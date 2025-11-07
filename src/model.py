import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForTokenClassification
from omegaconf import DictConfig

__all__ = [
    "build_model_with_adapters",
    "compute_adapter_params",
]

# -----------------------------------------------------------------------------
# Baseline LoRA wrapper
# -----------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if r > 0:
            self.lora_a = nn.Linear(base.in_features, r, bias=False)
            self.lora_b = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
            self.scaling = alpha / r
        else:
            self.lora_a = self.lora_b = None
            self.scaling = 1.0
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return out


# -----------------------------------------------------------------------------
# Proposed UniHyperLoRA components (simplified implementation)
# -----------------------------------------------------------------------------

class DimAwareProjector(nn.Module):
    def __init__(self, d0: int = 256):
        super().__init__()
        self.d0 = d0
        self.mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, d0 * 2)
        )

    def forward(self, dim: int) -> torch.Tensor:
        inp = torch.log10(
            torch.tensor([[float(dim)]], dtype=torch.float32, device=self.mlp[0].weight.device)
        )
        vec = self.mlp(inp).view(2, self.d0)
        row, col = vec[0], vec[1]
        return torch.outer(torch.ones(dim, device=row.device), row) + torch.outer(
            torch.arange(1, dim + 1, dtype=row.dtype, device=row.device), col
        )


class SharedUniHyperLoRA(nn.Module):
    def __init__(self, M: int = 128, d0: int = 256, router_top_k: int = 2):
        super().__init__()
        self.M = M
        self.d0 = d0
        self.router_top_k = router_top_k
        self.U = nn.Parameter(torch.randn(M, d0) * 0.02)
        self.V = nn.Parameter(torch.randn(M, d0) * 0.02)
        self.dim_proj = DimAwareProjector(d0)
        self.mixer = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, M)
        )


class UniHyperLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, shared: SharedUniHyperLoRA, depth_norm: float, layer_type_id: int):
        super().__init__()
        self.base = base
        self.shared = shared
        for p in self.base.parameters():
            p.requires_grad = False
        self.register_buffer("depth_norm", torch.tensor([depth_norm], dtype=torch.float32))
        self.register_buffer("type_id", torch.tensor([layer_type_id / 4.0], dtype=torch.float32))

    def forward(self, x):
        out = self.base(x)
        d_out, d_in = self.base.out_features, self.base.in_features
        P_out = self.shared.dim_proj(d_out).to(x.device)
        P_in = self.shared.dim_proj(d_in).to(x.device).t()
        gates = self.shared.mixer(torch.cat([self.depth_norm, self.type_id]).unsqueeze(0))[0]
        vals, idx = torch.topk(gates, k=self.shared.router_top_k)
        delta_W = torch.zeros(d_out, d_in, device=x.device)
        for val, j in zip(vals, idx):
            delta_W += val * (
                P_out @ torch.outer(self.shared.U[j], self.shared.V[j]) @ P_in
            )
        return out + F.linear(x, delta_W, bias=None)


# -----------------------------------------------------------------------------
# Helper utils
# -----------------------------------------------------------------------------

def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    if child_name.isdigit():
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def compute_adapter_params(model: nn.Module) -> float:
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return bytes_total / (1024 ** 2)  # MB


# -----------------------------------------------------------------------------
# Build model with adapters
# -----------------------------------------------------------------------------

def build_model_with_adapters(cfg: DictConfig, num_labels: int, device):
    model_name = cfg.model.host_model_name

    base_cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=".cache/")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, config=base_cfg, cache_dir=".cache/"
    )

    if cfg.method.lower().startswith("proposed") or cfg.adapter.name.lower() == "unihyperlora":
        shared = SharedUniHyperLoRA(
            M=cfg.adapter.atoms,
            d0=cfg.adapter.anchor_dim,
            router_top_k=cfg.adapter.router_top_k,
        )
        linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
        total = len(linear_modules)
        idx = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                parent = _get_parent_module(model, name)
                child_name = name.split(".")[-1]
                depth = idx / max(1, total - 1)
                wrapped = UniHyperLoRALinear(module, shared, depth, layer_type_id=0)
                _set_child_module(parent, child_name, wrapped)
                idx += 1
    elif cfg.method.lower().startswith("comparative") or cfg.adapter.name.lower() == "sat-lora":
        targets: List[str] = cfg.adapter.target_modules
        for name, module in list(model.named_modules()):
            if any(t in name for t in targets) and isinstance(module, nn.Linear):
                parent = _get_parent_module(model, name)
                child_name = name.split(".")[-1]
                wrapped = LoRALinear(
                    module,
                    r=cfg.adapter.lora_rank,
                    alpha=cfg.adapter.lora_alpha,
                )
                _set_child_module(parent, child_name, wrapped)
    else:
        raise ValueError(f"Unknown adapter type: {cfg.adapter.name}")

    model.to(device)
    return model