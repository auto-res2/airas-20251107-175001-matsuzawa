import os
import sys
import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import hydra
from omegaconf import OmegaConf, DictConfig

import wandb
import optuna

from .preprocess import build_dataloaders
from .model import (
    build_model_with_adapters,
    compute_adapter_params,
)

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _apply_mode_overrides(cfg: DictConfig) -> None:
    """Apply trial/full specific overrides in-place."""
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.max_steps = 2
        cfg.training.epochs = 1
        cfg.evaluation.eval_steps = 1
        cfg.training.batch_size = min(int(cfg.training.batch_size), 2)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("cfg.mode must be 'trial' or 'full'")


def _ensure_results_dir(cfg: DictConfig) -> None:
    cfg.results_dir = os.path.abspath(cfg.results_dir)
    os.makedirs(cfg.results_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metric_name: str,
) -> Dict[str, float]:
    import evaluate as hf_evaluate

    metric = hf_evaluate.load(metric_name, cache_dir=CACHE_DIR)
    model.eval()

    id2label: Dict[int, str] = {
        int(k): v for k, v in getattr(model.config, "id2label", {}).items()
    }

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        preds = logits.argmax(-1).cpu()
        labels_cpu = labels.cpu()

        if metric_name == "seqeval":
            preds_list: List[List[str]] = []
            refs_list: List[List[str]] = []
            for pred_seq, label_seq in zip(preds, labels_cpu):
                p_sent: List[str] = []
                l_sent: List[str] = []
                for p_id, l_id in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l_id == -100:
                        continue
                    p_sent.append(id2label.get(p_id, str(p_id)))
                    l_sent.append(id2label.get(l_id, str(l_id)))
                preds_list.append(p_sent)
                refs_list.append(l_sent)
            metric.add_batch(predictions=preds_list, references=refs_list)
        else:
            metric.add_batch(predictions=preds, references=labels_cpu)

    result = metric.compute()
    if metric_name == "seqeval":
        return {
            "f1": result.get("overall_f1", 0.0),
            "accuracy": result.get("overall_accuracy", 0.0),
        }
    if isinstance(result, dict):
        return result
    return {metric_name: float(result)}


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def _grad_global_norm(model: torch.nn.Module) -> float:
    grads: List[torch.Tensor] = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()


def _training_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Returns (loss, grad_norm)."""
    model.train()
    labels = batch.pop("labels").to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    if scaler is not None:
        with torch.cuda.amp.autocast():
            loss = model(**batch, labels=labels).loss
        scaler.scale(loss).backward()
        grad_norm = _grad_global_norm(model)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = model(**batch, labels=labels).loss
        loss.backward()
        grad_norm = _grad_global_norm(model)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
    return loss.item(), grad_norm


# -----------------------------------------------------------------------------
# Optuna helpers
# -----------------------------------------------------------------------------

def _suggest_and_apply(trial: optuna.Trial, cfg: DictConfig):
    search_space: Dict[str, Any] = cfg.optuna.get("search_space", {})
    for path, space in search_space.items():
        if space["type"] == "loguniform":
            value = trial.suggest_float(path, space["low"], space["high"], log=True)
        elif space["type"] == "uniform":
            value = trial.suggest_float(path, space["low"], space["high"], log=False)
        elif space["type"] == "categorical":
            value = trial.suggest_categorical(path, space["choices"])
        elif space["type"] == "int":
            value = trial.suggest_int(path, space["low"], space["high"])
        else:
            raise ValueError(f"Unsupported Optuna space type {space['type']}")
        OmegaConf.update(cfg, path, value, merge=True)


def _objective(
    trial: optuna.Trial,
    cfg: DictConfig,
    model_init_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):
    cfg_tmp = copy.deepcopy(cfg)
    _suggest_and_apply(trial, cfg_tmp)

    model = model_init_fn(cfg_tmp)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg_tmp.training.learning_rate,
        weight_decay=cfg_tmp.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    max_steps = min(100, cfg_tmp.training.max_steps)
    eval_every = max(10, max_steps // 5)

    train_iter = iter(train_loader)
    best_val = -float("inf")
    step = 0
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss, _ = _training_step(model, batch, optimizer, None, device, scaler)
        step += 1
        if step % eval_every == 0 or step == max_steps:
            metrics = _evaluate(model, val_loader, device, cfg_tmp.evaluation.metric)
            key = "f1" if "f1" in metrics else list(metrics.keys())[0]
            val_score = metrics[key]
            trial.report(val_score, step)
            best_val = max(best_val, val_score)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best_val


# -----------------------------------------------------------------------------
# wandb helpers
# -----------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig):
    if cfg.wandb.mode == "disabled":
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=getattr(cfg, "run_id", cfg.run),
        resume="allow",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg.results_dir,
    )
    print(f"wandb URL: {run.url}")
    return run


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Basic sanity ----------------------------------------------------------------
    if cfg.run is None:
        raise ValueError("Parameter 'run' must be supplied: python -m src.train run=<run_id> ...")

    # Load run-specific configuration
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    run_config_path = Path(__file__).parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if run_config_path.exists():
        import yaml
        with open(run_config_path, 'r') as f:
            run_cfg = OmegaConf.create(yaml.safe_load(f))
        # Disable struct mode to allow merging new keys
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, run_cfg)
        OmegaConf.set_struct(cfg, True)

    _apply_mode_overrides(cfg)
    _ensure_results_dir(cfg)

    # Determinism -----------------------------------------------------------------
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data ------------------------------------------------------------------------
    train_loader, val_loader, label_list = build_dataloaders(cfg, CACHE_DIR)
    num_labels = len(label_list)

    def model_init(local_cfg: DictConfig):
        return build_model_with_adapters(local_cfg, num_labels, device)

    # Optuna hyper-parameter search ----------------------------------------------
    if cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(
            lambda trial: _objective(
                trial, cfg, model_init, train_loader, val_loader, device
            ),
            n_trials=int(cfg.optuna.n_trials),
        )
        for k, v in study.best_trial.params.items():
            OmegaConf.update(cfg, k, v, merge=True)
        print(
            f"Optuna best trial {study.best_trial.number}: value={study.best_value:.4f} params={study.best_trial.params}"
        )

    # Final training -------------------------------------------------------------
    model = model_init(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    lr_lambda = lambda cur: max(0.0, 1.0 - (cur / cfg.training.max_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    wb_run = _init_wandb(cfg)
    if wb_run is not None:
        wb_run.summary["adapter_param_MB"] = compute_adapter_params(model)

    global_step = 0
    pbar = tqdm(total=cfg.training.max_steps, desc="Training")
    while global_step < cfg.training.max_steps:
        for batch in train_loader:
            loss_val, grad_norm_val = _training_step(
                model, batch, optimizer, scheduler, device, scaler
            )
            global_step += 1
            pbar.update(1)

            # Logging -----------------------------------------------------------
            if wb_run is not None:
                metrics_to_log = {
                    "train_loss": loss_val,
                    "grad_norm": grad_norm_val,
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                }
                if torch.cuda.is_available():
                    metrics_to_log["gpu_mem_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
                wandb.log(metrics_to_log, step=global_step)

            # Evaluation --------------------------------------------------------
            if (
                global_step % cfg.evaluation.eval_steps == 0
                or global_step == cfg.training.max_steps
            ):
                eval_metrics = _evaluate(
                    model, val_loader, device, cfg.evaluation.metric
                )
                if wb_run is not None:
                    wandb.log(
                        {f"eval_{k}": v for k, v in eval_metrics.items()},
                        step=global_step,
                    )
            if global_step >= cfg.training.max_steps:
                break
    pbar.close()

    final_metrics = _evaluate(model, val_loader, device, cfg.evaluation.metric)
    print(f"Final metrics: {final_metrics}")

    # -------------------------------------------------------------------------
    # Save checkpoint and finish wandb
    # -------------------------------------------------------------------------
    if wb_run is not None:
        for k, v in final_metrics.items():
            wb_run.summary[f"final_{k}"] = v

        model_fp = Path(cfg.results_dir) / f"{cfg.run}_final_model.pth"
        torch.save(model.state_dict(), model_fp)
        artifact = wandb.Artifact(name=f"{cfg.run}_model", type="model")
        artifact.add_file(str(model_fp))
        wb_run.log_artifact(artifact)
        wb_run.finish()


if __name__ == "__main__":
    main()