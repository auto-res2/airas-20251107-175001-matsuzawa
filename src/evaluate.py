import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch  # Must be imported before use throughout file
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import (
    confusion_matrix,
)

from .preprocess import build_dataloaders
from .model import build_model_with_adapters

PRIMARY_METRIC_STRING = (
    "Target-model dev score under 1 MB / 10 %-FLOP budget (report zero-shot and post-tune)."
)
CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_wandb_credentials():
    cfg_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_file)
    return cfg.wandb.entity, cfg.wandb.project


def _sanitize(s: str) -> str:
    """Return a safe filename component (no slashes, spaces, etc.)."""
    return re.sub(r"[^A-Za-z0-9\-_]", "_", s)


def _fetch_run(rid: str, entity: str, project: str):
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{rid}")


# -----------------------------------------------------------------------------
# In-depth run processing helpers
# -----------------------------------------------------------------------------

def _reconstruct_cfg(wandb_cfg: Dict[str, Any]) -> OmegaConf:
    """Convert the flattened wandb config dict back into OmegaConf style."""
    cfg_nested: Dict[str, Any] = {}
    for k, v in wandb_cfg.items():
        cur = cfg_nested
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return OmegaConf.create(cfg_nested)


def _generate_predictions(
    cfg: OmegaConf, model_state_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load model weights and dataset, then produce predictions (validation split)."""
    device = "cpu"

    # Build loaders & model ----------------------------------------------------
    loaders = build_dataloaders(cfg, CACHE_DIR)
    _, val_loader, label_list = loaders
    num_labels = len(label_list)

    model = build_model_with_adapters(cfg, num_labels=num_labels, device=device)
    state_dict = torch.load(model_state_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    preds_all: List[int] = []
    labels_all: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels")
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = logits.argmax(-1)
            if cfg.evaluation.metric == "seqeval":
                for p_seq, l_seq in zip(preds, labels):
                    for p_id, l_id in zip(p_seq.tolist(), l_seq.tolist()):
                        if l_id == -100:
                            continue
                        preds_all.append(p_id)
                        labels_all.append(l_id)
            else:
                preds_all.extend(preds.tolist())
                labels_all.extend(labels.tolist())
    return np.array(preds_all), np.array(labels_all), label_list


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Path,
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# STEP-1: Per-run processing
# -----------------------------------------------------------------------------

def _export_metrics_and_figures(run, save_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Export history & generate per-run figures. Returns (preds, labels) if computed."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save history & summary JSON ------------------------------------------
    history_df = run.history()
    summary_dict = dict(run.summary._json_dict)
    cfg_dict = dict(run.config)

    with open(save_dir / "metrics.json", "w") as fp:
        json.dump({"summary": summary_dict, "config": cfg_dict}, fp, indent=2)
    history_df.to_parquet(save_dir / "history.parquet")

    # 2. Learning-curve figure -------------------------------------------------
    fig_fp = save_dir / f"{_sanitize(run.id)}_learning_curve.pdf"
    plt.figure()
    metric_cols = [c for c in history_df.columns if re.match(r"(train|eval)_.*", str(c))]
    for col in metric_cols:
        sns.lineplot(x=history_df.index, y=history_df[col], label=col)
    plt.xlabel("Step")
    plt.ylabel("Metric value")
    plt.title(run.id)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_fp)
    plt.close()
    print(f"Saved figure: {fig_fp}")

    # 3. Confusion matrix ------------------------------------------------------
    preds_np = labels_np = None
    try:
        art = next(a for a in run.logged_artifacts() if a.type == "model")
        model_dir = art.download()
        state_path = next(Path(model_dir).glob("*.pth"))
        cfg = _reconstruct_cfg(cfg_dict)
        preds_np, labels_np, label_list = _generate_predictions(cfg, str(state_path))

        cm_path = save_dir / f"{_sanitize(run.id)}_confusion_matrix.pdf"
        _plot_confusion_matrix(labels_np, preds_np, label_list, cm_path)
        print(f"Saved figure: {cm_path}")

        # Save raw preds / labels
        np.save(save_dir / "preds.npy", preds_np)
        np.save(save_dir / "labels.npy", labels_np)
    except StopIteration:
        print(f"[WARN] No model artifact found for run {run.id}; skipping confusion matrix.")
    except Exception as e:
        print(f"[WARN] Failed to generate confusion matrix for {run.id}: {e}")

    return preds_np, labels_np


# -----------------------------------------------------------------------------
# STEP-2: Aggregated analysis & comparison
# -----------------------------------------------------------------------------

def _mcnemar(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """McNemar's exact binomial test between two classifiers."""
    assert y1.shape == y2.shape == y_true.shape
    n01 = ((y1 == y_true) & (y2 != y_true)).sum()
    n10 = ((y1 != y_true) & (y2 == y_true)).sum()
    from math import comb

    n = n01 + n10
    if n == 0:
        p_val = 1.0
    else:
        k = min(n01, n10)
        p_val = 2 * sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
        p_val = min(1.0, p_val)
    return {"n01": int(n01), "n10": int(n10), "p_value": float(p_val)}


def _aggregate(runs: Dict[str, wandb.apis.public.Run], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_name: Dict[str, Dict[str, float]] = {}
    preds_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Collect numerical summary metrics --------------------------------------
    for rid, run in runs.items():
        summary = run.summary._json_dict
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                metrics_by_name.setdefault(k, {})[rid] = v
        run_dir = out_dir.parent / _sanitize(rid)
        preds_fp = run_dir / "preds.npy"
        labels_fp = run_dir / "labels.npy"
        if preds_fp.exists() and labels_fp.exists():
            preds_map[rid] = (np.load(preds_fp), np.load(labels_fp))

    # Primary metric selection ------------------------------------------------
    pref_keys = [
        "final_f1",
        "final_accuracy",
        "f1",
        "accuracy",
        "eval_f1",
    ]
    selected_key = next((k for k in pref_keys if k in metrics_by_name), None)
    if selected_key is None:
        raise RuntimeError("Cannot find a common metric across runs.")

    proposed_vals = {
        rid: v for rid, v in metrics_by_name[selected_key].items() if "proposed" in rid
    }
    baseline_vals = {
        rid: v
        for rid, v in metrics_by_name[selected_key].items()
        if any(tok in rid for tok in ["baseline", "comparative"])
    }

    best_proposed_id = max(proposed_vals, key=proposed_vals.get)
    best_baseline_id = max(baseline_vals, key=baseline_vals.get)
    best_proposed_val = proposed_vals[best_proposed_id]
    best_baseline_val = baseline_vals[best_baseline_id]
    gap_pct = (best_proposed_val - best_baseline_val) / best_baseline_val * 100.0

    aggregate = {
        "primary_metric": PRIMARY_METRIC_STRING,
        "metrics": metrics_by_name,
        "best_proposed": {"run_id": best_proposed_id, "value": best_proposed_val},
        "best_baseline": {"run_id": best_baseline_id, "value": best_baseline_val},
        "gap": gap_pct,
    }

    # Statistical significance -----------------------------------------------
    if best_proposed_id in preds_map and best_baseline_id in preds_map:
        yp, yt = preds_map[best_proposed_id]
        yb, _ = preds_map[best_baseline_id]
        sig = _mcnemar(yp, yb, yt)
        aggregate["stat_tests"] = {"mcnemar": sig}
    else:
        aggregate["stat_tests"] = {"mcnemar": "NA"}

    out_fp = out_dir / "aggregated_metrics.json"
    with open(out_fp, "w") as fp:
        json.dump(aggregate, fp, indent=2)
    print(f"Aggregated metrics saved → {out_fp}")

    # Comparison figures ------------------------------------------------------
    bar_fig = out_dir / f"comparison_{_sanitize(selected_key)}_bar_chart.pdf"
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(metrics_by_name[selected_key].keys()),
        y=list(metrics_by_name[selected_key].values()),
        palette="viridis",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(selected_key)
    plt.title("Comparison on " + selected_key)
    for i, v in enumerate(metrics_by_name[selected_key].values()):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(bar_fig)
    plt.close()
    print(f"Saved figure: {bar_fig}")

    metrics_df = pd.DataFrame(metrics_by_name)
    melted = metrics_df.melt(var_name="metric", value_name="value", ignore_index=False)
    melted = melted.reset_index().rename(columns={"index": "run_id"})
    box_fig = out_dir / "comparison_all_metrics_boxplot.pdf"
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="metric", y="value")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Metric distribution across runs")
    plt.tight_layout()
    plt.savefig(box_fig)
    plt.close()
    print(f"Saved figure: {box_fig}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate multiple wandb runs and generate comparison reports."
    )
    p.add_argument("results_dir", type=str)
    p.add_argument(
        "run_ids",
        type=str,
        help="JSON string list of run IDs, e.g. '[\"run1\", \"run2\"]'",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results_root = Path(args.results_dir)
    run_id_list: List[str] = json.loads(args.run_ids)

    entity, project = _load_wandb_credentials()

    runs: Dict[str, wandb.apis.public.Run] = {}
    for rid in run_id_list:
        run = _fetch_run(rid, entity, project)
        runs[rid] = run
        _export_metrics_and_figures(run, results_root / _sanitize(rid))

    _aggregate(runs, results_root / "comparison")