"""Plots of one evaluated run, and of several runs side by side.

Read from `evaluation_<split>.json`, which `legal-risk-evaluate` writes, so any
evaluated run can be plotted without re-running a model. Needs matplotlib.

Accuracy is drawn because it is asked for, and labelled for what it is: most
windows and many contracts carry no clause, so a model that flags nothing still
scores high on it. Precision, recall and F1 are the numbers to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .labels import LABEL_NAMES

LEVELS = (("chunk_level", "Window level"), ("document_level", "Contract level"))
METRICS = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy"))
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#9aa1ab")
SHORT = {
    "Cap on Liability": "Cap on\nLiability",
    "Non-Compete": "Non-\nCompete",
    "License Grant": "License\nGrant",
    "Audit Rights": "Audit\nRights",
    "Termination for Convenience": "Termination\nfor Conv.",
    "Insurance": "Insurance",
}


def load_evaluation(run_dir: str | Path, split: str = "test") -> dict:
    path = Path(run_dir) / f"evaluation_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run legal-risk-evaluate --run_dir {run_dir} first")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if "tp" not in evaluation["chunk_level"]["per_class"][0]:
        raise ValueError(f"{path} predates confusion-matrix counts: re-run legal-risk-evaluate on it")
    return evaluation


def plot_run(run_dir: str | Path, split: str = "test", show: bool = True) -> list[Path]:
    """Confusion matrices and per-label metrics for one run. Returns the saved PNGs."""
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    evaluation = load_evaluation(run_dir, split)
    title = f"{run_dir.name} · {split} split"
    saved = []

    # Confusion matrices: one 2x2 per label, window level above, contract level below.
    fig, axes = plt.subplots(2, len(LABEL_NAMES), figsize=(3.1 * len(LABEL_NAMES), 6.6))
    for row, (level, level_name) in enumerate(LEVELS):
        for col, entry in enumerate(evaluation[level]["per_class"]):
            ax = axes[row, col]
            counts = np.array([[entry["tn"], entry["fp"]], [entry["fn"], entry["tp"]]])
            # Colour by share of the true row, so the rare positive row is as
            # readable as the huge negative one.
            shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
            ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
            for (i, j), n in np.ndenumerate(counts):
                ax.text(j, i, f"{n:,}\n{shares[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if shares[i, j] > 0.6 else "#14181f")
            ax.set_xticks([0, 1], ["no", "yes"])
            ax.set_yticks([0, 1], ["no", "yes"])
            ax.set_xlabel("predicted", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{level_name}\nactual", fontsize=9)
            if row == 0:
                ax.set_title(entry["label"], fontsize=10)
    fig.suptitle(f"Confusion matrices · {title}", fontsize=12)
    fig.tight_layout()
    path = run_dir / f"confusion_matrices_{split}.png"
    fig.savefig(path, dpi=130)
    saved.append(path)

    # Per-label precision, recall, F1 and accuracy, plus the overall scores.
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    x = np.arange(len(LABEL_NAMES))
    width = 0.2
    for ax, (level, level_name) in zip(axes, LEVELS):
        block = evaluation[level]
        for k, ((key, label), color) in enumerate(zip(METRICS, COLORS)):
            values = [entry[key] for entry in block["per_class"]]
            bars = ax.bar(x + (k - 1.5) * width, values, width, label=label, color=color)
            if key == "f1":
                ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(x, [SHORT[name] for name in LABEL_NAMES], fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        summary = (f"macro P {block['macro_precision']:.3f}   R {block['macro_recall']:.3f}   "
                   f"F1 {block['macro_f1']:.3f}   AP {block['macro_average_precision']:.3f}\n"
                   f"micro F1 {block['micro_f1']:.3f}   label accuracy {block['label_accuracy']:.3f}   "
                   f"exact match {block['exact_match_accuracy']:.3f}")
        ax.set_title(f"{level_name}\n{summary}", fontsize=9)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Metrics by label · {title}\n"
                 "Accuracy is high because most windows and contracts have no clause; read F1.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = run_dir / f"metrics_{split}.png"
    fig.savefig(path, dpi=130)
    saved.append(path)

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


def plot_comparison(run_dirs, split: str = "test", out: str | Path = "outputs/comparison.png",
                    show: bool = True) -> Path:
    """Contract-level scores of several runs side by side. Missing runs are skipped."""
    import matplotlib.pyplot as plt

    runs = []
    for run_dir in map(Path, run_dirs):
        try:
            evaluation = load_evaluation(run_dir, split)
        except (FileNotFoundError, ValueError) as error:
            print(f"skipping {run_dir}: {error}")
            continue
        runs.append((run_dir.name, evaluation["document_level"]))
    if not runs:
        raise ValueError("no evaluated runs to compare")

    keys = (("macro_precision", "Precision"), ("macro_recall", "Recall"), ("macro_f1", "F1"),
            ("macro_average_precision", "AP"), ("exact_match_accuracy", "Exact match"))
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(runs)), 5))
    x = np.arange(len(runs))
    width = 0.8 / len(keys)
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9aa1ab")
    for k, ((key, label), color) in enumerate(zip(keys, palette)):
        bars = ax.bar(x + (k - (len(keys) - 1) / 2) * width, [block[key] for _, block in runs],
                      width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, [name for name, _ in runs])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("macro score")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(keys), fontsize=9, loc="lower center", frameon=False)
    ax.set_title(f"Contract-level scores by model · {split} split")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out
