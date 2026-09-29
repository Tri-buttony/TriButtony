"""
Графики обучения ultralytics по results.csv одного или нескольких запусков.

    python -m services.plot_training runs/label_detector/train runs/label_detector/n30 \
        --labels "yolo26s b8" "yolo26n b8" --out runs/label_detector/curves.png

Можно запускать во время обучения — берутся уже завершённые эпохи.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # категориальные слоты 1–4
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

PANELS = [
    ("metrics/mAP50(B)", "mAP50 (val)"),
    ("metrics/mAP50-95(B)", "mAP50-95 (val)"),
    ("metrics/precision(B)", "Precision (val)"),
    ("metrics/recall(B)", "Recall (val)"),
    ("train/box_loss", "box loss (train)"),
    ("val/box_loss", "box loss (val)"),
    ("train/cls_loss", "cls loss (train)"),
    ("val/cls_loss", "cls loss (val)"),
    ("lr/pg0", "learning rate"),
]


def load(run: Path) -> dict:
    with open(run / "results.csv", newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", type=Path, default=Path("runs/label_detector/curves.png"))
    args = ap.parse_args()
    labels = args.labels or [r.name for r in args.runs]
    data = [load(r) for r in args.runs]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for d, lab, color in zip(data, labels, SERIES):
            if key not in d:
                continue
            ep, ys = d["epoch"], d[key]
            ax.plot(ep, ys, color=color, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=lab)
            if key.startswith("metrics/mAP"):  # подпись лучшей эпохи
                i = max(range(len(ys)), key=ys.__getitem__)
                ax.annotate(f"{ys[i]:.3f}", (ep[i], ys[i]), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8, color=INK2)
        ax.set_xlabel("эпоха")
        if key == "lr/pg0":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", ncol=len(labs), frameon=False, fontsize=10)
    fig.suptitle("Детектор этикеток — обучение", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
