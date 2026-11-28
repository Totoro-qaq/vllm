"""Generate PR figures from the two process medians in the local evidence file."""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parent / "figures"
results = json.loads(Path(sys.argv[1]).read_text())["trials"]
ink, muted, blue, gray, border = "#1c2731", "#4c5b69", "#276fbe", "#d9dedd", "#a8b2b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def label(ax, x, y, text, size=12, color=ink, weight="normal", **kw):
    ax.text(
        x, y, text, va="center", fontsize=size, color=color, fontweight=weight, **kw
    )


def rect(ax, x, y, w, h, fill):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="none"))


def figure(title):
    fig = plt.figure(figsize=(10, 5.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(0, 51))
    ax.axis("off")
    label(ax, 4, 47, title, 20, weight="bold")
    label(
        ax,
        4,
        43.7,
        "Inkling 2 Mini    ·    Desktop RTX 5090    ·    FP8 KV cache",
        10,
        muted,
    )
    return fig, ax


def latency(ax, arm):
    label(ax, 4, 18.5, "Time to first token", 13, weight="bold")
    rect(ax, 53, 17.9, 1.4, 1.2, gray)
    label(ax, 55, 18.5, "Stock", 10, muted)
    rect(ax, 65, 17.9, 1.4, 1.2, blue)
    label(ax, 67, 18.5, "With this change", 10, muted)
    for m, y in [(512, 14.8), (1024, 12.2), (2048, 9.6), (3325, 7.0), (4096, 4.4)]:

        def median(a, m=m):
            return statistics.mean(
                v[str(m)]["p50"] for k, v in results.items() if k.endswith("-" + a)
            )

        a, b = median("base"), median(arm)
        x = 21
        scale = 0.188
        label(ax, 4, y, "512 (control)" if m == 512 else f"{m:,} tokens", 10)
        rect(ax, x, y + 0.10, a * scale, 0.75, gray)
        rect(ax, x, y - 0.84, b * scale, 0.75, blue)
        label(ax, 63, y, f"{a:.2f} → {b:.2f} ms", 10)
        change = f"{100 * (b / a - 1):+.1f}%".replace("-", "−")
        label(ax, 87, y, change, 10, muted if m == 512 else blue, weight="bold")
    label(
        ax,
        4,
        1.5,
        "Warm streaming requests, one user. Each value averages two fresh-process medians.",
        9,
        muted,
    )


fig, ax = figure("Native MXFP8 in the last four expert layers")
label(ax, 4, 38.8, "Expert weights", 12, weight="bold")
for i in range(24):
    rect(ax, 22 + i * 3, 37.4, 2.83, 2.8, blue if i >= 20 else gray)
label(ax, 22, 34.8, "NVFP4  ·  layers 0–19", 10, muted)
label(ax, 94, 34.8, "MXFP8  ·  20–23", 10, blue, ha="right")
label(ax, 4, 29.6, "Stock", 11, muted)
label(ax, 22, 29.6, "Expand block scales   →   BF16 operands   →   matrix multiply", 11)
label(ax, 4, 25.4, "Long prefill", 11, blue, weight="bold")
label(
    ax, 22, 25.4, "E4M3 operands + E8M0 scales   →   native block-scaled MMA", 11, blue
)
ax.plot([4, 96], [21.8, 21.8], color=border, lw=0.75)
latency(ax, "native")
fig.savefig(root / "mxfp8-prefill.png", dpi=165)
plt.close(fig)

fig, ax = figure("A smaller intermediate for the loop controller")
label(ax, 4, 38.8, "Stock", 11, muted)
rect(ax, 22, 37.1, 25, 3.4, gray)
rect(ax, 47.15, 37.1, 25, 3.4, gray)
label(ax, 34.5, 38.8, "gain", 11, ha="center")
label(ax, 59.65, 38.8, "forget", 11, ha="center")
label(ax, 74, 38.8, "4,608 FP32 values", 10)
label(ax, 4, 32.8, "Fused", 11, blue, weight="bold")
rect(ax, 22, 31.1, 25, 3.4, blue)
label(ax, 34.5, 32.8, "blended", 11, "white", ha="center")
label(ax, 49, 32.8, "2,304 FP32 values", 10, blue)
label(
    ax,
    4,
    26.4,
    "Per token and stream, the up projection writes one blended row for RMSNorm.",
    10,
    muted,
)
ax.plot([4, 96], [21.8, 21.8], color=border, lw=0.75)
latency(ax, "controller")
fig.savefig(root / "fused-controller.png", dpi=165)
plt.close(fig)
