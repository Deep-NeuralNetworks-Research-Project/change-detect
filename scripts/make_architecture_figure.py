#!/usr/bin/env python3
"""Publication figure: proposed model architecture.

Vector PDF (docs/figure_standards.md) plus PNG preview. Okabe–Ito palette;
no red–green. Geometry matches ProposedModel in src/cdlib/models/proposed.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# Okabe–Ito (CVD-safe). Fills are light tints of the same hues.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "magenta": "#CC79A7",
    "grey": "#5A5A5A",
    "ink": "#1A1A1A",
    "rule": "#9A9A9A",
}

FILL = {
    "input": "#F3F3F3",
    "encoder": "#D7EEF8",
    "align": "#FBE7B2",
    "fusion": "#CDEADF",
    "decoder": "#CDE4F3",
    "head": "#F3D6E8",
    "dir": "#F8DCC8",
    "panel": "#FBFBFB",
    "band": "#F3F8FB",
}

EDGE = {
    "input": "#6E6E6E",
    "encoder": PALETTE["blue"],
    "align": PALETTE["orange"],
    "fusion": PALETTE["green"],
    "decoder": PALETTE["blue"],
    "head": PALETTE["magenta"],
    "dir": PALETTE["vermillion"],
    "panel": "#B4B4B4",
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"


def _box(ax, x, y, w, h, text, *, fill, edge, fs=7.8, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15",
            linewidth=1.05,
            facecolor=fill,
            edgecolor=edge,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=PALETTE["ink"],
        fontweight=weight,
        linespacing=1.28,
        zorder=4,
    )
    return x, y, w, h


def _arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.05, rad=0.0, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": style,
            "color": color or PALETTE["ink"],
            "lw": lw,
            "mutation_scale": 9,
            "connectionstyle": f"arc3,rad={rad}",
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=2,
    )


def _poly(ax, pts, *, color=None, lw=1.05):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color or PALETTE["ink"], lw=lw, zorder=2, solid_capstyle="round")
    _arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw)


def _label(ax, x, y, text, *, fs=7.2, color=None, ha="center", va="center", weight="normal", italic=False):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color or PALETTE["grey"],
        fontweight=weight,
        fontstyle="italic" if italic else "normal",
        zorder=5,
    )


def _panel(ax, x, y, w, h, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2",
            linewidth=0.95,
            facecolor=FILL["panel"],
            edgecolor=EDGE["panel"],
            zorder=1,
        )
    )
    ax.text(
        x + 1.6,
        y + h - 2.6,
        title,
        ha="left",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=2,
    )


def _band(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15",
            linewidth=0.0,
            facecolor=FILL["band"],
            edgecolor="none",
            zorder=0,
        )
    )


def draw(ax) -> None:
    ax.set_xlim(0, 178)
    ax.set_ylim(0, 110)
    ax.axis("off")

    ax.text(
        89,
        107.2,
        "Proposed model  —  pair-order consistent, uncertainty-aware change detection",
        ha="center",
        va="center",
        fontsize=12.4,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        89,
        104.0,
        r"ProposedModel:  encoder.forward_pair $\rightarrow$ bounded alignment $\rightarrow$ signed fusion $\rightarrow$ U-Net decoder $\rightarrow$ heads",
        ha="center",
        va="center",
        fontsize=7.5,
        color=PALETTE["grey"],
    )

    # Column geometry
    enc_w, box_h = 18.0, 8.6
    x_i1, x_i2 = 8.0, 30.0
    x_al, w_al = 56.0, 20.0
    x_fu, w_fu = 84.0, 18.0
    x_de, w_de = 110.0, 18.0
    x_hd, w_hd = 140.0, 32.0

    # Fine → coarse
    scales = [
        {"name": "C2", "stride": 4, "ch": 64, "y": 74.6},
        {"name": "C3", "stride": 8, "ch": 128, "y": 63.0},
        {"name": "C4", "stride": 16, "ch": 256, "y": 51.4},
        {"name": "C5", "stride": 32, "ch": 512, "y": 39.8},
    ]

    _band(ax, 6.2, 38.0, 44.6, 63.6)
    _band(ax, 54.4, 38.0, 23.4, 54.4)
    _band(ax, 82.4, 38.0, 21.4, 54.4)
    _band(ax, 108.4, 38.0, 21.4, 54.4)

    _label(ax, 28.0, 101.6, "Inputs + shared Siamese encoder", fs=7.6, weight="bold")
    _label(
        ax,
        28.0,
        99.6,
        r"batch-concat $[I_1;I_2]$  ·  one BN pass  ·  split",
        fs=6.4,
        color=PALETTE["blue"],
    )
    _label(ax, x_al + w_al / 2, 93.0, "Bounded alignment", fs=7.6, weight="bold", color=PALETTE["orange"])
    _label(ax, x_fu + w_fu / 2, 93.0, "Signed fusion", fs=7.6, weight="bold", color=PALETTE["green"])
    _label(ax, x_de + w_de / 2, 93.0, "U-Net decoder", fs=7.6, weight="bold", color=PALETTE["blue"])
    _label(ax, x_hd + w_hd / 2, 93.0, "Heads (frozen contract)", fs=7.6, weight="bold", color=PALETTE["magenta"])

    # Inputs sit on top of each Siamese tower
    _box(
        ax,
        x_i1,
        92.6,
        enc_w,
        6.2,
        r"$I_1$  reference" + "\n$[B,3,H,W]$",
        fill=FILL["input"],
        edge=EDGE["input"],
        fs=7.6,
        weight="bold",
    )
    _box(
        ax,
        x_i2,
        92.6,
        enc_w,
        6.2,
        r"$I_2$  edited" + "\n$[B,3,H,W]$",
        fill=FILL["input"],
        edge=EDGE["input"],
        fs=7.6,
        weight="bold",
    )

    _box(ax, x_i1, 85.0, enc_w, 5.8, "stem  /4", fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.6)
    _box(ax, x_i2, 85.0, enc_w, 5.8, "stem  /4", fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.6)
    _arrow(ax, x_i1 + enc_w / 2, 92.6, x_i1 + enc_w / 2, 90.8)
    _arrow(ax, x_i2 + enc_w / 2, 92.6, x_i2 + enc_w / 2, 90.8)

    ax.annotate(
        "",
        xy=(x_i2, 87.9),
        xytext=(x_i1 + enc_w, 87.9),
        arrowprops={
            "arrowstyle": "<->",
            "color": PALETTE["blue"],
            "lw": 1.05,
            "mutation_scale": 9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=2,
    )
    _label(ax, 28.0, 89.4, "shared weights", fs=6.3, color=PALETTE["blue"])

    for i, s in enumerate(scales):
        y = s["y"]
        enc_txt = f"{s['name']}   /{s['stride']}\n{s['ch']} ch"
        _box(ax, x_i1, y, enc_w, box_h, enc_txt, fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.5)
        _box(ax, x_i2, y, enc_w, box_h, enc_txt, fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.5)
        _box(
            ax,
            x_al,
            y,
            w_al,
            box_h,
            "offset + gate\n" + r"warp $f_2$",
            fill=FILL["align"],
            edge=EDGE["align"],
            fs=7.3,
        )
        fuse_ch = 2 * s["ch"]
        _box(
            ax,
            x_fu,
            y,
            w_fu,
            box_h,
            r"$[a{-}b;\; a\odot b]$" + f"\n{fuse_ch} ch",
            fill=FILL["fusion"],
            edge=EDGE["fusion"],
            fs=7.2,
        )
        dec_ch = max(32, fuse_ch // 2)
        _box(
            ax,
            x_de,
            y,
            w_de,
            box_h,
            f"D{s['name'][1:]}\n{dec_ch} ch",
            fill=FILL["decoder"],
            edge=EDGE["decoder"],
            fs=7.5,
        )

        cy = y + box_h / 2.0
        _arrow(ax, x_i1 + enc_w, cy, x_i2, cy, color=PALETTE["rule"], lw=0.9, style="-")
        _arrow(ax, x_i2 + enc_w, cy, x_al, cy)
        _arrow(ax, x_al + w_al, cy, x_fu, cy)
        _arrow(ax, x_fu + w_fu, cy, x_de, cy)
        if i == 0:
            _label(ax, x_i2 + enc_w + 3.6, cy + 1.35, r"$f_1,\,f_2$", fs=6.2, color=PALETTE["blue"])

        if i == 0:
            _arrow(ax, x_i1 + enc_w / 2, 85.0, x_i1 + enc_w / 2, y + box_h)
            _arrow(ax, x_i2 + enc_w / 2, 85.0, x_i2 + enc_w / 2, y + box_h)
        else:
            y_prev = scales[i - 1]["y"]
            _arrow(ax, x_i1 + enc_w / 2, y_prev, x_i1 + enc_w / 2, y + box_h)
            _arrow(ax, x_i2 + enc_w / 2, y_prev, x_i2 + enc_w / 2, y + box_h)
            # decoder: coarsest → finest
            _arrow(
                ax,
                x_de + w_de / 2,
                y + box_h,
                x_de + w_de / 2,
                y_prev,
                color=PALETTE["blue"],
            )

    _label(
        ax,
        x_de + w_de / 2,
        85.0,
        "upsample + concat skip\nConv–BN–ReLU",
        fs=6.3,
        color=PALETTE["blue"],
    )
    _label(
        ax,
        28.0,
        37.0,
        "ResNet-18 taps shown (C2–C5). EfficientNet-B0 is a drop-in encoder.",
        fs=6.3,
    )

    # Heads
    y_c2 = scales[0]["y"]
    y_c3 = scales[1]["y"]
    y_c5 = scales[3]["y"]

    head_h = box_h + 2.2
    _box(
        ax,
        x_hd,
        y_c2 - 0.4,
        w_hd,
        head_h,
        "logits   $[B,1,H,W]$\npre-sigmoid change mask",
        fill=FILL["head"],
        edge=EDGE["head"],
        fs=7.6,
        weight="bold",
    )
    _arrow(ax, x_de + w_de, y_c2 + box_h / 2, x_hd, y_c2 + head_h / 2 - 0.4)

    _box(
        ax,
        x_hd,
        y_c3 - 1.2,
        w_hd,
        head_h,
        "confidence   $[B,1,H,W]$\n1×1 conv on logits",
        fill=FILL["head"],
        edge=EDGE["head"],
        fs=7.4,
    )
    _arrow(ax, x_hd + w_hd / 2, y_c2 - 0.4, x_hd + w_hd / 2, y_c3 - 1.2 + head_h)

    # Aux outputs hang off the modules that emit them (not the decoder).
    aux_y, aux_h = 33.0, 5.6
    _box(
        ax,
        x_al,
        aux_y,
        w_al,
        aux_h,
        r"aux.alignment_offset  $[B,2,H,W]$",
        fill=FILL["align"],
        edge=EDGE["align"],
        fs=7.0,
    )
    _arrow(ax, x_al + w_al / 2, y_c5, x_al + w_al / 2, aux_y + aux_h, color=PALETTE["orange"])

    _box(
        ax,
        x_fu,
        aux_y,
        w_fu + (x_de - x_fu - w_fu) + w_de,
        aux_h,
        "directional logits  $[B,2,H,W]$   ·   from fused C5",
        fill=FILL["dir"],
        edge=EDGE["dir"],
        fs=7.0,
    )
    _arrow(ax, x_fu + w_fu / 2, y_c5, x_fu + w_fu / 2, aux_y + aux_h, color=PALETTE["vermillion"])

    # ---- insets ----
    _panel(ax, 4.0, 1.6, 54.0, 30.4, "(a)  Bounded, change-gated alignment")
    _panel(ax, 60.5, 1.6, 54.0, 30.4, "(b)  Why signed fusion needs a swap loss")
    _panel(ax, 117.0, 1.6, 57.0, 30.4, "(c)  Pair-order pass at train time")

    # (a)
    _box(ax, 7.2, 21.4, 22.0, 5.8, r"cat$(f_1,\,f_2)$", fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.3)
    _box(ax, 32.6, 21.4, 22.0, 5.8, "3×3 → ReLU → 3×3\nlast conv zero-init", fill=FILL["align"], edge=EDGE["align"], fs=6.5)
    _arrow(ax, 29.2, 24.3, 32.6, 24.3)

    _box(ax, 7.2, 13.6, 22.0, 5.8, r"offset $=\tanh(\mathrm{raw})\times 4$", fill=FILL["align"], edge=EDGE["align"], fs=6.8)
    _arrow(ax, 18.2, 21.4, 18.2, 19.4)
    _box(ax, 32.6, 13.6, 22.0, 5.8, r"grid_sample warp $f_2$", fill=FILL["align"], edge=EDGE["align"], fs=7.1)
    _arrow(ax, 29.2, 16.5, 32.6, 16.5)

    _box(ax, 7.2, 5.8, 22.0, 5.8, r"gate $\sigma(1{\times}1(|f_1-f_2|))$", fill=FILL["fusion"], edge=EDGE["fusion"], fs=6.6)
    _box(ax, 32.6, 5.8, 22.0, 5.8, r"$f_2'=g\cdot\mathrm{warp}+(1-g)f_2$", fill=FILL["align"], edge=EDGE["align"], fs=6.5)
    _arrow(ax, 29.2, 8.7, 32.6, 8.7)
    _arrow(ax, 43.6, 13.6, 43.6, 11.6)
    _label(
        ax,
        31.0,
        3.6,
        "Gate from abs-diff magnitude, not from the change head.",
        fs=6.2,
    )

    # (b)
    ax.text(
        87.5,
        24.8,
        r"Default mode  $\varphi(a,b)=\mathrm{concat}(a-b,\; a\odot b)$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=PALETTE["ink"],
        zorder=5,
    )
    ax.text(
        87.5,
        20.4,
        r"$|a-b|$ is swap-symmetric for any head $h$; direction is discarded.",
        ha="center",
        va="center",
        fontsize=7.1,
        color=PALETTE["grey"],
        zorder=5,
    )
    ax.text(
        87.5,
        16.2,
        r"Signed $a-b$ is antisymmetric:  $\varphi(b,a)=-\varphi(a,b)$.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=PALETTE["ink"],
        zorder=5,
    )
    ax.text(
        87.5,
        12.2,
        r"$h(b-a)=h(-(a-b))$ equals $h(a-b)$  iff  $h$ is even.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=PALETTE["ink"],
        zorder=5,
    )
    ax.text(
        87.5,
        6.8,
        "A generic conv head is not even. Pair-order consistency\n"
        "is a learned constraint, not a freebie of weight sharing.",
        ha="center",
        va="center",
        fontsize=7.1,
        color=PALETTE["vermillion"],
        zorder=5,
    )

    # (c)
    _box(ax, 121.5, 19.6, 21.5, 6.4, r"forward$(I_1,I_2)$", fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.3)
    _box(ax, 148.0, 19.6, 21.5, 6.4, r"forward$(I_2,I_1)$", fill=FILL["encoder"], edge=EDGE["encoder"], fs=7.3)
    _arrow(ax, 143.0, 22.8, 148.0, 22.8)
    ax.annotate(
        "",
        xy=(132.2, 19.6),
        xytext=(158.8, 19.6),
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["magenta"],
            "lw": 1.05,
            "connectionstyle": "arc3,rad=0.55",
        },
        zorder=2,
    )
    _label(ax, 145.5, 16.0, "swap", fs=6.6, color=PALETTE["magenta"])
    ax.text(
        145.5,
        12.0,
        r"binary:  $\sigma(\mathrm{logits})\approx\sigma(\mathrm{logits\_swapped})$",
        ha="center",
        va="center",
        fontsize=7.3,
        color=PALETTE["ink"],
        zorder=5,
    )
    ax.text(
        145.5,
        8.0,
        "direction: appeared ↔ disappeared channels permute",
        ha="center",
        va="center",
        fontsize=7.1,
        color=PALETTE["ink"],
        zorder=5,
    )
    ax.text(
        145.5,
        4.2,
        "Emitted only when training  ·  compute_swap=True",
        ha="center",
        va="center",
        fontsize=6.2,
        color=PALETTE["grey"],
        zorder=5,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(17.0, 10.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    pdf = OUT_DIR / "proposed_architecture.pdf"
    png = OUT_DIR / "proposed_architecture.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png, format="png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
