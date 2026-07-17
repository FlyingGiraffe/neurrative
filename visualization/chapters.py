"""
Chapter curve visualization: smooth each chapter's paragraph-to-paragraph
path into a curve using mean curvature flow (curve shortening flow).

Connecting consecutive paragraphs with straight segments produces a jagged
line -- every small semantic wobble between neighbouring paragraphs shows up
as a kink. Flowing the paragraph points directly isn't enough to fix this:
with only as many vertices as paragraphs and both endpoints pinned, a long
paragraph-to-paragraph jump has nowhere to bend and stays almost straight.
So before flowing, the polyline is subsampled uniformly in arc length --
long edges get proportionally more control points than short ones -- and
mean curvature flow is run on that dense polyline instead, revealing the
chapter's actual curved shape in the story manifold.

Example:
    python visualization/chapters.py --book alice_wonderland --model bge-m3
    python visualization/chapters.py --book alice_wonderland --model bge-m3 --chapter 5
    python visualization/chapters.py --book alice_wonderland --model bge-m3 --coords pca
"""

import os
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.curve_flow import uniform_arclength_resample, mean_curvature_flow, total_curvature


def smooth_chapter(coords, chapter, iterations, step_size, samples_per_edge=10):
    start, end = chapter["paragraph_range"]
    idx = np.arange(start, end + 1)
    raw = coords[idx]
    dense, orig_idx = uniform_arclength_resample(raw, samples_per_edge=samples_per_edge)
    smoothed = mean_curvature_flow(dense, iterations=iterations, step_size=step_size)
    return idx, raw, smoothed, orig_idx


def plot_single_chapter(raw, smoothed, orig_idx, chapter, book_title, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(raw[:, 0], raw[:, 1], color="lightgray", lw=1.0, alpha=0.7,
            zorder=1, label="raw (paragraph-to-paragraph)")
    ax.scatter(raw[:, 0], raw[:, 1], color="lightgray", s=14, alpha=0.8, zorder=2)

    ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1.5, alpha=0.6, zorder=3,
            label="smoothed (mean curvature flow)")

    at_paragraphs = smoothed[orig_idx]
    t = np.linspace(0, 1, len(at_paragraphs))
    ax.scatter(at_paragraphs[:, 0], at_paragraphs[:, 1], c=t, cmap="viridis", s=26,
               alpha=0.95, linewidths=0, zorder=4)

    ax.scatter(*raw[0],  color="lime", s=140, marker="*", zorder=5)
    ax.scatter(*raw[-1], color="red",  s=140, marker="X", zorder=5)

    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"Ch {chapter['chapter_number']}: {chapter['title']}\n{book_title}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def colored_curve_collection(points, cmap="viridis", lw=1.6, alpha=0.85, zorder=2):
    """LineCollection coloring a polyline by position along its length (0 -> 1)."""
    segments = np.stack([points[:-1], points[1:]], axis=1)
    t = np.linspace(0, 1, len(segments))
    lc = LineCollection(segments, cmap=cmap, array=t, linewidths=lw, alpha=alpha, zorder=zorder)
    return lc


def plot_all_chapters_grid(all_raw, all_smoothed, all_orig_idx, chapters, book_title, output_path):
    n = len(chapters)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    for ax, raw, smoothed, orig_idx, ch in zip(axes.flat, all_raw, all_smoothed, all_orig_idx, chapters):
        ax.plot(raw[:, 0], raw[:, 1], color="lightgray", lw=0.8, alpha=0.7, zorder=1)
        ax.add_collection(colored_curve_collection(smoothed))

        ax.scatter(*smoothed[0],  marker="o", s=90, facecolors="none",
                   edgecolors="black", linewidths=1.6, zorder=4)
        ax.scatter(*smoothed[-1], marker="x", s=90, color="black", linewidths=1.6, zorder=4)

        ax.set_title(f"Ch {ch['chapter_number']}: {ch['title'][:22]}", fontsize=9)
        ax.autoscale_view()
        ax.axis("off")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Chapter Curves (mean curvature flow) — {book_title}", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_overview(coords, all_smoothed, chapters, book_title, output_path):
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(coords[:, 0], coords[:, 1], color="lightgray", s=6, alpha=0.35,
               linewidths=0, zorder=1)

    colors = cm.tab20(np.linspace(0, 1, len(chapters)))
    for smoothed, ch, color in zip(all_smoothed, chapters, colors):
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, lw=2.2, alpha=0.9,
                zorder=2, label=f"Ch {ch['chapter_number']}")
        ax.scatter(*smoothed[0], color=color, s=50, marker="o", zorder=3,
                   edgecolors="white", linewidths=0.6)

    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(f"All Chapter Curves on the Story Manifold — {book_title}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book",       required=True, type=str)
    parser.add_argument("--model",      required=True, type=str)
    parser.add_argument("--coords",     default="umap", choices=["umap", "pca"],
                        help="Which 2D projection to smooth/draw the curves in")
    parser.add_argument("--chapter",    default=None, type=int,
                        help="Chapter number (1-indexed) to plot alone; default plots all chapters")
    parser.add_argument("--iterations", default=5, type=int,
                        help="Number of implicit mean-curvature-flow steps")
    parser.add_argument("--step-size",  default=0.15, type=float,
                        help="Dimensionless flow amount per step, auto-scaled to each "
                             "chapter's own length scale (larger = more smoothing)")
    parser.add_argument("--samples-per-edge", default=10, type=int,
                        help="Control points to subsample per paragraph-to-paragraph edge "
                             "before flowing (uniform in arc length, so longer edges get "
                             "proportionally more -- needed for the flow to actually curve)")
    args = parser.parse_args()

    book_dir = os.path.join("books", args.book)
    output_dir = os.path.join("output", args.book, args.model)
    os.makedirs(output_dir, exist_ok=True)
    book_title = args.book.replace("_", " ").title()

    with open(os.path.join(book_dir, "processed.json"), encoding="utf-8") as f:
        processed = json.load(f)
    chapters = processed["chapters"]

    coords = np.load(os.path.join(output_dir, f"{args.coords}.npy"))

    if args.chapter is not None:
        chapters = [c for c in chapters if c["chapter_number"] == _to_roman_or_str(args.chapter, chapters)]
        if not chapters:
            print(f"Chapter {args.chapter} not found.")
            return

    all_raw, all_smoothed, all_orig_idx = [], [], []
    for ch in chapters:
        _, raw, smoothed, orig_idx = smooth_chapter(
            coords, ch, args.iterations, args.step_size, args.samples_per_edge
        )
        all_raw.append(raw)
        all_smoothed.append(smoothed)
        all_orig_idx.append(orig_idx)

        raw_curv = total_curvature(raw)
        smooth_curv = total_curvature(smoothed)
        reduction = 100 * (1 - smooth_curv / raw_curv) if raw_curv > 1e-10 else 0.0
        print(f"Ch {ch['chapter_number']:>4} {ch['title'][:30]:<30}  "
              f"total curvature {raw_curv:6.2f} -> {smooth_curv:6.2f}  "
              f"({reduction:5.1f}% smoother)")

    if args.chapter is not None:
        ch = chapters[0]
        key = ch["chapter_number"]
        out_path = os.path.join(output_dir, f"chapter_curve_{key}_{args.coords}.png")
        plot_single_chapter(all_raw[0], all_smoothed[0], all_orig_idx[0], ch, book_title, out_path)
        return

    plot_all_chapters_grid(
        all_raw, all_smoothed, all_orig_idx, chapters, book_title,
        os.path.join(output_dir, f"chapter_curves_grid_{args.coords}.png"),
    )
    plot_overview(
        coords, all_smoothed, chapters, book_title,
        os.path.join(output_dir, f"chapter_curves_overview_{args.coords}.png"),
    )


def _to_roman_or_str(chapter_number, chapters):
    """Map a 1-indexed integer to whatever chapter_number label is stored (roman numeral or int)."""
    for i, ch in enumerate(chapters, start=1):
        if i == chapter_number:
            return ch["chapter_number"]
    return None


if __name__ == "__main__":
    main()
