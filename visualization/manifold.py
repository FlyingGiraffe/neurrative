"""
Manifold visualization: draw the KNN graph on a 2D layout, colour nodes by
narrative scalar fields (wonder, danger, etc.), and show gradient vector fields.

Example:
    python visualization/manifold.py --book alice_wonderland --model bge-m3
    python visualization/manifold.py --book alice_wonderland --model bge-m3 --score danger
"""

import os
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.knn import build_knn_graph


SCORE_FIELDS = ["wonder", "danger", "sadness", "humor", "confusion", "curiosity"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(book_dir, model):
    with open(os.path.join(book_dir, "processed.json"), encoding="utf-8") as f:
        processed = json.load(f)
    with open(os.path.join(book_dir, "paragraph_scores.json"), encoding="utf-8") as f:
        scores_raw = json.load(f)

    embeddings = np.load(os.path.join(book_dir, "embeddings", model, "embeddings.npy"))
    umap_coords = np.load(os.path.join("output", os.path.basename(book_dir), model, "umap.npy"))

    scores_by_id = {s["paragraph_id"]: s.get("scores", {}) for s in scores_raw}
    return processed, scores_by_id, embeddings, umap_coords


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_knn_edges(ax, coords, graph, max_edges=3000, alpha=0.15):
    """Draw KNN graph edges as thin grey lines (sample to avoid clutter)."""
    cx = graph.tocoo()
    mask = cx.row < cx.col  # avoid drawing each edge twice
    rows, cols = cx.row[mask], cx.col[mask]

    if len(rows) > max_edges:
        idx = np.random.choice(len(rows), max_edges, replace=False)
        rows, cols = rows[idx], cols[idx]

    segments_x = np.stack([coords[rows, 0], coords[cols, 0], np.full(len(rows), np.nan)], axis=1).ravel()
    segments_y = np.stack([coords[rows, 1], coords[cols, 1], np.full(len(rows), np.nan)], axis=1).ravel()
    ax.plot(segments_x, segments_y, color="gray", lw=0.3, alpha=alpha, zorder=1)


def plot_score_on_manifold(coords, graph, score_values, title, output_path,
                           cmap="YlOrRd", draw_edges=True):
    fig, ax = plt.subplots(figsize=(9, 9))

    if draw_edges:
        _draw_knn_edges(ax, coords, graph)

    norm = mcolors.Normalize(vmin=score_values.min(), vmax=score_values.max())
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=score_values, cmap=cmap, norm=norm,
                    s=22, alpha=0.95, linewidths=0, zorder=2)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title, fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gradient_field(coords, score_values, knn_indices, title, output_path):
    """
    Visualise the gradient of a scalar field as arrows on the manifold.

    The gradient at paragraph i points toward whichever neighbour has the
    highest increase in the scalar value — it shows where the narrative
    emotion is rising fastest.
    """
    n = len(coords)
    k = knn_indices.shape[1]

    # Gradient vector: direction of steepest ascent in 2D UMAP space
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    for i in range(n):
        neighbours = knn_indices[i]
        delta_score = score_values[neighbours] - score_values[i]
        delta_pos = coords[neighbours] - coords[i]  # (k, 2)
        # Weighted sum of displacement vectors, weighted by score increase
        weights = np.maximum(delta_score, 0)
        if weights.sum() > 1e-10:
            grad = (weights[:, None] * delta_pos).sum(axis=0)
            norm = np.linalg.norm(grad)
            if norm > 1e-10:
                grad_x[i] = grad[0] / norm
                grad_y[i] = grad[1] / norm

    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=score_values, cmap="YlOrRd",
                    s=18, alpha=0.8, linewidths=0, zorder=2)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

    # Subsample arrows so they don't overlap
    stride = max(1, n // 120)
    idx = np.arange(0, n, stride)
    ax.quiver(coords[idx, 0], coords[idx, 1],
              grad_x[idx], grad_y[idx],
              color="steelblue", alpha=0.7, scale=30, width=0.003, zorder=3)

    ax.set_title(title, fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book",   required=True, type=str)
    parser.add_argument("--model",  required=True, type=str)
    parser.add_argument("--score",  default="wonder",
                        choices=SCORE_FIELDS + ["all"],
                        help="Score field to visualise, or 'all' for a grid of all fields")
    parser.add_argument("--k",      default=10, type=int)
    parser.add_argument("--no-edges", action="store_true",
                        help="Omit KNN graph edges from the plot")
    args = parser.parse_args()

    book_dir = os.path.join("books", args.book)
    output_dir = os.path.join("output", args.book, args.model)
    os.makedirs(output_dir, exist_ok=True)

    processed, scores_by_id, embeddings, umap_coords = load_data(book_dir, args.model)
    paragraphs = processed["paragraphs"]
    book_title = args.book.replace("_", " ").title()

    graph, knn_indices, _ = build_knn_graph(embeddings, k=args.k)

    fields_to_plot = SCORE_FIELDS if args.score == "all" else [args.score]

    for field in fields_to_plot:
        score_values = np.array([
            scores_by_id.get(p["id"], {}).get(field, 0.0)
            for p in paragraphs
        ])

        # Single-field plot with KNN edges
        plot_score_on_manifold(
            umap_coords, graph, score_values,
            title=f'"{field.capitalize()}" on the Story Manifold — {book_title}',
            output_path=os.path.join(output_dir, f"manifold_{field}.png"),
            draw_edges=not args.no_edges,
        )

        # Gradient vector field
        plot_gradient_field(
            umap_coords, score_values, knn_indices,
            title=f'Gradient of "{field.capitalize()}" — {book_title}',
            output_path=os.path.join(output_dir, f"gradient_{field}.png"),
        )

    # --- Always produce the all-scores grid ---
    if args.score == "all":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, field in zip(axes.flat, SCORE_FIELDS):
            vals = np.array([
                scores_by_id.get(p["id"], {}).get(field, 0.0)
                for p in paragraphs
            ])
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                            c=vals, cmap="YlOrRd", s=10, alpha=0.9, linewidths=0)
            ax.set_title(field.capitalize(), fontsize=12)
            ax.axis("off")
            plt.colorbar(sc, ax=ax, fraction=0.05)

        fig.suptitle(f"Narrative Tension Fields — {book_title}", fontsize=16)
        plt.tight_layout()
        grid_path = os.path.join(output_dir, "manifold_all_scores.png")
        plt.savefig(grid_path, dpi=300)
        plt.close()
        print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
