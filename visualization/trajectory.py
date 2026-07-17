"""
Character trajectory visualization: trace a character's paragraphs as a path
on the story manifold and compute geometric properties (length, curvature).

Also computes the geodesic between the character's first and last appearance,
showing the shortest path through semantic space versus the actual narrative path.

Example:
    python visualization/trajectory.py --book alice_wonderland --model bge-m3 --character Alice
    python visualization/trajectory.py --book alice_wonderland --model bge-m3 --character "Queen"
"""

import os
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.knn import build_knn_graph, compute_geodesics, geodesic_path


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def path_length(coords):
    """Euclidean length of a polyline through a sequence of 2D points."""
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def discrete_curvature(coords):
    """
    Turning angle (in radians) at each interior point of a path.
    Returns an array of length len(coords) - 2.
    """
    if len(coords) < 3:
        return np.array([])
    v1 = coords[1:-1] - coords[:-2]   # incoming edge vectors
    v2 = coords[2:]   - coords[1:-1]  # outgoing edge vectors
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    # Avoid division by zero for zero-length edges
    safe = (n1 > 1e-10).ravel() & (n2 > 1e-10).ravel()
    cos_angle = np.ones(len(v1))
    cos_angle[safe] = np.clip(
        (v1[safe] * v2[safe]).sum(axis=1) / (n1[safe, 0] * n2[safe, 0]),
        -1.0, 1.0,
    )
    return np.arccos(cos_angle)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book",      required=True, type=str)
    parser.add_argument("--model",     required=True, type=str)
    parser.add_argument("--character", required=True, type=str,
                        help="Character name (case-insensitive partial match)")
    parser.add_argument("--k",         default=10, type=int)
    args = parser.parse_args()

    book_dir = os.path.join("books", args.book)
    output_dir = os.path.join("output", args.book, args.model)
    os.makedirs(output_dir, exist_ok=True)
    book_title = args.book.replace("_", " ").title()
    char_key = args.character.lower().replace(" ", "_")

    with open(os.path.join(book_dir, "processed.json"), encoding="utf-8") as f:
        processed = json.load(f)
    with open(os.path.join(book_dir, "paragraph_scores.json"), encoding="utf-8") as f:
        scores_raw = json.load(f)

    paragraphs = processed["paragraphs"]
    embeddings = np.load(os.path.join(book_dir, "embeddings", args.model, "embeddings.npy"))
    umap_coords = np.load(os.path.join("output", args.book, args.model, "umap.npy"))

    # Find paragraphs that mention the character (from LLM annotations)
    char_lower = args.character.lower()
    scores_by_id = {s["paragraph_id"]: s for s in scores_raw}

    char_indices = [
        i for i, p in enumerate(paragraphs)
        if any(char_lower in c.lower() for c in scores_by_id.get(p["id"], {}).get("characters", []))
    ]

    if len(char_indices) == 0:
        print(f"No paragraphs found mentioning '{args.character}'.")
        print("Available characters in first few paragraphs:")
        for s in scores_raw[:10]:
            print(" ", s.get("characters", []))
        return

    char_indices = np.array(char_indices)
    char_coords = umap_coords[char_indices]
    print(f"'{args.character}' appears in {len(char_indices)} paragraphs.")

    # --- Build KNN graph for geodesics ---
    print("Building KNN graph...")
    graph, knn_indices, _ = build_knn_graph(embeddings, k=args.k)

    # --- Plot 1: Character trajectory on full manifold ---
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
               c="lightgray", s=8, alpha=0.4, linewidths=0, zorder=1, label="all paragraphs")

    # Colour character paragraphs by their position in the story (0=start, 1=end)
    story_position = char_indices / (len(paragraphs) - 1)
    sc = ax.scatter(char_coords[:, 0], char_coords[:, 1],
                    c=story_position, cmap="plasma", s=60, alpha=0.95,
                    edgecolors="white", linewidths=0.4, zorder=3)

    # Connect consecutive appearances with a purple line
    ax.plot(char_coords[:, 0], char_coords[:, 1],
            color="purple", lw=0.9, alpha=0.5, zorder=2, label="narrative path")

    # Mark first and last appearance
    ax.scatter(*char_coords[0],  color="lime",  s=120, zorder=4, marker="*", label="first appearance")
    ax.scatter(*char_coords[-1], color="red",   s=120, zorder=4, marker="X", label="last appearance")

    plt.colorbar(sc, ax=ax, label="Story position (0=start, 1=end)", fraction=0.04, pad=0.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{args.character}'s Trajectory on the Story Manifold\n{book_title}", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    traj_path = os.path.join(output_dir, f"trajectory_{char_key}.png")
    plt.savefig(traj_path, dpi=300)
    plt.close()
    print(f"Saved: {traj_path}")

    # --- Plot 2: Geodesic vs narrative path ---
    src, tgt = int(char_indices[0]), int(char_indices[-1])
    geo_path = geodesic_path(graph, src, tgt)

    if geo_path:
        geo_coords = umap_coords[geo_path]
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                   c="lightgray", s=8, alpha=0.4, linewidths=0, zorder=1)

        # Narrative path
        ax.plot(char_coords[:, 0], char_coords[:, 1],
                color="purple", lw=1.2, alpha=0.6, zorder=2, label="narrative path")
        ax.scatter(char_coords[:, 0], char_coords[:, 1],
                   c="purple", s=20, alpha=0.7, linewidths=0, zorder=3)

        # Geodesic path
        ax.plot(geo_coords[:, 0], geo_coords[:, 1],
                color="gold", lw=2.0, alpha=0.9, zorder=4, label="geodesic (shortest path)")
        ax.scatter(geo_coords[:, 0], geo_coords[:, 1],
                   c="gold", s=30, alpha=0.9, linewidths=0, zorder=5)

        ax.scatter(*umap_coords[src], color="lime", s=150, zorder=6, marker="*")
        ax.scatter(*umap_coords[tgt], color="red",  s=150, zorder=6, marker="X")

        ax.legend(loc="upper left", fontsize=10)
        ax.set_title(f"Narrative Path vs Geodesic — {args.character}\n{book_title}", fontsize=13)
        ax.axis("off")
        plt.tight_layout()
        geo_path_png = os.path.join(output_dir, f"geodesic_{char_key}.png")
        plt.savefig(geo_path_png, dpi=300)
        plt.close()
        print(f"Saved: {geo_path_png}")

    # --- Plot 3: Chapter appearance bar chart ---
    chapter_ids = np.array([p["chapter_id"] for p in paragraphs])
    char_chapter_ids = chapter_ids[char_indices]
    chapters = processed["chapters"]
    chapter_counts = np.bincount(char_chapter_ids, minlength=len(chapters))
    chapter_labels = [f"Ch {c['chapter_number']}\n{c['title'][:18]}" for c in chapters]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(chapters)), chapter_counts, color="mediumpurple", edgecolor="white")
    ax.set_xticks(range(len(chapters)))
    ax.set_xticklabels(chapter_labels, fontsize=7)
    ax.set_ylabel("Paragraphs mentioning character")
    ax.set_title(f"'{args.character}' Appearances by Chapter — {book_title}")
    plt.tight_layout()
    ch_path = os.path.join(output_dir, f"character_{char_key}_chapters.png")
    plt.savefig(ch_path, dpi=300)
    plt.close()
    print(f"Saved: {ch_path}")

    # --- Geometric summary ---
    traj_len = path_length(char_coords)
    curvatures = discrete_curvature(char_coords)

    print(f"\nGeometric properties of '{args.character}'s trajectory:")
    print(f"  Appearances:          {len(char_indices)} paragraphs")
    print(f"  Narrative path length: {traj_len:.4f}  (UMAP units)")
    if len(curvatures) > 0:
        print(f"  Mean turning angle:   {np.degrees(curvatures.mean()):.2f}°")
        print(f"  Max turning angle:    {np.degrees(curvatures.max()):.2f}°")
    if geo_path:
        geo_len = path_length(geo_coords)
        print(f"  Geodesic length:      {geo_len:.4f}  (UMAP units)")
        print(f"  Path / geodesic ratio: {traj_len / geo_len:.2f}x  "
              f"(higher = more wandering)")

    # Save trajectory indices for downstream analysis
    np.save(os.path.join(output_dir, f"trajectory_{char_key}_indices.npy"), char_indices)


if __name__ == "__main__":
    main()
