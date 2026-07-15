"""
Spectral visualization: Laplacian eigenvalue spectrum and eigenvector maps.

The Laplacian eigenvectors are the "Fourier modes" of the story manifold.
Visualising them reveals which parts of the story are geometrically similar
and how the narrative decomposes into independent frequency components.

Example:
    python visualization/spectrum.py --book alice_wonderland --model bge-m3
    python visualization/spectrum.py --book alice_wonderland --model bge-m3 --k 15 --n-eigs 30
"""

import os
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.knn import build_knn_graph
from geometry.laplacian import build_laplacian, spectral_decomposition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book",   required=True, type=str)
    parser.add_argument("--model",  required=True, type=str)
    parser.add_argument("--k",      default=10, type=int,
                        help="Number of KNN neighbours for graph construction")
    parser.add_argument("--n-eigs", default=20, type=int,
                        help="Number of Laplacian eigenpairs to compute")
    args = parser.parse_args()

    book_dir = os.path.join("books", args.book)
    output_dir = os.path.join("output", args.book, args.model)
    os.makedirs(output_dir, exist_ok=True)
    book_title = args.book.replace("_", " ").title()

    with open(os.path.join(book_dir, "processed.json"), encoding="utf-8") as f:
        processed = json.load(f)
    paragraphs = processed["paragraphs"]
    paragraph_ids = np.arange(len(paragraphs))
    chapter_ids = np.array([p["chapter_id"] for p in paragraphs])

    embeddings = np.load(os.path.join(book_dir, "embeddings", args.model, "embeddings.npy"))
    umap_coords = np.load(os.path.join("output", args.book, args.model, "umap.npy"))

    # Build graph → Laplacian → eigenpairs
    print("Building KNN graph...")
    graph, _, _ = build_knn_graph(embeddings, k=args.k)

    print("Building Laplacian...")
    L, degree = build_laplacian(graph, normalized=True)

    print(f"Computing {args.n_eigs} eigenpairs...")
    eigenvalues, eigenvectors = spectral_decomposition(L, k=args.n_eigs)

    # --- Plot 1: Eigenvalue spectrum ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(eigenvalues)), eigenvalues, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="trivial (λ≈0)")
    ax.set_xlabel("Eigenvector index", fontsize=11)
    ax.set_ylabel("Eigenvalue λ", fontsize=11)
    ax.set_title(f"Laplacian Eigenvalue Spectrum — {book_title}", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eigenvalue_spectrum.png"), dpi=300)
    plt.close()
    print("Saved: eigenvalue_spectrum.png")

    # --- Plot 2: First 6 non-trivial eigenvectors on UMAP layout ---
    n_show = min(6, args.n_eigs - 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat[:n_show]):
        eig_idx = i + 1  # skip eigenvector 0 (constant, no information)
        vals = eigenvectors[:, eig_idx]
        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=vals, cmap="RdBu_r", s=14, alpha=0.9, linewidths=0)
        ax.set_title(f"Eigenvector {eig_idx}  (λ = {eigenvalues[eig_idx]:.4f})", fontsize=10)
        ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.05)

    fig.suptitle(f"Laplacian Eigenvectors on Story Manifold — {book_title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eigenvectors.png"), dpi=300)
    plt.close()
    print("Saved: eigenvectors.png")

    # --- Plot 3: Spectral embedding (eigenvectors 1 vs 2), coloured by story position ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (colors, label, cmap) in zip(axes, [
        (paragraph_ids, "Paragraph index (story position)", "plasma"),
        (chapter_ids,   "Chapter",                         "tab20"),
    ]):
        sc = ax.scatter(eigenvectors[:, 1], eigenvectors[:, 2],
                        c=colors, cmap=cmap, s=20, alpha=0.9, linewidths=0)
        ax.plot(eigenvectors[:, 1], eigenvectors[:, 2],
                color="black", lw=0.3, alpha=0.15, zorder=1)
        ax.set_xlabel("Eigenvector 1", fontsize=11)
        ax.set_ylabel("Eigenvector 2", fontsize=11)
        ax.set_title(f"Spectral Embedding — coloured by {label}", fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.05)

    plt.suptitle(f"Spectral Embedding — {book_title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectral_embedding.png"), dpi=300)
    plt.close()
    print("Saved: spectral_embedding.png")

    # --- Plot 4: Eigenvector values as time series (each eigenvector vs paragraph index) ---
    n_ts = min(5, args.n_eigs - 1)
    fig, axes = plt.subplots(n_ts, 1, figsize=(14, 2.5 * n_ts), sharex=True)
    for i, ax in enumerate(axes):
        eig_idx = i + 1
        ax.plot(paragraph_ids, eigenvectors[:, eig_idx], lw=0.8, color="steelblue")
        ax.set_ylabel(f"Eig {eig_idx}", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5, linestyle="--")

        # Shade chapters alternately for orientation
        for ch in processed["chapters"]:
            start, end = ch["paragraph_range"]
            if ch["chapter_id"] % 2 == 0:
                ax.axvspan(start, end, alpha=0.06, color="orange")

    axes[-1].set_xlabel("Paragraph index", fontsize=11)
    axes[0].set_title(f"Eigenvector Time Series — {book_title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eigenvector_timeseries.png"), dpi=300)
    plt.close()
    print("Saved: eigenvector_timeseries.png")

    # Save eigenpairs for downstream use
    np.save(os.path.join(output_dir, "eigenvalues.npy"), eigenvalues)
    np.save(os.path.join(output_dir, "eigenvectors.npy"), eigenvectors)
    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
