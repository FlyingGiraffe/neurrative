"""
Visualize paragraph embeddings.

Example:

python visualization/embedding.py --book alice_wonderland --model bge-m3
"""

import os
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


def scatter_plot(
  coords,
  colors,
  title,
  xlabel,
  ylabel,
  output_path,
  cmap="viridis",
  connect_story=False,
):

  plt.figure(figsize=(8, 8))

  if connect_story:
    plt.plot(
      coords[:, 0], coords[:, 1], color="black", linewidth=0.5, alpha=0.3, zorder=1
    )

  sc = plt.scatter(
    coords[:, 0], coords[:, 1], c=colors, cmap=cmap, s=20, alpha=0.9, linewidths=0, zorder=2
  )

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.colorbar(sc)

  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True, type=str)
  parser.add_argument("--model", required=True, type=str)
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--neighbors", default=5, type=int)
  parser.add_argument("--min-dist", default=0.1, type=float)
  args = parser.parse_args()

  book_dir = os.path.join("books", args.book)
  processed_file = os.path.join(book_dir, "processed.json")
  embedding_file = os.path.join(book_dir, "embeddings", args.model, "embeddings.npy")

  with open(processed_file, "r") as f:
    processed = json.load(f)
  
  paragraphs = processed["paragraphs"]
  chapter_ids = np.array([p["chapter_id"] for p in paragraphs])
  paragraph_ids = np.arange(len(paragraphs))

  embeddings = np.load(embedding_file)

  output_dir = os.path.join("output", args.book, args.model)
  os.makedirs(output_dir, exist_ok=True)

  print("Running PCA...")

  pca = PCA(n_components=3)

  pca_coords = pca.fit_transform(embeddings)
  np.save(os.path.join(output_dir, "pca.npy"), pca_coords)

  scatter_plot(
    pca_coords,
    chapter_ids,
    "PCA (colored by chapter)",
    "PC1",
    "PC2",
    os.path.join(output_dir, "pca_chapter.png"),
  )

  print("Running UMAP...")

  umap = UMAP(
    n_components=3,
    n_neighbors=args.neighbors,
    min_dist=args.min_dist,
    random_state=args.seed,
  )

  umap_coords = umap.fit_transform(embeddings)
  np.save(os.path.join(output_dir, "umap.npy"), umap_coords)

  scatter_plot(
    umap_coords,
    chapter_ids,
    "UMAP (colored by chapter)",
    "UMAP-1",
    "UMAP-2",
    os.path.join(output_dir, "umap_chapter.png"),
  )

  scatter_plot(
    umap_coords,
    paragraph_ids,
    "UMAP (colored by paragraph index)",
    "UMAP-1",
    "UMAP-2",
    os.path.join(output_dir, "umap_paragraph.png"),
    cmap="plasma",
  )

  scatter_plot(
    umap_coords,
    paragraph_ids,
    "Storyline through the semantic manifold",
    "UMAP-1",
    "UMAP-2",
    os.path.join(output_dir, "storyline.png"),
    cmap="plasma",
    connect_story=True,
  )

  print()
  print("Done.")
  print(output_dir)


if __name__ == "__main__":
  main()
