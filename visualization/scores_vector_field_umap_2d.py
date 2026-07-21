"""Create a 2D UMAP paragraph-to-paragraph score vector field.

Example:

python visualization/scores_vector_field_umap_2d.py --book alice_wonderland
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))
from umap import UMAP

from scores import PROJECT_ROOT, load_chapter_ids, load_scores, paragraph_global_indices
from scores_vector_field import build_vector_field, plot_vector_field


def main():
  parser = argparse.ArgumentParser(description="Create a 2D UMAP vector field from paragraph scores.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument("--scores-file", type=Path, help="Optional paragraph_scores.json path")
  parser.add_argument("--output-dir", type=Path, help="Optional output directory")
  parser.add_argument("--processed-file", type=Path, help="Optional processed.json path")
  parser.add_argument("--neighbors", type=int, default=15, help="UMAP neighbor count")
  parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP minimum distance")
  parser.add_argument("--metric", default="euclidean", help="UMAP distance metric")
  parser.add_argument("--seed", type=int, default=0, help="UMAP random seed")
  args = parser.parse_args()

  scores_file = args.scores_file or PROJECT_ROOT / "books" / args.book / "paragraph_scores.json"
  processed_file = args.processed_file or PROJECT_ROOT / "books" / args.book / "processed.json"
  output_dir = args.output_dir or PROJECT_ROOT / "output" / args.book / "scores"
  scores_file = scores_file.expanduser().resolve()
  processed_file = processed_file.expanduser().resolve()
  output_dir = output_dir.expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  if len(values) < 2 or not np.isfinite(values).all():
    raise ValueError("At least two paragraphs with complete finite scores are required")

  coordinates = UMAP(
    n_components=2,
    n_neighbors=args.neighbors,
    min_dist=args.min_dist,
    metric=args.metric,
    random_state=args.seed,
  ).fit_transform(values)
  global_indices = paragraph_global_indices(paragraph_ids)
  consecutive = np.diff(global_indices) == 1
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  transition_chapters = np.column_stack((chapter_ids[:-1][consecutive], chapter_ids[1:][consecutive]))
  field = build_vector_field(coordinates, global_indices)

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "score_vector_field_umap_2d.png"
  plot_vector_field(
    coordinates,
    field,
    None,
    output_path,
    projection="UMAP",
    transition_chapters=transition_chapters,
  )

  print(f"Loaded {len(paragraph_ids)} paragraphs with {len(score_names)} scores.")
  print(f"Drew {len(field)} exact UMAP paragraph-to-paragraph arrows.")
  print(output_path)


if __name__ == "__main__":
  main()
