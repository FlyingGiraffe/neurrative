"""Create an interactive 3D UMAP paragraph score vector field.

Example:

python visualization/scores_vector_field_umap_3d.py --book alice_wonderland
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))
from umap import UMAP

from scores_vector_field_3d import (
  PROJECT_ROOT,
  build_vector_field,
  load_chapter_ids,
  load_paragraph_texts,
  load_scores,
  make_figure,
  paragraph_global_indices,
)


def main():
  parser = argparse.ArgumentParser(description="Create an interactive 3D UMAP score vector field.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument("--scores-file", type=Path, help="Optional paragraph_scores.json path")
  parser.add_argument("--processed-file", type=Path, help="Optional processed.json path")
  parser.add_argument("--output-dir", type=Path, help="Optional output directory")
  parser.add_argument("--neighbors", type=int, default=15, help="UMAP neighbor count")
  parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP minimum distance")
  parser.add_argument("--metric", default="euclidean", help="UMAP distance metric")
  parser.add_argument("--seed", type=int, default=0, help="UMAP random seed")
  args = parser.parse_args()

  book_dir = PROJECT_ROOT / "books" / args.book
  scores_file = (args.scores_file or book_dir / "paragraph_scores.json").expanduser().resolve()
  processed_file = (args.processed_file or book_dir / "processed.json").expanduser().resolve()
  output_dir = (args.output_dir or PROJECT_ROOT / "output" / args.book / "scores").expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  texts = load_paragraph_texts(processed_file)
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  if len(values) < 2 or not np.isfinite(values).all():
    raise ValueError("At least two paragraphs with complete finite scores are required")

  coordinates = UMAP(
    n_components=3,
    n_neighbors=args.neighbors,
    min_dist=args.min_dist,
    metric=args.metric,
    random_state=args.seed,
  ).fit_transform(values)
  field = build_vector_field(coordinates, paragraph_global_indices(paragraph_ids))
  figure = make_figure(
    paragraph_ids,
    score_names,
    values,
    texts,
    coordinates,
    field,
    None,
    chapter_ids,
    projection="UMAP",
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "score_vector_field_umap_3d.html"
  figure.write_html(output_path, include_plotlyjs=True, full_html=True)

  print(f"Loaded {len(paragraph_ids)} paragraphs with {len(score_names)} scores.")
  print(f"Drew {len(field)} exact UMAP paragraph-to-paragraph vectors.")
  print(output_path)


if __name__ == "__main__":
  main()
