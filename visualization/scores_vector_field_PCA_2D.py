"""Create a vector field from paragraph-to-paragraph score changes.

Each paragraph is a six-dimensional score vector. PCA projects the score
vectors into two dimensions, then one exact arrow connects every paragraph to
the paragraph immediately following it.

Example:

python visualization/scores_vector_field.py --book alice_wonderland
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scores import PROJECT_ROOT, load_chapter_ids, load_scores, paragraph_global_indices


def build_vector_field(coordinates, global_indices=None):
  """Return one exact vector from every paragraph to its successor."""
  starts = coordinates[:-1]
  changes = coordinates[1:] - coordinates[:-1]
  if global_indices is not None:
    consecutive = np.diff(global_indices) == 1
    starts = starts[consecutive]
    changes = changes[consecutive]
  return np.column_stack((starts, changes))


def plot_vector_field(
  coordinates,
  field,
  explained_variance,
  output_path,
  projection="PCA",
  transition_chapters=None,
):
  """Draw every projected paragraph and exact paragraph-to-paragraph arrow."""
  fig, axis = plt.subplots(figsize=(11, 9))
  paragraph_order = np.arange(len(coordinates))

  points = axis.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=paragraph_order,
    cmap="plasma",
    s=12,
    alpha=1,
    linewidths=0,
    zorder=3,
  )

  if len(field):
    starts = field[:, :2]
    vectors = field[:, 2:4]
    ends = starts + vectors
    magnitudes = np.linalg.norm(vectors, axis=1)
    nonzero = magnitudes > 1e-12

    directions = vectors[nonzero] / magnitudes[nonzero, np.newaxis]
    perpendiculars = np.column_stack((-directions[:, 1], directions[:, 0]))
    plot_extent = np.linalg.norm(np.ptp(coordinates, axis=0))
    head_lengths = np.minimum(0.018 * plot_extent, 0.35 * magnitudes[nonzero])
    head_widths = 0.48 * head_lengths
    tips = ends[nonzero]
    bases = tips - directions * head_lengths[:, np.newaxis]
    wing_1 = bases + perpendiculars * head_widths[:, np.newaxis]
    wing_2 = bases - perpendiculars * head_widths[:, np.newaxis]

    if transition_chapters is None:
      transition_chapters = np.zeros((len(field), 2), dtype=int)
    transition_chapters = transition_chapters[nonzero]
    chapter_ids = np.unique(transition_chapters)
    chapter_colors = {
      chapter_id: matplotlib.colormaps["tab20"](index / max(len(chapter_ids) - 1, 1))
      for index, chapter_id in enumerate(chapter_ids)
    }
    midpoints = (starts[nonzero] + bases) / 2
    tail_segments = np.concatenate(
      (
        np.stack((starts[nonzero], midpoints), axis=1),
        np.stack((midpoints, bases), axis=1),
      )
    )
    tail_colors = [chapter_colors[chapter_id] for chapter_id in transition_chapters[:, 0]]
    tail_colors += [chapter_colors[chapter_id] for chapter_id in transition_chapters[:, 1]]
    head_colors = [chapter_colors[chapter_id] for chapter_id in transition_chapters[:, 1]]

    tails = LineCollection(
      tail_segments,
      colors=tail_colors,
      linewidths=1.8,
      alpha=0.18,
      zorder=1,
    )
    heads = PolyCollection(
      np.stack((tips, wing_1, wing_2), axis=1),
      facecolors=head_colors,
      edgecolors="none",
      alpha=1,
      zorder=2,
    )
    axis.add_collection(tails)
    axis.add_collection(heads)
    legend_options = {
      "ncols": 4,
      "loc": "upper center",
      "bbox_to_anchor": (0.5, -0.12),
    } if projection == "UMAP" else {
      "ncols": 2,
      "loc": "upper right",
    }
    axis.legend(
      handles=[Patch(color=chapter_colors[chapter_id], label=f"Chapter ID {chapter_id}") for chapter_id in chapter_ids],
      title="Arrow chapter",
      frameon=False,
      **legend_options,
    )

  order_colorbar = fig.colorbar(points, ax=axis, pad=0.02, fraction=0.046)
  order_colorbar.set_label("Paragraph order")

  if projection == "PCA":
    x_label = f"Score PC1 ({explained_variance[0]:.1%} variance)"
    y_label = f"Score PC2 ({explained_variance[1]:.1%} variance)"
  else:
    x_label = f"Score {projection}-1"
    y_label = f"Score {projection}-2"
  axis.set(
    title=f"Paragraph-to-paragraph 2D narrative vectors ({projection})",
    xlabel=x_label,
    ylabel=y_label,
  )
  axis.grid(alpha=0.12)
  axis.set_aspect("equal", adjustable="datalim")
  if projection == "UMAP":
    fig.tight_layout(rect=(0, 0.16, 1, 1))
  else:
    fig.tight_layout()
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Create a vector field from paragraph scores.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument(
    "--scores-file",
    type=Path,
    help="Optional path to paragraph_scores.json (overrides --book)",
  )
  parser.add_argument("--output-dir", type=Path, help="Optional output directory")
  parser.add_argument("--processed-file", type=Path, help="Optional processed.json path")
  args = parser.parse_args()

  scores_file = args.scores_file or PROJECT_ROOT / "books" / args.book / "paragraph_scores.json"
  processed_file = args.processed_file or PROJECT_ROOT / "books" / args.book / "processed.json"
  output_dir = args.output_dir or PROJECT_ROOT / "output" / args.book / "scores"
  scores_file = scores_file.expanduser().resolve()
  processed_file = processed_file.expanduser().resolve()
  output_dir = output_dir.expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  if len(values) < 2:
    raise ValueError("At least two paragraphs are required to calculate score changes")
  if not np.isfinite(values).all():
    raise ValueError("Every paragraph must contain a finite value for every score")

  pca = PCA(n_components=2)
  coordinates = pca.fit_transform(values)
  global_indices = paragraph_global_indices(paragraph_ids)
  consecutive = np.diff(global_indices) == 1
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  transition_chapters = np.column_stack((chapter_ids[:-1][consecutive], chapter_ids[1:][consecutive]))
  field = build_vector_field(coordinates, global_indices)

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "score_vector_field.png"
  plot_vector_field(
    coordinates,
    field,
    pca.explained_variance_ratio_,
    output_path,
    transition_chapters=transition_chapters,
  )

  print(f"Loaded {len(paragraph_ids)} paragraphs with {len(score_names)} scores.")
  print(f"Drew {len(field)} exact paragraph-to-paragraph arrows.")
  print(output_path)


if __name__ == "__main__":
  main()
