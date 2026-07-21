"""Compare chapter-only paths in one shared full-book PCA space.

Generates:
- A small-multiple grid: all points in every panel, arrows for one chapter.
- A combined plot: all points and only within-chapter arrows.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scores_vector_field_3d import (
  PROJECT_ROOT,
  load_chapter_ids,
  load_scores,
  paragraph_global_indices,
)
from scores_vector_fields_pca_by_chapter import ARROW_COLOR, emotion_colors


def draw_arrow_layer(axis, coordinates, global_indices, color, full_extent, tail_alpha=0.14):
  """Draw exact transitions for one chapter without bridging index gaps."""
  consecutive = np.diff(global_indices) == 1
  starts = coordinates[:-1][consecutive]
  vectors = (coordinates[1:] - coordinates[:-1])[consecutive]
  if not len(starts):
    return

  ends = starts + vectors
  magnitudes = np.linalg.norm(vectors, axis=1)
  nonzero = magnitudes > 1e-12
  directions = vectors[nonzero] / magnitudes[nonzero, np.newaxis]
  perpendiculars = np.column_stack((-directions[:, 1], directions[:, 0]))
  head_lengths = np.minimum(0.006 * full_extent, 0.28 * magnitudes[nonzero])
  head_widths = 0.42 * head_lengths
  tips = ends[nonzero]
  bases = tips - directions * head_lengths[:, np.newaxis]
  wing_1 = bases + perpendiculars * head_widths[:, np.newaxis]
  wing_2 = bases - perpendiculars * head_widths[:, np.newaxis]

  axis.add_collection(
    LineCollection(
      np.stack((starts[nonzero], bases), axis=1),
      colors=color,
      linewidths=1.2,
      alpha=tail_alpha,
      zorder=1,
    )
  )
  axis.add_collection(
    PolyCollection(
      np.stack((tips, wing_1, wing_2), axis=1),
      facecolors=color,
      edgecolors="none",
      alpha=1,
      zorder=2,
    )
  )


def shared_limits(coordinates):
  """Return padded limits used by every comparison panel."""
  minimum = coordinates.min(axis=0)
  maximum = coordinates.max(axis=0)
  padding = np.maximum((maximum - minimum) * 0.06, 0.02)
  return (minimum[0] - padding[0], maximum[0] + padding[0]), (
    minimum[1] - padding[1], maximum[1] + padding[1]
  )


def draw_comparison_grid(
  coordinates,
  global_indices,
  chapter_ids,
  score_names,
  values,
  explained_variance,
  output_path,
):
  """Show all points in every panel and arrows for one chapter at a time."""
  chapters = np.unique(chapter_ids)
  dominant = np.argmax(values, axis=1)
  colors = emotion_colors(score_names)
  x_limits, y_limits = shared_limits(coordinates)
  full_extent = np.linalg.norm(np.ptp(coordinates, axis=0))
  columns = 3
  rows = int(np.ceil(len(chapters) / columns))
  fig, axes = plt.subplots(rows, columns, figsize=(18, 5.2 * rows), sharex=True, sharey=True)
  axes = np.asarray(axes).ravel()

  for axis, chapter_id in zip(axes, chapters):
    active = chapter_ids == chapter_id
    for score_index, emotion in enumerate(score_names):
      background = (~active) & (dominant == score_index)
      if background.any():
        axis.scatter(
          coordinates[background, 0],
          coordinates[background, 1],
          color=colors[emotion],
          s=8,
          linewidths=0,
          zorder=0,
        )
    draw_arrow_layer(
      axis,
      coordinates[active],
      global_indices[active],
      ARROW_COLOR,
      full_extent,
      tail_alpha=0.16,
    )
    for score_index, emotion in enumerate(score_names):
      mask = active & (dominant == score_index)
      if mask.any():
        axis.scatter(
          coordinates[mask, 0],
          coordinates[mask, 1],
          color=colors[emotion],
          s=14,
          linewidths=0,
          zorder=3,
        )
    axis.set_title(f"Chapter ID {chapter_id}")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.10)

  for axis in axes[len(chapters):]:
    axis.set_visible(False)
  for row in range(rows):
    axes[row * columns].set_ylabel(f"Score PC2 ({explained_variance[1]:.1%})")
  for axis in axes[-columns:]:
    if axis.get_visible():
      axis.set_xlabel(f"Score PC1 ({explained_variance[0]:.1%})")

  emotion_handles = [
    Line2D([0], [0], marker="o", linestyle="none", color=colors[name], label=name.title())
    for name in score_names
  ]
  emotion_handles.append(Line2D([0], [0], color=ARROW_COLOR, alpha=0.35, label="Active chapter path"))
  fig.suptitle("Full-book PCA space with one chapter path active per panel")
  fig.legend(handles=emotion_handles, loc="lower center", ncols=7, frameon=False)
  fig.tight_layout(rect=(0, 0.045, 1, 0.97))
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def draw_all_chapter_paths(
  coordinates,
  global_indices,
  chapter_ids,
  score_names,
  values,
  explained_variance,
  output_path,
):
  """Draw all points and only within-chapter paths in one PCA plot."""
  chapters = np.unique(chapter_ids)
  dominant = np.argmax(values, axis=1)
  colors = emotion_colors(score_names)
  chapter_palette = matplotlib.colormaps["tab20"]
  chapter_colors = {
    chapter_id: chapter_palette(index / max(len(chapters) - 1, 1))
    for index, chapter_id in enumerate(chapters)
  }
  full_extent = np.linalg.norm(np.ptp(coordinates, axis=0))
  fig, axis = plt.subplots(figsize=(14, 11))

  for chapter_id in chapters:
    mask = chapter_ids == chapter_id
    draw_arrow_layer(
      axis,
      coordinates[mask],
      global_indices[mask],
      chapter_colors[chapter_id],
      full_extent,
      tail_alpha=0.18,
    )

  for score_index, emotion in enumerate(score_names):
    mask = dominant == score_index
    axis.scatter(
      coordinates[mask, 0],
      coordinates[mask, 1],
      color=colors[emotion],
      s=16,
      linewidths=0,
      zorder=3,
    )

  emotion_handles = [
    Line2D([0], [0], marker="o", linestyle="none", color=colors[name], label=name.title())
    for name in score_names
  ]
  chapter_handles = [
    Patch(color=chapter_colors[chapter_id], label=f"Chapter ID {chapter_id}")
    for chapter_id in chapters
  ]
  emotion_legend = axis.legend(
    handles=emotion_handles,
    title="Dominant emotion",
    loc="upper left",
    frameon=False,
    ncols=2,
  )
  axis.add_artist(emotion_legend)
  axis.legend(
    handles=chapter_handles,
    title="Within-chapter paths",
    loc="upper right",
    frameon=False,
    ncols=2,
  )
  axis.set(
    title="Full-book PCA with disconnected within-chapter paths",
    xlabel=f"Score PC1 ({explained_variance[0]:.1%} variance)",
    ylabel=f"Score PC2 ({explained_variance[1]:.1%} variance)",
  )
  axis.set_aspect("equal", adjustable="datalim")
  axis.grid(alpha=0.10)
  fig.tight_layout()
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Generate full-book PCA chapter comparisons.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument("--scores-file", type=Path, help="Optional paragraph_scores.json path")
  parser.add_argument("--processed-file", type=Path, help="Optional processed.json path")
  parser.add_argument("--output-dir", type=Path, help="Optional output directory")
  args = parser.parse_args()

  book_dir = PROJECT_ROOT / "books" / args.book
  scores_file = (args.scores_file or book_dir / "paragraph_scores.json").expanduser().resolve()
  processed_file = (args.processed_file or book_dir / "processed.json").expanduser().resolve()
  output_dir = (
    args.output_dir or PROJECT_ROOT / "output" / args.book / "scores" / "chapters"
  ).expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  global_indices = paragraph_global_indices(paragraph_ids)
  pca = PCA(n_components=2)
  coordinates = pca.fit_transform(values)
  output_dir.mkdir(parents=True, exist_ok=True)

  grid_path = output_dir / "chapter_path_comparison_grid.png"
  combined_path = output_dir / "all_chapters_disconnected_paths.png"
  draw_comparison_grid(
    coordinates,
    global_indices,
    chapter_ids,
    score_names,
    values,
    pca.explained_variance_ratio_,
    grid_path,
  )
  draw_all_chapter_paths(
    coordinates,
    global_indices,
    chapter_ids,
    score_names,
    values,
    pca.explained_variance_ratio_,
    combined_path,
  )
  print(grid_path)
  print(combined_path)


if __name__ == "__main__":
  main()
