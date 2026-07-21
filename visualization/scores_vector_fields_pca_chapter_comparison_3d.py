"""Generate interactive 3D counterparts to the PCA chapter comparisons."""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from scores_vector_field_3d import (
  PROJECT_ROOT,
  load_chapter_ids,
  load_scores,
  paragraph_global_indices,
)
from scores_vector_fields_pca_by_chapter import ARROW_COLOR, emotion_colors


def add_arrows(figure, coordinates, global_indices, color, extent, row=None, col=None, name=None, showlegend=False):
  """Add translucent shafts and small solid triangular heads."""
  consecutive = np.diff(global_indices) == 1
  starts = coordinates[:-1][consecutive]
  vectors = (coordinates[1:] - coordinates[:-1])[consecutive]
  if not len(starts):
    return
  ends = starts + vectors
  count = len(starts)
  trace_args = {"row": row, "col": col} if row is not None else {}
  figure.add_trace(
    go.Scatter3d(
      x=np.column_stack((starts[:, 0], ends[:, 0], np.full(count, np.nan))).ravel(),
      y=np.column_stack((starts[:, 1], ends[:, 1], np.full(count, np.nan))).ravel(),
      z=np.column_stack((starts[:, 2], ends[:, 2], np.full(count, np.nan))).ravel(),
      mode="lines",
      line={"color": color, "width": 2.5},
      opacity=0.16,
      hoverinfo="skip",
      name=name or "Chapter path",
      showlegend=showlegend,
      legendgroup=name,
      connectgaps=False,
    ),
    **trace_args,
  )

  magnitudes = np.linalg.norm(vectors, axis=1)
  nonzero = magnitudes > 1e-12
  directions = vectors[nonzero] / magnitudes[nonzero, np.newaxis]
  helpers = np.tile(np.array([0.0, 0.0, 1.0]), (len(directions), 1))
  helpers[np.abs(directions[:, 2]) > 0.9] = np.array([0.0, 1.0, 0.0])
  perpendiculars = np.cross(directions, helpers)
  perpendiculars /= np.linalg.norm(perpendiculars, axis=1, keepdims=True)
  head_lengths = np.minimum(0.0045 * extent, 0.26 * magnitudes[nonzero])
  head_widths = 0.40 * head_lengths
  tips = ends[nonzero]
  bases = tips - directions * head_lengths[:, np.newaxis]
  wing_1 = bases + perpendiculars * head_widths[:, np.newaxis]
  wing_2 = bases - perpendiculars * head_widths[:, np.newaxis]
  vertices = np.column_stack((tips, wing_1, wing_2)).reshape(-1, 3)
  triangle_starts = np.arange(0, len(vertices), 3)
  figure.add_trace(
    go.Mesh3d(
      x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
      i=triangle_starts, j=triangle_starts + 1, k=triangle_starts + 2,
      color=color, opacity=1, flatshading=True, hoverinfo="skip", showlegend=False,
    ),
    **trace_args,
  )


def point_trace(coordinates, mask, color, name, customdata, size, showlegend):
  return go.Scatter3d(
    x=coordinates[mask, 0], y=coordinates[mask, 1], z=coordinates[mask, 2],
    mode="markers",
    name=name,
    customdata=customdata,
    marker={"size": size, "color": color, "opacity": 1},
    hovertemplate=(
      "<b>%{customdata[0]}</b><br>Global index: %{customdata[1]}<br>"
      "Chapter ID: %{customdata[2]}<br>Dominant emotion: %{customdata[3]}<extra></extra>"
    ),
    showlegend=showlegend,
    legendgroup=name,
  )


def comparison_grid(coordinates, paragraph_ids, global_indices, chapter_ids, score_names, values, variance):
  chapters = np.unique(chapter_ids)
  dominant = np.argmax(values, axis=1)
  colors = emotion_colors(score_names)
  rows, columns = 4, 3
  figure = make_subplots(
    rows=rows,
    cols=columns,
    specs=[[{"type": "scene"} for _ in range(columns)] for _ in range(rows)],
    subplot_titles=[f"Chapter ID {chapter_id}" for chapter_id in chapters],
    horizontal_spacing=0.025,
    vertical_spacing=0.045,
  )
  extent = np.linalg.norm(np.ptp(coordinates, axis=0))
  ranges = []
  for dimension in range(3):
    low, high = coordinates[:, dimension].min(), coordinates[:, dimension].max()
    padding = max((high - low) * 0.05, 0.02)
    ranges.append([low - padding, high + padding])

  for panel, chapter_id in enumerate(chapters):
    row, col = panel // columns + 1, panel % columns + 1
    active = chapter_ids == chapter_id
    for score_index, emotion in enumerate(score_names):
      background = (~active) & (dominant == score_index)
      if not background.any():
        continue
      background_data = [
        [paragraph_ids[index], global_indices[index], chapter_ids[index], emotion.title()]
        for index in np.flatnonzero(background)
      ]
      figure.add_trace(
        point_trace(
          coordinates,
          background,
          colors[emotion],
          emotion.title(),
          background_data,
          2.4,
          panel == 0,
        ),
        row=row,
        col=col,
      )
    add_arrows(
      figure,
      coordinates[active],
      global_indices[active],
      ARROW_COLOR,
      extent,
      row=row,
      col=col,
      name="Active chapter path",
      showlegend=panel == 0,
    )
    for score_index, emotion in enumerate(score_names):
      mask = active & (dominant == score_index)
      if not mask.any():
        continue
      customdata = [
        [paragraph_ids[index], global_indices[index], chapter_ids[index], emotion.title()]
        for index in np.flatnonzero(mask)
      ]
      figure.add_trace(
        point_trace(coordinates, mask, colors[emotion], emotion.title(), customdata, 4.2, False),
        row=row,
        col=col,
      )

  figure.update_scenes(
    xaxis={"title": "PC1", "range": ranges[0]},
    yaxis={"title": "PC2", "range": ranges[1]},
    zaxis={"title": "PC3", "range": ranges[2]},
    aspectmode="data",
    camera={"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
  )
  figure.update_layout(
    title=f"Full-book 3D PCA: one chapter path active per scene · PC1 {variance[0]:.1%}, PC2 {variance[1]:.1%}, PC3 {variance[2]:.1%}",
    template="plotly_white",
    width=1500,
    height=1750,
    margin={"l": 0, "r": 180, "t": 80, "b": 0},
    legend={"title": "Points and active path", "itemsizing": "constant"},
    hovermode="closest",
  )
  return figure


def combined_plot(coordinates, paragraph_ids, global_indices, chapter_ids, score_names, values, variance):
  chapters = np.unique(chapter_ids)
  dominant = np.argmax(values, axis=1)
  emotion_map = emotion_colors(score_names)
  palette = qualitative.Dark24
  chapter_colors = {chapter: palette[index % len(palette)] for index, chapter in enumerate(chapters)}
  extent = np.linalg.norm(np.ptp(coordinates, axis=0))
  figure = go.Figure()

  for chapter_id in chapters:
    mask = chapter_ids == chapter_id
    add_arrows(
      figure,
      coordinates[mask],
      global_indices[mask],
      chapter_colors[chapter_id],
      extent,
      name=f"Chapter ID {chapter_id} path",
      showlegend=True,
    )
  for score_index, emotion in enumerate(score_names):
    mask = dominant == score_index
    customdata = [
      [paragraph_ids[index], global_indices[index], chapter_ids[index], emotion.title()]
      for index in np.flatnonzero(mask)
    ]
    figure.add_trace(point_trace(coordinates, mask, emotion_map[emotion], emotion.title(), customdata, 5, True))

  figure.update_layout(
    title="Full-book 3D PCA with disconnected within-chapter paths",
    template="plotly_white",
    margin={"l": 0, "r": 220, "t": 60, "b": 0},
    legend={"title": "Chapter paths and emotions", "itemsizing": "constant"},
    scene={
      "xaxis_title": f"PC1 ({variance[0]:.1%})",
      "yaxis_title": f"PC2 ({variance[1]:.1%})",
      "zaxis_title": f"PC3 ({variance[2]:.1%})",
      "aspectmode": "data",
    },
    hovermode="closest",
    hoverdistance=20,
  )
  return figure


def main():
  parser = argparse.ArgumentParser(description="Generate 3D PCA chapter comparisons.")
  parser.add_argument("--book", default="alice_wonderland")
  parser.add_argument("--scores-file", type=Path)
  parser.add_argument("--processed-file", type=Path)
  parser.add_argument("--output-dir", type=Path)
  args = parser.parse_args()

  book_dir = PROJECT_ROOT / "books" / args.book
  scores_file = (args.scores_file or book_dir / "paragraph_scores.json").expanduser().resolve()
  processed_file = (args.processed_file or book_dir / "processed.json").expanduser().resolve()
  output_dir = (args.output_dir or PROJECT_ROOT / "output" / args.book / "scores" / "chapters").expanduser().resolve()
  paragraph_ids, score_names, values = load_scores(scores_file)
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  global_indices = paragraph_global_indices(paragraph_ids)
  pca = PCA(n_components=3)
  coordinates = pca.fit_transform(values)
  output_dir.mkdir(parents=True, exist_ok=True)

  grid_path = output_dir / "chapter_path_comparison_grid_3d.html"
  combined_path = output_dir / "all_chapters_disconnected_paths_3d.html"
  comparison_grid(coordinates, paragraph_ids, global_indices, chapter_ids, score_names, values, pca.explained_variance_ratio_).write_html(
    grid_path, include_plotlyjs=True, full_html=True
  )
  combined_plot(coordinates, paragraph_ids, global_indices, chapter_ids, score_names, values, pca.explained_variance_ratio_).write_html(
    combined_path, include_plotlyjs=True, full_html=True
  )
  print(grid_path)
  print(combined_path)


if __name__ == "__main__":
  main()
