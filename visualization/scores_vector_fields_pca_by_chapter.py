"""Generate separate 2D and interactive 3D PCA score fields per chapter.

Example:

python visualization/scores_vector_fields_pca_by_chapter.py --book alice_wonderland
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scores_vector_field_3d import (
  PROJECT_ROOT,
  first_sentence,
  load_chapter_ids,
  load_paragraph_texts,
  load_scores,
  paragraph_global_indices,
)


ARROW_COLOR = "#626b73"
EMOTION_PALETTE = (
  "#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499",
  "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888",
)


def chapter_vectors(coordinates, global_indices):
  """Return exact consecutive transitions without bridging filtered gaps."""
  consecutive = np.diff(global_indices) == 1
  starts = coordinates[:-1][consecutive]
  changes = (coordinates[1:] - coordinates[:-1])[consecutive]
  sources = np.flatnonzero(consecutive)
  return starts, changes, sources


def emotion_colors(score_names):
  """Return a stable color for every score/emotion name."""
  return {name: EMOTION_PALETTE[index % len(EMOTION_PALETTE)] for index, name in enumerate(score_names)}


def draw_2d(
  coordinates,
  global_indices,
  score_names,
  values,
  chapter_id,
  explained_variance,
  output_path,
):
  """Draw one chapter with translucent gray tails and small solid heads."""
  dominant = np.argmax(values, axis=1)
  colors = emotion_colors(score_names)
  fig, axis = plt.subplots(figsize=(11, 9))

  starts, vectors, _ = chapter_vectors(coordinates, global_indices)
  if len(starts):
    ends = starts + vectors
    magnitudes = np.linalg.norm(vectors, axis=1)
    nonzero = magnitudes > 1e-12
    directions = vectors[nonzero] / magnitudes[nonzero, np.newaxis]
    perpendiculars = np.column_stack((-directions[:, 1], directions[:, 0]))
    extent = np.linalg.norm(np.ptp(coordinates, axis=0))
    head_lengths = np.minimum(0.008 * extent, 0.30 * magnitudes[nonzero])
    head_widths = 0.42 * head_lengths
    tips = ends[nonzero]
    bases = tips - directions * head_lengths[:, np.newaxis]
    wing_1 = bases + perpendiculars * head_widths[:, np.newaxis]
    wing_2 = bases - perpendiculars * head_widths[:, np.newaxis]

    axis.add_collection(
      LineCollection(
        np.stack((starts[nonzero], bases), axis=1),
        colors=ARROW_COLOR,
        linewidths=1.4,
        alpha=0.14,
        zorder=1,
      )
    )
    axis.add_collection(
      PolyCollection(
        np.stack((tips, wing_1, wing_2), axis=1),
        facecolors=ARROW_COLOR,
        edgecolors="none",
        alpha=1,
        zorder=2,
      )
    )

  for score_index, emotion in enumerate(score_names):
    mask = dominant == score_index
    if mask.any():
      axis.scatter(
        coordinates[mask, 0],
        coordinates[mask, 1],
        color=colors[emotion],
        s=24,
        linewidths=0,
        label=emotion.title(),
        zorder=3,
      )

  axis.set(
    title=f"Chapter ID {chapter_id}: paragraph-to-paragraph PCA vectors",
    xlabel=f"Score PC1 ({explained_variance[0]:.1%} variance)",
    ylabel=f"Score PC2 ({explained_variance[1]:.1%} variance)",
  )
  axis.grid(alpha=0.12)
  axis.set_aspect("equal", adjustable="datalim")
  handles, labels = axis.get_legend_handles_labels()
  handles.append(Line2D([0], [0], color=ARROW_COLOR, linewidth=2, alpha=0.35))
  labels.append("Paragraph transition")
  axis.legend(handles, labels, title="Dominant emotion", frameon=False, ncols=2)
  fig.tight_layout()
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def make_3d(
  paragraph_ids,
  coordinates,
  global_indices,
  score_names,
  values,
  texts,
  chapter_id,
  explained_variance,
):
  """Build an interactive chapter plot with point and vector details."""
  dominant = np.argmax(values, axis=1)
  colors = emotion_colors(score_names)
  figure = go.Figure()
  point_traces = []

  starts, vectors, transition_sources = chapter_vectors(coordinates, global_indices)
  if len(starts):
    ends = starts + vectors
    vector_text = []
    for source, vector in zip(transition_sources, vectors):
      vector_text.append(
        f"{paragraph_ids[source]} → {paragraph_ids[source + 1]}<br>"
        f"Global index {global_indices[source]} → {global_indices[source + 1]}<br>"
        f"Vector magnitude: {np.linalg.norm(vector):.3f}"
      )

    segment_count = len(starts)
    figure.add_trace(
      go.Scatter3d(
        x=np.column_stack((starts[:, 0], ends[:, 0], np.full(segment_count, np.nan))).ravel(),
        y=np.column_stack((starts[:, 1], ends[:, 1], np.full(segment_count, np.nan))).ravel(),
        z=np.column_stack((starts[:, 2], ends[:, 2], np.full(segment_count, np.nan))).ravel(),
        mode="lines",
        text=np.column_stack((vector_text, vector_text, [""] * segment_count)).ravel(),
        hoverinfo="text",
        name="Paragraph transition",
        line={"color": ARROW_COLOR, "width": 3},
        opacity=0.16,
        connectgaps=False,
      )
    )

    magnitudes = np.linalg.norm(vectors, axis=1)
    nonzero = magnitudes > 1e-12
    directions = vectors[nonzero] / magnitudes[nonzero, np.newaxis]
    helpers = np.tile(np.array([0.0, 0.0, 1.0]), (len(directions), 1))
    helpers[np.abs(directions[:, 2]) > 0.9] = np.array([0.0, 1.0, 0.0])
    perpendiculars = np.cross(directions, helpers)
    perpendiculars /= np.linalg.norm(perpendiculars, axis=1, keepdims=True)
    extent = np.linalg.norm(np.ptp(coordinates, axis=0))
    head_lengths = np.minimum(0.006 * extent, 0.30 * magnitudes[nonzero])
    head_widths = 0.42 * head_lengths
    tips = ends[nonzero]
    bases = tips - directions * head_lengths[:, np.newaxis]
    wing_1 = bases + perpendiculars * head_widths[:, np.newaxis]
    wing_2 = bases - perpendiculars * head_widths[:, np.newaxis]
    head_points = np.column_stack((tips, wing_1, wing_2)).reshape(-1, 3)
    triangle_starts = np.arange(0, len(head_points), 3)
    figure.add_trace(
      go.Mesh3d(
        x=head_points[:, 0],
        y=head_points[:, 1],
        z=head_points[:, 2],
        i=triangle_starts,
        j=triangle_starts + 1,
        k=triangle_starts + 2,
        color=ARROW_COLOR,
        opacity=1,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
      )
    )

  for score_index, emotion in enumerate(score_names):
    mask = dominant == score_index
    if not mask.any():
      continue
    customdata = []
    for paragraph_index in np.flatnonzero(mask):
      scores_html = "<br>".join(
        f"{name.title()}: {values[paragraph_index, column]:.2f}"
        for column, name in enumerate(score_names)
      )
      customdata.append(
        [
          paragraph_ids[paragraph_index],
          global_indices[paragraph_index],
          emotion.title(),
          scores_html,
          first_sentence(texts.get(paragraph_ids[paragraph_index], "")),
        ]
      )
    point_traces.append(
      go.Scatter3d(
        x=coordinates[mask, 0],
        y=coordinates[mask, 1],
        z=coordinates[mask, 2],
        mode="markers",
        name=emotion.title(),
        customdata=customdata,
        marker={"size": 6, "opacity": 1, "color": colors[emotion]},
        hovertemplate=(
          "<b>%{customdata[0]}</b> · global index %{customdata[1]}<br>"
          "Emotion: <b>%{customdata[2]}</b><br><br>"
          "%{customdata[3]}<br><br>Sentence:<br>%{customdata[4]}<extra></extra>"
        ),
      )
    )

  figure.add_traces(point_traces)
  figure.update_layout(
    title=f"Chapter ID {chapter_id}: paragraph-to-paragraph 3D PCA vectors",
    template="plotly_white",
    legend={"title": "Dominant emotion", "itemsizing": "constant"},
    margin={"l": 0, "r": 0, "t": 55, "b": 0},
    scene={
      "xaxis_title": f"Score PC1 ({explained_variance[0]:.1%})",
      "yaxis_title": f"Score PC2 ({explained_variance[1]:.1%})",
      "zaxis_title": f"Score PC3 ({explained_variance[2]:.1%})",
      "aspectmode": "data",
    },
    hovermode="closest",
    hoverdistance=20,
    hoverlabel={"align": "left"},
  )
  return figure


def main():
  parser = argparse.ArgumentParser(description="Generate 2D and 3D PCA score fields per chapter.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument("--scores-file", type=Path, help="Optional paragraph_scores.json path")
  parser.add_argument("--processed-file", type=Path, help="Optional processed.json path")
  parser.add_argument("--output-dir", type=Path, help="Optional chapter output root")
  args = parser.parse_args()

  book_dir = PROJECT_ROOT / "books" / args.book
  scores_file = (args.scores_file or book_dir / "paragraph_scores.json").expanduser().resolve()
  processed_file = (args.processed_file or book_dir / "processed.json").expanduser().resolve()
  output_root = (
    args.output_dir or PROJECT_ROOT / "output" / args.book / "scores" / "chapters"
  ).expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  texts = load_paragraph_texts(processed_file)
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  global_indices = paragraph_global_indices(paragraph_ids)
  if len(values) < 3 or not np.isfinite(values).all():
    raise ValueError("At least three paragraphs with complete finite scores are required")

  pca = PCA(n_components=3)
  all_coordinates = pca.fit_transform(values)
  output_root.mkdir(parents=True, exist_ok=True)

  for chapter_id in np.unique(chapter_ids):
    mask = chapter_ids == chapter_id
    chapter_dir = output_root / f"chapter_{chapter_id:02d}"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    chapter_coordinates = all_coordinates[mask]
    chapter_values = values[mask]
    chapter_paragraph_ids = [paragraph_id for paragraph_id, keep in zip(paragraph_ids, mask) if keep]
    chapter_global_indices = global_indices[mask]

    draw_2d(
      chapter_coordinates[:, :2],
      chapter_global_indices,
      score_names,
      chapter_values,
      chapter_id,
      pca.explained_variance_ratio_,
      chapter_dir / "pca_vector_field_2d.png",
    )
    figure = make_3d(
      chapter_paragraph_ids,
      chapter_coordinates,
      chapter_global_indices,
      score_names,
      chapter_values,
      texts,
      chapter_id,
      pca.explained_variance_ratio_,
    )
    figure.write_html(
      chapter_dir / "pca_vector_field_3d.html",
      include_plotlyjs=True,
      full_html=True,
    )
    print(f"Chapter ID {chapter_id}: {len(chapter_paragraph_ids)} paragraphs → {chapter_dir}")

  print(f"Done. Generated {len(np.unique(chapter_ids)) * 2} chapter visualizations in {output_root}")


if __name__ == "__main__":
  main()
