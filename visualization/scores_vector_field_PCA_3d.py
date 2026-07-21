"""Create an interactive 3D vector field from paragraph scores.

Example:

python visualization/scores_vector_field_3d.py --book alice_wonderland
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_GLOBAL_INDICES = frozenset(range(17, 20)) | frozenset(range(27, 30)) | frozenset(range(199, 202))


def load_scores(path):
  """Return paragraph IDs, score names, and a paragraph-by-score matrix."""
  with path.open("r", encoding="utf-8") as file:
    paragraphs = json.load(file)
  if not paragraphs or not paragraphs[0].get("scores"):
    raise ValueError(f"No paragraph scores found in {path}")

  paragraphs = [
    item for global_index, item in enumerate(paragraphs)
    if global_index not in EXCLUDED_GLOBAL_INDICES
  ]
  score_names = list(paragraphs[0]["scores"])
  paragraph_ids = [item.get("paragraph_id", str(index)) for index, item in enumerate(paragraphs)]
  values = np.asarray(
    [[item.get("scores", {}).get(name, np.nan) for name in score_names] for item in paragraphs],
    dtype=float,
  )
  return paragraph_ids, score_names, values


def paragraph_global_indices(paragraph_ids):
  """Extract numeric global indices from IDs such as raw_p0199."""
  return np.asarray([int(paragraph_id.rsplit("p", 1)[-1]) for paragraph_id in paragraph_ids])


def load_paragraph_texts(path):
  """Map paragraph IDs to their original text."""
  with path.open("r", encoding="utf-8") as file:
    processed = json.load(file)
  return {paragraph["id"]: paragraph.get("text", "") for paragraph in processed["paragraphs"]}


def load_chapter_ids(path, paragraph_ids):
  """Return chapter IDs aligned to the filtered paragraph IDs."""
  with path.open("r", encoding="utf-8") as file:
    processed = json.load(file)
  chapters_by_id = {paragraph["id"]: paragraph["chapter_id"] for paragraph in processed["paragraphs"]}
  return np.asarray([chapters_by_id[paragraph_id] for paragraph_id in paragraph_ids], dtype=int)


def first_sentence(text):
  """Return a compact sentence for the point tooltip."""
  normalized = " ".join(text.split())
  parts = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
  sentence = parts[0] if parts else normalized
  if len(sentence) > 420:
    sentence = sentence[:417].rstrip() + "…"
  return "<br>".join(textwrap.wrap(sentence, width=72))


def build_vector_field(coordinates, global_indices=None):
  """Return one exact vector from every paragraph to its successor."""
  starts = coordinates[:-1]
  changes = coordinates[1:] - coordinates[:-1]
  if global_indices is not None:
    consecutive = np.diff(global_indices) == 1
    starts = starts[consecutive]
    changes = changes[consecutive]
  return np.column_stack((starts, changes))


def make_figure(
  paragraph_ids,
  score_names,
  values,
  texts,
  coordinates,
  field,
  variance,
  chapter_ids=None,
  projection="PCA",
):
  """Build the interactive paragraph points and exact transition vectors."""
  dominant_indices = np.argmax(values, axis=1)
  emotions = np.array([score_names[index] for index in dominant_indices])
  palette = qualitative.Safe
  figure = go.Figure()
  point_traces = []
  point_size = 5.5 if projection == "UMAP" else 7
  shaft_width = 2.5 if projection == "UMAP" else 5
  head_scale = 0.009 if projection == "UMAP" else 0.022

  for emotion_index, emotion in enumerate(score_names):
    mask = emotions == emotion
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
          paragraph_index,
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
        marker={"size": point_size, "opacity": 1, "color": palette[emotion_index % len(palette)]},
        hovertemplate=(
          "<b>%{customdata[0]}</b> · paragraph %{customdata[1]}<br>"
          "Emotion: <b>%{customdata[2]}</b><br><br>"
          "%{customdata[3]}<br><br>"
          "Sentence:<br>%{customdata[4]}<extra></extra>"
        ),
      )
    )

  if len(field):
    magnitudes = np.linalg.norm(field[:, 3:6], axis=1)
    global_indices = paragraph_global_indices(paragraph_ids)
    transition_sources = np.flatnonzero(np.diff(global_indices) == 1)
    if chapter_ids is None:
      chapter_ids = np.zeros(len(paragraph_ids), dtype=int)
    transition_chapters = np.column_stack(
      (chapter_ids[transition_sources], chapter_ids[transition_sources + 1])
    )
    vector_text = [
      (
        f"{paragraph_ids[source]} → {paragraph_ids[source + 1]}<br>"
        f"Global index {global_indices[source]} → {global_indices[source + 1]}<br>"
        f"Vector magnitude: {magnitude:.3f}"
      )
      for source, magnitude in zip(transition_sources, magnitudes)
    ]

    starts = field[:, :3]
    ends = starts + field[:, 3:6]
    midpoints = (starts + ends) / 2
    half_starts = np.concatenate((starts, midpoints))
    half_ends = np.concatenate((midpoints, ends))
    half_chapters = np.concatenate((transition_chapters[:, 0], transition_chapters[:, 1]))
    half_text = np.concatenate((vector_text, vector_text))
    unique_chapters = np.unique(chapter_ids)
    chapter_palette = qualitative.Dark24
    chapter_colors = {
      chapter_id: chapter_palette[index % len(chapter_palette)]
      for index, chapter_id in enumerate(unique_chapters)
    }

    for chapter_id in unique_chapters:
      chapter_mask = half_chapters == chapter_id
      segment_count = int(chapter_mask.sum())
      if not segment_count:
        continue
      segment_starts = half_starts[chapter_mask]
      segment_ends = half_ends[chapter_mask]
      segment_text = half_text[chapter_mask]
      figure.add_trace(
        go.Scatter3d(
          x=np.column_stack((segment_starts[:, 0], segment_ends[:, 0], np.full(segment_count, np.nan))).ravel(),
          y=np.column_stack((segment_starts[:, 1], segment_ends[:, 1], np.full(segment_count, np.nan))).ravel(),
          z=np.column_stack((segment_starts[:, 2], segment_ends[:, 2], np.full(segment_count, np.nan))).ravel(),
          mode="lines",
          text=np.column_stack((segment_text, segment_text, [""] * segment_count)).ravel(),
          hoverinfo="text",
          name=f"Arrow: Chapter ID {chapter_id}",
          line={"color": chapter_colors[chapter_id], "width": shaft_width},
          connectgaps=False,
          legendgroup=f"arrow-chapter-{chapter_id}",
        )
      )

    nonzero = magnitudes > 1e-12
    plot_extent = np.linalg.norm(np.ptp(coordinates, axis=0))
    directions = field[nonzero, 3:6] / magnitudes[nonzero, np.newaxis]
    helpers = np.tile(np.array([0.0, 0.0, 1.0]), (len(directions), 1))
    nearly_parallel = np.abs(directions[:, 2]) > 0.9
    helpers[nearly_parallel] = np.array([0.0, 1.0, 0.0])
    perpendiculars = np.cross(directions, helpers)
    perpendiculars /= np.linalg.norm(perpendiculars, axis=1, keepdims=True)

    head_lengths = np.minimum(head_scale * plot_extent, 0.35 * magnitudes[nonzero])
    head_widths = 0.45 * head_lengths
    tips = ends[nonzero]
    head_bases = tips - directions * head_lengths[:, np.newaxis]
    wing_1 = head_bases + perpendiculars * head_widths[:, np.newaxis]
    wing_2 = head_bases - perpendiculars * head_widths[:, np.newaxis]

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
        hoverinfo="skip",
        name="Arrowheads",
        facecolor=[chapter_colors[chapter_id] for chapter_id in transition_chapters[nonzero, 1]],
        opacity=1,
        flatshading=True,
        showlegend=False,
      )
    )

  # Add points last so arrow geometry cannot intercept their hover targets.
  figure.add_traces(point_traces)

  if projection == "PCA":
    axis_titles = [
      f"Score PC1 ({variance[0]:.1%})",
      f"Score PC2 ({variance[1]:.1%})",
      f"Score PC3 ({variance[2]:.1%})",
    ]
  else:
    axis_titles = [f"Score {projection}-{dimension}" for dimension in range(1, 4)]

  figure.update_layout(
    title=f"Paragraph-to-paragraph 3D narrative vectors ({projection})",
    template="plotly_white",
    legend={"title": "Points: emotion · Arrows: chapter ID", "itemsizing": "constant"},
    margin={"l": 0, "r": 0, "t": 55, "b": 0},
    scene={
      "xaxis_title": axis_titles[0],
      "yaxis_title": axis_titles[1],
      "zaxis_title": axis_titles[2],
      "aspectmode": "data",
      "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
    },
    hoverlabel={"align": "left"},
    hovermode="closest",
    hoverdistance=20,
  )
  return figure


def main():
  parser = argparse.ArgumentParser(description="Create an interactive 3D score vector field.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument("--scores-file", type=Path, help="Optional paragraph_scores.json path")
  parser.add_argument("--processed-file", type=Path, help="Optional processed.json path")
  parser.add_argument("--output-dir", type=Path, help="Optional output directory")
  args = parser.parse_args()

  book_dir = PROJECT_ROOT / "books" / args.book
  scores_file = (args.scores_file or book_dir / "paragraph_scores.json").expanduser().resolve()
  processed_file = (args.processed_file or book_dir / "processed.json").expanduser().resolve()
  output_dir = (args.output_dir or PROJECT_ROOT / "output" / args.book / "scores").expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  texts = load_paragraph_texts(processed_file)
  chapter_ids = load_chapter_ids(processed_file, paragraph_ids)
  if len(values) < 2:
    raise ValueError("At least two paragraphs are required")
  if not np.isfinite(values).all():
    raise ValueError("Every paragraph must contain a finite value for every score")

  pca = PCA(n_components=3)
  coordinates = pca.fit_transform(values)
  field = build_vector_field(coordinates, paragraph_global_indices(paragraph_ids))
  figure = make_figure(
    paragraph_ids,
    score_names,
    values,
    texts,
    coordinates,
    field,
    pca.explained_variance_ratio_,
    chapter_ids,
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "score_vector_field_3d.html"
  figure.write_html(output_path, include_plotlyjs=True, full_html=True)

  print(f"Loaded {len(paragraph_ids)} paragraphs with {len(score_names)} scores.")
  print(f"Drew {len(field)} exact paragraph-to-paragraph vectors.")
  print(output_path)


if __name__ == "__main__":
  main()
