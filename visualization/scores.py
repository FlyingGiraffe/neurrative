"""Visualize paragraph-level narrative scores for a book.

Example:

python visualization/scores.py --book alice_wonderland
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_GLOBAL_INDICES = frozenset(range(17, 20)) | frozenset(range(27, 30)) | frozenset(range(199, 202))


def load_scores(path):
  """Return paragraph IDs, score names, and a paragraph-by-score matrix."""
  with path.open("r", encoding="utf-8") as file:
    paragraphs = json.load(file)

  if not paragraphs:
    raise ValueError(f"No paragraph scores found in {path}")

  paragraphs = [
    item for global_index, item in enumerate(paragraphs)
    if global_index not in EXCLUDED_GLOBAL_INDICES
  ]
  score_names = list(paragraphs[0].get("scores", {}))
  if not score_names:
    raise ValueError(f"The first paragraph has no scores in {path}")

  paragraph_ids = [item.get("paragraph_id", str(index)) for index, item in enumerate(paragraphs)]
  values = np.array(
    [[item.get("scores", {}).get(name, np.nan) for name in score_names] for item in paragraphs],
    dtype=float,
  )
  return paragraph_ids, score_names, values


def paragraph_global_indices(paragraph_ids):
  """Extract numeric global indices from IDs such as raw_p0199."""
  return np.asarray([int(paragraph_id.rsplit("p", 1)[-1]) for paragraph_id in paragraph_ids])


def load_chapter_ids(path, paragraph_ids):
  """Return chapter IDs aligned to the filtered paragraph IDs."""
  with path.open("r", encoding="utf-8") as file:
    processed = json.load(file)
  chapters_by_id = {paragraph["id"]: paragraph["chapter_id"] for paragraph in processed["paragraphs"]}
  return np.asarray([chapters_by_id[paragraph_id] for paragraph_id in paragraph_ids], dtype=int)


def plot_score_trajectory(score_names, values, output_path):
  """Plot every narrative score across paragraph order."""
  paragraph_numbers = np.arange(len(values))
  fig, axis = plt.subplots(figsize=(14, 7))

  for column, name in enumerate(score_names):
    axis.plot(paragraph_numbers, values[:, column], linewidth=1.2, alpha=0.9, label=name.title())

  axis.set(
    title="Narrative scores across paragraphs",
    xlabel="Paragraph order",
    ylabel="Score",
    xlim=(0, max(len(values) - 1, 1)),
    ylim=(0, 1),
  )
  axis.grid(alpha=0.2)
  axis.legend(ncols=min(3, len(score_names)), frameon=False)
  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)


def plot_score_heatmap(score_names, values, output_path):
  """Plot a compact overview of all scores and paragraphs."""
  fig_width = max(12, min(24, len(values) / 12))
  fig, axis = plt.subplots(figsize=(fig_width, 5))
  image = axis.imshow(values.T, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="magma")

  axis.set(
    title="Paragraph score heatmap",
    xlabel="Paragraph order",
    ylabel="Score",
  )
  axis.set_yticks(np.arange(len(score_names)), labels=[name.title() for name in score_names])
  colorbar = fig.colorbar(image, ax=axis, pad=0.015)
  colorbar.set_label("Score")
  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Visualize paragraph-level narrative scores.")
  parser.add_argument("--book", default="alice_wonderland", help="Book folder inside books/")
  parser.add_argument(
    "--scores-file",
    type=Path,
    help="Optional path to paragraph_scores.json (overrides --book)",
  )
  parser.add_argument("--output-dir", type=Path, help="Optional directory for generated PNG files")
  args = parser.parse_args()

  scores_file = args.scores_file or PROJECT_ROOT / "books" / args.book / "paragraph_scores.json"
  output_dir = args.output_dir or PROJECT_ROOT / "output" / args.book / "scores"
  scores_file = scores_file.expanduser().resolve()
  output_dir = output_dir.expanduser().resolve()

  paragraph_ids, score_names, values = load_scores(scores_file)
  output_dir.mkdir(parents=True, exist_ok=True)

  trajectory_path = output_dir / "score_trajectory.png"
  heatmap_path = output_dir / "score_heatmap.png"
  plot_score_trajectory(score_names, values, trajectory_path)
  plot_score_heatmap(score_names, values, heatmap_path)

  print(f"Loaded {len(paragraph_ids)} paragraphs with {len(score_names)} scores.")
  print(trajectory_path)
  print(heatmap_path)


if __name__ == "__main__":
  main()
