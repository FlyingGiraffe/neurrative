import os
import argparse
import json
import re


START_RE = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG")
END_RE = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG")

ACT_RE = re.compile(r"^\s*ACT\s+([IVXLC]+)", re.IGNORECASE)

# Example:
#   _Hor._ Tush! tush! 'twill not appear.
#   _Ber._ Come, let us...
SPEAKER_RE = re.compile(r"^\s*_([^_]+)\._\s*(.*)$")


def strip_gutenberg(lines):
  start = 0
  end = len(lines)

  for i, line in enumerate(lines):
    if START_RE.search(line):
      start = i + 1
      break

  for i in range(len(lines) - 1, -1, -1):
    if END_RE.search(lines[i]):
      end = i
      break

  return lines[start:end]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True)
  args = parser.parse_args()

  book_path = os.path.join("books", args.book, "raw.txt")
  output_path = os.path.join("books", args.book, "processed.json")

  with open(book_path, encoding="utf8") as f:
    lines = strip_gutenberg(f.readlines())

  chapters = []
  paragraphs = []

  current_act = None

  speech = []

  chapter_id = -1
  paragraph_index = 0
  global_index = 0

  def flush():
    nonlocal speech, paragraph_index, global_index

    if current_act is None:
      speech = []
      return

    text = " ".join(speech).strip()

    if len(text) == 0:
      speech = []
      return

    paragraphs.append(
      {
        "id": f"raw_p{global_index:04d}",
        "global_index": global_index,
        "chapter_id": chapter_id,
        "paragraph_index": paragraph_index,
        "text": text,
      }
    )
    paragraph_index += 1
    global_index += 1
    speech = []

  for raw in lines:
    line = " ".join(raw.strip().split())

    if len(line) == 0:
      continue

    m = ACT_RE.match(line)
    if m:
      flush()
      # Finish previous chapter
      if len(chapters) > 0:
        chapters[-1]["paragraph_range"][1] = global_index - 1
      chapter_id += 1
      paragraph_index = 0
      current_act = f"ACT {m.group(1)}"

      chapters.append(
        {
          "chapter_id": chapter_id,
          "chapter_number": m.group(1),
          "title": current_act,
          "paragraph_range": [global_index, None],
        }
      )
      continue

    # Ignore everything before ACT I
    if current_act is None:
      continue

    m = SPEAKER_RE.match(line)

    if m:
      flush()
      speaker = m.group(1).strip()
      first_line = m.group(2).strip()

      # Skip stage directions
      if any(word in speaker.lower() for word in ["enter", "exit", "exeunt", "re-enter"]):
        speech = []
        continue

      speech = []

      if len(first_line):
        speech.append(f"{speaker}: {first_line}")
      else:
        speech.append(f"{speaker}:")

      continue

    # Ignore standalone stage directions
    if line.startswith("_") and line.endswith("_"):
      continue

    speech.append(line)

  flush()

  # Finish final chapter
  if len(chapters) > 0:
    chapters[-1]["paragraph_range"][1] = global_index - 1

  output = {
    "metadata": {
      "title": args.book,
      "source": "Project Gutenberg",
    },
    "chapters": chapters,
    "paragraphs": paragraphs,
  }

  with open(output_path, "w", encoding="utf8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

  print(f"{len(chapters)} chapters")
  print(f"{len(paragraphs)} paragraphs")


if __name__ == "__main__":
  main()