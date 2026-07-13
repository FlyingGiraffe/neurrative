#!/usr/bin/env python3

"""
Preprocess a Project Gutenberg book into a structured JSON file.

Output format:

{
  "metadata": {...},
  "chapters": [...],
  "paragraphs": [...]
}
"""

import argparse
import json
import os
import re


HEADER_RE = r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
FOOTER_RE = r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*"


def remove_gutenberg_header_footer(text):
  """
  Remove the Project Gutenberg header and footer.
  This implementation is robust to different Gutenberg editions.
  """
  lines = text.splitlines()
  start = 0
  end = len(lines)

  for i, line in enumerate(lines):
    if "*** START OF" in line.upper():
      start = i + 1
      break

  for i, line in enumerate(lines):
    if "*** END OF" in line.upper() or line == "THE END":
      end = i
      break

  return "\n".join(lines[start:end]).strip()


def split_chapters(text):
  """
  Split chapters.

  Supports formats such as:
    CHAPTER I
    CHAPTER II
    CHAPTER 1

  Returns:
    list[(title, body)]
  """

  pattern = re.compile(
    r"\n\s*CHAPTER\s+([A-Z0-9IVXLCDM]+)\.?\s*\n",
    flags=re.IGNORECASE,
  )

  matches = list(pattern.finditer(text))

  if len(matches) == 0:
    return [("Book", text)]

  chapters = []

  for i, m in enumerate(matches):

    title = m.group(1)
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    body = text[start:end].strip()
    lines = body.splitlines()

    chapter_name = ""

    while lines and lines[0].strip() == "":
      lines.pop(0)

    if lines:
      chapter_name = lines[0].strip()
      body = "\n".join(lines[1:]).strip()

    chapters.append((title, chapter_name, body))

  return chapters


def split_paragraphs(text):
  """
  Split paragraphs using blank lines.
  """
  paras = re.split(r"\n\s*\n", text)
  paras = [
    " ".join(p.split())
    for p in paras
  ]
  paras = [p for p in paras if len(p) > 0]
  return paras


def preprocess_book(book_path, output_path):

  with open(book_path, "r", encoding="utf-8") as f:
    text = f.read()

  text = remove_gutenberg_header_footer(text)
  chapters = split_chapters(text)

  book = {
    "metadata": {
      "title": os.path.splitext(os.path.basename(book_path))[0],
      "source": "Project Gutenberg",
    },
    "chapters": [],
    "paragraphs": [],
  }

  global_idx = 0
  book_id = os.path.splitext(os.path.basename(book_path))[0].lower()

  for chapter_idx, (chapter_number, chapter_title, body) in enumerate(chapters):
    paragraphs = split_paragraphs(body)

    start_idx = global_idx
    for local_idx, para in enumerate(paragraphs):
      pid = f"{book_id}_p{global_idx:04d}"
      book["paragraphs"].append(
        {
          "id": pid,
          "global_index": global_idx,
          "chapter_id": chapter_idx,
          "paragraph_index": local_idx,
          "text": para,
        }
      )
      global_idx += 1

    end_idx = global_idx - 1

    book["chapters"].append(
      {
        "chapter_id": chapter_idx,
        "chapter_number": chapter_number,
        "title": chapter_title,
        "paragraph_range": [start_idx, end_idx],
      }
    )

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(book, f, indent=2, ensure_ascii=False)

  print(f"Parsed {len(book['chapters'])} chapters")
  print(f"Parsed {len(book['paragraphs'])} paragraphs")
  print(f"Saved to {output_path}")


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("book")
  args = parser.parse_args()

  book_path = os.path.join("books", args.book, "raw.txt")
  output_path = os.path.join("books", args.book, "processed.json")

  preprocess_book(book_path, output_path)