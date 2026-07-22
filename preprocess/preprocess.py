"""
Robust Project Gutenberg preprocessor.

This parser is designed to work across a wide range of Gutenberg books,
including:

- Pride and Prejudice
- Alice's Adventures in Wonderland
- A Christmas Carol
- The Call of the Wild
- Sherlock Holmes
- Treasure Island
- Dracula
- Jane Eyre
- etc.

Features
--------
* Removes Gutenberg header/footer
* Removes front matter
* Removes Contents pages
* Removes illustrations / footnotes / page numbers
* Robust chapter detection
* Robust paragraph extraction
* Keeps chapter metadata
"""

import argparse
import json
import os
import re


###############################################################################
# Configuration
###############################################################################


CHAPTER_KEYWORDS = ["CHAPTER", "BOOK", "PART", "STAVE", "LETTER", "VOLUME"]
CHAPTER_RE = re.compile(
  rf"^\s*({'|'.join(CHAPTER_KEYWORDS)})\s+([A-Za-z0-9IVXLCDM-]+)\b(.*)$",
  re.IGNORECASE,
)
HEADER_RE = re.compile(r"\*\*\*\s*START OF .*PROJECT GUTENBERG", re.IGNORECASE)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF .*PROJECT GUTENBERG", re.IGNORECASE)


###############################################################################
# Utility
###############################################################################


def normalize_whitespace(text):
  """
  Collapse repeated whitespace while preserving paragraph breaks.
  """
  text = text.replace("\r\n", "\n")
  text = text.replace("\r", "\n")
  return text


def clean_inline_markup(text):
  """
  Remove inline Gutenberg markup while preserving surrounding text.
  """
  substitutions = [
    # illustrations
    (r"\[Illustration:.*?\]", ""),
    (r"\[Illustration\]", ""),
    (r"\[Frontispiece.*?\]", ""),
    # editorial notes
    (r"\[Transcriber's Note:.*?\]", ""),
    (r"\[Editor's Note:.*?\]", ""),
    (r"\[Footnote:.*?\]", ""),
    # page markers
    (r"\[Page\s+\d+\]", ""),
    # copyright notes
    (r"\[_?Copyright.*?\]", ""),
  ]

  for pattern, repl in substitutions:
    text = re.sub(pattern, repl, text, flags=re.IGNORECASE | re.DOTALL)

  return text


###############################################################################
# Gutenberg cleanup
###############################################################################


def remove_gutenberg_header_footer(text):
  lines = text.splitlines()
  start = 0
  end = len(lines)

  for i, line in enumerate(lines):
    if HEADER_RE.search(line):
      start = i + 1
      break

  for i in range(len(lines) - 1, -1, -1):
    if FOOTER_RE.search(lines[i]):
      end = i
      break
    if lines[i].strip().upper() == "THE END":
      end = i
      break

  return "\n".join(lines[start:end])


def clean_text(text):
  """
  Remove common Project Gutenberg artifacts.
  """
  patterns = [
    r"\[Illustration:.*?\]",
    r"\[Illustration\]",
    r"\[Frontispiece.*?\]",
    r"\[Ornament.*?\]",
    r"\[Decorative.*?\]",
    r"\[Footnote:.*?\]",
    r"\[Transcriber's Note:.*?\]",
    r"\[Editor's Note:.*?\]",
  ]
  for pattern in patterns:
    text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
  return text


###############################################################################
# Paragraph filtering
###############################################################################


def should_skip_paragraph(text):
  t = text.strip()
  if not t:
    return True
  lower = t.lower()

  patterns = [
    r"^\[illustration.*\]$",
    r"^\[frontispiece.*\]$",
    r"^\[ornament.*\]$",
    r"^\[decorative.*\]$",
    r"^\[transcriber's note.*\]$",
    r"^\[editor.?s note.*\]$",
    r"^\[footnote.*\]$",
    r"^\[_?copyright.*\]$",
    r"^\[page\s+\d+\]$",
  ]

  for p in patterns:
    if re.match(p, lower):
      return True

  if re.fullmatch(r"[*=_\-~. ]{5,}", t):
    return True

  return False


###############################################################################
# Chapter helpers
###############################################################################

def is_chapter_heading(line):
  return CHAPTER_RE.match(line.strip()) is not None


def looks_like_chapter_title(line):
  line = line.strip()
  if (
    (not line) or
    (len(line) > 80) or
    (len(line.split()) > 10) or
    (line.endswith((".", "!", "?")))
  ):
    return False
  return True


###############################################################################
# Front matter
###############################################################################


def remove_front_matter(text):
  """
  Find the first genuine chapter heading.
  We intentionally skip title pages, prefaces, contents pages, etc.
  """

  lines = text.splitlines()
  first_heading = None

  for i, line in enumerate(lines):
    if not is_chapter_heading(line):
      continue
    first_heading = i
    break

  if first_heading is None:
    return text

  return "\n".join(lines[first_heading:])


###############################################################################
# Paragraph splitting
###############################################################################


def split_paragraphs(text):
  paragraphs = re.split(r"\n\s*\n", text)

  cleaned = []
  for p in paragraphs:
    p = normalize_whitespace(p)
    p = " ".join(p.split())

    if should_skip_paragraph(p):
        continue
    if not p:
        continue
    
    cleaned.append(p)

  return cleaned


###############################################################################
# Chapter detection
###############################################################################

def parse_heading(line):
  """
  Parse a chapter heading.

  Returns:
    (keyword, number, inline_title) or None.
  """
  m = CHAPTER_RE.match(line.strip())

  if m is None:
    return None

  keyword = m.group(1).upper()
  number = m.group(2).strip()

  title = m.group(3).strip()
  title = re.sub(r"^[\s:.\-–—]+", "", title)

  return keyword, number, title


def looks_like_contents_page(lines, idx):
  """
  Determine whether a heading belongs to the table of contents.

  We inspect the next ~20 lines. If several additional chapter
  headings immediately follow, we are almost certainly inside
  a TOC.
  """
  count = 0
  for j in range(idx + 1, min(len(lines), idx + 20)):
    if parse_heading(lines[j]):
      count += 1

  return count >= 2


def extract_multiline_title(body_lines):
  """
  Extract chapter subtitle.

  Examples:
    CHAPTER I
    The White Whale
  or
    STAVE I
    MARLEY'S GHOST
  """
  title = ""

  while body_lines and not body_lines[0].strip():
    body_lines.pop(0)

  if not body_lines:
    return title, body_lines

  first = body_lines[0].strip()
  if looks_like_chapter_title(first):
    title = first
    body_lines = body_lines[1:]

  while body_lines and not body_lines[0].strip():
    body_lines.pop(0)

  return title, body_lines


def find_chapter_headers(lines):
  headers = []
  for i, line in enumerate(lines):
    heading = parse_heading(line)
    if (
      (heading is None) or
      (line.strip() != line.strip()) or  # Heading must occupy its own line.
      (i > 0 and lines[i - 1].strip())  # Previous line should usually be blank.
    ):
      continue

    keyword, number, title = heading
    headers.append((i, keyword, number, title))

  if not headers:
    return []

  # Remove duplicated TOC headings.
  final = []
  i = 0
  while i < len(headers):
    current = headers[i]
    number = current[2]
    # If we later see Chapter I again, then everything before that was a TOC.
    if (number == "I" and i > 0):
      remaining = [h[2] for h in headers[i:i+5]]
      if (len(remaining) >= 2 and remaining[0] == "I" and remaining[1] == "II"):
        final = headers[i:]
        break
    final.append(current)
    i += 1

  return final


###############################################################################
# Chapter splitting
###############################################################################

def split_chapters(text):
  """
  Split a Gutenberg novel into chapters.

  Returns:
    list(chapter_number, chapter_title, chapter_body)
  """
  lines = text.splitlines()
  headers = find_chapter_headers(lines)

  # No chapters.
  if len(headers) == 0:
    return [("1", "", text.strip())]

  chapters = []

  for idx, (line_idx, keyword, number, inline_title) in enumerate(headers):

    start = line_idx + 1

    if idx + 1 < len(headers):
      end = headers[idx + 1][0]
    else:
      end = len(lines)

    body_lines = lines[start:end]

    title = inline_title
    # Multi-line title.
    if not title:
      title, body_lines = extract_multiline_title(body_lines)

    # Trim blank lines.
    while body_lines and not body_lines[0].strip():
      body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
      body_lines.pop()

    body = "\n".join(body_lines)
    chapters.append((number, title, body))

  return chapters


###############################################################################
# Front matter removal
###############################################################################

def remove_front_matter(text):
  """
  Remove title pages, publication information, prefaces, contents pages, etc.

  We don't simply start at the first chapter heading, because many books have
  a Contents page containing every chapter heading. Instead we look for the
  first heading whose body actually contains real paragraphs.
  """
  lines = text.splitlines()
  candidates = []

  # Find every possible chapter heading.
  for i, line in enumerate(lines):
    heading = parse_heading(line)
    if heading is None:
      continue
    candidates.append(i)

  if not candidates:
    return text

  # Evaluate each candidate.
  for idx, start in enumerate(candidates):
    # Where does the next heading begin?
    if idx + 1 < len(candidates):
      end = candidates[idx + 1]
    else:
      end = len(lines)

    body = "\n".join(lines[start:end])
    paragraphs = split_paragraphs(body)

    # A real chapter almost always has several paragraphs.
    if len(paragraphs) >= 3:
      return "\n".join(lines[start:])

  # Fallback
  return text


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True)
  args = parser.parse_args()

  book_path = os.path.join("books", "standard", args.book, "raw.txt")
  output_path = os.path.join("books", "standard", args.book, "processed.json")

  with open(book_path, "r", encoding="utf-8") as f:
    text = f.read()

  # Clean raw Gutenberg text.
  text = remove_gutenberg_header_footer(text)
  text = clean_text(text)
  text = remove_front_matter(text)

  chapters = split_chapters(text)

  book = {
    "metadata": {
      "title": args.book,
      "source": "Project Gutenberg",
    },
    "chapters": [],
    "paragraphs": [],
  }

  global_index = 0
  chapter_id = 0

  for chapter_number, chapter_title, body in chapters:

    paragraphs = split_paragraphs(body)

    # Skip empty chapters (can happen from duplicated TOCs)
    if len(paragraphs) == 0:
      continue

    start_index = global_index
    for paragraph_index, para in enumerate(paragraphs):
      book["paragraphs"].append(
        {
          "id": f"raw_p{global_index:04d}",
          "global_index": global_index,
          "chapter_id": chapter_id,
          "paragraph_index": paragraph_index,
          "text": para,
        }
      )
      global_index += 1

    end_index = global_index - 1

    book["chapters"].append(
      {
        "chapter_id": chapter_id,
        "chapter_number": chapter_number,
        "title": chapter_title,
        "paragraph_range": [start_index, end_index],
      }
    )
    chapter_id += 1

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(book, f, indent=2, ensure_ascii=False)

  print(f"Parsed {len(book['chapters'])} chapters")
  print(f"Parsed {len(book['paragraphs'])} paragraphs")
  print(f"Saved to {output_path}")


if __name__ == "__main__":
  main()