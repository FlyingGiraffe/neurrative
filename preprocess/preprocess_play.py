# Robust Project Gutenberg play preprocessor.
# Supports Shakespeare (_Ham._), Wilde/Ibsen (NORA., MRS. MARCHMONT.)
# and ACT/FIRST ACT formats.

import argparse, json, os, re

START_RE = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG", re.I)
ACT_RE = re.compile(
  r"^(?:ACT\s+([IVXLCDM]+)|"
  r"(FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+ACT)\.?$", re.I
)

ORD = {"FIRST":"I", "SECOND":"II", "THIRD":"III", "FOURTH":"IV", "FIFTH":"V"}

SHAKESPEARE = re.compile(r"^_([^_]+)\._\s*(.*)$")
PLAIN = re.compile(r"^([A-Z][A-Z .,'’\-]+?)\.\s*(.*)$")

EDITORIAL = [
  re.compile(r"^\[illustration.*\]$", re.I),
  re.compile(r"^\[footnote:.*\]$", re.I),
  re.compile(r"^\[transcriber's note:.*\]$", re.I),
  re.compile(r"^\[editor.?s note:.*\]$", re.I),
]

def strip(lines):
  s, e = 0, len(lines)

  for i, l in enumerate(lines):
    if START_RE.search(l):
      s = i + 1
      break
  
  for i in range(len(lines)-1, -1, -1):
    if END_RE.search(lines[i]):
      e = i
      break
  
  return lines[s:e]


def norm(x):
  return " ".join(x.strip().split())


def parse_speaker(line):
  m = SHAKESPEARE.match(line)

  if m:
    return m.group(1).strip(), m.group(2).strip()
  
  m = PLAIN.match(line)
  if m:
    name = m.group(1).strip()
    if len(name.split()) > 8:
      return None
    return name.title(), m.group(2).strip()
  
  return None


def stage(line):
  l = line.lower()

  if (line.startswith("_") and line.endswith("_")) or (line.startswith("[") and line.endswith("]")):
    return True
  
  for w in ["enter", "exit", "exeunt", "aside", "scene", "curtain", "flourish", "alarum"]:
    if w in l:
      return True
  
  return False


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True)
  args = parser.parse_args()

  inp = os.path.join("books", "play", args.book, "raw.txt")
  out = os.path.join("books", "play", args.book, "processed.json")
  
  lines = strip(open(inp, encoding="utf8").readlines())

  chapters = []
  paragraphs = []

  pending = None
  current = -1
  pidx = 0
  gidx = 0
  speech = []
  seen = False

  def flush():
    nonlocal speech, pidx, gidx

    if current < 0 or not speech:
      speech=[]
      return
    
    txt=" ".join(speech).strip()
    if txt:
      paragraphs.append({
        "id": f"raw_p{gidx:04d}",
        "global_index": gidx,
        "chapter_id": current,
        "paragraph_index": pidx,
        "text": txt}
      )
      gidx += 1
      pidx += 1
    speech=[]

  def start(act):
    nonlocal current, pidx
    if chapters:
      chapters[-1]["paragraph_range"][1] = gidx-1
    current += 1
    pidx = 0
    chapters.append({
      "chapter_id": current,
      "chapter_number": act,
      "title": f"ACT {act}",
      "paragraph_range": [gidx, None]}
    )

  for raw in lines:
    line = norm(raw)
    if (
      (not line) or
      (any(r.match(line) for r in EDITORIAL)) or
      (re.fullmatch(r"[\-*_=~. ]+", line)) or
      (line.lower() in {"contents", "the persons of the play", "dramatis personae", "scene"})
    ):
      continue

    m = ACT_RE.match(line)
    if m:
      flush()
      pending = m.group(1) or ORD[m.group(2).upper()]
      continue

    sp = parse_speaker(line)
    if sp:
      if pending is not None:
        start(pending)
        pending = None

      if current < 0:
        continue

      seen=True
      flush()
      n, first = sp
      speech = [f"{n}: {first}" if first else f"{n}:"]
      continue

    if (not seen) or (stage(line)): continue

    line=re.sub(r"_\[[^\]]*\]_\.?", "", line).strip()
    if line: 
      speech.append(line)

  flush()
  if chapters:
    chapters[-1]["paragraph_range"][1] = gidx - 1
  
  json.dump(
    {
      "metadata": {"title": args.book, "source": "Project Gutenberg"},
      "chapters": chapters,
      "paragraphs": paragraphs,
    },
    open(out, "w", encoding="utf8"), indent=2, ensure_ascii=False)
  
  print(len(chapters), "chapters")
  print(len(paragraphs), "paragraphs")

if __name__=="__main__":
  main()
