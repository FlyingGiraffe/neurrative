import os
import argparse
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_score_dimensions(book):
  filename = os.path.join("books", book, "metadata", "score_dimensions.txt")

  with open(filename) as f:
    return [
      line.strip()
      for line in f
      if line.strip() and not line.startswith("#")
    ]

def build_system_prompt(score_dimensions):
  score_lines = "\n".join(
    f'    "{name}": float,'
    for name in score_dimensions
  )

  return (
f"""You are a literary analysis assistant.

For the given paragraph, output ONLY valid JSON.

The JSON format is

{{
  "scores": {{
{score_lines}
  }},
  "summary": "...",
  "keywords": ["...", "...", "..."],
  "characters": ["...", "..."]
}}

All scores must be between 0 and 1.

Each score should represent how strongly that aspect is expressed in this paragraph.

Do not output Markdown.
Do not explain anything.
"""
  )


def build_prompt(paragraph):
  return f"Paragraph: {paragraph}"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True)
  parser.add_argument("--model",default="Qwen/Qwen2.5-7B-Instruct")
  parser.add_argument("--device", default="cuda")
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
  )

  model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
  )

  processed_file = os.path.join("books", args.book, "processed.json")
  with open(processed_file) as f:
    processed = json.load(f)
  paragraphs = processed["paragraphs"]

  score_dimensions = load_score_dimensions(args.book)
  system_prompt = build_system_prompt(score_dimensions)

  results = []
  for paragraph in tqdm(paragraphs):
    messages = [
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": build_prompt(paragraph["text"]),
      },
    ]

    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response = tokenizer.decode(
      outputs[0][inputs.input_ids.shape[1]:],
      skip_special_tokens=True,
    )

    try:
      annotation = json.loads(response)
    except Exception:
      annotation = {"parse_error": response}
    annotation["paragraph_id"] = paragraph["id"]

    results.append(annotation)

  output_dir = os.path.join("books", args.book)
  os.makedirs(output_dir, exist_ok=True)

  with open(os.path.join(output_dir, "paragraph_scores.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
  main()