import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from models.base import EmbeddingModel


class BGEM3Embedding(EmbeddingModel):

  MODEL_NAME = "BAAI/bge-m3"

  def __init__(self, device):
    self.device = device

    self.tokenizer = AutoTokenizer.from_pretrained(
      self.MODEL_NAME, trust_remote_code=True,
    )
    self.model = AutoModel.from_pretrained(
      self.MODEL_NAME, trust_remote_code=True,
    ).to(device)

    self.model.eval()
  
  @property
  def name(self):
    return "bge-m3"

  @property
  def hf_model(self):
    return "BAAI/bge-m3"

  @torch.no_grad()
  def encode(self, texts, batch_size=16):

    outputs = []
    for i in tqdm(range(0, len(texts), batch_size)):
      batch = texts[i:i + batch_size]
      inputs = self.tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
      )
      inputs = {k: v.to(self.device) for k, v in inputs.items()}

      out = self.model(**inputs)
      emb = out.last_hidden_state[:, 0]
      emb = torch.nn.functional.normalize(emb, dim=-1)
      outputs.append(emb.cpu().numpy())

    return np.concatenate(outputs, axis=0)
