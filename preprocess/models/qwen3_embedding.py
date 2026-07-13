import numpy as np
from sentence_transformers import SentenceTransformer

from models.base import EmbeddingModel


class Qwen3Embedding(EmbeddingModel):

  MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

  def __init__(self, device):
    self.model = SentenceTransformer(
      self.MODEL_NAME,
      device=device,
      trust_remote_code=True,
    )

  @property
  def name(self):
    return "qwen3-embedding"

  @property
  def hf_model(self):
    return "Qwen/Qwen3-Embedding-4B"

  def encode(self, texts, batch_size=8):
    emb = self.model.encode(
      texts,
      batch_size=batch_size,
      normalize_embeddings=True,
      convert_to_numpy=True,
      show_progress_bar=True,
    )
    return emb