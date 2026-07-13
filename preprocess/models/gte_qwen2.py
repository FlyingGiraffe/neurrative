from sentence_transformers import SentenceTransformer

from models.base import EmbeddingModel


class GTEQwen2Embedding(EmbeddingModel):

  MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"

  def __init__(self, device):
    self.model = SentenceTransformer(
      self.MODEL_NAME,
      device=device,
      trust_remote_code=True,
    )

  @property
  def name(self):
    return "gte-qwen2"

  @property
  def hf_model(self):
    return self.MODEL_NAME

  def encode(self, texts, batch_size=8):
    emb = self.model.encode(
      texts,
      batch_size=batch_size,
      normalize_embeddings=True,
      convert_to_numpy=True,
      show_progress_bar=True,
    )
    return emb