from sentence_transformers import SentenceTransformer

from models.base import EmbeddingModel


class E5LargeV2Embedding(EmbeddingModel):

  MODEL_NAME = "intfloat/e5-large-v2"

  def __init__(self, device):
    self.model = SentenceTransformer(
      self.MODEL_NAME,
      device=device,
    )

  @property
  def name(self):
    return "e5-large-v2"

  @property
  def hf_model(self):
    return self.MODEL_NAME

  def encode(self, texts, batch_size=16):
    # E5 expects every input to have a prefix.
    texts = ["query: " + text for text in texts]
    emb = self.model.encode(
      texts,
      batch_size=batch_size,
      normalize_embeddings=True,
      convert_to_numpy=True,
      show_progress_bar=True,
    )
    return emb