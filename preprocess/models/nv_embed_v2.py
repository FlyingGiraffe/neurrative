from sentence_transformers import SentenceTransformer

from models.base import EmbeddingModel


class NVEmbedV2(EmbeddingModel):

  MODEL_NAME = "nvidia/NV-Embed-v2"

  def __init__(self, device):
    self.model = SentenceTransformer(
      self.MODEL_NAME,
      device=device,
      trust_remote_code=True,
    )
  
  @property
  def name(self):
    return "nv-embed-v2"

  @property
  def hf_model(self):
    return "nvidia/NV-Embed-v2"

  def encode(self, texts, batch_size=4):
    emb = self.model.encode(
      texts,
      batch_size=batch_size,
      normalize_embeddings=True,
      convert_to_numpy=True,
      show_progress_bar=True,
    )
    return emb