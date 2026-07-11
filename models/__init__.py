from models.bge_m3 import BGEM3Embedding
from models.qwen3_embedding import Qwen3Embedding
from models.e5_large_v2 import E5LargeV2Embedding
# from models.nv_embed_v2 import NVEmbedV2
# from models.gte_qwen2 import GTEQwen2Embedding



MODELS = {
  "bge-m3": BGEM3Embedding,
  "qwen3-embedding": Qwen3Embedding,
  "e5-large-v2": E5LargeV2Embedding,
  # "nv-embed-v2": NVEmbedV2,  # need an older version of transformers (4.51.3) to run this model
  # "gte-qwen2": GTEQwen2Embedding,  # need an older version of transformers (4.51.3) to run this model
}


def build_model(name, device="cuda"):

  name = name.lower()

  if name not in MODELS:
    raise ValueError(
      f"Unknown policy model '{name}'. "
      f"Available models: {list(MODELS.keys())}"
    )

  return MODELS[name](device)