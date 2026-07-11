from abc import ABC, abstractmethod


class EmbeddingModel(ABC):

  @property
  def name(self):
    return "base"

  @property
  def hf_model(self):
    return "none"
  
  @abstractmethod
  def encode(self, texts, batch_size=16):
    pass