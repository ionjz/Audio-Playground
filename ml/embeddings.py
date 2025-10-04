from sentence_transformers import SentenceTransformer
import numpy as np
import os

_MODEL_NAME = os.environ.get('SBERT_MODEL', 'all-MiniLM-L6-v2')
_model = SentenceTransformer(_MODEL_NAME)

def sbert_cosine_similarity(s1: str, s2: str, device=None) -> float:
  emb1 = _model.encode(s1 or '', convert_to_numpy=True, device=device)
  emb2 = _model.encode(s2 or '', convert_to_numpy=True, device=device)
  if emb1.ndim == 1:
    emb1 = emb1.reshape(1, -1)
  if emb2.ndim == 1:
    emb2 = emb2.reshape(1, -1)
  dot = np.dot(emb1, emb2.T).item()
  denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)).item()
  if denom == 0:
    return 0.0
  return float(dot / denom)


