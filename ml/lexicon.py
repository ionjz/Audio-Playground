import os
import pandas as pd
from collections import defaultdict

LEXICON_FILE = os.environ.get('NRC_LEXICON_PATH', 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

_emotion_dict = None
_all_emotions = None

def _load_lexicon():
  global _emotion_dict, _all_emotions
  if _emotion_dict is not None:
    return
  df = pd.read_csv(LEXICON_FILE, sep='\t', names=['word', 'emotion', 'association'])
  emotion_dict = defaultdict(list)
  all_emotions = set()
  for _, row in df.iterrows():
    try:
      if int(row['association']) == 1:
        emotion_dict[str(row['word'])].append(str(row['emotion']))
        all_emotions.add(str(row['emotion']))
    except Exception:
      continue
  _emotion_dict = emotion_dict
  _all_emotions = sorted(list(all_emotions))

def get_normalized_emotions(text: str) -> dict:
  _load_lexicon()
  words = (text or '').lower().split()
  counts = defaultdict(int)
  for w in words:
    for emo in _emotion_dict.get(w, []):
      counts[emo] += 1
  full_counts = {emo: counts.get(emo, 0) for emo in _all_emotions}
  max_count = max(full_counts.values()) if full_counts else 1
  if max_count <= 0:
    max_count = 1
  return {emo: (full_counts[emo] / max_count) for emo in _all_emotions}


