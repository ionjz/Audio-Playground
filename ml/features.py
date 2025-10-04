from .lexicon import get_normalized_emotions
from .embeddings import sbert_cosine_similarity
from .profanity import count_insults, count_bad_verbs

def collect_params(text: str) -> dict:
  return {
    'swear_count': count_insults(text),
    'bad_verb_count': count_bad_verbs(text),
  }

def extract_features(sentence: str) -> dict:
  counters = collect_params(sentence)
  emotions = get_normalized_emotions(sentence)
  sim = sbert_cosine_similarity(sentence, 'I hate every immigrant')
  countersim = sbert_cosine_similarity(sentence, 'Everything is perfect')
  feats = {
    **counters,
    **emotions,
    'extremism_similarity': sim,
    'extremism_countersimilarity': countersim,
  }
  return feats

def extract_features_for_training(pair) -> dict:
  sentence, y_val = pair
  feats = extract_features(sentence)
  feats['IsExtremist'] = int(y_val)
  return feats


