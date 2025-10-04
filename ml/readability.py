import nltk
import re
import string
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

def count_syllables(word: str) -> int:
  groups = re.findall(r'[aeiouy]+', (word or '').lower())
  return max(1, len(groups))

def flesch_score(text: str):
  sentences = sent_tokenize(text or '')
  if not sentences:
    return None
  words = []
  for sent in sentences:
    words.extend([w for w in word_tokenize(sent) if any(c.isalpha() for c in w)])
  if not words:
    return None
  syllables = sum(count_syllables(w.lower()) for w in words)
  return 206.835 - 1.015 * (len(words)/len(sentences)) - 84.6 * (syllables/len(words))

def _get_random_synonym(word: str, pos_tag: str):
  tag_map = {
    'NN': wordnet.NOUN, 'NNS': wordnet.NOUN,
    'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
    'JJ': wordnet.ADJ,  'JJR': wordnet.ADJ,
    'JJS': wordnet.ADJ,
  }
  if pos_tag not in tag_map:
    return None
  wn_tag = tag_map[pos_tag]
  synonyms = set()
  for syn in wordnet.synsets(word, pos=wn_tag):
    for lemma in syn.lemmas():
      candidate = lemma.name().replace('_', ' ')
      if (candidate.lower() != word.lower() and candidate.isalpha() and len(candidate.split()) == 1):
        synonyms.add(candidate)
  if not synonyms:
    return None
  import random
  return random.choice(list(synonyms))

def replace_one_if_possible(text: str) -> str:
  words = (text or '').split()
  clean = [w.strip(string.punctuation) for w in words]
  pos_tags = nltk.pos_tag(clean)
  candidates = [i for i, (_, t) in enumerate(pos_tags) if t.startswith(('NN','JJ','VB'))]
  if not candidates:
    return text
  import random
  idx = random.choice(candidates)
  word, tag = pos_tags[idx]
  syn = _get_random_synonym(word, tag)
  if not syn:
    return text
  orig = words[idx]
  loc = orig.find(clean[idx])
  prefix = orig[:loc] if loc > 0 else ''
  suffix = orig[loc + len(clean[idx]):]
  words[idx] = f"{prefix}{syn}{suffix}"
  return ' '.join(words)

def give_sentence_with_synonym(sentence: str) -> str:
  original_score = flesch_score(sentence) or 0.0
  tried = set()
  while True:
    candidate = replace_one_if_possible(sentence)
    if candidate == sentence or candidate in tried:
      return sentence
    tried.add(candidate)
    cand_score = flesch_score(candidate)
    if cand_score is not None and cand_score < original_score:
      return candidate

def get_harder(original: str, N: int = 2):
  generated = [original]
  scores = [flesch_score(original)]
  seen = {original}
  while len(generated) < N:
    prev, prev_score = generated[-1], scores[-1]
    best_cand, best_score = None, prev_score
    for _ in range(250):
      cand = replace_one_if_possible(prev)
      if cand in seen:
        continue
      sc = flesch_score(cand)
      if sc is None or sc > (prev_score or 0) - 1:
        continue
      if best_cand is None or sc > (best_score or -1e9):
        best_cand, best_score = cand, sc
    if best_cand:
      generated.append(best_cand)
      scores.append(best_score)
      seen.add(best_cand)
    else:
      break
  return generated, scores

def get_same(start: str, N: int = 5):
  generated = [start]
  scores = [flesch_score(start)]
  seen = {start}
  while len(generated) < N:
    prev, prev_score = generated[-1], scores[-1]
    best_cand, best_score = None, prev_score
    for _ in range(200):
      cand = replace_one_if_possible(prev)
      if cand in seen:
        continue
      sc = flesch_score(cand)
      if sc is None or not ((prev_score or 0) - 1 < sc <= (prev_score or 0)):
        continue
      if best_cand is None or sc > (best_score or -1e9):
        best_cand, best_score = cand, sc
    if best_cand:
      generated.append(best_cand)
      scores.append(best_score)
      seen.add(best_cand)
    else:
      break
  return generated, scores


