def read_file_to_list(filename):
  with open(filename, 'r', encoding='utf-8') as f:
    return [line.strip() for line in f.read().split('\n') if line.strip()]

def _contains(word, words):
  return word in words

def count_list_hits(text: str, list_path: str) -> int:
  words = read_file_to_list(list_path)
  tokens = (text or '').strip().lower().replace(',', '').replace('.', '').split()
  return sum(1 for t in tokens if _contains(t, words))

def count_insults(text: str, insults_path: str = 'terribleWordsForHackathon.txt.txt') -> int:
  return count_list_hits(text, insults_path)

def count_bad_verbs(text: str, verbs_path: str = 'bad_verbs.txt') -> int:
  return count_list_hits(text, verbs_path)


