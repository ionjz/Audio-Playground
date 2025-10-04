from .lexicon import get_normalized_emotions
from .embeddings import sbert_cosine_similarity
from .profanity import count_insults, count_bad_verbs
from .readability import flesch_score, give_sentence_with_synonym, get_harder, get_same
from .features import extract_features, extract_features_for_training
from .synthetic import synthetic_sentences
from .model import train_classifier, predict_extremism, build_training_dataframe, evaluate_classifier


