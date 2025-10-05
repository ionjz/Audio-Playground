from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from collections import defaultdict
import random
import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

try:
  import whisper  # openai-whisper
except Exception as exc:  # pragma: no cover
  whisper = None  # type: ignore

ROOT = Path(__file__).parent.resolve()

app = Flask(
  __name__,
  static_folder=str(ROOT),
  static_url_path=''  # serve static assets from root: /index.html, /styles.css, /app.js
)


# Load Whisper model once at startup (if available)
WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL', 'base')
WHISPER_MODEL = None
if whisper is not None:
  try:
    WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
  except Exception:
    WHISPER_MODEL = None

def _normalize_text(text: str) -> str:
  # Minimal normalization to mirror user's example
  return (text or '').replace(',', '').replace('.', '').lower()


# def find_timestamps(biased_sentence: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#   matches: List[Dict[str, Any]] = []
#   for segment in segments:
#     segment_text = segment.get('text') or ''
#     segment_modified = _normalize_text(segment_text)
#     if biased_sentence in segment_modified:
#       start_time = float(segment.get('start') or 0.0)
#       end_time = float(segment.get('end') or start_time)
#       matches.append({
#         'start': start_time,
#         'end': end_time,
#         'text': segment_text,
#       })
#   return matches


@app.get('/')
def index():
  return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/health')
def health():
  return jsonify({
    'status': 'ok',
    'whisperLoaded': WHISPER_MODEL is not None,
    'model': WHISPER_MODEL_NAME,
  })


@app.post('/api/analyze')
def analyze():
  uploaded = request.files.get('audio')
  if not uploaded:
    return jsonify({ 'error': 'missing file field "audio"' }), 400

  # Persist upload to a temp file for Whisper
  tmp_file = None
  suffix = ''
  if uploaded.filename and '.' in uploaded.filename:
    suffix = '.' + uploaded.filename.rsplit('.', 1)[-1]
  tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or '.audio')
  tmp_file = tmp.name
  uploaded.save(tmp_file)
  tmp.close()

  # Transcribe and return transcript with segments
  transcript_text = ''
  transcript_segments = []
  if WHISPER_MODEL is not None:
    try:
      result = WHISPER_MODEL.transcribe(tmp_file)
      transcript_text = result.get('text') or ''
      transcript_segments = result.get('segments') or []
    except FileNotFoundError as e:
      # ffmpeg missing, still return tone-only results
      pass
    except Exception:
      # transcription failed, still return tone-only results
      pass

  return jsonify({
    'transcript': transcript_text,
    'segments': [
      { 'start': float(s.get('start') or 0.0), 'end': float(s.get('end') or 0.0), 'text': s.get('text') or '' }
      for s in (transcript_segments or [])
    ]
  })






@app.post('/api/phrase_search')
def phrase_search():
  uploaded = request.files.get('audio')
  if not uploaded:
    return jsonify({ 'error': 'missing file field "audio"' }), 400
  phrase = request.form.get('phrase', '')
  if WHISPER_MODEL is None:
    return jsonify({ 'error': 'whisper_model_not_loaded' }), 500
  phrase_norm = _normalize_text(phrase)
  tmp_file = None

  suffix = ''
  if uploaded.filename and '.' in uploaded.filename:
    suffix = '.' + uploaded.filename.rsplit('.', 1)[-1]
  tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or '.audio')
  tmp_file = tmp.name
  uploaded.save(tmp_file)
  tmp.close()
  try:
    result = WHISPER_MODEL.transcribe(tmp_file)
  except FileNotFoundError as e:
    return jsonify({ 'error': 'ffmpeg_not_found', 'detail': str(e) }), 500
  except Exception as e:
    return jsonify({ 'error': 'transcription_failed', 'detail': str(e) }), 500

  finally:
    transcript_text = result.get('text') or ''
    transcript_segments = result.get('segments') or []

  with open('rf_extremism_model.pkl', 'rb') as f:
    clf = pickle.load(f)

  model = whisper.load_model("base")
  transcription = model.transcribe(tmp_file)
  fullText = transcription["text"]
  segments = transcription["segments"]
  nltk.download('punkt')
  nltk.download('punkt_tab')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('averaged_perceptron_tagger_eng')
  nltk.download('wordnet')

  textData = []
  textLine = ""
  count = 0;

  for segment in segments:
    start_time = segment["start"]
    end_time = segment["end"]
    text = segment["text"]
    print(text)
    textLine += text;
    count+=1
    if(count == 3):
      textData.append(textLine)
      textLine = ""
      count = 0
  print(f"[{start_time:.2f}s -> {end_time:.2f}s] {text}")

  if textLine:
    textData.append(textLine)
  #Load feature extraction


  # 1. Download NLTK data



  # 2. Load NRC Emotion Lexicon
  LEXICON_FILE = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
  nrc_df = pd.read_csv(LEXICON_FILE, sep='\t', names=['word','emotion','association'])
  emotion_dict = defaultdict(list)
  all_emotions = set()
  for _, r in nrc_df.iterrows():
      if r.association == 1:
          emotion_dict[r.word].append(r.emotion)
          all_emotions.add(r.emotion)


  # 3. Load SBERT model
  _MODEL_NAME = 'all-MiniLM-L6-v2'
  _model = SentenceTransformer(_MODEL_NAME)


  # 4. Feature extraction functions
  def get_normalized_emotions(text):
      words = re.findall(r'\w+', text.lower())
      counts = defaultdict(int)
      for w in words:
          for emo in emotion_dict.get(w, []):
              counts[emo] += 1
      full = {e: counts.get(e,0) for e in all_emotions}
      m = max(full.values()) or 1
      return {e: full[e]/m for e in all_emotions}


  def sbert_cosine_similarity(s1, s2):
      e1 = _model.encode(s1, convert_to_numpy=True)
      e2 = _model.encode(s2, convert_to_numpy=True)
      if e1.ndim==1: e1=e1.reshape(1,-1)
      if e2.ndim==1: e2=e2.reshape(1,-1)
      dot = np.dot(e1, e2.T).item()
      den = (np.linalg.norm(e1)*np.linalg.norm(e2)).item() or 1
      return dot/den


  def read_file_to_list(path):
      with open(path,'r',encoding='utf-8') as f:
          return [l.strip() for l in f if l.strip()]


  def CountInsults(text):
      insults = read_file_to_list('terribleWordsForHackathon.txt.txt')
      words = re.findall(r'\w+', text.lower())
      return sum(1 for w in words if w in insults)


  def CountBadVerbs(text):
      bads = read_file_to_list('bad_verbs.txt')
      words = re.findall(r'\w+', text.lower())
      return sum(1 for w in words if w in bads)


  def CollectParams(text):
      return {'swear_count':CountInsults(text),'bad_verb_count':CountBadVerbs(text)}

  _PRED_CLF = None

  def _get_pickle_classifier():
    nonlocal _PRED_CLF
    if _PRED_CLF is not None:
      return _PRED_CLF
    try:
      with open('rf_extremism_model.pkl', 'rb') as f:
        _PRED_CLF = pickle.load(f)
    except Exception:
      _PRED_CLF = None
    return _PRED_CLF

  def find_timestamps(biased_sentence: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Return one element per segment whose predicted label == 1
    clf = _get_pickle_classifier()
    results: List[Dict[str, Any]] = []
    for segment in segments:
      seg_text = segment.get('text') or ''
      label = 0
      if clf is not None:
        try:
          # Prefer numeric feature path if classifier expects numeric inputs
          feats = extract_features(seg_text)
          X = pd.DataFrame([feats])
          if hasattr(clf, 'feature_names_in_'):
            X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
          pred = clf.predict(X)[0]
          label = int(pred)
        except Exception:
          # Fallback: assume pipeline that accepts raw text
          try:
            pred = clf.predict([seg_text])[0]
            label = int(pred)
          except Exception:
            label = 0
      if label == 1:
        start_time = float(segment.get('start') or 0.0)
        end_time = float(segment.get('end') or start_time)
        results.append({
          'start': start_time,
          'end': end_time,
          'text': seg_text,
          'prediction': 1,
        })
    return results

  def extract_features(sent):
      f = {}
      f.update(CollectParams(sent))
      f.update(get_normalized_emotions(sent))
      f['extremism_similarity'] = sbert_cosine_similarity(sent,"I hate every immigrant")
      f['extremism_countersimilarity'] = sbert_cosine_similarity(sent,"Everything is perfect")
      return f


  # predictions = [clf.predict([i])[0] for i in textData]
  predictions = []
  for i in textData:
    features = extract_features(text)
    X = pd.DataFrame([features])
    if hasattr(clf, 'feature_names_in_'):
      X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
    pred = clf.predict(X)[0]

    predictions.append(pred)



    is_extremist = 0
    for i in predictions:
      if i == 1:
        is_extremist = 1
    print(predictions[-1])
   
    matches = find_timestamps(i, transcript_segments)
  return jsonify({
    'phrase': phrase,
    'isExtremist': is_extremist,
    'segments': matches,
    'transcript': transcript_text
  })
  




  # finally:
  #   if tmp_file and os.path.exists(tmp_file):
  #     try:
  #       os.remove(tmp_file)
  #     except OSError:
  #       pass

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
