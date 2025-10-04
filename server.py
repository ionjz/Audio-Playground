from __future__ import annotations

import os
import math
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import io
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import torch
from transformers import pipeline
from ml.model import load_model, predict_proba_extremism

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

# Load trained extremism classifier (optional)
EXTREMISM_MODEL_PATH = os.environ.get('EXTREMISM_MODEL', str(ROOT / 'models' / 'extremism_rf.pkl'))
EXTREMISM_CLF = None
EXTREMISM_SCALER = None
try:
  if os.path.exists(EXTREMISM_MODEL_PATH):
    EXTREMISM_CLF, EXTREMISM_SCALER = load_model(EXTREMISM_MODEL_PATH)
except Exception:
  EXTREMISM_CLF, EXTREMISM_SCALER = None, None

# Load Whisper model once at startup (if available)
WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL', 'base')
WHISPER_MODEL = None
if whisper is not None:
  try:
    WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
  except Exception:
    WHISPER_MODEL = None


# ===================== Tone/Emotion Analysis Config =====================
TARGET_SR = 16000
WINDOW_SEC = 1.5
HOP_SEC = 0.5
MIN_SEGMENTS_FOR_SCORE = 3
ANGER_LABELS = {"anger", "angry"}  # model-dependent label names
RISK_SEGMENT_THRESHOLD = 0.55
GLOBAL_RISK_STRONG = 0.75
TOP_K_TONE_SEGMENTS = 5  # number of top tone segments to return
TEXT_SEGMENT_THRESHOLD = float(os.environ.get('TEXT_SEGMENT_THRESHOLD', '0.5'))

# Load emotion pipeline once at startup (optional)
EMOTION_PIPE = None
try:
  EMOTION_PIPE = pipeline(
    "audio-classification",
    model="superb/hubert-base-superb-er",
    device=0 if torch.cuda.is_available() else -1,
    top_k=None
  )
except Exception:
  EMOTION_PIPE = None


def _normalize_text(text: str) -> str:
  # Minimal normalization to mirror user's example
  return (text or '').replace(',', '').replace('.', '').lower()


def find_timestamps(biased_sentence: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  matches: List[Dict[str, Any]] = []
  for segment in segments:
    segment_text = segment.get('text') or ''
    segment_modified = _normalize_text(segment_text)
    if biased_sentence in segment_modified:
      start_time = float(segment.get('start') or 0.0)
      end_time = float(segment.get('end') or start_time)
      matches.append({
        'start': start_time,
        'end': end_time,
        'text': segment_text,
      })
  return matches


@app.get('/')
def index():
  return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/health')
def health():
  return jsonify({
    'status': 'ok',
    'whisperLoaded': WHISPER_MODEL is not None,
    'model': WHISPER_MODEL_NAME,
    'emotionLoaded': EMOTION_PIPE is not None,
    'extremismModelLoaded': EXTREMISM_CLF is not None,
  })


@app.post('/api/analyze')
def analyze():
  uploaded = request.files.get('audio')
  if not uploaded:
    return jsonify({ 'error': 'missing file field "audio"' }), 400

  # Persist upload to a temp file for Whisper
  tmp_file = None
  try:
    suffix = ''
    if uploaded.filename and '.' in uploaded.filename:
      suffix = '.' + uploaded.filename.rsplit('.', 1)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or '.audio')
    tmp_file = tmp.name
    uploaded.save(tmp_file)
    tmp.close()

    # 1) Tone analysis (audio-based risk and segments)
    try:
      y, sr = read_audio_any(tmp_file, target_sr=TARGET_SR)
    except FileNotFoundError as e:
      return jsonify({ 'error': 'ffmpeg_not_found', 'detail': str(e) }), 500
    except Exception as e:
      return jsonify({ 'error': 'decode_failed', 'detail': str(e) }), 400

    if y is None or len(y) == 0:
      return jsonify({ 'error': 'Could not decode audio' }), 400

    duration = len(y) / float(sr)
    seg_feats = []
    seg_times = []
    for start, end in segment_bounds(duration, WINDOW_SEC, HOP_SEC):
      feat = segment_paralinguistics(y, sr, start, end)
      if feat:
        seg_feats.append(feat)
        seg_times.append((start, end))

    tone_scores = []
    all_segments = []
    if seg_feats:
      rms_all = np.array([s['rms_mean'] for s in seg_feats], dtype=float)
      pitchstd_all = np.array([s['pitch_std'] for s in seg_feats], dtype=float)
      centroid_all = np.array([s['centroid_mean'] for s in seg_feats], dtype=float)

      for (start, end), feat in zip(seg_times, seg_feats):
        seg_wav = slice_audio(y, sr, start, end)
        probs = classify_emotion_segment(seg_wav, sr)
        anger_prob = 0.0
        for lab, p in probs.items():
          if any(a in lab for a in ANGER_LABELS):
            anger_prob = max(anger_prob, float(p))

        tone_score = compute_segment_tone_score(
          anger_prob=anger_prob,
          rms_mean=feat['rms_mean'],
          pitch_std=feat['pitch_std'],
          centroid_mean=feat['centroid_mean'],
          rms_all=rms_all,
          pitchstd_all=pitchstd_all,
          centroid_all=centroid_all,
        )
        tone_scores.append(tone_score)
        all_segments.append({
          'start': round(float(start), 2),
          'end': round(float(end), 2),
          'angerProb': round(float(anger_prob), 3),
          'toneScore': round(float(tone_score), 3),
          'label': 'angry' if anger_prob >= 0.5 else 'high-arousal',
        })

    tone_risk = robust_global_score(tone_scores, top_frac=0.30) if tone_scores else 0.0

    # 2) Transcribe and use transcript segments for text-based scoring
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

    # Also compute whole-clip per-emotion scores (optional)
    emotions = []
    try:
      if EMOTION_PIPE is not None:
        # Use entire resampled clip as ndarray
        outputs = EMOTION_PIPE(y.astype(np.float32), sampling_rate=sr)
        # outputs: list of {'label','score'}
        for d in (outputs or []):
          label = str(d.get('label', '')).lower()
          score = float(d.get('score', 0.0))
          emotions.append({ 'label': label, 'score': round(score, 3) })
    except Exception:
      emotions = []

    # 3) Text-based extremism prediction (global + per-transcript segment)
    text_prob = None
    text_label = None
    if EXTREMISM_CLF is not None and EXTREMISM_SCALER is not None and transcript_text:
      try:
        text_prob = predict_proba_extremism(EXTREMISM_CLF, EXTREMISM_SCALER, transcript_text)
        text_label = int((text_prob or 0.0) >= 0.5)
      except Exception:
        text_prob, text_label = None, None

    # Per-segment text scoring over Whisper segments
    text_segments = []
    if EXTREMISM_CLF is not None and EXTREMISM_SCALER is not None and transcript_segments:
      for seg in transcript_segments:
        s_text = (seg.get('text') or '').strip()
        if not s_text:
          continue
        try:
          s_prob = predict_proba_extremism(EXTREMISM_CLF, EXTREMISM_SCALER, s_text)
        except Exception:
          s_prob = None
        if isinstance(s_prob, float) and s_prob >= TEXT_SEGMENT_THRESHOLD:
          text_segments.append({
            'start': round(float(seg.get('start') or 0.0), 2),
            'end': round(float(seg.get('end') or 0.0), 2),
            'text': s_text,
            'textScore': round(float(s_prob), 3)
          })

    return jsonify({
      'isExtremist': text_label,
      'textScore': round(float(text_prob), 3) if isinstance(text_prob, float) else None,
      # For backward compat, leave riskScore but set to text score as requested
      'riskScore': round(float(text_prob), 3) if isinstance(text_prob, float) else None,
      'segments': text_segments,
      'transcript': transcript_text,
      'emotions': emotions
    })

  finally:
    if tmp_file and os.path.exists(tmp_file):
      try:
        os.remove(tmp_file)
      except OSError:
        pass


# ===================== Tone/Emotion Helpers =====================
def read_audio_any(in_path: str, target_sr: int = TARGET_SR):
  # Try librosa first
  try:
    y, sr = librosa.load(in_path, sr=target_sr, mono=True)
    return y.astype(np.float32), sr
  except Exception:
    pass

  # Fallback: pydub+ffmpeg decode to wav buffer, then read with soundfile
  audio = AudioSegment.from_file(in_path)
  audio = audio.set_channels(1).set_frame_rate(target_sr)
  buf = io.BytesIO()
  audio.export(buf, format='wav')
  buf.seek(0)
  y, sr = sf.read(buf, dtype="float32", always_2d=False)
  if getattr(y, 'ndim', 1) > 1:
    y = y.mean(axis=1)
  return y.astype(np.float32), sr


def segment_bounds(duration: float, win: float, hop: float):
  t = 0.0
  while t < max(1e-9, duration):
    start = t
    end = min(duration, t + win)
    yield start, end
    t += hop
    if end >= duration:
      break


def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float):
  a = int(start_s * sr)
  b = int(end_s * sr)
  return y[a:b]


def classify_emotion_segment(seg_wav: np.ndarray, sr: int):
  if EMOTION_PIPE is None:
    return {}
  try:
    # Transformers audio-classification expects ndarray/tensor + sampling_rate
    out = EMOTION_PIPE(seg_wav.astype(np.float32), sampling_rate=sr)
  except Exception:
    return {}
  return {d.get("label", "").lower(): float(d.get("score", 0.0)) for d in (out or [])}


def segment_paralinguistics(y: np.ndarray, sr: int, start: float, end: float):
  seg = slice_audio(y, sr, start, end)
  if seg.size == 0:
    return None
  rms = librosa.feature.rms(y=seg)[0]
  sc = librosa.feature.spectral_centroid(y=seg, sr=sr)[0]
  zcr = librosa.feature.zero_crossing_rate(seg)[0]
  try:
    f0 = librosa.yin(seg, fmin=50, fmax=300, sr=sr)
    f0 = f0[np.isfinite(f0)]
  except Exception:
    f0 = np.array([])

  return {
    'rms_mean': float(np.mean(rms)) if rms.size else 0.0,
    'centroid_mean': float(np.mean(sc)) if sc.size else 0.0,
    'zcr_mean': float(np.mean(zcr)) if zcr.size else 0.0,
    'pitch_std': float(np.std(f0)) if f0.size else 0.0,
  }


def compute_segment_tone_score(
  anger_prob: float,
  rms_mean: float,
  pitch_std: float,
  centroid_mean: float,
  rms_all: np.ndarray,
  pitchstd_all: np.ndarray,
  centroid_all: np.ndarray,
):
  # Z-score within the clip
  z_rms = (rms_mean - float(np.mean(rms_all))) / (float(np.std(rms_all)) + 1e-8)
  z_pitch = (pitch_std - float(np.mean(pitchstd_all))) / (float(np.std(pitchstd_all)) + 1e-8)
  z_cent = (centroid_mean - float(np.mean(centroid_all))) / (float(np.std(centroid_all)) + 1e-8)

  # Clip z-scores to limit outliers
  def _clip(x: float, a: float = -2.0, b: float = 2.0) -> float:
    return float(min(max(x, a), b))

  z_rms_c = _clip(z_rms)
  z_pitch_c = _clip(z_pitch)
  z_cent_c = _clip(z_cent)

  # Map z to [0,1]
  z_rms_01 = (z_rms_c + 2.0) / 4.0
  z_pitch_01 = (z_pitch_c + 2.0) / 4.0

  # Calibrate anger probability to boost above-baseline values
  # Everything <= floor maps to 0; >= ceil maps to 1
  floor, ceil = 0.10, 0.70
  cal_ang = (float(anger_prob) - floor) / (ceil - floor)
  cal_ang = float(min(1.0, max(0.0, cal_ang)))

  # Heuristic fusion (weights sum to 1)
  combined = 0.85 * cal_ang + 0.12 * z_rms_01 + 0.03 * z_pitch_01

  # Gamma to expand high region and compress low region
  gamma = 0.65
  combined = float(min(1.0, max(0.0, combined)))
  score = combined ** gamma
  return score


def robust_global_score(scores: list[float], top_frac: float = 0.30):
  if not scores:
    return 0.0
  s = np.array(sorted(scores), dtype=float)
  k = max(1, int(len(s) * top_frac))
  top = s[-k:]
  p95 = float(np.percentile(top, 95))
  top = np.clip(top, 0.0, p95)
  return float(np.mean(top))


@app.post('/api/analyze_tone')
def analyze_tone():
  if 'audio' not in request.files:
    return jsonify({ 'error': "No audio file part 'audio'" }), 400

  if EMOTION_PIPE is None:
    return jsonify({ 'error': 'emotion_model_not_loaded' }), 500

  f = request.files['audio']
  filename = secure_filename(f.filename or 'audio.webm')

  with tempfile.TemporaryDirectory() as td:
    raw_path = os.path.join(td, filename)
    f.save(raw_path)

    try:
      y, sr = read_audio_any(raw_path, target_sr=TARGET_SR)
    except FileNotFoundError as e:
      return jsonify({ 'error': 'ffmpeg_not_found', 'detail': str(e) }), 500
    except Exception as e:
      return jsonify({ 'error': 'decode_failed', 'detail': str(e) }), 400

    if y is None or len(y) == 0:
      return jsonify({ 'error': 'Could not decode audio' }), 400

    duration = len(y) / float(sr)
    if duration < 0.3:
      return jsonify({ 'error': 'Audio too short' }), 400

    # Extract per-segment features first for global stats
    seg_feats = []
    seg_times = []
    for start, end in segment_bounds(duration, WINDOW_SEC, HOP_SEC):
      feat = segment_paralinguistics(y, sr, start, end)
      if feat:
        seg_feats.append(feat)
        seg_times.append((start, end))

    if not seg_feats:
      return jsonify({ 'error': 'No valid segments' }), 400

    rms_all = np.array([s['rms_mean'] for s in seg_feats], dtype=float)
    pitchstd_all = np.array([s['pitch_std'] for s in seg_feats], dtype=float)
    centroid_all = np.array([s['centroid_mean'] for s in seg_feats], dtype=float)

    segments = []
    tone_scores = []
    for (start, end), feat in zip(seg_times, seg_feats):
      seg_wav = slice_audio(y, sr, start, end)
      probs = classify_emotion_segment(seg_wav, sr)
      anger_prob = 0.0
      for lab, p in probs.items():
        if any(a in lab for a in ANGER_LABELS):
          anger_prob = max(anger_prob, float(p))

      tone_score = compute_segment_tone_score(
        anger_prob=anger_prob,
        rms_mean=feat['rms_mean'],
        pitch_std=feat['pitch_std'],
        centroid_mean=feat['centroid_mean'],
        rms_all=rms_all,
        pitchstd_all=pitchstd_all,
        centroid_all=centroid_all,
      )
      tone_scores.append(tone_score)

      if tone_score >= RISK_SEGMENT_THRESHOLD:
        segments.append({
          'start': round(float(start), 2),
          'end': round(float(end), 2),
          'angerProb': round(float(anger_prob), 3),
          'toneScore': round(float(tone_score), 3),
          'label': 'angry' if anger_prob >= 0.5 else 'high-arousal',
        })

    risk_score = robust_global_score(tone_scores, top_frac=0.30)

    # Whole-clip emotions using the same model
    emotions = []
    try:
      if EMOTION_PIPE is not None:
        outputs = EMOTION_PIPE(y.astype(np.float32), sampling_rate=sr)
        for d in (outputs or []):
          label = str(d.get('label', '')).lower()
          score = float(d.get('score', 0.0))
          emotions.append({ 'label': label, 'score': round(score, 3) })
    except Exception:
      emotions = []

    return jsonify({
      'isExtremist': None,
      'riskScore': round(float(risk_score), 3),
      'segments': segments,
      'emotions': emotions
    })


@app.post('/api/emotions')
def emotions_route():
  if 'audio' not in request.files:
    return jsonify({ 'error': "No audio file part 'audio'" }), 400
  if EMOTION_PIPE is None:
    return jsonify({ 'error': 'emotion_model_not_loaded' }), 500
  f = request.files['audio']
  filename = secure_filename(f.filename or 'audio.webm')
  with tempfile.TemporaryDirectory() as td:
    raw_path = os.path.join(td, filename)
    f.save(raw_path)
    try:
      y, sr = read_audio_any(raw_path, target_sr=TARGET_SR)
    except FileNotFoundError as e:
      return jsonify({ 'error': 'ffmpeg_not_found', 'detail': str(e) }), 500
    except Exception as e:
      return jsonify({ 'error': 'decode_failed', 'detail': str(e) }), 400
    if y is None or len(y) == 0:
      return jsonify({ 'error': 'Could not decode audio' }), 400
    try:
      outputs = EMOTION_PIPE(y.astype(np.float32), sampling_rate=sr)
    except Exception as e:
      return jsonify({ 'error': 'emotion_inference_failed', 'detail': str(e) }), 500
    emotions = [ { 'label': str(d.get('label','')).lower(), 'score': round(float(d.get('score',0.0)), 3) } for d in (outputs or []) ]
    return jsonify({ 'emotions': emotions })


@app.post('/api/phrase_search')
def phrase_search():
  uploaded = request.files.get('audio')
  if not uploaded:
    return jsonify({ 'error': 'missing file field "audio"' }), 400
  phrase = request.form.get('phrase', '')
  if not phrase:
    return jsonify({ 'error': 'missing form field "phrase"' }), 400
  if WHISPER_MODEL is None:
    return jsonify({ 'error': 'whisper_model_not_loaded' }), 500
  phrase_norm = _normalize_text(phrase)
  tmp_file = None
  try:
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
    transcript_text = result.get('text') or ''
    transcript_segments = result.get('segments') or []
    matches = find_timestamps(phrase_norm, transcript_segments)
    return jsonify({
      'phrase': phrase,
      'matches': matches,
      'transcript': transcript_text
    })
  finally:
    if tmp_file and os.path.exists(tmp_file):
      try:
        os.remove(tmp_file)
      except OSError:
        pass

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
