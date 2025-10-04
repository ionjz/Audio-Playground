from __future__ import annotations

import math
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

ROOT = Path(__file__).parent.resolve()

app = Flask(
  __name__,
  static_folder=str(ROOT),
  static_url_path=''  # serve static assets from root: /index.html, /styles.css, /app.js
)


@app.get('/')
def index():
  # Serve the main page
  return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/health')
def health():
  return jsonify({ 'status': 'ok' })


@app.post('/api/analyze')
def analyze():
  uploaded = request.files.get('audio')
  if not uploaded:
    return jsonify({ 'error': 'missing file field "audio"' }), 400

  # Read the bytes; in a real system you would feed this into your model
  data = uploaded.read() or b''

  # Simple deterministic mock scoring based on file size
  size_bytes = len(data)
  seed = (size_bytes % 17) / 17.0
  risk_score = min(1.0, 0.25 + seed * 0.75)
  extremist = True if risk_score > 0.7 else False if risk_score < 0.4 else None

  # Make a couple of fake segments based on the seed
  total_seconds = 60 + int((seed * 100) % 90)  # 60..150
  segments = []
  if risk_score > 0.45:
    for i in range(int(1 + math.floor(seed * 3))):
      start = max(0, int((total_seconds / (2 + i)) - 8))
      end = min(total_seconds, start + 6 + i)
      segments.append({
        'start': float(start),
        'end': float(end),
        'label': 'flag',
        'confidence': round(min(1.0, 0.6 + seed * 0.4), 2)
      })

  return jsonify({
    'isExtremist': extremist,
    'riskScore': round(risk_score, 3),
    'segments': segments
  })


if __name__ == '__main__':
  # Run the dev server. Use a real WSGI server (gunicorn/uvicorn) in production.
  app.run(host='127.0.0.1', port=8000, debug=True)
