# Audio Analysis Playground

Minimal app to upload/record audio, transcribe with Whisper, and return predicted extremist segments.

## Quick start

Prereqs: Python 3.10+, ffmpeg on PATH (macOS: `brew install ffmpeg`).

```bash
cd "/Users/ivangvardeitsev/Computer Science/Projects/Junction2025"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000` and drop an audio file or record.

## API

POST `/api/phrase_search`
- Body: `multipart/form-data`
  - `audio`: file (required)
- Response (example):

```json
{
  "isExtremist": 0,
  "segments": [ { "start": 12.4, "end": 17.8, "text": "..." } ],
  "transcript": "..."
}
```

Notes: Whisper downloads a model on first run (default `base`). Long files may take a few minutes.
