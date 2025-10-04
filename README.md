# Audio Analysis Playground

A minimal web app to import or record audio, preview, and run an analysis. Ships with a mock analyzer and a Python backend endpoint at `/api/analyze` you can replace with real logic.

## Run locally (recommended)

Start the Python backend (serves the frontend too):

```bash
cd "/Users/ivangvardeitsev/Computer Science/Projects/Junction2025"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000`. Microphone access requires localhost or HTTPS.

## Use

- Click the card to select an audio file (or drag & drop)
- Or click "Start recording" to capture via the microphone
- Choose "Mock" or "Live" and click "Analyze audio"
  - Mock: built-in deterministic sample results
  - Live: posts `multipart/form-data` with field `audio` to `/api/analyze`

Results include a summary and a clickable list of segments. Clicking a segment jumps the player to that time.

## Backend API

- Endpoint: `POST /api/analyze`
- Body: `multipart/form-data` with `audio` file field
- Response shape (example):

```json
{
  "isExtremist": true,
  "riskScore": 0.82,
  "segments": [
    { "start": 12.5, "end": 18.2, "label": "flag", "confidence": 0.91 }
  ]
}
```

You can implement your real analysis in `server.py` inside the `/api/analyze` handler. The current implementation returns a deterministic mock based on the upload size.

## Notes

- Tested in latest Chrome and Edge. Safari supports MediaRecorder in recent versions.
- Recording requires localhost or HTTPS.
- Most audio formats supported by the browser should work.
