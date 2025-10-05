# Audio Analysis Playground

A minimal web app to import or record audio, preview, and search a phrase in the Whisper transcript via `/api/phrase_search`.

## Run backend + frontend

```bash
cd "/Users/ivangvardeitsev/Computer Science/Projects/Junction2025"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS: ensure ffmpeg is installed first:  brew install ffmpeg
python server.py
```

Open `http://127.0.0.1:8000`.

## Use

- Select an audio file or record with the microphone
- Enter a phrase and click "Search phrase"; the app uploads your audio to `/api/phrase_search` and shows matching transcript segments with timestamps

## Whisper integration

- The backend loads a Whisper model once at startup (`WHISPER_MODEL` env var; default `base`).
- On upload, Whisper transcribes and returns segments; any segment containing the phrase is returned with its text and time span.

### Dependencies

- `openai-whisper` (Python)
- `ffmpeg` must be available on your PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`

### Environment

- `WHISPER_MODEL` to choose model size (e.g. `tiny`, `base`, `small`, `medium`, `large`). Example:

```bash
WHISPER_MODEL=small python server.py
```

## API

- `POST /api/phrase_search`
  - Body: `multipart/form-data`
    - `audio`: file (required)
    - `phrase`: text (required)
  - Response:

```json
{ "phrase": "low iq", "segments": [ { "start": 12.4, "end": 17.8, "text": "..." } ], "transcript": "..." }
```

Notes: requires `ffmpeg` to be available on PATH.
