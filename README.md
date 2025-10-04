# Audio Analysis Playground

A minimal web app to import or record audio, preview, and analyze via a Python Whisper backend at `/api/analyze`.

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
- Click "Analyze audio"; the app uploads your audio to `/api/analyze` and shows top tone segments and per-emotion scores when available

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

- `POST /api/analyze`
  - Body: `multipart/form-data`
    - `audio`: file (required)
  - Response (tone + optional emotions):

```json
{
  "isExtremist": null,
  "riskScore": 0.64,
  "segments": [
    { "start": 12.3, "end": 13.8, "label": "angry", "toneScore": 0.81, "angerProb": 0.66, "text": "optional transcript snippet" }
  ],
  "emotions": [ { "label": "anger", "score": 0.62 } ]
}
```

- `POST /api/analyze_tone`
  - Body: `multipart/form-data`
    - `audio`: file (required)
  - Response:

```json
{
  "isExtremist": false,
  "riskScore": 0.37,
  "segments": [
    { "start": 3.5, "end": 5.0, "angerProb": 0.61, "toneScore": 0.58, "label": "angry" }
  ]
}
```

Notes: requires `ffmpeg`, `librosa`, `pydub`, `soundfile`, `torch`, and `transformers`. The emotion model runs on CPU by default unless CUDA is available.

- `POST /api/emotions`
  - Body: `multipart/form-data`
    - `audio`: file (required)
  - Response:

```json
{ "emotions": [ { "label": "anger", "score": 0.62 }, { "label": "sadness", "score": 0.03 } ] }
```

- `POST /api/phrase_search`
  - Body: `multipart/form-data`
    - `audio`: file (required)
    - `phrase`: text (required)
  - Response:

```json
{ "phrase": "low iq", "matches": [ { "start": 12.4, "end": 17.8, "text": "..." } ], "transcript": "..." }
```

## Extremism ML training

Install extra deps (already in `requirements.txt`) and download NLTK data once:

```bash
cd "/Users/ivangvardeitsev/Computer Science/Projects/Junction2025"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python - <<'PY'
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
PY
```

Prepare small lexicon files if you don't have them yet:

```bash
[ -f NRC-Emotion-Lexicon-Wordlevel-v0.92.txt ] || cat > NRC-Emotion-Lexicon-Wordlevel-v0.92.txt <<'TXT'
hate	anger	1
love	joy	1
fear	fear	1
kill	anger	1
bad	negative	1
good	positive	1
TXT

[ -f terribleWordsForHackathon.txt.txt ] || cat > terribleWordsForHackathon.txt.txt <<'TXT'
hate
kill
enemy
TXT

[ -f bad_verbs.txt ] || cat > bad_verbs.txt <<'TXT'
destroy
eliminate
crush
TXT
```

Train and evaluate the RandomForest model, and optionally predict a sample:

```bash
source .venv/bin/activate
python -m scripts.train_extremism --predict "I hate the ninja turtles"
```

Environment variables:

- `NRC_LEXICON_PATH` to point to a custom NRC lexicon file
- `SBERT_MODEL` to choose a Sentence-BERT model (default `all-MiniLM-L6-v2`)
