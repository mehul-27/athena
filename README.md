# Athena

Athena is a locally-runnable LLM assistant with **Chat**, **Web Search** and
**RAG** (retrieval-augmented chat). This README covers how to set the project up.

## Prerequisites

- Python 3.11 or newer (developed on 3.13)
- Git
- Internet access on first run, to download the local embedding model (optional — see below)

## Setup

Clone the repository and create a virtual environment:

```cmd
git clone https://github.com/mehul-27/athena.git
cd athena
python -m venv .venv
```

Install the dependencies.

Windows (cmd):

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Create your environment file from the template.

Windows (cmd):

```cmd
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

## Configure

Athena needs at least one LLM provider key. You can either:

- put one in `.env` (`NVIDIA_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or
  `GOOGLE_API_KEY`) as a one-time bootstrap, **or**
- just start Athena and add providers in the **Settings** tab, which is the
  primary configuration interface from then on.

Everything else in `.env` has a working default. In particular:

- ChromaDB runs **embedded** under `data/chroma` — no external service required.
- Embeddings run **locally** via fastembed. The first run downloads the ONNX
  model (~50 MB) into `data/fastembed_cache`. For a zero-download, fully offline
  run, set `EMBEDDING_PROVIDER=hashing`.

Do not commit real secrets — `.env` is gitignored.

## Run

Start the server:

```cmd
.venv\Scripts\python.exe -m backend.main
```

Or with uvicorn:

```cmd
.venv\Scripts\python.exe -m uvicorn backend.main:create_app --factory --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000**.

## Tests

Deterministic, offline, no API key required:

```cmd
.venv\Scripts\python.exe -m pytest
```

Opt-in real-LLM end-to-end suite (requires a configured provider):

```cmd
set ATHENA_E2E=1
.venv\Scripts\python.exe -m pytest tests/e2e -m e2e -v
```
