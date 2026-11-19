# About Brenda: a boxed career assistant (Mind Map V2)

A Streamlit app that answers questions about Quinsy Brenda Tebid's studies, roles, results and side projects **only from documents she provides**, and shows her career as metrics and a Where / How / Who / Why mind map.

- **Ask**: chat with cited sources. Career questions are answered as a Mind Map; factual questions get a direct answer. Saved job postings are used only for "fit" questions and are always labelled *not past work*.
- **Growth**: years active, organisations, countries, a timeline, seniority steps, toolset growth, measurable results, side projects.
- **Mind Map**: every role and degree as Where / How / Who / Why, with a button to ask the assistant about it.

It's retrieval-augmented generation (RAG), not a trained model: the LLM writes answers but may only use the passages retrieved from `knowledge/` plus the structured timeline, and must say so when something isn't covered.

## How it works

```text
knowledge/ ──► loaders ──► chunking (headings → semantic split → context headers)
                                │
                                ▼
               ChromaDB (all-MiniLM-L6-v2 via ONNX)  +  BM25 keyword index
                                │   hybrid search, type quotas, job-posting routing
                                ▼
       timeline.csv + top passages ──► Gemini → Groq (hosted) / Ollama (local) ──► answer + sources
```

## Project layout

```text
streamlit_app.py        entry point (multipage navigation)
app/                    chat.py · growth.py · mind_map.py · common.py
src/aboutme/            config · loaders · chunking · embeddings · index · retrieval · prompts · llm · assistant · timeline
knowledge/              the only content the assistant can see (see knowledge/README.md)
private/                originals: resume with contact details, offer letters, screenshots (git-ignored)
scripts/                build_index.py · ask.py · ocr_images.py · fetch_github.py
tests/                  golden questions, privacy check, timeline validation, LLM client
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # add GEMINI_API_KEY, or run Ollama:
ollama pull llama3.2:1b && ollama serve
streamlit run streamlit_app.py
```

The index builds itself on first run and again whenever anything in `knowledge/` changes. The first run downloads the ~80 MB embedding model.

Useful commands:

```bash
python scripts/build_index.py --query "What did she achieve at GIMAC-BEAC?"   # check retrieval
python scripts/ask.py "Map her career using Where, How, Who and Why"            # full answer in the terminal
pytest                                                                          # offline tests
```

`llama3.2:1b` is fast but follows the Mind Map format loosely; `llama3.2:3b` or larger gives noticeably better answers if your machine can run it (set `OLLAMA_MODEL`).

## Deploy publicly (Streamlit Community Cloud)

1. Check `private/` is not committed (`git status` should not list anything in it except `private/README.md`), then push to GitHub. A private repo works.
2. On share.streamlit.io: **Create app** → pick the repo → main file `streamlit_app.py` → Advanced settings → Python 3.12.
3. In **Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real keys. Use at least one of `GEMINI_API_KEY` (aistudio.google.com) and `GROQ_API_KEY` (console.groq.com, free tier with hosted Llama models). With both, Gemini answers first and Groq takes over whenever Gemini is busy.
4. Deploy. The first boot downloads the embedding model and builds the index (about a minute).
5. Smoke test: "Where did she do her master's?", "Did she work at Air Apps?" (correct answer: no, it's a saved posting), and the Career map button.

### When a model is busy or retired

Each provider retries busy (503), rate-limited (429) and timeout errors twice with a short backoff, then the next configured provider answers. The caption under each answer shows which model replied, and the sidebar shows the whole chain. If every provider fails, the chat says why and shows the matching passages instead. If a model is retired, update `GEMINI_MODEL` or `OPENAI_COMPAT_MODEL` in Secrets.

Your local Ollama model can't run on Streamlit Community Cloud: there is no way to run the Ollama server there, and free apps have too little memory. To use Llama in the public app, use Groq. To use a model on your own computer, expose your Ollama behind an authenticated tunnel and set `OLLAMA_HOST` and `OLLAMA_API_KEY`; the app is then down whenever your computer is. Ollama Cloud also works (`OLLAMA_HOST = "https://ollama.com"`, `OLLAMA_API_KEY`, and one of its cloud models).

## Docker

```bash
docker compose up --build        # uses Ollama on the host, or GEMINI_API_KEY from .env
```

## Updating the content

- New role or degree: add a row to `knowledge/timeline.csv` and a markdown file in `knowledge/experience/` or `knowledge/education/` with `timeline_id: <row id>` in the front matter.
- New side project: add a row to `knowledge/projects.csv`; optionally a longer write-up in `knowledge/projects/`.
- New job posting: OCR it with `scripts/ocr_images.py`, clean it up, save it in `knowledge/targets/` with `org:` in the front matter.
- Run `pytest` before pushing: it fails if a phone number, email, salary or signature block slipped into `knowledge/`.

## License

MIT for the code (see LICENSE). The content of `knowledge/` describes a real person and is not licensed for reuse.