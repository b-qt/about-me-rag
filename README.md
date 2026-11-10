# 📚 Brenda's Resume & Career RAG Chatbot (Mind Map V2)

An end-to-end, locally hosted Retrieval-Augmented Generation (RAG) system and Streamlit web application designed to query academic logs, official employment history, and targeted job roles with high precision.

Powered by **ChromaDB**, **SentenceTransformers**, **Ollama (`llama3.2:1b`)**, and **Tesseract OCR**, this system uses a custom **Mind Map V2 (Where, How, Who, Why)** framework to eliminate cross-company hallucinations, segregate universities from workplaces, and deliver verifiable career analytics.

---

## 🌟 Key Features

* **Local RAG Architecture:** 100% privacy-focused and running locally via Ollama (`llama3.2:1b`) and persistent ChromaDB vector storage.
* **OCR Ingestion Pipeline:** Enhanced image preprocessing script (`image_converter.py`) utilizing 2x bicubic upscaling, low-DPI screen font optimization, contrast tuning, and continuous-line post-processing that preserves bullet structure (`*`, `©`, numeric lists).
* **Semantic Document Chunking:** Custom `SemanticSplitter` leveraging sentence embedding cosine distances (`all-MiniLM-L6-v2`) to chunk text dynamically at natural semantic boundaries.
* **Mind Map V2 Framework:** System prompt guardrails structured around:
  * 🗺️ **Where:** Physical/virtual context (e.g., Remote in Spain, Harbour Space Institute, GIMAC-BEAC).
  * 🛠️ **How:** Technical mechanics, data modeling, SQL/Python pipelines, and ETL orchestration.
  * 👥 **Who:** Stakeholders, international policy-makers, and collaborating research/engineering teams.
  * 🎯 **Why:** Measurable business impact, system reliability improvements, and strategic transformations.
* **Domain Segregation & Anti-Hallucination Guardrails:** Strict prompt partitioning separating official resume history (Emerging Engineering, UNDP, GIMAC-BEAC), academic institutions, and interview target roles (Sagaci Research, Pillar, Air Apps).

---

## 🛠️ Tech Stack

* **UI & Frontend:** Streamlit
* **Vector Database:** ChromaDB (`PersistentClient`)
* **Embedding Model:** `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **Local LLM Orchestration:** Ollama (`llama3.2:1b`) via HTTP REST API
* **OCR & Document Processing:** PyTesseract, Pillow (`PIL`), PyPDF (`PdfReader`)
* **Languages & Analytics:** Python 3.10+, SQL, NumPy, Regex

---

## 📁 Repository Structure

```text
├── data/                         # Knowledge base directory (PDFs, TXT, OCR outputs)
│   └── chroma_db/                # Local ChromaDB persistent vector database (Git ignored)
├── src/
│   ├── app.py                    # Main Streamlit web application & session state handler
│   ├── rag_engine.py             # ChromaDB retriever & Ollama generation prompt engine
│   ├── ingestion.py              # ETL pipeline reading documents into ChromaDB
│   ├── semantic_splitter.py      # Cosine similarity-based semantic sentence chunker
│   ├── image_converter.py        # Tesseract OCR converter (PNG/JPG -> Clean formatted TXT)
│   └── github_scraper.py         # REST API scraper pulling profile & repository READMEs
├── .gitignore                    # Excludes vector DB, cache files, and credentials
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+** installed on your system.
2. **Tesseract OCR Engine** installed locally:
* **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr`
* **macOS:** `brew install tesseract`


3. **Ollama** installed and running with `llama3.2:1b`:
```bash
ollama pull llama3.2:1b
ollama serve
```



### Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/b-qt/YOUR_REPOSITORY_NAME.git](https://github.com/b-qt/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt
```



---

## ⚙️ Running the Pipeline

### Step 1: Preprocess Screenshots & OCR (Optional)

Convert document screenshots (PNG/JPG) into continuous-formatted `.txt` files in `./data/`:

```bash
python src/image_converter.py
```

### Step 2: Fetch GitHub Repositories Snapshot (Optional)

Generate a clean snapshot of public GitHub profile metadata and project READMEs:

```bash
python src/github_scraper.py
```

### Step 3: Run Document Ingestion

Load, chunk, embed, and store all documents inside ChromaDB:

```bash
python src/ingestion.py
```

### Step 4: Launch the Web UI

Start the Streamlit web application:

```bash
streamlit run src/app.py
```

---

## 🧠 Mind Map V2 Query Examples

Once the application is running, click the quick-action sidebar buttons or query the chatbot directly:

* **Career Evolution:** *"Give me a summary of Brenda's career evolution using the Where, How, Who, and Why framework."*
* **Quantifiable Impact:** *"What measurable transformations and system reliability improvements has Brenda delivered across her roles?"*
* **Role Alignment:** *"How does Brenda's data engineering background map to product data analyst requirements at target companies?"*

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.