[ Multi-Format Files ] ──► [ Unstructured Ingestion ] ──► [ Vector DB / Search ]
 (PDF, CSV, MD, DOCX)          (LangChain / LlamaIndex)        (Chroma / FAISS)
                                       │                              │
                                       ▼                              ▼
                              [ Postgres / SQLite ] ◄───► [ Local/API LLM Router ]
                               (Telemetry & BI Logs)            (Ollama / vLLM)
                                       │                              │
                                       ▼                              ▼
                              [ Streamlit / FastUI ] ◄───────── [ RAG Chat UI ]
                               (BI Performance Dashboard)