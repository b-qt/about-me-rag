""" Document: RAG Engine 
This script performs retrieval (finding relevant text) and generation (finding grounded responses) 

    What it does
- Connects to local db 'academic_logs'
- Embeds query using the same transformer model
- Retrieves the top n most semantically relevant chunks
- Sends retrieved context to local LLM
"""
import os
import json
import urllib.request
import chromadb 

from sentence_transformers import SentenceTransformer

DATA_DIR = os.getenv("DATA_DIR","./data")  # Directory to read documents from
DB_DIR = os.getenv("DB_DIR","./data/chroma_db")  # Directory to store the ChromaDB database
COLLECTION_NAME = os.getenv("COLLECTION_NAME",'academic_logs')
MODEL_NAME = os.getenv("MODEL_NAME","all-MiniLM-L6-v2")

OLLAMA_ENDPOINT = os.getenv(
    "OLLAMA_ENDPOINT",
    "http://localhost:11434/api/generate"
)
LLM_MODEL = os.getenv("OLLAMA_MODEL","llama3.2:1b")

class LocalRAGEngine:
    def __init__(self):
        print(f"Initializing RAG engine (Model name: {LLM_MODEL})")
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception:
            print(f"Warning!! Collection '{COLLECTION_NAME} not found; run ingestion script")
            self.collection = None
    
    def retrieve(self, query: str, top_k: int=5) -> list:
        if not self.collection:
            return []
        
        
        
        
        print(f"\n\nSearching vector databasse for {query} ...")
        query_embeddings = self.embedding_model.encode([query]).tolist()
        print(f"\n\t\t\tQuery embedding: {query_embeddings[:5]}... (length: {len(query_embeddings)})")
        
        results = self.collection.query( 
                                        query_embeddings = query_embeddings,
                                        n_results = top_k
                                        )
        print(f"\n\n\nRetrieved {len(results.get('documents',[[]])[0])} documents from ChromaDB for query '{query}'\nResults: {json.dumps(results, indent=2)}")
        documents = results.get("documents",[[]])[0]
        metadatas = results.get("metadatas",[[]])[0]
        
        return [
            {"content": doc,
             "source": meta.get("source","Unknown"),
             "page": meta.get("page",1)
            } for doc, meta in zip(documents, metadatas)
        ]

    def generate_responses(self, query:str, chunks:list):
        if not chunks:
            return "No information on this topic"
        
        # Build strictly partitioned context blocks
        context_blocks = []
        for c in chunks:
            src = c['source']
            content = c['content']
            
            # Map exact responsibilities to exact sources based on your real documents
            if "Resume" in src or "Data Engineer" in src:
                doc_desc = "OFFICIAL RESUME (Verified employment history: Emerging Engineering, UNDP, GIMAC-BEAC)"
            elif "Offer Letter" in src :
                doc_desc = "OFFER LETTER (Sagaci Research - Senior Data Analyst role starting May 2026)"
            elif "Internship" in src:
                doc_desc = "INTERNSHIP AGREEMENT (Harbour Space Institute - Data Science Internship)"
            else:
                #doc_desc = f"JOB DESCRIPTION / INTERVIEW TARGET ({src}) - NOT PAST WORK"
                doc_desc = f"Skills ({src})"
                
            context_blocks.append(f"[{doc_desc}] (Page {c['page']})\n{content}")

        context_str = "\n\n".join(context_blocks)
        
        prompt = f"""
        You are Brenda's warm, clever, and impeccably professional AI career assistant. Speak about Brenda in the 3rd person with high fidelity.

        CRITICAL STRUCTURAL RULES:
        1. NEVER output template placeholders like [Project Name], [GitHub Username], or [Hackathon Name]. If specific repository names are not in the context, name the explicit tools (e.g., Python, ChromaDB, Streamlit, Ollama, Tesseract OCR) or state: "I cannot find specific repository details in the loaded documents."
        2. GITHUB PROJECTS: When asked about side projects or GitHub, discuss her actual repos (e.g., local RAG pipelines, OCR tools, data visualization apps). DO NOT default to ICT University unless discussing her degree.
        3. NO HALLUCINATED METRICS: Use only exact facts provided in the text context.

        Structure your analysis:
        - **Where:** GitHub / Personal Development environment.
        - **How:** Specific frameworks, libraries, and languages used in her repos.
        - **Who:** Self-directed projects or open-source community contributions.
        - **Why:** Skill acquisition, continuous learning, and mastering modern data engineering tools.

        Context:
        {context_str}

        User Question: {query}

        Answer:
        """
        # 1. Try local Ollama first (for when you run it on your laptop)
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                OLLAMA_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "No response generated.")
        except Exception:
            # 2. Cloud Fallback: Use Hugging Face Transformers locally in the container
            try:
                from transformers import pipeline
                import torch
                
                # Load a lightweight, fast model suitable for CPU cloud environments
                generator = pipeline(
                    "text-generation", 
                    model="HuggingFaceTB/SmolLM-1.35B-Instruct", 
                    torch_dtype=torch.float32,
                    device=-1 # CPU mode
                )
                
                outputs = generator(
                    prompt, 
                    max_new_tokens=300, 
                    do_sample=True, 
                    temperature=0.2,
                    top_p=0.9
                )
                
                generated_text = outputs[0]["generated_text"]
                # Strip prompt echo if present
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                return generated_text
                
            except Exception as hf_err:
                return f"Error generating response on cloud backend: {hf_err}"


if __name__ == "__main__":
    # Test execution from command line
    engine = LocalRAGEngine()
    
    # Change this query to match whatever documents you ingested!
    test_query = "Who are we talking about?"
    
    # 1. Retrieve relevant chunks
    matched_chunks = engine.retrieve(test_query)
    
    print(f"\nFound {len(matched_chunks)} matching chunks:")
    for i, chunk in enumerate(matched_chunks, 1):
        print(f"  [{i}] Source: {chunk['source']} (Page {chunk['page']}) -> {chunk['content'][:80]}...")

    # 2. Generate final answer
    answer = engine.generate_responses(test_query, matched_chunks)
    
    print("\n-----------------------------------------")
    print("FINAL AI RESPONSE:")
    print("-----------------------------------------")
    print(answer)