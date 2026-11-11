""" Serve the app using streamlit. This is the main entry point for the application. It will:
    - Convert new images to text using image_converter
    - Scrape GitHub profile and repositories using github_scrapper
    - Load documents from the data directory
    - Chunk documents into semantically similar chunks using the SemanticSplitter class
    - Generate embeddings for the chunks using a local embedding model (SentenceTransformer)
    - Store the chunks and embeddings in a local ChromaDB database
    - Provide a Streamlit interface for users to input queries and retrieve relevant chunks from the database
"""

import os
import sys
import streamlit as st
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="About Me",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("About Me")
st.caption("Query about my experiences... The application uses a knowledge base of my academic logs and work experiences to answer questions.")

@st.cache_resource(show_spinner="Loading Query Engine & Vector Database...")
def get_rag_engine():
    from rag_engine import LocalRAGEngine
    engine = LocalRAGEngine()
    
    # If the cloud instance has an empty collection, ingest the committed text files automatically ONCE
    if not engine.collection or engine.collection.count() == 0:
        #st.toast("⚡ Initializing knowledge base on cloud instance...")
        try:
            from ingestion import read_documents, SemanticSplitter, store_chunks_in_chromadb, MODEL_NAME
            import uuid
            loaded_text = read_documents()
            if loaded_text:
                chunker = SemanticSplitter(model_name=MODEL_NAME, distance_threshold_percentile=60.0)
                chunks = []
                for doc in loaded_text:
                    doc_chunks = chunker.chunk_text(doc["text"])
                    for chunk in doc_chunks:
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "text": chunk["chunk"],
                            "metadata": {
                                "source": doc["source"],
                                "page": doc["page"],
                                "type": doc["type"]
                            }
                        })
                store_chunks_in_chromadb(chunks)
                # Re-fetch the collection after creation
                engine.collection = engine.client.get_collection(name="academic_logs")
        except Exception as e:
            #st.error(f"Auto-ingestion failed: {e}")
            pass
            
    return engine

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Error initializing RAG engine: {e}")
    st.stop()

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Number of retrieved chunks", 1, 10, 3, 1)
    
    st.divider()
    st.markdown("### Knowledge base information")
    if engine.collection:
        doc_count = engine.collection.count()
        st.metric("Ingested chunks", doc_count)
    else:
        st.warning("Collection not found.")
    
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    st.markdown("### 🧠 Mind Map Thought Prompts")
    if st.button("🗺️ Career Map"):
        st.session_state.active_prompt = "Give me a summary of Brenda's career evolution using the Where, How, Who, and Why framework."
        st.rerun()
    if st.button("🚀 Prove Growth & Impact"):
        st.session_state.active_prompt = "What measurable transformations and system reliability improvements has Brenda delivered across her roles?"
        st.rerun()
    if st.button("⏱️ Most recent experience"):
        st.session_state.active_prompt = "What are the most recent and measurable transformations and improvements Brenda has delivered across her most recent role?"
        st.rerun()
    if st.button("⏱️ Earliest experience"):
        st.session_state.active_prompt = "What are the earliest and measurable transformations and improvements Brenda had delivered across her earliest role?"
        st.rerun()
    if st.button("🚀 Side projects"):
        st.session_state.active_prompt = "Describe Brenda's side projects from github and their impact on her career growth and skill development."
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me about my work experiences...")

if "active_prompt" in st.session_state and st.session_state.active_prompt:
    prompt = st.session_state.active_prompt
    st.session_state.active_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            retrieved_chunks = engine.retrieve(prompt, top_k=top_k)
            response = engine.generate_responses(prompt, retrieved_chunks)
            st.markdown(response)
                        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response, 
        "sources": retrieved_chunks})
    st.rerun()