""" Serve the app using streamlit. This is the main entry point for the application. It will:
    - Load documents from the data directory
    - Chunk documents into semantically similar chunks using the SemanticSplitter class
    - Generate embeddings for the chunks using a local embedding model (SentenceTransformer)
    - Store the chunks and embeddings in a local ChromaDB database
    - Provide a Streamlit interface for users to input queries and retrieve relevant chunks from the database
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # -- Make sure the src directory is in the path for imports

# Page configurations
st.set_page_config(
    page_title="About Me",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("About Me")
st.caption(" Query about my experiences... The application uses a knowledge base of my academic logs and work experiences to answer questions. It uses a local LLM (LLaMA 3) and a local vector database (ChromaDB) for retrieval augmented generation (RAG).")

@st.cache_resource(show_spinner="Loading Query Engine...")
def get_rag_engine():
    from rag_engine import LocalRAGEngine
    return LocalRAGEngine()

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Error initializing RAG engine: {e}")
    st.stop()
    
# Automatic ingestion check for cloud deployments (Placed AFTER engine is initialized)
if not engine.collection or engine.collection.count() == 0:
    st.info("⚡ Initializing knowledge base on first launch...")
    try:
        from ingestion import run_ingestion  # Ensure ingestion script has a callable function
        run_ingestion()
        st.success("Knowledge base successfully loaded!")
        st.rerun()
    except Exception as e:
        st.error(f"Failed auto-ingestion: {e}")

# Initialize chat session state first so variables exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Number of retrieved chunks", 
                      min_value=1, 
                      max_value=10, 
                      value=3, 
                      step=1)
    
    st.divider()
    st.markdown("### Knowledge base information")
    if engine.collection:
        doc_count = engine.collection.count()
        st.metric("Ingested chunks", doc_count)
    else:
        st.warning("No collection found. Please run the ingestion script to load documents into the database.")
    
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
        st.session_state.active_prompt = "Describe Brenda's side projects from github and their impact on her career growth and skill development.In addition, include a summary of the most relevant skills and technologies she has used in her side projects. Then add a list of the repository urls and description summaries for each side project."
        st.rerun()

# Display past chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Determine if a prompt came from the chat input box OR a sidebar preset button
prompt = st.chat_input("Ask me about my work experiences...")

if "active_prompt" in st.session_state and st.session_state.active_prompt:
    prompt = st.session_state.active_prompt
    st.session_state.active_prompt = None  # Clear it so it doesn't loop

# If a prompt is triggered (either by typing or clicking a button)
if prompt:
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Retrieve relevant chunks from the database
            retrieved_chunks = engine.retrieve(prompt, top_k=top_k)
            
            # Generate response using the retrieved chunks
            response = engine.generate_responses(prompt, retrieved_chunks)
            
            # Display the response
            st.markdown(response)
                        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response, 
        "sources": retrieved_chunks})
    st.rerun()