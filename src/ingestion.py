""" Document: Ingestion Script ETL for Academic Logs into ChromaDB

    This script:
        - Reads documents from the specifiied folder. If there are no documents, it will create the folder and prompt the user to add documents. If the folder contains files which are not pdfs or txt files, it will skip those files and print a message to the user.
        - Chunks the documents into smaller pieces for semantic context.
        - Generates embeddings for the chunks using a local embedding model ... i'm using SentenceTransformer
        - Stores the chunks and their embeddings in ChromaDB for retrieval augmented generation (RAG) queries.
    I'm using ChromaDB over DuckDB because ChromaDB is optimized for vector storage and retrieval, which is essential for RAG queries. DuckDB is more suited for structured data and analytics, while ChromaDB provides efficient similarity search capabilities for embeddings.
"""

import os #-- For file and directory operations
import re #-- For formatting
import json #-- For saving and loading data in JSON format
import string #-- For string operations
from pathlib import Path #-- For creating directories

from pypdf import PdfReader #-- For reading PDF files
import uuid #-- For generating unique identifiers for documents

from sentence_transformers import SentenceTransformer #-- For generating embeddings (inputting semantic context)
import chromadb #-- Vector database and retrieval

from semantic_splitter import SemanticSplitter #-- For splitting documents into semantically similar chunks

DATA_DIR = "./data"  # Directory to read documents from
DB_DIR = "./data/chroma_db"  # Directory to store the ChromaDB database
COLLECTION_NAME = 'academic_logs'
MODEL_NAME = "all-MiniLM-L6-v2"

def read_documents(data_dir:str=DATA_DIR) -> list:
    """
    Arguments:
        data_dir (str): The path to the folder containing documents to read.
    PURPOSE:
        - Read documents from the specified folder. If there are no documents, it will create the folder and prompt the user to add documents. If the folder contains files which are not pdfs or txt files, it will skip those files and print a message to the user.
        - Load the documents and extract their content for further processing.
    """

    # Create the data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # List all files in the data directory
    files = os.listdir(data_dir)
    
    if not files:
        print(f"No documents found in {data_dir}. Please add PDF or TXT files.")
        return []

    valid_extensions = (".pdf", ".txt")
    documents = []

    # Iterate through the files and filter for valid document types
    for file_name in files:
        if file_name.lower().endswith(valid_extensions):
            file_path = os.path.join(data_dir, file_name)
            documents.append(file_path)
        else:
            print(f"\t\tSkipping unsupported item: {file_name} | Run the image_converter.py script to convert images to text or pdf.")
    
    # Load the content of the valid documents
    loaded_text = []
    
    for doc_path in documents:
        doc = doc_path.split("/")[-1]  # Extract the document name from the path
        ext = doc_path.split(".")[-1]  # Extract the file extension
        
        try:
            if ext == "pdf":
                loader = PdfReader(doc_path)
                for idx, page in enumerate(loader.pages):
                    text = page.extract_text()
                    if text:
                        cleaned_text = re.sub(r'\s+', ' ', text).strip()
                        doc_type = "resume" if "Brenda" in doc else "description"
                        loaded_text.append({"id":str(uuid.uuid4()), "text": cleaned_text, "source": doc, "page": idx + 1, "type": doc_type})
                    else:
                        print(f"\t\tNo text found on page {idx + 1} of {doc}.")
            elif ext == "txt":
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    cleaned_text = re.sub(r'\s+', ' ', text).strip()
                    loaded_text.append({"id":str(uuid.uuid4()), "text": cleaned_text, "source": doc, "page": 1, "type": "description"})
            else:
                print(f"\t\tUnsupported file type: {doc}. Only PDF and TXT files are supported.")
        except Exception as e:
            print(f"\t\tError reading {doc}: {e}")
    
    #print(f"\n\n\tLoaded {len(loaded_text)} documents from {data_dir}. \n {loaded_text[:3]} \n\n {type(loaded_text)}")
    return loaded_text

def chunk_documents(documents: list, chunk_size: int = 100) -> list:
    """
    Arguments:
        documents (list): List of loaded documents to be chunked.
        chunk_size (int): The maximum number of characters per chunk.
    PURPOSE:
        - Chunk the loaded documents into smaller pieces for semantic context.
        - Each chunk will have a maximum size defined by chunk_size.
    """
    if not documents:
        print("No documents to chunk. Please load documents first.")
        return []
    
    print(f"\n\nSample document before chunking: {json.dumps(documents[:3], indent=3, default=str, sort_keys=False)} \n\n {type(documents)} \nChunking documents into smaller pieces with a maximum size of {chunk_size} characters each ...\n\ń")
    
    chunks = []
        
    # Split the text into chunks based on the specified chunk size, if thechunk ends in the middle of a word, it will continue to the next chunkkeep the word intact
    for doc in documents:
            text = doc["text"] 
            source = doc["source"]
            page = doc["page"]
            
            text_length = len(text)
            start = 0
            
            while start < text_length:
                # Determine the ideal end position
                end = min(start + chunk_size, text_length)
                
                # If we aren't at the very end of the text, look for the last space 
                # within this window so we don't chop a word in half.
                if end < text_length:
                    space_idx = text.rfind(' ', start, end)
                    # If a space is found within the chunk window, cut at that space
                    if space_idx != -1 and space_idx > start:
                        end = space_idx + 1
                
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunks.append({
                        "id": str(uuid.uuid4()), 
                        "text": chunk_text, 
                        "metadata": {
                            "source": source, 
                            "page": page
                        }
                    })
                
                # Move the start pointer forward. 
                # If no space was found (e.g., an abnormally long word), force-advance by chunk_size to avoid infinite loops.
                if end < text_length and text.rfind(' ', start, end) != -1:
                    start = text.rfind(' ', start, end) + 1
                else:
                    start = end
    
    print(f"\n\n\tChunked {len(documents)} documents into {len(chunks)} chunks.\n")
    return chunks

def store_chunks_in_chromadb(chunks: list, embedding_model_name: str = MODEL_NAME) -> None:
    """ 
    Arguments:
        chunks (list): List of chunks to be stored
        embedding_model_name (str): what model we're using to build context for the data chunks
    PURPOSE:
        - Store document chunks together with semantic meanings (cosine differences) in ChromaDB
    """
    
    if not chunks:
        print(f"No chunks to store in DB")
        return
    
    #print(f"Loading {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    #print(f"Initializing ChromaDB and creating collection '{COLLECTION_NAME}' ...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Extract fields for chroma ingestion and construct correct metadata dictionaries
    ids = [chunk['id'] for chunk in chunks]
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    #print(f"Generating embeddings for {len(chunks)} chunks locally ...")
    embeddings = model.encode(texts).tolist()  # Generate embeddings for all chunks at once
    
    #print(f"Adding chunks to ChromaDB collection '{COLLECTION_NAME}'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )
    #print(f"Stored {collection.count()} chunks in ChromaDB successfully.")

if __name__ == "__main__":
    loaded_text = read_documents()
    if loaded_text:
        print(f"Loaded documents ...")
        # chunks = chunk_documents(loaded_text)
        
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
        
        print(f"Storing chunks in ChromaDB ..., {json.dumps(chunks[:3], indent=3, default=str, sort_keys=False)} \n\n {type(chunks)}")
        
        store_chunks_in_chromadb(chunks)
        print(f"Completed ingestion ...")
    else:
        print(f"No documents to load.")
        
    