import json
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import torch

# Load environment variables
load_dotenv()

# Configuration
QDRANT_PATH = "./qdrant_data" # Local persistent storage
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
COLLECTION_NAME = "physics_textbook"

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_and_process_data(file_input: Any) -> List[Dict[str, Any]]:
    if isinstance(file_input, str):
        with open(file_input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Assume file-like object (e.g. UploadedFile)
        data = json.load(file_input)
    
    processed_items = []
    # If the root is a list, iterate; if dict (single chapter), wrap in list
    chapters = data if isinstance(data, list) else [data]
    
    for chapter in chapters:
        chapter_title = chapter.get("chapter_title", "Unknown Chapter")
        
        for section in chapter.get("sections", []):
            section_number = section.get("section_number", "")
            section_title = section.get("section_title", "")
            
            # Process main section content
            if section.get("content"):
                content_text = "\n\n".join(section["content"])
                if content_text.strip():
                    processed_items.append({
                        "text": content_text,
                        "metadata": {
                            "chapter_title": chapter_title,
                            "section_number": section_number,
                            "section_title": section_title,
                            "subsection_number": None,
                            "subsection_title": None,
                            "hierarchy_level": "section",
                            "has_equations": False # Placeholder
                        }
                    })
            
            # Process subsections
            for subsection in section.get("subsections", []):
                subsection_number = subsection.get("subsection_number", "")
                subsection_title = subsection.get("subsection_title", "")
                
                if subsection.get("content"):
                    content_text = "\n\n".join(subsection["content"])
                    # Context injection
                    full_text = f"Section {section_number}: {section_title}\n\n{content_text}"
                    
                    if content_text.strip():
                        processed_items.append({
                            "text": full_text,
                            "metadata": {
                                "chapter_title": chapter_title,
                                "section_number": section_number,
                                "section_title": section_title,
                                "subsection_number": subsection_number,
                                "subsection_title": subsection_title,
                                "hierarchy_level": "subsection",
                                "has_equations": False # Placeholder
                            }
                        })
                    
    return processed_items

def ingest_data(file_input: Any, progress_callback=None, status_callback=None):
    """
    Ingests data from a file input (path or file object) into Qdrant.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Initialize Embedding Model
    log(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} on {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device, 'trust_remote_code': True}
    )
    
    # Initialize Semantic Chunker
    log("Initializing Semantic Chunker...")
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=100
    )
    
    # Load Data
    log("Loading data...")
    try:
        raw_items = load_and_process_data(file_input)
        log(f"Loaded {len(raw_items)} broad sections/subsections.")
    except Exception as e:
        log(f"Error loading data: {e}")
        return
    
    # Chunk Data
    log("Chunking data...")
    documents = []
    
    # Update progress for chunking (allocating 20% of progress)
    total_raw = len(raw_items)
    
    for idx, item in enumerate(raw_items):
        chunks = chunker.create_documents([item["text"]])
        for i, chunk in enumerate(chunks):
            # Enrich metadata
            chunk_metadata = item["metadata"].copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["estimated_tokens"] = len(chunk.page_content) // 4
            chunk_metadata["has_equations"] = "$" in chunk.page_content or "\\" in chunk.page_content
            
            documents.append(Document(page_content=chunk.page_content, metadata=chunk_metadata))
        
        if progress_callback:
            # First 20% for chunking
            progress_callback(int((idx + 1) / total_raw * 20))
            
    log(f"Generated {len(documents)} semantic chunks.")
    
    # Initialize Qdrant
    log(f"Initializing Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Re-create collection
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) # Nomic-embed-text-v1.5 is 768d
    )
    
    # Upload points
    log("Generating embeddings and uploading to Qdrant...")
    
    # Generate embeddings for all chunks in batches to manage memory
    batch_size = 32
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        
        current_batch_num = i // batch_size + 1
        log(f"Embedding batch {current_batch_num}/{total_batches}...")
        
        embeddings_list = embeddings.embed_documents(batch_texts)
        
        current_points = []
        for j, (doc, vector) in enumerate(zip(batch_docs, embeddings_list)):
            point_id = i + j
            current_points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            ))
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=current_points
        )
        
        if progress_callback:
            # Remaining 80% for embedding/uploading
            # 20 + (current_batch / total_batches * 80)
            progress = 20 + int(current_batch_num / total_batches * 80)
            progress_callback(min(progress, 100))
            
    log("Ingestion complete!")

if __name__ == "__main__":
    if os.path.exists("chapter_structure.json"):
        ingest_data("chapter_structure.json")
    else:
        print("chapter_structure.json not found.")
