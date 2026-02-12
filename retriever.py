import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
import torch

# Load environment variables
load_dotenv()

# Configuration
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "physics_textbook"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

device = "cuda" if torch.cuda.is_available() else "cpu"

class PhysicsRetriever:
    def __init__(self):
        print(f"Initializing Retriever on {device}...")
        
        # Initialize Qdrant Client
        self.client = QdrantClient(path=QDRANT_PATH)
        
        # Initialize Embedding Model
        print("Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device, 'trust_remote_code': True}
        )
        
        # Initialize Reranker
        print("Loading Reranker Model...")
        self.reranker = CrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            device=device,
            trust_remote_code=True
        )
        
        
    def check_connection(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Connection check failed: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(COLLECTION_NAME)
            return {
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status
            }
        except Exception:
            return {'vectors_count': 0, 'status': 'unknown'}

    def retrieve(self, query: str, top_k: int = 20) -> List[Any]:
        """
        Stage 1: Dense Retrieval from Qdrant
        """
        query_vector = self.embeddings.embed_query(query)
        
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        if COLLECTION_NAME not in collection_names:
            print(f"Warning: Collection '{COLLECTION_NAME}' does not exist.")
            return []

        search_results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.4 # Minimum relevance threshold
        )
        
        return search_results
    
    def rerank(self, query: str, initial_results: List[Any], top_k: int = 6) -> List[Any]:
        """
        Stage 2: Cross-Encoder Reranking
        """
        if not initial_results:
            return []
            
        # Prepare pairs for reranking: (query, document_text)
        pairs = [[query, result.payload['text']] for result in initial_results]
        
        # Predict scores
        scores = self.reranker.predict(pairs)
        
        # Combine results with scores
        for result, score in zip(initial_results, scores):
            # Add rerank score to payload for debugging/display
            result.payload['rerank_score'] = float(score)
            
        # Sort by rerank score descending
        # create a list of (score, result) tuples
        scored_results = sorted(
            zip(scores, initial_results),
            key=lambda x: x[0],
            reverse=True
        )
        
        # Select top K
        final_results = [result for score, result in scored_results[:top_k]]
        
        return final_results
        
    def search(self, query: str) -> List[Any]:
        """
        Full pipeline: Retrieve -> Rerank
        """
        initial_results = self.retrieve(query)
        final_results = self.rerank(query, initial_results)
        return final_results

# Test block
if __name__ == "__main__":
    retriever = PhysicsRetriever()
    test_query = "What is a rigid body? Give examples."
    print(f"\nTesting Query: {test_query}")
    
    results = retriever.search(test_query)
    
    print(f"\nTop {len(results)} Results:")
    for i, res in enumerate(results, 1):
        meta = res.payload['metadata']
        print(f"\n{i}. Section {meta['section_number']} ({meta.get('subsection_title') or meta['section_title']})")
        print(f"   Score: {res.payload.get('rerank_score', 0):.4f}")
        print(f"   Chunk Index: {meta.get('chunk_index')}")
        print(f"   Preview: {res.payload['text'][:150].replace(chr(10), ' ')}...")
