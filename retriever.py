import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector, Filter, FieldCondition, MatchValue
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
    def __init__(self, client):
        print(f"[DEBUG] Retriever.__init__ called. Device: {device}")
        print(f"[DEBUG] Received Qdrant client: {client}")
        
        # Use passed Qdrant Client
        self.client = client
        try:
            collections = self.client.get_collections()
            print(f"[DEBUG] Verified Qdrant connection. Collections: {collections}")
        except Exception as e:
            print(f"[ERROR] Failed to verify Qdrant connection in Retriever: {e}")
            raise e
        
        # Initialize Embedding Model
        print("Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device, 'trust_remote_code': True}
        )
        
        # Initialize Reranker
        # print("Loading Reranker Model...")
        # self.reranker = CrossEncoder(
        #     model_name=RERANKER_MODEL_NAME,
        #     device=device,
        #     trust_remote_code=True
        # )

    def build_sparse_query(self, text: str) -> SparseVector:
        """
        Build a simple sparse vector from query text.
        Uses hash-based indexing for query-time efficiency.
        """
        tokens = text.lower().split()
        token_counts = {}
        
        # Count token frequencies
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Build sparse vector using hash-based indices
        indices = []
        values = []
        
        for token, count in token_counts.items():
            # Use hash for index (modulo to keep reasonable range)
            idx = abs(hash(token)) % 100000
            indices.append(idx)
            values.append(float(count))
        
        return SparseVector(
            indices=indices,
            values=values
        )


    def check_connection(self) -> bool:
        # Simplified check
        return self.client is not None

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(COLLECTION_NAME)
            return {
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status
            }
        except Exception:
            return {'vectors_count': 0, 'status': 'unknown'}

    def retrieve(self, query: str, top_k: int = 20, chapter_filter: str = None) -> List[Any]:
        """
        Hybrid retrieval using query_points API with RRF fusion
        Combines dense semantic search + sparse BM25 keyword search
        """
        # Generate dense embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Generate sparse vector
        sparse_query = self.build_sparse_query(query)

        # Build filter
        query_filter = None
        if chapter_filter and chapter_filter != ["All Chapters"]:
            if isinstance(chapter_filter, list):
                # Handle empty list - no chapters selected means return nothing
                if len(chapter_filter) == 0:
                    return []
                query_filter = Filter(
                    should=[
                        FieldCondition(
                            key="metadata.chapter_id",
                            match=MatchValue(value=cid)
                        )
                        for cid in chapter_filter
                    ]
                )
            else:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.chapter_id",
                            match=MatchValue(value=chapter_filter)
                        )
                    ]
                )
        
        # Hybrid search with RRF fusion
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # Dense vector search
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=top_k
                ),
                # Sparse vector search
                models.Prefetch(
                    query=sparse_query,
                    using="bm25",
                    limit=top_k
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=0.3
        )
        
        # query_points returns QueryResponse, extract points
        return response.points

    def rerank(self, query: str, initial_results: List[Any], top_k: int = 6) -> List[Any]:
        """
        Stage 2: Cross-Encoder Reranking (DISABLED)
        """
        # Simply return top K from initial results to save time
        return initial_results[:top_k]
        
    def search(self, query: str, chapter_filter: str = None) -> List[Any]:
        initial_results = self.retrieve(query, chapter_filter=chapter_filter)
        final_results = self.rerank(query, initial_results)
        return final_results


    
    def close(self):
        """
        Explicitly closes the Qdrant client connection.
        """
        if self.client:
            print("Closing Qdrant client...")
            self.client.close()
            self.client = None

# Test block
if __name__ == "__main__":
    from qdrant_client import QdrantClient
    client = QdrantClient(path="./qdrant_data")
    retriever = PhysicsRetriever(client)
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
    
    retriever.close()
