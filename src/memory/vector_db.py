import chromadb
from chromadb.config import Settings
import uuid
import time
from typing import Dict, Any, List, Optional
import json

class VectorDB:
    """
    Persistent Memory utilizing ChromaDB.
    Acts as the 'Hippocampus' for the agent, storing validated Code-Vector pairs.
    """
    def __init__(self, persist_path: str = "./memory_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Create or get collection. 
        # We use cosine distance which is standard for high-dimensional vectors.
        self.collection = self.client.get_or_create_collection(
            name="friston_memory",
            metadata={"hnsw:space": "cosine"} 
        )
        print(f"🧠 Connected to Hippocampus (ChromaDB) at {persist_path}")
        print(f"   Current memory size: {self.collection.count()} items")

    def save_memory(self, 
                    vector: List[float], 
                    code: str, 
                    metadata: Dict[str, Any]):
        """
        Consolidate a memory trace.
        
        Args:
            vector: The HDC vector (list of floats).
            code: The validated code snippet (Payload).
            metadata: Context tags (e.g., {'type': 'fix', 'success': True}).
        """
        # Ensure metadata contains timestamp
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
            
        # Chroma requires IDs. We use UUIDs.
        mem_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[mem_id],
            embeddings=[vector],
            documents=[code], # We store the code in the 'document' field
            metadatas=[metadata]
        )
        # print(f"💾 Memory consolidated: {mem_id[:8]}...")

    def retrieve_memory(self, 
                        query_vector: List[float], 
                        threshold: float = 0.85, 
                        n_results: int = 1) -> Optional[Dict[str, Any]]:
        """
        Recall a memory based on vector similarity.
        
        Args:
            query_vector: The HDC vector of the current problem/code.
            threshold: Similarity threshold (0.0 to 1.0). 
                       Chroma returns 'distance'. For cosine, distance = 1 - similarity.
                       So max_distance = 1 - threshold.
            n_results: How many matches to check.
            
        Returns:
            Dict containing 'code' and 'metadata', or None if no match found.
        """
        # 1 - threshold because Chroma cosine distance is (1 - cos_sim)
        # E.g. threshold 0.85 means distance must be <= 0.15
        max_distance = 1.0 - threshold
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        # Check if we got any results
        if not results['ids'] or len(results['ids'][0]) == 0:
            return None
            
        distance = results['distances'][0][0]
        
        if distance <= max_distance:
            # Match found!
            return {
                "id": results['ids'][0][0],
                "code": results['documents'][0][0], #Payload
                "metadata": results['metadatas'][0][0],
                "score": 1.0 - distance
            }
        
        return None
