from src.core.agent import FEPAgent
import shutil
import os
import random

def test_memory_integration():
    print("🧠 Testing V2 Memory Persistence...")
    
    # Clean up old test db
    if os.path.exists("./test_memory_db"):
        shutil.rmtree("./test_memory_db")
    
    # Initialize Agent with local work_dir
    # Note: VectorDB path in agent is hardcoded to use work_dir/friston_memory
    # But in the code (agent.py:28):
    # db_path = os.path.join(work_dir if work_dir != "/workspace" else ".", "friston_memory")
    
    # We'll use local dir
    agent = FEPAgent(work_dir="./test_workspace")
    
    # Override memory path for testing to avoid polluting main DB
    # (Though simpler is just to let it use ./test_workspace/friston_memory)
    
    print(f"Agent Memory Type: {type(agent.memory)}")
    
    # Create fake vector (10k dim)
    # We use random vector to simulate HDC
    vec = [random.random() for _ in range(10000)]
    
    code_payload = "print('Hello Memory')"
    metadata = {"test": True}
    
    print("💾 Saving memory...")
    agent.memory.save_memory(vec, code_payload, metadata)
    
    print("🔍 Retrieving memory (Exact match)...")
    result = agent.memory.retrieve_memory(vec, threshold=0.99)
    
    if result:
        print(f"✅ Exact match found! Score: {result['score']}")
        print(f"   Code: {result['code']}")
    else:
        print("❌ Exact match failed!")
        
    # Test fuzzy retrieval
    print("🔍 Retrieving memory (Noisy)...")
    # Add noise
    noisy_vec = [v + random.uniform(-0.1, 0.1) for v in vec]
    # Re-normalize/clip? Chroma handles arbitrary vectors but cosine sim depends on angle.
    # HDC vectors usually binary or -1/1, but float is fine.
    
    result = agent.memory.retrieve_memory(noisy_vec, threshold=0.8)
    
    if result:
        print(f"✅ Fuzzy match found! Score: {result['score']}")
    else:
        print("❌ Fuzzy match failed (might be expected if noise is too high)")

if __name__ == "__main__":
    test_memory_integration()
