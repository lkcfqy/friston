import torch
import torch.nn.functional as F
from src.perception.parser import ASTParser
from src.perception.hdc import HDCEncoder
from src.memory.mhn import HopfieldMemory

def main():
    print("=== Phase 2: Neuro-Symbolic Memory (MHN) Demo ===")
    
    # 1. Initialize Components
    parser = ASTParser()
    encoder = HDCEncoder(dim=10000)
    memory = HopfieldMemory(dim=10000, beta=50.0) # High beta for precise recall
    
    # 2. Define "Canonical" Functions ( The Knowledge Base )
    # These represent correct coding patterns the agent "knows"
    library = {
        "bubble_sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
        "binary_search": """
def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1
""",
        "factorial": """
def factorial(n):
    if n == 1: 
        return 1
    else: 
        return n * factorial(n-1)
"""
    }
    
    # 3. Learn Patterns
    print(f"\n[Learning] Storing {len(library)} patterns in associative memory...")
    vectors = []
    keys = list(library.keys())
    
    for name in keys:
        code = library[name]
        tree = parser.parse(code)
        func_node = parser.get_functions(tree)[0]
        vec = encoder.encode_ast(func_node)
        vectors.append(vec)
        
    pattern_tensor = torch.stack(vectors)
    memory.learn(pattern_tensor)
    
    # 4. Retrieval Test with Noisy/Buggy Query
    print("\n[Query] Simulating a 'forgotten' or 'buggy' implementation of bubble sort...")
    
    # A modified bubble sort (renamed vars + slight structural change/bug)
    buggy_code = """
def sort_algo(data):
    # Forgot the lengths
    for x in range(100):
        for y in range(0, 100):
            if data[y] > data[y+1]:
                # Swapping
                temp = data[y]
                data[y] = data[y+1]
                data[y+1] = temp
    return data
"""
    print(f"Query Code Snippet:\n{buggy_code.strip()[:100]}...")
    
    tree_buggy = parser.parse(buggy_code)
    func_buggy = parser.get_functions(tree_buggy)[0]
    query_vec = encoder.encode_ast(func_buggy)
    
    # Retrieve from Memory
    print("\n[Retrieval] MHN Converging...")
    recovered_vec = memory.retrieve(query_vec)
    
    # 5. Verify which pattern was recalled
    print("\n[Analysis] Matching recovered pattern against library:")
    similarities = F.cosine_similarity(recovered_vec.unsqueeze(0), pattern_tensor)
    
    best_idx = torch.argmax(similarities).item()
    best_sim = similarities[best_idx].item()
    recalled_name = keys[best_idx]
    
    for i, name in enumerate(keys):
        print(f" - Similarity with '{name}': {similarities[i]:.4f}")
        
    print(f"\nResult: The noisy query triggered recall of '{recalled_name}' (Sim: {best_sim:.4f})")
    
    if recalled_name == "bubble_sort":
        print("✅ SUCCESS: Correctly identified the algorithm despite code changes!")
    else:
        print("❌ FAILURE: Wrong association.")

if __name__ == "__main__":
    main()
