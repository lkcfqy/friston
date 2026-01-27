import torch
from src.perception.parser import ASTParser
from src.perception.hdc import HDCEncoder

def main():
    print("=== Phase 1: Neuro-Symbolic Perception Demo ===")
    
    parser = ASTParser()
    encoder = HDCEncoder(dim=10000, device="cpu")

    # Code snippets
    code_base = """
def calculate_area(radius):
    pi = 3.14
    return pi * radius * radius
"""
    
    code_renamed = """
def compute_circle_size(r):
    p = 3.14
    return p * r * r
"""

    code_diff = """
def greet_user(name):
    print("Hello " + name)
    return True
"""

    print("\n[1] Parsing Code Snippets...")
    t1 = parser.parse(code_base)
    t2 = parser.parse(code_renamed)
    t3 = parser.parse(code_diff)

    f1 = parser.get_functions(t1)[0]
    f2 = parser.get_functions(t2)[0]
    f3 = parser.get_functions(t3)[0]

    print(f"Function 1: {parser.get_code_from_node(f1).splitlines()[0]} ...")
    print(f"Function 2: {parser.get_code_from_node(f2).splitlines()[0]} ... (Structurally Identical)")
    print(f"Function 3: {parser.get_code_from_node(f3).splitlines()[0]} ... (Different Logic)")

    print("\n[2] Generating Neuro-Symbolic Vectors (HDC)...")
    v1 = encoder.encode_ast(f1)
    v2 = encoder.encode_ast(f2)
    v3 = encoder.encode_ast(f3)

    print(f"Vector Dimensions: {v1.shape}")

    print("\n[3] Computing Structural Similarity (Cosine)...")
    sim_1_2 = encoder.similarity(v1, v2)
    sim_1_3 = encoder.similarity(v1, v3)

    print(f"Similarity (Base vs Renamed): {sim_1_2:.4f}")
    print(f"Similarity (Base vs Different): {sim_1_3:.4f}")

    if sim_1_2 > sim_1_3:
        print("\n✅ SUCCESS: Structural similarity detected despite variable renaming!")
    else:
        print("\n❌ FAILURE: Failed to distinguish structure.")

if __name__ == "__main__":
    main()
