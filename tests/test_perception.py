import pytest
import torch
from src.perception.parser import ASTParser
from src.perception.hdc import HDCEncoder

@pytest.fixture
def parser():
    return ASTParser()

@pytest.fixture
def encoder():
    return HDCEncoder(dim=10000)

def test_parser_basic(parser):
    code = "def foo(x): return x + 1"
    tree = parser.parse(code)
    assert tree.root_node.type == "module"
    functions = parser.get_functions(tree)
    assert len(functions) == 1
    assert parser.get_code_from_node(functions[0]).startswith("def foo")

def test_hdc_determinism(encoder, parser):
    code = "def foo(x): return x + 1"
    tree = parser.parse(code)
    v1 = encoder.encode_ast(tree.root_node)
    v2 = encoder.encode_ast(tree.root_node)
    assert torch.equal(v1, v2)
    assert encoder.similarity(v1, v2) > 0.99

def test_structural_similarity(encoder, parser):
    # Case 1: Same logic, different variable names
    code1 = """
def add_numbers(a, b):
    return a + b
"""
    code2 = """
def sum_values(x, y):
    return x + y
"""
    
    # Case 2: Different logic
    code3 = """
def multiply_numbers(a, b):
    print("multiplying")
    return a * b
"""

    tree1 = parser.parse(code1)
    tree2 = parser.parse(code2)
    tree3 = parser.parse(code3)

    # Get function nodes (skip module root to handle whitespace noise)
    func1 = parser.get_functions(tree1)[0]
    func2 = parser.get_functions(tree2)[0]
    func3 = parser.get_functions(tree3)[0]

    v1 = encoder.encode_ast(func1)
    v2 = encoder.encode_ast(func2)
    v3 = encoder.encode_ast(func3)

    sim_1_2 = encoder.similarity(v1, v2) # Should be high
    sim_1_3 = encoder.similarity(v1, v3) # Should be lower

    print(f"\nSimilarity(Identity, Renamed): {sim_1_2:.4f}")
    print(f"Similarity(Identity, Different): {sim_1_3:.4f}")

    # Expect higher similarity for structural match
    assert sim_1_2 > sim_1_3
    # Absolute threshold might vary, but renamed variables should still be very similar
    # The current simplified encoding includes token values (names) so it won't be 1.0, but structure dominates
    assert sim_1_2 > 0.6 
