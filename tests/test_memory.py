import pytest
import torch
import torch.nn.functional as F
from src.memory.mhn import HopfieldMemory

def test_memory_storage_retrieval():
    dim = 64
    memory = HopfieldMemory(dim=dim, beta=10.0) # High beta for sharp retrieval
    
    # Create 3 orthogonal patterns
    p1 = torch.zeros(dim)
    p1[:20] = 1.0
    
    p2 = torch.zeros(dim)
    p2[20:40] = 1.0
    
    p3 = torch.zeros(dim)
    p3[40:60] = 1.0
    
    patterns = torch.stack([p1, p2, p3])
    # Normalize expected patterns because memory stores normalized versions
    patterns = F.normalize(patterns, p=2, dim=1)
    
    memory.learn(patterns)
    
    # Test Exact Retrieval
    retrieved = memory.retrieve(patterns[0])
    # MHN is a continuous attractor, so it approaches the pattern but might have ε noise from other patterns due to softmax
    sim = F.cosine_similarity(retrieved.unsqueeze(0), patterns[0].unsqueeze(0))
    assert sim.item() > 0.9999

def test_noise_tolerance():
    dim = 128
    memory = HopfieldMemory(dim=dim, beta=5.0)
    
    # Generate random patterns
    torch.manual_seed(42)
    patterns = torch.randn(5, dim)
    patterns = F.normalize(patterns, p=2, dim=1)
    
    memory.learn(patterns)
    
    # Add noise to pattern 0
    target = patterns[0]
    noisy_input = target + 0.2 * torch.randn(dim)
    
    # Retrieve
    recovered = memory.retrieve(noisy_input)
    
    # Check similarity
    sim = F.cosine_similarity(recovered.unsqueeze(0), target.unsqueeze(0))
    print(f"Cosine Similarity (Recovered, Target): {sim.item()}")
    
    assert sim.item() > 0.95

def test_pattern_completion():
    dim = 100
    memory = HopfieldMemory(dim=dim, beta=8.0)
    
    # Pattern: [1, 1, ..., 1] vs [-1, -1, ..., -1] is too simple
    # Let's use random binary-like patterns
    p1 = torch.sign(torch.randn(dim))
    p2 = torch.sign(torch.randn(dim))
    
    patterns = torch.stack([p1, p2])
    patterns = F.normalize(patterns, p=2, dim=1)
    memory.learn(patterns)
    
    # Corrupt half of p1 (masking)
    corrupted = patterns[0].clone()
    corrupted[50:] = 0 # Zero out half
    
    recovered = memory.retrieve(corrupted)
    sim = F.cosine_similarity(recovered.unsqueeze(0), patterns[0].unsqueeze(0))
    
    assert sim.item() > 0.98
