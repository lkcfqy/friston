import pytest
import torch
import torch.nn.functional as F
from src.memory.mamba_context import MambaContext

def test_mamba_shapes():
    if not torch.cuda.is_available():
        pytest.skip("Skipping Mamba test because CUDA is not available")
        
    input_dim = 1024 # Use smaller dim for test speed
    ctx = MambaContext(input_dim=input_dim, hidden_dim=64, device="cuda")
    
    # Fake sequence of 10 vectors
    seq = torch.randn(10, input_dim)
    
    # Predict next
    pred = ctx.predict_next(seq)
    
    assert pred.shape == (input_dim,)
    assert pred.device.type == "cuda"
    assert not torch.isnan(pred).any()

def test_context_continuity():
    """Verify that model processes sequence without error."""
    if not torch.cuda.is_available():
        pytest.skip("Skipping Mamba test because CUDA is not available")
    
    ctx = MambaContext(input_dim=512, hidden_dim=64, device="cuda")
    seq = torch.randn(5, 512)
    pred1 = ctx.predict_next(seq)
    
    # Same input should yield same output (deterministic in eval)
    pred2 = ctx.predict_next(seq)
    
    assert torch.allclose(pred1, pred2)
