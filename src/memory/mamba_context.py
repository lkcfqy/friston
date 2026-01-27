import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaCodegen(nn.Module):
    """
    Toy Mamba model for code sequence modeling.
    Project Friston 'System 1' Prototype.
    """
    def __init__(self, d_model: int = 256, n_layers: int = 4, vocab_size: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        # In a real scenario, we might project HDC vectors (dim=10000) down to d_model
        # because running Mamba on dim=10000 might be expensive or memory hungy, 
        # though Mamba is efficient.
        # Let's assume input is HDC vector -> projection -> Mamba -> projection -> Output HDC
        
        self.encoder = nn.Linear(vocab_size, d_model) # Compression
        
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size) # Decompression to HDC space
        
    def forward(self, x):
        """
        x: [Batch, SeqLen, InputDim] (HDC Vectors)
        """
        x_emb = self.encoder(x)
        
        for layer in self.layers:
            x_emb = layer(x_emb)
            
        x_emb = self.norm(x_emb)
        output = self.decoder(x_emb)
        return output

class MambaContext:
    """
    Wrapper for the MambaCodegen model to handle state and inference.
    """
    def __init__(self, input_dim: int = 10000, hidden_dim: int = 256, device: str = "cuda"):
        self.device = device
        self.model = MambaCodegen(d_model=hidden_dim, n_layers=2, vocab_size=input_dim).to(device)
        self.model.eval() # Inference mode by default for now
        
    def predict_next(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict the next HDC vector given a sequence of vectors.
        sequence: [SeqLen, InputDim]
        returns: [InputDim]
        """
        with torch.no_grad():
            x = sequence.unsqueeze(0).to(self.device) # Add batch dim
            output = self.model(x)
            # Take the last token prediction
            last_token = output[0, -1, :]
            return last_token

    def train_step(self, sequence: torch.Tensor, target: torch.Tensor):
        # Placeholder for online learning
        pass
