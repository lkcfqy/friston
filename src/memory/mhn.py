import torch
import torch.nn.functional as F
from typing import Optional

class HopfieldMemory:
    """
    Continuous Modern Hopfield Network (Dense Associative Memory).
    
    Ref: Ramsauer, H., et al. "Hopfield Networks is All You Need." ICLR 2021.
    
    Update rule:
    xi_{t+1} = X * softmax(beta * X^T * xi_t)
    
    Where:
    - X: Stored patterns matrix (N_patterns x Dimension)
    - xi: State vector (Dimension)
    - beta: Inverse temperature parameter (controls retrieval sharpness)
    """
    def __init__(self, dim: int, beta: float = 1.0, device: str = "cpu"):
        self.dim = dim
        self.beta = beta
        self.device = device
        # X: Memory matrix [NumPatterns, Dim]
        self.X: Optional[torch.Tensor] = None 
        
    def learn(self, patterns: torch.Tensor):
        """
        Store patterns in the memory.
        patterns: Tensor of shape [NumPatterns, Dim]
        """
        if patterns.dim() != 2 or patterns.shape[1] != self.dim:
            raise ValueError(f"Expected patterns shape [N, {self.dim}], got {patterns.shape}")
        
        self.X = patterns.to(self.device)
        # Normalize patterns to lie on the sphere for cosine similarity based retrieval
        # Although the original update rule works with raw dot products, normalizing helps stability
        self.X = F.normalize(self.X, p=2, dim=1)

    def retrieve(self, query: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Retrieve memory associated with the query.
        
        Args:
            query: Query vector [Dim]
            num_steps: Number of update steps (usually 1 is enough for MHN attention mechanism)
            
        Returns:
            Recovered pattern [Dim]
        """
        if self.X is None:
            raise RuntimeError("Memory is empty. Call learn() first.")
        
        xi = query.to(self.device)
        xi = F.normalize(xi, p=2, dim=0) # Normalize query as well
        
        for _ in range(num_steps):
            # Attention logits: X * xi (dot product similarity)
            # Shape: [N_patterns, Dim] x [Dim] -> [N_patterns]
            logits = self.beta * (self.X @ xi)
            
            # Attention weights: softmax(logits)
            probs = F.softmax(logits, dim=0)
            
            # Update: weighted sum of patterns
            # Shape: [Dim, N_patterns] x [N_patterns] -> [Dim]
            xi_new = self.X.T @ probs
            
            # Optional: Normalize result if we want to stay on sphere
            xi = F.normalize(xi_new, p=2, dim=0)
            
        return xi

    def energy(self, state: torch.Tensor) -> float:
        """
        Energy function of the Continuous Hopfield Network.
        E = - lse(beta * X^T * state) / beta + 0.5 * state^T * state
        """
        if self.X is None:
            return 0.0
            
        state = state.to(self.device)
        # lse term
        logits = self.beta * (self.X @ state)
        lse = torch.logsumexp(logits, dim=0) / self.beta
        
        # quadratic term (if state is normalized, this is constant 0.5)
        quad = 0.5 * (state @ state)
        
        return -(lse - quad).item()
