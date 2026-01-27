import torch
import hashlib
from typing import List

class HDCEncoder:
    def __init__(self, dim: int = 10000, device: str = "cpu"):
        self.dim = dim
        self.device = device
        # Base vectors cache to ensure determinism for same token
        self.token_memory = {}
        
    def _get_token_vector(self, token: str) -> torch.Tensor:
        """Deterministic mapping from string token to hypervector using hash."""
        if token in self.token_memory:
            return self.token_memory[token]
        
        # Use hashing to create a deterministic seed
        sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
        seed = int(sha, 16) % (2**32)
        
        # Generate random bipolar vector {-1, 1}
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        # Random integers 0 or 1 -> map to -1 or 1
        # bernoulli with p=0.5
        v = torch.randint(0, 2, (self.dim,), generator=g, device=self.device, dtype=torch.float32)
        v = 2 * v - 1 # Map [0, 1] to [-1, 1]
        
        self.token_memory[token] = v
        return v

    def bind(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        HDC Binding operation.
        For bipolar vectors: Element-wise Multiplication (equivalent to XOR in binary).
        Preserves distance (noise tolerant) but changes similarity orthogonal to inputs.
        """
        return v1 * v2

    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        HDC Bundling (Superposition) operation.
        Element-wise addition followed by sign function (majority rule).
        Result approximates the set of input vectors.
        """
        if not vectors:
            return torch.zeros(self.dim, device=self.device)
        
        sum_vec = torch.stack(vectors).sum(dim=0)
        # Resolve ties randomly (usually not strictly necessary for high dim but good for theoretical purity)
        # Here we just use sign(x), if 0 we can default to 1 or -1. sign(0) is 0 in torch.
        res = torch.sign(sum_vec)
        # Fix zeros (ties) by setting to 1 (arbitrary but consistent)
        res[res == 0] = 1.0
        return res

    def permute(self, v: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """
        HDC Permutation operation (Cyclic shift).
        Used to encode order/sequence.
        """
        return torch.roll(v, shifts=shifts)
    
    def similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Cosine similarity for bipolar vectors."""
        return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    def encode_ast(self, node, depth=0) -> torch.Tensor:
        """
        Recursive encoding of AST node.
        Encoding strategy:
        TreeVec = Bind(TypeVec, Bundle( [Bind(Permute(ChildVec, i), PositionVec_i) for i, child in children] ))
        
        Simplified strategy for this prototype:
        NodeVector = Bundle(
            Bind(TokenType, "type"),
            Bind(TokenValue, "value"), (if leaf)
            Bind(Permute(Child_1, 1), "child_1"),
            Bind(Permute(Child_2, 2), "child_2"),
            ...
        )
        """
        # Feature 1: Node Type
        type_vec = self.bind(self._get_token_vector(node.type), self._get_token_vector("CTX_TYPE"))
        
        components = [type_vec]
        
        # Feature 2: Node Content (if leaf/identifier)
        if node.child_count == 0:
            # Leaf node, use its text
            if node.type in ["identifier", "string", "number", "integer"]:
                text_content = node.text.decode("utf-8")
                val_vec = self.bind(self._get_token_vector(text_content), self._get_token_vector("CTX_VAL"))
                components.append(val_vec)
        else:
            # Recursive Children
            # Positional encoding: Permute child vector by its index + 1
            for i, child in enumerate(node.children):
                child_vec = self.encode_ast(child, depth + 1)
                # Permute to encode structural position
                permuted_child = self.permute(child_vec, shifts=i+1)
                components.append(permuted_child)
        
        return self.bundle(components)
