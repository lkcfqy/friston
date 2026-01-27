import torch
import torch.nn.functional as F
from typing import Dict, Any

from src.perception.parser import ASTParser
from src.perception.hdc import HDCEncoder
from src.memory.mhn import HopfieldMemory
from src.memory.mamba_context import MambaContext
from src.action.sandbox import DockerSandbox
from src.core.llm_interface import DeepSeekClient

class FEPAgent:
    """
    The Project Friston Agent.
    Implements the Active Inference Loop:
    1. Prediction (Mamba)
    2. Action (Docker / Edit)
    3. Sensory Input (Execution Result)
    4. Surprise Minimization (Reflexive vs Reflective)
    """
    def __init__(self, work_dir: str = "/workspace"):
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing FEPAgent on device: {self.device}")

        # Components
        self.parser = ASTParser()
        self.encoder = HDCEncoder(dim=10000, device=self.device)
        self.memory = HopfieldMemory(dim=10000) # Memory can stay CPU or move to CUDA, usually safer on CPU for large storage
        self.mamba = MambaContext(input_dim=10000, hidden_dim=256, device=self.device)
        self.sandbox = DockerSandbox(image="python:3.10-slim", work_dir=work_dir)
        self.llm = DeepSeekClient()
        
        # State
        self.surprise_threshold = 0.5
        
    def perceive(self, code: str) -> torch.Tensor:
        """Text -> AST -> HDC Vector"""
        tree = self.parser.parse(code)
        # Assuming single function for simplicity in prototype
        func_nodes = self.parser.get_functions(tree)
        if not func_nodes:
            # Fallback for scripts without functions
            return torch.zeros(10000, device=self.device) 
        return self.encoder.encode_ast(func_nodes[0])
        
    def act(self, filename: str, code: str) -> Dict[str, Any]:
        """
        Execute the code in the world (Docker).
        Returns sensory state (exit_code, logs).
        """
        with self.sandbox as sb:
            sb.inject_file(filename, code)
            exit_code, stdout, stderr = sb.exec_run(f"python {filename}", timeout=5)
            
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr
        }
        
    def compute_surprise(self, prediction: torch.Tensor, observation: Dict) -> float:
        """
        Surprise = -ln P(sensory_input | world_model)
        
        Simplified Model:
        Prior Belief: "Code should run successfully" (Exit Code 0)
        Observation: Actual Exit Code.
        
        Surprise is high if Exit Code != 0.
        """
        if observation['exit_code'] == 0:
            return 0.0 # Expected
        else:
            return 1.0 # High Surprise (Error)
            
    def create_feature(self, filename: str, requirement: str):
        """
        Creation Mode:
        1. Generate initial implementation (System 2).
        2. Enter Active Loop to verify and refine (System 1 + 2).
        """
        print(f"✨ Generating initial code for: {requirement}")
        initial_code = self.llm.generate_code(requirement)
        
        print("\n[Initial Draft Generated]")
        # print(initial_code)
        
        # Verify it immediately
        return self.run_loop(filename, initial_code)

    def run_loop(self, filename: str, initial_code: str, max_steps: int = 3):
        current_code = initial_code
        
        for step in range(max_steps):
            print(f"\n--- Step {step+1} ---")
            
            # 1. Prediction (Prior)
            # Mamba predicts the next state (conceptually). 
            # Ideally Mamba predicts "successful execution vector".
            # Here we implicitly hold the belief "Success".
            
            # 2. Action (Execute)
            print("Executing in Sandbox...")
            sensory = self.act(filename, current_code)
            
            # 3. Compute Surprise
            surprise = self.compute_surprise(None, sensory)
            print(f"Surprise: {surprise}")
            
            if surprise < 0.1:
                print("✅ State is stable (Low Entropy). Task Complete.")
                print(f"Output: {sensory['stdout'].strip()}")
                return current_code
                
            # 4. Minimize Surprise
            print("⚠️ High Surprise detected! Initiating repair...")
            print(f"Error Log: {sensory['stderr'].strip()}")
            
            # Policy Selection:
            # Path A: Reflexive (Memory Retrieval) - Fast
            # Path B: Reflective (LLM Thinking) - Slow
            
            # For prototype, we use LLM for correction as errors usually need logic fixes
            # In full version, check Memory first for similar stored fixes
            
            print("🧠 Activating System 2 (Qwen3)...")
            new_code = self.llm.generate_fix(current_code, sensory['stderr'])
            
            if new_code == current_code:
                print("❌ System 2 failed to propose changes.")
                break
                
            current_code = new_code
            print("Generated Fix. Re-entering loop...")
            
        return current_code
