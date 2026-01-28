import torch
import torch.nn.functional as F
from typing import Dict, Any
import json

from src.perception.parser import ASTParser
from src.perception.linter import Linter
from src.perception.hdc import HDCEncoder
from src.memory.vector_db import VectorDB
import os
from src.memory.mamba_context import MambaContext
from src.action.sandbox import DockerSandbox
from src.core.llm_interface import QwenClient

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
        db_path = os.path.join(work_dir if work_dir != "/workspace" else ".", "friston_memory")
        self.memory = VectorDB(persist_path=db_path) 
        self.mamba = MambaContext(input_dim=10000, hidden_dim=256, device=self.device)
        self.sandbox = DockerSandbox(image="python:3.10-slim", work_dir=work_dir)
        self.llm = QwenClient()
        
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
        Includes Pre-flight check (Linting).
        Returns sensory state (exit_code, logs, lint_errors).
        """
        with self.sandbox as sb:
            sb.inject_file(filename, code)
            
            # Pre-flight Check: Linting
            raw_lint = sb.lint_file(filename)
            lint_errors = []
            lint_errors.extend(Linter.parse_ruff(raw_lint["ruff"]))
            lint_errors.extend(Linter.parse_mypy(raw_lint["mypy"]))
            
            # Decide if we even run the code?
            # For "Active Inference", we might want to run it anyway to see runtime errors,
            # UNLESS there are critical syntax errors which prevent running.
            has_syntax_errors = any(e['type'] == 'syntax' or e['type'] == 'format' for e in lint_errors) # Ruff E9, F821 etc are often fatal
            
            if has_syntax_errors:
                print("🛑 Pre-flight Check Failed: Syntax Errors detected. Skipping execution.")
                exit_code = 1
                stdout = ""
                stderr = "Static Analysis failed (Syntax/Format)."
            else:
                exit_code, stdout, stderr = sb.exec_run(f"python {filename}", timeout=5)
            
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "lint_errors": lint_errors
        }
        
    def compute_surprise(self, prediction: torch.Tensor, observation: Dict) -> float:
        """
        Surprise = -ln P(sensory_input | world_model)
        
        Quantified Surprise (Weighted Loss):
        - Runtime Error (Exit Code != 0): 1.0 (High)
        - Syntax Error (Linter): 1.0 (Critical)
        - Type Error (Linter): 0.5 (Moderate)
        - Format/Style (Linter): 0.1 (Low)
        """
        surprise = 0.0
        
        # 1. Runtime Success
        if observation['exit_code'] != 0:
            surprise += 1.0
            
        # 2. Static Analysis Surprise
        # Weighted sum of lint errors
        if 'lint_errors' in observation:
            for error in observation['lint_errors']:
                if error['type'] == 'syntax':
                    surprise += 1.0
                elif error['type'] == 'type':
                    surprise += 0.5
                else: 
                    surprise += 0.1
                    
        # Clip to [0, 1] or allow > 1? 
        # For simplicity, if ANY major error, we treat as max surprise for now.
        # But to allow "Gradual" improvement, let's keep it additive but thresholded check.
        
        return min(surprise, 1.0) # Normalized surprise cap
            
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
        last_buggy_vector = None
        last_error_tags = []
        
        for step in range(max_steps):
            print(f"\n--- Step {step+1} ---")
            
            # 0. Perceive (for Memory)
            # Create vector for current state
            current_vector_tensor = self.perceive(current_code)
            current_vector = current_vector_tensor.cpu().tolist()
            
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
                
                # Consolidation: If we came from a buggy state, save the fix
                if last_buggy_vector is not None:
                    print(f"💾 Consolidating new memory (Bug -> Fix)...")
                    self.memory.save_memory(
                        vector=last_buggy_vector,
                        code=current_code,
                        metadata={
                            "type": "fix",
                            "steps": step,
                            "filename": filename,
                            "error_tags": json.dumps(last_error_tags)
                        }
                    )
                
                return current_code
                
            # 4. Minimize Surprise
            print("⚠️ High Surprise detected! Initiating repair...")
            print(f"Error Log: {sensory['stderr'].strip()}")
            
            last_buggy_vector = current_vector
            
            # Capture error signature for metadata
            last_error_tags = []
            if 'lint_errors' in sensory:
                # Extract error codes (e.g., E501, F821) or generic stderr summary
                last_error_tags = [str(e.get('code')) for e in sensory['lint_errors'] if e.get('code')]
            
            # Policy Selection:
            # Path A: Reflexive (Memory Retrieval) - Fast
            print("🔍 System 1: Searching Hippocampus for similar problems...")
            memory_hit = self.memory.retrieve_memory(current_vector)
            
            if memory_hit:
                print(f"💡 Aha! Recall triggered (Similarity: {memory_hit['score']:.2f})")
                print("Applying remembered fix...")
                new_code = memory_hit['code']
                if new_code != current_code:
                    current_code = new_code
                    continue
            
            # Path B: Reflective (LLM Thinking) - Slow
            # For prototype, we use LLM for correction as errors usually need logic fixes
            
            print("🧠 Activating System 2 (Qwen3)...")
            new_code = self.llm.generate_fix(current_code, sensory['stderr'])
            
            if new_code == current_code:
                print("❌ System 2 failed to propose changes.")
                break
                
            current_code = new_code
            print("Generated Fix. Re-entering loop...")
            
        return current_code
