from src.core.agent import FEPAgent
import os

def main():
    print("=== Phase 3: The Active Inference Loop ===")
    
    # Configuration for Local LLM (User can set env vars)
    # Defaulting to an example Local URL if not set
    if "LLM_BASE_URL" not in os.environ:
        print("Note: LLM_BASE_URL not set. Using default 'http://localhost:11434/v1'.")
        print("Ensure your local Qwen3/Ollama is running!")

    agent = FEPAgent()
    
    # The Task: A buggy Fibonacci script
    # Bug: Infinite recursion (no base case for 0? or typo?)
    # Let's make a syntax error + logic error
    buggy_script = """
def fib(n):
    if n <= 1: return n
    # Intentional NameError to trigger Active Inference (System 2)
    return n + fib_typo(n-1)

if __name__ == "__main__":
    print(f"Fib(5) = {fib(5)}")
"""
    filename = "fib_buggy.py"
    
    print(f"\n[Initial State] Injecting buggy code:\n{buggy_script}")
    
    final_code = agent.run_loop(filename, buggy_script)
    
    print("\n[Final State] Code after Active Inference:")
    print(final_code)

if __name__ == "__main__":
    main()
