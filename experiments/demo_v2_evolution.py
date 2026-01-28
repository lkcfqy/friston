from src.core.agent import FEPAgent
import os
import shutil

def main():
    print("=== Friston V2 Evolution Demo ===")
    print("Showcasing: Visual Cortex (LSP) + Hippocampus (Memory)")
    
    # 1. Initialize Agent
    # Clean memory db for fresh demo
    if os.path.exists("./demo_v2_memory"):
        shutil.rmtree("./demo_v2_memory")
        
    # We patch the agent's memory path (hacky but quick for demo)
    # The agent init uses work_dir logic.
    work_dir = os.path.abspath("./demo_v2_workspace")
    agent = FEPAgent(work_dir=work_dir)
    
    # Manually override memory path to separate from other tests
    # agent.memory = VectorDB("./demo_v2_memory") (Actually agent does this if work_dir is set)
    
    print("\n--- Scenario 1: Pre-flight Interception (Visual Cortex) ---")
    print("Goal: Feed a script with syntax error. Agent should REFUSE to run it.")
    
    buggy_syntax_script = """
def bad_syntax():
    print("Missing parenthesis"
"""
    filename1 = "syntax_error.py"
    print(f"Code:\n{buggy_syntax_script}")
    
    # Run simple 1 step loop
    agent.run_loop(filename1, buggy_syntax_script, max_steps=1)
    
    
    print("\n--- Scenario 2: Memory Learning & Recall (Hippocampus) ---")
    print("Goal: Feed a script with a logic bug. Agent fixes it. Then feed it AGAIN.")
    print("Expectation: First time = System 2 (LLM). Second time = System 1 (Memory).")
    
    buggy_logic_script = """
def add(a, b):
    # Bug: subtracts instead of adds
    return a - b

if __name__ == "__main__":
    if add(2, 2) != 4:
        raise ValueError(f"2+2 should be 4, but got {add(2,2)}")
    print("Math is correct.")
"""
    filename2 = "logic_bug.py"
    
    print("\n[Run 1: First Encounter]")
    fixed_code = agent.run_loop(filename2, buggy_logic_script, max_steps=3)
    
    print("\n[Run 2: Induced Recall]")
    # We pass the SAME buggy code again.
    # The agent perceive() will generate the same vector.
    # It should hit the vector DB and recall the 'fixed_code'.
    
    print("Resetting Agent Loop for same problem...")
    agent.run_loop(filename2, buggy_logic_script, max_steps=1)
    
if __name__ == "__main__":
    main()
