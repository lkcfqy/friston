from src.core.agent import FEPAgent
import os

def main():
    print("=== Friston Agent: Code Generation Demo ===")
    
    if "LLM_BASE_URL" not in os.environ:
        print("Note: LLM_BASE_URL not set. Using default 'http://localhost:11434/v1'.")
    
    agent = FEPAgent()
    
    # Task: Write a script that checks if a number is prime
    # This is simple, but we want to see if it generates runnable code and verifies it.
    requirement = "Write a function is_prime(n) and a main block that asserts is_prime(17) is True and is_prime(4) is False. Print 'Verification Passed' if successful."
    
    filename = "prime_check.py"
    
    final_code = agent.create_feature(filename, requirement)
    
    print("\n[Final Verified Code]:")
    print(final_code)

if __name__ == "__main__":
    main()
