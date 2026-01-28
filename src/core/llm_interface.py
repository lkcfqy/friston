import os
import requests
import json
from typing import Optional

class QwenClient:
    """
    Client for interacting with Qwen (or compatible Local LLM).
    Supports OpenAI-compatible API endpoints (e.g., Ollama, vLLM).
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3:8b"):
        self.base_url = os.getenv("LLM_BASE_URL", base_url)
        self.api_key = os.getenv("LLM_API_KEY", "sk-local") # Local usually ignores key
        self.model = os.getenv("LLM_MODEL", model)
        
    def generate_fix(self, code: str, error_log: str) -> str:
        """
        Reflective System 2: Generate a fix for the buggy code.
        """
        prompt = f"""
You are an expert Python software engineer.
The following code failed to execute.

CODE:
```python
{code}
```

ERROR:
{error_log}

Please provide the FIXED code. Return ONLY the python code block. Do not explain.
"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a code repair assistant. Output only valid Python code."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract code block if present
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return content
            
            
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            # Fallback: return original code (no fix)
            return code

    def generate_code(self, user_prompt: str) -> str:
        """
        System 2: Generate new code from scratch based on user prompt.
        """
        prompt = f"""
You are an expert Python software engineer.
Write a Python script that achieves the following goal:

GOAL:
{user_prompt}

Return ONLY the python code block. No explanation. 
Ensure the code includes a main block to run/test itself.
"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a code generator. Output only valid Python code."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract code block
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return content
            
        except Exception as e:
            print(f"LLM Generation Failed: {e}")
            return f"# Error generating code: {e}"
