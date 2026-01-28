import json
from typing import List, Dict, Any

class Linter:
    """
    The 'Visual Cortex' of the agent.
    Parses raw output from static analysis tools (Ruff, Mypy) into structured perceptions.
    """
    
    @staticmethod
    def parse_ruff(json_output: str) -> List[Dict[str, Any]]:
        """
        Parse Ruff JSON output.
        Format: List of dicts with 'code', 'message', 'location', etc.
        """
        try:
            # Ruff output might be empty or contain non-json lines if something went wrong
            if not json_output.strip():
                return []
            
            # Filter distinct JSON part if mixed with other output
            # Usually ruff --format=json outputs pure JSON.
            data = json.loads(json_output)
            
            errors = []
            for item in data:
                errors.append({
                    "tool": "ruff",
                    "code": item.get("code"),
                    "message": item.get("message"),
                    "line": item.get("location", {}).get("row"),
                    "col": item.get("location", {}).get("column"),
                    # Approximate severity mapping usually handled by surprise function
                    "type": "format" if item.get("code") in ["E501"] else "syntax" 
                })
            return errors
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse Ruff JSON: {json_output[:100]}...")
            return []

    @staticmethod
    def parse_mypy(json_output: str) -> List[Dict[str, Any]]:
        """
        Parse Mypy JSON output.
        Note: Mypy doesn't have a native --format=json in older versions, 
        but we can parse the standard output or use a flag if available.
        Standard format: file:line:col: error: message [code]
        
        Using: mypy file.py --no-error-summary --show-error-codes --json (if available) 
        OR just parse line by line.
        Let's assume we invoke mypy with proper flags or parse standard output.
        
        Actually, let's use a robust line parser for mypy standard output 
        as it is most reliable across versions.
        """
        errors = []
        for line in json_output.splitlines():
            try:
                # Basic parsing: "file.py:8: error: Incompatible types... [assignment]"
                parts = line.split(":")
                if len(parts) >= 4:
                    # simplistic check
                    if "error" in parts[2] or "error" in parts[3]:
                        message = ":".join(parts[3:]).strip()
                        errors.append({
                            "tool": "mypy",
                            "code": "TYPE_ERROR",
                            "message": message,
                            "line": int(parts[1]) if parts[1].isdigit() else 0,
                            "col": 0, # Mypy cols varies
                            "type": "type"
                        })
            except Exception:
                continue
                
        return errors
