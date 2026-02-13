# 🧠 Project Friston: Neuro-Symbolic Active Inference Agent (V2) ✨

Welcome to **Project Friston**! 🚀 This is a next-generation "homeostasis-maintaining" software engineer agent. It's not just another code completion tool—it's a digital organism built on the principles of **Active Inference** (Free Energy Principle). 

Its core mission? To keep your codebase in a cozy "low-entropy" state: bug-free, beautifully styled, and logically consistent! 🌿✨

Unlike traditional agents, Friston has a **Long-term Memory (Hippocampus)** 🐘 and **Multi-modal Perception (Visual Cortex)** 👁️. It can "see" static errors without even running the code and instantly recall solutions from past experiences! ⚡

---

## 🧬 Biological Architecture 🏗️

The system mimics a biological brain, consisting of four tightly integrated functional layers:

### 1. 👁️ Perception: The Neuro-Symbolic Eyes
Allows the agent to truly "understand" code structure and quality.
* **Structural Perception**: Combines **Tree-sitter** 🌳 and **Hyperdimensional Computing (HDC)** 🧮 to encode code into 10,000-dimensional "concept vectors". It can recognize identical logic even if variable names change!
* **Static Vision (LSP)**: Integrates **Ruff** and **Mypy** 🔍. Before executing code, it performs a "Pre-flight Check". Syntax errors trigger high "Surprise" (Free Energy), and the agent will refuse to run it. 🛑

### 2. 💾 Memory: System 1 (Fast Thinking)
Responsible for quick reflexes and experience accumulation, bypassing expensive LLM calls.
* **Hippocampus**: A Vector DB powered by **ChromaDB** 🗄️.
    * *Consolidation*: Only verified fixes (low surprise) are stored in long-term memory. 🏆
    * *Instant Recall*: When HDC vectors detect a familiar bug, it pulls the historical fix instantly! ⚡
* **Intuition**: Uses **Mamba** 🐍 (State Space Model) for linear-complexity code sequence prediction.

### 3. 🤔 Cognition: System 2 (Slow Thinking)
Handles slow thinking, logical reasoning, and novel problem-solving.
* **Large Language Model (LLM)**: Powered by **Qwen3** (via Ollama) 🤖. It only activates when System 1 fails (surprise is too high), saving massive token costs and latency! 💸⏱️

### 4. 🏃 Action: The Body
Interacts with the world safely.
* **Docker Sandbox**: Executes code in an isolated container environment. 🐳
* **Multi-dimensional Feedback**: Senses the world through Exit Codes, Logs (stdout/stderr), and Lint Reports. 📊

---

## 🚀 Quick Start 🛠️

### Prerequisites 📋
* Linux / WSL2 🐧
* Python 3.10+ 🐍
* Docker (Running) 🐳
* Local LLM (Recommended: Ollama + `qwen3:8b`) 🦙
* GPU (Optional, but great for HDC/Mamba acceleration) 🎮

### Installation 📥
```bash
# 1. Clone the repository 📂
git clone [https://github.com/yourusername/project-friston.git](https://github.com/yourusername/project-friston.git)
cd project-friston

# 2. Create an environment 🌱
conda create -n friston python=3.10
conda activate friston

# 3. Install dependencies 📦
pip install -r requirements.txt

```

---

## 🎮 Demos & Playgrounds 🎪

We provide several fun scripts to showcase the agent's evolution!

### 🌟 Flagship Demo: V2 Evolution Complete

Showcases the full workflow with Memory and LSP integration!

* **Scenario 1**: The agent uses LSP to **reject** code with syntax errors. 🙅‍♀️
* **Scenario 2**: The agent uses the LLM to fix a bug, learns it, and then **instantly solves it from memory** the next time! 🧠✨

```bash
python -m experiments.demo_v2_evolution

```

### 🧩 Other Cool Demos

* **Active Inference Loop (System 2)** 🔄: The classic "Predict -> Execute -> Surprise -> Fix" cycle.
```bash
python -m experiments.demo_phase3

```


* **Creation Mode 🎨**: End-to-end code generation from a natural language prompt (e.g., "write a prime checker").
```bash
python -m experiments.demo_creation

```


* **Structural Perception (System 1) 🏗️**: Watch HDC identify code similarity even after variable renaming!
```bash
python -m experiments.demo_phase1

```



---

## 📁 Project Directory 🗺️

* `src/core/` 🧠: The Active Inference Loop and LLM interface.
* `src/perception/` 👁️: Static analysis (Linter), AST Parser, HDC Encoding.
* `src/memory/` 💾: VectorDB, Associative Memory (MHN), Sequence Models (Mamba).
* `src/action/` 🏃: Docker Sandbox environment.
* `experiments/` 🧪: Test scripts for various agent capabilities.

---

*Built with ❤️ for a Low-Entropy Software Engineering future.* 🌌
