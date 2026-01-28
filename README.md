# Project Friston: 神经符号主动推理智能体 (V2)

**Project Friston** 是下一代“稳态维持型”软件工程师智能体。它不仅仅是一个代码补全工具，而是一个基于**主动推理 (Active Inference)** 原理构建的数字生物。它的核心使命是将代码库维持在“低熵”状态——无 Bug、风格统一、逻辑自洽。

与传统 Agent 不同，Friston 具备**长期记忆 (Hippocampus)** 和**多模态感知 (Visual Cortex)**，能够在不运行代码的情况下“看见”静态错误，并能从过往经验中瞬间回忆起解决方案。

---

## 🧠 核心架构 (Biological Architecture)

该系统模仿生物大脑，由四个紧密协作的功能层组成：

### 1. 👁️ 感知层 (Perception): 神经符号之眼
让智能体“理解”代码的结构与质量，而不仅仅是处理文本。
*   **结构感知 (Structure)**: 结合 **Tree-sitter** 和 **超维计算 (HDC)**，将代码编码为 10,000 维的“概念向量”。这使得智能体能识别出“变量名不同但逻辑相同”的代码模式。
*   **静态感知 (LSP Vision)**: 集成 **Ruff** 和 **Mypy**。在代码提交执行前，智能体能通过 LSP 接口进行“预检 (Pre-flight Check)”，发现语法错误或类型不匹配会产生高“惊奇值”并拒绝执行。

### 2. 🧠 记忆层 (Memory): 系统 1 (System 1)
负责快速反应和经验积累，无需每次都进行昂贵的逻辑推理。
*   **海马体 (Hippocampus)**: 基于 **ChromaDB** 的向量记忆库。
    *   **记忆巩固**: 只有验证通过（低惊奇度）的修复方案才会被存入长期记忆。
    *   **瞬间召回**: 当通过 HDC 向量识别到相似的 Bug 模式时，直接提取历史修复方案，跳过 LLM 思考过程。
*   **直觉预测**: 使用 **Mamba** (状态空间模型) 进行序列建模，提供线性复杂度的代码预测。

### 3. 🤖 认知层 (Cognition): 系统 2 (System 2)
负责慢思考、逻辑推理和解决新颖问题。
*   **大语言模型**: 使用 **Qwen3** (via Ollama) 处理高阶逻辑。仅当 System 1 无法解决问题（惊奇度过高）时才被激活，从而大幅降低 Token 消耗和延迟。

### 4. 🛠️ 行动层 (Action): 身体
以安全的方式改变世界。
*   **Docker 沙箱**: 代码在隔离容器中执行。
*   **多维反馈**: 智能体的“感官输入”包括退出代码 (Exit Code)、标准输出/错误 (Logs) 以及 LSP 静态分析报告 (Lint Report)。

---

## 🚀 快速开始

### 环境依赖
*   Linux / WSL2
*   Python 3.10+
*   Docker (运行中)
*   本地 LLM (推荐 Ollama + `qwen3:8b`)
*   GPU (可选，加速 HDC/Mamba)

### 安装
```bash
# 1. 克隆项目
git clone https://github.com/yourusername/project-friston.git
cd project-friston

# 2. 创建环境
conda create -n friston python=3.10
conda activate friston

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 🎮 功能演示 (Demos)

我们提供了一系列演示脚本，展示智能体从基础感知到完全进化的能力。

### 🌟 旗舰演示: V2 完全体进化
展示集成了记忆与 LSP 的完整工作流。
*   **场景 1**: 智能体通过 LSP **拒收** 包含严重语法错误的代码。
*   **场景 2**: 智能体第一次用 LLM 修复逻辑 Bug 并习得经验；第二次遇到同样问题时，**直接从记忆中秒解**。
```bash
python -m experiments.demo_v2_evolution
```

### 其他组件演示
*   **主动推理闭环 (System 2)**: 经典的“预测-执行-惊奇-修复”循环。
    ```bash
    python -m experiments.demo_phase3
    ```
*   **自主创造模式 (Creation Mode)**: 从自然语言需求（如“写个质数判断”）到完美代码的端到端生成。
    ```bash
    python -m experiments.demo_creation
    ```
*   **结构感知 (System 1)**: 演示 HDC 如何识别变量重命名后的代码相似性。
    ```bash
    python -m experiments.demo_phase1
    ```

---

## 📂 项目结构

*   `src/core/`: 智能体的主循环 (Active Inference Loop) 与 LLM 接口。
*   `src/perception/`: 静态分析 (Linter), AST 解析, HDC 编码。
*   `src/memory/`: 向量数据库 (VectorDB), 联想记忆 (MHN), 序列模型 (Mamba)。
*   `src/action/`: Docker 沙箱环境。
*   `experiments/`: 各阶段能力的验证脚本。

---
**Project Friston** - Towards Low-Entropy Software Engineering.
