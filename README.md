# Project Friston: 神经符号主动推理智能体

**Project Friston** 是下一代“稳态维持型”软件工程师智能体。与传统的 LLM 编程辅助工具（本质上是文本补全工具）不同，Friston 像一个数字生物，致力于将代码库维持在“低熵”状态。

它基于 **主动推理 (Active Inference)** 原理（自由能原理）驱动：当它感知到 Bug 或未实现的需求（产生高“惊奇/Surprise”）时，会主动采取行动（编码、调试）来消除这种不确定性。

## 🧠 核心架构

该系统模仿生物大脑，由四个核心功能层组成：

1.  **👁️ 感知层 (Perception): 神经符号之眼**
    *   **核心技术**: Tree-sitter + 超维计算 (Hyperdimensional Computing, HDC)。
    *   **能力**: 将代码编码为 10,000 维的“概念向量”。这让它能“看到”代码的**逻辑结构**，而不仅仅是字符，因此它能识别出变量名不同但逻辑相同的代码。

2.  **🧠 记忆层 (Memory): 系统 1 (System 1)**
    *   **核心技术**: Mamba (状态空间模型) + 现代霍普菲尔德网络 (Modern Hopfield Networks, MHN)。
    *   **能力**:
        *   **快思考**: Mamba 以线性复杂度预测代码序列，支持超长上下文。
        *   **联想记忆**: MHN 能从向量库中瞬间召回历史修复方案或算法原型。

3.  **🤖 认知层 (Cognition): 系统 2 (System 2)**
    *   **核心技术**: Qwen3 (通过 Ollama 接口)。
    *   **能力**: **慢思考**。仅当 System 1 无法解决问题（惊奇度过高）时激活。负责处理复杂的逻辑推理、架构设计和新功能生成。

4.  **🛠️ 行动层 (Action): 身体**
    *   **核心技术**: Docker Sandbox。
    *   **能力**: 在隔离环境中执行代码。Docker 返回的退出代码（Exit Code）、日志（Logs）构成了智能体的“感官输入”。

---

## 🚀 快速开始

### 环境要求

*   Linux / WSL2
*   Python 3.10+
*   Conda
*   Docker (需运行中且当前用户有权限访问)
*   **Ollama** (本地运行 `qwen3:8b` 或兼容模型)
*   NVIDIA GPU (推荐，用于加速 Mamba/HDC)

### 安装步骤

1.  **克隆仓库**
    ```bash
    git clone https://github.com/yourusername/project-friston.git
    cd project-friston
    ```

2.  **创建 Conda 环境**
    ```bash
    conda create -n friston python=3.10
    conda activate friston
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

### 配置

确保本地 LLM 已启动（默认连接到 `http://localhost:11434` 的 Ollama）。
如需自定义，请设置环境变量：
```bash
export LLM_BASE_URL="http://localhost:11434/v1"
```

---

## 🎮 演示实验 (Demos)

项目包含四个核心实验，展示不同阶段的能力：

### 1. 结构感知 (Structure Perception)
演示 HDC 编码器如何识别“变量重命名”后的代码结构相似性（即便是完全不同的变量名，结构相似度仍 > 0.85）。
```bash
python -m experiments.demo_phase1
```

### 2. 联想记忆 (Associative Memory)
演示霍普菲尔德网络（Hopfield Network）如何修复受损的记忆，从充满噪点的代码片段中联想回正确的算法原型。
```bash
python -m experiments.demo_phase2
```

### 3. 主动推理闭环 (Active Inference Loop)
**核心演示**。智能体拿到一个有 Bug 的脚本 (`fib_buggy.py`)。它预测运行成功 -> 实际失败 (产生惊奇) -> 激活 Qwen3 修复 -> 验证修复成功。
```bash
python -m experiments.demo_phase3
```

### 4. 自主创造模式 (Creation Mode)
演示智能体如何从零开始响应自然语言需求（例如“写一个质数判断函数”），生成代码并在 Docker 中反复验证修正，直到功能完美。
```bash
python -m experiments.demo_creation
```

---

## 📂 项目结构

*   `src/core/`: 智能体核心逻辑，主动推理循环，LLM 接口。
*   `src/perception/`: AST 解析与 HDC 向量编码器。
*   `src/memory/`: Mamba 上下文模型与 Hopfield 联想记忆。
*   `src/action/`: Docker 沙箱管理。
*   `experiments/`: 演示脚本。
*   `tests/`: 单元测试。

## 📝 许可证
MIT
