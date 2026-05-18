# friston

Project Friston 是一个主动推断风格的代码修复智能体原型。它把代码解析、超维表示、向量记忆、Mamba 上下文、Docker 沙箱执行和本地 LLM 修复组合成一个“感知代码、运行代码、计算 surprise、检索记忆或请求 LLM 修复”的闭环。

## 当前状态

仓库已经包含核心 `FEPAgent`、Docker 沙箱、AST/HDC 感知模块、Chroma 向量记忆、Mamba 上下文模块、Qwen/Ollama 兼容 LLM 客户端、演示脚本和 pytest 测试。

这是 coding-agent 研究原型。它可以生成和修复小型 Python 脚本，但不应直接视为安全的生产级自动编程系统。

## 工作流

1. `src/perception/parser.py` 把代码解析为 AST。
2. `src/perception/hdc.py` 把 AST 编码为高维向量。
3. `src/action/sandbox.py` 在 Docker 中注入文件、运行 ruff/mypy 和 Python 脚本。
4. `src/core/agent.py` 根据运行结果计算 surprise。
5. surprise 低时保存成功记忆；surprise 高时先检索 Chroma 记忆，再调用本地 LLM 生成修复。

## 主要目录

- `src/core/agent.py`：主动推断主循环。
- `src/core/llm_interface.py`：OpenAI-compatible LLM 客户端，默认指向 Ollama。
- `src/action/sandbox.py`：Docker 执行环境。
- `src/perception/`：AST、linter、HDC 编码。
- `src/memory/`：Mamba context、Modern Hopfield Memory、Chroma 向量库。
- `experiments/`：阶段演示和基础设施检查。
- `tests/`：感知、记忆、Mamba、沙箱测试。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

同时需要：

- 本机 Docker 可用。
- 本地或远程 OpenAI-compatible LLM 服务可用。
- 默认模型为 `qwen3:8b`，默认地址为 `http://localhost:11434/v1`。

可用环境变量覆盖：

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=sk-local
export LLM_MODEL=qwen3:8b
```

## 运行示例

```bash
python -m experiments.demo_phase1
python -m experiments.demo_v2_evolution
pytest
```

Mamba 相关测试可能需要 CUDA 和可编译的 `mamba-ssm` 环境；Docker 不可用时，沙箱测试也可能被跳过或失败。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
