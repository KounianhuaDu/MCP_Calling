**Quick Start**

在/zhan下载了一个python311，并配置初始环境~/zhan/python311/bin/python3.11 -m venv venv
依照READ.md配置环境
sudo apt-get install libpq-dev（password:password）
# .env配置
api用的azure,改动代码 mcpuniverse/llm/openai.py (48~56,def _build_client_and_model(), 139~140, 155~156, 175~176)

**mcp-universe的local model测评脚本 替换llm client部分为vllm server**
注意对应的config脚本要改model_name(benchmark_1.yaml: model_name: /home/knhdu/modelweights/Qwen3-8B)
system prompt 在agent/base.py
