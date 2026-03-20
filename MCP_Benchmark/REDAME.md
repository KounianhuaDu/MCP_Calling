## MCP Benchmark

| Category | Name | Proposer/Source | Key Features/Evaluation Focus | Remarks in MCP-Bench Related Tests |
| :--- | :--- | :--- | :--- | :--- |
| Closed-source/Commercial Models | Claude-Sonnet-4 | Anthropic | Meta-tool learning capability | Highest success rate (78.95%) in LiveMCPBench's 70-server/527-tool test |
| | Claude-Opus-4 | Anthropic | | High alignment with human evaluation in LiveMCPBench |
| | GPT-5 | OpenAI | | Performed well in MCP-Universe financial analysis tasks, but challenges remain with long context and unknown tool handling |
| | Gemini 2.5 Pro | Google | | Participated in the MCP-Universe evaluation |
| | Grok-4 | xAI | | Performed well in MCP-Universe browser automation tasks |
| Open-source Models | Qwen2.5-72B-Instruct | Alibaba | | ~75% alignment with human evaluators in LiveMCPBench assessment |
| | DeepSeek-V3 | DeepSeek | Often used as a "judge model" in evaluations | 81% alignment with human evaluators in LiveMCPBench assessment |
| | GLM-4.5 | Zhipu AI | | Rated as one of the best-performing open-source models in MCP-Universe tests |
| Evaluation Frameworks | MCPBench (Original) | | Accuracy, latency, token consumption for Web search, database search tasks | Evaluated MCP servers like Bing, Brave, DuckDuckGo |
| | LiveMCPBench | ISCAS | Large-scale tool navigation (70 servers/527 tools), real-world dynamic tasks, AI judging | Includes 95 real-life tasks |
| | MCP-Universe | Salesforce AI Research | Enterprise-level tasks (6 major domains), real MCP server interaction, long-context handling | Evaluates model performance in real business scenarios |
| | MCP-AttackBench (Security Focus) | Teams from Zhejiang University, CUHK, etc. | MCP security defense benchmark, 7 threat types, 70k+ samples | Used for training and evaluating defense systems like MCP-Guard |

- The baseline models listed in the table include both closed-source commercial models like **Claude-Sonnet-4** and **GPT-5**, as well as powerful open-source models like **Qwen2.5-72B-Instruct** and **DeepSeek-V3**. They are evaluated across different benchmarks, assessing capabilities such as tool usage accuracy, multi-step reasoning, long-context understanding, and handling unknown tools.
- The evaluation frameworks themselves, such as **LiveMCPBench** and **MCP-Universe**, also constitute important baseline references. They define task types, evaluation criteria, and environments, and subsequent research often uses them as a basis for comparing the performance of new models or systems. **MCP-AttackBench** focuses specifically on the security dimension.

