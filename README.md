# RoadMap

## Contents
- MCP Benchmark
- Summary of Methodologies and Evaluation Frameworks
- Core Challenges of MCP Tool Calling
- Config Introduction
- Usage Notes
- Server and Dataset Collections
- Tuning Factory
- Fast Evaluation

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

**Summary of Methodologies and Evaluation Frameworks**

| Category | Name/Concept | Core Idea | Main Problem Solved | Applicable Scenarios/Cases |
| :--- | :--- | :--- | :--- | :--- |
| Methodology | Declarative Interface | Shifts **complex parameter construction (e.g., SQL) from the LLM to the server-side**; the LLM only needs to describe intent in natural language. | Difficulty in parameter construction; low accuracy of LLM-generated complex queries. | XiYan MCP Server, Natural language to database query |
| Methodology | Explicit vs Implicit Invocation | **Explicit Invocation**: Developer manually controls every aspect of tool calls (parsing LLM request, executing tool, integrating results), **strong control**.<br>**Implicit Invocation**: LLM automatically handles tool selection, parameter generation, and result integration, **more efficient development**. | Balancing development efficiency vs control; handling unstructured or error-prone output; need for direct access to raw data. | Fixed service scenarios (Explicit), General Chat applications (Implicit) |
| Evaluation Framework | MCPBench | **Fairly evaluates** different MCP servers on **accuracy, latency, Token consumption** using the **same LLM and Agent configuration**. | Horizontal performance comparison of MCP servers; basic performance benchmarking. | Web search (Bing, DuckDuckGo), Database query (MySQL, PostgreSQL) |
| Evaluation Framework | LiveMCPBench | Evaluates the agent's ability to **navigate, discover, and combine tools** within a **large-scale toolset (70+ servers, 527+ tools)** to complete **real-world dynamic tasks**. | Agent's retrieval, planning, and composition capabilities in vast toolsets; handling the dynamics of real tasks. | 95 real-life tasks in office, finance, travel, etc. |
| Evaluation Framework | MCP-specific Evaluation Systems (e.g., LiveMCPEval) | Employs **automated evaluation** using "LLM as a Judge", dynamically determining success based on **task key points**. | High heterogeneity and instability of tool outputs; diversity of solutions; difficulty evaluating dynamic tasks. | Replaces manual evaluation, handles large-scale, dynamic task assessment |

**Core Challenges of MCP Tool Calling**

- Parameter Construction Difficulty: LLMs often need to generate complex structured parameters (e.g., SQL queries, API request parameters), which places high demands on their reasoning and planning abilities and is prone to errors.
- High Heterogeneity of Returned Results: Data formats, structures, and information density vary greatly between different tools (could be raw data, HTML snippets, structured JSON, etc.), requiring LLMs to have strong **information extraction, summarization, and integration capabilities**.
- Tool Dynamicity and Uncertainty: Real-world tools (especially web APIs) may return results that change over time, with input, or due to external state, requiring agents to have **robustness** and **adaptability**.

## Config Introduction
MCP services need to select the appropriate communication protocol type during registration and configuration based on the actual scenario:
| Protocol Type | Communication Method | Advantages | Disadvantages | Applicable Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| Stdio | Standard Input/Output (Command line) | Simple and direct, no network required | No network communication capability | Local development/debugging, offline environment verification |
| SSE | Server-Sent Events (Unidirectional stream) | Low latency, strong compatibility | Only supports unidirectional communication from server to client | Unidirectional real-time communication scenarios like API gateway config push |
| Streamable HTTP | HTTP (Supports bidirectional streams) | Supports bidirectional communication, suitable for cross-network deployment | More complex compared to SSE | Formal environment deployment, hybrid cloud/cross-VPC communication |
| HTTP | Synchronous Request/Response | Strong compatibility, universal | Does not support streaming communication | Most standard interface scenarios |
| Webflux | Asynchronous Non-blocking (Reactive) | Supports high concurrency and real-time data streams, high performance | Relatively complex to understand and implement | Scenarios requiring fast response and high concurrency |
| Spring Bean | Invocation within Spring container | Seamless integration with Spring ecosystem | Typically limited to Java Spring applications | MCP tools implemented based on Spring Beans |

- MCP Configuration File
MCP configuration files commonly use JSON format to define how to connect to and start MCP servers. A typical MCP configuration file contains the following main fields:

| Field Name | Required | Description |
| :--- | :--- | :--- |
| `mcpServers` | Yes | An object containing definitions for all MCP servers. |
| `server_name` (custom) | -- | Custom identifier for the service (e.g., `filesystem`, `fetch`). |
| `type` | Yes | Service type, e.g., `stdio` (local process communication) or `sse` (remote Server-Sent Events API). |
| `command` | Yes | Command to start the server (e.g., `python script.py`). |
| `args` | No | List of arguments passed to the command. |
| `env` | No | Key-value pairs for environment variables, used to pass API keys, path configurations, etc. |
| `url` | No | When the type is `sse`, specifies the URL of the remote server. |

- Configuration Examples
The configuration file differs depending on the deployment method:
1.  Using NPX deployment (often used for Node.js related MCP servers)
    ```json
    {
      "mcpServers": {
        "amap-maps": {
          "command": "npx",
          "args": [
            "-y",
            "@amap/amap-maps-mcp-server"
          ],
          "env": {
            "AMAP_MAPS_API_KEY": "Your_API_Key"
          }
        }
      }
    }
    ```
2.  Using UVX deployment (using the uvx tool for installation and execution)
    ```json
    {
      "mcpServers": {
        "MCP-timeserver": {
          "command": "uvx",
          "args": ["MCP-timeserver"]
        }
      }
    }
    ```
3.  Remote URL (SSE) method (connecting to a remote SSE service)
    ```json
    {
      "mcpServers": {
        "amap-maps-sse": {
          "url": "https://mcp.amap.com/sse?key=Your_AMap_API_Key"
        }
      }
    }
    ```
4.  Spring Bean method (suitable for Spring ecosystem)
    ```yaml
    spring:
      ai:
        mcp:
          server:
            name: my-spring-mcp-server
            type: SYNC
      alibaba:
        mcp:
          nacos:
            server-addr: your-nacos-server-addr
            namespace: public
    ```

## Usage Notes
- Environment Preparation: Many local (Stdio) MCP servers require pre-installing necessary runtime environments, such as **Node.js** (for npx), **Python**, or **UV** (for uvx).
- Dependency Installation: Ensure the required dependency packages for the MCP server are correctly installed. `npx -y` and `uvx` usually handle this automatically, but network issues can cause failures.
- Permission Management: Servers accessing local file systems or system resources may require appropriate permissions.
- Testing and Verification:
    - After configuration is complete, check the MCP server status in the client (often indicated by a green light or status indicator).
    - Try triggering the relevant tools in conversation and observe if they can be called normally and return results.

## Server and Dataset Collections
- Server Collections:
    - [Smithery](https://smithery.ai/)
    - [MCP.so](https://glama.ai/mcp/servers)
    - [MCP.ing](https://mcp.ing/)
    - [AWS MCPs](https://github.com/awslabs/mcp)
    - [Genai Toolbox](https://googleapis.github.io/genai-toolbox/getting-started/introduction/)
    - [Microsoft MCPs](https://github.com/microsoft/mcp)
    - [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
    - [Awesome-MCP-ZH](https://github.com/yzfly/Awesome-MCP-ZH) 

- Datasets:
    - [TOUCAN: SYNTHESIZING 1.5M TOOL-AGENTIC DATA FROM REAL-WORLD MCP ENVIRONMENTS](https://arxiv.org/pdf/2510.01179)

## Tuning Factory
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory)
- [Art](https://art.openpipe.ai/getting-started/about)
- [Verl](https://github.com/volcengine/verl)


## Fast Evaluation for MCP Calling
We offer the revised version of "LiveMCP" and "MCPUniverse" for convenient testing that supports local model evaluation, using vllm and requests to replace the openai wrapper.
- For LiveMCP, see Readme4LiveMCP.md.
