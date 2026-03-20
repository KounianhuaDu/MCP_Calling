## Contents
- MCP Benchmark
- Summary of Methodologies and Evaluation Frameworks
- Core Challenges of MCP Tool Calling
- Config Introduction
- Usage Notes
- Server and Dataset Collections
- Tuning Factory
- Fast Evaluation for MCP Calling
- Training data

## MCP Benchmark
See [`MCP_Benchmark`](MCP_Benchmark/)

## Summary of Methodologies and Evaluation Frameworks
See [`Summary_of_Methodologies_and_Evaluation_Frameworks`](Summary_of_Methodologies_and_Evaluation_Frameworks/)

## Core Challenges of MCP Tool Calling
See [`Core_Challenges_of_MCP_Tool_Calling`](Core_Challenges_of_MCP_Tool_Calling/)

## Config Introduction
See [`Config_Introduction`](Config_Introduction/)

## Usage Notes
See [`Usage_Notes`](Usage_Notes/)

## Server and Dataset Collections
- Server Collections:
    - [Smithery](https://smithery.ai/)
    - [MCP.so](https://glama.ai/mcp/servers)
    - [MCP.ing](https://mcp.ing/)
    - [MCP Market](https://mcpmarket.com/)
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

## Training data
We offer a training data set consisting of 8k+ samples synthesized from gpt-oss-20b using real-world servers, which is filtered from a 20w-samples collection and validated to be useful using a rule-based reward training pipeline.

------------------------
If you find this repo useful, please star us.
