**Core Challenges of MCP Tool Calling**

- Parameter Construction Difficulty: LLMs often need to generate complex structured parameters (e.g., SQL queries, API request parameters), which places high demands on their reasoning and planning abilities and is prone to errors.
- High Heterogeneity of Returned Results: Data formats, structures, and information density vary greatly between different tools (could be raw data, HTML snippets, structured JSON, etc.), requiring LLMs to have strong **information extraction, summarization, and integration capabilities**.
- Tool Dynamicity and Uncertainty: Real-world tools (especially web APIs) may return results that change over time, with input, or due to external state, requiring agents to have **robustness** and **adaptability**.
