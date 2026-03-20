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

