import asyncio
import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple

import dotenv
from mcp import ClientSession
from tqdm import tqdm
import argparse
import uuid
import re
from vllm import LLM

from utils.clogger import _set_logger
from utils.llm_vllm import ChatModel
from utils.mcp_client import MCPClient

_set_logger(
    exp_dir=pathlib.Path("./logs"),
    logging_level_stdout=logging.INFO,
    logging_level=logging.DEBUG,
    file_name="baseline.log",
)
dotenv.load_dotenv()
logger = logging.getLogger(__name__)

INPUT_QUERIES_FILE = "./baseline/data/example_queries.json"
CONVERSATION_RESULTS_FILE = f"./baseline/output/{os.getenv('MODEL', 'None').replace('/', '_')}_{os.getenv('EMBEDDING_MODEL', 'None').replace('/', '_')}.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=INPUT_QUERIES_FILE,
        help="Path to the input queries file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=CONVERSATION_RESULTS_FILE,
        help="Path to the output conversation results file.",
    )
    return parser.parse_args()


class LoggingMCPClient(MCPClient):
    def __init__(self):
        super().__init__(timeout=180, max_sessions=9999)
        print(os.getenv("BASE_URL"))
        self.chat_model = ChatModel(
            #model_name=os.getenv("MODEL"),
            #api_key=os.getenv("OPENAI_API_KEY"),
            model_url=os.getenv("BASE_URL"),
        )

    async def connect_copilot(self):
        if "mcp-copilot" not in self.sessions:
            # Forward CUDA / NVIDIA device env vars explicitly to the child server
            # Some subprocess or stdio client wrappers may not preserve the
            # parent process environment exactly; provide explicit mapping so
            # the mcp-copilot process uses the intended GPU(s).
            env_map = {}
            for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
                val = os.environ.get(key)
                if val is not None:
                    env_map[key] = f"${{{key}}}"
            if env_map:
                logger.info(f"Forwarding env to mcp-copilot: {list(env_map.keys())}")

            await self.config_connect(
                config={
                    "mcpServers": {
                        "mcp-copilot": {
                            "command": "python",
                            "args": ["-m", "baseline.mcp_copilot"],
                            # when provided, MCPClient._process_env_vars will
                            # substitute ${VAR} with the current os.environ value
                            # and pass the resulting env dict to the subprocess.
                            **({"env": env_map} if env_map else {}),
                        },
                    }
                },
            )
            logger.info("Connected to MCP Copilot server.")

    async def process_query(
        self,
        query: str,
        history: Optional[list] = None,
        max_tool_tokens: int = 10000,
    ) -> Tuple[str, List[dict]]:
        if history is None:
            messages = [
                {
                    "role": "system",
                    "content": """\
You are an agent designed to assist users with daily tasks by using external tools. You have access to two tools: a retrieval tool and an execution tool. The retrieval tool allows you to search a large toolset for relevant tools, and the execution tool lets you invoke the tools you retrieved. Whenever possible, you should use these tools to get accurate, up-to-date information and to perform file operations.

Note that you can only response to user once, so you should try to provide a complete answer in your response.
""",
                }
            ]
        else:
            messages = history.copy()

        messages.append({"role": "user", "content": query})

        available_tools = []
        for server in self.sessions:
            session = self.sessions[server]
            assert isinstance(session, ClientSession), (
                "Session must be an instance of ClientSession"
            )
            response = await session.list_tools()
            available_tools += [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in response.tools
            ]
        final_text = []
        stop_flag = False
        try:
            while not stop_flag and len(messages) < 30:
                '''request_payload = {
                    "messages": messages,
                    "tools": available_tools,
                }
                response = self.chat_model.complete_with_retry(**request_payload)
                '''
                request_payload = {
                    "messages": messages,
                    "tools": available_tools,
                }
                response = self.chat_model.chat_with_retry(message = messages, tools=available_tools)

                if hasattr(response, "error"):
                    raise Exception(
                        f"Error in OpenAI response: {response.error['metadata']['raw']}"
                    )

                response_message = response['choices'][0]['text']
                matches = re.findall(r'<FUNCTION_CALL>\s*(\{.*?\})\s*</FUNCTION_CALL>', response_message, re.DOTALL)
                tool_calls = []
                for match in matches:
                    if match.count('{') == match.count('}') + 1:
                        match = match + '}'
                    try:
                        function_call = json.loads(match)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        function_call = None
                    if function_call:
                        tool_call = {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": function_call["name"],
                                "arguments": json.dumps(function_call["arguments"], ensure_ascii=False)
                            }
                        }
                        tool_calls.append(tool_call)
                messages.append({"role": "assistant", "content": response_message, "tool_calls": tool_calls})
                content = response_message
                if (
                    content
                    and not tool_calls
                ):
                    final_text.append(content)
                    stop_flag = True
                else:
                    if not tool_calls:
                        logger.warning(
                            "Received empty response from LLM without content or tool calls."
                        )
                        break

                    for tool_call in tool_calls:
                        for _ in range(3):
                            try:
                                tool_name = tool_call["function"]["name"]
                                tool_args = json.loads(tool_call['function']['arguments'])
                                tool_id = tool_call["id"]
                                # There is only one server in our method
                                # We use mcp-copilot to route the servers
                                server_id = "mcp-copilot"
                                session = self.sessions[server_id]

                                logger.info(
                                    f"LLM is calling tool: {tool_name}({tool_args})"
                                )
                                # timeout
                                result = await asyncio.wait_for(
                                    session.call_tool(tool_name, tool_args), timeout=300
                                )
                                break   
                            except asyncio.TimeoutError:
                                logger.error(f"Tool call {tool_name} timed out.")
                                result = "Tool call timed out."
                                await self.cleanup_server("mcp-copilot")
                                await self.connect_copilot()
                            except Exception as e:
                                logger.error(f"Error calling tool {tool_name}: {e}")
                                result = f"Error: {str(e)}"
                        result = str(result)
                        result = result[:max_tool_tokens]
                        # logger.info(str(result))
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": str(result),
                            }
                        )
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            final_text.append(f"Error: {str(e)}")
            messages.append({"role": "assistant", "content": str(e)})
        self.history = messages
        # import pdb; pdb.set_trace()
        return "\n".join(final_text), messages


async def main(args):
    if not pathlib.Path(args.input_path).exists():
        logger.error(f"Input queries file {args.input_path} does not exist.")
        return
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"len(queries): {len(data)}")
    client = LoggingMCPClient()
    await client.connect_copilot()
    rerun = True
    if os.path.exists(args.output_path) and not rerun:
        with open(args.output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        exist_ids = {entry["task_id"] for entry in all_results}
    else:
        all_results = []
        exist_ids = set()
    error_queries = set()
    try:
        for entry in tqdm(data):
            task_id = entry["task_id"]
            if task_id in exist_ids:
                continue
            query = entry["Question"]
            logger.info(f"{query}")
            try:
                response, messages = await client.process_query(query, None)
                logger.info(f"{response}")
                entry["response"] = response
                entry["messages"] = messages
                all_results.append(entry)

            except Exception:
                error_queries.add(query)
                logger.error(traceback.format_exc())
    finally:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        await client.cleanup()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))