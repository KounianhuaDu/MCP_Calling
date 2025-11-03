import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from tqdm import tqdm

import mcp.types as types
import openai
import requests

import vllm
from vllm import LLM
import torch
import gc
# from .. import openai_client

load_dotenv()

logger = logging.getLogger(__name__)


# Define default paths for config and output files
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DEFAULT_CONFIG_PATH = Path("/home/jxliu/Live-master/tools/test/tools.json")
embedding_model="qwen_0_6"
abstract_model="qwen_8"
DEFAULT_OUTPUT_PATH = Path(
    PROJECT_ROOT / "config" / f"mcp_arg_{embedding_model}_{abstract_model}.json"
)

class McpArgGenerator:
    def __init__(
        self,
        config: List[Dict[str, Any]] | Path,
        output_file: str | Path,
    ):
        self.output_file = Path(output_file)

        if isinstance(config, List):
            self.config = config
        elif isinstance(config, Path):
            if not config.exists():
                raise FileNotFoundError(f"File not exist: {config}")
            with config.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise TypeError("Config must be a dictionary or a Path to a JSON file.")

        '''self.embedding_client = openai.AsyncOpenAI(
            api_key=embedding_api_key, base_url=embedding_api_url
        )'''

        # Lazily instantiate the vllm LLM embedding client to avoid loading the
        # model repeatedly when the MCP server gets restarted or reconnected.
        # The actual LLM instance will be created on first use by
        # `_ensure_embedding_client()`.
        self.embedding_client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        # keep path and cfg to use when creating the client

        self.summary_client_url = os.getenv("ABSTRACT_BASE_URL")
        '''openai.AsyncOpenAI(
            api_key=abstract_api_key, base_url=abstract_api_url
        )'''

    async def _get_embedding(
        self, text: str, model: str = embedding_model
    ) -> List[float]:
        if not text:
            logger.warning("Empty text provided for embedding, returning empty list.")
            return []
        try:
            # vllm LLM.embed is synchronous; call and convert to list
            outputs = self.embedding_client.embeddings.create(model="/ext0/jxliu/models/Qwen3-Embedding-0.6B", input=[text])
            embeddings = torch.tensor(outputs.data[0].embedding)
            return embeddings.numpy().tolist()[0]

        except Exception as e:
            logger.error(f"Embedding Error: {e}")
            return []

    async def _generate_summary(
        self,
        server_name: str,
        server_desc: str,
        tools: List[types.Tool],
        model: str = abstract_model,
    ) -> str:
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools]
        )

        prompt = f"""
You are an expert AI technical writer. Based on the following information about an MCP server, please generate a concise and accurate summary of its core purpose and capabilities.

**Server Name:** {server_name}

**Server Description:** {server_desc}

**Available Tools:**
{tool_descriptions}

Please return only the generated summary text, without any additional titles or preambles.
"""
        try:
            '''response = await self.summary_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical writer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()'''
            input_prompt = "<|im_start|>system\nYou are an expert technical writer.<|im_end|>\n"
            input_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
            input_prompt += "<|im_start|>assistant\n"

            headers = {"Content-Type": "application/json"}

            data = {
                "prompt": input_prompt,  # 这是必需的字段
                "max_tokens": 1024,
                "temperature": 0.2,
                "stop": ["<|im_end|>"]  # 添加停止标记
            }

            response = requests.post(self.summary_client_url, headers=headers, json=data)

            # 添加响应状态检查
            if response.status_code != 200:
                logger.error(f"Server returned error: {response.status_code}, {response.text}")
                return f"Error: Server returned {response.status_code}"

            return response.json()

        except Exception as e:
            logger.error(f"Summary Generation Error for '{server_name}': {e}")
            return f"Error generating summary for {server_name}"

    def _format_tool_parameters(self, tool: types.Tool) -> Dict[str, str]:
        formatted_params = {}
        schema = tool.inputSchema
        if not schema or "properties" not in schema:
            return formatted_params

        properties = schema.get("properties", {})
        required_params = schema.get("required", [])

        for param_name, param_details in properties.items():
            param_type = param_details.get("type", "any")
            param_desc = param_details.get("description", "")

            if param_name not in required_params:
                formatted_params[param_name] = f"(Optional, {param_type}) {param_desc}"
            else:
                formatted_params[param_name] = f"({param_type}) {param_desc}"
        return formatted_params

    async def generate(self) -> None:
        existing_servers_info = []
        existing_server_names = set()

        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        existing_servers_info = content
                        for server_data in existing_servers_info:
                            if "server_name" in server_data:
                                existing_server_names.add(server_data["server_name"])
                        logger.info(
                            f"loaded {len(existing_server_names)} existing server from {self.output_file}."
                        )
                    else:
                        logger.warning(
                            f"{self.output_file} does not contain a valid list of servers. "
                        )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(
                    f"Error reading existing servers from {self.output_file}: {e}"
                )

        all_servers_info = existing_servers_info.copy()
        new_servers_processed_count = 0

        for server in tqdm(self.config):
            server_config = server["config"]["mcpServers"]
            server_name = list(server_config.keys())[0]
            if server_name in existing_server_names:
                continue
            tools = [
                types.Tool(**tool)
                for tool in server["tools"].get(server_name, {}).get("tools", [])
            ]
            server_description = server["description"]
            logger.info(f"Indexing server: {server_name}")
            try:
                server_summary = await self._generate_summary(
                    server_name, server_description, tools
                )
                embedding_tasks = {
                    "server_desc": self._get_embedding(server_description),
                    "server_summary": self._get_embedding(server_summary),
                }
                for i, tool in enumerate(tools):
                    embedding_tasks[f"tool_{i}"] = self._get_embedding(tool.description)

                embeddings_results = await asyncio.gather(*embedding_tasks.values())
                embeddings = dict(zip(embedding_tasks.keys(), embeddings_results))

                formatted_tools = []
                for i, tool in enumerate(tools):
                    formatted_tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "description_embedding": embeddings.get(f"tool_{i}", []),
                            "parameter": self._format_tool_parameters(tool),
                        }
                    )

                server_output = {
                    "server_name": server_name,
                    "server_summary": server_summary,
                    "server_description": server_description,
                    "description_embedding": embeddings.get("server_desc", []),
                    "summary_embedding": embeddings.get("server_summary", []),
                    "tools": formatted_tools,
                }

                all_servers_info.append(server_output)

                try:
                    self.output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.output_file, "w", encoding="utf-8") as f:
                        json.dump(all_servers_info, f, indent=2, ensure_ascii=False)
                    new_servers_processed_count += 1
                except IOError as e:
                    logger.error(
                        f"Error writing to output file {self.output_file}: {e}"
                    )

            except Exception as e:
                logger.error(f"Error processing server '{server_name}': {e}")
                continue
        logger.info("Indexing completed.")
        if new_servers_processed_count > 0:
            logger.info(
                f"Add {new_servers_processed_count} new servers to {self.output_file}."
            )
        else:
            logger.info("No new servers were added.")


async def run_generation():
    try:
        generator = McpArgGenerator(
            config=DEFAULT_CONFIG_PATH, output_file=DEFAULT_OUTPUT_PATH
        )
        await generator.generate()
    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Error initializing McpArgGenerator: {e}")


if __name__ == "__main__":
    asyncio.run(run_generation())