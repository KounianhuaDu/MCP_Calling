"""
OpenAI LLMs
"""
# pylint: disable=broad-exception-caught
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Union, Optional, Type, List
from openai import OpenAI, AzureOpenAI
from openai import RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

from mcpuniverse.common.config import BaseConfig
from mcpuniverse.common.context import Context
from .base import BaseLLM

load_dotenv()


@dataclass
class OpenAIConfig(BaseConfig):
    """
    Configuration for OpenAI language models.

    Attributes:
        model_name (str): The name of the OpenAI model to use (default: "gpt-4o").
        api_key (str): The OpenAI API key (default: environment variable OPENAI_API_KEY).
        temperature (float): Controls randomness in output (default: 1.0).
        top_p (float): Controls diversity of output (default: 1.0).
        frequency_penalty (float): Penalizes frequent token use (default: 0.0).
        presence_penalty (float): Penalizes repeated topics (default: 0.0).
        max_completion_tokens (int): Maximum number of tokens in the completion (default: 2048).
        reasoning_effort (str): The reasoning effort to use (default: "medium").
        seed (int): Random seed for reproducibility (default: 12345).
    """
    model_name: str = os.getenv("OPENAI_MODEL_NAME", "Qwen3-8B")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 10000
    reasoning_effort: str = "medium"
    seed: int = 12345

    # 新增：支持 Azure & 代理
    api_type: str = os.getenv("OPENAI_API_TYPE", "").lower()         # 'azure' | ''(默认openai)
    base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()         # 可用于自建代理，形如 http://localhost:8081/v1

    # Azure 专用
    #azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    #azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    #azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
    
    # 兼容老字段名
    @property
    def reasoning(self) -> str:
        return self.reasoning_effort


class OpenAIModel(BaseLLM):
    """
    OpenAI language models.

    This class provides methods to interact with OpenAI's language models,
    including generating responses based on input messages.

    Attributes:
        config_class (Type[OpenAIConfig]): Configuration class for the model.
        alias (str): Alias for the model, used for identification.
    """
    config_class = OpenAIConfig
    alias = "openai"
    env_vars = ["OPENAI_API_KEY"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        #print("hhh")
        #print(f"config.model_name:{config}")
        self.config = OpenAIModel.config_class.load(config)

    def _build_client_and_model(self):
        """
        根据配置返回 (client, model_name_to_use)
        - 普通 OpenAI: client=OpenAI(...), model=cfg.model_name
        - Azure OpenAI: client=AzureOpenAI(...), model=cfg.azure_deployment
        - 若提供 OPENAI_BASE_URL：以 openai 客户端 + base_url 访问代理
        """
        cfg = self.config

        # 优先：Azure
        if cfg.api_type == "azure":
            if not (cfg.azure_endpoint and cfg.azure_deployment and cfg.api_key):
                raise RuntimeError(
                    "Azure 配置不完整：需要 AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT_NAME / OPENAI_API_KEY"
                )
            client = AzureOpenAI(
                azure_endpoint=cfg.azure_endpoint,   # 例如: https://gpt.yunstorm.com
                api_key=cfg.api_key,
                api_version=cfg.azure_api_version,   # 例如: 2025-01-01-preview
            )
            return client, cfg.azure_deployment  # Azure 的 model 传部署名

        # 其次：自定义 Base URL（代理为 OpenAI 形状 /v1/...）
        if cfg.base_url:
            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            #print("Using base model:{cfg.base_url}")
            return client, cfg.model_name

        # 默认：官方 OpenAI
        client = OpenAI(api_key=cfg.api_key)
        return client, cfg.model_name
    
    def _generate(
            self,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Generates content using the OpenAI model.

        Args:
            messages (List[dict[str, str]]): List of message dictionaries,
                each containing 'role' and 'content' keys.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                defining the structure of the desired output. If None, generates
                free-form text.
            **kwargs: Additional keyword arguments including:
                - max_retries (int): Maximum number of retry attempts (default: 5)
                - base_delay (float): Base delay in seconds for exponential backoff (default: 10.0)
                - timeout (int): Request timeout in seconds (default: 60)

        Returns:
            Union[str, PydanticBaseModel, None]: Generated content as a string
                if no response_format is provided, a Pydantic model instance if
                response_format is provided, or None if parsing structured output fails.
                Returns None if all retry attempts fail or non-retryable errors occur.
        """
        max_retries = kwargs.get("max_retries", 5)
        base_delay = kwargs.get("base_delay", 10.0)
        client, runtime_model = self._build_client_and_model()

        for attempt in range(max_retries + 1):
            print(f"Attempt:{attempt}")
            try:
                #client = OpenAI(api_key=self.config.api_key)
                #client, runtime_model = self._build_client_and_model()
                # Models support the 'reasoning_effort' parameter.
                # This set can be extended as new models are introduced.
                '''
                _models_with_reasoning_effort_support = {"gpt-5", "o3", "o4-mini", "gpt-5-high"}
                if any(prefix in self.config.model_name
                       for prefix in _models_with_reasoning_effort_support):
                    kwargs["reasoning_effort"] = self.config.reasoning_effort

                if "high" in self.config.model_name:
                    kwargs["reasoning_effort"] = "high"
                    self.config.model_name = "gpt-5"
                '''
                
                if response_format is None:
                    print("no_response_format")
                    #print(f"messages:{messages}")
                    #print(f"model:{runtime_model}")
                    #print(f"temperature:{self.config.temperature}")
                    chat = client.chat.completions.create(
                        messages=messages,
                        #model=self.config.model_name,
                        model=runtime_model,
                        temperature=self.config.temperature,
                        #response_format={"type": "json_object"},
                    )
                    #print("hhh")
                    #print(chat.choices[0].message.content)
                    # If tools are provided, return the entire response object
                    # so the caller can handle both content and tool_calls
                    if 'tools' in kwargs:
                        return chat
                    # For backward compatibility, return just content when no tools
                    return chat.choices[0].message.content

                #print("have_response_format")
                #print(f"response_format:{response_format}")
                chat = client.beta.chat.completions.parse(
                    messages=messages,
                    #model=self.config.model_name,
                    model=runtime_model,
                    temperature=self.config.temperature,
                    #timeout=int(kwargs.get("timeout", 60)),
                    #top_p=self.config.top_p,
                    #frequency_penalty=self.config.frequency_penalty,
                    #presence_penalty=self.config.presence_penalty,
                    #max_completion_tokens=self.config.max_completion_tokens,
                    #seed=self.config.seed,
                    response_format=response_format,
                    #**kwargs
                )
                # If tools are provided, return the entire response object
                # so the caller can handle both content and tool_calls
                if 'tools' in kwargs:
                    return chat
                # For backward compatibility, return just parsed content when no tools
                return chat.choices[0].message.parsed

            except (RateLimitError, APIError, APITimeoutError) as e:
                if attempt == max_retries:
                    # Last attempt failed, return None instead of raising
                    logging.warning("All %d attempts failed. Last error: %s", max_retries + 1, e)
                    return None

                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                logging.info("Attempt %d failed with error: %s. Retrying in %.1f seconds...",
                           attempt + 1, e, delay)
                time.sleep(delay)

            except Exception as e:
                # For non-retryable errors, return None instead of raising
                logging.error("Non-retryable error occurred: %s", e)
                return None

    def set_context(self, context: Context):
        """
        Set context, e.g., environment variables (API keys).
        """
        super().set_context(context)
        self.config.api_key = context.env.get("OPENAI_API_KEY", self.config.api_key)
