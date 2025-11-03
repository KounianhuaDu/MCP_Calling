import logging
from backoff import on_exception, expo
import os
import requests
import json

logger = logging.getLogger(__name__)

class ChatModel:
    def __init__(
        self,
        model_url=None,
    ):
        self.model_url = model_url

    def chat(self, messages, tools=None, max_tokens=2048, temperature=0.3):
        try:
            headers = {"Content-Type": "application/json"}

            # Separate system messages from conversation messages
            system_message = ""
            conversation_messages = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_message = content
                else:
                    conversation_messages.append(msg)

            # Build base prompt
            prompt = ""

            # Add system message
            if system_message:
                prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"

            # If tools are provided, add tool descriptions and usage instructions
            if tools and len(tools) > 0:
                # Add tool descriptions
                prompt += "<|im_start|>system\n"
                prompt += "You can use the following tools to help users solve problems:\n\n"

                for i, tool in enumerate(tools, 1):
                    func = tool['function']
                    prompt += f"{i}. {func['name']}:\n"
                    prompt += f"   Description: {func['description']}\n"

                    if 'parameters' in func and func['parameters']:
                        prompt += "   Parameters:\n"
                        params = func['parameters']['properties']
                        for param_name, param_info in params.items():
                            prompt += f"     - {param_name}: {param_info.get('type', 'string')}"
                            if 'title' in param_info:
                                prompt += f" ({param_info['title']})"
                            prompt += "\n"
                    prompt += "\n"

                # Add usage instructions
                prompt += """Usage Guide:
1. Analyze the user's request to determine if tool usage is needed
2. If tool usage is required, respond in the following format:
<FUNCTION_CALL>
{
"name": "tool_name",
"arguments": {
    "parameter_name": "parameter_value"
}
}
</FUNCTION_CALL>
3. If no tool is needed, respond directly with a natural language answer
4. Note: You can only respond once, so ensure your answer is complete\n"""
                prompt += "<|im_end|>\n"

            # Add conversation history
            for msg in conversation_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                elif role == "tool":
                    # tool_name = msg.get("name", "tool")  # 如果有 tool 名
                    prompt += f"<|im_start|>tool\n{content}<|im_end|>\n"

            # Add assistant start tag to indicate model response is expected
            prompt += "<|im_start|>assistant\n"

            # Build request data
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["<|im_end|>", "<|im_start|>"]  # Add stop tokens
            }

            # If tools are provided, adjust temperature for more deterministic responses
            if tools and len(tools) > 0:
                data["temperature"] = min(temperature, 0.3)  # Use lower temperature for tool calls

            response = requests.post(self.model_url, headers=headers, json=data)

            if response.status_code == 200:
                try:
                    result = response.json()
                    # print(result['choices'][0]["text"])
                except Exception as e:
                    print(f"解析JSON失败: {e}")
                    print(f"响应内容: {response.text}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")

            # Add response status check
            if response.status_code != 200:
                logger.error(f"Server returned error: {response.status_code}, {response.text}")
                return f"Error: Server returned {response.status_code}"

            return response.json()
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def chat_with_retry(self, message, tools=None, retry=4):
        @on_exception(expo, Exception, max_tries=retry)
        def _chat_with_retry(message, tools):
            return self.chat(messages=message, tools=tools)

        try:
            response = _chat_with_retry(message, tools)
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise e

    def complete_with_retry(self, **args):
        @on_exception(expo, Exception, max_tries=5)
        def _chat_with_retry(**args):
            return self.chat(**args)

        try:
            response = _chat_with_retry(**args)
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise e


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # 使用正确的 API 端点
    # 对于 vLLM，通常使用 /v1/completions 或 /v1/chat/completions
    # 根据您的服务器配置选择正确的端点
    chat_model = ChatModel("http://0.0.0.0:6000/generate")

    # 测试请求
    print(chat_model.chat_with_retry([{"role": "user", "content": "Hello, how are you?"}]))