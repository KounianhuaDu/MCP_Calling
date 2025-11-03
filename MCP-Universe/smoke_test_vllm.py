# 文件名：smoke_test_vllm.py
# 依赖：pip install openai
from openai import OpenAI
import sys

BASE_URL = "http://localhost:8002/v1"   # 如在远端：改为 http://<server-ip>:8000/v1
API_KEY  = "EMPTY"                       # 启动时 --api-key 配置的值；默认可用 "EMPTY"

def main():
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        # ① 列出模型，确认服务可用
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print("[OK] /v1/models 可用：", model_ids[:5], ("... 共%d个" % len(model_ids)))

        # ② 发起一次对话补全
        #   model 名建议使用你 serve 的名字，如 Qwen3-8B-Instruct 或 本地路径名
        model_name = model_ids[0] if model_ids else "Qwen3-8B-Instruct"

        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "\n<|start|>developer<|message|># Instructions\n\nYou are an agent for web searching. If you don't have enough information to answer the question, you can use the google-search tool. If you want to obtain the information with a specific URL, you can use the fetch tool.\n\nYour goal is to reason about the task and use tools to answer it accurately. \nPlease use only the tools that are explicitly defined below. \nAt each step, you can either use a tool or provide a final answer. \nDo **not** ask clarifying questions.\nYour MUST output the final answer within 50 steps. Be aware of the number of steps remaining.\nReturn the final answer in the final channel.\n\n\nYou MUST respond in the following **exact format** with two clearly separated channels:\n\n<|channel|>commentary\n{\"server\": \"<server_name>\", \"tool\": \"<tool_name>\", \"arguments\": { ... }}\n<|channel|>final\n[Your final answer here]\n\nRules:\n- The commentary channel must contain EXACTLY ONE JSON object (one line) representing the tool call.\n- The final channel must contain the final natural language answer.\n- Do NOT include any markdown, explanations, or extra text outside these two channels.\n- Do NOT wrap in code fences or output `json` literal.\n- The two channels must appear in this order and exactly once each.\n\n# Tools\n## functions\n\nnamespace functions {\n\n// --- server: fetch ---\n// Fetches a URL from the internet and optionally extracts its contents as markdown.\n\nAlthough originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.\ntype fetch__fetch = (_: {\n// URL to fetch\nurl: string,\n// Maximum number of characters to return.\nmax_length?: number,\n// On return output starting at this character index, useful if a previous fetch was truncated and more context is required.\nstart_index?: number,\n// Get the actual HTML content if the requested page, without simplification.\nraw?: boolean,\n}) => any;\n\n\n// --- server: google-search ---\n// A tool to execute the Google search and return the top results.\n\n        Args:\n            query: The search query string.\ntype google-search__search = (_: {\nquery: string,\n}) => any;\n\n\n} // namespace functions\n\n<|end|><|start|>user<|message|>I'm looking for someone based on the clues below: - Score 16 goals in 2024-25 season - Score 1 goal in UEFA Champions League 2024-25 season - Score 11 goals in 2021-22 season - Score 2 goals in the EFL Cup of 2020-21 season.<|end|><|start|>assistant<|channel|>analysis<|message|>",
            }],
            temperature=0,
            max_tokens=4096,
        )
        print("\n[OK] /v1/chat/completions 返回：")
        print(resp.choices[0].message.content.strip())

        # ③ 简单输出 token 统计（如 vLLM 版本支持）
        usage = getattr(resp, "usage", None)
        #print(usage)
        if usage:
            print(f"\n[INFO] tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")

        print("\n🎉 烟囱测试通过！")
    except Exception as e:
        print("[ERR] 调用失败：", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
