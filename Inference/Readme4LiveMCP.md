# Brief pipeline intro

1. Prepare env as in README.md, use npm and uv instead of docker.
2. uv sync --active
3. source .venv/bin/activate
4. run toolcheck iteratively until all the tools are ready
```bash
bash ./tools/scripts/tool_check.sh
```
5. Index server (generate summary and embedding for the servers)
```bash
uv run -m baseline.mcp_copilot.arg_generation
```
OR
use `LiveMCP/baseline/mcp_copilot/config/mcp_arg_Qwen3-Embedding-0.6B_qwen25_72b_int4_instruct.json` from the original repo directly.

Note: If you find some packages are lost, use
```bash
uv pip install [package_name]
```
6. Example run:
Deploy both inference model and embedding model with vLLM:
```bash
bash ./vllm.sh
bash ./vllm_emb.sh
```
Then run
```bash
bash ./baseline/scripts/run_example.sh
```
7. All tasks run:
```bash
bash ./vllm.sh
bash ./vllm_emb.sh
bash ./baseline/scripts/run_baslines.sh
```
8. Evaluate
Run
```bash
bash ./vllm.sh
bash ./evaluator/scripts/run_baseline.sh
```
for model judge.
Run
```bash
python ./evaluator/stat_success_rate.py --result_path /path/to/evaluation/
```
to get final success rate.

**Note**: Uv commands are not used in shell files above.

