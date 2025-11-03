#!/bin/bash

OUTPUT_DIR="./baseline/output"

# for file in "$OUTPUT_DIR"/*.json; do
#   echo "Running evaluator on: $file"
#   uv run -m evaluator.llm_as_judge_baseline \
#     --trajectory_path "$file"
# done
file="./baseline/output/Qwen3-8B_Qwen3-Embedding-0.6B.json"
python -m evaluator.llm_as_judge_baseline \
    --trajectory_path "$file"