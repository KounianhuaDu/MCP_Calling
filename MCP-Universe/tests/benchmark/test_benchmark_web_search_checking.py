import os
import sys
import unittest
import pytest

from mcpuniverse.tracer.collectors import FileCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks


def _project_root():
    # 确保从任何目录运行都能 import & 找到 configs
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))  # -> MCP-Universe/

class TestBenchmarkRunner(unittest.IsolatedAsyncioTestCase):

    # 移除 @pytest.mark.skip 才会执行；先注释掉：
    # @pytest.mark.skip
    async def test(self):
        # 1) 保证 PYTHONPATH 指到项目根（import mcpuniverse 不再报错）
        root = _project_root()
        if root not in sys.path:
            sys.path.insert(0, root)

        # 2) 检查搜索 API Key（无则给出清晰提示并跳过，避免 NoneType 错）
        serp = os.getenv("SERP_API_KEY", "").strip()
        if not serp:
            self.skipTest("SERP_API_KEY not set. Set it in .env to run web_search benchmark.")

        # 3) 确保日志目录存在
        os.makedirs(os.path.join(root, "log"), exist_ok=True)

        # 4) Benchmark 配置路径
        # 注意：BenchmarkRunner 的相对路径是相对 mcpuniverse/benchmark/configs/
        # 如果仓库里是 mcpuniverse/benchmark/configs/test/web_search.yaml
        # 这里就应写 "test/web_search.yaml"
        benchmark_cfg_rel = "test/web_search.yaml"

        trace_collector = FileCollector(log_file=os.path.join(root, "log", "web_search.log"))
        benchmark = BenchmarkRunner(benchmark_cfg_rel)

        # 5) 真正运行；打开中间过程打印便于定位
        results = await benchmark.run(
            trace_collector=trace_collector,
            callbacks=get_vprint_callbacks()
        )

        # 6) 健壮性检查：results 结构非空
        self.assertIsInstance(results, list, "BenchmarkRunner should return a list of results.")
        self.assertGreater(len(results), 0, "No benchmark result returned. Check your config & API keys.")

        # 某些失败情况下，task_results 可能不存在/为 None，做防护
        run0 = results[0]
        task_results = getattr(run0, "task_results", None)
        self.assertIsNotNone(task_results, "task_results is None. Likely the run aborted early (tool/LLM failure).")

        # 7) 生成报告（可选）
        report = BenchmarkReport(benchmark, trace_collector=trace_collector)
        report.dump()

        # 8) 友好打印评测结果（避免 NoneType）
        print('=' * 66)
        print('Evaluation Result')
        print('-' * 66)
        for task_name, info in task_results.items():
            print(task_name)
            print('-' * 66)
            eval_results = (info or {}).get("evaluation_results") or []
            if not eval_results:
                print("No evaluation_results. Check tool outputs / evaluator config.")
            for eval_result in eval_results:
                cfg = getattr(eval_result, "config", None)
                passed = getattr(eval_result, "passed", False)
                if cfg:
                    print("func:", getattr(cfg, "func", None))
                    print("op:", getattr(cfg, "op", None))
                    print("op_args:", getattr(cfg, "op_args", None))
                    print("value:", getattr(cfg, "value", None))
                print('Passed?:', "\033[32mTrue\033[0m" if passed else "\033[31mFalse\033[0m")
                print('-' * 66)


if __name__ == "__main__":
    # 允许从项目根目录以及 tests 子目录直接运行
    # 推荐先：export PYTHONPATH=.
    unittest.main()
