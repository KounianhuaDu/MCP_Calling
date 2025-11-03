import asyncio
from mcpuniverse.tracer.collectors import MemoryCollector  # You can also use SQLiteCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner

async def test():
    trace_collector = MemoryCollector()
    # Choose a benchmark config file under the folder "mcpuniverse/benchmark/configs"
    benchmark = BenchmarkRunner("dummy/benchmark_2.yaml")
    # Run the specified benchmark
    results = await benchmark.run(trace_collector=trace_collector)
    # Get traces
    trace_id = results[0].task_trace_ids["dummy/tasks/weather_1.json"]
    trace_records = trace_collector.get(trace_id)
    print("\n=== Benchmark 运行结果 ===")
    print(f"任务总数: {len(results)}")
    print(f"第一个任务 trace_id: {trace_id}")
    print(f"轨迹记录条数: {len(trace_records)}")
    print("第一条轨迹：", trace_records[0])

if __name__ == "__main__":
    asyncio.run(test())