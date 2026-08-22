"""
Pipeline 测试脚本：验证 3 种模式 × pos_neg_method 组合

通过 HTTP API 顺序创建 5 个生成任务，轮询等待完成/失败，汇总打印结果。

测试矩阵:
  1. qa_extraction              - 原始文档(3 docs), 无 embedding
  2. doc_to_training + llm      - 原始文档(3 docs), 无 embedding
  3. doc_to_training + retrieval - 原始文档(3 docs), 有 embedding
  4. qa_to_training  + llm      - QA 数据集,        无 embedding
  5. qa_to_training  + retrieval - QA 数据集,        有 embedding

用法:
  python tests/test_pipelines.py
"""

import time
import sys
import os
import requests

# ── 配置 ──────────────────────────────────────────────
BASE_URL = os.getenv("TEST_API_BASE", "http://localhost:18000/api").rstrip("/")
LLM_CONFIG_ID = "af2200ec-ce65-4620-9a7e-19740515f2c9"
EMBEDDING_CONFIG_ID = "92166c52-182d-445d-b537-2f105bbd8fcc"
QA_DATASET_ID = "022fa895-3ab7-49a8-bc2d-f850d04b3e46"
DOC_PATH = "/workspace/train-factory/data/rag_documents_sample_20.jsonl"

POLL_INTERVAL = 5       # 轮询间隔 (秒)
TIMEOUT = 300           # 超时 (秒)
NUM_TEST_DOCS = 3       # 取前 N 条文档


def prepare_test_docs() -> str:
    """从原始文档中取前 N 条写入项目 data 目录（容器可访问），返回路径。"""
    import os
    out_path = os.path.join(os.path.dirname(DOC_PATH), "test_docs_pipeline.jsonl")
    with open(DOC_PATH, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if i >= NUM_TEST_DOCS:
                break
            out.write(line)
    print(f"[准备] 测试文档: {out_path} ({NUM_TEST_DOCS} 条)")
    return out_path


def build_test_cases(doc_path: str) -> list[dict]:
    """构建 5 个测试用例。"""
    ts = int(time.time())
    return [
        {
            "name": "1. qa_extraction",
            "payload": {
                "task_name": f"test-qa-extract-{ts}",
                "generation_mode": "qa_extraction",
                "input_path": doc_path,
                "input_format": "jsonl",
                "content_field": "doc_content",
                "llm_config": {"config_id": LLM_CONFIG_ID},
                "steps": {
                    "doc_quality": {"enabled": False},
                    "qa_gen": {"enabled": True, "num_qa_per_doc": 2},
                },
                "worker_config": {"concurrency": 2, "timeout_per_doc": 120},
                "auto_register_dataset": False,
            },
        },
        {
            "name": "2. doc_to_training + llm",
            "payload": {
                "task_name": f"test-doc-llm-{ts}",
                "generation_mode": "doc_to_training",
                "pos_neg_method": "llm",
                "input_path": doc_path,
                "input_format": "jsonl",
                "content_field": "doc_content",
                "output_format": "universal",
                "llm_config": {"config_id": LLM_CONFIG_ID},
                "steps": {
                    "doc_quality": {"enabled": False},
                    "qa_gen": {"enabled": True, "num_qa_per_doc": 2},
                    "pos_neg_extraction": {
                        "enabled": True,
                        "num_positive": 3,
                        "num_negative": 5,
                    },
                },
                "worker_config": {"concurrency": 2, "timeout_per_doc": 120},
                "auto_register_dataset": False,
            },
        },
        {
            "name": "3. doc_to_training + retrieval",
            "payload": {
                "task_name": f"test-doc-retrieval-{ts}",
                "generation_mode": "doc_to_training",
                "pos_neg_method": "retrieval",
                "input_path": doc_path,
                "input_format": "jsonl",
                "content_field": "doc_content",
                "output_format": "universal",
                "llm_config": {"config_id": LLM_CONFIG_ID},
                "embedding_config": {"config_id": EMBEDDING_CONFIG_ID},
                "similarity_threshold": 0.85,
                "retrieval_top_k": 5,
                "steps": {
                    "doc_quality": {"enabled": False},
                    "qa_gen": {"enabled": True, "num_qa_per_doc": 2},
                    "pos_neg_extraction": {
                        "enabled": True,
                        "num_positive": 3,
                        "num_negative": 5,
                    },
                },
                "worker_config": {"concurrency": 2, "timeout_per_doc": 120},
                "auto_register_dataset": False,
            },
        },
        {
            "name": "4. qa_to_training + llm",
            "payload": {
                "task_name": f"test-qa-llm-{ts}",
                "generation_mode": "qa_to_training",
                "pos_neg_method": "llm",
                "dataset_id": QA_DATASET_ID,
                "output_format": "universal",
                "llm_config": {"config_id": LLM_CONFIG_ID},
                "steps": {
                    "pos_neg_extraction": {
                        "enabled": True,
                        "num_positive": 3,
                        "num_negative": 5,
                    },
                },
                "worker_config": {"concurrency": 2, "timeout_per_doc": 120},
                "auto_register_dataset": False,
            },
        },
        {
            "name": "5. qa_to_training + retrieval",
            "payload": {
                "task_name": f"test-qa-retrieval-{ts}",
                "generation_mode": "qa_to_training",
                "pos_neg_method": "retrieval",
                "dataset_id": QA_DATASET_ID,
                "output_format": "universal",
                "llm_config": {"config_id": LLM_CONFIG_ID},
                "embedding_config": {"config_id": EMBEDDING_CONFIG_ID},
                "similarity_threshold": 0.85,
                "retrieval_top_k": 5,
                "steps": {
                    "pos_neg_extraction": {
                        "enabled": True,
                        "num_positive": 3,
                        "num_negative": 5,
                    },
                },
                "worker_config": {"concurrency": 2, "timeout_per_doc": 120},
                "auto_register_dataset": False,
            },
        },
    ]


def create_task(payload: dict) -> str | None:
    """创建任务，返回 task_id。"""
    resp = requests.post(f"{BASE_URL}/generation/tasks", json=payload, timeout=30)
    if resp.status_code != 200:
        print(f"  [错误] 创建失败: {resp.status_code} {resp.text[:300]}")
        return None
    data = resp.json()
    task_id = data.get("task_id")
    print(f"  [创建] task_id={task_id}")
    return task_id


def poll_task(task_id: str) -> dict:
    """轮询等待任务完成，返回最终状态信息。"""
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > TIMEOUT:
            return {"status": "timeout", "elapsed": elapsed}

        resp = requests.get(f"{BASE_URL}/generation/tasks/{task_id}", timeout=15)
        if resp.status_code != 200:
            return {"status": "poll_error", "error": resp.text[:200], "elapsed": elapsed}

        data = resp.json()
        status = data.get("status", "unknown")
        progress = data.get("progress", 0)
        output_count = data.get("output_sample_count", 0)

        print(
            f"  [{elapsed:5.0f}s] status={status}  progress={progress:.0f}%  samples={output_count}",
            end="\r",
        )

        if status in ("completed", "failed", "stopped"):
            print()  # 换行
            return {
                "status": status,
                "progress": progress,
                "output_sample_count": output_count,
                "error_message": data.get("error_message"),
                "elapsed": elapsed,
            }

        time.sleep(POLL_INTERVAL)


def main():
    print("=" * 70)
    print("Pipeline 测试 — 3 模式 × pos_neg_method")
    print("=" * 70)

    # 1. 准备数据
    doc_path = prepare_test_docs()
    test_cases = build_test_cases(doc_path)

    # 2. 逐个运行
    results = []
    for tc in test_cases:
        print(f"\n{'─' * 60}")
        print(f"▶ {tc['name']}")
        print(f"{'─' * 60}")

        task_id = create_task(tc["payload"])
        if not task_id:
            results.append({"name": tc["name"], "status": "create_failed"})
            continue

        result = poll_task(task_id)
        result["name"] = tc["name"]
        result["task_id"] = task_id
        results.append(result)

    # 3. 汇总
    print(f"\n{'=' * 70}")
    print("汇总结果")
    print(f"{'=' * 70}")
    header = f"{'#':<3} {'测试用例':<35} {'状态':<12} {'样本数':<8} {'耗时(s)':<8}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for i, r in enumerate(results, 1):
        status = r.get("status", "?")
        samples = r.get("output_sample_count", "-")
        elapsed = r.get("elapsed", 0)
        mark = "✓" if status == "completed" else "✗"
        if status != "completed":
            all_ok = False
        print(f"{i:<3} {r['name']:<35} {mark} {status:<10} {str(samples):<8} {elapsed:<8.1f}")
        if r.get("error_message"):
            print(f"    └─ error: {r['error_message'][:120]}")

    print(f"\n{'全部通过 ✓' if all_ok else '存在失败 ✗'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
