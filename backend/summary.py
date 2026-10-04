import sqlite3
import sys
import os

db_path = os.path.join(os.path.dirname(__file__), "data", "benchmark.db")
run_id = sys.argv[1] if len(sys.argv) > 1 else "benchmark-val-10-v3"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
    SELECT
        r.pipeline,
        COUNT(r.result_id) as attempts,
        SUM(CASE WHEN r.status = 'success' THEN 1 ELSE 0 END) as successes,
        SUM(CASE WHEN e.status = 'error' THEN 1 ELSE 0 END) as eval_errors,
        AVG(r.latency_ms) as avg_latency,
        SUM(CASE WHEN r.retrieved_chunk_count > 0 THEN 1 ELSE 0 END) as answerable_retrieved,
        SUM(CASE WHEN r.orchestrator_fallback_used = 1 THEN 1 ELSE 0 END) as fallbacks_used,
        AVG(e.correctness) as avg_correctness,
        AVG(e.completeness) as avg_completeness,
        AVG(e.groundedness) as avg_groundedness
    FROM benchmark_results r
    LEFT JOIN evaluation_results e ON r.result_id = e.result_id
    WHERE r.run_id = ?
    GROUP BY r.pipeline
""", (run_id,))

rows = cursor.fetchall()
print(f"--- Benchmark Summary ({run_id}) ---")
for row in rows:
    attempts = row['attempts']
    successes = row['successes']
    eval_errors = row['eval_errors']

    avg_latency = row['avg_latency']
    answerable_retrieved = row['answerable_retrieved'] or 0
    fallbacks_used = row['fallbacks_used'] or 0

    avg_corr = row['avg_correctness']
    avg_comp = row['avg_completeness']
    avg_grnd = row['avg_groundedness']

    print(f"Pipeline: {row['pipeline'].upper()}")
    print(f"  Attempts: {attempts}")
    print(f"  Pipeline Success: {successes}/{attempts} ({(successes/attempts)*100:.1f}%)")
    print(f"  Evaluator Errors: {eval_errors}/{attempts} (excluded from scoring)")

    if avg_latency:
        print(f"  Avg Latency: {avg_latency / 1000:.2f}s")
    else:
        print(f"  Avg Latency: N/A")

    print(f"  Answerable Retrieval Rate: {answerable_retrieved}/{attempts} ({(answerable_retrieved/attempts)*100:.1f}%)")
    if row['pipeline'] == 'agentic':
        print(f"  Agentic Fallbacks Triggered: {fallbacks_used}/{attempts} ({(fallbacks_used/attempts)*100:.1f}%)")

    # Valid evaluations only
    valid_evals = attempts - eval_errors
    if valid_evals > 0:
        print(f"  Avg Correctness: {avg_corr:.2f}" if avg_corr is not None else "  Avg Correctness: N/A")
        print(f"  Avg Completeness: {avg_comp:.2f}" if avg_comp is not None else "  Avg Completeness: N/A")
        print(f"  Avg Groundedness: {avg_grnd:.2f}" if avg_grnd is not None else "  Avg Groundedness: N/A")
    else:
        print(f"  (All evaluations failed)")
    print()
