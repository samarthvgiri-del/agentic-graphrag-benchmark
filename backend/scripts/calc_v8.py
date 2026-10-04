import sqlite3
import json

db_path = "data/benchmark.db"
run_id = "benchmark-pub20-v8"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def print_metrics():
    print(f"Run ID: {run_id}")

    # 1. Pipeline Success & Retrieval Success
    cursor.execute("""
    SELECT pipeline,
           COUNT(*) as attempts,
           SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
           SUM(CASE WHEN length(retrieved_chunks) > 2 THEN 1 ELSE 0 END) as retrieval_success,
           AVG(latency_ms)/1000 as avg_latency
    FROM benchmark_results
    WHERE run_id = ?
    GROUP BY pipeline
    """, (run_id,))

    print("\n--- Pipeline Stats ---")
    pipelines = cursor.fetchall()
    for row in pipelines:
        print(f"[{row['pipeline']}] Attempts: {row['attempts']} | Success: {row['success']} | Retrieval: {row['retrieval_success']} | Avg Latency: {row['avg_latency']:.2f}s")

    # 2. Evaluation Success & Metrics
    cursor.execute("""
    SELECT b.pipeline,
           COUNT(e.correctness) as eval_success,
           AVG(e.correctness) as avg_correctness,
           AVG(e.completeness) as avg_completeness,
           AVG(e.groundedness) as avg_groundedness
    FROM benchmark_results b
    LEFT JOIN evaluation_results e ON b.result_id = e.result_id
    WHERE b.run_id = ? AND b.status = 'success'
    GROUP BY b.pipeline
    """, (run_id,))

    print("\n--- Evaluation Metrics ---")
    evals = cursor.fetchall()
    for row in evals:
        c = row['avg_correctness'] or 0
        comp = row['avg_completeness'] or 0
        g = row['avg_groundedness'] or 0
        print(f"[{row['pipeline']}] Eval Success: {row['eval_success']} | Correctness: {c:.2f} | Completeness: {comp:.2f} | Groundedness: {g:.2f}")

    # 3. Agentic Specific Metrics
    cursor.execute("""
    SELECT b.question_id, b.question, b.answer, b.error, b.latency_ms/1000 as lat,
           t.complete_trace, t.step_count, t.tools_called
    FROM benchmark_results b
    LEFT JOIN agentic_traces t ON b.result_id = t.result_id
    WHERE b.run_id = ? AND b.pipeline = 'agentic'
    """, (run_id,))

    agentic_rows = cursor.fetchall()
    fallbacks = 0
    total_steps = 0
    tools_used = {}
    timeout_count = 0
    genuine_decisions = 0
    total_decisions = 0

    # Trace logic
    repeated_identical = 0
    repeated_changed = 0

    print("\n--- Agentic Traces ---")

    for row in agentic_rows:
        err = row['error'] or ""
        ans = row['answer'] or ""
        if "time" in err.lower() or "time" in ans.lower():
            timeout_count += 1

        trace_str = row['complete_trace']
        if trace_str:
            complete_trace = json.loads(trace_str)
            steps_taken = row['step_count'] or 0
            total_steps += steps_taken

            fallback = any(s.get('fallback_used') for s in complete_trace)
            if fallback:
                fallbacks += 1

            # Tools
            tools = json.loads(row['tools_called'] or "[]")
            for t in tools:
                tools_used[t] = tools_used.get(t, 0) + 1

            print(f"[{row['question_id']}] {' -> '.join(tools)} (Fallback: {fallback})")

            prev_tool = None
            prev_target = None
            for step in complete_trace:
                if step.get('fallback_used'):
                    continue  # don't count fallback as genuine decision

                total_decisions += 1
                genuine_decisions += 1

                curr_tool = step.get('action')
                curr_target = step.get('target')

                if curr_tool == prev_tool:
                    if curr_target == prev_target:
                        repeated_identical += 1
                    else:
                        repeated_changed += 1

                prev_tool = curr_tool
                prev_target = curr_target

    avg_steps = total_steps / len(agentic_rows) if len(agentic_rows) > 0 else 0
    fallback_rate = fallbacks / len(agentic_rows) if len(agentic_rows) > 0 else 0
    gen_rate = genuine_decisions / total_decisions if total_decisions > 0 else 0

    print("\n--- Agentic Deep Dive ---")
    print(f"Questions Processed: {len(agentic_rows)}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"Fallback Rate: {fallbacks}/{len(agentic_rows)} ({fallback_rate:.1%})")
    print(f"Genuine Decisions: {genuine_decisions}/{total_decisions} ({gen_rate:.1%})")
    print(f"Timeout Count: {timeout_count}")
    print(f"Tool Distribution: {tools_used}")
    print(f"Repeated Identical Action: {repeated_identical}")
    print(f"Repeated Changed Target: {repeated_changed}")

if __name__ == "__main__":
    print_metrics()
