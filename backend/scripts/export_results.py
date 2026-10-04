import sys
import os
import sqlite3
import json
import csv
import argparse
from pathlib import Path

def export_run(run_id: str, out_dir: str):
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Export JSONL raw results
    jsonl_path = os.path.join(out_dir, f"{run_id}_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        rows = conn.execute("SELECT * FROM benchmark_results WHERE run_id = ?", (run_id,)).fetchall()
        for row in rows:
            d = dict(row)
            f.write(json.dumps(d) + "\n")

    # 2. Export CSV summary
    summary_path = os.path.join(out_dir, f"{run_id}_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pipeline", "Total", "Success", "Error", "Timeout", "Avg Latency (ms)", "Avg Correctness", "Avg Completeness", "Avg Groundedness", "Eval Success", "Eval Error"])

        pipelines = ["rag", "graphrag", "agentic"]
        for p in pipelines:
            stats = conn.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN r.status='success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN r.status='error' THEN 1 ELSE 0 END) as errors,
                    SUM(CASE WHEN r.status='timeout' THEN 1 ELSE 0 END) as timeouts,
                    AVG(r.latency_ms) as avg_latency,
                    AVG(e.correctness) as avg_corr,
                    AVG(e.completeness) as avg_comp,
                    AVG(e.groundedness) as avg_ground,
                    SUM(CASE WHEN e.status='success' THEN 1 ELSE 0 END) as eval_success,
                    SUM(CASE WHEN e.status='error' THEN 1 ELSE 0 END) as eval_error
                FROM benchmark_results r
                LEFT JOIN evaluation_results e ON r.result_id = e.result_id
                WHERE r.run_id = ? AND r.pipeline = ?
            ''', (run_id, p)).fetchone()

            writer.writerow([
                p,
                stats["total"],
                stats["successes"] or 0,
                stats["errors"] or 0,
                stats["timeouts"] or 0,
                f"{stats['avg_latency'] or 0:.1f}",
                f"{stats['avg_corr'] or 0:.2f}",
                f"{stats['avg_comp'] or 0:.2f}",
                f"{stats['avg_ground'] or 0:.2f}",
                stats["eval_success"] or 0,
                stats["eval_error"] or 0
            ])

    # 3. Export CSV per-question comparison
    cmp_path = os.path.join(out_dir, f"{run_id}_comparison.csv")
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question ID", "RAG_Status", "RAG_Correct", "GraphRAG_Status", "GraphRAG_Correct", "Agentic_Status", "Agentic_Correct", "Agentic_Steps"])

        qids = [row["question_id"] for row in conn.execute("SELECT DISTINCT question_id FROM benchmark_results WHERE run_id = ? ORDER BY question_id", (run_id,)).fetchall()]

        for qid in qids:
            row_data = [qid]
            for p in pipelines:
                res = conn.execute('''
                    SELECT r.status, e.correctness, r.result_id
                    FROM benchmark_results r
                    LEFT JOIN evaluation_results e ON r.result_id = e.result_id
                    WHERE r.run_id = ? AND r.question_id = ? AND r.pipeline = ?
                ''', (run_id, qid, p)).fetchone()

                status = res["status"] if res else "missing"
                correct = res["correctness"] if res and res["correctness"] is not None else ""
                row_data.extend([status, correct])

                if p == "agentic":
                    if res:
                        trace = conn.execute("SELECT complete_trace FROM agentic_traces WHERE result_id = ?", (res["result_id"],)).fetchone()
                        steps = len(json.loads(trace["complete_trace"])) if trace else 0
                        row_data.append(steps)
                    else:
                        row_data.append(0)

            writer.writerow(row_data)

    # 4. Export Agentic trace data
    trace_path = os.path.join(out_dir, f"{run_id}_traces.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        rows = conn.execute('''
            SELECT r.question_id, t.complete_trace
            FROM agentic_traces t
            JOIN benchmark_results r ON t.result_id = r.result_id
            WHERE r.run_id = ?
        ''', (run_id,)).fetchall()
        for row in rows:
            f.write(json.dumps({"question_id": row["question_id"], "trace": json.loads(row["complete_trace"])}) + "\n")

    print(f"Exported all files for {run_id} to {out_dir}")
    print(f" - {jsonl_path}")
    print(f" - {summary_path}")
    print(f" - {cmp_path}")
    print(f" - {trace_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="../data/exports")
    args = parser.parse_args()
    export_run(args.run_id, os.path.abspath(args.out_dir))
