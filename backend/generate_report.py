import sqlite3
import json

c = sqlite3.connect('data/benchmark.db')
run_id = 'benchmark-pub-20'

with open('report.md', 'w', encoding='utf-8') as f:
    f.write("### Benchmark Comparison Table\n\n")
    f.write("| Pipeline | Success Rate | Eval Error Rate | Avg Latency (ms) | Avg Correctness | Avg Completeness | Avg Groundedness | Avg Agent Steps |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")

    pipelines = ["rag", "graphrag", "agentic"]

    for p in pipelines:
        rows = c.execute(
            "SELECT status, latency_ms FROM benchmark_results WHERE run_id=? AND pipeline=?",
            (run_id, p)
        ).fetchall()

        if not rows:
            continue

        total = len(rows)
        success = sum(1 for r in rows if r[0] == 'success')
        latencies = [r[1] for r in rows if r[1] is not None]
        latency = sum(latencies) / len(latencies) if latencies else 0

        evals = c.execute(
            "SELECT e.status, e.correctness, e.completeness, e.groundedness FROM evaluation_results e "
            "JOIN benchmark_results r ON e.result_id = r.result_id "
            "WHERE r.run_id=? AND r.pipeline=?", (run_id, p)
        ).fetchall()

        eval_total = len(evals)
        eval_err = sum(1 for e in evals if e[0] == 'error')

        corr = [e[1] for e in evals if e[1] is not None]
        comp = [e[2] for e in evals if e[2] is not None]
        grou = [e[3] for e in evals if e[3] is not None]

        avg_corr = sum(corr)/len(corr) if corr else 0.0
        avg_comp = sum(comp)/len(comp) if comp else 0.0
        avg_grou = sum(grou)/len(grou) if grou else 0.0

        avg_steps = 0
        if p == "agentic":
            traces = c.execute("SELECT t.step_count FROM agentic_traces t JOIN benchmark_results r ON t.result_id = r.result_id WHERE r.run_id=? AND r.pipeline=? AND r.status='success'", (run_id, p)).fetchall()
            steps = [t[0] for t in traces if t[0] is not None]
            avg_steps = sum(steps)/len(steps) if steps else 0

        f.write(f"| {p.upper()} | {success}/{total} ({(success/total)*100:.1f}%) | {eval_err}/{eval_total} ({(eval_err/max(1, eval_total))*100:.1f}%) | {latency:.0f} | {avg_corr:.2f} | {avg_comp:.2f} | {avg_grou:.2f} | {avg_steps:.1f} |\n")

    f.write("\n### Per-Question Analysis\n\n")
    questions = c.execute("SELECT DISTINCT question FROM benchmark_results WHERE run_id=?", (run_id,)).fetchall()
    for q_idx, (q,) in enumerate(questions):
        f.write(f"**Q{q_idx+1}: {q}**\n")
        for p in pipelines:
            res = c.execute("SELECT status, answer, latency_ms FROM benchmark_results WHERE run_id=? AND pipeline=? AND question=?", (run_id, p, q)).fetchone()
            if res:
                status, ans, lat = res
                ans = ans or ""
                ans_clean = ans.replace('\n', ' ')[:150] + ('...' if len(ans) > 150 else '')
                f.write(f"- **{p.upper()}** ({lat}ms) [{status}]: {ans_clean}\n")
        f.write("\n")
