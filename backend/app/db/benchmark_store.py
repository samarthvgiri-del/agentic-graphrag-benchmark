"""
SQLite benchmark store for Phase 6A.

Manages persistent storage of benchmark runs, results, and agentic traces.
"""

import sqlite3
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark.db"


def _generate_id() -> str:
    return str(uuid.uuid4())[:12]


class BenchmarkStore:
    """SQLite-backed store for benchmark results."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model TEXT,
                embedding_model TEXT,
                qdrant_collection TEXT,
                tigergraph_graph TEXT,
                agent_limits TEXT,
                inference_settings TEXT,
                question_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            );

            CREATE TABLE IF NOT EXISTS benchmark_results (
                result_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question TEXT NOT NULL,
                question_type TEXT,
                pipeline TEXT NOT NULL,
                answer TEXT,
                gold_answer TEXT,
                gold_doc_ids TEXT,
                citations TEXT,
                retrieved_chunks TEXT,
                selected_context TEXT,
                latency_ms INTEGER,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES benchmark_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS agentic_traces (
                trace_id TEXT PRIMARY KEY,
                result_id TEXT NOT NULL,
                step_count INTEGER,
                tools_called TEXT,
                strategy_changes TEXT,
                evidence_count INTEGER,
                citation_count INTEGER,
                stop_reason TEXT,
                complete_trace TEXT,
                tokens_per_operation TEXT,
                latency_per_operation TEXT,
                FOREIGN KEY(result_id) REFERENCES benchmark_results(result_id)
            );

            CREATE TABLE IF NOT EXISTS evaluation_results (
                eval_id TEXT PRIMARY KEY,
                result_id TEXT NOT NULL,
                evaluator_version TEXT NOT NULL,
                evaluator_model TEXT,
                evaluation_method TEXT NOT NULL,
                status TEXT,
                correctness REAL,
                completeness REAL,
                groundedness REAL,
                eval_details TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(result_id) REFERENCES benchmark_results(result_id)
            );

            CREATE INDEX IF NOT EXISTS idx_results_run_id ON benchmark_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_results_question_id ON benchmark_results(question_id);
            CREATE INDEX IF NOT EXISTS idx_results_pipeline ON benchmark_results(pipeline);
            CREATE INDEX IF NOT EXISTS idx_traces_result_id ON agentic_traces(result_id);
            CREATE INDEX IF NOT EXISTS idx_eval_result_id ON evaluation_results(result_id);
        """)

        # Safe migrations for new columns
        columns_to_add = {
            "benchmark_results": [
                "extracted_phrases TEXT",
                "linked_entities TEXT",
                "graph_nodes_returned INTEGER DEFAULT 0",
                "retrieved_chunk_count INTEGER DEFAULT 0",
                "unique_source_count INTEGER DEFAULT 0",
                "orchestrator_fallback_used INTEGER DEFAULT 0"
            ]
        }

        for table, columns in columns_to_add.items():
            for col_def in columns:
                col_name = col_def.split()[0]
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                except sqlite3.OperationalError as e:
                    # Column already exists
                    pass

        self.conn.commit()

    # ------------------------------------------------------------------
    # Benchmark Runs
    # ------------------------------------------------------------------
    def create_run(self, config: Dict[str, Any]) -> str:
        """Create a new benchmark run and return its ID."""
        run_id = f"run-{_generate_id()}"
        self.conn.execute(
            """INSERT INTO benchmark_runs
               (run_id, timestamp, model, embedding_model, qdrant_collection,
                tigergraph_graph, agent_limits, inference_settings, question_count, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                config.get("model"),
                config.get("embedding_model"),
                config.get("qdrant_collection"),
                config.get("tigergraph_graph"),
                json.dumps(config.get("agent_limits", {})),
                json.dumps(config.get("inference_settings", {})),
                config.get("question_count", 0),
                "running",
            ),
        )
        self.conn.commit()
        logger.info("Created benchmark run %s", run_id)
        return run_id

    def finish_run(self, run_id: str, status: str = "completed"):
        self.conn.execute(
            "UPDATE benchmark_runs SET status = ? WHERE run_id = ?",
            (status, run_id),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def save_result(self, result: Dict[str, Any]) -> str:
        """Save a single pipeline result."""
        result_id = f"res-{_generate_id()}"
        self.conn.execute(
            """INSERT INTO benchmark_results
               (result_id, run_id, question_id, question, question_type, pipeline,
                answer, gold_answer, gold_doc_ids, citations, retrieved_chunks,
                selected_context, latency_ms, input_tokens, output_tokens,
                total_tokens, status, error, timestamp,
                extracted_phrases, linked_entities, graph_nodes_returned,
                retrieved_chunk_count, unique_source_count, orchestrator_fallback_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result_id,
                result["run_id"],
                result["question_id"],
                result["question"],
                result.get("question_type"),
                result["pipeline"],
                result.get("answer"),
                json.dumps(result.get("gold_answer")),
                json.dumps(result.get("gold_doc_ids")),
                json.dumps(result.get("citations", [])),
                json.dumps(result.get("retrieved_chunks", [])),
                result.get("selected_context"),
                result.get("latency_ms"),
                result.get("input_tokens", 0),
                result.get("output_tokens", 0),
                result.get("total_tokens", 0),
                result["status"],
                result.get("error"),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(result.get("extracted_phrases", [])),
                json.dumps(result.get("linked_entities", [])),
                result.get("graph_nodes_returned", 0),
                result.get("retrieved_chunk_count", 0),
                result.get("unique_source_count", 0),
                1 if result.get("orchestrator_fallback_used", False) else 0
            ),
        )
        self.conn.commit()
        return result_id

    def get_results_for_run(
        self, run_id: str, pipeline: Optional[str] = None
    ) -> List[Dict]:
        if pipeline:
            rows = self.conn.execute(
                "SELECT * FROM benchmark_results WHERE run_id = ? AND pipeline = ? ORDER BY question_id",
                (run_id, pipeline),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM benchmark_results WHERE run_id = ? ORDER BY question_id, pipeline",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_result(self, result_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM benchmark_results WHERE result_id = ?", (result_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Agentic Traces
    # ------------------------------------------------------------------
    def save_trace(self, result_id: str, trace_data: Dict[str, Any]) -> str:
        trace_id = f"trc-{_generate_id()}"
        self.conn.execute(
            """INSERT INTO agentic_traces
               (trace_id, result_id, step_count, tools_called, strategy_changes,
                evidence_count, citation_count, stop_reason, complete_trace,
                tokens_per_operation, latency_per_operation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                result_id,
                trace_data.get("step_count"),
                json.dumps(trace_data.get("tools_called", [])),
                json.dumps(trace_data.get("strategy_changes", [])),
                trace_data.get("evidence_count"),
                trace_data.get("citation_count"),
                trace_data.get("stop_reason"),
                json.dumps(trace_data.get("complete_trace", {})),
                json.dumps(trace_data.get("tokens_per_operation", [])),
                json.dumps(trace_data.get("latency_per_operation", [])),
            ),
        )
        self.conn.commit()
        return trace_id

    def get_trace_for_result(self, result_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM agentic_traces WHERE result_id = ?", (result_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def save_evaluation(self, eval_data: Dict[str, Any]) -> str:
        eval_id = f"evl-{_generate_id()}"
        self.conn.execute(
            """INSERT INTO evaluation_results
               (eval_id, result_id, evaluator_version, evaluator_model,
                evaluation_method, status, correctness, completeness, groundedness,
                eval_details, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                eval_id,
                eval_data["result_id"],
                eval_data["evaluator_version"],
                eval_data.get("evaluator_model"),
                eval_data["evaluation_method"],
                eval_data.get("status", "success"),
                eval_data.get("correctness"),
                eval_data.get("completeness"),
                eval_data.get("groundedness"),
                json.dumps(eval_data.get("eval_details", {})),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()
        return eval_id

    def get_evaluations_for_result(self, result_id: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM evaluation_results WHERE result_id = ? ORDER BY timestamp DESC",
            (result_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Summary / Aggregation
    # ------------------------------------------------------------------
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Aggregate statistics grouped by pipeline for a run."""
        rows = self.conn.execute(
            """SELECT
                 pipeline,
                 COUNT(*) as total,
                 SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                 SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                 SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeouts,
                 AVG(latency_ms) as avg_latency_ms,
                 AVG(total_tokens) as avg_tokens
               FROM benchmark_results
               WHERE run_id = ?
               GROUP BY pipeline""",
            (run_id,),
        ).fetchall()

        summary = {}
        for row in rows:
            row_dict = dict(row)
            pipe = row_dict.pop("pipeline")

            # Get evaluation averages and counts
            # Get evaluation averages and counts only for successful evaluations
            eval_row = self.conn.execute(
                """SELECT
                     AVG(CASE WHEN e.status = 'evaluation_success' THEN e.correctness ELSE NULL END) as avg_correctness,
                     AVG(CASE WHEN e.status = 'evaluation_success' THEN e.completeness ELSE NULL END) as avg_completeness,
                     AVG(CASE WHEN e.status = 'evaluation_success' THEN e.groundedness ELSE NULL END) as avg_groundedness,
                     SUM(CASE WHEN e.status = 'evaluation_success' THEN 1 ELSE 0 END) as eval_successes,
                     SUM(CASE WHEN e.status IN ('evaluation_error', 'pipeline_error') THEN 1 ELSE 0 END) as eval_errors,
                     SUM(CASE WHEN e.status = 'evaluation_timeout' THEN 1 ELSE 0 END) as eval_timeouts,
                     COUNT(e.result_id) as eval_total
                   FROM evaluation_results e
                   JOIN benchmark_results r ON e.result_id = r.result_id
                   WHERE r.run_id = ? AND r.pipeline = ?""",
                (run_id, pipe),
            ).fetchone()
            if eval_row:
                row_dict.update(dict(eval_row))

            if pipe == "agentic":
                trace_row = self.conn.execute(
                    """SELECT AVG(t.step_count) as avg_agent_steps
                       FROM agentic_traces t
                       JOIN benchmark_results r ON t.result_id = r.result_id
                       WHERE r.run_id = ? AND r.pipeline = 'agentic'""",
                    (run_id,)
                ).fetchone()
                if trace_row and trace_row[0] is not None:
                    row_dict["avg_agent_steps"] = trace_row[0]
            summary[pipe] = row_dict
        return summary

    def close(self):
        self.conn.close()
