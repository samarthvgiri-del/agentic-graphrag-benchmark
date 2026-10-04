import json
import re
import logging
import unicodedata
from typing import Dict, Any, List, Optional
from app.llm.base import LLMProvider
from app.models.schemas import EvaluationResultSchema, CorrectnessEval, GroundednessEval

logger = logging.getLogger(__name__)

EVALUATOR_VERSION = "1.1.0"

class Evaluator:
    def __init__(self, llm_client: LLMProvider):
        self.llm_client = llm_client
        self.model_name = getattr(llm_client, 'model', 'unknown')

    def normalize_text(self, text: str) -> str:
        """Normalize string for comparison (lowercase, strip punctuation)."""
        if not text:
            return ""
        text = text.lower().strip()
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace('_', '')
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_json(self, text: str) -> str:
        """Extract JSON block from text if the model included conversational filler."""
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx:end_idx+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        return text

    def check_deterministic_correctness(self, generated_answer: str, gold_answers: List[str]) -> Optional[float]:
        if not generated_answer or not gold_answers:
            return 0.0

        gen_norm = self.normalize_text(generated_answer)

        if "insufficient evidence" in gen_norm or "not provided in the corpus" in gen_norm or "timed out" in gen_norm or "failed" in gen_norm:
            return 0.0

        for gold in gold_answers:
            gold_norm = self.normalize_text(gold)

            # Numeric match
            if gold_norm.isdigit():
                gen_nums = re.findall(r'\b\d+\b', gen_norm)
                if gold_norm in gen_nums and len(gen_nums) == 1:
                    return 1.0
                elif len(gen_nums) == 1 and gold_norm not in gen_nums:
                    return 0.0 # Exactly one number, and it's wrong
                elif len(gen_nums) > 1 and gold_norm not in gen_nums:
                    return 0.0 # Multiple numbers, none are the gold answer
                # If len == 0 (no numbers) or len > 1 (multiple numbers and one is right), let LLM decide
            else:
                # Entity exact inclusion
                if gold_norm in gen_norm:
                    return 1.0

                # Partial entity match (>= 80% words match)
                gold_words = set(gold_norm.split())
                gen_words = set(gen_norm.split())
                if len(gold_words) > 1 and len(gold_words.intersection(gen_words)) / len(gold_words) >= 0.8:
                    return 1.0

        return None # Require LLM evaluation

    async def evaluate_semantic_correctness(self, question: str, generated_answer: str, gold_answers: List[str]) -> Optional[float]:
        prompt = f"""You are an impartial evaluator assessing the correctness of an answer.

Question: {question}
Expected/Gold Answer: {", ".join(gold_answers)}
Generated Answer: {generated_answer}

Is the Generated Answer correct and factually equivalent to the Expected Answer?
OUTPUT EXACTLY AND ONLY A SINGLE JSON OBJECT."""
        for attempt in range(3):
            try:
                resp = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    format=CorrectnessEval.model_json_schema(),
                    think=False,
                    num_predict=800
                )
                clean_json = self.extract_json(resp)
                data = CorrectnessEval.model_validate_json(clean_json)
                return 1.0 if data.correct else 0.0
            except Exception as e:
                logger.error(f"Semantic correctness evaluation failed on attempt {attempt+1}: {e}")
        return None

    def evaluate_completeness(self, generated_answer: str, gold_answers: List[str]) -> float:
        if not generated_answer or not gold_answers:
            return 0.0

        gen_norm = self.normalize_text(generated_answer)
        found = sum(1 for gold in gold_answers if self.normalize_text(gold) in gen_norm)
        return float(found) / len(gold_answers) if gold_answers else 0.0

    async def evaluate_groundedness(self, question: str, generated_answer: str, context_chunks: List[str]) -> Optional[float]:
        if not context_chunks:
            return 0.0 # No evidence = not grounded

        context_str = "\n".join(f"- {c}" for c in context_chunks)
        prompt = f"""You are an impartial evaluator assessing if an answer is grounded in the provided evidence.

Question: {question}
Provided Evidence:
{context_str}

Generated Answer: {generated_answer}

Is the Generated Answer FULLY supported by the Provided Evidence? It must not contain external facts (hallucinations).
OUTPUT EXACTLY AND ONLY A SINGLE JSON OBJECT."""
        for attempt in range(3):
            try:
                resp = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    format=GroundednessEval.model_json_schema(),
                    think=False,
                    num_predict=800
                )
                clean_json = self.extract_json(resp)
                data = GroundednessEval.model_validate_json(clean_json)
                return 1.0 if data.grounded else 0.0
            except Exception as e:
                logger.error(f"Groundedness evaluation failed on attempt {attempt+1}: {e}")
        return None

    async def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        gen_ans = result.get("answer", "")
        pipe_status = result.get("status")

        if not gen_ans or pipe_status != "success":
            return {
                "result_id": result.get("result_id", "unknown"),
                "evaluator_version": EVALUATOR_VERSION,
                "evaluator_model": self.model_name,
                "evaluation_method": "deterministic_failure",
                "status": "pipeline_error",
                "correctness": None,
                "completeness": None,
                "groundedness": None,
                "eval_details": {"reason": "Pipeline failed or no answer generated."}
            }

        question = result.get("question", "")
        gold_answers = result.get("gold_answer", [])
        if isinstance(gold_answers, str):
            try:
                gold_answers = json.loads(gold_answers)
            except:
                gold_answers = [gold_answers]

        method = "deterministic"

        # 1. Correctness
        correctness = self.check_deterministic_correctness(gen_ans, gold_answers)
        if correctness is None:
            method = "hybrid_llm"
            correctness = await self.evaluate_semantic_correctness(question, gen_ans, gold_answers)

        # 2. Completeness (Deterministic)
        completeness = self.evaluate_completeness(gen_ans, gold_answers)

        # 3. Groundedness (Semantic)
        chunks = result.get("retrieved_chunks", [])
        if isinstance(chunks, str):
            try:
                chunks = json.loads(chunks)
            except:
                chunks = []

        context_texts = [c.get("text", "") for c in chunks if isinstance(c, dict) and "text" in c]

        if not context_texts:
            groundedness = 0.0
        else:
            groundedness = await self.evaluate_groundedness(question, gen_ans, context_texts)

        status = "evaluation_success"
        if correctness is None or groundedness is None:
            status = "evaluation_error"

        return {
            "result_id": result.get("result_id", "unknown"),
            "evaluator_version": EVALUATOR_VERSION,
            "evaluator_model": self.model_name,
            "evaluation_method": method,
            "status": status,
            "correctness": correctness,
            "completeness": completeness,
            "groundedness": groundedness,
            "eval_details": {
                "generated_answer": gen_ans,
                "gold_answers": gold_answers,
                "error": "LLM Judge parsing failed" if status == "evaluation_error" else None
            }
        }
