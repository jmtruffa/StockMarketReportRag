# debug_logger.py
# Comprehensive debugging / tracing for the report-generation pipeline.
#
# Captures the full "conversation" between the *writer* agent (LLM that drafts
# the paragraph) and the *evaluator* agent (LLM that scores it), so the user
# can inspect every intermediate step.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class IterationRecord:
    """One round of generate → evaluate."""
    iteration: int
    # Writer agent
    writer_prompt_system: str          # system message sent to writer
    writer_prompt_user: str            # user message sent to writer (may include feedback)
    writer_response: str               # generated paragraph
    # Evaluator agent
    evaluator_prompt: str = ""         # full prompt sent to evaluator
    evaluator_raw_response: str = ""   # raw text response from evaluator
    eval_score: float = 0.0
    eval_ok: bool = False
    eval_reason: str = ""
    # Meta
    timestamp: str = ""


@dataclass
class DebugSession:
    """Holds all iteration records for one report-generation run."""
    session_id: str = ""
    market_id: str = "US"
    target_date: str = ""
    started_at: str = ""
    finished_at: str = ""
    final_score: float = 0.0
    final_answer: str = ""
    total_iterations: int = 0
    iterations: List[IterationRecord] = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────
    def start(self, market_id: str = "US", target_date: str = ""):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.market_id = market_id
        self.target_date = target_date
        self.started_at = datetime.now().isoformat()

    def add_iteration(
        self,
        iteration: int,
        writer_system: str,
        writer_user: str,
        writer_response: str,
        evaluator_prompt: str = "",
        evaluator_raw: str = "",
        eval_score: float = 0.0,
        eval_ok: bool = False,
        eval_reason: str = "",
    ) -> IterationRecord:
        rec = IterationRecord(
            iteration=iteration,
            writer_prompt_system=writer_system,
            writer_prompt_user=writer_user,
            writer_response=writer_response,
            evaluator_prompt=evaluator_prompt,
            evaluator_raw_response=evaluator_raw,
            eval_score=eval_score,
            eval_ok=eval_ok,
            eval_reason=eval_reason,
            timestamp=datetime.now().isoformat(),
        )
        self.iterations.append(rec)
        self.total_iterations = len(self.iterations)
        return rec

    def finish(self, final_answer: str, final_score: float):
        self.finished_at = datetime.now().isoformat()
        self.final_answer = final_answer
        self.final_score = final_score
