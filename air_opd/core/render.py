"""Prompt renderers for Socratic-OPSD.

Design (confirmed 2026-06-30):
- Every student attempt is distilled on its own on-policy trajectory ``y_t``.
  The student-side context is exactly the context that generated ``y_t`` (round 1
  = bare problem; round t>1 = guided context with prior wrong attempts + hints).
- Teacher privilege depends on the attempt:
    correct attempt -> reference solution (math) / gold option (sciknoweval)
    wrong attempt   -> the round's hint (hint-opsd style: "use the hint to repair
                       the incorrect approach"); teacher does NOT see the solution.
- Student NEVER sees the gold answer / reference solution.

Math prompts match the official OPSD training convention (no system message;
``"Problem: {p}\\n\\nPlease reason step by step, and put your final answer within
\\boxed{}."``) so training is consistent with OPSD and the AIME evaluator.
SciKnowEval keeps its system-message MCQ format (matches its evaluator).
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from .prompt_profiles import (
    TEACHER_PROFILE_LEGACY,
    normalize_teacher_prompt_profile,
    teacher_key_idea_profile_user_message,
)

SCIKNOWEVAL_SYSTEM_INSTRUCTION = (
    "Given a question and four options, please select the right answer. "
    "Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
    "For the answer, only output the letter corresponding to the correct option "
    "(A, B, C, or D), and nothing else. Do not restate the answer text. "
    'For example, if the answer is "A", just output:\n<answer>\nA\n</answer>\n'
)
# Backwards-compatible export used by eval/sci_eval_vllm.py
SYSTEM_INSTRUCTION = SCIKNOWEVAL_SYSTEM_INSTRUCTION

# OPSD official training/eval math instruction (in the USER message, no system).
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

_GUIDED_INSTRUCTION = (
    "Use the hint(s) above to correct your reasoning and answer again. "
    "Do not simply repeat a previous attempt."
)


def _history_block(attempts: Sequence[str], hints: Sequence[str]) -> str:
    blocks = []
    for t, att in enumerate(attempts):
        blocks.append(f"=== Previous attempt {t + 1} (incorrect) ===\n{str(att).strip()}")
        if t < len(hints):
            blocks.append(f"=== Hint after attempt {t + 1} ===\n{str(hints[t]).strip()}")
    return "\n\n".join(blocks)


def _problem_and_history(problem: str, attempts: Sequence[str], hints: Sequence[str], task_type: str) -> str:
    """Shared base (problem + optional guided history) for student and teacher."""
    head = f"Problem: {str(problem).strip()}" if task_type == "math" else str(problem).strip()
    if not attempts:
        return head
    return f"{head}\n\n{_history_block(attempts, hints)}"


def _trailing_instruction(task_type: str, guided: bool) -> str:
    parts = []
    if guided:
        parts.append(_GUIDED_INSTRUCTION)
    if task_type == "math":
        parts.append(MATH_INSTRUCTION)
    return ("\n\n".join(parts)).strip()


# --------------------------------------------------------------------------- #
# Pure user-message builders (unit-testable)
# --------------------------------------------------------------------------- #
def student_user_message(
    problem: str, attempts: Sequence[str] = (), hints: Sequence[str] = (), task_type: str = "sciknoweval"
) -> str:
    base = _problem_and_history(problem, attempts, hints, task_type)
    instr = _trailing_instruction(task_type, guided=bool(attempts))
    return f"{base}\n\n{instr}".strip() if instr else base


def repair_feedback_user_message(
    key_idea: str,
    repair_number: int,
    task_type: str = "sciknoweval",
) -> str:
    """Build one real user-role correction turn without exposing the answer."""
    prefix = (
        "The previous answer is incorrect."
        if int(repair_number) == 1
        else "The previous answer is still incorrect."
    )
    parts = [
        prefix,
        "Use the following repair guidance, which explains how to correct the reasoning:",
        f"=== Repair Guidance Begin ===\n{str(key_idea or '').strip()}\n=== Repair Guidance End ===",
        "Revise the reasoning and answer again. Do not simply repeat the previous attempt.",
    ]
    if task_type == "math":
        parts.append(MATH_INSTRUCTION)
    return "\n\n".join(parts)


def explicit_edit_user_message(
    problem: str,
    previous_attempt: str,
    key_idea: str,
    task_type: str = "sciknoweval",
) -> str:
    """Shared scheme-2 prompt body for actor generation and teacher scoring.

    The two sides must call this exact builder.  Their only textual difference
    is the Key Idea value: KI1 for the student and consolidated P2 for the
    teacher.  The trusted reference solution is never rendered here.
    """
    problem_text = str(problem or "").strip()
    attempt_text = str(previous_attempt or "").strip()
    idea_text = str(key_idea or "").strip()
    if not problem_text or not attempt_text or not idea_text:
        raise ValueError("explicit edit requires problem, previous attempt, and key idea")
    if task_type != "math":
        raise ValueError("scheme-2 explicit edit is currently frozen for math only")
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Previous incorrect attempt:\n{attempt_text}\n\n"
        f"Key Idea:\n{idea_text}\n\n"
        "Use the Key Idea to correct the previous attempt. Preserve any valid "
        "reasoning, but do not copy the previous attempt unchanged.\n"
        "Write a complete corrected step-by-step solution.\n\n"
        f"{MATH_INSTRUCTION}"
    )


def ahead_repair_user_message(
    problem: str,
    original_response: str,
    repair_state: str,
    task_type: str = "sciknoweval",
) -> str:
    """Stable answer interface for the multiround-ahead method.

    Every repair generation returns to the same original response.  The actor
    sees state S_t and the distillation teacher sees S_(t+1); all other text is
    identical.  The currently supervised response is deliberately absent.
    """
    problem_text = str(problem or "").strip()
    response_text = str(original_response or "").strip()
    state_text = str(repair_state or "").strip()
    if not problem_text or not response_text or not state_text:
        raise ValueError(
            "multiround-ahead repair requires problem, original response, and repair state"
        )
    if task_type != "math":
        raise ValueError("multiround-ahead repair is currently frozen for math only")
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Original response (use only as a draft):\n{response_text}\n\n"
        f"Repair Key Idea:\n{state_text}\n\n"
        "Answer the problem again using the Repair Key Idea and the useful parts "
        "of the original response. Re-derive any affected steps instead of "
        "describing an edit. Write a self-contained solution in a natural "
        "fresh-answer style; do not mention feedback, revision, attempts, or "
        "repair rounds.\n\n"
        f"{MATH_INSTRUCTION}"
    )


def student_repair_messages(
    problem: str,
    attempts: Sequence[str],
    key_ideas: Sequence[str],
    task_type: str = "sciknoweval",
) -> list[dict[str, str]]:
    """Build the structured chat used for a repair generation.

    Keeping this separate from chat-template rendering makes the user-role
    feedback contract directly testable and lets runtime audits record role
    sequences and content hashes without persisting raw trajectories or key
    ideas a second time.
    """
    if len(attempts) != len(key_ideas) or not attempts:
        raise ValueError("repair chat requires one key idea per prior attempt")
    messages: list[dict[str, str]] = []
    if task_type != "math":
        messages.append(
            {"role": "system", "content": SCIKNOWEVAL_SYSTEM_INSTRUCTION}
        )
    messages.append(
        {
            "role": "user",
            "content": student_user_message(problem, (), (), task_type),
        }
    )
    for repair_number, (attempt, key_idea) in enumerate(
        zip(attempts, key_ideas), start=1
    ):
        messages.append({"role": "assistant", "content": str(attempt)})
        messages.append(
            {
                "role": "user",
                "content": repair_feedback_user_message(
                    key_idea, repair_number, task_type
                ),
            }
        )
    return messages


def teacher_delta_repair_messages(
    problem: str,
    attempts: Sequence[str],
    prior_signals: Sequence[str],
    current_delta: str,
    task_type: str = "sciknoweval",
) -> list[dict[str, str]]:
    """Add one teacher-only delta inside the actor's latest feedback turn.

    Scheme 3b preserves the actor's role sequence and every earlier transcript
    turn.  The only teacher privilege is the current ``new_correction``, added
    to the final user feedback message that generated the scored completion.
    """
    messages = student_repair_messages(
        problem, attempts, prior_signals, task_type
    )
    delta = str(current_delta or "").strip()
    if not delta:
        raise ValueError("teacher delta repair requires a current correction")
    marker = (
        "Revise the reasoning and answer again. "
        "Do not simply repeat the previous attempt."
    )
    latest = str(messages[-1]["content"])
    if marker not in latest:
        raise ValueError("latest repair feedback is missing its revision instruction")
    addition = (
        "Use this additional correction for the current remaining defect:\n\n"
        f"=== New Correction Begin ===\n{delta}\n=== New Correction End ==="
    )
    messages[-1] = {
        "role": "user",
        "content": latest.replace(marker, f"{addition}\n\n{marker}", 1),
    }
    return messages


def teacher_solution_user_message(
    problem: str,
    attempts: Sequence[str],
    hints: Sequence[str],
    gold: str,
    teacher_context: str | None = None,
    task_type: str = "sciknoweval",
) -> str:
    """Correct-attempt teacher: privilege = reference solution (math) / gold (sci)."""
    base = _problem_and_history(problem, attempts, hints, task_type)
    if task_type == "math":
        sol = str(teacher_context or "").strip() or "(reference solution unavailable)"
        privilege = (
            "Reference solution (PRIVATE — do not copy verbatim; reason in your own words):\n"
            f"=== Reference Solution Begin ===\n{sol}\n=== Reference Solution End ===\n"
            f"Final answer: {str(gold).strip()}"
        )
        return f"{base}\n\n{privilege}\n\n{MATH_INSTRUCTION}"
    privilege = f"(Private reference for grading only — the correct option is {str(gold).strip()}.)"
    return f"{base}\n\n{privilege}"


def teacher_hint_user_message(
    problem: str,
    attempts: Sequence[str],
    hints: Sequence[str],
    hint: str,
    current_attempt: str,
    task_type: str = "sciknoweval",
) -> str:
    """Wrong-attempt teacher.

    The teacher sees the student's incorrect trajectory ``current_attempt`` plus
    this round's diagnostic ``hint`` (one round of history more than the student)
    and is asked to produce the CORRECTED reasoning. The incorrect trajectory is
    the scored completion, so the JSD pushes down the student's wrong-token
    probabilities at the mistake positions. No reference solution is exposed.
    """
    base = _problem_and_history(problem, attempts, hints, task_type)
    att = str(current_attempt or "").strip()
    h = str(hint or "").strip()
    block = (
        "=== Student's incorrect attempt (to be corrected) ===\n"
        f"{att}\n"
        "=== End of attempt ===\n\n"
        "Use the following hint to repair the first mistake in the attempt above, "
        "WITHOUT revealing the final answer:\n"
        f"=== Hint Begin ===\n{h}\n=== Hint End ===\n\n"
        "Now give the corrected step-by-step reasoning"
    )
    if task_type == "math":
        return f"{base}\n\n{block} and put the final answer in \\boxed{{}}."
    return f"{base}\n\n{block} and select the correct option."


def teacher_key_idea_user_message(
    problem: str,
    key_idea: str,
    current_attempt: str | None = None,
    task_type: str = "sciknoweval",
    *,
    prompt_profile: str | None = None,
    prompt_meta: Mapping[str, Any] | None = None,
) -> str:
    """Wrong-attempt teacher for ROSD-style privilege.

    By default the teacher sees only the original problem plus a private key
    corrective idea. When ``current_attempt`` is supplied, the teacher also sees
    the student's wrong trajectory and is asked to repair it using the key idea.
    """
    resolved_profile = normalize_teacher_prompt_profile(prompt_profile)
    if resolved_profile != TEACHER_PROFILE_LEGACY:
        return teacher_key_idea_profile_user_message(
            problem,
            key_idea,
            current_attempt=current_attempt,
            task_type=task_type,
            profile=resolved_profile,
            prompt_meta=prompt_meta,
            math_instruction=MATH_INSTRUCTION,
        )

    # Default builder inherited from the pre-profile implementation.  Its
    # structure is unchanged; only the student-facing name of the privileged
    # field was renamed from "key idea" to "repair guidance" so that the
    # prompts match the terminology used in the paper.
    base = f"Problem: {str(problem).strip()}" if task_type == "math" else str(problem).strip()
    idea = str(key_idea or "").strip()
    attempt = str(current_attempt or "").strip()
    attempt_block = ""
    if attempt:
        attempt_block = (
            "Student's incorrect attempt to repair:\n"
            f"=== Incorrect Attempt Begin ===\n{attempt}\n=== Incorrect Attempt End ===\n\n"
        )
    instruction = (
        "Use the repair guidance to repair the incorrect attempt and reason from the problem conditions."
        if attempt
        else "Use the repair guidance to reason from the problem conditions."
    )
    privilege = (
        attempt_block +
        "Repair guidance (PRIVATE reflection, do not quote as an answer):\n"
        f"=== Repair Guidance Begin ===\n{idea}\n=== Repair Guidance End ===\n\n"
        f"{instruction}"
    )
    if task_type == "math":
        return f"{base}\n\n{privilege}\n\n{MATH_INSTRUCTION}"
    return f"{base}\n\n{privilege}"


class Renderers:
    """Tokenizer-backed renderers used by the trainer / repair loop.

    Math: no system message (OPSD-train convention). SciKnowEval: system message
    carrying the strict MCQ format instruction.
    """

    def __init__(
        self,
        tokenizer,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        task_type: str = "sciknoweval",
        teacher_key_idea_prompt_profile: str | None = None,
    ):
        self.tok = tokenizer
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.task_type = task_type
        self.teacher_key_idea_prompt_profile = normalize_teacher_prompt_profile(
            teacher_key_idea_prompt_profile
            if teacher_key_idea_prompt_profile is not None
            else os.environ.get("SOCRATIC_TEACHER_KEY_IDEA_PROMPT_PROFILE")
        )
        # Repair loops use this capability marker to remain compatible with
        # historical/fake renderers that accept only the three legacy args.
        self.supports_teacher_prompt_profiles = True

    def _chat(self, user_content: str, thinking: bool) -> str:
        messages = []
        if self.task_type != "math":
            messages.append({"role": "system", "content": SCIKNOWEVAL_SYSTEM_INSTRUCTION})
        messages.append({"role": "user", "content": user_content})
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )

    def render_student(self, problem: str, attempts: Sequence[str] = (), hints: Sequence[str] = ()) -> str:
        return self._chat(student_user_message(problem, attempts, hints, self.task_type), self.student_thinking)

    def render_student_key_idea(
        self,
        problem: str,
        key_idea: str,
        *,
        prompt_meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Fresh-regeneration actor prompt: problem plus one Key Idea only."""
        user = teacher_key_idea_user_message(
            problem,
            key_idea,
            current_attempt=None,
            task_type=self.task_type,
            prompt_profile=self.teacher_key_idea_prompt_profile,
            prompt_meta=prompt_meta,
        )
        return self._chat(user, self.student_thinking)

    def render_student_repair(
        self,
        problem: str,
        attempts: Sequence[str],
        key_ideas: Sequence[str],
    ) -> str:
        """Render assistant attempts and corrective feedback as real chat turns."""
        messages = self.student_repair_messages(problem, attempts, key_ideas)
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.student_thinking,
        )

    def render_teacher_repair(
        self,
        problem: str,
        attempts: Sequence[str],
        key_ideas: Sequence[str],
    ) -> str:
        """Render the actor's repair chat with the teacher thinking setting.

        Scheme 3 keeps every role and every non-privileged field fixed. The
        caller supplies KI1 on the student side and consolidated P2 on the
        teacher side; the reference solution is never rendered here.
        """
        messages = self.student_repair_messages(problem, attempts, key_ideas)
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )

    def render_teacher_delta_repair(
        self,
        problem: str,
        attempts: Sequence[str],
        prior_signals: Sequence[str],
        current_delta: str,
    ) -> str:
        """Render Scheme 3b with one extra correction in the latest user turn."""
        messages = teacher_delta_repair_messages(
            problem,
            attempts,
            prior_signals,
            current_delta,
            self.task_type,
        )
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )

    def render_student_explicit_edit(
        self,
        problem: str,
        previous_attempt: str,
        key_idea: str,
    ) -> str:
        return self._chat(
            explicit_edit_user_message(
                problem, previous_attempt, key_idea, self.task_type
            ),
            self.student_thinking,
        )

    def render_teacher_explicit_edit(
        self,
        problem: str,
        previous_attempt: str,
        key_idea: str,
    ) -> str:
        return self._chat(
            explicit_edit_user_message(
                problem, previous_attempt, key_idea, self.task_type
            ),
            self.teacher_thinking,
        )

    def render_student_ahead_repair(
        self,
        problem: str,
        original_response: str,
        repair_state: str,
    ) -> str:
        return self._chat(
            ahead_repair_user_message(
                problem, original_response, repair_state, self.task_type
            ),
            self.student_thinking,
        )

    def render_teacher_ahead_repair(
        self,
        problem: str,
        original_response: str,
        repair_state: str,
    ) -> str:
        return self._chat(
            ahead_repair_user_message(
                problem, original_response, repair_state, self.task_type
            ),
            self.teacher_thinking,
        )

    def student_repair_messages(
        self,
        problem: str,
        attempts: Sequence[str],
        key_ideas: Sequence[str],
    ) -> list[dict[str, str]]:
        return student_repair_messages(
            problem, attempts, key_ideas, self.task_type
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tok(str(text), add_special_tokens=False)["input_ids"])

    def render_teacher_solution(
        self,
        problem: str,
        attempts: Sequence[str],
        hints: Sequence[str],
        gold: str,
        teacher_context: str | None = None,
    ) -> str:
        return self._chat(
            teacher_solution_user_message(problem, attempts, hints, gold, teacher_context, self.task_type),
            self.teacher_thinking,
        )

    def render_teacher_hint(
        self,
        problem: str,
        attempts: Sequence[str],
        hints: Sequence[str],
        hint: str,
        current_attempt: str,
    ) -> str:
        return self._chat(
            teacher_hint_user_message(problem, attempts, hints, hint, current_attempt, self.task_type),
            self.teacher_thinking,
        )

    def render_teacher_key_idea(
        self,
        problem: str,
        key_idea: str,
        current_attempt: str | None = None,
        *,
        prompt_meta: Mapping[str, Any] | None = None,
    ) -> str:
        return self._chat(
            teacher_key_idea_user_message(
                problem,
                key_idea,
                current_attempt,
                self.task_type,
                prompt_profile=self.teacher_key_idea_prompt_profile,
                prompt_meta=prompt_meta,
            ),
            self.teacher_thinking,
        )
