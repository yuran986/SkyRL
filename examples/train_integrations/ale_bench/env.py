from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


ALE_BENCH_REPO = os.getenv("ALE_BENCH_REPO")
if ALE_BENCH_REPO:
    ale_src = Path(ALE_BENCH_REPO).expanduser() / "src"
    if ale_src.is_dir() and str(ale_src) not in sys.path:
        sys.path.insert(0, str(ale_src))


LANGUAGE_ALIASES = {
    "c++": "cpp20",
    "c++17": "cpp17",
    "c++20": "cpp20",
    "c++23": "cpp23",
    "cpp": "cpp20",
    "cpp17": "cpp17",
    "cpp20": "cpp20",
    "cpp23": "cpp23",
    "python": "python",
    "py": "python",
    "pypy": "pypy",
    "rust": "rust",
    "rs": "rust",
}

CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)\s*\n(.*?)```", re.DOTALL)
TAG_RE = re.compile(
    r"<(?:solution|code)(?:\s+language=[\"']?([^\"'>\s]+)[\"']?)?\s*>(.*?)</(?:solution|code)>",
    re.DOTALL | re.IGNORECASE,
)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_language(value: Any, default: str) -> str:
    if value is None or value == "":
        return default
    raw = str(value).strip().lower()
    return LANGUAGE_ALIASES.get(raw, raw)


def extract_code_and_language(output: str, default_language: str) -> tuple[str, str]:
    """Extract one source file from a model response."""
    tag_match = TAG_RE.search(output)
    if tag_match:
        language = _normalize_language(tag_match.group(1), default_language)
        return html.unescape(tag_match.group(2)).strip(), language

    fence_matches = CODE_FENCE_RE.findall(output)
    if fence_matches:
        language, code = fence_matches[-1]
        return code.strip(), _normalize_language(language, default_language)

    stripped = output.strip()
    if not stripped:
        raise ValueError("empty model output; expected a source code block")
    return stripped, default_language


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... truncated {len(text) - max_chars} chars"


def _result_summary(result: Any, case_limit: int) -> str:
    judge_result = getattr(result, "overall_judge_result", None)
    judge_value = getattr(judge_result, "value", judge_result)
    case_lines = []
    for idx, case in enumerate(list(getattr(result, "case_results", []))[:case_limit]):
        case_lines.append(
            "case[{idx}] judge={judge} score={score} time={time:.3f}s memory={memory}".format(
                idx=idx,
                judge=getattr(getattr(case, "judge_result", None), "value", getattr(case, "judge_result", None)),
                score=getattr(case, "absolute_score", None),
                time=float(getattr(case, "execution_time", 0.0) or 0.0),
                memory=getattr(case, "memory_usage", None),
            )
        )
    if len(getattr(result, "case_results", [])) > case_limit:
        case_lines.append(f"... {len(result.case_results) - case_limit} more cases")
    return "\n".join(
        [
            f"judge={judge_value}",
            f"absolute_score={getattr(result, 'overall_absolute_score', None)}",
            f"relative_score={getattr(result, 'overall_relative_score', None)}",
            *case_lines,
        ]
    )


class AleBenchEnv(BaseTextEnv):
    """SkyRL text environment around ALE-Bench public evaluation."""

    def __init__(self, env_config: Any, extras: dict[str, Any] | None = None):
        super().__init__()
        self.env_config = env_config or {}
        self.extras = extras or {}
        self.problem_id = str(self.extras.get("problem_id", os.getenv("ALE_BENCH_PROBLEM_ID", "ahc001")))
        self.lite_version = _to_bool(self.extras.get("lite_version", os.getenv("ALE_BENCH_LITE_VERSION")), True)
        self.max_turns = int(self.extras.get("max_turns", self.extras.get("max_steps", 1)))
        self.code_language = _normalize_language(self.extras.get("code_language"), "cpp20")
        self.judge_version = str(self.extras.get("judge_version", "202301"))
        self.num_workers = int(self.extras.get("num_workers", os.getenv("ALE_BENCH_NUM_WORKERS", "1")))
        self.score_scale = float(self.extras.get("score_scale", 1.0e9))
        self.invalid_reward = float(self.extras.get("invalid_reward", -1.0))
        self.reward_mode = str(self.extras.get("reward_mode", "score" if self.max_turns <= 1 else "improvement"))
        self.statement_max_chars = int(self.extras.get("statement_max_chars", 12000))
        self.tool_readme_max_chars = int(self.extras.get("tool_readme_max_chars", 5000))
        self.case_feedback_limit = int(self.extras.get("case_feedback_limit", 5))
        self.skip_local_visualization = _to_bool(self.extras.get("skip_local_visualization"), True)

        self.session = None
        self.turns = 0
        self.best_signed_score: float | None = None
        self.best_absolute_score: int | None = None
        self.best_judge_result: str | None = None
        self.last_error: str | None = None
        self._init_session()

    def _init_session(self) -> None:
        try:
            import ale_bench
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ale_bench is not importable. Install ALE-Bench in the SkyRL environment, or set "
                "ALE_BENCH_REPO=/path/to/ALE-Bench so this integration can add its src/ directory."
            ) from exc

        self.session = ale_bench.start(
            problem_id=self.problem_id,
            lite_version=self.lite_version,
            use_same_time_scale=False,
            maximum_num_call_public_eval=max(1, self.max_turns),
            session_duration=float(self.extras.get("session_duration_seconds", 24 * 3600)),
            num_workers=self.num_workers,
            run_visualization_server=False,
        )

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        assert self.session is not None
        problem = self.session.problem
        text = "\n".join(
            [
                f"problem_id: {self.problem_id}",
                f"title: {problem.metadata.title}",
                f"score_type: {problem.metadata.score_type.value}",
                f"problem_type: {problem.metadata.problem_type.value}",
                f"time_limit: {problem.constraints.time_limit}",
                f"memory_limit: {problem.constraints.memory_limit}",
                f"default_code_language: {self.code_language}",
                "Return exactly one complete solution source file. Prefer a fenced code block, "
                "for example ```cpp20 ... ```.",
                "",
                "# Problem Statement",
                _truncate(problem.statement, self.statement_max_chars),
                "",
                "# Example Input",
                _truncate(problem.example_input or "", 2000),
                "",
                "# Example Output",
                _truncate(problem.example_output or "", 2000),
                "",
                "# Tool README",
                _truncate(problem.tool_readme or "", self.tool_readme_max_chars),
            ]
        )
        messages = list(prompt)
        messages.append({"role": "user", "content": text})
        return messages, {"problem_id": self.problem_id}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        assert self.session is not None
        self.turns += 1
        error = None
        result = None
        code = ""
        language = self.code_language

        try:
            code, language = extract_code_and_language(action, self.code_language)
            result = self.session.public_eval(
                code=code,
                code_language=language,
                judge_version=self.judge_version,
                skip_local_visualization=self.skip_local_visualization,
            )
            if self._is_scored_result(result):
                signed_score = self._signed_score(result)
                reward = self._reward_from_signed_score(signed_score)
                if self.best_signed_score is None or signed_score > self.best_signed_score:
                    self.best_signed_score = signed_score
                    self.best_absolute_score = int(result.overall_absolute_score)
                    self.best_judge_result = str(result.overall_judge_result.value)
            else:
                reward = self.invalid_reward
        except Exception as exc:
            error = str(exc)
            self.last_error = error
            reward = self.invalid_reward

        done = self.turns >= self.max_turns
        observations: ConversationType = []
        if not done:
            observations = [{"role": "user", "content": self._feedback_text(result, error)}]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=done,
            metadata={
                "problem_id": self.problem_id,
                "turns": self.turns,
                "code_language": language,
                "judge_version": self.judge_version,
                "valid_submission": error is None,
                "error": error,
                "reward_mode": self.reward_mode,
                "best_absolute_score": self.best_absolute_score,
                "best_signed_score": self.best_signed_score,
                "best_judge_result": self.best_judge_result,
                "code_chars": len(code),
                "result": self._result_metadata(result),
            },
        )

    def _signed_score(self, result: Any) -> float:
        problem = self.session.problem
        score = float(result.overall_absolute_score)
        if problem.metadata.score_type.value == "minimize":
            return -score
        return score

    def _is_scored_result(self, result: Any) -> bool:
        if result.overall_judge_result.value == "ACCEPTED":
            return True
        return bool(getattr(result, "allow_score_non_ac", False) and result.overall_absolute_score > 0)

    def _reward_from_signed_score(self, signed_score: float) -> float:
        if self.reward_mode == "improvement" and self.best_signed_score is not None:
            return max(0.0, signed_score - self.best_signed_score) / self.score_scale
        return signed_score / self.score_scale

    def _feedback_text(self, result: Any, error: str | None) -> str:
        if error is not None:
            return (
                "Your submission could not be evaluated.\n"
                f"error={error}\n"
                "Submit a complete source file in one fenced code block."
            )
        return "\n".join(
            [
                "Public evaluation result:",
                _result_summary(result, self.case_feedback_limit),
                f"best_absolute_score={self.best_absolute_score}",
                "You may submit an improved complete solution.",
            ]
        )

    def _result_metadata(self, result: Any) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "overall_judge_result": result.overall_judge_result.value,
            "overall_absolute_score": int(result.overall_absolute_score),
            "overall_relative_score": result.overall_relative_score,
            "num_cases": len(result.case_results),
            "case_results": [
                {
                    "judge_result": case.judge_result.value,
                    "absolute_score": case.absolute_score,
                    "execution_time": case.execution_time,
                    "memory_usage": case.memory_usage,
                    "message": case.message,
                }
                for case in result.case_results[: self.case_feedback_limit]
            ],
        }

    def get_metrics(self) -> dict[str, Any]:
        return {
            "best_signed_score": self.best_signed_score or 0.0,
            "best_absolute_score": self.best_absolute_score or 0.0,
            "turns": self.turns,
            "has_error": 1.0 if self.last_error else 0.0,
        }

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
