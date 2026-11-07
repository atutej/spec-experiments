"""SkyRL-SQL environment with an outcome-only reward, for a model with its own thinking template.

Subclasses the unmodified skyrl_gym SQLEnv and overrides only the reward. The stock reward returns -1 unless the transcript
has exactly one <solution> block AND a complete <think>...</think> pair (a format the model was not trained on). Here the
reward is the execution match of the LAST <solution>...</solution> block in the assistant turns: 1.0 on a match, 0.0 otherwise
(including no usable solution). Thinking markers are neither required nor penalised.
"""
import re

from skyrl_gym.envs.sql.env import SQLEnv
from skyrl_gym.envs.sql.utils import execute_sql_wrapper_single

SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
_TAGS = re.compile(r"</?(sql|observation|solution)\b", re.I)


def extract_last_solution(text: str):
    """Text of the last closed <solution> block (the nearest opener before the last closer), or None."""
    end = text.rfind(SOLUTION_END)
    if end < 0:
        return None
    start = text.rfind(SOLUTION_START, 0, end)
    if start < 0:
        return None
    sql = text[start + len(SOLUTION_START):end].strip()
    if not sql or _TAGS.search(sql):
        return None
    return sql


def native_reward(assistant_text: str, gold_sql: str, db_file: str, timeout: int = 30) -> float:
    pred_sql = extract_last_solution(assistant_text)
    if pred_sql is None:
        return 0.0
    pred = execute_sql_wrapper_single(db_file, pred_sql, timeout, assistant_text)
    ref = execute_sql_wrapper_single(db_file, gold_sql, timeout, assistant_text)
    pred_results, gt_results = pred[2], ref[2]
    return 1.0 if (pred_results is not None and gt_results is not None and pred_results == gt_results) else 0.0


class SQLNativeThinkEnv(SQLEnv):
    def _get_reward(self, action: str, done: bool) -> float:
        if not done:
            return 0
        assistant_text = "".join(m["content"] for m in self.chat_history if m["role"] == "assistant")
        return native_reward(assistant_text, self.gold_sql, self.db_file)
