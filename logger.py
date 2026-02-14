"""
Debug logger for the backtester. Writes one JSON object per line to a log file.
Ensures debugging continues to work across all modules.
"""
import json
import time

DEBUG_LOG_PATH = r"c:\Users\Waseem\Downloads\backtester\.cursor\debug.log"


def _agent_log(payload: dict) -> None:
    """
    Tiny debug logger.

    It writes one JSON object per line to DEBUG_LOG_PATH.
    This helps us see what happened if the app has a problem.
    """
    base = {
        "sessionId": "debug-session",
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({**base, **payload}) + "\n")
    except Exception:
        # If logging fails, we do NOT want the whole app to crash.
        pass
