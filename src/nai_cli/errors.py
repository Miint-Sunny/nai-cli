"""The error every command reports, with the public codes the JSON output and exit status carry."""

from typing import Any


class NaiError(Exception):
    """Error with a stable code; callers branch on code, never on message."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "message": self.message, **self.detail}


BAD_ARGS = "bad_args"
CONFIG_MISSING = "config_missing"
MODEL_UNSUPPORTED = "model_unsupported"
NEED_CONFIRM = "need_confirm"
# The ledger cannot be read, or the generation lock cannot be used
LEDGER = "ledger"
EMPTY_MASK = "empty_mask"
NO_METADATA = "no_metadata"
AUTH = "auth"
INSUFFICIENT_ANLAS = "insufficient_anlas"
RATE_LIMITED = "rate_limited"
SERVER = "server"
NETWORK = "network"
API = "api"
BUSY = "busy"
# Some jobs of a batch failed; each has its own line
BATCH_FAILED = "batch_failed"
# The request ran and was charged, but its result could not be written
SAVE_FAILED = "save_failed"
# Only printed by the command line itself
INTERRUPTED = "interrupted"
INTERNAL = "internal"
