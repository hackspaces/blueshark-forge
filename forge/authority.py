"""Harness-owned authority policy, independent of model capability."""
import os
import re
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, Mapping, Optional


class AuthorityLevel(IntEnum):
    OBSERVE = 0
    CONTRIBUTE = 1
    OPERATOR = 2
    ADMIN = 3

    @classmethod
    def parse(cls, value: Optional[str]) -> "AuthorityLevel":
        name = (value or "operator").strip().upper()
        return cls.__members__.get(name, cls.OPERATOR)


ACTION_AUTHORITY = {
    "read_file": AuthorityLevel.OBSERVE,
    "list_files": AuthorityLevel.OBSERVE,
    "grep": AuthorityLevel.OBSERVE,
    "glob": AuthorityLevel.OBSERVE,
    "say": AuthorityLevel.OBSERVE,
    "write_file": AuthorityLevel.CONTRIBUTE,
    "edit_file": AuthorityLevel.CONTRIBUTE,
    "run_tests": AuthorityLevel.CONTRIBUTE,
    "bash": AuthorityLevel.OPERATOR,
    "fleet_send": AuthorityLevel.OPERATOR,
}

# Commands that can destroy work, elevate privileges, install arbitrary remote code,
# expose common secret stores, or rewrite published history require explicit admin.
_ADMIN_SHELL = (
    re.compile(r"(^|[;&|]\s*)sudo\b"),
    re.compile(r"(^|[;&|]\s*)rm\s+[^\n]*-[^\n]*r[^\n]*f|(^|[;&|]\s*)rm\s+-rf\b"),
    re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|push\s+[^\n]*--force)\b"),
    re.compile(r"\b(curl|wget)\b[^\n|]*\|\s*(sh|bash|zsh)\b"),
    re.compile(r"(^|[;&|]\s*)(env|printenv)\b"),
    re.compile(r"\b(cat|less|head|tail)\s+[^\n]*(\.ssh|\.aws|\.gnupg|\.env)\b"),
)


@dataclass(frozen=True)
class AuthorityDecision:
    allowed: bool
    actual: AuthorityLevel
    required: AuthorityLevel
    code: str
    reason: str
    action: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["actual"] = self.actual.name.lower()
        result["required"] = self.required.name.lower()
        return result


class AuthorityPolicy:
    """Assign authority to the harness, never infer it from model intelligence."""

    def __init__(self, level: Optional[str] = None):
        self.level = AuthorityLevel.parse(
            level if level is not None else os.environ.get("FORGE_AUTHORITY"))

    def required_for(self, action: Mapping[str, Any]) -> AuthorityLevel:
        kind = str(action.get("action", ""))
        required = ACTION_AUTHORITY.get(kind, AuthorityLevel.ADMIN)
        if kind == "fleet_send":
            target = str(action.get("target", "")).strip().lower()
            if not action.get("message") or target in ("", "list", "sessions"):
                return AuthorityLevel.OBSERVE
        if kind == "bash":
            command = str(action.get("command", ""))
            if any(pattern.search(command) for pattern in _ADMIN_SHELL):
                return AuthorityLevel.ADMIN
        return required

    def evaluate(self, action: Mapping[str, Any]) -> AuthorityDecision:
        kind = str(action.get("action", ""))
        required = self.required_for(action)
        allowed = self.level >= required
        if allowed:
            return AuthorityDecision(
                True, self.level, required, "authority_granted",
                f"{self.level.name.lower()} authority permits {kind}", kind)
        return AuthorityDecision(
            False, self.level, required, "authority_denied",
            f"{kind} requires {required.name.lower()} authority; "
            f"this session has {self.level.name.lower()}", kind)

    def legal_actions(self):
        """Action kinds grammatically available at this authority level.

        Bash remains available to operator sessions; command-specific admin checks
        happen after parsing because the grammar cannot safely classify shell text.
        """
        return frozenset(
            action for action, required in ACTION_AUTHORITY.items()
            if self.level >= required or action == "fleet_send"
        )
