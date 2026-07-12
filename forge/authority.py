"""Harness-owned authority policy, independent of model capability."""
import os
import shlex
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, Mapping, Optional


class AuthorityLevel(IntEnum):
    OBSERVE = 0
    CONTRIBUTE = 1
    OPERATOR = 2
    ADMIN = 3

    @classmethod
    def parse(cls, value: Optional[str]) -> "AuthorityLevel":
        if value is None or not str(value).strip():
            return cls.OPERATOR
        name = str(value).strip().upper()
        # A misspelled security setting must not silently grant shell authority.
        return cls.__members__.get(name, cls.OBSERVE)


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

_SHELLS = frozenset(("sh", "bash", "zsh"))
_READ_COMMANDS = frozenset(("cat", "less", "head", "tail", "base64"))
_SAFE_ENV_EXAMPLES = frozenset((".env.example", ".env.sample", ".env.template"))
_CONTROL = frozenset((";", ";;", "&", "&&", "|", "||"))


def _tokens(command: str):
    """Shell-like tokens with control operators separated; None means fail closed."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except (TypeError, ValueError):
        return None


def _segments(tokens: Iterable[str]):
    segment = []
    for token in tokens:
        if token in _CONTROL or token and set(token) <= set(";&|"):
            if segment:
                yield segment
                segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def _basename(token: str) -> str:
    return os.path.basename(token or "")


def _command_head(segment):
    """Return (head, remaining tokens), skipping wrappers and assignments."""
    i = 0
    wrappers = frozenset(("command", "time", "nice", "nohup"))
    while i < len(segment):
        token = segment[i]
        if "=" in token and not token.startswith(("/", "./", "../")):
            i += 1
            continue
        if _basename(token) in wrappers:
            i += 1
            continue
        if _basename(token) == "env":
            i += 1
            while i < len(segment) and "=" in segment[i]:
                i += 1
            continue
        return _basename(token), segment[i + 1:]
    return "", []


def _option_letters(token: str):
    if token.startswith("--") or not token.startswith("-"):
        return frozenset()
    return frozenset(token[1:])


def _dangerous_target(target: str) -> bool:
    target = (target or "").strip()
    if not target:
        return False
    if target.startswith(("$", "~")) or os.path.isabs(target):
        return True
    norm = os.path.normpath(target)
    if norm in (".", "..", "*") or norm.startswith("../"):
        return True
    # An unexpanded target cannot be proven workspace-confined.
    if any(ch in target for ch in ("?", "[", "]")):
        return True
    return False


def _rm_requires_admin(args) -> bool:
    recursive = force = False
    targets = []
    options = True
    for token in args:
        if options and token == "--":
            options = False
            continue
        if options and token.startswith("-") and token != "-":
            if token == "--recursive":
                recursive = True
            elif token == "--force":
                force = True
            else:
                letters = _option_letters(token)
                recursive = recursive or bool(letters.intersection(("r", "R")))
                force = force or "f" in letters
            continue
        targets.append(token)
    return recursive and force and any(_dangerous_target(t) for t in targets)


def _secret_path(token: str) -> bool:
    token = (token or "").rstrip("/")
    if not token or token.startswith("-"):
        return False
    parts = [p for p in token.replace("\\", "/").split("/") if p]
    if any(p in (".ssh", ".aws", ".gnupg") for p in parts):
        return True
    base = parts[-1] if parts else token
    if base in _SAFE_ENV_EXAMPLES:
        return False
    return base == ".env" or base.startswith(".env.")


def _remote_script_pipe(tokens) -> bool:
    for i, token in enumerate(tokens):
        if token != "|":
            continue
        left = list(_segments(tokens[:i]))
        right = list(_segments(tokens[i + 1:]))
        if not left or not right:
            continue
        lhead, _ = _command_head(left[-1])
        rhead, _ = _command_head(right[0])
        if lhead in ("curl", "wget") and rhead in _SHELLS:
            return True
    return False


def _logical_lines(command: str):
    """Split on newlines that are genuine command separators — outside single/double
    quotes and not backslash-escaped (a line continuation). A newline inside a quoted
    string is data and stays with its line, so a quoted multi-line literal is never
    mistaken for separate commands."""
    lines, buf = [], []
    quote = None
    escaped = False
    for ch in command:
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
        elif quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == "\n":
            lines.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    lines.append("".join(buf))
    return lines


def _tokens_require_admin(tokens) -> bool:
    """Classify one already-tokenized command stream."""
    if _remote_script_pipe(tokens):
        return True
    for segment in _segments(tokens):
        head, args = _command_head(segment)
        if head == "sudo":
            return True
        if head == "rm" and _rm_requires_admin(args):
            return True
        if head == "git" and args:
            sub = args[0]
            rest = args[1:]
            if sub == "reset" and "--hard" in rest:
                return True
            if sub == "clean" and any("f" in _option_letters(x) or x == "--force" for x in rest):
                return True
            if sub == "push" and any(x == "--force" or x.startswith("--force=")
                                     or x == "--force-with-lease"
                                     or x.startswith("--force-with-lease=") for x in rest):
                return True
        if head in _READ_COMMANDS and any(_secret_path(token) for token in args):
            return True
    return False


def shell_requires_admin(command: str) -> bool:
    """Token-aware classification for recognizable high-risk shell operations.

    A single-line command is one token stream. A multi-line command is classified
    both as a whole (so a backslash-continued command is seen joined) AND line by
    line — an unquoted newline is a real command separator in shell, so a privileged
    command on its own line must not hide behind a benign first line. Lines that are
    fragments of a quoted multi-line string fail to tokenize and are skipped: they
    are data, not commands. Malformed shell as a whole fails closed to admin.
    """
    tokens = _tokens(command)
    if tokens is None:
        return True
    if _tokens_require_admin(tokens):
        return True
    if "\n" in command:
        for line in _logical_lines(command):
            line_tokens = _tokens(line)
            if line_tokens is not None and _tokens_require_admin(line_tokens):
                return True
    return False


@dataclass(frozen=True)
class AuthorityDecision:
    allowed: bool
    actual: AuthorityLevel
    required: AuthorityLevel
    code: str
    reason: str
    action: str
    effects: str = ""          # H06: the declared effects the required authority gates

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
        # H06: the floor comes from the tool's DECLARED effects (unknown tools fail
        # closed to admin via spec_for), not a bare name lookup.
        from . import effects as _effects
        required = _effects.spec_for(kind).min_authority
        if kind == "fleet_send":
            target = str(action.get("target", "")).strip().lower()
            if not action.get("message") or target in ("", "list", "sessions"):
                return AuthorityLevel.OBSERVE
        if kind == "bash" and shell_requires_admin(str(action.get("command", ""))):
            return AuthorityLevel.ADMIN
        return required

    def evaluate(self, action: Mapping[str, Any]) -> AuthorityDecision:
        kind = str(action.get("action", ""))
        required = self.required_for(action)
        allowed = self.level >= required
        from . import effects as _effects
        eff = _effects.describe(_effects.spec_for(kind).effects)
        if allowed:
            return AuthorityDecision(
                True, self.level, required, "authority_granted",
                f"{self.level.name.lower()} authority permits {kind}", kind, eff)
        return AuthorityDecision(
            False, self.level, required, "authority_denied",
            f"{kind} requires {required.name.lower()} authority to {eff}; "
            f"this session has {self.level.name.lower()}", kind, eff)

    def legal_actions(self):
        """Action kinds grammatically available at this authority level.

        Bash remains available to operator sessions; command-specific admin checks
        happen after parsing because the grammar cannot safely classify shell text.
        """
        return frozenset(
            action for action, required in ACTION_AUTHORITY.items()
            if self.level >= required or action == "fleet_send"
        )
