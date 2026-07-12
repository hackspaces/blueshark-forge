"""Declared tool effects and capabilities (H06).

Each built-in tool DECLARES the side effects it can have — filesystem read/write,
process spawn, network, secret access, external message, irreversible change —
together with the minimum authority it needs and its approval policy. Policy then
evaluates these DECLARED effects rather than re-inferring risk from a bare action
name, and denials can name the capability a session is missing.

The capability grant is a deterministic function of the harness-owned authority
level (never of the model), so a smarter or escalated model gains no extra reach.
Unknown tools fail closed: they are treated as having every effect and needing
admin. The token-aware shell classifier in `authority.py` remains as defense in
depth for the shell details a static declaration can't see (e.g. `sudo`, an
`rm -rf /`), so `bash`'s declared OPERATOR floor can still be raised per command.
"""
from dataclasses import dataclass
from enum import Flag, auto

from .authority import AuthorityLevel   # one-way: effects depends on the authority ladder


class Effect(Flag):
    NONE = 0
    FS_READ = auto()          # reads files / the workspace
    FS_WRITE = auto()         # creates, edits, or deletes files
    PROCESS = auto()          # spawns a subprocess
    NETWORK = auto()          # can reach the network
    SECRET = auto()           # can read credentials / secret stores
    EXTERNAL_MSG = auto()     # sends a message outside this session (e.g. the fleet)
    IRREVERSIBLE = auto()     # can make changes that are hard to undo


ALL_EFFECTS = (Effect.FS_READ | Effect.FS_WRITE | Effect.PROCESS | Effect.NETWORK
               | Effect.SECRET | Effect.EXTERNAL_MSG | Effect.IRREVERSIBLE)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    effects: Effect
    min_authority: AuthorityLevel
    approval: str = "none"    # none | plan-gated | manual


# The declaration table. `min_authority` here is the single source that authority's
# per-action level agrees with (a test binds the two so they can't drift).
TOOLS = {
    "read_file":  ToolSpec("read_file",  Effect.FS_READ, AuthorityLevel.OBSERVE),
    "list_files": ToolSpec("list_files", Effect.FS_READ, AuthorityLevel.OBSERVE),
    "grep":       ToolSpec("grep",       Effect.FS_READ, AuthorityLevel.OBSERVE),
    "glob":       ToolSpec("glob",       Effect.FS_READ, AuthorityLevel.OBSERVE),
    "say":        ToolSpec("say",        Effect.NONE,    AuthorityLevel.OBSERVE),
    "write_file": ToolSpec("write_file", Effect.FS_WRITE, AuthorityLevel.CONTRIBUTE, "plan-gated"),
    "edit_file":  ToolSpec("edit_file",  Effect.FS_WRITE, AuthorityLevel.CONTRIBUTE, "plan-gated"),
    "run_tests":  ToolSpec("run_tests",  Effect.FS_READ | Effect.PROCESS, AuthorityLevel.CONTRIBUTE),
    # bash is declared broad; the shell classifier narrows specific commands to ADMIN.
    "bash":       ToolSpec("bash", Effect.FS_WRITE | Effect.PROCESS | Effect.NETWORK | Effect.IRREVERSIBLE,
                           AuthorityLevel.OPERATOR, "plan-gated"),
    # fleet_send's SEND is operator; a read-only roster list stays observe (authority.py).
    "fleet_send": ToolSpec("fleet_send", Effect.EXTERNAL_MSG | Effect.NETWORK, AuthorityLevel.OPERATOR),
}


def spec_for(kind: str) -> ToolSpec:
    """The declared spec for a tool, or a FAIL-CLOSED spec (every effect, admin-only)
    for an unknown tool — an undeclared tool is never quietly granted."""
    return TOOLS.get(kind, ToolSpec(kind or "unknown", ALL_EFFECTS, AuthorityLevel.ADMIN, "manual"))


def granted_capabilities(level: AuthorityLevel):
    """The tools whose full capability an authority level is granted — a deterministic
    function of the harness-owned level, never of the model. (fleet_send's read-only
    roster is separately available below operator; see authority.required_for.)"""
    return sorted(name for name, spec in TOOLS.items() if level >= spec.min_authority)


_EFFECT_LABELS = [
    (Effect.FS_READ, "read files"),
    (Effect.FS_WRITE, "write files"),
    (Effect.PROCESS, "run processes"),
    (Effect.NETWORK, "use the network"),
    (Effect.SECRET, "read secrets"),
    (Effect.EXTERNAL_MSG, "message other sessions"),
    (Effect.IRREVERSIBLE, "make irreversible changes"),
]


def describe(effects: Effect) -> str:
    """A human phrase naming the effects, for denial messages ('no side effects' if none)."""
    names = [label for bit, label in _EFFECT_LABELS if effects & bit]
    return ", ".join(names) if names else "no side effects"
