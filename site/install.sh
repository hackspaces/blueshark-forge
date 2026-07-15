#!/bin/sh
# forge installer — one line to the model-agnostic agentic runtime.
#
#   curl -fsSL https://topk1.com/forge/install.sh | sh
#
# (mirror, if you'd rather fetch from the repo directly:
#  curl -fsSL https://raw.githubusercontent.com/hackspaces/blueshark-forge/main/site/install.sh | sh)
#
# Installs the `blueshark-forge` package (the `forge` CLI), checking prerequisites
# and telling you what it found. Idempotent — re-run to upgrade. Set
# FORGE_INSTALL_DRY_RUN=1 to inspect your environment without installing anything.
set -eu

PKG="blueshark-forge"
DRY="${FORGE_INSTALL_DRY_RUN:-}"

say()  { printf '  %s\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }
run()  { if [ -n "$DRY" ]; then say "(dry-run) would run: $*"; else "$@"; fi; }

printf '\n\033[1mforge\033[0m — model-agnostic agentic runtime\n\n'

# 1. Python 3.10+ (the only hard requirement — forge itself is stdlib-only).
PY=""
for c in python3 python; do
  command -v "$c" >/dev/null 2>&1 || continue
  v=$("$c" -c 'import sys; print("%d %d" % sys.version_info[:2])' 2>/dev/null || echo "0 0")
  maj=${v% *}; min=${v#* }
  if [ "$maj" -ge 3 ] 2>/dev/null && { [ "$maj" -gt 3 ] || [ "$min" -ge 10 ]; } 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  case "$(uname -s)" in
    Darwin) hint="brew install python" ;;
    Linux)  hint="sudo apt install python3 python3-pip   (or your distro's equivalent)" ;;
    *)      hint="install Python 3.10+ from https://python.org" ;;
  esac
  die "Python 3.10+ is required and wasn't found. Install it:  $hint"
fi
ok "Python $("$PY" -c 'import platform; print(platform.python_version())')  ($PY)"

# 2. Install forge — prefer pipx (its own isolated env), fall back to pip --user.
if command -v pipx >/dev/null 2>&1; then
  ok "pipx found — installing $PKG in an isolated environment"
  run pipx install --force "$PKG"
else
  warn "pipx not found — installing $PKG with pip (--user)"
  say "(pipx is the cleaner way for CLI tools: $PY -m pip install --user pipx)"
  run "$PY" -m pip install --user --upgrade "$PKG"
fi

# 3. Verify the CLI is reachable.
if [ -n "$DRY" ]; then
  ok "(dry-run) would verify: forge --version"
elif command -v forge >/dev/null 2>&1; then
  ok "installed: $(forge --version 2>/dev/null || echo "$PKG")"
else
  warn "forge is installed but not on your PATH yet."
  say "Open a new terminal, or run:  pipx ensurepath   (then reopen your shell)"
fi

# 4. An inference engine is optional — only local models need one.
printf '\n'
if command -v ollama >/dev/null 2>&1; then
  ok "Ollama detected — local models are ready:  forge models use <name>"
else
  warn "No local engine detected (optional)."
  say "For local models, install Ollama from https://ollama.com — or bring a"
  say "frontier model with your own key (OpenAI / Anthropic). forge drives any of them."
fi

# 5. Where to go next — the same three steps `forge` itself shows on a fresh run,
#    so the installer and the first-run welcome tell one story.
printf '\n\033[1mNext\033[0m\n'
say "forge                  see what this machine can run"
say "forge models use phi-2 a quick starter — pulled + ready in ~2 min"
say 'forge run "<a task>"   then put it to work'
printf '\n'
say "(or  forge setup  to pick a model ladder yourself)"
printf '\n'
