"""`forge models` — the curated catalog + turnkey provisioning.

    forge models              list the curated recipes + a RAM-fit estimate
    forge models show <name>  print the copy-pasteable runbook for one entry
    forge models use <name>   provision it and point forge's config at it
    forge models stop <name>  stop a server forge launched for it

`list`/`show` are offline (the registry is the data). `use` has side effects:
Ollama-native models are pulled and configured; llama.cpp models get their weights
fetched, a llama-server launched (recorded under ~/.forge/servers/ so `stop` can
kill it), and the config's context matched to the server. Bespoke runtimes forge
can't provision yet fall back to the honest runbook.
"""
import glob
import os
import shutil
import signal
import subprocess
import time
import urllib.request

from . import config
from . import registry
from .render import paint, fit, term_width

SERVERS = os.path.expanduser("~/.forge/servers")   # pidfiles + logs for forge-launched servers


def _machine():
    """The detected hardware summary: config's (written by setup) or a live detect."""
    hw = config.get("machine") or {}
    if not hw.get("ram_gb"):
        from . import setup as setupmod
        hw = setupmod.detect_machine()
    return hw


def _list(hw):
    ram = hw.get("ram_gb") or 0
    W = term_width()
    print(paint("forge models — curated recipes (hand-verified, not scraped)", "bold"))
    print(paint(f"  this machine: {hw.get('chip') or hw.get('arch') or '?'} · {ram or '?'}GB RAM", "dim"))
    print()
    head = f"  {'NAME':<18} {'ENGINE':<12} {'SIZE':<8} {'RAM~':<6} {'FIT?':<6} {'STATUS':<10} NOTES"
    print(paint(head, "dim"))
    for m in registry.MODELS:
        size = m.get("weights", {}).get("size_gb")
        fit_ok = registry.fits(m, ram)
        status = m.get("status", "?")
        # pad the PLAIN text first, then paint — ANSI codes must not count as width
        fitcell = paint(f"{'✓' if fit_ok else '✗':<6}", "green" if fit_ok else "red")
        stcell = paint(f"{status:<10}", "green" if status == "verified" else "yellow")
        row = (f"  {m['name']:<18} {m['engine']:<12} "
               f"{(f'{size}GB' if size else '?'):<8} {str(m.get('ram_gb_needed', '?')) + 'GB':<6} ")
        note = fit(m.get("notes", ""), max(10, W - 66))
        print(row + fitcell + " " + stcell + " " + paint(note, "dim"))
    print()
    print(paint("  RAM~ is an estimate (other apps shrink real headroom). "
                "`forge models show <name>` prints the runbook.", "dim"))


def _show(hw, name):
    m = registry.get(name)
    if not m:
        print(f"✗ no curated entry named '{name}'. Known: {', '.join(registry.names())}")
        return 1
    ram = hw.get("ram_gb") or 0
    print(paint(f"{m['name']}", "bold") + paint(f"  ·  {m['repo']}  ·  {m['arch']}", "dim"))
    verdict = ("fits" if registry.fits(m, ram) else "does NOT fit")
    style = "green" if registry.fits(m, ram) else "red"
    print(f"  needs ~{m.get('ram_gb_needed', '?')}GB · this machine has {ram or '?'}GB → "
          + paint(verdict, style) + paint("  (estimate)", "dim"))
    if m.get("notes"):
        print(paint(f"  {m['notes']}", "dim"))
    if m.get("report"):
        print(paint(f"  report: {m['report']}", "dim"))
    print()
    for line in registry.runbook(m):
        print("  " + (paint(line, "dim") if line.startswith("#") else line))
    return 0


def _use_ollama(hw, entry):
    """Turnkey provision an Ollama-native model: ensure it's pulled, point forge's
    config at it, and smoke-test — so `forge run` just works afterward."""
    from . import setup as setupmod
    ok, msg = setupmod._ollama_ok()
    if not ok:
        print(paint(f"  ✗ {msg}", "red"))
        return 1
    tag = entry["ollama_tag"]
    if setupmod._have_model(tag):
        print(paint(f"  ✓ {tag} already present", "green"))
    else:
        size = entry.get("weights", {}).get("size_gb")
        print(f"  ↓ pulling {tag} via Ollama{f' (~{size}GB, one time)' if size else ''} …")
        if subprocess.run(["ollama", "pull", tag]).returncode != 0:
            print(paint(f"  ✗ failed to pull {tag} (the tag may differ on your Ollama).", "red"))
            return 1
    cfg = config.load()
    cfg.update({"engine": "ollama", "base_url": "", "api_key": "",
                "ladder": [tag], "num_ctx": setupmod.num_ctx_for(hw.get("ram_gb") or 0),
                "machine": hw})
    config.save(cfg)
    print(paint(f"  ✓ config → engine ollama · model {tag}", "green"))
    try:                                             # smoke test — prove it actually answers
        from .backends import make_backend
        out = make_backend(tag, engine="ollama").chat(
            [{"role": "user", "content": "Reply with exactly: OK"}], temperature=0.0)
        print(paint(f"  ✓ smoke test — model responded: {(out or '').strip()[:40]!r}", "green"))
    except Exception as e:                            # a slow/cold model shouldn't fail the setup
        print(paint(f"  ⚠ smoke test skipped ({e})", "yellow"))
    print("\n  ready → " + paint(f"forge run \"<task>\"", "bold") + "   (or just: forge)")
    return 0


# ---- llama.cpp turnkey (self-hosted server) --------------------------------

def _libllama_paths():
    """Where the installed llama library might live, so we can probe its arch table."""
    prefix = ""
    if shutil.which("brew"):
        try:
            prefix = subprocess.check_output(["brew", "--prefix"], text=True, timeout=10).strip()
        except (subprocess.SubprocessError, OSError):
            prefix = ""
    roots = [prefix] if prefix else ["/opt/homebrew", "/usr/local", "/usr"]
    libs = []
    for r in roots:
        libs += glob.glob(os.path.join(r, "lib", "libllama*.dylib"))
        libs += glob.glob(os.path.join(r, "lib", "libllama*.so"))
    return libs


def _arch_missing(probe):
    """Is the model's arch absent from the installed llama build? True = definitely
    absent (a hard stop), False = present, None = couldn't locate the library (can't
    verify — proceed with a warning; the server will error clearly if unsupported)."""
    libs = _libllama_paths()
    if not libs:
        return None
    for lib in libs:
        try:
            out = subprocess.run(["strings", lib], capture_output=True, text=True, timeout=30).stdout
        except (subprocess.SubprocessError, OSError):
            continue
        if probe in out:
            return False
    return True


def _ensure_gguf(entry):
    """The path to the model's GGUF, downloading it (resumable) if absent. None on failure."""
    w = entry["weights"]
    dest_dir = os.path.expanduser(f"~/models/{entry['name']}")
    path = os.path.join(dest_dir, w["file"])
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(paint(f"  ✓ weights present ({w['file']})", "green"))
        return path
    size = w.get("size_gb")
    print(f"  ↓ downloading {w['file']}{f' (~{size}GB, one time)' if size else ''} …")
    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://huggingface.co/{w['gguf_repo']}/resolve/main/{w['file']}"
    rc = subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-o", path, url]).returncode
    if rc != 0:
        print(paint(f"  ✗ download failed (curl exit {rc}).", "red"))
        return None
    return path


def _server_files(name):
    return os.path.join(SERVERS, name + ".pid"), os.path.join(SERVERS, name + ".log")


def _launch_server(entry, gguf):
    """Spawn llama-server detached (survives this process), recording pid + log. The
    server must outlive `forge models use` so a later `forge run` can reach it."""
    os.makedirs(SERVERS, exist_ok=True)
    s = entry["serve"]
    pidf, logf = _server_files(entry["name"])
    cmd = ["llama-server", "-m", gguf, "--host", "127.0.0.1", "--port", str(s["port"]),
           "-c", str(s["ctx"]), "-ngl", str(s.get("ngl", 999)), "--alias", entry["name"]]
    if s.get("jinja"):
        cmd.append("--jinja")
    try:
        log = open(logf, "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        print(paint(f"  ✗ could not launch llama-server: {e}", "red"))
        return False
    with open(pidf, "w") as f:
        f.write(str(p.pid))
    print(f"  ↑ llama-server starting (pid {p.pid}) · log {logf}")
    return True


def _wait_health(port, timeout=180):
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(max(1, timeout // 2)):
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if getattr(r, "status", 200) == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _use_llamacpp(hw, entry):
    """Turnkey provision a llama.cpp-served model: ensure the runtime knows the arch,
    ensure the weights, launch the server, and point forge's config at it — matching
    the reported context to the server so forge compacts in time."""
    if not shutil.which("llama-server"):
        if shutil.which("brew"):
            print("  ↓ installing llama.cpp via Homebrew (one time) …")
            subprocess.run(["brew", "install", "llama.cpp"])
        if not shutil.which("llama-server"):
            print(paint("  ✗ llama-server not found. Install llama.cpp "
                        "(brew install llama.cpp) and retry.", "red"))
            return 1
    probe = entry.get("arch_probe")
    if probe:
        miss = _arch_missing(probe)
        if miss is True:
            print(paint(f"  ✗ your llama.cpp build lacks the '{probe}' arch "
                        f"(needs ≥ {entry.get('min_version', '?')}). Update llama.cpp and retry.", "red"))
            return 1
        print(paint(f"  ✓ runtime carries the '{probe}' arch", "green") if miss is False
              else paint(f"  ⚠ couldn't verify the '{probe}' arch — proceeding; "
                         "the server errors clearly if it's unsupported.", "yellow"))
    gguf = _ensure_gguf(entry)
    if not gguf:
        return 1
    if not _launch_server(entry, gguf):
        return 1
    s = entry["serve"]
    if not _wait_health(s["port"]):
        pidf, logf = _server_files(entry["name"])
        print(paint(f"  ✗ server did not become healthy in time — check {logf}.", "red"))
        return 1
    print(paint("  ✓ server healthy", "green"))
    from . import setup as setupmod
    url = f"http://127.0.0.1:{s['port']}/v1"
    cfg = config.load()
    cfg.update({"engine": "llamacpp", "base_url": url, "api_key": "",
                "ladder": [entry["name"]], "remote_ctx": s["ctx"],
                "num_ctx": max(setupmod.num_ctx_for(hw.get("ram_gb") or 0), s["ctx"]),
                "machine": hw})
    config.save(cfg)
    print(paint(f"  ✓ config → engine llamacpp · {url} · model {entry['name']}", "green"))
    print(paint(f"    (context matched to the server at {s['ctx']} — no manual FORGE_REMOTE_CTX)", "dim"))
    print("\n  ready → " + paint("forge run \"<task>\"", "bold"))
    print(paint(f"  stop the server later with:  forge models stop {entry['name']}", "dim"))
    return 0


def _stop(name):
    """Stop a server forge launched (llama.cpp). Ollama manages its own process."""
    m = registry.get(name)
    if m and m.get("engine") == "ollama":
        print(f"  {name} runs on Ollama, which manages its own process — nothing for forge to stop.")
        return 0
    pidf, _ = _server_files(name)
    if not os.path.exists(pidf):
        print(f"  no forge-launched server for '{name}'.")
        return 0
    try:
        pid = int(open(pidf).read().strip())
    except (OSError, ValueError):
        os.remove(pidf)
        print(f"  cleared a stale pidfile for '{name}'.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
        print(paint(f"  ✓ stopped {name} server (pid {pid})", "green"))
    except ProcessLookupError:
        print(f"  {name} server was not running.")
    except OSError as e:
        print(paint(f"  ✗ could not stop pid {pid}: {e}", "red"))
        return 1
    try:
        os.remove(pidf)
    except OSError:
        pass
    return 0


def _use(hw, name):
    m = registry.get(name)
    if not m:
        print(f"✗ no curated entry named '{name}'. Known: {', '.join(registry.names())}")
        return 1
    if not registry.fits(m, hw.get("ram_gb") or 0):
        print(paint(f"  ⚠ {name} needs ~{m.get('ram_gb_needed', '?')}GB and this machine "
                    f"has {hw.get('ram_gb') or '?'}GB — it may not run well here.", "yellow"))
    if m["engine"] == "ollama":
        return _use_ollama(hw, m)
    if m["engine"] == "llamacpp":
        return _use_llamacpp(hw, m)
    # a bespoke runtime (e.g. bitnet.cpp) forge can't auto-provision — be honest and
    # hand over the exact verified runbook rather than pretend or half-do it.
    print(paint(f"  forge can't auto-provision {m['engine']} models yet.", "yellow"))
    print(paint(f"  Here's the verified runbook for {name}:\n", "dim"))
    for line in registry.runbook(m):
        print("  " + (paint(line, "dim") if line.startswith("#") else line))
    return 2


def cmd_models(args):
    hw = _machine()
    action = getattr(args, "action", None) or "list"
    if action == "list":
        _list(hw)
        return 0
    name = getattr(args, "name", None)
    if action in ("show", "use", "stop") and not name:
        print(f"✗ usage: forge models {action} <name>")
        return 1
    if action == "show":
        return _show(hw, name)
    if action == "use":
        return _use(hw, name)
    if action == "stop":
        return _stop(name)
    print(f"✗ unknown action '{action}' — try `forge models`, `forge models show/use/stop <name>`.")
    return 1
