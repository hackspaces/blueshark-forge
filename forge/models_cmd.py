"""`forge models` — what THIS machine can run, and how to run it.

    forge models              what this machine can run (a spread of open models,
                              sized + speed-checked against your RAM + hardware)
    forge models show <name>  the recipe + fit/speed for one model
    forge models use <name>   provision it and point forge's config at it
    forge models stop <name>  stop a server forge launched for it

`list`/`show` are offline (the registry is the data). `use` has side effects:
Ollama-native models are pulled and configured; llama.cpp models get their weights
fetched, a llama-server launched (recorded under ~/.forge/servers/ so `stop` can
kill it), and the config's context matched to the server. Bespoke runtimes forge
can't provision yet fall back to the honest runbook.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from . import config
from . import registry
from .render import paint, fit, term_width

SERVERS = os.path.expanduser("~/.forge/servers")   # pidfiles + logs for forge-launched servers


def bench_lift(entry, path=None):
    """The harness lift for a catalog model — base (bare) vs full pass-rate delta in
    percentage points, read from ~/.forge/bench/results.jsonl. None if it hasn't been
    benched. This is the 'weak weights + strong harness' number the catalog exists to show."""
    from . import bench
    path = path or bench.RESULTS
    ids = [entry["name"]]
    if entry.get("ollama_tag"):
        ids.append(entry["ollama_tag"])            # a bench run may address it by its ollama tag
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if any(i in str(r.get("model", "")) for i in ids):
                    rows.append(r)
    except OSError:
        return None
    if not rows:
        return None
    bare = bench._rate([r for r in rows if bench.config_label(r.get("levers", [])) == "bare"])
    full = bench._rate([r for r in rows if bench.config_label(r.get("levers", [])) == "full"])
    if bare is None or full is None:
        return None
    return round(100 * (full[2] - bare[2]))


def _lift(entry):
    """(display string, color) for an entry's lift — live bench, else stored, else pending."""
    lift = bench_lift(entry)
    if lift is None:
        lift = entry.get("lift_pts")
    if not isinstance(lift, (int, float)):
        return "pending", "dim"
    return (f"+{lift}pts" if lift >= 0 else f"{lift}pts",
            "green" if lift > 0 else "yellow" if lift == 0 else "red")


def _machine():
    """The detected hardware summary: config's (written by setup) or a live detect."""
    hw = config.get("machine") or {}
    if not hw.get("ram_gb"):
        from . import setup as setupmod
        hw = setupmod.detect_machine()
    return hw


def _pnum(p):
    return f"{p:g}"                                   # 0.5→"0.5", 7.0→"7", 2.7→"2.7"


def _accelerated(hw):
    from . import setup as setupmod
    try:
        return bool(setupmod._is_accelerated(hw))
    except Exception:
        return False


def _hw_desc(hw, accel):
    """One-line hardware summary — names the memory pool that actually gates fit:
    VRAM on a GPU/multi-GPU/datacenter node, unified memory on Apple Silicon, else RAM."""
    ram = hw.get("ram_gb") or "?"
    gpus, vram = hw.get("gpus", 0), hw.get("vram_gb", 0)
    if gpus and vram:
        n = f"{gpus}× " if gpus > 1 else ""
        return f"{n}{hw.get('gpu_name') or 'GPU'} · {vram}GB VRAM · {ram}GB RAM"
    base = f"{hw.get('chip') or hw.get('arch') or '?'} · {ram}GB RAM"
    return base + (" · GPU / Metal (fast)" if accel else " · CPU-only")


_RUN_STYLE = {"good": "green", "warn": "yellow", "faint": "dim"}


def _installed_tags(engine="ollama"):
    """Model tags already pulled/installed on THIS machine (Ollama today). Empty when the
    engine isn't Ollama or Ollama isn't reachable — installed-state just isn't shown then,
    never guessed."""
    if engine != "ollama":
        return set()
    try:
        import subprocess
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        return {ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()}
    except (subprocess.SubprocessError, OSError):
        return set()


def _is_installed(name, installed):
    """A catalog model counts as installed if its tag, or its bare base name, is pulled
    (so 'gemma2:9b' matches an installed 'gemma2:9b', and 'phi-2' matches 'phi-2:latest')."""
    base = name.split(":")[0]
    return name in installed or any(t == name or t.split(":")[0] == base for t in installed)


def _list(hw):
    ram = hw.get("ram_gb") or 0
    accel = _accelerated(hw)
    vram = hw.get("vram_gb", 0)
    W = term_width()
    from . import config as _cfg
    installed = _installed_tags(_cfg.load().get("engine", "ollama"))
    print(paint("forge models — what this machine can run", "bold"))
    print(paint(f"  {_hw_desc(hw, accel)}", "dim"))
    cap_p, cap_name = registry.ceiling(ram, accel, vram)
    if cap_name:
        print(paint(f"  → runs well up to ~{_pnum(cap_p)}B    e.g. {cap_name}", "green"))
        if not accel and any((m.get("params_b") or 0) > 3.5 and registry.fits(m, ram)
                             for m in registry.MODELS):
            print(paint("    (bigger models fit but run slowly — no GPU/Metal here)", "dim"))
    else:
        print(paint("  → tight on RAM — only the smallest models fit.", "yellow"))
    print()
    head = (f"    {'MODEL':<18} {'PARAMS':>7}  {'ENGINE':<11} {'SIZE':>7}  "
            f"{'RAM~':>5}  {'RUNS':<18} NOTES")
    print(paint(head, "dim"))
    n_installed = 0
    for m in sorted(registry.MODELS, key=lambda x: x.get("params_b") or 0):
        size = m.get("weights", {}).get("size_gb")
        verdict, style = registry.runs(m, ram, accel, vram)
        runs_cell = paint(f"{verdict:<18}", _RUN_STYLE.get(style, "dim"))
        star = paint(" ★", "green") if m.get("status") == "verified" else "  "
        here = _is_installed(m["name"], installed)
        n_installed += here
        mark = paint("●", "green") if here else paint("○", "dim")   # installed vs available
        note = fit(m.get("notes", ""), max(8, W - 84))
        print(f"  {mark} {m['name']:<18} {_pnum(m.get('params_b') or 0) + 'B':>7}  {m['engine']:<11} "
              f"{(f'{size}GB' if size else '?'):>7}  {str(m.get('ram_gb_needed', '?')) + 'GB':>5}  "
              + runs_cell + " " + paint(note, "dim") + star)
    print()
    tally = (paint(f"  ● {n_installed} installed", "green") + paint("  ○ available", "dim")) if installed \
        else paint("  ○ = in the catalog (install-state needs Ollama)", "dim")
    print(tally)
    print(paint("  ★ forge has run it   ·   pull one:  forge models use <name>   ·   "
                "RAM/RUNS are estimates.   Full catalog:  forge models --all", "dim"))


def _scan(hw, refresh=False):
    """`forge models --all` — run the fit-math on the whole downloadable catalog
    (Ollama library + HuggingFace GGUF + MLX), not just the curated spread."""
    from . import catalog
    ram = hw.get("ram_gb") or 0
    accel = _accelerated(hw)
    vram = hw.get("vram_gb", 0)
    print(paint("forge models — scanning the downloadable catalog …", "bold"))

    tty = sys.stdout.isatty()
    def prog(src, d, t):
        if tty:                                # in-place progress only on a terminal; pipes/logs stay clean
            print(paint(f"  … {src} {d}/{t}", "dim") + "        ", end="\r", flush=True)
    try:
        entries, cached = catalog.load_catalog(hw, refresh=refresh, on_progress=prog)
    except Exception as e:
        print(paint(f"  ✗ couldn't reach the catalog ({e}). Try again, or use `forge models`.", "red"))
        return 1
    if tty:
        print(" " * 48, end="\r")
    srcs = sorted({e.get("source") for e in entries if e.get("source")})
    print(paint(f"  {_hw_desc(hw, accel)}   ·   sources: {', '.join(srcs) or '—'}"
                + ("   (cached)" if cached else ""), "dim"))
    graded = [(e, registry.runs(e, ram, accel, vram)) for e in entries]
    fitting = [(e, v) for e, v in graded if v[0] != "won't fit"]
    well = [e for e, v in graded if v[0] == "runs well"]
    if well:
        best = max(well, key=lambda e: e.get("params_b") or 0)
        print(paint(f"  → runs well up to ~{_pnum(best.get('params_b') or 0)}B    e.g. {best['name']}", "green"))
    print(paint(f"  → {len(fitting)} of {len(entries)} models fit this machine", "green" if fitting else "yellow"))
    print()
    print(paint(f"  {'MODEL':<46} {'PARAMS':>7}  {'ENGINE':<9} {'RUNS':<16} SOURCE", "dim"))
    fitting.sort(key=lambda ev: -(ev[0].get("params_b") or 0))          # biggest-you-can-run first
    CAP = 30
    for e, (verdict, style) in fitting[:CAP]:
        print(f"  {e['name'][:46]:<46} {_pnum(e.get('params_b') or 0) + 'B':>7}  {e['engine']:<9} "
              + paint(f"{verdict:<16}", _RUN_STYLE.get(style, "dim")) + " " + paint(e.get("source", ""), "dim"))
    if len(fitting) > CAP:
        print(paint(f"  …and {len(fitting) - CAP} more that fit.", "dim"))
    print()
    print(paint("  Ollama models are one command:  forge models use <name>   ·   "
                "sizes/runs are estimates.   Refresh:  forge models --all --refresh", "dim"))
    return 0


def _show(hw, name):
    m = registry.get(name)
    if not m:
        print(f"✗ no curated entry named '{name}'. Known: {', '.join(registry.names())}")
        return 1
    ram = hw.get("ram_gb") or 0
    accel = _accelerated(hw)
    vram = hw.get("vram_gb", 0)
    params = m.get("params_b")
    print(paint(f"{m['name']}", "bold")
          + paint(f"  ·  {f'{_pnum(params)}B · ' if params else ''}{m['repo']}  ·  {m['arch']}", "dim"))
    verdict, style = registry.runs(m, ram, accel, vram)
    print(f"  needs ~{m.get('ram_gb_needed', '?')}GB · this machine has {ram or '?'}GB → "
          + paint(verdict, _RUN_STYLE.get(style, "dim")) + paint("  (estimate)", "dim"))
    lift_s, lift_style = _lift(m)
    if lift_s == "pending":
        print(paint("  harness lift: pending — run `forge bench` and the base→full gain appears here", "dim"))
    else:
        print("  harness lift: " + paint(lift_s.replace("pts", " pts"), lift_style)
              + paint("  base→full (forge bench)", "dim"))
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
        with open(pidf) as f:
            pid = int(f.read().strip())
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
        if getattr(args, "all", False):
            return _scan(hw, refresh=getattr(args, "refresh", False))
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
