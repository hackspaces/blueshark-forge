"""`forge models` — the curated catalog, checked against THIS machine.

    forge models              list the curated recipes + a RAM-fit estimate
    forge models show <name>  print the copy-pasteable runbook for one entry

Offline by design (Phase 1): no HF calls, no subprocess — the registry is the data,
the machine facts come from config (or a live detect if setup never ran).
"""
import subprocess

from . import config
from . import registry
from .render import paint, fit, term_width


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
    # non-Ollama (llamacpp / bespoke) turnkey provisioning is not built yet — be honest,
    # and hand over the exact verified runbook rather than pretend or half-do it.
    print(paint(f"  forge can't auto-provision {m['engine']} models yet — that's next.", "yellow"))
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
    if action in ("show", "use") and not name:
        print(f"✗ usage: forge models {action} <name>")
        return 1
    if action == "show":
        return _show(hw, name)
    if action == "use":
        return _use(hw, name)
    print(f"✗ unknown action '{action}' — try `forge models`, `forge models show <name>`, or `forge models use <name>`.")
    return 1
