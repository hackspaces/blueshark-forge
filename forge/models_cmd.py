"""`forge models` — the curated catalog, checked against THIS machine.

    forge models              list the curated recipes + a RAM-fit estimate
    forge models show <name>  print the copy-pasteable runbook for one entry

Offline by design (Phase 1): no HF calls, no subprocess — the registry is the data,
the machine facts come from config (or a live detect if setup never ran).
"""
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


def cmd_models(args):
    hw = _machine()
    action = getattr(args, "action", None) or "list"
    if action == "list":
        _list(hw)
        return 0
    if action == "show":
        if not getattr(args, "name", None):
            print("✗ usage: forge models show <name>")
            return 1
        return _show(hw, args.name)
    print(f"✗ unknown action '{action}' — try `forge models` or `forge models show <name>`.")
    return 1
