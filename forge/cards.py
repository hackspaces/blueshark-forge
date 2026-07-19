"""Agent cards — a deterministic-yet-unique profile of every model a user forges into an
agent, built from the model's real attributes and the user's real work.

forge's premise is that raw weights become a capable AGENT through the harness — so a card
measures AGENTIC work, not creature combat. The six attributes are the things that decide
whether a small model gets real work done: does it reason, land its edits, recover from a
failure, finish honestly, keep pace. The scoring MATH is a well-known deterministic stat
formula (nothing ownable), but every attribute, class, and term here is forge's own.

Two properties, by design:
  - DETERMINISTIC + GLOBALLY UNIQUE. Each install mints a 128-bit Forge ID once. A specimen
    is a pure function of (forge_id, model): its Grain (innate roll), Temperament, and Foil
    are fixed at first forge and regenerate identically anywhere — so a shared card
    reproduces exactly, yet no two people ever roll the same one (128 bits can't collide).
  - DRIVEN BY REAL WORK. The model's real metadata sets its class + base attributes; the
    user's real sessions/edits raise its Mastery and Temper. Use it more, the card grows.
"""
import hashlib
import json
import math
import os

from . import render as _r

FORGE_DIR = os.path.expanduser("~/.forge")
_ID_PATH = os.path.join(FORGE_DIR, "forge_id.json")

# The six AGENT ATTRIBUTES. `stm` (Stamina) is the endurance attribute — it uses the
# endurance stat formula (no temperament modifier) the way HP does; the other five can be
# tilted by temperament. Order is fixed (it seeds Grain + the formula).
ATTRS = ("stm", "pre", "res", "rea", "rel", "pac")
ATTR_LABELS = {"stm": "Stamina", "pre": "Precision", "res": "Resilience",
               "rea": "Reasoning", "rel": "Reliability", "pac": "Pace"}
ATTR_BLURB = {
    "stm": "sustains long autonomous runs", "pre": "lands edits right the first time",
    "res": "recovers from a failure without spiraling", "rea": "depth of planning + thought",
    "rel": "finishes honestly — verified, not just claimed", "pac": "throughput — steps to done",
}

# Temperament tilts ONE attribute up 10% and another down 10% (never Stamina), over the five
# tunable attributes; a diagonal roll is neutral. 25 dispositions — the name is flavor, the
# +X/-Y is shown, exactly as the disposition is learned in play.
_TEMPER_ATTRS = ("pre", "res", "pac", "rea", "rel")
_TEMPERAMENTS = (
    "Methodical", "Reckless", "Dogged", "Terse", "Cautious",
    "Bold", "Patient", "Restless", "Meticulous", "Blunt",
    "Nimble", "Hasty", "Balanced", "Eager", "Scattered",
    "Thoughtful", "Curious", "Deliberate", "Plain", "Impulsive",
    "Careful", "Gentle", "Stubborn", "Diligent", "Quirky",
)

FOIL_THRESHOLD = 16          # (id ⊕ pid) low bits < 16  →  ~1/4096, a rare finish

# Rarity by Forge Rating (attribute total).
RATING_TIERS = [(600, "mythic"), (540, "master"), (460, "forged"), (380, "tempered"), (0, "raw")]
_RARITY_STYLE = {"mythic": "yellow", "master": "magenta", "forged": "cyan",
                 "tempered": "green", "raw": "dim"}
_RARITY_PIP = {"mythic": "◆◆◆◆◆", "master": "◆◆◆◆", "forged": "◆◆◆", "tempered": "◆◆", "raw": "◆"}


# ---- Forge ID: the per-install uniqueness root ------------------------------
def forge_id():
    """This install's 128-bit Forge ID, minted once and reused forever — the guarantee no
    two people ever forge the same specimen. Every card is seeded from it."""
    try:
        with open(_ID_PATH) as f:
            fid = json.load(f).get("id")
            if fid:
                return fid
    except (OSError, ValueError):
        pass
    fid = os.urandom(16).hex()
    try:
        os.makedirs(FORGE_DIR, exist_ok=True)
        with open(_ID_PATH, "w") as f:
            json.dump({"id": fid}, f)
    except OSError:
        pass
    return fid


def _h(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest(), 16)


def _clamp(v, lo=15, hi=200):
    return max(lo, min(hi, int(round(v))))


# ---- the model's forged form (base attributes from real metadata) -----------
def forged_form(model):
    """Base attributes + class + a stable index for a model, DETERMINISTIC from its real
    metadata. Small models are fast but frail (high Pace, low Stamina); big models endure
    and reason but are slow; a forge-VERIFIED ★ or measured harness-lift raises Resilience +
    Reliability — the card rewards models forge has actually proven, not just big ones."""
    p = float(model.get("params_b") or 1)
    size = float((model.get("weights") or {}).get("size_gb") or p)
    lift = model.get("lift_pts") or 0
    verified = model.get("status") == "verified"
    reasoning = bool(model.get("reasoning"))
    name = model.get("name", "?")
    is_code = "cod" in (name + " " + (model.get("notes") or "")).lower()

    power = 40 + 60 * math.log10(1 + p)
    base = {
        "stm": _clamp(45 + 20 * math.log10(1 + size) + p),        # endurance ~ size/context
        "pre": _clamp(power + (18 if is_code else 0)),            # coders land edits
        "res": _clamp(power * 0.8 + 3 * lift + (20 if verified else 0)),   # proven → recovers
        "rea": _clamp(power + (18 if reasoning else 0)),          # reasoning models think deeper
        "rel": _clamp(power * 0.8 + 2 * lift + (18 if verified else 0)),   # verified → honest done
        "pac": _clamp(210 - 90 * math.log10(1 + p)),             # small is FAST
    }
    return {"name": name, "base": base, "rating": sum(base.values()),
            "classes": _classes(model, is_code, reasoning),
            "no": _h("no", name) % 1000 + 1}


def _classes(model, is_code, reasoning):
    p = float(model.get("params_b") or 1)
    out = []
    if is_code:
        out.append("Coder")
    if reasoning:
        out.append("Reasoner")
    if model.get("kind") == "moe":
        out.append("Sparse")           # mixture-of-experts: sparse-active
    if p >= 30 and "Anvil" not in out:
        out.append("Anvil")            # big, heavy, tanky
    if p < 2:
        out.append("Sprinter")         # tiny, quick
    if not out:
        out.append("Generalist")
    return out[:2]


def rarity(rating):
    for cutoff, tier in RATING_TIERS:
        if rating >= cutoff:
            return tier
    return "raw"


# ---- the specimen (fixed at first forge, unique per Forge ID + model) -------
def pid(fid, model_name):
    """The 32-bit specimen value — fixed per (forge_id, model). Drives Temperament, Grain,
    and (with the Forge ID) Foil."""
    return _h("pid", fid, model_name) & 0xFFFFFFFF


def temperament(pid_val):
    """(name, up_attr, down_attr) — index = pid % 25. Neutral dispositions have up==down."""
    i = pid_val % 25
    up, down = _TEMPER_ATTRS[i // 5], _TEMPER_ATTRS[i % 5]
    return _TEMPERAMENTS[i], (None, None) if up == down else (up, down)


def grain(pid_val):
    """The six Grain values (0-31) — the innate quality of this specimen, its genetic roll,
    derived from the specimen value so two forges of one model differ."""
    hb = hashlib.sha256(f"grain:{pid_val}".encode()).digest()
    return {a: hb[i] % 32 for i, a in enumerate(ATTRS)}


def is_foil(fid, pid_val):
    """A rare finish (~1/4096): (Forge ID ⊕ specimen) low bits below the threshold. Fixed
    per (forge_id, model) — you either forged a foil of it or you didn't."""
    fid_int = int(fid, 16) if isinstance(fid, str) else int(fid)
    fhi, flo = (fid_int >> 16) & 0xFFFF, fid_int & 0xFFFF
    phi, plo = (pid_val >> 16) & 0xFFFF, pid_val & 0xFFFF
    return (fhi ^ flo ^ phi ^ plo) < FOIL_THRESHOLD


# ---- Mastery + Temper (grow with real work) ---------------------------------
def training(telemetry):
    """Mastery (1-100) and a Temper spread from the user's REAL work — the seeds of work.
    `telemetry`: {sessions, edits}. A never-run model is a fresh, un-mastered specimen; use
    levels its Mastery and pours Temper into what it exercised. Empty → the base specimen."""
    edits = int((telemetry or {}).get("edits", 0))
    sessions = int((telemetry or {}).get("sessions", 0))
    mastery = max(1, min(100, 5 + 3 * edits + sessions))
    temper = {a: 0 for a in ATTRS}
    earned = min(510, 8 * (edits * 2 + sessions))            # capped 510 total, 252/attr
    temper["res"] = temper["rel"] = min(252, earned // 3)    # sustained use hardens the proven attrs
    temper["pre"] = temper["rea"] = min(252, earned // 4)
    return mastery, temper


# ---- the deterministic attribute formula ------------------------------------
def _attr(attr, base, grain_v, temper_v, mastery, up, down):
    core = ((2 * base + grain_v + temper_v // 4) * mastery) // 100
    if attr == "stm":                                        # endurance: no temperament tilt
        return core + mastery + 10
    val = core + 5
    if attr == up:
        val = (val * 110) // 100
    elif attr == down:
        val = (val * 90) // 100
    return val


def card(model, fid=None, telemetry=None):
    """Assemble the full agent card: the model's forged form + this install's unique specimen
    (Grain/Temperament/Foil) + training (Mastery/Temper) → the six computed attributes. Fully
    determined by (fid, model, telemetry), so it regenerates identically anywhere."""
    fid = fid or forge_id()
    form = forged_form(model)
    pv = pid(fid, form["name"])
    temp_name, (up, down) = temperament(pv)
    gr = grain(pv)
    mastery, temper = training(telemetry)
    attrs = {a: _attr(a, form["base"][a], gr[a], temper[a], mastery, up, down) for a in ATTRS}
    return {
        "name": form["name"], "no": form["no"], "classes": form["classes"],
        "base": form["base"], "rating": form["rating"],
        "rarity": rarity(form["rating"]), "foil": is_foil(fid, pv),
        "temperament": temp_name, "temper_up": up, "temper_down": down,
        "grain": gr, "temper": temper, "mastery": mastery, "attrs": attrs,
        "grain_total": sum(gr.values()),
    }


# ---- rendering (built on the display-width foundation) ----------------------
_CARD_W = 42


def _bar(value, hi, width=12):
    filled = max(0, min(width, round(value / max(1, hi) * width)))
    return "█" * filled + "·" * (width - filled)


def render_card(c, width=_CARD_W):
    """A boxed card for one specimen. Every line is padded to the same DISPLAY width (the
    render foundation), so the box holds even with wide model names / emoji. Coloured by
    rarity; a foil gets a ✦ and a gold frame. Content sits at fixed columns so it aligns."""
    inner = width - 2
    rs = _RARITY_STYLE.get(c["rarity"], "dim")
    frame = "yellow" if c["foil"] else rs

    def line(body=""):
        body = _r.clip(body, inner)
        pad = " " * max(0, inner - _r.display_width(body))
        return _r.paint("│", frame) + body + pad + _r.paint("│", frame)

    def rule(l, r):
        return _r.paint(l + "─" * inner + r, frame)

    foil = _r.paint(" ✦", "yellow") if c["foil"] else ""
    classes = " ".join(f"[{t}]" for t in c["classes"])
    temp = (f"+{ATTR_LABELS[c['temper_up']]}/-{ATTR_LABELS[c['temper_down']]}"
            if c["temper_up"] else "neutral")
    hi = max(c["attrs"].values())
    out = [rule("╭", "╮"),
           line(f" No.{c['no']:03d}  " + _r.paint(c["name"], "bold")
                + _r.paint(f"  M{c['mastery']}", "dim") + foil),
           line("  " + _r.paint(f"{c['rarity'].upper()} {_RARITY_PIP[c['rarity']]}", rs)
                + _r.paint(f"   {classes}", "dim")),
           line("  " + _r.paint(f"Temperament {c['temperament']} ({temp})", "dim")),
           rule("├", "┤")]
    for a in ATTRS:
        out.append(line(f"  {ATTR_LABELS[a]:<11}{c['attrs'][a]:>4}  {_bar(c['attrs'][a], hi)}"))
    out.append(rule("├", "┤"))
    out.append(line(_r.paint(f"  Forge Rating {c['rating']}   Grain {c['grain_total']}/186", "dim")))
    out.append(rule("╰", "╯"))
    return "\n".join(out)


def _telemetry(model_name):
    """Real work with a model on THIS machine, for Mastery/Temper — from the profile store's
    actual counters. Sessions are experience; successful edits are the productive work. A
    model never run returns empties (a fresh specimen)."""
    try:
        from . import profile
        c = profile.load(model_name).get("counts", {})
        edits = int(c.get("exact_edit", 0)) + int(c.get("fuzzy_edit", 0))
        return {"edits": edits, "sessions": int(c.get("session", 0))}
    except Exception:
        return {}


def collection(installed_only=True):
    """Every model this install has forged into an agent, as a card. With installed_only,
    only the models actually pulled on this machine."""
    from . import registry
    from .models_cmd import _installed_tags, _is_installed
    from . import config as _cfg
    fid = forge_id()
    installed = _installed_tags(_cfg.load().get("engine", "ollama"))
    out = []
    for m in registry.MODELS:
        here = _is_installed(m["name"], installed)
        if installed_only and not here:
            continue
        c = card(m, fid=fid, telemetry=_telemetry(m["name"]))
        c["_installed"] = here
        out.append(c)
    return fid, out, len(registry.MODELS)


def render_roster(fid, owned, total):
    """The collection: one line per card, sorted by rarity then Forge Rating, with how many
    of the catalog you've forged and your Forge ID."""
    order = {t: i for i, (_, t) in enumerate(RATING_TIERS)}
    owned = sorted(owned, key=lambda c: (order.get(c["rarity"], 9), -c["rating"]))
    lines = [_r.paint(f"forge roster — forge {fid[:8]}", "bold"),
             _r.paint(f"  {len(owned)} forged · {total} in the catalog", "dim"), ""]
    foils = sum(1 for c in owned if c["foil"])
    for c in owned:
        mark = _r.paint("✦", "yellow") if c["foil"] else " "
        rs = _RARITY_STYLE.get(c["rarity"], "dim")
        badge = _r.paint(f"{c['rarity'][:4].upper():4}", rs)
        name = _r.fit(c["name"], 22)
        lines.append(f"  No.{c['no']:03d} {mark} {name:<22}  {badge}  M{c['mastery']:<3}  "
                     + _r.paint(f"FR {c['rating']}  {'/'.join(c['classes'])}", "dim"))
    if foils:
        lines += ["", _r.paint(f"  ✦ {foils} foil", "yellow")]
    return "\n".join(lines)
