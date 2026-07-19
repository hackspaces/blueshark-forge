"""Model cards — a Pokémon-authentic, deterministic-yet-unique profile of every model a
user runs, built from the model's REAL attributes and the user's REAL work.

The science is Pokémon's, faithfully: six stats (HP/Atk/Def/SpA/SpD/Spe), IVs 0-31, 25
natures (+10%/-10% one stat, never HP), the Gen-3 stat formula, and a shiny roll from the
Trainer-ID ⊕ PID trick (~1/4096). The *randomness* is deterministic — a specimen is a pure
function of (trainer_id, model), so it never changes and can be regenerated anywhere (the
website, a reshare) from those inputs alone. The *uniqueness* is guaranteed: the Trainer ID
is a 128-bit value minted once per install, so no two people ever roll the same specimen —
Pokémon's own mechanic, with a wide enough ID that its real-game TID collisions can't happen.

What's the model vs what's the trainer:
  - SPECIES (base stats, types, dex no.) come from the model's real metadata — params, size,
    harness-lift, whether forge has verified it. A 0.5B is a fast glass-cannon; a 70B a slow
    tanky powerhouse (Speed falls with size, like a real early- vs late-route Pokémon).
  - The INDIVIDUAL (IVs, nature, shininess) is fixed at first encounter from H(trainer_id‖model)
    — your unique specimen of that species.
  - LEVEL and EVs grow with your actual WORK (verified tasks / sessions) — training, the way
    a Pokémon levels. Use the model more, the card grows.
"""
import hashlib
import json
import math
import os

FORGE_DIR = os.path.expanduser("~/.forge")
_TRAINER_PATH = os.path.join(FORGE_DIR, "trainer.json")

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
STAT_LABELS = {"hp": "HP", "atk": "Attack", "def": "Defense",
               "spa": "Sp.Atk", "spd": "Sp.Def", "spe": "Speed"}

# The 25 natures in canonical index order. up/down index into the FIVE nature-affected stats
# (never HP): [atk, def, spe, spa, spd]. up==down → neutral (×1.0 everywhere).
_NATURE_STATS = ("atk", "def", "spe", "spa", "spd")
_NATURE_NAMES = (
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty",
    "Bold", "Docile", "Relaxed", "Impish", "Lax",
    "Timid", "Hasty", "Serious", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Bashful", "Rash",
    "Calm", "Gentle", "Sassy", "Careful", "Quirky",
)

SHINY_THRESHOLD = 16          # (tid ⊕ pid) & 0xFFFF < 16  →  1/4096, the modern shiny rate

RARITY_TIERS = [              # by base-stat total, Pokémon-style
    (600, "legendary"), (540, "epic"), (460, "rare"), (380, "uncommon"), (0, "common"),
]


# ---- Trainer ID: the per-install uniqueness root ----------------------------
def trainer_id():
    """This install's 128-bit Trainer ID, minted once and reused forever. It is the
    guarantee that no two people ever generate the same specimens — every card the user
    owns is seeded from it, and 128 bits makes a collision across all installs impossible
    in practice (unlike Pokémon's 16-bit TID, which really does collide)."""
    try:
        with open(_TRAINER_PATH) as f:
            tid = json.load(f).get("tid")
            if tid:
                return tid
    except (OSError, ValueError):
        pass
    tid = os.urandom(16).hex()                 # 128 bits
    try:
        os.makedirs(FORGE_DIR, exist_ok=True)
        with open(_TRAINER_PATH, "w") as f:
            json.dump({"tid": tid}, f)
    except OSError:
        pass
    return tid


def _h(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest(), 16)


# ---- species (from the model's real attributes) -----------------------------
def _clamp(v, lo=15, hi=200):
    return max(lo, min(hi, int(round(v))))


def species(model):
    """Base stats + types + dex number for a model, DETERMINISTIC from its real metadata.
    `model` is a registry entry (or any dict with params_b / weights / lift_pts / status).
    Speed falls with size (small = fast); power/HP rise with it; harness-lift and a forge-
    verified ★ harden Defense/Sp.Def — the card rewards models forge has actually proven."""
    p = float(model.get("params_b") or 1)
    size = float((model.get("weights") or {}).get("size_gb") or p)
    lift = model.get("lift_pts") or 0
    verified = model.get("status") == "verified"
    reasoning = bool(model.get("reasoning"))
    name = model.get("name", "?")
    is_code = "cod" in (name + " " + (model.get("notes") or "")).lower()

    power = 40 + 60 * math.log10(1 + p)                       # rises with params
    base = {
        "hp":  _clamp(45 + 6 * math.log10(1 + size) * 10 / 3 + p),   # endurance ~ size
        "atk": _clamp(power + (15 if is_code else 0)),
        "def": _clamp(power * 0.8 + 3 * lift + (20 if verified else 0)),
        "spa": _clamp(power + (18 if reasoning else 0)),
        "spd": _clamp(power * 0.8 + 2 * lift + (15 if verified else 0)),
        "spe": _clamp(210 - 90 * math.log10(1 + p)),          # FALLS with size — small is fast
    }
    return {"name": name, "base": base, "bst": sum(base.values()),
            "types": _types(model, is_code, reasoning),
            "dex": _h("dex", name) % 1000 + 1}                # a stable dex number


def _types(model, is_code, reasoning):
    p = float(model.get("params_b") or 1)
    types = []
    if is_code:
        types.append("Steel")                                 # precise, hard-edged
    if reasoning:
        types.append("Psychic")
    if (model.get("kind") == "moe"):
        types.append("Electric")                              # sparky, sparse-active
    if p >= 30 and "Dragon" not in types:
        types.append("Dragon")                                # big, rare
    if p < 2:
        types.append("Flying")                                # tiny, nimble
    if not types:
        types.append("Normal")
    return types[:2]


def rarity(bst, verified=False):
    for cutoff, tier in RARITY_TIERS:
        if bst >= cutoff:
            return tier
    return "common"


# ---- the individual (fixed at first encounter, unique per trainer+model) -----
def pid(tid, model_name):
    """The 32-bit Personality Value — fixed per (trainer, model). Drives nature, IVs, and
    (with the Trainer ID) shininess, exactly as Pokémon's PID does."""
    return _h("pid", tid, model_name) & 0xFFFFFFFF


def nature(pid_val):
    """(name, up_stat, down_stat) — nature = PID % 25. Neutral natures have up==down."""
    i = pid_val % 25
    up, down = _NATURE_STATS[i // 5], _NATURE_STATS[i % 5]
    return _NATURE_NAMES[i], (None, None) if up == down else (up, down)


def ivs(pid_val):
    """The six Individual Values (0-31), the genetic uniqueness — derived from the PID via
    a hash chain so two trainers' specimens of the same species differ."""
    h = hashlib.sha256(f"iv:{pid_val}".encode()).digest()
    return {s: h[i] % 32 for i, s in enumerate(STATS)}


def is_shiny(tid, pid_val):
    """Pokémon's shiny test, adapted: (TID ⊕ PID) low bits < threshold → ~1/4096.
    Deterministic per (trainer, model); you either own a shiny of it or you don't."""
    tid_int = int(tid, 16) if isinstance(tid, str) else int(tid)
    thi, tlo = (tid_int >> 16) & 0xFFFF, tid_int & 0xFFFF
    phi, plo = (pid_val >> 16) & 0xFFFF, pid_val & 0xFFFF
    return (thi ^ tlo ^ phi ^ plo) < SHINY_THRESHOLD


# ---- training (level + EVs grow with real work) ------------------------------
def training(telemetry):
    """Level (1-100) and an EV spread from the user's REAL work with the model — the
    'seeds of work'. `telemetry` is a dict of counts (verified tasks, sessions, actions).
    A never-used model is a freshly-encountered Lv.5; work levels it and pours EVs into the
    stat it exercised most. Empty telemetry → the base specimen."""
    verified = int((telemetry or {}).get("verified", 0))
    sessions = int((telemetry or {}).get("sessions", 0))
    level = max(1, min(100, 5 + 3 * verified + sessions))
    # EVs earned by work, weighted toward the stat the model proved (verified → def/spd,
    # raw runs → atk/spa), capped like the game (252/stat, 510 total).
    evs = {s: 0 for s in STATS}
    earned = min(510, 8 * (verified * 2 + sessions))
    evs["def"] = evs["spd"] = min(252, earned // 3)
    evs["atk"] = evs["spa"] = min(252, earned // 4)
    return level, evs


# ---- the Gen-3 stat formula --------------------------------------------------
def _stat(stat, base, iv, ev, level, nat_up, nat_down):
    core = ((2 * base + iv + ev // 4) * level) // 100
    if stat == "hp":
        return core + level + 10
    val = core + 5
    if stat == nat_up:
        val = (val * 110) // 100
    elif stat == nat_down:
        val = (val * 90) // 100
    return val


def card(model, tid=None, telemetry=None):
    """Assemble the full model card: species (from the model) + this trainer's unique
    individual (IVs/nature/shiny) + training (level/EVs from work) → the six computed stats.
    Fully determined by (tid, model, telemetry), so it regenerates identically anywhere."""
    tid = tid or trainer_id()
    sp = species(model)
    pv = pid(tid, sp["name"])
    nat_name, (up, down) = nature(pv)
    iv = ivs(pv)
    level, evs = training(telemetry)
    stats = {s: _stat(s, sp["base"][s], iv[s], evs[s], level, up, down) for s in STATS}
    return {
        "name": sp["name"], "dex": sp["dex"], "types": sp["types"],
        "base": sp["base"], "bst": sp["bst"],
        "rarity": rarity(sp["bst"], model.get("status") == "verified"),
        "shiny": is_shiny(tid, pv),
        "nature": nat_name, "nature_up": up, "nature_down": down,
        "ivs": iv, "evs": evs, "level": level, "stats": stats,
        "iv_total": sum(iv.values()),
    }
