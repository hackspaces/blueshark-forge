"""forged — the fleet autopilot for forge. One loop, three layers, over every
live forge session: TRUST (verify done-claims) + COORDINATE (file collisions) +
LEARN (harvest & share facts). Model-agnostic; the checker/extractor model is
configurable (defaults to a small local one)."""
import hashlib
import json
import os
import sys
import time

from . import session as sessmod
from . import fleet
from .util import slurp

STATE = fleet.STATE


def _load(f, d):
    try:
        return json.loads(slurp(os.path.join(STATE, f)))
    except (OSError, json.JSONDecodeError):
        return d


def _save(f, v):
    with open(os.path.join(STATE, f), "w") as fh:
        json.dump(v, fh)


def _hash(s):
    return hashlib.md5(s.encode("utf-8", "replace")).hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Forged:
    def __init__(self, model, interval=20):
        self.model = model
        self.interval = interval

    def verify_pass(self):
        seen = _load("seen-claims.json", {})
        for e in sessmod.registry():
            if e["status"] == "working":
                continue
            text = fleet.last_say(e["sid"])
            if not text or not fleet.CLAIM_RE.search(text):
                continue
            h = _hash(text[:2000])
            if seen.get(e["sid"]) == h:
                continue
            seen[e["sid"]] = h; _save("seen-claims.json", seen)
            log(f'TRUST: "{e["name"]}" claims done → verifying ({self.model})')
            r = fleet.verify(text[:1500], e["cwd"], self.model)
            with open(fleet.RECEIPTS, "a") as f:
                f.write(json.dumps({"ts": time.time(), "sid": e["sid"], "cwd": e["cwd"],
                                    "verdict": r["verdict"], "evidence": r["evidence"]}) + "\n")
            if r["confirmed"]:
                log(f'  ✓ CONFIRMED ({e["name"]})')
            else:
                log(f'  ✗ {r["verdict"]} ({e["name"]}): {r["evidence"][:110]}')
                try:
                    fleet.send(e["sid"], f"[verify] Your completion claim failed independent verification. {r['evidence']} Please fix and re-check.", sender="verifier")
                except Exception:
                    pass

    def guard_pass(self):
        warned = _load("warned.json", {})
        live = sessmod.registry()
        owners = {}
        for e in live:
            for f in fleet.edited_files(e["sid"], e["cwd"]):
                owners.setdefault(f, []).append(e)
        for fpath, es in owners.items():
            uniq = {e["sid"]: e for e in es}.values()
            uniq = list(uniq)
            if len(uniq) < 2:
                continue
            key = fpath + "::" + "|".join(sorted(e["sid"] for e in uniq))
            if warned.get(key):
                continue
            warned[key] = time.time(); _save("warned.json", warned)
            log(f"COORDINATE: collision on {fpath} between {' & '.join(e['name'] for e in uniq)}")
            for e in uniq:
                others = ", ".join(o["name"] for o in uniq if o["sid"] != e["sid"])
                try:
                    fleet.send(e["sid"], f"[guard] {others} is also editing {fpath}. Coordinate before you commit.", sender="guard")
                except Exception:
                    pass

    def learn_pass(self):
        done = _load("harvested.json", {})
        live = sessmod.registry()
        for e in live:
            if e["status"] == "working":
                continue
            text = fleet.last_say(e["sid"]) or ""
            h = _hash(text[:2000] + e["sid"])
            if done.get(e["sid"]) == h:
                continue
            done[e["sid"]] = h; _save("harvested.json", done)
            fresh = fleet.harvest(e["sid"], e["cwd"], self.model)
            if not fresh:
                continue
            log(f'LEARN: "{e["name"]}" +{len(fresh)} fact(s): {fresh[0][:70]}')
            for peer in live:
                if peer["sid"] != e["sid"] and peer["cwd"] == e["cwd"]:
                    try:
                        fleet.send(peer["sid"], f"[learn] Another session here just learned: {' | '.join(fresh)}", sender="learn")
                    except Exception:
                        pass

    def tick(self):
        try:
            self.verify_pass(); self.guard_pass(); self.learn_pass()
        except Exception as ex:
            log(f"tick error: {ex}")

    def run(self):
        log(f"forged up (pid {os.getpid()}), every {self.interval}s, model {self.model}: TRUST + COORDINATE + LEARN")
        self.tick()
        if os.environ.get("FORGED_ONCE"):
            return
        while True:
            time.sleep(self.interval)
            self.tick()


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FORGE_VERIFIER_MODEL", "qwen3.6:latest")
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    Forged(model, interval).run()
