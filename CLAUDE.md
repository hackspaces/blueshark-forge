# Working on forge

Repo rules for agents (and humans) working in this repository. forge itself pins
this file into every session it runs here, so keep it short and true.

## What forge is

A model-agnostic agentic runtime. **Python 3.10+, stdlib-only, zero runtime
dependencies** — that constraint is load-bearing, not incidental. Don't add a
runtime dependency. Test-only or dev-only tooling is fine; anything a user's
`pip install blueshark-forge` would pull in is not.

The intelligence lives in the harness, not the weights: a small local model and a
frontier model go through the same loop, and the harness is what makes either one
capable. Authority is harness-owned (`FORGE_AUTHORITY`), never model-owned — see
`SECURITY.md`.

## Shipping: a merge is not the finish line

**A merged fix sitting unreleased on `main` is worthless to users.** Don't stop at
the merge. The full flow, in one go:

1. **Branch → PR.** Every PR carries a **label** (`enhancement` / `bug` /
   `documentation`, plus area labels) and the current **milestone**.
2. **Merge on green.** `main` requires the `test (3.12)` check.
3. **Cut the release** — immediately, in the same sitting (below).
4. **Verify the user-facing install path** — not just that CI was green (below).

Releases are cut **deliberately by hand**; there is intentionally no
tag-on-merge automation.

### Cutting a release

```bash
# on main, after the merge
#   1. bump the version
#      forge/__init__.py:  __version__ = "X.Y.Z"
#   2. its OWN commit — the GitHub Release takes its title from the tagged
#      commit's subject, so never tag a merge commit
git commit -am 'release: vX.Y.Z — <headline>'
git tag vX.Y.Z
git push origin main vX.Y.Z
```

The tag triggers `.github/workflows/publish.yml`, which runs the full test matrix,
**asserts the tag equals `forge.__version__`**, builds and smoke-tests the wheel in
a clean environment, publishes to PyPI via OIDC, and cuts the GitHub Release.

### Verifying the release actually reached users

CI going green is not proof a user can install it. Check the real front door:

```bash
# the canonical one-liner, in the installer's own no-side-effect mode
curl -fsSL https://topk1.com/forge/install.sh | FORGE_INSTALL_DRY_RUN=1 sh

# and that the published package is the new version
pip install --no-cache-dir --upgrade "blueshark-forge==X.Y.Z" && forge --version
```

**PyPI lags the upload, and lags itself.** The JSON API can report the new version
while pip still resolves the old one — they propagate separately, and pip reads the
*simple index*. `--no-cache-dir` does not help; only waiting does. So the check that
counts is a real install in a clean venv with **no version pin**:

```bash
python3 -m venv /tmp/v && /tmp/v/bin/pip install --no-cache-dir blueshark-forge
/tmp/v/bin/forge --version      # must equal the tag
```

Don't report a release as shipped until *that* serves it.

### Then publish the changelog

Once the release exists, regenerate it — it derives from the GitHub Releases, so it
can only be built *after* the tag ships:

```bash
python3 tools/changelog.py      # writes CHANGELOG.md + site/changelog.html
git commit -am 'docs: changelog for vX.Y.Z' && git push
```

`tools/changelog.py --check` fails if either file is stale. Never hand-edit them —
the releases are the source of truth, which is the whole point.

`https://topk1.com/forge/install.sh` is the **canonical install URL**. The repo
path `main/site/install.sh` serves the same file as a mirror. If you change one,
the README, `site/index.html`, and `site/README.md` must agree — they're the same
promise in four places.

## Tests

```bash
python3 -m unittest discover -s tests -q
```

Offline and stdlib-only, like the runtime. Tests must never launch a real model
server or hit the network — a past test did, and quietly ran a 30B model. Mock the
provisioning path.

### Python 3.10 is the floor, and your interpreter is probably newer

`requires-python = ">=3.10"`. A green local run proves nothing about 3.10/3.11 if
you're on 3.12+, because **newer syntax compiles silently for you and is a
SyntaxError for a supported user**. This has already shipped once: a quoted
literal inside an f-string expression (`f"{'a \"b\"':<{w}}"`) is fine from 3.12
(PEP 701) and a hard SyntaxError on 3.10/3.11 — it passed locally on 3.14 and
broke both older jobs.

There is no static guard for this: `ast.parse(..., feature_version=(3,10))` does
**not** catch it. The only real check is running an old interpreter:

```bash
python3.11 -m compileall -q forge/ tests/     # syntax floor
python3.11 -m unittest discover -s tests -q   # and the suite
```

CI runs the full 3.10–3.13 matrix. **Read every matrix job before merging, not
just the one required check** — `test (3.12)` is the only *required* status, so a
3.10/3.11 failure will not block the merge button. Green-on-3.12 is not green.

## The site

`site/` is a static, no-build landing page deployed by Vercel from `main`
(Root Directory `site`). Two constraints that are easy to break:

- **Desktop is deliberately ONE viewport with no scroll.** It's built on
  `html,body{height:100%}` + a flex column.
- **That frame cannot survive a phone** — below 820px the layout drops the fixed
  height and scrolls as a normal document. Use `minmax(0,1fr)`, never a bare
  `1fr`: a grid track's `auto` minimum won't shrink below its content, and the
  terminal panel holds a `white-space:nowrap` table that will otherwise set the
  width of the entire page.

Verify at 390/360/320px after touching the CSS; horizontal overflow must be 0.
