# Releasing forge

There are two ways to release, both fully automated after the trigger:

- **One click** (recommended) — run the `release` workflow. It does the manual
  half for you: computes + bumps the version, keeps `SECURITY.md` in sync, writes
  the notes from the commits since the last tag, then commits + tags + pushes.
- **By hand** — bump `__version__`, write the `release:` commit, push the tag.

Either way the `v*` tag triggers `publish.yml`, which runs the tests, proves the
tag matches the code, builds, and publishes to PyPI + creates the GitHub Release.

## One-click release (the `release` workflow)

From the **Actions → release → Run workflow** button (or `gh workflow run
release.yml -f bump=minor -f summary="..."`):

1. Pick a `bump`: `auto` (infer from commits — `feat` → minor, `fix` → patch,
   a breaking change → major), or force `patch` / `minor` / `major`.
2. Optionally give a one-line `summary` (becomes the release title suffix).

`.github/scripts/prep_release.py` then bumps `forge/__init__.py`, advances the
`SECURITY.md` supported-versions table to the new minor line, and renders the
`release:` commit notes from `git log <last-tag>..HEAD` (grouped by
`feat`/`fix`/`docs`). The workflow commits, tags, and pushes — and the tag
triggers `publish.yml`.

> **One-time setup:** add a repo secret **`RELEASE_PAT`** — a fine-grained PAT
> with **contents: write**. GitHub deliberately does not let a workflow's default
> token trigger another workflow, so without the PAT the release commit + tag are
> pushed but `publish.yml` won't fire. Push the tag yourself to ship
> (`git push origin vX.Y.Z --force`), or add the secret so it's fully hands-off.

The prep logic is unit-tested in `tests/test_prep_release.py` (version math, bump
inference, `SECURITY.md` sync, notes rendering) — no need to trust the YAML.

## Cut a release by hand

1. Bump the version in `forge/__init__.py`:

   ```python
   __version__ = "0.8.3"
   ```

2. Commit it as a `release:` commit whose **body is the release notes** (the
   pipeline uses this verbatim; the `Co-Authored-By:` trailer is stripped):

   ```
   release: v0.8.3 — <one-line summary>

   <the release notes — what changed and why, in the same voice as prior
   releases; this becomes the GitHub Release body>
   ```

3. Tag and push:

   ```bash
   git tag v0.8.3
   git push origin main v0.8.3
   ```

That's it. The tag push runs the pipeline. When it's green, `v0.8.3` is on PyPI
and its GitHub Release exists.

## What the pipeline does (and guarantees)

The `v*` tag triggers `publish.yml`, which runs four ordered, **fail-closed**
stages — if any stage fails, nothing downstream runs and nothing is published:

| Stage | What it proves |
|---|---|
| `tests` | the full Python 3.10–3.13 suite is green (the same `test.yml` that runs on every push to `main` and every PR) |
| `guard` | the tag equals `forge.__version__`, and the built wheel installs into a clean venv and starts the CLI (`forge --version` / `--help`) |
| `pypi` | the **vetted** wheel/sdist is published via PyPI trusted publishing (OIDC — no token to manage) |
| `github-release` | the GitHub Release is created, titled and noted from the `release:` commit body |

Because `github-release` runs **last**, a Release existing means the version is
already live on PyPI. Because `guard` asserts tag == version, a mistaken or
forgotten version bump **blocks the release** instead of shipping a mislabelled
artifact.

## Rules of thumb

- **The tag is the "go" signal.** Bumping `__version__` in a commit does nothing
  on its own — only pushing a `vX.Y.Z` tag releases. This is deliberate: you can
  land version-bearing work without shipping until you tag.
- **Tag name must match the code.** `v0.8.3` requires `__version__ == "0.8.3"`.
  A mismatch fails `guard` (loudly) rather than publishing.
- **Version is a clean `X.Y.Z`.** A malformed version fails the in-repo semver
  test before you ever tag.
- **No manual `gh release create`.** The pipeline owns the GitHub Release so its
  notes always match the release commit.
- **Re-releasing a version won't happen silently.** PyPI rejects a duplicate
  version, so a re-pushed tag fails at the publish step rather than clobbering.

## If a release fails

Read the failed stage in the Actions run:

- `guard` tag mismatch → fix `forge/__init__.py`, delete and re-push the tag.
- `tests` red → the release is correctly blocked; fix on `main`, re-tag.
- `github-release` fails *after* `pypi` already published → just re-push the same
  tag. The publish step is idempotent (`skip-existing`), so the re-run skips the
  already-uploaded artifact and completes the GitHub Release.
- `pypi` genuinely rejects (not a duplicate) → read the error; fix, bump, re-tag.
