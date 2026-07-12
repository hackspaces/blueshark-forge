# Releasing forge

Releasing is **one push of a version tag**. Everything else — tests, coherence
checks, the PyPI upload, and the GitHub Release — is automated by
`.github/workflows/publish.yml`, which triggers on any `v*` tag.

## Cut a release

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
| `tests` | the full Python 3.10–3.13 suite is green (the same `test.yml` every commit runs) |
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
- `pypi` duplicate version → that version already shipped; bump and tag again.
