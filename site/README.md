# forge — landing site

The public landing page for forge. Deliberately simple: one static `index.html`
(no build, no framework) plus `install.sh`, so a single origin serves both the
page and the one-line installer.

## Deploy (Vercel, straight from this repo)

1. Vercel → **New Project** → import `hackspaces/blueshark-forge`.
2. **Root Directory** → `site`   ·   **Framework Preset** → *Other* (no build step).
3. Deploy. Vercel then auto-redeploys on every push to `main` — the site is built
   from the repo, nothing to hand-upload.
4. Add your domain under the project's **Domains** (e.g. `topk1.com`).

`vercel.json` rewrites `/forge` and `/forge/*` back to the site root, so forge lives
under a **path** and the domain root stays free for anything else:

- `https://topk1.com/forge/` — the landing page
- `curl -fsSL https://topk1.com/forge/install.sh | sh` — installs the `forge` CLI

(The domain root `https://topk1.com/` and the bare `.vercel.app` URL still serve the
page too — the rewrite only *adds* the `/forge` path.)

## Files

- `index.html` — the landing page (static; forge's gold/Geist visual language).
- `install.sh` — the one-line installer (also the raw-GitHub install source).
