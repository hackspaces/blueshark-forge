#!/usr/bin/env python3
"""Generate CHANGELOG.md + site/changelog.html from the GitHub Releases.

The releases are the source of truth — they are what the tag-gated publish actually
shipped, and they carry the hand-written notes. Deriving both files from them means
the changelog can never drift from reality the way a hand-maintained list does.

    python3 tools/changelog.py            # write both files
    python3 tools/changelog.py --check    # non-zero if they're stale (for CI)

Stdlib only (it shells out to `gh`, which is already a release prerequisite). This is
a dev tool — it is not shipped in the package.
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "CHANGELOG.md"
HTML = ROOT / "site" / "changelog.html"


def releases():
    """Every published release, newest first."""
    out = subprocess.run(
        ["gh", "release", "list", "--limit", "100",
         "--json", "tagName,name,publishedAt,isPrerelease"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rels = [r for r in json.loads(out) if not r.get("isPrerelease")]
    rels.sort(key=lambda r: _ver(r["tagName"]), reverse=True)
    for r in rels:
        body = subprocess.run(
            ["gh", "release", "view", r["tagName"], "--json", "body", "-q", ".body"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        r["body"] = body
    return rels


def _ver(tag):
    return tuple(int(x) for x in re.findall(r"\d+", tag)[:3]) or (0,)


def _headline(r):
    """'v0.11.0 — the thesis phases are done' → 'the thesis phases are done'."""
    name = (r.get("name") or "").strip()
    m = re.match(r"^v?[\d.]+\s*[—–-]\s*(.+)$", name)
    return (m.group(1) if m else name) or r["tagName"]


def _date(r):
    return (r.get("publishedAt") or "")[:10]


# ---- markdown ---------------------------------------------------------------

def render_md(rels):
    out = ["# Changelog", "",
           "Every published release of `blueshark-forge`, newest first.",
           "",
           "Generated from the GitHub Releases by `tools/changelog.py` — the releases are",
           "what the tag-gated publish actually shipped, so this cannot drift. Don't hand-edit.",
           ""]
    for r in rels:
        out.append(f"## [{r['tagName']}](https://github.com/hackspaces/blueshark-forge/releases/tag/{r['tagName']}) — {_headline(r)}")
        out.append("")
        out.append(f"*{_date(r)} · `pip install blueshark-forge=={r['tagName'].lstrip('v')}`*")
        out.append("")
        body = (r.get("body") or "").strip()
        if body:
            out.append(body)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---- html -------------------------------------------------------------------

def _md_to_html(md):
    """A deliberately small markdown subset — enough for release notes, no deps.
    Everything is escaped first, so notes can never inject markup into the page."""
    lines = (md or "").split("\n")
    out, in_code, in_list = [], False, False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for ln in lines:
        if ln.strip().startswith("```"):
            close_list()
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(ln))
            continue
        s = html.escape(ln.strip())
        if not s:
            close_list()
            continue
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            close_list()
            lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{m.group(2)}</h{lvl}>")
            continue
        if re.match(r"^[-*]\s+", s):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + re.sub(r"^[-*]\s+", "", s) + "</li>")
            continue
        if s.startswith("&gt;"):
            close_list()
            out.append(f"<blockquote>{s[4:].strip()}</blockquote>")
            continue
        close_list()
        out.append(f"<p>{s}</p>")
    close_list()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def render_html(rels):
    items = []
    for r in rels:
        tag = html.escape(r["tagName"])
        items.append(f"""
      <article class="rel" id="{tag}">
        <header>
          <a class="tag" href="https://github.com/hackspaces/blueshark-forge/releases/tag/{tag}">{tag}</a>
          <h2>{html.escape(_headline(r))}</h2>
          <p class="meta">{_date(r)} · <code>pip install blueshark-forge=={tag.lstrip('v')}</code></p>
        </header>
        <div class="body">
{_md_to_html(r.get('body') or '')}
        </div>
      </article>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>forge — changelog</title>
<meta name="description" content="Every published release of forge, newest first.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%23111'/%3E%3Ctext x='16' y='23' font-family='monospace' font-size='20' fill='%23fff' text-anchor='middle'%3Ef%3C/text%3E%3C/svg%3E">
<style>
  :root{{
    --bg:#ffffff; --ink:#111111; --muted:#666666; --faint:#8a8a8a; --line:#e2e2e2; --inline:#f4f4f4;
    --inv-bg:#111111; --inv-ink:#f4f4f4;
  }}
  @media (prefers-color-scheme:dark){{
    :root{{
      --bg:#0c0c0c; --ink:#ededed; --muted:#9a9a9a; --faint:#6d6d6d; --line:#242424; --inline:#181818;
      --inv-bg:#ededed; --inv-ink:#0c0c0c;
    }}
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0}}
  body{{
    background:var(--bg); color:var(--ink);
    font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,"Liberation Mono",monospace;
    font-size:14px; line-height:1.65; -webkit-font-smoothing:antialiased;
  }}
  a{{color:inherit; text-underline-offset:2px}}
  a:hover{{background:var(--ink); color:var(--bg); text-decoration:none}}
  ::selection{{background:var(--ink); color:var(--bg)}}
  :focus-visible{{outline:2px solid var(--ink); outline-offset:2px}}

  .top{{display:flex; align-items:center; gap:1rem; padding:1rem 1.6rem; border-bottom:1px solid var(--line)}}
  .top .logo{{font-weight:700; font-size:1.05rem; letter-spacing:-.02em; text-decoration:none}}
  .top .logo b{{background:var(--ink); color:var(--bg); padding:0 .18em}}
  .top .sp{{flex:1}}
  .top a.nav{{margin-left:1.3rem; color:var(--faint); font-size:.85rem}}

  main{{max-width:760px; margin:0 auto; padding:2.4rem 1.6rem 4rem}}
  h1{{font-size:1.9rem; letter-spacing:-.03em; margin:0}}
  .lede{{color:var(--muted); margin:.7rem 0 0}}

  .rel{{border-top:1px solid var(--line); margin-top:2.4rem; padding-top:1.6rem}}
  .rel:first-of-type{{border-top:0}}
  .tag{{display:inline-block; background:var(--inv-bg); color:var(--inv-ink); padding:.1rem .5rem;
        font-size:.8rem; text-decoration:none; margin-bottom:.5rem}}
  .rel h2{{font-size:1.15rem; letter-spacing:-.02em; margin:.3rem 0 0; font-weight:700}}
  .rel .meta{{color:var(--faint); font-size:.8rem; margin:.35rem 0 0}}
  .rel .body{{margin-top:1rem}}
  .rel .body h3,.rel .body h4,.rel .body h5,.rel .body h6{{
    font-size:.92rem; margin:1.4rem 0 .4rem; letter-spacing:-.01em}}
  .rel .body p{{margin:.6rem 0; color:var(--muted)}}
  .rel .body strong{{color:var(--ink)}}
  .rel .body ul{{margin:.6rem 0; padding-left:1.1rem}}
  .rel .body li{{color:var(--muted); margin:.25rem 0}}
  .rel .body blockquote{{margin:.8rem 0; padding-left:.9rem; border-left:2px solid var(--line); color:var(--faint)}}
  code{{background:var(--inline); padding:.05em .3em; font-size:.88em}}
  /* wide content scrolls in its own box — the page body never scrolls sideways */
  pre{{background:var(--inv-bg); color:var(--inv-ink); padding:.85rem 1rem; overflow-x:auto; margin:.8rem 0}}
  pre code{{background:none; padding:0; font-size:.8rem; white-space:pre}}
  table{{display:block; overflow-x:auto; border-collapse:collapse; margin:.8rem 0}}

  footer{{border-top:1px solid var(--line); padding:1rem 1.6rem; color:var(--faint); font-size:.78rem;
    display:flex; flex-wrap:wrap; gap:.3rem 1rem}}

  @media (max-width:560px){{
    .top{{padding:.85rem 1.15rem}} .top a.nav{{margin-left:1rem; font-size:.78rem}}
    main{{padding:1.6rem 1.15rem 3rem}} h1{{font-size:1.5rem}}
    footer{{padding:.85rem 1.15rem; font-size:.72rem}}
  }}
</style>
</head>
<body>
  <div class="top">
    <a class="logo" href="/forge/"><b>forge</b></a>
    <span class="sp"></span>
    <a class="nav" href="/forge/">home</a>
    <a class="nav" href="https://github.com/hackspaces/blueshark-forge">github</a>
    <a class="nav" href="https://pypi.org/project/blueshark-forge/">pypi</a>
  </div>

  <main>
    <h1>Changelog</h1>
    <p class="lede">Every published release, newest first. Each one shipped through the
    tag-gated pipeline: full test matrix, tag&#8202;=&#8202;version assertion, wheel smoke-test,
    then PyPI.</p>
{"".join(items)}
  </main>

  <footer>
    <span>model-agnostic agentic runtime · stdlib-only, zero runtime deps · MIT</span>
    <span class="sp"></span>
    <span>generated from the GitHub Releases</span>
  </footer>
</body>
</html>
"""


def main():
    check = "--check" in sys.argv
    rels = releases()
    if not rels:
        print("no releases found", file=sys.stderr)
        return 1
    md, page = render_md(rels), render_html(rels)
    if check:
        stale = [p.name for p, new in ((MD, md), (HTML, page))
                 if not p.exists() or p.read_text() != new]
        if stale:
            print("stale (run `python3 tools/changelog.py`): " + ", ".join(stale), file=sys.stderr)
            return 1
        print("changelog up to date")
        return 0
    MD.write_text(md)
    HTML.write_text(page)
    print(f"wrote {MD.relative_to(ROOT)} and {HTML.relative_to(ROOT)} — {len(rels)} releases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
