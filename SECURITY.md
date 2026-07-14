# Security Policy

## Supported versions

BlueShark Forge is released from `main` and published to PyPI as
[`blueshark-forge`](https://pypi.org/project/blueshark-forge/). Only the latest
release line receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 0.10.x  | :white_check_mark: |
| < 0.10  | :x:                |

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report it privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the issue, the affected version, and a reproduction if you have one.

You can expect an initial response within a few days. If the report is accepted,
we will work on a fix, coordinate a disclosure timeline with you, and credit you in
the release notes unless you prefer to remain anonymous.

## Scope and threat model

Forge runs local models as autonomous agents against a workspace you point it at.
A few things worth knowing when assessing impact:

- **Authority is harness-owned, not model-owned.** `FORGE_AUTHORITY`
  (`observe | contribute | operator | admin`, default `operator`) gates which
  effects an action may have, independent of the model. A stronger or escalated
  model gains no extra capability. Admin-only shell (e.g. `sudo`, recursive-force
  deletes of absolute/home paths, remote-script pipes, secret-store reads) requires
  `FORGE_AUTHORITY=admin`.
- **Forge executes real actions with your OS privileges** (file writes, shell,
  network via tools). Run it in repositories and directories you trust, or constrain
  it with OS-level sandboxing and a lower `FORGE_AUTHORITY`.
- **Untrusted input is data, not authority.** Project file contents, fleet messages,
  and model output are treated as data; they do not grant capability.

Reports that most interest us: a path by which a model or crafted project/fleet
input escalates its own authority, forges completion evidence, or executes an
effect the configured authority level should have blocked.
