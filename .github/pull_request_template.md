<!--
Thanks for contributing to BlueShark Forge! Keep PRs to one focused change.
Delete any section that doesn't apply.
-->

## What & why

<!-- One or two sentences: what does this change do, and why? -->

## Changes

<!-- The notable changes, as bullets. -->
-

## Testing

<!-- How did you verify this? The full suite runs with: python -m unittest discover -s tests -->
- [ ] `python -m unittest discover -s tests` passes locally
- [ ] Added or updated tests for the change (and failure paths, not just the happy path)

## Notes

<!--
- Compatibility: does this change transcript/record shapes, replay, or public behavior?
  Old transcripts and fixtures must stay readable.
- Runtime stays standard-library only (no new third-party runtime dependencies).
- If this implements a roadmap slice, name it (e.g. "H0x") and confirm its acceptance checks.
-->

## Checklist

- [ ] Single, focused change (no unrelated refactors bundled in)
- [ ] No new third-party **runtime** dependencies
- [ ] Backwards compatible, or the break is called out above
- [ ] Docs/README updated if user-visible behavior changed
