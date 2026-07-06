# Contributing to forge

Thanks for wanting to help. forge aims to be a small, sharp, dependency-free
agentic runtime — contributions should keep it that way.

## Ground rules

- **Stdlib only.** No third-party runtime dependencies. If you think you need one,
  open an issue first to discuss.
- **Tests are required.** Every behavior change or bug fix comes with a test in
  `tests/test_forge.py`. If you fix an edge case, add a test that fails without the fix.
- **Keep it readable.** Match the surrounding style: clear names, short functions,
  comments that explain *why*, not *what*.
- **Security matters.** The `bash` tool is intentionally unsandboxed, but file
  tools are confined and the fleet inbox is authenticated — don't weaken those
  without discussion. See the Security section of the README.

## Workflow

1. **Fork** the repo and create a branch off `main`:
   `git checkout -b fix/short-description`
2. Make your change **with a test**.
3. Run the suite locally — it must pass:
   ```bash
   python -m unittest discover -s tests
   ```
4. **Open a pull request** against `main` with a clear description of what and why.
5. CI runs the tests on Python 3.10–3.13. A maintainer reviews and merges.

`main` is protected: no direct pushes, PRs require a passing CI run and review.

## Good first contributions

- More engine presets / better error messages for a specific inference server.
- Additional tools (e.g. a proper `apply_patch`) behind the constrained schema.
- Tests for under-covered paths (the fleet daemon, the TUI line editor).
- Docs and examples for running forge against vLLM / MLX / llama.cpp.

## Reporting bugs & security issues

Open a GitHub issue. For security-sensitive reports, note it clearly in the title
so a maintainer can respond before broad disclosure.

## License

By contributing you agree your contributions are licensed under the MIT License.
