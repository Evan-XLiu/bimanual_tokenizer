# Documentation Checks

Docs live under:

```text
docs/
```

They use VitePress.

## Cheap Content Check

From the repo root:

```bash
uv run python docs/scripts/check_snippets.py
```

This check is what CI or reviewers should run before a docs-only change when
Node/npm is unavailable.

## VitePress Build

From `docs/`:

```bash
npm install
npm run docs:build
```

`npm run docs:check` is an npm wrapper around the same Python script:

```bash
npm run docs:check
```

The cheap check verifies:

- Python fenced code blocks parse;
- Bash fenced code blocks pass `bash -n`;
- public Markdown avoids stale command patterns such as direct virtualenv
  executable paths, narrow LIBERO-suite examples, and dynamic-prefixing
  declared config keys;
- VitePress frontmatter scalar values containing `: ` are quoted;
- the policy catalog imports without heavyweight ML dependencies;
- the codec registry source contains expected entries;
- `bar_oat_pali` and `tokens/oat` are present.

`docs:build` verifies the VitePress site. It requires a working Node/npm
environment; if `npm --version` hangs on the cluster login image, use a Node
module or another environment where npm responds before running the build.

## Content Rules

- Explain the comparison axes first: method, backbone, codec.
- Use `bar_oat_pali` only as the quickstart/example path, not as the promoted
  project recommendation.
- Include math for BAR, FM, KI, and each codec family.
- Keep benchmark rollout details in `praxis-eval` docs.
- Keep operational launch details out of the conceptual documentation path.
- Prefer links to source files and Hydra config groups over copied implementation details.
- Publish through the GitHub Pages workflow, not directly from `main:/docs`.
