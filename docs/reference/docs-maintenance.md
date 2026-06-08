# Docs Maintenance

The docs are VitePress Markdown plus a cheap repository-specific content
checker.

## Cheap Content Check

From the repository root:

```bash
uv run python docs/scripts/check_snippets.py
```

This check validates:

- Python fenced code syntax;
- Bash fenced code syntax;
- frontmatter scalar quoting;
- forbidden public-doc patterns;
- policy catalog and codec registry smoke checks.

## VitePress Build

From `docs/`:

```bash
npm ci --no-audit --no-fund
npm run docs:build
```

Use a working Node 20 or newer environment if the system Node is broken or too
old for Vite.

## Content Rules

- Public docs should describe stable interfaces, not private launch scripts.
- Keep benchmark rollout details in `praxis-eval` docs.
- Keep heavy cluster-specific commands out of the core onboarding path.
- Update [Policy Cells](/foundations/policy-cells) when the public matrix
  changes.
