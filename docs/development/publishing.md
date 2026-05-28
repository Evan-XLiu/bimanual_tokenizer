# Publishing Docs

Use GitHub Actions as the GitHub Pages publishing source.

Do not publish directly from `main:/docs`. The `docs/` directory contains
VitePress source files, not the static site. The static output is:

```text
docs/.vitepress/dist/
```

## One-Time GitHub Setting

In the GitHub repository:

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

After that, pushes to `main` build and deploy the site. Pull requests and
pushes to `doc` build the site without deploying it. Manual workflow runs can
deploy when the `deploy` input is enabled.

## Base Path

Project Pages usually serve at:

```text
https://<owner>.github.io/<repo>/
```

The workflow sets:

```text
VITEPRESS_BASE=/<repo>/
```

For this repository, that means `/praxis/`.

If the site later moves to a custom domain or a user/organization Pages root,
set the repository variable:

```text
VITEPRESS_BASE=/
```

## Workflow

The workflow lives at:

```text
.github/workflows/docs.yml
```

It performs six steps:

1. sets up Python for cheap docs checks;
2. installs Node dependencies in `docs/`;
3. runs `npm run docs:check`;
4. runs `npm run docs:build`;
5. uploads `docs/.vitepress/dist` as a Pages artifact for deployable runs;
6. deploys the artifact with `actions/deploy-pages`.

Pull requests build the docs but do not deploy them.

## Local Checks

The cheap docs content check does not require Node:

```bash
uv run python docs/scripts/check_snippets.py
```

The full local site build requires a working Node/npm environment:

```bash
cd docs
npm install
npm run docs:build
```
