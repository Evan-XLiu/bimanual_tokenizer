#!/usr/bin/env python3
"""Cheap documentation checks that do not require training or simulators."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SRC = ROOT / "src"
CODEC_FACTORY = SRC / "praxis" / "codecs" / "factory.py"
POLICY_CATALOG = SRC / "praxis" / "policies" / "catalog.py"
FORBIDDEN_MARKDOWN_PATTERNS = {
    ".venv" + "/bin/": "use `uv run ...` in public docs",
    "libero" + "_10": "use the full `libero` dataset/task name in public docs",
    "+" + "server_address=": (
        "`server_address` is a declared config key; do not prefix it with `+`"
    ),
    "scripts" + "/shell": "keep operational launch details out of public docs",
}


def markdown_paths() -> list[Path]:
    paths = [ROOT / "README.md"]
    paths.extend(DOCS.rglob("*.md"))
    return [
        path
        for path in sorted(paths)
        if "node_modules" not in path.parts
        and ".vitepress" not in path.parts
        and path.is_file()
    ]


def docs_markdown_paths() -> list[Path]:
    return [
        path
        for path in sorted(DOCS.rglob("*.md"))
        if "node_modules" not in path.parts
        and ".vitepress" not in path.parts
        and path.is_file()
    ]


def check_python_fences() -> None:
    failures: list[str] = []
    fence_count = 0

    for path in markdown_paths():
        rel = path.relative_to(ROOT)
        in_block = False
        language = ""
        start_line = 0
        buffer: list[str] = []

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.startswith("```"):
                if not in_block:
                    in_block = True
                    language = line[3:].strip().split()[0] if line[3:].strip() else ""
                    start_line = line_number + 1
                    buffer = []
                    continue

                if language == "python":
                    fence_count += 1
                    code = "\n".join(buffer) + "\n"
                    try:
                        ast.parse(code, filename=f"{rel}:{start_line}", mode="exec")
                    except SyntaxError as exc:
                        failures.append(f"{rel}:{start_line}: {exc.msg}")

                in_block = False
                language = ""
                start_line = 0
                buffer = []
                continue

            if in_block:
                buffer.append(line)

        if in_block:
            failures.append(f"{rel}:{start_line}: unclosed fenced code block")

    if failures:
        print("Python fence syntax failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Compiled {fence_count} Python fenced code blocks.")


def check_bash_fences() -> None:
    failures: list[str] = []
    fence_count = 0

    for path in markdown_paths():
        rel = path.relative_to(ROOT)
        in_block = False
        language = ""
        start_line = 0
        buffer: list[str] = []

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.startswith("```"):
                if not in_block:
                    in_block = True
                    language = line[3:].strip().split()[0] if line[3:].strip() else ""
                    start_line = line_number + 1
                    buffer = []
                    continue

                if language == "bash":
                    fence_count += 1
                    proc = subprocess.run(
                        ["bash", "-n"],
                        input="\n".join(buffer) + "\n",
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        message = proc.stderr.strip() or "bash -n failed"
                        failures.append(f"{rel}:{start_line}: {message}")

                in_block = False
                language = ""
                start_line = 0
                buffer = []
                continue

            if in_block:
                buffer.append(line)

    if failures:
        print("Bash fence syntax failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Checked {fence_count} Bash fenced code blocks.")


def check_frontmatter_scalars() -> None:
    failures: list[str] = []
    for path in markdown_paths():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "---":
            continue
        for line_number, line in enumerate(lines[1:], 2):
            if line == "---":
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ": " not in stripped:
                continue
            value = stripped.split(": ", 1)[1].strip()
            if not value or value[0] in "\"'[{>|&*!":
                continue
            if ": " in value:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: "
                    "quote frontmatter scalar values containing ': '"
                )
    if failures:
        print("Frontmatter scalar failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Checked Markdown frontmatter scalar quoting.")


def check_forbidden_markdown_patterns() -> None:
    failures: list[str] = []
    for path in docs_markdown_paths():
        text = path.read_text(encoding="utf-8")
        for pattern, message in FORBIDDEN_MARKDOWN_PATTERNS.items():
            if pattern in text:
                failures.append(
                    f"{path.relative_to(ROOT)}: contains {pattern!r}; {message}"
                )
    if failures:
        print("Forbidden Markdown patterns:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Checked forbidden Markdown patterns.")


def codec_registry_names_from_source() -> tuple[str, ...]:
    tree = ast.parse(
        CODEC_FACTORY.read_text(encoding="utf-8"), filename=str(CODEC_FACTORY)
    )
    for node in tree.body:
        value: ast.expr | None = None
        if (
            (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "_CODEC_REGISTRY"
            )
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_CODEC_REGISTRY"
                for target in node.targets
            )
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            raise SystemExit("_CODEC_REGISTRY must be a dict literal")
        names: list[str] = []
        for key in value.keys:
            if not isinstance(key, ast.Tuple) or len(key.elts) != 2:
                raise SystemExit("_CODEC_REGISTRY keys must be (kind, name) tuples")
            kind_node, name_node = key.elts
            if not isinstance(kind_node, ast.Constant) or not isinstance(
                name_node, ast.Constant
            ):
                raise SystemExit("_CODEC_REGISTRY keys must be string constants")
            names.append(f"{kind_node.value}/{name_node.value}")
        return tuple(sorted(names))
    raise SystemExit("Could not find _CODEC_REGISTRY in codec factory")


def run_registry_smoke() -> None:
    spec = importlib.util.spec_from_file_location(
        "_praxis_docs_policy_catalog", POLICY_CATALOG
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load policy catalog from {POLICY_CATALOG}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    names = module.supported_policy_names()
    if "bar_oat_pali" not in names:
        raise SystemExit("bar_oat_pali missing from policy catalog")
    codecs = codec_registry_names_from_source()
    if "tokens/oat" not in codecs:
        raise SystemExit("tokens/oat missing from codec registry")
    print(f"Checked policy catalog / codec registry source: {len(names)} {len(codecs)}")


def main() -> None:
    check_python_fences()
    check_bash_fences()
    check_frontmatter_scalars()
    check_forbidden_markdown_patterns()
    run_registry_smoke()


if __name__ == "__main__":
    main()
