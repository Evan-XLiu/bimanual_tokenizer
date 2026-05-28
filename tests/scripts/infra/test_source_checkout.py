from __future__ import annotations

import sys

from scripts.infra.launch.source_checkout import install_source_checkout_imports
from tests._paths import repo_root


def test_install_source_checkout_imports_adds_src_and_infra_once(
    monkeypatch,
) -> None:
    root = repo_root()
    infra_dir = root / "scripts" / "infra"
    monkeypatch.setattr(sys, "path", ["existing"])

    install_source_checkout_imports(infra_dir)
    install_source_checkout_imports(infra_dir)

    assert sys.path[:2] == [str(root / "src"), str(infra_dir)]
    assert sys.path.count(str(root / "src")) == 1
    assert sys.path.count(str(infra_dir)) == 1
    assert sys.path[2:] == ["existing"]
