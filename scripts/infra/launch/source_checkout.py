# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path


def install_source_checkout_imports(infra_dir: Path) -> None:
    """Make launcher modules work from an uninstalled source checkout.

    Cluster entrypoints are often run as files from ``scripts/infra`` before
    the repo is installed as a package. Adding both the launcher package and
    ``src`` keeps imports identical between local dry-runs and submitted jobs.
    """
    repo_root = infra_dir.parents[1]
    for path in (infra_dir, repo_root / "src"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
