"""Lightweight distributed smoke test for torchrun launch behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("PRAXIS_RUN_DISTRIBUTED_SMOKE") != "1",
    reason="Set PRAXIS_RUN_DISTRIBUTED_SMOKE=1 to run distributed smoke tests.",
)
def test_torchrun_two_process_cpu_smoke(tmp_path: Path) -> None:
    """Launch 2 local torchrun processes and verify rank/world-size coordination."""
    script_path = tmp_path / "ddp_smoke_worker.py"
    out_dir = tmp_path / "out"
    script_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import json",
                "import os",
                "from pathlib import Path",
                "",
                "import torch.distributed as dist",
                "",
                "dist.init_process_group('gloo')",
                "out = Path(os.environ['PRAXIS_DDP_SMOKE_OUT'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "rank = int(os.environ['RANK'])",
                "world = int(os.environ['WORLD_SIZE'])",
                "(out / f'rank_{rank}.json').write_text(json.dumps({'rank': rank, 'world': world}) + '\\n')",
                "dist.barrier()",
                "if rank == 0:",
                "    files = sorted(out.glob('rank_*.json'))",
                "    if len(files) != world:",
                "        raise RuntimeError(f'Expected {world} rank files, found {len(files)}')",
                "    (out / 'main_ok.txt').write_text('ok\\n')",
                "dist.destroy_process_group()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PRAXIS_DDP_SMOKE_OUT"] = str(out_dir)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(script_path),
    ]
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"torchrun distributed smoke test failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    assert (out_dir / "main_ok.txt").exists()
    assert (out_dir / "rank_0.json").exists()
    assert (out_dir / "rank_1.json").exists()
