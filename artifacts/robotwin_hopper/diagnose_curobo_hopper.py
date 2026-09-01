#!/usr/bin/env python3
"""GPU-only RoboTwin/cuRobo diagnostic; does not import or run any policy."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path


def distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


def main() -> None:
    robotwin_root = Path(os.environ["ROBOTWIN_ROOT"]).resolve()
    curobo_src = robotwin_root / "envs/curobo/src"
    sys.path.insert(0, str(curobo_src))

    import torch
    import warp
    import curobo
    from curobo.curobolib import lbfgs_step_cu, line_search_cu
    from curobo.opt.newton.lbfgs import LBFGSOpt
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    print("=== hardware/runtime ===", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    nvcc = shutil.which("nvcc")
    print("nvcc_path", nvcc or "NOT_FOUND", flush=True)
    if nvcc:
        subprocess.run([nvcc, "--version"], check=False)
    print("gpu_name", torch.cuda.get_device_name(0), flush=True)
    print("gpu_capability", torch.cuda.get_device_capability(0), flush=True)
    print("torch_version", torch.__version__, flush=True)
    print("torch_cuda_version", torch.version.cuda, flush=True)
    print("torch_path", torch.__file__, flush=True)
    print("curobo_version", distribution_version("nvidia-curobo", "curobo"), flush=True)
    print("curobo_path", curobo.__file__, flush=True)
    print("warp_version", getattr(warp, "__version__", distribution_version("warp-lang", "warp")), flush=True)
    print("warp_path", warp.__file__, flush=True)
    print("lbfgs_step_cu_path", lbfgs_step_cu.__file__, flush=True)
    print("line_search_cu_path", line_search_cu.__file__, flush=True)

    print("=== cuobjdump lbfgs ===", flush=True)
    cuobjdump = shutil.which("cuobjdump")
    print("cuobjdump_path", cuobjdump or "NOT_FOUND", flush=True)
    if cuobjdump:
        subprocess.run([cuobjdump, "--list-elf", lbfgs_step_cu.__file__], check=False)

    yml_path = robotwin_root / "assets/embodiments/aloha-agilex/curobo_left.yml"
    world_config = {
        "cuboid": {
            "table": {
                "dims": [0.7, 2.0, 0.04],
                "pose": [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    print("=== minimal MotionGen warmup ===", flush=True)
    print("robot_config", yml_path, flush=True)
    disable_fused_lbfgs = os.environ.get("CUROBO_DISABLE_FUSED_LBFGS", "0") == "1"
    print("use_cuda_graph", False, flush=True)
    print("disable_fused_lbfgs", disable_fused_lbfgs, flush=True)
    if disable_fused_lbfgs:
        original_lbfgs_init = LBFGSOpt.__init__

        def patched_lbfgs_init(self, config=None):
            if config is not None and hasattr(config, "use_cuda_kernel"):
                config.use_cuda_kernel = False
            original_lbfgs_init(self, config)
            self.use_cuda_kernel = False

        LBFGSOpt.__init__ = patched_lbfgs_init
    config = MotionGenConfig.load_from_robot_config(
        str(yml_path),
        world_config,
        interpolation_dt=1 / 250,
        num_trajopt_seeds=1,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(config)
    print("motion_gen_created", flush=True)
    motion_gen.warmup()
    torch.cuda.synchronize()
    print("MOTION_GEN_WARMUP_OK", flush=True)


if __name__ == "__main__":
    main()
