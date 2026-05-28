"""gRPC serving entry point."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from praxis.core.devices import require_available_device

CONFIG_PATH = str(Path(__file__).parents[1] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="serve")
def main(cfg: DictConfig) -> None:
    """Start the gRPC policy server.

    Policy is loaded self-contained from checkpoint (weights + PolicyIO + norm stats).
    No dataset download is required.
    """
    print(OmegaConf.to_yaml(cfg))

    from praxis.policies.factory import make_policy
    from praxis.serving.server import PolicyServer

    device = require_available_device(cfg.device, purpose="Serving")

    # Build policy via factory — self-contained from checkpoint.
    policy = make_policy(cfg.policy)

    # Start server
    server = PolicyServer(
        policy=policy,
        device=str(device),
        host=cfg.host,
        port=cfg.port,
    )
    server.serve(block=True)


if __name__ == "__main__":
    main()
