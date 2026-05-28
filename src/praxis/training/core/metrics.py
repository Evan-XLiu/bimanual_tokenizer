"""Structured metric tracking aligned with LeRobot's training utilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AverageMeter:
    """Running average meter."""

    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


@dataclass
class MetricsTracker:
    """Track training progress counters + rolling meters."""

    batch_size: int
    world_size: int = 1
    steps: int = 0
    samples: int = 0
    meters: dict[str, AverageMeter] = field(default_factory=dict)

    def step(self, samples: int | None = None) -> None:
        self.steps += 1
        if samples is None:
            samples = self.batch_size * self.world_size
        self.samples += int(samples)

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n=n)

    def get(self, name: str) -> float:
        meter = self.meters.get(name)
        return meter.avg if meter is not None else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "steps": float(self.steps),
            "samples": float(self.samples),
            **{name: meter.avg for name, meter in self.meters.items()},
        }
