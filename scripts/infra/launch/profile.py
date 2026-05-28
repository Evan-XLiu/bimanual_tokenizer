from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SlurmClusterProfile:
    validate_partition_list: Callable[[list[str]], list[str]]
    resolve_partition_spec: Callable[[str, str, list[str]], str]
    resolve_account_for_partition_spec: Callable[[str], str]
    is_requeue_partition_spec: Callable[[str], bool]
    mail_type: str | None = None
    mail_user: str | None = None
