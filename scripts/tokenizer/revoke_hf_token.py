#!/usr/bin/env python3
"""Revoke a Hugging Face token through the public credential revocation API."""

from __future__ import annotations

import getpass

import requests


def main() -> None:
    token = getpass.getpass("Old token to revoke (input hidden): ").strip()
    if not token.startswith("hf_") or any(char.isspace() for char in token):
        raise SystemExit("Invalid token format. No request was sent.")

    try:
        response = requests.post(
            "https://huggingface.co/api/credentials/revoke",
            json={"credentials": [token]},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"Revocation request failed: {type(exc).__name__}") from None
    if not response.ok:
        raise SystemExit(f"Revocation request returned HTTP {response.status_code}.")

    try:
        check = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(
            f"Revocation request accepted; verification failed: {type(exc).__name__}"
        ) from None
    if check.status_code in (401, 403):
        print("Old token is invalid. Revocation verified.")
    else:
        print(f"Revocation request accepted; verification returned HTTP {check.status_code}.")


if __name__ == "__main__":
    main()
