#!/usr/bin/env python3
"""Serve a saved BiCAT FM policy through the YAM-style HTTP /act protocol.

The checkpoint stays on the GPU host. This server accepts one observation and
returns the full predicted continuous joint-action chunk. It does not perform
robot control or convert between joint and end-effector coordinates.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

MAX_REQUEST_BYTES = 24 * 1024 * 1024
CAMERAS = ("top", "left", "right")
LOG = logging.getLogger("praxis.fm_box_http")


def _decode_numpy_object(value: dict[str, Any]) -> Any:
    """Decode the numeric ndarray representation emitted by json_numpy.dumps."""
    if "__numpy__" not in value:
        return value
    dtype = np.lib.format.descr_to_dtype(value["dtype"])
    if dtype.hasobject or dtype.fields is not None:
        raise ValueError("object and structured NumPy arrays are unsupported")
    shape = tuple(int(dim) for dim in value["shape"])
    if len(shape) > 4 or any(dim < 0 for dim in shape):
        raise ValueError("invalid NumPy array shape")
    raw = base64.b64decode(value["__numpy__"], validate=True)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError("NumPy array byte count does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _encode_numpy_object(value: Any) -> Any:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Cannot encode {type(value).__name__}")
    contiguous = np.ascontiguousarray(value)
    return {
        "__numpy__": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        "dtype": np.lib.format.dtype_to_descr(contiguous.dtype),
        "shape": list(contiguous.shape),
    }


def _image_array(value: Any, *, camera: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{camera} image must have shape (height, width, 3)")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[0] * image.shape[1] > 4_000_000:
        raise ValueError(f"{camera} image has an unsupported size")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError(f"{camera} image must contain numeric pixels")
        image = image.astype(np.float32)
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError(f"{camera} image pixels must be finite and in [0, 255]")
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.rint(image).astype(np.uint8)
    return np.ascontiguousarray(image)


def parse_observation(payload: Any) -> tuple[dict[str, Any], int | None]:
    """Accept the goal-prior camera names and the released YAM camera aliases."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite 14-dimensional joint vector")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    camera_names = (
        {"top": "top", "left": "left", "right": "right"}
        if all(name in payload for name in CAMERAS)
        else {"top": "top_cam", "left": "left_cam", "right": "right_cam"}
    )
    missing = [source for source in camera_names.values() if source not in payload]
    if missing:
        raise ValueError(f"missing camera image(s): {', '.join(missing)}")
    observation: dict[str, Any] = {"state": state, "instruction": instruction}
    for camera, source in camera_names.items():
        observation[camera] = _image_array(payload[source], camera=camera)
    steps = payload.get("num_steps")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
        raise ValueError("num_steps must be a positive integer")
    return observation, steps


class FMBoxRuntime:
    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch

        from praxis.policies.factory import get_policy_class

        self.checkpoint = checkpoint.resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This Molmo2-ER FM checkpoint requires an available CUDA GPU")
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing model.safetensors in {self.checkpoint}")
        policy_class = get_policy_class("fm_molmo")
        self.policy = policy_class.from_pretrained(str(self.checkpoint))
        self.policy.to(self.device).eval()
        self.io = self.policy.io
        if self.io is None:
            raise RuntimeError("Checkpoint is missing PolicyIO")
        if (
            self.io.action_dim != 14
            or self.io.action_feature is None
            or self.io.action_feature.key != "action_joint_angles"
            or len(self.io.state_features) != 1
            or self.io.state_features[0].key != "observation.state_joint_angles"
        ):
            raise RuntimeError("Expected the BiCAT box 14-dimensional joint policy")
        expected_cameras = {f"observation.images.{camera}" for camera in CAMERAS}
        if {feature.key for feature in self.io.image_features} != expected_cameras:
            raise RuntimeError("Checkpoint camera keys differ from top, left, right")
        self.lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint": str(self.checkpoint),
            "camera_keys": list(CAMERAS),
            "state_dim": 14,
            "action_dim": 14,
            "action_space": "joint angles in the checkpoint's action_joint_angles convention",
            "chunk_size": int(self.policy.config.action_horizon),
            "execute_steps": int(self.policy.config.n_action_steps),
            "num_flow_steps": int(self.policy.config.flow_sampling_steps),
        }

    def predict(self, observation: dict[str, Any], *, num_steps: int | None) -> np.ndarray:
        import torch

        from praxis.policies.eval_adapter import resize_policy_io_images

        configured_steps = int(self.policy.config.flow_sampling_steps)
        if num_steps is not None and num_steps != configured_steps:
            raise ValueError(f"This checkpoint serves num_steps={configured_steps}; got {num_steps}")
        batch: dict[str, Any] = {
            "observation.state_joint_angles": torch.from_numpy(observation["state"])
            .unsqueeze(0)
            .to(self.device),
            "task": [observation["instruction"]],
        }
        for camera in CAMERAS:
            image = torch.from_numpy(observation[camera]).permute(2, 0, 1).contiguous()
            batch[f"observation.images.{camera}"] = image.unsqueeze(0).to(self.device)
        batch = resize_policy_io_images(batch, io=self.io)
        with self.lock, torch.inference_mode(), self.policy._autocast_context():
            actions, _ = self.policy.predict_action_chunk(batch)
        result = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected = (1, int(self.policy.config.action_horizon), 14)
        if result.shape != expected or not np.isfinite(result).all():
            raise RuntimeError(f"Model produced invalid action shape or values: {result.shape}")
        return result[0]


def make_handler(runtime: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=_encode_numpy_object, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/act":
                self._respond(200, runtime.info())
            elif self.path == "/healthz":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            if self.path != "/act":
                self._respond(404, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > MAX_REQUEST_BYTES:
                    raise ValueError(f"request size must be between 1 and {MAX_REQUEST_BYTES} bytes")
                payload = json.loads(self.rfile.read(size), object_hook=_decode_numpy_object)
                observation, num_steps = parse_observation(payload)
                start = time.perf_counter()
                actions = runtime.predict(observation, num_steps=num_steps)
                response: dict[str, Any] = {
                    "actions": actions,
                    "dt_ms": round((time.perf_counter() - start) * 1000, 2),
                    "execute_steps": runtime.info()["execute_steps"],
                }
                if "timestamp" in payload:
                    response["timestamp"] = payload["timestamp"]
                self._respond(200, response)
            except (ValueError, TypeError, KeyError) as exc:
                self._respond(400, {"error": str(exc)})
            except Exception:
                LOG.exception("FM inference failed")
                self._respond(500, {"error": "FM inference failed; see server log"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8213)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = FMBoxRuntime(args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    LOG.info("Serving %s on %s:%s", runtime.checkpoint, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
