#!/usr/bin/env python3
"""Quest WebXR input recorder. Python stdlib only; never imports robot SDKs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid


WEB = Path(__file__).with_name("quest_web")
SCHEMA = "quest_input_v1"


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_pose(pose):
    if pose is None:
        return
    if not isinstance(pose, dict):
        raise ValueError("pose must be null or an object")
    for key, size in (("position", 3), ("orientation_xyzw", 4)):
        values = pose.get(key)
        if not isinstance(values, list) or len(values) != size or not all(map(number, values)):
            raise ValueError(f"invalid pose {key}")
    norm = sum(v * v for v in pose["orientation_xyzw"])
    if abs(norm - 1) > 0.02 or type(pose.get("emulated_position")) is not bool:
        raise ValueError("invalid quaternion or emulated_position")


def validate_record(row):
    if not isinstance(row, dict) or type(row.get("seq")) is not int or row["seq"] < 0:
        raise ValueError("invalid sequence")
    if not number(row.get("client_time_ms")) or row["client_time_ms"] < 0:
        raise ValueError("invalid client timestamp")
    if row.get("type") == "event":
        if not isinstance(row.get("name"), str) or len(row["name"]) > 100:
            raise ValueError("invalid event")
        return
    if row.get("type") != "frame" or not number(row.get("xr_frame_time_ms")):
        raise ValueError("invalid frame")
    if type(row.get("reference_epoch")) is not int or row["reference_epoch"] < 0:
        raise ValueError("invalid reference epoch")
    if row.get("visibility") not in ("visible", "visible-blurred", "hidden"):
        raise ValueError("invalid visibility")
    validate_pose(row.get("head"))
    for side in ("left", "right"):
        controller = row.get(side)
        if controller is None:
            continue
        if not isinstance(controller, dict):
            raise ValueError("invalid controller")
        validate_pose(controller.get("grip"))
        validate_pose(controller.get("target_ray"))
        buttons, axes = controller.get("buttons"), controller.get("axes")
        if not isinstance(buttons, list) or len(buttons) > 20:
            raise ValueError("invalid buttons")
        for button in buttons:
            if (not isinstance(button, dict) or not number(button.get("value"))
                    or not 0 <= button["value"] <= 1
                    or type(button.get("pressed")) is not bool
                    or type(button.get("touched")) is not bool):
                raise ValueError("invalid button")
        if not isinstance(axes, list) or len(axes) > 16 or not all(number(a) and -1 <= a <= 1 for a in axes):
            raise ValueError("invalid axes")


class Recorder:
    def __init__(self, output, stale_s=2.0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.stale_s = stale_s
        self.lock = threading.RLock()
        self.active = None
        self.last_summary = None
        self.latest = None

    def start(self, metadata, now=None):
        now = time.monotonic() if now is None else now
        if not isinstance(metadata, dict) or metadata.get("reference_space") != "local-floor":
            raise ValueError("local-floor reference space required")
        if not number(metadata.get("performance_time_origin_ms")):
            raise ValueError("browser clock origin required")
        with self.lock:
            if self.active:
                raise ValueError("another episode is active; end it or wait for timeout")
            episode = uuid.uuid4().hex
            directory = self.output / episode
            directory.mkdir()
            meta = {"schema": SCHEMA, "episode_id": episode, "session_id": self.output.name,
                    "kind": "quest_input_only", "robot_commands": False,
                    "host_start_monotonic_s": now, "host_start_unix_s": time.time(),
                    "coordinate_system": "WebXR right-handed: +X right, +Y up, -Z forward; metres; quaternion xyzw",
                    "time_semantics": "XR frame time is browser predicted display time, NOT device sensor time; host receive time is per batch; clocks are NOT synchronized",
                    "client": metadata}
            (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            self.active = {"id": episode, "file": (directory / "raw.jsonl").open("x"),
                           "directory": directory, "last_receive": now, "start": now,
                           "last_seq": -1, "last_client": -1, "last_xr": None,
                           "frames": 0, "valid_both": 0, "events": 0,
                           "frame_gaps_ms": [], "max_receive_gap_s": 0.0}
            self.latest = None
            print(f"Episode started: {episode}", flush=True)
            return episode

    def batch(self, episode, records, now=None):
        now = time.monotonic() if now is None else now
        if not isinstance(records, list) or not 1 <= len(records) <= 512:
            raise ValueError("batch must contain 1..512 records")
        with self.lock:
            state = self.active
            if not state or episode != state["id"]:
                raise ValueError("episode is not active; start a new XR session")
            if now - state["last_receive"] > self.stale_s:
                self.finish("stream_timeout", now)
                raise ValueError("stream timed out; start a new XR session")
            seq, client, xr = state["last_seq"], state["last_client"], state["last_xr"]
            # Validate the complete batch before writing any of it.
            for row in records:
                validate_record(row)
                if row["seq"] != seq + 1 or row["client_time_ms"] < client:
                    raise ValueError("sequence gap/duplicate or browser clock reversal")
                seq, client = row["seq"], row["client_time_ms"]
                if row["type"] == "frame":
                    if xr is not None and row["xr_frame_time_ms"] <= xr:
                        raise ValueError("XR frame clock did not advance")
                    xr = row["xr_frame_time_ms"]
            serialized = [json.dumps({"episode_id": episode, "host_receive_monotonic_s": now,
                                      "input": row}, allow_nan=False) + "\n" for row in records]
            for row, line in zip(records, serialized):
                state["file"].write(line)
                if row["type"] == "frame":
                    state["frames"] += 1
                    valid = (row.get("head") is not None and not row["head"]["emulated_position"]
                             and row["visibility"] == "visible")
                    for side in ("left", "right"):
                        ctrl = row.get(side)
                        valid = valid and bool(ctrl and ctrl.get("grip") and not ctrl["grip"]["emulated_position"])
                    state["valid_both"] += int(bool(valid))
                    if state["last_xr"] is not None:
                        state["frame_gaps_ms"].append(row["xr_frame_time_ms"] - state["last_xr"])
                    state["last_xr"] = row["xr_frame_time_ms"]
                    self.latest = row
                else:
                    state["events"] += 1
            state["file"].flush()
            state["max_receive_gap_s"] = max(state["max_receive_gap_s"], now - state["last_receive"])
            state.update(last_receive=now, last_seq=seq, last_client=client)
            return seq

    def finish(self, reason, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            state = self.active
            if not state:
                return self.last_summary
            gaps = sorted(state["frame_gaps_ms"])
            summary = {"episode_id": state["id"], "reason": reason,
                       "frames": state["frames"], "valid_both_frames": state["valid_both"],
                       "events": state["events"], "duration_host_s": now - state["start"],
                       "max_batch_receive_gap_s": state["max_receive_gap_s"],
                       "frame_gap_p95_ms": gaps[math.ceil(len(gaps) * .95) - 1] if gaps else None,
                       "frame_gap_max_ms": max(gaps) if gaps else None,
                       "input_only": True}
            state["file"].write(json.dumps({"type": "host_episode_end", "host_monotonic_s": now,
                                           "summary": summary}) + "\n")
            state["file"].flush()
            os.fsync(state["file"].fileno())
            state["file"].close()
            (state["directory"] / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.active = None
            self.last_summary = summary
            self.latest = None
            print(json.dumps(summary), flush=True)
            return summary

    def status(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            state = self.active
            if state and now - state["last_receive"] > self.stale_s:
                self.finish("stream_timeout", now)
                state = None
            return {"active": state["id"] if state else None,
                    "frames": state["frames"] if state else 0,
                    "receive_age_s": now - state["last_receive"] if state else None,
                    "latest": self.latest, "last_summary": self.last_summary}


def make_server(bind, port, recorder, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def respond(self, code, data, content_type="application/json"):
            payload = json.dumps(data).encode() if content_type == "application/json" else data
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            return secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/client.js": ("client.js", "text/javascript; charset=utf-8")}
            if path in assets:
                name, mime = assets[path]
                self.respond(200, (WEB / name).read_bytes(), mime)
            elif path == "/api/status" and self.authorized():
                self.respond(200, recorder.status())
            else:
                self.respond(404, {"error": "not found or unauthorized"})

        def do_POST(self):
            if not self.authorized():
                self.respond(403, {"error": "invalid token"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("object required")
                with recorder.lock:
                    if self.path == "/api/start":
                        result = {"episode_id": recorder.start(data)}
                    elif self.path == "/api/batch":
                        result = {"ack": recorder.batch(data.get("episode_id"), data.get("records"))}
                    elif self.path == "/api/end":
                        if not recorder.active or data.get("episode_id") != recorder.active["id"]:
                            raise ValueError("episode is not active")
                        result = recorder.finish("client_end")
                    else:
                        self.respond(404, {"error": "unknown endpoint"})
                        return
                self.respond(200, result)
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})

    class Server(ThreadingHTTPServer):
        daemon_threads = False

        def service_actions(self):
            recorder.status()

    return Server((bind, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new external session directory")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--stale-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.stale_seconds) or args.stale_seconds <= 0:
        parser.error("--stale-seconds must be positive and finite")
    if bool(args.cert) != bool(args.key):
        parser.error("--cert and --key must be supplied together")
    if args.bind not in ("127.0.0.1", "localhost") and not args.cert:
        parser.error("LAN binding requires HTTPS --cert and --key")
    context = None
    if args.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
    recorder = Recorder(args.output, args.stale_seconds)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WEB, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    sources = [Path(__file__), WEB / "index.html", WEB / "client.js"]
    (args.output / "session.json").write_text(json.dumps({"schema": SCHEMA, "git_revision": revision,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "note": "Working tree may contain uncommitted changes; archive recorder sources with experiment.",
        "stale_seconds": args.stale_seconds}, indent=2) + "\n")
    token = secrets.token_urlsafe(12)
    server = make_server(args.bind, args.port, recorder, token)
    if context:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    scheme = "https" if context else "http"
    print(f"Quest input only. Open {scheme}://{args.bind}:{server.server_port}/#token={token}", flush=True)
    print(f"Output: {args.output.resolve()} | Ctrl-C to finish. For 0.0.0.0 use this PC's LAN IP.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        recorder.finish("server_shutdown")


if __name__ == "__main__":
    main()
