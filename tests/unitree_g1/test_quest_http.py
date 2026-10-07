"""Loopback HTTP integration; requires permission to bind a local socket."""

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from test_quest_record import frame, quest


class QuestHTTPTests(unittest.TestCase):
    def test_transport_and_single_writer(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            recorder = quest.Recorder(Path(tmp) / "session")
            server = quest.make_server("127.0.0.1", 0, recorder, "test-token")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def request(path, data=None, token="test-token"):
                req = Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                              headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
                with urlopen(req, timeout=3) as response:
                    return json.load(response)

            try:
                with urlopen(base + "/", timeout=3) as response:
                    self.assertIn(b"client.js", response.read())
                with self.assertRaises(HTTPError) as error:
                    request("/api/start", {}, token="wrong")
                self.assertEqual(error.exception.code, 403)
                metadata = {"reference_space": "local-floor", "performance_time_origin_ms": 1}
                episode = request("/api/start", metadata)["episode_id"]
                with self.assertRaises(HTTPError):
                    request("/api/start", metadata)
                self.assertEqual(request("/api/batch", {"episode_id": episode, "records": [frame()]})["ack"], 0)
                self.assertEqual(request("/api/status")["frames"], 1)
                summary = request("/api/end", {"episode_id": episode})
                self.assertEqual(summary["frames"], 1)
                self.assertEqual(summary["reason"], "client_end")
                self.assertIsNone(request("/api/status")["active"])
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=3)
                recorder.finish("test_shutdown")


if __name__ == "__main__":
    unittest.main()
