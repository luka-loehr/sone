#!/usr/bin/env python3
"""Checks of the lab server: other websites and rebound DNS names cannot read or write verdicts, and a
key cannot point the learner at files outside the page's renders. Run: python3 site/test_server.py
(works on a temporary copy of the lab's stores; the repo's data/owner is never touched)."""
import http.client, importlib.util, json, os, tempfile, threading, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lab_server", os.path.join(HERE, "server.py"))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LabServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = self.tmp.name
        self.root = os.path.join(base, "site")
        os.makedirs(os.path.join(self.root, "src", "blind"))
        server.ROOT = self.root
        server.STORE = os.path.join(self.root, "judgments.json")
        server.REPO_VERDICTS = os.path.join(base, "verdicts.json")
        spec_ = [{"a": {"id": "A"}, "b": {"id": "B"}, "recipe": "sneak"}]
        for where in (os.path.join(self.root, "src", "blind", "x01"), os.path.join(base, "outside")):
            json.dump(spec_, open(where + ".spec.json", "w"))
            json.dump({"lag_ms": 1}, open(where + ".report.json", "w"))
        with open(os.path.join(self.root, "index.html"), "w") as f:
            f.write("<p>lab</p>")
        self.httpd = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": f"127.0.0.1:{self.port}"}
        h.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            h["Content-Length"] = str(len(data))
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        out = r.status, r.read()
        conn.close()
        return out

    def verdicts(self):
        try:
            return json.load(open(server.REPO_VERDICTS))
        except FileNotFoundError:
            return None

    def test_lab_page_saves_a_verdict(self):
        origin = {"Origin": f"http://localhost:{self.port}", "Host": f"localhost:{self.port}"}
        status, _ = self.request("POST", "/api/judge", {"key": "blind:x01", "note": "B's voice on the one"}, origin)
        self.assertEqual(status, 200)
        self.assertEqual(self.verdicts()["blind:x01"]["note"], "B's voice on the one")
        status, body = self.request("GET", "/api/judgments", headers=origin)
        self.assertEqual(status, 200)
        self.assertIn("blind:x01", json.loads(body))
        self.assertEqual(self.request("GET", "/index.html")[0], 200)

    def test_client_without_origin_still_works(self):
        status, _ = self.request("POST", "/api/judge", {"key": "blind:x01", "stars": 4})
        self.assertEqual(status, 200)

    def test_other_website_cannot_write(self):
        for origin in ("https://evil.example", "null", f"http://127.0.0.1.evil.example:{self.port}"):
            status, _ = self.request("POST", "/api/judge", {"key": "blind:x01", "note": "fake"}, {"Origin": origin})
            self.assertEqual(status, 403, origin)
        self.assertFalse(os.path.exists(server.STORE))
        self.assertIsNone(self.verdicts())

    def test_rebound_dns_name_cannot_read_or_write(self):
        host = {"Host": f"evil.example:{self.port}"}
        self.assertEqual(self.request("GET", "/api/judgments", headers=host)[0], 403)
        self.assertEqual(self.request("GET", "/index.html", headers=host)[0], 403)
        self.assertEqual(self.request("HEAD", "/index.html", headers=host)[0], 403)
        self.assertEqual(self.request("POST", "/api/judge", {"key": "blind:x01", "note": "fake"}, host)[0], 403)
        self.assertIsNone(self.verdicts())

    def test_key_cannot_leave_the_page_renders(self):
        status, _ = self.request("POST", "/api/judge", {"key": "blind:../../../outside", "note": "probe"})
        self.assertEqual(status, 200)  # stored as a judgment like any other key, but never learned from
        self.assertIsNone(self.verdicts())

    def test_bad_bodies_are_refused(self):
        self.assertEqual(self.request("POST", "/api/judge", ["blind:x01"])[0], 400)
        self.assertEqual(self.request("POST", "/api/judge", {"key": 5})[0], 400)


if __name__ == "__main__":
    unittest.main()
