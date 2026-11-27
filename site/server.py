#!/usr/bin/env python3
"""DJ transition lab: serves the lab and stores judgments in judgments.json."""
import http.server, json, os, re, threading, time
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.abspath(__file__))
# A transition id on a system page (x25, b12, p01, intro): never a path.
TRANSITION_ID = re.compile(r"[A-Za-z0-9_-]+")

def is_local(netloc):
    """True when a Host header or an Origin's host[:port] names this machine."""
    try:
        return urlsplit("//" + netloc).hostname in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False
STORE = os.path.join(ROOT, "judgments.json")
LOCK = threading.Lock()

def load():
    try:
        return json.load(open(STORE))
    except Exception:
        return {}

# Every verdict on a transition the system made goes into the repo as training data, with the plan
# and the measurements it was made from: data/owner/verdicts.json, one entry per transition, every
# distinct note kept.
REPO_VERDICTS = os.path.expanduser("~/Documents/orin-app/orin-radio-ml/data/owner/verdicts.json")
SYSTEM_PAGES = {"auto", "new", "blind", "blindnew"}

def learn(key, entry):
    page, _, tid = key.partition(":")
    if page not in SYSTEM_PAGES or not TRANSITION_ID.fullmatch(tid) or not (entry.get("note") or entry.get("stars")):
        return
    src = os.path.join(ROOT, "src", page)
    try:
        spec_path = os.path.join(src, tid + ".spec.json")
        spec = json.load(open(spec_path))[0]
        report = json.load(open(os.path.join(src, tid + ".report.json")))
        # A note written before this render belongs to an earlier one (archived in verdicts_<round>.json).
        if time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(spec_path))) > entry["at"]:
            return
    except FileNotFoundError:
        return
    try:
        all_ = json.load(open(REPO_VERDICTS))
    except Exception:
        all_ = {}
    v = all_.get(key, {"page": page, "id": tid, "notes": []})
    v.update({"at": entry["at"], "stars": entry.get("stars"), "tags": entry.get("tags"), "note": entry.get("note", "")})
    if spec:
        v.update({"a": spec["a"]["id"], "b": spec["b"]["id"], "recipe": spec.get("recipe"), "spec": spec, "report": report})
    note = entry.get("note", "")
    # The page saves while the owner types: keep a note when it is not just the start of the next one.
    v["notes"] = [n for n in v["notes"] if not note.startswith(n["note"])] + ([{"at": entry["at"], "note": note}] if note else [])
    all_[key] = v
    tmp = REPO_VERDICTS + ".tmp"
    json.dump(all_, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, REPO_VERDICTS)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _foreign(self):
        """A request another website sent through the owner's browser, or one for a rebound DNS name:
        its verdicts would land in the training data, so it gets nothing."""
        if not is_local(self.headers.get("Host", "")):
            return True
        origin = self.headers.get("Origin")
        return origin is not None and not (origin.startswith(("http://", "https://")) and is_local(urlsplit(origin).netloc))

    def do_GET(self):
        if self._foreign():
            return self._json({"error": "forbidden"}, 403)
        if self.path == "/api/judgments":
            return self._json(load())
        return super().do_GET()

    def do_HEAD(self):
        if self._foreign():
            return self._json({"error": "forbidden"}, 403)
        return super().do_HEAD()

    def do_POST(self):
        if self._foreign():
            return self._json({"error": "forbidden"}, 403)
        if self.path != "/api/judge":
            return self._json({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
        key = body.get("key") if isinstance(body, dict) else None
        if not isinstance(key, str) or not key:
            return self._json({"error": "key"}, 400)
        with LOCK:
            data = load()
            entry = data.get(key, {})
            entry.update({k: v for k, v in body.items() if k != "key"})
            entry["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            data[key] = entry
            tmp = STORE + ".tmp"
            json.dump(data, open(tmp, "w"), indent=1)
            os.replace(tmp, STORE)
            learn(key, entry)
        return self._json({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print(f"DJ lab on http://localhost:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
