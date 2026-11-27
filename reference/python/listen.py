#!/usr/bin/env python3
"""Gemini 3.8 Flash as a second pair of ears for one hand-built transition.
listen.py <clip> <start_s> <one_s> <end_s> [runs=3] → JSON per run and a summary.
It is asked for concrete defects with time stamps, not for taste."""
import base64, json, os, subprocess, sys, tempfile, time, urllib.request

MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
KEY = os.environ["GEMINI_API_KEY"]
PROMPT = """You are an experienced club and radio DJ and mastering engineer. This clip is ONE transition between two songs, built the way a DJ with stem separation would do it.
Timeline of the clip: song A plays alone until {start:.0f} s. From {start:.0f} s parts of song B come in underneath. At {one:.0f} s the bass and kick change from A to B. By {end:.0f} s song A is gone and song B plays alone.
Listen to the whole clip, above all {start:.0f}-{end:.0f} s, and answer precisely and critically:
1. kicks_flam: do you hear double or galloping kicks/snares (two beats slightly apart) while both songs overlap? (true/false, and at which seconds)
2. two_vocals: are two different lead vocals audible at the same time anywhere? (true/false, seconds)
3. key_clash: do melodies/chords/bass of the two songs sound out of tune against each other? (true/false, seconds)
4. level_problem: any audible volume jump, dip, hole or pumping? (true/false, seconds)
5. abrupt: does anything start or stop abruptly in a way that sounds like a mistake (not like a deliberate DJ cut on the beat)? (true/false, seconds)
6. artifacts: any watery, phasey, thin or metallic sound (stem separation or time-stretch artifacts)? (true/false, seconds)
7. energy: does the energy hold through the transition (1 = collapses, 10 = holds or lifts)?
8. b_entry: does song B arrive at a musically sensible place and on the phrase (1-10)?
9. pro: how close is this to a transition by a top professional DJ (1-10)?
10. what_happens: describe in two sentences what you actually hear between {start:.0f} s and {end:.0f} s.
11. fix: the single most useful change to make it better, concrete.
Answer ONLY JSON: {{"kicks_flam":{{"yes":bool,"at":"..."}},"two_vocals":{{"yes":bool,"at":"..."}},"key_clash":{{"yes":bool,"at":"..."}},"level_problem":{{"yes":bool,"at":"..."}},"abrupt":{{"yes":bool,"at":"..."}},"artifacts":{{"yes":bool,"at":"..."}},"energy":n,"b_entry":n,"pro":n,"what_happens":"...","fix":"..."}}"""

def to_mp3(path):
    out = tempfile.mktemp(suffix=".mp3")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-ac", "2", "-ar", "44100", "-b:a", "128k", out], check=True)
    return out

def ask(mp3, start, one, end):
    body = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(open(mp3, "rb").read()).decode()}},
            {"text": PROMPT.format(start=start, one=one, end=end)},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                ans = json.load(r)
            return json.loads(ans["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            err = e
            time.sleep(2 + attempt * 3)
    raise err

def listen(path, start, one, end, runs=3):
    mp3 = to_mp3(path)
    try:
        res = [ask(mp3, start, one, end) for _ in range(runs)]
    finally:
        os.remove(mp3)
    flags = ["kicks_flam", "two_vocals", "key_clash", "level_problem", "abrupt", "artifacts"]
    out = {f: sum(1 for r in res if (r.get(f) or {}).get("yes")) for f in flags}
    for k in ("energy", "b_entry", "pro"):
        out[k] = round(sum(float(r.get(k, 0)) for r in res) / len(res), 1)
    out["notes"] = [{f: (r.get(f) or {}).get("at") for f in flags if (r.get(f) or {}).get("yes")} for r in res]
    out["what"] = [r.get("what_happens") for r in res]
    out["fix"] = [r.get("fix") for r in res]
    return out

if __name__ == "__main__":
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    print(json.dumps(listen(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), runs), indent=1, ensure_ascii=False))
