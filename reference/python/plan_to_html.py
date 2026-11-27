#!/usr/bin/env python3
"""PLAN.md → plan.html for the lab site. plan_to_html.py <PLAN.md> <out.html>…"""
import re, html, sys
src = open(sys.argv[1]).read().split("\n")
def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
out, i = [], 0
while i < len(src):
    ln = src[i]
    if ln.startswith("```"):
        j = i + 1; code = []
        while not src[j].startswith("```"): code.append(src[j]); j += 1
        out.append("<pre>" + html.escape("\n".join(code)) + "</pre>"); i = j + 1; continue
    if ln.startswith("|"):
        rows = []
        while i < len(src) and src[i].startswith("|"):
            rows.append([c.strip() for c in src[i].strip().strip("|").split("|")]); i += 1
        out.append("<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[2:]) + "</tbody></table></div>")
        continue
    m = re.match(r"^(#{1,3}) (.*)", ln)
    if m: out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>"); i += 1; continue
    if re.match(r"^(- |\d+\. )", ln):
        tag = "ul" if ln.startswith("- ") else "ol"; items = []
        while i < len(src) and re.match(r"^(- |\d+\. )", src[i]): items.append(re.sub(r"^(- |\d+\. )", "", src[i])); i += 1
        out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>"); continue
    if not ln.strip(): i += 1; continue
    para = [ln]; i += 1
    while i < len(src) and src[i].strip() and not re.match(r"^(#|\||- |\d+\. |```)", src[i]): para.append(src[i]); i += 1
    out.append("<p>" + inline(" ".join(para)) + "</p>")
page = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Plan: Echtzeit-DJ-Modell</title>
<style>
:root { --bg:#0d0e12; --panel:#16181f; --panel2:#1d2029; --line:#2a2e3a; --text:#e9ebf1; --dim:#8a90a2; --b:#3dd6ff; }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif; }
header { position:sticky; top:0; background:rgba(13,14,18,.92); backdrop-filter:blur(10px); border-bottom:1px solid var(--line); }
.bar { max-width:900px; margin:0 auto; padding:12px 16px; display:flex; gap:14px; align-items:center; } .bar b { font-size:17px; } .bar a { color:var(--b); margin-left:auto; }
main { max-width:900px; margin:0 auto; padding:8px 16px 60px; }
h1 { font-size:26px; margin:28px 0 6px; } h2 { font-size:20px; margin:38px 0 10px; padding-top:18px; border-top:1px solid var(--line); } h3 { font-size:16px; margin:24px 0 8px; }
p, li { color:#d3d6df; } b { color:var(--text); } ul, ol { padding-left:22px; }
code { background:var(--panel2); border:1px solid var(--line); border-radius:4px; padding:0 5px; font:13px ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; overflow-x:auto; font:12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color:#c9ccd6; }
.tw { overflow-x:auto; margin:12px 0; } table { width:100%; border-collapse:collapse; font-size:14px; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
th, td { text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; } th { color:var(--dim); font-weight:500; background:var(--panel2); } tr:last-child td { border-bottom:0; }
</style></head><body><header><div class="bar"><b>Plan</b><span style="color:var(--dim)">Echtzeit-DJ-Modell auf der CPU</span><a href="/">zurück zu den Übergängen</a></div></header><main>""" + "\n".join(out) + "</main></body></html>"
for p in sys.argv[2:]: open(p, "w").write(page)
