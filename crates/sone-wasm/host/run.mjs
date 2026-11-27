// Plans and renders one transition inside the WebAssembly module, the way a Worker would:
// only the windows of the stems the plan asks for are handed over.
//
//   node run.mjs <sone_wasm.wasm> <beats dir> <raw stems dir> <A> <B> <out.wav> [options json]
//
// Prints what was planned, how long planning and rendering took and how much memory the
// module had taken at the end (WebAssembly memory only grows, so that is its peak).
import fs from "node:fs";
import path from "node:path";

const [wasmPath, beatsDir, rawDir, a, b, outPath, optionsText] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), {});
const x = instance.exports;
const MB = 1024 * 1024;
const mem = () => x.memory.buffer.byteLength / MB;
const put = (bytes) => {
  const p = x.sone_alloc(bytes.length);
  new Uint8Array(x.memory.buffer, p, bytes.length).set(bytes);
  return p;
};
const result = () => new TextDecoder().decode(new Uint8Array(x.memory.buffer, x.sone_result_ptr(), x.sone_result_len()).slice());
const check = (code) => {
  if (code !== 0) throw new Error(result());
  return JSON.parse(result());
};

const base = mem();
const song = (id) => {
  const table = fs.readFileSync(path.join(beatsDir, `${id}.json`));
  const meta = JSON.parse(fs.readFileSync(path.join(rawDir, id, "meta.json"), "utf8"));
  const p = put(table);
  const h = x.sone_song(p, table.length, meta.lufs, meta.frames);
  x.sone_free(p, table.length);
  if (h === 0) throw new Error(result());
  return { id, h, meta };
};
const A = song(a), B = song(b);
const tablesMem = mem();

let t0 = performance.now();
const options = new TextEncoder().encode(optionsText ?? "{}");
const po = put(options);
const plan = check(x.sone_plan(A.h, B.h, po, options.length));
x.sone_free(po, options.length);
const planMs = performance.now() - t0;

const STEMS = ["vocals", "drums", "bass", "other"];
let given = 0;
for (const [s, needs] of [[A, plan.needs.a], [B, plan.needs.b]]) {
  for (const need of needs) {
    const start = Math.max(0, Math.floor(need.from_s * 44100));
    const frames = Math.min(s.meta.frames, Math.ceil(need.to_s * 44100)) - start;
    if (frames <= 0) continue;
    for (const [k, name] of STEMS.entries()) {
      if (need.stems === "drums" && name !== "drums") continue;
      const fd = fs.openSync(path.join(rawDir, s.id, `${name}.s16`), "r");
      const buf = Buffer.alloc(frames * 4);
      fs.readSync(fd, buf, 0, buf.length, start * 4);
      fs.closeSync(fd);
      const p = put(buf);
      if (x.sone_stems(s.h, k, BigInt(start), p, frames) !== 0) throw new Error("sone_stems");
      x.sone_free(p, buf.length);
      given += buf.length;
    }
  }
}
const stemsMem = mem();

t0 = performance.now();
const placed = check(x.sone_render());
const renderMs = performance.now() - t0;
const renderMem = mem();

const pcm = new Int16Array(x.memory.buffer, x.sone_audio_ptr(), x.sone_audio_len()).slice();
const header = Buffer.alloc(44);
header.write("RIFF", 0); header.writeUInt32LE(36 + pcm.byteLength, 4); header.write("WAVEfmt ", 8);
header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(2, 22);
header.writeUInt32LE(44100, 24); header.writeUInt32LE(44100 * 4, 28); header.writeUInt16LE(4, 32); header.writeUInt16LE(16, 34);
header.write("data", 36); header.writeUInt32LE(pcm.byteLength, 40);
fs.writeFileSync(outPath, Buffer.concat([header, Buffer.from(pcm.buffer)]));
fs.writeFileSync(outPath.replace(/\.wav$/, ".json"), JSON.stringify({ plan, placed }, null, 1));

const seconds = placed.frames / 44100;
console.log(JSON.stringify({
  pair: `${a} -> ${b}`, recipe: plan.recipe, one_s: +plan.one_s.toFixed(2), stretch_pct: +plan.stretch_pct.toFixed(2),
  clip_seconds: +seconds.toFixed(1), beats_apart_ms: placed.report.align?.all?.[0] ?? null,
  plan_ms: +planMs.toFixed(1), render_ms: +renderMs.toFixed(0), render_per_audio_second_ms: +(renderMs / seconds).toFixed(1),
  needs: plan.needs,
  memory_mb: { module_alone: +base.toFixed(1), with_both_tables: +tablesMem.toFixed(1), with_stems: +stemsMem.toFixed(1), after_rendering_peak: +renderMem.toFixed(1), stems_given: +(given / MB).toFixed(1) },
}, null, 1));
