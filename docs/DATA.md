# Data

## In this repository

| Path | What | Count |
|---|---|---|
| `data/labels.json` | every note of the owner: verbatim text, time, version, verdict, issue tags, rule | 28 |
| `data/final_verdicts.json` | state of every transition at the end of 2026-09-29 | 13 |
| `data/versions/S1…S7.json` | full spec of every transition in every version shown on the site | 75 specs |
| `data/renders/S*/hNN.report.json` | tempo, stretch, beat lag, level of every stem of both decks per bar | 75 |
| `data/gemini/` | Gemini evaluation notes and rankings on transition variants (audio excluded) | |
| `data/analysis/struct/<id>.json` | per song: beats, downbeats, per bar level of every stem | 40 songs |
| `data/analysis/keys.json` | key, Camelot code, tuning | 40 songs |
| `data/round0/` | first lab: two engines, five styles, star ratings | 26 ratings |
| `data/radio/planned_transitions.json` | what the production radio planned on the days that started this | 45 |
| `data/external/` | DJ Mix Dataset (github.com/mir-aidj/djmix-dataset, djsone-analysis): 5,040 mixes with track lists. Third party, no licence file in the source. | |

Songs are named by their Spotify track id.

## Not in this repository

| What | Why |
|---|---|
| Source recordings (44.1 kHz stereo) | copyright |
| Separated stems (`vocals`, `drums`, `bass`, `other`) | copyright |
| Legacy lab `.m4a` renders | copyright (only Creative Commons `CC-BY` / `CC-BY-SA` demos are distributed) |

Stems per song: `vocals_r.flac` (BS-RoFormer), `drums_i.flac` and `bass_i.flac` (htdemucs_ft on the song
without its voice), and the rest as "other". `teachers/` makes them.

## About the renders

S1 to S6 were rendered again at the end with the final renderer and stems. What the owner heard at the time
had the earlier stems (some voice leaked) and, for Blessings, the bar line one beat late.
