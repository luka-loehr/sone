# Data

## In this repository

| Path | What | Count |
|---|---|---|
| `data/labels.json` | every note of the owner: verbatim text, time, version, verdict, issue tags, rule | 28 |
| `data/final_verdicts.json` | state of every transition at the end of 2026-09-29 | 13 |
| `data/versions/S1…S7.json` | full spec of every transition in every version shown on the site | 75 specs |
| `data/renders/S*/hNN.m4a` | every version rendered, about 60 s around the transition | 75 clips |
| `data/renders/S*/hNN.report.json` | tempo, stretch, beat lag, level of every stem of both decks per bar | 75 |
| `data/gemini/` | nine versions of h10 (87 → 130 BPM) and Gemini's rankings | 9 clips |
| `data/analysis/struct/<id>.json` | per song: beats, downbeats, per bar level of every stem | 40 songs |
| `data/analysis/keys.json` | key, Camelot code, tuning | 40 songs |
| `data/round0/` | first lab: two engines, five styles, star ratings | 26 ratings |
| `data/radio/planned_transitions.json` | what the production radio planned on the days that started this | 45 |
| `data/external/` | DJ Mix Dataset (github.com/mir-aidj/djmix-dataset, djsone-analysis): 5,040 mixes with track lists. Third party, no licence file in the source. | |

Songs are named by their Spotify track id.

## Not in this repository

| What | Where | Why |
|---|---|---|
| The 40 songs (44.1 kHz float) | atlas `~/djlab/tracks` | copyright, 2.8 GB |
| Their stems | atlas `~/djlab/stems/<id>/` | copyright, 6.4 GB |

Stems per song: `vocals_r.flac` (BS-RoFormer), `drums_i.flac` and `bass_i.flac` (htdemucs_ft on the song
without its voice), and the rest as "other". `teachers/` makes them.

## About the renders

S1 to S6 were rendered again at the end with the final renderer and stems. What the owner heard at the time
had the earlier stems (some voice leaked) and, for Blessings, the bar line one beat late.
