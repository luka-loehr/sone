#!/usr/bin/env bash
# Copies every fetched song to atlas and prepares it there (stems, analysis), until the fetch is over
# and every fetched song is ready.
F=/Users/luka/Documents/orin-app/orml-fetch
while true; do
  ids=$(python3 -c "import json;print(' '.join(m['uri'].split(':')[-1] for m in json.load(open('$F/songs/meta.json'))))" 2>/dev/null)
  todo=""
  for id in $ids; do
    ssh atlas "test -f ~/djlab/orml-main/data/analysis/beats/$id.json" || todo="$todo $id"
  done
  if [ -n "$todo" ]; then
    for id in $todo; do rsync -a $F/songs/$id.f32 atlas:djlab/tracks/; done
    ssh atlas "cd ~/djlab/orml-main && bash prep_new.sh $todo"
  elif ! pgrep -f "fetch-songs songs" >/dev/null; then
    echo "all ready"; break
  else
    sleep 30
  fi
done
