#!/usr/bin/env bash
# Poll the Desktop platform builds run without draining the battery:
# one cheap HTTPS call per check, exits when the run finishes.
set -u
TOKEN=$(printf "protocol=https\nhost=github.com\n\n" | git credential fill 2>/dev/null | grep "^password=" | cut -d= -f2)
RUN_ID=${1:-35363562195}
BASE="https://api.github.com/repos/RileyK05/NLP-Suite/actions"

while true; do
  STATUS=$(curl -s -H "Authorization: token $TOKEN" \
    "$BASE/runs/$RUN_ID" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('status'), d.get('conclusion'))")
  set -- $STATUS
  STATE=$1; CONCLUSION=${2:-}
  echo "$(date +%H:%M) run $RUN_ID: $STATE ${CONCLUSION}"
  case "$STATE" in
    completed)
      echo "== finished: $CONCLUSION =="
      curl -s -H "Authorization: token $TOKEN" "$BASE/runs/$RUN_ID/jobs?per_page=10" | \
        python -c "
import json, sys
for j in json.load(sys.stdin).get('jobs', []):
    print(f\"{j['name'][:40]:42s} {j['status']:12s} {j.get('conclusion') or '-'}\")"
      if [ "$CONCLUSION" = "success" ]; then
        echo
        echo "Download the Windows installer artifact:"
        curl -s -H "Authorization: token $TOKEN" "$BASE/runs/$RUN_ID/artifacts" | \
          python -c "
import json, sys
for a in json.load(sys.stdin).get('artifacts', []):
    print(f\"  {a['name']}  ({a['size_in_bytes']//1_000_000} MB)  id={a['id']}\")"
      fi
      exit 0
      ;;
  esac
  sleep 120
done