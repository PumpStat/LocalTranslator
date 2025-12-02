#!/usr/bin/env bash
set -euo pipefail

# Call the local translator API and print the translated text.
#
# Examples:
#   bash translate.sh -src en -dst ko -t "Hello world"
#   bash translate.sh -s en -d es -t "Stamina with lots of runs" -u http://127.0.0.1:8000
#
# Short and long flags are both supported for src/dst/text.

SRC="ko"
DST="en"
TEXT=""
URL="http://127.0.0.1:8000"
USE_DEFAULT_EXAMPLES="true"

err() { echo "$*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -src|--src|-s)
      SRC="${2:-}"; shift 2;;
    -dst|--dst|-d)
      DST="${2:-}"; shift 2;;
    -t|--text)
      TEXT="${2:-}"; shift 2;;
    -u|--url)
      URL="${2:-}"; shift 2;;
    --no-default-examples)
      USE_DEFAULT_EXAMPLES="false"; shift;;
    -h|--help)
      cat <<USAGE
Usage: bash translate.sh -src <en|ko|es> -dst <en|ko|es> -t "text" [options]

Options:
  -src, --src, -s        Source language (en|ko|es)
  -dst, --dst, -d        Target language (en|ko|es)
  -t,   --text           Text to translate
  -u,   --url            API base URL (default: http://127.0.0.1:8000)
       --no-default-examples  Do not include default few-shot examples
  -h,   --help           Show this help
USAGE
      exit 0;;
    *) err "[translate] Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$SRC" || -z "$DST" || -z "$TEXT" ]]; then
  err "[translate] Missing required args. See -h for usage."; exit 1
fi

export SRC DST TEXT USE_DEFAULT_EXAMPLES

# Heuristic language check to catch common src/dst mistakes
python3 - <<'PY' 2>/dev/null || true
import os, sys
t = os.environ.get('TEXT','')
src = os.environ.get('SRC','')
def detect_lang(s:str):
    hangul = sum(0x3130 <= ord(ch) <= 0x318F or 0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
    latin = sum(('a' <= ch.lower() <= 'z') for ch in s)
    es_marks = sum(ch in 'áéíóúñ¡¿ÁÉÍÓÚÑ' for ch in s)
    if hangul > 10 and hangul > latin:
        return 'ko'
    if es_marks >= 1:
        return 'es'
    if latin > 5:
        return 'en'
    return ''
det = detect_lang(t)
if det and src and det != src:
    sys.stderr.write(f"[translate] Warning: text looks like '{det}' but --src='{src}'.\n")
    sys.stderr.write("             If you intended Korean→English, use: -s ko -d en\n")
PY

BODY=$(python3 - <<PY
import json, os
print(json.dumps({
  "text": os.environ.get("TEXT"),
  "source": os.environ.get("SRC"),
  "target": os.environ.get("DST"),
  "use_default_examples": os.environ.get("USE_DEFAULT_EXAMPLES") == "true"
}, ensure_ascii=False))
PY
)

RESP=$(curl -sS -X POST "$URL/translate" \
  -H 'Content-Type: application/json' \
  --data "$BODY" || true)

# Try to extract the translated field; fallback to raw response
if command -v jq >/dev/null 2>&1; then
  echo "$RESP" | jq -r '.translated // .'
else
  PYBIN=$(command -v python3 || command -v python || true)
  if [[ -n "$PYBIN" ]]; then
    echo "$RESP" | "$PYBIN" - <<'PY'
import sys, json
raw = sys.stdin.read()
try:
    obj = json.loads(raw)
    print(obj.get("translated", raw))
except Exception:
    print(raw)
PY
  else
    # No jq or python: print raw JSON
    echo "$RESP"
  fi
fi
