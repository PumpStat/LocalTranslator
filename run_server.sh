#!/usr/bin/env bash
set -euo pipefail

# Start the local translator API with a GPU profile.
#
# Examples:
#   bash run_server.sh -g 3080
#   bash run_server.sh -g 5090 -p 8000
#   bash run_server.sh -m Qwen/Qwen2.5-14B-Instruct --no-4bit
#   bash run_server.sh --host 0.0.0.0 --port 18000
#
# Env vars respected:
#   MODEL_NAME, USE_4BIT, GPU_PROFILE (are set from flags if provided)

GPU_PROFILE=""
MODEL_NAME=""
USE_4BIT=""
HOST="127.0.0.1"
PORT="8000"
BACKEND="${BACKEND:-}"  # allow pre-set

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      GPU_PROFILE="${2:-}"; shift 2;;
    -m|--model)
      MODEL_NAME="${2:-}"; shift 2;;
    --no-4bit)
      USE_4BIT="false"; shift;;
    --4bit)
      USE_4BIT="true"; shift;;
    -p|--port)
      PORT="${2:-}"; shift 2;;
    --host)
      HOST="${2:-}"; shift 2;;
    -b|--backend)
      BACKEND="${2:-}"; shift 2;;
    -h|--help)
      cat <<USAGE
Usage: bash run_server.sh [options]

Options:
  -g, --gpu <name>        GPU profile preset (e.g., 3080, 5090)
  -m, --model <hf_id>     Explicit model override (takes precedence)
      --no-4bit           Disable 4-bit quantization
      --4bit              Force 4-bit quantization
  -p, --port <num>        Port (default: 8000)
      --host <addr>       Host (default: 127.0.0.1)
  -b, --backend <llm|nllb> Backend (default: llm)
  -h, --help              Show this help

Environment overrides:
  MODEL_NAME, USE_4BIT, GPU_PROFILE
USAGE
      exit 0;;
    *) echo "[run_server] Unknown argument: $1" >&2; exit 1;;
  esac
done

# Export envs for server.py to consume
if [[ -n "$GPU_PROFILE" ]]; then export GPU_PROFILE; fi
if [[ -n "$MODEL_NAME" ]]; then export MODEL_NAME; fi
if [[ -n "$USE_4BIT" ]]; then export USE_4BIT; fi
if [[ -n "$BACKEND" ]]; then export BACKEND; fi

# Prefer local venv's uvicorn if present
UVICORN_BIN="uvicorn"
if [[ -x ".venv/bin/uvicorn" ]]; then
  UVICORN_BIN=".venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
  UVICORN_BIN="$(command -v uvicorn)"
fi

echo "[run_server] Using $UVICORN_BIN"
echo "[run_server] HOST=$HOST PORT=$PORT BACKEND=${BACKEND:-llm} GPU_PROFILE=${GPU_PROFILE:-} MODEL_NAME=${MODEL_NAME:-} USE_4BIT=${USE_4BIT:-}"

exec "$UVICORN_BIN" server:app --host "$HOST" --port "$PORT"
