#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
LLAMA_BIN="$AI_ROOT/llama.cpp/build/bin/llama-server"
MODEL_PATH="$AI_ROOT/models/qwen3-1.7b.Q8_0.gguf"

LLAMA_CTX_SIZE="128"
LLAMA_THREADS="8"
LLAMA_PORT="8080"

BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found or not executable: $LLAMA_BIN"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH"
  exit 1
fi

cleanup() {
  if [[ -n "${LLAMA_PID:-}" ]]; then
    kill "$LLAMA_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting llama-server on port $LLAMA_PORT..."
"$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --ctx-size "$LLAMA_CTX_SIZE" \
  --threads "$LLAMA_THREADS" \
  --port "$LLAMA_PORT" &
LLAMA_PID=$!

echo "Starting FastAPI backend on port $BACKEND_PORT..."
cd "$ROOT_DIR/backend"
conda run -n thesis python src/server.py &
BACKEND_PID=$!

wait "$LLAMA_PID" "$BACKEND_PID"