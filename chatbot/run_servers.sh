#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
LLAMA_BIN="$AI_ROOT/llama-server.exe"
MODEL_PATH="$ROOT_DIR/backend/models/qwen3-1.7b.Q8_0.gguf"

LLAMA_CTX_SIZE="128"
LLAMA_THREADS="8"
LLAMA_PORT="8080"

BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"

SHUTDOWN_TIMEOUT=10          # thời gian chờ graceful shutdown (giây)
SHUTDOWN_WARN_INTERVAL=2     # interval kill dần

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found or not executable: $LLAMA_BIN"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH"
  exit 1
fi

# ── Hàm kill process trên 1 port ──
kill_port() {
  local port="$1"
  local signal="$2"   # TERM hoặc KILL
  if command -v fuser &>/dev/null; then
    fuser -k -"$signal" "${port}/tcp" 2>/dev/null || true
  elif command -v lsof &>/dev/null; then
    local pids
    pids="$(lsof -ti :"$port" 2>/dev/null)" || true
    if [[ -n "$pids" ]]; then
      while IFS= read -r pid; do
        kill -"$signal" "$pid" 2>/dev/null || true
      done <<< "$pids"
    fi
  fi
}

# ── Kill process group (PID + children) ──
kill_pg() {
  local pid="$1"
  local signal="$2"
  # Gửi signal cho cả process group (dấu -)
  kill -"$signal" -- "-$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null || true
  kill -"$signal" "$pid" 2>/dev/null || true
}

# ── Cleanup chính ──
cleanup() {
  local exit_code=$?
  echo ""
  echo "=== Shutdown signal received, cleaning up ==="

  # Step 1: Graceful — gửi SIGTERM cho tất cả process
  echo "[1/${SHUTDOWN_TIMEOUT}s] Graceful shutdown (SIGTERM)..."

  # Backend trước (để nó cleanup resource)
  if [[ -n "${BACKEND_PID:-}" ]]; then
    echo "  → Killing backend (PID $BACKEND_PID)..."
    kill_pg "$BACKEND_PID" TERM
  fi

  # LLaMA server sau
  if [[ -n "${LLAMA_PID:-}" ]]; then
    echo "  → Killing llama-server (PID $LLAMA_PID)..."
    kill_pg "$LLAMA_PID" TERM
  fi

  # Force kill process trên port (dự phòng)
  kill_port "$BACKEND_PORT" TERM
  kill_port "$LLAMA_PORT" TERM

  # Step 2: Chờ xong → force kill nếu còn sống
  local waited=0
  while [[ $waited -lt $SHUTDOWN_TIMEOUT ]]; do
    local alive=0
    if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
      alive=1
    fi
    if [[ -n "${LLAMA_PID:-}" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; then
      alive=1
    fi
    # Cũng check port
    if command -v lsof &>/dev/null; then
      if lsof -i :"$BACKEND_PORT" -i :"$LLAMA_PORT" &>/dev/null; then
        alive=1
      fi
    fi
    if [[ $alive -eq 0 ]]; then
      break
    fi
    sleep 1
    waited=$((waited + 1))
    if [[ $((waited % SHUTDOWN_WARN_INTERVAL)) -eq 0 ]]; then
      echo "  ... waiting ${waited}s for processes to stop"
    fi
  done

  # Step 3: Force kill (SIGKILL) nếu còn sống sau timeout
  if [[ $waited -ge $SHUTDOWN_TIMEOUT ]]; then
    echo "[TIMEOUT ${SHUTDOWN_TIMEOUT}s] Force killing remaining processes..."
    if [[ -n "${BACKEND_PID:-}" ]]; then
      kill_pg "$BACKEND_PID" KILL 2>/dev/null || true
    fi
    if [[ -n "${LLAMA_PID:-}" ]]; then
      kill_pg "$LLAMA_PID" KILL 2>/dev/null || true
    fi
    kill_port "$BACKEND_PORT" KILL
    kill_port "$LLAMA_PORT" KILL
    echo "Force kill done."
  else
    echo "All processes stopped gracefully after ${waited}s."
  fi

  # Step 4: Final check — giải phóng port dứt điểm
  kill_port "$BACKEND_PORT" KILL
  kill_port "$LLAMA_PORT" KILL

  echo "=== Cleanup complete ==="
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

# ── Đảm bảo port chưa bị chiếm trước khi start ──
echo "Checking port availability..."
kill_port "$BACKEND_PORT" KILL 2>/dev/null || true
kill_port "$LLAMA_PORT" KILL 2>/dev/null || true
sleep 1

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

echo ""
echo "=== Servers running ==="
echo "  LLaMA server  : PID $LLAMA_PID  on port $LLAMA_PORT"
echo "  FastAPI backend: PID $BACKEND_PID on port $BACKEND_PORT"
echo "  Stop with: Ctrl+C"
echo ""

wait "$LLAMA_PID" "$BACKEND_PID"