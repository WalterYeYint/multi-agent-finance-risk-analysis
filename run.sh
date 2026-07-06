#!/usr/bin/env bash
#
# Start the full local stack: frontend dev server (:3000), Flask backend (:8000),
# and the worker (foreground, keeps this script alive). Ctrl+C stops all three.
#
# set -e: exit on error. NOTE it does NOT catch a failure in a backgrounded
# (`&`) command — that's why the port pre-flight below matters.
set -e

# --- Pre-flight: free the ports from any lingering previous run ----------------
# Backgrounded services (the Flask reloader child, the node dev server) can
# survive a Ctrl+C and keep holding :3000/:8000. If that happens, the backend we
# start below can't bind — and since it's backgrounded, `set -e` won't catch the
# failure, so you'd silently keep talking to the STALE backend (old code). Clear
# the ports first so every run starts clean.
for port in 3000 8000; do
  lingering="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$lingering" ]; then
    echo "⚠️  freeing port $port (killing lingering PIDs: $lingering)"
    echo "$lingering" | xargs kill 2>/dev/null || true
  fi
done
sleep 1

# --- Cleanup: kill our background children (and their whole tree) on exit ------
# Traps Ctrl+C (INT), termination (TERM), and normal/error exit (EXIT) so the
# frontend + backend never outlive this script.
CHILD_PIDS=()
kill_tree() {
  local pid="$1"
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  kill "$pid" 2>/dev/null || true
}
cleanup() {
  trap - INT TERM EXIT
  echo ""
  echo "🧹 shutting down frontend + backend..."
  for pid in "${CHILD_PIDS[@]}"; do kill_tree "$pid"; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- Frontend dev server (background) ------------------------------------------
( cd frontend && npm start ) &
CHILD_PIDS+=("$!")

# --- Backend (background) ------------------------------------------------------
python -m backend.app &
CHILD_PIDS+=("$!")

# --- Worker (foreground — keeps the script alive) -----------------------------
python -m src.worker
