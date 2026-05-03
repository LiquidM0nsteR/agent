#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

PYTHON_BIN="../enter/envs/agent/bin/python"
MYSQL_BASE="../local/mysql"
MYSQL_CNF="${MYSQL_BASE}/etc/my.cnf"
MYSQL_LOG="${MYSQL_BASE}/logs/mysqld_safe.out"
MYSQL_DATA_FILE="${MYSQL_BASE}/data/ibdata1"
MYSQLADMIN_BIN="${MYSQL_BASE}/bin/mysqladmin"

REDIS_BIN="../local/redis-stack-server-6.2.6/bin/redis-stack-server"
REDIS_CLI="../local/redis-stack-server-6.2.6/bin/redis-cli"
REDIS_HOST="127.0.0.1"
REDIS_PORT="6379"
REDIS_DATA_DIR="../local/redis-stack-data"
REDIS_LOG="../local/redis-stack-log/redis-stack.log"

VLLM_PID_FILE="data/vllm.pid"
VLLM_PORT="${VLLM_PORT:-8080}"
BACKEND_LOG="${AGENT_BACKEND_LOG:-data/backend_log.txt}"
export VLLM_BIN="${VLLM_BIN:-../enter/envs/agent/bin/vllm}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"

STARTED_MYSQL=0
STARTED_REDIS=0
UVICORN_PID=""
STOP_EXISTING_SERVICES_ON_EXIT="${AGENT_STOP_EXISTING_SERVICES_ON_EXIT:-0}"
STOP_EXISTING_VLLM_ON_EXIT="${AGENT_STOP_EXISTING_VLLM_ON_EXIT:-0}"

mysql_ready() {
  "${PYTHON_BIN}" - <<'PY'
import pymysql

try:
    connection = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="123456",
        connect_timeout=2,
    )
except Exception:
    raise SystemExit(1)
else:
    connection.close()
PY
}

mysql_data_locked() {
  "${PYTHON_BIN}" - "${MYSQL_DATA_FILE}" <<'PY'
import errno
import fcntl
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    handle = path.open("r+b")
except FileNotFoundError:
    raise SystemExit(1)

with handle:
    try:
        fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise SystemExit(0)
        raise
    else:
        fcntl.lockf(handle, fcntl.LOCK_UN)
        raise SystemExit(1)
PY
}

redis_ready() {
  "${REDIS_CLI}" -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping >/dev/null 2>&1
}

start_mysql() {
  if mysql_ready; then
    echo "MySQL ready on 127.0.0.1:3306"
    if [ "${STOP_EXISTING_SERVICES_ON_EXIT}" = "1" ]; then
      STARTED_MYSQL=1
    fi
    return
  fi

  if mysql_data_locked; then
    echo "MySQL data directory is already locked: ${MYSQL_DATA_FILE}" >&2
    echo "127.0.0.1:3306 is not reachable from this shell, so refusing to start a second mysqld." >&2
    echo "Run the backend in the same environment as the running MySQL, or stop the existing mysqld before running ./start.sh." >&2
    exit 1
  fi

  mkdir -p "${MYSQL_BASE}/logs"
  echo "Starting MySQL with ${MYSQL_CNF}"
  "${MYSQL_BASE}/bin/mysqld_safe" --defaults-file="${MYSQL_CNF}" >>"${MYSQL_LOG}" 2>&1 &
  STARTED_MYSQL=1

  for _ in $(seq 1 120); do
    if mysql_ready; then
      echo "MySQL ready on 127.0.0.1:3306"
      return
    fi
    sleep 1
  done

  echo "MySQL failed to become ready. Last log lines:" >&2
  tail -n 80 "${MYSQL_LOG}" >&2 || true
  tail -n 120 "${MYSQL_BASE}/logs/error.log" >&2 || true
  exit 1
}

start_redis() {
  if redis_ready; then
    echo "Redis ready on ${REDIS_HOST}:${REDIS_PORT}"
    if [ "${STOP_EXISTING_SERVICES_ON_EXIT}" = "1" ]; then
      STARTED_REDIS=1
    fi
    return
  fi

  mkdir -p "${REDIS_DATA_DIR}" "$(dirname "${REDIS_LOG}")"
  echo "Starting Redis Stack on ${REDIS_HOST}:${REDIS_PORT}"
  "${REDIS_BIN}" \
    --dir "${REDIS_DATA_DIR}" \
    --bind "${REDIS_HOST}" \
    --port "${REDIS_PORT}" \
    --appendonly yes \
    --daemonize yes \
    --logfile "${REDIS_LOG}"
  STARTED_REDIS=1

  for _ in $(seq 1 60); do
    if redis_ready; then
      echo "Redis ready on ${REDIS_HOST}:${REDIS_PORT}"
      return
    fi
    sleep 1
  done

  echo "Redis failed to become ready. Last log lines:" >&2
  tail -n 120 "${REDIS_LOG}" >&2 || true
  exit 1
}

wait_until_mysql_stopped() {
  for _ in $(seq 1 30); do
    if ! mysql_ready; then
      return
    fi
    sleep 1
  done
}

wait_until_redis_stopped() {
  for _ in $(seq 1 30); do
    if ! redis_ready; then
      return
    fi
    sleep 1
  done
}

stop_mysql() {
  if [ "${STARTED_MYSQL}" != "1" ]; then
    return
  fi
  if ! mysql_ready; then
    return
  fi
  echo "Stopping MySQL started by this script"
  "${MYSQLADMIN_BIN}" --host=127.0.0.1 --port=3306 --user=root --password=123456 shutdown >/dev/null 2>&1 || true
  wait_until_mysql_stopped
}

stop_redis() {
  if [ "${STARTED_REDIS}" != "1" ]; then
    return
  fi
  if ! redis_ready; then
    return
  fi
  echo "Stopping Redis started by this script"
  "${REDIS_CLI}" -h "${REDIS_HOST}" -p "${REDIS_PORT}" shutdown nosave >/dev/null 2>&1 || true
  wait_until_redis_stopped
}

stop_uvicorn() {
  if [ -n "${UVICORN_PID}" ] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}

wait_until_pid_stopped() {
  local pid="$1"
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return
    fi
    sleep 1
  done
}

listener_pids_for_port() {
  local port="$1"
  "${PYTHON_BIN}" - "${port}" <<'PY'
import os
import sys
from pathlib import Path

port = int(sys.argv[1])
uid = os.getuid()
inodes: set[str] = set()

for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
    try:
        lines = table.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        local_port = int(fields[1].rsplit(":", 1)[1], 16)
        if local_port == port:
            inodes.add(fields[9])

if not inodes:
    raise SystemExit(0)

pids: set[int] = set()
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != uid:
            continue
    except OSError:
        continue
    fd_dir = proc / "fd"
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        continue
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
            pids.add(int(proc.name))
            break

for pid in sorted(pids):
    print(pid)
PY
}

force_stop_port_listeners() {
  local service="$1"
  local port="$2"
  local pids

  pids="$(listener_pids_for_port "${port}" || true)"
  if [ -z "${pids}" ]; then
    echo "${service} port ${port} is free"
    return
  fi

  echo "${service} port ${port} still has listeners: ${pids}"
  for pid in ${pids}; do
    if [ -n "${UVICORN_PID}" ] && [ "${pid}" = "${UVICORN_PID}" ]; then
      continue
    fi
    echo "Stopping ${service} listener pid=${pid}"
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  done

  for _ in $(seq 1 10); do
    pids="$(listener_pids_for_port "${port}" || true)"
    if [ -z "${pids}" ]; then
      echo "${service} port ${port} released"
      return
    fi
    sleep 1
  done

  pids="$(listener_pids_for_port "${port}" || true)"
  for pid in ${pids}; do
    echo "Force stopping ${service} listener pid=${pid}"
    kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
  done

  for _ in $(seq 1 10); do
    pids="$(listener_pids_for_port "${port}" || true)"
    if [ -z "${pids}" ]; then
      echo "${service} port ${port} released"
      return
    fi
    sleep 1
  done

  pids="$(listener_pids_for_port "${port}" || true)"
  if [ -n "${pids}" ]; then
    echo "WARNING: ${service} port ${port} still has listeners after cleanup: ${pids}" >&2
  fi
}

stop_vllm_pid() {
  local pid="$1"
  if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  echo "Stopping vLLM pid=${pid}"
  kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  wait_until_pid_stopped "${pid}"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Force stopping vLLM pid=${pid}"
    kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    wait_until_pid_stopped "${pid}"
  fi
}

stop_vllm() {
  if [ -f "${VLLM_PID_FILE}" ]; then
    local pid
    pid="$(tr -d '[:space:]' < "${VLLM_PID_FILE}" || true)"
    stop_vllm_pid "${pid}"
    rm -f "${VLLM_PID_FILE}"
    echo "vLLM stopped; GPU memory released by process exit"
    return
  fi

  if [ "${STOP_EXISTING_VLLM_ON_EXIT}" = "1" ] || [ "${STOP_EXISTING_SERVICES_ON_EXIT}" = "1" ]; then
    local pids
    pids="$(pgrep -u "$(id -u)" -f "vllm serve .*--port ${VLLM_PORT}" || true)"
    for pid in ${pids}; do
      stop_vllm_pid "${pid}"
    done
    if [ -n "${pids}" ]; then
      echo "vLLM stopped; GPU memory released by process exit"
    fi
  fi
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  stop_uvicorn
  stop_vllm
  stop_redis
  stop_mysql
  force_stop_port_listeners "vLLM" "${VLLM_PORT}"
  force_stop_port_listeners "Redis" "${REDIS_PORT}"
  force_stop_port_listeners "MySQL" "3306"
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_mysql
start_redis

if [ "$#" -eq 0 ]; then
  set -- --host 127.0.0.1 --port 8000
fi
mkdir -p "$(dirname "${BACKEND_LOG}")"
echo "Starting backend; log=${BACKEND_LOG}"
"${PYTHON_BIN}" -m uvicorn backend.main:app "$@" >>"${BACKEND_LOG}" 2>&1 &
UVICORN_PID=$!
wait "${UVICORN_PID}"
