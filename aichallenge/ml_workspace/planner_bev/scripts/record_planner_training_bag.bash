#!/usr/bin/env bash
# Record a rosbag2 (mcap + zstd) with topics needed for planner_bev / P1 extraction.
# Prerequisites: source your ROS 2 workspace (e.g. aichallenge/workspace/install/setup.bash).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_BEV_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_TOPICS_FILE="${PLANNER_BEV_ROOT}/config/planner_record_topics.txt"
DEFAULT_QOS_OVERRIDES="${PLANNER_BEV_ROOT}/config/planner_record_qos_overrides.yaml"

usage() {
  sed -n '1,80p' <<'EOF'
Usage: record_planner_training_bag.bash [options]

  -o, --out-dir DIR     Base directory for this capture (default: ./datasets/rosbag2_planner)
  -t, --topics-file F   Topic list file (default: config/planner_record_topics.txt)
  -d, --duration SEC    Max duration in seconds (omit or 0 = until SIGINT)
      --no-compression  Disable zstd file compression
      --no-qos-overrides  Do not pass --qos-profile-overrides-path (not recommended)
      --check-topics  Only verify topics (list + publishers); do not record, then exit 0/1
  -h, --help            This help

Environment:
  AICHALLENGE_WORKSPACE  If set, sources install/setup.bash before recording.
  If unset and /aichallenge/workspace/install/setup.bash exists, that file is
  sourced (typical in evaluation Docker). Sourcing is done with nounset
  disabled briefly so colcon setup.bash does not fail on unset COLCON_TRACE.
  SKIP_RECORD_PREFLIGHT  If set to 1, skip all topic checks (not recommended).

Topic checks (default before recording, or with --check-topics):
  1) Each topic appears in ros2 topic list
  2) ros2 topic info: Publisher count >= 1 for all except /parameter_events
     (/aichallenge/objects with 0 publishers → WARN only; empty world)

QoS:
  Uses config/planner_record_qos_overrides.yaml so BEST_EFFORT trajectory/objects
  and TRANSIENT_LOCAL /tf_static match the recorder (fixes clock-only bags).

Writes:
  <out-dir>/planner_<timestamp>/              # ros2 bag record -o (created by ros2)
  <out-dir>/planner_<timestamp>.dataset_manifest.json  # sidecar (before record; avoids "folder exists")
  copy → <out-dir>/planner_<timestamp>/dataset_manifest.json after record if bag dir exists

Example:
  ./scripts/record_planner_training_bag.bash -o ~/bags/kart_run1
  ./scripts/record_planner_training_bag.bash --check-topics
  ./scripts/record_planner_training_bag.bash --check-topics -t config/planner_record_topics_extended.txt
EOF
}

OUT_BASE="${PWD}/datasets/rosbag2_planner"
TOPICS_FILE="${DEFAULT_TOPICS_FILE}"
DURATION_SEC=""
COMPRESS=1
USE_QOS_OVERRIDES=1
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out-dir) OUT_BASE="$2"; shift 2 ;;
    -t|--topics-file) TOPICS_FILE="$2"; shift 2 ;;
    -d|--duration) DURATION_SEC="$2"; shift 2 ;;
    --no-compression) COMPRESS=0; shift ;;
    --no-qos-overrides) USE_QOS_OVERRIDES=0; shift ;;
    --check-topics) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# Colcon's setup.bash references vars like COLCON_TRACE; with bash -u that errors unless we relax nounset here.
_source_workspace_setup() {
  local setup_bash="$1"
  if [[ ! -f "${setup_bash}" ]]; then
    echo "Workspace setup not found: ${setup_bash}" >&2
    return 1
  fi
  set +u
  # shellcheck disable=SC1091
  source "${setup_bash}"
  set -u
}

if [[ -n "${AICHALLENGE_WORKSPACE:-}" ]]; then
  _source_workspace_setup "${AICHALLENGE_WORKSPACE}/install/setup.bash"
elif [[ -f "/aichallenge/workspace/install/setup.bash" ]]; then
  _source_workspace_setup "/aichallenge/workspace/install/setup.bash"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 not found. Source your workspace first, e.g.:" >&2
  echo "  source /path/to/aichallenge/workspace/install/setup.bash" >&2
  exit 1
fi

if [[ ! -f "${TOPICS_FILE}" ]]; then
  echo "Topics file not found: ${TOPICS_FILE}" >&2
  exit 1
fi

mapfile -t TOPICS < <(grep -v '^[[:space:]]*#' "${TOPICS_FILE}" | grep -v '^[[:space:]]*$' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
if [[ ${#TOPICS[@]} -eq 0 ]]; then
  echo "No topics parsed from ${TOPICS_FILE}" >&2
  exit 1
fi

_preflight_topics() {
  if [[ "${SKIP_RECORD_PREFLIGHT:-}" == "1" ]]; then
    echo "[topic-check] skipped (SKIP_RECORD_PREFLIGHT=1)"
    return 0
  fi
  local all
  if ! all="$(ros2 topic list 2>/dev/null)"; then
    echo "ERROR: ros2 topic list failed (daemon not running or workspace not sourced?)" >&2
    return 1
  fi

  local missing=()
  for t in "${TOPICS[@]}"; do
    if ! grep -qxF "${t}" <<<"${all}"; then
      missing+=("$t")
    fi
  done
  if ((${#missing[@]})); then
    echo "ERROR: These topics are not in ros2 topic list (no publisher / wrong ROS_DOMAIN_ID / stack not up):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "Hint: start sim + reference.launch, then e.g.  ros2 topic hz /bev_scene_stack/tensor" >&2
    return 1
  fi
  echo "[topic-check] all ${#TOPICS[@]} topics appear in ros2 topic list."

  local t pc
  local pub_fail=()
  for t in "${TOPICS[@]}"; do
    if [[ "$t" == "/parameter_events" ]]; then
      echo "  [topic-check] $t — skipping Publisher count (often 0; optional topic)."
      continue
    fi
    pc="$(_topic_publisher_count "$t")"
    if [[ "$pc" == "-1" ]]; then
      echo "  WARN: $t — could not parse Publisher count from ros2 topic info (check ros2 CLI / locale)." >&2
      continue
    fi
    if [[ "$pc" -eq 0 ]]; then
      if [[ "$t" == "/aichallenge/objects" ]]; then
        echo "  WARN: $t — Publisher count is 0 (no object source; BEV obstacle channel may be empty)." >&2
      else
        echo "  ERROR: $t — Publisher count is 0 (no data will be recorded for this topic)." >&2
        pub_fail+=("$t")
      fi
    else
      echo "  [topic-check] $t — Publisher count: ${pc}"
    fi
  done

  if ((${#pub_fail[@]})); then
    echo "ERROR: One or more required topics have no publisher (see above)." >&2
    return 1
  fi
  echo "[topic-check] publisher check passed."
  return 0
}

_topic_publisher_count() {
  local topic="$1"
  local line
  line="$(ros2 topic info "${topic}" 2>/dev/null | grep -iE '^Publisher count:' | head -1 || true)"
  if [[ -z "${line}" ]]; then
    echo -1
    return 0
  fi
  # Typical: "Publisher count: 1"
  local n
  n="$(sed -n 's/^[Pp]ublisher count:[[:space:]]*\([0-9][0-9]*\).*/\1/p' <<<"${line}")"
  if [[ -z "${n}" ]]; then
    echo -1
    return 0
  fi
  echo "${n}"
}

if ! _preflight_topics; then
  exit 1
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "[check-topics] OK — exiting without recording."
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_BASE}"
OUT_DIR="${OUT_BASE}/planner_${TS}"
# Sidecar path: ros2 bag record refuses -o if that directory already exists, so we must not mkdir OUT_DIR first.
MANIFEST_SIDECAR="${OUT_BASE}/planner_${TS}.dataset_manifest.json"

echo "Recording to: ${OUT_DIR}"
echo "Topics (${#TOPICS[@]}):"
printf '  %s\n' "${TOPICS[@]}"

python3 "${SCRIPT_DIR}/write_planner_record_manifest.py" \
  --topics-file "${TOPICS_FILE}" \
  --bag-directory "${OUT_DIR}" \
  --manifest-path "${MANIFEST_SIDECAR}" \
  --note "planner_bev record_planner_training_bag.bash"

RECORD_CMD=(ros2 bag record -s mcap -o "${OUT_DIR}")
if [[ "${USE_QOS_OVERRIDES}" -eq 1 && -f "${DEFAULT_QOS_OVERRIDES}" ]]; then
  RECORD_CMD+=(--qos-profile-overrides-path "${DEFAULT_QOS_OVERRIDES}")
  echo "[qos] using overrides: ${DEFAULT_QOS_OVERRIDES}"
elif [[ "${USE_QOS_OVERRIDES}" -eq 1 ]]; then
  echo "WARN: QoS overrides file missing: ${DEFAULT_QOS_OVERRIDES}" >&2
fi
if [[ "${COMPRESS}" -eq 1 ]]; then
  RECORD_CMD+=(--compression-mode file --compression-format zstd)
fi
if [[ -n "${DURATION_SEC}" && "${DURATION_SEC}" != "0" ]]; then
  RECORD_CMD+=(-d "${DURATION_SEC}")
fi
RECORD_CMD+=("${TOPICS[@]}")

cleanup() {
  local ec=$?
  echo "" >&2
  echo "Stopping bag recorder (exit code ${ec})..." >&2
}
trap cleanup INT TERM

set +e
"${RECORD_CMD[@]}"
RC=$?
set -e

if [[ -f "${MANIFEST_SIDECAR}" ]]; then
  python3 - "${MANIFEST_SIDECAR}" "${RC}" <<'PY'
import json, pathlib, sys
import datetime as dt

p = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
data = json.loads(p.read_text(encoding="utf-8"))
data["recorder_finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
data["ros2_bag_exit_code"] = rc
p.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
fi

if [[ -d "${OUT_DIR}" && -f "${MANIFEST_SIDECAR}" ]]; then
  cp -f "${MANIFEST_SIDECAR}" "${OUT_DIR}/dataset_manifest.json"
  echo "Done. Manifest: ${OUT_DIR}/dataset_manifest.json (copy of ${MANIFEST_SIDECAR})"
else
  echo "Done. Manifest (sidecar only): ${MANIFEST_SIDECAR}"
  if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Note: bag directory was not created (ros2 may have failed before writing)." >&2
  fi
fi
exit "${RC}"
