#!/usr/bin/env bash
# Run a flappy command on a fresh Google Colab GPU, fetch its logs, release the VM.
#
# Usage:
#   scripts/colab_job.sh [--gpu A100] [--branch BRANCH] [--name NAME] \
#       [--fetch logs/curious-student] [--poll 60] -- python3 scripts/run_curious_student.py ...
#
# Needs the Colab CLI (`uv tool install google-colab-cli`) signed in to a Google
# account with Colab compute units. The VM clones BRANCH from GitHub, installs
# peft (Colab ships torch, transformers and tensorboard), starts the command in
# the background, and is polled until it exits. The command's output directory
# (--fetch) is tarred and unpacked into logs/colab/<name>/ locally. The VM is
# stopped on every exit path. No local secrets are sent.
set -euo pipefail

GPU=A100
BRANCH=claude/project-thread-mkmtbe
NAME="flappy-$(date +%Y%m%d%H%M%S)"
FETCH=logs/curious-student
POLL=60
REPO=https://github.com/lydakis/flappy

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --fetch) FETCH="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ $# -gt 0 ]] || { echo "missing command after --" >&2; exit 2; }

CMD_JSON=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")
LOCAL_DIR="logs/colab/$NAME"
mkdir -p "$LOCAL_DIR"

remote() {  # run Python from stdin on the VM
    colab exec -s "$NAME" --timeout "${1:-120}"
}

cleanup() {
    colab stop -s "$NAME" || echo "WARNING: colab stop failed; check 'colab sessions'" >&2
}

started=$(date +%s)
colab new -s "$NAME" --gpu "$GPU"
trap cleanup EXIT
echo "provisioned in $(( $(date +%s) - started ))s"

remote 900 <<EOF
import subprocess
subprocess.run(["git", "clone", "--depth", "1", "-b", "$BRANCH", "$REPO", "/content/flappy"], check=True)
subprocess.run(["pip", "install", "-q", "peft"], check=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
EOF

remote 60 <<EOF
import subprocess
job_log = open("/content/job.log", "w")
job = subprocess.Popen($CMD_JSON, cwd="/content/flappy", stdout=job_log, stderr=subprocess.STDOUT)
print("started pid", job.pid)
EOF

# Poll until the job exits. Each poll is a kernel execution in the same kernel
# (so job is still defined), which also keeps the session alive.
while true; do
    sleep "$POLL"
    status=$(remote 60 <<'EOF'
lines = open("/content/job.log").read().splitlines()
code = job.poll()
print(("RUNNING" if code is None else f"DONE exit={code}") + " | " + (lines[-1][-160:] if lines else ""))
EOF
)
    echo "[$(( ($(date +%s) - started) / 60 ))m] $status"
    [[ "$status" == *DONE* ]] && break
done

remote 300 <<EOF
import subprocess
subprocess.run(["tar", "czf", "/content/out.tgz", "-C", "/content/flappy", "$FETCH"], check=False)
EOF
colab download -s "$NAME" /content/out.tgz "$LOCAL_DIR/out.tgz" || true
colab download -s "$NAME" /content/job.log "$LOCAL_DIR/job.log" || true
if [[ -s "$LOCAL_DIR/out.tgz" ]]; then tar xzf "$LOCAL_DIR/out.tgz" -C "$LOCAL_DIR"; fi
echo "wall clock $(( $(date +%s) - started ))s; logs in $LOCAL_DIR"
