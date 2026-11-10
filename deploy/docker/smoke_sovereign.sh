#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
IMG="${IMG:-crispro-oncology-ruo:72h}"
docker build -f deploy/docker/Dockerfile.sovereign -t "$IMG" .
CID="$(docker run -d -p 18080:8080 "$IMG")"
cleanup() { docker rm -f "$CID" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:18080/healthz" >/dev/null \
    || curl -sf "http://127.0.0.1:18080/health" >/dev/null; then
    echo "{\"status\":\"ok\",\"image\":\"$IMG\",\"healthz\":200}"
    exit 0
  fi
  sleep 2
done
echo "{\"status\":\"failed\",\"error\":\"healthz_timeout\"}" >&2
exit 1
