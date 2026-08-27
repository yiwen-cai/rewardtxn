#!/usr/bin/env bash
# 断点续传下载 AReaL 镜像全部 blob (ghcr.io 直连, HTTP/1.1, curl -C -)
# 用法: bash download_areal_layers.sh <digests_file> <outdir>
set -uo pipefail

DIGESTS_FILE=${1:-/tmp/areal_digests.txt}
OUTDIR=${2:-/tmp/areal-oci/blobs/sha256}
REPO="areal-project/areal-runtime"
REGISTRY="https://ghcr.io/v2"
mkdir -p "$OUTDIR"

fetch_blob() {
  local digest="$1"
  local file="$OUTDIR/$digest"
  local want="${digest#sha256:}"
  # 已完整下载则跳过
  if [ -f "$file" ] && [ "$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1)" = "$want" ]; then
    echo "SKIP $digest"
    return 0
  fi
  local token
  token=$(timeout 20 curl -s "https://ghcr.io/token?service=ghcr.io&scope=repository:${REPO}:pull" | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
  for attempt in $(seq 1 200); do
    if [ -f "$file" ] && [ "$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1)" = "$want" ]; then
      echo "OK   $digest"
      return 0
    fi
    # 小文件(未完成时可能 416)或首次: 不带 -C - ; 已有部分内容时带 -C -
    local resume=""
    if [ -f "$file" ] && [ "$(stat -c%s "$file")" -gt 0 ]; then
      resume="-C -"
    fi
    timeout 900 curl -s --http1.1 $resume -L -H "Authorization: Bearer $token" \
      -o "$file" "$REGISTRY/${REPO}/blobs/$digest"
    local rc=$?
    if [ "$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1)" = "$want" ]; then
      echo "OK   $digest"
      return 0
    fi
    echo "RETRY($attempt) $digest rc=$rc size=$(stat -c%s "$file" 2>/dev/null)" >> /tmp/areal_dl_retries.log
  done
  echo "FAIL $digest"
  return 1
}

export -f fetch_blob
export OUTDIR REPO REGISTRY

tail -n +1 "$DIGESTS_FILE" | xargs -P 4 -I{} bash -c 'fetch_blob "$1"' _ {}
echo "ALL DONE"
