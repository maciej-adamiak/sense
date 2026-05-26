#!/bin/sh

set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 <input.bin|.npz>" >&2; exit 1; }

[ -f /mnt/system/usr/bin/envs_tpu_sdk.sh ] && . /mnt/system/usr/bin/envs_tpu_sdk.sh

exec python3 "${HERE}/run_yamnet.py" "$@"
