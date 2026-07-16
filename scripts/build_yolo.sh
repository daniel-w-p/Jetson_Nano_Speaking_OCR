#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/vendor/JetsonYolov5"

if ! python3 -c 'import torch' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
PyTorch is required only to convert yolov5n.pt to .wts.
Install the JetPack 4 / Python 3.6 wheel described in docs/SD_CARD_SETUP.md,
then run this script again.
EOF
  exit 1
fi

mkdir -p "$ROOT/vendor"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/mailrocketsystems/JetsonYolov5.git "$REPO"
fi

cd "$REPO"
python3 gen_wts.py -w yolov5n.pt -o yolov5n.wts
mkdir -p yolov5/build
cp yolov5n.wts yolov5/build/
cd yolov5/build
cmake ..
make -j"${BUILD_JOBS:-2}"
./yolov5_det -s yolov5n.wts yolov5n.engine n

echo "TensorRT engine ready: $REPO/yolov5/build/yolov5n.engine"
