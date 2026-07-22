#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/vendor/JetsonYolov5"
PLUGIN="$REPO/yolov5/build/libmyplugins.so"
ENGINE="$REPO/yolov5/build/yolov5n.engine"

if [[ -f "$PLUGIN" && -f "$ENGINE" && "${FORCE_REBUILD:-0}" != "1" ]]; then
  echo "YOLO/TensorRT artifacts are already built; skipping."
  echo "Use FORCE_REBUILD=1 $0 to rebuild them."
  exit 0
fi

if ! python3 -c 'import torch, torchvision, cv2, numpy, pandas, requests, yaml, PIL, scipy, psutil, tqdm, seaborn, imutils' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
One or more YOLO build dependencies are missing or incompatible.
Run: bash ./scripts/install_yolo_build_deps.sh
Then run this script again.
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

echo "TensorRT engine ready: $ENGINE"
