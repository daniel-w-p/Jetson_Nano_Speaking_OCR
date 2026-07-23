#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== Platform =="
cat /etc/nv_tegra_release 2>/dev/null || echo "Not running L4T"
uname -a
python3 --version
echo "== Memory =="
free -h

echo "== Camera =="
CAMERA_LIST="$(v4l2-ctl --list-devices 2>/dev/null || true)"
if [[ -n "$CAMERA_LIST" ]]; then
  printf '%s\n' "$CAMERA_LIST"
else
  echo "WARN No V4L2 camera detected. Connect the camera and check its cable/USB port."
fi

shopt -s nullglob
VIDEO_DEVICES=(/dev/video*)
shopt -u nullglob
if (( ${#VIDEO_DEVICES[@]} > 0 )); then
  printf 'OK   %s\n' "${VIDEO_DEVICES[@]}"
else
  echo "WARN No /dev/video* device nodes found."
fi

echo "== Audio =="
aplay -l 2>/dev/null || true

echo "== OCR =="
tesseract --version 2>/dev/null | head -n 1 || true
tesseract --list-langs 2>/dev/null | grep -x pol || echo "Polish OCR data missing"

echo "== Bootstrap artifacts =="
for path in \
  "$ROOT/models/piper/pl_PL-gosia-medium.onnx"; do
  [[ -e "$path" ]] && echo "OK  $path" || echo "MISS $path"
done

PIPER_BIN="$ROOT/bin/piper-jetson/piper"
if [[ -x "$PIPER_BIN" ]] && "$PIPER_BIN" --help >/dev/null 2>&1; then
  echo "OK  $PIPER_BIN"
else
  echo "MISS Jetson-compatible Piper: $PIPER_BIN"
  echo "NEXT Run: bash ./scripts/build_piper_jetson.sh"
fi

echo "== Generated YOLO/TensorRT artifacts =="
YOLO_PLUGIN="$ROOT/vendor/JetsonYolov5/yolov5/build/libmyplugins.so"
YOLO_ENGINE="$ROOT/vendor/JetsonYolov5/yolov5/build/yolov5n.engine"
YOLO_PENDING=0
for path in "$YOLO_PLUGIN" "$YOLO_ENGINE"; do
  if [[ -e "$path" ]]; then
    echo "OK      $path"
  else
    echo "PENDING $path"
    YOLO_PENDING=1
  fi
done

if (( YOLO_PENDING )); then
  echo "These files are generated locally, not downloaded by bootstrap_jetson.sh."
  if python3 -c 'import torch, torchvision, cv2, numpy, pandas, requests, yaml, PIL, scipy, psutil, tqdm, seaborn, imutils' >/dev/null 2>&1; then
    echo "NEXT YOLO build dependencies are ready. Run: ./scripts/build_yolo.sh"
  else
    echo "NEXT Run: bash ./scripts/install_yolo_build_deps.sh"
    echo "     Then run: ./scripts/build_yolo.sh"
  fi
fi
