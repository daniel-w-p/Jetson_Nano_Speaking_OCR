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
v4l2-ctl --list-devices 2>/dev/null || true
ls /dev/video* 2>/dev/null || true
echo "== Audio =="
aplay -l 2>/dev/null || true
echo "== OCR =="
tesseract --version 2>/dev/null | head -n 1 || true
tesseract --list-langs 2>/dev/null | grep -x pol || echo "Polish OCR data missing"
echo "== Project artifacts =="
for path in \
  "$ROOT/bin/piper/piper" \
  "$ROOT/models/piper/pl_PL-gosia-medium.onnx" \
  "$ROOT/vendor/JetsonYolov5/yolov5/build/libmyplugins.so" \
  "$ROOT/vendor/JetsonYolov5/yolov5/build/yolov5n.engine"; do
  [[ -e "$path" ]] && echo "OK  $path" || echo "MISS $path"
done
