#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_VERSION="2023.11.14-2"
VOICE="pl_PL-gosia-medium"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer must run on the Jetson (aarch64)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  alsa-utils build-essential cmake curl fswebcam git htop nano \
  python3-dev python3-numpy python3-opencv python3-pip python3-pycuda \
  tesseract-ocr tesseract-ocr-pol v4l-utils wget \
  gfortran libblas-dev libfreetype6-dev libjpeg-dev liblapack-dev \
  libopenblas-base libopenblas-dev libopenmpi-dev zlib1g-dev

# Optional at runtime, but pinned here so the same image is ready for button control.
python3 -m pip install --user "Jetson.GPIO==2.1.6"
GPIO_RULE="$(python3 -c 'import Jetson.GPIO, os; print(os.path.join(os.path.dirname(Jetson.GPIO.__file__), "99-gpio.rules"))')"
sudo groupadd -f -r gpio
sudo usermod -a -G gpio "$USER"
sudo cp "$GPIO_RULE" /etc/udev/rules.d/99-gpio.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

mkdir -p "$ROOT/bin" "$ROOT/models/piper" "$ROOT/tmp" "$ROOT/vendor"
if [[ ! -f "$ROOT/config/config.json" ]]; then
  cp "$ROOT/config/config.example.json" "$ROOT/config/config.json"
fi

ARCHIVE="$ROOT/tmp/piper_linux_aarch64.tar.gz"
if [[ ! -x "$ROOT/bin/piper/piper" ]]; then
  wget -O "$ARCHIVE" \
    "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
  tar -xzf "$ARCHIVE" -C "$ROOT/bin"
fi

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium"
for suffix in onnx onnx.json; do
  target="$ROOT/models/piper/${VOICE}.${suffix}"
  [[ -f "$target" ]] || wget -O "$target" "$BASE/${VOICE}.${suffix}"
done

echo "Bootstrap complete. Test with: python3 $ROOT/src/say.py"
