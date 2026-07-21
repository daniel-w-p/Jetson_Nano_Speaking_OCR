#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_VERSION="2023.11.14-2"
VOICE="pl_PL-gosia-medium"
PYCUDA_VERSION="2022.1"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer must run on the Jetson (aarch64)." >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  echo "Run this script as the regular login user, without sudo." >&2
  echo "The script requests sudo only for the steps that need it." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  alsa-utils build-essential cmake curl fswebcam git htop nano \
  python3-dev python3-numpy python3-opencv python3-pip \
  tesseract-ocr tesseract-ocr-pol v4l-utils wget \
  gfortran libblas-dev libfreetype6-dev libjpeg-dev liblapack-dev \
  libboost-all-dev libopenblas-base libopenblas-dev libopenmpi-dev zlib1g-dev

if [[ ! -d /usr/local/cuda/include || ! -d /usr/local/cuda/lib64 ]]; then
  echo "CUDA toolkit not found under /usr/local/cuda." >&2
  echo "Verify the JetPack installation before running this script again." >&2
  exit 1
fi

# Make CUDA available both to this build and to future interactive shells.
ensure_bashrc_line() {
  local line="$1"
  touch "$HOME/.bashrc"
  grep -Fqx "$line" "$HOME/.bashrc" || printf '\n%s\n' "$line" >>"$HOME/.bashrc"
}

ensure_bashrc_line 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
ensure_bashrc_line 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_INC_DIR=/usr/local/cuda/include
export CUDA_NDARRAY_CUDA_H=1
nvcc --version

# JetPack 4.6.1 uses Python 3.6. Build the last PyCUDA line supporting it
# instead of relying on the unavailable python3-pycuda APT package.
python3 -m pip install --user --upgrade \
  "pip<22" "setuptools<60" "wheel<0.38"
python3 -m pip install --user "Cython==0.29.36" numpy
if ! python3 -c 'import pycuda.driver' >/dev/null 2>&1; then
  python3 -m pip install --user --no-cache-dir \
    --global-option=build_ext \
    --global-option="-I${CUDA_INC_DIR}" \
    --global-option="-L/usr/local/cuda/lib64" \
    "pycuda==${PYCUDA_VERSION}"
fi
python3 -c "import pycuda.driver as cuda; print('CUDA version:', cuda.get_version())"

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
