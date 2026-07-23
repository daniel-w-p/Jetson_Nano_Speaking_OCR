#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

APT_PACKAGES=(
  alsa-utils build-essential cmake curl fswebcam git htop nano
  python3-dev python3-numpy python3-opencv python3-pip
  tesseract-ocr tesseract-ocr-pol v4l-utils wget
  gfortran libblas-dev libfreetype6-dev libjpeg-dev liblapack-dev
  libboost-all-dev libopenblas-base libopenblas-dev libopenmpi-dev zlib1g-dev
)

MISSING_APT_PACKAGES=()
for package in "${APT_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null | grep -Fxq 'install ok installed'; then
    MISSING_APT_PACKAGES+=("$package")
  fi
done

if (( ${#MISSING_APT_PACKAGES[@]} > 0 )); then
  echo "Installing missing APT packages: ${MISSING_APT_PACKAGES[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING_APT_PACKAGES[@]}"
else
  echo "All required APT packages are already installed; skipping APT."
fi

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
if python3 -c 'import pkg_resources; pkg_resources.require(["pip>=21.3,<22", "setuptools>=59,<60", "wheel>=0.37,<0.38"])' >/dev/null 2>&1; then
  echo "Compatible pip, setuptools and wheel are already installed; skipping."
else
  python3 -m pip install --user --upgrade \
    "pip>=21.3,<22" "setuptools>=59,<60" "wheel>=0.37,<0.38"
fi

if python3 -c 'import Cython, numpy; raise SystemExit(Cython.__version__ != "0.29.36")' >/dev/null 2>&1; then
  echo "Cython 0.29.36 and NumPy are already installed; skipping."
else
  python3 -m pip install --user "Cython==0.29.36" numpy
fi

if python3 -c 'import pycuda.driver' >/dev/null 2>&1; then
  echo "PyCUDA is already installed and importable; skipping compilation."
else
  python3 -m pip install --user --no-cache-dir \
    --global-option=build_ext \
    --global-option="-I${CUDA_INC_DIR}" \
    --global-option="-L/usr/local/cuda/lib64" \
    "pycuda==${PYCUDA_VERSION}"
fi
python3 -c "import pycuda.driver as cuda; print('CUDA version:', cuda.get_version())"

# Do not import Jetson.GPIO before its udev rule and group are configured: the
# module checks /dev/gpiochip0 permissions during import and would abort here.
if python3 -c 'import pkg_resources; raise SystemExit(pkg_resources.get_distribution("Jetson.GPIO").version != "2.1.6")' >/dev/null 2>&1; then
  echo "Jetson.GPIO 2.1.6 is already installed; skipping package installation."
else
  python3 -m pip install --user "Jetson.GPIO==2.1.6"
fi

GPIO_SITE="$(python3 -m pip show Jetson.GPIO | sed -n 's/^Location: //p')"
GPIO_RULE="$GPIO_SITE/Jetson/GPIO/99-gpio.rules"
if [[ -z "$GPIO_SITE" || ! -f "$GPIO_RULE" ]]; then
  echo "Jetson.GPIO udev rule was not found after package installation." >&2
  exit 1
fi

sudo groupadd -f -r gpio
if id -nG "$USER" | tr ' ' '\n' | grep -Fxq gpio; then
  echo "User $USER is already a member of the gpio group."
else
  sudo usermod -a -G gpio "$USER"
fi

GPIO_RULE_TARGET=/etc/udev/rules.d/99-gpio.rules
if sudo cmp -s "$GPIO_RULE" "$GPIO_RULE_TARGET"; then
  echo "Jetson.GPIO udev rule is already current."
else
  sudo cp "$GPIO_RULE" "$GPIO_RULE_TARGET"
  sudo udevadm control --reload-rules
  sudo udevadm trigger
fi

if [[ ! -e /dev/gpiochip0 ]]; then
  echo "Warning: /dev/gpiochip0 is not present; verify the Jetson kernel/device tree." >&2
fi
echo "GPIO permissions configured. Reboot before importing Jetson.GPIO."

mkdir -p "$ROOT/bin" "$ROOT/models/piper" "$ROOT/tmp" "$ROOT/vendor"
if [[ ! -f "$ROOT/config/config.json" ]]; then
  cp "$ROOT/config/config.example.json" "$ROOT/config/config.json"
fi

PIPER_BIN="$ROOT/bin/piper-jetson/piper"
if [[ ! -x "$PIPER_BIN" ]] || ! "$PIPER_BIN" --help >/dev/null 2>&1; then
  bash "$ROOT/scripts/build_piper_jetson.sh"
else
  echo "Jetson-compatible Piper is already installed; skipping source build."
fi

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium"
for suffix in onnx onnx.json; do
  target="$ROOT/models/piper/${VOICE}.${suffix}"
  [[ -f "$target" ]] || wget -O "$target" "$BASE/${VOICE}.${suffix}"
done

PIPER_TEST_WAV="$ROOT/tmp/piper-bootstrap-test.wav"
printf '%s\n' "Test syntezy mowy." | \
  "$PIPER_BIN" \
    --model "$ROOT/models/piper/${VOICE}.onnx" \
    --output_file "$PIPER_TEST_WAV"
rm -f "$PIPER_TEST_WAV"

echo "Bootstrap complete. Test with: python3 $ROOT/src/say.py"
