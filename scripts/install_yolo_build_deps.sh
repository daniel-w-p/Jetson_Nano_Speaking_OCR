#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_VERSION_REQUIRED="1.10.0"
TORCHVISION_VERSION_REQUIRED="0.11.1"
TORCH_WHEEL="/tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
TORCHVISION_REPO="$ROOT/vendor/torchvision"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer must run on the Jetson (aarch64)." >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  echo "Run this script as the regular login user, without sudo." >&2
  exit 1
fi

APT_PACKAGES=(
  build-essential git wget python3-dev python3-numpy python3-opencv python3-pip
  libjpeg-dev libopenblas-dev libopenmpi-dev libpng-dev zlib1g-dev
  python3-pandas python3-psutil python3-requests python3-scipy
  python3-seaborn python3-yaml
)

MISSING_APT_PACKAGES=()
for package in "${APT_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null | grep -Fxq 'install ok installed'; then
    MISSING_APT_PACKAGES+=("$package")
  fi
done

if (( ${#MISSING_APT_PACKAGES[@]} > 0 )); then
  echo "Installing missing YOLO build packages: ${MISSING_APT_PACKAGES[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING_APT_PACKAGES[@]}"
else
  echo "YOLO APT build dependencies are already installed; skipping."
fi

if python3 -c 'import PIL, imutils, tqdm; raise SystemExit(PIL.__version__ != "8.4.0" or tqdm.__version__ != "4.64.1")' >/dev/null 2>&1; then
  echo "Compatible Pillow, tqdm and imutils are already installed; skipping."
else
  python3 -m pip install --user \
    "Pillow==8.4.0" "tqdm==4.64.1" "imutils==0.5.4"
fi

TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [[ "$TORCH_VERSION" == "$TORCH_VERSION_REQUIRED"* ]]; then
  echo "PyTorch $TORCH_VERSION is already installed; skipping."
elif [[ -n "$TORCH_VERSION" ]]; then
  echo "Incompatible PyTorch version found: $TORCH_VERSION" >&2
  echo "JetPack 4 / Python 3.6 requires PyTorch $TORCH_VERSION_REQUIRED for this build." >&2
  exit 1
else
  wget -O "$TORCH_WHEEL" "$TORCH_URL"
  python3 -m pip install --user "$TORCH_WHEEL"
fi

TORCHVISION_VERSION="$(python3 -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || true)"
if [[ "$TORCHVISION_VERSION" == "$TORCHVISION_VERSION_REQUIRED"* ]]; then
  echo "torchvision $TORCHVISION_VERSION is already installed; skipping source build."
else
  mkdir -p "$ROOT/vendor"
  if [[ -e "$TORCHVISION_REPO" && ! -d "$TORCHVISION_REPO/.git" ]]; then
    echo "$TORCHVISION_REPO exists but is not a Git checkout; move it aside and retry." >&2
    exit 1
  fi
  if [[ ! -d "$TORCHVISION_REPO/.git" ]]; then
    git clone --depth 1 --branch "v$TORCHVISION_VERSION_REQUIRED" \
      https://github.com/pytorch/vision.git "$TORCHVISION_REPO"
  fi

  cd "$TORCHVISION_REPO"
  BUILD_VERSION="$TORCHVISION_VERSION_REQUIRED" \
    MAX_JOBS="${BUILD_JOBS:-2}" \
    python3 -m pip install --user --no-deps .
fi

python3 - <<'PY'
import importlib

modules = (
    "torch", "torchvision", "cv2", "numpy", "pandas", "requests",
    "yaml", "PIL", "scipy", "psutil", "tqdm", "seaborn", "imutils",
)
for module in modules:
    importlib.import_module(module)
print("YOLO Python build dependencies: OK")
PY
