#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_VERSION="2023.11.14-2"
PIPER_PHONEMIZE_VERSION="2023.11.14-4"
PIPER_SOURCE="$ROOT/vendor/piper-source"
PIPER_BUILD="$PIPER_SOURCE/build-jetson-gcc8"
PIPER_INSTALL="$ROOT/bin/piper-jetson"
PIPER_BIN="$PIPER_INSTALL/piper"
PIPER_PHONEMIZE_SOURCE="$ROOT/vendor/piper-phonemize-source"
PIPER_PHONEMIZE_BUILD="$PIPER_PHONEMIZE_SOURCE/build-jetson-gcc8"
PIPER_PHONEMIZE_INSTALL="$ROOT/vendor/piper-phonemize-jetson"
CMAKE_VERSION="3.22.6"
CC_BIN="/usr/bin/gcc-8"
CXX_BIN="/usr/bin/g++-8"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This build must run on the Jetson (aarch64)." >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  echo "Run this script as the regular login user, without sudo." >&2
  exit 1
fi

if [[ -x "$PIPER_BIN" ]] && "$PIPER_BIN" --help >/dev/null 2>&1; then
  echo "Jetson-compatible Piper is already built; skipping."
  exit 0
fi

APT_PACKAGES=(
  autoconf automake build-essential g++-8 gcc-8 git libtool pkg-config
  patchelf python3-pip unzip wget
)
MISSING_APT_PACKAGES=()
for package in "${APT_PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null | grep -Fxq 'install ok installed'; then
    MISSING_APT_PACKAGES+=("$package")
  fi
done

if (( ${#MISSING_APT_PACKAGES[@]} > 0 )); then
  echo "Installing missing Piper build packages: ${MISSING_APT_PACKAGES[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING_APT_PACKAGES[@]}"
else
  echo "Piper APT build dependencies are already installed; skipping."
fi

if [[ ! -x "$CC_BIN" || ! -x "$CXX_BIN" ]]; then
  echo "GCC/G++ 8 is required to compile the C++17 <filesystem> header." >&2
  exit 1
fi

USER_BASE="$(python3 -m site --user-base)"
CMAKE_BIN="$USER_BASE/bin/cmake"
if [[ ! -x "$CMAKE_BIN" ]] || ! "$CMAKE_BIN" --version | head -n 1 | grep -Fq "$CMAKE_VERSION"; then
  python3 -m pip install --user "cmake==$CMAKE_VERSION"
fi
export PATH="$(dirname "$CMAKE_BIN")${PATH:+:${PATH}}"
export CC="$CC_BIN"
export CXX="$CXX_BIN"

mkdir -p "$ROOT/vendor" "$ROOT/bin"
if [[ -e "$PIPER_SOURCE" && ! -d "$PIPER_SOURCE/.git" ]]; then
  echo "$PIPER_SOURCE exists but is not a Git checkout; move it aside and retry." >&2
  exit 1
fi
if [[ ! -d "$PIPER_SOURCE/.git" ]]; then
  git clone --depth 1 --branch "$PIPER_VERSION" \
    https://github.com/rhasspy/piper.git "$PIPER_SOURCE"
fi

SOURCE_VERSION="$(git -C "$PIPER_SOURCE" describe --tags --exact-match 2>/dev/null || true)"
if [[ "$SOURCE_VERSION" != "$PIPER_VERSION" ]]; then
  echo "Unexpected Piper source version in $PIPER_SOURCE: ${SOURCE_VERSION:-unknown}" >&2
  echo "Expected tag: $PIPER_VERSION" >&2
  exit 1
fi

FILESYSTEM_CHECK="$(mktemp /tmp/piper-filesystem-check.XXXXXX)"
trap 'rm -f "$FILESYSTEM_CHECK"' EXIT
if ! "$CXX_BIN" -std=c++17 -x c++ - -lstdc++fs -o "$FILESYSTEM_CHECK" <<'CPP'
#include <filesystem>
int main() { return std::filesystem::path(".").empty(); }
CPP
then
  echo "G++ 8 cannot compile and link C++17 <filesystem> with stdc++fs." >&2
  exit 1
fi
rm -f "$FILESYSTEM_CHECK"
trap - EXIT

if [[ -e "$PIPER_PHONEMIZE_SOURCE" && ! -d "$PIPER_PHONEMIZE_SOURCE/.git" ]]; then
  echo "$PIPER_PHONEMIZE_SOURCE exists but is not a Git checkout; move it aside and retry." >&2
  exit 1
fi
if [[ ! -d "$PIPER_PHONEMIZE_SOURCE/.git" ]]; then
  git clone --depth 1 --branch "$PIPER_PHONEMIZE_VERSION" \
    https://github.com/rhasspy/piper-phonemize.git "$PIPER_PHONEMIZE_SOURCE"
fi

PHONEMIZE_SOURCE_VERSION="$(
  git -C "$PIPER_PHONEMIZE_SOURCE" describe --tags --exact-match 2>/dev/null || true
)"
if [[ "$PHONEMIZE_SOURCE_VERSION" != "$PIPER_PHONEMIZE_VERSION" ]]; then
  echo "Unexpected piper-phonemize source version in $PIPER_PHONEMIZE_SOURCE:" \
    "${PHONEMIZE_SOURCE_VERSION:-unknown}" >&2
  echo "Expected tag: $PIPER_PHONEMIZE_VERSION" >&2
  exit 1
fi

echo "Building piper-phonemize with GCC 8..."
"$CMAKE_BIN" -S "$PIPER_PHONEMIZE_SOURCE" -B "$PIPER_PHONEMIZE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PIPER_PHONEMIZE_INSTALL" \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs
"$CMAKE_BIN" --build "$PIPER_PHONEMIZE_BUILD" --parallel "${BUILD_JOBS:-2}"
"$CMAKE_BIN" --install "$PIPER_PHONEMIZE_BUILD"

# The eSpeak fork builds ucd as a shared dependency but does not install it.
LIBUCD_SOURCE="$(
  find "$PIPER_PHONEMIZE_BUILD" -type f -name 'libucd.so*' -print -quit
)"
if [[ -z "$LIBUCD_SOURCE" ]]; then
  echo "Built libucd.so was not found under $PIPER_PHONEMIZE_BUILD." >&2
  exit 1
fi
install -D -m 0755 "$LIBUCD_SOURCE" \
  "$PIPER_PHONEMIZE_INSTALL/lib/libucd.so"

echo "Building Piper with GCC 8..."
"$CMAKE_BIN" -S "$PIPER_SOURCE" -B "$PIPER_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PIPER_INSTALL" \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs \
  -DPIPER_PHONEMIZE_DIR="$PIPER_PHONEMIZE_INSTALL"
"$CMAKE_BIN" --build "$PIPER_BUILD" --parallel "${BUILD_JOBS:-2}"
"$CMAKE_BIN" --install "$PIPER_BUILD"

# Keep all runtime dependencies in the portable Piper directory. RPATH on
# dependent shared libraries is required because executable RUNPATH is not
# applied transitively by the dynamic loader.
install -m 0755 "$PIPER_PHONEMIZE_INSTALL/lib/libucd.so" \
  "$PIPER_INSTALL/libucd.so"
for elf_file in \
  "$PIPER_BIN" \
  "$PIPER_INSTALL"/libespeak-ng.so* \
  "$PIPER_INSTALL"/libpiper_phonemize.so*; do
  if [[ -f "$elf_file" ]]; then
    patchelf --set-rpath '$ORIGIN' "$elf_file"
  fi
done

MISSING_SHARED_LIBS="$(
  ldd "$PIPER_BIN" 2>&1 | awk '/not found/ { print $1 }' || true
)"
if [[ -n "$MISSING_SHARED_LIBS" ]]; then
  echo "Piper still has missing shared libraries:" >&2
  while IFS= read -r missing_library; do
    echo "  $missing_library" >&2
  done <<<"$MISSING_SHARED_LIBS"
  exit 1
fi

"$PIPER_BIN" --help >/dev/null
echo "Jetson-compatible Piper ready: $PIPER_BIN"
