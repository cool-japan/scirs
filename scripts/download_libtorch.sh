#!/usr/bin/env bash
set -euo pipefail

# Download and prepare the official LibTorch (C++ distribution)
# Usage:
#   bash scripts/download_libtorch.sh [cpu|mps|cuda] [version]
# Examples:
#   bash scripts/download_libtorch.sh cpu 2.4.1
#   bash scripts/download_libtorch.sh mps 2.4.1   # macOS with Apple Silicon (MPS)
#   bash scripts/download_libtorch.sh cuda 12.1   # Linux CUDA 12.1

BACKEND=${1:-cpu}
VERSION=${2:-2.4.1}

UNAME_S=$(uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M=$(uname -m)

OUT_DIR=${OUT_DIR:-"$PWD/libtorch-dist"}
mkdir -p "$OUT_DIR"

echo "Target OS: $UNAME_S, Arch: $UNAME_M, Backend: $BACKEND, Version: $VERSION"

URL=""
ARCHIVE=""

if [[ "$UNAME_S" == "darwin" ]]; then
  # macOS builds
  case "$BACKEND" in
    cpu)
      # Universal CPU build (works for x86_64/arm64 via Rosetta if needed)
      URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip"
      ;;
    mps)
      # MPS-enabled build for Apple Silicon (arm64)
      URL="https://download.pytorch.org/libtorch/cu121/libtorch-macos-mps-${VERSION}.zip"
      ;;
    *)
      echo "Unsupported backend '$BACKEND' for macOS. Use 'cpu' or 'mps'." >&2
      exit 1
      ;;
  esac
  ARCHIVE="$OUT_DIR/libtorch-${BACKEND}-${VERSION}-macos.zip"
elif [[ "$UNAME_S" == "linux" ]]; then
  # Linux builds
  case "$BACKEND" in
    cpu)
      URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip"
      ;;
    cuda)
      CUDA_VER=${CUDA_VER:-12.1}
      # Note: Adjust CUDA_VER to match the desired CUDA runtime
      URL="https://download.pytorch.org/libtorch/cu${CUDA_VER/./}/libtorch-shared-with-deps-${VERSION}%2Bcu${CUDA_VER/./}.zip"
      ;;
    *)
      echo "Unsupported backend '$BACKEND' for Linux. Use 'cpu' or 'cuda'." >&2
      exit 1
      ;;
  esac
  ARCHIVE="$OUT_DIR/libtorch-${BACKEND}-${VERSION}-linux.zip"
else
  echo "Unsupported OS: $UNAME_S" >&2
  exit 1
fi

echo "Downloading: $URL"
curl -L --fail --progress-bar "$URL" -o "$ARCHIVE"

echo "Unpacking to: $OUT_DIR"
unzip -q -o "$ARCHIVE" -d "$OUT_DIR"

# The archive extracts into $OUT_DIR/libtorch
LIBTORCH_DIR="$OUT_DIR/libtorch"
if [[ ! -d "$LIBTORCH_DIR" ]]; then
  echo "Error: Expected directory not found: $LIBTORCH_DIR" >&2
  exit 1
fi

cat <<EOF

LibTorch is ready at: $LIBTORCH_DIR

Add these to your shell before building crates that depend on tch/torch-sys:

  export LIBTORCH="$LIBTORCH_DIR"
  case "\${OSTYPE:-}" in
    darwin*) export DYLD_LIBRARY_PATH="\$LIBTORCH/lib:\${DYLD_LIBRARY_PATH:-}" ;;
    linux*)  export LD_LIBRARY_PATH="\$LIBTORCH/lib:\${LD_LIBRARY_PATH:-}" ;;
  esac

Then build, for example:
  cargo build -p scirs2-transform --features auto-feature-engineering

EOF

echo "Done."
