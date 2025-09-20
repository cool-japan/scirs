#!/usr/bin/env bash
set -euo pipefail

# Configure environment so torch-sys (tch) can find libtorch
# Usage: source scripts/setup_libtorch_env.sh

PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python and PyTorch first." >&2
  return 1 2>/dev/null || exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(1 if importlib.util.find_spec("torch") is None else 0)
PY
then
  echo "Error: Python package 'torch' not found. Install it, e.g.:" >&2
  echo "  $PYTHON_BIN -m pip install --upgrade pip" >&2
  echo "  $PYTHON_BIN -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu" >&2
  return 1 2>/dev/null || exit 1
fi

TORCH_INFO=$("$PYTHON_BIN" - <<'PY'
import os, torch
pkg_dir = os.path.dirname(torch.__file__)
lib_dir = os.path.join(pkg_dir, 'lib')
cmake_prefix = getattr(torch.utils, 'cmake_prefix_path', None)
print(torch.__version__)
print(pkg_dir)
print(lib_dir)
print(cmake_prefix if cmake_prefix else '')
PY
)

TORCH_VER=$(echo "$TORCH_INFO" | sed -n '1p')
TORCH_PKG_DIR=$(echo "$TORCH_INFO" | sed -n '2p')
TORCH_LIB_DIR=$(echo "$TORCH_INFO" | sed -n '3p')
TORCH_CMAKE_PREFIX=$(echo "$TORCH_INFO" | sed -n '4p')

export LIBTORCH_USE_PYTORCH=1
# Optionally bypass version check if your torch version differs from torch-sys expectation
export LIBTORCH_BYPASS_VERSION_CHECK=${LIBTORCH_BYPASS_VERSION_CHECK:-1}

# On macOS, ensure runtime can locate dylibs when running binaries/tests.
case "${OSTYPE:-}" in
  darwin*)
    if [ -d "$TORCH_LIB_DIR" ]; then
      # Prepend to DYLD_LIBRARY_PATH if not already present
      case ":${DYLD_LIBRARY_PATH-}:" in
        *:"$TORCH_LIB_DIR":*) ;;
        *) export DYLD_LIBRARY_PATH="$TORCH_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" ;;
      esac
    fi
    ;;
esac

echo "Configured libtorch from Python:'$PYTHON_BIN'"
echo "- torch version: $TORCH_VER"
echo "- torch package: $TORCH_PKG_DIR"
echo "- torch lib dir: $TORCH_LIB_DIR"
if [ -n "$TORCH_CMAKE_PREFIX" ]; then
  echo "- cmake prefix : $TORCH_CMAKE_PREFIX"
fi
echo "Environment set: LIBTORCH_USE_PYTORCH=1, LIBTORCH_BYPASS_VERSION_CHECK=${LIBTORCH_BYPASS_VERSION_CHECK:-}"
case "${OSTYPE:-}" in
  darwin*) echo "DYLD_LIBRARY_PATH includes: $TORCH_LIB_DIR";;
esac
