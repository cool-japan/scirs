#!/usr/bin/env bash
# check-wasm-size.sh — Verify the compiled WASM binary does not exceed the size budget.
#
# Usage:
#   ./scripts/check-wasm-size.sh [--pkg-dir <path>]
#
# Default pkg-dir is <repo-root>/scirs2-wasm/pkg (wasm-pack bundler output).
#
# Exit codes:
#   0  — WASM binary is within budget, or no binary found (skip, not fail).
#   1  — WASM binary exceeds the size budget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── configurable ──────────────────────────────────────────────────────────────
PKG_DIR="${REPO_ROOT}/scirs2-wasm/pkg"
LIMIT_BYTES=2097152  # 2 MiB — matches WASM_BINARY_SIZE_LIMIT_BYTES in binary_size_check.rs
WASM_FILENAME="scirs2_wasm_bg.wasm"

# Allow caller to override the pkg dir
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pkg-dir)
            PKG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

WASM_FILE="${PKG_DIR}/${WASM_FILENAME}"

if [[ ! -f "${WASM_FILE}" ]]; then
    echo "INFO: WASM binary not found at ${WASM_FILE} — skipping size check."
    echo "      Run 'wasm-pack build --release -p scirs2-wasm' first."
    exit 0
fi

# Get file size in bytes (portable: wc -c works on both macOS and Linux)
SIZE=$(wc -c < "${WASM_FILE}" | tr -d ' ')
SIZE_KB=$(( SIZE / 1024 ))
LIMIT_KB=$(( LIMIT_BYTES / 1024 ))

echo "WASM binary: ${WASM_FILE}"
echo "Size:        ${SIZE} bytes  (${SIZE_KB} KiB)"
echo "Limit:       ${LIMIT_BYTES} bytes  (${LIMIT_KB} KiB)"

if [[ "${SIZE}" -gt "${LIMIT_BYTES}" ]]; then
    OVER=$(( SIZE - LIMIT_BYTES ))
    echo ""
    echo "ERROR: WASM binary exceeds size budget by ${OVER} bytes ($(( OVER / 1024 )) KiB)."
    echo "       Consider:"
    echo "         • wasm-opt -Oz to optimize the binary"
    echo "         • Auditing new dependencies with 'cargo bloat --release'"
    echo "         • Moving large modules behind feature flags"
    exit 1
fi

echo ""
echo "OK: WASM binary is within the ${LIMIT_KB} KiB budget."
exit 0
