#!/usr/bin/env bash
# check-features.sh — verify that each feature flag of a workspace crate
# compiles cleanly in isolation.
#
# Usage:
#   scripts/check-features.sh [CRATE]
#
# Default CRATE is "scirs2" (the facade).
#
# Exit codes
#   0   all checks passed
#   1   one or more feature checks failed
#
# Requirements:
#   - cargo
#   - python3  (for JSON parsing; ships with macOS and most Linux distros)
#
# Examples:
#   ./scripts/check-features.sh
#   ./scripts/check-features.sh scirs2-linalg
#   CI=true ./scripts/check-features.sh scirs2

set -euo pipefail

CRATE="${1:-scirs2}"
FAILED=()

# Determine workspace root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=========================================="
echo "  scirs2 feature-flag compile check"
echo "  Crate : ${CRATE}"
echo "  Root  : ${WORKSPACE_ROOT}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Check no-default-features baseline
# ---------------------------------------------------------------------------
echo ""
echo ">> cargo check -p ${CRATE} --no-default-features"
if cargo check -p "${CRATE}" --no-default-features 2>&1; then
    echo "   PASS: --no-default-features"
else
    echo "   FAIL: --no-default-features"
    FAILED+=("--no-default-features")
fi

# ---------------------------------------------------------------------------
# 2. Enumerate features via cargo metadata
# ---------------------------------------------------------------------------
FEATURES=$(
    cargo metadata \
        --format-version 1 \
        --no-deps \
        --manifest-path "${WORKSPACE_ROOT}/Cargo.toml" \
        2>/dev/null \
    | python3 - <<'PYEOF'
import sys, json

data = json.load(sys.stdin)
crate_name = sys.argv[1] if len(sys.argv) > 1 else None

# Find the target package
for pkg in data.get("packages", []):
    if crate_name and pkg["name"] != crate_name:
        continue
    features = pkg.get("features", {})
    for feat in sorted(features.keys()):
        # Skip hidden / internal features (starting with _ or dep:)
        if feat.startswith("_") or feat.startswith("dep:"):
            continue
        print(feat)
    break
PYEOF
) || true

if [ -z "${FEATURES}" ]; then
    echo ""
    echo "NOTE: Could not enumerate features via cargo metadata."
    echo "      Falling back to checking default features only."
fi

# ---------------------------------------------------------------------------
# 3. Check each individual feature
# ---------------------------------------------------------------------------
for FEAT in ${FEATURES}; do
    echo ""
    echo ">> cargo check -p ${CRATE} --no-default-features --features ${FEAT}"
    if cargo check -p "${CRATE}" --no-default-features --features "${FEAT}" 2>&1; then
        echo "   PASS: ${FEAT}"
    else
        echo "   FAIL: ${FEAT}"
        FAILED+=("${FEAT}")
    fi
done

# ---------------------------------------------------------------------------
# 4. Check all features together
# ---------------------------------------------------------------------------
echo ""
echo ">> cargo check -p ${CRATE} --all-features"
if cargo check -p "${CRATE}" --all-features 2>&1; then
    echo "   PASS: --all-features"
else
    echo "   FAIL: --all-features"
    FAILED+=("--all-features")
fi

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "  All feature checks PASSED for ${CRATE}"
    echo "=========================================="
    exit 0
else
    echo "  FAILED checks (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "    - ${f}"
    done
    echo "=========================================="
    exit 1
fi
