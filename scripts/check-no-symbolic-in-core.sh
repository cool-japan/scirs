#!/usr/bin/env bash
# check-no-symbolic-in-core.sh — Cycle-prevention CI gate
#
# Enforces two rules from ADR-0001:
#   1. scirs2-core MUST NOT depend on scirs2-symbolic (would create cycle:
#      scirs2-symbolic → scirs2-core → scirs2-symbolic).
#   2. oxieml MUST NOT appear in the production dependency graph
#      (only in [dev-dependencies] of scirs2-symbolic).
#
# Exit codes:
#   0 — both rules satisfied
#   1 — Rule 1 violated (scirs2-core depends on scirs2-symbolic)
#   2 — Rule 2 violated (oxieml in production deps)
#   3 — Tool dependencies missing (cargo or jq)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

# Tool check
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found in PATH" >&2
    exit 3
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found in PATH (required for dependency graph parsing)" >&2
    echo "       Install with: brew install jq  (macOS)" >&2
    echo "       Install with: apt-get install jq  (Debian/Ubuntu)" >&2
    exit 3
fi

echo "Running cargo metadata (this may take a moment)..."
METADATA=$(cargo metadata --format-version 1 --no-deps 2>/dev/null) || {
    echo "ERROR: cargo metadata failed" >&2
    exit 3
}

# Rule 1: scirs2-core must not depend on scirs2-symbolic
#
# Walk all `dependencies` of `scirs2-core` (Cargo.toml direct deps).
# Production deps appear with `kind == null`; dev/build deps have kind = "dev"/"build".
echo "[1/2] Checking that scirs2-core does not depend on scirs2-symbolic..."
SYMBOLIC_IN_CORE=$(echo "${METADATA}" | jq -r '
    .packages[]
    | select(.name == "scirs2-core")
    | .dependencies[]
    | select(.name == "scirs2-symbolic" and .kind == null)
    | .name
' || true)

if [[ -n "${SYMBOLIC_IN_CORE}" ]]; then
    echo "FAIL: scirs2-core depends on scirs2-symbolic (production dep)" >&2
    echo "      This would create a dependency cycle." >&2
    echo "      Remove the dep from scirs2-core/Cargo.toml." >&2
    exit 1
fi
echo "      OK"

# Rule 2: oxieml must not be a production dep of any workspace crate
#
# Iterate every workspace package; for each, list production deps; check oxieml absence.
echo "[2/2] Checking that oxieml is not a production dep of any workspace crate..."
OXIEML_PROD_DEPS=$(echo "${METADATA}" | jq -r '
    .packages[]
    | select(.source == null)         # workspace crates only (path deps have source = null)
    | {pkg: .name, deps: [.dependencies[] | select(.name == "oxieml" and .kind == null) | .name]}
    | select(.deps | length > 0)
    | .pkg
' || true)

if [[ -n "${OXIEML_PROD_DEPS}" ]]; then
    echo "FAIL: oxieml appears as production dep in: ${OXIEML_PROD_DEPS}" >&2
    echo "      ADR-0001 mandates oxieml as [dev-dependencies] only." >&2
    echo "      Move oxieml under [dev-dependencies] in the affected Cargo.toml." >&2
    exit 2
fi
echo "      OK"

echo ""
echo "All cycle-prevention rules satisfied."
exit 0
