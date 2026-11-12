#!/usr/bin/env bash
# save_baseline.sh — SciRS2 Performance Baseline Capture Tool
#
# Runs the v0.3.0 benchmark suites via `cargo bench` and saves the Criterion
# mean point estimates into benches/performance_baseline.json using the
# regression_check.py --update-baseline mechanism.
#
# Usage:
#   ./benches/save_baseline.sh                         # Run all v0.3.0 suites
#   ./benches/save_baseline.sh --dry-run               # Show what would be done
#   ./benches/save_baseline.sh --bench-filter v030_autograd  # One suite only
#   ./benches/save_baseline.sh --bench-filter v030_comprehensive --bench-filter v030_series
#
# Must be run from the repository root:
#   cd /path/to/scirs && ./benches/save_baseline.sh
#
# After saving, commit the updated baseline with:
#   git add benches/performance_baseline.json
#   git commit -m "perf: update performance baseline $(date +%Y-%m-%d)"
#
# Copyright: COOLJAPAN OU (Team KitaSan)
# License: Apache-2.0

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BENCH_DIR="${REPO_ROOT}/benches"
BASELINE_FILE="${BENCH_DIR}/performance_baseline.json"
CRITERION_DIR="${REPO_ROOT}/target/criterion"
REGRESSION_SCRIPT="${BENCH_DIR}/regression_check.py"

# Default benchmark suites to run (v0.3.0 set)
DEFAULT_SUITES=(
    "v030_comprehensive_suite"
    "v030_autograd_benchmarks"
    "v030_series_benchmarks"
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=false
BENCH_FILTERS=()

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run               Show commands that would be executed without running them"
    echo "  --bench-filter NAME     Run only the named benchmark suite (repeatable)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Default suites: ${DEFAULT_SUITES[*]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --bench-filter)
            if [[ -z "${2:-}" ]]; then
                echo -e "${RED}Error: --bench-filter requires a value${NC}" >&2
                exit 1
            fi
            BENCH_FILTERS+=("$2")
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: unknown argument '$1'${NC}" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

# Determine which suites to run
if [[ ${#BENCH_FILTERS[@]} -gt 0 ]]; then
    SUITES_TO_RUN=("${BENCH_FILTERS[@]}")
else
    SUITES_TO_RUN=("${DEFAULT_SUITES[@]}")
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_dry()     { echo -e "${CYAN}[DRY-RUN]${NC} would run: $*"; }

run_cmd() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_dry "$*"
        return 0
    fi
    "$@"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          SciRS2 Performance Baseline Capture Tool         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [[ ! -d "${REPO_ROOT}" ]]; then
    log_error "Repository root not found: ${REPO_ROOT}"
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/Cargo.toml" ]]; then
    log_error "Must be run from within the SciRS2 repository (Cargo.toml not found at ${REPO_ROOT})"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    log_error "python3 is required but not found in PATH"
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    log_error "cargo is required but not found in PATH"
    exit 1
fi

if [[ ! -f "${REGRESSION_SCRIPT}" ]]; then
    log_error "Regression script not found: ${REGRESSION_SCRIPT}"
    exit 1
fi

log_info "Repository root : ${REPO_ROOT}"
log_info "Baseline file   : ${BASELINE_FILE}"
log_info "Criterion dir   : ${CRITERION_DIR}"
log_info "Suites to run   : ${SUITES_TO_RUN[*]}"

if [[ "${DRY_RUN}" == "true" ]]; then
    log_warn "DRY-RUN mode: no commands will actually be executed"
fi

echo ""

# ---------------------------------------------------------------------------
# Capture baseline from existing Criterion results (if present)
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "false" ]] && [[ -d "${CRITERION_DIR}" ]]; then
    EXISTING_COUNT=$(find "${CRITERION_DIR}" -name "estimates.json" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${EXISTING_COUNT}" -gt 0 ]]; then
        log_info "Found ${EXISTING_COUNT} existing criterion result(s) in ${CRITERION_DIR}"
    fi
fi

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
START_TS=$(date +%s)
FAILED_SUITES=()

for suite in "${SUITES_TO_RUN[@]}"; do
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log_info "Running benchmark suite: ${suite}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    BENCH_CMD=(
        cargo bench
        --package scirs2-benchmarks
        --bench "${suite}"
    )

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_dry "RUSTFLAGS='-C target-cpu=native -C opt-level=3' ${BENCH_CMD[*]}"
    else
        if RUSTFLAGS="-C target-cpu=native -C opt-level=3" "${BENCH_CMD[@]}"; then
            log_ok "Suite '${suite}' completed successfully"
        else
            log_warn "Suite '${suite}' failed (exit $?); continuing to next suite"
            FAILED_SUITES+=("${suite}")
        fi
    fi
    echo ""
done

END_TS=$(date +%s)
DURATION=$(( END_TS - START_TS ))
MINUTES=$(( DURATION / 60 ))
SECONDS=$(( DURATION % 60 ))

echo ""
log_info "Benchmark run time: ${MINUTES}m ${SECONDS}s"

if [[ ${#FAILED_SUITES[@]} -gt 0 ]]; then
    log_warn "The following suites encountered errors: ${FAILED_SUITES[*]}"
    log_warn "Continuing with baseline update for available results."
fi

# ---------------------------------------------------------------------------
# Count Criterion results available
# ---------------------------------------------------------------------------
CRITERION_COUNT=0
if [[ "${DRY_RUN}" == "false" ]] && [[ -d "${CRITERION_DIR}" ]]; then
    CRITERION_COUNT=$(find "${CRITERION_DIR}" -name "estimates.json" 2>/dev/null | wc -l | tr -d ' ')
fi

if [[ "${DRY_RUN}" == "false" ]] && [[ "${CRITERION_COUNT}" -eq 0 ]]; then
    log_error "No Criterion results found in ${CRITERION_DIR}."
    log_error "Benchmarks may have failed to compile or run."
    exit 1
fi

if [[ "${DRY_RUN}" == "false" ]]; then
    log_info "Found ${CRITERION_COUNT} Criterion estimates to save as baseline"
fi

# ---------------------------------------------------------------------------
# Save baseline via regression_check.py
# ---------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log_info "Saving baseline to ${BASELINE_FILE}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

SAVE_CMD=(
    python3 "${REGRESSION_SCRIPT}"
    --update-baseline
    --baselines "${BASELINE_FILE}"
    --criterion-dir "${CRITERION_DIR}"
)

if run_cmd "${SAVE_CMD[@]}"; then
    if [[ "${DRY_RUN}" == "false" ]]; then
        log_ok "Baseline saved successfully"
    fi
else
    log_error "Failed to save baseline (exit $?)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  Baseline Save Complete                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [[ "${DRY_RUN}" == "false" ]]; then
    # Count entries in saved baseline
    if command -v python3 &>/dev/null && [[ -f "${BASELINE_FILE}" ]]; then
        ENTRY_COUNT=$(python3 -c "
import json
with open('${BASELINE_FILE}') as f:
    data = json.load(f)
entries = {k: v for k, v in data.items() if not k.startswith('_')}
print(len(entries))
" 2>/dev/null || echo "unknown")
        log_ok "Baseline file now contains ${ENTRY_COUNT} entries"
    fi

    log_ok "Baseline file : ${BASELINE_FILE}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo -e "  ${BOLD}git diff benches/performance_baseline.json${NC}  # Review changes"
    echo -e "  ${BOLD}git add benches/performance_baseline.json${NC}"
    echo -e "  ${BOLD}git commit -m \"perf: update performance baseline $(date +%Y-%m-%d)\"${NC}"
    echo ""
    echo -e "  To verify the new baseline runs clean:"
    echo -e "  ${BOLD}python3 benches/regression_check.py${NC}"
else
    log_info "Dry-run complete; no files were modified"
fi

exit 0
