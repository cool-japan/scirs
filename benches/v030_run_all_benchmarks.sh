#!/bin/bash
# SciRS2 v0.3.0 Comprehensive Benchmark Runner
#
# This script runs all v0.3.0 benchmark suites and generates reports
#
# Usage:
#   ./v030_run_all_benchmarks.sh [--quick]
#
# Options:
#   --quick  Run quick benchmarks (reduced measurement time)
#
# Copyright: COOLJAPAN OU (Team Kitasan)
# License: Apache-2.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/tmp/scirs2_v030_benchmarks"
CRITERION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/target/criterion"

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}Running in QUICK mode (reduced measurement time)${NC}"
fi

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║     SciRS2 v0.3.0 Comprehensive Benchmark Suite Runner    ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run a benchmark
run_benchmark() {
    local bench_name=$1
    local description=$2

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Running: ${description}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Set RUSTFLAGS for performance
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

    # Run the benchmark
    if cargo bench --package scirs2-benchmarks --bench "$bench_name"; then
        echo -e "${GREEN}✓ $description completed successfully${NC}"
        echo ""
    else
        echo -e "${RED}✗ $description failed${NC}"
        echo ""
        return 1
    fi
}

# Start timestamp
START_TIME=$(date +%s)
echo -e "${YELLOW}Start time: $(date)${NC}"
echo ""

# Run benchmarks in sequence
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Phase 1/4: Comprehensive Suite (Core Operations)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
run_benchmark "v030_comprehensive_suite" "Comprehensive Suite (Linear Algebra, FFT, Stats, Clustering)"

echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Phase 2/4: Autograd Benchmarks${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
run_benchmark "v030_autograd_benchmarks" "Autograd Benchmarks (Forward/Backward Pass, Gradients)"

echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Phase 3/4: Neural Network Benchmarks${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
run_benchmark "v030_neural_benchmarks" "Neural Network Benchmarks (MNIST, CIFAR-10, Inference)"

echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Phase 4/4: Time Series Benchmarks${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
run_benchmark "v030_series_benchmarks" "Time Series Benchmarks (ARIMA, SARIMA, Decomposition)"

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Benchmark Summary                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ All benchmarks completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Total execution time: ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${YELLOW}End time: $(date)${NC}"
echo ""

# Check for result files
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Benchmark Results:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -f "/tmp/scirs2_v030_comprehensive_results.json" ]; then
    echo -e "${GREEN}  ✓ Comprehensive Suite: /tmp/scirs2_v030_comprehensive_results.json${NC}"
fi

if [ -f "/tmp/scirs2_v030_autograd_results.json" ]; then
    echo -e "${GREEN}  ✓ Autograd: /tmp/scirs2_v030_autograd_results.json${NC}"
fi

if [ -f "/tmp/scirs2_v030_neural_results.json" ]; then
    echo -e "${GREEN}  ✓ Neural Networks: /tmp/scirs2_v030_neural_results.json${NC}"
fi

if [ -f "/tmp/scirs2_v030_series_results.json" ]; then
    echo -e "${GREEN}  ✓ Time Series: /tmp/scirs2_v030_series_results.json${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}HTML Reports:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "$CRITERION_DIR" ]; then
    echo -e "${GREEN}  ✓ Criterion reports: $CRITERION_DIR${NC}"
    echo -e "${YELLOW}    Open the following in a browser:${NC}"
    echo -e "${YELLOW}    file://$CRITERION_DIR/report/index.html${NC}"
else
    echo -e "${RED}  ✗ No Criterion reports found${NC}"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Benchmark Run Complete!                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Optional: Aggregate results
echo -e "${YELLOW}To aggregate results, run:${NC}"
echo -e "${YELLOW}  python3 ${BENCH_DIR}/aggregate_v030_results.py${NC}"
echo ""

exit 0
