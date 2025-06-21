#!/bin/bash

# Comprehensive benchmark runner for SciRS2 performance analysis
# This script runs all benchmarks and generates comparison reports

set -e

echo "=== SciRS2 Comprehensive Performance Benchmarking ==="
echo "Starting at: $(date)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p target/benchmark-results
mkdir -p target/plots

echo -e "${BLUE}Setting up benchmark environment...${NC}"

# Check if Python is available for SciPy comparison
PYTHON_AVAILABLE=false
if command -v python3 &> /dev/null; then
    if python3 -c "import scipy, numpy" &> /dev/null; then
        PYTHON_AVAILABLE=true
        echo -e "${GREEN}Python with SciPy available - will run comparison benchmarks${NC}"
    else
        echo -e "${YELLOW}Python available but SciPy not found - skipping comparison${NC}"
    fi
else
    echo -e "${YELLOW}Python not available - skipping SciPy comparison${NC}"
fi

# Function to run a benchmark with error handling
run_benchmark() {
    local bench_name=$1
    local description=$2
    
    echo -e "${BLUE}Running $description...${NC}"
    
    if cargo bench --bench $bench_name -- --output-format html > target/benchmark-results/${bench_name}_output.log 2>&1; then
        echo -e "${GREEN}✓ $description completed successfully${NC}"
        
        # Move HTML reports to organized location
        if [ -d "target/criterion" ]; then
            mv target/criterion target/benchmark-results/${bench_name}_criterion_reports
        fi
    else
        echo -e "${RED}✗ $description failed${NC}"
        echo "Check target/benchmark-results/${bench_name}_output.log for details"
        return 1
    fi
}

# Function to check for required dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check if criterion features are available
    if ! cargo check --bench linalg_benchmarks &> /dev/null; then
        echo -e "${RED}Failed to compile benchmarks. Check dependencies.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Dependencies check passed${NC}"
}

# Main benchmark execution
main() {
    check_dependencies
    
    # Clean previous results
    echo -e "${BLUE}Cleaning previous benchmark results...${NC}"
    rm -rf target/criterion target/benchmark-results/*
    
    # Run Rust benchmarks
    echo -e "${YELLOW}=== Running Rust Benchmarks ===${NC}"
    
    run_benchmark "linalg_benchmarks" "Linear Algebra Performance Benchmarks"
    sleep 2
    
    run_benchmark "memory_efficiency" "Memory Efficiency Benchmarks"
    sleep 2
    
    run_benchmark "numerical_stability" "Numerical Stability Tests"
    sleep 2
    
    run_benchmark "scipy_comparison" "SciPy Comparison Benchmarks"
    sleep 2
    
    # Run Python benchmarks if available
    if [ "$PYTHON_AVAILABLE" = true ]; then
        echo -e "${YELLOW}=== Running SciPy Comparison Benchmarks ===${NC}"
        
        cd benches
        if python3 scipy_benchmark.py > ../target/benchmark-results/scipy_output.log 2>&1; then
            echo -e "${GREEN}✓ SciPy benchmarks completed${NC}"
        else
            echo -e "${RED}✗ SciPy benchmarks failed${NC}"
            echo "Check target/benchmark-results/scipy_output.log for details"
        fi
        cd ..
    fi
    
    # Generate comprehensive report
    echo -e "${YELLOW}=== Generating Comprehensive Report ===${NC}"
    generate_report
    
    # Create visualizations if Python is available
    if [ "$PYTHON_AVAILABLE" = true ]; then
        echo -e "${YELLOW}=== Creating Performance Visualizations ===${NC}"
        create_visualizations
    fi
    
    echo -e "${GREEN}=== Benchmarking Complete ===${NC}"
    echo "Results available in target/benchmark-results/"
    echo "HTML reports available in target/benchmark-results/*_criterion_reports/"
    
    if [ "$PYTHON_AVAILABLE" = true ]; then
        echo "Comparison report: target/benchmark_comparison.json"
        echo "Visualizations: target/plots/"
    fi
}

# Generate comprehensive report
generate_report() {
    cat > target/benchmark-results/BENCHMARK_SUMMARY.md << 'EOF'
# SciRS2 Performance Benchmark Summary

This report contains the results of comprehensive performance benchmarking for SciRS2.

## Benchmark Categories

### 1. Linear Algebra Performance (`linalg_benchmarks`)
- Basic matrix operations (determinant, inverse, norms)
- Matrix decompositions (LU, QR, SVD, Cholesky)
- Linear system solvers (general, least squares, triangular)
- Eigenvalue computations
- Performance analysis with algorithmic complexity

### 2. Memory Efficiency (`memory_efficiency`)
- Buffer pool efficiency
- Chunked operations for large matrices
- Memory usage patterns
- Allocation strategies
- Fragmentation resistance
- Zero-copy operations

### 3. Numerical Stability (`numerical_stability`)
- Well-conditioned matrix tests
- Pathological matrix handling (Hilbert, Vandermonde, etc.)
- Decomposition accuracy verification
- Edge cases with extreme values
- Condition number analysis

### 4. SciPy Comparison (`scipy_comparison`)
- Direct performance comparison with SciPy
- Cross-platform performance analysis
- Speedup/slowdown analysis
- Memory usage comparison

## Files Generated

- `*_criterion_reports/`: Detailed HTML reports from Criterion
- `rust_benchmark_results.json`: Raw Rust benchmark data
- `python_benchmark_results.json`: Raw Python/SciPy benchmark data
- `benchmark_comparison.json`: Comprehensive comparison analysis
- `stability_test_results.json`: Numerical stability test results
- `complexity_analysis.json`: Algorithmic complexity analysis

## Reading the Results

### Criterion HTML Reports
Open `*_criterion_reports/report/index.html` in a browser for interactive results.

### JSON Data Files
Use the JSON files for programmatic analysis or custom visualization.

### Performance Metrics
- **Throughput**: Operations per second
- **Latency**: Time per operation
- **Memory Usage**: Peak memory consumption
- **Numerical Accuracy**: Relative error measurements
- **Stability**: Success rate under challenging conditions

EOF

    echo -e "${GREEN}✓ Benchmark summary generated${NC}"
}

# Create performance visualizations
create_visualizations() {
    cat > target/create_plots.py << 'EOF'
#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_json_safe(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {filename}")
        return None

def create_comparison_plot():
    """Create Rust vs SciPy performance comparison plot."""
    comparison_data = load_json_safe('target/benchmark_comparison.json')
    if not comparison_data:
        return
    
    results = comparison_data.get('results', [])
    operations = []
    speedups = []
    sizes = []
    
    for result in results:
        if result.get('speedup') is not None:
            operations.append(f"{result['operation']} ({result['size']})")
            speedups.append(result['speedup'])
            sizes.append(result['size'])
    
    if not speedups:
        print("No comparison data available")
        return
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if s > 1.0 else 'red' for s in speedups]
    bars = plt.bar(range(len(operations)), speedups, color=colors, alpha=0.7)
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Operations')
    plt.ylabel('Speedup (Rust/Python)')
    plt.title('SciRS2 vs SciPy Performance Comparison')
    plt.xticks(range(len(operations)), operations, rotation=45, ha='right')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add speedup labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('target/plots/rust_vs_scipy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created Rust vs SciPy comparison plot")

def create_stability_plot():
    """Create numerical stability analysis plot."""
    stability_data = load_json_safe('target/stability_test_results.json')
    if not stability_data:
        return
    
    # Group by condition number
    cond_numbers = {}
    for result in stability_data:
        cond = result.get('condition_number', 0)
        if cond != float('inf') and cond > 0:
            if cond not in cond_numbers:
                cond_numbers[cond] = {'success': 0, 'total': 0}
            cond_numbers[cond]['total'] += 1
            if result.get('success', False):
                cond_numbers[cond]['success'] += 1
    
    if not cond_numbers:
        print("No stability data available")
        return
    
    conds = sorted(cond_numbers.keys())
    success_rates = [cond_numbers[c]['success'] / cond_numbers[c]['total'] * 100 
                    for c in conds]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(conds, success_rates, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Condition Number')
    plt.ylabel('Success Rate (%)')
    plt.title('Numerical Stability vs Matrix Condition Number')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('target/plots/numerical_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created numerical stability plot")

def create_complexity_plot():
    """Create algorithmic complexity analysis plot."""
    complexity_data = load_json_safe('target/complexity_analysis.json')
    if not complexity_data:
        return
    
    # Extract matrix multiplication data
    matmul_data = {}
    det_data = {}
    
    for key, time_ns in complexity_data.items():
        if 'matmul_' in key:
            size = int(key.split('_')[1])
            matmul_data[size] = time_ns / 1e9  # Convert to seconds
        elif 'det_' in key:
            size = int(key.split('_')[1])
            det_data[size] = time_ns / 1e9
    
    if matmul_data:
        sizes = sorted(matmul_data.keys())
        times = [matmul_data[s] for s in sizes]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(sizes, times, 'ro-', label='Matrix Multiplication', linewidth=2)
        
        # Theoretical O(n^3) line for comparison
        theoretical = [times[0] * (s/sizes[0])**3 for s in sizes]
        plt.loglog(sizes, theoretical, 'r--', alpha=0.5, label='O(n³) theoretical')
        
        if det_data:
            det_times = [det_data.get(s, 0) for s in sizes if s in det_data]
            det_sizes = [s for s in sizes if s in det_data]
            if det_times:
                plt.loglog(det_sizes, det_times, 'bo-', label='Determinant', linewidth=2)
        
        plt.xlabel('Matrix Size (n)')
        plt.ylabel('Time (seconds)')
        plt.title('Algorithmic Complexity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('target/plots/complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created complexity analysis plot")

def main():
    os.makedirs('target/plots', exist_ok=True)
    
    print("Creating performance visualization plots...")
    create_comparison_plot()
    create_stability_plot()
    create_complexity_plot()
    print("Visualization plots created in target/plots/")

if __name__ == '__main__':
    main()
EOF

    if python3 target/create_plots.py > target/benchmark-results/plotting_output.log 2>&1; then
        echo -e "${GREEN}✓ Performance visualizations created${NC}"
    else
        echo -e "${RED}✗ Visualization creation failed${NC}"
        echo "Check target/benchmark-results/plotting_output.log for details"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-python)
            PYTHON_AVAILABLE=false
            echo "Skipping Python benchmarks as requested"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-python] [--help]"
            echo "  --skip-python  Skip SciPy comparison benchmarks"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main benchmarking
main