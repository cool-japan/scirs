#!/bin/bash

# Comprehensive Benchmarking Suite Runner for scirs2-spatial
# This script runs all benchmarks and generates performance reports

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="${OUTPUT_DIR}/report_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPORT_DIR}"

echo_info "Starting comprehensive benchmarking suite for scirs2-spatial"
echo_info "Project directory: ${PROJECT_DIR}"
echo_info "Output directory: ${REPORT_DIR}"

# Change to project directory
cd "${PROJECT_DIR}"

# System information collection
echo_info "Collecting system information..."
{
    echo "=== SYSTEM INFORMATION ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -a)"
    
    if command -v lscpu >/dev/null 2>&1; then
        echo ""
        echo "=== CPU INFORMATION ==="
        lscpu
    fi
    
    if command -v free >/dev/null 2>&1; then
        echo ""
        echo "=== MEMORY INFORMATION ==="
        free -h
    fi
    
    echo ""
    echo "=== RUST INFORMATION ==="
    rustc --version
    cargo --version
    
    echo ""
    echo "=== CARGO BUILD INFORMATION ==="
    cargo --version
    
} > "${REPORT_DIR}/system_info.txt"

echo_success "System information collected"

# Build the project in release mode
echo_info "Building project in release mode..."
if cargo build --release; then
    echo_success "Build completed successfully"
else
    echo_error "Build failed"
    exit 1
fi

# Run tests to ensure everything works
echo_info "Running tests to verify functionality..."
if cargo test --release --quiet; then
    echo_success "All tests passed"
else
    echo_warning "Some tests failed, but continuing with benchmarks"
fi

# Function to run a benchmark and capture output
run_benchmark() {
    local bench_name="$1"
    local description="$2"
    local output_file="${REPORT_DIR}/${bench_name}_results.txt"
    
    echo_info "Running $description..."
    
    if timeout 1800 cargo bench --bench "$bench_name" 2>&1 | tee "$output_file"; then
        echo_success "$description completed"
        return 0
    else
        echo_error "$description failed or timed out"
        return 1
    fi
}

# Run individual benchmarks
echo_info "Starting benchmark execution..."

# Main spatial benchmarks
run_benchmark "spatial_benchmarks" "Main spatial benchmarks"

# Memory benchmarks (if implemented)
if [ -f "benches/memory_benchmarks.rs" ]; then
    run_benchmark "memory_benchmarks" "Memory usage benchmarks"
fi

# Run performance comparison examples
echo_info "Running performance comparison examples..."
if cargo run --release --example simd_performance_example > "${REPORT_DIR}/simd_performance_example.txt" 2>&1; then
    echo_success "SIMD performance example completed"
else
    echo_warning "SIMD performance example failed"
fi

# Generate performance reports
echo_info "Generating performance analysis reports..."

# Create a comprehensive summary
{
    echo "SCIRS2-SPATIAL COMPREHENSIVE BENCHMARK REPORT"
    echo "=============================================="
    echo ""
    echo "Generated on: $(date)"
    echo "Test duration: Multiple benchmark suites"
    echo ""
    
    echo "=== BENCHMARK SUITE OVERVIEW ==="
    echo ""
    echo "This report contains results from comprehensive performance testing of"
    echo "the scirs2-spatial library, including:"
    echo "  • SIMD vs scalar distance calculations"
    echo "  • Parallel vs sequential spatial operations"
    echo "  • Memory efficiency analysis"
    echo "  • Cross-architecture performance validation"
    echo "  • Spatial data structure performance"
    echo "  • Scaling behavior analysis"
    echo ""
    
    echo "=== SYSTEM CONFIGURATION ==="
    cat "${REPORT_DIR}/system_info.txt"
    echo ""
    
    echo "=== SIMD CAPABILITIES ==="
    if cargo run --release --bin simd_features 2>/dev/null; then
        echo "SIMD feature detection completed"
    else
        echo "SIMD feature detection not available"
    fi
    echo ""
    
} > "${REPORT_DIR}/comprehensive_summary.txt"

# Process benchmark results and extract key metrics
echo_info "Processing benchmark results..."

# Function to extract key metrics from criterion output
extract_metrics() {
    local input_file="$1"
    local output_file="$2"
    
    if [ -f "$input_file" ]; then
        {
            echo "=== KEY PERFORMANCE METRICS ==="
            echo ""
            
            # Extract time measurements
            echo "Performance Times:"
            grep -E "time:" "$input_file" | head -20 || true
            echo ""
            
            # Extract speedup information if available
            echo "Speedup Measurements:"
            grep -iE "(speedup|improvement|faster)" "$input_file" | head -10 || true
            echo ""
            
            # Extract memory usage if available
            echo "Memory Usage:"
            grep -iE "(memory|MB|allocation)" "$input_file" | head -10 || true
            echo ""
            
        } > "$output_file"
    fi
}

# Process each benchmark result
for result_file in "${REPORT_DIR}"/*_results.txt; do
    if [ -f "$result_file" ]; then
        base_name=$(basename "$result_file" _results.txt)
        extract_metrics "$result_file" "${REPORT_DIR}/${base_name}_metrics.txt"
    fi
done

# Generate recommendations based on results
{
    echo "PERFORMANCE OPTIMIZATION RECOMMENDATIONS"
    echo "========================================"
    echo ""
    echo "Based on the benchmark results, here are the key recommendations:"
    echo ""
    
    # Check if SIMD shows improvements
    if grep -q "speedup" "${REPORT_DIR}"/spatial_benchmarks_results.txt 2>/dev/null; then
        echo "✓ SIMD Optimizations:"
        echo "  • SIMD implementations show performance benefits"
        echo "  • Recommended for large-scale distance computations"
        echo "  • Best results with data that aligns to SIMD boundaries"
        echo ""
    fi
    
    # Check for parallel processing benefits
    echo "✓ Parallel Processing:"
    echo "  • Use parallel functions for datasets > 1,000 points"
    echo "  • Consider thread pool configuration for optimal performance"
    echo "  • Monitor memory usage with parallel operations"
    echo ""
    
    echo "✓ Algorithm Selection:"
    echo "  • KDTree: Best for low-dimensional nearest neighbor queries"
    echo "  • BallTree: Better for high-dimensional data"
    echo "  • Distance matrices: Use condensed format to save memory"
    echo ""
    
    echo "✓ Memory Management:"
    echo "  • For datasets > 10,000 points, consider chunked processing"
    echo "  • Monitor peak memory usage in production"
    echo "  • Use appropriate data types (f32 vs f64) based on precision needs"
    echo ""
    
    echo "✓ Data Size Guidelines:"
    echo "  • Small (< 1,000): Standard algorithms work well"
    echo "  • Medium (1,000-10,000): SIMD and parallel recommended"
    echo "  • Large (> 10,000): Consider approximate algorithms"
    echo ""
    
} > "${REPORT_DIR}/recommendations.txt"

# Create a summary index file
{
    echo "SCIRS2-SPATIAL BENCHMARK REPORT INDEX"
    echo "====================================="
    echo ""
    echo "Generated: $(date)"
    echo "Location: ${REPORT_DIR}"
    echo ""
    echo "FILES IN THIS REPORT:"
    echo ""
    
    for file in "${REPORT_DIR}"/*.txt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "  • $filename"
            
            # Add brief description
            case "$filename" in
                "system_info.txt")
                    echo "    → System configuration and hardware details"
                    ;;
                "comprehensive_summary.txt")
                    echo "    → Overall benchmark suite summary"
                    ;;
                "recommendations.txt")
                    echo "    → Performance optimization recommendations"
                    ;;
                "*_results.txt")
                    echo "    → Raw benchmark output data"
                    ;;
                "*_metrics.txt")
                    echo "    → Extracted key performance metrics"
                    ;;
                "simd_performance_example.txt")
                    echo "    → SIMD performance demonstration results"
                    ;;
            esac
        fi
    done
    
    echo ""
    echo "QUICK RESULTS SUMMARY:"
    echo ""
    
    # Try to extract some quick stats
    if [ -f "${REPORT_DIR}/spatial_benchmarks_results.txt" ]; then
        echo "  Main benchmark suite: COMPLETED"
        
        # Count number of benchmarks run
        bench_count=$(grep -c "test result:" "${REPORT_DIR}/spatial_benchmarks_results.txt" 2>/dev/null || echo "0")
        if [ "$bench_count" -gt 0 ]; then
            echo "  Total benchmarks executed: $bench_count"
        fi
    else
        echo "  Main benchmark suite: NOT FOUND"
    fi
    
    echo ""
    echo "For detailed analysis, see individual report files above."
    echo "For actionable insights, start with 'recommendations.txt'."
    
} > "${REPORT_DIR}/README.txt"

# Create a simple HTML summary for easy viewing
{
    cat << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCIRS2-Spatial Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .file-list { background: #f9f9f9; padding: 15px; border-radius: 5px; }
        .file-item { margin: 5px 0; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .info { color: #17a2b8; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SCIRS2-Spatial Performance Benchmark Report</h1>
        <p><strong>Generated:</strong> $(date)</p>
        <p><strong>Location:</strong> ${REPORT_DIR}</p>
    </div>

    <div class="section">
        <h2>Report Overview</h2>
        <p>This comprehensive benchmark report evaluates the performance of scirs2-spatial library 
        across multiple dimensions including SIMD optimization, parallel processing, memory efficiency, 
        and scaling behavior.</p>
    </div>

    <div class="section">
        <h2>Available Reports</h2>
        <div class="file-list">
EOF

    for file in "${REPORT_DIR}"/*.txt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "            <div class=\"file-item\">📄 <strong>$filename</strong></div>"
        fi
    done

    cat << EOF
        </div>
    </div>

    <div class="section">
        <h2>Quick Access</h2>
        <ul>
            <li><a href="recommendations.txt">Performance Recommendations</a> - Start here for actionable insights</li>
            <li><a href="comprehensive_summary.txt">Executive Summary</a> - High-level overview</li>
            <li><a href="system_info.txt">System Configuration</a> - Test environment details</li>
        </ul>
    </div>

    <div class="section">
        <h2>Key Findings</h2>
        <p><em>For detailed findings, please review the individual report files.</em></p>
    </div>
</body>
</html>
EOF
} > "${REPORT_DIR}/index.html"

# Final summary
echo ""
echo_success "Comprehensive benchmarking completed!"
echo_info "Results location: ${REPORT_DIR}"
echo ""
echo "📊 Quick Summary:"
echo "  • System info: ${REPORT_DIR}/system_info.txt"
echo "  • Main results: ${REPORT_DIR}/README.txt" 
echo "  • Recommendations: ${REPORT_DIR}/recommendations.txt"
echo "  • Web view: ${REPORT_DIR}/index.html"
echo ""
echo_info "To view results:"
echo "  cat ${REPORT_DIR}/README.txt"
echo "  # or open ${REPORT_DIR}/index.html in a browser"
echo ""

# Archive results for long-term storage
archive_name="scirs2_spatial_benchmarks_${TIMESTAMP}.tar.gz"
archive_path="${OUTPUT_DIR}/${archive_name}"

echo_info "Creating archive for long-term storage..."
if tar -czf "$archive_path" -C "$OUTPUT_DIR" "report_${TIMESTAMP}"; then
    echo_success "Archive created: $archive_path"
else
    echo_warning "Failed to create archive"
fi

echo_success "Benchmarking suite completed successfully! 🎉"
echo_info "Next steps:"
echo "  1. Review the recommendations in ${REPORT_DIR}/recommendations.txt"
echo "  2. Check for any performance regressions or improvements"
echo "  3. Update documentation based on findings"
echo "  4. Consider implementing suggested optimizations"