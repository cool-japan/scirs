#!/bin/bash

# Quick Benchmark Runner for scirs2-spatial
# This script provides a fast way to run specific benchmarks

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

show_help() {
    echo "Quick Benchmark Runner for scirs2-spatial"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -s, --simd         Run SIMD vs scalar benchmarks"
    echo "  -p, --parallel     Run parallel vs sequential benchmarks"
    echo "  -m, --memory       Run memory efficiency benchmarks"
    echo "  -k, --knn          Run k-nearest neighbors benchmarks"
    echo "  -d, --distance     Run distance metrics comparison"
    echo "  -a, --all          Run all benchmark categories"
    echo "  -q, --quick        Run quick validation benchmarks"
    echo "  --example          Run SIMD performance example"
    echo "  --features         Show available SIMD features"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --simd          # Test SIMD performance"
    echo "  $0 --quick         # Quick validation"
    echo "  $0 --all           # Full benchmark suite"
}

# Quick validation function
run_quick_validation() {
    echo_info "Running quick validation benchmarks..."
    
    # Build first
    cargo build --release --quiet
    
    # Run a subset of tests
    echo_info "Testing basic functionality..."
    cargo test --release --quiet euclidean
    cargo test --release --quiet kdtree
    cargo test --release --quiet simd
    
    echo_success "Quick validation completed"
}

# SIMD feature detection
show_simd_features() {
    echo_info "Detecting SIMD features..."
    
    # Build and run a simple feature detection
    cat > /tmp/simd_detect.rs << 'EOF'
fn main() {
    println!("SIMD Feature Detection:");
    
    #[cfg(target_arch = "x86_64")]
    {
        println!("  Architecture: x86_64");
        println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
        println!("  AVX: {}", is_x86_feature_detected!("avx"));
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        println!("  Architecture: aarch64");
        println!("  NEON: {}", std::arch::is_aarch64_feature_detected!("neon"));
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("  Architecture: Other (scalar fallbacks)");
    }
}
EOF

    rustc /tmp/simd_detect.rs -o /tmp/simd_detect && /tmp/simd_detect
    rm -f /tmp/simd_detect.rs /tmp/simd_detect
}

# Run specific benchmark categories
run_simd_benchmarks() {
    echo_info "Running SIMD vs scalar benchmarks..."
    cargo bench --bench spatial_benchmarks simd_vs_scalar
}

run_parallel_benchmarks() {
    echo_info "Running parallel vs sequential benchmarks..."
    cargo bench --bench spatial_benchmarks parallel_vs_sequential
}

run_memory_benchmarks() {
    echo_info "Running memory efficiency benchmarks..."
    if [ -f "benches/memory_benchmarks.rs" ]; then
        cargo bench --bench memory_benchmarks
    else
        cargo bench --bench spatial_benchmarks memory_efficiency
    fi
}

run_knn_benchmarks() {
    echo_info "Running k-nearest neighbors benchmarks..."
    cargo bench --bench spatial_benchmarks knn_performance
}

run_distance_benchmarks() {
    echo_info "Running distance metrics comparison..."
    cargo bench --bench spatial_benchmarks distance_metrics
}

run_example() {
    echo_info "Running SIMD performance example..."
    cargo run --release --example simd_performance_example
}

# Parse command line arguments
case "${1:-}" in
    -s|--simd)
        run_simd_benchmarks
        ;;
    -p|--parallel)
        run_parallel_benchmarks
        ;;
    -m|--memory)
        run_memory_benchmarks
        ;;
    -k|--knn)
        run_knn_benchmarks
        ;;
    -d|--distance)
        run_distance_benchmarks
        ;;
    -a|--all)
        echo_info "Running all benchmark categories..."
        cargo bench --bench spatial_benchmarks
        ;;
    -q|--quick)
        run_quick_validation
        ;;
    --example)
        run_example
        ;;
    --features)
        show_simd_features
        ;;
    -h|--help|"")
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac