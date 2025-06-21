#!/bin/bash

# Run Acceleration Benchmarks Script
# This script runs comprehensive benchmarks for all acceleration features

echo "🚀 Running SciRS2-FFT Acceleration Benchmarks"
echo "=============================================="

# Check if criterion is available
if ! cargo bench --list 2>/dev/null | grep -q acceleration_benchmarks; then
    echo "❌ Benchmark dependencies not available"
    echo "💡 Install criterion with: cargo install criterion"
    exit 1
fi

echo "📊 Running CPU Sparse FFT benchmarks..."
cargo bench --bench acceleration_benchmarks -- cpu_sparse_fft --noplot

echo "🎮 Running GPU Sparse FFT benchmarks..."
cargo bench --bench acceleration_benchmarks -- gpu_sparse_fft --noplot

echo "🔄 Running Multi-GPU benchmarks..."
cargo bench --bench acceleration_benchmarks -- multi_gpu_sparse_fft --noplot

echo "⚡ Running Specialized Hardware benchmarks..."
cargo bench --bench acceleration_benchmarks -- specialized_hardware --noplot

echo "📈 Running Sparsity Scaling benchmarks..."
cargo bench --bench acceleration_benchmarks -- sparsity_scaling --noplot

echo "🔍 Running Algorithm Comparison benchmarks..."
cargo bench --bench acceleration_benchmarks -- algorithm_comparison --noplot

echo "🧠 Running Memory Efficiency benchmarks..."
cargo bench --bench acceleration_benchmarks -- memory_efficiency --noplot

echo "✅ All acceleration benchmarks completed!"
echo "📁 Results saved to: target/criterion/"
echo ""
echo "💡 To view detailed results:"
echo "   open target/criterion/index.html"
echo ""
echo "🚀 To run all benchmarks with plots:"
echo "   cargo bench --bench acceleration_benchmarks"