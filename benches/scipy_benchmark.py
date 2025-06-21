#!/usr/bin/env python3
"""
SciPy benchmark script for comparison with SciRS2 performance.
This script runs equivalent operations to the Rust benchmarks and saves results
for comparative analysis.
"""

import json
import time
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import psutil
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Constants matching Rust benchmarks
SEED = 42
COMPARISON_SIZES = [50, 100, 200, 500]

class BenchmarkResult:
    def __init__(self, operation: str, size: int, python_time_ns: int, 
                 memory_usage_mb: Optional[float] = None):
        self.operation = operation
        self.size = size
        self.python_time_ns = python_time_ns
        self.memory_usage_mb = memory_usage_mb

def generate_test_data(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test data matching Rust implementation."""
    np.random.seed(SEED)
    matrix = np.random.uniform(-1.0, 1.0, (size, size))
    vector = np.random.uniform(-1.0, 1.0, size)
    return matrix, vector

def generate_spd_matrix(size: int) -> np.ndarray:
    """Generate symmetric positive definite matrix."""
    np.random.seed(SEED)
    a = np.random.uniform(-1.0, 1.0, (size, size))
    return a.T @ a + np.eye(size) * 0.1

def measure_memory_usage(func, *args):
    """Measure peak memory usage of a function."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = peak_memory - initial_memory
    
    return result, memory_used

def benchmark_basic_operations() -> List[BenchmarkResult]:
    """Benchmark basic linear algebra operations."""
    results = []
    
    print("Benchmarking basic operations...")
    for size in COMPARISON_SIZES:
        matrix, vector = generate_test_data(size)
        
        # Matrix determinant
        start = time.perf_counter_ns()
        det_result = scipy.linalg.det(matrix)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "determinant", size, end - start
        ))
        print(f"  Determinant {size}x{size}: {(end - start) / 1e6:.2f} ms")
        
        # Matrix inverse (for smaller matrices)
        if size <= 200:
            start = time.perf_counter_ns()
            inv_result = scipy.linalg.inv(matrix)
            end = time.perf_counter_ns()
            
            results.append(BenchmarkResult(
                "inverse", size, end - start
            ))
            print(f"  Inverse {size}x{size}: {(end - start) / 1e6:.2f} ms")
        
        # Frobenius norm
        start = time.perf_counter_ns()
        norm_result = scipy.linalg.norm(matrix, 'fro')
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "frobenius_norm", size, end - start
        ))
        print(f"  Frobenius norm {size}x{size}: {(end - start) / 1e6:.2f} ms")
    
    return results

def benchmark_decompositions() -> List[BenchmarkResult]:
    """Benchmark matrix decompositions."""
    results = []
    
    print("Benchmarking decompositions...")
    for size in COMPARISON_SIZES:
        matrix, vector = generate_test_data(size)
        spd_matrix = generate_spd_matrix(size)
        
        # LU decomposition
        start = time.perf_counter_ns()
        p, l, u = scipy.linalg.lu(matrix)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "lu_decomposition", size, end - start
        ))
        print(f"  LU decomposition {size}x{size}: {(end - start) / 1e6:.2f} ms")
        
        # QR decomposition
        start = time.perf_counter_ns()
        q, r = scipy.linalg.qr(matrix)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "qr_decomposition", size, end - start
        ))
        print(f"  QR decomposition {size}x{size}: {(end - start) / 1e6:.2f} ms")
        
        # Cholesky decomposition
        start = time.perf_counter_ns()
        l = scipy.linalg.cholesky(spd_matrix, lower=True)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "cholesky_decomposition", size, end - start
        ))
        print(f"  Cholesky decomposition {size}x{size}: {(end - start) / 1e6:.2f} ms")
    
    return results

def benchmark_solvers() -> List[BenchmarkResult]:
    """Benchmark linear system solvers."""
    results = []
    
    print("Benchmarking solvers...")
    for size in COMPARISON_SIZES:
        matrix, vector = generate_test_data(size)
        
        # General linear solve
        start = time.perf_counter_ns()
        solution = scipy.linalg.solve(matrix, vector)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "linear_solve", size, end - start
        ))
        print(f"  Linear solve {size}x{size}: {(end - start) / 1e6:.2f} ms")
        
        # Least squares solve
        start = time.perf_counter_ns()
        solution, residuals, rank, s = scipy.linalg.lstsq(matrix, vector)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "least_squares", size, end - start
        ))
        print(f"  Least squares {size}x{size}: {(end - start) / 1e6:.2f} ms")
    
    return results

def benchmark_eigenvalues() -> List[BenchmarkResult]:
    """Benchmark eigenvalue computations."""
    results = []
    
    print("Benchmarking eigenvalues...")
    # Limit to smaller sizes due to computational cost
    eigen_sizes = [50, 100]
    
    for size in eigen_sizes:
        spd_matrix = generate_spd_matrix(size)
        
        # Symmetric eigenvalues only
        start = time.perf_counter_ns()
        eigenvals = scipy.linalg.eigvalsh(spd_matrix)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "symmetric_eigenvalues", size, end - start
        ))
        print(f"  Symmetric eigenvalues {size}x{size}: {(end - start) / 1e6:.2f} ms")
        
        # Symmetric eigenvalues and eigenvectors
        start = time.perf_counter_ns()
        eigenvals, eigenvecs = scipy.linalg.eigh(spd_matrix)
        end = time.perf_counter_ns()
        
        results.append(BenchmarkResult(
            "symmetric_eigenvectors", size, end - start
        ))
        print(f"  Symmetric eigenvectors {size}x{size}: {(end - start) / 1e6:.2f} ms")
    
    return results

def analyze_numerical_stability() -> List[BenchmarkResult]:
    """Analyze numerical stability with ill-conditioned matrices."""
    results = []
    
    print("Analyzing numerical stability...")
    test_sizes = [50, 100]
    condition_numbers = [1e3, 1e6, 1e12]
    
    for size in test_sizes:
        for cond in condition_numbers:
            # Generate ill-conditioned matrix
            u, s, vt = scipy.linalg.svd(np.random.randn(size, size))
            s = np.linspace(1.0, 1.0/cond, size)
            matrix = u @ np.diag(s) @ vt
            
            vector = np.ones(size)
            
            # Test solve with ill-conditioned matrix
            try:
                start = time.perf_counter_ns()
                solution = scipy.linalg.solve(matrix, vector)
                end = time.perf_counter_ns()
                
                results.append(BenchmarkResult(
                    f"solve_cond_{cond:.0e}", size, end - start
                ))
                print(f"  Solve cond {cond:.0e} {size}x{size}: {(end - start) / 1e6:.2f} ms")
            except Exception as e:
                print(f"  Solve failed for cond {cond:.0e} {size}x{size}: {e}")
            
            # Test lstsq with ill-conditioned matrix
            try:
                start = time.perf_counter_ns()
                solution, residuals, rank, s_vals = scipy.linalg.lstsq(matrix, vector)
                end = time.perf_counter_ns()
                
                results.append(BenchmarkResult(
                    f"lstsq_cond_{cond:.0e}", size, end - start
                ))
                print(f"  Lstsq cond {cond:.0e} {size}x{size}: {(end - start) / 1e6:.2f} ms")
            except Exception as e:
                print(f"  Lstsq failed for cond {cond:.0e} {size}x{size}: {e}")
    
    return results

def create_comparison_report(python_results: List[BenchmarkResult]) -> Dict:
    """Create comprehensive comparison report."""
    
    # Try to load Rust results
    rust_results = {}
    try:
        with open('target/rust_benchmark_results.json', 'r') as f:
            rust_data = json.load(f)
            for result in rust_data:
                key = f"{result['operation']}_{result['size']}"
                rust_results[key] = result['rust_time_ns']
    except FileNotFoundError:
        print("Warning: Rust benchmark results not found. Run Rust benchmarks first.")
    
    # Merge results and calculate speedups
    comparison_results = []
    rust_faster_count = 0
    python_faster_count = 0
    speedups = []
    
    for py_result in python_results:
        key = f"{py_result.operation}_{py_result.size}"
        rust_time = rust_results.get(key)
        
        result_dict = {
            "operation": py_result.operation,
            "size": py_result.size,
            "rust_time_ns": rust_time,
            "python_time_ns": py_result.python_time_ns,
            "speedup": None,
            "memory_usage_mb": py_result.memory_usage_mb
        }
        
        if rust_time is not None:
            speedup = py_result.python_time_ns / rust_time
            result_dict["speedup"] = speedup
            speedups.append(speedup)
            
            if speedup > 1.0:
                rust_faster_count += 1
            else:
                python_faster_count += 1
        
        comparison_results.append(result_dict)
    
    # Calculate summary statistics
    summary = {
        "total_operations": len(comparison_results),
        "rust_faster_count": rust_faster_count,
        "python_faster_count": python_faster_count,
        "average_speedup": np.mean(speedups) if speedups else 0.0,
        "max_speedup": np.max(speedups) if speedups else 0.0,
        "min_speedup": np.min(speedups) if speedups else 0.0,
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "results": comparison_results,
        "summary": summary
    }

def run_memory_analysis():
    """Run detailed memory usage analysis."""
    print("Running memory analysis...")
    
    memory_results = []
    for size in [100, 200, 500]:
        matrix, vector = generate_test_data(size)
        
        # Memory usage for matrix operations
        _, memory_used = measure_memory_usage(scipy.linalg.det, matrix)
        memory_results.append({
            "operation": "determinant",
            "size": size,
            "memory_mb": memory_used
        })
        
        if size <= 200:
            _, memory_used = measure_memory_usage(scipy.linalg.inv, matrix)
            memory_results.append({
                "operation": "inverse", 
                "size": size,
                "memory_mb": memory_used
            })
    
    # Save memory analysis
    with open('target/python_memory_analysis.json', 'w') as f:
        json.dump(memory_results, f, indent=2)
    
    print("Memory analysis complete.")

def main():
    """Main benchmark execution function."""
    print("=== SciPy Performance Benchmarks ===")
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {scipy.__version__}")
    
    # Ensure target directory exists
    os.makedirs('target', exist_ok=True)
    
    # Run all benchmarks
    all_results = []
    all_results.extend(benchmark_basic_operations())
    all_results.extend(benchmark_decompositions())
    all_results.extend(benchmark_solvers())
    all_results.extend(benchmark_eigenvalues())
    all_results.extend(analyze_numerical_stability())
    
    # Run memory analysis
    run_memory_analysis()
    
    # Create comparison report
    report = create_comparison_report(all_results)
    
    # Save results
    with open('target/python_benchmark_results.json', 'w') as f:
        json.dump([{
            "operation": r.operation,
            "size": r.size,
            "python_time_ns": r.python_time_ns,
            "memory_usage_mb": r.memory_usage_mb
        } for r in all_results], f, indent=2)
    
    with open('target/benchmark_comparison.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Total operations benchmarked: {len(all_results)}")
    if report['summary']['total_operations'] > 0:
        print(f"Rust vs Python comparison available")
        print(f"Average speedup: {report['summary']['average_speedup']:.2f}x")
    else:
        print("Run Rust benchmarks first for comparison")
    
    print("\nResults saved to target/ directory")
    print("Run 'cargo bench' to compare with Rust performance")

if __name__ == "__main__":
    main()