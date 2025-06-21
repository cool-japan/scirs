#!/usr/bin/env python3
"""
Comprehensive benchmark comparison between scirs2-integrate and SciPy.

This script runs both Rust and Python benchmarks, analyzes the results,
and generates detailed comparison reports with visualizations.
"""

import json
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

@dataclass
class BenchmarkComparison:
    """Container for benchmark comparison data."""
    name: str
    rust_time: float
    rust_std: float
    scipy_time: float
    scipy_std: float
    speedup: float
    rust_accuracy: Optional[float] = None
    scipy_accuracy: Optional[float] = None
    rust_extra: Optional[Dict] = None
    scipy_extra: Optional[Dict] = None

class BenchmarkRunner:
    """Manages running and comparing benchmarks."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.results_dir = project_dir / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def run_scipy_benchmarks(self, n_runs: int = 10) -> Dict:
        """Run SciPy reference benchmarks."""
        print("Running SciPy benchmarks...")
        
        script_path = self.project_dir / "benches" / "scipy_reference.py"
        output_file = self.results_dir / "scipy_results.json"
        
        cmd = [
            sys.executable, str(script_path),
            "--runs", str(n_runs),
            "--output", str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)
            if result.returncode != 0:
                print(f"SciPy benchmark failed: {result.stderr}")
                return {}
            
            with open(output_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error running SciPy benchmarks: {e}")
            return {}
    
    def run_rust_benchmarks(self) -> Dict:
        """Run Rust criterion benchmarks."""
        print("Running Rust benchmarks...")
        
        # Run criterion benchmark and capture JSON output
        cmd = [
            "cargo", "bench", "--bench", "scipy_comparison",
            "--", "--output-format", "json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)
            if result.returncode != 0:
                print(f"Rust benchmark failed: {result.stderr}")
                return {}
            
            # Parse criterion JSON output (this is simplified - real criterion output is more complex)
            # For now, we'll use a simplified approach
            return self._parse_criterion_output(result.stdout)
            
        except Exception as e:
            print(f"Error running Rust benchmarks: {e}")
            return {}
    
    def _parse_criterion_output(self, output: str) -> Dict:
        """Parse criterion output to extract timing data."""
        # This is a simplified parser - in practice, criterion output is complex
        # For now, return mock data that would come from criterion
        return {}
    
    def run_simplified_rust_benchmarks(self) -> Dict:
        """Run simplified Rust benchmarks using a custom timing approach."""
        print("Running simplified Rust benchmarks...")
        
        # Create a simple timing benchmark script
        timing_script = """
use std::time::Instant;
use ndarray::Array1;
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use scirs2_integrate::quad::{quad, QuadOptions};

fn time_function<F, R>(f: F, name: &str) -> (f64, R) 
where 
    F: Fn() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("{}: {:.6} seconds", name, duration.as_secs_f64());
    (duration.as_secs_f64(), result)
}

fn main() {
    // ODE benchmarks
    let exponential_decay = |t: f64, y: ndarray::ArrayView1<f64>| Array1::from_vec(vec![-y[0]]);
    let harmonic_oscillator = |t: f64, y: ndarray::ArrayView1<f64>| Array1::from_vec(vec![y[1], -y[0]]);
    
    // Exponential decay with RK45
    let (time, result) = time_function(|| {
        let y0 = Array1::from_vec(vec![1.0]);
        let opts = ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-6,
            atol: 1e-9,
            ..Default::default()
        };
        solve_ivp(exponential_decay, [0.0, 1.0], y0, Some(opts))
    }, "ode_exponential_decay_RK45");
    
    // Harmonic oscillator with DOP853
    let (time, result) = time_function(|| {
        let y0 = Array1::from_vec(vec![1.0, 0.0]);
        let opts = ODEOptions {
            method: ODEMethod::DOP853,
            rtol: 1e-6,
            atol: 1e-9,
            ..Default::default()
        };
        solve_ivp(harmonic_oscillator, [0.0, 10.0], y0, Some(opts))
    }, "ode_harmonic_oscillator_DOP853");
    
    // Quadrature benchmarks
    let polynomial_cubic = |x: f64| x * x * x;
    let oscillatory = |x: f64| (10.0 * x).sin();
    
    let (time, result) = time_function(|| {
        let opts = QuadOptions {
            epsabs: 1e-10,
            epsrel: 1e-10,
            limit: 1000,
            ..Default::default()
        };
        quad(polynomial_cubic, 0.0, 1.0, Some(opts))
    }, "quad_polynomial_cubic");
    
    let (time, result) = time_function(|| {
        let opts = QuadOptions {
            epsabs: 1e-10,
            epsrel: 1e-10,
            limit: 1000,
            ..Default::default()
        };
        quad(oscillatory, 0.0, 1.0, Some(opts))
    }, "quad_oscillatory");
}
"""
        
        # Write timing script to temporary file
        timing_file = self.project_dir / "benchmark_timing.rs"
        with open(timing_file, 'w') as f:
            f.write(timing_script)
        
        try:
            # Compile and run the timing script
            cmd = ["rustc", "--edition", "2021", "-L", "target/debug/deps", 
                   str(timing_file), "-o", "benchmark_timing"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)
            
            if result.returncode == 0:
                # Run the compiled benchmark
                result = subprocess.run(["./benchmark_timing"], capture_output=True, text=True, cwd=self.project_dir)
                return self._parse_timing_output(result.stdout)
            else:
                print(f"Compilation failed: {result.stderr}")
                return {}
                
        except Exception as e:
            print(f"Error running simplified benchmarks: {e}")
            return {}
        finally:
            # Clean up
            for file in [timing_file, self.project_dir / "benchmark_timing"]:
                if file.exists():
                    file.unlink()
    
    def _parse_timing_output(self, output: str) -> Dict:
        """Parse timing output into structured data."""
        results = {}
        lines = output.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                name, time_str = line.split(': ')
                time_val = float(time_str.split()[0])
                results[name] = {
                    'mean_time': time_val,
                    'std_time': 0.0,  # Single run, no std
                    'extra_info': {}
                }
        
        return results
    
    def compare_results(self, rust_results: Dict, scipy_results: List[Dict]) -> List[BenchmarkComparison]:
        """Compare Rust and SciPy benchmark results."""
        comparisons = []
        
        # Convert scipy results to dict for easier lookup
        scipy_dict = {result['name']: result for result in scipy_results}
        
        # Find matching benchmarks
        for rust_name, rust_data in rust_results.items():
            # Try to find matching scipy benchmark
            scipy_data = None
            
            # Simple name matching (would need more sophisticated matching in practice)
            for scipy_name, scipy_result in scipy_dict.items():
                if self._benchmarks_match(rust_name, scipy_name):
                    scipy_data = scipy_result
                    break
            
            if scipy_data:
                speedup = scipy_data['mean_time'] / rust_data['mean_time']
                
                comparison = BenchmarkComparison(
                    name=rust_name,
                    rust_time=rust_data['mean_time'],
                    rust_std=rust_data.get('std_time', 0.0),
                    scipy_time=scipy_data['mean_time'],
                    scipy_std=scipy_data['std_time'],
                    speedup=speedup,
                    rust_accuracy=rust_data.get('accuracy'),
                    scipy_accuracy=scipy_data.get('accuracy'),
                    rust_extra=rust_data.get('extra_info'),
                    scipy_extra=scipy_data.get('extra_info')
                )
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _benchmarks_match(self, rust_name: str, scipy_name: str) -> bool:
        """Check if rust and scipy benchmark names refer to the same test."""
        # Simplified matching logic
        rust_parts = rust_name.lower().replace('_', ' ').split()
        scipy_parts = scipy_name.lower().replace('_', ' ').split()
        
        # Check for common terms
        common_terms = set(rust_parts) & set(scipy_parts)
        return len(common_terms) >= 2
    
    def generate_report(self, comparisons: List[BenchmarkComparison]) -> str:
        """Generate a comprehensive comparison report."""
        report = []
        report.append("# scirs2-integrate vs SciPy Performance Comparison")
        report.append("=" * 60)
        report.append("")
        
        if not comparisons:
            report.append("No matching benchmarks found for comparison.")
            return "\n".join(report)
        
        # Summary statistics
        speedups = [c.speedup for c in comparisons if c.speedup != float('inf')]
        if speedups:
            report.append("## Summary Statistics")
            report.append(f"- Number of benchmarks: {len(comparisons)}")
            report.append(f"- Average speedup: {np.mean(speedups):.2f}x")
            report.append(f"- Median speedup: {np.median(speedups):.2f}x")
            report.append(f"- Best speedup: {np.max(speedups):.2f}x")
            report.append(f"- Worst speedup: {np.min(speedups):.2f}x")
            report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        report.append(f"{'Benchmark':<30} {'Rust (ms)':<12} {'SciPy (ms)':<12} {'Speedup':<10} {'Accuracy':<15}")
        report.append("-" * 85)
        
        for comp in sorted(comparisons, key=lambda x: x.speedup, reverse=True):
            rust_ms = comp.rust_time * 1000
            scipy_ms = comp.scipy_time * 1000
            speedup_str = f"{comp.speedup:.2f}x" if comp.speedup != float('inf') else "∞"
            
            accuracy_str = "N/A"
            if comp.rust_accuracy is not None and comp.scipy_accuracy is not None:
                if comp.rust_accuracy <= comp.scipy_accuracy:
                    accuracy_str = "✓ Better"
                else:
                    accuracy_str = "✗ Worse"
            
            report.append(f"{comp.name:<30} {rust_ms:<12.3f} {scipy_ms:<12.3f} {speedup_str:<10} {accuracy_str:<15}")
        
        report.append("")
        report.append("## Performance Categories")
        
        # Categorize results
        much_faster = [c for c in comparisons if c.speedup > 2.0]
        faster = [c for c in comparisons if 1.2 <= c.speedup <= 2.0]
        similar = [c for c in comparisons if 0.8 <= c.speedup < 1.2]
        slower = [c for c in comparisons if c.speedup < 0.8]
        
        report.append(f"- Much faster (>2x): {len(much_faster)} benchmarks")
        report.append(f"- Faster (1.2-2x): {len(faster)} benchmarks")
        report.append(f"- Similar (0.8-1.2x): {len(similar)} benchmarks")
        report.append(f"- Slower (<0.8x): {len(slower)} benchmarks")
        
        if much_faster:
            report.append(f"  Best performers: {', '.join([c.name for c in much_faster[:5]])}")
        
        if slower:
            report.append(f"  Needs improvement: {', '.join([c.name for c in slower[:5]])}")
        
        return "\n".join(report)
    
    def save_comparison_data(self, comparisons: List[BenchmarkComparison], filename: str):
        """Save comparison data to JSON file."""
        data = []
        for comp in comparisons:
            data.append({
                'name': comp.name,
                'rust_time': comp.rust_time,
                'rust_std': comp.rust_std,
                'scipy_time': comp.scipy_time,
                'scipy_std': comp.scipy_std,
                'speedup': comp.speedup,
                'rust_accuracy': comp.rust_accuracy,
                'scipy_accuracy': comp.scipy_accuracy,
                'rust_extra': comp.rust_extra,
                'scipy_extra': comp.scipy_extra,
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_visualizations(self, comparisons: List[BenchmarkComparison]):
        """Create visualization plots for the comparison."""
        if not comparisons:
            return
        
        # Speedup comparison
        plt.figure(figsize=(12, 8))
        
        names = [c.name for c in comparisons]
        speedups = [c.speedup for c in comparisons]
        
        # Truncate long names
        short_names = [name[:20] + "..." if len(name) > 20 else name for name in names]
        
        colors = ['green' if s > 1 else 'red' for s in speedups]
        
        plt.barh(range(len(speedups)), speedups, color=colors, alpha=0.7)
        plt.yticks(range(len(speedups)), short_names)
        plt.xlabel('Speedup (Rust vs SciPy)')
        plt.title('Performance Comparison: scirs2-integrate vs SciPy')
        plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Timing comparison scatter plot
        if len(comparisons) > 1:
            plt.figure(figsize=(10, 8))
            
            rust_times = [c.rust_time * 1000 for c in comparisons]  # Convert to ms
            scipy_times = [c.scipy_time * 1000 for c in comparisons]
            
            plt.scatter(scipy_times, rust_times, alpha=0.7, s=50)
            
            # Add diagonal line (equal performance)
            max_time = max(max(rust_times), max(scipy_times))
            plt.plot([0, max_time], [0, max_time], 'k--', alpha=0.5, label='Equal performance')
            
            plt.xlabel('SciPy Time (ms)')
            plt.ylabel('Rust Time (ms)')
            plt.title('Timing Comparison: Rust vs SciPy')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Log scale if there's a wide range
            if max_time / min(min(rust_times), min(scipy_times)) > 100:
                plt.xscale('log')
                plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'timing_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Run the complete benchmark comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark comparison between Rust and SciPy")
    parser.add_argument("--scipy-runs", type=int, default=10, help="Number of SciPy benchmark runs")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust benchmarks")
    parser.add_argument("--skip-scipy", action="store_true", help="Skip SciPy benchmarks")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    args = parser.parse_args()
    
    # Determine project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    if args.output_dir:
        project_dir = Path(args.output_dir)
    
    runner = BenchmarkRunner(project_dir)
    
    print("Starting comprehensive benchmark comparison...")
    print(f"Project directory: {project_dir}")
    print("=" * 60)
    
    # Run benchmarks
    rust_results = {}
    scipy_results = []
    
    if not args.skip_scipy:
        scipy_results = runner.run_scipy_benchmarks(args.scipy_runs)
    
    if not args.skip_rust:
        # Try criterion first, fall back to simplified
        rust_results = runner.run_rust_benchmarks()
        if not rust_results:
            print("Criterion benchmarks failed, trying simplified approach...")
            rust_results = runner.run_simplified_rust_benchmarks()
    
    # Compare results
    if rust_results and scipy_results:
        comparisons = runner.compare_results(rust_results, scipy_results)
        
        # Generate report
        report = runner.generate_report(comparisons)
        
        # Save results
        report_file = runner.results_dir / "benchmark_comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        data_file = runner.results_dir / "comparison_data.json"
        runner.save_comparison_data(comparisons, data_file)
        
        # Create visualizations
        try:
            runner.create_visualizations(comparisons)
            print(f"Visualizations saved to {runner.results_dir}")
        except ImportError:
            print("Matplotlib not available, skipping visualizations")
        
        # Print report
        print(report)
        print(f"\nDetailed results saved to {report_file}")
        print(f"Raw data saved to {data_file}")
        
    else:
        print("No benchmark results to compare.")
        if not rust_results:
            print("- Rust benchmarks failed")
        if not scipy_results:
            print("- SciPy benchmarks failed")

if __name__ == "__main__":
    main()