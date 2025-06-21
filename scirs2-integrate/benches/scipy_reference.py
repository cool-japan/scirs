#!/usr/bin/env python3
"""
Reference SciPy benchmarks for comparison with scirs2-integrate.

This script runs the same problems as the Rust benchmarks using SciPy,
providing baseline performance measurements for comparison.
"""

import numpy as np
import scipy.integrate as integrate
import time
import json
import sys
from typing import Dict, List, Tuple, Callable, Any

class BenchmarkResult:
    """Container for benchmark timing results."""
    
    def __init__(self, name: str, mean_time: float, std_time: float, 
                 accuracy: float = None, extra_info: Dict[str, Any] = None):
        self.name = name
        self.mean_time = mean_time
        self.std_time = std_time
        self.accuracy = accuracy
        self.extra_info = extra_info or {}

class ScipyBenchmarks:
    """SciPy benchmark suite matching the Rust implementation."""
    
    def __init__(self, n_runs: int = 10):
        self.n_runs = n_runs
        self.results: List[BenchmarkResult] = []
    
    def time_function(self, func: Callable, *args, **kwargs) -> Tuple[float, float, Any]:
        """Time a function call multiple times and return mean, std, and result."""
        times = []
        result = None
        
        for _ in range(self.n_runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return np.mean(times), np.std(times), result
    
    # ODE Problems
    @staticmethod
    def exponential_decay(t, y):
        """dy/dt = -y, y(0) = 1"""
        return -y
    
    @staticmethod
    def harmonic_oscillator(t, y):
        """d²x/dt² + x = 0, as first order system"""
        return [y[1], -y[0]]
    
    @staticmethod
    def van_der_pol(mu):
        """Van der Pol oscillator with parameter mu"""
        def rhs(t, y):
            return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
        return rhs
    
    @staticmethod
    def lotka_volterra(a, b, c, d):
        """Lotka-Volterra predator-prey model"""
        def rhs(t, y):
            x, y_val = y
            return [a * x - b * x * y_val, -c * y_val + d * x * y_val]
        return rhs
    
    @staticmethod
    def three_body_problem(t, y):
        """Simplified 3-body gravitational problem"""
        dydt = np.zeros_like(y)
        
        # Positions and velocities
        for i in range(3):
            dydt[i * 2] = y[6 + i * 2]
            dydt[i * 2 + 1] = y[6 + i * 2 + 1]
        
        # Gravitational forces
        m = [1.0, 1.0, 1.0]
        for i in range(3):
            fx, fy = 0.0, 0.0
            for j in range(3):
                if i != j:
                    dx = y[j * 2] - y[i * 2]
                    dy = y[j * 2 + 1] - y[i * 2 + 1]
                    r = np.sqrt(dx**2 + dy**2)
                    r3 = r**3 + 1e-10  # softening
                    
                    fx += m[j] * dx / r3
                    fy += m[j] * dy / r3
            
            dydt[6 + i * 2] = fx
            dydt[6 + i * 2 + 1] = fy
        
        return dydt
    
    # Quadrature Problems
    @staticmethod
    def polynomial_cubic(x):
        """f(x) = x^3"""
        return x**3
    
    @staticmethod
    def oscillatory(x):
        """f(x) = sin(10*x)"""
        return np.sin(10 * x)
    
    @staticmethod
    def gaussian(x):
        """f(x) = exp(-x^2)"""
        return np.exp(-x**2)
    
    @staticmethod
    def nearly_singular(x):
        """f(x) = 1/sqrt(x)"""
        return 1.0 / np.sqrt(np.maximum(x, 1e-10))
    
    @staticmethod
    def multivariate_gaussian(x):
        """f(x) = exp(-||x||^2) for multidimensional x"""
        if np.isscalar(x):
            return np.exp(-x**2)
        return np.exp(-np.sum(x**2))
    
    def bench_ode_solvers(self):
        """Benchmark ODE solvers."""
        print("Benchmarking ODE solvers...")
        
        problems = [
            ("exponential_decay", self.exponential_decay, [0.0, 1.0], [1.0]),
            ("harmonic_oscillator", self.harmonic_oscillator, [0.0, 10.0], [1.0, 0.0]),
            ("van_der_pol_mild", self.van_der_pol(1.0), [0.0, 10.0], [1.0, 0.0]),
            ("van_der_pol_stiff", self.van_der_pol(100.0), [0.0, 10.0], [1.0, 0.0]),
            ("lotka_volterra", self.lotka_volterra(1.5, 1.0, 3.0, 1.0), [0.0, 15.0], [10.0, 5.0]),
            ("three_body", self.three_body_problem, [0.0, 5.0], 
             [1.0, 0.0, -0.5, 0.866, -0.5, -0.866, 0.0, 0.5, -0.433, -0.25, 0.433, -0.25]),
        ]
        
        # SciPy methods mapping to our methods
        methods = [
            ("RK45", "RK45"),
            ("DOP853", "DOP853"), 
            ("BDF", "BDF"),
            ("Radau", "Radau"),
            ("LSODA", "LSODA"),
        ]
        
        for prob_name, func, t_span, y0 in problems:
            for method_name, scipy_method in methods:
                benchmark_name = f"ode_{prob_name}_{method_name}"
                print(f"  Running {benchmark_name}...")
                
                try:
                    mean_time, std_time, result = self.time_function(
                        integrate.solve_ivp,
                        func, t_span, y0,
                        method=scipy_method,
                        rtol=1e-6, atol=1e-9
                    )
                    
                    accuracy = None
                    if result.success:
                        # Calculate some measure of accuracy for known problems
                        if prob_name == "exponential_decay":
                            # Exact solution: y = exp(-t)
                            t_final = result.t[-1]
                            exact = np.exp(-t_final)
                            accuracy = abs(result.y[0][-1] - exact)
                        elif prob_name == "harmonic_oscillator":
                            # Energy conservation: E = 0.5 * (x^2 + v^2)
                            energy = 0.5 * (result.y[0][-1]**2 + result.y[1][-1]**2)
                            accuracy = abs(energy - 0.5)  # Initial energy = 0.5
                    
                    extra_info = {
                        'nfev': result.nfev,
                        'njev': result.njev if hasattr(result, 'njev') else 0,
                        'nlu': result.nlu if hasattr(result, 'nlu') else 0,
                        'success': result.success,
                        'message': result.message,
                        'final_time': result.t[-1] if len(result.t) > 0 else 0,
                        'n_steps': len(result.t) - 1,
                    }
                    
                    self.results.append(BenchmarkResult(
                        benchmark_name, mean_time, std_time, accuracy, extra_info
                    ))
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    self.results.append(BenchmarkResult(
                        benchmark_name, float('inf'), 0.0, None, {'error': str(e)}
                    ))
    
    def bench_quadrature_methods(self):
        """Benchmark quadrature methods."""
        print("Benchmarking quadrature methods...")
        
        problems = [
            ("polynomial_cubic", self.polynomial_cubic, 0.0, 1.0, 0.25),  # exact = 1/4
            ("oscillatory", self.oscillatory, 0.0, 1.0, None),
            ("gaussian", self.gaussian, -3.0, 3.0, None),
            ("nearly_singular", self.nearly_singular, 1e-6, 1.0, 2.0 * (1.0 - (1e-6)**0.5)),  # exact
        ]
        
        for prob_name, func, a, b, exact in problems:
            benchmark_name = f"quad_{prob_name}"
            print(f"  Running {benchmark_name}...")
            
            try:
                mean_time, std_time, result = self.time_function(
                    integrate.quad,
                    func, a, b,
                    epsabs=1e-10, epsrel=1e-10, limit=1000
                )
                
                integral_value, estimated_error = result
                accuracy = abs(integral_value - exact) if exact is not None else estimated_error
                
                extra_info = {
                    'integral_value': integral_value,
                    'estimated_error': estimated_error,
                    'exact_value': exact,
                }
                
                self.results.append(BenchmarkResult(
                    benchmark_name, mean_time, std_time, accuracy, extra_info
                ))
                
            except Exception as e:
                print(f"    Error: {e}")
                self.results.append(BenchmarkResult(
                    benchmark_name, float('inf'), 0.0, None, {'error': str(e)}
                ))
    
    def bench_multidimensional_integration(self):
        """Benchmark multidimensional integration."""
        print("Benchmarking multidimensional integration...")
        
        dimensions = [2, 3, 4, 5, 6]
        
        for dim in dimensions:
            # Monte Carlo style integration (using scipy's version)
            benchmark_name = f"monte_carlo_gaussian_{dim}d"
            print(f"  Running {benchmark_name}...")
            
            try:
                # Create integration domain
                ranges = [(-2.0, 2.0) for _ in range(dim)]
                
                def integrand(*args):
                    x = np.array(args)
                    return self.multivariate_gaussian(x)
                
                # Use scipy's nquad for comparison (it's deterministic)
                mean_time, std_time, result = self.time_function(
                    integrate.nquad,
                    integrand, ranges,
                    opts={'epsrel': 1e-6, 'epsabs': 1e-9}
                )
                
                integral_value, estimated_error = result
                
                # Exact value for Gaussian integral: (pi)^(d/2)
                exact = np.pi**(dim/2.0) * (2.0**dim)  # scaled by domain size
                accuracy = abs(integral_value - exact)
                
                extra_info = {
                    'integral_value': integral_value,
                    'estimated_error': estimated_error,
                    'exact_value': exact,
                    'dimensions': dim,
                }
                
                self.results.append(BenchmarkResult(
                    benchmark_name, mean_time, std_time, accuracy, extra_info
                ))
                
            except Exception as e:
                print(f"    Error: {e}")
                self.results.append(BenchmarkResult(
                    benchmark_name, float('inf'), 0.0, None, {'error': str(e)}
                ))
    
    def bench_accuracy_performance_tradeoff(self):
        """Benchmark accuracy vs performance trade-offs."""
        print("Benchmarking accuracy vs performance trade-offs...")
        
        tolerances = [1e-3, 1e-6, 1e-9, 1e-12]
        
        for tol in tolerances:
            benchmark_name = f"ode_tolerance_rtol_{tol:.0e}"
            print(f"  Running {benchmark_name}...")
            
            try:
                mean_time, std_time, result = self.time_function(
                    integrate.solve_ivp,
                    self.harmonic_oscillator, [0.0, 10.0], [1.0, 0.0],
                    method="DOP853",
                    rtol=tol, atol=tol * 1e-3
                )
                
                # Energy conservation accuracy
                if result.success:
                    energy = 0.5 * (result.y[0][-1]**2 + result.y[1][-1]**2)
                    accuracy = abs(energy - 0.5)
                else:
                    accuracy = float('inf')
                
                extra_info = {
                    'tolerance': tol,
                    'nfev': result.nfev,
                    'n_steps': len(result.t) - 1,
                    'success': result.success,
                }
                
                self.results.append(BenchmarkResult(
                    benchmark_name, mean_time, std_time, accuracy, extra_info
                ))
                
            except Exception as e:
                print(f"    Error: {e}")
                self.results.append(BenchmarkResult(
                    benchmark_name, float('inf'), 0.0, None, {'error': str(e)}
                ))
    
    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("Starting SciPy benchmark suite...")
        print(f"Running {self.n_runs} iterations per benchmark")
        print("=" * 50)
        
        self.bench_ode_solvers()
        self.bench_quadrature_methods() 
        self.bench_multidimensional_integration()
        self.bench_accuracy_performance_tradeoff()
        
        print("=" * 50)
        print("SciPy benchmarks completed!")
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        data = []
        for result in self.results:
            data.append({
                'name': result.name,
                'mean_time': result.mean_time,
                'std_time': result.std_time,
                'accuracy': result.accuracy,
                'extra_info': result.extra_info,
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\nBenchmark Summary:")
        print("=" * 80)
        print(f"{'Benchmark':<40} {'Time (ms)':<12} {'Std (ms)':<12} {'Accuracy':<12}")
        print("-" * 80)
        
        for result in self.results:
            time_ms = result.mean_time * 1000
            std_ms = result.std_time * 1000
            acc_str = f"{result.accuracy:.2e}" if result.accuracy is not None else "N/A"
            
            print(f"{result.name:<40} {time_ms:<12.3f} {std_ms:<12.3f} {acc_str:<12}")

def main():
    """Run the SciPy benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SciPy benchmark suite")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per benchmark")
    parser.add_argument("--output", type=str, default="scipy_benchmarks.json", 
                       help="Output JSON file")
    args = parser.parse_args()
    
    # Run benchmarks
    benchmarks = ScipyBenchmarks(n_runs=args.runs)
    benchmarks.run_all_benchmarks()
    
    # Display and save results
    benchmarks.print_summary()
    benchmarks.save_results(args.output)

if __name__ == "__main__":
    main()