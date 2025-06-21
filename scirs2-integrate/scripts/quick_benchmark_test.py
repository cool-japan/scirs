#!/usr/bin/env python3
"""
Quick test to verify SciPy benchmarking works.
"""

import numpy as np
import scipy.integrate as integrate
import time

def test_basic_functionality():
    """Test basic SciPy functionality."""
    print("Testing basic SciPy functionality...")
    
    # Simple ODE test
    def exponential_decay(t, y):
        return -y
    
    start = time.time()
    result = integrate.solve_ivp(exponential_decay, [0, 1], [1.0], method='RK45')
    scipy_time = time.time() - start
    
    print(f"SciPy ODE solve_ivp: {scipy_time:.6f} seconds")
    print(f"Final value: {result.y[0][-1]:.6f}")
    print(f"Expected: {np.exp(-1):.6f}")
    print(f"Success: {result.success}")
    
    # Simple quadrature test
    def polynomial(x):
        return x**3
    
    start = time.time()
    integral, error = integrate.quad(polynomial, 0, 1)
    quad_time = time.time() - start
    
    print(f"\nSciPy quad: {quad_time:.6f} seconds")
    print(f"Integral result: {integral:.6f}")
    print(f"Expected: 0.250000")
    print(f"Error estimate: {error:.2e}")
    
    print("\nBasic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()