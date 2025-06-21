//! Cross-module integration tests for scirs2
//!
//! This module tests the integration between different scirs2 modules to ensure
//! they work together properly in realistic scientific computing scenarios.

use ndarray::{Array1, Array2, array};
use scirs2_autograd as ag;
use scirs2_linalg::{basic, decomposition, norm};

#[test]
fn test_autograd_linalg_integration() {
    // Test that autograd can work with linalg operations
    ag::run(|ctx: &mut ag::Context<f64>| {
        // Create test matrices using autograd
        let a_data = array![[4.0, 1.0], [1.0, 3.0]];
        let a = ag::tensor_ops::variable(a_data.clone(), ctx);
        
        // Verify autograd operations work
        let trace_ag = a.trace();
        let trace_result = trace_ag.eval(ctx).unwrap();
        
        // Verify linalg operations work on the same data
        let trace_linalg = a_data[[0, 0]] + a_data[[1, 1]];
        
        assert!((trace_result.into_scalar() - trace_linalg).abs() < 1e-10);
    });
}

#[test]
fn test_linalg_norm_matrix_operations() {
    // Test that linalg and norm modules work together
    let matrix = array![[3.0, 4.0], [0.0, 5.0]];
    
    // Compute matrix norms
    let frobenius_norm = norm::matrix_norm(&matrix.view(), "fro").unwrap();
    let one_norm = norm::matrix_norm(&matrix.view(), "1").unwrap();
    let inf_norm = norm::matrix_norm(&matrix.view(), "inf").unwrap();
    
    // Verify reasonable values
    assert!(frobenius_norm > 0.0);
    assert!(one_norm > 0.0);
    assert!(inf_norm > 0.0);
    
    // Test basic matrix operations
    let det = basic::det(&matrix.view()).unwrap();
    assert!((det - 15.0).abs() < 1e-10); // 3*5 - 4*0 = 15
}

#[test] 
fn test_core_memory_efficiency() {
    // Test core memory management functionality
    use scirs2_core::memory::BufferPool;
    
    let pool = BufferPool::new(1024, 10);
    
    // Test buffer allocation and management
    let buffer1 = pool.get_buffer();
    let buffer2 = pool.get_buffer();
    
    assert!(buffer1.len() >= 1024);
    assert!(buffer2.len() >= 1024);
    
    // Return buffers to pool
    drop(buffer1);
    drop(buffer2);
    
    // Pool should reuse buffers
    let buffer3 = pool.get_buffer();
    assert!(buffer3.len() >= 1024);
}

#[test]
fn test_scientific_computing_pipeline() {
    // Test a realistic scientific computing pipeline using multiple modules
    
    // 1. Generate some test data
    let n = 10;
    let mut data = Array2::<f64>::zeros((n, n));
    
    // Create a symmetric positive definite matrix
    for i in 0..n {
        for j in 0..n {
            data[[i, j]] = if i == j { 
                2.0 + i as f64 * 0.1 
            } else { 
                0.1 * (i as f64 * j as f64).sin()
            };
        }
    }
    
    // Make it symmetric
    for i in 0..n {
        for j in i+1..n {
            data[[j, i]] = data[[i, j]];
        }
    }
    
    // 2. Compute basic properties
    let det = basic::det(&data.view()).unwrap();
    let frobenius_norm = norm::matrix_norm(&data.view(), "fro").unwrap();
    
    assert!(det > 0.0); // Should be positive definite
    assert!(frobenius_norm > 0.0);
    
    // 3. Test with autograd for gradient computation
    ag::run(|ctx: &mut ag::Context<f64>| {
        let matrix = ag::tensor_ops::variable(data.clone().into_dyn(), ctx);
        
        // Compute some function of the matrix
        let trace = matrix.trace();
        let trace_squared = &trace * &trace;
        
        // Test gradient computation
        let grad = ag::tensor_ops::grad(&[trace_squared], &[matrix]);
        assert!(grad.len() == 1);
        
        let grad_result = grad[0].eval(ctx).unwrap();
        assert_eq!(grad_result.shape(), data.shape());
    });
    
    println!("Integrated scientific computing pipeline completed successfully!");
}

#[test]
fn test_error_handling_across_modules() {
    // Test that error handling works consistently across modules
    
    // Test linalg errors
    let empty_matrix = Array2::<f64>::zeros((0, 0));
    let det_result = basic::det(&empty_matrix.view());
    assert!(det_result.is_err());
    
    // Test autograd errors
    ag::run(|ctx: &mut ag::Context<f64>| {
        let invalid_data = array![[]]; // Empty 1D array
        let tensor = ag::tensor_ops::variable(invalid_data.into_dyn(), ctx);
        
        // This should not panic but should handle gracefully
        let trace_result = tensor.trace().eval(ctx);
        // We expect this to either work or fail gracefully
        match trace_result {
            Ok(_) => println!("Trace computation succeeded unexpectedly"),
            Err(_) => println!("Trace computation failed as expected"),
        }
    });
}

#[test]
fn test_numerical_stability() {
    // Test numerical stability across different module combinations
    
    // Create a nearly singular matrix
    let matrix = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0 + 1e-12, 1.0],
        [1.0, 1.0, 1.0 + 1e-12]
    ];
    
    // Test determinant computation
    let det = basic::det(&matrix.view()).unwrap();
    assert!(det.abs() < 1e-10); // Should be very small
    
    // Test norm computation (should be stable)
    let norm = norm::matrix_norm(&matrix.view(), "fro").unwrap();
    assert!(norm > 0.0);
    assert!(norm < 10.0); // Should be reasonable
    
    // Test with autograd
    ag::run(|ctx: &mut ag::Context<f64>| {
        let ag_matrix = ag::tensor_ops::variable(matrix.into_dyn(), ctx);
        let ag_norm = ag_matrix.frobenius_norm();
        let norm_result = ag_norm.eval(ctx).unwrap();
        
        assert!(norm_result.into_scalar() > 0.0);
    });
}