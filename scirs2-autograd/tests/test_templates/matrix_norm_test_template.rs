// Test template for matrix norm gradient verification
// Use this template when implementing gradient calculations for matrix norms

use crate::{
    ndarray::{array, Array2},
    tensor_ops as T,
};
use scirs2_autograd as ag;

// Utility function for comparing matrices with tolerance
fn assert_matrix_close<F: ag::Float>(
    actual: &Array2<F>, 
    expected: &Array2<F>, 
    tolerance: F, 
    operation: &str
) {
    assert_eq!(actual.shape(), expected.shape(), 
        "{} matrices must have same shape", operation);
    
    for ((i, j), (a, e)) in actual.indexed_iter().zip(expected.iter()) {
        assert!(
            (*a - *e).abs() < tolerance,
            "{} gradient mismatch at [{}, {}]: got {}, expected {}, diff: {}",
            operation, i, j, *a, *e, (*a - *e).abs()
        );
    }
}

// Finite difference gradient verification
fn verify_norm_gradient_finite_diff<F: ag::Float>(
    norm_fn: impl Fn(&ag::Tensor<F>) -> ag::Tensor<F>,
    matrix: Array2<F>,
    h: F,
    tolerance: F,
) {
    ag::run::<F, _, _>(|ctx| {
        let a = T::convert_to_tensor(matrix.clone(), ctx);
        
        // Compute analytical gradient
        let norm = norm_fn(&a);
        let grad = T::grad(&[norm], &[&a])[0];
        let analytical_grad = grad.eval(ctx).unwrap()
            .into_dimensionality::<ag::ndarray::Ix2>()
            .unwrap();
        
        // Compute numerical gradient using finite differences
        let mut numerical_grad = Array2::<F>::zeros(matrix.raw_dim());
        
        for ((i, j), _) in matrix.indexed_iter() {
            // Forward difference: f(x + h) - f(x - h) / (2h)
            let mut matrix_plus = matrix.clone();
            let mut matrix_minus = matrix.clone();
            
            matrix_plus[[i, j]] = matrix_plus[[i, j]] + h;
            matrix_minus[[i, j]] = matrix_minus[[i, j]] - h;
            
            let a_plus = T::convert_to_tensor(matrix_plus, ctx);
            let a_minus = T::convert_to_tensor(matrix_minus, ctx);
            
            let norm_plus = norm_fn(&a_plus).eval(ctx).unwrap()[[0]];
            let norm_minus = norm_fn(&a_minus).eval(ctx).unwrap()[[0]];
            
            numerical_grad[[i, j]] = (norm_plus - norm_minus) / (h + h);
        }
        
        // Compare analytical and numerical gradients
        assert_matrix_close(&analytical_grad, &numerical_grad, tolerance, "gradient");
    });
}

#[cfg(test)]
mod template_tests {
    use super::*;
    
    #[test]
    fn test_frobenius_norm_gradient_verification() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        
        verify_norm_gradient_finite_diff(
            |a| T::frobenius_norm(a),
            matrix,
            1e-6,  // h for finite differences
            1e-4   // tolerance for comparison
        );
    }
    
    #[test]
    fn test_spectral_norm_gradient_verification() {
        // Use a diagonal matrix for simpler verification
        let matrix = array![[2.0, 0.0], [0.0, 3.0]];
        
        verify_norm_gradient_finite_diff(
            |a| T::spectral_norm(a),
            matrix,
            1e-6,
            1e-4
        );
    }
    
    #[test]
    fn test_nuclear_norm_gradient_verification() {
        // Use a diagonal matrix for simpler verification
        let matrix = array![[1.0, 0.0], [0.0, 2.0]];
        
        verify_norm_gradient_finite_diff(
            |a| T::nuclear_norm(a),
            matrix,
            1e-6,
            1e-4
        );
    }
    
    #[test]
    fn test_matrix_norm_edge_cases() {
        // Test with nearly zero matrix
        let near_zero = array![[1e-10, 0.0], [0.0, 1e-10]];
        
        ag::run::<f64, _, _>(|ctx| {
            let a = T::convert_to_tensor(near_zero, ctx);
            
            // All norms should handle near-zero matrices gracefully
            let frob_norm = T::frobenius_norm(&a);
            let spec_norm = T::spectral_norm(&a);
            let nucl_norm = T::nuclear_norm(&a);
            
            // Gradients should not contain NaN or infinity
            let frob_grad = T::grad(&[frob_norm], &[&a])[0].eval(ctx).unwrap();
            let spec_grad = T::grad(&[spec_norm], &[&a])[0].eval(ctx).unwrap();
            let nucl_grad = T::grad(&[nucl_norm], &[&a])[0].eval(ctx).unwrap();
            
            assert!(!frob_grad.iter().any(|&x| x.is_nan() || x.is_infinite()));
            assert!(!spec_grad.iter().any(|&x| x.is_nan() || x.is_infinite()));
            assert!(!nucl_grad.iter().any(|&x| x.is_nan() || x.is_infinite()));
        });
    }
}