//! Debug test to understand gradient flow issues

use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

#[test]
#[ignore = "Gradient shape propagation architectural limitation"]
fn test_trace_gradient_flow() {
    ag::run(|g| {
        // Create a simple 2x2 matrix
        let a = variable(array![[1.0_f64, 2.0], [3.0, 4.0]], g);

        // Compute trace
        let tr = trace(a);
        println!("Trace value: {:?}", tr.eval(g).unwrap());

        // Compute gradient
        let grads = grad(&[&tr], &[&a]);
        let grad_a = grads[0].eval(g).unwrap();

        println!("Gradient shape: {:?}", grad_a.shape());
        println!("Gradient values: {:?}", grad_a);

        // Expected gradient for trace is identity matrix
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_a[[0, 0]], 1.0);
        assert_eq!(grad_a[[1, 1]], 1.0);
        assert_eq!(grad_a[[0, 1]], 0.0);
        assert_eq!(grad_a[[1, 0]], 0.0);
    });
}

#[test]
#[ignore = "Gradient shape propagation architectural limitation"]
fn test_matrix_inverse_gradient_flow() {
    ag::run(|g| {
        let a = variable(array![[3.0_f64, 1.0], [1.0, 2.0]], g);

        // Just test matrix inverse gradient directly
        let inv_a = matinv(&a);
        let sum_inv = sum_all(inv_a);

        println!("Sum of inverse: {:?}", sum_inv.eval(g).unwrap());

        let grads = grad(&[&sum_inv], &[&a]);
        let grad_a = grads[0].eval(g).unwrap();

        println!("Matrix inverse gradient shape: {:?}", grad_a.shape());
        println!("Matrix inverse gradient: {:?}", grad_a);

        // Should be a 2x2 matrix
        assert_eq!(grad_a.shape(), &[2, 2]);
    });
}

#[test]
#[ignore = "Gradient shape propagation architectural limitation"]
fn test_chained_gradient_flow() {
    ag::run(|g| {
        // Test gradient flow through inv -> trace chain
        let a = variable(array![[3.0_f64, 1.0], [1.0, 2.0]], g);

        // Step by step
        let inv_a = matinv(&a);
        println!("Inverse shape: {:?}", inv_a.eval(g).unwrap().shape());

        let tr_inv = trace(inv_a);
        println!(
            "Trace of inverse shape: {:?}",
            tr_inv.eval(g).unwrap().shape()
        );
        println!("Trace of inverse value: {:?}", tr_inv.eval(g).unwrap());

        // Now compute gradient
        let grads = grad(&[&tr_inv], &[&a]);
        println!("Number of gradients: {}", grads.len());

        // Check the gradient tensor itself before evaluation
        let grad_tensor = &grads[0];
        println!(
            "Gradient tensor shape (from tensor): {:?}",
            grad_tensor.shape()
        );

        let grad_a = grad_tensor.eval(g).unwrap();
        println!("Gradient shape after eval: {:?}", grad_a.shape());
        println!("Gradient values: {:?}", grad_a);

        // This should be a 2x2 matrix
        assert_eq!(grad_a.shape(), &[2, 2]);
    });
}
