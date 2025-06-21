use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    println!("Running simple SVD gradient test...");

    ag::run::<f64, _, _>(|g| {
        // Create a simple 2x2 matrix
        let matrix_data = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix = variable(matrix_data.clone(), g);
        println!(
            "Original matrix shape: {:?}",
            matrix.eval(g).unwrap().shape()
        );

        // Compute SVD
        let (u, s, v) = svd(matrix);
        println!(
            "SVD shapes: U={:?}, S={:?}, V={:?}",
            u.eval(g).unwrap().shape(),
            s.eval(g).unwrap().shape(),
            v.eval(g).unwrap().shape()
        );

        // Test 1: Gradient through sum of U
        println!("\nTest 1: Gradient through sum of U");
        let loss_u = sum_all(u);
        let grads_u = grad(&[loss_u], &[&matrix]);
        let grad_matrix_u = &grads_u[0];

        // Print the gradient
        println!("Gradient of sum(U) with respect to matrix:");
        match grad_matrix_u.eval(g) {
            Ok(arr) => {
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);

                // Check that the gradient shape matches the input matrix shape
                assert_eq!(arr.shape(), matrix_data.shape());
                println!("Shape match check: PASSED");
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }

        // Test 2: Gradient through sum of S
        println!("\nTest 2: Gradient through sum of S");
        let loss_s = sum_all(s);
        let grads_s = grad(&[loss_s], &[&matrix]);
        let grad_matrix_s = &grads_s[0];

        // Print the gradient
        println!("Gradient of sum(S) with respect to matrix:");
        match grad_matrix_s.eval(g) {
            Ok(arr) => {
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);

                // Check that the gradient shape matches the input matrix shape
                assert_eq!(arr.shape(), matrix_data.shape());
                println!("Shape match check: PASSED");
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }

        // Test 3: Gradient through reconstruction loss
        println!("\nTest 3: Gradient through reconstruction loss");

        // Reconstruct the matrix from SVD components
        let s_diag = diag(s);
        let us = matmul(u, s_diag);
        let v_t = transpose(v, &[1, 0]);
        let reconstructed = matmul(us, v_t);

        // Compute reconstruction loss
        let diff = sub(reconstructed, matrix);
        let loss_recon = sum_all(square(diff));

        // Compute gradient of the loss with respect to the input matrix
        let grads_recon = grad(&[loss_recon], &[&matrix]);
        let grad_matrix_recon = &grads_recon[0];

        // Print the gradient
        println!("Gradient of reconstruction loss with respect to matrix:");
        match grad_matrix_recon.eval(g) {
            Ok(arr) => {
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);

                // Check that the gradient shape matches the input matrix shape
                assert_eq!(arr.shape(), matrix_data.shape());
                println!("Shape match check: PASSED");
            }
            Err(e) => println!("Error computing gradient: {:?}", e),
        }

        println!("\nSVD gradient test completed!");
    });
}
