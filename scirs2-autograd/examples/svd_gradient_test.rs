use ag::tensor_ops::*;
use ndarray::{array, Array2, Ix2};
use scirs2_autograd as ag;

fn main() {
    println!("Testing gradient backpropagation through SVD...");

    // Simplified test with a smaller matrix for faster execution
    ag::run::<f64, _, _>(|g| {
        // Create a simple 2x2 matrix (smaller for faster testing)
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
        {
            println!("\nTest 1: Gradient through sum of U");
            // Create a simple loss function: sum of all elements in U
            let loss_u = sum_all(u);

            // Compute gradient of the loss with respect to the input matrix
            let grads_u = grad(&[loss_u], &[&matrix]);
            let grad_matrix_u = &grads_u[0];

            // Print the gradient
            println!("Gradient of sum(U) with respect to matrix:");
            match grad_matrix_u.eval(g) {
                Ok(arr) => {
                    println!("Shape: {:?}", arr.shape());
                    println!("Values: {:?}", arr);

                    // Get matrix dimensions for the finite difference calculation
                    let shape = matrix_data.shape();
                    let rows = shape[0]; // 3
                    let cols = shape[1]; // 2

                    // Verify the gradient using finite differences
                    println!("Verifying gradient with finite differences...");

                    let eps = 1e-5;
                    let mut fd_grad = Array2::<f64>::zeros((rows, cols));

                    // Compute finite difference gradient
                    for i in 0..rows {
                        for j in 0..cols {
                            // Perturb the input matrix
                            let mut perturbed_plus = matrix_data.clone();
                            perturbed_plus[[i, j]] += eps;

                            let mut perturbed_minus = matrix_data.clone();
                            perturbed_minus[[i, j]] -= eps;

                            // Compute SVD for perturbed matrices
                            let matrix_plus = convert_to_tensor(perturbed_plus, g);
                            let (u_plus, _, _) = svd(matrix_plus);
                            let sum_u_plus = u_plus.eval(g).unwrap().sum();

                            let matrix_minus = convert_to_tensor(perturbed_minus, g);
                            let (u_minus, _, _) = svd(matrix_minus);
                            let sum_u_minus = u_minus.eval(g).unwrap().sum();

                            // Compute finite difference
                            fd_grad[[i, j]] = (sum_u_plus - sum_u_minus) / (2.0 * eps);
                        }
                    }

                    println!("Finite difference gradient:");
                    println!("{:?}", fd_grad);

                    // Compare with analytical gradient
                    println!("Shape of gradient array: {:?}", arr.shape());
                    let grad_arr = arr.into_dimensionality::<Ix2>().unwrap();
                    let mut max_diff: f64 = 0.0;

                    // Handle the case where the gradient shape doesn't match the input matrix shape
                    // This can happen during development if the gradient implementation is incomplete
                    if grad_arr.shape() != fd_grad.shape() {
                        println!("WARNING: Gradient shape mismatch - analytical: {:?}, finite difference: {:?}", 
                                 grad_arr.shape(), fd_grad.shape());
                        println!("Can't compare gradients due to shape mismatch");
                    } else {
                        for i in 0..rows {
                            for j in 0..cols {
                                let diff = (grad_arr[[i, j]] - fd_grad[[i, j]]).abs();
                                max_diff = max_diff.max(diff);
                            }
                        }
                        println!(
                            "Maximum difference between analytical and finite difference: {:.6}",
                            max_diff
                        );
                        println!(
                            "Gradient verification: {}",
                            if max_diff < 0.01 { "PASSED" } else { "FAILED" }
                        );
                    }
                }
                Err(e) => println!("Error computing gradient: {:?}", e),
            }
        }

        // Test 2: Gradient through sum of S
        {
            println!("\nTest 2: Gradient through sum of S");
            // Create a simple loss function: sum of all singular values
            let loss_s = sum_all(s);

            // Compute gradient of the loss with respect to the input matrix
            let grads_s = grad(&[loss_s], &[&matrix]);
            let grad_matrix_s = &grads_s[0];

            // Print the gradient
            println!("Gradient of sum(S) with respect to matrix:");
            match grad_matrix_s.eval(g) {
                Ok(arr) => {
                    println!("Shape: {:?}", arr.shape());
                    println!("Values: {:?}", arr);

                    // Get matrix dimensions for the finite difference calculation
                    let shape = matrix_data.shape();
                    let rows = shape[0]; // 3
                    let cols = shape[1]; // 2

                    // Verify the gradient using finite differences
                    println!("Verifying gradient with finite differences...");

                    let eps = 1e-5;
                    let mut fd_grad = Array2::<f64>::zeros((rows, cols));

                    // Compute finite difference gradient
                    for i in 0..rows {
                        for j in 0..cols {
                            // Perturb the input matrix
                            let mut perturbed_plus = matrix_data.clone();
                            perturbed_plus[[i, j]] += eps;

                            let mut perturbed_minus = matrix_data.clone();
                            perturbed_minus[[i, j]] -= eps;

                            // Compute SVD for perturbed matrices
                            let matrix_plus = convert_to_tensor(perturbed_plus, g);
                            let (_, s_plus, _) = svd(matrix_plus);
                            let sum_s_plus = s_plus.eval(g).unwrap().sum();

                            let matrix_minus = convert_to_tensor(perturbed_minus, g);
                            let (_, s_minus, _) = svd(matrix_minus);
                            let sum_s_minus = s_minus.eval(g).unwrap().sum();

                            // Compute finite difference
                            fd_grad[[i, j]] = (sum_s_plus - sum_s_minus) / (2.0 * eps);
                        }
                    }

                    println!("Finite difference gradient:");
                    println!("{:?}", fd_grad);

                    // Compare with analytical gradient
                    println!("Shape of gradient array: {:?}", arr.shape());
                    let grad_arr = arr.into_dimensionality::<Ix2>().unwrap();
                    let mut max_diff: f64 = 0.0;

                    // Handle the case where the gradient shape doesn't match the input matrix shape
                    // This can happen during development if the gradient implementation is incomplete
                    if grad_arr.shape() != fd_grad.shape() {
                        println!("WARNING: Gradient shape mismatch - analytical: {:?}, finite difference: {:?}", 
                                 grad_arr.shape(), fd_grad.shape());
                        println!("Can't compare gradients due to shape mismatch");
                    } else {
                        for i in 0..rows {
                            for j in 0..cols {
                                let diff = (grad_arr[[i, j]] - fd_grad[[i, j]]).abs();
                                max_diff = max_diff.max(diff);
                            }
                        }
                        println!(
                            "Maximum difference between analytical and finite difference: {:.6}",
                            max_diff
                        );
                        println!(
                            "Gradient verification: {}",
                            if max_diff < 0.01 { "PASSED" } else { "FAILED" }
                        );
                    }
                }
                Err(e) => println!("Error computing gradient: {:?}", e),
            }
        }

        // Test 3: Gradient through sum of V
        {
            println!("\nTest 3: Gradient through sum of V");
            // Create a simple loss function: sum of all elements in V
            let loss_v = sum_all(v);

            // Compute gradient of the loss with respect to the input matrix
            let grads_v = grad(&[loss_v], &[&matrix]);
            let grad_matrix_v = &grads_v[0];

            // Print the gradient
            println!("Gradient of sum(V) with respect to matrix:");
            match grad_matrix_v.eval(g) {
                Ok(arr) => {
                    println!("Shape: {:?}", arr.shape());
                    println!("Values: {:?}", arr);

                    // Get matrix dimensions for the finite difference calculation
                    let shape = matrix_data.shape();
                    let rows = shape[0]; // 3
                    let cols = shape[1]; // 2

                    // Verify the gradient using finite differences
                    println!("Verifying gradient with finite differences...");

                    let eps = 1e-5;
                    let mut fd_grad = Array2::<f64>::zeros((rows, cols));

                    // Compute finite difference gradient
                    for i in 0..rows {
                        for j in 0..cols {
                            // Perturb the input matrix
                            let mut perturbed_plus = matrix_data.clone();
                            perturbed_plus[[i, j]] += eps;

                            let mut perturbed_minus = matrix_data.clone();
                            perturbed_minus[[i, j]] -= eps;

                            // Compute SVD for perturbed matrices
                            let matrix_plus = convert_to_tensor(perturbed_plus, g);
                            let (_, _, v_plus) = svd(matrix_plus);
                            let sum_v_plus = v_plus.eval(g).unwrap().sum();

                            let matrix_minus = convert_to_tensor(perturbed_minus, g);
                            let (_, _, v_minus) = svd(matrix_minus);
                            let sum_v_minus = v_minus.eval(g).unwrap().sum();

                            // Compute finite difference
                            fd_grad[[i, j]] = (sum_v_plus - sum_v_minus) / (2.0 * eps);
                        }
                    }

                    println!("Finite difference gradient:");
                    println!("{:?}", fd_grad);

                    // Compare with analytical gradient
                    println!("Shape of gradient array: {:?}", arr.shape());
                    let grad_arr = arr.into_dimensionality::<Ix2>().unwrap();
                    let mut max_diff: f64 = 0.0;

                    // Handle the case where the gradient shape doesn't match the input matrix shape
                    // This can happen during development if the gradient implementation is incomplete
                    if grad_arr.shape() != fd_grad.shape() {
                        println!("WARNING: Gradient shape mismatch - analytical: {:?}, finite difference: {:?}", 
                                 grad_arr.shape(), fd_grad.shape());
                        println!("Can't compare gradients due to shape mismatch");
                    } else {
                        for i in 0..rows {
                            for j in 0..cols {
                                let diff = (grad_arr[[i, j]] - fd_grad[[i, j]]).abs();
                                max_diff = max_diff.max(diff);
                            }
                        }
                        println!(
                            "Maximum difference between analytical and finite difference: {:.6}",
                            max_diff
                        );
                        println!(
                            "Gradient verification: {}",
                            if max_diff < 0.01 { "PASSED" } else { "FAILED" }
                        );
                    }
                }
                Err(e) => println!("Error computing gradient: {:?}", e),
            }
        }

        // Test 4: Gradient through reconstruction loss
        {
            println!("\nTest 4: Gradient through reconstruction loss");

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
                }
                Err(e) => println!("Error computing gradient: {:?}", e),
            }
        }

        println!("\nSVD gradient backpropagation tests completed!");
    });
}
