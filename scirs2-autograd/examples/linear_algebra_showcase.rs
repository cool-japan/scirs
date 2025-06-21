use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    println!("=== scirs2-autograd Linear Algebra Showcase ===\n");

    ag::run::<f64, _, _>(|g| {
        // 1. Basic Linear Algebra Operations
        println!("1. Basic Linear Algebra Operations");

        let a = convert_to_tensor(array![[2.0, 1.0], [1.0, 3.0]], g);
        let eye_matrix = eye(2, g);
        let tr = trace(a);

        println!("Matrix A:");
        println!("{:?}", a.eval(g).unwrap());
        println!("Identity matrix:");
        println!("{:?}", eye_matrix.eval(g).unwrap());
        println!("Trace of A: {}", tr.eval(g).unwrap()[[]]);

        let diag_values = convert_to_tensor(array![4.0, 5.0], g);
        let diag_matrix = diag(diag_values);
        println!("Diagonal matrix from [4, 5]:");
        println!("{:?}", diag_matrix.eval(g).unwrap());

        // 2. Matrix Operations with Gradients
        println!("\n2. Matrix Operations with Gradients");

        let a_var = variable(array![[3.0, 1.0], [1.0, 2.0]], g);
        let inv_a = matinv(&a_var); // Using the new matinv alias (inv conflicts with reciprocal)
        let det_a = det(&a_var); // Using the new det alias

        println!("Inverse of A:");
        println!("{:?}", inv_a.eval(g).unwrap());
        println!("Determinant of A: {}", det_a.eval(g).unwrap()[[]]);

        // Compute gradient of determinant
        let grads = grad(&[&det_a], &[&a_var]);
        println!("Gradient of determinant w.r.t. A:");
        println!("{:?}", grads[0].eval(g).unwrap());

        // 3. Matrix Decompositions
        println!("\n3. Matrix Decompositions");

        // QR decomposition
        let (q, r) = qr(a);
        println!("QR decomposition:");
        println!("Q:\n{:?}", q.eval(g).unwrap());
        println!("R:\n{:?}", r.eval(g).unwrap());

        // Eigenvalue decomposition
        let symmetric = convert_to_tensor(array![[4.0, 1.0], [1.0, 3.0]], g);
        let (eigenvals, eigenvecs) = eig(&symmetric); // Using the new eig alias
        println!("\nEigenvalue decomposition:");
        println!("Eigenvalues: {:?}", eigenvals.eval(g).unwrap());
        println!("Eigenvectors:\n{:?}", eigenvecs.eval(g).unwrap());

        // SVD
        let matrix = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], g);
        let (u, s, v) = svd(matrix);
        println!("\nSVD of 3x2 matrix:");

        println!("U:");
        match u.eval(g) {
            Ok(result) => println!("{:?}", result),
            Err(e) => println!("Error evaluating U: {:?}", e),
        }

        println!("S:");
        match s.eval(g) {
            Ok(result) => println!("{:?}", result),
            Err(e) => println!("Error evaluating S: {:?}", e),
        }

        println!("V:");
        match v.eval(g) {
            Ok(result) => println!("{:?}", result),
            Err(e) => println!("Error evaluating V: {:?}", e),
        }

        // 4. Linear System Solving
        println!("\n4. Linear System Solving");

        let a_system = convert_to_tensor(array![[3.0, 1.0], [1.0, 2.0]], g);
        let b_system = convert_to_tensor(array![[9.0], [8.0]], g);
        let x = solve(a_system, b_system);

        println!("Solving Ax = b:");
        println!("A:\n{:?}", a_system.eval(g).unwrap());
        println!("b:\n{:?}", b_system.eval(g).unwrap());
        println!("x:\n{:?}", x.eval(g).unwrap());

        // Verify solution
        let ax = matmul(a_system, x);
        println!("Verification (Ax):\n{:?}", ax.eval(g).unwrap());

        // 5. Matrix Functions
        println!("\n5. Matrix Functions");

        let small_matrix = convert_to_tensor(array![[0.5, 0.1], [0.1, 0.3]], g);
        let exp_matrix = matrix_exp(&small_matrix);
        let log_exp = logm(&exp_matrix); // Using the new logm alias

        println!("Original matrix:");
        println!("{:?}", small_matrix.eval(g).unwrap());
        println!("exp(A):");
        println!("{:?}", exp_matrix.eval(g).unwrap());
        println!("log(exp(A)) - should equal A:");
        println!("{:?}", log_exp.eval(g).unwrap());

        // Matrix square root
        let pos_def = convert_to_tensor(array![[4.0, 1.0], [1.0, 3.0]], g);
        let sqrt_matrix = sqrtm(&pos_def); // Using the new sqrtm alias
        println!("\nMatrix square root:");
        println!("{:?}", sqrt_matrix.eval(g).unwrap());

        // 6. Special Matrix Operations
        println!("\n6. Special Matrix Operations");

        // Positive definite matrix for Cholesky
        let pd_matrix = convert_to_tensor(array![[4.0, 2.0], [2.0, 5.0]], g);
        let chol = cholesky(&pd_matrix);
        println!("Cholesky decomposition of positive definite matrix:");
        println!("{:?}", chol.eval(g).unwrap());

        // Verify: L * L^T = A
        let reconstructed = matmul(chol, transpose(chol, &[1, 0]));
        println!("L * L^T (should equal original):");
        println!("{:?}", reconstructed.eval(g).unwrap());

        // Extract triangular parts
        let matrix_3x3 =
            convert_to_tensor(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], g);
        let lower = tril(&matrix_3x3, 0);
        let upper = triu(&matrix_3x3, 0);

        println!("\nLower triangular part:");
        println!("{:?}", lower.eval(g).unwrap());
        println!("Upper triangular part:");
        println!("{:?}", upper.eval(g).unwrap());

        // 7. Complex Example: Principal Component Analysis (PCA)
        println!("\n7. Complex Example: PCA-like computation");

        // Create some data
        let data = convert_to_tensor(
            array![
                [2.5, 2.4],
                [0.5, 0.7],
                [2.2, 2.9],
                [1.9, 2.2],
                [3.1, 3.0],
                [2.3, 2.7],
            ],
            g,
        );

        // Center the data
        let mean = reduce_mean(data, &[0], true);
        let centered = sub(data, mean);

        // Compute covariance matrix
        let cov = matmul(transpose(centered, &[1, 0]), centered);
        let cov_scaled = scalar_mul(cov, 1.0 / 5.0); // n-1 = 5

        // Eigendecomposition of covariance matrix
        let (eigenvalues, eigenvectors) = eig(&cov_scaled); // Using the new eig alias

        println!("Data covariance matrix:");
        println!("{:?}", cov_scaled.eval(g).unwrap());
        println!("Eigenvalues (variances along principal components):");
        println!("{:?}", eigenvalues.eval(g).unwrap());
        println!("Eigenvectors (principal components):");
        println!("{:?}", eigenvectors.eval(g).unwrap());

        // 8. Gradient Flow Through Complex Operations
        println!("\n8. Gradient Flow Through Complex Operations");

        let a_grad = variable(array![[2.0, 1.0], [1.0, 3.0]], g);
        let b_grad = variable(array![[1.0], [2.0]], g);

        // Complex computation with gradients using aliases
        let _inv = matinv(&a_grad); // Using the new matinv alias
        let x_sol = solve(&a_grad, &b_grad);
        let det_val = det(&a_grad); // Using the new det alias
        let norm = frobenius_norm(&a_grad);

        // Combine all results
        let combined = add(add(sum_all(x_sol), det_val), norm);

        // Compute gradients
        let grads = grad(&[&combined], &[&a_grad, &b_grad]);

        println!("Gradient of combined result w.r.t. A:");
        println!("{:?}", grads[0].eval(g).unwrap());
        println!("Gradient of combined result w.r.t. b:");
        println!("{:?}", grads[1].eval(g).unwrap());

        println!("\n=== All linear algebra operations completed successfully! ===");
    });
}
