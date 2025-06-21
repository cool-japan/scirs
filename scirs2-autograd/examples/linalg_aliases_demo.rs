//! Demonstration of linear algebra operation aliases in scirs2-autograd
//!
//! This example shows how to use the convenient short aliases for common
//! linear algebra operations that are now implemented with autodiff support.

use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    println!("=== Linear Algebra Aliases Demo ===\n");

    ag::run::<f64, _, _>(|g| {
        println!("1. Matrix Inverse and Determinant Aliases");

        let a = variable(array![[3.0, 1.0], [1.0, 2.0]], g);

        // Using the short aliases
        let inverse = matinv(&a); // Note: inv conflicts with arithmetic::inv (reciprocal)
        let determinant = det(&a);

        println!("Matrix A:");
        println!("{:?}", a.eval(g).unwrap());
        println!("\nUsing matinv(A):");
        println!("{:?}", inverse.eval(g).unwrap());
        println!("\nUsing det(A): {}", determinant.eval(g).unwrap()[[]]);

        // Verify: A * inv(A) = I
        let identity_check = matmul(a, inverse);
        println!("\nA * inv(A) (should be identity):");
        println!("{:?}", identity_check.eval(g).unwrap());

        // Gradient of determinant
        let det_grad = grad(&[&determinant], &[&a]);
        println!("\nGradient of det w.r.t. A:");
        println!("{:?}", det_grad[0].eval(g).unwrap());

        println!("\n2. Pseudo-Inverse Alias");

        let rect = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], g);
        let pseudo_inv = pinv(&rect);

        println!("Rectangular matrix:");
        println!("{:?}", rect.eval(g).unwrap());
        println!("\nUsing pinv():");
        println!("{:?}", pseudo_inv.eval(g).unwrap());

        // Verify: A * pinv(A) * A ≈ A
        let check = matmul(matmul(rect, pseudo_inv), rect);
        println!("\nA * pinv(A) * A (should equal A):");
        println!("{:?}", check.eval(g).unwrap());

        println!("\n3. Eigendecomposition Alias");

        let symmetric = convert_to_tensor(array![[4.0, 1.0], [1.0, 3.0]], g);
        let (eigenvals, eigenvecs) = eig(&symmetric);

        println!("Symmetric matrix:");
        println!("{:?}", symmetric.eval(g).unwrap());
        println!("\nUsing eig() - Eigenvalues:");
        println!("{:?}", eigenvals.eval(g).unwrap());
        println!("Eigenvectors:");
        println!("{:?}", eigenvecs.eval(g).unwrap());

        println!("\n4. Matrix Functions Aliases");

        // Matrix square root
        let pos_def = convert_to_tensor(array![[4.0, 1.0], [1.0, 3.0]], g);
        let sqrt_mat = sqrtm(&pos_def);

        println!("Positive definite matrix:");
        println!("{:?}", pos_def.eval(g).unwrap());
        println!("\nUsing sqrtm():");
        println!("{:?}", sqrt_mat.eval(g).unwrap());

        // Verify: sqrtm(A) * sqrtm(A) = A
        let sqrt_squared = matmul(sqrt_mat, sqrt_mat);
        println!("\nsqrtm(A) * sqrtm(A) (should equal A):");
        println!("{:?}", sqrt_squared.eval(g).unwrap());

        // Matrix logarithm and exponential
        let small_mat = convert_to_tensor(array![[0.5, 0.1], [0.1, 0.3]], g);
        let exp_mat = matrix_exp(&small_mat);
        let log_mat = logm(&exp_mat);

        println!("\n5. Matrix Logarithm Alias");
        println!("Original matrix:");
        println!("{:?}", small_mat.eval(g).unwrap());
        println!("\nexp(A):");
        println!("{:?}", exp_mat.eval(g).unwrap());
        println!("\nUsing logm(exp(A)) (should equal A):");
        println!("{:?}", log_mat.eval(g).unwrap());

        println!("\n6. Combined Operations with Aliases");

        // Complex expression using multiple aliases
        let x = variable(array![[2.0, 0.5], [0.5, 3.0]], g);

        // Compute: tr(inv(A)) + det(A) - ||sqrtm(A)||_F
        let inv_x = matinv(&x);
        let tr_inv = trace(inv_x);
        let det_x = det(&x);
        let sqrt_x = sqrtm(&x);
        let norm_sqrt = frobenius_norm(sqrt_x);

        let result = sub(add(tr_inv, det_x), norm_sqrt);

        println!("Complex expression: tr(inv(A)) + det(A) - ||sqrtm(A)||_F");
        println!("Result: {}", result.eval(g).unwrap()[[]]);

        // Compute gradient
        let grads = grad(&[&result], &[&x]);
        println!("\nGradient w.r.t. A:");
        println!("{:?}", grads[0].eval(g).unwrap());

        println!("\n=== All aliases working correctly! ===");
    });
}
