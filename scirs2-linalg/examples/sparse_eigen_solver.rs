use ndarray::{s, Array2};
use scirs2_linalg::prelude::*;
use scirs2_linalg::{largest_k_eigh, smallest_k_eigh};

fn main() {
    // Create a symmetric test matrix
    println!("Creating test matrix...");
    let n = 100;
    let mut a = Array2::<f64>::zeros((n, n));

    // Fill diagonal with values 1 to n
    for i in 0..n {
        a[[i, i]] = (i + 1) as f64;
    }

    // Add some off-diagonal elements to make it more interesting
    for i in 0..n - 1 {
        a[[i, i + 1]] = 0.5;
        a[[i + 1, i]] = 0.5;
    }

    println!("Matrix shape: {:?}", a.shape());
    println!("First 5x5 submatrix:");
    print_submatrix(&a, 5);

    // Compute only the 5 largest eigenvalues
    println!("\nComputing 5 largest eigenvalues...");
    let start = std::time::Instant::now();
    let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 5, 1000, 1e-8).unwrap();
    let duration = start.elapsed();

    println!("Largest eigenvalues:");
    for (i, &val) in eigenvalues.iter().enumerate() {
        println!("λ{}: {:.6}", i + 1, val);
    }
    println!("Time taken: {:?}", duration);

    // Verify that these are eigenvectors by computing ||Av - λv||
    println!("\nVerifying eigenvectors (showing residual ||Av - λv||):");
    for i in 0..5 {
        let v = eigenvectors.slice(s![.., i]);
        let av = a.dot(&v);
        let lambda_v = &v * eigenvalues[i];
        let residual = (&av - &lambda_v).mapv(|x| x * x).sum().sqrt();
        println!("Residual for eigenvector {}: {:.6e}", i + 1, residual);
    }

    // Compute only the 5 smallest eigenvalues
    println!("\nComputing 5 smallest eigenvalues...");
    let start = std::time::Instant::now();
    let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 5, 1000, 1e-8).unwrap();
    let duration = start.elapsed();

    println!("Smallest eigenvalues:");
    for (i, &val) in eigenvalues.iter().enumerate() {
        println!("λ{}: {:.6}", i + 1, val);
    }
    println!("Time taken: {:?}", duration);

    // Verify that these are eigenvectors
    println!("\nVerifying eigenvectors (showing residual ||Av - λv||):");
    for i in 0..5 {
        let v = eigenvectors.slice(s![.., i]);
        let av = a.dot(&v);
        let lambda_v = &v * eigenvalues[i];
        let residual = (&av - &lambda_v).mapv(|x| x * x).sum().sqrt();
        println!("Residual for eigenvector {}: {:.6e}", i + 1, residual);
    }

    // Compare with the standard eigenvalue solver for reference
    println!("\nComputing all eigenvalues with standard solver for comparison...");
    let start = std::time::Instant::now();
    let (all_eigenvalues, _) = eigh(&a.view()).unwrap();
    let duration = start.elapsed();

    println!("5 largest eigenvalues from standard solver:");
    let mut sorted_eigenvalues = all_eigenvalues.to_vec();
    sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for (i, &val) in sorted_eigenvalues.iter().take(5).enumerate() {
        println!("λ{}: {:.6}", i + 1, val);
    }

    println!("\n5 smallest eigenvalues from standard solver:");
    sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (i, &val) in sorted_eigenvalues.iter().take(5).enumerate() {
        println!("λ{}: {:.6}", i + 1, val);
    }

    println!("Time taken for standard solver: {:?}", duration);
}

// Helper function to print a submatrix
fn print_submatrix<T: std::fmt::Display>(matrix: &Array2<T>, size: usize) {
    let n = std::cmp::min(size, matrix.shape()[0]);
    let m = std::cmp::min(size, matrix.shape()[1]);

    for i in 0..n {
        for j in 0..m {
            print!("{:7.3} ", matrix[[i, j]]);
        }
        println!();
    }
}
