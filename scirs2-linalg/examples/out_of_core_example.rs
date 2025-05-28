//! Example demonstrating out-of-core quantized matrix operations
//!
//! This example shows how to use the out-of-core functionality to work with
//! matrices that are too large to fit in memory, even when quantized.

use ndarray::{Array1, Array2};
use rand::Rng;
use scirs2_linalg::quantization::{out_of_core::ChunkedQuantizedMatrix, QuantizationMethod};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    println!("=== Out-of-Core Quantized Matrix Operations Example ===");

    // Create a temporary file path
    let file_path = get_temp_file_path("example");
    println!("Using temporary file: {}", file_path.to_str().unwrap());

    // Example 1: Creating and using a small chunked quantized matrix
    println!("\n--- Example 1: Small Matrix Operations ---");
    example_small_matrix(&file_path);

    // Example 2: Solving a linear system with a medium-sized matrix
    println!("\n--- Example 2: Medium Matrix Solver ---");
    example_medium_matrix(&file_path);

    // Example 3: Very large matrix demonstration with performance comparison
    println!("\n--- Example 3: Large Matrix Performance Comparison ---");
    example_large_matrix_performance(&file_path);

    // Clean up
    std::fs::remove_file(&file_path).unwrap_or_default();
    println!("\nTemporary file removed");
}

/// Example using a small matrix to demonstrate basic functionality
fn example_small_matrix(file_path: &Path) {
    println!("Creating a small 10x10 matrix...");
    let matrix = create_random_spd_matrix(10);

    // Create a chunked quantized matrix with 8-bit quantization
    println!("Creating chunked quantized matrix with 8-bit quantization...");
    let chunked = ChunkedQuantizedMatrix::new(
        &matrix.view(),
        8,
        QuantizationMethod::Symmetric,
        file_path.to_str().unwrap(),
    )
    .unwrap()
    .symmetric()
    .positive_definite();

    // Create a test vector
    let x = Array1::from_vec((0..10).map(|i| i as f32).collect());
    println!("Input vector: {:?}", x);

    // Apply the matrix to the vector
    println!("Applying matrix to vector...");
    let y = chunked.apply(&x.view()).unwrap();
    println!("Result vector: {:?}", y);

    // Compare with direct multiplication
    println!("Comparing with direct multiplication...");
    let expected = matrix.dot(&x);

    // Calculate relative error
    let mut max_rel_error: f32 = 0.0;
    for i in 0..y.len() {
        let rel_error = if expected[i].abs() > 1e-10 {
            (y[i] - expected[i]).abs() / expected[i].abs()
        } else {
            (y[i] - expected[i]).abs()
        };
        max_rel_error = max_rel_error.max(rel_error);
    }

    println!("Maximum relative error: {:.6}%", max_rel_error * 100.0);
}

/// Example solving a linear system with a medium-sized matrix
fn example_medium_matrix(file_path: &Path) {
    let size = 100;
    println!("Creating a medium {}x{} matrix...", size, size);
    let matrix = create_random_spd_matrix(size);

    // Create a chunked quantized matrix with 8-bit quantization
    println!("Creating chunked quantized matrix with 8-bit quantization...");
    let chunked = ChunkedQuantizedMatrix::new(
        &matrix.view(),
        8,
        QuantizationMethod::Symmetric,
        file_path.to_str().unwrap(),
    )
    .unwrap()
    .symmetric()
    .positive_definite();

    // Create a right-hand side vector
    println!("Creating right-hand side vector...");
    let mut b = Array1::zeros(size);
    for i in 0..size {
        b[i] = (i % 10) as f32;
    }

    // Solve the system using out-of-core conjugate gradient
    println!("Solving system using out-of-core conjugate gradient...");
    let start = Instant::now();
    let x = chunked
        .solve_conjugate_gradient(&b, 1000, 1e-6, true)
        .unwrap();
    let elapsed = start.elapsed();
    println!("Solution found in {:.2?}", elapsed);

    // Compute residual
    println!("Computing residual...");
    let r = &matrix.dot(&x) - &b;
    let residual_norm = (r.dot(&r)).sqrt();
    let b_norm = (b.dot(&b)).sqrt();
    let relative_residual = residual_norm / b_norm;

    println!("Relative residual: {:.6e}", relative_residual);
    println!(
        "First few elements of solution: [{:.4}, {:.4}, {:.4}, ...]",
        x[0], x[1], x[2]
    );
}

/// Example comparing performance with a large matrix
fn example_large_matrix_performance(file_path: &Path) {
    let size = 1000;
    println!(
        "Creating a large {}x{} matrix (this would be much larger in a real scenario)...",
        size, size
    );

    // For demonstration purposes we'll use a relatively small "large" matrix
    // In a real scenario, this would be much larger (e.g., 100,000 x 100,000)
    let matrix = create_random_spd_matrix(size);

    // Create vectors for different bit-width chunked quantized matrices
    println!("Creating chunked quantized matrices with different bit widths...");

    // 8-bit quantization
    let file_path_8bit = file_path.with_file_name("matrix_8bit.bin");
    let start = Instant::now();
    let chunked_8bit = ChunkedQuantizedMatrix::new(
        &matrix.view(),
        8,
        QuantizationMethod::Symmetric,
        file_path_8bit.to_str().unwrap(),
    )
    .unwrap()
    .symmetric()
    .positive_definite();
    println!("8-bit quantization took {:.2?}", start.elapsed());

    // 4-bit quantization
    let file_path_4bit = file_path.with_file_name("matrix_4bit.bin");
    let start = Instant::now();
    let chunked_4bit = ChunkedQuantizedMatrix::new(
        &matrix.view(),
        4,
        QuantizationMethod::Symmetric,
        file_path_4bit.to_str().unwrap(),
    )
    .unwrap()
    .symmetric()
    .positive_definite();
    println!("4-bit quantization took {:.2?}", start.elapsed());

    // Create a right-hand side vector
    println!("Creating right-hand side vector...");
    let mut b = Array1::zeros(size);
    for i in 0..size {
        b[i] = (i % 10) as f32;
    }

    // Solve using 8-bit quantization
    println!("\nSolving with 8-bit quantization...");
    let start = Instant::now();
    let x_8bit = chunked_8bit
        .solve_conjugate_gradient(&b, 1000, 1e-6, true)
        .unwrap();
    let elapsed_8bit = start.elapsed();
    println!("Solution found in {:.2?}", elapsed_8bit);

    let r_8bit = &matrix.dot(&x_8bit) - &b;
    let residual_norm_8bit = (r_8bit.dot(&r_8bit)).sqrt();
    let b_norm = (b.dot(&b)).sqrt();
    let relative_residual_8bit = residual_norm_8bit / b_norm;
    println!("Relative residual: {:.6e}", relative_residual_8bit);

    // Solve using 4-bit quantization
    println!("\nSolving with 4-bit quantization...");
    let start = Instant::now();
    let x_4bit = chunked_4bit
        .solve_conjugate_gradient(&b, 1000, 1e-6, true)
        .unwrap();
    let elapsed_4bit = start.elapsed();
    println!("Solution found in {:.2?}", elapsed_4bit);

    let r_4bit = &matrix.dot(&x_4bit) - &b;
    let residual_norm_4bit = (r_4bit.dot(&r_4bit)).sqrt();
    let relative_residual_4bit = residual_norm_4bit / b_norm;
    println!("Relative residual: {:.6e}", relative_residual_4bit);

    // Calculate file sizes
    let file_size_8bit = std::fs::metadata(&file_path_8bit)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    let file_size_4bit = std::fs::metadata(&file_path_4bit)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    println!("\nMemory usage comparison:");
    println!("8-bit quantization file size: {:.2} MB", file_size_8bit);
    println!("4-bit quantization file size: {:.2} MB", file_size_4bit);
    println!(
        "Memory reduction with 4-bit: {:.1}%",
        (1.0 - file_size_4bit / file_size_8bit) * 100.0
    );

    println!("\nPerformance comparison:");
    println!("8-bit solve time: {:.2?}", elapsed_8bit);
    println!("4-bit solve time: {:.2?}", elapsed_4bit);
    println!(
        "Speed difference: {:.1}%",
        (elapsed_8bit.as_secs_f64() / elapsed_4bit.as_secs_f64() - 1.0) * 100.0
    );

    println!("\nAccuracy comparison:");
    println!("8-bit relative residual: {:.6e}", relative_residual_8bit);
    println!("4-bit relative residual: {:.6e}", relative_residual_4bit);
    println!(
        "Accuracy difference: {:.1}x",
        relative_residual_4bit / relative_residual_8bit
    );

    // Clean up
    std::fs::remove_file(&file_path_8bit).unwrap_or_default();
    std::fs::remove_file(&file_path_4bit).unwrap_or_default();
}

/// Create a random symmetric positive definite matrix of the given size
fn create_random_spd_matrix(size: usize) -> Array2<f32> {
    // Create a random matrix with values in the range [-1.0, 1.0)
    let mut rng = rand::rng();
    let mut matrix = Array2::zeros((size, size));

    for i in 0..size {
        for j in 0..size {
            matrix[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }

    // Make it symmetric
    for i in 0..size {
        for j in i + 1..size {
            matrix[[j, i]] = matrix[[i, j]];
        }
    }

    // Add to the diagonal to ensure positive definiteness
    for i in 0..size {
        matrix[[i, i]] += size as f32;
    }

    matrix
}

/// Helper to get a temporary file path
fn get_temp_file_path(name: &str) -> PathBuf {
    let mut path = env::temp_dir();
    path.push(format!("quantized_matrix_{}.bin", name));
    path
}
