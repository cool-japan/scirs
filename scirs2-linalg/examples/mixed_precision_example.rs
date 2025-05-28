//! Mixed-precision linear algebra example
//!
//! This example demonstrates the use of mixed-precision operations
//! to balance accuracy and performance in scientific computing.

use ndarray::{array, Array1, Array2};
use scirs2_linalg::prelude::*;

// Simple benchmark function to compare execution times
fn benchmark_fn<F>(name: &str, mut f: F) -> std::time::Duration
where
    F: FnMut(),
{
    use std::time::Instant;

    // Warm up
    for _ in 0..5 {
        f();
    }

    // Actual measurement
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        f();
    }

    let duration = start.elapsed() / iterations;
    println!("{}: {:?} per iteration", name, duration);

    duration
}

fn main() {
    println!("Mixed-Precision Linear Algebra Examples");
    println!("=======================================\n");

    // Example 1: Basic type conversion
    println!("Example 1: Basic type conversion");
    println!("---------------------------------");

    // Create a high-precision array
    let arr_f64 = array![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Original f64 array: {:?}", arr_f64);

    // Convert to lower precision
    let arr_f32 = convert::<f64, f32>(&arr_f64.view());
    println!("Converted to f32: {:?}", arr_f32);

    // Convert back to higher precision
    let arr_f64_again = convert::<f32, f64>(&arr_f32.view());
    println!("Converted back to f64: {:?}", arr_f64_again);

    // Show that the conversion is lossless for these simple values
    println!("Conversion error: {:?}", &arr_f64 - &arr_f64_again);
    println!();

    // Example 2: 2D array conversion
    println!("Example 2: 2D array conversion");
    println!("------------------------------");

    // Create a high-precision 2D array
    let mat_f64 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
    println!("Original f64 matrix:\n{:?}", mat_f64);

    // Convert to lower precision
    let mat_f32 = convert_2d::<f64, f32>(&mat_f64.view());
    println!("Converted to f32:\n{:?}", mat_f32);
    println!();

    // Example 3: Mixed-precision matrix-vector multiplication
    println!("Example 3: Mixed-precision matrix-vector multiplication");
    println!("------------------------------------------------------");

    // Create a matrix and vector in f32 precision
    let a_f32 = array![
        [1.0f32, 2.0f32, 3.0f32],
        [4.0f32, 5.0f32, 6.0f32],
        [7.0f32, 8.0f32, 9.0f32],
    ];
    let x_f32 = array![0.1f32, 0.2f32, 0.3f32];

    println!("Matrix A (f32):\n{:?}", a_f32);
    println!("Vector x (f32):\n{:?}", x_f32);

    // Compute in f32
    let y_f32 = a_f32.dot(&x_f32);
    println!("Standard computation (f32):\n{:?}", y_f32);

    // Compute with internal f64 precision but result in f32
    let y_mp = mixed_precision_matvec::<f32, f32, f32, f64>(&a_f32.view(), &x_f32.view()).unwrap();
    println!("Mixed-precision computation (f32->f64->f32):\n{:?}", y_mp);

    // Check difference
    println!("Difference: {:?}", &y_f32 - &y_mp);
    println!();

    // Example 4: Mixed-precision solving of linear systems
    println!("Example 4: Mixed-precision linear system solver");
    println!("----------------------------------------------");

    // Create an ill-conditioned system in f32 precision
    // Hilbert matrix is notoriously ill-conditioned
    let mut hilbert_f32 = Array2::<f32>::zeros((5, 5));
    for i in 0..5 {
        for j in 0..5 {
            hilbert_f32[[i, j]] = 1.0 / ((i + j + 1) as f32);
        }
    }

    // Create a simple right-hand side
    let b_f32 = array![1.0f32, 0.0, 0.0, 0.0, 0.0];

    println!("Hilbert matrix (5x5, f32):");
    for i in 0..5 {
        println!("{:?}", hilbert_f32.row(i));
    }
    println!("Right-hand side: {:?}", b_f32);

    // First solve using pure f32 precision
    let x_pure_f32 =
        mixed_precision_solve::<f32, f32, f32, f32>(&hilbert_f32.view(), &b_f32.view()).unwrap();

    // Then solve using internal f64 precision
    let x_mp =
        mixed_precision_solve::<f32, f32, f32, f64>(&hilbert_f32.view(), &b_f32.view()).unwrap();

    println!("Solution with pure f32 precision: {:?}", x_pure_f32);
    println!("Solution with mixed precision (internal f64): {:?}", x_mp);

    // Check how well the solutions satisfy A*x = b
    let ax_f32 = hilbert_f32.dot(&x_pure_f32);
    let ax_mp = hilbert_f32.dot(&x_mp);

    println!("Residual with pure f32: {:?}", &ax_f32 - &b_f32);
    println!("Residual with mixed precision: {:?}", &ax_mp - &b_f32);

    // Compute residual norms to see which solution is more accurate
    let residual_norm_f32 = (&ax_f32 - &b_f32)
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let residual_norm_mp = (&ax_mp - &b_f32).iter().map(|&x| x * x).sum::<f32>().sqrt();

    println!("Residual norm (pure f32): {:.6e}", residual_norm_f32);
    println!("Residual norm (mixed precision): {:.6e}", residual_norm_mp);
    println!(
        "Improvement factor: {:.2}x",
        residual_norm_f32 / residual_norm_mp
    );
    println!();

    // Example 5: Working with very small or large numbers
    println!("Example 5: Working with very small or large numbers");
    println!("--------------------------------------------------");

    // Create a matrix with small and large values
    let a_extreme = array![[1.0e-6f32, 1.0e6f32], [1.0e6f32, 1.0e-6f32],];
    let b_extreme = array![1.0f32, 1.0f32];

    println!("Matrix with extreme values (f32):\n{:?}", a_extreme);
    println!("Right-hand side: {:?}", b_extreme);

    // Solve with pure f32
    let x_f32_extreme =
        mixed_precision_solve::<f32, f32, f32, f32>(&a_extreme.view(), &b_extreme.view()).unwrap();

    // Solve with internal f64
    let x_mp_extreme =
        mixed_precision_solve::<f32, f32, f32, f64>(&a_extreme.view(), &b_extreme.view()).unwrap();

    println!("Solution with pure f32: {:?}", x_f32_extreme);
    println!("Solution with mixed precision: {:?}", x_mp_extreme);

    // Check residuals
    let ax_f32_extreme = a_extreme.dot(&x_f32_extreme);
    let ax_mp_extreme = a_extreme.dot(&x_mp_extreme);

    println!("Residual with pure f32: {:?}", &ax_f32_extreme - &b_extreme);
    println!(
        "Residual with mixed precision: {:?}",
        &ax_mp_extreme - &b_extreme
    );

    // Example 6: Matrix multiplication with mixed precision
    println!("Example 6: Matrix multiplication with mixed precision");
    println!("-------------------------------------------------");

    // Create two matrices in f32 precision
    let a_mult = array![[1.0e-4f32, 1.0e4f32], [1.0e4f32, 1.0e-4f32]];

    let b_mult = array![[1.0e-4f32, 1.0e4f32], [1.0e4f32, 1.0e-4f32]];

    println!("Matrix A:\n{:?}", a_mult);
    println!("Matrix B:\n{:?}", b_mult);

    // Compute product with standard f32 precision
    let c_f32 = a_mult.dot(&b_mult);
    println!("Standard computation (f32):\n{:?}", c_f32);

    // Compute with internal f64 precision but result in f32
    let c_mp =
        mixed_precision_matmul::<f32, f32, f32, f64>(&a_mult.view(), &b_mult.view()).unwrap();
    println!("Mixed-precision computation (f32->f64->f32):\n{:?}", c_mp);

    // Check difference
    println!("Difference: {:?}", &c_f32 - &c_mp);
    println!();

    // Example 7: Condition number computation with mixed precision
    println!("Example 7: Condition number computation with mixed precision");
    println!("----------------------------------------------------------");

    // Create increasingly ill-conditioned matrices
    println!("Condition numbers of matrices with different conditioning:");

    // Well-conditioned matrix
    let a_well = array![[3.0f32, 2.0f32], [1.0f32, 5.0f32]];

    // Moderately ill-conditioned matrix
    let a_moderate = array![[1.0f32, 2.0f32], [1.001f32, 2.001f32]];

    // Severely ill-conditioned matrix (nearly singular)
    let a_severe = array![[1.0f32, 2.0f32], [1.0f32, 2.0000001f32]];

    // Compute condition numbers with both f32 and f64 internal precision
    let cond_well_f32 = mixed_precision_cond::<f32, f32, f32>(&a_well.view(), None).unwrap();
    let cond_well_f64 = mixed_precision_cond::<f32, f32, f64>(&a_well.view(), None).unwrap();

    let cond_moderate_f32 =
        mixed_precision_cond::<f32, f32, f32>(&a_moderate.view(), None).unwrap();
    let cond_moderate_f64 =
        mixed_precision_cond::<f32, f32, f64>(&a_moderate.view(), None).unwrap();

    let cond_severe_f32 = mixed_precision_cond::<f32, f32, f32>(&a_severe.view(), None).unwrap();
    let cond_severe_f64 = mixed_precision_cond::<f32, f32, f64>(&a_severe.view(), None).unwrap();

    println!("Well-conditioned matrix:");
    println!("  f32 precision: {:.2e}", cond_well_f32);
    println!("  f64 precision: {:.2e}", cond_well_f64);
    println!(
        "  Relative difference: {:.2}%",
        100.0 * (cond_well_f64 - cond_well_f32).abs() / cond_well_f32
    );
    println!();

    println!("Moderately ill-conditioned matrix:");
    println!("  f32 precision: {:.2e}", cond_moderate_f32);
    println!("  f64 precision: {:.2e}", cond_moderate_f64);
    println!(
        "  Relative difference: {:.2}%",
        100.0 * (cond_moderate_f64 - cond_moderate_f32).abs() / cond_moderate_f32
    );
    println!();

    println!("Severely ill-conditioned matrix:");
    println!("  f32 precision: {:.2e}", cond_severe_f32);
    println!("  f64 precision: {:.2e}", cond_severe_f64);
    println!(
        "  Relative difference: {:.2}%",
        100.0 * (cond_severe_f64 - cond_severe_f32).abs() / cond_severe_f32
    );
    println!();

    println!("\nConclusion:");
    println!("Mixed-precision operations can significantly improve accuracy");
    println!("for numerically challenging problems while maintaining the");
    println!("memory efficiency of lower precision storage.");
    println!();
    println!("This is particularly important for:");
    println!("1. Ill-conditioned linear systems");
    println!("2. Matrices with very large condition numbers");
    println!("3. Computations involving very small or very large numbers");
    println!("4. Maintaining both speed and accuracy in scientific computing");

    // Example 8: Performance comparison
    println!("\nExample 8: Performance comparison");
    println!("--------------------------------");

    // Create a larger matrix and vector for meaningful benchmarks
    let size = 500; // 500x500 matrix
    let large_matrix =
        Array2::<f32>::from_shape_fn((size, size), |(i, j)| if i == j { 1.0 } else { 0.01 });

    let large_vector = Array1::<f32>::from_shape_fn(size, |i| (i % 10) as f32 / 10.0);

    println!("Benchmarking with {}x{} matrix", size, size);
    println!("----------------------------------");

    // Benchmark matrix-vector multiplication
    println!("\nMatrix-vector multiplication:");

    let f32_time = benchmark_fn("Standard f32", || {
        let _ = large_matrix.dot(&large_vector);
    });

    let mp_time = benchmark_fn("Mixed precision (f32->f64->f32)", || {
        let _ = mixed_precision_matvec::<f32, f32, f32, f64>(
            &large_matrix.view(),
            &large_vector.view(),
        )
        .unwrap();
    });

    let speedup = f32_time.as_nanos() as f64 / mp_time.as_nanos() as f64;
    println!("Relative performance: {:.2}x", speedup);

    // Create an ill-conditioned matrix for solver benchmarks
    println!("\nLinear system solving (ill-conditioned):");

    // Create a simple, slightly ill-conditioned matrix
    let matrix_size = 10;
    let mut test_matrix = Array2::<f32>::eye(matrix_size);
    // Add small off-diagonal elements to make it slightly ill-conditioned
    // but not singular
    for i in 0..matrix_size - 1 {
        test_matrix[[i, i + 1]] = 0.1;
        test_matrix[[i + 1, i]] = 0.1;
    }

    let b_large = Array1::<f32>::from_shape_fn(matrix_size, |i| if i == 0 { 1.0 } else { 0.0 });

    let f32_solve_time = benchmark_fn("Pure f32 solver", || {
        let _ = mixed_precision_solve::<f32, f32, f32, f32>(&test_matrix.view(), &b_large.view())
            .unwrap();
    });

    let mp_solve_time = benchmark_fn("Mixed precision solver (f64 internal)", || {
        let _ = mixed_precision_solve::<f32, f32, f32, f64>(&test_matrix.view(), &b_large.view())
            .unwrap();
    });

    let solve_speedup = f32_solve_time.as_nanos() as f64 / mp_solve_time.as_nanos() as f64;
    println!("Relative solver performance: {:.2}x", solve_speedup);

    println!("\nNOTE: The mixed-precision versions often take slightly longer");
    println!("due to type conversions, but provide better numerical accuracy,");
    println!("especially for ill-conditioned problems or when working with");
    println!("values of vastly different magnitudes.");

    // Example 9: SIMD-accelerated mixed precision operations
    #[cfg(feature = "simd")]
    {
        println!("\nExample 9: SIMD-accelerated mixed precision operations");
        println!("---------------------------------------------------");

        // Create test matrices and vectors with a mix of large and small values
        // to highlight precision issues
        let size = 1000;
        let a_vec = Array1::<f32>::from_shape_fn(size, |i| if i % 2 == 0 { 1.0e-6 } else { 1.0e6 });

        let b_vec = Array1::<f32>::from_shape_fn(size, |i| if i % 2 == 0 { 1.0e6 } else { 1.0e-6 });

        // Create smaller matrices for matmul demonstration
        let mat_size = 100;
        let a_mat = Array2::<f32>::from_shape_fn((mat_size, mat_size), |(i, j)| {
            if (i + j) % 2 == 0 {
                1.0e-6
            } else {
                1.0e6
            }
        });

        let b_mat = Array2::<f32>::from_shape_fn((mat_size, mat_size), |(i, j)| {
            if (i + j) % 3 == 0 {
                1.0e6
            } else {
                1.0e-6
            }
        });

        println!("\nComparing SIMD-accelerated vs standard mixed precision:");

        // Benchmark dot product operations
        println!("\nDot product (vector length: {}):", size);

        let standard_time = benchmark_fn("Standard f32 dot product", || {
            let _ = a_vec.dot(&b_vec);
        });

        let standard_mp_time = benchmark_fn("Regular mixed precision dot product", || {
            let _ =
                mixed_precision_dot::<f32, f32, f32, f64>(&a_vec.view(), &b_vec.view()).unwrap();
        });

        let simd_time = benchmark_fn("SIMD-accelerated mixed precision dot product", || {
            let _ = simd_mixed_precision_dot_f32_f64::<f32>(&a_vec.view(), &b_vec.view()).unwrap();
        });

        println!(
            "SIMD speedup vs standard f32: {:.2}x",
            standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );
        println!(
            "SIMD speedup vs regular mixed precision: {:.2}x",
            standard_mp_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        );

        // Benchmark matrix-vector multiplication
        let row_vec = a_mat.row(0).to_owned();

        println!(
            "\nMatrix-vector multiplication ({}x{} matrix):",
            mat_size, mat_size
        );

        let standard_mv_time = benchmark_fn("Standard f32 matrix-vector multiplication", || {
            let _ = a_mat.dot(&row_vec);
        });

        let standard_mp_mv_time = benchmark_fn(
            "Regular mixed precision matrix-vector multiplication",
            || {
                let _ =
                    mixed_precision_matvec::<f32, f32, f32, f64>(&a_mat.view(), &row_vec.view())
                        .unwrap();
            },
        );

        let simd_mv_time = benchmark_fn(
            "SIMD-accelerated mixed precision matrix-vector multiplication",
            || {
                let _ = simd_mixed_precision_matvec_f32_f64::<f32>(&a_mat.view(), &row_vec.view())
                    .unwrap();
            },
        );

        println!(
            "SIMD speedup vs standard f32: {:.2}x",
            standard_mv_time.as_nanos() as f64 / simd_mv_time.as_nanos() as f64
        );
        println!(
            "SIMD speedup vs regular mixed precision: {:.2}x",
            standard_mp_mv_time.as_nanos() as f64 / simd_mv_time.as_nanos() as f64
        );

        // Benchmark matrix multiplication
        println!(
            "\nMatrix multiplication ({}x{} matrices):",
            mat_size, mat_size
        );

        let standard_mm_time = benchmark_fn("Standard f32 matrix multiplication", || {
            let _ = a_mat.dot(&b_mat);
        });

        let standard_mp_mm_time =
            benchmark_fn("Regular mixed precision matrix multiplication", || {
                let _ = mixed_precision_matmul::<f32, f32, f32, f64>(&a_mat.view(), &b_mat.view())
                    .unwrap();
            });

        let simd_mm_time = benchmark_fn(
            "SIMD-accelerated mixed precision matrix multiplication",
            || {
                let _ = simd_mixed_precision_matmul_f32_f64::<f32>(&a_mat.view(), &b_mat.view())
                    .unwrap();
            },
        );

        println!(
            "SIMD speedup vs standard f32: {:.2}x",
            standard_mm_time.as_nanos() as f64 / simd_mm_time.as_nanos() as f64
        );
        println!(
            "SIMD speedup vs regular mixed precision: {:.2}x",
            standard_mp_mm_time.as_nanos() as f64 / simd_mm_time.as_nanos() as f64
        );

        // Check accuracy differences
        println!("\nAccuracy comparison:");

        // Create vectors with values that benefit from higher precision
        let a_precision = array![1.0e-7f32, 2.0e7, 3.0e-7, 4.0e7, 5.0e-7, 6.0e7, 7.0e-7, 8.0e7];
        let b_precision = array![9.0e-7f32, 8.0e7, 7.0e-7, 6.0e7, 5.0e-7, 4.0e7, 3.0e-7, 2.0e7];

        // Compute standard f32 dot product
        let dot_f32 = a_precision
            .iter()
            .zip(b_precision.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f32>();

        // Compute mixed precision dot product
        let dot_simd_f64 =
            simd_mixed_precision_dot_f32_f64::<f64>(&a_precision.view(), &b_precision.view())
                .unwrap();

        // Compute "ground truth" with explicit f64 conversions
        let ground_truth = a_precision
            .iter()
            .zip(b_precision.iter())
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum::<f64>();

        println!("Standard f32 result: {:.10e}", dot_f32);
        println!("SIMD mixed precision result: {:.10e}", dot_simd_f64);
        println!("Ground truth (manual f64): {:.10e}", ground_truth);

        println!(
            "Relative error (standard f32): {:.10e}",
            ((dot_f32 as f64) - ground_truth).abs() / ground_truth
        );
        println!(
            "Relative error (SIMD mixed precision): {:.10e}",
            (dot_simd_f64 - ground_truth).abs() / ground_truth
        );

        // Summary
        println!("\nSIMD-accelerated mixed precision provides:");
        println!("1. Better performance than regular mixed precision");
        println!("2. Better numerical accuracy than standard f32");
        println!("3. Efficient handling of values with large dynamic range");
        println!("4. Ideal for performance-critical scientific computing applications");
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("\nExample 9: SIMD-accelerated operations (not available)");
        println!("---------------------------------------------------");
        println!("To enable SIMD-accelerated mixed-precision operations,");
        println!("rebuild with the 'simd' feature enabled:");
        println!("    cargo build --features=\"simd\"");
        println!("    cargo run --example mixed_precision_example --features=\"simd\"");
    }
}
