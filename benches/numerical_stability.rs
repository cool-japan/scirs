use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, ArrayView2};
use rand::distr::Uniform;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scirs2_linalg::{det, inv, lu, matrix_norm, qr, solve, svd, vector_norm, LinalgError};
use serde::{Deserialize, Serialize};
use std::f64;
use std::fs;
use std::hint::black_box;

const SEED: u64 = 42;

#[derive(Serialize, Deserialize)]
struct StabilityTestResult {
    test_name: String,
    matrix_size: usize,
    condition_number: f64,
    success: bool,
    relative_error: Option<f64>,
    computation_time_ns: u64,
    notes: String,
}

/// Generate matrices with specific condition numbers for stability testing
#[allow(dead_code)]
fn generate_conditioned_matrix(size: usize, condition_number: f64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    // Generate random orthogonal matrices U and V via QR decomposition
    let uniform = Uniform::new(-1.0, 1.0).unwrap();
    let u_raw = Array2::from_shape_fn((size, size), |_| rng.sample(uniform));
    let v_raw = Array2::from_shape_fn((size, size), |_| rng.sample(uniform));

    let (u, _) = qr(&u_raw.view(), None).unwrap();
    let (v, _) = qr(&v_raw.view(), None).unwrap();

    // Create singular values with specified condition number
    let mut singular_values = Array1::linspace(1.0, 1.0 / condition_number, size);

    // Ensure all singular values are positive and well-ordered
    for i in 0..size {
        singular_values[i] = singular_values[i].abs().max(1e-16);
    }

    // Construct matrix A = U * Σ * V^T
    let mut result = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                result[[i, j]] += u[[i, k]] * singular_values[k] * v[[j, k]];
            }
        }
    }

    result
}

/// Generate matrices with specific pathological properties
#[allow(dead_code)]
fn generate_pathological_matrix(size: usize, test_type: &str) -> Array2<f64> {
    match test_type {
        "hilbert" => {
            // Hilbert matrix - notoriously ill-conditioned
            let mut matrix = Array2::zeros((size, size));
            for i in 0..size {
                for j in 0..size {
                    matrix[[i, j]] = 1.0 / ((i + j + 1) as f64);
                }
            }
            matrix
        }
        "vandermonde" => {
            // Vandermonde matrix - also ill-conditioned
            let mut matrix = Array2::zeros((size, size));
            let points: Vec<f64> = (0..size).map(|i| i as f64 / (size - 1) as f64).collect();

            for i in 0..size {
                for j in 0..size {
                    matrix[[i, j]] = points[i].powi(j as i32);
                }
            }
            matrix
        }
        "near_singular" => {
            // Nearly singular matrix
            let mut matrix = Array2::eye(size);
            matrix[[size - 1, size - 1]] = 1e-14; // Make the last diagonal element tiny
            matrix
        }
        "rank_deficient" => {
            // Rank deficient matrix
            let mut matrix = Array2::zeros((size, size));
            // Only fill first (size-1) columns to make it rank deficient
            for i in 0..size {
                for j in 0..(size - 1) {
                    matrix[[i, j]] = ((i + j + 1) as f64).sin();
                }
            }
            matrix
        }
        _ => Array2::eye(size), // Default to identity
    }
}

/// Test numerical accuracy of linear solvers
#[allow(dead_code)]
fn test_solve_accuracy(matrix: &ArrayView2<f64>, known_solution: &Array1<f64>) -> (bool, f64) {
    let rhs = matrix.dot(known_solution);

    match solve(matrix, &rhs.view(), None) {
        Ok(computed_solution) => {
            let error = &computed_solution - known_solution;
            let relative_error = vector_norm(&error.view(), 2).unwrap()
                / vector_norm(&known_solution.view(), 2).unwrap();

            let success = relative_error < 1e-10; // Tolerance for success
            (success, relative_error)
        }
        Err(_) => (false, f64::INFINITY),
    }
}

/// Test numerical accuracy of matrix inversion
#[allow(dead_code)]
fn test_inverse_accuracy(matrix: &ArrayView2<f64>) -> (bool, f64) {
    match inv(matrix, None) {
        Ok(inv_matrix) => {
            let product = matrix.dot(&inv_matrix);
            let identity = Array2::eye(matrix.nrows());
            let error: Array2<f64> = &product - &identity;
            let relative_error = matrix_norm::<f64>(&error.view(), "frobenius", None).unwrap()
                / matrix_norm::<f64>(&identity.view(), "frobenius", None).unwrap();

            let success = relative_error < 1e-10;
            (success, relative_error)
        }
        Err(_) => (false, f64::INFINITY),
    }
}

/// Test numerical accuracy of matrix decompositions
#[allow(dead_code)]
fn test_decomposition_accuracy(matrix: &ArrayView2<f64>, decomp_type: &str) -> (bool, f64) {
    match decomp_type {
        "lu" => match lu(matrix, None) {
            Ok((p, l, u)) => {
                let reconstructed = p.dot(&l).dot(&u);
                let error: Array2<f64> = &reconstructed - matrix;
                let relative_error = matrix_norm(&error.view(), "frobenius", None).unwrap()
                    / matrix_norm(matrix, "frobenius", None).unwrap();
                (relative_error < 1e-12, relative_error)
            }
            Err(_) => (false, f64::INFINITY),
        },
        "qr" => match qr(matrix, None) {
            Ok((q, r)) => {
                let reconstructed = q.dot(&r);
                let error: Array2<f64> = &reconstructed - matrix;
                let relative_error = matrix_norm(&error.view(), "frobenius", None).unwrap()
                    / matrix_norm(matrix, "frobenius", None).unwrap();
                (relative_error < 1e-12, relative_error)
            }
            Err(_) => (false, f64::INFINITY),
        },
        "svd" => {
            match svd(matrix, false, None) {
                Ok((u, s, vt)) => {
                    // Reconstruct matrix: A = U * Σ * V^T
                    let sigma = Array2::from_diag(&s);
                    let reconstructed = u.dot(&sigma).dot(&vt);
                    let error: Array2<f64> = &reconstructed - matrix;
                    let relative_error = matrix_norm(&error.view(), "frobenius", None).unwrap()
                        / matrix_norm(matrix, "frobenius", None).unwrap();
                    (relative_error < 1e-12, relative_error)
                }
                Err(_) => (false, f64::INFINITY),
            }
        }
        _ => (false, f64::INFINITY),
    }
}

/// Benchmark numerical stability with well-conditioned matrices
#[allow(dead_code)]
fn bench_well_conditioned_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("well_conditioned_stability");
    let mut results = Vec::new();

    let sizes = vec![50, 100, 200];
    let condition_numbers = vec![1e2, 1e4, 1e6];

    for size in sizes {
        for cond_num in &condition_numbers {
            let matrix = generate_conditioned_matrix(size, *cond_num);
            let known_solution = Array1::ones(size);

            // Test linear solver stability
            group.bench_with_input(
                BenchmarkId::new(format!("solve_cond_{:.0e}", cond_num), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let start = std::time::Instant::now();
                        let (success, error) = test_solve_accuracy(&matrix.view(), &known_solution);
                        let elapsed = start.elapsed().as_nanos() as u64;

                        results.push(StabilityTestResult {
                            test_name: format!("solve_cond_{:.0e}", cond_num),
                            matrix_size: size,
                            condition_number: *cond_num,
                            success,
                            relative_error: if error.is_finite() { Some(error) } else { None },
                            computation_time_ns: elapsed,
                            notes: if success {
                                "Converged".to_string()
                            } else {
                                "Failed".to_string()
                            },
                        });

                        black_box((success, error))
                    })
                },
            );

            // Test matrix inversion stability
            if size <= 100 {
                // Limit inversion tests to smaller matrices
                group.bench_with_input(
                    BenchmarkId::new(format!("inverse_cond_{:.0e}", cond_num), size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let start = std::time::Instant::now();
                            let (success, error) = test_inverse_accuracy(&matrix.view());
                            let elapsed = start.elapsed().as_nanos() as u64;

                            results.push(StabilityTestResult {
                                test_name: format!("inverse_cond_{:.0e}", cond_num),
                                matrix_size: size,
                                condition_number: *cond_num,
                                success,
                                relative_error: if error.is_finite() { Some(error) } else { None },
                                computation_time_ns: elapsed,
                                notes: if success {
                                    "Converged".to_string()
                                } else {
                                    "Failed".to_string()
                                },
                            });

                            black_box((success, error))
                        })
                    },
                );
            }
        }
    }

    // Save stability test results
    save_stability_results(&results);
    group.finish();
}

/// Benchmark numerical stability with pathological matrices
#[allow(dead_code)]
fn bench_pathological_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathological_matrices");
    let mut results = Vec::new();

    let sizes = vec![20, 50, 100];
    let matrix_types = vec!["hilbert", "vandermonde", "near_singular", "rank_deficient"];

    for size in sizes {
        for matrix_type in &matrix_types {
            let matrix = generate_pathological_matrix(size, matrix_type);
            let known_solution = Array1::ones(size);

            group.bench_with_input(
                BenchmarkId::new(format!("solve_{}", matrix_type), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let start = std::time::Instant::now();
                        let (success, error) = test_solve_accuracy(&matrix.view(), &known_solution);
                        let elapsed = start.elapsed().as_nanos() as u64;

                        // Estimate condition number for pathological matrices
                        let cond_estimate = estimate_condition_number(&matrix.view());

                        results.push(StabilityTestResult {
                            test_name: format!("solve_{}", matrix_type),
                            matrix_size: size,
                            condition_number: cond_estimate,
                            success,
                            relative_error: if error.is_finite() { Some(error) } else { None },
                            computation_time_ns: elapsed,
                            notes: format!("Matrix type: {}", matrix_type),
                        });

                        black_box((success, error))
                    })
                },
            );
        }
    }

    save_stability_results(&results);
    group.finish();
}

/// Benchmark decomposition stability
#[allow(dead_code)]
fn bench_decomposition_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition_stability");
    let mut results = Vec::new();

    let sizes = vec![50, 100];
    let decomp_types = vec!["lu", "qr", "svd"];
    let condition_numbers = vec![1e4, 1e8, 1e12];

    for size in sizes {
        for decomp_type in &decomp_types {
            for cond_num in &condition_numbers {
                let matrix = generate_conditioned_matrix(size, *cond_num);

                group.bench_with_input(
                    BenchmarkId::new(format!("{}_{:.0e}", decomp_type, cond_num), size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let start = std::time::Instant::now();
                            let (success, error) =
                                test_decomposition_accuracy(&matrix.view(), decomp_type);
                            let elapsed = start.elapsed().as_nanos() as u64;

                            results.push(StabilityTestResult {
                                test_name: format!("{}_{:.0e}", decomp_type, cond_num),
                                matrix_size: size,
                                condition_number: *cond_num,
                                success,
                                relative_error: if error.is_finite() { Some(error) } else { None },
                                computation_time_ns: elapsed,
                                notes: format!("Decomposition: {}", decomp_type),
                            });

                            black_box((success, error))
                        })
                    },
                );
            }
        }
    }

    save_stability_results(&results);
    group.finish();
}

/// Test edge cases and extreme values
#[allow(dead_code)]
fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");
    let mut results = Vec::new();

    // Test with very small matrices
    let tiny_matrix = Array2::from_shape_vec((2, 2), vec![1e-15, 1e-14, 1e-14, 1e-13]).unwrap();

    group.bench_function("tiny_matrix_det", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = det(&tiny_matrix.view(), None);
            let elapsed = start.elapsed().as_nanos() as u64;

            let success = result.is_ok();
            results.push(StabilityTestResult {
                test_name: "tiny_matrix_det".to_string(),
                matrix_size: 2,
                condition_number: f64::INFINITY,
                success,
                relative_error: None,
                computation_time_ns: elapsed,
                notes: "Tiny values test".to_string(),
            });

            black_box(result)
        })
    });

    // Test with very large values
    let large_matrix = Array2::from_shape_vec((2, 2), vec![1e15, 1e14, 1e14, 1e13]).unwrap();

    group.bench_function("large_matrix_det", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result: Result<f64, LinalgError> = det(&large_matrix.view(), None);
            let elapsed = start.elapsed().as_nanos() as u64;

            let success = result.is_ok() && result.as_ref().unwrap().is_finite();
            results.push(StabilityTestResult {
                test_name: "large_matrix_det".to_string(),
                matrix_size: 2,
                condition_number: f64::INFINITY,
                success,
                relative_error: None,
                computation_time_ns: elapsed,
                notes: "Large values test".to_string(),
            });

            black_box(result)
        })
    });

    save_stability_results(&results);
    group.finish();
}

/// Estimate condition number using singular values
#[allow(dead_code)]
fn estimate_condition_number(matrix: &ArrayView2<f64>) -> f64 {
    match svd(matrix, false, None) {
        Ok((_, s, _)) => {
            let max_sv = s.iter().cloned().fold(0.0, f64::max);
            let min_sv = s.iter().cloned().fold(f64::INFINITY, f64::min);
            if min_sv > 0.0 {
                max_sv / min_sv
            } else {
                f64::INFINITY
            }
        }
        Err(_) => f64::INFINITY,
    }
}

/// Save stability test results to JSON file
#[allow(dead_code)]
fn save_stability_results(results: &[StabilityTestResult]) {
    std::fs::create_dir_all("target").unwrap_or_default();

    let json = serde_json::to_string_pretty(results).unwrap();
    fs::write("target/stability_test_results.json", json).unwrap_or_else(|e| {
        eprintln!("Failed to save stability test results: {}", e);
    });

    // Print summary
    let total_tests = results.len();
    let successful_tests = results.iter().filter(|r| r.success).count();
    let success_rate = (successful_tests as f64 / total_tests as f64) * 100.0;

    println!("\n=== Numerical Stability Summary ===");
    println!("Total tests: {}", total_tests);
    println!("Successful tests: {}", successful_tests);
    println!("Success rate: {:.1}%", success_rate);

    // Identify problematic condition numbers
    let mut failed_high_cond = 0;
    for result in results {
        if !result.success && result.condition_number > 1e10 {
            failed_high_cond += 1;
        }
    }

    if failed_high_cond > 0 {
        println!(
            "Failed tests with high condition numbers (>1e10): {}",
            failed_high_cond
        );
    }
}

criterion_group!(
    benches,
    bench_well_conditioned_stability,
    bench_pathological_matrices,
    bench_decomposition_stability,
    bench_edge_cases
);

criterion_main!(benches);
