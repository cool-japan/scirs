//! Utility functions for matrix operations using work-stealing
//!
//! This module provides various utility functions including band matrix solvers,
//! matrix function computations, batch operations, and specialized norm computations.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, One, Zero};
use std::iter::Sum;

use super::super::core::{BandSolveWorkItem, WorkItem};
use super::super::scheduler::WorkStealingScheduler;
use super::gemm::parallel_gemm_work_stealing;

/// Parallel Band matrix solver with optimized memory access
pub fn parallel_band_solve<F>(
    bandmatrix: &ArrayView2<F>,
    rhs: &ArrayView1<F>,
    bandwidth: usize,
    num_workers: usize,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = bandmatrix.nrows();
    if n != rhs.len() {
        return Err(LinalgError::ShapeError(
            "Matrix and RHS dimensions don't match".to_string(),
        ));
    }

    let mut x = rhs.to_owned();
    let scheduler = WorkStealingScheduler::new(num_workers);

    // Forward substitution with parallel band processing
    for i in 0..n {
        let start_j = i.saturating_sub(bandwidth);
        let end_j = (i + bandwidth + 1).min(n);

        if end_j > i + 1 {
            let work_items: Vec<BandSolveWorkItem<F>> = ((i + 1)..end_j)
                .map(|j| WorkItem::new(j, (i, j, start_j, bandmatrix.to_owned(), x.clone())))
                .collect();

            if !work_items.is_empty() {
                scheduler.submit_work(work_items)?;
                let results = scheduler.execute(move |(i, j, start_j, matrix, x_vec)| {
                    let mut sum = F::zero();
                    for k in start_j..i {
                        sum += matrix[(j, k)] * x_vec[k];
                    }
                    (j, sum)
                })?;

                // Update x vector
                for (j, sum) in results {
                    x[j] -= sum / bandmatrix[(j, j)];
                }
            }
        }
    }

    Ok(x)
}

/// Parallel matrix exponential computation using work stealing
///
/// Computes the matrix exponential exp(A) using parallel scaling and squaring
/// method with Padé approximation.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads
///
/// # Returns
///
/// * Matrix exponential exp(A)
pub fn parallel_matrix_exponential_work_stealing<F>(
    a: &ArrayView2<F>,
    workers: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for matrix exponential".to_string(),
        ));
    }

    // For small matrices, use sequential algorithm
    if n < 32 || workers == 1 {
        return crate::matrix_functions::expm(a, None);
    }

    // Compute matrix norm for scaling
    let norm_a = crate::norm::matrix_norm(a, "1", Some(workers))?;
    let log2_norm = norm_a.ln() / F::from(2.0).unwrap().ln();
    let scaling_factor = log2_norm.ceil().max(F::zero()).to_usize().unwrap_or(0);

    // Scale matrix
    let scaled_factor = F::from(2.0).unwrap().powi(-(scaling_factor as i32));
    let mut scaled_matrix = a.to_owned();
    scaled_matrix *= scaled_factor;

    // Parallel Padé approximation
    let result = parallel_pade_approximation(&scaled_matrix.view(), 13, workers)?;

    // Square the result `scaling_factor` times
    let mut final_result = result;
    for _ in 0..scaling_factor {
        final_result =
            parallel_gemm_work_stealing(&final_result.view(), &final_result.view(), workers)?;
    }

    Ok(final_result)
}

/// Parallel matrix square root computation using work stealing
///
/// Computes the matrix square root using parallel Newton-Schulz iteration.
///
/// # Arguments
///
/// * `a` - Input positive definite matrix
/// * `workers` - Number of worker threads
///
/// # Returns
///
/// * Matrix square root
pub fn parallel_matrix_sqrt_work_stealing<F>(
    a: &ArrayView2<F>,
    workers: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for matrix square root".to_string(),
        ));
    }

    // For small matrices, use sequential algorithm
    if n < 32 || workers == 1 {
        let max_iter = 20;
        let tolerance = F::epsilon().sqrt();
        return crate::matrix_functions::sqrtm(a, max_iter, tolerance);
    }

    // Initialize with scaled identity matrix
    let trace = (0..n).map(|i| a[[i, i]]).fold(F::zero(), |acc, x| acc + x);
    let initial_scaling = (trace / F::from(n).unwrap()).sqrt();
    let mut x = Array2::eye(n) * initial_scaling;
    let mut z = Array2::eye(n);

    let max_iterations = 20;
    let tolerance = F::epsilon().sqrt();

    for _iter in 0..max_iterations {
        // Newton-Schulz iteration with parallel matrix operations
        let x_squared = parallel_gemm_work_stealing(&x.view(), &x.view(), workers)?;
        let z_squared = parallel_gemm_work_stealing(&z.view(), &z.view(), workers)?;

        // Convergence check
        let error_matrix = &x_squared - a;
        let error_norm = parallel_matrix_norm_work_stealing(&error_matrix.view(), "fro", workers)?;

        if error_norm < tolerance {
            break;
        }

        // Update x and z using Newton-Schulz iteration
        let three = F::from(3.0).unwrap();
        let two = F::from(2.0).unwrap();

        // Create 3*I - Z² where I is identity matrix
        let three_i = Array2::eye(n) * three;
        let three_minus_z_squared = three_i - &z_squared;

        let temp_x = &x * &three_minus_z_squared / two;
        let temp_z = &z * &three_minus_z_squared / two;

        x = temp_x;
        z = temp_z;
    }

    Ok(x)
}

/// Parallel batch matrix operations using work stealing
///
/// Performs the same operation on multiple matrices in parallel.
///
/// # Arguments
///
/// * `matrices` - Vector of input matrices
/// * `operation` - Function to apply to each matrix
/// * `workers` - Number of worker threads
///
/// # Returns
///
/// * Vector of results
pub fn parallel_batch_operations_work_stealing<F, Op, R>(
    matrices: &[ArrayView2<F>],
    operation: Op,
    workers: usize,
) -> LinalgResult<Vec<R>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    Op: Fn(&ArrayView2<F>) -> LinalgResult<R> + Send + Sync,
    R: Send + Sync,
{
    if matrices.is_empty() {
        return Ok(Vec::new());
    }

    // For small batches, use sequential processing
    if matrices.len() < workers || workers == 1 {
        return matrices.iter().map(&operation).collect();
    }

    // Process matrices in parallel using chunks
    let chunk_size = matrices.len().div_ceil(workers);

    let results = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|worker_id| {
                let start_idx = worker_id * chunk_size;
                let end_idx = ((worker_id + 1) * chunk_size).min(matrices.len());
                let op_ref = &operation;

                s.spawn(move || {
                    matrices[start_idx..end_idx]
                        .iter()
                        .map(op_ref)
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .collect();

        let mut results = Vec::new();
        for handle in handles {
            let chunk_results = handle.join().unwrap()?;
            results.extend(chunk_results);
        }
        Ok::<Vec<R>, LinalgError>(results)
    })?;

    Ok(results)
}

/// Parallel specialized matrix norm computation using work stealing
///
/// Computes various matrix norms using parallel algorithms optimized
/// for different norm types.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `norm_type` - Type of norm ("fro", "nuc", "1", "2", "inf")
/// * `workers` - Number of worker threads
///
/// # Returns
///
/// * Computed norm value
pub fn parallel_matrix_norm_work_stealing<F>(
    a: &ArrayView2<F>,
    norm_type: &str,
    workers: usize,
) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    match norm_type {
        "fro" | "frobenius" => parallel_frobenius_norm(a, workers),
        "nuc" | "nuclear" => parallel_nuclear_norm(a, workers),
        "1" => parallel_matrix_1_norm(a, workers),
        "2" | "spectral" => parallel_spectral_norm(a, workers),
        "inf" | "infinity" => parallel_matrix_inf_norm(a, workers),
        _ => Err(LinalgError::InvalidInputError(format!(
            "Unknown norm type: {norm_type}"
        ))),
    }
}

/// Parallel Frobenius norm computation
pub fn parallel_frobenius_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let scheduler = WorkStealingScheduler::new(workers);

    // Create work items for chunks of the matrix
    let chunk_size = (m * n / workers).max(1);
    let mut work_items = Vec::new();

    for chunk_start in (0..m * n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(m * n);
        let indices: Vec<(usize, usize)> = (chunk_start..chunk_end)
            .map(|idx| (idx / n, idx % n))
            .collect();

        work_items.push(WorkItem::new(chunk_start, (indices, a.to_owned())));
    }

    scheduler.submit_work(work_items)?;

    let results = scheduler.execute(|(indices, matrix)| {
        indices
            .into_iter()
            .map(|(i, j)| {
                let val = matrix[(i, j)];
                val * val
            })
            .sum::<F>()
    })?;

    let sum_of_squares: F = results.into_iter().sum();
    Ok(sum_of_squares.sqrt())
}

/// Parallel nuclear norm computation
pub fn parallel_nuclear_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Nuclear norm is the sum of singular values
    use super::decomposition::parallel_svd_work_stealing;
    let (_, s, _) = parallel_svd_work_stealing(a, workers)?;
    Ok(s.iter().cloned().sum())
}

/// Parallel matrix 1-norm computation (maximum column sum)
pub fn parallel_matrix_1_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (_, n) = a.dim();
    let scheduler = WorkStealingScheduler::new(workers);

    // Create work items for each column
    let work_items: Vec<WorkItem<(usize, Array2<F>)>> = (0..n)
        .map(|j| WorkItem::new(j, (j, a.to_owned())))
        .collect();

    scheduler.submit_work(work_items)?;

    let results =
        scheduler.execute(|(j, matrix)| matrix.column(j).iter().map(|x| x.abs()).sum::<F>())?;

    Ok(results.into_iter().fold(F::zero(), |acc, x| acc.max(x)))
}

/// Parallel spectral norm computation (largest singular value)
pub fn parallel_spectral_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Spectral norm is the largest singular value
    use super::decomposition::parallel_svd_work_stealing;
    let (_, s, _) = parallel_svd_work_stealing(a, workers)?;
    Ok(s.iter().fold(F::zero(), |acc, &x| acc.max(x)))
}

/// Parallel matrix infinity norm computation (maximum row sum)
pub fn parallel_matrix_inf_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, _) = a.dim();
    let scheduler = WorkStealingScheduler::new(workers);

    // Create work items for each row
    let work_items: Vec<WorkItem<(usize, Array2<F>)>> = (0..m)
        .map(|i| WorkItem::new(i, (i, a.to_owned())))
        .collect();

    scheduler.submit_work(work_items)?;

    let results =
        scheduler.execute(|(i, matrix)| matrix.row(i).iter().map(|x| x.abs()).sum::<F>())?;

    Ok(results.into_iter().fold(F::zero(), |acc, x| acc.max(x)))
}

/// Parallel Padé approximation for matrix exponential
fn parallel_pade_approximation<F>(
    a: &ArrayView2<F>,
    degree: usize,
    workers: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let mut result = Array2::<F>::eye(n);
    let mut power = Array2::<F>::eye(n);

    // Compute powers of A and coefficients in parallel
    for k in 1..=degree {
        power = parallel_gemm_work_stealing(&power.view(), a, workers)?;

        // Padé coefficient (simplified)
        let coeff = F::from(1.0).unwrap() / F::from(factorial(k)).unwrap();
        result = result + power.mapv(|x| x * coeff);
    }

    Ok(result)
}

/// Simple factorial function for Padé coefficients
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).fold(1.0, |acc, x| acc * x as f64)
    }
}
