//! GEMM (General Matrix Multiply) operations using work-stealing
//!
//! This module provides work-stealing implementations for matrix-vector multiplication,
//! matrix-matrix multiplication, and optimized block matrix multiplication operations.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use std::iter::Sum;

use super::super::core::WorkItem;
use super::super::scheduler::WorkStealingScheduler;

/// Work-stealing matrix-vector multiplication
pub fn parallel_matvec_work_stealing<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    num_workers: usize,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    if n != vector.len() {
        return Err(crate::error::LinalgError::ShapeError(
            "Matrix and vector dimensions don't match".to_string(),
        ));
    }

    let scheduler = WorkStealingScheduler::new(num_workers);
    let mut result = Array1::zeros(m);

    // Create work items for each row
    let work_items: Vec<WorkItem<(usize, Array1<F>, F)>> = (0..m)
        .map(|i| {
            let row = matrix.row(i).to_owned();
            let dot_product = row.dot(vector);
            WorkItem::new(i, (i, row, dot_product))
        })
        .collect();

    scheduler.submit_work(work_items)?;

    // Execute work and collect results
    let results = scheduler.execute(|(i, row, dot_product)| (i, dot_product))?;

    // Assemble final result
    for (i, value) in results {
        result[i] = value;
    }

    Ok(result)
}

/// Work-stealing matrix multiplication
pub fn parallel_gemm_work_stealing<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    num_workers: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(crate::error::LinalgError::ShapeError(
            "Matrix dimensions don't match for multiplication".to_string(),
        ));
    }

    let scheduler = WorkStealingScheduler::new(num_workers);
    let mut result = Array2::zeros((m, n));

    // Create work items for blocks of the result matrix
    let blocksize = (m * n / (num_workers * 4)).max(1);
    let mut work_items = Vec::new();

    for block_start in (0..m * n).step_by(blocksize) {
        let block_end = (block_start + blocksize).min(m * n);
        let indices: Vec<(usize, usize)> = (block_start..block_end)
            .map(|idx| (idx / n, idx % n))
            .collect();

        work_items.push(WorkItem::new(
            block_start,
            (indices, a.to_owned(), b.to_owned()),
        ));
    }

    scheduler.submit_work(work_items)?;

    // Execute work and collect results
    let results = scheduler.execute(|(indices, a_copy, b_copy)| {
        indices
            .into_iter()
            .map(|(i, j)| {
                let value = a_copy.row(i).dot(&b_copy.column(j));
                (i, j, value)
            })
            .collect::<Vec<_>>()
    })?;

    // Assemble final result
    for block_results in results {
        for (i, j, value) in block_results {
            result[(i, j)] = value;
        }
    }

    Ok(result)
}

/// Parallel block matrix multiplication with advanced cache optimization
pub fn parallel_block_gemm<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    num_workers: usize,
    blocksize: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(
            "Matrix dimensions don't match for multiplication".to_string(),
        ));
    }

    // Adaptive block size based on cache size and matrix dimensions
    let optimal_blocksize = blocksize.unwrap_or_else(|| {
        let l1_cachesize = 32 * 1024; // 32KB L1 cache assumption
        let elementsize = std::mem::size_of::<F>();
        (l1_cachesize / (3 * elementsize)).clamp(64, 512)
    });

    let mut result = Array2::zeros((m, n));
    let scheduler = WorkStealingScheduler::new(num_workers);

    // Create work items for each block
    let mut work_items = Vec::new();
    let mut block_id = 0;

    for i in (0..m).step_by(optimal_blocksize) {
        for j in (0..n).step_by(optimal_blocksize) {
            let i_end = (i + optimal_blocksize).min(m);
            let j_end = (j + optimal_blocksize).min(n);

            work_items.push(WorkItem::new(
                block_id,
                (i, j, i_end, j_end, a.to_owned(), b.to_owned()),
            ));
            block_id += 1;
        }
    }

    scheduler.submit_work(work_items)?;

    let results = scheduler.execute(move |(i_start, j_start, i_end, j_end, a_copy, b_copy)| {
        let mut block_result = Array2::zeros((i_end - i_start, j_end - j_start));

        // Block multiplication with cache-friendly access pattern
        for k in (0..k1).step_by(optimal_blocksize) {
            let k_end = (k + optimal_blocksize).min(k1);

            for i in 0..(i_end - i_start) {
                for j in 0..(j_end - j_start) {
                    let mut sum = F::zero();
                    for kk in k..k_end {
                        sum += a_copy[(i_start + i, kk)] * b_copy[(kk, j_start + j)];
                    }
                    block_result[(i, j)] += sum;
                }
            }
        }

        (i_start, j_start, i_end, j_end, block_result)
    })?;

    // Assemble final result
    for (i_start, j_start, i_end, j_end, block_result) in results {
        for i in 0..(i_end - i_start) {
            for j in 0..(j_end - j_start) {
                result[(i_start + i, j_start + j)] = block_result[(i, j)];
            }
        }
    }

    Ok(result)
}
