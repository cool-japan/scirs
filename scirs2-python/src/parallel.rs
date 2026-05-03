//! Parallel batch processing utilities.
//!
//! Provides Python-facing functions that run Rayon-based parallel computations
//! while releasing the Python GIL so that Python threads are not blocked.
//!
//! # Example (Python)
//! ```python
//! import scirs2
//!
//! # Compute means of multiple float arrays in parallel.
//! arrays = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
//! means = scirs2.parallel_map_mean(arrays)
//! # means == [2.0, 4.5, 7.5]
//!
//! # Parallel batch matrix-vector multiply.
//! mats = [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]]
//! vecs = [[3.0, 4.0], [3.0, 4.0]]
//! results = scirs2.parallel_batch_matvec(mats, vecs, 2, 2)
//! # results == [[3.0, 4.0], [6.0, 8.0]]
//! ```

use pyo3::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// parallel_map_mean
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the arithmetic mean of each sub-array in parallel.
///
/// The GIL is released during Rayon computation, allowing Python threads to run
/// concurrently.
///
/// Empty sub-arrays return `0.0`.
///
/// # Arguments
/// * `arrays` – list of float arrays, one mean value is produced per element.
///
/// # Returns
/// A list of `f64` mean values with the same length as `arrays`.
#[pyfunction]
pub fn parallel_map_mean(py: Python<'_>, arrays: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    let result = py.detach(|| {
        use rayon::prelude::*;
        arrays
            .par_iter()
            .map(|arr| {
                if arr.is_empty() {
                    0.0
                } else {
                    arr.iter().sum::<f64>() / arr.len() as f64
                }
            })
            .collect::<Vec<f64>>()
    });
    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────────
// parallel_batch_matvec
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a batch of matrix-vector multiplications in parallel.
///
/// Each pair `(matrices[i], vectors[i])` represents a multiplication
/// `A * v` where `A` is stored in row-major order with shape
/// `(n_rows, n_cols)` and `v` is a vector of length `n_cols`.
///
/// The GIL is released during Rayon computation.
///
/// # Arguments
/// * `matrices` – list of flat row-major matrices, each with `n_rows * n_cols` elements.
/// * `vectors`  – list of vectors, each with `n_cols` elements.
/// * `n_rows`   – number of rows in each matrix.
/// * `n_cols`   – number of columns in each matrix (= length of each vector).
///
/// # Errors
/// Returns `ValueError` if:
/// - `matrices` and `vectors` have different lengths.
/// - Any matrix does not have exactly `n_rows * n_cols` elements.
/// - Any vector does not have exactly `n_cols` elements.
#[pyfunction]
pub fn parallel_batch_matvec(
    py: Python<'_>,
    matrices: Vec<Vec<f64>>,
    vectors: Vec<Vec<f64>>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if matrices.len() != vectors.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "matrices ({}) and vectors ({}) must have equal length",
            matrices.len(),
            vectors.len()
        )));
    }

    let expected_mat = n_rows * n_cols;
    for (i, mat) in matrices.iter().enumerate() {
        if mat.len() != expected_mat {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "matrix[{i}] has {} elements but expected {expected_mat} (n_rows={n_rows}, n_cols={n_cols})",
                mat.len()
            )));
        }
    }
    for (i, vec) in vectors.iter().enumerate() {
        if vec.len() != n_cols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vector[{i}] has {} elements but expected {n_cols} (n_cols)",
                vec.len()
            )));
        }
    }

    let result = py.detach(|| {
        use rayon::prelude::*;
        matrices
            .par_iter()
            .zip(vectors.par_iter())
            .map(|(mat, vec)| {
                (0..n_rows)
                    .map(|r| {
                        (0..n_cols)
                            .map(|c| mat[r * n_cols + c] * vec[c])
                            .sum::<f64>()
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    });
    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────────
// Module registration
// ──────────────────────────────────────────────────────────────────────────────

/// Register parallel batch-processing functions into a PyO3 module.
///
/// Exposes [`parallel_map_mean`] and [`parallel_batch_matvec`].
pub fn register_parallel_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_map_mean, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_batch_matvec, m)?)?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_map_mean_empty_arrays() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let arrays: Vec<Vec<f64>> = vec![vec![], vec![1.0, 2.0, 3.0], vec![]];
            let means = parallel_map_mean(py, arrays).expect("parallel_map_mean failed");
            assert_eq!(means.len(), 3);
            assert!((means[0]).abs() < f64::EPSILON); // empty → 0.0
            assert!((means[1] - 2.0).abs() < f64::EPSILON);
            assert!((means[2]).abs() < f64::EPSILON); // empty → 0.0
        });
    }

    #[test]
    fn parallel_batch_matvec_identity_matrix() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // 2×2 identity × [3.0, 4.0] should give [3.0, 4.0].
            let matrices = vec![vec![1.0, 0.0, 0.0, 1.0], vec![2.0, 0.0, 0.0, 2.0]];
            let vectors = vec![vec![3.0, 4.0], vec![3.0, 4.0]];
            let results =
                parallel_batch_matvec(py, matrices, vectors, 2, 2).expect("matvec failed");
            assert_eq!(results.len(), 2);
            assert!((results[0][0] - 3.0).abs() < f64::EPSILON);
            assert!((results[0][1] - 4.0).abs() < f64::EPSILON);
            assert!((results[1][0] - 6.0).abs() < f64::EPSILON);
            assert!((results[1][1] - 8.0).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn parallel_batch_matvec_mismatched_lengths_errors() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let matrices = vec![vec![1.0, 0.0, 0.0, 1.0]];
            let vectors = vec![vec![1.0], vec![2.0]]; // 2 vectors, 1 matrix → error
            let result = parallel_batch_matvec(py, matrices, vectors, 2, 2);
            assert!(result.is_err());
        });
    }
}
