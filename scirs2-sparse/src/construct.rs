// Construction utilities for sparse arrays
//
// This module provides functions for constructing sparse arrays,
// including identity matrices, diagonal matrices, random arrays, etc.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use ndarray::Array1;
use num_traits::Float;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use crate::coo_array::CooArray;
use crate::csr_array::CsrArray;
use crate::dok_array::DokArray;
use crate::error::{SparseError, SparseResult};
use crate::lil_array::LilArray;
use crate::sparray::SparseArray;

// Import parallel operations from scirs2-core
use scirs2_core::parallel_ops::*;

/// Creates a sparse identity array of size n x n
///
/// # Arguments
/// * `n` - Size of the square array
/// * `format` - Format of the output array ("csr" or "coo")
///
/// # Returns
/// A sparse array representing the identity matrix
///
/// # Examples
///
/// ```
/// use scirs2_sparse::construct::eye_array;
///
/// let eye: Box<dyn scirs2_sparse::SparseArray<f64>> = eye_array(3, "csr").unwrap();
/// assert_eq!(eye.shape(), (3, 3));
/// assert_eq!(eye.nnz(), 3);
/// assert_eq!(eye.get(0, 0), 1.0);
/// assert_eq!(eye.get(1, 1), 1.0);
/// assert_eq!(eye.get(2, 2), 1.0);
/// assert_eq!(eye.get(0, 1), 0.0);
/// ```
#[allow(dead_code)]
pub fn eye_array<T>(n: usize, format: &str) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    if n == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimension must be positive".to_string(),
        ));
    }

    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(T::one());
    }

    match format.to_lowercase().as_str() {
        "csr" => CsrArray::from_triplets(&rows, &cols, &data, (n, n), true)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "coo" => CooArray::from_triplets(&rows, &cols, &data, (n, n), true)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "dok" => DokArray::from_triplets(&rows, &cols, &data, (n, n))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "lil" => LilArray::from_triplets(&rows, &cols, &data, (n, n))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        _ => Err(SparseError::ValueError(format!(
            "Unknown sparse format: {format}. Supported formats are 'csr', 'coo', 'dok', and 'lil'"
        ))),
    }
}

/// Creates a sparse identity array of size m x n with k-th diagonal filled with ones
///
/// # Arguments
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `k` - Diagonal index (0 = main diagonal, >0 = above main, <0 = below main)
/// * `format` - Format of the output array ("csr" or "coo")
///
/// # Returns
/// A sparse array with ones on the specified diagonal
///
/// # Examples
///
/// ```
/// use scirs2_sparse::construct::eye_array_k;
///
/// // Identity with main diagonal (k=0)
/// let eye: Box<dyn scirs2_sparse::SparseArray<f64>> = eye_array_k(3, 3, 0, "csr").unwrap();
/// assert_eq!(eye.get(0, 0), 1.0);
/// assert_eq!(eye.get(1, 1), 1.0);
/// assert_eq!(eye.get(2, 2), 1.0);
///
/// // Superdiagonal (k=1)
/// let superdiag: Box<dyn scirs2_sparse::SparseArray<f64>> = eye_array_k(3, 4, 1, "csr").unwrap();
/// assert_eq!(superdiag.get(0, 1), 1.0);
/// assert_eq!(superdiag.get(1, 2), 1.0);
/// assert_eq!(superdiag.get(2, 3), 1.0);
///
/// // Subdiagonal (k=-1)
/// let subdiag: Box<dyn scirs2_sparse::SparseArray<f64>> = eye_array_k(4, 3, -1, "csr").unwrap();
/// assert_eq!(subdiag.get(1, 0), 1.0);
/// assert_eq!(subdiag.get(2, 1), 1.0);
/// assert_eq!(subdiag.get(3, 2), 1.0);
/// ```
#[allow(dead_code)]
pub fn eye_array_k<T>(
    m: usize,
    n: usize,
    k: isize,
    format: &str,
) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    // Calculate diagonal elements
    if k >= 0 {
        let k_usize = k as usize;
        let len = std::cmp::min(m, n.saturating_sub(k_usize));

        for i in 0..len {
            rows.push(i);
            cols.push(i + k_usize);
            data.push(T::one());
        }
    } else {
        let k_abs = (-k) as usize;
        let len = std::cmp::min(m.saturating_sub(k_abs), n);

        for i in 0..len {
            rows.push(i + k_abs);
            cols.push(i);
            data.push(T::one());
        }
    }

    match format.to_lowercase().as_str() {
        "csr" => CsrArray::from_triplets(&rows, &cols, &data, (m, n), true)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "coo" => CooArray::from_triplets(&rows, &cols, &data, (m, n), true)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "dok" => DokArray::from_triplets(&rows, &cols, &data, (m, n))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "lil" => LilArray::from_triplets(&rows, &cols, &data, (m, n))
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        _ => Err(SparseError::ValueError(format!(
            "Unknown sparse format: {format}. Supported formats are 'csr', 'coo', 'dok', and 'lil'"
        ))),
    }
}

/// Creates a sparse array from the specified diagonals
///
/// # Arguments
/// * `diagonals` - Data for the diagonals
/// * `offsets` - Offset for each diagonal (0 = main, >0 = above main, <0 = below main)
/// * `shape` - Shape of the output array (m, n)
/// * `format` - Format of the output array ("csr" or "coo")
///
/// # Returns
/// A sparse array with the specified diagonals
///
/// # Examples
///
/// ```
/// use scirs2_sparse::construct::diags_array;
/// use ndarray::Array1;
///
/// let diags = vec![
///     Array1::from_vec(vec![1.0, 2.0, 3.0]), // main diagonal
///     Array1::from_vec(vec![4.0, 5.0])       // superdiagonal
/// ];
/// let offsets = vec![0, 1];
/// let shape = (3, 3);
///
/// let result = diags_array(&diags, &offsets, shape, "csr").unwrap();
/// assert_eq!(result.shape(), (3, 3));
/// assert_eq!(result.get(0, 0), 1.0);
/// assert_eq!(result.get(1, 1), 2.0);
/// assert_eq!(result.get(2, 2), 3.0);
/// assert_eq!(result.get(0, 1), 4.0);
/// assert_eq!(result.get(1, 2), 5.0);
/// ```
#[allow(dead_code)]
pub fn diags_array<T>(
    diagonals: &[Array1<T>],
    offsets: &[isize],
    shape: (usize, usize),
    format: &str,
) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    if diagonals.len() != offsets.len() {
        return Err(SparseError::InconsistentData {
            reason: "Number of diagonals must match number of offsets".to_string(),
        });
    }

    if shape.0 == 0 || shape.1 == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    let (m, n) = shape;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (i, (diag, &offset)) in diagonals.iter().zip(offsets.iter()).enumerate() {
        if offset >= 0 {
            let offset_usize = offset as usize;
            let max_len = std::cmp::min(m, n.saturating_sub(offset_usize));

            if diag.len() > max_len {
                return Err(SparseError::InconsistentData {
                    reason: format!("Diagonal {i} is too long ({} > {})", diag.len(), max_len),
                });
            }

            for (j, &value) in diag.iter().enumerate() {
                if !value.is_zero() {
                    rows.push(j);
                    cols.push(j + offset_usize);
                    data.push(value);
                }
            }
        } else {
            let offset_abs = (-offset) as usize;
            let max_len = std::cmp::min(m.saturating_sub(offset_abs), n);

            if diag.len() > max_len {
                return Err(SparseError::InconsistentData {
                    reason: format!("Diagonal {i} is too long ({} > {})", diag.len(), max_len),
                });
            }

            for (j, &value) in diag.iter().enumerate() {
                if !value.is_zero() {
                    rows.push(j + offset_abs);
                    cols.push(j);
                    data.push(value);
                }
            }
        }
    }

    match format.to_lowercase().as_str() {
        "csr" => CsrArray::from_triplets(&rows, &cols, &data, shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "coo" => CooArray::from_triplets(&rows, &cols, &data, shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "dok" => DokArray::from_triplets(&rows, &cols, &data, shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "lil" => LilArray::from_triplets(&rows, &cols, &data, shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        _ => Err(SparseError::ValueError(format!(
            "Unknown sparse format: {format}. Supported formats are 'csr', 'coo', 'dok', and 'lil'"
        ))),
    }
}

/// Creates a random sparse array with specified density
///
/// # Arguments
/// * `shape` - Shape of the output array (m, n)
/// * `density` - Density of non-zero elements (between 0.0 and 1.0)
/// * `seed` - Optional seed for the random number generator
/// * `format` - Format of the output array ("csr" or "coo")
///
/// # Returns
/// A sparse array with random non-zero elements
///
/// # Examples
///
/// ```
/// use scirs2_sparse::construct::random_array;
///
/// // Create a 10x10 array with 30% non-zero elements
/// let random = random_array::<f64>((10, 10), 0.3, None, "csr").unwrap();
/// assert_eq!(random.shape(), (10, 10));
///
/// // Create a random array with a specific seed
/// let seeded = random_array::<f64>((5, 5), 0.5, Some(42), "coo").unwrap();
/// assert_eq!(seeded.shape(), (5, 5));
/// ```
#[allow(dead_code)]
pub fn random_array<T>(
    shape: (usize, usize),
    density: f64,
    seed: Option<u64>,
    format: &str,
) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    let (m, n) = shape;

    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::ValueError(
            "Density must be between 0.0 and 1.0".to_string(),
        ));
    }

    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    // Calculate the number of non-zero elements
    let nnz = (m * n) as f64 * density;
    let nnz = nnz.round() as usize;

    // Create random indices
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    // Create RNG
    let mut rng = if let Some(seed_value) = seed {
        rand::rngs::StdRng::seed_from_u64(seed_value)
    } else {
        // For a random seed, use rng
        let seed = rand::random::<u64>();
        rand::rngs::StdRng::seed_from_u64(seed)
    };

    // Generate random elements
    let total = m * n;

    if density > 0.4 {
        // For high densities, more efficient to generate a mask
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(&mut rng);

        for &idx in indices.iter().take(nnz) {
            let row = idx / n;
            let col = idx % n;

            rows.push(row);
            cols.push(col);

            // Generate random non-zero value
            // For simplicity, using values between -1 and 1
            let mut val: f64 = rng.random_range(-1.0..1.0);
            // Make sure the value is not zero
            while val.abs() < 1e-10 {
                val = rng.random_range(-1.0..1.0);
            }
            data.push(T::from(val).unwrap());
        }
    } else {
        // For low densities..use a set to track already-chosen positions
        let mut positions = std::collections::HashSet::with_capacity(nnz);

        while positions.len() < nnz {
            let row = rng.random_range(0..m);
            let col = rng.random_range(0..n);
            let pos = row * n + col; // Using row/col as usize indices

            if positions.insert(pos) {
                rows.push(row);
                cols.push(col);

                // Generate random non-zero value
                let mut val: f64 = rng.random_range(-1.0..1.0);
                // Make sure the value is not zero
                while val.abs() < 1e-10 {
                    val = rng.random_range(-1.0..1.0);
                }
                data.push(T::from(val).unwrap());
            }
        }
    }

    // Create the output array
    match format.to_lowercase().as_str() {
        "csr" => CsrArray::from_triplets(&rows, &cols, &data, shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "coo" => CooArray::from_triplets(&rows, &cols, &data, shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "dok" => DokArray::from_triplets(&rows, &cols, &data, shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "lil" => LilArray::from_triplets(&rows, &cols, &data, shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        _ => Err(SparseError::ValueError(format!(
            "Unknown sparse format: {format}. Supported formats are 'csr', 'coo', 'dok', and 'lil'"
        ))),
    }
}

/// Creates a large sparse random array using parallel processing
///
/// This function uses parallel construction for improved performance when creating
/// large sparse arrays with many non-zero elements.
///
/// # Arguments
/// * `shape` - Shape of the array (rows, cols)
/// * `density` - Density of non-zero elements (0.0 to 1.0)
/// * `seed` - Optional random seed for reproducibility
/// * `format` - Format of the output array ("csr" or "coo")
/// * `parallel_threshold` - Minimum number of elements to use parallel construction
///
/// # Returns
/// A sparse array with randomly distributed non-zero elements
///
/// # Examples
///
/// ```
/// use scirs2_sparse::construct::random_array_parallel;
///
/// // Create a large random sparse array
/// let large_random = random_array_parallel::<f64>((1000, 1000), 0.01, Some(42), "csr", 10000).unwrap();
/// assert_eq!(large_random.shape(), (1000, 1000));
/// assert!(large_random.nnz() > 5000); // Approximately 10000 non-zeros expected
/// ```
#[allow(dead_code)]
pub fn random_array_parallel<T>(
    shape: (usize, usize),
    density: f64,
    seed: Option<u64>,
    format: &str,
    parallel_threshold: usize,
) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + Send
        + Sync
        + 'static,
{
    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::ValueError(
            "Density must be between 0.0 and 1.0".to_string(),
        ));
    }

    let (rows, cols) = shape;
    if rows == 0 || cols == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    let total_elements = rows * cols;
    let expected_nnz = (total_elements as f64 * density) as usize;

    // Use parallel construction for large matrices
    if total_elements >= parallel_threshold && expected_nnz >= 1000 {
        parallel_random_construction(shape, density, seed, format)
    } else {
        // Fall back to sequential construction for small matrices
        random_array(shape, density, seed, format)
    }
}

/// Internal parallel construction function
#[allow(dead_code)]
fn parallel_random_construction<T>(
    shape: (usize, usize),
    density: f64,
    seed: Option<u64>,
    format: &str,
) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + Send
        + Sync
        + 'static,
{
    let (rows, cols) = shape;
    let total_elements = rows * cols;
    let expected_nnz = (total_elements as f64 * density) as usize;

    // Determine number of chunks based on available parallelism
    let num_chunks = std::cmp::min(scirs2_core::parallel_ops::get_num_threads(), rows.min(cols));
    let chunk_size = std::cmp::max(1, rows / num_chunks);

    // Create row chunks for parallel processing
    let row_chunks: Vec<_> = (0..rows)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    // Generate random elements in parallel using enumerate to get chunk index
    let chunk_data: Vec<_> = row_chunks.iter().enumerate().collect();
    let results: Vec<_> = parallel_map(&chunk_data, |(chunk_idx, row_chunk)| {
        let mut local_rows = Vec::new();
        let mut local_cols = Vec::new();
        let mut local_data = Vec::new();

        // Use a different seed for each chunk to ensure good randomization
        let chunk_seed = seed.unwrap_or(42) + *chunk_idx as u64 * 1000007; // Large prime offset
        let mut rng = rand::rngs::StdRng::seed_from_u64(chunk_seed);

        for &row in row_chunk.iter() {
            // Determine how many elements to generate for this row
            let row_elements = cols;
            let row_expected_nnz = std::cmp::max(1, (row_elements as f64 * density) as usize);

            // Generate random column indices for this row
            let mut col_indices: Vec<usize> = (0..cols).collect();
            col_indices.shuffle(&mut rng);

            // Take the first row_expected_nnz columns
            for &col in col_indices.iter().take(row_expected_nnz) {
                // Generate random value
                let mut val = rng.random_range(-1.0..1.0);
                // Make sure the value is not zero
                while val.abs() < 1e-10 {
                    val = rng.random_range(-1.0..1.0);
                }

                local_rows.push(row);
                local_cols.push(col);
                local_data.push(T::from(val).unwrap());
            }
        }

        (local_rows, local_cols, local_data)
    });

    // Combine results from all chunks
    let mut all_rows = Vec::new();
    let mut all_cols = Vec::new();
    let mut all_data = Vec::new();

    for (mut rowschunk, mut cols_chunk, mut data_chunk) in results {
        all_rows.extend(rowschunk);
        all_cols.append(&mut cols_chunk);
        all_data.append(&mut data_chunk);
    }

    // Create the output array
    match format.to_lowercase().as_str() {
        "csr" => CsrArray::from_triplets(&all_rows, &all_cols, &all_data, shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "coo" => CooArray::from_triplets(&all_rows, &all_cols, &all_data, shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "dok" => DokArray::from_triplets(&all_rows, &all_cols, &all_data, shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        "lil" => LilArray::from_triplets(&all_rows, &all_cols, &all_data, shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>),
        _ => Err(SparseError::ValueError(format!(
            "Unknown sparse format: {format}. Supported formats are 'csr', 'coo', 'dok', and 'lil'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_array() {
        let eye = eye_array::<f64>(3, "csr").unwrap();

        assert_eq!(eye.shape(), (3, 3));
        assert_eq!(eye.nnz(), 3);
        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(2, 2), 1.0);
        assert_eq!(eye.get(0, 1), 0.0);

        // Try COO format
        let eye_coo = eye_array::<f64>(3, "coo").unwrap();
        assert_eq!(eye_coo.shape(), (3, 3));
        assert_eq!(eye_coo.nnz(), 3);

        // Try DOK format
        let eye_dok = eye_array::<f64>(3, "dok").unwrap();
        assert_eq!(eye_dok.shape(), (3, 3));
        assert_eq!(eye_dok.nnz(), 3);
        assert_eq!(eye_dok.get(0, 0), 1.0);
        assert_eq!(eye_dok.get(1, 1), 1.0);
        assert_eq!(eye_dok.get(2, 2), 1.0);

        // Try LIL format
        let eye_lil = eye_array::<f64>(3, "lil").unwrap();
        assert_eq!(eye_lil.shape(), (3, 3));
        assert_eq!(eye_lil.nnz(), 3);
        assert_eq!(eye_lil.get(0, 0), 1.0);
        assert_eq!(eye_lil.get(1, 1), 1.0);
        assert_eq!(eye_lil.get(2, 2), 1.0);
    }

    #[test]
    fn test_eye_array_k() {
        // Identity with main diagonal (k=0)
        let eye = eye_array_k::<f64>(3, 3, 0, "csr").unwrap();
        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(2, 2), 1.0);

        // Superdiagonal (k=1)
        let superdiag = eye_array_k::<f64>(3, 4, 1, "csr").unwrap();
        assert_eq!(superdiag.get(0, 1), 1.0);
        assert_eq!(superdiag.get(1, 2), 1.0);
        assert_eq!(superdiag.get(2, 3), 1.0);

        // Subdiagonal (k=-1)
        let subdiag = eye_array_k::<f64>(4, 3, -1, "csr").unwrap();
        assert_eq!(subdiag.get(1, 0), 1.0);
        assert_eq!(subdiag.get(2, 1), 1.0);
        assert_eq!(subdiag.get(3, 2), 1.0);

        // Try LIL format
        let eye_lil = eye_array_k::<f64>(3, 3, 0, "lil").unwrap();
        assert_eq!(eye_lil.get(0, 0), 1.0);
        assert_eq!(eye_lil.get(1, 1), 1.0);
        assert_eq!(eye_lil.get(2, 2), 1.0);
    }

    #[test]
    fn test_diags_array() {
        let diags = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // main diagonal
            Array1::from_vec(vec![4.0, 5.0]),      // superdiagonal
        ];
        let offsets = vec![0, 1];
        let shape = (3, 3);

        let result = diags_array(&diags, &offsets, shape, "csr").unwrap();
        assert_eq!(result.shape(), (3, 3));
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(1, 1), 2.0);
        assert_eq!(result.get(2, 2), 3.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 2), 5.0);

        // Try with multiple diagonals and subdiagonals
        let diags = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // main diagonal
            Array1::from_vec(vec![4.0, 5.0]),      // superdiagonal
            Array1::from_vec(vec![6.0, 7.0]),      // subdiagonal
        ];
        let offsets = vec![0, 1, -1];
        let shape = (3, 3);

        let result = diags_array(&diags, &offsets, shape, "csr").unwrap();
        assert_eq!(result.shape(), (3, 3));
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(1, 1), 2.0);
        assert_eq!(result.get(2, 2), 3.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 2), 5.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(2, 1), 7.0);

        // Try LIL format
        let result_lil = diags_array(&diags, &offsets, shape, "lil").unwrap();
        assert_eq!(result_lil.shape(), (3, 3));
        assert_eq!(result_lil.get(0, 0), 1.0);
        assert_eq!(result_lil.get(1, 1), 2.0);
        assert_eq!(result_lil.get(2, 2), 3.0);
        assert_eq!(result_lil.get(0, 1), 4.0);
        assert_eq!(result_lil.get(1, 2), 5.0);
        assert_eq!(result_lil.get(1, 0), 6.0);
        assert_eq!(result_lil.get(2, 1), 7.0);
    }

    #[test]
    fn test_random_array() {
        let shape = (10, 10);
        let density = 0.3;

        let random = random_array::<f64>(shape, density, None, "csr").unwrap();

        // Check shape and sparsity
        assert_eq!(random.shape(), shape);
        let nnz = random.nnz();
        let expected_nnz = (shape.0 * shape.1) as f64 * density;

        // Allow for some random variation, but should be close to expected density
        assert!(
            (nnz as f64) > expected_nnz * 0.7,
            "Too few non-zeros: {nnz}"
        );
        assert!(
            (nnz as f64) < expected_nnz * 1.3,
            "Too many non-zeros: {nnz}"
        );

        // Test with custom RNG seed
        let random_seeded = random_array::<f64>(shape, density, Some(42), "csr").unwrap();
        assert_eq!(random_seeded.shape(), shape);

        // Test LIL format
        let random_lil = random_array::<f64>((5, 5), 0.5, Some(42), "lil").unwrap();
        assert_eq!(random_lil.shape(), (5, 5));
        let nnz_lil = random_lil.nnz();
        let expected_nnz_lil = 25.0 * 0.5;
        assert!(
            (nnz_lil as f64) > expected_nnz_lil * 0.7,
            "Too few non-zeros in LIL: {nnz_lil}"
        );
        assert!(
            (nnz_lil as f64) < expected_nnz_lil * 1.3,
            "Too many non-zeros in LIL: {nnz_lil}"
        );
    }
}
