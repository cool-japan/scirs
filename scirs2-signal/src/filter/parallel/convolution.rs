//! Parallel convolution operations
//!
//! This module provides parallel implementations of convolution operations
//! including 1D and 2D convolution with various modes and boundary conditions.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

/// Parallel convolution using overlap-save method
///
/// Performs convolution of two signals using parallel processing
/// with the overlap-save method for efficiency.
///
/// # Arguments
///
/// * `a` - First signal
/// * `v` - Second signal (kernel)
/// * `mode` - Convolution mode ("full", "same", "valid")
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Convolution result
pub fn parallel_convolve<T, U>(
    a: &[T],
    v: &[U],
    mode: &str,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
    U: Float + NumCast + Debug + Send + Sync,
{
    // Convert inputs to Array1
    let a_array = Array1::from_iter(
        a.iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    let v_array = Array1::from_iter(
        v.iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    // Use overlap-save for efficiency with long signals
    if a_array.len() > 1000 && v_array.len() > 10 {
        parallel_convolve_overlap_save(&a_array, &v_array, mode, chunk_size)
    } else {
        // For small signals, use direct convolution
        parallel_convolve_direct(&a_array, &v_array, mode)
    }
}

/// Overlap-save convolution for parallel processing
pub(crate) fn parallel_convolve_overlap_save(
    a: &Array1<f64>,
    v: &Array1<f64>,
    mode: &str,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();

    // For overlap-save, process in chunks
    let chunk = chunk_size.unwrap_or(4096);
    let overlap = nv - 1;

    // Full convolution length
    let n_full = na + nv - 1;
    let mut result = vec![0.0; n_full];

    // Process chunks sequentially (simplified version)
    let n_chunks = (na + chunk - overlap - 1) / (chunk - overlap);
    let mut chunk_results = Vec::with_capacity(n_chunks);

    for i in 0..n_chunks {
        let start = i * (chunk - overlap);
        let end = (start + chunk).min(na);

        // Extract chunk with zero padding if needed
        let mut chunk_data = vec![0.0; chunk];
        for j in start..end {
            chunk_data[j - start] = a[j];
        }

        // Convolve chunk with kernel
        let mut chunk_result = vec![0.0; chunk + nv - 1];
        for j in 0..chunk {
            for k in 0..nv {
                chunk_result[j + k] += chunk_data[j] * v[k];
            }
        }

        chunk_results.push(chunk_result);
    }

    // Combine chunk results
    for (i, chunk_res) in chunk_results.iter().enumerate() {
        let start = i * (chunk - overlap);
        for (j, &val) in chunk_res.iter().enumerate() {
            if start + j < n_full {
                result[start + j] += val;
            }
        }
    }

    // Apply mode
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            Ok(result[start..end].to_vec())
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel must not be larger than signal".to_string(),
                ));
            }
            let start = nv - 1;
            let end = n_full - (nv - 1);
            Ok(result[start..end].to_vec())
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Direct convolution for small signals
pub(crate) fn parallel_convolve_direct(
    a: &Array1<f64>,
    v: &Array1<f64>,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();
    let n_full = na + nv - 1;

    // Use sequential iteration (simplified version)
    let mut result = Vec::with_capacity(n_full);
    for i in 0..n_full {
        let mut sum = 0.0;
        for j in 0..nv {
            if i >= j && i - j < na {
                sum += a[i - j] * v[j];
            }
        }
        result.push(sum);
    }

    // Apply mode
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            Ok(result[start..end].to_vec())
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel must not be larger than signal".to_string(),
                ));
            }
            let start = nv - 1;
            let end = n_full - (nv - 1);
            Ok(result[start..end].to_vec())
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Parallel 2D convolution for image filtering
///
/// # Arguments
///
/// * `image` - 2D input array (image)
/// * `kernel` - 2D convolution kernel
/// * `mode` - Convolution mode
/// * `boundary` - Boundary handling ("zero", "reflect", "wrap")
///
/// # Returns
///
/// * Filtered 2D array
pub fn parallel_convolve2d(
    image: &Array2<f64>,
    kernel: &Array2<f64>,
    mode: &str,
    boundary: &str,
) -> SignalResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    // Validate inputs
    if ker_rows > img_rows || ker_cols > img_cols {
        return Err(SignalError::ValueError(
            "Kernel dimensions must not exceed image dimensions".to_string(),
        ));
    }

    // Determine output size based on mode
    let (out_rows, out_cols) = match mode {
        "full" => (img_rows + ker_rows - 1, img_cols + ker_cols - 1),
        "same" => (img_rows, img_cols),
        "valid" => (img_rows - ker_rows + 1, img_cols - ker_cols + 1),
        _ => return Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    };

    // Padding for boundary handling
    let pad_rows = ker_rows - 1;
    let pad_cols = ker_cols - 1;

    // Create padded image based on boundary condition
    let padded = pad_image(image, pad_rows, pad_cols, boundary)?;

    // Sequential convolution over rows (simplified version)
    let mut result_vec = Vec::with_capacity(out_rows);

    for i in 0..out_rows {
        let mut row_result = vec![0.0; out_cols];

        // Adjust indices based on mode
        let row_offset = match mode {
            "full" => 0,
            "same" => ker_rows / 2,
            "valid" => ker_rows - 1,
            _ => 0,
        };

        let col_offset = match mode {
            "full" => 0,
            "same" => ker_cols / 2,
            "valid" => ker_cols - 1,
            _ => 0,
        };

        for j in 0..out_cols {
            let mut sum = 0.0;

            // Convolution at position (i, j)
            for ki in 0..ker_rows {
                for kj in 0..ker_cols {
                    let pi = i + row_offset + ki;
                    let pj = j + col_offset + kj;

                    if pi < padded.nrows() && pj < padded.ncols() {
                        sum += padded[[pi, pj]] * kernel[[ker_rows - 1 - ki, ker_cols - 1 - kj]];
                    }
                }
            }

            row_result[j] = sum;
        }

        result_vec.push(row_result);
    }

    // Convert to Array2
    let mut output = Array2::zeros((out_rows, out_cols));
    for (i, row) in result_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            output[[i, j]] = val;
        }
    }

    Ok(output)
}

/// Pad image for boundary handling
pub(crate) fn pad_image(
    image: &Array2<f64>,
    pad_rows: usize,
    pad_cols: usize,
    boundary: &str,
) -> SignalResult<Array2<f64>> {
    let (rows, cols) = image.dim();
    let padded_rows = rows + 2 * pad_rows;
    let padded_cols = cols + 2 * pad_cols;

    let mut padded = Array2::zeros((padded_rows, padded_cols));

    // Copy original image to center
    for i in 0..rows {
        for j in 0..cols {
            padded[[i + pad_rows, j + pad_cols]] = image[[i, j]];
        }
    }

    // Apply boundary condition
    match boundary {
        "zero" => {
            // Already zero-padded
        }
        "reflect" => {
            // Reflect padding
            // Top and bottom
            for i in 0..pad_rows {
                for j in 0..cols {
                    padded[[i, j + pad_cols]] = image[[pad_rows - i - 1, j]];
                    padded[[rows + pad_rows + i, j + pad_cols]] = image[[rows - i - 1, j]];
                }
            }

            // Left and right (including corners)
            for i in 0..padded_rows {
                for j in 0..pad_cols {
                    padded[[i, j]] = padded[[i, 2 * pad_cols - j - 1]];
                    padded[[i, cols + pad_cols + j]] = padded[[i, cols + pad_cols - j - 1]];
                }
            }
        }
        "wrap" => {
            // Periodic boundary
            for i in 0..padded_rows {
                for j in 0..padded_cols {
                    let src_i = (i + rows - pad_rows) % rows;
                    let src_j = (j + cols - pad_cols) % cols;
                    padded[[i, j]] = image[[src_i, src_j]];
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown boundary condition: {}",
                boundary
            )));
        }
    }

    Ok(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_convolve() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let kernel = vec![0.25, 0.5, 0.25];

        let result = parallel_convolve(&signal, &kernel, "same", None);
        assert!(result.is_ok());
        let convolved = result.unwrap();
        assert_eq!(convolved.len(), signal.len());
    }

    #[test]
    fn test_parallel_convolve_modes() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 0.5];

        // Test "full" mode
        let full_result = parallel_convolve(&signal, &kernel, "full", None).unwrap();
        assert_eq!(full_result.len(), signal.len() + kernel.len() - 1);

        // Test "same" mode
        let same_result = parallel_convolve(&signal, &kernel, "same", None).unwrap();
        assert_eq!(same_result.len(), signal.len());

        // Test "valid" mode
        let valid_result = parallel_convolve(&signal, &kernel, "valid", None).unwrap();
        assert_eq!(valid_result.len(), signal.len() - kernel.len() + 1);
    }

    #[test]
    fn test_parallel_convolve2d() {
        let image =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let kernel = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.25, 0.25]).unwrap();

        let result = parallel_convolve2d(&image, &kernel, "valid", "zero");
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.dim(), (2, 2));
    }

    #[test]
    fn test_pad_image_zero() {
        let image = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let padded = pad_image(&image, 1, 1, "zero").unwrap();

        assert_eq!(padded.dim(), (4, 4));
        assert_eq!(padded[[1, 1]], 1.0); // Original top-left
        assert_eq!(padded[[0, 0]], 0.0); // Padded region
    }

    #[test]
    fn test_pad_image_reflect() {
        let image = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let padded = pad_image(&image, 1, 1, "reflect").unwrap();

        assert_eq!(padded.dim(), (4, 4));
        assert_eq!(padded[[1, 1]], 1.0); // Original top-left
    }
}
