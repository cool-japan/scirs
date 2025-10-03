//! Parallel morphological filtering operations
//!
//! This module provides parallel implementations of morphological operations
//! including erosion, dilation, opening, and closing for shape-based filtering.

use super::types::MorphologicalOperation;
use crate::error::{SignalError, SignalResult};
use scirs2_core::parallel_ops::*;

/// Helper for parallel operations (temporary replacement)
fn par_iter_with_setup<I, IT, S, F, R, RF, E>(
    items: I,
    _setup: S,
    map_fn: F,
    reduce_fn: RF,
) -> Result<Vec<R>, E>
where
    I: IntoIterator<Item = IT>,
    IT: Copy,
    S: Fn(),
    F: Fn((), IT) -> Result<R, E>,
    RF: Fn(&mut Vec<R>, Result<R, E>) -> Result<(), E>,
    E: std::fmt::Debug,
{
    let mut results = Vec::new();
    for item in items {
        let result = map_fn((), item);
        reduce_fn(&mut results, result)?;
    }
    Ok(results)
}

/// Parallel morphological filtering operations
///
/// Applies morphological operations (erosion, dilation, opening, closing)
/// in parallel for efficient shape-based filtering.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `structuring_element` - Structuring element for morphological operation
/// * `operation` - Type of morphological operation
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Morphologically filtered signal
pub fn parallel_morphological_filter(
    signal: &[f64],
    structuring_element: &[f64],
    operation: MorphologicalOperation,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let se_len = structuring_element.len();
    let half_se = se_len / 2;
    let chunk = chunk_size.unwrap_or(1024.min((n / num_cpus::get()).max(n / 4).max(1)));
    let overlap = half_se;

    // Process signal in overlapping chunks (safe arithmetic to prevent overflow)
    let effective_chunk = chunk.saturating_sub(overlap).max(1); // Ensure at least size 1
    let n_chunks = n.div_ceil(effective_chunk);

    let results = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, i| {
            let start = i * effective_chunk;
            let end = (start + chunk).min(n);
            let chunk_start = start.saturating_sub(overlap);
            let chunk_end = (end + overlap).min(n);

            // Extract chunk with padding
            let chunk_data = &signal[chunk_start..chunk_end];
            let mut chunk_result = Vec::with_capacity(end - start);

            // Apply morphological operation to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;

                let result_val = match operation {
                    MorphologicalOperation::Erosion => {
                        apply_erosion(chunk_data, local_idx, structuring_element, se_len)
                    }
                    MorphologicalOperation::Dilation => {
                        apply_dilation(chunk_data, local_idx, structuring_element, se_len)
                    }
                    MorphologicalOperation::Opening => {
                        // Opening = erosion followed by dilation
                        let eroded =
                            apply_erosion(chunk_data, local_idx, structuring_element, se_len);
                        apply_dilation(&[eroded], 0, structuring_element, se_len)
                    }
                    MorphologicalOperation::Closing => {
                        // Closing = dilation followed by erosion
                        let dilated =
                            apply_dilation(chunk_data, local_idx, structuring_element, se_len);
                        apply_erosion(&[dilated], 0, structuring_element, se_len)
                    }
                };

                chunk_result.push(result_val);
            }

            Ok(chunk_result)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }
    output.truncate(n);

    Ok(output)
}

/// Apply erosion operation at a specific index
pub(crate) fn apply_erosion(signal: &[f64], idx: usize, se: &[f64], selen: usize) -> f64 {
    let half_se = selen / 2;
    let mut min_val = f64::INFINITY;

    for (k, &se_val) in se.iter().enumerate() {
        if se_val > 0.0 {
            let sig_idx = idx + k;
            if sig_idx >= half_se && sig_idx - half_se < signal.len() {
                min_val = min_val.min(signal[sig_idx - half_se]);
            }
        }
    }

    if min_val == f64::INFINITY {
        0.0
    } else {
        min_val
    }
}

/// Apply dilation operation at a specific index
pub(crate) fn apply_dilation(signal: &[f64], idx: usize, se: &[f64], selen: usize) -> f64 {
    let half_se = selen / 2;
    let mut max_val = f64::NEG_INFINITY;

    for (k, &se_val) in se.iter().enumerate() {
        if se_val > 0.0 {
            let sig_idx = idx + k;
            if sig_idx >= half_se && sig_idx - half_se < signal.len() {
                max_val = max_val.max(signal[sig_idx - half_se]);
            }
        }
    }

    if max_val == f64::NEG_INFINITY {
        0.0
    } else {
        max_val
    }
}

/// Parallel gray-scale morphological reconstruction
///
/// Performs morphological reconstruction using parallel processing.
/// Useful for advanced morphological operations like filling holes.
///
/// # Arguments
///
/// * `marker` - Marker image/signal
/// * `mask` - Mask image/signal (constraint)
/// * `structuring_element` - Structuring element
/// * `operation` - Either dilation or erosion based reconstruction
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
///
/// * Reconstructed signal
pub fn parallel_morphological_reconstruction(
    marker: &[f64],
    mask: &[f64],
    structuring_element: &[f64],
    operation: MorphologicalOperation,
    max_iterations: usize,
) -> SignalResult<Vec<f64>> {
    if marker.len() != mask.len() {
        return Err(SignalError::ValueError(
            "Marker and mask must have the same length".to_string(),
        ));
    }

    let mut current = marker.to_vec();
    let mut previous;

    for _iteration in 0..max_iterations {
        previous = current.clone();

        // Apply morphological operation
        current = parallel_morphological_filter(&current, structuring_element, operation, None)?;

        // Apply constraint (point-wise min/max with mask)
        match operation {
            MorphologicalOperation::Dilation | MorphologicalOperation::Closing => {
                // For dilation-based reconstruction, take minimum with mask
                for (i, &mask_val) in mask.iter().enumerate() {
                    current[i] = current[i].min(mask_val);
                }
            }
            MorphologicalOperation::Erosion | MorphologicalOperation::Opening => {
                // For erosion-based reconstruction, take maximum with mask
                for (i, &mask_val) in mask.iter().enumerate() {
                    current[i] = current[i].max(mask_val);
                }
            }
        }

        // Check for convergence
        let mut converged = true;
        for (curr, prev) in current.iter().zip(previous.iter()) {
            if (curr - prev).abs() > 1e-10 {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }
    }

    Ok(current)
}

/// Parallel top-hat transform
///
/// Computes white and black top-hat transforms using parallel processing.
/// Useful for extracting bright and dark features respectively.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `structuring_element` - Structuring element
/// * `top_hat_type` - "white" for white top-hat, "black" for black top-hat
///
/// # Returns
///
/// * Top-hat transformed signal
pub fn parallel_top_hat_transform(
    signal: &[f64],
    structuring_element: &[f64],
    top_hat_type: &str,
) -> SignalResult<Vec<f64>> {
    match top_hat_type {
        "white" => {
            // White top-hat = original - opening
            let opened = parallel_morphological_filter(
                signal,
                structuring_element,
                MorphologicalOperation::Opening,
                None,
            )?;

            let result: Vec<f64> = signal
                .iter()
                .zip(opened.iter())
                .map(|(&orig, &open)| orig - open)
                .collect();

            Ok(result)
        }
        "black" => {
            // Black top-hat = closing - original
            let closed = parallel_morphological_filter(
                signal,
                structuring_element,
                MorphologicalOperation::Closing,
                None,
            )?;

            let result: Vec<f64> = closed
                .iter()
                .zip(signal.iter())
                .map(|(&close, &orig)| close - orig)
                .collect();

            Ok(result)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown top-hat type: {}. Use 'white' or 'black'",
            top_hat_type
        ))),
    }
}

/// Parallel morphological gradient
///
/// Computes morphological gradient using parallel processing.
/// Gradient = dilation - erosion
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `structuring_element` - Structuring element
///
/// # Returns
///
/// * Morphological gradient signal
pub fn parallel_morphological_gradient(
    signal: &[f64],
    structuring_element: &[f64],
) -> SignalResult<Vec<f64>> {
    // Compute dilation and erosion in parallel
    let operations = [
        MorphologicalOperation::Dilation,
        MorphologicalOperation::Erosion,
    ];

    let results: Vec<Vec<f64>> = par_iter_with_setup(
        operations.iter().enumerate(),
        || {},
        |_, (_idx, &operation)| {
            parallel_morphological_filter(signal, structuring_element, operation, None)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    let dilated = &results[0];
    let eroded = &results[1];

    // Compute gradient = dilation - erosion
    let gradient: Vec<f64> = dilated
        .iter()
        .zip(eroded.iter())
        .map(|(&d, &e)| d - e)
        .collect();

    Ok(gradient)
}

/// Create standard structuring elements
///
/// Generates common structuring elements for morphological operations.
///
/// # Arguments
///
/// * `shape` - Shape of structuring element ("line", "disk", "rectangle")
/// * `size` - Size parameter (length for line, radius for disk, etc.)
///
/// # Returns
///
/// * Structuring element as vector
pub fn create_structuring_element(shape: &str, size: usize) -> SignalResult<Vec<f64>> {
    match shape {
        "line" => {
            // Line structuring element
            Ok(vec![1.0; size])
        }
        "disk" => {
            // Approximation of disk structuring element for 1D
            // Use a symmetric pattern that approximates circular shape
            if size == 0 {
                return Err(SignalError::ValueError(
                    "Size must be greater than 0".to_string(),
                ));
            }

            let radius = size as f64;
            let length = 2 * size + 1;
            let mut element = vec![0.0; length];
            let center = size;

            for i in 0..length {
                let distance = (i as f64 - center as f64).abs();
                if distance <= radius {
                    element[i] = 1.0;
                }
            }

            Ok(element)
        }
        "rectangle" => {
            // Rectangle structuring element (same as line for 1D)
            Ok(vec![1.0; size])
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown structuring element shape: {}",
            shape
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_morphological_filter() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let structuring_element = vec![1.0, 1.0, 1.0];

        let result = parallel_morphological_filter(
            &signal,
            &structuring_element,
            MorphologicalOperation::Erosion,
            None,
        );

        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_morphological_operations() {
        let signal = vec![1.0, 3.0, 2.0, 4.0, 1.0];
        let se = vec![1.0, 1.0, 1.0];

        // Test erosion
        let eroded =
            parallel_morphological_filter(&signal, &se, MorphologicalOperation::Erosion, None)
                .unwrap();
        assert_eq!(eroded.len(), signal.len());

        // Test dilation
        let dilated =
            parallel_morphological_filter(&signal, &se, MorphologicalOperation::Dilation, None)
                .unwrap();
        assert_eq!(dilated.len(), signal.len());

        // Test opening
        let opened =
            parallel_morphological_filter(&signal, &se, MorphologicalOperation::Opening, None)
                .unwrap();
        assert_eq!(opened.len(), signal.len());

        // Test closing
        let closed =
            parallel_morphological_filter(&signal, &se, MorphologicalOperation::Closing, None)
                .unwrap();
        assert_eq!(closed.len(), signal.len());
    }

    #[test]
    fn test_parallel_top_hat_transform() {
        let signal = vec![0.0, 1.0, 5.0, 1.0, 0.0, 3.0, 0.0];
        let se = vec![1.0, 1.0, 1.0];

        // Test white top-hat
        let white_top_hat = parallel_top_hat_transform(&signal, &se, "white").unwrap();
        assert_eq!(white_top_hat.len(), signal.len());

        // Test black top-hat
        let black_top_hat = parallel_top_hat_transform(&signal, &se, "black").unwrap();
        assert_eq!(black_top_hat.len(), signal.len());

        // Test invalid type
        let result = parallel_top_hat_transform(&signal, &se, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_morphological_gradient() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let se = vec![1.0, 1.0, 1.0];

        let gradient = parallel_morphological_gradient(&signal, &se).unwrap();
        assert_eq!(gradient.len(), signal.len());

        // Gradient should be non-negative
        for &val in &gradient {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_create_structuring_element() {
        // Test line element
        let line = create_structuring_element("line", 5).unwrap();
        assert_eq!(line.len(), 5);
        assert!(line.iter().all(|&x| x == 1.0));

        // Test disk element
        let disk = create_structuring_element("disk", 3).unwrap();
        assert_eq!(disk.len(), 7); // 2*3 + 1

        // Test invalid shape
        let result = create_structuring_element("invalid", 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_morphological_reconstruction() {
        let marker = vec![0.5, 1.0, 0.5, 2.0, 0.5];
        let mask = vec![1.0, 2.0, 1.0, 3.0, 1.0];
        let se = vec![1.0, 1.0, 1.0];

        let reconstructed = parallel_morphological_reconstruction(
            &marker,
            &mask,
            &se,
            MorphologicalOperation::Dilation,
            10,
        )
        .unwrap();

        assert_eq!(reconstructed.len(), marker.len());

        // Check that result is constrained by mask
        for (i, &val) in reconstructed.iter().enumerate() {
            assert!(val <= mask[i]);
        }
    }

    #[test]
    fn test_apply_erosion_dilation() {
        let signal = vec![1.0, 3.0, 2.0, 4.0, 1.0];
        let se = vec![1.0, 1.0, 1.0];

        let eroded = apply_erosion(&signal, 2, &se, 3);
        let dilated = apply_dilation(&signal, 2, &se, 3);

        assert!(eroded <= signal[2]); // Erosion should not increase values
        assert!(dilated >= signal[2]); // Dilation should not decrease values
    }
}
