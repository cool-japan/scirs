//! Morphological filtering operations for denoising
//!
//! This module provides morphological operations (erosion, dilation, opening, closing)
//! that can be used for signal denoising and shape-based filtering.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;

/// Morphological opening for denoising (erosion followed by dilation)
pub fn denoise_morphological_opening(
    signal: &Array1<f64>,
    radius: usize,
) -> SignalResult<Array1<f64>> {
    if radius == 0 {
        return Err(SignalError::ValueError(
            "Radius must be greater than 0".to_string(),
        ));
    }

    let eroded = morphological_erosion(signal, radius);
    let opened = morphological_dilation(&eroded, radius);
    Ok(opened)
}

/// Morphological closing for denoising (dilation followed by erosion)
pub fn denoise_morphological_closing(
    signal: &Array1<f64>,
    radius: usize,
) -> SignalResult<Array1<f64>> {
    if radius == 0 {
        return Err(SignalError::ValueError(
            "Radius must be greater than 0".to_string(),
        ));
    }

    let dilated = morphological_dilation(signal, radius);
    let closed = morphological_erosion(&dilated, radius);
    Ok(closed)
}

/// Morphological erosion operation
pub fn morphological_erosion(signal: &Array1<f64>, radius: usize) -> Array1<f64> {
    let n = signal.len();
    let mut eroded = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(n);

        let mut min_val = f64::INFINITY;
        for j in start..end {
            min_val = min_val.min(signal[j]);
        }

        eroded[i] = min_val;
    }

    eroded
}

/// Morphological dilation operation
pub fn morphological_dilation(signal: &Array1<f64>, radius: usize) -> Array1<f64> {
    let n = signal.len();
    let mut dilated = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(n);

        let mut max_val = f64::NEG_INFINITY;
        for j in start..end {
            max_val = max_val.max(signal[j]);
        }

        dilated[i] = max_val;
    }

    dilated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphological_opening() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
        let result = denoise_morphological_opening(&signal, 1);
        assert!(result.is_ok());
        let opened = result.unwrap();
        assert_eq!(opened.len(), signal.len());
    }

    #[test]
    fn test_morphological_closing() {
        let signal = Array1::from_vec(vec![4.0, 2.0, 3.0, 1.0, 4.0]);
        let result = denoise_morphological_closing(&signal, 1);
        assert!(result.is_ok());
        let closed = result.unwrap();
        assert_eq!(closed.len(), signal.len());
    }

    #[test]
    fn test_erosion_dilation() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);

        let eroded = morphological_erosion(&signal, 1);
        let dilated = morphological_dilation(&signal, 1);

        assert_eq!(eroded.len(), signal.len());
        assert_eq!(dilated.len(), signal.len());

        // Erosion should not increase values
        for i in 0..signal.len() {
            assert!(eroded[i] <= signal[i]);
        }

        // Dilation should not decrease values
        for i in 0..signal.len() {
            assert!(dilated[i] >= signal[i]);
        }
    }

    #[test]
    fn test_error_conditions() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = denoise_morphological_opening(&signal, 0);
        assert!(result.is_err());

        let result = denoise_morphological_closing(&signal, 0);
        assert!(result.is_err());
    }
}
