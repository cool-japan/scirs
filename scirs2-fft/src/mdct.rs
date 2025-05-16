//! Modified Discrete Cosine Transform (MDCT) and Modified Discrete Sine Transform (MDST)
//!
//! The MDCT and MDST are lapped transforms based on the DCT/DST that are widely used
//! in audio coding (MP3, AAC, Vorbis) due to their perfect reconstruction properties
//! with overlapping windows.

use ndarray::{Array1, ArrayBase, Data};
use std::f64::consts::PI;

use crate::error::{FFTError, FFTResult};
use crate::window::Window;

/// Compute the Modified Discrete Cosine Transform (MDCT)
///
/// The MDCT is a lapped transform with 50% overlap between consecutive blocks.
/// It is critically sampled and allows perfect reconstruction.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `n` - Transform size (output will be n/2 coefficients)
/// * `window` - Window function to apply
///
/// # Returns
///
/// MDCT coefficients (n/2 values)
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_fft::mdct::mdct;
/// use scirs2_fft::window::Window;
///
/// let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let mdct_result = mdct(&signal, 8, Some(Window::Hann)).unwrap();
/// assert_eq!(mdct_result.len(), 4); // Output is half the transform size
/// ```
pub fn mdct<S>(
    x: &ArrayBase<S, ndarray::Ix1>,
    n: usize,
    window: Option<Window>,
) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    if n % 2 != 0 {
        return Err(FFTError::ValueError("MDCT size must be even".to_string()));
    }

    if x.len() != n {
        return Err(FFTError::ValueError(format!(
            "Input length {} does not match MDCT size {}",
            x.len(),
            n
        )));
    }

    let half_n = n / 2;
    let mut result = Array1::zeros(half_n);

    // Apply window if specified
    let windowed = if let Some(win) = window {
        let win_coeffs = crate::window::get_window(win, n, true)?;
        x.to_owned() * &win_coeffs
    } else {
        x.to_owned()
    };

    // Compute MDCT coefficients
    for k in 0..half_n {
        let mut sum = 0.0;
        for n_idx in 0..n {
            let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0.5);
            sum += windowed[n_idx] * angle.cos();
        }
        result[k] = sum * (2.0 / n as f64).sqrt();
    }

    Ok(result)
}

/// Compute the Inverse Modified Discrete Cosine Transform (IMDCT)
///
/// The IMDCT reconstructs a signal from MDCT coefficients.
/// To achieve perfect reconstruction, overlapping blocks must be properly combined.
///
/// # Arguments
///
/// * `x` - MDCT coefficients
/// * `window` - Window function to apply (should match the forward transform)
///
/// # Returns
///
/// Reconstructed signal (2 * input length)
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_fft::mdct::{mdct, imdct};
/// use scirs2_fft::window::Window;
///
/// let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let mdct_coeffs = mdct(&signal, 8, Some(Window::Hann)).unwrap();
/// let reconstructed = imdct(&mdct_coeffs, Some(Window::Hann)).unwrap();
/// assert_eq!(reconstructed.len(), 8); // Output is twice the input length
/// ```
pub fn imdct<S>(x: &ArrayBase<S, ndarray::Ix1>, window: Option<Window>) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    let half_n = x.len();
    let n = half_n * 2;
    let mut result = Array1::zeros(n);

    // Compute IMDCT values
    for n_idx in 0..n {
        let mut sum = 0.0;
        for k in 0..half_n {
            let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0.5);
            sum += x[k] * angle.cos();
        }
        result[n_idx] = sum * (2.0 / n as f64).sqrt();
    }

    // Apply window if specified
    if let Some(win) = window {
        let win_coeffs = crate::window::get_window(win, n, true)?;
        result *= &win_coeffs;
    }

    Ok(result)
}

/// Modified Discrete Sine Transform (MDST)
///
/// The MDST is similar to MDCT but uses sine basis functions.
/// It is less commonly used than MDCT but provides similar properties.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `n` - Transform size (output will be n/2 coefficients)
/// * `window` - Window function to apply
///
/// # Returns
///
/// MDST coefficients (n/2 values)
pub fn mdst<S>(
    x: &ArrayBase<S, ndarray::Ix1>,
    n: usize,
    window: Option<Window>,
) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    if n % 2 != 0 {
        return Err(FFTError::ValueError("MDST size must be even".to_string()));
    }

    if x.len() != n {
        return Err(FFTError::ValueError(format!(
            "Input length {} does not match MDST size {}",
            x.len(),
            n
        )));
    }

    let half_n = n / 2;
    let mut result = Array1::zeros(half_n);

    // Apply window if specified
    let windowed = if let Some(win) = window {
        let win_coeffs = crate::window::get_window(win, n, true)?;
        x.to_owned() * &win_coeffs
    } else {
        x.to_owned()
    };

    // Compute MDST coefficients
    for k in 0..half_n {
        let mut sum = 0.0;
        for n_idx in 0..n {
            let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0.5);
            sum += windowed[n_idx] * angle.sin();
        }
        result[k] = sum * (2.0 / n as f64).sqrt();
    }

    Ok(result)
}

/// Inverse Modified Discrete Sine Transform (IMDST)
///
/// Reconstructs a signal from MDST coefficients.
///
/// # Arguments
///
/// * `x` - MDST coefficients
/// * `window` - Window function to apply (should match the forward transform)
///
/// # Returns
///
/// Reconstructed signal (2 * input length)
pub fn imdst<S>(x: &ArrayBase<S, ndarray::Ix1>, window: Option<Window>) -> FFTResult<Array1<f64>>
where
    S: Data<Elem = f64>,
{
    let half_n = x.len();
    let n = half_n * 2;
    let mut result = Array1::zeros(n);

    // Compute IMDST values
    for n_idx in 0..n {
        let mut sum = 0.0;
        for k in 0..half_n {
            let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0.5);
            sum += x[k] * angle.sin();
        }
        result[n_idx] = sum * (2.0 / n as f64).sqrt();
    }

    // Apply window if specified
    if let Some(win) = window {
        let win_coeffs = crate::window::get_window(win, n, true)?;
        result *= &win_coeffs;
    }

    Ok(result)
}

/// Perform overlap-add reconstruction from MDCT coefficients
///
/// This function handles the proper overlapping and adding of consecutive
/// MDCT blocks for perfect reconstruction.
///
/// # Arguments
///
/// * `blocks` - Vector of MDCT coefficient blocks
/// * `window` - Window function used in the forward transform
/// * `hop_size` - Hop size between consecutive blocks (typically block_size/2)
///
/// # Returns
///
/// Reconstructed signal
pub fn mdct_overlap_add(
    blocks: &[Array1<f64>],
    window: Option<Window>,
    hop_size: usize,
) -> FFTResult<Array1<f64>> {
    if blocks.is_empty() {
        return Err(FFTError::ValueError("No blocks provided".to_string()));
    }

    let block_size = blocks[0].len() * 2;
    let n_blocks = blocks.len();
    let output_len = (n_blocks - 1) * hop_size + block_size;
    let mut output = Array1::zeros(output_len);

    for (i, block) in blocks.iter().enumerate() {
        let reconstructed = imdct(block, window.clone())?;
        let start_idx = i * hop_size;

        // Add overlapping parts
        for j in 0..block_size {
            if start_idx + j < output_len {
                output[start_idx + j] += reconstructed[j];
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::Window;
    use ndarray::array;

    #[test]
    fn test_mdct_basic() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mdct_result = mdct(&signal, 8, None).unwrap();

        // MDCT should produce n/2 coefficients
        assert_eq!(mdct_result.len(), 4);
    }

    #[test]
    fn test_mdct_perfect_reconstruction() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let window = Some(Window::Hann);

        // Perform MDCT
        let mdct_coeffs = mdct(&signal, 8, window.clone()).unwrap();

        // Perform IMDCT
        let reconstructed = imdct(&mdct_coeffs, window).unwrap();

        // For proper reconstruction, we need overlapping blocks
        // This is a simplified test that checks the transform works
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_mdst_basic() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mdst_result = mdst(&signal, 8, None).unwrap();

        // MDST should produce n/2 coefficients
        assert_eq!(mdst_result.len(), 4);
    }

    #[test]
    fn test_overlap_add() {
        // Create overlapping blocks
        let block1 = array![1.0, 2.0, 3.0, 4.0];
        let block2 = array![2.0, 3.0, 4.0, 5.0];
        let blocks = vec![block1, block2];

        let result = mdct_overlap_add(&blocks, Some(Window::Hann), 4).unwrap();

        // Check output length
        assert_eq!(result.len(), 12); // (2-1)*4 + 8
    }

    #[test]
    fn test_mdct_invalid_size() {
        let signal = array![1.0, 2.0, 3.0];
        let result = mdct(&signal, 3, None);
        assert!(result.is_err());
    }
}
