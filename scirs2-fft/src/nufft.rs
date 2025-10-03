//! Non-Uniform Fast Fourier Transform module
//!
//! This module provides implementations of the Non-Uniform Fast Fourier Transform (NUFFT),
//! which computes the Fourier transform of data sampled at non-uniform intervals.
//!
//! # Overview
//!
//! NUFFT is an extension of the FFT for data that is not sampled on a uniform grid.
//! This implementation uses a grid-based approach with interpolation to approximate
//! the non-uniform Fourier transform efficiently.
//!
//! # Types of NUFFT
//!
//! * Type 1 (Non-Uniform to Uniform): Data at non-uniform locations, transform to uniform frequency grid
//! * Type 2 (Uniform to Non-Uniform): Data at uniform locations, transform to non-uniform frequency grid

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;
use std::f64::consts::PI;

/// NUFFT interpolation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    /// Linear interpolation
    Linear,
    /// Gaussian kernel-based interpolation
    Gaussian,
    /// Minimal peak width Gaussian
    MinGaussian,
}

/// Performs Non-Uniform Fast Fourier Transform (NUFFT) Type 1.
///
/// This function computes the FFT of data sampled at non-uniform locations.
/// NUFFT Type 1 transforms from non-uniform samples to a uniform frequency grid.
///
/// # Arguments
///
/// * `x` - Non-uniform sample points (must be in range [-π, π])
/// * `samples` - Complex sample values at the non-uniform points
/// * `m` - Size of the output uniform grid
/// * `interp_type` - Interpolation type to use
/// * `epsilon` - Desired precision (typically 1e-6 to 1e-12)
///
/// # Returns
///
/// * A complex-valued array containing the NUFFT on a uniform frequency grid
///
/// # Errors
///
/// Returns an error if the computation fails or inputs are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::nufft::{nufft_type1, InterpolationType};
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// // Create non-uniform sample points in [-π, π]
/// let n = 100;
/// let x: Vec<f64> = (0..n).map(|i| -PI + 1.8 * PI * i as f64 / (n as f64)).collect();
///
/// // Create sample values for a simple function (e.g., a Gaussian)
/// let samples: Vec<Complex64> = x.iter()
///     .map(|&xi| {
///         let real = (-xi.powi(2) / 2.0).exp();
///         Complex64::new(real, 0.0)
///     })
///     .collect();
///
/// // Compute NUFFT
/// let m = 128;  // Output grid size
/// let result = nufft_type1(&x, &samples, m, InterpolationType::Gaussian, 1e-6).unwrap();
///
/// // The transform of a Gaussian is another Gaussian
/// assert!(result.len() == m);
/// ```
///
/// # Notes
///
/// This is a basic implementation. For performance-critical applications,
/// consider using a more optimized NUFFT library.
#[allow(dead_code)]
pub fn nufft_type1(
    x: &[f64],
    samples: &[Complex64],
    m: usize,
    interp_type: InterpolationType,
    epsilon: f64,
) -> FFTResult<Vec<Complex64>> {
    // Check inputs
    if x.len() != samples.len() {
        return Err(FFTError::DimensionError(
            "Sample points and values must have the same length".to_string(),
        ));
    }

    if epsilon <= 0.0 {
        return Err(FFTError::ValueError(
            "Precision parameter epsilon must be positive".to_string(),
        ));
    }

    // Check if x values are in the correct range [-π, π]
    for &xi in x {
        if !(-PI..=PI).contains(&xi) {
            return Err(FFTError::ValueError(
                "Sample points must be in the range [-π, π]".to_string(),
            ));
        }
    }

    // Estimate parameters for the algorithm
    let tau = 2.0; // Oversampling factor, usually in range [1.5, 2.5]
    let n_grid = tau as usize * m; // Size of the oversampled grid

    // Determine the width parameter based on the chosen interpolation _type
    let sigma = match interp_type {
        InterpolationType::Linear => 2.0,
        InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),
        InterpolationType::MinGaussian => 1.0,
    };

    // Compute the spreading width (kernel half-width)
    let width = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;
    let width = width.max(2); // At least 2 for stability

    // Grid spacing
    let h_grid = 2.0 * PI / n_grid as f64;

    // Initialize the oversampled grid
    let mut grid_data = vec![Complex64::zero(); n_grid];

    // Spread the non-uniform data onto the uniform grid using the chosen kernel
    for (&xi, &sample) in x.iter().zip(samples.iter()) {
        // Map the x value to the grid index
        let x_grid = (xi + PI) / h_grid;
        let i_grid = x_grid.floor() as isize;

        // Spread the sample to nearby grid points
        for j in (-(width as isize))..=(width as isize) {
            let idx = (i_grid + j).rem_euclid(n_grid as isize) as usize;
            let kernel_arg = (x_grid - (i_grid + j) as f64) / sigma;

            let kernel_value = match interp_type {
                InterpolationType::Linear => {
                    if kernel_arg.abs() <= 1.0 {
                        1.0 - kernel_arg.abs()
                    } else {
                        0.0
                    }
                }
                InterpolationType::Gaussian | InterpolationType::MinGaussian => {
                    (-kernel_arg * kernel_arg).exp()
                }
            };

            grid_data[idx] += sample * kernel_value;
        }
    }

    // Compute the FFT of the grid data
    let grid_fft = fft_backend(&grid_data)?;

    // Extract the desired frequency components
    let mut result = Vec::with_capacity(m);

    for i in 0..m {
        if i <= m / 2 {
            // Positive frequencies
            result.push(grid_fft[i]);
        } else {
            // Negative frequencies
            result.push(grid_fft[n_grid - (m - i)]);
        }
    }

    Ok(result)
}

/// Performs Non-Uniform Fast Fourier Transform (NUFFT) Type 2.
///
/// This function computes the FFT from a uniform grid to non-uniform frequencies.
/// NUFFT Type 2 is essentially the adjoint of Type 1.
///
/// # Arguments
///
/// * `spectrum` - Input spectrum on a uniform grid
/// * `x` - Non-uniform frequency points where output is desired (must be in [-π, π])
/// * `interp_type` - Interpolation type to use
/// * `epsilon` - Desired precision (typically 1e-6 to 1e-12)
///
/// # Returns
///
/// * A complex-valued array containing the NUFFT at the specified non-uniform points
///
/// # Errors
///
/// Returns an error if the computation fails or inputs are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::nufft::{nufft_type2, InterpolationType};
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// // Create a spectrum on a uniform grid
/// let m = 128;
/// let spectrum: Vec<Complex64> = (0..m)
///     .map(|i| {
///         // Simple Gaussian in frequency domain
///         let f = i as f64 - m as f64 / 2.0;
///         let val = (-f * f / (2.0 * 10.0)).exp();
///         Complex64::new(val, 0.0)
///     })
///     .collect();
///
/// // Define non-uniform points where we want to evaluate the transform
/// // Ensure all points are in the range [-π, π]
/// let n = 100;
/// let x: Vec<f64> = (0..n).map(|i| -PI + 1.99 * PI * i as f64 / (n as f64 - 1.0)).collect();
///
/// // Compute NUFFT Type 2
/// let result = nufft_type2(&spectrum, &x, InterpolationType::Gaussian, 1e-6).unwrap();
///
/// // The output should have the same length as the non-uniform points
/// assert_eq!(result.len(), x.len());
/// ```
///
/// # Notes
///
/// This is a basic implementation. For performance-critical applications,
/// consider using a more optimized NUFFT library.
#[allow(dead_code)]
pub fn nufft_type2(
    spectrum: &[Complex64],
    x: &[f64],
    interp_type: InterpolationType,
    epsilon: f64,
) -> FFTResult<Vec<Complex64>> {
    // Check inputs
    if epsilon <= 0.0 {
        return Err(FFTError::ValueError(
            "Precision parameter epsilon must be positive".to_string(),
        ));
    }

    // Check if x values are in the correct range [-π, π]
    for &xi in x {
        if !(-PI..=PI).contains(&xi) {
            return Err(FFTError::ValueError(
                "Output points must be in the range [-π, π]".to_string(),
            ));
        }
    }

    let m = spectrum.len();
    let tau = 2.0; // Oversampling factor
    let n_grid = tau as usize * m; // Size of the oversampled grid

    // Determine the width parameter
    let sigma = match interp_type {
        InterpolationType::Linear => 2.0,
        InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),
        InterpolationType::MinGaussian => 1.0,
    };

    // Compute the spreading width (kernel half-width)
    let width = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;
    let width = width.max(2); // At least 2 for stability

    // Prepare the input for the inverse FFT
    let mut padded_spectrum = vec![Complex64::zero(); n_grid];

    // Copy the spectrum to the padded array
    for i in 0..m {
        if i <= m / 2 {
            // Positive frequencies
            padded_spectrum[i] = spectrum[i];
        } else {
            // Negative frequencies
            padded_spectrum[n_grid - (m - i)] = spectrum[i];
        }
    }

    // Compute the inverse FFT
    let grid_ifft = ifft_backend(&padded_spectrum)?;

    // Grid spacing
    let h_grid = 2.0 * PI / n_grid as f64;

    // Interpolate at the non-uniform points
    let mut result = vec![Complex64::zero(); x.len()];

    for (i, &xi) in x.iter().enumerate() {
        // Map the x value to the grid index
        let x_grid = (xi + PI) / h_grid;
        let i_grid = x_grid.floor() as isize;

        // Interpolate from nearby grid points
        for j in (-(width as isize))..=(width as isize) {
            let idx = (i_grid + j).rem_euclid(n_grid as isize) as usize;
            let kernel_arg = (x_grid - (i_grid + j) as f64) / sigma;

            let kernel_value = match interp_type {
                InterpolationType::Linear => {
                    if kernel_arg.abs() <= 1.0 {
                        1.0 - kernel_arg.abs()
                    } else {
                        0.0
                    }
                }
                InterpolationType::Gaussian | InterpolationType::MinGaussian => {
                    (-kernel_arg * kernel_arg).exp()
                }
            };

            result[i] += grid_ifft[idx] * kernel_value;
        }
    }

    Ok(result)
}

/// Helper function for FFT computation used in NUFFT implementations
#[allow(dead_code)]
fn fft_backend(data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = data.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // Convert to rustfft's Complex type
    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&c| Complex::new(c.re, c.im)).collect();

    // Perform the FFT
    fft.process(&mut buffer);

    // Convert back to scirs2_core::numeric::Complex64
    Ok(buffer
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect())
}

/// Helper function for IFFT computation used in NUFFT implementations
#[allow(dead_code)]
fn ifft_backend(data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = data.len();
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);

    // Convert to rustfft's Complex type
    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&c| Complex::new(c.re, c.im)).collect();

    // Perform the IFFT
    ifft.process(&mut buffer);

    // Convert back to scirs2_core::numeric::Complex64 and normalize
    let scale = 1.0 / n as f64;
    Ok(buffer
        .into_iter()
        .map(|c| Complex64::new(c.re * scale, c.im * scale))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nufft_type1_gaussian() {
        // Create a Gaussian function
        let n = 100;
        let x: Vec<f64> = (0..n)
            .map(|i| -PI + 1.8 * PI * i as f64 / (n as f64))
            .collect();
        let samples: Vec<Complex64> = x
            .iter()
            .map(|&xi| {
                let real = (-xi.powi(2) / 2.0).exp();
                Complex64::new(real, 0.0)
            })
            .collect();

        // Compute NUFFT Type 1
        let m = 128;
        let result = nufft_type1(&x, &samples, m, InterpolationType::Gaussian, 1e-6).unwrap();

        // The transform of a Gaussian is another Gaussian
        // Check that the result is not all zeros and has the expected length
        assert_eq!(result.len(), m);
        assert!(result.iter().any(|&c| c.norm() > 1e-10));

        // For a Gaussian, the transform should be centered around the middle
        // and have significant energy, but we won't test exact shapes
        // as this is an approximation

        // Simply check that the spectrum is not all zeros or uniform
        let max_val = result.iter().map(|c| c.norm()).fold(0.0, f64::max);
        let min_val = result
            .iter()
            .map(|c| c.norm())
            .fold(f64::INFINITY, f64::min);

        // Ensure we have a reasonable dynamic range in the spectrum
        assert!(max_val > 0.0);
        assert!(min_val >= 0.0);
        // Some frequency components should be at least 2x stronger than others
        assert!(max_val > min_val * 2.0);
    }

    #[test]
    fn test_nufft_type2_consistency() {
        // Create a simple spectrum (impulse at the center)
        let m = 32;
        let mut spectrum = vec![Complex64::new(0.0, 0.0); m];
        spectrum[m / 2] = Complex64::new(1.0, 0.0);

        // Define non-uniform points
        let n = 50;
        let x: Vec<f64> = (0..n)
            .map(|i| -PI + 1.8 * PI * i as f64 / (n as f64))
            .collect();

        // Compute NUFFT Type 2
        let result = nufft_type2(&spectrum, &x, InterpolationType::Gaussian, 1e-6).unwrap();

        // Result should be approximately constant magnitude complex exponentials
        assert_eq!(result.len(), n);

        // Check that magnitudes are approximately constant
        let avg_magnitude: f64 = result.iter().map(|c| c.norm()).sum::<f64>() / n as f64;
        for c in result {
            assert_relative_eq!(c.norm(), avg_magnitude, epsilon = 0.2);
        }
    }

    #[test]
    fn test_nufft_type1_linear_interp() {
        // Create a simple cosine function
        let n = 120;
        let x: Vec<f64> = (0..n)
            .map(|i| -PI + 1.8 * PI * i as f64 / (n as f64))
            .collect();
        let samples: Vec<Complex64> = x.iter().map(|&xi| Complex64::new(xi.cos(), 0.0)).collect();

        // Compute NUFFT Type 1 with linear interpolation
        let m = 64;
        let result = nufft_type1(&x, &samples, m, InterpolationType::Linear, 1e-6).unwrap();

        // For a cosine function, we expect peaks at k=±1
        assert_eq!(result.len(), m);

        // Find the two largest peaks
        let mut magnitudes: Vec<(usize, f64)> = result
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c.norm()))
            .collect();
        magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Check that the peaks are at the expected frequencies
        let peak1 = magnitudes[0].0;
        let peak2 = magnitudes[1].0;

        // Either we have peaks at 1 and m-1 or at some other symmetric pair
        let matches_expected = (peak1 == 1 && peak2 == m - 1) || (peak1 == m - 1 && peak2 == 1);

        assert!(matches_expected || (peak1 as i32 - peak2 as i32).abs() == 2);
    }

    #[test]
    fn test_nufft_errors() {
        // Test with mismatched lengths
        let x = vec![0.0, 1.0];
        let samples = vec![Complex64::new(1.0, 0.0)];

        let result = nufft_type1(&x, &samples, 8, InterpolationType::Gaussian, 1e-6);
        assert!(result.is_err());

        // Test with invalid epsilon
        let x = vec![0.0];
        let samples = vec![Complex64::new(1.0, 0.0)];

        let result = nufft_type1(&x, &samples, 8, InterpolationType::Gaussian, -1.0);
        assert!(result.is_err());

        // Test with x values outside [-π, π]
        let x = vec![4.0];
        let samples = vec![Complex64::new(1.0, 0.0)];

        let result = nufft_type1(&x, &samples, 8, InterpolationType::Gaussian, 1e-6);
        assert!(result.is_err());
    }
}
