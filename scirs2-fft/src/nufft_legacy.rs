//! Legacy NUFFT API preserved for backward compatibility.
//!
//! The original `nufft.rs` API exposed `nufft_type1` / `nufft_type2` with an
//! `InterpolationType` parameter.  That API is retained here under the names
//! `nufft_type1_legacy` / `nufft_type2_legacy` and re-exported from
//! [`crate::nufft`] as `nufft_type1_legacy` / `nufft_type2_legacy`.

use crate::error::{FFTError, FFTResult};
#[cfg(feature = "oxifft")]
use crate::oxifft_plan_cache;
#[cfg(feature = "oxifft")]
use oxifft::{Complex as OxiComplex, Direction};
// NOTE: rustfft-backend removed; this block is now dead code
#[cfg(not(feature = "oxifft"))]
use rustfft::{num_complex::Complex as RustComplex, FftPlanner};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;
use std::f64::consts::PI;

/// NUFFT interpolation type (legacy API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    /// Linear interpolation
    Linear,
    /// Gaussian kernel-based interpolation
    Gaussian,
    /// Minimal peak width Gaussian
    MinGaussian,
}

/// Legacy Type-1 NUFFT with explicit `InterpolationType`.
///
/// Prefer [`crate::nufft::nufft_type1`] for new code.
///
/// # Arguments
///
/// * `x`          – non-uniform sample points in `[-π, π]`
/// * `samples`    – complex sample values
/// * `m`          – output grid size
/// * `interp_type` – interpolation kernel
/// * `epsilon`    – desired precision
///
/// # Errors
///
/// Returns an error on dimension mismatch, invalid epsilon, or out-of-range `x`.
pub fn nufft_type1(
    x: &[f64],
    samples: &[Complex64],
    m: usize,
    interp_type: InterpolationType,
    epsilon: f64,
) -> FFTResult<Vec<Complex64>> {
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
    for &xi in x {
        if !(-PI..=PI).contains(&xi) {
            return Err(FFTError::ValueError(
                "Sample points must be in the range [-π, π]".to_string(),
            ));
        }
    }

    let tau = 2.0_f64;
    let n_grid = tau as usize * m;

    let sigma = match interp_type {
        InterpolationType::Linear => 2.0,
        InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),
        InterpolationType::MinGaussian => 1.0,
    };

    let width = {
        let raw = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;
        raw.max(2)
    };

    let h_grid = 2.0 * PI / n_grid as f64;
    let mut grid_data = vec![Complex64::zero(); n_grid];

    for (&xi, &sample) in x.iter().zip(samples.iter()) {
        let x_grid = (xi + PI) / h_grid;
        let i_grid = x_grid.floor() as isize;

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

    let grid_fft = fft_backend_legacy(&grid_data)?;

    let mut result = Vec::with_capacity(m);
    for i in 0..m {
        if i <= m / 2 {
            result.push(grid_fft[i]);
        } else {
            result.push(grid_fft[n_grid - (m - i)]);
        }
    }

    Ok(result)
}

/// Legacy Type-2 NUFFT with explicit `InterpolationType`.
///
/// Prefer [`crate::nufft::nufft_type2`] for new code.
///
/// # Arguments
///
/// * `spectrum`   – input spectrum on a uniform grid
/// * `x`          – non-uniform output points in `[-π, π]`
/// * `interp_type` – interpolation kernel
/// * `epsilon`    – desired precision
///
/// # Errors
///
/// Returns an error on invalid epsilon or out-of-range `x`.
pub fn nufft_type2(
    spectrum: &[Complex64],
    x: &[f64],
    interp_type: InterpolationType,
    epsilon: f64,
) -> FFTResult<Vec<Complex64>> {
    if epsilon <= 0.0 {
        return Err(FFTError::ValueError(
            "Precision parameter epsilon must be positive".to_string(),
        ));
    }
    for &xi in x {
        if !(-PI..=PI).contains(&xi) {
            return Err(FFTError::ValueError(
                "Output points must be in the range [-π, π]".to_string(),
            ));
        }
    }

    let m = spectrum.len();
    let tau = 2.0_f64;
    let n_grid = tau as usize * m;

    let sigma = match interp_type {
        InterpolationType::Linear => 2.0,
        InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),
        InterpolationType::MinGaussian => 1.0,
    };

    let width = {
        let raw = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;
        raw.max(2)
    };

    let mut padded_spectrum = vec![Complex64::zero(); n_grid];
    for i in 0..m {
        if i <= m / 2 {
            padded_spectrum[i] = spectrum[i];
        } else {
            padded_spectrum[n_grid - (m - i)] = spectrum[i];
        }
    }

    let grid_ifft = ifft_backend_legacy(&padded_spectrum)?;

    let h_grid = 2.0 * PI / n_grid as f64;
    let mut result = vec![Complex64::zero(); x.len()];

    for (i, &xi) in x.iter().enumerate() {
        let x_grid = (xi + PI) / h_grid;
        let i_grid = x_grid.floor() as isize;

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

// ─── Internal FFT backend helpers ────────────────────────────────────────────

fn fft_backend_legacy(data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = data.len();

    #[cfg(feature = "oxifft")]
    {
        let input_oxi: Vec<OxiComplex<f64>> =
            data.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
        let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); n];
        oxifft_plan_cache::execute_c2c(&input_oxi, &mut output, Direction::Forward)?;
        Ok(output.into_iter().map(|c| Complex64::new(c.re, c.im)).collect())
    }

    // NOTE: rustfft-backend removed; this block is now dead code
    #[cfg(not(feature = "oxifft"))]
    {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut buffer: Vec<RustComplex<f64>> =
            data.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();
        fft.process(&mut buffer);
        Ok(buffer.into_iter().map(|c| Complex64::new(c.re, c.im)).collect())
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = n;
        Err(FFTError::ComputationError(
            "No FFT backend available. Enable 'oxifft' or 'rustfft-backend'.".to_string(),
        ))
    }
}

fn ifft_backend_legacy(data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = data.len();

    #[cfg(feature = "oxifft")]
    {
        let input_oxi: Vec<OxiComplex<f64>> =
            data.iter().map(|c| OxiComplex::new(c.re, c.im)).collect();
        let mut output: Vec<OxiComplex<f64>> = vec![OxiComplex::zero(); n];
        oxifft_plan_cache::execute_c2c(&input_oxi, &mut output, Direction::Backward)?;
        let scale = 1.0 / n as f64;
        Ok(output
            .into_iter()
            .map(|c| Complex64::new(c.re * scale, c.im * scale))
            .collect())
    }

    // NOTE: rustfft-backend removed; this block is now dead code
    #[cfg(not(feature = "oxifft"))]
    {
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);
        let mut buffer: Vec<RustComplex<f64>> =
            data.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();
        ifft.process(&mut buffer);
        let scale = 1.0 / n as f64;
        Ok(buffer
            .into_iter()
            .map(|c| Complex64::new(c.re * scale, c.im * scale))
            .collect())
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = n;
        Err(FFTError::ComputationError(
            "No FFT backend available. Enable 'oxifft' or 'rustfft-backend'.".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_legacy_type1_gaussian() {
        let n = 100usize;
        let x: Vec<f64> = (0..n)
            .map(|i| -PI + 1.8 * PI * i as f64 / n as f64)
            .collect();
        let samples: Vec<Complex64> = x
            .iter()
            .map(|&xi| Complex64::new((-xi.powi(2) / 2.0).exp(), 0.0))
            .collect();

        let m = 128usize;
        let result =
            nufft_type1(&x, &samples, m, InterpolationType::Gaussian, 1e-6).expect("type1");
        assert_eq!(result.len(), m);
        assert!(result.iter().any(|&c| c.norm() > 1e-10));

        let max_val = result.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
        let min_val = result
            .iter()
            .map(|c| c.norm())
            .fold(f64::INFINITY, f64::min);
        assert!(max_val > min_val * 2.0);
    }

    #[test]
    fn test_legacy_type2_consistency() {
        let m = 32usize;
        let mut spectrum = vec![Complex64::new(0.0, 0.0); m];
        spectrum[m / 2] = Complex64::new(1.0, 0.0);

        let n = 50usize;
        let x: Vec<f64> = (0..n)
            .map(|i| -PI + 1.8 * PI * i as f64 / n as f64)
            .collect();

        let result =
            nufft_type2(&spectrum, &x, InterpolationType::Gaussian, 1e-6).expect("type2");
        assert_eq!(result.len(), n);

        let avg_magnitude: f64 = result.iter().map(|c| c.norm()).sum::<f64>() / n as f64;
        for c in &result {
            assert_relative_eq!(c.norm(), avg_magnitude, epsilon = 0.25);
        }
    }

    #[test]
    fn test_legacy_errors() {
        let x = vec![0.0, 1.0];
        let samples = vec![Complex64::new(1.0, 0.0)];
        let res = nufft_type1(&x, &samples, 8, InterpolationType::Gaussian, 1e-6);
        assert!(res.is_err());

        let x2 = vec![0.0];
        let samples2 = vec![Complex64::new(1.0, 0.0)];
        let res2 = nufft_type1(&x2, &samples2, 8, InterpolationType::Gaussian, -1.0);
        assert!(res2.is_err());

        let x3 = vec![4.0];
        let samples3 = vec![Complex64::new(1.0, 0.0)];
        let res3 = nufft_type1(&x3, &samples3, 8, InterpolationType::Gaussian, 1e-6);
        assert!(res3.is_err());
    }
}
