//! Fast Fourier Transform operations for spectral methods
//!
//! This module provides FFT operations for 2D and 3D arrays using simplified DFT implementations.
//! In practice, optimized FFT libraries like FFTW would be used for better performance.

use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array2, Array3};
use scirs2_core::numeric::Complex;

/// Type alias for FFT result
pub type FFTResult<T> = IntegrateResult<T>;

/// FFT operations trait for spectral methods
pub struct FFTOperations;

impl FFTOperations {
    /// Forward 2D FFT (simplified implementation)
    ///
    /// This is a simplified FFT implementation using DFT for demonstration.
    /// In practice, would use optimized FFT library like FFTW.
    #[allow(dead_code)]
    pub fn fft_2d_forward(field: &Array2<f64>) -> FFTResult<Array2<Complex<f64>>> {
        let (nx, ny) = field.dim();
        let mut result = Array2::zeros((nx, ny));

        // Simplified FFT implementation using DFT for demonstration
        // In practice, would use optimized FFT library like FFTW
        for kx in 0..nx {
            for ky in 0..ny {
                let mut sum = Complex::new(0.0, 0.0);

                for x in 0..nx {
                    for y in 0..ny {
                        let phase = -2.0
                            * std::f64::consts::PI
                            * (kx as f64 * x as f64 / nx as f64 + ky as f64 * y as f64 / ny as f64);
                        let exp_factor = Complex::new(phase.cos(), phase.sin());
                        sum += field[[x, y]] * exp_factor;
                    }
                }

                result[[kx, ky]] = sum;
            }
        }

        Ok(result)
    }

    /// Backward 2D FFT (simplified implementation)
    ///
    /// Performs inverse FFT to transform from frequency domain back to spatial domain.
    #[allow(dead_code)]
    pub fn fft_2d_backward(fieldhat: &Array2<Complex<f64>>) -> FFTResult<Array2<f64>> {
        let (nx, ny) = fieldhat.dim();
        let mut result = Array2::zeros((nx, ny));
        let norm = 1.0 / (nx * ny) as f64;

        for x in 0..nx {
            for y in 0..ny {
                let mut sum = Complex::new(0.0, 0.0);

                for kx in 0..nx {
                    for ky in 0..ny {
                        let phase = 2.0
                            * std::f64::consts::PI
                            * (kx as f64 * x as f64 / nx as f64 + ky as f64 * y as f64 / ny as f64);
                        let exp_factor = Complex::new(phase.cos(), phase.sin());
                        sum += fieldhat[[kx, ky]] * exp_factor;
                    }
                }

                result[[x, y]] = sum.re * norm;
            }
        }

        Ok(result)
    }

    /// Forward 3D FFT (simplified implementation)
    ///
    /// This is a simplified 3D FFT implementation using DFT for demonstration.
    /// In practice, would use optimized FFT library like FFTW.
    #[allow(dead_code)]
    pub fn fft_3d_forward(field: &Array3<f64>) -> FFTResult<Array3<Complex<f64>>> {
        let (nx, ny, nz) = field.dim();
        let mut result = Array3::zeros((nx, ny, nz));

        for kx in 0..nx {
            for ky in 0..ny {
                for kz in 0..nz {
                    let mut sum = Complex::new(0.0, 0.0);

                    for x in 0..nx {
                        for y in 0..ny {
                            for z in 0..nz {
                                let phase = -2.0
                                    * std::f64::consts::PI
                                    * (kx as f64 * x as f64 / nx as f64
                                        + ky as f64 * y as f64 / ny as f64
                                        + kz as f64 * z as f64 / nz as f64);
                                let exp_factor = Complex::new(phase.cos(), phase.sin());
                                sum += field[[x, y, z]] * exp_factor;
                            }
                        }
                    }

                    result[[kx, ky, kz]] = sum;
                }
            }
        }

        Ok(result)
    }

    /// Backward 3D FFT (simplified implementation)
    ///
    /// Performs inverse 3D FFT to transform from frequency domain back to spatial domain.
    #[allow(dead_code)]
    pub fn fft_3d_backward(fieldhat: &Array3<Complex<f64>>) -> FFTResult<Array3<f64>> {
        let (nx, ny, nz) = fieldhat.dim();
        let mut result = Array3::zeros((nx, ny, nz));
        let norm = 1.0 / (nx * ny * nz) as f64;

        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let mut sum = Complex::new(0.0, 0.0);

                    for kx in 0..nx {
                        for ky in 0..ny {
                            for kz in 0..nz {
                                let phase = 2.0
                                    * std::f64::consts::PI
                                    * (kx as f64 * x as f64 / nx as f64
                                        + ky as f64 * y as f64 / ny as f64
                                        + kz as f64 * z as f64 / nz as f64);
                                let exp_factor = Complex::new(phase.cos(), phase.sin());
                                sum += fieldhat[[kx, ky, kz]] * exp_factor;
                            }
                        }
                    }

                    result[[x, y, z]] = sum.re * norm;
                }
            }
        }

        Ok(result)
    }

    /// Compute energy spectrum from 2D field
    ///
    /// Useful for analyzing turbulent flows and energy cascades.
    pub fn compute_energy_spectrum_2d(field: &Array2<f64>) -> FFTResult<Vec<f64>> {
        let field_hat = Self::fft_2d_forward(field)?;
        let (nx, ny) = field_hat.dim();

        let max_k = (nx.min(ny) / 2) as f64;
        let n_bins = (max_k as usize).min(50);
        let mut spectrum = vec![0.0; n_bins];
        let mut counts = vec![0; n_bins];

        for i in 0..nx {
            for j in 0..ny {
                let kx = if i <= nx / 2 {
                    i as f64
                } else {
                    (i as i32 - nx as i32) as f64
                };
                let ky = if j <= ny / 2 {
                    j as f64
                } else {
                    (j as i32 - ny as i32) as f64
                };

                let k_mag = (kx * kx + ky * ky).sqrt();
                let bin = ((k_mag / max_k) * (n_bins as f64)).floor() as usize;

                if bin < n_bins {
                    let energy = field_hat[[i, j]].norm_sqr();
                    spectrum[bin] += energy;
                    counts[bin] += 1;
                }
            }
        }

        // Normalize by bin counts
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                spectrum[i] /= *count as f64;
            }
        }

        Ok(spectrum)
    }

    /// Compute energy spectrum from 3D field
    ///
    /// Useful for analyzing 3D turbulent flows and energy cascades.
    pub fn compute_energy_spectrum_3d(field: &Array3<f64>) -> FFTResult<Vec<f64>> {
        let field_hat = Self::fft_3d_forward(field)?;
        let (nx, ny, nz) = field_hat.dim();

        let max_k = (nx.min(ny).min(nz) / 2) as f64;
        let n_bins = (max_k as usize).min(50);
        let mut spectrum = vec![0.0; n_bins];
        let mut counts = vec![0; n_bins];

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let kx = if i <= nx / 2 {
                        i as f64
                    } else {
                        (i as i32 - nx as i32) as f64
                    };
                    let ky = if j <= ny / 2 {
                        j as f64
                    } else {
                        (j as i32 - ny as i32) as f64
                    };
                    let kz = if k <= nz / 2 {
                        k as f64
                    } else {
                        (k as i32 - nz as i32) as f64
                    };

                    let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                    let bin = ((k_mag / max_k) * (n_bins as f64)).floor() as usize;

                    if bin < n_bins {
                        let energy = field_hat[[i, j, k]].norm_sqr();
                        spectrum[bin] += energy;
                        counts[bin] += 1;
                    }
                }
            }
        }

        // Normalize by bin counts
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                spectrum[i] /= *count as f64;
            }
        }

        Ok(spectrum)
    }

    /// Apply low-pass filter in frequency domain
    ///
    /// Filters out high-frequency components above the cutoff.
    pub fn low_pass_filter_2d(
        field: &Array2<f64>,
        cutoff_frequency: f64,
    ) -> FFTResult<Array2<f64>> {
        let field_hat = Self::fft_2d_forward(field)?;
        let (nx, ny) = field_hat.dim();
        let mut filtered_hat = field_hat.clone();

        for i in 0..nx {
            for j in 0..ny {
                let kx = if i <= nx / 2 {
                    i as f64
                } else {
                    (i as i32 - nx as i32) as f64
                };
                let ky = if j <= ny / 2 {
                    j as f64
                } else {
                    (j as i32 - ny as i32) as f64
                };

                let k_mag = (kx * kx + ky * ky).sqrt();

                if k_mag > cutoff_frequency {
                    filtered_hat[[i, j]] = Complex::new(0.0, 0.0);
                }
            }
        }

        Self::fft_2d_backward(&filtered_hat)
    }

    /// Apply high-pass filter in frequency domain
    ///
    /// Filters out low-frequency components below the cutoff.
    pub fn high_pass_filter_2d(
        field: &Array2<f64>,
        cutoff_frequency: f64,
    ) -> FFTResult<Array2<f64>> {
        let field_hat = Self::fft_2d_forward(field)?;
        let (nx, ny) = field_hat.dim();
        let mut filtered_hat = field_hat.clone();

        for i in 0..nx {
            for j in 0..ny {
                let kx = if i <= nx / 2 {
                    i as f64
                } else {
                    (i as i32 - nx as i32) as f64
                };
                let ky = if j <= ny / 2 {
                    j as f64
                } else {
                    (j as i32 - ny as i32) as f64
                };

                let k_mag = (kx * kx + ky * ky).sqrt();

                if k_mag < cutoff_frequency {
                    filtered_hat[[i, j]] = Complex::new(0.0, 0.0);
                }
            }
        }

        Self::fft_2d_backward(&filtered_hat)
    }

    /// Compute phase correlations between two fields
    ///
    /// Useful for analyzing flow structures and coherent patterns.
    pub fn phase_correlation_2d(
        field1: &Array2<f64>,
        field2: &Array2<f64>,
    ) -> FFTResult<Array2<f64>> {
        let field1_hat = Self::fft_2d_forward(field1)?;
        let field2_hat = Self::fft_2d_forward(field2)?;
        let (nx, ny) = field1_hat.dim();
        let mut correlation_hat = Array2::zeros((nx, ny));

        for i in 0..nx {
            for j in 0..ny {
                let f1 = field1_hat[[i, j]];
                let f2 = field2_hat[[i, j]].conj();
                let norm = (f1.norm() * f2.norm()).max(1e-12);
                correlation_hat[[i, j]] = (f1 * f2) / norm;
            }
        }

        Self::fft_2d_backward(&correlation_hat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_fft_round_trip_2d() {
        let nx = 8;
        let ny = 8;
        let mut field = Array2::zeros((nx, ny));

        // Create a simple test pattern
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j]] = (i as f64).sin() * (j as f64).cos();
            }
        }

        // Forward and backward FFT should recover original field
        let field_hat = FFTOperations::fft_2d_forward(&field).unwrap();
        let recovered = FFTOperations::fft_2d_backward(&field_hat).unwrap();

        // Check that we recover the original field (within numerical precision)
        for i in 0..nx {
            for j in 0..ny {
                assert!((field[[i, j]] - recovered[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_energy_spectrum_2d() {
        let nx = 16;
        let ny = 16;
        let mut field = Array2::zeros((nx, ny));

        // Create a field with known frequency content
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * 2.0 * std::f64::consts::PI / nx as f64;
                let y = j as f64 * 2.0 * std::f64::consts::PI / ny as f64;
                field[[i, j]] = x.sin() + (2.0 * y).cos();
            }
        }

        let spectrum = FFTOperations::compute_energy_spectrum_2d(&field).unwrap();

        // Spectrum should have finite length
        assert!(!spectrum.is_empty());

        // Energy should be non-negative
        for energy in &spectrum {
            assert!(*energy >= 0.0);
        }
    }

    #[test]
    fn test_low_pass_filter_2d() {
        let nx = 16;
        let ny = 16;
        let mut field = Array2::zeros((nx, ny));

        // Create field with high and low frequency components
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * 2.0 * std::f64::consts::PI / nx as f64;
                let y = j as f64 * 2.0 * std::f64::consts::PI / ny as f64;
                field[[i, j]] = x.sin() + (8.0 * x).sin() + y.cos() + (8.0 * y).cos();
            }
        }

        let filtered = FFTOperations::low_pass_filter_2d(&field, 2.0).unwrap();

        // Filtered field should have same dimensions
        assert_eq!(filtered.dim(), field.dim());

        // High frequency components should be reduced
        let original_max = field.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let filtered_max = filtered.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!(filtered_max <= original_max);
    }
}
