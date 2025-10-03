//! Chirp Z-Transform (CZT) implementation
//!
//! This module provides implementation of the Chirp Z-Transform,
//! which enables evaluation of the Z-transform along arbitrary
//! contours in the complex plane, including non-uniform frequency spacing.

use crate::{next_fast_len, FFTError, FFTResult};
use scirs2_core::ndarray::{
    s, Array, Array1, ArrayBase, ArrayD, Axis, Data, Dimension, RemoveAxis, Zip,
};
use scirs2_core::numeric::Complex;
use std::f64::consts::PI;

/// Compute points at which the chirp z-transform samples
///
/// Returns points on the z-plane where CZT evaluates the z-transform.
/// The points follow a logarithmic spiral defined by `a` and `w`.
///
/// # Parameters
/// - `m`: Number of output points
/// - `a`: Starting point on the complex plane (default: 1+0j)
/// - `w`: Ratio between consecutive points (default: exp(-2j*pi/m))
///
/// # Returns
/// Array of complex points where z-transform is evaluated
#[allow(dead_code)]
pub fn czt_points(
    m: usize,
    a: Option<Complex<f64>>,
    w: Option<Complex<f64>>,
) -> Array1<Complex<f64>> {
    let a = a.unwrap_or(Complex::new(1.0, 0.0));
    let k = Array1::linspace(0.0, (m - 1) as f64, m);

    if let Some(w) = w {
        // Use specified w value
        k.mapv(|ki| a * w.powf(-ki))
    } else {
        // Default to FFT-like equally spaced points on unit circle
        k.mapv(|ki| a * (Complex::new(0.0, 2.0 * PI * ki / m as f64)).exp())
    }
}

/// Chirp Z-Transform implementation
///
/// This structure pre-computes constant values for efficient CZT computation
#[derive(Clone)]
pub struct CZT {
    n: usize,
    m: usize,
    w: Option<Complex<f64>>,
    a: Complex<f64>,
    nfft: usize,
    awk2: Array1<Complex<f64>>,
    fwk2: Array1<Complex<f64>>,
    wk2: Array1<Complex<f64>>,
}

impl CZT {
    /// Create a new CZT transform
    ///
    /// # Parameters
    /// - `n`: Size of input signal
    /// - `m`: Number of output points (default: n)
    /// - `w`: Ratio between points (default: exp(-2j*pi/m))
    /// - `a`: Starting point in complex plane (default: 1+0j)
    pub fn new(
        n: usize,
        m: Option<usize>,
        w: Option<Complex<f64>>,
        a: Option<Complex<f64>>,
    ) -> FFTResult<Self> {
        if n < 1 {
            return Err(FFTError::ValueError("n must be positive".to_string()));
        }

        let m = m.unwrap_or(n);
        if m < 1 {
            return Err(FFTError::ValueError("m must be positive".to_string()));
        }

        let a = a.unwrap_or(Complex::new(1.0, 0.0));
        let max_size = n.max(m);
        let k = Array1::linspace(0.0, (max_size - 1) as f64, max_size);

        let (w, wk2) = if let Some(w) = w {
            // User-specified w
            let wk2 = k.mapv(|ki| w.powf(ki * ki / 2.0));
            (Some(w), wk2)
        } else {
            // Default to FFT-like
            let w = (-2.0 * PI * Complex::<f64>::i() / m as f64).exp();
            let wk2 = k.mapv(|ki| {
                let ki_i64 = ki as i64;
                let phase = -(PI * ((ki_i64 * ki_i64) % (2 * m as i64)) as f64) / m as f64;
                Complex::from_polar(1.0, phase)
            });
            (Some(w), wk2)
        };

        // Compute length for FFT
        let nfft = next_fast_len(n + m - 1, false);

        // Pre-compute A(k) * w_k^2 for the first n values
        let awk2: Array1<Complex<f64>> = (0..n).map(|k| a.powf(-(k as f64)) * wk2[k]).collect();

        // Pre-compute FFT of the reciprocal chirp
        let mut chirp_vec = vec![Complex::new(0.0, 0.0); nfft];

        // Fill with 1/wk2 values in specific order
        for i in 1..n {
            chirp_vec[n - 1 - i] = Complex::new(1.0, 0.0) / wk2[i];
        }
        for i in 0..m {
            chirp_vec[n - 1 + i] = Complex::new(1.0, 0.0) / wk2[i];
        }

        let chirp_array = Array1::from_vec(chirp_vec);
        let fwk2_vec = crate::fft::fft(&chirp_array.to_vec(), None)?;
        let fwk2 = Array1::from_vec(fwk2_vec);

        Ok(CZT {
            n,
            m,
            w,
            a,
            nfft,
            awk2,
            fwk2,
            wk2: wk2.slice(s![..m]).to_owned(),
        })
    }

    /// Compute the points where this CZT evaluates the z-transform
    pub fn points(&self) -> Array1<Complex<f64>> {
        czt_points(self.m, Some(self.a), self.w)
    }

    /// Apply the chirp z-transform to a signal
    ///
    /// # Parameters
    /// - `x`: Input signal
    /// - `axis`: Axis along which to compute CZT (default: -1)
    pub fn transform<S, D>(
        &self,
        x: &ArrayBase<S, D>,
        axis: Option<i32>,
    ) -> FFTResult<ArrayD<Complex<f64>>>
    where
        S: Data<Elem = Complex<f64>>,
        D: Dimension + RemoveAxis,
    {
        let ndim = x.ndim();
        let axis = if let Some(ax) = axis {
            if ax < 0 {
                let ax_pos = (ndim as i32 + ax) as usize;
                if ax_pos >= ndim {
                    return Err(FFTError::ValueError("Invalid axis".to_string()));
                }
                ax_pos
            } else {
                ax as usize
            }
        } else {
            ndim - 1
        };

        let axis_len = x.shape()[axis];
        if axis_len != self.n {
            return Err(FFTError::ValueError(format!(
                "Input size ({}) doesn't match CZT size ({})",
                axis_len, self.n
            )));
        }

        // Create output shape - same as input but with m points along specified axis
        let mut outputshape = x.shape().to_vec();
        outputshape[axis] = self.m;
        let mut result = Array::<Complex<f64>, _>::zeros(outputshape).into_dyn();

        // Apply CZT along the specified axis
        // For 1D array, directly apply the transform
        if x.ndim() == 1 {
            let x_1d: Array1<Complex<f64>> = x
                .to_owned()
                .into_shape_with_order(x.len())
                .map_err(|e| {
                    FFTError::ComputationError(format!("Failed to reshape input array to 1D: {e}"))
                })?
                .into_dimensionality()
                .map_err(|e| {
                    FFTError::ComputationError(format!(
                        "Failed to convert array dimensionality: {e}"
                    ))
                })?;
            let y = self.transform_1d(&x_1d)?;
            return Ok(y.into_dyn());
        }

        // For higher dimensions, iterate over axis
        for (i, x_slice) in x.axis_iter(Axis(axis)).enumerate() {
            // Convert slice to Array1
            let x_1d: Array1<Complex<f64>> = x_slice
                .to_owned()
                .into_shape_with_order(x_slice.len())
                .map_err(|e| {
                    FFTError::ComputationError(format!("Failed to reshape slice to 1D array: {e}"))
                })?;
            let y = self.transform_1d(&x_1d)?;

            // Dynamic slicing based on the number of dimensions
            match result.ndim() {
                2 => {
                    if axis == 0 {
                        let mut result_slice = result.slice_mut(s![i, ..]);
                        result_slice.assign(&y);
                    } else {
                        let mut result_slice = result.slice_mut(s![.., i]);
                        result_slice.assign(&y);
                    }
                }
                _ => {
                    // For higher dimensions, we need more complex handling
                    return Err(FFTError::ValueError(
                        "CZT currently only supports 1D and 2D arrays".to_string(),
                    ));
                }
            }
        }

        Ok(result)
    }

    /// Transform a 1D signal
    fn transform_1d(&self, x: &Array1<Complex<f64>>) -> FFTResult<Array1<Complex<f64>>> {
        if x.len() != self.n {
            return Err(FFTError::ValueError(format!(
                "Input size ({}) doesn't match CZT size ({})",
                x.len(),
                self.n
            )));
        }

        // Multiply input by A(k) * w_k^2
        let x_weighted: Array1<Complex<f64>> = Zip::from(x)
            .and(&self.awk2)
            .map_collect(|&xi, &awki| xi * awki);

        // Create zero-padded array for FFT
        let mut padded = Array1::zeros(self.nfft);
        padded.slice_mut(s![..self.n]).assign(&x_weighted);

        // Forward FFT
        let x_fft_vec = crate::fft::fft(&padded.to_vec(), None)?;
        let x_fft = Array1::from_vec(x_fft_vec);

        // Multiply by pre-computed FFT of reciprocal chirp
        let product: Array1<Complex<f64>> = Zip::from(&x_fft)
            .and(&self.fwk2)
            .map_collect(|&xi, &fi| xi * fi);

        // Inverse FFT
        let y_full_vec = crate::fft::ifft(&product.to_vec(), None)?;
        let y_full = Array1::from_vec(y_full_vec);

        // Extract relevant portion and multiply by w_k^2
        let y_slice = y_full.slice(s![self.n - 1..self.n - 1 + self.m]);
        let result: Array1<Complex<f64>> = Zip::from(&y_slice)
            .and(&self.wk2)
            .map_collect(|&yi, &wki| yi * wki);

        Ok(result)
    }
}

/// Functional interface to chirp z-transform
///
/// # Parameters
/// - `x`: Input signal
/// - `m`: Number of output points (default: length of x)
/// - `w`: Ratio between points (default: exp(-2j*pi/m))
/// - `a`: Starting point in complex plane (default: 1+0j)
/// - `axis`: Axis along which to compute CZT (default: -1)
#[allow(dead_code)]
pub fn czt<S, D>(
    x: &ArrayBase<S, D>,
    m: Option<usize>,
    w: Option<Complex<f64>>,
    a: Option<Complex<f64>>,
    axis: Option<i32>,
) -> FFTResult<ArrayD<Complex<f64>>>
where
    S: Data<Elem = Complex<f64>>,
    D: Dimension + RemoveAxis,
{
    let axis_actual = if let Some(ax) = axis {
        if ax < 0 {
            (x.ndim() as i32 + ax) as usize
        } else {
            ax as usize
        }
    } else {
        x.ndim() - 1
    };

    let n = x.shape()[axis_actual];
    let transform = CZT::new(n, m, w, a)?;
    transform.transform(x, axis)
}

/// Compute a zoom FFT - partial DFT on a specified frequency range
///
/// Efficiently evaluates the DFT over a subset of frequency range.
///
/// # Parameters
/// - `x`: Input signal
/// - `m`: Number of output points
/// - `f0`: Starting normalized frequency (0 to 1)
/// - `f1`: Ending normalized frequency (0 to 1)
/// - `oversampling`: Oversampling factor for frequency resolution
#[allow(dead_code)]
pub fn zoom_fft<S, D>(
    x: &ArrayBase<S, D>,
    m: usize,
    f0: f64,
    f1: f64,
    oversampling: Option<f64>,
) -> FFTResult<ArrayD<Complex<f64>>>
where
    S: Data<Elem = Complex<f64>>,
    D: Dimension + RemoveAxis,
{
    if !(0.0..=1.0).contains(&f0) || !(0.0..=1.0).contains(&f1) {
        return Err(FFTError::ValueError(
            "Frequencies must be in range [0, 1]".to_string(),
        ));
    }

    if f0 >= f1 {
        return Err(FFTError::ValueError("f0 must be less than f1".to_string()));
    }

    let oversampling = oversampling.unwrap_or(2.0);
    if oversampling < 1.0 {
        return Err(FFTError::ValueError(
            "Oversampling must be >= 1".to_string(),
        ));
    }

    let ndim = x.ndim();
    let axis = ndim - 1;
    let n = x.shape()[axis];

    // Compute CZT parameters for zoom FFT
    let k0_float = f0 * n as f64 * oversampling;
    let k1_float = f1 * n as f64 * oversampling;
    let step = (k1_float - k0_float) / (m - 1) as f64;

    let phi = 2.0 * PI * k0_float / (n as f64 * oversampling);
    let a = Complex::from_polar(1.0, phi);

    let theta = -2.0 * PI * step / (n as f64 * oversampling);
    let w = Complex::from_polar(1.0, theta);

    czt(x, Some(m), Some(w), Some(a), Some(axis as i32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_czt_points() {
        // Test default points (FFT-like)
        let points = czt_points(4, None, None);
        assert_eq!(points.len(), 4);

        // Check that they lie on unit circle
        for p in points.iter() {
            assert_abs_diff_eq!(p.norm(), 1.0, epsilon = 1e-10);
        }

        // Test custom spiral
        let a = Complex::new(0.8, 0.0);
        let w = Complex::from_polar(0.95, 0.1);
        let points = czt_points(5, Some(a), Some(w));
        assert_eq!(points.len(), 5);
        assert!((points[0] - a).norm() < 1e-10);
    }

    #[test]
    fn test_czt_as_fft() {
        // CZT with default parameters should match FFT
        let n = 8;
        let x: Array1<Complex<f64>> = Array1::linspace(0.0, 7.0, n).mapv(|v| Complex::new(v, 0.0));

        let czt_result = czt(&x.view(), None, None, None, None)
            .expect("CZT computation should succeed for test data");

        // czt returns ArrayD, need to convert to Array1
        assert_eq!(czt_result.ndim(), 1);
        let czt_result_1d: Array1<Complex<f64>> = czt_result
            .into_dimensionality()
            .expect("CZT result should convert to 1D array");

        let fft_result_vec = crate::fft::fft(&x.to_vec(), None)
            .expect("FFT computation should succeed for test data");
        let fft_result = Array1::from_vec(fft_result_vec);

        for i in 0..n {
            assert!((czt_result_1d[i].re - fft_result[i].re).abs() < 1e-10);
            assert!((czt_result_1d[i].im - fft_result[i].im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zoom_fft() {
        // Create a simple signal with a clear frequency peak
        let n = 64;
        let t: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let x: Array1<Complex<f64>> = t.mapv(|ti| {
            let s = (2.0 * PI * 5.0 * ti).sin(); // Single frequency for simplicity
            Complex::new(s, 0.0)
        });

        // Zoom in on a wider frequency range to ensure we capture the signal
        let m = 16;
        let zoom_result =
            zoom_fft(&x.view(), m, 0.0, 0.5, None).expect("Zoom FFT should succeed for test data");

        // Basic validation - check that we got a result and it's the right size
        assert_eq!(zoom_result.ndim(), 1);
        let zoom_result_1d: Array1<Complex<f64>> = zoom_result
            .into_dimensionality()
            .expect("Zoom FFT result should convert to 1D array");
        assert_eq!(zoom_result_1d.len(), m);

        // Simple check - there should be some non-zero values in the result
        let has_nonzero = zoom_result_1d.iter().any(|&c| c.norm() > 1e-10);
        assert!(has_nonzero, "Zoom FFT should produce some non-zero values");
    }
}
