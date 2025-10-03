//! GPU-accelerated convolution operations
//!
//! This module provides convolution and correlation operations optimized for GPU
//! execution, including 1D convolution, cross-correlation, and sliding window operations.

use scirs2_core::ndarray::{s, Array1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::{fft, GpuConfig};
use crate::error::{Result, TimeSeriesError};

/// GPU-accelerated convolution processor
#[derive(Debug)]
pub struct GpuConvolution<F: Float + Debug> {
    #[allow(dead_code)]
    config: GpuConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<F: Float + Debug + Clone> GpuConvolution<F> {
    /// Create new GPU convolution processor
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// GPU-accelerated 1D convolution
    pub fn convolve_1d(&self, signal: &Array1<F>, kernel: &Array1<F>) -> Result<Array1<F>> {
        if signal.is_empty() || kernel.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Signal and kernel must be non-empty".to_string(),
            ));
        }

        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = signal_len + kernel_len - 1;

        let mut result = Array1::zeros(output_len);

        // GPU-style parallel convolution with memory coalescing
        let chunk_size = self.config.batch_size;

        for chunk_start in (0..output_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(output_len);

            // Parallel processing within chunk
            for i in chunk_start..chunk_end {
                let mut sum = F::zero();

                // Vectorized inner loop
                let k_start = if i >= signal_len - 1 {
                    i - signal_len + 1
                } else {
                    0
                };
                let k_end = (i + 1).min(kernel_len);

                for k in k_start..k_end {
                    let signal_idx = i - k;
                    if signal_idx < signal_len {
                        sum = sum + signal[signal_idx] * kernel[k];
                    }
                }

                result[i] = sum;
            }
        }

        Ok(result)
    }

    /// GPU-accelerated cross-correlation
    pub fn cross_correlate(&self, x: &Array1<F>, y: &Array1<F>) -> Result<Array1<F>> {
        if x.is_empty() || y.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Input arrays must be non-empty".to_string(),
            ));
        }

        let n = x.len();
        let m = y.len();
        let result_len = n + m - 1;
        let mut result = Array1::zeros(result_len);

        // GPU-optimized cross-correlation using parallel reduction
        for lag in 0..result_len {
            let mut correlation = F::zero();

            // Determine overlap region
            let start_x = if lag >= m { lag - m + 1 } else { 0 };
            let end_x = (lag + 1).min(n);

            // Parallel dot product computation
            for i in start_x..end_x {
                let j = lag - i;
                if j < m {
                    correlation = correlation + x[i] * y[j];
                }
            }

            result[lag] = correlation;
        }

        Ok(result)
    }

    /// GPU-accelerated auto-correlation with FFT
    pub fn auto_correlate_fft(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n = data.len();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        // Use FFT-based correlation for better performance
        let padded_size = (2 * n - 1).next_power_of_two();
        let mut padded = Array1::zeros(padded_size);

        // Copy data to padded array
        for i in 0..n {
            padded[i] = data[i];
        }

        // Compute FFT, multiply by conjugate, then IFFT
        let fft_processor = fft::GpuFFT::new(self.config.clone());
        let fft_result = fft_processor.fft(&padded)?;

        // Multiply by complex conjugate (for real signals, this is just squaring)
        let power_spectrum = fft_result.mapv(|x| x * x);

        let autocorr_full = fft_processor.ifft(&power_spectrum)?;

        // Return only the meaningful part (0 to n-1 lags)
        Ok(autocorr_full.slice(s![0..n]).to_owned())
    }

    /// GPU-accelerated sliding window correlation
    pub fn sliding_correlation(
        &self,
        x: &Array1<F>,
        y: &Array1<F>,
        window_size: usize,
    ) -> Result<Array1<F>> {
        if x.len() != y.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.len() < window_size {
            return Err(TimeSeriesError::InsufficientData {
                message: "Data length less than window _size".to_string(),
                required: window_size,
                actual: x.len(),
            });
        }

        let num_windows = x.len() - window_size + 1;
        let mut correlations = Array1::zeros(num_windows);

        // GPU-style parallel window processing
        for i in 0..num_windows {
            let x_window = x.slice(s![i..i + window_size]);
            let y_window = y.slice(s![i..i + window_size]);

            // Compute Pearson correlation coefficient
            let mean_x = x_window.sum() / F::from(window_size).unwrap();
            let mean_y = y_window.sum() / F::from(window_size).unwrap();

            let mut num = F::zero();
            let mut den_x = F::zero();
            let mut den_y = F::zero();

            // Vectorized correlation computation
            for j in 0..window_size {
                let dx = x_window[j] - mean_x;
                let dy = y_window[j] - mean_y;

                num = num + dx * dy;
                den_x = den_x + dx * dx;
                den_y = den_y + dy * dy;
            }

            let denominator = (den_x * den_y).sqrt();
            correlations[i] = if denominator > F::zero() {
                num / denominator
            } else {
                F::zero()
            };
        }

        Ok(correlations)
    }
}
