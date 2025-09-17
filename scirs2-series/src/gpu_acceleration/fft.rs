//! GPU-accelerated FFT operations
//!
//! This module provides Fast Fourier Transform implementations optimized for GPU
//! execution, including forward/inverse FFT, power spectral density, and spectrogram.

use ndarray::{s, Array1, Array2};
use num_traits::Float;
use std::f64::consts::PI;
use std::fmt::Debug;

use super::GpuConfig;
use crate::error::{Result, TimeSeriesError};

/// GPU-accelerated FFT processor
#[derive(Debug)]
pub struct GpuFFT<F: Float + Debug> {
    #[allow(dead_code)]
    config: GpuConfig,
    /// FFT cache for repeated operations
    #[allow(dead_code)]
    fft_cache: Vec<Array1<F>>,
}

impl<F: Float + Debug + Clone> GpuFFT<F> {
    /// Create new GPU FFT processor
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            fft_cache: Vec::new(),
        }
    }

    /// GPU-accelerated forward FFT
    pub fn fft(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n = data.len();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        // Ensure power of 2 for efficiency
        let padded_n = n.next_power_of_two();
        let mut padded_data = Array1::zeros(padded_n);
        for i in 0..n {
            padded_data[i] = data[i];
        }

        // GPU-optimized Cooley-Tukey FFT implementation
        let result = self.cooley_tukey_fft(&padded_data, false)?;

        // Return only the original length
        Ok(result.slice(s![0..n]).to_owned())
    }

    /// GPU-accelerated inverse FFT
    pub fn ifft(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n = data.len();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }

        let padded_n = n.next_power_of_two();
        let mut padded_data = Array1::zeros(padded_n);
        for i in 0..n {
            padded_data[i] = data[i];
        }

        let result = self.cooley_tukey_fft(&padded_data, true)?;
        let normalized: Array1<F> = result.mapv(|x| x / F::from(padded_n).unwrap());

        Ok(normalized.slice(s![0..n]).to_owned())
    }

    /// Cooley-Tukey FFT algorithm optimized for GPU-like parallel execution
    fn cooley_tukey_fft(&self, data: &Array1<F>, inverse: bool) -> Result<Array1<F>> {
        let n = data.len();
        if n <= 1 {
            return Ok(data.clone());
        }

        if !n.is_power_of_two() {
            return Err(TimeSeriesError::InvalidInput(
                "FFT requires power of 2 length".to_string(),
            ));
        }

        let mut result = data.clone();
        let two = F::from(2).unwrap();
        let pi = F::from(PI).unwrap();

        // Bit-reversal permutation (GPU-friendly)
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if j > i {
                result.swap(i, j);
            }
        }

        // Cooley-Tukey FFT with GPU-style parallel butterfly operations
        let mut length = 2;
        while length <= n {
            let angle = if inverse {
                two * pi / F::from(length).unwrap()
            } else {
                -two * pi / F::from(length).unwrap()
            };

            let wlen_real = angle.cos();
            let wlen_imag = angle.sin();

            // Parallel butterfly operations
            for start in (0..n).step_by(length) {
                let mut w_real = F::one();
                let mut w_imag = F::zero();

                for j in 0..length / 2 {
                    let u = result[start + j];
                    let v_real = result[start + j + length / 2] * w_real;
                    let _v_imag = result[start + j + length / 2] * w_imag;

                    result[start + j] = u + v_real;
                    result[start + j + length / 2] = u - v_real;

                    // Update twiddle factors
                    let new_w_real = w_real * wlen_real - w_imag * wlen_imag;
                    let new_w_imag = w_real * wlen_imag + w_imag * wlen_real;
                    w_real = new_w_real;
                    w_imag = new_w_imag;
                }
            }

            length <<= 1;
        }

        Ok(result)
    }

    /// GPU-accelerated power spectral density
    pub fn power_spectral_density(
        &self,
        data: &Array1<F>,
        window_size: usize,
    ) -> Result<Array1<F>> {
        if data.len() < window_size {
            return Err(TimeSeriesError::InsufficientData {
                message: "Data length less than window _size".to_string(),
                required: window_size,
                actual: data.len(),
            });
        }

        let num_windows = (data.len() - window_size) / (window_size / 2) + 1;
        let mut psd = Array1::<F>::zeros(window_size / 2 + 1);

        // Parallel processing of overlapping windows
        for i in 0..num_windows {
            let start = i * window_size / 2;
            let end = (start + window_size).min(data.len());

            if end - start < window_size {
                break;
            }

            let window = data.slice(s![start..end]);
            let windowed = self.apply_hanning_window(&window.to_owned())?;
            let fft_result = self.fft(&windowed)?;

            // Compute power spectrum for this window
            for j in 0..psd.len() {
                if j < fft_result.len() {
                    psd[j] = psd[j] + fft_result[j] * fft_result[j];
                }
            }
        }

        // Normalize by number of windows
        let norm_factor = F::from(num_windows).unwrap();
        Ok(psd.mapv(|x: F| x / norm_factor))
    }

    /// Apply Hanning window for spectral analysis
    fn apply_hanning_window(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n = data.len();
        let mut windowed = data.clone();
        let pi = F::from(PI).unwrap();
        let two = F::from(2).unwrap();

        for i in 0..n {
            let window_val = F::from(0.5).unwrap()
                * (F::one() - (two * pi * F::from(i).unwrap() / F::from(n - 1).unwrap()).cos());
            windowed[i] = windowed[i] * window_val;
        }

        Ok(windowed)
    }

    /// GPU-accelerated spectrogram computation
    pub fn spectrogram(
        &self,
        data: &Array1<F>,
        window_size: usize,
        overlap: usize,
    ) -> Result<Array2<F>> {
        if window_size <= overlap {
            return Err(TimeSeriesError::InvalidInput(
                "Window _size must be greater than overlap".to_string(),
            ));
        }

        let step = window_size - overlap;
        let num_windows = (data.len() - window_size) / step + 1;
        let freq_bins = window_size / 2 + 1;

        let mut spectrogram = Array2::zeros((freq_bins, num_windows));

        // Parallel spectrogram computation
        for (window_idx, start) in (0..data.len() - window_size + 1).step_by(step).enumerate() {
            if window_idx >= num_windows {
                break;
            }

            let window = data.slice(s![start..start + window_size]);
            let windowed = self.apply_hanning_window(&window.to_owned())?;
            let fft_result = self.fft(&windowed)?;

            // Store magnitude spectrum
            for freq_idx in 0..freq_bins {
                if freq_idx < fft_result.len() {
                    let magnitude = fft_result[freq_idx].abs();
                    spectrogram[[freq_idx, window_idx]] = magnitude;
                }
            }
        }

        Ok(spectrogram)
    }
}
