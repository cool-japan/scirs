//! STFT algorithms implementation
//!
//! This module contains the forward and inverse STFT algorithms,
//! including FFT operations and frame processing.

use super::core::ShortTimeFft;
use super::types::*;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

impl ShortTimeFft {
    /// Perform the Short-Time Fourier Transform
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    ///
    /// # Returns
    ///
    /// * Complex-valued STFT matrix
    pub fn stft<T>(&self, x: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug,
    {
        if x.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        // Convert input to f64
        let x_f64: Vec<f64> = x
            .iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?;

        // Create padded signal with zeros
        let k_min = self.k_min();
        let k_max = self.k_max(x.len());
        let signal_len = (k_max - k_min) as usize;

        let mut padded_signal = Array1::zeros(signal_len);

        // Copy input signal to padded signal
        for (i, &val) in x_f64.iter().enumerate() {
            let idx = (i as isize - k_min) as usize;
            if idx < signal_len {
                padded_signal[idx] = val;
            }
        }

        // Calculate number of time frames
        let p_min = self.p_min();
        let p_max = self.p_max(x.len());
        let p_num = (p_max - p_min) as usize;

        // Initialize STFT matrix
        let f_pts = self.f_pts();
        let mut stft_matrix = Array2::zeros((f_pts, p_num));

        // Apply window and FFT for each frame
        for (p_idx, p) in (p_min..p_max).enumerate() {
            // Get start index for this frame
            let start = (p * self.hop as isize - k_min) as usize;

            // Extract frame
            let mut frame = Array1::zeros(self.mfft);
            for (i, win_i) in self.win.iter().enumerate().take(self.m_num) {
                if start + i < signal_len {
                    frame[i] = padded_signal[start + i] * win_i;
                }
            }

            // Apply FFT
            let frame_spectrum = self.fft(&frame)?;

            // Add to STFT matrix
            for (q, &val) in frame_spectrum.iter().enumerate() {
                if q < f_pts {
                    stft_matrix[[q, p_idx]] = val;
                }
            }
        }

        // Apply scaling if needed
        if self.scaling != ScalingMode::None {
            let scale_factor = match self.scaling {
                ScalingMode::Magnitude => 1.0 / self.win.mapv(|x| x * x).sum().sqrt(),
                ScalingMode::Psd => 1.0 / self.win.mapv(|x| x * x).sum(),
                _ => 1.0,
            };

            stft_matrix.mapv_inplace(|x| x * scale_factor);
        }

        Ok(stft_matrix)
    }

    /// Perform the inverse Short-Time Fourier Transform
    ///
    /// # Arguments
    ///
    /// * `X` - STFT matrix
    /// * `k0` - Start sample (optional)
    /// * `k1` - End sample (optional)
    ///
    /// # Returns
    ///
    /// * Reconstructed signal
    pub fn istft(
        &self,
        x: &Array2<Complex64>,
        k0: Option<usize>,
        k1: Option<usize>,
    ) -> SignalResult<Vec<f64>> {
        if x.is_empty() {
            return Err(SignalError::ValueError("STFT matrix is empty".to_string()));
        }

        // Ensure the STFT is invertible
        let dual_window = self.calc_dual_canonical_window()?;

        // Get dimensions
        let f_pts = x.shape()[0];
        let p_num = x.shape()[1];

        if f_pts != self.f_pts() {
            return Err(SignalError::ValueError(format!(
                "STFT matrix has {} frequency points, expected {}",
                f_pts,
                self.f_pts()
            )));
        }

        // Calculate signal boundaries
        let p_min = self.p_min();
        let k_min = self.k_min();
        let k_max = if let Some(k1_val) = k1 {
            k1_val as isize
        } else {
            k_min + (p_min + p_num as isize) * self.hop as isize + self.m_num as isize
        };

        let k0_val = k0.map(|k| k as isize).unwrap_or(k_min);

        if k0_val >= k_max {
            return Err(SignalError::ValueError(format!(
                "k0 ({}) must be less than k_max ({})",
                k0_val, k_max
            )));
        }

        // Initialize output signal
        let signal_len = (k_max - k0_val) as usize;
        let mut output = Array1::zeros(signal_len);
        let mut weight = Array1::<f64>::zeros(signal_len);

        // Process each frame
        for (p_idx, p) in (p_min..(p_min + p_num as isize)).enumerate() {
            // Inverse FFT for this frame
            let frame_spectrum = self.get_stft_frame(x, p_idx)?;
            let frame = self.ifft(&frame_spectrum)?;

            // Add to output with dual window
            let start = p * self.hop as isize - k0_val;

            for i in 0..self.m_num {
                let idx = start + i as isize;
                if idx >= 0 && (idx as usize) < signal_len {
                    let idx_u = idx as usize;
                    output[idx_u] += frame[i].re * dual_window[i];
                    weight[idx_u] += dual_window[i] * dual_window[i];
                }
            }
        }

        // Normalize by weights
        for i in 0..signal_len {
            if weight[i] > 1e-12 {
                output[i] /= weight[i];
            }
        }

        Ok(output.to_vec())
    }

    /// Compute spectrogram (magnitude squared of STFT)
    pub fn spectrogram<T>(&self, x: &[T]) -> SignalResult<Array2<f64>>
    where
        T: Float + NumCast + Debug,
    {
        let stft_result = self.stft(x)?;
        Ok(stft_result.mapv(|c| c.norm_sqr()))
    }

    /// Get border points for visualization
    pub fn border_points(&self, n: usize) -> ((isize, isize), (isize, isize)) {
        let t_lo = self.p_min();
        let t_hi = self.p_max(n);
        let f_lo = self.k_min();
        let f_hi = self.k_max(n);
        ((t_lo, t_hi), (f_lo, f_hi))
    }

    // Helper methods for FFT operations

    /// Apply FFT to a frame
    pub(super) fn fft(&self, frame: &Array1<f64>) -> SignalResult<Array1<Complex64>> {
        let n = frame.len();
        let mut result = Array1::<Complex64>::zeros(n);

        // Convert to complex
        for (i, &val) in frame.iter().enumerate() {
            result[i] = Complex64::new(val, 0.0);
        }

        // Apply simple DFT (placeholder for proper FFT)
        let mut fft_result = Array1::<Complex64>::zeros(n);
        for k in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for t in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                sum += result[t] * Complex64::new(angle.cos(), angle.sin());
            }
            fft_result[k] = sum;
        }

        // Handle different FFT modes
        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                let mut onesided = Array1::<Complex64>::zeros(n / 2 + 1);
                for i in 0..=n / 2 {
                    onesided[i] = fft_result[i];
                    if self.fft_mode == FftMode::OneSided2X && i > 0 && i < n / 2 {
                        onesided[i] *= 2.0;
                    }
                }
                Ok(onesided)
            }
            FftMode::TwoSided => Ok(fft_result),
            FftMode::Centered => {
                // Fftshift operation
                let mut centered = Array1::<Complex64>::zeros(n);
                let mid = n / 2;
                for i in 0..n {
                    let src_idx = (i + mid) % n;
                    centered[i] = fft_result[src_idx];
                }
                Ok(centered)
            }
        }
    }

    /// Apply inverse FFT to a frame spectrum
    pub(super) fn ifft(&self, spectrum: &Array1<Complex64>) -> SignalResult<Array1<Complex64>> {
        let f_pts = spectrum.len();
        let n = match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => (f_pts - 1) * 2,
            _ => f_pts,
        };

        let mut full_spectrum = Array1::<Complex64>::zeros(n);

        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                // Reconstruct two-sided spectrum
                for i in 0..f_pts {
                    full_spectrum[i] = spectrum[i];
                }

                // Mirror negative frequencies (except DC and Nyquist)
                for i in 1..f_pts - 1 {
                    full_spectrum[n - i] = spectrum[i].conj();
                }
            }
            FftMode::TwoSided => {
                full_spectrum = spectrum.clone();
            }
            FftMode::Centered => {
                // Inverse fftshift
                let mid = n / 2;
                for i in 0..n {
                    let dst_idx = (i + mid) % n;
                    full_spectrum[dst_idx] = spectrum[i];
                }
            }
        }

        // Apply inverse DFT (placeholder for proper IFFT)
        let mut result = Array1::<Complex64>::zeros(n);
        for t in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                let angle = 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                sum += full_spectrum[k] * Complex64::new(angle.cos(), angle.sin());
            }
            result[t] = sum / n as f64;
        }

        Ok(result)
    }

    /// Get a single frame from STFT matrix
    pub(super) fn get_stft_frame(
        &self,
        stft_matrix: &Array2<Complex64>,
        frame_idx: usize,
    ) -> SignalResult<Array1<Complex64>> {
        if frame_idx >= stft_matrix.shape()[1] {
            return Err(SignalError::ValueError(format!(
                "Frame index {} out of bounds for matrix with {} frames",
                frame_idx,
                stft_matrix.shape()[1]
            )));
        }

        let f_pts = stft_matrix.shape()[0];
        let mut frame = Array1::<Complex64>::zeros(f_pts);

        for i in 0..f_pts {
            frame[i] = stft_matrix[[i, frame_idx]];
        }

        Ok(frame)
    }
}