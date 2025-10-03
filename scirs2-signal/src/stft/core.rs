//! Core STFT implementation
//!
//! This module contains the main ShortTimeFft structure and its core methods
//! for computing forward and inverse STFT transforms.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use crate::window;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::f64::consts::PI;

/// A parametrized discrete Short-time Fourier transform (STFT)
/// and its inverse (ISTFT).
///
/// The STFT calculates sequential FFTs by sliding a window over an input signal
/// by hop increments. It can be used to quantify the change of the spectrum over time.
#[derive(Debug, Clone)]
pub struct ShortTimeFft {
    /// Window function
    pub win: Array1<f64>,

    /// Hop size (samples between consecutive frames)
    pub hop: usize,

    /// Sampling frequency
    pub fs: f64,

    /// FFT mode
    pub fft_mode: FftMode,

    /// FFT length
    pub mfft: usize,

    /// Dual window for inverse STFT
    pub dual_win: Option<Array1<f64>>,

    /// Scaling mode
    pub scaling: ScalingMode,

    /// Phase shift
    pub phase_shift: Option<isize>,

    /// Window length
    pub m_num: usize,

    /// Center index of window
    pub m_num_mid: usize,
}

impl ShortTimeFft {
    /// Create a new ShortTimeFft instance
    pub fn new(
        win: &[f64],
        hop: usize,
        fs: f64,
        config: Option<StftConfig>,
    ) -> SignalResult<Self> {
        // Use default config if none provided
        let config = config.unwrap_or_default();

        // Validate input parameters
        if win.is_empty() {
            return Err(SignalError::ValueError(
                "Window cannot be empty".to_string(),
            ));
        }

        if hop == 0 {
            return Err(SignalError::ValueError(
                "Hop size must be greater than 0".to_string(),
            ));
        }

        if fs <= 0.0 {
            return Err(SignalError::ValueError(
                "Sampling frequency must be positive".to_string(),
            ));
        }

        // Parse FFT mode
        let fft_mode_val = match config.fft_mode {
            Some(ref mode) => mode.parse::<FftMode>()?,
            None => FftMode::default(),
        };

        // Default mfft to window length
        let mfft_val = config.mfft.unwrap_or(win.len());

        // Validate mfft
        if mfft_val < win.len() {
            return Err(SignalError::ValueError(
                "FFT length must be at least as large as window length".to_string(),
            ));
        }

        // Parse scaling mode
        let scaling_val = match config.scale_to {
            Some(ref mode) => mode.parse::<ScalingMode>()?,
            None => ScalingMode::default(),
        };

        // Convert window to Array1
        let win_array = Array1::from_vec(win.to_vec());

        // Convert dual window to Array1 if provided
        let dual_win_array = if let Some(ref dw) = config.dual_win {
            if dw.len() != win.len() {
                return Err(SignalError::ValueError(
                    "Dual window must have the same length as window".to_string(),
                ));
            }
            Some(Array1::from_vec(dw.clone()))
        } else {
            None
        };

        // Window length and midpoint
        let m_num = win.len();
        let m_num_mid = m_num / 2;

        Ok(ShortTimeFft {
            win: win_array,
            hop,
            fs,
            fft_mode: fft_mode_val,
            mfft: mfft_val,
            dual_win: dual_win_array,
            scaling: scaling_val,
            phase_shift: config.phase_shift,
            m_num,
            m_num_mid,
        })
    }

    /// Create a ShortTimeFft instance from a named window
    pub fn from_window(
        window_type: &str,
        fs: f64,
        nperseg: usize,
        noverlap: usize,
        config: Option<StftConfig>,
    ) -> SignalResult<Self> {
        // Validate noverlap
        if noverlap >= nperseg {
            return Err(SignalError::ValueError(
                "noverlap must be less than nperseg".to_string(),
            ));
        }

        // Create window
        let win = window::get_window(window_type, nperseg, false)?;

        // Calculate hop size
        let hop = nperseg - noverlap;

        // Create ShortTimeFft
        Self::new(&win, hop, fs, config)
    }

    /// Create a ShortTimeFft instance where the window equals its dual
    pub fn from_win_equals_dual(
        win: &[f64],
        hop: usize,
        fs: f64,
        config: Option<StftConfig>,
    ) -> SignalResult<Self> {
        let mut config = config.unwrap_or_default();
        config.dual_win = Some(win.to_vec());
        Self::new(win, hop, fs, Some(config))
    }

    /// Create a ShortTimeFft instance from a dual window
    pub fn from_dual(
        dual_win: &[f64],
        hop: usize,
        fs: f64,
        config: Option<StftConfig>,
    ) -> SignalResult<Self> {
        // Create analysis window (placeholder - in practice would compute optimal window)
        let win = dual_win.to_vec();
        let mut config = config.unwrap_or_default();
        config.dual_win = Some(dual_win.to_vec());
        Self::new(&win, hop, fs, Some(config))
    }

    /// Check if the STFT setup is invertible
    pub fn invertible(&self) -> bool {
        if self.dual_win.is_none() {
            return false;
        }

        // Check if windowing satisfies COLA (Constant OverLap Add) condition
        let overlap = self.m_num - self.hop;
        overlap >= self.m_num / 2 // Simplified invertibility check
    }

    /// Calculate dual canonical window for inverse STFT
    pub fn calc_dual_canonical_window(&self) -> SignalResult<Array1<f64>> {
        if let Some(ref dual_win) = self.dual_win {
            Ok(dual_win.clone())
        } else {
            // Calculate dual window using pseudo-inverse approach
            let mut dual = Array1::zeros(self.m_num);
            let sum_win_sq = self.win.mapv(|w| w * w).sum();

            if sum_win_sq > 1e-12 {
                for i in 0..self.m_num {
                    dual[i] = self.win[i] / sum_win_sq;
                }
            }
            Ok(dual)
        }
    }

    /// Get time resolution
    pub fn t(&self) -> f64 {
        self.hop as f64 / self.fs
    }

    /// Get time step
    pub fn delta_t(&self) -> f64 {
        self.hop as f64 / self.fs
    }

    /// Get frequency resolution
    pub fn delta_f(&self) -> f64 {
        self.fs / self.mfft as f64
    }

    /// Get minimum time index
    pub fn p_min(&self) -> isize {
        -((self.m_num_mid) as isize)
    }

    /// Get maximum time index for given signal length
    pub fn p_max(&self, n: usize) -> isize {
        ((n + self.m_num_mid - 1) / self.hop) as isize
    }

    /// Get number of time frames for given signal length
    pub fn p_num(&self, n: usize) -> usize {
        let p_min = self.p_min();
        let p_max = self.p_max(n);
        (p_max - p_min) as usize
    }

    /// Get minimum frequency index
    pub fn k_min(&self) -> isize {
        match self.fft_mode {
            FftMode::Centered => -((self.mfft / 2) as isize),
            _ => 0,
        }
    }

    /// Get maximum frequency index
    pub fn k_max(&self, n: usize) -> isize {
        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => (self.mfft / 2) as isize,
            _ => (self.mfft - 1) as isize,
        }
    }

    /// Check if using one-sided FFT
    pub fn onesided_fft(&self) -> bool {
        matches!(self.fft_mode, FftMode::OneSided | FftMode::OneSided2X)
    }

    /// Get number of frequency points
    pub fn f_pts(&self) -> usize {
        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => self.mfft / 2 + 1,
            _ => self.mfft,
        }
    }

    /// Get frequency values
    pub fn f(&self) -> Array1<f64> {
        let f_pts = self.f_pts();
        let mut f = Array1::zeros(f_pts);

        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                for (i, f_i) in f.iter_mut().enumerate() {
                    *f_i = i as f64 * self.fs / self.mfft as f64;
                }
            }
            FftMode::TwoSided => {
                for (i, f_i) in f.iter_mut().enumerate().take(self.mfft) {
                    if i <= self.mfft / 2 {
                        *f_i = i as f64 * self.fs / self.mfft as f64;
                    } else {
                        *f_i = (i as f64 - self.mfft as f64) * self.fs / self.mfft as f64;
                    }
                }
            }
            FftMode::Centered => {
                for (i, f_i) in f.iter_mut().enumerate().take(self.mfft) {
                    *f_i = (i as f64 - self.mfft as f64 / 2.0) * self.fs / self.mfft as f64;
                }
            }
        }

        f
    }

    /// Get time vector for given signal length
    pub fn t_vec(&self, n: usize) -> Array1<f64> {
        let p_min = self.p_min();
        let p_max = self.p_max(n);
        let p_num = (p_max - p_min) as usize;
        let mut t = Array1::zeros(p_num);

        for (i, p) in (p_min..p_max).enumerate() {
            t[i] = p as f64 * self.hop as f64 / self.fs;
        }

        t
    }

    /// Get extent of STFT for plotting
    pub fn extent(
        &self,
        n: usize,
        axes_seq: Option<&str>,
        center_bins: Option<bool>,
    ) -> (f64, f64, f64, f64) {
        let center_bins = center_bins.unwrap_or(false);
        let t_vec = self.t_vec(n);
        let f_vec = self.f();

        let t_start = t_vec[0];
        let t_end = t_vec[t_vec.len() - 1];
        let f_start = f_vec[0];
        let f_end = f_vec[f_vec.len() - 1];

        if center_bins {
            let dt = if t_vec.len() > 1 { (t_end - t_start) / (t_vec.len() - 1) as f64 } else { 0.0 };
            let df = if f_vec.len() > 1 { (f_end - f_start) / (f_vec.len() - 1) as f64 } else { 0.0 };

            (t_start - dt/2.0, t_end + dt/2.0, f_start - df/2.0, f_end + df/2.0)
        } else {
            (t_start, t_end, f_start, f_end)
        }
    }
}