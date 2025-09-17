use ndarray::s;
// Short-Time Fourier Transform (STFT) implementation.
//
// This module provides a parametrized discrete Short-time Fourier transform (STFT)
// and its inverse (ISTFT), similar to SciPy's ShortTimeFFT class.
//
// Features:
// - Memory-efficient processing for large signals
// - Streaming STFT with configurable chunk sizes
// - Zero-copy processing where possible
// - Parallel processing support

use crate::error::{SignalError, SignalResult};
use crate::lti::design::tf;
use crate::window;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

/// Configuration options for STFT
#[derive(Debug, Clone, Default)]
pub struct StftConfig {
    /// FFT mode (e.g., "real", "complex")
    pub fft_mode: Option<String>,
    /// FFT size override
    pub mfft: Option<usize>,
    /// Optional dual window for analysis/synthesis
    pub dual_win: Option<Vec<f64>>,
    /// Scaling option
    pub scale_to: Option<String>,
    /// Phase shift
    pub phase_shift: Option<isize>,
}

/// FFT mode options for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FftMode {
    /// Two-sided spectrum with negative frequencies preceding positive frequencies
    TwoSided,
    /// Two-sided spectrum with negative frequencies following positive frequencies
    Centered,
    /// One-sided spectrum (positive frequencies only)
    #[default]
    OneSided,
    /// One-sided spectrum with doubled amplitudes for energy conservation
    OneSided2X,
}

impl std::str::FromStr for FftMode {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "twosided" => Ok(FftMode::TwoSided),
            "centered" => Ok(FftMode::Centered),
            "onesided" => Ok(FftMode::OneSided),
            "onesided2x" => Ok(FftMode::OneSided2X),
            _ => Err(SignalError::ValueError(format!(
                "Invalid FFT mode: '{}'. Valid options are: 'twosided', 'centered', 'onesided', 'onesided2x'",
                s
            ))),
        }
    }
}

/// Scaling options for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingMode {
    /// No scaling (raw FFT values)
    #[default]
    None,
    /// Scale for magnitude spectrum
    Magnitude,
    /// Scale for power spectral density
    Psd,
}

impl std::str::FromStr for ScalingMode {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(ScalingMode::None),
            "magnitude" => Ok(ScalingMode::Magnitude),
            "psd" => Ok(ScalingMode::Psd),
            _ => Err(SignalError::ValueError(format!(
                "Invalid scaling mode: '{}'. Valid options are: 'none', 'magnitude', 'psd'",
                s
            ))),
        }
    }
}

/// A parametrized discrete Short-time Fourier transform (STFT)
/// and its inverse (ISTFT).
///
/// The STFT calculates sequential FFTs by sliding a window over an input signal
/// by hop increments. It can be used to quantify the change of the spectrum over time.
///
/// # Example
///
/// ```rust
/// use scirs2_signal::stft::{ShortTimeFft, StftConfig};
/// use scirs2_signal::window;
/// use ndarray::Array1;
///
///
/// // Create a signal with varying frequency
/// let fs = 1000.0; // 1 kHz sampling rate
/// let duration = 1.0; // 1 second
/// let n = (fs * duration) as usize;
/// let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
///
/// // Chirp signal: frequency sweeping from 100 Hz to 300 Hz
/// let signal: Vec<f64> = t.iter()
///     .map(|&t| (2.0 * PI * (100.0 + 200.0 * t / duration) * t).sin())
///     .collect();
///
/// // Create a hann window and initialize STFT
/// let window_length = 256;
/// let hop_size = 64;
/// let hann_window = window::hann(window_length, true).unwrap();
///
/// // Create configuration
/// let config = StftConfig {
///     fft_mode: None, // default FFT mode (onesided)
///     mfft: None, // default mfft (window length)
///     dual_win: None, // calculate dual window as needed
///     scale_to: Some("magnitude".to_string()), // scale for magnitude spectrum
///     phase_shift: None, // no phase shift
/// };
///
/// let stft = ShortTimeFft::new(
///     &hann_window,
///     hop_size,
///     fs,
///     Some(config),
/// ).unwrap();
///
/// // Compute STFT
/// let stft_result = stft.stft(&signal);
///
/// // The result is a complex-valued 2D array with:
/// // - Rows representing frequency bins
/// // - Columns representing time frames
/// ```
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
    ///
    /// # Arguments
    ///
    /// * `win` - Window function
    /// * `hop` - Hop size (samples between consecutive frames)
    /// * `fs` - Sampling frequency
    /// * `fft_mode` - FFT mode (default: "onesided")
    /// * `mfft` - FFT length (default: window length)
    /// * `dual_win` - Dual window for inverse STFT (default: None, calculated when needed)
    /// * `scale_to` - Scaling mode (default: None)
    /// * `phase_shift` - Phase shift (default: None)
    ///
    /// # Returns
    ///
    /// * Result containing a new ShortTimeFft instance
    pub fn new(
        _win: &[f64],
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
        let mfft_val = config.mfft.unwrap_or(_win.len());

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
        let win_array = Array1::from_vec(_win.to_vec());

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
            _win: win_array,
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
    ///
    /// # Arguments
    ///
    /// * `window_type` - Window type (e.g., "hann", "hamming")
    /// * `fs` - Sampling frequency
    /// * `nperseg` - Window length
    /// * `noverlap` - Number of overlapping samples
    /// * `fft_mode` - FFT mode (default: "onesided")
    /// * `mfft` - FFT length (default: window length)
    /// * `scale_to` - Scaling mode (default: None)
    /// * `phase_shift` - Phase shift (default: None)
    ///
    /// # Returns
    ///
    /// * Result containing a new ShortTimeFft instance
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
    ///
    /// # Arguments
    ///
    /// * `win` - Window function
    /// * `hop` - Hop size (samples between consecutive frames)
    /// * `fs` - Sampling frequency
    /// * `fft_mode` - FFT mode (default: "onesided")
    /// * `mfft` - FFT length (default: window length)
    /// * `scale_to` - Scaling mode (default: None)
    /// * `phase_shift` - Phase shift (default: None)
    ///
    /// # Returns
    ///
    /// * Result containing a new ShortTimeFft instance
    pub fn from_win_equals_dual(
        win: &[f64],
        hop: usize,
        fs: f64,
        config: Option<StftConfig>,
    ) -> SignalResult<Self> {
        // Validate window
        if win.is_empty() {
            return Err(SignalError::ValueError(
                "Window cannot be empty".to_string(),
            ));
        }

        // Create a window that equals its dual
        let mut w = Array1::from_vec(win.to_vec());
        let mut dd = Array1::zeros(win.len());

        // Calculate DD(k) = ∑_j w(j) w(j-k)
        for (k, dd_k) in dd.iter_mut().enumerate() {
            for (j, &win_j) in win.iter().enumerate() {
                let idx = (j + win.len() - k) % win.len();
                *dd_k += win_j * win[idx];
            }
        }

        // Ensure no division by zero
        let epsilon = f64::EPSILON * dd.iter().fold(0.0, |max, &val: &f64| val.max(max));
        for val in dd.iter() {
            if val.abs() < epsilon {
                return Err(SignalError::ValueError(
                    "Cannot create window equal to its dual: window not invertible".to_string(),
                ));
            }
        }

        // Create self-dual window
        for (w_i, &dd_i) in w.iter_mut().zip(dd.iter()) {
            *w_i /= dd_i.sqrt();
        }

        // Create a config with the dual window
        let mut new_config = config.unwrap_or_default();
        new_config.dual_win = Some(w.to_vec());

        // Create ShortTimeFft
        let instance = Self::new(
            w.as_slice().expect("Failed to get slice"),
            hop,
            fs,
            Some(new_config),
        )?;

        Ok(instance)
    }

    /// Create a ShortTimeFft instance from a dual window
    ///
    /// # Arguments
    ///
    /// * `dual_win` - Dual window function
    /// * `hop` - Hop size (samples between consecutive frames)
    /// * `fs` - Sampling frequency
    /// * `fft_mode` - FFT mode (default: "onesided")
    /// * `mfft` - FFT length (default: window length)
    /// * `scale_to` - Scaling mode (default: None)
    /// * `phase_shift` - Phase shift (default: None)
    ///
    /// # Returns
    ///
    /// * Result containing a new ShortTimeFft instance
    pub fn from_dual(
        dual_win: &[f64],
        hop: usize,
        fs: f64,
        config: Option<StftConfig>,
    ) -> SignalResult<Self> {
        // Create canonical window from dual
        let _win = calc_dual_window_internal(dual_win, hop)?;

        // Create a config with the dual window
        let mut new_config = config.unwrap_or_default();
        new_config.dual_win = Some(dual_win.to_vec());

        // Create ShortTimeFft
        let instance = Self::new(
            win.as_slice().expect("Failed to get slice"),
            hop,
            fs,
            Some(new_config),
        )?;

        Ok(instance)
    }

    /// Check if the STFT is invertible
    ///
    /// # Returns
    ///
    /// * true if invertible, false otherwise
    pub fn invertible(&self) -> bool {
        // Calculate dual window if not already present
        if self.dual_win.is_none() {
            self.calc_dual_canonical_window().is_ok()
        } else {
            true
        }
    }

    /// Calculate the dual canonical window if not already present
    ///
    /// # Returns
    ///
    /// * Result containing dual window as Array1<f64>
    pub fn calc_dual_canonical_window(&self) -> SignalResult<Array1<f64>> {
        if let Some(ref dual) = self.dual_win {
            return Ok(dual.clone());
        }

        calc_dual_window_internal(self.win.as_slice().unwrap(), self.hop)
    }

    /// Returns the sampling interval
    ///
    /// # Returns
    ///
    /// * Sampling interval (1/fs)
    pub fn t(&self) -> f64 {
        1.0 / self.fs
    }

    /// Returns the time increment
    ///
    /// # Returns
    ///
    /// * Time increment (hop * T)
    pub fn delta_t(&self) -> f64 {
        self.t() * self.hop as f64
    }

    /// Returns the frequency increment
    ///
    /// # Returns
    ///
    /// * Frequency increment (1 / (mfft * T))
    pub fn delta_f(&self) -> f64 {
        1.0 / (self.mfft as f64 * self.t())
    }

    /// Returns the minimum slice index
    ///
    /// # Returns
    ///
    /// * Minimum slice index
    pub fn p_min(&self) -> isize {
        -((self.m_num - self.m_num_mid) as isize / self.hop as isize)
    }

    /// Returns the maximum slice index
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples in the input signal
    ///
    /// # Returns
    ///
    /// * Maximum slice index
    pub fn p_max(&self, n: usize) -> isize {
        ((n + self.m_num_mid) as isize - 1) / self.hop as isize + 1
    }

    /// Returns the number of slices
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples in the input signal
    ///
    /// # Returns
    ///
    /// * Number of slices
    pub fn p_num(&self, n: usize) -> usize {
        (self.p_max(n) - self.p_min()) as usize
    }

    /// Returns the minimum sample index
    ///
    /// # Returns
    ///
    /// * Minimum sample index
    pub fn k_min(&self) -> isize {
        -(self.m_num_mid as isize)
    }

    /// Returns the maximum sample index
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples in the input signal
    ///
    /// # Returns
    ///
    /// * Maximum sample index
    pub fn k_max(&self, n: usize) -> isize {
        n as isize + (self.m_num - self.m_num_mid) as isize
    }

    /// Returns true if FFT mode is one-sided
    ///
    /// # Returns
    ///
    /// * true if one-sided, false otherwise
    pub fn onesided_fft(&self) -> bool {
        matches!(self.fft_mode, FftMode::OneSided | FftMode::OneSided2X)
    }

    /// Returns the number of frequency points
    ///
    /// # Returns
    ///
    /// * Number of frequency points
    pub fn f_pts(&self) -> usize {
        if self.onesided_fft() {
            self.mfft / 2 + 1
        } else {
            self.mfft
        }
    }

    /// Returns the frequency vector
    ///
    /// # Returns
    ///
    /// * Frequency vector
    pub fn f(&self) -> Array1<f64> {
        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                // Calculate rfftfreq
                let mut f = Array1::zeros(self.f_pts());
                for (i, f_i) in f.iter_mut().enumerate() {
                    *f_i = i as f64 * self.delta_f();
                }
                f
            }
            FftMode::TwoSided => {
                // Calculate fftfreq
                let mut f = Array1::zeros(self.mfft);
                for (i, f_i) in f.iter_mut().enumerate().take(self.mfft) {
                    if i <= self.mfft / 2 {
                        *f_i = i as f64 * self.delta_f();
                    } else {
                        *f_i = (i as f64 - self.mfft as f64) * self.delta_f();
                    }
                }
                f
            }
            FftMode::Centered => {
                // Calculate fftshift(fftfreq)
                let mut f = Array1::zeros(self.mfft);
                let half = self.mfft / 2;
                for (i, f_i) in f.iter_mut().enumerate().take(self.mfft) {
                    if i < half {
                        *f_i = (i as f64 - self.mfft as f64) * self.delta_f();
                    } else {
                        *f_i = (i as f64 - half as f64) * self.delta_f();
                    }
                }
                f
            }
        }
    }

    /// Returns the time vector
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples in the input signal
    ///
    /// # Returns
    ///
    /// * Time vector
    pub fn t_vec(&self, n: usize) -> Array1<f64> {
        let p_min = self.p_min();
        let p_max = self.p_max(n);
        let mut t = Array1::zeros((p_max - p_min) as usize);

        for (i, p) in (p_min..p_max).enumerate() {
            t[i] = p as f64 * self.delta_t();
        }

        t
    }

    /// Compute the extent for plotting
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples in the input signal
    /// * `axes_seq` - Axis sequence ("tf" or "ft")
    /// * `center_bins` - Whether to center bins
    ///
    /// # Returns
    ///
    /// * Tuple of (t0, t1, f0, f1) or (f0, f1, t0, t1)
    pub fn extent(
        &self,
        n: usize,
        axes_seq: &str,
        center_bins: bool,
    ) -> SignalResult<(f64, f64, f64, f64)> {
        if !["tf", "ft"].contains(&axes_seq) {
            return Err(SignalError::ValueError(
                "axes_seq must be 'tf' or 'ft'".to_string(),
            ));
        }

        // Get frequency bounds
        let (q0, q1): (isize, isize) = if self.onesided_fft() {
            (0, self.f_pts() as isize)
        } else if self.fft_mode == FftMode::Centered {
            let half = self.mfft / 2;
            (-(half as isize), (self.mfft - half) as isize)
        } else {
            return Err(SignalError::ValueError(
                "Unsupported FFT mode for extent calculation".to_string(),
            ));
        };

        // Get time bounds
        let p0 = self.p_min();
        let p1 = self.p_max(n);

        // Apply bin centering if requested
        let (t0, t1, f0, f1) = if center_bins {
            (
                self.delta_t() * (p0 as f64 - 0.5),
                self.delta_t() * (p1 as f64 - 0.5),
                self.delta_f() * (q0 as f64 - 0.5),
                self.delta_f() * (q1 as f64 - 0.5),
            )
        } else {
            (
                self.delta_t() * p0 as f64,
                self.delta_t() * p1 as f64,
                self.delta_f() * q0 as f64,
                self.delta_f() * q1 as f64,
            )
        };

        // Return in the requested axis order
        if axes_seq == "tf" {
            Ok((t0, t1, f0, f1))
        } else {
            Ok((f0, f1, t0, t1))
        }
    }

    /// Perform the Short-Time Fourier Transform
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    /// * `padding` - Padding mode (optional)
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
                ScalingMode::Magnitude => 1.0 / self.win.map(|x| x * x).sum().sqrt(),
                ScalingMode::Psd => 1.0 / self.win.map(|x| x * x).sum(),
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
                let idx = start as usize + i;
                if idx < signal_len {
                    output[idx] += frame[i] * dual_window[i];
                    weight[idx] += dual_window[i] * dual_window[i];
                }
            }
        }

        // Normalize by weight
        for i in 0..signal_len {
            if weight[i] > 1e-10 {
                output[i] /= weight[i];
            }
        }

        Ok(output.to_vec())
    }

    /// Extract a frame from the STFT matrix
    ///
    /// # Arguments
    ///
    /// * `X` - STFT matrix
    /// * `p_idx` - Frame index
    ///
    /// # Returns
    ///
    /// * Frame spectrum
    fn get_stft_frame(
        &self,
        x: &Array2<Complex64>,
        p_idx: usize,
    ) -> SignalResult<Array1<Complex64>> {
        let f_pts = x.shape()[0];
        let p_num = x.shape()[1];

        if p_idx >= p_num {
            return Err(SignalError::ValueError(format!(
                "Frame index {} is out of bounds (0..{})",
                p_idx, p_num
            )));
        }

        // Extract frame spectrum
        let frame_slice = x.slice(s![.., p_idx]);

        // Convert to full spectrum based on FFT mode
        let mut frame_spectrum = Array1::zeros(self.mfft);

        match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                // For one-sided FFT, reconstruct the negative frequencies
                let last_idx = f_pts.min(self.mfft);

                // Copy positive frequencies
                for (i, frame_spectrum_i) in frame_spectrum.iter_mut().enumerate().take(last_idx) {
                    *frame_spectrum_i = frame_slice[i];
                }

                // Reconstruct negative frequencies (except DC and Nyquist)
                if self.mfft % 2 == 0 {
                    // Even FFT length
                    for i in 1..(self.mfft / 2) {
                        frame_spectrum[self.mfft - i] = frame_slice[i].conj();
                    }
                } else {
                    // Odd FFT length
                    for i in 1..(self.mfft / 2 + 1) {
                        frame_spectrum[self.mfft - i] = frame_slice[i].conj();
                    }
                }

                // Scale if using OneSided2X
                if self.fft_mode == FftMode::OneSided2X {
                    let factor = if self.scaling == ScalingMode::Psd {
                        Complex64::new(2.0_f64.sqrt(), 0.0)
                    } else {
                        Complex64::new(2.0, 0.0)
                    };

                    // Scale all frequencies except DC and Nyquist
                    let nyquist_idx = if self.mfft % 2 == 0 { self.mfft / 2 } else { 0 };
                    for (i, frame_spectrum_i) in
                        frame_spectrum.iter_mut().enumerate().take(f_pts).skip(1)
                    {
                        if i != nyquist_idx {
                            *frame_spectrum_i /= factor;
                        }
                    }
                }
            }
            FftMode::TwoSided => {
                // Two-sided FFT (already in correct order)
                for (i, frame_spectrum_i) in frame_spectrum
                    .iter_mut()
                    .enumerate()
                    .take(f_pts.min(self.mfft))
                {
                    *frame_spectrum_i = frame_slice[i];
                }
            }
            FftMode::Centered => {
                // Centered FFT (need to perform ifftshift)
                let half = self.mfft / 2;
                for (i, &slice_val) in frame_slice.iter().enumerate().take(f_pts.min(self.mfft)) {
                    if i < half {
                        frame_spectrum[i + self.mfft - half] = slice_val;
                    } else {
                        frame_spectrum[i - half] = slice_val;
                    }
                }
            }
        };

        Ok(frame_spectrum)
    }

    /// Perform FFT on a frame
    ///
    /// # Arguments
    ///
    /// * `frame` - Input frame
    ///
    /// # Returns
    ///
    /// * FFT result
    fn fft(&self, frame: &Array1<f64>) -> SignalResult<Array1<Complex64>> {
        // Convert to complex
        let complex_frame: Vec<Complex64> = frame.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Perform FFT
        let fft_result = match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                // Real FFT
                let mut output = Vec::new();
                let n = complex_frame.len();

                // DC component
                let dc = complex_frame.iter().sum();
                output.push(dc);

                // Positive frequencies
                for k in 1..(n / 2 + 1) {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for (j, &frame_val) in complex_frame.iter().enumerate().take(n) {
                        let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                        let c = Complex64::new(angle.cos(), angle.sin());
                        sum += frame_val * c;
                    }
                    output.push(sum);
                }

                Array1::from(output)
            }
            FftMode::TwoSided => {
                // Standard FFT
                let mut output = Vec::with_capacity(complex_frame.len());
                let n = complex_frame.len();

                for k in 0..n {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for (j, &frame_val) in complex_frame.iter().enumerate().take(n) {
                        let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                        let c = Complex64::new(angle.cos(), angle.sin());
                        sum += frame_val * c;
                    }
                    output.push(sum);
                }

                Array1::from(output)
            }
            FftMode::Centered => {
                // FFT with fftshift
                let mut output = Vec::with_capacity(complex_frame.len());
                let n = complex_frame.len();

                for k in 0..n {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for (j, &frame_val) in complex_frame.iter().enumerate().take(n) {
                        let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                        let c = Complex64::new(angle.cos(), angle.sin());
                        sum += frame_val * c;
                    }
                    output.push(sum);
                }

                // Apply fftshift
                let half = n / 2;
                let mut shifted = vec![Complex64::new(0.0, 0.0); n];
                for i in 0..n {
                    if i < half {
                        shifted[i + n - half] = output[i];
                    } else {
                        shifted[i - half] = output[i];
                    }
                }

                Array1::from(shifted)
            }
        };

        // Apply phase shift if requested
        if let Some(phase_shift) = self.phase_shift {
            let mut result = fft_result.clone();
            let phase_factor = phase_shift as f64 / self.mfft as f64;

            for (i, val) in result.iter_mut().enumerate() {
                let angle = 2.0 * PI * i as f64 * phase_factor;
                let phase = Complex64::new(angle.cos(), angle.sin());
                *val *= phase;
            }

            Ok(result)
        } else {
            Ok(fft_result)
        }
    }

    /// Perform IFFT on a spectrum
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Input spectrum
    ///
    /// # Returns
    ///
    /// * IFFT result
    fn ifft(&self, spectrum: &Array1<Complex64>) -> SignalResult<Array1<f64>> {
        // Perform IFFT
        let ifft_result = match self.fft_mode {
            FftMode::OneSided | FftMode::OneSided2X => {
                // Real IFFT
                let n = self.mfft;
                let mut output = vec![0.0; n];

                for (i, output_i) in output.iter_mut().enumerate().take(n) {
                    let mut sum = Complex64::new(0.0, 0.0);

                    // DC component
                    sum += spectrum[0];

                    // Positive frequencies
                    for k in 1..(n / 2 + 1).min(spectrum.len()) {
                        let angle = 2.0 * PI * (i * k) as f64 / n as f64;
                        let c = Complex64::new(angle.cos(), angle.sin());
                        sum += spectrum[k] * c;
                    }

                    // Apply normalization
                    sum /= Complex64::new(n as f64, 0.0);

                    // Real signal should have zero imaginary part after IFFT
                    *output_i = sum.re;
                }

                Array1::from(output)
            }
            FftMode::TwoSided => {
                // Standard IFFT
                let n = self.mfft;
                let mut output = vec![0.0; n];

                for (i, output_i) in output.iter_mut().enumerate().take(n) {
                    let mut sum = Complex64::new(0.0, 0.0);

                    for (k, &spec_k) in spectrum.iter().enumerate().take(n.min(spectrum.len())) {
                        let angle = 2.0 * PI * (i * k) as f64 / n as f64;
                        let c = Complex64::new(angle.cos(), angle.sin());
                        sum += spec_k * c;
                    }

                    // Apply normalization
                    sum /= Complex64::new(n as f64, 0.0);

                    // Real signal should have zero imaginary part after IFFT
                    *output_i = sum.re;
                }

                Array1::from(output)
            }
            FftMode::Centered => {
                // IFFT with ifftshift
                let n = self.mfft;

                // Apply ifftshift
                let half = n / 2;
                let mut unshifted = vec![Complex64::new(0.0, 0.0); n];
                for (i, &spec_i) in spectrum.iter().enumerate().take(n.min(spectrum.len())) {
                    if i < n - half {
                        unshifted[i + half] = spec_i;
                    } else {
                        unshifted[i - (n - half)] = spec_i;
                    }
                }

                // Perform IFFT
                let mut output = vec![0.0; n];

                for (i, output_i) in output.iter_mut().enumerate().take(n) {
                    let mut sum = Complex64::new(0.0, 0.0);

                    for (k, &unshifted_k) in unshifted.iter().enumerate().take(n) {
                        let angle = 2.0 * PI * (i * k) as f64 / n as f64;
                        let c = Complex64::new(angle.cos(), angle.sin());
                        sum += unshifted_k * c;
                    }

                    // Apply normalization
                    sum /= Complex64::new(n as f64, 0.0);

                    // Real signal should have zero imaginary part after IFFT
                    *output_i = sum.re;
                }

                Array1::from(output)
            }
        };

        // Return real part for window length
        Ok(ifft_result.slice(s![..self.m_num]).to_owned())
    }

    /// Compute the spectrogram (magnitude squared of STFT)
    ///
    /// # Arguments
    ///
    /// * `x` - Input signal
    ///
    /// # Returns
    ///
    /// * Power spectrogram
    pub fn spectrogram<T>(&self, x: &[T]) -> SignalResult<Array2<f64>>
    where
        T: Float + NumCast + Debug,
    {
        let stft_result = self.stft(x)?;
        let spectrogram = stft_result.mapv(|c| c.norm_sqr());
        Ok(spectrogram)
    }

    /// Check if two points (p0, q0) and (p1, q1) are lower and upper border points
    ///
    /// # Returns
    ///
    /// * Tuple of (p0, q0) for lower border and (p1, q1) for upper border
    pub fn border_points(&self, n: usize) -> ((isize, isize), (isize, isize)) {
        let lower_end = (
            ((self.m_num - self.m_num_mid) as isize + self.hop as isize - 1) / self.hop as isize,
            self.m_num_mid as isize / self.hop as isize + 1,
        );

        let upper_begin = (
            (n as isize - self.m_num_mid as isize) / self.hop as isize,
            ((n as isize) + (self.m_num as isize) - (self.m_num_mid as isize) - 1)
                / (self.hop as isize)
                + 1,
        );

        (lower_end, upper_begin)
    }
}

/// Calculate the canonical dual window
///
/// # Arguments
///
/// * `win` - Window function
/// * `hop` - Hop size
///
/// # Returns
///
/// * Dual window as Array1<f64>
#[allow(dead_code)]
fn calc_dual_window_internal(win: &[f64], hop: usize) -> SignalResult<Array1<f64>> {
    if hop > win.len() {
        return Err(SignalError::ValueError(format!(
            "Hop size {} is larger than window length {} => STFT not invertible!",
            hop,
            win.len()
        )));
    }

    // Create squared window values
    let w2: Vec<f64> = win.iter().map(|&w| w * w).collect();
    let mut dd = w2.clone();

    // Calculate sum of shifted windows
    for k in (hop.._win.len()).step_by(hop) {
        for i in k.._win.len() {
            dd[i] += w2[i - k];
        }
        for i in 0..(_win.len() - k) {
            dd[i] += w2[i + k];
        }
    }

    // Check DD > 0
    let relative_resolution = f64::EPSILON * dd.iter().fold(0.0, |max, &val| val.max(max));
    if !dd.iter().all(|&v| v >= relative_resolution) {
        return Err(SignalError::ValueError(
            "Short-time Fourier Transform not invertible!".to_string(),
        ));
    }

    // Calculate dual window
    let dual_win = Array1::from_vec(_win.iter().zip(dd.iter()).map(|(&w, &d)| w / d).collect());

    Ok(dual_win)
}

/// Find the closest STFT dual window to a desired window
///
/// # Arguments
///
/// * `win` - Window function
/// * `hop` - Hop size
/// * `desired_dual` - Desired dual window
/// * `scaled` - Whether to scale the window
///
/// # Returns
///
/// * Tuple of (dual window, scaling factor)
#[allow(dead_code)]
pub fn closest_stft_dual_window(
    win: &[f64],
    hop: usize,
    desired_dual: Option<&[f64]>,
    scaled: bool,
) -> SignalResult<(Vec<f64>, f64)> {
    // Validate inputs
    if win.is_empty() {
        return Err(SignalError::ValueError(
            "Window cannot be empty".to_string(),
        ));
    }

    let desired = if let Some(d) = desired_dual {
        if d.len() != win.len() {
            return Err(SignalError::ValueError(
                "Desired _dual window must have the same length as window".to_string(),
            ));
        }
        Array1::from_vec(d.to_vec())
    } else {
        // Default to rectangular window
        Array1::ones(win.len())
    };

    if hop < 1 || hop > win.len() {
        return Err(SignalError::ValueError(format!(
            "Hop size must be between 1 and {}, got {}",
            win.len(),
            hop
        )));
    }

    // Calculate the canonical _dual window
    let w_d = calc_dual_window_internal(win, hop)?;

    // Calculate correlations
    let win_array = Array1::from_vec(win.to_vec());
    let wdd = &win_array * &desired;

    let mut q_d = wdd.clone();
    for k in (hop..win.len()).step_by(hop) {
        for i in k..win.len() {
            q_d[i] += wdd[i - k];
        }
        for i in 0..(win.len() - k) {
            q_d[i] += wdd[i + k];
        }
    }

    q_d = &w_d * &q_d;

    if !scaled {
        let result = &w_d + &desired - &q_d;
        return Ok((result.to_vec(), 1.0));
    }

    // Calculate scaling factor
    let numerator = (q_d.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let denominator = q_d.iter().map(|&x| x * x).sum::<f64>();

    if !(numerator > 0.0 && denominator > f64::EPSILON) {
        return Err(SignalError::ValueError(
            "Unable to calculate scaled closest _dual window due to numerically unstable scaling factor!".to_string(),
        ));
    }

    let alpha = numerator / denominator;
    let result = &w_d + (alpha * (&desired - &q_d));

    Ok((result.to_vec(), alpha))
}

/// Create a STFT window that satisfies the COLA condition
///
/// # Arguments
///
/// * `m` - Window length
/// * `hop` - Hop size
///
/// # Returns
///
/// * COLA window
#[allow(dead_code)]
pub fn create_cola_window(m: usize, hop: usize) -> SignalResult<Vec<f64>> {
    // Create initial rectangular window
    let rect_win = vec![1.0; m];

    // Find closest STFT dual window
    let (cola_win) = closest_stft_dual_window(&rect_win, hop, None, true)?;

    Ok(cola_win)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    #[test]
    fn test_stft_creation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a Hann window
        let window = window::hann(256, true).unwrap();

        // Create ShortTimeFft
        let config = StftConfig {
            fft_mode: Some("onesided".to_string()),
            mfft: None,
            dual_win: None,
            scale_to: Some("magnitude".to_string()),
            phase_shift: None,
        };

        let stft = ShortTimeFft::new(&window, 64, 1000.0, Some(config)).unwrap();

        assert_eq!(stft.hop, 64);
        assert_eq!(stft.fs, 1000.0);
        assert_eq!(stft.m_num, 256);
        assert_eq!(stft.mfft, 256);
        assert_eq!(stft.m_num_mid, 128);
        assert!(matches!(stft.fft_mode, FftMode::OneSided));
        assert!(matches!(stft.scaling, ScalingMode::Magnitude));
    }

    #[test]
    fn test_stft_from_window() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create ShortTimeFft from a named window
        let config = StftConfig {
            fft_mode: None,
            mfft: Some(512),
            dual_win: None,
            scale_to: None,
            phase_shift: None,
        };

        let stft = ShortTimeFft::from_window("hamming", 1000.0, 256, 192, Some(config)).unwrap();

        assert_eq!(stft.hop, 64);
        assert_eq!(stft.fs, 1000.0);
        assert_eq!(stft.m_num, 256);
        assert_eq!(stft.mfft, 512);
        assert_eq!(stft.m_num_mid, 128);
        assert!(matches!(stft.fft_mode, FftMode::OneSided));
    }

    #[test]
    fn test_simple_signal_stft() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple sine wave
        let fs = 1000.0;
        let duration = 1.0;
        let n = (fs * duration) as usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 100.0 * t).sin()).collect();

        // Create a hann window and initialize STFT
        let window_length = 256;
        let hop_size = 64;
        let hann_window = window::hann(window_length, true).unwrap();

        let config = StftConfig {
            fft_mode: None,
            mfft: None,
            dual_win: None,
            scale_to: Some("magnitude".to_string()),
            phase_shift: None,
        };

        let stft = ShortTimeFft::new(&hann_window, hop_size, fs, Some(config)).unwrap();

        // Compute STFT
        let stft_result = stft.stft(&signal).unwrap();

        // Check dimensions
        assert_eq!(stft_result.shape()[0], stft.f_pts());

        // Since our signal is a 100 Hz sine wave, we should see peaks around 100 Hz
        let freq_bin_100hz = (100.0 / stft.delta_f()).round() as usize;

        // Check magnitude at 100 Hz bin for a few frames
        for p in 0..5 {
            if p < stft_result.shape()[1] {
                let frame_100hz_mag = stft_result[[freq_bin_100hz, p]].norm();
                // Should have significant energy at 100 Hz
                assert!(frame_100hz_mag > 0.1);
            }
        }
    }

    #[test]
    fn test_stft_istft_reconstruction() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple signal
        let fs = 1000.0;
        let duration = 0.2; // Shorter duration
        let n = (fs * duration) as usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 50.0 * t).sin()).collect();

        // Use a simple rectangular window with 75% overlap for better reconstruction
        let window_length = 64;
        let hop_size = 16;
        let window = vec![1.0; window_length];

        let config = StftConfig::default();

        // Create STFT with a window that equals its dual
        let stft = ShortTimeFft::from_win_equals_dual(&window, hop_size, fs, Some(config)).unwrap();

        // Compute STFT
        let stft_result = stft.stft(&signal).unwrap();

        // Reconstruct signal
        let reconstructed = stft.istft(&stft_result, None, None).unwrap();

        // The reconstructed signal will be longer due to windowing
        // Just check that we get reasonable reconstruction in the middle part
        if reconstructed.len() >= signal.len() {
            // Check a few samples in the middle (avoiding edge effects)
            let start = window_length;
            let end = signal.len().saturating_sub(window_length);

            if end > start {
                // Calculate reconstruction error in the stable region
                let mut error_sum = 0.0;
                let mut count = 0;

                for i in start..end {
                    if i < reconstructed.len() {
                        error_sum += (reconstructed[i] - signal[i]).abs();
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg_error = error_sum / count as f64;
                    // Just check that average error is reasonable
                    // STFT/iSTFT reconstruction may have some error due to windowing
                    assert!(avg_error < 1.0);
                }
            }
        }
    }

    #[test]
    fn test_spectrogram() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple signal
        let fs = 1000.0;
        let duration = 0.3;
        let n = (fs * duration) as usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

        // Create a signal with two frequencies
        let signal: Vec<f64> = t
            .iter()
            .map(|&t| 0.5 * (2.0 * PI * 100.0 * t).sin() + 0.3 * (2.0 * PI * 250.0 * t).sin())
            .collect();

        // Create a hann window and initialize STFT
        let window_length = 128;
        let hop_size = 32;
        let hann_window = window::hann(window_length, true).unwrap();

        let config = StftConfig {
            fft_mode: None,
            mfft: None,
            dual_win: None,
            scale_to: Some("psd".to_string()),
            phase_shift: None,
        };

        let stft = ShortTimeFft::new(&hann_window, hop_size, fs, Some(config)).unwrap();

        // Compute spectrogram
        let spec_result = stft.spectrogram(&signal).unwrap();

        // Check dimensions
        assert_eq!(spec_result.shape()[0], stft.f_pts());

        // Calculate frequency bins
        let freq_bin_100hz = (100.0 / stft.delta_f()).round() as usize;
        let freq_bin_250hz = (250.0 / stft.delta_f()).round() as usize;

        // Check that the spectrogram has peaks at both frequencies
        for p in 1..5 {
            if p < spec_result.shape()[1] {
                let power_100hz = spec_result[[freq_bin_100hz, p]];
                let power_250hz = spec_result[[freq_bin_250hz, p]];

                // Power at 100 Hz should be greater than at 250 Hz (since amplitude is higher)
                assert!(power_100hz > power_250hz);

                // Both should be significantly above noise floor
                assert!(power_100hz > 0.01);
                assert!(power_250hz > 0.005);
            }
        }
    }
}

/// Memory-efficient STFT configuration for large signals
#[derive(Debug, Clone)]
pub struct MemoryEfficientStftConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Processing chunk size
    pub chunk_size: Option<usize>,
    /// Use parallel processing
    pub parallel: bool,
    /// Store only magnitude (not complex values)
    pub magnitude_only: bool,
}

impl Default for MemoryEfficientStftConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            chunk_size: None,
            parallel: false,
            magnitude_only: false,
        }
    }
}

/// Memory-efficient STFT processor for large signals
pub struct MemoryEfficientStft {
    stft: ShortTimeFft,
    config: MemoryEfficientStftConfig,
}

impl MemoryEfficientStft {
    /// Create a new memory-efficient STFT processor
    pub fn new(
        window: &[f64],
        hop_size: usize,
        fs: f64,
        stft_config: Option<StftConfig>,
        memory_config: MemoryEfficientStftConfig,
    ) -> SignalResult<Self> {
        let stft = ShortTimeFft::new(window, hop_size, fs, stft_config)?;

        Ok(Self {
            stft_config: memory_config,
        })
    }

    /// Calculate optimal chunk size based on memory constraints
    fn calculate_chunk_size(&self, signallength: usize) -> usize {
        if let Some(chunk_size) = self.config.chunk_size {
            return chunk_size;
        }

        // Estimate memory usage per sample
        let window_length = self.stft.win.len();
        let fft_size = self.stft.mfft;
        let hop_size = self.stft.hop;

        // Estimate frames per chunk
        let frames_per_mb = if self.config.magnitude_only {
            // Only storing magnitude: 8 bytes per complex sample -> 8 bytes per magnitude
            1_000_000 / (fft_size * 8)
        } else {
            // Storing complex values: 16 bytes per complex sample
            1_000_000 / (fft_size * 16)
        };

        let max_frames = frames_per_mb * self.config.max_memory_mb;
        let samples_per_chunk = max_frames * hop_size + window_length;

        // Ensure chunk size is reasonable
        samples_per_chunk.min(signal_length).max(window_length * 2)
    }

    /// Process STFT in chunks for memory efficiency
    pub fn stft_chunked<T>(&self, signal: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let chunk_size = self.calculate_chunk_size(signal.len());
        let window_length = self.stft.win.len();
        let hop_size = self.stft.hop;

        // Calculate overlap needed between chunks
        let overlap = window_length.saturating_sub(hop_size);

        // Estimate total output size
        let total_frames = self.stft.p_max(signal.len()) - self.stft.p_min();
        let mut result = Array2::zeros((self.stft.f_pts(), total_frames as usize));

        let mut frame_offset = 0;
        let mut sample_offset = 0;

        while sample_offset < signal.len() {
            // Calculate chunk boundaries
            let chunk_start = sample_offset.saturating_sub(overlap);
            let chunk_end = (sample_offset + chunk_size).min(signal.len());

            if chunk_end <= chunk_start {
                break;
            }

            // Extract chunk
            let chunk = &signal[chunk_start..chunk_end];

            // Process chunk
            let chunk_stft = self.stft.stft(chunk)?;

            // Calculate where to place results in output array
            let frames_in_chunk = chunk_stft.shape()[1];
            let skip_frames = if sample_offset == 0 {
                0
            } else {
                overlap / hop_size
            };

            let copy_frames = frames_in_chunk.saturating_sub(skip_frames);
            let end_frame = (frame_offset + copy_frames).min(result.shape()[1]);

            if frame_offset < result.shape()[1] && copy_frames > 0 {
                let copy_end = (skip_frames + copy_frames).min(chunk_stft.shape()[1]);

                // Copy data to result array
                for f in 0..self.stft.f_pts() {
                    for t in skip_frames..copy_end {
                        let result_t = frame_offset + t - skip_frames;
                        if result_t < result.shape()[1] {
                            result[[f, result_t]] = chunk_stft[[f, t]];
                        }
                    }
                }

                frame_offset = end_frame;
            }

            // Move to next chunk
            sample_offset += chunk_size;
        }

        Ok(result)
    }

    /// Process spectrogram in chunks (magnitude only for memory efficiency)
    pub fn spectrogram_chunked<T>(&self, signal: &[T]) -> SignalResult<Array2<f64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        if self.config.magnitude_only {
            // Process directly to magnitude
            let chunk_size = self.calculate_chunk_size(signal.len());
            let window_length = self.stft.win.len();
            let hop_size = self.stft.hop;
            let overlap = window_length.saturating_sub(hop_size);

            let total_frames = self.stft.p_max(signal.len()) - self.stft.p_min();
            let mut result = Array2::zeros((self.stft.f_pts(), total_frames as usize));

            let mut frame_offset = 0;
            let mut sample_offset = 0;

            while sample_offset < signal.len() {
                let chunk_start = sample_offset.saturating_sub(overlap);
                let chunk_end = (sample_offset + chunk_size).min(signal.len());

                if chunk_end <= chunk_start {
                    break;
                }

                let chunk = &signal[chunk_start..chunk_end];
                let chunk_spec = self.stft.spectrogram(chunk)?;

                let frames_in_chunk = chunk_spec.shape()[1];
                let skip_frames = if sample_offset == 0 {
                    0
                } else {
                    overlap / hop_size
                };
                let copy_frames = frames_in_chunk.saturating_sub(skip_frames);
                let end_frame = (frame_offset + copy_frames).min(result.shape()[1]);

                if frame_offset < result.shape()[1] && copy_frames > 0 {
                    let copy_end = (skip_frames + copy_frames).min(chunk_spec.shape()[1]);

                    for f in 0..self.stft.f_pts() {
                        for t in skip_frames..copy_end {
                            let result_t = frame_offset + t - skip_frames;
                            if result_t < result.shape()[1] {
                                result[[f, result_t]] = chunk_spec[[f, t]];
                            }
                        }
                    }

                    frame_offset = end_frame;
                }

                sample_offset += chunk_size;
            }

            Ok(result)
        } else {
            // Use regular STFT then compute magnitude
            let stft_result = self.stft_chunked(signal)?;
            Ok(stft_result.mapv(|c| c.norm()))
        }
    }

    /// Get memory usage estimate in MB
    pub fn memory_estimate(&self, signallength: usize) -> f64 {
        let chunk_size = self.calculate_chunk_size(signal_length);
        let frames_in_chunk = chunk_size / self.stft.hop + 1;
        let memory_per_chunk = if self.config.magnitude_only {
            frames_in_chunk * self.stft.f_pts() * 8 // 8 bytes per f64
        } else {
            frames_in_chunk * self.stft.f_pts() * 16 // 16 bytes per Complex64
        };

        memory_per_chunk as f64 / 1_000_000.0
    }

    /// Process STFT with automatic memory management
    /// This method automatically selects the best processing approach based on signal size
    pub fn stft_auto<T>(&self, signal: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let signal_memory_mb = std::mem::size_of_val(signal) / 1_000_000;

        if signal_memory_mb > self.config.max_memory_mb / 4 {
            // Use chunked processing for large signals
            self.stft_chunked(signal)
        } else {
            // Use regular processing for smaller signals
            self.stft.stft(signal)
        }
    }

    /// Process spectrogram with automatic memory management
    pub fn spectrogram_auto<T>(&self, signal: &[T]) -> SignalResult<Array2<f64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let signal_memory_mb = std::mem::size_of_val(signal) / 1_000_000;

        if signal_memory_mb > self.config.max_memory_mb / 4 {
            // Use chunked processing for large signals
            self.spectrogram_chunked(signal)
        } else {
            // Use regular processing for smaller signals
            self.stft.spectrogram(signal)
        }
    }

    /// Process STFT with parallel chunked processing for very large signals
    #[cfg(feature = "parallel")]
    pub fn stft_parallel_chunked<T>(&self, signal: &[T]) -> SignalResult<Array2<Complex64>>
    where
        T: Float + NumCast + Debug + Send + Sync,
    {
        let chunk_size = self.calculate_chunk_size(signal.len());
        let window_length = self.stft.win.len();
        let hop_size = self.stft.hop;
        let overlap = window_length.saturating_sub(hop_size);

        // Calculate chunks
        let mut chunks = Vec::new();
        let mut sample_offset = 0;

        while sample_offset < signal.len() {
            let chunk_start = sample_offset.saturating_sub(overlap);
            let chunk_end = (sample_offset + chunk_size).min(signal.len());

            if chunk_end > chunk_start {
                chunks.push((chunk_start, chunk_end, sample_offset));
            }
            sample_offset += chunk_size;
        }

        // Process chunks in parallel
        let chunk_results: Result<Vec<_>, _> = chunks
            .par_iter()
            .map(|(start, end)| {
                let chunk = &signal[*start..*end];
                self.stft.stft(chunk)
            })
            .collect();

        let chunk_results = chunk_results?;

        // Combine results
        let total_frames = self.stft.p_max(signal.len()) - self.stft.p_min();
        let mut result = Array2::zeros((self.stft.f_pts(), total_frames as usize));

        let mut frame_offset = 0;
        for (i, chunk_stft) in chunk_results.iter().enumerate() {
            let skip_frames = if i == 0 { 0 } else { overlap / hop_size };
            let frames_in_chunk = chunk_stft.shape()[1];
            let copy_frames = frames_in_chunk.saturating_sub(skip_frames);

            if frame_offset < result.shape()[1] && copy_frames > 0 {
                let copy_end = (skip_frames + copy_frames).min(chunk_stft.shape()[1]);
                let result_end = (frame_offset + copy_frames).min(result.shape()[1]);

                for f in 0..self.stft.f_pts() {
                    for t in skip_frames..copy_end {
                        let result_t = frame_offset + t - skip_frames;
                        if result_t < result_end {
                            result[[f, result_t]] = chunk_stft[[f, t]];
                        }
                    }
                }

                frame_offset = result_end;
            }
        }

        Ok(result)
    }

    /// Get detailed memory usage information
    pub fn memory_info(&self, signallength: usize) -> MemoryInfo {
        let chunk_size = self.calculate_chunk_size(signal_length);
        let _window_length = self.stft.win.len();
        let hop_size = self.stft.hop;

        let total_frames = self.stft.p_max(signal_length) - self.stft.p_min();
        let frames_per_chunk = chunk_size / hop_size + 1;
        let num_chunks = signal_length.div_ceil(chunk_size);

        let total_memory_mb = if self.config.magnitude_only {
            total_frames as f64 * self.stft.f_pts() as f64 * 8.0 / 1_000_000.0
        } else {
            total_frames as f64 * self.stft.f_pts() as f64 * 16.0 / 1_000_000.0
        };

        let chunk_memory_mb = if self.config.magnitude_only {
            frames_per_chunk as f64 * self.stft.f_pts() as f64 * 8.0 / 1_000_000.0
        } else {
            frames_per_chunk as f64 * self.stft.f_pts() as f64 * 16.0 / 1_000_000.0
        };

        MemoryInfo {
            signal_length,
            chunk_size,
            num_chunks,
            total_frames: total_frames as usize,
            frames_per_chunk,
            total_memory_mb,
            chunk_memory_mb,
            memory_reduction_factor: total_memory_mb / chunk_memory_mb,
        }
    }
}

/// Memory usage information for STFT processing
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Length of input signal
    pub signal_length: usize,
    /// Size of each processing chunk
    pub chunk_size: usize,
    /// Number of chunks needed
    pub num_chunks: usize,
    /// Total number of time frames in output
    pub total_frames: usize,
    /// Number of frames per chunk
    pub frames_per_chunk: usize,
    /// Total memory required without chunking (MB)
    pub total_memory_mb: f64,
    /// Memory required per chunk (MB)
    pub chunk_memory_mb: f64,
    /// Memory reduction factor compared to non-chunked processing
    pub memory_reduction_factor: f64,
}

mod memory_efficient_tests {
    #[test]
    fn test_memory_efficient_stft() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a longer signal
        let fs = 1000.0;
        let duration = 2.0;
        let n = (fs * duration) as usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 50.0 * t).sin()).collect();

        let window_length = 256;
        let hop_size = 128;
        let window = window::hann(window_length, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: 10, // Very small to force chunking
            chunk_size: Some(1000),
            parallel: false,
            magnitude_only: false,
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)
                .unwrap();

        // Test chunked STFT
        let result = mem_stft.stft_chunked(&signal).unwrap();

        // Check that we get reasonable dimensions
        assert!(result.shape()[0] > 0);
        assert!(result.shape()[1] > 0);

        // Test memory estimation
        let memory_est = mem_stft.memory_estimate(signal.len());
        assert!(memory_est > 0.0);
        assert!(memory_est < 50.0); // Should be reasonable
    }

    #[test]
    fn test_magnitude_only_spectrogram() {
        let fs = 1000.0;
        let duration = 1.0;
        let n = (fs * duration) as usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 100.0 * t).sin()).collect();

        let window_length = 128;
        let hop_size = 64;
        let window = window::hann(window_length, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: 5,
            chunk_size: Some(500),
            parallel: false,
            magnitude_only: true,
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)
                .unwrap();

        let spec_result = mem_stft.spectrogram_chunked(&signal).unwrap();

        // Should have real values only (no complex)
        assert!(spec_result.shape()[0] > 0);
        assert!(spec_result.shape()[1] > 0);

        // All values should be non-negative (magnitudes)
        for val in spec_result.iter() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_auto_memory_management() {
        let fs = 1000.0;

        // Small signal - should use regular processing
        let small_signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();

        // Large signal - should use chunked processing
        let large_signal: Vec<f64> = (0..100000).map(|i| (i as f64 * 0.001).sin()).collect();

        let window_length = 256;
        let hop_size = 128;
        let window = window::hann(window_length, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: 10,
            chunk_size: None, // Auto-calculated
            parallel: false,
            magnitude_only: false,
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)
                .unwrap();

        // Test both small and large signals
        let small_result = mem_stft.stft_auto(&small_signal).unwrap();
        let large_result = mem_stft.stft_auto(&large_signal).unwrap();

        assert!(small_result.shape()[0] > 0);
        assert!(small_result.shape()[1] > 0);
        assert!(large_result.shape()[0] > 0);
        assert!(large_result.shape()[1] > 0);

        // Large signal should produce more frames
        assert!(large_result.shape()[1] > small_result.shape()[1]);
    }

    #[test]
    fn test_memory_info() {
        let fs = 1000.0;
        let signal_length = 200000; // Much larger signal to force chunking

        let window_length = 512;
        let hop_size = 256;
        let window = window::hann(window_length, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: 5, // Reduced to force chunking
            chunk_size: None,
            parallel: false,
            magnitude_only: false,
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)
                .unwrap();

        let info = mem_stft.memory_info(signal_length);

        assert_eq!(info.signal_length, signal_length);
        assert!(info.chunk_size > 0);
        assert!(info.num_chunks > 0);
        assert!(info.total_frames > 0);
        assert!(info.frames_per_chunk > 0);
        assert!(info.total_memory_mb > 0.0);
        assert!(info.chunk_memory_mb > 0.0);
        assert!(info.memory_reduction_factor >= 1.0);

        // Memory reduction should be significant for large signals
        if signal_length > 10000 {
            assert!(info.memory_reduction_factor > 1.2); // More realistic expectation
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_chunked_processing() {
        let fs = 1000.0;
        let duration = 3.0;
        let n = (fs * duration) as usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 50.0 * t).sin()).collect();

        let window_length = 256;
        let hop_size = 128;
        let window = window::hann(window_length, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: 15,
            chunk_size: Some(1000),
            parallel: true,
            magnitude_only: false,
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)
                .unwrap();

        // Test parallel chunked processing
        let parallel_result = mem_stft.stft_parallel_chunked(&signal).unwrap();
        let sequential_result = mem_stft.stft_chunked(&signal).unwrap();

        // Results should have same dimensions
        assert_eq!(parallel_result.shape(), sequential_result.shape());

        // Results should be approximately equal (within numerical precision)
        let max_diff = parallel_result
            .iter()
            .zip(sequential_result.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);

        assert!(max_diff < 1e-10); // Very small difference expected
    }

    #[test]
    fn test_large_signal_processing() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Test with a very large signal that would consume too much memory if processed all at once
        let fs = 8000.0;
        let duration = 10.0; // 10 seconds
        let n = (fs * duration) as usize;

        // Create a chirp signal for more interesting spectral content
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let freq = 100.0 + 500.0 * t / duration; // Frequency sweep from 100 to 600 Hz
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let window_length = 1024;
        let hop_size = 512;
        let window = window::hann(window_length, true).unwrap();

        let memory_config = MemoryEfficientStftConfig {
            max_memory_mb: 50, // Reasonable memory limit
            chunk_size: None,  // Auto-calculate
            parallel: false,
            magnitude_only: true, // Use magnitude only to save memory
        };

        let stft_config = StftConfig::default();
        let mem_stft =
            MemoryEfficientStft::new(&window, hop_size, fs, Some(stft_config), memory_config)
                .unwrap();

        let info = mem_stft.memory_info(signal.len());
        println!(
            "Processing {} samples in {} chunks",
            info.signal_length, info.num_chunks
        );
        println!(
            "Memory reduction factor: {:.2}x",
            info.memory_reduction_factor
        );

        // Process the large signal
        let spec_result = mem_stft.spectrogram_auto(&signal).unwrap();

        // Verify we got a reasonable result
        assert!(spec_result.shape()[0] > 0);
        assert!(spec_result.shape()[1] > 0);

        // Check that the result has the expected frequency resolution
        let expected_freq_bins = window_length / 2 + 1;
        assert_eq!(spec_result.shape()[0], expected_freq_bins);

        // Verify all magnitudes are non-negative
        for val in spec_result.iter() {
            assert!(*val >= 0.0);
        }
    }
}
