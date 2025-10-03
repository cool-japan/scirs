use scirs2_core::ndarray::s;
// Higher-order spectral analysis module
//
// This module implements higher-order spectral analysis methods, including:
// - Bispectrum: third-order spectrum for detecting quadratic phase coupling
// - Bicoherence: normalized bispectrum for phase coupling detection
// - Trispectrum: fourth-order spectrum for cubic phase coupling detection
// - Various estimators and windowing techniques
//
// Higher-order spectra can reveal non-linear coupling between frequency components
// that are not visible in traditional power spectral density estimates.
//
// # Example
// ```
// use scirs2_core::ndarray::Array1;
// use scirs2_signal::higher_order::{bispectrum, bicoherence};
//
// // Create a signal with quadratic phase coupling
// let n = 1024;
// let fs = 1000.0;
// let t = Array1::linspace(0.0, (n as f64 - 1.0) / fs, n);
//
// let f1 = 50.0;
// let f2 = 120.0;
// let f3 = f1 + f2; // Sum frequency - will show phase coupling
//
// // Signal with phase coupling: sin(2πf₁t) + sin(2πf₂t) + sin(2π(f₁+f₂)t + θ)
// // When θ = 0, there is perfect phase coupling
// let phase_coupling = 0.0;
// let signal = t.mapv(|ti| (2.0 * PI * f1 * ti).sin() +
//                          (2.0 * PI * f2 * ti).sin() +
//                          0.5 * (2.0 * PI * f3 * ti + phase_coupling).sin());
//
// // Compute bispectrum
// let nfft = 256;
// let (bispec, f1_axis, f2_axis) = bispectrum(&signal, nfft, Some("hann"), None, fs).unwrap();
//
// // Compute bicoherence (normalized bispectrum)
// let (bicoh_) = bicoherence(&signal, nfft, Some("hann"), None, fs).unwrap();
//
// // Look for peaks in bicoherence to detect phase coupling
// // Strong peaks at (f1, f2) indicate quadratic phase coupling between f1, f2, and f1+f2
// ```

use crate::error::{SignalError, SignalResult};
use crate::window;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Complex64, ComplexFloat};
use scirs2_fft;

#[allow(unused_imports)]
// Type aliases for complex return types
/// Result type for bicoherence and biamplitude functions containing 2D array and frequency axes
pub type BicoherenceResult = (Array2<f64>, (Array1<f64>, Array1<f64>));

/// Method for bispectrum/bicoherence estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BispecEstimator {
    /// Direct FFT-based estimate (single segment)
    Direct,

    /// Indirect estimate using triple correlation
    Indirect,

    /// Welch-like method using averaged segments
    Welch,
}

/// Configuration parameters for higher-order spectral analysis
#[derive(Debug, Clone)]
pub struct HigherOrderConfig {
    /// Estimator method to use
    pub estimator: BispecEstimator,

    /// Sampling frequency
    pub fs: f64,

    /// Window function to use
    pub window: Option<String>,

    /// Number of segments for Welch method (None = auto)
    pub n_segments: Option<usize>,

    /// Overlap fraction between segments (0.0 to 0.99)
    pub overlap: f64,

    /// FFT size (None = auto)
    pub nfft: Option<usize>,

    /// Whether to apply detrending to segments
    pub detrend: bool,

    /// Whether to zero-pad segments
    pub pad: bool,

    /// Whether to return only the non-redundant region
    pub non_redundant: bool,
}

impl Default for HigherOrderConfig {
    fn default() -> Self {
        Self {
            estimator: BispecEstimator::Welch,
            fs: 1.0,
            window: Some("hann".to_string()),
            n_segments: None,
            overlap: 0.5,
            nfft: None,
            detrend: true,
            pad: true,
            non_redundant: true,
        }
    }
}

/// Computes the bispectrum of a signal
///
/// The bispectrum is a third-order spectrum that measures the correlation between
/// frequencies f1, f2, and f1+f2. It can detect quadratic phase coupling between
/// frequency components.
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name (e.g., "hann", "hamming")
/// * `n_segments` - Number of segments for Welch-like estimation (None = auto)
/// * `fs` - Sampling frequency
///
/// # Returns
/// * `bispectrum` - 2D array containing the bispectrum magnitude
/// * `f1_axis` - Frequency axis for the first dimension
/// * `f2_axis` - Frequency axis for the second dimension
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::higher_order::bispectrum;
///
/// let signal = Array1::from_vec(vec![1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]);
/// let (bis, f1, f2) = bispectrum(&signal, 16, Some("hann"), None, 1.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn bispectrum(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    n_segments: Option<usize>,
    fs: f64,
) -> SignalResult<(Array2<f64>, Array1<f64>, Array1<f64>)> {
    // Create configuration
    let config = HigherOrderConfig {
        fs,
        nfft: Some(nfft),
        window: window.map(|w| w.to_string()),
        n_segments,
        ..HigherOrderConfig::default()
    };

    // Compute bispectrum
    let (bis_complex, f1_axis, f2_axis) = compute_bispectrum(signal, &config)?;

    // Convert to magnitude
    let bis_mag = bis_complex.mapv(|c| c.norm());

    Ok((bis_mag, f1_axis, f2_axis))
}

/// Computes the bicoherence of a signal
///
/// The bicoherence is a normalized version of the bispectrum, with values between 0 and 1.
/// It measures the amount of phase coupling between frequencies f1, f2, and f1+f2,
/// independent of their amplitudes.
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name (e.g., "hann", "hamming")
/// * `n_segments` - Number of segments for Welch-like estimation (None = auto)
/// * `fs` - Sampling frequency
///
/// # Returns
/// * `bicoherence` - 2D array containing the bicoherence magnitude (0-1)
/// * `frequency_axes` - (f1_axis, f2_axis) frequency axes
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::higher_order::bicoherence;
///
/// let signal = Array1::from_vec(vec![1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]);
/// let (bic, (f1, f2)) = bicoherence(&signal, 16, Some("hann"), None, 1.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn bicoherence(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    n_segments: Option<usize>,
    fs: f64,
) -> SignalResult<BicoherenceResult> {
    // Create configuration
    let config = HigherOrderConfig {
        fs,
        nfft: Some(nfft),
        window: window.map(|w| w.to_string()),
        n_segments,
        ..HigherOrderConfig::default()
    };

    // Compute bispectrum and power spectrum for normalization
    let (bis_complex, f1_axis, f2_axis) = compute_bispectrum(signal, &config)?;
    let power_spectrum = compute_power_spectrum(signal, &config)?;

    // Create the bicoherence
    let mut bicoherence = Array2::zeros(bis_complex.raw_dim());

    // Compute the bicoherence using the normalization formula
    for i in 0..f1_axis.len() {
        let f1 = f1_axis[i];
        let i_idx = (f1 * nfft as f64 / fs).round() as usize;

        for j in 0..f2_axis.len() {
            let f2 = f2_axis[j];
            let j_idx = (f2 * nfft as f64 / fs).round() as usize;

            // Calculate f1+f2 index
            let sum_freq = f1 + f2;
            let sum_idx = (sum_freq * nfft as f64 / fs).round() as usize % nfft;

            // Check for valid indices
            if i_idx < power_spectrum.len()
                && j_idx < power_spectrum.len()
                && sum_idx < power_spectrum.len()
            {
                // Compute the normalization factor
                let norm_factor =
                    (power_spectrum[i_idx] * power_spectrum[j_idx] * power_spectrum[sum_idx])
                        .sqrt();

                if norm_factor > 1e-10 {
                    // Compute bicoherence
                    bicoherence[[i, j]] = bis_complex[[i, j]].norm() / norm_factor;
                }
            }
        }
    }

    Ok((bicoherence, (f1_axis, f2_axis)))
}

/// Internal function to compute the bispectrum
#[allow(dead_code)]
fn compute_bispectrum(
    signal: &Array1<f64>,
    config: &HigherOrderConfig,
) -> SignalResult<(Array2<Complex64>, Array1<f64>, Array1<f64>)> {
    // Validate parameters
    if signal.len() < 4 {
        return Err(SignalError::ValueError(
            "Signal must have at least 4 data points".to_string(),
        ));
    }

    // Determine number of segments and FFT size
    let n = signal.len();
    let nfft = config.nfft.unwrap_or_else(|| {
        let mut size = 2usize.pow((n as f64).log2().ceil() as u32);
        if size < 256 {
            size = 256; // Minimum FFT size
        }
        size
    });

    // Create frequency axes
    let _freq_step = config.fs / nfft as f64;
    let max_freq = config.fs / 2.0;
    let freq_bins = (nfft / 2) + 1;

    let f1_axis = Array1::linspace(0.0, max_freq, freq_bins);
    let f2_axis = f1_axis.clone();

    // Select the estimator method
    match config.estimator {
        BispecEstimator::Direct => compute_direct_bispectrum(signal, nfft, config),
        BispecEstimator::Indirect => compute_indirect_bispectrum(signal, nfft, config),
        BispecEstimator::Welch => compute_welch_bispectrum(signal, nfft, config),
    }
    .map(|bis| (bis, f1_axis, f2_axis))
}

/// Computes the bispectrum using the direct FFT-based method
#[allow(dead_code)]
fn compute_direct_bispectrum(
    signal: &Array1<f64>,
    nfft: usize,
    config: &HigherOrderConfig,
) -> SignalResult<Array2<Complex64>> {
    // Apply window if specified
    let windowed_signal = if let Some(window_name) = &config.window {
        apply_window(signal, window_name)?
    } else {
        signal.clone()
    };

    // Compute FFT
    let signal_fft = compute_fft(&windowed_signal, nfft)?;

    // Number of frequency bins (for non-redundant region)
    let n_bins = (nfft / 2) + 1;

    // Initialize the bispectrum matrix
    let mut bispectrum = Array2::zeros((n_bins, n_bins));

    // Compute direct bispectrum: B(f1, f2) = X(f1) * X(f2) * X*(f1+f2)
    for i in 0..n_bins {
        for j in 0..=i {
            // Use triangular region for efficiency
            // Calculate the index for f1+f2
            let k = (i + j) % nfft;

            // Compute bispectrum value
            let b_val = signal_fft[i] * signal_fft[j] * signal_fft[k].conj();

            // Store the value
            bispectrum[[i, j]] = b_val;

            // Mirror for symmetry if needed
            if !config.non_redundant || i != j {
                bispectrum[[j, i]] = b_val;
            }
        }
    }

    Ok(bispectrum)
}

/// Computes the bispectrum using the indirect method via triple correlation
#[allow(dead_code)]
fn compute_indirect_bispectrum(
    signal: &Array1<f64>,
    nfft: usize,
    config: &HigherOrderConfig,
) -> SignalResult<Array2<Complex64>> {
    // Apply window if specified
    let windowed_signal = if let Some(window_name) = &config.window {
        apply_window(signal, window_name)?
    } else {
        signal.clone()
    };

    // Number of frequency bins (for non-redundant region)
    let n_bins = (nfft / 2) + 1;

    // Compute triple correlation (3rd-order cumulant)
    let triple_corr = compute_triple_correlation(&windowed_signal, n_bins)?;

    // Apply 2D FFT to get bispectrum
    let bispectrum = compute_2d_fft(&triple_corr, nfft)?;

    Ok(bispectrum)
}

/// Computes the bispectrum using the Welch-like method (segment averaging)
#[allow(dead_code)]
fn compute_welch_bispectrum(
    signal: &Array1<f64>,
    nfft: usize,
    config: &HigherOrderConfig,
) -> SignalResult<Array2<Complex64>> {
    // Determine segment size and overlap
    let n = signal.len();

    // Default segment size is nfft or signal length, whichever is smaller
    let segment_size = nfft.min(n);

    // Determine segment step based on overlap
    let overlap_samples = (segment_size as f64 * config.overlap).round() as usize;
    let step = segment_size - overlap_samples;

    if step == 0 {
        return Err(SignalError::ValueError(
            "Overlap too large, resulting in zero step size".to_string(),
        ));
    }

    // Determine number of segments
    let n_segments = if let Some(segs) = config.n_segments {
        segs
    } else {
        // Auto-determine number of segments
        ((n - overlap_samples) as f64 / step as f64).floor() as usize
    };

    if n_segments == 0 {
        return Err(SignalError::ValueError(
            "Signal too short for the specified segment size and overlap".to_string(),
        ));
    }

    // Number of frequency bins (for non-redundant region)
    let n_bins = (nfft / 2) + 1;

    // Initialize the averaged bispectrum
    let mut bispectrum_avg = Array2::zeros((n_bins, n_bins));

    // Process each segment
    for i in 0..n_segments {
        // Extract segment
        let start = i * step;
        let end = (start + segment_size).min(n);

        if end - start < 4 {
            continue; // Skip very short segments
        }

        let segment = signal.slice(s![start..end]).to_owned();

        // Apply window if specified
        let windowed_segment = if let Some(window_name) = &config.window {
            apply_window(&segment, window_name)?
        } else {
            segment
        };

        // Compute segment bispectrum using direct method
        let seg_bispec = compute_direct_bispectrum(&windowed_segment, nfft, config)?;

        // Accumulate
        bispectrum_avg += &seg_bispec;
    }

    // Normalize by number of segments
    bispectrum_avg.mapv_inplace(|x| x / n_segments as f64);

    Ok(bispectrum_avg)
}

/// Apply a window function to a signal
#[allow(dead_code)]
fn apply_window(_signal: &Array1<f64>, windowname: &str) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Get the window function
    let win = window::get_window(window_name, n, true)?;

    // Apply the window
    let win_arr = Array1::from(win);
    let windowed = _signal * &win_arr;

    Ok(windowed)
}

/// Compute the Fast Fourier Transform of a signal
#[allow(dead_code)]
fn compute_fft(signal: &Array1<f64>, nfft: usize) -> SignalResult<Vec<Complex64>> {
    let signal_vec = signal.to_vec();

    // Perform FFT
    let fft_result = match scirs2_fft::fft(&signal_vec, Some(nfft)) {
        Ok(result) => result,
        Err(_) => {
            return Err(SignalError::ComputationError(
                "Failed to compute FFT".to_string(),
            ))
        }
    };

    Ok(fft_result)
}

/// Compute the power spectrum of a signal
#[allow(dead_code)]
fn compute_power_spectrum(
    signal: &Array1<f64>,
    config: &HigherOrderConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Apply window if specified
    let windowed_signal = if let Some(window_name) = &config.window {
        apply_window(signal, window_name)?
    } else {
        signal.clone()
    };

    // Determine FFT size
    let nfft = config.nfft.unwrap_or_else(|| {
        let mut size = 2usize.pow((signal.len() as f64).log2().ceil() as u32);
        if size < 256 {
            size = 256; // Minimum FFT size
        }
        size
    });

    // Calculate frequency axis
    let _freq_step = config.fs / nfft as f64;
    let max_freq = config.fs / 2.0;
    let freq_bins = (nfft / 2) + 1;

    let frequency_axis = Array1::linspace(0.0, max_freq, freq_bins);

    // Compute FFT
    let fft_result = compute_fft(&windowed_signal, nfft)?;

    // Compute power spectrum (only positive frequencies)
    let mut power_spectrum = Array1::zeros(freq_bins);
    for i in 0..freq_bins {
        power_spectrum[i] = fft_result[i].norm_sqr() / nfft as f64;
    }

    // Scale by 2 for all but DC and Nyquist
    if freq_bins > 2 {
        for i in 1..(freq_bins - 1) {
            power_spectrum[i] *= 2.0;
        }
    }

    Ok((power_spectrum, frequency_axis))
}

/// Compute the triple correlation (third-order cumulant) of a signal
#[allow(dead_code)]
fn compute_triple_correlation(signal: &Array1<f64>, size: usize) -> SignalResult<Array2<f64>> {
    let n = signal.len();

    // Center the _signal
    let mean = signal.sum() / n as f64;
    let centered = signal.mapv(|x| x - mean);

    // Define the maximum lag
    let max_lag = size.min(n / 3);
    if max_lag < 2 {
        return Err(SignalError::ValueError(
            "Signal too short for triple correlation calculation".to_string(),
        ));
    }

    // Initialize the triple correlation matrix
    let mut triple_corr = Array2::zeros((2 * max_lag - 1, 2 * max_lag - 1));

    // Compute the triple correlation
    for i in 0..n {
        for tau1 in 0..max_lag {
            if i + tau1 >= n {
                continue;
            }

            for tau2 in 0..max_lag {
                if i + tau2 >= n {
                    continue;
                }

                // C(τ₁, τ₂) = E[x(t) * x(t+τ₁) * x(t+τ₂)]
                let val = centered[i] * centered[i + tau1] * centered[i + tau2];

                // Store in the matrix (with shifted indices)
                let idx1 = tau1 + max_lag - 1;
                let idx2 = tau2 + max_lag - 1;
                triple_corr[[idx1, idx2]] += val;
            }
        }
    }

    // Normalize
    triple_corr.mapv_inplace(|x| x / n as f64);

    Ok(triple_corr)
}

/// Compute 2D FFT of a matrix
#[allow(dead_code)]
fn compute_2d_fft(matrix: &Array2<f64>, nfft: usize) -> SignalResult<Array2<Complex64>> {
    let (rows, cols) = matrix.dim();

    // Convert to complex for FFT
    let mut complex_matrix = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            complex_matrix[[i, j]] = Complex64::new(_matrix[[i, j]], 0.0);
        }
    }

    // Apply 2D FFT using row-column algorithm
    // First, compute FFT for each row
    let mut row_fft = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = complex_matrix.row(i).to_vec();
        let row_padded = if row.len() < nfft {
            let mut padded = row.clone();
            padded.resize(nfft, Complex64::new(0.0, 0.0));
            padded
        } else {
            row
        };

        match scirs2_fft::fft(&row_padded, None) {
            Ok(fft_result) => row_fft.push(fft_result),
            Err(_) => {
                return Err(SignalError::ComputationError(
                    "Failed to compute row FFT".to_string(),
                ))
            }
        }
    }

    // Then, compute FFT for each column
    let mut result = Array2::zeros((nfft / 2 + 1, nfft / 2 + 1));
    for j in 0..(nfft / 2 + 1) {
        let mut col = Vec::with_capacity(rows);
        for row in row_fft.iter().take(rows) {
            col.push(row[j]);
        }

        let col_padded = if col.len() < nfft {
            let mut padded = col.clone();
            padded.resize(nfft, Complex64::new(0.0, 0.0));
            padded
        } else {
            col
        };

        match scirs2_fft::fft(&col_padded, None) {
            Ok(fft_result) => {
                for i in 0..(nfft / 2 + 1) {
                    result[[i, j]] = fft_result[i];
                }
            }
            Err(_) => {
                return Err(SignalError::ComputationError(
                    "Failed to compute column FFT".to_string(),
                ))
            }
        }
    }

    Ok(result)
}

/// Compute the trispectrum (fourth-order spectrum) of a signal
///
/// The trispectrum detects cubic phase coupling between frequency components.
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name
/// * `fs` - Sampling frequency
///
/// # Returns
/// * A structure containing the trispectrum values (partial implementation)
#[allow(dead_code)]
pub fn trispectrum(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    fs: f64,
) -> SignalResult<Array2<f64>> {
    // Create configuration
    let config = HigherOrderConfig {
        fs,
        nfft: Some(nfft),
        window: window.map(|w| w.to_string()),
        ..HigherOrderConfig::default()
    };

    // For now, just return a 2D slice of the trispectrum
    // A full 4D trispectrum would be very memory-intensive

    // Apply window if specified
    let windowed_signal = if let Some(window_name) = &config.window {
        apply_window(signal, window_name)?
    } else {
        signal.clone()
    };

    // Compute FFT
    let signal_fft = compute_fft(&windowed_signal, nfft)?;

    // Number of frequency bins
    let n_bins = (nfft / 2) + 1;

    // Initialize a 2D slice of the trispectrum (fixed f3=f1, f4=f2)
    let mut trispec_slice = Array2::zeros((n_bins, n_bins));

    // Compute a 2D slice of the trispectrum: T(f1, f2, f1, f2)
    for i in 0..n_bins {
        for j in 0..n_bins {
            // In this slice, we compute T(f1, f2, f1, f2) = E[X(f1)X(f2)X*(f1)X*(f2)]
            let val = (signal_fft[i] * signal_fft[j] * signal_fft[i].conj() * signal_fft[j].conj())
                .norm();
            trispec_slice[[i, j]] = val;
        }
    }

    Ok(trispec_slice)
}

/// Computes the biamplitude of a signal
///
/// The biamplitude is similar to the bispectrum, but measures amplitude coupling
/// rather than phase coupling between frequency components.
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name
/// * `fs` - Sampling frequency
///
/// # Returns
/// * `biamplitude` - 2D array containing the biamplitude values
/// * `frequency_axes` - (f1_axis, f2_axis) frequency axes
#[allow(dead_code)]
pub fn biamplitude(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    fs: f64,
) -> SignalResult<BicoherenceResult> {
    // Create configuration
    let config = HigherOrderConfig {
        fs,
        nfft: Some(nfft),
        window: window.map(|w| w.to_string()),
        ..HigherOrderConfig::default()
    };

    // Apply window if specified
    let windowed_signal = if let Some(window_name) = &config.window {
        apply_window(signal, window_name)?
    } else {
        signal.clone()
    };

    // Compute FFT
    let signal_fft = compute_fft(&windowed_signal, nfft)?;

    // Number of frequency bins
    let n_bins = (nfft / 2) + 1;

    // Create frequency axes
    let _freq_step = fs / nfft as f64;
    let max_freq = fs / 2.0;

    let f1_axis = Array1::linspace(0.0, max_freq, n_bins);
    let f2_axis = f1_axis.clone();

    // Initialize the biamplitude matrix
    let mut biamplitude = Array2::zeros((n_bins, n_bins));

    // Compute biamplitude: measure of amplitude coupling between frequencies
    for i in 0..n_bins {
        for j in 0..n_bins {
            let k = (i + j) % nfft;
            if k < n_bins {
                // BA(f1, f2) = |X(f1)| * |X(f2)| * |X(f1+f2)|
                biamplitude[[i, j]] =
                    signal_fft[i].norm() * signal_fft[j].norm() * signal_fft[k].norm();
            }
        }
    }

    Ok((biamplitude, (f1_axis, f2_axis)))
}

/// Computes the cumulative bispectrum, which integrates the bispectrum over
/// a sequence of bandwidths to measure the overall level of nonlinearity
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name
/// * `fs` - Sampling frequency
///
/// # Returns
/// * `cumulative_bispectrum` - Array containing the cumulative bispectrum values
/// * `bandwidth` - Array of bandwidth values used for integration
#[allow(dead_code)]
pub fn cumulative_bispectrum(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    fs: f64,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Compute bispectrum
    let (bis_mag) = bispectrum(signal, nfft, window, None, fs)?;

    // Number of frequency bins
    let n_bins = bis_mag.dim().0;

    // Define bandwidths for integration
    let n_bandwidths = 10;
    let max_bandwidth = n_bins / 2;
    let bandwidth = Array1::linspace(1.0, max_bandwidth as f64, n_bandwidths);

    // Initialize cumulative bispectrum
    let mut cumulative = Array1::zeros(n_bandwidths);

    // Compute cumulative bispectrum for each bandwidth
    for (i, &bw) in bandwidth.iter().enumerate() {
        let bw_int = bw.round() as usize;
        if bw_int > 0 {
            // Sum over a square region in the bispectrum
            let mut sum = 0.0;
            let mut count = 0;

            for i1 in 0..bw_int.min(n_bins) {
                for i2 in 0..bw_int.min(n_bins) {
                    sum += bis_mag[[i1, i2]];
                    count += 1;
                }
            }

            if count > 0 {
                cumulative[i] = sum / count as f64;
            }
        }
    }

    Ok((cumulative, bandwidth))
}

/// Compute the skewness spectrum of a signal, which is related to the bispectrum
/// and measures the asymmetry of the probability distribution at each frequency
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name
/// * `fs` - Sampling frequency
///
/// # Returns
/// * `skewness_spectrum` - Array containing the skewness spectrum values
/// * `frequency` - Frequency axis
#[allow(dead_code)]
pub fn skewness_spectrum(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    fs: f64,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Create configuration
    let config = HigherOrderConfig {
        fs,
        nfft: Some(nfft),
        window: window.map(|w| w.to_string()),
        ..HigherOrderConfig::default()
    };

    // Compute bispectrum for diagonal slice (f1 = f2)
    let (bis_complex, f1_axis) = compute_bispectrum(signal, &config)?;

    // Compute power spectrum for normalization
    let (power_spectrum) = compute_power_spectrum(signal, &config)?;

    // Number of frequency bins
    let n_bins = (nfft / 2) + 1;

    // Initialize skewness spectrum
    let mut skewness = Array1::zeros(n_bins);

    // Compute skewness spectrum: S(f) = B(f,f) / P(f)^1.5
    for i in 0..n_bins {
        if power_spectrum[i] > 1e-10 {
            skewness[i] = bis_complex[[i, i]].norm() / power_spectrum[i].powf(1.5);
        }
    }

    Ok((skewness, f1_axis))
}

/// Detect quadratic phase coupling in a signal using the bicoherence
///
/// Returns peaks in the bicoherence that exceed a specified threshold,
/// indicating frequencies that show significant phase coupling.
///
/// # Arguments
/// * `signal` - Input signal
/// * `nfft` - FFT size
/// * `window` - Window function name
/// * `fs` - Sampling frequency
/// * `threshold` - Detection threshold (default: 0.5)
///
/// # Returns
/// * Vector of (f1, f2, bicoherence_value) tuples for detected peaks
#[allow(dead_code)]
pub fn detect_phase_coupling(
    signal: &Array1<f64>,
    nfft: usize,
    window: Option<&str>,
    fs: f64,
    threshold: Option<f64>,
) -> SignalResult<Vec<(f64, f64, f64)>> {
    // Set threshold
    let thresh = threshold.unwrap_or(0.5);

    // Compute bicoherence
    let (bicoh, (f1_axis, f2_axis)) = bicoherence(signal, nfft, window, None, fs)?;

    // Find peaks that exceed the threshold
    let mut peaks = Vec::new();
    for i in 1..(bicoh.dim().0 - 1) {
        for j in 1..(bicoh.dim().1 - 1) {
            let val = bicoh[[i, j]];

            // Check if value exceeds threshold
            if val > thresh {
                // Check if it's a local maximum (simple peak detection)
                let neighbors = [
                    bicoh[[i - 1, j]],
                    bicoh[[i + 1, j]],
                    bicoh[[i, j - 1]],
                    bicoh[[i, j + 1]],
                    bicoh[[i - 1, j - 1]],
                    bicoh[[i + 1, j + 1]],
                    bicoh[[i - 1, j + 1]],
                    bicoh[[i + 1, j - 1]],
                ];

                if neighbors.iter().all(|&n| val >= n) {
                    peaks.push((f1_axis[i], f2_axis[j], val));
                }
            }
        }
    }

    // Sort by bicoherence value (descending)
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    Ok(peaks)
}
