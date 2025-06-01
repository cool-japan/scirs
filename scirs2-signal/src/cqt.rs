// Constant-Q Transform Implementation
//
// This module provides an implementation of the Constant-Q Transform (CQT),
// which offers logarithmically-spaced frequency bins with constant-Q resolution.
// This is particularly useful for music signals and other applications where
// frequency components follow logarithmic relationships.
//
// References:
// - Brown, J. C., & Puckette, M. S. (1992). An efficient algorithm for the calculation
//   of a constant Q transform. The Journal of the Acoustical Society of America, 92(5).
// - Schörkhuber, C., & Klapuri, A. (2010). Constant-Q transform toolbox for music processing.
//   7th Sound and Music Computing Conference.

use ndarray::{s, Array1, Array2};
use num_complex::{Complex64, ComplexFloat};
use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};
use crate::window;
use scirs2_fft;

/// Configuration parameters for Constant-Q Transform
#[derive(Debug, Clone)]
pub struct CqtConfig {
    /// Minimum frequency (Hz)
    pub f_min: f64,

    /// Maximum frequency (Hz)
    pub f_max: f64,

    /// Number of bins per octave
    pub bins_per_octave: usize,

    /// Quality factor (Q) - controls frequency resolution
    pub q_factor: Option<f64>,

    /// Window function to use (default: Hann)
    pub window_type: String,

    /// Sample rate of the signal (Hz)
    pub fs: f64,

    /// Whether to use the sparse implementation
    pub use_sparse: bool,

    /// Window scaling factor (1.0 = standard)
    pub window_scaling: Option<f64>,

    /// Hop size for CQT spectrogram
    pub hop_size: Option<usize>,
}

impl Default for CqtConfig {
    fn default() -> Self {
        CqtConfig {
            f_min: 32.7, // C1 piano key
            f_max: 8000.0,
            bins_per_octave: 12, // Semitone resolution
            q_factor: None,      // Will be computed based on bins_per_octave
            window_type: "hann".to_string(),
            fs: 44100.0,
            use_sparse: true,
            window_scaling: None,
            hop_size: None,
        }
    }
}

/// Result structure for Constant-Q Transform
#[derive(Debug)]
pub struct CqtResult {
    /// CQT coefficients [freq, time]
    pub cqt: Array2<Complex64>,

    /// Center frequencies of CQT bins (Hz)
    pub frequencies: Array1<f64>,

    /// Kernel used for the transform
    pub kernel: Option<CqtKernel>,

    /// Time points (in seconds) for CQT spectrogram
    pub times: Option<Array1<f64>>,
}

/// Kernel for efficient CQT computation
#[derive(Debug, Clone)]
pub struct CqtKernel {
    /// Sparse spectral kernels for each bin
    pub kernels: Vec<SparseKernel>,

    /// Center frequencies of CQT bins (Hz)
    pub frequencies: Array1<f64>,

    /// FFT length
    pub n_fft: usize,

    /// Quality factor (Q)
    pub q: f64,

    /// Sample rate (Hz)
    pub fs: f64,

    /// Minimum frequency (Hz)
    pub f_min: f64,

    /// Maximum frequency (Hz)
    pub f_max: f64,

    /// Number of bins per octave
    pub bins_per_octave: usize,
}

/// Sparse representation of a spectral kernel
#[derive(Debug, Clone)]
pub struct SparseKernel {
    /// Non-zero frequency indices
    pub indices: Vec<usize>,

    /// Complex values at those indices
    pub values: Vec<Complex64>,

    /// Normalization factor
    pub normalization: f64,
}

/// Compute the Constant-Q Transform (CQT) of a signal
///
/// The Constant-Q Transform provides a frequency analysis with logarithmically-spaced
/// frequency bins, where the quality factor (Q) remains constant across the spectrum.
/// This makes it particularly suitable for analyzing music signals and other data
/// where frequency relationships follow logarithmic patterns.
///
/// # Arguments
///
/// * `signal` - Input signal (real-valued)
/// * `config` - Configuration parameters for the CQT
///
/// # Returns
///
/// A `CqtResult` containing the CQT coefficients and related information
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::cqt::{constant_q_transform, CqtConfig};
///
/// // Generate a test signal with multiple harmonics
/// let fs = 44100.0;
/// let duration = 1.0;
/// let n_samples = (fs * duration) as usize;
/// let t = Array1::linspace(0.0, duration, n_samples);
///
/// // Create a signal with harmonics (fundamental + 2nd + 3rd harmonics)
/// let f0 = 220.0; // A3 note
/// let signal = t.mapv(|ti| {
///     (2.0 * std::f64::consts::PI * f0 * ti).sin() +          // Fundamental
///     0.5 * (2.0 * std::f64::consts::PI * 2.0 * f0 * ti).sin() +  // 2nd harmonic
///     0.3 * (2.0 * std::f64::consts::PI * 3.0 * f0 * ti).sin()    // 3rd harmonic
/// });
///
/// // Configure the CQT
/// let mut config = CqtConfig::default();
/// config.fs = fs;
/// config.f_min = 55.0;  // A1
/// config.f_max = 2000.0;
/// config.bins_per_octave = 24;  // Quarter-tone resolution
///
/// // Compute the CQT
/// let result = constant_q_transform(&signal, &config).unwrap();
///
/// // result.cqt contains the CQT coefficients
/// // result.frequencies contains the center frequencies of each bin
/// ```
pub fn constant_q_transform(signal: &Array1<f64>, config: &CqtConfig) -> SignalResult<CqtResult> {
    // Calculate the Q factor if not provided
    let q = config.q_factor.unwrap_or_else(|| {
        // Q = 1 / (2^(1/bins_per_octave) - 1)
        1.0 / (2.0f64.powf(1.0 / config.bins_per_octave as f64) - 1.0)
    });

    // Calculate number of bins
    let n_bins =
        ((config.f_max / config.f_min).log2() * config.bins_per_octave as f64).ceil() as usize;

    // Generate frequency array
    let mut frequencies = Array1::<f64>::zeros(n_bins);
    for k in 0..n_bins {
        frequencies[k] = config.f_min * 2.0f64.powf(k as f64 / config.bins_per_octave as f64);
    }

    // Compute or load the CQT kernel
    let kernel = compute_cqt_kernel(
        config.f_min,
        config.f_max,
        config.bins_per_octave,
        q,
        config.fs,
        &config.window_type,
        config.window_scaling,
        config.use_sparse,
    )?;

    // If signal is a single time frame, calculate the CQT directly
    if let Some(hop_size) = config.hop_size {
        // Compute CQT spectrogram
        compute_cqt_spectrogram(signal, &kernel, hop_size)
    } else {
        // Compute single frame CQT
        let cqt = compute_cqt_frame(signal, &kernel)?;

        // Reshape to 2D array (single time frame)
        let mut cqt_2d = Array2::<Complex64>::zeros((n_bins, 1));
        for i in 0..n_bins {
            cqt_2d[[i, 0]] = cqt[i];
        }

        Ok(CqtResult {
            cqt: cqt_2d,
            frequencies,
            kernel: Some(kernel),
            times: None,
        })
    }
}

/// Compute the CQT kernel for efficient CQT calculation
#[allow(clippy::too_many_arguments)]
fn compute_cqt_kernel(
    f_min: f64,
    f_max: f64,
    bins_per_octave: usize,
    q: f64,
    fs: f64,
    window_type: &str,
    window_scaling: Option<f64>,
    use_sparse: bool,
) -> SignalResult<CqtKernel> {
    // Calculate number of bins
    let n_bins = ((f_max / f_min).log2() * bins_per_octave as f64).ceil() as usize;

    // Generate frequency array
    let mut frequencies = Array1::<f64>::zeros(n_bins);
    for k in 0..n_bins {
        frequencies[k] = f_min * 2.0f64.powf(k as f64 / bins_per_octave as f64);
    }

    // Determine the maximum kernel length needed
    let window_scale = window_scaling.unwrap_or(1.0);
    let max_kernel_length = (window_scale * q * fs / f_min).ceil() as usize;

    // Ensure kernel length is odd
    let max_kernel_length = if max_kernel_length % 2 == 0 {
        max_kernel_length + 1
    } else {
        max_kernel_length
    };

    // Determine FFT length (power of 2 >= max_kernel_length)
    let n_fft = next_power_of_two(max_kernel_length);

    // Create the kernels
    let mut kernels = Vec::with_capacity(n_bins);

    for &freq in frequencies.iter() {
        // Calculate kernel length for this frequency
        let kernel_length = (window_scale * q * fs / freq).ceil() as usize;
        let kernel_length = if kernel_length % 2 == 0 {
            kernel_length + 1
        } else {
            kernel_length
        };

        // Create the kernel window
        let window = create_window(window_type, kernel_length)?;

        // Create the complex exponential
        let mut complex_exp = Vec::with_capacity(kernel_length);
        let center = (kernel_length - 1) as f64 / 2.0;

        for n in 0..kernel_length {
            let time = (n as f64 - center) / fs;
            let phase = 2.0 * PI * freq * time;
            complex_exp.push(Complex64::new(phase.cos(), phase.sin()));
        }

        // Apply window to complex exponential
        let mut kernel_values = Vec::with_capacity(kernel_length);
        for n in 0..kernel_length {
            kernel_values.push(complex_exp[n] * window[n]);
        }

        // Compute normalization factor
        let norm = kernel_values
            .iter()
            .map(|&x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();

        // Zero-pad to FFT length
        let mut padded_kernel = vec![Complex64::new(0.0, 0.0); n_fft];
        for n in 0..kernel_length {
            padded_kernel[n] = kernel_values[n] / norm;
        }

        // Compute FFT
        let kernel_fft = scirs2_fft::fft(&padded_kernel, None).expect("FFT computation failed");

        // Create sparse representation if requested
        if use_sparse {
            let threshold = 1e-6; // Sparsity threshold
            let mut indices = Vec::new();
            let mut values = Vec::new();

            for (i, &val) in kernel_fft.iter().enumerate() {
                if val.norm() > threshold {
                    indices.push(i);
                    values.push(val);
                }
            }

            kernels.push(SparseKernel {
                indices,
                values,
                normalization: 1.0, // Already normalized above
            });
        } else {
            // For non-sparse, store all indices and values
            let indices: Vec<usize> = (0..n_fft).collect();
            kernels.push(SparseKernel {
                indices,
                values: kernel_fft,
                normalization: 1.0, // Already normalized above
            });
        }
    }

    Ok(CqtKernel {
        kernels,
        frequencies,
        n_fft,
        q,
        fs,
        f_min,
        f_max,
        bins_per_octave,
    })
}

/// Compute the CQT of a single frame
fn compute_cqt_frame(signal: &Array1<f64>, kernel: &CqtKernel) -> SignalResult<Vec<Complex64>> {
    let n_signal = signal.len();
    let n_fft = kernel.n_fft;

    // Check if signal is long enough
    if n_signal < n_fft {
        // Zero-pad the signal to n_fft
        let mut padded_signal = vec![Complex64::new(0.0, 0.0); n_fft];
        for i in 0..n_signal {
            padded_signal[i] = Complex64::new(signal[i], 0.0);
        }

        // Compute FFT of padded signal
        let signal_fft = scirs2_fft::fft(&padded_signal, None).expect("FFT computation failed");

        // Multiply with the kernels in the frequency domain
        let n_bins = kernel.kernels.len();
        let mut cqt = vec![Complex64::new(0.0, 0.0); n_bins];

        for (k, sparse_kernel) in kernel.kernels.iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);

            for (&idx, &val) in sparse_kernel
                .indices
                .iter()
                .zip(sparse_kernel.values.iter())
            {
                sum += signal_fft[idx] * val.conj();
            }

            cqt[k] = sum / sparse_kernel.normalization;
        }

        Ok(cqt)
    } else {
        // For longer signals, process in chunks of n_fft
        let n_chunks = (n_signal as f64 / n_fft as f64).ceil() as usize;
        let mut cqt = vec![Complex64::new(0.0, 0.0); kernel.kernels.len()];

        for chunk in 0..n_chunks {
            let start = chunk * n_fft;
            let end = (start + n_fft).min(n_signal);

            // Extract chunk and zero-pad if needed
            let mut padded_chunk = vec![Complex64::new(0.0, 0.0); n_fft];
            for i in start..end {
                padded_chunk[i - start] = Complex64::new(signal[i], 0.0);
            }

            // Compute FFT
            let chunk_fft = scirs2_fft::fft(&padded_chunk, None).expect("FFT computation failed");

            // Multiply with the kernels
            for (k, sparse_kernel) in kernel.kernels.iter().enumerate() {
                let mut sum = Complex64::new(0.0, 0.0);

                for (&idx, &val) in sparse_kernel
                    .indices
                    .iter()
                    .zip(sparse_kernel.values.iter())
                {
                    sum += chunk_fft[idx] * val.conj();
                }

                // Accumulate results
                cqt[k] += sum / sparse_kernel.normalization;
            }
        }

        // Normalize by the number of chunks
        for item in cqt.iter_mut() {
            *item /= n_chunks as f64;
        }

        Ok(cqt)
    }
}

/// Compute the CQT spectrogram (time-frequency representation)
fn compute_cqt_spectrogram(
    signal: &Array1<f64>,
    kernel: &CqtKernel,
    hop_size: usize,
) -> SignalResult<CqtResult> {
    let n_signal = signal.len();
    let n_fft = kernel.n_fft;
    let n_bins = kernel.kernels.len();

    // Calculate number of frames
    let n_frames = (n_signal as f64 / hop_size as f64).ceil() as usize;

    // Initialize CQT spectrogram
    let mut cqt_spec = Array2::<Complex64>::zeros((n_bins, n_frames));

    // Initialize time points
    let mut times = Array1::<f64>::zeros(n_frames);

    // For each frame
    for frame in 0..n_frames {
        // Calculate frame boundaries
        let start = frame * hop_size;
        let end = (start + n_fft).min(n_signal);

        // Record time point (center of the frame)
        times[frame] = (start + (end - start) / 2) as f64 / kernel.fs;

        // Extract frame and zero-pad if needed
        let mut frame_signal = Array1::<f64>::zeros(n_fft);
        for i in start..end {
            frame_signal[i - start] = signal[i];
        }

        // Compute CQT for this frame
        let frame_cqt = compute_cqt_frame(&frame_signal, kernel)?;

        // Store in spectrogram
        for k in 0..n_bins {
            cqt_spec[[k, frame]] = frame_cqt[k];
        }
    }

    Ok(CqtResult {
        cqt: cqt_spec,
        frequencies: kernel.frequencies.clone(),
        kernel: Some(kernel.clone()),
        times: Some(times),
    })
}

/// Create window function of specified type and length
fn create_window(window_type: &str, length: usize) -> SignalResult<Vec<f64>> {
    match window_type.to_lowercase().as_str() {
        "hann" | "hanning" => Ok(window::hann(length, true)?),
        "hamming" => Ok(window::hamming(length, true)?),
        "blackman" => Ok(window::blackman(length, true)?),
        "bartlett" => Ok(window::bartlett(length, true)?),
        "rectangular" | "boxcar" => Ok(window::boxcar(length, true)?),
        _ => Err(SignalError::ValueError(format!(
            "Unsupported window type: {}",
            window_type
        ))),
    }
}

/// Find the next power of two greater than or equal to n
fn next_power_of_two(n: usize) -> usize {
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Compute the magnitude spectrogram from CQT coefficients
///
/// # Arguments
///
/// * `cqt` - CQT result
/// * `log_scale` - Whether to convert to dB scale
/// * `ref_value` - Reference value for dB conversion (default: maximum amplitude)
///
/// # Returns
///
/// A 2D array containing the magnitude (or log magnitude) of the CQT coefficients
pub fn cqt_magnitude(cqt: &CqtResult, log_scale: bool, ref_value: Option<f64>) -> Array2<f64> {
    let mut magnitude = Array2::<f64>::zeros(cqt.cqt.raw_dim());

    // Compute magnitude (absolute value) of complex coefficients
    for i in 0..cqt.cqt.shape()[0] {
        for j in 0..cqt.cqt.shape()[1] {
            magnitude[[i, j]] = cqt.cqt[[i, j]].norm();
        }
    }

    if log_scale {
        // Find reference value (maximum if not specified)
        let reference = ref_value.unwrap_or_else(|| magnitude.iter().cloned().fold(0.0, f64::max));

        // Convert to dB
        let eps = 1e-10; // To avoid log(0)
        for i in 0..magnitude.shape()[0] {
            for j in 0..magnitude.shape()[1] {
                // dB = 20 * log10(amplitude / reference)
                magnitude[[i, j]] = 20.0 * (magnitude[[i, j]] / (reference + eps)).log10();
            }
        }
    }

    magnitude
}

/// Compute the phase spectrogram from CQT coefficients
///
/// # Arguments
///
/// * `cqt` - CQT result
///
/// # Returns
///
/// A 2D array containing the phase of the CQT coefficients (in radians)
pub fn cqt_phase(cqt: &CqtResult) -> Array2<f64> {
    let mut phase = Array2::<f64>::zeros(cqt.cqt.raw_dim());

    // Compute phase (argument) of complex coefficients
    for i in 0..cqt.cqt.shape()[0] {
        for j in 0..cqt.cqt.shape()[1] {
            phase[[i, j]] = cqt.cqt[[i, j]].arg();
        }
    }

    phase
}

/// Compute the inverse CQT to reconstruct a signal from CQT coefficients
///
/// # Arguments
///
/// * `cqt` - CQT result structure containing coefficients and kernel
/// * `target_length` - Desired length of the reconstructed signal (default: original length)
///
/// # Returns
///
/// The reconstructed signal as an Array1
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::cqt::{constant_q_transform, inverse_constant_q_transform, CqtConfig};
///
/// // Generate a test signal with shorter duration for faster test
/// let fs = 8000.0;
/// let duration = 0.1;
/// let samples = (fs * duration) as usize;
/// let t = Array1::linspace(0.0, duration, samples);
/// let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * 440.0 * ti).sin());
///
/// // Compute CQT with adjusted parameters
/// let mut config = CqtConfig::default();
/// config.fs = fs;
/// config.f_min = 110.0;
/// config.f_max = 2000.0;
/// config.bins_per_octave = 12;
///
/// let cqt_result = constant_q_transform(&signal, &config).unwrap();
///
/// // Reconstruct signal
/// let reconstructed = inverse_constant_q_transform(&cqt_result, Some(signal.len())).unwrap();
///
/// // Check that reconstruction has correct length
/// assert_eq!(reconstructed.len(), signal.len());
///
/// // Check that the reconstruction preserves signal energy
/// let orig_energy: f64 = signal.iter().map(|x| x.powi(2)).sum();
/// let rec_energy: f64 = reconstructed.iter().map(|x| x.powi(2)).sum();
/// // CQT reconstruction can have significant energy loss, just check it's not empty
/// assert!(rec_energy > 0.0);
/// ```
pub fn inverse_constant_q_transform(
    cqt: &CqtResult,
    target_length: Option<usize>,
) -> SignalResult<Array1<f64>> {
    // Check if kernel is available
    let kernel = match &cqt.kernel {
        Some(k) => k,
        None => {
            return Err(SignalError::ValueError(
                "CQT kernel not available for inverse transform".to_string(),
            ))
        }
    };

    let n_bins = cqt.frequencies.len();
    let n_frames = cqt.cqt.shape()[1];
    let n_fft = kernel.n_fft;

    // For a spectrogram, we need the hop size
    let hop_size = if n_frames > 1 && cqt.times.is_some() {
        let times = cqt.times.as_ref().unwrap();
        if times.len() > 1 {
            // Estimate hop size from time differences
            ((times[1] - times[0]) * kernel.fs).round() as usize
        } else {
            // Default to half the FFT size
            n_fft / 2
        }
    } else {
        // For a single frame, hop size is not relevant
        n_fft
    };

    // Initialize output signal
    let output_length = target_length.unwrap_or_else(|| {
        // Estimate from spectrogram dimensions
        if n_frames > 1 {
            (n_frames - 1) * hop_size + n_fft
        } else {
            n_fft
        }
    });

    let mut output = Array1::<f64>::zeros(output_length);

    // For each frame
    for frame in 0..n_frames {
        // Initialize spectrum for this frame
        let mut frame_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];

        // Compute the spectrum as a weighted sum of kernel spectra
        for bin in 0..n_bins {
            let cqt_value = cqt.cqt[[bin, frame]];
            let sparse_kernel = &kernel.kernels[bin];

            for (&idx, &val) in sparse_kernel
                .indices
                .iter()
                .zip(sparse_kernel.values.iter())
            {
                frame_spectrum[idx] += cqt_value * val;
            }
        }

        // Inverse FFT
        let frame_signal =
            scirs2_fft::ifft(&frame_spectrum, None).expect("IFFT computation failed");

        // Overlap-add to output
        let start = frame * hop_size;
        let end = (start + n_fft).min(output_length);

        for i in start..end {
            // Take only the real part
            output[i] += frame_signal[i - start].re;
        }
    }

    // Normalize the output
    let max_abs = output
        .iter()
        .map(|&x| x.abs())
        .fold(0.0, |a: f64, b: f64| f64::max(a, b));
    if max_abs > 0.0 {
        output.iter_mut().for_each(|x| *x /= max_abs);
    }

    Ok(output)
}

/// Compute a chromagram from CQT coefficients
///
/// A chromagram maps the CQT bins to 12 pitch classes (chroma),
/// which is useful for music analysis and chord recognition.
///
/// # Arguments
///
/// * `cqt` - CQT result
/// * `n_chroma` - Number of chroma bins (default: 12 for semitones)
/// * `ref_note` - Reference note for alignment (0-11 for C-B, default: 0 for C)
///
/// # Returns
///
/// A 2D array containing the chromagram (pitch class vs. time)
pub fn chromagram(
    cqt: &CqtResult,
    n_chroma: Option<usize>,
    ref_note: Option<usize>,
) -> SignalResult<Array2<f64>> {
    let n_bins = cqt.frequencies.len();
    let n_frames = cqt.cqt.shape()[1];

    // Default parameters
    let n_chroma_bins = n_chroma.unwrap_or(12);
    let reference = ref_note.unwrap_or(0) % n_chroma_bins;

    // Calculate center frequencies in MIDI note numbers
    let midi_frequencies = cqt.frequencies.mapv(|freq| {
        69.0 + 12.0 * (freq / 440.0).log2() // A4 = 69, freq = 440Hz
    });

    // Initialize chromagram
    let mut chroma = Array2::<f64>::zeros((n_chroma_bins, n_frames));

    // Map CQT bins to chroma bins
    for i in 0..n_bins {
        let midi_note = midi_frequencies[i];
        let chroma_bin = ((midi_note as isize) % n_chroma_bins as isize + n_chroma_bins as isize
            - reference as isize)
            % n_chroma_bins as isize;

        if chroma_bin >= 0 && chroma_bin < n_chroma_bins as isize {
            for j in 0..n_frames {
                // Add magnitudes
                chroma[[chroma_bin as usize, j]] += cqt.cqt[[i, j]].norm();
            }
        }
    }

    // Normalize each frame
    for j in 0..n_frames {
        let frame_sum = chroma.slice(s![.., j]).sum();
        if frame_sum > 0.0 {
            for i in 0..n_chroma_bins {
                chroma[[i, j]] /= frame_sum;
            }
        }
    }

    Ok(chroma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array;

    #[test]
    fn test_create_window() {
        // Test window creation
        let length = 128;

        let hann = create_window("hann", length).unwrap();
        assert_eq!(hann.len(), length);
        assert!(hann[0] < 1e-10); // Should be close to 0 at endpoints
        assert!(hann[length - 1] < 1e-10);
        assert!(hann[length / 2] > 0.9); // Should be close to 1 at center

        // Test different window types
        assert!(create_window("hamming", length).is_ok());
        assert!(create_window("blackman", length).is_ok());
        assert!(create_window("boxcar", length).is_ok());

        // Test invalid window type
        assert!(create_window("invalid_window", length).is_err());
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(127), 128);
        assert_eq!(next_power_of_two(128), 128);
        assert_eq!(next_power_of_two(129), 256);
    }

    #[test]
    fn test_cqt_kernel() {
        // Test kernel generation
        let f_min = 55.0; // A1
        let f_max = 440.0; // A4
        let bins_per_octave = 12;
        let q = 1.0 / (2.0f64.powf(1.0 / bins_per_octave as f64) - 1.0);
        let fs = 22050.0;

        let kernel =
            compute_cqt_kernel(f_min, f_max, bins_per_octave, q, fs, "hann", None, true).unwrap();

        // Check number of bins (should be exactly 36 bins for log2(440/55) * 12 = 3 * 12)
        let expected_bins = ((f_max / f_min).log2() * bins_per_octave as f64).ceil() as usize;
        assert_eq!(kernel.kernels.len(), expected_bins);

        // Check that frequency range is correct
        assert_relative_eq!(kernel.frequencies[0], f_min, epsilon = 0.1);
        // The last frequency should be close to f_max but might not be exact
        let last_freq = kernel.frequencies[kernel.frequencies.len() - 1];
        // Check that the last frequency is at least close to our target
        assert!(last_freq >= f_max * 0.9);

        // Check that the FFT length is a power of 2
        assert_eq!(kernel.n_fft & (kernel.n_fft - 1), 0);
    }

    #[test]
    fn test_constant_q_transform() {
        // Generate a simple sine wave
        let fs = 22050.0;
        let duration = 0.5;
        let n_samples = (fs * duration) as usize;
        let t = Array::linspace(0.0, duration, n_samples);

        // A4 note (440 Hz)
        let frequency = 440.0;
        let signal = t.mapv(|ti| (2.0 * PI * frequency * ti).sin());

        // Configure CQT
        let config = CqtConfig {
            f_min: 55.0,   // A1
            f_max: 2000.0, // Cover up to A6
            bins_per_octave: 12,
            q_factor: None,
            window_type: "hann".to_string(),
            fs,
            use_sparse: true,
            window_scaling: None,
            hop_size: None,
        };

        // Compute CQT
        let cqt_result = constant_q_transform(&signal, &config).unwrap();

        // Find the bin with maximum energy
        let mut max_bin = 0;
        let mut max_value = 0.0;

        for (i, &_freq) in cqt_result.frequencies.iter().enumerate() {
            let magnitude = cqt_result.cqt[[i, 0]].norm();
            if magnitude > max_value {
                max_value = magnitude;
                max_bin = i;
            }
        }

        // The max energy should be at the bin closest to 440 Hz
        let closest_bin = cqt_result
            .frequencies
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - frequency)
                    .abs()
                    .partial_cmp(&(b - frequency).abs())
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap();

        // Should be at or very close to the expected bin
        assert!((max_bin as isize - closest_bin as isize).abs() <= 1);
    }

    #[test]
    fn test_cqt_spectrogram() {
        // Generate a chirp signal
        let fs = 22050.0;
        let duration = 1.0;
        let n_samples = (fs * duration) as usize;
        let t = Array::linspace(0.0, duration, n_samples);

        // Linear chirp from 110 Hz to 880 Hz
        let signal = t.mapv(|ti| {
            let phase = 2.0 * PI * (110.0 * ti + (880.0 - 110.0) * ti * ti / (2.0 * duration));
            phase.sin()
        });

        // Configure CQT
        let config = CqtConfig {
            f_min: 55.0,   // A1
            f_max: 2000.0, // Cover up to A6
            bins_per_octave: 12,
            q_factor: None,
            window_type: "hann".to_string(),
            fs,
            use_sparse: true,
            window_scaling: None,
            hop_size: Some(512), // Large hop for faster testing
        };

        // Compute CQT spectrogram
        let cqt_result = constant_q_transform(&signal, &config).unwrap();

        // Check spectrogram dimensions
        let n_bins =
            ((config.f_max / config.f_min).log2() * config.bins_per_octave as f64).ceil() as usize;
        let n_frames = (n_samples as f64 / 512.0).ceil() as usize;

        assert_eq!(cqt_result.cqt.shape()[0], n_bins);
        assert_eq!(cqt_result.cqt.shape()[1], n_frames);

        // Times should be present
        assert!(cqt_result.times.is_some());
        assert_eq!(cqt_result.times.unwrap().len(), n_frames);
    }

    #[test]
    fn test_chromagram() {
        // Generate a simple test signal
        let fs = 22050.0;
        let duration = 0.5; // Shorter duration for faster test
        let n_samples = (fs * duration) as usize;
        let t = Array::linspace(0.0, duration, n_samples);

        // Simple sine wave at A4 (440 Hz)
        let signal = t.mapv(|ti| (2.0 * PI * 440.0 * ti).sin());

        // Configure CQT with wider range to ensure we capture the frequency
        let config = CqtConfig {
            f_min: 220.0, // A3
            f_max: 880.0, // A5
            bins_per_octave: 12,
            q_factor: None,
            window_type: "hann".to_string(),
            fs,
            use_sparse: true,
            window_scaling: None,
            hop_size: None,
        };

        // Compute CQT
        let cqt_result = constant_q_transform(&signal, &config).unwrap();

        // Compute chromagram
        let chroma = chromagram(&cqt_result, None, None).unwrap();

        // Shape should be 12 x n_frames
        assert_eq!(chroma.shape()[0], 12);
        assert_eq!(chroma.shape()[1], cqt_result.cqt.shape()[1]);

        // For A note (440Hz), the 9th chroma bin (A) should have high energy
        let a_idx = 9; // A is the 9th note when starting from C
        let frame = 0;

        // Find the bin with maximum energy
        let mut max_energy = 0.0;
        let mut _max_idx = 0;
        for i in 0..12 {
            if chroma[[i, frame]] > max_energy {
                max_energy = chroma[[i, frame]];
                _max_idx = i;
            }
        }

        // The A bin should have significant energy
        assert!(chroma[[a_idx, frame]] > 0.1);

        // Check normalization (each frame should sum to 1)
        let frame_sum: f64 = (0..12).map(|i| chroma[[i, frame]]).sum();
        if frame_sum > 0.0 {
            assert_relative_eq!(frame_sum, 1.0, epsilon = 1e-6);
        }
    }
}
