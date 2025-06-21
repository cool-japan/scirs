//! Parallel spectral analysis algorithms using Rayon for high-performance processing
//!
//! This module provides parallel implementations of spectral analysis algorithms
//! that can significantly improve performance on multi-core systems when processing
//! large datasets or multiple signals simultaneously.

use crate::error::{SignalError, SignalResult};
use crate::window;
use ndarray::Array2;

type SpectrogramResult = (Vec<f64>, Vec<f64>, Array2<f64>);
type TimeFrequencyCoherenceResult = (Vec<f64>, Vec<f64>, Array2<f64>);
use num_complex::Complex64;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for parallel spectral processing
#[derive(Debug, Clone)]
pub struct ParallelSpectralConfig {
    /// Number of threads to use (None for automatic)
    pub num_threads: Option<usize>,
    /// Minimum chunk size for parallel processing
    pub min_chunk_size: usize,
    /// Enable SIMD optimizations where available
    pub enable_simd: bool,
    /// Memory usage optimization level (1-3)
    pub memory_optimization: u8,
}

impl Default for ParallelSpectralConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            min_chunk_size: 1000,
            enable_simd: true,
            memory_optimization: 2,
        }
    }
}

/// Parallel batch spectral analysis processor
pub struct ParallelSpectralProcessor {
    #[allow(dead_code)]
    config: ParallelSpectralConfig,
    fft_planner: Arc<std::sync::Mutex<FftPlanner<f64>>>,
}

impl ParallelSpectralProcessor {
    /// Create a new parallel spectral processor
    pub fn new(config: ParallelSpectralConfig) -> Self {
        // Configure Rayon thread pool if specified
        #[cfg(feature = "parallel")]
        if let Some(num_threads) = config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap_or_else(|_| {
                    eprintln!("Warning: Could not set thread count to {}", num_threads);
                });
        }

        #[cfg(not(feature = "parallel"))]
        if config.num_threads.is_some() {
            eprintln!("Warning: Parallel feature not enabled, ignoring thread count configuration");
        }

        Self {
            config,
            fft_planner: Arc::new(std::sync::Mutex::new(FftPlanner::new())),
        }
    }

    /// Parallel batch periodogram computation
    ///
    /// # Arguments
    ///
    /// * `signals` - Multiple signals to process
    /// * `fs` - Sampling frequency
    /// * `window_type` - Window function type
    /// * `nfft` - FFT size (None for automatic)
    ///
    /// # Returns
    ///
    /// * Vector of (frequencies, power_spectrum) pairs for each signal
    pub fn batch_periodogram(
        &self,
        signals: &[&[f64]],
        fs: f64,
        window_type: Option<&str>,
        nfft: Option<usize>,
    ) -> SignalResult<Vec<(Vec<f64>, Vec<f64>)>> {
        if signals.is_empty() {
            return Err(SignalError::ValueError("No signals provided".to_string()));
        }

        // Process signals in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|&signal| self.single_periodogram(signal, fs, window_type, nfft))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signals
            .iter()
            .map(|&signal| self.single_periodogram(signal, fs, window_type, nfft))
            .collect();

        results
    }

    /// Parallel batch spectrogram computation
    ///
    /// # Arguments
    ///
    /// * `signals` - Multiple signals to process
    /// * `fs` - Sampling frequency
    /// * `window_size` - Window size for STFT
    /// * `hop_size` - Hop size for STFT
    /// * `window_type` - Window function type
    ///
    /// # Returns
    ///
    /// * Vector of (frequencies, times, spectrogram) tuples for each signal
    pub fn batch_spectrogram(
        &self,
        signals: &[&[f64]],
        fs: f64,
        window_size: usize,
        hop_size: usize,
        window_type: Option<&str>,
    ) -> SignalResult<Vec<SpectrogramResult>> {
        if signals.is_empty() {
            return Err(SignalError::ValueError("No signals provided".to_string()));
        }

        // Process signals in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|&signal| self.single_spectrogram(signal, fs, window_size, hop_size, window_type))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signals
            .iter()
            .map(|&signal| self.single_spectrogram(signal, fs, window_size, hop_size, window_type))
            .collect();

        results
    }

    /// Parallel cross-spectral density matrix computation
    ///
    /// # Arguments
    ///
    /// * `signals` - Matrix of signals (each row is a signal)
    /// * `fs` - Sampling frequency
    /// * `window_size` - Window size
    /// * `overlap` - Overlap ratio (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// * Cross-spectral density matrix and frequency vector
    pub fn cross_spectral_density_matrix(
        &self,
        signals: &Array2<f64>,
        fs: f64,
        window_size: usize,
        overlap: f64,
    ) -> SignalResult<(Vec<f64>, Array2<Complex64>)> {
        let (n_signals, n_samples) = signals.dim();

        if n_signals == 0 || n_samples == 0 {
            return Err(SignalError::ValueError("Empty signal matrix".to_string()));
        }

        let hop_size = ((1.0 - overlap) * window_size as f64) as usize;
        let n_freq_bins = window_size / 2 + 1;

        // Create frequency vector
        let frequencies: Vec<f64> = (0..n_freq_bins)
            .map(|i| i as f64 * fs / window_size as f64)
            .collect();

        // Compute all signal spectrograms in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let spectrograms: Result<Vec<_>, SignalError> = (0..n_signals)
            .into_par_iter()
            .map(|i| {
                let signal = signals.row(i).to_vec();
                self.single_stft(&signal, window_size, hop_size)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let spectrograms: Result<Vec<_>, SignalError> = (0..n_signals)
            .map(|i| {
                let signal = signals.row(i).to_vec();
                self.single_stft(&signal, window_size, hop_size)
            })
            .collect();

        let spectrograms = spectrograms?;
        let n_time_frames = spectrograms[0].shape()[1];

        // Compute cross-spectral density matrix for each frequency bin
        let mut csd_matrix = Array2::zeros((n_freq_bins, n_signals * n_signals));

        for freq_bin in 0..n_freq_bins {
            #[cfg(feature = "parallel")]
            let cross_spectra: Vec<Complex64> = (0..n_signals)
                .into_par_iter()
                .map(|i| {
                    (0..n_signals)
                        .map(|j| {
                            // Compute cross-spectral density for frequency bin
                            let mut csd = Complex64::new(0.0, 0.0);
                            for t in 0..n_time_frames {
                                let x_i = spectrograms[i][[freq_bin, t]];
                                let x_j = spectrograms[j][[freq_bin, t]];
                                csd += x_i * x_j.conj();
                            }
                            csd / n_time_frames as f64
                        })
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect();

            #[cfg(not(feature = "parallel"))]
            let cross_spectra: Vec<Complex64> = (0..n_signals)
                .flat_map(|i| {
                    (0..n_signals)
                        .map(|j| {
                            // Compute cross-spectral density for frequency bin
                            let mut csd = Complex64::new(0.0, 0.0);
                            for t in 0..n_time_frames {
                                let x_i = spectrograms[i][[freq_bin, t]];
                                let x_j = spectrograms[j][[freq_bin, t]];
                                csd += x_i * x_j.conj();
                            }
                            csd / n_time_frames as f64
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            for (idx, &value) in cross_spectra.iter().enumerate() {
                csd_matrix[[freq_bin, idx]] = value;
            }
        }

        // Reshape to proper CSD matrix format
        let mut result_matrix = Array2::zeros((n_freq_bins, n_signals * n_signals));
        for freq in 0..n_freq_bins {
            for i in 0..n_signals {
                for j in 0..n_signals {
                    result_matrix[[freq, i * n_signals + j]] =
                        csd_matrix[[freq, i * n_signals + j]];
                }
            }
        }

        Ok((frequencies, result_matrix))
    }

    /// Parallel coherence estimation between multiple signal pairs
    ///
    /// # Arguments
    ///
    /// * `signal_pairs` - Vector of signal pairs to analyze
    /// * `fs` - Sampling frequency
    /// * `window_size` - Window size for analysis
    /// * `overlap` - Overlap ratio
    ///
    /// # Returns
    ///
    /// * Vector of (frequencies, coherence) pairs for each signal pair
    pub fn batch_coherence(
        &self,
        signal_pairs: &[(&[f64], &[f64])],
        fs: f64,
        window_size: usize,
        overlap: f64,
    ) -> SignalResult<Vec<(Vec<f64>, Vec<f64>)>> {
        if signal_pairs.is_empty() {
            return Err(SignalError::ValueError(
                "No signal pairs provided".to_string(),
            ));
        }

        // Process signal pairs in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signal_pairs
            .par_iter()
            .map(|&(signal1, signal2)| {
                self.single_coherence(signal1, signal2, fs, window_size, overlap)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signal_pairs
            .iter()
            .map(|&(signal1, signal2)| {
                self.single_coherence(signal1, signal2, fs, window_size, overlap)
            })
            .collect();

        results
    }

    /// Parallel multi-taper spectral estimation
    ///
    /// # Arguments
    ///
    /// * `signals` - Multiple signals to process
    /// * `nw` - Time-bandwidth product
    /// * `k` - Number of tapers
    /// * `fs` - Sampling frequency
    ///
    /// # Returns
    ///
    /// * Vector of (frequencies, power_spectrum) pairs for each signal
    pub fn batch_multitaper_psd(
        &self,
        signals: &[&[f64]],
        nw: f64,
        k: usize,
        fs: f64,
    ) -> SignalResult<Vec<(Vec<f64>, Vec<f64>)>> {
        if signals.is_empty() {
            return Err(SignalError::ValueError("No signals provided".to_string()));
        }

        // Generate DPSS tapers once (shared across all signals)
        let n = signals[0].len();
        let tapers = self.generate_dpss_tapers(n, nw, k)?;

        // Process signals in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|&signal| self.single_multitaper_psd(signal, &tapers, fs))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signals
            .iter()
            .map(|&signal| self.single_multitaper_psd(signal, &tapers, fs))
            .collect();

        results
    }

    // Private helper methods

    fn single_periodogram(
        &self,
        signal: &[f64],
        fs: f64,
        window_type: Option<&str>,
        nfft: Option<usize>,
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        let n = signal.len();
        let nfft_actual = nfft.unwrap_or(n);

        // Apply window
        let windowed_signal = if let Some(window_type) = window_type {
            let window = window::get_window(window_type, n, false)?;
            signal
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect()
        } else {
            signal.to_vec()
        };

        // Zero-pad if necessary
        let mut fft_input: Vec<Complex<f64>> = windowed_signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        fft_input.resize(nfft_actual, Complex::new(0.0, 0.0));

        // Compute FFT
        let mut planner = self.fft_planner.lock().unwrap();
        let fft = planner.plan_fft_forward(nfft_actual);
        drop(planner);

        fft.process(&mut fft_input);

        // Compute power spectral density
        let n_freq_bins = nfft_actual / 2 + 1;
        let mut psd = vec![0.0; n_freq_bins];
        let normalization = fs * n as f64;

        for i in 0..n_freq_bins {
            let magnitude_sq = fft_input[i].norm_sqr();
            psd[i] = if i == 0 || (i == nfft_actual / 2 && nfft_actual % 2 == 0) {
                magnitude_sq / normalization
            } else {
                2.0 * magnitude_sq / normalization
            };
        }

        // Create frequency vector
        let frequencies: Vec<f64> = (0..n_freq_bins)
            .map(|i| i as f64 * fs / nfft_actual as f64)
            .collect();

        Ok((frequencies, psd))
    }

    fn single_spectrogram(
        &self,
        signal: &[f64],
        fs: f64,
        window_size: usize,
        hop_size: usize,
        _window_type: Option<&str>,
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)> {
        let stft_result = self.single_stft(signal, window_size, hop_size)?;

        // Convert complex STFT to magnitude spectrogram
        let spectrogram = stft_result.mapv(|c| c.norm());

        // Create frequency and time vectors
        let n_freq_bins = stft_result.shape()[0];
        let n_time_frames = stft_result.shape()[1];

        let frequencies: Vec<f64> = (0..n_freq_bins)
            .map(|i| i as f64 * fs / (2 * (n_freq_bins - 1)) as f64)
            .collect();

        let times: Vec<f64> = (0..n_time_frames)
            .map(|i| i as f64 * hop_size as f64 / fs)
            .collect();

        Ok((frequencies, times, spectrogram))
    }

    fn single_stft(
        &self,
        signal: &[f64],
        window_size: usize,
        hop_size: usize,
    ) -> SignalResult<Array2<Complex64>> {
        let n_samples = signal.len();
        if n_samples < window_size {
            return Err(SignalError::ValueError(
                "Signal length must be at least window size".to_string(),
            ));
        }

        // Calculate number of frames
        let n_frames = (n_samples - window_size) / hop_size + 1;
        let n_freq_bins = window_size / 2 + 1;

        // Create result array
        let mut stft_result = Array2::zeros((n_freq_bins, n_frames));

        // Create window
        let window = window::hann(window_size, true)?;

        // Process frames in parallel chunks (if feature enabled) or sequentially
        let frames: Vec<usize> = (0..n_frames).collect();

        #[cfg(feature = "parallel")]
        let chunk_size = (n_frames / rayon::current_num_threads()).max(1);
        #[cfg(not(feature = "parallel"))]
        let chunk_size = n_frames;

        #[cfg(feature = "parallel")]
        let results: Vec<_> = frames
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_results = Vec::new();
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(window_size);

                for &frame_idx in chunk {
                    let start = frame_idx * hop_size;
                    let end = start + window_size;

                    if end <= n_samples {
                        // Apply window and prepare for FFT
                        let mut fft_input: Vec<Complex<f64>> = signal[start..end]
                            .iter()
                            .zip(window.iter())
                            .map(|(&s, &w)| Complex::new(s * w, 0.0))
                            .collect();

                        // Compute FFT
                        fft.process(&mut fft_input);

                        // Extract positive frequencies
                        let frame_spectrum: Vec<Complex64> = fft_input[..n_freq_bins]
                            .iter()
                            .map(|&c| Complex64::new(c.re, c.im))
                            .collect();

                        local_results.push((frame_idx, frame_spectrum));
                    }
                }
                local_results
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Vec<_> = frames
            .chunks(chunk_size)
            .map(|chunk| {
                let mut local_results = Vec::new();
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(window_size);

                for &frame_idx in chunk {
                    let start = frame_idx * hop_size;
                    let end = start + window_size;

                    if end <= n_samples {
                        // Apply window and prepare for FFT
                        let mut fft_input: Vec<Complex<f64>> = signal[start..end]
                            .iter()
                            .zip(window.iter())
                            .map(|(&s, &w)| Complex::new(s * w, 0.0))
                            .collect();

                        // Compute FFT
                        fft.process(&mut fft_input);

                        // Extract positive frequencies
                        let frame_spectrum: Vec<Complex64> = fft_input[..n_freq_bins]
                            .iter()
                            .map(|&c| Complex64::new(c.re, c.im))
                            .collect();

                        local_results.push((frame_idx, frame_spectrum));
                    }
                }
                local_results
            })
            .collect();

        // Collect results back into the main array
        for chunk_results in results {
            for (frame_idx, spectrum) in chunk_results {
                for (freq_idx, &value) in spectrum.iter().enumerate() {
                    stft_result[[freq_idx, frame_idx]] = value;
                }
            }
        }

        Ok(stft_result)
    }

    fn single_coherence(
        &self,
        signal1: &[f64],
        signal2: &[f64],
        fs: f64,
        window_size: usize,
        overlap: f64,
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        if signal1.len() != signal2.len() {
            return Err(SignalError::ValueError(
                "Signals must have the same length".to_string(),
            ));
        }

        let hop_size = ((1.0 - overlap) * window_size as f64) as usize;

        // Compute cross-spectral densities
        let stft1 = self.single_stft(signal1, window_size, hop_size)?;
        let stft2 = self.single_stft(signal2, window_size, hop_size)?;

        let n_freq_bins = stft1.shape()[0];
        let n_frames = stft1.shape()[1];

        // Compute coherence for each frequency bin in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let coherence: Vec<f64> = (0..n_freq_bins)
            .into_par_iter()
            .map(|freq_bin| {
                let mut pxx = 0.0;
                let mut pyy = 0.0;
                let mut pxy = Complex64::new(0.0, 0.0);

                for frame in 0..n_frames {
                    let x = stft1[[freq_bin, frame]];
                    let y = stft2[[freq_bin, frame]];

                    pxx += x.norm_sqr();
                    pyy += y.norm_sqr();
                    pxy += x * y.conj();
                }

                if pxx > 0.0 && pyy > 0.0 {
                    (pxy.norm_sqr()) / (pxx * pyy)
                } else {
                    0.0
                }
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let coherence: Vec<f64> = (0..n_freq_bins)
            .map(|freq_bin| {
                let mut pxx = 0.0;
                let mut pyy = 0.0;
                let mut pxy = Complex64::new(0.0, 0.0);

                for frame in 0..n_frames {
                    let x = stft1[[freq_bin, frame]];
                    let y = stft2[[freq_bin, frame]];

                    pxx += x.norm_sqr();
                    pyy += y.norm_sqr();
                    pxy += x * y.conj();
                }

                if pxx > 0.0 && pyy > 0.0 {
                    (pxy.norm_sqr()) / (pxx * pyy)
                } else {
                    0.0
                }
            })
            .collect();

        // Create frequency vector
        let frequencies: Vec<f64> = (0..n_freq_bins)
            .map(|i| i as f64 * fs / (2 * (n_freq_bins - 1)) as f64)
            .collect();

        Ok((frequencies, coherence))
    }

    fn generate_dpss_tapers(&self, n: usize, nw: f64, k: usize) -> SignalResult<Vec<Vec<f64>>> {
        // Simplified DPSS taper generation
        // In a full implementation, this would solve the eigenvalue problem
        // for the Slepian sequences

        let mut tapers = Vec::new();

        for taper_idx in 0..k {
            let mut taper = vec![0.0; n];
            let beta = 2.0 * std::f64::consts::PI * nw * taper_idx as f64 / n as f64;

            for (i, tap) in taper.iter_mut().enumerate().take(n) {
                let t = (i as f64 - n as f64 / 2.0) / n as f64;
                let w = 2.0 * nw / n as f64;

                // Simplified taper calculation (approximation)
                *tap = if (beta * t).abs() < 1e-10 {
                    1.0
                } else {
                    (w * std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
                } * (1.0 - 2.0 * taper_idx as f64 / k as f64).max(0.0);
            }

            // Normalize taper
            let norm: f64 = taper.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for val in &mut taper {
                    *val /= norm;
                }
            }

            tapers.push(taper);
        }

        Ok(tapers)
    }

    fn single_multitaper_psd(
        &self,
        signal: &[f64],
        tapers: &[Vec<f64>],
        fs: f64,
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        let n = signal.len();
        let n_freq_bins = n / 2 + 1;

        // Compute periodogram for each taper in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let periodograms: Result<Vec<_>, SignalError> = tapers
            .par_iter()
            .map(|taper| {
                // Apply taper
                let tapered_signal: Vec<f64> = signal
                    .iter()
                    .zip(taper.iter())
                    .map(|(&s, &t)| s * t)
                    .collect();

                // Compute periodogram
                self.single_periodogram(&tapered_signal, fs, None, Some(n))
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let periodograms: Result<Vec<_>, SignalError> = tapers
            .iter()
            .map(|taper| {
                // Apply taper
                let tapered_signal: Vec<f64> = signal
                    .iter()
                    .zip(taper.iter())
                    .map(|(&s, &t)| s * t)
                    .collect();

                // Compute periodogram
                self.single_periodogram(&tapered_signal, fs, None, Some(n))
            })
            .collect();

        let periodograms = periodograms?;

        // Average across tapers
        let mut avg_psd = vec![0.0; n_freq_bins];
        for (_, psd) in &periodograms {
            for (i, &value) in psd.iter().enumerate() {
                avg_psd[i] += value / tapers.len() as f64;
            }
        }

        // Use frequency vector from first periodogram
        let frequencies = periodograms[0].0.clone();

        Ok((frequencies, avg_psd))
    }
}

/// Parallel welch periodogram estimation
///
/// # Arguments
///
/// * `signals` - Multiple signals to process
/// * `fs` - Sampling frequency
/// * `window_size` - Window size for segments
/// * `overlap` - Overlap ratio (0.0 to 1.0)
/// * `window_type` - Window function type
///
/// # Returns
///
/// * Vector of (frequencies, power_spectrum) pairs for each signal
pub fn parallel_welch(
    signals: &[&[f64]],
    fs: f64,
    window_size: usize,
    overlap: f64,
    window_type: Option<&str>,
) -> SignalResult<Vec<(Vec<f64>, Vec<f64>)>> {
    let _processor = ParallelSpectralProcessor::new(ParallelSpectralConfig::default());

    // Process each signal in parallel (if feature enabled) or sequentially
    #[cfg(feature = "parallel")]
    let results: Result<Vec<_>, SignalError> = signals
        .par_iter()
        .map(|&signal| single_welch(signal, fs, window_size, overlap, window_type))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let results: Result<Vec<_>, SignalError> = signals
        .iter()
        .map(|&signal| single_welch(signal, fs, window_size, overlap, window_type))
        .collect();

    results
}

/// Single signal Welch periodogram (helper function)
fn single_welch(
    signal: &[f64],
    fs: f64,
    window_size: usize,
    overlap: f64,
    window_type: Option<&str>,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    if n < window_size {
        return Err(SignalError::ValueError(
            "Signal length must be at least window size".to_string(),
        ));
    }

    let hop_size = ((1.0 - overlap) * window_size as f64) as usize;
    let n_segments = (n - window_size) / hop_size + 1;

    if n_segments == 0 {
        return Err(SignalError::ValueError(
            "No valid segments found".to_string(),
        ));
    }

    // Create window
    let window = if let Some(window_type) = window_type {
        window::get_window(window_type, window_size, false)?
    } else {
        vec![1.0; window_size]
    };

    let n_freq_bins = window_size / 2 + 1;
    let mut avg_psd = vec![0.0; n_freq_bins];

    // Process segments
    for segment_idx in 0..n_segments {
        let start = segment_idx * hop_size;
        let end = start + window_size;

        if end <= n {
            // Apply window
            let windowed_segment: Vec<f64> = signal[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute FFT
            let mut fft_input: Vec<Complex<f64>> = windowed_segment
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();

            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(window_size);
            fft.process(&mut fft_input);

            // Add to average PSD
            let normalization = fs * window.iter().map(|&w| w * w).sum::<f64>();
            for i in 0..n_freq_bins {
                let magnitude_sq = fft_input[i].norm_sqr();
                avg_psd[i] += if i == 0 || (i == window_size / 2 && window_size % 2 == 0) {
                    magnitude_sq / normalization
                } else {
                    2.0 * magnitude_sq / normalization
                };
            }
        }
    }

    // Average across segments
    for psd_val in &mut avg_psd {
        *psd_val /= n_segments as f64;
    }

    // Create frequency vector
    let frequencies: Vec<f64> = (0..n_freq_bins)
        .map(|i| i as f64 * fs / window_size as f64)
        .collect();

    Ok((frequencies, avg_psd))
}

/// Advanced parallel spectral analysis algorithms
impl ParallelSpectralProcessor {
    /// Parallel batch cross-power spectral density
    ///
    /// Computes cross-PSD between pairs of signals in parallel
    ///
    /// # Arguments
    ///
    /// * `signal_pairs` - Vector of signal pairs for cross-PSD computation
    /// * `fs` - Sampling frequency
    /// * `window_size` - Window size for Welch method
    /// * `overlap` - Overlap ratio (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// * Vector of (frequencies, cross_psd) pairs
    pub fn batch_cross_psd(
        &self,
        signal_pairs: &[(&[f64], &[f64])],
        fs: f64,
        window_size: usize,
        overlap: f64,
    ) -> SignalResult<Vec<(Vec<f64>, Vec<Complex64>)>> {
        if signal_pairs.is_empty() {
            return Err(SignalError::ValueError(
                "No signal pairs provided".to_string(),
            ));
        }

        // Process signal pairs in parallel
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signal_pairs
            .par_iter()
            .map(|(sig1, sig2)| self.single_cross_psd(sig1, sig2, fs, window_size, overlap))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signal_pairs
            .iter()
            .map(|(sig1, sig2)| self.single_cross_psd(sig1, sig2, fs, window_size, overlap))
            .collect();

        results
    }

    /// Parallel batch phase-amplitude coupling analysis
    ///
    /// Computes phase-amplitude coupling between signals using parallel processing
    ///
    /// # Arguments
    ///
    /// * `signals` - Vector of signals to analyze
    /// * `phase_band` - Frequency band for phase (low, high)
    /// * `amplitude_band` - Frequency band for amplitude (low, high)
    /// * `fs` - Sampling frequency
    ///
    /// # Returns
    ///
    /// * Vector of coupling strength values
    pub fn batch_phase_amplitude_coupling(
        &self,
        signals: &[&[f64]],
        phase_band: (f64, f64),
        amplitude_band: (f64, f64),
        fs: f64,
    ) -> SignalResult<Vec<f64>> {
        if signals.is_empty() {
            return Err(SignalError::ValueError("No signals provided".to_string()));
        }

        // Process signals in parallel
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|&signal| self.single_pac_analysis(signal, phase_band, amplitude_band, fs))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signals
            .iter()
            .map(|&signal| self.single_pac_analysis(signal, phase_band, amplitude_band, fs))
            .collect();

        results
    }

    /// Parallel time-frequency coherence analysis
    ///
    /// Computes time-frequency coherence between signal pairs using parallel processing
    ///
    /// # Arguments
    ///
    /// * `signal_pairs` - Vector of signal pairs
    /// * `fs` - Sampling frequency
    /// * `window_size` - Window size for STFT
    /// * `hop_size` - Hop size for STFT
    ///
    /// # Returns
    ///
    /// * Vector of (frequencies, times, coherence_matrix) tuples
    pub fn batch_time_frequency_coherence(
        &self,
        signal_pairs: &[(&[f64], &[f64])],
        fs: f64,
        window_size: usize,
        hop_size: usize,
    ) -> SignalResult<Vec<TimeFrequencyCoherenceResult>> {
        if signal_pairs.is_empty() {
            return Err(SignalError::ValueError(
                "No signal pairs provided".to_string(),
            ));
        }

        // Process signal pairs in parallel
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signal_pairs
            .par_iter()
            .map(|(sig1, sig2)| self.single_tf_coherence(sig1, sig2, fs, window_size, hop_size))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signal_pairs
            .iter()
            .map(|(sig1, sig2)| self.single_tf_coherence(sig1, sig2, fs, window_size, hop_size))
            .collect();

        results
    }

    /// Parallel spectral entropy computation
    ///
    /// Computes spectral entropy for multiple signals in parallel
    ///
    /// # Arguments
    ///
    /// * `signals` - Vector of signals
    /// * `fs` - Sampling frequency
    /// * `method` - Entropy method ("shannon", "renyi", "tsallis")
    /// * `q` - Parameter for Renyi/Tsallis entropy (ignored for Shannon)
    ///
    /// # Returns
    ///
    /// * Vector of spectral entropy values
    pub fn batch_spectral_entropy(
        &self,
        signals: &[&[f64]],
        fs: f64,
        method: &str,
        q: f64,
    ) -> SignalResult<Vec<f64>> {
        if signals.is_empty() {
            return Err(SignalError::ValueError("No signals provided".to_string()));
        }

        // Process signals in parallel
        #[cfg(feature = "parallel")]
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|&signal| self.single_spectral_entropy(signal, fs, method, q))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<_>, SignalError> = signals
            .iter()
            .map(|&signal| self.single_spectral_entropy(signal, fs, method, q))
            .collect();

        results
    }

    /// Parallel band power computation
    ///
    /// Computes power in specific frequency bands for multiple signals in parallel
    ///
    /// # Arguments
    ///
    /// * `signals` - Vector of signals
    /// * `bands` - Vector of frequency bands (low, high)
    /// * `fs` - Sampling frequency
    ///
    /// # Returns
    ///
    /// * Vector of band power matrices (signals x bands)
    pub fn batch_band_power(
        &self,
        signals: &[&[f64]],
        bands: &[(f64, f64)],
        fs: f64,
    ) -> SignalResult<Array2<f64>> {
        if signals.is_empty() || bands.is_empty() {
            return Err(SignalError::ValueError(
                "No signals or bands provided".to_string(),
            ));
        }

        let mut power_matrix = Array2::zeros((signals.len(), bands.len()));

        // Process signals in parallel
        #[cfg(feature = "parallel")]
        {
            let results: Result<Vec<_>, SignalError> = signals
                .par_iter()
                .map(|&signal| self.single_band_power(signal, bands, fs))
                .collect();

            let band_powers = results?;
            for (i, powers) in band_powers.into_iter().enumerate() {
                for (j, power) in powers.into_iter().enumerate() {
                    power_matrix[[i, j]] = power;
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (i, &signal) in signals.iter().enumerate() {
                let powers = self.single_band_power(signal, bands, fs)?;
                for (j, power) in powers.into_iter().enumerate() {
                    power_matrix[[i, j]] = power;
                }
            }
        }

        Ok(power_matrix)
    }

    // Private helper methods for advanced algorithms

    fn single_cross_psd(
        &self,
        signal1: &[f64],
        signal2: &[f64],
        fs: f64,
        window_size: usize,
        overlap: f64,
    ) -> SignalResult<(Vec<f64>, Vec<Complex64>)> {
        if signal1.len() != signal2.len() {
            return Err(SignalError::ValueError(
                "Signals must have the same length".to_string(),
            ));
        }

        let n = signal1.len();
        if n < window_size {
            return Err(SignalError::ValueError(
                "Signal length must be at least window size".to_string(),
            ));
        }

        let hop_size = ((1.0 - overlap) * window_size as f64) as usize;
        let window = window::hann(window_size, true)
            .map_err(|_| SignalError::ValueError("Failed to create window".to_string()))?;

        let mut fft_planner = self.fft_planner.lock().map_err(|_| {
            SignalError::ComputationError("Failed to acquire FFT planner".to_string())
        })?;
        let fft = fft_planner.plan_fft_forward(window_size);

        let mut cross_psd_sum = vec![Complex64::new(0.0, 0.0); window_size / 2 + 1];
        let mut n_segments = 0;

        let mut idx = 0;
        while idx + window_size <= n {
            // Apply window to both signals
            let windowed1: Vec<Complex64> = signal1[idx..idx + window_size]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex64::new(s * w, 0.0))
                .collect();

            let windowed2: Vec<Complex64> = signal2[idx..idx + window_size]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex64::new(s * w, 0.0))
                .collect();

            // Compute FFTs
            let mut fft1 = windowed1;
            let mut fft2 = windowed2;
            fft.process(&mut fft1);
            fft.process(&mut fft2);

            // Compute cross-power spectrum
            for i in 0..cross_psd_sum.len() {
                cross_psd_sum[i] += fft1[i] * fft2[i].conj();
            }

            n_segments += 1;
            idx += hop_size;
        }

        // Average and normalize
        for cps in &mut cross_psd_sum {
            *cps /= n_segments as f64;
        }

        // Create frequency vector
        let frequencies: Vec<f64> = (0..cross_psd_sum.len())
            .map(|i| i as f64 * fs / window_size as f64)
            .collect();

        Ok((frequencies, cross_psd_sum))
    }

    fn single_pac_analysis(
        &self,
        signal: &[f64],
        phase_band: (f64, f64),
        amplitude_band: (f64, f64),
        fs: f64,
    ) -> SignalResult<f64> {
        use crate::filter::butter;
        use crate::hilbert::hilbert;

        // Design filters for phase and amplitude bands
        let (b_phase, a_phase) = butter(
            4,
            phase_band.1 / (fs / 2.0),
            crate::filter::FilterType::Lowpass,
        )?;
        let (b_amp, a_amp) = butter(
            4,
            amplitude_band.1 / (fs / 2.0),
            crate::filter::FilterType::Lowpass,
        )?;

        // Filter signals (simplified - in practice would use proper bandpass filters)
        let phase_filtered = self.apply_filter(signal, &b_phase, &a_phase)?;
        let amplitude_filtered = self.apply_filter(signal, &b_amp, &a_amp)?;

        // Extract phase from low-frequency component
        let analytic_phase = hilbert(&phase_filtered)?;
        let phases: Vec<f64> = analytic_phase.iter().map(|c| c.arg()).collect();

        // Extract amplitude envelope from high-frequency component
        let analytic_amp = hilbert(&amplitude_filtered)?;
        let amplitudes: Vec<f64> = analytic_amp.iter().map(|c| c.norm()).collect();

        // Compute phase-amplitude coupling using mean vector length
        let n = phases.len().min(amplitudes.len());
        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for i in 0..n {
            let phase = phases[i];
            let amplitude = amplitudes[i];
            sum_real += amplitude * phase.cos();
            sum_imag += amplitude * phase.sin();
        }

        let mean_vector_length = (sum_real * sum_real + sum_imag * sum_imag).sqrt()
            / (amplitudes.iter().sum::<f64>() / n as f64);

        Ok(mean_vector_length)
    }

    fn single_tf_coherence(
        &self,
        signal1: &[f64],
        signal2: &[f64],
        fs: f64,
        window_size: usize,
        hop_size: usize,
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)> {
        if signal1.len() != signal2.len() {
            return Err(SignalError::ValueError(
                "Signals must have the same length".to_string(),
            ));
        }

        let n = signal1.len();
        let n_frames = (n - window_size) / hop_size + 1;
        let n_freqs = window_size / 2 + 1;

        let mut coherence = Array2::zeros((n_freqs, n_frames));
        let window = window::hann(window_size, true)
            .map_err(|_| SignalError::ValueError("Failed to create window".to_string()))?;

        let mut fft_planner = self.fft_planner.lock().map_err(|_| {
            SignalError::ComputationError("Failed to acquire FFT planner".to_string())
        })?;
        let fft = fft_planner.plan_fft_forward(window_size);

        for frame in 0..n_frames {
            let start = frame * hop_size;
            let end = start + window_size;

            if end <= n {
                // Windowed signals
                let windowed1: Vec<Complex64> = signal1[start..end]
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| Complex64::new(s * w, 0.0))
                    .collect();

                let windowed2: Vec<Complex64> = signal2[start..end]
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| Complex64::new(s * w, 0.0))
                    .collect();

                // Compute FFTs
                let mut fft1 = windowed1;
                let mut fft2 = windowed2;
                fft.process(&mut fft1);
                fft.process(&mut fft2);

                // Compute coherence for each frequency
                for freq in 0..n_freqs {
                    let cross_spectrum = fft1[freq] * fft2[freq].conj();
                    let auto1 = fft1[freq].norm_sqr();
                    let auto2 = fft2[freq].norm_sqr();

                    let coherence_val = if auto1 > 0.0 && auto2 > 0.0 {
                        cross_spectrum.norm_sqr() / (auto1 * auto2)
                    } else {
                        0.0
                    };

                    coherence[[freq, frame]] = coherence_val;
                }
            }
        }

        // Create frequency and time vectors
        let frequencies: Vec<f64> = (0..n_freqs)
            .map(|i| i as f64 * fs / window_size as f64)
            .collect();

        let times: Vec<f64> = (0..n_frames)
            .map(|i| i as f64 * hop_size as f64 / fs)
            .collect();

        Ok((frequencies, times, coherence))
    }

    fn single_spectral_entropy(
        &self,
        signal: &[f64],
        fs: f64,
        method: &str,
        q: f64,
    ) -> SignalResult<f64> {
        // Compute power spectral density
        let (_, psd) = self.single_periodogram(signal, fs, Some("hann"), None)?;

        // Normalize PSD to create probability distribution
        let total_power: f64 = psd.iter().sum();
        if total_power <= 0.0 {
            return Ok(0.0);
        }

        let probabilities: Vec<f64> = psd.iter().map(|&p| p / total_power).collect();

        // Compute entropy based on method
        let entropy = match method.to_lowercase().as_str() {
            "shannon" => -probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f64>(),
            "renyi" => {
                if (q - 1.0).abs() < 1e-10 {
                    // Limit case q -> 1 gives Shannon entropy
                    -probabilities
                        .iter()
                        .filter(|&&p| p > 0.0)
                        .map(|&p| p * p.ln())
                        .sum::<f64>()
                } else {
                    let sum_q: f64 = probabilities
                        .iter()
                        .filter(|&&p| p > 0.0)
                        .map(|&p| p.powf(q))
                        .sum();
                    (1.0 / (1.0 - q)) * sum_q.ln()
                }
            }
            "tsallis" => {
                let sum_q: f64 = probabilities
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| p.powf(q))
                    .sum();
                (1.0 - sum_q) / (q - 1.0)
            }
            _ => {
                return Err(SignalError::ValueError(
                    "Unknown entropy method. Use 'shannon', 'renyi', or 'tsallis'".to_string(),
                ))
            }
        };

        Ok(entropy)
    }

    fn single_band_power(
        &self,
        signal: &[f64],
        bands: &[(f64, f64)],
        fs: f64,
    ) -> SignalResult<Vec<f64>> {
        // Compute power spectral density
        let (frequencies, psd) = self.single_periodogram(signal, fs, Some("hann"), None)?;

        let mut band_powers = Vec::with_capacity(bands.len());

        for &(low_freq, high_freq) in bands {
            let power = frequencies
                .iter()
                .zip(psd.iter())
                .filter(|(&freq, _)| freq >= low_freq && freq <= high_freq)
                .map(|(_, &power)| power)
                .sum::<f64>();

            band_powers.push(power);
        }

        Ok(band_powers)
    }

    fn apply_filter(&self, signal: &[f64], b: &[f64], a: &[f64]) -> SignalResult<Vec<f64>> {
        // Simplified filter implementation - in practice would use proper IIR filtering
        crate::filter::lfilter(b, a, signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::f64::consts::PI;

    #[test]
    fn test_parallel_periodogram() {
        let fs = 1000.0;
        let duration = 1.0;
        let n = (fs * duration) as usize;

        // Create test signals
        let signal1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let signal2: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
            .collect();

        let signals = vec![signal1.as_slice(), signal2.as_slice()];

        let processor = ParallelSpectralProcessor::new(ParallelSpectralConfig::default());
        let results = processor
            .batch_periodogram(&signals, fs, Some("hann"), None)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(!results[0].0.is_empty());
        assert!(!results[0].1.is_empty());
    }

    #[test]
    fn test_parallel_spectrogram() {
        let fs = 1000.0;
        let duration = 2.0;
        let n = (fs * duration) as usize;

        // Create chirp signal
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let freq = 50.0 + 100.0 * t; // Linear chirp
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let signals = vec![signal.as_slice()];

        let processor = ParallelSpectralProcessor::new(ParallelSpectralConfig::default());
        let results = processor
            .batch_spectrogram(&signals, fs, 256, 128, Some("hann"))
            .unwrap();

        assert_eq!(results.len(), 1);
        let (frequencies, times, spectrogram) = &results[0];
        assert!(!frequencies.is_empty());
        assert!(!times.is_empty());
        assert_eq!(spectrogram.shape()[0], frequencies.len());
        assert_eq!(spectrogram.shape()[1], times.len());
    }

    #[test]
    fn test_parallel_coherence() {
        let fs = 1000.0;
        let duration = 1.0;
        let n = (fs * duration) as usize;

        // Create correlated signals
        let signal1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let mut rng = rand::rng();
        let signal2: Vec<f64> = signal1
            .iter()
            .map(|&x| x + 0.1 * rng.random::<f64>())
            .collect();

        let signal_pairs = vec![(signal1.as_slice(), signal2.as_slice())];

        let processor = ParallelSpectralProcessor::new(ParallelSpectralConfig::default());
        let results = processor
            .batch_coherence(&signal_pairs, fs, 256, 0.5)
            .unwrap();

        assert_eq!(results.len(), 1);
        let (frequencies, coherence) = &results[0];
        assert_eq!(frequencies.len(), coherence.len());

        // Coherence should be high at 50 Hz
        let freq_50hz_idx = frequencies
            .iter()
            .position(|&f| (f - 50.0).abs() < 5.0)
            .unwrap();
        assert!(coherence[freq_50hz_idx] > 0.5);
    }

    #[test]
    fn test_parallel_welch() {
        let fs = 1000.0;
        let duration = 2.0;
        let n = (fs * duration) as usize;

        // Create test signals
        let signal1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let signal2: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
            .collect();

        let signals = vec![signal1.as_slice(), signal2.as_slice()];

        let results = parallel_welch(&signals, fs, 512, 0.5, Some("hann")).unwrap();

        assert_eq!(results.len(), 2);

        for (frequencies, psd) in &results {
            assert_eq!(frequencies.len(), psd.len());
            assert!(!frequencies.is_empty());
        }
    }

    #[test]
    fn test_processor_configuration() {
        let config = ParallelSpectralConfig {
            num_threads: Some(2),
            min_chunk_size: 500,
            enable_simd: true,
            memory_optimization: 3,
        };

        let processor = ParallelSpectralProcessor::new(config.clone());
        assert_eq!(processor.config.num_threads, Some(2));
        assert_eq!(processor.config.min_chunk_size, 500);
        assert!(processor.config.enable_simd);
        assert_eq!(processor.config.memory_optimization, 3);
    }
}
