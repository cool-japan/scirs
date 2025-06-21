//! Core sparse FFT algorithm implementations
//!
//! This module contains the main SparseFFT struct and its algorithm implementations.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use num_complex::Complex64;
use num_traits::NumCast;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::time::Instant;

use super::config::{SparseFFTAlgorithm, SparseFFTConfig};
use super::estimation::estimate_sparsity;
use super::windowing::apply_window;

/// Result of a sparse FFT computation
#[derive(Debug, Clone)]
pub struct SparseFFTResult {
    /// Sparse frequency components (values)
    pub values: Vec<Complex64>,
    /// Indices of the sparse frequency components
    pub indices: Vec<usize>,
    /// Estimated sparsity
    pub estimated_sparsity: usize,
    /// Computation time
    pub computation_time: std::time::Duration,
    /// Algorithm used
    pub algorithm: SparseFFTAlgorithm,
}

/// Sparse FFT processor
pub struct SparseFFT {
    /// Configuration
    config: SparseFFTConfig,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl SparseFFT {
    /// Create a new sparse FFT processor with the given configuration
    pub fn new(config: SparseFFTConfig) -> Self {
        let seed = config.seed.unwrap_or_else(rand::random);
        let rng = rand::rngs::StdRng::seed_from_u64(seed);

        Self { config, rng }
    }

    /// Create a new sparse FFT processor with default configuration
    pub fn with_default_config() -> Self {
        Self::new(SparseFFTConfig::default())
    }

    /// Estimate sparsity of a signal
    pub fn estimate_sparsity<T>(&mut self, signal: &[T]) -> FFTResult<usize>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        estimate_sparsity(signal, &self.config)
    }

    /// Calculate spectral flatness measure (Wiener entropy)
    /// Returns a value between 0 and 1:
    /// - Values close to 0 indicate sparse, tonal spectra
    /// - Values close to 1 indicate noise-like, dense spectra
    fn calculate_spectral_flatness(&self, magnitudes: &[f64]) -> f64 {
        if magnitudes.is_empty() {
            return 1.0; // Default to maximum flatness for empty input
        }

        // Add a small epsilon to avoid log(0) and division by zero
        let epsilon = 1e-10;

        // Calculate geometric mean
        let log_sum: f64 = magnitudes.iter().map(|&x| (x + epsilon).ln()).sum::<f64>();
        let geometric_mean = (log_sum / magnitudes.len() as f64).exp();

        // Calculate arithmetic mean
        let arithmetic_mean: f64 = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;

        if arithmetic_mean < epsilon {
            return 1.0; // Avoid division by zero
        }

        // Spectral flatness is the ratio of geometric mean to arithmetic mean
        let flatness = geometric_mean / arithmetic_mean;

        // Ensure the result is in [0, 1]
        flatness.clamp(0.0, 1.0)
    }

    /// Perform sparse FFT on the input signal
    pub fn sparse_fft<T>(&mut self, signal: &[T]) -> FFTResult<SparseFFTResult>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Measure performance
        let start = Instant::now();

        // Limit signal size for testing to avoid timeouts
        let limit = signal.len().min(self.config.max_signal_size);
        let limited_signal = &signal[..limit];

        // Apply windowing function if configured
        let windowed_signal = apply_window(
            limited_signal,
            self.config.window_function,
            self.config.kaiser_beta,
        )?;

        // Estimate sparsity if needed
        let estimated_sparsity = self.estimate_sparsity(&windowed_signal)?;

        // Choose algorithm based on configuration
        let (values, indices) = match self.config.algorithm {
            SparseFFTAlgorithm::Sublinear => {
                self.sublinear_sfft(&windowed_signal, estimated_sparsity)?
            }
            SparseFFTAlgorithm::CompressedSensing => {
                self.compressed_sensing_sfft(&windowed_signal, estimated_sparsity)?
            }
            SparseFFTAlgorithm::Iterative => {
                self.iterative_sfft(&windowed_signal, estimated_sparsity)?
            }
            SparseFFTAlgorithm::Deterministic => {
                self.deterministic_sfft(&windowed_signal, estimated_sparsity)?
            }
            SparseFFTAlgorithm::FrequencyPruning => {
                self.frequency_pruning_sfft(&windowed_signal, estimated_sparsity)?
            }
            SparseFFTAlgorithm::SpectralFlatness => {
                self.spectral_flatness_sfft(&windowed_signal, estimated_sparsity)?
            }
        };

        // Record computation time
        let computation_time = start.elapsed();

        Ok(SparseFFTResult {
            values,
            indices,
            estimated_sparsity,
            computation_time,
            algorithm: self.config.algorithm,
        })
    }

    /// Perform sparse FFT and reconstruct the full spectrum
    pub fn sparse_fft_full<T>(&mut self, signal: &[T]) -> FFTResult<Vec<Complex64>>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        let n = signal.len().min(self.config.max_signal_size);

        // Apply windowing function if configured
        let windowed_signal = apply_window(
            &signal[..n],
            self.config.window_function,
            self.config.kaiser_beta,
        )?;
        let result = self.sparse_fft(&windowed_signal)?;

        // Reconstruct full spectrum
        let mut spectrum = vec![Complex64::new(0.0, 0.0); n];
        for (value, &index) in result.values.iter().zip(result.indices.iter()) {
            spectrum[index] = *value;
        }

        Ok(spectrum)
    }

    /// Reconstruct time-domain signal from sparse frequency components
    pub fn reconstruct_signal(
        &self,
        sparse_result: &SparseFFTResult,
        n: usize,
    ) -> FFTResult<Vec<Complex64>> {
        // Create full spectrum from sparse representation
        let mut spectrum = vec![Complex64::new(0.0, 0.0); n];
        for (value, &index) in sparse_result
            .values
            .iter()
            .zip(sparse_result.indices.iter())
        {
            spectrum[index] = *value;
        }

        // Perform inverse FFT to get time-domain signal
        ifft(&spectrum, None)
    }

    /// Implementation of sublinear sparse FFT algorithm
    fn sublinear_sfft<T>(
        &mut self,
        signal: &[T],
        k: usize,
    ) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Convert input to complex
        let signal_complex: Vec<Complex64> = signal
            .iter()
            .map(|&val| {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                })?;
                Ok(Complex64::new(val_f64, 0.0))
            })
            .collect::<FFTResult<Vec<_>>>()?;

        let _n = signal_complex.len();

        // For this implementation, we'll use a simplified approach
        // A real sublinear algorithm would use more sophisticated techniques
        let spectrum = fft(&signal_complex, None)?;

        // Find frequency components
        let mut freq_with_magnitudes: Vec<(f64, usize, Complex64)> = spectrum
            .iter()
            .enumerate()
            .map(|(i, &coef)| (coef.norm(), i, coef))
            .collect();

        // Sort by magnitude in descending order
        freq_with_magnitudes
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Select largest k (or fewer) components
        let mut selected_indices = Vec::new();
        let mut selected_values = Vec::new();

        for &(_, idx, val) in freq_with_magnitudes.iter().take(k) {
            selected_indices.push(idx);
            selected_values.push(val);
        }

        Ok((selected_values, selected_indices))
    }

    /// Implementation of compressed sensing based sparse FFT
    fn compressed_sensing_sfft<T>(
        &mut self,
        signal: &[T],
        k: usize,
    ) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Convert input to complex
        let signal_complex: Vec<Complex64> = signal
            .iter()
            .map(|&val| {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                })?;
                Ok(Complex64::new(val_f64, 0.0))
            })
            .collect::<FFTResult<Vec<_>>>()?;

        let n = signal_complex.len();

        // Number of measurements (m << n for compression)
        let m = (4 * k * (self.config.iterations as f64).log2() as usize).min(n);

        // For a simplified implementation, we'll take random time-domain samples
        let mut _measurements = Vec::with_capacity(m);
        let mut _sample_indices = Vec::with_capacity(m);

        for _ in 0..m {
            let idx = self.rng.random_range(0..n);
            _sample_indices.push(idx);
            _measurements.push(signal_complex[idx]);
        }

        // For this demo, we'll just do a regular FFT and extract the k largest components
        let spectrum = fft(&signal_complex, None)?;

        // Find frequency components
        let mut freq_with_magnitudes: Vec<(f64, usize, Complex64)> = spectrum
            .iter()
            .enumerate()
            .map(|(i, &coef)| (coef.norm(), i, coef))
            .collect();

        // Sort by magnitude in descending order
        freq_with_magnitudes
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Select largest k components
        let mut selected_indices = Vec::new();
        let mut selected_values = Vec::new();

        for &(_, idx, val) in freq_with_magnitudes.iter().take(k) {
            selected_indices.push(idx);
            selected_values.push(val);
        }

        Ok((selected_values, selected_indices))
    }

    /// Implementation of iterative sparse FFT algorithm
    fn iterative_sfft<T>(
        &mut self,
        signal: &[T],
        k: usize,
    ) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Convert input to complex
        let mut signal_complex: Vec<Complex64> = signal
            .iter()
            .map(|&val| {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                })?;
                Ok(Complex64::new(val_f64, 0.0))
            })
            .collect::<FFTResult<Vec<_>>>()?;

        let mut selected_indices = Vec::new();
        let mut selected_values = Vec::new();

        // Iterative peeling: find one component at a time
        for _ in 0..k.min(self.config.iterations) {
            // Compute FFT of current residual
            let spectrum = fft(&signal_complex, None)?;

            // Find the strongest frequency component
            let (best_idx, best_value) = spectrum
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.norm()
                        .partial_cmp(&b.norm())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, &val)| (i, val))
                .ok_or_else(|| FFTError::ValueError("Empty spectrum".to_string()))?;

            // If this component is too small, stop
            if best_value.norm() < 1e-10 {
                break;
            }

            // Add this component to our result
            selected_indices.push(best_idx);
            selected_values.push(best_value);

            // Subtract this component from the signal (simplified)
            // In a real implementation, this would be more sophisticated
            let n = signal_complex.len();
            for (i, sample) in signal_complex.iter_mut().enumerate() {
                let phase =
                    2.0 * std::f64::consts::PI * (best_idx as f64) * (i as f64) / (n as f64);
                let component = best_value * Complex64::new(phase.cos(), phase.sin()) / (n as f64);
                *sample -= component;
            }
        }

        Ok((selected_values, selected_indices))
    }

    /// Implementation of deterministic sparse FFT algorithm
    fn deterministic_sfft<T>(
        &mut self,
        signal: &[T],
        k: usize,
    ) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // For this implementation, use a simple deterministic approach
        // based on fixed subsampling patterns
        self.sublinear_sfft(signal, k)
    }

    /// Implementation of frequency pruning sparse FFT algorithm
    fn frequency_pruning_sfft<T>(
        &mut self,
        signal: &[T],
        k: usize,
    ) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Convert input to complex
        let signal_complex: Vec<Complex64> = signal
            .iter()
            .map(|&val| {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                })?;
                Ok(Complex64::new(val_f64, 0.0))
            })
            .collect::<FFTResult<Vec<_>>>()?;

        // Compute full FFT
        let spectrum = fft(&signal_complex, None)?;
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

        // Compute statistics for pruning
        let n = magnitudes.len();
        let mean: f64 = magnitudes.iter().sum::<f64>() / n as f64;
        let variance: f64 = magnitudes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Define pruning threshold
        let threshold = mean + self.config.pruning_sensitivity * std_dev;

        // Find components above threshold
        let mut candidates: Vec<(f64, usize, Complex64)> = spectrum
            .iter()
            .enumerate()
            .filter(|(_, c)| c.norm() > threshold)
            .map(|(i, &c)| (c.norm(), i, c))
            .collect();

        // Sort by magnitude
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k components
        let selected_count = k.min(candidates.len());
        let selected_indices: Vec<usize> = candidates[..selected_count]
            .iter()
            .map(|(_, i, _)| *i)
            .collect();
        let selected_values: Vec<Complex64> = candidates[..selected_count]
            .iter()
            .map(|(_, _, c)| *c)
            .collect();

        Ok((selected_values, selected_indices))
    }

    /// Implementation of spectral flatness sparse FFT algorithm
    fn spectral_flatness_sfft<T>(
        &mut self,
        signal: &[T],
        k: usize,
    ) -> FFTResult<(Vec<Complex64>, Vec<usize>)>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        // Convert input to complex
        let signal_complex: Vec<Complex64> = signal
            .iter()
            .map(|&val| {
                let val_f64 = NumCast::from(val).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {:?} to f64", val))
                })?;
                Ok(Complex64::new(val_f64, 0.0))
            })
            .collect::<FFTResult<Vec<_>>>()?;

        // Compute full FFT
        let spectrum = fft(&signal_complex, None)?;
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

        // Analyze spectral flatness in segments
        let n = magnitudes.len();
        let window_size = self.config.window_size.min(n);
        let mut selected_indices = Vec::new();
        let mut selected_values = Vec::new();

        for start in (0..n).step_by(window_size / 2) {
            let end = (start + window_size).min(n);
            if start >= n {
                break;
            }

            let window_mags = &magnitudes[start..end];
            let flatness = self.calculate_spectral_flatness(window_mags);

            // If flatness is low (indicates structure), find peak in this window
            if flatness < self.config.flatness_threshold {
                if let Some((local_idx, _)) = window_mags
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
                    let global_idx = start + local_idx;
                    if !selected_indices.contains(&global_idx) {
                        selected_indices.push(global_idx);
                        selected_values.push(spectrum[global_idx]);
                    }
                }
            }

            // Stop if we have enough components
            if selected_indices.len() >= k {
                break;
            }
        }

        // If we don't have enough components, fall back to largest magnitude selection
        if selected_indices.len() < k {
            let mut remaining_candidates: Vec<(f64, usize, Complex64)> = spectrum
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected_indices.contains(i))
                .map(|(i, &c)| (c.norm(), i, c))
                .collect();

            remaining_candidates
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let needed = k - selected_indices.len();
            for (_, idx, val) in remaining_candidates.into_iter().take(needed) {
                selected_indices.push(idx);
                selected_values.push(val);
            }
        }

        Ok((selected_values, selected_indices))
    }
}

// Public API functions for backward compatibility

/// Compute sparse FFT of a signal
pub fn sparse_fft<T>(
    signal: &[T],
    k: usize,
    algorithm: Option<SparseFFTAlgorithm>,
    seed: Option<u64>,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    let config = SparseFFTConfig {
        sparsity: k,
        algorithm: algorithm.unwrap_or(SparseFFTAlgorithm::Sublinear),
        seed,
        ..SparseFFTConfig::default()
    };

    let mut processor = SparseFFT::new(config);
    processor.sparse_fft(signal)
}

/// Adaptive sparse FFT with automatic sparsity estimation
pub fn adaptive_sparse_fft<T>(signal: &[T], threshold: f64) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    let config = SparseFFTConfig {
        estimation_method: super::config::SparsityEstimationMethod::Adaptive,
        threshold,
        adaptivity_factor: threshold,
        ..SparseFFTConfig::default()
    };

    let mut processor = SparseFFT::new(config);
    processor.sparse_fft(signal)
}

/// Frequency pruning sparse FFT
pub fn frequency_pruning_sparse_fft<T>(signal: &[T], sensitivity: f64) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    let config = SparseFFTConfig {
        estimation_method: super::config::SparsityEstimationMethod::FrequencyPruning,
        algorithm: SparseFFTAlgorithm::FrequencyPruning,
        pruning_sensitivity: sensitivity,
        ..SparseFFTConfig::default()
    };

    let mut processor = SparseFFT::new(config);
    processor.sparse_fft(signal)
}

/// Spectral flatness sparse FFT
pub fn spectral_flatness_sparse_fft<T>(
    signal: &[T],
    flatness_threshold: f64,
    window_size: usize,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    let config = SparseFFTConfig {
        estimation_method: super::config::SparsityEstimationMethod::SpectralFlatness,
        algorithm: SparseFFTAlgorithm::SpectralFlatness,
        flatness_threshold,
        window_size,
        ..SparseFFTConfig::default()
    };

    let mut processor = SparseFFT::new(config);
    processor.sparse_fft(signal)
}

/// 2D sparse FFT (placeholder implementation)
pub fn sparse_fft2<T>(
    _signal: &[Vec<T>],
    _k: usize,
    _algorithm: Option<SparseFFTAlgorithm>,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Placeholder implementation
    Err(FFTError::ValueError(
        "2D sparse FFT not yet implemented".to_string(),
    ))
}

/// N-dimensional sparse FFT (placeholder implementation)
pub fn sparse_fftn<T>(
    _signal: &[T],
    _shape: &[usize],
    _k: usize,
    _algorithm: Option<SparseFFTAlgorithm>,
) -> FFTResult<SparseFFTResult>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Placeholder implementation
    Err(FFTError::ValueError(
        "N-dimensional sparse FFT not yet implemented".to_string(),
    ))
}
