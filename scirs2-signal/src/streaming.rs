// Comprehensive Streaming Signal Processing Framework
//
// This module provides a unified framework for real-time signal processing
// that combines multiple streaming capabilities including filtering, spectral
// analysis, feature extraction, and adaptive processing.
//
// ## Features
//
// - **Real-time Filtering**: Low-latency IIR and FIR filtering
// - **Streaming Spectral Analysis**: STFT, spectrograms, and power spectral density
// - **Feature Extraction**: Real-time extraction of signal features
// - **Adaptive Processing**: Dynamic parameter adjustment based on signal characteristics
// - **Multi-channel Support**: Simultaneous processing of multiple signals
// - **Memory Efficient**: Bounded memory usage with configurable buffer sizes
// - **Low Latency**: Optimized for real-time applications
//
// ## Example Usage
//
// ```rust
// use ndarray::Array1;
// use scirs2_signal::streaming::{StreamingProcessor, StreamingConfig};
// # fn main() -> Result<(), Box<dyn std::error::Error>> {
//
// // Configure streaming processor
// let config = StreamingConfig {
//     sample_rate: 44100.0,
//     buffer_size: 512,
//     enable_spectral_analysis: true,
//     enable_feature_extraction: true,
//     ..Default::default()
// };
//
// let mut processor = StreamingProcessor::new(config)?;
//
// // Process real-time audio
// let audio_frame = Array1::from_vec(vec![0.0; 512]);
// let result = processor.process_frame(&audio_frame)?;
//
// println!("RMS: {:.3}", result.features.rms);
// println!("Spectral centroid: {:.3}", result.features.spectral_centroid);
// # Ok(())
// # }
// ```

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::WindowType;
use crate::streaming_stft::{StreamingStft, StreamingStftConfig};
use crate::utilities::spectral::spectral_centroid;
use crate::utilities::spectral::spectral_flux;
use crate::utilities::spectral::spectral_rolloff;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_positive;
use statrs::statistics::Statistics;
use std::collections::VecDeque;

#[allow(unused_imports)]
// TODO: sosfilt_zi function not implemented yet
// use crate::filter::sosfilt_zi;
/// Configuration for streaming signal processor
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Buffer size for processing blocks
    pub buffer_size: usize,
    /// Maximum latency in samples
    pub max_latency_samples: usize,
    /// Enable spectral analysis
    pub enable_spectral_analysis: bool,
    /// Enable feature extraction
    pub enable_feature_extraction: bool,
    /// Enable adaptive processing
    pub enable_adaptive: bool,
    /// Number of channels
    pub num_channels: usize,
    /// Spectral analysis configuration
    pub spectral_config: SpectralAnalysisConfig,
    /// Feature extraction configuration
    pub feature_config: FeatureExtractionConfig,
    /// Adaptive processing configuration
    pub adaptive_config: AdaptiveProcessingConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            buffer_size: 512,
            max_latency_samples: 2048,
            enable_spectral_analysis: true,
            enable_feature_extraction: true,
            enable_adaptive: false,
            num_channels: 1,
            spectral_config: SpectralAnalysisConfig::default(),
            feature_config: FeatureExtractionConfig::default(),
            adaptive_config: AdaptiveProcessingConfig::default(),
        }
    }
}

/// Configuration for spectral analysis
#[derive(Debug, Clone)]
pub struct SpectralAnalysisConfig {
    /// STFT frame length
    pub stft_frame_length: usize,
    /// STFT hop length
    pub stft_hop_length: usize,
    /// Window function
    pub window: String,
    /// Compute power spectral density
    pub compute_psd: bool,
    /// PSD smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)
    pub psd_smoothing: f64,
    /// Frequency bands for analysis
    pub frequency_bands: Vec<(f64, f64)>,
}

impl Default for SpectralAnalysisConfig {
    fn default() -> Self {
        Self {
            stft_frame_length: 512,
            stft_hop_length: 256,
            window: WindowType::Hann.to_string(),
            compute_psd: true,
            psd_smoothing: 0.8,
            frequency_bands: vec![
                (20.0, 250.0),    // Low
                (250.0, 2000.0),  // Mid
                (2000.0, 8000.0), // High
            ],
        }
    }
}

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// Window length for time-domain features
    pub time_window_length: usize,
    /// Compute spectral features
    pub compute_spectral_features: bool,
    /// Compute temporal features
    pub compute_temporal_features: bool,
    /// Update rate for features (in samples)
    pub feature_update_rate: usize,
    /// Zero-crossing rate threshold
    pub zcr_threshold: f64,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            time_window_length: 1024,
            compute_spectral_features: true,
            compute_temporal_features: true,
            feature_update_rate: 512,
            zcr_threshold: 0.01,
        }
    }
}

/// Configuration for adaptive processing
#[derive(Debug, Clone)]
pub struct AdaptiveProcessingConfig {
    /// Adaptation rate
    pub adaptation_rate: f64,
    /// Noise gate threshold
    pub noise_gate_threshold: f64,
    /// Auto gain control
    pub enable_agc: bool,
    /// AGC target level
    pub agc_target_level: f64,
    /// AGC time constants
    pub agc_attack_time: f64,
    pub agc_release_time: f64,
}

impl Default for AdaptiveProcessingConfig {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.01,
            noise_gate_threshold: -60.0, // dB
            enable_agc: false,
            agc_target_level: -20.0, // dB
            agc_attack_time: 0.01,   // seconds
            agc_release_time: 0.1,   // seconds
        }
    }
}

/// Streaming signal processor result
#[derive(Debug, Clone)]
pub struct StreamingResult {
    /// Time-domain output
    pub output: Array1<f64>,
    /// Spectral analysis results
    pub spectral: Option<SpectralResult>,
    /// Extracted features
    pub features: SignalFeatures,
    /// Processing statistics
    pub statistics: ProcessingStatistics,
}

/// Spectral analysis results
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// STFT spectrum
    pub spectrum: Option<Array1<Complex64>>,
    /// Power spectral density
    pub psd: Option<Array1<f64>>,
    /// Frequency vector
    pub frequencies: Array1<f64>,
    /// Band powers
    pub band_powers: Vec<f64>,
    /// Dominant frequency
    pub dominant_frequency: f64,
}

/// Extracted signal features
#[derive(Debug, Clone)]
pub struct SignalFeatures {
    /// Root mean square
    pub rms: f64,
    /// Peak amplitude
    pub peak: f64,
    /// Zero crossing rate
    pub zero_crossing_rate: f64,
    /// Spectral centroid
    pub spectral_centroid: f64,
    /// Spectral bandwidth
    pub spectral_bandwidth: f64,
    /// Spectral rolloff
    pub spectral_rolloff: f64,
    /// Spectral flux
    pub spectral_flux: f64,
    /// Mel-frequency cepstral coefficients (first 13)
    pub mfcc: Array1<f64>,
    /// Band energy ratios
    pub band_energy_ratios: Vec<f64>,
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStatistics {
    /// Samples processed
    pub samples_processed: usize,
    /// Frames processed
    pub frames_processed: usize,
    /// Current latency in samples
    pub current_latency: usize,
    /// CPU usage estimate (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// Main streaming signal processor
pub struct StreamingProcessor {
    /// Configuration
    config: StreamingConfig,
    /// Multi-channel input buffers
    input_buffers: Vec<VecDeque<f64>>,
    /// Streaming STFT processors
    stft_processors: Vec<StreamingStft>,
    /// Previous PSD for smoothing
    previous_psd: Option<Array1<f64>>,
    /// Feature extraction buffers
    feature_buffers: Vec<VecDeque<f64>>,
    /// Previous spectrum for spectral flux
    previous_spectrum: Option<Array1<f64>>,
    /// AGC state
    agc_state: AgcState,
    /// Noise gate state
    noise_gate_state: NoiseGateState,
    /// Processing statistics
    stats: ProcessingStatistics,
    /// MEL filter bank
    mel_filter_bank: Option<Array2<f64>>,
}

/// Auto Gain Control state
#[derive(Debug, Clone)]
struct AgcState {
    gain: f64,
    envelope: f64,
    attack_coeff: f64,
    release_coeff: f64,
}

/// Noise gate state
#[derive(Debug, Clone)]
struct NoiseGateState {
    is_open: bool,
    envelope: f64,
    threshold_linear: f64,
}

impl StreamingProcessor {
    /// Create a new streaming processor
    pub fn new(config: StreamingConfig) -> SignalResult<Self> {
        // Validate configuration
        check_positive(_config.sample_rate, "sample_rate")?;
        check_positive(_config.buffer_size as f64, "buffer_size")?;

        if config.num_channels == 0 {
            return Err(SignalError::ValueError(
                "Number of channels must be greater than 0".to_string(),
            ));
        }

        // Initialize input buffers
        let input_buffers = (0.._config.num_channels)
            .map(|_| VecDeque::with_capacity(_config.max_latency_samples))
            .collect();

        // Initialize STFT processors
        let stft_config = StreamingStftConfig {
            frame_length: config.spectral_config.stft_frame_length,
            hop_length: config.spectral_config.stft_hop_length,
            window: config.spectral_config.window.clone(),
            center: true,
            magnitude_only: false,
            ..Default::default()
        };

        let stft_processors = if config.enable_spectral_analysis {
            (0.._config.num_channels)
                .map(|_| StreamingStft::new(stft_config.clone()))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };

        // Initialize feature extraction buffers
        let feature_buffers = (0.._config.num_channels)
            .map(|_| VecDeque::with_capacity(_config.feature_config.time_window_length))
            .collect();

        // Initialize AGC state
        let attack_coeff =
            (-1.0 / (_config.adaptive_config.agc_attack_time * config.sample_rate)).exp();
        let release_coeff =
            (-1.0 / (_config.adaptive_config.agc_release_time * config.sample_rate)).exp();

        let agc_state = AgcState {
            gain: 1.0,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
        };

        // Initialize noise gate state
        let threshold_linear = 10.0_f64.powf(_config.adaptive_config.noise_gate_threshold / 20.0);
        let noise_gate_state = NoiseGateState {
            is_open: true,
            envelope: 0.0,
            threshold_linear,
        };

        // Initialize MEL filter bank if needed
        let mel_filter_bank = if config.feature_config.compute_spectral_features {
            Some(create_mel_filter_bank(
                config.spectral_config.stft_frame_length / 2 + 1,
                config.sample_rate,
                13, // Number of MEL coefficients
            )?)
        } else {
            None
        };

        let stats = ProcessingStatistics {
            samples_processed: 0,
            frames_processed: 0,
            current_latency: 0,
            cpu_usage: 0.0,
            memory_usage: 0,
        };

        Ok(Self {
            config,
            input_buffers,
            stft_processors,
            previous_psd: None,
            feature_buffers,
            previous_spectrum: None,
            agc_state,
            noise_gate_state,
            stats,
            mel_filter_bank,
        })
    }

    /// Process a frame of input data
    pub fn process_frame(&mut self, input: &Array1<f64>) -> SignalResult<StreamingResult> {
        let start_time = std::time::Instant::now();

        // Validate input
        if input.len() != self.config.buffer_size {
            return Err(SignalError::ValueError(format!(
                "Input size {} does not match buffer size {}",
                input.len(),
                self.config.buffer_size
            )));
        }

        // Add input to buffers
        for (i, &sample) in input.iter().enumerate() {
            if i < self.input_buffers.len() {
                self.input_buffers[i].push_back(sample);

                // Maintain buffer size
                while self.input_buffers[i].len() > self.config.max_latency_samples {
                    self.input_buffers[i].pop_front();
                }
            }
        }

        // Process each channel
        let mut output = input.clone();
        let mut spectral_result = None;

        // Apply adaptive processing
        if self.config.enable_adaptive {
            output = self.apply_adaptive_processing(&output)?;
        }

        // Spectral analysis
        if self.config.enable_spectral_analysis && !self.stft_processors.is_empty() {
            spectral_result = self.compute_spectral_analysis(&input)?;
        }

        // Feature extraction
        let features = self.extract_features(&input, &spectral_result)?;

        // Update statistics
        self.stats.samples_processed += input.len();
        self.stats.frames_processed += 1;
        self.stats.current_latency = self.calculate_current_latency();
        self.stats.cpu_usage = self.estimate_cpu_usage(start_time.elapsed());
        self.stats.memory_usage = self.estimate_memory_usage();

        Ok(StreamingResult {
            output,
            spectral: spectral_result,
            features,
            statistics: self.stats.clone(),
        })
    }

    /// Process multi-channel input
    pub fn process_multichannel(
        &mut self,
        input: &Array2<f64>,
    ) -> SignalResult<Vec<StreamingResult>> {
        if input.ncols() != self.config.num_channels {
            return Err(SignalError::ValueError(format!(
                "Input channels {} does not match config channels {}",
                input.ncols(),
                self.config.num_channels
            )));
        }

        let mut results = Vec::with_capacity(self.config.num_channels);

        for ch in 0..self.config.num_channels {
            let channel_input = input.column(ch).to_owned();
            let result = self.process_frame(&channel_input)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Apply adaptive processing
    fn apply_adaptive_processing(&mut self, input: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let mut output = input.clone();

        // Auto Gain Control
        if self.config.adaptive_config.enable_agc {
            output = self.apply_agc(&output)?;
        }

        // Noise Gate
        output = self.apply_noise_gate(&output)?;

        Ok(output)
    }

    /// Apply Auto Gain Control
    fn apply_agc(&mut self, input: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let target_linear = 10.0_f64.powf(self.config.adaptive_config.agc_target_level / 20.0);
        let mut output = Array1::zeros(input.len());

        for (i, &sample) in input.iter().enumerate() {
            let abs_sample = sample.abs();

            // Update envelope
            if abs_sample > self.agc_state.envelope {
                self.agc_state.envelope = self.agc_state.envelope * self.agc_state.attack_coeff
                    + abs_sample * (1.0 - self.agc_state.attack_coeff);
            } else {
                self.agc_state.envelope = self.agc_state.envelope * self.agc_state.release_coeff
                    + abs_sample * (1.0 - self.agc_state.release_coeff);
            }

            // Calculate gain
            if self.agc_state.envelope > 1e-10 {
                let desired_gain = target_linear / self.agc_state.envelope;
                self.agc_state.gain = self.agc_state.gain * 0.99 + desired_gain * 0.01;
            }

            // Apply gain with limiting
            output[i] = (sample * self.agc_state.gain).max(-1.0).min(1.0);
        }

        Ok(output)
    }

    /// Apply noise gate
    fn apply_noise_gate(&mut self, input: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let mut output = Array1::zeros(input.len());

        for (i, &sample) in input.iter().enumerate() {
            let abs_sample = sample.abs();

            // Update envelope
            self.noise_gate_state.envelope =
                self.noise_gate_state.envelope * 0.999 + abs_sample * 0.001;

            // Gate decision
            if self.noise_gate_state.envelope > self.noise_gate_state.threshold_linear {
                self.noise_gate_state.is_open = true;
            } else if self.noise_gate_state.envelope < self.noise_gate_state.threshold_linear * 0.5
            {
                self.noise_gate_state.is_open = false;
            }

            output[i] = if self.noise_gate_state.is_open {
                sample
            } else {
                0.0
            };
        }

        Ok(output)
    }

    /// Compute spectral analysis
    fn compute_spectral_analysis(
        &mut self,
        input: &Array1<f64>,
    ) -> SignalResult<Option<SpectralResult>> {
        if self.stft_processors.is_empty() {
            return Ok(None);
        }

        // Process with first channel's STFT
        let spectrum_opt = self.stft_processors[0].process_frame(input)?;

        if let Some(spectrum) = spectrum_opt {
            let n_freq = spectrum.len();
            let fs = self.config.sample_rate;
            let frequencies =
                Array1::from_shape_fn(n_freq, |i| i as f64 * fs / (2.0 * (n_freq - 1) as f64));

            // Compute magnitude spectrum
            let magnitude = spectrum.mapv(|c| c.norm());

            // Compute PSD with smoothing
            let psd = if self.config.spectral_config.compute_psd {
                let current_psd = magnitude.mapv(|m| m * m);

                let smoothed_psd = if let Some(ref prev_psd) = self.previous_psd {
                    let alpha = self.config.spectral_config.psd_smoothing;
                    prev_psd * alpha + &current_psd * (1.0 - alpha)
                } else {
                    current_psd.clone()
                };

                self.previous_psd = Some(smoothed_psd.clone());
                Some(smoothed_psd)
            } else {
                None
            };

            // Compute band powers
            let band_powers = self.compute_band_powers(&magnitude, &frequencies)?;

            // Find dominant frequency
            let dominant_frequency = self.find_dominant_frequency(&magnitude, &frequencies);

            Ok(Some(SpectralResult {
                spectrum: Some(spectrum),
                psd,
                frequencies,
                band_powers,
                dominant_frequency,
            }))
        } else {
            Ok(None)
        }
    }

    /// Compute band powers
    fn compute_band_powers(
        &self,
        magnitude: &Array1<f64>,
        frequencies: &Array1<f64>,
    ) -> SignalResult<Vec<f64>> {
        let mut band_powers = Vec::new();

        for &(low_freq, high_freq) in &self.config.spectral_config.frequency_bands {
            let mut power = 0.0;
            let mut count = 0;

            for (i, &freq) in frequencies.iter().enumerate() {
                if freq >= low_freq && freq <= high_freq {
                    power += magnitude[i] * magnitude[i];
                    count += 1;
                }
            }

            band_powers.push(if count > 0 { power / count as f64 } else { 0.0 });
        }

        Ok(band_powers)
    }

    /// Find dominant frequency
    fn find_dominant_frequency(&self, magnitude: &Array1<f64>, frequencies: &Array1<f64>) -> f64 {
        let max_idx = magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        frequencies[max_idx]
    }

    /// Extract signal features
    fn extract_features(
        &mut self,
        input: &Array1<f64>,
        spectral: &Option<SpectralResult>,
    ) -> SignalResult<SignalFeatures> {
        // Add to feature buffer
        for &sample in input.iter() {
            self.feature_buffers[0].push_back(sample);

            while self.feature_buffers[0].len() > self.config.feature_config.time_window_length {
                self.feature_buffers[0].pop_front();
            }
        }

        let buffer_vec: Vec<f64> = self.feature_buffers[0].iter().cloned().collect();
        let buffer_array = Array1::from_vec(buffer_vec);

        // Time-domain features
    let sq = buffer_array.mapv(|x| x * x);
    let mean_sq = if sq.len() > 0 { sq.sum() / sq.len() as f64 } else { 0.0 };
    let rms = mean_sq.sqrt();
        let peak = buffer_array.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let zero_crossing_rate = self.compute_zero_crossing_rate(&buffer_array);

        // Spectral features
        let (spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flux, mfcc) =
            if let Some(ref spec) = spectral {
                self.compute_spectral_features(spec)?
            } else {
                (0.0, 0.0, 0.0, 0.0, Array1::zeros(13))
            };

        // Band energy ratios
        let band_energy_ratios = if let Some(ref spec) = spectral {
            self.compute_band_energy_ratios(&spec.band_powers)
        } else {
            vec![0.0; self.config.spectral_config.frequency_bands.len()]
        };

        Ok(SignalFeatures {
            rms,
            peak,
            zero_crossing_rate,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flux,
            mfcc,
            band_energy_ratios,
        })
    }

    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, signal: &Array1<f64>) -> f64 {
        let mut crossings = 0;
        let threshold = self.config.feature_config.zcr_threshold;

        for i in 1..signal.len() {
            if (signal[i] > threshold && signal[i - 1] < -threshold)
                || (signal[i] < -threshold && signal[i - 1] > threshold)
            {
                crossings += 1;
            }
        }

        crossings as f64 / signal.len() as f64
    }

    /// Compute spectral features
    fn compute_spectral_features(
        &mut self,
        spectral: &SpectralResult,
    ) -> SignalResult<(f64, f64, f64, f64, Array1<f64>)> {
        let magnitude = if let Some(ref spectrum) = spectral.spectrum {
            spectrum.mapv(|c| c.norm())
        } else {
            return Ok((0.0, 0.0, 0.0, 0.0, Array1::zeros(13)));
        };

        let frequencies = &spectral.frequencies;

        // Spectral centroid
        let total_magnitude: f64 = magnitude.sum();
        let spectral_centroid = if total_magnitude > 1e-10 {
            frequencies
                .iter()
                .zip(magnitude.iter())
                .map(|(f, m)| f * m)
                .sum::<f64>()
                / total_magnitude
        } else {
            0.0
        };

        // Spectral bandwidth
        let spectral_bandwidth = if total_magnitude > 1e-10 {
            let variance = frequencies
                .iter()
                .zip(magnitude.iter())
                .map(|(f, m)| (f - spectral_centroid).powi(2) * m)
                .sum::<f64>()
                / total_magnitude;
            variance.sqrt()
        } else {
            0.0
        };

        // Spectral rolloff (frequency below which 85% of energy lies)
        let mut cumulative_energy = 0.0;
        let total_energy = magnitude.mapv(|m| m * m).sum();
        let threshold = 0.85 * total_energy;
        let mut spectral_rolloff = frequencies[frequencies.len() - 1];

        for (i, &mag) in magnitude.iter().enumerate() {
            cumulative_energy += mag * mag;
            if cumulative_energy >= threshold {
                spectral_rolloff = frequencies[i];
                break;
            }
        }

        // Spectral flux
        let spectral_flux = if let Some(ref prev_spectrum) = self.previous_spectrum {
            magnitude
                .iter()
                .zip(prev_spectrum.iter())
                .map(|(curr, prev)| (curr - prev).powi(2))
                .sum::<f64>()
                .sqrt()
        } else {
            0.0
        };

        self.previous_spectrum = Some(magnitude.clone());

        // MFCC computation
        let mfcc = if let Some(ref mel_filters) = self.mel_filter_bank {
            self.compute_mfcc(&magnitude, mel_filters)?
        } else {
            Array1::zeros(13)
        };

        Ok((
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flux,
            mfcc,
        ))
    }

    /// Compute MFCC features
    fn compute_mfcc(
        &self,
        magnitude: &Array1<f64>,
        mel_filters: &Array2<f64>,
    ) -> SignalResult<Array1<f64>> {
        // Apply mel filter bank
        let mel_spectrum = mel_filters.dot(magnitude);

        // Log compression
        let log_mel = mel_spectrum.mapv(|x| (x + 1e-10).ln());

        // DCT (simplified version)
        let mut mfcc = Array1::zeros(13);
        let n_mel = log_mel.len();

        for k in 0..13 {
            let mut sum = 0.0;
            for n in 0..n_mel {
                sum += log_mel[n] * (PI * k as f64 * (n as f64 + 0.5) / n_mel as f64).cos();
            }
            mfcc[k] = sum;
        }

        Ok(mfcc)
    }

    /// Compute band energy ratios
    fn compute_band_energy_ratios(&self, bandpowers: &[f64]) -> Vec<f64> {
        let total_power: f64 = band_powers.iter().sum();

        if total_power > 1e-10 {
            band_powers
                .iter()
                .map(|&power| power / total_power)
                .collect()
        } else {
            vec![0.0; band_powers.len()]
        }
    }

    /// Calculate current latency
    fn calculate_current_latency(&self) -> usize {
        let buffer_latency = self.input_buffers.get(0).map(|buf| buf.len()).unwrap_or(0);

        let stft_latency = self
            .stft_processors
            .get(0)
            .map(|stft| stft.get_latency_samples())
            .unwrap_or(0);

        buffer_latency + stft_latency
    }

    /// Estimate CPU usage
    fn estimate_cpu_usage(&self, processingtime: std::time::Duration) -> f64 {
        let samples_per_second = self.config.sample_rate;
        let frame_time = self.config.buffer_size as f64 / samples_per_second;
        let processing_time_seconds = processing_time.as_secs_f64();

        (processing_time_seconds / frame_time).min(1.0)
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let mut usage = 0;

        // Input buffers
        for buffer in &self.input_buffers {
            usage += buffer.len() * std::mem::size_of::<f64>();
        }

        // Feature buffers
        for buffer in &self.feature_buffers {
            usage += buffer.len() * std::mem::size_of::<f64>();
        }

        // STFT processors (estimate)
        usage += self.stft_processors.len() * 4096; // Rough estimate

        // MEL filter bank
        if let Some(ref mel_filters) = self.mel_filter_bank {
            usage += mel_filters.len() * std::mem::size_of::<f64>();
        }

        usage
    }

    /// Get current configuration
    pub fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> &ProcessingStatistics {
        &self.stats
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        for buffer in &mut self.input_buffers {
            buffer.clear();
        }

        for buffer in &mut self.feature_buffers {
            buffer.clear();
        }

        for stft in &mut self.stft_processors {
            stft.reset();
        }

        self.previous_psd = None;
        self.previous_spectrum = None;

        self.agc_state.gain = 1.0;
        self.agc_state.envelope = 0.0;

        self.noise_gate_state.is_open = true;
        self.noise_gate_state.envelope = 0.0;

        self.stats = ProcessingStatistics {
            samples_processed: 0,
            frames_processed: 0,
            current_latency: 0,
            cpu_usage: 0.0,
            memory_usage: 0,
        };
    }
}

/// Create MEL filter bank
#[allow(dead_code)]
fn create_mel_filter_bank(
    n_fft: usize,
    sample_rate: f64,
    n_mels: usize,
) -> SignalResult<Array2<f64>> {
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);

    // Mel points
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    // Convert to FFT bin numbers
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz * n_fft as f64) / sample_rate).round() as usize)
        .collect();

    let mut filter_bank = Array2::zeros((n_mels, n_fft));

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        // Triangular filter
        for k in left..=right {
            if k < n_fft {
                if k < center {
                    filter_bank[[m, k]] = (k - left) as f64 / (center - left) as f64;
                } else {
                    filter_bank[[m, k]] = (right - k) as f64 / (right - center) as f64;
                }
            }
        }
    }

    Ok(filter_bank)
}

/// Convert Hz to Mel scale
#[allow(dead_code)]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + _hz / 700.0).log10()
}

/// Convert Mel to Hz scale
#[allow(dead_code)]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(_mel / 2595.0) - 1.0)
}

mod tests {
    #[test]
    fn test_streaming_processor_creation() {
        let config = StreamingConfig::default();
        let processor = StreamingProcessor::new(config).unwrap();

        assert_eq!(processor.config.buffer_size, 512);
        assert_eq!(processor.config.num_channels, 1);
    }

    #[test]
    fn test_streaming_processor_basic() {
        let config = StreamingConfig {
            buffer_size: 256,
            enable_spectral_analysis: true,
            enable_feature_extraction: true,
            ..Default::default()
        };

        let mut processor = StreamingProcessor::new(config).unwrap();

        // Generate test signal
        let fs = 1000.0;
        let freq = 100.0;
        let input: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let input_array = Array1::from(input);
        let result = processor.process_frame(&input_array).unwrap();

        assert_eq!(result.output.len(), 256);
        assert!(result.features.rms > 0.0);
        assert!(result.features.peak > 0.0);
    }

    #[test]
    fn test_agc_functionality() {
        let config = StreamingConfig {
            buffer_size: 128,
            enable_adaptive: true,
            adaptive_config: AdaptiveProcessingConfig {
                enable_agc: true,
                agc_target_level: -20.0,
                ..Default::default()
            },
            enable_spectral_analysis: false,
            enable_feature_extraction: false,
            ..Default::default()
        };

        let mut processor = StreamingProcessor::new(config).unwrap();

        // Test with loud signal
        let loud_input = Array1::from_vec(vec![0.9; 128]);
        let result1 = processor.process_frame(&loud_input).unwrap();

        // Test with quiet signal
        let quiet_input = Array1::from_vec(vec![0.1; 128]);
        let result2 = processor.process_frame(&quiet_input).unwrap();

        // AGC should adjust levels
        assert!(result1.output.iter().all(|&x: &f64| x.abs() <= 1.0));
        assert!(result2.output.iter().all(|&x: &f64| x.abs() <= 1.0));
    }

    #[test]
    fn test_feature_extraction() {
        let _config = StreamingConfig {
            buffer_size: 512,
            enable_spectral_analysis: true,
            enable_feature_extraction: true,
            ..Default::default()
        };

        let mut processor = StreamingProcessor::new(config).unwrap();

        // Generate complex test signal
        let fs = 1000.0;
        let input: Vec<f64> = (0..512)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 100.0 * t).sin() + 0.5 * (2.0 * PI * 200.0 * t).sin()
            })
            .collect();

        let input_array = Array1::from(input);
        let result = processor.process_frame(&input_array).unwrap();

        assert!(result.features.spectral_centroid > 0.0);
        assert!(result.features.spectral_bandwidth > 0.0);
        assert_eq!(result.features.mfcc.len(), 13);
    }

    #[test]
    fn test_multichannel_processing() {
        let config = StreamingConfig {
            buffer_size: 256,
            num_channels: 2,
            ..Default::default()
        };

        let mut processor = StreamingProcessor::new(config).unwrap();

        let input = Array2::from_shape_fn((256, 2), |(i, ch)| (i as f64 + ch as f64).sin());

        let results = processor.process_multichannel(&input).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_processor_reset() {
        let config = StreamingConfig::default();
        let mut processor = StreamingProcessor::new(config).unwrap();

        let input = Array1::from_vec(vec![0.5; 512]);
        let _ = processor.process_frame(&input).unwrap();

        assert!(processor.stats.samples_processed > 0);

        processor.reset();
        assert_eq!(processor.stats.samples_processed, 0);
    }

    #[test]
    fn test_mel_filter_bank() {
        let filter_bank = create_mel_filter_bank(257, 8000.0, 13).unwrap();
        assert_eq!(filter_bank.nrows(), 13);
        assert_eq!(filter_bank.ncols(), 257);

        // Check that filters sum to reasonable values
        for row in filter_bank.rows() {
            let sum: f64 = row.sum();
            assert!(sum >= 0.0);
        }
    }

    #[test]
    fn test_frequency_conversions() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 1e-6);
    }
}
