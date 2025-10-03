//! Audio quality assessment metrics
//!
//! This module provides comprehensive metrics for evaluating audio quality,
//! including perceptual metrics (PESQ, STOI), objective metrics (SNR, SDR),
//! and intelligibility measures for speech enhancement and audio processing tasks.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};

/// Audio quality assessment metrics
#[derive(Debug, Clone)]
pub struct AudioQualityMetrics {
    /// Perceptual evaluation metrics
    perceptual_metrics: PerceptualAudioMetrics,
    /// Objective quality metrics
    objective_metrics: ObjectiveAudioMetrics,
    /// Intelligibility metrics
    intelligibility_metrics: IntelligibilityMetrics,
}

/// Perceptual audio quality metrics
#[derive(Debug, Clone, Default)]
pub struct PerceptualAudioMetrics {
    /// PESQ (Perceptual Evaluation of Speech Quality)
    pesq: Option<f64>,
    /// STOI (Short-Time Objective Intelligibility)
    stoi: Option<f64>,
    /// MOSNet predicted MOS score
    mosnet_score: Option<f64>,
    /// DNSMOS predicted MOS score
    dnsmos_score: Option<f64>,
    /// SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    si_sdr: Option<f64>,
}

/// Objective audio quality metrics
#[derive(Debug, Clone, Default)]
pub struct ObjectiveAudioMetrics {
    /// Signal-to-Noise Ratio
    snr: f64,
    /// Signal-to-Distortion Ratio
    sdr: f64,
    /// Signal-to-Interference Ratio
    sir: f64,
    /// Signal-to-Artifacts Ratio
    sar: f64,
    /// Frequency-weighted SNR
    fw_snr: f64,
    /// Spectral distortion measures
    spectral_distortion: SpectralDistortionMetrics,
}

/// Spectral distortion metrics
#[derive(Debug, Clone, Default)]
pub struct SpectralDistortionMetrics {
    /// Log-spectral distance
    log_spectral_distance: f64,
    /// Itakura-Saito distance
    itakura_saito_distance: f64,
    /// Mel-cepstral distortion
    mel_cepstral_distortion: f64,
    /// Bark spectral distortion
    bark_spectral_distortion: f64,
}

/// Speech intelligibility metrics
#[derive(Debug, Clone, Default)]
pub struct IntelligibilityMetrics {
    /// Normalized Covariance Measure (NCM)
    ncm: f64,
    /// Coherence Speech Intelligibility Index (CSII)
    csii: f64,
    /// Hearing Aid Speech Quality Index (HASQI)
    hasqi: Option<f64>,
    /// Extended Short-Time Objective Intelligibility (ESTOI)
    estoi: Option<f64>,
}

/// Audio quality assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityResults {
    /// PESQ score
    pub pesq: Option<f64>,
    /// STOI score
    pub stoi: Option<f64>,
    /// Signal-to-Noise Ratio
    pub snr: f64,
    /// Signal-to-Distortion Ratio
    pub sdr: f64,
    /// SI-SDR score
    pub si_sdr: Option<f64>,
}

impl AudioQualityMetrics {
    /// Create new audio quality metrics
    pub fn new() -> Self {
        Self {
            perceptual_metrics: PerceptualAudioMetrics::default(),
            objective_metrics: ObjectiveAudioMetrics::default(),
            intelligibility_metrics: IntelligibilityMetrics::default(),
        }
    }

    /// Compute comprehensive audio quality assessment
    pub fn compute_quality_metrics<F: Float>(
        &mut self,
        clean_signal: ArrayView1<F>,
        processed_signal: ArrayView1<F>,
        noise_signal: Option<ArrayView1<F>>,
        sample_rate: f64,
    ) -> Result<AudioQualityResults> {
        if clean_signal.len() != processed_signal.len() {
            return Err(MetricsError::InvalidInput(
                "Clean and processed signals must have the same length".to_string(),
            ));
        }

        // Compute objective metrics
        self.objective_metrics
            .compute_snr(clean_signal, processed_signal)?;
        self.objective_metrics
            .compute_sdr(clean_signal, processed_signal)?;

        if let Some(noise) = noise_signal {
            self.objective_metrics
                .compute_sir(clean_signal, processed_signal, noise)?;
        }

        // Compute perceptual metrics
        self.perceptual_metrics
            .compute_pesq(clean_signal, processed_signal, sample_rate)?;
        self.perceptual_metrics
            .compute_stoi(clean_signal, processed_signal, sample_rate)?;
        self.perceptual_metrics
            .compute_si_sdr(clean_signal, processed_signal)?;

        // Compute intelligibility metrics
        self.intelligibility_metrics
            .compute_ncm(clean_signal, processed_signal)?;
        self.intelligibility_metrics
            .compute_csii(clean_signal, processed_signal, sample_rate)?;

        Ok(AudioQualityResults {
            pesq: self.perceptual_metrics.pesq,
            stoi: self.perceptual_metrics.stoi,
            snr: self.objective_metrics.snr,
            sdr: self.objective_metrics.sdr,
            si_sdr: self.perceptual_metrics.si_sdr,
        })
    }

    /// Compute PESQ score
    pub fn compute_pesq<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        degraded: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<f64> {
        self.perceptual_metrics
            .compute_pesq(reference, degraded, sample_rate)
    }

    /// Compute STOI score
    pub fn compute_stoi<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        degraded: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<f64> {
        self.perceptual_metrics
            .compute_stoi(reference, degraded, sample_rate)
    }

    /// Compute SNR
    pub fn compute_snr<F: Float>(
        &mut self,
        signal: ArrayView1<F>,
        noise: ArrayView1<F>,
    ) -> Result<f64> {
        self.objective_metrics.compute_snr(signal, noise)
    }

    /// Compute SDR (Signal-to-Distortion Ratio)
    pub fn compute_sdr<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        estimate: ArrayView1<F>,
    ) -> Result<f64> {
        self.objective_metrics.compute_sdr(reference, estimate)
    }

    /// Get comprehensive quality results
    pub fn get_results(&self) -> AudioQualityResults {
        AudioQualityResults {
            pesq: self.perceptual_metrics.pesq,
            stoi: self.perceptual_metrics.stoi,
            snr: self.objective_metrics.snr,
            sdr: self.objective_metrics.sdr,
            si_sdr: self.perceptual_metrics.si_sdr,
        }
    }

    /// Evaluate audio quality (alias for backward compatibility)
    pub fn evaluate_quality<F>(
        &mut self,
        reference_audio: ArrayView1<F>,
        degraded_audio: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<AudioQualityResults>
    where
        F: Float + std::fmt::Debug + std::iter::Sum,
    {
        self.compute_quality_metrics(reference_audio, degraded_audio, None, sample_rate)
    }
}

impl PerceptualAudioMetrics {
    /// Compute PESQ (Perceptual Evaluation of Speech Quality)
    pub fn compute_pesq<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        degraded: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<f64> {
        if reference.len() != degraded.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and degraded signals must have the same length".to_string(),
            ));
        }

        // Simplified PESQ implementation - would use actual ITU-T P.862 algorithm
        let min_length = 8000; // Minimum 1 second at 8kHz
        if reference.len() < min_length {
            return Err(MetricsError::InvalidInput(
                "Signal too short for PESQ computation".to_string(),
            ));
        }

        // Basic correlation-based approximation
        let correlation = self.compute_correlation(reference, degraded);
        let pesq_score = (correlation * 4.5).max(1.0).min(4.5); // PESQ range: 1.0-4.5

        self.pesq = Some(pesq_score);
        Ok(pesq_score)
    }

    /// Compute STOI (Short-Time Objective Intelligibility)
    pub fn compute_stoi<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        degraded: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<f64> {
        if reference.len() != degraded.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and degraded signals must have the same length".to_string(),
            ));
        }

        // Simplified STOI implementation - would use actual third-octave band analysis
        let frame_length = (sample_rate * 0.025) as usize; // 25ms frames
        let hop_length = frame_length / 2;

        if reference.len() < frame_length {
            return Err(MetricsError::InvalidInput(
                "Signal too short for STOI computation".to_string(),
            ));
        }

        let mut stoi_values = Vec::new();

        for i in (0..reference.len() - frame_length).step_by(hop_length) {
            let ref_frame = reference.slice(s![i..i + frame_length]);
            let deg_frame = degraded.slice(s![i..i + frame_length]);

            let correlation = self.compute_correlation(ref_frame, deg_frame);
            stoi_values.push(correlation.max(0.0).min(1.0));
        }

        let stoi_score = if !stoi_values.is_empty() {
            stoi_values.iter().sum::<f64>() / stoi_values.len() as f64
        } else {
            0.0
        };

        self.stoi = Some(stoi_score);
        Ok(stoi_score)
    }

    /// Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    pub fn compute_si_sdr<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        estimate: ArrayView1<F>,
    ) -> Result<f64> {
        if reference.len() != estimate.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and estimate signals must have the same length".to_string(),
            ));
        }

        // Convert to f64 for computation
        let ref_vec: Vec<f64> = reference
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();
        let est_vec: Vec<f64> = estimate
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();

        // Compute optimal scaling factor
        let numerator: f64 = ref_vec.iter().zip(&est_vec).map(|(r, e)| r * e).sum();
        let denominator: f64 = ref_vec.iter().map(|r| r * r).sum();

        if denominator == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }

        let alpha = numerator / denominator;

        // Compute scaled reference
        let scaled_ref: Vec<f64> = ref_vec.iter().map(|r| alpha * r).collect();

        // Compute signal and noise powers
        let signal_power: f64 = scaled_ref.iter().map(|s| s * s).sum();
        let noise_power: f64 = scaled_ref
            .iter()
            .zip(&est_vec)
            .map(|(s, e)| (s - e).powi(2))
            .sum();

        let si_sdr = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        self.si_sdr = Some(si_sdr);
        Ok(si_sdr)
    }

    /// Compute correlation between two signals
    fn compute_correlation<F: Float>(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let x_vec: Vec<f64> = x.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect();
        let y_vec: Vec<f64> = y.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect();

        let mean_x = x_vec.iter().sum::<f64>() / x_vec.len() as f64;
        let mean_y = y_vec.iter().sum::<f64>() / y_vec.len() as f64;

        let numerator: f64 = x_vec
            .iter()
            .zip(&y_vec)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let var_x: f64 = x_vec.iter().map(|x| (x - mean_x).powi(2)).sum();
        let var_y: f64 = y_vec.iter().map(|y| (y - mean_y).powi(2)).sum();

        let denominator = (var_x * var_y).sqrt();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl ObjectiveAudioMetrics {
    /// Compute Signal-to-Noise Ratio (SNR)
    pub fn compute_snr<F: Float>(
        &mut self,
        signal: ArrayView1<F>,
        noise: ArrayView1<F>,
    ) -> Result<f64> {
        let signal_power = self.compute_power(signal);
        let noise_power = self.compute_power(noise);

        self.snr = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        Ok(self.snr)
    }

    /// Compute Signal-to-Distortion Ratio (SDR)
    pub fn compute_sdr<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        estimate: ArrayView1<F>,
    ) -> Result<f64> {
        if reference.len() != estimate.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and estimate signals must have the same length".to_string(),
            ));
        }

        let signal_power = self.compute_power(reference);

        // Compute distortion power
        let distortion_power: f64 = reference
            .iter()
            .zip(estimate.iter())
            .map(|(&r, &e)| {
                let diff = r.to_f64().unwrap_or(0.0) - e.to_f64().unwrap_or(0.0);
                diff * diff
            })
            .sum::<f64>()
            / reference.len() as f64;

        self.sdr = if distortion_power > 0.0 {
            10.0 * (signal_power / distortion_power).log10()
        } else {
            f64::INFINITY
        };

        Ok(self.sdr)
    }

    /// Compute Signal-to-Interference Ratio (SIR)
    pub fn compute_sir<F: Float>(
        &mut self,
        signal: ArrayView1<F>,
        estimate: ArrayView1<F>,
        interference: ArrayView1<F>,
    ) -> Result<f64> {
        let signal_power = self.compute_power(signal);
        let interference_power = self.compute_power(interference);

        self.sir = if interference_power > 0.0 {
            10.0 * (signal_power / interference_power).log10()
        } else {
            f64::INFINITY
        };

        Ok(self.sir)
    }

    /// Compute power of a signal
    fn compute_power<F: Float>(&self, signal: ArrayView1<F>) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }

        signal
            .iter()
            .map(|&x| {
                let val = x.to_f64().unwrap_or(0.0);
                val * val
            })
            .sum::<f64>()
            / signal.len() as f64
    }

    /// Compute spectral distortion metrics
    pub fn compute_spectral_distortion<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        estimate: ArrayView1<F>,
    ) -> Result<()> {
        self.spectral_distortion
            .compute_log_spectral_distance(reference, estimate)?;
        self.spectral_distortion
            .compute_itakura_saito_distance(reference, estimate)?;
        Ok(())
    }
}

impl SpectralDistortionMetrics {
    /// Compute log-spectral distance
    pub fn compute_log_spectral_distance<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        estimate: ArrayView1<F>,
    ) -> Result<f64> {
        // Simplified implementation - would use actual FFT-based spectral analysis
        let ref_spectrum = self.compute_simple_spectrum(reference);
        let est_spectrum = self.compute_simple_spectrum(estimate);

        if ref_spectrum.len() != est_spectrum.len() {
            return Err(MetricsError::InvalidInput(
                "Spectrum lengths must match".to_string(),
            ));
        }

        let mut distance_sum = 0.0;
        let mut valid_bins = 0;

        for (ref_bin, est_bin) in ref_spectrum.iter().zip(est_spectrum.iter()) {
            if *ref_bin > 0.0 && *est_bin > 0.0 {
                distance_sum += (ref_bin.ln() - est_bin.ln()).powi(2);
                valid_bins += 1;
            }
        }

        self.log_spectral_distance = if valid_bins > 0 {
            (distance_sum / valid_bins as f64).sqrt()
        } else {
            0.0
        };

        Ok(self.log_spectral_distance)
    }

    /// Compute Itakura-Saito distance
    pub fn compute_itakura_saito_distance<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        estimate: ArrayView1<F>,
    ) -> Result<f64> {
        let ref_spectrum = self.compute_simple_spectrum(reference);
        let est_spectrum = self.compute_simple_spectrum(estimate);

        let mut distance_sum = 0.0;
        let mut valid_bins = 0;

        for (ref_bin, est_bin) in ref_spectrum.iter().zip(est_spectrum.iter()) {
            if *ref_bin > 0.0 && *est_bin > 0.0 {
                distance_sum += (ref_bin / est_bin) - (ref_bin / est_bin).ln() - 1.0;
                valid_bins += 1;
            }
        }

        self.itakura_saito_distance = if valid_bins > 0 {
            distance_sum / valid_bins as f64
        } else {
            0.0
        };

        Ok(self.itakura_saito_distance)
    }

    /// Compute simple power spectrum (placeholder)
    fn compute_simple_spectrum<F: Float>(&self, signal: ArrayView1<F>) -> Vec<f64> {
        // Simplified spectrum computation - would use actual FFT
        let window_size = signal.len().min(1024);
        let mut spectrum = Vec::with_capacity(window_size / 2);

        for i in 0..window_size / 2 {
            let start = i * 2;
            let end = (start + window_size).min(signal.len());

            if start < signal.len() {
                let power: f64 = signal
                    .slice(s![start..end])
                    .iter()
                    .map(|&x| {
                        let val = x.to_f64().unwrap_or(0.0);
                        val * val
                    })
                    .sum::<f64>()
                    / (end - start) as f64;

                spectrum.push(power.max(1e-10)); // Avoid log(0)
            }
        }

        spectrum
    }
}

impl IntelligibilityMetrics {
    /// Compute Normalized Covariance Measure (NCM)
    pub fn compute_ncm<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        degraded: ArrayView1<F>,
    ) -> Result<f64> {
        if reference.len() != degraded.len() {
            return Err(MetricsError::InvalidInput(
                "Reference and degraded signals must have the same length".to_string(),
            ));
        }

        // Simplified NCM computation
        let correlation = self.compute_cross_correlation(reference, degraded);
        self.ncm = correlation.abs();
        Ok(self.ncm)
    }

    /// Compute Coherence Speech Intelligibility Index (CSII)
    pub fn compute_csii<F: Float>(
        &mut self,
        reference: ArrayView1<F>,
        degraded: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<f64> {
        // Simplified CSII computation - would use actual coherence analysis
        let frame_length = (sample_rate * 0.032) as usize; // 32ms frames
        let hop_length = frame_length / 2;

        let mut coherence_values = Vec::new();

        for i in (0..reference.len() - frame_length).step_by(hop_length) {
            let ref_frame = reference.slice(s![i..i + frame_length]);
            let deg_frame = degraded.slice(s![i..i + frame_length]);

            let coherence = self.compute_frame_coherence(ref_frame, deg_frame);
            coherence_values.push(coherence);
        }

        self.csii = if !coherence_values.is_empty() {
            coherence_values.iter().sum::<f64>() / coherence_values.len() as f64
        } else {
            0.0
        };

        Ok(self.csii)
    }

    /// Compute cross-correlation between signals
    fn compute_cross_correlation<F: Float>(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let x_vec: Vec<f64> = x.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect();
        let y_vec: Vec<f64> = y.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect();

        let mean_x = x_vec.iter().sum::<f64>() / x_vec.len() as f64;
        let mean_y = y_vec.iter().sum::<f64>() / y_vec.len() as f64;

        let numerator: f64 = x_vec
            .iter()
            .zip(&y_vec)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let var_x: f64 = x_vec.iter().map(|x| (x - mean_x).powi(2)).sum();
        let var_y: f64 = y_vec.iter().map(|y| (y - mean_y).powi(2)).sum();

        let denominator = (var_x * var_y).sqrt();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Compute frame-level coherence
    fn compute_frame_coherence<F: Float>(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> f64 {
        // Simplified coherence computation
        self.compute_cross_correlation(x, y).abs()
    }
}

// Import necessary ndarray features
use scirs2_core::ndarray::s;

impl Default for AudioQualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}
