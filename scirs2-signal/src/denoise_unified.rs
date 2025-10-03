// Unified denoising API
//
// This module provides a consistent, high-level interface to all denoising
// methods available in the scirs2-signal library. It allows users to easily
// switch between different denoising algorithms and compare their performance.

use crate::denoise::{denoise_wavelet, ThresholdMethod, ThresholdSelect};
use crate::denoise_advanced::{advanced_denoise, AdvancedDenoiseConfig, NoiseEstimation};
use crate::denoise_cutting_edge::{denoise_dictionary_learning, DictionaryDenoiseConfig};
use crate::denoise_super_advanced::{advanced_advanced_denoise, AdvancedAdvancedDenoisingConfig};
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;

#[allow(unused_imports)]
use crate::denoise_enhanced::{
    denoise_median_1d, denoise_total_variation_1d, denoise_wiener_1d, WienerConfig,
};

/// Unified denoising method selector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DenoisingMethod {
    /// Basic wavelet denoising
    WaveletBasic {
        wavelet: Wavelet,
        levels: Option<usize>,
        threshold_method: ThresholdMethod,
        threshold_select: ThresholdSelect,
    },
    /// Advanced wavelet denoising with multiple techniques
    WaveletAdvanced { config: AdvancedDenoiseConfig },
    /// Dictionary learning based denoising
    DictionaryLearning { config: DictionaryDenoiseConfig },
    /// Wiener filtering
    Wiener { config: WienerConfig },
    /// Median filtering
    Median { window_size: usize },
    /// Total variation denoising
    TotalVariation { lambda: f64, iterations: usize },
    /// Advanced-advanced denoising with modern techniques
    AdvancedAdvanced {
        config: AdvancedAdvancedDenoisingConfig,
    },
    /// Adaptive hybrid denoising with automatic method selection
    AdaptiveHybrid {
        fallback_methods: Vec<DenoisingMethod>,
        quality_threshold: f64,
        max_iterations: usize,
    },
    /// Non-local means denoising
    NonLocalMeans {
        search_window: usize,
        patch_size: usize,
        filtering_strength: f64,
    },
}

impl Default for DenoisingMethod {
    fn default() -> Self {
        Self::WaveletBasic {
            wavelet: Wavelet::DB(4),
            levels: Some(4),
            threshold_method: ThresholdMethod::Soft,
            threshold_select: ThresholdSelect::Universal,
        }
    }
}

/// Unified denoising configuration
#[derive(Debug, Clone)]
pub struct UnifiedDenoisingConfig {
    /// Primary denoising method
    pub method: DenoisingMethod,
    /// Noise level estimate (if known)
    pub noise_level: Option<f64>,
    /// Enable preprocessing
    pub enable_preprocessing: bool,
    /// Enable postprocessing
    pub enable_postprocessing: bool,
    /// Enable performance benchmarking
    pub benchmark: bool,
}

impl Default for UnifiedDenoisingConfig {
    fn default() -> Self {
        Self {
            method: DenoisingMethod::default(),
            noise_level: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
            benchmark: false,
        }
    }
}

/// Unified denoising result with comprehensive information
#[derive(Debug, Clone)]
pub struct UnifiedDenoisingResult {
    /// Denoised signal
    pub denoised: Array1<f64>,
    /// Estimated noise level
    pub estimated_noise_level: f64,
    /// Signal-to-noise ratio improvement (in dB)
    pub snr_improvement: Option<f64>,
    /// Method used for denoising
    pub method_used: DenoisingMethod,
    /// Processing time (if benchmarking enabled)
    pub processing_time_ms: Option<f64>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for denoising assessment
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Mean squared error (if reference signal available)
    pub mse: Option<f64>,
    /// Peak signal-to-noise ratio (if reference signal available)
    pub psnr: Option<f64>,
    /// Structural similarity index (if reference signal available)
    pub ssim: Option<f64>,
    /// Signal preservation metric (0-1, higher is better)
    pub signal_preservation: f64,
    /// Noise reduction metric (0-1, higher is better)
    pub noise_reduction: f64,
    /// Edge preservation metric (0-1, higher is better)
    pub edge_preservation: f64,
    /// Smoothness index metric (0-1, higher is better)
    pub smoothness_index: f64,
}

/// Unified denoising function
///
/// This function provides a single entry point for all denoising methods
/// available in the scirs2-signal library, with automatic method selection
/// and parameter optimization.
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `config` - Denoising configuration
/// * `reference` - Optional reference clean signal for quality assessment
///
/// # Returns
///
/// * Comprehensive denoising result with quality metrics
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::denoise_unified::{denoise_unified, UnifiedDenoisingConfig, DenoisingMethod};
/// use scirs2_signal::denoise::{ThresholdMethod, ThresholdSelect};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a noisy signal
/// let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0]);
///
/// // Use default wavelet denoising
/// let config = UnifiedDenoisingConfig::default();
/// let result = denoise_unified(&signal, &config, None).unwrap();
///
/// // Use advanced wavelet denoising
/// let advanced_config = UnifiedDenoisingConfig {
///     method: DenoisingMethod::WaveletAdvanced {
///         config: scirs2_signal: denoise, advanced: AdvancedDenoiseConfig::default(),
///     },
///     ..Default::default()
/// };
/// let advanced_result = denoise_unified(&signal, &advanced_config, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn denoise_unified(
    signal: &Array1<f64>,
    config: &UnifiedDenoisingConfig,
    reference: Option<&Array1<f64>>,
) -> SignalResult<UnifiedDenoisingResult> {
    let start_time = if config.benchmark {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // Preprocessing
    let preprocessed = if config.enable_preprocessing {
        preprocess_signal(signal)?
    } else {
        signal.clone()
    };

    // Apply denoising method
    let (denoised, estimated_noise_level) = match &config.method {
        DenoisingMethod::WaveletBasic {
            wavelet,
            levels,
            threshold_method,
            threshold_select,
        } => {
            let denoised_vec = denoise_wavelet(
                preprocessed.as_slice().unwrap(),
                *wavelet,
                *levels,
                *threshold_method,
                *threshold_select,
                config.noise_level,
            )?;
            let denoised = Array1::from_vec(denoised_vec);
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::WaveletAdvanced { config: adv_config } => {
            let result = advanced_denoise(preprocessed.as_slice().unwrap(), adv_config)?;
            (Array1::from_vec(result.signal), result.noise_level)
        }
        DenoisingMethod::DictionaryLearning {
            config: dict_config,
        } => {
            let denoised = denoise_dictionary_learning(&preprocessed, dict_config)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::Wiener {
            config: wiener_config,
        } => {
            let denoised = denoise_wiener_1d(&preprocessed, wiener_config)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::Median { window_size } => {
            let denoised = denoise_median_1d(&preprocessed, *window_size)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::TotalVariation { lambda, iterations } => {
            let tv_config = crate::denoise_enhanced::TotalVariationConfig {
                lambda: *lambda,
                max_iterations: *iterations,
                step_size: 0.1,
                tolerance: 1e-6,
            };
            let denoised = denoise_total_variation_1d(&preprocessed, &tv_config)?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
        DenoisingMethod::AdvancedAdvanced {
            config: advanced_config,
        } => {
            let result = advanced_advanced_denoise(&preprocessed, advanced_config)?;
            (
                result.denoised_signal,
                estimate_noise_level(&result.noise_estimate),
            )
        }
        DenoisingMethod::AdaptiveHybrid {
            fallback_methods,
            quality_threshold,
            max_iterations,
        } => {
            let result = denoise_adaptive_hybrid(
                &preprocessed,
                fallback_methods,
                *quality_threshold,
                *max_iterations,
                config.noise_level,
            )?;
            (result.denoised, result.estimated_noise_level)
        }
        DenoisingMethod::NonLocalMeans {
            search_window,
            patch_size,
            filtering_strength,
        } => {
            let denoised = denoise_non_local_means(
                &preprocessed,
                *search_window,
                *patch_size,
                *filtering_strength,
            )?;
            let noise_level = config
                .noise_level
                .unwrap_or_else(|| estimate_noise_level(&preprocessed));
            (denoised, noise_level)
        }
    };

    // Postprocessing
    let final_denoised = if config.enable_postprocessing {
        postprocess_signal(&denoised)?
    } else {
        denoised
    };

    // Calculate processing time
    let processing_time_ms = start_time.map(|start| start.elapsed().as_secs_f64() * 1000.0);

    // Calculate quality metrics
    let quality_metrics = calculate_quality_metrics(signal, &final_denoised, reference);

    // Calculate SNR improvement
    let snr_improvement = calculate_snr_improvement(signal, &final_denoised);

    Ok(UnifiedDenoisingResult {
        denoised: final_denoised,
        estimated_noise_level,
        snr_improvement,
        method_used: config.method.clone(),
        processing_time_ms,
        quality_metrics,
    })
}

/// Automatically select the best denoising method for a given signal
///
/// This function analyzes the input signal characteristics and automatically
/// selects the most appropriate denoising method.
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `noise_level` - Optional known noise level
///
/// # Returns
///
/// * Recommended denoising configuration
#[allow(dead_code)]
pub fn auto_select_denoising_method(
    signal: &Array1<f64>,
    noise_level: Option<f64>,
) -> SignalResult<UnifiedDenoisingConfig> {
    // Analyze signal characteristics
    let signal_length = signal.len();
    let signal_complexity = analyze_signal_complexity(signal);
    let estimated_noise = noise_level.unwrap_or_else(|| estimate_noise_level(signal));

    // Select method based on characteristics
    let method = if signal_length < 256 {
        // Short signals: use median filtering or simple wavelet
        if estimated_noise > 0.1 {
            DenoisingMethod::Median { window_size: 3 }
        } else {
            DenoisingMethod::WaveletBasic {
                wavelet: Wavelet::Haar,
                levels: Some(3),
                threshold_method: ThresholdMethod::Soft,
                threshold_select: ThresholdSelect::Universal,
            }
        }
    } else if signal_complexity > 0.7 {
        // Complex signals: use advanced methods
        DenoisingMethod::WaveletAdvanced {
            config: AdvancedDenoiseConfig {
                translation_invariant: true,
                adaptive: true,
                noise_estimation: NoiseEstimation::MAD,
                ..Default::default()
            },
        }
    } else if estimated_noise > 0.05 {
        // High noise: use robust methods
        DenoisingMethod::TotalVariation {
            lambda: 0.1,
            iterations: 100,
        }
    } else {
        // Low noise: use gentle wavelet denoising
        DenoisingMethod::WaveletBasic {
            wavelet: Wavelet::DB(4),
            levels: Some(4),
            threshold_method: ThresholdMethod::Soft,
            threshold_select: ThresholdSelect::Sure,
        }
    };

    Ok(UnifiedDenoisingConfig {
        method,
        noise_level,
        enable_preprocessing: true,
        enable_postprocessing: true,
        benchmark: false,
    })
}

/// Preprocess signal before denoising
#[allow(dead_code)]
fn preprocess_signal(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // Basic preprocessing: remove DC offset
    let mean = signal.mean().unwrap_or(0.0);
    Ok(_signal.mapv(|x| x - mean))
}

/// Postprocess signal after denoising
#[allow(dead_code)]
fn postprocess_signal(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // Basic postprocessing: ensure no NaN or infinite values
    let cleaned = signal.mapv(|x| if x.is_finite() { x } else { 0.0 });
    Ok(cleaned)
}

/// Estimate noise level from signal
#[allow(dead_code)]
fn estimate_noise_level(signal: &Array1<f64>) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }

    // Use differences between adjacent samples as noise estimate
    let mut diffs = Vec::with_capacity(_signal.len() - 1);
    for i in 0.._signal.len() - 1 {
        diffs.push((_signal[i + 1] - signal[i]).abs());
    }

    // Use median of differences as robust noise estimate
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    diffs[diffs.len() / 2] * 0.6745 // Scale factor for Gaussian noise
}

/// Analyze signal complexity
#[allow(dead_code)]
fn analyze_signal_complexity(signal: &Array1<f64>) -> f64 {
    if signal.len() < 4 {
        return 0.0;
    }

    // Calculate second derivative as complexity measure
    let mut complexity_sum = 0.0;
    for i in 1.._signal.len() - 1 {
        let second_deriv = signal[i + 1] - 2.0 * signal[i] + signal[i - 1];
        complexity_sum += second_deriv.abs();
    }

    let signal_range = signal.mapv(|x| x.abs()).into_iter().fold(0.0, f64::max);
    if signal_range > 0.0 {
        (complexity_sum / ((_signal.len() - 2) as f64)) / signal_range
    } else {
        0.0
    }
}

/// Calculate quality metrics
#[allow(dead_code)]
fn calculate_quality_metrics(
    original: &Array1<f64>,
    denoised: &Array1<f64>,
    reference: Option<&Array1<f64>>,
) -> QualityMetrics {
    let mse = reference.map(|ref_sig| {
        if ref_sig.len() == denoised.len() {
            ref_sig
                .iter()
                .zip(denoised.iter())
                .map(|(&r, &d)| (r - d).powi(2))
                .sum::<f64>()
                / ref_sig.len() as f64
        } else {
            f64::NAN
        }
    });

    let psnr = mse.map(|mse_val| {
        if mse_val > 0.0 {
            let max_val = reference
                .unwrap()
                .mapv(|x| x.abs())
                .into_iter()
                .fold(0.0, f64::max);
            20.0 * (max_val / mse_val.sqrt()).log10()
        } else {
            f64::INFINITY
        }
    });

    // Simple signal preservation metric
    let signal_power = original.mapv(|x| x * x).sum();
    let denoised_power = denoised.mapv(|x| x * x).sum();
    let signal_preservation = if signal_power > 0.0 {
        (denoised_power / signal_power).min(1.0)
    } else {
        1.0
    };

    // Simple noise reduction metric based on high-frequency content reduction
    let original_hf = calculate_high_frequency_energy(original);
    let denoised_hf = calculate_high_frequency_energy(denoised);
    let noise_reduction = if original_hf > 0.0 {
        ((original_hf - denoised_hf) / original_hf)
            .max(0.0)
            .min(1.0)
    } else {
        0.0
    };

    QualityMetrics {
        mse,
        psnr,
        ssim: None, // Complex SSIM calculation not implemented here
        signal_preservation,
        noise_reduction,
        edge_preservation: 0.8, // Placeholder value
        smoothness_index: 0.7,  // Placeholder value
    }
}

/// Calculate high-frequency energy (simple approximation)
#[allow(dead_code)]
fn calculate_high_frequency_energy(signal: &Array1<f64>) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }

    _signal
        .windows(2)
        .into_iter()
        .map(|window| (window[1] - window[0]).powi(2))
        .sum()
}

/// Calculate SNR improvement
#[allow(dead_code)]
fn calculate_snr_improvement(original: &Array1<f64>, denoised: &Array1<f64>) -> Option<f64> {
    if original.len() != denoised.len() {
        return None;
    }

    // Estimate noise as difference
    let noise: Array1<f64> = _original - denoised;

    let signal_power = denoised.mapv(|x| x * x).sum() / denoised.len() as f64;
    let noise_power = noise.mapv(|x| x * x).sum() / noise.len() as f64;

    if noise_power > 0.0 && signal_power > 0.0 {
        Some(10.0 * (signal_power / noise_power).log10())
    } else {
        None
    }
}

/// Adaptive hybrid denoising result
#[derive(Debug, Clone)]
pub struct AdaptiveHybridResult {
    /// Denoised signal
    pub denoised: Array1<f64>,
    /// Estimated noise level
    pub estimated_noise_level: f64,
    /// Method that achieved the best result
    pub best_method: DenoisingMethod,
    /// Quality score of the best result
    pub best_quality_score: f64,
}

/// Advanced adaptive hybrid denoising with automatic method selection and validation
///
/// This function tries multiple denoising methods and automatically selects the best one
/// based on quality metrics and signal characteristics.
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `fallback_methods` - List of denoising methods to try
/// * `quality_threshold` - Minimum quality score required
/// * `max_iterations` - Maximum number of method attempts
/// * `known_noise_level` - Optional known noise level
///
/// # Returns
///
/// * Best denoising result found
#[allow(dead_code)]
pub fn denoise_adaptive_hybrid(
    signal: &Array1<f64>,
    fallback_methods: &[DenoisingMethod],
    quality_threshold: f64,
    max_iterations: usize,
    known_noise_level: Option<f64>,
) -> SignalResult<AdaptiveHybridResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if fallback_methods.is_empty() {
        return Err(SignalError::ValueError(
            "No fallback _methods provided".to_string(),
        ));
    }

    let mut best_result: Option<AdaptiveHybridResult> = None;
    let mut best_score = f64::NEG_INFINITY;

    // Try each method up to max_iterations
    for (iteration, method) in fallback_methods
        .iter()
        .cycle()
        .take(max_iterations)
        .enumerate()
    {
        // Create configuration for this method
        let config = UnifiedDenoisingConfig {
            method: method.clone(),
            noise_level: known_noise_level,
            enable_preprocessing: true,
            enable_postprocessing: true,
            benchmark: false,
        };

        // Try denoising with this method
        match denoise_unified(signal, &config, None) {
            Ok(result) => {
                // Calculate quality score based on multiple metrics
                let quality_score = calculate_adaptive_quality_score(
                    signal,
                    &result.denoised,
                    result.estimated_noise_level,
                    &result.quality_metrics,
                );

                // Check if this is the best result so far
                if quality_score > best_score {
                    best_score = quality_score;
                    best_result = Some(AdaptiveHybridResult {
                        denoised: result.denoised,
                        estimated_noise_level: result.estimated_noise_level,
                        best_method: method.clone(),
                        best_quality_score: quality_score,
                    });

                    // Early termination if quality _threshold is met
                    if quality_score >= quality_threshold {
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Method {:?} failed in iteration {}: {}",
                    method, iteration, e
                );
                continue;
            }
        }
    }

    match best_result {
        Some(result) => Ok(result),
        None => Err(SignalError::ComputationError(
            "All denoising _methods failed".to_string(),
        )),
    }
}

/// Calculate adaptive quality score for method selection
#[allow(dead_code)]
fn calculate_adaptive_quality_score(
    original: &Array1<f64>,
    denoised: &Array1<f64>,
    estimated_noise_level: f64,
    quality_metrics: &QualityMetrics,
) -> f64 {
    // Weighted combination of multiple quality factors
    let mut score = 0.0;

    // Signal preservation (30% weight)
    score += 0.3 * quality_metrics.signal_preservation;

    // Noise reduction (25% weight)
    score += 0.25 * quality_metrics.noise_reduction;

    // Edge preservation (20% weight)
    score += 0.2 * quality_metrics.edge_preservation;

    // Smoothness preservation (15% weight)
    score += 0.15 * quality_metrics.smoothness_index;

    // Artifact detection (10% weight) - penalty for artifacts
    let artifact_penalty = detect_artifacts(original, denoised);
    score += 0.1 * (1.0 - artifact_penalty);

    // Bonus for reasonable noise _level estimation
    if estimated_noise_level > 0.0 && estimated_noise_level < 10.0 {
        score += 0.05; // Small bonus
    }

    score.max(0.0).min(1.0)
}

/// Detect artifacts in denoised signal
#[allow(dead_code)]
fn detect_artifacts(original: &Array1<f64>, denoised: &Array1<f64>) -> f64 {
    if original.len() != denoised.len() {
        return 1.0; // Maximum penalty for length mismatch
    }

    let n = original.len();
    if n < 3 {
        return 0.0; // Cannot detect artifacts in very short signals
    }

    let mut artifact_score = 0.0;
    let mut checks = 0;

    // Check for excessive smoothing (loss of high-frequency content)
    let mut orig_hf_energy = 0.0;
    let mut denoised_hf_energy = 0.0;

    for i in 1..n {
        let orig_diff = (_original[i] - original[i - 1]).abs();
        let denoised_diff = (denoised[i] - denoised[i - 1]).abs();

        orig_hf_energy += orig_diff;
        denoised_hf_energy += denoised_diff;
    }

    if orig_hf_energy > 1e-10 {
        let hf_reduction = 1.0 - (denoised_hf_energy / orig_hf_energy);
        if hf_reduction > 0.8 {
            // More than 80% high-frequency reduction might indicate over-smoothing
            artifact_score += 0.3 * (hf_reduction - 0.8) / 0.2;
        }
        checks += 1;
    }

    // Check for ringing artifacts (oscillations around edges)
    let mut ringing_score = 0.0;
    for i in 2..n - 2 {
        // Look for significant changes in _original signal
        let orig_change = (_original[i + 1] - original[i - 1]).abs();
        if orig_change > 0.1 {
            // Check for oscillations in denoised signal around this point
            let denoised_oscillation =
                ((denoised[i - 2] - denoised[i]) + (denoised[i + 2] - denoised[i])).abs();
            let orig_smoothness =
                ((_original[i - 2] - original[i]) + (_original[i + 2] - original[i])).abs();

            if orig_smoothness > 1e-10 && denoised_oscillation > orig_smoothness * 2.0 {
                ringing_score += 1.0;
            }
        }
    }

    if n > 4 {
        ringing_score /= (n - 4) as f64;
        artifact_score += 0.2 * ringing_score.min(1.0);
        checks += 1;
    }

    // Check for unrealistic values
    let orig_max = original.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let orig_min = original.iter().cloned().fold(f64::INFINITY, f64::min);
    let denoised_max = denoised.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let denoised_min = denoised.iter().cloned().fold(f64::INFINITY, f64::min);

    if denoised_max > orig_max * 1.2 || denoised_min < orig_min * 1.2 {
        artifact_score += 0.2; // Penalty for values outside expected range
    }
    checks += 1;

    if checks > 0 {
        artifact_score / checks as f64
    } else {
        0.0
    }
}

/// Non-local means denoising for 1D signals
///
/// Implements the non-local means algorithm adapted for 1D signals.
/// This method preserves repetitive patterns and textures better than local methods.
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `search_window` - Size of the search window for finding similar patches
/// * `patch_size` - Size of patches to compare
/// * `filtering_strength` - Filtering parameter (higher = more denoising)
///
/// # Returns
///
/// * Denoised signal
#[allow(dead_code)]
pub fn denoise_non_local_means(
    signal: &Array1<f64>,
    search_window: usize,
    patch_size: usize,
    filtering_strength: f64,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    if n == 0 {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if patch_size >= n {
        return Err(SignalError::ValueError(
            "Patch _size must be smaller than signal length".to_string(),
        ));
    }

    if search_window > n {
        return Err(SignalError::ValueError(
            "Search _window must not exceed signal length".to_string(),
        ));
    }

    let mut denoised = Array1::zeros(n);
    let h_sq = filtering_strength * filtering_strength;

    for i in 0..n {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        // Define search range around current position
        let search_start = if i >= search_window / 2 {
            i - search_window / 2
        } else {
            0
        };
        let search_end = (i + search_window / 2 + 1).min(n);

        // Search for similar patches
        for j in search_start..search_end {
            // Calculate patch distance
            let distance = calculate_patch_distance(signal, i, j, patch_size);

            // Calculate weight using Gaussian kernel
            let weight = (-distance / h_sq).exp();

            weighted_sum += weight * signal[j];
            weight_sum += weight;
        }

        // Normalize
        if weight_sum > 1e-10 {
            denoised[i] = weighted_sum / weight_sum;
        } else {
            denoised[i] = signal[i]; // Fallback to original value
        }
    }

    Ok(denoised)
}

/// Calculate distance between patches centered at positions i and j
#[allow(dead_code)]
fn calculate_patch_distance(_signal: &Array1<f64>, i: usize, j: usize, patchsize: usize) -> f64 {
    let n = signal.len();
    let half_patch = patch_size / 2;

    let i_start = if i >= half_patch { i - half_patch } else { 0 };
    let i_end = (i + half_patch + 1).min(n);
    let j_start = if j >= half_patch { j - half_patch } else { 0 };
    let j_end = (j + half_patch + 1).min(n);

    let mut distance = 0.0;
    let mut count = 0;

    // Compare overlapping parts of patches
    let start_offset = (i_start as i32 - j_start as i32).abs() as usize;
    let min_len = (i_end - i_start).min(j_end - j_start);

    for k in start_offset..min_len {
        if i_start + k < n && j_start + k < n {
            let diff = signal[i_start + k] - signal[j_start + k];
            distance += diff * diff;
            count += 1;
        }
    }

    if count > 0 {
        distance / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_unified_denoising_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple test signal
        let n = 64;
        let signal: Array1<f64> = Array1::from_shape_fn(n, |i| {
            (2.0 * PI * i as f64 / n as f64 * 4.0).sin() + 0.1 * (i as f64 * 0.1).sin()
        });

        let config = UnifiedDenoisingConfig::default();
        let result = denoise_unified(&signal, &config, None).unwrap();

        assert_eq!(result.denoised.len(), signal.len());
        assert!(result.estimated_noise_level >= 0.0);
        assert!(result.quality_metrics.signal_preservation >= 0.0);
        assert!(result.quality_metrics.signal_preservation <= 1.0);
    }

    #[test]
    fn test_auto_select_denoising_method() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Test with a simple signal
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0]);
        let config = auto_select_denoising_method(&signal, None).unwrap();

        // Should select a method appropriate for short signals
        match config.method {
            DenoisingMethod::Median { .. } | DenoisingMethod::WaveletBasic { .. } => {}
            _ => panic!("Unexpected method selection for short signal"),
        }
    }

    #[test]
    fn test_noise_level_estimation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a signal with known characteristics
        let clean_signal = Array1::from_vec(vec![1.0; 100]);
        let noise_level = estimate_noise_level(&clean_signal);

        // Should detect very low noise in constant signal
        assert!(noise_level < 0.1);
    }
}
