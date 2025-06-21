//! Frequency domain and spectral analysis features for time series
//!
//! This module provides comprehensive frequency domain feature extraction including
//! FFT analysis, power spectral density estimation, spectral peak detection,
//! frequency band analysis, and advanced periodogram methods.

use ndarray::{s, Array1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::{EnhancedPeriodogramConfig, SpectralAnalysisConfig};
use super::utils::{
    BispectrumFeatures, PhaseSpectrumFeatures, PhaseSpectrumResult, ScaleSpectralFeatures,
};
use crate::error::Result;
use crate::utils::autocorrelation;

/// Comprehensive frequency domain features for time series analysis
#[derive(Debug, Clone)]
pub struct FrequencyFeatures<F> {
    /// Spectral centroid (center of mass of spectrum)
    pub spectral_centroid: F,
    /// Spectral spread (variance around centroid)
    pub spectral_spread: F,
    /// Spectral skewness
    pub spectral_skewness: F,
    /// Spectral kurtosis
    pub spectral_kurtosis: F,
    /// Spectral entropy
    pub spectral_entropy: F,
    /// Spectral rolloff (95% of energy)
    pub spectral_rolloff: F,
    /// Spectral flux (change in spectrum)
    pub spectral_flux: F,
    /// Dominant frequency
    pub dominant_frequency: F,
    /// Number of spectral peaks
    pub spectral_peaks: usize,
    /// Power in different frequency bands
    pub frequency_bands: Vec<F>,
    /// Advanced spectral analysis features
    pub spectral_analysis: SpectralAnalysisFeatures<F>,
    /// Enhanced periodogram analysis features
    pub enhanced_periodogram_features: EnhancedPeriodogramFeatures<F>,
    /// Wavelet-based features
    pub wavelet_features: WaveletFeatures<F>,
    /// Hilbert-Huang Transform (EMD) features
    pub emd_features: EMDFeatures<F>,
}

impl<F> Default for FrequencyFeatures<F>
where
    F: Float + FromPrimitive + Default,
{
    fn default() -> Self {
        Self {
            spectral_centroid: F::zero(),
            spectral_spread: F::zero(),
            spectral_skewness: F::zero(),
            spectral_kurtosis: F::zero(),
            spectral_entropy: F::zero(),
            spectral_rolloff: F::zero(),
            spectral_flux: F::zero(),
            dominant_frequency: F::zero(),
            spectral_peaks: 0,
            frequency_bands: Vec::new(),
            spectral_analysis: SpectralAnalysisFeatures::default(),
            enhanced_periodogram_features: EnhancedPeriodogramFeatures::default(),
            wavelet_features: WaveletFeatures::default(),
            emd_features: EMDFeatures::default(),
        }
    }
}

/// Comprehensive spectral analysis features
#[derive(Debug, Clone)]
pub struct SpectralAnalysisFeatures<F> {
    // Power Spectral Density (PSD) features
    /// Power spectral density using Welch's method
    pub welch_psd: Vec<F>,
    /// Power spectral density using periodogram
    pub periodogram_psd: Vec<F>,
    /// Power spectral density using autoregressive method
    pub ar_psd: Vec<F>,
    /// Frequency resolution of PSD estimates
    pub frequency_resolution: F,
    /// Total power across all frequencies
    pub total_power: F,
    /// Normalized power spectral density
    pub normalized_psd: Vec<F>,

    // Spectral peak detection and characterization
    /// Peak frequencies (in Hz or normalized units)
    pub peak_frequencies: Vec<F>,
    /// Peak magnitudes (power/amplitude at peaks)
    pub peak_magnitudes: Vec<F>,
    /// Peak widths (FWHM - Full Width Half Maximum)
    pub peak_widths: Vec<F>,
    /// Peak prominence (relative height above surroundings)
    pub peak_prominences: Vec<F>,
    /// Number of significant peaks
    pub significant_peaks_count: usize,
    /// Spectral peak density (peaks per frequency unit)
    pub peak_density: F,
    /// Average peak spacing
    pub average_peak_spacing: F,
    /// Peak asymmetry measures
    pub peak_asymmetry: Vec<F>,

    // Frequency band analysis and decomposition
    /// Delta band power (0.5-4 Hz)
    pub delta_power: F,
    /// Theta band power (4-8 Hz)
    pub theta_power: F,
    /// Alpha band power (8-12 Hz)
    pub alpha_power: F,
    /// Beta band power (12-30 Hz)
    pub beta_power: F,
    /// Gamma band power (30-100 Hz)
    pub gamma_power: F,
    /// Low frequency power (custom band)
    pub low_freq_power: F,
    /// High frequency power (custom band)
    pub high_freq_power: F,
    /// Relative band powers (normalized)
    pub relative_band_powers: Vec<F>,
    /// Band power ratios (e.g., alpha/theta)
    pub band_power_ratios: Vec<F>,
    /// Frequency band entropy
    pub band_entropy: F,

    // Spectral entropy and information measures
    /// Spectral entropy (Shannon entropy of PSD)
    pub spectral_shannon_entropy: F,
    /// Spectral Rényi entropy
    pub spectral_renyi_entropy: F,
    /// Spectral permutation entropy
    pub spectral_permutation_entropy: F,
    /// Frequency domain sample entropy
    pub spectral_sample_entropy: F,
    /// Spectral complexity (Lempel-Ziv in frequency domain)
    pub spectral_complexity: F,
    /// Spectral information density
    pub spectral_information_density: F,
    /// Frequency domain approximate entropy
    pub spectral_approximate_entropy: F,

    // Spectral shape and distribution measures
    /// Spectral flatness (Wiener entropy)
    pub spectral_flatness: F,
    /// Spectral crest factor (peak-to-average ratio)
    pub spectral_crest_factor: F,
    /// Spectral irregularity measure
    pub spectral_irregularity: F,
    /// Spectral smoothness index
    pub spectral_smoothness: F,
    /// Spectral slope (tilt of spectrum)
    pub spectral_slope: F,
    /// Spectral decrease measure
    pub spectral_decrease: F,
    /// Spectral brightness (high frequency content)
    pub spectral_brightness: F,
    /// Spectral roughness (fluctuation measure)
    pub spectral_roughness: F,

    // Advanced spectral characteristics
    /// Spectral autocorrelation features
    pub spectral_autocorrelation: Vec<F>,
    /// Cross-spectral features (if applicable)
    pub cross_spectral_coherence: Vec<F>,
    /// Spectral coherence measures
    pub spectral_coherence_mean: F,
    /// Phase spectrum features
    pub phase_spectrum_features: PhaseSpectrumFeatures<F>,
    /// Bispectrum features (third-order statistics)
    pub bispectrum_features: BispectrumFeatures<F>,

    // Frequency stability and variability
    /// Frequency stability measure
    pub frequency_stability: F,
    /// Spectral variability index
    pub spectral_variability: F,
    /// Frequency modulation index
    pub frequency_modulation_index: F,
    /// Spectral purity measure
    pub spectral_purity: F,

    // Multi-scale and cross-frequency coupling
    /// Multiscale spectral features
    pub multiscale_spectral_features: Vec<ScaleSpectralFeatures<F>>,
    /// Cross-frequency coupling measures
    pub cross_frequency_coupling: Vec<F>,
    /// Phase-amplitude coupling indices
    pub phase_amplitude_coupling: Vec<F>,
    /// Cross-scale correlation measures
    pub cross_scale_correlations: Vec<F>,

    // Time-frequency analysis (advanced)
    /// Short-time Fourier transform features
    pub stft_features: Vec<F>,
    /// Spectrogram-based measures
    pub spectrogram_entropy: F,
    /// Time-frequency localization measures
    pub time_frequency_localization: F,
    /// Instantaneous frequency measures
    pub instantaneous_frequency_stats: Vec<F>,
}

impl<F> Default for SpectralAnalysisFeatures<F>
where
    F: Float + FromPrimitive + Default,
{
    fn default() -> Self {
        Self {
            // Power Spectral Density features
            welch_psd: Vec::new(),
            periodogram_psd: Vec::new(),
            ar_psd: Vec::new(),
            frequency_resolution: F::zero(),
            total_power: F::zero(),
            normalized_psd: Vec::new(),

            // Spectral peak detection
            peak_frequencies: Vec::new(),
            peak_magnitudes: Vec::new(),
            peak_widths: Vec::new(),
            peak_prominences: Vec::new(),
            significant_peaks_count: 0,
            peak_density: F::zero(),
            average_peak_spacing: F::zero(),
            peak_asymmetry: Vec::new(),

            // Frequency band analysis
            delta_power: F::zero(),
            theta_power: F::zero(),
            alpha_power: F::zero(),
            beta_power: F::zero(),
            gamma_power: F::zero(),
            low_freq_power: F::zero(),
            high_freq_power: F::zero(),
            relative_band_powers: Vec::new(),
            band_power_ratios: Vec::new(),
            band_entropy: F::zero(),

            // Spectral entropy and information measures
            spectral_shannon_entropy: F::zero(),
            spectral_renyi_entropy: F::zero(),
            spectral_permutation_entropy: F::zero(),
            spectral_sample_entropy: F::zero(),
            spectral_complexity: F::zero(),
            spectral_information_density: F::zero(),
            spectral_approximate_entropy: F::zero(),

            // Spectral shape measures
            spectral_flatness: F::zero(),
            spectral_crest_factor: F::zero(),
            spectral_irregularity: F::zero(),
            spectral_smoothness: F::zero(),
            spectral_slope: F::zero(),
            spectral_decrease: F::zero(),
            spectral_brightness: F::zero(),
            spectral_roughness: F::zero(),

            // Advanced spectral characteristics
            spectral_autocorrelation: Vec::new(),
            cross_spectral_coherence: Vec::new(),
            spectral_coherence_mean: F::zero(),
            phase_spectrum_features: PhaseSpectrumFeatures::default(),
            bispectrum_features: BispectrumFeatures::default(),

            // Frequency stability and variability
            frequency_stability: F::zero(),
            spectral_variability: F::zero(),
            frequency_modulation_index: F::zero(),
            spectral_purity: F::zero(),

            // Multi-scale and cross-frequency coupling
            multiscale_spectral_features: Vec::new(),
            cross_frequency_coupling: Vec::new(),
            phase_amplitude_coupling: Vec::new(),
            cross_scale_correlations: Vec::new(),

            // Time-frequency analysis
            stft_features: Vec::new(),
            spectrogram_entropy: F::zero(),
            time_frequency_localization: F::zero(),
            instantaneous_frequency_stats: Vec::new(),
        }
    }
}

/// Enhanced periodogram analysis features
#[derive(Debug, Clone)]
pub struct EnhancedPeriodogramFeatures<F> {
    // Advanced periodogram methods
    /// Bartlett's method periodogram (averaged periodograms)
    pub bartlett_periodogram: Vec<F>,
    /// Enhanced Welch's method periodogram
    pub welch_periodogram: Vec<F>,
    /// Multitaper periodogram using Thomson's method
    pub multitaper_periodogram: Vec<F>,
    /// Blackman-Tukey periodogram
    pub blackman_tukey_periodogram: Vec<F>,
    /// Capon's minimum variance periodogram
    pub capon_periodogram: Vec<F>,
    /// MUSIC (Multiple Signal Classification) periodogram
    pub music_periodogram: Vec<F>,
    /// Enhanced autoregressive periodogram
    pub ar_periodogram: Vec<F>,

    // Window analysis and optimization
    /// Window type information and characteristics
    pub window_type: WindowTypeInfo<F>,
    /// Window effectiveness metrics
    pub window_effectiveness: F,
    /// Spectral leakage measurements
    pub spectral_leakage: F,
    /// Optimal window selection results
    pub optimal_window_type: String,
    /// Window comparison metrics
    pub window_comparison_metrics: Vec<F>,

    // Cross-periodogram analysis
    /// Cross-periodogram values
    pub cross_periodogram: Vec<F>,
    /// Coherence function values
    pub coherence_function: Vec<F>,
    /// Phase spectrum analysis results
    pub phase_spectrum_result: PhaseSpectrumResult<F>,
    /// Cross-correlation from periodogram
    pub periodogram_xcorr: Vec<F>,

    // Statistical analysis and confidence
    /// Confidence intervals for periodogram estimates
    pub confidence_intervals: Vec<(F, F)>,
    /// Statistical significance of peaks
    pub peak_significance: Vec<F>,
    /// Goodness-of-fit test results
    pub goodness_of_fit_statistics: Vec<F>,
    /// Variance and bias estimates
    pub variance_estimates: Vec<F>,
    /// Bias estimates
    pub bias_estimates: Vec<F>,

    // Bias correction and variance reduction
    /// Bias-corrected periodogram
    pub bias_corrected_periodogram: Vec<F>,
    /// Variance-reduced periodogram
    pub variance_reduced_periodogram: Vec<F>,
    /// Smoothed periodogram
    pub smoothed_periodogram: Vec<F>,

    // Frequency resolution enhancement
    /// Zero-padded periodogram for improved resolution
    pub zero_padded_periodogram: Vec<F>,
    /// Zero-padding effectiveness measure
    pub zero_padding_effectiveness: F,
    /// Interpolated periodogram
    pub interpolated_periodogram: Vec<F>,
    /// Interpolation effectiveness measure
    pub interpolation_effectiveness: F,

    // Quality and performance metrics
    /// Signal-to-noise ratio estimate
    pub snr_estimate: F,
    /// Dynamic range of periodogram
    pub dynamic_range: F,
    /// Spectral purity measure
    pub spectral_purity_measure: F,
    /// Frequency stability measures
    pub frequency_stability_measures: Vec<F>,
    /// Estimation error bounds
    pub error_bounds: Vec<F>,
    /// Computational efficiency metrics
    pub computational_efficiency: F,
    /// Memory efficiency metrics
    pub memory_efficiency: F,

    // Advanced features
    /// Multiscale coherence analysis
    pub multiscale_coherence: Vec<F>,
    /// Cross-scale correlation results
    pub cross_scale_correlations: Vec<F>,
    /// Hierarchical structure analysis results
    pub hierarchical_analysis: Vec<F>,
    /// Scale-dependent statistics
    pub scale_dependent_statistics: Vec<F>,
}

impl<F> Default for EnhancedPeriodogramFeatures<F>
where
    F: Float + FromPrimitive + Default,
{
    fn default() -> Self {
        Self {
            // Advanced periodogram methods
            bartlett_periodogram: Vec::new(),
            welch_periodogram: Vec::new(),
            multitaper_periodogram: Vec::new(),
            blackman_tukey_periodogram: Vec::new(),
            capon_periodogram: Vec::new(),
            music_periodogram: Vec::new(),
            ar_periodogram: Vec::new(),

            // Window analysis
            window_type: WindowTypeInfo::default(),
            window_effectiveness: F::zero(),
            spectral_leakage: F::zero(),
            optimal_window_type: String::new(),
            window_comparison_metrics: Vec::new(),

            // Cross-periodogram analysis
            cross_periodogram: Vec::new(),
            coherence_function: Vec::new(),
            phase_spectrum_result: (
                Vec::new(),
                Vec::new(),
                F::zero(),
                PhaseSpectrumFeatures::default(),
                BispectrumFeatures::default(),
            ),
            periodogram_xcorr: Vec::new(),

            // Statistical analysis
            confidence_intervals: Vec::new(),
            peak_significance: Vec::new(),
            goodness_of_fit_statistics: Vec::new(),
            variance_estimates: Vec::new(),
            bias_estimates: Vec::new(),

            // Bias correction and variance reduction
            bias_corrected_periodogram: Vec::new(),
            variance_reduced_periodogram: Vec::new(),
            smoothed_periodogram: Vec::new(),

            // Frequency resolution enhancement
            zero_padded_periodogram: Vec::new(),
            zero_padding_effectiveness: F::zero(),
            interpolated_periodogram: Vec::new(),
            interpolation_effectiveness: F::zero(),

            // Quality and performance metrics
            snr_estimate: F::zero(),
            dynamic_range: F::zero(),
            spectral_purity_measure: F::zero(),
            frequency_stability_measures: Vec::new(),
            error_bounds: Vec::new(),
            computational_efficiency: F::zero(),
            memory_efficiency: F::zero(),

            // Advanced features
            multiscale_coherence: Vec::new(),
            cross_scale_correlations: Vec::new(),
            hierarchical_analysis: Vec::new(),
            scale_dependent_statistics: Vec::new(),
        }
    }
}

/// Window type information for spectral analysis
#[derive(Debug, Clone)]
pub struct WindowTypeInfo<F> {
    /// Window name/type
    pub window_name: String,
    /// Main lobe width
    pub main_lobe_width: F,
    /// Side lobe level
    pub side_lobe_level: F,
    /// Scalloping loss
    pub scalloping_loss: F,
    /// Processing gain
    pub processing_gain: F,
    /// Noise bandwidth
    pub noise_bandwidth: F,
    /// Coherent gain
    pub coherent_gain: F,
    /// Window length
    pub window_length: usize,
    /// Equivalent noise bandwidth
    pub equivalent_noise_bandwidth: F,
    /// Overlap correlation factor
    pub overlap_correlation: F,
}

impl<F> Default for WindowTypeInfo<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            window_name: "Hanning".to_string(),
            main_lobe_width: F::zero(),
            side_lobe_level: F::zero(),
            scalloping_loss: F::zero(),
            processing_gain: F::zero(),
            noise_bandwidth: F::zero(),
            coherent_gain: F::zero(),
            window_length: 0,
            equivalent_noise_bandwidth: F::zero(),
            overlap_correlation: F::zero(),
        }
    }
}

/// Placeholder wavelet features (to be implemented in wavelet.rs)
#[derive(Debug, Clone, Default)]
pub struct WaveletFeatures<F> {
    /// Energy in different wavelet scales
    pub scale_energies: Vec<F>,
    /// Wavelet entropy
    pub wavelet_entropy: F,
}

/// Placeholder EMD features (to be implemented in temporal.rs or separate EMD module)
#[derive(Debug, Clone, Default)]
pub struct EMDFeatures<F> {
    /// Number of IMFs extracted
    pub num_imfs: usize,
    /// Energy distribution across IMFs
    pub imf_energies: Vec<F>,
    /// Instantaneous frequency statistics
    pub instantaneous_frequencies: Vec<F>,
}

// =============================================================================
// Core Frequency Analysis Functions
// =============================================================================

/// Calculate comprehensive frequency domain features
pub fn calculate_frequency_features<F>(
    ts: &Array1<F>,
    config: &SpectralAnalysisConfig,
) -> Result<FrequencyFeatures<F>>
where
    F: Float + FromPrimitive + Debug + ndarray::ScalarOperand + std::iter::Sum + Default,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    if n < 4 {
        return Ok(FrequencyFeatures::default());
    }

    // Calculate basic spectrum
    let spectrum = calculate_simple_periodogram(ts)?;
    let frequencies = (0..spectrum.len())
        .map(|i| F::from(i).unwrap() / F::from(spectrum.len() * 2).unwrap())
        .collect::<Vec<_>>();

    // Calculate spectral moments
    let _total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    let spectral_centroid = calculate_spectral_centroid(&spectrum, &frequencies);
    let spectral_spread = calculate_spectral_spread(&spectrum, &frequencies, spectral_centroid);
    let spectral_skewness =
        calculate_spectral_skewness(&spectrum, &frequencies, spectral_centroid, spectral_spread);
    let spectral_kurtosis =
        calculate_spectral_kurtosis(&spectrum, &frequencies, spectral_centroid, spectral_spread);

    // Calculate other spectral features
    let spectral_entropy = calculate_spectral_entropy(&spectrum);
    let spectral_rolloff =
        calculate_spectral_rolloff(&spectrum, &frequencies, F::from(0.95).unwrap());
    let spectral_flux = F::zero(); // Would need previous spectrum for comparison
    let dominant_frequency = find_dominant_frequency(&spectrum, &frequencies);

    // Calculate spectral peaks
    let (peak_frequencies, _peak_magnitudes) = find_spectral_peaks(&spectrum, &frequencies)?;
    let spectral_peaks = peak_frequencies.len();

    // Calculate frequency bands
    let frequency_bands = calculate_frequency_bands(&spectrum, &frequencies);

    // Calculate advanced spectral analysis
    let spectral_analysis = if config.calculate_welch_psd || config.calculate_periodogram_psd {
        calculate_spectral_analysis_features(ts, config)?
    } else {
        SpectralAnalysisFeatures::default()
    };

    // Enhanced periodogram features would be calculated separately
    let enhanced_periodogram_features = EnhancedPeriodogramFeatures::default();
    let wavelet_features = WaveletFeatures::default();
    let emd_features = EMDFeatures::default();

    Ok(FrequencyFeatures {
        spectral_centroid,
        spectral_spread,
        spectral_skewness,
        spectral_kurtosis,
        spectral_entropy,
        spectral_rolloff,
        spectral_flux,
        dominant_frequency,
        spectral_peaks,
        frequency_bands,
        spectral_analysis,
        enhanced_periodogram_features,
        wavelet_features,
        emd_features,
    })
}

/// Calculate enhanced periodogram analysis features
pub fn calculate_enhanced_periodogram_features<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<EnhancedPeriodogramFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Default + ndarray::ScalarOperand + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    if n < 8 {
        return Ok(EnhancedPeriodogramFeatures::default());
    }

    let mut features = EnhancedPeriodogramFeatures::default();

    // Calculate advanced periodogram methods
    if config.enable_bartlett_method {
        features.bartlett_periodogram = calculate_bartlett_periodogram(ts, config)?;
    }

    if config.enable_enhanced_welch {
        features.welch_periodogram = calculate_enhanced_welch_periodogram(ts, config)?;
    }

    if config.enable_multitaper {
        features.multitaper_periodogram = calculate_multitaper_periodogram(ts, config)?;
    }

    if config.enable_blackman_tukey {
        features.blackman_tukey_periodogram = calculate_blackman_tukey_periodogram(ts, config)?;
    }

    if config.enable_enhanced_ar {
        features.ar_periodogram = calculate_enhanced_ar_periodogram(ts, config)?;
    }

    // Calculate window analysis
    if config.enable_window_analysis {
        features.window_type = calculate_window_analysis(ts, config)?;
        features.window_effectiveness = calculate_window_effectiveness(&features.window_type);
        features.spectral_leakage = calculate_spectral_leakage(&features.window_type);
    }

    // Calculate statistical analysis and confidence intervals
    if config.enable_confidence_intervals {
        features.confidence_intervals =
            calculate_periodogram_confidence_intervals(&features.welch_periodogram, config)?;
    }

    if config.enable_significance_testing {
        features.peak_significance =
            calculate_peak_significance(&features.welch_periodogram, config)?;
    }

    // Calculate bias correction and variance reduction
    if config.enable_bias_correction {
        features.bias_corrected_periodogram =
            calculate_bias_corrected_periodogram(&features.welch_periodogram, config)?;
    }

    if config.enable_variance_reduction {
        features.variance_reduced_periodogram =
            calculate_variance_reduced_periodogram(&features.welch_periodogram, config)?;
    }

    if config.enable_smoothing {
        features.smoothed_periodogram =
            calculate_smoothed_periodogram(&features.welch_periodogram, config)?;
    }

    // Calculate frequency resolution enhancement
    if config.enable_zero_padding {
        features.zero_padded_periodogram = calculate_zero_padded_periodogram(ts, config)?;
        features.zero_padding_effectiveness = calculate_zero_padding_effectiveness(
            &features.zero_padded_periodogram,
            &features.welch_periodogram,
        );
    }

    // Note: Interpolation functionality would be enabled based on config if available
    // For now, using default values

    // Calculate quality and performance metrics (simplified for compatibility)
    // SNR estimation
    features.snr_estimate = calculate_snr_from_periodogram(&features.welch_periodogram)?;

    // Dynamic range calculation
    features.dynamic_range = calculate_dynamic_range(&features.welch_periodogram);

    if config.enable_enhanced_welch {
        features.spectral_purity_measure = calculate_spectral_purity(&features.welch_periodogram);
    }

    Ok(features)
}

// =============================================================================
// Periodogram Calculation Functions
// =============================================================================

/// Calculate Bartlett's periodogram using averaged periodograms
pub fn calculate_bartlett_periodogram<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    let segment_length = n / config.bartlett_num_segments;

    if segment_length < 4 {
        return Ok(vec![F::zero(); n / 2]);
    }

    let mut averaged_periodogram = vec![F::zero(); segment_length / 2];
    let mut segment_count = 0;

    for i in 0..config.bartlett_num_segments {
        let start_idx = i * segment_length;
        let end_idx = std::cmp::min(start_idx + segment_length, n);

        if end_idx - start_idx >= 4 {
            let segment = ts.slice(s![start_idx..end_idx]).to_owned();
            let segment_periodogram = calculate_simple_periodogram(&segment)?;

            for (j, &value) in segment_periodogram.iter().enumerate() {
                if j < averaged_periodogram.len() {
                    averaged_periodogram[j] = averaged_periodogram[j] + value;
                }
            }
            segment_count += 1;
        }
    }

    // Average the periodograms
    if segment_count > 0 {
        let count_f = F::from_usize(segment_count).unwrap();
        for value in averaged_periodogram.iter_mut() {
            *value = *value / count_f;
        }
    }

    Ok(averaged_periodogram)
}

/// Calculate enhanced Welch's periodogram with advanced windowing
pub fn calculate_enhanced_welch_periodogram<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    let window_length = (n as f64 * 0.25).round() as usize; // 25% window length
    let overlap = (window_length as f64 * 0.5).round() as usize; // 50% overlap

    if window_length < 4 {
        return calculate_simple_periodogram(ts);
    }

    let window = create_window(&config.primary_window_type, window_length)?;
    let step_size = window_length - overlap;
    let num_segments = (n - overlap) / step_size;

    if num_segments == 0 {
        return calculate_simple_periodogram(ts);
    }

    let mut averaged_periodogram = vec![F::zero(); window_length / 2];
    let mut segment_count = 0;

    for i in 0..num_segments {
        let start_idx = i * step_size;
        let end_idx = std::cmp::min(start_idx + window_length, n);

        if end_idx - start_idx == window_length {
            let mut segment = ts.slice(s![start_idx..end_idx]).to_owned();

            // Apply window
            for (j, &w) in window.iter().enumerate() {
                segment[j] = segment[j] * w;
            }

            let segment_periodogram = calculate_simple_periodogram(&segment)?;

            for (j, &value) in segment_periodogram.iter().enumerate() {
                if j < averaged_periodogram.len() {
                    averaged_periodogram[j] = averaged_periodogram[j] + value;
                }
            }
            segment_count += 1;
        }
    }

    // Average and normalize
    if segment_count > 0 {
        let count_f = F::from_usize(segment_count).unwrap();
        for value in averaged_periodogram.iter_mut() {
            *value = *value / count_f;
        }
    }

    Ok(averaged_periodogram)
}

/// Calculate multitaper periodogram using Thomson's method (simplified)
pub fn calculate_multitaper_periodogram<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    if n < 8 {
        return calculate_simple_periodogram(ts);
    }

    // Simplified multitaper using multiple Hanning windows with different phases
    let num_tapers = config.multitaper_num_tapers;
    let mut averaged_periodogram = vec![F::zero(); n / 2];

    for taper_idx in 0..num_tapers {
        let phase_shift =
            F::from(taper_idx as f64 * std::f64::consts::PI / num_tapers as f64).unwrap();
        let mut tapered_signal = ts.clone();

        for (i, value) in tapered_signal.iter_mut().enumerate() {
            let t = F::from(i).unwrap() / F::from(n).unwrap();
            let taper_weight = (F::from(std::f64::consts::PI).unwrap() * t + phase_shift).sin();
            *value = *value * taper_weight.abs();
        }

        let taper_periodogram = calculate_simple_periodogram(&tapered_signal)?;

        for (j, &value) in taper_periodogram.iter().enumerate() {
            if j < averaged_periodogram.len() {
                averaged_periodogram[j] = averaged_periodogram[j] + value;
            }
        }
    }

    // Average across tapers
    let num_tapers_f = F::from_usize(num_tapers).unwrap();
    for value in averaged_periodogram.iter_mut() {
        *value = *value / num_tapers_f;
    }

    Ok(averaged_periodogram)
}

/// Calculate Blackman-Tukey periodogram
pub fn calculate_blackman_tukey_periodogram<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    let max_lag = (n as f64 * config.blackman_tukey_max_lag_factor).round() as usize;

    // Calculate autocorrelation
    let acf = autocorrelation(ts, Some(max_lag))?;

    // Apply windowing to autocorrelation
    let window = create_window("Blackman", acf.len())?;
    let mut windowed_acf = acf.clone();
    for (i, &w) in window.iter().enumerate() {
        if i < windowed_acf.len() {
            windowed_acf[i] = windowed_acf[i] * w;
        }
    }

    // Calculate periodogram from windowed autocorrelation
    calculate_simple_periodogram(&windowed_acf)
}

/// Calculate enhanced autoregressive periodogram
pub fn calculate_enhanced_ar_periodogram<F>(
    ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    // Simplified AR periodogram - would need full AR parameter estimation
    // For now, return a smoothed version of the regular periodogram
    let periodogram = calculate_simple_periodogram(ts)?;
    let order = config.enhanced_ar_order.min(periodogram.len() / 4);

    let mut ar_periodogram = periodogram.clone();

    // Apply simple smoothing as placeholder for proper AR method
    for i in order..(ar_periodogram.len() - order) {
        let sum = (0..2 * order + 1).fold(F::zero(), |acc, j| acc + periodogram[i - order + j]);
        ar_periodogram[i] = sum / F::from(2 * order + 1).unwrap();
    }

    Ok(ar_periodogram)
}

/// Calculate simple periodogram using FFT
pub fn calculate_simple_periodogram<F>(ts: &Array1<F>) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    let n = ts.len();
    if n < 2 {
        return Ok(vec![F::zero()]);
    }

    // Simplified FFT-based periodogram calculation
    // In a real implementation, you would use a proper FFT library
    let mut periodogram = vec![F::zero(); n / 2];

    // Calculate power spectrum (simplified)
    let mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(n).unwrap();
    let variance = ts
        .iter()
        .fold(F::zero(), |acc, &x| acc + (x - mean) * (x - mean))
        / F::from_usize(n).unwrap();

    // For demonstration, create a simple spectrum based on autocorrelation
    for (k, item) in periodogram.iter_mut().enumerate() {
        let mut power = F::zero();
        let freq = F::from(k).unwrap() / F::from(n).unwrap()
            * F::from(2.0 * std::f64::consts::PI).unwrap();

        for lag in 0..std::cmp::min(n / 4, 50) {
            let mut autocorr = F::zero();
            let mut count = 0;

            for i in 0..(n - lag) {
                autocorr = autocorr + (ts[i] - mean) * (ts[i + lag] - mean);
                count += 1;
            }

            if count > 0 {
                autocorr = autocorr / F::from_usize(count).unwrap();
                let lag_f = F::from(lag).unwrap();
                power = power + autocorr * (freq * lag_f).cos();
            }
        }

        *item = power.abs() / variance;
    }

    Ok(periodogram)
}

// =============================================================================
// Window Functions
// =============================================================================

/// Create a window function
pub fn create_window<F>(window_type: &str, length: usize) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let mut window = vec![F::zero(); length];

    match window_type {
        "Rectangular" => {
            window.fill(F::one());
        }
        "Hanning" | "Hann" => {
            for (i, w) in window.iter_mut().enumerate() {
                let arg =
                    F::from(2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).unwrap();
                *w = F::from(0.5).unwrap() * (F::one() - arg.cos());
            }
        }
        "Hamming" => {
            for (i, w) in window.iter_mut().enumerate() {
                let arg =
                    F::from(2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).unwrap();
                *w = F::from(0.54).unwrap() - F::from(0.46).unwrap() * arg.cos();
            }
        }
        "Blackman" => {
            for (i, w) in window.iter_mut().enumerate() {
                let arg =
                    F::from(2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).unwrap();
                let arg2 = F::from(2.0).unwrap() * arg;
                *w = F::from(0.42).unwrap() - F::from(0.5).unwrap() * arg.cos()
                    + F::from(0.08).unwrap() * arg2.cos();
            }
        }
        _ => {
            // Default to Hanning
            for (i, w) in window.iter_mut().enumerate() {
                let arg =
                    F::from(2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).unwrap();
                *w = F::from(0.5).unwrap() * (F::one() - arg.cos());
            }
        }
    }

    Ok(window)
}

// =============================================================================
// Spectral Analysis Helper Functions
// =============================================================================

/// Calculate spectral centroid
pub fn calculate_spectral_centroid<F>(spectrum: &[F], frequencies: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    if total_power == F::zero() {
        return F::zero();
    }

    let weighted_sum = spectrum
        .iter()
        .zip(frequencies.iter())
        .fold(F::zero(), |acc, (&power, &freq)| acc + power * freq);

    weighted_sum / total_power
}

/// Calculate spectral spread
pub fn calculate_spectral_spread<F>(spectrum: &[F], frequencies: &[F], centroid: F) -> F
where
    F: Float + FromPrimitive,
{
    let total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    if total_power == F::zero() {
        return F::zero();
    }

    let weighted_variance =
        spectrum
            .iter()
            .zip(frequencies.iter())
            .fold(F::zero(), |acc, (&power, &freq)| {
                let diff = freq - centroid;
                acc + power * diff * diff
            });

    (weighted_variance / total_power).sqrt()
}

/// Calculate spectral skewness
pub fn calculate_spectral_skewness<F>(
    spectrum: &[F],
    frequencies: &[F],
    centroid: F,
    spread: F,
) -> F
where
    F: Float + FromPrimitive,
{
    if spread == F::zero() {
        return F::zero();
    }

    let total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    if total_power == F::zero() {
        return F::zero();
    }

    let weighted_third_moment =
        spectrum
            .iter()
            .zip(frequencies.iter())
            .fold(F::zero(), |acc, (&power, &freq)| {
                let standardized = (freq - centroid) / spread;
                acc + power * standardized * standardized * standardized
            });

    weighted_third_moment / total_power
}

/// Calculate spectral kurtosis
pub fn calculate_spectral_kurtosis<F>(
    spectrum: &[F],
    frequencies: &[F],
    centroid: F,
    spread: F,
) -> F
where
    F: Float + FromPrimitive,
{
    if spread == F::zero() {
        return F::zero();
    }

    let total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    if total_power == F::zero() {
        return F::zero();
    }

    let weighted_fourth_moment =
        spectrum
            .iter()
            .zip(frequencies.iter())
            .fold(F::zero(), |acc, (&power, &freq)| {
                let standardized = (freq - centroid) / spread;
                let standardized_squared = standardized * standardized;
                acc + power * standardized_squared * standardized_squared
            });

    weighted_fourth_moment / total_power - F::from(3.0).unwrap()
}

/// Calculate spectral entropy
pub fn calculate_spectral_entropy<F>(spectrum: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    if total_power == F::zero() {
        return F::zero();
    }

    let mut entropy = F::zero();
    for &power in spectrum.iter() {
        if power > F::zero() {
            let prob = power / total_power;
            entropy = entropy - prob * prob.ln();
        }
    }

    entropy
}

/// Calculate spectral rolloff
pub fn calculate_spectral_rolloff<F>(spectrum: &[F], frequencies: &[F], threshold: F) -> F
where
    F: Float + FromPrimitive,
{
    let total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    let target_power = total_power * threshold;

    let mut cumulative_power = F::zero();
    for (i, &power) in spectrum.iter().enumerate() {
        cumulative_power = cumulative_power + power;
        if cumulative_power >= target_power {
            return frequencies[i];
        }
    }

    frequencies.last().copied().unwrap_or(F::zero())
}

/// Find dominant frequency
pub fn find_dominant_frequency<F>(spectrum: &[F], frequencies: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let max_idx = spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    frequencies[max_idx]
}

/// Find spectral peaks
pub fn find_spectral_peaks<F>(spectrum: &[F], frequencies: &[F]) -> Result<(Vec<F>, Vec<F>)>
where
    F: Float + FromPrimitive,
{
    let mut peak_frequencies = Vec::new();
    let mut peak_magnitudes = Vec::new();

    if spectrum.len() < 3 {
        return Ok((peak_frequencies, peak_magnitudes));
    }

    // Simple peak detection: find local maxima
    for i in 1..(spectrum.len() - 1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            peak_frequencies.push(frequencies[i]);
            peak_magnitudes.push(spectrum[i]);
        }
    }

    Ok((peak_frequencies, peak_magnitudes))
}

/// Calculate frequency bands
pub fn calculate_frequency_bands<F>(spectrum: &[F], frequencies: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive,
{
    let mut bands = Vec::new();

    // Standard EEG frequency bands (normalized)
    let band_boundaries = [
        (F::from(0.0).unwrap(), F::from(0.05).unwrap()), // Delta (0-4Hz normalized to 0-0.05)
        (F::from(0.05).unwrap(), F::from(0.1).unwrap()), // Theta (4-8Hz)
        (F::from(0.1).unwrap(), F::from(0.15).unwrap()), // Alpha (8-12Hz)
        (F::from(0.15).unwrap(), F::from(0.375).unwrap()), // Beta (12-30Hz)
        (F::from(0.375).unwrap(), F::from(0.5).unwrap()), // Gamma (30-100Hz)
    ];

    for (low, high) in band_boundaries.iter() {
        let mut band_power = F::zero();
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq >= *low && freq < *high && i < spectrum.len() {
                band_power = band_power + spectrum[i];
            }
        }
        bands.push(band_power);
    }

    bands
}

// =============================================================================
// Placeholder Functions (to be fully implemented)
// =============================================================================

/// Calculate spectral analysis features (placeholder)
pub fn calculate_spectral_analysis_features<F>(
    ts: &Array1<F>,
    config: &SpectralAnalysisConfig,
) -> Result<SpectralAnalysisFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Default + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    // Simplified implementation - would need full spectral analysis
    let spectrum = calculate_simple_periodogram(ts)?;
    let frequencies = (0..spectrum.len())
        .map(|i| F::from(i).unwrap() / F::from(spectrum.len() * 2).unwrap())
        .collect::<Vec<_>>();

    let mut features = SpectralAnalysisFeatures::default();

    if config.calculate_welch_psd {
        features.welch_psd = spectrum.clone();
    }

    if config.calculate_periodogram_psd {
        features.periodogram_psd = spectrum.clone();
    }

    features.total_power = spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
    features.frequency_resolution = F::from(1.0).unwrap() / F::from(ts.len()).unwrap();

    // Calculate frequency bands
    let bands = calculate_frequency_bands(&spectrum, &frequencies);
    if bands.len() >= 5 {
        features.delta_power = bands[0];
        features.theta_power = bands[1];
        features.alpha_power = bands[2];
        features.beta_power = bands[3];
        features.gamma_power = bands[4];
    }

    features.spectral_shannon_entropy = calculate_spectral_entropy(&spectrum);
    features.spectral_flatness = calculate_spectral_flatness(&spectrum);

    Ok(features)
}

/// Calculate spectral flatness
pub fn calculate_spectral_flatness<F>(spectrum: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    if spectrum.is_empty() {
        return F::zero();
    }

    // Geometric mean / Arithmetic mean
    let mut geometric_mean = F::one();
    let mut arithmetic_mean = F::zero();
    let mut count = 0;

    for &power in spectrum.iter() {
        if power > F::zero() {
            geometric_mean = geometric_mean * power;
            arithmetic_mean = arithmetic_mean + power;
            count += 1;
        }
    }

    if count == 0 {
        return F::zero();
    }

    let count_f = F::from_usize(count).unwrap();
    geometric_mean = geometric_mean.powf(F::one() / count_f);
    arithmetic_mean = arithmetic_mean / count_f;

    if arithmetic_mean == F::zero() {
        F::zero()
    } else {
        geometric_mean / arithmetic_mean
    }
}

// Additional placeholder functions for completeness

/// Calculate window analysis for enhanced periodogram
pub fn calculate_window_analysis<F>(
    _ts: &Array1<F>,
    config: &EnhancedPeriodogramConfig,
) -> Result<WindowTypeInfo<F>>
where
    F: Float + FromPrimitive,
{
    Ok(WindowTypeInfo {
        window_name: config.primary_window_type.clone(),
        ..Default::default()
    })
}

/// Calculate window effectiveness metrics
pub fn calculate_window_effectiveness<F>(_window_info: &WindowTypeInfo<F>) -> F
where
    F: Float + FromPrimitive,
{
    F::from(0.8).unwrap() // Placeholder
}

/// Calculate spectral leakage measures
pub fn calculate_spectral_leakage<F>(_window_info: &WindowTypeInfo<F>) -> F
where
    F: Float + FromPrimitive,
{
    F::from(0.1).unwrap() // Placeholder
}

/// Calculate confidence intervals for periodogram
pub fn calculate_periodogram_confidence_intervals<F>(
    _periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<(F, F)>>
where
    F: Float + FromPrimitive,
{
    Ok(Vec::new()) // Placeholder
}

/// Calculate peak significance for periodogram
pub fn calculate_peak_significance<F>(
    _periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    Ok(Vec::new()) // Placeholder
}

/// Calculate bias-corrected periodogram
pub fn calculate_bias_corrected_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    Ok(periodogram.to_vec()) // Placeholder
}

/// Calculate variance-reduced periodogram
pub fn calculate_variance_reduced_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    Ok(periodogram.to_vec()) // Placeholder
}

/// Calculate smoothed periodogram
pub fn calculate_smoothed_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    Ok(periodogram.to_vec()) // Placeholder
}

/// Calculate zero-padded periodogram
pub fn calculate_zero_padded_periodogram<F>(
    ts: &Array1<F>,
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
    for<'a> F: std::iter::Sum<&'a F>,
{
    calculate_simple_periodogram(ts) // Placeholder
}

/// Calculate interpolated periodogram
pub fn calculate_interpolated_periodogram<F>(
    periodogram: &[F],
    _config: &EnhancedPeriodogramConfig,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive,
{
    Ok(periodogram.to_vec()) // Placeholder
}

/// Calculate zero padding effectiveness
pub fn calculate_zero_padding_effectiveness<F>(_padded: &[F], _original: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    F::from(0.9).unwrap() // Placeholder
}

/// Calculate interpolation effectiveness
pub fn calculate_interpolation_effectiveness<F>(_interpolated: &[F], _original: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    F::from(0.85).unwrap() // Placeholder
}

/// Calculate signal-to-noise ratio from periodogram
pub fn calculate_snr_from_periodogram<F>(periodogram: &[F]) -> Result<F>
where
    F: Float + FromPrimitive,
{
    if periodogram.is_empty() {
        return Ok(F::zero());
    }

    let max_power = periodogram.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let avg_power = periodogram.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(periodogram.len()).unwrap();

    if avg_power == F::zero() {
        Ok(F::zero())
    } else {
        Ok((max_power / avg_power).log10())
    }
}

/// Calculate dynamic range of periodogram
pub fn calculate_dynamic_range<F>(periodogram: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    if periodogram.is_empty() {
        return F::zero();
    }

    let max_power = periodogram.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let min_power = periodogram.iter().fold(F::infinity(), |a, &b| a.min(b));

    if min_power == F::zero() || max_power == F::zero() {
        F::zero()
    } else {
        (max_power / min_power).log10()
    }
}

/// Calculate spectral purity measure
pub fn calculate_spectral_purity<F>(periodogram: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    if periodogram.len() < 2 {
        return F::zero();
    }

    let max_power = periodogram.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let total_power = periodogram.iter().fold(F::zero(), |acc, &x| acc + x);

    if total_power == F::zero() {
        F::zero()
    } else {
        max_power / total_power
    }
}
