//! Configuration structures for time series feature extraction
//!
//! This module contains all configuration structures used to customize
//! feature extraction algorithms and their parameters.

/// Configuration for turning points detection and analysis
#[derive(Debug, Clone)]
pub struct TurningPointsConfig {
    /// Minimum relative threshold for detecting turning points
    pub min_turning_point_threshold: f64,
    /// Window size for local extrema detection
    pub extrema_window_size: usize,
    /// Threshold for major vs minor trend reversals
    pub major_reversal_threshold: f64,
    /// Enable detection of advanced patterns (double peaks, head-shoulders, etc.)
    pub detect_advanced_patterns: bool,
    /// Smoothing window sizes for multi-scale analysis
    pub smoothing_windows: Vec<usize>,
    /// Calculate temporal autocorrelation of turning points
    pub calculate_temporal_patterns: bool,
    /// Maximum lag for turning point autocorrelation
    pub max_autocorr_lag: usize,
    /// Enable clustering analysis of turning points
    pub analyze_clustering: bool,
    /// Minimum sequence length for momentum persistence
    pub min_sequence_length: usize,
    /// Enable multi-scale turning point analysis
    pub multiscale_analysis: bool,
}

impl Default for TurningPointsConfig {
    fn default() -> Self {
        Self {
            min_turning_point_threshold: 0.01, // 1% relative threshold
            extrema_window_size: 3,            // 3-point window for local extrema
            major_reversal_threshold: 0.05,    // 5% threshold for major reversals
            detect_advanced_patterns: true,
            smoothing_windows: vec![3, 5, 7, 10, 15], // Multiple smoothing scales
            calculate_temporal_patterns: true,
            max_autocorr_lag: 20,
            analyze_clustering: true,
            min_sequence_length: 3,
            multiscale_analysis: true,
        }
    }
}

/// Configuration for advanced spectral analysis feature calculation
#[derive(Debug, Clone)]
pub struct SpectralAnalysisConfig {
    // Power Spectral Density estimation parameters
    /// Calculate Welch's method PSD
    pub calculate_welch_psd: bool,
    /// Calculate periodogram PSD
    pub calculate_periodogram_psd: bool,
    /// Calculate autoregressive PSD
    pub calculate_ar_psd: bool,
    /// Window length for Welch's method (as fraction of signal length)
    pub welch_window_length_factor: f64,
    /// Overlap for Welch's method (as fraction of window length)
    pub welch_overlap_factor: f64,
    /// Order for autoregressive PSD estimation
    pub ar_order: usize,

    // Spectral peak detection parameters
    /// Enable spectral peak detection
    pub detect_spectral_peaks: bool,
    /// Minimum peak height (as fraction of max power)
    pub min_peak_height: f64,
    /// Minimum peak distance (in frequency bins)
    pub min_peak_distance: usize,
    /// Peak prominence threshold
    pub peak_prominence_threshold: f64,
    /// Maximum number of peaks to detect
    pub max_peaks: usize,

    // Frequency band analysis parameters
    /// Enable standard EEG frequency band analysis
    pub calculate_eeg_bands: bool,
    /// Enable custom frequency band analysis
    pub calculate_custom_bands: bool,
    /// Custom frequency band boundaries (in Hz or normalized units)
    pub custom_band_boundaries: Vec<f64>,
    /// Enable relative band power calculation
    pub calculate_relative_band_powers: bool,
    /// Enable band power ratio calculation
    pub calculate_band_ratios: bool,

    // Spectral entropy and information measures
    /// Calculate spectral Shannon entropy
    pub calculate_spectral_shannon_entropy: bool,
    /// Calculate spectral Rényi entropy
    pub calculate_spectral_renyi_entropy: bool,
    /// Rényi entropy alpha parameter
    pub renyi_alpha: f64,
    /// Calculate spectral permutation entropy
    pub calculate_spectral_permutation_entropy: bool,
    /// Permutation order for spectral permutation entropy
    pub spectral_permutation_order: usize,
    /// Calculate spectral sample entropy
    pub calculate_spectral_sample_entropy: bool,
    /// Sample entropy tolerance for spectral domain
    pub spectral_sample_entropy_tolerance: f64,
    /// Calculate spectral complexity measures
    pub calculate_spectral_complexity: bool,

    // Spectral shape and distribution measures
    /// Calculate spectral flatness (Wiener entropy)
    pub calculate_spectral_flatness: bool,
}

impl Default for SpectralAnalysisConfig {
    fn default() -> Self {
        Self {
            // Power Spectral Density estimation parameters
            calculate_welch_psd: true,
            calculate_periodogram_psd: true,
            calculate_ar_psd: false,          // More expensive
            welch_window_length_factor: 0.25, // 25% of signal length
            welch_overlap_factor: 0.5,        // 50% overlap
            ar_order: 10,

            // Spectral peak detection parameters
            detect_spectral_peaks: true,
            min_peak_height: 0.1, // 10% of max power
            min_peak_distance: 2,
            peak_prominence_threshold: 0.05,
            max_peaks: 20,

            // Frequency band analysis parameters
            calculate_eeg_bands: true,
            calculate_custom_bands: false,
            custom_band_boundaries: vec![], // Empty by default
            calculate_relative_band_powers: true,
            calculate_band_ratios: true,

            // Spectral entropy and information measures
            calculate_spectral_shannon_entropy: true,
            calculate_spectral_renyi_entropy: false,
            renyi_alpha: 2.0,
            calculate_spectral_permutation_entropy: false,
            spectral_permutation_order: 3,
            calculate_spectral_sample_entropy: false,
            spectral_sample_entropy_tolerance: 0.2,
            calculate_spectral_complexity: true,

            // Spectral shape and distribution measures
            calculate_spectral_flatness: true,
        }
    }
}

/// Configuration for enhanced periodogram analysis
#[derive(Debug, Clone)]
pub struct EnhancedPeriodogramConfig {
    // Advanced Periodogram Methods
    /// Enable Bartlett's method (averaged periodograms)
    pub enable_bartlett_method: bool,
    /// Number of segments for Bartlett's method
    pub bartlett_num_segments: usize,
    /// Enable enhanced Welch's method
    pub enable_enhanced_welch: bool,
    /// Enable multitaper periodogram using Thomson's method
    pub enable_multitaper: bool,
    /// Number of tapers for multitaper method
    pub multitaper_num_tapers: usize,
    /// Time-bandwidth product for multitaper
    pub multitaper_bandwidth: f64,
    /// Enable Blackman-Tukey periodogram
    pub enable_blackman_tukey: bool,
    /// Maximum lag for Blackman-Tukey method (fraction of signal length)
    pub blackman_tukey_max_lag_factor: f64,
    /// Enable Capon's minimum variance method
    pub enable_capon_method: bool,
    /// Enable MUSIC (Multiple Signal Classification) method
    pub enable_music_method: bool,
    /// Number of signal sources for MUSIC method
    pub music_num_sources: usize,
    /// Enable enhanced autoregressive periodogram
    pub enable_enhanced_ar: bool,
    /// Enhanced AR model order
    pub enhanced_ar_order: usize,

    // Window Analysis and Optimization
    /// Enable window analysis and optimization
    pub enable_window_analysis: bool,
    /// Primary window type to use
    pub primary_window_type: String,
    /// Enable automatic window selection
    pub enable_auto_window_selection: bool,
    /// Window selection criteria
    pub window_selection_criteria: String,
    /// Calculate window effectiveness metrics
    pub calculate_window_effectiveness: bool,
    /// Calculate spectral leakage measures
    pub calculate_spectral_leakage: bool,
    /// Leakage threshold for warnings
    pub spectral_leakage_threshold: f64,

    // Cross-Periodogram Analysis
    /// Enable cross-periodogram analysis
    pub enable_cross_periodogram: bool,
    /// Enable coherence function calculation
    pub enable_coherence_analysis: bool,
    /// Coherence confidence level
    pub coherence_confidence_level: f64,
    /// Enable phase spectrum analysis
    pub enable_phase_spectrum: bool,
    /// Phase unwrapping method
    pub phase_unwrapping_method: String,
    /// Calculate cross-correlation from periodogram
    pub calculate_periodogram_xcorr: bool,
    /// Maximum lag for cross-correlation analysis
    pub xcorr_max_lag: usize,

    // Statistical Analysis and Confidence
    /// Enable confidence interval calculation
    pub enable_confidence_intervals: bool,
    /// Confidence level for intervals (e.g., 0.95)
    pub confidence_level: f64,
    /// Enable statistical significance testing
    pub enable_significance_testing: bool,
    /// Significance testing method
    pub significance_test_method: String,
    /// Enable goodness-of-fit testing
    pub enable_goodness_of_fit: bool,
    /// Null hypothesis spectral model
    pub null_hypothesis_model: String,
    /// Enable variance and bias estimation
    pub enable_variance_bias_estimation: bool,

    // Bias Correction and Variance Reduction
    /// Enable bias correction methods
    pub enable_bias_correction: bool,
    /// Enable variance reduction methods
    pub enable_variance_reduction: bool,
    /// Enable smoothing methods
    pub enable_smoothing: bool,
    /// Enable zero padding for frequency resolution enhancement
    pub enable_zero_padding: bool,
}

impl Default for EnhancedPeriodogramConfig {
    fn default() -> Self {
        Self {
            // Advanced Periodogram Methods
            enable_bartlett_method: true,
            bartlett_num_segments: 8,
            enable_enhanced_welch: true,
            enable_multitaper: false, // More expensive
            multitaper_num_tapers: 4,
            multitaper_bandwidth: 4.0,
            enable_blackman_tukey: false, // More expensive
            blackman_tukey_max_lag_factor: 0.25,
            enable_capon_method: false, // More expensive
            enable_music_method: false, // Most expensive
            music_num_sources: 1,
            enable_enhanced_ar: true,
            enhanced_ar_order: 10,

            // Window Analysis and Optimization
            enable_window_analysis: true,
            primary_window_type: "Hanning".to_string(),
            enable_auto_window_selection: false,
            window_selection_criteria: "MinSidelobes".to_string(),
            calculate_window_effectiveness: true,
            calculate_spectral_leakage: true,
            spectral_leakage_threshold: 0.1,

            // Cross-Periodogram Analysis
            enable_cross_periodogram: false,
            enable_coherence_analysis: false,
            coherence_confidence_level: 0.95,
            enable_phase_spectrum: false,
            phase_unwrapping_method: "Quality".to_string(),
            calculate_periodogram_xcorr: false,
            xcorr_max_lag: 50,

            // Statistical Analysis and Confidence
            enable_confidence_intervals: false,
            confidence_level: 0.95,
            enable_significance_testing: false,
            significance_test_method: "WhiteNoise".to_string(),
            enable_goodness_of_fit: false,
            null_hypothesis_model: "WhiteNoise".to_string(),
            enable_variance_bias_estimation: false,

            // Bias Correction and Variance Reduction
            enable_bias_correction: false,
            enable_variance_reduction: false,
            enable_smoothing: false,
            enable_zero_padding: false,
        }
    }
}

/// Configuration for entropy-based feature calculation
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Calculate classical entropy measures (Shannon, Rényi, Tsallis)
    pub calculate_classical_entropy: bool,
    /// Calculate differential entropy measures (ApEn, SampEn, PermEn)
    pub calculate_differential_entropy: bool,
    /// Calculate multiscale entropy measures
    pub calculate_multiscale_entropy: bool,
    /// Calculate conditional and joint entropy measures
    pub calculate_conditional_entropy: bool,
    /// Calculate spectral entropy measures
    pub calculate_spectral_entropy: bool,
    /// Calculate time-frequency entropy measures
    pub calculate_timefrequency_entropy: bool,
    /// Calculate symbolic entropy measures
    pub calculate_symbolic_entropy: bool,
    /// Calculate distribution-based entropy measures
    pub calculate_distribution_entropy: bool,
    /// Calculate complexity and regularity measures
    pub calculate_complexity_measures: bool,
    /// Calculate fractal and scaling entropy measures
    pub calculate_fractal_entropy: bool,
    /// Calculate cross-scale entropy measures
    pub calculate_crossscale_entropy: bool,

    // Parameters for entropy calculations
    /// Number of bins for discretization (for classical entropy)
    pub n_bins: usize,
    /// Embedding dimension for approximate entropy
    pub embedding_dimension: usize,
    /// Tolerance for approximate entropy (as fraction of std dev)
    pub tolerance_fraction: f64,
    /// Order for permutation entropy
    pub permutation_order: usize,
    /// Maximum lag for conditional entropy
    pub max_lag: usize,
    /// Number of scales for multiscale entropy
    pub n_scales: usize,
    /// Rényi entropy parameter α
    pub renyi_alpha: f64,
    /// Tsallis entropy parameter q
    pub tsallis_q: f64,
    /// Number of symbols for symbolic encoding
    pub n_symbols: usize,
    /// Use fast approximations for expensive calculations
    pub use_fast_approximations: bool,
    /// Window size for instantaneous entropy
    pub instantaneous_window_size: usize,
    /// Overlap for instantaneous entropy windows
    pub instantaneous_overlap: f64,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            calculate_classical_entropy: true,
            calculate_differential_entropy: true,
            calculate_multiscale_entropy: true,
            calculate_conditional_entropy: true,
            calculate_spectral_entropy: true,
            calculate_timefrequency_entropy: false, // Expensive
            calculate_symbolic_entropy: true,
            calculate_distribution_entropy: true,
            calculate_complexity_measures: true,
            calculate_fractal_entropy: false,    // Expensive
            calculate_crossscale_entropy: false, // Expensive

            n_bins: 10,
            embedding_dimension: 2,
            tolerance_fraction: 0.2,
            permutation_order: 3,
            max_lag: 5,
            n_scales: 5,
            renyi_alpha: 2.0,
            tsallis_q: 2.0,
            n_symbols: 3,
            use_fast_approximations: false,
            instantaneous_window_size: 50,
            instantaneous_overlap: 0.5,
        }
    }
}

/// Configuration for wavelet analysis
#[derive(Debug, Clone)]
pub struct WaveletConfig {
    /// Wavelet family to use
    pub family: WaveletFamily,
    /// Number of decomposition levels
    pub levels: usize,
    /// Whether to calculate CWT features
    pub calculate_cwt: bool,
    /// CWT scale range (min, max)
    pub cwt_scales: Option<(f64, f64)>,
    /// Number of CWT scales
    pub cwt_scale_count: usize,
    /// Whether to calculate denoising-based features
    pub calculate_denoising: bool,
    /// Denoising threshold method
    pub denoising_method: DenoisingMethod,
}

impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            family: WaveletFamily::Daubechies(4),
            levels: 5,
            calculate_cwt: false,
            cwt_scales: None,
            cwt_scale_count: 32,
            calculate_denoising: false,
            denoising_method: DenoisingMethod::Soft,
        }
    }
}

/// Configuration for multi-window analysis
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Small window size (high temporal resolution)
    pub small_window_size: usize,
    /// Medium window size (balanced resolution)
    pub medium_window_size: usize,
    /// Large window size (low temporal resolution)
    pub large_window_size: usize,
    /// Whether to calculate cross-window correlations
    pub calculate_cross_correlations: bool,
    /// Whether to perform change detection
    pub detect_changes: bool,
    /// Whether to calculate Bollinger bands
    pub calculate_bollinger_bands: bool,
    /// Whether to calculate MACD features
    pub calculate_macd: bool,
    /// Whether to calculate RSI
    pub calculate_rsi: bool,
    /// RSI period
    pub rsi_period: usize,
    /// MACD fast period
    pub macd_fast_period: usize,
    /// MACD slow period
    pub macd_slow_period: usize,
    /// MACD signal period
    pub macd_signal_period: usize,
    /// Bollinger band standard deviations
    pub bollinger_std_dev: f64,
    /// EWMA smoothing factor
    pub ewma_alpha: f64,
    /// Change detection threshold
    pub change_threshold: f64,
    /// Change detection threshold (alias for compatibility)
    pub change_detection_threshold: f64,
    /// Bollinger band window size
    pub bollinger_window: usize,
    /// Bollinger band multiplier (alias for std_dev)
    pub bollinger_multiplier: f64,
    /// Normalization window size
    pub normalization_window: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            small_window_size: 5,
            medium_window_size: 20,
            large_window_size: 50,
            calculate_cross_correlations: true,
            detect_changes: true,
            calculate_bollinger_bands: true,
            calculate_macd: true,
            calculate_rsi: true,
            rsi_period: 14,
            macd_fast_period: 12,
            macd_slow_period: 26,
            macd_signal_period: 9,
            bollinger_std_dev: 2.0,
            ewma_alpha: 0.1,
            change_threshold: 2.0,
            change_detection_threshold: 2.0,
            bollinger_window: 20,
            bollinger_multiplier: 2.0,
            normalization_window: 20,
        }
    }
}

/// Configuration for expanded statistical analysis
#[derive(Debug, Clone)]
pub struct ExpandedStatisticalConfig {
    /// Enable higher-order moments calculation (5th and 6th moments)
    pub calculate_higher_order_moments: bool,
    /// Enable robust statistics (trimmed means, winsorized mean, MAD)
    pub calculate_robust_statistics: bool,
    /// Enable percentile-based measures (P5, P10, P90, P95, P99)
    pub calculate_percentiles: bool,
    /// Enable distribution characteristics (L-moments, skewness variants)
    pub calculate_distribution_characteristics: bool,
    /// Enable tail statistics (outlier counts, tail ratios)
    pub calculate_tail_statistics: bool,
    /// Enable central tendency variations (harmonic, geometric, quadratic means)
    pub calculate_central_tendency_variations: bool,
    /// Enable advanced variability measures
    pub calculate_variability_measures: bool,
    /// Enable normality tests (Jarque-Bera, Anderson-Darling, etc.)
    pub calculate_normality_tests: bool,
    /// Enable advanced shape measures (biweight, Qn/Sn estimators)
    pub calculate_advanced_shape_measures: bool,
    /// Enable count-based statistics (zero crossings, local extrema)
    pub calculate_count_statistics: bool,
    /// Enable concentration measures (Herfindahl, Shannon diversity)
    pub calculate_concentration_measures: bool,
    /// Trimming fraction for trimmed means (default: 0.1 for 10% trimming)
    pub trimming_fraction_10: f64,
    /// Trimming fraction for trimmed means (default: 0.2 for 20% trimming)
    pub trimming_fraction_20: f64,
    /// Winsorizing fraction (default: 0.05 for 5% winsorizing)
    pub winsorizing_fraction: f64,
    /// Number of bins for mode approximation (default: sqrt(n))
    pub mode_bins: Option<usize>,
    /// Confidence level for normality tests (default: 0.05)
    pub normality_alpha: f64,
    /// Whether to use fast approximations for computationally expensive measures
    pub use_fast_approximations: bool,
}

impl Default for ExpandedStatisticalConfig {
    fn default() -> Self {
        Self {
            // Enable all categories by default for comprehensive analysis
            calculate_higher_order_moments: true,
            calculate_robust_statistics: true,
            calculate_percentiles: true,
            calculate_distribution_characteristics: true,
            calculate_tail_statistics: true,
            calculate_central_tendency_variations: true,
            calculate_variability_measures: true,
            calculate_normality_tests: true,
            calculate_advanced_shape_measures: true,
            calculate_count_statistics: true,
            calculate_concentration_measures: true,

            // Default parameter values
            trimming_fraction_10: 0.1,
            trimming_fraction_20: 0.2,
            winsorizing_fraction: 0.05,
            mode_bins: None, // Use sqrt(n) by default
            normality_alpha: 0.05,
            use_fast_approximations: false,
        }
    }
}

/// Configuration for Empirical Mode Decomposition (EMD) analysis
#[derive(Debug, Clone)]
pub struct EMDConfig {
    /// Maximum number of IMFs to extract
    pub max_imfs: usize,
    /// Stopping criterion for sifting (standard deviation)
    pub sifting_tolerance: f64,
    /// Maximum number of sifting iterations per IMF
    pub max_sifting_iterations: usize,
    /// Whether to calculate Hilbert spectral features
    pub calculate_hilbert_spectrum: bool,
    /// Whether to calculate instantaneous features
    pub calculate_instantaneous: bool,
    /// Whether to calculate EMD-based entropies
    pub calculate_emd_entropies: bool,
    /// Interpolation method for envelope generation
    pub interpolation_method: InterpolationMethod,
    /// Edge effect handling method
    pub edge_method: EdgeMethod,
}

impl Default for EMDConfig {
    fn default() -> Self {
        Self {
            max_imfs: 10,
            sifting_tolerance: 0.2,
            max_sifting_iterations: 100,
            calculate_hilbert_spectrum: true,
            calculate_instantaneous: true,
            calculate_emd_entropies: false,
            interpolation_method: InterpolationMethod::CubicSpline,
            edge_method: EdgeMethod::Mirror,
        }
    }
}

/// Wavelet family types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Daubechies wavelets (db1-db10)
    Daubechies(usize),
    /// Haar wavelet (simplest case of Daubechies)
    Haar,
    /// Biorthogonal wavelets
    Biorthogonal(usize, usize),
    /// Coiflets wavelets
    Coiflets(usize),
    /// Morlet wavelet (for CWT)
    Morlet,
    /// Mexican hat wavelet (Ricker)
    MexicanHat,
}

/// Denoising threshold methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoisingMethod {
    /// Hard thresholding
    Hard,
    /// Soft thresholding
    Soft,
    /// Sure thresholding
    Sure,
    /// Minimax thresholding
    Minimax,
}

/// Window types for spectral analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Rectangular window
    Rectangular,
    /// Hanning window
    Hanning,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser(f64), // Beta parameter
}

/// Interpolation methods for EMD envelope generation  
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Cubic spline interpolation
    CubicSpline,
    /// Linear interpolation
    Linear,
    /// Piecewise cubic Hermite interpolation
    Pchip,
}

/// Edge effect handling methods for EMD
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMethod {
    /// Mirror reflection at boundaries
    Mirror,
    /// Zero padding at boundaries
    ZeroPadding,
    /// Extend with constant values
    Constant,
    /// Polynomial extrapolation
    Polynomial,
}

/// Configuration for general feature extraction
#[derive(Debug, Clone, Default)]
pub struct FeatureExtractionOptions {
    /// Turning points configuration
    pub turning_points: TurningPointsConfig,
    /// Enhanced periodogram configuration
    pub enhanced_periodogram: EnhancedPeriodogramConfig,
    /// Entropy features configuration
    pub entropy: EntropyConfig,
    /// Spectral analysis configuration
    pub spectral_analysis: SpectralAnalysisConfig,
    /// Window-based features configuration
    pub window: WindowConfig,
    /// Expanded statistical features configuration
    pub expanded_statistical: ExpandedStatisticalConfig,
    /// Wavelet features configuration
    pub wavelet: WaveletConfig,
    /// EMD features configuration
    pub emd: EMDConfig,
}
