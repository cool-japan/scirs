//! Feature Extraction for Bifurcation Prediction
//!
//! This module contains feature extraction configurations for time series,
//! phase space, frequency domain, topological, and statistical features.

/// Feature extraction configuration
#[derive(Debug, Clone)]
pub struct FeatureExtraction {
    /// Time series features
    pub time_series_features: TimeSeriesFeatures,
    /// Phase space features
    pub phase_space_features: PhaseSpaceFeatures,
    /// Frequency domain features
    pub frequency_features: FrequencyFeatures,
    /// Topological features
    pub topological_features: TopologicalFeatures,
    /// Statistical features
    pub statistical_features: StatisticalFeatures,
    /// Feature normalization method
    pub normalization: FeatureNormalization,
}

/// Time series feature extraction
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures {
    /// Window size for feature extraction
    pub window_size: usize,
    /// Overlap between windows
    pub overlap: f64,
    /// Extract trend features
    pub trend_features: bool,
    /// Extract seasonality features
    pub seasonality_features: bool,
    /// Extract autocorrelation features
    pub autocorr_features: bool,
    /// Maximum lag for autocorrelation
    pub max_lag: usize,
    /// Extract change point features
    pub change_point_features: bool,
}

/// Phase space feature extraction
#[derive(Debug, Clone)]
pub struct PhaseSpaceFeatures {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Time delay for embedding
    pub time_delay: usize,
    /// Extract attractor features
    pub attractor_features: bool,
    /// Extract recurrence features
    pub recurrence_features: bool,
    /// Recurrence threshold
    pub recurrence_threshold: f64,
    /// Extract Poincaré map features
    pub poincare_features: bool,
}

/// Frequency domain features
#[derive(Debug, Clone)]
pub struct FrequencyFeatures {
    /// Extract power spectral density features
    pub psd_features: bool,
    /// Number of frequency bins
    pub frequency_bins: usize,
    /// Extract dominant frequency features
    pub dominant_freq_features: bool,
    /// Extract spectral entropy
    pub spectral_entropy: bool,
    /// Extract wavelet features
    pub wavelet_features: bool,
    /// Wavelet type
    pub wavelet_type: WaveletType,
}

/// Wavelet types for feature extraction
#[derive(Debug, Clone, Copy)]
pub enum WaveletType {
    Daubechies(usize),
    Morlet,
    Mexican,
    Gabor,
}

/// Topological feature extraction
#[derive(Debug, Clone)]
pub struct TopologicalFeatures {
    /// Extract persistent homology features
    pub persistent_homology: bool,
    /// Maximum persistence dimension
    pub max_dimension: usize,
    /// Extract Betti numbers
    pub betti_numbers: bool,
    /// Extract topological complexity measures
    pub complexity_measures: bool,
}

/// Statistical feature extraction
#[derive(Debug, Clone)]
pub struct StatisticalFeatures {
    /// Extract moment-based features
    pub moments: bool,
    /// Extract quantile features
    pub quantiles: bool,
    /// Quantile levels to extract
    pub quantile_levels: Vec<f64>,
    /// Extract distribution shape features
    pub distributionshape: bool,
    /// Extract correlation features
    pub correlation_features: bool,
    /// Extract entropy measures
    pub entropy_measures: bool,
}

/// Feature normalization methods
#[derive(Debug, Clone, Copy)]
pub enum FeatureNormalization {
    /// No normalization
    None,
    /// Z-score normalization
    ZScore,
    /// Min-max scaling
    MinMax,
    /// Robust scaling (median and IQR)
    Robust,
    /// Quantile uniform transformation
    QuantileUniform,
    /// Power transformation (Box-Cox)
    PowerTransform,
}

impl Default for FeatureExtraction {
    fn default() -> Self {
        Self {
            time_series_features: TimeSeriesFeatures::default(),
            phase_space_features: PhaseSpaceFeatures::default(),
            frequency_features: FrequencyFeatures::default(),
            topological_features: TopologicalFeatures::default(),
            statistical_features: StatisticalFeatures::default(),
            normalization: FeatureNormalization::ZScore,
        }
    }
}

impl Default for TimeSeriesFeatures {
    fn default() -> Self {
        Self {
            window_size: 100,
            overlap: 0.5,
            trend_features: true,
            seasonality_features: true,
            autocorr_features: true,
            max_lag: 20,
            change_point_features: false,
        }
    }
}

impl Default for PhaseSpaceFeatures {
    fn default() -> Self {
        Self {
            embedding_dim: 3,
            time_delay: 1,
            attractor_features: true,
            recurrence_features: false,
            recurrence_threshold: 0.1,
            poincare_features: false,
        }
    }
}

impl Default for FrequencyFeatures {
    fn default() -> Self {
        Self {
            psd_features: true,
            frequency_bins: 64,
            dominant_freq_features: true,
            spectral_entropy: false,
            wavelet_features: false,
            wavelet_type: WaveletType::Daubechies(4),
        }
    }
}

impl Default for TopologicalFeatures {
    fn default() -> Self {
        Self {
            persistent_homology: false,
            max_dimension: 2,
            betti_numbers: false,
            complexity_measures: false,
        }
    }
}

impl Default for StatisticalFeatures {
    fn default() -> Self {
        Self {
            moments: true,
            quantiles: true,
            quantile_levels: vec![0.25, 0.5, 0.75],
            distributionshape: true,
            correlation_features: false,
            entropy_measures: false,
        }
    }
}
