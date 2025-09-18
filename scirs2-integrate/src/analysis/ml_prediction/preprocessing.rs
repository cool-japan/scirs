//! Data Preprocessing Pipeline
//!
//! This module contains structures for data preprocessing, quality checks,
//! validation rules, and data cleaning operations.

use std::collections::HashMap;

/// Data preprocessing pipeline
#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    /// Preprocessing steps
    pub steps: Vec<PreprocessingStep>,
    /// Quality checks
    pub quality_checks: Vec<QualityCheck>,
    /// Data validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Preprocessing step types
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    /// Remove outliers
    OutlierRemoval {
        method: OutlierDetectionMethod,
        threshold: f64,
    },
    /// Smooth data
    Smoothing {
        method: SmoothingMethod,
        window_size: usize,
    },
    /// Normalize features
    Normalization {
        method: super::features::FeatureNormalization,
    },
    /// Filter noise
    NoiseFiltering {
        filter_type: FilterType,
        cutoff_frequency: f64,
    },
    /// Interpolate missing values
    Interpolation { method: InterpolationMethod },
}

/// Outlier detection methods
#[derive(Debug, Clone, Copy)]
pub enum OutlierDetectionMethod {
    ZScore,
    IQR,
    IsolationForest,
    LocalOutlierFactor,
    EllipticEnvelope,
}

/// Smoothing methods
#[derive(Debug, Clone, Copy)]
pub enum SmoothingMethod {
    MovingAverage,
    ExponentialSmoothing,
    SavitzkyGolay,
    Gaussian,
    Median,
}

/// Filter types for noise removal
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    BandStop,
    Butterworth,
    Chebyshev,
}

/// Interpolation methods
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    Linear,
    Cubic,
    Spline,
    Polynomial,
    NearestNeighbor,
}

/// Data quality checks
#[derive(Debug, Clone)]
pub enum QualityCheck {
    /// Check for missing values
    MissingValues { max_missing_ratio: f64 },
    /// Check data range
    RangeCheck { min_value: f64, max_value: f64 },
    /// Check for constant values
    ConstantValues { tolerance: f64 },
    /// Check sampling rate
    SamplingRate { expected_rate: f64, tolerance: f64 },
    /// Check for duplicate values
    Duplicates { max_duplicate_ratio: f64 },
}

/// Data validation rules
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// Physical constraints
    PhysicalConstraints { constraints: Vec<Constraint> },
    /// Statistical tests
    StatisticalTests { tests: Vec<StatisticalTest> },
    /// Trend validation
    TrendValidation { max_trend_change: f64 },
    /// Correlation validation
    CorrelationValidation {
        expected_correlations: HashMap<String, f64>,
    },
}

/// Physical constraint types
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Variable bounds
    Bounds {
        variable: String,
        min: f64,
        max: f64,
    },
    /// Conservation laws
    Conservation {
        law_type: ConservationLaw,
        tolerance: f64,
    },
    /// Rate limits
    RateLimit { variable: String, max_rate: f64 },
}

/// Conservation law types
#[derive(Debug, Clone, Copy)]
pub enum ConservationLaw {
    Energy,
    Mass,
    Momentum,
    AngularMomentum,
    Charge,
}

/// Statistical test types
#[derive(Debug, Clone, Copy)]
pub enum StatisticalTest {
    Normality,
    Stationarity,
    Independence,
    Homoscedasticity,
    Linearity,
}
