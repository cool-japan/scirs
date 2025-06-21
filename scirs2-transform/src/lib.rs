//! Data transformation module for SciRS2
//!
//! This module provides utilities for transforming data in ways that are useful
//! for machine learning and data analysis. The main functionalities include:
//!
//! - Data normalization and standardization
//! - Feature engineering
//! - Dimensionality reduction

#![warn(missing_docs)]

/// Error handling for the transformation module
pub mod error;

/// Basic normalization methods for data
pub mod normalize;

/// Feature engineering techniques
pub mod features;

/// Dimensionality reduction algorithms
pub mod reduction;

/// Advanced scaling and transformation methods
pub mod scaling;

/// Missing value imputation utilities
pub mod impute;

/// Categorical data encoding utilities
pub mod encoding;

/// Feature selection utilities
pub mod selection;

// Re-export important types and functions
pub use encoding::{BinaryEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder};
pub use error::{Result, TransformError};
pub use features::{
    binarize, discretize_equal_frequency, discretize_equal_width, log_transform, power_transform,
    PolynomialFeatures, PowerTransformer,
};
pub use impute::{
    DistanceMetric, ImputeStrategy, IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer,
    WeightingScheme,
};
pub use normalize::{normalize_array, normalize_vector, NormalizationMethod, Normalizer};
pub use reduction::{trustworthiness, TruncatedSVD, LDA, PCA, TSNE};
pub use scaling::{MaxAbsScaler, QuantileTransformer};
pub use selection::VarianceThreshold;
