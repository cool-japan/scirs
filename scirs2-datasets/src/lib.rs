//! Datasets module for SciRS2
//!
//! This module provides dataset loading utilities similar to scikit-learn's datasets module.
//! It includes toy datasets, sample datasets, time series datasets, data generators,
//! and utilities for loading and processing datasets.
//!
//! # Features
//!
//! - **Toy datasets**: Classic datasets like Iris, Boston Housing, Breast Cancer, and Digits
//! - **Data generators**: Create synthetic datasets for classification, regression, clustering, and time series
//! - **Cross-validation utilities**: K-fold, stratified, and time series cross-validation
//! - **Dataset utilities**: Train/test splitting, normalization, and metadata handling
//! - **Caching**: Efficient caching system for downloaded datasets
//! - **Registry**: Centralized registry for dataset metadata and locations
//!
//! # Examples
//!
//! ## Loading toy datasets
//!
//! ```rust
//! use scirs2_datasets::{load_iris, load_boston};
//!
//! // Load the classic Iris dataset
//! let iris = load_iris().unwrap();
//! println!("Iris dataset: {} samples, {} features", iris.n_samples(), iris.n_features());
//!
//! // Load the Boston housing dataset
//! let boston = load_boston().unwrap();
//! println!("Boston dataset: {} samples, {} features", boston.n_samples(), boston.n_features());
//! ```
//!
//! ## Generating synthetic datasets
//!
//! ```rust
//! use scirs2_datasets::{make_classification, make_regression, make_blobs, make_spirals, make_moons};
//!
//! // Generate a classification dataset
//! let classification = make_classification(100, 5, 3, 2, 4, Some(42)).unwrap();
//! println!("Classification dataset: {} samples, {} features, {} classes",
//!          classification.n_samples(), classification.n_features(), 3);
//!
//! // Generate a regression dataset
//! let regression = make_regression(50, 4, 3, 0.1, Some(42)).unwrap();
//! println!("Regression dataset: {} samples, {} features",
//!          regression.n_samples(), regression.n_features());
//!
//! // Generate a clustering dataset
//! let blobs = make_blobs(80, 3, 4, 1.0, Some(42)).unwrap();
//! println!("Blobs dataset: {} samples, {} features, {} clusters",
//!          blobs.n_samples(), blobs.n_features(), 4);
//!
//! // Generate non-linear patterns
//! let spirals = make_spirals(200, 2, 0.1, Some(42)).unwrap();
//! let moons = make_moons(150, 0.05, Some(42)).unwrap();
//! ```
//!
//! ## Cross-validation
//!
//! ```rust
//! use scirs2_datasets::{load_iris, k_fold_split, stratified_k_fold_split};
//!
//! let iris = load_iris().unwrap();
//!
//! // K-fold cross-validation
//! let k_folds = k_fold_split(iris.n_samples(), 5, true, Some(42)).unwrap();
//! println!("Created {} folds for K-fold CV", k_folds.len());
//!
//! // Stratified K-fold cross-validation
//! if let Some(target) = &iris.target {
//!     let stratified_folds = stratified_k_fold_split(target, 5, true, Some(42)).unwrap();
//!     println!("Created {} stratified folds", stratified_folds.len());
//! }
//! ```
//!
//! ## Dataset manipulation
//!
//! ```rust
//! use scirs2_datasets::{load_iris, Dataset};
//!
//! let iris = load_iris().unwrap();
//!
//! // Access dataset properties
//! println!("Dataset: {} samples, {} features", iris.n_samples(), iris.n_features());
//! if let Some(feature_names) = iris.feature_names() {
//!     println!("Features: {:?}", feature_names);
//! }
//! ```

#![warn(missing_docs)]

pub mod cache;
pub mod error;
pub mod generators;
pub mod loaders;
pub mod registry;
pub mod sample;
pub mod time_series;
pub mod toy;
/// Core utilities for working with datasets
///
/// This module provides the Dataset struct and helper functions for
/// manipulating and transforming datasets.
pub mod utils;

// Temporary module to test method resolution conflict
mod method_resolution_test;

// Re-export commonly used functionality
pub use cache::{
    get_cache_dir, BatchOperations, BatchResult, CacheFileInfo, CacheManager, CacheStats,
    DatasetCache, DetailedCacheStats,
};
pub use generators::{
    add_time_series_noise, inject_missing_data, inject_outliers, make_anisotropic_blobs,
    make_blobs, make_circles, make_classification, make_corrupted_dataset,
    make_hierarchical_clusters, make_moons, make_regression, make_spirals, make_swiss_roll,
    make_time_series, MissingPattern, OutlierType,
};
pub use registry::*;
pub use sample::*;
pub use toy::*;
pub use utils::{
    create_balanced_dataset, create_binned_features, generate_synthetic_samples, importance_sample,
    k_fold_split, min_max_scale, polynomial_features, random_oversample, random_sample,
    random_undersample, robust_scale, statistical_features, stratified_k_fold_split,
    stratified_sample, time_series_split, BalancingStrategy, BinningStrategy, CrossValidationFolds,
    Dataset,
};
