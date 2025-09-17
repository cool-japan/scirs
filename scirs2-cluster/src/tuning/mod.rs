//! Hyperparameter tuning for clustering algorithms
//!
//! This module provides comprehensive hyperparameter optimization capabilities
//! for all clustering algorithms in the scirs2-cluster crate. It supports
//! various search strategies including grid search, random search, Bayesian
//! optimization, and evolutionary approaches.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_cluster::tuning::{AutoTuner, TuningConfig, SearchStrategy, StandardSearchSpaces};
//! use ndarray::Array2;
//!
//! // Create sample data
//! let data = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
//!
//! // Configure tuning
//! let config = TuningConfig {
//!     strategy: SearchStrategy::RandomSearch { n_trials: 50 },
//!     ..Default::default()
//! };
//!
//! // Create tuner and search space
//! let tuner = AutoTuner::new(config);
//! let search_space = StandardSearchSpaces::kmeans();
//!
//! // Tune hyperparameters
//! let result = tuner.tune_kmeans(data.view(), search_space).unwrap();
//! println!("Best score: {}", result.best_score);
//! ```

pub mod algorithm_tuners;
pub mod bayesian_optimization;
pub mod config;
pub mod cross_validation;
pub mod optimization_strategies;
pub mod search_spaces;
pub mod utilities;

// Re-export main types for convenience
pub use algorithm_tuners::AutoTuner;
pub use config::*;
pub use cross_validation::CrossValidator;
pub use optimization_strategies::ParameterGenerator;
pub use search_spaces::StandardSearchSpaces;
pub use utilities::*;

// Re-export for backward compatibility with the old tuning_old.rs interface
pub use algorithm_tuners::AutoTuner as Tuner;

/// Convenience function to create a default tuning configuration
pub fn default_tuning_config() -> TuningConfig {
    TuningConfig::default()
}

/// Convenience function to create a random search configuration
pub fn random_search_config(n_trials: usize) -> TuningConfig {
    TuningConfig {
        strategy: SearchStrategy::RandomSearch { n_trials },
        ..Default::default()
    }
}

/// Convenience function to create a Bayesian optimization configuration
pub fn bayesian_optimization_config(n_initial_points: usize) -> TuningConfig {
    TuningConfig {
        strategy: SearchStrategy::BayesianOptimization {
            n_initial_points,
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
        },
        ..Default::default()
    }
}

/// Convenience function to create a grid search configuration
pub fn grid_search_config() -> TuningConfig {
    TuningConfig {
        strategy: SearchStrategy::GridSearch,
        ..Default::default()
    }
}

/// Convenience function for quick K-means tuning
pub fn quick_tune_kmeans<F>(
    data: ndarray::ArrayView2<F>,
    n_trials: Option<usize>,
) -> crate::error::Result<TuningResult>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + 'static
        + std::iter::Sum
        + std::fmt::Display
        + Send
        + Sync
        + ndarray::ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + PartialOrd,
    f64: From<F>,
{
    let config = random_search_config(n_trials.unwrap_or(20));
    let tuner = AutoTuner::new(config);
    let search_space = StandardSearchSpaces::kmeans();
    tuner.tune_kmeans(data, search_space)
}

/// Convenience function for quick DBSCAN tuning
pub fn quick_tune_dbscan<F>(
    data: ndarray::ArrayView2<F>,
    n_trials: Option<usize>,
) -> crate::error::Result<TuningResult>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + 'static
        + std::iter::Sum
        + std::fmt::Display
        + Send
        + Sync
        + ndarray::ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + PartialOrd,
    f64: From<F>,
{
    let config = random_search_config(n_trials.unwrap_or(20));
    let tuner = AutoTuner::new(config);
    let search_space = StandardSearchSpaces::dbscan();
    tuner.tune_dbscan(data, search_space)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_default_tuning_config() {
        let config = default_tuning_config();
        assert_eq!(config.max_evaluations, 100);
    }

    #[test]
    fn test_random_search_config() {
        let config = random_search_config(50);
        match config.strategy {
            SearchStrategy::RandomSearch { n_trials } => assert_eq!(n_trials, 50),
            _ => panic!("Expected RandomSearch strategy"),
        }
    }

    #[test]
    fn test_bayesian_optimization_config() {
        let config = bayesian_optimization_config(10);
        match config.strategy {
            SearchStrategy::BayesianOptimization {
                n_initial_points, ..
            } => {
                assert_eq!(n_initial_points, 10)
            }
            _ => panic!("Expected BayesianOptimization strategy"),
        }
    }

    #[test]
    fn test_quick_tune_kmeans() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let result = quick_tune_kmeans(data.view(), Some(5));
        assert!(result.is_ok());
    }
}
