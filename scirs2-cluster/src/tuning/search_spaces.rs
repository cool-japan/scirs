//! Standard search spaces for clustering algorithms
//!
//! This module provides predefined hyperparameter search spaces
//! for common clustering algorithms.

use std::collections::HashMap;

use super::config::*;

/// Standard search spaces for clustering algorithms
pub struct StandardSearchSpaces;

impl StandardSearchSpaces {
    /// K-means search space
    pub fn kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 300, 500, 1000],
            },
        );

        parameters.insert(
            "tolerance".to_string(),
            HyperParameter::LogUniform {
                min: 1e-6,
                max: 1e-2,
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// DBSCAN search space
    pub fn dbscan() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "eps".to_string(),
            HyperParameter::Float { min: 0.1, max: 2.0 },
        );

        parameters.insert(
            "min_samples".to_string(),
            HyperParameter::Integer { min: 3, max: 20 },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// OPTICS search space
    pub fn optics() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "min_samples".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        parameters.insert(
            "max_eps".to_string(),
            HyperParameter::Float {
                min: 0.5,
                max: 10.0,
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Spectral clustering search space
    pub fn spectral() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        parameters.insert(
            "n_neighbors".to_string(),
            HyperParameter::Integer { min: 5, max: 50 },
        );

        parameters.insert(
            "gamma".to_string(),
            HyperParameter::LogUniform {
                min: 0.001,
                max: 10.0,
            },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 300, 500],
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Affinity Propagation search space
    pub fn affinity_propagation() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "damping".to_string(),
            HyperParameter::Float {
                min: 0.5,
                max: 0.99,
            },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![200, 500, 1000],
            },
        );

        parameters.insert(
            "convergence_iter".to_string(),
            HyperParameter::Integer { min: 10, max: 50 },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// BIRCH search space
    pub fn birch() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "branching_factor".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![25, 50, 100, 200],
            },
        );

        parameters.insert(
            "threshold".to_string(),
            HyperParameter::Float { min: 0.1, max: 2.0 },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Gaussian Mixture Model search space
    pub fn gmm() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_components".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 200, 500],
            },
        );

        parameters.insert(
            "tol".to_string(),
            HyperParameter::LogUniform {
                min: 1e-6,
                max: 1e-2,
            },
        );

        parameters.insert(
            "reg_covar".to_string(),
            HyperParameter::LogUniform {
                min: 1e-8,
                max: 1e-4,
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Mean Shift search space
    pub fn mean_shift() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "bandwidth".to_string(),
            HyperParameter::Float { min: 0.1, max: 5.0 },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 300, 500],
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Hierarchical clustering search space
    pub fn hierarchical() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        parameters.insert(
            "linkage".to_string(),
            HyperParameter::Categorical {
                choices: vec![
                    "ward".to_string(),
                    "complete".to_string(),
                    "average".to_string(),
                    "single".to_string(),
                ],
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Get search space by algorithm name
    pub fn get_search_space(algorithm: &str) -> Option<SearchSpace> {
        match algorithm.to_lowercase().as_str() {
            "kmeans" | "k-means" => Some(Self::kmeans()),
            "dbscan" => Some(Self::dbscan()),
            "optics" => Some(Self::optics()),
            "spectral" => Some(Self::spectral()),
            "affinity_propagation" | "affinity-propagation" => Some(Self::affinity_propagation()),
            "birch" => Some(Self::birch()),
            "gmm" | "gaussian_mixture" => Some(Self::gmm()),
            "mean_shift" | "mean-shift" => Some(Self::mean_shift()),
            "hierarchical" | "agglomerative" => Some(Self::hierarchical()),
            _ => None,
        }
    }

    /// Create a custom search space with specified parameter ranges
    pub fn custom(parameters: HashMap<String, HyperParameter>) -> SearchSpace {
        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Create a search space with constraints
    pub fn with_constraints(
        mut search_space: SearchSpace,
        constraints: Vec<ParameterConstraint>,
    ) -> SearchSpace {
        search_space.constraints = constraints;
        search_space
    }

    /// Create a minimal search space for quick testing
    pub fn minimal_kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![2, 3, 5, 8],
            },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 300],
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Create an extensive search space for thorough optimization
    pub fn extensive_kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 50 },
        );

        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::Integer { min: 50, max: 2000 },
        );

        parameters.insert(
            "tolerance".to_string(),
            HyperParameter::LogUniform {
                min: 1e-8,
                max: 1e-1,
            },
        );

        parameters.insert(
            "n_init".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![1, 5, 10, 20],
            },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }

    /// Create search space for ensemble methods
    pub fn ensemble() -> SearchSpace {
        let mut parameters = HashMap::new();

        parameters.insert(
            "n_estimators".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![3, 5, 10, 15, 20],
            },
        );

        parameters.insert(
            "base_algorithm".to_string(),
            HyperParameter::Categorical {
                choices: vec![
                    "kmeans".to_string(),
                    "dbscan".to_string(),
                    "spectral".to_string(),
                ],
            },
        );

        parameters.insert(
            "consensus_threshold".to_string(),
            HyperParameter::Float { min: 0.5, max: 1.0 },
        );

        SearchSpace {
            parameters,
            constraints: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_search_space() {
        let search_space = StandardSearchSpaces::kmeans();
        assert!(search_space.parameters.contains_key("n_clusters"));
        assert!(search_space.parameters.contains_key("max_iter"));
        assert!(search_space.parameters.contains_key("tolerance"));
    }

    #[test]
    fn test_get_search_space() {
        let search_space = StandardSearchSpaces::get_search_space("kmeans");
        assert!(search_space.is_some());

        let search_space = StandardSearchSpaces::get_search_space("unknown");
        assert!(search_space.is_none());
    }

    #[test]
    fn test_dbscan_search_space() {
        let search_space = StandardSearchSpaces::dbscan();
        assert!(search_space.parameters.contains_key("eps"));
        assert!(search_space.parameters.contains_key("min_samples"));
    }

    #[test]
    fn test_custom_search_space() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "test_param".to_string(),
            HyperParameter::Float { min: 0.0, max: 1.0 },
        );

        let search_space = StandardSearchSpaces::custom(parameters);
        assert!(search_space.parameters.contains_key("test_param"));
    }
}
