//! Algorithm-specific hyperparameter tuners
//!
//! This module contains the main AutoTuner implementation and methods
//! for tuning hyperparameters of specific clustering algorithms.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::{rng, Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::advanced::{
    adaptive_online_clustering, quantum_kmeans, rl_clustering, AdaptiveOnlineConfig, QuantumConfig,
    RLClusteringConfig,
};
use crate::affinity::{affinity_propagation, AffinityPropagationOptions};
use crate::birch::{birch, BirchOptions};
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions};
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use crate::spectral::{spectral_clustering, AffinityMode, SpectralClusteringOptions};
use crate::stability::OptimalKSelector;
use crate::vq::{kmeans, kmeans2};

use super::config::*;
use super::cross_validation::CrossValidator;
use super::optimization_strategies::ParameterGenerator;
use super::utilities::*;

use statrs::statistics::Statistics;

/// Main hyperparameter tuning engine for clustering algorithms
pub struct AutoTuner<F: Float> {
    config: TuningConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
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
    > AutoTuner<F>
where
    f64: From<F>,
{
    /// Create a new auto tuner with specified configuration
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Tune K-means hyperparameters
    pub fn tune_kmeans(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            if let Some(max_time) = self.config.resource_constraints.max_total_time {
                if start_time.elapsed().as_secs_f64() > max_time {
                    break;
                }
            }

            let eval_start = std::time::Instant::now();

            let k = params.get("n_clusters").map(|&x| x as usize).unwrap_or(3);
            let max_iter = params.get("max_iter").map(|&x| x as usize);
            let tol = params.get("tolerance").copied();
            let seed = rng.random_range(0..u64::MAX);

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let cv_scores = cv_validator.cross_validate_kmeans(
                data,
                k,
                max_iter,
                tol,
                Some(seed),
                &self.config.metric,
            )?;

            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let cv_std = calculate_std_dev(&cv_scores);
            let eval_time = eval_start.elapsed().as_secs_f64();

            let result = EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores,
                cv_std,
                metadata: HashMap::new(),
            };

            let is_better = is_score_better(mean_score, best_score, &self.config.metric);
            if is_better {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            evaluation_history.push(result);

            if let Some(ref early_stop) = self.config.early_stopping {
                if should_stop_early(&evaluation_history, early_stop) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let convergence_info =
            create_convergence_info(&evaluation_history, self.config.max_evaluations);
        let exploration_stats = calculate_exploration_stats(&evaluation_history);

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info,
            exploration_stats,
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune DBSCAN hyperparameters
    pub fn tune_dbscan(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            let eval_start = std::time::Instant::now();

            let eps = params.get("eps").copied().unwrap_or(0.5);
            let min_samples = params.get("min_samples").map(|&x| x as usize).unwrap_or(5);

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let cv_scores =
                cv_validator.cross_validate_dbscan(data, eps, min_samples, &self.config.metric)?;

            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let cv_std = calculate_std_dev(&cv_scores);
            let eval_time = eval_start.elapsed().as_secs_f64();

            let result = EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores,
                cv_std,
                metadata: HashMap::new(),
            };

            let is_better = is_score_better(mean_score, best_score, &self.config.metric);
            if is_better {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            evaluation_history.push(result);

            if let Some(ref early_stop) = self.config.early_stopping {
                if should_stop_early(&evaluation_history, early_stop) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let convergence_info =
            create_convergence_info(&evaluation_history, self.config.max_evaluations);
        let exploration_stats = calculate_exploration_stats(&evaluation_history);

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info,
            exploration_stats,
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune OPTICS hyperparameters
    pub fn tune_optics(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        for combination in &parameter_combinations {
            let min_samples = combination
                .get("min_samples")
                .ok_or_else(|| {
                    ClusteringError::InvalidInput("min_samples parameter not found".to_string())
                })?
                .round() as usize;
            let max_eps = combination.get("max_eps").copied().unwrap_or(5.0);

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let scores = cv_validator.cross_validate_optics(
                data,
                min_samples,
                Some(F::from(max_eps).unwrap()),
                &self.config.metric,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores,
                cv_std: 0.0,
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune Spectral clustering hyperparameters
    pub fn tune_spectral(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        for combination in &parameter_combinations {
            let n_clusters = combination
                .get("n_clusters")
                .ok_or_else(|| {
                    ClusteringError::InvalidInput("n_clusters parameter not found".to_string())
                })?
                .round() as usize;
            let n_neighbors = combination
                .get("n_neighbors")
                .copied()
                .unwrap_or(10.0)
                .round() as usize;
            let gamma = combination.get("gamma").copied().unwrap_or(1.0);
            let max_iter = combination
                .get("max_iter")
                .copied()
                .unwrap_or(300.0)
                .round() as usize;

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let scores = cv_validator.cross_validate_spectral(
                data,
                n_clusters,
                n_neighbors,
                F::from(gamma).unwrap(),
                max_iter,
                &self.config.metric,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune Affinity Propagation hyperparameters
    pub fn tune_affinity_propagation(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        for combination in &parameter_combinations {
            let damping = combination.get("damping").copied().unwrap_or(0.5);
            let max_iter = combination
                .get("max_iter")
                .copied()
                .unwrap_or(200.0)
                .round() as usize;
            let convergence_iter = combination
                .get("convergence_iter")
                .copied()
                .unwrap_or(15.0)
                .round() as usize;

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let scores = cv_validator.cross_validate_affinity_propagation(
                data,
                F::from(damping).unwrap(),
                max_iter,
                convergence_iter,
                &self.config.metric,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune BIRCH hyperparameters
    pub fn tune_birch(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        for combination in &parameter_combinations {
            let branching_factor = combination
                .get("branching_factor")
                .copied()
                .unwrap_or(50.0)
                .round() as usize;
            let threshold = combination.get("threshold").copied().unwrap_or(0.5);

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let scores = cv_validator.cross_validate_birch(
                data,
                branching_factor,
                F::from(threshold).unwrap(),
                &self.config.metric,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune GMM hyperparameters
    pub fn tune_gmm(&self, data: ArrayView2<F>, search_space: SearchSpace) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_generator = ParameterGenerator::new(&self.config);
        let parameter_combinations = parameter_generator.generate_combinations(&search_space)?;

        for combination in &parameter_combinations {
            let n_components = combination
                .get("n_components")
                .ok_or_else(|| {
                    ClusteringError::InvalidInput("n_components parameter not found".to_string())
                })?
                .round() as usize;
            let max_iter = combination
                .get("max_iter")
                .copied()
                .unwrap_or(100.0)
                .round() as usize;
            let tol = combination.get("tol").copied().unwrap_or(1e-3);
            let reg_covar = combination.get("reg_covar").copied().unwrap_or(1e-6);

            let cv_validator = CrossValidator::new(&self.config.cv_config);
            let scores = cv_validator.cross_validate_gmm(
                data,
                n_components,
                max_iter,
                F::from(tol).unwrap(),
                F::from(reg_covar).unwrap(),
                &self.config.metric,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }
}
