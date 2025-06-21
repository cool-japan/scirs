//! Curriculum optimization for adaptive training
//!
//! This module provides curriculum learning capabilities including task difficulty progression,
//! sample importance weighting, and adversarial training support.

use crate::error::{OptimError, Result};
use ndarray::{Array, Dimension, ScalarOperand, Zip};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::marker::PhantomData;

/// Curriculum learning strategy
#[derive(Debug, Clone)]
pub enum CurriculumStrategy {
    /// Linear difficulty progression
    Linear {
        /// Starting difficulty (0.0 to 1.0)
        start_difficulty: f64,
        /// Ending difficulty (0.0 to 1.0)
        end_difficulty: f64,
        /// Number of steps to reach end difficulty
        num_steps: usize,
    },
    /// Exponential difficulty progression
    Exponential {
        /// Starting difficulty (0.0 to 1.0)
        start_difficulty: f64,
        /// Ending difficulty (0.0 to 1.0)
        end_difficulty: f64,
        /// Growth rate
        growth_rate: f64,
    },
    /// Performance-based curriculum
    PerformanceBased {
        /// Threshold for advancing difficulty
        advance_threshold: f64,
        /// Threshold for reducing difficulty
        reduce_threshold: f64,
        /// Difficulty adjustment step size
        adjustment_step: f64,
        /// Window size for performance averaging
        window_size: usize,
    },
    /// Custom curriculum with predefined schedule
    Custom {
        /// Difficulty schedule (step -> difficulty)
        schedule: HashMap<usize, f64>,
        /// Default difficulty for unspecified steps
        default_difficulty: f64,
    },
}

/// Sample importance weighting strategy
#[derive(Debug, Clone)]
pub enum ImportanceWeightingStrategy {
    /// Uniform weighting (all samples equal)
    Uniform,
    /// Loss-based weighting (higher loss = higher weight)
    LossBased {
        /// Temperature parameter for softmax weighting
        temperature: f64,
        /// Minimum weight to avoid zero weights
        min_weight: f64,
    },
    /// Gradient norm based weighting
    GradientNormBased {
        /// Temperature parameter
        temperature: f64,
        /// Minimum weight
        min_weight: f64,
    },
    /// Uncertainty-based weighting
    UncertaintyBased {
        /// Temperature parameter
        temperature: f64,
        /// Minimum weight
        min_weight: f64,
    },
    /// Age-based weighting (older samples get higher weight)
    AgeBased {
        /// Decay factor for age
        decay_factor: f64,
    },
}

/// Adversarial training configuration
#[derive(Debug, Clone)]
pub struct AdversarialConfig<A: Float> {
    /// Adversarial perturbation magnitude
    pub epsilon: A,
    /// Number of adversarial steps
    pub num_steps: usize,
    /// Step size for adversarial perturbation
    pub step_size: A,
    /// Type of adversarial attack
    pub attack_type: AdversarialAttack,
    /// Regularization weight for adversarial loss
    pub adversarial_weight: A,
}

/// Types of adversarial attacks
#[derive(Debug, Clone, Copy)]
pub enum AdversarialAttack {
    /// Fast Gradient Sign Method (FGSM)
    FGSM,
    /// Projected Gradient Descent (PGD)
    PGD,
    /// Basic Iterative Method (BIM)
    BIM,
    /// Momentum Iterative Method (MIM)
    MIM,
}

/// Curriculum learning manager
#[derive(Debug)]
pub struct CurriculumManager<A: Float, D: Dimension> {
    /// Curriculum strategy
    strategy: CurriculumStrategy,
    /// Current difficulty level
    current_difficulty: f64,
    /// Current step count
    step_count: usize,
    /// Performance history
    performance_history: VecDeque<A>,
    /// Sample difficulty scores
    sample_difficulties: HashMap<usize, f64>,
    /// Importance weighting strategy
    importance_strategy: ImportanceWeightingStrategy,
    /// Sample weights
    sample_weights: HashMap<usize, A>,
    /// Adversarial training configuration
    adversarial_config: Option<AdversarialConfig<A>>,
    /// Phantom data for dimension
    _phantom: PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> CurriculumManager<A, D> {
    /// Create a new curriculum manager
    pub fn new(
        strategy: CurriculumStrategy,
        importance_strategy: ImportanceWeightingStrategy,
    ) -> Self {
        let initial_difficulty = match &strategy {
            CurriculumStrategy::Linear {
                start_difficulty, ..
            } => *start_difficulty,
            CurriculumStrategy::Exponential {
                start_difficulty, ..
            } => *start_difficulty,
            CurriculumStrategy::PerformanceBased { .. } => 0.1, // Start easy
            CurriculumStrategy::Custom {
                default_difficulty, ..
            } => *default_difficulty,
        };

        Self {
            strategy,
            current_difficulty: initial_difficulty,
            step_count: 0,
            performance_history: VecDeque::new(),
            sample_difficulties: HashMap::new(),
            importance_strategy,
            sample_weights: HashMap::new(),
            adversarial_config: None,
            _phantom: PhantomData,
        }
    }

    /// Enable adversarial training
    pub fn enable_adversarial_training(&mut self, config: AdversarialConfig<A>) {
        self.adversarial_config = Some(config);
    }

    /// Disable adversarial training
    pub fn disable_adversarial_training(&mut self) {
        self.adversarial_config = None;
    }

    /// Update curriculum based on performance
    pub fn update_curriculum(&mut self, performance: A) -> Result<()> {
        self.performance_history.push_back(performance);
        self.step_count += 1;

        // Update difficulty based on strategy
        match &self.strategy {
            CurriculumStrategy::Linear {
                start_difficulty,
                end_difficulty,
                num_steps,
            } => {
                let progress = (self.step_count as f64) / (*num_steps as f64);
                let progress = progress.min(1.0);
                self.current_difficulty =
                    start_difficulty + progress * (end_difficulty - start_difficulty);
            }
            CurriculumStrategy::Exponential {
                start_difficulty,
                end_difficulty,
                growth_rate,
            } => {
                let progress = 1.0 - (-growth_rate * self.step_count as f64).exp();
                self.current_difficulty =
                    start_difficulty + progress * (end_difficulty - start_difficulty);
            }
            CurriculumStrategy::PerformanceBased {
                advance_threshold,
                reduce_threshold,
                adjustment_step,
                window_size,
            } => {
                if self.performance_history.len() >= *window_size {
                    // Keep only recent performance
                    while self.performance_history.len() > *window_size {
                        self.performance_history.pop_front();
                    }

                    // Calculate average performance
                    let avg_performance = self
                        .performance_history
                        .iter()
                        .fold(A::zero(), |acc, &perf| acc + perf)
                        / A::from(self.performance_history.len()).unwrap();

                    let avg_perf_f64 = avg_performance.to_f64().unwrap_or(0.0);

                    // Adjust difficulty based on performance
                    if avg_perf_f64 > *advance_threshold {
                        self.current_difficulty =
                            (self.current_difficulty + adjustment_step).min(1.0);
                    } else if avg_perf_f64 < *reduce_threshold {
                        self.current_difficulty =
                            (self.current_difficulty - adjustment_step).max(0.0);
                    }
                }
            }
            CurriculumStrategy::Custom {
                schedule,
                default_difficulty,
            } => {
                self.current_difficulty = schedule
                    .get(&self.step_count)
                    .copied()
                    .unwrap_or(*default_difficulty);
            }
        }

        Ok(())
    }

    /// Set difficulty score for a sample
    pub fn set_sample_difficulty(&mut self, sample_id: usize, difficulty: f64) {
        self.sample_difficulties.insert(sample_id, difficulty);
    }

    /// Check if sample should be included based on current difficulty
    pub fn should_include_sample(&self, sample_id: usize) -> bool {
        if let Some(&sample_difficulty) = self.sample_difficulties.get(&sample_id) {
            sample_difficulty <= self.current_difficulty
        } else {
            true // Include unknown samples
        }
    }

    /// Get current difficulty level
    pub fn get_current_difficulty(&self) -> f64 {
        self.current_difficulty
    }

    /// Compute importance weights for samples
    pub fn compute_sample_weights(
        &mut self,
        sample_ids: &[usize],
        losses: &[A],
        gradient_norms: Option<&[A]>,
        uncertainties: Option<&[A]>,
    ) -> Result<()> {
        if sample_ids.len() != losses.len() {
            return Err(OptimError::DimensionMismatch(
                "Sample IDs and losses must have same length".to_string(),
            ));
        }

        match &self.importance_strategy {
            ImportanceWeightingStrategy::Uniform => {
                let uniform_weight = A::one();
                for &sample_id in sample_ids {
                    self.sample_weights.insert(sample_id, uniform_weight);
                }
            }
            ImportanceWeightingStrategy::LossBased {
                temperature,
                min_weight,
            } => {
                self.compute_loss_based_weights(sample_ids, losses, *temperature, *min_weight)?;
            }
            ImportanceWeightingStrategy::GradientNormBased {
                temperature,
                min_weight,
            } => {
                if let Some(grad_norms) = gradient_norms {
                    self.compute_gradient_norm_weights(
                        sample_ids,
                        grad_norms,
                        *temperature,
                        *min_weight,
                    )?;
                } else {
                    // Fall back to uniform weights
                    for &sample_id in sample_ids {
                        self.sample_weights.insert(sample_id, A::one());
                    }
                }
            }
            ImportanceWeightingStrategy::UncertaintyBased {
                temperature,
                min_weight,
            } => {
                if let Some(uncertainties_array) = uncertainties {
                    self.compute_uncertainty_weights(
                        sample_ids,
                        uncertainties_array,
                        *temperature,
                        *min_weight,
                    )?;
                } else {
                    // Fall back to uniform weights
                    for &sample_id in sample_ids {
                        self.sample_weights.insert(sample_id, A::one());
                    }
                }
            }
            ImportanceWeightingStrategy::AgeBased { decay_factor } => {
                self.compute_age_based_weights(sample_ids, *decay_factor)?;
            }
        }

        Ok(())
    }

    /// Compute loss-based importance weights
    fn compute_loss_based_weights(
        &mut self,
        sample_ids: &[usize],
        losses: &[A],
        temperature: f64,
        min_weight: f64,
    ) -> Result<()> {
        // Compute softmax weights based on losses
        let temp = A::from(temperature).unwrap();
        let min_w = A::from(min_weight).unwrap();

        // Find max loss for numerical stability
        let max_loss = losses.iter().fold(A::neg_infinity(), |a, &b| A::max(a, b));

        // Compute unnormalized weights
        let mut unnormalized_weights = Vec::new();
        for &loss in losses {
            let normalized_loss = (loss - max_loss) / temp;
            unnormalized_weights.push(A::exp(normalized_loss));
        }

        // Normalize weights
        let sum_weights: A = unnormalized_weights
            .iter()
            .fold(A::zero(), |acc, &w| acc + w);

        for (i, &sample_id) in sample_ids.iter().enumerate() {
            let weight = A::max(min_w, unnormalized_weights[i] / sum_weights);
            self.sample_weights.insert(sample_id, weight);
        }

        Ok(())
    }

    /// Compute gradient norm based weights
    fn compute_gradient_norm_weights(
        &mut self,
        sample_ids: &[usize],
        gradient_norms: &[A],
        temperature: f64,
        min_weight: f64,
    ) -> Result<()> {
        let temp = A::from(temperature).unwrap();
        let min_w = A::from(min_weight).unwrap();

        // Find max gradient norm for numerical stability
        let max_norm = gradient_norms
            .iter()
            .fold(A::neg_infinity(), |a, &b| A::max(a, b));

        // Compute softmax weights
        let mut unnormalized_weights = Vec::new();
        for &norm in gradient_norms {
            let normalized_norm = (norm - max_norm) / temp;
            unnormalized_weights.push(A::exp(normalized_norm));
        }

        let sum_weights: A = unnormalized_weights
            .iter()
            .fold(A::zero(), |acc, &w| acc + w);

        for (i, &sample_id) in sample_ids.iter().enumerate() {
            let weight = A::max(min_w, unnormalized_weights[i] / sum_weights);
            self.sample_weights.insert(sample_id, weight);
        }

        Ok(())
    }

    /// Compute uncertainty-based weights
    fn compute_uncertainty_weights(
        &mut self,
        sample_ids: &[usize],
        uncertainties: &[A],
        temperature: f64,
        min_weight: f64,
    ) -> Result<()> {
        let temp = A::from(temperature).unwrap();
        let min_w = A::from(min_weight).unwrap();

        // Find max uncertainty for numerical stability
        let max_uncertainty = uncertainties
            .iter()
            .fold(A::neg_infinity(), |a, &b| A::max(a, b));

        // Compute softmax weights (higher uncertainty = higher weight)
        let mut unnormalized_weights = Vec::new();
        for &uncertainty in uncertainties {
            let normalized_uncertainty = (uncertainty - max_uncertainty) / temp;
            unnormalized_weights.push(A::exp(normalized_uncertainty));
        }

        let sum_weights: A = unnormalized_weights
            .iter()
            .fold(A::zero(), |acc, &w| acc + w);

        for (i, &sample_id) in sample_ids.iter().enumerate() {
            let weight = A::max(min_w, unnormalized_weights[i] / sum_weights);
            self.sample_weights.insert(sample_id, weight);
        }

        Ok(())
    }

    /// Compute age-based weights
    fn compute_age_based_weights(&mut self, sample_ids: &[usize], decay_factor: f64) -> Result<()> {
        let decay = A::from(decay_factor).unwrap();

        for &sample_id in sample_ids {
            // Simple age-based weighting (older samples get exponentially higher weight)
            let age = A::from(self.step_count.saturating_sub(sample_id)).unwrap();
            let weight = A::exp(decay * age);
            self.sample_weights.insert(sample_id, weight);
        }

        Ok(())
    }

    /// Get importance weight for a sample
    pub fn get_sample_weight(&self, sample_id: usize) -> A {
        self.sample_weights
            .get(&sample_id)
            .copied()
            .unwrap_or_else(|| A::one())
    }

    /// Generate adversarial examples
    pub fn generate_adversarial_examples(
        &self,
        inputs: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<Array<A, D>> {
        if let Some(config) = &self.adversarial_config {
            match config.attack_type {
                AdversarialAttack::FGSM => self.fgsm_attack(inputs, gradients, config),
                AdversarialAttack::PGD => self.pgd_attack(inputs, gradients, config),
                AdversarialAttack::BIM => self.bim_attack(inputs, gradients, config),
                AdversarialAttack::MIM => self.mim_attack(inputs, gradients, config),
            }
        } else {
            Ok(inputs.clone()) // No adversarial training
        }
    }

    /// Fast Gradient Sign Method (FGSM)
    fn fgsm_attack(
        &self,
        inputs: &Array<A, D>,
        gradients: &Array<A, D>,
        config: &AdversarialConfig<A>,
    ) -> Result<Array<A, D>> {
        let mut adversarial = inputs.clone();

        // Sign of gradients
        let sign_gradients = gradients.mapv(|x| if x >= A::zero() { A::one() } else { -A::one() });

        // Add perturbation
        Zip::from(&mut adversarial)
            .and(&sign_gradients)
            .for_each(|x, &sign| {
                *x = *x + config.epsilon * sign;
            });

        Ok(adversarial)
    }

    /// Projected Gradient Descent (PGD)
    fn pgd_attack(
        &self,
        inputs: &Array<A, D>,
        gradients: &Array<A, D>,
        config: &AdversarialConfig<A>,
    ) -> Result<Array<A, D>> {
        let mut adversarial = inputs.clone();

        // Multiple PGD steps
        for _ in 0..config.num_steps {
            // Gradient step
            let sign_gradients =
                gradients.mapv(|x| if x >= A::zero() { A::one() } else { -A::one() });

            Zip::from(&mut adversarial)
                .and(&sign_gradients)
                .for_each(|x, &sign| {
                    *x = *x + config.step_size * sign;
                });

            // Project back to epsilon ball
            Zip::from(&mut adversarial)
                .and(inputs)
                .for_each(|adv, &orig| {
                    let diff = *adv - orig;
                    let clamped_diff = A::max(-config.epsilon, A::min(config.epsilon, diff));
                    *adv = orig + clamped_diff;
                });
        }

        Ok(adversarial)
    }

    /// Basic Iterative Method (BIM)
    fn bim_attack(
        &self,
        inputs: &Array<A, D>,
        gradients: &Array<A, D>,
        config: &AdversarialConfig<A>,
    ) -> Result<Array<A, D>> {
        // BIM is similar to PGD but with smaller steps
        let mut modified_config = config.clone();
        modified_config.step_size = config.epsilon / A::from(config.num_steps).unwrap();

        self.pgd_attack(inputs, gradients, &modified_config)
    }

    /// Momentum Iterative Method (MIM)
    fn mim_attack(
        &self,
        inputs: &Array<A, D>,
        gradients: &Array<A, D>,
        config: &AdversarialConfig<A>,
    ) -> Result<Array<A, D>> {
        let mut adversarial = inputs.clone();
        let mut momentum = Array::zeros(inputs.raw_dim());
        let decay_factor = A::from(1.0).unwrap(); // Momentum decay factor

        for _ in 0..config.num_steps {
            // Update momentum
            let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
            let normalized_gradients = if grad_norm > A::zero() {
                gradients.mapv(|x| x / grad_norm)
            } else {
                gradients.clone()
            };

            Zip::from(&mut momentum)
                .and(&normalized_gradients)
                .for_each(|m, &g| {
                    *m = decay_factor * *m + g;
                });

            // Apply momentum-based update
            let momentum_signs =
                momentum.mapv(|x| if x >= A::zero() { A::one() } else { -A::one() });

            Zip::from(&mut adversarial)
                .and(&momentum_signs)
                .for_each(|x, &sign| {
                    *x = *x + config.step_size * sign;
                });

            // Project back to epsilon ball
            Zip::from(&mut adversarial)
                .and(inputs)
                .for_each(|adv, &orig| {
                    let diff = *adv - orig;
                    let clamped_diff = A::max(-config.epsilon, A::min(config.epsilon, diff));
                    *adv = orig + clamped_diff;
                });
        }

        Ok(adversarial)
    }

    /// Get filtered samples based on current curriculum
    pub fn filter_samples(&self, sample_ids: &[usize]) -> Vec<usize> {
        sample_ids
            .iter()
            .copied()
            .filter(|&id| self.should_include_sample(id))
            .collect()
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &VecDeque<A> {
        &self.performance_history
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset curriculum state
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.performance_history.clear();
        self.sample_weights.clear();
        self.current_difficulty = match &self.strategy {
            CurriculumStrategy::Linear {
                start_difficulty, ..
            } => *start_difficulty,
            CurriculumStrategy::Exponential {
                start_difficulty, ..
            } => *start_difficulty,
            CurriculumStrategy::PerformanceBased { .. } => 0.1,
            CurriculumStrategy::Custom {
                default_difficulty, ..
            } => *default_difficulty,
        };
    }

    /// Export curriculum state for analysis
    pub fn export_state(&self) -> CurriculumState<A> {
        CurriculumState {
            current_difficulty: self.current_difficulty,
            step_count: self.step_count,
            performance_history: self.performance_history.clone(),
            sample_weights: self.sample_weights.clone(),
            has_adversarial: self.adversarial_config.is_some(),
        }
    }
}

/// Curriculum state for analysis and visualization
#[derive(Debug, Clone)]
pub struct CurriculumState<A: Float> {
    /// Current difficulty level
    pub current_difficulty: f64,
    /// Current step count
    pub step_count: usize,
    /// Performance history
    pub performance_history: VecDeque<A>,
    /// Sample weights
    pub sample_weights: HashMap<usize, A>,
    /// Whether adversarial training is enabled
    pub has_adversarial: bool,
}

/// Adaptive curriculum that automatically adjusts strategy
#[derive(Debug)]
pub struct AdaptiveCurriculum<A: Float, D: Dimension> {
    /// Collection of curriculum managers with different strategies
    curricula: Vec<CurriculumManager<A, D>>,
    /// Current active curriculum index
    active_curriculum: usize,
    /// Performance tracking for each curriculum
    curriculum_performance: Vec<VecDeque<A>>,
    /// Switch threshold for changing curriculum
    switch_threshold: A,
    /// Minimum steps before switching
    min_steps_before_switch: usize,
    /// Steps since last switch
    steps_since_switch: usize,
    /// Phantom data for dimension
    _phantom: PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> AdaptiveCurriculum<A, D> {
    /// Create a new adaptive curriculum
    pub fn new(curricula: Vec<CurriculumManager<A, D>>, switch_threshold: A) -> Self {
        let num_curricula = curricula.len();
        Self {
            curricula,
            active_curriculum: 0,
            curriculum_performance: vec![VecDeque::new(); num_curricula],
            switch_threshold,
            min_steps_before_switch: 100,
            steps_since_switch: 0,
            _phantom: PhantomData,
        }
    }

    /// Update with performance and potentially switch curriculum
    pub fn update(&mut self, performance: A) -> Result<()> {
        // Update current curriculum
        self.curricula[self.active_curriculum].update_curriculum(performance)?;
        self.curriculum_performance[self.active_curriculum].push_back(performance);
        self.steps_since_switch += 1;

        // Consider switching if enough steps have passed
        if self.steps_since_switch >= self.min_steps_before_switch {
            self.consider_curriculum_switch()?;
        }

        Ok(())
    }

    /// Consider switching to a better performing curriculum
    fn consider_curriculum_switch(&mut self) -> Result<()> {
        let current_performance = self.get_average_performance(self.active_curriculum);
        let mut best_curriculum = self.active_curriculum;
        let mut best_performance = current_performance;

        // Find best performing curriculum
        for (i, _) in self.curricula.iter().enumerate() {
            if i != self.active_curriculum {
                let perf = self.get_average_performance(i);
                if perf > best_performance + self.switch_threshold {
                    best_performance = perf;
                    best_curriculum = i;
                }
            }
        }

        // Switch if a better curriculum is found
        if best_curriculum != self.active_curriculum {
            self.active_curriculum = best_curriculum;
            self.steps_since_switch = 0;
        }

        Ok(())
    }

    /// Get average performance for a curriculum
    fn get_average_performance(&self, curriculum_idx: usize) -> A {
        let perf_history = &self.curriculum_performance[curriculum_idx];
        if perf_history.is_empty() {
            A::zero()
        } else {
            let sum = perf_history.iter().fold(A::zero(), |acc, &perf| acc + perf);
            sum / A::from(perf_history.len()).unwrap()
        }
    }

    /// Get active curriculum manager
    pub fn active_curriculum(&self) -> &CurriculumManager<A, D> {
        &self.curricula[self.active_curriculum]
    }

    /// Get mutable active curriculum manager
    pub fn active_curriculum_mut(&mut self) -> &mut CurriculumManager<A, D> {
        &mut self.curricula[self.active_curriculum]
    }

    /// Get active curriculum index
    pub fn active_curriculum_index(&self) -> usize {
        self.active_curriculum
    }

    /// Get performance comparison across curricula
    pub fn get_curriculum_comparison(&self) -> Vec<(usize, A)> {
        (0..self.curricula.len())
            .map(|i| (i, self.get_average_performance(i)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_linear_curriculum() {
        let strategy = CurriculumStrategy::Linear {
            start_difficulty: 0.1,
            end_difficulty: 1.0,
            num_steps: 10,
        };

        let importance_strategy = ImportanceWeightingStrategy::Uniform;
        let mut curriculum =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

        // Test initial difficulty
        assert_relative_eq!(curriculum.get_current_difficulty(), 0.1, epsilon = 1e-6);

        // Update curriculum multiple times
        for _ in 0..5 {
            curriculum.update_curriculum(0.8).unwrap();
        }

        // Difficulty should have increased
        assert!(curriculum.get_current_difficulty() > 0.1);
        assert!(curriculum.get_current_difficulty() <= 1.0);
    }

    #[test]
    fn test_performance_based_curriculum() {
        let strategy = CurriculumStrategy::PerformanceBased {
            advance_threshold: 0.8,
            reduce_threshold: 0.4,
            adjustment_step: 0.1,
            window_size: 3,
        };

        let importance_strategy = ImportanceWeightingStrategy::Uniform;
        let mut curriculum =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

        let initial_difficulty = curriculum.get_current_difficulty();

        // Simulate good performance (should increase difficulty)
        for _ in 0..5 {
            curriculum.update_curriculum(0.9).unwrap();
        }

        assert!(curriculum.get_current_difficulty() > initial_difficulty);
    }

    #[test]
    fn test_sample_filtering() {
        let strategy = CurriculumStrategy::Linear {
            start_difficulty: 0.5,
            end_difficulty: 0.5,
            num_steps: 10,
        };

        let importance_strategy = ImportanceWeightingStrategy::Uniform;
        let mut curriculum =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

        // Set sample difficulties
        curriculum.set_sample_difficulty(1, 0.3); // Easy
        curriculum.set_sample_difficulty(2, 0.7); // Hard
        curriculum.set_sample_difficulty(3, 0.5); // Medium

        let sample_ids = vec![1, 2, 3, 4]; // 4 has no difficulty set
        let filtered = curriculum.filter_samples(&sample_ids);

        // Should include samples 1, 3, 4 (difficulty <= 0.5 or unknown)
        assert_eq!(filtered.len(), 3);
        assert!(filtered.contains(&1));
        assert!(filtered.contains(&3));
        assert!(filtered.contains(&4));
        assert!(!filtered.contains(&2));
    }

    #[test]
    fn test_loss_based_importance_weighting() {
        let strategy = CurriculumStrategy::Linear {
            start_difficulty: 0.5,
            end_difficulty: 0.5,
            num_steps: 10,
        };

        let importance_strategy = ImportanceWeightingStrategy::LossBased {
            temperature: 1.0,
            min_weight: 0.1,
        };

        let mut curriculum =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

        let sample_ids = vec![1, 2, 3];
        let losses = vec![0.1, 1.0, 0.5]; // Low, high, medium loss

        curriculum
            .compute_sample_weights(&sample_ids, &losses, None, None)
            .unwrap();

        // Sample with highest loss should have highest weight
        let weight1 = curriculum.get_sample_weight(1);
        let weight2 = curriculum.get_sample_weight(2);
        let weight3 = curriculum.get_sample_weight(3);

        assert!(weight2 > weight3); // High loss > medium loss
        assert!(weight3 > weight1); // Medium loss > low loss
    }

    #[test]
    fn test_adversarial_config() {
        let strategy = CurriculumStrategy::Linear {
            start_difficulty: 0.5,
            end_difficulty: 0.5,
            num_steps: 10,
        };

        let importance_strategy = ImportanceWeightingStrategy::Uniform;
        let mut curriculum =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

        let adversarial_config = AdversarialConfig {
            epsilon: 0.1,
            num_steps: 5,
            step_size: 0.02,
            attack_type: AdversarialAttack::FGSM,
            adversarial_weight: 0.5,
        };

        curriculum.enable_adversarial_training(adversarial_config);

        let inputs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, -0.2, 0.3]);

        let adversarial = curriculum
            .generate_adversarial_examples(&inputs, &gradients)
            .unwrap();

        // Adversarial examples should be different from original
        assert_ne!(adversarial.as_slice().unwrap(), inputs.as_slice().unwrap());

        // Check that perturbation is bounded
        for (orig, adv) in inputs.iter().zip(adversarial.iter()) {
            assert!((adv - orig).abs() <= 0.1 + 1e-6); // epsilon + small tolerance
        }
    }

    #[test]
    fn test_adaptive_curriculum() {
        let strategy1 = CurriculumStrategy::Linear {
            start_difficulty: 0.1,
            end_difficulty: 0.5,
            num_steps: 100,
        };

        let strategy2 = CurriculumStrategy::Linear {
            start_difficulty: 0.2,
            end_difficulty: 0.8,
            num_steps: 100,
        };

        let importance_strategy = ImportanceWeightingStrategy::Uniform;
        let curriculum1 =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy1, importance_strategy.clone());
        let curriculum2 =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy2, importance_strategy);

        let mut adaptive = AdaptiveCurriculum::new(vec![curriculum1, curriculum2], 0.1);

        assert_eq!(adaptive.active_curriculum_index(), 0);

        // Update with some performance values
        for _ in 0..150 {
            adaptive.update(0.7).unwrap();
        }

        // Should potentially have switched curriculum
        let comparison = adaptive.get_curriculum_comparison();
        assert_eq!(comparison.len(), 2);
    }

    #[test]
    fn test_curriculum_state_export() {
        let strategy = CurriculumStrategy::Linear {
            start_difficulty: 0.1,
            end_difficulty: 1.0,
            num_steps: 10,
        };

        let importance_strategy = ImportanceWeightingStrategy::Uniform;
        let mut curriculum =
            CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

        curriculum.update_curriculum(0.8).unwrap();
        let state = curriculum.export_state();

        assert_eq!(state.step_count, 1);
        assert_eq!(state.performance_history.len(), 1);
        assert!(!state.has_adversarial);
    }
}
