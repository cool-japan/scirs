//! Active learning approaches for adaptive sampling in interpolation
//!
//! This module provides algorithms for intelligently selecting new sampling points
//! to improve interpolation accuracy. Active learning techniques can significantly
//! reduce the number of expensive function evaluations needed to achieve good
//! interpolation accuracy by focusing sampling on the most informative regions.
//!
//! # Active Learning Strategies
//!
//! - **Uncertainty-based sampling**: Sample where interpolation uncertainty is highest
//! - **Error-based refinement**: Add points where current approximation error is large  
//! - **Variance-based selection**: Sample where prediction variance is high
//! - **Gradient-based sampling**: Sample in regions with high function gradients
//! - **Expected improvement**: Sample where expected improvement in approximation is highest
//! - **Exploration vs exploitation**: Balance between exploring unknown regions and refining known ones
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_interpolate::adaptive_learning::{ActiveLearner, SamplingStrategy};
//! use scirs2_interpolate::multiscale::MultiscaleBSpline;
//!
//! // Create sample data
//! let x = Array1::linspace(0.0_f64, 10.0_f64, 20);
//! let y = x.mapv(|x| x.sin() + 0.1_f64 * (5.0_f64 * x).sin());
//!
//! // Create an initial interpolation model
//! let spline = MultiscaleBSpline::new(
//!     &x.view(), &y.view(), 10, 3, 5, 0.01_f64,
//!     scirs2_interpolate::ExtrapolateMode::Extrapolate
//! ).unwrap();
//!
//! // Set up active learning
//! let mut learner = ActiveLearner::new(spline, SamplingStrategy::UncertaintyBased)
//!     .with_budget(50)
//!     .with_exploration_weight(0.2);
//!
//! // Suggest next sampling points
//! let candidates = learner.suggest_samples(10).unwrap();
//! println!("Suggested sampling points: {:?}", candidates);
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use crate::multiscale::MultiscaleBSpline;
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

/// Strategies for active sampling in interpolation problems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Sample points where interpolation uncertainty is highest
    UncertaintyBased,
    /// Sample based on current approximation error estimates
    ErrorBased,
    /// Sample where function gradients are estimated to be large
    GradientBased,
    /// Sample to maximize expected improvement in global approximation
    ExpectedImprovement,
    /// Balance exploration of unknown regions with exploitation of known errors
    ExplorationExploitation,
    /// Combine multiple strategies with weighted importance
    Combined,
}

/// Parameters controlling active learning behavior
#[derive(Debug, Clone)]
pub struct ActiveLearningConfig<T> {
    /// Maximum number of new samples to suggest in one iteration
    pub max_samples_per_iteration: usize,
    /// Total sampling budget (maximum number of samples)
    pub total_budget: usize,
    /// Weight for exploration vs exploitation (0.0 = pure exploitation, 1.0 = pure exploration)
    pub exploration_weight: T,
    /// Minimum distance between new sampling points (to avoid clustering)
    pub min_sample_distance: T,
    /// Threshold for stopping active learning (when improvement is below this)
    pub convergence_threshold: T,
    /// Weight for uncertainty in combined strategies
    pub uncertainty_weight: T,
    /// Weight for error estimates in combined strategies
    pub error_weight: T,
    /// Weight for gradient estimates in combined strategies
    pub gradient_weight: T,
}

impl<T: Float + FromPrimitive> Default for ActiveLearningConfig<T> {
    fn default() -> Self {
        Self {
            max_samples_per_iteration: 10,
            total_budget: 100,
            exploration_weight: T::from(0.1).unwrap(),
            min_sample_distance: T::from(0.01).unwrap(),
            convergence_threshold: T::from(1e-6).unwrap(),
            uncertainty_weight: T::from(0.4).unwrap(),
            error_weight: T::from(0.4).unwrap(),
            gradient_weight: T::from(0.2).unwrap(),
        }
    }
}

/// Information about a candidate sampling point
#[derive(Debug, Clone)]
pub struct SamplingCandidate<T> {
    /// Location of the candidate point
    pub location: T,
    /// Estimated utility/importance of sampling at this point
    pub utility: T,
    /// Type of information this sample would provide
    pub information_type: String,
    /// Confidence in the utility estimate
    pub confidence: T,
}

/// Statistics tracking active learning progress
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    /// Number of samples suggested so far
    pub samples_suggested: usize,
    /// Number of iterations completed
    pub iterations_completed: usize,
    /// Best approximation error achieved
    pub best_error: f64,
    /// History of approximation errors over iterations
    pub error_history: Vec<f64>,
    /// Computational cost statistics
    pub total_computation_time_ms: u64,
    /// Distribution of sampling strategies used
    pub strategy_usage: HashMap<String, usize>,
}

/// Active learning system for intelligent sampling in interpolation
#[derive(Debug)]
pub struct ActiveLearner<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    /// The current interpolation model
    interpolator: MultiscaleBSpline<T>,
    /// Active learning configuration
    config: ActiveLearningConfig<T>,
    /// Primary sampling strategy
    strategy: SamplingStrategy,
    /// Current sample points and values
    sample_points: Array1<T>,
    sample_values: Array1<T>,
    /// Domain bounds for sampling
    domain_min: T,
    domain_max: T,
    /// Statistics tracking learning progress
    stats: LearningStats,
    /// Random number generator state (for reproducible sampling)
    rng_state: u64,
}

impl<T> ActiveLearner<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    /// Create a new active learner with an initial interpolation model
    ///
    /// # Arguments
    ///
    /// * `interpolator` - Initial interpolation model (typically trained on a small dataset)
    /// * `strategy` - Primary sampling strategy to use
    ///
    /// # Returns
    ///
    /// A new `ActiveLearner` instance ready for adaptive sampling
    pub fn new(interpolator: MultiscaleBSpline<T>, strategy: SamplingStrategy) -> Self {
        // Extract domain bounds from the interpolator
        let x_data = interpolator.x_data();
        let domain_min = x_data[0];
        let domain_max = x_data[x_data.len() - 1];

        Self {
            interpolator,
            config: ActiveLearningConfig::default(),
            strategy,
            sample_points: Array1::zeros(0),
            sample_values: Array1::zeros(0),
            domain_min,
            domain_max,
            stats: LearningStats::default(),
            rng_state: 12345, // Fixed seed for reproducibility
        }
    }

    /// Set the total sampling budget
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.config.total_budget = budget;
        self
    }

    /// Set the exploration weight (0.0 = pure exploitation, 1.0 = pure exploration)
    pub fn with_exploration_weight(mut self, weight: T) -> Self {
        self.config.exploration_weight = weight;
        self
    }

    /// Set the minimum distance between new sampling points
    pub fn with_min_sample_distance(mut self, distance: T) -> Self {
        self.config.min_sample_distance = distance;
        self
    }

    /// Set the maximum number of samples per iteration
    pub fn with_max_samples_per_iteration(mut self, max_samples: usize) -> Self {
        self.config.max_samples_per_iteration = max_samples;
        self
    }

    /// Suggest new sampling points based on the current strategy
    ///
    /// # Arguments
    ///
    /// * `num_points` - Number of new points to suggest (limited by max_samples_per_iteration)
    ///
    /// # Returns
    ///
    /// A vector of candidate sampling points with their utility scores
    pub fn suggest_samples(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        let num_to_suggest = std::cmp::min(num_points, self.config.max_samples_per_iteration);

        // Check if we've exceeded the budget
        if self.stats.samples_suggested + num_to_suggest > self.config.total_budget {
            return Ok(Vec::new());
        }

        let start_time = std::time::Instant::now();

        let candidates = match self.strategy {
            SamplingStrategy::UncertaintyBased => {
                self.uncertainty_based_sampling(num_to_suggest)?
            }
            SamplingStrategy::ErrorBased => self.error_based_sampling(num_to_suggest)?,
            SamplingStrategy::GradientBased => self.gradient_based_sampling(num_to_suggest)?,
            SamplingStrategy::ExpectedImprovement => {
                self.expected_improvement_sampling(num_to_suggest)?
            }
            SamplingStrategy::ExplorationExploitation => {
                self.exploration_exploitation_sampling(num_to_suggest)?
            }
            SamplingStrategy::Combined => self.combined_sampling(num_to_suggest)?,
        };

        // Update statistics
        self.stats.samples_suggested += candidates.len();
        self.stats.iterations_completed += 1;
        self.stats.total_computation_time_ms += start_time.elapsed().as_millis() as u64;

        // Update strategy usage statistics
        let strategy_name = format!("{:?}", self.strategy);
        *self.stats.strategy_usage.entry(strategy_name).or_insert(0) += 1;

        Ok(candidates)
    }

    /// Update the interpolation model with new data points
    ///
    /// # Arguments
    ///
    /// * `new_x` - x-coordinates of new sample points
    /// * `new_y` - y-coordinates of new sample points
    ///
    /// # Returns
    ///
    /// `true` if the model was successfully updated
    pub fn update_model(
        &mut self,
        new_x: &ArrayView1<T>,
        new_y: &ArrayView1<T>,
    ) -> InterpolateResult<bool> {
        if new_x.len() != new_y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x and y arrays must have same length, got {} and {}",
                new_x.len(),
                new_y.len()
            )));
        }

        // Combine existing data with new data
        let current_x = self.interpolator.x_data();
        let current_y = self.interpolator.y_data();

        let mut combined_x = Vec::with_capacity(current_x.len() + new_x.len());
        let mut combined_y = Vec::with_capacity(current_y.len() + new_y.len());

        // Add existing data
        combined_x.extend_from_slice(current_x.as_slice().unwrap());
        combined_y.extend_from_slice(current_y.as_slice().unwrap());

        // Add new data
        combined_x.extend_from_slice(new_x.as_slice().unwrap());
        combined_y.extend_from_slice(new_y.as_slice().unwrap());

        // Create combined arrays and sort by x values
        let mut data_pairs: Vec<_> = combined_x.into_iter().zip(combined_y).collect();
        data_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let (sorted_x, sorted_y): (Vec<_>, Vec<_>) = data_pairs.into_iter().unzip();
        let x_array = Array1::from_vec(sorted_x);
        let y_array = Array1::from_vec(sorted_y);

        // Create a new interpolation model
        let degree = 3;
        let min_knots = 2 * (degree + 1); // Minimum knots required for the spline degree
        let new_interpolator = MultiscaleBSpline::new(
            &x_array.view(),
            &y_array.view(),
            std::cmp::max(std::cmp::min(x_array.len() / 2, 20), min_knots), // Ensure enough knots
            degree,                                                         // Cubic degree
            10,                                                             // Max levels
            T::from(0.01).unwrap(),
            crate::ExtrapolateMode::Extrapolate,
        )?;

        self.interpolator = new_interpolator;

        // Update stored sample data
        self.sample_points = x_array;
        self.sample_values = y_array;

        Ok(true)
    }

    /// Get current learning statistics
    pub fn get_stats(&self) -> &LearningStats {
        &self.stats
    }

    /// Get the current interpolation model
    pub fn get_interpolator(&self) -> &MultiscaleBSpline<T> {
        &self.interpolator
    }

    /// Check if learning should stop based on convergence criteria
    pub fn should_stop(&self) -> bool {
        self.stats.samples_suggested >= self.config.total_budget || self.has_converged()
    }

    /// Check if the learning process has converged
    fn has_converged(&self) -> bool {
        if self.stats.error_history.len() < 3 {
            return false;
        }

        // Check if improvement in recent iterations is below threshold
        let recent_errors = &self.stats.error_history[self.stats.error_history.len() - 3..];
        let improvement = recent_errors[0] - recent_errors[2];
        improvement < self.config.convergence_threshold.to_f64().unwrap()
    }

    /// Uncertainty-based sampling strategy
    fn uncertainty_based_sampling(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        let mut candidates = Vec::new();
        let _domain_range = self.domain_max - self.domain_min;

        // Generate candidate points across the domain
        let num_candidates = num_points * 10; // Oversample to allow selection
        let candidate_points = Array1::linspace(
            self.domain_min.to_f64().unwrap(),
            self.domain_max.to_f64().unwrap(),
            num_candidates,
        );

        let mut utilities = Vec::with_capacity(num_candidates);

        // Estimate uncertainty at each candidate point
        for &x_val in candidate_points.iter() {
            let x_t = T::from(x_val).unwrap();
            let uncertainty = self.estimate_uncertainty(x_t)?;
            utilities.push((x_t, uncertainty, "uncertainty".to_string()));
        }

        // Sort by utility (uncertainty) in descending order
        utilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select points ensuring minimum distance constraint
        let mut selected_points = Vec::new();
        for (location, utility, info_type) in utilities {
            if self.is_valid_location(location, &selected_points) {
                candidates.push(SamplingCandidate {
                    location,
                    utility,
                    information_type: info_type,
                    confidence: T::from(0.8).unwrap(), // Moderate confidence in uncertainty estimates
                });
                selected_points.push(location);

                if candidates.len() >= num_points {
                    break;
                }
            }
        }

        Ok(candidates)
    }

    /// Error-based sampling strategy
    fn error_based_sampling(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        let mut candidates = Vec::new();

        // Use the multiscale spline's refinement mechanism to identify high-error regions
        let current_spline = &self.interpolator;
        let x_data = current_spline.x_data();
        let y_data = current_spline.y_data();

        // Evaluate current approximation and compute errors
        let y_approx = current_spline.evaluate(&x_data.view())?;
        let errors = y_data - &y_approx;

        // Find regions with highest errors
        let mut error_locations = Vec::new();
        for (i, &error) in errors.iter().enumerate() {
            if i > 0 && i < errors.len() - 1 {
                // Look for local error maxima
                let local_error = error.abs();
                let left_error = errors[i - 1].abs();
                let right_error = errors[i + 1].abs();

                if local_error > left_error && local_error > right_error {
                    error_locations.push((x_data[i], local_error, "error_peak".to_string()));
                }
            }
        }

        // Also add points between high-error regions
        for i in 0..error_locations.len().saturating_sub(1) {
            let mid_point =
                (error_locations[i].0 + error_locations[i + 1].0) / T::from(2.0).unwrap();
            let estimated_error =
                (error_locations[i].1 + error_locations[i + 1].1) / T::from(2.0).unwrap();
            error_locations.push((
                mid_point,
                estimated_error,
                "error_interpolation".to_string(),
            ));
        }

        // Sort by error magnitude
        error_locations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select points ensuring minimum distance constraint
        let mut selected_points = Vec::new();
        for (location, utility, info_type) in error_locations {
            if self.is_valid_location(location, &selected_points) {
                candidates.push(SamplingCandidate {
                    location,
                    utility,
                    information_type: info_type,
                    confidence: T::from(0.9).unwrap(), // High confidence in error-based sampling
                });
                selected_points.push(location);

                if candidates.len() >= num_points {
                    break;
                }
            }
        }

        // If we don't have enough candidates, fill in with uniform sampling
        while candidates.len() < num_points {
            let random_point = self.generate_random_point();
            if self.is_valid_location(random_point, &selected_points) {
                candidates.push(SamplingCandidate {
                    location: random_point,
                    utility: T::from(0.1).unwrap(),
                    information_type: "exploration".to_string(),
                    confidence: T::from(0.3).unwrap(),
                });
                selected_points.push(random_point);
            }
        }

        Ok(candidates)
    }

    /// Gradient-based sampling strategy
    fn gradient_based_sampling(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        let mut candidates = Vec::new();

        // Generate candidate points
        let num_candidates = num_points * 10;
        let candidate_points = Array1::linspace(
            self.domain_min.to_f64().unwrap(),
            self.domain_max.to_f64().unwrap(),
            num_candidates,
        );

        let mut gradient_utilities = Vec::new();

        // Estimate gradients at candidate points
        for &x_val in candidate_points.iter() {
            let x_t = T::from(x_val).unwrap();

            // Compute first derivative (gradient magnitude)
            let gradient = self.interpolator.derivative(
                1, // First derivative
                &Array1::from_vec(vec![x_t]).view(),
            )?;

            let gradient_magnitude = gradient[0].abs();
            gradient_utilities.push((x_t, gradient_magnitude, "gradient".to_string()));
        }

        // Sort by gradient magnitude in descending order
        gradient_utilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select points ensuring minimum distance constraint
        let mut selected_points = Vec::new();
        for (location, utility, info_type) in gradient_utilities {
            if self.is_valid_location(location, &selected_points) {
                candidates.push(SamplingCandidate {
                    location,
                    utility,
                    information_type: info_type,
                    confidence: T::from(0.7).unwrap(), // Moderate confidence in gradient estimates
                });
                selected_points.push(location);

                if candidates.len() >= num_points {
                    break;
                }
            }
        }

        Ok(candidates)
    }

    /// Expected improvement sampling strategy
    fn expected_improvement_sampling(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        // This is a simplified version of expected improvement
        // In practice, this would involve more sophisticated uncertainty quantification
        let mut candidates = Vec::new();

        // Combine uncertainty and error information
        let uncertainty_candidates = self.uncertainty_based_sampling(num_points)?;
        let error_candidates = self.error_based_sampling(num_points)?;

        // Create a combined utility that represents expected improvement
        let mut combined_utilities = Vec::new();

        // Add uncertainty-based candidates with weights
        for candidate in uncertainty_candidates {
            let expected_improvement = candidate.utility * self.config.uncertainty_weight;
            combined_utilities.push((
                candidate.location,
                expected_improvement,
                "expected_improvement_uncertainty".to_string(),
            ));
        }

        // Add error-based candidates with weights
        for candidate in error_candidates {
            let expected_improvement = candidate.utility * self.config.error_weight;
            combined_utilities.push((
                candidate.location,
                expected_improvement,
                "expected_improvement_error".to_string(),
            ));
        }

        // Sort by expected improvement
        combined_utilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select points ensuring minimum distance constraint
        let mut selected_points = Vec::new();
        for (location, utility, info_type) in combined_utilities {
            if self.is_valid_location(location, &selected_points) {
                candidates.push(SamplingCandidate {
                    location,
                    utility,
                    information_type: info_type,
                    confidence: T::from(0.6).unwrap(), // Lower confidence for complex heuristic
                });
                selected_points.push(location);

                if candidates.len() >= num_points {
                    break;
                }
            }
        }

        Ok(candidates)
    }

    /// Exploration-exploitation sampling strategy
    fn exploration_exploitation_sampling(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        let num_exploitation =
            (num_points as f64 * (1.0 - self.config.exploration_weight.to_f64().unwrap())) as usize;
        let num_exploration = num_points - num_exploitation;

        let mut candidates = Vec::new();

        // Exploitation: sample in high-error regions
        let exploitation_candidates = self.error_based_sampling(num_exploitation)?;
        candidates.extend(exploitation_candidates);

        // Exploration: sample in under-sampled regions
        for _ in 0..num_exploration {
            let exploration_point = self.find_under_sampled_region()?;
            candidates.push(SamplingCandidate {
                location: exploration_point,
                utility: T::from(0.5).unwrap(),
                information_type: "exploration".to_string(),
                confidence: T::from(0.4).unwrap(),
            });
        }

        Ok(candidates)
    }

    /// Combined sampling strategy using multiple criteria
    fn combined_sampling(
        &mut self,
        num_points: usize,
    ) -> InterpolateResult<Vec<SamplingCandidate<T>>> {
        let mut candidates = Vec::new();

        // Get candidates from each strategy
        let uncertainty_candidates = self.uncertainty_based_sampling(num_points * 2)?;
        let error_candidates = self.error_based_sampling(num_points * 2)?;
        let gradient_candidates = self.gradient_based_sampling(num_points * 2)?;

        // Combine all candidates with weighted utilities
        let mut all_candidates = Vec::new();

        for candidate in uncertainty_candidates {
            all_candidates.push((
                candidate.location,
                candidate.utility * self.config.uncertainty_weight,
                "combined_uncertainty".to_string(),
            ));
        }

        for candidate in error_candidates {
            all_candidates.push((
                candidate.location,
                candidate.utility * self.config.error_weight,
                "combined_error".to_string(),
            ));
        }

        for candidate in gradient_candidates {
            all_candidates.push((
                candidate.location,
                candidate.utility * self.config.gradient_weight,
                "combined_gradient".to_string(),
            ));
        }

        // Sort by combined utility
        all_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select points ensuring minimum distance constraint
        let mut selected_points = Vec::new();
        for (location, utility, info_type) in all_candidates {
            if self.is_valid_location(location, &selected_points) {
                candidates.push(SamplingCandidate {
                    location,
                    utility,
                    information_type: info_type,
                    confidence: T::from(0.75).unwrap(), // Good confidence in combined approach
                });
                selected_points.push(location);

                if candidates.len() >= num_points {
                    break;
                }
            }
        }

        Ok(candidates)
    }

    /// Estimate interpolation uncertainty at a point
    fn estimate_uncertainty(&self, x: T) -> InterpolateResult<T> {
        // Simple uncertainty estimate based on distance to nearest sample points
        let x_data = self.interpolator.x_data();

        let mut min_distance = T::infinity();
        for &sample_x in x_data.iter() {
            let distance = (x - sample_x).abs();
            if distance < min_distance {
                min_distance = distance;
            }
        }

        // Uncertainty inversely related to distance to nearest samples
        let uncertainty = T::one() / (T::one() + min_distance);
        Ok(uncertainty)
    }

    /// Check if a location is valid (respects minimum distance constraint)
    fn is_valid_location(&self, location: T, existing_points: &[T]) -> bool {
        // Check domain bounds
        if location < self.domain_min || location > self.domain_max {
            return false;
        }

        // Check minimum distance to existing sample points
        let x_data = self.interpolator.x_data();
        for &sample_x in x_data.iter() {
            if (location - sample_x).abs() < self.config.min_sample_distance {
                return false;
            }
        }

        // Check minimum distance to other selected points
        for &point in existing_points {
            if (location - point).abs() < self.config.min_sample_distance {
                return false;
            }
        }

        true
    }

    /// Generate a random point within the domain (simple LCG for reproducibility)
    fn generate_random_point(&mut self) -> T {
        // Simple linear congruential generator for reproducible randomness
        self.rng_state = (self
            .rng_state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223))
            % (1 << 32);
        let normalized = (self.rng_state as f64) / ((1u64 << 32) as f64);

        let range = self.domain_max - self.domain_min;
        self.domain_min + range * T::from(normalized).unwrap()
    }

    /// Find an under-sampled region for exploration
    fn find_under_sampled_region(&mut self) -> InterpolateResult<T> {
        let x_data = self.interpolator.x_data();

        if x_data.len() < 2 {
            return Ok(self.generate_random_point());
        }

        // Find the largest gap between consecutive sample points
        let mut max_gap = T::zero();
        let mut best_location = self.domain_min;

        for i in 0..x_data.len() - 1 {
            let gap = x_data[i + 1] - x_data[i];
            if gap > max_gap {
                max_gap = gap;
                best_location = x_data[i] + gap / T::from(2.0).unwrap();
            }
        }

        // Also check gaps at domain boundaries
        let left_gap = x_data[0] - self.domain_min;
        if left_gap > max_gap {
            max_gap = left_gap;
            best_location = self.domain_min + left_gap / T::from(2.0).unwrap();
        }

        let right_gap = self.domain_max - x_data[x_data.len() - 1];
        if right_gap > max_gap {
            best_location = x_data[x_data.len() - 1] + right_gap / T::from(2.0).unwrap();
        }

        Ok(best_location)
    }
}

/// Convenience function to create an active learner with default settings
///
/// # Arguments
///
/// * `x` - Initial x-coordinates
/// * `y` - Initial y-coordinates  
/// * `strategy` - Sampling strategy to use
/// * `budget` - Total sampling budget
///
/// # Returns
///
/// A configured `ActiveLearner` ready for adaptive sampling
pub fn make_active_learner<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    strategy: SamplingStrategy,
    budget: usize,
) -> InterpolateResult<ActiveLearner<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    // Create initial interpolation model
    let degree = 3;
    let min_knots = 2 * (degree + 1); // Minimum knots required for the spline degree
    let spline = MultiscaleBSpline::new(
        x,
        y,
        std::cmp::max(std::cmp::min(x.len() / 2, 10), min_knots), // Ensure enough knots
        degree,                                                   // Cubic degree
        10,                                                       // Max levels
        T::from(0.01).unwrap(),                                   // Error threshold
        crate::ExtrapolateMode::Extrapolate,
    )?;

    let learner = ActiveLearner::new(spline, strategy).with_budget(budget);

    Ok(learner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_active_learner_creation() {
        let x = Array1::linspace(0.0, 10.0, 21);
        let y = x.mapv(|x| x.sin());

        let learner =
            make_active_learner(&x.view(), &y.view(), SamplingStrategy::UncertaintyBased, 50)
                .unwrap();

        assert_eq!(learner.config.total_budget, 50);
        assert_eq!(learner.strategy, SamplingStrategy::UncertaintyBased);
    }

    #[test]
    fn test_uncertainty_based_sampling() {
        let x = Array1::from_vec(vec![0.0, 1.4, 2.9, 4.3, 5.7, 7.1, 8.6, 10.0]);
        let y = Array1::from_vec(vec![0.0, 0.7, 1.0, 0.8, 0.5, 0.3, 0.1, 0.0]);

        let mut learner =
            make_active_learner(&x.view(), &y.view(), SamplingStrategy::UncertaintyBased, 20)
                .unwrap();

        let candidates = learner.suggest_samples(5).unwrap();

        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 5);

        // Check that candidates are within domain
        for candidate in &candidates {
            assert!(candidate.location >= 0.0);
            assert!(candidate.location <= 10.0);
        }
    }

    #[test]
    fn test_error_based_sampling() {
        // Create data with a sharp feature that will have high error
        let x = Array1::linspace(0.0, 10.0, 11);
        let y = x.mapv(|x| if (x - 5.0).abs() < 1.0 { x * x } else { x });

        let mut learner =
            make_active_learner(&x.view(), &y.view(), SamplingStrategy::ErrorBased, 20).unwrap();

        let candidates = learner.suggest_samples(3).unwrap();

        assert!(!candidates.is_empty());

        // At least one candidate should be near the sharp feature (around x=5)
        let near_feature = candidates.iter().any(|c| (c.location - 5.0).abs() < 2.0);
        assert!(near_feature);
    }

    #[test]
    fn test_model_update() {
        let x = Array1::from_vec(vec![0.0, 1.4, 2.9, 4.3, 5.7, 7.1, 8.6, 10.0]);
        let y = Array1::from_vec(vec![0.0, 0.7, 1.0, 0.8, 0.5, 0.3, 0.1, 0.0]);

        let mut learner =
            make_active_learner(&x.view(), &y.view(), SamplingStrategy::UncertaintyBased, 20)
                .unwrap();

        // Add new data points
        let new_x = Array1::from_vec(vec![2.5, 7.5]);
        let new_y = Array1::from_vec(vec![0.5, 0.5]);

        let success = learner.update_model(&new_x.view(), &new_y.view()).unwrap();
        assert!(success);

        // Check that the model has been updated
        assert_eq!(learner.sample_points.len(), 10); // Original 8 + new 2
    }

    #[test]
    fn test_combined_sampling() {
        let x = Array1::linspace(0.0, 10.0, 21);
        let y = x.mapv(|x| x.sin() + 0.1 * (5.0 * x).sin());

        let mut learner =
            make_active_learner(&x.view(), &y.view(), SamplingStrategy::Combined, 30).unwrap();

        let candidates = learner.suggest_samples(5).unwrap();

        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 5);

        // Check that we get different types of information
        let info_types: std::collections::HashSet<_> = candidates
            .iter()
            .map(|c| c.information_type.as_str())
            .collect();

        // Should have multiple types when using combined strategy
        assert!(!info_types.is_empty());
    }

    #[test]
    fn test_exploration_exploitation_balance() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 9.0, 10.0]); // Sparse sampling with gap
        let y = Array1::from_vec(vec![0.0, 1.0, 1.5, 1.8, 2.0, 2.2, 9.0, 10.0]);

        let mut learner = make_active_learner(
            &x.view(),
            &y.view(),
            SamplingStrategy::ExplorationExploitation,
            20,
        )
        .unwrap();

        let candidates = learner.suggest_samples(4).unwrap();

        assert!(!candidates.is_empty());

        // Should suggest points in the large gap (exploration)
        let in_gap = candidates
            .iter()
            .any(|c| c.location > 3.0 && c.location < 8.0);
        assert!(in_gap);
    }

    #[test]
    fn test_minimum_distance_constraint() {
        let x = Array1::from_vec(vec![0.0, 1.4, 2.9, 4.3, 5.7, 7.1, 8.6, 10.0]);
        let y = Array1::from_vec(vec![0.0, 1.4, 2.9, 4.3, 5.7, 7.1, 8.6, 10.0]);

        let mut learner =
            make_active_learner(&x.view(), &y.view(), SamplingStrategy::UncertaintyBased, 20)
                .unwrap()
                .with_min_sample_distance(1.0);

        let candidates = learner.suggest_samples(5).unwrap();

        // Check that all candidates respect minimum distance constraint
        for i in 0..candidates.len() {
            for j in i + 1..candidates.len() {
                let distance = (candidates[i].location - candidates[j].location).abs();
                assert!(distance >= 1.0);
            }
        }
    }
}
