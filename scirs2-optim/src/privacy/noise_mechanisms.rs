//! Noise mechanisms for differential privacy
//!
//! This module implements various noise mechanisms used in differential privacy,
//! including Gaussian, Laplace, Exponential, and advanced mechanisms like
//! tree aggregation and truncated noise.

use ndarray::{s, Array, Array1, Array2, ArrayBase, Data, DataMut, Dimension};
// Removed unused import Distribution
use num_traits::Float;
use rand::Rng;
use std::marker::PhantomData;

use crate::error::{OptimError, Result};

/// Trait for differential privacy noise mechanisms
pub trait NoiseMechanism<T: Float> {
    /// Add noise to maintain differential privacy for 1D arrays
    fn add_noise_1d(
        &mut self,
        data: &mut Array<T, ndarray::Ix1>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()>;

    /// Add noise to maintain differential privacy for 2D arrays
    fn add_noise_2d(
        &mut self,
        data: &mut Array<T, ndarray::Ix2>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()>;

    /// Add noise to maintain differential privacy for 3D arrays
    fn add_noise_3d(
        &mut self,
        data: &mut Array<T, ndarray::Ix3>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()>;

    /// Get the mechanism name
    fn name(&self) -> &'static str;

    /// Check if mechanism supports (ε, δ)-DP
    fn supports_delta(&self) -> bool;

    /// Get noise distribution parameters
    fn get_parameters(&self) -> NoiseParameters<T>;
}

/// Noise mechanism parameters
#[derive(Debug, Clone)]
pub struct NoiseParameters<T: Float> {
    pub mechanism_type: String,
    pub scale: T,
    pub sensitivity: T,
    pub epsilon: T,
    pub delta: Option<T>,
    pub shape: Option<T>,
    pub rate: Option<T>,
}

/// Gaussian noise mechanism for (ε, δ)-differential privacy
pub struct GaussianMechanism<T: Float> {
    rng: scirs2_core::random::Random<rand::prelude::StdRng>,
    _phantom: PhantomData<T>,
}

/// Laplace noise mechanism for ε-differential privacy
pub struct LaplaceMechanism<T: Float> {
    rng: scirs2_core::random::Random<rand::prelude::StdRng>,
    _phantom: PhantomData<T>,
}

/// Exponential mechanism for discrete optimization
pub struct ExponentialMechanism<T: Float> {
    rng: scirs2_core::random::Random<rand::prelude::StdRng>,
    _qualityfunction: Box<dyn Fn(&T) -> T + Send + Sync>,
    _phantom: PhantomData<T>,
}

/// Truncated noise mechanism for bounded sensitivity
pub struct TruncatedNoiseMechanism<T: Float> {
    basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    truncationbound: T,
    _phantom: PhantomData<T>,
}

/// Tree aggregation mechanism for hierarchical noise
pub struct TreeAggregationMechanism<T: Float> {
    tree_height: usize,
    basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    _phantom: PhantomData<T>,
}

/// Sparse Vector Technique mechanism
pub struct SparseVectorMechanism<T: Float> {
    threshold: T,
    budget_fraction: T,
    queries_answered: usize,
    max_queries: usize,
    basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    _phantom: PhantomData<T>,
}

/// Smooth sensitivity mechanism
pub struct SmoothSensitivityMechanism<T: Float> {
    beta: T,
    sensitivity_function: Box<dyn Fn(&[T]) -> T + Send + Sync>,
    _phantom: PhantomData<T>,
}

/// Advanced noise calibration
pub struct NoiseCalibrator<T: Float> {
    /// Target privacy parameters
    target_epsilon: T,
    target_delta: Option<T>,

    /// Sensitivity bounds
    l2_sensitivity: T,
    l1_sensitivity: T,
    linf_sensitivity: T,

    /// Mechanism selection strategy
    selection_strategy: MechanismSelectionStrategy,

    /// Adaptive noise scaling
    adaptive_scaling: bool,
    scaling_factor: T,
    _phantom: PhantomData<T>,
}

/// Strategy for selecting noise mechanism
#[derive(Debug, Clone, Copy)]
pub enum MechanismSelectionStrategy {
    /// Always use Gaussian mechanism
    AlwaysGaussian,

    /// Always use Laplace mechanism
    AlwaysLaplace,

    /// Choose based on privacy parameters
    PrivacyOptimal,

    /// Choose based on utility optimization
    UtilityOptimal,

    /// Adaptive selection based on data characteristics
    Adaptive,
}

impl<T> GaussianMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    /// Create a new Gaussian mechanism
    pub fn new() -> Self {
        Self {
            rng: scirs2_core::random::Random::seed(42), // Use seeded RNG for thread safety
            _phantom: PhantomData,
        }
    }

    /// Compute noise scale for Gaussian mechanism
    pub fn compute_noise_scale(sensitivity: T, epsilon: T, delta: T) -> Result<T> {
        if epsilon <= T::zero() || delta <= T::zero() || delta >= T::one() {
            return Err(OptimError::InvalidConfig(
                "Invalid privacy parameters for Gaussian mechanism".to_string(),
            ));
        }

        // Standard Gaussian mechanism: σ = √(2 ln(1.25/δ)) * Δ / ε
        let ln_term = (T::one() + T::from(0.25).unwrap() / delta).ln();
        let sigma = (T::from(2.0).unwrap() * ln_term).sqrt() * sensitivity / epsilon;

        Ok(sigma)
    }

    /// Generic noise addition implementation
    fn add_noise_generic<S, D>(
        &mut self,
        data: &mut ArrayBase<S, D>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()>
    where
        S: DataMut<Elem = T>,
        D: Dimension,
    {
        let delta = delta.ok_or_else(|| {
            OptimError::InvalidConfig("Gaussian mechanism requires delta parameter".to_string())
        })?;

        let sigma = Self::compute_noise_scale(sensitivity, epsilon, delta)?;
        let sigma_f64 = sigma.to_f64().unwrap_or(1.0);

        data.mapv_inplace(|x| {
            // Use Box-Muller transformation to generate normal random numbers
            // since direct sampling with scirs2_core::Random has trait issues
            let u1: f64 = self.rng.gen_range(0.0..1.0);
            let u2: f64 = self.rng.gen_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let noise = T::from(z0 * sigma_f64).unwrap();
            x + noise
        });

        Ok(())
    }
}

impl<T> NoiseMechanism<T> for GaussianMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    fn add_noise_1d(
        &mut self,
        data: &mut Array<T, ndarray::Ix1>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.add_noise_generic(data, sensitivity, epsilon, delta)
    }

    fn add_noise_2d(
        &mut self,
        data: &mut Array<T, ndarray::Ix2>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.add_noise_generic(data, sensitivity, epsilon, delta)
    }

    fn add_noise_3d(
        &mut self,
        data: &mut Array<T, ndarray::Ix3>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.add_noise_generic(data, sensitivity, epsilon, delta)
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }

    fn supports_delta(&self) -> bool {
        true
    }

    fn get_parameters(&self) -> NoiseParameters<T> {
        NoiseParameters {
            mechanism_type: "Gaussian".to_string(),
            scale: T::zero(), // Will be computed dynamically
            sensitivity: T::zero(),
            epsilon: T::zero(),
            delta: Some(T::zero()),
            shape: None,
            rate: None,
        }
    }
}

impl<T> LaplaceMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    /// Create a new Laplace mechanism
    pub fn new() -> Self {
        Self {
            rng: scirs2_core::random::Random::seed(43), // Use seeded RNG for thread safety
            _phantom: PhantomData,
        }
    }

    /// Compute noise scale for Laplace mechanism
    pub fn compute_noise_scale(sensitivity: T, epsilon: T) -> Result<T> {
        if epsilon <= T::zero() {
            return Err(OptimError::InvalidConfig(
                "Epsilon must be positive for Laplace mechanism".to_string(),
            ));
        }

        // Laplace mechanism: b = Δ / ε
        Ok(sensitivity / epsilon)
    }

    /// Generic noise addition implementation
    fn add_noise_generic<S, D>(
        &mut self,
        data: &mut ArrayBase<S, D>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()>
    where
        S: DataMut<Elem = T>,
        D: Dimension,
    {
        let scale = Self::compute_noise_scale(sensitivity, epsilon)?;
        let scale_f64 = scale.to_f64().unwrap_or(1.0);

        // Implement Laplace distribution using transformation method
        // If U is uniform on [0,1], then Laplace(μ, b) = μ - b*sgn(U-0.5)*ln(1-2|U-0.5|)

        data.mapv_inplace(|x| {
            let u: f64 = self.rng.gen_range(0.0..1.0);
            let laplace_sample = if u < 0.5 {
                scale_f64 * (2.0 * u).ln()
            } else {
                -scale_f64 * (2.0 * (1.0 - u)).ln()
            };
            let noise = T::from(laplace_sample).unwrap();
            x + noise
        });

        Ok(())
    }
}

impl<T> NoiseMechanism<T> for LaplaceMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    fn add_noise_1d(
        &mut self,
        data: &mut Array<T, ndarray::Ix1>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.add_noise_generic(data, sensitivity, epsilon, delta)
    }

    fn add_noise_2d(
        &mut self,
        data: &mut Array<T, ndarray::Ix2>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.add_noise_generic(data, sensitivity, epsilon, delta)
    }

    fn add_noise_3d(
        &mut self,
        data: &mut Array<T, ndarray::Ix3>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        self.add_noise_generic(data, sensitivity, epsilon, delta)
    }

    fn name(&self) -> &'static str {
        "Laplace"
    }

    fn supports_delta(&self) -> bool {
        false
    }

    fn get_parameters(&self) -> NoiseParameters<T> {
        NoiseParameters {
            mechanism_type: "Laplace".to_string(),
            scale: T::zero(),
            sensitivity: T::zero(),
            epsilon: T::zero(),
            delta: None,
            shape: None,
            rate: None,
        }
    }
}

impl<T> ExponentialMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    /// Create a new exponential mechanism
    pub fn new(_qualityfunction: Box<dyn Fn(&T) -> T + Send + Sync>) -> Self {
        Self {
            rng: scirs2_core::random::Random::seed(44), // Use seeded RNG for thread safety
            _qualityfunction,
            _phantom: PhantomData,
        }
    }

    /// Select output using exponential mechanism
    pub fn select_output(&mut self, candidates: &[T], sensitivity: T, epsilon: T) -> Result<T> {
        if candidates.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No candidates provided".to_string(),
            ));
        }

        // Compute quality scores
        let scores: Vec<T> = candidates
            .iter()
            .map(|x| (self._qualityfunction)(x))
            .collect();

        // Compute exponential weights
        let max_score = scores.iter().cloned().fold(T::neg_infinity(), T::max);
        let weights: Vec<f64> = scores
            .iter()
            .map(|&score| {
                let normalized_score = score - max_score;
                let exponent = epsilon * normalized_score / (T::from(2.0).unwrap() * sensitivity);
                exponent.to_f64().unwrap_or(0.0).exp()
            })
            .collect();

        // Sample according to weights
        let total_weight: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        let random_val: f64 = self.rng.gen_range(0.0..total_weight);

        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if random_val <= cumulative {
                return Ok(candidates[i]);
            }
        }

        // Fallback (should not happen)
        Ok(candidates[candidates.len() - 1])
    }
}

impl<T> TruncatedNoiseMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    /// Create a new truncated noise mechanism
    pub fn new(_base_mechanism: Box<dyn NoiseMechanism<T> + Send>, truncationbound: T) -> Self {
        Self {
            basemechanism: _base_mechanism,
            truncationbound,
            _phantom: PhantomData,
        }
    }
}

impl<T> NoiseMechanism<T> for TruncatedNoiseMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    fn add_noise_1d(
        &mut self,
        data: &mut Array<T, ndarray::Ix1>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        // Apply base mechanism first
        self.basemechanism
            .add_noise_1d(data, sensitivity, epsilon, delta)?;

        // Truncate to bounds
        data.mapv_inplace(|x| x.max(-self.truncationbound).min(self.truncationbound));

        Ok(())
    }

    fn add_noise_2d(
        &mut self,
        data: &mut Array<T, ndarray::Ix2>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        // Apply base mechanism first
        self.basemechanism
            .add_noise_2d(data, sensitivity, epsilon, delta)?;

        // Truncate to bounds
        data.mapv_inplace(|x| x.max(-self.truncationbound).min(self.truncationbound));

        Ok(())
    }

    fn add_noise_3d(
        &mut self,
        data: &mut Array<T, ndarray::Ix3>,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<()> {
        // Apply base mechanism first
        self.basemechanism
            .add_noise_3d(data, sensitivity, epsilon, delta)?;

        // Truncate to bounds
        data.mapv_inplace(|x| x.max(-self.truncationbound).min(self.truncationbound));

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Truncated"
    }

    fn supports_delta(&self) -> bool {
        self.basemechanism.supports_delta()
    }

    fn get_parameters(&self) -> NoiseParameters<T> {
        let mut params = self.basemechanism.get_parameters();
        params.mechanism_type = format!("Truncated_{}", params.mechanism_type);
        params
    }
}

impl<T> TreeAggregationMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform + std::iter::Sum,
{
    /// Create a new tree aggregation mechanism
    pub fn new(_tree_height: usize, basemechanism: Box<dyn NoiseMechanism<T> + Send>) -> Self {
        Self {
            tree_height: _tree_height,
            basemechanism,
            _phantom: PhantomData,
        }
    }

    /// Aggregate values using binary tree with noise
    pub fn aggregate_with_tree(
        &mut self,
        values: &[T],
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<T> {
        if values.is_empty() {
            return Ok(T::zero());
        }

        let mut current_level = values.to_vec();
        let level_epsilon = epsilon / T::from(self.tree_height).unwrap();

        // Aggregate level by level
        for _level in 0..self.tree_height {
            if current_level.len() <= 1 {
                break;
            }

            let mut next_level = Vec::new();

            // Pair up values and add noise to sums
            for chunk in current_level.chunks(2) {
                let mut sum = Array1::from_vec(vec![chunk.iter().cloned().sum()]);
                self.basemechanism
                    .add_noise_1d(&mut sum, sensitivity, level_epsilon, delta)?;
                next_level.push(sum[0]);
            }

            current_level = next_level;
        }

        Ok(current_level.into_iter().sum())
    }
}

impl<T> SparseVectorMechanism<T>
where
    T: Float + Default + Clone + Send + Sync + rand_distr::uniform::SampleUniform,
{
    /// Create a new sparse vector mechanism
    pub fn new(
        threshold: T,
        budget_fraction: T,
        max_queries: usize,
        basemechanism: Box<dyn NoiseMechanism<T> + Send>,
    ) -> Self {
        Self {
            threshold,
            budget_fraction,
            queries_answered: 0,
            max_queries,
            basemechanism,
            _phantom: PhantomData,
        }
    }

    /// Answer query if above threshold
    pub fn answer_query(
        &mut self,
        query_result: T,
        sensitivity: T,
        epsilon: T,
        delta: Option<T>,
    ) -> Result<Option<T>> {
        if self.queries_answered >= self.max_queries {
            return Ok(None);
        }

        // Add noise to threshold
        let threshold_epsilon = epsilon * self.budget_fraction;
        let mut noisy_threshold = Array1::from_vec(vec![self.threshold]);
        self.basemechanism.add_noise_1d(
            &mut noisy_threshold,
            sensitivity,
            threshold_epsilon,
            delta,
        )?;
        let noisy_threshold = noisy_threshold[0];

        // Check if query _result exceeds noisy threshold
        if query_result > noisy_threshold {
            // Add noise to actual _result
            let result_epsilon = epsilon * (T::one() - self.budget_fraction);
            let mut noisy_result = Array1::from_vec(vec![query_result]);
            self.basemechanism.add_noise_1d(
                &mut noisy_result,
                sensitivity,
                result_epsilon,
                delta,
            )?;

            self.queries_answered += 1;
            Ok(Some(noisy_result[0]))
        } else {
            Ok(Some(T::zero())) // Below threshold
        }
    }
}

impl<T> NoiseCalibrator<T>
where
    T: Float
        + Default
        + Clone
        + Send
        + Sync
        + rand_distr::uniform::SampleUniform
        + std::iter::Sum
        + 'static,
{
    /// Create a new noise calibrator
    pub fn new(
        target_epsilon: T,
        target_delta: Option<T>,
        l2_sensitivity: T,
        selection_strategy: MechanismSelectionStrategy,
    ) -> Self {
        Self {
            target_epsilon,
            target_delta,
            l2_sensitivity,
            l1_sensitivity: l2_sensitivity, // Default assumption
            linf_sensitivity: l2_sensitivity,
            selection_strategy,
            adaptive_scaling: false,
            scaling_factor: T::one(),
            _phantom: PhantomData,
        }
    }

    /// Select optimal noise mechanism
    pub fn select_mechanism(&self) -> Box<dyn NoiseMechanism<T> + Send> {
        match self.selection_strategy {
            MechanismSelectionStrategy::AlwaysGaussian => Box::new(GaussianMechanism::new()),
            MechanismSelectionStrategy::AlwaysLaplace => Box::new(LaplaceMechanism::new()),
            MechanismSelectionStrategy::PrivacyOptimal => {
                if self.target_delta.is_some() {
                    Box::new(GaussianMechanism::new())
                } else {
                    Box::new(LaplaceMechanism::new())
                }
            }
            MechanismSelectionStrategy::UtilityOptimal => {
                // Gaussian typically provides better utility for same privacy
                if self.target_delta.is_some() {
                    Box::new(GaussianMechanism::new())
                } else {
                    Box::new(LaplaceMechanism::new())
                }
            }
            MechanismSelectionStrategy::Adaptive => {
                // Choose based on sensitivity characteristics
                if self.l2_sensitivity < self.l1_sensitivity * T::from(0.7).unwrap() {
                    Box::new(GaussianMechanism::new())
                } else {
                    Box::new(LaplaceMechanism::new())
                }
            }
        }
    }

    /// Calibrate noise for optimal privacy-utility tradeoff
    pub fn calibrate_noise<S, D>(
        &mut self,
        data: &mut ArrayBase<S, D>,
        actual_sensitivity: Option<T>,
    ) -> Result<NoiseCalibrationResult<T>>
    where
        S: DataMut<Elem = T>,
        D: Dimension,
    {
        let sensitivity = actual_sensitivity.unwrap_or(self.l2_sensitivity);

        // Adaptive scaling based on data characteristics
        if self.adaptive_scaling {
            let data_scale = self.estimate_data_scale(data);
            self.scaling_factor = data_scale / sensitivity;
        }

        let adjusted_sensitivity = sensitivity * self.scaling_factor;
        let mut mechanism = self.select_mechanism();

        let start_time = std::time::Instant::now();

        // Dispatch to appropriate dimension-specific method
        match data.ndim() {
            1 => {
                // Cast to 1D array
                let data_1d: &mut Array<T, ndarray::Ix1> = unsafe { std::mem::transmute(data) };
                mechanism.add_noise_1d(
                    data_1d,
                    adjusted_sensitivity,
                    self.target_epsilon,
                    self.target_delta,
                )?;
            }
            2 => {
                // Cast to 2D array
                let data_2d: &mut Array<T, ndarray::Ix2> = unsafe { std::mem::transmute(data) };
                mechanism.add_noise_2d(
                    data_2d,
                    adjusted_sensitivity,
                    self.target_epsilon,
                    self.target_delta,
                )?;
            }
            3 => {
                // Cast to 3D array
                let data_3d: &mut Array<T, ndarray::Ix3> = unsafe { std::mem::transmute(data) };
                mechanism.add_noise_3d(
                    data_3d,
                    adjusted_sensitivity,
                    self.target_epsilon,
                    self.target_delta,
                )?;
            }
            _ => {
                return Err(OptimError::InvalidConfig(
                    "Unsupported array dimension for noise calibration".to_string(),
                ));
            }
        }

        let calibration_time = start_time.elapsed();

        Ok(NoiseCalibrationResult {
            mechanism_used: mechanism.name().to_string(),
            noise_scale: adjusted_sensitivity / self.target_epsilon,
            sensitivity_used: adjusted_sensitivity,
            scaling_factor: self.scaling_factor,
            calibration_time_us: calibration_time.as_micros() as u64,
            privacy_parameters: PrivacyParameters {
                epsilon: self.target_epsilon,
                delta: self.target_delta,
            },
        })
    }

    fn estimate_data_scale<S, D>(&self, data: &ArrayBase<S, D>) -> T
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        // Estimate scale of data for adaptive noise calibration
        let sum_squares = data.iter().map(|&x| x * x).sum::<T>();
        let n = T::from(data.len()).unwrap();
        (sum_squares / n).sqrt()
    }
}

/// Result of noise calibration
#[derive(Debug, Clone)]
pub struct NoiseCalibrationResult<T: Float> {
    pub mechanism_used: String,
    pub noise_scale: T,
    pub sensitivity_used: T,
    pub scaling_factor: T,
    pub calibration_time_us: u64,
    pub privacy_parameters: PrivacyParameters<T>,
}

/// Privacy parameters used
#[derive(Debug, Clone)]
pub struct PrivacyParameters<T: Float> {
    pub epsilon: T,
    pub delta: Option<T>,
}

/// Generate correlated noise for matrix operations
#[allow(dead_code)]
pub fn generate_correlated_gaussian_noise<T>(
    shape: (usize, usize),
    correlation_matrix: &Array2<T>,
    scale: T,
    rng: &mut scirs2_core::random::Random,
) -> Result<Array2<T>>
where
    T: Float + Default + Clone + rand_distr::uniform::SampleUniform + 'static,
{
    let (rows, cols) = shape;

    if correlation_matrix.nrows() != cols || correlation_matrix.ncols() != cols {
        return Err(OptimError::InvalidConfig(
            "Correlation _matrix dimensions mismatch".to_string(),
        ));
    }

    let mut noise = Array2::zeros((rows, cols));
    let scale_f64 = scale.to_f64().unwrap_or(1.0);

    // Generate independent noise using Box-Muller transformation
    for i in 0..rows {
        for j in 0..cols {
            let u1: f64 = rng.gen_range(0.0..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let gaussian_sample = z0 * scale_f64;
            noise[[i, j]] = T::from(gaussian_sample).unwrap();
        }
    }

    // Apply correlation (simplified - would use Cholesky decomposition in practice)
    for i in 0..rows {
        let row_slice = noise.slice(s![i, ..]);
        let correlated_row = correlation_matrix.dot(&row_slice);
        for (j, &val) in correlated_row.iter().enumerate() {
            noise[[i, j]] = val;
        }
    }

    Ok(noise)
}

/// Validate differential privacy parameters
#[allow(dead_code)]
pub fn validate_privacy_parameters<T: Float>(epsilon: T, delta: Option<T>) -> Result<()> {
    if epsilon <= T::zero() {
        return Err(OptimError::InvalidConfig(
            "Epsilon must be positive".to_string(),
        ));
    }

    if let Some(d) = delta {
        if d < T::zero() || d >= T::one() {
            return Err(OptimError::InvalidConfig(
                "Delta must be in [0, 1)".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mechanism() {
        let mut mechanism = GaussianMechanism::<f64>::new();
        let mut data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = mechanism.add_noise_1d(&mut data, 1.0, 1.0, Some(1e-5));
        assert!(result.is_ok());
        assert_eq!(mechanism.name(), "Gaussian");
        assert!(mechanism.supports_delta());
    }

    #[test]
    fn test_laplace_mechanism() {
        let mut mechanism = LaplaceMechanism::<f64>::new();
        let mut data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = mechanism.add_noise_1d(&mut data, 1.0, 1.0, None);
        assert!(result.is_ok());
        assert_eq!(mechanism.name(), "Laplace");
        assert!(!mechanism.supports_delta());
    }

    #[test]
    fn test_noise_scale_computation() {
        let gaussian_scale = GaussianMechanism::<f64>::compute_noise_scale(1.0, 1.0, 1e-5);
        assert!(gaussian_scale.is_ok());
        assert!(gaussian_scale.unwrap() > 0.0);

        let laplace_scale = LaplaceMechanism::<f64>::compute_noise_scale(1.0, 1.0);
        assert!(laplace_scale.is_ok());
        assert_eq!(laplace_scale.unwrap(), 1.0);
    }

    #[test]
    fn test_truncated_mechanism() {
        let base = Box::new(LaplaceMechanism::<f64>::new());
        let mut truncated = TruncatedNoiseMechanism::new(base, 5.0);
        let mut data = Array1::from_vec(vec![100.0]); // Large value

        let result = truncated.add_noise_1d(&mut data, 1.0, 0.1, None);
        assert!(result.is_ok());
        assert!(data[0].abs() <= 5.0); // Should be truncated
    }

    #[test]
    fn test_exponential_mechanism() {
        let quality_fn = Box::new(|x: &f64| -*x); // Prefer smaller values
        let mut mechanism = ExponentialMechanism::new(quality_fn);

        let candidates = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mechanism.select_output(&candidates, 1.0, 1.0);

        assert!(result.is_ok());
        assert!(candidates.contains(&result.unwrap()));
    }

    #[test]
    fn test_noise_calibrator() {
        let calibrator = NoiseCalibrator::<f64>::new(
            1.0,
            Some(1e-5),
            1.0,
            MechanismSelectionStrategy::PrivacyOptimal,
        );

        let mechanism = calibrator.select_mechanism();
        assert_eq!(mechanism.name(), "Gaussian");
    }

    #[test]
    fn test_sparse_vector_mechanism() {
        let base = Box::new(LaplaceMechanism::<f64>::new());
        let mut svm = SparseVectorMechanism::new(5.0, 0.5, 3, base);

        // Query above threshold
        let result1 = svm.answer_query(10.0, 1.0, 1.0, None);
        assert!(result1.is_ok());
        assert!(result1.unwrap().is_some());

        // Query below threshold
        let result2 = svm.answer_query(1.0, 1.0, 1.0, None);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_privacy_parameter_validation() {
        assert!(validate_privacy_parameters(1.0, Some(1e-5)).is_ok());
        assert!(validate_privacy_parameters(-1.0, Some(1e-5)).is_err());
        assert!(validate_privacy_parameters(1.0, Some(1.5)).is_err());
    }
}
