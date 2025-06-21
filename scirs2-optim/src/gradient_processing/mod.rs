//! Gradient processing utilities for machine learning optimization
//!
//! This module provides comprehensive gradient manipulation utilities including
//! various clipping strategies, normalization, and other processing techniques.

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};

/// Gradient clipping configuration
#[derive(Debug, Clone)]
pub struct GradientClipConfig<A: Float> {
    /// Maximum allowed value for individual gradient elements
    pub max_value: Option<A>,
    /// Minimum allowed value for individual gradient elements  
    pub min_value: Option<A>,
    /// Maximum allowed L2 norm for the entire gradient vector
    pub max_norm: Option<A>,
    /// Maximum allowed L1 norm
    pub max_l1_norm: Option<A>,
    /// Whether to apply gradient centralization
    pub centralization: bool,
    /// Threshold for zeroing small gradients
    pub zero_threshold: Option<A>,
}

impl<A: Float> Default for GradientClipConfig<A> {
    fn default() -> Self {
        Self {
            max_value: None,
            min_value: None,
            max_norm: None,
            max_l1_norm: None,
            centralization: false,
            zero_threshold: None,
        }
    }
}

/// Gradient clipping processor
pub struct GradientProcessor<A: Float> {
    config: GradientClipConfig<A>,
}

impl<A: Float + ScalarOperand + Debug> Default for GradientProcessor<A> {
    fn default() -> Self {
        Self {
            config: GradientClipConfig::default(),
        }
    }
}

impl<A: Float + ScalarOperand + Debug> GradientProcessor<A> {
    /// Create a new gradient processor with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new gradient processor with a specific configuration
    pub fn with_config(config: GradientClipConfig<A>) -> Self {
        Self { config }
    }

    /// Set max value clipping
    pub fn set_max_value(&mut self, value: A) -> &mut Self {
        self.config.max_value = Some(value);
        self
    }

    /// Set min value clipping
    pub fn set_min_value(&mut self, value: A) -> &mut Self {
        self.config.min_value = Some(value);
        self
    }

    /// Set max L2 norm clipping
    pub fn set_max_norm(&mut self, value: A) -> &mut Self {
        self.config.max_norm = Some(value);
        self
    }

    /// Set max L1 norm clipping
    pub fn set_max_l1_norm(&mut self, value: A) -> &mut Self {
        self.config.max_l1_norm = Some(value);
        self
    }

    /// Enable gradient centralization
    pub fn set_centralization(&mut self, enabled: bool) -> &mut Self {
        self.config.centralization = enabled;
        self
    }

    /// Set threshold for zeroing small gradients
    pub fn set_zero_threshold(&mut self, value: A) -> &mut Self {
        self.config.zero_threshold = Some(value);
        self
    }

    /// Set value clipping range
    pub fn set_value_clip(&mut self, min: A, max: A) -> &mut Self {
        self.config.min_value = Some(min);
        self.config.max_value = Some(max);
        self
    }

    /// Set norm clipping
    pub fn set_norm_clip(&mut self, max_norm: A) -> &mut Self {
        self.config.max_norm = Some(max_norm);
        self
    }

    /// Set L1 norm clipping
    pub fn set_l1_norm_clip(&mut self, max_l1_norm: A) -> &mut Self {
        self.config.max_l1_norm = Some(max_l1_norm);
        self
    }

    /// Enable gradient centralization
    pub fn enable_centralization(&mut self) -> &mut Self {
        self.config.centralization = true;
        self
    }

    /// Process gradients according to configuration
    pub fn process<D: Dimension>(&self, gradients: &mut Array<A, D>) -> Result<()> {
        // Apply value clipping if configured
        if let (Some(min), Some(max)) = (self.config.min_value, self.config.max_value) {
            clip_gradients_by_value(gradients, min, max);
        }

        // Apply L2 norm clipping if configured
        if let Some(max_norm) = self.config.max_norm {
            clip_gradient_norm(gradients, max_norm)?;
        }

        // Apply L1 norm clipping if configured
        if let Some(max_l1_norm) = self.config.max_l1_norm {
            clip_gradient_l1_norm(gradients, max_l1_norm)?;
        }

        // Apply gradient centralization if enabled
        if self.config.centralization {
            gradient_centralization(gradients);
        }

        // Zero small gradients if threshold is set
        if let Some(threshold) = self.config.zero_threshold {
            zero_small_gradients(gradients, threshold);
        }

        Ok(())
    }
}

/// Clip gradient values to a specified range
pub fn clip_gradients_by_value<A, D>(
    gradients: &mut Array<A, D>,
    min_value: A,
    max_value: A,
) -> &mut Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    gradients.mapv_inplace(|x| {
        if x < min_value {
            min_value
        } else if x > max_value {
            max_value
        } else {
            x
        }
    });
    gradients
}

/// Clip gradient L2 norm (global gradient clipping)
pub fn clip_gradient_norm<A, D>(
    gradients: &mut Array<A, D>,
    max_norm: A,
) -> Result<&mut Array<A, D>>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    if max_norm <= A::zero() {
        return Err(OptimError::InvalidConfig(
            "max_norm must be positive".to_string(),
        ));
    }

    // Calculate current L2 norm
    let norm = gradients
        .iter()
        .fold(A::zero(), |acc, &x| acc + x * x)
        .sqrt();

    // If norm exceeds max_norm, scale gradients
    if norm > max_norm {
        let scale = max_norm / norm;
        gradients.mapv_inplace(|x| x * scale);
    }

    Ok(gradients)
}

/// Clip gradient L1 norm
pub fn clip_gradient_l1_norm<A, D>(
    gradients: &mut Array<A, D>,
    max_l1_norm: A,
) -> Result<&mut Array<A, D>>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    if max_l1_norm <= A::zero() {
        return Err(OptimError::InvalidConfig(
            "max_l1_norm must be positive".to_string(),
        ));
    }

    // Calculate current L1 norm
    let l1_norm = gradients.iter().fold(A::zero(), |acc, &x| acc + x.abs());

    // If norm exceeds max_l1_norm, scale gradients
    if l1_norm > max_l1_norm {
        let scale = max_l1_norm / l1_norm;
        gradients.mapv_inplace(|x| x * scale);
    }

    Ok(gradients)
}

/// Compute gradient centralization
pub fn gradient_centralization<A, D>(gradients: &mut Array<A, D>) -> &mut Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    // Calculate mean
    let sum = gradients.iter().fold(A::zero(), |acc, &x| acc + x);
    let mean = sum / A::from(gradients.len()).unwrap_or(A::one());

    // Subtract mean from each element
    gradients.mapv_inplace(|x| x - mean);

    gradients
}

/// Zero out small gradient values
pub fn zero_small_gradients<A, D>(gradients: &mut Array<A, D>, threshold: A) -> &mut Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    let abs_threshold = threshold.abs();

    gradients.mapv_inplace(|x| {
        if x.abs() < abs_threshold {
            A::zero()
        } else {
            x
        }
    });

    gradients
}

/// Gradient accumulation utility
#[derive(Debug, Clone)]
pub struct GradientAccumulator<A: Float, D: Dimension> {
    /// Accumulated gradients
    accumulated_gradients: Option<Array<A, D>>,
    /// Number of accumulated micro-batches
    num_accumulated: usize,
    /// Target number of micro-batches before step
    accumulation_steps: usize,
    /// Whether to average gradients (vs sum)
    average_gradients: bool,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GradientAccumulator<A, D> {
    /// Create a new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `accumulation_steps` - Number of micro-batches to accumulate before stepping
    /// * `average_gradients` - Whether to average gradients (true) or sum them (false)
    pub fn new(accumulation_steps: usize, average_gradients: bool) -> Self {
        Self {
            accumulated_gradients: None,
            num_accumulated: 0,
            accumulation_steps,
            average_gradients,
        }
    }

    /// Add gradients from a micro-batch
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradients from the current micro-batch
    ///
    /// # Returns
    ///
    /// `true` if enough gradients have been accumulated and it's time to step
    pub fn accumulate(&mut self, gradients: &Array<A, D>) -> bool {
        if self.accumulated_gradients.is_none() {
            self.accumulated_gradients = Some(gradients.clone());
        } else {
            let acc = self.accumulated_gradients.as_mut().unwrap();
            for (acc_val, &grad_val) in acc.iter_mut().zip(gradients.iter()) {
                *acc_val = *acc_val + grad_val;
            }
        }

        self.num_accumulated += 1;
        self.num_accumulated >= self.accumulation_steps
    }

    /// Get the accumulated gradients and reset the accumulator
    ///
    /// # Returns
    ///
    /// The accumulated gradients, ready for optimization step
    pub fn get_and_reset(&mut self) -> Option<Array<A, D>> {
        if let Some(mut gradients) = self.accumulated_gradients.take() {
            if self.average_gradients && self.num_accumulated > 0 {
                let scale = A::one() / A::from(self.num_accumulated).unwrap_or(A::one());
                gradients.mapv_inplace(|x| x * scale);
            }
            self.num_accumulated = 0;
            Some(gradients)
        } else {
            None
        }
    }

    /// Get current accumulation progress
    pub fn progress(&self) -> (usize, usize) {
        (self.num_accumulated, self.accumulation_steps)
    }

    /// Check if ready for optimization step
    pub fn is_ready(&self) -> bool {
        self.num_accumulated >= self.accumulation_steps
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.accumulated_gradients = None;
        self.num_accumulated = 0;
    }

    /// Change accumulation steps
    pub fn set_accumulation_steps(&mut self, steps: usize) {
        self.accumulation_steps = steps;
    }
}

/// Adaptive gradient clipping
///
/// Clips gradients based on the ratio of gradient norm to parameter norm.
/// This is particularly useful for transformer models.
pub fn adaptive_gradient_clipping<'a, A, D>(
    gradients: &'a mut Array<A, D>,
    parameters: &Array<A, D>,
    max_ratio: A,
) -> Result<&'a mut Array<A, D>>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    if max_ratio <= A::zero() {
        return Err(OptimError::InvalidConfig(
            "max_ratio must be positive".to_string(),
        ));
    }

    let grad_norm = gradients
        .iter()
        .fold(A::zero(), |acc, &x| acc + x * x)
        .sqrt();

    let param_norm = parameters
        .iter()
        .fold(A::zero(), |acc, &x| acc + x * x)
        .sqrt();

    if param_norm > A::zero() && grad_norm > A::zero() {
        let ratio = grad_norm / param_norm;
        if ratio > max_ratio {
            let scale = max_ratio / ratio;
            gradients.mapv_inplace(|x| x * scale);
        }
    }

    Ok(gradients)
}

/// Add noise to gradients for regularization
///
/// # Arguments
///
/// * `gradients` - Gradients to add noise to
/// * `noise_std` - Standard deviation of Gaussian noise to add
/// * `seed` - Optional seed for reproducible results
pub fn add_gradient_noise<A, D>(
    gradients: &mut Array<A, D>,
    noise_std: A,
    seed: Option<u64>,
) -> &mut Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

    if noise_std <= A::zero() {
        return gradients;
    }

    let mut rng = if let Some(s) = seed {
        ndarray_rand::rand::rngs::StdRng::seed_from_u64(s)
    } else {
        ndarray_rand::rand::rngs::StdRng::from_entropy()
    };

    let normal = Normal::new(0.0, noise_std.to_f64().unwrap_or(0.01)).unwrap();
    let noise = Array::random_using(gradients.raw_dim(), normal, &mut rng);

    gradients.zip_mut_with(&noise, |g, &n| {
        *g = *g + A::from(n).unwrap_or(A::zero());
    });

    gradients
}

/// Gradient masking and freezing utilities
///
/// Allows selective gradient updates by masking certain parameters
#[derive(Debug, Clone)]
pub struct GradientMask<A: Float, D: Dimension> {
    /// Mask indicating which parameters to update (true = update, false = freeze)
    mask: Array<bool, D>,
    /// Optional learning rate multipliers for each parameter
    lr_multipliers: Option<Array<A, D>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GradientMask<A, D> {
    /// Create a new gradient mask
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean mask indicating which parameters to update
    pub fn new(mask: Array<bool, D>) -> Self {
        Self {
            mask,
            lr_multipliers: None,
        }
    }

    /// Create a mask that freezes all parameters
    pub fn freeze_all(shape: D) -> Self {
        Self {
            mask: Array::from_elem(shape, false),
            lr_multipliers: None,
        }
    }

    /// Create a mask that updates all parameters
    pub fn update_all(shape: D) -> Self {
        Self {
            mask: Array::from_elem(shape, true),
            lr_multipliers: None,
        }
    }

    /// Set learning rate multipliers for different parameters
    pub fn with_lr_multipliers(mut self, multipliers: Array<A, D>) -> Self {
        self.lr_multipliers = Some(multipliers);
        self
    }

    /// Apply the mask to gradients
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradients to mask
    ///
    /// # Returns
    ///
    /// Masked gradients where frozen parameters have zero gradients
    pub fn apply_mask<'a>(&self, gradients: &'a mut Array<A, D>) -> &'a mut Array<A, D> {
        gradients.zip_mut_with(&self.mask, |grad, &should_update| {
            if !should_update {
                *grad = A::zero();
            }
        });

        // Apply learning rate multipliers if present
        if let Some(multipliers) = &self.lr_multipliers {
            gradients.zip_mut_with(multipliers, |grad, &mult| {
                *grad = *grad * mult;
            });
        }

        gradients
    }

    /// Freeze specific parameters by indices
    pub fn freeze_indices(&mut self, indices: &[usize]) -> Result<()> {
        let flat_mask = self.mask.as_slice_mut().ok_or_else(|| {
            OptimError::InvalidConfig("Cannot access mask as flat slice".to_string())
        })?;

        for &idx in indices {
            if idx < flat_mask.len() {
                flat_mask[idx] = false;
            } else {
                return Err(OptimError::InvalidConfig(format!(
                    "Index {} out of bounds for mask of size {}",
                    idx,
                    flat_mask.len()
                )));
            }
        }
        Ok(())
    }

    /// Unfreeze specific parameters by indices
    pub fn unfreeze_indices(&mut self, indices: &[usize]) -> Result<()> {
        let flat_mask = self.mask.as_slice_mut().ok_or_else(|| {
            OptimError::InvalidConfig("Cannot access mask as flat slice".to_string())
        })?;

        for &idx in indices {
            if idx < flat_mask.len() {
                flat_mask[idx] = true;
            } else {
                return Err(OptimError::InvalidConfig(format!(
                    "Index {} out of bounds for mask of size {}",
                    idx,
                    flat_mask.len()
                )));
            }
        }
        Ok(())
    }

    /// Get the number of frozen parameters
    pub fn num_frozen(&self) -> usize {
        self.mask.iter().filter(|&&x| !x).count()
    }

    /// Get the number of active (unfrozen) parameters
    pub fn num_active(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_gradient_processor() {
        let mut config = GradientClipConfig::default();
        config.max_value = Some(5.0);
        config.min_value = Some(-5.0);
        config.max_norm = Some(10.0);

        let processor = GradientProcessor::with_config(config);

        let mut gradients = Array1::from_vec(vec![-8.0, 3.0, 7.0, -2.0, 6.0]);
        processor.process(&mut gradients).unwrap();

        // Check value clipping
        assert_eq!(gradients[0], -5.0);
        assert_eq!(gradients[2], 5.0);
        assert_eq!(gradients[4], 5.0);
    }

    #[test]
    fn test_adaptive_clipping() {
        let mut gradients = Array1::from_vec(vec![3.0, 4.0]); // norm = 5
        let parameters = Array1::from_vec(vec![1.0, 0.0]); // norm = 1

        // Gradient/parameter ratio = 5/1 = 5, max_ratio = 2
        adaptive_gradient_clipping(&mut gradients, &parameters, 2.0).unwrap();

        // After clipping, ratio should be 2
        let new_grad_norm = gradients.iter().fold(0.0, |acc, &x| acc + x * x).sqrt();
        assert!((new_grad_norm - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = GradientAccumulator::new(3, true);

        // First micro-batch
        let grad1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(!accumulator.accumulate(&grad1));
        assert_eq!(accumulator.progress(), (1, 3));

        // Second micro-batch
        let grad2 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        assert!(!accumulator.accumulate(&grad2));
        assert_eq!(accumulator.progress(), (2, 3));

        // Third micro-batch - should trigger ready
        let grad3 = Array1::from_vec(vec![3.0, 4.0, 5.0]);
        assert!(accumulator.accumulate(&grad3));
        assert!(accumulator.is_ready());

        // Get accumulated gradients (should be averaged)
        let final_grads = accumulator.get_and_reset().unwrap();
        assert_relative_eq!(final_grads[0], 2.0, epsilon = 1e-6); // (1+2+3)/3
        assert_relative_eq!(final_grads[1], 3.0, epsilon = 1e-6); // (2+3+4)/3
        assert_relative_eq!(final_grads[2], 4.0, epsilon = 1e-6); // (3+4+5)/3

        // Should be reset now
        assert_eq!(accumulator.progress(), (0, 3));
        assert!(!accumulator.is_ready());
    }

    #[test]
    fn test_gradient_accumulator_sum_mode() {
        let mut accumulator = GradientAccumulator::new(2, false); // sum mode

        let grad1 = Array1::from_vec(vec![1.0, 2.0]);
        let grad2 = Array1::from_vec(vec![3.0, 4.0]);

        accumulator.accumulate(&grad1);
        accumulator.accumulate(&grad2);

        let final_grads = accumulator.get_and_reset().unwrap();
        assert_relative_eq!(final_grads[0], 4.0, epsilon = 1e-6); // 1+3
        assert_relative_eq!(final_grads[1], 6.0, epsilon = 1e-6); // 2+4
    }

    #[test]
    fn test_gradient_noise() {
        let mut gradients = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let original = gradients.clone();

        // Add noise with fixed seed for reproducibility
        add_gradient_noise(&mut gradients, 0.1, Some(42));

        // Gradients should be different but close to original
        for (i, (&orig, &noisy)) in original.iter().zip(gradients.iter()).enumerate() {
            assert!(
                (orig - noisy).abs() < 1.0,
                "Index {}: {} vs {}",
                i,
                orig,
                noisy
            );
        }
    }

    #[test]
    fn test_gradient_noise_zero_std() {
        let mut gradients = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let original = gradients.clone();

        // Zero noise should leave gradients unchanged
        add_gradient_noise(&mut gradients, 0.0, Some(42));

        for (orig, noisy) in original.iter().zip(gradients.iter()) {
            assert_relative_eq!(*orig, *noisy, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gradient_mask_creation() {
        let mask = Array1::from_vec(vec![true, false, true]);
        let grad_mask: GradientMask<f64, ndarray::Ix1> = GradientMask::new(mask);

        assert_eq!(grad_mask.num_active(), 2);
        assert_eq!(grad_mask.num_frozen(), 1);
    }

    #[test]
    fn test_gradient_mask_apply() {
        let mask = Array1::from_vec(vec![true, false, true]);
        let grad_mask: GradientMask<f64, ndarray::Ix1> = GradientMask::new(mask);
        let mut gradients = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        grad_mask.apply_mask(&mut gradients);

        assert_eq!(gradients.as_slice().unwrap(), &[1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_gradient_mask_freeze_unfreeze() {
        let mask = Array1::from_vec(vec![true, true, true]);
        let mut grad_mask: GradientMask<f64, ndarray::Ix1> = GradientMask::new(mask);

        // Freeze some indices
        grad_mask.freeze_indices(&[0, 2]).unwrap();
        assert_eq!(grad_mask.num_frozen(), 2);
        assert_eq!(grad_mask.num_active(), 1);

        // Unfreeze one index
        grad_mask.unfreeze_indices(&[0]).unwrap();
        assert_eq!(grad_mask.num_frozen(), 1);
        assert_eq!(grad_mask.num_active(), 2);
    }

    #[test]
    fn test_gradient_mask_with_lr_multipliers() {
        let mask = Array1::from_vec(vec![true, true, true]);
        let multipliers = Array1::from_vec(vec![1.0, 0.5, 2.0]);
        let grad_mask: GradientMask<f64, ndarray::Ix1> =
            GradientMask::new(mask).with_lr_multipliers(multipliers);
        let mut gradients = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        grad_mask.apply_mask(&mut gradients);

        assert_relative_eq!(gradients[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(gradients[1], 1.0, epsilon = 1e-6); // 2.0 * 0.5
        assert_relative_eq!(gradients[2], 6.0, epsilon = 1e-6); // 3.0 * 2.0
    }

    #[test]
    fn test_gradient_mask_freeze_all() {
        let grad_mask = GradientMask::<f64, ndarray::Ix1>::freeze_all(ndarray::Ix1(3));
        assert_eq!(grad_mask.num_frozen(), 3);
        assert_eq!(grad_mask.num_active(), 0);
    }

    #[test]
    fn test_gradient_mask_update_all() {
        let grad_mask = GradientMask::<f64, ndarray::Ix1>::update_all(ndarray::Ix1(3));
        assert_eq!(grad_mask.num_frozen(), 0);
        assert_eq!(grad_mask.num_active(), 3);
    }
}
