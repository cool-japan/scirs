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

#[cfg(test)]
mod tests {
    use super::*;
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
}
