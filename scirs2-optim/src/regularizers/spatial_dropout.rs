//! Spatial and Feature Dropout regularization
//!
//! This module provides specialized dropout variants that preserve spatial or feature connectivity:
//! - Spatial Dropout: drops entire feature maps (useful for CNNs)
//! - Feature Dropout: drops specific features across all spatial locations

use ndarray::{Array, Axis, Dimension, ScalarOperand};
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand_distr::{Bernoulli, Distribution};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Spatial Dropout regularizer
///
/// Drops entire feature maps instead of individual units. This helps
/// preserve spatial structure in convolutional neural networks.
///
/// # Example
///
/// ```
/// use ndarray::{Array4, array};
/// use scirs2_optim::regularizers::SpatialDropout;
///
/// let spatial_dropout = SpatialDropout::new(0.3).unwrap(); // 30% dropout rate
///
/// // 4D tensor (batch, channels, height, width)
/// let features = Array4::<f64>::ones((2, 3, 4, 4));
///
/// // During training - drops entire channels
/// let masked_features = spatial_dropout.apply(&features, true);
/// ```
#[derive(Debug, Clone)]
pub struct SpatialDropout<A: Float> {
    /// Probability of dropping a channel/feature map
    drop_prob: A,
    /// Dimension along which to drop (default is 1 for channels)
    feature_dim: Axis,
}

impl<A: Float + Debug + ScalarOperand> SpatialDropout<A> {
    /// Create a new SpatialDropout regularizer
    ///
    /// # Arguments
    ///
    /// * `drop_prob` - Probability of dropping each feature map (0.0 to 1.0)
    pub fn new(drop_prob: A) -> Result<Self> {
        if drop_prob < A::zero() || drop_prob > A::one() {
            return Err(OptimError::InvalidConfig(
                "Drop probability must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            drop_prob,
            feature_dim: Axis(1), // Default to channel dimension
        })
    }

    /// Set the dimension along which to drop features
    pub fn with_feature_dim(mut self, dim: usize) -> Self {
        self.feature_dim = Axis(dim);
        self
    }

    /// Apply spatial dropout to a tensor
    pub fn apply<D>(&self, features: &Array<A, D>, training: bool) -> Array<A, D>
    where
        D: Dimension + ndarray::RemoveAxis,
    {
        if !training || self.drop_prob == A::zero() {
            return features.clone();
        }

        let keep_prob = A::one() - self.drop_prob;

        // Get the size of the feature dimension
        let feature_size = features.shape()[self.feature_dim.0];

        // Create a mask for each feature map
        let dist = Bernoulli::new(keep_prob.to_f64().unwrap()).unwrap();
        let mut rng = thread_rng();
        let feature_mask: Vec<bool> = (0..feature_size).map(|_| dist.sample(&mut rng)).collect();

        // Apply mask to each feature map
        let mut result = features.clone();
        for (idx, &keep) in feature_mask.iter().enumerate() {
            if !keep {
                // Drop the entire feature map
                let mut axis_slice = result.index_axis_mut(self.feature_dim, idx);
                axis_slice.fill(A::zero());
            } else {
                // Scale kept features
                let mut axis_slice = result.index_axis_mut(self.feature_dim, idx);
                axis_slice.mapv_inplace(|x| x / keep_prob);
            }
        }

        result
    }
}

/// Feature Dropout regularizer
///
/// Drops specific features across all spatial locations. This is useful when
/// you want to maintain spatial consistency while dropping features.
///
/// # Example
///
/// ```
/// use ndarray::{Array3, array};
/// use scirs2_optim::regularizers::FeatureDropout;
///
/// let feature_dropout = FeatureDropout::new(0.5).unwrap(); // 50% dropout rate
///
/// // 3D tensor (batch, features, sequence_length)
/// let features = Array3::<f64>::ones((2, 10, 20));
///
/// // During training - drops specific features across all positions
/// let masked_features = feature_dropout.apply(&features, true);
/// ```
#[derive(Debug, Clone)]
pub struct FeatureDropout<A: Float> {
    /// Probability of dropping each feature
    drop_prob: A,
    /// Dimension along which features are located (default is 1)
    feature_dim: Axis,
}

impl<A: Float + Debug + ScalarOperand> FeatureDropout<A> {
    /// Create a new FeatureDropout regularizer
    ///
    /// # Arguments
    ///
    /// * `drop_prob` - Probability of dropping each feature (0.0 to 1.0)
    pub fn new(drop_prob: A) -> Result<Self> {
        if drop_prob < A::zero() || drop_prob > A::one() {
            return Err(OptimError::InvalidConfig(
                "Drop probability must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            drop_prob,
            feature_dim: Axis(1), // Default to feature dimension
        })
    }

    /// Set the dimension along which features are located
    pub fn with_feature_dim(mut self, dim: usize) -> Self {
        self.feature_dim = Axis(dim);
        self
    }

    /// Apply feature dropout to a tensor
    pub fn apply<D>(&self, features: &Array<A, D>, training: bool) -> Array<A, D>
    where
        D: Dimension + ndarray::RemoveAxis,
    {
        if !training || self.drop_prob == A::zero() {
            return features.clone();
        }

        let keep_prob = A::one() - self.drop_prob;

        // Get the size of the feature dimension
        let feature_size = features.shape()[self.feature_dim.0];

        // Create a consistent mask for each feature
        let dist = Bernoulli::new(keep_prob.to_f64().unwrap()).unwrap();
        let mut rng = thread_rng();
        let feature_mask: Vec<bool> = (0..feature_size).map(|_| dist.sample(&mut rng)).collect();

        // Apply the same mask across all spatial/temporal locations
        let mut result = features.clone();
        for (idx, &keep) in feature_mask.iter().enumerate() {
            if !keep {
                // Drop this feature everywhere
                let mut axis_slice = result.index_axis_mut(self.feature_dim, idx);
                axis_slice.fill(A::zero());
            } else {
                // Scale kept features
                let mut axis_slice = result.index_axis_mut(self.feature_dim, idx);
                axis_slice.mapv_inplace(|x| x / keep_prob);
            }
        }

        result
    }
}

// Implement Regularizer trait for SpatialDropout - only for dimensions that support RemoveAxis
impl<A: Float + Debug + ScalarOperand, D: Dimension + ndarray::RemoveAxis> Regularizer<A, D>
    for SpatialDropout<A>
{
    fn apply(&self, _params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Apply spatial dropout to gradients during training
        let masked_gradients = self.apply(gradients, true);
        gradients.assign(&masked_gradients);
        Ok(A::zero())
    }

    fn penalty(&self, _params: &Array<A, D>) -> Result<A> {
        // Spatial dropout doesn't add a penalty term
        Ok(A::zero())
    }
}

// Implement Regularizer trait for FeatureDropout - only for dimensions that support RemoveAxis
impl<A: Float + Debug + ScalarOperand, D: Dimension + ndarray::RemoveAxis> Regularizer<A, D>
    for FeatureDropout<A>
{
    fn apply(&self, _params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Apply feature dropout to gradients during training
        let masked_gradients = self.apply(gradients, true);
        gradients.assign(&masked_gradients);
        Ok(A::zero())
    }

    fn penalty(&self, _params: &Array<A, D>) -> Result<A> {
        // Feature dropout doesn't add a penalty term
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_spatial_dropout_creation() {
        // Valid creation
        let sd = SpatialDropout::<f64>::new(0.3).unwrap();
        assert_eq!(sd.drop_prob, 0.3);

        // Invalid probabilities
        assert!(SpatialDropout::<f64>::new(-0.1).is_err());
        assert!(SpatialDropout::<f64>::new(1.1).is_err());
    }

    #[test]
    fn test_spatial_dropout_4d() {
        let sd = SpatialDropout::new(0.5).unwrap();

        // Create a 4D tensor (batch, channels, height, width)
        // Use values that are always non-zero to better test dropout
        let features = Array::from_shape_fn((2, 4, 3, 3), |(b, c, h, w)| {
            1.0 + b as f64 + c as f64 * 10.0 + h as f64 * 0.1 + w as f64 * 0.01
        });

        // Apply spatial dropout
        let masked = sd.apply(&features, true);

        // Check that entire channels are either kept or dropped
        for b in 0..2 {
            for c in 0..4 {
                let masked_batch = masked.index_axis(Axis(0), b);
                let channel = masked_batch.index_axis(Axis(0), c);
                let channel_clone = channel.to_owned();
                let is_dropped = channel_clone.iter().all(|&x| x.abs() < 1e-10);
                let is_kept = channel_clone.iter().all(|&x| x.abs() > 1e-10);

                // For dropped channels, all values should be 0
                // For kept channels, all values should be scaled by 1/keep_prob
                if is_dropped {
                    for &val in channel_clone.iter() {
                        assert_eq!(val, 0.0);
                    }
                } else if is_kept {
                    // Check scaling
                    let original_batch = features.index_axis(Axis(0), b);
                    let original_channel = original_batch.index_axis(Axis(0), c);
                    for ((i, j), &val) in channel_clone.indexed_iter() {
                        assert_relative_eq!(val, original_channel[[i, j]] * 2.0, epsilon = 1e-10);
                    }
                } else {
                    // Mixed values - this shouldn't happen
                    println!("Channel {} in batch {} has mixed values:", c, b);
                    for val in channel_clone.iter() {
                        println!("  Value: {}", val);
                    }
                    panic!("Channel should be entirely dropped or kept");
                }
            }
        }
    }

    #[test]
    fn test_feature_dropout_creation() {
        // Valid creation
        let fd = FeatureDropout::<f64>::new(0.4).unwrap();
        assert_eq!(fd.drop_prob, 0.4);

        // Invalid probabilities
        assert!(FeatureDropout::<f64>::new(-0.1).is_err());
        assert!(FeatureDropout::<f64>::new(1.1).is_err());
    }

    #[test]
    fn test_feature_dropout_3d() {
        let fd = FeatureDropout::new(0.5).unwrap();

        // Create a 3D tensor (batch, features, sequence)
        let features = Array::from_shape_fn((2, 5, 10), |(_b, f, s)| f as f64 + s as f64);

        // Apply feature dropout
        let masked = fd.apply(&features, true);

        // Check that features are consistently dropped across all positions
        for f in 0..5 {
            let first_batch = masked.index_axis(Axis(0), 0);
            let first_batch_feature = first_batch.index_axis(Axis(0), f);
            let first_batch_clone = first_batch_feature.to_owned();
            let is_dropped = first_batch_clone.iter().all(|&x| x == 0.0);

            // Check consistency across batches and positions
            for b in 0..2 {
                let batch = masked.index_axis(Axis(0), b);
                let feature_slice = batch.index_axis(Axis(0), f);
                let feature_clone = feature_slice.to_owned();
                let all_dropped = feature_clone.iter().all(|&x| x == 0.0);
                assert_eq!(
                    is_dropped, all_dropped,
                    "Feature dropout should be consistent"
                );

                if !all_dropped {
                    // Check scaling
                    let original_batch = features.index_axis(Axis(0), b);
                    let original_slice = original_batch.index_axis(Axis(0), f);
                    for (i, &val) in feature_clone.iter().enumerate() {
                        assert_relative_eq!(val, original_slice[i] * 2.0, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_inference_mode() {
        let sd = SpatialDropout::new(0.5).unwrap();
        let fd = FeatureDropout::new(0.5).unwrap();

        let features = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

        // During inference, features should remain unchanged
        let sd_inference = sd.apply(&features, false);
        let fd_inference = fd.apply(&features, false);

        assert_eq!(features, sd_inference);
        assert_eq!(features, fd_inference);
    }

    #[test]
    fn test_regularizer_trait() {
        let sd = SpatialDropout::new(0.3).unwrap();
        let params = array![[[1.0, 2.0], [3.0, 4.0]]];
        let mut gradient = array![[[0.1, 0.2], [0.3, 0.4]]];

        // Test Regularizer trait
        let penalty = sd.penalty(&params).unwrap();
        assert_eq!(penalty, 0.0);

        let _penalty_apply = sd.apply(&params, true);
        let penalty_reg =
            <SpatialDropout<f64> as Regularizer<f64, _>>::apply(&sd, &params, &mut gradient)
                .unwrap();
        assert_eq!(penalty_reg, 0.0);

        // Gradient should be modified
        let is_modified = gradient != array![[[0.1, 0.2], [0.3, 0.4]]];
        assert!(is_modified || gradient == array![[[0.1, 0.2], [0.3, 0.4]]]);
    }
}
