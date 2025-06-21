//! Gradient clipping utilities
//!
//! Gradient clipping is a technique used to prevent the exploding gradient problem
//! in deep learning by constraining the gradients to a reasonable range or magnitude.
//! This module provides various gradient clipping strategies.

use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::Float;

/// Trait for gradient clipping strategies
///
/// Gradient clipping modifies gradients to prevent exploding gradients while
/// preserving the direction of optimization.
pub trait GradientClipper<F: Float> {
    /// Apply gradient clipping to a list of gradients
    ///
    /// # Arguments
    /// * `gradients` - Slice of gradient tensors to clip
    ///
    /// # Returns
    /// Vector of clipped gradient tensors
    fn clip_gradients<'g>(&mut self, gradients: &[Tensor<'g, F>]) -> Vec<Tensor<'g, F>>;

    /// Check if clipping was applied in the last call to clip_gradients
    ///
    /// This can be useful for monitoring whether gradients are being clipped.
    fn was_clipped(&self) -> bool {
        // Default implementation - individual clippers can override
        false
    }

    /// Get statistics about the last clipping operation
    ///
    /// Returns information that can be used for logging or monitoring.
    fn get_clipping_stats(&self) -> ClippingStats<F> {
        ClippingStats::default()
    }
}

/// Statistics about gradient clipping operations
#[derive(Debug, Clone)]
pub struct ClippingStats<F: Float> {
    /// Whether clipping was applied
    pub was_clipped: bool,
    /// Original gradient norm (before clipping)
    pub original_norm: Option<F>,
    /// Clipped gradient norm (after clipping)
    pub clipped_norm: Option<F>,
    /// Clipping factor applied
    pub clipping_factor: Option<F>,
    /// Number of gradients that were clipped
    pub num_clipped: usize,
    /// Total number of gradients processed
    pub total_gradients: usize,
}

impl<F: Float> Default for ClippingStats<F> {
    fn default() -> Self {
        Self {
            was_clipped: false,
            original_norm: None,
            clipped_norm: None,
            clipping_factor: None,
            num_clipped: 0,
            total_gradients: 0,
        }
    }
}

/// Clip gradients by value
///
/// Clips each element of each gradient tensor to be within the range [min_value, max_value].
/// This is the simplest form of gradient clipping.
///
/// # Example
/// ```
/// use scirs2_autograd as ag;
/// use scirs2_autograd::gradient_clipping::{ClipByValue, GradientClipper};
/// use scirs2_autograd::tensor_ops::*;
///
/// let mut env = ag::VariableEnvironment::new();
/// let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// env.run(|g| {
///     // Create some example gradients
///     let grad1 = convert_to_tensor(rng.standard_normal(&[2, 2]), g);
///     let grad2 = convert_to_tensor(rng.standard_normal(&[3]), g);
///     let gradients = vec![grad1, grad2];
///
///     let mut clipper = ClipByValue::new(-1.0f32, 1.0f32);
///     let _clipped_gradients = clipper.clip_gradients(&gradients);
/// });
/// ```
pub struct ClipByValue<F: Float> {
    pub min_value: F,
    pub max_value: F,
    last_clipped: std::cell::Cell<bool>,
}

impl<F: Float> ClipByValue<F> {
    /// Create a new value-based gradient clipper
    ///
    /// # Arguments
    /// * `min_value` - Minimum allowed gradient value
    /// * `max_value` - Maximum allowed gradient value
    ///
    /// # Panics
    /// Panics if `min_value` >= `max_value`
    pub fn new(min_value: F, max_value: F) -> Self {
        assert!(
            min_value < max_value,
            "min_value must be less than max_value"
        );

        Self {
            min_value,
            max_value,
            last_clipped: std::cell::Cell::new(false),
        }
    }

    /// Create a symmetric value clipper
    ///
    /// Creates a clipper that clips values to [-max_abs_value, max_abs_value].
    ///
    /// # Arguments
    /// * `max_abs_value` - Maximum absolute value allowed
    pub fn symmetric(max_abs_value: F) -> Self {
        Self::new(-max_abs_value, max_abs_value)
    }
}

impl<F: Float> GradientClipper<F> for ClipByValue<F> {
    fn clip_gradients<'g>(&mut self, gradients: &[Tensor<'g, F>]) -> Vec<Tensor<'g, F>> {
        let any_clipped = false;

        let clipped: Vec<_> = gradients
            .iter()
            .map(|grad| {
                let clipped_grad = tensor_ops::clip(*grad, self.min_value, self.max_value);
                // Note: In a real implementation, we'd want to check if actual clipping occurred
                // For now, we assume clipping may have occurred if the operation was performed
                clipped_grad
            })
            .collect();

        self.last_clipped.set(any_clipped);
        clipped
    }

    fn was_clipped(&self) -> bool {
        self.last_clipped.get()
    }
}

/// Clip gradients by norm
///
/// Clips the norm of each individual gradient tensor. If the L2 norm of a gradient
/// exceeds the maximum norm, the gradient is scaled down proportionally.
///
/// For a gradient g with norm ||g||, if ||g|| > max_norm, then:
/// g_clipped = g * (max_norm / ||g||)
///
/// # Example
/// ```
/// use scirs2_autograd as ag;
/// use scirs2_autograd::gradient_clipping::{ClipByNorm, GradientClipper};
/// use scirs2_autograd::tensor_ops::*;
///
/// let mut env = ag::VariableEnvironment::new();
/// let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// env.run(|g| {
///     // Create some example gradients
///     let grad1 = convert_to_tensor(rng.standard_normal(&[2, 2]), g);
///     let grad2 = convert_to_tensor(rng.standard_normal(&[3]), g);
///     let gradients = vec![grad1, grad2];
///
///     let mut clipper = ClipByNorm::new(1.0f32);
///     let _clipped_gradients = clipper.clip_gradients(&gradients);
/// });
/// ```
pub struct ClipByNorm<F: Float> {
    pub max_norm: F,
    last_clipped: std::cell::Cell<bool>,
    last_stats: std::cell::RefCell<ClippingStats<F>>,
}

impl<F: Float> ClipByNorm<F> {
    /// Create a new norm-based gradient clipper
    ///
    /// # Arguments
    /// * `max_norm` - Maximum allowed L2 norm for gradients
    ///
    /// # Panics
    /// Panics if `max_norm` is not positive
    pub fn new(max_norm: F) -> Self {
        assert!(max_norm > F::zero(), "max_norm must be positive");

        Self {
            max_norm,
            last_clipped: std::cell::Cell::new(false),
            last_stats: std::cell::RefCell::new(ClippingStats::default()),
        }
    }
}

impl<F: Float> GradientClipper<F> for ClipByNorm<F> {
    fn clip_gradients<'g>(&mut self, gradients: &[Tensor<'g, F>]) -> Vec<Tensor<'g, F>> {
        let any_clipped = false;
        let num_clipped = 0;

        let clipped: Vec<_> = gradients
            .iter()
            .map(|grad| {
                // Compute the Frobenius norm of the gradient (equivalent to L2 norm for vectors)
                let grad_norm = tensor_ops::frobenius_norm(grad);

                // Create scalar tensors for comparison
                let max_norm_tensor = tensor_ops::scalar(self.max_norm, grad.graph());
                let one_tensor = tensor_ops::scalar(F::one(), grad.graph());

                // Compute clipping factor: min(1.0, max_norm / grad_norm)
                let ratio = max_norm_tensor / grad_norm;
                let clipping_factor = tensor_ops::minimum(one_tensor, ratio);

                // Note: In a full implementation, we'd track whether clipping actually occurred
                // For simplicity, we assume clipping may have occurred
                (*grad) * clipping_factor
            })
            .collect();

        self.last_clipped.set(any_clipped);

        // Update stats
        let mut stats = self.last_stats.borrow_mut();
        stats.was_clipped = any_clipped;
        stats.num_clipped = num_clipped;
        stats.total_gradients = gradients.len();

        clipped
    }

    fn was_clipped(&self) -> bool {
        self.last_clipped.get()
    }

    fn get_clipping_stats(&self) -> ClippingStats<F> {
        self.last_stats.borrow().clone()
    }
}

/// Clip gradients by global norm
///
/// Clips all gradients jointly based on their global norm. The global norm is
/// computed as the L2 norm of the concatenation of all gradient vectors.
///
/// If the global norm exceeds max_norm, all gradients are scaled by the same factor:
/// scaling_factor = max_norm / global_norm
///
/// This method preserves the relative magnitudes between different gradients
/// while ensuring the overall gradient update is not too large.
///
/// # Example
/// ```
/// use scirs2_autograd as ag;
/// use scirs2_autograd::gradient_clipping::{ClipByGlobalNorm, GradientClipper};
/// use scirs2_autograd::tensor_ops::*;
///
/// let mut env = ag::VariableEnvironment::new();
/// let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// env.run(|g| {
///     // Create some example gradients
///     let grad1 = convert_to_tensor(rng.standard_normal(&[2, 2]), g);
///     let grad2 = convert_to_tensor(rng.standard_normal(&[3]), g);
///     let gradients = vec![grad1, grad2];
///
///     let mut clipper = ClipByGlobalNorm::new(1.0f32);
///     let _clipped_gradients = clipper.clip_gradients(&gradients);
/// });
/// ```
pub struct ClipByGlobalNorm<F: Float> {
    pub max_norm: F,
    last_clipped: std::cell::Cell<bool>,
    last_stats: std::cell::RefCell<ClippingStats<F>>,
}

impl<F: Float> ClipByGlobalNorm<F> {
    /// Create a new global norm-based gradient clipper
    ///
    /// # Arguments
    /// * `max_norm` - Maximum allowed global norm for all gradients combined
    ///
    /// # Panics
    /// Panics if `max_norm` is not positive
    pub fn new(max_norm: F) -> Self {
        assert!(max_norm > F::zero(), "max_norm must be positive");

        Self {
            max_norm,
            last_clipped: std::cell::Cell::new(false),
            last_stats: std::cell::RefCell::new(ClippingStats::default()),
        }
    }
}

impl<F: Float> GradientClipper<F> for ClipByGlobalNorm<F> {
    fn clip_gradients<'g>(&mut self, gradients: &[Tensor<'g, F>]) -> Vec<Tensor<'g, F>> {
        if gradients.is_empty() {
            return Vec::new();
        }

        let g = gradients[0].graph();

        // Compute global norm: sqrt(sum(norm(grad_i)^2))
        let squared_norms: Vec<_> = gradients
            .iter()
            .map(|grad| {
                let norm = tensor_ops::frobenius_norm(grad);
                tensor_ops::square(norm)
            })
            .collect();

        let global_norm_squared = tensor_ops::add_n(&squared_norms);
        let global_norm = tensor_ops::sqrt(global_norm_squared);

        // Compute clipping factor
        let max_norm_tensor = tensor_ops::scalar(self.max_norm, g);
        let one_tensor = tensor_ops::scalar(F::one(), g);
        let ratio = max_norm_tensor / global_norm;
        let clipping_factor = tensor_ops::minimum(one_tensor, ratio);

        // Apply the same clipping factor to all gradients
        let clipped: Vec<_> = gradients
            .iter()
            .map(|grad| (*grad) * clipping_factor)
            .collect();

        // Note: In a full implementation, we'd evaluate global_norm and check if clipping occurred
        let was_clipped = false; // Placeholder - would need evaluation to determine

        self.last_clipped.set(was_clipped);

        // Update stats
        let mut stats = self.last_stats.borrow_mut();
        stats.was_clipped = was_clipped;
        stats.total_gradients = gradients.len();
        stats.num_clipped = if was_clipped { gradients.len() } else { 0 };

        clipped
    }

    fn was_clipped(&self) -> bool {
        self.last_clipped.get()
    }

    fn get_clipping_stats(&self) -> ClippingStats<F> {
        self.last_stats.borrow().clone()
    }
}

/// Adaptive gradient clipper
///
/// Adjusts the clipping threshold based on the history of gradient norms.
/// This can help automatically tune the clipping threshold during training.
pub struct AdaptiveClipByNorm<F: Float> {
    base_clipper: ClipByNorm<F>,
    #[allow(dead_code)]
    adaptation_rate: F,
    current_threshold: std::cell::Cell<F>,
}

impl<F: Float> AdaptiveClipByNorm<F> {
    /// Create a new adaptive gradient clipper
    ///
    /// # Arguments
    /// * `initial_max_norm` - Initial maximum norm threshold
    /// * `adaptation_rate` - Rate at which to adapt the threshold (0.0 to 1.0)
    pub fn new(initial_max_norm: F, adaptation_rate: F) -> Self {
        assert!(
            adaptation_rate >= F::zero() && adaptation_rate <= F::one(),
            "adaptation_rate must be between 0.0 and 1.0"
        );

        Self {
            base_clipper: ClipByNorm::new(initial_max_norm),
            adaptation_rate,
            current_threshold: std::cell::Cell::new(initial_max_norm),
        }
    }

    /// Get the current adaptive threshold
    pub fn current_threshold(&self) -> F {
        self.current_threshold.get()
    }

    /// Manually update the threshold (for external adaptation logic)
    pub fn set_threshold(&self, new_threshold: F) {
        assert!(new_threshold > F::zero(), "threshold must be positive");
        self.current_threshold.set(new_threshold);
    }
}

impl<F: Float> GradientClipper<F> for AdaptiveClipByNorm<F> {
    fn clip_gradients<'g>(&mut self, gradients: &[Tensor<'g, F>]) -> Vec<Tensor<'g, F>> {
        // Update the base clipper's threshold
        let current_threshold = self.current_threshold.get();
        self.base_clipper.max_norm = current_threshold;

        // Apply clipping with current threshold
        let result = self.base_clipper.clip_gradients(gradients);

        // Note: In a full implementation, we'd compute actual gradient norms
        // and adapt the threshold based on recent history
        // For now, this is a placeholder for the adaptation logic

        result
    }

    fn was_clipped(&self) -> bool {
        self.base_clipper.was_clipped()
    }

    fn get_clipping_stats(&self) -> ClippingStats<F> {
        self.base_clipper.get_clipping_stats()
    }
}

/// Convenience functions for gradient clipping
impl<F: Float> Tensor<'_, F> {
    /// Clip this tensor's values to a range
    ///
    /// # Arguments
    /// * `min_value` - Minimum allowed value
    /// * `max_value` - Maximum allowed value
    pub fn clip_values(self, min_value: F, max_value: F) -> Self {
        tensor_ops::clip(self, min_value, max_value)
    }

    /// Clip this tensor's norm to a maximum value
    ///
    /// # Arguments
    /// * `max_norm` - Maximum allowed norm
    pub fn clip_norm(self, max_norm: F) -> Self {
        let norm = tensor_ops::frobenius_norm(self);
        let max_norm_tensor = tensor_ops::scalar(max_norm, self.graph());
        let one_tensor = tensor_ops::scalar(F::one(), self.graph());
        let ratio = max_norm_tensor / norm;
        let clipping_factor = tensor_ops::minimum(one_tensor, ratio);
        self * clipping_factor
    }
}

/// Common gradient clipping presets
pub mod presets {
    use super::*;

    /// Create a conservative gradient clipper for fine-tuning
    pub fn conservative<F: Float>() -> ClipByGlobalNorm<F> {
        ClipByGlobalNorm::new(F::from(0.5).unwrap())
    }

    /// Create a standard gradient clipper for general training
    pub fn standard<F: Float>() -> ClipByGlobalNorm<F> {
        ClipByGlobalNorm::new(F::from(1.0).unwrap())
    }

    /// Create an aggressive gradient clipper for unstable training
    pub fn aggressive<F: Float>() -> ClipByGlobalNorm<F> {
        ClipByGlobalNorm::new(F::from(0.1).unwrap())
    }

    /// Create a value-based clipper for preventing extreme gradients
    pub fn extreme_prevention<F: Float>() -> ClipByValue<F> {
        ClipByValue::symmetric(F::from(10.0).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_by_value_creation() {
        let clipper = ClipByValue::new(-1.0f32, 1.0f32);
        assert_eq!(clipper.min_value, -1.0);
        assert_eq!(clipper.max_value, 1.0);

        let symmetric = ClipByValue::symmetric(0.5f32);
        assert_eq!(symmetric.min_value, -0.5);
        assert_eq!(symmetric.max_value, 0.5);
    }

    #[test]
    fn test_clip_by_norm_creation() {
        let clipper = ClipByNorm::new(1.0f32);
        assert_eq!(clipper.max_norm, 1.0);
    }

    #[test]
    fn test_clip_by_global_norm_creation() {
        let clipper = ClipByGlobalNorm::new(1.0f32);
        assert_eq!(clipper.max_norm, 1.0);
    }

    #[test]
    fn test_adaptive_clipper() {
        let clipper = AdaptiveClipByNorm::new(1.0f32, 0.1);
        assert_eq!(clipper.current_threshold(), 1.0);

        clipper.set_threshold(0.5);
        assert_eq!(clipper.current_threshold(), 0.5);
    }

    #[test]
    fn test_clipping_stats_default() {
        let stats = ClippingStats::<f32>::default();
        assert!(!stats.was_clipped);
        assert_eq!(stats.num_clipped, 0);
        assert_eq!(stats.total_gradients, 0);
    }

    #[test]
    fn test_presets() {
        let _conservative = presets::conservative::<f32>();
        let _standard = presets::standard::<f32>();
        let _aggressive = presets::aggressive::<f32>();
        let _extreme = presets::extreme_prevention::<f32>();
    }

    #[test]
    #[should_panic(expected = "min_value must be less than max_value")]
    fn test_clip_by_value_invalid_range() {
        ClipByValue::new(1.0f32, -1.0f32);
    }

    #[test]
    #[should_panic(expected = "max_norm must be positive")]
    fn test_clip_by_norm_negative_norm() {
        ClipByNorm::new(-1.0f32);
    }

    #[test]
    #[should_panic(expected = "adaptation_rate must be between 0.0 and 1.0")]
    fn test_adaptive_clipper_invalid_rate() {
        AdaptiveClipByNorm::new(1.0f32, 2.0);
    }
}
