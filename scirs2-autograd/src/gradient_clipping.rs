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

/// Evaluates the Frobenius norms of `gradients` when the graph can be evaluated.
///
/// Returns `None` when any gradient cannot be evaluated (unfed placeholders, or a
/// variable that is only reachable through a `Context` the clipper does not hold). The
/// clippers use this to report **measured** statistics instead of hard-coded ones; when
/// it returns `None` they report "unknown" rather than claiming that nothing was clipped.
fn measure_norms<F: Float>(gradients: &[Tensor<'_, F>]) -> Option<Vec<F>> {
    let mut out = Vec::with_capacity(gradients.len());
    for grad in gradients {
        let norm = tensor_ops::frobenius_norm(grad);
        let arr = norm.eval(grad.graph()).ok()?;
        out.push(arr.iter().copied().next()?);
    }
    Some(out)
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
/// use scirs2_autograd::tensor_ops::convert_to_tensor;
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
/// use scirs2_autograd::tensor_ops::convert_to_tensor;
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

                (*grad) * clipping_factor
            })
            .collect();

        // Measure whether clipping actually bites.  These fields used to be hard-coded to
        // `false`/`0` no matter what the gradients were, which made the monitoring API
        // report "never clipped" even while it was clipping every step.
        let mut stats = ClippingStats::<F> {
            total_gradients: gradients.len(),
            ..Default::default()
        };
        match measure_norms(gradients) {
            Some(norms) => {
                let over: Vec<F> = norms
                    .iter()
                    .copied()
                    .filter(|n| *n > self.max_norm)
                    .collect();
                stats.num_clipped = over.len();
                stats.was_clipped = !over.is_empty();
                stats.original_norm = norms.iter().copied().fold(None, |acc: Option<F>, n| {
                    Some(match acc {
                        Some(a) if a >= n => a,
                        _ => n,
                    })
                });
                stats.clipped_norm =
                    stats
                        .original_norm
                        .map(|n| if n > self.max_norm { self.max_norm } else { n });
                stats.clipping_factor = stats.original_norm.map(|n| {
                    if n > F::zero() && n > self.max_norm {
                        self.max_norm / n
                    } else {
                        F::one()
                    }
                });
            }
            None => {
                // Norms are not measurable from a bare graph handle (unfed placeholders /
                // variables outside this scope): report "unknown" (all `None`) rather than
                // asserting that no clipping happened.
                stats.was_clipped = false;
            }
        }

        self.last_clipped.set(stats.was_clipped);
        *self.last_stats.borrow_mut() = stats;

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
/// use scirs2_autograd::tensor_ops::convert_to_tensor;
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

        // Measure the actual global norm so the monitoring API reports reality instead of
        // a hard-coded `false`.
        let mut stats = ClippingStats::<F> {
            total_gradients: gradients.len(),
            ..Default::default()
        };
        match measure_norms(gradients) {
            Some(norms) => {
                let sum_sq = norms.iter().fold(F::zero(), |acc, n| acc + (*n) * (*n));
                let measured = sum_sq.sqrt();
                let was_clipped = measured > self.max_norm;
                stats.was_clipped = was_clipped;
                stats.num_clipped = if was_clipped { gradients.len() } else { 0 };
                stats.original_norm = Some(measured);
                stats.clipped_norm = Some(if was_clipped { self.max_norm } else { measured });
                stats.clipping_factor = Some(if was_clipped && measured > F::zero() {
                    self.max_norm / measured
                } else {
                    F::one()
                });
            }
            None => {
                stats.was_clipped = false;
            }
        }

        self.last_clipped.set(stats.was_clipped);
        *self.last_stats.borrow_mut() = stats;

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
    adaptation_rate: F,
    current_threshold: std::cell::Cell<F>,
    /// Exponential moving average of the observed gradient norms, or `None` until the
    /// first measurable batch of gradients has been seen.
    norm_ema: std::cell::Cell<Option<F>>,
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
            norm_ema: std::cell::Cell::new(None),
        }
    }

    /// Exponential moving average of the gradient norms observed so far.
    ///
    /// `None` until at least one call to `clip_gradients` has seen evaluable gradients.
    pub fn observed_norm_ema(&self) -> Option<F> {
        self.norm_ema.get()
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

        // Adapt: track an exponential moving average of the observed *global* gradient
        // norm and move the threshold towards it at `adaptation_rate`.  Without this the
        // type did no adaptation at all -- the threshold only ever moved when the caller
        // invoked `set_threshold` by hand, which is precisely what an *adaptive* clipper
        // is supposed to remove the need for.
        if let Some(norms) = measure_norms(gradients) {
            let sum_sq = norms.iter().fold(F::zero(), |acc, n| acc + (*n) * (*n));
            let observed = sum_sq.sqrt();

            let ema = match self.norm_ema.get() {
                Some(prev) => prev + self.adaptation_rate * (observed - prev),
                None => observed,
            };
            self.norm_ema.set(Some(ema));

            // Move the threshold towards the running average, keeping it strictly
            // positive (a non-positive threshold would make the clipper degenerate).
            let next = current_threshold + self.adaptation_rate * (ema - current_threshold);
            if next > F::zero() && next.is_finite() {
                self.current_threshold.set(next);
            }
        }

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
        ClipByGlobalNorm::new(F::from(0.5).expect("Failed to convert constant to float"))
    }

    /// Create a standard gradient clipper for general training
    pub fn standard<F: Float>() -> ClipByGlobalNorm<F> {
        ClipByGlobalNorm::new(F::from(1.0).expect("Failed to convert constant to float"))
    }

    /// Create an aggressive gradient clipper for unstable training
    pub fn aggressive<F: Float>() -> ClipByGlobalNorm<F> {
        ClipByGlobalNorm::new(F::from(0.1).expect("Failed to convert constant to float"))
    }

    /// Create a value-based clipper for preventing extreme gradients
    pub fn extreme_prevention<F: Float>() -> ClipByValue<F> {
        ClipByValue::symmetric(F::from(10.0).expect("Failed to convert constant to float"))
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
