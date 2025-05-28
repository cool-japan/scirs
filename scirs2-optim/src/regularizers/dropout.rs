//! Dropout regularization

use ndarray::{Array, Dimension, ScalarOperand, Zip};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::Regularizer;

/// Dropout regularization
///
/// Randomly sets a fraction of the input units to 0 at each update during training,
/// which helps prevent overfitting. During inference, all units are used with appropriate
/// scaling to maintain the same expected output.
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::regularizers::Dropout;
/// use rand::SeedableRng;
/// use rand::rngs::SmallRng;
///
/// // Create a dropout regularizer with 0.5 dropout rate
/// let seed = 42;
/// let mut rng = SmallRng::seed_from_u64(seed);
/// let mut dropout = Dropout::new(0.5f64, &mut rng);
///
/// // Set to training mode
/// dropout.train();
///
/// // Check the dropout rate
/// assert_eq!(dropout.rate(), 0.5);
///
/// // Set to evaluation mode
/// dropout.eval();
/// assert!(!dropout.is_training());
/// ```
#[derive(Debug, Clone)]
pub struct Dropout<A: Float + Debug> {
    /// Dropout rate (fraction of units that are dropped)
    rate: A,
    /// Random number generator
    rng: SmallRng,
    /// Boolean indicating whether in training mode
    training: bool,
    /// Cached dropout mask
    mask: Option<Array<A, ndarray::IxDyn>>,
}

impl<A: Float + Debug> Dropout<A> {
    /// Create a new dropout regularizer
    ///
    /// # Arguments
    ///
    /// * `rate` - Dropout rate (0.0 to 1.0, fraction of units that are dropped)
    /// * `rng` - Random number generator
    pub fn new<R: Rng>(rate: A, rng: &mut R) -> Self {
        // Ensure rate is between 0 and 1
        let rate = rate.max(A::zero()).min(A::one());

        // Create a new RNG from the provided one
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let seed = u64::from_ne_bytes(seed_bytes);
        let rng = SmallRng::seed_from_u64(seed);

        Self {
            rate,
            rng,
            training: true,
            mask: None,
        }
    }

    /// Get the dropout rate
    pub fn rate(&self) -> A {
        self.rate
    }

    /// Set the dropout rate
    ///
    /// # Arguments
    ///
    /// * `rate` - Dropout rate (0.0 to 1.0, fraction of units that are dropped)
    pub fn set_rate(&mut self, rate: A) -> &mut Self {
        // Ensure rate is between 0 and 1
        self.rate = rate.max(A::zero()).min(A::one());
        // Clear the mask cache
        self.mask = None;
        self
    }

    /// Set to training mode (apply dropout)
    pub fn train(&mut self) -> &mut Self {
        self.training = true;
        self
    }

    /// Set to inference mode (no dropout, scale outputs)
    pub fn eval(&mut self) -> &mut Self {
        self.training = false;
        self
    }

    /// Get the training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Create a new dropout mask for the given shape
    ///
    /// During training, randomly sets units to 0 with probability `rate`,
    /// and scales the remaining by 1/(1-rate) to maintain the same expected output.
    fn create_mask<D: Dimension>(&mut self, shape: D) -> Array<A, D> {
        if !self.training || self.rate <= A::zero() {
            // In eval mode or with 0 dropout rate, no masking is applied
            return Array::ones(shape);
        }

        // The scale factor for the kept units is 1/(1-rate)
        // This maintains the expected sum of the layer outputs
        let keep_prob = A::one() - self.rate;
        let scale = A::one() / keep_prob;

        // Create a mask where units are kept with probability (1-rate)
        // and scaled by 1/(1-rate)
        let mut mask = Array::zeros(shape);
        for elem in mask.iter_mut() {
            let rand_val = A::from(self.rng.random_range(0.0..1.0)).unwrap();
            if rand_val > self.rate {
                *elem = scale;
            }
        }

        mask
    }
}

impl<A, D> Regularizer<A, D> for Dropout<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension<Pattern = D>,
{
    fn apply(&self, _params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        if !self.training || self.rate <= A::zero() {
            // In eval mode or with 0 dropout rate, no dropout is applied
            return Ok(A::zero());
        }

        // Create or get the dropout mask
        let mask = match &self.mask {
            Some(m) if m.shape() == gradients.shape() => {
                // Use cached mask if shapes match
                m.clone().into_dimensionality::<D>().unwrap()
            }
            _ => {
                // Create a new mask
                let mut dropout = self.clone();
                // We would cache the mask here in a mutable context
                dropout.create_mask(gradients.dim())
            }
        };

        // Apply the mask to the gradients
        Zip::from(gradients).and(&mask).for_each(|grad, &mask_val| {
            *grad = *grad * mask_val;
        });

        // Dropout doesn't add a penalty term to the loss
        Ok(A::zero())
    }

    fn penalty(&self, _params: &Array<A, D>) -> Result<A> {
        // Dropout doesn't add a penalty term to the loss
        Ok(A::zero())
    }
}
