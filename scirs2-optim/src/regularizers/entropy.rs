use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::Regularizer;

/// Entropy regularization
///
/// Entropy regularization encourages a model to produce more confident outputs (lower entropy),
/// or more uncertain outputs (higher entropy), depending on the settings. This is often used
/// in reinforcement learning, semi-supervised learning, and some classifier applications.
///
/// # Types
///
/// * `MaximizeEntropy`: Encourages high entropy (uniform, uncertain predictions)
/// * `MinimizeEntropy`: Encourages low entropy (confident, peaked predictions)
///
/// # Parameters
///
/// * `lambda`: Regularization strength coefficient, controls the amount of regularization applied
/// * `epsilon`: Small value for numerical stability (prevents log(0))
///
#[derive(Debug, Clone, Copy)]
pub enum EntropyRegularizerType {
    /// Maximize entropy (encourages uniform distributions)
    MaximizeEntropy,
    /// Minimize entropy (encourages confident predictions)
    MinimizeEntropy,
}

/// Entropy regularization for probability distributions
///
/// This regularizer can either encourage high entropy (more uniform distributions) or
/// low entropy (more peaked distributions) depending on the selected regularizer type.
///
/// It's commonly used in reinforcement learning algorithms, semi-supervised learning,
/// and some classification tasks where controlling the certainty of outputs is desired.
#[derive(Debug, Clone, Copy)]
pub struct EntropyRegularization<A: Float + FromPrimitive + Debug> {
    /// Regularization strength
    pub lambda: A,
    /// Small value for numerical stability
    pub epsilon: A,
    /// Type of entropy regularization
    pub reg_type: EntropyRegularizerType,
}

impl<A: Float + FromPrimitive + Debug> EntropyRegularization<A> {
    /// Create a new entropy regularization
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength coefficient
    /// * `reg_type` - Type of entropy regularization (maximize or minimize)
    ///
    /// # Returns
    ///
    /// An entropy regularization with default epsilon
    pub fn new(lambda: A, reg_type: EntropyRegularizerType) -> Self {
        let epsilon = A::from_f64(1e-8).unwrap();
        Self {
            lambda,
            epsilon,
            reg_type,
        }
    }

    /// Create a new entropy regularization with custom epsilon
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength coefficient
    /// * `epsilon` - Small value for numerical stability
    /// * `reg_type` - Type of entropy regularization (maximize or minimize)
    ///
    /// # Returns
    ///
    /// An entropy regularization with custom epsilon
    pub fn new_with_epsilon(lambda: A, epsilon: A, reg_type: EntropyRegularizerType) -> Self {
        Self {
            lambda,
            epsilon,
            reg_type,
        }
    }

    /// Calculate the entropy of a probability distribution
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability distribution (should sum to 1 along the appropriate axis)
    ///
    /// # Returns
    ///
    /// The entropy value
    fn calculate_entropy<S, D>(&self, probs: &ArrayBase<S, D>) -> A
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        // Clip probabilities to avoid log(0)
        let safe_probs = probs.mapv(|p| {
            if p < self.epsilon {
                self.epsilon
            } else if p > (A::one() - self.epsilon) {
                A::one() - self.epsilon
            } else {
                p
            }
        });

        // Calculate entropy: -sum(p * log(p))
        let neg_entropy = safe_probs.mapv(|p| p * p.ln()).sum();
        -neg_entropy
    }

    /// Calculate gradient of entropy with respect to input probabilities
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability distribution
    ///
    /// # Returns
    ///
    /// The gradient of entropy with respect to probabilities
    fn entropy_gradient<S, D>(&self, probs: &ArrayBase<S, D>) -> Array<A, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        // Clip probabilities to avoid log(0)
        let safe_probs = probs.mapv(|p| {
            if p < self.epsilon {
                self.epsilon
            } else if p > (A::one() - self.epsilon) {
                A::one() - self.epsilon
            } else {
                p
            }
        });

        // Gradient of entropy: -(1 + log(p))
        let gradient = safe_probs.mapv(|p| -(A::one() + p.ln()));

        // For minimizing entropy, we negate the gradient
        match self.reg_type {
            EntropyRegularizerType::MaximizeEntropy => gradient,
            EntropyRegularizerType::MinimizeEntropy => gradient.mapv(|g| -g),
        }
    }
}

impl<A, D> Regularizer<A, D> for EntropyRegularization<A>
where
    A: Float + ScalarOperand + Debug + FromPrimitive,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Calculate entropy penalty
        let entropy = self.calculate_entropy(params);

        // Calculate entropy gradients
        let entropy_grads = self.entropy_gradient(params);

        // Scale gradients by lambda and add to input gradients
        gradients.zip_mut_with(&entropy_grads, |g, &e| *g = *g + self.lambda * e);

        // Return the regularization term to be added to the loss:
        // For maximizing entropy, we return -lambda * entropy (to minimize -entropy)
        // For minimizing entropy, we return lambda * entropy (to minimize entropy)
        let penalty = match self.reg_type {
            EntropyRegularizerType::MaximizeEntropy => -self.lambda * entropy,
            EntropyRegularizerType::MinimizeEntropy => self.lambda * entropy,
        };

        Ok(penalty)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // Calculate entropy penalty
        let entropy = self.calculate_entropy(params);

        // For maximizing entropy, we return -lambda * entropy (to minimize -entropy)
        // For minimizing entropy, we return lambda * entropy (to minimize entropy)
        let penalty = match self.reg_type {
            EntropyRegularizerType::MaximizeEntropy => -self.lambda * entropy,
            EntropyRegularizerType::MinimizeEntropy => self.lambda * entropy,
        };

        Ok(penalty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_entropy_regularization_creation() {
        let er = EntropyRegularization::new(0.1f64, EntropyRegularizerType::MaximizeEntropy);
        assert_eq!(er.lambda, 0.1);
        assert_eq!(er.epsilon, 1e-8);
        match er.reg_type {
            EntropyRegularizerType::MaximizeEntropy => (),
            _ => panic!("Wrong regularizer type"),
        }

        let er = EntropyRegularization::new_with_epsilon(
            0.2f64,
            1e-10,
            EntropyRegularizerType::MinimizeEntropy,
        );
        assert_eq!(er.lambda, 0.2);
        assert_eq!(er.epsilon, 1e-10);
        match er.reg_type {
            EntropyRegularizerType::MinimizeEntropy => (),
            _ => panic!("Wrong regularizer type"),
        }
    }

    #[test]
    fn test_calculate_entropy() {
        // Uniform distribution (maximum entropy)
        let uniform = Array1::from_vec(vec![0.25f64, 0.25, 0.25, 0.25]);
        let er = EntropyRegularization::new(1.0f64, EntropyRegularizerType::MaximizeEntropy);
        let entropy = er.calculate_entropy(&uniform);

        // Entropy of uniform distribution should be ln(n)
        let expected = (4.0f64).ln();
        assert_abs_diff_eq!(entropy, expected, epsilon = 1e-6);

        // Peaked distribution (low entropy)
        let peaked = Array1::from_vec(vec![0.01f64, 0.01, 0.97, 0.01]);
        let entropy = er.calculate_entropy(&peaked);
        assert!(entropy < expected); // Should be less than uniform entropy
    }

    #[test]
    fn test_entropy_gradient() {
        let er = EntropyRegularization::new(1.0f64, EntropyRegularizerType::MaximizeEntropy);

        // For uniform distribution, gradients should be approximately equal
        let uniform = Array1::from_vec(vec![0.25f64, 0.25, 0.25, 0.25]);
        let grads = er.entropy_gradient(&uniform);

        // Expected gradient: -(1 + ln(0.25))
        let expected = -(1.0 + 0.25f64.ln());
        for &g in grads.iter() {
            assert_abs_diff_eq!(g, expected, epsilon = 1e-6);
        }

        // For peaked distribution, gradients should be different for different probabilities
        let peaked = Array1::from_vec(vec![0.1f64, 0.1, 0.7, 0.1]);
        let grads = er.entropy_gradient(&peaked);

        // The gradient for larger probability should have a smaller absolute value
        // because ln(0.7) is greater (less negative) than ln(0.1)
        // So -(1 + ln(0.7)) has smaller magnitude than -(1 + ln(0.1))
        assert!(grads[2].abs() < grads[0].abs());
    }

    #[test]
    fn test_maximize_entropy_penalty() {
        // For maximizing entropy, we want to minimize -entropy
        let er = EntropyRegularization::new(1.0f64, EntropyRegularizerType::MaximizeEntropy);

        // Uniform distribution (high entropy)
        let uniform = Array1::from_vec(vec![0.25f64, 0.25, 0.25, 0.25]);
        let penalty = er.penalty(&uniform).unwrap();

        // Peaked distribution (low entropy)
        let peaked = Array1::from_vec(vec![0.01f64, 0.01, 0.97, 0.01]);
        let peaked_penalty = er.penalty(&peaked).unwrap();

        // The penalty for peaked should be greater than for uniform
        // because we're trying to maximize entropy
        assert!(peaked_penalty > penalty);
    }

    #[test]
    fn test_minimize_entropy_penalty() {
        // For minimizing entropy, we want to minimize entropy
        let er = EntropyRegularization::new(1.0f64, EntropyRegularizerType::MinimizeEntropy);

        // Uniform distribution (high entropy)
        let uniform = Array1::from_vec(vec![0.25f64, 0.25, 0.25, 0.25]);
        let penalty = er.penalty(&uniform).unwrap();

        // Peaked distribution (low entropy)
        let peaked = Array1::from_vec(vec![0.01f64, 0.01, 0.97, 0.01]);
        let peaked_penalty = er.penalty(&peaked).unwrap();

        // The penalty for uniform should be greater than for peaked
        // because we're trying to minimize entropy
        assert!(penalty > peaked_penalty);
    }

    #[test]
    fn test_apply_gradients() {
        let lambda = 0.5f64;
        let er = EntropyRegularization::new(lambda, EntropyRegularizerType::MaximizeEntropy);

        let probs = Array1::from_vec(vec![0.25f64, 0.25, 0.25, 0.25]);
        let mut gradients = Array1::zeros(4);

        let penalty = er.apply(&probs, &mut gradients).unwrap();

        // Check that gradients have been modified
        assert!(gradients.iter().all(|&g| g != 0.0));

        // For uniform distribution, all gradients should be equal
        let first = gradients[0];
        assert!(gradients.iter().all(|&g| (g - first).abs() < 1e-6));

        // Expected gradient: -lambda * (1 + ln(0.25))
        let expected_grad = -lambda * (1.0 + 0.25f64.ln());
        assert_abs_diff_eq!(gradients[0], expected_grad, epsilon = 1e-6);

        // Check penalty matches expected value
        let entropy = (4.0f64).ln(); // Entropy of uniform distribution
        let expected_penalty = -lambda * entropy; // For maximizing entropy
        assert_abs_diff_eq!(penalty, expected_penalty, epsilon = 1e-6);
    }

    #[test]
    fn test_regularizer_trait() {
        // Test that EntropyRegularization implements Regularizer trait correctly
        let er = EntropyRegularization::new(0.1f64, EntropyRegularizerType::MaximizeEntropy);

        let probs = Array1::from_vec(vec![0.25f64, 0.25, 0.25, 0.25]);
        let mut gradients = Array1::zeros(4);

        // Both methods should return the same penalty for the same input
        let penalty1 = er.apply(&probs, &mut gradients).unwrap();
        let penalty2 = er.penalty(&probs).unwrap();

        assert_abs_diff_eq!(penalty1, penalty2, epsilon = 1e-10);
    }
}
