//! Information theory functions
//!
//! This module provides functions related to information theory, including
//! entropy, Kullback-Leibler divergence, and Huber loss functions.
//!
//! # Mathematical Background
//!
//! ## Shannon Entropy
//!
//! The Shannon entropy H(X) of a discrete random variable X with probability mass
//! function P(X = xᵢ) = pᵢ is defined as:
//!
//! ```text
//! H(X) = -∑ᵢ pᵢ log₂(pᵢ)
//! ```
//!
//! The base of the logarithm determines the unit of measurement:
//! - Base 2: bits (binary digits)
//! - Base e: nats (natural units)
//! - Base 10: dits (decimal digits)
//!
//! ### Properties of Shannon Entropy
//!
//! 1. **Non-negativity**: H(X) ≥ 0, with equality if and only if X is deterministic
//! 2. **Maximum entropy**: H(X) ≤ log n for n possible outcomes, achieved by uniform distribution
//! 3. **Concavity**: H is a concave function of the probability distribution
//! 4. **Continuity**: H is continuous in the probabilities
//!
//! ### Differential Entropy
//!
//! For continuous random variables with probability density function f(x):
//!
//! ```text
//! h(X) = -∫ f(x) log f(x) dx
//! ```
//!
//! Note: Unlike discrete entropy, differential entropy can be negative.
//!
//! ## Kullback-Leibler Divergence
//!
//! The Kullback-Leibler (KL) divergence D_KL(P||Q) measures the difference between
//! two probability distributions P and Q:
//!
//! ```text
//! D_KL(P||Q) = ∑ᵢ P(i) log(P(i)/Q(i))
//! ```
//!
//! For continuous distributions:
//!
//! ```text
//! D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
//! ```
//!
//! ### Properties of KL Divergence
//!
//! 1. **Non-negativity**: D_KL(P||Q) ≥ 0 (Gibbs' inequality)
//! 2. **Zero if and only if identical**: D_KL(P||Q) = 0 ⟺ P = Q almost everywhere
//! 3. **Asymmetry**: D_KL(P||Q) ≠ D_KL(Q||P) in general
//! 4. **Convexity**: D_KL(·||Q) is convex in the first argument
//!
//! ### Mathematical Proof of Non-negativity (Gibbs' Inequality)
//!
//! **Theorem**: For probability distributions P and Q, D_KL(P||Q) ≥ 0.
//!
//! **Proof**: By Jensen's inequality, since log is concave:
//!
//! ```text
//! -D_KL(P||Q) = ∑ᵢ P(i) log(Q(i)/P(i))
//!               ≤ log(∑ᵢ P(i) · Q(i)/P(i))  [Jensen's inequality]
//!               = log(∑ᵢ Q(i))
//!               = log(1) = 0
//! ```
//!
//! Therefore, D_KL(P||Q) ≥ 0, with equality if and only if P(i) = Q(i) for all i.
//!
//! ## Cross-Entropy
//!
//! The cross-entropy H(P,Q) between distributions P and Q is:
//!
//! ```text
//! H(P,Q) = -∑ᵢ P(i) log Q(i) = H(P) + D_KL(P||Q)
//! ```
//!
//! This decomposition shows that cross-entropy equals the entropy of P plus
//! the additional "cost" of using Q instead of P.
//!
//! ## Mutual Information
//!
//! The mutual information I(X;Y) between random variables X and Y quantifies
//! the amount of information obtained about one variable through the other:
//!
//! ```text
//! I(X;Y) = D_KL(P(X,Y)||P(X)⊗P(Y)) = ∑ᵢⱼ P(x,y) log(P(x,y)/(P(x)P(y)))
//! ```
//!
//! ### Properties of Mutual Information
//!
//! 1. **Symmetry**: I(X;Y) = I(Y;X)
//! 2. **Non-negativity**: I(X;Y) ≥ 0
//! 3. **Bounds**: 0 ≤ I(X;Y) ≤ min(H(X), H(Y))
//! 4. **Chain rule**: I(X;Y,Z) = I(X;Y) + I(X;Z|Y)
//!
//! ## Information-Theoretic Inequalities
//!
//! ### Fano's Inequality
//!
//! For a Markov chain X → Y → X̂ where X̂ is an estimate of X:
//!
//! ```text
//! H(P_e) + P_e log(|𝒳| - 1) ≥ H(X|X̂)
//! ```
//!
//! where P_e = Pr(X ≠ X̂) is the error probability.
//!
//! ### Data Processing Inequality
//!
//! For a Markov chain X → Y → Z:
//!
//! ```text
//! I(X;Z) ≤ I(X;Y) and I(X;Z) ≤ I(Y;Z)
//! ```
//!
//! This states that processing cannot increase information.
//!
//! ## Applications
//!
//! ### Machine Learning
//! - **Loss functions**: Cross-entropy loss for classification
//! - **Feature selection**: Mutual information for feature ranking
//! - **Model selection**: Information criteria (AIC, BIC)
//!
//! ### Statistics
//! - **Hypothesis testing**: Likelihood ratio tests
//! - **Estimation**: Maximum likelihood estimation
//! - **Model comparison**: Information criteria
//!
//! ### Physics
//! - **Statistical mechanics**: Connection to thermodynamic entropy
//! - **Quantum information**: Von Neumann entropy
//! - **Black hole physics**: Bekenstein-Hawking entropy
//!
//! ## Computational Considerations
//!
//! ### Numerical Stability
//!
//! The computation of x log(x) for small x requires careful handling:
//! - For x = 0: Define 0 log(0) = 0 (by continuity)
//! - For small x: Use series expansion x log(x) ≈ x(log(x₀) + (x-x₀)/x₀) near x₀
//!
//! ### Algorithmic Complexity
//!
//! - **Entropy computation**: O(n) for n probability values
//! - **KL divergence**: O(n) for discrete distributions
//! - **Mutual information estimation**: O(n log n) with k-NN methods

use crate::error::{SpecialError, SpecialResult};
use crate::validation::{check_finite, check_non_negative};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayViewMut1};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};

/// Shannon entropy function
///
/// Computes -x * log(x) for x > 0, and 0 for x = 0.
/// This is the entropy contribution of a single probability.
///
/// # Arguments
/// * `x` - Input value (typically a probability)
///
/// # Returns
/// The entropy value -x * log(x)
///
/// # Examples
/// ```
/// use scirs2_special::information_theory::entr;
///
/// let h = entr(0.5);
/// assert!((h - 0.34657359027997264f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn entr<T>(x: T) -> T
where
    T: Float + FromPrimitive + Zero,
{
    if x.is_zero() {
        T::zero()
    } else if x < T::zero() {
        -T::infinity()
    } else {
        -x * x.ln()
    }
}

/// Relative entropy (Kullback-Leibler divergence term)
///
/// Computes x * log(x/y) for x > 0, y > 0.
/// Special cases:
/// - If x = 0, returns 0
/// - If y = 0 and x > 0, returns infinity
///
/// # Arguments
/// * `x` - First probability
/// * `y` - Second probability
///
/// # Returns
/// The relative entropy term x * log(x/y)
#[allow(dead_code)]
pub fn rel_entr<T>(x: T, y: T) -> T
where
    T: Float + FromPrimitive + Zero,
{
    if x.is_zero() {
        T::zero()
    } else if y.is_zero() {
        T::infinity()
    } else {
        x * (x / y).ln()
    }
}

/// Kullback-Leibler divergence
///
/// Computes the KL divergence: x * log(x/y) - x + y
/// This is a symmetrized version that's more numerically stable.
///
/// # Arguments
/// * `x` - First value
/// * `y` - Second value
///
/// # Returns
/// The KL divergence value
#[allow(dead_code)]
pub fn kl_div<T>(x: T, y: T) -> T
where
    T: Float + FromPrimitive + Zero,
{
    if x.is_zero() {
        y
    } else if y.is_zero() {
        T::infinity()
    } else {
        x * (x / y).ln() - x + y
    }
}

/// Huber loss function
///
/// The Huber loss is a robust loss function that behaves like squared error
/// for small errors and like absolute error for large errors.
///
/// huber(δ, r) = { r²/2           if |r| <= δ
///               { δ|r| - δ²/2    if |r| > δ
///
/// # Arguments
/// * `delta` - Threshold parameter (must be positive)
/// * `r` - Residual value
///
/// # Returns
/// The Huber loss value
#[allow(dead_code)]
pub fn huber<T>(delta: T, r: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(delta, "delta value")?;
    check_finite(r, "r value")?;

    if delta <= T::zero() {
        return Err(SpecialError::DomainError(
            "huber: delta must be positive".to_string(),
        ));
    }

    let abs_r = r.abs();

    if abs_r <= delta {
        Ok(r * r / T::from_f64(2.0).unwrap())
    } else {
        Ok(delta * abs_r - delta * delta / T::from_f64(2.0).unwrap())
    }
}

/// Pseudo-Huber loss function
///
/// A smooth approximation to the Huber loss function:
/// pseudo_huber(δ, r) = δ² * (sqrt(1 + (r/δ)²) - 1)
///
/// # Arguments
/// * `delta` - Scale parameter (must be positive)
/// * `r` - Residual value
///
/// # Returns
/// The pseudo-Huber loss value
#[allow(dead_code)]
pub fn pseudo_huber<T>(delta: T, r: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(delta, "delta value")?;
    check_finite(r, "r value")?;

    if delta <= T::zero() {
        return Err(SpecialError::DomainError(
            "pseudo_huber: delta must be positive".to_string(),
        ));
    }

    let r_overdelta = r / delta;
    let delta_squared = delta * delta;

    Ok(delta_squared * ((T::one() + r_overdelta * r_overdelta).sqrt() - T::one()))
}

/// Apply entropy function to array
///
/// Computes -x * log(x) element-wise for an array.
#[allow(dead_code)]
pub fn entr_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Zero + Send + Sync,
{
    x.mapv(entr)
}

/// Compute total entropy of a probability distribution
///
/// Computes H(p) = -Σ p_i * log(p_i)
///
/// # Arguments
/// * `p` - Probability distribution (must sum to 1)
///
/// # Returns
/// The Shannon entropy of the distribution
#[allow(dead_code)]
pub fn entropy<T>(p: &ArrayView1<T>) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Zero + Display + Debug,
{
    // Check that probabilities are non-negative
    for &pi in p.iter() {
        check_non_negative(pi, "probability")?;
    }

    // Compute entropy
    let mut h = T::zero();
    for &pi in p.iter() {
        h = h + entr(pi);
    }

    Ok(h)
}

/// Compute KL divergence between two probability distributions
///
/// Computes D_KL(p || q) = Σ p_i * log(p_i / q_i)
///
/// # Arguments
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
/// The KL divergence from q to p
#[allow(dead_code)]
pub fn kl_divergence<T>(p: &ArrayView1<T>, q: &ArrayView1<T>) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Zero + Display + Debug,
{
    if p.len() != q.len() {
        return Err(SpecialError::ValueError(
            "kl_divergence: arrays must have the same length".to_string(),
        ));
    }

    let mut kl = T::zero();
    for i in 0..p.len() {
        kl = kl + rel_entr(p[i], q[i]);
    }

    Ok(kl)
}

/// Apply Huber loss to arrays of predictions and targets
///
/// # Arguments
/// * `delta` - Threshold parameter
/// * `predictions` - Predicted values
/// * `targets` - True values
/// * `output` - Output array for losses
#[allow(dead_code)]
pub fn huber_loss<T>(
    delta: T,
    predictions: &ArrayView1<T>,
    targets: &ArrayView1<T>,
    output: &mut ArrayViewMut1<T>,
) -> SpecialResult<()>
where
    T: Float + FromPrimitive + Display + Debug,
{
    if predictions.len() != targets.len() || predictions.len() != output.len() {
        return Err(SpecialError::ValueError(
            "huber_loss: all arrays must have the same length".to_string(),
        ));
    }

    for i in 0..predictions.len() {
        let residual = predictions[i] - targets[i];
        output[i] = huber(delta, residual)?;
    }

    Ok(())
}

/// Binary entropy function
///
/// Computes the binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
///
/// # Returns
/// The binary entropy
#[allow(dead_code)]
pub fn binary_entropy<T>(p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    crate::validation::check_probability(p, "p")?;

    if p.is_zero() || p == T::one() {
        return Ok(T::zero());
    }

    Ok(entr(p) + entr(T::one() - p))
}

/// Cross entropy between two probability distributions
///
/// Computes H(p, q) = -Σ p_i * log(q_i)
///
/// # Arguments
/// * `p` - True probability distribution
/// * `q` - Predicted probability distribution
///
/// # Returns
/// The cross entropy
#[allow(dead_code)]
pub fn cross_entropy<T>(p: &ArrayView1<T>, q: &ArrayView1<T>) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Zero + Display + Debug,
{
    if p.len() != q.len() {
        return Err(SpecialError::ValueError(
            "cross_entropy: arrays must have the same length".to_string(),
        ));
    }

    let mut ce = T::zero();
    for i in 0..p.len() {
        if p[i] > T::zero() {
            if q[i].is_zero() {
                return Ok(T::infinity());
            }
            ce = ce - p[i] * q[i].ln();
        }
    }

    Ok(ce)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_entr() {
        assert_eq!(entr(0.0), 0.0);
        assert_relative_eq!(entr(0.5), 0.34657359027997264, epsilon = 1e-10);
        assert_relative_eq!(entr(1.0), 0.0, epsilon = 1e-10);
        assert!(entr(-1.0).is_infinite() && entr(-1.0) < 0.0);
    }

    #[test]
    fn test_rel_entr() {
        assert_eq!(rel_entr(0.0, 1.0), 0.0);
        assert!(rel_entr(1.0, 0.0).is_infinite());
        assert_relative_eq!(rel_entr(0.5, 0.5), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rel_entr(0.7, 0.3), 0.5931085022710425, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_div() {
        assert_eq!(kl_div(0.0, 1.0), 1.0);
        assert!(kl_div(1.0, 0.0).is_infinite());
        assert_relative_eq!(kl_div(0.5, 0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_huber() {
        let delta = 1.0;

        // Small residuals (quadratic region)
        assert_relative_eq!(huber(delta, 0.5).unwrap(), 0.125, epsilon = 1e-10);
        assert_relative_eq!(huber(delta, -0.5).unwrap(), 0.125, epsilon = 1e-10);

        // Large residuals (linear region)
        assert_relative_eq!(huber(delta, 2.0).unwrap(), 1.5, epsilon = 1e-10);
        assert_relative_eq!(huber(delta, -2.0).unwrap(), 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pseudo_huber() {
        let delta = 1.0;

        assert_relative_eq!(pseudo_huber(delta, 0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(
            pseudo_huber(delta, 1.0).unwrap(),
            0.41421356237309515,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_entropy() {
        let uniform = arr1(&[0.25, 0.25, 0.25, 0.25]);
        let h = entropy(&uniform.view()).unwrap();
        assert_relative_eq!(h, 1.3862943611198906, epsilon = 1e-10); // log(4)

        let certain = arr1(&[1.0, 0.0, 0.0, 0.0]);
        let h = entropy(&certain.view()).unwrap();
        assert_relative_eq!(h, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let p = arr1(&[0.5, 0.5]);
        let q = arr1(&[0.9, 0.1]);
        let kl = kl_divergence(&p.view(), &q.view()).unwrap();
        assert!(kl > 0.0); // KL divergence is always non-negative
    }

    #[test]
    fn test_binary_entropy() {
        assert_eq!(binary_entropy(0.0).unwrap(), 0.0);
        assert_eq!(binary_entropy(1.0).unwrap(), 0.0);
        assert_relative_eq!(
            binary_entropy(0.5).unwrap(),
            std::f64::consts::LN_2,
            epsilon = 1e-10
        ); // log(2)
    }
}
