//! Stateful SGD optimizer variants
//!
//! This module provides stateful, struct-based optimizer objects that hold
//! their own parameter state across `step()` calls. These are suitable for
//! use in ML training loops where the same optimizer instance is updated
//! repeatedly.
//!
//! # Algorithms
//!
//! | Type | Description |
//! |------|-------------|
//! | `SgdOptimizer` | SGD with optional momentum (classical & Nesterov), weight decay |
//! | `AdaGradOptimizer` | Adaptive learning rates via accumulated squared gradients |
//! | `AdaDeltaOptimizer` | Adaptive learning rates without a global LR (Zeiler 2012) |
//!
//! # References
//!
//! - Polyak (1964). "Some methods of speeding up the convergence of iteration methods".
//! - Nesterov (1983). "A method of solving a convex programming problem".
//! - Duchi et al. (2011). "Adaptive Subgradient Methods for Online Learning". *JMLR*.
//! - Zeiler (2012). "ADADELTA: An Adaptive Learning Rate Method". arXiv:1212.5701.

use crate::error::OptimizeError;

// ─── SGD ─────────────────────────────────────────────────────────────────────

/// Stateful SGD optimizer with optional momentum, Nesterov momentum, and
/// L2 weight decay.
///
/// # Update rule (no Nesterov)
/// ```text
/// v_t = μ·v_{t-1} + g_t + λ·θ_{t-1}
/// θ_t = θ_{t-1} - α·v_t
/// ```
///
/// # Update rule (Nesterov)
/// ```text
/// v_t = μ·v_{t-1} + g_t + λ·θ_{t-1}
/// θ_t = θ_{t-1} - α·(g_t + μ·v_t)
/// ```
///
/// where α = `lr`, μ = `momentum`, λ = `weight_decay`.
#[derive(Debug, Clone)]
pub struct SgdOptimizer {
    /// Learning rate
    pub lr: f64,
    /// Momentum coefficient (0 = vanilla SGD)
    pub momentum: f64,
    /// Use Nesterov momentum
    pub nesterov: bool,
    /// L2 weight-decay coefficient
    pub weight_decay: f64,
    /// Velocity buffer (accumulated momentum); populated lazily on first step
    velocity: Vec<f64>,
}

impl SgdOptimizer {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// * `lr` - Learning rate (must be > 0)
    /// * `momentum` - Momentum factor in [0, 1)
    /// * `nesterov` - Whether to use Nesterov lookahead momentum
    /// * `weight_decay` - L2 regularisation strength (≥ 0)
    pub fn new(lr: f64, momentum: f64, nesterov: bool, weight_decay: f64) -> Self {
        Self {
            lr,
            momentum,
            nesterov,
            weight_decay,
            velocity: Vec::new(),
        }
    }

    /// Vanilla SGD with default hyperparameters (lr=0.01, no momentum).
    pub fn vanilla(lr: f64) -> Self {
        Self::new(lr, 0.0, false, 0.0)
    }

    /// Perform one SGD update step.
    ///
    /// # Arguments
    /// * `params` - Mutable parameter vector; updated in-place
    /// * `grad` - Gradient vector (same length as `params`)
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if `params` and `grad` have
    /// different lengths.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "params length {} != grad length {}",
                n,
                grad.len()
            )));
        }

        // Lazy initialisation of velocity buffer
        if self.velocity.len() != n {
            self.velocity = vec![0.0; n];
        }

        for i in 0..n {
            // Add L2 regularisation to gradient
            let g = grad[i] + self.weight_decay * params[i];

            if self.momentum == 0.0 {
                // Vanilla SGD
                params[i] -= self.lr * g;
            } else {
                // Update velocity
                self.velocity[i] = self.momentum * self.velocity[i] + g;

                if self.nesterov {
                    // Nesterov: use the "lookahead" gradient
                    params[i] -= self.lr * (g + self.momentum * self.velocity[i]);
                } else {
                    params[i] -= self.lr * self.velocity[i];
                }
            }
        }
        Ok(())
    }

    /// Reset velocity buffer (useful when restarting training).
    pub fn reset(&mut self) {
        self.velocity.clear();
    }
}

// ─── AdaGrad ─────────────────────────────────────────────────────────────────

/// AdaGrad optimizer.
///
/// Adapts the learning rate for each parameter by accumulating squared
/// gradients. Parameters that receive large, frequent gradients see smaller
/// effective learning rates.
///
/// # Update rule
/// ```text
/// G_t = G_{t-1} + g_t ⊙ g_t
/// θ_t = θ_{t-1} - α / (√G_t + ε) ⊙ g_t
/// ```
///
/// Reference: Duchi et al. (2011).
#[derive(Debug, Clone)]
pub struct AdaGradOptimizer {
    /// Global learning rate
    pub lr: f64,
    /// Numerical stability constant
    pub eps: f64,
    /// Accumulated squared gradients
    pub accum: Vec<f64>,
}

impl AdaGradOptimizer {
    /// Create a new AdaGrad optimizer.
    pub fn new(lr: f64, eps: f64) -> Self {
        Self {
            lr,
            eps,
            accum: Vec::new(),
        }
    }

    /// Create with default hyperparameters (lr=0.01, eps=1e-8).
    pub fn default_params(lr: f64) -> Self {
        Self::new(lr, 1e-8)
    }

    /// Perform one AdaGrad update step.
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if length mismatch.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "params length {} != grad length {}",
                n,
                grad.len()
            )));
        }

        if self.accum.len() != n {
            self.accum = vec![0.0; n];
        }

        for i in 0..n {
            self.accum[i] += grad[i] * grad[i];
            params[i] -= self.lr / (self.accum[i].sqrt() + self.eps) * grad[i];
        }
        Ok(())
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        self.accum.clear();
    }
}

// ─── AdaDelta ────────────────────────────────────────────────────────────────

/// AdaDelta optimizer.
///
/// Extends AdaGrad to avoid its monotonically decreasing learning rate by
/// using an exponentially decaying window of past squared gradients.
/// Importantly, no global learning rate is required.
///
/// # Update rule
/// ```text
/// E[g²]_t    = ρ·E[g²]_{t-1}    + (1-ρ)·g_t²
/// Δθ_t       = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) · g_t
/// E[Δθ²]_t   = ρ·E[Δθ²]_{t-1}  + (1-ρ)·Δθ_t²
/// θ_t        = θ_{t-1} + Δθ_t
/// ```
///
/// Reference: Zeiler (2012), "ADADELTA: An Adaptive Learning Rate Method".
#[derive(Debug, Clone)]
pub struct AdaDeltaOptimizer {
    /// Decay rate for running averages
    pub rho: f64,
    /// Numerical stability constant
    pub eps: f64,
    /// Running average of squared gradients: E[g²]
    pub accum_grad: Vec<f64>,
    /// Running average of squared updates: E[Δθ²]
    pub accum_update: Vec<f64>,
}

impl AdaDeltaOptimizer {
    /// Create a new AdaDelta optimizer.
    ///
    /// # Arguments
    /// * `rho` - Decay factor for exponential moving averages (typically 0.95)
    /// * `eps` - Numerical stability (typically 1e-6)
    pub fn new(rho: f64, eps: f64) -> Self {
        Self {
            rho,
            eps,
            accum_grad: Vec::new(),
            accum_update: Vec::new(),
        }
    }

    /// Create with default hyperparameters (rho=0.95, eps=1e-6).
    pub fn default_params() -> Self {
        Self::new(0.95, 1e-6)
    }

    /// Perform one AdaDelta update step.
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if length mismatch.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "params length {} != grad length {}",
                n,
                grad.len()
            )));
        }

        if self.accum_grad.len() != n {
            self.accum_grad = vec![0.0; n];
            self.accum_update = vec![0.0; n];
        }

        for i in 0..n {
            // Update running average of squared gradients
            self.accum_grad[i] =
                self.rho * self.accum_grad[i] + (1.0 - self.rho) * grad[i] * grad[i];

            // Compute parameter update using RMS of past updates
            let rms_update = (self.accum_update[i] + self.eps).sqrt();
            let rms_grad = (self.accum_grad[i] + self.eps).sqrt();
            let delta = -(rms_update / rms_grad) * grad[i];

            // Update running average of squared updates
            self.accum_update[i] =
                self.rho * self.accum_update[i] + (1.0 - self.rho) * delta * delta;

            params[i] += delta;
        }
        Ok(())
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        self.accum_grad.clear();
        self.accum_update.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| 2.0 * xi).collect()
    }

    #[test]
    fn test_sgd_vanilla_converges() {
        let mut opt = SgdOptimizer::vanilla(0.1);
        let mut params = vec![1.0, -2.0, 0.5];
        for _ in 0..200 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_sgd_momentum_converges() {
        let mut opt = SgdOptimizer::new(0.05, 0.9, false, 0.0);
        let mut params = vec![2.0, -1.5];
        for _ in 0..300 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_sgd_nesterov_converges() {
        let mut opt = SgdOptimizer::new(0.05, 0.9, true, 0.0);
        let mut params = vec![1.5, -1.0];
        for _ in 0..300 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_sgd_weight_decay() {
        // With weight decay, minimum shifts; check that update is applied
        let mut opt = SgdOptimizer::new(0.01, 0.0, false, 0.1);
        let mut params = vec![1.0];
        let init = params[0];
        let g = vec![0.0]; // zero gradient; only weight decay should pull
        opt.step(&mut params, &g).expect("step failed");
        assert!(params[0] < init, "weight decay should reduce param");
    }

    #[test]
    fn test_sgd_length_mismatch() {
        let mut opt = SgdOptimizer::vanilla(0.1);
        let mut params = vec![1.0, 2.0];
        let grad = vec![0.1]; // wrong length
        assert!(opt.step(&mut params, &grad).is_err());
    }

    #[test]
    fn test_adagrad_converges() {
        let mut opt = AdaGradOptimizer::default_params(0.5);
        let mut params = vec![3.0, -2.0];
        for _ in 0..500 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_adadelta_converges() {
        let mut opt = AdaDeltaOptimizer::default_params();
        let mut params = vec![2.0, -1.0];
        for _ in 0..2000 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_adadelta_length_mismatch() {
        let mut opt = AdaDeltaOptimizer::default_params();
        let mut params = vec![1.0, 2.0];
        let grad = vec![0.1, 0.2, 0.3]; // wrong length
        assert!(opt.step(&mut params, &grad).is_err());
    }
}
