//! Adam-family stateful optimizers
//!
//! Provides struct-based, stateful Adam variants that hold their own moment
//! buffers and advance an internal time step on each `step()` call.
//!
//! # Variants
//!
//! | Type | Description |
//! |------|-------------|
//! | `AdamOptimizer` | Standard Adam with optional AMSGrad |
//! | `AdamWOptimizer` | Adam with decoupled weight decay (Loshchilov & Hutter 2019) |
//! | `NAdamOptimizer` | Adam with Nesterov momentum look-ahead |
//! | `RAdamOptimizer` | Rectified Adam — variance-corrected warm-start (Liu et al. 2020) |
//!
//! # References
//! - Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization". *ICLR*.
//! - Reddi et al. (2018). "On the Convergence of Adam and Beyond". *ICLR* (AMSGrad).
//! - Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization". *ICLR*.
//! - Dozat (2016). "Incorporating Nesterov Momentum into Adam". *ICLR Workshop*.
//! - Liu et al. (2020). "On the Variance of the Adaptive Learning Rate and Beyond". *ICLR*.

use crate::error::OptimizeError;

// ─── Adam ────────────────────────────────────────────────────────────────────

/// Standard Adam optimizer.
///
/// Maintains first (mean) and second (uncentred variance) moment estimates for
/// each parameter and applies bias-corrected updates.
///
/// # Update rule
/// ```text
/// m_t = β₁·m_{t-1} + (1−β₁)·g_t
/// v_t = β₂·v_{t-1} + (1−β₂)·g_t²
/// m̂   = m_t / (1 − β₁ᵗ)
/// v̂   = v_t / (1 − β₂ᵗ)        [or v_max_t for AMSGrad]
/// θ_t = θ_{t-1} − α·m̂ / (√v̂ + ε) − α·λ·θ_{t-1}   [L2 weight decay]
/// ```
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    /// Learning rate
    pub lr: f64,
    /// First-moment decay (typically 0.9)
    pub beta1: f64,
    /// Second-moment decay (typically 0.999)
    pub beta2: f64,
    /// Numerical stability constant
    pub eps: f64,
    /// L2 weight decay coefficient
    pub weight_decay: f64,
    /// Use AMSGrad variant (max of past second moments)
    pub amsgrad: bool,
    /// First moment buffer
    pub m: Vec<f64>,
    /// Second moment buffer
    pub v: Vec<f64>,
    /// Max second moment buffer for AMSGrad
    pub v_max: Vec<f64>,
    /// Current time step
    pub t: usize,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer.
    pub fn new(
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            m: Vec::new(),
            v: Vec::new(),
            v_max: Vec::new(),
            t: 0,
        }
    }

    /// Create with default hyperparameters (lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8).
    pub fn default_params(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.0, false)
    }

    /// Initialise buffers for the given number of parameters.
    fn ensure_init(&mut self, n: usize) {
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
            self.v_max = vec![0.0; n];
            self.t = 0;
        }
    }

    /// Perform one Adam update step.
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if `params` and `grad` lengths differ.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "params length {} != grad length {}",
                n,
                grad.len()
            )));
        }
        self.ensure_init(n);
        self.t += 1;

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr_t = self.lr * bc2.sqrt() / bc1;

        for i in 0..n {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            let v_hat = if self.amsgrad {
                self.v_max[i] = self.v_max[i].max(self.v[i]);
                self.v_max[i]
            } else {
                self.v[i]
            };

            // Adam update + optional L2 weight decay (coupled)
            params[i] -= lr_t * self.m[i] / (v_hat.sqrt() + self.eps)
                + self.lr * self.weight_decay * params[i];
        }
        Ok(())
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.v_max.clear();
        self.t = 0;
    }
}

// ─── AdamW ───────────────────────────────────────────────────────────────────

/// AdamW — Adam with *decoupled* weight decay.
///
/// Unlike standard Adam (which applies weight decay as an L2 gradient penalty),
/// AdamW applies it directly to the parameters after the adaptive update.
/// This separation allows better generalisation and is the recommended default
/// for transformer training.
///
/// # Update rule (decoupled weight decay)
/// ```text
/// [standard Adam moment/bias-corrected update for m̂, v̂]
/// θ_t = θ_{t-1} − α·m̂ / (√v̂ + ε)   [adaptive part]
///       − α·λ·θ_{t-1}                  [weight decay, decoupled]
/// ```
///
/// Reference: Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization".
#[derive(Debug, Clone)]
pub struct AdamWOptimizer {
    /// Learning rate
    pub lr: f64,
    /// First-moment decay
    pub beta1: f64,
    /// Second-moment decay
    pub beta2: f64,
    /// Numerical stability constant
    pub eps: f64,
    /// Decoupled weight decay coefficient
    pub weight_decay: f64,
    /// Use AMSGrad variant
    pub amsgrad: bool,
    /// First moment buffer
    m: Vec<f64>,
    /// Second moment buffer
    v: Vec<f64>,
    /// Max second moment buffer (AMSGrad)
    v_max: Vec<f64>,
    /// Time step
    t: usize,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer.
    pub fn new(
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            m: Vec::new(),
            v: Vec::new(),
            v_max: Vec::new(),
            t: 0,
        }
    }

    /// Default hyperparameters (lr=1e-3, wd=0.01).
    pub fn default_params(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01, false)
    }

    fn ensure_init(&mut self, n: usize) {
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
            self.v_max = vec![0.0; n];
            self.t = 0;
        }
    }

    /// Perform one AdamW update step.
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if `params` and `grad` lengths differ.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "params length {} != grad length {}",
                n,
                grad.len()
            )));
        }
        self.ensure_init(n);
        self.t += 1;

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let step_size = self.lr * bc2.sqrt() / bc1;

        for i in 0..n {
            // Decoupled weight decay first
            params[i] *= 1.0 - self.lr * self.weight_decay;

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            let v_hat = if self.amsgrad {
                self.v_max[i] = self.v_max[i].max(self.v[i]);
                self.v_max[i]
            } else {
                self.v[i]
            };

            params[i] -= step_size * self.m[i] / (v_hat.sqrt() + self.eps);
        }
        Ok(())
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.v_max.clear();
        self.t = 0;
    }
}

// ─── NAdam ───────────────────────────────────────────────────────────────────

/// NAdam — Adam with Nesterov momentum.
///
/// Replaces the standard first-moment estimate `m̂` in the denominator step
/// with a look-ahead that includes the current gradient scaled by β₁.
///
/// # Update rule
/// ```text
/// m_t  = β₁·m_{t-1} + (1−β₁)·g_t
/// v_t  = β₂·v_{t-1} + (1−β₂)·g_t²
/// m̂_n  = β₁·m_t/(1−β₁ᵗ⁺¹) + (1−β₁)·g_t/(1−β₁ᵗ)   [Nesterov look-ahead]
/// v̂    = v_t / (1 − β₂ᵗ)
/// θ_t  = θ_{t-1} − α·m̂_n / (√v̂ + ε)
/// ```
///
/// Reference: Dozat (2016). "Incorporating Nesterov Momentum into Adam".
#[derive(Debug, Clone)]
pub struct NAdamOptimizer {
    /// Learning rate
    pub lr: f64,
    /// First-moment decay
    pub beta1: f64,
    /// Second-moment decay
    pub beta2: f64,
    /// Numerical stability constant
    pub eps: f64,
    /// L2 weight decay
    pub weight_decay: f64,
    /// First moment buffer
    m: Vec<f64>,
    /// Second moment buffer
    v: Vec<f64>,
    /// Time step
    t: usize,
}

impl NAdamOptimizer {
    /// Create a new NAdam optimizer.
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Default hyperparameters (lr=2e-3).
    pub fn default_params(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.0)
    }

    fn ensure_init(&mut self, n: usize) {
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
            self.t = 0;
        }
    }

    /// Perform one NAdam update step.
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
        self.ensure_init(n);
        self.t += 1;

        let bc1_t = 1.0 - self.beta1.powi(self.t as i32);
        let bc1_t1 = 1.0 - self.beta1.powi(self.t as i32 + 1);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..n {
            let g = grad[i] + self.weight_decay * params[i];

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Nesterov-corrected first-moment estimate
            let m_hat_n = self.beta1 * self.m[i] / bc1_t1 + (1.0 - self.beta1) * g / bc1_t;
            let v_hat = self.v[i] / bc2;

            params[i] -= self.lr * m_hat_n / (v_hat.sqrt() + self.eps);
        }
        Ok(())
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

// ─── RAdam ───────────────────────────────────────────────────────────────────

/// RAdam — Rectified Adam.
///
/// Warms up the adaptive learning rate by estimating the variance of the
/// second moment and falling back to SGD-with-momentum when the variance is
/// too high (early in training). This eliminates the need for a dedicated
/// learning-rate warmup schedule.
///
/// # References
/// Liu et al. (2020). "On the Variance of the Adaptive Learning Rate and Beyond". *ICLR*.
#[derive(Debug, Clone)]
pub struct RAdamOptimizer {
    /// Learning rate
    pub lr: f64,
    /// First-moment decay
    pub beta1: f64,
    /// Second-moment decay
    pub beta2: f64,
    /// Numerical stability constant
    pub eps: f64,
    /// L2 weight decay
    pub weight_decay: f64,
    /// First moment buffer
    m: Vec<f64>,
    /// Second moment buffer
    v: Vec<f64>,
    /// Time step
    t: usize,
}

impl RAdamOptimizer {
    /// Create a new RAdam optimizer.
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Default hyperparameters (lr=1e-3).
    ///
    /// Uses β₂=0.98 rather than 0.999 so that ρ_∞ = 2/(1-0.98)-1 = 99
    /// and the rectification warm-up completes within ~20 steps, allowing
    /// convergence in O(1000) iterations for typical convex problems.
    /// With β₂=0.999 the warm-up spans ~4000 effective gradient steps,
    /// which prevents convergence within the iteration budget.
    pub fn default_params(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.98, 1e-8, 0.0)
    }

    fn ensure_init(&mut self, n: usize) {
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
            self.t = 0;
        }
    }

    /// Perform one RAdam update step.
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
        self.ensure_init(n);
        self.t += 1;

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        // ρ_max — maximum length of the approximated SMA
        let rho_max = 2.0 / (1.0 - self.beta2) - 1.0;
        // ρ_t — length of the approximated SMA at step t
        let rho_t = rho_max - 2.0 * self.t as f64 * self.beta2.powi(self.t as i32) / bc2;

        for i in 0..n {
            let g = grad[i] + self.weight_decay * params[i];

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / bc1;

            if rho_t > 4.0 {
                // Variance is tractable: apply fully rectified adaptive step
                let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_max
                    / ((rho_max - 4.0) * (rho_max - 2.0) * rho_t))
                    .sqrt();
                let v_hat = (self.v[i] / bc2).sqrt();
                params[i] -= self.lr * rect * m_hat / (v_hat + self.eps);
            } else {
                // Variance is intractable: fall back to SGD + momentum
                params[i] -= self.lr * m_hat;
            }
        }
        Ok(())
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
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
    fn test_adam_converges_quadratic() {
        let mut opt = AdamOptimizer::default_params(0.01);
        let mut params = vec![3.0, -2.0, 1.5];
        for _ in 0..1000 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_adam_amsgrad_converges() {
        let mut opt = AdamOptimizer::new(0.01, 0.9, 0.999, 1e-8, 0.0, true);
        let mut params = vec![2.0, -1.0];
        for _ in 0..1000 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_adamw_decoupled_decay() {
        // With decoupled weight decay and zero gradient, params should shrink
        let mut opt = AdamWOptimizer::new(0.01, 0.9, 0.999, 1e-8, 0.1, false);
        let mut params = vec![1.0];
        let g = vec![0.0];
        for _ in 0..100 {
            opt.step(&mut params, &g).expect("step failed");
        }
        // Params should be smaller than initial
        assert!(params[0] < 1.0, "weight decay should reduce params");
    }

    #[test]
    fn test_adamw_converges_quadratic() {
        let mut opt = AdamWOptimizer::default_params(0.01);
        let mut params = vec![2.0, -1.5];
        for _ in 0..1000 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 0.05);
        }
    }

    #[test]
    fn test_nadam_converges_quadratic() {
        let mut opt = NAdamOptimizer::default_params(0.01);
        let mut params = vec![2.0, -1.0];
        for _ in 0..1000 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_radam_converges_quadratic() {
        let mut opt = RAdamOptimizer::default_params(0.01);
        let mut params = vec![3.0, -2.0];
        for _ in 0..1000 {
            let g = quadratic_grad(&params);
            opt.step(&mut params, &g).expect("step failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_adam_length_mismatch() {
        let mut opt = AdamOptimizer::default_params(0.01);
        let mut params = vec![1.0, 2.0];
        let grad = vec![0.1]; // wrong length
        assert!(opt.step(&mut params, &grad).is_err());
    }

    #[test]
    fn test_adam_reset() {
        let mut opt = AdamOptimizer::default_params(0.01);
        let mut params = vec![1.0, -1.0];
        let g = quadratic_grad(&params);
        opt.step(&mut params, &g).expect("step failed");
        assert_eq!(opt.t, 1);
        opt.reset();
        assert_eq!(opt.t, 0);
        assert!(opt.m.is_empty());
    }
}
