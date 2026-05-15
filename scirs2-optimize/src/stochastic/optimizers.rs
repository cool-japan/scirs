//! Stateful first-order stochastic optimizers with learning rate schedules.
//!
//! Provides clean, struct-based implementations of:
//! - **SGD** — vanilla, momentum, Nesterov momentum, weight-decay
//! - **Adam** — adaptive moment estimation
//! - **AdaGrad** — cumulative gradient-square scaling
//! - **RMSprop** — exponential moving-average second moment
//! - **AdamW** — Adam with decoupled weight decay
//! - **SVRG** — Stochastic Variance Reduced Gradient
//! - **LrSchedule** — a rich enum of learning rate schedules
//!
//! # Usage pattern
//!
//! ```rust
//! use scirs2_optimize::stochastic::optimizers::{Sgd, LrSchedule};
//!
//! let mut opt = Sgd::new(0.01, 0.9);
//! let mut params = vec![1.0_f64, -2.0];
//! for _ in 0..100 {
//!     let grad = params.iter().map(|&p| 2.0 * p).collect::<Vec<_>>();
//!     opt.step(&mut params, &grad).expect("valid input");
//! }
//! ```

use crate::error::{OptimizeError, OptimizeResult};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Learning rate schedule
// ─────────────────────────────────────────────────────────────────────────────

/// Learning rate schedule variants.
///
/// All variants are pure functions of the current step count and return the
/// effective learning rate to use at that step.
#[derive(Debug, Clone)]
pub enum LrSchedule {
    /// Constant learning rate.
    Constant(f64),

    /// Exponential decay: lr = initial * decay^step.
    ExponentialDecay {
        /// Initial learning rate.
        initial: f64,
        /// Per-step multiplicative decay factor (0 < decay < 1).
        decay: f64,
    },

    /// Cosine annealing: lr oscillates between lr_min and lr_max over t_max steps.
    CosineAnnealing {
        /// Maximum (initial) learning rate.
        lr_max: f64,
        /// Minimum learning rate at the end of the cycle.
        lr_min: f64,
        /// Half-period (number of steps).
        t_max: usize,
    },

    /// Linear warmup followed by cosine annealing.
    WarmupCosine {
        /// Number of linear warmup steps (lr 0 → lr_peak).
        warmup_steps: usize,
        /// Peak learning rate reached after warmup.
        lr_peak: f64,
        /// Minimum learning rate at the end.
        lr_min: f64,
        /// Total number of steps (warmup + cosine phase).
        total_steps: usize,
    },

    /// Step decay: lr is multiplied by gamma every step_size steps.
    StepLr {
        /// Initial learning rate.
        initial: f64,
        /// Number of steps between reductions.
        step_size: usize,
        /// Multiplicative reduction factor per epoch.
        gamma: f64,
    },
}

impl LrSchedule {
    /// Return the effective learning rate at the given step index.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_optimize::stochastic::optimizers::LrSchedule;
    /// let sched = LrSchedule::Constant(0.01);
    /// assert!((sched.lr_at(42) - 0.01).abs() < 1e-14);
    /// ```
    pub fn lr_at(&self, step: usize) -> f64 {
        match self {
            LrSchedule::Constant(lr) => *lr,

            LrSchedule::ExponentialDecay { initial, decay } => initial * decay.powi(step as i32),

            LrSchedule::CosineAnnealing {
                lr_max,
                lr_min,
                t_max,
            } => {
                let t = (step % (2 * (*t_max).max(1))) as f64;
                let t_m = *t_max as f64;
                let cos_inner = PI * t / t_m;
                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_inner.cos())
            }

            LrSchedule::WarmupCosine {
                warmup_steps,
                lr_peak,
                lr_min,
                total_steps,
            } => {
                let ws = *warmup_steps;
                let ts = (*total_steps).max(ws + 1);
                if step < ws {
                    // Linear warmup
                    lr_peak * step as f64 / ws.max(1) as f64
                } else {
                    // Cosine decay from lr_peak to lr_min
                    let progress = (step - ws) as f64 / (ts - ws) as f64;
                    lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + (PI * progress).cos())
                }
            }

            LrSchedule::StepLr {
                initial,
                step_size,
                gamma,
            } => {
                let n_decays = step / (*step_size).max(1);
                initial * gamma.powi(n_decays as i32)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent with optional momentum, Nesterov lookahead,
/// and L2 weight decay.
///
/// Update rule (with momentum):
///   v ← momentum · v + lr · (grad + weight_decay · params)
///   params ← params - v        (vanilla momentum)
///   params ← params - (momentum · v + lr · grad)   (Nesterov)
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Base learning rate.
    pub learning_rate: f64,
    /// Momentum coefficient (0 = vanilla SGD).
    pub momentum: f64,
    /// L2 weight-decay coefficient.
    pub weight_decay: f64,
    /// Use Nesterov momentum instead of classical momentum.
    pub nesterov: bool,
    /// Velocity buffer (initialised lazily).
    velocity: Vec<f64>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
            velocity: Vec::new(),
        }
    }

    /// Perform one SGD update step.
    ///
    /// # Errors
    /// Returns [`OptimizeError::InvalidInput`] if `params` and `grad` have
    /// different lengths.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> OptimizeResult<()> {
        if params.len() != grad.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "params length {} != grad length {}",
                params.len(),
                grad.len()
            )));
        }

        let n = params.len();
        if self.velocity.len() != n {
            self.velocity = vec![0.0; n];
        }

        let lr = self.learning_rate;
        let mu = self.momentum;
        let wd = self.weight_decay;

        if self.nesterov {
            for i in 0..n {
                let g = grad[i] + wd * params[i];
                self.velocity[i] = mu * self.velocity[i] + g;
                params[i] -= lr * (mu * self.velocity[i] + g);
            }
        } else {
            for i in 0..n {
                let g = grad[i] + wd * params[i];
                self.velocity[i] = mu * self.velocity[i] + g;
                params[i] -= lr * self.velocity[i];
            }
        }
        Ok(())
    }

    /// Reset velocity buffer to zeros (for new training run).
    pub fn zero_velocity(&mut self, n: usize) {
        self.velocity = vec![0.0; n];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba 2015).
///
/// Update rule:
///   m ← β₁ m + (1-β₁) g
///   v ← β₂ v + (1-β₂) g²
///   m̂ = m / (1-β₁ᵗ),  v̂ = v / (1-β₂ᵗ)
///   params ← params - lr · m̂ / (√v̂ + ε)
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate.
    pub lr: f64,
    /// First moment decay (default 0.9).
    pub beta1: f64,
    /// Second moment decay (default 0.999).
    pub beta2: f64,
    /// Numerical stability constant (default 1e-8).
    pub eps: f64,
    /// L2 weight-decay (added to gradient, not decoupled).
    pub weight_decay: f64,
    /// First moment estimate.
    m: Vec<f64>,
    /// Second moment estimate.
    v: Vec<f64>,
    /// Step counter (1-indexed).
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer with default β₁=0.9, β₂=0.999, ε=1e-8.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Perform one Adam update step.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> OptimizeResult<()> {
        if params.len() != grad.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "params length {} != grad length {}",
                params.len(),
                grad.len()
            )));
        }

        let n = params.len();
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
        }

        self.t += 1;
        let t = self.t as f64;
        let bias_corr1 = 1.0 - self.beta1.powf(t);
        let bias_corr2 = 1.0 - self.beta2.powf(t);

        for i in 0..n {
            let g = grad[i] + self.weight_decay * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
        Ok(())
    }

    /// Reset all state (moments and step counter).
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaGrad
// ─────────────────────────────────────────────────────────────────────────────

/// AdaGrad optimizer (Duchi et al. 2011).
///
/// Accumulates the sum of squared gradients per parameter and scales the
/// learning rate accordingly:
///   G ← G + g²
///   params ← params - lr · g / (√G + ε)
#[derive(Debug, Clone)]
pub struct AdaGrad {
    /// Base learning rate.
    pub lr: f64,
    /// Numerical stability constant.
    pub eps: f64,
    /// L2 weight-decay (coupled).
    pub weight_decay: f64,
    /// Accumulated squared-gradient sum.
    sum_sq_grad: Vec<f64>,
}

impl AdaGrad {
    /// Create a new AdaGrad optimizer.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            eps: 1e-8,
            weight_decay: 0.0,
            sum_sq_grad: Vec::new(),
        }
    }

    /// Perform one AdaGrad update step.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> OptimizeResult<()> {
        if params.len() != grad.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "params/grad length mismatch: {} vs {}",
                params.len(),
                grad.len()
            )));
        }

        let n = params.len();
        if self.sum_sq_grad.len() != n {
            self.sum_sq_grad = vec![0.0; n];
        }

        for i in 0..n {
            let g = grad[i] + self.weight_decay * params[i];
            self.sum_sq_grad[i] += g * g;
            params[i] -= self.lr * g / (self.sum_sq_grad[i].sqrt() + self.eps);
        }
        Ok(())
    }

    /// Reset accumulated state.
    pub fn reset_state(&mut self) {
        self.sum_sq_grad.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSprop
// ─────────────────────────────────────────────────────────────────────────────

/// RMSprop optimizer (Hinton 2012).
///
/// Maintains an exponential moving average of squared gradients:
///   E\[g²\] ← α·E\[g²\] + (1-α)·g²
///   params ← params - lr · g / (√E\[g²\] + ε)
///
/// With momentum > 0:
///   vel ← momentum·vel + lr · g / (√E\[g²\] + ε)
///   params ← params - vel
#[derive(Debug, Clone)]
pub struct RmsProp {
    /// Base learning rate.
    pub lr: f64,
    /// Decay factor for the moving average (default 0.99).
    pub alpha: f64,
    /// Numerical stability constant.
    pub eps: f64,
    /// Momentum coefficient (0 = no momentum).
    pub momentum: f64,
    /// Moving-average of squared gradients.
    sq_avg: Vec<f64>,
    /// Momentum buffer.
    velocity: Vec<f64>,
}

impl RmsProp {
    /// Create a new RMSprop optimizer with default α=0.99.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            sq_avg: Vec::new(),
            velocity: Vec::new(),
        }
    }

    /// Perform one RMSprop update step.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> OptimizeResult<()> {
        if params.len() != grad.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "params/grad length mismatch: {} vs {}",
                params.len(),
                grad.len()
            )));
        }

        let n = params.len();
        if self.sq_avg.len() != n {
            self.sq_avg = vec![0.0; n];
            self.velocity = vec![0.0; n];
        }

        for i in 0..n {
            let g = grad[i];
            self.sq_avg[i] = self.alpha * self.sq_avg[i] + (1.0 - self.alpha) * g * g;
            let denom = self.sq_avg[i].sqrt() + self.eps;
            if self.momentum > 0.0 {
                self.velocity[i] = self.momentum * self.velocity[i] + self.lr * g / denom;
                params[i] -= self.velocity[i];
            } else {
                params[i] -= self.lr * g / denom;
            }
        }
        Ok(())
    }

    /// Reset accumulated state.
    pub fn reset_state(&mut self) {
        self.sq_avg.clear();
        self.velocity.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdamW
// ─────────────────────────────────────────────────────────────────────────────

/// AdamW optimizer (Loshchilov & Hutter 2017) — Adam with decoupled weight decay.
///
/// Unlike Adam (which couples weight decay into the gradient), AdamW applies
/// weight decay directly to the parameters before the gradient update:
///   params ← params - lr · weight_decay · params   (L2 shrinkage)
///   then Adam update on the raw gradient
#[derive(Debug, Clone)]
pub struct AdamW {
    /// Learning rate.
    pub lr: f64,
    /// First moment decay.
    pub beta1: f64,
    /// Second moment decay.
    pub beta2: f64,
    /// Numerical stability constant.
    pub eps: f64,
    /// Decoupled L2 weight-decay coefficient.
    pub weight_decay: f64,
    /// First moment.
    m: Vec<f64>,
    /// Second moment.
    v: Vec<f64>,
    /// Step counter.
    t: usize,
}

impl AdamW {
    /// Create a new AdamW optimizer with default β₁=0.9, β₂=0.999, ε=1e-8,
    /// weight_decay=0.01.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Perform one AdamW update step.
    pub fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) -> OptimizeResult<()> {
        if params.len() != grad.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "params/grad length mismatch: {} vs {}",
                params.len(),
                grad.len()
            )));
        }

        let n = params.len();
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
        }

        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for i in 0..n {
            // Decoupled weight decay: shrink parameters first
            params[i] *= 1.0 - self.lr * self.weight_decay;

            let g = grad[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
        Ok(())
    }

    /// Reset all state.
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SVRG
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic Variance Reduced Gradient (SVRG; Johnson & Zhang 2013).
///
/// SVRG maintains a snapshot of parameters and the corresponding full gradient,
/// and uses a control variate to reduce variance of the stochastic gradient
/// estimate:
///
///   g̃ = g_i(x) - g_i(x̃) + ∇f(x̃)
///   x ← x - lr · g̃
///
/// The snapshot (x̃, ∇f(x̃)) must be updated every `update_freq` inner steps
/// by calling [`Svrg::update_snapshot`].
#[derive(Debug, Clone)]
pub struct Svrg {
    /// Learning rate.
    pub lr: f64,
    /// Dataset size (used for normalization in documentation, not internally).
    pub n: usize,
    /// Number of inner steps between snapshot updates.
    pub update_freq: usize,
    /// Snapshot of parameters x̃.
    snapshot_params: Vec<f64>,
    /// Full gradient at snapshot ∇f(x̃).
    snapshot_grad: Vec<f64>,
    /// Inner iteration counter.
    inner_t: usize,
}

impl Svrg {
    /// Create a new SVRG optimizer.
    ///
    /// # Arguments
    /// * `lr`          – Learning rate.
    /// * `n`           – Dataset size.
    /// * `update_freq` – Inner-loop length (snapshot updated every this many steps).
    pub fn new(lr: f64, n: usize, update_freq: usize) -> Self {
        Self {
            lr,
            n,
            update_freq,
            snapshot_params: Vec::new(),
            snapshot_grad: Vec::new(),
            inner_t: 0,
        }
    }

    /// Perform one SVRG inner-loop update.
    ///
    /// # Arguments
    /// * `params`           – Current parameters (modified in place).
    /// * `stochastic_grad`  – Mini-batch gradient at current `params`.
    /// * `snapshot_grad_i`  – Mini-batch gradient at the snapshot params (same mini-batch).
    ///
    /// # Errors
    /// Returns [`OptimizeError::InvalidInput`] on length mismatches, or if
    /// `update_snapshot` has not been called first.
    pub fn step(
        &mut self,
        params: &mut Vec<f64>,
        stochastic_grad: &[f64],
        snapshot_grad_i: &[f64],
    ) -> OptimizeResult<()> {
        let n = params.len();

        if stochastic_grad.len() != n || snapshot_grad_i.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "SVRG gradient/param length mismatch: params={}, sg={}, sgi={}",
                n,
                stochastic_grad.len(),
                snapshot_grad_i.len()
            )));
        }

        if self.snapshot_grad.len() != n {
            return Err(OptimizeError::InvalidInput(
                "SVRG: snapshot not initialised — call update_snapshot first".to_string(),
            ));
        }

        // Variance-reduced gradient estimate
        for i in 0..n {
            let g_tilde = stochastic_grad[i] - snapshot_grad_i[i] + self.snapshot_grad[i];
            params[i] -= self.lr * g_tilde;
        }

        self.inner_t += 1;
        Ok(())
    }

    /// Update the snapshot with current parameters and full gradient.
    ///
    /// Should be called at the start of each outer epoch (every `update_freq`
    /// inner steps).
    pub fn update_snapshot(&mut self, params: &[f64], full_grad: &[f64]) {
        self.snapshot_params = params.to_vec();
        self.snapshot_grad = full_grad.to_vec();
        self.inner_t = 0;
    }

    /// Whether the inner loop has completed and a snapshot update is due.
    pub fn needs_snapshot_update(&self) -> bool {
        self.inner_t >= self.update_freq
    }

    /// Current snapshot parameters.
    pub fn snapshot_params(&self) -> &[f64] {
        &self.snapshot_params
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn quadratic_grad(params: &[f64]) -> Vec<f64> {
        params.iter().map(|&p| 2.0 * p).collect()
    }

    // ── LrSchedule ───────────────────────────────────────────────────────────

    #[test]
    fn test_constant_schedule() {
        let s = LrSchedule::Constant(0.01);
        assert_abs_diff_eq!(s.lr_at(0), 0.01, epsilon = 1e-14);
        assert_abs_diff_eq!(s.lr_at(1000), 0.01, epsilon = 1e-14);
    }

    #[test]
    fn test_exponential_decay_schedule() {
        let s = LrSchedule::ExponentialDecay {
            initial: 0.1,
            decay: 0.9,
        };
        assert_abs_diff_eq!(s.lr_at(0), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(s.lr_at(1), 0.09, epsilon = 1e-10);
        assert_abs_diff_eq!(s.lr_at(10), 0.1 * 0.9_f64.powi(10), epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_annealing_at_zero() {
        let s = LrSchedule::CosineAnnealing {
            lr_max: 0.1,
            lr_min: 0.0,
            t_max: 100,
        };
        // At step 0: cos(0) = 1 → lr = lr_min + 0.5*(lr_max-lr_min)*2 = lr_max
        assert_abs_diff_eq!(s.lr_at(0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_annealing_at_t_max() {
        let s = LrSchedule::CosineAnnealing {
            lr_max: 0.1,
            lr_min: 0.001,
            t_max: 50,
        };
        // At step t_max: cos(π) = -1 → lr = lr_min
        assert_abs_diff_eq!(s.lr_at(50), 0.001, epsilon = 1e-10);
    }

    #[test]
    fn test_warmup_cosine_warmup_phase() {
        let s = LrSchedule::WarmupCosine {
            warmup_steps: 10,
            lr_peak: 0.1,
            lr_min: 0.0,
            total_steps: 110,
        };
        // Step 5 of 10 warmup → lr = 0.1 * 5/10 = 0.05
        assert_abs_diff_eq!(s.lr_at(5), 0.05, epsilon = 1e-10);
        // After warmup start: lr = lr_peak at step 10 (cos(0) phase)
        let lr10 = s.lr_at(10);
        assert!(
            lr10 >= 0.09 && lr10 <= 0.1 + 1e-9,
            "lr at warmup end ≈ peak, got {}",
            lr10
        );
    }

    #[test]
    fn test_step_lr_schedule() {
        let s = LrSchedule::StepLr {
            initial: 0.1,
            step_size: 10,
            gamma: 0.5,
        };
        assert_abs_diff_eq!(s.lr_at(0), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(s.lr_at(9), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(s.lr_at(10), 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(s.lr_at(20), 0.025, epsilon = 1e-12);
    }

    // ── SGD ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_sgd_converges_quadratic() {
        let mut opt = Sgd::new(0.1, 0.0);
        let mut p = vec![1.0, -2.0];
        for _ in 0..200 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert_abs_diff_eq!(p[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(p[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sgd_momentum_converges() {
        let mut opt = Sgd::new(0.05, 0.9);
        let mut p = vec![2.0, -1.5];
        for _ in 0..500 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert_abs_diff_eq!(p[0], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(p[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_sgd_nesterov() {
        let mut opt = Sgd {
            nesterov: true,
            ..Sgd::new(0.05, 0.9)
        };
        let mut p = vec![1.0, 1.0];
        for _ in 0..500 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert_abs_diff_eq!(p[0], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_sgd_weight_decay() {
        let mut opt = Sgd {
            weight_decay: 0.1,
            ..Sgd::new(0.01, 0.0)
        };
        let mut p = vec![1.0];
        opt.step(&mut p, &[0.0]).expect("step failed");
        // Weight decay pulls toward 0: p_new = p - lr * wd * p = p * (1 - lr*wd)
        assert!(p[0] < 1.0, "weight decay should shrink param");
    }

    #[test]
    fn test_sgd_length_mismatch() {
        let mut opt = Sgd::new(0.01, 0.0);
        let mut p = vec![1.0, 2.0];
        assert!(opt.step(&mut p, &[0.1]).is_err());
    }

    #[test]
    fn test_sgd_zero_velocity() {
        let mut opt = Sgd::new(0.01, 0.9);
        opt.zero_velocity(5);
        assert_eq!(opt.velocity.len(), 5);
        assert!(opt.velocity.iter().all(|&v| v == 0.0));
    }

    // ── Adam ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_adam_converges() {
        let mut opt = Adam::new(0.01);
        let mut p = vec![3.0, -3.0];
        for _ in 0..1000 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert_abs_diff_eq!(p[0], 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(p[1], 0.0, epsilon = 1e-2);
    }

    #[test]
    fn test_adam_reset_state() {
        let mut opt = Adam::new(0.01);
        let mut p = vec![1.0];
        opt.step(&mut p, &[0.5]).expect("step failed");
        assert_eq!(opt.t, 1);
        opt.reset_state();
        assert_eq!(opt.t, 0);
        assert!(opt.m.is_empty());
        assert!(opt.v.is_empty());
    }

    #[test]
    fn test_adam_weight_decay_coupled() {
        let mut opt = Adam {
            weight_decay: 0.01,
            ..Adam::new(0.001)
        };
        let mut p = vec![1.0];
        let p_before = p[0];
        opt.step(&mut p, &[0.0]).expect("step failed");
        // With pure weight decay (grad=0), param should shrink
        assert!(p[0] < p_before, "weight decay should reduce param");
    }

    // ── AdaGrad ──────────────────────────────────────────────────────────────

    #[test]
    fn test_adagrad_converges() {
        let mut opt = AdaGrad::new(0.5);
        let mut p = vec![3.0, -2.0];
        for _ in 0..2000 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert!(p[0].abs() < 0.5, "adagrad should converge, p[0]={}", p[0]);
    }

    #[test]
    fn test_adagrad_reset() {
        let mut opt = AdaGrad::new(0.1);
        let mut p = vec![1.0];
        opt.step(&mut p, &[1.0]).expect("step failed");
        assert_eq!(opt.sum_sq_grad.len(), 1);
        opt.reset_state();
        assert!(opt.sum_sq_grad.is_empty());
    }

    // ── RMSprop ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rmsprop_converges() {
        let mut opt = RmsProp::new(0.01);
        let mut p = vec![2.0, -2.0];
        for _ in 0..1000 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert!(p[0].abs() < 0.1, "rmsprop p[0]={}", p[0]);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let mut opt = RmsProp {
            momentum: 0.9,
            ..RmsProp::new(0.01)
        };
        let mut p = vec![1.0, 1.0];
        for _ in 0..500 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert!(p[0].abs() < 0.5, "rmsprop+momentum p[0]={}", p[0]);
    }

    #[test]
    fn test_rmsprop_length_mismatch() {
        let mut opt = RmsProp::new(0.01);
        let mut p = vec![1.0, 2.0];
        assert!(opt.step(&mut p, &[0.1]).is_err());
    }

    // ── AdamW ────────────────────────────────────────────────────────────────

    #[test]
    fn test_adamw_decoupled_wd() {
        let mut opt = AdamW {
            weight_decay: 0.1,
            ..AdamW::new(0.001)
        };
        let mut p = vec![1.0];
        let p_before = p[0];
        opt.step(&mut p, &[0.0]).expect("step failed");
        // Decoupled WD: p ← p * (1 - lr * wd) then Adam on grad=0
        assert!(p[0] < p_before, "decoupled WD should shrink param");
    }

    #[test]
    fn test_adamw_converges() {
        let mut opt = AdamW {
            weight_decay: 0.0,
            ..AdamW::new(0.01)
        };
        let mut p = vec![2.0, -2.0];
        for _ in 0..1000 {
            let g = quadratic_grad(&p);
            opt.step(&mut p, &g).expect("step failed");
        }
        assert!(p[0].abs() < 0.1, "adamw p[0]={}", p[0]);
    }

    #[test]
    fn test_adamw_reset() {
        let mut opt = AdamW::new(0.001);
        let mut p = vec![1.0];
        opt.step(&mut p, &[0.5]).expect("step failed");
        assert_eq!(opt.t, 1);
        opt.reset_state();
        assert_eq!(opt.t, 0);
        assert!(opt.m.is_empty());
    }

    // ── SVRG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_svrg_needs_snapshot() {
        let mut svrg = Svrg::new(0.01, 100, 10);
        let mut p = vec![1.0, 2.0];
        let sg = vec![0.1, 0.2];
        let sgi = vec![0.05, 0.1];
        // No snapshot: should error
        assert!(svrg.step(&mut p, &sg, &sgi).is_err());
    }

    #[test]
    fn test_svrg_step_after_snapshot() {
        let mut svrg = Svrg::new(0.01, 100, 10);
        let mut p = vec![1.0, 1.0];
        let full_grad = vec![2.0, 2.0]; // grad at snapshot
        svrg.update_snapshot(&p, &full_grad);

        let sg = vec![2.1, 1.9];
        let sgi = vec![2.0, 2.0];
        svrg.step(&mut p, &sg, &sgi).expect("step failed");
        // Effective grad = sg - sgi + full_grad = [2.1-2+2, 1.9-2+2] = [2.1, 1.9]
        // p[0] ← 1 - 0.01*2.1 = 0.979
        assert_abs_diff_eq!(p[0], 1.0 - 0.01 * 2.1, epsilon = 1e-12);
    }

    #[test]
    fn test_svrg_update_freq() {
        let mut svrg = Svrg::new(0.01, 100, 3);
        let mut p = vec![1.0];
        svrg.update_snapshot(&p, &[0.0]);
        assert!(!svrg.needs_snapshot_update());

        for _ in 0..3 {
            svrg.step(&mut p, &[0.0], &[0.0]).expect("step");
        }
        assert!(svrg.needs_snapshot_update());
    }

    #[test]
    fn test_svrg_snapshot_params() {
        let mut svrg = Svrg::new(0.01, 100, 10);
        let snap = vec![3.0, 4.0];
        svrg.update_snapshot(&snap, &[0.0, 0.0]);
        assert_eq!(svrg.snapshot_params(), &[3.0, 4.0]);
    }

    #[test]
    fn test_svrg_length_mismatch() {
        let mut svrg = Svrg::new(0.01, 100, 10);
        let mut p = vec![1.0, 2.0];
        svrg.update_snapshot(&p, &[0.0, 0.0]);
        // Wrong length stochastic grad
        assert!(svrg.step(&mut p, &[0.1], &[0.0, 0.0]).is_err());
    }
}
