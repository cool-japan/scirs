//! Stateful Variance Reduction Optimizers
//!
//! Provides struct-based, stateful wrappers around the SVRG, SARAH, and SPIDER
//! algorithms. Unlike the functional forms in `variance_reduction.rs`, these
//! types hold their snapshot/auxiliary state internally and operate directly on
//! `Vec<f64>` parameter vectors via a user-supplied gradient closure.
//!
//! # Motivation
//! Standard mini-batch SGD has O(1/√T) convergence on strongly-convex
//! problems. Variance-reduced methods achieve *linear* convergence O((1-μ/L)ᵀ)
//! by periodically correcting for mini-batch noise with a reference gradient.
//!
//! # Algorithms
//!
//! | Type | Reference | Gradient cost |
//! |------|-----------|---------------|
//! | `SvrgOptimizer` | Johnson & Zhang (2013) | n + 2·inner |
//! | `SarahOptimizer` | Nguyen et al. (2017) | n + inner |
//! | `SpiderOptimizer` | Fang et al. (2018) | n + batch·inner |

use crate::error::OptimizeError;

// ─── SVRG ─────────────────────────────────────────────────────────────────────

/// SVRG — Stochastic Variance Reduced Gradient.
///
/// At the start of each epoch the optimizer computes the full gradient at a
/// "snapshot" θ̃, and uses it as a control variate during inner-loop SGD steps:
///
/// ```text
/// snapshot gradient:  μ̃ = ∇f(θ̃)   [full batch once per epoch]
/// inner update:       g_s = ∇f_s(θ) − ∇f_s(θ̃) + μ̃
///                     θ   = θ − α·g_s
/// ```
///
/// Reference: Johnson & Zhang (2013).
#[derive(Debug, Clone)]
pub struct SvrgOptimizer {
    /// Step size α
    pub lr: f64,
    /// Number of inner SGD steps per epoch
    pub inner_iters: usize,
    /// Current snapshot parameter vector
    snapshot: Vec<f64>,
    /// Full gradient at the snapshot
    full_grad: Vec<f64>,
    /// Inner step counter
    inner_step: usize,
}

impl SvrgOptimizer {
    /// Create a new SVRG optimizer.
    ///
    /// # Arguments
    /// * `lr` - Step size (learning rate)
    /// * `inner_iters` - Number of inner SGD iterations per epoch
    pub fn new(lr: f64, inner_iters: usize) -> Self {
        Self {
            lr,
            inner_iters,
            snapshot: Vec::new(),
            full_grad: Vec::new(),
            inner_step: 0,
        }
    }

    /// Update the snapshot and recompute the full gradient.
    ///
    /// Call this at the start of each epoch with the current parameters and
    /// the *full-batch* gradient.
    ///
    /// # Arguments
    /// * `params` - Current parameter vector
    /// * `full_gradient` - Full-batch gradient at `params`
    pub fn update_snapshot(&mut self, params: &[f64], full_gradient: Vec<f64>) {
        self.snapshot = params.to_vec();
        self.full_grad = full_gradient;
        self.inner_step = 0;
    }

    /// Perform one inner SVRG update step.
    ///
    /// # Arguments
    /// * `params` - Mutable parameter vector; updated in-place
    /// * `stoch_grad_current` - Mini-batch gradient at `params`
    /// * `stoch_grad_snapshot` - Mini-batch gradient at the *same* mini-batch
    ///   evaluated at the current snapshot θ̃
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` if any vector lengths differ.
    pub fn step(
        &mut self,
        params: &mut Vec<f64>,
        stoch_grad_current: &[f64],
        stoch_grad_snapshot: &[f64],
    ) -> Result<(), OptimizeError> {
        let n = params.len();
        if stoch_grad_current.len() != n || stoch_grad_snapshot.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "length mismatch: params={}, current_grad={}, snapshot_grad={}",
                n,
                stoch_grad_current.len(),
                stoch_grad_snapshot.len()
            )));
        }
        if self.full_grad.len() != n {
            return Err(OptimizeError::ValueError(
                "Snapshot not initialised; call update_snapshot first".to_string(),
            ));
        }

        for i in 0..n {
            // SVRG control variate: g = ∇f_s(θ) − ∇f_s(θ̃) + μ̃
            let g = stoch_grad_current[i] - stoch_grad_snapshot[i] + self.full_grad[i];
            params[i] -= self.lr * g;
        }
        self.inner_step += 1;
        Ok(())
    }

    /// Returns `true` if the epoch's inner loop is complete.
    pub fn epoch_done(&self) -> bool {
        self.inner_step >= self.inner_iters
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.snapshot.clear();
        self.full_grad.clear();
        self.inner_step = 0;
    }

    /// Reference to the current snapshot.
    pub fn snapshot(&self) -> &[f64] {
        &self.snapshot
    }

    /// Convenience: run a complete epoch using a gradient closure.
    ///
    /// The closure receives `(params, snapshot)` and returns
    /// `(stoch_grad_current, stoch_grad_snapshot)`.
    pub fn run_epoch<F>(
        &mut self,
        params: &mut Vec<f64>,
        full_gradient: Vec<f64>,
        mut grad_fn: F,
    ) -> Result<(), OptimizeError>
    where
        F: FnMut(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
    {
        self.update_snapshot(params, full_gradient);
        let snapshot_copy = self.snapshot.clone();
        for _ in 0..self.inner_iters {
            let (gc, gs) = grad_fn(params, &snapshot_copy);
            self.step(params, &gc, &gs)?;
        }
        Ok(())
    }
}

// ─── SARAH ────────────────────────────────────────────────────────────────────

/// SARAH — Stochastic Recursive gradient Algorithm with Historical gradient.
///
/// Unlike SVRG, SARAH uses a recursively-updated gradient estimate instead of
/// a snapshot + full-gradient correction. The estimator has lower variance but
/// is biased; in practice it achieves similar linear convergence.
///
/// ```text
/// at start:  v₀ = ∇f(θ₀)   [full gradient]
/// inner:     v_t = ∇f_s(θ_t) − ∇f_s(θ_{t-1}) + v_{t-1}
///            θ_{t+1} = θ_t − α·v_t
/// ```
///
/// Reference: Nguyen et al. (2017). "SARAH: A Novel Method for Machine Learning
/// Problems Using Stochastic Recursive Gradient". *ICML*.
#[derive(Debug, Clone)]
pub struct SarahOptimizer {
    /// Step size α
    pub lr: f64,
    /// Number of inner iterations before snapshot refresh
    pub inner_iters: usize,
    /// Current recursive gradient estimate v_t
    v: Vec<f64>,
    /// Previous parameter snapshot θ_{t-1}
    prev_params: Vec<f64>,
    /// Inner step counter
    inner_step: usize,
}

impl SarahOptimizer {
    /// Create a new SARAH optimizer.
    ///
    /// # Arguments
    /// * `lr` - Step size
    /// * `inner_iters` - Inner iterations per outer epoch
    pub fn new(lr: f64, inner_iters: usize) -> Self {
        Self {
            lr,
            inner_iters,
            v: Vec::new(),
            prev_params: Vec::new(),
            inner_step: 0,
        }
    }

    /// Initialise (or refresh) the recursive gradient estimate.
    ///
    /// Call at the start of each outer epoch with the full-batch gradient.
    pub fn init_epoch(&mut self, params: &[f64], full_gradient: Vec<f64>) {
        self.prev_params = params.to_vec();
        self.v = full_gradient;
        self.inner_step = 0;
    }

    /// Perform one SARAH inner step.
    ///
    /// # Arguments
    /// * `params` - Current parameters; updated in-place
    /// * `grad_current` - Mini-batch gradient at `params`
    /// * `grad_prev` - Mini-batch gradient at `prev_params` (same mini-batch)
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` on length mismatch or uninitialised state.
    pub fn step(
        &mut self,
        params: &mut Vec<f64>,
        grad_current: &[f64],
        grad_prev: &[f64],
    ) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad_current.len() != n || grad_prev.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "length mismatch: params={}, grad_current={}, grad_prev={}",
                n,
                grad_current.len(),
                grad_prev.len()
            )));
        }
        if self.v.len() != n {
            return Err(OptimizeError::ValueError(
                "SARAH not initialised; call init_epoch first".to_string(),
            ));
        }

        // Save the pre-update parameters as the new snapshot BEFORE modifying params.
        // If we saved AFTER the update, the next outer-loop call to `prev_params()`
        // would return params_{k+1} instead of params_k, making the stochastic gradient
        // terms equal (grad_current ≡ grad_prev) and zeroing out the recursive correction.
        let old_params = params.clone();
        for i in 0..n {
            // Recursive gradient update: v_{k+1} = ∇f_s(θ_{k+1}) − ∇f_s(θ_k) + v_k
            self.v[i] = grad_current[i] - grad_prev[i] + self.v[i];
            params[i] -= self.lr * self.v[i];
        }
        self.prev_params = old_params;
        self.inner_step += 1;
        Ok(())
    }

    /// Whether the inner loop is done.
    pub fn epoch_done(&self) -> bool {
        self.inner_step >= self.inner_iters
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.v.clear();
        self.prev_params.clear();
        self.inner_step = 0;
    }

    /// Reference to previous parameters.
    pub fn prev_params(&self) -> &[f64] {
        &self.prev_params
    }

    /// Convenience: run a complete epoch using a gradient closure.
    ///
    /// The closure receives `(params, prev_params)` and returns
    /// `(grad_current, grad_prev)`.
    pub fn run_epoch<F>(
        &mut self,
        params: &mut Vec<f64>,
        full_gradient: Vec<f64>,
        mut grad_fn: F,
    ) -> Result<(), OptimizeError>
    where
        F: FnMut(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
    {
        self.init_epoch(params, full_gradient);
        for _ in 0..self.inner_iters {
            let prev = self.prev_params.clone();
            let (gc, gp) = grad_fn(params, &prev);
            self.step(params, &gc, &gp)?;
        }
        Ok(())
    }
}

// ─── SPIDER ───────────────────────────────────────────────────────────────────

/// SPIDER — Stochastic Path-Integrated Differential Estimator.
///
/// Similar to SARAH but with a mini-batch size schedule and provable
/// near-optimal complexity for non-convex objectives.
///
/// ```text
/// every q steps:  v = (1/n)·Σ ∇f_i(θ)   [large batch refresh]
/// otherwise:      v = ∇f_B(θ_t) − ∇f_B(θ_{t-1}) + v_{t-1}   [small batch recursive]
/// θ_{t+1} = θ_t − α·v_t
/// ```
///
/// Reference: Fang et al. (2018). "SPIDER: Near-Optimal Non-Convex Optimization
/// via Stochastic Path-Integrated Differential Estimator". *NeurIPS*.
#[derive(Debug, Clone)]
pub struct SpiderOptimizer {
    /// Step size α
    pub lr: f64,
    /// Mini-batch size for recursive updates
    pub batch_size: usize,
    /// Number of inner iterations before a full-gradient refresh
    pub inner_iters: usize,
    /// Current gradient estimate v
    v: Vec<f64>,
    /// Previous parameter vector θ_{t-1}
    prev_params: Vec<f64>,
    /// Inner step counter
    inner_step: usize,
}

impl SpiderOptimizer {
    /// Create a new SPIDER optimizer.
    ///
    /// # Arguments
    /// * `lr` - Step size
    /// * `batch_size` - Mini-batch size for recursive updates
    /// * `inner_iters` - Recursive inner steps before next large-batch refresh
    pub fn new(lr: f64, batch_size: usize, inner_iters: usize) -> Self {
        Self {
            lr,
            batch_size,
            inner_iters,
            v: Vec::new(),
            prev_params: Vec::new(),
            inner_step: 0,
        }
    }

    /// Refresh the gradient estimate using a large/full batch.
    ///
    /// Call at the start of each outer iteration or whenever `refresh_needed()` returns `true`.
    pub fn refresh(&mut self, params: &[f64], large_batch_grad: Vec<f64>) {
        self.prev_params = params.to_vec();
        self.v = large_batch_grad;
        self.inner_step = 0;
    }

    /// Whether a large-batch refresh is due.
    pub fn refresh_needed(&self) -> bool {
        self.inner_step >= self.inner_iters || self.v.is_empty()
    }

    /// Perform one SPIDER recursive inner step.
    ///
    /// # Arguments
    /// * `params` - Current parameters; updated in-place
    /// * `grad_current` - Mini-batch gradient at `params`
    /// * `grad_prev` - Same mini-batch gradient at `prev_params`
    ///
    /// # Errors
    /// Returns `OptimizeError::ValueError` on length mismatch.
    pub fn step(
        &mut self,
        params: &mut Vec<f64>,
        grad_current: &[f64],
        grad_prev: &[f64],
    ) -> Result<(), OptimizeError> {
        let n = params.len();
        if grad_current.len() != n || grad_prev.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "length mismatch: params={}, grad_current={}, grad_prev={}",
                n,
                grad_current.len(),
                grad_prev.len()
            )));
        }
        if self.v.len() != n {
            return Err(OptimizeError::ValueError(
                "SPIDER not initialised; call refresh first".to_string(),
            ));
        }

        for i in 0..n {
            self.v[i] = grad_current[i] - grad_prev[i] + self.v[i];
            params[i] -= self.lr * self.v[i];
        }
        self.prev_params = params.clone();
        self.inner_step += 1;
        Ok(())
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.v.clear();
        self.prev_params.clear();
        self.inner_step = 0;
    }

    /// Reference to previous parameters.
    pub fn prev_params(&self) -> &[f64] {
        &self.prev_params
    }

    /// Convenience: run one outer iteration (refresh + inner loop) using closures.
    ///
    /// `large_grad_fn` computes a full/large-batch gradient at a parameter vector.
    /// `mini_grad_fn` computes `(grad_current, grad_prev)` for a mini-batch.
    pub fn run_outer_iter<F, G>(
        &mut self,
        params: &mut Vec<f64>,
        mut large_grad_fn: F,
        mut mini_grad_fn: G,
    ) -> Result<(), OptimizeError>
    where
        F: FnMut(&[f64]) -> Vec<f64>,
        G: FnMut(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
    {
        let large_grad = large_grad_fn(params);
        self.refresh(params, large_grad);
        for _ in 0..self.inner_iters {
            let prev = self.prev_params.clone();
            let (gc, gp) = mini_grad_fn(params, &prev);
            self.step(params, &gc, &gp)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn full_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| 2.0 * xi).collect()
    }

    #[test]
    fn test_svrg_converges() {
        let mut opt = SvrgOptimizer::new(0.05, 20);
        let mut params = vec![2.0, -1.5];

        for _epoch in 0..50 {
            let fg = full_grad(&params);
            opt.run_epoch(&mut params, fg, |p, snap| (full_grad(p), full_grad(snap)))
                .expect("epoch failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_svrg_epoch_done() {
        let mut opt = SvrgOptimizer::new(0.1, 5);
        let params = vec![1.0, 1.0];
        let fg = full_grad(&params);
        opt.update_snapshot(&params, fg);
        assert!(!opt.epoch_done());
        for _ in 0..5 {
            let snap = opt.snapshot().to_vec();
            let mut p = params.clone();
            let fg = full_grad(&p);
            let sg = full_grad(&snap);
            opt.step(&mut p, &fg, &sg).expect("step failed");
        }
        assert!(opt.epoch_done());
    }

    #[test]
    fn test_sarah_converges() {
        let mut opt = SarahOptimizer::new(0.05, 20);
        let mut params = vec![3.0, -1.0];

        for _epoch in 0..50 {
            let fg = full_grad(&params);
            opt.run_epoch(&mut params, fg, |p, prev| (full_grad(p), full_grad(prev)))
                .expect("epoch failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_spider_converges() {
        let mut opt = SpiderOptimizer::new(0.05, 10, 10);
        let mut params = vec![2.0, -2.0];

        for _outer in 0..50 {
            opt.run_outer_iter(
                &mut params,
                |p| full_grad(p),
                |p, prev| (full_grad(p), full_grad(prev)),
            )
            .expect("outer iter failed");
        }
        for &p in &params {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_svrg_length_mismatch() {
        let mut opt = SvrgOptimizer::new(0.1, 5);
        let params = vec![1.0, 2.0];
        opt.update_snapshot(&params, vec![0.0, 0.0]);
        let mut p = params.clone();
        // wrong length gradient
        let result = opt.step(&mut p, &[0.1], &[0.1, 0.2]);
        assert!(result.is_err());
    }
}
