//! Pure-ndarray differentiable projection head for SimCSE.
//!
//! Implements a two-layer MLP (d_in → d_hidden → d_out, Tanh activations)
//! with **analytic SGD backpropagation** using `ndarray` arithmetic.
//!
//! # Architecture
//!
//! ```text
//! input (d_in)  → Linear + Tanh → hidden (d_hidden) → Linear + Tanh → output (d_out)
//! ```
//!
//! Two forward passes of the **same input** through the projection (with
//! independently sampled Bernoulli dropout masks) produce `(h_i, h_i⁺)` — the
//! SimCSE unsupervised positive pair — without requiring labelled data.
//!
//! # Why not `scirs2-autograd`?
//!
//! `scirs2-autograd`'s current gradient backend evaluates gradient tensors via
//! `.eval(ctx)` *without a placeholder feeder*, so any gradient op that
//! transitively depends on an input placeholder silently returns `None` instead
//! of propagating the gradient.  Additionally, `Variable::compute` contains an
//! `unreachable!()` that fires during optimiser updates (see the crate's own
//! `simple_neural_network.rs` comment: *"There's an `unreachable!()` in the
//! Variable::compute method that gets called during optimization"*).
//!
//! This module therefore implements the projection head and its SGD update in
//! pure ndarray.  The public API is identical to what an autograd-backed
//! implementation would expose, so callers need no changes when the upstream
//! limitation is resolved.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::TextError;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Error type for projection operations.
pub type ProjResult<T> = Result<T, TextError>;

// ── ProjectionConfig ──────────────────────────────────────────────────────────

/// Configuration for the differentiable projection head.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Input dimensionality (must match the frozen encoder output dimension).
    pub d_in: usize,
    /// Hidden layer dimensionality.
    pub d_hidden: usize,
    /// Output / projection dimensionality.
    pub d_out: usize,
    /// Dropout probability applied during training (SimCSE trick).
    pub dropout_rate: f32,
    /// SGD learning rate.
    pub learning_rate: f32,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        ProjectionConfig {
            d_in: 768,
            d_hidden: 768,
            d_out: 768,
            dropout_rate: 0.1,
            learning_rate: 1e-4,
        }
    }
}

// ── DifferentiableProjection ──────────────────────────────────────────────────

/// Two-layer MLP projection head with analytic SGD weight updates.
///
/// Weights are `Array2<f32>` tensors initialised with Glorot uniform.  The
/// forward pass applies two linear + Tanh layers with Bernoulli dropout (for
/// training).  The backward pass computes exact gradients analytically and
/// applies a vanilla SGD update.
///
/// # InfoNCE loss
///
/// Given two augmented views `ha, hb ∈ ℝ^{n×d}`, the InfoNCE loss is:
///
/// ```text
/// L = -mean_i [ sim(ha_i, hb_i)/τ - logsumexp_j(sim(ha_i, hb_j)/τ) ]
///   = mean_i sparse_xent(logits_i, label=i)
/// ```
///
/// where `logits = (ha @ hb.T) / τ` (shape `n×n`).
///
/// # Gradient derivation
///
/// `d(L)/d(logits_{ij}) = (softmax(logits)_{ij} - delta_{ij}) / n`
///
/// Gradients then flow backwards through the tanh layers via:
/// - `d(tanh(z))/dz = 1 - tanh²(z)` (Jacobian is pointwise)
/// - linear layer: `dL/dW = X^T dL/dZ`, `dL/dX = dL/dZ W^T`
pub struct DifferentiableProjection {
    config: ProjectionConfig,
    /// Layer-1 weight: `[d_in, d_hidden]`.
    w1: Array2<f32>,
    /// Layer-1 bias:   `[1, d_hidden]`.
    b1: Array2<f32>,
    /// Layer-2 weight: `[d_hidden, d_out]`.
    w2: Array2<f32>,
    /// Layer-2 bias:   `[1, d_out]`.
    b2: Array2<f32>,
    /// LCG state for deterministic dropout.
    rng_state: u64,
    /// Number of SGD steps completed.
    steps: u64,
}

impl DifferentiableProjection {
    /// Construct a new projection head with Glorot-uniform weight initialisation.
    pub fn new(config: ProjectionConfig) -> Self {
        let d_in = config.d_in;
        let d_h = config.d_hidden;
        let d_out = config.d_out;

        let w1 = glorot_uniform(d_in, d_h, 42);
        let b1 = Array2::zeros((1, d_h));
        let w2 = glorot_uniform(d_h, d_out, 137);
        let b2 = Array2::zeros((1, d_out));

        DifferentiableProjection {
            config,
            w1,
            b1,
            w2,
            b2,
            rng_state: 0xDEAD_BEEF_1234_5678,
            steps: 0,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Perform a single SGD training step.
    ///
    /// Two dropout-augmented forward passes produce positive pairs `(ha, hb)`.
    /// The InfoNCE gradient is backpropagated analytically through both MLP paths.
    ///
    /// Returns the InfoNCE loss value.
    pub fn update_step(&mut self, embeddings: &Array2<f32>, temperature: f32) -> ProjResult<f32> {
        let n = embeddings.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        if embeddings.ncols() != self.config.d_in {
            return Err(TextError::InvalidInput(format!(
                "Expected d_in={}, got {}",
                self.config.d_in,
                embeddings.ncols()
            )));
        }

        // Forward pass A (with dropout)
        let (ha, cache_a) = self.forward_train(embeddings);
        // Forward pass B (with different dropout mask)
        let (hb, cache_b) = self.forward_train(embeddings);

        // InfoNCE loss + gradient w.r.t. logits
        let (loss, d_logits) = infonce_loss_and_grad(&ha, &hb, temperature);

        // Backprop: d_logits is [n, n]
        // logits = ha @ hb.T  →  d_ha = d_logits @ hb / τ,  d_hb = d_logits.T @ ha / τ
        let inv_tau = 1.0_f32 / temperature;
        let d_ha = d_logits.dot(&hb) * inv_tau; // [n, d_out]
        let d_hb = d_logits.t().dot(&ha) * inv_tau; // [n, d_out]

        // Accumulate gradients from both paths
        let (dw1_a, db1_a, dw2_a, db2_a) = self.backward(&d_ha, &cache_a, embeddings);
        let (dw1_b, db1_b, dw2_b, db2_b) = self.backward(&d_hb, &cache_b, embeddings);

        let lr = self.config.learning_rate;
        let inv_two = 0.5_f32; // average over two paths

        // SGD update: θ ← θ - lr * (grad_a + grad_b) / 2
        self.w1 = &self.w1 - &((&dw1_a + &dw1_b) * (lr * inv_two));
        self.b1 = &self.b1 - &((&db1_a + &db1_b) * (lr * inv_two));
        self.w2 = &self.w2 - &((&dw2_a + &dw2_b) * (lr * inv_two));
        self.b2 = &self.b2 - &((&db2_a + &db2_b) * (lr * inv_two));

        self.steps += 1;
        Ok(loss)
    }

    /// Run a forward pass in inference mode (no dropout).
    ///
    /// Returns `(batch_size × d_out)`.
    pub fn forward_inference(&self, embeddings: &Array2<f32>) -> ProjResult<Array2<f32>> {
        if embeddings.ncols() != self.config.d_in {
            return Err(TextError::InvalidInput(format!(
                "Expected d_in={}, got {}",
                self.config.d_in,
                embeddings.ncols()
            )));
        }

        // Layer 1: Z1 = X @ W1 + b1,  H1 = tanh(Z1)
        let z1 = embeddings.dot(&self.w1) + &self.b1;
        let h1 = z1.mapv(f32::tanh);

        // Layer 2: Z2 = H1 @ W2 + b2,  H2 = tanh(Z2)
        let z2 = h1.dot(&self.w2) + &self.b2;
        let output = z2.mapv(f32::tanh);

        Ok(output)
    }

    /// Return the number of completed SGD steps.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Return the projection configuration.
    pub fn config(&self) -> &ProjectionConfig {
        &self.config
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Forward pass with Bernoulli dropout.  Returns (output, cache).
    ///
    /// Cache = (h1, z1, mask1, z2, mask2) for use in backward.
    fn forward_train(&mut self, x: &Array2<f32>) -> (Array2<f32>, ForwardCache) {
        let rate = self.config.dropout_rate;
        let scale = if rate < 1.0 { 1.0 / (1.0 - rate) } else { 1.0 };

        // Layer 1
        let z1 = x.dot(&self.w1) + &self.b1;
        let mask1 = self.bernoulli_mask(z1.nrows(), z1.ncols(), rate, scale);
        let h1 = z1.mapv(f32::tanh) * &mask1;

        // Layer 2
        let z2 = h1.dot(&self.w2) + &self.b2;
        let mask2 = self.bernoulli_mask(z2.nrows(), z2.ncols(), rate, scale);
        let output = z2.mapv(f32::tanh) * &mask2;

        let cache = ForwardCache {
            h1,
            z1,
            mask1,
            z2,
            mask2,
        };
        (output, cache)
    }

    /// Analytic backward pass through the two-layer MLP.
    ///
    /// `d_out` — gradient of the loss w.r.t. `forward_train` output, shape `[n, d_out]`.
    /// Returns `(dW1, db1, dW2, db2)`.
    fn backward(
        &self,
        d_out: &Array2<f32>,
        cache: &ForwardCache,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        // Layer 2 backward
        // d_output_pre_mask = d_out * mask2
        let d_h2 = d_out * &cache.mask2;
        // d_tanh(z2) = d_h2 * (1 - tanh²(z2))
        let tanh_z2 = cache.z2.mapv(f32::tanh);
        let d_z2 = &d_h2 * &(1.0 - &tanh_z2.mapv(|v| v * v));

        // dW2 = h1.T @ d_z2   shape [d_hidden, d_out]
        let dw2 = cache.h1.t().dot(&d_z2);
        // db2 = sum(d_z2, axis=0, keepdims)  shape [1, d_out]
        let db2 = d_z2.sum_axis(Axis(0)).insert_axis(Axis(0));
        // d_h1 = d_z2 @ W2.T   shape [n, d_hidden]
        let d_h1_raw = d_z2.dot(&self.w2.t());

        // Layer 1 backward
        // d_h1_pre_mask = d_h1_raw * mask1
        let d_h1 = d_h1_raw * &cache.mask1;
        // d_tanh(z1) = d_h1 * (1 - tanh²(z1))
        let tanh_z1 = cache.z1.mapv(f32::tanh);
        let d_z1 = &d_h1 * &(1.0 - &tanh_z1.mapv(|v| v * v));

        // dW1 = x.T @ d_z1   shape [d_in, d_hidden]
        let dw1 = x.t().dot(&d_z1);
        // db1 = sum(d_z1, axis=0, keepdims)  shape [1, d_hidden]
        let db1 = d_z1.sum_axis(Axis(0)).insert_axis(Axis(0));

        (dw1, db1, dw2, db2)
    }

    /// Sample a Bernoulli dropout mask using LCG randomness.
    fn bernoulli_mask(&mut self, rows: usize, cols: usize, rate: f32, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (self.rng_state >> 33) as f32 / u32::MAX as f32;
            if u < rate {
                0.0
            } else {
                scale
            }
        })
    }
}

// ── ForwardCache ──────────────────────────────────────────────────────────────

/// Intermediate activations cached during a training forward pass.
struct ForwardCache {
    /// Pre-dropout Tanh output of layer 1: `tanh(Z1) * mask1`.
    h1: Array2<f32>,
    /// Pre-activation of layer 1: `X @ W1 + b1`.
    z1: Array2<f32>,
    /// Dropout mask applied after layer 1 tanh.
    mask1: Array2<f32>,
    /// Pre-activation of layer 2: `H1 @ W2 + b2`.
    z2: Array2<f32>,
    /// Dropout mask applied after layer 2 tanh.
    mask2: Array2<f32>,
}

// ── InfoNCE ───────────────────────────────────────────────────────────────────

/// Compute InfoNCE loss and its gradient w.r.t. logits.
///
/// Given `ha, hb ∈ ℝ^{n×d}`, computes `logits = ha @ hb.T` (unnormalised by τ
/// since the caller scales gradients by `1/τ`).
///
/// Returns `(loss, d_logits)` where:
/// - `loss` is the mean sparse cross-entropy.
/// - `d_logits = (softmax(logits/τ) - I_onehot) / n`.
fn infonce_loss_and_grad(
    ha: &Array2<f32>,
    hb: &Array2<f32>,
    temperature: f32,
) -> (f32, Array2<f32>) {
    let n = ha.nrows();
    let inv_n = 1.0_f32 / n as f32;
    let inv_tau = 1.0_f32 / temperature;

    // Logits: [n, n]  (scaled by 1/τ for softmax)
    let sim = ha.dot(&hb.t()); // [n, n]
    let logits = sim.mapv(|v| v * inv_tau);

    // Numerically stable softmax per row.
    let mut softmax = Array2::<f32>::zeros((n, n));
    let mut total_loss = 0.0_f32;

    for i in 0..n {
        let row = logits.row(i);
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let log_sum_exp = sum_exp.ln() + max_val;

        // Cross-entropy: -logit[i, i] + log_sum_exp[i]
        let pos_logit = logits[[i, i]];
        total_loss += -(pos_logit - log_sum_exp);

        for j in 0..n {
            softmax[[i, j]] = exps[j] / sum_exp.max(1e-30);
        }
    }

    let loss = total_loss * inv_n;

    // Gradient: d_logits[i, j] = (softmax[i, j] - delta_{ij}) / n
    let mut d_logits = softmax;
    for i in 0..n {
        d_logits[[i, i]] -= 1.0;
    }
    d_logits.mapv_inplace(|v| v * inv_n);

    (loss, d_logits)
}

// ── Weight initialisation ─────────────────────────────────────────────────────

/// Glorot uniform initialiser.  Weights drawn from `U[-limit, limit]` where
/// `limit = sqrt(6 / (fan_in + fan_out))`.
fn glorot_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Array2<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    let mut state = seed;
    Array2::from_shape_fn((fan_in, fan_out), |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (state >> 12) as f64 / (1u64 << 52) as f64;
        (u as f32 * 2.0 - 1.0) * limit
    })
}

impl std::fmt::Debug for DifferentiableProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DifferentiableProjection")
            .field("d_in", &self.config.d_in)
            .field("d_hidden", &self.config.d_hidden)
            .field("d_out", &self.config.d_out)
            .field("steps", &self.steps)
            .finish()
    }
}
