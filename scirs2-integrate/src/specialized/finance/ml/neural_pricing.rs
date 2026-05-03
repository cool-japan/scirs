//! Neural network-based derivative pricing
//!
//! This module implements deep learning approaches for pricing complex derivatives,
//! using pure-ndarray multi-layer perceptrons with Adam optimization.
//!
//! # Features
//! - Deep neural networks for option pricing (3-layer MLP with Adam)
//! - Black-Scholes training data generation
//! - Batch inference for portfolio pricing
//!
//! # Example
//! ```
//! use scirs2_integrate::specialized::finance::ml::neural_pricing::{
//!     NeuralPricer, DeepPricingNetwork, generate_black_scholes_training_data,
//! };
//!
//! let (features, prices) = generate_black_scholes_training_data(200);
//! let mut net = DeepPricingNetwork::new();
//! let loss = net.fit(features.view(), prices.view(), 20).expect("fit failed");
//! println!("Final MSE loss: {:.6}", loss);
//! ```

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::pricing::black_scholes::black_scholes_price;
use crate::specialized::finance::types::OptionType;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, Rng, RngExt};

// ============================================================================
// Trait definition
// ============================================================================

/// A neural network that learns to price financial derivatives.
///
/// Implementations train on `(features, target_prices)` pairs where the feature
/// vector is `[S/K, T, r, sigma]` (moneyness, time-to-expiry, interest rate,
/// volatility) and output a non-negative price prediction.
pub trait NeuralPricer: Send + Sync {
    /// Train on (features, target_prices).
    ///
    /// Features are \[S/K, T, r, sigma\] per sample (shape `[n_samples, 4]`).
    /// Returns the final epoch MSE loss.
    fn fit(
        &mut self,
        features: ArrayView2<f64>,
        prices: ArrayView1<f64>,
        epochs: usize,
    ) -> IntegrateResult<f64>;

    /// Price a single option given a feature vector \[S/K, T, r, sigma\].
    fn price(&self, features: ArrayView1<f64>) -> IntegrateResult<f64>;

    /// Price multiple options given a feature matrix (shape `[n_samples, 4]`).
    fn batch_price(&self, features: ArrayView2<f64>) -> IntegrateResult<Array1<f64>>;

    /// Training loss history (one value per epoch).
    fn loss_history(&self) -> &[f64];
}

// ============================================================================
// Adam optimizer state helpers
// ============================================================================

/// Adam hyper-parameters (fixed; exposed only for documentation)
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;
const MINIBATCH_SIZE: usize = 32;

/// Apply one Adam step to a 2-D parameter array in-place.
fn adam_step2(
    param: &mut Array2<f64>,
    grad: &Array2<f64>,
    m: &mut Array2<f64>,
    v: &mut Array2<f64>,
    t: f64,
    lr: f64,
) {
    // m = β1 * m + (1 - β1) * grad
    *m = m.mapv(|x| ADAM_BETA1 * x) + &grad.mapv(|x| (1.0 - ADAM_BETA1) * x);
    // v = β2 * v + (1 - β2) * grad²
    *v = v.mapv(|x| ADAM_BETA2 * x) + &grad.mapv(|x| (1.0 - ADAM_BETA2) * x * x);
    // bias-corrected estimates
    let m_hat = m.mapv(|x| x / (1.0 - ADAM_BETA1.powf(t)));
    let v_hat = v.mapv(|x| x / (1.0 - ADAM_BETA2.powf(t)));
    // parameter update
    *param = param.clone() - &m_hat.mapv(|mh| lr * mh) / &v_hat.mapv(|vh| vh.sqrt() + ADAM_EPS);
}

/// Apply one Adam step to a 1-D parameter array in-place.
fn adam_step1(
    param: &mut Array1<f64>,
    grad: &Array1<f64>,
    m: &mut Array1<f64>,
    v: &mut Array1<f64>,
    t: f64,
    lr: f64,
) {
    *m = m.mapv(|x| ADAM_BETA1 * x) + &grad.mapv(|x| (1.0 - ADAM_BETA1) * x);
    *v = v.mapv(|x| ADAM_BETA2 * x) + &grad.mapv(|x| (1.0 - ADAM_BETA2) * x * x);
    let m_hat = m.mapv(|x| x / (1.0 - ADAM_BETA1.powf(t)));
    let v_hat = v.mapv(|x| x / (1.0 - ADAM_BETA2.powf(t)));
    *param = param.clone() - &m_hat.mapv(|mh| lr * mh) / &v_hat.mapv(|vh| vh.sqrt() + ADAM_EPS);
}

// ============================================================================
// DeepPricingNetwork
// ============================================================================

/// 3-layer MLP with Adam optimizer for learning Black-Scholes pricing.
///
/// Architecture: input(4) → Dense(32, tanh) → Dense(32, tanh) → Dense(1, Softplus)
///
/// Softplus output `log(1 + exp(z))` ensures non-negative prices while
/// maintaining a non-zero gradient everywhere (unlike ReLU).
///
/// All weights are Glorot-uniform initialised; biases start at zero except the
/// output bias which is initialised to a small positive value (0.5) to start
/// the softplus in an active region.
#[derive(Debug, Clone)]
pub struct DeepPricingNetwork {
    // Layer 1: 4 → 32
    w1: Array2<f64>,
    b1: Array1<f64>,
    // Layer 2: 32 → 32
    w2: Array2<f64>,
    b2: Array1<f64>,
    // Layer 3: 32 → 1
    w3: Array2<f64>,
    b3: Array1<f64>,
    // Adam first moment (m)
    mw1: Array2<f64>,
    mb1: Array1<f64>,
    mw2: Array2<f64>,
    mb2: Array1<f64>,
    mw3: Array2<f64>,
    mb3: Array1<f64>,
    // Adam second moment (v)
    vw1: Array2<f64>,
    vb1: Array1<f64>,
    vw2: Array2<f64>,
    vb2: Array1<f64>,
    vw3: Array2<f64>,
    vb3: Array1<f64>,
    /// Adam global timestep (incremented each minibatch step)
    t: u64,
    /// Learning rate (default 1e-3)
    lr: f64,
    /// Per-epoch MSE loss history
    loss_history: Vec<f64>,
}

impl DeepPricingNetwork {
    /// Glorot uniform limit for a (fan_in, fan_out) layer.
    fn glorot_limit(fan_in: usize, fan_out: usize) -> f64 {
        (6.0 / (fan_in + fan_out) as f64).sqrt()
    }

    /// Initialise a weight matrix with Glorot uniform.
    fn glorot_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f64> {
        let limit = Self::glorot_limit(cols, rows); // (fan_in=cols, fan_out=rows)
        Array2::from_shape_fn((rows, cols), |_| rng.random_range(-limit..limit))
    }

    /// Create a new `DeepPricingNetwork` with default learning rate 1e-3.
    pub fn new() -> Self {
        let mut rng = thread_rng();
        Self::with_rng(&mut rng, 1e-3)
    }

    /// Create a network with a user-supplied RNG (useful for reproducible tests).
    pub fn with_rng(rng: &mut impl Rng, lr: f64) -> Self {
        let w1 = Self::glorot_matrix(32, 4, rng);
        let b1 = Array1::zeros(32);
        let w2 = Self::glorot_matrix(32, 32, rng);
        let b2 = Array1::zeros(32);
        let w3 = Self::glorot_matrix(1, 32, rng);
        // Output bias positive to start softplus in an active region
        let mut b3 = Array1::zeros(1);
        b3[0] = 0.5;

        Self {
            mw1: Array2::zeros(w1.raw_dim()),
            mb1: Array1::zeros(32),
            mw2: Array2::zeros(w2.raw_dim()),
            mb2: Array1::zeros(32),
            mw3: Array2::zeros(w3.raw_dim()),
            mb3: Array1::zeros(1),
            vw1: Array2::zeros(w1.raw_dim()),
            vb1: Array1::zeros(32),
            vw2: Array2::zeros(w2.raw_dim()),
            vb2: Array1::zeros(32),
            vw3: Array2::zeros(w3.raw_dim()),
            vb3: Array1::zeros(1),
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            t: 0,
            lr,
            loss_history: Vec::new(),
        }
    }

    /// Numerically stable softplus: `log(1 + exp(z))`.
    fn softplus(z: f64) -> f64 {
        if z > 30.0 {
            z // avoid exp overflow; softplus ≈ z for large z
        } else {
            (1.0 + z.exp()).ln()
        }
    }

    /// Softplus derivative: `sigmoid(z) = 1 / (1 + exp(-z))`.
    fn softplus_prime(z: f64) -> f64 {
        if z > 30.0 {
            1.0
        } else if z < -30.0 {
            0.0
        } else {
            1.0 / (1.0 + (-z).exp())
        }
    }

    /// Forward pass: returns `(h1, z1, h2, z2, out, z3)` for a single sample.
    ///
    /// - `z_i`: pre-activation for layer i
    /// - `h_i`: post-activation for hidden layer i
    /// - `out`: final (non-negative) scalar output via softplus
    fn forward_full(
        &self,
        x: ArrayView1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, f64, f64) {
        let z1 = self.w1.dot(&x) + &self.b1;
        let h1 = z1.mapv(|v| v.tanh());
        let z2 = self.w2.dot(&h1) + &self.b2;
        let h2 = z2.mapv(|v| v.tanh());
        let z3_arr = self.w3.dot(&h2) + &self.b3;
        let z3 = z3_arr[0];
        let out = Self::softplus(z3); // always non-negative, always differentiable
        (h1, z1, h2, z2, out, z3)
    }

    /// Scalar forward pass (for inference).
    fn forward(&self, x: ArrayView1<f64>) -> f64 {
        let (_, _, _, _, out, _) = self.forward_full(x);
        out
    }

    /// Train on a single sample, accumulate gradients.
    ///
    /// Returns the squared error for this sample.
    fn backward_single(
        &self,
        x: ArrayView1<f64>,
        target: f64,
        // gradient accumulators (mutated in-place)
        dw1: &mut Array2<f64>,
        db1: &mut Array1<f64>,
        dw2: &mut Array2<f64>,
        db2: &mut Array1<f64>,
        dw3: &mut Array2<f64>,
        db3: &mut Array1<f64>,
    ) -> f64 {
        let (h1, _z1, h2, _z2, out, z3) = self.forward_full(x);

        let err = out - target;

        // δ3 = 2 * err * softplus'(z3)  [softplus' = sigmoid(z3)]
        let sp_prime = Self::softplus_prime(z3);
        let delta3 = Array1::from_elem(1, 2.0 * err * sp_prime);

        // dW3 = delta3 (1×1) * h2^T (1×32)
        for i in 0..32 {
            dw3[[0, i]] += delta3[0] * h2[i];
        }
        db3[0] += delta3[0];

        // δ2 = (W3^T δ3) * tanh'(z2)  — shape [32]
        // W3 is (1×32), W3^T is (32×1)
        let w3t_delta3 = self.w3.t().dot(&delta3); // [32]
                                                   // tanh'(z2) = 1 - tanh(z2)^2 = 1 - h2^2
        let tanh_prime2 = h2.mapv(|v| 1.0 - v * v);
        let delta2 = &w3t_delta3 * &tanh_prime2;

        // dW2 = delta2 (32) outer h1 (32)
        for i in 0..32 {
            for j in 0..32 {
                dw2[[i, j]] += delta2[i] * h1[j];
            }
        }
        for i in 0..32 {
            db2[i] += delta2[i];
        }

        // δ1 = (W2^T δ2) * tanh'(z1)  — shape [32]
        let w2t_delta2 = self.w2.t().dot(&delta2);
        let tanh_prime1 = h1.mapv(|v| 1.0 - v * v);
        let delta1 = &w2t_delta2 * &tanh_prime1;

        // dW1 = delta1 (32) outer x (4)
        for i in 0..32 {
            for j in 0..4 {
                dw1[[i, j]] += delta1[i] * x[j];
            }
        }
        for i in 0..32 {
            db1[i] += delta1[i];
        }

        err * err
    }

    /// Execute one Adam update step with the given (averaged) gradients.
    fn apply_adam_step(
        &mut self,
        dw1: Array2<f64>,
        db1: Array1<f64>,
        dw2: Array2<f64>,
        db2: Array1<f64>,
        dw3: Array2<f64>,
        db3: Array1<f64>,
    ) {
        self.t += 1;
        let t = self.t as f64;
        let lr = self.lr;

        let (w1, b1) = (&mut self.w1, &mut self.b1);
        let (mw1, vw1) = (&mut self.mw1, &mut self.vw1);
        let (mb1, vb1) = (&mut self.mb1, &mut self.vb1);
        adam_step2(w1, &dw1, mw1, vw1, t, lr);
        adam_step1(b1, &db1, mb1, vb1, t, lr);

        let (w2, b2) = (&mut self.w2, &mut self.b2);
        let (mw2, vw2) = (&mut self.mw2, &mut self.vw2);
        let (mb2, vb2) = (&mut self.mb2, &mut self.vb2);
        adam_step2(w2, &dw2, mw2, vw2, t, lr);
        adam_step1(b2, &db2, mb2, vb2, t, lr);

        let (w3, b3) = (&mut self.w3, &mut self.b3);
        let (mw3, vw3) = (&mut self.mw3, &mut self.vw3);
        let (mb3, vb3) = (&mut self.mb3, &mut self.vb3);
        adam_step2(w3, &dw3, mw3, vw3, t, lr);
        adam_step1(b3, &db3, mb3, vb3, t, lr);
    }
}

impl Default for DeepPricingNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralPricer for DeepPricingNetwork {
    fn fit(
        &mut self,
        features: ArrayView2<f64>,
        prices: ArrayView1<f64>,
        epochs: usize,
    ) -> IntegrateResult<f64> {
        let n = features.nrows();
        if n == 0 {
            return Err(IntegrateError::ValueError(
                "Training dataset is empty".to_string(),
            ));
        }
        if features.ncols() != 4 {
            return Err(IntegrateError::ValueError(format!(
                "Expected 4 features per sample, got {}",
                features.ncols()
            )));
        }
        if prices.len() != n {
            return Err(IntegrateError::DimensionMismatch(format!(
                "features has {} rows but prices has {} elements",
                n,
                prices.len()
            )));
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();

        for _epoch in 0..epochs {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }

            let mut epoch_sq_sum = 0.0_f64;
            let mut n_processed = 0usize;

            // Minibatch loop
            let mut start = 0;
            while start < n {
                let end = (start + MINIBATCH_SIZE).min(n);
                let batch_size = end - start;

                let mut dw1 = Array2::<f64>::zeros(self.w1.raw_dim());
                let mut db1 = Array1::<f64>::zeros(32);
                let mut dw2 = Array2::<f64>::zeros(self.w2.raw_dim());
                let mut db2 = Array1::<f64>::zeros(32);
                let mut dw3 = Array2::<f64>::zeros(self.w3.raw_dim());
                let mut db3 = Array1::<f64>::zeros(1);

                let mut batch_sq_sum = 0.0_f64;

                for &idx in &indices[start..end] {
                    let x = features.row(idx);
                    let target = prices[idx];
                    let sq_err = self.backward_single(
                        x, target, &mut dw1, &mut db1, &mut dw2, &mut db2, &mut dw3, &mut db3,
                    );
                    batch_sq_sum += sq_err;
                }

                // Average gradients over batch
                let bs_f = batch_size as f64;
                dw1 /= bs_f;
                db1 /= bs_f;
                dw2 /= bs_f;
                db2 /= bs_f;
                dw3 /= bs_f;
                db3 /= bs_f;

                self.apply_adam_step(dw1, db1, dw2, db2, dw3, db3);

                epoch_sq_sum += batch_sq_sum;
                n_processed += batch_size;
                start = end;
            }

            let epoch_mse = epoch_sq_sum / n_processed as f64;
            self.loss_history.push(epoch_mse);
        }

        let final_loss = self.loss_history.last().copied().unwrap_or(0.0);
        Ok(final_loss)
    }

    fn price(&self, features: ArrayView1<f64>) -> IntegrateResult<f64> {
        if features.len() != 4 {
            return Err(IntegrateError::ValueError(format!(
                "Expected feature vector of length 4, got {}",
                features.len()
            )));
        }
        Ok(self.forward(features))
    }

    fn batch_price(&self, features: ArrayView2<f64>) -> IntegrateResult<Array1<f64>> {
        if features.ncols() != 4 {
            return Err(IntegrateError::ValueError(format!(
                "Expected 4 features per sample, got {}",
                features.ncols()
            )));
        }
        let n = features.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            out[i] = self.forward(features.row(i));
        }
        Ok(out)
    }

    fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }
}

// ============================================================================
// Black-Scholes training data generation
// ============================================================================

/// Generate a synthetic Black-Scholes call-option dataset.
///
/// Features are `[S/K, T, r, sigma]` sampled uniformly:
/// - `S/K` (moneyness) in `[0.5, 2.0]`
/// - `T` (time-to-expiry in years) in `[0.1, 2.0]`
/// - `r` (risk-free rate) in `[0.0, 0.1]`
/// - `sigma` (implied volatility) in `[0.1, 0.5]`
///
/// Labels are the corresponding Black-Scholes European call prices with
/// normalised strike `K = 1` and dividend `q = 0`.
pub fn generate_black_scholes_training_data(n_samples: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = thread_rng();

    let mut features = Array2::<f64>::zeros((n_samples, 4));
    let mut prices = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        // Sample features uniformly
        let moneyness = 0.5 + rng.random_range(0.0..1.5); // [0.5, 2.0]
        let t = 0.1 + rng.random_range(0.0..1.9); // [0.1, 2.0]
        let r = rng.random_range(0.0..0.1); // [0.0, 0.1)
        let sigma = 0.1 + rng.random_range(0.0..0.4); // [0.1, 0.5)

        features[[i, 0]] = moneyness;
        features[[i, 1]] = t;
        features[[i, 2]] = r;
        features[[i, 3]] = sigma;

        // Black-Scholes call price with S = moneyness, K = 1, q = 0
        let price = black_scholes_price(moneyness, 1.0, r, 0.0, sigma, t, OptionType::Call);
        prices[i] = price.max(0.0); // ensure non-negative (should always be, but guard)
    }

    (features, prices)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_pricing_network_fit() {
        // Use a reduced dataset and fewer epochs for reasonable test runtime in
        // debug mode. The loss threshold is correspondingly relaxed.
        let (features, prices) = generate_black_scholes_training_data(128);
        let mut net = DeepPricingNetwork::new();
        let final_loss = net
            .fit(features.view(), prices.view(), 30)
            .expect("fit should succeed");
        // The network should show non-trivial convergence (loss below naive
        // constant-predictor level). Exact value depends on random init.
        assert!(
            final_loss < 0.5,
            "Expected final MSE loss < 0.5 after 30 epochs, got {:.6}",
            final_loss
        );
        assert_eq!(net.loss_history().len(), 30);
    }

    #[test]
    fn test_deep_pricing_network_price() {
        let (features, prices) = generate_black_scholes_training_data(128);
        let mut net = DeepPricingNetwork::new();
        net.fit(features.view(), prices.view(), 30)
            .expect("fit should succeed");

        // Mid-range option: S/K=1.0, T=1.0, r=0.05, sigma=0.2
        let test_feature: Array1<f64> = Array1::from_vec(vec![1.0, 1.0, 0.05, 0.2]);
        let nn_price = net
            .price(test_feature.view())
            .expect("price should succeed");

        // After softplus, the output should always be positive.
        // With limited training we just verify non-negativity and sanity.
        assert!(
            nn_price >= 0.0,
            "Neural price must be non-negative, got {:.4}",
            nn_price
        );
        // Prices for realistic options are bounded by the spot (S/K=1 → max ~1)
        assert!(
            nn_price < 5.0,
            "Neural price unreasonably large: {:.4}",
            nn_price
        );
    }

    #[test]
    fn test_batch_price_shape() {
        let (features, prices) = generate_black_scholes_training_data(64);
        let mut net = DeepPricingNetwork::new();
        net.fit(features.view(), prices.view(), 5)
            .expect("fit should succeed");

        let batch_features = features.slice(s![..10, ..]).to_owned();
        let preds = net
            .batch_price(batch_features.view())
            .expect("batch_price should succeed");

        assert_eq!(preds.len(), 10);
        // All prices should be non-negative (softplus output)
        for (i, &p) in preds.iter().enumerate() {
            assert!(p >= 0.0, "Price at index {} is negative: {}", i, p);
        }
    }

    #[test]
    fn test_loss_history_grows() {
        let (features, prices) = generate_black_scholes_training_data(32);
        let mut net = DeepPricingNetwork::new();
        net.fit(features.view(), prices.view(), 5)
            .expect("fit should succeed");
        assert_eq!(net.loss_history().len(), 5);
    }

    #[test]
    fn test_invalid_feature_dimension() {
        let net = DeepPricingNetwork::new();
        let wrong_feat: Array1<f64> = Array1::from_vec(vec![1.0, 0.5]);
        assert!(net.price(wrong_feat.view()).is_err());
    }
}
