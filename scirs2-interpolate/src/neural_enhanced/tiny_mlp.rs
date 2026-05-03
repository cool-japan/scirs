//! Tiny multi-layer perceptron with analytic backpropagation.
//!
//! Pure-ndarray implementation (no external autograd) used by `ResidualMlpRbf`.
//! Supports `Tanh`, `Relu`, and `GeluApprox` activations; Glorot initialisation;
//! and mini-batch SGD with L2 regularisation.
//!
//! # Output-layer zero initialisation
//!
//! The **last layer weights and bias are initialised to zero** so that the MLP
//! outputs zero before any training.  This means that when `epochs == 0` the
//! residual MLP behaves exactly like the base RBF interpolant, satisfying the
//! test `residual_rbf_matches_pure_rbf_when_epochs_zero`.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

/// XorShift64 deterministic PRNG — same as `deep_kriging::mlp_kriging`.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 + 0.5) / (u64::MAX as f64 + 1.0)
    }

    /// Box-Muller Normal(0,1) sample.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

/// Activation function for `TinyMlp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// Hyperbolic tangent.
    Tanh,
    /// Rectified linear unit.
    Relu,
    /// Approximate GELU: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
    GeluApprox,
}

impl Activation {
    #[inline]
    pub(crate) fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => (x as f64).tanh() as f32,
            Activation::Relu => x.max(0.0),
            Activation::GeluApprox => {
                let xf = x as f64;
                let c = (2.0 / std::f64::consts::PI).sqrt();
                let inner = c * (xf + 0.044715 * xf * xf * xf);
                (0.5 * xf * (1.0 + inner.tanh())) as f32
            }
        }
    }

    /// Derivative of activation with respect to its pre-activation input.
    #[inline]
    pub(crate) fn backward(&self, pre: f32) -> f32 {
        match self {
            Activation::Tanh => {
                let t = (pre as f64).tanh() as f32;
                1.0 - t * t
            }
            Activation::Relu => {
                if pre > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::GeluApprox => {
                // Approximate derivative via finite difference — accurate enough for training.
                let h = 1e-4f32;
                let fp = self.forward(pre + h);
                let fm = self.forward(pre - h);
                (fp - fm) / (2.0 * h)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TinyMlp
// ---------------------------------------------------------------------------

/// Small trainable MLP.
///
/// All hidden layers use `activation`; the output layer is linear (no activation).
/// The output layer is initialised to zero weights + zero bias so that
/// `forward([any_x]) == 0` before training begins.
#[derive(Debug, Clone)]
pub struct TinyMlp {
    /// Weight matrices, one per layer.  `weights[l]` has shape `(out, in)`.
    pub weights: Vec<Array2<f32>>,
    /// Bias vectors, one per layer.  `biases[l]` has length `out`.
    pub biases: Vec<Array1<f32>>,
    /// Activation applied to every hidden layer.
    pub activation: Activation,
    /// Architecture: input dim + hidden sizes + output dim=1.
    pub layer_sizes: Vec<usize>,
}

impl TinyMlp {
    /// Create a new `TinyMlp` with Glorot-initialised hidden layers and
    /// **zero-initialised output layer**.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` — `[input_dim, hidden_1, ..., hidden_k, output_dim]`.
    ///   Must have at least 2 elements.
    /// * `activation`  — Applied to every hidden layer.
    /// * `seed`        — Seed for the deterministic PRNG.
    pub fn new(
        layer_sizes: &[usize],
        activation: Activation,
        seed: u64,
    ) -> InterpolateResult<Self> {
        if layer_sizes.len() < 2 {
            return Err(InterpolateError::invalid_input(
                "layer_sizes must have at least 2 elements (input and output)".to_string(),
            ));
        }
        for (i, &s) in layer_sizes.iter().enumerate() {
            if s == 0 {
                return Err(InterpolateError::invalid_input(format!(
                    "layer size at index {i} must be > 0"
                )));
            }
        }

        let mut rng = XorShift64::new(seed);
        let n_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let in_d = layer_sizes[l];
            let out_d = layer_sizes[l + 1];

            if l == n_layers - 1 {
                // Output layer: zero-init so MLP starts as identity-zero.
                weights.push(Array2::<f32>::zeros((out_d, in_d)));
                biases.push(Array1::<f32>::zeros(out_d));
            } else {
                // Glorot (Xavier) uniform: scale = sqrt(6 / (fan_in + fan_out))
                let scale = ((6.0 / (in_d + out_d) as f64).sqrt()) as f32;
                let w: Array2<f32> = Array2::from_shape_fn((out_d, in_d), |_| {
                    (rng.next_normal() * scale as f64) as f32
                });
                let b = Array1::<f32>::zeros(out_d);
                weights.push(w);
                biases.push(b);
            }
        }

        Ok(Self {
            weights,
            biases,
            activation,
            layer_sizes: layer_sizes.to_vec(),
        })
    }

    /// Forward pass — returns the scalar output.
    pub fn forward(&self, x: &Array1<f32>) -> InterpolateResult<Array1<f32>> {
        self.forward_with_cache(x).map(|(out, _)| out)
    }

    /// Forward pass with pre-activation cache (used in backprop).
    ///
    /// Returns `(output, pre_activations)` where `pre_activations[l]` holds the
    /// pre-activation vector for layer `l` (before applying the activation).
    /// The last entry is the output layer's pre-activation.
    pub fn forward_with_cache(
        &self,
        x: &Array1<f32>,
    ) -> InterpolateResult<(Array1<f32>, Vec<Array1<f32>>)> {
        let n_layers = self.weights.len();
        let mut current = x.clone();
        let mut pre_activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let w = &self.weights[l];
            let b = &self.biases[l];
            // pre = W * current + b
            let pre: Array1<f32> = w.dot(&current) + b;
            pre_activations.push(pre.clone());
            if l < n_layers - 1 {
                // Hidden layer: apply activation
                current = pre.mapv(|v| self.activation.forward(v));
            } else {
                // Output layer: linear
                current = pre;
            }
        }

        Ok((current, pre_activations))
    }

    /// Backpropagation for a single training example with MSE loss.
    ///
    /// Returns `(grad_weights, grad_biases)` — gradients of the MSE loss
    /// `0.5 * (output - target)^2` w.r.t. all weights and biases.
    ///
    /// # Arguments
    ///
    /// * `x`      — Input vector.
    /// * `target` — Scalar target value.
    /// * `pre_activations` — Cache from `forward_with_cache`.
    pub fn backward_pub(
        &self,
        x: &Array1<f32>,
        target: f32,
        pre_activations: &[Array1<f32>],
    ) -> InterpolateResult<(Vec<Array2<f32>>, Vec<Array1<f32>>)> {
        self.backward(x, target, pre_activations)
    }

    fn backward(
        &self,
        x: &Array1<f32>,
        target: f32,
        pre_activations: &[Array1<f32>],
    ) -> InterpolateResult<(Vec<Array2<f32>>, Vec<Array1<f32>>)> {
        let n_layers = self.weights.len();

        // Reconstruct activations at each layer.
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(n_layers + 1);
        activations.push(x.clone());
        for l in 0..n_layers - 1 {
            let act = pre_activations[l].mapv(|v| self.activation.forward(v));
            activations.push(act);
        }
        // Last "activation" is output (linear)
        activations.push(pre_activations[n_layers - 1].clone());

        // MSE loss gradient at the output: d_loss/d_output = (output - target)
        let output = &pre_activations[n_layers - 1];
        if output.len() != 1 {
            return Err(InterpolateError::DimensionMismatch(
                "TinyMlp output must be scalar (length 1)".to_string(),
            ));
        }
        let mut delta: Array1<f32> = Array1::from(vec![output[0] - target]);

        let mut grad_w: Vec<Array2<f32>> = vec![Array2::zeros((0, 0)); n_layers];
        let mut grad_b: Vec<Array1<f32>> = vec![Array1::zeros(0); n_layers];

        for l in (0..n_layers).rev() {
            // Gradient of bias = delta (no chain rule needed beyond this layer)
            grad_b[l] = delta.clone();

            // Gradient of weights = delta.outer(activations[l])
            let in_act = &activations[l];
            let gw =
                Array2::from_shape_fn((delta.len(), in_act.len()), |(i, j)| delta[i] * in_act[j]);
            grad_w[l] = gw;

            if l > 0 {
                // Backprop through weight matrix: W^T * delta
                let w = &self.weights[l];
                let wt_delta: Array1<f32> = w.t().dot(&delta);
                // Apply activation derivative
                delta = Array1::from_iter(
                    wt_delta
                        .iter()
                        .zip(pre_activations[l - 1].iter())
                        .map(|(&d, &pre)| d * self.activation.backward(pre)),
                );
            }
        }

        Ok((grad_w, grad_b))
    }

    /// Perform one SGD step with L2 regularisation.
    ///
    /// # Arguments
    ///
    /// * `x`      — Input vector.
    /// * `target` — Scalar target.
    /// * `lr`     — Learning rate.
    /// * `l2`     — L2 regularisation strength.
    pub fn train_step(
        &mut self,
        x: &Array1<f32>,
        target: f32,
        lr: f32,
        l2: f32,
    ) -> InterpolateResult<()> {
        let (output, pre_activations) = self.forward_with_cache(x)?;
        let (grad_w, grad_b) = self.backward(x, target, &pre_activations)?;

        let n_layers = self.weights.len();
        for l in 0..n_layers {
            // SGD update with L2: w = w - lr * (grad_w + l2 * w)
            let reg_term = self.weights[l].mapv(|w| w * l2);
            let delta_w = (grad_w[l].clone() + reg_term) * lr;
            self.weights[l] = self.weights[l].clone() - delta_w;
            let delta_b = grad_b[l].mapv(|g| g * lr);
            self.biases[l] = self.biases[l].clone() - delta_b;
        }

        // Suppress unused variable warning
        let _ = output;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glorot_init_hidden_variance_within_range() {
        // Hidden-layer weights should have variance approximately 2/(fan_in+fan_out).
        let mlp = TinyMlp::new(&[4, 16, 8, 1], Activation::Tanh, 42).expect("construction");

        // Check first hidden layer: fan_in=4, fan_out=16, scale=sqrt(6/20)
        let w = &mlp.weights[0];
        let var: f32 = w.iter().map(|&v| v * v).sum::<f32>() / w.len() as f32;
        let expected_var = 2.0f32 / (4.0 + 16.0); // Glorot expected variance
        let ratio = var / expected_var;
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "First layer weight variance ratio out of range: {ratio}"
        );
    }

    #[test]
    fn output_layer_zero_init() {
        let mlp = TinyMlp::new(&[3, 8, 1], Activation::Tanh, 7).expect("construction");
        let x = Array1::from(vec![1.0f32, 2.0, 3.0]);
        let out = mlp.forward(&x).expect("forward");
        assert_eq!(out.len(), 1);
        assert!(
            out[0].abs() < 1e-6,
            "output should be zero before training, got {}",
            out[0]
        );
    }

    #[test]
    fn forward_backward_gradient_check() {
        // Numerical gradient check on a single weight.
        let mut mlp = TinyMlp::new(&[2, 4, 1], Activation::Tanh, 123).expect("construction");

        // Give the network some non-trivial weights (perturb first hidden layer).
        mlp.weights[0][[0, 0]] = 0.5;
        mlp.weights[0][[1, 0]] = -0.3;
        // Also give the output layer a non-zero weight so gradient is non-trivial.
        mlp.weights[1][[0, 0]] = 0.8;
        mlp.weights[1][[0, 1]] = -0.4;
        mlp.weights[1][[0, 2]] = 0.6;
        mlp.weights[1][[0, 3]] = 0.2;

        let x = Array1::from(vec![0.7f32, -0.3]);
        let target = 0.4f32;

        // Analytic gradient of weight[0][0,0].
        let (_, pres) = mlp.forward_with_cache(&x).expect("fwd");
        let (gw, _) = mlp.backward(&x, target, &pres).expect("bwd");
        let analytic = gw[0][[0, 0]];

        // Numerical gradient.
        let h = 1e-4f32;
        mlp.weights[0][[0, 0]] += h;
        let out_plus = mlp.forward(&x).expect("fwd+")[0];
        mlp.weights[0][[0, 0]] -= 2.0 * h;
        let out_minus = mlp.forward(&x).expect("fwd-")[0];
        mlp.weights[0][[0, 0]] += h; // restore

        let loss = |o: f32| 0.5 * (o - target).powi(2);
        let numerical = (loss(out_plus) - loss(out_minus)) / (2.0 * h);

        assert!(
            (analytic - numerical).abs() < 1e-3,
            "gradient check failed: analytic={analytic:.6}, numerical={numerical:.6}"
        );
    }
}
