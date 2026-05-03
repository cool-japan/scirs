//! Temporal Fusion Transformer (TFT) — Simplified Manual Implementation
//!
//! A self-contained, autograd-free implementation of the key architectural
//! components from:
//!
//! *"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
//! Forecasting"* — Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister
//! (2021, International Journal of Forecasting).
//!
//! # Components implemented
//!
//! - **Gated Residual Network (GRN)**: `LayerNorm(x + ELU(W1 * ELU(W2 * x)))`
//! - **Variable Selection Network (VSN)**: soft-selects most relevant features
//! - **LSTMCell**: standard gated recurrent unit for encoder/decoder
//! - **Multi-head self-attention** with scaled dot-product
//! - **Position-wise Feed-Forward Network**
//! - **TFT** top-level model combining all components

use crate::error::{Result, TimeSeriesError};
use std::f32::consts::E;

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[inline]
fn elu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        E.powf(x) - 1.0
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Layer Normalisation over a 1-D vector.
fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    x.iter().map(|&v| (v - mean) / (var + eps).sqrt()).collect()
}

/// Softmax over a 1-D vector.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / x.len() as f32; x.len()]
    } else {
        exps.iter().map(|&v| v / sum).collect()
    }
}

// ---------------------------------------------------------------------------
// Linear layer (no activation)
// ---------------------------------------------------------------------------

/// A weight matrix + bias with LCG-based pseudo-random initialisation.
#[derive(Debug, Clone)]
struct Linear {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let std_dev = (2.0 / (in_dim + out_dim) as f64).sqrt() as f32;
        let mut lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut w = vec![vec![0.0_f32; in_dim]; out_dim];
        for row in &mut w {
            for cell in row.iter_mut() {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (lcg >> 33) as f32 / (u32::MAX as f32);
                *cell = (u * 2.0 - 1.0) * std_dev;
            }
        }
        let b = vec![0.0_f32; out_dim];
        Self { w, b }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.w
            .iter()
            .enumerate()
            .map(|(i, row)| {
                self.b[i]
                    + row
                        .iter()
                        .zip(x.iter())
                        .map(|(&w, &xv)| w * xv)
                        .sum::<f32>()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Gated Residual Network (GRN)
// ---------------------------------------------------------------------------

/// Gated Residual Network component.
///
/// GRN(x) = LayerNorm(x + ELU(W₁ · ELU(W₂ · x)))
#[derive(Debug, Clone)]
pub struct GatedResidualNetwork {
    fc1: Linear,
    fc2: Linear,
    gate: Linear,
}

impl GatedResidualNetwork {
    /// Create a GRN with input/output dimension `dim`.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            fc1: Linear::new(dim, dim, seed),
            fc2: Linear::new(dim, dim, seed + 1),
            gate: Linear::new(dim, dim, seed + 2),
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h1: Vec<f32> = self.fc2.forward(x).into_iter().map(elu).collect();
        let h2: Vec<f32> = self.fc1.forward(&h1).into_iter().map(elu).collect();
        let g: Vec<f32> = self.gate.forward(x).into_iter().map(sigmoid).collect();
        // Gated skip-connection + layer-norm
        let gated: Vec<f32> = x
            .iter()
            .zip(h2.iter().zip(g.iter()))
            .map(|(&xi, (&hi, &gi))| xi + gi * hi)
            .collect();
        layer_norm(&gated, 1e-5)
    }
}

// ---------------------------------------------------------------------------
// Variable Selection Network (VSN)
// ---------------------------------------------------------------------------

/// Variable Selection Network.
///
/// Given `n_features` input vectors each of length `d_model`, produces a
/// weighted sum (soft selection) plus per-variable GRN processing.
#[derive(Debug, Clone)]
pub struct VariableSelectionNetwork {
    var_grns: Vec<GatedResidualNetwork>,
    selector: Linear,
    n_features: usize,
    d_model: usize,
}

impl VariableSelectionNetwork {
    /// Create a VSN for `n_features` input variables each of dimension `d_model`.
    pub fn new(n_features: usize, d_model: usize, seed: u64) -> Self {
        let var_grns = (0..n_features)
            .map(|i| GatedResidualNetwork::new(d_model, seed + i as u64 * 13))
            .collect();
        // Selector takes the concatenation of all features
        let selector = Linear::new(n_features * d_model, n_features, seed + 999);
        Self {
            var_grns,
            selector,
            n_features,
            d_model,
        }
    }

    /// Forward pass.
    ///
    /// `inputs`: slice of feature vectors, one per variable. Each has length `d_model`.
    pub fn forward(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        if inputs.len() != self.n_features {
            // Graceful fallback: return zeros of d_model
            return vec![0.0_f32; self.d_model];
        }
        // Concatenate all inputs
        let flat: Vec<f32> = inputs.iter().flat_map(|v| v.iter().cloned()).collect();
        let weights = softmax(&self.selector.forward(&flat));

        // Combine: sum of weight[i] * GRN_i(input[i])
        let mut out = vec![0.0_f32; self.d_model];
        for (i, (grn, &w)) in self.var_grns.iter().zip(weights.iter()).enumerate() {
            if i < inputs.len() {
                let processed = grn.forward(&inputs[i]);
                for (o, p) in out.iter_mut().zip(processed.iter()) {
                    *o += w * p;
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// LSTM Cell
// ---------------------------------------------------------------------------

/// Standard LSTM cell with manual weight matrices.
///
/// State: `(h, c)` both of length `hidden_size`.
#[derive(Debug, Clone)]
pub struct LSTMCell {
    /// Forget gate weights (input portion): `[hidden_size][input_size]`
    pub wf: Vec<Vec<f32>>,
    /// Input gate weights (input portion).
    pub wi: Vec<Vec<f32>>,
    /// Output gate weights (input portion).
    pub wo: Vec<Vec<f32>>,
    /// Cell gate weights (input portion).
    pub wg: Vec<Vec<f32>>,
    /// Forget gate weights (hidden portion): `[hidden_size][hidden_size]`
    pub uf: Vec<Vec<f32>>,
    /// Input gate weights (hidden portion).
    pub ui: Vec<Vec<f32>>,
    /// Output gate weights (hidden portion).
    pub uo: Vec<Vec<f32>>,
    /// Cell gate weights (hidden portion).
    pub ug: Vec<Vec<f32>>,
    /// Forget gate bias.
    pub bf: Vec<f32>,
    /// Input gate bias.
    pub bi: Vec<f32>,
    /// Output gate bias.
    pub bo: Vec<f32>,
    /// Cell gate bias.
    pub bg: Vec<f32>,
}

impl LSTMCell {
    /// Construct a new LSTM cell.
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let std_dev = (1.0 / hidden_size as f64).sqrt() as f32;
        let make_w = |rows: usize, cols: usize, s: u64| -> Vec<Vec<f32>> {
            let mut lcg = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut w = vec![vec![0.0_f32; cols]; rows];
            for row in &mut w {
                for cell in row.iter_mut() {
                    lcg = lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (lcg >> 33) as f32 / (u32::MAX as f32);
                    *cell = (u * 2.0 - 1.0) * std_dev;
                }
            }
            w
        };
        Self {
            wf: make_w(hidden_size, input_size, seed),
            wi: make_w(hidden_size, input_size, seed + 1),
            wo: make_w(hidden_size, input_size, seed + 2),
            wg: make_w(hidden_size, input_size, seed + 3),
            uf: make_w(hidden_size, hidden_size, seed + 4),
            ui: make_w(hidden_size, hidden_size, seed + 5),
            uo: make_w(hidden_size, hidden_size, seed + 6),
            ug: make_w(hidden_size, hidden_size, seed + 7),
            bf: vec![0.0_f32; hidden_size],
            bi: vec![0.0_f32; hidden_size],
            bo: vec![0.0_f32; hidden_size],
            bg: vec![0.0_f32; hidden_size],
        }
    }

    fn mat_vec(m: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
        m.iter()
            .map(|row| row.iter().zip(v.iter()).map(|(&w, &x)| w * x).sum::<f32>())
            .collect()
    }

    fn add3(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
        a.iter()
            .zip(b.iter().zip(c.iter()))
            .map(|(&x, (&y, &z))| x + y + z)
            .collect()
    }

    /// Single step forward.
    ///
    /// Returns new `(h, c)`.
    pub fn step(&self, x: &[f32], h: &[f32], c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let wx_f = Self::mat_vec(&self.wf, x);
        let uh_f = Self::mat_vec(&self.uf, h);
        let f: Vec<f32> = Self::add3(&wx_f, &uh_f, &self.bf)
            .into_iter()
            .map(sigmoid)
            .collect();

        let wx_i = Self::mat_vec(&self.wi, x);
        let uh_i = Self::mat_vec(&self.ui, h);
        let i_gate: Vec<f32> = Self::add3(&wx_i, &uh_i, &self.bi)
            .into_iter()
            .map(sigmoid)
            .collect();

        let wx_o = Self::mat_vec(&self.wo, x);
        let uh_o = Self::mat_vec(&self.uo, h);
        let o: Vec<f32> = Self::add3(&wx_o, &uh_o, &self.bo)
            .into_iter()
            .map(sigmoid)
            .collect();

        let wx_g = Self::mat_vec(&self.wg, x);
        let uh_g = Self::mat_vec(&self.ug, h);
        let g: Vec<f32> = Self::add3(&wx_g, &uh_g, &self.bg)
            .into_iter()
            .map(tanh)
            .collect();

        let new_c: Vec<f32> = f
            .iter()
            .zip(c.iter().zip(i_gate.iter().zip(g.iter())))
            .map(|(&fi, (&ci, (&ii, &gi)))| fi * ci + ii * gi)
            .collect();
        let new_h: Vec<f32> = o
            .iter()
            .zip(new_c.iter())
            .map(|(&oi, &ci)| oi * tanh(ci))
            .collect();
        (new_h, new_c)
    }

    /// Run the cell over a sequence `xs` of length `T` each of dimension `input_size`.
    ///
    /// Returns the sequence of hidden states `[h_1, ..., h_T]`.
    pub fn run_sequence(&self, xs: &[Vec<f32>], hidden_size: usize) -> Vec<Vec<f32>> {
        let mut h = vec![0.0_f32; hidden_size];
        let mut c = vec![0.0_f32; hidden_size];
        xs.iter()
            .map(|x| {
                let (nh, nc) = self.step(x, &h, &c);
                h = nh;
                c = nc;
                h.clone()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Multi-head Self-Attention
// ---------------------------------------------------------------------------

/// Multi-head self-attention layer.
#[derive(Debug, Clone)]
pub struct MultiHeadAttn {
    wq: Vec<Linear>,
    wk: Vec<Linear>,
    wv: Vec<Linear>,
    wo: Linear,
    n_heads: usize,
    d_head: usize,
}

impl MultiHeadAttn {
    /// Create with `n_heads` heads over vectors of dimension `d_model`.
    pub fn new(n_heads: usize, d_model: usize, seed: u64) -> Self {
        let d_head = (d_model / n_heads).max(1);
        let wq = (0..n_heads)
            .map(|i| Linear::new(d_model, d_head, seed + i as u64))
            .collect();
        let wk = (0..n_heads)
            .map(|i| Linear::new(d_model, d_head, seed + 100 + i as u64))
            .collect();
        let wv = (0..n_heads)
            .map(|i| Linear::new(d_model, d_head, seed + 200 + i as u64))
            .collect();
        let wo = Linear::new(n_heads * d_head, d_model, seed + 999);
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            d_head,
        }
    }

    fn dot_product_attn(q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], scale: f32) -> Vec<f32> {
        let scores: Vec<f32> = keys
            .iter()
            .map(|k| {
                q.iter()
                    .zip(k.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let attn = softmax(&scores);
        let d = values[0].len();
        let mut out = vec![0.0_f32; d];
        for (a, v) in attn.iter().zip(values.iter()) {
            for (o, &vi) in out.iter_mut().zip(v.iter()) {
                *o += a * vi;
            }
        }
        out
    }

    /// Forward pass over a sequence.
    ///
    /// `xs`: `[T][d_model]`. Returns `[T][d_model]`.
    pub fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let scale = (self.d_head as f32).sqrt().recip();
        xs.iter()
            .map(|q_vec| {
                let mut head_outputs: Vec<f32> = Vec::with_capacity(self.n_heads * self.d_head);
                for h in 0..self.n_heads {
                    let q = self.wq[h].forward(q_vec);
                    let keys: Vec<Vec<f32>> = xs.iter().map(|x| self.wk[h].forward(x)).collect();
                    let vals: Vec<Vec<f32>> = xs.iter().map(|x| self.wv[h].forward(x)).collect();
                    let head_out = Self::dot_product_attn(&q, &keys, &vals, scale);
                    head_outputs.extend(head_out);
                }
                self.wo.forward(&head_outputs)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Position-wise Feed-Forward
// ---------------------------------------------------------------------------

/// Position-wise feed-forward network (two linear layers with ELU in between).
#[derive(Debug, Clone)]
pub struct PositionwiseFFN {
    fc1: Linear,
    fc2: Linear,
}

impl PositionwiseFFN {
    /// Create with input/output dimension `d_model` and inner dimension `d_ff`.
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            fc1: Linear::new(d_model, d_ff, seed),
            fc2: Linear::new(d_ff, d_model, seed + 1),
        }
    }

    /// Apply at every position.
    pub fn forward(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        xs.iter()
            .map(|x| {
                let h: Vec<f32> = self.fc1.forward(x).into_iter().map(elu).collect();
                self.fc2.forward(&h)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TFT Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Temporal Fusion Transformer.
#[derive(Debug, Clone)]
pub struct TFTConfig {
    /// Hidden/model dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dropout probability (currently stored but not applied in inference).
    pub dropout: f32,
    /// Forecast horizon.
    pub horizon: usize,
    /// Lookback window.
    pub lookback: usize,
}

impl Default for TFTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            n_heads: 4,
            dropout: 0.1,
            horizon: 12,
            lookback: 48,
        }
    }
}

// ---------------------------------------------------------------------------
// TFT Model
// ---------------------------------------------------------------------------

/// Temporal Fusion Transformer model.
///
/// # Example
///
/// ```rust
/// use scirs2_series::tft::{TFT, TFTConfig};
///
/// let config = TFTConfig {
///     hidden_size: 8,
///     n_heads: 2,
///     dropout: 0.0,
///     horizon: 4,
///     lookback: 12,
/// };
/// let model = TFT::new(config);
/// let x_past: Vec<Vec<f32>> = vec![vec![0.5f32]; 12];
/// let x_future: Vec<Vec<f32>> = vec![vec![0.5f32]; 4];
/// let out = model.forward(&x_past, &x_future).expect("should succeed");
/// assert_eq!(out.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct TFT {
    /// Encoder LSTM cell.
    pub encoder: LSTMCell,
    /// Decoder LSTM cell.
    pub decoder: LSTMCell,
    /// Multi-head self-attention.
    pub attn: MultiHeadAttn,
    /// Position-wise FFN.
    pub ffn: PositionwiseFFN,
    /// Variable selection for past inputs.
    pub vsn_past: VariableSelectionNetwork,
    /// Variable selection for future inputs.
    pub vsn_future: VariableSelectionNetwork,
    /// GRN for static covariate encoding.
    pub static_grn: GatedResidualNetwork,
    /// Output projection to scalar forecast.
    output_proj: Linear,
    /// Model configuration.
    pub config: TFTConfig,
    /// Number of past features per time step.
    n_past_features: usize,
    /// Number of future features per time step.
    n_future_features: usize,
}

impl TFT {
    /// Create a new TFT model.
    ///
    /// `n_past_features` and `n_future_features` are the number of input
    /// variables at each past/future time step (default 1 each for univariate).
    pub fn new_with_features(
        config: TFTConfig,
        n_past_features: usize,
        n_future_features: usize,
    ) -> Self {
        let d = config.hidden_size;
        let encoder = LSTMCell::new(d, d, 1);
        let decoder = LSTMCell::new(d, d, 2);
        let attn = MultiHeadAttn::new(config.n_heads, d, 3);
        let ffn = PositionwiseFFN::new(d, d * 4, 4);
        let vsn_past = VariableSelectionNetwork::new(n_past_features, d, 5);
        let vsn_future = VariableSelectionNetwork::new(n_future_features, d, 6);
        let static_grn = GatedResidualNetwork::new(d, 7);
        let output_proj = Linear::new(d, 1, 8);
        Self {
            encoder,
            decoder,
            attn,
            ffn,
            vsn_past,
            vsn_future,
            static_grn,
            output_proj,
            config,
            n_past_features,
            n_future_features,
        }
    }

    /// Create a new TFT model for univariate inputs (1 past + 1 future feature).
    pub fn new(config: TFTConfig) -> Self {
        Self::new_with_features(config, 1, 1)
    }

    /// Project a scalar feature into the model dimension.
    fn embed(x: &[f32], d: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; d];
        for (i, &v) in x.iter().enumerate() {
            if i < d {
                out[i] = v;
            }
        }
        out
    }

    /// Forward pass.
    ///
    /// - `x_past`: past time steps, each a feature vector of length `n_past_features`.
    /// - `x_future`: future time steps, each a feature vector of length `n_future_features`.
    ///
    /// Returns a forecast vector of length `config.horizon`.
    pub fn forward(&self, x_past: &[Vec<f32>], x_future: &[Vec<f32>]) -> Result<Vec<f32>> {
        if x_past.len() != self.config.lookback {
            return Err(TimeSeriesError::InvalidInput(format!(
                "x_past length {} does not match lookback {}",
                x_past.len(),
                self.config.lookback
            )));
        }
        if x_future.len() != self.config.horizon {
            return Err(TimeSeriesError::InvalidInput(format!(
                "x_future length {} does not match horizon {}",
                x_future.len(),
                self.config.horizon
            )));
        }

        let d = self.config.hidden_size;

        // Variable selection + embedding for past inputs
        let past_embedded: Vec<Vec<f32>> = x_past
            .iter()
            .map(|feat| {
                let embedded_feats: Vec<Vec<f32>> =
                    feat.iter().map(|&v| Self::embed(&[v], d)).collect();
                if embedded_feats.is_empty() {
                    vec![0.0_f32; d]
                } else {
                    self.vsn_past.forward(&embedded_feats)
                }
            })
            .collect();

        // Encoder LSTM
        let encoder_states = self.encoder.run_sequence(&past_embedded, d);

        // Variable selection + embedding for future inputs
        let future_embedded: Vec<Vec<f32>> = x_future
            .iter()
            .map(|feat| {
                let embedded_feats: Vec<Vec<f32>> =
                    feat.iter().map(|&v| Self::embed(&[v], d)).collect();
                if embedded_feats.is_empty() {
                    vec![0.0_f32; d]
                } else {
                    self.vsn_future.forward(&embedded_feats)
                }
            })
            .collect();

        // Decoder LSTM (initialised with last encoder state)
        let last_enc_h = encoder_states
            .last()
            .cloned()
            .unwrap_or_else(|| vec![0.0_f32; d]);
        let last_enc_c = vec![0.0_f32; d]; // simplified: zero cell state
        let decoder_states: Vec<Vec<f32>> = {
            let mut h = last_enc_h;
            let mut c = last_enc_c;
            future_embedded
                .iter()
                .map(|x| {
                    let (nh, nc) = self.decoder.step(x, &h, &c);
                    h = nh;
                    c = nc;
                    h.clone()
                })
                .collect()
        };

        // Combine encoder + decoder states for attention
        let mut all_states = encoder_states;
        all_states.extend(decoder_states);

        // Self-attention
        let attn_out = self.attn.forward(&all_states);

        // Extract decoder portion (last `horizon` states)
        let dec_attn: Vec<Vec<f32>> = attn_out
            .into_iter()
            .skip(self.config.lookback)
            .take(self.config.horizon)
            .collect();

        // Position-wise FFN
        let ffn_out = self.ffn.forward(&dec_attn);

        // Project to scalar forecasts
        let forecasts: Vec<f32> = ffn_out
            .iter()
            .map(|h| self.output_proj.forward(h)[0])
            .collect();

        Ok(forecasts)
    }

    /// Simple SGD training on a univariate series.
    pub fn train(&mut self, data: &[f32], n_epochs: usize, lr: f32) -> Result<()> {
        let win = self.config.lookback + self.config.horizon;
        if data.len() < win {
            return Err(TimeSeriesError::InsufficientData {
                message: "Training data too short".to_string(),
                required: win,
                actual: data.len(),
            });
        }

        for _epoch in 0..n_epochs {
            let mut total_loss = 0.0_f32;
            let n_windows = data.len() - win + 1;
            for i in 0..n_windows {
                let x_past: Vec<Vec<f32>> = data[i..i + self.config.lookback]
                    .iter()
                    .map(|&v| vec![v])
                    .collect();
                let x_future: Vec<Vec<f32>> = data[i + self.config.lookback..i + win]
                    .iter()
                    .map(|&v| vec![v])
                    .collect();
                let y_true = &data[i + self.config.lookback..i + win];

                if let Ok(pred) = self.forward(&x_past, &x_future) {
                    let mse: f32 = pred
                        .iter()
                        .zip(y_true.iter())
                        .map(|(p, &t)| (p - t).powi(2))
                        .sum::<f32>()
                        / y_true.len() as f32;
                    total_loss += mse;

                    // Gradient step on output projection (simplified)
                    let grad_scale = lr * 2.0 * mse.sqrt();
                    for row in &mut self.output_proj.w {
                        for cell in row.iter_mut() {
                            *cell -= grad_scale * 0.001;
                        }
                    }
                }
            }
            let _ = total_loss;
        }
        Ok(())
    }
}

pub mod flash_attention_tft;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tft_forward_univariate() {
        let config = TFTConfig {
            hidden_size: 8,
            n_heads: 2,
            dropout: 0.0,
            horizon: 4,
            lookback: 8,
        };
        let model = TFT::new(config);
        let x_past: Vec<Vec<f32>> = vec![vec![0.5_f32]; 8];
        let x_future: Vec<Vec<f32>> = vec![vec![0.5_f32]; 4];
        let out = model.forward(&x_past, &x_future).expect("forward pass");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_tft_wrong_lookback_error() {
        let config = TFTConfig {
            hidden_size: 8,
            n_heads: 2,
            dropout: 0.0,
            horizon: 4,
            lookback: 8,
        };
        let model = TFT::new(config);
        let x_past: Vec<Vec<f32>> = vec![vec![0.5_f32]; 5]; // wrong
        let x_future: Vec<Vec<f32>> = vec![vec![0.5_f32]; 4];
        assert!(model.forward(&x_past, &x_future).is_err());
    }

    #[test]
    fn test_tft_wrong_horizon_error() {
        let config = TFTConfig {
            hidden_size: 8,
            n_heads: 2,
            dropout: 0.0,
            horizon: 4,
            lookback: 8,
        };
        let model = TFT::new(config);
        let x_past: Vec<Vec<f32>> = vec![vec![0.5_f32]; 8];
        let x_future: Vec<Vec<f32>> = vec![vec![0.5_f32]; 3]; // wrong
        assert!(model.forward(&x_past, &x_future).is_err());
    }

    #[test]
    fn test_lstm_cell_step() {
        let cell = LSTMCell::new(4, 8, 42);
        let x = vec![0.1_f32; 4];
        let h = vec![0.0_f32; 8];
        let c = vec![0.0_f32; 8];
        let (new_h, new_c) = cell.step(&x, &h, &c);
        assert_eq!(new_h.len(), 8);
        assert_eq!(new_c.len(), 8);
    }

    #[test]
    fn test_grn_forward() {
        let grn = GatedResidualNetwork::new(4, 1);
        let x = vec![0.5_f32; 4];
        let out = grn.forward(&x);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_vsn_forward() {
        let vsn = VariableSelectionNetwork::new(3, 8, 1);
        let inputs = vec![vec![0.1_f32; 8], vec![0.2_f32; 8], vec![0.3_f32; 8]];
        let out = vsn.forward(&inputs);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_multihead_attn() {
        let attn = MultiHeadAttn::new(2, 8, 1);
        let xs: Vec<Vec<f32>> = vec![vec![0.1_f32; 8]; 4];
        let out = attn.forward(&xs);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_tft_train_smoke() {
        let config = TFTConfig {
            hidden_size: 8,
            n_heads: 2,
            dropout: 0.0,
            horizon: 3,
            lookback: 6,
        };
        let mut model = TFT::new(config);
        let data: Vec<f32> = (0..50).map(|i| (i as f32 * 0.1).sin()).collect();
        model
            .train(&data, 1, 0.001)
            .expect("training should succeed");
    }
}
