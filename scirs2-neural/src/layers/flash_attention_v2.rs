//! Flash Attention V2 implementation
//!
//! This module implements the improved Flash Attention algorithm from:
//! "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
//! by Tri Dao (2023).
//!
//! Key improvements over V1:
//! - Reduced non-matmul FLOPs by tracking separate alpha/beta correction factors
//!   in the online softmax, fusing the rescale into a single multiply
//! - Better parallelism: each Q block is processed independently, enabling
//!   forward-pass parallelism over N/B_r blocks
//! - Causal masking with early-exit: skip entire KV blocks beyond the diagonal

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{s, Array, Array2, Array4, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Configuration for Flash Attention V2
#[derive(Debug, Clone)]
pub struct FlashAttentionV2Config {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Block size for query tiling (B_r in the paper)
    pub block_size_q: usize,
    /// Block size for key/value tiling (B_c in the paper)
    pub block_size_kv: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Custom scaling factor (default: 1/sqrt(head_dim))
    pub scale: Option<f64>,
}

impl Default for FlashAttentionV2Config {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            block_size_q: 128,
            block_size_kv: 128,
            causal: false,
            dropout_prob: 0.0,
            scale: None,
        }
    }
}

impl FlashAttentionV2Config {
    /// Create a new FlashAttentionV2Config
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            ..Default::default()
        }
    }

    /// Set block size for queries (B_r)
    pub fn with_block_size_q(mut self, block_size: usize) -> Self {
        self.block_size_q = block_size;
        self
    }

    /// Set block size for keys/values (B_c)
    pub fn with_block_size_kv(mut self, block_size: usize) -> Self {
        self.block_size_kv = block_size;
        self
    }

    /// Enable causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout_prob: f64) -> Self {
        self.dropout_prob = dropout_prob;
        self
    }

    /// Set custom scale factor
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }
}

/// Forward-pass cache for Flash Attention V2 backward computation.
#[derive(Debug)]
struct ForwardCacheV2<F> {
    /// Row-max per (batch, head), each Vec of length seq_len
    m: Vec<Vec<F>>,
    /// Row-sum per (batch, head), each Vec of length seq_len
    l: Vec<Vec<F>>,
    /// Q after projection [batch, seq, num_heads, head_dim]
    q4d: Array<F, IxDyn>,
    /// K after projection [batch, seq, num_heads, head_dim]
    k4d: Array<F, IxDyn>,
    /// V after projection [batch, seq, num_heads, head_dim]
    v4d: Array<F, IxDyn>,
    /// Per-head output before W_O [batch, seq, num_heads, head_dim]
    o4d: Array<F, IxDyn>,
    /// Input to the layer [batch*seq, d_model]
    input2d: Array<F, IxDyn>,
    batch_size: usize,
    seq_len: usize,
}

/// Flash Attention V2 layer
///
/// Implements the improved Flash Attention algorithm with fused online softmax.
///
/// # Algorithm (per head)
///
/// ```text
/// for each Q block i (rows [i*B_r .. (i+1)*B_r]):
///     O_i = 0, m_i = -inf, l_i = 0
///     for each KV block j (rows [j*B_c .. (j+1)*B_c]):
///         S_ij = Q_i @ K_j^T * scale
///         if causal: mask S_ij where col > row
///         m_ij = rowmax(S_ij)
///         P_ij = exp(S_ij - m_ij)
///         l_ij = rowsum(P_ij)
///         m_new = max(m_i, m_ij)
///         alpha = exp(m_i - m_new)
///         beta  = exp(m_ij - m_new)
///         l_i   = alpha * l_i + beta * l_ij
///         O_i   = alpha * O_i + beta * P_ij @ V_j
///         m_i   = m_new
///     O_i = O_i / l_i
/// ```
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{FlashAttentionV2, FlashAttentionV2Config, Layer};
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let config = FlashAttentionV2Config::new(4, 16).with_causal(true);
/// let attn = FlashAttentionV2::<f64>::new(64, config, &mut rng).expect("failed");
///
/// let input = Array3::<f64>::from_elem((2, 32, 64), 0.1).into_dyn();
/// let output = attn.forward(&input).expect("failed");
/// assert_eq!(output.shape(), &[2, 32, 64]);
/// ```
pub struct FlashAttentionV2<F: Float + Debug + Send + Sync + NumAssign> {
    /// Model dimension
    d_model: usize,
    /// Configuration
    config: FlashAttentionV2Config,
    /// Query projection weights [d_model, d_model]
    w_query: Array<F, IxDyn>,
    /// Key projection weights [d_model, d_model]
    w_key: Array<F, IxDyn>,
    /// Value projection weights [d_model, d_model]
    w_value: Array<F, IxDyn>,
    /// Output projection weights [d_model, d_model]
    w_output: Array<F, IxDyn>,
    /// Scaling factor
    scale: F,
    /// Forward pass cache (interior mutability for &self forward)
    cache: Arc<RwLock<Option<ForwardCacheV2<F>>>>,
    /// Gradient of query weights
    dw_query: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of key weights
    dw_key: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of value weights
    dw_value: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of output weights
    dw_output: Arc<RwLock<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + Send + Sync + NumAssign> std::fmt::Debug for FlashAttentionV2<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlashAttentionV2")
            .field("d_model", &self.d_model)
            .field("num_heads", &self.config.num_heads)
            .field("head_dim", &self.config.head_dim)
            .field("causal", &self.config.causal)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> FlashAttentionV2<F> {
    /// Create a new Flash Attention V2 layer
    pub fn new<R: Rng>(
        d_model: usize,
        config: FlashAttentionV2Config,
        rng: &mut R,
    ) -> Result<Self> {
        let total_dim = config.num_heads * config.head_dim;
        if total_dim != d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_heads * head_dim ({}) must equal d_model ({})",
                total_dim, d_model
            )));
        }

        let xavier_std = (F::from(2.0)
            .ok_or_else(|| NeuralError::InvalidArchitecture("float conversion failed".into()))?
            / F::from(d_model + d_model).ok_or_else(|| {
                NeuralError::InvalidArchitecture("float conversion failed".into())
            })?)
        .sqrt();

        let w_query = Self::init_weight(d_model, d_model, xavier_std, rng)?;
        let w_key = Self::init_weight(d_model, d_model, xavier_std, rng)?;
        let w_value = Self::init_weight(d_model, d_model, xavier_std, rng)?;
        let w_output = Self::init_weight(d_model, d_model, xavier_std, rng)?;

        let scale = config
            .scale
            .and_then(|s| F::from(s))
            .or_else(|| {
                let hd = F::from(config.head_dim)?;
                Some(F::one() / hd.sqrt())
            })
            .ok_or_else(|| NeuralError::InvalidArchitecture("Failed to compute scale".into()))?;

        let zeros = Array::zeros(IxDyn(&[d_model, d_model]));

        Ok(Self {
            d_model,
            config,
            w_query,
            w_key,
            w_value,
            w_output,
            scale,
            cache: Arc::new(RwLock::new(None)),
            dw_query: Arc::new(RwLock::new(zeros.clone())),
            dw_key: Arc::new(RwLock::new(zeros.clone())),
            dw_value: Arc::new(RwLock::new(zeros.clone())),
            dw_output: Arc::new(RwLock::new(zeros)),
        })
    }

    /// Initialize a weight matrix with Box-Muller Xavier initialization
    fn init_weight<R: Rng>(
        in_dim: usize,
        out_dim: usize,
        std_val: F,
        rng: &mut R,
    ) -> Result<Array<F, IxDyn>> {
        let mut weights = Array::zeros(IxDyn(&[in_dim, out_dim]));
        for w in weights.iter_mut() {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let u1_clamped = if u1 < 1e-15 { 1e-15 } else { u1 };
            let z = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *w = F::from(z)
                .ok_or_else(|| NeuralError::InvalidArchitecture("float conversion".into()))?
                * std_val;
        }
        Ok(weights)
    }

    /// Flash Attention V2 core: per-head attention with fused online softmax.
    ///
    /// Returns `(output, final_row_max, final_row_sum)` for backward recomputation.
    fn flash_v2_forward(
        &self,
        query: &Array2<F>,
        key: &Array2<F>,
        value: &Array2<F>,
    ) -> Result<(Array2<F>, Vec<F>, Vec<F>)> {
        let seq_len_q = query.nrows();
        let seq_len_kv = key.nrows();
        let head_dim = query.ncols();

        let br = self.config.block_size_q.min(seq_len_q).max(1);
        let bc = self.config.block_size_kv.min(seq_len_kv).max(1);

        let num_blocks_q = seq_len_q.div_ceil(br);
        let num_blocks_kv = seq_len_kv.div_ceil(bc);

        let mut output = Array2::<F>::zeros((seq_len_q, head_dim));
        // Final row-max and row-sum saved for backward
        let mut final_m = vec![F::neg_infinity(); seq_len_q];
        let mut final_l = vec![F::zero(); seq_len_q];

        for qi in 0..num_blocks_q {
            let q_start = qi * br;
            let q_end = (q_start + br).min(seq_len_q);
            let q_len = q_end - q_start;

            let mut o_block = Array2::<F>::zeros((q_len, head_dim));
            let mut m_i = vec![F::neg_infinity(); q_len];
            let mut l_i = vec![F::zero(); q_len];

            let kv_limit = if self.config.causal {
                q_end.div_ceil(bc).min(num_blocks_kv)
            } else {
                num_blocks_kv
            };

            for kj in 0..kv_limit {
                let kv_start = kj * bc;
                let kv_end = (kv_start + bc).min(seq_len_kv);
                let kv_len = kv_end - kv_start;

                // S_ij = Q_block @ K_block^T * scale
                let mut s_block = Array2::<F>::zeros((q_len, kv_len));
                for i in 0..q_len {
                    for j in 0..kv_len {
                        let mut dot = F::zero();
                        for d in 0..head_dim {
                            dot += query[[q_start + i, d]] * key[[kv_start + j, d]];
                        }
                        s_block[[i, j]] = dot * self.scale;
                    }
                }

                // Causal mask
                if self.config.causal {
                    for i in 0..q_len {
                        let q_pos = q_start + i;
                        for j in 0..kv_len {
                            let k_pos = kv_start + j;
                            if k_pos > q_pos {
                                s_block[[i, j]] = F::neg_infinity();
                            }
                        }
                    }
                }

                // V2 fused online softmax update
                for i in 0..q_len {
                    let mut m_ij = F::neg_infinity();
                    for j in 0..kv_len {
                        if s_block[[i, j]] > m_ij {
                            m_ij = s_block[[i, j]];
                        }
                    }

                    let mut l_ij = F::zero();
                    let mut p_row = vec![F::zero(); kv_len];
                    for j in 0..kv_len {
                        if s_block[[i, j]] > F::neg_infinity() {
                            let p = (s_block[[i, j]] - m_ij).exp();
                            p_row[j] = p;
                            l_ij += p;
                        }
                    }

                    let m_new = if m_i[i] > m_ij { m_i[i] } else { m_ij };

                    let alpha = if m_i[i] == F::neg_infinity() {
                        F::zero()
                    } else {
                        (m_i[i] - m_new).exp()
                    };

                    let beta = if m_ij == F::neg_infinity() {
                        F::zero()
                    } else {
                        (m_ij - m_new).exp()
                    };

                    // O_i = alpha * O_i + beta * P_ij @ V_j
                    for d in 0..head_dim {
                        o_block[[i, d]] = alpha * o_block[[i, d]];
                        for j in 0..kv_len {
                            o_block[[i, d]] += beta * p_row[j] * value[[kv_start + j, d]];
                        }
                    }

                    l_i[i] = alpha * l_i[i] + beta * l_ij;
                    m_i[i] = m_new;
                }
            }

            // Final normalization for this Q block
            for i in 0..q_len {
                if l_i[i] > F::zero() {
                    let inv = F::one() / l_i[i];
                    for d in 0..head_dim {
                        o_block[[i, d]] *= inv;
                    }
                }
                for d in 0..head_dim {
                    output[[q_start + i, d]] = o_block[[i, d]];
                }
                // Save final statistics for backward
                final_m[q_start + i] = m_i[i];
                final_l[q_start + i] = l_i[i];
            }
        }

        Ok((output, final_m, final_l))
    }

    /// Flash Attention V2 backward per head (Algorithm 4 from Dao 2023).
    ///
    /// Returns `(dq, dk, dv)` each of shape [seq_len, head_dim].
    fn flash_v2_backward_head(
        &self,
        q: &Array2<F>,
        k: &Array2<F>,
        v: &Array2<F>,
        o: &Array2<F>,
        do_: &Array2<F>,
        m: &[F],
        l: &[F],
    ) -> Result<(Array2<F>, Array2<F>, Array2<F>)> {
        let seq_len = q.nrows();
        let head_dim = q.ncols();

        let br = self.config.block_size_q.min(seq_len).max(1);
        let bc = self.config.block_size_kv.min(seq_len).max(1);
        let n_q_blocks = seq_len.div_ceil(br);
        let n_kv_blocks = seq_len.div_ceil(bc);

        // D_i = rowsum(dO ⊙ O), shape [seq_len]
        let mut d_vec = vec![F::zero(); seq_len];
        for i in 0..seq_len {
            let mut s = F::zero();
            for d in 0..head_dim {
                s += do_[[i, d]] * o[[i, d]];
            }
            d_vec[i] = s;
        }

        let mut dq = Array2::<F>::zeros((seq_len, head_dim));
        let mut dk = Array2::<F>::zeros((seq_len, head_dim));
        let mut dv = Array2::<F>::zeros((seq_len, head_dim));

        for qi in 0..n_q_blocks {
            let q_start = qi * br;
            let q_end = (q_start + br).min(seq_len);
            let q_len = q_end - q_start;

            let kv_limit = if self.config.causal {
                q_end.div_ceil(bc).min(n_kv_blocks)
            } else {
                n_kv_blocks
            };

            for kj in 0..kv_limit {
                let kv_start = kj * bc;
                let kv_end = (kv_start + bc).min(seq_len);
                let kv_len = kv_end - kv_start;

                // ---- Recompute S_ij ----
                let mut s_ij = Array2::<F>::zeros((q_len, kv_len));
                for i in 0..q_len {
                    for j in 0..kv_len {
                        let mut dot = F::zero();
                        for d in 0..head_dim {
                            dot += q[[q_start + i, d]] * k[[kv_start + j, d]];
                        }
                        s_ij[[i, j]] = dot * self.scale;
                    }
                }

                // Causal mask
                if self.config.causal {
                    for i in 0..q_len {
                        let q_pos = q_start + i;
                        for j in 0..kv_len {
                            if kv_start + j > q_pos {
                                s_ij[[i, j]] = F::neg_infinity();
                            }
                        }
                    }
                }

                // ---- P_ij = exp(S_ij - m_i) / l_i ----
                let mut p_ij = Array2::<F>::zeros((q_len, kv_len));
                for i in 0..q_len {
                    let mi = m[q_start + i];
                    let li = l[q_start + i];
                    let inv_l = if li > F::zero() {
                        F::one() / li
                    } else {
                        F::zero()
                    };
                    for j in 0..kv_len {
                        let s = s_ij[[i, j]];
                        let p = if s > F::neg_infinity() {
                            (s - mi).exp() * inv_l
                        } else {
                            F::zero()
                        };
                        p_ij[[i, j]] = p;
                    }
                }

                // ---- dV_j += P_ij^T @ dO_i ----
                for i in 0..q_len {
                    for j in 0..kv_len {
                        for d in 0..head_dim {
                            dv[[kv_start + j, d]] += p_ij[[i, j]] * do_[[q_start + i, d]];
                        }
                    }
                }

                // ---- dP_ij = dO_i @ V_j^T ----
                let mut dp_ij = Array2::<F>::zeros((q_len, kv_len));
                for i in 0..q_len {
                    for j in 0..kv_len {
                        let mut dot = F::zero();
                        for d in 0..head_dim {
                            dot += do_[[q_start + i, d]] * v[[kv_start + j, d]];
                        }
                        dp_ij[[i, j]] = dot;
                    }
                }

                // ---- dS_ij[r,c] = P_ij[r,c] * (dP_ij[r,c] - D_i[r]) ----
                let mut ds_ij = Array2::<F>::zeros((q_len, kv_len));
                for i in 0..q_len {
                    let di = d_vec[q_start + i];
                    for j in 0..kv_len {
                        ds_ij[[i, j]] = p_ij[[i, j]] * (dp_ij[[i, j]] - di);
                    }
                }

                // ---- dQ_i += dS_ij @ K_j * scale ----
                for i in 0..q_len {
                    for d in 0..head_dim {
                        let mut acc = F::zero();
                        for j in 0..kv_len {
                            acc += ds_ij[[i, j]] * k[[kv_start + j, d]];
                        }
                        dq[[q_start + i, d]] += acc * self.scale;
                    }
                }

                // ---- dK_j += dS_ij^T @ Q_i * scale ----
                for j in 0..kv_len {
                    for d in 0..head_dim {
                        let mut acc = F::zero();
                        for i in 0..q_len {
                            acc += ds_ij[[i, j]] * q[[q_start + i, d]];
                        }
                        dk[[kv_start + j, d]] += acc * self.scale;
                    }
                }
            }
        }

        Ok((dq, dk, dv))
    }

    /// Get the configuration
    pub fn config(&self) -> &FlashAttentionV2Config {
        &self.config
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F> Layer<F> for FlashAttentionV2<F>
where
    F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "FlashAttentionV2 expects 3D input [batch, seq_len, d_model], got {}D",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        if d_model != self.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input dim {} != model dim {}",
                d_model, self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        let input_2d = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape input: {e}")))?;

        let input_2d_view = input_2d
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("to 2D failed".into()))?;

        let w_q_2d = self
            .w_query
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("Q weights 2D".into()))?;
        let w_k_2d = self
            .w_key
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("K weights 2D".into()))?;
        let w_v_2d = self
            .w_value
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("V weights 2D".into()))?;
        let w_o_2d = self
            .w_output
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("O weights 2D".into()))?;

        let q_proj = input_2d_view.dot(&w_q_2d);
        let k_proj = input_2d_view.dot(&w_k_2d);
        let v_proj = input_2d_view.dot(&w_v_2d);

        let q_4d = q_proj
            .into_shape_with_order((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("reshape Q: {e}")))?;
        let k_4d = k_proj
            .into_shape_with_order((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("reshape K: {e}")))?;
        let v_4d = v_proj
            .into_shape_with_order((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("reshape V: {e}")))?;

        let mut output_4d = Array4::<F>::zeros((batch_size, seq_len, num_heads, head_dim));

        let n_heads_total = batch_size * num_heads;
        let mut cache_m: Vec<Vec<F>> = Vec::with_capacity(n_heads_total);
        let mut cache_l: Vec<Vec<F>> = Vec::with_capacity(n_heads_total);

        for b in 0..batch_size {
            for h in 0..num_heads {
                let q_head: Array2<F> = q_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("Q head: {e}")))?;
                let k_head: Array2<F> = k_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("K head: {e}")))?;
                let v_head: Array2<F> = v_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("V head: {e}")))?;

                let (attn_out, row_max, row_sum) =
                    self.flash_v2_forward(&q_head, &k_head, &v_head)?;

                cache_m.push(row_max);
                cache_l.push(row_sum);

                for i in 0..seq_len {
                    for d in 0..head_dim {
                        output_4d[[b, i, h, d]] = attn_out[[i, d]];
                    }
                }
            }
        }

        // Save cache for backward
        let o4d_dyn = output_4d
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("cache o4d: {e}")))?;

        let q4d_dyn = q_4d
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("cache q4d: {e}")))?;
        let k4d_dyn = k_4d
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("cache k4d: {e}")))?;
        let v4d_dyn = v_4d
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("cache v4d: {e}")))?;

        {
            let mut guard = self
                .cache
                .write()
                .map_err(|_| NeuralError::InferenceError("cache write lock poisoned".into()))?;
            *guard = Some(ForwardCacheV2 {
                m: cache_m,
                l: cache_l,
                q4d: q4d_dyn,
                k4d: k4d_dyn,
                v4d: v4d_dyn,
                o4d: o4d_dyn,
                input2d: input_2d,
                batch_size,
                seq_len,
            });
        }

        let output_3d = output_4d
            .into_shape_with_order((batch_size, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape output: {e}")))?;

        let output_2d = output_3d
            .into_shape_with_order((batch_size * seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape O proj: {e}")))?;

        let final_output = output_2d.dot(&w_o_2d);

        let result = final_output
            .into_shape_with_order((batch_size, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape final: {e}")))?;

        Ok(result.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        if grad_output.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "FlashAttentionV2 backward expects 3D grad_output, got {}D",
                grad_output.ndim()
            )));
        }

        let cache_guard = self
            .cache
            .read()
            .map_err(|_| NeuralError::InferenceError("cache read lock poisoned".into()))?;
        let fc = cache_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "FlashAttentionV2 backward called before forward".to_string(),
            )
        })?;

        let batch_size = fc.batch_size;
        let seq_len = fc.seq_len;
        let d_model = self.d_model;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // ----------------------------------------------------------------
        // Step 1: backprop through W_O
        // ----------------------------------------------------------------
        let grad_2d = grad_output
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape grad_output: {e}")))?;

        let grad_2d_view = grad_2d
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("grad_2d Ix2".into()))?;

        let w_o_2d = self
            .w_output
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("W_O Ix2".into()))?;

        // d_output_concat [B*S, D] = grad_output [B*S, D] @ W_O^T [D, D]
        let d_output_concat = grad_2d_view.dot(&w_o_2d.t());

        // dW_O = o4d_2d^T @ grad_2d
        let o4d_2d = fc
            .o4d
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("o4d to 2d: {e}")))?;

        let o4d_2d_view = o4d_2d
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("o4d_2d Ix2".into()))?;

        let dw_o_update = o4d_2d_view.t().dot(&grad_2d_view);
        {
            let mut g = self
                .dw_output
                .write()
                .map_err(|_| NeuralError::InferenceError("dw_output lock".into()))?;
            let gv = g
                .view_mut()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("dw_output Ix2".into()))?;
            Zip::from(gv)
                .and(dw_o_update.view())
                .for_each(|a, &b| *a += b);
        }

        // ----------------------------------------------------------------
        // Step 2: reshape to per-head gradients and run backward per head
        // ----------------------------------------------------------------
        let do_4d = d_output_concat
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("do_4d reshape: {e}")))?;

        let mut dq_4d = Array4::<F>::zeros((batch_size, seq_len, num_heads, head_dim));
        let mut dk_4d = Array4::<F>::zeros((batch_size, seq_len, num_heads, head_dim));
        let mut dv_4d = Array4::<F>::zeros((batch_size, seq_len, num_heads, head_dim));

        for b in 0..batch_size {
            for h in 0..num_heads {
                let idx = b * num_heads + h;

                let q_head: Array2<F> = fc
                    .q4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("q_head bwd: {e}")))?;
                let k_head: Array2<F> = fc
                    .k4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("k_head bwd: {e}")))?;
                let v_head: Array2<F> = fc
                    .v4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("v_head bwd: {e}")))?;
                let o_head: Array2<F> = fc
                    .o4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("o_head bwd: {e}")))?;
                let do_head: Array2<F> = do_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| NeuralError::InferenceError(format!("do_head bwd: {e}")))?;

                let m_head = &fc.m[idx];
                let l_head = &fc.l[idx];

                let (dq_h, dk_h, dv_h) = self.flash_v2_backward_head(
                    &q_head, &k_head, &v_head, &o_head, &do_head, m_head, l_head,
                )?;

                for i in 0..seq_len {
                    for d in 0..head_dim {
                        dq_4d[[b, i, h, d]] = dq_h[[i, d]];
                        dk_4d[[b, i, h, d]] = dk_h[[i, d]];
                        dv_4d[[b, i, h, d]] = dv_h[[i, d]];
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // Step 3: backprop through W_Q, W_K, W_V
        // ----------------------------------------------------------------
        let dq_flat = dq_4d
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("dq_flat: {e}")))?;
        let dk_flat = dk_4d
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("dk_flat: {e}")))?;
        let dv_flat = dv_4d
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("dv_flat: {e}")))?;

        let dq_flat_2d = dq_flat
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("dq_flat Ix2".into()))?;
        let dk_flat_2d = dk_flat
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("dk_flat Ix2".into()))?;
        let dv_flat_2d = dv_flat
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("dv_flat Ix2".into()))?;

        let input2d_view = fc
            .input2d
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("input2d Ix2".into()))?;

        let w_q_2d = self
            .w_query
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("W_Q Ix2".into()))?;
        let w_k_2d = self
            .w_key
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("W_K Ix2".into()))?;
        let w_v_2d = self
            .w_value
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("W_V Ix2".into()))?;

        let dw_q_update = input2d_view.t().dot(&dq_flat_2d);
        let dw_k_update = input2d_view.t().dot(&dk_flat_2d);
        let dw_v_update = input2d_view.t().dot(&dv_flat_2d);

        {
            let mut g = self
                .dw_query
                .write()
                .map_err(|_| NeuralError::InferenceError("dw_query lock".into()))?;
            let gv = g
                .view_mut()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("dw_query Ix2".into()))?;
            Zip::from(gv)
                .and(dw_q_update.view())
                .for_each(|a, &b| *a += b);
        }
        {
            let mut g = self
                .dw_key
                .write()
                .map_err(|_| NeuralError::InferenceError("dw_key lock".into()))?;
            let gv = g
                .view_mut()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("dw_key Ix2".into()))?;
            Zip::from(gv)
                .and(dw_k_update.view())
                .for_each(|a, &b| *a += b);
        }
        {
            let mut g = self
                .dw_value
                .write()
                .map_err(|_| NeuralError::InferenceError("dw_value lock".into()))?;
            let gv = g
                .view_mut()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("dw_value Ix2".into()))?;
            Zip::from(gv)
                .and(dw_v_update.view())
                .for_each(|a, &b| *a += b);
        }

        // d_input = dq_flat @ W_Q^T + dk_flat @ W_K^T + dv_flat @ W_V^T
        let d_input_2d =
            dq_flat_2d.dot(&w_q_2d.t()) + dk_flat_2d.dot(&w_k_2d.t()) + dv_flat_2d.dot(&w_v_2d.t());

        let d_input = d_input_2d
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("d_input reshape: {e}")))?;

        Ok(d_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        macro_rules! apply_grad {
            ($weight:expr, $grad_lock:expr, $name:literal) => {{
                {
                    let dw = $grad_lock.read().map_err(|_| {
                        NeuralError::InferenceError(concat!($name, " read lock").into())
                    })?;
                    let dw_view = dw
                        .view()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|_| NeuralError::InferenceError(concat!($name, " Ix2").into()))?;
                    let mut w_view = $weight
                        .view_mut()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|_| {
                            NeuralError::InferenceError(concat!($name, " w Ix2").into())
                        })?;
                    Zip::from(w_view.view_mut())
                        .and(dw_view)
                        .for_each(|w, &dw_val| *w -= learning_rate * dw_val);
                }
                {
                    let mut g = $grad_lock.write().map_err(|_| {
                        NeuralError::InferenceError(concat!($name, " write lock").into())
                    })?;
                    g.fill(F::zero());
                }
            }};
        }

        apply_grad!(self.w_query, self.dw_query, "dw_query");
        apply_grad!(self.w_key, self.dw_key, "dw_key");
        apply_grad!(self.w_value, self.dw_value, "dw_value");
        apply_grad!(self.w_output, self.dw_output, "dw_output");

        Ok(())
    }

    fn layer_type(&self) -> &str {
        "FlashAttentionV2"
    }
}

/// Standalone Flash Attention V2 compute function (no projection weights)
///
/// Computes attention using the V2 algorithm with tiling and fused online softmax.
///
/// # Arguments
/// * `query` - [batch, seq_q, head_dim]
/// * `key`   - [batch, seq_k, head_dim]
/// * `value` - [batch, seq_k, head_dim]
/// * `causal` - Whether to apply causal masking
/// * `block_size_q` - Block size for Q tiling
/// * `block_size_kv` - Block size for KV tiling
///
/// # Returns
/// Output tensor [batch, seq_q, head_dim]
pub fn flash_attention_v2_compute<F: Float + Debug + ScalarOperand + NumAssign>(
    query: &Array<F, IxDyn>,
    key: &Array<F, IxDyn>,
    value: &Array<F, IxDyn>,
    causal: bool,
    block_size_q: usize,
    block_size_kv: usize,
) -> Result<Array<F, IxDyn>> {
    if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
        return Err(NeuralError::InvalidArchitecture(
            "Q, K, V must be 3D tensors [batch, seq, dim]".into(),
        ));
    }

    let batch_size = query.shape()[0];
    let seq_len_q = query.shape()[1];
    let seq_len_kv = key.shape()[1];
    let head_dim = query.shape()[2];

    if key.shape()[2] != head_dim || value.shape()[2] != head_dim {
        return Err(NeuralError::InvalidArchitecture(
            "Q, K, V head_dim mismatch".into(),
        ));
    }

    let scale = F::one()
        / F::from(head_dim)
            .ok_or_else(|| NeuralError::InvalidArchitecture("float conv".into()))?
            .sqrt();

    let br = block_size_q.min(seq_len_q).max(1);
    let bc = block_size_kv.min(seq_len_kv).max(1);

    let mut output = Array::zeros(IxDyn(&[batch_size, seq_len_q, head_dim]));

    for b in 0..batch_size {
        let num_blocks_q = seq_len_q.div_ceil(br);
        let num_blocks_kv = seq_len_kv.div_ceil(bc);

        for qi in 0..num_blocks_q {
            let q_start = qi * br;
            let q_end = (q_start + br).min(seq_len_q);
            let q_len = q_end - q_start;

            let mut o_block = vec![F::zero(); q_len * head_dim];
            let mut m_i = vec![F::neg_infinity(); q_len];
            let mut l_i = vec![F::zero(); q_len];

            let kv_limit = if causal {
                q_end.div_ceil(bc).min(num_blocks_kv)
            } else {
                num_blocks_kv
            };

            for kj in 0..kv_limit {
                let kv_start = kj * bc;
                let kv_end = (kv_start + bc).min(seq_len_kv);
                let kv_len = kv_end - kv_start;

                for i in 0..q_len {
                    let q_pos = q_start + i;

                    let mut scores = vec![F::zero(); kv_len];
                    let mut m_ij = F::neg_infinity();

                    for (j, score) in scores.iter_mut().enumerate().take(kv_len) {
                        let k_pos = kv_start + j;
                        if causal && k_pos > q_pos {
                            *score = F::neg_infinity();
                        } else {
                            let mut dot = F::zero();
                            for d in 0..head_dim {
                                dot += query[[b, q_pos, d]] * key[[b, k_pos, d]];
                            }
                            *score = dot * scale;
                        }
                        if *score > m_ij {
                            m_ij = *score;
                        }
                    }

                    let mut l_ij = F::zero();
                    let mut p_row = vec![F::zero(); kv_len];
                    for j in 0..kv_len {
                        if scores[j] > F::neg_infinity() {
                            let p = (scores[j] - m_ij).exp();
                            p_row[j] = p;
                            l_ij += p;
                        }
                    }

                    let m_new = if m_i[i] > m_ij { m_i[i] } else { m_ij };
                    let alpha = if m_i[i] == F::neg_infinity() {
                        F::zero()
                    } else {
                        (m_i[i] - m_new).exp()
                    };
                    let beta = if m_ij == F::neg_infinity() {
                        F::zero()
                    } else {
                        (m_ij - m_new).exp()
                    };

                    for d in 0..head_dim {
                        let idx = i * head_dim + d;
                        o_block[idx] = alpha * o_block[idx];
                        for j in 0..kv_len {
                            o_block[idx] += beta * p_row[j] * value[[b, kv_start + j, d]];
                        }
                    }

                    l_i[i] = alpha * l_i[i] + beta * l_ij;
                    m_i[i] = m_new;
                }
            }

            for i in 0..q_len {
                let inv = if l_i[i] > F::zero() {
                    F::one() / l_i[i]
                } else {
                    F::zero()
                };
                for d in 0..head_dim {
                    output[[b, q_start + i, d]] = o_block[i * head_dim + d] * inv;
                }
            }
        }
    }

    Ok(output)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::flash_attention::flash_attention_compute;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_flash_v2_config() {
        let config = FlashAttentionV2Config::new(8, 64)
            .with_causal(true)
            .with_block_size_q(128)
            .with_block_size_kv(64)
            .with_dropout(0.05)
            .with_scale(0.125);

        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
        assert_eq!(config.block_size_q, 128);
        assert_eq!(config.block_size_kv, 64);
        assert!((config.dropout_prob - 0.05).abs() < 1e-10);
        assert!((config.scale.unwrap_or(0.0) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_flash_v2_creation() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(4, 16);
        let result = FlashAttentionV2::<f64>::new(64, config, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flash_v2_forward_shape() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(4, 16)
            .with_block_size_q(8)
            .with_block_size_kv(8);
        let attn = FlashAttentionV2::<f64>::new(64, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((2, 16, 64), 0.1).into_dyn();
        let output = attn.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[2, 16, 64]);
    }

    #[test]
    fn test_flash_v2_causal_masking() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(2, 8)
            .with_causal(true)
            .with_block_size_q(4)
            .with_block_size_kv(4);
        let attn = FlashAttentionV2::<f64>::new(16, config, &mut rng).expect("creation failed");

        let mut input = Array3::<f64>::zeros((1, 8, 16));
        for i in 0..8 {
            for j in 0..16 {
                input[[0, i, j]] = (i as f64 + 1.0) * 0.1 + j as f64 * 0.01;
            }
        }

        let output = attn.forward(&input.into_dyn()).expect("forward failed");
        assert_eq!(output.shape(), &[1, 8, 16]);

        for val in output.iter() {
            assert!(val.is_finite(), "causal output has non-finite value");
        }
    }

    #[test]
    fn test_flash_v2_matches_standard_attention() {
        let query = Array3::<f64>::from_elem((1, 4, 8), 0.5).into_dyn();
        let key = query.clone();
        let value = query.clone();

        let v2_output = flash_attention_v2_compute(&query, &key, &value, false, 2, 2)
            .expect("v2 compute failed");

        // Standard attention
        let scale = 1.0 / (8.0_f64).sqrt();
        let mut scores = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for _d in 0..8 {
                    dot += 0.5 * 0.5;
                }
                scores[[i, j]] = dot * scale;
            }
        }

        let mut attention = scores.clone();
        for i in 0..4 {
            let max_val = attention.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for j in 0..4 {
                let exp_val = (attention[[i, j]] - max_val).exp();
                attention[[i, j]] = exp_val;
                sum += exp_val;
            }
            for j in 0..4 {
                attention[[i, j]] /= sum;
            }
        }

        let mut standard_output = Array2::<f64>::zeros((4, 8));
        for i in 0..4 {
            for d in 0..8 {
                let mut sum = 0.0;
                for j in 0..4 {
                    sum += attention[[i, j]] * 0.5;
                }
                standard_output[[i, d]] = sum;
            }
        }

        for i in 0..4 {
            for d in 0..8 {
                assert!(
                    (v2_output[[0, i, d]] - standard_output[[i, d]]).abs() < 1e-10,
                    "V2 mismatch at [{i}, {d}]"
                );
            }
        }
    }

    #[test]
    fn test_flash_v2_different_block_sizes_same_result() {
        let mut query = Array3::<f64>::zeros((1, 6, 4)).into_dyn();
        let mut key = Array3::<f64>::zeros((1, 6, 4)).into_dyn();
        let mut value = Array3::<f64>::zeros((1, 6, 4)).into_dyn();

        for t in 0..6 {
            for d in 0..4 {
                let v = ((t * 4 + d) as f64) * 0.1;
                query[[0, t, d]] = v;
                key[[0, t, d]] = v * 0.8;
                value[[0, t, d]] = v * 0.5 + 0.1;
            }
        }

        let out_bs2 =
            flash_attention_v2_compute(&query, &key, &value, false, 2, 2).expect("bs2 failed");
        let out_bs3 =
            flash_attention_v2_compute(&query, &key, &value, false, 3, 3).expect("bs3 failed");
        let out_bs6 =
            flash_attention_v2_compute(&query, &key, &value, false, 6, 6).expect("bs6 failed");

        for t in 0..6 {
            for d in 0..4 {
                let a = out_bs2[[0, t, d]];
                let b = out_bs3[[0, t, d]];
                let c = out_bs6[[0, t, d]];
                assert!(
                    (a - b).abs() < 1e-10 && (b - c).abs() < 1e-10,
                    "block size mismatch at [{t}, {d}]: bs2={a}, bs3={b}, bs6={c}"
                );
            }
        }
    }

    #[test]
    fn test_flash_v2_causal_matches_v1_causal() {
        let mut query = Array3::<f64>::zeros((1, 5, 6)).into_dyn();
        let mut key = Array3::<f64>::zeros((1, 5, 6)).into_dyn();
        let mut value = Array3::<f64>::zeros((1, 5, 6)).into_dyn();

        for t in 0..5 {
            for d in 0..6 {
                let v = ((t + 1) as f64 * 0.15) + (d as f64 * 0.03);
                query[[0, t, d]] = v;
                key[[0, t, d]] = v * 1.1;
                value[[0, t, d]] = v * 0.7;
            }
        }

        let v1_out = flash_attention_compute(&query, &key, &value, true, 2).expect("v1 failed");
        let v2_out =
            flash_attention_v2_compute(&query, &key, &value, true, 2, 2).expect("v2 failed");

        for t in 0..5 {
            for d in 0..6 {
                assert!(
                    (v1_out[[0, t, d]] - v2_out[[0, t, d]]).abs() < 1e-10,
                    "v1 vs v2 causal mismatch at [{t}, {d}]"
                );
            }
        }
    }

    #[test]
    fn test_flash_v2_numerical_stability() {
        let mut query = Array3::<f64>::zeros((1, 4, 4)).into_dyn();
        for t in 0..4 {
            for d in 0..4 {
                query[[0, t, d]] = (t as f64 + 1.0) * 10.0;
            }
        }
        let key = query.clone();
        let value = Array3::<f64>::from_elem((1, 4, 4), 1.0).into_dyn();

        let out = flash_attention_v2_compute(&query, &key, &value, false, 2, 2).expect("failed");

        for val in out.iter() {
            assert!(val.is_finite(), "non-finite in stability test");
        }
    }

    #[test]
    fn test_flash_v2_invalid_input() {
        let q_2d = Array2::<f64>::zeros((4, 8)).into_dyn();
        let k = Array3::<f64>::zeros((1, 4, 8)).into_dyn();
        let v = k.clone();

        let result = flash_attention_v2_compute(&q_2d, &k, &v, false, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_v2_backward_shape() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(2, 8)
            .with_block_size_q(4)
            .with_block_size_kv(4);
        let attn = FlashAttentionV2::<f64>::new(16, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 8, 16), 0.1).into_dyn();
        let output = attn.forward(&input).expect("forward failed");
        let grad = Array::ones(output.raw_dim());
        let grad_input = attn.backward(&input, &grad).expect("backward failed");

        assert_eq!(
            grad_input.shape(),
            input.shape(),
            "V2 backward grad_input shape should match input"
        );
    }

    #[test]
    fn test_flash_v2_backward_finite() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(2, 4)
            .with_block_size_q(2)
            .with_block_size_kv(2);
        let attn = FlashAttentionV2::<f64>::new(8, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 4, 8), 0.1).into_dyn();
        let out = attn.forward(&input).expect("forward failed");
        let grad = Array::ones(out.raw_dim());
        let grad_in = attn.backward(&input, &grad).expect("backward failed");

        for val in grad_in.iter() {
            assert!(
                val.is_finite(),
                "V2 backward grad contains non-finite value"
            );
        }
    }

    #[test]
    fn test_flash_v2_gradient_check() {
        // End-to-end numerical gradient check for V2 layer.
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(1, 4)
            .with_block_size_q(2)
            .with_block_size_kv(2);
        let attn = FlashAttentionV2::<f64>::new(4, config, &mut rng).expect("creation failed");

        let input = Array::from_shape_vec(
            IxDyn(&[1, 4, 4]),
            (0..16).map(|x| x as f64 * 0.05).collect::<Vec<_>>(),
        )
        .expect("input creation");

        let out = attn.forward(&input).expect("forward");
        let loss = out.sum();
        let grad_out = Array::ones(out.raw_dim());
        let grad_in = attn.backward(&input, &grad_out).expect("backward");

        let eps = 1e-5_f64;
        let mut input_plus = input.clone();
        input_plus[[0, 0, 0]] += eps;
        let out_plus = attn.forward(&input_plus).expect("forward+");
        let loss_plus = out_plus.sum();

        let numerical_grad = (loss_plus - loss) / eps;
        let analytical_grad = grad_in[[0, 0, 0]];

        let rel_err = (numerical_grad - analytical_grad).abs()
            / (numerical_grad.abs().max(analytical_grad.abs()) + 1e-8);
        assert!(
            rel_err < 1e-3,
            "V2 gradient check failed: numerical={numerical_grad:.6}, analytical={analytical_grad:.6}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn test_flash_v2_update() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionV2Config::new(2, 4)
            .with_block_size_q(2)
            .with_block_size_kv(2);
        let mut attn = FlashAttentionV2::<f64>::new(8, config, &mut rng).expect("creation");

        let input = Array3::<f64>::from_elem((1, 4, 8), 0.1).into_dyn();
        let out = attn.forward(&input).expect("forward");
        let grad = Array::ones(out.raw_dim());
        attn.backward(&input, &grad).expect("backward");
        attn.update(0.01).expect("update");
    }
}
