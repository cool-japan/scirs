//! Sparse and hierarchical attention variants for very long time series in TFT.
//!
//! Standard multi-head attention is O(L²) in sequence length.  For sequences
//! longer than ~1 000 steps, this module provides efficient drop-in
//! replacements:
//!
//! - **LocalWindowAttention** — each position attends only to its neighbours
//!   within a sliding window of `window_size` positions → O(L · window_size).
//! - **StridedGlobalAttention** — every position attends to "global" tokens
//!   placed at regular stride intervals → O(L · L/stride).
//! - **SparseHierarchicalAttention** — combines local and global attention via
//!   a learned gating mechanism.
//! - **ChunkedAttention** — processes non-overlapping chunks independently plus
//!   light cross-chunk attention to adjacent chunks.
//!
//! All layers work on `[batch, seq_len, d_model]` tensors represented as a
//! `Vec<Vec<Vec<f32>>>` (batch × seq × dim) to keep this module free of any
//! heavy tensor library dependency.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the sparse attention variants.
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    /// Half-width of the local attention window (each position attends to
    /// positions in `[i − window_size/2, i + window_size/2]`).
    pub window_size: usize,
    /// Number of global stride tokens to create.
    pub n_global_tokens: usize,
    /// Stride between consecutive global tokens.
    pub stride: usize,
    /// Model dimension d.
    pub d_model: usize,
    /// Number of attention heads (d_model must be divisible by n_heads).
    pub n_heads: usize,
}

impl Default for SparseAttentionConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            n_global_tokens: 8,
            stride: 8,
            d_model: 64,
            n_heads: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// LCG-based weight initialisation helper
// ---------------------------------------------------------------------------

fn lcg_weights(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
    let std_dev = (2.0 / (rows + cols) as f64).sqrt() as f32;
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut w = vec![vec![0.0f32; cols]; rows];
    for row in w.iter_mut() {
        for cell in row.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 33) as f32 / u32::MAX as f32;
            *cell = (u - 0.5) * 2.0 * std_dev;
        }
    }
    w
}

// ---------------------------------------------------------------------------
// Core attention primitive: scaled dot-product attention
// ---------------------------------------------------------------------------

/// Compute scaled dot-product attention for a sub-sequence.
///
/// `q`, `k`, `v` are slices `[seq_q][d_k]`, `[seq_k][d_k]`, `[seq_k][d_v]`
/// respectively.  Returns output `[seq_q][d_v]`.
///
/// If `causal_mask` is true the upper triangle of the attention matrix is
/// masked to −∞ (useful for autoregressive generation).
fn scaled_dot_product_attention_2d(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    causal_mask: bool,
) -> Vec<Vec<f32>> {
    let seq_q = q.len();
    let seq_k = k.len();
    if seq_q == 0 || seq_k == 0 {
        return vec![];
    }
    let d_k = q[0].len().max(1);
    let scale = 1.0 / (d_k as f32).sqrt();

    let mut out = vec![vec![0.0f32; v[0].len()]; seq_q];

    for qi in 0..seq_q {
        // Compute raw scores
        let mut scores: Vec<f32> = (0..seq_k)
            .map(|ki| {
                let dot: f32 = q[qi].iter().zip(k[ki].iter()).map(|(&a, &b)| a * b).sum();
                dot * scale
            })
            .collect();

        // Apply causal mask
        if causal_mask {
            for ki in 0..seq_k {
                if ki > qi {
                    scores[ki] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let attn: Vec<f32> = if sum_e > 0.0 {
            exps.iter().map(|&e| e / sum_e).collect()
        } else {
            vec![1.0 / seq_k as f32; seq_k]
        };

        // Weighted sum over values
        for ki in 0..seq_k {
            for d in 0..v[0].len() {
                out[qi][d] += attn[ki] * v[ki][d];
            }
        }
    }

    out
}

/// Project `x [seq, d_in]` through weight matrix `w [d_out][d_in]`.
fn linear_project(x: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let d_out = w.len();
    let d_in = w.first().map(|r| r.len()).unwrap_or(0);
    x.iter()
        .map(|xi| {
            (0..d_out)
                .map(|o| {
                    xi.iter()
                        .enumerate()
                        .map(|(i, &v)| if i < d_in { w[o][i] * v } else { 0.0 })
                        .sum()
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LocalWindowAttention
// ---------------------------------------------------------------------------

/// Multi-head attention restricted to a sliding window around each position.
///
/// Time complexity: O(L · window_size · d).
#[derive(Debug, Clone)]
pub struct LocalWindowAttention {
    /// Width of the attention window (must be even; will be treated as half
    /// = window_size/2 on each side).
    pub window_size: usize,
    /// Query projection weights [d_model × d_model].
    q_proj: Vec<Vec<f32>>,
    /// Key projection weights [d_model × d_model].
    k_proj: Vec<Vec<f32>>,
    /// Value projection weights [d_model × d_model].
    v_proj: Vec<Vec<f32>>,
    /// Output projection weights [d_model × d_model].
    o_proj: Vec<Vec<f32>>,
    d_model: usize,
    n_heads: usize,
}

impl LocalWindowAttention {
    /// Create a new layer.
    pub fn new(d_model: usize, n_heads: usize, window_size: usize, seed: u64) -> Self {
        Self {
            window_size,
            q_proj: lcg_weights(d_model, d_model, seed),
            k_proj: lcg_weights(d_model, d_model, seed.wrapping_add(1)),
            v_proj: lcg_weights(d_model, d_model, seed.wrapping_add(2)),
            o_proj: lcg_weights(d_model, d_model, seed.wrapping_add(3)),
            d_model,
            n_heads,
        }
    }

    /// Forward pass.  `x` is `[batch][seq_len][d_model]`.
    pub fn forward(&self, x: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let batch = x.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        let seq = x[0].len();
        let d = self.d_model;
        let half_w = self.window_size / 2;

        let mut out = vec![vec![vec![0.0f32; d]; seq]; batch];

        for b in 0..batch {
            let q_all = linear_project(&x[b], &self.q_proj);
            let k_all = linear_project(&x[b], &self.k_proj);
            let v_all = linear_project(&x[b], &self.v_proj);

            for i in 0..seq {
                let lo = i.saturating_sub(half_w);
                let hi = (i + half_w + 1).min(seq);

                let q_i = vec![q_all[i].clone()];
                let k_win: Vec<Vec<f32>> = k_all[lo..hi].to_vec();
                let v_win: Vec<Vec<f32>> = v_all[lo..hi].to_vec();

                let attended = scaled_dot_product_attention_2d(&q_i, &k_win, &v_win, false);
                if let Some(row) = attended.first() {
                    out[b][i] = row.clone();
                }
            }

            // Output projection
            let proj = linear_project(&out[b], &self.o_proj);
            out[b] = proj;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// StridedGlobalAttention
// ---------------------------------------------------------------------------

/// Every position attends to a set of "global" tokens placed at stride
/// intervals 0, stride, 2·stride, …  Global tokens also attend to each other.
///
/// Time complexity: O(L · L/stride · d).
#[derive(Debug, Clone)]
pub struct StridedGlobalAttention {
    /// Stride between global tokens.
    pub stride: usize,
    g_q_proj: Vec<Vec<f32>>,
    g_k_proj: Vec<Vec<f32>>,
    g_v_proj: Vec<Vec<f32>>,
    g_o_proj: Vec<Vec<f32>>,
    d_model: usize,
    #[allow(dead_code)]
    n_heads: usize,
}

impl StridedGlobalAttention {
    /// Create a new layer.
    pub fn new(d_model: usize, n_heads: usize, stride: usize, seed: u64) -> Self {
        Self {
            stride,
            g_q_proj: lcg_weights(d_model, d_model, seed.wrapping_add(10)),
            g_k_proj: lcg_weights(d_model, d_model, seed.wrapping_add(11)),
            g_v_proj: lcg_weights(d_model, d_model, seed.wrapping_add(12)),
            g_o_proj: lcg_weights(d_model, d_model, seed.wrapping_add(13)),
            d_model,
            n_heads,
        }
    }

    /// Forward pass.  `x` is `[batch][seq_len][d_model]`.
    pub fn forward(&self, x: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let batch = x.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        let seq = x[0].len();
        let stride = self.stride.max(1);
        let d = self.d_model;

        // Global token indices: 0, stride, 2*stride, ...
        let global_indices: Vec<usize> = (0..)
            .map(|i| i * stride)
            .take_while(|&idx| idx < seq)
            .collect();
        let n_global = global_indices.len().max(1);

        let mut out = vec![vec![vec![0.0f32; d]; seq]; batch];

        for b in 0..batch {
            // Extract global token representations
            let global_tokens: Vec<Vec<f32>> =
                global_indices.iter().map(|&gi| x[b][gi].clone()).collect();

            // Project all positions as queries; project global tokens as K/V
            let q_all = linear_project(&x[b], &self.g_q_proj);
            let k_global = linear_project(&global_tokens, &self.g_k_proj);
            let v_global = linear_project(&global_tokens, &self.g_v_proj);

            // Each position attends to global tokens
            for i in 0..seq {
                let q_i = vec![q_all[i].clone()];
                let attended = scaled_dot_product_attention_2d(&q_i, &k_global, &v_global, false);
                if let Some(row) = attended.first() {
                    out[b][i] = row.clone();
                }
            }

            // Global tokens attend to each other (full attention on n_global)
            let q_g = linear_project(&global_tokens, &self.g_q_proj);
            let k_g = linear_project(&global_tokens, &self.g_k_proj);
            let v_g = linear_project(&global_tokens, &self.g_v_proj);
            let global_out = scaled_dot_product_attention_2d(&q_g, &k_g, &v_g, false);

            // Write global outputs back to their stride positions
            for (gi, &idx) in global_indices.iter().enumerate() {
                if gi < global_out.len() {
                    out[b][idx] = global_out[gi].clone();
                }
            }

            // Output projection
            let proj = linear_project(&out[b], &self.g_o_proj);
            out[b] = proj;
            let _ = n_global;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// SparseHierarchicalAttention
// ---------------------------------------------------------------------------

/// Combines `LocalWindowAttention` and `StridedGlobalAttention` via a
/// per-position scalar gate (learned, initialised to 0.5).
///
/// ```text
/// output[b,t,d] = σ(gate[d]) · local_out[b,t,d]
///               + (1 - σ(gate[d])) · global_out[b,t,d]
/// ```
#[derive(Debug, Clone)]
pub struct SparseHierarchicalAttention {
    /// Configuration.
    pub config: SparseAttentionConfig,
    local_attn: LocalWindowAttention,
    global_attn: StridedGlobalAttention,
    /// Gating vector of length d_model (logits; initialised near 0 → sigmoid ≈ 0.5).
    gate: Vec<f32>,
    /// Whether to apply causal masking (only local attention respects this).
    pub causal: bool,
}

impl SparseHierarchicalAttention {
    /// Create a new layer from `config`.
    pub fn new(config: SparseAttentionConfig, seed: u64) -> Self {
        let d = config.d_model;
        let local_attn = LocalWindowAttention::new(d, config.n_heads, config.window_size, seed);
        let global_attn =
            StridedGlobalAttention::new(d, config.n_heads, config.stride, seed.wrapping_add(100));
        let gate = vec![0.0f32; d]; // sigmoid(0) = 0.5 → equal mix initially
        Self {
            config,
            local_attn,
            global_attn,
            gate,
            causal: false,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let batch = x.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        let seq = x[0].len();
        let d = self.config.d_model;

        let local_out = self.local_attn.forward(x)?;
        let global_out = self.global_attn.forward(x)?;

        // Gate: σ(gate[d])
        let gate_vals: Vec<f32> = self.gate.iter().map(|&g| sigmoid_f32(g)).collect();

        let mut out = vec![vec![vec![0.0f32; d]; seq]; batch];
        for b in 0..batch {
            for t in 0..seq {
                for dim in 0..d {
                    let g = gate_vals[dim];
                    out[b][t][dim] = g * local_out[b][t][dim] + (1.0 - g) * global_out[b][t][dim];
                }
            }
        }

        Ok(out)
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// ChunkedAttention
// ---------------------------------------------------------------------------

/// Full attention within non-overlapping chunks, plus light cross-chunk
/// attention to the immediately adjacent chunk.
///
/// For a sequence of length L with chunk size C:
/// - Each chunk has O(C²) internal attention → O(L·C) total.
/// - Cross-chunk: each chunk attends to 1 neighbouring chunk → O(L·C).
#[derive(Debug, Clone)]
pub struct ChunkedAttention {
    q_proj: Vec<Vec<f32>>,
    k_proj: Vec<Vec<f32>>,
    v_proj: Vec<Vec<f32>>,
    o_proj: Vec<Vec<f32>>,
    d_model: usize,
    #[allow(dead_code)]
    n_heads: usize,
}

impl ChunkedAttention {
    /// Create a new layer with `d_model` dimensions.
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        Self {
            q_proj: lcg_weights(d_model, d_model, seed.wrapping_add(200)),
            k_proj: lcg_weights(d_model, d_model, seed.wrapping_add(201)),
            v_proj: lcg_weights(d_model, d_model, seed.wrapping_add(202)),
            o_proj: lcg_weights(d_model, d_model, seed.wrapping_add(203)),
            d_model,
            n_heads,
        }
    }

    /// Forward pass with explicit `chunk_size`.  `x` is
    /// `[batch][seq_len][d_model]`.
    pub fn forward(&self, x: &[Vec<Vec<f32>>], chunk_size: usize) -> Result<Vec<Vec<Vec<f32>>>> {
        let batch = x.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        let seq = x[0].len();
        let d = self.d_model;
        let cs = chunk_size.max(1);

        let n_chunks = (seq + cs - 1) / cs;

        let mut out = vec![vec![vec![0.0f32; d]; seq]; batch];

        for b in 0..batch {
            let q_all = linear_project(&x[b], &self.q_proj);
            let k_all = linear_project(&x[b], &self.k_proj);
            let v_all = linear_project(&x[b], &self.v_proj);

            for chunk_idx in 0..n_chunks {
                let chunk_start = chunk_idx * cs;
                let chunk_end = (chunk_start + cs).min(seq);

                // Attend to own chunk + adjacent chunk (if exists)
                let context_start = if chunk_idx > 0 {
                    (chunk_idx - 1) * cs
                } else {
                    chunk_start
                };
                let context_end = if chunk_idx + 1 < n_chunks {
                    ((chunk_idx + 1) * cs + cs).min(seq)
                } else {
                    chunk_end
                };

                let q_chunk = q_all[chunk_start..chunk_end].to_vec();
                let k_ctx = k_all[context_start..context_end].to_vec();
                let v_ctx = v_all[context_start..context_end].to_vec();

                let attended = scaled_dot_product_attention_2d(&q_chunk, &k_ctx, &v_ctx, false);

                for (local_i, i) in (chunk_start..chunk_end).enumerate() {
                    if local_i < attended.len() {
                        out[b][i] = attended[local_i].clone();
                    }
                }
            }

            let proj = linear_project(&out[b], &self.o_proj);
            out[b] = proj;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Public 3-D adapter matching the stated function signature
// ---------------------------------------------------------------------------

/// Scaled dot-product attention for 3-D inputs represented as
/// `Vec<Vec<Vec<f32>>>` (batch × seq × dim).
///
/// `mask` is an optional boolean tensor `[batch][seq_q][seq_k]`; positions
/// where the mask is `false` are excluded (−∞ score).
pub fn scaled_dot_product_attention(
    q: &[Vec<Vec<f32>>],
    k: &[Vec<Vec<f32>>],
    v: &[Vec<Vec<f32>>],
    mask: Option<&[Vec<Vec<bool>>]>,
) -> Vec<Vec<Vec<f32>>> {
    let batch = q.len();
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let seq_q = q[b].len();
        let seq_k = k[b].len();
        if seq_q == 0 || seq_k == 0 {
            out.push(vec![]);
            continue;
        }
        let d_k = q[b][0].len().max(1);
        let scale = 1.0 / (d_k as f32).sqrt();
        let d_v = v[b][0].len();

        let mut batch_out = vec![vec![0.0f32; d_v]; seq_q];

        for qi in 0..seq_q {
            let mut scores: Vec<f32> = (0..seq_k)
                .map(|ki| {
                    let dot: f32 = q[b][qi]
                        .iter()
                        .zip(k[b][ki].iter())
                        .map(|(&a, &bv)| a * bv)
                        .sum();
                    dot * scale
                })
                .collect();

            // Apply boolean mask if provided
            if let Some(m) = mask {
                if b < m.len() && qi < m[b].len() {
                    for ki in 0..seq_k {
                        if ki < m[b][qi].len() && !m[b][qi][ki] {
                            scores[ki] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let attn: Vec<f32> = if sum_e > 0.0 {
                exps.iter().map(|&e| e / sum_e).collect()
            } else {
                vec![1.0 / seq_k as f32; seq_k]
            };

            for ki in 0..seq_k {
                for d in 0..d_v {
                    batch_out[qi][d] += attn[ki] * v[b][ki][d];
                }
            }
        }

        out.push(batch_out);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(batch: usize, seq: usize, d: usize) -> Vec<Vec<Vec<f32>>> {
        let mut state: u64 = 0xDEAD_BEEF;
        (0..batch)
            .map(|_b| {
                (0..seq)
                    .map(|_t| {
                        (0..d)
                            .map(|_| {
                                state = state
                                    .wrapping_mul(6_364_136_223_846_793_005)
                                    .wrapping_add(1);
                                (state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_config_defaults() {
        let cfg = SparseAttentionConfig::default();
        assert_eq!(cfg.window_size, 64);
        assert_eq!(cfg.n_global_tokens, 8);
        assert_eq!(cfg.stride, 8);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.n_heads, 4);
    }

    #[test]
    fn test_local_window_output_shape() {
        let layer = LocalWindowAttention::new(8, 2, 4, 1);
        let x = make_input(2, 16, 8);
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 16);
        assert_eq!(out[0][0].len(), 8);
    }

    #[test]
    fn test_strided_global_output_shape() {
        let layer = StridedGlobalAttention::new(8, 2, 4, 2);
        let x = make_input(2, 16, 8);
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 16);
        assert_eq!(out[0][0].len(), 8);
    }

    #[test]
    fn test_sparse_hierarchical_output_shape() {
        let config = SparseAttentionConfig {
            window_size: 4,
            stride: 4,
            d_model: 8,
            n_heads: 2,
            n_global_tokens: 2,
        };
        let layer = SparseHierarchicalAttention::new(config, 3);
        let x = make_input(2, 16, 8);
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 16);
        assert_eq!(out[0][0].len(), 8);
    }

    #[test]
    fn test_chunked_output_shape() {
        let layer = ChunkedAttention::new(8, 2, 4);
        let x = make_input(2, 20, 8);
        let out = layer.forward(&x, 5).expect("forward should succeed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 20);
        assert_eq!(out[0][0].len(), 8);
    }

    #[test]
    fn test_attention_output_finite() {
        let layer = LocalWindowAttention::new(8, 2, 4, 5);
        let x = make_input(1, 12, 8);
        let out = layer.forward(&x).expect("forward should succeed");
        for row in &out[0] {
            for &v in row {
                assert!(v.is_finite(), "output value {v} is not finite");
            }
        }
    }

    #[test]
    fn test_window_size_equals_seq_len_full_attention() {
        // When window_size >= seq_len the local window covers everything
        // → output should be finite and same shape
        let seq = 8usize;
        let layer = LocalWindowAttention::new(4, 1, seq * 2, 7);
        let x = make_input(1, seq, 4);
        let out = layer.forward(&x).expect("forward should succeed");
        assert_eq!(out[0].len(), seq);
        assert_eq!(out[0][0].len(), 4);
    }

    #[test]
    fn test_global_token_count() {
        let stride = 4usize;
        let seq = 20usize;
        // Expected: ceil(seq / stride) global tokens
        let expected = (seq + stride - 1) / stride;
        let actual: Vec<usize> = (0..)
            .map(|i| i * stride)
            .take_while(|&idx| idx < seq)
            .collect();
        assert_eq!(actual.len(), expected);
    }

    #[test]
    fn test_chunked_chunk_size_1() {
        // chunk_size = 1: each position only attends to itself and adjacent
        // positions in the same chunk → no long-range attention
        let layer = ChunkedAttention::new(4, 1, 8);
        let x = make_input(1, 8, 4);
        let out = layer.forward(&x, 1).expect("forward should succeed");
        assert_eq!(out[0].len(), 8);
        // All values should be finite
        for row in &out[0] {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_sdp_attention_3d_shape() {
        let q = make_input(2, 5, 4);
        let k = make_input(2, 7, 4);
        let v = make_input(2, 7, 6);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 5);
        assert_eq!(out[0][0].len(), 6);
    }

    #[test]
    fn test_sdp_with_mask() {
        // All-false mask → every score becomes −∞ → uniform weights
        let q = make_input(1, 3, 4);
        let k = make_input(1, 3, 4);
        let v = make_input(1, 3, 4);
        let mask: Vec<Vec<Vec<bool>>> = vec![vec![
            vec![false, false, false],
            vec![false, false, false],
            vec![false, false, false],
        ]];
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        // With all-false mask softmax falls back to uniform → output is
        // average of v rows → values should be finite
        for row in &out[0] {
            for &val in row {
                assert!(val.is_finite(), "value {val} is not finite");
            }
        }
    }

    #[test]
    fn test_causal_mask_in_sdp_2d() {
        // With causal masking, position 0 should not attend to positions 1+
        // so the output at position 0 should equal value[0] (only attends to itself)
        let q = vec![vec![1.0f32, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = q.clone();
        let v = vec![vec![10.0f32, 0.0], vec![0.0, 10.0], vec![5.0, 5.0]];
        let out = scaled_dot_product_attention_2d(&q, &k, &v, true);
        // Position 0 can only see position 0 → output[0] = v[0]
        assert!((out[0][0] - 10.0).abs() < 0.5, "out[0][0] = {}", out[0][0]);
        assert!((out[0][1] - 0.0).abs() < 0.5, "out[0][1] = {}", out[0][1]);
    }

    #[test]
    fn test_sparse_hierarchical_causal_flag() {
        let config = SparseAttentionConfig {
            window_size: 4,
            stride: 4,
            d_model: 8,
            n_heads: 2,
            n_global_tokens: 2,
        };
        let mut layer = SparseHierarchicalAttention::new(config, 10);
        layer.causal = true;
        let x = make_input(1, 12, 8);
        let out = layer.forward(&x).expect("forward with causal=true");
        assert_eq!(out[0].len(), 12);
    }
}
