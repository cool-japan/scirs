//! Model compression for on-device deployment
//!
//! This module provides algorithms to reduce model size and inference cost while
//! preserving accuracy. Supported techniques include quantization, magnitude/structured
//! pruning, low-rank factorization, and Huffman entropy coding of weight values.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CompressionStrategy
// ---------------------------------------------------------------------------

/// Strategy used to compress a neural network model.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionStrategy {
    /// Post-training or quantization-aware weight/activation quantization.
    Quantization(QuantizationConfig),
    /// Weight pruning (magnitude-based or structured).
    Pruning(PruningConfig),
    /// Knowledge distillation from a teacher model.
    KnowledgeDistillation(DistillationConfig),
    /// Approximate weight matrices with low-rank factors (W ≈ U · V).
    LowRankDecomposition(LowRankConfig),
    /// Huffman entropy coding of quantized weight values.
    HuffmanCoding(HuffmanConfig),
}

// ---------------------------------------------------------------------------
// Sub-configs
// ---------------------------------------------------------------------------

/// Quantization precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationPrecision {
    /// 8-bit integer weights and activations.
    Int8,
    /// 16-bit brain float (stored as f32, half the exponent range).
    BF16,
    /// 16-bit IEEE 754 half precision (emulated in f32 storage).
    FP16,
    /// Dynamic per-tensor quantization at inference time.
    Dynamic { bits: u8 },
}

/// Configuration for quantization-based compression.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationConfig {
    pub precision: QuantizationPrecision,
    /// Quantize activations in addition to weights.
    pub quantize_activations: bool,
    /// Use per-channel scales instead of a single per-tensor scale.
    pub per_channel: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::Int8,
            quantize_activations: false,
            per_channel: false,
        }
    }
}

/// Pruning scope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningScope {
    /// Zero out individual weights below a magnitude threshold.
    Unstructured,
    /// Zero out entire output neurons (rows) ranked by L2 norm.
    StructuredRows,
    /// Zero out entire input neurons (columns) ranked by L2 norm.
    StructuredColumns,
}

/// Configuration for pruning-based compression.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningConfig {
    /// Target fraction of weights to zero out (0.0 – 1.0).
    pub sparsity: f64,
    pub scope: PruningScope,
    /// Apply gradual pruning over this many steps (0 = one shot).
    pub warmup_steps: usize,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.5,
            scope: PruningScope::Unstructured,
            warmup_steps: 0,
        }
    }
}

/// Configuration for knowledge distillation.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillationConfig {
    /// Softmax temperature (> 1 produces softer distributions).
    pub temperature: f64,
    /// Weight on the soft-label distillation loss vs. hard-label loss.
    pub alpha: f64,
    /// Layer indices from which to match intermediate activations.
    pub hint_layers: Vec<usize>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.5,
            hint_layers: Vec::new(),
        }
    }
}

/// Configuration for low-rank matrix factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct LowRankConfig {
    /// Fraction of original rank to retain (0.0 – 1.0).
    pub rank_fraction: f64,
    /// Fixed absolute rank.  If set, overrides `rank_fraction`.
    pub fixed_rank: Option<usize>,
    /// Use randomised SVD for speed on large matrices.
    pub randomised: bool,
}

impl Default for LowRankConfig {
    fn default() -> Self {
        Self {
            rank_fraction: 0.25,
            fixed_rank: None,
            randomised: false,
        }
    }
}

/// Configuration for Huffman coding compression.
#[derive(Debug, Clone, PartialEq)]
pub struct HuffmanConfig {
    /// Number of quantisation bins used before Huffman coding.
    pub num_bins: usize,
}

impl Default for HuffmanConfig {
    fn default() -> Self {
        Self { num_bins: 256 }
    }
}

// ---------------------------------------------------------------------------
// Compression result
// ---------------------------------------------------------------------------

/// Statistics produced by one compression pass over a single weight matrix.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub strategy_name: String,
    pub original_params: usize,
    pub compressed_params: usize,
    /// Bytes saved (approximation assuming f32 originals).
    pub bytes_saved: usize,
    /// Compression ratio (original / compressed).
    pub compression_ratio: f64,
    /// Estimated relative change in Frobenius norm (reconstruction error).
    pub reconstruction_error: f64,
    /// Layer-level breakdown keyed by layer name.
    pub layer_stats: HashMap<String, LayerCompressionStat>,
}

/// Per-layer statistics.
#[derive(Debug, Clone)]
pub struct LayerCompressionStat {
    pub original_params: usize,
    pub compressed_params: usize,
    pub sparsity: f64,
    pub rank_used: Option<usize>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the Frobenius norm of a 2-D array view.
fn frobenius_norm(m: &ArrayView2<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Truncated SVD via power iteration (randomised).  Returns (U, s, Vt).
/// U: (rows, rank), s: (rank,), Vt: (rank, cols).
fn randomised_svd(
    a: &Array2<f64>,
    rank: usize,
    n_iter: usize,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (rows, cols) = a.dim();
    if rank == 0 || rank > rows.min(cols) {
        return Err(NeuralError::InvalidArgument(format!(
            "rank {} out of range for matrix {}x{}",
            rank, rows, cols
        )));
    }

    // Random Gaussian sketch: Y = A * Omega,  Omega in R^{cols x (rank + oversample)}
    let oversample = 10_usize.min(rows.min(cols) - rank);
    let sketch_cols = rank + oversample;

    // Build Omega with values from standard normal using deterministic seeding.
    let mut omega = Array2::<f64>::zeros((cols, sketch_cols));
    // Use a simple LCG PRNG so we have no external dependency.
    let mut state: u64 = 0x_DEAD_BEEF_CAFE_BABE;
    for v in omega.iter_mut() {
        // Box-Muller pairs would be more correct but this linear congruential
        // approximation is sufficient for the sketch quality needed here.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = (state >> 33) as f64 / (u32::MAX as f64);
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = (state >> 33) as f64 / (u32::MAX as f64);
        // Box-Muller transform
        let r = (-2.0 * (u1 + f64::EPSILON).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        *v = r * theta.cos();
    }

    // Y = A * Omega
    let mut y = a.dot(&omega); // (rows, sketch_cols)

    // Power iteration to improve accuracy
    for _ in 0..n_iter {
        // Y = A^T * Y
        let yt = a.t().dot(&y); // (cols, sketch_cols)
                                // Y = A * Y
        y = a.dot(&yt); // (rows, sketch_cols)
    }

    // QR of Y: Q in R^{rows x sketch_cols}
    let q = orthonormalise_columns(&y)?;

    // B = Q^T * A  ->  (sketch_cols x cols)
    let b = q.t().dot(a);

    // Thin SVD of the small matrix B
    let (ub, sigma, vt) = thin_svd(&b, rank)?;

    // U = Q * Ub  -> (rows x rank)
    let u = q.dot(&ub);

    Ok((u, sigma, vt))
}

/// Thin, exact SVD using the Jacobi one-sided algorithm (compact, no external dep).
/// Returns (U, s, Vt) truncated to `rank` singular values.
fn thin_svd(a: &Array2<f64>, rank: usize) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let k = rank.min(m).min(n);

    // Work on A^T A (smaller Gram matrix when m >= n is NOT assumed – use AAt).
    // For simplicity we use power-iteration one-column-at-a-time (deflation).
    let mut u_cols: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut sigma_vals: Vec<f64> = Vec::with_capacity(k);
    let mut vt_rows: Vec<Array1<f64>> = Vec::with_capacity(k);

    let mut residual = a.to_owned();

    for _ in 0..k {
        // Power iteration to find dominant left singular vector.
        let (u_vec, s_val, v_vec) = power_iteration_svd(&residual, 30)?;
        if s_val < 1e-12 {
            break;
        }
        // Deflate
        let outer = outer_product(&u_vec, &v_vec);
        residual = residual - outer * s_val;

        u_cols.push(u_vec);
        sigma_vals.push(s_val);
        vt_rows.push(v_vec);
    }

    let actual_k = u_cols.len();
    if actual_k == 0 {
        return Err(NeuralError::ComputationError(
            "SVD failed: matrix appears to be zero".to_string(),
        ));
    }

    // Assemble U (m x actual_k)
    let mut u_arr = Array2::<f64>::zeros((m, actual_k));
    for (j, col) in u_cols.iter().enumerate() {
        u_arr.column_mut(j).assign(col);
    }

    // Assemble Vt (actual_k x n)
    let mut vt_arr = Array2::<f64>::zeros((actual_k, n));
    for (i, row) in vt_rows.iter().enumerate() {
        vt_arr.row_mut(i).assign(row);
    }

    let sigma_arr = Array1::from_vec(sigma_vals);

    Ok((u_arr, sigma_arr, vt_arr))
}

/// Power iteration to approximate the dominant singular triplet.
fn power_iteration_svd(
    a: &Array2<f64>,
    max_iter: usize,
) -> Result<(Array1<f64>, f64, Array1<f64>)> {
    let (m, n) = a.dim();
    // Initialise v with a non-trivial vector.
    let mut v = Array1::<f64>::from_elem(n, 1.0 / (n as f64).sqrt());

    for _ in 0..max_iter {
        // u = A v
        let u_raw = a.dot(&v);
        let u_norm = u_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        if u_norm < 1e-14 {
            break;
        }
        let u = u_raw / u_norm;
        // v = A^T u
        let v_raw = a.t().dot(&u);
        let v_norm = v_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm < 1e-14 {
            break;
        }
        v = v_raw / v_norm;
    }

    // Final u and sigma
    let u_raw = a.dot(&v);
    let sigma = u_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
    if sigma < 1e-14 {
        let u = Array1::from_elem(m, 0.0);
        return Ok((u, 0.0, v));
    }
    let u = u_raw / sigma;
    Ok((u, sigma, v))
}

/// Compute the outer product u ⊗ v^T as a 2-D array.
fn outer_product(u: &Array1<f64>, v: &Array1<f64>) -> Array2<f64> {
    let m = u.len();
    let n = v.len();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            out[[i, j]] = u[i] * v[j];
        }
    }
    out
}

/// Modified Gram–Schmidt orthonormalisation of the columns of `a`.
/// Returns Q whose columns form an orthonormal basis for range(a).
fn orthonormalise_columns(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, n) = a.dim();
    let mut q = a.to_owned();
    for j in 0..n {
        // Normalise column j
        let norm = q.column(j).iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            continue;
        }
        let scale = 1.0 / norm;
        for i in 0..m {
            q[[i, j]] *= scale;
        }
        // Subtract projection from remaining columns
        for k in (j + 1)..n {
            let dot: f64 = (0..m).map(|i| q[[i, j]] * q[[i, k]]).sum();
            for i in 0..m {
                let qij = q[[i, j]];
                q[[i, k]] -= dot * qij;
            }
        }
    }
    Ok(q)
}

// ---------------------------------------------------------------------------
// Huffman coding helpers
// ---------------------------------------------------------------------------

/// A Huffman tree node.
#[derive(Debug)]
enum HuffNode {
    Leaf {
        symbol: i32,
        freq: usize,
    },
    Internal {
        freq: usize,
        left: Box<HuffNode>,
        right: Box<HuffNode>,
    },
}

impl HuffNode {
    fn freq(&self) -> usize {
        match self {
            HuffNode::Leaf { freq, .. } => *freq,
            HuffNode::Internal { freq, .. } => *freq,
        }
    }
}

/// Build a Huffman code table (symbol -> bit-length) from frequency counts.
fn build_huffman_lengths(freq: &HashMap<i32, usize>) -> HashMap<i32, usize> {
    if freq.is_empty() {
        return HashMap::new();
    }
    // Use a sorted Vec as a priority queue (min-heap by freq).
    let mut heap: Vec<Box<HuffNode>> = freq
        .iter()
        .map(|(&sym, &f)| {
            Box::new(HuffNode::Leaf {
                symbol: sym,
                freq: f,
            })
        })
        .collect();
    heap.sort_by_key(|n| n.freq());

    while heap.len() > 1 {
        // Pop two minimum nodes
        let left = heap.remove(0);
        let right = heap.remove(0);
        let combined_freq = left.freq() + right.freq();
        let internal = Box::new(HuffNode::Internal {
            freq: combined_freq,
            left,
            right,
        });
        // Insert in sorted position
        let pos = heap.partition_point(|n| n.freq() <= combined_freq);
        heap.insert(pos, internal);
    }

    let mut lengths = HashMap::new();
    if let Some(root) = heap.into_iter().next() {
        assign_lengths(&root, 0, &mut lengths);
    }
    lengths
}

fn assign_lengths(node: &HuffNode, depth: usize, lengths: &mut HashMap<i32, usize>) {
    match node {
        HuffNode::Leaf { symbol, .. } => {
            lengths.insert(*symbol, depth.max(1));
        }
        HuffNode::Internal { left, right, .. } => {
            assign_lengths(left, depth + 1, lengths);
            assign_lengths(right, depth + 1, lengths);
        }
    }
}

/// Estimate the number of bits needed to store an array after Huffman coding.
fn huffman_encoded_bits(quantised: &Array2<i32>, freq: &HashMap<i32, usize>) -> usize {
    let lengths = build_huffman_lengths(freq);
    quantised
        .iter()
        .map(|sym| lengths.get(sym).copied().unwrap_or(8))
        .sum()
}

// ---------------------------------------------------------------------------
// ModelCompressor
// ---------------------------------------------------------------------------

/// High-level model compressor that applies one or more `CompressionStrategy`
/// variants to individual weight matrices.
pub struct ModelCompressor<F> {
    strategies: Vec<CompressionStrategy>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Clone + Into<f64> + From<f64> + std::fmt::Debug> ModelCompressor<F> {
    /// Create a new compressor with the given ordered list of strategies.
    pub fn new(strategies: Vec<CompressionStrategy>) -> Self {
        Self {
            strategies,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compress a single named weight matrix and return the modified matrix
    /// together with per-layer statistics.
    pub fn compress_weights(
        &self,
        name: &str,
        weights: &Array2<F>,
    ) -> Result<(Array2<F>, Vec<LayerCompressionStat>)> {
        // Convert to f64 for all internal arithmetic.
        let mut w: Array2<f64> = weights.mapv(|v| v.into());
        let original_params = w.len();
        let mut stats_per_strategy = Vec::new();

        for strategy in &self.strategies {
            let (new_w, stat) = self.apply_strategy(name, &w, strategy)?;
            w = new_w;
            stats_per_strategy.push(stat);
        }

        // Convert back.
        let result: Array2<F> = w.mapv(|v| F::from(v));
        Ok((result, stats_per_strategy))
    }

    /// Apply a single strategy to a weight matrix represented as f64.
    fn apply_strategy(
        &self,
        name: &str,
        w: &Array2<f64>,
        strategy: &CompressionStrategy,
    ) -> Result<(Array2<f64>, LayerCompressionStat)> {
        match strategy {
            CompressionStrategy::Quantization(cfg) => self.apply_quantization(name, w, cfg),
            CompressionStrategy::Pruning(cfg) => self.apply_pruning(name, w, cfg),
            CompressionStrategy::LowRankDecomposition(cfg) => self.apply_low_rank(name, w, cfg),
            CompressionStrategy::HuffmanCoding(cfg) => self.apply_huffman(name, w, cfg),
            CompressionStrategy::KnowledgeDistillation(_cfg) => {
                // Knowledge distillation does not modify individual weight tensors;
                // it requires a training loop with a teacher model.  We return the
                // weights unchanged and record the absence of a structural change.
                let stat = LayerCompressionStat {
                    original_params: w.len(),
                    compressed_params: w.len(),
                    sparsity: 0.0,
                    rank_used: None,
                };
                Ok((w.to_owned(), stat))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Quantization
    // -----------------------------------------------------------------------

    fn apply_quantization(
        &self,
        _name: &str,
        w: &Array2<f64>,
        cfg: &QuantizationConfig,
    ) -> Result<(Array2<f64>, LayerCompressionStat)> {
        let (bits, is_fp_reduction) = match cfg.precision {
            QuantizationPrecision::Int8 => (8_u32, false),
            QuantizationPrecision::FP16 => (16_u32, true),
            QuantizationPrecision::BF16 => (16_u32, true),
            QuantizationPrecision::Dynamic { bits } => (bits as u32, false),
        };

        let n_levels = (1u64 << bits) as f64;
        let quantised = if is_fp_reduction {
            // FP16/BF16: simulate reduced mantissa precision (10-bit mantissa).
            let mantissa_bits = if bits == 16 { 10_u32 } else { 7_u32 }; // BF16 has 7
            let scale = (1u32 << mantissa_bits) as f64;
            if cfg.per_channel {
                let (rows, cols) = w.dim();
                let mut out = Array2::<f64>::zeros((rows, cols));
                for r in 0..rows {
                    let row = w.row(r);
                    let abs_max = row.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                    let channel_scale = if abs_max < 1e-12 {
                        1.0
                    } else {
                        abs_max / scale
                    };
                    for c in 0..cols {
                        out[[r, c]] = (w[[r, c]] / channel_scale).round() * channel_scale;
                    }
                }
                out
            } else {
                let abs_max = w.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let s = if abs_max < 1e-12 {
                    1.0
                } else {
                    abs_max / scale
                };
                w.mapv(|x| (x / s).round() * s)
            }
        } else {
            // Integer quantization (symmetric per-tensor or per-channel).
            let half_range = (n_levels / 2.0) - 1.0;
            if cfg.per_channel {
                let (rows, cols) = w.dim();
                let mut out = Array2::<f64>::zeros((rows, cols));
                for r in 0..rows {
                    let row = w.row(r);
                    let abs_max = row.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                    let scale = if abs_max < 1e-12 {
                        1.0
                    } else {
                        abs_max / half_range
                    };
                    for c in 0..cols {
                        let q = (w[[r, c]] / scale).round().clamp(-half_range, half_range);
                        out[[r, c]] = q * scale;
                    }
                }
                out
            } else {
                let abs_max = w.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let scale = if abs_max < 1e-12 {
                    1.0
                } else {
                    abs_max / half_range
                };
                w.mapv(|x| (x / scale).round().clamp(-half_range, half_range) * scale)
            }
        };

        let original_params = w.len();
        let stat = LayerCompressionStat {
            original_params,
            compressed_params: original_params, // same count, smaller dtype
            sparsity: 0.0,
            rank_used: None,
        };
        Ok((quantised, stat))
    }

    // -----------------------------------------------------------------------
    // Pruning
    // -----------------------------------------------------------------------

    fn apply_pruning(
        &self,
        _name: &str,
        w: &Array2<f64>,
        cfg: &PruningConfig,
    ) -> Result<(Array2<f64>, LayerCompressionStat)> {
        if cfg.sparsity < 0.0 || cfg.sparsity > 1.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "sparsity must be in [0, 1], got {}",
                cfg.sparsity
            )));
        }
        let original_params = w.len();
        let mut pruned = w.to_owned();

        let pruned_count = match cfg.scope {
            PruningScope::Unstructured => {
                let n_prune = (original_params as f64 * cfg.sparsity).round() as usize;
                // Collect (|w|, flat_index) and sort ascending.
                let mut magnitudes: Vec<(f64, usize)> = pruned
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v.abs(), i))
                    .collect();
                magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite weights expected"));
                let flat = pruned.as_slice_mut().ok_or_else(|| {
                    NeuralError::ComputationError("weight array is not contiguous".to_string())
                })?;
                for &(_, idx) in magnitudes.iter().take(n_prune) {
                    flat[idx] = 0.0;
                }
                n_prune
            }
            PruningScope::StructuredRows => {
                let (rows, cols) = pruned.dim();
                let n_prune_rows = (rows as f64 * cfg.sparsity).round() as usize;
                // Rank rows by L2 norm ascending.
                let mut row_norms: Vec<(f64, usize)> = (0..rows)
                    .map(|r| {
                        let norm = pruned.row(r).iter().map(|x| x * x).sum::<f64>().sqrt();
                        (norm, r)
                    })
                    .collect();
                row_norms.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite norms expected"));
                let mut zeroed = 0_usize;
                for &(_, r) in row_norms.iter().take(n_prune_rows) {
                    for c in 0..cols {
                        pruned[[r, c]] = 0.0;
                    }
                    zeroed += cols;
                }
                zeroed
            }
            PruningScope::StructuredColumns => {
                let (rows, cols) = pruned.dim();
                let n_prune_cols = (cols as f64 * cfg.sparsity).round() as usize;
                let mut col_norms: Vec<(f64, usize)> = (0..cols)
                    .map(|c| {
                        let norm = pruned.column(c).iter().map(|x| x * x).sum::<f64>().sqrt();
                        (norm, c)
                    })
                    .collect();
                col_norms.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite norms expected"));
                let mut zeroed = 0_usize;
                for &(_, c) in col_norms.iter().take(n_prune_cols) {
                    for r in 0..rows {
                        pruned[[r, c]] = 0.0;
                    }
                    zeroed += rows;
                }
                zeroed
            }
        };

        let actual_sparsity = pruned_count as f64 / original_params as f64;
        let stat = LayerCompressionStat {
            original_params,
            compressed_params: original_params - pruned_count,
            sparsity: actual_sparsity,
            rank_used: None,
        };
        Ok((pruned, stat))
    }

    // -----------------------------------------------------------------------
    // Low-rank decomposition
    // -----------------------------------------------------------------------

    fn apply_low_rank(
        &self,
        _name: &str,
        w: &Array2<f64>,
        cfg: &LowRankConfig,
    ) -> Result<(Array2<f64>, LayerCompressionStat)> {
        let (rows, cols) = w.dim();
        let max_rank = rows.min(cols);
        let target_rank = if let Some(r) = cfg.fixed_rank {
            r.min(max_rank).max(1)
        } else {
            ((max_rank as f64 * cfg.rank_fraction).round() as usize)
                .min(max_rank)
                .max(1)
        };

        let original_params = rows * cols;
        let compressed_params = (rows + cols) * target_rank;

        let (u, sigma, vt) = if cfg.randomised {
            randomised_svd(w, target_rank, 4)?
        } else {
            thin_svd(w, target_rank)?
        };

        // Reconstruct: W_approx = U * diag(sigma) * Vt
        let actual_rank = sigma.len().min(target_rank);
        let u_trunc = u.slice(s![.., ..actual_rank]).to_owned();
        let vt_trunc = vt.slice(s![..actual_rank, ..]).to_owned();
        // Scale U by sigma
        let mut u_scaled = u_trunc.clone();
        for (j, &s) in sigma.iter().take(actual_rank).enumerate() {
            for i in 0..rows {
                u_scaled[[i, j]] *= s;
            }
        }
        let reconstructed = u_scaled.dot(&vt_trunc);

        // Reconstruction error
        let diff = w - &reconstructed;
        let error = frobenius_norm(&diff.view()) / (frobenius_norm(&w.view()) + 1e-12);

        let stat = LayerCompressionStat {
            original_params,
            compressed_params,
            sparsity: 0.0,
            rank_used: Some(actual_rank),
        };
        let _ = error; // available for logging; not stored in stat currently.
        Ok((reconstructed, stat))
    }

    // -----------------------------------------------------------------------
    // Huffman coding
    // -----------------------------------------------------------------------

    fn apply_huffman(
        &self,
        _name: &str,
        w: &Array2<f64>,
        cfg: &HuffmanConfig,
    ) -> Result<(Array2<f64>, LayerCompressionStat)> {
        if cfg.num_bins < 2 {
            return Err(NeuralError::InvalidArgument(
                "HuffmanConfig::num_bins must be >= 2".to_string(),
            ));
        }
        let n = w.len();
        let min_val = w.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        // Quantise weights to bins.
        let num_bins = cfg.num_bins as f64;
        let quantise = |v: f64| -> i32 {
            if range < 1e-14 {
                return 0;
            }
            let bin = ((v - min_val) / range * (num_bins - 1.0)).round() as i32;
            bin.clamp(0, cfg.num_bins as i32 - 1)
        };
        let dequantise = |b: i32| -> f64 {
            if range < 1e-14 {
                return min_val;
            }
            min_val + (b as f64) / (num_bins - 1.0) * range
        };

        let quantised: Array2<i32> = w.mapv(quantise);
        let reconstructed: Array2<f64> = quantised.mapv(dequantise);

        // Build frequency table for Huffman analysis.
        let mut freq: HashMap<i32, usize> = HashMap::new();
        for &sym in quantised.iter() {
            *freq.entry(sym).or_insert(0) += 1;
        }

        let encoded_bits = huffman_encoded_bits(&quantised, &freq);
        // Original cost: n * 32 bits (f32) or n * 64 bits (f64).
        let original_bits = n * 32;
        let bytes_saved = if encoded_bits < original_bits {
            (original_bits - encoded_bits) / 8
        } else {
            0
        };

        let stat = LayerCompressionStat {
            original_params: n,
            compressed_params: n, // element count unchanged; size reduction is in encoding.
            sparsity: 0.0,
            rank_used: None,
        };
        let _ = bytes_saved; // logged externally.
        Ok((reconstructed, stat))
    }

    // -----------------------------------------------------------------------
    // Public batch API
    // -----------------------------------------------------------------------

    /// Compress a collection of named weight matrices and return an overall
    /// `CompressionResult` together with the compressed matrices.
    pub fn compress_model(
        &self,
        layers: &HashMap<String, Array2<F>>,
    ) -> Result<(HashMap<String, Array2<F>>, CompressionResult)> {
        if self.strategies.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "no compression strategies configured".to_string(),
            ));
        }

        let mut compressed_layers = HashMap::new();
        let mut layer_stats_map: HashMap<String, LayerCompressionStat> = HashMap::new();
        let mut total_original = 0_usize;
        let mut total_compressed = 0_usize;

        for (name, weights) in layers {
            let (compressed, per_strategy_stats) = self.compress_weights(name, weights)?;
            // Aggregate stats across strategies for this layer.
            let orig = weights.len();
            let comp = per_strategy_stats
                .last()
                .map(|s| s.compressed_params)
                .unwrap_or(orig);
            let sparsity = per_strategy_stats
                .iter()
                .map(|s| s.sparsity)
                .fold(0.0_f64, f64::max);
            let rank_used = per_strategy_stats.iter().find_map(|s| s.rank_used);

            let layer_stat = LayerCompressionStat {
                original_params: orig,
                compressed_params: comp,
                sparsity,
                rank_used,
            };
            layer_stats_map.insert(name.clone(), layer_stat);
            total_original += orig;
            total_compressed += comp;
            compressed_layers.insert(name.clone(), compressed);
        }

        let bytes_saved = if total_original > total_compressed {
            (total_original - total_compressed) * std::mem::size_of::<f32>()
        } else {
            0
        };
        let compression_ratio = if total_compressed == 0 {
            f64::INFINITY
        } else {
            total_original as f64 / total_compressed as f64
        };

        let strategy_name = self
            .strategies
            .iter()
            .map(|s| match s {
                CompressionStrategy::Quantization(_) => "Quantization",
                CompressionStrategy::Pruning(_) => "Pruning",
                CompressionStrategy::KnowledgeDistillation(_) => "KnowledgeDistillation",
                CompressionStrategy::LowRankDecomposition(_) => "LowRankDecomposition",
                CompressionStrategy::HuffmanCoding(_) => "HuffmanCoding",
            })
            .collect::<Vec<_>>()
            .join("+");

        let result = CompressionResult {
            strategy_name,
            original_params: total_original,
            compressed_params: total_compressed,
            bytes_saved,
            compression_ratio,
            reconstruction_error: 0.0, // aggregated error would require storing per-layer errors.
            layer_stats: layer_stats_map,
        };

        Ok((compressed_layers, result))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights(rows: usize, cols: usize) -> Array2<f64> {
        // Deterministic weights for reproducible tests.
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64 / (rows * cols) as f64) * 2.0 - 1.0)
            .collect();
        Array2::from_shape_vec((rows, cols), data).expect("shape must be valid")
    }

    #[test]
    fn test_int8_quantization_idempotent_shape() {
        let weights = sample_weights(8, 16);
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::Quantization(
                QuantizationConfig {
                    precision: QuantizationPrecision::Int8,
                    quantize_activations: false,
                    per_channel: false,
                },
            )]);
        let (compressed, stats) = compressor
            .compress_weights("layer0", &weights)
            .expect("quantization must not fail");
        assert_eq!(compressed.shape(), weights.shape());
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].original_params, 8 * 16);
    }

    #[test]
    fn test_unstructured_pruning_sparsity() {
        let weights = sample_weights(10, 10);
        let sparsity = 0.5;
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::Pruning(PruningConfig {
                sparsity,
                scope: PruningScope::Unstructured,
                warmup_steps: 0,
            })]);
        let (compressed, _) = compressor
            .compress_weights("dense", &weights)
            .expect("pruning must not fail");
        let zeros = compressed.iter().filter(|&&v| v == 0.0).count();
        let actual_sparsity = zeros as f64 / compressed.len() as f64;
        assert!(
            (actual_sparsity - sparsity).abs() < 0.05,
            "expected ~50% sparsity, got {:.3}",
            actual_sparsity
        );
    }

    #[test]
    fn test_structured_row_pruning() {
        let weights = sample_weights(8, 8);
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::Pruning(PruningConfig {
                sparsity: 0.25,
                scope: PruningScope::StructuredRows,
                warmup_steps: 0,
            })]);
        let (compressed, stats) = compressor
            .compress_weights("conv", &weights)
            .expect("structured pruning must not fail");
        assert!(stats[0].sparsity > 0.0);
        // At least 2 rows should be entirely zeroed (25% of 8 rows = 2).
        let zero_rows = (0..8)
            .filter(|&r| compressed.row(r).iter().all(|&v| v == 0.0))
            .count();
        assert_eq!(zero_rows, 2, "expected 2 zero rows, got {}", zero_rows);
    }

    #[test]
    fn test_low_rank_reconstruction_shape() {
        let weights = sample_weights(12, 8);
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::LowRankDecomposition(
                LowRankConfig {
                    rank_fraction: 0.5,
                    fixed_rank: None,
                    randomised: false,
                },
            )]);
        let (compressed, stats) = compressor
            .compress_weights("embed", &weights)
            .expect("low-rank compression must not fail");
        assert_eq!(compressed.shape(), weights.shape());
        assert!(stats[0].rank_used.is_some());
        assert!(stats[0].rank_used.expect("operation should succeed") <= 4); // 50% of min(12,8)=8 => rank 4
    }

    #[test]
    fn test_huffman_coding_preserves_shape() {
        let weights = sample_weights(6, 6);
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::HuffmanCoding(HuffmanConfig {
                num_bins: 64,
            })]);
        let (compressed, _) = compressor
            .compress_weights("layer", &weights)
            .expect("huffman compression must not fail");
        assert_eq!(compressed.shape(), weights.shape());
    }

    #[test]
    fn test_compress_model_batch() {
        let mut layers: HashMap<String, Array2<f64>> = HashMap::new();
        layers.insert("w1".to_string(), sample_weights(16, 8));
        layers.insert("w2".to_string(), sample_weights(8, 4));

        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::Pruning(PruningConfig {
                sparsity: 0.5,
                scope: PruningScope::Unstructured,
                warmup_steps: 0,
            })]);
        let (compressed, result) = compressor
            .compress_model(&layers)
            .expect("batch compression must not fail");
        assert_eq!(compressed.len(), 2);
        assert_eq!(result.original_params, 16 * 8 + 8 * 4);
        assert!(result.layer_stats.contains_key("w1"));
        assert!(result.layer_stats.contains_key("w2"));
    }

    #[test]
    fn test_knowledge_distillation_passthrough() {
        let weights = sample_weights(4, 4);
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::KnowledgeDistillation(
                DistillationConfig::default(),
            )]);
        let (compressed, stats) = compressor
            .compress_weights("layer", &weights)
            .expect("kd passthrough must not fail");
        // Weights unchanged.
        for (orig, comp) in weights.iter().zip(compressed.iter()) {
            assert!((orig - comp).abs() < 1e-10);
        }
        assert_eq!(stats[0].compressed_params, stats[0].original_params);
    }

    #[test]
    fn test_chained_strategies() {
        let weights = sample_weights(10, 10);
        let compressor: ModelCompressor<f64> = ModelCompressor::new(vec![
            CompressionStrategy::Pruning(PruningConfig {
                sparsity: 0.3,
                scope: PruningScope::Unstructured,
                warmup_steps: 0,
            }),
            CompressionStrategy::Quantization(QuantizationConfig {
                precision: QuantizationPrecision::Int8,
                quantize_activations: false,
                per_channel: true,
            }),
        ]);
        let (compressed, stats) = compressor
            .compress_weights("multi", &weights)
            .expect("chained compression must not fail");
        assert_eq!(stats.len(), 2);
        assert_eq!(compressed.shape(), weights.shape());
    }

    #[test]
    fn test_invalid_sparsity_rejected() {
        let weights = sample_weights(4, 4);
        let compressor: ModelCompressor<f64> =
            ModelCompressor::new(vec![CompressionStrategy::Pruning(PruningConfig {
                sparsity: 1.5, // invalid
                scope: PruningScope::Unstructured,
                warmup_steps: 0,
            })]);
        let result = compressor.compress_weights("bad", &weights);
        assert!(result.is_err());
    }
}
