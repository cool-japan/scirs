//! Wavelet Packet Transform (WPT)
//!
//! Wavelet packets generalize the discrete wavelet transform by allowing full
//! decomposition of both approximation and detail subbands at each level.
//! This produces a complete binary tree of subband coefficients.
//!
//! The best-basis algorithm (Coifman–Wickerhauser 1992) selects an optimal
//! orthonormal basis from the packet tree by minimising an additive cost function
//! (e.g. Shannon entropy or log-energy).
//!
//! # References
//! - Coifman, R.R. & Wickerhauser, M.V. (1992). Entropy-based algorithms for best
//!   basis selection. IEEE Trans. Inf. Theory, 38(2), 713–718.
//! - Mallat, S. (1999). A Wavelet Tour of Signal Processing. Academic Press.

use std::collections::HashMap;
use std::f64::consts::LN_2;

use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
// Wavelet filter definitions
// ─────────────────────────────────────────────────────────────────────────────

/// Supported orthonormal wavelet families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Wavelet {
    /// Haar wavelet (db1)
    Haar,
    /// Daubechies 4-tap (db2)
    Db2,
    /// Daubechies 6-tap (db3)
    Db3,
    /// Daubechies 8-tap (db4)
    Db4,
    /// Daubechies 10-tap (db5)
    Db5,
    /// Symlet 4-tap (sym2)
    Sym2,
    /// Symlet 8-tap (sym4)
    Sym4,
    /// Coiflet 6-tap (coif1)
    Coif1,
    /// Biorthogonal 2.2 (bior2.2) – for approximation only; analysis filters
    Bior22,
}

/// Low-pass (scaling) and high-pass (wavelet) analysis filters for a wavelet.
#[derive(Debug, Clone)]
pub struct WaveletFilters {
    /// Low-pass decomposition filter h₀
    pub lo_d: Vec<f64>,
    /// High-pass decomposition filter h₁
    pub hi_d: Vec<f64>,
    /// Low-pass reconstruction filter g₀
    pub lo_r: Vec<f64>,
    /// High-pass reconstruction filter g₁
    pub hi_r: Vec<f64>,
}

impl WaveletFilters {
    /// Return filters for the given wavelet.
    pub fn for_wavelet(w: Wavelet) -> Self {
        match w {
            Wavelet::Haar => {
                let s = 1.0_f64 / 2.0_f64.sqrt();
                let lo = vec![s, s];
                let hi = vec![s, -s];
                // For orthogonal wavelets with the transpose synthesis formula,
                // the synthesis filters equal the analysis filters (lo_r = lo_d, hi_r = hi_d).
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Db2 => {
                let s3 = 3.0_f64.sqrt();
                let norm = 4.0 * 2.0_f64.sqrt(); // 4*sqrt(2)
                let lo = vec![
                    (1.0 + s3) / norm,
                    (3.0 + s3) / norm,
                    (3.0 - s3) / norm,
                    (1.0 - s3) / norm,
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Db3 => {
                // Daubechies db3 (6-tap) coefficients
                let lo = vec![
                    0.035226291882100656,
                    -0.08544127388202666,
                    -0.13501102001039084,
                    0.4598775021193313,
                    0.8068915093133388,
                    0.3326705529509569,
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Db4 => {
                // Daubechies db4 (8-tap) coefficients
                let lo = vec![
                    -0.010597401784997278,
                    0.032883011666982945,
                    0.030841381835986965,
                    -0.18703481171888114,
                    -0.027_983_769_416_983_85,
                    0.6308807679295904,
                    0.7148465705525415,
                    0.23037781330885523,
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Db5 => {
                // Daubechies db5 (10-tap) coefficients
                let lo = vec![
                    0.003335725285001549,
                    -0.012580751999015526,
                    -0.006241490213011705,
                    0.07757149384006515,
                    -0.03224486958502952,
                    -0.24229488706619015,
                    0.13842814590110342,
                    0.7243085284377729,
                    0.6038292697974729,
                    0.160102397974125,
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Sym2 => {
                // Symlet sym2 = db2 (same energy, different phase)
                let s3 = 3.0_f64.sqrt();
                let lo = vec![
                    (1.0 - s3) / 8.0_f64.sqrt(),
                    (3.0 - s3) / 8.0_f64.sqrt(),
                    (3.0 + s3) / 8.0_f64.sqrt(),
                    (1.0 + s3) / 8.0_f64.sqrt(),
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Sym4 => {
                // Symlet sym4 (8-tap)
                let lo = vec![
                    -0.07576571478927333,
                    -0.02963552764599851,
                    0.49761866763201545,
                    0.8037387518059161,
                    0.29785779560527736,
                    -0.09921954357684722,
                    -0.012603967262037833,
                    0.032_223_100_604_042_7,
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Coif1 => {
                // Coiflet coif1 (6-tap)
                let lo = vec![
                    -0.015655728135960927,
                    -0.07273261951285047,
                    0.3848648565381134,
                    0.8525720202122554,
                    0.3378976624578092,
                    -0.07273261951285047,
                ];
                let hi = qmf_hi(&lo);
                // Orthogonal wavelet: synthesis filters equal analysis filters for transpose synthesis.
                let lo_r = lo.clone();
                let hi_r = hi.clone();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
            Wavelet::Bior22 => {
                // Biorthogonal 2.2 analysis filters
                let lo = vec![-0.125, 0.25, 0.75, 0.25, -0.125];
                let hi = vec![-0.25, 0.5, -0.25];
                let lo_r: Vec<f64> = lo.iter().rev().cloned().collect();
                let hi_r: Vec<f64> = hi.iter().rev().cloned().collect();
                WaveletFilters {
                    lo_d: lo,
                    hi_d: hi,
                    lo_r,
                    hi_r,
                }
            }
        }
    }
}

/// Build the high-pass QMF filter from a low-pass filter.
///
/// h₁[n] = (-1)^n · h₀[L-1-n]
fn qmf_hi(lo: &[f64]) -> Vec<f64> {
    let n = lo.len();
    lo.iter()
        .rev()
        .enumerate()
        .map(|(k, &v)| if (n - 1 - k) % 2 == 0 { v } else { -v })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core convolution / subsampling helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve `signal` with `filter` using periodic (circular) boundary extension
/// and then down-sample by 2 (keep even-indexed samples).
///
/// The output length is `ceil(signal.len() / 2)` = `(signal.len() + 1) / 2`.
/// Using circular (periodic) boundary means the output length depends only on
/// the signal length, not the filter length.
fn conv_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let out_len = n.div_ceil(2); // ceil(n/2), independent of filter length
    let mut out = vec![0.0_f64; out_len];
    for k in 0..out_len {
        let t = 2 * k;
        let mut acc = 0.0_f64;
        for (j, &h) in filter.iter().enumerate() {
            // periodic (circular) boundary
            let idx = ((t as isize - j as isize).rem_euclid(n as isize)) as usize;
            acc += signal[idx] * h;
        }
        out[k] = acc;
    }
    out
}

/// Synthesis step: transpose of `conv_downsample` with periodic (circular) boundary.
///
/// This is the adjoint (transpose) of the analysis step, ensuring perfect
/// reconstruction for orthogonal wavelets.  The formula is:
///
///   xhat[n] = Σₖ input[k] · filter[(2k − n) mod target_len]
///
/// where the modular index is only applied when it falls within the filter support
/// `[0, filter.len())`.
///
/// `target_len` must equal the length of the signal that was passed to
/// `conv_downsample` to produce `input`.
fn synthesis_step(input: &[f64], filter: &[f64], target_len: usize) -> Vec<f64> {
    let n_in = input.len();
    let flen = filter.len();
    let n_out = target_len;
    let mut out = vec![0.0_f64; n_out];
    for n_idx in 0..n_out {
        let mut acc = 0.0_f64;
        for k in 0..n_in {
            let j = ((2 * k as isize - n_idx as isize).rem_euclid(n_out as isize)) as usize;
            if j < flen {
                acc += input[k] * filter[j];
            }
        }
        out[n_idx] = acc;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Node & tree structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the wavelet packet tree.
///
/// The node stores the subband coefficients and its position in the tree.
/// Position is identified by `(level, index)` where `index ∈ [0, 2^level)`.
#[derive(Debug, Clone)]
pub struct WaveletPacketNode {
    /// Subband coefficients at this node.
    pub coeffs: Vec<f64>,
    /// Decomposition level (0 = root, i.e. the original signal).
    pub level: usize,
    /// Node index within the level (frequency-ordered).
    pub index: usize,
}

impl WaveletPacketNode {
    /// Create a new node.
    pub fn new(coeffs: Vec<f64>, level: usize, index: usize) -> Self {
        WaveletPacketNode {
            coeffs,
            level,
            index,
        }
    }

    /// Returns `true` if this node is the root (level 0).
    pub fn is_root(&self) -> bool {
        self.level == 0
    }

    /// Flat key used for `HashMap` storage: `level * OFFSET + index`.
    fn key(level: usize, index: usize) -> u64 {
        (level as u64) << 32 | (index as u64)
    }
}

/// A full binary tree of wavelet packet nodes.
///
/// Nodes are stored in a `HashMap` keyed by `(level, index)`.
/// The tree is built by `wpd` and each interior node stores *both* the
/// node's own coefficients and its children (low/high subbands).
#[derive(Debug, Clone)]
pub struct WaveletPacketTree {
    /// All computed nodes, keyed by `WaveletPacketNode::key(level, index)`.
    nodes: HashMap<u64, WaveletPacketNode>,
    /// Maximum decomposition depth.
    pub max_level: usize,
    /// Wavelet used to build this tree.
    pub wavelet: Wavelet,
    /// Length of the original signal (needed for reconstruction).
    pub signal_len: usize,
}

impl WaveletPacketTree {
    /// Create an empty tree.
    pub fn new(wavelet: Wavelet, max_level: usize, signal_len: usize) -> Self {
        WaveletPacketTree {
            nodes: HashMap::new(),
            max_level,
            wavelet,
            signal_len,
        }
    }

    /// Insert a node into the tree.
    pub fn insert(&mut self, node: WaveletPacketNode) {
        let key = WaveletPacketNode::key(node.level, node.index);
        self.nodes.insert(key, node);
    }

    /// Retrieve a node by `(level, index)`.
    pub fn get(&self, level: usize, index: usize) -> Option<&WaveletPacketNode> {
        self.nodes.get(&WaveletPacketNode::key(level, index))
    }

    /// Iterate over all nodes at a given `level`.
    pub fn nodes_at_level(&self, level: usize) -> impl Iterator<Item = &WaveletPacketNode> {
        self.nodes.values().filter(move |n| n.level == level)
    }

    /// All nodes in the tree.
    pub fn all_nodes(&self) -> impl Iterator<Item = &WaveletPacketNode> {
        self.nodes.values()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wavelet Packet Decomposition (WPD)
// ─────────────────────────────────────────────────────────────────────────────

/// Perform a full wavelet packet decomposition up to `max_level`.
///
/// Every node (approximation *and* detail) at every level is recursively
/// decomposed, producing a complete binary tree with `2^(max_level+1) - 1` nodes.
///
/// # Arguments
///
/// * `signal`    – Real-valued input signal.
/// * `wavelet`   – Wavelet to use (determines analysis filters).
/// * `max_level` – Maximum decomposition depth.  The root is level 0.
///
/// # Errors
///
/// Returns `FFTError::ValueError` if `signal` is empty or `max_level == 0`.
///
/// # Example
///
/// ```
/// use scirs2_fft::wavelet_packets::{wpd, Wavelet};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
/// let tree = wpd(&signal, Wavelet::Db4, 3).expect("decomposition failed");
/// // Tree has nodes at levels 0 through 3
/// assert!(tree.get(0, 0).is_some());
/// assert!(tree.get(3, 7).is_some());
/// ```
pub fn wpd(signal: &[f64], wavelet: Wavelet, max_level: usize) -> FFTResult<WaveletPacketTree> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("signal must be non-empty".to_string()));
    }
    if max_level == 0 {
        return Err(FFTError::ValueError("max_level must be >= 1".to_string()));
    }

    let filters = WaveletFilters::for_wavelet(wavelet);
    let signal_len = signal.len();
    let mut tree = WaveletPacketTree::new(wavelet, max_level, signal_len);

    // Root node (level 0, index 0) = original signal
    tree.insert(WaveletPacketNode::new(signal.to_vec(), 0, 0));

    // BFS decomposition
    for level in 0..max_level {
        let num_nodes = 1_usize << level;
        for index in 0..num_nodes {
            let coeffs = match tree.get(level, index) {
                Some(n) => n.coeffs.clone(),
                None => {
                    return Err(FFTError::InternalError(format!(
                        "missing node ({level}, {index})"
                    )))
                }
            };

            // Low-pass child (approximation) → (level+1, 2*index)
            let lo = conv_downsample(&coeffs, &filters.lo_d);
            tree.insert(WaveletPacketNode::new(lo, level + 1, 2 * index));

            // High-pass child (detail) → (level+1, 2*index+1)
            let hi = conv_downsample(&coeffs, &filters.hi_d);
            tree.insert(WaveletPacketNode::new(hi, level + 1, 2 * index + 1));
        }
    }

    Ok(tree)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cost functions
// ─────────────────────────────────────────────────────────────────────────────

/// Shannon entropy cost function.
///
/// E(s) = -∑ |s_i|² log₂(|s_i|²)
///
/// Zero coefficients are excluded from the sum (lim_{p→0} p log p = 0).
///
/// # Example
///
/// ```
/// use scirs2_fft::wavelet_packets::shannon_entropy;
///
/// let coeffs = vec![0.5, -0.5, 0.5, -0.5];
/// let e = shannon_entropy(&coeffs);
/// assert!(e >= 0.0);
/// ```
pub fn shannon_entropy(coeffs: &[f64]) -> f64 {
    coeffs
        .iter()
        .filter_map(|&c| {
            let p = c * c;
            if p > 0.0 {
                Some(-p * p.log2())
            } else {
                None
            }
        })
        .sum()
}

/// Log-energy entropy cost function.
///
/// E(s) = ∑ log(|s_i|²)   (non-zero coefficients only)
pub fn log_energy_entropy(coeffs: &[f64]) -> f64 {
    coeffs
        .iter()
        .filter_map(|&c| {
            let p = c * c;
            if p > 0.0 {
                Some(p.ln() / LN_2)
            } else {
                None
            }
        })
        .sum()
}

/// Lp-norm (p ≠ 2) cost function – measures sparsity.
///
/// E(s) = ∑ |s_i|^p
pub fn lp_norm_cost(coeffs: &[f64], p: f64) -> f64 {
    coeffs.iter().map(|&c| c.abs().powf(p)).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Best Basis Selection (Coifman–Wickerhauser)
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best orthonormal basis from a wavelet packet tree.
///
/// The algorithm minimises an additive cost function `cost_fn` using a
/// bottom-up pass: a parent node is kept when its cost is *less than or equal*
/// to the sum of the costs of its two children.
///
/// # Arguments
///
/// * `tree`    – Packet tree produced by `wpd`.
/// * `cost_fn` – Additive cost function; must satisfy `cost(A∪B) = cost(A) + cost(B)`.
///
/// # Returns
///
/// A `Vec<WaveletPacketNode>` that forms a partition of the time-frequency
/// plane (i.e. a valid orthonormal basis).
///
/// # Errors
///
/// Returns `FFTError::ValueError` if the tree is empty.
///
/// # Example
///
/// ```
/// use scirs2_fft::wavelet_packets::{wpd, best_basis, shannon_entropy, Wavelet};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
/// let tree = wpd(&signal, Wavelet::Haar, 3).expect("decomp");
/// let basis = best_basis(&tree, shannon_entropy).expect("basis");
/// assert!(!basis.is_empty());
/// ```
pub fn best_basis<F>(tree: &WaveletPacketTree, cost_fn: F) -> FFTResult<Vec<WaveletPacketNode>>
where
    F: Fn(&[f64]) -> f64,
{
    if tree.max_level == 0 {
        return Err(FFTError::ValueError("tree is empty".to_string()));
    }

    // Pre-compute costs for every node in the tree
    let mut costs: HashMap<u64, f64> = HashMap::new();
    for node in tree.all_nodes() {
        let key = WaveletPacketNode::key(node.level, node.index);
        costs.insert(key, cost_fn(&node.coeffs));
    }

    // best_flag[key] = true  →  keep this node (do NOT split)
    let mut best_flag: HashMap<u64, bool> = HashMap::new();

    // Bottom-up: iterate from max_level - 1 down to 0
    for level in (0..tree.max_level).rev() {
        let num_nodes = 1_usize << level;
        for index in 0..num_nodes {
            let parent_key = WaveletPacketNode::key(level, index);
            let left_key = WaveletPacketNode::key(level + 1, 2 * index);
            let right_key = WaveletPacketNode::key(level + 1, 2 * index + 1);

            let parent_cost = match costs.get(&parent_key) {
                Some(&c) => c,
                None => continue,
            };

            // Children cost is the sum; if a child is already "split", we use
            // the *best* cost that the subtree achieves (propagated upward).
            let left_cost = effective_cost(&costs, &best_flag, level + 1, 2 * index);
            let right_cost = effective_cost(&costs, &best_flag, level + 1, 2 * index + 1);
            let children_cost = left_cost + right_cost;

            if parent_cost <= children_cost {
                // Parent is better (or equal) → keep parent, prune children
                best_flag.insert(parent_key, false); // false = "not split"
                costs.insert(parent_key, parent_cost);
            } else {
                // Split is better → mark parent as "split"
                best_flag.insert(parent_key, true);
                // Update the effective cost of this node to the children sum
                // so grandparents can compare correctly
                costs.insert(parent_key, children_cost);
            }

            // Ensure leaf flags exist for the children (they have no children of their own)
            best_flag.entry(left_key).or_insert(false);
            best_flag.entry(right_key).or_insert(false);
        }
    }

    // Collect the basis by selecting nodes that are NOT split
    let mut basis: Vec<WaveletPacketNode> = Vec::new();
    collect_basis(tree, &best_flag, 0, 0, &mut basis)?;

    Ok(basis)
}

/// Recursively collect basis nodes starting from `(level, index)`.
fn collect_basis(
    tree: &WaveletPacketTree,
    best_flag: &HashMap<u64, bool>,
    level: usize,
    index: usize,
    out: &mut Vec<WaveletPacketNode>,
) -> FFTResult<()> {
    let key = WaveletPacketNode::key(level, index);
    let is_split = best_flag.get(&key).copied().unwrap_or(false);

    if !is_split || level == tree.max_level {
        // Leaf of best basis tree
        if let Some(node) = tree.get(level, index) {
            out.push(node.clone());
        }
    } else {
        collect_basis(tree, best_flag, level + 1, 2 * index, out)?;
        collect_basis(tree, best_flag, level + 1, 2 * index + 1, out)?;
    }
    Ok(())
}

/// Return the effective (post-best-basis) cost for a node.
fn effective_cost(
    costs: &HashMap<u64, f64>,
    best_flag: &HashMap<u64, bool>,
    level: usize,
    index: usize,
) -> f64 {
    // If the node has already been processed (and possibly "split"), its
    // cost in the map already reflects the best achievable cost.
    let key = WaveletPacketNode::key(level, index);
    costs.get(&key).copied().unwrap_or(f64::INFINITY)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct the signal from a set of wavelet packet nodes forming a basis.
///
/// The nodes must constitute a valid partition of the time-frequency plane
/// (e.g. those returned by `best_basis`).  Mixed-level bases (where some nodes
/// are at depth 2 and others at depth 3, etc.) are fully supported.
///
/// # Arguments
///
/// * `tree`       – The original packet tree (provides wavelet & signal length).
/// * `basis_nodes` – A valid wavelet packet basis (partition of the root).
///
/// # Errors
///
/// Returns `FFTError::InternalError` if reconstruction encounters a missing node.
///
/// # Example
///
/// ```
/// use scirs2_fft::wavelet_packets::{wpd, best_basis, wp_reconstruct, shannon_entropy, Wavelet};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
/// let tree = wpd(&signal, Wavelet::Haar, 3).expect("decomp");
/// let basis = best_basis(&tree, shannon_entropy).expect("basis");
/// let recon = wp_reconstruct(&tree, &basis).expect("recon");
/// assert_eq!(recon.len(), signal.len());
/// // Perfect reconstruction (approx)
/// for (a, b) in signal.iter().zip(recon.iter()) {
///     assert!((a - b).abs() < 1e-10, "mismatch: {} vs {}", a, b);
/// }
/// ```
pub fn wp_reconstruct(
    tree: &WaveletPacketTree,
    basis_nodes: &[WaveletPacketNode],
) -> FFTResult<Vec<f64>> {
    if basis_nodes.is_empty() {
        return Err(FFTError::ValueError(
            "basis_nodes must be non-empty".to_string(),
        ));
    }

    let filters = WaveletFilters::for_wavelet(tree.wavelet);

    // Map each basis node into the tree storage so we can do upward synthesis
    let mut node_map: HashMap<u64, Vec<f64>> = HashMap::new();
    for node in basis_nodes {
        let key = WaveletPacketNode::key(node.level, node.index);
        node_map.insert(key, node.coeffs.clone());
    }

    // We need to know the coefficient length at each level.
    // We reuse the already-computed nodes in the tree.
    // Bottom-up synthesis from max_level to level 0
    for level in (1..=tree.max_level).rev() {
        let num_nodes = 1_usize << level;
        let parent_level = level - 1;
        let num_parents = 1_usize << parent_level;

        for p_idx in 0..num_parents {
            let left_key = WaveletPacketNode::key(level, 2 * p_idx);
            let right_key = WaveletPacketNode::key(level, 2 * p_idx + 1);
            let parent_key = WaveletPacketNode::key(parent_level, p_idx);

            // Skip if parent already exists in the basis (it was a leaf)
            if node_map.contains_key(&parent_key) {
                continue;
            }

            // Both children must be present to reconstruct the parent
            let left_coeffs = match node_map.get(&left_key) {
                Some(c) => c.clone(),
                None => continue,
            };
            let right_coeffs = match node_map.get(&right_key) {
                Some(c) => c.clone(),
                None => continue,
            };

            // Target length: get from the tree if available, else estimate
            let target_len = tree
                .get(parent_level, p_idx)
                .map(|n| n.coeffs.len())
                .unwrap_or_else(|| {
                    // Estimate: parent length ≈ 2 * child length
                    left_coeffs.len() * 2
                });

            // Synthesis: lo branch + hi branch (transpose of analysis)
            let lo_rec = synthesis_step(&left_coeffs, &filters.lo_r, target_len);
            let hi_rec = synthesis_step(&right_coeffs, &filters.hi_r, target_len);
            let parent_coeffs: Vec<f64> = lo_rec
                .iter()
                .zip(hi_rec.iter())
                .map(|(a, b)| a + b)
                .collect();

            node_map.insert(parent_key, parent_coeffs);
        }

        // We no longer need the children at this level to save memory
        for idx in 0..num_nodes {
            // Only remove if both siblings have been consumed
            let left_key = WaveletPacketNode::key(level, idx);
            // Keep it if still needed (might be a basis leaf)
            let _ = left_key;
        }
    }

    // The reconstructed signal is the root (level 0, index 0)
    let root_key = WaveletPacketNode::key(0, 0);
    node_map.remove(&root_key).ok_or_else(|| {
        FFTError::InternalError("reconstruction failed: root not reached".to_string())
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// WPT Denoising
// ─────────────────────────────────────────────────────────────────────────────

/// Thresholding method for wavelet denoising.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Hard thresholding: coefficients with |c| < τ are set to 0.
    Hard,
    /// Soft thresholding: shrinks coefficients toward zero by τ.
    Soft,
    /// Garrote (non-negative garrote): c → c - τ²/c  for |c| > τ.
    Garrote,
    /// Firm (semi-soft): linear transition between hard and soft.
    Firm { t2: f64 },
}

/// Apply a scalar threshold to a coefficient vector.
fn threshold_coeffs(coeffs: &[f64], tau: f64, method: ThresholdMethod) -> Vec<f64> {
    coeffs
        .iter()
        .map(|&c| apply_threshold(c, tau, method))
        .collect()
}

/// Apply threshold to a single coefficient.
fn apply_threshold(c: f64, tau: f64, method: ThresholdMethod) -> f64 {
    match method {
        ThresholdMethod::Hard => {
            if c.abs() >= tau {
                c
            } else {
                0.0
            }
        }
        ThresholdMethod::Soft => {
            if c > tau {
                c - tau
            } else if c < -tau {
                c + tau
            } else {
                0.0
            }
        }
        ThresholdMethod::Garrote => {
            if c.abs() <= tau {
                0.0
            } else {
                c - tau * tau / c
            }
        }
        ThresholdMethod::Firm { t2 } => {
            let t1 = tau;
            let abs_c = c.abs();
            if abs_c <= t1 {
                0.0
            } else if abs_c >= t2 {
                c
            } else {
                // Linear ramp
                c.signum() * t1 * (abs_c - t1) / (t2 - t1)
            }
        }
    }
}

/// Denoise a signal using the Wavelet Packet Transform.
///
/// The procedure is:
/// 1. Compute the full WPT up to `max_level`.
/// 2. Select the best basis using Shannon entropy.
/// 3. Threshold the coefficients in the best-basis nodes.
/// 4. Reconstruct the signal.
///
/// # Arguments
///
/// * `signal`    – Noisy input signal.
/// * `wavelet`   – Wavelet to use.
/// * `max_level` – Maximum decomposition depth.
/// * `threshold` – Threshold value τ.
/// * `method`    – Thresholding method.
///
/// # Errors
///
/// Propagates any error from `wpd`, `best_basis`, or `wp_reconstruct`.
///
/// # Example
///
/// ```
/// use scirs2_fft::wavelet_packets::{wp_denoising, ThresholdMethod, Wavelet};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
/// let denoised = wp_denoising(&signal, Wavelet::Db4, 3, 0.05, ThresholdMethod::Soft)
///     .expect("denoising failed");
/// assert_eq!(denoised.len(), signal.len());
/// ```
pub fn wp_denoising(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
    threshold: f64,
    method: ThresholdMethod,
) -> FFTResult<Vec<f64>> {
    // 1. Decompose
    let tree = wpd(signal, wavelet, max_level)?;

    // 2. Best basis
    let basis = best_basis(&tree, shannon_entropy)?;

    // 3. Threshold coefficients (do NOT threshold the root / approximation leaf)
    let thresholded: Vec<WaveletPacketNode> = basis
        .into_iter()
        .map(|mut node| {
            if node.level > 0 {
                node.coeffs = threshold_coeffs(&node.coeffs, threshold, method);
            }
            node
        })
        .collect();

    // 4. Reconstruct
    let mut recon = wp_reconstruct(&tree, &thresholded)?;

    // Trim or extend to original signal length
    recon.truncate(signal.len());
    while recon.len() < signal.len() {
        recon.push(0.0);
    }

    Ok(recon)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple test signal.
    fn test_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * std::f64::consts::PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 13.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_haar_decomp_shape() {
        let sig = test_signal(64);
        let tree = wpd(&sig, Wavelet::Haar, 3).expect("wpd failed");
        // All nodes at level 3 should exist
        for idx in 0..8 {
            assert!(tree.get(3, idx).is_some(), "missing node (3, {idx})");
        }
    }

    #[test]
    fn test_qmf_energy_preservation() {
        // lo and hi filters of db2 should each have unit energy
        let filters = WaveletFilters::for_wavelet(Wavelet::Db2);
        let e_lo: f64 = filters.lo_d.iter().map(|&c| c * c).sum();
        let e_hi: f64 = filters.hi_d.iter().map(|&c| c * c).sum();
        assert!((e_lo - 1.0).abs() < 1e-10, "lo energy {e_lo}");
        assert!((e_hi - 1.0).abs() < 1e-10, "hi energy {e_hi}");
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform signal (all equal nonzero): entropy should be positive
        let coeffs = vec![0.5_f64; 8];
        let e = shannon_entropy(&coeffs);
        assert!(e > 0.0, "expected positive entropy, got {e}");
    }

    #[test]
    fn test_shannon_entropy_sparse() {
        // A single non-zero coefficient → minimum entropy (sparse)
        let mut coeffs = vec![0.0_f64; 64];
        coeffs[0] = 1.0;
        let e = shannon_entropy(&coeffs);
        assert!((e - 0.0).abs() < 1e-12, "sparse signal entropy {e}");
    }

    #[test]
    fn test_best_basis_returns_valid_partition() {
        let sig = test_signal(64);
        let tree = wpd(&sig, Wavelet::Db2, 3).expect("wpd");
        let basis = best_basis(&tree, shannon_entropy).expect("best_basis");

        // Basis must be non-empty
        assert!(!basis.is_empty(), "basis is empty");

        // All nodes in basis must exist in the tree
        for node in &basis {
            assert!(
                tree.get(node.level, node.index).is_some(),
                "basis node ({}, {}) not in tree",
                node.level,
                node.index
            );
        }
    }

    #[test]
    fn test_haar_perfect_reconstruction() {
        let sig = test_signal(64);
        let tree = wpd(&sig, Wavelet::Haar, 2).expect("wpd");
        // Use all leaf nodes as basis (no simplification)
        let basis: Vec<WaveletPacketNode> = (0..4_usize)
            .filter_map(|idx| tree.get(2, idx).cloned())
            .collect();
        let recon = wp_reconstruct(&tree, &basis).expect("recon");
        for (i, (&s, &r)) in sig.iter().zip(recon.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-10,
                "mismatch at {i}: orig={s}, recon={r}"
            );
        }
    }

    #[test]
    fn test_denoising_length_preserved() {
        let sig = test_signal(64);
        let denoised =
            wp_denoising(&sig, Wavelet::Db4, 3, 0.1, ThresholdMethod::Soft).expect("denoise");
        assert_eq!(denoised.len(), sig.len());
    }

    #[test]
    fn test_threshold_hard() {
        let coeffs = vec![1.0, -0.5, 0.3, -0.1, 2.0];
        let out = threshold_coeffs(&coeffs, 0.4, ThresholdMethod::Hard);
        assert_eq!(out, vec![1.0, -0.5, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_threshold_soft() {
        let out = threshold_coeffs(&[1.0, -1.5, 0.2], 0.5, ThresholdMethod::Soft);
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!((out[1] - (-1.0)).abs() < 1e-12);
        assert!((out[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_wpd_error_on_empty() {
        let result = wpd(&[], Wavelet::Haar, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_wpd_error_on_zero_level() {
        let result = wpd(&[1.0, 2.0, 3.0], Wavelet::Haar, 0);
        assert!(result.is_err());
    }
}
