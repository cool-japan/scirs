//! Graph signal sampling: optimal set selection, bandlimited reconstruction,
//! bandwidth estimation (Gershgorin), and the graph uncertainty principle.
//!
//! ## Overview
//!
//! Classical Nyquist–Shannon sampling requires that a signal's bandwidth be at
//! most half the sampling rate.  On graphs, an analogous theory exists where
//! "frequency" is defined through the Laplacian spectrum:
//!
//! - A signal is **k-bandlimited** if its GFT coefficients beyond index `k` are zero.
//! - A **sampling set** `S ⊆ V` is *k-valid* if the restriction `U_k|_S` (top-k
//!   eigenvectors restricted to rows in `S`) has full column rank.
//! - **Gershgorin circles** provide fast spectral radius bounds without computing
//!   the full eigendecomposition.
//! - The **graph uncertainty principle** quantifies the trade-off between spatial
//!   spread and spectral spread of a signal.
//!
//! ## References
//! - Pesenson (2008). "Sampling in Paley-Wiener spaces on combinatorial graphs."
//! - Shomorony & Avestimehr (2014). "Sampling large graphs."
//! - Agaskar & Lu (2013). "A spectral graph uncertainty principle."
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::signal_processing::sampling::{GraphSampling, BandlimitedReconstruction};
//! use scirs2_graph::signal_processing::gsp::GraphFourierTransform;
//!
//! let mut adj = Array2::<f64>::zeros((6, 6));
//! for i in 0..5_usize { adj[[i, i+1]] = 1.0; adj[[i+1, i]] = 1.0; }
//! let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
//!
//! // Find a 2-valid sampling set
//! let sampler = GraphSampling::new(2);
//! let set = sampler.greedy_sampling_set(&gft).unwrap();
//! println!("Sampling set: {set:?}");
//! ```

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{GraphError, Result};
use crate::signal_processing::gsp::GraphFourierTransform;

// ─────────────────────────────────────────────────────────────────────────────
// GraphSampling — greedy optimal sampling set
// ─────────────────────────────────────────────────────────────────────────────

/// Optimal graph signal sampling set selection.
///
/// Selects a set of `k` nodes from which a `k`-bandlimited graph signal can
/// be uniquely reconstructed.  The greedy algorithm maximises the cut-off
/// frequency achievable from the selected nodes by greedily picking nodes that
/// maximise the determinant (volume) of the submatrix formed by the `k`
/// smoothest eigenvectors restricted to the sampling set.
///
/// The greedy maximisation of the log-determinant is equivalent to maximising
/// the minimum singular value of the restricted eigenvector matrix, which
/// directly determines the stability of the reconstruction.
#[derive(Debug, Clone)]
pub struct GraphSampling {
    /// Number of frequency components (bandwidth) of the signal.
    pub bandwidth: usize,
}

impl GraphSampling {
    /// Create a `GraphSampling` instance for signals with `bandwidth` components.
    pub fn new(bandwidth: usize) -> Self {
        Self { bandwidth }
    }

    /// Greedy sampling set selection.
    ///
    /// Returns a list of `bandwidth` node indices that form a valid sampling set
    /// (i.e. the restriction of the first `bandwidth` eigenvectors to these rows
    /// has full column rank).
    ///
    /// The algorithm iteratively adds the node that maximises the volume
    /// (log-det of `Aᵀ A` where `A` is the current restricted eigenvector
    /// submatrix).  This is a classic D-optimal experimental design step.
    pub fn greedy_sampling_set(&self, gft: &GraphFourierTransform) -> Result<Vec<usize>> {
        let n = gft.num_nodes();
        let k = self.bandwidth;
        if k == 0 {
            return Ok(Vec::new());
        }
        if k > n {
            return Err(GraphError::InvalidParameter {
                param: "bandwidth".into(),
                value: k.to_string(),
                expected: format!("<= n ({})", n),
                context: "GraphSampling".into(),
            });
        }

        // The first k columns of the eigenvector matrix U (shape n×k)
        let u_k = gft
            .eigenvectors
            .slice(scirs2_core::ndarray::s![.., ..k])
            .to_owned();

        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut remaining: Vec<usize> = (0..n).collect();

        for _ in 0..k {
            // Find the node that maximises the volume increment.
            // Volume criterion: choose node `r` maximising the squared norm of
            // the projection of u_k[r, :] onto the complement of the span of
            // already-selected rows.
            let best = Self::pick_best_node(&u_k, &selected, &remaining, n, k)?;
            selected.push(best);
            remaining.retain(|&x| x != best);
        }

        Ok(selected)
    }

    /// Pick the node index from `remaining` that maximises the leverage score
    /// (projection onto the complement of the span of currently selected rows).
    fn pick_best_node(
        u_k: &Array2<f64>,
        selected: &[usize],
        remaining: &[usize],
        _n: usize,
        k: usize,
    ) -> Result<usize> {
        // Build the current selection matrix (rows of u_k for selected nodes)
        // Compute the orthogonal projection onto the span of selected rows.
        // Leverage score of row r = ‖(I − Π) u_k[r,:]‖²

        let mut best_score = -1.0_f64;
        let mut best_node = remaining[0];

        for &r in remaining {
            let row = u_k.row(r);
            let score = if selected.is_empty() {
                // No projection yet; score = ‖row‖²
                row.iter().map(|&x| x * x).sum::<f64>()
            } else {
                // Project row onto the complement of the span of selected rows.
                let s = selected.len();
                // Assemble sub-matrix S (s × k)
                let mut sub = Array2::<f64>::zeros((s, k));
                for (new_r, &old_r) in selected.iter().enumerate() {
                    for c in 0..k {
                        sub[[new_r, c]] = u_k[[old_r, c]];
                    }
                }
                // QR-free: use Gram-Schmidt residual
                let row_vec: Vec<f64> = row.to_vec();
                let residual = gram_schmidt_residual(&row_vec, &sub);
                residual.iter().map(|&x| x * x).sum::<f64>()
            };
            if score > best_score {
                best_score = score;
                best_node = r;
            }
        }

        Ok(best_node)
    }

    /// Check whether a given node set is a valid `bandwidth`-sampling set,
    /// i.e. the restricted eigenvector matrix has full column rank.
    ///
    /// Validity is checked by verifying that the minimum singular value of
    /// the restricted matrix exceeds `tol`.
    pub fn is_valid_sampling_set(
        &self,
        gft: &GraphFourierTransform,
        set: &[usize],
        tol: f64,
    ) -> Result<bool> {
        let k = self.bandwidth;
        let n = gft.num_nodes();
        if set.len() < k {
            return Ok(false);
        }
        for &s in set {
            if s >= n {
                return Err(GraphError::InvalidParameter {
                    param: "set node index".into(),
                    value: s.to_string(),
                    expected: format!("< {n}"),
                    context: "is_valid_sampling_set".into(),
                });
            }
        }
        // Build restricted matrix (|set| × k)
        let m = set.len();
        let mut r_mat = Array2::<f64>::zeros((m, k));
        for (new_r, &old_r) in set.iter().enumerate() {
            for c in 0..k {
                r_mat[[new_r, c]] = gft.eigenvectors[[old_r, c]];
            }
        }
        // Minimum singular value via power iteration on (A^T A)
        let min_sv = min_singular_value(&r_mat);
        Ok(min_sv > tol)
    }
}

/// Gram-Schmidt residual of `v` w.r.t. rows of `basis` (each row is a basis vector).
fn gram_schmidt_residual(v: &[f64], basis: &Array2<f64>) -> Vec<f64> {
    let k = v.len();
    let m = basis.nrows();
    let mut res = v.to_vec();
    for i in 0..m {
        let row = basis.row(i);
        let dot: f64 = res.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
        let norm_sq: f64 = row.iter().map(|&x| x * x).sum();
        if norm_sq > 1e-14 {
            for (r, &b) in res.iter_mut().zip(row.iter()) {
                *r -= (dot / norm_sq) * b;
            }
        }
    }
    let _ = k; // used implicitly
    res
}

/// Approximate minimum singular value of matrix `a` via inverse power iteration
/// applied to `aᵀ a`.  Returns a lower bound.
fn min_singular_value(a: &Array2<f64>) -> f64 {
    let m = a.nrows();
    let k = a.ncols();
    if m < k {
        return 0.0;
    }
    // Compute ata = a^T a  (k × k)
    let mut ata = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0_f64;
            for r in 0..m {
                acc += a[[r, i]] * a[[r, j]];
            }
            ata[[i, j]] = acc;
        }
    }

    // Power iteration on ata to find max eigenvalue, then invert
    // (approximation: we use the ratio of norms as a cheap estimate)
    // A more robust approach: Frobenius norm as upper bound, then
    // check conditioning.
    let frob: f64 = ata.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if frob < 1e-14 {
        return 0.0;
    }

    // One step of inverse power iteration with random start
    let mut v: Vec<f64> = (0..k).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    for _ in 0..30 {
        // Solve ata * w = v  (via Gaussian elimination on small k×k system)
        let w = solve_linear(&ata, &v);
        let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            return 0.0;
        }
        v = w.iter().map(|&x| x / norm).collect();
    }

    // Rayleigh quotient: v^T ata v / v^T v
    let ata_v = matvec(&ata, &v);
    let num: f64 = v.iter().zip(ata_v.iter()).map(|(&a, &b)| a * b).sum();
    let den: f64 = v.iter().map(|&x| x * x).sum();
    if den < 1e-14 {
        return 0.0;
    }
    // This Rayleigh quotient converges to the MIN eigenvalue of ata,
    // hence the min singular value of a is sqrt(min_eig(ata)).
    (num / den).max(0.0).sqrt()
}

fn matvec(a: &Array2<f64>, v: &[f64]) -> Vec<f64> {
    let n = a.nrows();
    let k = a.ncols();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..k {
            out[i] += a[[i, j]] * v[j];
        }
    }
    out
}

/// Solve `a x = b` via Gaussian elimination with partial pivoting (small k×k).
fn solve_linear(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let n = a.nrows();
    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            // Singular; return zeros
            return vec![0.0; n];
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                let val = aug[col][k];
                aug[row][k] -= factor * val;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// BandlimitedReconstruction — reconstruct from samples
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct a `k`-bandlimited graph signal from its samples on a node set.
///
/// Given samples `y = x[S]` at nodes `S`, and assuming the signal `x` is
/// `k`-bandlimited (only first `k` GFT components are non-zero), recover `x`
/// by solving the least-squares system:
///
///   U_k[S, :] α = y   =>   α = (U_k[S,:]ᵀ U_k[S,:])⁻¹ U_k[S,:]ᵀ y
///   x = U_k α
///
/// where `U_k` is the matrix of the `k` smoothest eigenvectors.
#[derive(Debug, Clone)]
pub struct BandlimitedReconstruction {
    /// Number of frequency components.
    pub bandwidth: usize,
}

impl BandlimitedReconstruction {
    /// Create a reconstruction instance for `k`-bandlimited signals.
    pub fn new(bandwidth: usize) -> Self {
        Self { bandwidth }
    }

    /// Reconstruct the full graph signal from samples.
    ///
    /// # Arguments
    /// * `gft` — precomputed GFT for the graph.
    /// * `sampling_set` — indices of sampled nodes.
    /// * `samples` — signal values at the sampled nodes (length = `|sampling_set|`).
    ///
    /// # Returns
    /// Reconstructed signal of length `n` (all nodes).
    pub fn reconstruct(
        &self,
        gft: &GraphFourierTransform,
        sampling_set: &[usize],
        samples: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = gft.num_nodes();
        let k = self.bandwidth;
        let s = sampling_set.len();

        if samples.len() != s {
            return Err(GraphError::InvalidParameter {
                param: "samples.len()".into(),
                value: samples.len().to_string(),
                expected: format!("{s} (= |sampling_set|)"),
                context: "BandlimitedReconstruction".into(),
            });
        }
        if s < k {
            return Err(GraphError::InvalidParameter {
                param: "sampling_set.len()".into(),
                value: s.to_string(),
                expected: format!(">= bandwidth ({})", k),
                context: "BandlimitedReconstruction".into(),
            });
        }

        // Build U_k[S, :] — shape (s, k)
        let mut u_s = Array2::<f64>::zeros((s, k));
        for (new_r, &old_r) in sampling_set.iter().enumerate() {
            if old_r >= n {
                return Err(GraphError::InvalidParameter {
                    param: "sampling_set node".into(),
                    value: old_r.to_string(),
                    expected: format!("< {n}"),
                    context: "BandlimitedReconstruction".into(),
                });
            }
            for c in 0..k {
                u_s[[new_r, c]] = gft.eigenvectors[[old_r, c]];
            }
        }

        // Normal equations: (U_s^T U_s) α = U_s^T y
        // Build Gram matrix G = U_s^T U_s  (k × k)
        let mut gram = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0_f64;
                for r in 0..s {
                    acc += u_s[[r, i]] * u_s[[r, j]];
                }
                gram[[i, j]] = acc;
            }
        }

        // Build right-hand side: rhs = U_s^T y  (length k)
        let rhs: Vec<f64> = (0..k)
            .map(|c| (0..s).map(|r| u_s[[r, c]] * samples[r]).sum::<f64>())
            .collect();

        // Solve Gram α = rhs
        let alpha = solve_linear(&gram, &rhs);

        // Reconstruct: x = U_k α  (length n)
        let mut x = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0_f64;
            for c in 0..k {
                acc += gft.eigenvectors[[i, c]] * alpha[c];
            }
            x[i] = acc;
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GershgorinBound — graph signal bandwidth estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Gershgorin-circle-based bounds on the graph Laplacian spectrum.
///
/// The Gershgorin circle theorem states that every eigenvalue `λ` of a matrix
/// `M` lies in at least one Gershgorin disc centred at `M[i,i]` with radius
/// `R_i = Σ_{j≠i} |M[i,j]|`.
///
/// For the graph Laplacian `L = D − A`:
///   - `L[i,i] = degree(i)`
///   - `R_i = degree(i)` (sum of off-diagonal magnitudes)
///
/// Therefore all eigenvalues lie in `[0, 2 * max_degree]` and we get
/// a fast upper bound on the graph bandwidth without computing eigenvalues.
#[derive(Debug, Clone)]
pub struct GershgorinBound {
    /// Upper bound on the spectral radius (max Laplacian eigenvalue).
    pub lambda_max_upper: f64,
    /// Lower bound: 0 (Laplacian is PSD).
    pub lambda_min_lower: f64,
    /// Per-node Gershgorin radii.
    pub radii: Vec<f64>,
    /// Per-node centres (= degree of node i).
    pub centres: Vec<f64>,
}

impl GershgorinBound {
    /// Compute Gershgorin bounds from a weighted adjacency matrix.
    pub fn from_adjacency(adj: &Array2<f64>) -> Result<Self> {
        let n = adj.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty adjacency".into()));
        }
        let mut centres = vec![0.0_f64; n];
        let mut radii = vec![0.0_f64; n];
        for i in 0..n {
            let deg = adj.row(i).iter().map(|&x| x.abs()).sum::<f64>();
            centres[i] = deg; // L[i,i] = degree
            radii[i] = deg; // R_i = sum of |L[i,j]| for j != i = same deg for L
        }
        let lambda_max_upper = centres
            .iter()
            .zip(radii.iter())
            .map(|(&c, &r)| c + r)
            .fold(0.0_f64, f64::max);
        Ok(Self {
            lambda_max_upper,
            lambda_min_lower: 0.0,
            radii,
            centres,
        })
    }

    /// Estimate the effective bandwidth of a graph signal based on the
    /// fraction of spectral energy concentrated in low frequencies.
    ///
    /// Computes the GFT and returns the smallest `k` such that the
    /// cumulative spectral energy in the first `k` components exceeds `threshold`.
    pub fn signal_bandwidth(
        gft: &GraphFourierTransform,
        signal: &Array1<f64>,
        threshold: f64,
    ) -> Result<usize> {
        let x_hat = gft.transform(signal)?;
        let total_energy: f64 = x_hat.iter().map(|&c| c * c).sum();
        if total_energy < 1e-14 {
            return Ok(0);
        }
        let mut cumulative = 0.0_f64;
        for (k, &c) in x_hat.iter().enumerate() {
            cumulative += c * c;
            if cumulative / total_energy >= threshold {
                return Ok(k + 1);
            }
        }
        Ok(x_hat.len())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphUncertaintyPrinciple
// ─────────────────────────────────────────────────────────────────────────────

/// Graph uncertainty principle: spatial spread vs. spectral spread tradeoff.
///
/// Analogous to Heisenberg's uncertainty principle, signals on graphs cannot
/// be simultaneously concentrated in both the vertex domain and the spectral
/// domain.
///
/// **Spatial spread** of signal `x` about reference node `v₀`:
///   Δ_V²(x) = Σ_i d(i, v₀)² x_i² / ‖x‖²
///
/// **Spectral spread** about reference frequency `λ₀`:
///   Δ_S²(x) = Σ_k (λ_k − λ₀)² x̂_k² / ‖x‖²
///
/// The product `Δ_V Δ_S ≥ C` for some graph-dependent constant.
#[derive(Debug, Clone)]
pub struct GraphUncertaintyPrinciple {
    /// Squared pairwise distances from each node to node `v₀`.
    pub spatial_distances_sq: Array1<f64>,
    /// Reference frequency `λ₀` (usually the DC frequency = 0).
    pub ref_frequency: f64,
}

impl GraphUncertaintyPrinciple {
    /// Build from BFS shortest-path distances from reference node `v0`.
    ///
    /// # Arguments
    /// * `adj` — weighted adjacency matrix.
    /// * `v0` — reference node index (typically the "centre" of the signal).
    /// * `ref_frequency` — spectral reference point (default: 0 for DC).
    pub fn new(adj: &Array2<f64>, v0: usize, ref_frequency: f64) -> Result<Self> {
        let n = adj.nrows();
        if v0 >= n {
            return Err(GraphError::InvalidParameter {
                param: "v0".into(),
                value: v0.to_string(),
                expected: format!("< {n}"),
                context: "GraphUncertaintyPrinciple".into(),
            });
        }
        // Compute shortest-path distances from v0 via Dijkstra (unweighted BFS)
        let mut dist = vec![f64::INFINITY; n];
        dist[v0] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(v0);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[[u, v]] != 0.0 && dist[v].is_infinite() {
                    dist[v] = dist[u] + 1.0;
                    queue.push_back(v);
                }
            }
        }
        let spatial_distances_sq = Array1::from_iter(dist.iter().map(|&d| d * d));
        Ok(Self {
            spatial_distances_sq,
            ref_frequency,
        })
    }

    /// Compute the spatial spread Δ_V(x) for signal `x`.
    pub fn spatial_spread(&self, signal: &Array1<f64>) -> Result<f64> {
        let n = signal.len();
        if n != self.spatial_distances_sq.len() {
            return Err(GraphError::InvalidParameter {
                param: "signal.len()".into(),
                value: n.to_string(),
                expected: self.spatial_distances_sq.len().to_string(),
                context: "spatial_spread".into(),
            });
        }
        let norm_sq: f64 = signal.iter().map(|&x| x * x).sum();
        if norm_sq < 1e-14 {
            return Ok(0.0);
        }
        let spread_sq: f64 = signal
            .iter()
            .zip(self.spatial_distances_sq.iter())
            .map(|(&x, &d2)| d2 * x * x)
            .sum::<f64>()
            / norm_sq;
        Ok(spread_sq.sqrt())
    }

    /// Compute the spectral spread Δ_S(x) for signal `x` using the GFT.
    pub fn spectral_spread(
        &self,
        gft: &GraphFourierTransform,
        signal: &Array1<f64>,
    ) -> Result<f64> {
        let x_hat = gft.transform(signal)?;
        let norm_sq: f64 = x_hat.iter().map(|&c| c * c).sum();
        if norm_sq < 1e-14 {
            return Ok(0.0);
        }
        let spread_sq: f64 = x_hat
            .iter()
            .zip(gft.eigenvalues.iter())
            .map(|(&c, &lam)| {
                let d = lam - self.ref_frequency;
                d * d * c * c
            })
            .sum::<f64>()
            / norm_sq;
        Ok(spread_sq.sqrt())
    }

    /// Compute both spatial and spectral spreads and their product.
    ///
    /// Returns `(spatial_spread, spectral_spread, product)`.
    pub fn uncertainty(
        &self,
        gft: &GraphFourierTransform,
        signal: &Array1<f64>,
    ) -> Result<(f64, f64, f64)> {
        let dv = self.spatial_spread(signal)?;
        let ds = self.spectral_spread(gft, signal)?;
        Ok((dv, ds, dv * ds))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn path_adj(n: usize) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    #[test]
    fn test_greedy_sampling_set_size() {
        let adj = path_adj(8);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let sampler = GraphSampling::new(3);
        let set = sampler.greedy_sampling_set(&gft).unwrap();
        assert_eq!(set.len(), 3);
        // All indices should be valid
        for &s in &set {
            assert!(s < 8);
        }
        // All indices should be unique
        let mut uniq = set.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), set.len());
    }

    #[test]
    fn test_bandlimited_reconstruction() {
        let n = 6;
        let adj = path_adj(n);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let k = 2;

        // Build a k-bandlimited signal (only first k GFT components)
        let mut x_hat = Array1::<f64>::zeros(n);
        x_hat[0] = 2.0;
        x_hat[1] = 1.0;
        let original = gft.inverse(&x_hat).unwrap();

        // Sample all nodes (trivial reconstruction)
        let set: Vec<usize> = (0..n).collect();
        let samples = Array1::from_iter(set.iter().map(|&i| original[i]));
        let rec = BandlimitedReconstruction::new(k)
            .reconstruct(&gft, &set, &samples)
            .unwrap();

        for (a, b) in original.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-8, "Reconstruction mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_gershgorin_bounds() {
        let adj = path_adj(5);
        let bounds = GershgorinBound::from_adjacency(&adj).unwrap();
        assert!(bounds.lambda_min_lower == 0.0);
        assert!(bounds.lambda_max_upper > 0.0);
        // For a path graph, all eigenvalues are in [0, 4], so upper bound <= 4
        assert!(bounds.lambda_max_upper <= 4.0 + 1e-9);
    }

    #[test]
    fn test_signal_bandwidth() {
        let adj = path_adj(8);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        // DC signal: all energy in component 0
        let dc = Array1::from_vec(vec![1.0; 8]);
        let bw = GershgorinBound::signal_bandwidth(&gft, &dc, 0.99).unwrap();
        assert!(bw <= 2, "DC signal should have bandwidth 1 (got {bw})");
    }

    #[test]
    fn test_uncertainty_principle() {
        let adj = path_adj(7);
        let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
        let up = GraphUncertaintyPrinciple::new(&adj, 3, 0.0).unwrap();
        // Localised signal at node 3
        let mut local = Array1::<f64>::zeros(7);
        local[3] = 1.0;
        let (dv, ds, prod) = up.uncertainty(&gft, &local).unwrap();
        assert!(dv >= 0.0);
        assert!(ds >= 0.0);
        assert!(prod >= 0.0);
        // A localised vertex signal should have small spatial spread
        assert!(
            dv < 1.0,
            "Vertex-localised signal should have small dv: {dv}"
        );
    }
}
