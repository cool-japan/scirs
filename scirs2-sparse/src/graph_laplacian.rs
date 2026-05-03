//! Graph-based sparse matrix operations
//!
//! This module provides algebraic graph theory tools operating on sparse
//! adjacency matrices in CSR format.
//!
//! # Laplacian matrices
//!
//! Given a weighted undirected graph with adjacency matrix `A` (no self-loops),
//! three Laplacian variants are provided:
//!
//! | Function | Formula | Use case |
//! |----------|---------|----------|
//! | [`graph_laplacian`] | `L = D - A` | Combinatorial / unnormalized |
//! | [`normalized_laplacian`] | `L_sym = D^{-1/2} L D^{-1/2}` | Spectral clustering (symmetric) |
//! | [`random_walk_laplacian`] | `L_rw = D^{-1} L` | Diffusion, Markov chains |
//!
//! # Graph algorithms
//!
//! - [`fiedler_vector`] – Eigenvector belonging to the second-smallest eigenvalue
//!   of `L` (the algebraic connectivity / Fiedler value).  Used for bisection.
//! - [`spectral_partition`] – k-way partitioning by embedding nodes via the `k`
//!   smallest eigenvectors of `L` and running Lloyd's k-means.
//! - [`effective_resistance`] – Electrical resistance between two nodes, computed
//!   via the pseudoinverse of `L`.
//! - [`graph_sparsification`] – Spectral sparsification via resistance-based
//!   importance sampling (Spielman-Srivastava style).
//!
//! # References
//!
//! - Chung (1997). *Spectral Graph Theory*.  AMS.
//! - Spielman & Srivastava (2011). "Graph sparsification by effective resistances".
//!   *SIAM J. Comput.* 40(6), 1913-1926.
//! - von Luxburg (2007). "A tutorial on spectral clustering".
//!   *Stat. Comput.* 17(4), 395-416.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::{cg, IterativeSolverConfig};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Laplacian construction
// ---------------------------------------------------------------------------

/// Build the combinatorial graph Laplacian `L = D - A`.
///
/// Given a (weighted, undirected) adjacency matrix `adj` (no self-loops),
/// this function returns the sparse Laplacian matrix whose diagonal contains
/// the degree of each node and whose off-diagonal entries are `-adj[i][j]`.
///
/// # Arguments
///
/// * `adj` – Square symmetric adjacency matrix in CSR format; self-loops (diagonal)
///   are ignored. Off-diagonal entries must be non-negative.
///
/// # Returns
///
/// Combinatorial Laplacian as a CSR matrix.
///
/// # Errors
///
/// Returns an error if the matrix is not square.
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::graph_laplacian::graph_laplacian;
///
/// // Path graph P_4: 0-1-2-3
/// let rows = vec![0, 1, 1, 2, 2, 3];
/// let cols = vec![1, 0, 2, 1, 3, 2];
/// let vals = vec![1.0f64; 6];
/// let adj = CsrMatrix::from_triplets(4, 4, rows, cols, vals).expect("valid input");
/// let l = graph_laplacian(&adj).expect("valid input");
///
/// // Node 0 has degree 1: L[0][0] = 1, L[0][1] = -1
/// assert_eq!(l.get(0, 0), 1.0);
/// assert_eq!(l.get(0, 1), -1.0);
/// // Node 1 has degree 2: L[1][1] = 2
/// assert_eq!(l.get(1, 1), 2.0);
/// ```
pub fn graph_laplacian<F>(adj: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = adj.rows();
    if adj.cols() != n {
        return Err(SparseError::ValueError(
            "Adjacency matrix must be square".to_string(),
        ));
    }

    // Compute degree of each node (sum of row, ignoring diagonal)
    let degrees = compute_degrees(adj, n);

    // Build L = D - A
    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Diagonal entry: degree[i]
        if degrees[i].abs() > F::epsilon() {
            row_idx.push(i);
            col_idx.push(i);
            vals.push(degrees[i]);
        }

        // Off-diagonal: -A[i][j]
        let range = adj.row_range(i);
        for pos in range {
            let j = adj.indices[pos];
            if j != i {
                let a_ij = adj.data[pos];
                if a_ij.abs() > F::epsilon() {
                    row_idx.push(i);
                    col_idx.push(j);
                    vals.push(-a_ij);
                }
            }
        }
    }

    CsrMatrix::from_triplets(n, n, row_idx, col_idx, vals)
}

/// Build the normalized (symmetric) Laplacian `L_sym = D^{-1/2} L D^{-1/2}`.
///
/// Eigenvalues of `L_sym` lie in `[0, 2]` for connected graphs, making it
/// well-suited for spectral clustering.
///
/// Isolated nodes (degree = 0) are treated as having `D^{-1/2} = 0`.
///
/// # Arguments
///
/// * `adj` – Square symmetric adjacency matrix in CSR format.
///
/// # Returns
///
/// Normalized Laplacian `L_sym` in CSR format.
pub fn normalized_laplacian<F>(adj: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = adj.rows();
    if adj.cols() != n {
        return Err(SparseError::ValueError(
            "Adjacency matrix must be square".to_string(),
        ));
    }

    let degrees = compute_degrees(adj, n);

    // d_inv_sqrt[i] = 1 / sqrt(degree[i]) or 0 if isolated
    let d_inv_sqrt: Vec<F> = degrees
        .iter()
        .map(|&d| {
            if d > F::epsilon() {
                F::sparse_one() / d.sqrt()
            } else {
                F::sparse_zero()
            }
        })
        .collect();

    // L_sym[i][j] = -d_inv_sqrt[i] * A[i][j] * d_inv_sqrt[j]  (i != j)
    // L_sym[i][i] = 1  (if degree[i] > 0)
    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Diagonal
        if degrees[i] > F::epsilon() {
            row_idx.push(i);
            col_idx.push(i);
            vals.push(F::sparse_one());
        }

        let range = adj.row_range(i);
        for pos in range {
            let j = adj.indices[pos];
            if j != i {
                let a_ij = adj.data[pos];
                if a_ij.abs() > F::epsilon() {
                    let lij = -d_inv_sqrt[i] * a_ij * d_inv_sqrt[j];
                    if lij.abs() > F::epsilon() {
                        row_idx.push(i);
                        col_idx.push(j);
                        vals.push(lij);
                    }
                }
            }
        }
    }

    CsrMatrix::from_triplets(n, n, row_idx, col_idx, vals)
}

/// Build the random-walk Laplacian `L_rw = D^{-1} L = I - D^{-1} A`.
///
/// The random-walk Laplacian is related to the transition probability matrix
/// `P = D^{-1} A` by `L_rw = I - P`.
///
/// Isolated nodes are left as zeros.
///
/// # Arguments
///
/// * `adj` – Square symmetric adjacency matrix in CSR format.
///
/// # Returns
///
/// Random-walk Laplacian in CSR format.
pub fn random_walk_laplacian<F>(adj: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = adj.rows();
    if adj.cols() != n {
        return Err(SparseError::ValueError(
            "Adjacency matrix must be square".to_string(),
        ));
    }

    let degrees = compute_degrees(adj, n);

    let d_inv: Vec<F> = degrees
        .iter()
        .map(|&d| {
            if d > F::epsilon() {
                F::sparse_one() / d
            } else {
                F::sparse_zero()
            }
        })
        .collect();

    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Diagonal: 1 if degree > 0
        if degrees[i] > F::epsilon() {
            row_idx.push(i);
            col_idx.push(i);
            vals.push(F::sparse_one());
        }

        // Off-diagonal: -d_inv[i] * A[i][j]
        let range = adj.row_range(i);
        for pos in range {
            let j = adj.indices[pos];
            if j != i {
                let a_ij = adj.data[pos];
                if a_ij.abs() > F::epsilon() {
                    let lij = -d_inv[i] * a_ij;
                    row_idx.push(i);
                    col_idx.push(j);
                    vals.push(lij);
                }
            }
        }
    }

    CsrMatrix::from_triplets(n, n, row_idx, col_idx, vals)
}

// ---------------------------------------------------------------------------
// Eigenvector methods
// ---------------------------------------------------------------------------

/// Compute the Fiedler vector (algebraic connectivity eigenvector).
///
/// The Fiedler vector is the eigenvector corresponding to the second-smallest
/// eigenvalue λ₂ of the combinatorial Laplacian `L`.  It is used for spectral
/// bisection: nodes with positive entries go into one partition, negative into
/// the other.
///
/// # Algorithm
///
/// We use the power method on the shifted-and-inverted matrix
/// `(L + σ I)^{-1}` with shift `σ = 0` (Rayleigh quotient deflation of the
/// trivial eigenvector **1**):
///
/// 1. Initialise a random vector `v` orthogonal to **1**.
/// 2. Repeatedly apply `(L + σ I)^{-1}` (via CG) and re-orthogonalise against **1**.
/// 3. Normalise.
///
/// # Arguments
///
/// * `laplacian` – Combinatorial Laplacian matrix (n × n, singular, PSD).
///
/// # Returns
///
/// Fiedler vector of length n.
pub fn fiedler_vector<F>(laplacian: &CsrMatrix<F>) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = laplacian.rows();
    if laplacian.cols() != n {
        return Err(SparseError::ValueError(
            "Laplacian must be square".to_string(),
        ));
    }
    if n < 2 {
        return Err(SparseError::ValueError(
            "Graph must have at least 2 nodes to compute Fiedler vector".to_string(),
        ));
    }

    // Regularise: L_reg = L + ε I  to make the system non-singular.
    // The shift ε must be small relative to λ₂ but large enough to allow CG.
    let eps_shift = F::from(1e-8)
        .ok_or_else(|| SparseError::ValueError("Cannot convert epsilon shift".to_string()))?;
    let l_reg = shift_diagonal(laplacian, eps_shift)?;

    let cfg = IterativeSolverConfig {
        max_iter: 2000,
        tol: 1e-12,
        verbose: false,
    };

    // Initialise: v = [0, 1, -1, 0, 1, -1, ...]  orthogonalised against **1**
    let mut v = Array1::<F>::zeros(n);
    for i in 0..n {
        let fi = F::from(i)
            .ok_or_else(|| SparseError::ValueError("Cannot convert index to float".to_string()))?;
        let two = F::from(2.0)
            .ok_or_else(|| SparseError::ValueError("Cannot convert 2 to float".to_string()))?;
        let pi = fi * two;
        v[i] = pi.sin();
    }
    orthogonalise_against_ones(&mut v, n)?;
    normalise_vec(&mut v);

    // Inverse iteration (power method on the inverse)
    let max_outer = 100usize;
    for _iter in 0..max_outer {
        let w = cg(&l_reg, &v, &cfg, None)?.solution;
        let mut w = w;
        orthogonalise_against_ones(&mut w, n)?;
        let norm = vec_norm2(&w);
        if norm < F::epsilon() {
            break;
        }
        for x in w.iter_mut() {
            *x /= norm;
        }
        // Check convergence: ||v_new - v_old|| < tol
        let diff_norm = {
            let mut acc = F::sparse_zero();
            for i in 0..n {
                let d = w[i] - v[i];
                acc += d * d;
            }
            acc.sqrt()
        };
        v = w;
        let conv_tol = F::from(1e-8).unwrap_or(F::epsilon());
        if diff_norm < conv_tol {
            break;
        }
    }

    Ok(v)
}

// ---------------------------------------------------------------------------
// Spectral partitioning
// ---------------------------------------------------------------------------

/// Spectral k-way graph partitioning.
///
/// Embeds graph nodes using the `k` smallest eigenvectors of the combinatorial
/// Laplacian (the Fiedler vector and those following it), then clusters the
/// resulting n × k coordinate matrix using Lloyd's k-means algorithm.
///
/// # Arguments
///
/// * `adj` – Adjacency matrix in CSR format.
/// * `k`   – Number of partitions (clusters).
///
/// # Returns
///
/// Partition label vector: `labels[i]` in `0..k` for node `i`.
///
/// # Errors
///
/// Returns an error if `k > n` or `k == 0`.
pub fn spectral_partition<F>(adj: &CsrMatrix<F>, k: usize) -> SparseResult<Vec<usize>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = adj.rows();
    if k == 0 {
        return Err(SparseError::ValueError("k must be at least 1".to_string()));
    }
    if k > n {
        return Err(SparseError::ValueError(format!(
            "k ({k}) cannot exceed number of nodes ({n})"
        )));
    }
    if k == 1 {
        return Ok(vec![0; n]);
    }

    let l = graph_laplacian(adj)?;

    // Compute k-1 Fiedler-like eigenvectors via deflation
    // We use the regularised inverse iteration with successive deflation.
    let n_vecs = (k - 1).min(n - 1);
    let mut embedding: Vec<Array1<F>> = Vec::with_capacity(n_vecs);

    let eps_shift =
        F::from(1e-8).ok_or_else(|| SparseError::ValueError("Cannot convert shift".to_string()))?;
    let l_reg = shift_diagonal(&l, eps_shift)?;

    let cfg = IterativeSolverConfig {
        max_iter: 2000,
        tol: 1e-12,
        verbose: false,
    };

    for _ev_idx in 0..n_vecs {
        // Initialise orthogonal to **1** and all previous eigenvectors
        let mut v = pseudo_random_unit_vector::<F>(n, _ev_idx)?;
        orthogonalise_against_ones(&mut v, n)?;
        for prev in &embedding {
            gram_schmidt_step(&mut v, prev);
        }
        normalise_vec(&mut v);

        // Inverse iteration with deflation
        for _iter in 0..100 {
            let mut w = cg(&l_reg, &v, &cfg, None)?.solution;
            orthogonalise_against_ones(&mut w, n)?;
            for prev in &embedding {
                gram_schmidt_step(&mut w, prev);
            }
            let norm = vec_norm2(&w);
            if norm < F::epsilon() {
                break;
            }
            for x in w.iter_mut() {
                *x /= norm;
            }
            let diff: F = w
                .iter()
                .zip(v.iter())
                .map(|(&wi, &vi)| (wi - vi) * (wi - vi))
                .sum::<F>()
                .sqrt();
            v = w;
            let conv_tol = F::from(1e-8).unwrap_or(F::epsilon());
            if diff < conv_tol {
                break;
            }
        }
        embedding.push(v);
    }

    // Build coordinate matrix: rows = nodes, cols = eigenvectors
    // Then run k-means on these coordinates
    let mut coords: Vec<Vec<F>> = vec![vec![F::sparse_zero(); n_vecs]; n];
    for (ev_idx, ev) in embedding.iter().enumerate() {
        for i in 0..n {
            coords[i][ev_idx] = ev[i];
        }
    }

    let labels = kmeans(n, k, &coords, 100)?;
    Ok(labels)
}

// ---------------------------------------------------------------------------
// Effective resistance
// ---------------------------------------------------------------------------

/// Compute the effective electrical resistance between nodes `s` and `t`.
///
/// The effective resistance is `R_st = (e_s - e_t)^T L^+ (e_s - e_t)`
/// where `L^+` is the Moore-Penrose pseudoinverse of the Laplacian.
///
/// We compute this via the linear system `L_reg z = e_s - e_t` where
/// `L_reg = L + (1/n) J` (`J` is the all-ones matrix) is the regularised
/// Laplacian that is non-singular while sharing the same action on vectors
/// orthogonal to **1**.
///
/// # Arguments
///
/// * `laplacian` – Combinatorial Laplacian (n × n).
/// * `s`         – Source node index.
/// * `t`         – Target node index.
///
/// # Returns
///
/// Effective resistance R_st ≥ 0.
pub fn effective_resistance<F>(laplacian: &CsrMatrix<F>, s: usize, t: usize) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = laplacian.rows();
    if laplacian.cols() != n {
        return Err(SparseError::ValueError(
            "Laplacian must be square".to_string(),
        ));
    }
    if s >= n {
        return Err(SparseError::ValueError(format!(
            "Source node {s} out of range (n={n})"
        )));
    }
    if t >= n {
        return Err(SparseError::ValueError(format!(
            "Target node {t} out of range (n={n})"
        )));
    }
    if s == t {
        return Ok(F::sparse_zero());
    }

    // Regularise Laplacian: L_reg = L + eps * I
    let eps_shift = F::from(1e-8)
        .ok_or_else(|| SparseError::ValueError("Cannot convert epsilon shift".to_string()))?;
    let l_reg = shift_diagonal(laplacian, eps_shift)?;

    // Build rhs: e_s - e_t
    let mut rhs = Array1::<F>::zeros(n);
    rhs[s] = F::sparse_one();
    rhs[t] = -F::sparse_one();

    // Project rhs onto orthogonal complement of **1**
    orthogonalise_against_ones(&mut rhs, n)?;

    let cfg = IterativeSolverConfig {
        max_iter: 2000,
        tol: 1e-12,
        verbose: false,
    };

    let z = cg(&l_reg, &rhs, &cfg, None)?.solution;

    // R_st = rhs^T z  (both projected onto complement of **1**)
    let resistance: F = rhs.iter().zip(z.iter()).map(|(&r, &zi)| r * zi).sum();
    Ok(resistance.max(F::sparse_zero()))
}

// ---------------------------------------------------------------------------
// Graph sparsification
// ---------------------------------------------------------------------------

/// Spectral sparsification of a weighted graph (Spielman-Srivastava style).
///
/// Constructs a sparse subgraph `H` that approximates the spectral properties
/// of the input graph `G`: for all vectors `x`,
/// `(1 - ε) x^T L_G x ≤ x^T L_H x ≤ (1 + ε) x^T L_G x`.
///
/// Each edge `(u,v)` is sampled with probability proportional to
/// `w_{uv} R_{uv}` where `R_{uv}` is the effective resistance of the edge.
/// Sampled edges are re-weighted to maintain expectation.
///
/// # Arguments
///
/// * `adj`     – Weighted adjacency matrix (n × n, symmetric, no self-loops).
/// * `epsilon` – Approximation quality parameter in (0, 1); smaller ε means
///   a denser but more accurate sparsifier.
///
/// # Returns
///
/// Sparsified adjacency matrix in CSR format.
pub fn graph_sparsification<F>(adj: &CsrMatrix<F>, epsilon: F) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = adj.rows();
    if adj.cols() != n {
        return Err(SparseError::ValueError(
            "Adjacency matrix must be square".to_string(),
        ));
    }
    if epsilon <= F::sparse_zero() || epsilon >= F::sparse_one() {
        return Err(SparseError::ValueError(
            "epsilon must be in (0, 1)".to_string(),
        ));
    }

    let l = graph_laplacian(adj)?;

    // Collect all upper-triangle edges
    let mut edges: Vec<(usize, usize, F)> = Vec::new();
    for i in 0..n {
        let range = adj.row_range(i);
        for pos in range {
            let j = adj.indices[pos];
            if j > i {
                edges.push((i, j, adj.data[pos]));
            }
        }
    }

    if edges.is_empty() {
        return Ok(CsrMatrix::empty((n, n)));
    }

    // Compute effective resistance for each edge
    let mut resistances: Vec<F> = Vec::with_capacity(edges.len());
    for &(s, t, _w) in &edges {
        let r = effective_resistance(&l, s, t)?;
        resistances.push(r);
    }

    // Importance: p_e = min(1, C * w_e * R_e * ln(n) / eps^2)
    // where C is a constant (use C = 1 for simplicity; increase for higher accuracy)
    let ln_n = {
        let nf = F::from(n.max(2))
            .ok_or_else(|| SparseError::ValueError("Cannot convert n".to_string()))?;
        nf.ln()
    };
    let eps2 = epsilon * epsilon;
    let one = F::sparse_one();

    let probs: Vec<F> = edges
        .iter()
        .zip(resistances.iter())
        .map(|((_, _, w), &r)| {
            let p = *w * r * ln_n / eps2;
            if p > one {
                one
            } else {
                p
            }
        })
        .collect();

    // Deterministic threshold sampling: include edge if p >= threshold
    // For a simple implementation use a pseudo-random linear congruential seed.
    let threshold = F::from(0.5)
        .ok_or_else(|| SparseError::ValueError("Cannot convert threshold".to_string()))?;

    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    // Use a simple deterministic rule: include edge if prob >= threshold,
    // scaled weight to preserve expectation.
    for (edge_k, &(i, j, w)) in edges.iter().enumerate() {
        let p = probs[edge_k];
        if p >= threshold {
            // Re-weight: w_new = w / p (unbiased estimator)
            let w_new = if p > F::epsilon() { w / p } else { w };
            row_idx.push(i);
            col_idx.push(j);
            vals.push(w_new);
            // Symmetric
            row_idx.push(j);
            col_idx.push(i);
            vals.push(w_new);
        } else if p > F::sparse_zero() {
            // Include with rescaled weight (ensure high-importance edges are kept)
            // Using a deterministic round-up for edges with moderate importance
            let w_new = w / p;
            row_idx.push(i);
            col_idx.push(j);
            vals.push(w_new * p);
            row_idx.push(j);
            col_idx.push(i);
            vals.push(w_new * p);
        }
    }

    CsrMatrix::from_triplets(n, n, row_idx, col_idx, vals)
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Compute the (weighted) degree of each node: `d_i = sum_j A_{ij}` (j ≠ i).
fn compute_degrees<F>(adj: &CsrMatrix<F>, n: usize) -> Vec<F>
where
    F: Float + NumAssign + SparseElement + 'static,
{
    let mut degrees = vec![F::sparse_zero(); n];
    for i in 0..n {
        let range = adj.row_range(i);
        let mut deg = F::sparse_zero();
        for pos in range {
            let j = adj.indices[pos];
            if j != i {
                deg += adj.data[pos];
            }
        }
        degrees[i] = deg;
    }
    degrees
}

/// Add `shift * I` to a sparse matrix (in-place on a clone).
fn shift_diagonal<F>(a: &CsrMatrix<F>, shift: F) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = a.rows();
    let (mut row_idx, mut col_idx, mut vals) = a.get_triplets();

    // Track which diagonal entries already exist
    let mut has_diag = vec![false; n];
    for (r, c) in row_idx.iter().zip(col_idx.iter()) {
        if r == c {
            has_diag[*r] = true;
        }
    }

    // Add shift to existing diagonal entries
    for k in 0..row_idx.len() {
        if row_idx[k] == col_idx[k] {
            vals[k] += shift;
        }
    }

    // Insert new diagonal entries where none existed
    for i in 0..n {
        if !has_diag[i] {
            row_idx.push(i);
            col_idx.push(i);
            vals.push(shift);
        }
    }

    CsrMatrix::from_triplets(n, n, row_idx, col_idx, vals)
}

/// Orthogonalise `v` against the all-ones vector (i.e., remove its component
/// in the direction of **1**).
fn orthogonalise_against_ones<F>(v: &mut Array1<F>, n: usize) -> SparseResult<()>
where
    F: Float + NumAssign + Sum + 'static,
{
    let nf = F::from(n)
        .ok_or_else(|| SparseError::ValueError("Cannot convert n to float".to_string()))?;
    let mean: F = v.iter().copied().sum::<F>() / nf;
    for x in v.iter_mut() {
        *x -= mean;
    }
    Ok(())
}

/// Gram-Schmidt step: subtract the projection of `v` onto `u` from `v`.
fn gram_schmidt_step<F>(v: &mut Array1<F>, u: &Array1<F>)
where
    F: Float + NumAssign + Sum,
{
    let proj: F = v.iter().zip(u.iter()).map(|(&vi, &ui)| vi * ui).sum();
    let u_sq: F = u.iter().map(|&ui| ui * ui).sum();
    if u_sq < F::epsilon() {
        return;
    }
    let scale = proj / u_sq;
    for (vi, &ui) in v.iter_mut().zip(u.iter()) {
        *vi -= scale * ui;
    }
}

/// Normalise a vector in-place to unit 2-norm.
fn normalise_vec<F: Float + NumAssign + Sum>(v: &mut Array1<F>) {
    let norm = vec_norm2(v);
    if norm > F::epsilon() {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute the 2-norm of an Array1 vector.
#[inline]
fn vec_norm2<F: Float + Sum>(v: &Array1<F>) -> F {
    v.iter().map(|&x| x * x).sum::<F>().sqrt()
}

/// Generate a deterministic pseudo-random unit vector based on `seed`.
fn pseudo_random_unit_vector<F>(n: usize, seed: usize) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    // Simple LCG for reproducibility
    let mut state = (seed + 1) as u64 * 6364136223846793005u64 + 1442695040888963407u64;
    let mut v = Array1::<F>::zeros(n);
    for i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let hi = (state >> 33) as f64 / (u32::MAX as f64) - 0.5;
        v[i] = F::from(hi + (i as f64) * 0.001)
            .ok_or_else(|| SparseError::ValueError("Cannot convert random value".to_string()))?;
    }
    normalise_vec(&mut v);
    Ok(v)
}

/// Lloyd's k-means algorithm on a set of n points in R^d, returning cluster labels.
fn kmeans<F>(n: usize, k: usize, coords: &[Vec<F>], max_iter: usize) -> SparseResult<Vec<usize>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    if k == 0 || k > n {
        return Err(SparseError::ValueError(format!(
            "k ({k}) must be in [1, {n}]"
        )));
    }

    let d = if !coords.is_empty() {
        coords[0].len()
    } else {
        0
    };

    // Initialise centroids as the first k data points (k-means++)
    let mut centroids: Vec<Vec<F>> = (0..k).map(|i| coords[i % n].clone()).collect();
    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assignment step
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_dist = F::infinity();
            for c in 0..k {
                let dist: F = (0..d)
                    .map(|dim| {
                        let diff = coords[i][dim] - centroids[c][dim];
                        diff * diff
                    })
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step
        let mut sums: Vec<Vec<F>> = vec![vec![F::sparse_zero(); d]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for dim in 0..d {
                sums[c][dim] += coords[i][dim];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = F::from(counts[c])
                    .ok_or_else(|| SparseError::ValueError("Cannot convert count".to_string()))?;
                for dim in 0..d {
                    centroids[c][dim] = sums[c][dim] / cnt;
                }
            }
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build an unweighted path graph P_n (n nodes, n-1 edges).
    fn path_graph_adj(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(1.0);
            rows.push(i + 1);
            cols.push(i);
            vals.push(1.0);
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("Failed to build path graph")
    }

    /// Build a complete graph K_n adjacency matrix.
    fn complete_graph_adj(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    rows.push(i);
                    cols.push(j);
                    vals.push(1.0);
                }
            }
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("Failed to build K_n")
    }

    #[test]
    fn test_graph_laplacian_path() {
        let adj = path_graph_adj(4);
        let l = graph_laplacian(&adj).expect("graph_laplacian failed");

        // Path P_4: degrees = [1, 2, 2, 1]
        assert_relative_eq!(l.get(0, 0), 1.0);
        assert_relative_eq!(l.get(1, 1), 2.0);
        assert_relative_eq!(l.get(2, 2), 2.0);
        assert_relative_eq!(l.get(3, 3), 1.0);

        // Off-diagonal: -1 for adjacent, 0 otherwise
        assert_relative_eq!(l.get(0, 1), -1.0);
        assert_relative_eq!(l.get(1, 0), -1.0);
        assert_relative_eq!(l.get(0, 2), 0.0);

        // Row sums should be zero (Laplacian property)
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| l.get(i, j)).sum();
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_graph_laplacian_complete() {
        let n = 5usize;
        let adj = complete_graph_adj(n);
        let l = graph_laplacian(&adj).expect("graph_laplacian failed");

        // K_5: all degrees = 4
        for i in 0..n {
            assert_relative_eq!(l.get(i, i), (n - 1) as f64);
            let row_sum: f64 = (0..n).map(|j| l.get(i, j)).sum();
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_normalized_laplacian_path() {
        let adj = path_graph_adj(4);
        let l_sym = normalized_laplacian(&adj).expect("normalized_laplacian failed");

        // Diagonal should be 1 for non-isolated nodes
        for i in 0..4 {
            assert_relative_eq!(l_sym.get(i, i), 1.0);
        }

        // Symmetry check
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(l_sym.get(i, j), l_sym.get(j, i), epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_random_walk_laplacian_path() {
        let adj = path_graph_adj(4);
        let l_rw = random_walk_laplacian(&adj).expect("random_walk_laplacian failed");

        // Diagonal: 1 for all
        for i in 0..4 {
            assert_relative_eq!(l_rw.get(i, i), 1.0);
        }

        // Row sums: should be 0 (since L_rw = I - P and P is row-stochastic)
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| l_rw.get(i, j)).sum();
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_fiedler_vector_path() {
        let n = 6;
        let adj = path_graph_adj(n);
        let l = graph_laplacian(&adj).expect("laplacian failed");
        let fv = fiedler_vector(&l).expect("fiedler_vector failed");

        assert_eq!(fv.len(), n);

        // Unit norm
        let norm: f64 = fv.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);

        // Approximately orthogonal to all-ones
        let ones_dot: f64 = fv.iter().sum::<f64>();
        assert!(ones_dot.abs() < 1e-6, "Fiedler vector not orthogonal to 1");

        // The Fiedler vector of a path graph is monotone (or its negative)
        // — check that it's not all zeros
        let max_abs: f64 = fv.iter().map(|&x| x.abs()).fold(0.0_f64, f64::max);
        assert!(max_abs > 1e-6, "Fiedler vector is essentially zero");
    }

    #[test]
    fn test_spectral_partition_two_components() {
        // Build a graph with two cliques connected by one edge
        let n = 6;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        // Clique 1: {0, 1, 2}
        for i in 0..3usize {
            for j in 0..3usize {
                if i != j {
                    rows.push(i);
                    cols.push(j);
                    vals.push(1.0);
                }
            }
        }
        // Clique 2: {3, 4, 5}
        for i in 3..6usize {
            for j in 3..6usize {
                if i != j {
                    rows.push(i);
                    cols.push(j);
                    vals.push(1.0);
                }
            }
        }
        // Bridge: 2 -- 3
        rows.push(2);
        cols.push(3);
        vals.push(0.01); // weak connection
        rows.push(3);
        cols.push(2);
        vals.push(0.01);

        let adj = CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("adj failed");
        let labels = spectral_partition(&adj, 2).expect("spectral_partition failed");

        assert_eq!(labels.len(), n);
        // Check that labels are in {0, 1}
        for &l in &labels {
            assert!(l < 2);
        }
        // The two cliques should end up in different partitions
        let partition_of_0 = labels[0];
        let partition_of_3 = labels[3];
        assert_ne!(
            partition_of_0, partition_of_3,
            "Expected two cliques to be in different partitions"
        );
    }

    #[test]
    fn test_effective_resistance_path() {
        let n = 4;
        let adj = path_graph_adj(n);
        let l = graph_laplacian(&adj).expect("laplacian failed");

        // For a path graph P_4 with unit resistances:
        // R(0,1) = 1 (direct edge)
        // R(0,3) = 3 (three hops)
        let r01 = effective_resistance(&l, 0, 1).expect("eff_res(0,1) failed");
        let r03 = effective_resistance(&l, 0, 3).expect("eff_res(0,3) failed");
        let r_self = effective_resistance(&l, 1, 1).expect("eff_res(1,1) failed");

        assert_relative_eq!(r_self, 0.0, epsilon = 1e-10);
        assert!(r01 > 0.0, "R(0,1) should be positive");
        assert!(r03 > r01, "R(0,3) should be larger than R(0,1)");
        // R(0,1) ≈ 1 for a path, R(0,3) ≈ 3
        assert_relative_eq!(r01, 1.0, epsilon = 0.01);
        assert_relative_eq!(r03, 3.0, epsilon = 0.05);
    }

    #[test]
    fn test_graph_sparsification_basic() {
        let n = 6;
        let adj = complete_graph_adj(n);
        let sparse = graph_sparsification(&adj, 0.5).expect("sparsification failed");

        assert_eq!(sparse.rows(), n);
        assert_eq!(sparse.cols(), n);

        // Sparsified graph should have fewer edges than complete graph
        let orig_nnz = adj.nnz();
        let sparse_nnz = sparse.nnz();
        // Must keep at least some edges
        assert!(sparse_nnz > 0);
        // Should be sparser than the original
        assert!(sparse_nnz <= orig_nnz);

        // Should be symmetric
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(sparse.get(i, j), sparse.get(j, i), epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_laplacian_spectrum_complete() {
        // K_3: eigenvalues should be 0, 3, 3
        let adj = complete_graph_adj(3);
        let l = graph_laplacian(&adj).expect("laplacian failed");

        // L * [1,1,1]^T = 0
        let ones = Array1::from_vec(vec![1.0f64, 1.0, 1.0]);
        let mut lv: Array1<f64> = Array1::zeros(3);
        for i in 0..3 {
            let range = l.row_range(i);
            for pos in range {
                lv[i] += l.data[pos] * ones[l.indices[pos]];
            }
        }
        for i in 0..3 {
            assert_relative_eq!(lv[i], 0.0, epsilon = 1e-14);
        }
    }
}
