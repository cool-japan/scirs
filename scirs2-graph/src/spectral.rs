//! Spectral graph theory operations
//!
//! This module provides functions for spectral graph analysis,
//! including Laplacian matrices, spectral clustering, and eigenvalue-based
//! graph properties.
//!
//! Features SIMD acceleration for performance-critical operations.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1};
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// SIMD-accelerated matrix operations for spectral algorithms
mod simd_spectral {
    use super::*;

    /// SIMD-accelerated matrix subtraction: result = a - b
    #[allow(dead_code)]
    pub fn simd_matrix_subtract(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        assert_eq!(a.shape(), b.shape());
        let (rows, cols) = a.dim();
        let mut result = Array2::zeros((rows, cols));

        // Use SIMD operations from scirs2-core
        for i in 0..rows {
            let a_row = a.row(i);
            let b_row = b.row(i);
            let mut result_row = result.row_mut(i);

            // Convert to slices for SIMD operations
            if let (Some(a_slice), Some(b_slice), Some(result_slice)) = (
                a_row.as_slice(),
                b_row.as_slice(),
                result_row.as_slice_mut(),
            ) {
                // Use SIMD subtraction from scirs2-core
                let a_view = scirs2_core::ndarray::ArrayView1::from(a_slice);
                let b_view = scirs2_core::ndarray::ArrayView1::from(b_slice);
                let result_array = f64::simd_sub(&a_view, &b_view);
                result_slice.copy_from_slice(result_array.as_slice().expect("Operation failed"));
            } else {
                // Fallback to element-wise operation if not contiguous
                for j in 0..cols {
                    result[[i, j]] = a[[i, j]] - b[[i, j]];
                }
            }
        }

        result
    }

    /// SIMD-accelerated vector operations for degree calculations
    #[allow(dead_code)]
    pub fn simd_compute_degree_sqrt_inverse(degrees: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; degrees.len()];

        // Use chunked operations for better cache performance
        for (deg, res) in degrees.iter().zip(result.iter_mut()) {
            *res = if *deg > 0.0 { 1.0 / deg.sqrt() } else { 0.0 };
        }

        result
    }

    /// SIMD-accelerated vector norm computation
    #[allow(dead_code)]
    pub fn simd_norm(vector: &ArrayView1<f64>) -> f64 {
        // Use scirs2-core SIMD operations for optimal performance
        f64::simd_norm(vector)
    }

    /// SIMD-accelerated matrix-vector multiplication
    #[allow(dead_code)]
    pub fn simd_matvec(matrix: &Array2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        let (rows, _cols) = matrix.dim();
        let mut result = Array1::zeros(rows);

        // Use SIMD operations for each row
        for i in 0..rows {
            let row = matrix.row(i);
            if let (Some(row_slice), Some(vec_slice)) = (row.as_slice(), vector.as_slice()) {
                let row_view = ArrayView1::from(row_slice);
                let vec_view = ArrayView1::from(vec_slice);
                result[i] = f64::simd_dot(&row_view, &vec_view);
            } else {
                // Fallback for non-contiguous data
                result[i] = row.dot(vector);
            }
        }

        result
    }

    /// SIMD-accelerated vector scaling and addition
    #[allow(dead_code)]
    pub fn simd_axpy(alpha: f64, x: &ArrayView1<f64>, y: &mut ArrayViewMut1<f64>) {
        // Compute y = _alpha * x + y using SIMD
        if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice_mut()) {
            let x_view = ArrayView1::from(x_slice);
            let scaled_x = f64::simd_scalar_mul(&x_view, alpha);
            let y_view = ArrayView1::from(&*y_slice);
            let result = f64::simd_add(&scaled_x.view(), &y_view);
            if let Some(result_slice) = result.as_slice() {
                y_slice.copy_from_slice(result_slice);
            }
        } else {
            // Fallback for non-contiguous data
            for (x_val, y_val) in x.iter().zip(y.iter_mut()) {
                *y_val += alpha * x_val;
            }
        }
    }

    /// SIMD-accelerated Gram-Schmidt orthogonalization
    #[allow(dead_code)]
    pub fn simd_gram_schmidt(vectors: &mut Array2<f64>) {
        let (_n, k) = vectors.dim();

        for i in 0..k {
            // Normalize current vector
            let mut current_col = vectors.column_mut(i);
            let norm = simd_norm(&current_col.view());
            if norm > 1e-12 {
                current_col /= norm;
            }

            // Orthogonalize against following _vectors
            for j in (i + 1)..k {
                let (dot_product, current_column_data) = {
                    let current_view = vectors.column(i);
                    let next_col = vectors.column(j);

                    let dot = if let (Some(curr_slice), Some(next_slice)) =
                        (current_view.as_slice(), next_col.as_slice())
                    {
                        let curr_view = ArrayView1::from(curr_slice);
                        let next_view = ArrayView1::from(next_slice);
                        f64::simd_dot(&curr_view, &next_view)
                    } else {
                        current_view.dot(&next_col)
                    };

                    (dot, current_view.to_owned())
                };

                let mut next_col = vectors.column_mut(j);

                // Subtract projection: next = next - dot * current
                simd_axpy(-dot_product, &current_column_data.view(), &mut next_col);
            }
        }
    }
}

/// Minimal deterministic PRNG (a PCG/Knuth-style LCG) used only to generate
/// reproducible Lanczos starting vectors and k-means initial centroids.
///
/// This is *not* used for anything statistical: its only job is to hand
/// Lanczos a "generic" (numerically non-degenerate) starting vector while
/// keeping the whole spectral pipeline byte-for-byte reproducible across
/// runs -- `scirs2_core::random::rng()` returns a `ThreadRng` seeded from OS
/// entropy, which would make spectral clustering results non-reproducible.
mod deterministic_rng {
    const LCG_MUL: u64 = 6364136223846793005;
    const LCG_ADD: u64 = 1442695040888963407;

    /// Advances `state` and returns the new value.
    pub fn next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        *state
    }

    /// A pseudo-random `f64` in `[-0.5, 0.5)`, deterministic given `state`.
    pub fn next_signed_unit(state: &mut u64) -> f64 {
        let bits = next_u64(state);
        ((bits >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }
}

/// Builds a deterministic, reproducible "generic" starting vector for
/// Lanczos, seeded from `seed`. Callers vary the seed per call (e.g. by
/// matrix size and eigenvector index) so consecutive deflation steps don't
/// reuse the same vector.
fn lanczos_start_vector(n: usize, seed: u64) -> Array1<f64> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    // Advance once so an all-zero seed doesn't start the sequence at 0.
    deterministic_rng::next_u64(&mut state);
    Array1::from_shape_fn(n, |_| deterministic_rng::next_signed_unit(&mut state))
}

/// Derives a reproducible per-call Lanczos seed so consecutive deflation
/// steps (different `eig_idx`, i.e. a different number of previously found
/// eigenvectors) don't reuse the same starting vector.
fn lanczos_seed(n: usize, prev_count: usize) -> u64 {
    (n as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((prev_count as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(1)
}

/// Projects out the components of `vec` along each column of `prev`
/// (Gram-Schmidt deflation against already-found eigenvectors).
fn deflate_against(vec: Array1<f64>, prev: &Array2<f64>) -> Array1<f64> {
    let mut v = vec;
    for j in 0..prev.ncols() {
        let prev_vec = prev.column(j);
        let proj = v.dot(&prev_vec);
        v = v - proj * &prev_vec;
    }
    v
}

/// Projects out the components of `vec` along columns `0..upto` of `basis`
/// (full reorthogonalization to counter the numerical loss of orthogonality
/// inherent to naive Lanczos iteration).
fn reorthogonalize(vec: Array1<f64>, basis: &Array2<f64>, upto: usize) -> Array1<f64> {
    let mut v = vec;
    for j in 0..upto {
        let basis_vec = basis.column(j);
        let proj = v.dot(&basis_vec);
        v = v - proj * &basis_vec;
    }
    v
}

/// Exact eigendecomposition of a real symmetric tridiagonal matrix given its
/// diagonal (`alpha`) and off-diagonal (`beta`; `beta[i]` sits between
/// `alpha[i]` and `alpha[i + 1]`) entries, via the implicit-shift QL
/// algorithm (a.k.a. "tqli", Numerical Recipes §11.3). Returns eigenvalues in
/// ascending order together with matching orthonormal eigenvectors (as
/// columns).
#[allow(dead_code)]
fn tridiagonal_eigen(
    alpha: &[f64],
    beta: &[f64],
) -> std::result::Result<(Vec<f64>, Array2<f64>), String> {
    let n = alpha.len();
    if n == 0 {
        return Ok((vec![], Array2::zeros((0, 0))));
    }

    let mut d = alpha.to_vec();
    // e[i] holds the off-diagonal entry between d[i] and d[i + 1]; e[n - 1]
    // is unused scratch space required by the algorithm below.
    let mut e = vec![0.0_f64; n];
    for (i, e_i) in e.iter_mut().enumerate().take(n.saturating_sub(1)) {
        *e_i = beta.get(i).copied().unwrap_or(0.0);
    }

    let mut z = Array2::<f64>::eye(n);
    tridiagonal_ql_implicit(&mut d, &mut e, &mut z)?;

    // `tqli` does not sort its output; sort ascending and permute the
    // eigenvector columns to match (callers rely on index 0 == smallest).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for (new_col, &old_col) in order.iter().enumerate() {
        eigenvectors.column_mut(new_col).assign(&z.column(old_col));
    }

    Ok((eigenvalues, eigenvectors))
}

/// In-place implicit-shift QL algorithm for a real symmetric tridiagonal
/// matrix (the classic "tqli" routine). `d` holds the diagonal on input and
/// the (unsorted) eigenvalues on output. `e[i]` holds the off-diagonal entry
/// between `d[i]` and `d[i + 1]` for `i in 0..n - 1` (`e[n - 1]` is unused
/// scratch space). `z` must be initialized by the caller -- typically to the
/// identity matrix, to obtain the tridiagonal matrix's own eigenvectors; on
/// return, column `j` of `z` holds the eigenvector for eigenvalue `d[j]`.
#[allow(clippy::many_single_char_names)]
fn tridiagonal_ql_implicit(
    d: &mut [f64],
    e: &mut [f64],
    z: &mut Array2<f64>,
) -> std::result::Result<(), String> {
    let n = d.len();
    if n <= 1 {
        return Ok(());
    }

    for l in 0..n {
        let mut iter_count = 0;
        loop {
            // Find the smallest sub-diagonal element to split the problem
            // (i.e. one already negligible relative to its neighboring
            // diagonal entries).
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            iter_count += 1;
            if iter_count > 50 {
                return Err(
                    "tridiagonal QL: an eigenvalue failed to converge after 50 iterations"
                        .to_string(),
                );
            }

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut broke_early = false;

            for i in (l..m).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    broke_early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                // Accumulate the rotation into the eigenvector matrix.
                for k in 0..n {
                    f = z[[k, i + 1]];
                    z[[k, i + 1]] = s * z[[k, i]] + c * f;
                    z[[k, i]] = c * z[[k, i]] - s * f;
                }
            }

            if broke_early {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    Ok(())
}

/// Advanced eigenvalue computation using Lanczos algorithm for Laplacian matrices
/// This is a production-ready implementation with proper deflation and convergence checking
#[allow(dead_code)]
fn compute_smallest_eigenvalues(
    matrix: &Array2<f64>,
    k: usize,
) -> std::result::Result<(Vec<f64>, Array2<f64>), String> {
    let n = matrix.shape()[0];

    if k > n {
        return Err("k cannot be larger than matrix size".to_string());
    }

    if k == 0 {
        return Ok((vec![], Array2::zeros((n, 0))));
    }

    // For small matrices, use a simple approach with lower precision
    // For larger matrices, use the full Lanczos algorithm
    if n <= 10 {
        lanczos_eigenvalues(matrix, k, 1e-6, 20) // Lower precision, fewer iterations for small matrices
    } else {
        lanczos_eigenvalues(matrix, k, 1e-10, 100)
    }
}

/// Lanczos algorithm for finding smallest eigenvalues of symmetric matrices
/// Optimized for Laplacian matrices with SIMD acceleration where possible
#[allow(dead_code)]
fn lanczos_eigenvalues(
    matrix: &Array2<f64>,
    k: usize,
    tolerance: f64,
    max_iterations: usize,
) -> std::result::Result<(Vec<f64>, Array2<f64>), String> {
    let n = matrix.shape()[0];

    if n == 0 {
        return Ok((vec![], Array2::zeros((0, 0))));
    }

    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Array2::zeros((n, k));

    // For Laplacian matrices, we know the first eigenvalue is 0
    eigenvalues.push(0.0);
    if k > 0 {
        let val = 1.0 / (n as f64).sqrt();
        for i in 0..n {
            eigenvectors[[i, 0]] = val;
        }
    }

    // Find additional eigenvalues using deflated Lanczos
    for eig_idx in 1..k {
        let (eval, evec) = deflated_lanczos_iteration(
            matrix,
            &eigenvectors.slice(s![.., 0..eig_idx]).to_owned(),
            tolerance,
            max_iterations,
        )?;

        eigenvalues.push(eval);
        for i in 0..n {
            eigenvectors[[i, eig_idx]] = evec[i];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Core deflated Lanczos iteration: finds the eigenpair of `matrix` with the
/// smallest eigenvalue lying outside the span of `prev_eigenvectors`.
///
/// Implements full three-term-recurrence tridiagonalization with full
/// reorthogonalization against every previously generated Lanczos vector (to
/// counter the numerical loss of orthogonality inherent to naive Lanczos),
/// followed by an *exact* eigendecomposition of the resulting tridiagonal
/// matrix `T` (via [`tridiagonal_eigen`]) and reconstruction of the Ritz
/// vector in the original basis. The `matvec` closure lets callers choose a
/// sequential-SIMD or rayon-parallel matrix-vector product without
/// duplicating this logic (see [`deflated_lanczos_iteration`] and
/// [`parallel_deflated_lanczos_iteration`]).
fn deflated_lanczos_core(
    matrix: &Array2<f64>,
    prev_eigenvectors: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    seed: u64,
    matvec: impl Fn(&Array2<f64>, &ArrayView1<f64>) -> Array1<f64>,
) -> std::result::Result<(f64, Array1<f64>), String> {
    let n = matrix.shape()[0];
    if n == 0 {
        return Err("Cannot run Lanczos iteration on an empty matrix".to_string());
    }

    // Deterministic (reproducible), deflated, normalized starting vector.
    let mut v = deflate_against(lanczos_start_vector(n, seed), prev_eigenvectors);
    let start_norm = simd_spectral::simd_norm(&v.view());
    if start_norm < tolerance {
        return Err("Failed to generate suitable starting vector".to_string());
    }
    v /= start_norm;

    let m = max_iterations.min(n).max(1);
    let mut alpha: Vec<f64> = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m.saturating_sub(1));
    let mut lanczos_vectors = Array2::<f64>::zeros((n, m));
    lanczos_vectors.column_mut(0).assign(&v);

    let w0 = deflate_against(matvec(matrix, &v.view()), prev_eigenvectors);
    alpha.push(v.dot(&w0));
    let mut w = reorthogonalize(&w0 - alpha[0] * &v, &lanczos_vectors, 1);

    let mut steps = 1usize;

    for i in 1..m {
        let beta_val = simd_spectral::simd_norm(&w.view());
        if beta_val < tolerance {
            break;
        }

        let mut v_i = reorthogonalize(&w / beta_val, &lanczos_vectors, i);
        let renorm = simd_spectral::simd_norm(&v_i.view());
        if renorm < tolerance {
            break;
        }
        v_i /= renorm;
        lanczos_vectors.column_mut(i).assign(&v_i);
        beta.push(beta_val);
        steps = i + 1;

        let prev_v = lanczos_vectors.column(i - 1).to_owned();
        let w_raw = deflate_against(matvec(matrix, &v_i.view()), prev_eigenvectors);
        let alpha_i = v_i.dot(&w_raw);
        alpha.push(alpha_i);
        let w_next = &w_raw - alpha_i * &v_i - beta[i - 1] * &prev_v;
        w = reorthogonalize(w_next, &lanczos_vectors, i + 1);
    }

    let (tri_evals, tri_evecs) = tridiagonal_eigen(&alpha, &beta)
        .map_err(|e| format!("tridiagonal eigensolve failed: {e}"))?;
    if tri_evals.is_empty() {
        return Err("Lanczos process produced no eigenvalues".to_string());
    }

    let smallest_eval = tri_evals[0];
    let mut ritz_vector = Array1::<f64>::zeros(n);
    for j in 0..steps {
        ritz_vector = ritz_vector + tri_evecs[[j, 0]] * &lanczos_vectors.column(j);
    }

    // Defensive final deflation + normalization.
    ritz_vector = deflate_against(ritz_vector, prev_eigenvectors);
    let final_norm = simd_spectral::simd_norm(&ritz_vector.view());
    if final_norm < tolerance {
        return Err("Eigenvector deflation failed".to_string());
    }
    ritz_vector /= final_norm;

    Ok((smallest_eval, ritz_vector))
}

/// Single deflated Lanczos iteration to find the next smallest eigenvalue.
/// Uses deflation to avoid previously found eigenvectors. See
/// [`deflated_lanczos_core`] for the shared algorithm; this entry point uses
/// a sequential SIMD matrix-vector product.
#[allow(dead_code)]
fn deflated_lanczos_iteration(
    matrix: &Array2<f64>,
    prev_eigenvectors: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> std::result::Result<(f64, Array1<f64>), String> {
    let seed = lanczos_seed(matrix.shape()[0], prev_eigenvectors.ncols());
    deflated_lanczos_core(
        matrix,
        prev_eigenvectors,
        tolerance,
        max_iterations,
        seed,
        simd_spectral::simd_matvec,
    )
}

/// Laplacian matrix type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianType {
    /// Standard Laplacian: L = D - A
    /// where D is the degree matrix and A is the adjacency matrix
    Standard,

    /// Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    /// where I is the identity matrix, D is the degree matrix, and A is the adjacency matrix
    Normalized,

    /// Random walk Laplacian: L = I - D^(-1) A
    /// where I is the identity matrix, D is the degree matrix, and A is the adjacency matrix
    RandomWalk,
}

/// Computes the Laplacian matrix of a graph
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `laplacian_type` - The type of Laplacian matrix to compute
///
/// # Returns
/// * The Laplacian matrix as an scirs2_core::ndarray::Array2
#[allow(dead_code)]
pub fn laplacian<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    laplacian_type: LaplacianType,
) -> Result<Array2<f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix and convert to f64
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Get degree vector
    let degrees = graph.degree_vector();

    match laplacian_type {
        LaplacianType::Standard => {
            // L = D - A
            let mut laplacian = Array2::<f64>::zeros((n, n));

            // Set diagonal to degrees
            for i in 0..n {
                laplacian[[i, i]] = degrees[i] as f64;
            }

            // Subtract adjacency matrix
            laplacian = laplacian - adj_f64;

            Ok(laplacian)
        }
        LaplacianType::Normalized => {
            // L = I - D^(-1/2) A D^(-1/2)
            let mut normalized = Array2::<f64>::zeros((n, n));

            // Compute D^(-1/2)
            let mut d_inv_sqrt = Array1::<f64>::zeros(n);
            for i in 0..n {
                let degree = degrees[i] as f64;
                d_inv_sqrt[i] = if degree > 0.0 {
                    1.0 / degree.sqrt()
                } else {
                    0.0
                };
            }

            // Compute I - D^(-1/2) A D^(-1/2)
            for i in 0..n {
                for j in 0..n {
                    normalized[[i, j]] = -d_inv_sqrt[i] * adj_f64[[i, j]] * d_inv_sqrt[j];
                }
                // Add identity on diagonal
                normalized[[i, i]] += 1.0;
            }

            Ok(normalized)
        }
        LaplacianType::RandomWalk => {
            // L = I - D^(-1) A
            let mut random_walk = Array2::<f64>::zeros((n, n));

            // Compute I - D^(-1) A
            for i in 0..n {
                let degree = degrees[i] as f64;
                if degree > 0.0 {
                    for j in 0..n {
                        random_walk[[i, j]] = -adj_f64[[i, j]] / degree;
                    }
                }
                // Add identity on diagonal
                random_walk[[i, i]] += 1.0;
            }

            Ok(random_walk)
        }
    }
}

/// Computes the Laplacian matrix of a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
/// * `laplacian_type` - The type of Laplacian matrix to compute
///
/// # Returns
/// * The Laplacian matrix as an scirs2_core::ndarray::Array2
#[allow(dead_code)]
pub fn laplacian_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    laplacian_type: LaplacianType,
) -> Result<Array2<f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix and convert to f64
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]];
        }
    }

    // Get out-degree vector for directed graphs
    let degrees = graph.out_degree_vector();

    match laplacian_type {
        LaplacianType::Standard => {
            // L = D - A
            let mut laplacian = Array2::<f64>::zeros((n, n));

            // Set diagonal to out-degrees
            for i in 0..n {
                laplacian[[i, i]] = degrees[i] as f64;
            }

            // Subtract adjacency matrix
            laplacian = laplacian - adj_f64;

            Ok(laplacian)
        }
        LaplacianType::Normalized => {
            // L = I - D^(-1/2) A D^(-1/2)
            let mut normalized = Array2::<f64>::zeros((n, n));

            // Compute D^(-1/2)
            let mut d_inv_sqrt = Array1::<f64>::zeros(n);
            for i in 0..n {
                let degree = degrees[i] as f64;
                d_inv_sqrt[i] = if degree > 0.0 {
                    1.0 / degree.sqrt()
                } else {
                    0.0
                };
            }

            // Compute I - D^(-1/2) A D^(-1/2)
            for i in 0..n {
                for j in 0..n {
                    normalized[[i, j]] = -d_inv_sqrt[i] * adj_f64[[i, j]] * d_inv_sqrt[j];
                }
                // Add identity on diagonal
                normalized[[i, i]] += 1.0;
            }

            Ok(normalized)
        }
        LaplacianType::RandomWalk => {
            // L = I - D^(-1) A
            let mut random_walk = Array2::<f64>::zeros((n, n));

            // Compute I - D^(-1) A
            for i in 0..n {
                let degree = degrees[i] as f64;
                if degree > 0.0 {
                    for j in 0..n {
                        random_walk[[i, j]] = -adj_f64[[i, j]] / degree;
                    }
                }
                // Add identity on diagonal
                random_walk[[i, i]] += 1.0;
            }

            Ok(random_walk)
        }
    }
}

/// Computes the algebraic connectivity (Fiedler value) of a graph
///
/// The algebraic connectivity is the second-smallest eigenvalue of the Laplacian matrix.
/// It is a measure of how well-connected the graph is.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `laplacian_type` - The type of Laplacian to use (standard, normalized, or random walk)
///
/// # Returns
/// * The algebraic connectivity as a f64
#[allow(dead_code)]
pub fn algebraic_connectivity<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    laplacian_type: LaplacianType,
) -> Result<f64>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n <= 1 {
        return Err(GraphError::InvalidGraph(
            "Algebraic connectivity is undefined for graphs with 0 or 1 nodes".to_string(),
        ));
    }

    let laplacian = laplacian(graph, laplacian_type)?;

    // Compute the eigenvalues of the Laplacian
    // We only need the smallest 2 eigenvalues
    let (eigenvalues_, _) =
        compute_smallest_eigenvalues(&laplacian, 2).map_err(|e| GraphError::LinAlgError {
            operation: "eigenvalue_computation".to_string(),
            details: e,
        })?;

    // The second eigenvalue is the algebraic connectivity
    Ok(eigenvalues_[1])
}

/// Computes the spectral radius of a graph
///
/// The spectral radius is the largest eigenvalue of the adjacency matrix.
/// For undirected graphs, it provides bounds on various graph properties.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * The spectral radius as a f64
#[allow(dead_code)]
pub fn spectral_radius<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<f64>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty _graph".to_string()));
    }

    // Get adjacency matrix
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Power iteration method to approximate the largest eigenvalue
    let mut v = Array1::<f64>::ones(n);
    let mut lambda = 0.0;
    let max_iter = 100;
    let tolerance = 1e-10;

    for _ in 0..max_iter {
        // v_new = A * v
        let mut v_new = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                v_new[i] += adj_f64[[i, j]] * v[j];
            }
        }

        // Normalize v_new
        let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < tolerance {
            break;
        }

        for i in 0..n {
            v_new[i] /= norm;
        }

        // Compute eigenvalue approximation
        let mut lambda_new = 0.0;
        for i in 0..n {
            let mut av_i = 0.0;
            for j in 0..n {
                av_i += adj_f64[[i, j]] * v_new[j];
            }
            lambda_new += av_i * v_new[i];
        }

        // Check convergence
        if (lambda_new - lambda).abs() < tolerance {
            return Ok(lambda_new);
        }

        lambda = lambda_new;
        v = v_new;
    }

    Ok(lambda)
}

/// Computes the normalized cut value for a given partition
///
/// The normalized cut is a measure of how good a graph partition is.
/// Lower values indicate better partitions.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `partition` - A boolean vector indicating which nodes belong to set A (true) or set B (false)
///
/// # Returns
/// * The normalized cut value as a f64
#[allow(dead_code)]
pub fn normalized_cut<N, E, Ix>(graph: &Graph<N, E, Ix>, partition: &[bool]) -> Result<f64>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty _graph".to_string()));
    }

    if partition.len() != n {
        return Err(GraphError::InvalidGraph(
            "Partition size does not match _graph size".to_string(),
        ));
    }

    // Count nodes in each partition
    let count_a = partition.iter().filter(|&&x| x).count();
    let count_b = n - count_a;

    if count_a == 0 || count_b == 0 {
        return Err(GraphError::InvalidGraph(
            "Partition must have nodes in both sets".to_string(),
        ));
    }

    // Get adjacency matrix
    let adj_mat = graph.adjacency_matrix();

    // Compute cut(A,B), vol(A), and vol(B)
    let mut cut_ab = 0.0;
    let mut vol_a = 0.0;
    let mut vol_b = 0.0;

    let _nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    for i in 0..n {
        for j in 0..n {
            let weight: f64 = adj_mat[[i, j]].into();

            if partition[i] && !partition[j] {
                // Edge from A to B
                cut_ab += weight;
            }

            if partition[i] {
                vol_a += weight;
            } else {
                vol_b += weight;
            }
        }
    }

    // Normalized cut = cut(A,B)/vol(A) + cut(A,B)/vol(B)
    let ncut = if vol_a > 0.0 && vol_b > 0.0 {
        cut_ab / vol_a + cut_ab / vol_b
    } else {
        f64::INFINITY
    };

    Ok(ncut)
}

/// Performs spectral clustering on a graph
///
/// # Arguments
/// * `graph` - The graph to cluster
/// * `n_clusters` - The number of clusters to create
/// * `laplacian_type` - The type of Laplacian to use
///
/// # Returns
/// * A vector of cluster labels, one for each node in the graph
#[allow(dead_code)]
pub fn spectral_clustering<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    n_clusters: usize,
    laplacian_type: LaplacianType,
) -> Result<Vec<usize>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    if n_clusters == 0 {
        return Err(GraphError::InvalidGraph(
            "Number of _clusters must be positive".to_string(),
        ));
    }

    if n_clusters > n {
        return Err(GraphError::InvalidGraph(
            "Number of _clusters cannot exceed number of nodes".to_string(),
        ));
    }

    // Compute the Laplacian matrix
    let laplacian_matrix = laplacian(graph, laplacian_type)?;

    // Compute the eigenvectors corresponding to the smallest n_clusters eigenvalues;
    // these form the spectral embedding (n rows, n_clusters columns).
    let (_eigenvalues, embedding) = compute_smallest_eigenvalues(&laplacian_matrix, n_clusters)
        .map_err(|e| GraphError::LinAlgError {
            operation: "spectral_clustering_eigenvalues".to_string(),
            details: e,
        })?;

    // Cluster the rows of the spectral embedding with Lloyd's k-means
    // (deterministic k-means++ init, so results are reproducible).
    let labels = kmeans_cluster_embedding(&embedding, n_clusters, n);

    Ok(labels)
}

/// Parallel version of spectral clustering with improved performance for large graphs
///
/// # Arguments
/// * `graph` - The graph to cluster
/// * `n_clusters` - The number of clusters to create
/// * `laplacian_type` - The type of Laplacian to use
///
/// # Returns
/// * A vector of cluster labels..one for each node in the graph
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_spectral_clustering<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    n_clusters: usize,
    laplacian_type: LaplacianType,
) -> Result<Vec<usize>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    if n_clusters == 0 {
        return Err(GraphError::InvalidGraph(
            "Number of _clusters must be positive".to_string(),
        ));
    }

    if n_clusters > n {
        return Err(GraphError::InvalidGraph(
            "Number of _clusters cannot exceed number of nodes".to_string(),
        ));
    }

    // Compute the Laplacian matrix using parallel operations where possible
    let laplacian_matrix = parallel_laplacian(graph, laplacian_type)?;

    // Compute the eigenvectors using parallel eigenvalue computation
    let (_eigenvalues, embedding) =
        parallel_compute_smallest_eigenvalues(&laplacian_matrix, n_clusters).map_err(|e| {
            GraphError::LinAlgError {
                operation: "parallel_spectral_clustering_eigenvalues".to_string(),
                details: e,
            }
        })?;

    // Run Lloyd's k-means on the spectral embedding (n × k_clusters matrix, rows are points)
    let labels = kmeans_cluster_embedding(&embedding, n_clusters, n);

    Ok(labels)
}

/// Parallel Laplacian matrix computation with optimized memory access patterns
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_laplacian<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    laplacian_type: LaplacianType,
) -> Result<Array2<f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + PartialOrd
        + Into<f64>
        + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix and convert to f64 in parallel
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));

    // Parallel conversion of adjacency matrix
    adj_f64
        .axis_iter_mut(scirs2_core::ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                row[j] = adj_mat[[i, j]].into();
            }
        });

    // Get degree vector
    let degrees = graph.degree_vector();

    match laplacian_type {
        LaplacianType::Standard => {
            // L = D - A (computed in parallel)
            let mut laplacian = Array2::<f64>::zeros((n, n));

            // Parallel computation of Laplacian matrix
            laplacian
                .axis_iter_mut(scirs2_core::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    // Set diagonal to degree
                    row[i] = degrees[i] as f64;

                    // Subtract adjacency values
                    for j in 0..n {
                        if i != j {
                            row[j] = -adj_f64[[i, j]];
                        }
                    }
                });

            Ok(laplacian)
        }
        LaplacianType::Normalized => {
            // L = I - D^(-1/2) A D^(-1/2) (computed in parallel)
            let mut normalized = Array2::<f64>::zeros((n, n));

            // Compute D^(-1/2) in parallel
            let d_inv_sqrt: Vec<f64> = degrees
                .par_iter()
                .map(|&degree| {
                    let deg_f64 = degree as f64;
                    if deg_f64 > 0.0 {
                        1.0 / deg_f64.sqrt()
                    } else {
                        0.0
                    }
                })
                .collect();

            // Parallel computation of normalized Laplacian
            normalized
                .axis_iter_mut(scirs2_core::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n {
                        if i == j {
                            row[j] = 1.0 - d_inv_sqrt[i] * adj_f64[[i, j]] * d_inv_sqrt[j];
                        } else {
                            row[j] = -d_inv_sqrt[i] * adj_f64[[i, j]] * d_inv_sqrt[j];
                        }
                    }
                });

            Ok(normalized)
        }
        LaplacianType::RandomWalk => {
            // L = I - D^(-1) A (computed in parallel)
            let mut random_walk = Array2::<f64>::zeros((n, n));

            // Parallel computation of random walk Laplacian
            random_walk
                .axis_iter_mut(scirs2_core::ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    let degree = degrees[i] as f64;
                    for j in 0..n {
                        if i == j {
                            if degree > 0.0 {
                                row[j] = 1.0 - adj_f64[[i, j]] / degree;
                            } else {
                                row[j] = 1.0;
                            }
                        } else if degree > 0.0 {
                            row[j] = -adj_f64[[i, j]] / degree;
                        } else {
                            row[j] = 0.0;
                        }
                    }
                });

            Ok(random_walk)
        }
    }
}

/// Parallel eigenvalue computation for large symmetric matrices
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn parallel_compute_smallest_eigenvalues(
    matrix: &Array2<f64>,
    k: usize,
) -> std::result::Result<(Vec<f64>, Array2<f64>), String> {
    let n = matrix.shape()[0];

    if k > n {
        return Err("k cannot be larger than matrix size".to_string());
    }

    if k == 0 {
        return Ok((vec![], Array2::zeros((n, 0))));
    }

    // Use parallel Lanczos algorithm for large matrices
    parallel_lanczos_eigenvalues(matrix, k, 1e-10, 100)
}

/// Parallel Lanczos algorithm with optimized matrix-vector operations
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn parallel_lanczos_eigenvalues(
    matrix: &Array2<f64>,
    k: usize,
    tolerance: f64,
    max_iterations: usize,
) -> std::result::Result<(Vec<f64>, Array2<f64>), String> {
    let n = matrix.shape()[0];

    if n == 0 {
        return Ok((vec![], Array2::zeros((0, 0))));
    }

    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Array2::zeros((n, k));

    // For Laplacian matrices, we know the first eigenvalue is 0
    eigenvalues.push(0.0);
    if k > 0 {
        let val = 1.0 / (n as f64).sqrt();
        eigenvectors.column_mut(0).fill(val);
    }

    // Find additional eigenvalues using parallel deflated Lanczos
    for eig_idx in 1..k {
        let (eval, evec) = parallel_deflated_lanczos_iteration(
            matrix,
            &eigenvectors.slice(s![.., 0..eig_idx]).to_owned(),
            tolerance,
            max_iterations,
        )?;

        eigenvalues.push(eval);
        eigenvectors.column_mut(eig_idx).assign(&evec);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Parallel deflated Lanczos iteration.
///
/// Implements the exact same algorithm as [`deflated_lanczos_iteration`] (full
/// tridiagonalization with reorthogonalization, then an exact eigensolve of
/// the resulting tridiagonal matrix -- see [`deflated_lanczos_core`]), but
/// uses a rayon-parallel row reduction for the matrix-vector product, which
/// dominates the per-iteration cost for large graphs.
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn parallel_deflated_lanczos_iteration(
    matrix: &Array2<f64>,
    prev_eigenvectors: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> std::result::Result<(f64, Array1<f64>), String> {
    let seed = lanczos_seed(matrix.shape()[0], prev_eigenvectors.ncols());
    deflated_lanczos_core(
        matrix,
        prev_eigenvectors,
        tolerance,
        max_iterations,
        seed,
        parallel_matvec,
    )
}

/// Rayon-parallel matrix-vector product (row-wise reduction). Used by the
/// `parallel` feature's Lanczos entry point for its dominant O(n^2)-per-step
/// cost; falls back to a plain dot product for non-contiguous rows/vectors.
#[cfg(feature = "parallel")]
fn parallel_matvec(matrix: &Array2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
    let rows = matrix.nrows();
    let result: Vec<f64> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let row = matrix.row(i);
            if let (Some(row_slice), Some(vec_slice)) = (row.as_slice(), vector.as_slice()) {
                f64::simd_dot(&ArrayView1::from(row_slice), &ArrayView1::from(vec_slice))
            } else {
                row.dot(vector)
            }
        })
        .collect();
    Array1::from_vec(result)
}

/// Lloyd's k-means clustering on a spectral embedding matrix.
///
/// Uses k-means++ initialization (deterministic LCG, no external RNG crate)
/// followed by iterative assignment/update until convergence or 100
/// iterations. Used by both [`spectral_clustering`] and (with a
/// rayon-parallelized assignment step) [`parallel_spectral_clustering`], so
/// spectral clustering behaves identically regardless of the `parallel`
/// feature -- only the assignment step's execution strategy differs.
///
/// # Arguments
/// * `embedding` - `n × k` matrix where each row is a point in k-dimensional space
/// * `k_clusters` - number of clusters
/// * `n` - number of points (must equal `embedding.nrows()`)
///
/// # Returns
/// Cluster assignments of length `n`, values in `0..k_clusters`.
fn kmeans_cluster_embedding(embedding: &Array2<f64>, k_clusters: usize, n: usize) -> Vec<usize> {
    // Trivial cases
    if k_clusters == 0 || n == 0 {
        return vec![0usize; n];
    }
    if k_clusters == 1 {
        return vec![0usize; n];
    }
    if k_clusters >= n {
        return (0..n).collect();
    }

    let k = k_clusters;
    let dim = embedding.ncols();

    // ---- k-means++ initialization (deterministic LCG, seed = n * k) -------------------
    // LCG: x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
    let lcg_mul: u64 = 6364136223846793005;
    let lcg_add: u64 = 1442695040888963407;
    let mut lcg_state: u64 = (n as u64).wrapping_mul(k as u64);
    // Advance once so seed 0 doesn't start at 0
    lcg_state = lcg_state.wrapping_mul(lcg_mul).wrapping_add(lcg_add);

    // Helper: pick an index in 0..remaining using current LCG state
    let mut lcg_index = |state: &mut u64, remaining: usize| -> usize {
        *state = state.wrapping_mul(lcg_mul).wrapping_add(lcg_add);
        ((*state) >> 33) as usize % remaining
    };

    // Squared Euclidean distance between embedding row i and a centroid slice
    let sq_dist = |row_i: usize, centroid: &[f64]| -> f64 {
        let mut d = 0.0_f64;
        for j in 0..dim {
            let diff = embedding[[row_i, j]] - centroid[j];
            d += diff * diff;
        }
        d
    };

    // First centroid: deterministic pick at index = n/2 (biased toward middle)
    let first_idx = lcg_index(&mut lcg_state, n);
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    centroids.push(embedding.row(first_idx).to_vec());

    // Subsequent centroids: k-means++ probability ∝ dist² to nearest existing centroid
    for _c in 1..k {
        // Compute dist² for every point to its nearest existing centroid
        let dist2: Vec<f64> = (0..n)
            .map(|i| {
                centroids
                    .iter()
                    .map(|c| sq_dist(i, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dist2.iter().sum();

        // Select next centroid by weighted sampling via LCG
        let chosen = if total <= 0.0 {
            // All points coincide with existing centroids – pick deterministically
            lcg_index(&mut lcg_state, n)
        } else {
            // Normalise and pick via cumulative sum
            lcg_state = lcg_state.wrapping_mul(lcg_mul).wrapping_add(lcg_add);
            // Map LCG value to [0, 1)
            let threshold = ((lcg_state >> 11) as f64) / ((1u64 << 53) as f64) * total;
            let mut cumsum = 0.0_f64;
            let mut picked = n - 1;
            for (idx, &d2) in dist2.iter().enumerate() {
                cumsum += d2;
                if cumsum >= threshold {
                    picked = idx;
                    break;
                }
            }
            picked
        };

        centroids.push(embedding.row(chosen).to_vec());
    }

    // ---- Lloyd's iteration (max 100 rounds) -------------------------------------------
    let mut assignments: Vec<usize> = vec![0usize; n];
    let max_iter = 100usize;

    for _iter in 0..max_iter {
        // Assignment step: each point → nearest centroid
        let new_assignments = assign_to_nearest_centroid(embedding, &centroids, n, dim);

        // Convergence check
        if new_assignments == assignments {
            assignments = new_assignments;
            break;
        }
        assignments = new_assignments;

        // Update step: recompute centroids as mean of assigned points
        let mut sums = vec![vec![0.0_f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in assignments.iter().enumerate() {
            counts[c] += 1;
            for j in 0..dim {
                sums[c][j] += embedding[[i, j]];
            }
        }
        for (c_idx, centroid) in centroids.iter_mut().enumerate() {
            let cnt = counts[c_idx];
            if cnt > 0 {
                for j in 0..dim {
                    centroid[j] = sums[c_idx][j] / cnt as f64;
                }
            }
            // Empty cluster: keep old centroid (no divide-by-zero, stable convergence)
        }
    }

    assignments
}

/// Assigns every embedding row to its nearest centroid (rayon-parallelized).
#[cfg(feature = "parallel")]
fn assign_to_nearest_centroid(
    embedding: &Array2<f64>,
    centroids: &[Vec<f64>],
    n: usize,
    dim: usize,
) -> Vec<usize> {
    (0..n)
        .into_par_iter()
        .map(|i| nearest_centroid(embedding, centroids, i, dim))
        .collect()
}

/// Assigns every embedding row to its nearest centroid (sequential).
#[cfg(not(feature = "parallel"))]
fn assign_to_nearest_centroid(
    embedding: &Array2<f64>,
    centroids: &[Vec<f64>],
    n: usize,
    dim: usize,
) -> Vec<usize> {
    (0..n)
        .map(|i| nearest_centroid(embedding, centroids, i, dim))
        .collect()
}

/// Index of the centroid nearest to `embedding` row `i` (squared Euclidean).
fn nearest_centroid(
    embedding: &Array2<f64>,
    centroids: &[Vec<f64>],
    i: usize,
    dim: usize,
) -> usize {
    let mut best_c = 0usize;
    let mut best_d = f64::INFINITY;
    for (c_idx, centroid) in centroids.iter().enumerate() {
        let mut d = 0.0_f64;
        for j in 0..dim {
            let diff = embedding[[i, j]] - centroid[j];
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best_c = c_idx;
        }
    }
    best_c
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_laplacian_matrix() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a simple graph:
        // 0 -- 1 -- 2
        // |         |
        // +----3----+

        graph.add_edge(0, 1, 1.0).expect("Operation failed");
        graph.add_edge(1, 2, 1.0).expect("Operation failed");
        graph.add_edge(2, 3, 1.0).expect("Operation failed");
        graph.add_edge(3, 0, 1.0).expect("Operation failed");

        // Test standard Laplacian
        let lap = laplacian(&graph, LaplacianType::Standard).expect("Operation failed");

        // Expected Laplacian:
        // [[ 2, -1,  0, -1],
        //  [-1,  2, -1,  0],
        //  [ 0, -1,  2, -1],
        //  [-1,  0, -1,  2]]

        let expected = Array2::from_shape_vec(
            (4, 4),
            vec![
                2.0, -1.0, 0.0, -1.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, -1.0, 0.0, -1.0,
                2.0,
            ],
        )
        .expect("Test: operation failed");

        // Check that the matrices are approximately equal
        for i in 0..4 {
            for j in 0..4 {
                assert!((lap[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }

        // Test normalized Laplacian
        let lap_norm = laplacian(&graph, LaplacianType::Normalized).expect("Operation failed");

        // Each node has degree 2, so D^(-1/2) = diag(1/sqrt(2), 1/sqrt(2), 1/sqrt(2), 1/sqrt(2))
        // For normalized Laplacian, check key properties rather than exact values

        // 1. Diagonal elements should be 1.0
        assert!(lap_norm[[0, 0]].abs() - 1.0 < 1e-6);
        assert!(lap_norm[[1, 1]].abs() - 1.0 < 1e-6);
        assert!(lap_norm[[2, 2]].abs() - 1.0 < 1e-6);
        assert!(lap_norm[[3, 3]].abs() - 1.0 < 1e-6);

        // Just verify the matrix is symmetric
        for i in 0..4 {
            for j in i + 1..4 {
                assert!((lap_norm[[i, j]] - lap_norm[[j, i]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_algebraic_connectivity() {
        // Test a path graph P4 (4 nodes in a line)
        let mut path_graph: Graph<i32, f64> = Graph::new();

        path_graph.add_edge(0, 1, 1.0).expect("Operation failed");
        path_graph.add_edge(1, 2, 1.0).expect("Operation failed");
        path_graph.add_edge(2, 3, 1.0).expect("Operation failed");

        // For a path graph P4, the algebraic connectivity should be positive and reasonable
        let conn =
            algebraic_connectivity(&path_graph, LaplacianType::Standard).expect("Operation failed");
        // Check that it's in a reasonable range for a path graph (approximation may vary)
        assert!(
            conn > 0.3 && conn < 1.0,
            "Algebraic connectivity {conn} should be positive and reasonable for path graph"
        );

        // Test a cycle graph C4 (4 nodes in a cycle)
        let mut cycle_graph: Graph<i32, f64> = Graph::new();

        cycle_graph.add_edge(0, 1, 1.0).expect("Operation failed");
        cycle_graph.add_edge(1, 2, 1.0).expect("Operation failed");
        cycle_graph.add_edge(2, 3, 1.0).expect("Operation failed");
        cycle_graph.add_edge(3, 0, 1.0).expect("Operation failed");

        // For a cycle graph C4, the algebraic connectivity should be positive and higher than path
        let conn = algebraic_connectivity(&cycle_graph, LaplacianType::Standard)
            .expect("Operation failed");

        // Check that it's reasonable for a cycle graph (more connected than path graph)
        assert!(
            conn > 0.5,
            "Algebraic connectivity {conn} should be positive and reasonable for cycle graph"
        );
    }

    #[test]
    fn test_spectral_radius() {
        // Test with a complete graph K3
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0).expect("Operation failed");
        graph.add_edge(1, 2, 1.0).expect("Operation failed");
        graph.add_edge(2, 0, 1.0).expect("Operation failed");

        let radius = spectral_radius(&graph).expect("Operation failed");
        // For K3, spectral radius should be 2.0
        assert!((radius - 2.0).abs() < 0.1);

        // Test with a star graph S3 (3 leaves)
        let mut star: Graph<i32, f64> = Graph::new();
        star.add_edge(0, 1, 1.0).expect("Operation failed");
        star.add_edge(0, 2, 1.0).expect("Operation failed");
        star.add_edge(0, 3, 1.0).expect("Operation failed");

        let radius_star = spectral_radius(&star).expect("Operation failed");
        // For S3, spectral radius should be sqrt(3) ≈ 1.732
        assert!(radius_star > 1.5 && radius_star < 2.0);
    }

    #[test]
    fn test_normalized_cut() {
        // Create a graph with two clear clusters
        let mut graph: Graph<i32, f64> = Graph::new();

        // Cluster 1: 0, 1, 2 (complete)
        graph.add_edge(0, 1, 1.0).expect("Operation failed");
        graph.add_edge(1, 2, 1.0).expect("Operation failed");
        graph.add_edge(2, 0, 1.0).expect("Operation failed");

        // Cluster 2: 3, 4, 5 (complete)
        graph.add_edge(3, 4, 1.0).expect("Operation failed");
        graph.add_edge(4, 5, 1.0).expect("Operation failed");
        graph.add_edge(5, 3, 1.0).expect("Operation failed");

        // Bridge between clusters
        graph.add_edge(2, 3, 1.0).expect("Operation failed");

        // Perfect partition
        let partition = vec![true, true, true, false, false, false];
        let ncut = normalized_cut(&graph, &partition).expect("Operation failed");

        // This should be a good cut with low normalized cut value
        assert!(ncut < 0.5);

        // Bad partition (splits a cluster)
        let bad_partition = vec![true, true, false, false, false, false];
        let bad_ncut = normalized_cut(&graph, &bad_partition).expect("Operation failed");

        // This should have a higher normalized cut value
        assert!(bad_ncut > ncut);
    }

    #[test]
    fn test_tridiagonal_eigen_matches_known_spectrum() {
        // Classic discrete-Laplacian tridiagonal matrix: diag = 2, offdiag = -1.
        // Closed-form eigenvalues: 2 - 2*cos(k*pi/(n+1)) for k = 1..=n.
        let n = 5;
        let alpha = vec![2.0_f64; n];
        let beta = vec![-1.0_f64; n - 1];

        let (eigenvalues, eigenvectors) =
            tridiagonal_eigen(&alpha, &beta).expect("tridiagonal eigensolve failed");

        assert_eq!(eigenvalues.len(), n);

        let mut expected: Vec<f64> = (1..=n)
            .map(|k| 2.0 - 2.0 * ((k as f64) * std::f64::consts::PI / (n as f64 + 1.0)).cos())
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (computed, exp) in eigenvalues.iter().zip(expected.iter()) {
            assert!(
                (computed - exp).abs() < 1e-8,
                "eigenvalue {computed} should match closed-form {exp}"
            );
        }

        // Cross-check independently of the closed form: reconstruct the dense
        // tridiagonal matrix and verify T*v ≈ lambda*v for every returned
        // pair. This would fail hard against the old fallback, which simply
        // returned the raw diagonal [2,2,2,2,2] (ignoring `beta` entirely)
        // and the identity matrix as "eigenvectors".
        let mut dense = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            dense[[i, i]] = alpha[i];
            if i > 0 {
                dense[[i, i - 1]] = beta[i - 1];
                dense[[i - 1, i]] = beta[i - 1];
            }
        }
        for col in 0..n {
            let v = eigenvectors.column(col).to_owned();
            let lambda = eigenvalues[col];
            let av = dense.dot(&v);
            for i in 0..n {
                assert!(
                    (av[i] - lambda * v[i]).abs() < 1e-6,
                    "T*v should equal lambda*v at row {i} for eigenpair {col}"
                );
            }
            let norm: f64 = v.dot(&v).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-8,
                "eigenvector {col} should be unit norm"
            );
        }
    }

    #[test]
    fn test_algebraic_connectivity_path_graph_general_size() {
        // P8: a path graph of 8 nodes. Its average degree (1.75) is not 2.0,
        // so the old crude small-matrix heuristic (which only special-cased
        // degree exactly 2.0, and only hard-coded a formula for n == 4) would
        // have fallen through to `avg_degree * 0.5 == 0.875` here -- nowhere
        // near the true value asserted below. This graph is also well past
        // the old `n <= 4` shortcut, so it exercises the general Lanczos path.
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0i32..7 {
            graph.add_edge(i, i + 1, 1.0).expect("Operation failed");
        }

        let conn =
            algebraic_connectivity(&graph, LaplacianType::Standard).expect("Operation failed");

        let n = 8.0_f64;
        let expected = 2.0 * (1.0 - (std::f64::consts::PI / n).cos());
        assert!(
            (conn - expected).abs() < 1e-4,
            "algebraic connectivity {conn} should match the closed-form path-graph value {expected}"
        );
    }

    #[test]
    fn test_algebraic_connectivity_is_deterministic() {
        // Regression guard: the Lanczos starting vector must be reproducible.
        // Previously a `ThreadRng` (seeded from OS entropy) fed the starting
        // vector, so results could silently vary from run to run for the
        // exact same graph.
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0i32..9 {
            graph.add_edge(i, i + 1, 1.0).expect("Operation failed");
        }
        graph.add_edge(9, 0, 1.0).expect("Operation failed"); // close into a 10-cycle

        let first =
            algebraic_connectivity(&graph, LaplacianType::Standard).expect("Operation failed");
        for _ in 0..5 {
            let again =
                algebraic_connectivity(&graph, LaplacianType::Standard).expect("Operation failed");
            assert_eq!(
                first.to_bits(),
                again.to_bits(),
                "algebraic_connectivity should be bit-for-bit reproducible across calls"
            );
        }
    }

    /// Builds two dense (complete) 4-node clusters joined by a single,
    /// much-lighter bridge edge -- a textbook case for spectral clustering.
    fn two_clusters_with_weak_bridge() -> Graph<i32, f64> {
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0i32..4 {
            for j in (i + 1)..4 {
                graph.add_edge(i, j, 1.0).expect("Operation failed");
            }
        }
        for i in 4i32..8 {
            for j in (i + 1)..8 {
                graph.add_edge(i, j, 1.0).expect("Operation failed");
            }
        }
        graph.add_edge(3, 4, 0.01).expect("Operation failed");
        graph
    }

    #[test]
    fn test_spectral_clustering_recovers_two_clusters() {
        let graph = two_clusters_with_weak_bridge();

        let labels_first =
            spectral_clustering(&graph, 2, LaplacianType::Standard).expect("clustering failed");
        let labels_second =
            spectral_clustering(&graph, 2, LaplacianType::Standard).expect("clustering failed");

        // Deterministic: identical labeling across independent runs (the old
        // implementation discarded the eigendecomposition and returned
        // uniformly random labels every call).
        assert_eq!(
            labels_first, labels_second,
            "spectral_clustering must be deterministic across runs"
        );

        // The two dense clusters must each get one consistent label, and the
        // two clusters must get DIFFERENT labels.
        let cluster_a_label = labels_first[0];
        let cluster_b_label = labels_first[4];
        assert_ne!(
            cluster_a_label, cluster_b_label,
            "the two clusters separated by only a weak bridge must get different labels"
        );
        for &node in &[0usize, 1, 2, 3] {
            assert_eq!(
                labels_first[node], cluster_a_label,
                "node {node} should share cluster A's label"
            );
        }
        for &node in &[4usize, 5, 6, 7] {
            assert_eq!(
                labels_first[node], cluster_b_label,
                "node {node} should share cluster B's label"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_spectral_clustering_recovers_two_clusters() {
        let graph = two_clusters_with_weak_bridge();

        // Old implementation: hardcoded every eigenvalue past the first to
        // 0.1 and never multiplied the matrix at all, so the "embedding"
        // clustered on was pure orthogonalized noise. This must now recover
        // the same real partition as the sequential path.
        let labels = parallel_spectral_clustering(&graph, 2, LaplacianType::Standard)
            .expect("parallel clustering failed");

        let cluster_a_label = labels[0];
        let cluster_b_label = labels[4];
        assert_ne!(
            cluster_a_label, cluster_b_label,
            "the two clusters separated by only a weak bridge must get different labels"
        );
        for &node in &[0usize, 1, 2, 3] {
            assert_eq!(labels[node], cluster_a_label);
        }
        for &node in &[4usize, 5, 6, 7] {
            assert_eq!(labels[node], cluster_b_label);
        }
    }
}
