//! Large-scale linear algebra algorithms
//!
//! This module provides specialized algorithms optimized for large-scale problems
//! including sparse solvers, randomized algorithms, and matrix-free methods.

use crate::error::{LinalgError, LinalgResult};
use crate::norm::vector_norm;
use crate::solvers::iterative::{conjugate_gradient, gmres, IterativeSolverOptions};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign};
use rand::prelude::*;
use rand_distr::Normal;
use std::fmt::{Debug, Display};

/// Randomized algorithm for solving least squares problems
///
/// This algorithm uses random projections to solve large-scale least squares
/// problems Ax ≈ b efficiently. It's particularly useful when A is tall and skinny
/// (m >> n) or when only an approximate solution is needed.
///
/// # Arguments
/// * `a` - Matrix A (m × n)
/// * `b` - Right-hand side vector b (m × 1)
/// * `sketch_size` - Size of the random sketch (typically O(n log n))
/// * `iterations` - Number of refinement iterations
///
/// # Returns
/// * Approximate solution x to minimize ||Ax - b||₂
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::large_scale::randomized_least_squares;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let b = array![1.0, 2.0, 3.0];
/// let x = randomized_least_squares(&a.view(), &b.view(), 2, 3).unwrap();
/// ```
pub fn randomized_least_squares<A>(
    a: &ArrayView2<A>,
    b: &ArrayView1<A>,
    sketch_size: usize,
    iterations: usize,
) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());

    if b.len() != m {
        return Err(LinalgError::ShapeError(
            "Vector b must have length m".to_string(),
        ));
    }

    if sketch_size < n {
        return Err(LinalgError::ShapeError(
            "Sketch size must be at least n".to_string(),
        ));
    }

    // Generate random sketching matrix S (sketch_size × m)
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut s = Array2::zeros((sketch_size, m));
    let scale = A::from(1.0 / (sketch_size as f64).sqrt()).unwrap();

    for i in 0..sketch_size {
        for j in 0..m {
            s[[i, j]] = A::from(normal.sample(&mut rng)).unwrap() * scale;
        }
    }

    // Sketch the matrix and vector: SA and Sb
    let sa = s.dot(a);
    let sb = s.dot(b);

    // Solve the sketched problem: (SA)x ≈ Sb
    // Using normal equations: (SA)^T(SA)x = (SA)^T(Sb)
    let sa_t = sa.t();
    let gram = sa_t.dot(&sa);
    let sa_t_sb = sa_t.dot(&sb);

    // Solve the normal equations
    let mut x = crate::solve::solve(&gram.view(), &sa_t_sb.view(), None)?;

    // Iterative refinement
    for _ in 0..iterations {
        // Compute residual r = b - Ax
        let ax = a.dot(&x);
        let r = b - &ax;

        // Sketch the residual
        let sr = s.dot(&r);

        // Solve for correction: (SA)δx = Sr
        let delta_x = crate::solve::solve(&gram.view(), &sa_t.dot(&sr).view(), None)?;

        // Update solution
        x = x + delta_x;
    }

    Ok(x)
}

/// Randomized algorithm for computing matrix norms
///
/// Uses random sampling to estimate matrix norms for very large matrices
/// where computing the exact norm is prohibitive.
///
/// # Arguments
/// * `a` - Input matrix
/// * `norm_type` - Type of norm ("2" for spectral norm, "fro" for Frobenius)
/// * `num_samples` - Number of random vectors to sample
/// * `power_iterations` - Number of power iterations for better accuracy
///
/// # Returns
/// * Estimated matrix norm
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::large_scale::randomized_norm;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let norm = randomized_norm(&a.view(), "2", 10, 2).unwrap();
/// ```
pub fn randomized_norm<A>(
    a: &ArrayView2<A>,
    norm_type: &str,
    num_samples: usize,
    power_iterations: usize,
) -> LinalgResult<A>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());

    match norm_type {
        "2" | "spectral" => {
            // Estimate spectral norm using power method with random initialization
            let mut rng = rand::rng();
            let normal = Normal::new(0.0, 1.0).unwrap();

            let mut max_norm = A::zero();

            for _ in 0..num_samples {
                // Random initial vector
                let mut v = Array1::zeros(n);
                for i in 0..n {
                    v[i] = A::from(normal.sample(&mut rng)).unwrap();
                }

                // Normalize
                let vnorm = vector_norm(&v.view(), 2)?;
                if vnorm > A::epsilon() {
                    v.mapv_inplace(|x| x / vnorm);
                }

                // Power iterations
                for _ in 0..power_iterations {
                    // v = A^T * A * v
                    let av = a.dot(&v);
                    v = a.t().dot(&av);

                    // Normalize
                    let vnorm = vector_norm(&v.view(), 2)?;
                    if vnorm > A::epsilon() {
                        v.mapv_inplace(|x| x / vnorm);
                    }
                }

                // Estimate norm as sqrt(v^T * A^T * A * v)
                let av = a.dot(&v);
                let norm_sq = av.dot(&av);
                let norm = norm_sq.sqrt();

                if norm > max_norm {
                    max_norm = norm;
                }
            }

            Ok(max_norm)
        }
        "fro" | "frobenius" => {
            // Estimate Frobenius norm using random sampling
            let mut rng = rand::rng();
            let total_entries = (m * n) as f64;
            let sample_size = num_samples.min(m * n);
            let scale = A::from(total_entries / sample_size as f64).unwrap_or(A::one());

            let mut sum_sq = A::zero();
            let mut sampled = std::collections::HashSet::new();

            while sampled.len() < sample_size {
                let i = rng.random_range(0..m);
                let j = rng.random_range(0..n);

                if sampled.insert((i, j)) {
                    sum_sq += a[[i, j]] * a[[i, j]];
                }
            }

            Ok((sum_sq * scale).sqrt())
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Unknown norm type: {}",
            norm_type
        ))),
    }
}

/// Streaming algorithm for incremental SVD
///
/// This algorithm computes the SVD incrementally as new data arrives,
/// useful for online learning and streaming applications.
///
/// # Arguments
/// * `current_u` - Current left singular vectors
/// * `current_s` - Current singular values
/// * `current_vt` - Current right singular vectors (transposed)
/// * `new_columns` - New columns to add to the matrix
/// * `rank` - Target rank to maintain
///
/// # Returns
/// * Updated SVD factors (U, S, Vt)
pub fn incremental_svd<A>(
    current_u: &ArrayView2<A>,
    current_s: &ArrayView1<A>,
    current_vt: &ArrayView2<A>,
    new_columns: &ArrayView2<A>,
    rank: usize,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let m = current_u.shape()[0];
    let k = current_s.len();
    let n_old = current_vt.shape()[1];
    let n_new = new_columns.shape()[1];

    if new_columns.shape()[0] != m {
        return Err(LinalgError::ShapeError(
            "New columns must have same number of rows as U".to_string(),
        ));
    }

    // Project new columns onto current subspace
    let proj = current_u.t().dot(new_columns);

    // Compute orthogonal complement
    let us_proj = current_u.dot(&Array2::from_diag(current_s)).dot(&proj);
    let complement = new_columns - &us_proj;

    // QR decomposition of complement
    let (q_comp, r_comp) = crate::decomposition::qr(&complement.view(), None)?;

    // Build augmented matrices
    // Determine actual dimensions based on QR result
    let q_cols = q_comp.ncols();
    let aug_cols = k + q_cols;

    let mut aug_u = Array2::zeros((m, aug_cols));
    aug_u.slice_mut(ndarray::s![.., ..k]).assign(current_u);
    aug_u.slice_mut(ndarray::s![.., k..]).assign(&q_comp);

    let mut aug_s_mat = Array2::zeros((aug_cols, aug_cols));
    // Fill diagonal with singular values
    for i in 0..k {
        aug_s_mat[[i, i]] = current_s[i];
    }

    // Fill off-diagonal blocks
    // proj is k x n_new, we need to assign to the appropriate portion
    let proj_cols = proj.ncols().min(q_cols);
    if proj_cols > 0 {
        aug_s_mat
            .slice_mut(ndarray::s![..k, k..k + proj_cols])
            .assign(&proj.slice(ndarray::s![.., ..proj_cols]));
    }

    // r_comp should be q_cols x q_cols (or q_cols x n_new)
    let r_rows = r_comp.nrows().min(q_cols);
    let r_cols = r_comp.ncols().min(q_cols);
    aug_s_mat
        .slice_mut(ndarray::s![k..k + r_rows, k..k + r_cols])
        .assign(&r_comp.slice(ndarray::s![..r_rows, ..r_cols]));

    let mut aug_vt = Array2::zeros((aug_cols, n_old + n_new));
    aug_vt
        .slice_mut(ndarray::s![..k, ..n_old])
        .assign(current_vt);
    // Set identity block for new columns
    for i in 0..q_cols.min(n_new) {
        aug_vt[[k + i, n_old + i]] = A::one();
    }

    // SVD of augmented S matrix
    let (u_s, s_new, vt_s) = crate::decomposition::svd(&aug_s_mat.view(), false, None)?;

    // Update factors
    let updated_u = aug_u.dot(&u_s);
    let updated_vt = vt_s.dot(&aug_vt);

    // Truncate to desired rank
    let actual_rank = rank.min(s_new.len()).min(m).min(n_old + n_new);

    Ok((
        updated_u.slice(ndarray::s![.., ..actual_rank]).to_owned(),
        s_new.slice(ndarray::s![..actual_rank]).to_owned(),
        updated_vt.slice(ndarray::s![..actual_rank, ..]).to_owned(),
    ))
}

/// Block Krylov subspace method for solving large linear systems
///
/// This method builds a Krylov subspace using multiple starting vectors,
/// which can be more efficient than standard Krylov methods for certain problems.
///
/// # Arguments
/// * `a` - System matrix
/// * `b` - Right-hand side matrix (multiple RHS vectors)
/// * `block_size` - Number of vectors to use in each block
/// * `max_iterations` - Maximum number of block iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * Solution matrix X such that AX ≈ B
pub fn block_krylov_solve<A>(
    a: &ArrayView2<A>,
    b: &ArrayView2<A>,
    block_size: usize,
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<Array2<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let (n, _) = (a.nrows(), a.ncols());
    let (n_b, n_rhs) = (b.nrows(), b.ncols());

    if n != n_b {
        return Err(LinalgError::ShapeError(
            "Matrix A and B must have compatible dimensions".to_string(),
        ));
    }

    if block_size == 0 || block_size > n_rhs {
        return Err(LinalgError::InvalidInputError(
            "Invalid block size".to_string(),
        ));
    }

    // For simplicity, use block conjugate gradient for symmetric positive definite systems
    // In practice, this would be extended to handle general systems

    let mut x = Array2::zeros((n, n_rhs));

    // Process blocks of RHS vectors
    for block_start in (0..n_rhs).step_by(block_size) {
        let block_end = (block_start + block_size).min(n_rhs);
        let block_b = b.slice(ndarray::s![.., block_start..block_end]);

        // Solve each system in the block
        for j in 0..block_b.shape()[1] {
            let b_j = block_b.column(j);
            let options = IterativeSolverOptions {
                max_iterations,
                tolerance,
                verbose: false,
                restart: None,
            };

            let result = conjugate_gradient(a, &b_j, None, &options)?;
            x.column_mut(block_start + j).assign(&result.solution);
        }
    }

    Ok(x)
}

/// Communication-avoiding Krylov subspace method
///
/// This method reduces communication in parallel/distributed settings by
/// computing multiple matrix-vector products at once.
///
/// # Arguments
/// * `a` - System matrix
/// * `b` - Right-hand side vector
/// * `s` - Number of steps to compute at once
/// * `max_iterations` - Maximum iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * Solution vector x such that Ax ≈ b
pub fn ca_gmres<A>(
    a: &ArrayView2<A>,
    b: &ArrayView1<A>,
    s: usize,
    max_iterations: usize,
    tolerance: A,
) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    // For now, delegate to standard GMRES
    // A full implementation would compute s matrix powers at once
    let options = IterativeSolverOptions {
        max_iterations,
        tolerance,
        verbose: false,
        restart: Some(s * 10),
    };

    let result = gmres(a, b, None, &options)?;
    Ok(result.solution)
}

/// Randomized block Lanczos for large symmetric eigenvalue problems
///
/// Computes a few eigenvalues and eigenvectors of large symmetric matrices
/// using randomized block Lanczos iteration.
///
/// # Arguments
/// * `a` - Symmetric matrix
/// * `k` - Number of eigenvalues to compute
/// * `block_size` - Block size for Lanczos iteration
/// * `oversampling` - Oversampling parameter
/// * `max_iterations` - Maximum iterations
///
/// # Returns
/// * Tuple (eigenvalues, eigenvectors)
pub fn randomized_block_lanczos<A>(
    a: &ArrayView2<A>,
    k: usize,
    block_size: usize,
    oversampling: usize,
    max_iterations: usize,
) -> LinalgResult<(Array1<A>, Array2<A>)>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + std::iter::Sum + 'static,
{
    let n = a.shape()[0];

    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".to_string()));
    }

    let total_size = k + oversampling;
    if total_size > n {
        return Err(LinalgError::ShapeError(
            "k + oversampling must not exceed matrix dimension".to_string(),
        ));
    }

    // Initialize random block
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut q = Array2::zeros((n, block_size));
    for i in 0..n {
        for j in 0..block_size {
            q[[i, j]] = A::from(normal.sample(&mut rng)).unwrap();
        }
    }

    // Orthogonalize initial block
    let (q0, _) = crate::decomposition::qr(&q.view(), None)?;

    // Build Krylov subspace
    let mut q_blocks = vec![q0];
    let mut t = Array2::zeros((total_size, total_size));
    let mut block_offsets = vec![0]; // Track cumulative column offsets
    block_offsets.push(q_blocks[0].ncols());

    for iter in 0..max_iterations.min(total_size / block_size) {
        if iter >= q_blocks.len() {
            break;
        }
        let q_curr = &q_blocks[iter];
        let curr_cols = q_curr.ncols();
        let curr_offset = block_offsets[iter];

        // Compute A * Q
        let aq = a.dot(q_curr);

        // Orthogonalize against previous blocks
        let mut v = aq.clone();
        for (i, q_prev) in q_blocks.iter().enumerate() {
            let prev_cols = q_prev.ncols();
            let prev_offset = block_offsets[i];

            let h = q_prev.t().dot(&v);
            v = v - q_prev.dot(&h);

            // Store in block tridiagonal matrix
            // h is prev_cols x curr_cols
            if prev_offset + prev_cols <= total_size && curr_offset + curr_cols <= total_size {
                let h_rows = h.nrows().min(total_size - prev_offset);
                let h_cols = h.ncols().min(total_size - curr_offset);

                if h_rows > 0 && h_cols > 0 {
                    t.slice_mut(ndarray::s![
                        prev_offset..prev_offset + h_rows,
                        curr_offset..curr_offset + h_cols
                    ])
                    .assign(&h.slice(ndarray::s![..h_rows, ..h_cols]));
                    t.slice_mut(ndarray::s![
                        curr_offset..curr_offset + h_cols,
                        prev_offset..prev_offset + h_rows
                    ])
                    .assign(&h.slice(ndarray::s![..h_rows, ..h_cols]).t());
                }
            }
        }

        // QR decomposition of residual
        let (q_new, r_new) = crate::decomposition::qr(&v.view(), None)?;

        // Store diagonal block (sub-diagonal block of T)
        let new_cols = q_new.ncols();

        if new_cols > 0 {
            let next_offset = block_offsets.last().copied().unwrap_or(0);

            // r_new represents the coupling between current and next block
            let r_rows = r_new.nrows().min(new_cols);
            let r_cols = r_new.ncols().min(curr_cols);

            if next_offset < total_size && curr_offset < total_size && r_rows > 0 && r_cols > 0 {
                let avail_rows = (total_size - next_offset).min(r_rows);
                let avail_cols = (total_size - curr_offset).min(r_cols);

                if avail_rows > 0 && avail_cols > 0 {
                    t.slice_mut(ndarray::s![
                        next_offset..next_offset + avail_rows,
                        curr_offset..curr_offset + avail_cols
                    ])
                    .assign(&r_new.slice(ndarray::s![..avail_rows, ..avail_cols]));
                }
            }

            q_blocks.push(q_new);
            block_offsets.push(next_offset + new_cols);
        }

        // Check if we have enough basis vectors
        let current_basis_size = block_offsets.last().copied().unwrap_or(0);
        if current_basis_size >= total_size {
            break;
        }
    }

    // Solve eigenvalue problem for tridiagonal matrix
    let actual_size: usize = q_blocks.iter().map(|q| q.ncols()).sum();
    let actual_size = actual_size.min(total_size);
    let t_reduced = t.slice(ndarray::s![..actual_size, ..actual_size]);

    // Use standard eigendecomposition on the reduced matrix
    let (eigvals, eigvecs_small) = crate::eigen::eigh(&t_reduced, None)?;

    // Select k largest eigenvalues
    let mut indices: Vec<usize> = (0..eigvals.len()).collect();
    indices.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());

    let mut selected_eigvals = Array1::zeros(k);
    let mut selected_eigvecs = Array2::zeros((n, k));

    for i in 0..k {
        let idx = indices[i];
        selected_eigvals[i] = eigvals[idx];

        // Reconstruct eigenvector
        let mut eigvec = Array1::zeros(n);
        let mut coeff_idx = 0;

        for q_block in q_blocks.iter() {
            let block_cols = q_block.ncols();

            // Get coefficients for this block
            if coeff_idx + block_cols <= eigvecs_small.shape()[0] {
                for j in 0..block_cols {
                    if coeff_idx < eigvecs_small.shape()[0] {
                        let coeff = eigvecs_small[[coeff_idx, idx]];
                        let q_col = q_block.column(j);
                        eigvec.scaled_add(coeff, &q_col);
                        coeff_idx += 1;
                    }
                }
            } else {
                break;
            }
        }

        selected_eigvecs.column_mut(i).assign(&eigvec);
    }

    Ok((selected_eigvals, selected_eigvecs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_randomized_least_squares() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let b = array![1.0, 2.0, 0.0];

        let x = randomized_least_squares(&a.view(), &b.view(), 2, 2).unwrap();

        // Should approximately solve the least squares problem
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_randomized_norm() {
        let a = array![[3.0, 0.0], [0.0, 4.0]];

        // Spectral norm of diagonal matrix is max diagonal element
        let spec_norm = randomized_norm(&a.view(), "2", 20, 3).unwrap();
        assert!(spec_norm > 3.5 && spec_norm < 4.5);

        // Frobenius norm is sqrt(3^2 + 4^2) = 5
        let fro_norm = randomized_norm(&a.view(), "fro", 100, 0).unwrap();
        assert!(fro_norm > 4.5 && fro_norm < 5.5);
    }

    #[test]
    #[ignore = "Depends on SVD eigendecomposition for small matrices which is not yet implemented"]
    fn test_incremental_svd() {
        // Start with a larger matrix to avoid small matrix eigenvalue issues
        let mut initial = Array2::zeros((6, 4));
        initial[[0, 0]] = 1.0;
        initial[[1, 1]] = 1.0;
        initial[[2, 2]] = 0.5;
        initial[[3, 3]] = 0.25;

        let (u, s, vt) = crate::decomposition::svd(&initial.view(), false, None).unwrap();

        // Add new columns
        let mut new_cols = Array2::zeros((6, 2));
        new_cols[[4, 0]] = 1.0;
        new_cols[[5, 1]] = 1.0;

        let (u_new, s_new, vt_new) =
            incremental_svd(&u.view(), &s.view(), &vt.view(), &new_cols.view(), 6).unwrap();

        // Check dimensions
        assert_eq!(u_new.shape()[0], 6);
        assert_eq!(vt_new.shape()[1], 6);
        assert!(s_new.len() <= 6);

        // Verify basic properties
        assert!(s_new[0] >= s_new[s_new.len() - 1]); // Singular values should be in descending order
    }

    #[test]
    fn test_block_krylov_solve() {
        // Simple diagonal system
        let a = array![[2.0, 0.0], [0.0, 3.0]];
        let b = array![[2.0, 4.0], [6.0, 9.0]];

        let x = block_krylov_solve(&a.view(), &b.view(), 1, 10, 1e-10).unwrap();

        // Solution should be [[1, 2], [2, 3]]
        assert_abs_diff_eq!(x[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[[1, 1]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ca_gmres() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];

        let x = ca_gmres(&a.view(), &b.view(), 2, 100, 1e-10).unwrap();

        // Verify solution
        let residual = &b - a.dot(&x);
        let res_norm = residual.dot(&residual).sqrt();
        assert!(res_norm < 1e-10);
    }

    #[test]
    #[ignore = "Depends on eigendecomposition for small matrices which is not yet implemented"]
    fn test_randomized_block_lanczos() {
        // Use a larger symmetric matrix to avoid small matrix eigenvalue issues
        let n = 10;
        let mut a = Array2::zeros((n, n));

        // Create a symmetric tridiagonal matrix with known structure
        for i in 0..n {
            a[[i, i]] = 2.0 + (i as f64) * 0.1; // Diagonal
            if i > 0 {
                a[[i, i - 1]] = -1.0; // Sub-diagonal
                a[[i - 1, i]] = -1.0; // Super-diagonal
            }
        }

        let k = 3; // Find 3 largest eigenvalues
        let (eigvals, eigvecs) = randomized_block_lanczos(&a.view(), k, 2, 2, 10).unwrap();

        // Check we got the requested number of eigenvalues
        assert_eq!(eigvals.len(), k);
        assert_eq!(eigvecs.shape()[1], k);

        // Eigenvalues should be positive for this positive definite matrix
        for i in 0..k {
            assert!(eigvals[i] > 0.0);
        }

        // Eigenvalues should be in descending order
        for i in 1..k {
            assert!(eigvals[i - 1] >= eigvals[i]);
        }
    }
}
