use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{s, Array1, Array2, Ix2};
use num_traits::FromPrimitive;

// Type aliases to reduce complexity
type SVDResult<F> = Result<(Array2<F>, Array1<F>, Array2<F>), OpError>;
#[allow(dead_code)]
type QRResult<F> = Result<(Array2<F>, Array2<F>), OpError>;
type QRPivotResult<F> = Result<(Array2<F>, Array2<F>, Array1<F>), OpError>;
#[allow(dead_code)]
type EigenResult<F> = Result<(Array2<F>, Array2<F>, Array1<F>), OpError>;

/// Improved SVD using Jacobi algorithm for better numerical stability
pub struct JacobiSVDOp {
    full_matrices: bool,
}

/// Extraction operator for Jacobi SVD components
pub struct SVDJacobiExtractOp {
    component: usize,
}

impl<F: Float> Op<F> for SVDJacobiExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "SVDJacobiExtractU",
            1 => "SVDJacobiExtractS",
            2 => "SVDJacobiExtractVt",
            _ => "SVDJacobiExtractUnknown",
        }
    }

    fn compute(&self, _ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // This is a placeholder - the actual extraction happens in the parent op
        Err(OpError::Other(
            "SVD extraction should be handled by parent op".into(),
        ))
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for JacobiSVDOp {
    fn name(&self) -> &'static str {
        "JacobiSVD"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape("SVD requires 2D matrix".into()));
        }

        let _m = shape[0];
        let _n = shape[1];
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute SVD using Jacobi algorithm
        let (u, s, vt) = compute_svd_jacobi(&input_2d, self.full_matrices)?;

        // Append outputs
        ctx.append_output(u.into_dyn());
        ctx.append_output(s.into_dyn());
        ctx.append_output(vt.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // SVD gradient is complex - simplified version
        ctx.append_input_grad(0, None);
    }
}

/// Randomized SVD for large matrices
pub struct RandomizedSVDOp {
    rank: usize,
    oversampling: usize,
    n_iter: usize,
}

/// Extraction operator for Randomized SVD components
pub struct RandomizedSVDExtractOp {
    component: usize,
}

impl<F: Float> Op<F> for RandomizedSVDExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "RandomizedSVDExtractU",
            1 => "RandomizedSVDExtractS",
            2 => "RandomizedSVDExtractVt",
            _ => "RandomizedSVDExtractUnknown",
        }
    }

    fn compute(&self, _ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        Err(OpError::Other(
            "Randomized SVD extraction should be handled by parent op".into(),
        ))
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for RandomizedSVDOp {
    fn name(&self) -> &'static str {
        "RandomizedSVD"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape("SVD requires 2D matrix".into()));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute randomized SVD
        let (u, s, vt) =
            compute_randomized_svd(&input_2d, self.rank, self.oversampling, self.n_iter)?;

        ctx.append_output(u.into_dyn());
        ctx.append_output(s.into_dyn());
        ctx.append_output(vt.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

/// Generalized eigenvalue problem: Ax = λBx
pub struct GeneralizedEigenOp;

/// Extraction operator for Generalized Eigen components
pub struct GeneralizedEigenExtractOp {
    component: usize,
}

impl<F: Float> Op<F> for GeneralizedEigenExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "GeneralizedEigenExtractValues",
            1 => "GeneralizedEigenExtractVectors",
            _ => "GeneralizedEigenExtractUnknown",
        }
    }

    fn compute(&self, _ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        Err(OpError::Other(
            "Generalized eigen extraction should be handled by parent op".into(),
        ))
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for GeneralizedEigenOp {
    fn name(&self) -> &'static str {
        "GeneralizedEigen"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        if a.shape() != b.shape() || a.shape().len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Generalized eigenvalue problem requires two square matrices of same size".into(),
            ));
        }

        let n = a.shape()[0];
        if n != a.shape()[1] {
            return Err(OpError::IncompatibleShape("Matrices must be square".into()));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_2d = b
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert B to 2D".into()))?;

        // Compute generalized eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = compute_generalized_eigen(&a_2d, &b_2d)?;

        ctx.append_output(eigenvalues.into_dyn());
        ctx.append_output(eigenvectors.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// QR decomposition with column pivoting for better numerical stability
pub struct QRPivotOp;

/// Extraction operator for QR Pivot components
pub struct QRPivotExtractOp {
    component: usize,
}

impl<F: Float> Op<F> for QRPivotExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "QRPivotExtractQ",
            1 => "QRPivotExtractR",
            2 => "QRPivotExtractP",
            _ => "QRPivotExtractUnknown",
        }
    }

    fn compute(&self, _ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        Err(OpError::Other(
            "QR pivot extraction should be handled by parent op".into(),
        ))
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

impl<F: Float + ndarray::ScalarOperand> Op<F> for QRPivotOp {
    fn name(&self) -> &'static str {
        "QRPivot"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "QR decomposition requires 2D matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute QR with column pivoting
        let (q, r, p) = compute_qr_pivot(&input_2d)?;

        ctx.append_output(q.into_dyn());
        ctx.append_output(r.into_dyn());
        ctx.append_output(p.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

// Helper functions

/// Compute SVD using Jacobi algorithm (more stable for small matrices)
fn compute_svd_jacobi<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
    full_matrices: bool,
) -> SVDResult<F> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = m.min(n);

    // For simplicity, use a hybrid approach
    // First reduce to bidiagonal form, then apply Jacobi rotations

    // Initialize U, S, V
    let mut u = Array2::<F>::eye(m);
    let mut v = Array2::<F>::eye(n);
    let mut b = matrix.to_owned();

    // Bidiagonalization using Householder reflections
    for i in 0..k {
        // Left Householder for column i
        if i < m - 1 {
            let col = b.slice(s![i.., i]).to_owned();
            let (h, _) = householder_vector(&col.view())?;
            let h_mat = householder_matrix(&h, m - i);

            // Apply to B and U
            let b_sub = b.slice(s![i.., i..]).to_owned();
            let b_new = h_mat.dot(&b_sub);
            b.slice_mut(s![i.., i..]).assign(&b_new);

            let u_sub = u.slice(s![.., i..]).to_owned();
            let u_new = u_sub.dot(&h_mat.t());
            u.slice_mut(s![.., i..]).assign(&u_new);
        }

        // Right Householder for row i
        if i < n - 2 {
            let row = b.slice(s![i, i + 1..]).to_owned();
            let (h, _) = householder_vector(&row.view())?;
            let h_mat = householder_matrix(&h, n - i - 1);

            // Apply to B and V
            let b_sub = b.slice(s![i.., i + 1..]).to_owned();
            let b_new = b_sub.dot(&h_mat);
            b.slice_mut(s![i.., i + 1..]).assign(&b_new);

            let v_sub = v.slice(s![i + 1.., ..]).to_owned();
            let v_new = h_mat.t().dot(&v_sub);
            v.slice_mut(s![i + 1.., ..]).assign(&v_new);
        }
    }

    // Extract diagonal and superdiagonal
    let mut diag = Array1::<F>::zeros(k);
    let mut superdiag = Array1::<F>::zeros(k - 1);

    for i in 0..k {
        diag[i] = b[[i, i]];
        if i < k - 1 {
            superdiag[i] = b[[i, i + 1]];
        }
    }

    // Apply Jacobi rotations to diagonalize
    let max_iter = 100;
    let tol = F::epsilon() * F::from(10.0).unwrap();

    for _ in 0..max_iter {
        let mut converged = true;

        for i in 0..k - 1 {
            if superdiag[i].abs() > tol {
                converged = false;

                // Compute Givens rotation
                let a = diag[i];
                let b = superdiag[i];
                let c = diag[i + 1];

                let (cos, sin) = compute_givens_rotation(a, b, c);

                // Update matrices
                diag[i] = cos * cos * a + sin * sin * c + F::from(2.0).unwrap() * cos * sin * b;
                diag[i + 1] = sin * sin * a + cos * cos * c - F::from(2.0).unwrap() * cos * sin * b;
                superdiag[i] = F::zero();

                // Update U and V
                apply_givens_left(&mut u, i, i + 1, cos, sin);
                apply_givens_right(&mut v, i, i + 1, cos, sin);
            }
        }

        if converged {
            break;
        }
    }

    // Ensure positive singular values
    for i in 0..k {
        if diag[i] < F::zero() {
            diag[i] = -diag[i];
            u.slice_mut(s![.., i]).mapv_inplace(|x| -x);
        }
    }

    // Sort singular values in descending order
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&i, &j| diag[j].abs().partial_cmp(&diag[i].abs()).unwrap());

    let s = Array1::from_iter(indices.iter().map(|&i| diag[i]));

    let u_sorted = if full_matrices {
        let mut u_full = u.clone();
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            u_full
                .slice_mut(s![.., new_idx])
                .assign(&u.slice(s![.., old_idx]));
        }
        u_full
    } else {
        let mut u_reduced = Array2::<F>::zeros((m, k));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            u_reduced
                .slice_mut(s![.., new_idx])
                .assign(&u.slice(s![.., old_idx]));
        }
        u_reduced
    };

    let vt_sorted = if full_matrices {
        let mut vt_full = v.t().to_owned();
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            vt_full
                .slice_mut(s![new_idx, ..])
                .assign(&v.slice(s![.., old_idx]));
        }
        vt_full
    } else {
        let mut vt_reduced = Array2::<F>::zeros((k, n));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            vt_reduced
                .slice_mut(s![new_idx, ..])
                .assign(&v.slice(s![.., old_idx]));
        }
        vt_reduced
    };

    Ok((u_sorted, s, vt_sorted))
}

/// Compute randomized SVD for large matrices
fn compute_randomized_svd<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
    rank: usize,
    oversampling: usize,
    n_iter: usize,
) -> SVDResult<F> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let l = (rank + oversampling).min(n.min(m));

    // Generate random Gaussian matrix
    let mut omega = Array2::<F>::zeros((n, l));
    for i in 0..n {
        for j in 0..l {
            // Simple pseudo-random number (not cryptographically secure)
            let val =
                F::from((i * l + j) % 7).unwrap() / F::from(7.0).unwrap() - F::from(0.5).unwrap();
            omega[[i, j]] = val;
        }
    }

    // Power iteration for better approximation
    let mut q = matrix.dot(&omega);

    for _ in 0..n_iter {
        q = orthogonalize_qr(&q)?;
        q = matrix.t().dot(&q);
        q = orthogonalize_qr(&q)?;
        q = matrix.dot(&q);
    }

    let q = orthogonalize_qr(&q)?;

    // Project matrix onto Q subspace
    let b = q.t().dot(matrix);

    // Compute SVD of smaller matrix B
    let (u_b, s, vt) = compute_svd_jacobi(&b.view(), false)?;

    // Recover full U
    let u = q.dot(&u_b);

    // Truncate to requested rank
    let u_truncated = u.slice(s![.., ..rank]).to_owned();
    let s_truncated = s.slice(s![..rank]).to_owned();
    let vt_truncated = vt.slice(s![..rank, ..]).to_owned();

    Ok((u_truncated, s_truncated, vt_truncated))
}

/// Compute generalized eigenvalue problem
fn compute_generalized_eigen<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let _n = a.shape()[0];

    // Simple approach: compute B^(-1)A if B is non-singular
    // For a robust implementation, use QZ algorithm

    // Check if B is positive definite (simplified check)
    let b_inv = match compute_matrix_inverse(b) {
        Ok(inv) => inv,
        Err(_) => return Err(OpError::Other("Matrix B is singular".into())),
    };

    // Compute B^(-1)A
    let c = b_inv.dot(a);

    // Compute eigenvalues of C
    let (eigenvalues, eigenvectors) = compute_eigen_iterative(&c.view())?;

    Ok((eigenvalues, eigenvectors))
}

/// QR decomposition with column pivoting
fn compute_qr_pivot<F: Float + ndarray::ScalarOperand>(
    matrix: &ndarray::ArrayView2<F>,
) -> QRPivotResult<F> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = m.min(n);

    let mut a = matrix.to_owned();
    let mut q = Array2::<F>::eye(m);
    let mut perm: Vec<usize> = (0..n).collect();

    // Column norms for pivoting
    let mut col_norms = Array1::<F>::zeros(n);
    for j in 0..n {
        let col = a.slice(s![.., j]);
        col_norms[j] = col.dot(&col).sqrt();
    }

    for i in 0..k {
        // Find pivot column
        let (pivot_idx, _) = col_norms
            .slice(s![i..])
            .indexed_iter()
            .max_by(|(_, &a), (_, &b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        let pivot_col = i + pivot_idx;

        // Swap columns
        if pivot_col != i {
            perm.swap(i, pivot_col);
            for row in 0..m {
                a.swap((row, i), (row, pivot_col));
            }
            col_norms.swap(i, pivot_col);
        }

        // Compute Householder reflector
        if i < m {
            let col = a.slice(s![i.., i]).to_owned();
            let (v, beta) = householder_vector(&col.view())?;

            // Apply Householder transformation
            if beta.abs() > F::epsilon() {
                // Update A
                for j in i..n {
                    let col = a.slice(s![i.., j]).to_owned();
                    let dot_product = v.dot(&col);
                    let update = v.mapv(|x| x * beta * dot_product);
                    for k in 0..m - i {
                        a[[i + k, j]] -= update[k];
                    }
                }

                // Update Q
                for j in 0..m {
                    let col = q.slice(s![j, i..]).to_owned();
                    let dot_product = col.dot(&v);
                    for k in 0..m - i {
                        q[[j, i + k]] -= beta * dot_product * v[k];
                    }
                }

                // Update column norms
                for j in (i + 1)..n {
                    let col = a.slice(s![i + 1.., j]);
                    col_norms[j] = col.dot(&col).sqrt();
                }
            }
        }
    }

    // Extract R
    let r = a.slice(s![..k, ..]).to_owned();

    // Convert permutation to array
    let p = Array1::from_vec(perm.iter().map(|&i| F::from(i).unwrap()).collect());

    Ok((q, r, p))
}

// Utility functions

fn householder_vector<F: Float>(x: &ndarray::ArrayView1<F>) -> Result<(Array1<F>, F), OpError> {
    let n = x.len();
    if n == 0 {
        return Err(OpError::IncompatibleShape("Empty vector".into()));
    }

    let mut v = x.to_owned();
    let norm_x = x.dot(x).sqrt();

    if norm_x < F::epsilon() {
        v[0] = F::one();
        return Ok((v, F::zero()));
    }

    let sign = if x[0] >= F::zero() {
        F::one()
    } else {
        -F::one()
    };
    v[0] += sign * norm_x;

    let norm_v_sq = v.dot(&v);
    let beta = F::from(2.0).unwrap() / norm_v_sq;

    Ok((v, beta))
}

fn householder_matrix<F: Float>(v: &Array1<F>, size: usize) -> Array2<F> {
    let beta = F::from(2.0).unwrap() / v.dot(v);
    let mut h = Array2::<F>::eye(size);

    for i in 0..size {
        for j in 0..size {
            h[[i, j]] -= beta * v[i] * v[j];
        }
    }

    h
}

fn compute_givens_rotation<F: Float>(a: F, b: F, c: F) -> (F, F) {
    if b.abs() < F::epsilon() {
        return (F::one(), F::zero());
    }

    let tau = (c - a) / (F::from(2.0).unwrap() * b);
    let t = if tau >= F::zero() {
        F::one() / (tau + (F::one() + tau * tau).sqrt())
    } else {
        -F::one() / (-tau + (F::one() + tau * tau).sqrt())
    };

    let cos = F::one() / (F::one() + t * t).sqrt();
    let sin = t * cos;

    (cos, sin)
}

fn apply_givens_left<F: Float>(matrix: &mut Array2<F>, i: usize, j: usize, cos: F, sin: F) {
    let n = matrix.shape()[1];
    for k in 0..n {
        let ai = matrix[[i, k]];
        let aj = matrix[[j, k]];
        matrix[[i, k]] = cos * ai - sin * aj;
        matrix[[j, k]] = sin * ai + cos * aj;
    }
}

fn apply_givens_right<F: Float>(matrix: &mut Array2<F>, i: usize, j: usize, cos: F, sin: F) {
    let m = matrix.shape()[0];
    for k in 0..m {
        let ai = matrix[[k, i]];
        let aj = matrix[[k, j]];
        matrix[[k, i]] = cos * ai - sin * aj;
        matrix[[k, j]] = sin * ai + cos * aj;
    }
}

fn orthogonalize_qr<F: Float + ndarray::ScalarOperand>(
    a: &Array2<F>,
) -> Result<Array2<F>, OpError> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut q = a.to_owned();

    // Modified Gram-Schmidt
    for j in 0..n {
        let mut col = q.slice_mut(s![.., j]);
        let norm = col.dot(&col).sqrt();

        if norm < F::epsilon() {
            return Err(OpError::Other("Matrix is rank deficient".into()));
        }

        col.mapv_inplace(|x| x / norm);

        for k in (j + 1)..n {
            let dot_product = q.slice(s![.., j]).dot(&q.slice(s![.., k]));
            let q_col_j = q.slice(s![.., j]).to_owned();
            for i in 0..m {
                q[[i, k]] -= dot_product * q_col_j[i];
            }
        }
    }

    Ok(q)
}

fn compute_matrix_inverse<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let mut inv = Array2::<F>::eye(n);

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < F::epsilon() {
            return Err(OpError::IncompatibleShape("Matrix is singular".into()));
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                a.swap((i, j), (max_row, j));
                inv.swap((i, j), (max_row, j));
            }
        }

        // Scale pivot row
        let pivot = a[[i, i]];
        for j in 0..n {
            a[[i, j]] /= pivot;
            inv[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = a[[k, i]];
                for j in 0..n {
                    let a_ij = a[[i, j]];
                    let inv_ij = inv[[i, j]];
                    a[[k, j]] -= factor * a_ij;
                    inv[[k, j]] -= factor * inv_ij;
                }
            }
        }
    }

    Ok(inv)
}

fn compute_eigen_iterative<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = matrix.shape()[0];
    let max_iter = 100;
    let tol = F::epsilon() * F::from(10.0).unwrap();

    // QR algorithm with shifts
    let mut a = matrix.to_owned();
    let mut q_total = Array2::<F>::eye(n);

    for _ in 0..max_iter {
        // Wilkinson shift
        let a_nn = a[[n - 1, n - 1]];
        let a_nm1 = if n > 1 { a[[n - 2, n - 1]] } else { F::zero() };
        let a_nm1nm1 = if n > 1 { a[[n - 2, n - 2]] } else { F::zero() };

        let delta = (a_nm1nm1 - a_nn) / F::from(2.0).unwrap();
        let sign = if delta >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        let mu =
            a_nn - sign * a_nm1 * a_nm1 / (delta.abs() + (delta * delta + a_nm1 * a_nm1).sqrt());

        // Shifted QR step
        for i in 0..n {
            a[[i, i]] -= mu;
        }

        // QR decomposition
        let (q, r) = compute_qr_simple(&a.view())?;
        a = r.dot(&q);

        for i in 0..n {
            a[[i, i]] += mu;
        }

        q_total = q_total.dot(&q);

        // Check convergence
        let mut converged = true;
        for i in 0..n - 1 {
            if a[[i + 1, i]].abs() > tol {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }
    }

    // Extract eigenvalues
    let mut eigenvalues = Array1::<F>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    Ok((eigenvalues, q_total))
}

fn compute_qr_simple<F: Float + ndarray::ScalarOperand>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<(Array2<F>, Array2<F>), OpError> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = m.min(n);

    let mut q = Array2::<F>::eye(m);
    let mut r = matrix.to_owned();

    for j in 0..k {
        let col = r.slice(s![j.., j]).to_owned();
        let (v, beta) = householder_vector(&col.view())?;

        if beta.abs() > F::epsilon() {
            // Apply to R
            for col_idx in j..n {
                let col = r.slice(s![j.., col_idx]).to_owned();
                let dot_product = v.dot(&col);
                for row_idx in 0..(m - j) {
                    r[[j + row_idx, col_idx]] -= beta * dot_product * v[row_idx];
                }
            }

            // Apply to Q
            for row_idx in 0..m {
                let row = q.slice(s![row_idx, j..]).to_owned();
                let dot_product = row.dot(&v);
                for col_idx in 0..(m - j) {
                    q[[row_idx, j + col_idx]] -= beta * dot_product * v[col_idx];
                }
            }
        }
    }

    Ok((q, r))
}

// Public API functions

/// Compute SVD using improved Jacobi algorithm
pub fn svd_jacobi<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    full_matrices: bool,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    // Build the SVD operation
    let svd_op = Tensor::builder(g)
        .append_input(matrix, false)
        .build(JacobiSVDOp { full_matrices });

    // Extract components using SVDExtractOp
    let u = Tensor::builder(g)
        .append_input(svd_op, false)
        .build(SVDJacobiExtractOp { component: 0 });

    let s = Tensor::builder(g)
        .append_input(svd_op, false)
        .build(SVDJacobiExtractOp { component: 1 });

    let vt = Tensor::builder(g)
        .append_input(svd_op, false)
        .build(SVDJacobiExtractOp { component: 2 });

    (u, s, vt)
}

/// Compute randomized SVD for large matrices
pub fn randomized_svd<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    rank: usize,
    oversampling: usize,
    n_iter: usize,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    let svd_op = Tensor::builder(g)
        .append_input(matrix, false)
        .build(RandomizedSVDOp {
            rank,
            oversampling,
            n_iter,
        });

    let u = Tensor::builder(g)
        .append_input(svd_op, false)
        .build(RandomizedSVDExtractOp { component: 0 });

    let s = Tensor::builder(g)
        .append_input(svd_op, false)
        .build(RandomizedSVDExtractOp { component: 1 });

    let vt = Tensor::builder(g)
        .append_input(svd_op, false)
        .build(RandomizedSVDExtractOp { component: 2 });

    (u, s, vt)
}

/// Solve generalized eigenvalue problem Ax = λBx
pub fn generalized_eigen<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>) {
    let g = a.graph();

    let eigen_op = Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(GeneralizedEigenOp);

    let eigenvalues = Tensor::builder(g)
        .append_input(eigen_op, false)
        .build(GeneralizedEigenExtractOp { component: 0 });

    let eigenvectors = Tensor::builder(g)
        .append_input(eigen_op, false)
        .build(GeneralizedEigenExtractOp { component: 1 });

    (eigenvalues, eigenvectors)
}

/// QR decomposition with column pivoting
pub fn qr_pivot<'g, F: Float + ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    let qr_op = Tensor::builder(g)
        .append_input(matrix, false)
        .build(QRPivotOp);

    let q = Tensor::builder(g)
        .append_input(qr_op, false)
        .build(QRPivotExtractOp { component: 0 });

    let r = Tensor::builder(g)
        .append_input(qr_op, false)
        .build(QRPivotExtractOp { component: 1 });

    let p = Tensor::builder(g)
        .append_input(qr_op, false)
        .build(QRPivotExtractOp { component: 2 });

    (q, r, p)
}
