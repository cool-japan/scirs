use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::matrix_calculus::{self, MatrixFnKind, MatrixFnVjpOp};
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix2};
use scirs2_core::numeric::FromPrimitive;

/// Matrix inverse operation
pub struct MatrixInverseOp;

impl<F: Float> Op<F> for MatrixInverseOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Get input as ndarray
        let input_array = input.view();
        let shape = input_array.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix inverse requires square matrix".into(),
            ));
        }

        let n = shape[0];

        let input_2d = input_array
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute inverse using Gauss-Jordan elimination
        let inv = compute_inverse(&input_2d)?;

        // No need to reshape, just use the computed inverse directly
        // but make a deep copy of it to ensure we have a clean array
        let output_inv = inv.to_owned();

        // Verify shape before output
        assert_eq!(output_inv.shape(), &[n, n]);

        // Append the array as output
        ctx.append_output(output_inv.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let output = ctx.output(); // This is the inverse
        let g = ctx.graph();

        // Evaluate tensors
        let output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Gradient of matrix inverse: -A^{-T} @ grad_output @ A^{-T}
        let inv = match output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_out_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let inv_t = inv.t();
        let temp = inv_t.dot(&grad_out_2d);
        let grad_input = -temp.dot(&inv_t);

        // Convert gradient to tensor
        let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// Matrix pseudo-inverse (Moore-Penrose) operation
pub struct PseudoInverseOp;

impl<F: Float> Op<F> for PseudoInverseOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute pseudo-inverse using SVD
        let pinv = compute_pseudo_inverse(&input_2d)?;

        ctx.append_output(pinv.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Golub-Pereyra derivative of the Moore-Penrose pseudo-inverse.
        let a = *ctx.input(0);
        let p = *ctx.output();
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(a, false)
            .append_input(p, false)
            .append_input(gy, false)
            .build(PseudoInverseVjpOp);
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Backward node of [`PseudoInverseOp`].
///
/// Inputs are `(A, P, gy)` with `P = A⁺` (taken from the forward output rather than
/// recomputed) and `gy` the cotangent of `P`.
///
/// Differentiating the Moore-Penrose conditions (Golub & Pereyra, 1973) gives
///
/// ```text
///   dA⁺ = -A⁺ dA A⁺ + A⁺ A⁺ᵀ dAᵀ (I - A A⁺) + (I - A⁺ A) dAᵀ A⁺ᵀ A⁺
/// ```
///
/// and taking the adjoint of each term against `gy` yields
///
/// ```text
///   Ā = -Pᵀ gy Pᵀ + (I_m - A P) gyᵀ P Pᵀ + Pᵀ P gyᵀ (I_n - P A)
/// ```
///
/// The last two terms are the *range* and *null-space* corrections; they vanish when `A`
/// is square and invertible, where the rule correctly collapses to the matrix-inverse VJP
/// `-A⁻ᵀ gy A⁻ᵀ`.
///
/// The previous implementation used
/// `PᵀP gy Aᵀ Pᵀ` and `Pᵀ Aᵀ gy Pᵀᵀ Pᵀ` for those two corrections. Neither is the adjoint
/// of anything in the expression above, and neither vanishes for an invertible `A`: on a
/// 3x3 symmetric positive definite matrix the reported gradient was
/// `+0.0344` where the true value is `-0.0473`.
pub struct PseudoInverseVjpOp;

impl<F: Float> Op<F> for PseudoInverseVjpOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_in = ctx.input(0);
        let p_in = ctx.input(1);
        let gy_in = ctx.input(2);

        let a = a_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("pseudo-inverse backward: A is not 2-D".into())
        })?;
        let p = p_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("pseudo-inverse backward: A+ is not 2-D".into())
        })?;
        let gy = gy_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("pseudo-inverse backward: cotangent is not 2-D".into())
        })?;

        let m = a.nrows();
        let n = a.ncols();
        if p.nrows() != n || p.ncols() != m {
            return Err(OpError::IncompatibleShape(
                "pseudo-inverse backward: A+ does not have the transposed shape of A".into(),
            ));
        }
        if gy.shape() != p.shape() {
            return Err(OpError::IncompatibleShape(
                "pseudo-inverse backward: cotangent shape does not match A+".into(),
            ));
        }

        let pt = p.t();
        let gyt = gy.t();

        let term1 = pt.dot(&gy).dot(&pt).mapv(|v| -v);

        // (I_m - A P) gyᵀ P Pᵀ
        let range_projector = Array2::<F>::eye(m) - a.dot(&p);
        let term2 = range_projector.dot(&gyt).dot(&p).dot(&pt);

        // Pᵀ P gyᵀ (I_n - P A)
        let null_projector = Array2::<F>::eye(n) - p.dot(&a);
        let term3 = pt.dot(&p).dot(&gyt).dot(&null_projector);

        let grad = term1 + term2 + term3;
        ctx.append_output(grad.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        matrix_calculus::append_unsupported_grad(
            ctx,
            "pseudo-inverse: second-order differentiation is not implemented".into(),
        );
    }
}

/// Matrix determinant for larger matrices
pub struct GeneralDeterminantOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for GeneralDeterminantOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Get input as ndarray
        let input_view = input.view();
        let shape = input_view.shape().to_vec(); // Clone the shape

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Determinant requires square matrix".into(),
            ));
        }

        let input_2d = input_view
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let det = compute_determinant_lu(&input_2d)?;

        // Create a scalar (0-dimensional) array with the determinant value
        // Use explicit arr0 to ensure we get a 0-dimensional array
        let det_array = scirs2_core::ndarray::arr0(det);

        // Verify the shape to make sure we're creating a scalar
        assert_eq!(det_array.ndim(), 0);

        // Output the determinant as a 0-dimensional array
        ctx.append_output(det_array.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // d det(A) / dA = det(A) * A^{-T}, so the VJP for an upstream scalar cotangent
        // `gy` is `gy * det(A) * A^{-T}`.
        //
        // Built lazily out of graph ops.  The previous implementation evaluated `gy`,
        // `det(A)` and `A` during graph construction and baked a constant tensor into the
        // graph: that collapses the tape (no second derivative), needs every placeholder
        // to already be fed, and indexed the 0-d determinant output with `[[0]]`, which
        // panics with "Attempted to index with [0] in array with 0 axes".
        let input = ctx.input(0);
        let output = ctx.output();
        let gy = ctx.output_grad();

        let inv_a = crate::tensor_ops::matrix_inverse(input);
        let inv_a_t = crate::tensor_ops::transpose(inv_a, &[1, 0]);
        let scaled = crate::tensor_ops::mul(gy, output);
        ctx.append_input_grad(0, Some(crate::tensor_ops::mul(scaled, inv_a_t)));
    }
}

// Helper functions
#[allow(dead_code)]
fn compute_inverse<F: Float>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
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
                    a[[k, j]] = a[[k, j]] - factor * a[[i, j]];
                    inv[[k, j]] = inv[[k, j]] - factor * inv[[i, j]];
                }
            }
        }
    }

    Ok(inv)
}

#[allow(dead_code)]
fn compute_pseudo_inverse<F: Float>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    // Simplified pseudo-inverse using transpose
    // For a full implementation, use SVD
    let m = matrix.shape()[0];
    let n = matrix.shape()[1];

    if m >= n {
        // A^+ = (A^T A)^(-1) A^T
        let at = matrix.t();
        let ata = at.dot(matrix);
        let ata_inv = compute_inverse(&ata.view())?;
        Ok(ata_inv.dot(&at))
    } else {
        // A^+ = A^T (A A^T)^(-1)
        let at = matrix.t();
        let aat = matrix.dot(&at);
        let aat_inv = compute_inverse(&aat.view())?;
        Ok(at.dot(&aat_inv))
    }
}

#[allow(dead_code)]
fn compute_determinant_lu<F: Float>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<F, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let mut det = F::one();
    let mut swaps = 0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[[i, k]].abs() > a[[max_row, k]].abs() {
                max_row = i;
            }
        }

        if a[[max_row, k]].abs() < F::epsilon() {
            return Ok(F::zero()); // Singular _matrix
        }

        // Swap rows
        if max_row != k {
            for j in k..n {
                a.swap((k, j), (max_row, j));
            }
            swaps += 1;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = a[[i, k]] / a[[k, k]];
            for j in (k + 1)..n {
                a[[i, j]] = a[[i, j]] - factor * a[[k, j]];
            }
        }

        det *= a[[k, k]];
    }

    // Account for row swaps
    if swaps % 2 == 1 {
        det = -det;
    }

    Ok(det)
}

/// Matrix exponential using Padé approximation (method 2)
pub struct MatrixExp2Op;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixExp2Op {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix exponential requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Use improved Padé approximation
        let result = compute_matrix_exp_pade(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_matrix_exp_grad(ctx);
    }
}

/// Matrix exponential using eigendecomposition (method 3)
pub struct MatrixExp3Op;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixExp3Op {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix exponential requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Use eigendecomposition method
        let result = compute_matrix_exp_eigen(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_matrix_exp_grad(ctx);
    }
}

/// Backward rule shared by [`MatrixExp2Op`] and [`MatrixExp3Op`].
///
/// Both compute `exp(A)` (by different forward algorithms), so both have the same VJP:
/// the adjoint of the Fréchet derivative of `exp` at `A` applied to the output cotangent,
/// evaluated as the top-right block of `exp([[Aᵀ, gy], [0, Aᵀ]])`.
///
/// The node is built lazily instead of evaluated here. The previous code called
/// `eval` on the input, the output *and* the cotangent, threw all three results away, and
/// returned the cotangent unchanged — so `d expm(A) / dA` was the identity.
fn append_matrix_exp_grad<F: Float>(ctx: &mut GradientContext<F>) {
    let a = *ctx.input(0);
    let gy = *ctx.output_grad();
    let g = ctx.graph();
    let gx = Tensor::builder(g)
        .append_input(a, false)
        .append_input(gy, false)
        .build(MatrixFnVjpOp {
            kind: MatrixFnKind::Exp,
        });
    ctx.append_input_grad(0, Some(gx));
}

/// Compute matrix exponential using Padé approximation
#[allow(dead_code)]
fn compute_matrix_exp_pade<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Compute norm of matrix
    let mut norm = F::zero();
    for i in 0..n {
        let mut row_sum = F::zero();
        for j in 0..n {
            row_sum += matrix[[i, j]].abs();
        }
        if row_sum > norm {
            norm = row_sum;
        }
    }

    // Scaling parameter.  The order-6 Padé approximant is accurate to `f64` rounding
    // only once the scaled matrix has norm <= 1/2, so scale to that bound rather than to
    // the norm <= 1 the previous `ceil(log2(norm))` produced.
    let two = F::from(2.0).expect("Failed to convert constant to float");
    let half = F::from(0.5).expect("Failed to convert constant to float");
    let s = if norm > half {
        ((norm / half).ln() / two.ln()).ceil()
    } else {
        F::zero()
    };

    let scale = F::from(2.0)
        .expect("Failed to convert constant to float")
        .powf(s);
    let scaled_matrix = matrix.mapv(|x| x / scale);

    // Coefficients of the diagonal Padé approximant of order (6, 6):
    //   b_j = (12 - j)! 6! / ( 12! j! (6 - j)! )
    // i.e. 1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280.
    //
    // The previous constants (1/12, 1/120, 1/3360, 1/30240, 1/1209600) are not the
    // order-6 Padé coefficients, and the odd part was built as `A * (b1*A + ...)`, which
    // shifts every term one power too high and makes the "odd" polynomial even. Both are
    // fixed here; `matrix_calculus::expm` is used as the cross-check in the unit test.
    let b0 = F::from(1.0).expect("Failed to convert constant to float");
    let b1 = F::from(0.5).expect("Failed to convert constant to float");
    let b2 = F::from(44.0)
        .expect("Failed to convert constant to float")
        .recip()
        * F::from(5.0).expect("Failed to convert constant to float");
    let b3 = F::from(66.0)
        .expect("Failed to convert constant to float")
        .recip();
    let b4 = F::from(792.0)
        .expect("Failed to convert constant to float")
        .recip();
    let b5 = F::from(15840.0)
        .expect("Failed to convert constant to float")
        .recip();
    let b6 = F::from(665280.0)
        .expect("Failed to convert constant to float")
        .recip();

    // Compute powers of matrix
    let i = Array2::<F>::eye(n);
    let a2 = scaled_matrix.dot(&scaled_matrix);
    let a4 = a2.dot(&a2);
    let a6 = a4.dot(&a2);

    // Odd part  U = A (b1 I + b3 A^2 + b5 A^4)
    // Even part V = b0 I + b2 A^2 + b4 A^4 + b6 A^6
    let u = &i * b1 + &a2 * b3 + &a4 * b5;
    let u = scaled_matrix.dot(&u);

    let v = &i * b0 + &a2 * b2 + &a4 * b4 + &a6 * b6;

    // Solve (V - U) * R = (V + U)
    let v_minus_u = &v - &u;
    let v_plus_u = &v + &u;

    // Use Gaussian elimination to solve
    let mut result = solve_matrix_equation(&v_minus_u.view(), &v_plus_u.view())?;

    // Square the result s times
    for _ in 0..s.to_usize().unwrap_or(0) {
        result = result.dot(&result);
    }

    Ok(result)
}

/// Compute matrix exponential using eigendecomposition
#[allow(dead_code)]
fn compute_matrix_exp_eigen<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Check if matrix is symmetric
    let is_symmetric = is_symmetric_matrix(matrix);

    if is_symmetric {
        // For symmetric matrices, use real eigendecomposition
        let (eigenvalues, eigenvectors) = compute_symmetric_eigen_simple(matrix)?;

        // exp(A) = V * diag(exp(λ)) * V^T
        let mut exp_eigenvalues = Array1::<F>::zeros(n);
        for i in 0..n {
            exp_eigenvalues[i] = eigenvalues[i].exp();
        }

        // Compute V * diag(exp(λ))
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                temp[[i, j]] = eigenvectors[[i, j]] * exp_eigenvalues[j];
            }
        }

        // Compute (V * diag(exp(λ))) * V^T
        let result = temp.dot(&eigenvectors.t());
        Ok(result)
    } else {
        // A non-symmetric matrix has no real orthogonal eigendecomposition, so fall back
        // to scaling-and-squaring.  The previous fallback was a bare 20-term Taylor
        // series with no argument reduction, which loses all accuracy once ||A|| grows
        // past a few units.
        matrix_calculus::expm(matrix)
    }
}

/// Solve matrix equation AX = B
#[allow(dead_code)]
fn solve_matrix_equation<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = a.shape()[0];
    let mut aug = Array2::<F>::zeros((n, 2 * n));

    // Create augmented matrix [A|B]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
            aug[[i, j + n]] = b[[i, j]];
        }
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if aug[[max_row, i]].abs() < F::epsilon() {
            return Err(OpError::IncompatibleShape("Matrix is singular".into()));
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                aug.swap((i, j), (max_row, j));
            }
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract solution
    let mut x = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(x)
}

/// Check if matrix is symmetric
#[allow(dead_code)]
fn is_symmetric_matrix<F: Float>(matrix: &scirs2_core::ndarray::ArrayView2<F>) -> bool {
    let n = matrix.shape()[0];
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs()
                > F::epsilon() * F::from(10.0).expect("Failed to convert constant to float")
            {
                return false;
            }
        }
    }
    true
}

/// Symmetric eigendecomposition, `(values descending, vectors as columns)`.
///
/// Cyclic Jacobi via [`crate::tensor_ops::matrix_calculus::symmetric_eigen`]. The previous
/// implementation solved the 2x2 case analytically and, for `n >= 3`, returned the
/// diagonal of the matrix as its "eigenvalues" together with the identity as its
/// "eigenvectors" — so `expm3` silently degenerated to `exp` of the diagonal for every
/// matrix bigger than 2x2.
#[allow(dead_code)]
fn compute_symmetric_eigen_simple<
    F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive,
>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    matrix_calculus::symmetric_eigen(matrix)
}

// Public API functions
#[allow(dead_code)]
pub fn matrix_inverse<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape tensor from the input
    let matrixshape = crate::tensor_ops::shape(matrix);

    // Build the tensor with shape information
    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixInverseOp)
}

#[allow(dead_code)]
pub fn pseudo_inverse<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape tensor from the input
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(PseudoInverseOp)
}

#[allow(dead_code)]
pub fn determinant<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();

    // For determinant, we're creating a scalar output (0-dimensional tensor)
    // We'll use zeros(0) to create a scalar tensor shape
    let scalarshape = crate::tensor_ops::zeros(&[0], g);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&scalarshape)
        .build(GeneralDeterminantOp)
}

/// Matrix exponential using improved Padé approximation (method 2)
#[allow(dead_code)]
pub fn expm2<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixExp2Op)
}

/// Matrix exponential using eigendecomposition (method 3)  
#[allow(dead_code)]
pub fn expm3<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixExp3Op)
}
