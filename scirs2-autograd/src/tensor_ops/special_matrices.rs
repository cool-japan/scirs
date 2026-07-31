use crate::op::*;
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::Float;
use scirs2_core::ndarray::Array2;
use scirs2_core::ndarray::ScalarOperand;
// BLAS dependencies removed - using core abstractions
// use ndarray_linalg::{Lapack, UPLO};

/// Cholesky decomposition operation with gradient support
#[derive(Clone)]
pub(crate) struct CholeskyOp;

impl<F: Float + ScalarOperand> Op<F> for CholeskyOp {
    fn name(&self) -> &'static str {
        "Cholesky"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::Other("Cholesky requires square matrix".into()));
        }

        // Get ndarray data directly
        let matrix = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("Failed to convert to 2D array".into()))?;

        // Pure-Rust Cholesky decomposition: A = L Lᵀ
        // Algorithm: for j in 0..n:
        //   L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)
        //   for i in j+1..n:
        //     L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]*L[j,k]) / L[j,j]
        let n = shape[0];
        let mut l = Array2::<F>::zeros((n, n));

        for j in 0..n {
            // Diagonal element
            let mut diag_sum = F::zero();
            for k in 0..j {
                diag_sum += l[[j, k]] * l[[j, k]];
            }
            let diag_val = matrix[[j, j]] - diag_sum;
            if diag_val <= F::zero() {
                return Err(OpError::Other("Matrix is not positive definite".into()));
            }
            l[[j, j]] = diag_val.sqrt();

            // Sub-diagonal elements in column j
            for i in (j + 1)..n {
                let mut off_sum = F::zero();
                for k in 0..j {
                    off_sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (matrix[[i, j]] - off_sum) / l[[j, j]];
            }
        }

        ctx.append_output(l.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // NOTE: this must NOT eagerly `.eval()` `y`/`gy` here (a previous version did).
        // `Op::grad` only ever has access to a bare `&Graph` (`ctx.graph()`), never the
        // `Context`/`VariableEnvironment` that resolves `Variable` nodes -- evaluating a
        // tensor that traces back to a `Variable` from here fails *honestly* rather than
        // fabricating a value (see `Graph::eval_tensors_in`'s `var_env` parameter). That
        // honest failure silently became a *shape* bug: the failed eval fell through to
        // `ctx.append_input_grad(0, None)`, and `tensor_ops::grad`'s "no gradient
        // accumulated" fallback then invented a zero gradient from `Tensor::shape()`'s
        // (empty, since it is never set for this node) hint -- a 0-d "gradient" in place
        // of an n×n matrix. Building a lazy `CholeskyBackwardOp` instead defers the
        // Murray-2016 computation to normal (non-eager) graph evaluation, exactly like
        // every other backward op in this crate.
        let y = *ctx.output();
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(y, false)
            .append_input(gy, false)
            .build(CholeskyBackwardOp);
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Iain Murray (2016) reverse-mode gradient for Cholesky decomposition, computed lazily
/// (as an ordinary graph node evaluated in topological order) rather than eagerly inside
/// `Op::grad` -- see the comment on `CholeskyOp::grad` for why eager evaluation there is
/// unsound.
///
/// Inputs are `(L, dL)`: `L = chol(A)` (the forward output) and `dL` (the upstream
/// cotangent). Output is `dA`.
struct CholeskyBackwardOp;

impl<F: Float + ScalarOperand> Op<F> for CholeskyBackwardOp {
    fn name(&self) -> &'static str {
        "CholeskyBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let l_view = ctx.input(0);
        let l = l_view
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("CholeskyBackward: L must be 2D".into()))?;
        let gy_view = ctx.input(1);
        let gy_2d = gy_view
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("CholeskyBackward: dL must be 2D".into()))?;

        let n = l.shape()[0];
        let two = F::from(2.0_f64)
            .ok_or_else(|| OpError::Other("CholeskyBackward: could not represent 2.0".into()))?;

        // Given L = chol(A) and upstream gradient dL = gy_2d:
        //   S     = L^T * dL                              (n×n)
        //   Phi(S) = tril(S, -1) + diag(diag(S)) / 2     (keep strict lower tri, halve diagonal)
        //   dA    = L^{-T} * Phi(S) * L^{-1}             (two triangular solves)
        //   dA    = (dA + dA^T) / 2                       (symmetrise)

        // Step 1: S = L^T * dL
        let mut s = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut acc = F::zero();
                // L^T[i,k] = L[k,i], non-zero only when k >= i (lower tri)
                for k in i..n {
                    acc += l[[k, i]] * gy_2d[[k, j]];
                }
                s[[i, j]] = acc;
            }
        }

        // Step 2: Phi(S) — zero strict upper triangle, halve diagonal
        let mut phi = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    phi[[i, j]] = s[[i, j]] / two;
                } else {
                    phi[[i, j]] = s[[i, j]];
                }
            }
        }

        // Step 3: M = L^{-T} * Phi(S) — solve L^T * M = Phi(S) column by column
        // L^T is upper triangular; back substitution: for each col of Phi
        let mut m = Array2::<F>::zeros((n, n));
        for col in 0..n {
            // Solve L^T x = phi[:,col] using back substitution
            let mut x = vec![F::zero(); n];
            for i in (0..n).rev() {
                let mut rhs = phi[[i, col]];
                for k in (i + 1)..n {
                    // L^T[i, k] = L[k, i]
                    rhs -= l[[k, i]] * x[k];
                }
                x[i] = rhs / l[[i, i]];
            }
            for i in 0..n {
                m[[i, col]] = x[i];
            }
        }

        // Step 4: dA_raw = M * L^{-1}
        // Equivalently, transposing each row: solve L^T * da[row,:]^T = m[row,:]^T via back-sub.
        // L^T is upper triangular, so back-substitute from bottom to top.
        let mut da = Array2::<F>::zeros((n, n));
        for row in 0..n {
            // Solve L^T x = m[row,:] using back substitution (L^T upper triangular)
            let mut x = vec![F::zero(); n];
            for i in (0..n).rev() {
                let mut rhs = m[[row, i]];
                for k in (i + 1)..n {
                    // L^T[i, k] = L[k, i]
                    rhs -= l[[k, i]] * x[k];
                }
                x[i] = rhs / l[[i, i]];
            }
            for j in 0..n {
                da[[row, j]] = x[j];
            }
        }

        // Step 5: symmetrise
        let mut grad = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                grad[[i, j]] = (da[[i, j]] + da[[j, i]]) / two;
            }
        }

        ctx.append_output(grad.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        crate::tensor_ops::matrix_calculus::append_unsupported_grad(
            ctx,
            "Cholesky: second-order differentiation is not implemented.".into(),
        );
    }
}

/// Symmetric matrix operation - makes a matrix symmetric by averaging with its transpose
#[derive(Clone)]
pub(crate) struct SymmetrizeOp;

impl<F: Float + ScalarOperand> Op<F> for SymmetrizeOp {
    fn name(&self) -> &'static str {
        "Symmetrize"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::Other("Symmetrize requires square matrix".into()));
        }

        // Get ndarray data directly
        let matrix = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("Failed to convert to 2D array".into()))?;

        // Symmetrize manually: (A + A^T) / 2
        let mut symmetric = Array2::<F>::zeros((shape[0], shape[1]));
        let half = F::from(0.5).expect("Failed to convert constant to float");

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                symmetric[[i, j]] = (matrix[[i, j]] + matrix[[j, i]]) * half;
            }
        }

        ctx.append_output(symmetric.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Symmetrize, `S(A) = (A + A^T) / 2`, is self-adjoint under the Frobenius inner
        // product (`<S(A), B> = <A, S(B)>`), so its VJP is itself: re-apply `SymmetrizeOp`
        // to the upstream cotangent. Building this as a lazy graph node (rather than
        // eagerly evaluating `gy` right here, as a previous version did) is required:
        // `Op::grad` only ever has a bare `&Graph`, not the `Context`/`VariableEnvironment`
        // needed to resolve a `Variable` upstream of `gy`, so an eager `.eval()` here can
        // fail even when the *lazy* gradient graph would evaluate just fine later.
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(gy, false)
            .build(SymmetrizeOp);
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Lower triangular extraction operation
#[derive(Clone)]
pub(crate) struct LowerTriangularOp {
    diagonal: i32, // k=0 for main diagonal, k<0 for below diagonal
}

impl<F: Float> Op<F> for LowerTriangularOp {
    fn name(&self) -> &'static str {
        "LowerTriangular"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::Other(
                "Lower triangular extraction requires 2D matrix".into(),
            ));
        }

        // Get ndarray data directly
        let matrix = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("Failed to convert to 2D array".into()))?;

        let mut lower = matrix.to_owned();
        let (rows, cols) = (lower.shape()[0], lower.shape()[1]);

        // Zero out elements above the specified diagonal
        for i in 0..rows {
            for j in 0..cols {
                if j as i32 > i as i32 - self.diagonal {
                    lower[[i, j]] = F::zero();
                }
            }
        }

        ctx.append_output(lower.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Extracting a triangle is a coordinate projection (each output entry either
        // copies or zeros the matching input entry), which is self-adjoint: its VJP is
        // itself. Re-apply the same masking op to the upstream cotangent, built as a lazy
        // graph node -- see `CholeskyOp::grad`'s comment for why this must not eagerly
        // `.eval()` `gy` here.
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(gy, false)
            .build(LowerTriangularOp {
                diagonal: self.diagonal,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Upper triangular extraction operation
#[derive(Clone)]
pub(crate) struct UpperTriangularOp {
    diagonal: i32, // k=0 for main diagonal, k>0 for above diagonal
}

impl<F: Float> Op<F> for UpperTriangularOp {
    fn name(&self) -> &'static str {
        "UpperTriangular"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::Other(
                "Upper triangular extraction requires 2D matrix".into(),
            ));
        }

        // Get ndarray data directly
        let matrix = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("Failed to convert to 2D array".into()))?;

        let mut upper = matrix.to_owned();
        let (rows, cols) = (upper.shape()[0], upper.shape()[1]);

        // Zero out elements below the specified diagonal
        for i in 0..rows {
            for j in 0..cols {
                if (j as i32) < (i as i32 + self.diagonal) {
                    upper[[i, j]] = F::zero();
                }
            }
        }

        ctx.append_output(upper.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Self-adjoint coordinate projection -- see `LowerTriangularOp::grad`.
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(gy, false)
            .build(UpperTriangularOp {
                diagonal: self.diagonal,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Band matrix extraction operation
#[derive(Clone)]
pub(crate) struct BandMatrixOp {
    lower: i32, // number of subdiagonals
    upper: i32, // number of superdiagonals
}

impl<F: Float> Op<F> for BandMatrixOp {
    fn name(&self) -> &'static str {
        "BandMatrix"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::Other(
                "Band matrix extraction requires 2D matrix".into(),
            ));
        }

        // Get ndarray data directly
        let matrix = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| OpError::Other("Failed to convert to 2D array".into()))?;

        let mut band = matrix.to_owned();
        let (rows, cols) = (band.shape()[0], band.shape()[1]);

        // Zero out elements outside the band
        for i in 0..rows {
            for j in 0..cols {
                let diag_offset = j as i32 - i as i32;
                if diag_offset < -self.lower || diag_offset > self.upper {
                    band[[i, j]] = F::zero();
                }
            }
        }

        ctx.append_output(band.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Self-adjoint coordinate projection -- see `LowerTriangularOp::grad`.
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(gy, false)
            .build(BandMatrixOp {
                lower: self.lower,
                upper: self.upper,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

// Public API functions

/// Compute Cholesky decomposition with gradient support
#[allow(dead_code)]
pub fn cholesky<'g, F: Float + ScalarOperand>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(CholeskyOp)
}

/// Make a matrix symmetric by averaging with its transpose
#[allow(dead_code)]
pub fn symmetrize<'g, F: Float + ScalarOperand>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(SymmetrizeOp)
}

/// Extract lower triangular part of a matrix
#[allow(dead_code)]
pub fn tril<'g, F: Float>(matrix: &Tensor<'g, F>, diagonal: i32) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape of the input tensor for setting the output shape
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)  // Preserve shape information
        .build(LowerTriangularOp { diagonal })
}

/// Extract upper triangular part of a matrix
#[allow(dead_code)]
pub fn triu<'g, F: Float>(matrix: &Tensor<'g, F>, diagonal: i32) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape of the input tensor for setting the output shape
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)  // Preserve shape information
        .build(UpperTriangularOp { diagonal })
}

/// Extract band from a matrix
#[allow(dead_code)]
pub fn band_matrix<'g, F: Float>(matrix: &Tensor<'g, F>, lower: i32, upper: i32) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape of the input tensor for setting the output shape
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)  // Preserve shape information
        .build(BandMatrixOp { lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    use scirs2_core::ndarray::array;

    /// Test basic 2×2 SPD matrix: A = [[4,2],[2,3]]
    /// Expected L = [[2,0],[1,sqrt(2)]]
    #[test]
    fn test_cholesky_2x2_spd() {
        crate::run(|g| {
            let a = convert_to_tensor(array![[4.0_f64, 2.0], [2.0, 3.0]], g);
            let l_tensor = cholesky(&a);
            let l = l_tensor.eval(g).expect("Cholesky eval failed");

            // L[0,0] = sqrt(4) = 2
            assert!(
                (l[[0, 0]] - 2.0_f64).abs() < 1e-10,
                "L[0,0] expected 2.0, got {}",
                l[[0, 0]]
            );
            // L[0,1] must be zero (lower triangular)
            assert!(
                l[[0, 1]].abs() < 1e-10,
                "L[0,1] expected 0.0, got {}",
                l[[0, 1]]
            );
            // L[1,0] = (A[1,0] - 0) / L[0,0] = 2/2 = 1
            assert!(
                (l[[1, 0]] - 1.0_f64).abs() < 1e-10,
                "L[1,0] expected 1.0, got {}",
                l[[1, 0]]
            );
            // L[1,1] = sqrt(3 - 1²) = sqrt(2)
            let expected_l11 = 2.0_f64.sqrt();
            assert!(
                (l[[1, 1]] - expected_l11).abs() < 1e-10,
                "L[1,1] expected {expected_l11}, got {}",
                l[[1, 1]]
            );
        });
    }

    /// Verify L @ Lᵀ ≈ A for a larger SPD matrix
    #[test]
    fn test_cholesky_reconstruction() {
        crate::run(|g| {
            // Build a 4×4 SPD matrix: A = Mᵀ M + 4I for random-ish M
            let raw = array![
                [10.0_f64, 2.0, 1.0, 0.5],
                [2.0, 8.0, 1.5, 0.3],
                [1.0, 1.5, 7.0, 0.8],
                [0.5, 0.3, 0.8, 6.0],
            ];
            let a = convert_to_tensor(raw.clone(), g);
            let l_tensor = cholesky(&a);
            let l = l_tensor.eval(g).expect("Cholesky eval failed");

            let l_2d = l
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("dim");
            let n = l_2d.shape()[0];

            // Compute L @ Lᵀ
            let mut reconstructed = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        reconstructed[[i, j]] += l_2d[[i, k]] * l_2d[[j, k]];
                    }
                }
            }

            // Verify element-wise match with original
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (reconstructed[[i, j]] - raw[[i, j]]).abs() < 1e-8,
                        "Mismatch at [{i},{j}]: reconstructed={}, original={}",
                        reconstructed[[i, j]],
                        raw[[i, j]]
                    );
                }
            }
        });
    }

    /// Non-SPD matrix must return an error
    #[test]
    fn test_cholesky_non_spd_returns_error() {
        crate::run(|g| {
            // Matrix with a negative diagonal — not positive definite
            let a = convert_to_tensor(array![[-1.0_f64, 0.0], [0.0, 1.0]], g);
            let l_tensor = cholesky(&a);
            let result = l_tensor.eval(g);
            assert!(result.is_err(), "Expected error for non-SPD matrix, got Ok");
        });
    }

    /// Verify the Iain Murray (2016) backward formula by evaluating it directly.
    ///
    /// For A = [[4,2],[2,5]], L = [[2,0],[1,2]], upstream gy = 2*L (from sum(L^2)):
    /// The expected gradient dA is [[1,0],[0,1]] (computed by hand).
    #[test]
    fn test_cholesky_gradient_murray_formula() {
        // A = [[4,2],[2,5]], L = [[2,0],[1,2]]
        // gy = dLoss/dL where Loss = sum(L^2), so gy = 2*L = [[4,0],[2,4]]
        //
        // Hand-calculated Murray (2016) result:
        //   S = L^T * gy = [[2,1],[0,2]] * [[4,0],[2,4]] = [[10,4],[4,8]]
        //   Phi(S) = [[5,0],[4,4]]  (half diagonal, zero strict upper)
        //   M = L^{-T} * Phi(S): solve L^T * M = Phi(S)
        //     col 0: [5,4] -> x[1]=2, x[0]=1.5 => M[:,0]=[1.5,2]
        //     col 1: [0,4] -> x[1]=2, x[0]=-1   => M[:,1]=[-1,2]
        //   da = M * L^{-1}: solve L^T * da[row] = m[row] (back-sub)
        //     row 0: [1.5,-1] -> x[1]=-0.5, x[0]=1
        //     row 1: [2,2]    -> x[1]=1,    x[0]=0.5
        //   dA = (da + da^T)/2 = [[1,0],[0,1]]
        let n = 2_usize;
        let l_data = [[2.0_f64, 0.0], [1.0, 2.0]];
        let gy_data = [[4.0_f64, 0.0], [2.0, 4.0]]; // 2 * L

        let two = 2.0_f64;

        // Step 1: S = L^T * gy
        let mut s = [[0.0_f64; 2]; 2];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in i..n {
                    acc += l_data[k][i] * gy_data[k][j];
                }
                s[i][j] = acc;
            }
        }

        // Step 2: Phi(S)
        let mut phi = [[0.0_f64; 2]; 2];
        for i in 0..n {
            for j in 0..=i {
                phi[i][j] = if i == j { s[i][j] / two } else { s[i][j] };
            }
        }

        // Step 3: M = L^{-T} * Phi(S) via back-sub
        let mut m = [[0.0_f64; 2]; 2];
        for col in 0..n {
            let mut x = [0.0_f64; 2];
            for i in (0..n).rev() {
                let mut rhs = phi[i][col];
                for k in (i + 1)..n {
                    rhs -= l_data[k][i] * x[k];
                }
                x[i] = rhs / l_data[i][i];
            }
            for i in 0..n {
                m[i][col] = x[i];
            }
        }

        // Step 4: da = M * L^{-1} via back-sub (solve L^T da[row] = m[row])
        let mut da = [[0.0_f64; 2]; 2];
        for row in 0..n {
            let mut x = [0.0_f64; 2];
            for i in (0..n).rev() {
                let mut rhs = m[row][i];
                for k in (i + 1)..n {
                    rhs -= l_data[k][i] * x[k];
                }
                x[i] = rhs / l_data[i][i];
            }
            da[row][..n].copy_from_slice(&x[..n]);
        }

        // Step 5: symmetrise
        let mut grad_a = [[0.0_f64; 2]; 2];
        for i in 0..n {
            for j in 0..n {
                grad_a[i][j] = (da[i][j] + da[j][i]) / two;
            }
        }

        // Expected: [[1,0],[0,1]]
        let expected = [[1.0_f64, 0.0], [0.0, 1.0]];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (grad_a[i][j] - expected[i][j]).abs() < 1e-10,
                    "Murray formula mismatch at [{i},{j}]: got {}, expected {}",
                    grad_a[i][j],
                    expected[i][j]
                );
            }
        }
    }
}
