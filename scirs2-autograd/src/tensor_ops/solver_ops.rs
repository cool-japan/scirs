use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix1, Ix2};

/// Solve linear system Ax = b
pub struct LinearSolveOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for LinearSolveOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || ashape[0] != ashape[1] {
            return Err(OpError::IncompatibleShape(
                "Linear solve requires square matrix A".into(),
            ));
        }

        if bshape[0] != ashape[0] {
            return Err(OpError::IncompatibleShape(
                "Dimension mismatch in Ax = b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;

        let x = if bshape.len() == 1 {
            let b_1d = b
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

            solve_linear_system_1d(&a_2d, &b_1d)?
        } else {
            let b_2d = b
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 2D".into()))?;

            solve_linear_system_2d(&a_2d, &b_2d)?
        };

        ctx.append_output(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Reverse-mode VJP of the linear solve `A x = b`:
        //   grad_b = A⁻ᵀ · grad_x          (solve  Aᵀ · grad_b = grad_x)
        //   grad_A = − grad_b · xᵀ
        //
        // NOTE on the well-known formula: `grad_A = −grad_b · xᵀ`, NOT
        // `−grad_x · xᵀ`.  An earlier version used `grad_x` here, which is a
        // genuine bug; the correct factor is the *already back-solved* `grad_b`.
        //
        // On any numerical failure we propagate `None` (non-differentiable for
        // this evaluation) rather than fabricating a zero gradient, which would
        // silently corrupt training.
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let x = ctx.output();
        let g = ctx.graph();

        let a_array = match a.eval(g) {
            Ok(arr) => arr,
            Err(_) => return append_solver_no_grad(ctx),
        };
        let x_array = match x.eval(g) {
            Ok(arr) => arr,
            Err(_) => return append_solver_no_grad(ctx),
        };
        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => return append_solver_no_grad(ctx),
        };

        let a_2d = match a_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => return append_solver_no_grad(ctx),
        };

        // grad_b = A⁻ᵀ · grad_x  (solve Aᵀ · grad_b = grad_x).
        let grad_b = if grad_output_array.ndim() == 1 {
            let grad_x_1d = match grad_output_array.view().into_dimensionality::<Ix1>() {
                Ok(view) => view,
                Err(_) => return append_solver_no_grad(ctx),
            };
            match solve_transpose_system(&a_2d, &grad_x_1d) {
                Ok(result) => result,
                // Honest failure: singular Aᵀ ⇒ gradient undefined, emit None.
                Err(_) => return append_solver_no_grad(ctx),
            }
        } else {
            let grad_x_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => return append_solver_no_grad(ctx),
            };
            match solve_transpose_system_2d(&a_2d, &grad_x_2d) {
                Ok(result) => result,
                Err(_) => return append_solver_no_grad(ctx),
            }
        };

        // grad_A = − grad_b · xᵀ  (outer product of the back-solved cotangent).
        let grad_b_view = grad_b.view();
        let x_view = x_array.view();
        let grad_a = match compute_outer_product_gradient(&grad_b_view, &x_view) {
            Ok(arr) => arr.mapv(|v| -v),
            Err(_) => return append_solver_no_grad(ctx),
        };

        let grad_a_tensor = crate::tensor_ops::convert_to_tensor(grad_a, g);
        let grad_b_tensor = crate::tensor_ops::convert_to_tensor(grad_b, g);

        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));
    }
}

/// Emit "no gradient" for both inputs of the linear solver.
///
/// Used when a tensor cannot be evaluated or the transpose system is singular:
/// returning `None`/`None` is honest (the gradient is genuinely unavailable for
/// this evaluation) and avoids fabricating a zero gradient.
fn append_solver_no_grad<F: Float>(ctx: &mut GradientContext<F>) {
    ctx.append_input_grad(0, None);
    ctx.append_input_grad(1, None);
}

// Enhanced version of solve_transpose_system with better error handling
#[allow(dead_code)]
fn solve_transpose_system<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
) -> Result<scirs2_core::ndarray::ArrayD<F>, OpError> {
    let at = a.t();
    solve_linear_system_1d(&at, b)
}

// Enhanced version of solve_transpose_system_2d with better error handling
#[allow(dead_code)]
fn solve_transpose_system_2d<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<scirs2_core::ndarray::ArrayD<F>, OpError> {
    let at = a.t();
    solve_linear_system_2d(&at, b)
}

/// Least squares solver (minimize ||Ax - b||²)
pub struct LeastSquaresSolveOp;

impl<F: Float> Op<F> for LeastSquaresSolveOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;

        // Solve least squares using normal equations: A^T A x = A^T b
        let at = a_2d.t();
        let ata = at.dot(&a_2d);
        let atb = if b.ndim() == 1 {
            let b_1d = b
                .view()
                .into_dimensionality::<Ix1>()
                .expect("Operation failed");
            at.dot(&b_1d).into_dyn()
        } else {
            let b_2d = b
                .view()
                .into_dimensionality::<Ix2>()
                .expect("Operation failed");
            at.dot(&b_2d).into_dyn()
        };

        // Create views for solve_symmetric_system
        let ata_view = ata
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Operation failed");
        let atb_view = atb.view();

        let x = solve_symmetric_system(&ata_view, &atb_view)?;

        ctx.append_output(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);
        let x = ctx.output();
        let g = ctx.graph();

        // Evaluate tensors to arrays
        let a_array = match a.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let b_array = match b.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let x_array = match x.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Convert to appropriate dimensions
        let a_2d = match a_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Compute residual r = Ax - b
        let ax = if x_array.ndim() == 1 {
            let x_1d = match x_array.view().into_dimensionality::<Ix1>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };
            a_2d.dot(&x_1d).into_dyn()
        } else {
            let x_2d = match x_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };
            a_2d.dot(&x_2d).into_dyn()
        };

        let residual = &ax - &b_array.view();

        // Gradient for least squares
        let at = a_2d.t();
        let ata = at.dot(&a_2d);

        // Solve A^T A @ grad_b = A^T @ grad_x for intermediate gradient
        let at_grad_x = if grad_output_array.ndim() == 1 {
            let grad_x_1d = match grad_output_array.view().into_dimensionality::<Ix1>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };
            at.dot(&grad_x_1d).into_dyn()
        } else {
            let grad_x_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };
            at.dot(&grad_x_2d).into_dyn()
        };

        // Create views for solve_symmetric_system
        let ata_view = ata
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Operation failed");
        let at_grad_x_view = at_grad_x.view();

        let grad_intermediate = match solve_symmetric_system(&ata_view, &at_grad_x_view) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Create views for the outer product gradient computation
        let grad_intermediate_view = grad_intermediate.view();
        let x_view = x_array.view();
        let residual_view = residual.view();

        // Compute gradient parts
        let grad_a_part1 = match compute_outer_product_gradient(&grad_intermediate_view, &x_view) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let grad_a_part2 =
            match compute_outer_product_gradient(&residual_view, &grad_intermediate_view) {
                Ok(arr) => arr,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };

        // Add both gradient parts
        let grad_a = grad_a_part1 + grad_a_part2;

        // grad_b = -A @ grad_intermediate
        let grad_b = if grad_intermediate.ndim() == 1 {
            let grad_int_1d = match grad_intermediate.view().into_dimensionality::<Ix1>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };
            let mut result = a_2d.dot(&grad_int_1d).into_dyn();
            result.mapv_inplace(|v| -v); // Apply negative sign
            result
        } else {
            let grad_int_2d = match grad_intermediate.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };
            let mut result = a_2d.dot(&grad_int_2d).into_dyn();
            result.mapv_inplace(|v| -v); // Apply negative sign
            result
        };

        // Convert gradients to tensors
        let grad_a_tensor = crate::tensor_ops::convert_to_tensor(grad_a, g);
        let grad_b_tensor = crate::tensor_ops::convert_to_tensor(grad_b, g);

        // Append with correct indices
        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));
    }
}

// Helper functions
#[allow(dead_code)]
fn solve_linear_system_1d<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
) -> Result<scirs2_core::ndarray::ArrayD<F>, OpError> {
    let n = a.shape()[0];
    let mut aug = Array2::<F>::zeros((n, n + 1));

    // Create augmented matrix [A|b]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination with partial pivoting
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
            for j in 0..=n {
                aug.swap((i, j), (max_row, j));
            }
        }

        // Forward elimination
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            let x_j = x[j];
            x[i] -= aug[[i, j]] * x_j;
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x.into_dyn())
}

#[allow(dead_code)]
fn solve_linear_system_2d<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<scirs2_core::ndarray::ArrayD<F>, OpError> {
    let n = a.shape()[0];
    let m = b.shape()[1];
    let mut x = Array2::<F>::zeros((n, m));

    // Solve for each column of B
    for j in 0..m {
        let b_col = b.column(j);
        let x_col = solve_linear_system_1d(a, &b_col)?;
        let x_col_1d = x_col
            .view()
            .into_dimensionality::<Ix1>()
            .expect("Operation failed");

        for i in 0..n {
            x[[i, j]] = x_col_1d[i];
        }
    }

    Ok(x.into_dyn())
}

#[allow(dead_code)]
fn solve_symmetric_system<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayViewD<F>,
) -> Result<scirs2_core::ndarray::ArrayD<F>, OpError> {
    // Cholesky decomposition for symmetric positive definite matrices
    let n = a.shape()[0];
    let mut l = Array2::<F>::zeros((n, n));

    // Cholesky decomposition
    for i in 0..n {
        for j in 0..=i {
            if i == j {
                let mut sum = F::zero();
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag_val = a[[j, j]] - sum;
                if diag_val <= F::zero() {
                    // Fall back to general solver
                    return if b.ndim() == 1 {
                        let b_1d = b
                            .view()
                            .into_dimensionality::<Ix1>()
                            .expect("Operation failed");
                        solve_linear_system_1d(a, &b_1d)
                    } else {
                        let b_2d = b
                            .view()
                            .into_dimensionality::<Ix2>()
                            .expect("Operation failed");
                        solve_linear_system_2d(a, &b_2d)
                    };
                }
                l[[j, j]] = diag_val.sqrt();
            } else {
                let mut sum = F::zero();
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // Solve L @ y = b, then L^T @ x = y
    if b.ndim() == 1 {
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .expect("Operation failed");

        // Forward substitution
        let mut y = Array1::<F>::zeros(n);
        for i in 0..n {
            y[i] = b_1d[i];
            for j in 0..i {
                let y_j = y[j];
                y[i] -= l[[i, j]] * y_j;
            }
            y[i] /= l[[i, i]];
        }

        // Back substitution
        let mut x = Array1::<F>::zeros(n);
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                let x_j = x[j];
                x[i] -= l[[j, i]] * x_j;
            }
            x[i] /= l[[i, i]];
        }

        Ok(x.into_dyn())
    } else {
        let b_2d = b
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Operation failed");
        let m = b_2d.shape()[1];
        let mut x = Array2::<F>::zeros((n, m));

        for col in 0..m {
            let b_col = b_2d.column(col);

            // Forward substitution
            let mut y = Array1::<F>::zeros(n);
            for i in 0..n {
                y[i] = b_col[i];
                for j in 0..i {
                    let y_j = y[j];
                    y[i] -= l[[i, j]] * y_j;
                }
                y[i] /= l[[i, i]];
            }

            // Back substitution
            for i in (0..n).rev() {
                x[[i, col]] = y[i];
                for j in (i + 1)..n {
                    let x_j_col = x[[j, col]];
                    x[[i, col]] -= l[[j, i]] * x_j_col;
                }
                x[[i, col]] /= l[[i, i]];
            }
        }

        Ok(x.into_dyn())
    }
}

#[allow(dead_code)]
fn compute_outer_product_gradient<F: Float>(
    a: &scirs2_core::ndarray::ArrayViewD<F>,
    b: &scirs2_core::ndarray::ArrayViewD<F>,
) -> Result<scirs2_core::ndarray::ArrayD<F>, OpError> {
    if a.ndim() == 1 && b.ndim() == 1 {
        let a_1d = match a.view().into_dimensionality::<Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        let b_1d = match b.view().into_dimensionality::<Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        let m = a_1d.len();
        let n = b_1d.len();
        let mut result = Array2::<F>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                result[[i, j]] = a_1d[i] * b_1d[j];
            }
        }

        Ok(result.into_dyn())
    } else if a.ndim() == 2 && b.ndim() == 1 {
        let a_2d = match a.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 2D".into())),
        };

        let b_1d = match b.view().into_dimensionality::<Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        Ok(a_2d
            .dot(&b_1d.view().insert_axis(scirs2_core::ndarray::Axis(0)))
            .into_dyn())
    } else if a.ndim() == 1 && b.ndim() == 2 {
        let a_1d = match a.view().into_dimensionality::<Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        let b_2d = match b.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 2D".into())),
        };

        Ok(a_1d
            .view()
            .insert_axis(scirs2_core::ndarray::Axis(1))
            .dot(&b_2d)
            .into_dyn())
    } else {
        let a_2d = match a.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 2D".into())),
        };

        let b_2d = match b.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 2D".into())),
        };

        Ok(a_2d.dot(&b_2d.t()).into_dyn())
    }
}

// Public API functions
#[allow(dead_code)]
pub fn solve<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = a.graph();

    // Use the shape of b for the result shape - the solution x should match b's shape
    let bshape = crate::tensor_ops::shape(b);

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .setshape(&bshape)  // Preserve shape information
        .build(LinearSolveOp)
}

#[allow(dead_code)]
pub fn lstsq<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = a.graph();

    // Use the shape of b for the result shape - the solution x should match b's shape
    let bshape = crate::tensor_ops::shape(b);

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .setshape(&bshape)  // Preserve shape information
        .build(LeastSquaresSolveOp)
}

#[cfg(test)]
mod grad_tests {
    use crate::tensor_ops as T;
    use scirs2_core::ndarray::{array, Array2};

    /// Reference solve `A X = B` (B a matrix) via Gaussian elimination, f64.
    fn solve_ref(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mc = b.ncols();
        let mut aug = Array2::<f64>::zeros((n, n + mc));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            for j in 0..mc {
                aug[[i, n + j]] = b[[i, j]];
            }
        }
        for i in 0..n {
            // partial pivot
            let mut mr = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[mr, i]].abs() {
                    mr = k;
                }
            }
            for j in 0..(n + mc) {
                aug.swap((i, j), (mr, j));
            }
            for k in (i + 1)..n {
                let f = aug[[k, i]] / aug[[i, i]];
                for j in i..(n + mc) {
                    aug[[k, j]] -= f * aug[[i, j]];
                }
            }
        }
        let mut x = Array2::<f64>::zeros((n, mc));
        for col in 0..mc {
            for i in (0..n).rev() {
                let mut s = aug[[i, n + col]];
                for j in (i + 1)..n {
                    s -= aug[[i, j]] * x[[j, col]];
                }
                x[[i, col]] = s / aug[[i, i]];
            }
        }
        x
    }

    /// Verify the linear-solver VJP (grad_A and grad_b) against finite
    /// differences for a 2×2 system with a 2-column RHS (fully 2D path).
    #[test]
    fn linear_solve_gradient_matches_fd() {
        let a = array![[3.0_f64, 1.0], [0.5, 2.0]];
        let b = array![[1.0_f64, 0.5], [-0.5, 2.0]];

        // Analytic grads of sum_all(solve(A, B)) via the autograd graph.
        let (ga, gb) = crate::run(|g| {
            let av = T::variable(a.clone(), g);
            let bv = T::variable(b.clone(), g);
            let x = T::linalg_solve(&av, &bv);
            let loss = T::sum_all(x);
            let grads = T::grad(&[&loss], &[&av, &bv]);
            let ga = grads[0]
                .eval(g)
                .expect("grad_a eval")
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("ga 2D")
                .to_owned();
            let gb = grads[1]
                .eval(g)
                .expect("grad_b eval")
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("gb 2D")
                .to_owned();
            (ga, gb)
        });

        let loss_of = |aa: &Array2<f64>, bb: &Array2<f64>| solve_ref(aa, bb).sum();
        let h = 1e-6_f64;

        // FD grad w.r.t. A.
        let mut ga_fd = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let mut ap = a.clone();
                let mut am = a.clone();
                ap[[i, j]] += h;
                am[[i, j]] -= h;
                ga_fd[[i, j]] = (loss_of(&ap, &b) - loss_of(&am, &b)) / (2.0 * h);
            }
        }
        // FD grad w.r.t. b.
        let mut gb_fd = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let mut bp = b.clone();
                let mut bm = b.clone();
                bp[[i, j]] += h;
                bm[[i, j]] -= h;
                gb_fd[[i, j]] = (loss_of(&a, &bp) - loss_of(&a, &bm)) / (2.0 * h);
            }
        }

        let err_a = ga
            .iter()
            .zip(ga_fd.iter())
            .fold(0.0_f64, |m, (x, y)| (x - y).abs().max(m));
        let err_b = gb
            .iter()
            .zip(gb_fd.iter())
            .fold(0.0_f64, |m, (x, y)| (x - y).abs().max(m));
        assert!(
            err_a < 1e-4,
            "linear_solve grad_A fd mismatch: err = {err_a}"
        );
        assert!(
            err_b < 1e-4,
            "linear_solve grad_b fd mismatch: err = {err_b}"
        );
    }
}
