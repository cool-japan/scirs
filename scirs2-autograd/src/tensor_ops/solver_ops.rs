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

        println!("Solving linear system: A({ashape:?}) * x = b({bshape:?})");

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

            println!("Solving 1D system (vector b)");
            let x_result = solve_linear_system_1d(&a_2d, &b_1d)?;
            println!("Solution x shape: {:?}", x_result.shape());
            x_result
        } else {
            let b_2d = b
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 2D".into()))?;

            println!("Solving 2D system (matrix b)");
            let x_result = solve_linear_system_2d(&a_2d, &b_2d)?;
            println!("Solution x shape: {:?}", x_result.shape());
            x_result
        };

        // Verify the solution has the expected shape
        println!("Final solution shape: {:?}", x.shape());

        ctx.append_output(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);
        let x = ctx.output();
        let g = ctx.graph();

        println!("Computing gradient for linear solver");

        // Gradient computation for linear solve
        // If Ax = b, then:
        // dL/dA = -dL/dx @ x^T
        // dL/db = A^{-T} @ dL/dx

        // Evaluate tensors to arrays
        let a_array = match a.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate matrix A");
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let b_array = match b.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate vector b");
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let x_array = match x.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate solution x");
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate gradient");
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        println!(
            "A shape: {:?}, b shape: {:?}, x shape: {:?}, grad shape: {:?}",
            a_array.shape(),
            b_array.shape(),
            x_array.shape(),
            grad_output_array.shape()
        );

        // Convert to appropriate dimensions
        let a_2d = match a_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                println!("Failed to convert A to 2D");
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // First calculate gradient with respect to b: grad_b = A^{-T} @ grad_x
        // This involves solving the transpose system: A^T @ grad_b = grad_x
        println!("Computing gradient with respect to b (solving A^T @ grad_b = grad_x)");

        let grad_b = if grad_output_array.ndim() == 1 {
            // For 1D grad_output (vector case)
            let grad_x_1d = match grad_output_array.view().into_dimensionality::<Ix1>() {
                Ok(view) => view,
                Err(_) => {
                    println!("Failed to convert gradient to 1D");
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };

            // Solve A^T @ grad_b = grad_x
            match solve_transpose_system(&a_2d, &grad_x_1d) {
                Ok(result) => result.into_dyn(),
                Err(e) => {
                    println!("Error solving transpose system (1D): {e:?}");
                    // Use regularized system instead
                    let n = a_2d.shape()[0];
                    let eps = F::epsilon() * F::from(10.0).unwrap();
                    let regularized = &a_2d + &(Array2::<F>::eye(n) * eps);

                    match solve_transpose_system(&regularized.view(), &grad_x_1d) {
                        Ok(result) => result.into_dyn(),
                        Err(e2) => {
                            println!("Error solving regularized system: {e2:?}");
                            // Return zero gradient as fallback
                            Array1::<F>::zeros(grad_x_1d.len()).into_dyn()
                        }
                    }
                }
            }
        } else {
            // For 2D grad_output (matrix case)
            let grad_x_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    println!("Failed to convert gradient to 2D");
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };

            // Solve A^T @ grad_b = grad_x for each column
            match solve_transpose_system_2d(&a_2d, &grad_x_2d) {
                Ok(result) => result.into_dyn(),
                Err(e) => {
                    println!("Error solving transpose system (2D): {e:?}");
                    // Use regularized system instead
                    let n = a_2d.shape()[0];
                    let eps = F::epsilon() * F::from(10.0).unwrap();
                    let regularized = &a_2d + &(Array2::<F>::eye(n) * eps);

                    match solve_transpose_system_2d(&regularized.view(), &grad_x_2d) {
                        Ok(result) => result.into_dyn(),
                        Err(e2) => {
                            println!("Error solving regularized system: {e2:?}");
                            // Return zero gradient as fallback
                            Array2::<F>::zeros(grad_x_2d.raw_dim()).into_dyn()
                        }
                    }
                }
            }
        };

        println!("Gradient for b computed, shape: {:?}", grad_b.shape());

        // Now calculate gradient with respect to A: grad_A = -grad_x @ x^T
        println!("Computing gradient with respect to A (outer product: -grad_x @ x^T)");

        // Create view arrays for compute_outer_product_gradient
        let grad_output_view = grad_output_array.view();
        let x_view = x_array.view();

        // Compute outer product: grad_x @ x^T
        let grad_a_result = compute_outer_product_gradient(&grad_output_view, &x_view);
        let grad_a = match grad_a_result {
            Ok(arr) => {
                // Apply negative sign
                arr.mapv(|v| -v)
            }
            Err(e) => {
                println!("Error computing outer product: {e:?}");
                // Return zero gradient as fallback
                Array2::<F>::zeros((a_2d.shape()[0], a_2d.shape()[1])).into_dyn()
            }
        };

        println!("Gradient for A computed, shape: {:?}", grad_a.shape());

        // Convert gradients to tensors
        let grad_a_tensor = crate::tensor_ops::convert_to_tensor(grad_a, g);
        let grad_b_tensor = crate::tensor_ops::convert_to_tensor(grad_b, g);

        // Append with correct indices
        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));

        println!("Linear solver gradient computation complete");
    }
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
            let b_1d = b.view().into_dimensionality::<Ix1>().unwrap();
            at.dot(&b_1d).into_dyn()
        } else {
            let b_2d = b.view().into_dimensionality::<Ix2>().unwrap();
            at.dot(&b_2d).into_dyn()
        };

        // Create views for solve_symmetric_system
        let ata_view = ata.view().into_dimensionality::<Ix2>().unwrap();
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
        let ata_view = ata.view().into_dimensionality::<Ix2>().unwrap();
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
        let x_col_1d = x_col.view().into_dimensionality::<Ix1>().unwrap();

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
                        let b_1d = b.view().into_dimensionality::<Ix1>().unwrap();
                        solve_linear_system_1d(a, &b_1d)
                    } else {
                        let b_2d = b.view().into_dimensionality::<Ix2>().unwrap();
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
        let b_1d = b.view().into_dimensionality::<Ix1>().unwrap();

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
        let b_2d = b.view().into_dimensionality::<Ix2>().unwrap();
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
