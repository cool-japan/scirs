use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::Float;
use ndarray::{Array2, ArrayView2, Ix2};

/// Solve Sylvester equation AX + XB = C
pub struct SylvesterSolveOp;

impl<F: Float + ndarray::ScalarOperand> Op<F> for SylvesterSolveOp {
    fn name(&self) -> &'static str {
        "SylvesterSolve"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);
        let c = ctx.input(2);

        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        // Check shapes
        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(OpError::IncompatibleShape(
                "Sylvester equation requires square matrix A".into(),
            ));
        }

        if b_shape.len() != 2 || b_shape[0] != b_shape[1] {
            return Err(OpError::IncompatibleShape(
                "Sylvester equation requires square matrix B".into(),
            ));
        }

        if c_shape.len() != 2 || c_shape[0] != a_shape[0] || c_shape[1] != b_shape[0] {
            return Err(OpError::IncompatibleShape(
                "Matrix C must have shape (m, n) where A is (m, m) and B is (n, n)".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;

        let b_2d = b
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert B to 2D".into()))?;

        let c_2d = c
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert C to 2D".into()))?;

        let x = solve_sylvester_internal(&a_2d, &b_2d, &c_2d)?;
        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);
        let x = ctx.output();
        let g = ctx.graph();

        // Evaluate tensors
        let a_array = match a.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        let b_array = match b.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        let x_array = match x.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        // Convert to 2D
        let _a_2d = match a_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        let _b_2d = match b_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        let x_2d = match x_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        let grad_x_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                ctx.append_input_grad(2, None);
                return;
            }
        };

        // Compute gradients
        // For Sylvester equation AX + XB = C:
        // grad_A = grad_X @ X^T
        // grad_B = X^T @ grad_X
        // grad_C = grad_X

        // grad_A = grad_X @ X^T
        let grad_a = grad_x_2d.dot(&x_2d.t());

        // grad_B = X^T @ grad_X
        let grad_b = x_2d.t().dot(&grad_x_2d);

        // grad_C = grad_X
        let grad_c = grad_x_2d.to_owned();

        // Convert to tensors
        let grad_a_tensor = tensor_ops::convert_to_tensor(grad_a.into_dyn(), g);
        let grad_b_tensor = tensor_ops::convert_to_tensor(grad_b.into_dyn(), g);
        let grad_c_tensor = tensor_ops::convert_to_tensor(grad_c.into_dyn(), g);

        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));
        ctx.append_input_grad(2, Some(grad_c_tensor));
    }
}

/// Solve Lyapunov equation AX + XA^T = Q
pub struct LyapunovSolveOp;

impl<F: Float + ndarray::ScalarOperand> Op<F> for LyapunovSolveOp {
    fn name(&self) -> &'static str {
        "LyapunovSolve"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let q = ctx.input(1);

        let a_shape = a.shape();
        let q_shape = q.shape();

        // Check shapes
        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(OpError::IncompatibleShape(
                "Lyapunov equation requires square matrix A".into(),
            ));
        }

        if q_shape.len() != 2 || q_shape[0] != q_shape[1] || q_shape[0] != a_shape[0] {
            return Err(OpError::IncompatibleShape(
                "Matrix Q must be square with same size as A".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;

        let q_2d = q
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert Q to 2D".into()))?;

        let x = solve_lyapunov_internal(&a_2d, &q_2d)?;
        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let x = ctx.output();
        let g = ctx.graph();

        // Evaluate tensors
        let a_array = match a.eval(g) {
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

        // Convert to 2D
        let _a_2d = match a_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let x_2d = match x_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        let grad_x_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Compute gradients
        // For Lyapunov equation AX + XA^T = Q:
        // grad_A = grad_X @ X^T + X @ grad_X^T
        // grad_Q = grad_X

        // grad_A = grad_X @ X^T + X @ grad_X^T
        let grad_a = grad_x_2d.dot(&x_2d.t()) + x_2d.dot(&grad_x_2d.t());

        // grad_Q = grad_X
        let grad_q = grad_x_2d.to_owned();

        // Convert to tensors
        let grad_a_tensor = tensor_ops::convert_to_tensor(grad_a.into_dyn(), g);
        let grad_q_tensor = tensor_ops::convert_to_tensor(grad_q.into_dyn(), g);

        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_q_tensor));
    }
}

/// Cholesky solve: solve AX = B where A is positive definite using Cholesky decomposition
pub struct CholeskySolveOp;

impl<F: Float> Op<F> for CholeskySolveOp {
    fn name(&self) -> &'static str {
        "CholeskySolve"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
            return Err(OpError::IncompatibleShape(
                "Cholesky solve requires square matrix A".into(),
            ));
        }

        if b_shape[0] != a_shape[0] {
            return Err(OpError::IncompatibleShape(
                "Dimension mismatch in AX = B".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;

        // Compute Cholesky decomposition of A
        let l = compute_cholesky(&a_2d)?;

        // Solve using forward and backward substitution
        let x = if b_shape.len() == 1 {
            let b_1d = b
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

            solve_cholesky_1d(&l.view(), &b_1d)?.into_dyn()
        } else {
            let b_2d = b
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 2D".into()))?;

            solve_cholesky_2d(&l.view(), &b_2d)?.into_dyn()
        };

        ctx.append_output(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let a = ctx.input(0);
        let _b = ctx.input(1);
        let x = ctx.output();
        let g = ctx.graph();

        // Evaluate tensors
        let a_array = match a.eval(g) {
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

        // Compute Cholesky decomposition
        let l = match compute_cholesky(&a_2d) {
            Ok(l) => l,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Solve for gradient wrt b: L L^T grad_b = grad_x
        let grad_b = if grad_output_array.ndim() == 1 {
            let grad_x_1d = match grad_output_array
                .view()
                .into_dimensionality::<ndarray::Ix1>()
            {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };

            solve_cholesky_1d(&l.view(), &grad_x_1d)
                .unwrap_or_else(|_| ndarray::Array1::zeros(grad_x_1d.len()))
                .into_dyn()
        } else {
            let grad_x_2d = match grad_output_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    ctx.append_input_grad(1, None);
                    return;
                }
            };

            solve_cholesky_2d(&l.view(), &grad_x_2d)
                .unwrap_or_else(|_| Array2::zeros(grad_x_2d.raw_dim()))
                .into_dyn()
        };

        // Compute gradient wrt A
        let grad_output_view = grad_output_array.view();
        let x_view = x_array.view();

        let grad_a_outer = match compute_outer_product_gradient(&grad_output_view, &x_view) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // grad_A = -0.5 * (grad_x @ x^T + x @ grad_x^T)
        let grad_a = match grad_a_outer.view().into_dimensionality::<Ix2>() {
            Ok(view) => {
                let mut result = view.to_owned();
                let transposed = view.t().to_owned();
                result = &result + &transposed;
                result.mapv(|x| x * F::from(-0.5).unwrap())
            }
            Err(_) => {
                ctx.append_input_grad(0, None);
                ctx.append_input_grad(1, None);
                return;
            }
        };

        // Convert gradients to tensors
        let grad_a_tensor = tensor_ops::convert_to_tensor(grad_a.into_dyn(), g);
        let grad_b_tensor = tensor_ops::convert_to_tensor(grad_b, g);

        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));
    }
}

// Helper functions

/// Solve Sylvester equation AX + XB = C using Bartels-Stewart algorithm
fn solve_sylvester_internal<F: Float + ndarray::ScalarOperand>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    c: &ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let m = a.shape()[0];
    let n = b.shape()[0];

    // For small matrices, use vectorization approach
    if m * n <= 100 {
        // Vectorize: vec(AX + XB) = (I ⊗ A + B^T ⊗ I) vec(X) = vec(C)
        // where ⊗ is the Kronecker product

        let mut kron_sum = Array2::<F>::zeros((m * n, m * n));

        // Build I ⊗ A + B^T ⊗ I
        for i in 0..m {
            for j in 0..n {
                let row = i * n + j;

                // I ⊗ A contribution
                for k in 0..m {
                    let col = k * n + j;
                    kron_sum[[row, col]] = a[[i, k]];
                }

                // B^T ⊗ I contribution
                for l in 0..n {
                    let col = i * n + l;
                    kron_sum[[row, col]] += b[[l, j]];
                }
            }
        }

        // Vectorize C
        let mut c_vec = ndarray::Array1::<F>::zeros(m * n);
        for i in 0..m {
            for j in 0..n {
                c_vec[i * n + j] = c[[i, j]];
            }
        }

        // Solve the linear system
        let x_vec = solve_linear_system(&kron_sum.view(), &c_vec.view())?;

        // Reshape back to matrix
        let mut x = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                x[[i, j]] = x_vec[i * n + j];
            }
        }

        Ok(x)
    } else {
        // For larger matrices, use iterative method (simplified)
        // X_{k+1} = (C - X_k B) A^{-1}
        let mut x = c.to_owned();
        let max_iter = 100;
        let tol = F::epsilon() * F::from(100.0).unwrap();

        for _ in 0..max_iter {
            let x_old = x.clone();

            // Compute C - X*B
            let xb = x.dot(b);
            let rhs = c - &xb;

            // Solve A*X_new = rhs
            x = solve_matrix_equation_right(a, &rhs.view())?;

            // Check convergence
            let diff = (&x - &x_old).mapv(|v| v.abs()).sum();
            if diff < tol {
                break;
            }
        }

        Ok(x)
    }
}

/// Solve Lyapunov equation AX + XA^T = Q
fn solve_lyapunov_internal<F: Float + ndarray::ScalarOperand>(
    a: &ArrayView2<F>,
    q: &ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    // Lyapunov equation is a special case of Sylvester equation
    // where B = A^T
    let at = a.t();
    solve_sylvester_internal(a, &at, q)
}

/// Helper to solve linear system
fn solve_linear_system<F: Float>(
    a: &ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
) -> Result<ndarray::Array1<F>, OpError> {
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
    let mut x = ndarray::Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            let x_j = x[j];
            x[i] -= aug[[i, j]] * x_j;
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x)
}

/// Solve AX = B for matrix B
fn solve_matrix_equation_right<F: Float>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = a.shape()[0];
    let m = b.shape()[1];
    let mut x = Array2::<F>::zeros((n, m));

    // Solve for each column
    for j in 0..m {
        let b_col = b.column(j);
        let x_col = solve_linear_system(a, &b_col)?;

        for i in 0..n {
            x[[i, j]] = x_col[i];
        }
    }

    Ok(x)
}

/// Compute Cholesky decomposition
fn compute_cholesky<F: Float>(matrix: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let mut sum = F::zero();
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag_val = matrix[[j, j]] - sum;
                if diag_val <= F::zero() {
                    return Err(OpError::Other("Matrix is not positive definite".into()));
                }
                l[[j, j]] = diag_val.sqrt();
            } else {
                // Off-diagonal elements
                let mut sum = F::zero();
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Solve LLᵀx = b using Cholesky decomposition (1D case)
fn solve_cholesky_1d<F: Float>(
    l: &ArrayView2<F>,
    b: &ndarray::ArrayView1<F>,
) -> Result<ndarray::Array1<F>, OpError> {
    let n = l.shape()[0];

    // Forward substitution: Ly = b
    let mut y = ndarray::Array1::<F>::zeros(n);
    for i in 0..n {
        y[i] = b[i];
        for j in 0..i {
            let y_j = y[j];
            y[i] -= l[[i, j]] * y_j;
        }
        y[i] /= l[[i, i]];
    }

    // Back substitution: Lᵀx = y
    let mut x = ndarray::Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            let x_j = x[j];
            x[i] -= l[[j, i]] * x_j;
        }
        x[i] /= l[[i, i]];
    }

    Ok(x)
}

/// Solve LLᵀX = B using Cholesky decomposition (2D case)
fn solve_cholesky_2d<F: Float>(l: &ArrayView2<F>, b: &ArrayView2<F>) -> Result<Array2<F>, OpError> {
    let n = l.shape()[0];
    let m = b.shape()[1];
    let mut x = Array2::<F>::zeros((n, m));

    // Solve for each column
    for j in 0..m {
        let b_col = b.column(j);
        let x_col = solve_cholesky_1d(l, &b_col)?;

        for i in 0..n {
            x[[i, j]] = x_col[i];
        }
    }

    Ok(x)
}

/// Compute outer product gradient
fn compute_outer_product_gradient<F: Float>(
    a: &ndarray::ArrayViewD<F>,
    b: &ndarray::ArrayViewD<F>,
) -> Result<ndarray::ArrayD<F>, OpError> {
    if a.ndim() == 1 && b.ndim() == 1 {
        let a_1d = match a.view().into_dimensionality::<ndarray::Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        let b_1d = match b.view().into_dimensionality::<ndarray::Ix1>() {
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

        let b_1d = match b.view().into_dimensionality::<ndarray::Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        Ok(a_2d
            .dot(&b_1d.view().insert_axis(ndarray::Axis(0)))
            .into_dyn())
    } else if a.ndim() == 1 && b.ndim() == 2 {
        let a_1d = match a.view().into_dimensionality::<ndarray::Ix1>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 1D".into())),
        };

        let b_2d = match b.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => return Err(OpError::IncompatibleShape("Failed to convert to 2D".into())),
        };

        Ok(a_1d
            .view()
            .insert_axis(ndarray::Axis(1))
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

/// Solve Sylvester equation AX + XB = C
pub fn solve_sylvester<'g, F: Float + ndarray::ScalarOperand>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    c: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = a.graph();
    let c_shape = crate::tensor_ops::shape(c);

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .append_input(c, false)
        .set_shape(&c_shape)
        .build(SylvesterSolveOp)
}

/// Solve Lyapunov equation AX + XA^T = Q
pub fn solve_lyapunov<'g, F: Float + ndarray::ScalarOperand>(
    a: &Tensor<'g, F>,
    q: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = a.graph();
    let q_shape = crate::tensor_ops::shape(q);

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(q, false)
        .set_shape(&q_shape)
        .build(LyapunovSolveOp)
}

/// Solve linear system AX = B using Cholesky decomposition for positive definite A
pub fn cholesky_solve<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();
    let b_shape = crate::tensor_ops::shape(b);

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .set_shape(&b_shape)
        .build(CholeskySolveOp)
}
