use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array1, Array2, Ix2};
use num_traits::FromPrimitive;

/// Matrix sine function
pub struct MatrixSineOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSineOp {
    fn name(&self) -> &'static str {
        "MatrixSine"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix sine requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let result = compute_matrix_sine(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Gradient of sin(A) is cos(A) ⊙ grad_output (element-wise product)
        if let Ok(input_array) = input.eval(g) {
            if let Ok(input_2d) = input_array.view().into_dimensionality::<Ix2>() {
                if let Ok(cos_a) = compute_matrix_cosine(&input_2d) {
                    if let Ok(grad_array) = grad_output.eval(g) {
                        if let Ok(grad_2d) = grad_array.view().into_dimensionality::<Ix2>() {
                            // Element-wise multiplication
                            let grad_input = cos_a * grad_2d;
                            let grad_tensor =
                                crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
                            ctx.append_input_grad(0, Some(grad_tensor));
                            return;
                        }
                    }
                }
            }
        }

        ctx.append_input_grad(0, None);
    }
}

/// Matrix cosine function
pub struct MatrixCosineOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixCosineOp {
    fn name(&self) -> &'static str {
        "MatrixCosine"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix cosine requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let result = compute_matrix_cosine(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Gradient of cos(A) is -sin(A) ⊙ grad_output
        if let Ok(input_array) = input.eval(g) {
            if let Ok(input_2d) = input_array.view().into_dimensionality::<Ix2>() {
                if let Ok(sin_a) = compute_matrix_sine(&input_2d) {
                    if let Ok(grad_array) = grad_output.eval(g) {
                        if let Ok(grad_2d) = grad_array.view().into_dimensionality::<Ix2>() {
                            let grad_input = -sin_a * grad_2d;
                            let grad_tensor =
                                crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
                            ctx.append_input_grad(0, Some(grad_tensor));
                            return;
                        }
                    }
                }
            }
        }

        ctx.append_input_grad(0, None);
    }
}

/// Matrix sign function
pub struct MatrixSignOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSignOp {
    fn name(&self) -> &'static str {
        "MatrixSign"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix sign requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let result = compute_matrix_sign(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Gradient of sign function is complex, using simplified version
        let grad_output = ctx.output_grad();
        ctx.append_input_grad(0, Some(*grad_output));
    }
}

/// Matrix hyperbolic sine function
pub struct MatrixSinhOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSinhOp {
    fn name(&self) -> &'static str {
        "MatrixSinh"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix sinh requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let result = compute_matrix_sinh(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Gradient of sinh(A) is cosh(A) ⊙ grad_output
        if let Ok(input_array) = input.eval(g) {
            if let Ok(input_2d) = input_array.view().into_dimensionality::<Ix2>() {
                if let Ok(cosh_a) = compute_matrix_cosh(&input_2d) {
                    if let Ok(grad_array) = grad_output.eval(g) {
                        if let Ok(grad_2d) = grad_array.view().into_dimensionality::<Ix2>() {
                            let grad_input = cosh_a * grad_2d;
                            let grad_tensor =
                                crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
                            ctx.append_input_grad(0, Some(grad_tensor));
                            return;
                        }
                    }
                }
            }
        }

        ctx.append_input_grad(0, None);
    }
}

/// Matrix hyperbolic cosine function
pub struct MatrixCoshOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixCoshOp {
    fn name(&self) -> &'static str {
        "MatrixCosh"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix cosh requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let result = compute_matrix_cosh(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Gradient of cosh(A) is sinh(A) ⊙ grad_output
        if let Ok(input_array) = input.eval(g) {
            if let Ok(input_2d) = input_array.view().into_dimensionality::<Ix2>() {
                if let Ok(sinh_a) = compute_matrix_sinh(&input_2d) {
                    if let Ok(grad_array) = grad_output.eval(g) {
                        if let Ok(grad_2d) = grad_array.view().into_dimensionality::<Ix2>() {
                            let grad_input = sinh_a * grad_2d;
                            let grad_tensor =
                                crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
                            ctx.append_input_grad(0, Some(grad_tensor));
                            return;
                        }
                    }
                }
            }
        }

        ctx.append_input_grad(0, None);
    }
}

/// General matrix function using eigendecomposition
pub struct MatrixFunctionOp<F: Float> {
    function: fn(F) -> F,
    name: &'static str,
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixFunctionOp<F> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix function requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let result = compute_matrix_function(&input_2d, self.function)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Simplified gradient
        let grad_output = ctx.output_grad();
        ctx.append_input_grad(0, Some(*grad_output));
    }
}

// Helper functions

/// Compute matrix sine using Taylor series
fn compute_matrix_sine<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut result = matrix.to_owned();
    let mut term = matrix.to_owned();
    let a2 = matrix.dot(matrix);

    // sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
    for k in 1..10 {
        term = -term.dot(&a2) / F::from((2 * k) * (2 * k + 1)).unwrap();
        let old_result = result.clone();
        result += &term;

        // Check convergence
        let diff = (&result - &old_result).mapv(|x| x.abs()).sum();
        if diff < F::epsilon() * F::from(n as f64).unwrap() {
            break;
        }
    }

    Ok(result)
}

/// Compute matrix cosine using Taylor series
fn compute_matrix_cosine<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut result = Array2::<F>::eye(n);
    let mut term = Array2::<F>::eye(n);
    let a2 = matrix.dot(matrix);

    // cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
    for k in 1..10 {
        term = -term.dot(&a2) / F::from((2 * k - 1) * (2 * k)).unwrap();
        let old_result = result.clone();
        result += &term;

        // Check convergence
        let diff = (&result - &old_result).mapv(|x| x.abs()).sum();
        if diff < F::epsilon() * F::from(n as f64).unwrap() {
            break;
        }
    }

    Ok(result)
}

/// Compute matrix sign function using Newton iteration
fn compute_matrix_sign<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut x = matrix.to_owned();
    let max_iter = 20;
    let tol = F::epsilon() * F::from(100.0).unwrap();

    // Newton iteration: X_{k+1} = (X_k + X_k^{-1}) / 2
    for _ in 0..max_iter {
        let x_inv = compute_matrix_inverse(&x.view())?;
        let x_new = (&x + &x_inv) / F::from(2.0).unwrap();

        // Check convergence
        let diff = (&x_new - &x).mapv(|x| x.abs()).sum();
        x = x_new;

        if diff < tol * F::from(n as f64).unwrap() {
            break;
        }
    }

    Ok(x)
}

/// Compute matrix hyperbolic sine
fn compute_matrix_sinh<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    // sinh(A) = (exp(A) - exp(-A)) / 2
    let exp_a = compute_matrix_exp(matrix)?;
    let neg_a = matrix.mapv(|x| -x);
    let exp_neg_a = compute_matrix_exp(&neg_a.view())?;

    Ok((exp_a - exp_neg_a) / F::from(2.0).unwrap())
}

/// Compute matrix hyperbolic cosine
fn compute_matrix_cosh<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    // cosh(A) = (exp(A) + exp(-A)) / 2
    let exp_a = compute_matrix_exp(matrix)?;
    let neg_a = matrix.mapv(|x| -x);
    let exp_neg_a = compute_matrix_exp(&neg_a.view())?;

    Ok((exp_a + exp_neg_a) / F::from(2.0).unwrap())
}

/// Compute matrix exponential (from matrix_ops.rs)
fn compute_matrix_exp<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut result = Array2::<F>::eye(n);
    let mut term = Array2::<F>::eye(n);

    // Use Taylor series with more terms for accuracy
    for k in 1..=20 {
        term = term.dot(matrix) / F::from(k).unwrap();
        let old_result = result.clone();
        result += &term;

        // Check convergence
        let diff = (&result - &old_result).mapv(|x| x.abs()).sum();
        if diff < F::epsilon() * F::from(n as f64).unwrap() {
            break;
        }
    }

    Ok(result)
}

/// Compute matrix inverse
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

/// Compute general matrix function using eigendecomposition
fn compute_matrix_function<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
    func: fn(F) -> F,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Check if matrix is symmetric
    let is_symmetric = is_symmetric_matrix(matrix);

    if is_symmetric {
        // For symmetric matrices, use real eigendecomposition
        let (eigenvalues, eigenvectors) = compute_symmetric_eigen(matrix)?;

        // Apply function to eigenvalues
        let mut func_eigenvalues = Array1::<F>::zeros(n);
        for i in 0..n {
            func_eigenvalues[i] = func(eigenvalues[i]);
        }

        // Reconstruct: f(A) = V * diag(f(λ)) * V^T
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                temp[[i, j]] = eigenvectors[[i, j]] * func_eigenvalues[j];
            }
        }

        let result = temp.dot(&eigenvectors.t());
        Ok(result)
    } else {
        // For general matrices, use Taylor series or other approximation
        // This is a placeholder - implement based on specific function
        Ok(matrix.to_owned())
    }
}

/// Check if matrix is symmetric
fn is_symmetric_matrix<F: Float>(matrix: &ndarray::ArrayView2<F>) -> bool {
    let n = matrix.shape()[0];
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > F::epsilon() * F::from(10.0).unwrap() {
                return false;
            }
        }
    }
    true
}

/// Simple symmetric eigendecomposition
fn compute_symmetric_eigen<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = matrix.shape()[0];

    // For small matrices, use analytical solution
    if n == 2 {
        let a = matrix[[0, 0]];
        let b = matrix[[0, 1]];
        let c = matrix[[1, 1]];

        let trace = a + c;
        let det = a * c - b * b;
        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(OpError::Other("Complex eigenvalues".into()));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();

        let eigenvalues = Array1::from_vec(vec![lambda1, lambda2]);

        // Compute eigenvectors
        let mut eigenvectors = Array2::<F>::zeros((2, 2));

        if b.abs() > F::epsilon() {
            // First eigenvector
            let v1_0 = lambda1 - c;
            let v1_1 = b;
            let norm1 = (v1_0 * v1_0 + v1_1 * v1_1).sqrt();
            eigenvectors[[0, 0]] = v1_0 / norm1;
            eigenvectors[[1, 0]] = v1_1 / norm1;

            // Second eigenvector
            let v2_0 = lambda2 - c;
            let v2_1 = b;
            let norm2 = (v2_0 * v2_0 + v2_1 * v2_1).sqrt();
            eigenvectors[[0, 1]] = v2_0 / norm2;
            eigenvectors[[1, 1]] = v2_1 / norm2;
        } else {
            // Matrix is diagonal
            eigenvectors[[0, 0]] = F::one();
            eigenvectors[[1, 1]] = F::one();
        }

        return Ok((eigenvalues, eigenvectors));
    }

    // For larger matrices, use simplified Jacobi method
    let mut eigenvalues = Array1::<F>::zeros(n);
    let mut eigenvectors = Array2::<F>::eye(n);
    let mut a = matrix.to_owned();

    let max_iter = 50;
    let tol = F::epsilon() * F::from(10.0).unwrap();

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = F::zero();
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in i + 1..n {
                if a[[i, j]].abs() > max_val {
                    max_val = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation
        let theta = (a[[q, q]] - a[[p, p]]) / (F::from(2.0).unwrap() * a[[p, q]]);
        let t = if theta >= F::zero() {
            F::one() / (theta + (F::one() + theta * theta).sqrt())
        } else {
            -F::one() / (-theta + (F::one() + theta * theta).sqrt())
        };

        let c = F::one() / (F::one() + t * t).sqrt();
        let s = t * c;

        // Update matrix
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        a[[p, p]] = c * c * app - F::from(2.0).unwrap() * s * c * apq + s * s * aqq;
        a[[q, q]] = s * s * app + F::from(2.0).unwrap() * s * c * apq + c * c * aqq;
        a[[p, q]] = F::zero();
        a[[q, p]] = F::zero();

        for i in 0..n {
            if i != p && i != q {
                let aip = a[[i, p]];
                let aiq = a[[i, q]];
                a[[i, p]] = c * aip - s * aiq;
                a[[p, i]] = a[[i, p]];
                a[[i, q]] = s * aip + c * aiq;
                a[[q, i]] = a[[i, q]];
            }
        }

        // Update eigenvectors
        for i in 0..n {
            let vip = eigenvectors[[i, p]];
            let viq = eigenvectors[[i, q]];
            eigenvectors[[i, p]] = c * vip - s * viq;
            eigenvectors[[i, q]] = s * vip + c * viq;
        }
    }

    // Extract diagonal as eigenvalues
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    Ok((eigenvalues, eigenvectors))
}

// Public API functions

/// Compute matrix sine
pub fn sinm<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixSineOp)
}

/// Compute matrix cosine
pub fn cosm<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixCosineOp)
}

/// Compute matrix sign function
pub fn signm<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixSignOp)
}

/// Compute matrix hyperbolic sine
pub fn sinhm<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixSinhOp)
}

/// Compute matrix hyperbolic cosine
pub fn coshm<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixCoshOp)
}

/// Compute general matrix function
pub fn funm<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    func: fn(F) -> F,
    name: &'static str,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixFunctionOp {
            function: func,
            name,
        })
}
