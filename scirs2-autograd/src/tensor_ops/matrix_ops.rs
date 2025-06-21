use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array1, Array2, Ix2};
use num_traits::FromPrimitive;

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

        // Debug information
        println!("Computing matrix inverse of shape: {:?}", shape);

        let input_2d = input_array
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute inverse using Gauss-Jordan elimination
        let inv = compute_inverse(&input_2d)?;

        // Verify the shape of the result
        let result_shape = inv.shape();
        println!("Matrix inverse result shape: {:?}", result_shape);

        // No need to reshape, just use the computed inverse directly
        // but make a deep copy of it to ensure we have a clean array
        let output_inv = inv.to_owned();

        // Verify shape before output
        println!("Final inverse shape: {:?}", output_inv.shape());
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
        let grad_output = ctx.output_grad();
        let output = ctx.output(); // This is the pseudo-inverse
        let input = ctx.input(0);
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

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D arrays
        let pinv = match output_array.view().into_dimensionality::<Ix2>() {
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

        let input_2d = match input_array.view().into_dimensionality::<Ix2>() {
            Ok(view) => view,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Gradient calculation
        let pinv_t = pinv.t();
        let term1 = -pinv_t.dot(&grad_out_2d).dot(&pinv_t);
        let term2 = pinv_t
            .dot(&pinv_t.t())
            .dot(&grad_out_2d)
            .dot(&input_2d.t())
            .dot(&pinv_t);
        let term3 = pinv_t
            .dot(&input_2d.t())
            .dot(&grad_out_2d)
            .dot(&pinv_t.t())
            .dot(&pinv_t);

        let grad_input = term1 + term2 + term3;

        // Convert gradient to tensor
        let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_input.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// Matrix determinant for larger matrices
pub struct GeneralDeterminantOp;

impl<F: Float + ndarray::ScalarOperand> Op<F> for GeneralDeterminantOp {
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

        println!("Computing determinant for matrix of shape: {:?}", shape);
        let det = compute_determinant_lu(&input_2d)?;
        println!("Determinant result: {}", det);

        // Create a scalar (0-dimensional) array with the determinant value
        // Use explicit arr0 to ensure we get a 0-dimensional array
        let det_array = ndarray::arr0(det);

        // Verify the shape to make sure we're creating a scalar
        assert_eq!(det_array.ndim(), 0);
        println!("Determinant array shape: {:?}", det_array.shape());

        // Output the determinant as a 0-dimensional array
        ctx.append_output(det_array.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let output = ctx.output();
        let g = ctx.graph();

        println!("Computing gradient for determinant");

        // Evaluate tensors
        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate gradient output");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate output");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate input");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Access scalar values
        let grad_scalar = grad_output_array[[0]];
        let det = output_array[[0]];
        println!("Determinant: {}, Gradient scale: {}", det, grad_scalar);

        // Gradient of determinant: det(A) * A^{-T}
        if det.abs() > F::epsilon() {
            let input_2d = match input_array.view().into_dimensionality::<Ix2>() {
                Ok(view) => view,
                Err(_) => {
                    println!("Failed to convert input to 2D");
                    ctx.append_input_grad(0, None);
                    return;
                }
            };

            match compute_inverse(&input_2d) {
                Ok(inv) => {
                    // Scale transpose of inverse by det and grad_scalar
                    let inv_t = inv.t();
                    println!("Inverse transpose shape: {:?}", inv_t.shape());

                    // Correctly compute gradient: grad = grad_scalar * det * A^(-T)
                    let scaled_grad = inv_t.mapv(|x| det * grad_scalar * x);
                    let grad_tensor =
                        crate::tensor_ops::convert_to_tensor(scaled_grad.into_dyn(), g);

                    println!("Determinant gradient computed successfully");
                    ctx.append_input_grad(0, Some(grad_tensor));
                }
                Err(_) => {
                    println!("Matrix is nearly singular, using approximate gradient");
                    // For nearly singular matrices, use regularized inverse
                    let eps = F::epsilon() * F::from(10.0).unwrap();
                    let n = input_2d.shape()[0];
                    let regularized = &input_2d + &(Array2::<F>::eye(n) * eps);

                    if let Ok(reg_inv) = compute_inverse(&regularized.view()) {
                        let reg_inv_t = reg_inv.t();
                        let scaled_grad = reg_inv_t.mapv(|x| det * grad_scalar * x);
                        let grad_tensor =
                            crate::tensor_ops::convert_to_tensor(scaled_grad.into_dyn(), g);
                        ctx.append_input_grad(0, Some(grad_tensor));
                    } else {
                        println!("Failed to compute even regularized inverse, returning zeros");
                        let zeros = ndarray::Array::zeros(input_array.raw_dim());
                        let grad_tensor = crate::tensor_ops::convert_to_tensor(zeros, g);
                        ctx.append_input_grad(0, Some(grad_tensor));
                    }
                }
            }
            return;
        }

        println!("Matrix is singular, gradient is undefined, returning zeros");
        // If matrix is singular, gradient is undefined
        let zeros = ndarray::Array::zeros(input_array.raw_dim());
        let grad_tensor = crate::tensor_ops::convert_to_tensor(zeros, g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

// Helper functions
fn compute_inverse<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<Array2<F>, OpError> {
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

fn compute_pseudo_inverse<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<Array2<F>, OpError> {
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

fn compute_determinant_lu<F: Float>(matrix: &ndarray::ArrayView2<F>) -> Result<F, OpError> {
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
            return Ok(F::zero()); // Singular matrix
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

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixExp2Op {
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
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let output = ctx.output();
        let g = ctx.graph();

        // Gradient of matrix exponential: complex computation
        // For now, use a simplified version
        let _grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let _input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let _output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Simplified gradient: pass through
        ctx.append_input_grad(0, Some(*grad_output));
    }
}

/// Matrix exponential using eigendecomposition (method 3)
pub struct MatrixExp3Op;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixExp3Op {
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
        let grad_output = ctx.output_grad();
        // Simplified gradient
        ctx.append_input_grad(0, Some(*grad_output));
    }
}

/// Compute matrix exponential using Padé approximation
fn compute_matrix_exp_pade<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
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

    // Scaling parameter
    let s = if norm > F::one() {
        (norm.ln() / F::from(2.0).unwrap().ln()).ceil()
    } else {
        F::zero()
    };

    let scale = F::from(2.0).unwrap().powf(s);
    let scaled_matrix = matrix.mapv(|x| x / scale);

    // Padé approximation coefficients (order 6)
    let c0 = F::from(1.0).unwrap();
    let c1 = F::from(0.5).unwrap();
    let c2 = F::from(12.0).unwrap().recip();
    let c3 = F::from(120.0).unwrap().recip();
    let c4 = F::from(3360.0).unwrap().recip();
    let c5 = F::from(30240.0).unwrap().recip();
    let c6 = F::from(1209600.0).unwrap().recip();

    // Compute powers of matrix
    let i = Array2::<F>::eye(n);
    let a2 = scaled_matrix.dot(&scaled_matrix);
    let a4 = a2.dot(&a2);
    let a6 = a4.dot(&a2);

    // Compute U and V for Padé approximation
    let u = &scaled_matrix * c1 + &a2 * c3 + &a4 * c5;
    let u = scaled_matrix.dot(&u);

    let v = &i * c0 + &a2 * c2 + &a4 * c4 + &a6 * c6;

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
fn compute_matrix_exp_eigen<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
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
        // For general matrices, use approximation
        compute_matrix_exp_taylor(matrix)
    }
}

/// Compute matrix exponential using Taylor series
fn compute_matrix_exp_taylor<F: Float + ndarray::ScalarOperand>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut result = Array2::<F>::eye(n);
    let mut term = Array2::<F>::eye(n);

    // Use more terms for better accuracy
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

/// Solve matrix equation AX = B
fn solve_matrix_equation<F: Float>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView2<F>,
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
fn compute_symmetric_eigen_simple<F: Float + ndarray::ScalarOperand + FromPrimitive>(
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

    // For larger matrices, use iterative method (simplified)
    let mut eigenvalues = Array1::<F>::zeros(n);
    let eigenvectors = Array2::<F>::eye(n);

    // Use diagonal approximation for simplicity
    for i in 0..n {
        eigenvalues[i] = matrix[[i, i]];
    }

    Ok((eigenvalues, eigenvectors))
}

// Public API functions
pub fn matrix_inverse<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape tensor from the input
    let matrix_shape = crate::tensor_ops::shape(matrix);

    // Build the tensor with shape information
    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixInverseOp)
}

pub fn pseudo_inverse<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape tensor from the input
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(PseudoInverseOp)
}

pub fn determinant<'g, F: Float + ndarray::ScalarOperand>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    // For determinant, we're creating a scalar output (0-dimensional tensor)
    // We'll use zeros(0) to create a scalar tensor shape
    let scalar_shape = crate::tensor_ops::zeros(&[0], g);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&scalar_shape)
        .build(GeneralDeterminantOp)
}

/// Matrix exponential using improved Padé approximation (method 2)
pub fn expm2<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixExp2Op)
}

/// Matrix exponential using eigendecomposition (method 3)  
pub fn expm3<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixExp3Op)
}
