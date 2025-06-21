use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array1, Array2, Ix2};
use num_traits::FromPrimitive;

/// Matrix square root operation
pub struct MatrixSqrtOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSqrtOp {
    fn name(&self) -> &'static str {
        "MatrixSqrt"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix square root requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute matrix square root
        let result = compute_matrix_sqrt(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let g = ctx.graph();
        let input = ctx.input(0);
        let shape = input.shape().to_vec();

        // For matrix square root, the gradient is computed by solving
        // the Sylvester equation: dA/2 = X * dY + dY * X
        // where X = sqrt(A), Y is the gradient w.r.t output
        // This gives us: dA = solve_sylvester(X, X, 2*dY)

        // For now, we'll use a simplified approach with zeros
        // to maintain the correct shape
        if shape.len() == 2 {
            let grad_zeros = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
            let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_zeros, g);
            ctx.append_input_grad(0, Some(grad_tensor));
        } else {
            ctx.append_input_grad(0, None);
        }
    }
}

/// Matrix logarithm operation
pub struct MatrixLogOp;

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixLogOp {
    fn name(&self) -> &'static str {
        "MatrixLog"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix logarithm requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute matrix logarithm
        let result = compute_matrix_log(&input_2d)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let g = ctx.graph();
        let input = ctx.input(0);
        let shape = input.shape().to_vec();

        // For matrix logarithm, the gradient involves solving
        // a complex equation involving the Fréchet derivative
        // For now, we'll use a simplified approach with zeros
        // to maintain the correct shape
        if shape.len() == 2 {
            let grad_zeros = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
            let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_zeros, g);
            ctx.append_input_grad(0, Some(grad_tensor));
        } else {
            ctx.append_input_grad(0, None);
        }
    }
}

/// Matrix power operation
pub struct MatrixPowOp {
    pub power: f64,
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixPowOp {
    fn name(&self) -> &'static str {
        "MatrixPow"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix power requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute matrix power
        let result = compute_matrix_pow(&input_2d, self.power)?;
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let g = ctx.graph();
        let input = ctx.input(0);
        let shape = input.shape().to_vec();

        // Gradient of matrix power: p * A^(p-1) for scalar gradient
        // For matrix gradient it's more complex
        // For now, we'll use a simplified approach with zeros
        // to maintain the correct shape
        if shape.len() == 2 {
            let grad_zeros = ndarray::ArrayD::zeros(ndarray::IxDyn(&shape));
            let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_zeros, g);
            ctx.append_input_grad(0, Some(grad_tensor));
        } else {
            ctx.append_input_grad(0, None);
        }
    }
}

// Helper functions

/// Compute matrix square root using eigendecomposition
fn compute_matrix_sqrt<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Check if matrix is symmetric and positive semi-definite
    if is_symmetric_matrix(matrix) && is_positive_semidefinite(matrix)? {
        // Use eigendecomposition for symmetric positive semi-definite matrices
        let (eigenvalues, eigenvectors) = compute_symmetric_eigen(matrix)?;

        // Check all eigenvalues are non-negative
        for &lambda in eigenvalues.iter() {
            if lambda < -F::epsilon() * F::from(10.0).unwrap() {
                return Err(OpError::Other(
                    "Matrix has negative eigenvalues, cannot compute real square root".into(),
                ));
            }
        }

        // Compute sqrt of eigenvalues
        let mut sqrt_eigenvalues = Array1::<F>::zeros(n);
        for i in 0..n {
            sqrt_eigenvalues[i] = eigenvalues[i].abs().sqrt();
        }

        // Reconstruct: sqrt(A) = V * diag(sqrt(λ)) * V^T
        // Check if eigenvectors are identity (diagonal matrix case)
        let is_diagonal = {
            let mut diag = true;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        if (eigenvectors[[i, j]] - F::one()).abs()
                            > F::epsilon() * F::from(10.0).unwrap()
                        {
                            diag = false;
                            break;
                        }
                    } else if eigenvectors[[i, j]].abs() > F::epsilon() * F::from(10.0).unwrap() {
                        diag = false;
                        break;
                    }
                }
                if !diag {
                    break;
                }
            }
            diag
        };

        if is_diagonal {
            // Matrix is diagonal, just return diagonal sqrt
            let mut result = Array2::<F>::zeros((n, n));
            for i in 0..n {
                result[[i, i]] = sqrt_eigenvalues[i];
            }
            Ok(result)
        } else {
            let mut temp = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    temp[[i, j]] = eigenvectors[[i, j]] * sqrt_eigenvalues[j];
                }
            }

            let result = temp.dot(&eigenvectors.t());
            Ok(result)
        }
    } else {
        // For general matrices, use Schur decomposition method
        // For now, return an approximation using Denman-Beavers iteration
        compute_matrix_sqrt_denman_beavers(matrix)
    }
}

/// Compute matrix logarithm using eigendecomposition
fn compute_matrix_log<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    // Check if matrix is symmetric
    if is_symmetric_matrix(matrix) {
        // Use eigendecomposition for symmetric matrices
        let (eigenvalues, eigenvectors) = compute_symmetric_eigen(matrix)?;

        // Check all eigenvalues are positive
        for &lambda in eigenvalues.iter() {
            if lambda <= F::epsilon() {
                return Err(OpError::Other(
                    "Matrix has non-positive eigenvalues, cannot compute real logarithm".into(),
                ));
            }
        }

        // Compute log of eigenvalues
        let mut log_eigenvalues = Array1::<F>::zeros(n);
        for i in 0..n {
            log_eigenvalues[i] = eigenvalues[i].ln();
        }

        // Reconstruct: log(A) = V * diag(log(λ)) * V^T
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                temp[[i, j]] = eigenvectors[[i, j]] * log_eigenvalues[j];
            }
        }

        let result = temp.dot(&eigenvectors.t());
        Ok(result)
    } else {
        // For general matrices, use inverse scaling and squaring method
        compute_matrix_log_inverse_scaling(matrix)
    }
}

/// Compute matrix power using eigendecomposition or repeated squaring
fn compute_matrix_pow<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
    power: f64,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let p = F::from(power).ok_or(OpError::Other("Invalid power value".into()))?;

    // Special cases
    if power == 0.0 {
        return Ok(Array2::<F>::eye(n));
    } else if power == 1.0 {
        return Ok(matrix.to_owned());
    } else if power == -1.0 {
        return compute_matrix_inverse(matrix);
    }

    // For integer powers, use repeated squaring
    if power.fract() == 0.0 && power.abs() < 100.0 {
        let int_power = power as i32;
        return compute_matrix_pow_integer(matrix, int_power);
    }

    // For symmetric matrices, use eigendecomposition
    if is_symmetric_matrix(matrix) {
        let (eigenvalues, eigenvectors) = compute_symmetric_eigen(matrix)?;

        // Check for negative eigenvalues if power is not integer
        if power.fract() != 0.0 {
            for &lambda in eigenvalues.iter() {
                if lambda < -F::epsilon() * F::from(10.0).unwrap() {
                    return Err(OpError::Other(
                        "Matrix has negative eigenvalues, cannot compute real fractional power"
                            .into(),
                    ));
                }
            }
        }

        // Compute power of eigenvalues
        let mut pow_eigenvalues = Array1::<F>::zeros(n);
        for i in 0..n {
            if eigenvalues[i].abs() > F::epsilon() {
                pow_eigenvalues[i] = eigenvalues[i].powf(p);
            } else {
                pow_eigenvalues[i] = F::zero();
            }
        }

        // Reconstruct: A^p = V * diag(λ^p) * V^T
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                temp[[i, j]] = eigenvectors[[i, j]] * pow_eigenvalues[j];
            }
        }

        let result = temp.dot(&eigenvectors.t());
        Ok(result)
    } else {
        // For general matrices with fractional powers, use exp(p * log(A))
        let log_a = compute_matrix_log_inverse_scaling(matrix)?;
        let p_log_a = log_a.mapv(|x| x * p);
        compute_matrix_exp_pade(&p_log_a.view())
    }
}

/// Denman-Beavers iteration for matrix square root
fn compute_matrix_sqrt_denman_beavers<F: Float + ndarray::ScalarOperand>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut y = matrix.to_owned();
    let mut z = Array2::<F>::eye(n);

    let max_iter = 50;
    let tol = F::epsilon() * F::from(100.0).unwrap();

    for _ in 0..max_iter {
        let y_old = y.clone();

        // Compute inverses
        let y_inv = compute_matrix_inverse(&y.view())?;
        let z_inv = compute_matrix_inverse(&z.view())?;

        // Update Y and Z
        y = (&y + &z_inv) / F::from(2.0).unwrap();
        z = (&z + &y_inv) / F::from(2.0).unwrap();

        // Check convergence
        let diff = (&y - &y_old).mapv(|x| x.abs()).sum();
        if diff < tol {
            break;
        }
    }

    Ok(y)
}

/// Inverse scaling and squaring method for matrix logarithm
fn compute_matrix_log_inverse_scaling<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let i = Array2::<F>::eye(n);

    // Find s such that ||A^(1/2^s) - I|| < 0.5
    let mut s = 0;
    loop {
        let norm = (&a - &i).mapv(|x| x.abs()).sum() / F::from(n * n).unwrap();
        if norm < F::from(0.5).unwrap() {
            break;
        }
        // Take square root
        a = compute_matrix_sqrt_denman_beavers(&a.view())?;
        s += 1;
        if s > 20 {
            return Err(OpError::Other("Matrix logarithm failed to converge".into()));
        }
    }

    // Compute log using Padé approximation for log(I + X) where X = A - I
    let x = &a - &i;
    let mut log_a = compute_log_pade(&x)?;

    // Scale back
    log_a *= F::from(2.0_f64.powi(s)).unwrap();

    Ok(log_a)
}

/// Padé approximation for log(I + X)
fn compute_log_pade<F: Float + ndarray::ScalarOperand>(
    x: &Array2<F>,
) -> Result<Array2<F>, OpError> {
    let n = x.shape()[0];

    // Use Padé [3/3] approximation
    // log(I + X) ≈ X * (I + X/2 + X²/10) / (I + X/2 + 3X²/10)
    let x2 = x.dot(x);
    let half = F::from(0.5).unwrap();
    let tenth = F::from(0.1).unwrap();
    let three_tenths = F::from(0.3).unwrap();

    let i = Array2::<F>::eye(n);
    let numerator = &i + &(x * half) + &(&x2 * tenth);
    let denominator = &i + &(x * half) + &(&x2 * three_tenths);

    // Solve denominator * result = x * numerator
    let rhs = x.dot(&numerator);
    solve_matrix_equation(&denominator.view(), &rhs.view())
}

/// Integer matrix power using repeated squaring
fn compute_matrix_pow_integer<F: Float + ndarray::ScalarOperand>(
    matrix: &ndarray::ArrayView2<F>,
    power: i32,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    if power == 0 {
        return Ok(Array2::<F>::eye(n));
    }

    let abs_power = power.unsigned_abs();
    let mut result = Array2::<F>::eye(n);
    let mut base = if power > 0 {
        matrix.to_owned()
    } else {
        compute_matrix_inverse(matrix)?
    };

    let mut p = abs_power;
    while p > 0 {
        if p & 1 == 1 {
            result = result.dot(&base);
        }
        base = base.dot(&base);
        p >>= 1;
    }

    Ok(result)
}

// Utility functions (reuse from other modules or implement here)

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

fn is_positive_semidefinite<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<bool, OpError> {
    if !is_symmetric_matrix(matrix) {
        return Ok(false);
    }

    // Check eigenvalues
    let (eigenvalues, _) = compute_symmetric_eigen(matrix)?;
    for &lambda in eigenvalues.iter() {
        if lambda < -F::epsilon() * F::from(10.0).unwrap() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn compute_symmetric_eigen<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    // This is a simplified version - in practice, use LAPACK or a robust algorithm
    let n = matrix.shape()[0];

    // For 2x2 matrices, use analytical solution
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
            // For diagonal matrices, eigenvalues should match diagonal order
            let eigenvalues = Array1::from_vec(vec![a, c]);
            return Ok((eigenvalues, eigenvectors));
        }

        return Ok((eigenvalues, eigenvectors));
    }

    // For larger matrices, use a simplified approach
    // In production, use a proper eigenvalue algorithm
    let mut eigenvalues = Array1::<F>::zeros(n);
    let eigenvectors = Array2::<F>::eye(n);

    // Use diagonal approximation for now
    for i in 0..n {
        eigenvalues[i] = matrix[[i, i]];
    }

    Ok((eigenvalues, eigenvectors))
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

fn solve_matrix_equation<F: Float>(
    a: &ndarray::ArrayView2<F>,
    b: &ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    // Solve AX = B using LU decomposition or direct inversion
    let a_inv = compute_matrix_inverse(a)?;
    Ok(a_inv.dot(b))
}

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

// Public API functions

/// Compute matrix square root
pub fn matrix_sqrt<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixSqrtOp)
}

/// Compute matrix logarithm
pub fn matrix_log<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixLogOp)
}

/// Compute matrix power
pub fn matrix_power<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    power: f64,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrix_shape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .set_shape(&matrix_shape)
        .build(MatrixPowOp { power })
}

// Aliases
pub use self::matrix_log as logm;
pub use self::matrix_power as powm;
pub use self::matrix_sqrt as sqrtm;
