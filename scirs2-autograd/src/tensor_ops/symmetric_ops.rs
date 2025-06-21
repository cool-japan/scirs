use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::Float;
use ndarray::{Array1, Array2, ArrayView2, Ix2};
use num_traits::FromPrimitive;

/// Eigendecomposition for symmetric/Hermitian matrices
pub struct SymmetricEigenOp;

/// Extract operation for symmetric eigenvalue decomposition
pub struct SymmetricEigenExtractOp {
    pub component: usize, // 0 for eigenvalues, 1 for eigenvectors
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for SymmetricEigenOp {
    fn name(&self) -> &'static str {
        "SymmetricEigen"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Symmetric eigendecomposition requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Check if matrix is symmetric
        if !is_symmetric(&input_2d) {
            return Err(OpError::Other("Matrix must be symmetric".into()));
        }

        let (eigenvalues, eigenvectors) = compute_symmetric_eigen(&input_2d)?;

        // Ensure shapes are correct
        assert_eq!(eigenvalues.len(), shape[0]);
        assert_eq!(eigenvectors.shape(), &[shape[0], shape[0]]);

        ctx.append_output(eigenvalues.into_dyn());
        ctx.append_output(eigenvectors.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Get arrays from tensors
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let y_array = match y.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let n = input_array.shape()[0];

        // Extract eigenvalues and eigenvectors from output
        let values_size = n;
        let vectors_start = values_size;

        let eigen_vals = y_array.slice(ndarray::s![0..values_size]);
        let eigen_vecs = y_array.slice(ndarray::s![vectors_start..]);

        let eigen_vals_1d = eigen_vals.to_shape(n).unwrap().to_owned();
        let eigen_vecs_2d = eigen_vecs.to_shape((n, n)).unwrap().to_owned();

        // Get gradients
        let grad_vals = gy_array
            .slice(ndarray::s![0..values_size])
            .to_shape(n)
            .unwrap()
            .to_owned();
        let grad_vecs = gy_array
            .slice(ndarray::s![vectors_start..])
            .to_shape((n, n))
            .unwrap()
            .to_owned();

        // Compute gradient for symmetric eigendecomposition
        let grad_input = symmetric_eigen_gradient(
            &eigen_vals_1d.view(),
            &eigen_vecs_2d.view(),
            &grad_vals.view(),
            &grad_vecs.view(),
        );

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad_input.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

impl<F: Float + ndarray::ScalarOperand + FromPrimitive> Op<F> for SymmetricEigenExtractOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Symmetric eigendecomposition requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Check if matrix is symmetric
        if !is_symmetric(&input_2d) {
            return Err(OpError::Other("Matrix must be symmetric".into()));
        }

        let (eigenvalues, eigenvectors) = compute_symmetric_eigen(&input_2d)?;

        // Return the requested component
        match self.component {
            0 => ctx.append_output(eigenvalues.into_dyn()),
            1 => ctx.append_output(eigenvectors.into_dyn()),
            _ => {
                return Err(OpError::Other(
                    "Invalid component index for symmetric eigen extraction".into(),
                ))
            }
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // For extraction operations, just pass through the gradient
        let _gy = ctx.output_grad();
        let g = ctx.graph();

        // Create a zero gradient for the input since this is just extraction
        let zeros = crate::tensor_ops::zeros(&crate::tensor_ops::shape(ctx.input(0)), g);
        ctx.append_input_grad(0, Some(zeros));
    }
}

// Helper functions

/// Check if matrix is symmetric
fn is_symmetric<F: Float>(matrix: &ArrayView2<F>) -> bool {
    let n = matrix.shape()[0];
    let tol = F::epsilon() * F::from(10.0).unwrap();

    for i in 0..n {
        for j in i + 1..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Compute eigendecomposition for symmetric matrix using Jacobi method
fn compute_symmetric_eigen<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = matrix.shape()[0];

    // Handle special cases
    if n == 1 {
        let eigenvalue = matrix[[0, 0]];
        let eigenvalues = Array1::from_vec(vec![eigenvalue]);
        let eigenvectors = Array2::from_shape_vec((1, 1), vec![F::one()]).unwrap();
        return Ok((eigenvalues, eigenvectors));
    } else if n == 2 {
        // For 2x2 symmetric matrices, use analytical formula
        let a = matrix[[0, 0]];
        let b = matrix[[0, 1]];
        let d = matrix[[1, 1]];

        let trace = a + d;
        let det = a * d - b * b;
        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        if discriminant < F::zero() {
            return Err(OpError::Other(
                "Complex eigenvalues for symmetric matrix".into(),
            ));
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();
        let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();

        let eigenvalues = if lambda1 >= lambda2 {
            Array1::from_vec(vec![lambda1, lambda2])
        } else {
            Array1::from_vec(vec![lambda2, lambda1])
        };

        // Compute eigenvectors
        let mut eigenvectors = Array2::<F>::zeros((2, 2));

        if b.abs() > F::epsilon() {
            // Non-diagonal case
            if lambda1 >= lambda2 {
                // First eigenvector (for lambda1)
                let v1_x = b;
                let v1_y = lambda1 - a;
                let norm1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
                eigenvectors[[0, 0]] = v1_x / norm1;
                eigenvectors[[1, 0]] = v1_y / norm1;

                // Second eigenvector (for lambda2)
                let v2_x = b;
                let v2_y = lambda2 - a;
                let norm2 = (v2_x * v2_x + v2_y * v2_y).sqrt();
                eigenvectors[[0, 1]] = v2_x / norm2;
                eigenvectors[[1, 1]] = v2_y / norm2;
            } else {
                // Swap order
                let v1_x = b;
                let v1_y = lambda2 - a;
                let norm1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
                eigenvectors[[0, 0]] = v1_x / norm1;
                eigenvectors[[1, 0]] = v1_y / norm1;

                let v2_x = b;
                let v2_y = lambda1 - a;
                let norm2 = (v2_x * v2_x + v2_y * v2_y).sqrt();
                eigenvectors[[0, 1]] = v2_x / norm2;
                eigenvectors[[1, 1]] = v2_y / norm2;
            }
        } else {
            // Diagonal matrix
            if lambda1 >= lambda2 {
                eigenvectors[[0, 0]] = F::one();
                eigenvectors[[1, 1]] = F::one();
            } else {
                eigenvectors[[1, 0]] = F::one();
                eigenvectors[[0, 1]] = F::one();
            }
        }

        return Ok((eigenvalues, eigenvectors));
    }

    // For larger matrices, use Jacobi rotation method
    let mut a = matrix.to_owned();
    let mut v = Array2::<F>::eye(n);

    // Jacobi iterations
    let max_iter = 100;
    let tol = F::epsilon() * F::from(100.0).unwrap();

    for _iter in 0..max_iter {
        // Find off-diagonal element with largest magnitude
        let mut max_val = F::zero();
        let mut p = 0;
        let mut q = 0;

        for i in 0..n {
            for j in i + 1..n {
                if a[[i, j]].abs() > max_val {
                    max_val = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        if max_val < tol {
            break;
        }

        // Compute rotation angle
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        let theta = if (app - aqq).abs() < F::epsilon() {
            F::from(std::f64::consts::FRAC_PI_4).unwrap()
        } else {
            F::from(0.5).unwrap() * ((F::from(2.0).unwrap() * apq) / (aqq - app)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();
        let c2 = c * c;
        let s2 = s * s;
        let cs = c * s;

        // Update matrix A
        let app_new = c2 * app + s2 * aqq - F::from(2.0).unwrap() * cs * apq;
        let aqq_new = s2 * app + c2 * aqq + F::from(2.0).unwrap() * cs * apq;

        a[[p, p]] = app_new;
        a[[q, q]] = aqq_new;
        a[[p, q]] = F::zero();
        a[[q, p]] = F::zero();

        // Update other elements
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

        // Update eigenvector matrix
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];

            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = Array1::<F>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    // Sort eigenvalues and eigenvectors in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

    let mut sorted_eigenvalues = Array1::<F>::zeros(n);
    let mut sorted_eigenvectors = Array2::<F>::zeros((n, n));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_eigenvalues[new_idx] = eigenvalues[old_idx];
        for row in 0..n {
            sorted_eigenvectors[[row, new_idx]] = v[[row, old_idx]];
        }
    }

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

/// Compute gradient for symmetric eigendecomposition
fn symmetric_eigen_gradient<F: Float + ndarray::ScalarOperand + FromPrimitive>(
    eigenvalues: &ndarray::ArrayView1<F>,
    eigenvectors: &ArrayView2<F>,
    grad_vals: &ndarray::ArrayView1<F>,
    grad_vecs: &ArrayView2<F>,
) -> Array2<F> {
    let n = eigenvalues.len();
    let mut grad = Array2::<F>::zeros((n, n));

    // Gradient for eigenvalues part
    for i in 0..n {
        let vi = eigenvectors.slice(ndarray::s![.., i]);
        for j in 0..n {
            for k in 0..n {
                grad[[j, k]] += grad_vals[i] * vi[j] * vi[k];
            }
        }
    }

    // Gradient for eigenvectors part
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let lambda_diff = eigenvalues[j] - eigenvalues[i];

                if lambda_diff.abs() > F::epsilon() * F::from(10.0).unwrap() {
                    let factor = F::one() / lambda_diff;

                    // Compute v_i^T @ grad_vecs[:, j]
                    let mut dot_product = F::zero();
                    for k in 0..n {
                        dot_product += eigenvectors[[k, i]] * grad_vecs[[k, j]];
                    }

                    // Add contribution to gradient
                    for p in 0..n {
                        for q in 0..n {
                            let term = eigenvectors[[p, j]] * dot_product * eigenvectors[[q, i]];
                            grad[[p, q]] += factor * term;
                        }
                    }
                }
            }
        }
    }

    // For symmetric matrices, ensure gradient is symmetric
    let mut sym_grad = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym_grad[[i, j]] = (grad[[i, j]] + grad[[j, i]]) / F::from(2.0).unwrap();
        }
    }

    sym_grad
}

// Public API functions

/// Compute eigenvalues and eigenvectors of a symmetric matrix
pub fn eigh<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    // Extract eigenvalues
    let values = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SymmetricEigenExtractOp { component: 0 });

    // Extract eigenvectors
    let vectors = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SymmetricEigenExtractOp { component: 1 });

    (values, vectors)
}

/// Compute only the eigenvalues of a symmetric matrix (more efficient)
pub fn eigvalsh<'g, F: Float + ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();

    Tensor::builder(g)
        .append_input(matrix, false)
        .build(SymmetricEigenExtractOp { component: 0 })
}
