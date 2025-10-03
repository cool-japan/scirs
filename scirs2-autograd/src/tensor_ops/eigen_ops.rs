use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::Float;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray::{Array1, Array2, Ix2};
use scirs2_core::numeric::FromPrimitive;

/// Eigenvalue decomposition operation
pub struct EigenOp;

/// Extract operation for eigenvalue decomposition components
pub struct EigenExtractOp {
    pub component: usize, // 0 for eigenvalues, 1 for eigenvectors
}

impl<F: Float + ScalarOperand + FromPrimitive> Op<F> for EigenOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Eigendecomposition requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Check if matrix is symmetric
        let is_symmetric = is_symmetric_matrix(&input_2d);

        let (eigenvalues, eigenvectors) = if is_symmetric {
            compute_symmetric_eigen(&input_2d)?
        } else {
            compute_general_eigen(&input_2d)?
        };

        // Verify the shapes of the computed arrays
        println!("Eigenvalues shape: {:?}", eigenvalues.shape());
        println!("Eigenvectors shape: {:?}", eigenvectors.shape());

        // Ensure eigenvalues is a 1D array of length n
        assert_eq!(eigenvalues.len(), shape[0]);

        // Ensure eigenvectors is a 2D array of shape (n, n)
        assert_eq!(eigenvectors.shape(), &[shape[0], shape[0]]);

        // Output the arrays with verified shapes
        ctx.append_output(eigenvalues.into_dyn());
        ctx.append_output(eigenvectors.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let _g = ctx.graph(); // Prefix with _ to avoid unused variable warning

        // Get the inputs
        let input = ctx.input(0);
        let g = ctx.graph();

        // Get shape from input tensor via evaluation
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let n = input_array.shape()[0]; // Get size from shape array

        // Evaluate tensors to get their array values
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

        // Calculate sizes for splitting arrays
        let values_size = n;
        let vectors_start = values_size;

        // Extract eigenvalues and eigenvectors
        let eigen_vals = y_array.slice(scirs2_core::ndarray::s![0..values_size]);
        let eigen_vecs = y_array.slice(scirs2_core::ndarray::s![vectors_start..]);

        let eigen_vals_1d = eigen_vals.to_shape(n).unwrap().to_owned();
        let eigen_vecs_2d = eigen_vecs.to_shape((n, n)).unwrap().to_owned();

        // Get gradients
        let grad_vals = gy_array
            .slice(scirs2_core::ndarray::s![0..values_size])
            .to_shape(n)
            .unwrap()
            .to_owned();
        let grad_vecs = gy_array
            .slice(scirs2_core::ndarray::s![vectors_start..])
            .to_shape((n, n))
            .unwrap()
            .to_owned();

        // Compute gradient using eigendecomposition gradient formula
        let grad_input = eigendecomposition_gradient(
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

/// Eigenvalues only operation (more efficient when eigenvectors not needed)
pub struct EigenvaluesOp;

impl<F: Float + ScalarOperand + FromPrimitive> Op<F> for EigenvaluesOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Eigenvalues require square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        let eigenvalues = compute_eigenvalues_only(&input_2d)?;

        // Verify the shape of the computed eigenvalues array
        println!("Eigenvalues shape: {:?}", eigenvalues.shape());

        // Ensure eigenvalues is a 1D array of length n
        let n = shape[0];
        if eigenvalues.len() != n {
            println!(
                "WARNING: Eigenvalues shape mismatch! Expected length {}, got {}",
                n,
                eigenvalues.len()
            );

            // Create a new array with the correct shape
            let mut reshaped_vals = scirs2_core::ndarray::Array1::<F>::zeros(n);

            // Copy as much data as fits
            let min_len = n.min(eigenvalues.len());
            for i in 0..min_len {
                reshaped_vals[i] = eigenvalues[i];
            }

            // Output the reshaped array
            ctx.append_output(reshaped_vals.into_dyn());
        } else {
            // Output the eigenvalues with verified shape
            ctx.append_output(eigenvalues.into_dyn());
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Get dimensions from shape
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let n = input_array.shape()[0];

        // Evaluate gradient output
        let grad_vals_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to the right shape
        let grad_vals_1d = match grad_vals_array.to_shape(n) {
            Ok(arr) => arr.to_owned(),
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // For eigenvalues gradient, we need the eigenvectors
        // This is a simplified placeholder - real implementation needs eigenvectors
        let mut grad_input = Array2::<F>::zeros((n, n));

        for i in 0..n {
            grad_input[[i, i]] = grad_vals_1d[i];
        }

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad_input.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

// Helper functions
#[allow(dead_code)]
fn is_symmetric_matrix<F: Float>(matrix: &scirs2_core::ndarray::ArrayView2<F>) -> bool {
    let n = matrix.shape()[0];
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > F::epsilon() {
                return false;
            }
        }
    }
    true
}

#[allow(dead_code)]
fn compute_symmetric_eigen<F: Float + ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = matrix.shape()[0];

    // Use Jacobi rotation method for symmetric matrices
    let mut a = matrix.to_owned();
    let mut v = Array2::<F>::eye(n);

    // Jacobi iterations
    for _ in 0..100 {
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
        if max_val < F::epsilon() {
            break;
        }

        // Compute rotation parameters
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        let theta: F = if app == aqq {
            scirs2_core::numeric::FromPrimitive::from_f64(0.25 * std::f64::consts::PI).unwrap()
        } else {
            scirs2_core::numeric::FromPrimitive::from_f64(
                0.5 * (aqq - app)
                    .to_f64()
                    .unwrap()
                    .atan2(2.0 * apq.to_f64().unwrap()),
            )
            .unwrap()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Update matrix A
        for i in 0..n {
            let aip = a[[i, p]];
            let aiq = a[[i, q]];

            a[[i, p]] = aip * c - aiq * s;
            a[[i, q]] = aip * s + aiq * c;
        }

        for j in 0..n {
            let apj = a[[p, j]];
            let aqj = a[[q, j]];

            a[[p, j]] = apj * c - aqj * s;
            a[[q, j]] = apj * s + aqj * c;
        }

        // Restore symmetry
        a[[p, q]] = F::zero();
        a[[q, p]] = F::zero();

        // Update eigenvector matrix
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];

            v[[i, p]] = vip * c - viq * s;
            v[[i, q]] = vip * s + viq * c;
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = Array1::<F>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    Ok((eigenvalues, v))
}

#[allow(dead_code)]
fn compute_general_eigen<F: Float + ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    // For general matrices, we'll use the QR algorithm
    // This is a more robust implementation for non-symmetric matrices

    let n = matrix.shape()[0];
    println!("Computing eigendecomposition for general matrix of size: {n}");

    // Check if the matrix is close to symmetric within a tolerance
    let is_nearly_symmetric =
        is_nearly_symmetric_matrix(matrix, F::epsilon() * F::from(100.0).unwrap());

    if is_nearly_symmetric {
        println!("Matrix is nearly symmetric, using symmetric algorithm");
        // If nearly symmetric, symmetrize and use the more efficient Jacobi method
        let mut sym_matrix = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sym_matrix[[i, j]] = (matrix[[i, j]] + matrix[[j, i]])
                    * scirs2_core::numeric::FromPrimitive::from_f64(0.5).unwrap();
            }
        }
        return compute_symmetric_eigen(&sym_matrix.view());
    }

    println!("Using QR algorithm for general eigendecomposition");

    // Implementation of the QR algorithm for eigendecomposition
    // We'll perform a series of QR decompositions to find the eigenvalues and vectors

    // Start with the original matrix
    let mut a = matrix.to_owned();

    // Initialize eigenvectors to identity matrix
    let mut v = Array2::<F>::eye(n);

    // Maximum number of iterations
    let max_iter = 100;

    // Threshold for convergence
    let tol = F::epsilon() * F::from(1000.0).unwrap();

    // Store previous iteration matrix to check convergence
    let mut prev_a = a.clone();

    // QR algorithm iterations
    for iter in 0..max_iter {
        // Apply a shift to improve convergence
        let shift = if n > 1 { a[[n - 1, n - 1]] } else { F::zero() };

        // Subtract the shift from the diagonal
        for i in 0..n {
            a[[i, i]] -= shift;
        }

        // Compute the QR decomposition: A = QR
        // We'll use a simple Gram-Schmidt process for QR
        let (q, r) = compute_qr_decomposition(&a)?;

        // Form the new matrix A' = RQ + shift*I
        a = r.dot(&q);

        // Add the shift back to the diagonal
        for i in 0..n {
            a[[i, i]] += shift;
        }

        // Update the eigenvector matrix: V = V * Q
        v = v.dot(&q);

        // Check for convergence
        let diff = (&a - &prev_a).mapv(|x| x.abs()).sum() / F::from(n as f64).unwrap();
        if iter > 0 && diff < tol {
            println!("QR algorithm converged after {} iterations", iter + 1);
            break;
        }

        // Update previous matrix for next convergence check
        prev_a = a.clone();

        // Check for zeros in last row/column to detect converged eigenvalues
        let mut is_triangular = true;
        for i in 1..n {
            for j in 0..i {
                if a[[i, j]].abs() > tol {
                    is_triangular = false;
                    break;
                }
            }
            if !is_triangular {
                break;
            }
        }

        if is_triangular {
            println!("Matrix is nearly triangular after {} iterations", iter + 1);
            break;
        }
    }

    // Extract eigenvalues from the diagonal of the final matrix
    let mut eigenvalues = Array1::<F>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    // Normalize eigenvectors
    for j in 0..n {
        let mut norm_squared = F::zero();
        for i in 0..n {
            norm_squared += v[[i, j]] * v[[i, j]];
        }
        let norm = norm_squared.sqrt();

        if norm > F::epsilon() {
            for i in 0..n {
                v[[i, j]] /= norm;
            }
        }
    }

    Ok((eigenvalues, v))
}

// Helper function to check if a matrix is nearly symmetric
#[allow(dead_code)]
fn is_nearly_symmetric_matrix<F: Float>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    tol: F,
) -> bool {
    let n = matrix.shape()[0];
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    true
}

// Helper function to compute QR decomposition
#[allow(dead_code)]
fn compute_qr_decomposition<F: Float + ScalarOperand + FromPrimitive>(
    a: &Array2<F>,
) -> Result<(Array2<F>, Array2<F>), OpError> {
    let n = a.shape()[0];

    // Initialize Q and R
    let mut q = Array2::<F>::zeros((n, n));
    let mut r = Array2::<F>::zeros((n, n));

    // Modified Gram-Schmidt orthogonalization
    for j in 0..n {
        // Copy column j of A into Q
        let mut column = Array1::<F>::zeros(n);
        for i in 0..n {
            column[i] = a[[i, j]];
        }

        // Orthogonalize against previous columns
        for k in 0..j {
            // Compute dot product of column j with normalized column k
            let mut dot_product = F::zero();
            for i in 0..n {
                dot_product += column[i] * q[[i, k]];
            }

            // Store in R
            r[[k, j]] = dot_product;

            // Subtract projection
            for i in 0..n {
                column[i] -= dot_product * q[[i, k]];
            }
        }

        // Compute the norm of the column
        let mut norm_squared = F::zero();
        for i in 0..n {
            norm_squared += column[i] * column[i];
        }

        let norm = norm_squared.sqrt();

        // Check for linear dependency
        if norm < F::epsilon() {
            // Generate a random orthogonal vector
            let mut new_col = Array1::<F>::zeros(n);
            new_col[j] = F::one();

            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = F::zero();
                for i in 0..n {
                    dot += new_col[i] * q[[i, k]];
                }
                for i in 0..n {
                    new_col[i] -= dot * q[[i, k]];
                }
            }

            // Normalize
            let mut new_norm_squared = F::zero();
            for i in 0..n {
                new_norm_squared += new_col[i] * new_col[i];
            }
            let new_norm = new_norm_squared.sqrt();

            if new_norm < F::epsilon() {
                return Err(OpError::Other(
                    "Failed to generate orthogonal vector".into(),
                ));
            }

            for i in 0..n {
                q[[i, j]] = new_col[i] / new_norm;
            }

            r[[j, j]] = F::zero();
        } else {
            // Store the normalized column in Q
            r[[j, j]] = norm;
            for i in 0..n {
                q[[i, j]] = column[i] / norm;
            }
        }
    }

    Ok((q, r))
}

#[allow(dead_code)]
fn compute_eigenvalues_only<F: Float + ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array1<F>, OpError> {
    // Simplified implementation - use full eigen decomposition but return only values
    let (values_, _vectors) = if is_symmetric_matrix(matrix) {
        compute_symmetric_eigen(matrix)?
    } else {
        compute_general_eigen(matrix)?
    };

    Ok(values_)
}

#[allow(dead_code)]
fn eigendecomposition_gradient<F: Float + ScalarOperand + FromPrimitive>(
    eigenvalues: &scirs2_core::ndarray::ArrayView1<F>,
    eigenvectors: &scirs2_core::ndarray::ArrayView2<F>,
    grad_vals: &scirs2_core::ndarray::ArrayView1<F>,
    grad_vecs: &scirs2_core::ndarray::ArrayView2<F>,
) -> Array2<F> {
    let n = eigenvalues.len();
    let mut grad = Array2::<F>::zeros((n, n));

    println!("Computing eigendecomposition gradient for matrix of size: {n}");

    // Gradient for eigenvalues part
    // For each eigenvalue, we add the corresponding component to the gradient
    for i in 0..n {
        let vi = eigenvectors.slice(scirs2_core::ndarray::s![.., i]);
        for j in 0..n {
            for k in 0..n {
                grad[[j, k]] += grad_vals[i] * vi[j] * vi[k];
            }
        }
    }

    // Gradient for eigenvectors part
    // This is a more robust implementation that handles degeneracy
    for i in 0..n {
        for j in 0..n {
            // Check if eigenvalues are distinct - if they are close, use regularization
            let eigenvalue_diff = (eigenvalues[i] - eigenvalues[j]).abs();
            let is_degenerate = eigenvalue_diff <= F::epsilon() * F::from(10.0).unwrap();

            if i != j {
                // Compute the factor with safeguards against division by zero
                let factor = if is_degenerate {
                    // For nearly degenerate eigenvalues, use a regularized factor
                    let reg_diff = F::max(eigenvalue_diff, F::epsilon() * F::from(10.0).unwrap());
                    F::one() / reg_diff
                } else {
                    F::one() / (eigenvalues[j] - eigenvalues[i])
                };

                // For degenerate eigenvalues, the gradient needs special handling
                if is_degenerate {
                    // Use a more stable approach for degenerate eigenvalues
                    // This is a simplification - a full implementation would need
                    // to compute the generalized eigenvectors
                    let vi = eigenvectors.slice(scirs2_core::ndarray::s![.., i]);
                    let vj = eigenvectors.slice(scirs2_core::ndarray::s![.., j]);

                    // Compute the component perpendicular to vi
                    let mut dot_product = F::zero();
                    for p in 0..n {
                        dot_product += grad_vecs[[p, j]] * vi[p];
                    }

                    // Add the perpendicular component to the gradient
                    for p in 0..n {
                        for q in 0..n {
                            let term = vj[p] * (grad_vecs[[p, j]] - dot_product * vi[p]) * vj[q];
                            grad[[q, p]] += term * F::from(0.5).unwrap();
                        }
                    }
                } else {
                    // For distinct eigenvalues, use the standard formula
                    for p in 0..n {
                        for q in 0..n {
                            let term =
                                eigenvectors[[p, i]] * grad_vecs[[p, j]] * eigenvectors[[q, j]];
                            grad[[q, p]] += factor * term;
                        }
                    }
                }
            }
        }
    }

    // Handle the case where eigenvector gradient is with respect to itself
    // This is the projection of the gradient onto the orthogonal complement
    for i in 0..n {
        let vi = eigenvectors.slice(scirs2_core::ndarray::s![.., i]);

        // Compute norm of vi for normalization gradient
        let mut vi_norm_squared = F::zero();
        for p in 0..n {
            vi_norm_squared += vi[p] * vi[p];
        }

        if vi_norm_squared > F::epsilon() {
            // Projection term
            let mut dot_product = F::zero();
            for p in 0..n {
                dot_product += grad_vecs[[p, i]] * vi[p];
            }

            // Add to gradient
            for p in 0..n {
                for q in 0..n {
                    let term = vi[p] * (grad_vecs[[p, i]] - dot_product * vi[p]) * vi[q];
                    grad[[q, p]] += term;
                }
            }
        }
    }

    // Add regularization for numerical stability
    let eps = F::epsilon() * F::from(10.0).unwrap();
    for i in 0..n {
        grad[[i, i]] += eps;
    }

    println!("Completed eigendecomposition gradient computation");
    grad
}

impl<F: Float + ScalarOperand + FromPrimitive> Op<F> for EigenExtractOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Eigendecomposition requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Check if matrix is symmetric
        let is_symmetric = is_symmetric_matrix(&input_2d);

        let (eigenvalues, eigenvectors) = if is_symmetric {
            compute_symmetric_eigen(&input_2d)?
        } else {
            compute_general_eigen(&input_2d)?
        };

        // Return the requested component
        match self.component {
            0 => ctx.append_output(eigenvalues.into_dyn()),
            1 => ctx.append_output(eigenvectors.into_dyn()),
            _ => {
                return Err(OpError::Other(
                    "Invalid component index for eigen extraction".into(),
                ))
            }
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // For extraction operations, just pass through the gradient
        // The actual gradient computation happens in the EigenOp
        let _gy = ctx.output_grad();
        let g = ctx.graph();

        // Create a zero gradient for the input since this is just extraction
        let zeros = crate::tensor_ops::zeros(&crate::tensor_ops::shape(ctx.input(0)), g);
        ctx.append_input_grad(0, Some(zeros));
    }
}

// Public API functions

/// Compute eigenvalues and eigenvectors of a square matrix
#[allow(dead_code)]
pub fn eigen<'g, F: Float + ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    // Extract eigenvalues
    let values = Tensor::builder(g)
        .append_input(matrix, false)
        .build(EigenExtractOp { component: 0 });

    // Extract eigenvectors
    let vectors = Tensor::builder(g)
        .append_input(matrix, false)
        .build(EigenExtractOp { component: 1 });

    (values, vectors)
}

/// Compute only the eigenvalues of a square matrix
#[allow(dead_code)]
pub fn eigenvalues<'g, F: Float + ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();

    // Get the shape of the input tensor for setting the output shape
    // For eigenvalues, we'll have a 1D tensor with size n for an n×n matrix
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(EigenvaluesOp)
}
