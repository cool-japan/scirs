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

        // TODO: Replace with scirs2-core linear algebra when available
        // For now, return an error as Cholesky decomposition requires BLAS
        return Err(OpError::Other(
            "Cholesky decomposition not yet implemented - waiting for scirs2-core linear algebra module".to_string(),
        ));

        #[allow(unreachable_code)]
        {
            // When implemented, we'll need to create a mutable copy and process it
            let mut matrix_data = matrix.to_owned();

            // The result is stored in-place in matrix_data (lower triangular part)
            // Zero out the upper triangular part to get a clean L matrix
            for i in 0..shape[0] {
                for j in (i + 1)..shape[1] {
                    matrix_data[[i, j]] = F::zero();
                }
            }

            ctx.append_output(matrix_data.into_dyn());
            Ok(())
        } // End unreachable block
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let g = ctx.graph();

        println!("Computing gradient for Cholesky decomposition");

        // Get arrays for gradient computation
        let y_array = match y.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate output tensor for Cholesky gradient");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to evaluate gradient tensor for Cholesky gradient");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D arrays
        let l = match y_array.into_dimensionality::<scirs2_core::ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to convert Cholesky output to 2D array");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let gy_2d = match gy_array.into_dimensionality::<scirs2_core::ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(_) => {
                println!("Failed to convert Cholesky gradient to 2D array");
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let n = l.shape()[0];
        println!("Cholesky gradient computation for matrix of size: {n}");

        // Initialize gradient matrix
        let mut grad = Array2::<F>::zeros((n, n));

        // Create a view of L that's properly triangular (ensuring zeros above diagonal)
        let mut l_clean = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                l_clean[[i, j]] = l[[i, j]];
            }
        }

        // Compute the gradient using the chain rule for Cholesky decomposition
        // Based on the formula: dA = L * (dL * L^T + L * dL^T) * L^T
        // We need to solve for dL given dA
        // We use a forward substitution approach

        // First, mask gradients to be lower triangular (same shape as L)
        let mut d_l = Array2::<F>::zeros((n, n));

        // The diagonal elements have a special formula
        for i in 0..n {
            d_l[[i, i]] = gy_2d[[i, i]] / (F::from(2.0).unwrap() * l_clean[[i, i]]);
        }

        // Process row by row
        for i in 1..n {
            for j in 0..i {
                // Compute the right-hand side for the equation
                let mut rhs = gy_2d[[i, j]];

                // Subtract the effect of already computed elements
                for k in 0..j {
                    rhs = rhs - d_l[[i, k]] * l_clean[[j, k]] - l_clean[[i, k]] * d_l[[j, k]];
                }

                // Solve for d_l[i,j]
                d_l[[i, j]] = rhs / l_clean[[j, j]];
            }
        }

        // Convert to gradient of A by making it symmetric
        // dA/dL = 0.5 * (dL + dL^T) for non-diagonal elements
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal elements
                    grad[[i, i]] = d_l[[i, i]] * F::from(2.0).unwrap();
                } else {
                    // Off-diagonal elements, symmetrize
                    let val = if i > j { d_l[[i, j]] } else { d_l[[j, i]] };
                    grad[[i, j]] = val;
                    grad[[j, i]] = val;
                }
            }
        }

        // Add regularization for numerical stability
        let eps = F::epsilon() * F::from(10.0).unwrap();
        for i in 0..n {
            grad[[i, i]] += eps;
        }

        println!("Completed Cholesky gradient computation");

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
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
        let half = F::from(0.5).unwrap();

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                symmetric[[i, j]] = (matrix[[i, j]] + matrix[[j, i]]) * half;
            }
        }

        ctx.append_output(symmetric.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();

        // Get array for gradient computation
        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D array
        let gy_2d = match gy_array.into_dimensionality::<scirs2_core::ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Gradient of symmetrize: (dY + dY^T) / 2
        let mut grad = Array2::<F>::zeros(gy_2d.dim());
        let half = F::from(0.5).unwrap();

        for i in 0..gy_2d.shape()[0] {
            for j in 0..gy_2d.shape()[1] {
                grad[[i, j]] = (gy_2d[[i, j]] + gy_2d[[j, i]]) * half;
            }
        }

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
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

        println!(
            "Computing lower triangular with diagonal={}, input shape: {:?}",
            self.diagonal, shape
        );

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

        println!("Processing lower triangular matrix: {rows} rows x {cols} columns");

        // Zero out elements above the specified diagonal
        for i in 0..rows {
            for j in 0..cols {
                if j as i32 > i as i32 - self.diagonal {
                    lower[[i, j]] = F::zero();
                }
            }
        }

        // Verify the output shape
        println!("Lower triangular result shape: {:?}", lower.shape());

        ctx.append_output(lower.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();

        // Get array for gradient computation
        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D array
        let gy_2d = match gy_array.into_dimensionality::<scirs2_core::ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let mut grad = gy_2d.to_owned();
        let (rows, cols) = (grad.shape()[0], grad.shape()[1]);

        // Zero out gradients for elements that were zeroed in forward pass
        for i in 0..rows {
            for j in 0..cols {
                if j as i32 > i as i32 - self.diagonal {
                    grad[[i, j]] = F::zero();
                }
            }
        }

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
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

        println!(
            "Computing upper triangular with diagonal={}, input shape: {:?}",
            self.diagonal, shape
        );

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

        println!("Processing upper triangular matrix: {rows} rows x {cols} columns");

        // Zero out elements below the specified diagonal
        for i in 0..rows {
            for j in 0..cols {
                if (j as i32) < (i as i32 + self.diagonal) {
                    upper[[i, j]] = F::zero();
                }
            }
        }

        // Verify the output shape
        println!("Upper triangular result shape: {:?}", upper.shape());

        ctx.append_output(upper.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();

        // Get array for gradient computation
        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D array
        let gy_2d = match gy_array.into_dimensionality::<scirs2_core::ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let mut grad = gy_2d.to_owned();
        let (rows, cols) = (grad.shape()[0], grad.shape()[1]);

        // Zero out gradients for elements that were zeroed in forward pass
        for i in 0..rows {
            for j in 0..cols {
                if (j as i32) < (i as i32 + self.diagonal) {
                    grad[[i, j]] = F::zero();
                }
            }
        }

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
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

        println!(
            "Computing band matrix with lower={}, upper={}, input shape: {:?}",
            self.lower, self.upper, shape
        );

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

        println!("Processing band matrix: {rows} rows x {cols} columns");

        // Zero out elements outside the band
        for i in 0..rows {
            for j in 0..cols {
                let diag_offset = j as i32 - i as i32;
                if diag_offset < -self.lower || diag_offset > self.upper {
                    band[[i, j]] = F::zero();
                }
            }
        }

        // Verify the output shape
        println!("Band matrix result shape: {:?}", band.shape());

        ctx.append_output(band.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();

        // Get array for gradient computation
        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Convert to 2D array
        let gy_2d = match gy_array.into_dimensionality::<scirs2_core::ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let mut grad = gy_2d.to_owned();
        let (rows, cols) = (grad.shape()[0], grad.shape()[1]);

        // Zero out gradients for elements outside the band
        for i in 0..rows {
            for j in 0..cols {
                let diag_offset = j as i32 - i as i32;
                if diag_offset < -self.lower || diag_offset > self.upper {
                    grad[[i, j]] = F::zero();
                }
            }
        }

        // Convert gradient to tensor and append
        let grad_tensor = convert_to_tensor(grad.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
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
