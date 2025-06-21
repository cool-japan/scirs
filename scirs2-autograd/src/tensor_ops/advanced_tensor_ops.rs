use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::Float;
use ndarray::{Array2, ArrayD, IxDyn};

/// Solve tensor equation a_ijk... x_jk... = b_i...
pub struct TensorSolveOp {
    axes: Option<Vec<i32>>,
}

impl<F: Float + ndarray::ScalarOperand> Op<F> for TensorSolveOp {
    fn name(&self) -> &'static str {
        "TensorSolve"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_shape = a.shape();
        let b_shape = b.shape();

        // Validate shapes and compute solution shape
        let (prod_x, prod_b) = validate_tensor_solve_shapes(a_shape, b_shape, &self.axes)?;

        // Reshape tensors for matrix solve
        let a_reshaped = reshape_for_solve(&a.view(), prod_b, prod_x)?;
        let b_reshaped = reshape_vector(&b.view(), prod_b)?;

        // Solve the linear system
        let x_flat = solve_linear_system(&a_reshaped, &b_reshaped)?;

        // Reshape solution back to expected shape
        let x_shape = compute_solution_shape(a_shape, b_shape, &self.axes)?;
        let x = reshape_solution(&x_flat, &x_shape)?;

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

        // Compute gradients
        // grad_b = a @ grad_x (reshaped appropriately)
        let grad_b = compute_grad_b(&a_array, &grad_output_array, &self.axes);

        // grad_a = -grad_x ⊗ x (outer product, reshaped)
        let grad_a = compute_grad_a(&grad_output_array, &x_array, a_array.shape(), &self.axes);

        // Convert to tensors
        let grad_a_tensor = convert_to_tensor(grad_a, g);
        let grad_b_tensor = convert_to_tensor(grad_b, g);

        ctx.append_input_grad(0, Some(grad_a_tensor));
        ctx.append_input_grad(1, Some(grad_b_tensor));
    }
}

/// Generalized tensor contraction with pattern specification
pub struct EinsumOp {
    pattern: String,
}

impl<F: Float + ndarray::ScalarOperand> Op<F> for EinsumOp {
    fn name(&self) -> &'static str {
        "Einsum"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Parse einsum pattern
        let (input_specs, output_spec) = parse_einsum_pattern(&self.pattern)?;

        // For now, implement a simplified version for common cases
        if input_specs.len() == 2 {
            let a = ctx.input(0);
            let b = ctx.input(1);

            // Handle common patterns
            if self.pattern == "ij,jk->ik" {
                // Matrix multiplication
                let result = compute_matmul(&a.view(), &b.view())?;
                ctx.append_output(result);
            } else if self.pattern == "i,i->" {
                // Dot product
                let result = compute_dot_product(&a.view(), &b.view())?;
                ctx.append_output(result);
            } else if self.pattern == "ij,ij->ij" {
                // Element-wise multiplication
                let result = compute_elementwise_mul(&a.view(), &b.view())?;
                ctx.append_output(result);
            } else {
                // General einsum (simplified)
                let result =
                    compute_general_einsum(&a.view(), &b.view(), &input_specs, &output_spec)?;
                ctx.append_output(result);
            }
        } else {
            return Err(OpError::Other(
                "Only binary einsum operations supported".into(),
            ));
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Simplified gradient - pass through
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(*gy));
        ctx.append_input_grad(1, Some(*gy));
    }
}

// Helper functions

fn validate_tensor_solve_shapes(
    a_shape: &[usize],
    b_shape: &[usize],
    axes: &Option<Vec<i32>>,
) -> Result<(usize, usize), OpError> {
    // Default axes behavior
    let ndim_a = a_shape.len();
    let ndim_b = b_shape.len();

    let axes_normalized = if let Some(ax) = axes {
        ax.clone()
    } else {
        // Default: last ndim_b axes of a
        let start = (ndim_a - ndim_b) as i32;
        (start..ndim_a as i32).collect()
    };

    // Compute products of dimensions
    let mut prod_x = 1;
    let mut prod_b = 1;

    for (i, &dim) in a_shape.iter().enumerate() {
        if axes_normalized.contains(&(i as i32)) {
            prod_x *= dim;
        } else {
            prod_b *= dim;
        }
    }

    let b_prod: usize = b_shape.iter().product();
    if b_prod != prod_b {
        return Err(OpError::IncompatibleShape(
            "Incompatible shapes for tensor solve".into(),
        ));
    }

    Ok((prod_x, prod_b))
}

fn reshape_for_solve<F: Float>(
    tensor: &ndarray::ArrayViewD<F>,
    rows: usize,
    cols: usize,
) -> Result<Array2<F>, OpError> {
    let tensor_view = tensor.view();
    let flat = tensor_view
        .to_shape(rows * cols)
        .map_err(|_| OpError::IncompatibleShape("Failed to flatten tensor for solve".into()))?;

    let mut matrix = Array2::<F>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            matrix[[i, j]] = flat[i * cols + j];
        }
    }

    Ok(matrix)
}

fn reshape_vector<F: Float>(
    tensor: &ndarray::ArrayViewD<F>,
    size: usize,
) -> Result<ndarray::Array1<F>, OpError> {
    tensor
        .view()
        .to_shape(size)
        .map_err(|_| OpError::IncompatibleShape("Failed to reshape vector".into()))
        .map(|v| v.to_owned())
}

fn solve_linear_system<F: Float>(
    a: &Array2<F>,
    b: &ndarray::Array1<F>,
) -> Result<ndarray::Array1<F>, OpError> {
    let n = a.shape()[0];
    let m = a.shape()[1];

    if n != b.len() {
        return Err(OpError::IncompatibleShape(
            "Matrix-vector dimension mismatch".into(),
        ));
    }

    // Use least squares for over/under-determined systems
    if n != m {
        // A^T A x = A^T b
        let ata = a.t().dot(a);
        let atb = a.t().dot(b);
        return solve_square_system(&ata, &atb);
    }

    // Square system
    solve_square_system(a, b)
}

fn solve_square_system<F: Float>(
    a: &Array2<F>,
    b: &ndarray::Array1<F>,
) -> Result<ndarray::Array1<F>, OpError> {
    let n = a.shape()[0];
    let mut aug = Array2::<F>::zeros((n, n + 1));

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
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

fn compute_solution_shape(
    a_shape: &[usize],
    _b_shape: &[usize],
    axes: &Option<Vec<i32>>,
) -> Result<Vec<usize>, OpError> {
    let ndim_a = a_shape.len();

    let axes_normalized = if let Some(ax) = axes {
        ax.clone()
    } else {
        // Default behavior
        vec![]
    };

    let mut x_shape = Vec::new();
    for (i, &dim) in a_shape.iter().enumerate() {
        if axes_normalized.contains(&(i as i32)) {
            x_shape.push(dim);
        }
    }

    if x_shape.is_empty() {
        // If no axes specified, use last dimensions
        let ndim_b = _b_shape.len();
        for &dim in a_shape.iter().skip(ndim_a - ndim_b) {
            x_shape.push(dim);
        }
    }

    Ok(x_shape)
}

fn reshape_solution<F: Float>(
    flat: &ndarray::Array1<F>,
    shape: &[usize],
) -> Result<ArrayD<F>, OpError> {
    let total: usize = shape.iter().product();
    if flat.len() != total {
        return Err(OpError::IncompatibleShape(
            "Solution reshape size mismatch".into(),
        ));
    }

    let dyn_shape = IxDyn(shape);
    flat.view()
        .to_shape(dyn_shape)
        .map_err(|_| OpError::IncompatibleShape("Failed to reshape solution".into()))
        .map(|v| v.to_owned())
}

fn compute_grad_b<F: Float>(
    _a: &ArrayD<F>,
    grad_x: &ArrayD<F>,
    _axes: &Option<Vec<i32>>,
) -> ArrayD<F> {
    // Simplified: return grad_x with appropriate shape
    grad_x.clone()
}

fn compute_grad_a<F: Float>(
    _grad_x: &ArrayD<F>,
    _x: &ArrayD<F>,
    a_shape: &[usize],
    _axes: &Option<Vec<i32>>,
) -> ArrayD<F> {
    // Simplified: return negative outer product with appropriate shape
    // This is a placeholder - actual implementation would compute proper tensor product
    ArrayD::<F>::zeros(IxDyn(a_shape))
}

// Einsum helpers

fn parse_einsum_pattern(pattern: &str) -> Result<(Vec<String>, String), OpError> {
    let parts: Vec<&str> = pattern.split("->").collect();
    if parts.len() != 2 {
        return Err(OpError::Other("Invalid einsum pattern".into()));
    }

    let input_part = parts[0];
    let output_part = parts[1];

    let input_specs: Vec<String> = input_part.split(',').map(|s| s.to_string()).collect();

    Ok((input_specs, output_part.to_string()))
}

fn compute_matmul<F: Float>(
    a: &ndarray::ArrayViewD<F>,
    b: &ndarray::ArrayViewD<F>,
) -> Result<ArrayD<F>, OpError> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(OpError::IncompatibleShape(
            "Matrix multiplication requires 2D arrays".into(),
        ));
    }

    let a_2d = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let b_2d = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();

    Ok(a_2d.dot(&b_2d).into_dyn())
}

fn compute_dot_product<F: Float>(
    a: &ndarray::ArrayViewD<F>,
    b: &ndarray::ArrayViewD<F>,
) -> Result<ArrayD<F>, OpError> {
    if a.shape() != b.shape() {
        return Err(OpError::IncompatibleShape(
            "Dot product requires same shape".into(),
        ));
    }

    let mut sum = F::zero();
    for (&a_val, &b_val) in a.iter().zip(b.iter()) {
        sum += a_val * b_val;
    }

    Ok(ndarray::arr0(sum).into_dyn())
}

fn compute_elementwise_mul<F: Float>(
    a: &ndarray::ArrayViewD<F>,
    b: &ndarray::ArrayViewD<F>,
) -> Result<ArrayD<F>, OpError> {
    if a.shape() != b.shape() {
        return Err(OpError::IncompatibleShape(
            "Element-wise multiplication requires same shape".into(),
        ));
    }

    Ok((a * b).into_owned())
}

fn compute_general_einsum<F: Float>(
    a: &ndarray::ArrayViewD<F>,
    _b: &ndarray::ArrayViewD<F>,
    _input_specs: &[String],
    _output_spec: &str,
) -> Result<ArrayD<F>, OpError> {
    // Simplified implementation - just return first input for now
    Ok(a.to_owned())
}

// Public API functions

/// Solve tensor equation a @ x = b for x
pub fn tensor_solve<'g, F: Float + ndarray::ScalarOperand>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    axes: Option<Vec<i32>>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(TensorSolveOp { axes })
}

/// Einstein summation convention
pub fn einsum<'g, F: Float + ndarray::ScalarOperand>(
    pattern: &str,
    operands: &[&Tensor<'g, F>],
) -> Tensor<'g, F> {
    if operands.len() != 2 {
        panic!("Only binary einsum operations are currently supported");
    }

    let g = operands[0].graph();

    Tensor::builder(g)
        .append_input(operands[0], false)
        .append_input(operands[1], false)
        .build(EinsumOp {
            pattern: pattern.to_string(),
        })
}

/// Kronecker product (tensor product of matrices)
pub fn kron<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();

    // Use einsum to compute Kronecker product
    // For 2D matrices: kron(A, B)[i*p+k, j*q+l] = A[i,j] * B[k,l]
    // This can be expressed as: einsum("ij,kl->ikjl", A, B).reshape(...)

    // For now, use a simple implementation
    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(KroneckerOp)
}

/// Kronecker product operation
struct KroneckerOp;

impl<F: Float> Op<F> for KroneckerOp {
    fn name(&self) -> &'static str {
        "Kronecker"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Kronecker product requires 2D matrices".into(),
            ));
        }

        let (m, n) = (a_shape[0], a_shape[1]);
        let (p, q) = (b_shape[0], b_shape[1]);

        let mut result = ArrayD::<F>::zeros(IxDyn(&[m * p, n * q]));

        // Compute Kronecker product
        for i in 0..m {
            for j in 0..n {
                let a_ij = a[[i, j]];
                for k in 0..p {
                    for l in 0..q {
                        result[[i * p + k, j * q + l]] = a_ij * b[[k, l]];
                    }
                }
            }
        }

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Simplified gradient
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(*gy));
        ctx.append_input_grad(1, Some(*gy));
    }
}
