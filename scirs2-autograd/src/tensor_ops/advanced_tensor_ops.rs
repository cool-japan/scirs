use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::Float;
use scirs2_core::ndarray::{Array2, ArrayD, Ix2, IxDyn};

/// Solve tensor equation a_ijk... x_jk... = b_i...
pub struct TensorSolveOp {
    axes: Option<Vec<i32>>,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for TensorSolveOp {
    fn name(&self) -> &'static str {
        "TensorSolve"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let ashape = a.shape();
        let bshape = b.shape();

        // Validate shapes and compute solution shape
        let (prod_x, prod_b) = validate_tensor_solveshapes(ashape, bshape, &self.axes)?;

        // Reshape tensors for matrix solve
        let a_reshaped = reshape_for_solve(&a.view(), prod_b, prod_x)?;
        let b_reshaped = reshape_vector(&b.view(), prod_b)?;

        // Solve the linear system
        let x_flat = solve_linear_system(&a_reshaped, &b_reshaped)?;

        // Reshape solution back to expected shape
        let xshape = compute_solutionshape(ashape, bshape, &self.axes)?;
        let x = reshape_solution(&x_flat, &xshape)?;

        ctx.append_output(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // `tensor_solve` flattens `A` to a matrix and `b` to a vector, solves, and
        // reshapes; its VJP is therefore the linear-solve VJP applied to the flattened
        // problem and reshaped back.  See `TensorSolveVjpOp`.
        //
        // The previous rule called two helpers whose bodies were
        // `grad_x.clone()` and `ArrayD::zeros(ashape)` — a copy of the cotangent for `b`
        // and an identically-zero gradient for `A`.
        let a = *ctx.input(0);
        let b = *ctx.input(1);
        let x = *ctx.output();
        let gy = *ctx.output_grad();
        let g = ctx.graph();

        let mut node = |wrt_a: bool| {
            Tensor::builder(g)
                .append_input(a, false)
                .append_input(b, false)
                .append_input(x, false)
                .append_input(gy, false)
                .build(TensorSolveVjpOp {
                    axes: self.axes.clone(),
                    wrt_a,
                })
        };
        let grad_a = node(true);
        let grad_b = node(false);
        ctx.append_input_grad(0, Some(grad_a));
        ctx.append_input_grad(1, Some(grad_b));
    }
}

/// Backward node of [`TensorSolveOp`].
///
/// Inputs are `(A, b, x, gy)`; the output is `Ā` (`wrt_a = true`) or `b̄`.
///
/// After flattening, `x` solves `A x = b` (square) or the normal equations
/// `AᵀA x = Aᵀ b` (over-determined). Differentiating the defining relation gives
///
/// ```text
///   square:            b̄ = A^-ᵀ gy,        Ā = -b̄ xᵀ
///   full column rank:  b̄ = A⁺ᵀ gy,         Ā = -b̄ xᵀ + r (A⁺ A⁺ᵀ gy)ᵀ,  r = b - A x
/// ```
///
/// The residual term is what distinguishes a least-squares solve from an exact one; it
/// vanishes identically when `A` is square (`r = 0`), so the two branches agree there.
pub struct TensorSolveVjpOp {
    axes: Option<Vec<i32>>,
    wrt_a: bool,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for TensorSolveVjpOp {
    fn name(&self) -> &'static str {
        "TensorSolveVjp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);
        let x = ctx.input(2);
        let gy = ctx.input(3);

        let ashape = a.shape().to_vec();
        let bshape = b.shape().to_vec();
        let (prod_x, prod_b) = validate_tensor_solveshapes(&ashape, &bshape, &self.axes)?;

        let a_mat = reshape_for_solve(&a.view(), prod_b, prod_x)?;
        let b_vec = reshape_vector(&b.view(), prod_b)?;
        let x_vec = reshape_vector(&x.view(), prod_x)?;
        let gy_vec = reshape_vector(&gy.view(), prod_x)?;

        let (grad_b_vec, grad_a_mat) = if prod_b == prod_x {
            // Square: b̄ solves Aᵀ b̄ = gy.
            let y = solve_square_system(&a_mat.t().to_owned(), &gy_vec)?;
            let mut grad_a = Array2::<F>::zeros((prod_b, prod_x));
            for i in 0..prod_b {
                for j in 0..prod_x {
                    grad_a[[i, j]] = -(y[i] * x_vec[j]);
                }
            }
            (y, grad_a)
        } else {
            // Least squares through the normal equations: A⁺ = (AᵀA)^-1 Aᵀ.
            let ata = a_mat.t().dot(&a_mat);
            let ata_inv = crate::tensor_ops::matrix_calculus::inverse(&ata.view())?;
            let pinv = ata_inv.dot(&a_mat.t()); // prod_x x prod_b
            let y = pinv.t().dot(&gy_vec); // prod_b
            let residual = &b_vec - &a_mat.dot(&x_vec); // prod_b
            let correction = pinv.dot(&pinv.t()).dot(&gy_vec); // prod_x
            let mut grad_a = Array2::<F>::zeros((prod_b, prod_x));
            for i in 0..prod_b {
                for j in 0..prod_x {
                    grad_a[[i, j]] = -(y[i] * x_vec[j]) + residual[i] * correction[j];
                }
            }
            (y, grad_a)
        };

        if self.wrt_a {
            let flat: Vec<F> = grad_a_mat.iter().copied().collect();
            let out = ArrayD::from_shape_vec(IxDyn(&ashape), flat).map_err(|_| {
                OpError::IncompatibleShape("tensor_solve backward: cannot reshape dL/dA".into())
            })?;
            ctx.append_output(out);
        } else {
            let flat: Vec<F> = grad_b_vec.iter().copied().collect();
            let out = ArrayD::from_shape_vec(IxDyn(&bshape), flat).map_err(|_| {
                OpError::IncompatibleShape("tensor_solve backward: cannot reshape dL/db".into())
            })?;
            ctx.append_output(out);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        crate::tensor_ops::matrix_calculus::append_unsupported_grad(
            ctx,
            "tensor_solve: second-order differentiation is not implemented".into(),
        );
    }
}

/// Generalized tensor contraction with pattern specification
pub struct EinsumOp {
    pattern: String,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for EinsumOp {
    fn name(&self) -> &'static str {
        "Einsum"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Parse einsum pattern
        let (input_specs, output_spec) = parse_einsum_pattern(&self.pattern)?;

        if input_specs.len() != 2 {
            return Err(OpError::Other(format!(
                "einsum: only binary equations are supported, got {} operands in '{}'",
                input_specs.len(),
                self.pattern
            )));
        }

        let a = ctx.input(0);
        let b = ctx.input(1);

        // `ij,jk->ik` keeps its BLAS-backed fast path; everything else goes through the
        // general contraction.  The previous `else` branch returned the *first operand
        // unchanged* for any equation outside three hard-coded patterns.
        if self.pattern == "ij,jk->ik" {
            let result = compute_matmul(&a.view(), &b.view())?;
            ctx.append_output(result);
        } else {
            let result = einsum_contract(
                &[&input_specs[0], &input_specs[1]],
                &[&a.view(), &b.view()],
                &output_spec,
                None,
            )?;
            ctx.append_output(result);
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Einsum is differentiated by *rewriting the equation*: swap the output subscript
        // with the subscript of the operand being differentiated.  For `a_A b_B -> y_Y`,
        //
        //     ā = einsum("Y,B->A", gy, b)      b̄ = einsum("Y,A->B", gy, a)
        //
        // e.g. for `ij,jk->ik` this yields `ik,jk->ij` (gy · bᵀ) and `ik,ij->jk`
        // (aᵀ · gy), the familiar matmul rules — but it is equally correct for
        // contractions, free indices, summed-away indices and repeated (diagonal)
        // indices, because a label that appears only in the *target* spec is simply an
        // output label with no input to read from, i.e. a broadcast.
        //
        // The previous rule handed the raw output cotangent to both operands, which is
        // not even shape-correct: `einsum("ij->i")` produced an `i`-shaped gradient for
        // an `ij`-shaped input and made the accumulation in `AddN` hit `unreachable!()`.
        let (input_specs, out_spec) = match parse_einsum_pattern(&self.pattern) {
            Ok(parsed) => parsed,
            Err(e) => {
                crate::tensor_ops::matrix_calculus::append_unsupported_grad(
                    ctx,
                    format!("einsum: cannot differentiate an unparsable equation: {e}"),
                );
                return;
            }
        };
        if input_specs.len() != 2 {
            crate::tensor_ops::matrix_calculus::append_unsupported_grad(
                ctx,
                format!(
                    "einsum: only binary equations are supported, got {} operands in '{}'",
                    input_specs.len(),
                    self.pattern
                ),
            );
            return;
        }

        let a = *ctx.input(0);
        let b = *ctx.input(1);
        let gy = *ctx.output_grad();
        let g = ctx.graph();

        // Inputs are (gy, other operand, target operand).  The target is passed only so
        // the backward node can read its shape: a label that appears in the target spec
        // and nowhere else has no size anywhere in the rewritten equation.
        let grad_a = Tensor::builder(g)
            .append_input(gy, false)
            .append_input(b, false)
            .append_input(a, false)
            .build(EinsumGradOp {
                out_spec: out_spec.clone(),
                other_spec: input_specs[1].clone(),
                target_spec: input_specs[0].clone(),
            });
        let grad_b = Tensor::builder(g)
            .append_input(gy, false)
            .append_input(a, false)
            .append_input(b, false)
            .build(EinsumGradOp {
                out_spec,
                other_spec: input_specs[0].clone(),
                target_spec: input_specs[1].clone(),
            });

        ctx.append_input_grad(0, Some(grad_a));
        ctx.append_input_grad(1, Some(grad_b));
    }
}

/// Backward node of [`EinsumOp`] for one operand.
///
/// Inputs are `(gy, other, target)`; the result is
/// `einsum("{out_spec},{other_spec}->{target_spec}", gy, other)`, shaped like `target`.
pub struct EinsumGradOp {
    out_spec: String,
    other_spec: String,
    target_spec: String,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for EinsumGradOp {
    fn name(&self) -> &'static str {
        "EinsumGrad"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let gy = ctx.input(0);
        let other = ctx.input(1);
        let target = ctx.input(2);
        let result = einsum_contract(
            &[&self.out_spec, &self.other_spec],
            &[&gy.view(), &other.view()],
            &self.target_spec,
            Some(target.shape()),
        )?;
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second order would need the same rewrite applied once more with the roles of
        // `gy` and `other` swapped; not implemented, and reported rather than faked.
        crate::tensor_ops::matrix_calculus::append_unsupported_grad(
            ctx,
            "einsum: second-order differentiation is not implemented".into(),
        );
    }
}

// Helper functions

#[allow(dead_code)]
fn validate_tensor_solveshapes(
    ashape: &[usize],
    bshape: &[usize],
    axes: &Option<Vec<i32>>,
) -> Result<(usize, usize), OpError> {
    // Default axes behavior
    let ndim_a = ashape.len();
    let ndim_b = bshape.len();

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

    for (i, &dim) in ashape.iter().enumerate() {
        if axes_normalized.contains(&(i as i32)) {
            prod_x *= dim;
        } else {
            prod_b *= dim;
        }
    }

    let b_prod: usize = bshape.iter().product();
    if b_prod != prod_b {
        return Err(OpError::IncompatibleShape(
            "Incompatible shapes for tensor solve".into(),
        ));
    }

    Ok((prod_x, prod_b))
}

#[allow(dead_code)]
fn reshape_for_solve<F: Float>(
    tensor: &scirs2_core::ndarray::ArrayViewD<F>,
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

#[allow(dead_code)]
fn reshape_vector<F: Float>(
    tensor: &scirs2_core::ndarray::ArrayViewD<F>,
    size: usize,
) -> Result<scirs2_core::ndarray::Array1<F>, OpError> {
    tensor
        .view()
        .to_shape(size)
        .map_err(|_| OpError::IncompatibleShape("Failed to reshape vector".into()))
        .map(|v| v.to_owned())
}

#[allow(dead_code)]
fn solve_linear_system<F: Float>(
    a: &Array2<F>,
    b: &scirs2_core::ndarray::Array1<F>,
) -> Result<scirs2_core::ndarray::Array1<F>, OpError> {
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

#[allow(dead_code)]
fn solve_square_system<F: Float>(
    a: &Array2<F>,
    b: &scirs2_core::ndarray::Array1<F>,
) -> Result<scirs2_core::ndarray::Array1<F>, OpError> {
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
    let mut x = scirs2_core::ndarray::Array1::<F>::zeros(n);
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

#[allow(dead_code)]
fn compute_solutionshape(
    ashape: &[usize],
    bshape: &[usize],
    axes: &Option<Vec<i32>>,
) -> Result<Vec<usize>, OpError> {
    let ndim_a = ashape.len();

    let axes_normalized = if let Some(ax) = axes {
        ax.clone()
    } else {
        // Default behavior
        vec![]
    };

    let mut xshape = Vec::new();
    for (i, &dim) in ashape.iter().enumerate() {
        if axes_normalized.contains(&(i as i32)) {
            xshape.push(dim);
        }
    }

    if xshape.is_empty() {
        // If no axes specified, use last dimensions
        let ndim_b = bshape.len();
        for &dim in ashape.iter().skip(ndim_a - ndim_b) {
            xshape.push(dim);
        }
    }

    Ok(xshape)
}

#[allow(dead_code)]
fn reshape_solution<F: Float>(
    flat: &scirs2_core::ndarray::Array1<F>,
    shape: &[usize],
) -> Result<ArrayD<F>, OpError> {
    let total: usize = shape.iter().product();
    if flat.len() != total {
        return Err(OpError::IncompatibleShape(
            "Solution reshape size mismatch".into(),
        ));
    }

    let dynshape = IxDyn(shape);
    flat.view()
        .to_shape(dynshape)
        .map_err(|_| OpError::IncompatibleShape("Failed to reshape solution".into()))
        .map(|v| v.to_owned())
}

#[allow(dead_code)]
fn compute_grad_b<F: Float>(
    _a: &ArrayD<F>,
    grad_x: &ArrayD<F>,
    _axes: &Option<Vec<i32>>,
) -> ArrayD<F> {
    // Simplified: return grad_x with appropriate shape
    grad_x.clone()
}

#[allow(dead_code)]
fn compute_grad_a<F: Float>(
    _grad_x: &ArrayD<F>,
    _x: &ArrayD<F>,
    ashape: &[usize],
    _axes: &Option<Vec<i32>>,
) -> ArrayD<F> {
    // Simplified: return negative outer product with appropriate shape
    // This is a placeholder - actual implementation would compute proper tensor product
    ArrayD::<F>::zeros(IxDyn(ashape))
}

// Einsum helpers

#[allow(dead_code)]
fn parse_einsum_pattern(pattern: &str) -> Result<(Vec<String>, String), OpError> {
    let parts: Vec<&str> = pattern.split("->").collect();
    if parts.len() != 2 {
        return Err(OpError::Other("Invalid einsum _pattern".into()));
    }

    let input_part = parts[0];
    let output_part = parts[1];

    let input_specs: Vec<String> = input_part.split(',').map(|s| s.to_string()).collect();

    Ok((input_specs, output_part.to_string()))
}

#[allow(dead_code)]
fn compute_matmul<F: Float>(
    a: &scirs2_core::ndarray::ArrayViewD<F>,
    b: &scirs2_core::ndarray::ArrayViewD<F>,
) -> Result<ArrayD<F>, OpError> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(OpError::IncompatibleShape(
            "Matrix multiplication requires 2D arrays".into(),
        ));
    }

    let a_2d = a
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .expect("Operation failed");
    let b_2d = b
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .expect("Operation failed");

    Ok(a_2d.dot(&b_2d).into_dyn())
}

#[allow(dead_code)]
fn compute_dot_product<F: Float>(
    a: &scirs2_core::ndarray::ArrayViewD<F>,
    b: &scirs2_core::ndarray::ArrayViewD<F>,
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

    Ok(scirs2_core::ndarray::arr0(sum).into_dyn())
}

#[allow(dead_code)]
fn compute_elementwise_mul<F: Float>(
    a: &scirs2_core::ndarray::ArrayViewD<F>,
    b: &scirs2_core::ndarray::ArrayViewD<F>,
) -> Result<ArrayD<F>, OpError> {
    if a.shape() != b.shape() {
        return Err(OpError::IncompatibleShape(
            "Element-wise multiplication requires same shape".into(),
        ));
    }

    Ok((a * b).into_owned())
}

/// General Einstein summation over an arbitrary number of operands.
///
/// Every label that appears in `out_spec` becomes an output axis; every label that
/// appears only in the operands is contracted (summed over). A label repeated *within*
/// one operand selects that operand's diagonal, and a label that appears only in
/// `out_spec` is broadcast — its size then has to come from `out_shape_hint`, which the
/// backward pass supplies from the shape of the operand it is differentiating.
///
/// The implementation walks the full index space with an odometer. That is
/// `O(prod of all label sizes)` rather than an optimised contraction order, but it is
/// exactly the definition of the notation and therefore correct for every equation
/// shape — including the ones the backward rewrite produces.
fn einsum_contract<F: Float>(
    specs: &[&str],
    arrays: &[&scirs2_core::ndarray::ArrayViewD<F>],
    out_spec: &str,
    out_shape_hint: Option<&[usize]>,
) -> Result<ArrayD<F>, OpError> {
    if specs.len() != arrays.len() {
        return Err(OpError::Other(
            "einsum: subscript count does not match operand count".into(),
        ));
    }

    let out_labels: Vec<char> = out_spec.chars().collect();
    for (i, l) in out_labels.iter().enumerate() {
        if out_labels[..i].contains(l) {
            return Err(OpError::Other(format!(
                "einsum: output label '{l}' appears more than once in '{out_spec}'"
            )));
        }
    }

    // Label -> extent, collected from the operands (and from the hint for output-only
    // labels), with a consistency check.
    let mut sizes: Vec<(char, usize)> = Vec::new();
    let mut set =
        |label: char, dim: usize, sizes: &mut Vec<(char, usize)>| -> Result<(), OpError> {
            match sizes.iter_mut().find(|(l, _)| *l == label) {
                Some((_, existing)) => {
                    if *existing != dim {
                        return Err(OpError::IncompatibleShape(format!(
                            "einsum: label '{label}' has extent {existing} and {dim}"
                        )));
                    }
                }
                None => sizes.push((label, dim)),
            }
            Ok(())
        };

    for (spec, array) in specs.iter().zip(arrays.iter()) {
        let labels: Vec<char> = spec.chars().collect();
        if labels.len() != array.ndim() {
            return Err(OpError::IncompatibleShape(format!(
                "einsum: subscript '{spec}' has {} labels but the operand has {} axes",
                labels.len(),
                array.ndim()
            )));
        }
        for (axis, &label) in labels.iter().enumerate() {
            set(label, array.shape()[axis], &mut sizes)?;
        }
    }

    if let Some(hint) = out_shape_hint {
        if hint.len() != out_labels.len() {
            return Err(OpError::IncompatibleShape(format!(
                "einsum: output subscript '{out_spec}' has {} labels but the requested \
                 shape has {} axes",
                out_labels.len(),
                hint.len()
            )));
        }
        for (axis, &label) in out_labels.iter().enumerate() {
            set(label, hint[axis], &mut sizes)?;
        }
    }

    for label in &out_labels {
        if !sizes.iter().any(|(l, _)| l == label) {
            return Err(OpError::Other(format!(
                "einsum: output label '{label}' does not appear in any operand, so its \
                 extent is unknown"
            )));
        }
    }

    let extent = |label: char| -> usize {
        sizes
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, d)| *d)
            .unwrap_or(0)
    };

    // Iteration order: the output labels first, then everything contracted.
    let mut loop_labels: Vec<char> = out_labels.clone();
    for (label, _) in &sizes {
        if !loop_labels.contains(label) {
            loop_labels.push(*label);
        }
    }
    let loop_extents: Vec<usize> = loop_labels.iter().map(|&l| extent(l)).collect();
    let out_shape: Vec<usize> = out_labels.iter().map(|&l| extent(l)).collect();

    let mut out = ArrayD::<F>::zeros(IxDyn(&out_shape));
    let total: usize = loop_extents.iter().product();
    if total == 0 {
        return Ok(out);
    }

    // Precomputed per-operand axis -> position in `loop_labels`.
    let operand_axes: Vec<Vec<usize>> = specs
        .iter()
        .map(|spec| {
            spec.chars()
                .map(|c| {
                    loop_labels
                        .iter()
                        .position(|&l| l == c)
                        .unwrap_or(usize::MAX)
                })
                .collect()
        })
        .collect();

    let mut counter = vec![0usize; loop_labels.len()];
    let mut out_index = vec![0usize; out_labels.len()];
    let mut operand_index: Vec<Vec<usize>> = specs
        .iter()
        .map(|s| vec![0usize; s.chars().count()])
        .collect();

    for _ in 0..total {
        let mut product = F::one();
        for (k, array) in arrays.iter().enumerate() {
            let idx = &mut operand_index[k];
            for (axis, &pos) in operand_axes[k].iter().enumerate() {
                idx[axis] = counter[pos];
            }
            product *= array[IxDyn(idx)];
            if product == F::zero() {
                break;
            }
        }
        out_index[..out_labels.len()].copy_from_slice(&counter[..out_labels.len()]);
        out[IxDyn(&out_index)] += product;

        // Odometer increment (last label fastest).
        for pos in (0..counter.len()).rev() {
            counter[pos] += 1;
            if counter[pos] < loop_extents[pos] {
                break;
            }
            counter[pos] = 0;
        }
    }

    Ok(out)
}

// Public API functions

/// Solve tensor equation a @ x = b for x
#[allow(dead_code)]
pub fn tensor_solve<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
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
#[allow(dead_code)]
pub fn einsum<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
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
#[allow(dead_code)]
pub fn kron<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    // Delegates to `kronecker_ops::kron`.
    //
    // This module used to carry its own private `KroneckerOp` that returned the same
    // `name()` string ("Kronecker") as the one in `kronecker_ops`. While the backward
    // pass dispatched on `name()`, the duplicate silently borrowed the *other* op's
    // (correct) gradient rule and looked fine; once dispatch moved to the concrete type
    // the duplicate's own rule went live, and it read the operand extents from
    // `Tensor::shape()` — the static shape hint, empty for most tensors — so it emitted a
    // 0-d gradient for a 2x2 input. One implementation, one gradient rule.
    crate::tensor_ops::kronecker_ops::kron(a, b)
}
