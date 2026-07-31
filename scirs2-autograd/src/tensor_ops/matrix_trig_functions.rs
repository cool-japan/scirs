use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::matrix_calculus::{
    self, MatrixFnKind, MatrixFnVjpOp, ScalarFnSymmetricVjpOp,
};
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix2};
use scirs2_core::numeric::FromPrimitive;

/// Builds the backward node of a matrix function: `MatrixFnVjp(A, gy)`.
///
/// Every op in this module is a *matrix* function `f(A)`, not an element-wise one, so its
/// VJP is the adjoint of the Fréchet derivative `L_f(A, .)` applied to the cotangent —
/// evaluated as the top-right block of `f([[Aᵀ, gy], [0, Aᵀ]])`.
///
/// The rules replaced here were element-wise: `sin(A)` used `cos(A) * gy` entry by entry,
/// which is the chain rule for `mapv(sin)`, a completely different function. `sign(A)` and
/// `funm` simply returned the cotangent unchanged.
fn append_matrix_fn_grad<F: Float>(ctx: &mut GradientContext<F>, kind: MatrixFnKind) {
    let a = *ctx.input(0);
    let gy = *ctx.output_grad();
    let g = ctx.graph();
    let gx = Tensor::builder(g)
        .append_input(a, false)
        .append_input(gy, false)
        .build(MatrixFnVjpOp { kind });
    ctx.append_input_grad(0, Some(gx));
}

/// Matrix sine function
pub struct MatrixSineOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSineOp {
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
        append_matrix_fn_grad(ctx, MatrixFnKind::Sin);
    }
}

/// Matrix cosine function
pub struct MatrixCosineOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixCosineOp {
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
        append_matrix_fn_grad(ctx, MatrixFnKind::Cos);
    }
}

/// Matrix sign function
pub struct MatrixSignOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSignOp {
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
        // `sign` is piecewise constant *along the real axis*, but sign(A) is not a
        // constant function of the matrix A: rotating the invariant subspaces moves it.
        // The Fréchet derivative is the solution of the Sylvester equation
        // `S L + L S = E - S E S`, obtained here through the same block-matrix trick
        // (the Newton iteration preserves block-triangular structure).
        append_matrix_fn_grad(ctx, MatrixFnKind::Sign);
    }
}

/// Matrix hyperbolic sine function
pub struct MatrixSinhOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixSinhOp {
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
        append_matrix_fn_grad(ctx, MatrixFnKind::Sinh);
    }
}

/// Matrix hyperbolic cosine function
pub struct MatrixCoshOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixCoshOp {
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
        append_matrix_fn_grad(ctx, MatrixFnKind::Cosh);
    }
}

/// General matrix function using eigendecomposition
pub struct MatrixFunctionOp<F: Float> {
    function: fn(F) -> F,
    name: &'static str,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for MatrixFunctionOp<F> {
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
        // `funm` applies an arbitrary scalar `fn(F) -> F` through the spectrum, so the
        // Fréchet derivative is the Daleckii-Krein formula built from divided differences
        // of that scalar function.  See `matrix_calculus::scalar_fn_symmetric_vjp`.
        let a = *ctx.input(0);
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(a, false)
            .append_input(gy, false)
            .build(ScalarFnSymmetricVjpOp {
                function: self.function,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

// Helper functions

/// Matrix sine `sin(A)`.
///
/// Scaling plus double-angle recovery, shared with the backward pass so that forward and
/// gradient are consistent by construction (see
/// [`crate::tensor_ops::matrix_calculus::sin_cos_m`]).
#[allow(dead_code)]
fn compute_matrix_sine<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::sin_cos_m(matrix).map(|(sin, _)| sin)
}

/// Matrix cosine `cos(A)`.
#[allow(dead_code)]
fn compute_matrix_cosine<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::sin_cos_m(matrix).map(|(_, cos)| cos)
}

/// Matrix sign function, by the Newton iteration `X <- (X + X^-1) / 2`.
#[allow(dead_code)]
fn compute_matrix_sign<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::signm(matrix)
}

/// Matrix hyperbolic sine `sinh(A) = (exp(A) - exp(-A)) / 2`.
#[allow(dead_code)]
fn compute_matrix_sinh<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::sinh_cosh_m(matrix).map(|(sinh, _)| sinh)
}

/// Matrix hyperbolic cosine `cosh(A) = (exp(A) + exp(-A)) / 2`.
#[allow(dead_code)]
fn compute_matrix_cosh<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::sinh_cosh_m(matrix).map(|(_, cosh)| cosh)
}

/// Applies a scalar function through the spectrum: `f(A) = V diag(f(lambda)) Vᵀ`.
///
/// # Errors
///
/// Returns an error for a non-symmetric argument. A real non-symmetric matrix has no real
/// orthogonal eigendecomposition, so `f` cannot be pushed through its spectrum this way;
/// the correct general algorithm is Schur-Parlett, which this crate does not implement.
/// The previous code returned the *input matrix unchanged* in that case, which silently
/// reported `f(A) = A` for every non-symmetric input.
#[allow(dead_code)]
fn compute_matrix_function<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    func: fn(F) -> F,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];

    if !matrix_calculus::is_symmetric(matrix) {
        return Err(OpError::Other(
            "funm: applying a scalar function through the spectrum is only supported for \
             symmetric matrices (a non-symmetric argument needs a Schur-Parlett \
             evaluation, which this crate does not provide)"
                .into(),
        ));
    }

    let (eigenvalues, eigenvectors) = matrix_calculus::symmetric_eigen(matrix)?;

    let mut temp = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            temp[[i, j]] = eigenvectors[[i, j]] * func(eigenvalues[j]);
        }
    }
    Ok(temp.dot(&eigenvectors.t()))
}

// Public API functions

/// Compute matrix sine
#[allow(dead_code)]
pub fn sinm<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixSineOp)
}

/// Compute matrix cosine
#[allow(dead_code)]
pub fn cosm<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixCosineOp)
}

/// Compute matrix sign function
#[allow(dead_code)]
pub fn signm<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixSignOp)
}

/// Compute matrix hyperbolic sine
#[allow(dead_code)]
pub fn sinhm<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixSinhOp)
}

/// Compute matrix hyperbolic cosine
#[allow(dead_code)]
pub fn coshm<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixCoshOp)
}

/// Compute general matrix function
#[allow(dead_code)]
pub fn funm<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    func: fn(F) -> F,
    name: &'static str,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let matrixshape = crate::tensor_ops::shape(matrix);

    Tensor::builder(g)
        .append_input(matrix, false)
        .setshape(&matrixshape)
        .build(MatrixFunctionOp {
            function: func,
            name,
        })
}
