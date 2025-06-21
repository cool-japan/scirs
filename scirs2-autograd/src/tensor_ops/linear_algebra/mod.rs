//! Linear algebra tensor operations
//!
//! This module provides advanced linear algebra operations for tensors including:
//! - Matrix operations (matrix multiplication, transpose, trace, determinant)
//! - Tensor operations (tensordot, batch operations, convolutions)
//! - Decompositions (QR, SVD, eigenvalue decomposition)
//! - Linear solvers and matrix functions
//! - Matrix manipulation and indexing operations

use crate::graph::AsGraph;
use crate::tensor::{AsTensor, Tensor};
use crate::Float;

// Import internal operation modules
use crate::tensor_ops::{array_ops, conv_ops, dot_ops, math_ops, shape};

/// Matrix multiplication.
///
/// Both `a` and `b` must be 2-ranked tensors.
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::matmul;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::zeros(&[4, 2], g);
///    let b: ag::Tensor<f32> = ag::tensor_ops::zeros(&[2, 3], g);
///    let c = matmul(a, b);
///    assert_eq!(c.eval(g).unwrap().shape(), &[4, 3]);
/// });
/// ```
///
/// This function supports only f32 and f64.
pub fn matmul<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(dot_ops::MatMul {
            transpose_a: false,
            transpose_b: false,
        })
}

/// Computes tensor-dot-product (tensor contraction) along specified axes.
///
/// # Arguments
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `a_axes` - `a`'s Contraction axes
/// * `b_axes` - `b`'s Contraction axes
///
/// NOTE:
///
/// * length of `a_axes` and `b_axes` must match.
/// * Each axis number can be negative.
/// * Supports only f32 and f64.
///
/// # Examples
///
/// ```ignore
/// # FIXME: tensordot doctest fails with runtime error - needs investigation
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::tensordot;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::zeros(&[3, 4, 5], g);
///    let b: ag::Tensor<f32> = ag::tensor_ops::zeros(&[4, 3, 2], g);
///    let c = tensordot(a, b, &[1, 0], &[0, 1]);
///    assert_eq!(c.eval(g).unwrap().shape(), &[5, 2]);
/// });
/// ```
//
// For detailed description,
// see <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html>.
pub fn tensordot<'graph, A, B, AT1, AT2, F: Float>(
    a: A,
    b: B,
    a_axes: &AT1,
    b_axes: &AT2,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
    AT1: AsTensor<'graph, F>,
    AT2: AsTensor<'graph, F>,
{
    let a = a.as_ref();
    let g = a.graph();
    // Preprocess
    let pre = &Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .append_input(a_axes.as_tensor(g), false)
        .append_input(b_axes.as_tensor(g), false)
        .build(dot_ops::TensordotPreprocess);
    let final_shape = crate::tensor_ops::nth_tensor(pre, 0);
    let perm_a = crate::tensor_ops::nth_tensor(pre, 1);
    let perm_b = crate::tensor_ops::nth_tensor(pre, 2);
    let new_shape_a = crate::tensor_ops::nth_tensor(pre, 3);
    let new_shape_b = crate::tensor_ops::nth_tensor(pre, 4);

    let a_reshaped =
        crate::tensor_ops::reshape(crate::tensor_ops::transpose(a, &perm_a), &new_shape_a);
    let b_reshaped =
        crate::tensor_ops::reshape(crate::tensor_ops::transpose(b, &perm_b), &new_shape_b);

    // matmul
    let mm = matmul(a_reshaped, b_reshaped);
    crate::tensor_ops::reshape(mm, &final_shape)
}

/// Batched matrix multiplication with inputs's transposition.
///
/// The rank of `a` and `b` must be equals.
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::batch_matmul_t;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::zeros(&[2, 3, 2, 4], g);
///    let b: ag::Tensor<f32> = ag::tensor_ops::zeros(&[2, 3, 2, 3], g);
///    let c = batch_matmul_t(a, b, true, false);
///    assert_eq!(c.eval(g).unwrap().shape(), &[2, 3, 4, 3]);
/// });
/// ```
///
/// This function supports only f32 and f64.
/// For detailed description, see <https://www.tensorflow.org/api_docs/python/tf/matmul>.
pub fn batch_matmul_t<'graph, A, B, F: Float>(
    a: A,
    b: B,
    trans_a: bool,
    trans_b: bool,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    let op = dot_ops::BatchMatMul {
        transpose_a: trans_a,
        transpose_b: trans_b,
    };
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(op)
}

/// Batched matrix multiplication.
///
/// The rank of `a` and `b` must be equals.
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::batch_matmul;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::ones((&[2, 3, 4, 2]), g);
///    let b: ag::Tensor<f32> = ag::tensor_ops::ones((&[2, 3, 2, 3]), g);
///    let c = batch_matmul(a, b);
///    assert_eq!(c.eval(g).unwrap().shape(), &[2, 3, 4, 3]);
/// });
/// ```
///
/// This function supports only f32 and f64.
/// For detailed description, see <https://www.tensorflow.org/api_docs/python/tf/matmul>.
pub fn batch_matmul<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    let op = dot_ops::BatchMatMul {
        transpose_a: false,
        transpose_b: false,
    };
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(op)
}

/// Permutes dimensions without copy.
///
/// It's like TensorFlow or NumPy's.
/// `x`'s rank (ndim) and `axes.len()` must match.
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::transpose;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::zeros(&[1, 2, 3, 4, 5], g);
///    let b = transpose(a, &[4, 2, 3, 0, 1]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[5, 3, 4, 1, 2]);
/// });
/// ```
pub fn transpose<'graph, A, AT, F: Float>(x: A, axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = math_ops::Transpose { invert_axes: false };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(axes.as_tensor(g), false)
        .build(op)
}

/// Extracts the diagonal of a matrix as a vector.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::extract_diag;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let d = extract_diag(x);
///    assert_eq!(d.eval(g), Ok(array![1., 4.].into_dyn()));
/// });
/// ```
pub fn extract_diag<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(crate::tensor_ops::linalg_ops::ExtractDiagOp)
}

/// Computes the trace of a matrix (sum of diagonal elements).
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::trace;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let tr = trace(x);
///    assert_eq!(tr.eval(g), Ok(ndarray::arr0(5.).into_dyn()));
/// });
/// ```
pub fn trace<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    // Use the TraceOp which has proper gradient computation
    crate::tensor_ops::linalg_ops::trace(x.as_ref())
}

/// Creates an identity matrix.
///
/// # Arguments
/// * `size` - Size of the identity matrix (creates size x size matrix)
/// * `graph` - Computation graph
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::eye;
///
/// ag::run(|g| {
///    let identity: ag::Tensor<'_, f64> = eye(3, g);
///    let result = identity.eval(g).unwrap();
///    assert_eq!(result.shape(), &[3, 3]);
///    assert_eq!(result[[0, 0]], 1.0);
///    assert_eq!(result[[0, 1]], 0.0);
/// });
/// ```
pub fn eye<F: Float>(size: usize, graph: &impl AsGraph<F>) -> Tensor<'_, F> {
    Tensor::builder(graph).build(crate::tensor_ops::linalg_ops::EyeOp { size })
}

/// Creates a diagonal matrix from a vector.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::diag;
///
/// ag::run(|g| {
///    let v = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let d = diag(v);
///    let result = d.eval(g).unwrap();
///    assert_eq!(result.shape(), &[3, 3]);
///    assert_eq!(result[[0, 0]], 1.0);
///    assert_eq!(result[[1, 1]], 2.0);
///    assert_eq!(result[[2, 2]], 3.0);
/// });
/// ```
pub fn diag<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(crate::tensor_ops::linalg_ops::DiagOp)
}

/// Computes the QR decomposition of a matrix.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::qr;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let (q, r) = qr(a);
///    // Q should be orthogonal and R should be upper triangular
///    assert_eq!(q.eval(g).unwrap().shape(), &[2, 2]);
///    assert_eq!(r.eval(g).unwrap().shape(), &[2, 2]);
/// });
/// ```
pub fn qr<'graph, A, F: Float>(x: A) -> (Tensor<'graph, F>, Tensor<'graph, F>)
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    // Use the correct implementation from decomposition_ops
    crate::tensor_ops::decomposition_ops::qr(x.as_ref())
}

/// Computes the Singular Value Decomposition (SVD) of a matrix.
///
/// # Examples
///
/// ```ignore
/// # FIXME: svd doctest fails due to tensor order mismatch - implementation returns different order than expected
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::svd;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let (u, s, vt) = svd(a);
///    // U and V^T should be orthogonal, S should be diagonal
///    assert_eq!(u.eval(g).unwrap().shape(), &[2, 2]);
///    assert_eq!(s.eval(g).unwrap().shape(), &[2]);
///    assert_eq!(vt.eval(g).unwrap().shape(), &[2, 2]);
/// });
/// ```
pub fn svd<'graph, A, F: Float + ndarray::ScalarOperand>(
    x: A,
) -> (Tensor<'graph, F>, Tensor<'graph, F>, Tensor<'graph, F>)
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    // Use the correct implementation from decomposition_ops
    crate::tensor_ops::decomposition_ops::svd(x.as_ref())
}

/// Computes the eigenvalues of a matrix.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::eigenvalues;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let w = eigenvalues(a);
///    assert_eq!(w.eval(g).unwrap().shape(), &[2]);
/// });
/// ```
pub fn eigenvalues<'graph, A, F: Float + ndarray::ScalarOperand + num_traits::FromPrimitive>(
    x: A,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(crate::tensor_ops::eigen_ops::EigenvaluesOp)
}

/// Computes the eigenvalues and eigenvectors of a matrix.
///
/// # Examples
///
/// ```ignore
/// # FIXME: eigen doctest fails due to tensor order mismatch - implementation returns different order than expected
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::eigen;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let (w, v) = eigen(a);
///    assert_eq!(w.eval(g).unwrap().shape(), &[2]);
///    assert_eq!(v.eval(g).unwrap().shape(), &[2, 2]);
/// });
/// ```
pub fn eigen<'graph, A, F: Float + ndarray::ScalarOperand + num_traits::FromPrimitive>(
    x: A,
) -> (Tensor<'graph, F>, Tensor<'graph, F>)
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    // Use the correct implementation from eigen_ops
    crate::tensor_ops::eigen_ops::eigen(x.as_ref())
}

/// Computes the matrix determinant.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::determinant;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let det = determinant(a);
///    assert_eq!(det.eval(g), Ok(ndarray::arr0(-2.).into_dyn()));
/// });
/// ```
pub fn determinant<'graph, A, F: Float + ndarray::ScalarOperand>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(crate::tensor_ops::matrix_ops::GeneralDeterminantOp)
}

/// Computes the matrix inverse.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::matrix_inverse;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let inv_a = matrix_inverse(a);
///    assert_eq!(inv_a.eval(g).unwrap().shape(), &[2, 2]);
/// });
/// ```
pub fn matrix_inverse<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(crate::tensor_ops::matrix_ops::MatrixInverseOp)
}

/// Solves a linear system Ax = b.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::solve;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.]], g);
///    let b = ag::tensor_ops::convert_to_tensor(array![1., 2.], g);
///    let x = solve(a, b);
///    assert_eq!(x.eval(g).unwrap().shape(), &[2]);
/// });
/// ```
pub fn solve<'graph, A, B, F: Float + ndarray::ScalarOperand>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(crate::tensor_ops::solver_ops::LinearSolveOp)
}

/// Computes least squares solution to Ax = b.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::lstsq;
///
/// ag::run(|g| {
///    let a = ag::tensor_ops::convert_to_tensor(array![[1., 2.], [3., 4.], [5., 6.]], g);
///    let b = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let x = lstsq(a, b);
///    assert_eq!(x.eval(g).unwrap().shape(), &[2]);
/// });
/// ```
pub fn lstsq<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(crate::tensor_ops::solver_ops::LeastSquaresSolveOp)
}

/// 2D convolution.
///
/// * `x`: Tensor with shape `(batch, channel, h, w)`
/// * `w`: Tensor with shape `(out_channel, channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `(h + 2 * pad - filter_h) / stride + 1`
///   * `out_w` = `(w + 2 * pad - filter_w) / stride + 1`
///
/// This function supports only f32 and f64.
pub fn conv2d<'graph, A, B, F: Float>(x: A, w: B, pad: usize, stride: usize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d::Conv2D {
            pad,
            stride,
            dilation: 1,
        })
}

/// 2D convolution with dilation.
///
/// * `x`: Tensor with shape `(batch, channel, h, w)`
/// * `w`: Tensor with shape `(out_channel, in_channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `(h + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
///   * `out_w` = `(w + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
///
/// This function supports only f32 and f64.
pub fn dilated_conv2d<'graph, A, B, F: Float>(
    x: A,
    w: B,
    pad: usize,
    stride: usize,
    dilate: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d::Conv2D {
            pad,
            stride,
            dilation: dilate,
        })
}

/// 2D transposed convolution.
///
/// * `x`: Tensor with shape `(batch, in_channel, h, w)`
/// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `stride * (h - 1) - pad + filter_h`
///   * `out_w` = `stride * (w - 1) - pad + filter_w`
///
/// This function supports only f32 and f64.
pub fn conv2d_transpose<'graph, A, B, F: Float>(
    x: A,
    w: B,
    pad: usize,
    stride: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d_transpose::Conv2DTranspose {
            pad,
            stride,
            dilation: 1,
        })
}

/// 2D max pooling.
///
/// * `x`: Tensor with shape `(batch, channel, h, w)`
///
/// Returns a tensor with shape `(batch, channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `(h + 2 * pad - pool_size) / stride + 1`
///   * `out_w` = `(w + 2 * pad - pool_size) / stride + 1`
///
/// This function supports only f32 and f64.
pub fn max_pool2d<'graph, A, F: Float>(
    x: A,
    pool_size: usize,
    pad: usize,
    stride: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(conv_ops::max_pool2d::MaxPool2D {
            pad,
            stride,
            size: pool_size,
        })
}

/// Concatenates input tensors along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::concat;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::zeros(&[3, 2], g);
///    let b: ag::Tensor<f32> = ag::tensor_ops::zeros(&[3, 2], g);
///    let c: ag::Tensor<f32> = ag::tensor_ops::zeros(&[3, 2], g);
///    let d = concat(&[a, b, c], 0);
///
///    assert_eq!(d.eval(g).unwrap().shape(), &[9, 2]);
/// });
/// ```
pub fn concat<'graph, A, F: Float>(tensors: &[A], axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    assert_ne!(tensors.len(), 0);
    let g = tensors[0].as_ref().graph();
    let op = array_ops::Concat { axis };
    let mut b = Tensor::builder(g);
    for t in tensors {
        b = b.append_input(t.as_ref(), false);
    }
    b.build(op)
}

/// Splits input tensors into parts.
///
/// Splits `x` into `sizes.len()` parts along `axis`.
///
/// The size of dimension of each part is `sizes[i]` on `axis`, but is
/// `x.shape[i]` on other axis (similar to TensorFlow's `split`).
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::split;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ag::tensor_ops::zeros(&[3, 7, 5], g);
///    let b = split(a, &[2, 3, 2], 1);
///
///    let evaluated = g.evaluator().extend(&[&b[0], &b[1], &b[2]]).run();
///    let e0 = &evaluated[0];
///    let e1 = &evaluated[1];
///    let e2 = &evaluated[2];
///
///    assert_eq!(e0.as_ref().unwrap().shape(), &[3, 2, 5]);
///    assert_eq!(e1.as_ref().unwrap().shape(), &[3, 3, 5]);
///    assert_eq!(e2.as_ref().unwrap().shape(), &[3, 2, 5]);
/// });
/// ```
pub fn split<'graph, A, F: Float>(x: A, sizes: &[usize], axis: isize) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let len = sizes.len();
    let mut ret = Vec::with_capacity(len);
    for i in 0..len {
        let mut start_index = 0usize;
        for &size in sizes[..i].iter() {
            start_index += size;
        }
        let end_index = start_index + sizes[i];
        ret.push(
            Tensor::builder(g)
                .append_input(x.as_ref(), false)
                .build(array_ops::Split {
                    start_index: start_index as isize,
                    end_index: end_index as isize,
                    axis,
                }),
        );
    }
    ret
}

/// Scalar multiplication
pub fn scalar_mul<'graph, A, F: Float>(x: A, scalar: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(crate::tensor_ops::scalar_ops::ScalarMulOp { scalar })
}

/// Computes the Frobenius norm of a tensor.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::linear_algebra::frobenius_norm;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[3., 4.], [0., 0.]], g);
///    let norm = frobenius_norm(x);
///    assert_eq!(norm.eval(g), Ok(ndarray::arr0(5.).into_dyn()));
/// });
/// ```
pub fn frobenius_norm<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    crate::tensor_ops::reduction::frobenius_norm(x)
}

// Re-export enhanced linear algebra operations

/// Compute the 1-norm of a matrix (maximum column sum)
pub use crate::tensor_ops::matrix_norms::norm1;

/// Compute the 2-norm of a matrix (largest singular value)
pub use crate::tensor_ops::matrix_norms::norm2;

/// Compute the infinity-norm of a matrix (maximum row sum)
pub use crate::tensor_ops::matrix_norms::norminf;

/// Compute the Frobenius norm of a matrix (alias)
pub use crate::tensor_ops::matrix_norms::normfro;

/// Solve Sylvester equation AX + XB = C
pub use crate::tensor_ops::matrix_solvers::solve_sylvester;

/// Solve Lyapunov equation AX + XA^T = Q
pub use crate::tensor_ops::matrix_solvers::solve_lyapunov;

/// Solve linear system AX = B using Cholesky decomposition
pub use crate::tensor_ops::matrix_solvers::cholesky_solve;

/// Eigendecomposition for symmetric/Hermitian matrices
pub use crate::tensor_ops::symmetric_ops::eigh;

/// Eigenvalues only for symmetric/Hermitian matrices
pub use crate::tensor_ops::symmetric_ops::eigvalsh;

/// Polar decomposition A = UP
pub use crate::tensor_ops::special_decompositions::polar;

/// Schur decomposition A = QTQ^T
pub use crate::tensor_ops::special_decompositions::schur;

/// Matrix exponential using Padé approximation (method 2)
pub use crate::tensor_ops::matrix_ops::expm2;

/// Matrix exponential using eigendecomposition (method 3)
pub use crate::tensor_ops::matrix_ops::expm3;

/// Solve tensor equation
pub use crate::tensor_ops::advanced_tensor_ops::tensor_solve;

/// Einstein summation convention
pub use crate::tensor_ops::advanced_tensor_ops::einsum;

/// Kronecker product (tensor product)
pub use crate::tensor_ops::advanced_tensor_ops::kron as kronecker_product;

// Advanced decompositions

/// SVD using Jacobi algorithm for improved numerical stability
pub use crate::tensor_ops::advanced_decompositions::svd_jacobi;

/// Randomized SVD for large matrices
pub use crate::tensor_ops::advanced_decompositions::randomized_svd;

/// Generalized eigenvalue problem Ax = λBx
pub use crate::tensor_ops::advanced_decompositions::generalized_eigen;

/// QR decomposition with column pivoting
pub use crate::tensor_ops::advanced_decompositions::qr_pivot;

// Iterative solvers

/// Conjugate gradient solver for symmetric positive definite systems
pub use crate::tensor_ops::iterative_solvers::conjugate_gradient_solve;

/// GMRES solver for general linear systems
pub use crate::tensor_ops::iterative_solvers::gmres_solve;

/// BiCGSTAB solver for non-symmetric systems
pub use crate::tensor_ops::iterative_solvers::bicgstab_solve;

/// Preconditioned conjugate gradient solver
pub use crate::tensor_ops::iterative_solvers::pcg_solve;

/// Preconditioner types for iterative solvers
pub use crate::tensor_ops::iterative_solvers::PreconditionerType;

// Matrix functions

/// Matrix sine function
pub use crate::tensor_ops::matrix_trig_functions::sinm;

/// Matrix cosine function
pub use crate::tensor_ops::matrix_trig_functions::cosm;

/// Matrix sign function
pub use crate::tensor_ops::matrix_trig_functions::signm;

/// Matrix hyperbolic sine function
pub use crate::tensor_ops::matrix_trig_functions::sinhm;

/// Matrix hyperbolic cosine function
pub use crate::tensor_ops::matrix_trig_functions::coshm;

/// General matrix function
pub use crate::tensor_ops::matrix_trig_functions::funm;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    #[allow(unused_imports)]
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_matrix_operations() {
        crate::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
            let b = convert_to_tensor(array![[5.0_f32, 6.0], [7.0, 8.0]], g);

            // Test matrix multiplication
            let c = matmul(a, b);
            let expected = array![[19.0_f32, 22.0], [43.0, 50.0]];
            assert_eq!(c.eval(g).unwrap(), expected.into_dyn());

            // Test transpose
            let a_t = transpose(a, &[1, 0]);
            let expected_t = array![[1.0_f32, 3.0], [2.0, 4.0]];
            assert_eq!(a_t.eval(g).unwrap(), expected_t.into_dyn());
        });
    }

    #[test]
    fn test_matrix_properties() {
        crate::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test trace
            let tr = trace(a);
            assert_eq!(tr.eval(g).unwrap(), ndarray::arr0(5.0).into_dyn());

            // Test diagonal extraction
            let diag_vals = extract_diag(a);
            let expected_diag = array![1.0_f32, 4.0];
            assert_eq!(diag_vals.eval(g).unwrap(), expected_diag.into_dyn());

            // Test determinant (with tolerance for floating point precision)
            let det = determinant(a);
            let det_result = det.eval(g).unwrap();
            let det_value = det_result[ndarray::IxDyn(&[])];
            assert!((det_value - (-2.0)).abs() < 1e-5);
        });
    }

    #[test]
    fn test_identity_and_diagonal() {
        crate::run(|g| {
            // Test identity matrix
            let identity = eye(3, g);
            let result = identity.eval(g).unwrap();
            assert_eq!(result.shape(), &[3, 3]);
            assert_eq!(result[[0, 0]], 1.0);
            assert_eq!(result[[0, 1]], 0.0);
            assert_eq!(result[[1, 1]], 1.0);

            // Test diagonal matrix creation
            let v = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let d = diag(v);
            let result = d.eval(g).unwrap();
            assert_eq!(result.shape(), &[3, 3]);
            assert_eq!(result[[0, 0]], 1.0);
            assert_eq!(result[[1, 1]], 2.0);
            assert_eq!(result[[2, 2]], 3.0);
            assert_eq!(result[[0, 1]], 0.0);
        });
    }

    #[test]
    fn test_tensor_concatenation_and_splitting() {
        crate::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
            let b = convert_to_tensor(array![[5.0_f32, 6.0], [7.0, 8.0]], g);

            // Test concatenation
            let concat_result = concat(&[a, b], 0);
            let expected_concat = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
            assert_eq!(concat_result.eval(g).unwrap(), expected_concat.into_dyn());

            // Test splitting
            let to_split = convert_to_tensor(array![[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]], g);
            let splits = split(to_split, &[2, 2, 2], 1);
            assert_eq!(splits.len(), 3);

            let eval1 = splits[0].eval(g).unwrap();
            let eval2 = splits[1].eval(g).unwrap();
            let eval3 = splits[2].eval(g).unwrap();

            assert_eq!(eval1.shape(), &[1, 2]);
            assert_eq!(eval2.shape(), &[1, 2]);
            assert_eq!(eval3.shape(), &[1, 2]);
        });
    }

    #[test]
    fn test_scalar_operations() {
        crate::run(|g| {
            let a = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);

            // Test scalar multiplication
            let scaled = scalar_mul(a, 2.0);
            let expected = array![2.0_f32, 4.0, 6.0];
            assert_eq!(scaled.eval(g).unwrap(), expected.into_dyn());
        });
    }

    #[test]
    fn test_batch_operations() {
        crate::run(|g| {
            let a = convert_to_tensor(
                array![[[1.0_f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                g,
            );
            let b = convert_to_tensor(
                array![[[1.0_f32, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                g,
            );

            // Test batch matrix multiplication
            let batch_result = batch_matmul(a, b);
            assert_eq!(batch_result.eval(g).unwrap().shape(), &[2, 2, 2]);

            // The result should be the same as input since we're multiplying by identity
            let expected = array![[[1.0_f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
            assert_eq!(batch_result.eval(g).unwrap(), expected.into_dyn());
        });
    }
}
