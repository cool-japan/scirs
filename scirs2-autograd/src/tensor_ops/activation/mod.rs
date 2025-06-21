//! Activation functions for neural networks
//!
//! This module provides various activation functions commonly used in neural networks:
//! - Basic activations (sigmoid, tanh, ReLU, leaky ReLU)
//! - Advanced activations (ELU, Swish, GELU, Mish)
//! - Softmax and log-softmax functions
//! - Loss functions (cross-entropy variants)
//! - Normalization functions

use crate::tensor::{AsTensor, Tensor};
use crate::Float;

// Import internal operation modules
use crate::tensor_ops::{activation_ops, scalar, shape, xent_ops};

/// Elementwise logistic sigmoid function.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::sigmoid;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![0., 1., -1.], g);
///    let y = sigmoid(x);
///    let result = y.eval(g).unwrap();
///    // sigmoid(0) ≈ 0.5, sigmoid(1) ≈ 0.73, sigmoid(-1) ≈ 0.27
/// });
/// ```
pub fn sigmoid<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::Sigmoid)
}

/// Elementwise hyperbolic tangent function.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::tanh;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![0., 1., -1.], g);
///    let y = tanh(x);
///    let result = y.eval(g).unwrap();
///    // tanh(0) = 0, tanh(1) ≈ 0.76, tanh(-1) ≈ -0.76
/// });
/// ```
pub fn tanh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(crate::tensor_ops::math_ops::Tanh)
}

/// Elementwise rectified linear unit.
///
/// ReLU(x) = max(0, x)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::relu;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![-1., 0., 1., 2.], g);
///    let y = relu(x);
///    assert_eq!(y.eval(g), Ok(array![0., 0., 1., 2.].into_dyn()));
/// });
/// ```
pub fn relu<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::ReLU)
}

/// Elementwise leaky relu.
///
/// LeakyReLU(x) = max(alpha * x, x)
/// In common, `alpha` is around 0.1 ~ 0.2.
///
/// https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::leaky_relu;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![-1., 0., 1., 2.], g);
///    let y = leaky_relu(x, 0.1);
///    assert_eq!(y.eval(g), Ok(array![-0.1, 0., 1., 2.].into_dyn()));
/// });
/// ```
pub fn leaky_relu<'graph, A, F: Float>(x: A, alpha: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    crate::tensor_ops::arithmetic::maximum(x, scalar(alpha, g) * x)
}

/// Elementwise exponential linear unit.
///
/// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
/// See <https://arxiv.org/abs/1511.07289>
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::elu;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![-1., 0., 1., 2.], g);
///    let y = elu(x, 1.0);
///    let result = y.eval(g).unwrap();
///    // ELU smoothly transitions from exponential to linear
/// });
/// ```
pub fn elu<'graph, A, F: Float>(x: A, alpha: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::Elu { alpha })
}

/// Elementwise softplus.
///
/// Softplus(x) = log(1 + exp(x))
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::softplus;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![0., 1., -1.], g);
///    let y = softplus(x);
///    let result = y.eval(g).unwrap();
///    // Softplus is a smooth approximation to ReLU
/// });
/// ```
pub fn softplus<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::Softplus)
}

/// Elementwise Swish activation function: x * sigmoid(x).
///
/// Also known as SiLU (Sigmoid Linear Unit).
/// See <https://arxiv.org/abs/1710.05941>
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::swish;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![0., 1., -1.], g);
///    let y = swish(x);
///    let result = y.eval(g).unwrap();
///    // Swish: x * sigmoid(x)
/// });
/// ```
pub fn swish<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::Swish)
}

/// Elementwise GELU (Gaussian Error Linear Unit) activation function.
///
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// See <https://arxiv.org/abs/1606.08415>
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::gelu;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![0., 1., -1.], g);
///    let y = gelu(x);
///    let result = y.eval(g).unwrap();
///    // GELU is commonly used in transformer models
/// });
/// ```
pub fn gelu<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::Gelu)
}

/// Elementwise Mish activation function: x * tanh(softplus(x)).
///
/// Mish(x) = x * tanh(ln(1 + exp(x)))
/// See <https://arxiv.org/abs/1908.08681>
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::mish;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![0., 1., -1.], g);
///    let y = mish(x);
///    let result = y.eval(g).unwrap();
///    // Mish: x * tanh(softplus(x))
/// });
/// ```
pub fn mish<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(activation_ops::Mish)
}

/// Computes softmax along specified axis
///
/// Softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::softmax;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2., 3.], [4., 5., 6.]], g);
///    let y = softmax(x, 1); // softmax along columns
///    let result = y.eval(g).unwrap();
///    // Each row should sum to 1
/// });
/// ```
pub fn softmax<'graph, A, F: Float>(x: A, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = activation_ops::Softmax { axis };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Log softmax function.
///
/// Computes `softmax(x)` along specified axis and
/// takes logarithm of it.
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::log_softmax;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![[1., 2., 3.], [4., 5., 6.]], g);
///    let y = log_softmax(x, 1); // log softmax along columns
///    let result = y.eval(g).unwrap();
///    // Log of softmax probabilities
/// });
/// ```
pub fn log_softmax<'graph, A, F: Float>(x: A, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(xent_ops::LogSoftmax { axis })
}

/// Computes `binary_cross_entropy(sigmoid(y), t)`.
///
/// This function is better than that combination in that it can prevent
/// underflow of `log(sigmoid)`.
///
/// # Arguments
/// * `y` - Tensor with arbitrary shape
/// * `t` - Ground-truth Tensor with same shape as `y`
///
/// # Panics
/// When y.shape != t.shape.
///
/// # Returns
/// Loss tensor with same shape as inputs's shapes
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::sigmoid_cross_entropy;
///
/// ag::run(|g| {
///    let logits = ag::tensor_ops::convert_to_tensor(array![0.5, -0.5, 1.0], g);
///    let targets = ag::tensor_ops::convert_to_tensor(array![1., 0., 1.], g);
///    let loss = sigmoid_cross_entropy(logits, targets);
///    let result = loss.eval(g).unwrap();
///    // Binary cross-entropy loss
/// });
/// ```
pub fn sigmoid_cross_entropy<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let y = y.as_ref();
    let g = y.graph();
    let op = xent_ops::SigmoidCrossEntropy;
    Tensor::builder(g)
        .set_shape(&shape(y))
        .append_input(y.as_ref(), false)
        .append_input(t.as_ref(), false)
        .build(op)
}

/// Computes `categorical_cross_entropy(softmax(y), t)`.
///
/// This function is better than that combination in that it can prevent
/// underflow of `log(softmax)`.
///
/// # Arguments
/// * `y` - Tensor with shape (batch_size, num_classes)
/// * `t` - Tensor with shape (batch_size, num_classes)
///
/// # Returns
/// Loss tensor with shape (batch_size, 1)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::softmax_cross_entropy;
///
/// ag::run(|g| {
///    let logits = ag::tensor_ops::convert_to_tensor(array![[1., 2., 3.], [4., 5., 6.]], g);
///    let targets = ag::tensor_ops::convert_to_tensor(array![[0., 0., 1.], [1., 0., 0.]], g);
///    let loss = softmax_cross_entropy(logits, targets);
///    let result = loss.eval(g).unwrap();
///    // Categorical cross-entropy loss
/// });
/// ```
pub fn softmax_cross_entropy<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let y = y.as_ref();
    let g = y.graph();
    let op = xent_ops::SoftmaxCrossEntropy;
    Tensor::builder(g)
        .append_input(y.as_ref(), false)
        .append_input(t.as_ref(), false)
        .build(op)
}

/// A variant of `softmax_cross_entropy`.
///
/// The behavior of this function is same as `softmax_cross_entropy`
/// except that `t` is **not** batch of one-hot distributions but batch of ground truth label ids.
///
/// # Arguments
/// * `y` - Tensor with shape (batch_size, num_classes)
/// * `t` - Tensor with shape (batch_size,) or (batch_size, 1)
///
/// # Returns
/// Loss tensor with shape (batch_size, 1)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::sparse_softmax_cross_entropy;
///
/// ag::run(|g| {
///    let logits = ag::tensor_ops::convert_to_tensor(array![[1., 2., 3.], [4., 5., 6.]], g);
///    let labels = ag::tensor_ops::convert_to_tensor(array![2., 0.], g); // class indices
///    let loss = sparse_softmax_cross_entropy(logits, labels);
///    let result = loss.eval(g).unwrap();
///    // Sparse categorical cross-entropy loss
/// });
/// ```
pub fn sparse_softmax_cross_entropy<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let y = y.as_ref();
    let g = y.graph();
    let op = xent_ops::SparseSoftmaxCrossEntropy;
    Tensor::builder(g)
        .append_input(y.as_ref(), false)
        .append_input(t.as_ref(), false)
        .build(op)
}

/// Computes mean squared error
///
/// Note that the mean axis is the last one.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::mean_squared_error;
///
/// ag::run(|g| {
///    let predictions = ag::tensor_ops::convert_to_tensor(array![1., 2., 3.], g);
///    let targets = ag::tensor_ops::convert_to_tensor(array![1.5, 2.5, 2.5], g);
///    let loss = mean_squared_error(predictions, targets);
///    let result = loss.eval(g).unwrap();
///    // MSE loss
/// });
/// ```
pub fn mean_squared_error<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    crate::tensor_ops::reduction::reduce_mean(
        crate::tensor_ops::arithmetic::square(y.as_ref() - t.as_ref()),
        &[-1],
        false,
    )
}

/// Normalizes the input tensor with its mean and variance along specified axis.
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::normalize;
///
/// ag::run(|g| {
///    let x: ag::Tensor<f32> = ag::tensor_ops::standard_normal(&[3, 4], g);
///    let y1 = normalize(x, &[0]);
///    let y2 = normalize(x, &[0]);
///
///    let evaluated = g.evaluator().extend(&[y1, y2]).run();
///    let e0 = &evaluated[0];
///    let e1 = &evaluated[1];
///    assert_eq!(e0.as_ref().unwrap().shape(), &[3, 4]);
///    assert_eq!(e1.as_ref().unwrap().shape(), &[3, 4]);
/// });
/// ```
pub fn normalize<'graph, A, AT, F: Float>(_x: A, _axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = _x.as_ref();
    let g = x.graph();
    let axes = _axes.as_tensor(g);
    let mean = crate::tensor_ops::reduction::reduce_mean(x, &axes, true);
    let centered = x - mean;
    let variance = crate::tensor_ops::reduction::reduce_mean(
        crate::tensor_ops::arithmetic::square(centered),
        &axes,
        true,
    );
    let em5 = scalar(F::from(1e-5).unwrap(), g);
    centered * crate::tensor_ops::arithmetic::inv_sqrt(variance + em5)
}

/// Applies batch normalization.
///
/// `scale` and `shift` should be shared variables.
/// Since normalization is performed along 1st axis of `x`,
/// both of them should have shape `(1, x.shape[1])`
///
/// # Examples
///
/// ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::batch_norm;
/// use ag::prelude::*;
///
/// let mut env = ag::VariableEnvironment::new();
/// let scale = env.set(ag::ndarray_ext::ones::<f32>(&[1, 4]));
/// let shift = env.set(ag::ndarray_ext::zeros::<f32>(&[1, 4]));
///
/// env.run(|g| {
///    let x = ag::tensor_ops::standard_normal(&[3, 4], g);
///    let norm = batch_norm(x, g.variable(scale), g.variable(shift));
///
///    assert_eq!(norm.eval(g).unwrap().shape(), &[3, 4]);
/// });
/// ```
pub fn batch_norm<'graph, A, B, C, F: Float>(x: A, scale: B, shift: C) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
    C: AsRef<Tensor<'graph, F>> + Copy,
{
    normalize(x, &[0]) * scale.as_ref() + shift.as_ref()
}

/// Hard sigmoid activation function.
///
/// HardSigmoid(x) = max(0, min(1, (x + 1) / 2))
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::hard_sigmoid;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![-2., 0., 2.], g);
///    let y = hard_sigmoid(x);
///    assert_eq!(y.eval(g), Ok(array![0., 0.5, 1.].into_dyn()));
/// });
/// ```
pub fn hard_sigmoid<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let one = scalar(F::one(), g);
    let two = scalar(F::from(2.0).unwrap(), g);
    let zero = scalar(F::zero(), g);

    let shifted = (x + one) / two;
    crate::tensor_ops::arithmetic::maximum(
        zero,
        crate::tensor_ops::arithmetic::minimum(one, shifted),
    )
}

/// Hard tanh activation function.
///
/// HardTanh(x) = max(-1, min(1, x))
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::hard_tanh;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![-2., 0., 2.], g);
///    let y = hard_tanh(x);
///    assert_eq!(y.eval(g), Ok(array![-1., 0., 1.].into_dyn()));
/// });
/// ```
pub fn hard_tanh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let one = scalar(F::one(), g);
    let neg_one = scalar(-F::one(), g);

    crate::tensor_ops::arithmetic::maximum(neg_one, crate::tensor_ops::arithmetic::minimum(one, x))
}

/// ReLU6 activation function.
///
/// ReLU6(x) = min(max(0, x), 6)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_autograd as ag;
/// use ag::tensor_ops::activation::relu6;
///
/// ag::run(|g| {
///    let x = ag::tensor_ops::convert_to_tensor(array![-1., 3., 8.], g);
///    let y = relu6(x);
///    assert_eq!(y.eval(g), Ok(array![0., 3., 6.].into_dyn()));
/// });
/// ```
pub fn relu6<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let zero = scalar(F::zero(), g);
    let six = scalar(F::from(6.0).unwrap(), g);

    crate::tensor_ops::arithmetic::minimum(crate::tensor_ops::arithmetic::maximum(zero, x), six)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_basic_activations() {
        crate::run(|g| {
            let x = convert_to_tensor(array![-1.0_f32, 0.0, 1.0], g);

            // Test ReLU
            let relu_result = relu(x);
            let expected_relu = array![0.0_f32, 0.0, 1.0];
            assert_eq!(relu_result.eval(g).unwrap(), expected_relu.into_dyn());

            // Test Leaky ReLU
            let leaky_result = leaky_relu(x, 0.1);
            let expected_leaky = array![-0.1_f32, 0.0, 1.0];
            assert_eq!(leaky_result.eval(g).unwrap(), expected_leaky.into_dyn());

            // Test sigmoid
            let sigmoid_result = sigmoid(x);
            let actual_sigmoid = sigmoid_result.eval(g).unwrap();
            // sigmoid(-1) ≈ 0.27, sigmoid(0) = 0.5, sigmoid(1) ≈ 0.73
            assert_relative_eq!(actual_sigmoid[0], 0.2689414, epsilon = 1e-6);
            assert_relative_eq!(actual_sigmoid[1], 0.5, epsilon = 1e-6);
            assert_relative_eq!(actual_sigmoid[2], 0.7310586, epsilon = 1e-6);
        });
    }

    #[test]
    fn test_advanced_activations() {
        crate::run(|g| {
            let x = convert_to_tensor(array![-1.0_f32, 0.0, 1.0], g);

            // Test ELU
            let elu_result = elu(x, 1.0);
            let actual_elu = elu_result.eval(g).unwrap();
            // elu(-1, 1.0) ≈ -0.632, elu(0) = 0, elu(1) = 1
            assert_relative_eq!(actual_elu[1], 0.0, epsilon = 1e-6);
            assert_relative_eq!(actual_elu[2], 1.0, epsilon = 1e-6);

            // Test softplus
            let softplus_result = softplus(x);
            let actual_softplus = softplus_result.eval(g).unwrap();
            // All values should be positive
            assert!(actual_softplus[0] > 0.0);
            assert!(actual_softplus[1] > 0.0);
            assert!(actual_softplus[2] > 0.0);
        });
    }

    #[test]
    fn test_softmax() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0, 3.0]], g);

            // Test softmax with 2D input (batch of 1)
            let softmax_result = softmax(x, 1); // Apply softmax along last axis
            let actual = softmax_result.eval(g).unwrap();

            // Check that probabilities sum to 1
            let sum: f32 = actual.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

            // Test log softmax
            let log_softmax_result = log_softmax(x, 1);
            let log_actual = log_softmax_result.eval(g).unwrap();

            // Log probabilities should be negative - access as 2D array
            let log_slice = log_actual.index_axis(ndarray::Axis(0), 0);
            assert!(log_slice[0] < 0.0);
            assert!(log_slice[1] < 0.0);
            assert!(log_slice[2] < 0.0);
        });
    }

    #[test]
    fn test_loss_functions() {
        crate::run(|g| {
            let logits = convert_to_tensor(array![0.5_f32, -0.5], g);
            let targets = convert_to_tensor(array![1.0_f32, 0.0], g);

            // Test sigmoid cross entropy
            let loss = sigmoid_cross_entropy(logits, targets);
            let loss_val = loss.eval(g).unwrap();

            // Loss should be positive
            assert!(loss_val[0] > 0.0);
            assert!(loss_val[1] > 0.0);

            // Test MSE
            let predictions = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let true_vals = convert_to_tensor(array![1.5_f32, 2.5, 2.5], g);
            let mse = mean_squared_error(predictions, true_vals);
            let mse_val = mse.eval(g).unwrap();

            // MSE should be positive
            assert!(mse_val[ndarray::IxDyn(&[])] > 0.0);
        });
    }

    #[test]
    fn test_hard_activations() {
        crate::run(|g| {
            let x = convert_to_tensor(array![-2.0_f32, 0.0, 2.0], g);

            // Test hard sigmoid
            let hard_sig_result = hard_sigmoid(x);
            let expected_hard_sig = array![0.0_f32, 0.5, 1.0];
            assert_eq!(
                hard_sig_result.eval(g).unwrap(),
                expected_hard_sig.into_dyn()
            );

            // Test hard tanh
            let hard_tanh_result = hard_tanh(x);
            let expected_hard_tanh = array![-1.0_f32, 0.0, 1.0];
            assert_eq!(
                hard_tanh_result.eval(g).unwrap(),
                expected_hard_tanh.into_dyn()
            );

            // Test ReLU6
            let x2 = convert_to_tensor(array![-1.0_f32, 3.0, 8.0], g);
            let relu6_result = relu6(x2);
            let expected_relu6 = array![0.0_f32, 3.0, 6.0];
            assert_eq!(relu6_result.eval(g).unwrap(), expected_relu6.into_dyn());
        });
    }

    #[test]
    fn test_normalization() {
        crate::run(|g| {
            let x = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Test normalize
            let normalized = normalize(x, &[0]);
            let result = normalized.eval(g).unwrap();

            // After normalization along axis 0, mean should be close to 0
            assert_eq!(result.shape(), &[2, 2]);
        });
    }
}
