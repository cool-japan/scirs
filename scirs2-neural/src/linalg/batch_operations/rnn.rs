//! RNN and LSTM operations for neural networks
//!
//! This module contains implementations of recurrent neural network operations,
//! including LSTM cells and related functions.

use ndarray::{Array, Array2, ArrayView, ArrayView1, ArrayView2, Axis, Dimension};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{NeuralError, Result};

/// Type alias for LSTM forward return values
type LSTMForwardReturn<F> = (
    Array2<F>,
    Array2<F>,
    (Array2<F>, Array2<F>, Array2<F>, Array2<F>, Array2<F>),
);

/// Type alias for Adam update return values  
type AdamUpdateReturn<F, D> = (Array<F, D>, Array<F, D>, Array<F, D>);

/// Performs an LSTM cell forward computation.
///
/// # Arguments
///
/// * `x` - Input tensor with shape [batch_size, input_size]
/// * `h_prev` - Previous hidden state with shape [batch_size, hidden_size]
/// * `c_prev` - Previous cell state with shape [batch_size, hidden_size]
/// * `w_ih` - Input-to-hidden weights with shape [4*hidden_size, input_size]
/// * `w_hh` - Hidden-to-hidden weights with shape [4*hidden_size, hidden_size]
/// * `b_ih` - Input-to-hidden bias with shape [4*hidden_size]
/// * `b_hh` - Hidden-to-hidden bias with shape [4*hidden_size]
///
/// # Returns
///
/// * Tuple of (h_next, c_next, cache) where:
///   - h_next: Next hidden state with shape [batch_size, hidden_size]
///   - c_next: Next cell state with shape [batch_size, hidden_size]
///   - cache: Cached values for backpropagation
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Array2};
/// use scirs2_neural::linalg::lstm_cell;
///
/// // Sample dimensions
/// let batch_size = 2;
/// let input_size = 3;
/// let hidden_size = 4;
///
/// // Initialize inputs and parameters
/// let x = Array::from_shape_fn((batch_size, input_size), |_| 0.1);
/// let h_prev = Array::from_shape_fn((batch_size, hidden_size), |_| 0.0);
/// let c_prev = Array::from_shape_fn((batch_size, hidden_size), |_| 0.0);
///
/// let w_ih = Array::from_shape_fn((4 * hidden_size, input_size), |_| 0.1);
/// let w_hh = Array::from_shape_fn((4 * hidden_size, hidden_size), |_| 0.1);
/// let b_ih = Array::from_shape_fn(4 * hidden_size, |_| 0.1);
/// let b_hh = Array::from_shape_fn(4 * hidden_size, |_| 0.1);
///
/// // Forward pass
/// let (h_next, c_next, _) = lstm_cell(
///     &x.view(), &h_prev.view(), &c_prev.view(),
///     &w_ih.view(), &w_hh.view(), &b_ih.view(), &b_hh.view()
/// ).unwrap();
///
/// assert_eq!(h_next.shape(), &[batch_size, hidden_size]);
/// assert_eq!(c_next.shape(), &[batch_size, hidden_size]);
/// ```
pub fn lstm_cell<F>(
    x: &ArrayView2<F>,
    h_prev: &ArrayView2<F>,
    c_prev: &ArrayView2<F>,
    w_ih: &ArrayView2<F>,
    w_hh: &ArrayView2<F>,
    b_ih: &ArrayView1<F>,
    b_hh: &ArrayView1<F>,
) -> Result<LSTMForwardReturn<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = x.shape()[0];
    let input_size = x.shape()[1];
    let hidden_size = h_prev.shape()[1];

    // Validate shapes
    if h_prev.shape()[0] != batch_size {
        return Err(NeuralError::ShapeMismatch(format!(
            "Hidden state batch size mismatch in lstm_cell: x batch_size={}, h_prev batch_size={}",
            batch_size,
            h_prev.shape()[0]
        )));
    }

    if c_prev.shape()[0] != batch_size || c_prev.shape()[1] != hidden_size {
        return Err(NeuralError::ShapeMismatch(format!(
            "Cell state shape mismatch in lstm_cell: c_prev shape {:?}, expected [{}, {}]",
            c_prev.shape(),
            batch_size,
            hidden_size
        )));
    }

    if w_ih.shape()[0] != 4 * hidden_size || w_ih.shape()[1] != input_size {
        return Err(NeuralError::ShapeMismatch(
            format!("Input-to-hidden weights shape mismatch in lstm_cell: w_ih shape {:?}, expected [{}, {}]",
                   w_ih.shape(), 4 * hidden_size, input_size)
        ));
    }

    if w_hh.shape()[0] != 4 * hidden_size || w_hh.shape()[1] != hidden_size {
        return Err(NeuralError::ShapeMismatch(
            format!("Hidden-to-hidden weights shape mismatch in lstm_cell: w_hh shape {:?}, expected [{}, {}]",
                   w_hh.shape(), 4 * hidden_size, hidden_size)
        ));
    }

    if b_ih.shape()[0] != 4 * hidden_size {
        return Err(NeuralError::ShapeMismatch(format!(
            "Input-to-hidden bias shape mismatch in lstm_cell: b_ih shape {:?}, expected [{}]",
            b_ih.shape(),
            4 * hidden_size
        )));
    }

    if b_hh.shape()[0] != 4 * hidden_size {
        return Err(NeuralError::ShapeMismatch(format!(
            "Hidden-to-hidden bias shape mismatch in lstm_cell: b_hh shape {:?}, expected [{}]",
            b_hh.shape(),
            4 * hidden_size
        )));
    }

    // Compute gates: input, forget, cell, output
    let mut gates = Array2::<F>::zeros((batch_size, 4 * hidden_size));

    // Compute w_ih * x + b_ih
    for b in 0..batch_size {
        for i in 0..(4 * hidden_size) {
            let mut sum = b_ih[i];

            for j in 0..input_size {
                sum = sum + w_ih[[i, j]] * x[[b, j]];
            }

            gates[[b, i]] = sum;
        }
    }

    // Add w_hh * h_prev + b_hh
    for b in 0..batch_size {
        for i in 0..(4 * hidden_size) {
            let mut sum = b_hh[i];

            for j in 0..hidden_size {
                sum = sum + w_hh[[i, j]] * h_prev[[b, j]];
            }

            gates[[b, i]] = gates[[b, i]] + sum;
        }
    }

    // Extract gates
    let mut i_gate = Array2::<F>::zeros((batch_size, hidden_size));
    let mut f_gate = Array2::<F>::zeros((batch_size, hidden_size));
    let mut g_gate = Array2::<F>::zeros((batch_size, hidden_size));
    let mut o_gate = Array2::<F>::zeros((batch_size, hidden_size));

    for b in 0..batch_size {
        for h in 0..hidden_size {
            // Input gate: sigmoid
            i_gate[[b, h]] = sigmoid(gates[[b, h]]);

            // Forget gate: sigmoid
            f_gate[[b, h]] = sigmoid(gates[[b, h + hidden_size]]);

            // Cell gate: tanh
            g_gate[[b, h]] = gates[[b, h + 2 * hidden_size]].tanh();

            // Output gate: sigmoid
            o_gate[[b, h]] = sigmoid(gates[[b, h + 3 * hidden_size]]);
        }
    }

    // Compute next cell state
    let mut c_next = Array2::<F>::zeros((batch_size, hidden_size));

    for b in 0..batch_size {
        for h in 0..hidden_size {
            c_next[[b, h]] = f_gate[[b, h]] * c_prev[[b, h]] + i_gate[[b, h]] * g_gate[[b, h]];
        }
    }

    // Compute next hidden state
    let mut h_next = Array2::<F>::zeros((batch_size, hidden_size));

    for b in 0..batch_size {
        for h in 0..hidden_size {
            h_next[[b, h]] = o_gate[[b, h]] * c_next[[b, h]].tanh();
        }
    }

    // Cache values for backward pass
    let cache = (i_gate, f_gate, g_gate, o_gate, c_next.clone());

    Ok((h_next, c_next, cache))
}

// Helper function for sigmoid activation
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

/// Performs dropout operation for regularizing neural networks.
///
/// Randomly sets a fraction of input units to 0 at each update during training time,
/// which helps prevent overfitting. During inference, no dropout is applied.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `dropout_rate` - Probability of setting a value to zero
/// * `rng` - Random number generator object
/// * `training` - Whether in training mode (true) or inference mode (false)
///
/// # Returns
///
/// * Tuple of (output, mask) where:
///   - output: Result after applying dropout
///   - mask: Binary mask used for dropout (1 for kept elements, 0 for dropped)
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2};
/// use rand::prelude::*;
/// use scirs2_neural::linalg::dropout;
///
/// // Create input tensor
/// let x = Array::from_shape_fn((3, 4), |_| 1.0);
///
/// // Apply dropout in training mode
/// let mut rng = StdRng::seed_from_u64(42);
/// let (y_train, mask) = dropout(&x.view(), 0.5, &mut rng, true).unwrap();
///
/// // Apply dropout in inference mode
/// let (y_test, _) = dropout(&x.view(), 0.5, &mut rng, false).unwrap();
///
/// // In inference mode, no elements should be dropped
/// assert_eq!(y_test, x);
/// ```
pub fn dropout<F, D, R>(
    x: &ArrayView<F, D>,
    dropout_rate: F,
    rng: &mut R,
    training: bool,
) -> Result<(Array<F, D>, Array<F, D>)>
where
    F: Float + Debug + std::fmt::Display,
    D: Dimension,
    R: rand::Rng,
{
    // Validate dropout rate
    if dropout_rate < F::zero() || dropout_rate >= F::one() {
        return Err(NeuralError::InvalidArgument(format!(
            "Dropout rate must be in [0, 1) range, got {}",
            dropout_rate
        )));
    }

    // Create mask and output arrays
    let mut mask = Array::ones(x.raw_dim());
    let mut output = x.to_owned();

    // Apply dropout only in training mode
    if training {
        let keep_prob = F::from(1.0).unwrap() - dropout_rate;
        let scale = F::from(1.0).unwrap() / keep_prob;

        // Generate random mask
        for val in mask.iter_mut() {
            let rand_val = F::from(rng.random::<f64>()).unwrap();
            if rand_val < dropout_rate {
                *val = F::from(0.0).unwrap();
            } else {
                *val = scale;
            }
        }

        // Apply mask
        for (o, m) in output.iter_mut().zip(mask.iter()) {
            *o = *o * *m;
        }
    }

    Ok((output, mask))
}

/// Computes the backward pass for dropout operation.
///
/// # Arguments
///
/// * `dout` - Gradient of loss with respect to dropout output
/// * `mask` - Dropout mask from forward pass
/// * `dropout_rate` - Dropout rate used in forward pass
///
/// # Returns
///
/// * Gradient with respect to input before dropout
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2};
/// use rand::prelude::*;
/// use scirs2_neural::linalg::{dropout, dropout_backward};
///
/// // Setup (similar to forward example)
/// let x = Array::from_shape_fn((3, 4), |_| 1.0);
/// let mut rng = StdRng::seed_from_u64(42);
/// let dropout_rate = 0.5;
///
/// // Forward pass
/// let (y, mask) = dropout(&x.view(), dropout_rate, &mut rng, true).unwrap();
///
/// // Gradient of loss with respect to dropout output
/// let dout = Array::from_shape_fn(x.raw_dim(), |_| 0.1);
///
/// // Backward pass
/// let dx = dropout_backward(&dout.view(), &mask.view(), dropout_rate).unwrap();
///
/// assert_eq!(dx.shape(), x.shape());
/// ```
pub fn dropout_backward<F, D>(
    dout: &ArrayView<F, D>,
    mask: &ArrayView<F, D>,
    dropout_rate: F,
) -> Result<Array<F, D>>
where
    F: Float + Debug + std::fmt::Display,
    D: Dimension,
{
    // Validate dropout rate
    if dropout_rate < F::zero() || dropout_rate >= F::one() {
        return Err(NeuralError::InvalidArgument(format!(
            "Dropout rate must be in [0, 1) range, got {}",
            dropout_rate
        )));
    }

    // Simply apply the mask to the gradient
    let mut dx = dout.to_owned();

    for (dx_val, mask_val) in dx.iter_mut().zip(mask.iter()) {
        *dx_val = *dx_val * *mask_val;
    }

    Ok(dx)
}

/// Computes log-softmax function for numerical stability.
///
/// Calculates log(softmax(x)) in a numerically stable way by subtracting the max
/// value before exponentiating.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `dim` - Dimension along which to compute softmax
///
/// # Returns
///
/// * Log-softmax values with the same shape as input
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2, Axis};
/// use scirs2_neural::linalg::log_softmax;
///
/// // Create sample input
/// let x = Array::from_shape_vec(
///     (2, 3),
///     vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).unwrap();
///
/// // Compute log-softmax along the last dimension (axis 1)
/// let log_sm = log_softmax(&x.view(), 1).unwrap();
///
/// assert_eq!(log_sm.shape(), x.shape());
///
/// // Check that exp(log_softmax) sums to 1 along the specified dimension
/// for row in 0..2 {
///     let mut sum = 0.0f64;
///     for col in 0..3 {
///         let val: f64 = log_sm[[row, col]];
///         sum += val.exp();
///     }
///     assert!((sum - 1.0).abs() < 1e-5);
/// }
/// ```
pub fn log_softmax<F, D>(x: &ArrayView<F, D>, dim: usize) -> Result<Array<F, D>>
where
    F: Float + Debug + std::fmt::Display,
    D: Dimension,
{
    if dim >= x.ndim() {
        return Err(NeuralError::InvalidArgument(format!(
            "Dimension out of bounds in log_softmax: dim={}, ndim={}",
            dim,
            x.ndim()
        )));
    }

    let mut output = x.to_owned();

    // Handle each slice along the specified dimension
    for mut slice in output.lanes_mut(Axis(dim)) {
        // Find max value for numerical stability
        let max_val = *slice
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Subtract max and compute exp
        let mut sum_exp = F::zero();
        for val in slice.iter_mut() {
            *val = (*val - max_val).exp();
            sum_exp = sum_exp + *val;
        }

        // Convert to log-softmax
        for val in slice.iter_mut() {
            *val = (*val / sum_exp).ln();
        }
    }

    Ok(output)
}

/// Performs weight update using the Adam optimizer.
///
/// Adam (Adaptive Moment Estimation) is an optimization algorithm that combines
/// the benefits of AdaGrad and RMSProp.
///
/// # Arguments
///
/// * `w` - Parameters to update
/// * `dw` - Gradient of loss with respect to parameters
/// * `m` - First moment vector (momentum)
/// * `v` - Second moment vector (velocity)
/// * `learning_rate` - Learning rate
/// * `beta1` - Exponential decay rate for first moment (typically 0.9)
/// * `beta2` - Exponential decay rate for second moment (typically 0.999)
/// * `epsilon` - Small constant for numerical stability
/// * `t` - Iteration count
///
/// # Returns
///
/// * Tuple of (updated_w, updated_m, updated_v) containing:
///   - updated_w: Updated parameters
///   - updated_m: Updated first moment
///   - updated_v: Updated second moment
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Ix1};
/// use scirs2_neural::linalg::adam_update;
///
/// // Setup parameters and gradients
/// let w = Array::from_vec(vec![1.0, 2.0, 3.0]);
/// let dw = Array::from_vec(vec![0.1, 0.2, 0.3]);
/// let mut m = Array::zeros(3);
/// let mut v = Array::zeros(3);
///
/// // Optimizer settings
/// let learning_rate = 0.001;
/// let beta1 = 0.9;
/// let beta2 = 0.999;
/// let epsilon = 1e-8;
/// let t = 1;
///
/// // Update parameters
/// let (w_new, m_new, v_new) = adam_update(
///     &w.view(), &dw.view(), &m.view(), &v.view(),
///     learning_rate, beta1, beta2, epsilon, t
/// ).unwrap();
///
/// assert_eq!(w_new.shape(), w.shape());
/// assert_eq!(m_new.shape(), m.shape());
/// assert_eq!(v_new.shape(), v.shape());
/// ```
#[allow(clippy::too_many_arguments)]
pub fn adam_update<F, D>(
    w: &ArrayView<F, D>,
    dw: &ArrayView<F, D>,
    m: &ArrayView<F, D>,
    v: &ArrayView<F, D>,
    learning_rate: F,
    beta1: F,
    beta2: F,
    epsilon: F,
    t: usize,
) -> Result<AdamUpdateReturn<F, D>>
where
    F: Float + Debug + std::fmt::Display,
    D: Dimension,
{
    // Validate shapes
    if w.shape() != dw.shape() || w.shape() != m.shape() || w.shape() != v.shape() {
        return Err(NeuralError::ShapeMismatch(
            format!("Shape mismatch in adam_update: w shape {:?}, dw shape {:?}, m shape {:?}, v shape {:?}",
                   w.shape(), dw.shape(), m.shape(), v.shape())
        ));
    }

    // Validate hyperparameters
    if learning_rate <= F::zero() {
        return Err(NeuralError::InvalidArgument(format!(
            "Learning rate must be positive in adam_update, got {}",
            learning_rate
        )));
    }

    if beta1 < F::zero() || beta1 >= F::one() {
        return Err(NeuralError::InvalidArgument(format!(
            "beta1 must be in [0, 1) range in adam_update, got {}",
            beta1
        )));
    }

    if beta2 < F::zero() || beta2 >= F::one() {
        return Err(NeuralError::InvalidArgument(format!(
            "beta2 must be in [0, 1) range in adam_update, got {}",
            beta2
        )));
    }

    if epsilon <= F::zero() {
        return Err(NeuralError::InvalidArgument(format!(
            "epsilon must be positive in adam_update, got {}",
            epsilon
        )));
    }

    // Initialize output arrays
    let mut w_new = w.to_owned();
    let mut m_new = Array::zeros(w.raw_dim());
    let mut v_new = Array::zeros(w.raw_dim());

    // Update biased first moment estimate
    for (m_val, m_prev, dw_val) in zip3(m_new.iter_mut(), m.iter(), dw.iter()) {
        *m_val = beta1 * *m_prev + (F::one() - beta1) * *dw_val;
    }

    // Update biased second moment estimate
    for (v_val, v_prev, dw_val) in zip3(v_new.iter_mut(), v.iter(), dw.iter()) {
        *v_val = beta2 * *v_prev + (F::one() - beta2) * (*dw_val * *dw_val);
    }

    // Bias correction
    let t_f = F::from(t).unwrap();
    let m_hat_factor = F::one() / (F::one() - beta1.powf(t_f));
    let v_hat_factor = F::one() / (F::one() - beta2.powf(t_f));

    // Update parameters
    for (w_val, m_val, v_val) in zip3(w_new.iter_mut(), m_new.iter(), v_new.iter()) {
        let m_hat = *m_val * m_hat_factor;
        let v_hat = *v_val * v_hat_factor;

        // Update rule
        *w_val = *w_val - learning_rate * m_hat / (v_hat.sqrt() + epsilon);
    }

    Ok((w_new, m_new, v_new))
}

// Helper function to zip three iterators for convenience
fn zip3<I1, I2, I3>(i1: I1, i2: I2, i3: I3) -> impl Iterator<Item = (I1::Item, I2::Item, I3::Item)>
where
    I1: IntoIterator,
    I2: IntoIterator,
    I3: IntoIterator,
{
    i1.into_iter()
        .zip(i2.into_iter().zip(i3))
        .map(|(a, (b, c))| (a, b, c))
}
