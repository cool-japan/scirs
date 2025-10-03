//! Normalization operations for batch processing in neural networks
//!
//! This module contains functions for various normalization techniques used in neural
//! networks, including batch normalization and layer normalization.

use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, Dimension};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use crate::error::{NeuralError, Result};
/// Type alias for batch normalization forward return values  
type BatchNormForwardReturn<F> = (
    Array2<F>,
    Option<(Array2<F>, Array1<F>, Array1<F>, Array1<F>, F)>,
);
/// Type alias for layer normalization forward return values
type LayerNormForwardReturn<F> = (Array2<F>, (Array2<F>, Array1<F>, Array1<F>, Array1<F>, F));
/// Performs batch normalization forward pass.
///
/// Normalizes input across batch dimension for each feature.
/// # Arguments
/// * `x` - Input data with shape [batch_size, features]
/// * `gamma` - Scale parameter with shape [features]
/// * `beta` - Shift parameter with shape [features]
/// * `eps` - Small constant for numerical stability
/// * `momentum` - Momentum factor for running mean/var updates (not used in inference)
/// * `running_mean` - Running mean with shape [features], updated during training
/// * `running_var` - Running variance with shape [features], updated during training
/// * `training` - Whether in training mode (true) or inference mode (false)
/// # Returns
/// * Tuple of (normalized_output, cache) where:
///   - normalized_output has the same shape as input
///   - cache contains intermediate values needed for backward pass (if in training mode)
/// # Examples
/// ```
/// use scirs2_core::ndarray::{Array, Array1, Array2};
/// use scirs2_neural::linalg::batch_norm_forward;
/// // Sample input: batch_size=3, features=2
/// let x = Array::from_shape_vec(
///     (3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).unwrap();
/// // Parameters
/// let gamma = Array::from_vec(vec![1.0, 1.0]);
/// let beta = Array::from_vec(vec![0.0, 0.0]);
/// let eps = 1e-5;
/// let momentum = 0.9;
/// let mut running_mean = Array::zeros(2);
/// let mut running_var = Array::ones(2);
/// // Forward pass (training mode)
/// let (out, cache) = batch_norm_forward(
///     &x.view(),
///     &gamma.view(),
///     &beta.view(),
///     eps,
///     momentum,
///     &mut running_mean,
///     &mut running_var,
///     true
/// assert_eq!(out.shape(), x.shape());
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn batch_norm_forward<F>(
    x: &ArrayView2<F>,
    gamma: &ArrayView1<F>,
    beta: &ArrayView1<F>,
    eps: F,
    momentum: F,
    running_mean: &mut Array1<F>,
    running_var: &mut Array1<F>,
    training: bool,
) -> Result<BatchNormForwardReturn<F>>
where
    F: Float + Debug + scirs2_core::numeric::FromPrimitive,
{
    let batch_size = x.shape()[0];
    let num_features = x.shape()[1];
    if gamma.shape()[0] != num_features || beta.shape()[0] != num_features {
        return Err(NeuralError::ShapeMismatch(
            format!("Parameter shape mismatch in batch_normforward: x shape {:?}, gamma shape {:?}, beta shape {:?}",
                   x.shape(), gamma.shape(), beta.shape())
        ));
    }
    if running_mean.shape()[0] != num_features || running_var.shape()[0] != num_features {
            format!("Running stats shape mismatch in batch_normforward: x shape {:?}, running_mean shape {:?}, running_var shape {:?}",
                   x.shape(), running_mean.shape(), running_var.shape())
    let mut out = Array2::<F>::zeros(x.raw_dim());
    if training {
        // Compute mean and variance for this batch
        let batch_mean = x.mean_axis(Axis(0)).unwrap();
        // Compute variance
        let mut batch_var = Array1::<F>::zeros(num_features);
        for i in 0..batch_size {
            for j in 0..num_features {
                let diff = x[[i, j]] - batch_mean[j];
                batch_var[j] = batch_var[j] + diff * diff;
            }
        }
        batch_var.mapv_inplace(|v| v / F::from(batch_size).unwrap());
        // Update running mean and variance
        for j in 0..num_features {
            running_mean[j] = momentum * running_mean[j] + (F::one() - momentum) * batch_mean[j];
            running_var[j] = momentum * running_var[j] + (F::one() - momentum) * batch_var[j];
        // Normalize
        let mut x_hat = Array2::<F>::zeros(x.raw_dim());
                x_hat[[i, j]] = (x[[i, j]] - batch_mean[j]) / (batch_var[j] + eps).sqrt();
                out[[i, j]] = gamma[j] * x_hat[[i, j]] + beta[j];
        // Return output and cache for backward pass
        let cache = (x_hat, batch_mean, batch_var, gamma.to_owned(), eps);
        Ok((out, Some(cache)))
    } else {
        // In inference mode, use running mean and variance
                let x_hat = (x[[i, j]] - running_mean[j]) / (running_var[j] + eps).sqrt();
                out[[i, j]] = gamma[j] * x_hat + beta[j];
        // No need for cache in inference mode
        Ok((out, None))
}
/// Computes the backward pass for batch normalization.
/// * `dout` - Gradient of loss with respect to batch norm output, shape [batch_size, features]
/// * `cache` - Cache from forward pass, contains:
///   - x_hat: Normalized inputs before scaling and shifting, shape [batch_size, features]
///   - batch_mean: Mean of each feature across batch, shape [features]
///   - batch_var: Variance of each feature across batch, shape [features]
///   - gamma: Scale parameter, shape [features]
///   - eps: Epsilon value used in forward pass
/// * Tuple of (dx, dgamma, dbeta) where:
///   - dx: Gradient with respect to input x, shape [batch_size, features]
///   - dgamma: Gradient with respect to gamma, shape [features]
///   - dbeta: Gradient with respect to beta, shape [features]
/// use scirs2_neural::linalg::{batch_norm_forward, batch_norm_backward};
/// // Setup (same as forward example)
/// let x = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// // Forward pass to get cache
/// let (out, cache_opt) = batch_norm_forward(
///     &x.view(), &gamma.view(), &beta.view(), eps, momentum,
///     &mut running_mean, &mut running_var, true
/// // Assume gradient of loss with respect to output
/// let dout = Array::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
/// // Backward pass
/// if let Some(cache) = cache_opt {
///     let (dx, dgamma, dbeta) = batch_norm_backward(&dout.view(), &cache).unwrap();
///     assert_eq!(dx.shape(), x.shape());
///     assert_eq!(dgamma.shape(), gamma.shape());
///     assert_eq!(dbeta.shape(), beta.shape());
/// }
#[allow(dead_code)]
pub fn batch_norm_backward<F>(
    dout: &ArrayView2<F>,
    cache: &(Array2<F>, Array1<F>, Array1<F>, Array1<F>, F),
) -> Result<(Array2<F>, Array1<F>, Array1<F>)>
    let (x_hat_batch_mean, batch_var, gamma, eps) = cache;
    let batch_size = dout.shape()[0];
    let num_features = dout.shape()[1];
    if x_hat.shape() != dout.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "Shape mismatch in batch_norm_backward: dout shape {:?}, x_hat shape {:?}",
            dout.shape(),
            x_hat.shape()
        )));
    // Initialize gradients
    let mut dx = Array2::<F>::zeros(dout.raw_dim());
    let mut dgamma = Array1::<F>::zeros(gamma.raw_dim());
    let mut dbeta = Array1::<F>::zeros(gamma.raw_dim());
    // Gradient with respect to beta: sum over batch dimension
    for j in 0..num_features {
            dbeta[j] = dbeta[j] + dout[[i, j]];
    // Gradient with respect to gamma: sum over batch dimension of dout * x_hat
            dgamma[j] = dgamma[j] + dout[[i, j]] * x_hat[[i, j]];
    // Gradient with respect to x_hat
    let mut dx_hat = Array2::<F>::zeros(dout.raw_dim());
    for i in 0..batch_size {
            dx_hat[[i, j]] = dout[[i, j]] * gamma[j];
    // Gradient with respect to input x
    let batch_size_f = F::from(batch_size).unwrap();
        let std_inv = F::one() / (batch_var[j] + *eps).sqrt();
        // Compute sum terms for the dx equation
        let mut sum_dx_hat = F::zero();
        let mut sum_dx_hat_x_hat = F::zero();
            sum_dx_hat = sum_dx_hat + dx_hat[[i, j]];
            sum_dx_hat_x_hat = sum_dx_hat_x_hat + dx_hat[[i, j]] * x_hat[[i, j]];
        // Apply the batch normalization backward formula for each input
            dx[[i, j]] =
                dx_hat[[i, j]] - (sum_dx_hat + x_hat[[i, j]] * sum_dx_hat_x_hat) / batch_size_f;
            dx[[i, j]] = dx[[i, j]] * std_inv;
    Ok((dx, dgamma, dbeta))
/// Performs layer normalization forward pass.
/// Normalizes input across feature dimension for each sample in the batch.
///   - cache contains intermediate values needed for backward pass
/// use scirs2_neural::linalg::layer_norm;
/// // Sample input: batch_size=3, features=4
///     (3, 4),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// let gamma = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
/// let beta = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
/// // Forward pass
/// let (out_) = layer_norm(&x.view(), &gamma.view(), &beta.view(), eps).unwrap();
#[allow(dead_code)]
pub fn layer_norm<F>(
) -> Result<LayerNormForwardReturn<F>>
            format!("Parameter shape mismatch in layernorm: x shape {:?}, gamma shape {:?}, beta shape {:?}",
    let mut x_hat = Array2::<F>::zeros(x.raw_dim());
    let mut mean = Array1::<F>::zeros(batch_size);
    let mut var = Array1::<F>::zeros(batch_size);
    // Compute mean and variance for each sample
        // Compute mean for this sample
        mean[i] = F::zero();
            mean[i] = mean[i] + x[[i, j]];
        mean[i] = mean[i] / F::from(num_features).unwrap();
        // Compute variance for this sample
        var[i] = F::zero();
            let diff = x[[i, j]] - mean[i];
            var[i] = var[i] + diff * diff;
        var[i] = var[i] / F::from(num_features).unwrap();
        // Normalize, scale, and shift
            x_hat[[i, j]] = (x[[i, j]] - mean[i]) / (var[i] + eps).sqrt();
            out[[i, j]] = gamma[j] * x_hat[[i, j]] + beta[j];
    // Return output and cache for backward pass
    let cache = (x_hat, mean, var, gamma.to_owned(), eps);
    Ok((out, cache))
/// Computes the backward pass for layer normalization.
/// * `dout` - Gradient of loss with respect to layer norm output, shape [batch_size, features]
///   - mean: Mean of each sample across features, shape [batch_size]
///   - var: Variance of each sample across features, shape [batch_size]
/// use scirs2_neural::linalg::{layer_norm, layer_norm_backward};
/// let (out, cache) = layer_norm(&x.view(), &gamma.view(), &beta.view(), eps).unwrap();
/// let dout = Array::from_shape_vec(
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
/// let (dx, dgamma, dbeta) = layer_norm_backward(&dout.view(), &cache).unwrap();
/// assert_eq!(dx.shape(), x.shape());
/// assert_eq!(dgamma.shape(), gamma.shape());
/// assert_eq!(dbeta.shape(), beta.shape());
#[allow(dead_code)]
pub fn layer_norm_backward<F>(
    let (x_hat_mean, var, gamma, eps) = cache;
            "Shape mismatch in layer_norm_backward: dout shape {:?}, x_hat shape {:?}",
    let num_features_f = F::from(num_features).unwrap();
        let std_inv = F::one() / (var[i] + *eps).sqrt();
        // Apply the layer normalization backward formula for each input
                dx_hat[[i, j]] - (sum_dx_hat + x_hat[[i, j]] * sum_dx_hat_x_hat) / num_features_f;
/// Clips gradient norms to prevent exploding gradients.
/// This function is commonly used in recurrent neural networks and
/// transformer models to stabilize training.
/// * `grad` - Gradient values to clip
/// * `max_norm` - Maximum allowed norm
/// * Clipped gradient values with the same shape as input
/// use scirs2_core::ndarray::{Array, Ix2};
/// use scirs2_neural::linalg::clip_grad_norm;
/// // Create a gradient with large values
/// let grad = Array::from_shape_vec(
///     (2, 3),
///     vec![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0]
/// // Clip gradient to max norm of 10.0
/// let clipped_grad = clip_grad_norm(&grad.view(), 10.0).unwrap();
/// assert_eq!(clipped_grad.shape(), grad.shape());
/// // Verify norm is reduced
/// let norm = clipped_grad.iter().map(|&v: &f64| v.powi(2)).sum::<f64>().sqrt();
/// assert!(norm <= 10.0 + 1e-5); // Allow small numerical error
#[allow(dead_code)]
pub fn clip_grad_norm<F, D>(_grad: &ArrayView<F, D>, maxnorm: F) -> Result<Array<F, D>>
    D: Dimension,
    let mut grad_squared_sum = F::zero();
    // Compute sum of squared gradients
    for &g in grad.iter() {
        grad_squared_sum = grad_squared_sum + g * g;
    let grad_norm = grad_squared_sum.sqrt();
    // If norm is smaller than max_norm, return original gradient
    if grad_norm <= max_norm {
        return Ok(grad.to_owned());
    // Otherwise, scale the gradient
    let scale = max_norm / grad_norm;
    // Apply scaling
    let mut clipped_grad = grad.to_owned();
    for g in clipped_grad.iter_mut() {
        *g = *g * scale;
    Ok(clipped_grad)
