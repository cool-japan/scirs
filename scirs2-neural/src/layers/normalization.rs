//! Normalization layers implementation
//!
//! This module provides implementations of various normalization techniques
//! such as Layer Normalization, Batch Normalization, etc.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, ArrayView1, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Layer Normalization layer
///
/// Implements layer normalization as described in "Layer Normalization"
/// by Ba, Kiros, and Hinton. It normalizes the inputs across the last dimension
/// and applies learnable scale and shift parameters.
#[derive(Debug)]
pub struct LayerNorm<F: Float + Debug + Send + Sync + NumAssign>
where
    F: SimdUnifiedOps,
{
    /// Dimensionality of the input features
    normalizedshape: Vec<usize>,
    /// Learnable scale parameter
    gamma: Array<F, IxDyn>,
    /// Learnable shift parameter
    beta: Array<F, IxDyn>,
    /// Gradient of gamma
    dgamma: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of beta
    dbeta: Arc<RwLock<Array<F, IxDyn>>>,
    /// Small constant for numerical stability
    eps: F,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cache of the standardized input `x_hat` (before the gamma/beta affine),
    /// shaped `[batch, features]`; required by the backward pass
    xhat_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Mean cache for backward pass
    mean_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Variance cache for backward pass
    var_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> Clone for LayerNorm<F>
where
    F: SimdUnifiedOps,
{
    fn clone(&self) -> Self {
        let input_cache_clone = match self.input_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };
        let xhat_cache_clone = match self.xhat_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };
        let mean_cache_clone = match self.mean_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };
        let var_cache_clone = match self.var_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };

        Self {
            normalizedshape: self.normalizedshape.clone(),
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            dgamma: Arc::new(RwLock::new(
                self.dgamma.read().expect("Operation failed").clone(),
            )),
            dbeta: Arc::new(RwLock::new(
                self.dbeta.read().expect("Operation failed").clone(),
            )),
            eps: self.eps,
            input_cache: Arc::new(RwLock::new(input_cache_clone)),
            xhat_cache: Arc::new(RwLock::new(xhat_cache_clone)),
            mean_cache: Arc::new(RwLock::new(mean_cache_clone)),
            var_cache: Arc::new(RwLock::new(var_cache_clone)),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> LayerNorm<F>
where
    F: SimdUnifiedOps,
{
    /// Create a new layer normalization layer
    pub fn new<R: Rng>(normalizedshape: usize, eps: f64, _rng: &mut R) -> Result<Self> {
        let gamma = Array::<F, IxDyn>::from_elem(IxDyn(&[normalizedshape]), F::one());
        let beta = Array::<F, IxDyn>::from_elem(IxDyn(&[normalizedshape]), F::zero());

        let dgamma = Arc::new(RwLock::new(Array::<F, IxDyn>::zeros(IxDyn(&[
            normalizedshape,
        ]))));
        let dbeta = Arc::new(RwLock::new(Array::<F, IxDyn>::zeros(IxDyn(&[
            normalizedshape,
        ]))));

        let eps = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert epsilon to type F".to_string())
        })?;

        Ok(Self {
            normalizedshape: vec![normalizedshape],
            gamma,
            beta,
            dgamma,
            dbeta,
            eps,
            input_cache: Arc::new(RwLock::new(None)),
            xhat_cache: Arc::new(RwLock::new(None)),
            mean_cache: Arc::new(RwLock::new(None)),
            var_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the normalized shape
    pub fn normalizedshape(&self) -> usize {
        self.normalizedshape[0]
    }

    /// Get the epsilon value
    #[allow(dead_code)]
    pub fn eps(&self) -> f64 {
        self.eps.to_f64().unwrap_or(1e-5)
    }
}

/// Threshold for using SIMD-accelerated LayerNorm
const LAYERNORM_SIMD_THRESHOLD: usize = 64;

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign> Layer<F>
    for LayerNorm<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        let inputshape = input.shape();
        let ndim = input.ndim();

        if ndim < 1 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 1 dimension".to_string(),
            ));
        }

        let feat_dim = inputshape[ndim - 1];
        if feat_dim != self.normalizedshape[0] {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Last dimension of input ({}) must match normalizedshape ({})",
                feat_dim, self.normalizedshape[0]
            )));
        }

        let batchshape: Vec<usize> = inputshape[..ndim - 1].to_vec();
        let batch_size: usize = batchshape.iter().product();

        // Reshape input to 2D: [batch_size, features]
        let reshaped = input
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch_size, feat_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?;

        // Compute mean and variance for each sample
        let mut mean = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, 1]));
        let mut var = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, 1]));

        // Use SIMD-accelerated path for larger feature dimensions (Phase 36+ optimization)
        if feat_dim >= LAYERNORM_SIMD_THRESHOLD {
            // SIMD path: use simd_mean for mean and simd_sum for variance
            for i in 0..batch_size {
                // Extract row as 1D view for SIMD operations
                let row_slice = reshaped.slice(scirs2_core::ndarray::s![i, ..]);
                let row_view: ArrayView1<F> =
                    row_slice.into_dimensionality().expect("Operation failed");

                // SIMD-accelerated mean computation
                let row_mean = F::simd_mean(&row_view);
                mean[[i, 0]] = row_mean;

                // SIMD-accelerated variance computation
                // variance = E[(x - mean)^2] = E[x^2] - mean^2
                // Using simd_dot for sum of squares
                let sum_sq = F::simd_dot(&row_view, &row_view);
                let mean_sq = row_mean * row_mean;
                let n = F::from(feat_dim).expect("Failed to convert to float");
                var[[i, 0]] = sum_sq / n - mean_sq;
            }
        } else {
            // Scalar fallback for small feature dimensions
            for i in 0..batch_size {
                let mut sum = F::zero();
                for j in 0..feat_dim {
                    sum += reshaped[[i, j]];
                }
                mean[[i, 0]] = sum / F::from(feat_dim).expect("Failed to convert to float");

                let mut sum_sq = F::zero();
                for j in 0..feat_dim {
                    let diff = reshaped[[i, j]] - mean[[i, 0]];
                    sum_sq += diff * diff;
                }
                var[[i, 0]] = sum_sq / F::from(feat_dim).expect("Failed to convert to float");
            }
        }

        // Cache mean and variance
        if let Ok(mut cache) = self.mean_cache.write() {
            *cache = Some(mean.clone());
        }
        if let Ok(mut cache) = self.var_cache.write() {
            *cache = Some(var.clone());
        }

        // Normalize and apply gamma/beta
        // Using SIMD for larger dimensions
        let mut xhat = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, feat_dim]));
        let mut normalized = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, feat_dim]));
        for i in 0..batch_size {
            let inv_std = (var[[i, 0]] + self.eps).sqrt().recip();
            let mean_i = mean[[i, 0]];

            for j in 0..feat_dim {
                let x_norm = (reshaped[[i, j]] - mean_i) * inv_std;
                xhat[[i, j]] = x_norm;
                normalized[[i, j]] = x_norm * self.gamma[[j]] + self.beta[[j]];
            }
        }

        // Cache the standardized input (pre-affine) for the backward pass
        if let Ok(mut cache) = self.xhat_cache.write() {
            *cache = Some(xhat);
        }

        // Reshape back to original shape
        let output = normalized
            .into_shape_with_order(IxDyn(inputshape))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {e}")))?;

        Ok(output)
    }

    /// Exact gradient of layer normalization.
    ///
    /// With `y_j = gamma_j * x_hat_j + beta_j` and
    /// `x_hat_j = (x_j - mean) / sqrt(var + eps)` over the last axis:
    ///
    /// ```text
    /// dgamma_j = sum_batch dy_j * x_hat_j
    /// dbeta_j  = sum_batch dy_j
    /// dx_j     = (dxhat_j - mean_k(dxhat_k) - x_hat_j * mean_k(dxhat_k x_hat_k)) / sqrt(var + eps)
    /// ```
    ///
    /// where `dxhat_j = dy_j * gamma_j`. Parameter gradients are stored for
    /// [`Layer::update`] and [`Layer::gradients`].
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let xhat_guard = self.xhat_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on xhat cache".to_string())
        })?;
        let var_guard = self.var_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on variance cache".to_string())
        })?;
        let missing = || {
            NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            )
        };
        let xhat = xhat_guard.as_ref().ok_or_else(missing)?;
        let var = var_guard.as_ref().ok_or_else(missing)?;

        let feat_dim = self.normalizedshape[0];
        let outshape = grad_output.shape().to_vec();
        let ndim = outshape.len();
        if ndim < 1 || outshape[ndim - 1] != feat_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient last dimension must be {feat_dim}, got {outshape:?}"
            )));
        }
        let batch_size: usize = outshape[..ndim - 1].iter().product();
        if xhat.shape() != [batch_size, feat_dim] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Cached activations have shape {:?} but the gradient implies [{batch_size}, {feat_dim}]",
                xhat.shape()
            )));
        }

        let grad2d = grad_output
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch_size, feat_dim]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape output gradient: {e}"))
            })?;

        let n = F::from(feat_dim).ok_or_else(|| {
            NeuralError::InferenceError("Failed to convert feature count".to_string())
        })?;

        let mut dgamma = Array::<F, IxDyn>::zeros(IxDyn(&[feat_dim]));
        let mut dbeta = Array::<F, IxDyn>::zeros(IxDyn(&[feat_dim]));
        let mut grad_input = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, feat_dim]));

        for i in 0..batch_size {
            let inv_std = (var[[i, 0]] + self.eps).sqrt().recip();

            let mut sum_dxhat = F::zero();
            let mut sum_dxhat_xhat = F::zero();
            for j in 0..feat_dim {
                let dy = grad2d[[i, j]];
                let xh = xhat[[i, j]];
                dgamma[j] += dy * xh;
                dbeta[j] += dy;
                let dxhat = dy * self.gamma[[j]];
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * xh;
            }

            for j in 0..feat_dim {
                let dxhat = grad2d[[i, j]] * self.gamma[[j]];
                grad_input[[i, j]] =
                    (dxhat - sum_dxhat / n - xhat[[i, j]] * sum_dxhat_xhat / n) * inv_std;
            }
        }

        if let Ok(mut cache) = self.dgamma.write() {
            *cache = dgamma;
        }
        if let Ok(mut cache) = self.dbeta.write() {
            *cache = dbeta;
        }

        grad_input
            .into_shape_with_order(IxDyn(&outshape))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape input gradient: {e}"))
            })
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        let dgamma = self
            .dgamma
            .read()
            .map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on dgamma".to_string())
            })?
            .clone();
        let dbeta = self
            .dbeta
            .read()
            .map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on dbeta".to_string())
            })?
            .clone();

        for (param, grad) in [(&mut self.gamma, &dgamma), (&mut self.beta, &dbeta)] {
            if param.shape() != grad.shape() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Parameter shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
            scirs2_core::ndarray::Zip::from(&mut *param)
                .and(grad)
                .for_each(|w, &g| *w -= learningrate * g);
        }
        Ok(())
    }

    fn gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        ParamLayer::get_gradients(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn parameter_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn params(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_params(&mut self, params: &[Array<F, scirs2_core::ndarray::IxDyn>]) -> Result<()> {
        if params.len() >= 2 {
            self.gamma = params[0].clone();
            self.beta = params[1].clone();
        } else if params.len() == 1 {
            self.gamma = params[0].clone();
        }
        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static + NumAssign>
    ParamLayer<F> for LayerNorm<F>
{
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    /// Gradients of `[gamma, beta]`; zero until `backward` has run.
    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        let read = |cell: &Arc<RwLock<Array<F, IxDyn>>>| match cell.read() {
            Ok(guard) => guard.clone(),
            Err(_) => Array::zeros(IxDyn(&[0])),
        };
        vec![read(&self.dgamma), read(&self.dbeta)]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        if params[0].shape() != self.gamma.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Gamma shape mismatch: expected {:?}, got {:?}",
                self.gamma.shape(),
                params[0].shape()
            )));
        }

        if params[1].shape() != self.beta.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Beta shape mismatch: expected {:?}, got {:?}",
                self.beta.shape(),
                params[1].shape()
            )));
        }

        self.gamma = params[0].clone();
        self.beta = params[1].clone();

        Ok(())
    }
}

/// Batch Normalization layer
///
/// Implements batch normalization as described in "Batch Normalization:
/// Accelerating Deep Network Training by Reducing Internal Covariate Shift"
/// (Ioffe & Szegedy, 2015).
///
/// For every channel `c`, the layer normalizes over the batch dimension and
/// any trailing spatial dimensions (i.e. over all `(n, s)` for input shape
/// `[N, C, ...]`), then applies a learnable affine transform:
///
/// ```text
/// mean_c = mean_{n,s}(x[n, c, s])
/// var_c  = mean_{n,s}((x[n, c, s] - mean_c)^2)
/// x_hat[n, c, s] = (x[n, c, s] - mean_c) / sqrt(var_c + eps)
/// y[n, c, s]     = gamma_c * x_hat[n, c, s] + beta_c
/// ```
///
/// In training mode the batch statistics above are used, and an exponential
/// moving average (the "running" mean/variance) is updated for later use at
/// inference time. In evaluation mode (`set_training(false)`) the running
/// statistics are used directly instead of recomputing batch statistics,
/// matching standard BatchNorm semantics.
///
/// # Input shape
/// `[N, C, ...]` — batch, channel, then zero or more trailing spatial
/// dimensions (e.g. `[N, C]` for dense features or `[N, C, H, W]` for
/// images).
#[derive(Debug)]
pub struct BatchNorm<F: Float + Debug + Send + Sync + NumAssign> {
    /// Number of features (channels)
    num_features: usize,
    /// Learnable scale parameter, shape `[C]`
    gamma: Array<F, IxDyn>,
    /// Learnable shift parameter, shape `[C]`
    beta: Array<F, IxDyn>,
    /// Small constant for numerical stability
    eps: F,
    /// Momentum for the running-statistics exponential moving average (the
    /// weight given to the newly observed batch statistic; PyTorch
    /// convention, typically `0.1`)
    momentum: F,
    /// Whether we're in training mode
    training: bool,
    /// Running mean, shape `[C]`; used for normalization in evaluation mode
    running_mean: Arc<RwLock<Array<F, IxDyn>>>,
    /// Running (unbiased) variance, shape `[C]`; used for normalization in
    /// evaluation mode
    running_var: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of gamma, shape `[C]`
    dgamma: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of beta, shape `[C]`
    dbeta: Arc<RwLock<Array<F, IxDyn>>>,
    /// Cache of the standardized `x_hat` from the last forward pass, shaped
    /// `[N, C, S]` (`S` = product of trailing spatial dims)
    xhat_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cache of `1 / sqrt(var + eps)` per channel from the last forward pass
    inv_std_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Original input shape from the last forward pass, for reshaping the
    /// gradient back
    inputshape_cache: Arc<RwLock<Option<Vec<usize>>>>,
    /// Whether the cached forward pass used batch statistics (training) or
    /// running statistics (evaluation); the backward formula differs.
    forward_was_training: Arc<RwLock<bool>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> Clone for BatchNorm<F> {
    fn clone(&self) -> Self {
        let clone_arr = |cell: &Arc<RwLock<Array<F, IxDyn>>>| -> Array<F, IxDyn> {
            cell.read()
                .map(|g| g.clone())
                .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.num_features])))
        };
        Self {
            num_features: self.num_features,
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            eps: self.eps,
            momentum: self.momentum,
            training: self.training,
            running_mean: Arc::new(RwLock::new(clone_arr(&self.running_mean))),
            running_var: Arc::new(RwLock::new(clone_arr(&self.running_var))),
            dgamma: Arc::new(RwLock::new(clone_arr(&self.dgamma))),
            dbeta: Arc::new(RwLock::new(clone_arr(&self.dbeta))),
            xhat_cache: Arc::new(RwLock::new(
                self.xhat_cache.read().map(|c| c.clone()).unwrap_or(None),
            )),
            inv_std_cache: Arc::new(RwLock::new(
                self.inv_std_cache.read().map(|c| c.clone()).unwrap_or(None),
            )),
            inputshape_cache: Arc::new(RwLock::new(
                self.inputshape_cache
                    .read()
                    .map(|c| c.clone())
                    .unwrap_or(None),
            )),
            forward_was_training: Arc::new(RwLock::new(
                self.forward_was_training.read().map(|g| *g).unwrap_or(true),
            )),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> BatchNorm<F> {
    /// Create a new batch normalization layer.
    ///
    /// # Arguments
    /// * `num_features` - number of channels `C`.
    /// * `momentum` - weight given to the current batch statistic when
    ///   updating the running mean/variance (typical: `0.1`).
    /// * `eps` - small constant added to the variance for numerical
    ///   stability (typical: `1e-5`).
    pub fn new<R: Rng>(num_features: usize, momentum: f64, eps: f64, _rng: &mut R) -> Result<Self> {
        if num_features == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "BatchNorm: num_features must be non-zero".to_string(),
            ));
        }

        let gamma = Array::<F, IxDyn>::from_elem(IxDyn(&[num_features]), F::one());
        let beta = Array::<F, IxDyn>::from_elem(IxDyn(&[num_features]), F::zero());

        let momentum_f = F::from(momentum).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert momentum to type F".to_string())
        })?;

        let eps_f = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert epsilon to type F".to_string())
        })?;

        Ok(Self {
            num_features,
            gamma,
            beta,
            eps: eps_f,
            momentum: momentum_f,
            training: true,
            running_mean: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_features])))),
            running_var: Arc::new(RwLock::new(Array::from_elem(
                IxDyn(&[num_features]),
                F::one(),
            ))),
            dgamma: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_features])))),
            dbeta: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_features])))),
            xhat_cache: Arc::new(RwLock::new(None)),
            inv_std_cache: Arc::new(RwLock::new(None)),
            inputshape_cache: Arc::new(RwLock::new(None)),
            forward_was_training: Arc::new(RwLock::new(true)),
        })
    }

    /// Set the training mode on the concrete type.
    ///
    /// Prefer going through [`Layer::set_training`] when the layer is stored
    /// as a `Box<dyn Layer<F>>` (e.g. inside [`crate::layers::Sequential`]) —
    /// that trait method is overridden below to do the same thing, so both
    /// paths stay in sync.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get the running mean (all zeros until at least one training-mode
    /// forward pass has run).
    pub fn running_mean(&self) -> Array<F, IxDyn> {
        self.running_mean
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.num_features])))
    }

    /// Get the running (unbiased) variance (all ones until at least one
    /// training-mode forward pass has run).
    pub fn running_var(&self) -> Array<F, IxDyn> {
        self.running_var
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::from_elem(IxDyn(&[self.num_features]), F::one()))
    }

    /// Validate `shape` is `[N, C, ...]` with `C == num_features` and return
    /// `(batch, channels, trailing_spatial_size)`.
    fn split_shape(&self, shape: &[usize]) -> Result<(usize, usize, usize)> {
        if shape.len() < 2 {
            return Err(NeuralError::InferenceError(format!(
                "BatchNorm requires an input of shape [N, C, ...], got {shape:?}"
            )));
        }
        let n = shape[0];
        let c = shape[1];
        if c != self.num_features {
            return Err(NeuralError::InvalidArchitecture(format!(
                "BatchNorm: expected {} channels, got {}",
                self.num_features, c
            )));
        }
        let s: usize = shape[2..].iter().product();
        Ok((n, c, s))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> Layer<F>
    for BatchNorm<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let inputshape = input.shape().to_vec();
        let (n, c, s) = self.split_shape(&inputshape)?;

        let flat = input
            .to_owned()
            .into_shape_with_order(IxDyn(&[n, c, s]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?;

        let mut xhat = Array::<F, IxDyn>::zeros(IxDyn(&[n, c, s]));
        let mut output = Array::<F, IxDyn>::zeros(IxDyn(&[n, c, s]));
        let mut inv_std = Array::<F, IxDyn>::zeros(IxDyn(&[c]));
        let was_training = self.training;

        if was_training {
            let count = n * s;
            if count == 0 {
                return Err(NeuralError::InferenceError(
                    "BatchNorm: cannot compute batch statistics over an empty batch".to_string(),
                ));
            }
            let count_f = F::from(count).ok_or_else(|| {
                NeuralError::InferenceError("Failed to convert element count".to_string())
            })?;

            let mut mean = Array::<F, IxDyn>::zeros(IxDyn(&[c]));
            let mut var = Array::<F, IxDyn>::zeros(IxDyn(&[c]));

            for ci in 0..c {
                let mut sum = F::zero();
                for ni in 0..n {
                    for si in 0..s {
                        sum += flat[[ni, ci, si]];
                    }
                }
                mean[[ci]] = sum / count_f;
            }
            for ci in 0..c {
                let mu = mean[[ci]];
                let mut sq_sum = F::zero();
                for ni in 0..n {
                    for si in 0..s {
                        let diff = flat[[ni, ci, si]] - mu;
                        sq_sum += diff * diff;
                    }
                }
                var[[ci]] = sq_sum / count_f;
            }
            for ci in 0..c {
                inv_std[[ci]] = (var[[ci]] + self.eps).sqrt().recip();
            }
            for ci in 0..c {
                let mu = mean[[ci]];
                let is = inv_std[[ci]];
                let g = self.gamma[[ci]];
                let b = self.beta[[ci]];
                for ni in 0..n {
                    for si in 0..s {
                        let xh = (flat[[ni, ci, si]] - mu) * is;
                        xhat[[ni, ci, si]] = xh;
                        output[[ni, ci, si]] = xh * g + b;
                    }
                }
            }

            // Update the running statistics. The batch itself is normalized
            // with the biased variance above, but (matching common practice)
            // the running average tracks the *unbiased* estimator.
            let unbiased_scale = if count > 1 {
                count_f / F::from(count - 1).unwrap_or(F::one())
            } else {
                F::one()
            };
            if let (Ok(mut rm), Ok(mut rv)) = (self.running_mean.write(), self.running_var.write())
            {
                let keep = F::one() - self.momentum;
                for ci in 0..c {
                    rm[[ci]] = keep * rm[[ci]] + self.momentum * mean[[ci]];
                    rv[[ci]] = keep * rv[[ci]] + self.momentum * (var[[ci]] * unbiased_scale);
                }
            }
        } else {
            let rm = self.running_mean.read().map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on running mean".to_string(),
                )
            })?;
            let rv = self.running_var.read().map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on running variance".to_string(),
                )
            })?;
            for ci in 0..c {
                inv_std[[ci]] = (rv[[ci]] + self.eps).sqrt().recip();
            }
            for ci in 0..c {
                let mu = rm[[ci]];
                let is = inv_std[[ci]];
                let g = self.gamma[[ci]];
                let b = self.beta[[ci]];
                for ni in 0..n {
                    for si in 0..s {
                        let xh = (flat[[ni, ci, si]] - mu) * is;
                        xhat[[ni, ci, si]] = xh;
                        output[[ni, ci, si]] = xh * g + b;
                    }
                }
            }
        }

        if let Ok(mut cache) = self.xhat_cache.write() {
            *cache = Some(xhat);
        }
        if let Ok(mut cache) = self.inv_std_cache.write() {
            *cache = Some(inv_std);
        }
        if let Ok(mut cache) = self.inputshape_cache.write() {
            *cache = Some(inputshape.clone());
        }
        if let Ok(mut cache) = self.forward_was_training.write() {
            *cache = was_training;
        }

        output
            .into_shape_with_order(IxDyn(&inputshape))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {e}")))
    }

    /// Gradient of batch normalization.
    ///
    /// In training mode this is the standard BN backward (mean/var are
    /// functions of the batch, so their contribution to `dx` is included):
    ///
    /// ```text
    /// dgamma_c = sum_{n,s} dy[n,c,s] * x_hat[n,c,s]
    /// dbeta_c  = sum_{n,s} dy[n,c,s]
    /// dx[n,c,s] = (gamma_c * inv_std_c / count)
    ///     * (count*dy[n,c,s] - sum_dy_c - x_hat[n,c,s] * sum_dy_xhat_c)
    /// ```
    ///
    /// In evaluation mode the running statistics are fixed constants (not
    /// functions of the input), so the map degenerates to a plain per-element
    /// affine: `dx = dy * gamma_c * inv_std_c`.
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let xhat_guard = self.xhat_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on xhat cache".to_string())
        })?;
        let inv_std_guard = self.inv_std_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on inv_std cache".to_string())
        })?;
        let shape_guard = self.inputshape_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on shape cache".to_string())
        })?;
        let training_guard = self.forward_was_training.read().map_err(|_| {
            NeuralError::InferenceError(
                "Failed to acquire read lock on training-mode cache".to_string(),
            )
        })?;

        let missing = || {
            NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            )
        };
        let xhat = xhat_guard.as_ref().ok_or_else(missing)?;
        let inv_std = inv_std_guard.as_ref().ok_or_else(missing)?;
        let cachedshape = shape_guard.as_ref().ok_or_else(missing)?;
        let was_training = *training_guard;

        let outshape = grad_output.shape().to_vec();
        if &outshape != cachedshape {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient shape {outshape:?} does not match the cached forward shape {cachedshape:?}"
            )));
        }
        let (n, c, s) = self.split_shape(&outshape)?;
        if xhat.shape() != [n, c, s] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Cached activations have shape {:?} but the gradient implies [{n}, {c}, {s}]",
                xhat.shape()
            )));
        }

        let grad_flat = grad_output
            .to_owned()
            .into_shape_with_order(IxDyn(&[n, c, s]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape output gradient: {e}"))
            })?;

        let count = n * s;
        let count_f = F::from(count).ok_or_else(|| {
            NeuralError::InferenceError("Failed to convert element count".to_string())
        })?;

        let mut dgamma = Array::<F, IxDyn>::zeros(IxDyn(&[c]));
        let mut dbeta = Array::<F, IxDyn>::zeros(IxDyn(&[c]));
        let mut grad_input = Array::<F, IxDyn>::zeros(IxDyn(&[n, c, s]));

        for ci in 0..c {
            let mut sum_dy = F::zero();
            let mut sum_dy_xhat = F::zero();
            for ni in 0..n {
                for si in 0..s {
                    let dy = grad_flat[[ni, ci, si]];
                    let xh = xhat[[ni, ci, si]];
                    sum_dy += dy;
                    sum_dy_xhat += dy * xh;
                }
            }
            dgamma[[ci]] = sum_dy_xhat;
            dbeta[[ci]] = sum_dy;

            let g = self.gamma[[ci]];
            let is = inv_std[[ci]];

            if was_training {
                for ni in 0..n {
                    for si in 0..s {
                        let dy = grad_flat[[ni, ci, si]];
                        let xh = xhat[[ni, ci, si]];
                        grad_input[[ni, ci, si]] =
                            (g * is / count_f) * (count_f * dy - sum_dy - xh * sum_dy_xhat);
                    }
                }
            } else {
                for ni in 0..n {
                    for si in 0..s {
                        grad_input[[ni, ci, si]] = grad_flat[[ni, ci, si]] * g * is;
                    }
                }
            }
        }

        if let Ok(mut cache) = self.dgamma.write() {
            *cache = dgamma;
        }
        if let Ok(mut cache) = self.dbeta.write() {
            *cache = dbeta;
        }

        grad_input
            .into_shape_with_order(IxDyn(&outshape))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape input gradient: {e}"))
            })
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        let dgamma = self
            .dgamma
            .read()
            .map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on dgamma".to_string())
            })?
            .clone();
        let dbeta = self
            .dbeta
            .read()
            .map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on dbeta".to_string())
            })?
            .clone();

        for (param, grad) in [(&mut self.gamma, &dgamma), (&mut self.beta, &dbeta)] {
            if param.shape() != grad.shape() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Parameter shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
            scirs2_core::ndarray::Zip::from(&mut *param)
                .and(grad)
                .for_each(|w, &g| *w -= learningrate * g);
        }
        Ok(())
    }

    fn gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        ParamLayer::get_gradients(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Switch between batch-statistic (training) and running-statistic
    /// (evaluation) normalization. Overridden so that dispatch through a
    /// `Box<dyn Layer<F>>` — e.g. `Sequential::set_training` — actually
    /// reaches the field; the default `Layer::set_training` is a no-op.
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// See [`Layer::set_training`]: overridden for the same reason.
    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "BatchNorm"
    }

    fn parameter_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn params(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_params(&mut self, params: &[Array<F, scirs2_core::ndarray::IxDyn>]) -> Result<()> {
        if params.len() >= 2 {
            self.gamma = params[0].clone();
            self.beta = params[1].clone();
        } else if params.len() == 1 {
            self.gamma = params[0].clone();
        }
        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> ParamLayer<F>
    for BatchNorm<F>
{
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    /// Gradients of `[gamma, beta]`; zero until `backward` has run.
    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        let read = |cell: &Arc<RwLock<Array<F, IxDyn>>>| match cell.read() {
            Ok(guard) => guard.clone(),
            Err(_) => Array::zeros(IxDyn(&[0])),
        };
        vec![read(&self.dgamma), read(&self.dbeta)]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        if params[0].shape() != self.gamma.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Gamma shape mismatch: expected {:?}, got {:?}",
                self.gamma.shape(),
                params[0].shape()
            )));
        }
        if params[1].shape() != self.beta.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Beta shape mismatch: expected {:?}, got {:?}",
                self.beta.shape(),
                params[1].shape()
            )));
        }

        self.gamma = params[0].clone();
        self.beta = params[1].clone();

        Ok(())
    }
}
