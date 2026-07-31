//! Normalization layer variants
//!
//! This module provides additional normalization techniques beyond the
//! standard BatchNorm and LayerNorm:
//!
//! - **RMSNorm**: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
//! - **Group Normalization** (Wu & He, 2018): Divides channels into groups.
//! - **Instance Normalization** (Ulyanov et al., 2016): Per-instance, per-channel.
//! - **Weight Normalization** (Salimans & Kingma, 2016): Reparameterizes weights.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

// ===========================================================================
// 1. RMSNorm
// ===========================================================================

/// Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
///
/// Normalizes the input by dividing by the RMS of the features, without
/// centering (no mean subtraction). This is computationally cheaper than
/// LayerNorm and is used in LLaMA, Gemma, and other modern architectures.
///
/// y = x / RMS(x) * gamma
/// RMS(x) = sqrt(mean(x^2) + eps)
///
/// # Input: (..., d_model)
/// # Output: same shape, normalized
#[derive(Debug)]
pub struct RMSNorm<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    eps: F,
    /// Learnable scale parameter [d_model]
    gamma: Array<F, IxDyn>,
    /// Gradient of gamma
    dgamma: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Clone for RMSNorm<F> {
    fn clone(&self) -> Self {
        let dgamma_clone = self
            .dgamma
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.d_model])));
        Self {
            d_model: self.d_model,
            eps: self.eps,
            gamma: self.gamma.clone(),
            dgamma: Arc::new(RwLock::new(dgamma_clone)),
            training: self.training,
            _phantom: PhantomData,
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> RMSNorm<F> {
    /// Create a new RMSNorm layer.
    ///
    /// # Arguments
    /// * `d_model` - Feature dimension (last axis).
    /// * `eps` - Small constant for numerical stability.
    pub fn new(d_model: usize, eps: f64) -> Result<Self> {
        if d_model == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "d_model must be > 0".into(),
            ));
        }
        let eps_f =
            F::from(eps).ok_or_else(|| NeuralError::InvalidArchitecture("eps conv".into()))?;
        let gamma = Array::from_elem(IxDyn(&[d_model]), F::one());
        let dgamma = Arc::new(RwLock::new(Array::zeros(IxDyn(&[d_model]))));

        Ok(Self {
            d_model,
            eps: eps_f,
            gamma,
            dgamma,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F> for RMSNorm<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        let ndim = shape.len();
        if ndim < 1 {
            return Err(NeuralError::InferenceError("Need >= 1D".into()));
        }
        let feat = shape[ndim - 1];
        if feat != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dim ({feat}) != d_model ({})",
                self.d_model
            )));
        }

        let outer: usize = shape[..ndim - 1].iter().product();
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[outer, feat]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;

        let n = F::from(feat).unwrap_or(F::one());
        let mut output = Array::zeros(IxDyn(&[outer, feat]));

        for i in 0..outer {
            // RMS = sqrt(mean(x^2) + eps)
            let mut sum_sq = F::zero();
            for j in 0..feat {
                let v = flat[[i, j]];
                sum_sq += v * v;
            }
            let rms = (sum_sq / n + self.eps).sqrt();
            let inv_rms = F::one() / rms;

            for j in 0..feat {
                output[[i, j]] = flat[[i, j]] * inv_rms * self.gamma[[j]];
            }
        }

        output
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| NeuralError::InferenceError(format!("reshape back: {e}")))
    }

    /// Exact gradient of `y = gamma * x / rms(x)`, `rms(x) = sqrt(mean(x^2) + eps)`.
    ///
    /// Recomputed from the `input` the caller hands back (rather than an
    /// internal cache): with `xnorm = x / rms(x)` and `dg = dy * gamma`,
    ///
    /// ```text
    /// dgamma_j = sum_batch dy_j * xnorm_j
    /// dx_k     = (1 / rms) * (dg_k - xnorm_k * mean_j(dg_j * xnorm_j))
    /// ```
    fn backward(&self, input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        let ndim = shape.len();
        if ndim < 1 {
            return Err(NeuralError::InferenceError("Need >= 1D".into()));
        }
        let feat = shape[ndim - 1];
        if feat != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dim ({feat}) != d_model ({})",
                self.d_model
            )));
        }
        if grad.shape() != shape {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient shape {:?} must match input shape {shape:?}",
                grad.shape()
            )));
        }

        let outer: usize = shape[..ndim - 1].iter().product();
        let flat_in = input
            .clone()
            .into_shape_with_order(IxDyn(&[outer, feat]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;
        let flat_grad = grad
            .clone()
            .into_shape_with_order(IxDyn(&[outer, feat]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape grad: {e}")))?;

        let n = F::from(feat).unwrap_or(F::one());
        let mut grad_input = Array::<F, IxDyn>::zeros(IxDyn(&[outer, feat]));
        let mut dgamma = Array::<F, IxDyn>::zeros(IxDyn(&[feat]));
        let mut dg_row = vec![F::zero(); feat];

        for i in 0..outer {
            let mut sum_sq = F::zero();
            for j in 0..feat {
                let v = flat_in[[i, j]];
                sum_sq += v * v;
            }
            let rms = (sum_sq / n + self.eps).sqrt();
            let inv_rms = F::one() / rms;

            let mut s = F::zero();
            for j in 0..feat {
                let xnorm = flat_in[[i, j]] * inv_rms;
                let dy = flat_grad[[i, j]];
                let dg = dy * self.gamma[[j]];
                dg_row[j] = dg;
                dgamma[j] += dy * xnorm;
                s += dg * xnorm;
            }
            let mean_s = s / n;
            for j in 0..feat {
                let xnorm = flat_in[[i, j]] * inv_rms;
                grad_input[[i, j]] = inv_rms * (dg_row[j] - xnorm * mean_s);
            }
        }

        if let Ok(mut cache) = self.dgamma.write() {
            *cache = dgamma;
        }

        grad_input
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| NeuralError::InferenceError(format!("reshape back: {e}")))
    }

    fn update(&mut self, lr: F) -> Result<()> {
        let dg = self
            .dgamma
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        for j in 0..self.d_model {
            self.gamma[[j]] -= lr * dg[[j]];
        }
        drop(dg);
        let mut dg = self
            .dgamma
            .write()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        dg.fill(F::zero());
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "RMSNorm"
    }
    fn parameter_count(&self) -> usize {
        self.d_model
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![self.gamma.clone()]
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if let Some(p) = params.first() {
            self.gamma = p.clone();
        }
        Ok(())
    }
}

// ===========================================================================
// 2. Group Normalization
// ===========================================================================

/// Group Normalization (Wu & He, 2018).
///
/// Divides channels into groups and normalizes within each group.
/// This is a compromise between LayerNorm (1 group) and InstanceNorm
/// (1 channel per group).
///
/// # Input: (batch, channels, ...) or (batch, seq, channels)
/// For the common transformer case of (batch, seq, channels), set `channel_dim = 2`.
///
/// # Output: same shape, normalized within groups
#[derive(Debug)]
pub struct GroupNorm<F: Float + Debug + Send + Sync + NumAssign> {
    num_groups: usize,
    num_channels: usize,
    eps: F,
    /// Learnable scale [num_channels]
    gamma: Array<F, IxDyn>,
    /// Learnable bias [num_channels]
    beta: Array<F, IxDyn>,
    dgamma: Arc<RwLock<Array<F, IxDyn>>>,
    dbeta: Arc<RwLock<Array<F, IxDyn>>>,
    /// Which axis is the channel axis (typically 1 for CNN, 2 for transformer).
    channel_axis: usize,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> GroupNorm<F> {
    /// Create a new group normalization layer.
    ///
    /// # Arguments
    /// * `num_groups` - Number of groups (must divide num_channels).
    /// * `num_channels` - Number of channels.
    /// * `eps` - Small constant for numerical stability.
    /// * `channel_axis` - Axis index for channel dimension (1 for NCHW, 2 for NLC).
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        channel_axis: usize,
    ) -> Result<Self> {
        if num_groups == 0 || num_channels == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_groups and num_channels must be > 0".into(),
            ));
        }
        if !num_channels.is_multiple_of(num_groups) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )));
        }

        let eps_f =
            F::from(eps).ok_or_else(|| NeuralError::InvalidArchitecture("eps conv".into()))?;

        let gamma = Array::from_elem(IxDyn(&[num_channels]), F::one());
        let beta = Array::zeros(IxDyn(&[num_channels]));

        Ok(Self {
            num_groups,
            num_channels,
            eps: eps_f,
            gamma,
            beta,
            dgamma: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_channels])))),
            dbeta: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_channels])))),
            channel_axis,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for GroupNorm<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(NeuralError::InferenceError("Need >= 2D".into()));
        }

        let ch_axis = self.channel_axis;
        if ch_axis >= ndim {
            return Err(NeuralError::InferenceError(format!(
                "channel_axis ({ch_axis}) >= ndim ({ndim})"
            )));
        }
        let num_ch = shape[ch_axis];
        if num_ch != self.num_channels {
            return Err(NeuralError::InferenceError(format!(
                "Channel dim ({num_ch}) != num_channels ({})",
                self.num_channels
            )));
        }

        let batch = shape[0];
        let channels_per_group = self.num_channels / self.num_groups;

        // For simplicity handle 3D (batch, seq/spatial, channels) and
        // standard 4D (batch, channels, H, W).
        let mut output = input.clone();

        if ndim == 3 && ch_axis == 2 {
            // (batch, seq, channels) -- transformer layout
            let seq = shape[1];
            for b in 0..batch {
                for g in 0..self.num_groups {
                    let c_start = g * channels_per_group;
                    let c_end = c_start + channels_per_group;

                    // Compute mean and variance over (seq, channels_in_group)
                    let count = seq * channels_per_group;
                    let n = F::from(count).unwrap_or(F::one());

                    let mut mean = F::zero();
                    for s in 0..seq {
                        for c in c_start..c_end {
                            mean += input[[b, s, c]];
                        }
                    }
                    mean /= n;

                    let mut var = F::zero();
                    for s in 0..seq {
                        for c in c_start..c_end {
                            let diff = input[[b, s, c]] - mean;
                            var += diff * diff;
                        }
                    }
                    var /= n;

                    let inv_std = (var + self.eps).sqrt().recip();

                    for s in 0..seq {
                        for c in c_start..c_end {
                            let x_norm = (input[[b, s, c]] - mean) * inv_std;
                            output[[b, s, c]] = x_norm * self.gamma[[c]] + self.beta[[c]];
                        }
                    }
                }
            }
        } else if ndim == 4 && ch_axis == 1 {
            // (batch, channels, H, W) -- CNN layout
            let h = shape[2];
            let w = shape[3];
            for b in 0..batch {
                for g in 0..self.num_groups {
                    let c_start = g * channels_per_group;
                    let c_end = c_start + channels_per_group;

                    let count = channels_per_group * h * w;
                    let n = F::from(count).unwrap_or(F::one());

                    let mut mean = F::zero();
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                mean += input[[b, c, y, x]];
                            }
                        }
                    }
                    mean /= n;

                    let mut var = F::zero();
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                let diff = input[[b, c, y, x]] - mean;
                                var += diff * diff;
                            }
                        }
                    }
                    var /= n;

                    let inv_std = (var + self.eps).sqrt().recip();
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                let x_norm = (input[[b, c, y, x]] - mean) * inv_std;
                                output[[b, c, y, x]] = x_norm * self.gamma[[c]] + self.beta[[c]];
                            }
                        }
                    }
                }
            }
        } else {
            // Generic fallback: treat last axis as channel if channel_axis == ndim-1
            // or first non-batch axis as channel otherwise
            return Err(NeuralError::InferenceError(format!(
                "GroupNorm: unsupported layout ndim={ndim}, channel_axis={ch_axis}. \
                 Supported: 3D ch=2 or 4D ch=1."
            )));
        }

        Ok(output)
    }

    /// Exact gradient of group normalization.
    ///
    /// Recomputed from the `input` the caller hands back (rather than an
    /// internal cache). Within one `(batch, group)` reduction set, `gamma`
    /// and `beta` vary by channel exactly like they do in
    /// [`super::normalization::LayerNorm`], so the same closed form applies
    /// per group:
    ///
    /// ```text
    /// dxhat_j  = dy_j * gamma_{c(j)}
    /// dx_j     = inv_std * (dxhat_j - mean_k(dxhat_k) - xhat_j * mean_k(dxhat_k * xhat_k))
    /// ```
    ///
    /// where `mean_k` ranges over the same `(channel-in-group, spatial)`
    /// reduction set that produced `mean`/`var`. `dgamma`/`dbeta` sum over
    /// the batch and spatial dims for each channel.
    fn backward(&self, input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if grad.shape() != shape {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient shape {:?} must match input shape {shape:?}",
                grad.shape()
            )));
        }
        let ndim = shape.len();
        let ch_axis = self.channel_axis;
        let channels_per_group = self.num_channels / self.num_groups;
        let batch = shape[0];

        let mut dgamma = Array::<F, IxDyn>::zeros(IxDyn(&[self.num_channels]));
        let mut dbeta = Array::<F, IxDyn>::zeros(IxDyn(&[self.num_channels]));
        let mut grad_input = Array::<F, IxDyn>::zeros(IxDyn(shape));

        if ndim == 3 && ch_axis == 2 {
            let seq = shape[1];
            let count = seq * channels_per_group;
            let n = F::from(count).unwrap_or(F::one());

            for b in 0..batch {
                for g in 0..self.num_groups {
                    let c_start = g * channels_per_group;
                    let c_end = c_start + channels_per_group;

                    let mut mean = F::zero();
                    for s in 0..seq {
                        for c in c_start..c_end {
                            mean += input[[b, s, c]];
                        }
                    }
                    mean /= n;
                    let mut var = F::zero();
                    for s in 0..seq {
                        for c in c_start..c_end {
                            let diff = input[[b, s, c]] - mean;
                            var += diff * diff;
                        }
                    }
                    var /= n;
                    let inv_std = (var + self.eps).sqrt().recip();

                    let mut sum_dxhat = F::zero();
                    let mut sum_dxhat_xhat = F::zero();
                    for s in 0..seq {
                        for c in c_start..c_end {
                            let xhat = (input[[b, s, c]] - mean) * inv_std;
                            let dy = grad[[b, s, c]];
                            dgamma[c] += dy * xhat;
                            dbeta[c] += dy;
                            let dxhat = dy * self.gamma[[c]];
                            sum_dxhat += dxhat;
                            sum_dxhat_xhat += dxhat * xhat;
                        }
                    }
                    for s in 0..seq {
                        for c in c_start..c_end {
                            let xhat = (input[[b, s, c]] - mean) * inv_std;
                            let dxhat = grad[[b, s, c]] * self.gamma[[c]];
                            grad_input[[b, s, c]] =
                                inv_std * (dxhat - sum_dxhat / n - xhat * sum_dxhat_xhat / n);
                        }
                    }
                }
            }
        } else if ndim == 4 && ch_axis == 1 {
            let h = shape[2];
            let w = shape[3];
            let count = channels_per_group * h * w;
            let n = F::from(count).unwrap_or(F::one());

            for b in 0..batch {
                for g in 0..self.num_groups {
                    let c_start = g * channels_per_group;
                    let c_end = c_start + channels_per_group;

                    let mut mean = F::zero();
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                mean += input[[b, c, y, x]];
                            }
                        }
                    }
                    mean /= n;
                    let mut var = F::zero();
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                let diff = input[[b, c, y, x]] - mean;
                                var += diff * diff;
                            }
                        }
                    }
                    var /= n;
                    let inv_std = (var + self.eps).sqrt().recip();

                    let mut sum_dxhat = F::zero();
                    let mut sum_dxhat_xhat = F::zero();
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                let xhat = (input[[b, c, y, x]] - mean) * inv_std;
                                let dy = grad[[b, c, y, x]];
                                dgamma[c] += dy * xhat;
                                dbeta[c] += dy;
                                let dxhat = dy * self.gamma[[c]];
                                sum_dxhat += dxhat;
                                sum_dxhat_xhat += dxhat * xhat;
                            }
                        }
                    }
                    for c in c_start..c_end {
                        for y in 0..h {
                            for x in 0..w {
                                let xhat = (input[[b, c, y, x]] - mean) * inv_std;
                                let dxhat = grad[[b, c, y, x]] * self.gamma[[c]];
                                grad_input[[b, c, y, x]] =
                                    inv_std * (dxhat - sum_dxhat / n - xhat * sum_dxhat_xhat / n);
                            }
                        }
                    }
                }
            }
        } else {
            return Err(NeuralError::InferenceError(format!(
                "GroupNorm: unsupported layout ndim={ndim}, channel_axis={ch_axis}. \
                 Supported: 3D ch=2 or 4D ch=1."
            )));
        }

        if let Ok(mut cache) = self.dgamma.write() {
            *cache = dgamma;
        }
        if let Ok(mut cache) = self.dbeta.write() {
            *cache = dbeta;
        }

        Ok(grad_input)
    }

    fn update(&mut self, lr: F) -> Result<()> {
        let dg = self
            .dgamma
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let db = self
            .dbeta
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        for j in 0..self.num_channels {
            self.gamma[[j]] -= lr * dg[[j]];
            self.beta[[j]] -= lr * db[[j]];
        }
        drop(dg);
        drop(db);
        if let Ok(mut dg) = self.dgamma.write() {
            dg.fill(F::zero());
        }
        if let Ok(mut db) = self.dbeta.write() {
            db.fill(F::zero());
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "GroupNorm"
    }
    fn parameter_count(&self) -> usize {
        2 * self.num_channels
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.len() >= 2 {
            self.gamma = params[0].clone();
            self.beta = params[1].clone();
        } else if params.len() == 1 {
            self.gamma = params[0].clone();
        }
        Ok(())
    }
}

// ===========================================================================
// 3. Instance Normalization
// ===========================================================================

/// Instance Normalization (Ulyanov et al., 2016).
///
/// Normalizes each channel of each sample independently. Equivalent to
/// GroupNorm with `num_groups = num_channels`.
///
/// # Input: (batch, channels, H, W) -- 4D CNN layout
/// # Output: same shape, normalized per (batch, channel)
#[derive(Debug)]
pub struct InstanceNorm<F: Float + Debug + Send + Sync + NumAssign> {
    num_channels: usize,
    eps: F,
    /// Optional learnable scale [num_channels]
    gamma: Option<Array<F, IxDyn>>,
    /// Optional learnable bias [num_channels]
    beta: Option<Array<F, IxDyn>>,
    /// Whether to use affine transform
    affine: bool,
    training: bool,
    /// Gradient of gamma (unused, kept zero, unless `affine`)
    dgamma: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of beta (unused, kept zero, unless `affine`)
    dbeta: Arc<RwLock<Array<F, IxDyn>>>,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> InstanceNorm<F> {
    /// Create a new instance normalization layer.
    ///
    /// # Arguments
    /// * `num_channels` - Number of channels (features).
    /// * `eps` - Epsilon for numerical stability.
    /// * `affine` - Whether to learn scale and bias parameters.
    pub fn new(num_channels: usize, eps: f64, affine: bool) -> Result<Self> {
        if num_channels == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_channels must be > 0".into(),
            ));
        }
        let eps_f =
            F::from(eps).ok_or_else(|| NeuralError::InvalidArchitecture("eps conv".into()))?;

        let (gamma, beta) = if affine {
            (
                Some(Array::from_elem(IxDyn(&[num_channels]), F::one())),
                Some(Array::zeros(IxDyn(&[num_channels]))),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            num_channels,
            eps: eps_f,
            gamma,
            beta,
            affine,
            training: true,
            dgamma: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_channels])))),
            dbeta: Arc::new(RwLock::new(Array::zeros(IxDyn(&[num_channels])))),
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for InstanceNorm<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        // Support 3D (batch, channels, spatial) and 4D (batch, channels, H, W)
        if shape.len() < 3 {
            return Err(NeuralError::InferenceError(
                "InstanceNorm requires >= 3D input".into(),
            ));
        }
        let batch = shape[0];
        let channels = shape[1];
        if channels != self.num_channels {
            return Err(NeuralError::InferenceError(format!(
                "Channel dim ({channels}) != num_channels ({})",
                self.num_channels
            )));
        }

        // Spatial dims: everything after batch and channel
        let spatial: usize = shape[2..].iter().product();
        let n = F::from(spatial).unwrap_or(F::one());

        // Flatten spatial: [batch, channels, spatial]
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch, channels, spatial]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;

        let mut output = flat.clone();

        for b in 0..batch {
            for c in 0..channels {
                // Mean
                let mut mean = F::zero();
                for s in 0..spatial {
                    mean += flat[[b, c, s]];
                }
                mean /= n;

                // Variance
                let mut var = F::zero();
                for s in 0..spatial {
                    let diff = flat[[b, c, s]] - mean;
                    var += diff * diff;
                }
                var /= n;

                let inv_std = (var + self.eps).sqrt().recip();

                let gamma_c = self.gamma.as_ref().map(|g| g[[c]]).unwrap_or(F::one());
                let beta_c = self.beta.as_ref().map(|b| b[[c]]).unwrap_or(F::zero());

                for s in 0..spatial {
                    let x_norm = (flat[[b, c, s]] - mean) * inv_std;
                    output[[b, c, s]] = x_norm * gamma_c + beta_c;
                }
            }
        }

        // Reshape back to original
        output
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| NeuralError::InferenceError(format!("reshape back: {e}")))
    }

    /// Exact gradient of instance normalization.
    ///
    /// Recomputed from the `input` the caller hands back. Each `(batch,
    /// channel)` pair normalizes independently over the spatial axis, and
    /// `gamma`/`beta` (when present) are constant across that reduction set,
    /// so the closed form matches [`super::normalization::BatchNorm`]'s
    /// per-channel case with the reduction restricted to one sample:
    ///
    /// ```text
    /// dx[b,c,s] = gamma_c * inv_std / S * (S*dy[b,c,s] - sum_s(dy) - xhat[b,c,s] * sum_s(dy*xhat))
    /// ```
    fn backward(&self, input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if grad.shape() != shape {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient shape {:?} must match input shape {shape:?}",
                grad.shape()
            )));
        }
        if shape.len() < 3 {
            return Err(NeuralError::InferenceError(
                "InstanceNorm requires >= 3D input".into(),
            ));
        }
        let batch = shape[0];
        let channels = shape[1];
        if channels != self.num_channels {
            return Err(NeuralError::InferenceError(format!(
                "Channel dim ({channels}) != num_channels ({})",
                self.num_channels
            )));
        }
        let spatial: usize = shape[2..].iter().product();
        let n = F::from(spatial).unwrap_or(F::one());

        let flat_in = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch, channels, spatial]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;
        let flat_grad = grad
            .clone()
            .into_shape_with_order(IxDyn(&[batch, channels, spatial]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape grad: {e}")))?;

        let mut dgamma = Array::<F, IxDyn>::zeros(IxDyn(&[channels]));
        let mut dbeta = Array::<F, IxDyn>::zeros(IxDyn(&[channels]));
        let mut grad_input = Array::<F, IxDyn>::zeros(IxDyn(&[batch, channels, spatial]));

        for b in 0..batch {
            for c in 0..channels {
                let mut mean = F::zero();
                for s in 0..spatial {
                    mean += flat_in[[b, c, s]];
                }
                mean /= n;
                let mut var = F::zero();
                for s in 0..spatial {
                    let diff = flat_in[[b, c, s]] - mean;
                    var += diff * diff;
                }
                var /= n;
                let inv_std = (var + self.eps).sqrt().recip();
                let gamma_c = self.gamma.as_ref().map(|g| g[[c]]).unwrap_or(F::one());

                let mut sum_dy = F::zero();
                let mut sum_dy_xhat = F::zero();
                for s in 0..spatial {
                    let xhat = (flat_in[[b, c, s]] - mean) * inv_std;
                    let dy = flat_grad[[b, c, s]];
                    dgamma[c] += dy * xhat;
                    dbeta[c] += dy;
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                }
                for s in 0..spatial {
                    let xhat = (flat_in[[b, c, s]] - mean) * inv_std;
                    let dy = flat_grad[[b, c, s]];
                    grad_input[[b, c, s]] =
                        (gamma_c * inv_std / n) * (n * dy - sum_dy - xhat * sum_dy_xhat);
                }
            }
        }

        if self.affine {
            if let Ok(mut cache) = self.dgamma.write() {
                *cache = dgamma;
            }
            if let Ok(mut cache) = self.dbeta.write() {
                *cache = dbeta;
            }
        }

        grad_input
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| NeuralError::InferenceError(format!("reshape back: {e}")))
    }

    fn update(&mut self, lr: F) -> Result<()> {
        if !self.affine {
            return Ok(());
        }
        let dg = self
            .dgamma
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?
            .clone();
        let db = self
            .dbeta
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?
            .clone();
        if let Some(ref mut gamma) = self.gamma {
            for c in 0..self.num_channels {
                gamma[[c]] -= lr * dg[[c]];
            }
        }
        if let Some(ref mut beta) = self.beta {
            for c in 0..self.num_channels {
                beta[[c]] -= lr * db[[c]];
            }
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "InstanceNorm"
    }
    fn parameter_count(&self) -> usize {
        if self.affine {
            2 * self.num_channels
        } else {
            0
        }
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = Vec::new();
        if let Some(ref g) = self.gamma {
            p.push(g.clone());
        }
        if let Some(ref b) = self.beta {
            p.push(b.clone());
        }
        p
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if self.affine && params.len() >= 2 {
            self.gamma = Some(params[0].clone());
            self.beta = Some(params[1].clone());
        }
        Ok(())
    }
}

// ===========================================================================
// 4. Weight Normalization
// ===========================================================================

/// Weight Normalization (Salimans & Kingma, 2016).
///
/// Reparameterizes weight vectors as `w = g * (v / ||v||)` where `g` is a
/// learnable scalar magnitude and `v` is the unnormalized direction vector.
///
/// This layer wraps a weight matrix and produces the normalized version.
/// It stores `v` [rows, cols] and `g` [cols], and computes
/// `W[:, j] = g[j] * v[:, j] / ||v[:, j]||` during forward.
///
/// The forward pass expects input (batch, in_features) and produces (batch, out_features).
#[derive(Debug)]
pub struct WeightNorm<F: Float + Debug + Send + Sync + NumAssign> {
    in_features: usize,
    out_features: usize,
    /// Direction vectors [in_features, out_features]
    v: Array<F, IxDyn>,
    /// Magnitude scalars [out_features]
    g: Array<F, IxDyn>,
    /// Bias [out_features]
    bias: Array<F, IxDyn>,
    training: bool,
    /// Gradient of `v`, shape `[in_features, out_features]`
    dv: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of `g`, shape `[out_features]`
    dg: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of `bias`, shape `[out_features]`
    dbias: Arc<RwLock<Array<F, IxDyn>>>,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> WeightNorm<F> {
    /// Create a new weight-normalized linear layer.
    ///
    /// # Arguments
    /// * `in_features` - Input dimension.
    /// * `out_features` - Output dimension.
    /// * `rng` - Random number generator.
    pub fn new<R: Rng>(in_features: usize, out_features: usize, rng: &mut R) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "in/out features must be > 0".into(),
            ));
        }

        let scale = (2.0 / (in_features + out_features) as f64).sqrt();
        let mut v_data = Vec::with_capacity(in_features * out_features);
        for _ in 0..(in_features * out_features) {
            let val: f64 = rng.random_range(-1.0..1.0);
            v_data.push(
                F::from(val * scale)
                    .ok_or_else(|| NeuralError::InvalidArchitecture("conv".into()))?,
            );
        }
        let v = Array::from_shape_vec(IxDyn(&[in_features, out_features]), v_data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))?;

        // Initialize g from the norms of v columns
        let mut g_data = Vec::with_capacity(out_features);
        for j in 0..out_features {
            let mut norm_sq = F::zero();
            for i in 0..in_features {
                let val = v[[i, j]];
                norm_sq += val * val;
            }
            g_data.push(norm_sq.sqrt());
        }
        let g = Array::from_shape_vec(IxDyn(&[out_features]), g_data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))?;

        let bias = Array::zeros(IxDyn(&[out_features]));

        Ok(Self {
            in_features,
            out_features,
            v,
            g,
            bias,
            training: true,
            dv: Arc::new(RwLock::new(Array::zeros(IxDyn(&[
                in_features,
                out_features,
            ])))),
            dg: Arc::new(RwLock::new(Array::zeros(IxDyn(&[out_features])))),
            dbias: Arc::new(RwLock::new(Array::zeros(IxDyn(&[out_features])))),
            _phantom: PhantomData,
        })
    }

    /// Compute the normalized weight matrix W = g * (v / ||v||).
    fn compute_weights(&self) -> Array<F, IxDyn> {
        let mut w = Array::zeros(IxDyn(&[self.in_features, self.out_features]));
        for j in 0..self.out_features {
            let mut norm_sq = F::zero();
            for i in 0..self.in_features {
                let val = self.v[[i, j]];
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            let eps = F::from(1e-12).unwrap_or(F::zero());
            let inv_norm = if norm > eps {
                F::one() / norm
            } else {
                F::one()
            };

            for i in 0..self.in_features {
                w[[i, j]] = self.g[[j]] * self.v[[i, j]] * inv_norm;
            }
        }
        w
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for WeightNorm<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(NeuralError::InferenceError("Need >= 2D".into()));
        }
        let last = shape[shape.len() - 1];
        if last != self.in_features {
            return Err(NeuralError::InferenceError(format!(
                "Last dim ({last}) != in_features ({})",
                self.in_features
            )));
        }

        let w = self.compute_weights();

        // Generalized matmul: (..., in) @ (in, out) -> (..., out)
        let outer: usize = shape[..shape.len() - 1].iter().product();
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[outer, self.in_features]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;

        let mut out = Array::zeros(IxDyn(&[outer, self.out_features]));
        for b in 0..outer {
            for o in 0..self.out_features {
                let mut acc = self.bias[[o]];
                for i in 0..self.in_features {
                    acc += flat[[b, i]] * w[[i, o]];
                }
                out[[b, o]] = acc;
            }
        }

        let mut out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        out_shape.push(self.out_features);
        out.into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| NeuralError::InferenceError(format!("reshape back: {e}")))
    }

    /// Exact gradient of `y = x @ (g * v / ||v||) + bias`.
    ///
    /// Recomputed from the `input` the caller hands back. First backprops
    /// through the linear map using the effective weight `w = g*v/||v||`,
    /// then through the weight-norm reparameterization itself (Salimans &
    /// Kingma, 2016):
    ///
    /// ```text
    /// dbias_o = sum_batch dy[b,o]
    /// dw[i,o] = sum_batch dy[b,o] * x[b,i]
    /// dx[b,i] = sum_o dy[b,o] * w[i,o]
    /// dg_o    = sum_i dw[i,o] * v[i,o] / ||v[:,o]||
    /// dv[i,o] = (g_o / ||v[:,o]||) * (dw[i,o] - v[i,o] * dot_o / ||v[:,o]||^2)
    /// ```
    ///
    /// where `dot_o = sum_i dw[i,o] * v[i,o]`.
    fn backward(&self, input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape().to_vec();
        if shape.len() < 2 {
            return Err(NeuralError::InferenceError("Need >= 2D".into()));
        }
        let last = shape[shape.len() - 1];
        if last != self.in_features {
            return Err(NeuralError::InferenceError(format!(
                "Last dim ({last}) != in_features ({})",
                self.in_features
            )));
        }
        let mut expectedshape = shape[..shape.len() - 1].to_vec();
        expectedshape.push(self.out_features);
        if grad.shape() != expectedshape.as_slice() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient shape {:?} must be {:?}",
                grad.shape(),
                expectedshape
            )));
        }

        let outer: usize = shape[..shape.len() - 1].iter().product();
        let flat_in = input
            .clone()
            .into_shape_with_order(IxDyn(&[outer, self.in_features]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;
        let flat_grad = grad
            .clone()
            .into_shape_with_order(IxDyn(&[outer, self.out_features]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape grad: {e}")))?;

        let w = self.compute_weights();

        let mut dbias = Array::<F, IxDyn>::zeros(IxDyn(&[self.out_features]));
        let mut dw = Array::<F, IxDyn>::zeros(IxDyn(&[self.in_features, self.out_features]));
        let mut grad_input = Array::<F, IxDyn>::zeros(IxDyn(&[outer, self.in_features]));

        for o in 0..self.out_features {
            let mut acc = F::zero();
            for b in 0..outer {
                acc += flat_grad[[b, o]];
            }
            dbias[[o]] = acc;
        }
        for i in 0..self.in_features {
            for o in 0..self.out_features {
                let mut acc = F::zero();
                for b in 0..outer {
                    acc += flat_grad[[b, o]] * flat_in[[b, i]];
                }
                dw[[i, o]] = acc;
            }
        }
        for b in 0..outer {
            for i in 0..self.in_features {
                let mut acc = F::zero();
                for o in 0..self.out_features {
                    acc += flat_grad[[b, o]] * w[[i, o]];
                }
                grad_input[[b, i]] = acc;
            }
        }

        // Backprop through w = g * v / ||v|| (per output column).
        let mut dv = Array::<F, IxDyn>::zeros(IxDyn(&[self.in_features, self.out_features]));
        let mut dg = Array::<F, IxDyn>::zeros(IxDyn(&[self.out_features]));
        let eps = F::from(1e-12).unwrap_or(F::zero());
        for o in 0..self.out_features {
            let mut norm_sq = F::zero();
            for i in 0..self.in_features {
                let val = self.v[[i, o]];
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();

            if norm <= eps {
                // Degenerate (near-)zero direction column: `compute_weights`
                // treats `inv_norm` as the constant `1` in this case, so the
                // norm is detached from `v` and w = g*v directly.
                let mut dg_o = F::zero();
                for i in 0..self.in_features {
                    dv[[i, o]] = dw[[i, o]] * self.g[[o]];
                    dg_o += dw[[i, o]] * self.v[[i, o]];
                }
                dg[[o]] = dg_o;
                continue;
            }

            let mut dot = F::zero();
            for i in 0..self.in_features {
                dot += dw[[i, o]] * self.v[[i, o]];
            }
            let g_o = self.g[[o]];
            let mut dg_o = F::zero();
            for i in 0..self.in_features {
                let v_io = self.v[[i, o]];
                dg_o += dw[[i, o]] * v_io / norm;
                dv[[i, o]] = (g_o / norm) * (dw[[i, o]] - v_io * dot / (norm * norm));
            }
            dg[[o]] = dg_o;
        }

        if let Ok(mut cache) = self.dv.write() {
            *cache = dv;
        }
        if let Ok(mut cache) = self.dg.write() {
            *cache = dg;
        }
        if let Ok(mut cache) = self.dbias.write() {
            *cache = dbias;
        }

        grad_input
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| NeuralError::InferenceError(format!("reshape back: {e}")))
    }

    fn update(&mut self, lr: F) -> Result<()> {
        let dv = self
            .dv
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?
            .clone();
        let dg = self
            .dg
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?
            .clone();
        let dbias = self
            .dbias
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?
            .clone();

        for i in 0..self.in_features {
            for o in 0..self.out_features {
                self.v[[i, o]] -= lr * dv[[i, o]];
            }
        }
        for o in 0..self.out_features {
            self.g[[o]] -= lr * dg[[o]];
            self.bias[[o]] -= lr * dbias[[o]];
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "WeightNorm"
    }
    fn parameter_count(&self) -> usize {
        self.in_features * self.out_features // v
        + self.out_features                  // g
        + self.out_features // bias
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![self.v.clone(), self.g.clone(), self.bias.clone()]
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.len() >= 3 {
            self.v = params[0].clone();
            self.g = params[1].clone();
            self.bias = params[2].clone();
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array3, Array4};
    use scirs2_core::random::rng;

    // ------- RMSNorm -------

    #[test]
    fn test_rmsnorm_creation() {
        let rms = RMSNorm::<f64>::new(16, 1e-6).expect("creation failed");
        assert_eq!(rms.layer_type(), "RMSNorm");
        assert_eq!(rms.parameter_count(), 16);
    }

    #[test]
    fn test_rmsnorm_forward_3d() {
        let rms = RMSNorm::<f64>::new(16, 1e-6).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 1.0).into_dyn();
        let out = rms.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_rmsnorm_forward_2d() {
        let rms = RMSNorm::<f64>::new(8, 1e-5).expect("creation failed");
        let input = Array::from_elem(IxDyn(&[4, 8]), 2.0);
        let out = rms.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[4, 8]);
        // RMS of constant 2.0 = 2.0, so normalized = 2/2 * 1 = 1
        for v in out.iter() {
            assert!((*v - 1.0).abs() < 1e-4, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_params() {
        let rms = RMSNorm::<f64>::new(16, 1e-6).expect("creation failed");
        assert_eq!(rms.params().len(), 1);
    }

    #[test]
    fn test_rmsnorm_clone() {
        let rms = RMSNorm::<f64>::new(8, 1e-5).expect("creation failed");
        let rms2 = rms.clone();
        assert_eq!(rms2.d_model, rms.d_model);
    }

    // ------- GroupNorm -------

    #[test]
    fn test_groupnorm_creation() {
        let gn = GroupNorm::<f64>::new(4, 16, 1e-5, 2).expect("creation failed");
        assert_eq!(gn.layer_type(), "GroupNorm");
        assert_eq!(gn.parameter_count(), 32);
    }

    #[test]
    fn test_groupnorm_3d() {
        let gn = GroupNorm::<f64>::new(4, 16, 1e-5, 2).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 1.0).into_dyn();
        let out = gn.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_groupnorm_4d_cnn() {
        let gn = GroupNorm::<f64>::new(4, 8, 1e-5, 1).expect("creation failed");
        let input = Array4::<f64>::from_elem((2, 8, 4, 4), 0.5).into_dyn();
        let out = gn.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 8, 4, 4]);
    }

    #[test]
    fn test_groupnorm_indivisible() {
        let result = GroupNorm::<f64>::new(3, 8, 1e-5, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_groupnorm_params() {
        let gn = GroupNorm::<f64>::new(4, 16, 1e-5, 2).expect("creation failed");
        assert_eq!(gn.params().len(), 2);
    }

    // ------- InstanceNorm -------

    #[test]
    fn test_instancenorm_creation() {
        let inst = InstanceNorm::<f64>::new(8, 1e-5, true).expect("creation failed");
        assert_eq!(inst.layer_type(), "InstanceNorm");
        assert_eq!(inst.parameter_count(), 16);
    }

    #[test]
    fn test_instancenorm_no_affine() {
        let inst = InstanceNorm::<f64>::new(8, 1e-5, false).expect("creation failed");
        assert_eq!(inst.parameter_count(), 0);
        assert_eq!(inst.params().len(), 0);
    }

    #[test]
    fn test_instancenorm_3d() {
        let inst = InstanceNorm::<f64>::new(4, 1e-5, true).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 4, 8), 1.0).into_dyn();
        let out = inst.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_instancenorm_4d() {
        let inst = InstanceNorm::<f64>::new(3, 1e-5, true).expect("creation failed");
        let input = Array4::<f64>::from_elem((2, 3, 4, 4), 0.5).into_dyn();
        let out = inst.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_instancenorm_constant_input() {
        // Constant input => normalized to zero (before affine)
        let inst = InstanceNorm::<f64>::new(2, 1e-5, false).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 2, 4), 5.0).into_dyn();
        let out = inst.forward(&input).expect("forward failed");
        // All outputs should be ~0 (mean subtracted from constant)
        for v in out.iter() {
            assert!(v.abs() < 1e-4, "expected ~0, got {v}");
        }
    }

    // ------- WeightNorm -------

    #[test]
    fn test_weightnorm_creation() {
        let mut r = rng();
        let wn = WeightNorm::<f64>::new(16, 8, &mut r).expect("creation failed");
        assert_eq!(wn.layer_type(), "WeightNorm");
        assert_eq!(wn.parameter_count(), 16 * 8 + 8 + 8);
    }

    #[test]
    fn test_weightnorm_forward_2d() {
        let mut r = rng();
        let wn = WeightNorm::<f64>::new(16, 8, &mut r).expect("creation failed");
        let input = Array::from_elem(IxDyn(&[4, 16]), 0.5);
        let out = wn.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[4, 8]);
    }

    #[test]
    fn test_weightnorm_forward_3d() {
        let mut r = rng();
        let wn = WeightNorm::<f64>::new(16, 8, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = wn.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_weightnorm_normalized_columns() {
        let mut r = rng();
        let wn = WeightNorm::<f64>::new(8, 4, &mut r).expect("creation failed");
        let w = wn.compute_weights();
        // Each column of w should have norm == g[j]
        for j in 0..4 {
            let mut norm_sq = 0.0;
            for i in 0..8 {
                norm_sq += w[[i, j]] * w[[i, j]];
            }
            let norm = norm_sq.sqrt();
            let g_j = wn.g[[j]];
            assert!(
                (norm - g_j).abs() < 1e-8,
                "column {j}: norm={norm}, g={g_j}"
            );
        }
    }

    #[test]
    fn test_weightnorm_params() {
        let mut r = rng();
        let wn = WeightNorm::<f64>::new(8, 4, &mut r).expect("creation failed");
        assert_eq!(wn.params().len(), 3); // v, g, bias
    }
}
