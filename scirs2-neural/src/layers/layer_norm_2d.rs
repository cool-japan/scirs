//! Layer Normalization for 2D feature maps (NCHW format).
//!
//! Normalizes across the spatial (H, W) dimensions per channel per sample,
//! then applies learnable scale (gamma) and bias (beta) parameters.
//!
//! Reference: "Layer Normalization", Ba et al. (2016).
//! Used in ConvNeXt as a spatial normalization equivalent to BatchNorm,
//! but without dependence on batch statistics.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, Array1, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Layer Normalization for 2D feature maps in NCHW format.
///
/// For an input of shape `[N, C, H, W]`, normalizes across the `(H, W)` spatial
/// dimensions independently per `(N, C)` pair, then applies learnable affine
/// parameters `weight` (gamma) and `bias` (beta) of shape `[C]`:
///
/// ```text
/// y[n,c,h,w] = (x[n,c,h,w] - mean[n,c]) / sqrt(var[n,c] + eps) * weight[c] + bias[c]
/// ```
///
/// # Shape
/// - Input: `[N, C, H, W]`
/// - Output: `[N, C, H, W]` (same shape)
///
/// # Parameters
/// - `weight` (gamma): shape `[C]`, initialized to ones
/// - `bias` (beta): shape `[C]`, initialized to zeros
pub struct LayerNorm2D<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Number of channels (C dimension)
    num_channels: usize,
    /// Learnable scale (gamma), shape `[C]`
    weight: Array1<F>,
    /// Learnable bias (beta), shape `[C]`
    bias: Array1<F>,
    /// Small epsilon for numerical stability
    eps: F,
    /// Optional name for this layer
    name: Option<String>,
    /// Gradient buffer for weight
    weight_grad: Arc<RwLock<Array1<F>>>,
    /// Gradient buffer for bias
    bias_grad: Arc<RwLock<Array1<F>>>,
    /// Cached normalized output for backward
    cached_x_norm: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cached per-(N,C) mean for backward
    cached_mean: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cached per-(N,C) variance for backward
    cached_var: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Number of spatial elements H*W (cached for backward)
    cached_spatial_size: Arc<RwLock<usize>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Debug
    for LayerNorm2D<F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNorm2D")
            .field("num_channels", &self.num_channels)
            .field("eps", &format!("{:?}", self.eps))
            .field("name", &self.name)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Clone
    for LayerNorm2D<F>
{
    fn clone(&self) -> Self {
        Self {
            num_channels: self.num_channels,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            eps: self.eps,
            name: self.name.clone(),
            weight_grad: Arc::new(RwLock::new(
                self.weight_grad
                    .read()
                    .map(|g| g.clone())
                    .unwrap_or_else(|_| Array1::zeros(self.num_channels)),
            )),
            bias_grad: Arc::new(RwLock::new(
                self.bias_grad
                    .read()
                    .map(|g| g.clone())
                    .unwrap_or_else(|_| Array1::zeros(self.num_channels)),
            )),
            cached_x_norm: Arc::new(RwLock::new(
                self.cached_x_norm.read().map(|c| c.clone()).unwrap_or(None),
            )),
            cached_mean: Arc::new(RwLock::new(
                self.cached_mean.read().map(|c| c.clone()).unwrap_or(None),
            )),
            cached_var: Arc::new(RwLock::new(
                self.cached_var.read().map(|c| c.clone()).unwrap_or(None),
            )),
            cached_spatial_size: Arc::new(RwLock::new(
                self.cached_spatial_size.read().map(|s| *s).unwrap_or(0),
            )),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> LayerNorm2D<F> {
    /// Create a new LayerNorm2D layer.
    ///
    /// # Arguments
    /// * `num_channels` — number of channels C in the NCHW input
    /// * `eps` — numerical stability constant (typical: `1e-6`)
    /// * `name` — optional human-readable name
    pub fn new<R>(_num_channels: usize, eps: f64, name: Option<&str>) -> Result<Self>
    where
        R: scirs2_core::random::Rng,
    {
        if _num_channels == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "LayerNorm2D: num_channels must be non-zero".to_string(),
            ));
        }
        let eps_f = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture(
                "LayerNorm2D: failed to convert eps to float type".to_string(),
            )
        })?;

        let weight = Array1::from_elem(_num_channels, F::one());
        let bias = Array1::zeros(_num_channels);

        Ok(Self {
            num_channels: _num_channels,
            weight,
            bias,
            eps: eps_f,
            name: name.map(|s| s.to_string()),
            weight_grad: Arc::new(RwLock::new(Array1::zeros(_num_channels))),
            bias_grad: Arc::new(RwLock::new(Array1::zeros(_num_channels))),
            cached_x_norm: Arc::new(RwLock::new(None)),
            cached_mean: Arc::new(RwLock::new(None)),
            cached_var: Arc::new(RwLock::new(None)),
            cached_spatial_size: Arc::new(RwLock::new(0)),
        })
    }

    /// Validate that input has 4 dimensions matching expected channels.
    fn validate_input(&self, input: &Array<F, IxDyn>) -> Result<(usize, usize, usize, usize)> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "LayerNorm2D expects 4-D input [N, C, H, W], got {:?}",
                shape
            )));
        }
        if shape[1] != self.num_channels {
            return Err(NeuralError::InferenceError(format!(
                "LayerNorm2D: expected {} channels, got {}",
                self.num_channels, shape[1]
            )));
        }
        Ok((shape[0], shape[1], shape[2], shape[3]))
    }
}

// Safety: interior mutability is guarded by RwLock
unsafe impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Send for LayerNorm2D<F> {}
unsafe impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Sync for LayerNorm2D<F> {}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for LayerNorm2D<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let (n, c, h, w) = self.validate_input(input)?;
        let spatial_size = h * w;

        let spatial_f = F::from(spatial_size).ok_or_else(|| {
            NeuralError::InferenceError("LayerNorm2D: spatial size overflows float".to_string())
        })?;

        // Allocate output and per-(n,c) statistics
        let mut output = Array::zeros(IxDyn(&[n, c, h, w]));
        // mean and var each: [N, C]
        let mut mean_arr = Array::zeros(IxDyn(&[n, c]));
        let mut var_arr = Array::zeros(IxDyn(&[n, c]));
        let mut x_norm_arr = Array::zeros(IxDyn(&[n, c, h, w]));

        for ni in 0..n {
            for ci in 0..c {
                // Compute spatial mean
                let mut sum = F::zero();
                for hi in 0..h {
                    for wi in 0..w {
                        sum += input[[ni, ci, hi, wi]];
                    }
                }
                let mean = sum / spatial_f;
                mean_arr[[ni, ci]] = mean;

                // Compute spatial variance (biased)
                let mut var_sum = F::zero();
                for hi in 0..h {
                    for wi in 0..w {
                        let diff = input[[ni, ci, hi, wi]] - mean;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / spatial_f;
                var_arr[[ni, ci]] = var;

                // Normalize and apply affine transform
                let inv_std = (var + self.eps).sqrt().recip();
                let gamma = self.weight[ci];
                let beta = self.bias[ci];

                for hi in 0..h {
                    for wi in 0..w {
                        let x_hat = (input[[ni, ci, hi, wi]] - mean) * inv_std;
                        x_norm_arr[[ni, ci, hi, wi]] = x_hat;
                        output[[ni, ci, hi, wi]] = gamma * x_hat + beta;
                    }
                }
            }
        }

        // Cache for backward pass
        {
            let mut cache_xn = self.cached_x_norm.write().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: cached_x_norm lock poisoned".to_string())
            })?;
            *cache_xn = Some(x_norm_arr);
        }
        {
            let mut cache_m = self.cached_mean.write().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: cached_mean lock poisoned".to_string())
            })?;
            *cache_m = Some(mean_arr);
        }
        {
            let mut cache_v = self.cached_var.write().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: cached_var lock poisoned".to_string())
            })?;
            *cache_v = Some(var_arr);
        }
        {
            let mut cache_s = self.cached_spatial_size.write().map_err(|_| {
                NeuralError::InferenceError(
                    "LayerNorm2D: cached_spatial_size lock poisoned".to_string(),
                )
            })?;
            *cache_s = spatial_size;
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let (n, c, h, w) = self.validate_input(grad_output)?;

        let x_norm = {
            let cache = self.cached_x_norm.read().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: cached_x_norm lock poisoned".to_string())
            })?;
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError(
                    "LayerNorm2D backward called before forward".to_string(),
                )
            })?
        };
        let var_cache = {
            let cache = self.cached_var.read().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: cached_var lock poisoned".to_string())
            })?;
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError(
                    "LayerNorm2D backward called before forward".to_string(),
                )
            })?
        };
        let spatial_size = {
            *self.cached_spatial_size.read().map_err(|_| {
                NeuralError::InferenceError(
                    "LayerNorm2D: cached_spatial_size lock poisoned".to_string(),
                )
            })?
        };

        let spatial_f = F::from(spatial_size).ok_or_else(|| {
            NeuralError::InferenceError("LayerNorm2D: spatial size overflows float".to_string())
        })?;

        // Gradients w.r.t. weight and bias
        let mut d_weight = Array1::zeros(c);
        let mut d_bias = Array1::zeros(c);

        for ci in 0..c {
            let mut dw_acc = F::zero();
            let mut db_acc = F::zero();
            for ni in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        let go = grad_output[[ni, ci, hi, wi]];
                        dw_acc += go * x_norm[[ni, ci, hi, wi]];
                        db_acc += go;
                    }
                }
            }
            d_weight[ci] = dw_acc;
            d_bias[ci] = db_acc;
        }

        // Store parameter gradients
        {
            let mut wg = self.weight_grad.write().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: weight_grad lock poisoned".to_string())
            })?;
            *wg = d_weight;
        }
        {
            let mut bg = self.bias_grad.write().map_err(|_| {
                NeuralError::InferenceError("LayerNorm2D: bias_grad lock poisoned".to_string())
            })?;
            *bg = d_bias;
        }

        // Gradient w.r.t. input (standard LayerNorm backward over spatial dims)
        // grad_x = gamma/sqrt(var+eps) * [ dL/dy - mean(dL/dy) - x_hat * mean(dL/dy * x_hat) ]
        let mut d_input = Array::zeros(IxDyn(&[n, c, h, w]));

        for ni in 0..n {
            for ci in 0..c {
                let gamma = self.weight[ci];
                let inv_std = (var_cache[[ni, ci]] + self.eps).sqrt().recip();

                // sum_dy  = sum_{h,w}(dL/dy[n,c,h,w])
                // sum_dy_xhat = sum_{h,w}(dL/dy[n,c,h,w] * x_hat[n,c,h,w])
                let mut sum_dy = F::zero();
                let mut sum_dy_xhat = F::zero();
                for hi in 0..h {
                    for wi in 0..w {
                        let dy = grad_output[[ni, ci, hi, wi]];
                        let xh = x_norm[[ni, ci, hi, wi]];
                        sum_dy += dy;
                        sum_dy_xhat += dy * xh;
                    }
                }

                for hi in 0..h {
                    for wi in 0..w {
                        let dy = grad_output[[ni, ci, hi, wi]];
                        let xh = x_norm[[ni, ci, hi, wi]];
                        // Standard LN backward for normalized mean-zero spatial dims
                        let dx = gamma
                            * inv_std
                            * (dy - sum_dy / spatial_f - xh * sum_dy_xhat / spatial_f);
                        d_input[[ni, ci, hi, wi]] = dx;
                    }
                }
            }
        }

        Ok(d_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        let dw = {
            self.weight_grad
                .read()
                .map_err(|_| {
                    NeuralError::InferenceError(
                        "LayerNorm2D: weight_grad lock poisoned".to_string(),
                    )
                })?
                .clone()
        };
        let db = {
            self.bias_grad
                .read()
                .map_err(|_| {
                    NeuralError::InferenceError("LayerNorm2D: bias_grad lock poisoned".to_string())
                })?
                .clone()
        };

        for ci in 0..self.num_channels {
            self.weight[ci] -= learning_rate * dw[ci];
            self.bias[ci] -= learning_rate * db[ci];
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "LayerNorm2D"
    }

    fn parameter_count(&self) -> usize {
        self.num_channels * 2 // weight + bias
    }

    fn layer_description(&self) -> String {
        format!(
            "type:LayerNorm2D, num_channels:{}, eps:{:?}, params:{}",
            self.num_channels,
            self.eps,
            self.parameter_count()
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![self.weight.clone().into_dyn(), self.bias.clone().into_dyn()]
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.len() < 2 {
            return Err(NeuralError::InvalidArchitecture(
                "LayerNorm2D set_params: expected 2 parameters (weight, bias)".to_string(),
            ));
        }
        // Convert IxDyn back to 1-D
        let w = params[0]
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!(
                    "LayerNorm2D set_params: weight reshape error: {e}"
                ))
            })?;
        let b = params[1]
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!(
                    "LayerNorm2D set_params: bias reshape error: {e}"
                ))
            })?;
        self.weight = w;
        self.bias = b;
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.num_channels])
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.num_channels])
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ParamLayer<F>
    for LayerNorm2D<F>
{
    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        self.params()
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        let dw = self
            .weight_grad
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array1::zeros(self.num_channels));
        let db = self
            .bias_grad
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array1::zeros(self.num_channels));
        vec![dw.into_dyn(), db.into_dyn()]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        self.set_params(&params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;
    use scirs2_core::random::rngs::SmallRng;

    fn make_layer(channels: usize) -> LayerNorm2D<f64> {
        LayerNorm2D::<f64>::new::<SmallRng>(channels, 1e-5, Some("test_ln2d"))
            .expect("Failed to create LayerNorm2D")
    }

    #[test]
    fn test_layer_norm_2d_output_shape() {
        let layer = make_layer(4);
        let input = Array::zeros(IxDyn(&[2, 4, 8, 8]));
        let output = layer.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_layer_norm_2d_normalizes_spatial_dims() {
        // With weight=1, bias=0, after normalization the per-(n,c) output
        // should have mean ≈ 0 and std ≈ 1 over the spatial H×W dimensions
        // (using population variance = biased var → std = sqrt(biased_var))
        let layer = make_layer(2);

        // Use random-ish input by filling with recognizable pattern
        let n = 1usize;
        let c = 2usize;
        let h = 4usize;
        let w = 4usize;
        let spatial = (h * w) as f64;
        let mut input = Array::zeros(IxDyn(&[n, c, h, w]));
        let mut val = 0.0f64;
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    input[[0, ci, hi, wi]] = val;
                    val += 1.0;
                }
            }
        }

        let output = layer.forward(&input).expect("Forward failed");

        // Verify mean ≈ 0 and std ≈ 1 per (n,c) slice
        for ci in 0..c {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for hi in 0..h {
                for wi in 0..w {
                    let v = output[[0, ci, hi, wi]];
                    sum += v;
                    sum_sq += v * v;
                }
            }
            let mean = sum / spatial;
            let var = sum_sq / spatial - mean * mean;
            assert!(mean.abs() < 1e-10, "channel {ci}: mean={mean} not ≈ 0");
            // The biased population variance after normalization equals
            // (M-1)/M where M = H*W, plus floating-point rounding; allow 1e-5 tolerance.
            assert!((var - 1.0).abs() < 1e-5, "channel {ci}: var={var} not ≈ 1");
        }
    }

    #[test]
    fn test_layer_norm_2d_backward_shape() {
        let layer = make_layer(3);
        let input = Array::zeros(IxDyn(&[2, 3, 6, 6]));
        let output = layer.forward(&input).expect("Forward failed");
        let grad_out = Array::ones(output.raw_dim());
        let grad_in = layer.backward(&input, &grad_out).expect("Backward failed");
        assert_eq!(grad_in.shape(), input.shape());
    }

    #[test]
    fn test_layer_norm_2d_parameter_count() {
        let layer = make_layer(16);
        assert_eq!(layer.parameter_count(), 32); // 16 + 16
    }

    #[test]
    fn test_layer_norm_2d_update() {
        let mut layer = make_layer(4);
        let input = Array::zeros(IxDyn(&[1, 4, 4, 4]));
        let output = layer.forward(&input).expect("Forward failed");
        let grad_out = Array::ones(output.raw_dim());
        layer.backward(&input, &grad_out).expect("Backward failed");
        layer.update(0.01f64).expect("Update failed");
    }

    #[test]
    fn test_layer_norm_2d_backward_gradient_finite() {
        // Verify gradients are finite for a non-trivial (non-zero) input
        let layer = make_layer(2);
        let mut input = Array::zeros(IxDyn(&[1, 2, 4, 4]));
        // Fill with nontrivial values
        let mut v = -8.0f64;
        for ci in 0..2 {
            for hi in 0..4 {
                for wi in 0..4 {
                    input[[0, ci, hi, wi]] = v;
                    v += 1.0;
                }
            }
        }
        let output = layer.forward(&input).expect("Forward failed");
        let grad_out = Array::ones(output.raw_dim());
        let grad_in = layer.backward(&input, &grad_out).expect("Backward failed");
        for &g in grad_in.iter() {
            assert!(g.is_finite(), "Non-finite gradient encountered: {g}");
        }
    }

    #[test]
    fn test_layer_norm_2d_invalid_channels() {
        let result = LayerNorm2D::<f64>::new::<scirs2_core::random::rngs::SmallRng>(0, 1e-5, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_2d_wrong_input_shape() {
        let layer = make_layer(4);
        // 3-D input should fail
        let bad_input = Array::zeros(IxDyn(&[2, 4, 8]));
        assert!(layer.forward(&bad_input).is_err());
    }

    #[test]
    fn test_layer_norm_2d_param_layer() {
        let mut layer = make_layer(4);
        let params = layer.get_parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].len(), 4);
        assert_eq!(params[1].len(), 4);
        // set_parameters round-trip
        layer
            .set_parameters(params.clone())
            .expect("set_parameters failed");
        let params2 = layer.get_parameters();
        for (a, b) in params[0].iter().zip(params2[0].iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
