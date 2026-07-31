// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Neural network layers and models using the array protocol.
//!
//! This module provides neural network layers and models that work with
//! any array type implementing the ArrayProtocol trait.

use ::ndarray::{Array, Array1, Array4, ArrayD, Axis, Ix1, Ix2, Ix4, IxDyn, Zip};

use rand::{Rng, RngExt, SeedableRng};

use crate::array_protocol::ml_ops::ActivationFunc;
use crate::array_protocol::operations::OperationError;
use crate::array_protocol::{ArrayProtocol, NdarrayWrapper};

/// Gradients produced by a single layer's [`Layer::backward`] pass.
pub struct LayerGrad {
    /// Gradient of the loss with respect to this layer's input, to be
    /// passed as `grad_output` to the previous layer's `backward`.
    pub grad_input: Box<dyn ArrayProtocol>,

    /// Gradients of the loss with respect to each of this layer's
    /// parameters, in the same order as [`Layer::parameters`] /
    /// [`Layer::parameter_names`].
    pub grad_params: Vec<Box<dyn ArrayProtocol>>,
}

/// Trait for neural network layers.
pub trait Layer: Send + Sync {
    /// Forward pass through the layer.
    /// Get the layer type name for serialization.
    fn layer_type(&self) -> &str;

    fn forward(&self, inputs: &dyn ArrayProtocol)
        -> Result<Box<dyn ArrayProtocol>, OperationError>;

    /// Backward pass through the layer: given the layer's original `input`
    /// (as seen by the preceding `forward` call) and `grad_output` (the
    /// gradient of the loss with respect to this layer's output), compute
    /// the gradient with respect to the input and every parameter.
    ///
    /// `forward` does not cache any intermediate state, so implementations
    /// that need it (e.g. a pre-activation value) recompute it from `input`
    /// — cheap relative to the training pipeline's own forward pass, and it
    /// keeps `Layer` object-safe and side-effect-free.
    ///
    /// The default returns [`OperationError::NotImplemented`] so that
    /// existing third-party `Layer` implementations continue to compile;
    /// every layer defined in this module overrides it with a real
    /// implementation (or, where forward itself is already a documented
    /// simplification, an explicit and equally honest error).
    fn backward(
        &self,
        _input: &dyn ArrayProtocol,
        _grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        Err(OperationError::NotImplemented(format!(
            "backward() is not implemented for layer type '{layertype}'",
            layertype = self.layer_type()
        )))
    }

    /// Get the layer's parameters.
    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>>;

    /// Get mutable references to the layer's parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>>;

    /// Update a specific parameter by name
    fn update_parameter(
        &mut self,
        name: &str,
        value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError>;

    /// Get parameter names
    fn parameter_names(&self) -> Vec<String>;

    /// Set the layer to training mode.
    fn train(&mut self);

    /// Set the layer to evaluation mode.
    fn eval(&mut self);

    /// Check if the layer is in training mode.
    fn is_training(&self) -> bool;

    /// Get the layer's name.
    fn name(&self) -> &str;
}

/// Downcasts an `ArrayProtocol` object to an owned `f64` array, regardless
/// of its concrete (static or dynamic) dimensionality. Used internally by
/// `Layer::backward` implementations, which only need to do plain
/// elementwise/linear-algebra math on the underlying data.
fn as_f64_array(a: &dyn ArrayProtocol) -> Result<ArrayD<f64>, OperationError> {
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
        return Ok(w.as_array().clone());
    }
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix1>>() {
        return Ok(w.as_array().clone().into_dyn());
    }
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
        return Ok(w.as_array().clone().into_dyn());
    }
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>() {
        return Ok(w.as_array().clone().into_dyn());
    }
    Err(OperationError::TypeMismatch(
        "Layer::backward currently only supports f64 NdarrayWrapper arrays (Ix1/Ix2/Ix4/IxDyn)"
            .to_string(),
    ))
}

/// Wraps `data` to match the concrete dimensionality that `reference`'s own
/// `NdarrayWrapper` uses, so the result stays dispatch-compatible with
/// operations (like `subtract`/`multiply_by_scalar_f64` in a gradient-descent
/// update) that require both operands to share the same concrete `D`.
/// Falls back to `IxDyn` when `reference` isn't one of the recognized
/// concrete dimensionalities.
fn wrap_like(
    reference: &dyn ArrayProtocol,
    data: ArrayD<f64>,
) -> Result<Box<dyn ArrayProtocol>, OperationError> {
    if reference
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix1>>()
        .is_some()
    {
        let arr = data
            .into_dimensionality::<Ix1>()
            .map_err(|e| OperationError::ShapeMismatch(format!("wrap_like (Ix1): {e}")))?;
        return Ok(Box::new(NdarrayWrapper::new(arr)));
    }
    if reference
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        .is_some()
    {
        let arr = data
            .into_dimensionality::<Ix2>()
            .map_err(|e| OperationError::ShapeMismatch(format!("wrap_like (Ix2): {e}")))?;
        return Ok(Box::new(NdarrayWrapper::new(arr)));
    }
    if reference
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
        .is_some()
    {
        let arr = data
            .into_dimensionality::<Ix4>()
            .map_err(|e| OperationError::ShapeMismatch(format!("wrap_like (Ix4): {e}")))?;
        return Ok(Box::new(NdarrayWrapper::new(arr)));
    }
    Ok(Box::new(NdarrayWrapper::new(data)))
}

/// Elementwise gradient of `ActivationFunc` (see `ml_ops::apply_activation`
/// for the matching forward computation): given the pre-activation `z` and
/// `grad_y` (`dL/dy` where `y = activation(z)`), returns `dL/dz`.
fn activation_grad(
    z: &ArrayD<f64>,
    act: ActivationFunc,
    grad_y: &ArrayD<f64>,
) -> Result<ArrayD<f64>, OperationError> {
    match act {
        ActivationFunc::ReLU => {
            Ok(Zip::from(z)
                .and(grad_y)
                .map_collect(|&zv, &gy| if zv > 0.0 { gy } else { 0.0 }))
        }
        ActivationFunc::Sigmoid => Ok(Zip::from(z).and(grad_y).map_collect(|&zv, &gy| {
            let s = 1.0 / (1.0 + (-zv).exp());
            gy * s * (1.0 - s)
        })),
        ActivationFunc::Tanh => Ok(Zip::from(z).and(grad_y).map_collect(|&zv, &gy| {
            let t = zv.tanh();
            gy * (1.0 - t * t)
        })),
        ActivationFunc::LeakyReLU(alpha) => Ok(Zip::from(z)
            .and(grad_y)
            .map_collect(|&zv, &gy| if zv > 0.0 { gy } else { gy * alpha })),
        ActivationFunc::Softmax => Err(OperationError::NotImplemented(
            "Softmax backward is not implemented: ml_ops::apply_activation's multi-dimensional \
             Softmax normalizes along the array's last axis regardless of what that axis \
             semantically represents for a given layer, so a generic gradient here would not \
             reliably correspond to forward's behavior. Use an element-wise activation (ReLU, \
             Sigmoid, Tanh, or LeakyReLU) on layers that need to be part of a differentiated model."
                .to_string(),
        )),
    }
}

/// Linear (dense/fully-connected) layer.
pub struct Linear {
    /// The layer's name.
    name: String,

    /// Weight matrix.
    weights: Box<dyn ArrayProtocol>,

    /// Bias vector.
    bias: Option<Box<dyn ArrayProtocol>>,

    /// Activation function.
    activation: Option<ActivationFunc>,

    /// Training mode flag.
    training: bool,
}

impl Linear {
    /// Create a new linear layer.
    pub fn new(
        name: &str,
        weights: Box<dyn ArrayProtocol>,
        bias: Option<Box<dyn ArrayProtocol>>,
        activation: Option<ActivationFunc>,
    ) -> Self {
        Self {
            name: name.to_string(),
            weights,
            bias,
            activation,
            training: true,
        }
    }

    /// Create a new linear layer with randomly initialized weights.
    pub fn new_random(
        name: &str,
        in_features: usize,
        out_features: usize,
        withbias: bool,
        activation: Option<ActivationFunc>,
    ) -> Self {
        // Create random weights using Xavier/Glorot initialization
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut rng = rand::rng();
        let weights = Array::from_shape_fn((out_features, in_features), |_| {
            (rng.random::<f64>() * 2.0_f64 - 1.0) * scale
        });

        // Create bias if needed
        let bias = if withbias {
            let bias_array: Array<f64, Ix1> = Array::zeros(out_features);
            Some(Box::new(NdarrayWrapper::new(bias_array)) as Box<dyn ArrayProtocol>)
        } else {
            None
        };

        Self {
            name: name.to_string(),
            weights: Box::new(NdarrayWrapper::new(weights)),
            bias,
            activation,
            training: true,
        }
    }
}

impl Layer for Linear {
    fn layer_type(&self) -> &str {
        "Linear"
    }

    fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Perform matrix multiplication: y = Wx
        let mut result = crate::array_protocol::matmul(self.weights.as_ref(), inputs)?;

        // Add bias if present: y = Wx + b
        if let Some(bias) = &self.bias {
            // Create a temporary for the intermediate result
            let intermediate = crate::array_protocol::add(result.as_ref(), bias.as_ref())?;
            result = intermediate;
        }

        // Apply activation if present
        if let Some(act_fn) = self.activation {
            // Create a temporary for the intermediate result
            let intermediate = crate::array_protocol::ml_ops::activation(result.as_ref(), act_fn)?;
            result = intermediate;
        }

        Ok(result)
    }

    fn backward(
        &self,
        input: &dyn ArrayProtocol,
        grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        let x = as_f64_array(input)?
            .into_dimensionality::<Ix2>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "Linear::backward expects a 2D input (in_features, batch): {e}"
                ))
            })?;
        let w = as_f64_array(self.weights.as_ref())?
            .into_dimensionality::<Ix2>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!("Linear::backward expects 2D weights: {e}"))
            })?;

        // Recompute the pre-activation z = Wx [+ b], since `forward` doesn't
        // cache it and it's needed to evaluate the activation's derivative.
        let mut z = w.dot(&x);
        if let Some(bias) = &self.bias {
            let b = as_f64_array(bias.as_ref())?
                .into_dimensionality::<Ix1>()
                .map_err(|e| {
                    OperationError::ShapeMismatch(format!(
                        "Linear::backward expects a 1D bias: {e}"
                    ))
                })?;
            for mut col in z.axis_iter_mut(Axis(1)) {
                col += &b;
            }
        }

        let grad_y = as_f64_array(grad_output)?
            .into_dimensionality::<Ix2>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "Linear::backward expects a 2D grad_output: {e}"
                ))
            })?;

        let grad_z = match self.activation {
            Some(act) => activation_grad(&z.into_dyn(), act, &grad_y.into_dyn())?
                .into_dimensionality::<Ix2>()
                .map_err(|e| {
                    OperationError::Other(format!(
                        "internal shape error recovering activation_grad's result: {e}"
                    ))
                })?,
            None => grad_y,
        };

        // dL/dW = dL/dz @ x^T ; dL/dx = W^T @ dL/dz
        let grad_w = grad_z.dot(&x.t());
        let grad_input = w.t().dot(&grad_z);

        let mut grad_params = vec![wrap_like(self.weights.as_ref(), grad_w.into_dyn())?];
        if let Some(bias) = &self.bias {
            // dL/db = sum over the batch axis of dL/dz
            let grad_b = grad_z.sum_axis(Axis(1));
            grad_params.push(wrap_like(bias.as_ref(), grad_b.into_dyn())?);
        }

        Ok(LayerGrad {
            grad_input: wrap_like(input, grad_input.into_dyn())?,
            grad_params,
        })
    }

    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        let mut params = vec![self.weights.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>> {
        let mut params = vec![&mut self.weights];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }

    fn update_parameter(
        &mut self,
        name: &str,
        value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError> {
        match name {
            "weights" => {
                self.weights = value;
                Ok(())
            }
            "bias" => {
                self.bias = Some(value);
                Ok(())
            }
            _ => Err(OperationError::Other(format!("Unknown parameter: {name}"))),
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names = vec!["weights".to_string()];
        if self.bias.is_some() {
            names.push("bias".to_string());
        }
        names
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Convolutional layer.
pub struct Conv2D {
    /// The layer's name.
    name: String,

    /// Filters tensor.
    filters: Box<dyn ArrayProtocol>,

    /// Bias vector.
    bias: Option<Box<dyn ArrayProtocol>>,

    /// Stride for the convolution.
    stride: (usize, usize),

    /// Padding for the convolution.
    padding: (usize, usize),

    /// Activation function.
    activation: Option<ActivationFunc>,

    /// Training mode flag.
    training: bool,
}

impl Conv2D {
    /// Create a new convolutional layer.
    pub fn new(
        name: &str,
        filters: Box<dyn ArrayProtocol>,
        bias: Option<Box<dyn ArrayProtocol>>,
        stride: (usize, usize),
        padding: (usize, usize),
        activation: Option<ActivationFunc>,
    ) -> Self {
        Self {
            name: name.to_string(),
            filters,
            bias,
            stride,
            padding,
            activation,
            training: true,
        }
    }

    /// Create a new convolutional layer with randomly initialized weights.
    #[allow(clippy::too_many_arguments)]
    pub fn withshape(
        name: &str,
        filter_height: usize,
        filter_width: usize,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        withbias: bool,
        activation: Option<ActivationFunc>,
    ) -> Self {
        // Create random filters using Kaiming initialization
        let fan_in = filter_height * filter_width * in_channels;
        let scale = (2.0 / fan_in as f64).sqrt();
        let mut rng = rand::rng();
        let filters = Array::from_shape_fn(
            (filter_height, filter_width, in_channels, out_channels),
            |_| (rng.random::<f64>() * 2.0_f64 - 1.0) * scale,
        );

        // Create bias if needed
        let bias = if withbias {
            let bias_array: Array<f64, Ix1> = Array::zeros(out_channels);
            Some(Box::new(NdarrayWrapper::new(bias_array)) as Box<dyn ArrayProtocol>)
        } else {
            None
        };

        Self {
            name: name.to_string(),
            filters: Box::new(NdarrayWrapper::new(filters)),
            bias,
            stride,
            padding,
            activation,
            training: true,
        }
    }
}

impl Layer for Conv2D {
    fn layer_type(&self) -> &str {
        "Conv2D"
    }

    fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Perform convolution
        let mut result = crate::array_protocol::ml_ops::conv2d(
            inputs,
            self.filters.as_ref(),
            self.stride,
            self.padding,
        )?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            result = crate::array_protocol::add(result.as_ref(), bias.as_ref())?;
        }

        // Apply activation if present
        if let Some(act_fn) = self.activation {
            result = crate::array_protocol::ml_ops::activation(result.as_ref(), act_fn)?;
        }

        Ok(result)
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(
        &self,
        input: &dyn ArrayProtocol,
        grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        let inputarr = as_f64_array(input)?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!("Conv2D::backward expects a 4D input: {e}"))
            })?;
        let filters = as_f64_array(self.filters.as_ref())?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!("Conv2D::backward expects 4D filters: {e}"))
            })?;

        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let batch_size = inputarr.shape()[0];
        let input_height = inputarr.shape()[1];
        let input_width = inputarr.shape()[2];
        let input_channels = inputarr.shape()[3];
        let filter_height = filters.shape()[0];
        let filter_width = filters.shape()[1];
        let filter_out_channels = filters.shape()[3];
        let out_height = (input_height - filter_height + 2 * pad_h) / stride_h + 1;
        let out_width = (input_width - filter_width + 2 * pad_w) / stride_w + 1;

        // Recompute the pre-activation convolution output (forward doesn't
        // cache it), needed to evaluate the activation's derivative.
        let mut conv_out =
            Array4::<f64>::zeros((batch_size, out_height, out_width, filter_out_channels));
        for b in 0..batch_size {
            for oc in 0..filter_out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;
                        for fh in 0..filter_height {
                            for fw in 0..filter_width {
                                let in_h = (oh * stride_h) as i32 + fh as i32 - pad_h as i32;
                                let in_w = (ow * stride_w) as i32 + fw as i32 - pad_w as i32;
                                if in_h >= 0
                                    && in_h < input_height as i32
                                    && in_w >= 0
                                    && in_w < input_width as i32
                                {
                                    for ic in 0..input_channels {
                                        sum += inputarr[[b, in_h as usize, in_w as usize, ic]]
                                            * filters[[fh, fw, ic, oc]];
                                    }
                                }
                            }
                        }
                        conv_out[[b, oh, ow, oc]] = sum;
                    }
                }
            }
        }
        if let Some(bias) = &self.bias {
            let bias_arr = as_f64_array(bias.as_ref())?;
            for b in 0..batch_size {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        for oc in 0..filter_out_channels {
                            conv_out[[b, oh, ow, oc]] += bias_arr[[oc]];
                        }
                    }
                }
            }
        }

        let grad_y = as_f64_array(grad_output)?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "Conv2D::backward expects a 4D grad_output: {e}"
                ))
            })?;

        let grad_preact = match self.activation {
            Some(act) => activation_grad(&conv_out.into_dyn(), act, &grad_y.into_dyn())?
                .into_dimensionality::<Ix4>()
                .map_err(|e| {
                    OperationError::Other(format!(
                        "internal shape error recovering activation_grad's result: {e}"
                    ))
                })?,
            None => grad_y,
        };

        // Backward via loop transposition of the exact forward computation:
        // every (b, oc, oh, ow, fh, fw, ic) contribution to `conv_out` above
        // contributes symmetrically to `grad_filters` and `grad_input`.
        let mut grad_input =
            Array4::<f64>::zeros((batch_size, input_height, input_width, input_channels));
        let mut grad_filters = Array4::<f64>::zeros((
            filter_height,
            filter_width,
            input_channels,
            filter_out_channels,
        ));

        for b in 0..batch_size {
            for oc in 0..filter_out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let g = grad_preact[[b, oh, ow, oc]];
                        for fh in 0..filter_height {
                            for fw in 0..filter_width {
                                let in_h = (oh * stride_h) as i32 + fh as i32 - pad_h as i32;
                                let in_w = (ow * stride_w) as i32 + fw as i32 - pad_w as i32;
                                if in_h >= 0
                                    && in_h < input_height as i32
                                    && in_w >= 0
                                    && in_w < input_width as i32
                                {
                                    let (in_h, in_w) = (in_h as usize, in_w as usize);
                                    for ic in 0..input_channels {
                                        grad_filters[[fh, fw, ic, oc]] +=
                                            g * inputarr[[b, in_h, in_w, ic]];
                                        grad_input[[b, in_h, in_w, ic]] +=
                                            g * filters[[fh, fw, ic, oc]];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut grad_params = vec![wrap_like(self.filters.as_ref(), grad_filters.into_dyn())?];
        if let Some(bias) = &self.bias {
            let mut grad_bias = Array1::<f64>::zeros(filter_out_channels);
            for b in 0..batch_size {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        for oc in 0..filter_out_channels {
                            grad_bias[oc] += grad_preact[[b, oh, ow, oc]];
                        }
                    }
                }
            }
            grad_params.push(wrap_like(bias.as_ref(), grad_bias.into_dyn())?);
        }

        Ok(LayerGrad {
            grad_input: wrap_like(input, grad_input.into_dyn())?,
            grad_params,
        })
    }

    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        let mut params = vec![self.filters.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>> {
        let mut params = vec![&mut self.filters];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }

    fn update_parameter(
        &mut self,
        name: &str,
        value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError> {
        match name {
            "filters" => {
                self.filters = value;
                Ok(())
            }
            "bias" => {
                self.bias = Some(value);
                Ok(())
            }
            _ => Err(OperationError::Other(format!("Unknown parameter: {name}"))),
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names = vec!["filters".to_string()];
        if self.bias.is_some() {
            names.push("bias".to_string());
        }
        names
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Builder for creating Conv2D layers
pub struct Conv2DBuilder {
    name: String,
    filter_height: usize,
    filter_width: usize,
    in_channels: usize,
    out_channels: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    withbias: bool,
    activation: Option<ActivationFunc>,
}

impl Conv2DBuilder {
    /// Create a new Conv2D builder
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            filter_height: 3,
            filter_width: 3,
            in_channels: 1,
            out_channels: 1,
            stride: (1, 1),
            padding: (0, 0),
            withbias: true,
            activation: None,
        }
    }

    /// Set filter dimensions
    pub const fn filter_size(mut self, height: usize, width: usize) -> Self {
        self.filter_height = height;
        self.filter_width = width;
        self
    }

    /// Set input and output channels
    pub const fn channels(mut self, input: usize, output: usize) -> Self {
        self.in_channels = input;
        self.out_channels = output;
        self
    }

    /// Set stride
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Set whether to include bias
    pub fn withbias(mut self, withbias: bool) -> Self {
        self.withbias = withbias;
        self
    }

    /// Set activation function
    pub fn activation(mut self, activation: ActivationFunc) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Build the Conv2D layer
    pub fn build(self) -> Conv2D {
        Conv2D::withshape(
            &self.name,
            self.filter_height,
            self.filter_width,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.padding,
            self.withbias,
            self.activation,
        )
    }
}

/// Max pooling layer.
#[allow(dead_code)]
pub struct MaxPool2D {
    /// The layer's name.
    name: String,

    /// Kernel size.
    kernel_size: (usize, usize),

    /// Stride.
    stride: (usize, usize),

    /// Padding.
    padding: (usize, usize),

    /// Training mode flag.
    training: bool,
}

impl MaxPool2D {
    /// Create a new max pooling layer.
    pub fn new(
        name: &str,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);

        Self {
            name: name.to_string(),
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }
}

impl Layer for MaxPool2D {
    fn layer_type(&self) -> &str {
        "MaxPool2D"
    }

    fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Use the max_pool2d implementation from ml_ops module
        crate::array_protocol::ml_ops::max_pool2d(
            inputs,
            self.kernel_size,
            self.stride,
            self.padding,
        )
    }

    fn backward(
        &self,
        input: &dyn ArrayProtocol,
        grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        let inputarr = as_f64_array(input)?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "MaxPool2D::backward expects a 4D input: {e}"
                ))
            })?;
        let grad_y = as_f64_array(grad_output)?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "MaxPool2D::backward expects a 4D grad_output: {e}"
                ))
            })?;

        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let batch_size = inputarr.shape()[0];
        let input_height = inputarr.shape()[1];
        let input_width = inputarr.shape()[2];
        let channels = inputarr.shape()[3];
        let out_height = grad_y.shape()[1];
        let out_width = grad_y.shape()[2];

        let mut grad_input =
            Array4::<f64>::zeros((batch_size, input_height, input_width, channels));

        // Route each output cell's gradient to whichever input cell was the
        // max in its pooling window (accumulating, since overlapping
        // windows can route to the same input cell more than once).
        for b in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut argmax: Option<(usize, usize)> = None;
                        for k_h in 0..kernel_h {
                            for k_w in 0..kernel_w {
                                let in_h = (out_h * stride_h) as i32 + k_h as i32 - pad_h as i32;
                                let in_w = (out_w * stride_w) as i32 + k_w as i32 - pad_w as i32;
                                if in_h >= 0
                                    && in_h < input_height as i32
                                    && in_w >= 0
                                    && in_w < input_width as i32
                                {
                                    let val = inputarr[[b, in_h as usize, in_w as usize, c]];
                                    if val > max_val {
                                        max_val = val;
                                        argmax = Some((in_h as usize, in_w as usize));
                                    }
                                }
                            }
                        }
                        if let Some((in_h, in_w)) = argmax {
                            grad_input[[b, in_h, in_w, c]] += grad_y[[b, out_h, out_w, c]];
                        }
                    }
                }
            }
        }

        Ok(LayerGrad {
            grad_input: wrap_like(input, grad_input.into_dyn())?,
            grad_params: Vec::new(),
        })
    }

    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        // Pooling layers have no parameters
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>> {
        // Pooling layers have no parameters
        Vec::new()
    }

    fn update_parameter(
        &mut self,
        name: &str,
        _value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError> {
        Err(OperationError::Other(format!(
            "MaxPool2D has no parameter: {name}"
        )))
    }

    fn parameter_names(&self) -> Vec<String> {
        // Pooling layers have no parameters
        Vec::new()
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Batch normalization layer.
pub struct BatchNorm {
    /// The layer's name.
    name: String,

    /// Scale parameter.
    scale: Box<dyn ArrayProtocol>,

    /// Offset parameter.
    offset: Box<dyn ArrayProtocol>,

    /// Running mean (for inference).
    running_mean: Box<dyn ArrayProtocol>,

    /// Running variance (for inference).
    running_var: Box<dyn ArrayProtocol>,

    /// Epsilon for numerical stability.
    epsilon: f64,

    /// Training mode flag.
    training: bool,
}

impl BatchNorm {
    /// Create a new batch normalization layer.
    pub fn new(
        name: &str,
        scale: Box<dyn ArrayProtocol>,
        offset: Box<dyn ArrayProtocol>,
        running_mean: Box<dyn ArrayProtocol>,
        running_var: Box<dyn ArrayProtocol>,
        epsilon: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            scale,
            offset,
            running_mean,
            running_var,
            epsilon,
            training: true,
        }
    }

    /// Create a new batch normalization layer with initialized parameters.
    pub fn withshape(
        name: &str,
        num_features: usize,
        epsilon: Option<f64>,
        _momentum: Option<f64>,
    ) -> Self {
        // Initialize parameters with explicit types
        let scale: Array<f64, Ix1> = Array::ones(num_features);
        let offset: Array<f64, Ix1> = Array::zeros(num_features);
        let running_mean: Array<f64, Ix1> = Array::zeros(num_features);
        let running_var: Array<f64, Ix1> = Array::ones(num_features);

        Self {
            name: name.to_string(),
            scale: Box::new(NdarrayWrapper::new(scale)),
            offset: Box::new(NdarrayWrapper::new(offset)),
            running_mean: Box::new(NdarrayWrapper::new(running_mean)),
            running_var: Box::new(NdarrayWrapper::new(running_var)),
            epsilon: epsilon.unwrap_or(1e-5),
            training: true,
        }
    }
}

impl Layer for BatchNorm {
    fn layer_type(&self) -> &str {
        "BatchNorm"
    }

    fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        crate::array_protocol::ml_ops::batch_norm(
            inputs,
            self.scale.as_ref(),
            self.offset.as_ref(),
            self.running_mean.as_ref(),
            self.running_var.as_ref(),
            self.epsilon,
        )
    }

    fn backward(
        &self,
        input: &dyn ArrayProtocol,
        grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        // `ml_ops::batch_norm`'s forward always normalizes with the given
        // `mean`/`variance` directly (it does not compute batch statistics
        // even in training mode), so — matching that — they're treated here
        // as constants rather than differentiated through: only `scale` and
        // `offset` are learnable parameters (see `parameters()` below).
        let x = as_f64_array(input)?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "BatchNorm::backward expects a 4D input: {e}"
                ))
            })?;
        let grad_y = as_f64_array(grad_output)?
            .into_dimensionality::<Ix4>()
            .map_err(|e| {
                OperationError::ShapeMismatch(format!(
                    "BatchNorm::backward expects a 4D grad_output: {e}"
                ))
            })?;
        let scale = as_f64_array(self.scale.as_ref())?;
        let mean = as_f64_array(self.running_mean.as_ref())?;
        let variance = as_f64_array(self.running_var.as_ref())?;

        let (batch_size, height, width, channels) = x.dim();
        let mut grad_input = Array4::<f64>::zeros((batch_size, height, width, channels));
        let mut grad_scale = Array1::<f64>::zeros(channels);
        let mut grad_offset = Array1::<f64>::zeros(channels);

        for c in 0..channels {
            let inv_std = 1.0 / (variance[[c]] + self.epsilon).sqrt();
            let m = mean[[c]];
            let s = scale[[c]];
            let mut grad_scale_c = 0.0;
            let mut grad_offset_c = 0.0;
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let gy = grad_y[[b, h, w, c]];
                        let normalized = (x[[b, h, w, c]] - m) * inv_std;
                        grad_scale_c += gy * normalized;
                        grad_offset_c += gy;
                        grad_input[[b, h, w, c]] = gy * s * inv_std;
                    }
                }
            }
            grad_scale[c] = grad_scale_c;
            grad_offset[c] = grad_offset_c;
        }

        Ok(LayerGrad {
            grad_input: wrap_like(input, grad_input.into_dyn())?,
            grad_params: vec![
                wrap_like(self.scale.as_ref(), grad_scale.into_dyn())?,
                wrap_like(self.offset.as_ref(), grad_offset.into_dyn())?,
            ],
        })
    }

    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        vec![self.scale.clone(), self.offset.clone()]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>> {
        vec![&mut self.scale, &mut self.offset]
    }

    fn update_parameter(
        &mut self,
        name: &str,
        value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError> {
        match name {
            "scale" => {
                self.scale = value;
                Ok(())
            }
            "offset" => {
                self.offset = value;
                Ok(())
            }
            _ => Err(OperationError::Other(format!("Unknown parameter: {name}"))),
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["scale".to_string(), "offset".to_string()]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Dropout layer.
pub struct Dropout {
    /// The layer's name.
    name: String,

    /// Dropout rate.
    rate: f64,

    /// Optional seed for reproducibility.
    seed: Option<u64>,

    /// Training mode flag.
    training: bool,
}

impl Dropout {
    /// Create a new dropout layer.
    pub fn new(name: &str, rate: f64, seed: Option<u64>) -> Self {
        Self {
            name: name.to_string(),
            rate,
            seed,
            training: true,
        }
    }
}

impl Layer for Dropout {
    fn layer_type(&self) -> &str {
        "Dropout"
    }

    fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        crate::array_protocol::ml_ops::dropout(inputs, self.rate, self.training, self.seed)
    }

    fn backward(
        &self,
        input: &dyn ArrayProtocol,
        grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        if !self.training {
            // Inference-mode dropout is the identity function.
            return Ok(LayerGrad {
                grad_input: grad_output.box_clone(),
                grad_params: Vec::new(),
            });
        }

        // `ml_ops::dropout`'s forward doesn't cache the mask it drew, so it
        // can only be reproduced deterministically (and thus differentiated
        // through) when the layer was constructed with a fixed seed.
        let seed = self.seed.ok_or_else(|| {
            OperationError::NotImplemented(
                "Dropout::backward requires a fixed `seed` to deterministically reproduce the \
                 forward mask (forward() does not cache it); construct the layer via \
                 `Dropout::new(name, rate, Some(seed))` to backpropagate through training-mode \
                 dropout"
                    .to_string(),
            )
        })?;

        let grad_y = as_f64_array(grad_output)?;
        let inputarr = as_f64_array(input)?;
        if inputarr.shape() != grad_y.shape() {
            return Err(OperationError::ShapeMismatch(format!(
                "Dropout::backward: input shape {inputshape:?} != grad_output shape {gradshape:?}",
                inputshape = inputarr.shape(),
                gradshape = grad_y.shape()
            )));
        }

        // Reproduce the exact mask `forward()` would have drawn for this
        // input shape, using the same seeded RNG sequence.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let rate = self.rate;
        let mask = Array::from_shape_fn(inputarr.raw_dim(), |_| {
            if rng.random::<f64>() >= rate {
                1.0
            } else {
                0.0
            }
        });
        let scale = 1.0 / (1.0 - self.rate);
        let grad_input = grad_y * &mask * scale;

        Ok(LayerGrad {
            grad_input: wrap_like(input, grad_input)?,
            grad_params: Vec::new(),
        })
    }

    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        // Dropout layers have no parameters
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>> {
        // Dropout layers have no parameters
        Vec::new()
    }

    fn update_parameter(
        &mut self,
        name: &str,
        _value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError> {
        Err(OperationError::Other(format!(
            "Dropout has no parameter: {name}"
        )))
    }

    fn parameter_names(&self) -> Vec<String> {
        // Dropout layers have no parameters
        Vec::new()
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Multi-head attention layer.
pub struct MultiHeadAttention {
    /// The layer's name.
    name: String,

    /// Query projection.
    wq: Box<dyn ArrayProtocol>,

    /// Key projection.
    wk: Box<dyn ArrayProtocol>,

    /// Value projection.
    wv: Box<dyn ArrayProtocol>,

    /// Output projection.
    wo: Box<dyn ArrayProtocol>,

    /// Number of attention heads.
    num_heads: usize,

    /// Model dimension.
    dmodel: usize,

    /// Training mode flag.
    training: bool,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer.
    pub fn new(
        name: &str,
        wq: Box<dyn ArrayProtocol>,
        wk: Box<dyn ArrayProtocol>,
        wv: Box<dyn ArrayProtocol>,
        wo: Box<dyn ArrayProtocol>,
        num_heads: usize,
        dmodel: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            wq,
            wk,
            wv,
            wo,
            num_heads,
            dmodel,
            training: true,
        }
    }

    /// Create a new multi-head attention layer with randomly initialized weights.
    pub fn with_params(name: &str, num_heads: usize, dmodel: usize) -> Self {
        // Check if dmodel is divisible by num_heads
        assert!(
            dmodel % num_heads == 0,
            "dmodel must be divisible by num_heads"
        );

        // Initialize parameters
        let scale = (1.0_f64 / dmodel as f64).sqrt();
        let mut rng = rand::rng();

        let wq = Array::from_shape_fn((dmodel, dmodel), |_| {
            (rng.random::<f64>() * 2.0_f64 - 1.0) * scale
        });

        let wk = Array::from_shape_fn((dmodel, dmodel), |_| {
            (rng.random::<f64>() * 2.0_f64 - 1.0) * scale
        });

        let wv = Array::from_shape_fn((dmodel, dmodel), |_| {
            (rng.random::<f64>() * 2.0_f64 - 1.0) * scale
        });

        let wo = Array::from_shape_fn((dmodel, dmodel), |_| {
            (rng.random::<f64>() * 2.0_f64 - 1.0) * scale
        });

        Self {
            name: name.to_string(),
            wq: Box::new(NdarrayWrapper::new(wq)),
            wk: Box::new(NdarrayWrapper::new(wk)),
            wv: Box::new(NdarrayWrapper::new(wv)),
            wo: Box::new(NdarrayWrapper::new(wo)),
            num_heads,
            dmodel,
            training: true,
        }
    }
}

impl Layer for MultiHeadAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadAttention"
    }

    fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // For a real implementation, this would:
        // 1. Project inputs to queries, keys, and values
        // 2. Reshape for multi-head attention
        // 3. Compute self-attention
        // 4. Reshape and project back to output space

        // This is a simplified placeholder implementation
        let queries = crate::array_protocol::matmul(self.wq.as_ref(), inputs)?;
        let keys = crate::array_protocol::matmul(self.wk.as_ref(), inputs)?;
        let values = crate::array_protocol::matmul(self.wv.as_ref(), inputs)?;

        // Compute self-attention
        let attention = crate::array_protocol::ml_ops::self_attention(
            queries.as_ref(),
            keys.as_ref(),
            values.as_ref(),
            None,
            Some((self.dmodel / self.num_heads) as f64),
        )?;

        // Project back to output space
        let output = crate::array_protocol::matmul(self.wo.as_ref(), attention.as_ref())?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &dyn ArrayProtocol,
        _grad_output: &dyn ArrayProtocol,
    ) -> Result<LayerGrad, OperationError> {
        // `forward()` above is already documented as "a simplified
        // placeholder implementation" that doesn't perform real per-head
        // reshape/splitting — differentiating it exactly wouldn't give a
        // gradient for the multi-head attention operation this layer is
        // meant to represent. Fixing `forward()` for real is a prerequisite
        // for a meaningful `backward()` here.
        Err(OperationError::NotImplemented(
            "MultiHeadAttention::backward is not implemented (forward() is itself a documented \
             simplified placeholder, not real multi-head attention)"
                .to_string(),
        ))
    }

    fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        vec![
            self.wq.clone(),
            self.wk.clone(),
            self.wv.clone(),
            self.wo.clone(),
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Box<dyn ArrayProtocol>> {
        vec![&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo]
    }

    fn update_parameter(
        &mut self,
        name: &str,
        value: Box<dyn ArrayProtocol>,
    ) -> Result<(), OperationError> {
        match name {
            "wq" => {
                self.wq = value;
                Ok(())
            }
            "wk" => {
                self.wk = value;
                Ok(())
            }
            "wv" => {
                self.wv = value;
                Ok(())
            }
            "wo" => {
                self.wo = value;
                Ok(())
            }
            _ => Err(OperationError::Other(format!("Unknown parameter: {name}"))),
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        vec![
            "wq".to_string(),
            "wk".to_string(),
            "wv".to_string(),
            "wo".to_string(),
        ]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Sequential model that chains layers together.
pub struct Sequential {
    /// The model's name.
    name: String,

    /// The layers in the model.
    layers: Vec<Box<dyn Layer>>,

    /// Training mode flag.
    training: bool,
}

impl Sequential {
    /// Create a new sequential model.
    pub fn new(name: &str, layers: Vec<Box<dyn Layer>>) -> Self {
        Self {
            name: name.to_string(),
            layers,
            training: true,
        }
    }

    /// Add a layer to the model.
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    /// Forward pass through the model.
    pub fn forward(
        &self,
        inputs: &dyn ArrayProtocol,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Clone the input to a Box
        let mut x: Box<dyn ArrayProtocol> = inputs.box_clone();

        for layer in &self.layers {
            // Get a reference from the box for the layer
            let x_ref: &dyn ArrayProtocol = x.as_ref();
            // Update x with the layer output
            x = layer.forward(x_ref)?;
        }

        Ok(x)
    }

    /// Get all parameters in the model.
    pub fn parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        let mut params = Vec::new();

        for layer in &self.layers {
            params.extend(layer.parameters());
        }

        params
    }

    /// Set the model to training mode.
    pub fn train(&mut self) {
        self.training = true;

        for layer in &mut self.layers {
            layer.train();
        }
    }

    /// Set the model to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;

        for layer in &mut self.layers {
            layer.eval();
        }
    }

    /// Get the model's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the layers in the model.
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    /// Get mutable access to the layers in the model — e.g. so a
    /// deserializer can restore saved parameter values in place.
    pub fn layers_mut(&mut self) -> &mut [Box<dyn Layer>] {
        &mut self.layers
    }

    /// Backward pass through the model: given the original `input` fed to
    /// `forward` and `grad_output` (the gradient of the loss with respect
    /// to the model's final output — e.g. `Loss::backward`'s result),
    /// computes the gradient for every parameter in every layer via
    /// backpropagation. The returned dictionary is keyed
    /// `"{layer_index}.{param_name}"`, matching [`Self::all_parameter_names`]
    /// and [`Self::update_parameter`]'s expected format.
    ///
    /// `Layer::forward` doesn't cache intermediate activations, so this
    /// first recomputes the forward pass (retaining each layer's input) and
    /// then walks the layers in reverse, threading each layer's
    /// `grad_input` to the previous layer as its `grad_output`.
    pub fn backward(
        &self,
        input: &dyn ArrayProtocol,
        grad_output: &dyn ArrayProtocol,
    ) -> Result<crate::array_protocol::grad::GradientDict, crate::error::CoreError> {
        let mut gradients = crate::array_protocol::grad::GradientDict::new();
        if self.layers.is_empty() {
            return Ok(gradients);
        }

        // Recompute the forward pass, caching each layer's input so that
        // `activations[i]` is the input seen by `self.layers[i]`.
        let mut activations: Vec<Box<dyn ArrayProtocol>> =
            Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.box_clone());
        for layer in &self.layers {
            let layer_input: &dyn ArrayProtocol = activations
                .last()
                .ok_or_else(|| {
                    crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(
                        "internal error: activation cache unexpectedly empty".to_string(),
                    ))
                })?
                .as_ref();
            let out = layer.forward(layer_input).map_err(|e| {
                crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "backward(): forward recompute failed in layer '{name}': {e}",
                    name = layer.name()
                )))
            })?;
            activations.push(out);
        }

        // Walk the layers in reverse, propagating the gradient and
        // collecting each layer's parameter gradients.
        let mut grad_current: Box<dyn ArrayProtocol> = grad_output.box_clone();
        for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
            let layer_input: &dyn ArrayProtocol = activations[layer_idx].as_ref();
            let layer_grad = layer
                .backward(layer_input, grad_current.as_ref())
                .map_err(|e| {
                    crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(
                        format!(
                            "backward() failed in layer '{name}' (index {layer_idx}): {e}",
                            name = layer.name()
                        ),
                    ))
                })?;

            let param_names = layer.parameter_names();
            for (param_idx, grad_param) in layer_grad.grad_params.into_iter().enumerate() {
                let param_name = param_names.get(param_idx).ok_or_else(|| {
                    crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(
                        format!(
                            "layer '{name}' (index {layer_idx}) returned {ngrads} parameter \
                             gradient(s) but parameter_names() only has {nnames}",
                            name = layer.name(),
                            ngrads = param_idx + 1,
                            nnames = param_names.len()
                        ),
                    ))
                })?;
                gradients.insert(format!("{layer_idx}.{param_name}"), grad_param);
            }

            grad_current = layer_grad.grad_input;
        }

        Ok(gradients)
    }

    /// Update a parameter in the model
    pub fn update_parameter(
        &mut self,
        param_name: &str,
        gradient: &dyn ArrayProtocol,
        learningrate: f64,
    ) -> Result<(), crate::error::CoreError> {
        // Parse parameter name: layer_index.parameter_name (e.g., "0.weights", "1.bias")
        let parts: Vec<&str> = param_name.split('.').collect();
        if parts.len() != 2 {
            return Err(crate::error::CoreError::ValueError(
                crate::error::ErrorContext::new(format!(
                    "Invalid parameter name format. Expected 'layer_index.param_name', got: {param_name}"
                )),
            ));
        }

        let layer_index: usize = parts[0].parse().map_err(|_| {
            crate::error::CoreError::ValueError(crate::error::ErrorContext::new(format!(
                "Invalid layer index: {layer_idx}",
                layer_idx = parts[0]
            )))
        })?;

        let param_name = parts[1];

        if layer_index >= self.layers.len() {
            return Err(crate::error::CoreError::ValueError(
                crate::error::ErrorContext::new(format!(
                    "Layer index {layer_index} out of bounds (model has {num_layers} layers)",
                    num_layers = self.layers.len()
                )),
            ));
        }

        // Get the current parameter value
        let layer = &mut self.layers[layer_index];
        let current_params = layer.parameters();
        let param_names = layer.parameter_names();

        // Find the parameter by name
        let param_idx = param_names
            .iter()
            .position(|name| name == param_name)
            .ok_or_else(|| {
                crate::error::CoreError::ValueError(crate::error::ErrorContext::new(format!(
                    "Parameter '{param_name}' not found in layer {layer_index}"
                )))
            })?;

        // Perform gradient descent update: param = param - learningrate * gradient
        let current_param = &current_params[param_idx];

        // Multiply gradient by learning _rate
        let scaled_gradient =
            crate::array_protocol::operations::multiply_by_scalar_f64(gradient, learningrate)
                .map_err(|e| {
                    crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(
                        format!("Failed to scale gradient: {e}"),
                    ))
                })?;

        // Subtract scaled gradient from current parameter
        let updated_param = crate::array_protocol::operations::subtract(
            current_param.as_ref(),
            scaled_gradient.as_ref(),
        )
        .map_err(|e| {
            crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to update parameter: {e}"
            )))
        })?;

        // Update the parameter in the layer
        layer
            .update_parameter(param_name, updated_param)
            .map_err(|e| {
                crate::error::CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to set parameter in layer: {e}"
                )))
            })?;

        Ok(())
    }

    /// Get all parameter names in the model with layer prefixes
    pub fn all_parameter_names(&self) -> Vec<String> {
        let mut all_names = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_param_names = layer.parameter_names();
            for param_name in layer_param_names {
                all_names.push(format!("{layer_idx}.{param_name}"));
            }
        }
        all_names
    }

    /// Get all parameters in the model
    pub fn all_parameters(&self) -> Vec<Box<dyn ArrayProtocol>> {
        let mut all_params = Vec::new();
        for layer in &self.layers {
            all_params.extend(layer.parameters());
        }
        all_params
    }
}

/// Example function to create a simple CNN model.
#[allow(dead_code)]
pub fn create_simple_cnn(inputshape: (usize, usize, usize), num_classes: usize) -> Sequential {
    let (height, width, channels) = inputshape;

    let mut model = Sequential::new("SimpleCNN", Vec::new());

    // First convolutional block
    model.add_layer(Box::new(Conv2D::withshape(
        "conv1",
        3,
        3, // Filter size
        channels,
        32,     // In/out channels
        (1, 1), // Stride
        (1, 1), // Padding
        true,   // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(MaxPool2D::new(
        "pool1",
        (2, 2), // Kernel size
        None,   // Stride (default to kernel size)
        (0, 0), // Padding
    )));

    // Second convolutional block
    model.add_layer(Box::new(Conv2D::withshape(
        "conv2",
        3,
        3, // Filter size
        32,
        64,     // In/out channels
        (1, 1), // Stride
        (1, 1), // Padding
        true,   // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(MaxPool2D::new(
        "pool2",
        (2, 2), // Kernel size
        None,   // Stride (default to kernel size)
        (0, 0), // Padding
    )));

    // Flatten layer (implemented as a Linear layer with reshape)

    // Fully connected layers
    model.add_layer(Box::new(Linear::new_random(
        "fc1",
        64 * (height / 4) * (width / 4), // Input features
        128,                             // Output features
        true,                            // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Dropout::new(
        "dropout", 0.5,  // Dropout rate
        None, // No fixed seed
    )));

    model.add_layer(Box::new(Linear::new_random(
        "fc2",
        128,         // Input features
        num_classes, // Output features
        true,        // With bias
        None,        // No activation (will be applied in loss function)
    )));

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol::{self, NdarrayWrapper};
    use ndarray::{Array1, Array2};

    #[test]
    fn test_linear_layer() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a linear layer
        let weights = Array2::<f64>::eye(3);
        let bias = Array1::<f64>::ones(3);

        let layer = Linear::new(
            "linear",
            Box::new(NdarrayWrapper::new(weights)),
            Some(Box::new(NdarrayWrapper::new(bias))),
            Some(ActivationFunc::ReLU),
        );

        // Create input - ensure we use a dynamic array
        // (commented out since we're not using it in the test now)
        // let x = array![[-1.0, 2.0, -3.0]].into_dyn();
        // let input = NdarrayWrapper::new(x);

        // We can't actually run the operation without proper implementation
        // Skip the actual forward pass for now
        // let output = layer.forward(&input).expect("Operation failed");

        // For now, just make sure the layer is created correctly
        assert_eq!(layer.name(), "linear");
        assert!(layer.is_training());
    }

    #[test]
    fn test_sequential_model() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a simple sequential model
        let mut model = Sequential::new("test_model", Vec::new());

        // Add linear layers
        model.add_layer(Box::new(Linear::new_random(
            "fc1",
            3,    // Input features
            2,    // Output features
            true, // With bias
            Some(ActivationFunc::ReLU),
        )));

        model.add_layer(Box::new(Linear::new_random(
            "fc2",
            2,    // Input features
            1,    // Output features
            true, // With bias
            Some(ActivationFunc::Sigmoid),
        )));

        // Just test that the model is constructed correctly
        assert_eq!(model.name(), "test_model");
        assert_eq!(model.layers().len(), 2);
        assert!(model.training);
    }

    #[test]
    fn test_simple_cnn_creation() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a simple CNN
        let model = create_simple_cnn((28, 28, 1), 10);

        // Check the model structure
        assert_eq!(model.layers().len(), 7);
        assert_eq!(model.name(), "SimpleCNN");

        // Check parameters
        let params = model.parameters();
        assert!(!params.is_empty());
    }
}

// Split out to keep neural.rs under the workspace's 2000-line-per-file limit
// (see grad.rs/grad_tests.rs for the same pattern).
#[cfg(test)]
#[path = "neural_backward_tests.rs"]
mod backward_tests;
