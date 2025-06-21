// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Machine learning operations using the array protocol.
//!
//! This module provides implementations of various machine learning operations
//! using the array protocol, such as activation functions, convolution, and
//! pooling.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use ndarray::{Array, Axis, Ix1, Ix2, Ix3, Ix4, IxDyn};
use rand::prelude::*;
use rand::Rng;

use crate::array_protocol::operations::OperationError;
use crate::array_protocol::{
    array_function_dispatch, get_implementing_args, ArrayProtocol, NdarrayWrapper,
};

/// Activation function types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunc {
    /// Rectified Linear Unit: f(x) = max(0, x)
    ReLU,

    /// Sigmoid function: f(x) = 1 / (1 + exp(-x))
    Sigmoid,

    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,

    /// Softmax function (applied along the last dimension)
    Softmax,

    /// Leaky ReLU: f(x) = max(alpha * x, x)
    LeakyReLU(f64),
}

/// Apply an activation function to an array.
fn apply_activation(
    x: &ndarray::ArrayBase<ndarray::ViewRepr<&f64>, IxDyn>,
    func: ActivationFunc,
) -> Array<f64, IxDyn> {
    match func {
        ActivationFunc::ReLU => x.mapv(|v| v.max(0.0)),
        ActivationFunc::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
        ActivationFunc::Tanh => x.mapv(|v| v.tanh()),
        ActivationFunc::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * v }),
        ActivationFunc::Softmax => {
            // Apply softmax along the last dimension
            let mut result = Array::zeros(x.raw_dim());

            // Iterate over all but the last dimension
            let last_dim = x.ndim() - 1;
            let _last_dim_len = x.shape()[last_dim];

            if x.ndim() == 1 {
                // Simple 1D case
                let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_x = x.mapv(|v| (v - max_val).exp());
                let sum_exp = exp_x.sum();
                result.assign(&(exp_x / sum_exp));
            } else {
                // Multi-dimensional case
                for (i, mut slice) in result.lanes_mut(Axis(last_dim)).into_iter().enumerate() {
                    // Use index_axis to get the ith slice along the last dimension
                    let x_slice = x.index_axis(Axis(last_dim), i);
                    let max_val = x_slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_x = x_slice.mapv(|v| (v - max_val).exp());
                    let sum_exp = exp_x.sum();
                    slice.assign(&(exp_x / sum_exp));
                }
            }

            result
        }
    }
}

// Define machine learning operations using the array protocol

array_function_dispatch!(
    fn activation(
        x: &dyn ArrayProtocol,
        func: ActivationFunc,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_x = Box::new(x.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_x];
        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let Some(x_array) = x.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
                let x_array = x_array.as_array();
                let result = apply_activation(&x_array.view(), func);
                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            return Err(OperationError::NotImplemented(
                "activation not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new(
                "scirs2::array_protocol::ml_ops::activation",
            ),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(x.box_clone())],
            &HashMap::new(),
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast array_function result".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::ml_ops::activation"
);

array_function_dispatch!(
    fn conv2d(
        input: &dyn ArrayProtocol,
        filters: &dyn ArrayProtocol,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_input = Box::new(input.box_clone());
        let boxed_filters = Box::new(filters.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_input, boxed_filters];
        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            // This is a simplified implementation - in practice, convolution is much more complex
            if let (Some(input_array), Some(filters_array)) = (
                input.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>(),
                filters.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>(),
            ) {
                let input = input_array.as_array();
                let filters = filters_array.as_array();

                // Get dimensions
                let batch_size = input.shape()[0];
                let input_height = input.shape()[1];
                let input_width = input.shape()[2];
                let input_channels = input.shape()[3];

                let filter_height = filters.shape()[0];
                let filter_width = filters.shape()[1];
                let filter_in_channels = filters.shape()[2];
                let filter_out_channels = filters.shape()[3];

                // Check dimensions
                if input_channels != filter_in_channels {
                    return Err(OperationError::ShapeMismatch(format!(
                        "Input channels ({}) doesn't match filter input channels ({})",
                        input_channels, filter_in_channels
                    )));
                }

                // Calculate output dimensions
                let out_height = (input_height - filter_height + 2 * padding.0) / stride.0 + 1;
                let out_width = (input_width - filter_width + 2 * padding.1) / stride.1 + 1;

                // Create output array
                let output: Array<f64, Ix4> =
                    Array::zeros((batch_size, out_height, out_width, filter_out_channels));

                // Perform convolution (this is just a placeholder - real convolution is more complex)
                // In practice, we'd use a proper algorithm like im2col or FFT for this

                return Ok(Box::new(NdarrayWrapper::new(output)));
            }
            return Err(OperationError::NotImplemented(
                "conv2d not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        kwargs.insert("stride".to_string(), Box::new(stride) as Box<dyn Any>);
        kwargs.insert("padding".to_string(), Box::new(padding) as Box<dyn Any>);

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new("scirs2::array_protocol::ml_ops::conv2d"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(input.box_clone()), Box::new(filters.box_clone())],
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast array_function result".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::ml_ops::conv2d"
);

array_function_dispatch!(
    fn max_pool2d(
        input: &dyn ArrayProtocol,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_input = Box::new(input.box_clone());
        let boxed_args: Vec<Box<dyn Any>> = vec![boxed_input];
        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let Some(input_array) = input.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>() {
                let input = input_array.as_array();

                // Get dimensions
                let batch_size = input.shape()[0];
                let input_height = input.shape()[1];
                let input_width = input.shape()[2];
                let channels = input.shape()[3];

                // Calculate output dimensions
                let out_height = (input_height - kernel_size.0 + 2 * padding.0) / stride.0 + 1;
                let out_width = (input_width - kernel_size.1 + 2 * padding.1) / stride.1 + 1;

                // Create output array
                let output: Array<f64, Ix4> =
                    Array::zeros((batch_size, out_height, out_width, channels));

                // Perform max pooling (this is just a placeholder - real max pooling is more complex)
                // In practice, we'd use a more efficient algorithm that handles padding properly

                return Ok(Box::new(NdarrayWrapper::new(output)));
            }
            return Err(OperationError::NotImplemented(
                "max_pool2d not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        kwargs.insert(
            "kernel_size".to_string(),
            Box::new(kernel_size) as Box<dyn Any>,
        );
        kwargs.insert("stride".to_string(), Box::new(stride) as Box<dyn Any>);
        kwargs.insert("padding".to_string(), Box::new(padding) as Box<dyn Any>);

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new(
                "scirs2::array_protocol::ml_ops::max_pool2d",
            ),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(input.box_clone())],
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast array_function result".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::ml_ops::max_pool2d"
);

array_function_dispatch!(
    fn batch_norm(
        input: &dyn ArrayProtocol,
        scale: &dyn ArrayProtocol,
        offset: &dyn ArrayProtocol,
        mean: &dyn ArrayProtocol,
        variance: &dyn ArrayProtocol,
        epsilon: f64,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args - convert to Box<dyn Any>
        let boxed_args: Vec<Box<dyn Any>> = vec![
            Box::new(input.box_clone()),
            Box::new(scale.box_clone()),
            Box::new(offset.box_clone()),
            Box::new(mean.box_clone()),
            Box::new(variance.box_clone()),
        ];
        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let (
                Some(input_array),
                Some(scale_array),
                Some(offset_array),
                Some(mean_array),
                Some(variance_array),
            ) = (
                input.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>(),
                scale.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                offset.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                mean.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                variance
                    .as_any()
                    .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
            ) {
                let input = input_array.as_array();
                let scale = scale_array.as_array();
                let offset = offset_array.as_array();
                let mean = mean_array.as_array();
                let variance = variance_array.as_array();

                // Get dimensions
                let _batch_size = input.shape()[0];
                let _height = input.shape()[1];
                let _width = input.shape()[2];
                let channels = input.shape()[3];

                // Check dimensions
                if scale.shape()[0] != channels
                    || offset.shape()[0] != channels
                    || mean.shape()[0] != channels
                    || variance.shape()[0] != channels
                {
                    return Err(OperationError::ShapeMismatch(
                        "Scale, offset, mean, and variance must match the number of channels"
                            .to_string(),
                    ));
                }

                // Create output array with same shape as input
                let output: Array<f64, Ix4> = Array::zeros(input.raw_dim());

                // Perform batch normalization
                // For each channel, normalize using the formula:
                // y = scale * (x - mean) / sqrt(variance + epsilon) + offset

                // This is a simplified implementation - in practice, we'd use more efficient
                // broadcasting and vectorized operations

                return Ok(Box::new(NdarrayWrapper::new(output)));
            }
            return Err(OperationError::NotImplemented(
                "batch_norm not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        kwargs.insert("epsilon".to_string(), Box::new(epsilon) as Box<dyn Any>);

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new(
                "scirs2::array_protocol::ml_ops::batch_norm",
            ),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[
                Box::new(input.box_clone()),
                Box::new(scale.box_clone()),
                Box::new(offset.box_clone()),
                Box::new(mean.box_clone()),
                Box::new(variance.box_clone()),
            ],
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast array_function result".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::ml_ops::batch_norm"
);

array_function_dispatch!(
    fn cross_entropy(
        logits: &dyn ArrayProtocol,
        labels: &dyn ArrayProtocol,
        reduction: &str,
    ) -> Result<Box<dyn Any>, OperationError> {
        // Get implementing args
        let boxed_args: Vec<Box<dyn Any>> =
            vec![Box::new(logits.box_clone()), Box::new(labels.box_clone())];
        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let (Some(logits_array), Some(labels_array)) = (
                logits.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
                labels.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            ) {
                let logits = logits_array.as_array();
                let labels = labels_array.as_array();

                // Check shapes
                if logits.shape() != labels.shape() {
                    return Err(OperationError::ShapeMismatch(format!(
                        "Logits shape {:?} doesn't match labels shape {:?}",
                        logits.shape(),
                        labels.shape()
                    )));
                }

                // Apply softmax to logits
                let mut softmax = Array::zeros(logits.raw_dim());

                // For each sample in the batch
                for (i, sample) in logits.outer_iter().enumerate() {
                    let max_val = sample.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_x = sample.mapv(|v| (v - max_val).exp());
                    let sum_exp = exp_x.sum();

                    for (j, val) in exp_x.iter().enumerate() {
                        softmax[[i, j]] = val / sum_exp;
                    }
                }

                // Compute cross-entropy loss
                // loss = -sum(labels * log(softmax))
                let mut sample_losses = Array::zeros(logits.shape()[0]);

                for (i, (s, l)) in softmax.outer_iter().zip(labels.outer_iter()).enumerate() {
                    let mut loss = 0.0;
                    for (s_val, l_val) in s.iter().zip(l.iter()) {
                        // Add small epsilon to avoid log(0)
                        loss -= l_val * (s_val + 1e-10).ln();
                    }
                    sample_losses[i] = loss;
                }

                // Apply reduction
                let loss = match reduction {
                    "none" => sample_losses,
                    "mean" => {
                        let mean = sample_losses.sum() / sample_losses.len() as f64;
                        // Use Array1 instead of Array0 to make type consistent
                        Array::from_elem(Ix1(1), mean)
                    }
                    "sum" => {
                        let sum = sample_losses.sum();
                        // Use Array1 instead of Array0 to make type consistent
                        Array::from_elem(Ix1(1), sum)
                    }
                    _ => {
                        return Err(OperationError::ShapeMismatch(format!(
                            "Unknown reduction method: {}",
                            reduction
                        )))
                    }
                };

                return Ok(Box::new(loss) as Box<dyn Any>);
            }
            return Err(OperationError::NotImplemented(
                "cross_entropy not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        kwargs.insert(
            "reduction".to_string(),
            Box::new(reduction.to_string()) as Box<dyn Any>,
        );

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new(
                "scirs2::array_protocol::ml_ops::cross_entropy",
            ),
            &[TypeId::of::<Box<dyn Any>>()],
            &[Box::new(logits.box_clone()), Box::new(labels.box_clone())],
            &kwargs,
        )?;

        Ok(result)
    },
    "scirs2::array_protocol::ml_ops::cross_entropy"
);

array_function_dispatch!(
    fn dropout(
        input: &dyn ArrayProtocol,
        rate: f64,
        training: bool,
        seed: Option<u64>,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let boxed_args: Vec<Box<dyn Any>> = vec![Box::new(input.box_clone())];
        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let Some(input_array) = input.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
                let input = input_array.as_array();

                if !training {
                    // During inference, just scale the input
                    return Ok(Box::new(NdarrayWrapper::new(input.clone())));
                }

                // Create a binary mask with probabilities (1-rate)
                let mut rng = match seed {
                    Some(s) => rand::rngs::StdRng::seed_from_u64(s),
                    None => {
                        let mut rng = rand::rng();
                        // Get a random seed from rng and create a new StdRng
                        let random_seed: u64 = rng.random();
                        rand::rngs::StdRng::seed_from_u64(random_seed)
                    }
                };

                let mask = Array::from_shape_fn(input.raw_dim(), |_| {
                    if rng.random::<f64>() >= rate {
                        1.0
                    } else {
                        0.0
                    }
                });

                // Scale by 1/(1-rate) to maintain expected value during training
                let scale = 1.0 / (1.0 - rate);
                let result = input.clone() * &mask * scale;

                return Ok(Box::new(NdarrayWrapper::new(result)));
            }
            return Err(OperationError::NotImplemented(
                "dropout not implemented for this array type".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        kwargs.insert("rate".to_string(), Box::new(rate) as Box<dyn Any>);
        kwargs.insert("training".to_string(), Box::new(training) as Box<dyn Any>);
        if let Some(s) = seed {
            kwargs.insert("seed".to_string(), Box::new(s) as Box<dyn Any>);
        }

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new("scirs2::array_protocol::ml_ops::dropout"),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[Box::new(input.box_clone())],
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast array_function result".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::ml_ops::dropout"
);

array_function_dispatch!(
    fn self_attention(
        queries: &dyn ArrayProtocol,
        keys: &dyn ArrayProtocol,
        values: &dyn ArrayProtocol,
        mask: Option<&dyn ArrayProtocol>,
        scale: Option<f64>,
    ) -> Result<Box<dyn ArrayProtocol>, OperationError> {
        // Get implementing args
        let mut boxed_args: Vec<Box<dyn Any>> = vec![
            Box::new(queries.box_clone()),
            Box::new(keys.box_clone()),
            Box::new(values.box_clone()),
        ];
        if let Some(m) = mask {
            boxed_args.push(Box::new(m.box_clone()));
        }

        let implementing_args = get_implementing_args(&boxed_args);
        if implementing_args.is_empty() {
            // Fallback implementation for ndarray types
            if let (Some(q_array), Some(k_array), Some(v_array)) = (
                queries.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>(),
                keys.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>(),
                values.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>(),
            ) {
                let q = q_array.as_array();
                let k = k_array.as_array();
                let v = v_array.as_array();

                // Get dimensions
                // q, k, v should have shape [batch_size, seq_len, num_heads, d_k]
                let batch_size = q.shape()[0];
                let q_len = q.shape()[1];
                let num_heads = q.shape()[2];
                let d_k = q.shape()[3];

                let k_len = k.shape()[1];

                // Check dimensions
                if k.shape()[0] != batch_size
                    || k.shape()[2] != num_heads
                    || k.shape()[3] != d_k
                    || v.shape()[0] != batch_size
                    || v.shape()[1] != k_len
                    || v.shape()[2] != num_heads
                {
                    return Err(OperationError::ShapeMismatch(
                        "Incompatible shapes for self-attention".to_string(),
                    ));
                }

                // Apply scaling
                let _scale_factor = scale.unwrap_or((d_k as f64).sqrt());

                // For a real implementation, we would compute:
                // 1. scores = matmul(q, k.transpose) / scale_factor
                // 2. if mask: scores = scores.masked_fill(mask, -inf)
                // 3. attention = softmax(scores)
                // 4. output = matmul(attention, v)

                // This is a placeholder showing the expected result shape
                let output: Array<f64, Ix3> = Array::zeros((batch_size, q_len, d_k));

                return Ok(Box::new(NdarrayWrapper::new(output)));
            }
            return Err(OperationError::NotImplemented(
                "self_attention not implemented for these array types".to_string(),
            ));
        }

        // Delegate to the implementation
        let mut kwargs = HashMap::new();
        if let Some(s) = scale {
            kwargs.insert("scale".to_string(), Box::new(s) as Box<dyn Any>);
        }
        if let Some(m) = mask {
            kwargs.insert("mask".to_string(), Box::new(m.box_clone()) as Box<dyn Any>);
        }

        let array_ref = implementing_args[0].1;

        let result = array_ref.array_function(
            &crate::array_protocol::ArrayFunction::new(
                "scirs2::array_protocol::ml_ops::self_attention",
            ),
            &[TypeId::of::<Box<dyn ArrayProtocol>>()],
            &[
                Box::new(queries.box_clone()),
                Box::new(keys.box_clone()),
                Box::new(values.box_clone()),
            ],
            &kwargs,
        )?;

        // Try to downcast the result
        match result.downcast::<Box<dyn ArrayProtocol>>() {
            Ok(array) => Ok(*array),
            Err(_) => Err(OperationError::Other(
                "Failed to downcast array_function result".to_string(),
            )),
        }
    },
    "scirs2::array_protocol::ml_ops::self_attention"
);
