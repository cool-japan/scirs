//! WebAssembly inference wrapper for neural networks
//!
//! This module provides a pure-Rust inference engine for WASM deployment.
//! No wasm-bindgen bindings are required at the struct level.
//!
//! # Key types
//!
//! - [`WasmTensor`] – heap-allocated f32 tensor
//! - [`WasmLayer`] – inference-only layer enum
//! - [`WasmNeuralNet`] – sequential stack with oxicode serialization

use crate::error::{NeuralError, Result};
use oxicode::{config as oxicode_config, serde as oxicode_serde};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// WasmTensor
// ─────────────────────────────────────────────────────────────────────────────

/// A heap-allocated f32 tensor for WebAssembly inference.
///
/// # Examples
/// ```
/// use scirs2_neural::wasm::WasmTensor;
/// let t = WasmTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]);
/// assert_eq!(t.shape(), &[2, 2]);
/// assert_eq!(t.numel(), 4);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl WasmTensor {
    /// Create a tensor from a data vector and a shape.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Create an all-zeros tensor.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0_f32; n],
            shape,
        }
    }

    /// Returns a reference to the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Raw data slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable raw data.
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    /// Consume `self` and return the raw data vector.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Batch size (first dimension).
    pub fn batch_size(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Reshape without copying. Returns an error if element counts differ.
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Result<Self> {
        let n: usize = new_shape.iter().product();
        if n != self.data.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "WasmTensor::reshape: old numel={} new numel={n}",
                self.data.len()
            )));
        }
        self.shape = new_shape;
        Ok(self)
    }

    /// Apply ReLU element-wise in-place.
    pub fn relu_inplace(&mut self) {
        for v in self.data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }

    /// Apply sigmoid element-wise in-place.
    pub fn sigmoid_inplace(&mut self) {
        for v in self.data.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    /// Apply tanh element-wise in-place.
    pub fn tanh_inplace(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.tanh();
        }
    }

    /// Apply row-wise softmax (last dimension) in-place.
    pub fn softmax_inplace(&mut self) {
        if self.shape.is_empty() || self.data.is_empty() {
            return;
        }
        let last_dim = *self.shape.last().unwrap_or(&1);
        if last_dim == 0 {
            return;
        }
        let batch = self.data.len() / last_dim;
        for b in 0..batch {
            let slice = &mut self.data[b * last_dim..(b + 1) * last_dim];
            let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for v in slice.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in slice.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WasmLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Inference-only layer variants for WASM deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmLayer {
    /// Fully-connected layer: `y = xW^T + b`
    Dense {
        in_features: usize,
        out_features: usize,
        /// Row-major weights `[out_features × in_features]`
        weights: Vec<f32>,
        bias: Vec<f32>,
    },
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation (last dimension)
    Softmax,
    /// Dropout (identity at inference)
    Dropout { rate: f32 },
    /// Layer normalisation
    LayerNorm {
        normalized_shape: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
        eps: f32,
    },
    /// Flatten: `[batch, ...rest]` → `[batch, rest.product()]`
    Flatten,
}

impl WasmLayer {
    /// Human-readable layer type name.
    pub fn type_name(&self) -> &str {
        match self {
            WasmLayer::Dense { .. } => "Dense",
            WasmLayer::ReLU => "ReLU",
            WasmLayer::Sigmoid => "Sigmoid",
            WasmLayer::Tanh => "Tanh",
            WasmLayer::Softmax => "Softmax",
            WasmLayer::Dropout { .. } => "Dropout",
            WasmLayer::LayerNorm { .. } => "LayerNorm",
            WasmLayer::Flatten => "Flatten",
        }
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        match self {
            WasmLayer::Dense { weights, bias, .. } => weights.len() + bias.len(),
            WasmLayer::LayerNorm { weight, bias, .. } => weight.len() + bias.len(),
            _ => 0,
        }
    }

    /// Forward pass.
    pub fn forward(&self, input: WasmTensor) -> Result<WasmTensor> {
        match self {
            WasmLayer::Dense {
                in_features,
                out_features,
                weights,
                bias,
            } => dense_forward(input, *in_features, *out_features, weights, bias),
            WasmLayer::ReLU => {
                let mut t = input;
                t.relu_inplace();
                Ok(t)
            }
            WasmLayer::Sigmoid => {
                let mut t = input;
                t.sigmoid_inplace();
                Ok(t)
            }
            WasmLayer::Tanh => {
                let mut t = input;
                t.tanh_inplace();
                Ok(t)
            }
            WasmLayer::Softmax => {
                let mut t = input;
                t.softmax_inplace();
                Ok(t)
            }
            WasmLayer::Dropout { .. } => Ok(input),
            WasmLayer::LayerNorm {
                normalized_shape,
                weight,
                bias,
                eps,
            } => layer_norm_forward(input, *normalized_shape, weight, bias, *eps),
            WasmLayer::Flatten => {
                let batch = input.batch_size();
                let rest = input.numel() / batch.max(1);
                input.reshape(vec![batch, rest])
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WasmNeuralNet
// ─────────────────────────────────────────────────────────────────────────────

/// Serializable, inference-only sequential neural network for WASM.
///
/// # Examples
/// ```
/// use scirs2_neural::wasm::{WasmNeuralNet, WasmLayer, WasmTensor};
///
/// let mut net = WasmNeuralNet::new("my_model");
/// net.add_layer(WasmLayer::Dense {
///     in_features: 4, out_features: 2,
///     weights: vec![0.1; 4 * 2], bias: vec![0.0; 2],
/// });
/// net.add_layer(WasmLayer::ReLU);
///
/// let input = WasmTensor::from_vec(vec![1.0, 0.0, -1.0, 0.5], vec![1, 4]);
/// let output = net.forward(input).expect("ok");
/// assert_eq!(output.shape(), &[1, 2]);
///
/// let bytes = net.to_bytes().expect("serialize ok");
/// let net2 = WasmNeuralNet::from_bytes(&bytes).expect("deserialize ok");
/// assert_eq!(net2.name(), "my_model");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmNeuralNet {
    name: String,
    layers: Vec<WasmLayer>,
    input_shape: Vec<usize>,
    metadata: HashMap<String, String>,
}

impl WasmNeuralNet {
    /// Create an empty network.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layers: Vec::new(),
            input_shape: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Network name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// All layers.
    pub fn layers(&self) -> &[WasmLayer] {
        &self.layers
    }

    /// Input shape (may be empty if not set).
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Set expected input shape (excluding batch).
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = shape;
    }

    /// Add a metadata entry.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Append a layer.
    pub fn add_layer(&mut self, layer: WasmLayer) {
        self.layers.push(layer);
    }

    /// Total parameter count.
    pub fn total_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum()
    }

    /// Run the full forward pass.
    pub fn forward(&self, input: WasmTensor) -> Result<WasmTensor> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x)?;
        }
        Ok(x)
    }

    /// Serialise to compact binary (oxicode).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let cfg = oxicode_config::standard();
        oxicode_serde::encode_to_vec(self, cfg)
            .map_err(|e| NeuralError::SerializationError(format!("oxicode encode: {e}")))
    }

    /// Deserialise from bytes produced by [`to_bytes`].
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let cfg = oxicode_config::standard();
        let (net, _) = oxicode_serde::decode_from_slice::<Self, _>(data, cfg)
            .map_err(|e| NeuralError::DeserializationError(format!("oxicode decode: {e}")))?;
        Ok(net)
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| NeuralError::SerializationError(format!("json encode: {e}")))
    }

    /// Deserialise from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| NeuralError::DeserializationError(format!("json decode: {e}")))
    }

    /// Print a brief summary of the network.
    pub fn summary(&self) -> String {
        let mut s = format!("WasmNeuralNet '{}'\n", self.name);
        for (i, layer) in self.layers.iter().enumerate() {
            s.push_str(&format!("  [{i}] {}\n", layer.type_name()));
        }
        s.push_str(&format!("Total parameters: {}\n", self.total_parameters()));
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer implementations
// ─────────────────────────────────────────────────────────────────────────────

fn dense_forward(
    input: WasmTensor,
    in_features: usize,
    out_features: usize,
    weights: &[f32],
    bias: &[f32],
) -> Result<WasmTensor> {
    let shape = input.shape().to_vec();
    if shape.len() < 2 {
        return Err(NeuralError::ShapeMismatch(
            "Dense: input must be at least 2-D [batch, features]".to_string(),
        ));
    }
    let feat_dim = *shape.last().unwrap_or(&0);
    if feat_dim != in_features {
        return Err(NeuralError::ShapeMismatch(format!(
            "Dense: expected in_features={in_features}, got {feat_dim}"
        )));
    }
    if weights.len() != out_features * in_features {
        return Err(NeuralError::ShapeMismatch(format!(
            "Dense: weights len {} != {out_features}×{in_features}",
            weights.len()
        )));
    }
    if bias.len() != out_features {
        return Err(NeuralError::ShapeMismatch(format!(
            "Dense: bias len {} != {out_features}",
            bias.len()
        )));
    }
    let batch: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let input_data = input.data();
    let mut output = vec![0.0_f32; batch * out_features];
    for b in 0..batch {
        for o in 0..out_features {
            let mut acc = bias[o];
            for i in 0..in_features {
                acc += input_data[b * in_features + i] * weights[o * in_features + i];
            }
            output[b * out_features + o] = acc;
        }
    }
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(out_features);
    Ok(WasmTensor::from_vec(output, out_shape))
}

fn layer_norm_forward(
    input: WasmTensor,
    normalized_shape: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<WasmTensor> {
    let shape = input.shape().to_vec();
    let feat_dim = *shape.last().unwrap_or(&0);
    if feat_dim != normalized_shape {
        return Err(NeuralError::ShapeMismatch(format!(
            "LayerNorm: expected {normalized_shape}, got {feat_dim}"
        )));
    }
    let batch: usize = (input.numel() / feat_dim.max(1)).max(1);
    let data = input.data().to_vec();
    let mut out_data = vec![0.0_f32; data.len()];
    for b in 0..batch {
        let slice = &data[b * feat_dim..(b + 1) * feat_dim];
        let mean: f32 = slice.iter().sum::<f32>() / feat_dim as f32;
        let var: f32 = slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / feat_dim as f32;
        let std_inv = 1.0 / (var + eps).sqrt();
        for (j, &v) in slice.iter().enumerate() {
            out_data[b * feat_dim + j] = (v - mean) * std_inv * weight[j] + bias[j];
        }
    }
    Ok(WasmTensor::from_vec(out_data, shape))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_net() -> WasmNeuralNet {
        let mut net = WasmNeuralNet::new("tiny");
        net.add_layer(WasmLayer::Dense {
            in_features: 2,
            out_features: 2,
            weights: vec![1.0_f32, 0.0, 0.0, 1.0], // identity
            bias: vec![0.0, 0.0],
        });
        net.add_layer(WasmLayer::ReLU);
        net.add_layer(WasmLayer::Dense {
            in_features: 2,
            out_features: 2,
            weights: vec![0.5_f32, 0.5, 0.5, 0.5],
            bias: vec![0.0, 0.0],
        });
        net
    }

    #[test]
    fn test_wasm_tensor_creation() {
        let t = WasmTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_wasm_tensor_reshape_ok() {
        let t = WasmTensor::from_vec(vec![1.0_f32; 6], vec![2, 3]);
        let t2 = t.reshape(vec![3, 2]).expect("ok");
        assert_eq!(t2.shape(), &[3, 2]);
    }

    #[test]
    fn test_wasm_tensor_reshape_err() {
        let t = WasmTensor::from_vec(vec![1.0_f32; 6], vec![2, 3]);
        assert!(t.reshape(vec![4, 2]).is_err());
    }

    #[test]
    fn test_relu_inplace() {
        let mut t = WasmTensor::from_vec(vec![-1.0_f32, 2.0, -3.0, 4.0], vec![1, 4]);
        t.relu_inplace();
        assert_eq!(t.data(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_sigmoid_range() {
        let mut t = WasmTensor::from_vec(vec![-100.0_f32, 0.0, 100.0], vec![1, 3]);
        t.sigmoid_inplace();
        let d = t.data();
        assert!(d[0] >= 0.0 && d[0] < 0.01);
        assert!((d[1] - 0.5).abs() < 1e-4);
        assert!(d[2] > 0.99 && d[2] <= 1.0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut t = WasmTensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![1, 3]);
        t.softmax_inplace();
        let sum: f32 = t.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn test_dense_identity() {
        let layer = WasmLayer::Dense {
            in_features: 2,
            out_features: 2,
            weights: vec![1.0_f32, 0.0, 0.0, 1.0],
            bias: vec![0.0, 0.0],
        };
        let input = WasmTensor::from_vec(vec![3.0_f32, 4.0], vec![1, 2]);
        let out = layer.forward(input).expect("ok");
        assert!((out.data()[0] - 3.0).abs() < 1e-5);
        assert!((out.data()[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_shape_mismatch_err() {
        let layer = WasmLayer::Dense {
            in_features: 3,
            out_features: 2,
            weights: vec![1.0_f32; 6],
            bias: vec![0.0; 2],
        };
        let input = WasmTensor::from_vec(vec![1.0_f32; 4], vec![1, 4]);
        assert!(layer.forward(input).is_err());
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let feat = 4;
        let layer = WasmLayer::LayerNorm {
            normalized_shape: feat,
            weight: vec![1.0_f32; feat],
            bias: vec![0.0_f32; feat],
            eps: 1e-5,
        };
        let input = WasmTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, feat]);
        let out = layer.forward(input).expect("ok");
        let mean: f32 = out.data().iter().sum::<f32>() / feat as f32;
        assert!(mean.abs() < 1e-4, "mean={mean}");
    }

    #[test]
    fn test_dropout_is_identity() {
        let layer = WasmLayer::Dropout { rate: 0.5 };
        let data = vec![1.0_f32, 2.0, 3.0];
        let input = WasmTensor::from_vec(data.clone(), vec![1, 3]);
        let out = layer.forward(input).expect("ok");
        assert_eq!(out.data(), data.as_slice());
    }

    #[test]
    fn test_flatten_layer() {
        let layer = WasmLayer::Flatten;
        let input = WasmTensor::from_vec(vec![1.0_f32; 24], vec![2, 3, 4]);
        let out = layer.forward(input).expect("ok");
        assert_eq!(out.shape(), &[2, 12]);
    }

    #[test]
    fn test_net_forward() {
        let net = make_tiny_net();
        let input = WasmTensor::from_vec(vec![1.0_f32, -1.0], vec![1, 2]);
        let out = net.forward(input).expect("ok");
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_net_total_params() {
        let net = make_tiny_net();
        assert_eq!(net.total_parameters(), 12); // (4+2) + 0 + (4+2)
    }

    #[test]
    fn test_net_binary_roundtrip() {
        let net = make_tiny_net();
        let bytes = net.to_bytes().expect("serialize ok");
        let net2 = WasmNeuralNet::from_bytes(&bytes).expect("deserialize ok");
        assert_eq!(net2.name(), "tiny");
        assert_eq!(net2.num_layers(), 3);
        assert_eq!(net2.total_parameters(), net.total_parameters());
    }

    #[test]
    fn test_net_json_roundtrip() {
        let net = make_tiny_net();
        let json = net.to_json().expect("json ok");
        let net2 = WasmNeuralNet::from_json(&json).expect("from json ok");
        assert_eq!(net2.name(), "tiny");
        assert_eq!(net2.num_layers(), 3);
    }

    #[test]
    fn test_net_summary() {
        let net = make_tiny_net();
        let s = net.summary();
        assert!(s.contains("tiny"));
        assert!(s.contains("Dense"));
        assert!(s.contains("ReLU"));
    }

    #[test]
    fn test_net_metadata() {
        let mut net = WasmNeuralNet::new("m");
        net.add_metadata("version", "1.0");
        assert_eq!(net.get_metadata("version"), Some("1.0"));
        assert_eq!(net.get_metadata("missing"), None);
    }

    #[test]
    fn test_from_bytes_invalid_err() {
        assert!(WasmNeuralNet::from_bytes(b"not valid data").is_err());
    }

    #[test]
    fn test_net_deterministic() {
        let net = make_tiny_net();
        let input = WasmTensor::from_vec(vec![2.0_f32, 3.0], vec![1, 2]);
        let out1 = net.forward(input.clone()).expect("ok");
        let out2 = net.forward(input).expect("ok");
        for (a, b) in out1.data().iter().zip(out2.data().iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_wasm_layer_type_names() {
        assert_eq!(WasmLayer::ReLU.type_name(), "ReLU");
        assert_eq!(WasmLayer::Sigmoid.type_name(), "Sigmoid");
        assert_eq!(WasmLayer::Flatten.type_name(), "Flatten");
        assert_eq!(WasmLayer::Softmax.type_name(), "Softmax");
        assert_eq!(WasmLayer::Tanh.type_name(), "Tanh");
    }
}
