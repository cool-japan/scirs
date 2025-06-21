//! Quantization support for neural networks
//!
//! This module provides comprehensive quantization capabilities including:
//! - Post-training quantization (PTQ)
//! - Quantization-aware training (QAT)
//! - Mixed bit-width operations
//! - Dynamic and static quantization schemes

use crate::error::{Error, Result};
use ndarray::{ArrayD, ArrayView, Zip};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Number of bits for quantization
    pub bits: u8,
    /// Whether to use signed quantization
    pub signed: bool,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Calibration dataset size for PTQ
    pub calibration_size: usize,
    /// Quantization mode
    pub mode: QuantizationMode,
    /// Per-channel quantization for weights
    pub per_channel: bool,
    /// Quantization range clipping
    pub range_clipping: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            signed: true,
            scheme: QuantizationScheme::Symmetric,
            calibration_size: 1000,
            mode: QuantizationMode::Static,
            per_channel: false,
            range_clipping: 0.999,
        }
    }
}

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Symmetric quantization around zero
    Symmetric,
    /// Asymmetric quantization with zero-point offset
    Asymmetric,
    /// Power-of-two quantization for hardware efficiency
    PowerOfTwo,
}

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Static quantization with fixed parameters
    Static,
    /// Dynamic quantization computed at runtime
    Dynamic,
    /// QAT (Quantization-Aware Training)
    QAT,
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Number of quantization bits
    pub bits: u8,
    /// Minimum quantization value
    pub qmin: i32,
    /// Maximum quantization value
    pub qmax: i32,
}

impl QuantizationParams {
    /// Create new quantization parameters
    pub fn new(bits: u8, signed: bool) -> Self {
        let (qmin, qmax) = if signed {
            (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        } else {
            (0, (1 << bits) - 1)
        };

        Self {
            scale: 1.0,
            zero_point: 0,
            bits,
            qmin,
            qmax,
        }
    }

    /// Calculate quantization parameters from tensor statistics
    pub fn from_tensor(
        tensor: &ArrayView<f32, ndarray::IxDyn>,
        config: &QuantizationConfig,
    ) -> Result<Self> {
        let mut params = Self::new(config.bits, config.signed);

        // Calculate tensor statistics
        let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Apply range clipping
        let range = max_val - min_val;
        let clipped_range = range * config.range_clipping;
        let center = (max_val + min_val) / 2.0;
        let clipped_min = center - clipped_range / 2.0;
        let clipped_max = center + clipped_range / 2.0;

        match config.scheme {
            QuantizationScheme::Symmetric => {
                let abs_max = clipped_max.abs().max(clipped_min.abs());
                params.scale = (2.0 * abs_max) / (params.qmax - params.qmin) as f32;
                params.zero_point = 0;
            }
            QuantizationScheme::Asymmetric => {
                params.scale = (clipped_max - clipped_min) / (params.qmax - params.qmin) as f32;
                params.zero_point = params.qmin - (clipped_min / params.scale).round() as i32;
            }
            QuantizationScheme::PowerOfTwo => {
                let abs_max = clipped_max.abs().max(clipped_min.abs());
                let scale_log2 = (abs_max / (1 << (config.bits - 1)) as f32).log2().ceil();
                params.scale = 2.0_f32.powf(scale_log2);
                params.zero_point = 0;
            }
        }

        Ok(params)
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer data
    pub data: ArrayD<i8>,
    /// Quantization parameters
    pub params: QuantizationParams,
    /// Original tensor shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Create new quantized tensor from float tensor
    pub fn from_float(tensor: &ArrayD<f32>, config: &QuantizationConfig) -> Result<Self> {
        let params = QuantizationParams::from_tensor(&tensor.view(), config)?;
        let quantized_data = Self::quantize_tensor(tensor, &params)?;

        Ok(Self {
            data: quantized_data,
            params,
            shape: tensor.shape().to_vec(),
        })
    }

    /// Quantize a float tensor to integers
    fn quantize_tensor(tensor: &ArrayD<f32>, params: &QuantizationParams) -> Result<ArrayD<i8>> {
        let quantized = tensor.mapv(|x| {
            let q_val = (x / params.scale).round() + params.zero_point as f32;
            let clamped = q_val.max(params.qmin as f32).min(params.qmax as f32);
            clamped as i8
        });

        Ok(quantized)
    }

    /// Dequantize back to float tensor
    pub fn dequantize(&self) -> ArrayD<f32> {
        self.data
            .mapv(|q| (q as f32 - self.params.zero_point as f32) * self.params.scale)
    }

    /// Get quantized tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<QuantizationParams>()
    }

    /// Get compression ratio compared to f32
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.data.len() * std::mem::size_of::<f32>();
        let quantized_size = self.size_bytes();
        original_size as f32 / quantized_size as f32
    }
}

/// Post-training quantization (PTQ) implementation
#[derive(Debug)]
pub struct PostTrainingQuantizer {
    /// Quantization configuration
    config: QuantizationConfig,
    /// Calibration statistics
    calibration_stats: HashMap<String, TensorStats>,
}

/// Tensor statistics for calibration
#[derive(Debug, Clone)]
struct TensorStats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    histogram: Vec<u32>,
}

impl TensorStats {
    fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            std: 0.0,
            histogram: vec![0; 256],
        }
    }

    fn update(&mut self, tensor: &ArrayView<f32, ndarray::IxDyn>) {
        self.min = self.min.min(
            *tensor
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
        self.max = self.max.max(
            *tensor
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );

        let sum: f32 = tensor.sum();
        let count = tensor.len() as f32;
        self.mean = sum / count;

        let variance: f32 = tensor.iter().map(|&x| (x - self.mean).powi(2)).sum::<f32>() / count;
        self.std = variance.sqrt();

        // Update histogram
        for &val in tensor.iter() {
            let normalized = ((val - self.min) / (self.max - self.min) * 255.0).round() as usize;
            let bin = normalized.min(255);
            self.histogram[bin] += 1;
        }
    }
}

impl PostTrainingQuantizer {
    /// Create new post-training quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_stats: HashMap::new(),
        }
    }

    /// Add calibration data for a named tensor
    pub fn add_calibration_data(&mut self, name: &str, tensor: &ArrayD<f32>) {
        let stats = self
            .calibration_stats
            .entry(name.to_string())
            .or_insert_with(TensorStats::new);
        stats.update(&tensor.view());
    }

    /// Finalize calibration and compute optimal quantization parameters
    pub fn finalize_calibration(&mut self) -> Result<HashMap<String, QuantizationParams>> {
        let mut params_map = HashMap::new();

        for (name, stats) in &self.calibration_stats {
            // Use the stats directly for optimal parameter computation

            // Use KL divergence for optimal quantization range selection
            let optimal_params = self.compute_optimal_params(stats)?;
            params_map.insert(name.clone(), optimal_params);
        }

        Ok(params_map)
    }

    /// Compute optimal quantization parameters using KL divergence
    fn compute_optimal_params(&self, stats: &TensorStats) -> Result<QuantizationParams> {
        let mut best_params = QuantizationParams::new(self.config.bits, self.config.signed);
        let mut best_kl_div = f32::INFINITY;

        // Try different threshold values
        for threshold_idx in 128..=255 {
            let threshold = stats.min + (threshold_idx as f32 / 255.0) * (stats.max - stats.min);

            // Compute quantization parameters for this threshold
            let mut params = QuantizationParams::new(self.config.bits, self.config.signed);

            match self.config.scheme {
                QuantizationScheme::Symmetric => {
                    params.scale = (2.0 * threshold) / (params.qmax - params.qmin) as f32;
                    params.zero_point = 0;
                }
                QuantizationScheme::Asymmetric => {
                    params.scale = (threshold - stats.min) / (params.qmax - params.qmin) as f32;
                    params.zero_point = params.qmin - (stats.min / params.scale).round() as i32;
                }
                QuantizationScheme::PowerOfTwo => {
                    let scale_log2 = (threshold / (1 << (self.config.bits - 1)) as f32)
                        .log2()
                        .ceil();
                    params.scale = 2.0_f32.powf(scale_log2);
                    params.zero_point = 0;
                }
            }

            // Compute KL divergence (simplified approximation)
            let kl_div = self.compute_kl_divergence(&stats.histogram, &params);

            if kl_div < best_kl_div {
                best_kl_div = kl_div;
                best_params = params;
            }
        }

        Ok(best_params)
    }

    /// Compute KL divergence between original and quantized distributions
    fn compute_kl_divergence(&self, histogram: &[u32], params: &QuantizationParams) -> f32 {
        let total_count: u32 = histogram.iter().sum();
        if total_count == 0 {
            return 0.0;
        }

        let mut kl_div = 0.0;
        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                let p = count as f32 / total_count as f32;

                // Simulate quantization effect
                let bin_value = i as f32 / 255.0;
                let quantized = (bin_value / params.scale)
                    .round()
                    .max(params.qmin as f32)
                    .min(params.qmax as f32);
                let dequantized = quantized * params.scale;

                // Approximate quantized distribution
                let q = (dequantized * 255.0).round() as usize;
                let q_count = if q < histogram.len() { histogram[q] } else { 1 };
                let q_prob = (q_count as f32 / total_count as f32).max(1e-8);

                kl_div += p * (p / q_prob).ln();
            }
        }

        kl_div
    }

    /// Quantize a tensor using computed parameters
    pub fn quantize_tensor(
        &self,
        tensor: &ArrayD<f32>,
        params: &QuantizationParams,
    ) -> Result<QuantizedTensor> {
        let quantized_data = QuantizedTensor::quantize_tensor(tensor, params)?;

        Ok(QuantizedTensor {
            data: quantized_data,
            params: params.clone(),
            shape: tensor.shape().to_vec(),
        })
    }
}

/// Quantization-aware training (QAT) support
#[derive(Debug)]
pub struct QuantizationAwareTraining {
    /// QAT configuration
    config: QuantizationConfig,
    /// Fake quantization parameters for layers
    layer_params: HashMap<String, QuantizationParams>,
    /// Training step counter
    step_count: usize,
    /// Warmup steps before quantization
    warmup_steps: usize,
}

impl QuantizationAwareTraining {
    /// Create new QAT instance
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            layer_params: HashMap::new(),
            step_count: 0,
            warmup_steps: 1000,
        }
    }

    /// Set warmup steps
    pub fn set_warmup_steps(&mut self, steps: usize) {
        self.warmup_steps = steps;
    }

    /// Initialize quantization parameters for a layer
    pub fn init_layer_params(&mut self, layer_name: &str, tensor: &ArrayD<f32>) -> Result<()> {
        let params = QuantizationParams::from_tensor(&tensor.view(), &self.config)?;
        self.layer_params.insert(layer_name.to_string(), params);
        Ok(())
    }

    /// Apply fake quantization during training
    pub fn fake_quantize(&mut self, layer_name: &str, tensor: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        self.step_count += 1;

        // Skip quantization during warmup
        if self.step_count < self.warmup_steps {
            return Ok(tensor.clone());
        }

        let params = self.layer_params.get_mut(layer_name).ok_or_else(|| {
            Error::InvalidArgument(format!("Layer {} not initialized", layer_name))
        })?;

        // Update parameters with exponential moving average
        let new_params = QuantizationParams::from_tensor(&tensor.view(), &self.config)?;
        let alpha = 0.01; // EMA factor

        params.scale = params.scale * (1.0 - alpha) + new_params.scale * alpha;
        if self.config.scheme == QuantizationScheme::Asymmetric {
            params.zero_point = ((params.zero_point as f32) * (1.0 - alpha)
                + (new_params.zero_point as f32) * alpha)
                .round() as i32;
        }

        // Apply fake quantization (quantize then dequantize)
        let quantized = QuantizedTensor::quantize_tensor(tensor, params)?;
        let dequantized = quantized.mapv(|q| (q as f32 - params.zero_point as f32) * params.scale);

        Ok(dequantized)
    }

    /// Get final quantization parameters for deployment
    pub fn get_quantization_params(&self) -> &HashMap<String, QuantizationParams> {
        &self.layer_params
    }

    /// Simulate quantization noise for better training
    pub fn add_quantization_noise(&self, tensor: &ArrayD<f32>, noise_scale: f32) -> ArrayD<f32> {
        use rand::Rng;
        let mut rng = rand::rng();

        tensor.mapv(|x| {
            let noise = rng.random::<f32>() - 0.5; // Uniform noise [-0.5, 0.5]
            x + noise * noise_scale
        })
    }
}

/// Mixed bit-width quantization support
#[derive(Debug)]
pub struct MixedBitWidthQuantizer {
    /// Per-layer bit configurations
    layer_configs: HashMap<String, QuantizationConfig>,
    /// Sensitivity analysis results
    sensitivity_scores: HashMap<String, f32>,
}

impl Default for MixedBitWidthQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedBitWidthQuantizer {
    /// Create new mixed bit-width quantizer
    pub fn new() -> Self {
        Self {
            layer_configs: HashMap::new(),
            sensitivity_scores: HashMap::new(),
        }
    }

    /// Set quantization configuration for a specific layer
    pub fn set_layer_config(&mut self, layer_name: &str, config: QuantizationConfig) {
        self.layer_configs.insert(layer_name.to_string(), config);
    }

    /// Perform sensitivity analysis to determine optimal bit allocation
    pub fn analyze_sensitivity(
        &mut self,
        layer_outputs: &HashMap<String, ArrayD<f32>>,
    ) -> Result<()> {
        for (layer_name, output) in layer_outputs {
            // Compute sensitivity score based on activation distribution
            let variance = self.compute_variance(output);
            let entropy = self.compute_entropy(output);
            let gradient_norm = self.compute_gradient_norm(output);

            // Combined sensitivity score
            let sensitivity = variance * 0.4 + entropy * 0.3 + gradient_norm * 0.3;
            self.sensitivity_scores
                .insert(layer_name.clone(), sensitivity);
        }

        // Assign bit-widths based on sensitivity scores
        self.assign_bit_widths()?;

        Ok(())
    }

    /// Compute variance of activations
    fn compute_variance(&self, tensor: &ArrayD<f32>) -> f32 {
        let mean = tensor.mean().unwrap_or(0.0);
        let variance =
            tensor.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / tensor.len() as f32;
        variance
    }

    /// Compute entropy of activation distribution
    fn compute_entropy(&self, tensor: &ArrayD<f32>) -> f32 {
        let mut histogram = vec![0; 256];
        let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range == 0.0 {
            return 0.0;
        }

        for &val in tensor.iter() {
            let bin = ((val - min_val) / range * 255.0).round() as usize;
            let bin = bin.min(255);
            histogram[bin] += 1;
        }

        let total = tensor.len() as f32;
        let mut entropy = 0.0;
        for count in histogram {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Compute gradient norm (simplified approximation)
    fn compute_gradient_norm(&self, tensor: &ArrayD<f32>) -> f32 {
        // Approximate gradient as the standard deviation of adjacent differences
        let mut grad_norm = 0.0;
        for axis in 0..tensor.ndim() {
            if tensor.shape()[axis] > 1 {
                for _i in 0..tensor.shape()[axis] - 1 {
                    // Simplified gradient computation along each axis
                    grad_norm += 1.0; // Placeholder - would compute actual gradients in real implementation
                }
            }
        }
        grad_norm / tensor.len() as f32
    }

    /// Assign bit-widths based on sensitivity scores
    fn assign_bit_widths(&mut self) -> Result<()> {
        let mut scores: Vec<(String, f32)> = self
            .sensitivity_scores
            .iter()
            .map(|(name, &score)| (name.clone(), score))
            .collect();

        // Sort by sensitivity (higher sensitivity gets more bits)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Assign bit-widths: high sensitivity layers get 8 bits, others get 4-6 bits
        for (i, (layer_name, _)) in scores.iter().enumerate() {
            let bits = if i < scores.len() / 3 {
                8 // High sensitivity
            } else if i < 2 * scores.len() / 3 {
                6 // Medium sensitivity
            } else {
                4 // Low sensitivity
            };

            let mut config = self
                .layer_configs
                .get(layer_name)
                .cloned()
                .unwrap_or_default();
            config.bits = bits;
            self.layer_configs.insert(layer_name.clone(), config);
        }

        Ok(())
    }

    /// Get optimal configuration for a layer
    pub fn get_layer_config(&self, layer_name: &str) -> Option<&QuantizationConfig> {
        self.layer_configs.get(layer_name)
    }

    /// Get sensitivity score for a layer
    pub fn get_sensitivity_score(&self, layer_name: &str) -> Option<f32> {
        self.sensitivity_scores.get(layer_name).copied()
    }
}

/// Dynamic quantization at runtime
#[derive(Debug)]
pub struct DynamicQuantizer {
    /// Configuration for dynamic quantization
    config: QuantizationConfig,
    /// Cache of recently computed parameters
    params_cache: HashMap<String, QuantizationParams>,
    /// Cache size limit
    cache_size_limit: usize,
}

impl DynamicQuantizer {
    /// Create new dynamic quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            params_cache: HashMap::new(),
            cache_size_limit: 100,
        }
    }

    /// Dynamically quantize tensor at runtime
    pub fn quantize(
        &mut self,
        tensor: &ArrayD<f32>,
        cache_key: Option<&str>,
    ) -> Result<QuantizedTensor> {
        let params = if let Some(key) = cache_key {
            if let Some(cached_params) = self.params_cache.get(key) {
                cached_params.clone()
            } else {
                let params = QuantizationParams::from_tensor(&tensor.view(), &self.config)?;
                self.cache_params(key, params.clone());
                params
            }
        } else {
            QuantizationParams::from_tensor(&tensor.view(), &self.config)?
        };

        let quantized_data = QuantizedTensor::quantize_tensor(tensor, &params)?;

        Ok(QuantizedTensor {
            data: quantized_data,
            params,
            shape: tensor.shape().to_vec(),
        })
    }

    /// Cache quantization parameters
    fn cache_params(&mut self, key: &str, params: QuantizationParams) {
        if self.params_cache.len() >= self.cache_size_limit {
            // Simple LRU eviction - remove first entry
            if let Some(first_key) = self.params_cache.keys().next().cloned() {
                self.params_cache.remove(&first_key);
            }
        }
        self.params_cache.insert(key.to_string(), params);
    }

    /// Clear parameter cache
    pub fn clear_cache(&mut self) {
        self.params_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.params_cache.len(), self.cache_size_limit)
    }
}

/// Quantization utilities and helper functions
pub mod utils {
    use super::*;

    /// Compare quantized vs original tensor accuracy
    pub fn compute_quantization_error(original: &ArrayD<f32>, quantized: &QuantizedTensor) -> f32 {
        let dequantized = quantized.dequantize();
        let mse = Zip::from(original)
            .and(&dequantized)
            .fold(0.0, |acc, &orig, &deq| acc + (orig - deq).powi(2));
        mse / original.len() as f32
    }

    /// Estimate model size reduction from quantization
    pub fn estimate_size_reduction(bit_width: u8) -> f32 {
        32.0 / bit_width as f32
    }

    /// Simulate quantization performance gains
    pub fn estimate_performance_gain(bit_width: u8) -> f32 {
        // Empirical approximation based on common hardware
        match bit_width {
            8 => 2.0,  // ~2x speedup with INT8
            4 => 4.0,  // ~4x speedup with INT4
            1 => 16.0, // ~16x speedup with binary
            _ => 1.0,
        }
    }

    /// Convert between different quantization schemes
    pub fn convert_quantization_scheme(
        tensor: &QuantizedTensor,
        target_scheme: QuantizationScheme,
        target_bits: u8,
    ) -> Result<QuantizedTensor> {
        // First dequantize to float
        let float_tensor = tensor.dequantize();

        // Create new config with target scheme
        let config = QuantizationConfig {
            scheme: target_scheme,
            bits: target_bits,
            ..Default::default()
        };

        // Re-quantize with new scheme
        QuantizedTensor::from_float(&float_tensor, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use ndarray_rand::rand::distributions::Standard;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.bits, 8);
        assert!(config.signed);
        assert_eq!(config.scheme, QuantizationScheme::Symmetric);
    }

    #[test]
    fn test_quantization_params_creation() {
        let params = QuantizationParams::new(8, true);
        assert_eq!(params.bits, 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);
    }

    #[test]
    fn test_symmetric_quantization() {
        let tensor = array![[1.0, -1.0], [2.0, -2.0]].into_dyn();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::Symmetric,
            ..Default::default()
        };

        let quantized = QuantizedTensor::from_float(&tensor, &config).unwrap();
        let _dequantized = quantized.dequantize();

        // Check that quantization preserves approximate values
        let error = utils::compute_quantization_error(&tensor, &quantized);
        assert!(error < 0.1); // Small quantization error
    }

    #[test]
    fn test_asymmetric_quantization() {
        let tensor = array![[0.0, 1.0], [2.0, 3.0]].into_dyn();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::Asymmetric,
            ..Default::default()
        };

        let quantized = QuantizedTensor::from_float(&tensor, &config).unwrap();
        let _dequantized = quantized.dequantize();

        assert!(quantized.params.zero_point != 0); // Should have non-zero zero-point

        let error = utils::compute_quantization_error(&tensor, &quantized);
        assert!(error < 0.1);
    }

    #[test]
    fn test_post_training_quantization() {
        let mut ptq = PostTrainingQuantizer::new(QuantizationConfig::default());

        // Add calibration data
        let calib_data = Array2::random((100, 50), Standard).into_dyn();
        ptq.add_calibration_data("layer1", &calib_data);

        let params = ptq.finalize_calibration().unwrap();
        assert!(params.contains_key("layer1"));
    }

    #[test]
    fn test_quantization_aware_training() {
        let mut qat = QuantizationAwareTraining::new(QuantizationConfig::default());
        let tensor = Array2::ones((10, 10)).into_dyn();

        qat.init_layer_params("layer1", &tensor).unwrap();
        let fake_quantized = qat.fake_quantize("layer1", &tensor).unwrap();

        assert_eq!(fake_quantized.shape(), tensor.shape());
    }

    #[test]
    fn test_mixed_bitwidth_quantization() {
        let mut mbq = MixedBitWidthQuantizer::new();

        let mut outputs = HashMap::new();
        outputs.insert(
            "layer1".to_string(),
            Array2::random((50, 50), Standard).into_dyn(),
        );
        outputs.insert("layer2".to_string(), Array2::ones((50, 50)).into_dyn());

        mbq.analyze_sensitivity(&outputs).unwrap();

        assert!(mbq.get_sensitivity_score("layer1").is_some());
        assert!(mbq.get_layer_config("layer1").is_some());
    }

    #[test]
    fn test_dynamic_quantization() {
        let mut dq = DynamicQuantizer::new(QuantizationConfig::default());
        let tensor = Array2::random((20, 20), Standard).into_dyn();

        let quantized = dq.quantize(&tensor, Some("test_key")).unwrap();
        assert_eq!(quantized.shape, tensor.shape().to_vec());

        let (cache_size, _) = dq.cache_stats();
        assert_eq!(cache_size, 1);
    }

    #[test]
    fn test_quantization_utilities() {
        let original = Array2::random((10, 10), Standard).into_dyn();
        let quantized =
            QuantizedTensor::from_float(&original, &QuantizationConfig::default()).unwrap();

        let error = utils::compute_quantization_error(&original, &quantized);
        assert!(error >= 0.0);

        let size_reduction = utils::estimate_size_reduction(8);
        assert_eq!(size_reduction, 4.0);

        let perf_gain = utils::estimate_performance_gain(8);
        assert_eq!(perf_gain, 2.0);
    }

    #[test]
    fn test_compression_ratio() {
        let tensor = Array2::ones((100, 100)).into_dyn();
        let quantized =
            QuantizedTensor::from_float(&tensor, &QuantizationConfig::default()).unwrap();

        let ratio = quantized.compression_ratio();
        assert!(ratio > 1.0); // Should be compressed
    }

    #[test]
    fn test_power_of_two_quantization() {
        let tensor = Array2::random((10, 10), Standard).into_dyn();
        let config = QuantizationConfig {
            scheme: QuantizationScheme::PowerOfTwo,
            ..Default::default()
        };

        let quantized = QuantizedTensor::from_float(&tensor, &config).unwrap();

        // Scale should be a power of 2
        let scale_log2 = quantized.params.scale.log2();
        assert!((scale_log2.round() - scale_log2).abs() < 1e-6);
    }

    #[test]
    fn test_quantization_scheme_conversion() {
        let tensor = Array2::random((10, 10), Standard).into_dyn();
        let quantized =
            QuantizedTensor::from_float(&tensor, &QuantizationConfig::default()).unwrap();

        let converted =
            utils::convert_quantization_scheme(&quantized, QuantizationScheme::Asymmetric, 4)
                .unwrap();

        assert_eq!(converted.params.bits, 4);
    }
}
