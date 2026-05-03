//! Quantization-aware training for reduced model size and faster inference

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero_point = n_levels / 2)
    Symmetric,
    /// Asymmetric (min-max) quantization
    Asymmetric,
    /// Per-channel quantization (falls back to Asymmetric here)
    PerChannel,
    /// Dynamic quantization: params computed at run time
    Dynamic,
}

/// Calibration method for determining quantization range
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Use absolute min/max
    MinMax,
    /// Clip outliers at the given percentile (0..1)
    Percentile(f32),
    /// Entropy-based calibration (simplified to MinMax here)
    Entropy,
    /// KL-divergence minimisation (simplified to MinMax here)
    KLDivergence,
}

/// Configuration for quantization-aware training
#[derive(Debug, Clone)]
pub struct QATConfig {
    /// Bits used to quantize weights
    pub weight_bits: u8,
    /// Bits used to quantize activations
    pub activation_bits: u8,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Insert fake-quantization nodes during forward
    pub fake_quantize: bool,
    /// Calibration strategy
    pub calibration: CalibrationMethod,
    /// Number of calibration batches
    pub calibration_batches: usize,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            weight_bits: 8,
            activation_bits: 8,
            scheme: QuantizationScheme::Symmetric,
            fake_quantize: true,
            calibration: CalibrationMethod::MinMax,
            calibration_batches: 100,
        }
    }
}

/// A quantized tensor (integer data + scale/zero-point)
pub struct QuantizedTensor {
    /// Quantized integer values
    pub data: Array2<i32>,
    /// Scale factor: `float_val = (int_val - zero_point) * scale`
    pub scale: f32,
    /// Zero point offset
    pub zero_point: f32,
    /// Bit width used for quantization
    pub bits: u8,
}

// ── Per-tensor statistics for calibration ─────────────────────────────────────

struct TensorStats {
    min: f32,
    max: f32,
    mean: f32,
    count: usize,
}

impl TensorStats {
    fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            count: 0,
        }
    }

    fn update(&mut self, tensor: &ArrayView2<f32>) {
        if tensor.is_empty() {
            return;
        }
        let current_min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let current_max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let current_mean = tensor.iter().sum::<f32>() / tensor.len() as f32;
        self.min = self.min.min(current_min);
        self.max = self.max.max(current_max);
        self.mean = (self.mean * self.count as f32 + current_mean) / (self.count + 1).max(1) as f32;
        self.count += 1;
    }
}

/// Calibration statistics store
struct CalibrationStats {
    stats: HashMap<String, TensorStats>,
}

impl CalibrationStats {
    fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    fn update(&mut self, name: &str, tensor: &ArrayView2<f32>) {
        let entry = self
            .stats
            .entry(name.to_string())
            .or_insert_with(TensorStats::new);
        entry.update(tensor);
    }
}

// ── Quantizer implementations ────────────────────────────────────────────────

trait Quantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor>;
}

struct SymmetricQuantizer {
    bits: u8,
}

impl SymmetricQuantizer {
    fn new(bits: u8) -> Self {
        Self { bits }
    }
}

impl Quantizer for SymmetricQuantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        let abs_max = tensor.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let n_levels = (1u32 << self.bits) as f32;
        let scale = if abs_max > 0.0 {
            (2.0 * abs_max) / (n_levels - 1.0)
        } else {
            1.0
        };
        let zero_point = (n_levels - 1.0) / 2.0;
        let quantized = tensor.mapv(|x| {
            let q = (x / scale + zero_point).round();
            q.clamp(0.0, n_levels - 1.0) as i32
        });
        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            bits: self.bits,
        })
    }
}

struct AsymmetricQuantizer {
    bits: u8,
}

impl AsymmetricQuantizer {
    fn new(bits: u8) -> Self {
        Self { bits }
    }
}

impl Quantizer for AsymmetricQuantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let n_levels = (1u32 << self.bits) as f32;
        let scale = if (max_val - min_val).abs() > 1e-10 {
            (max_val - min_val) / (n_levels - 1.0)
        } else {
            1.0
        };
        let zero_point = -min_val / scale;
        let quantized = tensor.mapv(|x| {
            let q = (x / scale + zero_point).round();
            q.clamp(0.0, n_levels - 1.0) as i32
        });
        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            bits: self.bits,
        })
    }
}

struct PerChannelQuantizer {
    bits: u8,
}

impl PerChannelQuantizer {
    fn new(bits: u8) -> Self {
        Self { bits }
    }
}

impl Quantizer for PerChannelQuantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        // Simplified: delegate to asymmetric quantizer
        AsymmetricQuantizer::new(self.bits).quantize(tensor)
    }
}

struct DynamicQuantizer {
    bits: u8,
}

impl DynamicQuantizer {
    fn new(bits: u8) -> Self {
        Self { bits }
    }
}

impl Quantizer for DynamicQuantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        // Dynamic quantization: compute params from the runtime tensor
        AsymmetricQuantizer::new(self.bits).quantize(tensor)
    }
}

// ── Main QAT manager ─────────────────────────────────────────────────────────

/// Quantization-aware training manager
pub struct QuantizationAwareTraining {
    config: QATConfig,
    calibration_stats: CalibrationStats,
}

impl QuantizationAwareTraining {
    /// Create a new QAT manager
    pub fn new(config: QATConfig) -> Self {
        Self {
            config,
            calibration_stats: CalibrationStats::new(),
        }
    }

    /// Quantize weight tensor
    pub fn quantize_weights(&self, weights: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        self.make_quantizer(self.config.weight_bits)
            .quantize(weights)
    }

    /// Quantize activation tensor
    pub fn quantize_activations(&self, activations: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        self.make_quantizer(self.config.activation_bits)
            .quantize(activations)
    }

    /// Simulate quantization during training (straight-through estimator)
    pub fn fake_quantize(&self, tensor: &ArrayView2<f32>, is_weight: bool) -> Result<Array2<f32>> {
        if !self.config.fake_quantize {
            return Ok(tensor.to_owned());
        }
        let bits = if is_weight {
            self.config.weight_bits
        } else {
            self.config.activation_bits
        };
        let quantized = self.quantize_tensor(tensor, bits)?;
        self.dequantize(&quantized)
    }

    /// Dequantize a quantized tensor back to floating-point
    pub fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Array2<f32>> {
        Ok(quantized
            .data
            .mapv(|x| (x as f32 - quantized.zero_point) * quantized.scale))
    }

    /// Update internal calibration statistics for a named tensor
    pub fn update_calibration_stats(&mut self, tensor: &ArrayView2<f32>, name: &str) {
        self.calibration_stats.update(name, tensor);
    }

    /// Print calibration summary (production use would apply per-tensor scales)
    pub fn apply_calibration(&mut self) -> Result<()> {
        for (name, stats) in &self.calibration_stats.stats {
            println!(
                "Calibration for {}: min={:.4}, max={:.4}, mean={:.4}",
                name, stats.min, stats.max, stats.mean
            );
        }
        Ok(())
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_quantizer(&self, bits: u8) -> Box<dyn Quantizer> {
        match self.config.scheme {
            QuantizationScheme::Symmetric => Box::new(SymmetricQuantizer::new(bits)),
            QuantizationScheme::Asymmetric => Box::new(AsymmetricQuantizer::new(bits)),
            QuantizationScheme::PerChannel => Box::new(PerChannelQuantizer::new(bits)),
            QuantizationScheme::Dynamic => Box::new(DynamicQuantizer::new(bits)),
        }
    }

    pub(crate) fn quantize_tensor(
        &self,
        tensor: &ArrayView2<f32>,
        bits: u8,
    ) -> Result<QuantizedTensor> {
        let (scale, zero_point) = self.compute_quantization_params(tensor, bits)?;
        let n_levels = (1u32 << bits) as f32;
        let quantized = tensor.mapv(|x| {
            let q = (x / scale + zero_point).round();
            q.clamp(0.0, n_levels - 1.0) as i32
        });
        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            bits,
        })
    }

    fn compute_quantization_params(
        &self,
        tensor: &ArrayView2<f32>,
        bits: u8,
    ) -> Result<(f32, f32)> {
        let n_levels = (1u32 << bits) as f32;
        let (min_val, max_val) = match self.config.calibration {
            CalibrationMethod::MinMax => {
                let min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            }
            CalibrationMethod::Percentile(p) => {
                let mut values: Vec<f32> = tensor.iter().cloned().collect();
                values.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN"));
                let low_idx = ((1.0 - p) * values.len() as f32) as usize;
                let high_idx = (p * values.len() as f32) as usize;
                let n = values.len();
                (values[low_idx.min(n - 1)], values[high_idx.min(n - 1)])
            }
            _ => {
                // Entropy/KL: simplified to min-max
                let min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            }
        };

        match self.config.scheme {
            QuantizationScheme::Symmetric => {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = if abs_max > 0.0 {
                    (2.0 * abs_max) / (n_levels - 1.0)
                } else {
                    1.0
                };
                let zero_point = (n_levels - 1.0) / 2.0;
                Ok((scale, zero_point))
            }
            _ => {
                let range = max_val - min_val;
                let scale = if range.abs() > 1e-10 {
                    range / (n_levels - 1.0)
                } else {
                    1.0
                };
                let zero_point = -min_val / scale;
                Ok((scale, zero_point))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization() {
        let quantizer = SymmetricQuantizer::new(8);
        let tensor =
            Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -0.5, 0.5, 0.8]).expect("shape ok");
        let quantized = quantizer.quantize(&tensor.view()).expect("quantize ok");
        assert_eq!(quantized.bits, 8);
        assert!(quantized.scale > 0.0);
    }

    #[test]
    fn test_fake_quantize_passthrough() {
        let config = QATConfig {
            fake_quantize: false,
            ..QATConfig::default()
        };
        let qat = QuantizationAwareTraining::new(config);
        let tensor =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape ok");
        let out = qat
            .fake_quantize(&tensor.view(), true)
            .expect("fake_quantize ok");
        for (a, b) in tensor.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fake_quantize_with_quantization() {
        let qat = QuantizationAwareTraining::new(QATConfig::default());
        let tensor = Array2::from_shape_vec((2, 3), vec![0.0, 0.25, 0.5, 0.75, 1.0, -1.0])
            .expect("shape ok");
        let fake_quantized = qat
            .fake_quantize(&tensor.view(), true)
            .expect("fake_quantize ok");
        assert_eq!(fake_quantized.shape(), tensor.shape());
        for (orig, quant) in tensor.iter().zip(fake_quantized.iter()) {
            // Quantization error should be small for 8-bit
            assert!((orig - quant).abs() < 0.1, "orig={orig}, quant={quant}");
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let qat = QuantizationAwareTraining::new(QATConfig::default());
        let tensor = Array2::from_shape_vec((2, 3), vec![0.0, 0.25, 0.5, 0.75, 1.0, -1.0])
            .expect("shape ok");
        let quantized = qat.quantize_tensor(&tensor.view(), 8).expect("quantize ok");
        let dequantized = qat.dequantize(&quantized).expect("dequantize ok");
        for (orig, dequant) in tensor.iter().zip(dequantized.iter()) {
            assert!(
                (orig - dequant).abs() < 0.02,
                "orig={orig}, dequant={dequant}"
            );
        }
    }

    #[test]
    fn test_asymmetric_quantization() {
        let quantizer = AsymmetricQuantizer::new(8);
        let tensor = Array2::from_shape_vec((1, 4), vec![0.0, 1.0, 2.0, 3.0]).expect("shape ok");
        let quantized = quantizer.quantize(&tensor.view()).expect("quantize ok");
        assert_eq!(quantized.bits, 8);
        assert!(quantized.scale > 0.0);
    }

    #[test]
    fn test_calibration_stats_update() {
        let mut qat = QuantizationAwareTraining::new(QATConfig::default());
        let t = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape ok");
        qat.update_calibration_stats(&t.view(), "layer_0");
        qat.apply_calibration().expect("calibration ok");
    }

    #[test]
    fn test_qat_config_default() {
        let config = QATConfig::default();
        assert_eq!(config.weight_bits, 8);
        assert_eq!(config.activation_bits, 8);
        assert!(config.fake_quantize);
        assert_eq!(config.calibration_batches, 100);
    }
}
