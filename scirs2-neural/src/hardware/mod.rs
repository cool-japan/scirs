//! Hardware-aware neural network operations
//!
//! This module provides hardware-specific optimizations for neural network inference,
//! including hardware profiling, per-layer quantization targeting specific chips,
//! and hardware-aware pruning strategies.
//!
//! # Key Types
//!
//! - [`HardwareProfile`] – describes capabilities of a target hardware device
//! - [`HardwareOptimizer`] – applies hardware-specific quantization and pruning
//! - [`QuantizationPrecision`] – precision modes supported by hardware

use crate::error::{NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

// ─────────────────────────────────────────────────────────────────────────────
// Hardware profile
// ─────────────────────────────────────────────────────────────────────────────

/// Precision modes available on hardware accelerators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    /// 32-bit floating point (full precision)
    FP32,
    /// 16-bit floating point (half precision)
    FP16,
    /// BFloat16 (brain float)
    BF16,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,
}

impl QuantizationPrecision {
    /// Returns the number of bits for this precision.
    pub fn bits(self) -> u8 {
        match self {
            QuantizationPrecision::FP32 => 32,
            QuantizationPrecision::FP16 | QuantizationPrecision::BF16 => 16,
            QuantizationPrecision::INT8 => 8,
            QuantizationPrecision::INT4 => 4,
        }
    }

    /// Returns whether this precision is floating-point.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            QuantizationPrecision::FP32 | QuantizationPrecision::FP16 | QuantizationPrecision::BF16
        )
    }
}

/// Describes the computational capabilities of a target hardware device.
///
/// # Examples
/// ```
/// use scirs2_neural::hardware::{HardwareProfile, QuantizationPrecision};
///
/// let cpu_profile = HardwareProfile::cpu_default();
/// assert!(cpu_profile.supported_precisions.contains(&QuantizationPrecision::FP32));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Human-readable device name (e.g. "Apple M2 Pro", "NVIDIA A100")
    pub name: String,
    /// Number of physical compute cores
    pub num_cores: usize,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gb_s: f64,
    /// Total on-chip memory / cache in MB
    pub cache_mb: f64,
    /// Supported quantization precisions
    pub supported_precisions: Vec<QuantizationPrecision>,
    /// SIMD vector width in bits (0 = no SIMD)
    pub simd_width_bits: usize,
    /// Whether the device has dedicated neural-network accelerator units
    pub has_npu: bool,
    /// Peak compute in TFLOPS at FP32
    pub peak_tflops_fp32: f64,
    /// Custom hardware properties
    pub properties: HashMap<String, String>,
}

impl HardwareProfile {
    /// Build a generic CPU profile.
    pub fn cpu_default() -> Self {
        Self {
            name: "Generic CPU".to_string(),
            num_cores: detect_num_cpus(),
            memory_bandwidth_gb_s: 50.0,
            cache_mb: 8.0,
            supported_precisions: vec![
                QuantizationPrecision::FP32,
                QuantizationPrecision::FP16,
                QuantizationPrecision::INT8,
            ],
            simd_width_bits: 256, // AVX2
            has_npu: false,
            peak_tflops_fp32: 0.5,
            properties: HashMap::new(),
        }
    }

    /// Build a mobile ARM profile (e.g. Cortex-A series).
    pub fn mobile_arm() -> Self {
        Self {
            name: "Mobile ARM".to_string(),
            num_cores: 8,
            memory_bandwidth_gb_s: 30.0,
            cache_mb: 4.0,
            supported_precisions: vec![
                QuantizationPrecision::FP32,
                QuantizationPrecision::FP16,
                QuantizationPrecision::INT8,
                QuantizationPrecision::INT4,
            ],
            simd_width_bits: 128, // NEON
            has_npu: true,
            peak_tflops_fp32: 0.1,
            properties: {
                let mut m = HashMap::new();
                m.insert("arch".to_string(), "arm64".to_string());
                m
            },
        }
    }

    /// Build an NVIDIA GPU profile.
    pub fn nvidia_gpu(name: &str, tflops: f64, bandwidth_gb_s: f64) -> Self {
        Self {
            name: name.to_string(),
            num_cores: 4096,
            memory_bandwidth_gb_s: bandwidth_gb_s,
            cache_mb: 40.0,
            supported_precisions: vec![
                QuantizationPrecision::FP32,
                QuantizationPrecision::FP16,
                QuantizationPrecision::BF16,
                QuantizationPrecision::INT8,
                QuantizationPrecision::INT4,
            ],
            simd_width_bits: 512,
            has_npu: true,
            peak_tflops_fp32: tflops,
            properties: {
                let mut m = HashMap::new();
                m.insert("vendor".to_string(), "NVIDIA".to_string());
                m
            },
        }
    }

    /// Returns the most efficient precision supported for inference.
    ///
    /// Chooses the smallest integer type available, falling back to FP16/FP32.
    pub fn preferred_inference_precision(&self) -> QuantizationPrecision {
        let priority = [
            QuantizationPrecision::INT4,
            QuantizationPrecision::INT8,
            QuantizationPrecision::FP16,
            QuantizationPrecision::BF16,
            QuantizationPrecision::FP32,
        ];
        for p in &priority {
            if self.supported_precisions.contains(p) {
                return *p;
            }
        }
        QuantizationPrecision::FP32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer quantization plan
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer quantization decision produced by [`HardwareOptimizer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantizationPlan {
    /// Layer name / identifier
    pub layer_name: String,
    /// Chosen weight precision
    pub weight_precision: QuantizationPrecision,
    /// Chosen activation precision
    pub activation_precision: QuantizationPrecision,
    /// Whether this layer should be pruned
    pub prune: bool,
    /// Pruning sparsity target (0 = no pruning, 1 = fully pruned)
    pub pruning_sparsity: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardware optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Strategies used by [`HardwareOptimizer`] to decide which precision to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OptimizationStrategy {
    /// Maximise throughput at the cost of accuracy
    MaxThroughput,
    /// Balance speed and numerical accuracy
    #[default]
    Balanced,
    /// Maximise accuracy, apply minimal quantization
    MaxAccuracy,
    /// Minimise power / energy consumption
    PowerEfficient,
}

/// Configuration for [`HardwareOptimizer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizerConfig {
    /// Optimisation goal
    pub strategy: OptimizationStrategy,
    /// Fraction of layers eligible for INT8 quantization (0.0–1.0)
    pub int8_fraction: f64,
    /// Whether to enable weight pruning
    pub enable_pruning: bool,
    /// Target pruning sparsity for eligible layers
    pub pruning_sparsity: f64,
    /// Layers that must stay at full FP32 precision (by name/prefix)
    pub sensitive_layers: Vec<String>,
}

impl Default for HardwareOptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Balanced,
            int8_fraction: 0.8,
            enable_pruning: false,
            pruning_sparsity: 0.3,
            sensitive_layers: vec!["output".to_string(), "classifier".to_string()],
        }
    }
}

/// Applies hardware-aware quantization and pruning decisions to a model description.
///
/// The optimizer does **not** mutate any live parameters; instead it produces a
/// [`Vec<LayerQuantizationPlan>`] that downstream code can apply.
///
/// # Examples
/// ```
/// use scirs2_neural::hardware::{HardwareProfile, HardwareOptimizer, HardwareOptimizerConfig};
///
/// let profile = HardwareProfile::mobile_arm();
/// let optimizer = HardwareOptimizer::new(profile, HardwareOptimizerConfig::default());
///
/// let layers = &["conv1", "bn1", "relu1", "conv2", "fc_output"];
/// let plan = optimizer.compute_quantization_plan(layers).expect("plan failed");
/// assert_eq!(plan.len(), layers.len());
/// ```
pub struct HardwareOptimizer {
    profile: HardwareProfile,
    config: HardwareOptimizerConfig,
}

impl HardwareOptimizer {
    /// Create a new hardware optimizer for the given device profile and config.
    pub fn new(profile: HardwareProfile, config: HardwareOptimizerConfig) -> Self {
        Self { profile, config }
    }

    /// Returns a reference to the hardware profile.
    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }

    /// Returns a reference to the optimizer configuration.
    pub fn config(&self) -> &HardwareOptimizerConfig {
        &self.config
    }

    /// Produce a per-layer quantization plan for the given layer names.
    ///
    /// Layers whose names start with any prefix in `config.sensitive_layers` are
    /// kept at FP32.  The remaining layers are quantized according to
    /// `config.strategy` and the hardware's supported precisions.
    pub fn compute_quantization_plan(
        &self,
        layer_names: &[&str],
    ) -> Result<Vec<LayerQuantizationPlan>> {
        if layer_names.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "layer_names must not be empty".to_string(),
            ));
        }

        let total = layer_names.len();
        let n_int8 = ((total as f64) * self.config.int8_fraction.clamp(0.0, 1.0)) as usize;
        let preferred = self.profile.preferred_inference_precision();

        let mut plans = Vec::with_capacity(total);
        let mut int8_assigned = 0usize;

        for (i, &name) in layer_names.iter().enumerate() {
            let is_sensitive = self.config.sensitive_layers.iter().any(|prefix| {
                let p = prefix.as_str();
                name.starts_with(p) || name.ends_with(p)
            });

            let precision = if is_sensitive {
                QuantizationPrecision::FP32
            } else if int8_assigned < n_int8
                && self
                    .profile
                    .supported_precisions
                    .contains(&QuantizationPrecision::INT8)
            {
                int8_assigned += 1;
                QuantizationPrecision::INT8
            } else {
                preferred
            };

            // Pruning: only apply to inner layers (not first/last) if enabled
            let prune =
                self.config.enable_pruning && !is_sensitive && i > 0 && i < total.saturating_sub(1);

            plans.push(LayerQuantizationPlan {
                layer_name: name.to_string(),
                weight_precision: precision,
                activation_precision: precision,
                prune,
                pruning_sparsity: if prune {
                    self.config.pruning_sparsity
                } else {
                    0.0
                },
            });
        }

        Ok(plans)
    }

    /// Estimate compressed model size in bytes given a base size and the plan.
    ///
    /// Uses the bit-width ratio of each layer's chosen precision vs FP32.
    pub fn estimate_compressed_size_bytes(
        &self,
        base_size_bytes: u64,
        plan: &[LayerQuantizationPlan],
    ) -> u64 {
        if plan.is_empty() {
            return base_size_bytes;
        }
        let weight_ratio: f64 = plan
            .iter()
            .map(|p| p.weight_precision.bits() as f64 / 32.0)
            .sum::<f64>()
            / plan.len() as f64;

        let prune_ratio: f64 =
            plan.iter().map(|p| 1.0 - p.pruning_sparsity).sum::<f64>() / plan.len() as f64;

        ((base_size_bytes as f64) * weight_ratio * prune_ratio) as u64
    }

    /// Quantize a flat f32 weight vector to i8 using symmetric INT8.
    ///
    /// Returns `(quantized_weights, scale)` where `scale` maps i8 → f32.
    pub fn quantize_to_int8(weights: &[f32]) -> Result<(Vec<i8>, f32)> {
        if weights.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "weights slice is empty".to_string(),
            ));
        }
        let abs_max = weights.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round();
                q.clamp(-128.0, 127.0) as i8
            })
            .collect();
        Ok((quantized, scale))
    }

    /// Dequantize i8 values back to f32 using the given scale.
    pub fn dequantize_from_int8(quantized: &[i8], scale: f32) -> Vec<f32> {
        quantized.iter().map(|&q| (q as f32) * scale).collect()
    }

    /// Quantize a flat f32 vector with asymmetric uint8 (FP16 simulation).
    ///
    /// Returns `(quantized, scale, zero_point)`.
    pub fn quantize_to_fp16_sim(weights: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        if weights.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "weights slice is empty".to_string(),
            ));
        }
        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (max - min).abs() > f32::EPSILON {
            (max - min) / 255.0
        } else {
            1.0
        };
        let zero_point = min;
        let quantized: Vec<u8> = weights
            .iter()
            .map(|&w| {
                let q = ((w - zero_point) / scale).round();
                q.clamp(0.0, 255.0) as u8
            })
            .collect();
        Ok((quantized, scale, zero_point))
    }
}

impl Debug for HardwareOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HardwareOptimizer")
            .field("profile", &self.profile.name)
            .field("strategy", &self.config.strategy)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the number of logical CPUs on the host (returns 1 if unavailable).
fn detect_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_cpu_default() {
        let p = HardwareProfile::cpu_default();
        assert!(!p.name.is_empty());
        assert!(p.num_cores >= 1);
        assert!(p
            .supported_precisions
            .contains(&QuantizationPrecision::FP32));
    }

    #[test]
    fn test_hardware_profile_mobile_arm() {
        let p = HardwareProfile::mobile_arm();
        assert!(p.has_npu);
        assert!(p
            .supported_precisions
            .contains(&QuantizationPrecision::INT8));
        assert_eq!(p.simd_width_bits, 128);
    }

    #[test]
    fn test_preferred_precision_mobile() {
        let p = HardwareProfile::mobile_arm();
        let pref = p.preferred_inference_precision();
        assert_eq!(pref, QuantizationPrecision::INT4);
    }

    #[test]
    fn test_preferred_precision_cpu() {
        let p = HardwareProfile::cpu_default();
        let pref = p.preferred_inference_precision();
        assert_eq!(pref, QuantizationPrecision::INT8);
    }

    #[test]
    fn test_compute_quantization_plan_basic() {
        let profile = HardwareProfile::mobile_arm();
        let config = HardwareOptimizerConfig {
            int8_fraction: 0.6,
            enable_pruning: false,
            ..Default::default()
        };
        let opt = HardwareOptimizer::new(profile, config);
        let layers = &["conv1", "bn1", "conv2", "bn2", "fc_output"];
        let plan = opt.compute_quantization_plan(layers).expect("plan ok");
        assert_eq!(plan.len(), 5);
        // fc_output is sensitive → FP32
        let fc = plan
            .iter()
            .find(|p| p.layer_name == "fc_output")
            .expect("fc");
        assert_eq!(fc.weight_precision, QuantizationPrecision::FP32);
    }

    #[test]
    fn test_compute_quantization_plan_empty_layers_err() {
        let opt = HardwareOptimizer::new(
            HardwareProfile::cpu_default(),
            HardwareOptimizerConfig::default(),
        );
        assert!(opt.compute_quantization_plan(&[]).is_err());
    }

    #[test]
    fn test_quantize_int8_roundtrip() {
        let weights: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0, 0.0, 0.25, -0.25];
        let (quant, scale) = HardwareOptimizer::quantize_to_int8(&weights).expect("quant ok");
        let dequant = HardwareOptimizer::dequantize_from_int8(&quant, scale);
        for (orig, deq) in weights.iter().zip(dequant.iter()) {
            assert!((orig - deq).abs() < 0.01, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn test_quantize_fp16_sim_roundtrip() {
        let weights: Vec<f32> = vec![0.1, 0.5, -0.3, 0.9, -0.9];
        let (quant, scale, zp) = HardwareOptimizer::quantize_to_fp16_sim(&weights).expect("ok");
        let dequant: Vec<f32> = quant.iter().map(|&q| (q as f32) * scale + zp).collect();
        for (orig, deq) in weights.iter().zip(dequant.iter()) {
            assert!((orig - deq).abs() < 0.02, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn test_quantize_int8_empty_err() {
        assert!(HardwareOptimizer::quantize_to_int8(&[]).is_err());
    }

    #[test]
    fn test_estimate_compressed_size() {
        let profile = HardwareProfile::cpu_default();
        let opt = HardwareOptimizer::new(profile, HardwareOptimizerConfig::default());
        let layers = &["layer1", "layer2"];
        let plan = opt.compute_quantization_plan(layers).expect("plan");
        let compressed = opt.estimate_compressed_size_bytes(1_000_000, &plan);
        // INT8 is 8 bits vs 32 bits FP32, so expect < original
        assert!(compressed < 1_000_000);
    }

    #[test]
    fn test_precision_bits() {
        assert_eq!(QuantizationPrecision::FP32.bits(), 32);
        assert_eq!(QuantizationPrecision::FP16.bits(), 16);
        assert_eq!(QuantizationPrecision::INT8.bits(), 8);
        assert_eq!(QuantizationPrecision::INT4.bits(), 4);
    }

    #[test]
    fn test_hardware_optimizer_debug() {
        let opt = HardwareOptimizer::new(
            HardwareProfile::cpu_default(),
            HardwareOptimizerConfig::default(),
        );
        let s = format!("{opt:?}");
        assert!(s.contains("HardwareOptimizer"));
    }
}
