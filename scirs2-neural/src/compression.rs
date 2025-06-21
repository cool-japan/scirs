//! Model compression utilities for neural networks
//!
//! This module provides tools for model compression including:
//! - Quantization (post-training and quantization-aware training)
//! - Pruning (magnitude-based, structured, and unstructured)
//! - Knowledge distillation
//! - Model compression analysis and optimization

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Quantization precision levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationBits {
    /// 8-bit quantization
    Int8,
    /// 16-bit quantization  
    Int16,
    /// 4-bit quantization (experimental)
    Int4,
    /// Mixed precision
    Mixed {
        /// Number of bits for weight quantization
        weight_bits: u8,
        /// Number of bits for activation quantization
        activation_bits: u8,
    },
}

/// Quantization scheme
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero-point is 0)
    Symmetric,
    /// Asymmetric quantization (with zero-point)
    Asymmetric,
    /// Dynamic quantization (calibrated during inference)
    Dynamic,
}

/// Quantization calibration method
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Use min/max values
    MinMax,
    /// Use percentiles to handle outliers
    Percentile {
        /// Lower percentile bound (0.0 to 100.0)
        lower: f64,
        /// Upper percentile bound (0.0 to 100.0)
        upper: f64,
    },
    /// KL-divergence based calibration
    KLDivergence,
    /// Entropy-based calibration
    Entropy,
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantizationParams<F: Float> {
    /// Scale factor
    pub scale: F,
    /// Zero point
    pub zero_point: i32,
    /// Quantization range
    pub qmin: i32,
    /// Quantization range
    pub qmax: i32,
}

/// Post-training quantization manager
pub struct PostTrainingQuantizer<F: Float + Debug> {
    /// Quantization bit precision
    bits: QuantizationBits,
    /// Quantization scheme
    scheme: QuantizationScheme,
    /// Calibration method
    calibration: CalibrationMethod,
    /// Layer-specific quantization parameters
    layer_params: HashMap<String, QuantizationParams<F>>,
    /// Calibration data statistics
    calibration_stats: HashMap<String, CalibrationStatistics<F>>,
}

/// Statistics collected during calibration
#[derive(Debug, Clone)]
pub struct CalibrationStatistics<F: Float> {
    /// Minimum observed value
    pub min_val: F,
    /// Maximum observed value
    pub max_val: F,
    /// Histogram for distribution analysis
    pub histogram: Vec<u32>,
    /// Histogram bin edges
    pub bin_edges: Vec<F>,
    /// Number of samples observed
    pub sample_count: usize,
}

impl<F: Float + Debug + 'static> PostTrainingQuantizer<F> {
    /// Create a new post-training quantizer
    pub fn new(
        bits: QuantizationBits,
        scheme: QuantizationScheme,
        calibration: CalibrationMethod,
    ) -> Self {
        Self {
            bits,
            scheme,
            calibration,
            layer_params: HashMap::new(),
            calibration_stats: HashMap::new(),
        }
    }

    /// Calibrate quantization parameters using sample data
    pub fn calibrate(&mut self, layer_name: String, activations: &ArrayD<F>) -> Result<()> {
        let min_val = activations.iter().cloned().fold(F::infinity(), F::min);
        let max_val = activations.iter().cloned().fold(F::neg_infinity(), F::max);

        // Create histogram for distribution analysis
        let num_bins = 2048;
        let range = max_val - min_val;
        let bin_width = range / F::from(num_bins).unwrap();

        let mut histogram = vec![0u32; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);

        for i in 0..=num_bins {
            bin_edges.push(min_val + bin_width * F::from(i).unwrap());
        }

        // Fill histogram
        for &val in activations.iter() {
            if val.is_finite() {
                let bin_idx = ((val - min_val) / bin_width).to_usize().unwrap_or(0);
                let bin_idx = bin_idx.min(num_bins - 1);
                histogram[bin_idx] += 1;
            }
        }

        let stats = CalibrationStatistics {
            min_val,
            max_val,
            histogram,
            bin_edges,
            sample_count: activations.len(),
        };

        self.calibration_stats.insert(layer_name.clone(), stats);

        // Compute quantization parameters
        let params = self.compute_quantization_params(&layer_name)?;
        self.layer_params.insert(layer_name, params);

        Ok(())
    }

    /// Compute quantization parameters for a layer
    fn compute_quantization_params(&self, layer_name: &str) -> Result<QuantizationParams<F>> {
        let stats = self.calibration_stats.get(layer_name).ok_or_else(|| {
            NeuralError::ComputationError("Calibration stats not found".to_string())
        })?;

        let (qmin, qmax) = match self.bits {
            QuantizationBits::Int8 => (-128i32, 127i32),
            QuantizationBits::Int16 => (-32768i32, 32767i32),
            QuantizationBits::Int4 => (-8i32, 7i32),
            QuantizationBits::Mixed { weight_bits, .. } => match weight_bits {
                8 => (-128i32, 127i32),
                16 => (-32768i32, 32767i32),
                4 => (-8i32, 7i32),
                _ => {
                    return Err(NeuralError::InvalidArchitecture(
                        "Unsupported bit width".to_string(),
                    ))
                }
            },
        };

        let (min_val, max_val) = match self.calibration {
            CalibrationMethod::MinMax => (stats.min_val, stats.max_val),
            CalibrationMethod::Percentile { lower, upper } => {
                self.compute_percentiles(stats, lower, upper)?
            }
            CalibrationMethod::KLDivergence => self.compute_kl_optimal_range(stats)?,
            CalibrationMethod::Entropy => self.compute_entropy_optimal_range(stats)?,
        };

        let (scale, zero_point) = match self.scheme {
            QuantizationScheme::Symmetric => {
                let max_abs = F::max(min_val.abs(), max_val.abs());
                let scale = max_abs / F::from(qmax).unwrap();
                (scale, 0i32)
            }
            QuantizationScheme::Asymmetric => {
                let scale = (max_val - min_val) / F::from(qmax - qmin).unwrap();
                let zero_point = qmin - (min_val / scale).round().to_i32().unwrap_or(0);
                (scale, zero_point)
            }
            QuantizationScheme::Dynamic => {
                // For dynamic quantization, we compute per-batch statistics
                let scale = (max_val - min_val) / F::from(qmax - qmin).unwrap();
                (scale, 0i32)
            }
        };

        Ok(QuantizationParams {
            scale,
            zero_point,
            qmin,
            qmax,
        })
    }

    fn compute_percentiles(
        &self,
        stats: &CalibrationStatistics<F>,
        lower: f64,
        upper: f64,
    ) -> Result<(F, F)> {
        let total_samples = stats.sample_count;
        let lower_threshold = (total_samples as f64 * lower / 100.0) as usize;
        let upper_threshold = (total_samples as f64 * upper / 100.0) as usize;

        let mut cumulative = 0usize;
        let mut min_val = stats.min_val;
        let mut max_val = stats.max_val;

        for (i, &count) in stats.histogram.iter().enumerate() {
            cumulative += count as usize;

            if cumulative >= lower_threshold && min_val == stats.min_val {
                min_val = stats.bin_edges[i];
            }

            if cumulative >= upper_threshold {
                max_val = stats.bin_edges[i + 1];
                break;
            }
        }

        Ok((min_val, max_val))
    }

    fn compute_kl_optimal_range(&self, stats: &CalibrationStatistics<F>) -> Result<(F, F)> {
        // Simplified KL divergence optimization
        // In practice, this would involve more sophisticated optimization
        let target_bins = 128; // Target quantization bins
        let total_bins = stats.histogram.len();

        if target_bins >= total_bins {
            return Ok((stats.min_val, stats.max_val));
        }

        let mut best_range = (stats.min_val, stats.max_val);
        let mut best_kl = F::infinity();

        // Try different ranges and find the one with minimum KL divergence
        for start in 0..(total_bins - target_bins) {
            let end = start + target_bins;
            let range_min = stats.bin_edges[start];
            let range_max = stats.bin_edges[end];

            let kl_div = self.compute_kl_divergence(stats, start, end);
            if kl_div < best_kl {
                best_kl = kl_div;
                best_range = (range_min, range_max);
            }
        }

        Ok(best_range)
    }

    fn compute_kl_divergence(
        &self,
        stats: &CalibrationStatistics<F>,
        start: usize,
        end: usize,
    ) -> F {
        // Simplified KL divergence calculation
        let mut kl_div = F::zero();
        let total_count: u32 = stats.histogram[start..end].iter().sum();

        if total_count == 0 {
            return F::infinity();
        }

        for i in start..end {
            let p = F::from(stats.histogram[i]).unwrap() / F::from(total_count).unwrap();
            if p > F::zero() {
                // Assume uniform quantized distribution for simplicity
                let q = F::one() / F::from(end - start).unwrap();
                kl_div = kl_div + p * (p / q).ln();
            }
        }

        kl_div
    }

    fn compute_entropy_optimal_range(&self, stats: &CalibrationStatistics<F>) -> Result<(F, F)> {
        // Find range that preserves most entropy
        let total_count: u32 = stats.histogram.iter().sum();
        if total_count == 0 {
            return Ok((stats.min_val, stats.max_val));
        }

        // Find the range that contains 99.9% of the data
        let threshold = (total_count as f64 * 0.999) as u32;
        let mut cumulative = 0u32;

        let mut start_idx = 0;
        let mut end_idx = stats.histogram.len() - 1;

        for (i, &count) in stats.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= threshold / 2 && start_idx == 0 {
                start_idx = i;
            }
            if cumulative >= threshold {
                end_idx = i;
                break;
            }
        }

        Ok((stats.bin_edges[start_idx], stats.bin_edges[end_idx + 1]))
    }

    /// Quantize a tensor using the calibrated parameters
    pub fn quantize_tensor(&self, layer_name: &str, tensor: &ArrayD<F>) -> Result<ArrayD<i32>> {
        let params = self.layer_params.get(layer_name).ok_or_else(|| {
            NeuralError::ComputationError("Quantization params not found".to_string())
        })?;

        let quantized = tensor.mapv(|x| {
            let scaled = x / params.scale;
            let shifted = scaled + F::from(params.zero_point).unwrap();
            let clamped = shifted
                .max(F::from(params.qmin).unwrap())
                .min(F::from(params.qmax).unwrap());
            clamped.round().to_i32().unwrap_or(0)
        });

        Ok(quantized)
    }

    /// Dequantize a tensor back to floating point
    pub fn dequantize_tensor(
        &self,
        layer_name: &str,
        quantized: &ArrayD<i32>,
    ) -> Result<ArrayD<F>> {
        let params = self.layer_params.get(layer_name).ok_or_else(|| {
            NeuralError::ComputationError("Quantization params not found".to_string())
        })?;

        let dequantized = quantized.mapv(|q| {
            let shifted = F::from(q).unwrap() - F::from(params.zero_point).unwrap();
            shifted * params.scale
        });

        Ok(dequantized)
    }

    /// Get compression ratio achieved
    pub fn get_compression_ratio(&self) -> f64 {
        match self.bits {
            QuantizationBits::Int8 => 32.0 / 8.0,
            QuantizationBits::Int16 => 32.0 / 16.0,
            QuantizationBits::Int4 => 32.0 / 4.0,
            QuantizationBits::Mixed {
                weight_bits,
                activation_bits,
            } => {
                // Weighted average assuming equal weight and activation sizes
                32.0 / ((weight_bits + activation_bits) as f64 / 2.0)
            }
        }
    }
}

/// Pruning method
#[derive(Debug, Clone, PartialEq)]
pub enum PruningMethod {
    /// Magnitude-based pruning
    MagnitudeBased {
        /// Threshold below which weights are pruned
        threshold: f64,
    },
    /// Top-k pruning (keep only k largest weights)
    TopK {
        /// Number of largest weights to keep
        k: usize,
    },
    /// Structured pruning (remove entire channels/filters)
    Structured {
        /// Granularity level for structured pruning
        granularity: StructuredGranularity,
    },
    /// Gradual magnitude pruning
    GradualMagnitude {
        /// Initial sparsity ratio (0.0 to 1.0)
        initial_sparsity: f64,
        /// Final sparsity ratio (0.0 to 1.0)
        final_sparsity: f64,
        /// Training step to begin pruning
        begin_step: usize,
        /// Training step to end pruning
        end_step: usize,
    },
}

/// Structured pruning granularity
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredGranularity {
    /// Channel-wise pruning
    Channel,
    /// Filter-wise pruning
    Filter,
    /// Block-wise pruning
    Block {
        /// Size of blocks to prune (height, width)
        block_size: (usize, usize),
    },
}

/// Neural network pruner
pub struct ModelPruner<F: Float + Debug> {
    /// Pruning method
    method: PruningMethod,
    /// Layer-specific pruning masks
    pruning_masks: HashMap<String, ArrayD<bool>>,
    /// Sparsity statistics per layer
    sparsity_stats: HashMap<String, SparsityStatistics>,
    /// Current training step (for gradual pruning)
    current_step: usize,
    /// Phantom data
    _phantom: std::marker::PhantomData<F>,
}

/// Sparsity statistics for a layer
#[derive(Debug, Clone)]
pub struct SparsityStatistics {
    /// Current sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    pub sparsity_ratio: f64,
    /// Number of pruned parameters
    pub pruned_params: usize,
    /// Total number of parameters
    pub total_params: usize,
    /// Memory reduction achieved
    pub memory_reduction: f64,
}

impl<F: Float + Debug + 'static> ModelPruner<F> {
    /// Create a new model pruner
    pub fn new(method: PruningMethod) -> Self {
        Self {
            method,
            pruning_masks: HashMap::new(),
            sparsity_stats: HashMap::new(),
            current_step: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate pruning mask for a layer
    pub fn generate_pruning_mask(
        &mut self,
        layer_name: String,
        weights: &ArrayD<F>,
    ) -> Result<ArrayD<bool>> {
        let mask = match &self.method {
            PruningMethod::MagnitudeBased { threshold } => {
                self.magnitude_based_mask(weights, *threshold)?
            }
            PruningMethod::TopK { k } => self.top_k_mask(weights, *k)?,
            PruningMethod::Structured { granularity } => {
                self.structured_mask(weights, granularity)?
            }
            PruningMethod::GradualMagnitude {
                initial_sparsity,
                final_sparsity,
                begin_step,
                end_step,
            } => {
                let current_sparsity = self.compute_gradual_sparsity(
                    *initial_sparsity,
                    *final_sparsity,
                    *begin_step,
                    *end_step,
                );
                self.magnitude_based_mask(weights, current_sparsity)?
            }
        };

        // Compute sparsity statistics
        let total_params = mask.len();
        let pruned_params = mask.iter().filter(|&&x| !x).count();
        let sparsity_ratio = pruned_params as f64 / total_params as f64;

        let stats = SparsityStatistics {
            sparsity_ratio,
            pruned_params,
            total_params,
            memory_reduction: sparsity_ratio,
        };

        self.sparsity_stats.insert(layer_name.clone(), stats);
        self.pruning_masks.insert(layer_name, mask.clone());

        Ok(mask)
    }

    fn magnitude_based_mask(&self, weights: &ArrayD<F>, threshold: f64) -> Result<ArrayD<bool>> {
        let threshold_val = F::from(threshold)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid threshold".to_string()))?;

        let mask = weights.mapv(|w| w.abs() >= threshold_val);
        Ok(mask)
    }

    fn top_k_mask(&self, weights: &ArrayD<F>, k: usize) -> Result<ArrayD<bool>> {
        let total_params = weights.len();
        if k >= total_params {
            return Ok(Array::from_elem(weights.raw_dim(), true));
        }

        // Find the k-th largest magnitude
        let mut magnitudes: Vec<F> = weights.iter().map(|&w| w.abs()).collect();
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = magnitudes[k - 1];
        let mask = weights.mapv(|w| w.abs() >= threshold);

        Ok(mask)
    }

    fn structured_mask(
        &self,
        weights: &ArrayD<F>,
        granularity: &StructuredGranularity,
    ) -> Result<ArrayD<bool>> {
        match granularity {
            StructuredGranularity::Channel => self.channel_wise_mask(weights),
            StructuredGranularity::Filter => self.filter_wise_mask(weights),
            StructuredGranularity::Block { block_size } => {
                self.block_wise_mask(weights, *block_size)
            }
        }
    }

    fn channel_wise_mask(&self, weights: &ArrayD<F>) -> Result<ArrayD<bool>> {
        if weights.ndim() < 2 {
            return Err(NeuralError::InvalidArchitecture(
                "Channel-wise pruning requires at least 2D weights".to_string(),
            ));
        }

        let mut mask = Array::from_elem(weights.raw_dim(), true);

        // For each output channel, compute L2 norm and decide whether to prune
        for i in 0..weights.shape()[0] {
            let channel_slice = weights.slice(ndarray::s![i, ..]);
            let l2_norm = channel_slice.mapv(|x| x * x).sum().sqrt();

            // Simple heuristic: prune channels with norm below median
            let threshold = F::from(0.1).unwrap(); // Simplified threshold
            if l2_norm < threshold {
                mask.slice_mut(ndarray::s![i, ..]).fill(false);
            }
        }

        Ok(mask)
    }

    fn filter_wise_mask(&self, weights: &ArrayD<F>) -> Result<ArrayD<bool>> {
        // Similar to channel-wise but operates on filters
        self.channel_wise_mask(weights) // Simplified implementation
    }

    fn block_wise_mask(
        &self,
        weights: &ArrayD<F>,
        block_size: (usize, usize),
    ) -> Result<ArrayD<bool>> {
        if weights.ndim() != 2 {
            return Err(NeuralError::InvalidArchitecture(
                "Block-wise pruning requires 2D weights".to_string(),
            ));
        }

        let (rows, cols) = (weights.shape()[0], weights.shape()[1]);
        let (block_h, block_w) = block_size;

        let mut mask = Array::from_elem(weights.raw_dim(), false);

        // Process each block
        for i in (0..rows).step_by(block_h) {
            for j in (0..cols).step_by(block_w) {
                let end_i = (i + block_h).min(rows);
                let end_j = (j + block_w).min(cols);

                let block = weights.slice(ndarray::s![i..end_i, j..end_j]);
                let block_norm = block.mapv(|x| x * x).sum().sqrt();

                // Keep block if norm is above threshold
                let threshold = F::from(0.1).unwrap();
                if block_norm >= threshold {
                    mask.slice_mut(ndarray::s![i..end_i, j..end_j]).fill(true);
                }
            }
        }

        Ok(mask)
    }

    fn compute_gradual_sparsity(
        &self,
        initial_sparsity: f64,
        final_sparsity: f64,
        begin_step: usize,
        end_step: usize,
    ) -> f64 {
        if self.current_step < begin_step {
            return initial_sparsity;
        }

        if self.current_step >= end_step {
            return final_sparsity;
        }

        let progress = (self.current_step - begin_step) as f64 / (end_step - begin_step) as f64;
        let sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress;

        sparsity.min(final_sparsity).max(initial_sparsity)
    }

    /// Apply pruning mask to weights
    pub fn apply_pruning_mask(&self, layer_name: &str, weights: &mut ArrayD<F>) -> Result<()> {
        let mask = self
            .pruning_masks
            .get(layer_name)
            .ok_or_else(|| NeuralError::ComputationError("Pruning mask not found".to_string()))?;

        if weights.shape() != mask.shape() {
            return Err(NeuralError::DimensionMismatch(
                "Weight and mask shapes don't match".to_string(),
            ));
        }

        // Zero out pruned weights
        for (weight, &keep) in weights.iter_mut().zip(mask.iter()) {
            if !keep {
                *weight = F::zero();
            }
        }

        Ok(())
    }

    /// Update current step (for gradual pruning)
    pub fn update_step(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Get overall model sparsity
    pub fn get_model_sparsity(&self) -> f64 {
        if self.sparsity_stats.is_empty() {
            return 0.0;
        }

        let total_params: usize = self.sparsity_stats.values().map(|s| s.total_params).sum();
        let pruned_params: usize = self.sparsity_stats.values().map(|s| s.pruned_params).sum();

        pruned_params as f64 / total_params as f64
    }

    /// Get sparsity statistics for all layers
    pub fn get_sparsity_statistics(&self) -> &HashMap<String, SparsityStatistics> {
        &self.sparsity_stats
    }
}

/// Model compression analyzer
pub struct CompressionAnalyzer {
    /// Original model size in bytes
    original_size: usize,
    /// Compressed model size in bytes  
    compressed_size: usize,
    /// Inference speed metrics
    speed_metrics: SpeedMetrics,
    /// Accuracy metrics
    accuracy_metrics: AccuracyMetrics,
}

/// Speed measurement metrics
#[derive(Debug, Clone)]
pub struct SpeedMetrics {
    /// Original inference time (ms)
    pub original_time_ms: f64,
    /// Compressed inference time (ms)
    pub compressed_time_ms: f64,
    /// Speedup ratio
    pub speedup_ratio: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
}

/// Accuracy measurement metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Original model accuracy
    pub original_accuracy: f64,
    /// Compressed model accuracy
    pub compressed_accuracy: f64,
    /// Accuracy degradation
    pub accuracy_loss: f64,
}

impl CompressionAnalyzer {
    /// Create a new compression analyzer
    pub fn new(original_size: usize) -> Self {
        Self {
            original_size,
            compressed_size: original_size,
            speed_metrics: SpeedMetrics {
                original_time_ms: 0.0,
                compressed_time_ms: 0.0,
                speedup_ratio: 1.0,
                memory_usage_mb: 0.0,
            },
            accuracy_metrics: AccuracyMetrics {
                original_accuracy: 0.0,
                compressed_accuracy: 0.0,
                accuracy_loss: 0.0,
            },
        }
    }

    /// Update compressed model size
    pub fn set_compressed_size(&mut self, size: usize) {
        self.compressed_size = size;
    }

    /// Update speed metrics
    pub fn update_speed_metrics(&mut self, metrics: SpeedMetrics) {
        self.speed_metrics = metrics;
    }

    /// Update accuracy metrics
    pub fn update_accuracy_metrics(&mut self, metrics: AccuracyMetrics) {
        self.accuracy_metrics = metrics;
    }

    /// Get compression ratio
    pub fn get_compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.compressed_size as f64
    }

    /// Get comprehensive analysis report
    pub fn generate_report(&self) -> CompressionReport {
        CompressionReport {
            compression_ratio: self.get_compression_ratio(),
            size_reduction_mb: (self.original_size - self.compressed_size) as f64
                / (1024.0 * 1024.0),
            speed_metrics: self.speed_metrics.clone(),
            accuracy_metrics: self.accuracy_metrics.clone(),
            efficiency_score: self.calculate_efficiency_score(),
        }
    }

    fn calculate_efficiency_score(&self) -> f64 {
        // Weighted score considering compression, speed, and accuracy
        let compression_score = (self.get_compression_ratio() - 1.0).min(10.0) / 10.0;
        let speed_score = (self.speed_metrics.speedup_ratio - 1.0).min(10.0) / 10.0;
        let accuracy_score = 1.0 - (self.accuracy_metrics.accuracy_loss / 100.0).clamp(0.0, 1.0);

        // Weighted average (accuracy is most important)
        0.5 * accuracy_score + 0.3 * compression_score + 0.2 * speed_score
    }
}

/// Comprehensive compression analysis report
#[derive(Debug, Clone)]
pub struct CompressionReport {
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Size reduction in MB
    pub size_reduction_mb: f64,
    /// Speed metrics
    pub speed_metrics: SpeedMetrics,
    /// Accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
    /// Overall efficiency score (0-1)
    pub efficiency_score: f64,
}

impl std::fmt::Display for CompressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model Compression Analysis Report")?;
        writeln!(f, "=================================")?;
        writeln!(f, "Compression Ratio: {:.2}x", self.compression_ratio)?;
        writeln!(f, "Size Reduction: {:.1} MB", self.size_reduction_mb)?;
        writeln!(
            f,
            "Speed Improvement: {:.2}x",
            self.speed_metrics.speedup_ratio
        )?;
        writeln!(
            f,
            "Accuracy Loss: {:.2}%",
            self.accuracy_metrics.accuracy_loss
        )?;
        writeln!(
            f,
            "Memory Usage: {:.1} MB",
            self.speed_metrics.memory_usage_mb
        )?;
        writeln!(f, "Efficiency Score: {:.3}/1.000", self.efficiency_score)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_post_training_quantizer() {
        let mut quantizer = PostTrainingQuantizer::<f64>::new(
            QuantizationBits::Int8,
            QuantizationScheme::Symmetric,
            CalibrationMethod::MinMax,
        );

        let activations =
            Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64 / 10.0).collect())
                .unwrap()
                .into_dyn();

        quantizer
            .calibrate("layer1".to_string(), &activations)
            .unwrap();

        let quantized = quantizer.quantize_tensor("layer1", &activations).unwrap();
        let dequantized = quantizer.dequantize_tensor("layer1", &quantized).unwrap();

        assert_eq!(quantized.shape(), activations.shape());
        assert_eq!(dequantized.shape(), activations.shape());

        let compression_ratio = quantizer.get_compression_ratio();
        assert_eq!(compression_ratio, 4.0); // 32-bit to 8-bit
    }

    #[test]
    fn test_magnitude_based_pruning() {
        let mut pruner = ModelPruner::<f64>::new(PruningMethod::MagnitudeBased { threshold: 0.5 });

        let weights =
            Array2::from_shape_vec((3, 3), vec![0.1, 0.6, 0.3, 0.8, 0.2, 0.7, 0.4, 0.9, 0.1])
                .unwrap()
                .into_dyn();

        let mask = pruner
            .generate_pruning_mask("layer1".to_string(), &weights)
            .unwrap();

        // Weights with magnitude >= 0.5 should be kept
        assert!(mask[[0, 1]]); // 0.6 >= 0.5
        assert!(mask[[1, 0]]); // 0.8 >= 0.5
        assert!(mask[[1, 2]]); // 0.7 >= 0.5
        assert!(mask[[2, 1]]); // 0.9 >= 0.5

        assert!(!mask[[0, 0]]); // 0.1 < 0.5
        assert!(!mask[[0, 2]]); // 0.3 < 0.5

        let stats = pruner.get_sparsity_statistics();
        assert!(stats.contains_key("layer1"));
    }

    #[test]
    fn test_top_k_pruning() {
        let mut pruner = ModelPruner::<f64>::new(PruningMethod::TopK { k: 3 });

        let weights = Array2::from_shape_vec((2, 3), vec![0.1, 0.6, 0.3, 0.8, 0.2, 0.7])
            .unwrap()
            .into_dyn();

        let mask = pruner
            .generate_pruning_mask("layer1".to_string(), &weights)
            .unwrap();

        // Should keep 3 largest weights: 0.8, 0.7, 0.6
        let kept_count = mask.iter().filter(|&&x| x).count();
        assert_eq!(kept_count, 3);

        assert!(mask[[1, 0]]); // 0.8
        assert!(mask[[1, 2]]); // 0.7
        assert!(mask[[0, 1]]); // 0.6
    }

    #[test]
    fn test_gradual_pruning() {
        let mut pruner = ModelPruner::<f64>::new(PruningMethod::GradualMagnitude {
            initial_sparsity: 0.0,
            final_sparsity: 0.8,
            begin_step: 10,
            end_step: 20,
        });

        // Test at different steps
        pruner.update_step(5);
        let sparsity_early = pruner.compute_gradual_sparsity(0.0, 0.8, 10, 20);
        assert_eq!(sparsity_early, 0.0); // Before begin_step

        pruner.update_step(15);
        let sparsity_mid = pruner.compute_gradual_sparsity(0.0, 0.8, 10, 20);
        assert_eq!(sparsity_mid, 0.4); // Middle of range

        pruner.update_step(25);
        let sparsity_late = pruner.compute_gradual_sparsity(0.0, 0.8, 10, 20);
        assert_eq!(sparsity_late, 0.8); // After end_step
    }

    #[test]
    fn test_compression_analyzer() {
        let mut analyzer = CompressionAnalyzer::new(1000000); // 1MB original
        analyzer.set_compressed_size(250000); // 250KB compressed

        analyzer.update_speed_metrics(SpeedMetrics {
            original_time_ms: 100.0,
            compressed_time_ms: 40.0,
            speedup_ratio: 2.5,
            memory_usage_mb: 25.0,
        });

        analyzer.update_accuracy_metrics(AccuracyMetrics {
            original_accuracy: 95.0,
            compressed_accuracy: 94.0,
            accuracy_loss: 1.0,
        });

        let report = analyzer.generate_report();
        assert_eq!(report.compression_ratio, 4.0);
        assert!((report.size_reduction_mb - 0.715).abs() < 0.01);
        assert!(report.efficiency_score > 0.6);
    }

    #[test]
    fn test_quantization_bits() {
        assert_eq!(QuantizationBits::Int8, QuantizationBits::Int8);

        let mixed = QuantizationBits::Mixed {
            weight_bits: 8,
            activation_bits: 16,
        };

        if let QuantizationBits::Mixed {
            weight_bits,
            activation_bits,
        } = mixed
        {
            assert_eq!(weight_bits, 8);
            assert_eq!(activation_bits, 16);
        }
    }
}
