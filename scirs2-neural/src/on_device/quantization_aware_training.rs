//! Quantization-aware training for reduced model size and faster inference

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use statrs::statistics::Statistics;
/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QATConfig {
    /// Number of bits for weights
    pub weight_bits: u8,
    /// Number of bits for activations
    pub activation_bits: u8,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Enable fake quantization during training
    pub fake_quantize: bool,
    /// Calibration method
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
/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero point = 0)
    Symmetric,
    /// Asymmetric quantization
    Asymmetric,
    /// Per-channel quantization
    PerChannel,
    /// Dynamic quantization
    Dynamic,
/// Calibration method for quantization
pub enum CalibrationMethod {
    /// Min-max calibration
    MinMax,
    /// Percentile calibration
    Percentile(f32),
    /// Entropy calibration
    Entropy,
    /// KL-divergence minimization
    KLDivergence,
/// Quantization-aware training manager
pub struct QuantizationAwareTraining {
    config: QATConfig,
    quantizers: Vec<Quantizer>,
    calibration_stats: CalibrationStats,
impl QuantizationAwareTraining {
    /// Create a new QAT manager
    pub fn new(config: QATConfig) -> Self {
            config,
            quantizers: Vec::new(),
            calibration_stats: CalibrationStats::new(),
    
    /// Quantize weights
    pub fn quantize_weights(&self, weights: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        let quantizer = match self._config.scheme {
            QuantizationScheme::Symmetric => SymmetricQuantizer::new(self._config.weight_bits),
            QuantizationScheme::Asymmetric => AsymmetricQuantizer::new(self._config.weight_bits),
            QuantizationScheme::PerChannel => PerChannelQuantizer::new(self._config.weight_bits),
            QuantizationScheme::Dynamic => DynamicQuantizer::new(self._config.weight_bits),
        };
        
        quantizer.quantize(weights)
    /// Quantize activations
    pub fn quantize_activations(&self, activations: &ArrayView2<f32>) -> Result<QuantizedTensor> {
            QuantizationScheme::Symmetric => SymmetricQuantizer::new(self.config.activation_bits),
            QuantizationScheme::Asymmetric => AsymmetricQuantizer::new(self.config.activation_bits),
            QuantizationScheme::PerChannel => PerChannelQuantizer::new(self.config.activation_bits),
            QuantizationScheme::Dynamic => DynamicQuantizer::new(self.config.activation_bits),
        quantizer.quantize(activations)
    /// Fake quantization for training
    pub fn fake_quantize(&self, tensor: &ArrayView2<f32>, isweight: bool) -> Result<Array2<f32>> {
        if !self.config.fake_quantize {
            return Ok(tensor.to_owned());
        let bits = if is_weight { self.config.weight_bits } else { self.config.activation_bits };
        let quantized = self.quantize_tensor(tensor, bits)?;
        self.dequantize(&quantized)
    /// Quantize tensor
    fn quantize_tensor(&self, tensor: &ArrayView2<f32>, bits: u8) -> Result<QuantizedTensor> {
        let (scale, zero_point) = self.compute_quantization_params(tensor, bits)?;
        let n_levels = (1 << bits) as f32;
        let quantized_data = tensor.mapv(|x| {
            let q = ((x / scale) + zero_point).round();
            q.clamp(0.0, n_levels - 1.0) as i32
        });
        Ok(QuantizedTensor {
            data: quantized_data,
            scale,
            zero_point,
            bits,
        })
    /// Dequantize tensor
    pub fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Array2<f32>> {
        Ok(quantized.data.mapv(|x| (x as f32 - quantized.zero_point) * quantized.scale))
    /// Compute quantization parameters
    fn compute_quantization_params(&self, tensor: &ArrayView2<f32>, bits: u8) -> Result<(f32, f32)> {
        let (min_val, max_val) = match self.config.calibration {
            CalibrationMethod::MinMax => {
                let min = tensor.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let max = tensor.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                (*min, *max)
            }
            CalibrationMethod::Percentile(p) => {
                let mut values: Vec<f32> = tensor.iter().cloned().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let low_idx = ((1.0 - p) * values.len() as f32) as usize;
                let high_idx = (p * values.len() as f32) as usize;
                (values[low_idx], values[high_idx.min(values.len() - 1)])
            _ => {
                // Simplified for other methods
        match self.config.scheme {
            QuantizationScheme::Symmetric => {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = (2.0 * abs_max) / (n_levels - 1.0);
                let zero_point = (n_levels - 1.0) / 2.0;
                Ok((scale, zero_point))
            QuantizationScheme::Asymmetric => {
                let scale = (max_val - min_val) / (n_levels - 1.0);
                let zero_point = -min_val / scale;
                // Simplified for other schemes
    /// Collect calibration statistics
    pub fn update_calibration_stats(&mut self, tensor: &ArrayView2<f32>, name: &str) {
        self.calibration_stats.update(name, tensor);
    /// Apply calibration
    pub fn apply_calibration(&mut self) -> Result<()> {
        // Apply calibration statistics to quantizers
        for (name, stats) in &self.calibration_stats.stats {
            println!("Calibration for {}: min={:.4}, max={:.4}, mean={:.4}",
                     name, stats.min, stats.max, stats.mean);
        Ok(())
/// Quantized tensor representation
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Array2<i32>,
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: f32,
    /// Number of bits
    pub bits: u8,
/// Base trait for quantizers
trait Quantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor>;
/// Symmetric quantizer
struct SymmetricQuantizer {
    bits: u8,
impl SymmetricQuantizer {
    fn new(bits: u8) -> Self {
        Self { _bits }
impl Quantizer for SymmetricQuantizer {
    fn quantize(&self, tensor: &ArrayView2<f32>) -> Result<QuantizedTensor> {
        let abs_max = tensor.iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);
        let n_levels = (1 << self.bits) as f32;
        let scale = (2.0 * abs_max) / (n_levels - 1.0);
        let zero_point = (n_levels - 1.0) / 2.0;
        let quantized = tensor.mapv(|x| {
            let q = (x / scale + zero_point).round();
            data: quantized,
            bits: self.bits,
/// Asymmetric quantizer
struct AsymmetricQuantizer {
impl AsymmetricQuantizer {
impl Quantizer for AsymmetricQuantizer {
        let min_val = tensor.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_val = tensor.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let scale = (max_val - min_val) / (n_levels - 1.0);
        let zero_point = -min_val / scale;
/// Per-channel quantizer
struct PerChannelQuantizer {
impl PerChannelQuantizer {
impl Quantizer for PerChannelQuantizer {
        // Simplified per-channel quantization
        // In practice, would compute scale/zero_point per channel
        AsymmetricQuantizer::new(self.bits).quantize(tensor)
/// Dynamic quantizer
struct DynamicQuantizer {
impl DynamicQuantizer {
impl Quantizer for DynamicQuantizer {
        // Dynamic quantization computes parameters at runtime
/// Calibration statistics
struct CalibrationStats {
    stats: std::collections::HashMap<String, TensorStats>,
impl CalibrationStats {
    fn new() -> Self {
            stats: std::collections::HashMap::new(),
    fn update(&mut self, name: &str, tensor: &ArrayView2<f32>) {
        let entry = self.stats.entry(name.to_string()).or_insert(TensorStats::new());
        entry.update(tensor);
/// Tensor statistics for calibration
struct TensorStats {
    min: f32,
    max: f32,
    mean: f32,
    count: usize,
impl TensorStats {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            count: 0,
    fn update(&mut self, tensor: &ArrayView2<f32>) {
        let current_min = tensor.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let current_max = tensor.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let current_mean = tensor.mean().unwrap();
        self.min = self.min.min(*current_min);
        self.max = self.max.max(*current_max);
        // Running average
        self.mean = (self.mean * self.count as f32 + current_mean) / (self.count + 1) as f32;
        self.count += 1;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_symmetric_quantization() {
        let quantizer = SymmetricQuantizer::new(8);
        let tensor = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -0.5, 0.5, 0.8]).unwrap();
        let quantized = quantizer.quantize(&tensor.view()).unwrap();
        assert_eq!(quantized.bits, 8);
        assert!(quantized.scale > 0.0);
    fn test_fake_quantize() {
        let qat = QuantizationAwareTraining::new(QATConfig::default());
        let fake_quantized = qat.fake_quantize(&tensor.view(), true).unwrap();
        assert_eq!(fake_quantized.shape(), tensor.shape());
        // Values should be slightly different due to quantization
        for (orig, quant) in tensor.iter().zip(fake_quantized.iter()) {
            assert!((orig - quant).abs() < 0.1); // Quantization error
    fn test_quantize_dequantize() {
        let quantized = qat.quantize_tensor(&tensor.view(), 8).unwrap();
        let dequantized = qat.dequantize(&quantized).unwrap();
        // Check round-trip error
        for (orig, dequant) in tensor.iter().zip(dequantized.iter()) {
            assert!((orig - dequant).abs() < 0.01); // Small quantization error expected
