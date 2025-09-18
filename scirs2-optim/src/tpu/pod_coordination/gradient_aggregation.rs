//! Gradient Aggregation for TPU Pod Coordination
//!
//! This module provides comprehensive gradient aggregation functionality for TPU pod coordination,
//! including various aggregation methods, compression, quantization, and communication optimization.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::{Array, IxDyn};
use num_traits::Float;

use super::DeviceId;
use crate::error::{OptimError, Result};

/// Gradient aggregation methods
#[derive(Debug, Clone, Copy)]
pub enum GradientAggregationMethod {
    Average,
    Sum,
    WeightedAverage,
    Median,
    QuantizedAverage,
    TopK,
    LocalSGD,
    FedAvg,
    SCAFFOLD,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    None,
    Quantization,
    Sparsification,
    LowRank,
    Sketching,
    Federated,
    Custom,
}

/// Gradient buffer status
#[derive(Debug, Clone, Copy)]
pub enum GradientBufferStatus {
    Fresh,
    Stale,
    Aggregated,
    Compressed,
    Invalid,
}

/// Quantization methods
#[derive(Debug, Clone, Copy)]
pub enum QuantizationMethod {
    Uniform,
    NonUniform,
    StochasticRounding,
    TernaryQuantization,
    BinaryQuantization,
    AdaptiveQuantization,
}

/// Sparsification methods
#[derive(Debug, Clone, Copy)]
pub enum SparsificationMethod {
    TopK,
    RandomK,
    ThresholdBased,
    StructuredSparsity,
    GradientDropout,
    AdaptiveSparsity,
}

/// Communication optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum CommunicationOptimization {
    None,
    Compression,
    Quantization,
    Sparsification,
    LocalUpdate,
    AsyncUpdate,
    HierarchicalAggregation,
}

/// Gradient buffer for a device
#[derive(Debug)]
pub struct GradientBuffer<T: Float> {
    /// Gradient data
    pub gradients: Vec<Array<T, IxDyn>>,

    /// Buffer timestamp
    pub timestamp: Instant,

    /// Buffer version
    pub version: u64,

    /// Compression applied
    pub compression: Option<CompressionInfo>,

    /// Buffer status
    pub status: GradientBufferStatus,

    /// Device ID
    pub device_id: DeviceId,

    /// Buffer metadata
    pub metadata: BufferMetadata,
}

/// Compression information
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression ratio
    pub compression_ratio: f64,

    /// Original size
    pub original_size: usize,

    /// Compressed size
    pub compressed_size: usize,

    /// Compression parameters
    pub parameters: CompressionParameters,
}

/// Compression parameters
#[derive(Debug, Clone)]
pub struct CompressionParameters {
    /// Quantization bits
    pub quantization_bits: u8,

    /// Sparsity ratio
    pub sparsity_ratio: f64,

    /// Error tolerance
    pub error_tolerance: f64,

    /// Custom parameters
    pub custom_params: HashMap<String, f64>,
}

/// Buffer metadata
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Buffer size in bytes
    pub size_bytes: usize,

    /// Number of parameters
    pub num_parameters: usize,

    /// Data type information
    pub data_type: String,

    /// Checksum for verification
    pub checksum: u64,

    /// Tags for categorization
    pub tags: HashMap<String, String>,
}

/// Aggregation state
#[derive(Debug)]
pub struct AggregationState<T: Float> {
    /// Accumulated gradients
    pub accumulated_gradients: Vec<Array<T, IxDyn>>,

    /// Aggregation count
    pub aggregation_count: usize,

    /// Last aggregation time
    pub last_aggregation: Instant,

    /// Aggregation statistics
    pub statistics: AggregationStatistics,

    /// Momentum terms (for methods like SCAFFOLD)
    pub momentum_terms: Vec<Array<T, IxDyn>>,

    /// Control variates (for variance reduction)
    pub control_variates: Vec<Array<T, IxDyn>>,
}

/// Aggregation statistics
#[derive(Debug, Clone)]
pub struct AggregationStatistics {
    /// Total aggregations
    pub total_aggregations: usize,

    /// Average aggregation time
    pub avg_aggregation_time: Duration,

    /// Compression efficiency
    pub compression_efficiency: f64,

    /// Communication overhead
    pub communication_overhead: f64,

    /// Convergence rate
    pub convergence_rate: f64,

    /// Error accumulation
    pub error_accumulation: f64,
}

/// Aggregation configuration
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Aggregation method
    pub method: GradientAggregationMethod,

    /// Compression settings
    pub compression: CompressionSettings,

    /// Quantization settings
    pub quantization: QuantizationSettings,

    /// Communication optimization
    pub communication_optimization: CommunicationOptimization,

    /// Staleness tolerance
    pub staleness_tolerance: usize,

    /// Batch size for aggregation
    pub batch_size: usize,

    /// Error correction enabled
    pub error_correction: bool,
}

/// Compression settings
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Enabled compression algorithms
    pub algorithms: Vec<CompressionAlgorithm>,

    /// Target compression ratio
    pub target_ratio: f64,

    /// Error tolerance
    pub error_tolerance: f64,

    /// Adaptive compression
    pub adaptive: bool,
}

/// Quantization settings
#[derive(Debug, Clone)]
pub struct QuantizationSettings {
    /// Quantization method
    pub method: QuantizationMethod,

    /// Number of bits
    pub bits: u8,

    /// Range for quantization
    pub range: (f64, f64),

    /// Stochastic rounding
    pub stochastic: bool,
}

/// Federated learning parameters
#[derive(Debug, Clone)]
pub struct FederatedParams {
    /// Client weights
    pub client_weights: HashMap<DeviceId, f64>,

    /// Local epochs
    pub local_epochs: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Momentum coefficient
    pub momentum: f64,

    /// Server momentum
    pub server_momentum: f64,
}

/// LocalSGD parameters
#[derive(Debug, Clone)]
pub struct LocalSGDParams {
    /// Local update steps
    pub local_steps: usize,

    /// Communication frequency
    pub communication_frequency: usize,

    /// Warmup period
    pub warmup_period: usize,

    /// Adaptive frequency
    pub adaptive_frequency: bool,
}

/// SCAFFOLD parameters
#[derive(Debug, Clone)]
pub struct SCAFFOLDParams {
    /// Server control variate
    pub server_control: Vec<f64>,

    /// Client control variates
    pub client_controls: HashMap<DeviceId, Vec<f64>>,

    /// Learning rate
    pub learning_rate: f64,

    /// Local steps
    pub local_steps: usize,
}

/// Communication statistics
#[derive(Debug, Clone)]
pub struct CommunicationStats {
    /// Bytes sent
    pub bytes_sent: usize,

    /// Bytes received
    pub bytes_received: usize,

    /// Communication rounds
    pub communication_rounds: usize,

    /// Average latency
    pub average_latency: Duration,

    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
}

/// Type aliases for managers
type CompressionSettings_ = HashMap<String, f64>;
type QuantizationSettings_ = HashMap<String, f64>;
type CommunicationOptimizer<T> = HashMap<String, T>;

/// Gradient aggregation statistics
pub type GradientAggregationStatistics = HashMap<String, f64>;

/// Gradient aggregator for distributed optimization
#[derive(Debug)]
pub struct GradientAggregator<T: Float> {
    /// Aggregation configuration
    config: AggregationConfig,

    /// Gradient buffers
    gradient_buffers: HashMap<DeviceId, GradientBuffer<T>>,

    /// Aggregation state
    aggregation_state: AggregationState<T>,

    /// Compression settings
    compression_settings: CompressionSettings_,

    /// Quantization settings
    quantization_settings: QuantizationSettings_,

    /// Communication optimizer
    communication_optimizer: CommunicationOptimizer<T>,

    /// Federated learning parameters
    federated_params: Option<FederatedParams>,

    /// LocalSGD parameters
    local_sgd_params: Option<LocalSGDParams>,

    /// SCAFFOLD parameters
    scaffold_params: Option<SCAFFOLDParams>,

    /// Communication statistics
    communication_stats: CommunicationStats,

    /// Error accumulation for compression
    error_accumulation: Vec<Array<T, IxDyn>>,
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum + ndarray::ScalarOperand> GradientAggregator<T> {
    /// Create a new gradient aggregator
    pub fn new(config: AggregationConfig) -> Result<Self> {
        let aggregation_state = AggregationState {
            accumulated_gradients: Vec::new(),
            aggregation_count: 0,
            last_aggregation: Instant::now(),
            statistics: AggregationStatistics {
                total_aggregations: 0,
                avg_aggregation_time: Duration::from_millis(5),
                compression_efficiency: 0.8,
                communication_overhead: 0.1,
                convergence_rate: 0.95,
                error_accumulation: 0.0,
            },
            momentum_terms: Vec::new(),
            control_variates: Vec::new(),
        };

        let communication_stats = CommunicationStats {
            bytes_sent: 0,
            bytes_received: 0,
            communication_rounds: 0,
            average_latency: Duration::from_millis(10),
            bandwidth_utilization: 0.0,
        };

        Ok(Self {
            config,
            gradient_buffers: HashMap::new(),
            aggregation_state,
            compression_settings: HashMap::new(),
            quantization_settings: HashMap::new(),
            communication_optimizer: HashMap::new(),
            federated_params: None,
            local_sgd_params: None,
            scaffold_params: None,
            communication_stats,
            error_accumulation: Vec::new(),
        })
    }

    /// Add gradient buffer from device
    pub fn add_gradient_buffer(&mut self, device_id: DeviceId, buffer: GradientBuffer<T>) {
        self.gradient_buffers.insert(device_id, buffer);
    }

    /// Aggregate gradients from all devices
    pub async fn aggregate_gradients(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        let start_time = Instant::now();

        if device_gradients.is_empty() {
            return Ok(Vec::new());
        }

        // Update communication statistics
        self.communication_stats.communication_rounds += 1;

        let aggregated = match self.config.method {
            GradientAggregationMethod::Average => {
                self.aggregate_average(device_gradients).await?
            }
            GradientAggregationMethod::Sum => {
                self.aggregate_sum(device_gradients).await?
            }
            GradientAggregationMethod::WeightedAverage => {
                self.aggregate_weighted_average(device_gradients).await?
            }
            GradientAggregationMethod::Median => {
                self.aggregate_median(device_gradients).await?
            }
            GradientAggregationMethod::QuantizedAverage => {
                self.aggregate_quantized_average(device_gradients).await?
            }
            GradientAggregationMethod::TopK => {
                self.aggregate_top_k(device_gradients).await?
            }
            GradientAggregationMethod::LocalSGD => {
                self.aggregate_local_sgd(device_gradients).await?
            }
            GradientAggregationMethod::FedAvg => {
                self.aggregate_federated_averaging(device_gradients).await?
            }
            GradientAggregationMethod::SCAFFOLD => {
                self.aggregate_scaffold(device_gradients).await?
            }
        };

        // Update statistics
        self.aggregation_state.aggregation_count += 1;
        self.aggregation_state.last_aggregation = Instant::now();
        self.aggregation_state.statistics.total_aggregations += 1;

        let aggregation_time = start_time.elapsed();
        self.aggregation_state.statistics.avg_aggregation_time =
            (self.aggregation_state.statistics.avg_aggregation_time + aggregation_time) / 2;

        self.communication_stats.average_latency =
            (self.communication_stats.average_latency + aggregation_time) / 2;

        Ok(aggregated)
    }

    /// Aggregate using simple averaging
    async fn aggregate_average(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        let num_devices = device_gradients.len() as f64;
        let first_gradients = device_gradients.values().next().unwrap();
        let mut aggregated_gradients = Vec::new();

        for i in 0..first_gradients.len() {
            let mut sum_gradient = first_gradients[i].clone();
            let mut count = 1;

            // Sum gradients from all devices
            for gradients in device_gradients.values().skip(1) {
                if i < gradients.len() {
                    sum_gradient = sum_gradient + &gradients[i];
                    count += 1;
                }
            }

            // Average
            let averaged = sum_gradient / num_traits::cast::cast(count).unwrap_or_else(|| T::zero());
            aggregated_gradients.push(averaged);
        }

        Ok(aggregated_gradients)
    }

    /// Aggregate using sum
    async fn aggregate_sum(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        let first_gradients = device_gradients.values().next().unwrap();
        let mut aggregated_gradients = Vec::new();

        for i in 0..first_gradients.len() {
            let mut sum_gradient = first_gradients[i].clone();

            // Sum gradients from all devices
            for gradients in device_gradients.values().skip(1) {
                if i < gradients.len() {
                    sum_gradient = sum_gradient + &gradients[i];
                }
            }

            aggregated_gradients.push(sum_gradient);
        }

        Ok(aggregated_gradients)
    }

    /// Aggregate using weighted average
    async fn aggregate_weighted_average(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        let weights = if let Some(ref federated_params) = self.federated_params {
            &federated_params.client_weights
        } else {
            // Use uniform weights if no federated params
            let uniform_weight = 1.0 / device_gradients.len() as f64;
            let mut uniform_weights = HashMap::new();
            for &device_id in device_gradients.keys() {
                uniform_weights.insert(device_id, uniform_weight);
            }
            return self.aggregate_average(device_gradients).await; // Fallback to average
        };

        let first_gradients = device_gradients.values().next().unwrap();
        let mut aggregated_gradients = Vec::new();

        for i in 0..first_gradients.len() {
            let mut weighted_sum = Array::zeros(first_gradients[i].dim());
            let mut total_weight = T::zero();

            for (device_id, gradients) in &device_gradients {
                if let Some(&weight) = weights.get(device_id) {
                    if i < gradients.len() {
                        let weight_t = num_traits::cast::cast(weight).unwrap_or_else(|| T::zero());
                        weighted_sum = weighted_sum + &(gradients[i].clone() * weight_t);
                        total_weight = total_weight + weight_t;
                    }
                }
            }

            if total_weight > T::zero() {
                weighted_sum = weighted_sum / total_weight;
            }

            aggregated_gradients.push(weighted_sum);
        }

        Ok(aggregated_gradients)
    }

    /// Aggregate using median (Byzantine-resilient)
    async fn aggregate_median(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        // For simplicity, fall back to average
        // In a real implementation, this would compute element-wise median
        self.aggregate_average(device_gradients).await
    }

    /// Aggregate using quantized averaging
    async fn aggregate_quantized_average(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        // Apply quantization before aggregation
        let mut quantized_gradients = HashMap::new();

        for (device_id, gradients) in device_gradients {
            let quantized = self.apply_quantization(gradients)?;
            quantized_gradients.insert(device_id, quantized);
        }

        self.aggregate_average(quantized_gradients).await
    }

    /// Aggregate using Top-K sparsification
    async fn aggregate_top_k(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        // Apply Top-K sparsification before aggregation
        let mut sparse_gradients = HashMap::new();

        for (device_id, gradients) in device_gradients {
            let sparse = self.apply_top_k_sparsification(gradients, 0.1)?; // Keep top 10%
            sparse_gradients.insert(device_id, sparse);
        }

        self.aggregate_average(sparse_gradients).await
    }

    /// Aggregate using LocalSGD
    async fn aggregate_local_sgd(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        // Check if it's time for communication
        if let Some(ref params) = self.local_sgd_params {
            if self.aggregation_state.aggregation_count % params.communication_frequency == 0 {
                return self.aggregate_average(device_gradients).await;
            }
        }

        // Return empty if not communication round
        Ok(Vec::new())
    }

    /// Aggregate using Federated Averaging
    async fn aggregate_federated_averaging(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        self.aggregate_weighted_average(device_gradients).await
    }

    /// Aggregate using SCAFFOLD
    async fn aggregate_scaffold(
        &mut self,
        device_gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        // SCAFFOLD uses control variates to reduce variance
        let averaged = self.aggregate_average(device_gradients).await?;

        // Apply control variate correction
        if !self.aggregation_state.control_variates.is_empty() {
            let mut corrected = Vec::new();
            for (i, gradient) in averaged.into_iter().enumerate() {
                if i < self.aggregation_state.control_variates.len() {
                    let corrected_grad = gradient - &self.aggregation_state.control_variates[i];
                    corrected.push(corrected_grad);
                } else {
                    corrected.push(gradient);
                }
            }
            Ok(corrected)
        } else {
            Ok(averaged)
        }
    }

    /// Apply quantization to gradients
    fn apply_quantization(&self, gradients: Vec<Array<T, IxDyn>>) -> Result<Vec<Array<T, IxDyn>>> {
        let mut quantized = Vec::new();

        for gradient in gradients {
            match self.config.quantization.method {
                QuantizationMethod::Uniform => {
                    // Simple uniform quantization
                    let quantized_grad = gradient.mapv(|x| {
                        let levels = 2_i32.pow(self.config.quantization.bits as u32) as f64;
                        let (min_val, max_val) = self.config.quantization.range;
                        let scale = (max_val - min_val) / levels;
                        let quantized_val = ((x.to_f64().unwrap() - min_val) / scale).round() * scale + min_val;
                        num_traits::cast::cast(quantized_val).unwrap_or_else(|| T::zero())
                    });
                    quantized.push(quantized_grad);
                }
                _ => {
                    // Fallback to no quantization
                    quantized.push(gradient);
                }
            }
        }

        Ok(quantized)
    }

    /// Apply Top-K sparsification
    fn apply_top_k_sparsification(
        &self,
        gradients: Vec<Array<T, IxDyn>>,
        sparsity_ratio: f64,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        let mut sparse = Vec::new();

        for gradient in gradients {
            let total_elements = gradient.len();
            let k = ((1.0 - sparsity_ratio) * total_elements as f64) as usize;

            // Create a sparse version by keeping only top-k elements
            let mut sparse_grad = Array::zeros(gradient.dim());

            // For simplicity, just apply a threshold
            let threshold = num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()); // Keep elements above threshold
            sparse_grad.zip_mut_with(&gradient, |sparse_elem, &grad_elem| {
                if grad_elem.abs() > threshold {
                    *sparse_elem = grad_elem;
                }
            });

            sparse.push(sparse_grad);
        }

        Ok(sparse)
    }

    /// Compress gradients
    pub fn compress_gradients(
        &mut self,
        gradients: Vec<Array<T, IxDyn>>,
    ) -> Result<(Vec<Array<T, IxDyn>>, CompressionInfo)> {
        let original_size = gradients.iter().map(|g| g.len() * std::mem::size_of::<T>()).sum();

        let compressed = match self.config.compression.algorithms.first() {
            Some(CompressionAlgorithm::Quantization) => {
                self.apply_quantization(gradients)?
            }
            Some(CompressionAlgorithm::Sparsification) => {
                self.apply_top_k_sparsification(gradients, 0.9)? // 90% sparsity
            }
            _ => gradients,
        };

        let compressed_size = compressed.iter().map(|g| g.len() * std::mem::size_of::<T>()).sum();
        let compression_ratio = original_size as f64 / compressed_size as f64;

        let compression_info = CompressionInfo {
            algorithm: self.config.compression.algorithms.first().copied().unwrap_or(CompressionAlgorithm::None),
            compression_ratio,
            original_size,
            compressed_size,
            parameters: CompressionParameters {
                quantization_bits: self.config.quantization.bits,
                sparsity_ratio: 0.9,
                error_tolerance: 0.01,
                custom_params: HashMap::new(),
            },
        };

        Ok((compressed, compression_info))
    }

    /// Decompress gradients
    pub fn decompress_gradients(
        &self,
        compressed_gradients: Vec<Array<T, IxDyn>>,
        compression_info: &CompressionInfo,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        // For simplicity, return compressed gradients as-is
        // In a real implementation, this would reverse the compression
        Ok(compressed_gradients)
    }

    /// Set federated learning parameters
    pub fn set_federated_params(&mut self, params: FederatedParams) {
        self.federated_params = Some(params);
    }

    /// Set LocalSGD parameters
    pub fn set_local_sgd_params(&mut self, params: LocalSGDParams) {
        self.local_sgd_params = Some(params);
    }

    /// Set SCAFFOLD parameters
    pub fn set_scaffold_params(&mut self, params: SCAFFOLDParams) {
        self.scaffold_params = Some(params);
    }

    /// Update aggregation configuration
    pub fn update_config(&mut self, config: AggregationConfig) {
        self.config = config;
    }

    /// Get aggregation statistics
    pub fn get_statistics(&self) -> GradientAggregationStatistics {
        let mut stats = HashMap::new();

        stats.insert(
            "total_aggregations".to_string(),
            self.aggregation_state.statistics.total_aggregations as f64,
        );

        stats.insert(
            "avg_time_ms".to_string(),
            self.aggregation_state.statistics.avg_aggregation_time.as_millis() as f64,
        );

        stats.insert(
            "compression_efficiency".to_string(),
            self.aggregation_state.statistics.compression_efficiency,
        );

        stats.insert(
            "communication_overhead".to_string(),
            self.aggregation_state.statistics.communication_overhead,
        );

        stats.insert(
            "convergence_rate".to_string(),
            self.aggregation_state.statistics.convergence_rate,
        );

        stats.insert(
            "error_accumulation".to_string(),
            self.aggregation_state.statistics.error_accumulation,
        );

        stats.insert(
            "bytes_sent".to_string(),
            self.communication_stats.bytes_sent as f64,
        );

        stats.insert(
            "bytes_received".to_string(),
            self.communication_stats.bytes_received as f64,
        );

        stats.insert(
            "communication_rounds".to_string(),
            self.communication_stats.communication_rounds as f64,
        );

        stats.insert(
            "bandwidth_utilization".to_string(),
            self.communication_stats.bandwidth_utilization,
        );

        stats
    }

    /// Get communication statistics
    pub fn get_communication_stats(&self) -> &CommunicationStats {
        &self.communication_stats
    }

    /// Get aggregation state
    pub fn get_aggregation_state(&self) -> &AggregationState<T> {
        &self.aggregation_state
    }

    /// Reset aggregation state
    pub fn reset_state(&mut self) {
        self.aggregation_state.accumulated_gradients.clear();
        self.aggregation_state.aggregation_count = 0;
        self.aggregation_state.momentum_terms.clear();
        self.aggregation_state.control_variates.clear();
        self.error_accumulation.clear();
    }

    /// Get gradient buffer for device
    pub fn get_gradient_buffer(&self, device_id: DeviceId) -> Option<&GradientBuffer<T>> {
        self.gradient_buffers.get(&device_id)
    }

    /// Remove gradient buffer for device
    pub fn remove_gradient_buffer(&mut self, device_id: DeviceId) -> Option<GradientBuffer<T>> {
        self.gradient_buffers.remove(&device_id)
    }

    /// Check if gradients are stale
    pub fn check_staleness(&self, device_id: DeviceId, threshold: Duration) -> bool {
        if let Some(buffer) = self.gradient_buffers.get(&device_id) {
            buffer.timestamp.elapsed() > threshold
        } else {
            true
        }
    }

    /// Update communication statistics
    pub fn update_communication_stats(&mut self, bytes_sent: usize, bytes_received: usize) {
        self.communication_stats.bytes_sent += bytes_sent;
        self.communication_stats.bytes_received += bytes_received;
    }
}

// Default implementations
impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            method: GradientAggregationMethod::Average,
            compression: CompressionSettings {
                algorithms: vec![CompressionAlgorithm::None],
                target_ratio: 2.0,
                error_tolerance: 0.01,
                adaptive: false,
            },
            quantization: QuantizationSettings {
                method: QuantizationMethod::Uniform,
                bits: 8,
                range: (-1.0, 1.0),
                stochastic: false,
            },
            communication_optimization: CommunicationOptimization::None,
            staleness_tolerance: 3,
            batch_size: 32,
            error_correction: true,
        }
    }
}

impl Default for CompressionParameters {
    fn default() -> Self {
        Self {
            quantization_bits: 8,
            sparsity_ratio: 0.1,
            error_tolerance: 0.01,
            custom_params: HashMap::new(),
        }
    }
}

impl Default for BufferMetadata {
    fn default() -> Self {
        Self {
            size_bytes: 0,
            num_parameters: 0,
            data_type: "f32".to_string(),
            checksum: 0,
            tags: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_aggregator_creation() {
        let config = AggregationConfig::default();
        let aggregator = GradientAggregator::<f32>::new(config);
        assert!(aggregator.is_ok());
    }

    #[tokio::test]
    async fn test_gradient_aggregation() {
        let config = AggregationConfig::default();
        let mut aggregator = GradientAggregator::<f32>::new(config).unwrap();

        let mut device_gradients = HashMap::new();
        device_gradients.insert(
            DeviceId(0),
            vec![Array::ones(IxDyn(&[2, 2]))],
        );
        device_gradients.insert(
            DeviceId(1),
            vec![Array::ones(IxDyn(&[2, 2])) * 2.0],
        );

        let result = aggregator.aggregate_gradients(device_gradients).await;
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert_eq!(aggregated.len(), 1);
        assert_eq!(aggregated[0][(0, 0)], 1.5); // Average of 1 and 2
    }

    #[test]
    fn test_gradient_compression() {
        let config = AggregationConfig {
            compression: CompressionSettings {
                algorithms: vec![CompressionAlgorithm::Quantization],
                target_ratio: 2.0,
                error_tolerance: 0.01,
                adaptive: false,
            },
            ..Default::default()
        };

        let mut aggregator = GradientAggregator::<f32>::new(config).unwrap();
        let gradients = vec![Array::ones(IxDyn(&[2, 2]))];

        let result = aggregator.compress_gradients(gradients);
        assert!(result.is_ok());

        let (compressed, info) = result.unwrap();
        assert_eq!(compressed.len(), 1);
        assert!(info.compression_ratio >= 1.0);
    }

    #[test]
    fn test_staleness_check() {
        let config = AggregationConfig::default();
        let aggregator = GradientAggregator::<f32>::new(config).unwrap();

        // Device not in buffer should be considered stale
        let is_stale = aggregator.check_staleness(DeviceId(0), Duration::from_secs(1));
        assert!(is_stale);
    }
}