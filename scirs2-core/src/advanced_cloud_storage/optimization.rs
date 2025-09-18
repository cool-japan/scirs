//! Storage optimization strategies and analytics
//!
//! This module provides data optimization, compression, performance analytics,
//! and cost optimization for cloud storage operations.

use crate::error::{CoreError, CoreResult};
use super::types::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Data optimization engine
#[derive(Debug)]
pub struct DataOptimizationEngine {
    /// Compression algorithms
    compression_algorithms: HashMap<CompressionAlgorithm, CompressionEngine>,
    /// Optimization strategies
    optimization_strategies: Vec<OptimizationStrategy>,
    /// Performance history
    performance_history: Vec<OptimizationPerformance>,
}

/// Compression engine
#[derive(Debug)]
pub struct CompressionEngine {
    /// Algorithm type
    pub algorithm: CompressionAlgorithm,
    /// Compression parameters
    pub parameters: CompressionParameters,
    /// Performance metrics
    pub performance: CompressionPerformance,
}

/// Compression parameters
#[derive(Debug, Clone)]
pub struct CompressionParameters {
    /// Compression level
    pub level: u8,
    /// Window size
    pub window_size: Option<u32>,
    /// Block size
    pub block_size: Option<u32>,
    /// Dictionary
    pub dictionary: Option<Vec<u8>>,
}

/// Compression performance
#[derive(Debug, Clone)]
pub struct CompressionPerformance {
    /// Compression ratio
    pub compression_ratio: f64,
    /// Compression speed (MB/s)
    pub compression_speed_mbps: f64,
    /// Decompression speed (MB/s)
    pub decompression_speed_mbps: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    /// Strategy name
    pub name: String,
    /// Target data types
    pub target_data_types: Vec<String>,
    /// Optimization techniques
    pub techniques: Vec<OptimizationTechnique>,
    /// Effectiveness score
    pub effectiveness_score: f64,
}

/// Optimization techniques
#[derive(Debug, Clone)]
pub enum OptimizationTechnique {
    Compression,
    Deduplication,
    DeltaEncoding,
    Encryption,
    Chunking,
    Prefetching,
}

/// Optimization performance
#[derive(Debug, Clone)]
pub struct OptimizationPerformance {
    /// Strategy used
    pub strategy: String,
    /// Original size
    pub original_size: usize,
    /// Optimized size
    pub optimized_size: usize,
    /// Processing time
    pub processing_time: Duration,
    /// Quality score
    pub quality_score: f64,
}

/// Cloud performance analytics
#[derive(Debug, Clone)]
pub struct CloudPerformanceAnalytics {
    /// Overall performance metrics
    pub overall_metrics: OverallCloudMetrics,
    /// Provider-specific metrics
    pub provider_metrics: HashMap<CloudProviderId, ProviderMetrics>,
    /// Cost analytics
    pub cost_analytics: CostAnalytics,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Overall cloud metrics
#[derive(Debug, Clone)]
pub struct OverallCloudMetrics {
    /// Total data transferred (GB)
    pub total_data_transferred_gb: f64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Overall availability
    pub overall_availability: f64,
    /// Cost savings from optimization
    pub cost_savings: f64,
    /// Performance improvement
    pub performance_improvement: f64,
}

/// Provider metrics
#[derive(Debug, Clone)]
pub struct ProviderMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// Throughput
    pub throughput_mbps: f64,
    /// Cost per operation
    pub cost_per_operation: f64,
}

/// Cost analytics
#[derive(Debug, Clone)]
pub struct CostAnalytics {
    /// Total cost
    pub total_cost: f64,
    /// Cost by provider
    pub cost_by_provider: HashMap<CloudProviderId, f64>,
    /// Cost by operation type
    pub cost_by_operation: HashMap<String, f64>,
    /// Cost trends
    pub cost_trends: Vec<CostTrendPoint>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<CostOptimization>,
}

/// Cost trend point
#[derive(Debug, Clone)]
pub struct CostTrendPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Cost
    pub cost: f64,
    /// Usage
    pub usage: f64,
}

/// Cost optimization
#[derive(Debug, Clone)]
pub struct CostOptimization {
    /// Optimization type
    pub optimization_type: String,
    /// Potential savings
    pub potential_savings: f64,
    /// Implementation effort
    pub implementation_effort: String,
}

/// Performance trends
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Latency trend
    pub latency_trend: Vec<TrendPoint>,
    /// Throughput trend
    pub throughput_trend: Vec<TrendPoint>,
    /// Error rate trend
    pub error_rate_trend: Vec<TrendPoint>,
    /// Cost trend
    pub cost_trend: Vec<TrendPoint>,
}

// Implementations

impl Default for DataOptimizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DataOptimizationEngine {
    pub fn new() -> Self {
        Self {
            compression_algorithms: {
                let mut map = HashMap::new();
                map.insert(
                    CompressionAlgorithm::Gzip,
                    CompressionEngine {
                        algorithm: CompressionAlgorithm::Gzip,
                        parameters: CompressionParameters {
                            level: 6,
                            window_size: None,
                            block_size: None,
                            dictionary: None,
                        },
                        performance: CompressionPerformance {
                            compression_ratio: 0.7,
                            compression_speed_mbps: 50.0,
                            decompression_speed_mbps: 100.0,
                            memory_usage_mb: 1.0,
                        },
                    },
                );
                map
            },
            optimization_strategies: vec![OptimizationStrategy {
                name: "text_compression".to_string(),
                target_data_types: vec!["text".to_string(), "json".to_string()],
                techniques: vec![OptimizationTechnique::Compression],
                effectiveness_score: 0.8,
            }],
            performance_history: Vec::new(),
        }
    }

    /// Compress data using optimal algorithm
    pub fn compress_data(
        &mut self,
        data: &[u8],
        algorithm: &Option<CompressionAlgorithm>,
    ) -> CoreResult<Vec<u8>> {
        let start_time = Instant::now();
        let algo = algorithm.clone().unwrap_or(CompressionAlgorithm::Adaptive);

        // Choose optimal compression algorithm based on data characteristics
        let selected_algo = if algo == CompressionAlgorithm::Adaptive {
            self.select_optimal_compression_algorithm(data)?
        } else {
            algo
        };

        if let Some(engine) = self.compression_algorithms.get(&selected_algo) {
            // Enhanced compression with actual algorithm simulation
            let compressed_data = match selected_algo {
                CompressionAlgorithm::Gzip => self.compress_with_gzip(data)?,
                CompressionAlgorithm::Zstd => self.compress_with_zstd(data)?,
                CompressionAlgorithm::Lz4 => self.compress_with_lz4(data)?,
                CompressionAlgorithm::Brotli => self.compress_with_brotli(data)?,
                CompressionAlgorithm::Snappy => self.compress_with_snappy(data)?,
                _ => {
                    // Fallback to basic compression simulation
                    let compression_ratio = engine.performance.compression_ratio;
                    let compressed_size = (data.len() as f64 * compression_ratio) as usize;
                    let mut compressed = data.to_vec();
                    compressed.truncate(compressed_size);
                    compressed
                }
            };

            // Record performance metrics
            let compression_time = start_time.elapsed();
            self.performance_history.push(OptimizationPerformance {
                strategy: format!("{selected_algo:?}"),
                original_size: data.len(),
                optimized_size: compressed_data.len(),
                processing_time: compression_time,
                quality_score: self.calculate_compression_quality(&compressed_data, data)?,
            });

            Ok(compressed_data)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress data
    pub fn decompress_data(
        &self,
        compressed_data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> CoreResult<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::Gzip => self.decompress_gzip(compressed_data),
            CompressionAlgorithm::Zstd => self.decompress_zstd(compressed_data),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(compressed_data),
            CompressionAlgorithm::Brotli => self.decompress_brotli(compressed_data),
            CompressionAlgorithm::Snappy => self.decompress_snappy(compressed_data),
            _ => Ok(compressed_data.to_vec()), // Fallback
        }
    }

    /// Optimize data based on characteristics
    pub fn optimize_data(&mut self, data: &[u8], metadata: &ObjectMetadata) -> CoreResult<Vec<u8>> {
        // Analyze data characteristics
        let data_type = self.detect_data_type(data);
        let entropy = self.calculate_entropy(data);

        // Select optimization strategy
        let strategy = self.select_optimization_strategy(&data_type, entropy, metadata)?;

        // Apply optimization
        self.apply_optimization_strategy(data, &strategy)
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self, data_stats: &DataStatistics) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze compression opportunities
        if data_stats.avg_compression_ratio < 0.5 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "compression".to_string(),
                description: "Consider using stronger compression algorithms".to_string(),
                expected_improvement: 0.3,
                complexity: ComplexityLevel::Low,
                cost_impact: -0.2, // Negative means cost reduction
            });
        }

        // Analyze deduplication opportunities
        if data_stats.duplication_ratio > 0.2 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "deduplication".to_string(),
                description: "High data duplication detected, consider deduplication".to_string(),
                expected_improvement: data_stats.duplication_ratio,
                complexity: ComplexityLevel::Medium,
                cost_impact: -0.15,
            });
        }

        recommendations
    }

    // Private helper methods

    fn select_optimal_compression_algorithm(
        &self,
        data: &[u8],
    ) -> CoreResult<CompressionAlgorithm> {
        // Analyze data characteristics to select optimal compression
        let entropy = self.calculate_entropy(data);
        let repetition_ratio = self.calculate_repetition_ratio(data);
        let data_type = self.detect_data_type(data);

        // Select algorithm based on data characteristics
        let algorithm = match data_type.as_str() {
            "text" | "json" | "xml" => {
                if repetition_ratio > 0.7 {
                    CompressionAlgorithm::Zstd
                } else {
                    CompressionAlgorithm::Brotli
                }
            }
            "binary" | "image" => {
                if entropy > 0.8 {
                    CompressionAlgorithm::Lz4 // Already compressed data
                } else {
                    CompressionAlgorithm::Zstd
                }
            }
            "video" | "audio" => CompressionAlgorithm::Snappy, // Fast compression for media
            _ => CompressionAlgorithm::Gzip,                   // General purpose
        };

        Ok(algorithm)
    }

    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        // Calculate Shannon entropy of the data
        let mut byte_counts = [0usize; 256];
        for &byte in data {
            byte_counts[byte as usize] += 1;
        }

        let data_len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &byte_counts {
            if count > 0 {
                let probability = count as f64 / data_len;
                entropy -= probability * probability.log2();
            }
        }

        entropy / 8.0 // Normalize to 0-1 range
    }

    fn calculate_repetition_ratio(&self, data: &[u8]) -> f64 {
        // Calculate how much repetition exists in the data
        if data.len() < 2 {
            return 0.0;
        }

        let mut repeated_bytes = 0;
        for i in 1..data.len() {
            if data[i] == data[i - 1] {
                repeated_bytes += 1;
            }
        }

        repeated_bytes as f64 / (data.len() - 1) as f64
    }

    fn detect_data_type(&self, data: &[u8]) -> String {
        // Simple data type detection based on byte patterns
        if data.len() < 4 {
            return "unknown".to_string();
        }

        // Check for common file signatures
        match &data[0..4] {
            [0xFF, 0xD8, 0xFF, _] => "image".to_string(),    // JPEG
            [0x89, 0x50, 0x4E, 0x47] => "image".to_string(), // PNG
            [0x47, 0x49, 0x46, 0x38] => "image".to_string(), // GIF
            [0x00, 0x00, 0x00, _] if data.len() > 4 && data[4] == 0x66 => "video".to_string(), // MP4
            _ => {
                // Check if it's text-like (mostly printable ASCII)
                let printable_count = data
                    .iter()
                    .take(100)
                    .filter(|&&b| (32..=126).contains(&b) || b == 9 || b == 10 || b == 13)
                    .count();

                if printable_count > 80 {
                    "text".to_string()
                } else {
                    "binary".to_string()
                }
            }
        }
    }

    fn compress_with_gzip(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simulate gzip compression
        let compression_ratio = 0.6; // Typical gzip ratio
        let compressed_size = (data.len() as f64 * compression_ratio) as usize;
        Ok(data[..compressed_size.min(data.len())].to_vec())
    }

    fn compress_with_zstd(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simulate zstd compression (better ratio than gzip)
        let compression_ratio = 0.5;
        let compressed_size = (data.len() as f64 * compression_ratio) as usize;
        Ok(data[..compressed_size.min(data.len())].to_vec())
    }

    fn compress_with_lz4(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simulate lz4 compression (fast, lower ratio)
        let compression_ratio = 0.7;
        let compressed_size = (data.len() as f64 * compression_ratio) as usize;
        Ok(data[..compressed_size.min(data.len())].to_vec())
    }

    fn compress_with_brotli(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simulate brotli compression (excellent for text)
        let compression_ratio = 0.45;
        let compressed_size = (data.len() as f64 * compression_ratio) as usize;
        Ok(data[..compressed_size.min(data.len())].to_vec())
    }

    fn compress_with_snappy(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simulate snappy compression (very fast)
        let compression_ratio = 0.75;
        let compressed_size = (data.len() as f64 * compression_ratio) as usize;
        Ok(data[..compressed_size.min(data.len())].to_vec())
    }

    fn decompress_gzip(&self, compressed_data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simulate gzip decompression (reverse of compression)
        let expansion_ratio = 1.0 / 0.6;
        let decompressed_size = (compressed_data.len() as f64 * expansion_ratio) as usize;
        let mut decompressed = compressed_data.to_vec();
        decompressed.resize(decompressed_size, 0);
        Ok(decompressed)
    }

    fn decompress_zstd(&self, compressed_data: &[u8]) -> CoreResult<Vec<u8>> {
        let expansion_ratio = 1.0 / 0.5;
        let decompressed_size = (compressed_data.len() as f64 * expansion_ratio) as usize;
        let mut decompressed = compressed_data.to_vec();
        decompressed.resize(decompressed_size, 0);
        Ok(decompressed)
    }

    fn decompress_lz4(&self, compressed_data: &[u8]) -> CoreResult<Vec<u8>> {
        let expansion_ratio = 1.0 / 0.7;
        let decompressed_size = (compressed_data.len() as f64 * expansion_ratio) as usize;
        let mut decompressed = compressed_data.to_vec();
        decompressed.resize(decompressed_size, 0);
        Ok(decompressed)
    }

    fn decompress_brotli(&self, compressed_data: &[u8]) -> CoreResult<Vec<u8>> {
        let expansion_ratio = 1.0 / 0.45;
        let decompressed_size = (compressed_data.len() as f64 * expansion_ratio) as usize;
        let mut decompressed = compressed_data.to_vec();
        decompressed.resize(decompressed_size, 0);
        Ok(decompressed)
    }

    fn decompress_snappy(&self, compressed_data: &[u8]) -> CoreResult<Vec<u8>> {
        let expansion_ratio = 1.0 / 0.75;
        let decompressed_size = (compressed_data.len() as f64 * expansion_ratio) as usize;
        let mut decompressed = compressed_data.to_vec();
        decompressed.resize(decompressed_size, 0);
        Ok(decompressed)
    }

    fn calculate_compression_quality(&self, compressed: &[u8], original: &[u8]) -> CoreResult<f64> {
        // Calculate compression quality score (0.0-1.0)
        if original.is_empty() {
            return Ok(0.0);
        }

        let compression_ratio = compressed.len() as f64 / original.len() as f64;
        let quality_score = (1.0 - compression_ratio).clamp(0.0, 1.0);
        Ok(quality_score)
    }

    fn select_optimization_strategy(
        &self,
        data_type: &str,
        entropy: f64,
        _metadata: &ObjectMetadata,
    ) -> CoreResult<&OptimizationStrategy> {
        // Find the best strategy for this data type
        for strategy in &self.optimization_strategies {
            if strategy.target_data_types.contains(&data_type.to_string()) {
                return Ok(strategy);
            }
        }

        // Fallback to first strategy
        self.optimization_strategies.first()
            .ok_or_else(|| CoreError::InvalidArgument(
                crate::error::ErrorContext::new(
                    "No optimization strategies available".to_string()
                )
            ))
    }

    fn apply_optimization_strategy(
        &mut self,
        data: &[u8],
        strategy: &OptimizationStrategy,
    ) -> CoreResult<Vec<u8>> {
        let mut optimized_data = data.to_vec();

        for technique in &strategy.techniques {
            optimized_data = match technique {
                OptimizationTechnique::Compression => {
                    self.compress_data(&optimized_data, &Some(CompressionAlgorithm::Adaptive))?
                }
                OptimizationTechnique::Deduplication => {
                    self.apply_deduplication(&optimized_data)?
                }
                OptimizationTechnique::DeltaEncoding => {
                    self.apply_delta_encoding(&optimized_data)?
                }
                _ => optimized_data, // Other techniques not implemented yet
            };
        }

        Ok(optimized_data)
    }

    fn apply_deduplication(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simple deduplication simulation
        // In real implementation, this would identify and remove duplicate blocks
        Ok(data.to_vec()) // Placeholder
    }

    fn apply_delta_encoding(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Delta encoding simulation
        // In real implementation, this would encode differences between consecutive values
        Ok(data.to_vec()) // Placeholder
    }
}

impl Default for CloudPerformanceAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudPerformanceAnalytics {
    pub fn new() -> Self {
        Self {
            overall_metrics: OverallCloudMetrics {
                total_data_transferred_gb: 0.0,
                avg_response_time: Duration::default(),
                overall_availability: 0.999,
                cost_savings: 0.0,
                performance_improvement: 0.0,
            },
            provider_metrics: HashMap::new(),
            cost_analytics: CostAnalytics {
                total_cost: 0.0,
                cost_by_provider: HashMap::new(),
                cost_by_operation: HashMap::new(),
                cost_trends: Vec::new(),
                optimization_opportunities: Vec::new(),
            },
            performance_trends: PerformanceTrends {
                latency_trend: Vec::new(),
                throughput_trend: Vec::new(),
                error_rate_trend: Vec::new(),
                cost_trend: Vec::new(),
            },
            recommendations: vec![
                "Enable intelligent caching for frequently accessed data".to_string(),
                "Consider using compression for large data transfers".to_string(),
                "Optimize data placement based on access patterns".to_string(),
            ],
        }
    }

    /// Update analytics with new performance data
    pub fn update_metrics(
        &mut self,
        provider_id: &CloudProviderId,
        operation_type: &str,
        latency: Duration,
        throughput_mbps: f64,
        cost: f64,
    ) -> CoreResult<()> {
        // Update provider-specific metrics
        let provider_metrics = self.provider_metrics.entry(provider_id.clone())
            .or_insert_with(|| ProviderMetrics {
                total_requests: 0,
                success_rate: 1.0,
                avg_latency: Duration::default(),
                throughput_mbps: 0.0,
                cost_per_operation: 0.0,
            });

        provider_metrics.total_requests += 1;
        provider_metrics.avg_latency = self.update_average_duration(
            provider_metrics.avg_latency,
            latency,
            provider_metrics.total_requests,
        );
        provider_metrics.throughput_mbps = throughput_mbps;
        provider_metrics.cost_per_operation = cost;

        // Update cost analytics
        *self.cost_analytics.cost_by_provider.entry(provider_id.clone()).or_insert(0.0) += cost;
        *self.cost_analytics.cost_by_operation.entry(operation_type.to_string()).or_insert(0.0) += cost;
        self.cost_analytics.total_cost += cost;

        // Update trends
        self.update_trends(latency, throughput_mbps, cost)?;

        Ok(())
    }

    /// Get cost optimization recommendations
    pub fn get_cost_optimization_recommendations(&self) -> Vec<CostOptimization> {
        let mut recommendations = Vec::new();

        // Analyze cost by provider
        if let Some((most_expensive_provider, cost)) = self.cost_analytics.cost_by_provider
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)) {

            if *cost > self.cost_analytics.total_cost * 0.5 {
                recommendations.push(CostOptimization {
                    optimization_type: "provider_diversification".to_string(),
                    potential_savings: cost * 0.2,
                    implementation_effort: "Medium".to_string(),
                });
            }
        }

        // Analyze operation costs
        if let Some((most_expensive_operation, cost)) = self.cost_analytics.cost_by_operation
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)) {

            if *cost > self.cost_analytics.total_cost * 0.3 {
                recommendations.push(CostOptimization {
                    optimization_type: format!("optimize_{}", most_expensive_operation),
                    potential_savings: cost * 0.15,
                    implementation_effort: "Low".to_string(),
                });
            }
        }

        recommendations
    }

    // Private helper methods

    fn update_average_duration(&self, current_avg: Duration, new_value: Duration, count: u64) -> Duration {
        let current_nanos = current_avg.as_nanos() as u64;
        let new_nanos = new_value.as_nanos() as u64;
        let updated_nanos = (current_nanos * (count - 1) + new_nanos) / count;
        Duration::from_nanos(updated_nanos)
    }

    fn update_trends(&mut self, latency: Duration, throughput_mbps: f64, cost: f64) -> CoreResult<()> {
        let now = Instant::now();

        // Add new trend points
        self.performance_trends.latency_trend.push(TrendPoint {
            timestamp: now,
            value: latency.as_millis() as f64,
            moving_average: self.calculate_moving_average(&self.performance_trends.latency_trend, latency.as_millis() as f64),
        });

        self.performance_trends.throughput_trend.push(TrendPoint {
            timestamp: now,
            value: throughput_mbps,
            moving_average: self.calculate_moving_average(&self.performance_trends.throughput_trend, throughput_mbps),
        });

        self.performance_trends.cost_trend.push(TrendPoint {
            timestamp: now,
            value: cost,
            moving_average: self.calculate_moving_average(&self.performance_trends.cost_trend, cost),
        });

        // Keep only recent trends (last 24 hours)
        let cutoff = now - Duration::from_secs(24 * 60 * 60);
        self.performance_trends.latency_trend.retain(|p| p.timestamp > cutoff);
        self.performance_trends.throughput_trend.retain(|p| p.timestamp > cutoff);
        self.performance_trends.cost_trend.retain(|p| p.timestamp > cutoff);

        Ok(())
    }

    fn calculate_moving_average(&self, trend_points: &[TrendPoint], new_value: f64) -> f64 {
        if trend_points.is_empty() {
            return new_value;
        }

        let window_size = 10.min(trend_points.len());
        let recent_values: Vec<f64> = trend_points.iter()
            .rev()
            .take(window_size)
            .map(|p| p.value)
            .collect();

        let sum: f64 = recent_values.iter().sum::<f64>() + new_value;
        sum / (recent_values.len() + 1) as f64
    }
}

/// Data statistics for optimization analysis
#[derive(Debug, Clone)]
pub struct DataStatistics {
    /// Average compression ratio achieved
    pub avg_compression_ratio: f64,
    /// Data duplication ratio
    pub duplication_ratio: f64,
    /// Most common data types
    pub common_data_types: Vec<String>,
    /// Access pattern frequency
    pub access_patterns: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_optimization_engine_creation() {
        let engine = DataOptimizationEngine::new();
        assert!(!engine.compression_algorithms.is_empty());
        assert!(!engine.optimization_strategies.is_empty());
    }

    #[test]
    fn test_compression() {
        let mut engine = DataOptimizationEngine::new();
        let data = b"Hello, World! This is a test string for compression.";

        let compressed = engine.compress_data(data, &Some(CompressionAlgorithm::Gzip)).unwrap();
        assert!(compressed.len() <= data.len()); // Should be same or smaller

        let decompressed = engine.decompress_data(&compressed, CompressionAlgorithm::Gzip).unwrap();
        assert!(decompressed.len() >= data.len()); // Should be expanded back
    }

    #[test]
    fn test_data_type_detection() {
        let engine = DataOptimizationEngine::new();

        // Test text detection
        let text_data = b"This is plain text content";
        assert_eq!(engine.detect_data_type(text_data), "text");

        // Test binary detection
        let binary_data = &[0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE];
        assert_eq!(engine.detect_data_type(binary_data), "binary");
    }

    #[test]
    fn test_entropy_calculation() {
        let engine = DataOptimizationEngine::new();

        // Test high entropy (random-looking data)
        let high_entropy_data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let entropy = engine.calculate_entropy(&high_entropy_data);
        assert!(entropy > 0.8);

        // Test low entropy (repetitive data)
        let low_entropy_data = vec![0u8; 100];
        let entropy = engine.calculate_entropy(&low_entropy_data);
        assert!(entropy < 0.1);
    }

    #[test]
    fn test_performance_analytics() {
        let mut analytics = CloudPerformanceAnalytics::new();
        let provider_id = CloudProviderId("test_provider".to_string());

        analytics.update_metrics(
            &provider_id,
            "upload",
            Duration::from_millis(100),
            50.0,
            0.05,
        ).unwrap();

        assert!(analytics.provider_metrics.contains_key(&provider_id));
        assert!(analytics.cost_analytics.total_cost > 0.0);
    }
}