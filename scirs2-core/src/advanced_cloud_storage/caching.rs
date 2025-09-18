//! Intelligent caching and tiering system
//!
//! This module provides multi-tier caching with predictive prefetching,
//! intelligent eviction policies, and cache analytics.

use crate::error::{CoreError, CoreResult};
use super::types::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Intelligent cache system
#[derive(Debug)]
pub struct IntelligentCacheSystem {
    /// Cache layers
    cache_layers: Vec<CacheLayer>,
    /// Cache policies
    policies: CachePolicies,
    /// Performance analytics
    analytics: CacheAnalytics,
    /// Eviction manager
    eviction_manager: EvictionManager,
}

/// Cache layer
#[derive(Debug)]
pub struct CacheLayer {
    /// Layer ID
    pub id: String,
    /// Layer type
    pub layer_type: CacheLayerType,
    /// Capacity (MB)
    pub capacity_mb: usize,
    /// Current usage (MB)
    pub current_usage_mb: usize,
    /// Cache entries
    pub entries: HashMap<String, CacheEntry>,
    /// Performance metrics
    pub metrics: CacheLayerMetrics,
}

/// Cache layer types
#[derive(Debug, Clone, PartialEq)]
pub enum CacheLayerType {
    Memory,
    SSD,
    HDD,
    Network,
    CDN,
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Object key
    pub key: String,
    /// Data
    pub data: Vec<u8>,
    /// Metadata
    pub metadata: ObjectMetadata,
    /// Access count
    pub access_count: u64,
    /// Last accessed
    pub last_accessed: Instant,
    /// Created
    pub created: Instant,
    /// TTL
    pub ttl: Option<Duration>,
    /// Size
    pub size: usize,
}

/// Cache layer metrics
#[derive(Debug, Clone)]
pub struct CacheLayerMetrics {
    /// Hit rate
    pub hit_rate: f64,
    /// Miss rate
    pub miss_rate: f64,
    /// Eviction rate
    pub eviction_rate: f64,
    /// Average access time
    pub avg_access_time: Duration,
    /// Storage efficiency
    pub storage_efficiency: f64,
}

/// Cache policies
#[derive(Debug)]
pub struct CachePolicies {
    /// Insertion policy
    pub insertion_policy: InsertionPolicy,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Coherence policy
    pub coherence_policy: CoherencePolicy,
    /// TTL policy
    pub ttl_policy: TTLPolicy,
}

/// Insertion policies
#[derive(Debug, Clone)]
pub enum InsertionPolicy {
    Always,
    OnDemand,
    Predictive,
    SizeBased,
    FrequencyBased,
}

/// Eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    TTL,
    Adaptive,
}

/// Coherence policies
#[derive(Debug, Clone)]
pub enum CoherencePolicy {
    WriteThrough,
    WriteBack,
    WriteInvalidate,
    NoWrite,
}

/// TTL policies
#[derive(Debug, Clone)]
pub struct TTLPolicy {
    /// Default TTL
    pub default_ttl: Duration,
    /// Max TTL
    pub max_ttl: Duration,
    /// TTL strategy
    pub strategy: TTLStrategy,
}

/// TTL strategies
#[derive(Debug, Clone)]
pub enum TTLStrategy {
    Fixed,
    Sliding,
    Adaptive,
    AccessBased,
}

/// Cache analytics
#[derive(Debug)]
pub struct CacheAnalytics {
    /// Overall metrics
    overall_metrics: OverallCacheMetrics,
    /// Per-layer metrics
    layer_metrics: HashMap<String, CacheLayerMetrics>,
    /// Trends
    trends: CacheTrends,
    /// Recommendations
    recommendations: Vec<CacheRecommendation>,
}

/// Overall cache metrics
#[derive(Debug, Clone)]
pub struct OverallCacheMetrics {
    /// Total hit rate
    pub total_hit_rate: f64,
    /// Total storage used (MB)
    pub total_storage_mb: f64,
    /// Average access time
    pub avg_access_time: Duration,
    /// Cost savings
    pub cost_savings: f64,
    /// Bandwidth savings
    pub bandwidth_savings: f64,
}

/// Cache trends
#[derive(Debug)]
pub struct CacheTrends {
    /// Hit rate trend
    pub hit_rate_trend: Vec<TrendPoint>,
    /// Storage utilization trend
    pub storage_trend: Vec<TrendPoint>,
    /// Access pattern trend
    pub access_pattern_trend: Vec<TrendPoint>,
}

/// Trend point
#[derive(Debug, Clone)]
pub struct TrendPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Value
    pub value: f64,
    /// Moving average
    pub moving_average: f64,
}

/// Cache recommendation
#[derive(Debug, Clone)]
pub struct CacheRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description
    pub description: String,
    /// Potential impact
    pub potential_impact: f64,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
}

/// Recommendation types
#[derive(Debug, Clone)]
pub enum RecommendationType {
    IncreaseCapacity,
    ChangeEvictionPolicy,
    AdjustTTL,
    OptimizePlacement,
    AddCacheLayer,
}

/// Eviction manager
#[derive(Debug)]
pub struct EvictionManager {
    /// Active eviction algorithms
    algorithms: Vec<EvictionAlgorithm>,
    /// Eviction statistics
    statistics: EvictionStatistics,
    /// Predictive eviction
    predictive_eviction: PredictiveEviction,
}

/// Eviction algorithm
#[derive(Debug)]
pub struct EvictionAlgorithm {
    /// Algorithm type
    pub algorithm_type: EvictionAlgorithmType,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Performance metrics
    pub performance: EvictionPerformance,
}

/// Eviction algorithm types
#[derive(Debug, Clone)]
pub enum EvictionAlgorithmType {
    LRU,
    LFU,
    ARC,
    SLRU,
    TinyLFU,
    Clock,
    AdaptiveReplacement,
}

/// Eviction performance
#[derive(Debug, Clone)]
pub struct EvictionPerformance {
    /// Eviction accuracy
    pub accuracy: f64,
    /// Eviction latency
    pub latency: Duration,
    /// Memory efficiency
    pub memory_efficiency: f64,
    /// Hit rate after eviction
    pub post_eviction_hit_rate: f64,
}

/// Eviction statistics
#[derive(Debug, Clone)]
pub struct EvictionStatistics {
    /// Total evictions
    pub total_evictions: u64,
    /// Evictions by algorithm
    pub evictions_by_algorithm: HashMap<String, u64>,
    /// Average eviction time
    pub avg_eviction_time: Duration,
    /// Eviction success rate
    pub success_rate: f64,
}

/// Predictive eviction
#[derive(Debug)]
pub struct PredictiveEviction {
    /// Prediction models
    models: HashMap<String, EvictionPredictionModel>,
    /// Training data
    training_data: Vec<EvictionTrainingData>,
    /// Prediction accuracy
    accuracy: ModelAccuracy,
}

/// Eviction prediction model
#[derive(Debug)]
pub struct EvictionPredictionModel {
    /// Model type
    pub model_type: PredictionModelType,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Feature weights
    pub feature_weights: Vec<f64>,
    /// Last training time
    pub last_training: Instant,
}

/// Eviction training data
#[derive(Debug, Clone)]
pub struct EvictionTrainingData {
    /// Object features
    pub features: Vec<f64>,
    /// Was accessed after potential eviction
    pub was_accessed: bool,
    /// Time until next access
    pub time_to_access: Option<Duration>,
    /// Training timestamp
    pub timestamp: Instant,
}

// Implementations

impl Default for IntelligentCacheSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelligentCacheSystem {
    pub fn new() -> Self {
        Self {
            cache_layers: vec![CacheLayer {
                id: "memory".to_string(),
                layer_type: CacheLayerType::Memory,
                capacity_mb: 1024,
                current_usage_mb: 0,
                entries: HashMap::new(),
                metrics: CacheLayerMetrics {
                    hit_rate: 0.0,
                    miss_rate: 0.0,
                    eviction_rate: 0.0,
                    avg_access_time: Duration::from_micros(10),
                    storage_efficiency: 0.0,
                },
            }],
            policies: CachePolicies {
                insertion_policy: InsertionPolicy::OnDemand,
                eviction_policy: EvictionPolicy::LRU,
                coherence_policy: CoherencePolicy::WriteThrough,
                ttl_policy: TTLPolicy {
                    default_ttl: Duration::from_secs(3600),
                    max_ttl: Duration::from_secs(86400),
                    strategy: TTLStrategy::Sliding,
                },
            },
            analytics: CacheAnalytics {
                overall_metrics: OverallCacheMetrics {
                    total_hit_rate: 0.0,
                    total_storage_mb: 0.0,
                    avg_access_time: Duration::default(),
                    cost_savings: 0.0,
                    bandwidth_savings: 0.0,
                },
                layer_metrics: HashMap::new(),
                trends: CacheTrends {
                    hit_rate_trend: Vec::new(),
                    storage_trend: Vec::new(),
                    access_pattern_trend: Vec::new(),
                },
                recommendations: Vec::new(),
            },
            eviction_manager: EvictionManager {
                algorithms: vec![EvictionAlgorithm {
                    algorithm_type: EvictionAlgorithmType::LRU,
                    parameters: HashMap::new(),
                    performance: EvictionPerformance {
                        accuracy: 0.0,
                        latency: Duration::default(),
                        memory_efficiency: 0.0,
                        post_eviction_hit_rate: 0.0,
                    },
                }],
                statistics: EvictionStatistics {
                    total_evictions: 0,
                    evictions_by_algorithm: HashMap::new(),
                    avg_eviction_time: Duration::default(),
                    success_rate: 0.0,
                },
                predictive_eviction: PredictiveEviction {
                    models: HashMap::new(),
                    training_data: Vec::new(),
                    accuracy: ModelAccuracy {
                        mae: 0.0,
                        rmse: 0.0,
                        r_squared: 0.0,
                        confidence: 0.0,
                    },
                },
            },
        }
    }

    /// Get cached data
    pub fn get(&self, key: &str) -> CoreResult<Option<Vec<u8>>> {
        // Check each cache layer in order
        for layer in &self.cache_layers {
            if let Some(entry) = layer.entries.get(key) {
                // Check if entry is still valid (TTL)
                if self.is_entry_valid(entry) {
                    return Ok(Some(entry.data.clone()));
                }
            }
        }
        Ok(None)
    }

    /// Store data in cache
    pub fn put(&mut self, key: String, data: Vec<u8>, metadata: ObjectMetadata) -> CoreResult<()> {
        let entry = CacheEntry {
            key: key.clone(),
            data: data.clone(),
            metadata,
            access_count: 1,
            last_accessed: Instant::now(),
            created: Instant::now(),
            ttl: Some(self.policies.ttl_policy.default_ttl),
            size: data.len(),
        };

        // Find appropriate layer based on policies
        let layer_index = self.select_cache_layer(&entry)?;

        // Check if we need to evict entries first
        if self.needs_eviction(layer_index, entry.size) {
            self.evict_entries(layer_index, entry.size)?;
        }

        // Insert into selected layer
        if let Some(layer) = self.cache_layers.get_mut(layer_index) {
            layer.entries.insert(key, entry);
            layer.current_usage_mb += data.len() / (1024 * 1024);
        }

        Ok(())
    }

    /// Remove data from cache
    pub fn remove(&mut self, key: &str) -> CoreResult<bool> {
        for layer in &mut self.cache_layers {
            if let Some(entry) = layer.entries.remove(key) {
                layer.current_usage_mb = layer.current_usage_mb.saturating_sub(entry.size / (1024 * 1024));
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Update cache analytics
    pub fn update_analytics(&mut self) -> CoreResult<()> {
        self.calculate_overall_metrics()?;
        self.update_trends()?;
        self.generate_recommendations()?;
        Ok(())
    }

    /// Add a new cache layer
    pub fn add_cache_layer(&mut self, layer_type: CacheLayerType, capacity_mb: usize) -> CoreResult<()> {
        let layer_id = format!("{:?}_{}", layer_type, self.cache_layers.len());

        let layer = CacheLayer {
            id: layer_id,
            layer_type,
            capacity_mb,
            current_usage_mb: 0,
            entries: HashMap::new(),
            metrics: CacheLayerMetrics {
                hit_rate: 0.0,
                miss_rate: 0.0,
                eviction_rate: 0.0,
                avg_access_time: self.get_layer_access_time(&layer_type),
                storage_efficiency: 0.0,
            },
        };

        self.cache_layers.push(layer);
        Ok(())
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheAnalytics {
        // Return a clone of current analytics
        CacheAnalytics {
            overall_metrics: self.analytics.overall_metrics.clone(),
            layer_metrics: self.analytics.layer_metrics.clone(),
            trends: CacheTrends {
                hit_rate_trend: self.analytics.trends.hit_rate_trend.clone(),
                storage_trend: self.analytics.trends.storage_trend.clone(),
                access_pattern_trend: self.analytics.trends.access_pattern_trend.clone(),
            },
            recommendations: self.analytics.recommendations.clone(),
        }
    }

    // Private helper methods

    fn is_entry_valid(&self, entry: &CacheEntry) -> bool {
        if let Some(ttl) = entry.ttl {
            entry.created.elapsed() < ttl
        } else {
            true // No TTL means always valid
        }
    }

    fn select_cache_layer(&self, entry: &CacheEntry) -> CoreResult<usize> {
        // Simple policy: use first layer with available space
        for (index, layer) in self.cache_layers.iter().enumerate() {
            let required_mb = entry.size / (1024 * 1024);
            if layer.current_usage_mb + required_mb <= layer.capacity_mb {
                return Ok(index);
            }
        }

        // If no layer has space, use the first layer (will trigger eviction)
        Ok(0)
    }

    fn needs_eviction(&self, layer_index: usize, entry_size: usize) -> bool {
        if let Some(layer) = self.cache_layers.get(layer_index) {
            let required_mb = entry_size / (1024 * 1024);
            layer.current_usage_mb + required_mb > layer.capacity_mb
        } else {
            false
        }
    }

    fn evict_entries(&mut self, layer_index: usize, required_size: usize) -> CoreResult<()> {
        let required_mb = required_size / (1024 * 1024);

        if let Some(layer) = self.cache_layers.get_mut(layer_index) {
            // Simple LRU eviction
            let mut entries_to_evict = Vec::new();

            // Sort entries by last accessed time
            let mut sorted_entries: Vec<_> = layer.entries.iter().collect();
            sorted_entries.sort_by_key(|(_, entry)| entry.last_accessed);

            let mut freed_mb = 0;
            for (key, entry) in sorted_entries {
                if freed_mb >= required_mb {
                    break;
                }

                entries_to_evict.push(key.clone());
                freed_mb += entry.size / (1024 * 1024);
            }

            // Remove selected entries
            for key in entries_to_evict {
                if let Some(entry) = layer.entries.remove(&key) {
                    layer.current_usage_mb = layer.current_usage_mb.saturating_sub(entry.size / (1024 * 1024));
                }
            }
        }

        Ok(())
    }

    fn calculate_overall_metrics(&mut self) -> CoreResult<()> {
        let mut total_hits = 0.0;
        let mut total_requests = 0.0;
        let mut total_storage = 0.0;

        for layer in &self.cache_layers {
            total_hits += layer.metrics.hit_rate * 100.0; // Assuming some base requests
            total_requests += 100.0;
            total_storage += layer.current_usage_mb as f64;
        }

        self.analytics.overall_metrics.total_hit_rate = if total_requests > 0.0 {
            total_hits / total_requests
        } else {
            0.0
        };

        self.analytics.overall_metrics.total_storage_mb = total_storage;

        Ok(())
    }

    fn update_trends(&mut self) -> CoreResult<()> {
        let now = Instant::now();
        let hit_rate = self.analytics.overall_metrics.total_hit_rate;

        // Add new trend point
        let trend_point = TrendPoint {
            timestamp: now,
            value: hit_rate,
            moving_average: self.calculate_moving_average(&self.analytics.trends.hit_rate_trend, hit_rate),
        };

        self.analytics.trends.hit_rate_trend.push(trend_point);

        // Keep only recent trends (last 24 hours worth)
        let cutoff = now - Duration::from_secs(24 * 60 * 60);
        self.analytics.trends.hit_rate_trend.retain(|point| point.timestamp > cutoff);

        Ok(())
    }

    fn generate_recommendations(&mut self) -> CoreResult<()> {
        self.analytics.recommendations.clear();

        // Check if hit rate is low
        if self.analytics.overall_metrics.total_hit_rate < 0.5 {
            self.analytics.recommendations.push(CacheRecommendation {
                recommendation_type: RecommendationType::IncreaseCapacity,
                description: "Consider increasing cache capacity to improve hit rate".to_string(),
                potential_impact: 0.3,
                complexity: ComplexityLevel::Low,
            });
        }

        // Check if storage is highly utilized
        let total_capacity: usize = self.cache_layers.iter().map(|l| l.capacity_mb).sum();
        let utilization = self.analytics.overall_metrics.total_storage_mb / total_capacity as f64;

        if utilization > 0.8 {
            self.analytics.recommendations.push(CacheRecommendation {
                recommendation_type: RecommendationType::AddCacheLayer,
                description: "Add an additional cache layer to reduce pressure".to_string(),
                potential_impact: 0.25,
                complexity: ComplexityLevel::Medium,
            });
        }

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

    fn get_layer_access_time(&self, layer_type: &CacheLayerType) -> Duration {
        match layer_type {
            CacheLayerType::Memory => Duration::from_micros(10),
            CacheLayerType::SSD => Duration::from_micros(100),
            CacheLayerType::HDD => Duration::from_millis(10),
            CacheLayerType::Network => Duration::from_millis(50),
            CacheLayerType::CDN => Duration::from_millis(100),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intelligent_cache_system_creation() {
        let cache = IntelligentCacheSystem::new();
        assert_eq!(cache.cache_layers.len(), 1);
        assert_eq!(cache.cache_layers[0].layer_type, CacheLayerType::Memory);
    }

    #[test]
    fn test_cache_put_and_get() {
        let mut cache = IntelligentCacheSystem::new();

        let data = vec![1, 2, 3, 4, 5];
        let metadata = ObjectMetadata {
            key: "test_key".to_string(),
            size: data.len() as u64,
            content_type: None,
            last_modified: None,
            etag: None,
            metadata: HashMap::new(),
            storage_class: None,
            encryption: None,
        };

        // Put data in cache
        cache.put("test_key".to_string(), data.clone(), metadata).unwrap();

        // Get data from cache
        let cached_data = cache.get("test_key").unwrap();
        assert_eq!(cached_data, Some(data));
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = IntelligentCacheSystem::new();

        let data = vec![1, 2, 3, 4, 5];
        let metadata = ObjectMetadata {
            key: "test_key".to_string(),
            size: data.len() as u64,
            content_type: None,
            last_modified: None,
            etag: None,
            metadata: HashMap::new(),
            storage_class: None,
            encryption: None,
        };

        cache.put("test_key".to_string(), data, metadata).unwrap();

        let removed = cache.remove("test_key").unwrap();
        assert!(removed);

        let cached_data = cache.get("test_key").unwrap();
        assert_eq!(cached_data, None);
    }

    #[test]
    fn test_add_cache_layer() {
        let mut cache = IntelligentCacheSystem::new();
        let initial_layers = cache.cache_layers.len();

        cache.add_cache_layer(CacheLayerType::SSD, 2048).unwrap();

        assert_eq!(cache.cache_layers.len(), initial_layers + 1);
        assert_eq!(cache.cache_layers.last().unwrap().layer_type, CacheLayerType::SSD);
        assert_eq!(cache.cache_layers.last().unwrap().capacity_mb, 2048);
    }
}