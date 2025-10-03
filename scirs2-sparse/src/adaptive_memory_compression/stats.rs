//! Statistics and metadata tracking for adaptive memory compression
//!
//! This module contains structures for tracking compression performance,
//! memory usage, and access patterns to guide optimization decisions.

use super::config::CompressionAlgorithm;

/// Statistics for compression operations
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    pub total_blocks: usize,
    pub compressed_blocks: usize,
    pub total_uncompressed_size: usize,
    pub total_compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time: f64,
    pub decompression_time: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub out_of_core_reads: usize,
    pub out_of_core_writes: usize,
}

/// Compression metadata
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time: f64,
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryStats {
    pub total_memory_budget: usize,
    pub current_memory_usage: usize,
    pub memory_usage_ratio: f64,
    pub compression_stats: CompressionStats,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_ratio: f64,
    pub out_of_core_enabled: bool,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub(crate) struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub hit_ratio: f64,
    pub total_size_bytes: usize,
    pub max_size_bytes: usize,
    pub utilization: f64,
}

/// Compression strategy performance tracking
#[derive(Debug)]
pub(crate) struct CompressionStrategy {
    pub algorithm: CompressionAlgorithm,
    pub block_size: usize,
    pub hierarchical: bool,
    pub predicted_ratio: f64,
    pub actual_ratio: f64,
    pub compression_speed: f64,
    pub decompression_speed: f64,
}

/// Sparsity pattern analysis
#[derive(Debug, Default, Clone)]
pub(crate) struct SparsityPatternAnalysis {
    pub avg_nnz_per_row: f64,
    pub max_nnz_per_row: usize,
    pub min_nnz_per_row: usize,
    pub sequential_patterns: usize,
    pub clustering_factor: f64,
    pub bandwidth: usize,
    pub diagonal_dominance: f64,
    pub fill_ratio: f64,
}

/// Access pattern information
#[derive(Debug, Default, Clone)]
pub(crate) struct AccessPatternInfo {
    pub total_accesses: usize,
    pub avg_temporal_locality: f64,
    pub avg_spatial_locality: f64,
    pub pattern_count: usize,
    pub sequential_access_ratio: f64,
    pub random_access_ratio: f64,
}

/// Performance metrics for different compression algorithms
#[derive(Debug, Clone)]
pub struct AlgorithmPerformanceMetrics {
    pub algorithm: CompressionAlgorithm,
    pub avg_compression_ratio: f64,
    pub avg_compression_time: f64,
    pub avg_decompression_time: f64,
    pub success_rate: f64,
    pub memory_overhead: f64,
    pub sample_count: usize,
}

/// Comprehensive performance summary
#[derive(Debug)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub total_compressed_bytes: usize,
    pub total_uncompressed_bytes: usize,
    pub overall_compression_ratio: f64,
    pub cache_performance: CachePerformance,
    pub algorithm_metrics: Vec<AlgorithmPerformanceMetrics>,
    pub memory_efficiency: MemoryEfficiency,
    pub access_patterns: AccessPatternSummary,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CachePerformance {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f64,
    pub avg_response_time_us: f64,
    pub eviction_count: usize,
    pub prefetch_accuracy: f64,
}

/// Memory efficiency metrics
#[derive(Debug, Clone)]
pub struct MemoryEfficiency {
    pub memory_budget_utilization: f64,
    pub compression_space_savings: f64,
    pub fragmentation_ratio: f64,
    pub out_of_core_usage: f64,
    pub memory_pressure_events: usize,
}

/// Access pattern summary
#[derive(Debug, Clone)]
pub struct AccessPatternSummary {
    pub predominant_pattern: AccessPatternType,
    pub temporal_locality_score: f64,
    pub spatial_locality_score: f64,
    pub predictability_score: f64,
    pub hotspot_concentration: f64,
}

/// Type of access pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessPatternType {
    Sequential,
    Random,
    Clustered,
    Mixed,
    #[default]
    Unknown,
}

impl CompressionStats {
    /// Create new compression statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with a new compression operation
    pub fn update_compression(
        &mut self,
        original_size: usize,
        compressed_size: usize,
        compression_time: f64,
    ) {
        self.total_blocks += 1;
        self.compressed_blocks += 1;
        self.total_uncompressed_size += original_size;
        self.total_compressed_size += compressed_size;
        self.compression_time += compression_time;

        // Update compression ratio
        self.compression_ratio = if self.total_uncompressed_size > 0 {
            self.total_compressed_size as f64 / self.total_uncompressed_size as f64
        } else {
            1.0
        };
    }

    /// Update decompression statistics
    pub fn update_decompression(&mut self, decompression_time: f64) {
        self.decompression_time += decompression_time;
    }

    /// Record cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Record out-of-core read
    pub fn record_out_of_core_read(&mut self) {
        self.out_of_core_reads += 1;
    }

    /// Record out-of-core write
    pub fn record_out_of_core_write(&mut self) {
        self.out_of_core_writes += 1;
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses > 0 {
            self.cache_hits as f64 / total_accesses as f64
        } else {
            0.0
        }
    }

    /// Get average compression time per block
    pub fn avg_compression_time(&self) -> f64 {
        if self.compressed_blocks > 0 {
            self.compression_time / self.compressed_blocks as f64
        } else {
            0.0
        }
    }

    /// Get average decompression time per block
    pub fn avg_decompression_time(&self) -> f64 {
        if self.compressed_blocks > 0 {
            self.decompression_time / self.compressed_blocks as f64
        } else {
            0.0
        }
    }

    /// Get compression efficiency (inverse of time)
    pub fn compression_efficiency(&self) -> f64 {
        let avg_time = self.avg_compression_time();
        if avg_time > 0.0 {
            1.0 / avg_time
        } else {
            0.0
        }
    }

    /// Get space savings in bytes
    pub fn space_savings(&self) -> usize {
        self.total_uncompressed_size
            .saturating_sub(self.total_compressed_size)
    }

    /// Get space savings ratio
    pub fn space_savings_ratio(&self) -> f64 {
        if self.total_uncompressed_size > 0 {
            self.space_savings() as f64 / self.total_uncompressed_size as f64
        } else {
            0.0
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Merge with another stats instance
    pub fn merge(&mut self, other: &CompressionStats) {
        self.total_blocks += other.total_blocks;
        self.compressed_blocks += other.compressed_blocks;
        self.total_uncompressed_size += other.total_uncompressed_size;
        self.total_compressed_size += other.total_compressed_size;
        self.compression_time += other.compression_time;
        self.decompression_time += other.decompression_time;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.out_of_core_reads += other.out_of_core_reads;
        self.out_of_core_writes += other.out_of_core_writes;

        // Recalculate compression ratio
        self.compression_ratio = if self.total_uncompressed_size > 0 {
            self.total_compressed_size as f64 / self.total_uncompressed_size as f64
        } else {
            1.0
        };
    }
}

impl CompressionMetadata {
    /// Create new compression metadata
    pub fn new(original_size: usize, compressed_size: usize, compression_time: f64) -> Self {
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            compression_time,
        }
    }

    /// Get space savings in bytes
    pub fn space_savings(&self) -> usize {
        self.original_size.saturating_sub(self.compressed_size)
    }

    /// Get space savings ratio
    pub fn space_savings_ratio(&self) -> f64 {
        if self.original_size > 0 {
            self.space_savings() as f64 / self.original_size as f64
        } else {
            0.0
        }
    }

    /// Get compression speed (bytes per second)
    pub fn compression_speed(&self) -> f64 {
        if self.compression_time > 0.0 {
            self.original_size as f64 / self.compression_time
        } else {
            0.0
        }
    }

    /// Check if compression was effective
    pub fn is_effective(&self, threshold: f64) -> bool {
        self.compression_ratio < threshold
    }
}

impl MemoryStats {
    /// Create new memory statistics
    pub fn new(memory_budget: usize, out_of_core_enabled: bool) -> Self {
        Self {
            total_memory_budget: memory_budget,
            current_memory_usage: 0,
            memory_usage_ratio: 0.0,
            compression_stats: CompressionStats::default(),
            cache_hits: 0,
            cache_misses: 0,
            cache_hit_ratio: 0.0,
            out_of_core_enabled,
        }
    }

    /// Update memory usage
    pub fn update_memory_usage(&mut self, current_usage: usize) {
        self.current_memory_usage = current_usage;
        self.memory_usage_ratio = if self.total_memory_budget > 0 {
            current_usage as f64 / self.total_memory_budget as f64
        } else {
            0.0
        };
    }

    /// Update cache statistics
    pub fn update_cache_stats(&mut self, hits: usize, misses: usize) {
        self.cache_hits = hits;
        self.cache_misses = misses;
        let total = hits + misses;
        self.cache_hit_ratio = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
    }

    /// Check if memory pressure exists
    pub fn has_memory_pressure(&self, threshold: f64) -> bool {
        self.memory_usage_ratio > threshold
    }

    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.total_memory_budget
            .saturating_sub(self.current_memory_usage)
    }

    /// Get memory efficiency score
    pub fn memory_efficiency_score(&self) -> f64 {
        let compression_benefit = self.compression_stats.space_savings_ratio();
        let cache_benefit = self.cache_hit_ratio;
        let usage_efficiency = 1.0 - self.memory_usage_ratio.min(1.0);

        (compression_benefit + cache_benefit + usage_efficiency) / 3.0
    }
}

impl SparsityPatternAnalysis {
    /// Create a new sparsity pattern analysis
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze sparsity pattern from matrix data
    pub fn analyze_pattern(
        &mut self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
    ) {
        let nnz = indices.len();
        self.fill_ratio = nnz as f64 / (rows * cols) as f64;

        // Calculate per-row statistics
        let mut nnz_per_row = Vec::new();
        for i in 0..rows {
            if i + 1 < indptr.len() {
                let row_nnz = indptr[i + 1] - indptr[i];
                nnz_per_row.push(row_nnz);
            }
        }

        if !nnz_per_row.is_empty() {
            self.avg_nnz_per_row =
                nnz_per_row.iter().sum::<usize>() as f64 / nnz_per_row.len() as f64;
            self.max_nnz_per_row = *nnz_per_row.iter().max().unwrap();
            self.min_nnz_per_row = *nnz_per_row.iter().min().unwrap();
        }

        // Calculate bandwidth (simplified)
        if !indices.is_empty() {
            let min_col = *indices.iter().min().unwrap();
            let max_col = *indices.iter().max().unwrap();
            self.bandwidth = max_col - min_col + 1;
        }

        // Analyze sequential patterns
        self.sequential_patterns = self.count_sequential_patterns(indptr, indices);

        // Calculate clustering factor (simplified)
        self.clustering_factor = self.calculate_clustering_factor(indptr, indices);

        // Calculate diagonal dominance
        self.diagonal_dominance = self.calculate_diagonal_dominance(rows, indptr, indices);
    }

    /// Count sequential access patterns
    fn count_sequential_patterns(&self, indptr: &[usize], indices: &[usize]) -> usize {
        let mut sequential_count = 0;

        for i in 0..indptr.len().saturating_sub(1) {
            let start = indptr[i];
            let end = indptr[i + 1];

            if end > start + 1 {
                let row_indices = &indices[start..end];
                let mut is_sequential = true;

                for j in 1..row_indices.len() {
                    if row_indices[j] != row_indices[j - 1] + 1 {
                        is_sequential = false;
                        break;
                    }
                }

                if is_sequential {
                    sequential_count += 1;
                }
            }
        }

        sequential_count
    }

    /// Calculate clustering factor
    fn calculate_clustering_factor(&self, indptr: &[usize], indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut cluster_score = 0.0;
        let mut total_comparisons = 0;

        for i in 0..indptr.len().saturating_sub(1) {
            let start = indptr[i];
            let end = indptr[i + 1];

            if end > start + 1 {
                let row_indices = &indices[start..end];
                for j in 1..row_indices.len() {
                    let distance = row_indices[j].saturating_sub(row_indices[j - 1]);
                    if distance <= 5 {
                        // Consider elements within 5 columns as clustered
                        cluster_score += 1.0;
                    }
                    total_comparisons += 1;
                }
            }
        }

        if total_comparisons > 0 {
            cluster_score / total_comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate diagonal dominance
    fn calculate_diagonal_dominance(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
    ) -> f64 {
        let mut diagonal_elements = 0;

        for i in 0..rows.min(indptr.len().saturating_sub(1)) {
            let start = indptr[i];
            let end = indptr[i + 1];

            for j in start..end {
                if j < indices.len() && indices[j] == i {
                    diagonal_elements += 1;
                    break;
                }
            }
        }

        diagonal_elements as f64 / rows as f64
    }

    /// Get pattern classification
    pub fn pattern_type(&self) -> AccessPatternType {
        if self.sequential_patterns as f64 / (self.avg_nnz_per_row + 1.0) > 0.7 {
            AccessPatternType::Sequential
        } else if self.clustering_factor > 0.6 {
            AccessPatternType::Clustered
        } else if self.diagonal_dominance > 0.8 {
            AccessPatternType::Sequential // Diagonal patterns are often sequential
        } else {
            AccessPatternType::Mixed
        }
    }
}

impl AccessPatternInfo {
    /// Create new access pattern info
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with new access information
    pub fn update(&mut self, temporal_locality: f64, spatial_locality: f64, is_sequential: bool) {
        self.total_accesses += 1;

        // Update running averages
        let n = self.total_accesses as f64;
        self.avg_temporal_locality =
            ((n - 1.0) * self.avg_temporal_locality + temporal_locality) / n;
        self.avg_spatial_locality = ((n - 1.0) * self.avg_spatial_locality + spatial_locality) / n;

        if is_sequential {
            self.sequential_access_ratio = ((n - 1.0) * self.sequential_access_ratio + 1.0) / n;
        } else {
            self.sequential_access_ratio = ((n - 1.0) * self.sequential_access_ratio) / n;
            self.random_access_ratio = 1.0 - self.sequential_access_ratio;
        }

        self.pattern_count += 1;
    }

    /// Get overall locality score
    pub fn locality_score(&self) -> f64 {
        (self.avg_temporal_locality + self.avg_spatial_locality) / 2.0
    }

    /// Determine predominant access pattern
    pub fn predominant_pattern(&self) -> AccessPatternType {
        if self.sequential_access_ratio > 0.7 {
            AccessPatternType::Sequential
        } else if self.random_access_ratio > 0.7 {
            AccessPatternType::Random
        } else if self.locality_score() > 0.6 {
            AccessPatternType::Clustered
        } else {
            AccessPatternType::Mixed
        }
    }
}

impl std::fmt::Display for AccessPatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessPatternType::Sequential => write!(f, "Sequential"),
            AccessPatternType::Random => write!(f, "Random"),
            AccessPatternType::Clustered => write!(f, "Clustered"),
            AccessPatternType::Mixed => write!(f, "Mixed"),
            AccessPatternType::Unknown => write!(f, "Unknown"),
        }
    }
}
