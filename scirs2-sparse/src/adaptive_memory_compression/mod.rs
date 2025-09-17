//! Adaptive Memory Compression for Advanced-Large Sparse Matrices
//!
//! This module provides advanced memory management and compression techniques
//! specifically designed for handling advanced-large sparse matrices that exceed
//! available system memory.
//!
//! ## Architecture
//!
//! The adaptive memory compression system consists of several interconnected components:
//!
//! - **Configuration**: Flexible configuration system for different compression strategies
//! - **Cache Management**: Intelligent block caching with LRU eviction and access tracking
//! - **Access Tracking**: Pattern analysis for optimizing compression and caching decisions
//! - **Compression**: Multiple compression algorithms optimized for sparse data
//! - **Out-of-Core Storage**: Seamless handling of matrices larger than available memory
//! - **Memory Mapping**: Efficient file-based storage with memory mapping support
//! - **Statistics**: Comprehensive performance tracking and optimization guidance
//!
//! ## Usage
//!
//! ```rust,ignore
//! use scirs2_sparse::adaptive_memory_compression::{
//!     AdaptiveMemoryCompressor, AdaptiveCompressionConfig, CompressionAlgorithm
//! };
//!
//! // Create configuration
//! let config = AdaptiveCompressionConfig::new()
//!     .with_memory_budget(8 * 1024 * 1024 * 1024) // 8GB
//!     .with_compression_algorithm(CompressionAlgorithm::Adaptive)
//!     .with_out_of_core(true);
//!
//! // Create compressor
//! let mut compressor = AdaptiveMemoryCompressor::new(config)?;
//!
//! // Compress a sparse matrix
//! let compressed = compressor.compress_matrix(
//!     1,          // matrix_id
//!     1000,       // rows
//!     &indptr,    // CSR indptr
//!     &indices,   // CSR indices
//!     &data       // CSR data
//! )?;
//! ```
//!
//! ## Performance Optimization
//!
//! The system automatically learns from access patterns and adapts compression strategies:
//!
//! ```rust,ignore
//! // Get performance statistics
//! let stats = compressor.get_stats();
//! println!("Compression ratio: {:.2}", stats.compression_ratio);
//! println!("Cache hit rate: {:.2}%", stats.cache_hit_ratio * 100.0);
//!
//! // Manual optimization based on access patterns
//! compressor.optimize_for_sequential_access();
//! compressor.optimize_for_random_access();
//! ```

pub mod access_tracking;
pub mod cache;
pub mod compressed_data;
pub mod compression;
pub mod compressor;
pub mod config;
pub mod memory_mapping;
pub mod out_of_core;
pub mod stats;

// Re-export main types for convenience
pub use cache::BlockId;
pub use compressed_data::{BlockType, CompressedBlock, CompressedMatrix};
pub use compressor::AdaptiveMemoryCompressor;
pub use config::{AdaptiveCompressionConfig, CompressionAlgorithm};
pub use stats::{CompressionMetadata, CompressionStats, MemoryStats};

// Re-export key internal types that might be useful
pub use access_tracking::AccessType;
pub use compression::{CompressionEngine, CompressionResult};
pub use memory_mapping::MemoryMappedFile;
pub use out_of_core::OutOfCoreManager;
pub use stats::AccessPatternType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AdaptiveCompressionConfig::new();
        assert_eq!(config.memory_budget, 8 * 1024 * 1024 * 1024);
        assert!(matches!(
            config.compression_algorithm,
            CompressionAlgorithm::Adaptive
        ));
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = AdaptiveCompressionConfig::new()
            .with_memory_budget(4 * 1024 * 1024 * 1024)
            .with_compression_algorithm(CompressionAlgorithm::LZ77)
            .with_out_of_core(false);

        assert_eq!(config.memory_budget, 4 * 1024 * 1024 * 1024);
        assert!(matches!(
            config.compression_algorithm,
            CompressionAlgorithm::LZ77
        ));
        assert!(!config.out_of_core);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AdaptiveCompressionConfig::new();
        assert!(config.validate().is_ok());

        config.memory_budget = 0;
        assert!(config.validate().is_err());

        config.memory_budget = 1024;
        config.compression_threshold = 1.5;
        assert!(config.validate().is_err());

        config.compression_threshold = 0.8;
        config.cache_size = config.memory_budget + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_predefined_configurations() {
        let lightweight = AdaptiveCompressionConfig::lightweight();
        assert_eq!(lightweight.memory_budget, 64 * 1024 * 1024);
        assert!(!lightweight.hierarchical_compression);

        let high_perf = AdaptiveCompressionConfig::high_performance();
        assert_eq!(high_perf.memory_budget, 32 * 1024 * 1024 * 1024);
        assert!(high_perf.hierarchical_compression);

        let mem_efficient = AdaptiveCompressionConfig::memory_efficient();
        assert_eq!(mem_efficient.memory_budget, 1024 * 1024 * 1024);
        assert!(matches!(
            mem_efficient.compression_algorithm,
            CompressionAlgorithm::LZ77
        ));
    }

    #[test]
    fn test_compression_algorithm_properties() {
        assert!(CompressionAlgorithm::Adaptive.supports_adaptive());
        assert!(CompressionAlgorithm::SparseOptimized.supports_adaptive());
        assert!(!CompressionAlgorithm::RLE.supports_adaptive());

        assert_eq!(CompressionAlgorithm::None.expected_compression_ratio(), 1.0);
        assert!(CompressionAlgorithm::Adaptive.expected_compression_ratio() < 0.5);

        assert!(
            CompressionAlgorithm::None.compression_speed()
                > CompressionAlgorithm::Adaptive.compression_speed()
        );
    }

    #[test]
    fn test_block_id_operations() {
        let block_id = BlockId::new(123, 10, 20);
        assert_eq!(block_id.matrixid, 123);
        assert_eq!(block_id.block_row, 10);
        assert_eq!(block_id.block_col, 20);

        let as_u64 = block_id.to_u64();
        let restored = BlockId::from_u64(as_u64);
        assert_eq!(block_id, restored);

        let as_string = block_id.as_string();
        let restored_from_string = BlockId::from_string(&as_string).unwrap();
        assert_eq!(block_id, restored_from_string);
    }

    #[test]
    fn test_compressed_block_creation() {
        let block_id = BlockId::new(1, 0, 0);
        let data = vec![1, 2, 3, 4, 5];
        let original_size = 100;

        let block = CompressedBlock::new(
            block_id.clone(),
            BlockType::Data,
            data.clone(),
            original_size,
            1,
        );

        assert_eq!(block.blockid, block_id);
        assert_eq!(block.block_type, BlockType::Data);
        assert_eq!(block.compressed_data, data);
        assert_eq!(block.original_size, original_size);
        assert!(block.checksum.is_some());
        assert!(block.verify_integrity());
    }

    #[test]
    fn test_compressed_matrix_operations() {
        let mut matrix =
            CompressedMatrix::<f64>::new(1, 1000, 1000, CompressionAlgorithm::RLE, 1024);

        let block_id = BlockId::new(1, 0, 0);
        let block =
            CompressedBlock::new(block_id.clone(), BlockType::Data, vec![1, 2, 3, 4], 100, 1);

        matrix.add_block(block);
        assert_eq!(matrix.block_count(), 1);
        assert!(matrix.get_block(&block_id).is_some());

        let removed = matrix.remove_block(&block_id);
        assert!(removed.is_some());
        assert_eq!(matrix.block_count(), 0);
    }

    #[test]
    fn test_block_type_properties() {
        assert_eq!(BlockType::Data.as_str(), "data");
        assert_eq!(BlockType::from_str("indices").unwrap(), BlockType::Indices);

        assert!(
            BlockType::Data.compression_priority() > BlockType::Metadata.compression_priority()
        );
        assert!(BlockType::Data.benefits_from_compression());
        assert!(!BlockType::IndPtr.benefits_from_compression());
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();

        stats.update_compression(1000, 500, 0.1);
        assert_eq!(stats.total_blocks, 1);
        assert_eq!(stats.compressed_blocks, 1);
        assert_eq!(stats.compression_ratio, 0.5);

        stats.record_cache_hit();
        stats.record_cache_miss();
        assert_eq!(stats.cache_hit_ratio(), 0.5);

        assert_eq!(stats.space_savings(), 500);
        assert_eq!(stats.space_savings_ratio(), 0.5);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new(1024 * 1024, true);

        stats.update_memory_usage(512 * 1024);
        assert_eq!(stats.memory_usage_ratio, 0.5);
        assert!(!stats.has_memory_pressure(0.8));
        assert!(stats.has_memory_pressure(0.3));

        assert_eq!(stats.available_memory(), 512 * 1024);
    }

    #[test]
    fn test_access_pattern_type_display() {
        assert_eq!(AccessPatternType::Sequential.to_string(), "Sequential");
        assert_eq!(AccessPatternType::Random.to_string(), "Random");
        assert_eq!(AccessPatternType::Clustered.to_string(), "Clustered");
        assert_eq!(AccessPatternType::Mixed.to_string(), "Mixed");
        assert_eq!(AccessPatternType::Unknown.to_string(), "Unknown");
    }
}
