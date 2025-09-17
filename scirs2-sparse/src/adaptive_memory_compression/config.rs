//! Configuration for adaptive memory compression
//!
//! This module contains configuration structures and enums that control
//! the behavior of the adaptive memory compression system.

/// Configuration for adaptive memory compression
#[derive(Debug, Clone)]
pub struct AdaptiveCompressionConfig {
    /// Maximum memory budget in bytes
    pub memory_budget: usize,
    /// Compression algorithm to use
    pub compression_algorithm: CompressionAlgorithm,
    /// Enable hierarchical compression
    pub hierarchical_compression: bool,
    /// Block size for compression
    pub block_size: usize,
    /// Compression threshold (compress when usage exceeds this ratio)
    pub compression_threshold: f64,
    /// Enable adaptive compression based on access patterns
    pub adaptive_compression: bool,
    /// Cache size for frequently accessed blocks
    pub cache_size: usize,
    /// Enable out-of-core processing
    pub out_of_core: bool,
    /// Temporary directory for out-of-core storage
    pub temp_directory: String,
    /// Enable memory mapping
    pub memory_mapping: bool,
}

/// Compression algorithms for sparse matrix data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Run-Length Encoding
    RLE,
    /// Delta encoding for indices
    Delta,
    /// Huffman coding
    Huffman,
    /// LZ77 compression
    LZ77,
    /// Sparse-specific compression
    SparseOptimized,
    /// Hybrid adaptive compression
    Adaptive,
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            memory_budget: 8 * 1024 * 1024 * 1024, // 8GB default
            compression_algorithm: CompressionAlgorithm::Adaptive,
            hierarchical_compression: true,
            block_size: 1024 * 1024, // 1MB blocks
            compression_threshold: 0.8,
            adaptive_compression: true,
            cache_size: 256 * 1024 * 1024, // 256MB cache
            out_of_core: true,
            temp_directory: "/tmp/scirs2_sparse".to_string(),
            memory_mapping: true,
        }
    }
}

impl AdaptiveCompressionConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set memory budget
    pub fn with_memory_budget(mut self, budget: usize) -> Self {
        self.memory_budget = budget;
        self
    }

    /// Set compression algorithm
    pub fn with_compression_algorithm(mut self, algorithm: CompressionAlgorithm) -> Self {
        self.compression_algorithm = algorithm;
        self
    }

    /// Enable or disable hierarchical compression
    pub fn with_hierarchical_compression(mut self, enabled: bool) -> Self {
        self.hierarchical_compression = enabled;
        self
    }

    /// Set block size for compression
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set compression threshold
    pub fn with_compression_threshold(mut self, threshold: f64) -> Self {
        self.compression_threshold = threshold;
        self
    }

    /// Enable or disable adaptive compression
    pub fn with_adaptive_compression(mut self, enabled: bool) -> Self {
        self.adaptive_compression = enabled;
        self
    }

    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Enable or disable out-of-core processing
    pub fn with_out_of_core(mut self, enabled: bool) -> Self {
        self.out_of_core = enabled;
        self
    }

    /// Set temporary directory for out-of-core storage
    pub fn with_temp_directory(mut self, dir: impl Into<String>) -> Self {
        self.temp_directory = dir.into();
        self
    }

    /// Enable or disable memory mapping
    pub fn with_memory_mapping(mut self, enabled: bool) -> Self {
        self.memory_mapping = enabled;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.memory_budget == 0 {
            return Err("Memory budget must be greater than 0".to_string());
        }

        if self.block_size == 0 {
            return Err("Block size must be greater than 0".to_string());
        }

        if self.compression_threshold < 0.0 || self.compression_threshold > 1.0 {
            return Err("Compression threshold must be between 0.0 and 1.0".to_string());
        }

        if self.cache_size == 0 {
            return Err("Cache size must be greater than 0".to_string());
        }

        if self.cache_size > self.memory_budget {
            return Err("Cache size cannot exceed memory budget".to_string());
        }

        if self.temp_directory.is_empty() {
            return Err("Temporary directory must be specified".to_string());
        }

        Ok(())
    }

    /// Create a lightweight configuration for testing
    pub fn lightweight() -> Self {
        Self {
            memory_budget: 64 * 1024 * 1024, // 64MB
            compression_algorithm: CompressionAlgorithm::RLE,
            hierarchical_compression: false,
            block_size: 64 * 1024, // 64KB blocks
            compression_threshold: 0.9,
            adaptive_compression: false,
            cache_size: 16 * 1024 * 1024, // 16MB cache
            out_of_core: false,
            temp_directory: "/tmp/scirs2_test".to_string(),
            memory_mapping: false,
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            memory_budget: 32 * 1024 * 1024 * 1024, // 32GB
            compression_algorithm: CompressionAlgorithm::Adaptive,
            hierarchical_compression: true,
            block_size: 4 * 1024 * 1024, // 4MB blocks
            compression_threshold: 0.7,
            adaptive_compression: true,
            cache_size: 2 * 1024 * 1024 * 1024, // 2GB cache
            out_of_core: true,
            temp_directory: "/tmp/scirs2_hiperf".to_string(),
            memory_mapping: true,
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            memory_budget: 1024 * 1024 * 1024, // 1GB
            compression_algorithm: CompressionAlgorithm::LZ77,
            hierarchical_compression: true,
            block_size: 256 * 1024, // 256KB blocks
            compression_threshold: 0.5,
            adaptive_compression: true,
            cache_size: 128 * 1024 * 1024, // 128MB cache
            out_of_core: true,
            temp_directory: "/tmp/scirs2_memeff".to_string(),
            memory_mapping: true,
        }
    }
}

impl CompressionAlgorithm {
    /// Check if the algorithm supports adaptive compression
    pub fn supports_adaptive(&self) -> bool {
        matches!(
            self,
            CompressionAlgorithm::Adaptive | CompressionAlgorithm::SparseOptimized
        )
    }

    /// Get expected compression ratio for the algorithm
    pub fn expected_compression_ratio(&self) -> f64 {
        match self {
            CompressionAlgorithm::None => 1.0,
            CompressionAlgorithm::RLE => 0.6,
            CompressionAlgorithm::Delta => 0.7,
            CompressionAlgorithm::Huffman => 0.5,
            CompressionAlgorithm::LZ77 => 0.4,
            CompressionAlgorithm::SparseOptimized => 0.3,
            CompressionAlgorithm::Adaptive => 0.25,
        }
    }

    /// Get relative compression speed (higher = faster)
    pub fn compression_speed(&self) -> f64 {
        match self {
            CompressionAlgorithm::None => 10.0,
            CompressionAlgorithm::RLE => 8.0,
            CompressionAlgorithm::Delta => 7.0,
            CompressionAlgorithm::Huffman => 4.0,
            CompressionAlgorithm::LZ77 => 3.0,
            CompressionAlgorithm::SparseOptimized => 5.0,
            CompressionAlgorithm::Adaptive => 2.0,
        }
    }

    /// Get description of the algorithm
    pub fn description(&self) -> &'static str {
        match self {
            CompressionAlgorithm::None => "No compression applied",
            CompressionAlgorithm::RLE => "Run-Length Encoding for repeated values",
            CompressionAlgorithm::Delta => "Delta encoding for sparse indices",
            CompressionAlgorithm::Huffman => "Huffman coding for optimal entropy",
            CompressionAlgorithm::LZ77 => "LZ77 dictionary-based compression",
            CompressionAlgorithm::SparseOptimized => "Specialized sparse matrix compression",
            CompressionAlgorithm::Adaptive => "Adaptive hybrid compression strategy",
        }
    }
}
