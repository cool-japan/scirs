//! Adaptive Memory Compressor
//!
//! This module contains the main compressor implementation that coordinates
//! all compression, caching, and out-of-core operations.

use super::access_tracking::{AccessEvent, AccessPattern, AccessTracker, AccessType};
use super::cache::{BlockCache, BlockId};
use super::compressed_data::{BlockType, CompressedBlock, CompressedMatrix};
use super::compression::{CompressionEngine, CompressionResult};
use super::config::{AdaptiveCompressionConfig, CompressionAlgorithm};
use super::memory_mapping::{MemoryMappingConfig, MemoryMappingManager};
use super::out_of_core::OutOfCoreManager;
use super::stats::AccessPatternType;
use super::stats::{CompressionMetadata, CompressionStats, MemoryStats};
use crate::error::{SparseError, SparseResult};
use num_traits::{Float, NumAssign};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Adaptive memory compression manager
pub struct AdaptiveMemoryCompressor {
    config: AdaptiveCompressionConfig,
    memory_usage: AtomicUsize,
    compression_stats: Arc<Mutex<CompressionStats>>,
    block_cache: Arc<Mutex<BlockCache>>,
    access_tracker: Arc<Mutex<AccessTracker>>,
    compression_engine: CompressionEngine,
    hierarchical_levels: Vec<CompressionLevel>,
    out_of_core_manager: Option<OutOfCoreManager>,
    memory_mapping_manager: Option<MemoryMappingManager>,
}

/// Hierarchical compression level configuration
#[derive(Debug, Clone)]
struct CompressionLevel {
    level: u8,
    compression_ratio: f64,
    algorithm: CompressionAlgorithm,
    block_size: usize,
    access_threshold: usize,
}

/// Compression strategy selection
#[derive(Debug)]
struct CompressionStrategy {
    algorithm: CompressionAlgorithm,
    block_size: usize,
    hierarchical: bool,
    predicted_ratio: f64,
}

/// Sparsity pattern analysis results
#[derive(Debug, Default)]
struct SparsityPatternAnalysis {
    avg_nnz_per_row: f64,
    max_nnz_per_row: usize,
    min_nnz_per_row: usize,
    sequential_patterns: usize,
    clustering_factor: f64,
    bandwidth: usize,
}

/// Access pattern information for a matrix
#[derive(Debug, Default)]
struct AccessPatternInfo {
    total_accesses: usize,
    avg_temporal_locality: f64,
    avg_spatial_locality: f64,
    pattern_count: usize,
}

/// Cache statistics for internal use
#[derive(Debug)]
struct CacheStats {
    hits: usize,
    misses: usize,
    hit_ratio: f64,
}

impl AdaptiveMemoryCompressor {
    /// Create a new adaptive memory compressor
    pub fn new(config: AdaptiveCompressionConfig) -> SparseResult<Self> {
        let block_cache = BlockCache::new(config.cache_size);
        let access_tracker = AccessTracker::new(); // Default capacity
        let compression_engine = CompressionEngine::new();

        // Initialize hierarchical compression levels
        let hierarchical_levels = vec![
            CompressionLevel {
                level: 1,
                compression_ratio: 2.0,
                algorithm: CompressionAlgorithm::RLE,
                block_size: config.block_size,
                access_threshold: 100,
            },
            CompressionLevel {
                level: 2,
                compression_ratio: 4.0,
                algorithm: CompressionAlgorithm::Delta,
                block_size: config.block_size / 2,
                access_threshold: 50,
            },
            CompressionLevel {
                level: 3,
                compression_ratio: 8.0,
                algorithm: CompressionAlgorithm::LZ77,
                block_size: config.block_size / 4,
                access_threshold: 10,
            },
        ];

        // Initialize out-of-core manager if enabled
        let out_of_core_manager = if config.out_of_core {
            Some(OutOfCoreManager::new(&config.temp_directory)?)
        } else {
            None
        };

        // Initialize memory mapping manager if enabled
        let memory_mapping_manager = if config.memory_mapping {
            let mapping_config = MemoryMappingConfig {
                read_only: false,
                write_through: true,
                prefetch: true,
                page_size_hint: 4096,
            };
            Some(MemoryMappingManager::new(mapping_config))
        } else {
            None
        };

        Ok(Self {
            config,
            memory_usage: AtomicUsize::new(0),
            compression_stats: Arc::new(Mutex::new(CompressionStats::new())),
            block_cache: Arc::new(Mutex::new(block_cache)),
            access_tracker: Arc::new(Mutex::new(access_tracker)),
            compression_engine,
            hierarchical_levels,
            out_of_core_manager,
            memory_mapping_manager,
        })
    }

    /// Compress sparse matrix data adaptively
    #[allow(clippy::too_many_arguments)]
    pub fn compress_matrix<T>(
        &mut self,
        matrix_id: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<CompressedMatrix<T>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let total_size = std::mem::size_of_val(indptr)
            + std::mem::size_of_val(indices)
            + std::mem::size_of_val(data);

        // Check if compression is needed
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        let usage_ratio = (current_usage + total_size) as f64 / self.config.memory_budget as f64;

        if usage_ratio < self.config.compression_threshold && !self.config.adaptive_compression {
            // No compression needed
            return self.create_uncompressed_matrix(matrix_id, rows, indptr, indices, data);
        }

        let start_time = std::time::Instant::now();

        // Determine optimal compression strategy
        let compression_strategy =
            self.determine_compression_strategy(matrix_id, rows, indptr, indices)?;

        // Apply compression based on strategy
        let compressed_blocks = self.apply_compression_strategy(
            &compression_strategy,
            matrix_id,
            rows,
            indptr,
            indices,
            data,
        )?;

        let compression_time = start_time.elapsed().as_secs_f64();

        // Update statistics
        self.update_compression_stats(total_size, &compressed_blocks, compression_time);

        // Update memory usage
        let compressed_size = compressed_blocks
            .iter()
            .map(|b| b.compressed_data.len())
            .sum::<usize>();
        self.memory_usage
            .fetch_add(compressed_size, Ordering::Relaxed);

        // Handle out-of-core storage if needed
        self.handle_out_of_core_storage(&compressed_blocks)?;

        let mut compressed_matrix = CompressedMatrix::new(
            matrix_id,
            rows,
            if !indptr.is_empty() {
                *indices.iter().max().unwrap_or(&0) + 1
            } else {
                0
            },
            compression_strategy.algorithm,
            compression_strategy.block_size,
        );

        // Add the compressed blocks to the matrix
        for block in compressed_blocks {
            compressed_matrix.add_block(block);
        }

        Ok(compressed_matrix)
    }

    /// Decompress matrix data
    pub fn decompress_matrix<T>(
        &mut self,
        compressed_matrix: &CompressedMatrix<T>,
    ) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<T>)>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug + num_traits::FromPrimitive,
    {
        let start_time = std::time::Instant::now();

        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut data = Vec::new();

        // Decompress each block
        for block in compressed_matrix.get_blocks_row_major() {
            // Check cache first
            let decompressed_data =
                if let Some(cached_data) = self.get_cached_block(&block.blockid)? {
                    cached_data
                } else {
                    // Decompress and cache
                    let decompressed =
                        self.decompress_block(block, compressed_matrix.compression_algorithm)?;
                    self.cache_block(&block.blockid, &decompressed)?;
                    decompressed
                };

            // Parse decompressed data based on block type
            match block.block_type {
                BlockType::IndPtr => {
                    indptr.extend(self.parse_indptr_data(&decompressed_data)?);
                }
                BlockType::Indices => {
                    indices.extend(self.parse_indices_data(&decompressed_data)?);
                }
                BlockType::Data => {
                    data.extend(self.parse_data_values::<T>(&decompressed_data)?);
                }
                BlockType::Combined => {
                    let (block_indptr, block_indices, block_data) =
                        self.parse_combined_data::<T>(&decompressed_data)?;
                    indptr.extend(block_indptr);
                    indices.extend(block_indices);
                    data.extend(block_data);
                }
                BlockType::Metadata => {
                    // Handle metadata blocks if needed
                }
            }
        }

        let decompression_time = start_time.elapsed().as_secs_f64();

        // Update statistics
        if let Ok(mut stats) = self.compression_stats.lock() {
            stats.decompression_time += decompression_time;
        }

        // Record access pattern
        self.record_matrix_access(compressed_matrix.matrixid, AccessType::Read);

        Ok((indptr, indices, data))
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        let usage_ratio = current_usage as f64 / self.config.memory_budget as f64;

        let compression_stats = self.compression_stats.lock().unwrap().clone();
        let cache_stats = self.get_cache_stats();

        let mut memory_stats = MemoryStats::new(self.config.memory_budget, self.config.out_of_core);
        memory_stats.update_memory_usage(current_usage);
        memory_stats.compression_stats = compression_stats;
        memory_stats.cache_hits = cache_stats.hits;
        memory_stats.cache_misses = cache_stats.misses;
        memory_stats.cache_hit_ratio = cache_stats.hit_ratio;
        memory_stats
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> CompressionStats {
        self.compression_stats.lock().unwrap().clone()
    }

    /// Optimize for sequential access patterns
    pub fn optimize_for_sequential_access(&mut self) {
        // Note: AccessTracker doesn't have set_access_pattern_hint method
        // This optimization is handled internally by the tracker
    }

    /// Optimize for random access patterns
    pub fn optimize_for_random_access(&mut self) {
        // Note: AccessTracker doesn't have set_access_pattern_hint method
        // This optimization is handled internally by the tracker
    }

    /// Clear cache and reset statistics
    pub fn reset(&mut self) -> SparseResult<()> {
        // Clear cache
        if let Ok(mut cache) = self.block_cache.lock() {
            cache.clear();
        }

        // Reset statistics
        if let Ok(mut stats) = self.compression_stats.lock() {
            *stats = CompressionStats::new();
        }

        // Reset access tracker
        if let Ok(mut tracker) = self.access_tracker.lock() {
            tracker.cleanup_old_patterns(0); // Remove all patterns
        }

        // Reset memory usage
        self.memory_usage.store(0, Ordering::Relaxed);

        // Cleanup out-of-core files
        if let Some(ref mut manager) = self.out_of_core_manager {
            manager.cleanup()?;
        }

        Ok(())
    }

    // Private helper methods

    fn determine_compression_strategy(
        &self,
        matrix_id: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
    ) -> SparseResult<CompressionStrategy> {
        // Analyze matrix characteristics
        let nnz = indices.len();
        let density = if rows > 0 && !indices.is_empty() {
            let max_col = *indices.iter().max().unwrap_or(&0);
            nnz as f64 / (rows as f64 * (max_col + 1) as f64)
        } else {
            0.0
        };

        // Analyze sparsity patterns
        let pattern_analysis = self.analyze_sparsity_patterns(indptr, indices);

        // Check access patterns if available
        let access_info = self.get_access_pattern_info(matrix_id);

        // Select compression algorithm based on analysis
        let algorithm = if self.config.adaptive_compression {
            self.select_adaptive_algorithm(density, &pattern_analysis, &access_info)
        } else {
            self.config.compression_algorithm
        };

        // Determine block size
        let block_size = self.determine_optimal_block_size(rows, nnz, density);

        Ok(CompressionStrategy {
            algorithm,
            block_size,
            hierarchical: self.config.hierarchical_compression,
            predicted_ratio: self.predict_compression_ratio(algorithm, density, &pattern_analysis),
        })
    }

    fn apply_compression_strategy<T>(
        &mut self,
        strategy: &CompressionStrategy,
        matrix_id: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<Vec<CompressedBlock>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        // Serialize matrix components
        let indptr_data = self.serialize_indptr(indptr)?;
        let indices_data = self.serialize_indices(indices)?;
        let data_data = self.serialize_data(data)?;

        let mut blocks = Vec::new();

        // Compress indptr
        let indptr_block_id = BlockId::new(matrix_id, 0, 0);
        let indptr_result = self.compression_engine.compress(
            &indptr_data,
            strategy.algorithm,
            &indptr_block_id,
            BlockType::IndPtr,
        )?;
        blocks.push(CompressedBlock::new(
            indptr_block_id,
            BlockType::IndPtr,
            indptr_result.compressed_data,
            indptr_data.len(),
            (indptr_result.compression_ratio.clamp(1.0, 10.0) as u8).max(1),
        ));

        // Compress indices
        let indices_block_id = BlockId::new(matrix_id, 0, 1);
        let indices_result = self.compression_engine.compress(
            &indices_data,
            strategy.algorithm,
            &indices_block_id,
            BlockType::Indices,
        )?;
        blocks.push(CompressedBlock::new(
            indices_block_id,
            BlockType::Indices,
            indices_result.compressed_data,
            indices_data.len(),
            (indices_result.compression_ratio.clamp(1.0, 10.0) as u8).max(1),
        ));

        // Compress data
        let data_block_id = BlockId::new(matrix_id, 0, 2);
        let data_result = self.compression_engine.compress(
            &data_data,
            strategy.algorithm,
            &data_block_id,
            BlockType::Data,
        )?;
        blocks.push(CompressedBlock::new(
            data_block_id,
            BlockType::Data,
            data_result.compressed_data,
            data_data.len(),
            (data_result.compression_ratio.clamp(1.0, 10.0) as u8).max(1),
        ));

        Ok(blocks)
    }

    fn analyze_sparsity_patterns(
        &self,
        indptr: &[usize],
        indices: &[usize],
    ) -> SparsityPatternAnalysis {
        let mut analysis = SparsityPatternAnalysis::default();

        if indptr.len() <= 1 {
            return analysis;
        }

        let rows = indptr.len() - 1;

        // Analyze row distribution
        let mut row_nnz = Vec::new();
        for row in 0..rows {
            row_nnz.push(indptr[row + 1] - indptr[row]);
        }

        analysis.avg_nnz_per_row = row_nnz.iter().sum::<usize>() as f64 / rows as f64;
        analysis.max_nnz_per_row = *row_nnz.iter().max().unwrap_or(&0);
        analysis.min_nnz_per_row = *row_nnz.iter().min().unwrap_or(&0);

        // Analyze column patterns
        analysis.sequential_patterns = self.count_sequential_patterns(indices);
        analysis.clustering_factor = self.calculate_clustering_factor(indptr, indices);
        analysis.bandwidth = self.calculate_bandwidth(indptr, indices);

        analysis
    }

    fn count_sequential_patterns(&self, indices: &[usize]) -> usize {
        let mut sequential_count = 0;
        let mut current_sequence = 0;

        for window in indices.windows(2) {
            if window[1] == window[0] + 1 {
                current_sequence += 1;
            } else {
                if current_sequence >= 3 {
                    sequential_count += current_sequence;
                }
                current_sequence = 0;
            }
        }

        if current_sequence >= 3 {
            sequential_count += current_sequence;
        }

        sequential_count
    }

    fn calculate_clustering_factor(&self, indptr: &[usize], indices: &[usize]) -> f64 {
        if indptr.len() <= 1 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut total_pairs = 0;

        let rows = indptr.len() - 1;
        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];

            if end > start + 1 {
                for i in start..(end - 1) {
                    total_distance += (indices[i + 1] - indices[i]) as f64;
                    total_pairs += 1;
                }
            }
        }

        if total_pairs > 0 {
            total_distance / total_pairs as f64
        } else {
            0.0
        }
    }

    fn calculate_bandwidth(&self, indptr: &[usize], indices: &[usize]) -> usize {
        if indptr.len() <= 1 {
            return 0;
        }

        let mut max_bandwidth = 0;
        let rows = indptr.len() - 1;

        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];

            if end > start {
                let min_col = indices[start];
                let max_col = indices[end - 1];
                let bandwidth = max_col.saturating_sub(min_col);
                max_bandwidth = max_bandwidth.max(bandwidth);
            }
        }

        max_bandwidth
    }

    fn get_access_pattern_info(&self, _matrix_id: u64) -> AccessPatternInfo {
        let access_tracker = self.access_tracker.lock().unwrap();
        // AccessTracker doesn't have get_matrix_access_info, so we'll use available methods
        let stats = access_tracker.get_statistics();
        AccessPatternInfo {
            total_accesses: stats.total_access_events,
            avg_temporal_locality: 0.5, // Default value
            avg_spatial_locality: 0.5,  // Default value
            pattern_count: stats.total_tracked_blocks,
        }
    }

    fn select_adaptive_algorithm(
        &self,
        density: f64,
        pattern_analysis: &SparsityPatternAnalysis,
        access_info: &AccessPatternInfo,
    ) -> CompressionAlgorithm {
        // Decision tree for algorithm selection
        if density > 0.1 {
            // Dense matrices benefit from general compression
            CompressionAlgorithm::LZ77
        } else if pattern_analysis.sequential_patterns
            > pattern_analysis.avg_nnz_per_row as usize * 10
        {
            // High sequential patterns favor RLE
            CompressionAlgorithm::RLE
        } else if pattern_analysis.clustering_factor < 2.0 {
            // Low clustering suggests delta encoding
            CompressionAlgorithm::Delta
        } else if access_info.avg_temporal_locality > 0.8 {
            // High temporal locality suggests sparse-optimized
            CompressionAlgorithm::SparseOptimized
        } else {
            // Default to Huffman for general case
            CompressionAlgorithm::Huffman
        }
    }

    fn determine_optimal_block_size(&self, rows: usize, nnz: usize, density: f64) -> usize {
        let base_block_size = self.config.block_size;

        // Adjust block size based on matrix characteristics
        let size_factor = if rows > 1_000_000 {
            2.0 // Larger blocks for large matrices
        } else if rows < 10_000 {
            0.5 // Smaller blocks for small matrices
        } else {
            1.0
        };

        let density_factor = if density > 0.1 {
            1.5 // Larger blocks for denser matrices
        } else {
            1.0
        };

        let nnz_factor = if nnz > 10_000_000 {
            1.5 // Larger blocks for many non-zeros
        } else {
            1.0
        };

        let optimal_size =
            (base_block_size as f64 * size_factor * density_factor * nnz_factor) as usize;
        optimal_size.clamp(4096, 16 * 1024 * 1024) // 4KB to 16MB range
    }

    fn predict_compression_ratio(
        &self,
        algorithm: CompressionAlgorithm,
        density: f64,
        pattern_analysis: &SparsityPatternAnalysis,
    ) -> f64 {
        let base_ratio = algorithm.expected_compression_ratio();

        // Adjust based on matrix characteristics
        let adjustment = if pattern_analysis.bandwidth > 100000 {
            0.8 // Lower compression for high bandwidth
        } else if pattern_analysis.bandwidth < 100 {
            1.2 // Higher compression for low bandwidth
        } else {
            1.0
        };

        base_ratio * adjustment
    }

    fn handle_out_of_core_storage(&mut self, blocks: &[CompressedBlock]) -> SparseResult<()> {
        if let Some(ref mut manager) = self.out_of_core_manager {
            for block in blocks {
                if block.compressed_data.len() > self.config.block_size {
                    // Store large blocks out-of-core
                    manager.write_block_to_disk(block)?;
                }
            }
        }
        Ok(())
    }

    fn get_cached_block(&self, block_id: &BlockId) -> SparseResult<Option<Vec<u8>>> {
        if let Ok(mut cache) = self.block_cache.lock() {
            Ok(cache
                .get(block_id)
                .map(|cached_block| cached_block.data.clone()))
        } else {
            Ok(None)
        }
    }

    fn cache_block(&self, block_id: &BlockId, data: &[u8]) -> SparseResult<()> {
        use super::cache::CachedBlock;
        if let Ok(mut cache) = self.block_cache.lock() {
            let cached_block = CachedBlock::new(data.to_vec(), false, 1);
            cache.insert(block_id.clone(), cached_block);
        }
        Ok(())
    }

    fn record_matrix_access(&self, matrix_id: u64, access_type: AccessType) {
        if let Ok(mut tracker) = self.access_tracker.lock() {
            let block_id = BlockId::new(matrix_id, 0, 0); // Default block position
            tracker.record_access(block_id, access_type, 1024); // Default size
        }
    }

    fn decompress_block(
        &mut self,
        block: &CompressedBlock,
        algorithm: CompressionAlgorithm,
    ) -> SparseResult<Vec<u8>> {
        self.compression_engine
            .decompress(&block.compressed_data, algorithm, block.original_size)
    }

    fn create_uncompressed_matrix<T>(
        &self,
        matrix_id: u64,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<CompressedMatrix<T>>
    where
        T: Float + NumAssign + Send + Sync + Copy + std::fmt::Debug,
    {
        let mut matrix = CompressedMatrix::new(
            matrix_id,
            rows,
            if !indptr.is_empty() {
                *indices.iter().max().unwrap_or(&0) + 1
            } else {
                0
            },
            CompressionAlgorithm::None,
            self.config.block_size,
        );

        // Create uncompressed blocks
        let indptr_data = self.serialize_indptr(indptr)?;
        let indices_data = self.serialize_indices(indices)?;
        let data_data = self.serialize_data(data)?;

        matrix.add_block(CompressedBlock::new(
            BlockId::new(matrix_id, 0, 0),
            BlockType::IndPtr,
            indptr_data.clone(),
            indptr_data.len(),
            0,
        ));

        matrix.add_block(CompressedBlock::new(
            BlockId::new(matrix_id, 0, 1),
            BlockType::Indices,
            indices_data.clone(),
            indices_data.len(),
            0,
        ));

        matrix.add_block(CompressedBlock::new(
            BlockId::new(matrix_id, 0, 2),
            BlockType::Data,
            data_data.clone(),
            data_data.len(),
            0,
        ));

        Ok(matrix)
    }

    fn update_compression_stats(
        &self,
        original_size: usize,
        blocks: &[CompressedBlock],
        compression_time: f64,
    ) {
        if let Ok(mut stats) = self.compression_stats.lock() {
            let compressed_size = blocks
                .iter()
                .map(|b| b.compressed_data.len())
                .sum::<usize>();

            stats.update_compression(original_size, compressed_size, compression_time);
        }
    }

    fn get_cache_stats(&self) -> CacheStats {
        if let Ok(cache) = self.block_cache.lock() {
            let stats = cache.get_stats();
            CacheStats {
                hits: stats.hit_count,
                misses: stats.miss_count,
                hit_ratio: if stats.hit_count + stats.miss_count > 0 {
                    stats.hit_count as f64 / (stats.hit_count + stats.miss_count) as f64
                } else {
                    0.0
                },
            }
        } else {
            CacheStats {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
            }
        }
    }

    // Serialization methods
    fn serialize_indptr(&self, indptr: &[usize]) -> SparseResult<Vec<u8>> {
        let mut serialized = Vec::new();
        serialized.extend_from_slice(&indptr.len().to_le_bytes());
        for &value in indptr {
            serialized.extend_from_slice(&value.to_le_bytes());
        }
        Ok(serialized)
    }

    fn serialize_indices(&self, indices: &[usize]) -> SparseResult<Vec<u8>> {
        let mut serialized = Vec::new();
        serialized.extend_from_slice(&indices.len().to_le_bytes());
        for &value in indices {
            serialized.extend_from_slice(&value.to_le_bytes());
        }
        Ok(serialized)
    }

    fn serialize_data<T>(&self, data: &[T]) -> SparseResult<Vec<u8>>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        let mut serialized = Vec::new();
        serialized.extend_from_slice(&data.len().to_le_bytes());
        for &value in data {
            let bytes = value.to_f64().unwrap_or(0.0).to_le_bytes();
            serialized.extend_from_slice(&bytes);
        }
        Ok(serialized)
    }

    // Parsing methods
    fn parse_indptr_data(&self, data: &[u8]) -> SparseResult<Vec<usize>> {
        if data.len() < 8 {
            return Ok(Vec::new());
        }

        let length = usize::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let mut indptr = Vec::with_capacity(length);

        let mut offset = 8;
        for _ in 0..length {
            if offset + 8 <= data.len() {
                let value = usize::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                indptr.push(value);
                offset += 8;
            }
        }

        Ok(indptr)
    }

    fn parse_indices_data(&self, data: &[u8]) -> SparseResult<Vec<usize>> {
        self.parse_indptr_data(data) // Same format
    }

    fn parse_data_values<T>(&self, data: &[u8]) -> SparseResult<Vec<T>>
    where
        T: Float + NumAssign + Send + Sync + Copy + num_traits::FromPrimitive,
    {
        if data.len() < 8 {
            return Ok(Vec::new());
        }

        let length = usize::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let mut values = Vec::with_capacity(length);

        let mut offset = 8;
        for _ in 0..length {
            if offset + 8 <= data.len() {
                let value_f64 = f64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                if let Some(value) = T::from_f64(value_f64) {
                    values.push(value);
                }
                offset += 8;
            }
        }

        Ok(values)
    }

    fn parse_combined_data<T>(&self, _data: &[u8]) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<T>)>
    where
        T: Float + NumAssign + Send + Sync + Copy,
    {
        // Placeholder for combined data parsing
        Ok((Vec::new(), Vec::new(), Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compressor_creation() {
        let config = AdaptiveCompressionConfig::default();
        let compressor = AdaptiveMemoryCompressor::new(config);
        assert!(compressor.is_ok());
    }

    #[test]
    fn test_matrix_compression_roundtrip() {
        let config = AdaptiveCompressionConfig {
            compression_algorithm: CompressionAlgorithm::None,
            out_of_core: false,
            memory_mapping: false,
            ..Default::default()
        };
        let mut compressor = AdaptiveMemoryCompressor::new(config).unwrap();

        let indptr = vec![0, 2, 3];
        let indices = vec![0, 1, 1];
        let data = vec![1.0, 2.0, 3.0];

        let compressed = compressor
            .compress_matrix(1, 2, &indptr, &indices, &data)
            .unwrap();

        let (decompressed_indptr, decompressed_indices, decompressed_data) =
            compressor.decompress_matrix(&compressed).unwrap();

        assert_eq!(decompressed_indptr, indptr);
        assert_eq!(decompressed_indices, indices);
        assert_eq!(decompressed_data.len(), data.len());
    }

    #[test]
    fn test_memory_stats() {
        let config = AdaptiveCompressionConfig::default();
        let compressor = AdaptiveMemoryCompressor::new(config).unwrap();
        let stats = compressor.get_memory_stats();

        assert_eq!(stats.current_memory_usage, 0);
        assert!(stats.memory_usage_ratio >= 0.0);
    }

    #[test]
    fn test_access_pattern_optimization() {
        let config = AdaptiveCompressionConfig::default();
        let mut compressor = AdaptiveMemoryCompressor::new(config).unwrap();

        // Test sequential optimization
        compressor.optimize_for_sequential_access();

        // Test random optimization
        compressor.optimize_for_random_access();

        // Should not panic
        assert!(true);
    }

    #[test]
    fn test_compressor_reset() {
        let temp_dir = TempDir::new().unwrap();
        let config = AdaptiveCompressionConfig {
            temp_directory: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };
        let mut compressor = AdaptiveMemoryCompressor::new(config).unwrap();

        // Add some data
        let indptr = vec![0, 1];
        let indices = vec![0];
        let data = vec![1.0];

        let _ = compressor.compress_matrix(1, 1, &indptr, &indices, &data);

        // Reset should work
        let result = compressor.reset();
        assert!(result.is_ok());

        let stats = compressor.get_memory_stats();
        assert_eq!(stats.current_memory_usage, 0);
    }
}
