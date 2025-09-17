//! Compression algorithms and engine for adaptive memory compression
//!
//! This module implements various compression algorithms optimized for sparse matrix data,
//! including run-length encoding, delta compression, Huffman coding, and adaptive strategies.

use super::cache::BlockId;
use super::compressed_data::{BlockType, CompressedBlock};
use super::config::CompressionAlgorithm;
use super::stats::SparsityPatternAnalysis;
use crate::error::{SparseError, SparseResult};
use num_traits::{Float, NumAssign};
use std::collections::HashMap;

/// Compression engine that handles all compression algorithms
#[derive(Debug)]
pub struct CompressionEngine {
    /// Current compression strategy
    current_strategy: CompressionStrategy,
    /// Algorithm performance metrics
    algorithm_metrics: HashMap<CompressionAlgorithm, AlgorithmMetrics>,
    /// Huffman tables cache
    huffman_tables: HashMap<String, HuffmanTable>,
}

/// Compression strategy configuration
#[derive(Debug, Clone)]
pub(crate) struct CompressionStrategy {
    pub algorithm: CompressionAlgorithm,
    pub block_size: usize,
    pub hierarchical: bool,
    pub predicted_ratio: f64,
    pub actual_ratio: f64,
    pub compression_speed: f64,
    pub decompression_speed: f64,
}

/// Algorithm performance metrics
#[derive(Debug, Clone)]
struct AlgorithmMetrics {
    total_operations: usize,
    total_compression_time: f64,
    total_decompression_time: f64,
    total_original_size: usize,
    total_compressed_size: usize,
    success_count: usize,
}

/// Compression result with metadata
#[derive(Debug)]
pub struct CompressionResult {
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub compression_ratio: f64,
    pub compression_time: f64,
    pub algorithm_used: CompressionAlgorithm,
}

/// Huffman coding table
#[derive(Debug, Clone)]
struct HuffmanTable {
    encode_table: HashMap<u8, Vec<bool>>,
    decode_tree: HuffmanNode,
}

/// Huffman tree node
#[derive(Debug, Clone, PartialEq, Eq)]
enum HuffmanNode {
    Leaf {
        value: u8,
    },
    Internal {
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

/// Frequency counter for Huffman coding
#[derive(Debug, Clone, Eq, PartialEq)]
struct FrequencyNode {
    frequency: usize,
    node: HuffmanNode,
}

impl std::cmp::Ord for FrequencyNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.frequency.cmp(&self.frequency) // Reverse for min-heap
    }
}

impl std::cmp::PartialOrd for FrequencyNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl CompressionEngine {
    /// Create a new compression engine
    pub fn new() -> Self {
        Self {
            current_strategy: CompressionStrategy::default(),
            algorithm_metrics: HashMap::new(),
            huffman_tables: HashMap::new(),
        }
    }

    /// Compress data using the specified algorithm
    pub fn compress(
        &mut self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        block_id: &BlockId,
        block_type: BlockType,
    ) -> SparseResult<CompressionResult> {
        let start_time = std::time::Instant::now();
        let original_size = data.len();

        let compressed_data = match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::RLE => self.compress_rle(data)?,
            CompressionAlgorithm::Delta => self.compress_delta(data)?,
            CompressionAlgorithm::Huffman => self.compress_huffman(data)?,
            CompressionAlgorithm::LZ77 => self.compress_lz77(data)?,
            CompressionAlgorithm::SparseOptimized => {
                self.compress_sparse_optimized(data, block_type)?
            }
            CompressionAlgorithm::Adaptive => self.compress_adaptive(data, block_type)?,
        };

        let compression_time = start_time.elapsed().as_secs_f64();
        let compression_ratio = if original_size > 0 {
            compressed_data.len() as f64 / original_size as f64
        } else {
            1.0
        };

        // Update algorithm metrics
        self.update_algorithm_metrics(
            algorithm,
            compression_time,
            original_size,
            compressed_data.len(),
            true,
        );

        Ok(CompressionResult {
            compressed_data,
            original_size,
            compression_ratio,
            compression_time,
            algorithm_used: algorithm,
        })
    }

    /// Decompress data using the specified algorithm
    pub fn decompress(
        &mut self,
        compressed_data: &[u8],
        algorithm: CompressionAlgorithm,
        original_size: usize,
    ) -> SparseResult<Vec<u8>> {
        let start_time = std::time::Instant::now();

        let decompressed_data = match algorithm {
            CompressionAlgorithm::None => compressed_data.to_vec(),
            CompressionAlgorithm::RLE => self.decompress_rle(compressed_data)?,
            CompressionAlgorithm::Delta => self.decompress_delta(compressed_data)?,
            CompressionAlgorithm::Huffman => self.decompress_huffman(compressed_data)?,
            CompressionAlgorithm::LZ77 => self.decompress_lz77(compressed_data)?,
            CompressionAlgorithm::SparseOptimized => {
                self.decompress_sparse_optimized(compressed_data)?
            }
            CompressionAlgorithm::Adaptive => self.decompress_adaptive(compressed_data)?,
        };

        let decompression_time = start_time.elapsed().as_secs_f64();

        // Update algorithm metrics
        self.update_algorithm_metrics(
            algorithm,
            decompression_time,
            original_size,
            compressed_data.len(),
            true,
        );

        if decompressed_data.len() != original_size {
            return Err(SparseError::ComputationError(format!(
                "Decompression size mismatch: expected {}, got {}",
                original_size,
                decompressed_data.len()
            )));
        }

        Ok(decompressed_data)
    }

    /// Run-Length Encoding compression
    fn compress_rle(&self, data: &[u8]) -> SparseResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();
        let mut current_byte = data[0];
        let mut count = 1u8;

        for &byte in &data[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        // Add the last run
        compressed.push(count);
        compressed.push(current_byte);

        Ok(compressed)
    }

    /// Run-Length Encoding decompression
    fn decompress_rle(&self, compressed_data: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed_data.len() % 2 != 0 {
            return Err(SparseError::ComputationError(
                "Invalid RLE data".to_string(),
            ));
        }

        let mut decompressed = Vec::new();

        for chunk in compressed_data.chunks(2) {
            let count = chunk[0] as usize;
            let byte = chunk[1];
            decompressed.extend(std::iter::repeat(byte).take(count));
        }

        Ok(decompressed)
    }

    /// Delta encoding compression (for sorted integer sequences)
    fn compress_delta(&self, data: &[u8]) -> SparseResult<Vec<u8>> {
        if data.len() < 4 {
            return Ok(data.to_vec()); // Too small for delta encoding
        }

        // Interpret as u32 integers
        let integers: Vec<u32> = data
            .chunks(4)
            .map(|chunk| {
                if chunk.len() == 4 {
                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                } else {
                    0
                }
            })
            .collect();

        if integers.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();

        // Store first value as-is
        compressed.extend(&integers[0].to_le_bytes());

        // Store deltas
        for i in 1..integers.len() {
            let delta = integers[i].wrapping_sub(integers[i - 1]);

            // Use variable-length encoding for deltas
            if delta < 128 {
                compressed.push(delta as u8);
            } else if delta < 32768 {
                compressed.push(0x80 | (delta as u8));
                compressed.push((delta >> 7) as u8);
            } else {
                compressed.push(0xFF);
                compressed.extend(&delta.to_le_bytes());
            }
        }

        Ok(compressed)
    }

    /// Delta encoding decompression
    fn decompress_delta(&self, compressed_data: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed_data.len() < 4 {
            return Ok(compressed_data.to_vec());
        }

        let mut decompressed = Vec::new();
        let mut pos = 0;

        // Read first value
        if compressed_data.len() < 4 {
            return Err(SparseError::ComputationError(
                "Invalid delta data".to_string(),
            ));
        }

        let first_value = u32::from_le_bytes([
            compressed_data[0],
            compressed_data[1],
            compressed_data[2],
            compressed_data[3],
        ]);
        decompressed.extend(&first_value.to_le_bytes());
        pos += 4;

        let mut current_value = first_value;

        // Read deltas
        while pos < compressed_data.len() {
            let delta = if compressed_data[pos] < 0x80 {
                let d = compressed_data[pos] as u32;
                pos += 1;
                d
            } else if compressed_data[pos] < 0xFF {
                if pos + 1 >= compressed_data.len() {
                    break;
                }
                let d = ((compressed_data[pos] & 0x7F) as u32)
                    | ((compressed_data[pos + 1] as u32) << 7);
                pos += 2;
                d
            } else {
                if pos + 4 >= compressed_data.len() {
                    break;
                }
                let d = u32::from_le_bytes([
                    compressed_data[pos + 1],
                    compressed_data[pos + 2],
                    compressed_data[pos + 3],
                    compressed_data[pos + 4],
                ]);
                pos += 5;
                d
            };

            current_value = current_value.wrapping_add(delta);
            decompressed.extend(&current_value.to_le_bytes());
        }

        Ok(decompressed)
    }

    /// Huffman encoding compression
    fn compress_huffman(&mut self, data: &[u8]) -> SparseResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let table = self.build_huffman_table(data);
        let mut bit_stream = Vec::new();
        let mut current_byte = 0u8;
        let mut bit_count = 0;

        for &byte in data {
            if let Some(code) = table.encode_table.get(&byte) {
                for &bit in code {
                    if bit {
                        current_byte |= 1 << (7 - bit_count);
                    }
                    bit_count += 1;

                    if bit_count == 8 {
                        bit_stream.push(current_byte);
                        current_byte = 0;
                        bit_count = 0;
                    }
                }
            }
        }

        // Add remaining bits
        if bit_count > 0 {
            bit_stream.push(current_byte);
        }

        // Serialize table and data
        let mut result = Vec::new();
        let table_data = self.serialize_huffman_table(&table)?;
        result.extend(&(table_data.len() as u32).to_le_bytes());
        result.extend(table_data);
        result.push(bit_count); // Store remaining bits count
        result.extend(bit_stream);

        Ok(result)
    }

    /// Huffman encoding decompression
    fn decompress_huffman(&self, compressed_data: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed_data.len() < 5 {
            return Ok(Vec::new());
        }

        let table_size = u32::from_le_bytes([
            compressed_data[0],
            compressed_data[1],
            compressed_data[2],
            compressed_data[3],
        ]) as usize;

        if compressed_data.len() < 4 + table_size + 1 {
            return Err(SparseError::ComputationError(
                "Invalid Huffman data".to_string(),
            ));
        }

        let table_data = &compressed_data[4..4 + table_size];
        let table = self.deserialize_huffman_table(table_data)?;

        let remaining_bits = compressed_data[4 + table_size];
        let bit_stream = &compressed_data[4 + table_size + 1..];

        let mut decompressed = Vec::new();
        let mut current_node = &table.decode_tree;

        for (i, &byte) in bit_stream.iter().enumerate() {
            let bit_limit = if i == bit_stream.len() - 1 && remaining_bits > 0 {
                remaining_bits
            } else {
                8
            };

            for bit_pos in 0..bit_limit {
                let bit = (byte >> (7 - bit_pos)) & 1 == 1;

                current_node = match current_node {
                    HuffmanNode::Internal { left, right } => {
                        if bit {
                            right
                        } else {
                            left
                        }
                    }
                    HuffmanNode::Leaf { value } => {
                        decompressed.push(*value);
                        &table.decode_tree
                    }
                };

                if let HuffmanNode::Leaf { value } = current_node {
                    decompressed.push(*value);
                    current_node = &table.decode_tree;
                }
            }
        }

        Ok(decompressed)
    }

    /// Simple LZ77 compression
    fn compress_lz77(&self, data: &[u8]) -> SparseResult<Vec<u8>> {
        const WINDOW_SIZE: usize = 4096;
        const LOOKAHEAD_SIZE: usize = 256;

        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let mut best_length = 0;
            let mut best_distance = 0;

            // Search for matches in the sliding window
            let window_start = pos.saturating_sub(WINDOW_SIZE);
            let lookahead_end = (pos + LOOKAHEAD_SIZE).min(data.len());

            for window_pos in window_start..pos {
                let mut length = 0;
                while window_pos + length < pos
                    && pos + length < lookahead_end
                    && data[window_pos + length] == data[pos + length]
                {
                    length += 1;
                }

                if length > best_length {
                    best_length = length;
                    best_distance = pos - window_pos;
                }
            }

            if best_length >= 3 {
                // Encode as (distance, length)
                compressed.push(0xFF); // Marker for encoded sequence
                compressed.extend(&(best_distance as u16).to_le_bytes());
                compressed.push(best_length as u8);
                pos += best_length;
            } else {
                // Encode as literal
                compressed.push(data[pos]);
                pos += 1;
            }
        }

        Ok(compressed)
    }

    /// Simple LZ77 decompression
    fn decompress_lz77(&self, compressed_data: &[u8]) -> SparseResult<Vec<u8>> {
        let mut decompressed = Vec::new();
        let mut pos = 0;

        while pos < compressed_data.len() {
            if compressed_data[pos] == 0xFF && pos + 3 < compressed_data.len() {
                // Decode encoded sequence
                let distance =
                    u16::from_le_bytes([compressed_data[pos + 1], compressed_data[pos + 2]])
                        as usize;
                let length = compressed_data[pos + 3] as usize;

                if distance == 0 || distance > decompressed.len() {
                    return Err(SparseError::ComputationError(
                        "Invalid LZ77 distance".to_string(),
                    ));
                }

                let start_pos = decompressed.len() - distance;
                for i in 0..length {
                    let byte = decompressed[start_pos + (i % distance)];
                    decompressed.push(byte);
                }

                pos += 4;
            } else {
                // Literal byte
                decompressed.push(compressed_data[pos]);
                pos += 1;
            }
        }

        Ok(decompressed)
    }

    /// Sparse-optimized compression
    fn compress_sparse_optimized(
        &mut self,
        data: &[u8],
        block_type: BlockType,
    ) -> SparseResult<Vec<u8>> {
        match block_type {
            BlockType::Indices => {
                // Use delta encoding for indices (usually sorted)
                self.compress_delta(data)
            }
            BlockType::IndPtr => {
                // Use delta encoding for indptr (monotonic)
                self.compress_delta(data)
            }
            BlockType::Data => {
                // Use RLE for data values (may have many zeros)
                self.compress_rle(data)
            }
            _ => {
                // Use adaptive compression for other types
                self.compress_adaptive(data, block_type)
            }
        }
    }

    /// Sparse-optimized decompression
    fn decompress_sparse_optimized(&self, compressed_data: &[u8]) -> SparseResult<Vec<u8>> {
        // Try different decompression methods and return the first successful one
        // In practice, you'd store the method used in the compressed data header

        if let Ok(result) = self.decompress_delta(compressed_data) {
            Ok(result)
        } else if let Ok(result) = self.decompress_rle(compressed_data) {
            Ok(result)
        } else {
            Ok(compressed_data.to_vec())
        }
    }

    /// Adaptive compression (tries multiple algorithms)
    fn compress_adaptive(&mut self, data: &[u8], block_type: BlockType) -> SparseResult<Vec<u8>> {
        let algorithms = vec![
            CompressionAlgorithm::RLE,
            CompressionAlgorithm::Delta,
            CompressionAlgorithm::LZ77,
        ];

        let mut best_result = data.to_vec();
        let mut best_algorithm = CompressionAlgorithm::None;

        for algorithm in algorithms {
            if let Ok(compressed) = match algorithm {
                CompressionAlgorithm::RLE => self.compress_rle(data),
                CompressionAlgorithm::Delta => self.compress_delta(data),
                CompressionAlgorithm::LZ77 => self.compress_lz77(data),
                _ => continue,
            } {
                if compressed.len() < best_result.len() {
                    best_result = compressed;
                    best_algorithm = algorithm;
                }
            }
        }

        // Prepend algorithm identifier
        let mut result = vec![best_algorithm as u8];
        result.extend(best_result);
        Ok(result)
    }

    /// Adaptive decompression
    fn decompress_adaptive(&mut self, compressed_data: &[u8]) -> SparseResult<Vec<u8>> {
        if compressed_data.is_empty() {
            return Ok(Vec::new());
        }

        let algorithm_id = compressed_data[0];
        let data = &compressed_data[1..];

        match algorithm_id {
            0 => Ok(data.to_vec()),             // None
            1 => self.decompress_rle(data),     // RLE
            2 => self.decompress_delta(data),   // Delta
            3 => self.decompress_huffman(data), // Huffman
            4 => self.decompress_lz77(data),    // LZ77
            _ => Err(SparseError::ComputationError(
                "Unknown compression algorithm".to_string(),
            )),
        }
    }

    /// Build Huffman table from data
    fn build_huffman_table(&mut self, data: &[u8]) -> HuffmanTable {
        // Count frequencies
        let mut frequencies = HashMap::new();
        for &byte in data {
            *frequencies.entry(byte).or_insert(0) += 1;
        }

        if frequencies.len() <= 1 {
            // Handle edge case: single character
            let byte = data[0];
            let mut encode_table = HashMap::new();
            encode_table.insert(byte, vec![false]);
            return HuffmanTable {
                encode_table,
                decode_tree: HuffmanNode::Leaf { value: byte },
            };
        }

        // Build Huffman tree
        let mut heap = std::collections::BinaryHeap::new();
        for (byte, freq) in frequencies {
            heap.push(FrequencyNode {
                frequency: freq,
                node: HuffmanNode::Leaf { value: byte },
            });
        }

        while heap.len() > 1 {
            let right = heap.pop().unwrap();
            let left = heap.pop().unwrap();

            heap.push(FrequencyNode {
                frequency: left.frequency + right.frequency,
                node: HuffmanNode::Internal {
                    left: Box::new(left.node),
                    right: Box::new(right.node),
                },
            });
        }

        let root = heap.pop().unwrap().node;

        // Build encoding table
        let mut encode_table = HashMap::new();
        self.build_codes(&root, Vec::new(), &mut encode_table);

        HuffmanTable {
            encode_table,
            decode_tree: root,
        }
    }

    /// Build Huffman codes recursively
    fn build_codes(
        &self,
        node: &HuffmanNode,
        code: Vec<bool>,
        encode_table: &mut HashMap<u8, Vec<bool>>,
    ) {
        match node {
            HuffmanNode::Leaf { value } => {
                encode_table.insert(*value, code);
            }
            HuffmanNode::Internal { left, right } => {
                let mut left_code = code.clone();
                left_code.push(false);
                self.build_codes(left, left_code, encode_table);

                let mut right_code = code;
                right_code.push(true);
                self.build_codes(right, right_code, encode_table);
            }
        }
    }

    /// Serialize Huffman table (simplified)
    fn serialize_huffman_table(&self, _table: &HuffmanTable) -> SparseResult<Vec<u8>> {
        // Simplified serialization - in practice you'd implement proper serialization
        Ok(vec![0])
    }

    /// Deserialize Huffman table (simplified)
    fn deserialize_huffman_table(&self, _data: &[u8]) -> SparseResult<HuffmanTable> {
        // Simplified deserialization - in practice you'd implement proper deserialization
        Err(SparseError::ComputationError(
            "Huffman table deserialization not implemented".to_string(),
        ))
    }

    /// Update algorithm performance metrics
    fn update_algorithm_metrics(
        &mut self,
        algorithm: CompressionAlgorithm,
        time: f64,
        original_size: usize,
        compressed_size: usize,
        success: bool,
    ) {
        let metrics = self
            .algorithm_metrics
            .entry(algorithm)
            .or_insert_with(|| AlgorithmMetrics {
                total_operations: 0,
                total_compression_time: 0.0,
                total_decompression_time: 0.0,
                total_original_size: 0,
                total_compressed_size: 0,
                success_count: 0,
            });

        metrics.total_operations += 1;
        metrics.total_compression_time += time;
        metrics.total_original_size += original_size;
        metrics.total_compressed_size += compressed_size;

        if success {
            metrics.success_count += 1;
        }
    }

    /// Get algorithm performance metrics
    pub fn get_algorithm_metrics(
        &self,
        algorithm: CompressionAlgorithm,
    ) -> Option<&AlgorithmMetrics> {
        self.algorithm_metrics.get(&algorithm)
    }

    /// Get best algorithm for given data characteristics
    pub fn select_best_algorithm(
        &self,
        data_size: usize,
        block_type: BlockType,
    ) -> CompressionAlgorithm {
        match block_type {
            BlockType::Indices | BlockType::IndPtr if data_size > 1024 => {
                CompressionAlgorithm::Delta
            }
            BlockType::Data if data_size > 4096 => CompressionAlgorithm::LZ77,
            _ if data_size > 512 => CompressionAlgorithm::RLE,
            _ => CompressionAlgorithm::None,
        }
    }
}

impl CompressionStrategy {
    /// Create a new compression strategy
    pub fn new(algorithm: CompressionAlgorithm, block_size: usize) -> Self {
        Self {
            algorithm,
            block_size,
            hierarchical: false,
            predicted_ratio: algorithm.expected_compression_ratio(),
            actual_ratio: 1.0,
            compression_speed: algorithm.compression_speed(),
            decompression_speed: algorithm.compression_speed() * 1.5, // Decompression usually faster
        }
    }

    /// Update actual performance metrics
    pub fn update_performance(
        &mut self,
        actual_ratio: f64,
        compression_speed: f64,
        decompression_speed: f64,
    ) {
        self.actual_ratio = actual_ratio;
        self.compression_speed = compression_speed;
        self.decompression_speed = decompression_speed;
    }

    /// Get efficiency score
    pub fn efficiency_score(&self) -> f64 {
        let ratio_score = (1.0 - self.actual_ratio).max(0.0);
        let speed_score = (self.compression_speed / 10.0).min(1.0);
        (ratio_score + speed_score) / 2.0
    }
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        Self::new(CompressionAlgorithm::Adaptive, 1024 * 1024)
    }
}

impl Default for CompressionEngine {
    fn default() -> Self {
        Self::new()
    }
}
