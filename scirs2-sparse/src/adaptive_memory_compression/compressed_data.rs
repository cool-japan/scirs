//! Compressed data structures for adaptive memory compression
//!
//! This module contains the data structures used to represent compressed
//! sparse matrices and their component blocks.

use super::cache::BlockId;
use super::config::CompressionAlgorithm;
use super::stats::CompressionMetadata;
use std::marker::PhantomData;

/// Compressed sparse matrix representation
#[derive(Debug)]
pub struct CompressedMatrix<T> {
    pub matrixid: u64,
    pub original_rows: usize,
    pub original_cols: usize,
    pub compressed_blocks: Vec<CompressedBlock>,
    pub compression_algorithm: CompressionAlgorithm,
    pub block_size: usize,
    pub metadata: CompressionMetadata,
    _phantom: PhantomData<T>,
}

/// Compressed block of matrix data
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    pub blockid: BlockId,
    pub block_type: BlockType,
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub compression_level: u8,
    pub checksum: Option<u64>,
    pub timestamp: u64,
}

/// Type of data stored in a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockType {
    IndPtr,
    Indices,
    #[default]
    Data,
    Combined,
    Metadata,
}

/// Block header for disk storage
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct BlockHeader {
    pub blockid: BlockId,
    pub block_type: u8,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_level: u8,
    pub checksum: u64,
    pub timestamp: u64,
}

/// Serialized block header (fixed size for disk storage)
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct BlockHeaderSerialized {
    pub blockid: u64,
    pub block_type: u8,
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_level: u8,
    pub checksum: u64,
    pub timestamp: u64,
    pub padding: [u8; 3], // Ensure proper alignment
}

impl<T> CompressedMatrix<T> {
    /// Create a new compressed matrix
    pub fn new(
        matrix_id: u64,
        original_rows: usize,
        original_cols: usize,
        compression_algorithm: CompressionAlgorithm,
        block_size: usize,
    ) -> Self {
        Self {
            matrixid: matrix_id,
            original_rows,
            original_cols,
            compressed_blocks: Vec::new(),
            compression_algorithm,
            block_size,
            metadata: CompressionMetadata::new(0, 0, 0.0),
            _phantom: PhantomData,
        }
    }

    /// Add a compressed block
    pub fn add_block(&mut self, block: CompressedBlock) {
        self.compressed_blocks.push(block);
        self.update_metadata();
    }

    /// Get block by ID
    pub fn get_block(&self, block_id: &BlockId) -> Option<&CompressedBlock> {
        self.compressed_blocks
            .iter()
            .find(|block| &block.blockid == block_id)
    }

    /// Get mutable block by ID
    pub fn get_block_mut(&mut self, block_id: &BlockId) -> Option<&mut CompressedBlock> {
        self.compressed_blocks
            .iter_mut()
            .find(|block| &block.blockid == block_id)
    }

    /// Remove a block
    pub fn remove_block(&mut self, block_id: &BlockId) -> Option<CompressedBlock> {
        if let Some(pos) = self
            .compressed_blocks
            .iter()
            .position(|block| &block.blockid == block_id)
        {
            let removed = self.compressed_blocks.remove(pos);
            self.update_metadata();
            Some(removed)
        } else {
            None
        }
    }

    /// Get blocks of specific type
    pub fn get_blocks_by_type(&self, block_type: BlockType) -> Vec<&CompressedBlock> {
        self.compressed_blocks
            .iter()
            .filter(|block| block.block_type == block_type)
            .collect()
    }

    /// Update metadata based on current blocks
    fn update_metadata(&mut self) {
        let total_original_size: usize = self
            .compressed_blocks
            .iter()
            .map(|block| block.original_size)
            .sum();

        let total_compressed_size: usize = self
            .compressed_blocks
            .iter()
            .map(|block| block.compressed_data.len())
            .sum();

        self.metadata = CompressionMetadata::new(
            total_original_size,
            total_compressed_size,
            0.0, // Compression time would be tracked separately
        );
    }

    /// Get total number of blocks
    pub fn block_count(&self) -> usize {
        self.compressed_blocks.len()
    }

    /// Get total compressed size
    pub fn compressed_size(&self) -> usize {
        self.compressed_blocks
            .iter()
            .map(|block| block.compressed_data.len())
            .sum()
    }

    /// Get total original size
    pub fn original_size(&self) -> usize {
        self.compressed_blocks
            .iter()
            .map(|block| block.original_size)
            .sum()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.metadata.compression_ratio
    }

    /// Check data integrity
    pub fn verify_integrity(&self) -> Result<(), String> {
        for block in &self.compressed_blocks {
            if let Some(expected_checksum) = block.checksum {
                let actual_checksum = Self::calculate_checksum(&block.compressed_data);
                if actual_checksum != expected_checksum {
                    return Err(format!("Checksum mismatch for block {}", block.blockid));
                }
            }
        }
        Ok(())
    }

    /// Calculate checksum for data
    fn calculate_checksum(data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    /// Get memory footprint
    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .compressed_blocks
                .iter()
                .map(|block| block.memory_footprint())
                .sum::<usize>()
    }

    /// Optimize block organization
    pub fn optimize_blocks(&mut self) {
        // Sort blocks by access frequency (if available) or by block ID
        self.compressed_blocks.sort_by(|a, b| {
            a.blockid
                .block_row
                .cmp(&b.blockid.block_row)
                .then_with(|| a.blockid.block_col.cmp(&b.blockid.block_col))
        });
    }

    /// Get blocks in row-major order
    pub fn get_blocks_row_major(&self) -> Vec<&CompressedBlock> {
        let mut blocks = self.compressed_blocks.iter().collect::<Vec<_>>();
        blocks.sort_by(|a, b| {
            a.blockid
                .block_row
                .cmp(&b.blockid.block_row)
                .then_with(|| a.blockid.block_col.cmp(&b.blockid.block_col))
        });
        blocks
    }

    /// Export metadata for persistence
    pub fn export_metadata(&self) -> MatrixMetadataExport {
        MatrixMetadataExport {
            matrix_id: self.matrixid,
            original_rows: self.original_rows,
            original_cols: self.original_cols,
            block_count: self.compressed_blocks.len(),
            compression_algorithm: self.compression_algorithm,
            block_size: self.block_size,
            total_original_size: self.original_size(),
            total_compressed_size: self.compressed_size(),
            compression_ratio: self.compression_ratio(),
            block_map: self
                .compressed_blocks
                .iter()
                .map(|block| (block.blockid.clone(), block.block_type))
                .collect(),
        }
    }
}

impl CompressedBlock {
    /// Create a new compressed block
    pub fn new(
        block_id: BlockId,
        block_type: BlockType,
        compressed_data: Vec<u8>,
        original_size: usize,
        compression_level: u8,
    ) -> Self {
        let checksum = Self::calculate_checksum(&compressed_data);

        Self {
            blockid: block_id,
            block_type,
            compressed_data,
            original_size,
            compression_level,
            checksum: Some(checksum),
            timestamp: Self::current_timestamp(),
        }
    }

    /// Create without checksum (for faster creation)
    pub fn new_unchecked(
        block_id: BlockId,
        block_type: BlockType,
        compressed_data: Vec<u8>,
        original_size: usize,
        compression_level: u8,
    ) -> Self {
        Self {
            blockid: block_id,
            block_type,
            compressed_data,
            original_size,
            compression_level,
            checksum: None,
            timestamp: Self::current_timestamp(),
        }
    }

    /// Get compression ratio for this block
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size > 0 {
            self.compressed_data.len() as f64 / self.original_size as f64
        } else {
            1.0
        }
    }

    /// Get space savings in bytes
    pub fn space_savings(&self) -> usize {
        self.original_size
            .saturating_sub(self.compressed_data.len())
    }

    /// Verify block integrity
    pub fn verify_integrity(&self) -> bool {
        if let Some(expected_checksum) = self.checksum {
            let actual_checksum = Self::calculate_checksum(&self.compressed_data);
            actual_checksum == expected_checksum
        } else {
            true // No checksum to verify
        }
    }

    /// Update checksum
    pub fn update_checksum(&mut self) {
        self.checksum = Some(Self::calculate_checksum(&self.compressed_data));
    }

    /// Calculate checksum for data
    fn calculate_checksum(data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Get memory footprint of this block
    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of::<Self>() + self.compressed_data.len()
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> u64 {
        Self::current_timestamp().saturating_sub(self.timestamp)
    }

    /// Check if block is old
    pub fn is_old(&self, max_age_seconds: u64) -> bool {
        self.age_seconds() > max_age_seconds
    }

    /// Clone data without metadata
    pub fn clone_data(&self) -> Vec<u8> {
        self.compressed_data.clone()
    }

    /// Get size information
    pub fn size_info(&self) -> BlockSizeInfo {
        BlockSizeInfo {
            original_size: self.original_size,
            compressed_size: self.compressed_data.len(),
            compression_ratio: self.compression_ratio(),
            space_savings: self.space_savings(),
        }
    }
}

impl BlockType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            BlockType::IndPtr => "indptr",
            BlockType::Indices => "indices",
            BlockType::Data => "data",
            BlockType::Combined => "combined",
            BlockType::Metadata => "metadata",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "indptr" => Some(BlockType::IndPtr),
            "indices" => Some(BlockType::Indices),
            "data" => Some(BlockType::Data),
            "combined" => Some(BlockType::Combined),
            "metadata" => Some(BlockType::Metadata),
            _ => None,
        }
    }

    /// Get priority for compression (higher = more important)
    pub fn compression_priority(&self) -> u8 {
        match self {
            BlockType::Data => 10,    // Highest priority - usually largest
            BlockType::Indices => 8,  // High priority - often compressible
            BlockType::Combined => 7, // High priority - mixed content
            BlockType::IndPtr => 5,   // Medium priority - usually small
            BlockType::Metadata => 3, // Lower priority - typically small
        }
    }

    /// Check if this block type typically benefits from compression
    pub fn benefits_from_compression(&self) -> bool {
        match self {
            BlockType::Data => true,      // Numerical data often compresses well
            BlockType::Indices => true,   // Sorted indices compress well
            BlockType::Combined => true,  // Mixed content varies
            BlockType::IndPtr => false,   // Usually small and regular
            BlockType::Metadata => false, // Usually small
        }
    }
}

impl BlockHeader {
    /// Create new block header
    pub fn new(
        block_id: BlockId,
        block_type: BlockType,
        original_size: usize,
        compressed_size: usize,
        compression_level: u8,
    ) -> Self {
        Self {
            blockid: block_id,
            block_type: block_type as u8,
            original_size,
            compressed_size,
            compression_level,
            checksum: 0, // Will be calculated separately
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Serialize header to bytes
    #[allow(dead_code)]
    pub fn serialize(&self) -> Vec<u8> {
        let serialized = BlockHeaderSerialized {
            blockid: self.blockid.to_u64(),
            block_type: self.block_type,
            original_size: self.original_size as u64,
            compressed_size: self.compressed_size as u64,
            compression_level: self.compression_level,
            checksum: self.checksum,
            timestamp: self.timestamp,
            padding: [0; 3],
        };

        // Convert to bytes
        unsafe {
            let ptr = &serialized as *const BlockHeaderSerialized as *const u8;
            std::slice::from_raw_parts(ptr, std::mem::size_of::<BlockHeaderSerialized>()).to_vec()
        }
    }

    /// Deserialize header from bytes
    #[allow(dead_code)]
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        if data.len() < std::mem::size_of::<BlockHeaderSerialized>() {
            return Err("Invalid header size".to_string());
        }

        // Convert from bytes
        let serialized: BlockHeaderSerialized = unsafe {
            let ptr = data.as_ptr() as *const BlockHeaderSerialized;
            ptr.read()
        };

        Ok(BlockHeader {
            blockid: BlockId::from_u64(serialized.blockid),
            block_type: serialized.block_type,
            original_size: serialized.original_size as usize,
            compressed_size: serialized.compressed_size as usize,
            compression_level: serialized.compression_level,
            checksum: serialized.checksum,
            timestamp: serialized.timestamp,
        })
    }

    /// Get header size in bytes
    pub fn size() -> usize {
        std::mem::size_of::<BlockHeaderSerialized>()
    }
}

/// Matrix metadata for export/import
#[derive(Debug, Clone)]
pub struct MatrixMetadataExport {
    pub matrix_id: u64,
    pub original_rows: usize,
    pub original_cols: usize,
    pub block_count: usize,
    pub compression_algorithm: CompressionAlgorithm,
    pub block_size: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub compression_ratio: f64,
    pub block_map: Vec<(BlockId, BlockType)>,
}

/// Block size information
#[derive(Debug, Clone)]
pub struct BlockSizeInfo {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub space_savings: usize,
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
