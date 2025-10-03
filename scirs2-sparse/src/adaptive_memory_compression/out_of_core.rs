//! Out-of-Core Memory Management
//!
//! This module handles storage and retrieval of compressed matrix blocks
//! when they exceed available memory capacity.

use super::cache::BlockId;
use super::compressed_data::{BlockHeader, BlockHeaderSerialized, BlockType, CompressedBlock};
use crate::error::{SparseError, SparseResult};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Manager for out-of-core block storage
#[derive(Debug)]
pub struct OutOfCoreManager {
    temp_dir: String,
    file_counter: AtomicUsize,
    active_files: HashMap<BlockId, String>,
    disk_usage: AtomicUsize,
}

impl OutOfCoreManager {
    /// Create a new out-of-core manager
    pub fn new(temp_dir: &str) -> SparseResult<Self> {
        std::fs::create_dir_all(temp_dir).map_err(SparseError::IoError)?;

        Ok(Self {
            temp_dir: temp_dir.to_string(),
            file_counter: AtomicUsize::new(0),
            active_files: HashMap::new(),
            disk_usage: AtomicUsize::new(0),
        })
    }

    /// Write compressed block to disk
    pub fn write_block_to_disk(&mut self, block: &CompressedBlock) -> SparseResult<String> {
        let file_id = self.file_counter.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("block_{}_{file_id}.dat", block.blockid);
        let filepath = Path::new(&self.temp_dir).join(&file_name);

        // Create and write to file
        let mut file = File::create(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to create file {filepath:?}: {e}")))?;

        // Write block header
        let header = BlockHeader::new(
            block.blockid.clone(),
            block.block_type,
            block.original_size,
            block.compressed_data.len(),
            block.compression_level,
        );

        file.write_all(&header.serialize())
            .map_err(|e| SparseError::Io(format!("Failed to write header: {e}")))?;

        // Write compressed data
        file.write_all(&block.compressed_data)
            .map_err(|e| SparseError::Io(format!("Failed to write data: {e}")))?;

        file.flush()
            .map_err(|e| SparseError::Io(format!("Failed to flush file: {e}")))?;

        // Update disk usage
        let file_size = BlockHeader::size() + block.compressed_data.len();
        self.disk_usage.fetch_add(file_size, Ordering::Relaxed);

        // Track the file
        self.active_files
            .insert(block.blockid.clone(), file_name.clone());

        Ok(file_name)
    }

    /// Read compressed block from disk
    pub fn read_block_from_disk(&self, block_id: &BlockId) -> SparseResult<CompressedBlock> {
        let file_name = self.active_files.get(block_id).ok_or_else(|| {
            SparseError::BlockNotFound(format!("Block {block_id} not found on disk"))
        })?;

        let filepath = Path::new(&self.temp_dir).join(file_name);
        let mut file = File::open(&filepath)
            .map_err(|e| SparseError::Io(format!("Failed to open file {filepath:?}: {e}")))?;

        // Read block header
        let mut header_bytes = vec![0u8; BlockHeader::size()];
        file.read_exact(&mut header_bytes)
            .map_err(|e| SparseError::Io(format!("Failed to read header: {e}")))?;

        let header = BlockHeader::deserialize(&header_bytes)?;

        // Read compressed data
        let mut compressed_data = vec![0u8; header.compressed_size];
        file.read_exact(&mut compressed_data)
            .map_err(|e| SparseError::Io(format!("Failed to read data: {e}")))?;

        Ok(CompressedBlock::new(
            header.blockid,
            self.block_type_from_u8(header.block_type)?,
            compressed_data,
            header.original_size,
            header.compression_level,
        ))
    }

    /// Check if block exists on disk
    pub fn has_block(&self, block_id: &BlockId) -> bool {
        if let Some(file_name) = self.active_files.get(block_id) {
            let filepath = Path::new(&self.temp_dir).join(file_name);
            filepath.exists()
        } else {
            false
        }
    }

    /// Remove block from disk
    pub fn remove_block(&mut self, block_id: &BlockId) -> SparseResult<()> {
        if let Some(file_name) = self.active_files.remove(block_id) {
            let filepath = Path::new(&self.temp_dir).join(&file_name);

            // Get file size before removal for disk usage tracking
            if let Ok(metadata) = std::fs::metadata(&filepath) {
                let file_size = metadata.len() as usize;
                self.disk_usage.fetch_sub(file_size, Ordering::Relaxed);
            }

            // Remove file from disk
            std::fs::remove_file(&filepath)
                .map_err(|e| SparseError::Io(format!("Failed to remove file {filepath:?}: {e}")))?;
        }
        Ok(())
    }

    /// Get total disk usage in bytes
    pub fn get_disk_usage(&self) -> usize {
        self.disk_usage.load(Ordering::Relaxed)
    }

    /// Get number of active files
    pub fn get_active_file_count(&self) -> usize {
        self.active_files.len()
    }

    /// List all active blocks
    pub fn list_active_blocks(&self) -> Vec<BlockId> {
        self.active_files.keys().cloned().collect()
    }

    /// Get storage efficiency (actual vs estimated disk usage)
    pub fn get_storage_efficiency(&self) -> f64 {
        let actual_usage = self.calculate_actual_disk_usage();
        let tracked_usage = self.get_disk_usage();

        if tracked_usage > 0 {
            actual_usage as f64 / tracked_usage as f64
        } else {
            1.0
        }
    }

    /// Cleanup all temporary files
    pub fn cleanup(&mut self) -> SparseResult<()> {
        let mut total_removed = 0;

        for file_name in self.active_files.values() {
            let filepath = Path::new(&self.temp_dir).join(file_name);
            if let Ok(metadata) = std::fs::metadata(&filepath) {
                total_removed += metadata.len() as usize;
            }
            let _ = std::fs::remove_file(&filepath); // Ignore errors during cleanup
        }

        self.active_files.clear();
        self.disk_usage.store(0, Ordering::Relaxed);

        // Try to remove the temp directory if it's empty
        let _ = std::fs::remove_dir(&self.temp_dir);

        Ok(())
    }

    /// Compact storage by defragmenting files
    pub fn compact_storage(&mut self) -> SparseResult<usize> {
        let mut blocks_to_rewrite = Vec::new();

        // Read all blocks from disk
        let block_ids: Vec<_> = self.active_files.keys().cloned().collect();
        for block_id in block_ids {
            match self.read_block_from_disk(&block_id) {
                Ok(block) => blocks_to_rewrite.push(block),
                Err(_) => {
                    // Remove invalid entries
                    self.active_files.remove(&block_id);
                }
            }
        }

        // Clear all files
        self.cleanup()?;

        // Recreate the temp directory
        std::fs::create_dir_all(&self.temp_dir)
            .map_err(|e| SparseError::Io(format!("Failed to recreate temp directory: {}", e)))?;

        // Rewrite all valid blocks
        let mut compacted_blocks = 0;
        for block in blocks_to_rewrite {
            if self.write_block_to_disk(&block).is_ok() {
                compacted_blocks += 1;
            }
        }

        Ok(compacted_blocks)
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> OutOfCoreStats {
        OutOfCoreStats {
            total_blocks: self.active_files.len(),
            disk_usage: self.get_disk_usage(),
            actual_disk_usage: self.calculate_actual_disk_usage(),
            storage_efficiency: self.get_storage_efficiency(),
            temp_directory: self.temp_dir.clone(),
            fragmentation_ratio: self.calculate_fragmentation_ratio(),
        }
    }

    /// Set temp directory (for configuration updates)
    pub fn set_temp_directory(&mut self, new_temp_dir: &str) -> SparseResult<()> {
        // Create new directory
        std::fs::create_dir_all(new_temp_dir).map_err(SparseError::IoError)?;

        // Move existing files if any
        if !self.active_files.is_empty() {
            self.migrate_files_to_directory(new_temp_dir)?;
        }

        self.temp_dir = new_temp_dir.to_string();
        Ok(())
    }

    // Private helper methods

    fn block_type_from_u8(&self, value: u8) -> SparseResult<BlockType> {
        match value {
            0 => Ok(BlockType::IndPtr),
            1 => Ok(BlockType::Indices),
            2 => Ok(BlockType::Data),
            3 => Ok(BlockType::Combined),
            4 => Ok(BlockType::Metadata),
            _ => Err(SparseError::InvalidFormat(format!(
                "Unknown block type: {value}"
            ))),
        }
    }

    fn calculate_actual_disk_usage(&self) -> usize {
        let mut total_size = 0;
        for file_name in self.active_files.values() {
            let filepath = Path::new(&self.temp_dir).join(file_name);
            if let Ok(metadata) = std::fs::metadata(&filepath) {
                total_size += metadata.len() as usize;
            }
        }
        total_size
    }

    fn calculate_fragmentation_ratio(&self) -> f64 {
        let file_count = self.active_files.len();
        if file_count == 0 {
            return 0.0;
        }

        let average_file_size = self.get_disk_usage() as f64 / file_count as f64;
        let mut variance = 0.0;

        for file_name in self.active_files.values() {
            let filepath = Path::new(&self.temp_dir).join(file_name);
            if let Ok(metadata) = std::fs::metadata(&filepath) {
                let size_diff = metadata.len() as f64 - average_file_size;
                variance += size_diff * size_diff;
            }
        }

        if file_count > 1 {
            variance /= (file_count - 1) as f64;
            (variance.sqrt() / average_file_size).min(1.0)
        } else {
            0.0
        }
    }

    fn migrate_files_to_directory(&mut self, new_dir: &str) -> SparseResult<()> {
        let old_dir = self.temp_dir.clone();
        let mut migration_errors = 0;

        for (block_id, file_name) in &self.active_files.clone() {
            let old_path = Path::new(&old_dir).join(file_name);
            let new_path = Path::new(new_dir).join(file_name);

            if std::fs::copy(&old_path, &new_path).is_err() {
                migration_errors += 1;
            } else {
                let _ = std::fs::remove_file(&old_path);
            }
        }

        if migration_errors > 0 {
            return Err(SparseError::Io(format!(
                "Failed to migrate {migration_errors} files to new directory"
            )));
        }

        Ok(())
    }
}

/// Statistics for out-of-core storage
#[derive(Debug, Clone)]
pub struct OutOfCoreStats {
    pub total_blocks: usize,
    pub disk_usage: usize,
    pub actual_disk_usage: usize,
    pub storage_efficiency: f64,
    pub temp_directory: String,
    pub fragmentation_ratio: f64,
}

impl std::fmt::Display for OutOfCoreStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OutOfCore Stats:\n\
             - Total blocks: {}\n\
             - Disk usage: {} bytes\n\
             - Actual usage: {} bytes\n\
             - Storage efficiency: {:.2}%\n\
             - Fragmentation: {:.2}%\n\
             - Temp directory: {}",
            self.total_blocks,
            self.disk_usage,
            self.actual_disk_usage,
            self.storage_efficiency * 100.0,
            self.fragmentation_ratio * 100.0,
            self.temp_directory
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_out_of_core_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = OutOfCoreManager::new(temp_dir.path().to_str().unwrap());
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert_eq!(manager.get_active_file_count(), 0);
        assert_eq!(manager.get_disk_usage(), 0);
    }

    #[test]
    fn test_block_write_read_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = OutOfCoreManager::new(temp_dir.path().to_str().unwrap()).unwrap();

        let block_id = BlockId::new(1, 0, 0);
        let original_block = CompressedBlock::new(
            block_id.clone(),
            BlockType::Data,
            vec![1, 2, 3, 4, 5],
            100,
            1,
        );

        // Write block to disk
        let file_name = manager.write_block_to_disk(&original_block).unwrap();
        assert!(!file_name.is_empty());
        assert_eq!(manager.get_active_file_count(), 1);
        assert!(manager.has_block(&block_id));

        // Read block from disk
        let read_block = manager.read_block_from_disk(&block_id).unwrap();
        assert_eq!(read_block.blockid, original_block.blockid);
        assert_eq!(read_block.block_type, original_block.block_type);
        assert_eq!(read_block.compressed_data, original_block.compressed_data);
        assert_eq!(read_block.original_size, original_block.original_size);
        assert_eq!(
            read_block.compression_level,
            original_block.compression_level
        );
    }

    #[test]
    fn test_block_removal() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = OutOfCoreManager::new(temp_dir.path().to_str().unwrap()).unwrap();

        let block_id = BlockId::new(1, 0, 0);
        let block = CompressedBlock::new(
            block_id.clone(),
            BlockType::Data,
            vec![1, 2, 3, 4, 5],
            100,
            1,
        );

        // Write and then remove block
        manager.write_block_to_disk(&block).unwrap();
        assert!(manager.has_block(&block_id));

        manager.remove_block(&block_id).unwrap();
        assert!(!manager.has_block(&block_id));
        assert_eq!(manager.get_active_file_count(), 0);
    }

    #[test]
    fn test_storage_stats() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = OutOfCoreManager::new(temp_dir.path().to_str().unwrap()).unwrap();

        let stats = manager.get_storage_stats();
        assert_eq!(stats.total_blocks, 0);
        assert_eq!(stats.disk_usage, 0);

        // Add a block and check stats
        let block = CompressedBlock::new(
            BlockId::new(1, 0, 0),
            BlockType::Data,
            vec![1, 2, 3, 4, 5],
            100,
            1,
        );
        manager.write_block_to_disk(&block).unwrap();

        let stats = manager.get_storage_stats();
        assert_eq!(stats.total_blocks, 1);
        assert!(stats.disk_usage > 0);
        assert!(stats.storage_efficiency > 0.0);
    }

    #[test]
    fn test_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = OutOfCoreManager::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add multiple blocks
        for i in 0..5 {
            let block = CompressedBlock::new(
                BlockId::new(1, i, 0),
                BlockType::Data,
                vec![1, 2, 3, 4, 5],
                100,
                1,
            );
            manager.write_block_to_disk(&block).unwrap();
        }

        assert_eq!(manager.get_active_file_count(), 5);

        // Cleanup all blocks
        manager.cleanup().unwrap();
        assert_eq!(manager.get_active_file_count(), 0);
        assert_eq!(manager.get_disk_usage(), 0);
    }

    #[test]
    fn test_storage_compaction() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = OutOfCoreManager::new(temp_dir.path().to_str().unwrap()).unwrap();

        // Add blocks
        for i in 0..3 {
            let block = CompressedBlock::new(
                BlockId::new(1, i, 0),
                BlockType::Data,
                vec![1, 2, 3, 4, 5],
                100,
                1,
            );
            manager.write_block_to_disk(&block).unwrap();
        }

        let initial_count = manager.get_active_file_count();
        let compacted = manager.compact_storage().unwrap();

        assert_eq!(compacted, initial_count);
        assert_eq!(manager.get_active_file_count(), initial_count);
    }
}
