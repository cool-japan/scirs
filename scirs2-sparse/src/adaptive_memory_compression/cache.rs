//! Block cache management for adaptive memory compression
//!
//! This module handles caching of frequently accessed compressed blocks
//! with LRU eviction and access pattern tracking.

use super::access_tracking::AccessType;
use std::collections::{HashMap, VecDeque};

/// Block identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BlockId {
    pub matrixid: u64,
    pub block_row: usize,
    pub block_col: usize,
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}-{}", self.matrixid, self.block_row, self.block_col)
    }
}

impl BlockId {
    /// Create a new block identifier
    pub fn new(matrix_id: u64, block_row: usize, block_col: usize) -> Self {
        Self {
            matrixid: matrix_id,
            block_row,
            block_col,
        }
    }

    /// Convert BlockId to u64 for serialization (using a hash-like approach)
    pub fn to_u64(&self) -> u64 {
        // Simple hash combining the fields
        self.matrixid
            .wrapping_mul(1000000)
            .wrapping_add((self.block_row as u64) * 1000)
            .wrapping_add(self.block_col as u64)
    }

    /// Create BlockId from u64 (for deserialization)
    pub fn from_u64(value: u64) -> Self {
        // This is a simplified reverse operation - in practice you'd want a proper bijection
        let matrixid = value / 1000000;
        let remainder = value % 1000000;
        let block_row = (remainder / 1000) as usize;
        let block_col = (remainder % 1000) as usize;
        Self {
            matrixid,
            block_row,
            block_col,
        }
    }

    /// Get a unique string representation
    pub fn as_string(&self) -> String {
        format!("{}_{}-{}", self.matrixid, self.block_row, self.block_col)
    }

    /// Parse BlockId from string representation
    pub fn from_string(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('_').collect();
        if parts.len() != 2 {
            return Err("Invalid BlockId string format".to_string());
        }

        let matrix_id = parts[0].parse::<u64>().map_err(|_| "Invalid matrix ID")?;

        let coords: Vec<&str> = parts[1].split('-').collect();
        if coords.len() != 2 {
            return Err("Invalid coordinate format".to_string());
        }

        let block_row = coords[0]
            .parse::<usize>()
            .map_err(|_| "Invalid block row")?;
        let block_col = coords[1]
            .parse::<usize>()
            .map_err(|_| "Invalid block column")?;

        Ok(Self {
            matrixid: matrix_id,
            block_row,
            block_col,
        })
    }
}

/// Cached block information
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct CachedBlock {
    pub data: Vec<u8>,
    pub compressed: bool,
    pub access_count: usize,
    pub last_access: u64,
    pub compression_level: u8,
    pub original_size: usize,
    pub compressed_size: usize,
}

impl CachedBlock {
    /// Create a new cached block
    pub fn new(data: Vec<u8>, compressed: bool, compression_level: u8) -> Self {
        let size = data.len();
        Self {
            data,
            compressed,
            access_count: 1,
            last_access: Self::current_timestamp(),
            compression_level,
            original_size: if compressed { 0 } else { size },
            compressed_size: if compressed { size } else { 0 },
        }
    }

    /// Update access information
    pub fn update_access(&mut self) {
        self.access_count += 1;
        self.last_access = Self::current_timestamp();
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            1.0
        } else {
            self.compressed_size as f64 / self.original_size as f64
        }
    }

    /// Get access frequency (accesses per second)
    pub fn access_frequency(&self) -> f64 {
        let current_time = Self::current_timestamp();
        let time_diff = current_time.saturating_sub(self.last_access).max(1);
        self.access_count as f64 / time_diff as f64
    }

    /// Get data size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if block is recently accessed
    pub fn is_recently_accessed(&self, threshold_seconds: u64) -> bool {
        let current_time = Self::current_timestamp();
        current_time.saturating_sub(self.last_access) < threshold_seconds
    }
}

/// Block cache for frequently accessed data
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct BlockCache {
    pub cache: HashMap<BlockId, CachedBlock>,
    pub access_order: VecDeque<BlockId>,
    _maxsize: usize,
    pub current_size: usize,
    max_cache_size: usize,
    hit_count: usize,
    miss_count: usize,
}

impl BlockCache {
    /// Create a new block cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            _maxsize: max_size,
            current_size: 0,
            max_cache_size: max_size,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Insert a block into the cache
    pub fn insert(&mut self, block_id: BlockId, block: CachedBlock) {
        let block_size = block.size();

        // Remove existing block if present
        if let Some(existing) = self.cache.remove(&block_id) {
            self.current_size = self.current_size.saturating_sub(existing.size());
            // Remove from access order
            if let Some(pos) = self.access_order.iter().position(|id| id == &block_id) {
                self.access_order.remove(pos);
            }
        }

        // Evict blocks if necessary
        while self.current_size + block_size > self.max_cache_size && !self.access_order.is_empty()
        {
            self.evict_lru();
        }

        // Insert new block
        if block_size <= self.max_cache_size {
            self.cache.insert(block_id.clone(), block);
            self.access_order.push_back(block_id);
            self.current_size += block_size;
        }
    }

    /// Get a block from the cache
    pub fn get(&mut self, block_id: &BlockId) -> Option<&CachedBlock> {
        if let Some(block) = self.cache.get(block_id) {
            self.hit_count += 1;
            // Move to back of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|id| id == block_id) {
                self.access_order.remove(pos);
                self.access_order.push_back(block_id.clone());
            }
            Some(block)
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Get a mutable reference to a block
    pub fn get_mut(&mut self, block_id: &BlockId) -> Option<&mut CachedBlock> {
        if self.cache.contains_key(block_id) {
            self.hit_count += 1;
            // Move to back of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|id| id == block_id) {
                self.access_order.remove(pos);
                self.access_order.push_back(block_id.clone());
            }
            self.cache.get_mut(block_id)
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Remove a block from the cache
    pub fn remove(&mut self, block_id: &BlockId) -> Option<CachedBlock> {
        if let Some(block) = self.cache.remove(block_id) {
            self.current_size = self.current_size.saturating_sub(block.size());
            // Remove from access order
            if let Some(pos) = self.access_order.iter().position(|id| id == block_id) {
                self.access_order.remove(pos);
            }
            Some(block)
        } else {
            None
        }
    }

    /// Check if a block exists in the cache
    pub fn contains(&self, block_id: &BlockId) -> bool {
        self.cache.contains_key(block_id)
    }

    /// Evict the least recently used block
    fn evict_lru(&mut self) {
        if let Some(lru_id) = self.access_order.pop_front() {
            if let Some(block) = self.cache.remove(&lru_id) {
                self.current_size = self.current_size.saturating_sub(block.size());
            }
        }
    }

    /// Clear all blocks from the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.current_size = 0;
        self.hit_count = 0;
        self.miss_count = 0;
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let total_accesses = self.hit_count + self.miss_count;
        let hit_rate = if total_accesses > 0 {
            self.hit_count as f64 / total_accesses as f64
        } else {
            0.0
        };

        CacheStats {
            total_blocks: self.cache.len(),
            current_size_bytes: self.current_size,
            max_size_bytes: self.max_cache_size,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            hit_rate,
            utilization: self.current_size as f64 / self.max_cache_size as f64,
        }
    }

    /// Get blocks sorted by access frequency
    pub fn get_most_accessed_blocks(&self, limit: usize) -> Vec<(BlockId, usize)> {
        let mut blocks: Vec<_> = self
            .cache
            .iter()
            .map(|(id, block)| (id.clone(), block.access_count))
            .collect();

        blocks.sort_by(|a, b| b.1.cmp(&a.1));
        blocks.truncate(limit);
        blocks
    }

    /// Get total number of blocks in cache
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get current memory usage ratio
    pub fn memory_usage_ratio(&self) -> f64 {
        self.current_size as f64 / self.max_cache_size as f64
    }

    /// Force eviction of old blocks based on age
    pub fn evict_old_blocks(&mut self, max_age_seconds: u64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let old_blocks: Vec<BlockId> = self
            .cache
            .iter()
            .filter(|(_, block)| current_time.saturating_sub(block.last_access) > max_age_seconds)
            .map(|(id, _)| id.clone())
            .collect();

        for block_id in old_blocks {
            self.remove(&block_id);
        }
    }

    /// Prefetch blocks based on spatial locality
    pub fn suggest_prefetch_candidates(
        &self,
        current_block: &BlockId,
        lookahead: usize,
    ) -> Vec<BlockId> {
        let mut candidates = Vec::new();

        // Suggest adjacent blocks
        for row_offset in 0..=lookahead {
            for col_offset in 0..=lookahead {
                if row_offset == 0 && col_offset == 0 {
                    continue; // Skip current block
                }

                let candidate = BlockId {
                    matrixid: current_block.matrixid,
                    block_row: current_block.block_row.saturating_add(row_offset),
                    block_col: current_block.block_col.saturating_add(col_offset),
                };

                if !self.contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }

        candidates
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_blocks: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub hit_count: usize,
    pub miss_count: usize,
    pub hit_rate: f64,
    pub utilization: f64,
}

impl CacheStats {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Cache: {} blocks, {:.1}% utilized, {:.1}% hit rate ({}/{} accesses)",
            self.total_blocks,
            self.utilization * 100.0,
            self.hit_rate * 100.0,
            self.hit_count,
            self.hit_count + self.miss_count
        )
    }
}
