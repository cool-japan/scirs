//! Access pattern tracking for adaptive memory compression
//!
//! This module tracks how blocks are accessed to optimize compression
//! strategies and caching decisions based on temporal and spatial locality.

use super::cache::BlockId;
use std::collections::{HashMap, VecDeque};

/// Type of access
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Access pattern tracking
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct AccessTracker {
    pub access_patterns: HashMap<BlockId, AccessPattern>,
    pub temporal_patterns: VecDeque<AccessEvent>,
    pub spatial_patterns: HashMap<usize, Vec<usize>>, // row -> accessed columns
    pub sequential_threshold: usize,
    max_temporal_history: usize,
    max_spatial_history: usize,
}

/// Access pattern for a block
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct AccessPattern {
    pub access_count: usize,
    pub last_access: u64,
    pub access_frequency: f64,
    pub sequential_accesses: usize,
    pub random_accesses: usize,
    pub temporal_locality: f64,
    pub spatial_locality: f64,
    first_access: u64,
    access_intervals: VecDeque<u64>,
}

/// Access event for pattern analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct AccessEvent {
    pub blockid: BlockId,
    pub timestamp: u64,
    pub access_type: AccessType,
    duration: Option<u64>,
    bytes_accessed: usize,
}

impl AccessTracker {
    /// Create a new access tracker
    pub fn new() -> Self {
        Self {
            access_patterns: HashMap::new(),
            temporal_patterns: VecDeque::new(),
            spatial_patterns: HashMap::new(),
            sequential_threshold: 3,
            max_temporal_history: 10000,
            max_spatial_history: 1000,
        }
    }

    /// Record an access event
    pub fn record_access(
        &mut self,
        block_id: BlockId,
        access_type: AccessType,
        bytes_accessed: usize,
    ) {
        let timestamp = Self::current_timestamp();

        // Record temporal event
        let event = AccessEvent {
            blockid: block_id.clone(),
            timestamp,
            access_type,
            duration: None,
            bytes_accessed,
        };

        self.temporal_patterns.push_back(event);

        // Trim temporal history
        while self.temporal_patterns.len() > self.max_temporal_history {
            self.temporal_patterns.pop_front();
        }

        // Update spatial patterns
        self.spatial_patterns
            .entry(block_id.block_row)
            .or_insert_with(Vec::new)
            .push(block_id.block_col);

        // Trim spatial history
        for column_list in self.spatial_patterns.values_mut() {
            if column_list.len() > self.max_spatial_history {
                column_list.drain(0..column_list.len() - self.max_spatial_history);
            }
        }

        // Update or create access pattern
        let pattern = self
            .access_patterns
            .entry(block_id.clone())
            .or_insert_with(|| AccessPattern::new(timestamp));

        pattern.update_access(timestamp, bytes_accessed);
        pattern.update_locality_metrics(&self.temporal_patterns, &self.spatial_patterns, &block_id);
        pattern.classify_access_type(
            &self.temporal_patterns,
            &block_id,
            self.sequential_threshold,
        );
    }

    /// Get access pattern for a block
    pub fn get_pattern(&self, block_id: &BlockId) -> Option<&AccessPattern> {
        self.access_patterns.get(block_id)
    }

    /// Predict next likely accessed blocks
    pub fn predict_next_accesses(&self, current_block: &BlockId, lookahead: usize) -> Vec<BlockId> {
        let mut predictions = Vec::new();

        // Spatial prediction based on recent access patterns
        if let Some(recent_columns) = self.spatial_patterns.get(&current_block.block_row) {
            let recent_cols: Vec<usize> = recent_columns
                .iter()
                .rev()
                .take(lookahead)
                .copied()
                .collect();

            for &col in &recent_cols {
                if col != current_block.block_col {
                    predictions.push(BlockId {
                        matrixid: current_block.matrixid,
                        block_row: current_block.block_row,
                        block_col: col,
                    });
                }
            }
        }

        // Temporal prediction based on sequential patterns
        if let Some(pattern) = self.get_pattern(current_block) {
            if pattern.is_sequential() {
                // Predict next sequential blocks
                for i in 1..=lookahead {
                    predictions.push(BlockId {
                        matrixid: current_block.matrixid,
                        block_row: current_block.block_row,
                        block_col: current_block.block_col + i,
                    });
                }
            }
        }

        predictions.truncate(lookahead);
        predictions
    }

    /// Get blocks with high access frequency
    pub fn get_hot_blocks(&self, limit: usize) -> Vec<(BlockId, f64)> {
        let mut blocks: Vec<_> = self
            .access_patterns
            .iter()
            .map(|(id, pattern)| (id.clone(), pattern.access_frequency))
            .collect();

        blocks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        blocks.truncate(limit);
        blocks
    }

    /// Get blocks with low access frequency (candidates for eviction)
    pub fn get_cold_blocks(&self, limit: usize) -> Vec<(BlockId, f64)> {
        let mut blocks: Vec<_> = self
            .access_patterns
            .iter()
            .map(|(id, pattern)| (id.clone(), pattern.access_frequency))
            .collect();

        blocks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        blocks.truncate(limit);
        blocks
    }

    /// Analyze overall access patterns
    pub fn analyze_patterns(&self) -> AccessAnalysis {
        let total_blocks = self.access_patterns.len();
        let total_accesses: usize = self.access_patterns.values().map(|p| p.access_count).sum();

        let sequential_blocks = self
            .access_patterns
            .values()
            .filter(|p| p.is_sequential())
            .count();

        let random_blocks = total_blocks - sequential_blocks;

        let avg_temporal_locality = if total_blocks > 0 {
            self.access_patterns
                .values()
                .map(|p| p.temporal_locality)
                .sum::<f64>()
                / total_blocks as f64
        } else {
            0.0
        };

        let avg_spatial_locality = if total_blocks > 0 {
            self.access_patterns
                .values()
                .map(|p| p.spatial_locality)
                .sum::<f64>()
                / total_blocks as f64
        } else {
            0.0
        };

        AccessAnalysis {
            total_blocks,
            total_accesses,
            sequential_blocks,
            random_blocks,
            avg_temporal_locality,
            avg_spatial_locality,
            hot_blocks_threshold: self.calculate_hot_blocks_threshold(),
            cold_blocks_threshold: self.calculate_cold_blocks_threshold(),
        }
    }

    /// Calculate threshold for hot blocks
    fn calculate_hot_blocks_threshold(&self) -> f64 {
        if self.access_patterns.is_empty() {
            return 0.0;
        }

        let frequencies: Vec<f64> = self
            .access_patterns
            .values()
            .map(|p| p.access_frequency)
            .collect();

        // Use 90th percentile as hot threshold
        let mut sorted_freq = frequencies;
        sorted_freq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (sorted_freq.len() as f64 * 0.9) as usize;
        sorted_freq.get(index).copied().unwrap_or(0.0)
    }

    /// Calculate threshold for cold blocks
    fn calculate_cold_blocks_threshold(&self) -> f64 {
        if self.access_patterns.is_empty() {
            return 0.0;
        }

        let frequencies: Vec<f64> = self
            .access_patterns
            .values()
            .map(|p| p.access_frequency)
            .collect();

        // Use 10th percentile as cold threshold
        let mut sorted_freq = frequencies;
        sorted_freq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (sorted_freq.len() as f64 * 0.1) as usize;
        sorted_freq.get(index).copied().unwrap_or(0.0)
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Clear old access patterns
    pub fn cleanup_old_patterns(&mut self, max_age_seconds: u64) {
        let current_time = Self::current_timestamp();
        let cutoff_time = current_time.saturating_sub(max_age_seconds);

        // Remove old access patterns
        self.access_patterns
            .retain(|_, pattern| pattern.last_access >= cutoff_time);

        // Remove old temporal events
        self.temporal_patterns
            .retain(|event| event.timestamp >= cutoff_time);
    }

    /// Get access statistics
    pub fn get_statistics(&self) -> AccessStatistics {
        let total_events = self.temporal_patterns.len();
        let read_events = self
            .temporal_patterns
            .iter()
            .filter(|e| matches!(e.access_type, AccessType::Read | AccessType::ReadWrite))
            .count();
        let write_events = self
            .temporal_patterns
            .iter()
            .filter(|e| matches!(e.access_type, AccessType::Write | AccessType::ReadWrite))
            .count();

        AccessStatistics {
            total_tracked_blocks: self.access_patterns.len(),
            total_access_events: total_events,
            read_events,
            write_events,
            spatial_localities: self.spatial_patterns.len(),
            avg_accesses_per_block: if self.access_patterns.is_empty() {
                0.0
            } else {
                self.access_patterns
                    .values()
                    .map(|p| p.access_count as f64)
                    .sum::<f64>()
                    / self.access_patterns.len() as f64
            },
        }
    }
}

impl AccessPattern {
    /// Create a new access pattern
    fn new(timestamp: u64) -> Self {
        Self {
            access_count: 0,
            last_access: timestamp,
            access_frequency: 0.0,
            sequential_accesses: 0,
            random_accesses: 0,
            temporal_locality: 0.0,
            spatial_locality: 0.0,
            first_access: timestamp,
            access_intervals: VecDeque::new(),
        }
    }

    /// Update access information
    fn update_access(&mut self, timestamp: u64, _bytes_accessed: usize) {
        if self.access_count > 0 {
            let interval = timestamp.saturating_sub(self.last_access);
            self.access_intervals.push_back(interval);

            // Keep only recent intervals for frequency calculation
            while self.access_intervals.len() > 100 {
                self.access_intervals.pop_front();
            }
        }

        self.access_count += 1;
        self.last_access = timestamp;

        // Update access frequency
        self.update_frequency(timestamp);
    }

    /// Update access frequency
    fn update_frequency(&mut self, current_timestamp: u64) {
        let time_span = current_timestamp.saturating_sub(self.first_access).max(1);
        self.access_frequency = self.access_count as f64 / time_span as f64;
    }

    /// Update locality metrics
    fn update_locality_metrics(
        &mut self,
        temporal_patterns: &VecDeque<AccessEvent>,
        spatial_patterns: &HashMap<usize, Vec<usize>>,
        block_id: &BlockId,
    ) {
        // Calculate temporal locality based on recent accesses
        let recent_accesses: Vec<_> = temporal_patterns
            .iter()
            .rev()
            .take(10)
            .filter(|e| e.blockid == *block_id)
            .collect();

        if recent_accesses.len() > 1 {
            let intervals: Vec<u64> = recent_accesses
                .windows(2)
                .map(|window| window[0].timestamp.saturating_sub(window[1].timestamp))
                .collect();

            let avg_interval = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
            self.temporal_locality = 1.0 / (1.0 + avg_interval / 100.0); // Normalize to 0-1
        }

        // Calculate spatial locality based on nearby block accesses
        if let Some(row_accesses) = spatial_patterns.get(&block_id.block_row) {
            let nearby_accesses = row_accesses
                .iter()
                .filter(|&&col| {
                    let distance = (col as i32 - block_id.block_col as i32).abs();
                    distance <= 5 // Within 5 blocks is considered spatial locality
                })
                .count();

            self.spatial_locality = nearby_accesses as f64 / row_accesses.len().max(1) as f64;
        }
    }

    /// Classify access type based on pattern
    fn classify_access_type(
        &mut self,
        temporal_patterns: &VecDeque<AccessEvent>,
        block_id: &BlockId,
        sequential_threshold: usize,
    ) {
        // Look for sequential access patterns
        let recent_same_row: Vec<_> = temporal_patterns
            .iter()
            .rev()
            .take(sequential_threshold * 2)
            .filter(|e| e.blockid.block_row == block_id.block_row)
            .collect();

        if recent_same_row.len() >= sequential_threshold {
            let columns: Vec<usize> = recent_same_row
                .iter()
                .map(|e| e.blockid.block_col)
                .collect();

            let is_sequential = columns
                .windows(2)
                .all(|window| window[1] == window[0] + 1 || window[1] == window[0]);

            if is_sequential {
                self.sequential_accesses += 1;
            } else {
                self.random_accesses += 1;
            }
        }
    }

    /// Check if this pattern indicates sequential access
    pub fn is_sequential(&self) -> bool {
        if self.sequential_accesses + self.random_accesses == 0 {
            return false;
        }
        let sequential_ratio = self.sequential_accesses as f64
            / (self.sequential_accesses + self.random_accesses) as f64;
        sequential_ratio > 0.7
    }

    /// Get access rate (accesses per second)
    pub fn access_rate(&self) -> f64 {
        self.access_frequency
    }

    /// Get predictability score
    pub fn predictability_score(&self) -> f64 {
        (self.temporal_locality + self.spatial_locality) / 2.0
    }
}

impl AccessEvent {
    /// Create a new access event
    pub fn new(block_id: BlockId, access_type: AccessType, bytes_accessed: usize) -> Self {
        Self {
            blockid: block_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            access_type,
            duration: None,
            bytes_accessed,
        }
    }

    /// Set access duration
    pub fn with_duration(mut self, duration: u64) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Get access bandwidth (bytes per second)
    pub fn bandwidth(&self) -> Option<f64> {
        self.duration.map(|d| {
            if d > 0 {
                self.bytes_accessed as f64 / d as f64
            } else {
                0.0
            }
        })
    }
}

/// Analysis results for access patterns
#[derive(Debug, Clone)]
pub struct AccessAnalysis {
    pub total_blocks: usize,
    pub total_accesses: usize,
    pub sequential_blocks: usize,
    pub random_blocks: usize,
    pub avg_temporal_locality: f64,
    pub avg_spatial_locality: f64,
    pub hot_blocks_threshold: f64,
    pub cold_blocks_threshold: f64,
}

/// Statistics for access tracking
#[derive(Debug, Clone)]
pub struct AccessStatistics {
    pub total_tracked_blocks: usize,
    pub total_access_events: usize,
    pub read_events: usize,
    pub write_events: usize,
    pub spatial_localities: usize,
    pub avg_accesses_per_block: f64,
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}
