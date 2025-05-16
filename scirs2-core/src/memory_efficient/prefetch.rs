//! Smart prefetching for memory-mapped arrays.
//!
//! This module provides intelligent prefetching capabilities for memory-mapped arrays,
//! which can significantly improve performance for workloads with predictable access patterns.
//! By analyzing access patterns and prefetching blocks that are likely to be needed soon,
//! the system can reduce latency and improve throughput.
//!
//! The prefetching system supports:
//! - Automatic detection of sequential access patterns
//! - Recognition of strided access patterns
//! - Adaptive prefetching based on historical access patterns
//! - Integration with the block cache system to manage prefetched blocks

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::compressed_memmap::CompressedMemMappedArray;
use super::memmap::MemoryMappedArray;
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Types of access patterns that can be detected and prefetched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access (consecutive blocks)
    Sequential,

    /// Strided access (blocks with a fixed stride)
    Strided(usize),

    /// Random access (no discernable pattern)
    Random,

    /// Custom pattern defined by a specific sequence of offsets
    Custom,
}

/// Configuration for the prefetching system.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Whether prefetching is enabled
    pub enabled: bool,

    /// Number of blocks to prefetch ahead of the current access
    pub prefetch_count: usize,

    /// Maximum number of blocks to keep in the prefetch history
    pub history_size: usize,

    /// Minimum number of accesses needed to detect a pattern
    pub min_pattern_length: usize,

    /// Whether to prefetch in a background thread
    pub async_prefetch: bool,

    /// Timeout for prefetch operations (to avoid blocking too long)
    pub prefetch_timeout: Duration,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefetch_count: 2,
            history_size: 32,
            min_pattern_length: 4,
            async_prefetch: true,
            prefetch_timeout: Duration::from_millis(100),
        }
    }
}

/// Builder for prefetch configuration.
#[derive(Debug, Clone)]
pub struct PrefetchConfigBuilder {
    config: PrefetchConfig,
}

impl PrefetchConfigBuilder {
    /// Create a new prefetch config builder with default settings.
    pub fn new() -> Self {
        Self {
            config: PrefetchConfig::default(),
        }
    }

    /// Enable or disable prefetching.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set the number of blocks to prefetch ahead of the current access.
    pub fn prefetch_count(mut self, count: usize) -> Self {
        self.config.prefetch_count = count;
        self
    }

    /// Set the maximum number of blocks to keep in the prefetch history.
    pub fn history_size(mut self, size: usize) -> Self {
        self.config.history_size = size;
        self
    }

    /// Set the minimum number of accesses needed to detect a pattern.
    pub fn min_pattern_length(mut self, length: usize) -> Self {
        self.config.min_pattern_length = length;
        self
    }

    /// Enable or disable asynchronous prefetching.
    pub fn async_prefetch(mut self, async_prefetch: bool) -> Self {
        self.config.async_prefetch = async_prefetch;
        self
    }

    /// Set the timeout for prefetch operations.
    pub fn prefetch_timeout(mut self, timeout: Duration) -> Self {
        self.config.prefetch_timeout = timeout;
        self
    }

    /// Build the prefetch configuration.
    pub fn build(self) -> PrefetchConfig {
        self.config
    }
}

/// Trait for tracking and predicting access patterns.
pub trait AccessPatternTracker {
    /// Record an access to a block.
    fn record_access(&mut self, block_idx: usize);

    /// Predict which blocks will be accessed next.
    fn predict_next_blocks(&self, count: usize) -> Vec<usize>;

    /// Get the current detected access pattern.
    fn current_pattern(&self) -> AccessPattern;

    /// Clear the access history.
    fn clear_history(&mut self);
}

/// Implementation of access pattern tracking.
#[derive(Debug)]
pub struct BlockAccessTracker {
    /// Configuration for the tracker
    config: PrefetchConfig,

    /// History of accessed blocks
    history: VecDeque<usize>,

    /// The currently detected pattern
    current_pattern: AccessPattern,

    /// For strided patterns, the stride value
    stride: Option<usize>,

    /// Last time the pattern was updated
    last_update: Instant,
}

impl BlockAccessTracker {
    /// Create a new block access tracker with the given configuration.
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(config.history_size),
            current_pattern: AccessPattern::Random,
            stride: None,
            last_update: Instant::now(),
        }
    }

    /// Detect the access pattern based on the history.
    fn detect_pattern(&mut self) {
        if self.history.len() < self.config.min_pattern_length {
            // Not enough history to detect a pattern
            self.current_pattern = AccessPattern::Random;
            return;
        }

        // Check for sequential access
        let mut is_sequential = true;
        let mut prev = *self.history.get(0).unwrap();

        for &block_idx in self.history.iter().skip(1) {
            if block_idx != prev + 1 {
                is_sequential = false;
                break;
            }
            prev = block_idx;
        }

        if is_sequential {
            self.current_pattern = AccessPattern::Sequential;
            return;
        }

        // Check for strided access
        let mut is_strided = true;
        let stride = self.history.get(1).unwrap() - self.history.get(0).unwrap();
        prev = *self.history.get(0).unwrap();

        for &block_idx in self.history.iter().skip(1) {
            if block_idx != prev + stride {
                is_strided = false;
                break;
            }
            prev = block_idx;
        }

        if is_strided {
            self.current_pattern = AccessPattern::Strided(stride);
            self.stride = Some(stride);
            return;
        }

        // If no pattern detected, mark as random
        self.current_pattern = AccessPattern::Random;
    }
}

impl AccessPatternTracker for BlockAccessTracker {
    fn record_access(&mut self, block_idx: usize) {
        // Add to history and remove oldest if needed
        self.history.push_back(block_idx);

        if self.history.len() > self.config.history_size {
            self.history.pop_front();
        }

        // Update pattern if we have enough history
        if self.history.len() >= self.config.min_pattern_length {
            self.detect_pattern();
        }

        // Update timestamp
        self.last_update = Instant::now();
    }

    fn predict_next_blocks(&self, count: usize) -> Vec<usize> {
        if self.history.is_empty() {
            return Vec::new();
        }

        let mut predictions = Vec::with_capacity(count);
        let latest = *self.history.back().unwrap();

        match self.current_pattern {
            AccessPattern::Sequential => {
                // For sequential access, predict the next 'count' blocks
                for i in 1..=count {
                    predictions.push(latest + i);
                }
            }
            AccessPattern::Strided(stride) => {
                // For strided access, predict the next 'count' blocks with the given stride
                for i in 1..=count {
                    predictions.push(latest + stride * i);
                }
            }
            _ => {
                // For random access, we can't make good predictions
                // but we could predict nearby blocks as a heuristic
                if latest > 0 {
                    predictions.push(latest - 1);
                }
                predictions.push(latest + 1);

                // Fill remaining predictions with adjacent blocks
                let mut offset = 2;
                while predictions.len() < count {
                    if latest >= offset {
                        predictions.push(latest - offset);
                    }
                    predictions.push(latest + offset);
                    offset += 1;
                }

                // Trim to requested count
                predictions.truncate(count);
            }
        }

        predictions
    }

    fn current_pattern(&self) -> AccessPattern {
        self.current_pattern
    }

    fn clear_history(&mut self) {
        self.history.clear();
        self.current_pattern = AccessPattern::Random;
        self.stride = None;
    }
}

/// Shared state for the prefetching system.
#[derive(Debug)]
pub struct PrefetchingState {
    /// Configuration for prefetching
    config: PrefetchConfig,

    /// Access pattern tracker
    tracker: Box<dyn AccessPatternTracker + Send + Sync>,

    /// Set of blocks that are currently being prefetched
    prefetching: HashSet<usize>,

    /// Set of blocks that have been prefetched and are available in the cache
    prefetched: HashSet<usize>,

    /// Statistics about prefetching
    stats: PrefetchStats,
}

/// Statistics about prefetching performance.
#[derive(Debug, Default, Clone)]
pub struct PrefetchStats {
    /// Total number of prefetch operations performed
    pub prefetch_count: usize,

    /// Number of cache hits on prefetched blocks
    pub prefetch_hits: usize,

    /// Number of accesses to blocks that were not prefetched
    pub prefetch_misses: usize,

    /// Hit rate (hits / (hits + misses))
    pub hit_rate: f64,
}

impl PrefetchingState {
    /// Create a new prefetching state with the given configuration.
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            tracker: Box::new(BlockAccessTracker::new(config.clone())),
            config,
            prefetching: HashSet::new(),
            prefetched: HashSet::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Record an access to a block.
    pub fn record_access(&mut self, block_idx: usize) {
        self.tracker.record_access(block_idx);

        // Update stats if this was a prefetched block
        if self.prefetched.contains(&block_idx) {
            self.stats.prefetch_hits += 1;
            self.prefetched.remove(&block_idx);
        } else {
            self.stats.prefetch_misses += 1;
        }

        // Update hit rate
        let total = self.stats.prefetch_hits + self.stats.prefetch_misses;
        if total > 0 {
            self.stats.hit_rate = self.stats.prefetch_hits as f64 / total as f64;
        }
    }

    /// Get the blocks that should be prefetched next.
    pub fn get_blocks_to_prefetch(&self) -> Vec<usize> {
        if !self.config.enabled {
            return Vec::new();
        }

        // Predict next blocks
        let predicted = self.tracker.predict_next_blocks(self.config.prefetch_count);

        // Filter out blocks that are already prefetched or being prefetched
        predicted
            .into_iter()
            .filter(|&block_idx| {
                !self.prefetched.contains(&block_idx) && !self.prefetching.contains(&block_idx)
            })
            .collect()
    }

    /// Mark a block as being prefetched.
    pub fn mark_prefetching(&mut self, block_idx: usize) {
        self.prefetching.insert(block_idx);
    }

    /// Mark a block as prefetched and available in the cache.
    pub fn mark_prefetched(&mut self, block_idx: usize) {
        self.prefetching.remove(&block_idx);
        self.prefetched.insert(block_idx);
        self.stats.prefetch_count += 1;
    }

    /// Get the current prefetching statistics.
    pub fn stats(&self) -> PrefetchStats {
        self.stats.clone()
    }
}

/// Trait for memory-mapped arrays that support prefetching.
pub trait Prefetching {
    /// Enable prefetching with the given configuration.
    fn enable_prefetching(&mut self, config: PrefetchConfig) -> CoreResult<()>;

    /// Disable prefetching.
    fn disable_prefetching(&mut self) -> CoreResult<()>;

    /// Get the current prefetching statistics.
    fn prefetch_stats(&self) -> CoreResult<PrefetchStats>;

    /// Prefetch a specific block.
    fn prefetch_block(&self, block_idx: usize) -> CoreResult<()>;

    /// Prefetch multiple blocks.
    fn prefetch_blocks(&self, block_indices: &[usize]) -> CoreResult<()>;

    /// Clear the prefetching state.
    fn clear_prefetch_state(&mut self) -> CoreResult<()>;
}

// Extended CompressedMemMappedArray struct with prefetching support
#[derive(Debug)]
pub struct PrefetchingCompressedArray<A: Clone + Copy + 'static> {
    /// The underlying compressed memory-mapped array
    array: CompressedMemMappedArray<A>,

    /// Prefetching state
    prefetch_state: Arc<Mutex<PrefetchingState>>,

    /// Prefetching enabled flag
    prefetching_enabled: bool,

    /// Background prefetching thread handle (if async prefetching is enabled)
    #[allow(dead_code)] // May be unused if async prefetching is disabled
    prefetch_thread: Option<std::thread::JoinHandle<()>>,

    /// Channel to send blocks to prefetch
    #[allow(dead_code)] // May be unused if async prefetching is disabled
    prefetch_sender: Option<std::sync::mpsc::Sender<PrefetchCommand>>,
}

/// Commands for the prefetching thread
enum PrefetchCommand {
    /// Prefetch a specific block
    Prefetch(usize),

    /// Stop the prefetching thread
    Stop,
}

impl<A: Clone + Copy + 'static> PrefetchingCompressedArray<A> {
    /// Create a new prefetching compressed array from an existing compressed memory-mapped array.
    pub fn new(array: CompressedMemMappedArray<A>) -> Self {
        // Create prefetching state with default config
        let prefetch_state = Arc::new(Mutex::new(PrefetchingState::new(PrefetchConfig::default())));

        Self {
            array,
            prefetch_state,
            prefetching_enabled: false,
            prefetch_thread: None,
            prefetch_sender: None,
        }
    }

    /// Create a new prefetching compressed array with the given configuration.
    pub fn new_with_config(
        array: CompressedMemMappedArray<A>,
        config: PrefetchConfig,
    ) -> CoreResult<Self> {
        let mut prefetching_array = Self::new(array);
        prefetching_array.enable_prefetching(config)?;
        Ok(prefetching_array)
    }

    /// Start the background prefetching thread.
    fn start_prefetch_thread(
        &mut self,
        prefetch_state: Arc<Mutex<PrefetchingState>>,
    ) -> CoreResult<()> {
        // Create channel for sending prefetch commands
        let (sender, receiver) = std::sync::mpsc::channel();
        self.prefetch_sender = Some(sender);

        // Clone array for the thread
        let array = self.array.clone();

        // Get the timeout from the config
        let timeout = {
            let guard = prefetch_state.lock().map_err(|_| {
                CoreError::MutexError(ErrorContext::new(
                    "Failed to lock prefetch state".to_string(),
                ))
            })?;
            guard.config.prefetch_timeout
        };

        // Start the thread
        let thread = std::thread::spawn(move || {
            // Background thread for prefetching
            loop {
                // Get the next command
                match receiver.recv_timeout(timeout) {
                    Ok(PrefetchCommand::Prefetch(block_idx)) => {
                        // Mark the block as being prefetched
                        {
                            if let Ok(mut guard) = prefetch_state.lock() {
                                guard.mark_prefetching(block_idx);
                            }
                        }

                        // Attempt to prefetch the block (ignoring errors)
                        if let Ok(_) = array.preload_block(block_idx) {
                            // Mark the block as prefetched
                            if let Ok(mut guard) = prefetch_state.lock() {
                                guard.mark_prefetched(block_idx);
                            }
                        }
                    }
                    Ok(PrefetchCommand::Stop) => {
                        // Stop the thread
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // Timeout, check if there are new blocks to prefetch
                        if let Ok(guard) = prefetch_state.lock() {
                            let blocks = guard.get_blocks_to_prefetch();

                            // If there are blocks to prefetch, we need to drop the lock
                            // and then prefetch them
                            if !blocks.is_empty() {
                                drop(guard);

                                for &block_idx in &blocks {
                                    // Mark the block as being prefetched
                                    if let Ok(mut guard) = prefetch_state.lock() {
                                        guard.mark_prefetching(block_idx);
                                    }

                                    // Attempt to prefetch the block (ignoring errors)
                                    if let Ok(_) = array.preload_block(block_idx) {
                                        // Mark the block as prefetched
                                        if let Ok(mut guard) = prefetch_state.lock() {
                                            guard.mark_prefetched(block_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        // Channel closed, exit thread
                        break;
                    }
                }
            }
        });

        self.prefetch_thread = Some(thread);
        Ok(())
    }

    /// Stop the background prefetching thread.
    fn stop_prefetch_thread(&mut self) -> CoreResult<()> {
        if let Some(sender) = self.prefetch_sender.take() {
            // Send stop command to the thread
            sender.send(PrefetchCommand::Stop).map_err(|_| {
                CoreError::ThreadError(ErrorContext::new("Failed to send stop command".to_string()))
            })?;

            // Wait for the thread to finish
            if let Some(thread) = self.prefetch_thread.take() {
                thread.join().map_err(|_| {
                    CoreError::ThreadError(ErrorContext::new(
                        "Failed to join prefetch thread".to_string(),
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Get access to the underlying compressed memory-mapped array.
    pub fn inner(&self) -> &CompressedMemMappedArray<A> {
        &self.array
    }

    /// Get mutable access to the underlying compressed memory-mapped array.
    pub fn inner_mut(&mut self) -> &mut CompressedMemMappedArray<A> {
        &mut self.array
    }

    /// Request prefetching of a specific block through the background thread.
    fn request_prefetch(&self, block_idx: usize) -> CoreResult<()> {
        if let Some(sender) = &self.prefetch_sender {
            sender
                .send(PrefetchCommand::Prefetch(block_idx))
                .map_err(|_| {
                    CoreError::ThreadError(ErrorContext::new(
                        "Failed to send prefetch command".to_string(),
                    ))
                })?;
        }

        Ok(())
    }
}

impl<A: Clone + Copy + 'static> Prefetching for PrefetchingCompressedArray<A> {
    fn enable_prefetching(&mut self, config: PrefetchConfig) -> CoreResult<()> {
        // Already enabled with the same config?
        if self.prefetching_enabled {
            // Check if we need to update the config
            let current_config = {
                let guard = self.prefetch_state.lock().map_err(|_| {
                    CoreError::MutexError(ErrorContext::new(
                        "Failed to lock prefetch state".to_string(),
                    ))
                })?;
                guard.config.clone()
            };

            if current_config.async_prefetch == config.async_prefetch
                && current_config.prefetch_count == config.prefetch_count
                && current_config.history_size == config.history_size
            {
                // No significant changes, just update the config
                let mut guard = self.prefetch_state.lock().map_err(|_| {
                    CoreError::MutexError(ErrorContext::new(
                        "Failed to lock prefetch state".to_string(),
                    ))
                })?;
                guard.config = config;
                return Ok(());
            }

            // Need to stop the current prefetching and restart with new config
            self.disable_prefetching()?;
        }

        // Create new prefetching state
        let prefetch_state = Arc::new(Mutex::new(PrefetchingState::new(config.clone())));
        self.prefetch_state = prefetch_state.clone();

        // Start background thread if async prefetching is enabled
        if config.async_prefetch {
            self.start_prefetch_thread(prefetch_state)?;
        }

        self.prefetching_enabled = true;
        Ok(())
    }

    fn disable_prefetching(&mut self) -> CoreResult<()> {
        if self.prefetching_enabled {
            // Stop background thread if it's running
            self.stop_prefetch_thread()?;

            // Clear prefetching state
            let mut guard = self.prefetch_state.lock().map_err(|_| {
                CoreError::MutexError(ErrorContext::new(
                    "Failed to lock prefetch state".to_string(),
                ))
            })?;

            // Disable prefetching in the config
            guard.config.enabled = false;

            self.prefetching_enabled = false;
        }

        Ok(())
    }

    fn prefetch_stats(&self) -> CoreResult<PrefetchStats> {
        let guard = self.prefetch_state.lock().map_err(|_| {
            CoreError::MutexError(ErrorContext::new(
                "Failed to lock prefetch state".to_string(),
            ))
        })?;

        Ok(guard.stats())
    }

    fn prefetch_block(&self, block_idx: usize) -> CoreResult<()> {
        if !self.prefetching_enabled {
            return Ok(());
        }

        // Check if the block is already prefetched
        let should_prefetch = {
            let guard = self.prefetch_state.lock().map_err(|_| {
                CoreError::MutexError(ErrorContext::new(
                    "Failed to lock prefetch state".to_string(),
                ))
            })?;

            !guard.prefetched.contains(&block_idx) && !guard.prefetching.contains(&block_idx)
        };

        if should_prefetch {
            // Check if we should do sync or async prefetching
            let is_async = {
                let guard = self.prefetch_state.lock().map_err(|_| {
                    CoreError::MutexError(ErrorContext::new(
                        "Failed to lock prefetch state".to_string(),
                    ))
                })?;

                guard.config.async_prefetch
            };

            if is_async {
                // Request async prefetching
                self.request_prefetch(block_idx)?;
            } else {
                // Mark the block as being prefetched
                {
                    let mut guard = self.prefetch_state.lock().map_err(|_| {
                        CoreError::MutexError(ErrorContext::new(
                            "Failed to lock prefetch state".to_string(),
                        ))
                    })?;

                    guard.mark_prefetching(block_idx);
                }

                // Prefetch the block
                self.array.preload_block(block_idx)?;

                // Mark the block as prefetched
                let mut guard = self.prefetch_state.lock().map_err(|_| {
                    CoreError::MutexError(ErrorContext::new(
                        "Failed to lock prefetch state".to_string(),
                    ))
                })?;

                guard.mark_prefetched(block_idx);
            }
        }

        Ok(())
    }

    fn prefetch_blocks(&self, block_indices: &[usize]) -> CoreResult<()> {
        if !self.prefetching_enabled {
            return Ok(());
        }

        for &block_idx in block_indices {
            self.prefetch_block(block_idx)?;
        }

        Ok(())
    }

    fn clear_prefetch_state(&mut self) -> CoreResult<()> {
        let mut guard = self.prefetch_state.lock().map_err(|_| {
            CoreError::MutexError(ErrorContext::new(
                "Failed to lock prefetch state".to_string(),
            ))
        })?;

        guard.prefetched.clear();
        guard.prefetching.clear();
        guard.tracker.clear_history();

        Ok(())
    }
}

// Extension methods for CompressedMemMappedArray to add prefetching support
impl<A: Clone + Copy + 'static> CompressedMemMappedArray<A> {
    /// Convert into a prefetching compressed array.
    pub fn with_prefetching(self) -> PrefetchingCompressedArray<A> {
        PrefetchingCompressedArray::new(self)
    }

    /// Convert into a prefetching compressed array with the given configuration.
    pub fn with_prefetching_config(
        self,
        config: PrefetchConfig,
    ) -> CoreResult<PrefetchingCompressedArray<A>> {
        PrefetchingCompressedArray::new_with_config(self, config)
    }
}

// For transparent pass-through to underlying array methods
impl<A: Clone + Copy + 'static> std::ops::Deref for PrefetchingCompressedArray<A> {
    type Target = CompressedMemMappedArray<A>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

// Implement wrapper method for get that records accesses
impl<A: Clone + Copy + 'static> PrefetchingCompressedArray<A> {
    /// Get a specific element from the array, with prefetching support.
    pub fn get(&self, indices: &[usize]) -> CoreResult<A> {
        // Calculate block index from the access
        let flat_index = self.calculate_flat_index(indices)?;
        let block_idx = flat_index / self.metadata.block_size;

        // Record the access
        if self.prefetching_enabled {
            let mut guard = self.prefetch_state.lock().map_err(|_| {
                CoreError::MutexError(ErrorContext::new(
                    "Failed to lock prefetch state".to_string(),
                ))
            })?;

            guard.record_access(block_idx);

            // Get blocks to prefetch
            let to_prefetch = guard.get_blocks_to_prefetch();

            // Drop the lock before prefetching
            drop(guard);

            // Request prefetching of predicted blocks
            for &idx in &to_prefetch {
                self.prefetch_block(idx)?;
            }
        }

        // Get the element from the underlying array
        self.array.get(indices)
    }

    /// Calculate the flat index from multidimensional indices.
    fn calculate_flat_index(&self, indices: &[usize]) -> CoreResult<usize> {
        // Check that the indices are valid
        if indices.len() != self.metadata.shape.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} indices, got {}",
                self.metadata.shape.len(),
                indices.len()
            ))));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.metadata.shape[i] {
                return Err(CoreError::IndexError(ErrorContext::new(format!(
                    "Index {} out of bounds for dimension {} (max {})",
                    idx,
                    i,
                    self.metadata.shape[i] - 1
                ))));
            }
        }

        // Calculate flat index
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            if i > 0 {
                stride *= self.metadata.shape[i];
            }
        }

        Ok(flat_index)
    }

    /// Slice the array with prefetching support.
    pub fn slice(
        &self,
        ranges: &[(usize, usize)],
    ) -> CoreResult<ndarray::Array<A, ndarray::IxDyn>> {
        // Record accesses for the blocks that will be accessed
        if self.prefetching_enabled {
            // Determine which blocks will be accessed
            let blocks = self.calculate_blocks_for_slice(ranges)?;

            // Record accesses and prefetch
            let mut guard = self.prefetch_state.lock().map_err(|_| {
                CoreError::MutexError(ErrorContext::new(
                    "Failed to lock prefetch state".to_string(),
                ))
            })?;

            // Record each block access
            for &block_idx in &blocks {
                guard.record_access(block_idx);
            }

            // Get blocks to prefetch
            let to_prefetch = guard.get_blocks_to_prefetch();

            // Drop the lock before prefetching
            drop(guard);

            // Request prefetching of predicted blocks
            for &idx in &to_prefetch {
                self.prefetch_block(idx)?;
            }
        }

        // Use the underlying array's slice method
        self.array.slice(ranges)
    }

    /// Calculate which blocks will be accessed for a slice operation.
    fn calculate_blocks_for_slice(&self, ranges: &[(usize, usize)]) -> CoreResult<HashSet<usize>> {
        // Check that the ranges are valid
        if ranges.len() != self.metadata.shape.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} ranges, got {}",
                self.metadata.shape.len(),
                ranges.len()
            ))));
        }

        // Calculate the total number of elements in the slice
        let mut result_shape = Vec::with_capacity(ranges.len());
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end {
                return Err(CoreError::ValueError(ErrorContext::new(format!(
                    "Invalid range for dimension {}: {}..{}",
                    i, start, end
                ))));
            }
            if end > self.metadata.shape[i] {
                return Err(CoreError::IndexError(ErrorContext::new(format!(
                    "Range {}..{} out of bounds for dimension {} (max {})",
                    start, end, i, self.metadata.shape[i]
                ))));
            }
            result_shape.push(end - start);
        }

        // Calculate the strides for each dimension
        let mut strides = Vec::with_capacity(self.metadata.shape.len());
        let mut stride = 1;
        for i in (0..self.metadata.shape.len()).rev() {
            strides.push(stride);
            if i > 0 {
                stride *= self.metadata.shape[i];
            }
        }
        strides.reverse();

        // Calculate the blocks that will be accessed
        let mut blocks = HashSet::new();
        let block_size = self.metadata.block_size;

        // Calculate the corners of the hypercube
        let mut corners = Vec::with_capacity(1 << ranges.len());
        corners.push(vec![0; ranges.len()]);

        for dim in 0..ranges.len() {
            let mut new_corners = Vec::new();
            for corner in &corners {
                let mut corner1 = corner.clone();
                let mut corner2 = corner.clone();
                corner1[dim] = 0;
                corner2[dim] = result_shape[dim] - 1;
                new_corners.push(corner1);
                new_corners.push(corner2);
            }
            corners = new_corners;
        }

        // Convert corners to flat indices and blocks
        for corner in corners {
            let mut flat_index = 0;
            for (dim, &offset) in corner.iter().enumerate() {
                flat_index += (ranges[dim].0 + offset) * strides[dim];
            }

            let block_idx = flat_index / block_size;
            blocks.insert(block_idx);
        }

        // For large slices, we should also check intermediate blocks along the edges
        // This is a simplification, but covers many common cases
        if blocks.len() > 1 {
            let min_block = *blocks.iter().min().unwrap();
            let max_block = *blocks.iter().max().unwrap();

            // Add all blocks in between
            for block_idx in min_block..=max_block {
                blocks.insert(block_idx);
            }
        }

        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_detection_sequential() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            ..Default::default()
        };

        let mut tracker = BlockAccessTracker::new(config);

        // Record sequential access
        for i in 0..10 {
            tracker.record_access(i);
        }

        // Check that the pattern was detected correctly
        assert_eq!(tracker.current_pattern(), AccessPattern::Sequential);

        // Check predictions
        let predictions = tracker.predict_next_blocks(3);
        assert_eq!(predictions, vec![10, 11, 12]);
    }

    #[test]
    fn test_access_pattern_detection_strided() {
        let config = PrefetchConfig {
            min_pattern_length: 4,
            ..Default::default()
        };

        let mut tracker = BlockAccessTracker::new(config);

        // Record strided access with stride 3
        for i in (0..30).step_by(3) {
            tracker.record_access(i);
        }

        // Check that the pattern was detected correctly
        assert_eq!(tracker.current_pattern(), AccessPattern::Strided(3));

        // Check predictions
        let predictions = tracker.predict_next_blocks(3);
        assert_eq!(predictions, vec![30, 33, 36]);
    }

    #[test]
    fn test_prefetching_state() {
        let config = PrefetchConfig {
            prefetch_count: 3,
            ..Default::default()
        };

        let mut state = PrefetchingState::new(config);

        // Record sequential access
        for i in 0..5 {
            state.record_access(i);
        }

        // Get blocks to prefetch
        let to_prefetch = state.get_blocks_to_prefetch();
        assert_eq!(to_prefetch, vec![5, 6, 7]);

        // Mark blocks as being prefetched
        for &block in &to_prefetch {
            state.mark_prefetching(block);
        }

        // Mark block 5 as prefetched
        state.mark_prefetched(5);

        // Access block 5 (should be a hit)
        state.record_access(5);

        // Check stats
        let stats = state.stats();
        assert_eq!(stats.prefetch_hits, 1);
        assert_eq!(stats.prefetch_misses, 5); // Initial 5 accesses
        assert!(stats.hit_rate > 0.0);
    }
}
