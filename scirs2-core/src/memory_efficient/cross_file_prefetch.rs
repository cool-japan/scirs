//! Cross-file prefetching for related data access.
//!
//! This module provides a system for tracking access patterns across multiple files
//! or arrays and prefetching related data. This is particularly useful for scientific
//! computing workloads where multiple datasets are often accessed together in predictable ways.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};

use super::prefetch::{PrefetchConfig, PrefetchStats};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Default correlation threshold for considering files related
const DEFAULT_CORRELATION_THRESHOLD: f64 = 0.6;

/// Default time window for correlation tracking
const DEFAULT_CORRELATION_WINDOW: Duration = Duration::from_secs(60);

/// Default minimum occurrences for establishing correlation
const DEFAULT_MIN_OCCURRENCES: usize = 3;

/// Default expiration time for correlations
const DEFAULT_CORRELATION_EXPIRY: Duration = Duration::from_secs(3600); // 1 hour

/// A unique identifier for a file or array.
#[derive(Debug, Clone, Eq)]
pub struct DatasetId {
    /// Path for file-backed datasets
    pub path: Option<PathBuf>,

    /// Memory address for in-memory datasets
    pub memory_address: Option<usize>,

    /// Unique name for the dataset
    pub name: String,
}

impl PartialEq for DatasetId {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(ref self_path), Some(ref other_path)) = (&self.path, &other.path) {
            return self_path == other_path;
        }

        if let (Some(self_addr), Some(other_addr)) = (self.memory_address, other.memory_address) {
            return self_addr == other_addr;
        }

        self.name == other.name
    }
}

impl Hash for DatasetId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(ref path) = self.path {
            path.hash(state);
        } else if let Some(addr) = self.memory_address {
            addr.hash(state);
        } else {
            self.name.hash(state);
        }
    }
}

impl DatasetId {
    /// Create a new dataset ID from a file path.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        let path_buf = path.as_ref().to_path_buf();
        let name = path_buf
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed_dataset".to_string());

        Self {
            path: Some(path_buf),
            memory_address: None,
            name,
        }
    }

    /// Create a new dataset ID from a memory address.
    pub fn from_address(address: usize, name: impl Into<String>) -> Self {
        Self {
            path: None,
            memory_address: Some(address),
            name: name.into(),
        }
    }

    /// Create a new dataset ID from a name.
    pub fn from_name(name: impl Into<String>) -> Self {
        Self {
            path: None,
            memory_address: None,
            name: name.into(),
        }
    }
}

/// An access to a specific part of a dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataAccess {
    /// Dataset being accessed
    pub dataset: DatasetId,

    /// Index or offset being accessed
    pub index: usize,

    /// Type of access (read, write, etc.)
    pub access_type: AccessType,

    /// Size of the access in bytes
    pub size: Option<usize>,

    /// Dimensions accessed (for multi-dimensional datasets)
    pub dimensions: Option<Vec<usize>>,
}

/// Types of data access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessType {
    /// Reading data
    Read,

    /// Writing data
    Write,

    /// Both reading and writing
    ReadWrite,

    /// Metadata access
    Metadata,
}

/// A record of a data access with timestamp.
#[derive(Debug, Clone)]
struct AccessRecord {
    /// The access details
    access: DataAccess,

    /// When the access occurred
    timestamp: Instant,

    /// Duration of the access
    duration: Option<Duration>,
}

/// Configuration for cross-file prefetching.
#[derive(Debug, Clone)]
pub struct CrossFilePrefetchConfig {
    /// Threshold for considering datasets correlated
    pub correlation_threshold: f64,

    /// Time window for correlation tracking
    pub correlation_window: Duration,

    /// Minimum occurrences for establishing correlation
    pub min_occurrences: usize,

    /// Maximum number of datasets to prefetch
    pub max_prefetch_datasets: usize,

    /// Maximum elements to prefetch per dataset
    pub max_prefetch_elements: usize,

    /// Whether to prefetch entire files when correlation is very high
    pub prefetch_entire_file: bool,

    /// Expiration time for correlations
    pub correlation_expiry: Duration,

    /// Whether to enable learning from access patterns
    pub enable_learning: bool,
}

impl Default for CrossFilePrefetchConfig {
    fn default() -> Self {
        Self {
            correlation_threshold: DEFAULT_CORRELATION_THRESHOLD,
            correlation_window: DEFAULT_CORRELATION_WINDOW,
            min_occurrences: DEFAULT_MIN_OCCURRENCES,
            max_prefetch_datasets: 3,
            max_prefetch_elements: 100,
            prefetch_entire_file: false,
            correlation_expiry: DEFAULT_CORRELATION_EXPIRY,
            enable_learning: true,
        }
    }
}

/// Builder for cross-file prefetch configuration.
#[derive(Debug, Clone)]
pub struct CrossFilePrefetchConfigBuilder {
    config: CrossFilePrefetchConfig,
}

impl CrossFilePrefetchConfigBuilder {
    /// Create a new config builder with default settings.
    pub fn new() -> Self {
        Self {
            config: CrossFilePrefetchConfig::default(),
        }
    }

    /// Set the correlation threshold.
    pub fn with_correlation_threshold(mut self, threshold: f64) -> Self {
        self.config.correlation_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the correlation window.
    pub fn with_correlation_window(mut self, window: Duration) -> Self {
        self.config.correlation_window = window;
        self
    }

    /// Set the minimum occurrences for correlation.
    pub fn with_min_occurrences(mut self, occurrences: usize) -> Self {
        self.config.min_occurrences = occurrences;
        self
    }

    /// Set the maximum number of datasets to prefetch.
    pub fn with_max_prefetch_datasets(mut self, max_datasets: usize) -> Self {
        self.config.max_prefetch_datasets = max_datasets;
        self
    }

    /// Set the maximum elements to prefetch per dataset.
    pub fn with_max_prefetch_elements(mut self, max_elements: usize) -> Self {
        self.config.max_prefetch_elements = max_elements;
        self
    }

    /// Enable or disable prefetching entire files.
    pub fn with_prefetch_entire_file(mut self, enable: bool) -> Self {
        self.config.prefetch_entire_file = enable;
        self
    }

    /// Set the correlation expiry time.
    pub fn with_correlation_expiry(mut self, expiry: Duration) -> Self {
        self.config.correlation_expiry = expiry;
        self
    }

    /// Enable or disable learning from access patterns.
    pub fn with_enable_learning(mut self, enable: bool) -> Self {
        self.config.enable_learning = enable;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> CrossFilePrefetchConfig {
        self.config
    }
}

/// A correlation between datasets.
#[derive(Debug, Clone)]
struct DatasetCorrelation {
    /// Primary dataset
    primary: DatasetId,

    /// Related dataset
    related: DatasetId,

    /// Correlation strength (0.0 to 1.0)
    strength: f64,

    /// Number of co-occurrences observed
    occurrences: usize,

    /// Last time this correlation was observed
    last_observed: Instant,

    /// Index correlations (mapping from primary index to related indices)
    index_correlations: HashMap<usize, Vec<usize>>,
}

impl DatasetCorrelation {
    /// Create a new dataset correlation.
    fn new(primary: DatasetId, related: DatasetId) -> Self {
        Self {
            primary,
            related,
            strength: 0.0,
            occurrences: 0,
            last_observed: Instant::now(),
            index_correlations: HashMap::new(),
        }
    }

    /// Update the correlation based on new evidence.
    fn update(&mut self, primary_index: usize, related_indices: &[usize]) {
        self.occurrences += 1;
        self.last_observed = Instant::now();

        // Update strength based on new occurrence
        // Simple learning strategy: increase strength with each occurrence
        self.strength = (self.strength * 0.9) + 0.1;

        // Update index correlations
        let entry = self
            .index_correlations
            .entry(primary_index)
            .or_insert_with(Vec::new);

        for &related_index in related_indices {
            if !entry.contains(&related_index) {
                entry.push(related_index);
            }
        }
    }

    /// Check if the correlation is still valid.
    fn is_valid(&self, expiry: Duration) -> bool {
        self.last_observed.elapsed() <= expiry
    }

    /// Get related indices for a primary index.
    fn get_related_indices(&self, primary_index: usize, max_count: usize) -> Vec<usize> {
        if let Some(indices) = self.index_correlations.get(&primary_index) {
            indices.iter().take(max_count).copied().collect()
        } else {
            // If no specific index correlation, return nearby indices based on the primary index
            let mut nearby = Vec::with_capacity(max_count);

            // Add the exact same index first
            nearby.push(primary_index);

            // Add some nearby indices
            for i in 1..=max_count / 2 {
                if primary_index >= i {
                    nearby.push(primary_index - i);
                }
                nearby.push(primary_index + i);
            }

            nearby.into_iter().take(max_count).collect()
        }
    }
}

/// Manager for cross-file prefetching.
pub struct CrossFilePrefetchManager {
    /// Configuration for cross-file prefetching
    config: CrossFilePrefetchConfig,

    /// Recent access history
    access_history: VecDeque<AccessRecord>,

    /// Dataset correlations
    correlations: HashMap<DatasetId, HashMap<DatasetId, DatasetCorrelation>>,

    /// Registered datasets with their prefetchers
    datasets: HashMap<DatasetId, Weak<dyn DatasetPrefetcher>>,

    /// Cached last access for each dataset
    last_dataset_access: HashMap<DatasetId, (usize, Instant)>,
}

impl CrossFilePrefetchManager {
    /// Create a new cross-file prefetch manager.
    pub fn new(config: CrossFilePrefetchConfig) -> Self {
        Self {
            config,
            access_history: VecDeque::with_capacity(1000),
            correlations: HashMap::new(),
            datasets: HashMap::new(),
            last_dataset_access: HashMap::new(),
        }
    }

    /// Record a data access and trigger prefetching of related data.
    pub fn record_access(&mut self, access: DataAccess) -> CoreResult<()> {
        let access_record = AccessRecord {
            access: access.clone(),
            timestamp: Instant::now(),
            duration: None,
        };

        // Update access history
        self.access_history.push_back(access_record);

        // Limit history size
        while self.access_history.len() > 1000 {
            self.access_history.pop_front();
        }

        // Update last access for this dataset
        self.last_dataset_access
            .insert(access.dataset.clone(), (access.index, Instant::now()));

        // Update correlations
        if self.config.enable_learning {
            self.update_correlations(&access);
        }

        // Trigger prefetching for related datasets
        self.prefetch_related_data(&access)
    }

    /// Complete a data access record with duration.
    pub fn complete_access(&mut self, dataset: &DatasetId, index: usize, duration: Duration) {
        // Find the most recent access to this dataset and index
        if let Some(record) = self.access_history.iter_mut().rev().find(|r| {
            r.access.dataset == *dataset && r.access.index == index && r.duration.is_none()
        }) {
            record.duration = Some(duration);
        }
    }

    /// Register a dataset with its prefetcher.
    pub fn register_dataset(&mut self, dataset: DatasetId, prefetcher: Arc<dyn DatasetPrefetcher>) {
        self.datasets.insert(dataset, Arc::downgrade(&prefetcher));
    }

    /// Unregister a dataset.
    pub fn unregister_dataset(&mut self, dataset: &DatasetId) {
        self.datasets.remove(dataset);

        // Remove any correlations involving this dataset
        self.correlations.remove(dataset);
        for (_, related_map) in self.correlations.iter_mut() {
            related_map.remove(dataset);
        }
    }

    /// Update correlations based on a new access.
    fn update_correlations(&mut self, access: &DataAccess) {
        let current_time = Instant::now();
        let window_start = current_time - self.config.correlation_window;

        // Collect recent accesses within the correlation window
        let recent_accesses: Vec<_> = self
            .access_history
            .iter()
            .filter(|record| {
                record.timestamp >= window_start && record.access.dataset != access.dataset
                // Only different datasets
            })
            .collect();

        // Group by dataset
        let mut dataset_accesses: HashMap<&DatasetId, Vec<&AccessRecord>> = HashMap::new();
        for record in &recent_accesses {
            dataset_accesses
                .entry(&record.access.dataset)
                .or_default()
                .push(record);
        }

        // Update correlations for each dataset
        for (related_dataset, records) in dataset_accesses {
            // Check if we have enough occurrences
            if records.len() < self.config.min_occurrences {
                continue;
            }

            // Extract indices from the related dataset
            let related_indices: Vec<usize> = records.iter().map(|r| r.access.index).collect();

            // Get or create correlation
            let correlation = self
                .correlations
                .entry(access.dataset.clone())
                .or_default()
                .entry(related_dataset.clone())
                .or_insert_with(|| {
                    DatasetCorrelation::new(access.dataset.clone(), related_dataset.clone())
                });

            // Update the correlation
            correlation.update(access.index, &related_indices);
        }

        // Clean up expired correlations
        self.clean_expired_correlations();
    }

    /// Remove expired correlations.
    fn clean_expired_correlations(&mut self) {
        let expiry = self.config.correlation_expiry;

        // For each primary dataset
        let primaries: Vec<_> = self.correlations.keys().cloned().collect();
        for primary in primaries {
            if let Some(related_map) = self.correlations.get_mut(&primary) {
                // Remove expired correlations
                related_map.retain(|_, corr| corr.is_valid(expiry));

                // If no correlations left, remove the primary dataset entry
                if related_map.is_empty() {
                    self.correlations.remove(&primary);
                }
            }
        }
    }

    /// Prefetch related data based on a current access.
    fn prefetch_related_data(&self, access: &DataAccess) -> CoreResult<()> {
        // Skip if we don't have correlations for this dataset
        if !self.correlations.contains_key(&access.dataset) {
            return Ok(());
        }

        // Get correlations for this dataset
        let related_datasets = self.correlations.get(&access.dataset).unwrap();

        // Sort by correlation strength
        let mut correlations: Vec<_> = related_datasets.values().collect();
        correlations.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

        // Prefetch from datasets with strongest correlations
        let mut prefetch_count = 0;
        for correlation in correlations {
            // Skip weak correlations
            if correlation.strength < self.config.correlation_threshold {
                continue;
            }

            // Lookup the prefetcher for this dataset
            if let Some(weak_prefetcher) = self.datasets.get(&correlation.related) {
                if let Some(prefetcher) = weak_prefetcher.upgrade() {
                    // Get the indices to prefetch
                    let indices = correlation
                        .get_related_indices(access.index, self.config.max_prefetch_elements);

                    // Prefetch the related indices
                    if self.config.prefetch_entire_file && correlation.strength > 0.9 {
                        prefetcher.prefetch_all()?;
                    } else {
                        prefetcher.prefetch_indices(&indices)?;
                    }

                    prefetch_count += 1;

                    // Limit the number of datasets we prefetch from
                    if prefetch_count >= self.config.max_prefetch_datasets {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get correlations for a dataset.
    pub fn get_correlations(&self, dataset: &DatasetId) -> Vec<(DatasetId, f64)> {
        let mut result = Vec::new();

        if let Some(related_map) = self.correlations.get(dataset) {
            for (related, correlation) in related_map {
                result.push((related.clone(), correlation.strength));
            }
        }

        // Sort by correlation strength (strongest first)
        result.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        result
    }

    /// Get the strongest related dataset for a given dataset.
    pub fn get_strongest_related(&self, dataset: &DatasetId) -> Option<(DatasetId, f64)> {
        if let Some(related_map) = self.correlations.get(dataset) {
            related_map
                .iter()
                .max_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
                .map(|(k, v)| (k.clone(), v.strength))
        } else {
            None
        }
    }

    /// Get all datasets that have been accessed.
    pub fn get_active_datasets(&self) -> Vec<DatasetId> {
        self.last_dataset_access.keys().cloned().collect()
    }
}

/// Interface for dataset-specific prefetchers.
pub trait DatasetPrefetcher: Send + Sync {
    /// Prefetch specific indices.
    fn prefetch_indices(&self, indices: &[usize]) -> CoreResult<()>;

    /// Prefetch the entire dataset.
    fn prefetch_all(&self) -> CoreResult<()>;

    /// Get the dataset ID.
    fn get_dataset_id(&self) -> DatasetId;
}

/// A global registry for cross-file prefetching.
pub struct CrossFilePrefetchRegistry {
    /// Shared manager instance
    manager: Arc<Mutex<CrossFilePrefetchManager>>,
}

impl CrossFilePrefetchRegistry {
    /// Create a new cross-file prefetch registry.
    pub fn new(config: CrossFilePrefetchConfig) -> Self {
        Self {
            manager: Arc::new(Mutex::new(CrossFilePrefetchManager::new(config))),
        }
    }

    /// Get the global registry instance.
    pub fn global() -> &'static Self {
        use std::sync::Once;
        static INIT: Once = Once::new();
        static mut INSTANCE: Option<CrossFilePrefetchRegistry> = None;

        INIT.call_once(|| {
            let registry = CrossFilePrefetchRegistry::new(CrossFilePrefetchConfig::default());
            unsafe {
                INSTANCE = Some(registry);
            }
        });

        unsafe { INSTANCE.as_ref().unwrap() }
    }

    /// Record a data access.
    pub fn record_access(&self, access: DataAccess) -> CoreResult<()> {
        match self.manager.lock() {
            Ok(mut manager) => manager.record_access(access),
            Err(_) => Err(CoreError::MutexError(ErrorContext::new(
                "Failed to acquire lock on cross-file prefetch manager".to_string(),
            ))),
        }
    }

    /// Complete a data access with duration.
    pub fn complete_access(
        &self,
        dataset: &DatasetId,
        index: usize,
        duration: Duration,
    ) -> CoreResult<()> {
        match self.manager.lock() {
            Ok(mut manager) => {
                manager.complete_access(dataset, index, duration);
                Ok(())
            }
            Err(_) => Err(CoreError::MutexError(ErrorContext::new(
                "Failed to acquire lock on cross-file prefetch manager".to_string(),
            ))),
        }
    }

    /// Register a dataset with its prefetcher.
    pub fn register_dataset(
        &self,
        dataset: DatasetId,
        prefetcher: Arc<dyn DatasetPrefetcher>,
    ) -> CoreResult<()> {
        match self.manager.lock() {
            Ok(mut manager) => {
                manager.register_dataset(dataset, prefetcher);
                Ok(())
            }
            Err(_) => Err(CoreError::MutexError(ErrorContext::new(
                "Failed to acquire lock on cross-file prefetch manager".to_string(),
            ))),
        }
    }

    /// Unregister a dataset.
    pub fn unregister_dataset(&self, dataset: &DatasetId) -> CoreResult<()> {
        match self.manager.lock() {
            Ok(mut manager) => {
                manager.unregister_dataset(dataset);
                Ok(())
            }
            Err(_) => Err(CoreError::MutexError(ErrorContext::new(
                "Failed to acquire lock on cross-file prefetch manager".to_string(),
            ))),
        }
    }

    /// Get correlations for a dataset.
    pub fn get_correlations(&self, dataset: &DatasetId) -> CoreResult<Vec<(DatasetId, f64)>> {
        match self.manager.lock() {
            Ok(manager) => Ok(manager.get_correlations(dataset)),
            Err(_) => Err(CoreError::MutexError(ErrorContext::new(
                "Failed to acquire lock on cross-file prefetch manager".to_string(),
            ))),
        }
    }

    /// Get all active datasets.
    pub fn get_active_datasets(&self) -> CoreResult<Vec<DatasetId>> {
        match self.manager.lock() {
            Ok(manager) => Ok(manager.get_active_datasets()),
            Err(_) => Err(CoreError::MutexError(ErrorContext::new(
                "Failed to acquire lock on cross-file prefetch manager".to_string(),
            ))),
        }
    }
}

/// Implementation of DatasetPrefetcher for CompressedMemMappedArray.
pub struct CompressedArrayPrefetcher<A: Clone + Copy + 'static> {
    /// Dataset ID
    dataset_id: DatasetId,

    /// Compressed array reference
    array: Arc<super::compressed_memmap::CompressedMemMappedArray<A>>,
}

impl<A: Clone + Copy + 'static> CompressedArrayPrefetcher<A> {
    /// Create a new prefetcher for a compressed array.
    pub fn new(
        dataset_id: DatasetId,
        array: Arc<super::compressed_memmap::CompressedMemMappedArray<A>>,
    ) -> Self {
        Self { dataset_id, array }
    }
}

impl<A: Clone + Copy + 'static> DatasetPrefetcher for CompressedArrayPrefetcher<A> {
    fn prefetch_indices(&self, indices: &[usize]) -> CoreResult<()> {
        if indices.is_empty() {
            return Ok(());
        }

        // For compressed arrays, we need to convert flat indices to block indices
        let block_size = self.array.metadata.block_size;

        // Calculate unique block indices to prefetch
        let mut block_indices = HashSet::new();
        for &idx in indices {
            let block_idx = idx / block_size;
            block_indices.insert(block_idx);
        }

        // Prefetch each block
        for block_idx in block_indices {
            self.array.preload_block(block_idx)?;
        }

        Ok(())
    }

    fn prefetch_all(&self) -> CoreResult<()> {
        // Prefetch all blocks
        let total_blocks = self.array.metadata.num_blocks;
        for block_idx in 0..total_blocks {
            self.array.preload_block(block_idx)?;
        }

        Ok(())
    }

    fn get_dataset_id(&self) -> DatasetId {
        self.dataset_id.clone()
    }
}

/// Implementation of DatasetPrefetcher for MemoryMappedArray.
pub struct MemoryMappedArrayPrefetcher<A: Clone + Copy + 'static> {
    /// Dataset ID
    dataset_id: DatasetId,

    /// Memory-mapped array reference
    array: Arc<super::memmap::MemoryMappedArray<A>>,

    /// Chunk size for prefetching
    chunk_size: usize,
}

impl<A: Clone + Copy + 'static> MemoryMappedArrayPrefetcher<A> {
    /// Create a new prefetcher for a memory-mapped array.
    pub fn new(
        dataset_id: DatasetId,
        array: Arc<super::memmap::MemoryMappedArray<A>>,
        chunk_size: usize,
    ) -> Self {
        Self {
            dataset_id,
            array,
            chunk_size,
        }
    }
}

impl<A: Clone + Copy + 'static> DatasetPrefetcher for MemoryMappedArrayPrefetcher<A> {
    fn prefetch_indices(&self, indices: &[usize]) -> CoreResult<()> {
        if indices.is_empty() {
            return Ok(());
        }

        // For memory-mapped arrays, we need to ensure the data is loaded
        // This is a simple approximation - doesn't actually load the data
        // but ensures the memory is mapped

        // Ensure the array is readable
        let _array = self.array.readonly_array()?;

        Ok(())
    }

    fn prefetch_all(&self) -> CoreResult<()> {
        // For memory-mapped arrays, just ensure the file is mapped
        let _array = self.array.readonly_array()?;

        Ok(())
    }

    fn get_dataset_id(&self) -> DatasetId {
        self.dataset_id.clone()
    }
}

/// Extension traits for compressed arrays to enable cross-file prefetching.
pub trait CompressedArrayPrefetchExt<A: Clone + Copy + 'static> {
    /// Register with the cross-file prefetching system.
    fn register_for_cross_prefetch(
        &self,
        dataset_id: DatasetId,
    ) -> CoreResult<Arc<CompressedArrayPrefetcher<A>>>;
}

impl<A: Clone + Copy + 'static> CompressedArrayPrefetchExt<A>
    for super::compressed_memmap::CompressedMemMappedArray<A>
{
    fn register_for_cross_prefetch(
        &self,
        dataset_id: DatasetId,
    ) -> CoreResult<Arc<CompressedArrayPrefetcher<A>>> {
        let array = Arc::new(self.clone());
        let prefetcher = Arc::new(CompressedArrayPrefetcher::new(dataset_id.clone(), array));

        CrossFilePrefetchRegistry::global().register_dataset(dataset_id, prefetcher.clone())?;

        Ok(prefetcher)
    }
}

/// Extension traits for memory-mapped arrays to enable cross-file prefetching.
pub trait MemoryMappedArrayPrefetchExt<A: Clone + Copy + 'static> {
    /// Register with the cross-file prefetching system.
    fn register_for_cross_prefetch(
        &self,
        dataset_id: DatasetId,
        chunk_size: usize,
    ) -> CoreResult<Arc<MemoryMappedArrayPrefetcher<A>>>;
}

impl<A: Clone + Copy + 'static> MemoryMappedArrayPrefetchExt<A>
    for super::memmap::MemoryMappedArray<A>
{
    fn register_for_cross_prefetch(
        &self,
        dataset_id: DatasetId,
        chunk_size: usize,
    ) -> CoreResult<Arc<MemoryMappedArrayPrefetcher<A>>> {
        let array = Arc::new(self.clone());
        let prefetcher = Arc::new(MemoryMappedArrayPrefetcher::new(
            dataset_id.clone(),
            array,
            chunk_size,
        ));

        CrossFilePrefetchRegistry::global().register_dataset(dataset_id, prefetcher.clone())?;

        Ok(prefetcher)
    }
}

/// Middleware wrapper for tracking array accesses.
pub struct TrackedArray<A: Clone + Copy + 'static, T> {
    /// Underlying array
    array: T,

    /// Dataset ID
    dataset_id: DatasetId,

    /// Phantom for type parameter
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Clone + Copy + 'static, T> TrackedArray<A, T> {
    /// Create a new tracked array.
    pub fn new(array: T, dataset_id: DatasetId) -> Self {
        Self {
            array,
            dataset_id,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a reference to the underlying array.
    pub fn inner(&self) -> &T {
        &self.array
    }

    /// Get a mutable reference to the underlying array.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.array
    }

    /// Get the dataset ID.
    pub fn dataset_id(&self) -> &DatasetId {
        &self.dataset_id
    }

    /// Record an access to the array.
    fn record_access(
        &self,
        index: usize,
        access_type: AccessType,
        dimensions: Option<Vec<usize>>,
    ) -> CoreResult<()> {
        let access = DataAccess {
            dataset: self.dataset_id.clone(),
            index,
            access_type,
            size: None,
            dimensions,
        };

        CrossFilePrefetchRegistry::global().record_access(access)
    }
}

/// Implementation of TrackedArray for CompressedMemMappedArray.
impl<A: Clone + Copy + 'static>
    TrackedArray<A, super::compressed_memmap::CompressedMemMappedArray<A>>
{
    /// Get an element from the array, recording the access.
    pub fn get(&self, indices: &[usize]) -> CoreResult<A> {
        let start = Instant::now();

        // Calculate flat index
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            if i > 0 {
                stride *= self.array.metadata.shape[i];
            }
        }

        // Record the access
        self.record_access(flat_index, AccessType::Read, Some(indices.to_vec()))?;

        // Get the element
        let result = self.array.get(indices);

        // Record the duration
        CrossFilePrefetchRegistry::global().complete_access(
            &self.dataset_id,
            flat_index,
            start.elapsed(),
        )?;

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_id() {
        let path_id = DatasetId::from_path("/path/to/data.bin");
        let mem_id = DatasetId::from_address(0x12345678, "memory_dataset");
        let name_id = DatasetId::from_name("named_dataset");

        // Test equality
        assert_eq!(path_id, DatasetId::from_path("/path/to/data.bin"));
        assert_eq!(mem_id, DatasetId::from_address(0x12345678, "other_name"));
        assert_eq!(name_id, DatasetId::from_name("named_dataset"));

        // Test inequality
        assert_ne!(path_id, mem_id);
        assert_ne!(path_id, name_id);
        assert_ne!(mem_id, name_id);
    }

    #[test]
    fn test_correlation_update() {
        let primary = DatasetId::from_name("dataset1");
        let related = DatasetId::from_name("dataset2");

        let mut correlation = DatasetCorrelation::new(primary.clone(), related.clone());

        // Initially zero strength
        assert_eq!(correlation.strength, 0.0);
        assert_eq!(correlation.occurrences, 0);

        // Update with some indices
        correlation.update(10, &[20, 30, 40]);

        // Strength should increase
        assert!(correlation.strength > 0.0);
        assert_eq!(correlation.occurrences, 1);

        // Should have recorded the index correlation
        assert!(correlation.index_correlations.contains_key(&10));
        assert_eq!(correlation.index_correlations[&10], vec![20, 30, 40]);

        // Update again
        correlation.update(10, &[20, 30, 50]);

        // Strength should increase further
        assert!(correlation.strength > 0.1);
        assert_eq!(correlation.occurrences, 2);

        // Should have updated the index correlation without duplicates
        assert_eq!(correlation.index_correlations[&10], vec![20, 30, 40, 50]);
    }

    #[test]
    fn test_prefetch_manager() {
        // Create a manager
        let config = CrossFilePrefetchConfig {
            correlation_threshold: 0.5,
            min_occurrences: 2,
            ..Default::default()
        };

        let mut manager = CrossFilePrefetchManager::new(config);

        // Create some datasets
        let dataset1 = DatasetId::from_name("dataset1");
        let dataset2 = DatasetId::from_name("dataset2");

        // Record some related accesses
        for i in 0..5 {
            let access1 = DataAccess {
                dataset: dataset1.clone(),
                index: i,
                access_type: AccessType::Read,
                size: None,
                dimensions: None,
            };

            let access2 = DataAccess {
                dataset: dataset2.clone(),
                index: i * 2,
                access_type: AccessType::Read,
                size: None,
                dimensions: None,
            };

            // Record access to both datasets in sequence
            manager.record_access(access1).unwrap();
            manager.record_access(access2).unwrap();
        }

        // Should have established a correlation
        let correlations = manager.get_correlations(&dataset1);
        assert!(!correlations.is_empty());

        // First correlation should be to dataset2
        assert_eq!(correlations[0].0, dataset2);

        // Correlation strength should be positive
        assert!(correlations[0].1 > 0.0);
    }

    // Define a mock prefetcher for testing
    struct MockPrefetcher {
        dataset_id: DatasetId,
        prefetched_indices: Arc<Mutex<Vec<usize>>>,
        prefetched_all: Arc<Mutex<bool>>,
    }

    impl MockPrefetcher {
        fn new(dataset_id: DatasetId) -> Self {
            Self {
                dataset_id,
                prefetched_indices: Arc::new(Mutex::new(Vec::new())),
                prefetched_all: Arc::new(Mutex::new(false)),
            }
        }
    }

    impl DatasetPrefetcher for MockPrefetcher {
        fn prefetch_indices(&self, indices: &[usize]) -> CoreResult<()> {
            let mut prefetched = self.prefetched_indices.lock().unwrap();
            prefetched.extend_from_slice(indices);
            Ok(())
        }

        fn prefetch_all(&self) -> CoreResult<()> {
            let mut prefetched_all = self.prefetched_all.lock().unwrap();
            *prefetched_all = true;
            Ok(())
        }

        fn get_dataset_id(&self) -> DatasetId {
            self.dataset_id.clone()
        }
    }

    #[test]
    fn test_cross_file_prefetching() {
        // Create a manager
        let config = CrossFilePrefetchConfig {
            correlation_threshold: 0.01, // Low threshold for testing
            min_occurrences: 1,
            max_prefetch_datasets: 5,
            ..Default::default()
        };

        let mut manager = CrossFilePrefetchManager::new(config);

        // Create some datasets with mock prefetchers
        let dataset1 = DatasetId::from_name("dataset1");
        let dataset2 = DatasetId::from_name("dataset2");

        let prefetcher1 = Arc::new(MockPrefetcher::new(dataset1.clone()));
        let prefetcher2 = Arc::new(MockPrefetcher::new(dataset2.clone()));

        // Register the datasets
        manager.register_dataset(dataset1.clone(), prefetcher1.clone());
        manager.register_dataset(dataset2.clone(), prefetcher2.clone());

        // Record some related accesses to establish correlation
        for i in 0..3 {
            let access1 = DataAccess {
                dataset: dataset1.clone(),
                index: i,
                access_type: AccessType::Read,
                size: None,
                dimensions: None,
            };

            let access2 = DataAccess {
                dataset: dataset2.clone(),
                index: i * 2,
                access_type: AccessType::Read,
                size: None,
                dimensions: None,
            };

            // Record access to both datasets
            manager.record_access(access1).unwrap();
            manager.record_access(access2).unwrap();
        }

        // Now trigger prefetching by accessing dataset1
        let access = DataAccess {
            dataset: dataset1.clone(),
            index: 10,
            access_type: AccessType::Read,
            size: None,
            dimensions: None,
        };

        manager.record_access(access).unwrap();

        // Check that dataset2 got prefetched
        let prefetched = prefetcher2.prefetched_indices.lock().unwrap();
        assert!(!prefetched.is_empty());

        // Should have prefetched index 20 (10 * 2) based on correlation
        assert!(prefetched.contains(&20));
    }
}
