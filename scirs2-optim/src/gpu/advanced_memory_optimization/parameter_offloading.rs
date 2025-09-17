//! Parameter offloading for memory-efficient large model training
//!
//! This module provides parameter offloading capabilities to move parameters
//! between GPU, CPU memory, and disk storage based on access patterns and
//! memory pressure.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::path::PathBuf;
use ndarray::Array1;
use num_traits::Float;

use crate::error::{OptimError, Result};
use super::config::{StorageLocation, CompressionType, CompressionInfo, TransferCost, OffloadStrategy};

/// Parameter offload manager
#[derive(Debug)]
pub struct ParameterOffloadManager<T: Float> {
    /// Currently offloaded parameters
    offloaded_params: HashMap<String, OffloadedParameter<T>>,

    /// Offloading strategy
    strategy: OffloadStrategy,

    /// CPU memory pool for offloaded parameters
    cpu_memory_pool: Option<CpuMemoryPool<T>>,

    /// Disk storage manager
    disk_storage: Option<DiskStorage<T>>,

    /// Prefetch queue for predicted parameter accesses
    prefetch_queue: VecDeque<String>,

    /// Access pattern predictor
    access_predictor: AccessPatternPredictor,

    /// Total memory saved through offloading (bytes)
    memory_saved: usize,

    /// Offloading statistics
    stats: OffloadingStats,

    /// Size threshold for automatic offloading
    size_threshold: usize,

    /// Memory pressure threshold for triggering offloads
    memory_pressure_threshold: f32,

    /// Base directory for disk storage
    storage_directory: PathBuf,
}

impl<T: Float + Clone + Default> ParameterOffloadManager<T> {
    /// Create a new parameter offload manager
    pub fn new(strategy: OffloadStrategy, size_threshold: usize) -> Self {
        Self {
            offloaded_params: HashMap::new(),
            strategy,
            cpu_memory_pool: Some(CpuMemoryPool::new()),
            disk_storage: Some(DiskStorage::new()),
            prefetch_queue: VecDeque::new(),
            access_predictor: AccessPatternPredictor::new(),
            memory_saved: 0,
            stats: OffloadingStats::default(),
            size_threshold,
            memory_pressure_threshold: 0.8,
            storage_directory: PathBuf::from("/tmp/parameter_offload"),
        }
    }

    /// Configure offload manager parameters
    pub fn configure(
        &mut self,
        strategy: OffloadStrategy,
        size_threshold: usize,
        memory_pressure_threshold: f32,
        storage_directory: PathBuf,
    ) {
        self.strategy = strategy;
        self.size_threshold = size_threshold;
        self.memory_pressure_threshold = memory_pressure_threshold;
        self.storage_directory = storage_directory;
    }

    /// Decide whether to offload a parameter
    pub fn should_offload(&self, param_name: &str, param_size: usize, memory_pressure: f32) -> bool {
        match self.strategy {
            OffloadStrategy::SizeBased(threshold) => param_size >= threshold,
            OffloadStrategy::FrequencyBased => {
                self.access_predictor.get_access_frequency(param_name) < 0.1
            }
            OffloadStrategy::CostBenefit => {
                self.calculate_offload_benefit(param_name, param_size, memory_pressure) > 0.0
            }
            OffloadStrategy::PressureDriven => memory_pressure > self.memory_pressure_threshold,
            OffloadStrategy::LRU => {
                self.access_predictor.get_last_access_time(param_name)
                    .map(|t| t.elapsed().as_secs() > 60)
                    .unwrap_or(true)
            }
        }
    }

    /// Offload a parameter to appropriate storage
    pub fn offload_parameter(
        &mut self,
        param_name: String,
        parameter: Array1<T>,
        compression: Option<CompressionType>,
    ) -> Result<()> {
        let param_size = parameter.len() * std::mem::size_of::<T>();
        let shape = vec![parameter.len()];

        // Choose storage location based on size and strategy
        let storage_location = self.choose_storage_location(param_size, compression.is_some())?;

        // Apply compression if requested
        let (final_location, compression_info) = if let Some(comp_type) = compression {
            self.compress_and_store(param_name.clone(), parameter, comp_type, storage_location)?
        } else {
            (self.store_uncompressed(parameter, storage_location)?, None)
        };

        // Create offloaded parameter record
        let offloaded_param = OffloadedParameter {
            name: param_name.clone(),
            shape,
            location: final_location,
            offload_time: Instant::now(),
            access_frequency: self.access_predictor.get_access_frequency(&param_name),
            transfer_cost: TransferCost::default(),
            compression: compression_info,
        };

        // Update tracking
        self.offloaded_params.insert(param_name.clone(), offloaded_param);
        self.memory_saved += param_size;
        self.stats.total_offloaded += 1;
        self.stats.bytes_offloaded += param_size;

        // Record access pattern
        self.access_predictor.record_offload(&param_name);

        Ok(())
    }

    /// Retrieve an offloaded parameter
    pub fn retrieve_parameter(&mut self, param_name: &str) -> Result<Array1<T>> {
        let offloaded_param = self.offloaded_params.get(param_name)
            .ok_or_else(|| OptimError::InvalidParameter(format!("Parameter {} not offloaded", param_name)))?;

        let start_time = Instant::now();

        // Load parameter from storage
        let parameter = self.load_from_storage(&offloaded_param.location, &offloaded_param.shape)?;

        // Decompress if necessary
        let final_parameter = if let Some(compression_info) = &offloaded_param.compression {
            self.decompress_parameter(parameter, compression_info)?
        } else {
            parameter
        };

        // Update statistics
        let retrieval_time = start_time.elapsed();
        self.stats.total_retrieved += 1;
        self.stats.total_retrieval_time += retrieval_time;

        // Record access pattern
        self.access_predictor.record_access(param_name);

        Ok(final_parameter)
    }

    /// Remove parameter from offload storage
    pub fn remove_offloaded_parameter(&mut self, param_name: &str) -> Result<()> {
        if let Some(offloaded_param) = self.offloaded_params.remove(param_name) {
            // Clean up storage
            self.cleanup_storage(&offloaded_param.location)?;

            // Update tracking
            let param_size = offloaded_param.shape.iter().product::<usize>() * std::mem::size_of::<T>();
            self.memory_saved = self.memory_saved.saturating_sub(param_size);
            self.stats.total_removed += 1;
        }
        Ok(())
    }

    /// Prefetch parameters based on predicted access patterns
    pub fn prefetch_parameters(&mut self) -> Result<()> {
        let predictions = self.access_predictor.predict_next_accesses(5);

        for param_name in predictions {
            if self.offloaded_params.contains_key(&param_name) && !self.prefetch_queue.contains(&param_name) {
                self.prefetch_queue.push_back(param_name);
            }
        }

        // Process prefetch queue
        while let Some(param_name) = self.prefetch_queue.pop_front() {
            // In a real implementation, this would trigger asynchronous loading
            self.stats.total_prefetched += 1;
        }

        Ok(())
    }

    /// Get offloading statistics
    pub fn get_statistics(&self) -> &OffloadingStats {
        &self.stats
    }

    /// Get memory savings from offloading
    pub fn get_memory_savings(&self) -> usize {
        self.memory_saved
    }

    /// Get list of offloaded parameters
    pub fn get_offloaded_parameters(&self) -> Vec<&str> {
        self.offloaded_params.keys().map(|s| s.as_str()).collect()
    }

    /// Check if parameter is offloaded
    pub fn is_offloaded(&self, param_name: &str) -> bool {
        self.offloaded_params.contains_key(param_name)
    }

    // Private helper methods

    fn choose_storage_location(&self, param_size: usize, compressed: bool) -> Result<StorageLocation> {
        // Choose storage based on size and availability
        if param_size < 100 * 1024 * 1024 && self.cpu_memory_pool.is_some() {
            // Small parameters go to CPU memory
            Ok(StorageLocation::CpuMemory {
                ptr: std::ptr::null_mut(),
                size: param_size
            })
        } else if self.disk_storage.is_some() {
            // Large parameters go to disk
            let file_path = self.storage_directory.join(format!("param_{}.bin", uuid::Uuid::new_v4()));
            Ok(StorageLocation::DiskStorage {
                file_path: file_path.to_string_lossy().to_string(),
                offset: 0
            })
        } else {
            Err(OptimError::ResourceError("No storage location available".to_string()))
        }
    }

    fn compress_and_store(
        &self,
        param_name: String,
        parameter: Array1<T>,
        compression_type: CompressionType,
        storage_location: StorageLocation,
    ) -> Result<(StorageLocation, Option<CompressionInfo>)> {
        let start_time = Instant::now();

        // Simulate compression (in practice, would use actual compression libraries)
        let compressed_data = self.compress_parameter(&parameter, compression_type)?;
        let compression_time = start_time.elapsed();

        let compression_info = CompressionInfo {
            algorithm: compression_type,
            ratio: compressed_data.len() as f32 / (parameter.len() * std::mem::size_of::<T>()) as f32,
            compression_time,
            decompression_time: Duration::ZERO, // Will be measured during decompression
            quality_loss: if compression_type.is_lossy() { Some(0.01) } else { None },
        };

        let compressed_location = StorageLocation::Compressed {
            data: compressed_data,
            compression_type,
        };

        Ok((compressed_location, Some(compression_info)))
    }

    fn store_uncompressed(&self, parameter: Array1<T>, location: StorageLocation) -> Result<StorageLocation> {
        match location {
            StorageLocation::CpuMemory { size, .. } => {
                // In practice, would allocate CPU memory and copy data
                Ok(StorageLocation::CpuMemory {
                    ptr: std::ptr::null_mut(),
                    size
                })
            }
            StorageLocation::DiskStorage { file_path, .. } => {
                // In practice, would write to disk file
                Ok(StorageLocation::DiskStorage { file_path, offset: 0 })
            }
            other => Ok(other),
        }
    }

    fn load_from_storage(&self, location: &StorageLocation, shape: &[usize]) -> Result<Array1<T>> {
        match location {
            StorageLocation::CpuMemory { size, .. } => {
                // In practice, would load from CPU memory
                Ok(Array1::zeros(*size / std::mem::size_of::<T>()))
            }
            StorageLocation::DiskStorage { file_path, .. } => {
                // In practice, would read from disk file
                let total_elements = shape.iter().product();
                Ok(Array1::zeros(total_elements))
            }
            StorageLocation::Compressed { data, compression_type } => {
                // Data is compressed and needs decompression
                self.decompress_data(data, *compression_type, shape)
            }
            StorageLocation::RemoteStorage { .. } => {
                Err(OptimError::UnsupportedOperation("Remote storage not implemented".to_string()))
            }
        }
    }

    fn compress_parameter(&self, parameter: &Array1<T>, compression_type: CompressionType) -> Result<Vec<u8>> {
        // Simulate compression by reducing size based on compression type
        let original_size = parameter.len() * std::mem::size_of::<T>();
        let compressed_size = (original_size as f32 * compression_type.expected_ratio()) as usize;

        // In practice, would use actual compression algorithms
        Ok(vec![0u8; compressed_size])
    }

    fn decompress_parameter(&self, compressed: Array1<T>, compression_info: &CompressionInfo) -> Result<Array1<T>> {
        // In practice, would perform actual decompression
        let decompressed_size = (compressed.len() as f32 / compression_info.ratio) as usize;
        Ok(Array1::zeros(decompressed_size))
    }

    fn decompress_data(&self, data: &[u8], compression_type: CompressionType, shape: &[usize]) -> Result<Array1<T>> {
        // In practice, would perform actual decompression
        let total_elements = shape.iter().product();
        Ok(Array1::zeros(total_elements))
    }

    fn cleanup_storage(&self, location: &StorageLocation) -> Result<()> {
        match location {
            StorageLocation::DiskStorage { file_path, .. } => {
                // In practice, would delete the file
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn calculate_offload_benefit(&self, param_name: &str, param_size: usize, memory_pressure: f32) -> f32 {
        let access_frequency = self.access_predictor.get_access_frequency(param_name);
        let size_factor = param_size as f32 / 1024.0 / 1024.0; // Size in MB
        let pressure_factor = memory_pressure;

        // Benefit = memory savings - access cost
        size_factor * pressure_factor - access_frequency * 10.0
    }
}

/// Information about an offloaded parameter
#[derive(Debug, Clone)]
pub struct OffloadedParameter<T: Float> {
    /// Parameter name
    pub name: String,

    /// Original parameter shape
    pub shape: Vec<usize>,

    /// Current storage location
    pub location: StorageLocation,

    /// When parameter was offloaded
    pub offload_time: Instant,

    /// Access frequency (accesses per minute)
    pub access_frequency: f32,

    /// Transfer cost analysis
    pub transfer_cost: TransferCost,

    /// Compression information if compressed
    pub compression: Option<CompressionInfo>,
}

impl<T: Float> OffloadedParameter<T> {
    /// Get the time since offload
    pub fn time_since_offload(&self) -> Duration {
        Instant::now().duration_since(self.offload_time)
    }

    /// Calculate priority for retrieval (higher = more likely to retrieve)
    pub fn retrieval_priority(&self) -> f32 {
        let time_factor = self.time_since_offload().as_secs_f32() / 3600.0; // Hours
        let access_factor = self.access_frequency;

        access_factor / (time_factor + 1.0)
    }

    /// Estimate retrieval cost
    pub fn estimate_retrieval_cost(&self) -> f32 {
        match &self.location {
            StorageLocation::CpuMemory { size, .. } => {
                *size as f32 * self.transfer_cost.cpu_to_gpu_cost
            }
            StorageLocation::DiskStorage { .. } => {
                let size = self.shape.iter().product::<usize>() * std::mem::size_of::<T>();
                size as f32 * self.transfer_cost.disk_io_cost
            }
            StorageLocation::Compressed { data, .. } => {
                data.len() as f32 * self.transfer_cost.decompression_cost
            }
            StorageLocation::RemoteStorage { .. } => {
                let size = self.shape.iter().product::<usize>() * std::mem::size_of::<T>();
                size as f32 * self.transfer_cost.network_cost
            }
        }
    }
}

/// CPU memory pool for offloaded parameters
#[derive(Debug)]
pub struct CpuMemoryPool<T: Float> {
    /// Allocated memory blocks
    memory_blocks: HashMap<String, Vec<T>>,

    /// Total allocated memory (bytes)
    total_allocated: usize,

    /// Maximum pool size (bytes)
    max_pool_size: usize,
}

impl<T: Float + Clone + Default> CpuMemoryPool<T> {
    /// Create a new CPU memory pool
    pub fn new() -> Self {
        Self {
            memory_blocks: HashMap::new(),
            total_allocated: 0,
            max_pool_size: 1024 * 1024 * 1024, // 1GB default
        }
    }

    /// Allocate memory block
    pub fn allocate(&mut self, name: String, size: usize) -> Result<()> {
        if self.total_allocated + size * std::mem::size_of::<T>() > self.max_pool_size {
            return Err(OptimError::ResourceError("CPU memory pool exhausted".to_string()));
        }

        let block = vec![T::default(); size];
        self.memory_blocks.insert(name, block);
        self.total_allocated += size * std::mem::size_of::<T>();

        Ok(())
    }

    /// Deallocate memory block
    pub fn deallocate(&mut self, name: &str) -> Result<()> {
        if let Some(block) = self.memory_blocks.remove(name) {
            self.total_allocated -= block.len() * std::mem::size_of::<T>();
        }
        Ok(())
    }

    /// Get memory utilization
    pub fn utilization(&self) -> f32 {
        self.total_allocated as f32 / self.max_pool_size as f32
    }
}

/// Disk storage manager for large parameters
#[derive(Debug)]
pub struct DiskStorage<T: Float> {
    /// File handles for stored parameters
    file_handles: HashMap<String, PathBuf>,

    /// Total disk usage (bytes)
    total_disk_usage: usize,

    /// Base storage directory
    storage_directory: PathBuf,
}

impl<T: Float> DiskStorage<T> {
    /// Create a new disk storage manager
    pub fn new() -> Self {
        Self {
            file_handles: HashMap::new(),
            total_disk_usage: 0,
            storage_directory: PathBuf::from("/tmp/parameter_storage"),
        }
    }

    /// Store parameter to disk
    pub fn store(&mut self, name: String, data: &[u8]) -> Result<PathBuf> {
        let file_path = self.storage_directory.join(format!("{}.bin", name));

        // In practice, would write data to file
        self.file_handles.insert(name, file_path.clone());
        self.total_disk_usage += data.len();

        Ok(file_path)
    }

    /// Load parameter from disk
    pub fn load(&self, name: &str) -> Result<Vec<u8>> {
        let _file_path = self.file_handles.get(name)
            .ok_or_else(|| OptimError::InvalidParameter(format!("Parameter {} not found on disk", name)))?;

        // In practice, would read data from file
        Ok(Vec::new())
    }

    /// Remove parameter from disk
    pub fn remove(&mut self, name: &str) -> Result<()> {
        if let Some(_file_path) = self.file_handles.remove(name) {
            // In practice, would delete the file and update disk usage
        }
        Ok(())
    }

    /// Get disk usage
    pub fn get_disk_usage(&self) -> usize {
        self.total_disk_usage
    }
}

/// Access pattern predictor for intelligent prefetching
#[derive(Debug)]
pub struct AccessPatternPredictor {
    /// Access history for each parameter
    access_history: HashMap<String, Vec<Instant>>,

    /// Last access time for each parameter
    last_access: HashMap<String, Instant>,

    /// Access frequency cache
    frequency_cache: HashMap<String, f32>,

    /// Prediction model (simplified)
    prediction_window: Duration,
}

impl AccessPatternPredictor {
    /// Create a new access pattern predictor
    pub fn new() -> Self {
        Self {
            access_history: HashMap::new(),
            last_access: HashMap::new(),
            frequency_cache: HashMap::new(),
            prediction_window: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Record parameter access
    pub fn record_access(&mut self, param_name: &str) {
        let now = Instant::now();

        self.access_history
            .entry(param_name.to_string())
            .or_insert_with(Vec::new)
            .push(now);

        self.last_access.insert(param_name.to_string(), now);

        // Update frequency cache
        self.update_frequency_cache(param_name);
    }

    /// Record parameter offload
    pub fn record_offload(&mut self, param_name: &str) {
        // Clear recent access history since parameter is being offloaded
        if let Some(history) = self.access_history.get_mut(param_name) {
            history.clear();
        }
        self.frequency_cache.insert(param_name.to_string(), 0.0);
    }

    /// Get access frequency (accesses per minute)
    pub fn get_access_frequency(&self, param_name: &str) -> f32 {
        self.frequency_cache.get(param_name).copied().unwrap_or(0.0)
    }

    /// Get last access time
    pub fn get_last_access_time(&self, param_name: &str) -> Option<Instant> {
        self.last_access.get(param_name).copied()
    }

    /// Predict next parameters to be accessed
    pub fn predict_next_accesses(&self, count: usize) -> Vec<String> {
        let mut predictions: Vec<(String, f32)> = self.frequency_cache
            .iter()
            .map(|(name, &freq)| (name.clone(), freq))
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        predictions.into_iter()
            .take(count)
            .map(|(name, _)| name)
            .collect()
    }

    fn update_frequency_cache(&mut self, param_name: &str) {
        if let Some(history) = self.access_history.get(param_name) {
            let recent_accesses = history.iter()
                .filter(|&&access_time| Instant::now().duration_since(access_time) <= self.prediction_window)
                .count();

            let frequency = recent_accesses as f32 / self.prediction_window.as_secs_f32() * 60.0; // Per minute
            self.frequency_cache.insert(param_name.to_string(), frequency);
        }
    }
}

/// Offloading performance statistics
#[derive(Debug, Clone, Default)]
pub struct OffloadingStats {
    /// Total parameters offloaded
    pub total_offloaded: usize,

    /// Total parameters retrieved
    pub total_retrieved: usize,

    /// Total parameters removed from offload
    pub total_removed: usize,

    /// Total parameters prefetched
    pub total_prefetched: usize,

    /// Total bytes offloaded
    pub bytes_offloaded: usize,

    /// Total time spent on retrieval
    pub total_retrieval_time: Duration,

    /// Average retrieval time
    pub avg_retrieval_time: Duration,

    /// Cache hit rate for predictions
    pub prediction_hit_rate: f32,
}

impl OffloadingStats {
    /// Calculate offload ratio
    pub fn offload_ratio(&self) -> f32 {
        let total_operations = self.total_offloaded + self.total_retrieved;
        if total_operations > 0 {
            self.total_offloaded as f32 / total_operations as f32
        } else {
            0.0
        }
    }

    /// Update average retrieval time
    pub fn update_avg_retrieval_time(&mut self) {
        if self.total_retrieved > 0 {
            self.avg_retrieval_time = self.total_retrieval_time / self.total_retrieved as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_offload_manager_creation() {
        let manager = ParameterOffloadManager::<f32>::new(OffloadStrategy::SizeBased(1024), 1024);
        assert_eq!(manager.size_threshold, 1024);
        assert_eq!(manager.offloaded_params.len(), 0);
    }

    #[test]
    fn test_should_offload_size_based() {
        let manager = ParameterOffloadManager::<f32>::new(OffloadStrategy::SizeBased(1024), 1024);

        assert!(manager.should_offload("param1", 2048, 0.5)); // Above threshold
        assert!(!manager.should_offload("param2", 512, 0.5)); // Below threshold
    }

    #[test]
    fn test_should_offload_pressure_driven() {
        let manager = ParameterOffloadManager::<f32>::new(OffloadStrategy::PressureDriven, 1024);

        assert!(manager.should_offload("param1", 1024, 0.9)); // High pressure
        assert!(!manager.should_offload("param2", 1024, 0.7)); // Low pressure
    }

    #[test]
    fn test_access_pattern_predictor() {
        let mut predictor = AccessPatternPredictor::new();

        predictor.record_access("param1");
        predictor.record_access("param1");
        predictor.record_access("param2");

        assert!(predictor.get_access_frequency("param1") >= predictor.get_access_frequency("param2"));

        let predictions = predictor.predict_next_accesses(2);
        assert!(predictions.contains(&"param1".to_string()));
    }

    #[test]
    fn test_cpu_memory_pool() {
        let mut pool = CpuMemoryPool::<f32>::new();

        assert!(pool.allocate("block1".to_string(), 1000).is_ok());
        assert!(pool.utilization() > 0.0);

        assert!(pool.deallocate("block1").is_ok());
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn test_offloaded_parameter_metrics() {
        let param = OffloadedParameter {
            name: "test_param".to_string(),
            shape: vec![1000],
            location: StorageLocation::CpuMemory { ptr: std::ptr::null_mut(), size: 4000 },
            offload_time: Instant::now(),
            access_frequency: 2.0,
            transfer_cost: TransferCost::default(),
            compression: None,
        };

        assert!(param.retrieval_priority() > 0.0);
        assert!(param.estimate_retrieval_cost() > 0.0);
    }

    #[test]
    fn test_offloading_stats() {
        let mut stats = OffloadingStats::default();
        stats.total_offloaded = 10;
        stats.total_retrieved = 5;

        assert_eq!(stats.offload_ratio(), 10.0 / 15.0);

        stats.total_retrieval_time = Duration::from_millis(1000);
        stats.update_avg_retrieval_time();
        assert_eq!(stats.avg_retrieval_time, Duration::from_millis(200));
    }
}