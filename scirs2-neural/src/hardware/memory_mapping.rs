//! Memory mapping and management for hardware accelerators

use crate::error::Result;
use crate::hardware::accelerator::DeviceBuffer;
use crate::hardware::MemoryStrategy;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
/// Memory requirements for a layer or operation
#[derive(Debug, Clone)]
pub struct MemoryMapRequirements {
    /// Minimum memory required in bytes
    pub min_bytes: usize,
    /// Preferred memory size in bytes
    pub preferred_bytes: usize,
    /// Memory alignment requirement
    pub alignment: usize,
    /// Whether the memory can be shared
    pub can_share: bool,
}
/// Memory statistics
pub struct MemoryStatistics {
    /// Total allocated memory
    pub total_allocated: usize,
    /// Peak allocated memory
    pub peak_allocated: usize,
    /// Number of allocations
    pub num_allocations: usize,
    /// Number of deallocations
    pub num_deallocations: usize,
/// Memory mapper for efficient device memory management
pub struct MemoryMapper {
    device: Arc<dyn crate::hardware::Accelerator>,
    strategy: MemoryStrategy,
    allocations: Arc<Mutex<HashMap<u64, BufferAllocation>>>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    total_allocated: Arc<Mutex<usize>>,
    peak_allocated: Arc<Mutex<usize>>,
impl MemoryMapper {
    /// Create a new memory mapper
    pub fn new(
        device: Arc<dyn crate::hardware::Accelerator>,
        strategy: MemoryStrategy,
    ) -> Result<Self> {
        let pool_size = match strategy {
            MemoryStrategy::PoolBased(size) => size_ => 0,
        };
        Ok(Self {
            device,
            strategy,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(pool_size))),
            total_allocated: Arc::new(Mutex::new(0)),
            peak_allocated: Arc::new(Mutex::new(0)),
        })
    }
    /// Allocate memory with specific layout
    pub fn allocate_with_layout(&self, layout: &MemoryLayout) -> Result<BufferAllocation> {
        match self.strategy {
            MemoryStrategy::Automatic => self.allocate_automatic(layout),
            MemoryStrategy::Preallocated => self.allocate_preallocated(layout),
            MemoryStrategy::OnDemand => self.allocate_on_demand(layout),
            MemoryStrategy::PoolBased(_) => self.allocate_from_pool(layout),
        }
    /// Allocate using automatic strategy
    fn allocate_automatic(&self, layout: &MemoryLayout) -> Result<BufferAllocation> {
        // Try pool first, fall back to on-demand
        if let Ok(alloc) = self.allocate_from_pool(layout) {
            return Ok(alloc);
        self.allocate_on_demand(layout)
    /// Allocate from preallocated memory
    fn allocate_preallocated(&self, layout: &MemoryLayout) -> Result<BufferAllocation> {
        // In real implementation, this would use a preallocated chunk
    /// Allocate on demand
    fn allocate_on_demand(&self, layout: &MemoryLayout) -> Result<BufferAllocation> {
        let aligned_size = Self::align_size(layout.size, layout.alignment);
        let buffer = self.device.allocate(aligned_size)?;
        let allocation = BufferAllocation {
            buffer,
            layout: layout.clone(),
            offset: 0,
            in_use: true,
        // Track allocation
        let mut allocations = self.allocations.lock().unwrap();
        allocations.insert(allocation.buffer.id, allocation.clone());
        // Update statistics
        let mut total = self.total_allocated.lock().unwrap();
        *total += aligned_size;
        let mut peak = self.peak_allocated.lock().unwrap();
        if *total > *peak {
            *peak = *total;
        Ok(allocation)
    /// Allocate from memory pool
    fn allocate_from_pool(&self, layout: &MemoryLayout) -> Result<BufferAllocation> {
        let mut pool = self.memory_pool.lock().unwrap();
        if let Some(buffer) = pool.allocate(layout.size, layout.alignment) {
            let allocation = BufferAllocation {
                buffer,
                layout: layout.clone(),
                offset: 0,
                in_use: true,
            };
            let mut allocations = self.allocations.lock().unwrap();
            allocations.insert(allocation.buffer.id, allocation.clone());
            Ok(allocation)
        } else {
            Err(crate::error::NeuralError::ComputationError(
                "No suitable buffer in pool".to_string(),
            ))
    /// Free an allocation
    pub fn free(&self, bufferid: u64) -> Result<()> {
        if let Some(allocation) = allocations.remove(&buffer_id) {
            let size = allocation.layout.size;
            // Return to pool if using pool strategy
            if matches!(self.strategy, MemoryStrategy::PoolBased(_)) {
                let mut pool = self.memory_pool.lock().unwrap();
                pool.return_buffer(allocation.buffer);
            }
            // Update statistics
            let mut total = self.total_allocated.lock().unwrap();
            *total = total.saturating_sub(size);
            Ok(())
            Err(crate::error::NeuralError::InvalidArgument(format!(
                "Buffer {} not found",
                buffer_id
            )))
    /// Optimize memory layout for multiple allocations
    pub fn optimize_layout(
        &self,
        requirements: &[MemoryMapRequirements],
    ) -> Result<OptimizedLayout> {
        let mut total_size = 0;
        let mut max_alignment = 64;
        let mut layouts = Vec::new();
        for req in requirements {
            let input_layout = MemoryLayout {
                size: req.input_size,
                alignment: req.alignment,
                access_pattern: AccessPattern::Sequential,
                usage: MemoryUsage::ReadOnly,
            let output_layout = MemoryLayout {
                size: req.output_size,
                usage: MemoryUsage::WriteOnly,
            let workspace_layout = MemoryLayout {
                size: req.workspace_size,
                access_pattern: AccessPattern::Random,
                usage: MemoryUsage::ReadWrite,
            total_size += Self::align_size(req.input_size, req.alignment);
            total_size += Self::align_size(req.output_size, req.alignment);
            total_size += Self::align_size(req.workspace_size, req.alignment);
            max_alignment = max_alignment.max(req.alignment);
            layouts.push(input_layout);
            layouts.push(output_layout);
            layouts.push(workspace_layout);
        // Try to coalesce allocations
        let coalesced = self.coalesce_layouts(&layouts)?;
        Ok(OptimizedLayout {
            total_size,
            max_alignment,
            layouts: coalesced,
            estimated_bandwidth: self.estimate_bandwidth(&layouts),
    /// Coalesce multiple layouts into fewer allocations
    fn coalesce_layouts(&self, layouts: &[MemoryLayout]) -> Result<Vec<MemoryLayout>> {
        // Simple strategy: group by usage pattern
        let mut read_only_size = 0;
        let mut write_only_size = 0;
        let mut read_write_size = 0;
        for layout in layouts {
            match layout.usage {
                MemoryUsage::ReadOnly => {
                    read_only_size += Self::align_size(layout.size, layout.alignment)
                }
                MemoryUsage::WriteOnly => {
                    write_only_size += Self::align_size(layout.size, layout.alignment)
                MemoryUsage::ReadWrite => {
                    read_write_size += Self::align_size(layout.size, layout.alignment)
        let mut coalesced = Vec::new();
        if read_only_size > 0 {
            coalesced.push(MemoryLayout {
                size: read_only_size,
                alignment: 64,
            });
        if write_only_size > 0 {
                size: write_only_size,
        if read_write_size > 0 {
                size: read_write_size,
        Ok(coalesced)
    /// Estimate bandwidth requirements
    fn estimate_bandwidth(&self, layouts: &[MemoryLayout]) -> f32 {
        let mut read_bytes = 0;
        let mut write_bytes = 0;
                MemoryUsage::ReadOnly => read_bytes += layout.size,
                MemoryUsage::WriteOnly => write_bytes += layout.size,
                    read_bytes += layout.size;
                    write_bytes += layout.size;
        // Assume 1ms kernel execution time for estimation
        ((read_bytes + write_bytes) as f32) / 1e6
    /// Get memory statistics
    pub fn get_statistics(&self) -> MemoryStatistics {
        let allocations = self.allocations.lock().unwrap();
        let total = *self.total_allocated.lock().unwrap();
        let peak = *self.peak_allocated.lock().unwrap();
        // Calculate fragmentation
        let mut used_blocks = 0;
        let mut total_block_size = 0;
        for (_, alloc) in allocations.iter() {
            if alloc.in_use {
                used_blocks += 1;
                total_block_size += alloc.layout.size;
        let fragmentation = if total > 0 {
            1.0 - (total_block_size as f32 / total as f32)
            0.0
        MemoryStatistics {
            allocated: total,
            used: total_block_size,
            peak,
            num_allocations: allocations.len(),
            fragmentation,
    /// Align size to alignment requirement
    fn align_size(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
/// Memory layout specification
pub struct MemoryLayout {
    /// Size in bytes
    pub size: usize,
    /// Alignment requirement
    /// Access pattern hint
    pub access_pattern: AccessPattern,
    /// Usage hint
    pub usage: MemoryUsage,
/// Memory access pattern
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    Sequential,
    Strided(usize),
    Random,
    Tiled(usize, usize),
/// Memory usage hint
pub enum MemoryUsage {
    ReadOnly,
    WriteOnly,
    ReadWrite,
/// Buffer allocation information
#[derive(Clone)]
pub struct BufferAllocation {
    /// The allocated buffer
    pub buffer: DeviceBuffer,
    /// Memory layout
    pub layout: MemoryLayout,
    /// Offset within buffer
    pub offset: usize,
    /// Whether buffer is currently in use
    pub in_use: bool,
/// Optimized memory layout
pub struct OptimizedLayout {
    /// Total size required
    pub total_size: usize,
    /// Maximum alignment requirement
    pub max_alignment: usize,
    /// Optimized layouts
    pub layouts: Vec<MemoryLayout>,
    /// Estimated bandwidth in GB/s
    pub estimated_bandwidth: f32,
/// Memory pool for buffer reuse
struct MemoryPool {
    /// Available buffers by size
    free_buffers: HashMap<usize, VecDeque<DeviceBuffer>>,
    /// Maximum pool size
    max_size: usize,
    /// Current pool size
    current_size: usize,
impl MemoryPool {
    /// Create a new memory pool
    fn new(_maxsize: usize) -> Self {
        Self {
            free_buffers: HashMap::new(),
            max_size,
            current_size: 0,
    /// Allocate from pool
    fn allocate(&mut self, size: usize, alignment: usize) -> Option<DeviceBuffer> {
        let aligned_size = MemoryMapper::align_size(size, alignment);
        // Look for exact size match first
        if let Some(buffers) = self.free_buffers.get_mut(&aligned_size) {
            if let Some(buffer) = buffers.pop_front() {
                self.current_size -= aligned_size;
                return Some(buffer);
        // Look for larger buffer
        for (&buffer_size, buffers) in self.free_buffers.iter_mut() {
            if buffer_size >= aligned_size && !buffers.is_empty() {
                if let Some(buffer) = buffers.pop_front() {
                    self.current_size -= buffer_size;
                    return Some(buffer);
        None
    /// Return buffer to pool
    fn return_buffer(&mut self, buffer: DeviceBuffer) {
        if self.current_size + buffer.size <= self.max_size {
            let size = buffer.size;
            self.free_buffers
                .entry(size)
                .or_insert_with(VecDeque::new)
                .push_back(buffer);
            self.current_size += size;
        // Otherwise, let the buffer be deallocated
/// Memory migration helper for multi-device scenarios
pub struct MemoryMigration;
impl MemoryMigration {
    /// Migrate buffer between devices
    pub fn migrate_buffer(
        src_device: &dyn crate::hardware::Accelerator,
        dst_device: &dyn crate::hardware::Accelerator,
        buffer: &DeviceBuffer,
    ) -> Result<DeviceBuffer> {
        // Download from source
        let data = src_device.download(buffer)?;
        // Upload to destination
        dst_device.upload(&data.view())
    /// Migrate with transformation
    pub fn migrate_with_transform<F>(
        transform: F,
    ) -> Result<DeviceBuffer>
    where
        F: FnOnce(scirs2_core::ndarray::Array2<f32>) -> scirs2_core::ndarray::Array2<f32>,
    {
        // Apply transformation
        let transformed = transform(data);
        dst_device.upload(&transformed.view())
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::accelerator::CPUAccelerator;
    #[test]
    fn test_memory_layout() {
        let layout = MemoryLayout {
            size: 1024,
            alignment: 64,
            access_pattern: AccessPattern::Sequential,
            usage: MemoryUsage::ReadOnly,
        assert_eq!(layout.size, 1024);
        assert_eq!(layout.alignment, 64);
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(10 * 1024 * 1024); // 10MB pool
        let buffer = DeviceBuffer::new(Box::into_raw(Box::new([0u8; 1024])) as *mut u8, 1024, 0);
        pool.return_buffer(buffer);
        assert_eq!(pool.current_size, 1024);
        let allocated = pool.allocate(1024, 64);
        assert!(allocated.is_some());
        assert_eq!(pool.current_size, 0);
    fn test_align_size() {
        assert_eq!(MemoryMapper::align_size(100, 64), 128);
        assert_eq!(MemoryMapper::align_size(64, 64), 64);
        assert_eq!(MemoryMapper::align_size(65, 64), 128);
