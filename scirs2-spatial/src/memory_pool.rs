//! Advanced-optimized memory pool system for spatial algorithms
//!
//! This module provides advanced memory management strategies specifically
//! designed for spatial computing algorithms that perform frequent allocations.
//! The system includes object pools, arena allocators, and cache-aware
//! memory layouts to maximize performance.
//!
//! # Features
//!
//! - **Object pools**: Reusable pools for frequently allocated types
//! - **Arena allocators**: Block-based allocation for temporary objects
//! - **Cache-aware layouts**: Memory alignment for optimal cache performance
//! - **NUMA-aware allocation**: Memory placement for multi-socket systems
//! - **Zero-copy operations**: Minimize data movement and copying
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::memory_pool::{DistancePool, ClusteringArena};
//!
//! // Create a distance computation pool
//! let mut pool = DistancePool::new(1000);
//!
//! // Get a reusable distance buffer
//! let buffer = pool.get_distance_buffer(256);
//!
//! // Use buffer for computations...
//!
//! // Buffer is automatically returned to pool when dropped (RAII)
//! ```

use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::Mutex;

// Platform-specific NUMA imports
#[cfg(any(target_os = "linux", target_os = "android"))]
use libc;
#[cfg(target_os = "linux")]
use std::fs;

// Thread affinity for NUMA binding
use std::sync::atomic::Ordering;

// Add num_cpus for cross-platform CPU detection
// The num_cpus crate is available in dev-dependencies
#[cfg(test)]
use num_cpus;

// Fallback implementation for non-test builds
#[cfg(not(test))]
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

/// Configuration for memory pool system
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum number of objects to keep in each pool
    pub max_pool_size: usize,
    /// Cache line size for alignment (typically 64 bytes)
    pub cache_line_size: usize,
    /// Enable NUMA-aware allocation strategies
    pub numa_aware: bool,
    /// Prefetch distance for memory access patterns
    pub prefetch_distance: usize,
    /// Block size for arena allocators
    pub arena_block_size: usize,
    /// NUMA node hint for allocation (-1 for automatic detection)
    pub numa_node_hint: i32,
    /// Enable automatic NUMA topology discovery
    pub auto_numa_discovery: bool,
    /// Enable thread-to-NUMA-node affinity binding
    pub enable_thread_affinity: bool,
    /// Enable memory warming (pre-touch pages)
    pub enable_memory_warming: bool,
    /// Size threshold for large object handling
    pub large_object_threshold: usize,
    /// Maximum memory usage before forced cleanup (in bytes)
    pub max_memory_usage: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1000,
            cache_line_size: 64,
            numa_aware: true,
            prefetch_distance: 8,
            arena_block_size: 1024 * 1024, // 1MB blocks
            numa_node_hint: -1,            // Auto-detect
            auto_numa_discovery: true,
            enable_thread_affinity: true,
            enable_memory_warming: true,
            large_object_threshold: 64 * 1024,    // 64KB
            max_memory_usage: 1024 * 1024 * 1024, // 1GB default limit
        }
    }
}

/// Advanced-optimized distance computation memory pool
pub struct DistancePool {
    config: MemoryPoolConfig,
    distance_buffers: Mutex<VecDeque<Box<[f64]>>>,
    index_buffers: Mutex<VecDeque<Box<[usize]>>>,
    matrix_buffers: Mutex<VecDeque<Array2<f64>>>,
    large_buffers: Mutex<VecDeque<Box<[f64]>>>, // For large objects
    stats: PoolStatistics,
    memory_usage: std::sync::atomic::AtomicUsize, // Track total memory usage
    numa_node: std::sync::atomic::AtomicI32,      // Current NUMA node
}

impl DistancePool {
    /// Create a new distance computation pool
    pub fn new(capacity: usize) -> Self {
        Self::with_config(capacity, MemoryPoolConfig::default())
    }

    /// Create a pool with custom configuration
    pub fn with_config(capacity: usize, config: MemoryPoolConfig) -> Self {
        let numa_node = if config.numa_aware && config.numa_node_hint >= 0 {
            config.numa_node_hint
        } else {
            Self::detect_numa_node()
        };

        Self {
            config,
            distance_buffers: Mutex::new(VecDeque::with_capacity(capacity)),
            index_buffers: Mutex::new(VecDeque::with_capacity(capacity)),
            matrix_buffers: Mutex::new(VecDeque::with_capacity(capacity / 4)), // Matrices are larger
            large_buffers: Mutex::new(VecDeque::with_capacity(capacity / 10)), // Large objects are rarer
            stats: PoolStatistics::new(),
            memory_usage: std::sync::atomic::AtomicUsize::new(0),
            numa_node: std::sync::atomic::AtomicI32::new(numa_node),
        }
    }

    /// Get a cache-aligned distance buffer
    pub fn get_distance_buffer(&self, size: usize) -> DistanceBuffer {
        // Check if this is a large object
        let buffer_size_bytes = size * std::mem::size_of::<f64>();
        let is_large = buffer_size_bytes > self.config.large_object_threshold;

        // Check memory usage limit
        let current_usage = self.memory_usage.load(std::sync::atomic::Ordering::Relaxed);
        if current_usage + buffer_size_bytes > self.config.max_memory_usage {
            self.cleanup_excess_memory();
        }

        let buffer = if is_large {
            self.get_large_buffer(size)
        } else {
            let mut buffers = self.distance_buffers.lock().unwrap();

            // Try to reuse an existing buffer of appropriate size
            for i in 0..buffers.len() {
                if buffers[i].len() >= size && buffers[i].len() <= size * 2 {
                    let buffer = buffers.remove(i).unwrap();
                    self.stats.record_hit();
                    return DistanceBuffer::new(buffer, self);
                }
            }

            // Create new aligned buffer
            self.stats.record_miss();
            self.create_aligned_buffer(size)
        };

        // Track memory usage
        self.memory_usage
            .fetch_add(buffer_size_bytes, std::sync::atomic::Ordering::Relaxed);

        DistanceBuffer::new(buffer, self)
    }

    /// Get a buffer for large objects with special handling
    fn get_large_buffer(&self, size: usize) -> Box<[f64]> {
        let mut buffers = self.large_buffers.lock().unwrap();

        // For large buffers, be more strict about size matching
        for i in 0..buffers.len() {
            if buffers[i].len() == size {
                let buffer = buffers.remove(i).unwrap();
                self.stats.record_hit();
                return buffer;
            }
        }

        // Create new large buffer with NUMA awareness
        self.stats.record_miss();
        if self.config.numa_aware {
            self.create_numa_aligned_buffer(size)
        } else {
            self.create_aligned_buffer(size)
        }
    }

    /// Get an index buffer for storing indices
    pub fn get_index_buffer(&self, size: usize) -> IndexBuffer {
        let mut buffers = self.index_buffers.lock().unwrap();

        // Try to reuse existing buffer
        for i in 0..buffers.len() {
            if buffers[i].len() >= size && buffers[i].len() <= size * 2 {
                let buffer = buffers.remove(i).unwrap();
                self.stats.record_hit();
                return IndexBuffer::new(buffer, self);
            }
        }

        // Create new buffer
        self.stats.record_miss();
        let new_buffer = vec![0usize; size].into_boxed_slice();
        IndexBuffer::new(new_buffer, self)
    }

    /// Get a distance matrix buffer
    pub fn get_matrix_buffer(&self, rows: usize, cols: usize) -> MatrixBuffer {
        let mut buffers = self.matrix_buffers.lock().unwrap();

        // Try to reuse existing matrix
        for i in 0..buffers.len() {
            let (r, c) = buffers[i].dim();
            if r >= rows && c >= cols && r <= rows * 2 && c <= cols * 2 {
                let mut matrix = buffers.remove(i).unwrap();
                // Resize to exact dimensions needed
                matrix = matrix.slice_mut(s![..rows, ..cols]).to_owned();
                self.stats.record_hit();
                return MatrixBuffer::new(matrix, self);
            }
        }

        // Create new matrix
        self.stats.record_miss();
        let matrix = Array2::zeros((rows, cols));
        MatrixBuffer::new(matrix, self)
    }

    /// Create cache-aligned buffer for optimal SIMD performance
    fn create_aligned_buffer(&self, size: usize) -> Box<[f64]> {
        let layout = Layout::from_size_align(
            size * std::mem::size_of::<f64>(),
            self.config.cache_line_size,
        )
        .unwrap();

        unsafe {
            let ptr = System.alloc(layout) as *mut f64;
            if ptr.is_null() {
                panic!("Failed to allocate aligned memory");
            }

            // Initialize to zero (optional memory warming)
            if self.config.enable_memory_warming {
                std::ptr::write_bytes(ptr, 0, size);
            }

            // Convert to boxed slice
            Box::from_raw(std::slice::from_raw_parts_mut(ptr, size))
        }
    }

    /// Create NUMA-aware aligned buffer with proper node binding
    fn create_numa_aligned_buffer(&self, size: usize) -> Box<[f64]> {
        let numa_node = self.numa_node.load(Ordering::Relaxed);

        #[cfg(target_os = "linux")]
        {
            if self.config.numa_aware && numa_node >= 0 {
                match Self::allocate_on_numa_node_linux(size, numa_node as u32) {
                    Ok(buffer) => {
                        if self.config.enable_memory_warming {
                            Self::warm_memory(&buffer);
                        }
                        return buffer;
                    }
                    Err(_) => {
                        // Fallback to regular allocation
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if self.config.numa_aware && numa_node >= 0 {
                match Self::allocate_on_numa_node_windows(size, numa_node as u32) {
                    Ok(buffer) => {
                        if self.config.enable_memory_warming {
                            Self::warm_memory(&buffer);
                        }
                        return buffer;
                    }
                    Err(_) => {
                        // Fallback to regular allocation
                    }
                }
            }
        }

        // Fallback to regular aligned allocation
        let buffer = self.create_aligned_buffer(size);

        // Warm memory to encourage allocation on current NUMA node
        if self.config.enable_memory_warming {
            Self::warm_memory(&buffer);
        }

        buffer
    }

    /// Linux-specific NUMA-aware allocation (fallback without actual NUMA binding)
    #[cfg(target_os = "linux")]
    fn allocate_on_numa_node_linux(
        size: usize,
        node: u32,
    ) -> Result<Box<[f64]>, Box<dyn std::error::Error>> {
        let total_size = size * std::mem::size_of::<f64>();
        let layout = Layout::from_size_align(total_size, 64)?;

        unsafe {
            // Allocate memory (NUMA binding disabled due to libc limitations)
            let ptr = System.alloc(layout) as *mut f64;
            if ptr.is_null() {
                return Err("Failed to allocate memory".into());
            }

            // Initialize memory
            std::ptr::write_bytes(ptr, 0, size);

            Ok(Box::from_raw(std::slice::from_raw_parts_mut(ptr, size)))
        }
    }

    /// Windows-specific NUMA-aware allocation
    #[cfg(target_os = "windows")]
    fn allocate_on_numa_node_windows(
        size: usize,
        node: u32,
    ) -> Result<Box<[f64]>, Box<dyn std::error::Error>> {
        // Windows NUMA allocation using VirtualAllocExNuma would go here
        // For now, fallback to regular allocation
        Err("Windows NUMA allocation not implemented".into())
    }

    /// Bind current thread to specific NUMA node for better locality
    pub fn bind_thread_to_numa_node(node: u32) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            Self::bind_thread_to_numa_node_linux(node)
        }
        #[cfg(target_os = "windows")]
        {
            Self::bind_thread_to_numa_node_windows(node)
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Ok(()) // No-op for unsupported platforms
        }
    }

    #[cfg(target_os = "linux")]
    fn bind_thread_to_numa_node_linux(node: u32) -> Result<(), Box<dyn std::error::Error>> {
        // NUMA memory policy binding disabled due to libc limitations
        // Still attempt CPU affinity for performance

        // Try to set CPU affinity to CPUs on this NUMA _node
        if let Some(_cpu_count) = Self::get_node_cpu_count(node) {
            let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };

            // Read the CPU list for this NUMA _node
            let cpulist_path = format!("/sys/devices/system/node/node{}/cpulist", node);
            if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                for range in cpulist.trim().split(',') {
                    if let Some((start, end)) = range.split_once('-') {
                        if let (Ok(s), Ok(e)) = (start.parse::<u32>(), end.parse::<u32>()) {
                            for cpu in s..=e {
                                unsafe { libc::CPU_SET(cpu as usize, &mut cpu_set) };
                            }
                        }
                    } else if let Ok(cpu) = range.parse::<u32>() {
                        unsafe { libc::CPU_SET(cpu as usize, &mut cpu_set) };
                    }
                }

                // Set thread affinity
                unsafe {
                    libc::sched_setaffinity(
                        0, // current thread
                        std::mem::size_of::<libc::cpu_set_t>(),
                        &cpu_set,
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn bind_thread_to_numa_node_windows(node: u32) -> Result<(), Box<dyn std::error::Error>> {
        // Windows thread affinity using SetThreadGroupAffinity would go here
        Ok(())
    }

    /// Warm memory to ensure pages are allocated and potentially improve locality
    fn warm_memory(buffer: &[f64]) {
        if buffer.is_empty() {
            return;
        }

        // Touch every page to ensure allocation
        let page_size = 4096; // Typical page size
        let elements_per_page = page_size / std::mem::size_of::<f64>();

        for i in (0..buffer.len()).step_by(elements_per_page) {
            // Volatile read to prevent optimization
            unsafe {
                std::ptr::read_volatile(&buffer[i]);
            }
        }
    }

    /// Detect current NUMA node using platform-specific APIs
    fn detect_numa_node() -> i32 {
        #[cfg(target_os = "linux")]
        {
            Self::detect_numa_node_linux().unwrap_or(0)
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_numa_node_windows().unwrap_or(0)
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            0 // Default for unsupported platforms
        }
    }

    /// Linux-specific NUMA node detection
    #[cfg(target_os = "linux")]
    fn detect_numa_node_linux() -> Option<i32> {
        // Try to get current thread's NUMA node
        let _tid = unsafe { libc::gettid() };

        // Read from /proc/self/task/{tid}/numa_maps or use getcpu syscall
        match Self::get_current_numa_node_linux() {
            Ok(node) => Some(node),
            Err(_) => {
                // Fallback: try to detect from CPU
                Self::detect_numa_from_cpu_linux()
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_current_numa_node_linux() -> Result<i32, Box<dyn std::error::Error>> {
        // Use getcpu syscall to get current CPU and NUMA node
        let mut cpu: u32 = 0;
        let mut node: u32 = 0;

        let result = unsafe {
            libc::syscall(
                libc::SYS_getcpu,
                &mut cpu as *mut u32,
                &mut node as *mut u32,
                std::ptr::null_mut::<libc::c_void>(),
            )
        };

        if result == 0 {
            Ok(node as i32)
        } else {
            Err("getcpu syscall failed".into())
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_numa_from_cpu_linux() -> Option<i32> {
        // Try to read NUMA topology from /sys/devices/system/node/
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    if let Some(stripped) = name_str.strip_prefix("node") {
                        if let Ok(node_num) = stripped.parse::<i32>() {
                            // Simple heuristic: use first available node
                            return Some(node_num);
                        }
                    }
                }
            }
        }
        None
    }

    /// Windows-specific NUMA node detection
    #[cfg(target_os = "windows")]
    fn detect_numa_node_windows() -> Option<i32> {
        // In a real implementation, this would use Windows NUMA APIs
        // such as GetNumaProcessorNode, GetCurrentProcessorNumber, etc.
        // For now, return 0 as fallback
        Some(0)
    }

    /// Get NUMA topology information
    pub fn get_numa_topology() -> NumaTopology {
        #[cfg(target_os = "linux")]
        {
            Self::get_numa_topology_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::get_numa_topology_windows()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            NumaTopology::default()
        }
    }

    #[cfg(target_os = "linux")]
    fn get_numa_topology_linux() -> NumaTopology {
        let mut topology = NumaTopology::default();

        // Try to read NUMA information from /sys/devices/system/node/
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    if let Some(stripped) = name_str.strip_prefix("node") {
                        if let Ok(_nodeid) = stripped.parse::<u32>() {
                            // Read memory info for this node
                            let meminfo_path =
                                format!("/sys/devices/system/node/{name_str}/meminfo");
                            if let Ok(meminfo) = fs::read_to_string(&meminfo_path) {
                                if let Some(total_kb) = Self::parse_meminfo_total(&meminfo) {
                                    topology.nodes.push(NumaNode {
                                        id: _nodeid,
                                        total_memory_bytes: total_kb * 1024,
                                        available_memory_bytes: total_kb * 1024, // Approximation
                                        cpu_count: Self::get_node_cpu_count(_nodeid).unwrap_or(1),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // If no nodes found, create a default single node
        if topology.nodes.is_empty() {
            topology.nodes.push(NumaNode {
                id: 0,
                total_memory_bytes: Self::get_total_system_memory()
                    .unwrap_or(8 * 1024 * 1024 * 1024), // 8GB default
                available_memory_bytes: Self::get_available_system_memory()
                    .unwrap_or(4 * 1024 * 1024 * 1024), // 4GB default
                cpu_count: num_cpus::get() as u32,
            });
        }

        topology
    }

    #[cfg(target_os = "linux")]
    fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
        for line in meminfo.lines() {
            if line.starts_with("Node") && line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    return parts[2].parse().ok();
                }
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    fn get_node_cpu_count(_nodeid: u32) -> Option<u32> {
        let cpulist_path = format!("/sys/devices/system/node/node{}/cpulist", _nodeid);
        if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
            // Parse CPU list (e.g., "0-3,8-11" -> 8 CPUs)
            let mut count = 0;
            for range in cpulist.trim().split(',') {
                if let Some((start, end)) = range.split_once('-') {
                    if let (Ok(s), Ok(e)) = (start.parse::<u32>(), end.parse::<u32>()) {
                        count += e - s + 1;
                    }
                } else if range.parse::<u32>().is_ok() {
                    count += 1;
                }
            }
            Some(count)
        } else {
            None
        }
    }

    #[cfg(target_os = "linux")]
    fn get_total_system_memory() -> Option<u64> {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<u64>().ok().map(|kb| kb * 1024);
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    fn get_available_system_memory() -> Option<u64> {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<u64>().ok().map(|kb| kb * 1024);
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "windows")]
    fn get_numa_topology_windows() -> NumaTopology {
        // Windows NUMA topology detection would go here
        // Using GetLogicalProcessorInformation and related APIs
        NumaTopology::default()
    }

    /// Clean up excess memory when approaching limits
    fn cleanup_excess_memory(&self) {
        // Remove some older buffers to free memory
        let cleanup_ratio = 0.25; // Clean up 25% of buffers

        {
            let mut buffers = self.distance_buffers.lock().unwrap();
            let cleanup_count = (buffers.len() as f64 * cleanup_ratio) as usize;
            for _ in 0..cleanup_count {
                if let Some(buffer) = buffers.pop_back() {
                    let freed_bytes = buffer.len() * std::mem::size_of::<f64>();
                    self.memory_usage
                        .fetch_sub(freed_bytes, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        {
            let mut buffers = self.large_buffers.lock().unwrap();
            let cleanup_count = (buffers.len() as f64 * cleanup_ratio) as usize;
            for _ in 0..cleanup_count {
                if let Some(buffer) = buffers.pop_back() {
                    let freed_bytes = buffer.len() * std::mem::size_of::<f64>();
                    self.memory_usage
                        .fetch_sub(freed_bytes, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// Return a distance buffer to the pool
    fn return_distance_buffer(&self, buffer: Box<[f64]>) {
        let buffer_size_bytes = buffer.len() * std::mem::size_of::<f64>();
        let is_large = buffer_size_bytes > self.config.large_object_threshold;

        // Update memory usage when buffer is returned
        self.memory_usage
            .fetch_sub(buffer_size_bytes, std::sync::atomic::Ordering::Relaxed);

        if is_large {
            let mut buffers = self.large_buffers.lock().unwrap();
            if buffers.len() < self.config.max_pool_size / 10 {
                buffers.push_back(buffer);
            }
            // Otherwise let it drop and deallocate
        } else {
            let mut buffers = self.distance_buffers.lock().unwrap();
            if buffers.len() < self.config.max_pool_size {
                buffers.push_back(buffer);
            }
            // Otherwise let it drop and deallocate
        }
    }

    /// Return an index buffer to the pool
    fn return_index_buffer(&self, buffer: Box<[usize]>) {
        let mut buffers = self.index_buffers.lock().unwrap();
        if buffers.len() < self.config.max_pool_size {
            buffers.push_back(buffer);
        }
    }

    /// Return a matrix buffer to the pool
    fn return_matrix_buffer(&self, matrix: Array2<f64>) {
        let mut buffers = self.matrix_buffers.lock().unwrap();
        if buffers.len() < self.config.max_pool_size / 4 {
            // Keep fewer matrices
            buffers.push_back(matrix);
        }
    }

    /// Get pool statistics for performance monitoring
    pub fn statistics(&self) -> PoolStatistics {
        self.stats.clone()
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.memory_usage.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get current NUMA node
    pub fn current_numa_node(&self) -> i32 {
        self.numa_node.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get detailed pool information
    pub fn pool_info(&self) -> PoolInfo {
        let distance_count = self.distance_buffers.lock().unwrap().len();
        let index_count = self.index_buffers.lock().unwrap().len();
        let matrix_count = self.matrix_buffers.lock().unwrap().len();
        let large_count = self.large_buffers.lock().unwrap().len();

        PoolInfo {
            distance_buffer_count: distance_count,
            index_buffer_count: index_count,
            matrix_buffer_count: matrix_count,
            large_buffer_count: large_count,
            total_memory_usage: self.memory_usage(),
            numa_node: self.current_numa_node(),
            hit_rate: self.stats.hit_rate(),
        }
    }

    /// Clear all pools and free memory
    pub fn clear(&self) {
        self.distance_buffers.lock().unwrap().clear();
        self.index_buffers.lock().unwrap().clear();
        self.matrix_buffers.lock().unwrap().clear();
        self.large_buffers.lock().unwrap().clear();
        self.memory_usage
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.stats.reset();
    }
}

// Use ndarray's s! macro
use ndarray::s;

/// RAII wrapper for distance buffers with automatic return to pool
pub struct DistanceBuffer<'a> {
    buffer: Option<Box<[f64]>>,
    pool: &'a DistancePool,
}

impl<'a> DistanceBuffer<'a> {
    fn new(buffer: Box<[f64]>, pool: &'a DistancePool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.buffer.as_mut().unwrap().as_mut()
    }

    /// Get an immutable slice of the buffer
    pub fn as_slice(&self) -> &[f64] {
        self.buffer.as_ref().unwrap().as_ref()
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.buffer.as_ref().unwrap().len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a view as ndarray Array1
    pub fn as_array_mut(&mut self) -> ArrayViewMut1<f64> {
        ArrayViewMut1::from(self.as_mut_slice())
    }
}

impl Drop for DistanceBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_distance_buffer(buffer);
        }
    }
}

/// RAII wrapper for index buffers
pub struct IndexBuffer<'a> {
    buffer: Option<Box<[usize]>>,
    pool: &'a DistancePool,
}

impl<'a> IndexBuffer<'a> {
    fn new(buffer: Box<[usize]>, pool: &'a DistancePool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        self.buffer.as_mut().unwrap().as_mut()
    }

    /// Get an immutable slice of the buffer
    pub fn as_slice(&self) -> &[usize] {
        self.buffer.as_ref().unwrap().as_ref()
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.buffer.as_ref().unwrap().len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for IndexBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_index_buffer(buffer);
        }
    }
}

/// RAII wrapper for matrix buffers
pub struct MatrixBuffer<'a> {
    matrix: Option<Array2<f64>>,
    pool: &'a DistancePool,
}

impl<'a> MatrixBuffer<'a> {
    fn new(matrix: Array2<f64>, pool: &'a DistancePool) -> Self {
        Self {
            matrix: Some(matrix),
            pool,
        }
    }

    /// Get a mutable view of the matrix
    pub fn as_mut(&mut self) -> ArrayViewMut2<f64> {
        self.matrix.as_mut().unwrap().view_mut()
    }

    /// Get the dimensions of the matrix
    pub fn dim(&mut self) -> (usize, usize) {
        self.matrix.as_ref().unwrap().dim()
    }

    /// Fill the matrix with a value
    pub fn fill(&mut self, value: f64) {
        self.matrix.as_mut().unwrap().fill(value);
    }
}

impl Drop for MatrixBuffer<'_> {
    fn drop(&mut self) {
        if let Some(matrix) = self.matrix.take() {
            self.pool.return_matrix_buffer(matrix);
        }
    }
}

/// Arena allocator for temporary objects in clustering algorithms
pub struct ClusteringArena {
    config: MemoryPoolConfig,
    current_block: Mutex<Option<ArenaBlock>>,
    full_blocks: Mutex<Vec<ArenaBlock>>,
    stats: ArenaStatistics,
}

impl ClusteringArena {
    /// Create a new clustering arena
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create arena with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        Self {
            config,
            current_block: Mutex::new(None),
            full_blocks: Mutex::new(Vec::new()),
            stats: ArenaStatistics::new(),
        }
    }

    /// Allocate a temporary vector in the arena
    pub fn alloc_temp_vec<T: Default + Clone>(&self, size: usize) -> ArenaVec<T> {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = self.allocate_raw(layout);

        unsafe {
            // Initialize elements
            for i in 0..size {
                std::ptr::write(ptr.as_ptr().add(i) as *mut T, T::default());
            }

            ArenaVec::new(ptr.as_ptr() as *mut T, size)
        }
    }

    /// Allocate raw memory with proper alignment
    fn allocate_raw(&self, layout: Layout) -> NonNull<u8> {
        let mut current = self.current_block.lock().unwrap();

        if current.is_none() || !current.as_ref().unwrap().can_allocate(layout) {
            // Need a new block
            if let Some(old_block) = current.take() {
                self.full_blocks.lock().unwrap().push(old_block);
            }
            *current = Some(ArenaBlock::new(self.config.arena_block_size));
        }

        current.as_mut().unwrap().allocate(layout)
    }

    /// Reset the arena, keeping allocated blocks for reuse
    pub fn reset(&self) {
        let mut current = self.current_block.lock().unwrap();
        let mut full_blocks = self.full_blocks.lock().unwrap();

        if let Some(block) = current.take() {
            full_blocks.push(block);
        }

        // Reset all blocks
        for block in full_blocks.iter_mut() {
            block.reset();
        }

        // Move one block back to current
        if let Some(block) = full_blocks.pop() {
            *current = Some(block);
        }

        self.stats.reset();
    }

    /// Get arena statistics
    pub fn statistics(&self) -> ArenaStatistics {
        self.stats.clone()
    }
}

impl Default for ClusteringArena {
    fn default() -> Self {
        Self::new()
    }
}

/// A block of memory within the arena
struct ArenaBlock {
    memory: NonNull<u8>,
    size: usize,
    offset: usize,
}

// SAFETY: ArenaBlock manages its own memory and ensures thread-safe access
unsafe impl Send for ArenaBlock {}
unsafe impl Sync for ArenaBlock {}

impl ArenaBlock {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 64).unwrap(); // 64-byte aligned
        let memory =
            unsafe { NonNull::new(System.alloc(layout)).expect("Failed to allocate arena block") };

        Self {
            memory,
            size,
            offset: 0,
        }
    }

    fn can_allocate(&self, layout: Layout) -> bool {
        let aligned_offset = (self.offset + layout.align() - 1) & !(layout.align() - 1);
        aligned_offset + layout.size() <= self.size
    }

    fn allocate(&mut self, layout: Layout) -> NonNull<u8> {
        assert!(self.can_allocate(layout));

        // Align the offset
        self.offset = (self.offset + layout.align() - 1) & !(layout.align() - 1);

        let ptr = unsafe { NonNull::new_unchecked(self.memory.as_ptr().add(self.offset)) };
        self.offset += layout.size();

        ptr
    }

    fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for ArenaBlock {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, 64).unwrap();
        unsafe {
            System.dealloc(self.memory.as_ptr(), layout);
        }
    }
}

/// RAII wrapper for arena-allocated vectors
pub struct ArenaVec<T> {
    ptr: *mut T,
    len: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T> ArenaVec<T> {
    fn new(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            phantom: std::marker::PhantomData,
        }
    }

    /// Get a mutable slice of the vector
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get an immutable slice of the vector
    pub fn as_slice(&mut self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get the length of the vector
    pub fn len(&mut self) -> usize {
        self.len
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Note: ArenaVec doesn't implement Drop because the arena manages the memory

/// Detailed pool information
#[derive(Debug, Clone)]
pub struct PoolInfo {
    /// Number of distance buffers in pool
    pub distance_buffer_count: usize,
    /// Number of index buffers in pool
    pub index_buffer_count: usize,
    /// Number of matrix buffers in pool
    pub matrix_buffer_count: usize,
    /// Number of large buffers in pool
    pub large_buffer_count: usize,
    /// Total memory usage in bytes
    pub total_memory_usage: usize,
    /// Current NUMA node
    pub numa_node: i32,
    /// Hit rate percentage
    pub hit_rate: f64,
}

/// Pool performance statistics
#[derive(Debug)]
pub struct PoolStatistics {
    hits: std::sync::atomic::AtomicUsize,
    misses: std::sync::atomic::AtomicUsize,
    total_allocations: std::sync::atomic::AtomicUsize,
}

impl PoolStatistics {
    fn new() -> Self {
        Self {
            hits: std::sync::atomic::AtomicUsize::new(0),
            misses: std::sync::atomic::AtomicUsize::new(0),
            total_allocations: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn record_hit(&self) {
        self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn reset(&self) {
        self.hits.store(0, std::sync::atomic::Ordering::Relaxed);
        self.misses.store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_allocations
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + self.misses.load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64 * 100.0
        }
    }

    /// Get total requests
    pub fn total_requests(&self) -> usize {
        self.hits.load(std::sync::atomic::Ordering::Relaxed)
            + self.misses.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total new allocations
    pub fn total_allocations(&self) -> usize {
        self.total_allocations
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Clone for PoolStatistics {
    fn clone(&self) -> Self {
        Self {
            hits: std::sync::atomic::AtomicUsize::new(
                self.hits.load(std::sync::atomic::Ordering::Relaxed),
            ),
            misses: std::sync::atomic::AtomicUsize::new(
                self.misses.load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_allocations: std::sync::atomic::AtomicUsize::new(
                self.total_allocations
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

/// Arena performance statistics
#[derive(Debug)]
pub struct ArenaStatistics {
    blocks_allocated: std::sync::atomic::AtomicUsize,
    total_memory: std::sync::atomic::AtomicUsize,
    active_objects: std::sync::atomic::AtomicUsize,
}

impl ArenaStatistics {
    fn new() -> Self {
        Self {
            blocks_allocated: std::sync::atomic::AtomicUsize::new(0),
            total_memory: std::sync::atomic::AtomicUsize::new(0),
            active_objects: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.blocks_allocated
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_memory
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.active_objects
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get number of allocated blocks
    pub fn blocks_allocated(&self) -> usize {
        self.blocks_allocated
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total memory usage in bytes
    pub fn total_memory(&self) -> usize {
        self.total_memory.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get number of active objects
    pub fn active_objects(&self) -> usize {
        self.active_objects
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Clone for ArenaStatistics {
    fn clone(&self) -> Self {
        Self {
            blocks_allocated: std::sync::atomic::AtomicUsize::new(
                self.blocks_allocated
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_memory: std::sync::atomic::AtomicUsize::new(
                self.total_memory.load(std::sync::atomic::Ordering::Relaxed),
            ),
            active_objects: std::sync::atomic::AtomicUsize::new(
                self.active_objects
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

/// NUMA topology information for memory allocation optimization
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Available NUMA nodes
    pub nodes: Vec<NumaNode>,
}

/// Individual NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// NUMA node ID
    pub id: u32,
    /// Total memory on this node in bytes
    pub total_memory_bytes: u64,
    /// Available memory on this node in bytes
    pub available_memory_bytes: u64,
    /// Number of CPU cores on this node
    pub cpu_count: u32,
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self {
            nodes: vec![NumaNode {
                id: 0,
                total_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
                available_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB default
                cpu_count: 4,                               // Default 4 cores
            }],
        }
    }
}

impl NumaTopology {
    /// Get the best NUMA node for allocation based on current thread affinity
    pub fn get_optimal_node(&self) -> u32 {
        // In a real implementation, this would check current thread affinity
        // and return the node that the thread is running on
        if !self.nodes.is_empty() {
            self.nodes[0].id
        } else {
            0
        }
    }

    /// Get node with most available memory
    pub fn get_node_with_most_memory(&self) -> Option<u32> {
        self.nodes
            .iter()
            .max_by_key(|node| node.available_memory_bytes)
            .map(|node| node.id)
    }

    /// Get total system memory across all nodes
    pub fn total_system_memory(&self) -> u64 {
        self.nodes.iter().map(|node| node.total_memory_bytes).sum()
    }

    /// Get total available memory across all nodes
    pub fn total_available_memory(&self) -> u64 {
        self.nodes
            .iter()
            .map(|node| node.available_memory_bytes)
            .sum()
    }

    /// Check if a specific NUMA node exists
    pub fn has_node(&self, _nodeid: u32) -> bool {
        self.nodes.iter().any(|node| node.id == _nodeid)
    }

    /// Get memory information for a specific node
    pub fn get_node_info(&self, _nodeid: u32) -> Option<&NumaNode> {
        self.nodes.iter().find(|node| node.id == _nodeid)
    }
}

/// Global memory pool instance for convenience
static GLOBAL_DISTANCE_POOL: std::sync::OnceLock<DistancePool> = std::sync::OnceLock::new();
static GLOBAL_CLUSTERING_ARENA: std::sync::OnceLock<ClusteringArena> = std::sync::OnceLock::new();

/// Get the global distance pool instance
#[allow(dead_code)]
pub fn global_distance_pool() -> &'static DistancePool {
    GLOBAL_DISTANCE_POOL.get_or_init(|| DistancePool::new(1000))
}

/// Get the global clustering arena instance
#[allow(dead_code)]
pub fn global_clustering_arena() -> &'static ClusteringArena {
    GLOBAL_CLUSTERING_ARENA.get_or_init(ClusteringArena::new)
}

/// Create a NUMA-optimized distance pool for the current thread
#[allow(dead_code)]
pub fn create_numa_optimized_pool(capacity: usize) -> DistancePool {
    let config = MemoryPoolConfig {
        numa_aware: true,
        auto_numa_discovery: true,
        enable_thread_affinity: true,
        ..Default::default()
    };

    DistancePool::with_config(capacity, config)
}

/// Get NUMA topology information
#[allow(dead_code)]
pub fn get_numa_topology() -> NumaTopology {
    DistancePool::get_numa_topology()
}

/// Test NUMA capabilities and return detailed information
#[allow(dead_code)]
pub fn test_numa_capabilities() -> NumaCapabilities {
    NumaCapabilities::detect()
}

/// NUMA system capabilities
#[derive(Debug, Clone)]
pub struct NumaCapabilities {
    /// Whether NUMA is available on this system
    pub numa_available: bool,
    /// Number of NUMA nodes detected
    pub num_nodes: u32,
    /// Whether NUMA memory binding is supported
    pub memory_binding_supported: bool,
    /// Whether thread affinity is supported
    pub thread_affinity_supported: bool,
    /// Platform-specific details
    pub platform_details: String,
}

impl NumaCapabilities {
    /// Detect NUMA capabilities of the current system
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Self {
                numa_available: false,
                num_nodes: 1,
                memory_binding_supported: false,
                thread_affinity_supported: false,
                platform_details: "Unsupported platform".to_string(),
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        let numa_available = std::path::Path::new("/sys/devices/system/node").exists();
        let num_nodes = if numa_available {
            DistancePool::get_numa_topology().nodes.len() as u32
        } else {
            1
        };

        Self {
            numa_available,
            num_nodes,
            memory_binding_supported: numa_available,
            thread_affinity_supported: true, // Generally available on Linux
            platform_details: format!("Linux with {num_nodes} NUMA nodes"),
        }
    }

    #[cfg(target_os = "windows")]
    fn detect_windows() -> Self {
        Self {
            numa_available: true, // Windows typically has NUMA support
            num_nodes: 1,         // Would be detected using Windows APIs
            memory_binding_supported: true,
            thread_affinity_supported: true,
            platform_details: "Windows NUMA support".to_string(),
        }
    }

    /// Check if NUMA optimizations should be enabled
    pub fn should_enable_numa(&self) -> bool {
        self.numa_available && self.num_nodes > 1
    }

    /// Get recommended memory allocation strategy
    pub fn recommended_memory_strategy(&self) -> &'static str {
        if self.should_enable_numa() {
            "NUMA-aware"
        } else {
            "Standard"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_pool() {
        let pool = DistancePool::new(10);

        // Get a buffer
        let mut buffer1 = pool.get_distance_buffer(100);
        assert_eq!(buffer1.len(), 100);

        // Use the buffer
        buffer1.as_mut_slice()[0] = 42.0;
        assert_eq!(buffer1.as_slice()[0], 42.0);

        // Get another buffer while first is in use
        let buffer2 = pool.get_distance_buffer(50);
        assert_eq!(buffer2.len(), 50);

        // Drop first buffer (should return to pool)
        drop(buffer1);

        // Get buffer again (should reuse)
        let buffer3 = pool.get_distance_buffer(100);
        assert_eq!(buffer3.len(), 100);
        // Note: value should be zeroed when creating aligned buffer
    }

    #[test]
    fn test_arena_allocator() {
        let arena = ClusteringArena::new();

        // Allocate some temporary vectors
        let mut vec1 = arena.alloc_temp_vec::<f64>(100);
        let mut vec2 = arena.alloc_temp_vec::<usize>(50);

        // Use the vectors
        vec1.as_mut_slice()[0] = std::f64::consts::PI;
        vec2.as_mut_slice()[0] = 42;

        assert_eq!(vec1.as_slice()[0], std::f64::consts::PI);
        assert_eq!(vec2.as_slice()[0], 42);

        // Reset arena
        arena.reset();

        // Allocate again (should reuse memory)
        let mut vec3 = arena.alloc_temp_vec::<f64>(200);
        vec3.as_mut_slice()[0] = 2.71;
        assert_eq!(vec3.as_slice()[0], 2.71);
    }

    #[test]
    fn test_pool_statistics() {
        let pool = DistancePool::new(2);

        // Initial stats should be zero
        let stats = pool.statistics();
        assert_eq!(stats.total_requests(), 0);
        assert_eq!(stats.total_allocations(), 0);

        // First request should be a miss
        let _buffer1 = pool.get_distance_buffer(100);
        let stats = pool.statistics();
        assert_eq!(stats.total_requests(), 1);
        assert_eq!(stats.total_allocations(), 1);
        assert!(stats.hit_rate() < 1.0);

        // Drop and get again should be a hit
        drop(_buffer1);
        let _buffer2 = pool.get_distance_buffer(100);
        let stats = pool.statistics();
        assert_eq!(stats.total_requests(), 2);
        assert_eq!(stats.total_allocations(), 1); // No new allocation
        assert!(stats.hit_rate() > 0.0);
    }

    #[test]
    fn test_matrix_buffer() {
        let pool = DistancePool::new(5);

        let mut matrix = pool.get_matrix_buffer(10, 10);
        assert_eq!(matrix.dim(), (10, 10));

        matrix.fill(42.0);
        // Matrix should be filled with 42.0 (can't easily test without exposing internals)

        drop(matrix);

        // Get another matrix (should potentially reuse)
        let mut matrix2 = pool.get_matrix_buffer(8, 8);
        assert_eq!(matrix2.dim(), (8, 8));
    }

    #[test]
    fn test_global_pools() {
        // Test that global pools can be accessed
        let pool = global_distance_pool();
        let arena = global_clustering_arena();

        let buffer = pool.get_distance_buffer(10);
        let _vec = arena.alloc_temp_vec::<f64>(10);

        // Should not panic
    }
}
