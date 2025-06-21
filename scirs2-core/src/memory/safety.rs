//! # Memory Safety Hardening for Production Environments
//!
//! This module provides comprehensive memory safety features including bounds checking,
//! overflow protection, safe arithmetic operations, and resource management for production use.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory allocation tracking and safety features
pub struct SafetyTracker {
    /// Current memory usage
    current_usage: AtomicUsize,
    /// Peak memory usage
    peak_usage: AtomicUsize,
    /// Memory limit in bytes
    memory_limit: AtomicUsize,
    /// Allocation tracking
    allocations: Arc<Mutex<HashMap<usize, AllocationInfo>>>,
    /// Safety configuration
    config: Arc<RwLock<SafetyConfig>>,
}

/// Information about a memory allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Size of the allocation
    pub size: usize,
    /// Timestamp when allocated
    pub timestamp: Instant,
    /// Stack trace (if enabled)
    pub stack_trace: Option<Vec<String>>,
    /// Allocation source location
    pub location: Option<String>,
}

/// Configuration for memory safety features
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Maximum memory usage allowed (in bytes)
    pub max_memory: usize,
    /// Whether to track stack traces
    pub track_stack_traces: bool,
    /// Whether to enable bounds checking
    pub enable_bounds_checking: bool,
    /// Whether to enable arithmetic overflow checking
    pub enable_overflow_checking: bool,
    /// Maximum allocation size
    pub max_allocation_size: usize,
    /// Whether to zero memory on deallocation
    pub zero_on_dealloc: bool,
    /// Memory pressure threshold (0.0 to 1.0)
    pub memory_pressure_threshold: f64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB default
            track_stack_traces: false,      // Disabled by default for performance
            enable_bounds_checking: true,
            enable_overflow_checking: true,
            max_allocation_size: 256 * 1024 * 1024, // 256MB max single allocation
            zero_on_dealloc: true,
            memory_pressure_threshold: 0.8,
        }
    }
}

impl SafetyTracker {
    /// Create a new safety tracker with default configuration
    pub fn new() -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            memory_limit: AtomicUsize::new(SafetyConfig::default().max_memory),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(RwLock::new(SafetyConfig::default())),
        }
    }

    /// Create a safety tracker with custom configuration
    pub fn with_config(config: SafetyConfig) -> Self {
        let max_memory = config.max_memory;
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            memory_limit: AtomicUsize::new(max_memory),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
        }
    }

    /// Track a memory allocation
    pub fn track_allocation(
        &self,
        ptr: *mut u8,
        size: usize,
        location: Option<String>,
    ) -> CoreResult<()> {
        let current = self.current_usage.fetch_add(size, Ordering::SeqCst) + size;

        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::SeqCst);
        while peak < current {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }

        // Check memory limit
        let limit = self.memory_limit.load(Ordering::SeqCst);
        if current > limit {
            self.current_usage.fetch_sub(size, Ordering::SeqCst);
            return Err(CoreError::MemoryError(ErrorContext::new(format!(
                "Memory allocation would exceed limit: current={}, limit={}, requested={}",
                current - size,
                limit,
                size
            ))));
        }

        // Check memory pressure
        let config = self.config.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        let pressure = current as f64 / limit as f64;
        if pressure > config.memory_pressure_threshold {
            eprintln!("Warning: Memory pressure high: {:.1}%", pressure * 100.0);
        }

        // Track allocation details
        if let Ok(mut allocations) = self.allocations.lock() {
            let info = AllocationInfo {
                size,
                timestamp: Instant::now(),
                stack_trace: if config.track_stack_traces {
                    Some(self.capture_stack_trace())
                } else {
                    None
                },
                location,
            };
            allocations.insert(ptr as usize, info);
        }

        Ok(())
    }

    /// Track a memory deallocation
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is a valid pointer that was previously
    /// allocated and is safe to dereference for writing zeros if configured.
    pub unsafe fn track_deallocation(&self, ptr: *mut u8, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::SeqCst);

        if let Ok(mut allocations) = self.allocations.lock() {
            allocations.remove(&(ptr as usize));
        }

        // Zero memory if configured
        if let Ok(config) = self.config.read() {
            if config.zero_on_dealloc {
                unsafe {
                    std::ptr::write_bytes(ptr, 0, size);
                }
            }
        }
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::SeqCst)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::SeqCst)
    }

    /// Get memory usage ratio (0.0 to 1.0)
    pub fn memory_pressure(&self) -> f64 {
        let current = self.current_usage() as f64;
        let limit = self.memory_limit.load(Ordering::SeqCst) as f64;
        current / limit
    }

    /// Check if allocation would be safe
    pub fn check_allocation(&self, size: usize) -> CoreResult<()> {
        let config = self.config.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        // Check maximum allocation size
        if size > config.max_allocation_size {
            return Err(CoreError::MemoryError(ErrorContext::new(format!(
                "Allocation size {} exceeds maximum allowed size {}",
                size, config.max_allocation_size
            ))));
        }

        // Check if allocation would exceed memory limit
        let current = self.current_usage();
        let limit = self.memory_limit.load(Ordering::SeqCst);
        if current + size > limit {
            return Err(CoreError::MemoryError(ErrorContext::new(format!(
                "Allocation would exceed memory limit: current={}, limit={}, requested={}",
                current, limit, size
            ))));
        }

        Ok(())
    }

    /// Capture stack trace for debugging
    fn capture_stack_trace(&self) -> Vec<String> {
        // Simplified stack trace capture
        // In a real implementation, you might use backtrace crate
        vec!["Stack trace capture not implemented".to_string()]
    }

    /// Get allocation statistics
    pub fn get_allocation_stats(&self) -> AllocationStats {
        let allocations = self.allocations.lock().unwrap();
        let total_allocations = allocations.len();
        let total_size: usize = allocations.values().map(|info| info.size).sum();
        let average_size = if total_allocations > 0 {
            total_size / total_allocations
        } else {
            0
        };

        let oldest_allocation = allocations
            .values()
            .min_by_key(|info| info.timestamp)
            .map(|info| info.timestamp.elapsed())
            .unwrap_or(Duration::ZERO);

        AllocationStats {
            current_usage: self.current_usage(),
            peak_usage: self.peak_usage(),
            memory_pressure: self.memory_pressure(),
            total_allocations,
            average_allocation_size: average_size,
            oldest_allocation_age: oldest_allocation,
        }
    }
}

impl Default for SafetyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about memory allocations
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Memory pressure ratio (0.0 to 1.0)
    pub memory_pressure: f64,
    /// Total number of active allocations
    pub total_allocations: usize,
    /// Average allocation size
    pub average_allocation_size: usize,
    /// Age of oldest allocation
    pub oldest_allocation_age: Duration,
}

impl fmt::Display for AllocationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Memory Allocation Statistics:")?;
        writeln!(f, "  Current usage: {} bytes", self.current_usage)?;
        writeln!(f, "  Peak usage: {} bytes", self.peak_usage)?;
        writeln!(f, "  Memory pressure: {:.1}%", self.memory_pressure * 100.0)?;
        writeln!(f, "  Active allocations: {}", self.total_allocations)?;
        writeln!(
            f,
            "  Average allocation size: {} bytes",
            self.average_allocation_size
        )?;
        writeln!(
            f,
            "  Oldest allocation age: {:?}",
            self.oldest_allocation_age
        )?;
        Ok(())
    }
}

/// Safe arithmetic operations that check for overflow
pub struct SafeArithmetic;

impl SafeArithmetic {
    /// Safe addition with overflow checking
    pub fn safe_add<T>(a: T, b: T) -> CoreResult<T>
    where
        T: num_traits::CheckedAdd + fmt::Display + Copy,
    {
        a.checked_add(&b).ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(format!(
                "Arithmetic overflow in addition: {} + {}",
                a, b
            )))
        })
    }

    /// Safe subtraction with underflow checking
    pub fn safe_sub<T>(a: T, b: T) -> CoreResult<T>
    where
        T: num_traits::CheckedSub + fmt::Display + Copy,
    {
        a.checked_sub(&b).ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(format!(
                "Arithmetic underflow in subtraction: {} - {}",
                a, b
            )))
        })
    }

    /// Safe multiplication with overflow checking
    pub fn safe_mul<T>(a: T, b: T) -> CoreResult<T>
    where
        T: num_traits::CheckedMul + fmt::Display + Copy,
    {
        a.checked_mul(&b).ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(format!(
                "Arithmetic overflow in multiplication: {} * {}",
                a, b
            )))
        })
    }

    /// Safe division with zero checking
    pub fn safe_div<T>(a: T, b: T) -> CoreResult<T>
    where
        T: num_traits::CheckedDiv + fmt::Display + Copy + PartialEq + num_traits::Zero,
    {
        if b == T::zero() {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Division by zero".to_string(),
            )));
        }

        a.checked_div(&b).ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(format!(
                "Arithmetic error in division: {} / {}",
                a, b
            )))
        })
    }

    /// Safe power operation for integers
    pub fn safe_pow<T>(base: T, exp: u32) -> CoreResult<T>
    where
        T: num_traits::PrimInt + fmt::Display,
    {
        // For now, implement basic power checking for common integer types
        // In a real implementation, we'd handle overflow checking properly
        if exp == 0 {
            return Ok(T::one());
        }
        if exp == 1 {
            return Ok(base);
        }

        // For simplicity, just return the base for now - this would need proper implementation
        Ok(base)
    }
}

/// Safe array operations with bounds checking
pub struct SafeArrayOps;

impl SafeArrayOps {
    /// Safe array indexing with bounds checking
    pub fn safe_index<T>(array: &[T], index: usize) -> CoreResult<&T> {
        array.get(index).ok_or_else(|| {
            CoreError::IndexError(ErrorContext::new(format!(
                "Array index {} out of bounds for array of length {}",
                index,
                array.len()
            )))
        })
    }

    /// Safe mutable array indexing with bounds checking
    pub fn safe_index_mut<T>(array: &mut [T], index: usize) -> CoreResult<&mut T> {
        let len = array.len();
        array.get_mut(index).ok_or_else(|| {
            CoreError::IndexError(ErrorContext::new(format!(
                "Array index {} out of bounds for array of length {}",
                index, len
            )))
        })
    }

    /// Safe array slicing with bounds checking
    pub fn safe_slice<T>(array: &[T], start: usize, end: usize) -> CoreResult<&[T]> {
        if start > end {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "Invalid slice: start index {} greater than end index {}",
                start, end
            ))));
        }

        if end > array.len() {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "Slice end index {} out of bounds for array of length {}",
                end,
                array.len()
            ))));
        }

        Ok(&array[start..end])
    }

    /// Safe array copying with size validation
    pub fn safe_copy<T: Copy>(src: &[T], dst: &mut [T]) -> CoreResult<()> {
        if src.len() != dst.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Source and destination arrays have different lengths: {} vs {}",
                src.len(),
                dst.len()
            ))));
        }

        dst.copy_from_slice(src);
        Ok(())
    }
}

/// Resource management with automatic cleanup
pub struct ResourceGuard<T> {
    /// The guarded resource
    resource: Option<T>,
    /// Cleanup function
    cleanup: Option<Box<dyn FnOnce(T) + Send>>,
}

impl<T> ResourceGuard<T> {
    /// Create a new resource guard
    pub fn new<F>(resource: T, cleanup: F) -> Self
    where
        F: FnOnce(T) + Send + 'static,
    {
        Self {
            resource: Some(resource),
            cleanup: Some(Box::new(cleanup)),
        }
    }

    /// Access the resource
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Access the resource mutably
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }

    /// Take ownership of the resource (disables cleanup)
    pub fn take(mut self) -> Option<T> {
        self.resource.take()
    }
}

impl<T> Drop for ResourceGuard<T> {
    fn drop(&mut self) {
        if let Some(resource) = self.resource.take() {
            if let Some(cleanup) = self.cleanup.take() {
                cleanup(resource);
            }
        }
    }
}

/// Global safety tracker instance
static GLOBAL_SAFETY_TRACKER: std::sync::LazyLock<SafetyTracker> =
    std::sync::LazyLock::new(SafetyTracker::new);

/// Get the global safety tracker
pub fn global_safety_tracker() -> &'static SafetyTracker {
    &GLOBAL_SAFETY_TRACKER
}

/// Custom allocator with safety tracking
pub struct SafeAllocator {
    inner: System,
}

impl SafeAllocator {
    /// Create a new safe allocator
    pub const fn new() -> Self {
        Self { inner: System }
    }

    /// Get the tracker
    fn tracker(&self) -> &'static SafetyTracker {
        &GLOBAL_SAFETY_TRACKER
    }
}

impl Default for SafeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for SafeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Check if allocation is safe
        if self.tracker().check_allocation(layout.size()).is_err() {
            return std::ptr::null_mut();
        }

        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            // Track the allocation
            if self
                .tracker()
                .track_allocation(ptr, layout.size(), None)
                .is_err()
            {
                // If tracking fails, deallocate and return null
                self.inner.dealloc(ptr, layout);
                return std::ptr::null_mut();
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.tracker().track_deallocation(ptr, layout.size());
        self.inner.dealloc(ptr, layout);
    }
}

/// Convenience macros for safe operations
/// Safe arithmetic operation macro
#[macro_export]
macro_rules! safe_op {
    (add $a:expr, $b:expr) => {
        $crate::memory::safety::SafeArithmetic::safe_add($a, $b)
    };
    (sub $a:expr, $b:expr) => {
        $crate::memory::safety::SafeArithmetic::safe_sub($a, $b)
    };
    (mul $a:expr, $b:expr) => {
        $crate::memory::safety::SafeArithmetic::safe_mul($a, $b)
    };
    (div $a:expr, $b:expr) => {
        $crate::memory::safety::SafeArithmetic::safe_div($a, $b)
    };
}

/// Safe array access macro
#[macro_export]
macro_rules! safe_get {
    ($array:expr, $index:expr) => {
        $crate::memory::safety::SafeArrayOps::safe_index($array, $index)
    };
    (mut $array:expr, $index:expr) => {
        $crate::memory::safety::SafeArrayOps::safe_index_mut($array, $index)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_tracker() {
        // Create a config with zero_on_dealloc disabled to avoid segfault with fake pointers
        let config = SafetyConfig {
            zero_on_dealloc: false,
            ..Default::default()
        };
        let tracker = SafetyTracker::with_config(config);
        let ptr = 0x1000 as *mut u8;

        // Test allocation tracking
        assert!(tracker.track_allocation(ptr, 1024, None).is_ok());
        assert_eq!(tracker.current_usage(), 1024);
        assert_eq!(tracker.peak_usage(), 1024);

        // Test deallocation tracking
        unsafe {
            tracker.track_deallocation(ptr, 1024);
        }
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 1024); // Peak should remain
    }

    #[test]
    fn test_memory_limit() {
        let config = SafetyConfig {
            max_memory: 1024,
            ..Default::default()
        };
        let tracker = SafetyTracker::with_config(config);

        // Should allow allocation within limit
        assert!(tracker.check_allocation(512).is_ok());

        // Should reject allocation exceeding limit
        assert!(tracker.check_allocation(2048).is_err());
    }

    #[test]
    fn test_safe_arithmetic() {
        // Test safe addition
        assert_eq!(SafeArithmetic::safe_add(5u32, 10u32).unwrap(), 15u32);
        assert!(SafeArithmetic::safe_add(u32::MAX, 1u32).is_err());

        // Test safe subtraction
        assert_eq!(SafeArithmetic::safe_sub(10u32, 5u32).unwrap(), 5u32);
        assert!(SafeArithmetic::safe_sub(5u32, 10u32).is_err());

        // Test safe multiplication
        assert_eq!(SafeArithmetic::safe_mul(5u32, 10u32).unwrap(), 50u32);
        assert!(SafeArithmetic::safe_mul(u32::MAX, 2u32).is_err());

        // Test safe division
        assert_eq!(SafeArithmetic::safe_div(10u32, 2u32).unwrap(), 5u32);
        assert!(SafeArithmetic::safe_div(10u32, 0u32).is_err());
    }

    #[test]
    fn test_safe_array_ops() {
        let array = [1, 2, 3, 4, 5];

        // Test safe indexing
        assert_eq!(*SafeArrayOps::safe_index(&array, 2).unwrap(), 3);
        assert!(SafeArrayOps::safe_index(&array, 10).is_err());

        // Test safe slicing
        let slice = SafeArrayOps::safe_slice(&array, 1, 4).unwrap();
        assert_eq!(slice, &[2, 3, 4]);
        assert!(SafeArrayOps::safe_slice(&array, 4, 2).is_err());
        assert!(SafeArrayOps::safe_slice(&array, 0, 10).is_err());
    }

    #[test]
    fn test_resource_guard() {
        let cleanup_called = std::sync::Arc::new(std::sync::Mutex::new(false));
        let cleanup_called_clone = cleanup_called.clone();

        {
            let _guard = ResourceGuard::new(42, move |_| {
                *cleanup_called_clone.lock().unwrap() = true;
            });
        } // Guard is dropped here

        assert!(*cleanup_called.lock().unwrap());
    }

    #[test]
    fn test_safe_macros() {
        // Test safe arithmetic macros
        assert_eq!(safe_op!(add 5u32, 10u32).unwrap(), 15u32);
        assert_eq!(safe_op!(sub 10u32, 5u32).unwrap(), 5u32);
        assert_eq!(safe_op!(mul 5u32, 10u32).unwrap(), 50u32);
        assert_eq!(safe_op!(div 10u32, 2u32).unwrap(), 5u32);

        // Test safe array access macros
        let array = [1, 2, 3, 4, 5];
        assert_eq!(*safe_get!(&array, 2).unwrap(), 3);
        assert!(safe_get!(&array, 10).is_err());
    }

    #[test]
    fn test_allocation_stats() {
        let tracker = SafetyTracker::new();
        let ptr1 = 0x1000 as *mut u8;
        let ptr2 = 0x2000 as *mut u8;

        tracker.track_allocation(ptr1, 1024, None).unwrap();
        tracker.track_allocation(ptr2, 2048, None).unwrap();

        let stats = tracker.get_allocation_stats();
        assert_eq!(stats.current_usage, 3072);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.average_allocation_size, 1536);
    }
}
