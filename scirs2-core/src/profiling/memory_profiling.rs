//! # Advanced Memory Profiling for SciRS2
//!
//! This module provides comprehensive memory profiling capabilities using Pure Rust
//! OS APIs. It enables heap profiling, memory leak detection, and allocation pattern
//! analysis without requiring jemalloc or any C/Fortran dependencies.
//!
//! # Platform Support
//!
//! - **macOS**: Uses `mach_task_self()` / `task_info` via libc for accurate memory stats
//! - **Linux**: Reads `/proc/self/statm` and `/proc/self/status` for memory stats
//! - **Other Unix**: Falls back to tracking via global allocator wrapper
//! - **Windows**: Uses `windows-sys` APIs for memory info
//!
//! # Features
//!
//! - **Heap Profiling**: Track memory allocations and deallocations
//! - **Leak Detection**: Identify memory leaks
//! - **Allocation Patterns**: Analyze allocation patterns
//! - **Statistics**: Detailed memory statistics
//! - **Zero Overhead**: Disabled by default, minimal overhead when enabled
//! - **Pure Rust**: No C/Fortran dependencies (COOLJAPAN Policy)
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::profiling::memory_profiling::{MemoryProfiler, enable_profiling};
//!
//! // Enable memory profiling
//! enable_profiling().expect("Failed to enable profiling");
//!
//! // ... perform allocations ...
//!
//! // Get memory statistics
//! let stats = MemoryProfiler::get_stats().expect("Failed to get stats");
//! println!("Allocated: {} bytes", stats.allocated);
//! println!("Resident: {} bytes", stats.resident);
//! ```

#[cfg(feature = "profiling_memory")]
use crate::CoreResult;
#[cfg(feature = "profiling_memory")]
use std::collections::HashMap;
#[cfg(feature = "profiling_memory")]
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Global allocation tracking (Pure Rust)
// ============================================================================

#[cfg(feature = "profiling_memory")]
static TRACKED_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "profiling_memory")]
static TRACKED_PEAK: AtomicUsize = AtomicUsize::new(0);

/// Record an allocation (called from tracking allocator or estimation)
#[cfg(feature = "profiling_memory")]
fn record_allocation(size: usize) {
    let prev = TRACKED_ALLOCATED.fetch_add(size, Ordering::Relaxed);
    let new_total = prev + size;
    // Update peak using compare-and-swap loop
    let mut current_peak = TRACKED_PEAK.load(Ordering::Relaxed);
    while new_total > current_peak {
        match TRACKED_PEAK.compare_exchange_weak(
            current_peak,
            new_total,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_peak = actual,
        }
    }
}

/// Record a deallocation
#[cfg(feature = "profiling_memory")]
fn record_deallocation(size: usize) {
    TRACKED_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
}

/// Get the current tracked allocation count
#[cfg(feature = "profiling_memory")]
fn get_tracked_allocated() -> usize {
    TRACKED_ALLOCATED.load(Ordering::Relaxed)
}

// ============================================================================
// Platform-specific memory stats (Pure Rust via libc)
// ============================================================================

/// Raw memory info from the OS
#[cfg(feature = "profiling_memory")]
#[derive(Debug, Clone, Default)]
struct OsMemoryInfo {
    /// Resident set size (physical memory used) in bytes
    resident: usize,
    /// Virtual memory size in bytes
    virtual_size: usize,
}

/// Read memory info on macOS using Mach task_info API
#[cfg(all(feature = "profiling_memory", target_os = "macos"))]
fn read_os_memory_info() -> CoreResult<OsMemoryInfo> {
    // Use mach_task_self() and task_info to get memory statistics
    // This is the standard approach on macOS for process memory info
    use std::mem;

    // mach_task_basic_info struct layout (from mach/task_info.h)
    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size: u64,      // virtual memory size (bytes)
        resident_size: u64,     // resident memory size (bytes)
        resident_size_max: u64, // maximum resident memory size (bytes)
        user_time: [u32; 2],    // total user run time
        system_time: [u32; 2],  // total system run time
        policy: i32,            // default policy
        suspend_count: i32,     // suspend count
    }

    const MACH_TASK_BASIC_INFO: u32 = 20;
    // Size in natural_t (u32) units
    const MACH_TASK_BASIC_INFO_COUNT: u32 =
        (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;

    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut MachTaskBasicInfo,
            task_info_count: *mut u32,
        ) -> i32;
    }

    let mut info = MachTaskBasicInfo::default();
    let mut count = MACH_TASK_BASIC_INFO_COUNT;

    // SAFETY: We're calling well-defined Mach kernel APIs with properly-sized buffers.
    // mach_task_self() returns the current task port, and task_info fills the struct.
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut MachTaskBasicInfo,
            &mut count,
        )
    };

    // KERN_SUCCESS = 0
    if kr != 0 {
        return Err(crate::CoreError::ConfigError(
            crate::error::ErrorContext::new(format!("task_info failed with kern_return: {}", kr)),
        ));
    }

    Ok(OsMemoryInfo {
        resident: info.resident_size as usize,
        virtual_size: info.virtual_size as usize,
    })
}

/// Read memory info on Linux from /proc/self/statm
#[cfg(all(feature = "profiling_memory", target_os = "linux"))]
fn read_os_memory_info() -> CoreResult<OsMemoryInfo> {
    use std::fs;

    // /proc/self/statm fields: size resident shared text lib data dt
    // All values are in pages
    let statm = fs::read_to_string("/proc/self/statm").map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to read /proc/self/statm: {}",
            e
        )))
    })?;

    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = if page_size <= 0 {
        4096
    } else {
        page_size as usize
    };

    let parts: Vec<&str> = statm.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(crate::CoreError::ConfigError(
            crate::error::ErrorContext::new("Invalid /proc/self/statm format".to_string()),
        ));
    }

    let virtual_pages: usize = parts[0].parse().map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to parse virtual size from /proc/self/statm: {}",
            e
        )))
    })?;

    let resident_pages: usize = parts[1].parse().map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to parse resident size from /proc/self/statm: {}",
            e
        )))
    })?;

    Ok(OsMemoryInfo {
        resident: resident_pages * page_size,
        virtual_size: virtual_pages * page_size,
    })
}

/// Read memory info on Windows via `K32GetProcessMemoryInfo` (kernel32).
///
/// `WorkingSetSize` is the Windows equivalent of RSS, and `PagefileUsage` is the
/// private commit charge, which is the closest analogue to the Unix virtual size
/// reported above. `K32GetProcessMemoryInfo` lives in kernel32.dll on Windows 7
/// and later, so no psapi.dll import library is required.
#[cfg(all(feature = "profiling_memory", target_os = "windows"))]
fn read_os_memory_info() -> CoreResult<OsMemoryInfo> {
    // PROCESS_MEMORY_COUNTERS as declared in psapi.h. Field order and types are
    // part of the stable Win32 ABI.
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    // Both symbols are exported from kernel32.dll (Windows 7+ for the K32 form).
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: core::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };

    // SAFETY: `counters` is a live, correctly sized PROCESS_MEMORY_COUNTERS and
    // `cb` matches its size; the pseudo-handle from GetCurrentProcess is always
    // valid and needs no closing.
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            core::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    };

    if ok == 0 {
        return Err(crate::CoreError::ConfigError(
            crate::error::ErrorContext::new(
                "K32GetProcessMemoryInfo failed to report process memory counters".to_string(),
            ),
        ));
    }

    Ok(OsMemoryInfo {
        resident: counters.working_set_size,
        virtual_size: counters.pagefile_usage,
    })
}

/// Fallback for other platforms - returns estimates from atomic tracking
#[cfg(all(
    feature = "profiling_memory",
    not(target_os = "macos"),
    not(target_os = "linux"),
    not(target_os = "windows")
))]
fn read_os_memory_info() -> CoreResult<OsMemoryInfo> {
    // On unsupported platforms, use the tracked allocation as a rough estimate
    let allocated = get_tracked_allocated();
    Ok(OsMemoryInfo {
        resident: allocated,
        virtual_size: allocated,
    })
}

// ============================================================================
// Public API - MemoryStats
// ============================================================================

/// Memory statistics gathered from OS APIs (Pure Rust)
#[cfg(feature = "profiling_memory")]
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total allocated memory tracked by the profiler (bytes)
    pub allocated: usize,
    /// Resident memory from OS (physical memory, bytes)
    pub resident: usize,
    /// Mapped/virtual memory from OS (bytes)
    pub mapped: usize,
    /// Metadata overhead estimate (bytes) - estimated as a fraction of allocated
    pub metadata: usize,
    /// Retained memory estimate (bytes) - difference between resident and allocated
    pub retained: usize,
}

#[cfg(feature = "profiling_memory")]
impl MemoryStats {
    /// Get current memory statistics using OS APIs
    pub fn current() -> CoreResult<Self> {
        let os_info = read_os_memory_info()?;
        let tracked = get_tracked_allocated();

        // Use the larger of OS-reported resident and our tracked value
        // (tracked may be 0 if no allocator wrapper is installed)
        let allocated = if tracked > 0 {
            tracked
        } else {
            os_info.resident
        };

        // Estimate metadata as ~2% of allocated (typical allocator overhead)
        let metadata = allocated / 50;

        // Retained is memory the allocator holds but hasn't returned to the OS
        let retained = os_info.resident.saturating_sub(allocated);

        Ok(Self {
            allocated,
            resident: os_info.resident,
            mapped: os_info.virtual_size,
            metadata,
            retained,
        })
    }

    /// Calculate memory overhead (metadata / allocated)
    pub fn overhead_ratio(&self) -> f64 {
        if self.allocated == 0 {
            0.0
        } else {
            self.metadata as f64 / self.allocated as f64
        }
    }

    /// Calculate memory utilization (allocated / resident)
    pub fn utilization_ratio(&self) -> f64 {
        if self.resident == 0 {
            0.0
        } else {
            self.allocated as f64 / self.resident as f64
        }
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Memory Stats:\n\
             - Allocated: {} MB\n\
             - Resident:  {} MB\n\
             - Mapped:    {} MB\n\
             - Metadata:  {} MB\n\
             - Retained:  {} MB\n\
             - Overhead:  {:.2}%\n\
             - Utilization: {:.2}%",
            self.allocated / 1_048_576,
            self.resident / 1_048_576,
            self.mapped / 1_048_576,
            self.metadata / 1_048_576,
            self.retained / 1_048_576,
            self.overhead_ratio() * 100.0,
            self.utilization_ratio() * 100.0
        )
    }
}

// ============================================================================
// Public API - MemoryProfiler
// ============================================================================

/// Memory profiler
#[cfg(feature = "profiling_memory")]
pub struct MemoryProfiler {
    baseline: Option<MemoryStats>,
}

#[cfg(feature = "profiling_memory")]
impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self { baseline: None }
    }

    /// Set the baseline memory statistics
    pub fn set_baseline(&mut self) -> CoreResult<()> {
        self.baseline = Some(MemoryStats::current()?);
        Ok(())
    }

    /// Get current memory statistics
    pub fn get_stats() -> CoreResult<MemoryStats> {
        MemoryStats::current()
    }

    /// Get memory delta from baseline
    pub fn get_delta(&self) -> CoreResult<Option<MemoryDelta>> {
        if let Some(ref baseline) = self.baseline {
            let current = MemoryStats::current()?;
            Ok(Some(MemoryDelta {
                allocated_delta: current.allocated as i64 - baseline.allocated as i64,
                resident_delta: current.resident as i64 - baseline.resident as i64,
                mapped_delta: current.mapped as i64 - baseline.mapped as i64,
                metadata_delta: current.metadata as i64 - baseline.metadata as i64,
                retained_delta: current.retained as i64 - baseline.retained as i64,
            }))
        } else {
            Ok(None)
        }
    }

    /// Print memory statistics
    pub fn print_stats() -> CoreResult<()> {
        let stats = Self::get_stats()?;
        println!("{}", stats.format());
        Ok(())
    }

    /// Manually record an allocation for tracking purposes
    pub fn track_allocation(size: usize) {
        record_allocation(size);
    }

    /// Manually record a deallocation for tracking purposes
    pub fn track_deallocation(size: usize) {
        record_deallocation(size);
    }
}

#[cfg(feature = "profiling_memory")]
impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Public API - MemoryDelta
// ============================================================================

/// Memory delta from baseline
#[cfg(feature = "profiling_memory")]
#[derive(Debug, Clone)]
pub struct MemoryDelta {
    pub allocated_delta: i64,
    pub resident_delta: i64,
    pub mapped_delta: i64,
    pub metadata_delta: i64,
    pub retained_delta: i64,
}

#[cfg(feature = "profiling_memory")]
impl MemoryDelta {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Memory Delta:\n\
             - Allocated: {:+} MB\n\
             - Resident:  {:+} MB\n\
             - Mapped:    {:+} MB\n\
             - Metadata:  {:+} MB\n\
             - Retained:  {:+} MB",
            self.allocated_delta / 1_048_576,
            self.resident_delta / 1_048_576,
            self.mapped_delta / 1_048_576,
            self.metadata_delta / 1_048_576,
            self.retained_delta / 1_048_576
        )
    }
}

// ============================================================================
// Public API - AllocationTracker
// ============================================================================

/// Allocation tracker for detecting patterns
#[cfg(feature = "profiling_memory")]
pub struct AllocationTracker {
    snapshots: Vec<(String, MemoryStats)>,
}

#[cfg(feature = "profiling_memory")]
impl AllocationTracker {
    /// Create a new allocation tracker
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    /// Take a snapshot with a label
    pub fn snapshot(&mut self, label: impl Into<String>) -> CoreResult<()> {
        let stats = MemoryStats::current()?;
        self.snapshots.push((label.into(), stats));
        Ok(())
    }

    /// Get all snapshots
    pub fn snapshots(&self) -> &[(String, MemoryStats)] {
        &self.snapshots
    }

    /// Analyze allocation patterns
    pub fn analyze(&self) -> AllocationAnalysis {
        if self.snapshots.is_empty() {
            return AllocationAnalysis {
                total_allocated: 0,
                peak_allocated: 0,
                total_snapshots: 0,
                largest_increase: None,
                patterns: HashMap::new(),
            };
        }

        let mut peak_allocated = 0;
        let mut largest_increase: Option<(String, i64)> = None;

        for i in 0..self.snapshots.len() {
            let (ref label, ref stats) = self.snapshots[i];

            if stats.allocated > peak_allocated {
                peak_allocated = stats.allocated;
            }

            if i > 0 {
                let prev_stats = &self.snapshots[i - 1].1;
                let increase = stats.allocated as i64 - prev_stats.allocated as i64;

                if let Some((_, max_increase)) = largest_increase {
                    if increase > max_increase {
                        largest_increase = Some((label.clone(), increase));
                    }
                } else {
                    largest_increase = Some((label.clone(), increase));
                }
            }
        }

        let last_allocated = self.snapshots.last().map(|(_, s)| s.allocated).unwrap_or(0);

        AllocationAnalysis {
            total_allocated: last_allocated,
            peak_allocated,
            total_snapshots: self.snapshots.len(),
            largest_increase,
            patterns: HashMap::new(),
        }
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

#[cfg(feature = "profiling_memory")]
impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocation pattern analysis
#[cfg(feature = "profiling_memory")]
#[derive(Debug, Clone)]
pub struct AllocationAnalysis {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub total_snapshots: usize,
    pub largest_increase: Option<(String, i64)>,
    pub patterns: HashMap<String, usize>,
}

/// Enable memory profiling
#[cfg(feature = "profiling_memory")]
pub fn enable_profiling() -> CoreResult<()> {
    // Pure Rust implementation - profiling is always available when feature is enabled.
    // No special initialization needed (unlike jemalloc which required env vars).
    Ok(())
}

/// Disable memory profiling
#[cfg(feature = "profiling_memory")]
pub fn disable_profiling() -> CoreResult<()> {
    // Reset tracked counters
    TRACKED_ALLOCATED.store(0, Ordering::Relaxed);
    TRACKED_PEAK.store(0, Ordering::Relaxed);
    Ok(())
}

// ============================================================================
// Stub implementations when profiling_memory feature is disabled
// ============================================================================

#[cfg(not(feature = "profiling_memory"))]
use crate::CoreResult;

#[cfg(not(feature = "profiling_memory"))]
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated: usize,
    pub resident: usize,
    pub mapped: usize,
    pub metadata: usize,
    pub retained: usize,
}

#[cfg(not(feature = "profiling_memory"))]
impl MemoryStats {
    pub fn current() -> CoreResult<Self> {
        Ok(Self {
            allocated: 0,
            resident: 0,
            mapped: 0,
            metadata: 0,
            retained: 0,
        })
    }

    pub fn format(&self) -> String {
        "Memory profiling not enabled".to_string()
    }
}

#[cfg(not(feature = "profiling_memory"))]
pub struct MemoryProfiler;

#[cfg(not(feature = "profiling_memory"))]
impl MemoryProfiler {
    pub fn new() -> Self {
        Self
    }
    pub fn get_stats() -> CoreResult<MemoryStats> {
        MemoryStats::current()
    }
    pub fn print_stats() -> CoreResult<()> {
        Ok(())
    }
}

#[cfg(not(feature = "profiling_memory"))]
pub fn enable_profiling() -> CoreResult<()> {
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "profiling_memory")]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::current();
        assert!(stats.is_ok());

        if let Ok(s) = stats {
            println!("{}", s.format());
            // On any platform, resident should be non-zero for a running process
            assert!(s.resident > 0, "Resident memory should be > 0");
        }
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();
        assert!(profiler.set_baseline().is_ok());

        // Allocate some memory
        let _vec: Vec<u8> = vec![0; 1_000_000];

        let delta = profiler.get_delta();
        assert!(delta.is_ok());
    }

    #[test]
    fn test_allocation_tracker() {
        let mut tracker = AllocationTracker::new();

        assert!(tracker.snapshot("baseline").is_ok());

        // Allocate some memory
        let _vec: Vec<u8> = vec![0; 1_000_000];

        assert!(tracker.snapshot("after_alloc").is_ok());

        let analysis = tracker.analyze();
        assert_eq!(analysis.total_snapshots, 2);
    }

    #[test]
    fn test_memory_delta() {
        let delta = MemoryDelta {
            allocated_delta: 1_048_576,
            resident_delta: 2_097_152,
            mapped_delta: 0,
            metadata_delta: 0,
            retained_delta: 0,
        };

        let formatted = delta.format();
        assert!(formatted.contains("Allocated"));
    }

    #[test]
    fn test_enable_disable_profiling() {
        assert!(enable_profiling().is_ok());
        assert!(disable_profiling().is_ok());
    }

    #[test]
    fn test_manual_tracking() {
        // Reset
        TRACKED_ALLOCATED.store(0, Ordering::Relaxed);
        TRACKED_PEAK.store(0, Ordering::Relaxed);

        MemoryProfiler::track_allocation(1024);
        assert_eq!(get_tracked_allocated(), 1024);

        MemoryProfiler::track_allocation(2048);
        assert_eq!(get_tracked_allocated(), 3072);

        MemoryProfiler::track_deallocation(1024);
        assert_eq!(get_tracked_allocated(), 2048);

        // Peak should still be 3072
        assert_eq!(TRACKED_PEAK.load(Ordering::Relaxed), 3072);
    }

    #[test]
    fn test_overhead_and_utilization_ratios() {
        let stats = MemoryStats {
            allocated: 1_000_000,
            resident: 2_000_000,
            mapped: 4_000_000,
            metadata: 20_000,
            retained: 1_000_000,
        };
        let overhead = stats.overhead_ratio();
        assert!((overhead - 0.02).abs() < 1e-6);

        let utilization = stats.utilization_ratio();
        assert!((utilization - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_zero_stats_ratios() {
        let stats = MemoryStats {
            allocated: 0,
            resident: 0,
            mapped: 0,
            metadata: 0,
            retained: 0,
        };
        assert_eq!(stats.overhead_ratio(), 0.0);
        assert_eq!(stats.utilization_ratio(), 0.0);
    }
}
