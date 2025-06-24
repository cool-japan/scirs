//! # Memory Detection and Analysis
//!
//! This module provides memory system detection and analysis for
//! optimizing memory-intensive operations.

use crate::error::CoreResult;
use crate::CoreError;
use std::fs;

/// Memory system information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total physical memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Memory page size in bytes
    pub page_size: usize,
    /// Memory bandwidth estimate (GB/s)
    pub bandwidth_gbps: f64,
    /// Memory latency estimate (nanoseconds)
    pub latency_ns: f64,
    /// NUMA nodes count
    pub numa_nodes: usize,
    /// Swap space information
    pub swap_info: SwapInfo,
    /// Memory pressure indicators
    pub pressure: MemoryPressure,
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            total_memory: 8 * 1024 * 1024 * 1024,     // 8GB default
            available_memory: 4 * 1024 * 1024 * 1024, // 4GB default
            page_size: 4096,
            bandwidth_gbps: 20.0,
            latency_ns: 100.0,
            numa_nodes: 1,
            swap_info: SwapInfo::default(),
            pressure: MemoryPressure::default(),
        }
    }
}

impl MemoryInfo {
    /// Detect memory information
    pub fn detect() -> CoreResult<Self> {
        #[cfg(target_os = "linux")]
        return Self::detect_linux();

        #[cfg(target_os = "windows")]
        return Self::detect_windows();

        #[cfg(target_os = "macos")]
        return Self::detect_macos();

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        return Ok(Self::default());
    }

    /// Detect memory information on Linux
    #[cfg(target_os = "linux")]
    fn detect_linux() -> CoreResult<Self> {
        let meminfo = fs::read_to_string("/proc/meminfo").map_err(|e| {
            CoreError::IoError(crate::error::ErrorContext::new(format!(
                "Failed to read /proc/meminfo: {}",
                e
            )))
        })?;

        let mut total_memory = 8 * 1024 * 1024 * 1024; // Default 8GB
        let mut available_memory = 4 * 1024 * 1024 * 1024; // Default 4GB
        let mut swap_total = 0;
        let mut swap_free = 0;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(value) = Self::parse_meminfo_value(line) {
                    total_memory = value * 1024; // Convert KB to bytes
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(value) = Self::parse_meminfo_value(line) {
                    available_memory = value * 1024; // Convert KB to bytes
                }
            } else if line.starts_with("SwapTotal:") {
                if let Some(value) = Self::parse_meminfo_value(line) {
                    swap_total = value * 1024; // Convert KB to bytes
                }
            } else if line.starts_with("SwapFree:") {
                if let Some(value) = Self::parse_meminfo_value(line) {
                    swap_free = value * 1024; // Convert KB to bytes
                }
            }
        }

        let page_size = Self::get_page_size();
        let numa_nodes = Self::detect_numa_nodes();
        let bandwidth_gbps = Self::estimate_memory_bandwidth();
        let latency_ns = Self::estimate_memory_latency();
        let pressure = Self::detect_memory_pressure();

        let swap_info = SwapInfo {
            total: swap_total,
            free: swap_free,
            used: swap_total - swap_free,
        };

        Ok(Self {
            total_memory,
            available_memory,
            page_size,
            bandwidth_gbps,
            latency_ns,
            numa_nodes,
            swap_info,
            pressure,
        })
    }

    /// Parse value from /proc/meminfo line
    #[allow(dead_code)]
    fn parse_meminfo_value(line: &str) -> Option<usize> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            parts[1].parse().ok()
        } else {
            None
        }
    }

    /// Get system page size
    #[allow(dead_code)]
    fn get_page_size() -> usize {
        // Use a simplified implementation without libc dependency
        4096 // Most common page size on modern systems
    }

    /// Detect NUMA nodes
    #[allow(dead_code)]
    fn detect_numa_nodes() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
                let node_count = entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with("node"))
                    .count();
                if node_count > 0 {
                    return node_count;
                }
            }
        }

        1 // Default to single NUMA node
    }

    /// Estimate memory bandwidth
    #[allow(dead_code)]
    fn estimate_memory_bandwidth() -> f64 {
        // This is a simplified estimation
        // In a real implementation, we might:
        // 1. Read from hardware databases
        // 2. Run memory benchmarks
        // 3. Query system information APIs

        #[cfg(target_os = "linux")]
        {
            // Try to estimate based on memory type
            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                if content.contains("MemTotal:") {
                    // Rough estimation based on typical memory configurations
                    return 25.0; // GB/s for typical DDR4
                }
            }
        }

        20.0 // Default estimation
    }

    /// Estimate memory latency
    #[allow(dead_code)]
    fn estimate_memory_latency() -> f64 {
        // Simplified estimation
        // Real implementation might benchmark memory access patterns
        80.0 // nanoseconds - typical DDR4 latency
    }

    /// Detect memory pressure
    #[allow(dead_code)]
    fn detect_memory_pressure() -> MemoryPressure {
        #[cfg(target_os = "linux")]
        {
            // Check memory pressure via PSI (Pressure Stall Information)
            if let Ok(content) = fs::read_to_string("/proc/pressure/memory") {
                for line in content.lines() {
                    if line.starts_with("some avg10=") {
                        if let Some(pressure_str) = line.split('=').nth(1) {
                            if let Some(pressure_val) = pressure_str.split_whitespace().next() {
                                if let Ok(pressure) = pressure_val.parse::<f64>() {
                                    return if pressure > 50.0 {
                                        MemoryPressure::High
                                    } else if pressure > 20.0 {
                                        MemoryPressure::Medium
                                    } else {
                                        MemoryPressure::Low
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }

        MemoryPressure::Low
    }

    /// Detect memory information on Windows
    #[cfg(target_os = "windows")]
    fn detect_windows() -> CoreResult<Self> {
        // Windows implementation would use GetPhysicallyInstalledSystemMemory
        // and GlobalMemoryStatusEx APIs
        Ok(Self::default())
    }

    /// Detect memory information on macOS
    #[cfg(target_os = "macos")]
    fn detect_macos() -> CoreResult<Self> {
        // macOS implementation would use sysctl
        Ok(Self::default())
    }

    /// Calculate performance score (0.0 to 1.0)
    pub fn performance_score(&self) -> f64 {
        let capacity_score =
            (self.total_memory as f64 / (64.0 * 1024.0 * 1024.0 * 1024.0)).min(1.0); // Normalize to 64GB
        let bandwidth_score = (self.bandwidth_gbps / 100.0).min(1.0); // Normalize to 100 GB/s
        let latency_score = (200.0 / self.latency_ns).min(1.0); // Lower latency is better
        let availability_score = self.available_memory as f64 / self.total_memory as f64;

        (capacity_score + bandwidth_score + latency_score + availability_score) / 4.0
    }

    /// Get optimal chunk size for memory operations
    pub fn optimal_chunk_size(&self) -> usize {
        // Base chunk size on available memory and page size
        let available_mb = self.available_memory / (1024 * 1024);
        let chunk_mb = if available_mb > 1024 {
            64 // 64MB for systems with >1GB available
        } else if available_mb > 256 {
            16 // 16MB for systems with >256MB available
        } else {
            4 // 4MB for smaller systems
        };

        (chunk_mb * 1024 * 1024).max(self.page_size * 16) // At least 16 pages
    }

    /// Check if memory pressure is high
    pub fn is_under_pressure(&self) -> bool {
        matches!(self.pressure, MemoryPressure::High)
            || (self.available_memory * 100 / self.total_memory) < 10 // Less than 10% available
    }

    /// Get recommended memory allocation strategy
    pub fn allocation_strategy(&self) -> AllocationStrategy {
        if self.numa_nodes > 1 {
            AllocationStrategy::NumaAware
        } else if self.is_under_pressure() {
            AllocationStrategy::Conservative
        } else {
            AllocationStrategy::Aggressive
        }
    }
}

/// Swap space information
#[derive(Debug, Clone, Default)]
pub struct SwapInfo {
    /// Total swap space in bytes
    pub total: usize,
    /// Free swap space in bytes
    pub free: usize,
    /// Used swap space in bytes
    pub used: usize,
}

impl SwapInfo {
    /// Check if swap is being used
    pub fn is_swap_active(&self) -> bool {
        self.used > 0
    }

    /// Get swap usage percentage
    pub fn usage_percentage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryPressure {
    /// Low memory pressure
    #[default]
    Low,
    /// Medium memory pressure
    Medium,
    /// High memory pressure
    High,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Aggressive allocation for performance
    Aggressive,
    /// Conservative allocation to avoid pressure
    Conservative,
    /// NUMA-aware allocation for multi-node systems
    NumaAware,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_detection() {
        let memory_info = MemoryInfo::detect();
        assert!(memory_info.is_ok());

        let memory = memory_info.unwrap();
        assert!(memory.total_memory > 0);
        assert!(memory.available_memory > 0);
        assert!(memory.available_memory <= memory.total_memory);
        assert!(memory.page_size > 0);
    }

    #[test]
    fn test_performance_score() {
        let memory = MemoryInfo::default();
        let score = memory.performance_score();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_optimal_chunk_size() {
        let memory = MemoryInfo::default();
        let chunk_size = memory.optimal_chunk_size();
        assert!(chunk_size >= memory.page_size * 16);
    }

    #[test]
    fn test_swap_info() {
        let swap = SwapInfo {
            total: 1024 * 1024 * 1024, // 1GB
            free: 512 * 1024 * 1024,   // 512MB
            used: 512 * 1024 * 1024,   // 512MB
        };

        assert!(swap.is_swap_active());
        assert_eq!(swap.usage_percentage(), 50.0);
    }

    #[test]
    fn test_allocation_strategy() {
        let mut memory = MemoryInfo::default();

        // Test conservative strategy under pressure
        memory.available_memory = memory.total_memory / 20; // 5% available
        assert_eq!(
            memory.allocation_strategy(),
            AllocationStrategy::Conservative
        );

        // Test NUMA-aware strategy
        memory.numa_nodes = 2;
        memory.available_memory = memory.total_memory / 2; // 50% available
        assert_eq!(memory.allocation_strategy(), AllocationStrategy::NumaAware);
    }

    #[test]
    fn test_memory_pressure() {
        assert_eq!(MemoryPressure::Low, MemoryPressure::Low);
        assert_ne!(MemoryPressure::Low, MemoryPressure::High);
    }
}
