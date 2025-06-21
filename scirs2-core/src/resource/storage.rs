//! # Storage Detection and Capabilities
//!
//! This module provides storage device detection for I/O optimization.

use crate::error::CoreResult;

/// Storage device information
#[derive(Debug, Clone)]
pub struct StorageInfo {
    /// Storage devices
    pub devices: Vec<StorageDevice>,
    /// Optimal I/O size in bytes
    pub optimal_io_size: usize,
    /// Queue depth
    pub queue_depth: usize,
    /// Total capacity
    pub capacity: usize,
    /// Available space
    pub available: usize,
}

impl Default for StorageInfo {
    fn default() -> Self {
        Self {
            devices: vec![StorageDevice::default()],
            optimal_io_size: 64 * 1024, // 64KB
            queue_depth: 32,
            capacity: 500 * 1024 * 1024 * 1024,  // 500GB
            available: 250 * 1024 * 1024 * 1024, // 250GB
        }
    }
}

impl StorageInfo {
    /// Detect storage information
    pub fn detect() -> CoreResult<Self> {
        // Simplified implementation
        Ok(Self::default())
    }

    /// Check if storage supports async I/O
    pub fn supports_async_io(&self) -> bool {
        self.devices.iter().any(|d| d.supports_async)
    }

    /// Check if primary storage is SSD
    pub fn is_ssd(&self) -> bool {
        self.devices.first().map(|d| d.is_ssd).unwrap_or(false)
    }
}

/// Storage device information
#[derive(Debug, Clone)]
pub struct StorageDevice {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: StorageType,
    /// Is SSD (vs HDD)
    pub is_ssd: bool,
    /// Supports async I/O
    pub supports_async: bool,
    /// Device capacity
    pub capacity: usize,
    /// Optimal I/O size
    pub optimal_io_size: usize,
}

impl Default for StorageDevice {
    fn default() -> Self {
        Self {
            name: "sda".to_string(),
            device_type: StorageType::Ssd,
            is_ssd: true,
            supports_async: true,
            capacity: 500 * 1024 * 1024 * 1024, // 500GB
            optimal_io_size: 64 * 1024,         // 64KB
        }
    }
}

/// Storage device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    /// Hard Disk Drive
    Hdd,
    /// Solid State Drive
    Ssd,
    /// NVMe SSD
    Nvme,
    /// Network attached storage
    Network,
    /// RAM disk
    Ram,
    /// Unknown
    Unknown,
}
