//! # CPU Detection and Capabilities
//!
//! This module provides CPU detection and capability assessment for
//! optimizing computational workloads.

use crate::error::{CoreError, CoreResult};
use std::fs;

/// CPU information and capabilities
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// CPU model name
    pub model: String,
    /// CPU vendor
    pub vendor: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (with hyperthreading)
    pub logical_cores: usize,
    /// Base frequency in GHz
    pub base_frequency_ghz: f64,
    /// Maximum frequency in GHz
    pub max_frequency_ghz: f64,
    /// L1 cache size in KB
    pub cache_l1_kb: usize,
    /// L2 cache size in KB
    pub cache_l2_kb: usize,
    /// L3 cache size in KB
    pub cache_l3_kb: usize,
    /// SIMD instruction set support
    pub simd_capabilities: SimdCapabilities,
    /// CPU architecture
    pub architecture: CpuArchitecture,
    /// Additional features
    pub features: Vec<String>,
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            model: "Unknown CPU".to_string(),
            vendor: "Unknown".to_string(),
            physical_cores: 1,
            logical_cores: 1,
            base_frequency_ghz: 1.0,
            max_frequency_ghz: 1.0,
            cache_l1_kb: 32,
            cache_l2_kb: 256,
            cache_l3_kb: 1024,
            simd_capabilities: SimdCapabilities::default(),
            architecture: CpuArchitecture::Unknown,
            features: Vec::new(),
        }
    }
}

impl CpuInfo {
    /// Detect CPU information
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

    /// Detect CPU information on Linux
    #[cfg(target_os = "linux")]
    fn detect_linux() -> CoreResult<Self> {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").map_err(|e| {
            CoreError::IoError(crate::error::ErrorContext::new(format!(
                "Failed to read /proc/cpuinfo: {}",
                e
            )))
        })?;

        let mut model = "Unknown CPU".to_string();
        let mut vendor = "Unknown".to_string();
        let mut logical_cores = 0;
        let mut cache_l1_kb = 32;
        let mut cache_l2_kb = 256;
        let mut cache_l3_kb = 1024;
        let mut flags = Vec::new();

        for line in cpuinfo.lines() {
            if line.starts_with("model name") {
                if let Some(value) = line.split(':').nth(1) {
                    model = value.trim().to_string();
                }
            } else if line.starts_with("vendor_id") {
                if let Some(value) = line.split(':').nth(1) {
                    vendor = value.trim().to_string();
                }
            } else if line.starts_with("processor") {
                logical_cores += 1;
            } else if line.starts_with("flags") {
                if let Some(value) = line.split(':').nth(1) {
                    flags = value.split_whitespace().map(|s| s.to_string()).collect();
                }
            }
        }

        // Try to get cache information from sysfs
        if let Ok(cache_info) = Self::read_cache_info_linux() {
            cache_l1_kb = cache_info.0;
            cache_l2_kb = cache_info.1;
            cache_l3_kb = cache_info.2;
        }

        // Get physical core count
        let physical_cores = Self::get_physical_cores_linux().unwrap_or(logical_cores);

        // Get frequency information
        let (base_freq, max_freq) = Self::get_frequency_info_linux().unwrap_or((2.0, 2.0));

        // Detect SIMD capabilities
        let simd_capabilities = SimdCapabilities::from_flags(&flags);

        // Detect architecture
        let architecture = CpuArchitecture::detect();

        Ok(Self {
            model,
            vendor,
            physical_cores,
            logical_cores,
            base_frequency_ghz: base_freq,
            max_frequency_ghz: max_freq,
            cache_l1_kb,
            cache_l2_kb,
            cache_l3_kb,
            simd_capabilities,
            architecture,
            features: flags,
        })
    }

    /// Get cache information on Linux
    #[cfg(target_os = "linux")]
    fn read_cache_info_linux() -> CoreResult<(usize, usize, usize)> {
        let mut l1_kb = 32;
        let mut l2_kb = 256;
        let mut l3_kb = 1024;

        // Try to read cache sizes from sysfs
        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
            if let Ok(size) = Self::parse_cache_size(content.trim()) {
                l1_kb = size;
            }
        }

        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Ok(size) = Self::parse_cache_size(content.trim()) {
                l2_kb = size;
            }
        }

        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            if let Ok(size) = Self::parse_cache_size(content.trim()) {
                l3_kb = size;
            }
        }

        Ok((l1_kb, l2_kb, l3_kb))
    }

    /// Parse cache size string (e.g., "32K", "256K", "8192K")
    #[allow(dead_code)]
    fn parse_cache_size(size_str: &str) -> CoreResult<usize> {
        if size_str.ends_with('K') || size_str.ends_with('k') {
            let num_str = &size_str[..size_str.len() - 1];
            let size = num_str.parse::<usize>().map_err(|e| {
                CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                    "Failed to parse cache size: {}",
                    e
                )))
            })?;
            Ok(size)
        } else if size_str.ends_with('M') || size_str.ends_with('m') {
            let num_str = &size_str[..size_str.len() - 1];
            let size = num_str.parse::<usize>().map_err(|e| {
                CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                    "Failed to parse cache size: {}",
                    e
                )))
            })? * 1024;
            Ok(size)
        } else {
            let size = size_str.parse::<usize>().map_err(|e| {
                CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                    "Failed to parse cache size: {}",
                    e
                )))
            })?;
            Ok(size)
        }
    }

    /// Get physical core count on Linux
    #[cfg(target_os = "linux")]
    fn get_physical_cores_linux() -> CoreResult<usize> {
        if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
            let mut core_ids = std::collections::HashSet::new();
            for line in content.lines() {
                if line.starts_with("core id") {
                    if let Some(value) = line.split(':').nth(1) {
                        if let Ok(core_id) = value.trim().parse::<usize>() {
                            core_ids.insert(core_id);
                        }
                    }
                }
            }
            if !core_ids.is_empty() {
                return Ok(core_ids.len());
            }
        }

        // Fallback: use available_parallelism
        Ok(std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1))
    }

    /// Get frequency information on Linux
    #[cfg(target_os = "linux")]
    fn get_frequency_info_linux() -> CoreResult<(f64, f64)> {
        let mut base_freq = 2.0;
        let mut max_freq = 2.0;

        // Try to read from cpufreq
        if let Ok(content) =
            fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")
        {
            if let Ok(freq_khz) = content.trim().parse::<f64>() {
                base_freq = freq_khz / 1_000_000.0; // Convert kHz to GHz
            }
        }

        if let Ok(content) =
            fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        {
            if let Ok(freq_khz) = content.trim().parse::<f64>() {
                max_freq = freq_khz / 1_000_000.0; // Convert kHz to GHz
            }
        }

        Ok((base_freq, max_freq))
    }

    /// Detect CPU information on Windows
    #[cfg(target_os = "windows")]
    fn detect_windows() -> CoreResult<Self> {
        // For Windows, we'd use WMI or registry queries
        // This is a simplified implementation
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let physical_cores = logical_cores / 2; // Rough estimate

        Ok(Self {
            model: "Windows CPU".to_string(),
            vendor: "Unknown".to_string(),
            physical_cores,
            logical_cores,
            base_frequency_ghz: 2.5,
            max_frequency_ghz: 3.0,
            cache_l1_kb: 32,
            cache_l2_kb: 256,
            cache_l3_kb: 1024,
            simd_capabilities: SimdCapabilities::detect(),
            architecture: CpuArchitecture::detect(),
            features: Vec::new(),
        })
    }

    /// Detect CPU information on macOS
    #[cfg(target_os = "macos")]
    fn detect_macos() -> CoreResult<Self> {
        // For macOS, we'd use sysctl
        // This is a simplified implementation
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let physical_cores = logical_cores; // Apple Silicon doesn't have hyperthreading

        Ok(Self {
            model: "macOS CPU".to_string(),
            vendor: "Apple".to_string(),
            physical_cores,
            logical_cores,
            base_frequency_ghz: 2.5,
            max_frequency_ghz: 3.0,
            cache_l1_kb: 128,
            cache_l2_kb: 4096,
            cache_l3_kb: 0, // Apple Silicon uses different cache hierarchy
            simd_capabilities: SimdCapabilities::detect(),
            architecture: CpuArchitecture::detect(),
            features: Vec::new(),
        })
    }

    /// Calculate performance score (0.0 to 1.0)
    pub fn performance_score(&self) -> f64 {
        let core_score = (self.physical_cores as f64 / 16.0).min(1.0); // Normalize to 16 cores
        let freq_score = (self.max_frequency_ghz / 4.0).min(1.0); // Normalize to 4 GHz
        let cache_score = (self.cache_l3_kb as f64 / 32768.0).min(1.0); // Normalize to 32MB
        let simd_score = if self.simd_capabilities.avx512 {
            1.0
        } else if self.simd_capabilities.avx2 {
            0.8
        } else if self.simd_capabilities.sse4_2 {
            0.6
        } else {
            0.3
        };

        (core_score + freq_score + cache_score + simd_score) / 4.0
    }

    /// Get optimal thread count for parallel operations
    pub fn optimal_thread_count(&self) -> usize {
        // Use physical cores for CPU-intensive tasks
        // Add some extra threads for I/O bound tasks
        let base_threads = self.physical_cores;
        let io_threads = (self.logical_cores - self.physical_cores).min(2);
        base_threads + io_threads
    }

    /// Get optimal chunk size based on cache hierarchy
    pub fn optimal_chunk_size(&self) -> usize {
        // Use L2 cache size as base, leave some room for other data
        let l2_bytes = self.cache_l2_kb * 1024;
        (l2_bytes * 3 / 4).max(4096) // At least 4KB
    }

    /// Check if CPU supports specific instruction set
    pub fn supports_instruction_set(&self, instruction_set: &str) -> bool {
        self.features
            .iter()
            .any(|f| f.eq_ignore_ascii_case(instruction_set))
    }
}

/// SIMD instruction set capabilities
#[derive(Debug, Clone, Default)]
pub struct SimdCapabilities {
    /// SSE 4.2 support
    pub sse4_2: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512 support
    pub avx512: bool,
    /// ARM NEON support
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities from CPU flags
    pub fn from_flags(flags: &[String]) -> Self {
        let mut capabilities = Self::default();

        for flag in flags {
            match flag.as_str() {
                "sse4_2" => capabilities.sse4_2 = true,
                "avx2" => capabilities.avx2 = true,
                "avx512f" | "avx512" => capabilities.avx512 = true,
                "neon" => capabilities.neon = true,
                _ => {}
            }
        }

        capabilities
    }

    /// Detect SIMD capabilities using runtime detection
    pub fn detect() -> Self {
        let mut capabilities = Self::default();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                capabilities.sse4_2 = true;
            }
            if is_x86_feature_detected!("avx2") {
                capabilities.avx2 = true;
            }
            if is_x86_feature_detected!("avx512f") {
                capabilities.avx512 = true;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is standard on AArch64
            capabilities.neon = true;
        }

        capabilities
    }

    /// Get the best available SIMD instruction set
    pub fn best_available(&self) -> SimdInstructionSet {
        if self.avx512 {
            SimdInstructionSet::Avx512
        } else if self.avx2 {
            SimdInstructionSet::Avx2
        } else if self.sse4_2 {
            SimdInstructionSet::Sse42
        } else if self.neon {
            SimdInstructionSet::Neon
        } else {
            SimdInstructionSet::None
        }
    }
}

/// SIMD instruction set types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdInstructionSet {
    /// No SIMD support
    None,
    /// SSE 4.2
    Sse42,
    /// AVX2
    Avx2,
    /// AVX-512
    Avx512,
    /// ARM NEON
    Neon,
}

impl SimdInstructionSet {
    /// Get vector width in bytes
    pub fn vector_width_bytes(&self) -> usize {
        match self {
            SimdInstructionSet::None => 8,
            SimdInstructionSet::Sse42 => 16,
            SimdInstructionSet::Avx2 => 32,
            SimdInstructionSet::Avx512 => 64,
            SimdInstructionSet::Neon => 16,
        }
    }

    /// Get vector width in f32 elements
    pub fn vector_width_f32(&self) -> usize {
        self.vector_width_bytes() / 4
    }

    /// Get vector width in f64 elements
    pub fn vector_width_f64(&self) -> usize {
        self.vector_width_bytes() / 8
    }
}

/// CPU architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArchitecture {
    /// x86-64 (AMD64)
    X86_64,
    /// AArch64 (ARM64)
    AArch64,
    /// 32-bit x86
    X86,
    /// 32-bit ARM
    Arm,
    /// Unknown architecture
    Unknown,
}

impl CpuArchitecture {
    /// Detect current CPU architecture
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        return CpuArchitecture::X86_64;

        #[cfg(target_arch = "aarch64")]
        return CpuArchitecture::AArch64;

        #[cfg(target_arch = "x86")]
        return CpuArchitecture::X86;

        #[cfg(target_arch = "arm")]
        return CpuArchitecture::Arm;

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "arm"
        )))]
        return CpuArchitecture::Unknown;
    }

    /// Check if architecture supports specific features
    pub fn supports_64bit(&self) -> bool {
        matches!(self, CpuArchitecture::X86_64 | CpuArchitecture::AArch64)
    }

    /// Get native pointer size in bytes
    pub fn pointer_size(&self) -> usize {
        match self {
            CpuArchitecture::X86_64 | CpuArchitecture::AArch64 => 8,
            CpuArchitecture::X86 | CpuArchitecture::Arm => 4,
            CpuArchitecture::Unknown => std::mem::size_of::<usize>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let cpu_info = CpuInfo::detect();
        assert!(cpu_info.is_ok());

        let cpu = cpu_info.unwrap();
        assert!(cpu.logical_cores > 0);
        assert!(cpu.physical_cores > 0);
        assert!(cpu.physical_cores <= cpu.logical_cores);
    }

    #[test]
    fn test_simd_detection() {
        let simd = SimdCapabilities::detect();
        let best = simd.best_available();

        // Should at least detect some capability
        assert_ne!(best, SimdInstructionSet::None);
    }

    #[test]
    fn test_architecture_detection() {
        let arch = CpuArchitecture::detect();
        assert_ne!(arch, CpuArchitecture::Unknown);

        // Test pointer size
        assert!(arch.pointer_size() > 0);
        assert_eq!(arch.pointer_size(), std::mem::size_of::<usize>());
    }

    #[test]
    fn test_performance_score() {
        let cpu = CpuInfo::default();
        let score = cpu.performance_score();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_optimal_parameters() {
        let cpu = CpuInfo::default();

        let thread_count = cpu.optimal_thread_count();
        assert!(thread_count > 0);

        let chunk_size = cpu.optimal_chunk_size();
        assert!(chunk_size >= 4096);
    }

    #[test]
    fn test_vector_widths() {
        let avx2 = SimdInstructionSet::Avx2;
        assert_eq!(avx2.vector_width_bytes(), 32);
        assert_eq!(avx2.vector_width_f32(), 8);
        assert_eq!(avx2.vector_width_f64(), 4);
    }

    #[test]
    fn test_cache_size_parsing() {
        assert_eq!(CpuInfo::parse_cache_size("32K").unwrap(), 32);
        assert_eq!(CpuInfo::parse_cache_size("256k").unwrap(), 256);
        assert_eq!(CpuInfo::parse_cache_size("8M").unwrap(), 8192);
        assert_eq!(CpuInfo::parse_cache_size("1024").unwrap(), 1024);
    }
}
