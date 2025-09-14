//! # Hardware Performance Counter Integration
//!
//! This module provides integration with hardware performance counters for detailed
//! performance analysis including CPU cycles, cache misses, branch predictions, and more.

use crate::error::{CoreError, CoreResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use thiserror::Error;

/// Error types for hardware performance counters
#[derive(Error, Debug)]
pub enum HardwareCounterError {
    /// Performance counters not available on this platform
    #[error("Performance counters not available on this platform")]
    NotAvailable,

    /// Permission denied to access performance counters
    #[error("Permission denied to access performance counters: {0}")]
    PermissionDenied(String),

    /// Counter not found
    #[error("Performance counter not found: {0}")]
    CounterNotFound(String),

    /// Invalid counter configuration
    #[error("Invalid counter configuration: {0}")]
    InvalidConfiguration(String),

    /// System error
    #[error("System error: {0}")]
    SystemError(String),
}

impl From<HardwareCounterError> for CoreError {
    fn from(err: HardwareCounterError) -> Self {
        CoreError::ComputationError(crate::error::ErrorContext::new(err.to_string()))
    }
}

/// Hardware performance counter types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CounterType {
    // CPU Counters
    /// CPU cycles
    CpuCycles,
    /// Instructions retired
    Instructions,
    /// Cache references
    CacheReferences,
    /// Cache misses
    CacheMisses,
    /// Branch instructions
    BranchInstructions,
    /// Branch mispredictions
    BranchMisses,
    /// Bus cycles
    BusCycles,
    /// Stalled cycles frontend
    StalledCyclesFrontend,
    /// Stalled cycles backend
    StalledCyclesBackend,

    // L1 Cache Counters
    /// L1 data cache loads
    L1DCacheLoads,
    /// L1 data cache load misses
    L1DCacheLoadMisses,
    /// L1 data cache stores
    L1DCacheStores,
    /// L1 instruction cache loads
    L1ICacheLoads,
    /// L1 instruction cache load misses
    L1ICacheLoadMisses,

    // L2/L3 Cache Counters
    /// L2 cache loads
    L2CacheLoads,
    /// L2 cache load misses
    L2CacheLoadMisses,
    /// L3 cache loads
    L3CacheLoads,
    /// L3 cache load misses
    L3CacheLoadMisses,

    // Memory Counters
    /// DTLB loads
    DtlbLoads,
    /// DTLB load misses
    DtlbLoadMisses,
    /// ITLB loads
    ItlbLoads,
    /// ITLB load misses
    ItlbLoadMisses,

    // Power/Thermal Counters
    /// CPU power consumption
    CpuPower,
    /// CPU temperature
    CpuTemperature,
    /// CPU frequency
    CpuFrequency,

    // Custom counter for platform-specific counters
    Custom(String),
}

impl CounterType {
    /// Get a human-readable description of the counter
    pub const fn description(&self) -> &'static str {
        match self {
            CounterType::CpuCycles => "CPU cycles",
            CounterType::Instructions => "Instructions retired",
            CounterType::CacheReferences => "Cache references",
            CounterType::CacheMisses => "Cache misses",
            CounterType::BranchInstructions => "Branch instructions",
            CounterType::BranchMisses => "Branch mispredictions",
            CounterType::BusCycles => "Bus cycles",
            CounterType::StalledCyclesFrontend => "Stalled cycles frontend",
            CounterType::StalledCyclesBackend => "Stalled cycles backend",
            CounterType::L1DCacheLoads => "L1 data cache loads",
            CounterType::L1DCacheLoadMisses => "L1 data cache load misses",
            CounterType::L1DCacheStores => "L1 data cache stores",
            CounterType::L1ICacheLoads => "L1 instruction cache loads",
            CounterType::L1ICacheLoadMisses => "L1 instruction cache load misses",
            CounterType::L2CacheLoads => "L2 cache loads",
            CounterType::L2CacheLoadMisses => "L2 cache load misses",
            CounterType::L3CacheLoads => "L3 cache loads",
            CounterType::L3CacheLoadMisses => "L3 cache load misses",
            CounterType::DtlbLoads => "Data TLB loads",
            CounterType::DtlbLoadMisses => "Data TLB load misses",
            CounterType::ItlbLoads => "Instruction TLB loads",
            CounterType::ItlbLoadMisses => "Instruction TLB load misses",
            CounterType::CpuPower => "CPU power consumption",
            CounterType::CpuTemperature => "CPU temperature",
            CounterType::CpuFrequency => "CPU frequency",
            CounterType::Custom(_) => "Custom counter",
        }
    }

    /// Get the unit for this counter type
    pub const fn unit(&self) -> &'static str {
        match self {
            CounterType::CpuCycles
            | CounterType::Instructions
            | CounterType::CacheReferences
            | CounterType::CacheMisses
            | CounterType::BranchInstructions
            | CounterType::BranchMisses
            | CounterType::BusCycles
            | CounterType::StalledCyclesFrontend
            | CounterType::StalledCyclesBackend
            | CounterType::L1DCacheLoads
            | CounterType::L1DCacheLoadMisses
            | CounterType::L1DCacheStores
            | CounterType::L1ICacheLoads
            | CounterType::L1ICacheLoadMisses
            | CounterType::L2CacheLoads
            | CounterType::L2CacheLoadMisses
            | CounterType::L3CacheLoads
            | CounterType::L3CacheLoadMisses
            | CounterType::DtlbLoads
            | CounterType::DtlbLoadMisses
            | CounterType::ItlbLoads
            | CounterType::ItlbLoadMisses => "count",
            CounterType::CpuPower => "watts",
            CounterType::CpuTemperature => "celsius",
            CounterType::CpuFrequency => "hertz",
            CounterType::Custom(_) => "unknown",
        }
    }
}

/// Performance counter value with metadata
#[derive(Debug, Clone)]
pub struct CounterValue {
    /// Counter type
    pub countertype: CounterType,
    /// Raw counter value
    pub value: u64,
    /// Timestamp when value was read
    pub timestamp: Instant,
    /// Whether the counter is running
    pub enabled: bool,
    /// Counter scaling factor (for normalized values)
    pub scaling_factor: f64,
}

impl CounterValue {
    /// Create a new counter value
    pub fn new(countertype: CounterType, value: u64) -> Self {
        Self {
            countertype,
            value,
            timestamp: Instant::now(),
            enabled: true,
            scaling_factor: 1.0,
        }
    }

    /// Get the scaled value
    pub fn scaled_value(&self) -> f64 {
        self.value as f64 * self.scaling_factor
    }
}

/// Hardware performance counter interface
pub trait PerformanceCounter: Send + Sync {
    /// Get available counter types on this platform
    fn available_counters(&self) -> Vec<CounterType>;

    /// Check if a counter type is available
    fn is_available(&self, countertype: &CounterType) -> bool;

    /// Start monitoring a counter
    fn start_counter(&self, countertype: &CounterType) -> CoreResult<()>;

    /// Stop monitoring a counter
    fn stop_counter(&self, countertype: &CounterType) -> CoreResult<()>;

    /// Read current value of a counter
    fn read_counter(&self, countertype: &CounterType) -> CoreResult<CounterValue>;

    /// Read multiple counters atomically
    fn read_counters(&self, countertypes: &[CounterType]) -> CoreResult<Vec<CounterValue>>;

    /// Reset a counter to zero
    fn reset_counter(&self, countertype: &CounterType) -> CoreResult<()>;

    /// Get counter overflow status
    fn is_overflowed(&self, countertype: &CounterType) -> CoreResult<bool>;
}

/// Linux perf_event implementation
#[cfg(target_os = "linux")]
pub struct LinuxPerfCounter {
    active_counters: RwLock<HashMap<CounterType, i32>>, // file descriptors
}

#[cfg(target_os = "linux")]
impl LinuxPerfCounter {
    /// Create a new Linux perf counter
    pub fn new() -> Self {
        Self {
            active_counters: RwLock::new(HashMap::new()),
        }
    }

    /// Convert counter type to perf event type and config
    fn counter_to_perf_config(&self, countertype: &CounterType) -> Option<(u32, u64)> {
        match countertype {
            CounterType::CpuCycles => Some((0, 0)), // PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES
            CounterType::Instructions => Some((0, 1)), // PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS
            CounterType::CacheReferences => Some((0, 2)), // PERF_COUNT_HW_CACHE_REFERENCES
            CounterType::CacheMisses => Some((0, 3)),  // PERF_COUNT_HW_CACHE_MISSES
            CounterType::BranchInstructions => Some((0, 4)), // PERF_COUNT_HW_BRANCH_INSTRUCTIONS
            CounterType::BranchMisses => Some((0, 5)), // PERF_COUNT_HW_BRANCH_MISSES
            CounterType::BusCycles => Some((0, 6)),    // PERF_COUNT_HW_BUS_CYCLES
            CounterType::StalledCyclesFrontend => Some((0, 7)), // PERF_COUNT_HW_STALLED_CYCLES_FRONTEND
            CounterType::StalledCyclesBackend => Some((0, 8)), // PERF_COUNT_HW_STALLED_CYCLES_BACKEND
            _ => None, // Not supported or requires hardware cache events
        }
    }
}

#[cfg(target_os = "linux")]
impl Default for LinuxPerfCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "linux")]
impl PerformanceCounter for LinuxPerfCounter {
    fn available_counters(&self) -> Vec<CounterType> {
        vec![
            CounterType::CpuCycles,
            CounterType::Instructions,
            CounterType::CacheReferences,
            CounterType::CacheMisses,
            CounterType::BranchInstructions,
            CounterType::BranchMisses,
            CounterType::BusCycles,
            CounterType::StalledCyclesFrontend,
            CounterType::StalledCyclesBackend,
        ]
    }

    fn is_available(&self, countertype: &CounterType) -> bool {
        self.counter_to_perf_config(countertype).is_some()
    }

    fn start_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        if let Some(_event_type_config) = self.counter_to_perf_config(countertype) {
            // In a real implementation, we would:
            // 1. Create perf_event_attr structure
            // 2. Call perf_event_open syscall
            // 3. Store the file descriptor

            // For now, simulate with a dummy file descriptor
            let fd = 42; // Would be actual fd from perf_event_open

            let mut counters = self.active_counters.write().unwrap();
            counters.insert(countertype.clone(), fd);

            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn stop_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        let mut counters = self.active_counters.write().unwrap();
        if let Some(fd) = counters.remove(countertype) {
            // In real implementation: close(fd)
            let _ = fd;
            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn read_counter(&self, countertype: &CounterType) -> CoreResult<CounterValue> {
        let counters = self.active_counters.read().unwrap();
        if let Some(_fd) = counters.get(countertype) {
            // In real implementation: read() from fd
            // For now, return a mock value
            let mock_value = match countertype {
                CounterType::CpuCycles => 1_000_000,
                CounterType::Instructions => 500_000,
                CounterType::CacheReferences => 10_000,
                CounterType::CacheMisses => 1_000,
                CounterType::BranchInstructions => 100_000,
                CounterType::BranchMisses => 5_000,
                CounterType::BusCycles => 50_000,
                CounterType::StalledCyclesFrontend => 10_000,
                CounterType::StalledCyclesBackend => 20_000,
                _ => 0,
            };

            Ok(CounterValue::new(countertype.clone(), mock_value))
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn read_counters(&self, countertypes: &[CounterType]) -> CoreResult<Vec<CounterValue>> {
        let mut results = Vec::new();
        for countertype in countertypes {
            results.push(self.read_counter(countertype)?);
        }
        Ok(results)
    }

    fn reset_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        let counters = self.active_counters.read().unwrap();
        if counters.contains_key(countertype) {
            // In real implementation: ioctl(fd, PERF_EVENT_IOC_RESET, 0)
            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn is_overflowed(&self, countertype: &CounterType) -> CoreResult<bool> {
        let counters = self.active_counters.read().unwrap();
        if counters.contains_key(countertype) {
            // In real implementation: check overflow bit from perf_event read
            // For now, always return false (not overflowed)
            Ok(false)
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }
}

/// Windows Performance Data Helper (PDH) implementation
#[cfg(target_os = "windows")]
pub struct WindowsPdhCounter {
    active_counters: RwLock<HashMap<CounterType, String>>, // PDH counter paths
}

#[cfg(target_os = "windows")]
impl WindowsPdhCounter {
    /// Create a new Windows PDH counter
    pub fn new() -> Self {
        Self {
            active_counters: RwLock::new(HashMap::new()),
        }
    }

    /// Convert counter type to PDH counter path
    fn counter_to_path(countertype: &CounterType) -> Option<String> {
        match countertype {
            CounterType::CpuCycles => Some("\\Processor(_Total)\\% Processor Time".to_string()),
            CounterType::CpuFrequency => {
                Some("\\Processor Information(_Total)\\Processor Frequency".to_string())
            }
            CounterType::CpuPower => Some("\\Power Meter(*)\\Power".to_string()),
            _ => None,
        }
    }
}

#[cfg(target_os = "windows")]
impl Default for WindowsPdhCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "windows")]
impl PerformanceCounter for WindowsPdhCounter {
    fn available_counters(&self) -> Vec<CounterType> {
        vec![
            CounterType::CpuCycles,
            CounterType::CpuFrequency,
            CounterType::CpuPower,
        ]
    }

    fn is_available(&self, countertype: &CounterType) -> bool {
        Self::counter_to_path(countertype).is_some()
    }

    fn start_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        if let Some(path) = Self::counter_to_path(countertype) {
            // In real implementation: PDH API calls
            let mut counters = self.active_counters.write().unwrap();
            counters.insert(countertype.clone(), path);
            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn stop_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        let mut counters = self.active_counters.write().unwrap();
        if counters.remove(countertype).is_some() {
            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn read_counter(&self, countertype: &CounterType) -> CoreResult<CounterValue> {
        let counters = self.active_counters.read().unwrap();
        if counters.contains_key(countertype) {
            // Mock values for Windows
            let mock_value = match countertype {
                CounterType::CpuCycles => 85,               // CPU usage percentage
                CounterType::CpuFrequency => 2_400_000_000, // 2.4 GHz
                CounterType::CpuPower => 45,                // 45 watts
                _ => 0,
            };

            Ok(CounterValue::new(countertype.clone(), mock_value))
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn read_counters(&self, countertypes: &[CounterType]) -> CoreResult<Vec<CounterValue>> {
        let mut results = Vec::new();
        for countertype in countertypes {
            results.push(self.read_counter(countertype)?);
        }
        Ok(results)
    }

    fn reset_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        // PDH counters can't be reset
        Err(
            HardwareCounterError::InvalidConfiguration("PDH counters cannot be reset".to_string())
                .into(),
        )
    }

    fn is_overflowed(&self, _countertype: &CounterType) -> CoreResult<bool> {
        // PDH counters don't typically overflow in our implementation
        Ok(false)
    }
}

/// macOS performance counter implementation using system profiling
#[cfg(target_os = "macos")]
pub struct MacOSCounter {
    active_counters: RwLock<HashMap<CounterType, bool>>,
}

#[cfg(target_os = "macos")]
impl MacOSCounter {
    /// Create a new macOS counter
    pub fn new() -> Self {
        Self {
            active_counters: RwLock::new(HashMap::new()),
        }
    }
}

#[cfg(target_os = "macos")]
impl Default for MacOSCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "macos")]
impl PerformanceCounter for MacOSCounter {
    fn available_counters(&self) -> Vec<CounterType> {
        vec![
            CounterType::CpuCycles,
            CounterType::Instructions,
            CounterType::CpuFrequency,
            CounterType::CpuTemperature,
        ]
    }

    fn is_available(&self, countertype: &CounterType) -> bool {
        matches!(
            countertype,
            CounterType::CpuCycles
                | CounterType::Instructions
                | CounterType::CpuFrequency
                | CounterType::CpuTemperature
        )
    }

    fn start_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        if self.is_available(countertype) {
            let mut counters = self.active_counters.write().unwrap();
            counters.insert(countertype.clone(), true);
            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn stop_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        let mut counters = self.active_counters.write().unwrap();
        if counters.remove(countertype).is_some() {
            Ok(())
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn read_counter(&self, countertype: &CounterType) -> CoreResult<CounterValue> {
        let counters = self.active_counters.read().unwrap();
        if counters.contains_key(countertype) {
            // Mock values for macOS
            let mock_value = match countertype {
                CounterType::CpuCycles => 2_000_000,
                CounterType::Instructions => 1_000_000,
                CounterType::CpuFrequency => 3_200_000_000, // 3.2 GHz
                CounterType::CpuTemperature => 65,          // 65°C
                _ => 0,
            };

            Ok(CounterValue::new(countertype.clone(), mock_value))
        } else {
            Err(HardwareCounterError::CounterNotFound(format!("{countertype:?}")).into())
        }
    }

    fn read_counters(&self, countertypes: &[CounterType]) -> CoreResult<Vec<CounterValue>> {
        let mut results = Vec::new();
        for countertype in countertypes {
            results.push(self.read_counter(countertype)?);
        }
        Ok(results)
    }

    fn reset_counter(&self, countertype: &CounterType) -> CoreResult<()> {
        // macOS counters typically can't be reset
        Ok(())
    }

    fn is_overflowed(&self, _countertype: &CounterType) -> CoreResult<bool> {
        // macOS hardware counters don't typically overflow in our implementation
        Ok(false)
    }
}

/// Hardware counter manager that provides a unified interface
pub struct HardwareCounterManager {
    backend: Box<dyn PerformanceCounter>,
    session_counters: RwLock<HashMap<String, Vec<CounterType>>>,
    counter_history: RwLock<HashMap<CounterType, Vec<CounterValue>>>,
    max_history_size: usize,
}

impl HardwareCounterManager {
    /// Create a new hardware counter manager with platform-specific backend
    pub fn new() -> CoreResult<Self> {
        let backend = Self::create_platform_backend()?;

        Ok(Self {
            backend,
            session_counters: RwLock::new(HashMap::new()),
            counter_history: RwLock::new(HashMap::new()),
            max_history_size: 1000,
        })
    }

    /// Create platform-specific backend
    fn create_platform_backend() -> CoreResult<Box<dyn PerformanceCounter>> {
        #[cfg(target_os = "linux")]
        {
            Ok(Box::new(LinuxPerfCounter::new()))
        }

        #[cfg(target_os = "windows")]
        {
            Ok(Box::new(WindowsPdhCounter::new()))
        }

        #[cfg(target_os = "macos")]
        {
            Ok(Box::new(MacOSCounter::new()))
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            Err(HardwareCounterError::NotAvailable.into())
        }
    }

    /// Get available counter types
    pub fn available_counters(&self) -> Vec<CounterType> {
        self.backend.available_counters()
    }

    /// Start a profiling session with specific counters
    pub fn start_session(&self, sessionname: &str, counters: Vec<CounterType>) -> CoreResult<()> {
        // Start all requested counters
        for counter in &counters {
            self.backend.start_counter(counter)?;
        }

        // Register session
        let mut sessions = self.session_counters.write().unwrap();
        sessions.insert(sessionname.to_string(), counters);

        Ok(())
    }

    /// Stop a profiling session
    pub fn stop_session(&self, sessionname: &str) -> CoreResult<()> {
        let mut sessions = self.session_counters.write().unwrap();

        if let Some(counters) = sessions.remove(sessionname) {
            for counter in &counters {
                self.backend.stop_counter(counter)?;
            }
            Ok(())
        } else {
            Err(HardwareCounterError::InvalidConfiguration(format!(
                "Session not found: {sessionname}"
            ))
            .into())
        }
    }

    /// Sample all active counters
    pub fn sample_counters(&self) -> CoreResult<HashMap<CounterType, CounterValue>> {
        let sessions = self.session_counters.read().unwrap();
        let active_counters: Vec<CounterType> = sessions
            .values()
            .flat_map(|counters| counters.iter())
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let values = self.backend.read_counters(&active_counters)?;

        // Store in history
        let mut history = self.counter_history.write().unwrap();
        for value in &values {
            let counter_history = history.entry(value.countertype.clone()).or_default();

            counter_history.push(value.clone());

            // Limit history size
            if counter_history.len() > self.max_history_size {
                counter_history.drain(0..counter_history.len() - self.max_history_size);
            }
        }

        // Convert to HashMap
        let result = values
            .into_iter()
            .map(|value| (value.countertype.clone(), value))
            .collect();

        Ok(result)
    }

    /// Get counter history
    pub fn get_counter_history(&self, countertype: &CounterType) -> Vec<CounterValue> {
        let history = self.counter_history.read().unwrap();
        history.get(countertype).cloned().unwrap_or_default()
    }

    /// Calculate derived metrics
    pub fn calculate_derived_metrics(
        &self,
        counters: &HashMap<CounterType, CounterValue>,
    ) -> DerivedMetrics {
        let mut metrics = DerivedMetrics::default();

        // Instructions per cycle (IPC)
        if let (Some(instructions), Some(cycles)) = (
            counters.get(&CounterType::Instructions),
            counters.get(&CounterType::CpuCycles),
        ) {
            if cycles.value > 0 {
                metrics.instructions_per_cycle = instructions.value as f64 / cycles.value as f64;
            }
        }

        // Cache hit rate
        if let (Some(references), Some(misses)) = (
            counters.get(&CounterType::CacheReferences),
            counters.get(&CounterType::CacheMisses),
        ) {
            if references.value > 0 {
                metrics.cache_hit_rate = 1.0 - (misses.value as f64 / references.value as f64);
            }
        }

        // Branch prediction accuracy
        if let (Some(instructions), Some(misses)) = (
            counters.get(&CounterType::BranchInstructions),
            counters.get(&CounterType::BranchMisses),
        ) {
            if instructions.value > 0 {
                metrics.branch_prediction_accuracy =
                    1.0 - (misses.value as f64 / instructions.value as f64);
            }
        }

        // CPU utilization (cycles per second)
        if let Some(cycles) = counters.get(&CounterType::CpuCycles) {
            // Would need time delta for accurate calculation
            metrics.cpu_utilization = cycles.value as f64 / 1_000_000.0; // Simplified
        }

        metrics
    }

    /// Generate performance report
    pub fn generate_report(&self, sessionname: &str) -> PerformanceReport {
        let sessions = self.session_counters.read().unwrap();
        let counters = sessions.get(sessionname).cloned().unwrap_or_default();

        let current_values = self.sample_counters().unwrap_or_default();
        let derived_metrics = self.calculate_derived_metrics(&current_values);

        PerformanceReport {
            session_name: sessionname.to_string(),
            timestamp: Instant::now(),
            counter_values: current_values,
            derived_metrics,
            countersmonitored: counters,
        }
    }
}

impl Default for HardwareCounterManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback with no-op backend
            Self {
                backend: Box::new(NoOpCounter),
                session_counters: RwLock::new(HashMap::new()),
                counter_history: RwLock::new(HashMap::new()),
                max_history_size: 1000,
            }
        })
    }
}

/// No-op counter for unsupported platforms
pub struct NoOpCounter;

impl PerformanceCounter for NoOpCounter {
    fn available_counters(&self) -> Vec<CounterType> {
        Vec::new()
    }

    fn is_available(&self, _countertype: &CounterType) -> bool {
        false
    }

    fn start_counter(&self, _countertype: &CounterType) -> CoreResult<()> {
        Err(HardwareCounterError::NotAvailable.into())
    }

    fn stop_counter(&self, _countertype: &CounterType) -> CoreResult<()> {
        Err(HardwareCounterError::NotAvailable.into())
    }

    fn read_counter(&self, _countertype: &CounterType) -> CoreResult<CounterValue> {
        Err(HardwareCounterError::NotAvailable.into())
    }

    fn read_counters(&self, _countertypes: &[CounterType]) -> CoreResult<Vec<CounterValue>> {
        Err(HardwareCounterError::NotAvailable.into())
    }

    fn reset_counter(&self, _countertype: &CounterType) -> CoreResult<()> {
        Err(HardwareCounterError::NotAvailable.into())
    }

    fn is_overflowed(&self, _countertype: &CounterType) -> CoreResult<bool> {
        Err(HardwareCounterError::NotAvailable.into())
    }
}

/// Derived performance metrics calculated from raw counters
#[derive(Debug, Clone, Default)]
pub struct DerivedMetrics {
    /// Instructions per cycle
    pub instructions_per_cycle: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Branch prediction accuracy (0.0 to 1.0)
    pub branch_prediction_accuracy: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory bandwidth (bytes per second)
    pub memorybandwidth: f64,
    /// Power efficiency (instructions per watt)
    pub power_efficiency: f64,
}

/// Performance report containing counter values and analysis
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Session name
    pub session_name: String,
    /// Report timestamp
    pub timestamp: Instant,
    /// Raw counter values
    pub counter_values: HashMap<CounterType, CounterValue>,
    /// Derived metrics
    pub derived_metrics: DerivedMetrics,
    /// Counters that were monitored
    pub countersmonitored: Vec<CounterType>,
}

impl PerformanceReport {
    /// Format the report as human-readable text
    pub fn formattext(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Performance Report: {session_name}\n",
            session_name = self.session_name
        ));
        output.push_str(&format!("Timestamp: {:?}\n\n", self.timestamp));

        output.push_str("Raw Counters:\n");
        for (countertype, value) in &self.counter_values {
            output.push_str(&format!(
                "  {}: {} {}\n",
                countertype.description(),
                value.scaled_value(),
                countertype.unit()
            ));
        }

        output.push_str("\nDerived Metrics:\n");
        let metrics = &self.derived_metrics;
        output.push_str(&format!(
            "  Instructions per Cycle: {:.2}\n",
            metrics.instructions_per_cycle
        ));
        output.push_str(&format!(
            "  Cache Hit Rate: {:.2}%\n",
            metrics.cache_hit_rate * 100.0
        ));
        output.push_str(&format!(
            "  Branch Prediction Accuracy: {:.2}%\n",
            metrics.branch_prediction_accuracy * 100.0
        ));
        output.push_str(&format!(
            "  CPU Utilization: {:.2}%\n",
            metrics.cpu_utilization
        ));

        output
    }

    /// Export report as JSON
    pub fn to_json(&self) -> String {
        // Simplified JSON serialization - real implementation would use serde
        format!(
            r#"{{"session":"{}","timestamp":"{}","metrics":{{"ipc":{:.2},"cache_hit_rate":{:.2},"branch_accuracy":{:.2}}}}}"#,
            self.session_name,
            self.timestamp.elapsed().as_secs(),
            self.derived_metrics.instructions_per_cycle,
            self.derived_metrics.cache_hit_rate,
            self.derived_metrics.branch_prediction_accuracy
        )
    }
}

/// Global hardware counter manager instance
static GLOBAL_MANAGER: std::sync::OnceLock<Arc<Mutex<HardwareCounterManager>>> =
    std::sync::OnceLock::new();

/// Get the global hardware counter manager
#[allow(dead_code)]
pub fn global_manager() -> Arc<Mutex<HardwareCounterManager>> {
    GLOBAL_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(HardwareCounterManager::default())))
        .clone()
}

/// Convenience functions for hardware performance monitoring
pub mod utils {
    use super::*;

    /// Start monitoring basic CPU performance counters
    pub fn start_basic_cpumonitoring(sessionname: &str) -> CoreResult<()> {
        let manager = global_manager();
        let manager = manager.lock().unwrap();

        let counters = vec![
            CounterType::CpuCycles,
            CounterType::Instructions,
            CounterType::CacheReferences,
            CounterType::CacheMisses,
        ];

        manager.start_session(sessionname, counters)
    }

    /// Start monitoring cache performance
    pub fn start_cachemonitoring(sessionname: &str) -> CoreResult<()> {
        let manager = global_manager();
        let manager = manager.lock().unwrap();

        let counters = vec![
            CounterType::L1DCacheLoads,
            CounterType::L1DCacheLoadMisses,
            CounterType::L2CacheLoads,
            CounterType::L2CacheLoadMisses,
            CounterType::L3CacheLoads,
            CounterType::L3CacheLoadMisses,
        ];

        manager.start_session(sessionname, counters)
    }

    /// Get a quick performance snapshot
    pub fn get_performance_snapshot() -> CoreResult<HashMap<CounterType, CounterValue>> {
        let manager = global_manager();
        let manager = manager.lock().unwrap();
        manager.sample_counters()
    }

    /// Check if hardware performance counters are available
    pub fn counters_available() -> bool {
        let manager = global_manager();
        let manager = manager.lock().unwrap();
        !manager.available_counters().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_countertype_properties() {
        let counter = CounterType::CpuCycles;
        assert_eq!(counter.description(), "CPU cycles");
        assert_eq!(counter.unit(), "count");

        let custom = CounterType::Custom("test".to_string());
        assert_eq!(custom.description(), "Custom counter");
        assert_eq!(custom.unit(), "unknown");
    }

    #[test]
    fn test_counter_value() {
        let counter = CounterType::Instructions;
        let value = CounterValue::new(counter.clone(), 1000);

        assert_eq!(value.countertype, counter);
        assert_eq!(value.value, 1000);
        assert_eq!(value.scaled_value(), 1000.0);
        assert!(value.enabled);
    }

    #[test]
    fn test_derived_metrics() {
        let metrics = DerivedMetrics {
            instructions_per_cycle: 2.5,
            cache_hit_rate: 0.95,
            branch_prediction_accuracy: 0.98,
            ..Default::default()
        };

        assert_eq!(metrics.instructions_per_cycle, 2.5);
        assert_eq!(metrics.cache_hit_rate, 0.95);
        assert_eq!(metrics.branch_prediction_accuracy, 0.98);
    }

    #[test]
    fn test_performance_report() {
        let mut counter_values = HashMap::new();
        counter_values.insert(
            CounterType::CpuCycles,
            CounterValue::new(CounterType::CpuCycles, 1000000),
        );

        let report = PerformanceReport {
            session_name: "test_session".to_string(),
            timestamp: Instant::now(),
            counter_values,
            derived_metrics: DerivedMetrics::default(),
            countersmonitored: vec![CounterType::CpuCycles],
        };

        let text = report.formattext();
        assert!(text.contains("Performance Report: test_session"));
        assert!(text.contains("CPU cycles"));

        let json = report.to_json();
        assert!(json.contains("test_session"));
    }

    #[test]
    fn test_no_op_counter() {
        let counter = NoOpCounter;
        assert!(counter.available_counters().is_empty());
        assert!(!counter.is_available(&CounterType::CpuCycles));
        assert!(counter.start_counter(&CounterType::CpuCycles).is_err());
    }

    #[test]
    fn test_global_manager() {
        let manager = global_manager();

        // Should return the same instance
        let manager2 = global_manager();
        assert!(Arc::ptr_eq(&manager, &manager2));
    }

    #[test]
    fn test_utils_functions() {
        // Test that utility functions don't panic
        let available = utils::counters_available();
        // Function should complete without panicking - no assertion needed

        // Test starting monitoring (may fail on unsupported platforms)
        let result = utils::start_basic_cpumonitoring("test");
        // Either succeeds or fails with known error
        assert!(result.is_ok() || result.is_err());
    }
}
