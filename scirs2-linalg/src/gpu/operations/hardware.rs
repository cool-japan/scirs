//! Hardware capability profiling for detailed device analysis

use std::collections::HashMap;

/// Hardware capability profiler for detailed device analysis
#[derive(Debug)]
pub struct HardwareCapabilityProfiler {
    /// Device-specific performance characteristics
    device_profiles: HashMap<String, DeviceProfile>,
    /// Benchmark results for different operation types
    benchmark_results: HashMap<String, HashMap<String, f64>>,
    /// Capability flags for different features
    capability_flags: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub peak_flops_sp: f64,
    pub peak_flops_dp: f64,
    pub memory_bandwidth: f64,
    pub l1_cachesize: usize,
    pub l2_cachesize: usize,
    pub shared_memory: usize,
    pub register_count: usize,
    pub tensor_core_support: bool,
    pub mixed_precision_support: bool,
}

impl HardwareCapabilityProfiler {
    pub fn new() -> Self {
        Self {
            device_profiles: HashMap::new(),
            benchmark_results: HashMap::new(),
            capability_flags: HashMap::new(),
        }
    }

    /// Add a device profile
    pub fn add_device_profile(&mut self, device_id: String, profile: DeviceProfile) {
        self.device_profiles.insert(device_id, profile);
    }

    /// Get device profile by ID
    pub fn get_device_profile(&self, device_id: &str) -> Option<&DeviceProfile> {
        self.device_profiles.get(device_id)
    }

    /// Record benchmark result for a device and operation
    pub fn record_benchmark(&mut self, device_id: String, operation: String, performance: f64) {
        self.benchmark_results
            .entry(device_id)
            .or_insert_with(HashMap::new)
            .insert(operation, performance);
    }

    /// Get benchmark result for a device and operation
    pub fn get_benchmark(&self, device_id: &str, operation: &str) -> Option<f64> {
        self.benchmark_results
            .get(device_id)
            .and_then(|ops| ops.get(operation))
            .copied()
    }

    /// Add capability flag for a device
    pub fn add_capability_flag(&mut self, device_id: String, capability: String) {
        self.capability_flags
            .entry(device_id)
            .or_insert_with(Vec::new)
            .push(capability);
    }

    /// Check if device has a specific capability
    pub fn has_capability(&self, device_id: &str, capability: &str) -> bool {
        self.capability_flags
            .get(device_id)
            .map(|caps| caps.contains(&capability.to_string()))
            .unwrap_or(false)
    }

    /// Get all capabilities for a device
    pub fn get_capabilities(&self, device_id: &str) -> Option<&Vec<String>> {
        self.capability_flags.get(device_id)
    }

    /// Get all available devices
    pub fn get_available_devices(&self) -> Vec<&str> {
        self.device_profiles.keys().map(|s| s.as_str()).collect()
    }

    /// Profile device capabilities by running benchmarks
    pub fn profile_device(&mut self, device_id: String) -> Result<DeviceProfile, String> {
        // This would normally run actual hardware benchmarks
        // For now, we'll create a mock profile

        let mock_profile = DeviceProfile {
            peak_flops_sp: 5000.0,         // 5 TFLOPS single precision
            peak_flops_dp: 2500.0,         // 2.5 TFLOPS double precision
            memory_bandwidth: 500.0,       // 500 GB/s
            l1_cachesize: 64 * 1024,       // 64KB L1 cache
            l2_cachesize: 2 * 1024 * 1024, // 2MB L2 cache
            shared_memory: 48 * 1024,      // 48KB shared memory
            register_count: 65536,         // 64K registers
            tensor_core_support: true,
            mixed_precision_support: true,
        };

        // Add some mock benchmark results
        let mut operations = HashMap::new();
        operations.insert("matmul_f32".to_string(), 1200.0); // GFLOPS
        operations.insert("matmul_f64".to_string(), 600.0);
        operations.insert("matvec_f32".to_string(), 800.0);
        operations.insert("elementwise_f32".to_string(), 2000.0);

        self.device_profiles
            .insert(device_id.clone(), mock_profile.clone());
        self.benchmark_results.insert(device_id.clone(), operations);

        // Add capability flags
        let mut capabilities = Vec::new();
        capabilities.push("cuda_compute_7_5".to_string());
        capabilities.push("tensor_cores".to_string());
        capabilities.push("mixed_precision".to_string());
        capabilities.push("unified_memory".to_string());
        self.capability_flags.insert(device_id, capabilities);

        Ok(mock_profile)
    }

    /// Compare performance between devices for a specific operation
    pub fn compare_devices(&self, operation: &str) -> Vec<(String, f64)> {
        let mut results = Vec::new();

        for (device_id, benchmarks) in &self.benchmark_results {
            if let Some(&performance) = benchmarks.get(operation) {
                results.push((device_id.clone(), performance));
            }
        }

        // Sort by performance (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get optimal device for a specific operation
    pub fn get_optimal_device(&self, operation: &str) -> Option<String> {
        self.compare_devices(operation)
            .first()
            .map(|(device_id, _)| device_id.clone())
    }

    /// Generate hardware compatibility report
    pub fn generate_compatibility_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Hardware Compatibility Report\n");
        report.push_str("============================\n\n");

        for (device_id, profile) in &self.device_profiles {
            report.push_str(&format!("Device: {}\n", device_id));
            report.push_str(&format!(
                "  Peak FP32 FLOPS: {:.1} GFLOPS\n",
                profile.peak_flops_sp
            ));
            report.push_str(&format!(
                "  Peak FP64 FLOPS: {:.1} GFLOPS\n",
                profile.peak_flops_dp
            ));
            report.push_str(&format!(
                "  Memory Bandwidth: {:.1} GB/s\n",
                profile.memory_bandwidth
            ));
            report.push_str(&format!("  L1 Cache: {} KB\n", profile.l1_cachesize / 1024));
            report.push_str(&format!(
                "  L2 Cache: {} MB\n",
                profile.l2_cachesize / (1024 * 1024)
            ));
            report.push_str(&format!(
                "  Tensor Cores: {}\n",
                if profile.tensor_core_support {
                    "Yes"
                } else {
                    "No"
                }
            ));
            report.push_str(&format!(
                "  Mixed Precision: {}\n",
                if profile.mixed_precision_support {
                    "Yes"
                } else {
                    "No"
                }
            ));

            if let Some(capabilities) = self.capability_flags.get(device_id) {
                report.push_str("  Capabilities: ");
                report.push_str(&capabilities.join(", "));
                report.push_str("\n");
            }

            if let Some(benchmarks) = self.benchmark_results.get(device_id) {
                report.push_str("  Benchmark Results:\n");
                for (op, perf) in benchmarks {
                    report.push_str(&format!("    {}: {:.1} GFLOPS\n", op, perf));
                }
            }

            report.push_str("\n");
        }

        report
    }

    /// Clear all profiles and benchmarks
    pub fn clear(&mut self) {
        self.device_profiles.clear();
        self.benchmark_results.clear();
        self.capability_flags.clear();
    }
}

impl Default for HardwareCapabilityProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self {
            peak_flops_sp: 1000.0,
            peak_flops_dp: 500.0,
            memory_bandwidth: 100.0,
            l1_cachesize: 32 * 1024,
            l2_cachesize: 1024 * 1024,
            shared_memory: 32 * 1024,
            register_count: 32768,
            tensor_core_support: false,
            mixed_precision_support: false,
        }
    }
}
