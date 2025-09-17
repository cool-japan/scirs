//! Configuration structures for advanced interpolation coordination
//!
//! This module contains all configuration types and settings used to
//! control the behavior of the advanced interpolation coordinator.

/// Configuration for advanced interpolation operations
#[derive(Debug, Clone)]
pub struct AdvancedInterpolationConfig {
    /// Enable intelligent method selection
    pub enable_method_selection: bool,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Enable quantum-inspired optimization
    pub enable_quantum_optimization: bool,
    /// Enable cross-domain knowledge transfer
    pub enable_knowledge_transfer: bool,
    /// Target accuracy tolerance
    pub target_accuracy: f64,
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Performance monitoring interval (operations)
    pub monitoring_interval: usize,
    /// Enable real-time learning
    pub enable_real_time_learning: bool,
    /// Enable error prediction
    pub enable_error_prediction: bool,
    /// Cache size limit (number of interpolants)
    pub cache_size_limit: usize,
    /// Adaptation threshold (performance improvement needed)
    pub adaptation_threshold: f64,
    /// Enable hardware-specific optimization
    pub enable_hardware_optimization: bool,
}

impl Default for AdvancedInterpolationConfig {
    fn default() -> Self {
        Self {
            enable_method_selection: true,
            enable_adaptive_optimization: true,
            enable_quantum_optimization: true,
            enable_knowledge_transfer: true,
            target_accuracy: 1e-6,
            max_memory_mb: 4096, // 4GB default (consistent with FFT)
            monitoring_interval: 50,
            enable_real_time_learning: true,
            enable_error_prediction: true,
            cache_size_limit: 500,
            adaptation_threshold: 0.05, // 5% improvement (consistent with FFT)
            enable_hardware_optimization: true,
        }
    }
}

/// Configuration builder for advanced interpolation
#[derive(Debug, Default)]
pub struct AdvancedInterpolationConfigBuilder {
    config: AdvancedInterpolationConfig,
}

impl AdvancedInterpolationConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: AdvancedInterpolationConfig::default(),
        }
    }

    /// Enable or disable intelligent method selection
    pub fn method_selection(mut self, enabled: bool) -> Self {
        self.config.enable_method_selection = enabled;
        self
    }

    /// Enable or disable adaptive optimization
    pub fn adaptive_optimization(mut self, enabled: bool) -> Self {
        self.config.enable_adaptive_optimization = enabled;
        self
    }

    /// Enable or disable quantum-inspired optimization
    pub fn quantum_optimization(mut self, enabled: bool) -> Self {
        self.config.enable_quantum_optimization = enabled;
        self
    }

    /// Enable or disable cross-domain knowledge transfer
    pub fn knowledge_transfer(mut self, enabled: bool) -> Self {
        self.config.enable_knowledge_transfer = enabled;
        self
    }

    /// Set target accuracy tolerance
    pub fn target_accuracy(mut self, accuracy: f64) -> Self {
        self.config.target_accuracy = accuracy;
        self
    }

    /// Set maximum memory usage in MB
    pub fn max_memory_mb(mut self, memory_mb: usize) -> Self {
        self.config.max_memory_mb = memory_mb;
        self
    }

    /// Set performance monitoring interval
    pub fn monitoring_interval(mut self, interval: usize) -> Self {
        self.config.monitoring_interval = interval;
        self
    }

    /// Enable or disable real-time learning
    pub fn real_time_learning(mut self, enabled: bool) -> Self {
        self.config.enable_real_time_learning = enabled;
        self
    }

    /// Enable or disable error prediction
    pub fn error_prediction(mut self, enabled: bool) -> Self {
        self.config.enable_error_prediction = enabled;
        self
    }

    /// Set cache size limit
    pub fn cache_size_limit(mut self, limit: usize) -> Self {
        self.config.cache_size_limit = limit;
        self
    }

    /// Set adaptation threshold
    pub fn adaptation_threshold(mut self, threshold: f64) -> Self {
        self.config.adaptation_threshold = threshold;
        self
    }

    /// Enable or disable hardware-specific optimization
    pub fn hardware_optimization(mut self, enabled: bool) -> Self {
        self.config.enable_hardware_optimization = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> AdvancedInterpolationConfig {
        self.config
    }
}

/// Preset configurations for common use cases
pub struct ConfigPresets;

impl ConfigPresets {
    /// High accuracy configuration (prioritizes accuracy over speed)
    pub fn high_accuracy() -> AdvancedInterpolationConfig {
        AdvancedInterpolationConfigBuilder::new()
            .target_accuracy(1e-12)
            .method_selection(true)
            .adaptive_optimization(true)
            .quantum_optimization(true)
            .knowledge_transfer(true)
            .max_memory_mb(8192)
            .build()
    }

    /// High performance configuration (prioritizes speed over accuracy)
    pub fn high_performance() -> AdvancedInterpolationConfig {
        AdvancedInterpolationConfigBuilder::new()
            .target_accuracy(1e-3)
            .method_selection(true)
            .adaptive_optimization(false)
            .quantum_optimization(false)
            .knowledge_transfer(false)
            .max_memory_mb(2048)
            .monitoring_interval(100)
            .build()
    }

    /// Balanced configuration (balanced accuracy and performance)
    pub fn balanced() -> AdvancedInterpolationConfig {
        AdvancedInterpolationConfig::default()
    }

    /// Memory constrained configuration (minimizes memory usage)
    pub fn memory_constrained() -> AdvancedInterpolationConfig {
        AdvancedInterpolationConfigBuilder::new()
            .max_memory_mb(512)
            .cache_size_limit(50)
            .adaptive_optimization(false)
            .quantum_optimization(false)
            .build()
    }

    /// Real-time configuration (optimized for real-time applications)
    pub fn real_time() -> AdvancedInterpolationConfig {
        AdvancedInterpolationConfigBuilder::new()
            .target_accuracy(1e-4)
            .method_selection(true)
            .adaptive_optimization(false)
            .quantum_optimization(false)
            .knowledge_transfer(false)
            .real_time_learning(true)
            .monitoring_interval(10)
            .build()
    }

    /// Research configuration (enables all features for experimentation)
    pub fn research() -> AdvancedInterpolationConfig {
        AdvancedInterpolationConfigBuilder::new()
            .method_selection(true)
            .adaptive_optimization(true)
            .quantum_optimization(true)
            .knowledge_transfer(true)
            .real_time_learning(true)
            .error_prediction(true)
            .hardware_optimization(true)
            .max_memory_mb(16384)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AdvancedInterpolationConfig::default();
        assert!(config.enable_method_selection);
        assert!(config.enable_adaptive_optimization);
        assert_eq!(config.target_accuracy, 1e-6);
    }

    #[test]
    fn test_config_builder() {
        let config = AdvancedInterpolationConfigBuilder::new()
            .target_accuracy(1e-9)
            .method_selection(false)
            .max_memory_mb(1024)
            .build();

        assert_eq!(config.target_accuracy, 1e-9);
        assert!(!config.enable_method_selection);
        assert_eq!(config.max_memory_mb, 1024);
    }

    #[test]
    fn test_preset_configurations() {
        let high_acc = ConfigPresets::high_accuracy();
        assert_eq!(high_acc.target_accuracy, 1e-12);
        assert_eq!(high_acc.max_memory_mb, 8192);

        let high_perf = ConfigPresets::high_performance();
        assert_eq!(high_perf.target_accuracy, 1e-3);
        assert!(!high_perf.enable_adaptive_optimization);

        let memory_const = ConfigPresets::memory_constrained();
        assert_eq!(memory_const.max_memory_mb, 512);
        assert_eq!(memory_const.cache_size_limit, 50);
    }
}
