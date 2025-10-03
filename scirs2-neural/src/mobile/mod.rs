//! Mobile deployment utilities for neural networks
//!
//! This module provides comprehensive mobile deployment support including:
//! - iOS framework generation with Metal Performance Shaders integration
//! - Android AAR packaging with NNAPI acceleration support
//! - Cross-platform model optimization for mobile constraints
//! - On-device training and fine-tuning capabilities
//! - Battery and thermal management for efficient inference
//! - Model quantization and compression for mobile deployment
//! # Module Organization
//! - [`platform`] - Platform definitions, configurations, and optimization settings
//! - [`generator`] - Core deployment generator and orchestration logic
//! - [`templates`] - Platform-specific code generation templates
//! - [`guides`] - Documentation and integration guide content

pub mod generator;
pub mod guides;
pub mod platform;
pub mod templates;
// Re-export main types and functions for backward compatibility
pub use generator::{
    ChangeType, CodeExample, ConfigurationChange, ConfigurationStep, IntegrationInstructions,
    LatencyMetrics, MemoryMetrics, MobileDeploymentGenerator, MobileDeploymentResult,
    OptimizationReport, OptimizationTechnique, PerformanceImprovement, PerformanceMetrics,
    PlatformPackage, PowerMetrics, ThermalMetrics, TroubleshootingStep,
};
pub use platform::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::models::sequential::Sequential;
    use scirs2_core::random::SeedableRng;
    use std::collections::HashMap;
    use tempfile::TempDir;
    #[test]
    fn test_mobile_platform_ios() {
        let platform = MobilePlatform::IOS {
            min_version: "12.0".to_string(),
            devices: vec![IOSDevice::IPhone, IOSDevice::IPad],
        };
        match platform {
            MobilePlatform::IOS {
                min_version,
                devices,
            } => {
                assert_eq!(min_version, "12.0");
                assert_eq!(devices.len(), 2);
                assert!(devices.contains(&IOSDevice::IPhone));
                assert!(devices.contains(&IOSDevice::IPad));
            }
            _ => unreachable!("Expected iOS platform"),
        }
    }
    fn test_mobile_platform_android() {
        let platform = MobilePlatform::Android {
            min_api_level: 21,
            architectures: vec![AndroidArchitecture::ARM64, AndroidArchitecture::ARMv7],
            MobilePlatform::Android {
                min_api_level,
                architectures,
                assert_eq!(min_api_level, 21);
                assert_eq!(architectures.len(), 2);
                assert!(architectures.contains(&AndroidArchitecture::ARM64));
                assert!(architectures.contains(&AndroidArchitecture::ARMv7));
            _ => unreachable!("Expected Android platform"),
    fn test_mobile_optimization_config_default() {
        let config = MobileOptimizationConfig::default();
        assert_eq!(
            config.compression.pruning.pruning_type,
            PruningType::Magnitude
        );
        assert_eq!(config.compression.pruning.sparsity_level, 0.5);
        assert!(config.compression.distillation.enable);
            config.quantization.strategy,
            QuantizationStrategy::PostTraining
        assert_eq!(config.quantization.precision.weights, 8);
        assert_eq!(config.power.power_mode, PowerMode::Balanced);
    fn test_mobile_deployment_generator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let mut model: Sequential<f32> = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(dense);
            devices: vec![IOSDevice::IPhone],
        let optimization = MobileOptimizationConfig::default();
        let metadata = crate::serving::PackageMetadata {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Test".to_string(),
            license: "MIT".to_string(),
            platforms: vec!["ios".to_string()],
            dependencies: HashMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            runtime_requirements: crate::serving::RuntimeRequirements {
                min_memory_mb: 256,
                cpu_requirements: crate::serving::CpuRequirements {
                    min_cores: 1,
                    instruction_sets: Vec::new(),
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: Vec::new(),
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
            checksum: "test".to_string(),
        let generator = MobileDeploymentGenerator::new(
            model,
            platform,
            optimization,
            metadata,
            temp_dir.path().to_path_buf(),
        match generator.platform() {
                ref min_version, ..
    fn test_quantization_precision() {
        let precision = QuantizationPrecision {
            weights: 8,
            activations: 8,
            bias: Some(32),
        assert_eq!(precision.weights, 8);
        assert_eq!(precision.activations, 8);
        assert_eq!(precision.bias, Some(32));
    fn test_thermal_thresholds() {
        let thresholds = ThermalThresholds {
            warning: 70.0,
            critical: 80.0,
            emergency: 90.0,
        assert_eq!(thresholds.warning, 70.0);
        assert_eq!(thresholds.critical, 80.0);
        assert_eq!(thresholds.emergency, 90.0);
        assert!(thresholds.warning < thresholds.critical);
        assert!(thresholds.critical < thresholds.emergency);
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            latency: LatencyMetrics {
                average_ms: 15.2,
                p95_ms: 23.1,
                p99_ms: 28.7,
                cold_start_ms: 45.3,
            memory: MemoryMetrics {
                peak_mb: 128.5,
                average_mb: 85.2,
                footprint_mb: 64.1,
                efficiency: 1.2,
            power: PowerMetrics {
                average_mw: 1250.0,
                peak_mw: 2100.0,
                energy_per_inference_mj: 19.0,
                battery_impact_hours: 8.5,
            thermal: ThermalMetrics {
                peak_temperature: 42.5,
                average_temperature: 38.2,
                throttling_events: 0,
                time_to_limit_s: 300.0,
        assert_eq!(metrics.latency.average_ms, 15.2);
        assert_eq!(metrics.memory.peak_mb, 128.5);
        assert_eq!(metrics.power.average_mw, 1250.0);
        assert_eq!(metrics.thermal.peak_temperature, 42.5);
    fn test_module_integration() {
        // Test that all modules work together
        // Create a simple model
        // Create configurations
        let wasm_config = MobileOptimizationConfig::default();
            name: "test-model".to_string(),
            description: "Test mobile model".to_string(),
            author: "SciRS2".to_string(),
        // Create generator
            wasm_config,
        // Test generator creation
            MobilePlatform::IOS { .. } => {
                // Generator created successfully
    #[allow(clippy::const_is_empty)]
    fn test_template_content() {
        // Test that template content is not empty
        assert!(!templates::IOS_INFO_PLIST.is_empty());
        assert!(!templates::SWIFT_WRAPPER.is_empty());
        assert!(!templates::OBJC_HEADER.is_empty());
        assert!(!templates::OBJC_IMPL.is_empty());
        assert!(!templates::JAVA_WRAPPER.is_empty());
        assert!(!templates::KOTLIN_WRAPPER.is_empty());
        assert!(!templates::JNI_HEADER.is_empty());
        assert!(!templates::JNI_IMPL.is_empty());
    fn test_guide_content() {
        // Test that guide content is not empty
        assert!(!guides::IOS_INTEGRATION_GUIDE.is_empty());
        assert!(!guides::ANDROID_INTEGRATION_GUIDE.is_empty());
        assert!(!guides::OPTIMIZATION_GUIDE.is_empty());
        // Test that guides contain expected sections
        assert!(guides::IOS_INTEGRATION_GUIDE.contains("# iOS Integration Guide"));
        assert!(guides::ANDROID_INTEGRATION_GUIDE.contains("# Android Integration Guide"));
        assert!(guides::OPTIMIZATION_GUIDE.contains("# Mobile Optimization Guide"));
}
