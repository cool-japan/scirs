//! WebAssembly target support for neural networks
//!
//! This module provides comprehensive WebAssembly compilation and deployment support including:
//! - WASM module generation with optimized neural network execution
//! - JavaScript/TypeScript bindings for web integration
//! - WebGL/WebGPU acceleration support
//! - Memory management and streaming for large models
//! - Web Workers integration for background inference
//! - Progressive loading and caching strategies
//!
//! # Module Organization
//!
//! - [`bindings`] - JavaScript and TypeScript binding generation
//! - [`memory`] - Memory management and configuration
//! - [`exports`] - WASM compilation and export configuration

pub mod bindings;
pub mod exports;
pub mod memory;

// Re-export main types and functions for backward compatibility

// From bindings module
pub use bindings::{
    BindingGenerator, BundleFormat, BundlingConfig, ModuleSystem, WebBindingConfig,
    WebBindingLanguage,
};

// From memory module
pub use memory::{
    CacheStorage, CacheStrategy, CachingConfig, LoadingStrategy, MemoryAlignment, MemoryBreakdown,
    MemoryGrowthStrategy, MemoryManager, MemoryRequirements, ParallelConfig, PreloadingConfig,
    ProgressiveLoadingConfig, VersioningStrategy, WasmMemoryConfig, WasmMemoryExport,
    WasmMemoryImport,
};

// From exports module
pub use exports::{
    BundleInfo, InlineLevel, MessagingStrategy, PerformanceHint, ProfilingConfig, ProfilingFormat,
    TextureFormat, WasmCompilationConfig, WasmCompilationResult, WasmCompiler, WasmDebugConfig,
    WasmExports, WasmFeatures, WasmFunctionExport, WasmFunctionImport, WasmGlobalExport,
    WasmGlobalImport, WasmImports, WasmOptimization, WasmSignature, WasmTableExport,
    WasmTableImport, WasmType, WasmVersion, WebAccelerationConfig, WebGLConfig, WebGPUConfig,
    WebIntegrationConfig, WorkerConfig, WorkerPoolConfig, WorkerType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::models::sequential::Sequential;
    use crate::serving::PackageMetadata;
    use rand::SeedableRng;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_wasm_module_integration() {
        // Test that all modules work together
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        // Create a simple model
        let mut model: Sequential<f32> = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(dense);

        // Create configurations
        let wasm_config = WasmCompilationConfig::default();
        let web_config = WebIntegrationConfig::default();
        let metadata = PackageMetadata {
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            description: "Test WebAssembly model".to_string(),
            author: "SciRS2".to_string(),
            license: "MIT".to_string(),
            platforms: vec!["wasm".to_string()],
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
        };

        // Create compiler
        let compiler = WasmCompiler::new(
            model,
            wasm_config,
            web_config,
            metadata,
            temp_dir.path().to_path_buf(),
        );

        // Test compilation process
        let result = compiler.compile();
        assert!(result.is_ok());

        let compilation_result = result.unwrap();
        assert!(compilation_result.wasm_module.exists());
        assert!(!compilation_result.bindings.is_empty());
        assert!(compilation_result.bundle_info.total_size > 0);
    }

    #[test]
    fn test_memory_manager_integration() {
        // Test memory manager with different configurations
        let performance_manager = MemoryManager::performance_optimized();
        let constrained_manager = MemoryManager::resource_constrained();

        let model_size = 10 * 1024 * 1024; // 10MB model

        let perf_requirements = performance_manager.calculate_memory_requirements(model_size);
        let constrained_requirements =
            constrained_manager.calculate_memory_requirements(model_size);

        // Performance config should use more memory
        assert!(perf_requirements.total > constrained_requirements.total);

        // Both should handle the model size
        assert!(performance_manager.is_suitable_for_model(model_size));
        assert!(constrained_manager.is_suitable_for_model(model_size));
    }

    #[test]
    fn test_binding_generator_integration() {
        let temp_dir = TempDir::new().unwrap();

        // Test different binding configurations
        let js_config = WebBindingConfig {
            target_language: WebBindingLanguage::JavaScript,
            module_system: ModuleSystem::ESModules,
            type_definitions: false,
            documentation: false,
            bundling: BundlingConfig {
                enable: false,
                format: BundleFormat::Single,
                minify: false,
                tree_shaking: false,
                code_splitting: false,
            },
        };

        let ts_config = WebBindingConfig {
            target_language: WebBindingLanguage::TypeScript,
            module_system: ModuleSystem::ESModules,
            type_definitions: true,
            documentation: true,
            bundling: BundlingConfig {
                enable: true,
                format: BundleFormat::Single,
                minify: true,
                tree_shaking: true,
                code_splitting: false,
            },
        };

        let both_config = WebBindingConfig {
            target_language: WebBindingLanguage::Both,
            module_system: ModuleSystem::ESModules,
            type_definitions: true,
            documentation: true,
            bundling: BundlingConfig {
                enable: true,
                format: BundleFormat::Chunked,
                minify: true,
                tree_shaking: true,
                code_splitting: true,
            },
        };

        // Test JavaScript generation
        let js_generator = BindingGenerator::new(temp_dir.path().to_path_buf(), js_config);
        let js_bindings = js_generator.generate_bindings().unwrap();
        assert_eq!(js_bindings.len(), 1);
        assert!(js_bindings[0].to_string_lossy().ends_with(".js"));

        // Test TypeScript generation
        let ts_generator = BindingGenerator::new(temp_dir.path().to_path_buf(), ts_config);
        let ts_bindings = ts_generator.generate_bindings().unwrap();
        assert_eq!(ts_bindings.len(), 1);
        assert!(ts_bindings[0].to_string_lossy().ends_with(".ts"));

        // Test both generation
        let both_generator = BindingGenerator::new(temp_dir.path().to_path_buf(), both_config);
        let both_bindings = both_generator.generate_bindings().unwrap();
        assert_eq!(both_bindings.len(), 2);
    }

    #[test]
    fn test_configuration_defaults() {
        // Test all default configurations are valid
        let wasm_config = WasmCompilationConfig::default();
        assert_eq!(wasm_config.target_version, WasmVersion::SIMD);
        assert!(wasm_config.features.simd);
        assert!(wasm_config.features.bulk_memory);
        assert!(wasm_config.optimization_level.lto);

        let web_config = WebIntegrationConfig::default();
        assert_eq!(
            web_config.bindings.target_language,
            WebBindingLanguage::Both
        );
        assert!(web_config.caching.enable);
        assert!(web_config.progressive_loading.enable);
        assert!(web_config.workers.enable);

        let memory_config = WasmMemoryConfig::default();
        assert_eq!(memory_config.initial_pages, 256);
        assert_eq!(memory_config.maximum_pages, Some(1024));
        assert_eq!(
            memory_config.growth_strategy,
            MemoryGrowthStrategy::OnDemand
        );
    }

    #[test]
    fn test_wasm_features_validation() {
        // Test WASM feature combinations
        let mvp_features = WasmFeatures {
            simd: false,
            threads: false,
            bulk_memory: false,
            reference_types: false,
            exception_handling: false,
            tail_calls: false,
            multi_value: false,
            wasi: false,
        };

        let modern_features = WasmFeatures {
            simd: true,
            threads: true,
            bulk_memory: true,
            reference_types: true,
            exception_handling: false,
            tail_calls: false,
            multi_value: true,
            wasi: false,
        };

        // Test version compatibility
        let mvp_config = WasmCompilationConfig {
            target_version: WasmVersion::MVP,
            features: mvp_features,
            ..Default::default()
        };

        let simd_threads_config = WasmCompilationConfig {
            target_version: WasmVersion::SIMDThreads,
            features: modern_features,
            ..Default::default()
        };

        // Verify configurations make sense
        assert_eq!(mvp_config.target_version, WasmVersion::MVP);
        assert!(!mvp_config.features.simd);

        assert_eq!(simd_threads_config.target_version, WasmVersion::SIMDThreads);
        assert!(simd_threads_config.features.simd);
        assert!(simd_threads_config.features.threads);
    }

    #[test]
    fn test_memory_requirements_calculation() {
        let manager = MemoryManager::performance_optimized();

        // Test different model sizes
        let small_model = 1024 * 1024; // 1MB
        let medium_model = 10 * 1024 * 1024; // 10MB
        let large_model = 100 * 1024 * 1024; // 100MB

        let small_req = manager.calculate_memory_requirements(small_model);
        let medium_req = manager.calculate_memory_requirements(medium_model);
        let large_req = manager.calculate_memory_requirements(large_model);

        // Larger models should require more memory
        assert!(small_req.total < medium_req.total);
        assert!(medium_req.total < large_req.total);

        // All should include model memory
        assert_eq!(small_req.model_memory, small_model);
        assert_eq!(medium_req.model_memory, medium_model);
        assert_eq!(large_req.model_memory, large_model);

        // Check breakdown percentages sum to 100%
        let breakdown = medium_req.breakdown_percentages();
        let total_percent = breakdown.base_percent
            + breakdown.model_percent
            + breakdown.cache_percent
            + breakdown.preload_percent
            + breakdown.worker_percent;
        assert!((total_percent - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_chunk_size_recommendations() {
        let manager = MemoryManager::performance_optimized();

        let small_model = 512 * 1024; // 512KB
        let medium_model = 10 * 1024 * 1024; // 10MB
        let large_model = 200 * 1024 * 1024; // 200MB

        let small_chunk = manager.recommended_chunk_size(small_model);
        let medium_chunk = manager.recommended_chunk_size(medium_model);
        let large_chunk = manager.recommended_chunk_size(large_model);

        // Verify chunk size adaptation
        assert!(small_chunk < medium_chunk);
        assert!(medium_chunk <= large_chunk);

        // Base chunk size should be used for medium models
        assert_eq!(medium_chunk, manager.progressive_config().chunk_size);
    }
}
