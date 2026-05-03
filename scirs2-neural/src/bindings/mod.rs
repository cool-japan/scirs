//! C/C++ binding generation utilities for neural networks
//!
//! This module provides comprehensive tools for generating C and C++ bindings including:
//! - Automatic header generation with proper type mappings
//! - Source code generation for implementation stubs
//! - Build system integration (CMake, Makefile)
//! - Cross-platform compatibility handling
//! - Advanced binding features (callbacks, memory management)
//! # Module Organization
//! - [`config`] - Configuration types and structures for binding generation
//! - [`generator`] - Core generator logic and orchestration
//! - [`header_generation`] - Header file generation with type definitions and API declarations
//! - [`source_generation`] - Source file generation and C++ wrapper implementation
//! - [`build_system`] - Build system file generation (CMake, Makefile, pkg-config)
//! - [`examples_docs`] - Examples and documentation generation
//! # Basic Usage
//! ```rust
//! use scirs2_neural::bindings::{BindingConfig, BindingLanguage};
//! use scirs2_neural::models::Sequential;
//! // Create a binding configuration
//! let mut config = BindingConfig::default();
//! config.language = BindingLanguage::Cpp;
//! config.library_name = "my_neural_lib".to_string();
//! // Create a model (simplified for example)
//! let model = Sequential::<f32>::new();
//! println!("Binding config created for {}", config.library_name);
//! println!("Model has {} layers", model.num_layers());
//! ```

pub mod build_system;
pub mod config;
pub mod examples_docs;
pub mod generator;
pub mod header_generation;
pub mod source_generation;
// Re-export main types and functions for backward compatibility
pub use config::{
    ApiStyle, ArrayMapping, BindingConfig, BindingLanguage, BindingResult, BuildSystem,
    BuildSystemConfig, CustomType, Dependency, ErrorHandling, InstallConfig, MemoryStrategy,
    StringMapping, SyncPrimitive, ThreadPoolConfig, ThreadSafety, ThreadingConfig, TypeMappings,
};
pub use generator::BindingGenerator;
pub use build_system::BuildSystemGenerator;
pub use examples_docs::ExamplesDocsGenerator;
pub use header_generation::HeaderGenerator;
pub use source_generation::SourceGenerator;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::models::sequential::Sequential;
    use crate::serving::PackageMetadata;
    use scirs2_core::random::SeedableRng;
    use tempfile::TempDir;

    #[test]
    fn test_binding_module_integration() {
        let temp_dir = TempDir::new().expect("TempDir::new failed");
        let config = BindingConfig::default();
        let metadata = PackageMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "Test model".to_string(),
            license: "Apache-2.0".to_string(),
            platforms: vec!["linux-x86_64".to_string()],
            dependencies: std::collections::HashMap::new(),
            input_specs: vec![],
            output_specs: vec![],
            runtime_requirements: crate::serving::RuntimeRequirements {
                min_memory_mb: 256,
                cpu_requirements: crate::serving::CpuRequirements {
                    min_cores: 1,
                    instruction_sets: vec![],
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: vec![],
            },
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            checksum: "test_checksum".to_string(),
        };
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).expect("Dense::new failed"));
        model.add_layer(Dense::new(5, 2, None, &mut rng).expect("Dense::new failed"));
        let generator =
            BindingGenerator::new(model, config, metadata, temp_dir.path().to_path_buf());
        let result = generator.create_directory_structure();
        assert!(result.is_ok());
        assert!(temp_dir.path().join("include").exists());
        assert!(temp_dir.path().join("src").exists());
        assert!(temp_dir.path().join("examples").exists());
        assert!(temp_dir.path().join("docs").exists());
        assert!(temp_dir.path().join("build").exists());
    }

    #[test]
    fn test_config_variations() {
        let mut config = BindingConfig {
            language: BindingLanguage::Cpp,
            api_style: ApiStyle::ObjectOriented,
            ..Default::default()
        };
        assert_eq!(config.language, BindingLanguage::Cpp);
        assert_eq!(config.api_style, ApiStyle::ObjectOriented);
        config.api_style = ApiStyle::Hybrid;
        assert_eq!(config.api_style, ApiStyle::Hybrid);
        let custom_type = CustomType {
            rust_name: "MyStruct".to_string(),
            c_name: "my_struct_t".to_string(),
            definition: "typedef struct { int x; float y; } my_struct_t;".to_string(),
            includes: vec!["stdint.h".to_string()],
        };
        config.type_mappings.custom_types.push(custom_type);
        assert_eq!(config.type_mappings.custom_types.len(), 1);
    }

    #[test]
    fn test_individual_generators() {
        let temp_dir = TempDir::new().expect("TempDir::new failed");
        let output_dir = temp_dir.path().to_path_buf();
        std::fs::create_dir_all(output_dir.join("include")).expect("mkdir failed");
        std::fs::create_dir_all(output_dir.join("src")).expect("mkdir failed");
        std::fs::create_dir_all(output_dir.join("examples")).expect("mkdir failed");
        let config = BindingConfig::default();
        let header_gen = HeaderGenerator::new(&config, &output_dir);
        assert!(header_gen.generate().is_ok());
        let source_gen = SourceGenerator::new(&config, &output_dir);
        assert!(source_gen.generate().is_ok());
        let build_gen = BuildSystemGenerator::new(&config, &output_dir);
        assert!(build_gen.generate().is_ok());
        let examples_gen = ExamplesDocsGenerator::new(&config, &output_dir);
        assert!(examples_gen.generate().is_ok());
    }
}
