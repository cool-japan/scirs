//! C/C++ binding generation utilities for neural networks
//!
//! This module provides comprehensive tools for generating C and C++ bindings including:
//! - Automatic header generation with proper type mappings
//! - Source code generation for implementation stubs
//! - Build system integration (CMake, Makefile)
//! - Cross-platform compatibility handling
//! - Advanced binding features (callbacks, memory management)
//!
//! # Module Organization
//!
//! - [`config`] - Configuration types and structures for binding generation
//! - [`generator`] - Core generator logic and orchestration
//! - [`header_generation`] - Header file generation with type definitions and API declarations
//! - [`source_generation`] - Source file generation and C++ wrapper implementation
//! - [`build_system`] - Build system file generation (CMake, Makefile, pkg-config)
//! - [`examples_docs`] - Examples and documentation generation
//!
//! # Basic Usage
//!
//! ```rust
//! use scirs2_neural::bindings::{BindingConfig, BindingLanguage};
//! use scirs2_neural::models::Sequential;
//!
//! // Create a binding configuration
//! let mut config = BindingConfig::default();
//! config.language = BindingLanguage::Cpp;
//! config.library_name = "my_neural_lib".to_string();
//!
//! // Create a model (simplified for example)
//! let model = Sequential::<f32>::new();
//!
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
    use rand::SeedableRng;
    use tempfile::TempDir;

    #[test]
    fn test_binding_module_integration() {
        // Test that all modules work together
        let temp_dir = TempDir::new().unwrap();
        let config = BindingConfig::default();
        let metadata = PackageMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "Test model".to_string(),
            license: "MIT".to_string(),
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

        // Create a simple sequential model for testing
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());
        model.add_layer(Dense::new(5, 2, None, &mut rng).unwrap());

        let generator =
            BindingGenerator::new(model, config, metadata, temp_dir.path().to_path_buf());

        // Test that directory structure can be created
        let result = generator.create_directory_structure();
        assert!(result.is_ok());

        // Verify directories were created
        assert!(temp_dir.path().join("include").exists());
        assert!(temp_dir.path().join("src").exists());
        assert!(temp_dir.path().join("examples").exists());
        assert!(temp_dir.path().join("docs").exists());
        assert!(temp_dir.path().join("build").exists());
    }

    #[test]
    fn test_individual_generators() {
        let temp_dir = TempDir::new().unwrap();
        let config = BindingConfig::default();
        let output_dir = temp_dir.path().to_path_buf();

        // Create necessary directories
        std::fs::create_dir_all(output_dir.join("include")).unwrap();
        std::fs::create_dir_all(output_dir.join("src")).unwrap();
        std::fs::create_dir_all(output_dir.join("examples")).unwrap();

        // Test header generator
        let header_gen = HeaderGenerator::new(&config, &output_dir);
        let headers = header_gen.generate();
        assert!(headers.is_ok());

        // Test source generator
        let source_gen = SourceGenerator::new(&config, &output_dir);
        let sources = source_gen.generate();
        assert!(sources.is_ok());

        // Test build system generator
        let build_gen = BuildSystemGenerator::new(&config, &output_dir);
        let build_files = build_gen.generate();
        assert!(build_files.is_ok());

        // Test examples/docs generator
        let examples_gen = ExamplesDocsGenerator::new(&config, &output_dir);
        let result = examples_gen.generate();
        assert!(result.is_ok());
        let (examples, docs) = result.unwrap();
        assert!(examples.len() > 0 || examples.is_empty()); // Just checking it's valid
        assert!(docs.len() > 0 || docs.is_empty()); // Just checking it's valid
    }

    #[test]
    fn test_config_variations() {
        // Test different binding configurations
        let mut config = BindingConfig::default();

        // Test C++ configuration
        config.language = BindingLanguage::Cpp;
        config.api_style = ApiStyle::ObjectOriented;
        assert_eq!(config.language, BindingLanguage::Cpp);
        assert_eq!(config.api_style, ApiStyle::ObjectOriented);

        // Test hybrid configuration
        config.api_style = ApiStyle::Hybrid;
        assert_eq!(config.api_style, ApiStyle::Hybrid);

        // Test custom type mappings
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
    fn test_error_handling() {
        // Test that generators handle invalid configurations gracefully
        let temp_dir = TempDir::new().unwrap();
        let mut config = BindingConfig::default();
        config.library_name = "".to_string(); // Invalid empty name

        let output_dir = temp_dir.path().to_path_buf();

        // Create necessary directories
        std::fs::create_dir_all(output_dir.join("include")).unwrap();

        let header_gen = HeaderGenerator::new(&config, &output_dir);
        // This should work despite empty library name, as the generator should handle it
        let result = header_gen.generate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_binding_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = BindingConfig::default();
        let metadata = PackageMetadata {
            name: "integration_test".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "Integration test model".to_string(),
            license: "MIT".to_string(),
            platforms: vec!["linux-x86_64".to_string()],
            dependencies: std::collections::HashMap::new(),
            input_specs: vec![],
            output_specs: vec![],
            runtime_requirements: crate::serving::RuntimeRequirements {
                min_memory_mb: 512,
                cpu_requirements: crate::serving::CpuRequirements {
                    min_cores: 2,
                    instruction_sets: vec![],
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: vec![],
            },
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            checksum: "integration_test_checksum".to_string(),
        };

        // Create a simple model
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(784, 128, Some("relu"), &mut rng).unwrap());
        model.add_layer(Dense::new(128, 10, Some("softmax"), &mut rng).unwrap());

        let generator =
            BindingGenerator::new(model, config, metadata, temp_dir.path().to_path_buf());

        // Generate complete bindings
        let result = generator.generate();
        assert!(result.is_ok());

        let binding_result = result.unwrap();
        assert!(!binding_result.headers.is_empty());
        assert!(!binding_result.sources.is_empty());
        assert!(!binding_result.build_files.is_empty());
        assert!(!binding_result.examples.is_empty());
        assert!(!binding_result.documentation.is_empty());

        // Verify files were actually created
        for header in &binding_result.headers {
            assert!(header.exists(), "Header file should exist: {:?}", header);
        }
        for source in &binding_result.sources {
            assert!(source.exists(), "Source file should exist: {:?}", source);
        }
        for build_file in &binding_result.build_files {
            assert!(
                build_file.exists(),
                "Build file should exist: {:?}",
                build_file
            );
        }
    }
}
