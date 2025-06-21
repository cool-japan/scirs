//! Core binding generator logic and orchestration
//!
//! This module contains the main BindingGenerator struct that orchestrates
//! the entire C/C++ binding generation process.

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::serving::PackageMetadata;
use num_traits::Float;
use std::fmt::Debug;
use std::fs;
use std::path::PathBuf;

use super::build_system::BuildSystemGenerator;
use super::config::{BindingConfig, BindingResult};
use super::examples_docs::ExamplesDocsGenerator;
use super::header_generation::HeaderGenerator;
use super::source_generation::SourceGenerator;

/// C/C++ binding generator
pub struct BindingGenerator<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model to generate bindings for
    #[allow(dead_code)]
    model: Sequential<F>,
    /// Binding configuration
    config: BindingConfig,
    /// Package metadata
    #[allow(dead_code)]
    metadata: PackageMetadata,
    /// Output directory
    output_dir: PathBuf,
}

impl<
        F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync,
    > BindingGenerator<F>
{
    /// Create a new binding generator
    pub fn new(
        model: Sequential<F>,
        config: BindingConfig,
        metadata: PackageMetadata,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            model,
            config,
            metadata,
            output_dir,
        }
    }

    /// Generate complete bindings
    pub fn generate(&self) -> Result<BindingResult> {
        // Create output directory structure
        self.create_directory_structure()?;

        let mut result = BindingResult {
            headers: Vec::new(),
            sources: Vec::new(),
            build_files: Vec::new(),
            examples: Vec::new(),
            documentation: Vec::new(),
        };

        // Generate header files
        let header_generator = HeaderGenerator::new(&self.config, &self.output_dir);
        let headers = header_generator.generate()?;
        result.headers.extend(headers);

        // Generate source files
        let source_generator = SourceGenerator::new(&self.config, &self.output_dir);
        let sources = source_generator.generate()?;
        result.sources.extend(sources);

        // Generate build system files
        let build_generator = BuildSystemGenerator::new(&self.config, &self.output_dir);
        let build_files = build_generator.generate()?;
        result.build_files.extend(build_files);

        // Generate examples and documentation
        let examples_docs_generator = ExamplesDocsGenerator::new(&self.config, &self.output_dir);
        let (examples, docs) = examples_docs_generator.generate()?;
        result.examples.extend(examples);
        result.documentation.extend(docs);

        Ok(result)
    }

    /// Create the directory structure for binding output
    pub fn create_directory_structure(&self) -> Result<()> {
        let dirs = vec!["include", "src", "examples", "docs", "build"];
        for dir in dirs {
            let path = self.output_dir.join(dir);
            fs::create_dir_all(&path).map_err(|e| {
                NeuralError::IOError(format!(
                    "Failed to create directory {}: {}",
                    path.display(),
                    e
                ))
            })?;
        }
        Ok(())
    }

    /// Get the binding configuration
    pub fn config(&self) -> &BindingConfig {
        &self.config
    }

    /// Get the output directory
    pub fn output_dir(&self) -> &PathBuf {
        &self.output_dir
    }

    /// Get the package metadata
    pub fn metadata(&self) -> &PackageMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use rand::SeedableRng;
    use tempfile::TempDir;

    #[test]
    fn test_binding_generator_creation() {
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

        assert_eq!(generator.config().library_name, "scirs2_model");
        assert_eq!(generator.output_dir(), &temp_dir.path().to_path_buf());
    }

    #[test]
    fn test_directory_structure_creation() {
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

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let generator =
            BindingGenerator::new(model, config, metadata, temp_dir.path().to_path_buf());

        let result = generator.create_directory_structure();
        assert!(result.is_ok());

        // Verify directories were created
        assert!(temp_dir.path().join("include").exists());
        assert!(temp_dir.path().join("src").exists());
        assert!(temp_dir.path().join("examples").exists());
        assert!(temp_dir.path().join("docs").exists());
        assert!(temp_dir.path().join("build").exists());
    }
}
