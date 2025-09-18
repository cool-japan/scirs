//! Advanced Plugin Template Generator
//!
//! This module provides comprehensive template generation capabilities for creating
//! sophisticated optimizer plugins with advanced features, testing, documentation,
//! and CI/CD integration.

use super::core::*;
use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Advanced template generator for plugin development
#[derive(Debug)]
pub struct AdvancedTemplateGenerator {
    /// Template configuration
    config: TemplateGeneratorConfig,
    /// Template registry
    template_registry: TemplateRegistry,
    /// Code generators
    generators: HashMap<GeneratorType, Box<dyn CodeGenerator>>,
    /// Validation engine
    validator: TemplateValidator,
}

/// Template generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateGeneratorConfig {
    /// Target Rust version
    pub rust_version: String,
    /// Include CI/CD templates
    pub include_cicd: bool,
    /// Include documentation templates
    pub include_docs: bool,
    /// Include benchmark templates
    pub include_benchmarks: bool,
    /// Include example templates
    pub include_examples: bool,
    /// Include GPU support templates
    pub include_gpu: bool,
    /// Include distributed training templates
    pub include_distributed: bool,
    /// Code style preferences
    pub code_style: CodeStyle,
    /// License type
    pub license: LicenseType,
    /// Testing framework
    pub testing_framework: TestingFramework,
}

/// Code style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStyle {
    /// Indentation type
    pub indentation: IndentationType,
    /// Maximum line length
    pub max_line_length: usize,
    /// Use trailing commas
    pub trailing_commas: bool,
    /// Import organization style
    pub import_style: ImportStyle,
    /// Documentation style
    pub doc_style: DocStyle,
}

/// Indentation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndentationType {
    Spaces(usize),
    Tabs,
}

/// Import organization style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportStyle {
    Grouped,
    Flat,
    SeparateStd,
}

/// Documentation style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocStyle {
    Brief,
    Comprehensive,
    Academic,
}

/// License types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LicenseType {
    MIT,
    Apache2,
    GPL3,
    BSD3Clause,
    Custom(String),
}

/// Testing framework options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestingFramework {
    Standard,
    Proptest,
    Quickcheck,
    Criterion,
}

/// Template registry for managing plugin templates
#[derive(Debug)]
pub struct TemplateRegistry {
    /// Built-in templates
    builtin_templates: HashMap<String, PluginTemplate>,
    /// Custom templates
    custom_templates: HashMap<String, PluginTemplate>,
    /// Template metadata
    template_metadata: HashMap<String, TemplateMetadata>,
}

/// Plugin template with enhanced capabilities
#[derive(Debug, Clone)]
pub struct PluginTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: TemplateCategory,
    /// Template complexity level
    pub complexity: ComplexityLevel,
    /// Template structure
    pub structure: EnhancedTemplateStructure,
    /// Required features
    pub required_features: Vec<String>,
    /// Template parameters
    pub parameters: HashMap<String, TemplateParameter>,
}

/// Template categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateCategory {
    BasicOptimizer,
    AdaptiveOptimizer,
    SecondOrderOptimizer,
    MetaLearningOptimizer,
    DistributedOptimizer,
    GPUOptimizer,
    SpecializedOptimizer,
    Utility,
}

/// Complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Enhanced template structure
#[derive(Debug, Clone)]
pub struct EnhancedTemplateStructure {
    /// Core source files
    pub core_files: Vec<TemplateFile>,
    /// Test files
    pub test_files: Vec<TemplateFile>,
    /// Documentation files
    pub documentation: Vec<TemplateFile>,
    /// Configuration files
    pub config_files: Vec<TemplateFile>,
    /// CI/CD files
    pub cicd_files: Vec<TemplateFile>,
    /// Example files
    pub example_files: Vec<TemplateFile>,
    /// Benchmark files
    pub benchmark_files: Vec<TemplateFile>,
    /// Resource files
    pub resource_files: Vec<TemplateFile>,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template version
    pub version: String,
    /// Author information
    pub author: String,
    /// Creation date
    pub created_at: String,
    /// Last modified
    pub modified_at: String,
    /// Usage count
    pub usage_count: usize,
    /// User ratings
    pub ratings: Vec<f64>,
    /// Tags
    pub tags: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Template parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default_value: Option<String>,
    /// Description
    pub description: String,
    /// Required parameter
    pub required: bool,
    /// Validation rules
    pub validation: Vec<ParameterValidation>,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Choice(Vec<String>),
    Array(Box<ParameterType>),
}

/// Parameter validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValidation {
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Range(f64, f64),
    Custom(String),
}

/// Code generator types
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum GeneratorType {
    OptimizerCore,
    TestSuite,
    Documentation,
    Benchmarks,
    Examples,
    CICD,
    Configuration,
}

/// Code generator trait
pub trait CodeGenerator: std::fmt::Debug {
    /// Generate code based on template and parameters
    fn generate(
        &self,
        template: &PluginTemplate,
        parameters: &HashMap<String, String>,
        config: &TemplateGeneratorConfig,
    ) -> Result<Vec<GeneratedFile>>;
    
    /// Get generator name
    fn name(&self) -> &str;
    
    /// Get supported template categories
    fn supported_categories(&self) -> Vec<TemplateCategory>;
}

/// Generated file
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    /// File path relative to project root
    pub path: PathBuf,
    /// File content
    pub content: String,
    /// File permissions (Unix-style)
    pub permissions: Option<u32>,
    /// File type
    pub file_type: GeneratedFileType,
}

/// Generated file types
#[derive(Debug, Clone)]
pub enum GeneratedFileType {
    RustSource,
    CargoToml,
    Documentation,
    Configuration,
    Script,
    Data,
}

/// Template validator
#[derive(Debug)]
pub struct TemplateValidator {
    /// Validation rules
    rules: Vec<Box<dyn TemplateValidationRule>>,
    /// Syntax checker
    syntax_checker: SyntaxChecker,
}

/// Template validation rule
pub trait TemplateValidationRule: std::fmt::Debug {
    /// Validate template
    fn validate(&self, template: &PluginTemplate) -> ValidationResult;
    
    /// Get rule name
    fn name(&self) -> &str;
    
    /// Get rule description
    fn description(&self) -> &str;
}

/// Syntax checker for generated code
#[derive(Debug)]
pub struct SyntaxChecker {
    /// Rust syntax validator
    rust_validator: RustSyntaxValidator,
    /// TOML validator
    toml_validator: TomlValidator,
    /// Markdown validator
    markdown_validator: MarkdownValidator,
}

/// Rust syntax validator
#[derive(Debug)]
pub struct RustSyntaxValidator;

/// TOML validator
#[derive(Debug)]
pub struct TomlValidator;

/// Markdown validator
#[derive(Debug)]
pub struct MarkdownValidator;

impl AdvancedTemplateGenerator {
    /// Create a new advanced template generator
    pub fn new(config: TemplateGeneratorConfig) -> Self {
        let mut template_registry = TemplateRegistry::new();
        template_registry.register_builtin_templates();
        
        let mut generators = HashMap::new();
        generators.insert(GeneratorType::OptimizerCore, Box::new(OptimizerCoreGenerator) as Box<dyn CodeGenerator>);
        generators.insert(GeneratorType::TestSuite, Box::new(TestSuiteGenerator) as Box<dyn CodeGenerator>);
        generators.insert(GeneratorType::Documentation, Box::new(DocumentationGenerator) as Box<dyn CodeGenerator>);
        generators.insert(GeneratorType::Benchmarks, Box::new(BenchmarkGenerator) as Box<dyn CodeGenerator>);
        generators.insert(GeneratorType::Examples, Box::new(ExampleGenerator) as Box<dyn CodeGenerator>);
        generators.insert(GeneratorType::CICD, Box::new(CICDGenerator) as Box<dyn CodeGenerator>);
        generators.insert(GeneratorType::Configuration, Box::new(ConfigurationGenerator) as Box<dyn CodeGenerator>);
        
        let validator = TemplateValidator::new();
        
        Self {
            config,
            template_registry,
            generators,
            validator,
        }
    }
    
    /// Generate a complete plugin project
    pub fn generate_plugin_project(
        &self,
        template_name: &str,
        project_name: &str,
        parameters: HashMap<String, String>,
        output_dir: &Path,
    ) -> Result<ProjectGenerationResult> {
        // Get template
        let template = self.template_registry.get_template(template_name)
            .ok_or_else(|| OptimError::InvalidConfig(
                format!("Template '{}' not found", template_name)
            ))?;
        
        // Validate template
        let validation_result = self.validator.validate_template(template)?;
        if !validation_result.is_valid {
            return Err(OptimError::InvalidConfig(
                format!("Template validation failed: {:?}", validation_result.errors)
            ));
        }
        
        // Validate parameters
        self.validate_parameters(template, &parameters)?;
        
        // Generate files
        let mut all_files = Vec::new();
        let mut generation_stats = GenerationStats::new();
        
        for (generator_type, generator) in &self.generators {
            if self.should_use_generator(generator_type) {
                let generated_files = generator.generate(template, &parameters, &self.config)?;
                generation_stats.update_for_generator(generator_type, generated_files.len());
                all_files.extend(generated_files);
            }
        }
        
        // Create project structure
        std::fs::create_dir_all(output_dir)?;
        
        // Write files
        let mut written_files = Vec::new();
        for file in all_files {
            let file_path = output_dir.join(&file.path);
            
            // Create parent directories
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            
            // Write file content
            std::fs::write(&file_path, &file.content)?;
            
            // Set permissions if specified
            #[cfg(unix)]
            if let Some(permissions) = file.permissions {
                use std::os::unix::fs::PermissionsExt;
                let perms = std::fs::Permissions::from_mode(permissions);
                std::fs::set_permissions(&file_path, perms)?;
            }
            
            written_files.push(file_path);
        }
        
        // Generate project metadata
        let project_metadata = ProjectMetadata {
            _name: project_name.to_string(),
            template_name: template_name.to_string(),
            template_version: template.version().unwrap_or("unknown".to_string()),
            generated_at: chrono::Utc::now().to_rfc3339(),
            parameters: parameters.clone(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        // Write metadata file
        let metadata_path = output_dir.join(".scirs2_plugin_metadata.json");
        let metadata_json = serde_json::to_string_pretty(&project_metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        
        Ok(ProjectGenerationResult {
            project_metadata,
            generated_files: written_files,
            generation_stats,
            validation_result,
        })
    }
    
    /// List available templates
    pub fn list_templates(&self) -> Vec<TemplateInfo> {
        self.template_registry.list_templates()
    }
    
    /// Get template details
    pub fn get_template_details(&self, templatename: &str) -> Option<TemplateDetails> {
        self.template_registry.get_template_details(template_name)
    }
    
    /// Add custom template
    pub fn add_custom_template(&mut self, template: PluginTemplate) -> Result<()> {
        // Validate template first
        let validation_result = self.validator.validate_template(&template)?;
        if !validation_result.is_valid {
            return Err(OptimError::InvalidConfig(
                format!("Template validation failed: {:?}", validation_result.errors)
            ));
        }
        
        self.template_registry.add_custom_template(template)?;
        Ok(())
    }
    
    /// Generate template from existing plugin
    pub fn generate_template_from_plugin(&self, plugin_dir: &Path, templatename: &str) -> Result<PluginTemplate> {
        // Analyze existing plugin structure
        let analyzer = PluginAnalyzer::new();
        let analysis = analyzer.analyze_plugin(plugin_dir)?;
        
        // Extract template from analysis
        let template_extractor = TemplateExtractor::new();
        let template = template_extractor.extract_template(analysis, template_name)?;
        
        Ok(template)
    }
    
    /// Validate parameters against template
    fn validate_parameters(&self, template: &PluginTemplate, parameters: &HashMap<String, String>) -> Result<()> {
        for (param_name, param_def) in &template.parameters {
            if param_def.required && !parameters.contains_key(param_name) {
                return Err(OptimError::InvalidConfig(
                    format!("Required parameter '{}' is missing", param_name)
                ));
            }
            
            if let Some(value) = parameters.get(param_name) {
                self.validate_parameter_value(param_def, value)?;
            }
        }
        
        Ok(())
    }
    
    /// Validate individual parameter value
    fn validate_parameter_value(&self, paramdef: &TemplateParameter, value: &str) -> Result<()> {
        for validation in &param_def.validation {
            match validation {
                ParameterValidation::MinLength(min_len) => {
                    if value.len() < *min_len {
                        return Err(OptimError::InvalidConfig(
                            format!("Parameter '{}' is too short (minimum: {})", param_def.name, min_len)
                        ));
                    }
                }
                ParameterValidation::MaxLength(max_len) => {
                    if value.len() > *max_len {
                        return Err(OptimError::InvalidConfig(
                            format!("Parameter '{}' is too long (maximum: {})", param_def.name, max_len)
                        ));
                    }
                }
                ParameterValidation::Pattern(pattern) => {
                    let regex = regex::Regex::new(pattern).map_err(|e| OptimError::InvalidConfig(
                        format!("Invalid regex pattern: {}", e)
                    ))?;
                    if !regex.is_match(value) {
                        return Err(OptimError::InvalidConfig(
                            format!("Parameter '{}' doesn't match pattern '{}'", param_def.name, pattern)
                        ));
                    }
                }
                ParameterValidation::Range(min, max) => {
                    let numeric_value: f64 = value.parse().map_err(|_| OptimError::InvalidConfig(
                        format!("Parameter '{}' must be a number", param_def.name)
                    ))?;
                    if numeric_value < *min || numeric_value > *max {
                        return Err(OptimError::InvalidConfig(
                            format!("Parameter '{}' must be between {} and {}", param_def.name, min, max)
                        ));
                    }
                }
                ParameterValidation::Custom(_) => {
                    // Custom validation would be implemented based on specific needs
                    continue;
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if generator should be used based on configuration
    fn should_use_generator(&self, generatortype: &GeneratorType) -> bool {
        match generator_type {
            GeneratorType::OptimizerCore => true, // Always include core
            GeneratorType::TestSuite => true, // Always include tests
            GeneratorType::Documentation => self.config.include_docs,
            GeneratorType::Benchmarks => self.config.include_benchmarks,
            GeneratorType::Examples => self.config.include_examples,
            GeneratorType::CICD => self.config.include_cicd,
            GeneratorType::Configuration => true, // Always include config
        }
    }
}

/// Project generation result
#[derive(Debug)]
pub struct ProjectGenerationResult {
    /// Project metadata
    pub project_metadata: ProjectMetadata,
    /// List of generated files
    pub generated_files: Vec<PathBuf>,
    /// Generation statistics
    pub generation_stats: GenerationStats,
    /// Template validation result
    pub validation_result: PluginValidationResult,
}

/// Project metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectMetadata {
    /// Project name
    pub name: String,
    /// Template used
    pub template_name: String,
    /// Template version
    pub template_version: String,
    /// Generation timestamp
    pub generated_at: String,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Generator version
    pub generator_version: String,
}

/// Generation statistics
#[derive(Debug)]
pub struct GenerationStats {
    /// Files generated by each generator
    pub files_by_generator: HashMap<GeneratorType, usize>,
    /// Total files generated
    pub total_files: usize,
    /// Generation time
    pub generation_time: std::time::Duration,
    /// Total lines of code generated
    pub total_loc: usize,
}

impl GenerationStats {
    fn new() -> Self {
        Self {
            files_by_generator: HashMap::new(),
            total_files: 0,
            generation_time: std::time::Duration::default(),
            total_loc: 0,
        }
    }
    
    fn update_for_generator(&mut self, generator_type: &GeneratorType, filecount: usize) {
        self.files_by_generator.insert(generator_type.clone(), file_count);
        self.total_files += file_count;
    }
}

/// Template information for listing
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: TemplateCategory,
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Required features
    pub required_features: Vec<String>,
    /// Metadata
    pub metadata: TemplateMetadata,
}

/// Detailed template information
#[derive(Debug, Clone)]
pub struct TemplateDetails {
    /// Basic template info
    pub info: TemplateInfo,
    /// Template structure
    pub structure: EnhancedTemplateStructure,
    /// Parameters
    pub parameters: HashMap<String, TemplateParameter>,
    /// Usage examples
    pub examples: Vec<TemplateUsageExample>,
}

/// Template usage example
#[derive(Debug, Clone)]
pub struct TemplateUsageExample {
    /// Example name
    pub name: String,
    /// Example description
    pub description: String,
    /// Parameter values
    pub parameters: HashMap<String, String>,
    /// Expected outcome
    pub expected_outcome: String,
}

/// Concrete code generators

/// Optimizer core generator
#[derive(Debug)]
struct OptimizerCoreGenerator;

impl CodeGenerator for OptimizerCoreGenerator {
    fn generate(
        &self,
        template: &PluginTemplate,
        parameters: &HashMap<String, String>,
        config: &TemplateGeneratorConfig,
    ) -> Result<Vec<GeneratedFile>> {
        let optimizer_name = parameters.get("name").unwrap_or(&template.name);
        let learning_rate_param = parameters.get("learning_rate").unwrap_or("0.001");
        
        let content = self.generate_optimizer_code(optimizer_name, learning_rate_param, config)?;
        
        Ok(vec![GeneratedFile {
            path: PathBuf::from("src/lib.rs"),
            content,
            permissions: None,
            file_type: GeneratedFileType::RustSource,
        }])
    }
    
    fn name(&self) -> &str {
        "OptimizerCore"
    }
    
    fn supported_categories(&self) -> Vec<TemplateCategory> {
        vec![
            TemplateCategory::BasicOptimizer,
            TemplateCategory::AdaptiveOptimizer,
            TemplateCategory::SecondOrderOptimizer,
        ]
    }
}

impl OptimizerCoreGenerator {
    fn generate_optimizer_code(&self, name: &str, learningrate: &str, config: &TemplateGeneratorConfig) -> Result<String> {
        let content = format!(
            r#"//! {} Optimizer Plugin
//!
//! This plugin implements the {} optimization algorithm.
//! Generated with scirs2-optim template generator v{}

use scirs2_optim::plugin::*;
use ndarray::{{Array1, Array2}};
use num_traits::Float;
use serde::{{Serialize, Deserialize}};

/// {} optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {}Config<T: Float> {{
    /// Learning _rate
    pub learning_rate: T,
    /// Add additional parameters here
}}

impl<T: Float> Default for {}Config<T> {{
    fn default() -> Self {{
        Self {{
            learning_rate: num_traits::cast::cast({}).unwrap_or_else(|| T::zero()),
        }}
    }}
}}

/// {} optimizer implementation
#[derive(Debug)]
pub struct {}<T: Float> {{
    /// Configuration
    config: {}Config<T>,
    /// Internal state
    state: {}State<T>,
    /// Plugin metadata
    info: PluginInfo,
    /// Capabilities
    capabilities: PluginCapabilities,
}}

/// Optimizer internal state
#[derive(Debug, Clone)]
struct {}State<T: Float> {{
    /// Step count
    step_count: usize,
    /// Previous gradients (for momentum, etc.)
    prev_gradients: Option<Array1<T>>,
    // Add state variables as needed
}}

impl<T: Float> Default for {}State<T> {{
    fn default() -> Self {{
        Self {{
            step_count: 0,
            prev_gradients: None,
        }}
    }}
}}

impl<T: Float> {}<T> {{
    /// Create a new {} optimizer
    pub fn new(config: {}Config<T>) -> Self {{
        let info = create_plugin_info("{}", env!("CARGO_PKG_VERSION"), "Auto-generated");
        let mut capabilities = create_basic_capabilities();
        capabilities.momentum = true; // Enable if using momentum
        capabilities.state_serialization = true;
        
        Self {{
            config,
            state: {}State::default(),
            info,
            capabilities,
        }}
    }}
    
    /// Update learning _rate
    pub fn set_learning_rate(&mut self, learningrate: T) {{
        self.config.learning_rate = learning_rate;
    }}
    
    /// Get current learning _rate
    pub fn learning_rate(&self) -> T {{
        self.config.learning_rate
    }}
}}

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> OptimizerPlugin<T> for {}<T> {{
    fn step(&mut self, params: &Array1<T>, gradients: &Array1<T>) -> Result<Array1<T>> {{
        // Implement your optimization step here
        // This is a basic gradient descent implementation
        let updated_params = params - &(gradients * self.config.learning_rate);
        
        // Update state
        self.state.step_count += 1;
        self.state.prev_gradients = Some(gradients.clone());
        
        Ok(updated_params)
    }}
    
    fn name(&self) -> &str {{
        "{}"
    }}
    
    fn version(&self) -> &str {{
        env!("CARGO_PKG_VERSION")
    }}
    
    fn plugin_info(&self) -> PluginInfo {{
        self.info.clone()
    }}
    
    fn capabilities(&self) -> PluginCapabilities {{
        self.capabilities.clone()
    }}
    
    fn initialize(&mut self, paramshape: &[usize]) -> Result<()> {{
        // Initialize state based on parameter shape
        Ok(())
    }}
    
    fn reset(&mut self) -> Result<()> {{
        self.state = {}State::default();
        Ok(())
    }}
    
    fn get_config(&self) -> OptimizerConfig {{
        OptimizerConfig {{
            learning_rate: self.config.learning_rate.to_f64().unwrap_or(0.001),
            weight_decay: 0.0,
            momentum: 0.0,
            gradient_clip: None,
            custom_params: std::collections::HashMap::new(),
        }}
    }}
    
    fn set_config(&mut self, config: OptimizerConfig) -> Result<()> {{
        self.config.learning_rate = T::from(config.learning_rate)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid learning _rate".to_string()))?;
        Ok(())
    }}
    
    fn get_state(&self) -> Result<OptimizerState> {{
        let mut state = OptimizerState::default();
        state.step_count = self.state.step_count;
        
        if let Some(ref prev_grad) = self.state.prev_gradients {{
            let grad_vec: Vec<f64> = prev_grad.iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect();
            state.state_vectors.insert("prev_gradients".to_string(), grad_vec);
        }}
        
        Ok(state)
    }}
    
    fn set_state(&mut self, state: OptimizerState) -> Result<()> {{
        self.state.step_count = state.step_count;
        
        if let Some(grad_vec) = state.state_vectors.get("prev_gradients") {{
            let grad_array: Array1<T> = grad_vec.iter()
                .map(|&x| T::from(x).unwrap_or_else(|| T::zero()))
                .collect();
            self.state.prev_gradients = Some(grad_array);
        }}
        
        Ok(())
    }}
    
    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<T>> {{
        Box::new(Self::new(self.config.clone()))
    }}
}}

/// Plugin factory for creating {} instances
#[derive(Debug)]
pub struct {}Factory;

impl<T: Float + std::fmt::Debug + Send + Sync + 'static> OptimizerPluginFactory<T> for {}Factory {{
    fn create_optimizer(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<T>>> {{
        let plugin_config = {}Config {{
            learning_rate: T::from(config.learning_rate)
                .ok_or_else(|| OptimError::InvalidConfig("Invalid learning _rate".to_string()))?,
        }};
        
        Ok(Box::new({}<T>::new(plugin_config)))
    }}
    
    fn factory_info(&self) -> PluginInfo {{
        create_plugin_info("{}", env!("CARGO_PKG_VERSION"), "Auto-generated")
    }}
    
    fn validate_config(&self, config: &OptimizerConfig) -> Result<()> {{
        if config.learning_rate <= 0.0 {{
            return Err(OptimError::InvalidConfig(
                "Learning _rate must be positive".to_string(),
            ));
        }}
        Ok(())
    }}
    
    fn default_config(&self) -> OptimizerConfig {{
        OptimizerConfig {{
            learning_rate: {},
            ..Default::default()
        }}
    }}
    
    fn config_schema(&self) -> ConfigSchema {{
        let mut schema = ConfigSchema {{
            fields: std::collections::HashMap::new(),
            required_fields: vec!["learning_rate".to_string()],
            version: "1.0".to_string(),
        }};
        
        schema.fields.insert(
            "learning_rate".to_string(),
            FieldSchema {{
                field_type: FieldType::Float {{ min: Some(0.0), max: None }},
                description: "Learning _rate for {} optimization".to_string(),
                default_value: Some(ConfigValue::Float({})),
                constraints: vec![ValidationConstraint::Positive],
                required: true,
            }},
        );
        
        schema
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_{}_creation() {{
        let config = {}Config::default();
        let optimizer = {}<f64>::new(config);
        assert_eq!(optimizer.name(), "{}");
    }}

    #[test]
    fn test_{}_step() {{
        let config = {}Config::default();
        let mut optimizer = {}<f64>::new(config);
        
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        
        let result = optimizer.step(&params, &gradients).unwrap();
        
        // Check that parameters were updated
        assert!((result[0] - 0.9999).abs() < 1e-6); // 1.0 - 0.001 * 0.1
        assert!((result[1] - 1.9998).abs() < 1e-6); // 2.0 - 0.001 * 0.2
        assert!((result[2] - 2.9997).abs() < 1e-6); // 3.0 - 0.001 * 0.3
    }}
}}
"#,
            name, name, env!("CARGO_PKG_VERSION"),
            name, name, name, learning_rate,
            name, name, name, name,
            name, name,
            name, name, name, name, name,
            name, name,
            name, name, name, name,
            name, name, name, name, name, name, learning_rate,
            name.to_lowercase(), name, name, name.to_lowercase(), name, name, name
        );
        
        Ok(content)
    }
}

// Additional generators would be implemented similarly...

/// Test suite generator
#[derive(Debug)]
struct TestSuiteGenerator;

impl CodeGenerator for TestSuiteGenerator {
    fn generate(
        &self,
        template: &PluginTemplate,
        parameters: &HashMap<String, String>,
        config: &TemplateGeneratorConfig,
    ) -> Result<Vec<GeneratedFile>> {
        let optimizer_name = parameters.get("name").unwrap_or(&template.name);
        
        let test_content = self.generate_test_code(optimizer_name, config)?;
        let bench_content = if config.include_benchmarks {
            Some(self.generate_benchmark_code(optimizer_name, config)?)
        } else {
            None
        };
        
        let mut files = vec![GeneratedFile {
            path: PathBuf::from("tests/integration_tests.rs"),
            content: test_content,
            permissions: None,
            file_type: GeneratedFileType::RustSource,
        }];
        
        if let Some(bench_content) = bench_content {
            files.push(GeneratedFile {
                path: PathBuf::from("benches/optimizer_benchmarks.rs"),
                content: bench_content,
                permissions: None,
                file_type: GeneratedFileType::RustSource,
            });
        }
        
        Ok(files)
    }
    
    fn name(&self) -> &str {
        "TestSuite"
    }
    
    fn supported_categories(&self) -> Vec<TemplateCategory> {
        vec![TemplateCategory::BasicOptimizer, TemplateCategory::AdaptiveOptimizer]
    }
}

impl TestSuiteGenerator {
    fn generate_test_code(&self, name: &str, config: &TemplateGeneratorConfig) -> Result<String> {
        let content = format!(
            r#"//! Integration tests for {} optimizer
//!
//! These tests validate the complete functionality of the {} optimizer plugin.

use scirs2_optim::plugin::*;
use ndarray::{{Array1, Array2}};

mod common;
use common::*;

#[test]
#[allow(dead_code)]
fn test_{}_basic_functionality() {{
    let _config = {}Config::default();
    let mut optimizer = {}<f64>::new(_config);
    
    // Test basic step
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    
    let result = optimizer.step(&params, &gradients).unwrap();
    
    // Verify the optimizer updated parameters in the expected direction
    for i in 0..params.len() {{
        assert!(result[i] != params[i], "Parameters should be updated");
    }}
}}

#[test]
#[allow(dead_code)]
fn test_{}_convergence() {{
    let _config = {}Config::default();
    let mut optimizer = {}<f64>::new(_config);
    
    // Test convergence on quadratic function
    let mut params = Array1::from_vec(vec![1.0, 1.0]);
    let target = Array1::from_vec(vec![0.0, 0.0]);
    
    for _iteration in 0..1000 {{
        // Compute gradients: gradient of ||x - target||^2 = 2(x - target)
        let gradients = (&params - &target) * 2.0;
        params = optimizer.step(&params, &gradients).unwrap();
        
        // Check if converged
        let distance = (&params - &target).mapv(|x| x * x).sum().sqrt();
        if distance < 1e-6 {{
            println!("Converged in {{}} iterations"_iteration + 1);
            break;
        }}
    }}
    
    // Verify convergence
    let final_distance = (&params - &target).mapv(|x| x * x).sum().sqrt();
    assert!(final_distance < 1e-3, "Should converge close to target");
}}

#[test]
#[allow(dead_code)]
fn test_{}_state_serialization() {{
    let _config = {}Config::default();
    let mut optimizer = {}<f64>::new(_config);
    
    // Take a few steps to build up state
    let params = Array1::from_vec(vec![1.0, 2.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2]);
    
    for _ in 0..5 {{
        optimizer.step(&params, &gradients).unwrap();
    }}
    
    // Save state
    let saved_state = optimizer.get_state().unwrap();
    
    // Create new optimizer and load state
    let config2 = {}Config::default();
    let mut optimizer2 = {}<f64>::new(config2);
    optimizer2.set_state(saved_state).unwrap();
    
    // Both optimizers should produce same results
    let result1 = optimizer.step(&params, &gradients).unwrap();
    let result2 = optimizer2.step(&params, &gradients).unwrap();
    
    for i in 0..result1.len() {{
        assert!((result1[i] - result2[i]).abs() < 1e-10, "Results should be identical");
    }}
}}

#[test]
#[allow(dead_code)]
fn test_{}_plugin_interface() {{
    let _config = {}Config::default();
    let optimizer = {}<f64>::new(_config);
    
    // Test plugin metadata
    assert_eq!(optimizer.name(), "{}");
    assert!(!optimizer.version().is_empty());
    
    let info = optimizer.plugin_info();
    assert_eq!(info.name, "{}");
    
    let capabilities = optimizer.capabilities();
    assert!(capabilities.state_serialization);
}}

#[test]
#[allow(dead_code)]
fn test_{}_factory() {{
    let factory = {}Factory;
    
    // Test default _config
    let default_config = factory.default_config();
    assert!(default_config.learning_rate > 0.0);
    
    // Test _config validation
    let mut invalid_config = default_config.clone();
    invalid_config.learning_rate = -1.0;
    assert!(factory.validate_config(&invalid_config).is_err());
    
    assert!(factory.validate_config(&default_config).is_ok());
    
    // Test optimizer creation
    let optimizer = factory.create_optimizer(default_config).unwrap();
    assert_eq!(optimizer.name(), "{}");
}}

#[test]
#[allow(dead_code)]
fn test_{}_performance_characteristics() {{
    let _config = {}Config::default();
    let mut optimizer = {}<f64>::new(_config);
    
    // Test with different problem sizes
    for size in &[10, 100, 1000] {{
        let params = Array1::ones(*size);
        let gradients = Array1::ones(*size) * 0.1;
        
        let start = std::time::Instant::now();
        let _result = optimizer.step(&params, &gradients).unwrap();
        let duration = start.elapsed();
        
        // Should complete reasonably quickly
        assert!(duration.as_millis() < 100, "Step should be fast for size {{}}", size);
    }}
}}
"#,
            name, name,
            name.to_lowercase(), name, name,
            name.to_lowercase(), name, name,
            name.to_lowercase(), name, name,
            name, name,
            name.to_lowercase(), name, name, name, name,
            name.to_lowercase(), name,
            name.to_lowercase(), name, name
        );
        
        Ok(content)
    }
    
    fn generate_benchmark_code(&self, name: &str, config: &TemplateGeneratorConfig) -> Result<String> {
        let content = format!(
            r#"//! Benchmarks for {} optimizer
//!
//! These benchmarks measure the performance characteristics of the {} optimizer.

use criterion::{{black_box, criterion_group, criterion_main, Criterion}};
use scirs2_optim::plugin::*;
use ndarray::Array1;

#[allow(dead_code)]
fn benchmark_{}_step(c: &mut Criterion) {{
    let _config = {}Config::default();
    let mut optimizer = {}<f64>::new(_config);
    
    let params = Array1::ones(1000);
    let gradients = Array1::ones(1000) * 0.1;
    
    c.bench_function("{} step", |b| {{
        b.iter(|| {{
            black_box(optimizer.step(black_box(&params), black_box(&gradients)).unwrap())
        }})
    }});
}}

#[allow(dead_code)]
fn benchmark_{}_scalability(c: &mut Criterion) {{
    let mut group = c.benchmark_group("{} scalability");
    
    for size in &[10, 100, 1000, 10000] {{
        let _config = {}Config::default();
        let mut optimizer = {}<f64>::new(_config);
        
        let params = Array1::ones(*size);
        let gradients = Array1::ones(*size) * 0.1;
        
        group.bench_with_input(
            criterion::BenchmarkId::new("size", size),
            size,
            |b_size| {{
                b.iter(|| {{
                    black_box(optimizer.step(black_box(&params), black_box(&gradients)).unwrap())
                }})
            }},
        );
    }}
    
    group.finish();
}}

criterion_group!(benches, benchmark_{}_step, benchmark_{}_scalability);
criterion_main!(benches);
"#,
            name, name,
            name.to_lowercase(), name, name, name,
            name.to_lowercase(), name, name, name,
            name.to_lowercase(), name.to_lowercase()
        );
        
        Ok(content)
    }
}

// Implement other generators...
#[derive(Debug)]
struct DocumentationGenerator;

impl CodeGenerator for DocumentationGenerator {
    fn generate(
        &self,
        template: &PluginTemplate,
        parameters: &HashMap<String, String>,
        config: &TemplateGeneratorConfig,
    ) -> Result<Vec<GeneratedFile>> {
        let optimizer_name = parameters.get("name").unwrap_or(&template.name);
        
        let readme_content = self.generate_readme(optimizer_name, parameters, config)?;
        let lib_docs = self.generate_lib_docs(optimizer_name, config)?;
        
        Ok(vec![
            GeneratedFile {
                path: PathBuf::from("README.md"),
                content: readme_content,
                permissions: None,
                file_type: GeneratedFileType::Documentation,
            },
            GeneratedFile {
                path: PathBuf::from("docs/lib.md"),
                content: lib_docs,
                permissions: None,
                file_type: GeneratedFileType::Documentation,
            },
        ])
    }
    
    fn name(&self) -> &str {
        "Documentation"
    }
    
    fn supported_categories(&self) -> Vec<TemplateCategory> {
        vec![
            TemplateCategory::BasicOptimizer,
            TemplateCategory::AdaptiveOptimizer,
            TemplateCategory::SecondOrderOptimizer,
        ]
    }
}

impl DocumentationGenerator {
    fn generate_readme(&self, name: &str, parameters: &HashMap<String, String>, config: &TemplateGeneratorConfig) -> Result<String> {
        let author = parameters.get("author").unwrap_or(&"Unknown".to_string());
        let description = parameters.get("description").unwrap_or(&format!("{} optimizer plugin", name));
        
        let content = format!(
            r#"# {} Optimizer Plugin

{}

## Overview

The {} optimizer is a custom optimization algorithm implemented as a plugin for the scirs2-optim framework. This plugin provides [describe key features and benefits here].

## Installation

Add this plugin to your `Cargo.toml`:

```toml
[dependencies]
{} = "0.1.0"
```

## Usage

### Basic Usage

```rust
use scirs2_optim::plugin::*;
use {}::*;
use ndarray::Array1;

// Create optimizer configuration
let _config = {}Config {{
    learning_rate: 0.001,
}};

// Create optimizer instance
let mut optimizer = {}<f64>::new(_config);

// Use in optimization loop
let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

let updated_params = optimizer.step(&params, &gradients)?;
```

### Advanced Usage

```rust
// Using the plugin factory
let factory = {}Factory;
let _config = factory.default_config();
let optimizer = factory.create_optimizer(_config)?;

// State serialization
let state = optimizer.get_state()?;
// ... save state to file ...
// ... load state from file ...
optimizer.set_state(loaded_state)?;
```

## Algorithm Details

[Describe the optimization algorithm, its mathematical foundation, convergence properties, etc.]

### Mathematical Formulation

The {} optimizer updates parameters according to:

```
θ_{t+1} = θ_t - α ∇f(θ_t)
```

Where:
- θ represents the parameters
- α is the learning rate
- ∇f(θ_t) is the gradient at step t

### Key Features

- **Feature 1**: Description
- **Feature 2**: Description
- **Feature 3**: Description

## Configuration

The optimizer supports the following configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| learning_rate | f64 | 0.001 | Learning rate for parameter updates |

## Benchmarks

Performance characteristics on common optimization problems:

| Problem Type | Convergence Rate | Memory Usage | Notes |
|--------------|------------------|--------------|-------|
| Quadratic | Fast | Low | Optimal for convex problems |
| Non-convex | Medium | Low | Good general-purpose performance |

## Testing

Run the test suite:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

{}

## Changelog

### Version 0.1.0
- Initial release
- Basic {} optimization algorithm
- Plugin interface implementation
- Test suite and benchmarks
"#,
            name, description, name, name.to_lowercase(), name.to_lowercase(), name, name, name, name, author, name
        );
        
        Ok(content)
    }
    
    fn generate_lib_docs(&self, name: &str, config: &TemplateGeneratorConfig) -> Result<String> {
        let content = format!(
            r#"# {} Optimizer Library Documentation

## Architecture

The {} optimizer plugin is structured as follows:

```
src/
├── lib.rs          # Main library with optimizer implementation
├── config.rs       # Configuration structures
├── state.rs        # State management
└── factory.rs      # Plugin factory implementation

tests/
├── integration_tests.rs    # Integration tests
└── common/                 # Test utilities

benches/
└── optimizer_benchmarks.rs # Performance benchmarks
```

## Core Components

### {}Config

Configuration structure for the optimizer:

```rust
pub struct {}Config<T: Float> {{
    pub learning_rate: T,
    // Additional configuration parameters
}}
```

### {} Implementation

The main optimizer struct implementing the `OptimizerPlugin` trait:

```rust
pub struct {}<T: Float> {{
    _config: {}Config<T>,
    state: {}State<T>,
    info: PluginInfo,
    capabilities: PluginCapabilities,
}}
```

### Plugin Factory

Factory for creating optimizer instances:

```rust
pub struct {}Factory;
```

## API Reference

### Methods

#### `step(params, gradients) -> Result<Array1<T>>`

Performs a single optimization step.

**Parameters:**
- `params`: Current parameter values
- `gradients`: Computed gradients

**Returns:**
- Updated parameter values

#### `reset() -> Result<()>`

Resets the optimizer state.

#### `get_state() -> Result<OptimizerState>`

Retrieves the current optimizer state for serialization.

#### `set_state(state) -> Result<()>`

Sets the optimizer state from deserialized data.

## Usage Patterns

### Pattern 1: Basic Optimization Loop

```rust
for epoch in 0..num_epochs {{
    let gradients = compute_gradients(&params);
    params = optimizer.step(&params, &gradients)?;
}}
```

### Pattern 2: With State Management

```rust
// Save state periodically
if epoch % checkpoint_interval == 0 {{
    let state = optimizer.get_state()?;
    save_checkpoint(&state)?;
}}

// Resume from checkpoint
let state = load_checkpoint()?;
optimizer.set_state(state)?;
```

### Pattern 3: Factory-based Creation

```rust
let factory = {}Factory;
let _config = factory.default_config();
let mut optimizer = factory.create_optimizer(_config)?;
```

## Performance Considerations

- The {} optimizer has O(n) memory complexity where n is the number of parameters
- Time complexity per step is O(n)
- Suitable for problems with up to 10^6 parameters

## Error Handling

The plugin uses the standard `Result<T>` pattern:

```rust
match optimizer.step(&params, &gradients) {{
    Ok(updated_params) => {{
        // Continue optimization
    }}
    Err(OptimError::InvalidConfig(msg)) => {{
        // Handle configuration error
    }}
    Err(e) => {{
        // Handle other errors
    }}
}}
```

## Extension Points

The plugin can be extended by:

1. Adding new configuration parameters
2. Implementing additional state variables
3. Extending the capabilities structure
4. Adding custom validation logic
"#,
            name, name, name, name, name, name, name, name, name, name, name
        );
        
        Ok(content)
    }
}

// Implement remaining generators...
#[derive(Debug)]
struct BenchmarkGenerator;

impl CodeGenerator for BenchmarkGenerator {
    fn generate(&self,
        template: &PluginTemplate, _parameters: &HashMap<String, String>, _config: &TemplateGeneratorConfig) -> Result<Vec<GeneratedFile>> {
        // Implementation would generate comprehensive benchmark suites
        Ok(vec![])
    }
    
    fn name(&self) -> &str { "Benchmark" }
    fn supported_categories(&self) -> Vec<TemplateCategory> { vec![] }
}

#[derive(Debug)]
struct ExampleGenerator;

impl CodeGenerator for ExampleGenerator {
    fn generate(&self,
        template: &PluginTemplate, _parameters: &HashMap<String, String>, _config: &TemplateGeneratorConfig) -> Result<Vec<GeneratedFile>> {
        // Implementation would generate usage examples
        Ok(vec![])
    }
    
    fn name(&self) -> &str { "Example" }
    fn supported_categories(&self) -> Vec<TemplateCategory> { vec![] }
}

#[derive(Debug)]
struct CICDGenerator;

impl CodeGenerator for CICDGenerator {
    fn generate(&self,
        template: &PluginTemplate, _parameters: &HashMap<String, String>, _config: &TemplateGeneratorConfig) -> Result<Vec<GeneratedFile>> {
        // Implementation would generate GitHub Actions, Travis CI, etc.
        Ok(vec![])
    }
    
    fn name(&self) -> &str { "CICD" }
    fn supported_categories(&self) -> Vec<TemplateCategory> { vec![] }
}

#[derive(Debug)]
struct ConfigurationGenerator;

impl CodeGenerator for ConfigurationGenerator {
    fn generate(&self,
        template: &PluginTemplate, _parameters: &HashMap<String, String>, _config: &TemplateGeneratorConfig) -> Result<Vec<GeneratedFile>> {
        // Implementation would generate Cargo.toml, plugin manifests, etc.
        Ok(vec![])
    }
    
    fn name(&self) -> &str { "Configuration" }
    fn supported_categories(&self) -> Vec<TemplateCategory> { vec![] }
}

// Implement template registry and other supporting structures...

impl TemplateRegistry {
    fn new() -> Self {
        Self {
            builtin_templates: HashMap::new(),
            custom_templates: HashMap::new(),
            template_metadata: HashMap::new(),
        }
    }
    
    fn register_builtin_templates(&mut self) {
        // Register built-in templates
        let basic_template = self.create_basic_optimizer_template();
        self.builtin_templates.insert("basic_optimizer".to_string(), basic_template);
        
        let adaptive_template = self.create_adaptive_optimizer_template();
        self.builtin_templates.insert("adaptive_optimizer".to_string(), adaptive_template);
    }
    
    fn create_basic_optimizer_template(&self) -> PluginTemplate {
        PluginTemplate {
            name: "BasicOptimizer".to_string(),
            description: "A basic gradient descent optimizer template".to_string(),
            category: TemplateCategory::BasicOptimizer,
            complexity: ComplexityLevel::Beginner,
            structure: EnhancedTemplateStructure {
                core_files: vec![],
                test_files: vec![],
                documentation: vec![],
                config_files: vec![],
                cicd_files: vec![],
                example_files: vec![],
                benchmark_files: vec![],
                resource_files: vec![],
            },
            required_features: vec!["optimizer".to_string()],
            parameters: {
                let mut params = HashMap::new();
                params.insert("name".to_string(), TemplateParameter {
                    name: "name".to_string(),
                    param_type: ParameterType::String,
                    default_value: Some("MyOptimizer".to_string()),
                    description: "Name of the optimizer".to_string(),
                    required: true,
                    validation: vec![ParameterValidation::MinLength(1)],
                });
                params.insert("learning_rate".to_string(), TemplateParameter {
                    name: "learning_rate".to_string(),
                    param_type: ParameterType::Float,
                    default_value: Some("0.001".to_string()),
                    description: "Default learning rate".to_string(),
                    required: false,
                    validation: vec![ParameterValidation::Range(0.0, 1.0)],
                });
                params
            },
        }
    }
    
    fn create_adaptive_optimizer_template(&self) -> PluginTemplate {
        PluginTemplate {
            name: "AdaptiveOptimizer".to_string(),
            description: "An adaptive optimizer template with momentum and learning rate adaptation".to_string(),
            category: TemplateCategory::AdaptiveOptimizer,
            complexity: ComplexityLevel::Intermediate,
            structure: EnhancedTemplateStructure {
                core_files: vec![],
                test_files: vec![],
                documentation: vec![],
                config_files: vec![],
                cicd_files: vec![],
                example_files: vec![],
                benchmark_files: vec![],
                resource_files: vec![],
            },
            required_features: vec!["optimizer".to_string(), "adaptive".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn get_template(&self, name: &str) -> Option<&PluginTemplate> {
        self.builtin_templates.get(name).or_else(|| self.custom_templates.get(name))
    }
    
    fn list_templates(&self) -> Vec<TemplateInfo> {
        let mut templates = Vec::new();
        
        for (name, template) in &self.builtin_templates {
            templates.push(TemplateInfo {
                name: name.clone(),
                description: template.description.clone(),
                category: template.category.clone(),
                complexity: template.complexity.clone(),
                required_features: template.required_features.clone(),
                metadata: self.template_metadata.get(name).cloned().unwrap_or_else(|| TemplateMetadata {
                    version: "1.0.0".to_string(),
                    author: "scirs2-optim".to_string(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    modified_at: chrono::Utc::now().to_rfc3339(),
                    usage_count: 0,
                    ratings: vec![],
                    tags: vec![],
                    dependencies: vec![],
                }),
            });
        }
        
        templates
    }
    
    fn get_template_details(&self, name: &str) -> Option<TemplateDetails> {
        self.get_template(name).map(|template| {
            TemplateDetails {
                info: TemplateInfo {
                    name: name.to_string(),
                    description: template.description.clone(),
                    category: template.category.clone(),
                    complexity: template.complexity.clone(),
                    required_features: template.required_features.clone(),
                    metadata: self.template_metadata.get(name).cloned().unwrap_or_default(),
                },
                structure: template.structure.clone(),
                parameters: template.parameters.clone(),
                examples: vec![], // Would be populated with usage examples
            }
        })
    }
    
    fn add_custom_template(&mut self, template: PluginTemplate) -> Result<()> {
        let name = template.name.clone();
        self.custom_templates.insert(name, template);
        Ok(())
    }
}

impl TemplateValidator {
    fn new() -> Self {
        Self {
            rules: vec![
                Box::new(NameValidationRule),
                Box::new(ParameterValidationRule),
                Box::new(StructureValidationRule),
            ],
            syntax_checker: SyntaxChecker::new(),
        }
    }
    
    fn validate_template(&self, template: &PluginTemplate) -> Result<PluginValidationResult> {
        let mut result = PluginValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
            benchmark_results: None,
        };
        
        for rule in &self.rules {
            let validation = rule.validate(template);
            if !validation.passed {
                result.is_valid = false;
                result.errors.push(validation.message);
            }
        }
        
        Ok(result)
    }
}

impl SyntaxChecker {
    fn new() -> Self {
        Self {
            rust_validator: RustSyntaxValidator,
            toml_validator: TomlValidator,
            markdown_validator: MarkdownValidator,
        }
    }
}

// Validation rules implementations
#[derive(Debug)]
struct NameValidationRule;

impl TemplateValidationRule for NameValidationRule {
    fn validate(&self, template: &PluginTemplate) -> ValidationResult {
        if template.name.is_empty() {
            ValidationResult {
                passed: false,
                message: "Template name cannot be empty".to_string(),
                severity: ValidationSeverity::Error,
                suggestions: vec!["Provide a descriptive name for the template".to_string()],
            }
        } else {
            ValidationResult {
                passed: true,
                message: "Template name is valid".to_string(),
                severity: ValidationSeverity::Info,
                suggestions: vec![],
            }
        }
    }
    
    fn name(&self) -> &str {
        "NameValidation"
    }
    
    fn description(&self) -> &str {
        "Validates template name requirements"
    }
}

#[derive(Debug)]
struct ParameterValidationRule;

impl TemplateValidationRule for ParameterValidationRule {
    fn validate(&self, template: &PluginTemplate) -> ValidationResult {
        for (param_name, param) in &template.parameters {
            if param.name.is_empty() {
                return ValidationResult {
                    passed: false,
                    message: format!("Parameter '{}' has empty name", param_name),
                    severity: ValidationSeverity::Error,
                    suggestions: vec!["Ensure all parameters have non-empty names".to_string()],
                };
            }
        }
        
        ValidationResult {
            passed: true,
            message: "All parameters are valid".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: vec![],
        }
    }
    
    fn name(&self) -> &str {
        "ParameterValidation"
    }
    
    fn description(&self) -> &str {
        "Validates template parameter definitions"
    }
}

#[derive(Debug)]
struct StructureValidationRule;

impl TemplateValidationRule for StructureValidationRule {
    fn validate(&self,
        template: &PluginTemplate) -> ValidationResult {
        // Validate _template structure
        ValidationResult {
            passed: true,
            message: "Template structure is valid".to_string(),
            severity: ValidationSeverity::Info,
            suggestions: vec![],
        }
    }
    
    fn name(&self) -> &str {
        "StructureValidation"
    }
    
    fn description(&self) -> &str {
        "Validates template structure requirements"
    }
}

// Plugin analyzer for reverse engineering templates
#[derive(Debug)]
struct PluginAnalyzer;

impl PluginAnalyzer {
    fn new() -> Self {
        Self
    }
    
    fn analyze_plugin(&self, _plugindir: &Path) -> Result<PluginAnalysis> {
        // Analyze existing plugin to extract structure and patterns
        Ok(PluginAnalysis {
            structure: AnalyzedStructure::default(),
            dependencies: vec![],
            patterns: vec![],
        })
    }
}

#[derive(Debug)]
struct PluginAnalysis {
    structure: AnalyzedStructure,
    dependencies: Vec<String>,
    patterns: Vec<String>,
}

#[derive(Debug, Default)]
struct AnalyzedStructure {
    source_files: Vec<String>,
    test_files: Vec<String>,
    config_files: Vec<String>,
}

// Template extractor for creating templates from analysis
#[derive(Debug)]
struct TemplateExtractor;

impl TemplateExtractor {
    fn new() -> Self {
        Self
    }
    
    fn extract_template(&self,
        analysis: PluginAnalysis, template_name: &str) -> Result<PluginTemplate> {
        // Extract template structure from _analysis
        Ok(PluginTemplate {
            _name: template_name.to_string(),
            description: "Extracted template".to_string(),
            category: TemplateCategory::BasicOptimizer,
            complexity: ComplexityLevel::Intermediate,
            structure: EnhancedTemplateStructure {
                core_files: vec![],
                test_files: vec![],
                documentation: vec![],
                config_files: vec![],
                cicd_files: vec![],
                example_files: vec![],
                benchmark_files: vec![],
                resource_files: vec![],
            },
            required_features: vec![],
            parameters: HashMap::new(),
        })
    }
}

// Implement trait for template
impl PluginTemplate {
    fn version(&self) -> Option<String> {
        Some("1.0.0".to_string())
    }
}

impl Default for TemplateMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            author: "Unknown".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            modified_at: chrono::Utc::now().to_rfc3339(),
            usage_count: 0,
            ratings: vec![],
            tags: vec![],
            dependencies: vec![],
        }
    }
}

impl Default for TemplateGeneratorConfig {
    fn default() -> Self {
        Self {
            rust_version: "1.70.0".to_string(),
            include_cicd: true,
            include_docs: true,
            include_benchmarks: true,
            include_examples: true,
            include_gpu: false,
            include_distributed: false,
            code_style: CodeStyle {
                indentation: IndentationType::Spaces(4),
                max_line_length: 100,
                trailing_commas: true,
                import_style: ImportStyle::Grouped,
                doc_style: DocStyle::Comprehensive,
            },
            license: LicenseType::MIT,
            testing_framework: TestingFramework::Standard,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_template_generator_creation() {
        let config = TemplateGeneratorConfig::default();
        let generator = AdvancedTemplateGenerator::new(config);
        
        assert!(!generator.list_templates().is_empty());
    }

    #[test]
    fn test_template_validation() {
        let config = TemplateGeneratorConfig::default();
        let generator = AdvancedTemplateGenerator::new(config);
        
        let templates = generator.list_templates();
        assert!(!templates.is_empty());
        
        let template_name = &templates[0].name;
        let details = generator.get_template_details(template_name).unwrap();
        
        let validation_result = generator.validator.validate_template(&details.info.into()).unwrap();
        assert!(validation_result.is_valid);
    }

    #[test]
    fn test_project_generation() {
        let config = TemplateGeneratorConfig::default();
        let generator = AdvancedTemplateGenerator::new(config);
        
        let temp_dir = TempDir::new().unwrap();
        let mut parameters = HashMap::new();
        parameters.insert("name".to_string(), "TestOptimizer".to_string());
        parameters.insert("author".to_string(), "Test Author".to_string());
        
        let result = generator.generate_plugin_project(
            "basic_optimizer",
            "test_project",
            parameters,
            temp_dir.path(),
        );
        
        assert!(result.is_ok());
        let generation_result = result.unwrap();
        assert!(!generation_result.generated_files.is_empty());
        
        // Verify main source file was created
        let lib_rs_path = temp_dir.path().join("src/lib.rs");
        assert!(lib_rs_path.exists());
    }

    // Helper function to convert TemplateInfo to PluginTemplate for testing
    impl From<TemplateInfo> for PluginTemplate {
        fn from(info: TemplateInfo) -> Self {
            PluginTemplate {
                name: info.name,
                description: info.description,
                category: info.category,
                complexity: info.complexity,
                structure: EnhancedTemplateStructure {
                    core_files: vec![],
                    test_files: vec![],
                    documentation: vec![],
                    config_files: vec![],
                    cicd_files: vec![],
                    example_files: vec![],
                    benchmark_files: vec![],
                    resource_files: vec![],
                },
                required_features: info.required_features,
                parameters: HashMap::new(),
            }
        }
    }
}
