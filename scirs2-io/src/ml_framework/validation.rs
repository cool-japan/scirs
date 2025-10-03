//! Model validation and compatibility checking between frameworks

use crate::error::Result;
use crate::ml_framework::{DataType, MLFramework, MLModel};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Model validator for checking compatibility between frameworks
pub struct ModelValidator {
    source_framework: MLFramework,
    target_framework: MLFramework,
    validation_config: ValidationConfig,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub check_data_types: bool,
    pub check_tensorshapes: bool,
    pub check_operations: bool,
    pub check_metadata: bool,
    pub strict_mode: bool,
    pub allow_type_conversion: bool,
    pub maxshape_dimension: Option<usize>,
    pub supported_dtypes: Option<HashSet<DataType>>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_data_types: true,
            check_tensorshapes: true,
            check_operations: true,
            check_metadata: true,
            strict_mode: false,
            allow_type_conversion: true,
            maxshape_dimension: Some(8), // Most frameworks support up to 8D tensors
            supported_dtypes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_compatible: bool,
    pub compatibility_score: f32, // 0.0 to 1.0
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub recommendations: Vec<ValidationRecommendation>,
    pub conversion_path: Option<ConversionPath>,
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub location: Option<String>, // e.g., tensor name, operation name
    pub fix_suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub category: WarningCategory,
    pub message: String,
    pub location: Option<String>,
    pub impact: WarningImpact,
}

#[derive(Debug, Clone)]
pub struct ValidationRecommendation {
    pub category: RecommendationCategory,
    pub message: String,
    pub priority: RecommendationPriority,
    pub estimated_effort: EstimatedEffort,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    DataType,
    Shape,
    Operation,
    Metadata,
    Framework,
    Version,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Critical, // Blocks conversion
    High,     // Likely to cause runtime errors
    Medium,   // May cause issues
    Low,      // Minor issues
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningCategory {
    Performance,
    Precision,
    Compatibility,
    BestPractice,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningImpact {
    High,   // Significant impact on model behavior
    Medium, // Moderate impact
    Low,    // Minor impact
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    Optimization,
    Conversion,
    Preprocessing,
    Alternative,
    BestPractice,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub enum EstimatedEffort {
    Minimal,  // < 1 hour
    Low,      // 1-4 hours
    Medium,   // 1-2 days
    High,     // 1 week
    VeryHigh, // > 1 week
}

#[derive(Debug, Clone)]
pub struct ConversionPath {
    pub steps: Vec<ConversionStep>,
    pub estimated_accuracy_loss: f32,      // 0.0 to 1.0
    pub estimated_performance_impact: f32, // Relative performance change
    pub complexity: ConversionComplexity,
}

#[derive(Debug, Clone)]
pub struct ConversionStep {
    pub operation: ConversionOperation,
    pub description: String,
    pub required_tools: Vec<String>,
    pub estimated_time: EstimatedEffort,
}

#[derive(Debug, Clone)]
pub enum ConversionOperation {
    DirectConversion,
    TypeConversion,
    ShapeReshaping,
    OperationMapping,
    ManualIntervention,
    AlternativeImplementation,
}

#[derive(Debug, Clone)]
pub enum ConversionComplexity {
    Trivial,     // Direct conversion possible
    Simple,      // Minor adjustments needed
    Moderate,    // Some manual work required
    Complex,     // Significant effort required
    VeryComplex, // Major rewrite needed
}

impl ModelValidator {
    pub fn new(source: MLFramework, target: MLFramework, config: ValidationConfig) -> Self {
        Self {
            source_framework: source,
            target_framework: target,
            validation_config: config,
        }
    }

    /// Validate model compatibility
    pub fn validate(&self, model: &MLModel) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Check framework compatibility
        let framework_compatibility = self.check_framework_compatibility(model);
        if let Some(error) = framework_compatibility.error {
            errors.push(error);
        }
        warnings.extend(framework_compatibility.warnings);
        recommendations.extend(framework_compatibility.recommendations);

        // Check data types
        if self.validation_config.check_data_types {
            let dtype_check = self.check_data_types(model);
            errors.extend(dtype_check.errors);
            warnings.extend(dtype_check.warnings);
            recommendations.extend(dtype_check.recommendations);
        }

        // Check tensor shapes
        if self.validation_config.check_tensorshapes {
            let shape_check = self.check_tensorshapes(model);
            errors.extend(shape_check.errors);
            warnings.extend(shape_check.warnings);
            recommendations.extend(shape_check.recommendations);
        }

        // Check operations (if applicable)
        if self.validation_config.check_operations {
            let ops_check = self.check_operations(model);
            errors.extend(ops_check.errors);
            warnings.extend(ops_check.warnings);
            recommendations.extend(ops_check.recommendations);
        }

        // Check metadata
        if self.validation_config.check_metadata {
            let metadata_check = self.check_metadata(model);
            errors.extend(metadata_check.errors);
            warnings.extend(metadata_check.warnings);
            recommendations.extend(metadata_check.recommendations);
        }

        // Calculate compatibility score
        let compatibility_score = self.calculate_compatibility_score(&errors, &warnings);
        let is_compatible = compatibility_score > 0.7
            && errors.iter().all(|e| e.severity != ErrorSeverity::Critical);

        // Generate conversion path if compatible
        let conversion_path = if is_compatible {
            Some(self.generate_conversion_path(model, &errors, &warnings)?)
        } else {
            None
        };

        Ok(ValidationReport {
            is_compatible,
            compatibility_score,
            errors,
            warnings,
            recommendations,
            conversion_path,
        })
    }

    /// Check framework compatibility
    fn check_framework_compatibility(&self, model: &MLModel) -> FrameworkCompatibilityResult {
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Check if frameworks are the same
        if self.source_framework == self.target_framework {
            return FrameworkCompatibilityResult {
                error: None,
                warnings,
                recommendations,
            };
        }

        // Check common conversion paths
        let compatibility_score = crate::ml_framework::validation::utils::quick_compatibility_check(
            self.source_framework,
            self.target_framework,
        );

        if compatibility_score < 0.5 {
            warnings.push(ValidationWarning {
                category: WarningCategory::Compatibility,
                message: format!(
                    "Low compatibility between {:?} and {:?} (score: {:.2})",
                    self.source_framework, self.target_framework, compatibility_score
                ),
                location: None,
                impact: WarningImpact::High,
            });

            recommendations.push(ValidationRecommendation {
                category: RecommendationCategory::Alternative,
                message: "Consider using ONNX as an intermediate format".to_string(),
                priority: RecommendationPriority::High,
                estimated_effort: EstimatedEffort::Medium,
            });
        }

        FrameworkCompatibilityResult {
            error: None,
            warnings,
            recommendations,
        }
    }

    /// Check data types compatibility
    fn check_data_types(&self, model: &MLModel) -> ValidationCheckResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let recommendations = Vec::new();

        for (tensor_name, tensor) in &model.weights {
            // Check if data type is supported by target framework
            if let Some(ref supported_dtypes) = self.validation_config.supported_dtypes {
                if !supported_dtypes.contains(&tensor.metadata.dtype) {
                    if self.validation_config.allow_type_conversion {
                        warnings.push(ValidationWarning {
                            category: WarningCategory::Precision,
                            message: format!(
                                "Tensor '{}' has unsupported data type {:?}, conversion may be needed",
                                tensor_name, tensor.metadata.dtype
                            ),
                            location: Some(tensor_name.clone()),
                            impact: WarningImpact::Medium,
                        });
                    } else {
                        errors.push(ValidationError {
                            category: ErrorCategory::DataType,
                            severity: ErrorSeverity::High,
                            message: format!(
                                "Tensor '{}' has unsupported data type {:?}",
                                tensor_name, tensor.metadata.dtype
                            ),
                            location: Some(tensor_name.clone()),
                            fix_suggestion: Some(
                                "Enable type conversion or change tensor data type".to_string(),
                            ),
                        });
                    }
                }
            }

            // Check for precision loss warnings
            if let (MLFramework::PyTorch, MLFramework::CoreML, DataType::Float64) = (
                &self.source_framework,
                &self.target_framework,
                &tensor.metadata.dtype,
            ) {
                warnings.push(ValidationWarning {
                    category: WarningCategory::Precision,
                    message: format!(
                        "Tensor '{}' uses Float64 which may be converted to Float32 in CoreML",
                        tensor_name
                    ),
                    location: Some(tensor_name.clone()),
                    impact: WarningImpact::Medium,
                });
            }
        }

        ValidationCheckResult {
            errors,
            warnings,
            recommendations,
        }
    }

    /// Check tensor shapes compatibility
    fn check_tensorshapes(&self, model: &MLModel) -> ValidationCheckResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let recommendations = Vec::new();

        for (tensor_name, tensor) in &model.weights {
            let shape = &tensor.metadata.shape;

            // Check maximum dimensions
            if let Some(max_dims) = self.validation_config.maxshape_dimension {
                if shape.len() > max_dims {
                    errors.push(ValidationError {
                        category: ErrorCategory::Shape,
                        severity: ErrorSeverity::High,
                        message: format!(
                            "Tensor '{}' has {} dimensions, but target framework supports max {}",
                            tensor_name,
                            shape.len(),
                            max_dims
                        ),
                        location: Some(tensor_name.clone()),
                        fix_suggestion: Some(
                            "Reshape tensor or use tensor decomposition".to_string(),
                        ),
                    });
                }
            }

            // Check for dynamic shapes (represented as 0 dimensions)
            if shape.contains(&0) {
                warnings.push(ValidationWarning {
                    category: WarningCategory::Compatibility,
                    message: format!(
                        "Tensor '{}' has dynamic shape dimensions which may not be supported",
                        tensor_name
                    ),
                    location: Some(tensor_name.clone()),
                    impact: WarningImpact::High,
                });
            }

            // Check for very large tensors
            let total_elements: usize = shape.iter().product();
            if total_elements > 1_000_000_000 {
                warnings.push(ValidationWarning {
                    category: WarningCategory::Performance,
                    message: format!(
                        "Tensor '{}' is very large ({} elements), may cause memory issues",
                        tensor_name, total_elements
                    ),
                    location: Some(tensor_name.clone()),
                    impact: WarningImpact::Medium,
                });
            }
        }

        ValidationCheckResult {
            errors,
            warnings,
            recommendations,
        }
    }

    /// Check operations compatibility (simplified implementation)
    fn check_operations(&self, model: &MLModel) -> ValidationCheckResult {
        let errors = Vec::new();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // In a real implementation, this would analyze the model graph and operations
        // For now, we'll just provide framework-specific warnings
        match (&self.source_framework, &self.target_framework) {
            (MLFramework::PyTorch, MLFramework::CoreML) => {
                warnings.push(ValidationWarning {
                    category: WarningCategory::Compatibility,
                    message: "Some PyTorch operations may not have direct CoreML equivalents"
                        .to_string(),
                    location: None,
                    impact: WarningImpact::Medium,
                });
            }
            (MLFramework::TensorFlow, MLFramework::PyTorch) => {
                recommendations.push(ValidationRecommendation {
                    category: RecommendationCategory::Conversion,
                    message: "Consider using ONNX as intermediate format for TensorFlow -> PyTorch conversion".to_string(),
                    priority: RecommendationPriority::Medium,
                    estimated_effort: EstimatedEffort::Low,
                });
            }
            _ => {}
        }

        ValidationCheckResult {
            errors,
            warnings,
            recommendations,
        }
    }

    /// Check metadata compatibility
    fn check_metadata(&self, model: &MLModel) -> ValidationCheckResult {
        let errors = Vec::new();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Check if framework version is compatible
        if let Some(ref framework_version) = model.metadata.framework_version {
            // This is a simplified check - in practice would have version compatibility matrices
            if framework_version.starts_with("0.") {
                warnings.push(ValidationWarning {
                    category: WarningCategory::Compatibility,
                    message: format!(
                        "Framework version {} appears to be a pre-release version",
                        framework_version
                    ),
                    location: None,
                    impact: WarningImpact::Low,
                });
            }
        }

        // Check for missing critical metadata
        if model.metadata.model_name.is_none() {
            recommendations.push(ValidationRecommendation {
                category: RecommendationCategory::BestPractice,
                message: "Consider adding a model name for better tracking".to_string(),
                priority: RecommendationPriority::Low,
                estimated_effort: EstimatedEffort::Minimal,
            });
        }

        // Check for empty configurations
        if model.config.is_empty() {
            warnings.push(ValidationWarning {
                category: WarningCategory::BestPractice,
                message: "Model configuration is empty, may cause issues during conversion"
                    .to_string(),
                location: None,
                impact: WarningImpact::Low,
            });
        }

        ValidationCheckResult {
            errors,
            warnings,
            recommendations,
        }
    }

    /// Calculate overall compatibility score
    fn calculate_compatibility_score(
        &self,
        errors: &[ValidationError],
        warnings: &[ValidationWarning],
    ) -> f32 {
        let base_score = crate::ml_framework::validation::utils::quick_compatibility_check(
            self.source_framework,
            self.target_framework,
        );

        // Reduce score based on errors and warnings
        let error_penalty: f32 = errors
            .iter()
            .map(|e| match e.severity {
                ErrorSeverity::Critical => 0.5,
                ErrorSeverity::High => 0.3,
                ErrorSeverity::Medium => 0.1,
                ErrorSeverity::Low => 0.05,
            })
            .sum();

        let warning_penalty: f32 = warnings
            .iter()
            .map(|w| match w.impact {
                WarningImpact::High => 0.1,
                WarningImpact::Medium => 0.05,
                WarningImpact::Low => 0.02,
            })
            .sum();

        (base_score - error_penalty - warning_penalty)
            .max(0.0)
            .min(1.0)
    }

    /// Generate conversion path
    fn generate_conversion_path(
        &self,
        _model: &MLModel,
        errors: &[ValidationError],
        warnings: &[ValidationWarning],
    ) -> Result<ConversionPath> {
        let mut steps = Vec::new();

        // Analyze errors and warnings to determine conversion steps
        let has_dtype_issues = errors.iter().any(|e| e.category == ErrorCategory::DataType)
            || warnings
                .iter()
                .any(|w| w.category == WarningCategory::Precision);

        let hasshape_issues = errors.iter().any(|e| e.category == ErrorCategory::Shape);

        if has_dtype_issues {
            steps.push(ConversionStep {
                operation: ConversionOperation::TypeConversion,
                description: "Convert incompatible data types".to_string(),
                required_tools: vec!["dtype_converter".to_string()],
                estimated_time: EstimatedEffort::Low,
            });
        }

        if hasshape_issues {
            steps.push(ConversionStep {
                operation: ConversionOperation::ShapeReshaping,
                description: "Reshape tensors for target framework".to_string(),
                required_tools: vec!["shape_converter".to_string()],
                estimated_time: EstimatedEffort::Medium,
            });
        }

        // Add main conversion step
        let conversion_complexity = if steps.is_empty() {
            ConversionComplexity::Trivial
        } else if steps.len() <= 2 {
            ConversionComplexity::Simple
        } else {
            ConversionComplexity::Moderate
        };

        steps.push(ConversionStep {
            operation: ConversionOperation::DirectConversion,
            description: format!(
                "Convert from {:?} to {:?}",
                self.source_framework, self.target_framework
            ),
            required_tools: vec![format!("{:?}_converter", self.target_framework)],
            estimated_time: match conversion_complexity {
                ConversionComplexity::Trivial => EstimatedEffort::Minimal,
                ConversionComplexity::Simple => EstimatedEffort::Low,
                _ => EstimatedEffort::Medium,
            },
        });

        Ok(ConversionPath {
            steps,
            estimated_accuracy_loss: if has_dtype_issues { 0.05 } else { 0.01 },
            estimated_performance_impact: if hasshape_issues { 0.1 } else { 0.02 },
            complexity: conversion_complexity,
        })
    }
}

/// Batch validation for multiple models
pub struct BatchValidator {
    validators: Vec<ModelValidator>,
    #[allow(dead_code)]
    parallel: bool,
}

impl Default for BatchValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchValidator {
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
            parallel: true,
        }
    }

    pub fn add_validation(
        &mut self,
        source: MLFramework,
        target: MLFramework,
        config: ValidationConfig,
    ) {
        self.validators
            .push(ModelValidator::new(source, target, config));
    }

    pub fn validate_all(&self, models: &[MLModel]) -> Result<Vec<ValidationReport>> {
        let mut reports = Vec::new();

        for model in models {
            for validator in &self.validators {
                reports.push(validator.validate(model)?);
            }
        }

        Ok(reports)
    }
}

/// Validation utilities
pub mod utils {
    use super::*;

    /// Quick compatibility check
    pub fn quick_compatibility_check(source: MLFramework, target: MLFramework) -> f32 {
        // Simplified compatibility check
        if source == target {
            1.0
        } else if matches!(
            (source, target),
            (MLFramework::PyTorch, MLFramework::ONNX)
                | (MLFramework::TensorFlow, MLFramework::ONNX)
                | (MLFramework::ONNX, MLFramework::PyTorch)
                | (MLFramework::ONNX, MLFramework::TensorFlow)
        ) {
            0.9
        } else {
            0.5
        }
    }

    /// Generate compatibility matrix for all frameworks
    pub fn generate_compatibility_matrix() -> BTreeMap<String, BTreeMap<String, f32>> {
        let frameworks = [
            MLFramework::PyTorch,
            MLFramework::TensorFlow,
            MLFramework::ONNX,
            MLFramework::SafeTensors,
            MLFramework::JAX,
            MLFramework::MXNet,
            MLFramework::CoreML,
            MLFramework::HuggingFace,
        ];

        let mut matrix = BTreeMap::new();

        for source in &frameworks {
            let mut row = BTreeMap::new();
            for target in &frameworks {
                let score = quick_compatibility_check(*source, *target);
                row.insert(format!("{:?}", target), score);
            }
            matrix.insert(format!("{:?}", source), row);
        }

        matrix
    }

    /// Find best conversion path between frameworks
    pub fn find_best_conversion_path(source: MLFramework, target: MLFramework) -> Vec<MLFramework> {
        // Simple pathfinding - in practice could use more sophisticated algorithms
        if source == target {
            return vec![source];
        }

        // Try direct conversion first
        if quick_compatibility_check(source, target) > 0.7 {
            return vec![source, target];
        }

        // Try via ONNX as intermediate
        if quick_compatibility_check(source, MLFramework::ONNX) > 0.7
            && quick_compatibility_check(MLFramework::ONNX, target) > 0.7
        {
            return vec![source, MLFramework::ONNX, target];
        }

        // Fallback to direct conversion
        vec![source, target]
    }
}

// Supporting structures
#[derive(Debug, Clone)]
struct FrameworkCompatibilityResult {
    error: Option<ValidationError>,
    warnings: Vec<ValidationWarning>,
    recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Clone)]
struct ValidationCheckResult {
    errors: Vec<ValidationError>,
    warnings: Vec<ValidationWarning>,
    recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Clone)]
struct FrameworkCompatibility {
    level: CompatibilityLevel,
    recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Clone)]
enum CompatibilityLevel {
    FullyCompatible,
    MostlyCompatible,
    PartiallyCompatible,
    #[allow(dead_code)]
    Incompatible,
}
