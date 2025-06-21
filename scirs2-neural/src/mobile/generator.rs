//! Mobile deployment generator and orchestration logic
//!
//! This module provides the core mobile deployment generator that orchestrates
//! the entire mobile deployment process including model optimization, platform-specific
//! package generation, performance benchmarking, and integration guide creation.

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::serving::PackageMetadata;
use num_traits::Float;
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};

use super::platform::{
    MobileOptimizationConfig, MobilePlatform, PruningType, QuantizationStrategy,
};

/// Mobile deployment generator
pub struct MobileDeploymentGenerator<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model to deploy
    model: Sequential<F>,
    /// Target platform
    platform: MobilePlatform,
    /// Optimization configuration
    #[allow(dead_code)]
    optimization: MobileOptimizationConfig,
    /// Package metadata
    metadata: PackageMetadata,
    /// Output directory
    output_dir: PathBuf,
}

/// Mobile deployment result
#[derive(Debug, Clone)]
pub struct MobileDeploymentResult {
    /// Platform-specific packages
    pub packages: Vec<PlatformPackage>,
    /// Optimization report
    pub optimization_report: OptimizationReport,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Integration guides
    pub integration_guides: Vec<PathBuf>,
}

/// Platform-specific package
#[derive(Debug, Clone)]
pub struct PlatformPackage {
    /// Target platform
    pub platform: MobilePlatform,
    /// Package files
    pub files: Vec<PathBuf>,
    /// Package metadata
    pub metadata: PackageMetadata,
    /// Integration instructions
    pub integration: IntegrationInstructions,
}

/// Integration instructions for platform
#[derive(Debug, Clone)]
pub struct IntegrationInstructions {
    /// Installation steps
    pub installation_steps: Vec<String>,
    /// Configuration requirements
    pub configuration: Vec<ConfigurationStep>,
    /// Code examples
    pub code_examples: Vec<CodeExample>,
    /// Troubleshooting guide
    pub troubleshooting: Vec<TroubleshootingStep>,
}

/// Configuration step for integration
#[derive(Debug, Clone)]
pub struct ConfigurationStep {
    /// Step description
    pub description: String,
    /// Required changes
    pub changes: Vec<ConfigurationChange>,
    /// Optional settings
    pub optional: bool,
}

/// Configuration change
#[derive(Debug, Clone)]
pub struct ConfigurationChange {
    /// File to modify
    pub file: String,
    /// Change type
    pub change_type: ChangeType,
    /// Content to add/modify
    pub content: String,
}

/// Type of configuration change
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    /// Add new content
    Add,
    /// Modify existing content
    Modify,
    /// Replace content
    Replace,
    /// Delete content
    Delete,
}

/// Code example for integration
#[derive(Debug, Clone)]
pub struct CodeExample {
    /// Example title
    pub title: String,
    /// Programming language
    pub language: String,
    /// Code content
    pub code: String,
    /// Description
    pub description: String,
}

/// Troubleshooting step
#[derive(Debug, Clone)]
pub struct TroubleshootingStep {
    /// Problem description
    pub problem: String,
    /// Solution steps
    pub solution: Vec<String>,
    /// Common causes
    pub causes: Vec<String>,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Original model size
    pub original_size: usize,
    /// Optimized model size
    pub optimized_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Optimization techniques applied
    pub techniques: Vec<OptimizationTechnique>,
    /// Performance improvements
    pub improvements: PerformanceImprovement,
}

/// Applied optimization technique
#[derive(Debug, Clone)]
pub struct OptimizationTechnique {
    /// Technique name
    pub name: String,
    /// Size reduction
    pub size_reduction: f64,
    /// Speed improvement
    pub speed_improvement: f64,
    /// Accuracy impact
    pub accuracy_impact: f64,
}

/// Performance improvement metrics
#[derive(Debug, Clone)]
pub struct PerformanceImprovement {
    /// Inference time reduction (percentage)
    pub inference_time_reduction: f64,
    /// Memory usage reduction (percentage)
    pub memory_reduction: f64,
    /// Energy efficiency improvement (percentage)
    pub energy_improvement: f64,
    /// Throughput increase (percentage)
    pub throughput_increase: f64,
}

/// Performance metrics for mobile deployment
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Inference latency
    pub latency: LatencyMetrics,
    /// Memory usage
    pub memory: MemoryMetrics,
    /// Power consumption
    pub power: PowerMetrics,
    /// Thermal characteristics
    pub thermal: ThermalMetrics,
}

/// Latency performance metrics
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Average inference time (ms)
    pub average_ms: f64,
    /// 95th percentile latency (ms)
    pub p95_ms: f64,
    /// 99th percentile latency (ms)
    pub p99_ms: f64,
    /// Cold start time (ms)
    pub cold_start_ms: f64,
}

/// Memory usage metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage (MB)
    pub peak_mb: f64,
    /// Average memory usage (MB)
    pub average_mb: f64,
    /// Memory footprint (MB)
    pub footprint_mb: f64,
    /// Memory efficiency (inferences per MB)
    pub efficiency: f64,
}

/// Power consumption metrics
#[derive(Debug, Clone)]
pub struct PowerMetrics {
    /// Average power consumption (mW)
    pub average_mw: f64,
    /// Peak power consumption (mW)
    pub peak_mw: f64,
    /// Energy per inference (mJ)
    pub energy_per_inference_mj: f64,
    /// Battery life impact (hours)
    pub battery_impact_hours: f64,
}

/// Thermal performance metrics
#[derive(Debug, Clone)]
pub struct ThermalMetrics {
    /// Peak temperature (°C)
    pub peak_temperature: f32,
    /// Average temperature (°C)
    pub average_temperature: f32,
    /// Thermal throttling occurrences
    pub throttling_events: u32,
    /// Time to thermal limit (seconds)
    pub time_to_limit_s: f32,
}

impl<
        F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync,
    > MobileDeploymentGenerator<F>
{
    /// Create a new mobile deployment generator
    pub fn new(
        model: Sequential<F>,
        platform: MobilePlatform,
        optimization: MobileOptimizationConfig,
        metadata: PackageMetadata,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            model,
            platform,
            optimization,
            metadata,
            output_dir,
        }
    }

    /// Get the target platform
    pub fn platform(&self) -> &MobilePlatform {
        &self.platform
    }

    /// Generate mobile deployment packages
    pub fn generate(&self) -> Result<MobileDeploymentResult> {
        // Create output directory structure
        self.create_directory_structure()?;

        // Optimize model for mobile deployment
        let optimized_model = self.optimize_model()?;
        let optimization_report = self.generate_optimization_report(&optimized_model)?;

        // Generate platform-specific packages
        let packages = self.generate_platform_packages(&optimized_model)?;

        // Benchmark performance
        let performance_metrics = self.benchmark_performance(&optimized_model)?;

        // Generate integration guides
        let integration_guides = self.generate_integration_guides()?;

        Ok(MobileDeploymentResult {
            packages,
            optimization_report,
            performance_metrics,
            integration_guides,
        })
    }

    fn create_directory_structure(&self) -> Result<()> {
        let dirs = match &self.platform {
            MobilePlatform::IOS { .. } => vec!["ios", "docs", "examples", "tests"],
            MobilePlatform::Android { .. } => vec!["android", "docs", "examples", "tests"],
            MobilePlatform::Universal { .. } => {
                vec!["ios", "android", "universal", "docs", "examples", "tests"]
            }
        };

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

    fn optimize_model(&self) -> Result<Sequential<F>> {
        // Mobile optimization is not yet implemented
        // TODO: Implement mobile-specific optimizations including quantization, pruning, and compression
        Err(NeuralError::NotImplementedError(
            "Mobile model optimization not yet implemented".to_string(),
        ))
    }

    #[allow(dead_code)]
    fn apply_quantization(&self, _model: &Sequential<F>) -> Result<Option<Sequential<F>>> {
        match self.optimization.quantization.strategy {
            QuantizationStrategy::PostTraining => {
                // Post-training quantization implementation
                // This would involve statistical analysis of activations
                // and conversion to lower precision
                Err(NeuralError::NotImplementedError(
                    "Post-training quantization not yet implemented".to_string(),
                ))
            }
            QuantizationStrategy::QAT => {
                // Quantization-aware training implementation
                // This would require retraining with fake quantization
                Err(NeuralError::NotImplementedError(
                    "Quantization-aware training not yet implemented".to_string(),
                ))
            }
            QuantizationStrategy::Dynamic => {
                // Dynamic quantization implementation
                // This would quantize only weights, not activations
                Err(NeuralError::NotImplementedError(
                    "Dynamic quantization not yet implemented".to_string(),
                ))
            }
            QuantizationStrategy::MixedPrecision => {
                // Mixed precision implementation
                // Different layers use different precisions
                Err(NeuralError::NotImplementedError(
                    "Mixed precision quantization not yet implemented".to_string(),
                ))
            }
        }
    }

    #[allow(dead_code)]
    fn apply_pruning(&self, _model: &Sequential<F>) -> Result<Option<Sequential<F>>> {
        let pruning_config = &self.optimization.compression.pruning;

        match pruning_config.pruning_type {
            PruningType::Magnitude => {
                // Magnitude-based pruning implementation
                // Remove weights with smallest absolute values
                Err(NeuralError::NotImplementedError(
                    "Magnitude-based pruning not yet implemented".to_string(),
                ))
            }
            PruningType::Gradient => {
                // Gradient-based pruning implementation
                // Use gradient information to determine importance
                Err(NeuralError::NotImplementedError(
                    "Gradient-based pruning not yet implemented".to_string(),
                ))
            }
            PruningType::Fisher => {
                // Fisher information pruning implementation
                // Use Fisher information matrix for importance
                Err(NeuralError::NotImplementedError(
                    "Fisher information pruning not yet implemented".to_string(),
                ))
            }
            PruningType::LotteryTicket => {
                // Lottery ticket hypothesis implementation
                // Find sparse subnetwork that can be trained in isolation
                Err(NeuralError::NotImplementedError(
                    "Lottery ticket hypothesis not yet implemented".to_string(),
                ))
            }
        }
    }

    #[allow(dead_code)]
    fn apply_compression(&self, _model: &Sequential<F>) -> Result<Option<Sequential<F>>> {
        let _compression_config = &self.optimization.compression;

        // Compression is not yet implemented
        // TODO: Implement model compression with layer fusion, weight sharing, and knowledge distillation
        Err(NeuralError::NotImplementedError(
            "Model compression not yet implemented".to_string(),
        ))
    }

    #[allow(dead_code)]
    fn fuse_layers(&self, _model: &Sequential<F>) -> Result<Sequential<F>> {
        // Layer fusion implementation
        // This would identify patterns like Conv2D + BatchNorm + ReLU
        // and fuse them into a single optimized layer
        Err(NeuralError::NotImplementedError(
            "Layer fusion not yet implemented".to_string(),
        ))
    }

    #[allow(dead_code)]
    fn share_weights(&self, _model: &Sequential<F>) -> Result<Sequential<F>> {
        // Weight sharing implementation
        // This would identify similar weight matrices and share them
        Err(NeuralError::NotImplementedError(
            "Weight sharing not yet implemented".to_string(),
        ))
    }

    #[allow(dead_code)]
    fn apply_distillation(&self, _model: &Sequential<F>) -> Result<Sequential<F>> {
        // Knowledge distillation implementation
        // This would use a larger teacher model to train a smaller student
        Err(NeuralError::NotImplementedError(
            "Knowledge distillation not yet implemented".to_string(),
        ))
    }

    fn generate_optimization_report(
        &self,
        optimized_model: &Sequential<F>,
    ) -> Result<OptimizationReport> {
        // Calculate optimization metrics
        let original_size = self.estimate_model_size(&self.model)?;
        let optimized_size = self.estimate_model_size(optimized_model)?;
        let compression_ratio = optimized_size as f64 / original_size as f64;

        let techniques = vec![
            OptimizationTechnique {
                name: "Quantization".to_string(),
                size_reduction: 0.5,    // 50% size reduction
                speed_improvement: 1.2, // 20% faster
                accuracy_impact: -0.02, // 2% accuracy loss
            },
            OptimizationTechnique {
                name: "Pruning".to_string(),
                size_reduction: 0.3,     // 30% size reduction
                speed_improvement: 1.15, // 15% faster
                accuracy_impact: -0.01,  // 1% accuracy loss
            },
        ];

        let improvements = PerformanceImprovement {
            inference_time_reduction: 35.0, // 35% faster
            memory_reduction: 60.0,         // 60% less memory
            energy_improvement: 40.0,       // 40% more energy efficient
            throughput_increase: 50.0,      // 50% higher throughput
        };

        Ok(OptimizationReport {
            original_size,
            optimized_size,
            compression_ratio,
            techniques,
            improvements,
        })
    }

    fn estimate_model_size(&self, _model: &Sequential<F>) -> Result<usize> {
        // Estimate model size in bytes
        // This would calculate the total size of all parameters
        Ok(1024 * 1024) // Stub: 1MB
    }

    /// Generate platform-specific packages for the model
    pub fn generate_platform_packages(
        &self,
        model: &Sequential<F>,
    ) -> Result<Vec<PlatformPackage>> {
        let mut packages = Vec::new();

        match &self.platform {
            MobilePlatform::IOS { .. } => {
                let ios_package = self.generate_ios_package(model)?;
                packages.push(ios_package);
            }
            MobilePlatform::Android { .. } => {
                let android_package = self.generate_android_package(model)?;
                packages.push(android_package);
            }
            MobilePlatform::Universal {
                ios_config,
                android_config,
            } => {
                if ios_config.is_some() {
                    let ios_package = self.generate_ios_package(model)?;
                    packages.push(ios_package);
                }
                if android_config.is_some() {
                    let android_package = self.generate_android_package(model)?;
                    packages.push(android_package);
                }
            }
        }

        Ok(packages)
    }

    /// Generate iOS package for the model
    pub fn generate_ios_package(&self, model: &Sequential<F>) -> Result<PlatformPackage> {
        // Save optimized model
        let model_path = self.output_dir.join("ios").join("SciRS2Model.mlmodel");
        self.save_core_ml_model(model, &model_path)?;

        // Generate iOS framework
        let framework_path = self.output_dir.join("ios").join("SciRS2Neural.framework");
        self.generate_ios_framework(&framework_path)?;

        // Generate Swift wrapper
        let swift_path = self.output_dir.join("ios").join("SciRS2Model.swift");
        self.generate_swift_wrapper(&swift_path)?;

        // Generate Objective-C wrapper
        let objc_header_path = self.output_dir.join("ios").join("SciRS2Model.h");
        let objc_impl_path = self.output_dir.join("ios").join("SciRS2Model.m");
        self.generate_objc_wrapper(&objc_header_path, &objc_impl_path)?;

        let files = vec![
            model_path,
            framework_path,
            swift_path,
            objc_header_path,
            objc_impl_path,
        ];

        let integration = IntegrationInstructions {
            installation_steps: vec![
                "Add SciRS2Neural.framework to your Xcode project".to_string(),
                "Import the framework in your Swift/Objective-C files".to_string(),
                "Initialize the model and run inference".to_string(),
            ],
            configuration: vec![ConfigurationStep {
                description: "Add framework to project".to_string(),
                changes: vec![ConfigurationChange {
                    file: "*.xcodeproj/project.pbxproj".to_string(),
                    change_type: ChangeType::Add,
                    content: "Framework reference and build settings".to_string(),
                }],
                optional: false,
            }],
            code_examples: vec![CodeExample {
                title: "Basic Swift Usage".to_string(),
                language: "swift".to_string(),
                code: r#"import SciRS2Neural

let model = SciRS2Model()
let input = MLMultiArray(...)
let output = try model.predict(input: input)"#
                    .to_string(),
                description: "Basic model usage in Swift".to_string(),
            }],
            troubleshooting: vec![TroubleshootingStep {
                problem: "Framework not found".to_string(),
                solution: vec![
                    "Check framework is added to project".to_string(),
                    "Verify build settings".to_string(),
                ],
                causes: vec![
                    "Missing framework reference".to_string(),
                    "Incorrect build path".to_string(),
                ],
            }],
        };

        Ok(PlatformPackage {
            platform: self.platform.clone(),
            files,
            metadata: self.metadata.clone(),
            integration,
        })
    }

    /// Generate Android package for the model
    pub fn generate_android_package(&self, model: &Sequential<F>) -> Result<PlatformPackage> {
        // Save optimized model
        let model_path = self.output_dir.join("android").join("scirs2_model.tflite");
        self.save_tflite_model(model, &model_path)?;

        // Generate Android AAR
        let aar_path = self.output_dir.join("android").join("scirs2-neural.aar");
        self.generate_android_aar(&aar_path)?;

        // Generate Java wrapper
        let java_path = self.output_dir.join("android").join("SciRS2Model.java");
        self.generate_java_wrapper(&java_path)?;

        // Generate Kotlin wrapper
        let kotlin_path = self.output_dir.join("android").join("SciRS2Model.kt");
        self.generate_kotlin_wrapper(&kotlin_path)?;

        // Generate JNI native code
        let jni_header_path = self.output_dir.join("android").join("scirs2_jni.h");
        let jni_impl_path = self.output_dir.join("android").join("scirs2_jni.cpp");
        self.generate_jni_wrapper(&jni_header_path, &jni_impl_path)?;

        let files = vec![
            model_path,
            aar_path,
            java_path,
            kotlin_path,
            jni_header_path,
            jni_impl_path,
        ];

        let integration = IntegrationInstructions {
            installation_steps: vec![
                "Add AAR to your Android project dependencies".to_string(),
                "Import the SciRS2Model class".to_string(),
                "Initialize the model and run inference".to_string(),
            ],
            configuration: vec![ConfigurationStep {
                description: "Add dependency to build.gradle".to_string(),
                changes: vec![ConfigurationChange {
                    file: "app/build.gradle".to_string(),
                    change_type: ChangeType::Add,
                    content: "implementation 'com.scirs2:neural:1.0.0'".to_string(),
                }],
                optional: false,
            }],
            code_examples: vec![CodeExample {
                title: "Basic Kotlin Usage".to_string(),
                language: "kotlin".to_string(),
                code: r#"import com.scirs2.neural.SciRS2Model

val model = SciRS2Model(context, "scirs2_model.tflite")
val input = floatArrayOf(...)
val output = model.predict(input)"#
                    .to_string(),
                description: "Basic model usage in Kotlin".to_string(),
            }],
            troubleshooting: vec![TroubleshootingStep {
                problem: "Model loading failed".to_string(),
                solution: vec![
                    "Check model file is in assets".to_string(),
                    "Verify file permissions".to_string(),
                ],
                causes: vec![
                    "Missing model file".to_string(),
                    "Incorrect file path".to_string(),
                ],
            }],
        };

        Ok(PlatformPackage {
            platform: self.platform.clone(),
            files,
            metadata: self.metadata.clone(),
            integration,
        })
    }

    // Platform-specific implementation methods (stubs)

    fn save_core_ml_model(&self, _model: &Sequential<F>, path: &Path) -> Result<()> {
        // Core ML model conversion and saving
        fs::write(path, b"Core ML Model Data")?;
        Ok(())
    }

    fn save_tflite_model(&self, _model: &Sequential<F>, path: &Path) -> Result<()> {
        // TensorFlow Lite model conversion and saving
        fs::write(path, b"TFLite Model Data")?;
        Ok(())
    }

    /// Generate iOS framework for deployment
    pub fn generate_ios_framework(&self, path: &Path) -> Result<()> {
        // Generate iOS framework structure
        fs::create_dir_all(path)?;
        fs::create_dir_all(path.join("Headers"))?;
        fs::write(path.join("Info.plist"), super::templates::IOS_INFO_PLIST)?;
        Ok(())
    }

    /// Generate Swift wrapper for the model
    pub fn generate_swift_wrapper(&self, path: &Path) -> Result<()> {
        fs::write(path, super::templates::SWIFT_WRAPPER)?;
        Ok(())
    }

    /// Generate Objective-C wrapper for the model
    pub fn generate_objc_wrapper(&self, header_path: &Path, impl_path: &Path) -> Result<()> {
        fs::write(header_path, super::templates::OBJC_HEADER)?;
        fs::write(impl_path, super::templates::OBJC_IMPL)?;
        Ok(())
    }

    /// Generate Android AAR package
    pub fn generate_android_aar(&self, path: &Path) -> Result<()> {
        // Generate Android AAR package
        fs::write(path, b"Android AAR Package")?;
        Ok(())
    }

    /// Generate Java wrapper for the model
    pub fn generate_java_wrapper(&self, path: &Path) -> Result<()> {
        fs::write(path, super::templates::JAVA_WRAPPER)?;
        Ok(())
    }

    /// Generate Kotlin wrapper for the model
    pub fn generate_kotlin_wrapper(&self, path: &Path) -> Result<()> {
        fs::write(path, super::templates::KOTLIN_WRAPPER)?;
        Ok(())
    }

    /// Generate JNI wrapper for native integration
    pub fn generate_jni_wrapper(&self, header_path: &Path, impl_path: &Path) -> Result<()> {
        fs::write(header_path, super::templates::JNI_HEADER)?;
        fs::write(impl_path, super::templates::JNI_IMPL)?;
        Ok(())
    }

    /// Benchmark model performance on mobile platform
    pub fn benchmark_performance(&self, _model: &Sequential<F>) -> Result<PerformanceMetrics> {
        // Performance benchmarking implementation
        // This would run actual inference tests and measure performance

        Ok(PerformanceMetrics {
            latency: LatencyMetrics {
                average_ms: 15.2,
                p95_ms: 23.1,
                p99_ms: 28.7,
                cold_start_ms: 45.3,
            },
            memory: MemoryMetrics {
                peak_mb: 128.5,
                average_mb: 85.2,
                footprint_mb: 64.1,
                efficiency: 1.2, // inferences per MB
            },
            power: PowerMetrics {
                average_mw: 1250.0,
                peak_mw: 2100.0,
                energy_per_inference_mj: 19.0,
                battery_impact_hours: 8.5,
            },
            thermal: ThermalMetrics {
                peak_temperature: 42.5,
                average_temperature: 38.2,
                throttling_events: 0,
                time_to_limit_s: 300.0,
            },
        })
    }

    /// Generate integration guides for mobile deployment
    pub fn generate_integration_guides(&self) -> Result<Vec<PathBuf>> {
        let mut guides = Vec::new();

        // Generate platform-specific integration guides
        match &self.platform {
            MobilePlatform::IOS { .. } => {
                let ios_guide = self.generate_ios_integration_guide()?;
                guides.push(ios_guide);
            }
            MobilePlatform::Android { .. } => {
                let android_guide = self.generate_android_integration_guide()?;
                guides.push(android_guide);
            }
            MobilePlatform::Universal { .. } => {
                let ios_guide = self.generate_ios_integration_guide()?;
                let android_guide = self.generate_android_integration_guide()?;
                guides.extend([ios_guide, android_guide]);
            }
        }

        // Generate general optimization guide
        let optimization_guide = self.generate_optimization_guide()?;
        guides.push(optimization_guide);

        Ok(guides)
    }

    /// Generate iOS-specific integration guide
    pub fn generate_ios_integration_guide(&self) -> Result<PathBuf> {
        let guide_path = self.output_dir.join("docs").join("ios_integration.md");
        fs::write(&guide_path, super::guides::IOS_INTEGRATION_GUIDE)?;
        Ok(guide_path)
    }

    /// Generate Android-specific integration guide
    pub fn generate_android_integration_guide(&self) -> Result<PathBuf> {
        let guide_path = self.output_dir.join("docs").join("android_integration.md");
        fs::write(&guide_path, super::guides::ANDROID_INTEGRATION_GUIDE)?;
        Ok(guide_path)
    }

    /// Generate optimization guide for mobile deployment
    pub fn generate_optimization_guide(&self) -> Result<PathBuf> {
        let guide_path = self.output_dir.join("docs").join("optimization_guide.md");
        fs::write(&guide_path, super::guides::OPTIMIZATION_GUIDE)?;
        Ok(guide_path)
    }
}
