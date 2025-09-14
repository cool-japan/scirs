#![allow(deprecated)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(clippy::for_loops_over_fallibles)]
#![allow(unexpected_cfgs)]
#![allow(unused_attributes)]
//! Machine Learning optimization module for SciRS2
//!
//! This module provides optimization algorithms specifically designed for machine learning,
//! including stochastic gradient descent variants, learning rate schedulers, and regularization techniques.
//!
//! # Features
//!
//! - Optimization algorithms: SGD, Adam, RMSprop, etc.
//! - Learning rate schedulers: ExponentialDecay, CosineAnnealing, etc.
//! - Regularization techniques: L1, L2, Dropout, etc.
//!
//! # Examples
//!
//! ## Traditional API
//! ```
//! use ndarray::{Array1, Array2};
//! use scirs2_optim::optimizers::{SGD, Optimizer};
//!
//! // Create a simple optimization problem
//! let params = Array1::zeros(5);
//! let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
//!
//! // Create an optimizer with learning rate 0.01
//! let mut optimizer = SGD::new(0.01);
//!
//! // Update parameters using the optimizer
//! let updated_params = optimizer.step(&params, &gradients);
//! // Parameters should be updated in the negative gradient direction
//! ```
//!
//! ## Unified API (PyTorch-style)
//! ```
//! use ndarray::Array1;
//! use scirs2_optim::{OptimizerConfig, OptimizerFactory, Parameter, UnifiedOptimizer};
//!
//! // Create parameters (similar to PyTorch)
//! let mut param1 = Parameter::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), "layer1.weight");
//! let mut param2 = Parameter::new(Array1::from_vec(vec![0.1, 0.2]), "layer1.bias");
//!
//! // Set gradients
//! param1.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));
//! param2.set_grad(Array1::from_vec(vec![0.05, 0.1]));
//!
//! // Create optimizer with configuration
//! let config = OptimizerConfig::new(0.001f64)
//!     .weight_decay(0.0001)
//!     .grad_clip(1.0);
//! let mut optimizer = OptimizerFactory::adam(config);
//!
//! // Update parameters
//! optimizer.step_param(&mut param1).unwrap();
//! optimizer.step_param(&mut param2).unwrap();
//! ```

#![allow(missing_docs)]

pub mod adaptive_selection;
pub mod autodiff;
pub mod benchmarking;
pub mod curriculum_optimization;
pub mod distributed;
pub mod domain_specific;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod gradient_accumulation;
pub mod gradient_processing;
pub mod hardware_aware;
pub mod learned_optimizers;
pub mod memory_efficient;
pub mod meta_learning;
pub mod metrics;
pub mod neural_architecture_search;
pub mod neural_integration;
pub mod online_learning;
pub mod optimizer_composition;
pub mod optimizers;
pub mod parameter_groups;
pub mod plugin;
pub mod privacy;
pub mod regularizers;
pub mod reinforcement_learning;
pub mod schedulers;
pub mod second_order;
pub mod self_tuning;
pub mod streaming;
pub mod tpu;
pub mod training_stabilization;
pub mod unified_api;
pub mod utils;
pub mod visualization;

// Re-exports for convenience
#[cfg(feature = "gpu")]
pub use gpu::{
    adagrad_gpu::AdagradGpu,
    adam_gpu::AdamGpu,
    adamw_gpu::AdamWGpu,
    lamb_gpu::LAMBGpu,
    memory_pool::{CudaMemoryPool, MemoryPoolConfig, MemoryStats, ThreadSafeMemoryPool},
    mixed_precision::{DynamicLossScaler, MixedPrecisionConfig, MixedPrecisionOptimizer},
    multi_gpu::{MultiGpuConfig, MultiGpuSetup, MultiGpuSync, SyncStrategy},
    rmsprop_gpu::RMSpropGpu,
    rocm_backend::{RocmBackend, RocmConfig},
    sgd_gpu::SGDGpu,
    GpuOptimError, GpuOptimizer, GpuOptimizerConfig,
};
pub use gradient_processing::*;
#[cfg(feature = "metrics_integration")]
pub use metrics::*;
pub use optimizer_composition::*;
pub use optimizers::*;
pub use regularizers::*;
pub use schedulers::*;

// Selective re-exports to avoid conflicts
pub use adaptive_selection::{
    AdaptiveOptimizerSelector, OptimizerStatistics, OptimizerType, PerformanceMetrics,
    ProblemCharacteristics, ProblemType, SelectionNetwork, SelectionStrategy,
};
pub use autodiff::{
    ADNode, AutodiffConfig, AutodiffEngine, AutodiffUtils, GradientCheckResult, GraphStats,
    HessianApproximation, Operation,
};
pub use benchmarking::visualization::{
    OptimizerDashboard, OptimizerStateSnapshot, OptimizerStateVisualizer, VisualizationExport,
};
pub use benchmarking::{
    BenchmarkResult, GradientFlowAnalyzer, GradientFlowStats, OptimizerBenchmark, VisualizationData,
};
pub use curriculum_optimization::{
    AdaptiveCurriculum, AdversarialAttack, AdversarialConfig, CurriculumManager, CurriculumState,
    CurriculumStrategy, ImportanceWeightingStrategy,
};
pub use distributed::{
    AveragingStrategy, CommunicationResult, CompressedGradient, CompressionStrategy,
    DistributedCoordinator, GradientCompressor, ParameterAverager, ParameterServer,
};
pub use domain_specific::{
    CrossDomainKnowledge, DomainOptimizationConfig, DomainPerformanceMetrics, DomainRecommendation,
    DomainSpecificSelector, DomainStrategy, LearningRateScheduleType, OptimizationContext,
    RecommendationType, RegularizationApproach, ResourceConstraints, TrainingConfiguration,
};
pub use gradient_accumulation::{
    AccumulationMode, GradientAccumulator as GradAccumulator, MicroBatchTrainer,
    VariableAccumulator,
};
// Temporarily commented out hardware-aware re-exports to resolve compilation issues
// pub use hardware_aware::{
//     AllReduceAlgorithm, CommunicationStrategy, GPUArchitecture, HardwareAwareOptimizer,
//     HardwareOptimizationConfig, HardwarePerformanceStats, HardwarePlatform, MemoryStrategy,
//     ParallelizationStrategy, PartitionStrategy, PerformanceProfiler, PrecisionStrategy,
//     QuantizationSupport, ResourceMonitor, SIMDSupport, TPUVersion, TuningStrategy,
// };
// Temporarily commented out complex re-exports to resolve compilation issues
// pub use learned_optimizers::optimization_coordinator::{
//     LandscapeFeatures, OptimizationContext as advancedOptimizationContext,
//     OptimizationRecommendation, OptimizerCapabilities, PerformanceSnapshot, AdvancedConfig,
//     AdvancedCoordinator, AdvancedResult,
// };
pub use learned_optimizers::{
    LSTMOptimizer, LearnedOptimizerConfig, LearnedOptimizerMetrics, LearnedOptimizerState,
    MetaOptimizationStrategy, NeuralOptimizerType, TransferResults,
};
pub use meta_learning::{
    AcquisitionFunction, HyperparameterOptimizer, HyperparameterPredictor, HyperparameterStrategy,
    MetaOptimizer, MetaOptimizerTrait, NeuralOptimizer, OptimizationTrajectory, SGDMetaOptimizer,
    UpdateNetwork,
};
pub use neural_architecture_search::{
    ArchitectureEncodingStrategy, EvaluationMetric, MultiObjectiveConfig, NASConfig,
    NeuralArchitectureSearch, OptimizerArchitecture, PerformanceEvaluator, SearchResults,
    SearchStrategyType,
};
pub use neural_integration::architecture_aware::{
    ArchitectureAwareOptimizer, ArchitectureStrategy,
};
pub use neural_integration::forward_backward::{BackwardHook, ForwardHook, NeuralIntegration};
pub use neural_integration::{
    LayerArchitecture, LayerId, OptimizationConfig, ParamId, ParameterManager, ParameterMetadata,
    ParameterOptimizer, ParameterType,
};
pub use plugin::core::{
    create_basic_capabilities, create_plugin_info, OptimizerPluginFactory, PluginCategory,
    PluginInfo,
};
pub use plugin::sdk::{BaseOptimizerPlugin, PluginTester};
pub use plugin::{
    OptimizerPlugin, PluginCapabilities, PluginLoader, PluginRegistry, PluginValidationFramework,
};
// Macro is exported at crate root via #[macro_export]
pub use online_learning::{
    ColumnGrowthStrategy, LearningRateAdaptation, LifelongOptimizer, LifelongStats,
    LifelongStrategy, MemoryExample, MemoryUpdateStrategy, MirrorFunction, OnlineLearningStrategy,
    OnlineOptimizer, OnlinePerformanceMetrics, SharedKnowledge, TaskGraph,
};
pub use privacy::{
    AccountingMethod, ClippingStats, DifferentialPrivacyConfig, DifferentiallyPrivateOptimizer,
    MomentsAccountant, NoiseMechanism, PrivacyBudget, PrivacyValidation,
};
pub use second_order::{HessianInfo, Newton, SecondOrderOptimizer, LBFGS as SecondOrderLBFGS};
pub use self_tuning::{
    OptimizerInfo, OptimizerTrait, PerformanceStats, SelfTuningConfig, SelfTuningOptimizer,
    SelfTuningStatistics, TargetMetric,
};
pub use streaming::{
    LearningRateAdaptation as StreamingLearningRateAdaptation, StreamingConfig, StreamingDataPoint,
    StreamingHealthStatus, StreamingMetrics, StreamingOptimizer,
};
// Temporarily commented out TPU re-exports to resolve compilation issues
// pub use tpu::{
//     PodTopology, TPUConfig, TPUMemoryOptimization, TPUOptimizer, TPUPerformanceMetrics,
//     TPUTopologyInfo, TPUVersion as TPUVersionEnum, XLAOptimizationLevel,
// };
pub use training_stabilization::{AveragingMethod, ModelEnsemble, PolyakAverager, WeightAverager};
pub use unified_api::{
    OptimizerConfig, OptimizerFactory, Parameter, TrainingLoop, UnifiedAdam, UnifiedOptimizer,
    UnifiedSGD,
};
pub use visualization::{
    ColorScheme, ConvergenceInfo, DataSeries, MemoryStats as VisualizationMemoryStats,
    OptimizationMetric, OptimizationVisualizer, OptimizerComparison, PlotType, VisualizationConfig,
};

// Re-export error types
pub use error::{OptimError, OptimizerError, Result};
