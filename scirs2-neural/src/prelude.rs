//! Common neural network functionality
//!
//! This module re-exports the most commonly used types and traits
//! to provide a convenient single import for users of the library.

pub use crate::activations::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU};
pub use crate::callbacks::{
    EarlyStopping, GradientClipping, GradientClippingMethod, ModelCheckpoint,
};
pub use crate::error::{Error, Result};
pub use crate::evaluation::{
    CrossValidationConfig, CrossValidator, EarlyStoppingConfig, EarlyStoppingMode,
    EvaluationConfig, Evaluator, Metric, MetricType, TestConfig, TestEvaluator, ValidationConfig,
    ValidationHandler,
};
pub use crate::layers::{
    ActivityRegularization, AdaptiveAvgPool2D, AdaptiveMaxPool2D, Dense, Dropout,
    L1ActivityRegularization, L2ActivityRegularization, Layer, LayerConfig, Sequential,
};
pub use crate::losses::{ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, TripletLoss};
pub use crate::models::Model;
pub use crate::optimizers::{Adam, Optimizer, RMSprop, SGD};
pub use crate::training::{
    GradientAccumulationConfig, GradientAccumulator, GradientStats, MixedPrecisionConfig,
    MixedPrecisionManager, Trainer, TrainingConfig, TrainingSession, ValidationSettings,
};
pub use crate::transformer::{TransformerDecoderLayer, TransformerEncoderLayer};
pub use crate::utils::positional_encoding::{PositionalEncoding, SinusoidalPositionalEncoding};

// Performance optimizations
pub use crate::performance::{
    OptimizationCapabilities, PerformanceOptimizer, PerformanceProfiler, ThreadPoolManager,
};

// JIT compilation - module not yet implemented
// pub use crate::performance::jit::{
//     CompiledElementOp, CompiledMatrixOp, JitContext, JitProfiler, JitStats, JitStrategy,
// };

// Data augmentation
pub use crate::augmentation::{
    AudioAugmentation, AugmentationManager, AugmentationPipelineBuilder, FillMode,
    ImageAugmentation, MixAugmentation, TextAugmentation,
};

// Enhanced evaluation tools
pub use crate::model_evaluation::{
    AveragingMethod, ClassificationMetric, CrossValidationStrategy, EvaluationBuilder,
    EvaluationMetric, ModelEvaluator, RegressionMetric,
};

// Model compression
pub use crate::compression::{
    CalibrationMethod, CompressionAnalyzer, ModelPruner, PostTrainingQuantizer, PruningMethod,
    QuantizationBits, QuantizationScheme,
};

// Knowledge distillation
pub use crate::distillation::{DistillationMethod, DistillationTrainer, FeatureAdaptation};

// Transfer learning
pub use crate::transfer_learning::{LayerState, TransferLearningManager, TransferStrategy};

// Model interpretation
pub use crate::interpretation::{
    AttributionMethod, BaselineMethod, ModelInterpreter, VisualizationMethod,
};

// Architecture specific imports
pub use crate::models::architectures::{
    AttentionType, BertConfig, BertModel, CLIPConfig, CLIPTextConfig, ConvNeXt, ConvNeXtConfig,
    ConvNeXtVariant, EfficientNet, EfficientNetConfig, FeatureFusion, FeatureFusionConfig,
    FusionMethod, GPTConfig, GPTModel, MobileNet, MobileNetConfig, MobileNetVersion, RNNCellType,
    ResNet, ResNetConfig, Seq2Seq, Seq2SeqConfig, ViTConfig, VisionTransformer, CLIP,
};
