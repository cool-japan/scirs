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
    CrossValidationConfig, CrossValidationStrategy, CrossValidator, EarlyStoppingConfig,
    EarlyStoppingMode, EvaluationConfig, Evaluator, Metric, MetricType, TestConfig, TestEvaluator,
    ValidationConfig, ValidationHandler,
};
pub use crate::layers::{Dense, Dropout, Layer, LayerConfig, Sequential};
pub use crate::losses::{ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, TripletLoss};
pub use crate::models::Model;
pub use crate::optimizers::{Adam, Optimizer, RMSprop, SGD};
pub use crate::training::{
    GradientAccumulationConfig, GradientAccumulator, GradientStats, MixedPrecisionConfig,
    MixedPrecisionManager, Trainer, TrainingConfig, TrainingSession, ValidationSettings,
};
pub use crate::transformer::{TransformerDecoderLayer, TransformerEncoderLayer};
pub use crate::utils::positional_encoding::{PositionalEncoding, SinusoidalPositionalEncoding};

// Architecture specific imports
pub use crate::models::architectures::{
    AttentionType, BertConfig, BertModel, CLIPConfig, CLIPTextConfig, ConvNeXt, ConvNeXtConfig,
    ConvNeXtVariant, EfficientNet, EfficientNetConfig, FeatureFusion, FeatureFusionConfig,
    FusionMethod, GPTConfig, GPTModel, MobileNet, MobileNetConfig, MobileNetVersion, RNNCellType,
    ResNet, ResNetConfig, Seq2Seq, Seq2SeqConfig, ViTConfig, VisionTransformer, CLIP,
};
