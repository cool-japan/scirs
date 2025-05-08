//! Pre-defined neural network architectures
//!
//! This module provides implementations of popular neural network architectures
//! for computer vision, natural language processing, and other domains.

pub mod bert;
pub mod clip;
pub mod convnext;
pub mod efficientnet;
pub mod fusion;
pub mod gpt;
pub mod mobilenet;
pub mod resnet;
pub mod seq2seq;
pub mod vit;

pub use bert::{BertConfig, BertModel};
pub use clip::{CLIPConfig, CLIPTextConfig, CLIPTextEncoder, CLIPVisionEncoder, CLIP};
pub use convnext::{ConvNeXt, ConvNeXtBlock, ConvNeXtConfig, ConvNeXtStage, ConvNeXtVariant};
pub use efficientnet::{EfficientNet, EfficientNetConfig, EfficientNetStage, MBConvConfig};
pub use fusion::{
    BilinearFusion, CrossModalAttention, FeatureAlignment, FeatureFusion, FeatureFusionConfig,
    FiLMModule, FusionMethod,
};
pub use gpt::{GPTConfig, GPTModel};
pub use mobilenet::{MobileNet, MobileNetConfig, MobileNetVersion};
pub use resnet::{ResNet, ResNetBlock, ResNetConfig, ResNetLayer};
pub use seq2seq::{
    Attention, AttentionType, RNNCellType, Seq2Seq, Seq2SeqConfig, Seq2SeqDecoder, Seq2SeqEncoder,
};
pub use vit::{ViTConfig, VisionTransformer};
