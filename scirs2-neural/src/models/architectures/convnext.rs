//! ConvNeXt architecture implementation
//!
//! This module implements the ConvNeXt architecture as described in
//! "A ConvNet for the 2020s" (<https://arxiv.org/abs/2201.03545>)
//! ConvNeXt modernizes ResNet architecture by incorporating design choices from
//! Vision Transformers, resulting in a pure convolutional model with excellent performance.

use crate::activations::GELU;
use crate::error::{NeuralError, Result};
use crate::layers::conv::PaddingMode;
use crate::layers::{Conv2D, Dense, Dropout, GlobalAvgPool2D, Layer, LayerNorm2D, Sequential};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Configuration for a ConvNeXt stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvNeXtStageConfig {
    /// Number of input channels
    pub input_channels: usize,
    /// Number of output channels
    pub output_channels: usize,
    /// Number of blocks in this stage
    pub num_blocks: usize,
    /// Stride for the first block (typically 2 for downsampling, 1 otherwise)
    pub stride: usize,
    /// Layer scale initialization value (typically 1e-6)
    pub layer_scale_init_value: f64,
    /// Dropout probability
    pub drop_path_prob: f64,
}

/// Configuration for a ConvNeXt model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvNeXtConfig {
    /// Model depth variant (Tiny, Small, Base, Large, XLarge)
    pub variant: ConvNeXtVariant,
    /// Number of input channels (typically 3 for RGB images)
    pub input_channels: usize,
    /// Depths for each stage
    pub depths: Vec<usize>,
    /// Dimensions (channels) for each stage
    pub dims: Vec<usize>,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate
    pub dropout_rate: Option<f64>,
    /// Layer scale initialization value
    pub layer_scale_init_value: f64,
    /// Whether to include the classification head
    pub include_top: bool,
}

impl Default for ConvNeXtConfig {
    fn default() -> Self {
        Self {
            variant: ConvNeXtVariant::Tiny,
            input_channels: 3,
            depths: vec![3, 3, 9, 3],
            dims: vec![96, 192, 384, 768],
            num_classes: 1000,
            dropout_rate: Some(0.0),
            layer_scale_init_value: 1e-6,
            include_top: true,
        }
    }
}

/// ConvNeXt model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvNeXtVariant {
    /// ConvNeXt-Tiny
    Tiny,
    /// ConvNeXt-Small
    Small,
    /// ConvNeXt-Base
    Base,
    /// ConvNeXt-Large
    Large,
    /// ConvNeXt-XLarge
    XLarge,
}

/// ConvNeXt residual block.
///
/// Each block applies: depthwise 7×7 conv → LayerNorm2D → 1×1 conv (×4 channels) →
/// GELU → 1×1 conv (back to original channels) → layer scale → skip connection.
#[derive(Debug, Clone)]
pub struct ConvNeXtBlock<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Depthwise convolution (7×7, same-padding)
    pub depthwise_conv: Conv2D<F>,
    /// Layer normalization over spatial dims
    pub norm: LayerNorm2D<F>,
    /// Pointwise convolution 1 (channels → channels×4)
    pub pointwise_conv1: Conv2D<F>,
    /// GELU activation
    pub gelu: GELU,
    /// Pointwise convolution 2 (channels×4 → channels)
    pub pointwise_conv2: Conv2D<F>,
    /// Layer scale gamma parameter, shape `[channels]`
    pub gamma: Array<F, IxDyn>,
    /// Whether to apply stochastic-depth scaling
    pub use_skip: bool,
    /// Scale factor for stochastic depth: `1 - drop_path_prob`
    pub skip_scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ConvNeXtBlock<F> {
    /// Create a new ConvNeXtBlock.
    pub fn new(channels: usize, layer_scale_init_value: f64, drop_path_prob: f64) -> Result<Self> {
        let depthwise_conv = Conv2D::<F>::new(channels, channels, (7, 7), (1, 1), None)
            .map(|c| c.with_padding(PaddingMode::Custom(3)))?;

        let norm = LayerNorm2D::<F>::new::<SmallRng>(channels, 1e-6, Some("norm"))?;

        let pointwise_conv1 = Conv2D::<F>::new(channels, channels * 4, (1, 1), (1, 1), None)
            .map(|c| c.with_padding(PaddingMode::Custom(0)))?;

        let gelu = GELU::new();

        let pointwise_conv2 = Conv2D::<F>::new(channels * 4, channels, (1, 1), (1, 1), None)
            .map(|c| c.with_padding(PaddingMode::Custom(0)))?;

        let gamma_value = F::from(layer_scale_init_value).ok_or_else(|| {
            NeuralError::InvalidArchitecture(
                "ConvNeXtBlock: failed to convert layer_scale_init_value to float".to_string(),
            )
        })?;
        let gamma = Array::<F, _>::from_elem(IxDyn(&[channels]), gamma_value);

        let skip_scale = F::from(1.0 - drop_path_prob).ok_or_else(|| {
            NeuralError::InvalidArchitecture(
                "ConvNeXtBlock: failed to convert drop_path_prob to float".to_string(),
            )
        })?;
        let use_skip = drop_path_prob > 0.0;

        Ok(Self {
            depthwise_conv,
            norm,
            pointwise_conv1,
            gelu,
            pointwise_conv2,
            gamma,
            use_skip,
            skip_scale,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for ConvNeXtBlock<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let identity = input.clone();

        // Depthwise conv → LayerNorm2D
        let mut x = self.depthwise_conv.forward(input)?;
        x = self.norm.forward(&x)?;

        // Pointwise expand → GELU → pointwise project
        x = self.pointwise_conv1.forward(&x)?;
        x = <GELU as Layer<F>>::forward(&self.gelu, &x)?;
        x = self.pointwise_conv2.forward(&x)?;

        // Apply layer scale: broadcast gamma [C] over [N,C,H,W]
        let shape = x.shape().to_vec();
        if shape.len() == 4 {
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            for ni in 0..n {
                for ci in 0..c {
                    let g = self.gamma[ci];
                    for hi in 0..h {
                        for wi in 0..w {
                            x[[ni, ci, hi, wi]] *= g;
                        }
                    }
                }
            }
        }

        // Stochastic depth
        if self.use_skip {
            x *= self.skip_scale;
        }

        Ok(x + identity)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let mut grad = grad_output.clone();
        let grad_skip = grad.clone();

        if self.use_skip {
            grad *= self.skip_scale;
        }

        // Undo layer scale
        let shape = grad.shape().to_vec();
        if shape.len() == 4 {
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            for ni in 0..n {
                for ci in 0..c {
                    let g = self.gamma[ci];
                    for hi in 0..h {
                        for wi in 0..w {
                            grad[[ni, ci, hi, wi]] *= g;
                        }
                    }
                }
            }
        }

        let grad_after_conv2 = self.pointwise_conv2.backward(&grad, &grad)?;
        let grad_after_gelu = grad_after_conv2.clone();
        let grad_after_conv1 = self
            .pointwise_conv1
            .backward(&grad_after_gelu, &grad_after_gelu)?;
        let grad_after_norm = self.norm.backward(&grad_after_conv1, &grad_after_conv1)?;
        let grad_after_dwconv = self.depthwise_conv.backward(input, &grad_after_norm)?;

        Ok(grad_after_dwconv + grad_skip)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.depthwise_conv.update(learning_rate)?;
        self.norm.update(learning_rate)?;
        self.pointwise_conv1.update(learning_rate)?;
        self.pointwise_conv2.update(learning_rate)?;

        // Small gradient-like update on gamma (simplified)
        let small_update = F::from(0.0001_f64).ok_or_else(|| {
            NeuralError::InvalidArchitecture(
                "ConvNeXtBlock: failed to convert small_update to float".to_string(),
            )
        })? * learning_rate;
        for elem in self.gamma.iter_mut() {
            *elem -= small_update;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.depthwise_conv.params());
        params.extend(self.norm.params());
        params.extend(self.pointwise_conv1.params());
        params.extend(self.pointwise_conv2.params());
        params.push(self.gamma.clone());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.depthwise_conv.set_training(training);
        self.norm.set_training(training);
        self.pointwise_conv1.set_training(training);
        self.pointwise_conv2.set_training(training);
        <GELU as Layer<F>>::set_training(&mut self.gelu, training);
    }

    fn is_training(&self) -> bool {
        self.depthwise_conv.is_training()
    }

    fn layer_type(&self) -> &str {
        "ConvNeXtBlock"
    }
}

/// ConvNeXt downsampling layer: LayerNorm2D followed by a strided convolution.
#[derive(Debug, Clone)]
pub struct ConvNeXtDownsample<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Layer normalization before convolution
    pub norm: LayerNorm2D<F>,
    /// Strided convolution for spatial downsampling
    pub conv: Conv2D<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ConvNeXtDownsample<F> {
    /// Create a new ConvNeXtDownsample.
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Result<Self> {
        let norm = LayerNorm2D::<F>::new::<SmallRng>(in_channels, 1e-6, Some("downsample_norm"))?;
        let conv = Conv2D::<F>::new(
            in_channels,
            out_channels,
            (stride, stride),
            (stride, stride),
            None,
        )
        .map(|c| c.with_padding(PaddingMode::Custom(0)))?;
        Ok(Self { norm, conv })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for ConvNeXtDownsample<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let x = self.norm.forward(input)?;
        self.conv.forward(&x)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let grad_after_conv = self.conv.backward(grad_output, grad_output)?;
        self.norm.backward(input, &grad_after_conv)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.norm.update(learning_rate)?;
        self.conv.update(learning_rate)?;
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.norm.params());
        params.extend(self.conv.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.norm.set_training(training);
        self.conv.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.norm.is_training()
    }

    fn layer_type(&self) -> &str {
        "ConvNeXtDownsample"
    }
}

/// A single ConvNeXt stage: optional downsampling layer followed by ConvNeXt blocks.
#[derive(Debug, Clone)]
pub struct ConvNeXtStage<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Optional downsampling layer (present when channels change or stride > 1)
    pub downsample: Option<ConvNeXtDownsample<F>>,
    /// ConvNeXt residual blocks
    pub blocks: Vec<ConvNeXtBlock<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ConvNeXtStage<F> {
    /// Create a new ConvNeXtStage.
    pub fn new(config: &ConvNeXtStageConfig) -> Result<Self> {
        let downsample = if config.input_channels != config.output_channels || config.stride > 1 {
            Some(ConvNeXtDownsample::<F>::new(
                config.input_channels,
                config.output_channels,
                config.stride,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            blocks.push(ConvNeXtBlock::<F>::new(
                config.output_channels,
                config.layer_scale_init_value,
                config.drop_path_prob,
            )?);
        }

        Ok(Self { downsample, blocks })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for ConvNeXtStage<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = if let Some(ref ds) = self.downsample {
            ds.forward(input)?
        } else {
            input.clone()
        };
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let mut grad = grad_output.clone();
        for block in self.blocks.iter().rev() {
            grad = block.backward(&grad, &grad)?;
        }
        if let Some(ref ds) = self.downsample {
            grad = ds.backward(input, &grad)?;
        }
        Ok(grad)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        if let Some(ref mut ds) = self.downsample {
            ds.update(learning_rate)?;
        }
        for block in &mut self.blocks {
            block.update(learning_rate)?;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        if let Some(ref ds) = self.downsample {
            params.extend(ds.params());
        }
        for block in &self.blocks {
            params.extend(block.params());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        if let Some(ref mut ds) = self.downsample {
            ds.set_training(training);
        }
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        if let Some(ref ds) = self.downsample {
            return ds.is_training();
        }
        if !self.blocks.is_empty() {
            return self.blocks[0].is_training();
        }
        true
    }

    fn layer_type(&self) -> &str {
        "ConvNeXtStage"
    }
}

/// Full ConvNeXt model: stem → stages → optional classification head.
#[derive(Debug)]
pub struct ConvNeXt<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Stem layer (4×4 strided conv + LayerNorm2D)
    pub stem: Sequential<F>,
    /// Main stages of the network
    pub stages: Vec<ConvNeXtStage<F>>,
    /// Classification head (if include_top is true)
    pub head: Option<Sequential<F>>,
    /// Model configuration
    pub config: ConvNeXtConfig,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static,
    > ConvNeXt<F>
{
    /// Create a new ConvNeXt model from config.
    pub fn new(config: ConvNeXtConfig) -> Result<Self> {
        let mut rng = SmallRng::from_seed([99u8; 32]);

        // Stem: 4×4 strided conv + LayerNorm2D
        let mut stem = Sequential::new();
        stem.add(
            Conv2D::<F>::new(config.input_channels, config.dims[0], (4, 4), (4, 4), None)
                .map(|c| c.with_padding(PaddingMode::Custom(0)))?,
        );
        stem.add(LayerNorm2D::<F>::new::<SmallRng>(
            config.dims[0],
            1e-6,
            Some("stem_norm"),
        )?);

        // Stages
        let mut stages = Vec::with_capacity(config.depths.len());
        let mut current_channels = config.dims[0];

        for (i, &depth) in config.depths.iter().enumerate() {
            let output_channels = config.dims[i];
            let stride = if i == 0 { 1 } else { 2 };

            let stage_config = ConvNeXtStageConfig {
                input_channels: current_channels,
                output_channels,
                num_blocks: depth,
                stride,
                layer_scale_init_value: config.layer_scale_init_value,
                drop_path_prob: 0.0,
            };

            stages.push(ConvNeXtStage::<F>::new(&stage_config)?);
            current_channels = output_channels;
        }

        // Head
        let head = if config.include_top {
            let last_dim = *config.dims.last().ok_or_else(|| {
                NeuralError::InvalidArchitecture("ConvNeXt: dims must be non-empty".to_string())
            })?;
            let mut head_seq = Sequential::new();
            head_seq.add(LayerNorm2D::<F>::new::<SmallRng>(
                last_dim,
                1e-6,
                Some("head_norm"),
            )?);
            // GlobalAvgPool2D::new returns Self (not Result), so no `?`
            head_seq.add(GlobalAvgPool2D::<F>::new(Some("head_pool")));
            if let Some(dropout_rate) = config.dropout_rate {
                if dropout_rate > 0.0 {
                    head_seq.add(Dropout::<F>::new(dropout_rate, &mut rng)?);
                }
            }
            head_seq.add(Dense::<F>::new(
                last_dim,
                config.num_classes,
                Some("classifier"),
                &mut rng,
            )?);
            Some(head_seq)
        } else {
            None
        };

        Ok(Self {
            stem,
            stages,
            head,
            config,
        })
    }

    /// Create a ConvNeXt-Tiny model.
    pub fn convnext_tiny(num_classes: usize, include_top: bool) -> Result<Self> {
        Self::new(ConvNeXtConfig {
            variant: ConvNeXtVariant::Tiny,
            input_channels: 3,
            depths: vec![3, 3, 9, 3],
            dims: vec![96, 192, 384, 768],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        })
    }

    /// Create a ConvNeXt-Small model.
    pub fn convnext_small(num_classes: usize, include_top: bool) -> Result<Self> {
        Self::new(ConvNeXtConfig {
            variant: ConvNeXtVariant::Small,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![96, 192, 384, 768],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        })
    }

    /// Create a ConvNeXt-Base model.
    pub fn convnext_base(num_classes: usize, include_top: bool) -> Result<Self> {
        Self::new(ConvNeXtConfig {
            variant: ConvNeXtVariant::Base,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![128, 256, 512, 1024],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        })
    }

    /// Create a ConvNeXt-Large model.
    pub fn convnext_large(num_classes: usize, include_top: bool) -> Result<Self> {
        Self::new(ConvNeXtConfig {
            variant: ConvNeXtVariant::Large,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![192, 384, 768, 1536],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        })
    }

    /// Create a ConvNeXt-XLarge model.
    pub fn convnext_xlarge(num_classes: usize, include_top: bool) -> Result<Self> {
        Self::new(ConvNeXtConfig {
            variant: ConvNeXtVariant::XLarge,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![256, 512, 1024, 2048],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        })
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static,
    > Layer<F> for ConvNeXt<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = self.stem.forward(input)?;
        for stage in &self.stages {
            x = stage.forward(&x)?;
        }
        if let Some(ref head) = self.head {
            x = head.forward(&x)?;
        }
        Ok(x)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let mut grad = grad_output.clone();
        if let Some(ref head) = self.head {
            grad = head.backward(&grad, &grad)?;
        }
        for stage in self.stages.iter().rev() {
            grad = stage.backward(&grad, &grad)?;
        }
        self.stem.backward(input, &grad)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.stem.update(learning_rate)?;
        for stage in &mut self.stages {
            stage.update(learning_rate)?;
        }
        if let Some(ref mut head) = self.head {
            head.update(learning_rate)?;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.stem.params());
        for stage in &self.stages {
            params.extend(stage.params());
        }
        if let Some(ref head) = self.head {
            params.extend(head.params());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.stem.set_training(training);
        for stage in &mut self.stages {
            stage.set_training(training);
        }
        if let Some(ref mut head) = self.head {
            head.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.stem.is_training()
    }

    fn layer_type(&self) -> &str {
        "ConvNeXt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convnext_config() {
        let config = ConvNeXtConfig::default();
        assert_eq!(config.variant, ConvNeXtVariant::Tiny);
        assert_eq!(config.input_channels, 3);
        assert_eq!(config.depths.len(), 4);
        assert_eq!(config.dims.len(), 4);
    }

    #[test]
    fn test_convnext_block_creation() {
        let block = ConvNeXtBlock::<f64>::new(64, 1e-6, 0.0);
        assert!(block.is_ok());
    }

    #[test]
    fn test_convnext_stage_config() {
        let config = ConvNeXtStageConfig {
            input_channels: 64,
            output_channels: 128,
            num_blocks: 3,
            stride: 2,
            layer_scale_init_value: 1e-6,
            drop_path_prob: 0.0,
        };
        let stage = ConvNeXtStage::<f64>::new(&config);
        assert!(stage.is_ok());
    }

    #[test]
    fn test_convnext_downsample() {
        let downsample = ConvNeXtDownsample::<f64>::new(64, 128, 2);
        assert!(downsample.is_ok());
    }

    #[test]
    fn test_convnext_variants() {
        assert_eq!(ConvNeXtVariant::Tiny, ConvNeXtVariant::Tiny);
        assert_ne!(ConvNeXtVariant::Tiny, ConvNeXtVariant::Base);
    }
}
