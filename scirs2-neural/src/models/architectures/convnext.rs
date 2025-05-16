//! ConvNeXt architecture implementation
//!
//! This module implements the ConvNeXt architecture as described in
//! "A ConvNet for the 2020s" (https://arxiv.org/abs/2201.03545)
//!
//! ConvNeXt modernizes ResNet architecture by incorporating design choices from
//! Vision Transformers, resulting in a pure convolutional model with excellent performance.

use crate::activations::GELU;
use crate::error::{Error, Result};
use crate::layers::{
    Conv2D, Dense, Dropout, GlobalAvgPool2D, Layer, LayerNorm2D, PaddingMode, Sequential,
};

use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::SeedableRng;
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

/// ConvNeXt block implementation
#[derive(Debug, Clone)]
pub struct ConvNeXtBlock<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Depthwise convolution
    pub depthwise_conv: Conv2D<F>,
    /// Layer normalization
    pub norm: LayerNorm2D<F>,
    /// Pointwise convolution 1
    pub pointwise_conv1: Conv2D<F>,
    /// GELU activation
    pub gelu: GELU,
    /// Pointwise convolution 2
    pub pointwise_conv2: Conv2D<F>,
    /// Layer scale gamma parameter
    pub gamma: Array<F, IxDyn>,
    /// Skip connection flag
    pub use_skip: bool,
    /// Skip connection scale for stochastic depth
    pub skip_scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> ConvNeXtBlock<F> {
    /// Create a new ConvNeXtBlock
    pub fn new(channels: usize, layer_scale_init_value: f64, drop_path_prob: f64) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);
        let depthwise_conv = Conv2D::<F>::new(
            channels,
            channels,
            (7, 7),
            (1, 1),
            PaddingMode::Custom(3),
            &mut rng,
        )?;

        let norm = LayerNorm2D::<F>::new::<SmallRng>(channels, 1e-6, Some("norm"))?;

        let pointwise_conv1 = Conv2D::<F>::new(
            channels,
            channels * 4,
            (1, 1),
            (1, 1),
            PaddingMode::Custom(0),
            &mut rng,
        )?;

        let gelu = GELU::new();

        let pointwise_conv2 = Conv2D::<F>::new(
            channels * 4,
            channels,
            (1, 1),
            (1, 1),
            PaddingMode::Custom(0),
            &mut rng,
        )?;

        // Initialize gamma as a learnable parameter
        let gamma_value = F::from(layer_scale_init_value).unwrap();
        let gamma = Array::<F, _>::from_elem([channels, 1, 1], gamma_value).into_dyn();

        // Stochastic depth rate
        let skip_scale = F::from(1.0 - drop_path_prob).unwrap();
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

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for ConvNeXtBlock<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Save input for skip connection
        let identity = input.clone();

        // Depthwise convolution
        let mut x = self.depthwise_conv.forward(input)?;

        // Normalization
        x = self.norm.forward(&x)?;

        // First pointwise convolution and activation
        x = self.pointwise_conv1.forward(&x)?;
        x = <GELU as Layer<F>>::forward(&self.gelu, &x)?;

        // Second pointwise convolution
        x = self.pointwise_conv2.forward(&x)?;

        // Apply layer scale
        let shape = x.shape().to_vec();
        let view = x
            .clone()
            .into_shape_with_order((shape[0], shape[1], shape[2] * shape[3]))?;
        let scaled = view * &self.gamma;
        x = scaled.into_shape_with_order(shape).unwrap();

        // Apply stochastic depth and skip connection
        if self.use_skip {
            // During training, scale the output by (1 - drop_path_prob)
            x = x * self.skip_scale;
        }

        // Add skip connection
        x = x + identity;

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Not implemented for this example
        Err(Error::NotImplementedError(
            "ConvNeXtBlock backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // Not implemented for this example
        Err(Error::NotImplementedError(
            "ConvNeXtBlock update not implemented".to_string(),
        ))
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
}

/// ConvNeXt downsampling layer
#[derive(Debug, Clone)]
pub struct ConvNeXtDownsample<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Layer normalization
    pub norm: LayerNorm2D<F>,
    /// Convolution for downsampling
    pub conv: Conv2D<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> ConvNeXtDownsample<F> {
    /// Create a new ConvNeXtDownsample
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);
        let norm = LayerNorm2D::<F>::new::<SmallRng>(in_channels, 1e-6, Some("downsample_norm"))?;

        let conv = Conv2D::<F>::new(
            in_channels,
            out_channels,
            (stride, stride),
            (stride, stride),
            PaddingMode::Custom(0),
            &mut rng,
        )?;

        Ok(Self { norm, conv })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for ConvNeXtDownsample<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let x = self.norm.forward(input)?;
        let x = self.conv.forward(&x)?;
        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(Error::NotImplementedError(
            "ConvNeXtDownsample backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(Error::NotImplementedError(
            "ConvNeXtDownsample update not implemented".to_string(),
        ))
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
}

/// ConvNeXt stage
#[derive(Debug, Clone)]
pub struct ConvNeXtStage<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Downsampling layer (optional)
    pub downsample: Option<ConvNeXtDownsample<F>>,
    /// Blocks in this stage
    pub blocks: Vec<ConvNeXtBlock<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> ConvNeXtStage<F> {
    /// Create a new ConvNeXtStage
    pub fn new(config: &ConvNeXtStageConfig) -> Result<Self> {
        // Create the downsampling layer if needed
        let downsample = if config.input_channels != config.output_channels || config.stride > 1 {
            Some(ConvNeXtDownsample::<F>::new(
                config.input_channels,
                config.output_channels,
                config.stride,
            )?)
        } else {
            None
        };

        // Create the blocks
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

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for ConvNeXtStage<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Apply downsampling if available
        let mut x = if let Some(ref downsample) = self.downsample {
            downsample.forward(input)?
        } else {
            input.clone()
        };

        // Apply all blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(Error::NotImplementedError(
            "ConvNeXtStage backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(Error::NotImplementedError(
            "ConvNeXtStage update not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.params());
        }
        for block in &self.blocks {
            params.extend(block.params());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        if let Some(ref mut downsample) = self.downsample {
            downsample.set_training(training);
        }
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        if let Some(ref downsample) = self.downsample {
            return downsample.is_training();
        }
        self.blocks[0].is_training()
    }
}

/// ConvNeXt model
#[derive(Debug, Clone)]
pub struct ConvNeXt<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Stem layer (initial convolution)
    pub stem: Sequential<F>,
    /// Main stages of the network
    pub stages: Vec<ConvNeXtStage<F>>,
    /// Classification head (if include_top is true)
    pub head: Option<Sequential<F>>,
    /// Model configuration
    pub config: ConvNeXtConfig,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> ConvNeXt<F> {
    /// Create a new ConvNeXt model
    pub fn new(config: ConvNeXtConfig) -> Result<Self> {
        // Create the stem layer
        let mut stem = Sequential::new();
        let mut rng = SmallRng::seed_from_u64(42);
        stem.add(Conv2D::<F>::new(
            config.input_channels,
            config.dims[0],
            (4, 4),
            (4, 4),
            PaddingMode::Custom(0),
            &mut rng,
        )?);
        stem.add(LayerNorm2D::<F>::new::<SmallRng>(
            config.dims[0],
            1e-6,
            Some("stem_norm"),
        )?);

        // Create the stages
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
                drop_path_prob: 0.0, // This would typically vary by block in a real implementation
            };

            stages.push(ConvNeXtStage::<F>::new(&stage_config)?);
            current_channels = output_channels;
        }

        // Create the head if needed
        let head = if config.include_top {
            let mut head_seq = Sequential::new();
            let mut rng = SmallRng::seed_from_u64(42);
            head_seq.add(LayerNorm2D::<F>::new::<SmallRng>(
                *config.dims.last().unwrap(),
                1e-6,
                Some("head_norm"),
            )?);
            head_seq.add(GlobalAvgPool2D::<F>::new(Some("head_pool"))?);

            if let Some(dropout_rate) = config.dropout_rate {
                head_seq.add(Dropout::<F>::new(dropout_rate, &mut rng)?);
            }

            head_seq.add(Dense::<F>::new(
                *config.dims.last().unwrap(),
                config.num_classes,
                Some("relu"),
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

    /// Create a ConvNeXt-Tiny model
    pub fn convnext_tiny(num_classes: usize, include_top: bool) -> Result<Self> {
        let config = ConvNeXtConfig {
            variant: ConvNeXtVariant::Tiny,
            input_channels: 3,
            depths: vec![3, 3, 9, 3],
            dims: vec![96, 192, 384, 768],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        };

        Self::new(config)
    }

    /// Create a ConvNeXt-Small model
    pub fn convnext_small(num_classes: usize, include_top: bool) -> Result<Self> {
        let config = ConvNeXtConfig {
            variant: ConvNeXtVariant::Small,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![96, 192, 384, 768],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        };

        Self::new(config)
    }

    /// Create a ConvNeXt-Base model
    pub fn convnext_base(num_classes: usize, include_top: bool) -> Result<Self> {
        let config = ConvNeXtConfig {
            variant: ConvNeXtVariant::Base,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![128, 256, 512, 1024],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        };

        Self::new(config)
    }

    /// Create a ConvNeXt-Large model
    pub fn convnext_large(num_classes: usize, include_top: bool) -> Result<Self> {
        let config = ConvNeXtConfig {
            variant: ConvNeXtVariant::Large,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![192, 384, 768, 1536],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        };

        Self::new(config)
    }

    /// Create a ConvNeXt-XLarge model
    pub fn convnext_xlarge(num_classes: usize, include_top: bool) -> Result<Self> {
        let config = ConvNeXtConfig {
            variant: ConvNeXtVariant::XLarge,
            input_channels: 3,
            depths: vec![3, 3, 27, 3],
            dims: vec![256, 512, 1024, 2048],
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top,
        };

        Self::new(config)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for ConvNeXt<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Apply stem
        let mut x = self.stem.forward(input)?;

        // Apply stages
        for stage in &self.stages {
            x = stage.forward(&x)?;
        }

        // Apply head if available
        if let Some(ref head) = self.head {
            x = head.forward(&x)?;
        }

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(Error::NotImplementedError(
            "ConvNeXt backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(Error::NotImplementedError(
            "ConvNeXt update not implemented".to_string(),
        ))
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
}
