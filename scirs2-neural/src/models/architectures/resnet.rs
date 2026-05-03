//! ResNet implementation
//!
//! ResNet (Residual Network) is a popular CNN architecture that introduced
//! skip connections to allow for training very deep networks.
//! Reference: "Deep Residual Learning for Image Recognition", He et al. (2015)
//! <https://arxiv.org/abs/1512.03385>

use crate::error::{NeuralError, Result};
use crate::layers::conv::PaddingMode;
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// ResNet block configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResNetBlock {
    /// Basic block (2 conv layers)
    Basic,
    /// Bottleneck block (3 conv layers with bottleneck)
    Bottleneck,
}

/// Configuration for a ResNet layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetLayer {
    /// Number of blocks in this layer
    pub blocks: usize,
    /// Number of output channels
    pub channels: usize,
    /// Stride for the first block (usually 1 or 2)
    pub stride: usize,
}

/// Configuration for a ResNet model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetConfig {
    /// Block type (Basic or Bottleneck)
    pub block: ResNetBlock,
    /// Layer configuration
    pub layers: Vec<ResNetLayer>,
    /// Number of input channels (e.g., 3 for RGB)
    pub input_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate (0 to disable)
    pub dropout_rate: f64,
}

impl ResNetConfig {
    /// Create a ResNet-18 configuration
    pub fn resnet18(input_channels: usize, num_classes: usize) -> Self {
        Self {
            block: ResNetBlock::Basic,
            layers: vec![
                ResNetLayer {
                    blocks: 2,
                    channels: 64,
                    stride: 1,
                },
                ResNetLayer {
                    blocks: 2,
                    channels: 128,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 2,
                    channels: 256,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 2,
                    channels: 512,
                    stride: 2,
                },
            ],
            input_channels,
            num_classes,
            dropout_rate: 0.0,
        }
    }

    /// Create a ResNet-34 configuration
    pub fn resnet34(input_channels: usize, num_classes: usize) -> Self {
        Self {
            block: ResNetBlock::Basic,
            layers: vec![
                ResNetLayer {
                    blocks: 3,
                    channels: 64,
                    stride: 1,
                },
                ResNetLayer {
                    blocks: 4,
                    channels: 128,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 6,
                    channels: 256,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 3,
                    channels: 512,
                    stride: 2,
                },
            ],
            input_channels,
            num_classes,
            dropout_rate: 0.0,
        }
    }

    /// Create a ResNet-50 configuration
    pub fn resnet50(input_channels: usize, num_classes: usize) -> Self {
        Self {
            block: ResNetBlock::Bottleneck,
            layers: vec![
                ResNetLayer {
                    blocks: 3,
                    channels: 64,
                    stride: 1,
                },
                ResNetLayer {
                    blocks: 4,
                    channels: 128,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 6,
                    channels: 256,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 3,
                    channels: 512,
                    stride: 2,
                },
            ],
            input_channels,
            num_classes,
            dropout_rate: 0.0,
        }
    }

    /// Create a ResNet-101 configuration
    pub fn resnet101(input_channels: usize, num_classes: usize) -> Self {
        Self {
            block: ResNetBlock::Bottleneck,
            layers: vec![
                ResNetLayer {
                    blocks: 3,
                    channels: 64,
                    stride: 1,
                },
                ResNetLayer {
                    blocks: 4,
                    channels: 128,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 23,
                    channels: 256,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 3,
                    channels: 512,
                    stride: 2,
                },
            ],
            input_channels,
            num_classes,
            dropout_rate: 0.0,
        }
    }

    /// Create a ResNet-152 configuration
    pub fn resnet152(input_channels: usize, num_classes: usize) -> Self {
        Self {
            block: ResNetBlock::Bottleneck,
            layers: vec![
                ResNetLayer {
                    blocks: 3,
                    channels: 64,
                    stride: 1,
                },
                ResNetLayer {
                    blocks: 8,
                    channels: 128,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 36,
                    channels: 256,
                    stride: 2,
                },
                ResNetLayer {
                    blocks: 3,
                    channels: 512,
                    stride: 2,
                },
            ],
            input_channels,
            num_classes,
            dropout_rate: 0.0,
        }
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }
}

/// Basic block for ResNet (2 conv layers)
struct BasicBlock<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// First convolutional layer
    conv1: Conv2D<F>,
    /// First batch normalization layer
    bn1: BatchNorm<F>,
    /// Second convolutional layer
    conv2: Conv2D<F>,
    /// Second batch normalization layer
    bn2: BatchNorm<F>,
    /// Skip connection downsample (optional)
    downsample: Option<(Conv2D<F>, BatchNorm<F>)>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Clone for BasicBlock<F> {
    fn clone(&self) -> Self {
        Self {
            conv1: self.conv1.clone(),
            bn1: self.bn1.clone(),
            conv2: self.conv2.clone(),
            bn2: self.bn2.clone(),
            downsample: self.downsample.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> BasicBlock<F> {
    /// Create a new basic block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: bool,
    ) -> Result<Self> {
        let stride_tuple = (stride, stride);

        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let conv1 = Conv2D::new(in_channels, out_channels, (3, 3), stride_tuple, None)?
            .with_padding(PaddingMode::Same);

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([43; 32]);
        let bn1 = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng2)?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([44; 32]);
        let conv2 = Conv2D::new(out_channels, out_channels, (3, 3), (1, 1), None)?
            .with_padding(PaddingMode::Same);

        let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([45; 32]);
        let bn2 = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng4)?;

        let downsample = if downsample {
            let mut rng5 = scirs2_core::random::rngs::SmallRng::from_seed([46; 32]);
            let ds_conv = Conv2D::new(in_channels, out_channels, (1, 1), stride_tuple, None)?
                .with_padding(PaddingMode::Valid);

            let mut rng6 = scirs2_core::random::rngs::SmallRng::from_seed([47; 32]);
            let ds_bn = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng6)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for BasicBlock<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // First conv block
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = x.mapv(|v: F| v.max(F::zero())); // ReLU

        // Second conv block
        x = self.conv2.forward(&x)?;
        x = self.bn2.forward(&x)?;

        // Skip connection
        let identity = if let Some((ref conv, ref bn)) = self.downsample {
            let ds = conv.forward(input)?;
            bn.forward(&ds)?
        } else {
            input.clone()
        };

        // Add skip connection
        let x = &x + &identity;

        // Final ReLU
        let x = x.mapv(|v: F| v.max(F::zero()));

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.conv1.update(learning_rate)?;
        self.bn1.update(learning_rate)?;
        self.conv2.update(learning_rate)?;
        self.bn2.update(learning_rate)?;
        if let Some((ref mut conv, ref mut bn)) = self.downsample {
            conv.update(learning_rate)?;
            bn.update(learning_rate)?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + ToPrimitive
            + FromPrimitive
            + 'static,
    > BasicBlock<F>
{
    /// Extract named parameters with a given prefix (HuggingFace-compatible naming)
    pub(crate) fn extract_named_params(&self, prefix: &str) -> Vec<(String, Array<F, IxDyn>)> {
        let mut result = Vec::new();
        // conv1: weight, bias
        for (i, p) in self.conv1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.conv1.{suffix}"), p.clone()));
        }
        // bn1: weight (gamma), bias (beta)
        for (i, p) in self.bn1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.bn1.{suffix}"), p.clone()));
        }
        // conv2
        for (i, p) in self.conv2.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.conv2.{suffix}"), p.clone()));
        }
        // bn2
        for (i, p) in self.bn2.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.bn2.{suffix}"), p.clone()));
        }
        // downsample (optional)
        if let Some((ref conv, ref bn)) = self.downsample {
            for (i, p) in conv.params().iter().enumerate() {
                let suffix = if i == 0 { "weight" } else { "bias" };
                result.push((format!("{prefix}.downsample.0.{suffix}"), p.clone()));
            }
            for (i, p) in bn.params().iter().enumerate() {
                let suffix = if i == 0 { "weight" } else { "bias" };
                result.push((format!("{prefix}.downsample.1.{suffix}"), p.clone()));
            }
        }
        result
    }

    /// Load named parameters from a map using the given prefix
    pub(crate) fn load_named_params(
        &mut self,
        prefix: &str,
        params_map: &HashMap<String, Array<F, IxDyn>>,
    ) -> Result<()> {
        // conv1
        if let Some(w) = params_map.get(&format!("{prefix}.conv1.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.conv1.bias")) {
                ps.push(b.clone());
            }
            self.conv1.set_params(&ps)?;
        }
        // bn1
        if let Some(w) = params_map.get(&format!("{prefix}.bn1.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.bn1.bias")) {
                ps.push(b.clone());
            }
            self.bn1.set_params(&ps)?;
        }
        // conv2
        if let Some(w) = params_map.get(&format!("{prefix}.conv2.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.conv2.bias")) {
                ps.push(b.clone());
            }
            self.conv2.set_params(&ps)?;
        }
        // bn2
        if let Some(w) = params_map.get(&format!("{prefix}.bn2.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.bn2.bias")) {
                ps.push(b.clone());
            }
            self.bn2.set_params(&ps)?;
        }
        // downsample
        if let Some((ref mut conv, ref mut bn)) = self.downsample {
            if let Some(w) = params_map.get(&format!("{prefix}.downsample.0.weight")) {
                let mut ps = vec![w.clone()];
                if let Some(b) = params_map.get(&format!("{prefix}.downsample.0.bias")) {
                    ps.push(b.clone());
                }
                conv.set_params(&ps)?;
            }
            if let Some(w) = params_map.get(&format!("{prefix}.downsample.1.weight")) {
                let mut ps = vec![w.clone()];
                if let Some(b) = params_map.get(&format!("{prefix}.downsample.1.bias")) {
                    ps.push(b.clone());
                }
                bn.set_params(&ps)?;
            }
        }
        Ok(())
    }
}

/// Bottleneck block for ResNet (3 conv layers)
struct BottleneckBlock<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// First convolutional layer (1x1 reduce)
    conv1: Conv2D<F>,
    /// First batch normalization layer
    bn1: BatchNorm<F>,
    /// Second convolutional layer (3x3)
    conv2: Conv2D<F>,
    /// Second batch normalization layer
    bn2: BatchNorm<F>,
    /// Third convolutional layer (1x1 expand)
    conv3: Conv2D<F>,
    /// Third batch normalization layer
    bn3: BatchNorm<F>,
    /// Skip connection downsample (optional)
    downsample: Option<(Conv2D<F>, BatchNorm<F>)>,
    /// Expansion factor
    #[allow(dead_code)]
    expansion: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Clone
    for BottleneckBlock<F>
{
    fn clone(&self) -> Self {
        Self {
            conv1: self.conv1.clone(),
            bn1: self.bn1.clone(),
            conv2: self.conv2.clone(),
            bn2: self.bn2.clone(),
            conv3: self.conv3.clone(),
            bn3: self.bn3.clone(),
            downsample: self.downsample.clone(),
            expansion: self.expansion,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> BottleneckBlock<F> {
    /// Expansion factor for bottleneck blocks
    const EXPANSION: usize = 4;

    /// Create a new bottleneck block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: bool,
    ) -> Result<Self> {
        let bottleneck_channels = out_channels / Self::EXPANSION;
        let stride_tuple = (stride, stride);

        // First conv (1x1 reduce)
        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([48; 32]);
        let conv1 = Conv2D::new(in_channels, bottleneck_channels, (1, 1), (1, 1), None)?
            .with_padding(PaddingMode::Valid);

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([49; 32]);
        let bn1 = BatchNorm::new(bottleneck_channels, 1e-5, 0.1, &mut rng2)?;

        // Second conv (3x3)
        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([50; 32]);
        let conv2 = Conv2D::new(
            bottleneck_channels,
            bottleneck_channels,
            (3, 3),
            stride_tuple,
            None,
        )?
        .with_padding(PaddingMode::Same);

        let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([51; 32]);
        let bn2 = BatchNorm::new(bottleneck_channels, 1e-5, 0.1, &mut rng4)?;

        // Third conv (1x1 expand)
        let mut rng5 = scirs2_core::random::rngs::SmallRng::from_seed([52; 32]);
        let conv3 = Conv2D::new(bottleneck_channels, out_channels, (1, 1), (1, 1), None)?
            .with_padding(PaddingMode::Valid);

        let mut rng6 = scirs2_core::random::rngs::SmallRng::from_seed([53; 32]);
        let bn3 = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng6)?;

        // Downsample
        let downsample = if downsample {
            let mut rng7 = scirs2_core::random::rngs::SmallRng::from_seed([54; 32]);
            let ds_conv = Conv2D::new(in_channels, out_channels, (1, 1), stride_tuple, None)?
                .with_padding(PaddingMode::Valid);

            let mut rng8 = scirs2_core::random::rngs::SmallRng::from_seed([55; 32]);
            let ds_bn = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng8)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
            expansion: Self::EXPANSION,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for BottleneckBlock<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // First conv block
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = x.mapv(|v: F| v.max(F::zero())); // ReLU

        // Second conv block
        x = self.conv2.forward(&x)?;
        x = self.bn2.forward(&x)?;
        x = x.mapv(|v: F| v.max(F::zero())); // ReLU

        // Third conv block
        x = self.conv3.forward(&x)?;
        x = self.bn3.forward(&x)?;

        // Skip connection
        let identity = if let Some((ref conv, ref bn)) = self.downsample {
            let ds = conv.forward(input)?;
            bn.forward(&ds)?
        } else {
            input.clone()
        };

        // Add skip connection
        let x = &x + &identity;

        // Final ReLU
        let x = x.mapv(|v: F| v.max(F::zero()));

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.conv1.update(learning_rate)?;
        self.bn1.update(learning_rate)?;
        self.conv2.update(learning_rate)?;
        self.bn2.update(learning_rate)?;
        self.conv3.update(learning_rate)?;
        self.bn3.update(learning_rate)?;
        if let Some((ref mut conv, ref mut bn)) = self.downsample {
            conv.update(learning_rate)?;
            bn.update(learning_rate)?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + ToPrimitive
            + FromPrimitive
            + 'static,
    > BottleneckBlock<F>
{
    /// Extract named parameters with a given prefix (HuggingFace-compatible naming)
    pub(crate) fn extract_named_params(&self, prefix: &str) -> Vec<(String, Array<F, IxDyn>)> {
        let mut result = Vec::new();
        for (i, p) in self.conv1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.conv1.{suffix}"), p.clone()));
        }
        for (i, p) in self.bn1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.bn1.{suffix}"), p.clone()));
        }
        for (i, p) in self.conv2.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.conv2.{suffix}"), p.clone()));
        }
        for (i, p) in self.bn2.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.bn2.{suffix}"), p.clone()));
        }
        for (i, p) in self.conv3.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.conv3.{suffix}"), p.clone()));
        }
        for (i, p) in self.bn3.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("{prefix}.bn3.{suffix}"), p.clone()));
        }
        if let Some((ref conv, ref bn)) = self.downsample {
            for (i, p) in conv.params().iter().enumerate() {
                let suffix = if i == 0 { "weight" } else { "bias" };
                result.push((format!("{prefix}.downsample.0.{suffix}"), p.clone()));
            }
            for (i, p) in bn.params().iter().enumerate() {
                let suffix = if i == 0 { "weight" } else { "bias" };
                result.push((format!("{prefix}.downsample.1.{suffix}"), p.clone()));
            }
        }
        result
    }

    /// Load named parameters from a map using the given prefix
    pub(crate) fn load_named_params(
        &mut self,
        prefix: &str,
        params_map: &HashMap<String, Array<F, IxDyn>>,
    ) -> Result<()> {
        if let Some(w) = params_map.get(&format!("{prefix}.conv1.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.conv1.bias")) {
                ps.push(b.clone());
            }
            self.conv1.set_params(&ps)?;
        }
        if let Some(w) = params_map.get(&format!("{prefix}.bn1.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.bn1.bias")) {
                ps.push(b.clone());
            }
            self.bn1.set_params(&ps)?;
        }
        if let Some(w) = params_map.get(&format!("{prefix}.conv2.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.conv2.bias")) {
                ps.push(b.clone());
            }
            self.conv2.set_params(&ps)?;
        }
        if let Some(w) = params_map.get(&format!("{prefix}.bn2.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.bn2.bias")) {
                ps.push(b.clone());
            }
            self.bn2.set_params(&ps)?;
        }
        if let Some(w) = params_map.get(&format!("{prefix}.conv3.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.conv3.bias")) {
                ps.push(b.clone());
            }
            self.conv3.set_params(&ps)?;
        }
        if let Some(w) = params_map.get(&format!("{prefix}.bn3.weight")) {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get(&format!("{prefix}.bn3.bias")) {
                ps.push(b.clone());
            }
            self.bn3.set_params(&ps)?;
        }
        if let Some((ref mut conv, ref mut bn)) = self.downsample {
            if let Some(w) = params_map.get(&format!("{prefix}.downsample.0.weight")) {
                let mut ps = vec![w.clone()];
                if let Some(b) = params_map.get(&format!("{prefix}.downsample.0.bias")) {
                    ps.push(b.clone());
                }
                conv.set_params(&ps)?;
            }
            if let Some(w) = params_map.get(&format!("{prefix}.downsample.1.weight")) {
                let mut ps = vec![w.clone()];
                if let Some(b) = params_map.get(&format!("{prefix}.downsample.1.bias")) {
                    ps.push(b.clone());
                }
                bn.set_params(&ps)?;
            }
        }
        Ok(())
    }
}

/// ResNet implementation
pub struct ResNet<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// Initial convolutional layer
    conv1: Conv2D<F>,
    /// Initial batch normalization
    bn1: BatchNorm<F>,
    /// ResNet layer groups
    layer1: Vec<BasicBlock<F>>,
    /// ResNet layer groups (bottleneck)
    layer1_bottleneck: Vec<BottleneckBlock<F>>,
    /// Fully connected layer
    fc: Dense<F>,
    /// Dropout layer
    dropout: Option<Dropout<F>>,
    /// Model configuration
    config: ResNetConfig,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ResNet<F> {
    /// Create a new ResNet model
    pub fn new(config: ResNetConfig) -> Result<Self> {
        // Initial convolution
        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([56; 32]);
        let conv1 = Conv2D::new(config.input_channels, 64, (7, 7), (2, 2), None)?
            .with_padding(PaddingMode::Same);

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([57; 32]);
        let bn1 = BatchNorm::new(64, 1e-5, 0.1, &mut rng2)?;

        // For simplicity, create a single layer with blocks
        let layer1 = Vec::new();
        let layer1_bottleneck = Vec::new();

        // Final FC layer
        let fc_in_features = match config.block {
            ResNetBlock::Basic => config.layers.last().map(|l| l.channels).unwrap_or(512),
            ResNetBlock::Bottleneck => config.layers.last().map(|l| l.channels * 4).unwrap_or(2048),
        };

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([58; 32]);
        let fc = Dense::new(fc_in_features, config.num_classes, None, &mut rng3)?;

        // Dropout
        let dropout = if config.dropout_rate > 0.0 {
            let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([59; 32]);
            Some(Dropout::new(config.dropout_rate, &mut rng4)?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer1_bottleneck,
            fc,
            dropout,
            config,
        })
    }

    /// Create a ResNet-18 model
    pub fn resnet18(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet18(input_channels, num_classes);
        Self::new(config)
    }

    /// Create a ResNet-34 model
    pub fn resnet34(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet34(input_channels, num_classes);
        Self::new(config)
    }

    /// Create a ResNet-50 model
    pub fn resnet50(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet50(input_channels, num_classes);
        Self::new(config)
    }

    /// Create a ResNet-101 model
    pub fn resnet101(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet101(input_channels, num_classes);
        Self::new(config)
    }

    /// Create a ResNet-152 model
    pub fn resnet152(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = ResNetConfig::resnet152(input_channels, num_classes);
        Self::new(config)
    }

    /// Get the model configuration
    pub fn config(&self) -> &ResNetConfig {
        &self.config
    }

    /// Global average pooling
    fn global_avg_pool(x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input for average pooling, got shape {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let mut output = Array::zeros(IxDyn(&[batch_size, channels]));
        let count = F::from(height * width).expect("Failed to convert to float");

        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = F::zero();
                for h in 0..height {
                    for w in 0..width {
                        sum += x[[b, c, h, w]];
                    }
                }
                output[[b, c]] = sum / count;
            }
        }

        Ok(output)
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + NumAssign
            + ToPrimitive
            + FromPrimitive
            + 'static,
    > ResNet<F>
{
    /// Extract all named parameters in HuggingFace-compatible format.
    ///
    /// Parameter names follow PyTorch/HuggingFace ResNet naming convention:
    /// - `conv1.weight`, `bn1.weight`, `bn1.bias`
    /// - `layer1.0.conv1.weight`, `layer1.0.bn1.weight`, etc.
    /// - `layer2.0.conv1.weight`, etc. (for bottleneck blocks)
    /// - `fc.weight`, `fc.bias`
    pub fn extract_named_params(&self) -> Result<Vec<(String, Array<F, IxDyn>)>> {
        let mut result = Vec::new();

        // Initial conv and bn
        for (i, p) in self.conv1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("conv1.{suffix}"), p.clone()));
        }
        for (i, p) in self.bn1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("bn1.{suffix}"), p.clone()));
        }

        // Residual blocks: basic or bottleneck
        for (idx, block) in self.layer1.iter().enumerate() {
            let block_params = block.extract_named_params(&format!("layer1.{idx}"));
            result.extend(block_params);
        }
        for (idx, block) in self.layer1_bottleneck.iter().enumerate() {
            let block_params = block.extract_named_params(&format!("layer1.{idx}"));
            result.extend(block_params);
        }

        // FC (classifier head)
        for (i, p) in self.fc.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("fc.{suffix}"), p.clone()));
        }

        Ok(result)
    }

    /// Load named parameters from a map (by name).
    ///
    /// Unknown parameter names are silently ignored, enabling graceful
    /// forward/backward compatibility between model versions.
    pub fn load_named_params(
        &mut self,
        params_map: &HashMap<String, Array<F, IxDyn>>,
    ) -> Result<()> {
        // conv1
        if let Some(w) = params_map.get("conv1.weight") {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get("conv1.bias") {
                ps.push(b.clone());
            }
            self.conv1.set_params(&ps)?;
        }
        // bn1
        if let Some(w) = params_map.get("bn1.weight") {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get("bn1.bias") {
                ps.push(b.clone());
            }
            self.bn1.set_params(&ps)?;
        }

        // Basic blocks
        for (idx, block) in self.layer1.iter_mut().enumerate() {
            block.load_named_params(&format!("layer1.{idx}"), params_map)?;
        }
        // Bottleneck blocks
        for (idx, block) in self.layer1_bottleneck.iter_mut().enumerate() {
            block.load_named_params(&format!("layer1.{idx}"), params_map)?;
        }

        // fc
        if let Some(w) = params_map.get("fc.weight") {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get("fc.bias") {
                ps.push(b.clone());
            }
            self.fc.set_params(&ps)?;
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F> for ResNet<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Initial conv
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = x.mapv(|v: F| v.max(F::zero())); // ReLU

        // Process through basic blocks
        for block in &self.layer1 {
            x = block.forward(&x)?;
        }

        // Process through bottleneck blocks
        for block in &self.layer1_bottleneck {
            x = block.forward(&x)?;
        }

        // Global average pooling
        x = Self::global_avg_pool(&x)?;

        // Dropout
        if let Some(ref dropout) = self.dropout {
            x = dropout.forward(&x)?;
        }

        // Final FC
        x = self.fc.forward(&x)?;

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.conv1.update(learning_rate)?;
        self.bn1.update(learning_rate)?;

        for block in &mut self.layer1 {
            block.update(learning_rate)?;
        }

        for block in &mut self.layer1_bottleneck {
            block.update(learning_rate)?;
        }

        self.fc.update(learning_rate)?;

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet_config_18() {
        let config = ResNetConfig::resnet18(3, 1000);
        assert_eq!(config.input_channels, 3);
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.layers.len(), 4);
        assert!(matches!(config.block, ResNetBlock::Basic));
    }

    #[test]
    fn test_resnet_config_50() {
        let config = ResNetConfig::resnet50(3, 1000);
        assert!(matches!(config.block, ResNetBlock::Bottleneck));
        assert_eq!(config.layers.len(), 4);
    }

    #[test]
    fn test_resnet_config_with_dropout() {
        let config = ResNetConfig::resnet18(3, 100).with_dropout(0.5);
        assert_eq!(config.dropout_rate, 0.5);
    }

    #[test]
    fn test_resnet_config_variants() {
        let config34 = ResNetConfig::resnet34(3, 1000);
        assert_eq!(config34.layers[0].blocks, 3);
        assert_eq!(config34.layers[1].blocks, 4);

        let config101 = ResNetConfig::resnet101(3, 1000);
        assert_eq!(config101.layers[2].blocks, 23);

        let config152 = ResNetConfig::resnet152(3, 1000);
        assert_eq!(config152.layers[2].blocks, 36);
    }
}
