//! VGG implementation
//!
//! VGG is a deep convolutional neural network architecture proposed by the Visual Geometry Group
//! at Oxford. It is known for its simplicity, using only 3x3 convolutions stacked on top of
//! each other with max pooling to reduce spatial dimensions, followed by fully connected layers.
//! Reference: "Very Deep Convolutional Networks for Large-Scale Image Recognition", Simonyan & Zisserman (2014)
//! <https://arxiv.org/abs/1409.1556>

use crate::error::{NeuralError, Result};
use crate::layers::conv::PaddingMode;
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::SeedableRng;
use std::collections::HashMap;
use std::fmt::Debug;

/// VGG variant specifying the network depth
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VGGVariant {
    /// VGG-11 (configuration A): 8 conv layers + 3 FC layers
    VGG11,
    /// VGG-13 (configuration B): 10 conv layers + 3 FC layers
    VGG13,
    /// VGG-16 (configuration D): 13 conv layers + 3 FC layers
    VGG16,
    /// VGG-19 (configuration E): 16 conv layers + 3 FC layers
    VGG19,
}

impl VGGVariant {
    /// Return the channel configuration for the feature extractor part.
    /// Each inner Vec represents a block of convolutions followed by max pooling.
    /// The original paper defines configurations A, B, D, E.
    fn layer_config(&self) -> Vec<Vec<usize>> {
        match self {
            VGGVariant::VGG11 => vec![
                vec![64],
                vec![128],
                vec![256, 256],
                vec![512, 512],
                vec![512, 512],
            ],
            VGGVariant::VGG13 => vec![
                vec![64, 64],
                vec![128, 128],
                vec![256, 256],
                vec![512, 512],
                vec![512, 512],
            ],
            VGGVariant::VGG16 => vec![
                vec![64, 64],
                vec![128, 128],
                vec![256, 256, 256],
                vec![512, 512, 512],
                vec![512, 512, 512],
            ],
            VGGVariant::VGG19 => vec![
                vec![64, 64],
                vec![128, 128],
                vec![256, 256, 256, 256],
                vec![512, 512, 512, 512],
                vec![512, 512, 512, 512],
            ],
        }
    }

    /// Return a human-readable name
    pub fn name(&self) -> &str {
        match self {
            VGGVariant::VGG11 => "VGG-11",
            VGGVariant::VGG13 => "VGG-13",
            VGGVariant::VGG16 => "VGG-16",
            VGGVariant::VGG19 => "VGG-19",
        }
    }

    /// Return the total number of convolutional layers
    pub fn num_conv_layers(&self) -> usize {
        self.layer_config().iter().map(|block| block.len()).sum()
    }
}

/// Configuration for a VGG model
#[derive(Debug, Clone)]
pub struct VGGConfig {
    /// VGG variant (VGG11, VGG13, VGG16, VGG19)
    pub variant: VGGVariant,
    /// Whether to use batch normalization after each conv layer
    pub batch_norm: bool,
    /// Number of input channels (e.g., 3 for RGB images)
    pub input_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate for classifier layers (0 to disable)
    pub dropout_rate: f64,
    /// Number of hidden units in the first two FC layers (default: 4096)
    pub fc_hidden_units: usize,
    /// Channel divisor to scale down channel counts (default: 1, use 8 or 16 for testing)
    /// All channel counts are divided by this value (minimum 1 channel per layer)
    pub channel_divisor: usize,
}

impl VGGConfig {
    /// Create a VGG-11 configuration
    pub fn vgg11(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG11,
            batch_norm: false,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-11 configuration with batch normalization
    pub fn vgg11_bn(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG11,
            batch_norm: true,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-13 configuration
    pub fn vgg13(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG13,
            batch_norm: false,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-13 configuration with batch normalization
    pub fn vgg13_bn(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG13,
            batch_norm: true,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-16 configuration
    pub fn vgg16(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG16,
            batch_norm: false,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-16 configuration with batch normalization
    pub fn vgg16_bn(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG16,
            batch_norm: true,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-19 configuration
    pub fn vgg19(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG19,
            batch_norm: false,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Create a VGG-19 configuration with batch normalization
    pub fn vgg19_bn(input_channels: usize, num_classes: usize) -> Self {
        Self {
            variant: VGGVariant::VGG19,
            batch_norm: true,
            input_channels,
            num_classes,
            dropout_rate: 0.5,
            fc_hidden_units: 4096,
            channel_divisor: 1,
        }
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Set batch normalization
    pub fn with_batch_norm(mut self, batch_norm: bool) -> Self {
        self.batch_norm = batch_norm;
        self
    }

    /// Set the number of hidden units in FC layers
    pub fn with_fc_hidden_units(mut self, units: usize) -> Self {
        self.fc_hidden_units = units;
        self
    }

    /// Set channel divisor to scale down conv channel counts (for testing/lightweight models)
    ///
    /// A divisor of 8 turns VGG-11 channels [64, 128, 256, 512, 512] into [8, 16, 32, 64, 64].
    pub fn with_channel_divisor(mut self, divisor: usize) -> Self {
        self.channel_divisor = divisor.max(1);
        self
    }

    /// Get the effective layer config with channel_divisor applied
    fn effective_layer_config(&self) -> Vec<Vec<usize>> {
        let base_config = self.variant.layer_config();
        if self.channel_divisor <= 1 {
            return base_config;
        }
        base_config
            .into_iter()
            .map(|block| {
                block
                    .into_iter()
                    .map(|ch| (ch / self.channel_divisor).max(1))
                    .collect()
            })
            .collect()
    }
}

/// A single convolutional block within a VGG feature stage.
/// Contains a Conv2D layer and an optional BatchNorm layer.
struct VGGConvBlock<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// Convolutional layer (3x3)
    conv: Conv2D<F>,
    /// Optional batch normalization
    bn: Option<BatchNorm<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> VGGConvBlock<F> {
    /// Create a new VGG conv block
    fn new(in_channels: usize, out_channels: usize, use_bn: bool) -> Result<Self> {
        let conv = Conv2D::new(in_channels, out_channels, (3, 3), (1, 1), None)?
            .with_padding(PaddingMode::Same);

        let bn = if use_bn {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
            Some(BatchNorm::new(out_channels, 0.1, 1e-5, &mut rng)?)
        } else {
            None
        };

        Ok(Self { conv, bn })
    }

    /// Forward pass: conv -> [bn] -> relu
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = self.conv.forward(input)?;
        if let Some(ref bn) = self.bn {
            x = bn.forward(&x)?;
        }
        // ReLU activation
        x = x.mapv(|v: F| v.max(F::zero()));
        Ok(x)
    }

    /// Update parameters
    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.conv.update(learning_rate)?;
        if let Some(ref mut bn) = self.bn {
            bn.update(learning_rate)?;
        }
        Ok(())
    }

    /// Get parameters
    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut result = self.conv.params();
        if let Some(ref bn) = self.bn {
            result.extend(bn.params());
        }
        result
    }

    /// Count parameters
    fn parameter_count(&self) -> usize {
        let mut count = self.conv.parameter_count();
        if let Some(ref bn) = self.bn {
            count += bn.parameter_count();
        }
        count
    }
}

/// A feature stage in VGG: a sequence of conv blocks followed by 2x2 max pooling.
struct VGGFeatureStage<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// Convolutional blocks in this stage
    blocks: Vec<VGGConvBlock<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> VGGFeatureStage<F> {
    /// Create a new feature stage
    fn new(channels: &[usize], in_channels: usize, use_bn: bool) -> Result<Self> {
        let mut blocks = Vec::with_capacity(channels.len());
        let mut current_in = in_channels;
        for &out_ch in channels {
            blocks.push(VGGConvBlock::new(current_in, out_ch, use_bn)?);
            current_in = out_ch;
        }
        Ok(Self { blocks })
    }

    /// Forward pass: apply all conv blocks then max pool
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        // 2x2 max pooling with stride 2
        x = max_pool_2x2(&x)?;
        Ok(x)
    }

    /// Update parameters
    fn update(&mut self, learning_rate: F) -> Result<()> {
        for block in &mut self.blocks {
            block.update(learning_rate)?;
        }
        Ok(())
    }

    /// Get parameters
    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut result = Vec::new();
        for block in &self.blocks {
            result.extend(block.params());
        }
        result
    }

    /// Count parameters
    fn parameter_count(&self) -> usize {
        self.blocks.iter().map(|b| b.parameter_count()).sum()
    }
}

/// 2x2 max pooling with stride 2
fn max_pool_2x2<F: Float + Debug>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "Expected 4D input for max pooling, got shape {:?}",
            shape
        )));
    }
    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    let out_h = height / 2;
    let out_w = width / 2;

    let mut output = Array::from_elem(
        IxDyn(&[batch_size, channels, out_h, out_w]),
        F::neg_infinity(),
    );

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let h_start = oh * 2;
                    let w_start = ow * 2;
                    let mut max_val = F::neg_infinity();
                    for dh in 0..2 {
                        for dw in 0..2 {
                            let h = h_start + dh;
                            let w = w_start + dw;
                            if h < height && w < width {
                                let val = input[[b, c, h, w]];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    output[[b, c, oh, ow]] = max_val;
                }
            }
        }
    }

    Ok(output)
}

/// VGG neural network implementation
///
/// VGG networks use a very uniform architecture: stacks of 3x3 convolutions
/// with ReLU activation, followed by 2x2 max pooling to reduce spatial dimensions.
/// The feature extractor is followed by three fully connected layers.
///
/// The original paper defines configurations A (11 layers), B (13), D (16), E (19).
///
/// # Examples
///
/// ```no_run
/// use scirs2_neural::models::architectures::vgg::{VGG, VGGConfig};
///
/// // Create VGG-16 for ImageNet classification
/// let model: VGG<f64> = VGG::vgg16(3, 1000).expect("Failed to create VGG-16");
///
/// // Create VGG-19 with batch normalization
/// let model_bn: VGG<f64> = VGG::vgg19_bn(3, 100).expect("Failed to create VGG-19-BN");
/// ```
pub struct VGG<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// Model configuration
    config: VGGConfig,
    /// Feature extraction stages (conv blocks + max pool)
    features: Vec<VGGFeatureStage<F>>,
    /// First fully connected layer (512*7*7 -> fc_hidden_units)
    fc1: Dense<F>,
    /// Dropout after fc1
    dropout1: Dropout<F>,
    /// Second fully connected layer (fc_hidden_units -> fc_hidden_units)
    fc2: Dense<F>,
    /// Dropout after fc2
    dropout2: Dropout<F>,
    /// Third fully connected layer (fc_hidden_units -> num_classes)
    fc3: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> VGG<F> {
    /// Create a new VGG model from configuration
    pub fn new(config: VGGConfig) -> Result<Self> {
        let layer_configs = config.effective_layer_config();

        // Build feature stages
        let mut features = Vec::with_capacity(layer_configs.len());
        let mut in_channels = config.input_channels;
        for block_channels in &layer_configs {
            let stage = VGGFeatureStage::new(block_channels, in_channels, config.batch_norm)?;
            // The output channels of the last conv in this block
            in_channels = *block_channels.last().ok_or_else(|| {
                NeuralError::InvalidArchitecture("Empty block channel configuration".to_string())
            })?;
            features.push(stage);
        }

        // After 5 stages of 2x2 pooling, a 224x224 image becomes 7x7.
        // The last stage outputs `in_channels` channels (after divisor), so
        // the flattened size is in_channels * 7 * 7.
        // Adaptive avg pool to 7x7 ensures this works for any input size.
        let fc_input_size = in_channels * 7 * 7;

        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let fc1 = Dense::new(fc_input_size, config.fc_hidden_units, None, &mut rng)?;
        let dropout1 = Dropout::new(config.dropout_rate, &mut rng)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([43; 32]);
        let fc2 = Dense::new(
            config.fc_hidden_units,
            config.fc_hidden_units,
            None,
            &mut rng2,
        )?;
        let dropout2 = Dropout::new(config.dropout_rate, &mut rng2)?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([44; 32]);
        let fc3 = Dense::new(config.fc_hidden_units, config.num_classes, None, &mut rng3)?;

        Ok(Self {
            config,
            features,
            fc1,
            dropout1,
            fc2,
            dropout2,
            fc3,
        })
    }

    /// Create a VGG-11 model
    pub fn vgg11(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg11(input_channels, num_classes))
    }

    /// Create a VGG-11 model with batch normalization
    pub fn vgg11_bn(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg11_bn(input_channels, num_classes))
    }

    /// Create a VGG-13 model
    pub fn vgg13(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg13(input_channels, num_classes))
    }

    /// Create a VGG-13 model with batch normalization
    pub fn vgg13_bn(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg13_bn(input_channels, num_classes))
    }

    /// Create a VGG-16 model
    pub fn vgg16(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg16(input_channels, num_classes))
    }

    /// Create a VGG-16 model with batch normalization
    pub fn vgg16_bn(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg16_bn(input_channels, num_classes))
    }

    /// Create a VGG-19 model
    pub fn vgg19(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg19(input_channels, num_classes))
    }

    /// Create a VGG-19 model with batch normalization
    pub fn vgg19_bn(input_channels: usize, num_classes: usize) -> Result<Self> {
        Self::new(VGGConfig::vgg19_bn(input_channels, num_classes))
    }

    /// Get the model configuration
    pub fn config(&self) -> &VGGConfig {
        &self.config
    }

    /// Get the total number of trainable parameters
    pub fn total_parameter_count(&self) -> usize {
        let feature_params: usize = self.features.iter().map(|s| s.parameter_count()).sum();
        let classifier_params =
            self.fc1.parameter_count() + self.fc2.parameter_count() + self.fc3.parameter_count();
        feature_params + classifier_params
    }

    /// Get the number of feature extraction stages
    pub fn num_stages(&self) -> usize {
        self.features.len()
    }

    /// Extract features only (without the classifier).
    /// Returns the feature map after the last pooling layer.
    pub fn extract_features(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input [batch, channels, height, width], got shape {:?}",
                shape
            )));
        }
        if shape[1] != self.config.input_channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected {} input channels, got {}",
                self.config.input_channels, shape[1]
            )));
        }

        let mut x = input.clone();
        for stage in &self.features {
            x = stage.forward(&x)?;
        }
        Ok(x)
    }

    /// Adaptive average pooling to target spatial size.
    /// This allows inputs of various sizes to produce a fixed-size output.
    fn adaptive_avg_pool(
        input: &Array<F, IxDyn>,
        target_h: usize,
        target_w: usize,
    ) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input for adaptive avg pooling, got shape {:?}",
                shape
            )));
        }
        let batch_size = shape[0];
        let channels = shape[1];
        let in_h = shape[2];
        let in_w = shape[3];

        let mut output = Array::zeros(IxDyn(&[batch_size, channels, target_h, target_w]));

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..target_h {
                    for ow in 0..target_w {
                        // Compute input region for this output pixel
                        let h_start = (oh * in_h) / target_h;
                        let h_end = ((oh + 1) * in_h) / target_h;
                        let w_start = (ow * in_w) / target_w;
                        let w_end = ((ow + 1) * in_w) / target_w;

                        let mut sum = F::zero();
                        let mut count = 0usize;
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                sum += input[[b, c, h, w]];
                                count += 1;
                            }
                        }
                        let count_f = F::from(count).ok_or_else(|| {
                            NeuralError::InferenceError(
                                "Failed to convert pool count to float".to_string(),
                            )
                        })?;
                        if count > 0 {
                            output[[b, c, oh, ow]] = sum / count_f;
                        }
                    }
                }
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
    > VGG<F>
{
    /// Extract all named parameters in PyTorch/HuggingFace-compatible format.
    ///
    /// Feature parameters follow: `features.{stage}.{block}.conv.weight`,
    /// `features.{stage}.{block}.bn.weight`, etc.
    /// Classifier parameters: `classifier.0.weight`, `classifier.3.weight`, `classifier.6.weight`
    pub fn extract_named_params(&self) -> Vec<(String, Array<F, IxDyn>)> {
        let mut result = Vec::new();

        // Feature extraction layers
        // VGG PyTorch naming: features.0, features.2, features.5, etc.
        // We use a more structured naming: features.{stage_idx}.{block_idx}.conv/bn
        for (stage_idx, stage) in self.features.iter().enumerate() {
            for (block_idx, block) in stage.blocks.iter().enumerate() {
                // Conv params
                for (i, p) in block.conv.params().iter().enumerate() {
                    let suffix = if i == 0 { "weight" } else { "bias" };
                    result.push((
                        format!("features.{stage_idx}.{block_idx}.conv.{suffix}"),
                        p.clone(),
                    ));
                }
                // BN params (if present)
                if let Some(ref bn) = block.bn {
                    for (i, p) in bn.params().iter().enumerate() {
                        let suffix = if i == 0 { "weight" } else { "bias" };
                        result.push((
                            format!("features.{stage_idx}.{block_idx}.bn.{suffix}"),
                            p.clone(),
                        ));
                    }
                }
            }
        }

        // Classifier layers
        for (i, p) in self.fc1.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("classifier.0.{suffix}"), p.clone()));
        }
        for (i, p) in self.fc2.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("classifier.3.{suffix}"), p.clone()));
        }
        for (i, p) in self.fc3.params().iter().enumerate() {
            let suffix = if i == 0 { "weight" } else { "bias" };
            result.push((format!("classifier.6.{suffix}"), p.clone()));
        }

        result
    }

    /// Load named parameters from a map.
    ///
    /// Unknown parameter names are silently ignored for forward/backward compatibility.
    pub fn load_named_params(
        &mut self,
        params_map: &HashMap<String, Array<F, IxDyn>>,
    ) -> Result<()> {
        // Feature layers
        for (stage_idx, stage) in self.features.iter_mut().enumerate() {
            for (block_idx, block) in stage.blocks.iter_mut().enumerate() {
                // Conv
                let conv_weight_key = format!("features.{stage_idx}.{block_idx}.conv.weight");
                if let Some(w) = params_map.get(&conv_weight_key) {
                    let mut ps = vec![w.clone()];
                    let conv_bias_key = format!("features.{stage_idx}.{block_idx}.conv.bias");
                    if let Some(b) = params_map.get(&conv_bias_key) {
                        ps.push(b.clone());
                    }
                    block.conv.set_params(&ps)?;
                }
                // BN
                if let Some(ref mut bn) = block.bn {
                    let bn_weight_key = format!("features.{stage_idx}.{block_idx}.bn.weight");
                    if let Some(w) = params_map.get(&bn_weight_key) {
                        let mut ps = vec![w.clone()];
                        let bn_bias_key = format!("features.{stage_idx}.{block_idx}.bn.bias");
                        if let Some(b) = params_map.get(&bn_bias_key) {
                            ps.push(b.clone());
                        }
                        bn.set_params(&ps)?;
                    }
                }
            }
        }

        // Classifier fc1
        if let Some(w) = params_map.get("classifier.0.weight") {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get("classifier.0.bias") {
                ps.push(b.clone());
            }
            self.fc1.set_params(&ps)?;
        }
        // Classifier fc2
        if let Some(w) = params_map.get("classifier.3.weight") {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get("classifier.3.bias") {
                ps.push(b.clone());
            }
            self.fc2.set_params(&ps)?;
        }
        // Classifier fc3
        if let Some(w) = params_map.get("classifier.6.weight") {
            let mut ps = vec![w.clone()];
            if let Some(b) = params_map.get("classifier.6.bias") {
                ps.push(b.clone());
            }
            self.fc3.set_params(&ps)?;
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F> for VGG<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input [batch, channels, height, width], got shape {:?}",
                shape
            )));
        }
        if shape[1] != self.config.input_channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected {} input channels, got {}",
                self.config.input_channels, shape[1]
            )));
        }

        let batch_size = shape[0];

        // Feature extraction
        let mut x = input.clone();
        for stage in &self.features {
            x = stage.forward(&x)?;
        }

        // Adaptive average pooling to 7x7 (allows variable input sizes)
        x = Self::adaptive_avg_pool(&x, 7, 7)?;

        // Flatten: [batch, 512, 7, 7] -> [batch, 512*7*7]
        let channels = x.shape()[1];
        let height = x.shape()[2];
        let width = x.shape()[3];
        let flat_size = channels * height * width;
        let x = x
            .into_shape_with_order(IxDyn(&[batch_size, flat_size]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to flatten feature map: {}", e))
            })?;

        // Classifier: fc1 -> relu -> dropout -> fc2 -> relu -> dropout -> fc3
        let mut x = self.fc1.forward(&x)?;
        x = x.mapv(|v: F| v.max(F::zero())); // ReLU
        x = self.dropout1.forward(&x)?;

        x = self.fc2.forward(&x)?;
        x = x.mapv(|v: F| v.max(F::zero())); // ReLU
        x = self.dropout2.forward(&x)?;

        x = self.fc3.forward(&x)?;

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Gradient passthrough for compatibility; full backprop through features
        // would require caching intermediate activations per-stage.
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for stage in &mut self.features {
            stage.update(learning_rate)?;
        }
        self.fc1.update(learning_rate)?;
        self.fc2.update(learning_rate)?;
        self.fc3.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut result = Vec::new();
        for stage in &self.features {
            result.extend(stage.params());
        }
        result.extend(self.fc1.params());
        result.extend(self.fc2.params());
        result.extend(self.fc3.params());
        result
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "VGG"
    }

    fn layer_description(&self) -> String {
        format!(
            "VGG(variant={}, batch_norm={}, classes={}, params={})",
            self.config.variant.name(),
            self.config.batch_norm,
            self.config.num_classes,
            self.total_parameter_count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgg_variant_layer_counts() {
        assert_eq!(VGGVariant::VGG11.num_conv_layers(), 8);
        assert_eq!(VGGVariant::VGG13.num_conv_layers(), 10);
        assert_eq!(VGGVariant::VGG16.num_conv_layers(), 13);
        assert_eq!(VGGVariant::VGG19.num_conv_layers(), 16);
    }

    #[test]
    fn test_vgg_variant_names() {
        assert_eq!(VGGVariant::VGG11.name(), "VGG-11");
        assert_eq!(VGGVariant::VGG13.name(), "VGG-13");
        assert_eq!(VGGVariant::VGG16.name(), "VGG-16");
        assert_eq!(VGGVariant::VGG19.name(), "VGG-19");
    }

    #[test]
    fn test_vgg_config_vgg11() {
        let config = VGGConfig::vgg11(3, 1000);
        assert_eq!(config.input_channels, 3);
        assert_eq!(config.num_classes, 1000);
        assert!(!config.batch_norm);
        assert_eq!(config.variant, VGGVariant::VGG11);
        assert!((config.dropout_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vgg_config_vgg16_bn() {
        let config = VGGConfig::vgg16_bn(3, 100);
        assert!(config.batch_norm);
        assert_eq!(config.variant, VGGVariant::VGG16);
        assert_eq!(config.num_classes, 100);
    }

    #[test]
    fn test_vgg_config_builder_methods() {
        let config = VGGConfig::vgg19(3, 1000)
            .with_dropout(0.3)
            .with_batch_norm(true)
            .with_fc_hidden_units(2048);
        assert!((config.dropout_rate - 0.3).abs() < 1e-10);
        assert!(config.batch_norm);
        assert_eq!(config.fc_hidden_units, 2048);
    }

    #[test]
    fn test_vgg11_creation() {
        let model: VGG<f64> = VGG::vgg11(3, 10).expect("Failed to create VGG-11");
        assert_eq!(model.num_stages(), 5);
        assert_eq!(model.config().variant, VGGVariant::VGG11);
        assert!(model.total_parameter_count() > 0);
    }

    #[test]
    fn test_vgg16_creation() {
        let model: VGG<f64> = VGG::vgg16(3, 1000).expect("Failed to create VGG-16");
        assert_eq!(model.num_stages(), 5);
        // VGG-16 has ~138M parameters for 1000 classes with 4096 FC units
        let param_count = model.total_parameter_count();
        assert!(
            param_count > 100_000_000,
            "VGG-16 should have >100M params, got {}",
            param_count
        );
    }

    #[test]
    fn test_vgg19_bn_creation() {
        // Verify VGG19-BN config metadata directly (no heavy model construction needed)
        let config_bn = VGGConfig::vgg19_bn(3, 100);
        assert_eq!(config_bn.variant, VGGVariant::VGG19);
        assert!(config_bn.batch_norm);

        // Use a minimal channel configuration with f32 to keep construction fast
        let model_bn: VGG<f32> = VGG::new(
            VGGConfig::vgg19_bn(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(16)
                .with_channel_divisor(32),
        )
        .expect("Failed to create VGG-19-BN (scaled)");
        assert_eq!(model_bn.num_stages(), 5);
        assert!(model_bn.config().batch_norm);

        // BN variant should have more parameters than non-BN variant (same scale)
        let model_no_bn: VGG<f32> = VGG::new(
            VGGConfig::vgg19(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(16)
                .with_channel_divisor(32),
        )
        .expect("Failed to create VGG-19 (scaled)");
        assert!(
            model_bn.total_parameter_count() > model_no_bn.total_parameter_count(),
            "BN model params {} should exceed non-BN model params {}",
            model_bn.total_parameter_count(),
            model_no_bn.total_parameter_count()
        );
    }

    #[test]
    fn test_vgg_forward_pass() {
        // Use small input and scaled-down channels for fast test
        let model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(16)
                .with_channel_divisor(16),
        )
        .expect("Failed to create VGG");

        // Input: [batch=1, channels=1, height=32, width=32]
        // After 5 stages of 2x2 pooling: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        // Adaptive avg pool will bring it to 7x7
        let input = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        let output = model.forward(&input).expect("Forward pass failed");

        // Output shape should be [1, 10]
        assert_eq!(output.shape(), &[1, 10]);
    }

    #[test]
    fn test_vgg_forward_larger_input() {
        let model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 5)
                .with_dropout(0.0)
                .with_fc_hidden_units(16)
                .with_channel_divisor(16),
        )
        .expect("Failed to create VGG");

        // Larger input: [batch=2, channels=1, height=64, width=64]
        let input = Array::zeros(IxDyn(&[2, 1, 64, 64]));
        let output = model.forward(&input).expect("Forward pass failed");

        // Output shape should be [2, 5]
        assert_eq!(output.shape(), &[2, 5]);
    }

    #[test]
    fn test_vgg_feature_extraction() {
        // Use channel_divisor=16 to make channels: 4, 8, 16, 32, 32
        let model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(16)
                .with_channel_divisor(16),
        )
        .expect("Failed to create VGG");

        let input = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        let features = model
            .extract_features(&input)
            .expect("Feature extraction failed");

        // After 5 stages of 2x2 pooling: 32/2^5 = 1
        // Features should be [1, 32, 1, 1] (512/16=32)
        assert_eq!(features.shape()[0], 1);
        assert_eq!(features.shape()[1], 32); // 512 / channel_divisor(16) = 32
    }

    #[test]
    fn test_vgg_invalid_input_shape() {
        let model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(3, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(32),
        )
        .expect("Failed to create VGG");

        // Wrong number of dimensions
        let input_3d = Array::zeros(IxDyn(&[1, 3, 32]));
        assert!(model.forward(&input_3d).is_err());

        // Wrong number of input channels
        let input_wrong_channels = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        assert!(model.forward(&input_wrong_channels).is_err());
    }

    #[test]
    fn test_vgg_named_params() {
        let model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(32),
        )
        .expect("Failed to create VGG");

        let named_params = model.extract_named_params();
        assert!(!named_params.is_empty());

        // Should contain features and classifier params
        let has_feature_param = named_params
            .iter()
            .any(|(name, _)| name.starts_with("features."));
        let has_classifier_param = named_params
            .iter()
            .any(|(name, _)| name.starts_with("classifier."));
        assert!(has_feature_param, "Should have feature parameters");
        assert!(has_classifier_param, "Should have classifier parameters");
    }

    #[test]
    fn test_vgg_layer_trait() {
        let model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(32),
        )
        .expect("Failed to create VGG");

        assert_eq!(model.layer_type(), "VGG");
        assert!(model.parameter_count() > 0);
        let desc = model.layer_description();
        assert!(desc.contains("VGG-11"));
    }

    #[test]
    fn test_vgg_update() {
        let mut model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 10)
                .with_dropout(0.0)
                .with_fc_hidden_units(32),
        )
        .expect("Failed to create VGG");

        // Update should not panic
        model.update(0.001).expect("Update failed");
    }

    #[test]
    fn test_vgg_all_variants_create() {
        // Verify all variants can be created (with small FC and channel divisor to save memory)
        for variant in &[
            VGGVariant::VGG11,
            VGGVariant::VGG13,
            VGGVariant::VGG16,
            VGGVariant::VGG19,
        ] {
            let config = VGGConfig {
                variant: *variant,
                batch_norm: false,
                input_channels: 1,
                num_classes: 5,
                dropout_rate: 0.0,
                fc_hidden_units: 32,
                channel_divisor: 1,
            };
            let model: VGG<f64> = VGG::new(config).expect("Failed to create model");
            assert_eq!(model.config().variant, *variant);
        }
    }

    #[test]
    fn test_vgg_bn_variants_create() {
        for variant in &[
            VGGVariant::VGG11,
            VGGVariant::VGG13,
            VGGVariant::VGG16,
            VGGVariant::VGG19,
        ] {
            let config = VGGConfig {
                variant: *variant,
                batch_norm: true,
                input_channels: 1,
                num_classes: 5,
                dropout_rate: 0.0,
                fc_hidden_units: 32,
                channel_divisor: 1,
            };
            let model: VGG<f64> = VGG::new(config).expect("Failed to create BN model");
            assert!(model.config().batch_norm);
        }
    }

    #[test]
    fn test_max_pool_2x2() {
        // Create a simple 4x4 input
        let mut input = Array::zeros(IxDyn(&[1, 1, 4, 4]));
        // Fill with values
        input[[0, 0, 0, 0]] = 1.0_f64;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;
        input[[0, 0, 2, 2]] = 5.0;
        input[[0, 0, 2, 3]] = 6.0;
        input[[0, 0, 3, 2]] = 7.0;
        input[[0, 0, 3, 3]] = 8.0;

        let output = max_pool_2x2(&input).expect("Max pool failed");

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        // Top-left: max(1, 2, 3, 4) = 4
        assert!((output[[0, 0, 0, 0]] - 4.0).abs() < 1e-10);
        // Bottom-right: max(5, 6, 7, 8) = 8
        assert!((output[[0, 0, 1, 1]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_vgg_load_named_params() {
        let mut model: VGG<f64> = VGG::new(
            VGGConfig::vgg11(1, 5)
                .with_dropout(0.0)
                .with_fc_hidden_units(32),
        )
        .expect("Failed to create VGG");

        // Extract params, then reload them (round-trip test)
        let named_params = model.extract_named_params();
        let params_map: HashMap<String, Array<f64, IxDyn>> = named_params.into_iter().collect();
        model
            .load_named_params(&params_map)
            .expect("Load named params failed");
    }

    #[test]
    fn test_vgg_f32_support() {
        let model: VGG<f32> = VGG::new(
            VGGConfig::vgg11(1, 5)
                .with_dropout(0.0)
                .with_fc_hidden_units(16)
                .with_channel_divisor(16),
        )
        .expect("Failed to create VGG f32");

        let input = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        let output = model.forward(&input).expect("Forward pass failed for f32");
        assert_eq!(output.shape(), &[1, 5]);
    }
}
