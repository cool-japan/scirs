//! EfficientNet implementation
//!
//! EfficientNet is a convolutional neural network architecture that uses
//! compound scaling to systematically scale all dimensions of depth, width, and resolution.
//!
//! Reference: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", Tan & Le (2019)
//! https://arxiv.org/abs/1905.11946

use crate::error::{NeuralError, Result};
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, PaddingMode};
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt::Debug;

/// Swish activation function used in EfficientNet
pub fn swish<F: Float>(x: F) -> F {
    x * (F::one() + (-x).exp()).recip()
}

/// Configuration for the MBConv block in EfficientNet
#[derive(Debug, Clone)]
pub struct MBConvConfig {
    /// Input channels
    pub input_channels: usize,
    /// Output channels
    pub output_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Expansion ratio
    pub expand_ratio: usize,
    /// Whether to use squeeze and excitation
    pub use_se: bool,
    /// Dropout rate for stochastic depth
    pub drop_connect_rate: f64,
}

/// Configuration for a stage of EfficientNet
#[derive(Debug, Clone)]
pub struct EfficientNetStage {
    /// MBConv block configuration
    pub mbconv_config: MBConvConfig,
    /// Number of blocks in this stage
    pub num_blocks: usize,
}

/// Configuration for an EfficientNet model
#[derive(Debug, Clone)]
pub struct EfficientNetConfig {
    /// Width multiplier
    pub width_coefficient: f64,
    /// Depth multiplier
    pub depth_coefficient: f64,
    /// Resolution multiplier
    pub resolution: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Stage configurations
    pub stages: Vec<EfficientNetStage>,
    /// Number of input channels (e.g., 3 for RGB)
    pub input_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
}

impl EfficientNetConfig {
    /// Create EfficientNet-B0 configuration
    pub fn efficientnet_b0(input_channels: usize, num_classes: usize) -> Self {
        let mut stages = Vec::new();

        // Stage 1: MBConv1, 16 channels, 1 block
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 32,
                output_channels: 16,
                kernel_size: 3,
                stride: 1,
                expand_ratio: 1,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 1,
        });

        // Stage 2: MBConv6, 24 channels, 2 blocks
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 16,
                output_channels: 24,
                kernel_size: 3,
                stride: 2,
                expand_ratio: 6,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 2,
        });

        // Stage 3: MBConv6, 40 channels, 2 blocks
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 24,
                output_channels: 40,
                kernel_size: 5,
                stride: 2,
                expand_ratio: 6,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 2,
        });

        // Stage 4: MBConv6, 80 channels, 3 blocks
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 40,
                output_channels: 80,
                kernel_size: 3,
                stride: 2,
                expand_ratio: 6,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 3,
        });

        // Stage 5: MBConv6, 112 channels, 3 blocks
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 80,
                output_channels: 112,
                kernel_size: 5,
                stride: 1,
                expand_ratio: 6,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 3,
        });

        // Stage 6: MBConv6, 192 channels, 4 blocks
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 112,
                output_channels: 192,
                kernel_size: 5,
                stride: 2,
                expand_ratio: 6,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 4,
        });

        // Stage 7: MBConv6, 320 channels, 1 block
        stages.push(EfficientNetStage {
            mbconv_config: MBConvConfig {
                input_channels: 192,
                output_channels: 320,
                kernel_size: 3,
                stride: 1,
                expand_ratio: 6,
                use_se: true,
                drop_connect_rate: 0.2,
            },
            num_blocks: 1,
        });

        Self {
            width_coefficient: 1.0,
            depth_coefficient: 1.0,
            resolution: 224,
            dropout_rate: 0.2,
            stages,
            input_channels,
            num_classes,
        }
    }

    /// Create EfficientNet-B1 configuration
    pub fn efficientnet_b1(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 1.0;
        config.depth_coefficient = 1.1;
        config.resolution = 240;
        config.dropout_rate = 0.2;
        config
    }

    /// Create EfficientNet-B2 configuration
    pub fn efficientnet_b2(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 1.1;
        config.depth_coefficient = 1.2;
        config.resolution = 260;
        config.dropout_rate = 0.3;
        config
    }

    /// Create EfficientNet-B3 configuration
    pub fn efficientnet_b3(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 1.2;
        config.depth_coefficient = 1.4;
        config.resolution = 300;
        config.dropout_rate = 0.3;
        config
    }

    /// Create EfficientNet-B4 configuration
    pub fn efficientnet_b4(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 1.4;
        config.depth_coefficient = 1.8;
        config.resolution = 380;
        config.dropout_rate = 0.4;
        config
    }

    /// Create EfficientNet-B5 configuration
    pub fn efficientnet_b5(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 1.6;
        config.depth_coefficient = 2.2;
        config.resolution = 456;
        config.dropout_rate = 0.4;
        config
    }

    /// Create EfficientNet-B6 configuration
    pub fn efficientnet_b6(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 1.8;
        config.depth_coefficient = 2.6;
        config.resolution = 528;
        config.dropout_rate = 0.5;
        config
    }

    /// Create EfficientNet-B7 configuration
    pub fn efficientnet_b7(input_channels: usize, num_classes: usize) -> Self {
        let mut config = Self::efficientnet_b0(input_channels, num_classes);
        config.width_coefficient = 2.0;
        config.depth_coefficient = 3.1;
        config.resolution = 600;
        config.dropout_rate = 0.5;
        config
    }

    /// Scale channels based on width coefficient
    pub fn scale_channels(&self, channels: usize) -> usize {
        let scaled = (channels as f64 * self.width_coefficient).round();
        // Ensure divisibility by 8
        ((scaled as usize) + 7) / 8 * 8
    }

    /// Scale depth based on depth coefficient
    pub fn scale_depth(&self, depth: usize) -> usize {
        (depth as f64 * self.depth_coefficient).ceil() as usize
    }
}

/// Squeeze and Excitation block
struct SqueezeExcitation<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Input channels
    input_channels: usize,
    /// Squeeze channels
    #[allow(dead_code)]
    squeeze_channels: usize,
    /// First convolution (squeeze)
    fc1: Conv2D<F>,
    /// Second convolution (excite)
    fc2: Conv2D<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> SqueezeExcitation<F> {
    /// Create a new Squeeze and Excitation block
    pub fn new(input_channels: usize, squeeze_channels: usize) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);

        // First 1x1 convolution (squeeze)
        let fc1 = Conv2D::new(
            input_channels,
            squeeze_channels,
            (1, 1),
            (1, 1),
            PaddingMode::Valid,
            &mut rng,
        )?;

        // Second 1x1 convolution (excite)
        let fc2 = Conv2D::new(
            squeeze_channels,
            input_channels,
            (1, 1),
            (1, 1),
            PaddingMode::Valid,
            &mut rng,
        )?;

        Ok(Self {
            input_channels,
            squeeze_channels,
            fc1,
            fc2,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for SqueezeExcitation<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Input shape [batch_size, channels, height, width]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input, got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        if channels != self.input_channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected {} input channels, got {}",
                self.input_channels, channels
            )));
        }

        // Global average pooling
        let mut x = Array::zeros(IxDyn(&[batch_size, channels, 1, 1]));

        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = F::zero();
                for h in 0..height {
                    for w in 0..width {
                        sum = sum + input[[b, c, h, w]];
                    }
                }
                x[[b, c, 0, 0]] = sum / F::from(height * width).unwrap();
            }
        }

        // Apply squeeze
        let x = self.fc1.forward(&x)?;

        // Apply ReLU
        let x = x.mapv(|v| v.max(F::zero()));

        // Apply excite
        let x = self.fc2.forward(&x)?;

        // Apply sigmoid
        let x = x.mapv(|v| F::one() / (F::one() + (-v).exp()));

        // Scale input
        let mut result = input.clone();
        for b in 0..batch_size {
            for c in 0..channels {
                let scale = x[[b, c, 0, 0]];
                for h in 0..height {
                    for w in 0..width {
                        result[[b, c, h, w]] = input[[b, c, h, w]] * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a real implementation, we'd compute gradients for all parameters
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.fc1.update(learning_rate)?;
        self.fc2.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Mobile Inverted Bottleneck Convolution (MBConv) block
struct MBConvBlock<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Block configuration
    #[allow(dead_code)]
    config: MBConvConfig,
    /// Whether to use skip connection
    has_skip_connection: bool,
    /// Expansion convolution (optional)
    expand_conv: Option<Conv2D<F>>,
    /// Expansion batch normalization (optional)
    expand_bn: Option<BatchNorm<F>>,
    /// Depthwise convolution
    depthwise_conv: Conv2D<F>,
    /// Depthwise batch normalization
    depthwise_bn: BatchNorm<F>,
    /// Squeeze and excitation block (optional)
    se: Option<SqueezeExcitation<F>>,
    /// Projection convolution
    project_conv: Conv2D<F>,
    /// Projection batch normalization
    project_bn: BatchNorm<F>,
    /// Drop connect rate
    drop_connect_rate: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> MBConvBlock<F> {
    /// Create a new MBConv block
    pub fn new(config: MBConvConfig) -> Result<Self> {
        let input_channels = config.input_channels;
        let output_channels = config.output_channels;
        let expand_ratio = config.expand_ratio;
        let kernel_size = config.kernel_size;
        let stride = config.stride;
        let use_se = config.use_se;
        let drop_connect_rate = F::from(config.drop_connect_rate).unwrap();

        // Check if we use skip connection
        let has_skip_connection = input_channels == output_channels && stride == 1;

        // Create expansion convolution if needed
        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            let expanded_channels = input_channels * expand_ratio;

            // Expansion convolution (1x1)
            let mut rng = SmallRng::seed_from_u64(42);
            let conv = Conv2D::new(
                input_channels,
                expanded_channels,
                (1, 1),
                (1, 1),
                PaddingMode::Valid,
                &mut rng,
            )?;

            // Batch normalization
            let mut rng = SmallRng::seed_from_u64(42);
            let bn = BatchNorm::new(expanded_channels, 1e-3, 0.01, &mut rng)?;

            (Some(conv), Some(bn))
        } else {
            (None, None)
        };

        // Get expanded channels
        let expanded_channels = if expand_ratio != 1 {
            input_channels * expand_ratio
        } else {
            input_channels
        };

        // Create depthwise convolution
        let mut rng = SmallRng::seed_from_u64(42);
        let depthwise_conv = Conv2D::new(
            expanded_channels,
            expanded_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            PaddingMode::Same,
            &mut rng,
        )?;

        // Depthwise batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let depthwise_bn = BatchNorm::new(expanded_channels, 1e-3, 0.01, &mut rng)?;

        // Create squeeze and excitation block if needed
        let se = if use_se {
            let squeeze_channels = (expanded_channels as f64 / 4.0).round() as usize;
            Some(SqueezeExcitation::new(expanded_channels, squeeze_channels)?)
        } else {
            None
        };

        // Create projection convolution (1x1)
        let mut rng = SmallRng::seed_from_u64(42);
        let project_conv = Conv2D::new(
            expanded_channels,
            output_channels,
            (1, 1),
            (1, 1),
            PaddingMode::Valid,
            &mut rng,
        )?;

        // Projection batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let project_bn = BatchNorm::new(output_channels, 1e-3, 0.01, &mut rng)?;

        Ok(Self {
            config,
            has_skip_connection,
            expand_conv,
            expand_bn,
            depthwise_conv,
            depthwise_bn,
            se,
            project_conv,
            project_bn,
            drop_connect_rate,
        })
    }

    /// Apply drop connection (stochastic depth)
    fn drop_connect<R: rand::Rng>(&self, input: &Array<F, IxDyn>, rng: &mut R) -> Array<F, IxDyn> {
        if self.drop_connect_rate <= F::zero() || !self.has_skip_connection {
            return input.clone();
        }

        // Generate a random tensor for binary mask
        let keep_prob = F::one() - self.drop_connect_rate;
        let shape = input.shape();
        let mut result = input.clone();

        if rng.random::<f64>() > self.drop_connect_rate.to_f64().unwrap() {
            // Correct the drop value to maintain same expectation
            result = result.mapv(|x| x / keep_prob);
        } else {
            // Drop entire residual path
            result = Array::zeros(IxDyn(shape));
        }

        result
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for MBConvBlock<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = input.clone();

        // Expansion phase
        if let (Some(ref expand_conv), Some(ref expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            x = expand_conv.forward(&x)?;
            x = expand_bn.forward(&x)?;
            x = x.mapv(swish); // Apply Swish activation
        }

        // Depthwise convolution phase
        x = self.depthwise_conv.forward(&x)?;
        x = self.depthwise_bn.forward(&x)?;
        x = x.mapv(swish); // Apply Swish activation

        // Squeeze and excitation phase
        if let Some(ref se) = self.se {
            x = se.forward(&x)?;
        }

        // Projection phase
        x = self.project_conv.forward(&x)?;
        x = self.project_bn.forward(&x)?;

        // Skip connection
        if self.has_skip_connection {
            // Apply stochastic depth (drop connect)
            let mut rng = rand::rng();
            x = self.drop_connect(&x, &mut rng);

            // Add skip connection
            let mut result = input.clone();
            for i in 0..result.len() {
                result[i] = result[i] + x[i];
            }
            x = result;
        }

        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a real implementation, we'd compute gradients for all parameters
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update expansion phase
        if let (Some(ref mut expand_conv), Some(ref mut expand_bn)) =
            (&mut self.expand_conv, &mut self.expand_bn)
        {
            expand_conv.update(learning_rate)?;
            expand_bn.update(learning_rate)?;
        }

        // Update depthwise convolution phase
        self.depthwise_conv.update(learning_rate)?;
        self.depthwise_bn.update(learning_rate)?;

        // Update squeeze and excitation phase
        if let Some(ref mut se) = self.se {
            se.update(learning_rate)?;
        }

        // Update projection phase
        self.project_conv.update(learning_rate)?;
        self.project_bn.update(learning_rate)?;

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// EfficientNet implementation
pub struct EfficientNet<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Model configuration
    config: EfficientNetConfig,
    /// Initial convolution
    stem_conv: Conv2D<F>,
    /// Initial batch normalization
    stem_bn: BatchNorm<F>,
    /// MBConv blocks
    blocks: Vec<MBConvBlock<F>>,
    /// Final convolution
    head_conv: Conv2D<F>,
    /// Final batch normalization
    head_bn: BatchNorm<F>,
    /// Classifier
    classifier: Dense<F>,
    /// Dropout
    dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> EfficientNet<F> {
    /// Create a new EfficientNet model
    pub fn new(config: EfficientNetConfig) -> Result<Self> {
        let input_channels = config.input_channels;
        let num_classes = config.num_classes;

        // Initial stem convolution
        let stem_channels = config.scale_channels(32);
        let mut rng = SmallRng::seed_from_u64(42);
        let stem_conv = Conv2D::new(
            input_channels,
            stem_channels,
            (3, 3),
            (2, 2),
            PaddingMode::Same,
            &mut rng,
        )?;

        // Initial batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let stem_bn = BatchNorm::new(stem_channels, 1e-3, 0.01, &mut rng)?;

        // Create MBConv blocks
        let mut blocks = Vec::new();
        let mut in_channels = stem_channels;

        for stage in &config.stages {
            // Scale stage parameters
            let num_blocks = config.scale_depth(stage.num_blocks);
            let out_channels = config.scale_channels(stage.mbconv_config.output_channels);

            // First block may have different input channels and stride
            let first_block_config = MBConvConfig {
                input_channels: in_channels,
                output_channels: out_channels,
                kernel_size: stage.mbconv_config.kernel_size,
                stride: stage.mbconv_config.stride,
                expand_ratio: stage.mbconv_config.expand_ratio,
                use_se: stage.mbconv_config.use_se,
                drop_connect_rate: stage.mbconv_config.drop_connect_rate,
            };

            blocks.push(MBConvBlock::new(first_block_config)?);

            // Remaining blocks have stride 1 and same input/output channels
            for _ in 1..num_blocks {
                let block_config = MBConvConfig {
                    input_channels: out_channels,
                    output_channels: out_channels,
                    kernel_size: stage.mbconv_config.kernel_size,
                    stride: 1,
                    expand_ratio: stage.mbconv_config.expand_ratio,
                    use_se: stage.mbconv_config.use_se,
                    drop_connect_rate: stage.mbconv_config.drop_connect_rate,
                };

                blocks.push(MBConvBlock::new(block_config)?);
            }

            in_channels = out_channels;
        }

        // Final convolution
        let head_channels = config.scale_channels(1280);
        let mut rng = SmallRng::seed_from_u64(42);
        let head_conv = Conv2D::new(
            in_channels,
            head_channels,
            (1, 1),
            (1, 1),
            PaddingMode::Valid,
            &mut rng,
        )?;

        // Final batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let head_bn = BatchNorm::new(head_channels, 1e-3, 0.01, &mut rng)?;

        // Classifier
        let mut rng = SmallRng::seed_from_u64(42);
        let classifier = Dense::new(head_channels, num_classes, None, &mut rng)?;

        // Dropout
        let mut rng = SmallRng::seed_from_u64(42);
        let dropout = Dropout::new(config.dropout_rate, &mut rng)?;

        Ok(Self {
            config,
            stem_conv,
            stem_bn,
            blocks,
            head_conv,
            head_bn,
            classifier,
            dropout,
        })
    }

    /// Create EfficientNet-B0 model
    pub fn efficientnet_b0(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b0(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B1 model
    pub fn efficientnet_b1(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b1(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B2 model
    pub fn efficientnet_b2(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b2(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B3 model
    pub fn efficientnet_b3(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b3(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B4 model
    pub fn efficientnet_b4(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b4(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B5 model
    pub fn efficientnet_b5(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b5(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B6 model
    pub fn efficientnet_b6(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b6(input_channels, num_classes);
        Self::new(config)
    }

    /// Create EfficientNet-B7 model
    pub fn efficientnet_b7(input_channels: usize, num_classes: usize) -> Result<Self> {
        let config = EfficientNetConfig::efficientnet_b7(input_channels, num_classes);
        Self::new(config)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for EfficientNet<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Check input shape
        let shape = input.shape();
        if shape.len() != 4 || shape[1] != self.config.input_channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, {}, height, width], got {:?}",
                self.config.input_channels, shape
            )));
        }

        let batch_size = shape[0];

        // Stem
        let mut x = self.stem_conv.forward(input)?;
        x = self.stem_bn.forward(&x)?;
        x = x.mapv(swish); // Apply Swish activation

        // Blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Head
        x = self.head_conv.forward(&x)?;
        x = self.head_bn.forward(&x)?;
        x = x.mapv(swish); // Apply Swish activation

        // Global average pooling
        let channels = x.shape()[1];
        let height = x.shape()[2];
        let width = x.shape()[3];
        let mut pooled = Array::zeros(IxDyn(&[batch_size, channels]));

        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = F::zero();
                for h in 0..height {
                    for w in 0..width {
                        sum = sum + x[[b, c, h, w]];
                    }
                }
                pooled[[b, c]] = sum / F::from(height * width).unwrap();
            }
        }

        // Dropout
        let pooled = self.dropout.forward(&pooled)?;

        // Classifier
        let logits = self.classifier.forward(&pooled)?;

        Ok(logits)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a real implementation, we'd compute gradients for all parameters
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update stem
        self.stem_conv.update(learning_rate)?;
        self.stem_bn.update(learning_rate)?;

        // Update blocks
        for block in &mut self.blocks {
            block.update(learning_rate)?;
        }

        // Update head
        self.head_conv.update(learning_rate)?;
        self.head_bn.update(learning_rate)?;

        // Update classifier
        self.classifier.update(learning_rate)?;

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
