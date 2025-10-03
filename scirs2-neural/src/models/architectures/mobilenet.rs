//! MobileNet implementation
//!
//! MobileNet is a lightweight convolutional neural network architecture designed for
//! mobile and embedded vision applications. It uses depthwise separable convolutions
//! to reduce the number of parameters and computational cost.
//! References:
//! - MobileNetV1: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", Howard et al. (2017)
//!   https://arxiv.org/abs/1704.04861
//! - MobileNetV2: "MobileNetV2: Inverted Residuals and Linear Bottlenecks", Sandler et al. (2018)
//!   https://arxiv.org/abs/1801.04381
//! - MobileNetV3: "Searching for MobileNetV3", Howard et al. (2019)
//!   https://arxiv.org/abs/1905.02244

use crate::error::{NeuralError, Result};
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, PaddingMode};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::SeedableRng;
use std::fmt::Debug;
/// ReLU6 activation function (min(max(x, 0), 6))
#[allow(dead_code)]
pub fn relu6<F: Float>(x: F) -> F {
    let zero = F::zero();
    let six = F::from(6.0).unwrap();
    x.max(zero).min(six)
}
/// Hard Sigmoid activation function used in MobileNetV3
#[allow(dead_code)]
pub fn hard_sigmoid<F: Float>(x: F) -> F {
    let _one = F::one(); // Not used but kept for consistency
    let three = F::from(3.0).unwrap();
    (x + three).max(zero).min(six) / six
/// Hard Swish activation function used in MobileNetV3
#[allow(dead_code)]
pub fn hard_swish<F: Float>(x: F) -> F {
    x * hard_sigmoid(x)
/// Version of MobileNet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobileNetVersion {
    /// MobileNetV1
    V1,
    /// MobileNetV2
    V2,
    /// MobileNetV3-Small
    V3Small,
    /// MobileNetV3-Large
    V3Large,
/// Configuration for a convolutional block in MobileNet
#[derive(Debug, Clone)]
pub struct ConvBlockConfig {
    /// Input channels
    pub input_channels: usize,
    /// Output channels
    pub output_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Whether to use residual connection
    pub use_residual: bool,
    /// Expansion factor (for MobileNetV2/V3)
    pub expand_ratio: usize,
    /// Whether to use squeeze and excitation (for MobileNetV3)
    pub use_se: bool,
    /// Activation function name
    pub activation: String,
/// Configuration for a MobileNet model
pub struct MobileNetConfig {
    /// Version of MobileNet
    pub version: MobileNetVersion,
    /// Width multiplier to scale model (alpha)
    pub width_multiplier: f64,
    /// Resolution multiplier (for MobileNetV1)
    pub resolution_multiplier: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Block configurations
    pub blocks: Vec<ConvBlockConfig>,
    /// Number of input channels (e.g., 3 for RGB)
    /// Number of output classes
    pub num_classes: usize,
impl MobileNetConfig {
    /// Create MobileNetV1 configuration
    pub fn mobilenet_v1(_input_channels: usize, numclasses: usize) -> Self {
        let mut blocks = Vec::new();
        // First standard convolution
        blocks.push(ConvBlockConfig {
            input_channels,
            output_channels: 32,
            kernel_size: 3,
            stride: 2,
            use_residual: false,
            expand_ratio: 1,
            use_se: false,
            activation: "relu".to_string(),
        });
        // Depthwise separable convolutions
        let channels = [
            64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024,
        ];
        let strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1];
        let mut input_c = 32;
        for i in 0..channels.len() {
            blocks.push(ConvBlockConfig {
                input_channels: input_c,
                output_channels: channels[i],
                kernel_size: 3,
                stride: strides[i],
                use_residual: false,
                expand_ratio: 1,
                use_se: false,
                activation: "relu".to_string(),
            });
            input_c = channels[i];
        }
        Self {
            version: MobileNetVersion::V1,
            width_multiplier: 1.0,
            resolution_multiplier: 1.0,
            dropout_rate: 0.001,
            blocks,
            num_classes,
    }
    /// Create MobileNetV2 configuration
    pub fn mobilenet_v2(_input_channels: usize, numclasses: usize) -> Self {
            activation: "relu6".to_string(),
        // Inverted residual blocks
        let t = [1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]; // Expansion factors
        let c = [
            16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320,
        ]; // Output channels
        let s = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1]; // Strides
        let r = [
            false, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, false,
        ]; // Residual
        for i in 0..t.len() {
                output_channels: c[i],
                stride: s[i],
                use_residual: r[i],
                expand_ratio: t[i],
                activation: "relu6".to_string(),
            input_c = c[i];
        // Final point-wise convolution
            input_channels: 320,
            output_channels: 1280,
            kernel_size: 1,
            stride: 1,
            version: MobileNetVersion::V2,
    /// Create MobileNetV3-Small configuration
    pub fn mobilenet_v3_small(_input_channels: usize, numclasses: usize) -> Self {
            output_channels: 16,
            activation: "hard_swish".to_string(),
        // Define the network architecture
        // [exp_ratio, out_ch, kernel, stride, use_se, activation, use_res]
        let configs = vec![
            (1.0, 16, 3, 2, true, "relu", false),
            (4.0, 24, 3, 2, false, "relu", false),
            (3.0, 24, 3, 1, false, "relu", true),
            (3.0, 40, 5, 2, true, "hard_swish", false),
            (3.0, 40, 5, 1, true, "hard_swish", true),
            (6.0, 80, 3, 2, false, "hard_swish", false),
            (2.5, 80, 3, 1, false, "hard_swish", true),
            (2.3, 80, 3, 1, false, "hard_swish", true),
            (6.0, 112, 3, 1, true, "hard_swish", false),
            (6.0, 112, 3, 1, true, "hard_swish", true),
            (6.0, 160, 5, 2, true, "hard_swish", false),
            (6.0, 160, 5, 1, true, "hard_swish", true),
        let mut input_c = 16;
        for (exp_ratio, out_ch, kernel, stride, use_se, activation, use_res) in configs {
            let expand_ratio = (exp_ratio * 100.0).round() as usize / 100;
                output_channels: out_ch,
                kernel_size: kernel,
                stride,
                use_residual: use_res,
                expand_ratio,
                use_se,
                activation: activation.to_string(),
            input_c = out_ch;
        // Final layers
            input_channels: 160,
            output_channels: 960,
            input_channels: 960,
            version: MobileNetVersion::V3Small,
            dropout_rate: 0.2,
    /// Create MobileNetV3-Large configuration
    pub fn mobilenet_v3_large(_input_channels: usize, numclasses: usize) -> Self {
            (1.0, 16, 3, 1, false, "relu", false),
            (3.0, 40, 5, 2, true, "relu", false),
            (3.0, 40, 5, 1, true, "relu", true),
            version: MobileNetVersion::V3Large,
    /// Scale channels based on width multiplier
    pub fn scale_channels(&self, channels: usize) -> usize {
        let scaled = (channels as f64 * self.width_multiplier).round();
        scaled.max(8.0) as usize // Ensure minimum of 8 channels
/// Squeeze and Excitation block for MobileNetV3
struct SqueezeExcitation<F: Float + Debug + ScalarOperand + Send + Sync> {
    input_channels: usize,
    /// Squeeze channels
    #[allow(dead_code)]
    squeeze_channels: usize,
    /// First convolution (squeeze)
    fc1: Conv2D<F>,
    /// Second convolution (excite)
    fc2: Conv2D<F>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> SqueezeExcitation<F> {
    /// Create a new Squeeze and Excitation block
    pub fn new(_input_channels: usize, squeezechannels: usize) -> Result<Self> {
        // First 1x1 convolution (squeeze)
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let kernel_size = (1, 1);
        let stride = (1, 1);
        let fc1 = Conv2D::new(
            squeeze_channels,
            kernel_size,
            stride,
            PaddingMode::Valid,
            &mut rng,
        )?;
        // Second 1x1 convolution (excite)
        let fc2 = Conv2D::new(
        Ok(Self {
            fc1,
            fc2,
        })
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for SqueezeExcitation<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Input shape [batch_size, channels, height, width]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input, got {:?}",
                shape
            )));
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        if channels != self.input_channels {
                "Expected {} input channels, got {}",
                self.input_channels, channels
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
        // Apply squeeze
        let x = self.fc1.forward(&x)?;
        // Apply ReLU
        let x = x.mapv(|v| v.max(F::zero()));
        // Apply excite
        let x = self.fc2.forward(&x)?;
        // Apply hard-sigmoid
        let x = x.mapv(hard_sigmoid);
        // Scale input
        let mut result = input.clone();
                let scale = x[[b, c, 0, 0]];
                        result[[b, c, h, w]] = input[[b, c, h, w]] * scale;
        Ok(result)
    fn backward(
        &mut self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a real implementation, we'd compute gradients for all parameters
        Ok(grad_output.clone())
    fn update(&mut self, learningrate: F) -> Result<()> {
        self.fc1.update(learning_rate)?;
        self.fc2.update(learning_rate)?;
        Ok(())
    fn as_any(&self) -> &dyn std::any::Any {
        self
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
/// Activation function
#[allow(dead_code)]
fn get_activation<F: Float + Debug + ScalarOperand + Send + Sync>(
    name: &str,
) -> Box<dyn Fn(F) -> F + Send + Sync> {
    match name {
        "relu" => Box::new(|x: F| x.max(F::zero())),
        "relu6" => Box::new(relu6),
        "hard_swish" => Box::new(hard_swish),
        "hard_sigmoid" => Box::new(hard_sigmoid, _ => Box::new(|x: F| x.max(F::zero())), // Default to ReLU
/// Mobile Inverted Bottleneck Convolution block for MobileNetV2/V3
struct InvertedResidualBlock<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Block configuration
    config: ConvBlockConfig,
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
    /// Activation function
    activation: Box<dyn Fn(F) -> F + Send + Sync>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> InvertedResidualBlock<F> {
    /// Create a new Inverted Residual block
    pub fn new(_config: ConvBlockConfig, widthmultiplier: f64) -> Result<Self> {
        let input_channels = (_config.input_channels as f64 * width_multiplier).round() as usize;
        let output_channels = (_config.output_channels as f64 * width_multiplier).round() as usize;
        let expand_ratio = config.expand_ratio;
        let kernel_size = config.kernel_size;
        let stride = config.stride;
        let use_se = config.use_se;
        // Check if we use skip connection
        let has_skip_connection =
            input_channels == output_channels && stride == 1 && config.use_residual;
        // Create expansion convolution if needed
        let (expand_conv, expand_bn) = if expand_ratio != 1 {
            let expanded_channels = input_channels * expand_ratio;
            // Expansion convolution (1x1)
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
            let kernel_size = (1, 1);
            let stride = (1, 1);
            let conv = Conv2D::new(
                input_channels,
                expanded_channels,
                kernel_size,
                PaddingMode::Valid,
                &mut rng,
            )?;
            // Batch normalization
            let bn = BatchNorm::new(expanded_channels, 1e-3, 0.01, &mut rng)?;
            (Some(conv), Some(bn))
        } else {
            (None, None)
        };
        // Get expanded channels
        let expanded_channels = if expand_ratio != 1 {
            input_channels * expand_ratio
            input_channels
        // Create depthwise convolution
        let kernel_size_tuple = (kernel_size, kernel_size);
        let stride_tuple = (stride, stride);
        let depthwise_conv = Conv2D::new(
            expanded_channels,
            kernel_size_tuple,
            stride_tuple,
            PaddingMode::Same,
        // Depthwise batch normalization
        let depthwise_bn = BatchNorm::new(expanded_channels, 1e-3, 0.01, &mut rng)?;
        // Create squeeze and excitation block if needed
        let se = if use_se {
            let squeeze_channels = (expanded_channels as f64 / 4.0).round() as usize;
            Some(SqueezeExcitation::new(expanded_channels, squeeze_channels)?)
            None
        // Create projection convolution (1x1)
        let project_conv = Conv2D::new(
            output_channels,
            (1, 1),
        // Projection batch normalization
        let project_bn = BatchNorm::new(output_channels, 1e-3, 0.01, &mut rng)?;
        // Activation function
        let activation = get_activation(&config.activation);
            config,
            has_skip_connection,
            expand_conv,
            expand_bn,
            depthwise_conv,
            depthwise_bn,
            se,
            project_conv,
            project_bn,
            activation,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for InvertedResidualBlock<F> {
        let mut x = input.clone();
        // Expansion phase
        if let (Some(ref expand_conv), Some(ref expand_bn)) = (&self.expand_conv, &self.expand_bn) {
            x = expand_conv.forward(&x)?;
            x = expand_bn.forward(&x)?;
            x = x.mapv(|v| (self.activation)(v)); // Apply activation
        // Depthwise convolution phase
        x = self.depthwise_conv.forward(&x)?;
        x = self.depthwise_bn.forward(&x)?;
        x = x.mapv(|v| (self.activation)(v)); // Apply activation
        // Squeeze and excitation phase
        if let Some(ref se) = self.se {
            x = se.forward(&x)?;
        // Projection phase
        x = self.project_conv.forward(&x)?;
        x = self.project_bn.forward(&x)?;
        // Skip connection (residual)
        if self.has_skip_connection {
            // Add skip connection
            let mut result = input.clone();
            for i in 0..result.len() {
                result[i] = result[i] + x[i];
            x = result;
        Ok(x)
        // Update expansion phase
        if let (Some(ref mut expand_conv), Some(ref mut expand_bn)) =
            (&mut self.expand_conv, &mut self.expand_bn)
        {
            expand_conv.update(learning_rate)?;
            expand_bn.update(learning_rate)?;
        // Update depthwise convolution phase
        self.depthwise_conv.update(learning_rate)?;
        self.depthwise_bn.update(learning_rate)?;
        // Update squeeze and excitation phase
        if let Some(ref mut se) = self.se {
            se.update(learning_rate)?;
        // Update projection phase
        self.project_conv.update(learning_rate)?;
        self.project_bn.update(learning_rate)?;
/// Depthwise Separable Convolution block for MobileNetV1
struct DepthwiseSeparableConv<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Pointwise convolution (1x1)
    pointwise_conv: Conv2D<F>,
    /// Pointwise batch normalization
    pointwise_bn: BatchNorm<F>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> DepthwiseSeparableConv<F> {
    /// Create a new Depthwise Separable Convolution block
        let depthwise_bn = BatchNorm::new(input_channels, 1e-3, 0.01, &mut rng)?;
        // Create pointwise convolution (1x1)
        let pointwise_conv = Conv2D::new(
        // Pointwise batch normalization
        let pointwise_bn = BatchNorm::new(output_channels, 1e-3, 0.01, &mut rng)?;
            pointwise_conv,
            pointwise_bn,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for DepthwiseSeparableConv<F> {
        // Depthwise convolution
        let mut x = self.depthwise_conv.forward(input)?;
        // Pointwise convolution
        x = self.pointwise_conv.forward(&x)?;
        x = self.pointwise_bn.forward(&x)?;
        // Update depthwise convolution
        // Update pointwise convolution
        self.pointwise_conv.update(learning_rate)?;
        self.pointwise_bn.update(learning_rate)?;
/// MobileNet implementation
pub struct MobileNet<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Model configuration
    config: MobileNetConfig,
    /// Initial convolution
    stem_conv: Conv2D<F>,
    /// Initial batch normalization
    stem_bn: BatchNorm<F>,
    /// Blocks (depthwise separable or inverted residual)
    blocks: Vec<Box<dyn Layer<F>>>,
    /// Classifier
    classifier: Dense<F>,
    /// Dropout
    dropout: Dropout<F>,
    /// Activation function for the stem
    stem_activation: Box<dyn Fn(F) -> F + Send + Sync>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> MobileNet<F> {
    /// Create a new MobileNet model
    pub fn new(config: MobileNetConfig) -> Result<Self> {
        let input_channels = config.input_channels;
        let num_classes = config.num_classes;
        let width_multiplier = config.width_multiplier;
        // Create the stem block (first convolution)
        let first_block = &_config.blocks[0];
        let stem_channels =
            (first_block.output_channels as f64 * width_multiplier).round() as usize;
        let kernel_size_tuple = (first_block.kernel_size, first_block.kernel_size);
        let stride_tuple = (first_block.stride, first_block.stride);
        let stem_conv = Conv2D::new(
            stem_channels,
        let stem_bn = BatchNorm::new(stem_channels, 1e-3, 0.01, &mut rng)?;
        let stem_activation = get_activation(&first_block.activation);
        // Create blocks based on the MobileNet version
        let mut blocks: Vec<Box<dyn Layer<F>>> = Vec::new();
        match config.version {
            MobileNetVersion::V1 => {
                // Depthwise separable blocks for MobileNetV1
                for i in 1.._config.blocks.len() {
                    let block =
                        DepthwiseSeparableConv::new(_config.blocks[i].clone(), width_multiplier)?;
                    blocks.push(Box::new(block));
            _ => {
                // Inverted residual blocks for MobileNetV2/V3
                        InvertedResidualBlock::new(_config.blocks[i].clone(), width_multiplier)?;
        // Get the output channels of the last block
        let last_channels = if config.version == MobileNetVersion::V1 {
            // For MobileNetV1, the last block is a depthwise separable conv
            let scaled =
                (_config.blocks.last().unwrap().output_channels as f64 * width_multiplier).round();
            scaled as usize
            // For MobileNetV2/V3, we use 1280 as the number of output channels
            // except for MobileNetV2 which uses 1001 for compatibility with original paper
            match config.version {
                MobileNetVersion::V2 => 1001_ => 1280,
        // Classifier
        let classifier = Dense::new(last_channels, num_classes, None, &mut rng)?;
        // Dropout
        let dropout = Dropout::new(_config.dropout_rate, &mut rng)?;
            stem_conv,
            stem_bn,
            classifier,
            dropout,
            stem_activation,
    /// Create MobileNetV1 model
    pub fn mobilenet_v1(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let _config = MobileNetConfig::mobilenet_v1(input_channels, num_classes);
        Self::new(_config)
    /// Create MobileNetV2 model
    pub fn mobilenet_v2(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let _config = MobileNetConfig::mobilenet_v2(input_channels, num_classes);
    /// Create MobileNetV3-Small model
    pub fn mobilenet_v3_small(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let _config = MobileNetConfig::mobilenet_v3_small(input_channels, num_classes);
    /// Create MobileNetV3-Large model
    pub fn mobilenet_v3_large(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let _config = MobileNetConfig::mobilenet_v3_large(input_channels, num_classes);
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for MobileNet<F> {
        // Check input shape
        if shape.len() != 4 || shape[1] != self._config.input_channels {
                "Expected input shape [batch_size, {}, height, width], got {:?}",
                self.config.input_channels, shape
        // Apply stem
        let mut x = self.stem_conv.forward(input)?;
        x = self.stem_bn.forward(&x)?;
        x = x.mapv(|v| (self.stem_activation)(v)); // Apply activation
        // Apply blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        let channels = x.shape()[1];
        let height = x.shape()[2];
        let width = x.shape()[3];
        let mut pooled = Array::zeros(IxDyn(&[batch_size, channels]));
                        sum = sum + x[[b, c, h, w]];
                pooled[[b, c]] = sum / F::from(height * width).unwrap();
        let pooled = self.dropout.forward(&pooled)?;
        let logits = self.classifier.forward(&pooled)?;
        Ok(logits)
        // Update stem
        self.stem_conv.update(learning_rate)?;
        self.stem_bn.update(learning_rate)?;
        // Update blocks
        for block in &mut self.blocks {
            block.update(learning_rate)?;
        // Update classifier
        self.classifier.update(learning_rate)?;
