//! ResNet implementation
//!
//! ResNet (Residual Network) is a popular CNN architecture that introduced
//! skip connections to allow for training very deep networks.
//! Reference: "Deep Residual Learning for Image Recognition", He et al. (2015)
//! https://arxiv.org/abs/1512.03385

use crate::error::{NeuralError, Result};
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, PaddingMode};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::SeedableRng;
use std::fmt::Debug;
/// Type alias for average pooling function
type AvgPoolFn<F> = Box<dyn Fn(&Array<F, IxDyn>) -> Result<Array<F, IxDyn>> + Send + Sync>;
/// ResNet block configuration
#[derive(Debug, Clone)]
pub enum ResNetBlock {
    /// Basic block (2 conv layers)
    Basic,
    /// Bottleneck block (3 conv layers with bottleneck)
    Bottleneck,
}
/// Configuration for a ResNet layer
pub struct ResNetLayer {
    /// Number of blocks in this layer
    pub blocks: usize,
    /// Number of output channels
    pub channels: usize,
    /// Stride for the first block (usually 1 or 2)
    pub stride: usize,
/// Configuration for a ResNet model
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
impl ResNetConfig {
    /// Create a ResNet-18 configuration
    pub fn resnet18(_input_channels: usize, numclasses: usize) -> Self {
        Self {
            block: ResNetBlock::Basic,
            layers: vec![
                ResNetLayer {
                    blocks: 2, channels: 64,
                    stride: 1,
                },
                    _channels: 128,
                    stride: 2, channels: 256,
                    _channels: 512,
            ],
            input_channels,
            num_classes,
            dropout_rate: 0.0,
        }
    }
    /// Create a ResNet-34 configuration
    pub fn resnet34(_input_channels: usize, numclasses: usize) -> Self {
                    blocks: 3,
                    blocks: 4,
                    blocks: 6,
    /// Create a ResNet-50 configuration
    pub fn resnet50(_input_channels: usize, numclasses: usize) -> Self {
            block: ResNetBlock::Bottleneck,
    /// Create a ResNet-101 configuration
    pub fn resnet101(_input_channels: usize, numclasses: usize) -> Self {
                    blocks: 23,
    /// Create a ResNet-152 configuration
    pub fn resnet152(_input_channels: usize, numclasses: usize) -> Self {
                    blocks: 8,
                    blocks: 36,
    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
/// Basic block for ResNet
struct BasicBlock<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// First convolutional layer
    conv1: Conv2D<F>,
    /// First batch normalization layer
    bn1: BatchNorm<F>,
    /// Second convolutional layer
    conv2: Conv2D<F>,
    /// Second batch normalization layer
    bn2: BatchNorm<F>,
    /// Skip connection (optional)
    downsample: Option<(Conv2D<F>, BatchNorm<F>)>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> BasicBlock<F> {
    /// Create a new basic block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: bool,
    ) -> Result<Self> {
        // First convolutional layer
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let kernel_size = (3, 3);
        let stride_tuple = if stride == 1 {
            (1, 1)
        } else {
            (stride, stride)
        };
        let conv1 = Conv2D::new(
            in_channels,
            out_channels,
            kernel_size,
            stride_tuple,
            PaddingMode::Same,
            &mut rng,
        )?;
        // First batch normalization
        let bn1 = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng)?;
        // Second convolutional layer
        let conv2 = Conv2D::new(
            (3, 3),
            (1, 1),
        // Second batch normalization
        let bn2 = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng)?;
        // Downsample if needed
        let downsample = if downsample {
            let mut ds_rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
            let ds_conv = Conv2D::new(
                in_channels,
                out_channels,
                (1, 1),
                stride_tuple,
                PaddingMode::Valid,
                &mut ds_rng,
            )?;
            let ds_bn = BatchNorm::new(out_channels, 1e-5, 0.1, &mut ds_rng)?;
            Some((ds_conv, ds_bn))
            None
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BasicBlock<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Save identity for skip connection
        let identity = input.clone();
        // First convolutional block
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = x.mapv(|v| v.max(F::zero())); // ReLU
        // Second convolutional block
        x = self.conv2.forward(&x)?;
        x = self.bn2.forward(&x)?;
        // Apply downsample if needed
        if let Some((ref conv, ref bn)) = self.downsample {
            let identity = conv.forward(input)?;
            let identity = bn.forward(&identity)?;
            // Add skip connection
            for i in 0..x.len() {
                x[i] = x[i] + identity[i];
            }
            // Add skip connection (identity)
            if x.shape() == identity.shape() {
                for i in 0..x.len() {
                    x[i] = x[i] + identity[i];
                }
            } else {
                return Err(NeuralError::InferenceError(format!(
                    "Shape mismatch in skip connection: {:?} vs {:?}",
                    x.shape(),
                    identity.shape()
                )));
        // Final activation
        Ok(x)
    fn backward(
        &mut self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // In a real implementation, we'd compute gradients for all parameters
        // For now, just return the input gradient
        Ok(grad_output.clone())
    fn update(&mut self, learningrate: F) -> Result<()> {
        // Update weights in all layers
        self.conv1.update(learning_rate)?;
        self.bn1.update(learning_rate)?;
        self.conv2.update(learning_rate)?;
        self.bn2.update(learning_rate)?;
        if let Some((ref mut conv, ref mut bn)) = self.downsample {
            conv.update(learning_rate)?;
            bn.update(learning_rate)?;
        Ok(())
    fn as_any(&self) -> &dyn std::any::Any {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
/// Bottleneck block for ResNet
struct BottleneckBlock<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// First convolutional layer (1x1)
    /// Second convolutional layer (3x3)
    /// Third convolutional layer (1x1)
    conv3: Conv2D<F>,
    /// Third batch normalization layer
    bn3: BatchNorm<F>,
    /// Expansion factor
    #[allow(dead_code)]
    expansion: usize,
impl<F: Float + Debug + ScalarOperand + Send + Sync> BottleneckBlock<F> {
    /// Create a new bottleneck block
        // Expansion factor is 4 for bottleneck blocks
        let expansion = 4;
        let bottleneck_channels = out_channels / expansion;
        // First convolutional layer (1x1 reduce dimensions)
            bottleneck_channels,
            PaddingMode::Valid,
        let bn1 = BatchNorm::new(bottleneck_channels, 1e-5, 0.1, &mut rng)?;
        // Second convolutional layer (3x3)
        let bn2 = BatchNorm::new(bottleneck_channels, 1e-5, 0.1, &mut rng)?;
        // Third convolutional layer (1x1 increase dimensions)
        let conv3 = Conv2D::new(
        // Third batch normalization
        let bn3 = BatchNorm::new(out_channels, 1e-5, 0.1, &mut rng)?;
            conv3,
            bn3,
            expansion,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BottleneckBlock<F> {
        // Third convolutional block
        x = self.conv3.forward(&x)?;
        x = self.bn3.forward(&x)?;
        self.conv3.update(learning_rate)?;
        self.bn3.update(learning_rate)?;
/// ResNet implementation
pub struct ResNet<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Initial convolutional layer
    /// Initial batch normalization
    /// ResNet layers
    layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    /// Global average pooling
    avg_pool: AvgPoolFn<F>,
    /// Fully connected layer
    fc: Dense<F>,
    /// Dropout layer
    dropout: Option<Dropout<F>>,
    /// Model configuration
    config: ResNetConfig,
impl<F: Float + Debug + ScalarOperand + Send + Sync> ResNet<F> {
    /// Create a new ResNet model
    pub fn new(config: ResNetConfig) -> Result<Self> {
        // Initial convolutional layer
            config.input_channels,
            64,
            (7, 7),
            (2, 2),
        // Initial batch normalization
        let bn1 = BatchNorm::new(64, 1e-5, 0.1, &mut rng)?;
        // Create ResNet layers
        let mut layers: Vec<Box<dyn Layer<F> + Send + Sync>> = Vec::new();
        let mut in_channels = 64;
        // Create each layer group
        for layer_config in &_config.layers {
            let mut layer_blocks: Vec<Box<dyn Layer<F> + Send + Sync>> = Vec::new();
            let out_channels = match config.block {
                ResNetBlock::Basic => layer_config.channels,
                ResNetBlock::Bottleneck => layer_config.channels * 4,
            };
            // First block may have stride > 1 and needs downsample
            match config.block {
                ResNetBlock::Basic => {
                    let block = BasicBlock::new(
                        in_channels,
                        out_channels,
                        layer_config.stride,
                        in_channels != out_channels || layer_config.stride != 1,
                    )?;
                    layer_blocks.push(Box::new(block));
                ResNetBlock::Bottleneck => {
                    let block = BottleneckBlock::new(
            // Remaining blocks have stride 1 and no downsample
            for _ in 1..layer_config.blocks {
                match config.block {
                    ResNetBlock::Basic => {
                        let block = BasicBlock::new(out_channels, out_channels, 1, false)?;
                        layer_blocks.push(Box::new(block));
                    }
                    ResNetBlock::Bottleneck => {
                        let block = BottleneckBlock::new(out_channels, out_channels, 1, false)?;
            // Add this layer group
            let layer = ResNetLayerGroup {
                blocks: layer_blocks,
            layers.push(Box::new(layer));
            // Update input channels for next layer
            in_channels = out_channels;
        // Global average pooling function
        let avg_pool = Box::new(|x: &Array<F, IxDyn>| {
            // Assuming input shape is [batch_size, channels, height, width]
            let shape = x.shape();
            if shape.len() != 4 {
                    "Expected 4D input for average pooling, got shape {:?}",
                    shape
            let batch_size = shape[0];
            let channels = shape[1];
            let height = shape[2];
            let width = shape[3];
            // Initialize output
            let mut output = Array::zeros(IxDyn(&[batch_size, channels]));
            // Compute average for each batch and channel
            for b in 0..batch_size {
                for c in 0..channels {
                    let mut sum = F::zero();
                    let count = F::from(height * width).unwrap();
                    for h in 0..height {
                        for w in 0..width {
                            sum = sum + x[[b, c, h, w]];
                        }
                    output[[b, c]] = sum / count;
            Ok(output)
        });
        // Last layer is fully connected
        let fc_in_features = match config.block {
            ResNetBlock::Basic => config.layers.last().unwrap().channels,
            ResNetBlock::Bottleneck => config.layers.last().unwrap().channels * 4,
        let fc = Dense::new(
            fc_in_features,
            config.num_classes,
            None, // No activation for final layer (usually used with softmax loss)
        // Add dropout if specified
        let dropout = if config.dropout_rate > 0.0 {
            Some(Dropout::new(config.dropout_rate, &mut rng)?)
            layers,
            avg_pool,
            fc,
            dropout,
            config,
    /// Create a ResNet-18 model
    pub fn resnet18(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let config = ResNetConfig::resnet18(input_channels, num_classes);
        Self::new(config)
    /// Create a ResNet-34 model
    pub fn resnet34(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let config = ResNetConfig::resnet34(input_channels, num_classes);
    /// Create a ResNet-50 model
    pub fn resnet50(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let config = ResNetConfig::resnet50(input_channels, num_classes);
    /// Create a ResNet-101 model
    pub fn resnet101(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let config = ResNetConfig::resnet101(input_channels, num_classes);
    /// Create a ResNet-152 model
    pub fn resnet152(_input_channels: usize, numclasses: usize) -> Result<Self> {
        let config = ResNetConfig::resnet152(input_channels, num_classes);
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for ResNet<F> {
        // MaxPool 3x3 with stride 2 (typically used in ResNet)
        // In a real implementation, we'd have a maxpool layer here
        // Forward through all layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
        // Global average pooling
        x = (self.avg_pool)(&x)?;
        // Apply dropout if present
        if let Some(ref dropout) = self.dropout {
            x = dropout.forward(&x)?;
        // Final fully connected layer
        x = self.fc.forward(&x)?;
        for layer in &mut self.layers {
            layer.update(learningrate)?;
        self.fc.update(learning_rate)?;
/// Group of ResNet blocks for a single layer
struct ResNetLayerGroup<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Blocks in this layer
    blocks: Vec<Box<dyn Layer<F> + Send + Sync>>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for ResNetLayerGroup<F> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        for block in &mut self.blocks {
            block.update(learning_rate)?;
