//! U-Net implementation
//!
//! U-Net is a convolutional neural network architecture designed for semantic segmentation.
//! It features an encoder (contracting) path that captures context and a symmetric decoder
//! (expanding) path that enables precise localization via skip connections.
//! Reference: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
//! Ronneberger et al. (2015)
//! <https://arxiv.org/abs/1505.04597>

use crate::error::{NeuralError, Result};
use crate::layers::conv::PaddingMode;
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::SeedableRng;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a U-Net model
#[derive(Debug, Clone)]
pub struct UNetConfig {
    /// Number of input channels (e.g. 1 for grayscale, 3 for RGB)
    pub input_channels: usize,
    /// Number of output channels / classes for the segmentation map
    pub output_channels: usize,
    /// Number of encoder / decoder stages (depth of the U)
    pub depth: usize,
    /// Number of base filters — doubled at each encoder stage
    pub base_filters: usize,
    /// Kernel size for all convolutions (square)
    pub kernel_size: usize,
    /// Whether to use batch normalization after each convolution
    pub use_batch_norm: bool,
    /// Dropout rate applied between encoder/decoder blocks (0 to disable)
    pub dropout_rate: f64,
    /// Channel divisor for lightweight / test models (default 1)
    pub channel_divisor: usize,
    /// Whether to use bilinear interpolation for upsampling (true) or transposed conv (false)
    pub bilinear_upsample: bool,
}

impl UNetConfig {
    /// Standard U-Net for biomedical image segmentation
    pub fn standard(input_channels: usize, output_channels: usize) -> Self {
        Self {
            input_channels,
            output_channels,
            depth: 4,
            base_filters: 64,
            kernel_size: 3,
            use_batch_norm: true,
            dropout_rate: 0.0,
            channel_divisor: 1,
            bilinear_upsample: true,
        }
    }

    /// Tiny U-Net for testing
    pub fn tiny(input_channels: usize, output_channels: usize) -> Self {
        Self {
            input_channels,
            output_channels,
            depth: 2,
            base_filters: 16,
            kernel_size: 3,
            use_batch_norm: false,
            dropout_rate: 0.0,
            channel_divisor: 1,
            bilinear_upsample: true,
        }
    }

    /// Create a custom U-Net configuration
    pub fn custom(
        input_channels: usize,
        output_channels: usize,
        depth: usize,
        base_filters: usize,
    ) -> Self {
        Self {
            input_channels,
            output_channels,
            depth,
            base_filters,
            kernel_size: 3,
            use_batch_norm: true,
            dropout_rate: 0.0,
            channel_divisor: 1,
            bilinear_upsample: true,
        }
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Set batch normalization
    pub fn with_batch_norm(mut self, use_bn: bool) -> Self {
        self.use_batch_norm = use_bn;
        self
    }

    /// Set channel divisor for lightweight models
    pub fn with_channel_divisor(mut self, divisor: usize) -> Self {
        self.channel_divisor = divisor.max(1);
        self
    }

    /// Set kernel size
    pub fn with_kernel_size(mut self, ks: usize) -> Self {
        self.kernel_size = ks;
        self
    }

    /// Compute the effective number of filters at a given encoder level
    fn filters_at(&self, level: usize) -> usize {
        let raw = self.base_filters * (1 << level);
        (raw / self.channel_divisor).max(1)
    }
}

// ---------------------------------------------------------------------------
// ConvBlock — two convolutions optionally followed by batch-norm + ReLU
// ---------------------------------------------------------------------------

/// Double convolution block: Conv -> [BN] -> ReLU -> Conv -> [BN] -> ReLU
struct UNetConvBlock<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    conv1: Conv2D<F>,
    bn1: Option<BatchNorm<F>>,
    conv2: Conv2D<F>,
    bn2: Option<BatchNorm<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> UNetConvBlock<F> {
    fn new(in_ch: usize, out_ch: usize, kernel_size: usize, use_bn: bool) -> Result<Self> {
        let conv1 = Conv2D::new(in_ch, out_ch, (kernel_size, kernel_size), (1, 1), None)?
            .with_padding(PaddingMode::Same);
        let conv2 = Conv2D::new(out_ch, out_ch, (kernel_size, kernel_size), (1, 1), None)?
            .with_padding(PaddingMode::Same);

        let bn1 = if use_bn {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([60; 32]);
            Some(BatchNorm::new(out_ch, 0.1, 1e-5, &mut rng)?)
        } else {
            None
        };
        let bn2 = if use_bn {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([61; 32]);
            Some(BatchNorm::new(out_ch, 0.1, 1e-5, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
        })
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // First conv + optional BN + ReLU
        let mut x = self.conv1.forward(input)?;
        if let Some(ref bn) = self.bn1 {
            x = bn.forward(&x)?;
        }
        x = x.mapv(|v| v.max(F::zero()));

        // Second conv + optional BN + ReLU
        x = self.conv2.forward(&x)?;
        if let Some(ref bn) = self.bn2 {
            x = bn.forward(&x)?;
        }
        x = x.mapv(|v| v.max(F::zero()));

        Ok(x)
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.conv1.update(lr)?;
        self.conv2.update(lr)?;
        if let Some(ref mut bn) = self.bn1 {
            bn.update(lr)?;
        }
        if let Some(ref mut bn) = self.bn2 {
            bn.update(lr)?;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = self.conv1.params();
        p.extend(self.conv2.params());
        if let Some(ref bn) = self.bn1 {
            p.extend(bn.params());
        }
        if let Some(ref bn) = self.bn2 {
            p.extend(bn.params());
        }
        p
    }

    fn parameter_count(&self) -> usize {
        let mut c = self.conv1.parameter_count() + self.conv2.parameter_count();
        if let Some(ref bn) = self.bn1 {
            c += bn.parameter_count();
        }
        if let Some(ref bn) = self.bn2 {
            c += bn.parameter_count();
        }
        c
    }
}

// ---------------------------------------------------------------------------
// Encoder stage — conv block followed by 2x2 max pool
// ---------------------------------------------------------------------------

struct UNetEncoderStage<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    conv_block: UNetConvBlock<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> UNetEncoderStage<F> {
    fn new(in_ch: usize, out_ch: usize, kernel_size: usize, use_bn: bool) -> Result<Self> {
        Ok(Self {
            conv_block: UNetConvBlock::new(in_ch, out_ch, kernel_size, use_bn)?,
        })
    }

    /// Returns (pooled_output, skip_connection)
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let conv_out = self.conv_block.forward(input)?;
        let pooled = max_pool_2x2(&conv_out)?;
        Ok((pooled, conv_out))
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.conv_block.update(lr)
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        self.conv_block.params()
    }

    fn parameter_count(&self) -> usize {
        self.conv_block.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Decoder stage — upsample + concat skip + conv block
// ---------------------------------------------------------------------------

struct UNetDecoderStage<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    /// 1x1 convolution to reduce channels after concatenation
    reduce_conv: Conv2D<F>,
    conv_block: UNetConvBlock<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> UNetDecoderStage<F> {
    fn new(
        in_ch: usize,
        skip_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        use_bn: bool,
    ) -> Result<Self> {
        // After upsampling in_ch channels are concatenated with skip_ch channels
        let cat_ch = in_ch + skip_ch;
        let reduce_conv =
            Conv2D::new(cat_ch, out_ch, (1, 1), (1, 1), None)?.with_padding(PaddingMode::Same);
        let conv_block = UNetConvBlock::new(out_ch, out_ch, kernel_size, use_bn)?;
        Ok(Self {
            reduce_conv,
            conv_block,
        })
    }

    /// Forward: upsample `input`, concat with `skip`, apply conv block
    fn forward(&self, input: &Array<F, IxDyn>, skip: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Bilinear-style nearest-neighbor 2x upsample
        let upsampled = upsample_2x(input)?;

        // Crop both tensors to their minimum spatial dimensions.
        // After 2x upsampling an odd-sized feature map, the upsampled spatial
        // size may differ from the encoder skip connection by ±1.  We center-crop
        // the larger of the two so they share identical H×W before concatenation.
        let (upsampled, skip_cropped) = crop_pair_to_min(&upsampled, skip)?;

        // Concatenate along channel axis
        let cat = concat_channels(&upsampled, &skip_cropped)?;

        // 1x1 reduce + conv block
        let reduced = self.reduce_conv.forward(&cat)?;
        self.conv_block.forward(&reduced)
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.reduce_conv.update(lr)?;
        self.conv_block.update(lr)
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = self.reduce_conv.params();
        p.extend(self.conv_block.params());
        p
    }

    fn parameter_count(&self) -> usize {
        self.reduce_conv.parameter_count() + self.conv_block.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Helper ops
// ---------------------------------------------------------------------------

/// 2x2 max pooling with stride 2
fn max_pool_2x2<F: Float + Debug>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "Expected 4D input for max pooling, got shape {:?}",
            shape
        )));
    }
    let (bs, ch, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (oh, ow) = (h / 2, w / 2);

    let mut output = Array::from_elem(IxDyn(&[bs, ch, oh, ow]), F::neg_infinity());
    for b in 0..bs {
        for c in 0..ch {
            for i in 0..oh {
                for j in 0..ow {
                    let mut mx = F::neg_infinity();
                    for di in 0..2 {
                        for dj in 0..2 {
                            let hi = i * 2 + di;
                            let wi = j * 2 + dj;
                            if hi < h && wi < w {
                                let v = input[[b, c, hi, wi]];
                                if v > mx {
                                    mx = v;
                                }
                            }
                        }
                    }
                    output[[b, c, i, j]] = mx;
                }
            }
        }
    }
    Ok(output)
}

/// Nearest-neighbor 2x upsampling
fn upsample_2x<F: Float + Debug>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "Expected 4D input for upsample, got shape {:?}",
            shape
        )));
    }
    let (bs, ch, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (oh, ow) = (h * 2, w * 2);
    let mut output = Array::zeros(IxDyn(&[bs, ch, oh, ow]));
    for b in 0..bs {
        for c in 0..ch {
            for i in 0..h {
                for j in 0..w {
                    let v = input[[b, c, i, j]];
                    output[[b, c, i * 2, j * 2]] = v;
                    output[[b, c, i * 2 + 1, j * 2]] = v;
                    output[[b, c, i * 2, j * 2 + 1]] = v;
                    output[[b, c, i * 2 + 1, j * 2 + 1]] = v;
                }
            }
        }
    }
    Ok(output)
}

/// Crop both 4-D tensors to their minimum spatial dimensions (center crop).
///
/// When the UNet encoder downsamples an odd-sized feature map the decoder's
/// 2x upsampled output and the encoder skip connection can differ by 1 pixel
/// in height and/or width.  This helper crops whichever is larger so that both
/// share identical H×W, enabling channel-wise concatenation without error.
fn crop_pair_to_min<F: Float + Debug>(
    a: &Array<F, IxDyn>,
    b: &Array<F, IxDyn>,
) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
    let sa = a.shape();
    let sb = b.shape();
    if sa.len() != 4 || sb.len() != 4 {
        return Err(NeuralError::InferenceError(
            "crop_pair_to_min requires 4D tensors".to_string(),
        ));
    }
    let min_h = sa[2].min(sb[2]);
    let min_w = sa[3].min(sb[3]);

    let crop_a = center_crop_4d(a, min_h, min_w)?;
    let crop_b = center_crop_4d(b, min_h, min_w)?;
    Ok((crop_a, crop_b))
}

/// Center-crop a 4D tensor `[B, C, H, W]` to `[B, C, target_h, target_w]`.
fn center_crop_4d<F: Float + Debug>(
    src: &Array<F, IxDyn>,
    target_h: usize,
    target_w: usize,
) -> Result<Array<F, IxDyn>> {
    let ss = src.shape();
    let (sh, sw) = (ss[2], ss[3]);
    if sh == target_h && sw == target_w {
        return Ok(src.clone());
    }
    if sh < target_h || sw < target_w {
        return Err(NeuralError::InferenceError(format!(
            "center_crop_4d: source [{}, {}] is smaller than target [{}, {}]",
            sh, sw, target_h, target_w
        )));
    }
    let dh = (sh - target_h) / 2;
    let dw = (sw - target_w) / 2;
    let bs = ss[0];
    let ch = ss[1];
    let mut output = Array::zeros(IxDyn(&[bs, ch, target_h, target_w]));
    for b in 0..bs {
        for c in 0..ch {
            for i in 0..target_h {
                for j in 0..target_w {
                    output[[b, c, i, j]] = src[[b, c, i + dh, j + dw]];
                }
            }
        }
    }
    Ok(output)
}

/// Crop `source` spatially to match `target` dimensions (center crop)
fn crop_to_match<F: Float + Debug>(
    source: &Array<F, IxDyn>,
    target: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    let ss = source.shape();
    let ts = target.shape();
    if ss.len() != 4 || ts.len() != 4 {
        return Err(NeuralError::InferenceError(
            "crop_to_match requires 4D tensors".to_string(),
        ));
    }
    let (sh, sw) = (ss[2], ss[3]);
    let (th, tw) = (ts[2], ts[3]);

    if sh == th && sw == tw {
        return Ok(source.clone());
    }

    let dh = (sh.saturating_sub(th)) / 2;
    let dw = (sw.saturating_sub(tw)) / 2;
    let bs = ss[0];
    let ch = ss[1];
    let out_h = th.min(sh);
    let out_w = tw.min(sw);

    let mut output = Array::zeros(IxDyn(&[bs, ch, out_h, out_w]));
    for b in 0..bs {
        for c in 0..ch {
            for i in 0..out_h {
                for j in 0..out_w {
                    output[[b, c, i, j]] = source[[b, c, i + dh, j + dw]];
                }
            }
        }
    }
    Ok(output)
}

/// Concatenate two 4D tensors along the channel (dim=1) axis
fn concat_channels<F: Float + Debug>(
    a: &Array<F, IxDyn>,
    b: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    let sa = a.shape();
    let sb = b.shape();
    if sa.len() != 4 || sb.len() != 4 {
        return Err(NeuralError::InferenceError(
            "concat_channels requires 4D tensors".to_string(),
        ));
    }
    if sa[0] != sb[0] || sa[2] != sb[2] || sa[3] != sb[3] {
        return Err(NeuralError::InferenceError(format!(
            "Spatial dims must match for concat: {:?} vs {:?}",
            sa, sb
        )));
    }

    let bs = sa[0];
    let ca = sa[1];
    let cb = sb[1];
    let h = sa[2];
    let w = sa[3];

    let mut output = Array::zeros(IxDyn(&[bs, ca + cb, h, w]));
    for batch in 0..bs {
        for c in 0..ca {
            for i in 0..h {
                for j in 0..w {
                    output[[batch, c, i, j]] = a[[batch, c, i, j]];
                }
            }
        }
        for c in 0..cb {
            for i in 0..h {
                for j in 0..w {
                    output[[batch, ca + c, i, j]] = b[[batch, c, i, j]];
                }
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// U-Net model
// ---------------------------------------------------------------------------

/// U-Net neural network for image segmentation
///
/// Consists of an encoder (contracting) path that progressively down-samples
/// the input, a bottleneck, and a decoder (expanding) path that up-samples
/// back to the original resolution. Skip connections pass feature maps from
/// corresponding encoder stages to decoder stages, preserving fine-grained
/// spatial information.
///
/// # Examples
///
/// ```no_run
/// use scirs2_neural::models::architectures::unet::{UNet, UNetConfig};
///
/// // Standard U-Net for binary segmentation of grayscale images
/// let model: UNet<f64> = UNet::new(UNetConfig::standard(1, 2))
///     .expect("Failed to create U-Net");
/// ```
pub struct UNet<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    config: UNetConfig,
    /// Encoder stages (each produces a skip connection)
    encoder_stages: Vec<UNetEncoderStage<F>>,
    /// Bottleneck conv block at the bottom of the U
    bottleneck: UNetConvBlock<F>,
    /// Decoder stages (in reverse order of the encoder)
    decoder_stages: Vec<UNetDecoderStage<F>>,
    /// Final 1x1 convolution to produce the output segmentation map
    final_conv: Conv2D<F>,
    /// Optional dropout
    dropout: Option<Dropout<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> UNet<F> {
    /// Create a new U-Net model from configuration
    pub fn new(config: UNetConfig) -> Result<Self> {
        if config.depth == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "U-Net depth must be at least 1".to_string(),
            ));
        }

        // Build encoder stages
        let mut encoder_stages = Vec::with_capacity(config.depth);
        let mut in_ch = config.input_channels;
        for level in 0..config.depth {
            let out_ch = config.filters_at(level);
            encoder_stages.push(UNetEncoderStage::new(
                in_ch,
                out_ch,
                config.kernel_size,
                config.use_batch_norm,
            )?);
            in_ch = out_ch;
        }

        // Bottleneck
        let bottleneck_ch = config.filters_at(config.depth);
        let bottleneck = UNetConvBlock::new(
            in_ch,
            bottleneck_ch,
            config.kernel_size,
            config.use_batch_norm,
        )?;

        // Build decoder stages (mirror of encoder)
        let mut decoder_stages = Vec::with_capacity(config.depth);
        let mut dec_in_ch = bottleneck_ch;
        for level in (0..config.depth).rev() {
            let skip_ch = config.filters_at(level);
            let out_ch = skip_ch; // match encoder channel count
            decoder_stages.push(UNetDecoderStage::new(
                dec_in_ch,
                skip_ch,
                out_ch,
                config.kernel_size,
                config.use_batch_norm,
            )?);
            dec_in_ch = out_ch;
        }

        // Final 1x1 conv to map to output channels
        let final_conv = Conv2D::new(dec_in_ch, config.output_channels, (1, 1), (1, 1), None)?
            .with_padding(PaddingMode::Same);

        let dropout = if config.dropout_rate > 0.0 {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([62; 32]);
            Some(Dropout::new(config.dropout_rate, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            config,
            encoder_stages,
            bottleneck,
            decoder_stages,
            final_conv,
            dropout,
        })
    }

    /// Convenience constructor for the standard U-Net
    pub fn standard(input_channels: usize, output_channels: usize) -> Result<Self> {
        Self::new(UNetConfig::standard(input_channels, output_channels))
    }

    /// Convenience constructor for a tiny U-Net (testing)
    pub fn tiny(input_channels: usize, output_channels: usize) -> Result<Self> {
        Self::new(UNetConfig::tiny(input_channels, output_channels))
    }

    /// Get the model configuration
    pub fn config(&self) -> &UNetConfig {
        &self.config
    }

    /// Total trainable parameter count
    pub fn total_parameter_count(&self) -> usize {
        let enc: usize = self
            .encoder_stages
            .iter()
            .map(|s| s.parameter_count())
            .sum();
        let bneck = self.bottleneck.parameter_count();
        let dec: usize = self
            .decoder_stages
            .iter()
            .map(|s| s.parameter_count())
            .sum();
        let fc = self.final_conv.parameter_count();
        enc + bneck + dec + fc
    }

    /// Number of encoder/decoder stages
    pub fn depth(&self) -> usize {
        self.config.depth
    }

    /// Extract features at the bottleneck (before decoder)
    pub fn encode(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input [batch, channels, height, width], got {:?}",
                shape
            )));
        }

        let mut x = input.clone();
        for stage in &self.encoder_stages {
            let (pooled, _skip) = stage.forward(&x)?;
            x = pooled;
        }
        self.bottleneck.forward(&x)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F> for UNet<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input [batch, channels, height, width], got {:?}",
                shape
            )));
        }
        if shape[1] != self.config.input_channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected {} input channels, got {}",
                self.config.input_channels, shape[1]
            )));
        }

        // Encoder path — collect skip connections
        let mut skips = Vec::with_capacity(self.config.depth);
        let mut x = input.clone();
        for stage in &self.encoder_stages {
            let (pooled, skip) = stage.forward(&x)?;
            skips.push(skip);
            x = pooled;
        }

        // Bottleneck
        x = self.bottleneck.forward(&x)?;
        if let Some(ref drop) = self.dropout {
            x = drop.forward(&x)?;
        }

        // Decoder path — use skip connections in reverse order
        for (i, stage) in self.decoder_stages.iter().enumerate() {
            let skip_idx = self.config.depth - 1 - i;
            x = stage.forward(&x, &skips[skip_idx])?;
        }

        // Final 1x1 conv
        self.final_conv.forward(&x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Gradient passthrough for compatibility
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for stage in &mut self.encoder_stages {
            stage.update(learning_rate)?;
        }
        self.bottleneck.update(learning_rate)?;
        for stage in &mut self.decoder_stages {
            stage.update(learning_rate)?;
        }
        self.final_conv.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = Vec::new();
        for stage in &self.encoder_stages {
            p.extend(stage.params());
        }
        p.extend(self.bottleneck.params());
        for stage in &self.decoder_stages {
            p.extend(stage.params());
        }
        p.extend(self.final_conv.params());
        p
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "UNet"
    }

    fn layer_description(&self) -> String {
        format!(
            "UNet(depth={}, base_filters={}, in_ch={}, out_ch={}, bn={}, params={})",
            self.config.depth,
            self.config.base_filters,
            self.config.input_channels,
            self.config.output_channels,
            self.config.use_batch_norm,
            self.total_parameter_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unet_config_standard() {
        let cfg = UNetConfig::standard(3, 2);
        assert_eq!(cfg.input_channels, 3);
        assert_eq!(cfg.output_channels, 2);
        assert_eq!(cfg.depth, 4);
        assert_eq!(cfg.base_filters, 64);
        assert!(cfg.use_batch_norm);
    }

    #[test]
    fn test_unet_config_tiny() {
        let cfg = UNetConfig::tiny(1, 1);
        assert_eq!(cfg.depth, 2);
        assert_eq!(cfg.base_filters, 16);
        assert!(!cfg.use_batch_norm);
    }

    #[test]
    fn test_unet_config_custom() {
        let cfg = UNetConfig::custom(1, 3, 3, 32);
        assert_eq!(cfg.depth, 3);
        assert_eq!(cfg.base_filters, 32);
    }

    #[test]
    fn test_unet_config_filters_at() {
        let cfg = UNetConfig::standard(1, 1);
        assert_eq!(cfg.filters_at(0), 64);
        assert_eq!(cfg.filters_at(1), 128);
        assert_eq!(cfg.filters_at(2), 256);
        assert_eq!(cfg.filters_at(3), 512);
    }

    #[test]
    fn test_unet_config_filters_divisor() {
        let cfg = UNetConfig::standard(1, 1).with_channel_divisor(8);
        assert_eq!(cfg.filters_at(0), 8);
        assert_eq!(cfg.filters_at(1), 16);
        assert_eq!(cfg.filters_at(2), 32);
    }

    #[test]
    fn test_unet_config_builder() {
        let cfg = UNetConfig::standard(1, 1)
            .with_dropout(0.3)
            .with_batch_norm(false)
            .with_kernel_size(5)
            .with_channel_divisor(4);
        assert!((cfg.dropout_rate - 0.3).abs() < 1e-10);
        assert!(!cfg.use_batch_norm);
        assert_eq!(cfg.kernel_size, 5);
        assert_eq!(cfg.channel_divisor, 4);
    }

    #[test]
    fn test_unet_creation_tiny() {
        let model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");
        assert_eq!(model.depth(), 2);
        assert!(model.total_parameter_count() > 0);
    }

    #[test]
    fn test_unet_creation_standard() {
        let model: UNet<f64> = UNet::new(UNetConfig::standard(1, 2).with_channel_divisor(16))
            .expect("Failed to create U-Net");
        assert_eq!(model.depth(), 4);
        assert!(model.total_parameter_count() > 0);
    }

    #[test]
    fn test_unet_forward_tiny() {
        let model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");

        // Input: [1, 1, 32, 32]
        let input = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        let output = model.forward(&input).expect("Forward pass failed");

        // Output should preserve spatial dims with output_channels
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 2); // output_channels
        assert_eq!(output.shape()[2], 32);
        assert_eq!(output.shape()[3], 32);
    }

    #[test]
    fn test_unet_forward_batch() {
        let model: UNet<f64> = UNet::tiny(1, 3).expect("Failed to create tiny U-Net");

        let input = Array::zeros(IxDyn(&[2, 1, 16, 16]));
        let output = model.forward(&input).expect("Forward pass failed");

        assert_eq!(output.shape(), &[2, 3, 16, 16]);
    }

    #[test]
    fn test_unet_forward_custom_depth() {
        let cfg = UNetConfig::custom(1, 1, 3, 8);
        let model: UNet<f64> = UNet::new(cfg).expect("Failed to create custom U-Net");

        let input = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        let output = model.forward(&input).expect("Forward pass failed");
        assert_eq!(output.shape(), &[1, 1, 32, 32]);
    }

    #[test]
    fn test_unet_encode() {
        let model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");

        let input = Array::zeros(IxDyn(&[1, 1, 32, 32]));
        let bottleneck = model.encode(&input).expect("Encode failed");

        // After depth=2 stages of 2x pooling: 32->16->8
        // Bottleneck channels = base_filters * 2^depth = 16 * 4 = 64
        assert_eq!(bottleneck.shape()[0], 1);
        assert_eq!(bottleneck.shape()[2], 8);
        assert_eq!(bottleneck.shape()[3], 8);
    }

    #[test]
    fn test_unet_invalid_input_shape() {
        let model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");

        // 3D input
        let input_3d = Array::zeros(IxDyn(&[1, 1, 32]));
        assert!(model.forward(&input_3d).is_err());

        // Wrong channels
        let input_wrong_ch = Array::zeros(IxDyn(&[1, 3, 32, 32]));
        assert!(model.forward(&input_wrong_ch).is_err());
    }

    #[test]
    fn test_unet_invalid_depth() {
        let cfg = UNetConfig {
            input_channels: 1,
            output_channels: 1,
            depth: 0,
            base_filters: 16,
            kernel_size: 3,
            use_batch_norm: false,
            dropout_rate: 0.0,
            channel_divisor: 1,
            bilinear_upsample: true,
        };
        assert!(UNet::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_unet_layer_trait() {
        let model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");
        assert_eq!(model.layer_type(), "UNet");
        assert!(model.parameter_count() > 0);
        let desc = model.layer_description();
        assert!(desc.contains("UNet"));
        assert!(desc.contains("depth=2"));
    }

    #[test]
    fn test_unet_update() {
        let mut model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");
        model.update(0.001).expect("Update should not fail");
    }

    #[test]
    fn test_unet_params() {
        let model: UNet<f64> = UNet::tiny(1, 2).expect("Failed to create tiny U-Net");
        let p = model.params();
        assert!(!p.is_empty());
    }

    #[test]
    fn test_unet_f32() {
        let model: UNet<f32> = UNet::tiny(1, 1).expect("Failed to create f32 U-Net");
        let input = Array::zeros(IxDyn(&[1, 1, 16, 16]));
        let output = model.forward(&input).expect("f32 forward failed");
        assert_eq!(output.shape(), &[1, 1, 16, 16]);
    }

    #[test]
    fn test_unet_with_dropout() {
        let cfg = UNetConfig::tiny(1, 1).with_dropout(0.5);
        let model: UNet<f64> = UNet::new(cfg).expect("Failed to create U-Net with dropout");
        assert!(model.dropout.is_some());
        let input = Array::zeros(IxDyn(&[1, 1, 16, 16]));
        let _output = model.forward(&input).expect("Forward with dropout failed");
    }

    #[test]
    fn test_unet_with_bn() {
        let cfg = UNetConfig::tiny(1, 1).with_batch_norm(true);
        let model: UNet<f64> = UNet::new(cfg).expect("Failed to create U-Net with BN");
        let input = Array::zeros(IxDyn(&[2, 1, 16, 16])); // batch > 1 for BN
        let output = model.forward(&input).expect("Forward with BN failed");
        assert_eq!(output.shape(), &[2, 1, 16, 16]);
    }

    // Helper tests for internal ops
    #[test]
    fn test_max_pool_2x2() {
        let mut input = Array::zeros(IxDyn(&[1, 1, 4, 4]));
        input[[0, 0, 0, 0]] = 1.0_f64;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;

        let output = max_pool_2x2(&input).expect("Pool failed");
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert!((output[[0, 0, 0, 0]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_upsample_2x() {
        let mut input = Array::zeros(IxDyn(&[1, 1, 2, 2]));
        input[[0, 0, 0, 0]] = 1.0_f64;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;

        let output = upsample_2x(&input).expect("Upsample failed");
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        assert!((output[[0, 0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((output[[0, 0, 0, 1]] - 1.0).abs() < 1e-10);
        assert!((output[[0, 0, 1, 0]] - 1.0).abs() < 1e-10);
        assert!((output[[0, 0, 1, 1]] - 1.0).abs() < 1e-10);
        assert!((output[[0, 0, 2, 2]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_concat_channels() {
        let a: Array<f64, IxDyn> = Array::ones(IxDyn(&[1, 2, 4, 4]));
        let b: Array<f64, IxDyn> = Array::zeros(IxDyn(&[1, 3, 4, 4]));
        let cat = concat_channels(&a, &b).expect("Concat failed");
        assert_eq!(cat.shape(), &[1, 5, 4, 4]);
        assert!((cat[[0, 0, 0, 0]] - 1.0_f64).abs() < 1e-10);
        assert!((cat[[0, 3, 0, 0]] - 0.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_crop_to_match_same_size() {
        let a: Array<f64, IxDyn> = Array::ones(IxDyn(&[1, 1, 4, 4]));
        let b: Array<f64, IxDyn> = Array::zeros(IxDyn(&[1, 1, 4, 4]));
        let cropped = crop_to_match(&a, &b).expect("Crop failed");
        assert_eq!(cropped.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_crop_to_match_different_size() {
        let a: Array<f64, IxDyn> = Array::ones(IxDyn(&[1, 1, 6, 6]));
        let b: Array<f64, IxDyn> = Array::zeros(IxDyn(&[1, 1, 4, 4]));
        let cropped = crop_to_match(&a, &b).expect("Crop failed");
        assert_eq!(cropped.shape(), &[1, 1, 4, 4]);
    }
}
