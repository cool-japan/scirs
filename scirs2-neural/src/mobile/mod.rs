//! Mobile neural network deployment and optimisation
//!
//! This module provides lightweight building blocks for deploying neural networks
//! on mobile and edge devices:
//!
//! - [`MobileNetConfig`] – configuration for MobileNet-style architectures
//! - [`DepthwiseSeparableConv`] – depthwise + pointwise convolution layer
//! - [`MobileNetV2Block`] – inverted residual block (MobileNetV2 bottleneck)
//! - [`MobileOptimizer`] – quantization / pruning utilities for mobile deployment
//!
//! # References
//! - Howard et al., "MobileNets", 2017 <https://arxiv.org/abs/1704.04861>
//! - Sandler et al., "MobileNetV2", 2018 <https://arxiv.org/abs/1801.04381>

use crate::error::{NeuralError, Result};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// MobileNetConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a MobileNet-style architecture.
///
/// # Examples
/// ```
/// use scirs2_neural::mobile::MobileNetConfig;
///
/// let cfg = MobileNetConfig::mobilenet_v1();
/// assert_eq!(cfg.input_resolution, 224);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileNetConfig {
    /// Width multiplier α (scales number of channels, typical 0.25–1.0)
    pub width_multiplier: f64,
    /// Input image resolution (square, typical 128/160/192/224)
    pub input_resolution: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// MobileNet version
    pub version: MobileNetVersion,
    /// Dropout rate before the final classifier
    pub dropout_rate: f64,
    /// Whether to use batch normalisation
    pub use_batch_norm: bool,
}

/// MobileNet architecture version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MobileNetVersion {
    /// MobileNetV1 – depthwise separable convolutions
    V1,
    /// MobileNetV2 – inverted residuals + linear bottlenecks
    V2,
    /// MobileNetV3 – hard-swish, SE blocks, NAS-searched
    V3Small,
    /// MobileNetV3 Large variant
    V3Large,
}

impl MobileNetConfig {
    /// Standard MobileNetV1 configuration (1× width, 224×224 input).
    pub fn mobilenet_v1() -> Self {
        Self {
            width_multiplier: 1.0,
            input_resolution: 224,
            num_classes: 1000,
            version: MobileNetVersion::V1,
            dropout_rate: 0.001,
            use_batch_norm: true,
        }
    }

    /// Standard MobileNetV2 configuration (1× width, 224×224 input).
    pub fn mobilenet_v2() -> Self {
        Self {
            width_multiplier: 1.0,
            input_resolution: 224,
            num_classes: 1000,
            version: MobileNetVersion::V2,
            dropout_rate: 0.2,
            use_batch_norm: true,
        }
    }

    /// Lightweight configuration suitable for low-power devices (0.25× width, 128×128).
    pub fn mobile_lite() -> Self {
        Self {
            width_multiplier: 0.25,
            input_resolution: 128,
            num_classes: 10,
            version: MobileNetVersion::V2,
            dropout_rate: 0.0,
            use_batch_norm: true,
        }
    }

    /// Compute the number of channels at a given base channel count after applying the width multiplier.
    pub fn scaled_channels(&self, base: usize) -> usize {
        ((base as f64) * self.width_multiplier).round() as usize
    }

    /// Estimate total parameter count for a simple 4-layer depthwise separable network.
    pub fn estimated_param_count(&self) -> usize {
        // Rough estimate: 3×3 depthwise + 1×1 pointwise × 4 stages × channel size
        let c = self.scaled_channels(32);
        let dw_params = 3 * 3 * c; // depthwise kernel
        let pw_params = c * (c * 2); // pointwise expanding
        (dw_params + pw_params) * 4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthwiseSeparableConv
// ─────────────────────────────────────────────────────────────────────────────

/// Depthwise-separable convolution layer (inference-only, f32 weights).
///
/// Factorises a standard K×K convolution into:
/// 1. **Depthwise** 3×3 conv (one filter per input channel)
/// 2. **Pointwise** 1×1 conv (mixes channels)
///
/// Weight layout:
/// - depthwise weights: `[in_channels, kH, kW]`
/// - pointwise weights: `[out_channels, in_channels]`
/// - biases: `[out_channels]`
///
/// # Examples
/// ```
/// use scirs2_neural::mobile::DepthwiseSeparableConv;
///
/// let dsc = DepthwiseSeparableConv::new(8, 16, (3, 3)).expect("dsc");
/// assert_eq!(dsc.in_channels(), 8);
/// assert_eq!(dsc.out_channels(), 16);
/// ```
#[derive(Debug, Clone)]
pub struct DepthwiseSeparableConv {
    in_ch: usize,
    out_ch: usize,
    kernel_size: (usize, usize),
    // depthwise weights [in_ch, kH, kW]
    dw_weights: Vec<f32>,
    // pointwise weights [out_ch, in_ch]
    pw_weights: Vec<f32>,
    // bias [out_ch]
    bias: Vec<f32>,
}

impl DepthwiseSeparableConv {
    /// Create a new layer with randomly-initialised He weights.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(NeuralError::InvalidArgument(
                "DepthwiseSeparableConv: channel counts must be > 0".to_string(),
            ));
        }
        let (kh, kw) = kernel_size;
        let dw_size = in_channels * kh * kw;
        let pw_size = out_channels * in_channels;

        // He initialisation scale
        let dw_scale = (2.0_f32 / (kh * kw) as f32).sqrt();
        let pw_scale = (2.0_f32 / in_channels as f32).sqrt();

        let dw_weights = pseudo_random_weights(dw_size, dw_scale, 1);
        let pw_weights = pseudo_random_weights(pw_size, pw_scale, 2);
        let bias = vec![0.0_f32; out_channels];

        Ok(Self {
            in_ch: in_channels,
            out_ch: out_channels,
            kernel_size,
            dw_weights,
            pw_weights,
            bias,
        })
    }

    /// Returns the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// Returns the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// Returns the kernel size.
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Total trainable parameter count.
    pub fn parameter_count(&self) -> usize {
        self.dw_weights.len() + self.pw_weights.len() + self.bias.len()
    }

    /// Forward pass on a flat `[batch × in_ch × H × W]` f32 slice.
    ///
    /// `input_shape` must be `[batch, in_ch, H, W]`.
    /// Returns a flat `[batch × out_ch × H_out × W_out]` vector.
    pub fn forward(
        &self,
        input: &[f32],
        input_shape: [usize; 4],
    ) -> Result<(Vec<f32>, [usize; 4])> {
        let [batch, in_ch, h, w] = input_shape;
        if in_ch != self.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "DepthwiseSeparableConv: expected in_ch={}, got {}",
                self.in_ch, in_ch
            )));
        }
        if input.len() != batch * in_ch * h * w {
            return Err(NeuralError::ShapeMismatch(
                "DepthwiseSeparableConv: input slice length mismatch".to_string(),
            ));
        }

        let (kh, kw) = self.kernel_size;
        let padding = (kh / 2, kw / 2);
        let h_out = (h + 2 * padding.0).saturating_sub(kh) + 1;
        let w_out = (w + 2 * padding.1).saturating_sub(kw) + 1;

        // ── 1. Depthwise conv ──────────────────────────────────────────────
        let dw_size = batch * in_ch * h_out * w_out;
        let mut dw_out = vec![0.0_f32; dw_size];

        for b in 0..batch {
            for c in 0..in_ch {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0_f32;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh + ki;
                                let iw = ow + kj;
                                // Padding check (implicit zero-padding)
                                let ih_src = ih.wrapping_sub(padding.0);
                                let iw_src = iw.wrapping_sub(padding.1);
                                if ih_src < h && iw_src < w {
                                    let in_idx =
                                        b * in_ch * h * w + c * h * w + ih_src * w + iw_src;
                                    let w_idx = c * kh * kw + ki * kw + kj;
                                    acc += input[in_idx] * self.dw_weights[w_idx];
                                }
                            }
                        }
                        // ReLU6
                        let idx = b * in_ch * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                        dw_out[idx] = acc.clamp(0.0, 6.0);
                    }
                }
            }
        }

        // ── 2. Pointwise conv (1×1) ────────────────────────────────────────
        let pw_size = batch * self.out_ch * h_out * w_out;
        let mut pw_out = vec![0.0_f32; pw_size];

        for b in 0..batch {
            for oc in 0..self.out_ch {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = self.bias[oc];
                        for ic in 0..in_ch {
                            let dw_idx =
                                b * in_ch * h_out * w_out + ic * h_out * w_out + oh * w_out + ow;
                            let pw_idx = oc * in_ch + ic;
                            acc += dw_out[dw_idx] * self.pw_weights[pw_idx];
                        }
                        let out_idx =
                            b * self.out_ch * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                        pw_out[out_idx] = acc.clamp(0.0, 6.0);
                    }
                }
            }
        }

        Ok((pw_out, [batch, self.out_ch, h_out, w_out]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MobileNetV2Block
// ─────────────────────────────────────────────────────────────────────────────

/// MobileNetV2 inverted residual block.
///
/// Structure (for stride=1 with residual shortcut):
/// ```text
/// input  ──[PW expand]──[DW 3×3]──[PW project]──⊕── output
///        └────────────────────────────────────────┘
/// ```
/// When `stride > 1` or `in_channels != out_channels`, no residual is added.
///
/// # Examples
/// ```
/// use scirs2_neural::mobile::MobileNetV2Block;
///
/// let block = MobileNetV2Block::new(32, 16, 6, 1).expect("block");
/// assert_eq!(block.out_channels(), 16);
/// ```
#[derive(Debug, Clone)]
pub struct MobileNetV2Block {
    in_ch: usize,
    out_ch: usize,
    expansion: usize,
    stride: usize,
    /// Expansion pointwise: in_ch → expanded_ch
    expand_pw: Option<PointwiseConv>,
    /// Depthwise: expanded_ch × 3×3
    dw: DepthwiseSeparableConv,
    /// Projection pointwise: expanded_ch → out_ch (no activation)
    project_pw: PointwiseConv,
    /// Whether to add a residual shortcut
    use_residual: bool,
}

impl MobileNetV2Block {
    /// Create a new inverted residual block.
    ///
    /// # Arguments
    /// * `in_channels` – input feature maps
    /// * `out_channels` – output feature maps
    /// * `expansion_factor` – channel expansion multiplier (typically 6)
    /// * `stride` – depthwise convolution stride (1 or 2)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        expansion_factor: usize,
        stride: usize,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(NeuralError::InvalidArgument(
                "MobileNetV2Block: channel counts must be > 0".to_string(),
            ));
        }
        if stride == 0 {
            return Err(NeuralError::InvalidArgument(
                "MobileNetV2Block: stride must be >= 1".to_string(),
            ));
        }

        let expanded_ch = in_channels * expansion_factor;

        // Expand PW (skip for expansion factor == 1)
        let expand_pw = if expansion_factor != 1 {
            Some(PointwiseConv::new(in_channels, expanded_ch)?)
        } else {
            None
        };

        // Depthwise conv on expanded channels
        let dw = DepthwiseSeparableConv::new(expanded_ch, expanded_ch, (3, 3))?;
        // Project PW (no activation)
        let project_pw = PointwiseConv::new(expanded_ch, out_channels)?;

        let use_residual = stride == 1 && in_channels == out_channels;

        Ok(Self {
            in_ch: in_channels,
            out_ch: out_channels,
            expansion: expansion_factor,
            stride,
            expand_pw,
            dw,
            project_pw,
            use_residual,
        })
    }

    /// Returns the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// Returns the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// Returns the expansion factor.
    pub fn expansion(&self) -> usize {
        self.expansion
    }

    /// Returns the stride.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns whether this block uses a residual shortcut.
    pub fn has_residual(&self) -> bool {
        self.use_residual
    }

    /// Total parameter count.
    pub fn parameter_count(&self) -> usize {
        let expand = self
            .expand_pw
            .as_ref()
            .map(|p| p.parameter_count())
            .unwrap_or(0);
        expand + self.dw.parameter_count() + self.project_pw.parameter_count()
    }

    /// Forward pass.
    ///
    /// Input: flat `[batch × in_ch × H × W]` f32 slice.
    /// Returns: `(output_flat, [batch, out_ch, H_out, W_out])`.
    pub fn forward(&self, input: &[f32], shape: [usize; 4]) -> Result<(Vec<f32>, [usize; 4])> {
        let [batch, in_ch, h, w] = shape;
        if in_ch != self.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "MobileNetV2Block: expected in_ch={}, got {}",
                self.in_ch, in_ch
            )));
        }

        // ── Expand PW ─────────────────────────────────────────────────────
        let (expanded, expanded_shape) = if let Some(ref pw) = self.expand_pw {
            pw.forward_with_relu6(input, shape)?
        } else {
            (input.to_vec(), shape)
        };

        // ── Depthwise conv ─────────────────────────────────────────────────
        // We only use the depthwise part (forward already includes pointwise,
        // but we manually call the depthwise and skip the pointwise channel-mix)
        let (dw_out, dw_shape) = depthwise_only(
            &expanded,
            expanded_shape,
            &self.dw.dw_weights,
            self.dw.kernel_size,
            self.stride,
        )?;

        // ── Project PW ────────────────────────────────────────────────────
        let (projected, proj_shape) = self
            .project_pw
            .forward_linear(dw_out.as_slice(), dw_shape)?;

        // ── Residual shortcut ─────────────────────────────────────────────
        let output = if self.use_residual {
            input
                .iter()
                .zip(projected.iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            projected
        };

        Ok((output, proj_shape))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PointwiseConv (internal helper)
// ─────────────────────────────────────────────────────────────────────────────

/// 1×1 pointwise convolution (internal helper).
#[derive(Debug, Clone)]
struct PointwiseConv {
    in_ch: usize,
    out_ch: usize,
    weights: Vec<f32>, // [out_ch, in_ch]
    bias: Vec<f32>,    // [out_ch]
}

impl PointwiseConv {
    fn new(in_channels: usize, out_channels: usize) -> Result<Self> {
        let size = out_channels * in_channels;
        let scale = (2.0_f32 / in_channels as f32).sqrt();
        Ok(Self {
            in_ch: in_channels,
            out_ch: out_channels,
            weights: pseudo_random_weights(size, scale, 3),
            bias: vec![0.0_f32; out_channels],
        })
    }

    fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Forward with ReLU6 activation.
    fn forward_with_relu6(
        &self,
        input: &[f32],
        shape: [usize; 4],
    ) -> Result<(Vec<f32>, [usize; 4])> {
        let [batch, in_ch, h, w] = shape;
        if in_ch != self.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "PointwiseConv: in_ch mismatch {} vs {}",
                self.in_ch, in_ch
            )));
        }
        let out_size = batch * self.out_ch * h * w;
        let mut out = vec![0.0_f32; out_size];

        for b in 0..batch {
            for oc in 0..self.out_ch {
                for ph in 0..h {
                    for pw_pos in 0..w {
                        let mut acc = self.bias[oc];
                        for ic in 0..in_ch {
                            let in_idx = b * in_ch * h * w + ic * h * w + ph * w + pw_pos;
                            acc += input[in_idx] * self.weights[oc * in_ch + ic];
                        }
                        let out_idx = b * self.out_ch * h * w + oc * h * w + ph * w + pw_pos;
                        out[out_idx] = acc.clamp(0.0, 6.0);
                    }
                }
            }
        }
        Ok((out, [batch, self.out_ch, h, w]))
    }

    /// Forward without activation (used for projection PW in V2 block).
    fn forward_linear(&self, input: &[f32], shape: [usize; 4]) -> Result<(Vec<f32>, [usize; 4])> {
        let [batch, in_ch, h, w] = shape;
        if in_ch != self.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "PointwiseConv(linear): in_ch mismatch {} vs {}",
                self.in_ch, in_ch
            )));
        }
        let out_size = batch * self.out_ch * h * w;
        let mut out = vec![0.0_f32; out_size];
        for b in 0..batch {
            for oc in 0..self.out_ch {
                for ph in 0..h {
                    for pw_pos in 0..w {
                        let mut acc = self.bias[oc];
                        for ic in 0..in_ch {
                            let in_idx = b * in_ch * h * w + ic * h * w + ph * w + pw_pos;
                            acc += input[in_idx] * self.weights[oc * in_ch + ic];
                        }
                        let out_idx = b * self.out_ch * h * w + oc * h * w + ph * w + pw_pos;
                        out[out_idx] = acc;
                    }
                }
            }
        }
        Ok((out, [batch, self.out_ch, h, w]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Depthwise-only helper
// ─────────────────────────────────────────────────────────────────────────────

/// Runs just the depthwise convolution part (no channel mixing).
fn depthwise_only(
    input: &[f32],
    shape: [usize; 4],
    weights: &[f32],
    kernel_size: (usize, usize),
    stride: usize,
) -> Result<(Vec<f32>, [usize; 4])> {
    let [batch, channels, h, w] = shape;
    let (kh, kw) = kernel_size;
    let padding = (kh / 2, kw / 2);
    let h_out = if stride == 1 {
        h
    } else {
        (h + 2 * padding.0).saturating_sub(kh) / stride + 1
    };
    let w_out = if stride == 1 {
        w
    } else {
        (w + 2 * padding.1).saturating_sub(kw) / stride + 1
    };

    let mut out = vec![0.0_f32; batch * channels * h_out * w_out];
    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0_f32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh * stride + ki;
                            let iw = ow * stride + kj;
                            let ih_src = ih.wrapping_sub(padding.0);
                            let iw_src = iw.wrapping_sub(padding.1);
                            if ih_src < h && iw_src < w {
                                let in_idx = b * channels * h * w + c * h * w + ih_src * w + iw_src;
                                let w_idx = c * kh * kw + ki * kw + kj;
                                acc += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                    let out_idx =
                        b * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                    out[out_idx] = acc.clamp(0.0, 6.0);
                }
            }
        }
    }
    Ok((out, [batch, channels, h_out, w_out]))
}

// ─────────────────────────────────────────────────────────────────────────────
// MobileOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Mobile-optimised model utilities.
///
/// Provides helpers to reduce model size and latency for mobile deployment.
pub struct MobileOptimizer {
    /// Target model size budget in kilobytes
    pub size_budget_kb: f64,
    /// Minimum acceptable accuracy drop (0.0–1.0)
    pub max_accuracy_drop: f64,
}

impl MobileOptimizer {
    /// Create a new mobile optimizer.
    pub fn new(size_budget_kb: f64, max_accuracy_drop: f64) -> Result<Self> {
        if size_budget_kb <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "size_budget_kb must be > 0".to_string(),
            ));
        }
        Ok(Self {
            size_budget_kb,
            max_accuracy_drop: max_accuracy_drop.clamp(0.0, 1.0),
        })
    }

    /// Estimate the byte size of a weight vector at a given bit-width.
    pub fn estimate_size_bytes(num_weights: usize, bits_per_weight: u8) -> usize {
        (num_weights * bits_per_weight as usize).div_ceil(8)
    }

    /// Quantise f32 weights to INT8 symmetric representation.
    ///
    /// Returns `(quantized, scale)`.
    pub fn quantize_int8(weights: &[f32]) -> Result<(Vec<i8>, f32)> {
        if weights.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "quantize_int8: empty weights".to_string(),
            ));
        }
        let abs_max = weights.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| (w / scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        Ok((quantized, scale))
    }

    /// Prune weights whose absolute value is below `threshold` (set to 0).
    pub fn magnitude_prune(weights: &mut [f32], sparsity: f64) {
        if weights.is_empty() || sparsity <= 0.0 {
            return;
        }
        let n = weights.len();
        let mut sorted_abs: Vec<f32> = weights.iter().map(|v| v.abs()).collect();
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff_idx = ((sparsity.clamp(0.0, 1.0) * n as f64) as usize).min(n.saturating_sub(1));
        let threshold = sorted_abs[cutoff_idx];
        for w in weights.iter_mut() {
            if w.abs() < threshold {
                *w = 0.0;
            }
        }
    }

    /// Returns whether a model fits within the configured size budget.
    pub fn fits_budget(&self, param_count: usize) -> bool {
        let bytes = Self::estimate_size_bytes(param_count, 32);
        (bytes as f64 / 1024.0) <= self.size_budget_kb
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate deterministic pseudo-random weights for testing (no external deps).
fn pseudo_random_weights(n: usize, scale: f32, seed_offset: u64) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_0000_0001u64.wrapping_add(seed_offset);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 33) as f32 / u32::MAX as f32; // [0, 1)
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_net_config_v1() {
        let cfg = MobileNetConfig::mobilenet_v1();
        assert_eq!(cfg.input_resolution, 224);
        assert_eq!(cfg.version, MobileNetVersion::V1);
        assert!((cfg.width_multiplier - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mobile_net_config_v2() {
        let cfg = MobileNetConfig::mobilenet_v2();
        assert_eq!(cfg.version, MobileNetVersion::V2);
    }

    #[test]
    fn test_scaled_channels() {
        let cfg = MobileNetConfig {
            width_multiplier: 0.5,
            ..MobileNetConfig::mobilenet_v2()
        };
        assert_eq!(cfg.scaled_channels(32), 16);
        assert_eq!(cfg.scaled_channels(64), 32);
    }

    #[test]
    fn test_depthwise_separable_conv_creation() {
        let dsc = DepthwiseSeparableConv::new(4, 8, (3, 3)).expect("dsc ok");
        assert_eq!(dsc.in_channels(), 4);
        assert_eq!(dsc.out_channels(), 8);
        assert!(dsc.parameter_count() > 0);
    }

    #[test]
    fn test_depthwise_separable_conv_forward() {
        let dsc = DepthwiseSeparableConv::new(2, 4, (3, 3)).expect("dsc ok");
        // batch=1, in_ch=2, H=8, W=8 (NCHW)
        let input = vec![0.5_f32; 2 * 8 * 8];
        let (output, out_shape) = dsc.forward(&input, [1, 2, 8, 8]).expect("forward ok");
        let [b, c, h, w] = out_shape;
        assert_eq!(b, 1);
        assert_eq!(c, 4);
        assert_eq!(h, 8); // same-padding
        assert_eq!(w, 8);
        assert_eq!(output.len(), b * c * h * w);
    }

    #[test]
    fn test_depthwise_separable_conv_channel_mismatch_err() {
        let dsc = DepthwiseSeparableConv::new(4, 8, (3, 3)).expect("dsc ok");
        let input = vec![0.0_f32; 2 * 4 * 4]; // wrong channels (batch=1 NCHW)
        let result = dsc.forward(&input, [1, 2, 4, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mobilenet_v2_block_creation() {
        let block = MobileNetV2Block::new(32, 16, 6, 1).expect("block ok");
        assert_eq!(block.in_channels(), 32);
        assert_eq!(block.out_channels(), 16);
        assert!(!block.has_residual()); // channels differ
    }

    #[test]
    fn test_mobilenet_v2_block_residual() {
        let block = MobileNetV2Block::new(16, 16, 6, 1).expect("block ok");
        assert!(block.has_residual());
    }

    #[test]
    fn test_mobilenet_v2_block_forward() {
        let block = MobileNetV2Block::new(8, 8, 6, 1).expect("block ok");
        let input = vec![0.1_f32; 8 * 4 * 4]; // batch=1 NCHW
        let (output, out_shape) = block.forward(&input, [1, 8, 4, 4]).expect("fwd ok");
        let [b, c, _h, _w] = out_shape;
        assert_eq!(b, 1);
        assert_eq!(c, 8);
        assert_eq!(output.len(), 8 * 4 * 4);
    }

    #[test]
    fn test_mobilenet_v2_block_stride2() {
        let block = MobileNetV2Block::new(8, 16, 6, 2).expect("block ok");
        assert!(!block.has_residual());
        let input = vec![0.1_f32; 8 * 8 * 8]; // batch=1 NCHW
        let (output, out_shape) = block.forward(&input, [1, 8, 8, 8]).expect("fwd ok");
        let [b, c, h, w] = out_shape;
        assert_eq!(b, 1);
        assert_eq!(c, 16);
        // With stride=2, spatial dims halve
        assert!(h <= 4 && w <= 4, "expected ≤4, got h={h} w={w}");
        assert_eq!(output.len(), b * c * h * w);
    }

    #[test]
    fn test_mobile_optimizer_quantize_int8() {
        let weights = vec![0.5_f32, -0.5, 1.0, -1.0, 0.0];
        let (q, scale) = MobileOptimizer::quantize_int8(&weights).expect("ok");
        assert_eq!(q.len(), weights.len());
        let dequant: Vec<f32> = q.iter().map(|&v| v as f32 * scale).collect();
        for (orig, deq) in weights.iter().zip(dequant.iter()) {
            assert!((orig - deq).abs() < 0.01, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn test_mobile_optimizer_prune() {
        let mut weights = vec![0.01_f32, 0.5, 0.001, 1.0, 0.002];
        MobileOptimizer::magnitude_prune(&mut weights, 0.6);
        // Bottom 60% (3 out of 5) should be zeroed
        let zeros = weights.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros >= 2, "expected ≥2 zeros, got {zeros}");
    }

    #[test]
    fn test_mobile_optimizer_budget() {
        let opt = MobileOptimizer::new(1000.0, 0.01).expect("ok");
        // 10 weights at FP32 = 40 bytes ≈ tiny, fits
        assert!(opt.fits_budget(10));
        // 10M weights at FP32 ≈ 40MB, won't fit in 1000 KB
        assert!(!opt.fits_budget(10_000_000));
    }

    #[test]
    fn test_depthwise_separable_conv_zero_channels_err() {
        assert!(DepthwiseSeparableConv::new(0, 8, (3, 3)).is_err());
    }
}
