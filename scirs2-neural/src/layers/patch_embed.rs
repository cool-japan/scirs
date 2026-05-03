//! Patch Embedding layer for Vision Transformers
//!
//! Implements the patch embedding operation used in ViT and related architectures.
//! An input image `[batch, channels, height, width]` is divided into non-overlapping
//! `patch_size × patch_size` patches, which are then linearly projected to an
//! embedding space of dimension `embed_dim`, yielding `[batch, num_patches, embed_dim]`.
//!
//! Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
//! Dosovitskiy et al. (2020). <https://arxiv.org/abs/2010.11929>

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Rng, RngExt, Uniform};
use std::fmt::Debug;
use std::sync::RwLock;

/// Patch Embedding layer.
///
/// Divides an image into fixed-size patches and projects each patch to an embedding
/// vector via a learned linear projection (optionally including a bias term).
///
/// # Shape
/// - Input: `[batch_size, in_channels, image_height, image_width]`
/// - Output: `[batch_size, num_patches, embed_dim]`
///   where `num_patches = (image_height / patch_size) * (image_width / patch_size)`
///
/// # Parameters
/// - `weight`: shape `[embed_dim, in_channels * patch_h * patch_w]` — the projection matrix
/// - `bias` (optional): shape `[embed_dim]` — per-embedding bias
pub struct PatchEmbedding<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Image size `(height, width)` expected as input
    image_size: (usize, usize),
    /// Patch size `(height, width)`
    patch_size: (usize, usize),
    /// Number of input channels
    in_channels: usize,
    /// Output embedding dimension
    embed_dim: usize,
    /// Number of patches along height
    num_patches_h: usize,
    /// Number of patches along width
    num_patches_w: usize,
    /// Flat patch dimension: `in_channels * patch_h * patch_w`
    patch_dim: usize,

    /// Projection weight matrix, shape `[embed_dim, patch_dim]`
    weight: Array<F, IxDyn>,
    /// Projection bias, shape `[embed_dim]` (optional but always allocated as zeros)
    bias: Array<F, IxDyn>,
    /// Whether a bias term is used
    use_bias: bool,

    /// Gradient for weight
    d_weight: RwLock<Array<F, IxDyn>>,
    /// Gradient for bias
    d_bias: RwLock<Array<F, IxDyn>>,
    /// Cached input from most recent forward pass (for backward)
    cached_patches: RwLock<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Debug
    for PatchEmbedding<F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchEmbedding")
            .field("image_size", &self.image_size)
            .field("patch_size", &self.patch_size)
            .field("in_channels", &self.in_channels)
            .field("embed_dim", &self.embed_dim)
            .field("num_patches", &(self.num_patches_h * self.num_patches_w))
            .field("use_bias", &self.use_bias)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Clone
    for PatchEmbedding<F>
{
    fn clone(&self) -> Self {
        Self {
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            embed_dim: self.embed_dim,
            num_patches_h: self.num_patches_h,
            num_patches_w: self.num_patches_w,
            patch_dim: self.patch_dim,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            use_bias: self.use_bias,
            d_weight: RwLock::new(
                self.d_weight
                    .read()
                    .expect("RwLock poisoned on d_weight read")
                    .clone(),
            ),
            d_bias: RwLock::new(
                self.d_bias
                    .read()
                    .expect("RwLock poisoned on d_bias read")
                    .clone(),
            ),
            cached_patches: RwLock::new(
                self.cached_patches
                    .read()
                    .expect("RwLock poisoned on cached_patches read")
                    .clone(),
            ),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> PatchEmbedding<F> {
    /// Create a new PatchEmbedding layer.
    ///
    /// # Arguments
    /// * `image_size` — expected input image dimensions `(height, width)`
    /// * `patch_size` — patch dimensions `(height, width)` — must divide `image_size` evenly
    /// * `in_channels` — number of input image channels
    /// * `embed_dim` — output embedding dimension
    /// * `use_bias` — whether to include a learnable bias term
    /// * `rng` — random number generator for weight initialisation
    pub fn new<R: Rng>(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        embed_dim: usize,
        use_bias: bool,
        rng: &mut R,
    ) -> Result<Self> {
        if image_size.0 == 0 || image_size.1 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "image_size dimensions must be non-zero".to_string(),
            ));
        }
        if patch_size.0 == 0 || patch_size.1 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "patch_size dimensions must be non-zero".to_string(),
            ));
        }
        if !image_size.0.is_multiple_of(patch_size.0) || !image_size.1.is_multiple_of(patch_size.1)
        {
            return Err(NeuralError::InvalidArchitecture(format!(
                "image_size {:?} must be divisible by patch_size {:?}",
                image_size, patch_size
            )));
        }
        if in_channels == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "in_channels must be non-zero".to_string(),
            ));
        }
        if embed_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "embed_dim must be non-zero".to_string(),
            ));
        }

        let num_patches_h = image_size.0 / patch_size.0;
        let num_patches_w = image_size.1 / patch_size.1;
        let patch_dim = in_channels * patch_size.0 * patch_size.1;

        // Kaiming / Xavier uniform initialisation: uniform(-bound, bound)
        // bound = sqrt(6 / (fan_in + fan_out)) for Xavier uniform
        let fan_in = patch_dim as f64;
        let fan_out = embed_dim as f64;
        let bound = f64::sqrt(6.0 / (fan_in + fan_out));

        let uniform = Uniform::new(-bound, bound).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create uniform distribution: {e}"))
        })?;

        // Weight: [embed_dim, patch_dim]
        let weight_vec: Vec<F> = (0..(embed_dim * patch_dim))
            .map(|_| {
                F::from(uniform.sample(rng))
                    .ok_or_else(|| {
                        NeuralError::InvalidArchitecture(
                            "Failed to convert random value to float type".to_string(),
                        )
                    })
                    .unwrap_or(F::zero())
            })
            .collect();

        let weight =
            Array::from_shape_vec(IxDyn(&[embed_dim, patch_dim]), weight_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to construct weight array: {e}"))
            })?;

        // Bias: [embed_dim], initialised to zero
        let bias = Array::zeros(IxDyn(&[embed_dim]));

        let d_weight = RwLock::new(Array::zeros(IxDyn(&[embed_dim, patch_dim])));
        let d_bias = RwLock::new(Array::zeros(IxDyn(&[embed_dim])));

        Ok(Self {
            image_size,
            patch_size,
            in_channels,
            embed_dim,
            num_patches_h,
            num_patches_w,
            patch_dim,
            weight,
            bias,
            use_bias,
            d_weight,
            d_bias,
            cached_patches: RwLock::new(None),
        })
    }

    /// Total number of patches produced from one image
    pub fn num_patches(&self) -> usize {
        self.num_patches_h * self.num_patches_w
    }

    /// Flat patch vector dimension: `in_channels * patch_h * patch_w`
    pub fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    /// Extract and flatten patches from the input image batch.
    ///
    /// Returns an array of shape `[batch, num_patches, patch_dim]`.
    fn extract_patches(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        let batch = shape[0];
        let num_patches = self.num_patches_h * self.num_patches_w;

        let mut patches = Array::zeros(IxDyn(&[batch, num_patches, self.patch_dim]));

        for b in 0..batch {
            for ph in 0..self.num_patches_h {
                for pw in 0..self.num_patches_w {
                    let patch_idx = ph * self.num_patches_w + pw;
                    // Top-left pixel of this patch in the image
                    let h_start = ph * self.patch_size.0;
                    let w_start = pw * self.patch_size.1;
                    let mut flat_idx = 0usize;
                    for c in 0..self.in_channels {
                        for dy in 0..self.patch_size.0 {
                            for dx in 0..self.patch_size.1 {
                                patches[[b, patch_idx, flat_idx]] =
                                    input[[b, c, h_start + dy, w_start + dx]];
                                flat_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(patches)
    }

    /// Apply the linear projection `patches @ weight.T + bias` over all patches.
    ///
    /// Input shape: `[batch, num_patches, patch_dim]`
    /// Output shape: `[batch, num_patches, embed_dim]`
    fn linear_project(&self, patches: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let batch = patches.shape()[0];
        let num_patches = patches.shape()[1];
        let mut output = Array::zeros(IxDyn(&[batch, num_patches, self.embed_dim]));

        for b in 0..batch {
            for p in 0..num_patches {
                for e in 0..self.embed_dim {
                    let mut acc = F::zero();
                    for k in 0..self.patch_dim {
                        acc += patches[[b, p, k]] * self.weight[[e, k]];
                    }
                    if self.use_bias {
                        acc += self.bias[e];
                    }
                    output[[b, p, e]] = acc;
                }
            }
        }

        Ok(output)
    }

    /// Validate input tensor shape for this layer.
    fn validate_input_shape(&self, input: &Array<F, IxDyn>) -> Result<()> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "PatchEmbedding expects 4-D input [batch, channels, height, width], got {:?}",
                shape
            )));
        }
        if shape[1] != self.in_channels {
            return Err(NeuralError::InferenceError(format!(
                "PatchEmbedding: expected {} input channels, got {}",
                self.in_channels, shape[1]
            )));
        }
        if shape[2] != self.image_size.0 || shape[3] != self.image_size.1 {
            return Err(NeuralError::InferenceError(format!(
                "PatchEmbedding: expected image size {:?}, got ({}, {})",
                self.image_size, shape[2], shape[3]
            )));
        }
        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for PatchEmbedding<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.validate_input_shape(input)?;

        // Extract patches: [batch, num_patches, patch_dim]
        let patches = self.extract_patches(input)?;

        // Cache flattened patches for backward pass
        {
            let mut cache = self
                .cached_patches
                .write()
                .expect("RwLock poisoned on cached_patches write");
            *cache = Some(patches.clone());
        }

        // Linear projection: [batch, num_patches, embed_dim]
        self.linear_project(&patches)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // grad_output shape: [batch, num_patches, embed_dim]
        let go_shape = grad_output.shape();
        if go_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "PatchEmbedding backward: grad_output must be 3-D [batch, num_patches, embed_dim], got {:?}",
                go_shape
            )));
        }
        let batch = go_shape[0];
        let num_patches = go_shape[1];

        let patches = {
            let cache = self
                .cached_patches
                .read()
                .expect("RwLock poisoned on cached_patches read");
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError(
                    "PatchEmbedding backward called before forward — no cached patches".to_string(),
                )
            })?
        };

        // Gradient w.r.t. weight: d_weight[e, k] = sum_{b,p} grad_output[b,p,e] * patches[b,p,k]
        let mut d_weight = Array::zeros(IxDyn(&[self.embed_dim, self.patch_dim]));
        for b in 0..batch {
            for p in 0..num_patches {
                for e in 0..self.embed_dim {
                    let go = grad_output[[b, p, e]];
                    for k in 0..self.patch_dim {
                        d_weight[[e, k]] += go * patches[[b, p, k]];
                    }
                }
            }
        }

        // Gradient w.r.t. bias: d_bias[e] = sum_{b,p} grad_output[b,p,e]
        let mut d_bias = Array::zeros(IxDyn(&[self.embed_dim]));
        if self.use_bias {
            for b in 0..batch {
                for p in 0..num_patches {
                    for e in 0..self.embed_dim {
                        d_bias[e] += grad_output[[b, p, e]];
                    }
                }
            }
        }

        // Store gradients
        {
            let mut dw = self
                .d_weight
                .write()
                .expect("RwLock poisoned on d_weight write");
            *dw = d_weight;
        }
        {
            let mut db = self
                .d_bias
                .write()
                .expect("RwLock poisoned on d_bias write");
            *db = d_bias;
        }

        // Gradient w.r.t. input patches: d_patches[b,p,k] = sum_e grad_output[b,p,e] * weight[e,k]
        let mut d_patches = Array::zeros(IxDyn(&[batch, num_patches, self.patch_dim]));
        for b in 0..batch {
            for p in 0..num_patches {
                for k in 0..self.patch_dim {
                    let mut acc = F::zero();
                    for e in 0..self.embed_dim {
                        acc += grad_output[[b, p, e]] * self.weight[[e, k]];
                    }
                    d_patches[[b, p, k]] = acc;
                }
            }
        }

        // Scatter gradient back into image-shaped tensor [batch, channels, H, W]
        let mut d_input = Array::zeros(IxDyn(&[
            batch,
            self.in_channels,
            self.image_size.0,
            self.image_size.1,
        ]));
        for b in 0..batch {
            for ph in 0..self.num_patches_h {
                for pw in 0..self.num_patches_w {
                    let patch_idx = ph * self.num_patches_w + pw;
                    let h_start = ph * self.patch_size.0;
                    let w_start = pw * self.patch_size.1;
                    let mut flat_idx = 0usize;
                    for c in 0..self.in_channels {
                        for dy in 0..self.patch_size.0 {
                            for dx in 0..self.patch_size.1 {
                                d_input[[b, c, h_start + dy, w_start + dx]] +=
                                    d_patches[[b, patch_idx, flat_idx]];
                                flat_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(d_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        let d_weight = {
            self.d_weight
                .read()
                .expect("RwLock poisoned on d_weight read")
                .clone()
        };
        let d_bias = {
            self.d_bias
                .read()
                .expect("RwLock poisoned on d_bias read")
                .clone()
        };

        // SGD update for weight
        for e in 0..self.embed_dim {
            for k in 0..self.patch_dim {
                self.weight[[e, k]] -= learning_rate * d_weight[[e, k]];
            }
        }

        // SGD update for bias
        if self.use_bias {
            for e in 0..self.embed_dim {
                self.bias[e] -= learning_rate * d_bias[e];
            }
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "PatchEmbedding"
    }

    fn parameter_count(&self) -> usize {
        let weight_params = self.embed_dim * self.patch_dim;
        let bias_params = if self.use_bias { self.embed_dim } else { 0 };
        weight_params + bias_params
    }

    fn layer_description(&self) -> String {
        format!(
            "type:PatchEmbedding, image_size:{:?}, patch_size:{:?}, in_channels:{}, embed_dim:{}, num_patches:{}, params:{}",
            self.image_size,
            self.patch_size,
            self.in_channels,
            self.embed_dim,
            self.num_patches(),
            self.parameter_count()
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        if self.use_bias {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![self.weight.clone()]
        }
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.is_empty() {
            return Err(NeuralError::InvalidArchitecture(
                "PatchEmbedding set_params: expected at least 1 parameter (weight)".to_string(),
            ));
        }
        self.weight = params[0].clone();
        if self.use_bias && params.len() >= 2 {
            self.bias = params[1].clone();
        }
        Ok(())
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.in_channels, self.image_size.0, self.image_size.1])
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.num_patches(), self.embed_dim])
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ParamLayer<F>
    for PatchEmbedding<F>
{
    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        self.params()
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        let dw = self
            .d_weight
            .read()
            .expect("RwLock poisoned on d_weight read")
            .clone();
        if self.use_bias {
            let db = self
                .d_bias
                .read()
                .expect("RwLock poisoned on d_bias read")
                .clone();
            vec![dw, db]
        } else {
            vec![dw]
        }
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        self.set_params(&params)
    }
}

// Safety: PatchEmbedding is safe to send across threads; interior mutability is via RwLock
unsafe impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Send for PatchEmbedding<F> {}
unsafe impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Sync for PatchEmbedding<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    fn make_embed(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        embed_dim: usize,
    ) -> PatchEmbedding<f64> {
        let mut rng = SmallRng::from_seed([0u8; 32]);
        PatchEmbedding::new(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
            true,
            &mut rng,
        )
        .expect("Failed to construct PatchEmbedding")
    }

    #[test]
    fn test_patch_embedding_output_shape() {
        // 8×8 image, 2×2 patches → 4×4 = 16 patches
        let layer = make_embed((8, 8), (2, 2), 3, 32);
        assert_eq!(layer.num_patches(), 16);
        assert_eq!(layer.patch_dim(), 3 * 2 * 2);

        let batch = 2usize;
        let input = Array::zeros(IxDyn(&[batch, 3, 8, 8]));
        let output = layer.forward(&input).expect("Forward pass failed");
        assert_eq!(output.shape(), &[batch, 16, 32]);
    }

    #[test]
    fn test_patch_embedding_parameter_count() {
        let layer = make_embed((16, 16), (4, 4), 3, 64);
        // patch_dim = 3*4*4 = 48; weight = 64*48 = 3072; bias = 64
        assert_eq!(layer.parameter_count(), 64 * 48 + 64);
    }

    #[test]
    fn test_patch_embedding_backward_shape() {
        let layer = make_embed((8, 8), (2, 2), 3, 32);
        let batch = 2usize;
        let input = Array::zeros(IxDyn(&[batch, 3, 8, 8]));
        let output = layer.forward(&input).expect("Forward failed");
        let grad_out = Array::ones(output.raw_dim());
        let grad_in = layer
            .backward(&input, &grad_out)
            .expect("Backward pass failed");
        // Gradient must match input shape
        assert_eq!(grad_in.shape(), input.shape());
    }

    #[test]
    fn test_patch_embedding_invalid_size() {
        let mut rng = SmallRng::from_seed([0u8; 32]);
        // 7 is not divisible by 4
        let result = PatchEmbedding::<f64>::new((7, 8), (4, 4), 3, 32, true, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_embedding_update() {
        let mut layer = make_embed((8, 8), (2, 2), 1, 16);
        let input = Array::zeros(IxDyn(&[1, 1, 8, 8]));
        let output = layer.forward(&input).expect("Forward failed");
        let grad_out = Array::ones(output.raw_dim());
        layer.backward(&input, &grad_out).expect("Backward failed");
        layer.update(0.01f64).expect("Update failed");
    }

    #[test]
    fn test_patch_embedding_round_trip_params() {
        // get_parameters → set_parameters must preserve forward output identically
        let mut layer = make_embed((8, 8), (2, 2), 3, 16);
        let input = Array::ones(IxDyn(&[1, 3, 8, 8]));

        let out_before = layer
            .forward(&input)
            .expect("Forward before round-trip failed");

        // Extract parameters, set them back
        let params = layer.get_parameters();
        layer
            .set_parameters(params.clone())
            .expect("set_parameters failed");

        let out_after = layer
            .forward(&input)
            .expect("Forward after round-trip failed");

        // Outputs must be bit-identical
        assert_eq!(out_before.shape(), out_after.shape());
        for (a, b) in out_before.iter().zip(out_after.iter()) {
            assert!((a - b).abs() < 1e-12, "round-trip mismatch: {a} vs {b}");
        }

        // Parameter count consistency
        // make_embed((8,8),(2,2),3,16): embed_dim=16, patch_dim=3*2*2=12
        // weight: 16*12=192 elements; bias: 16 elements
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].len(), 16 * 3 * 2 * 2); // embed_dim * patch_dim
        assert_eq!(params[1].len(), 16); // embed_dim
    }
}
