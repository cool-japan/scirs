//! Embedding layer implementations
//!
//! This module provides implementations of various embedding layers
//! such as word embeddings, positional embeddings, and patch embeddings for vision.

use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::prelude::*;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::layers::Layer;
use crate::utils::initializers;

/// Configuration for the Embedding layer
pub struct EmbeddingConfig {
    /// Number of embeddings in the embedding table
    pub num_embeddings: usize,
    /// Dimension of each embedding vector
    pub embedding_dim: usize,
    /// Optional padding index that will have its embedding vector filled with zeros
    pub padding_idx: Option<usize>,
    /// Maximum norm for embedding vectors
    pub max_norm: Option<f64>,
    /// Type of norm to use with max_norm
    pub norm_type: f64,
    /// Whether to scale gradients by the inverse of frequency of the indices
    pub scale_grad_by_freq: bool,
    /// Whether to use sparse gradients for the embedding matrix
    pub sparse: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            num_embeddings: 1,
            embedding_dim: 1,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        }
    }
}

/// Embedding layer that stores embeddings for discrete inputs
///
/// This layer is often used to store word embeddings and retrieve them using indices.
/// The input to the module is a list of indices, and the output is the corresponding
/// embedding vectors.
pub struct Embedding<F: Float + Debug + ScalarOperand> {
    /// Configuration for the embedding layer
    pub config: EmbeddingConfig,
    /// Weight matrix containing the embeddings
    pub weight: Array<F, IxDyn>,
    /// Gradient of the weight matrix
    weight_grad: Array<F, IxDyn>,
    /// Frequency counter for indices
    freq_counter: Option<Vec<usize>>,
}

impl<F: Float + Debug + ScalarOperand> Embedding<F> {
    /// Create a new Embedding layer with the given configuration
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        if config.num_embeddings == 0 {
            return Err(Error::InvalidArchitecture(
                "num_embeddings must be greater than 0".to_string(),
            ));
        }
        if config.embedding_dim == 0 {
            return Err(Error::InvalidArchitecture(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }

        // Validate padding_idx
        if let Some(idx) = config.padding_idx {
            if idx >= config.num_embeddings {
                return Err(Error::InvalidArchitecture(format!(
                    "padding_idx ({}) must be less than num_embeddings ({})",
                    idx, config.num_embeddings
                )));
            }
        }

        // Initialize weights with standard distribution
        let weight_shape = IxDyn(&[config.num_embeddings, config.embedding_dim]);

        // Use standard distribution and scale it
        let mut rng = rand::rng();
        let mut weight = Array::from_shape_fn(weight_shape.clone(), |_| {
            let value: f64 = rng.random::<f64>();
            // Scale to approximate normal distribution N(0, 1)
            let scaled_value = (value * 2.0 - 1.0) * 0.5;
            F::from(scaled_value).unwrap()
        });

        // Initialize gradients with zeros
        let weight_grad = Array::zeros(weight_shape.clone());

        // Set padding_idx embeddings to zero if specified
        if let Some(idx) = config.padding_idx {
            let mut slice = weight.slice_mut(ndarray::s![idx, ..]);
            for item in slice.iter_mut() {
                *item = F::zero();
            }
        }

        // Initialize frequency counter if needed
        let freq_counter = if config.scale_grad_by_freq {
            Some(vec![0; config.num_embeddings])
        } else {
            None
        };

        Ok(Self {
            config,
            weight,
            weight_grad,
            freq_counter,
        })
    }

    /// Create an Embedding layer from pretrained embeddings
    pub fn from_pretrained(
        embeddings: Array<F, IxDyn>,
        padding_idx: Option<usize>,
        max_norm: Option<f64>,
        norm_type: f64,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> Result<Self> {
        if embeddings.ndim() != 2 {
            return Err(Error::InvalidArchitecture(
                "Embeddings parameter is expected to be 2-dimensional".to_string(),
            ));
        }

        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        // Validate padding_idx
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(Error::InvalidArchitecture(format!(
                    "padding_idx ({}) must be less than num_embeddings ({})",
                    idx, num_embeddings
                )));
            }
        }

        let config = EmbeddingConfig {
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        };

        // Clone the weights
        let weight = embeddings.clone();
        let weight_grad = Array::zeros(IxDyn(&[num_embeddings, embedding_dim]));

        // Initialize frequency counter if needed
        let freq_counter = if scale_grad_by_freq {
            Some(vec![0; num_embeddings])
        } else {
            None
        };

        Ok(Self {
            config,
            weight,
            weight_grad,
            freq_counter,
        })
    }

    /// Reset parameters of the embedding layer
    pub fn reset_parameters(&mut self) -> Result<()> {
        // Re-initialize weights with standard normal distribution
        let mut rng = rand::rng();
        for item in self.weight.iter_mut() {
            *item = F::from(rng.random::<f64>()).unwrap();
        }

        // Set padding_idx embeddings to zero if specified
        if let Some(idx) = self.config.padding_idx {
            let mut slice = self.weight.slice_mut(ndarray::s![idx, ..]);
            for item in slice.iter_mut() {
                *item = F::zero();
            }
        }

        // Reset gradients
        self.weight_grad.fill(F::zero());

        // Reset frequency counter if needed
        if let Some(counter) = &mut self.freq_counter {
            counter.iter_mut().for_each(|c| *c = 0);
        }

        Ok(())
    }

    /// Apply max_norm to the embeddings if specified
    fn apply_max_norm(&mut self) -> Result<()> {
        if let Some(max_norm) = self.config.max_norm {
            let norm_type = self.config.norm_type;
            let p = F::from(norm_type).ok_or_else(|| {
                Error::InvalidArchitecture(format!("Invalid norm_type: {}", norm_type))
            })?;
            let max_norm = F::from(max_norm).ok_or_else(|| {
                Error::InvalidArchitecture(format!("Invalid max_norm: {}", max_norm))
            })?;

            // Calculate norms for each embedding vector
            for i in 0..self.config.num_embeddings {
                let mut norm = F::zero();
                // Calculate p-norm
                for j in 0..self.config.embedding_dim {
                    let val = self.weight[[i, j]];
                    if p == F::from(2.0).unwrap() {
                        norm = norm + val * val;
                    } else {
                        norm = norm + val.abs().powf(p);
                    }
                }

                if p == F::from(2.0).unwrap() {
                    norm = norm.sqrt();
                } else {
                    norm = norm.powf(F::one() / p);
                }

                // Apply max_norm if needed
                if norm > max_norm {
                    let scale = max_norm / norm;
                    for j in 0..self.config.embedding_dim {
                        self.weight[[i, j]] = self.weight[[i, j]] * scale;
                    }
                }
            }
        }

        Ok(())
    }

    /// Internal forward pass implementation
    fn forward_impl<D: Dimension>(
        &mut self,
        indices: &ArrayBase<impl Data<Elem = usize>, D>,
    ) -> Result<Array<F, IxDyn>> {
        // Validate indices
        for &idx in indices.iter() {
            if idx >= self.config.num_embeddings {
                return Err(Error::InvalidArchitecture(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    idx, self.config.num_embeddings
                )));
            }
        }

        // Apply max_norm if specified
        self.apply_max_norm()?;

        // Update frequency counter if needed
        if let Some(counter) = &mut self.freq_counter {
            for &idx in indices.iter() {
                counter[idx] += 1;
            }
        }

        // Create output array
        let mut output_shape = Vec::with_capacity(indices.ndim() + 1);
        output_shape.extend_from_slice(indices.shape());
        output_shape.push(self.config.embedding_dim);

        let mut output = Array::zeros(IxDyn(output_shape.as_slice()));

        // Lookup embeddings
        let indices_flat = indices
            .view()
            .into_shape_with_order(IxDyn(&[indices.len()]))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();

        for (flat_idx, &idx) in indices_flat.iter().enumerate() {
            // Skip padding indices
            if let Some(padding_idx) = self.config.padding_idx {
                if idx == padding_idx {
                    // Already filled with zeros
                    continue;
                }
            }

            // Compute output index
            let mut output_idx = Vec::with_capacity(indices.ndim() + 1);
            let mut remaining = flat_idx;
            for &dim in indices.shape().iter().rev() {
                output_idx.push(remaining % dim);
                remaining /= dim;
            }
            output_idx.reverse();
            output_idx.push(0); // Starting embedding dimension index

            // Copy embedding to output
            for j in 0..self.config.embedding_dim {
                output_idx.last_mut().unwrap().clone_from(&j);
                let emb_val = self.weight[[idx, j]];
                output[IxDyn(output_idx.as_slice())] = emb_val;
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for Embedding<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Convert input to indices
        let indices = input
            .mapv(|x| x.to_usize().unwrap_or(0))
            .into_dimensionality::<IxDyn>()?;

        // Forward using the immutable version
        let mut embedding_mut = Embedding {
            config: EmbeddingConfig {
                num_embeddings: self.config.num_embeddings,
                embedding_dim: self.config.embedding_dim,
                padding_idx: self.config.padding_idx,
                max_norm: self.config.max_norm,
                norm_type: self.config.norm_type,
                scale_grad_by_freq: self.config.scale_grad_by_freq,
                sparse: self.config.sparse,
            },
            weight: self.weight.clone(),
            weight_grad: self.weight_grad.clone(),
            freq_counter: self.freq_counter.clone(),
        };

        embedding_mut.forward_impl(&indices)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Embedding has no meaningful upstream gradient since indices are discrete
        // Return zeros of the same shape as the input (indices)
        let input_shape = &input.shape();
        Ok(Array::zeros(IxDyn(input_shape)))
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights using accumulated gradients
        let lr = learning_rate;

        // Handle frequency-based scaling
        if let Some(counter) = &self.freq_counter {
            for i in 0..self.config.num_embeddings {
                // Skip padding indices
                if let Some(padding_idx) = self.config.padding_idx {
                    if i == padding_idx {
                        continue;
                    }
                }

                let scale = if counter[i] > 0 {
                    F::from(1.0 / counter[i] as f64).unwrap()
                } else {
                    F::one()
                };

                for j in 0..self.config.embedding_dim {
                    self.weight[[i, j]] =
                        self.weight[[i, j]] - lr * scale * self.weight_grad[[i, j]];
                }
            }
        } else {
            // Standard gradient update
            for i in 0..self.config.num_embeddings {
                // Skip padding indices
                if let Some(padding_idx) = self.config.padding_idx {
                    if i == padding_idx {
                        continue;
                    }
                }

                for j in 0..self.config.embedding_dim {
                    self.weight[[i, j]] = self.weight[[i, j]] - lr * self.weight_grad[[i, j]];
                }
            }
        }

        // Reset gradients
        self.weight_grad.fill(F::zero());

        // Reset frequency counter if needed
        if let Some(counter) = &mut self.freq_counter {
            counter.iter_mut().for_each(|c| *c = 0);
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

/// Positional Embedding layer for transformers and sequence models
///
/// This layer adds positional information to embeddings to help models
/// understand the position of elements in a sequence.
pub struct PositionalEmbedding<F: Float + Debug + ScalarOperand> {
    /// Maximum sequence length supported
    pub max_seq_length: usize,
    /// Dimension of each embedding vector
    pub embedding_dim: usize,
    /// Whether to use learned positional embeddings (true) or fixed sinusoidal (false)
    pub learned: bool,
    /// Weight matrix for learned positional embeddings
    pub weight: Option<Array<F, IxDyn>>,
    /// Gradient of the weight matrix
    weight_grad: Option<Array<F, IxDyn>>,
}

impl<F: Float + Debug + ScalarOperand> PositionalEmbedding<F> {
    /// Create a new PositionalEmbedding layer
    pub fn new(max_seq_length: usize, embedding_dim: usize, learned: bool) -> Result<Self> {
        if max_seq_length == 0 {
            return Err(Error::InvalidArchitecture(
                "max_seq_length must be greater than 0".to_string(),
            ));
        }
        if embedding_dim == 0 {
            return Err(Error::InvalidArchitecture(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }

        if learned {
            // Initialize learned positional embeddings
            let weight_shape = IxDyn(&[max_seq_length, embedding_dim]);
            let weight = Some(initializers::xavier_uniform::<F>(weight_shape.clone())?);
            let weight_grad = Some(Array::zeros(weight_shape));

            Ok(Self {
                max_seq_length,
                embedding_dim,
                learned,
                weight,
                weight_grad,
            })
        } else {
            // Sinusoidal positional embeddings are computed on the fly
            Ok(Self {
                max_seq_length,
                embedding_dim,
                learned,
                weight: None,
                weight_grad: None,
            })
        }
    }

    /// Generate sinusoidal positional embeddings
    fn generate_sinusoidal_embeddings(&self, seq_length: usize) -> Result<Array<F, IxDyn>> {
        if seq_length > self.max_seq_length {
            return Err(Error::InvalidArchitecture(format!(
                "Sequence length {} exceeds maximum supported length {}",
                seq_length, self.max_seq_length
            )));
        }

        // Initialize output array
        let mut pos_embeddings = Array::zeros(IxDyn(&[seq_length, self.embedding_dim]));

        // Generate sinusoidal positional embeddings
        for pos in 0..seq_length {
            for i in 0..self.embedding_dim {
                let div_term =
                    F::from((10000.0f64).powf(2.0 * (i / 2) as f64 / self.embedding_dim as f64))
                        .unwrap();

                if i % 2 == 0 {
                    // Sine for even dimensions
                    pos_embeddings[[pos, i]] = F::from(pos as f64 / div_term.to_f64().unwrap())
                        .unwrap()
                        .sin();
                } else {
                    // Cosine for odd dimensions
                    pos_embeddings[[pos, i]] = F::from(pos as f64 / div_term.to_f64().unwrap())
                        .unwrap()
                        .cos();
                }
            }
        }

        Ok(pos_embeddings)
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for PositionalEmbedding<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Validate input shape - at least 2D with last dimension being embedding_dim
        if input.ndim() < 2 {
            return Err(Error::InvalidArchitecture(
                "Input to PositionalEmbedding must be at least 2D".to_string(),
            ));
        }

        let last_dim = input.shape().last().unwrap();
        if *last_dim != self.embedding_dim {
            return Err(Error::InvalidArchitecture(format!(
                "Input embedding dimension {} doesn't match layer embedding dimension {}",
                last_dim, self.embedding_dim
            )));
        }

        // Get sequence length from the input shape
        let seq_dim = input.ndim() - 2;
        let seq_length = input.shape()[seq_dim];

        if seq_length > self.max_seq_length {
            return Err(Error::InvalidArchitecture(format!(
                "Input sequence length {} exceeds maximum supported length {}",
                seq_length, self.max_seq_length
            )));
        }

        if self.learned {
            // Use learned positional embeddings
            let pos_embeddings = self
                .weight
                .as_ref()
                .unwrap()
                .slice(ndarray::s![0..seq_length, ..]);

            // Add positional embeddings to input
            // Need to broadcast positional embeddings to match input shape
            let mut output = input.clone();

            // Iterate over all batch elements and add positional embeddings
            let batch_shape = &input.shape()[..seq_dim];
            let batch_size: usize = batch_shape.iter().product();

            for batch_idx in 0..batch_size {
                // Calculate the multi-dimensional batch index
                let mut multi_idx = Vec::with_capacity(seq_dim);
                let mut remaining = batch_idx;
                for &dim in batch_shape.iter().rev() {
                    multi_idx.push(remaining % dim);
                    remaining /= dim;
                }
                multi_idx.reverse();

                // For each position in the sequence
                for pos in 0..seq_length {
                    // Full index includes batch indices, sequence position, and embedding dimension
                    let mut full_idx = multi_idx.clone();
                    full_idx.push(pos);

                    // Add positional embeddings to each embedding dimension
                    for dim in 0..self.embedding_dim {
                        full_idx.push(dim);
                        let pos_val = pos_embeddings[[pos, dim]];
                        output[IxDyn(full_idx.as_slice())] =
                            output[IxDyn(full_idx.as_slice())] + pos_val;
                        full_idx.pop();
                    }
                }
            }

            Ok(output)
        } else {
            // Generate sinusoidal positional embeddings
            let pos_embeddings = self.generate_sinusoidal_embeddings(seq_length)?;

            // Add positional embeddings to input
            let mut output = input.clone();

            // Iterate over all batch elements and add positional embeddings
            let batch_shape = &input.shape()[..seq_dim];
            let batch_size: usize = batch_shape.iter().product();

            for batch_idx in 0..batch_size {
                // Calculate the multi-dimensional batch index
                let mut multi_idx = Vec::with_capacity(seq_dim);
                let mut remaining = batch_idx;
                for &dim in batch_shape.iter().rev() {
                    multi_idx.push(remaining % dim);
                    remaining /= dim;
                }
                multi_idx.reverse();

                // For each position in the sequence
                for pos in 0..seq_length {
                    // Full index includes batch indices, sequence position, and embedding dimension
                    let mut full_idx = multi_idx.clone();
                    full_idx.push(pos);

                    // Add positional embeddings to each embedding dimension
                    for dim in 0..self.embedding_dim {
                        full_idx.push(dim);
                        let pos_val = pos_embeddings[[pos, dim]];
                        output[IxDyn(full_idx.as_slice())] =
                            output[IxDyn(full_idx.as_slice())] + pos_val;
                        full_idx.pop();
                    }
                }
            }

            Ok(output)
        }
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // For PositionalEmbedding, gradients flow through directly
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Only update weights if using learned positional embeddings
        if self.learned {
            if let (Some(weight), Some(weight_grad)) = (&mut self.weight, &self.weight_grad) {
                // Update weights using accumulated gradients
                let lr = learning_rate;
                for i in 0..self.max_seq_length {
                    for j in 0..self.embedding_dim {
                        weight[[i, j]] = weight[[i, j]] - lr * weight_grad[[i, j]];
                    }
                }

                // Reset gradients
                self.weight_grad = Some(Array::zeros(IxDyn(&[
                    self.max_seq_length,
                    self.embedding_dim,
                ])));
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
}

/// Patch Embedding layer for vision transformers
///
/// This layer converts image patches into embeddings for vision transformers.
/// It applies a convolution to extract patches and flatten them into embedding vectors.
#[derive(Debug, Clone)]
pub struct PatchEmbedding<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Size of input images (height, width)
    pub image_size: (usize, usize),
    /// Size of patches (height, width)
    pub patch_size: (usize, usize),
    /// Number of input channels (e.g., 3 for RGB)
    pub in_channels: usize,
    /// Dimension of each embedding vector
    pub embedding_dim: usize,
    /// Weight matrix for patch extraction
    pub weight: Array<F, IxDyn>,
    /// Bias vector
    pub bias: Option<Array<F, IxDyn>>,
    /// Gradient of the weight matrix
    weight_grad: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of the bias vector
    bias_grad: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Input cache for backpropagation
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> PatchEmbedding<F> {
    /// Create a new PatchEmbedding layer
    pub fn new(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        embedding_dim: usize,
        use_bias: bool,
    ) -> Result<Self> {
        // Validate parameters
        if image_size.0 == 0 || image_size.1 == 0 {
            return Err(Error::InvalidArchitecture(
                "Image height and width must be greater than 0".to_string(),
            ));
        }
        if patch_size.0 == 0 || patch_size.1 == 0 {
            return Err(Error::InvalidArchitecture(
                "Patch height and width must be greater than 0".to_string(),
            ));
        }
        if in_channels == 0 {
            return Err(Error::InvalidArchitecture(
                "Number of input channels must be greater than 0".to_string(),
            ));
        }
        if embedding_dim == 0 {
            return Err(Error::InvalidArchitecture(
                "Embedding dimension must be greater than 0".to_string(),
            ));
        }

        // Check if image is divisible by patch size
        if image_size.0 % patch_size.0 != 0 || image_size.1 % patch_size.1 != 0 {
            return Err(Error::InvalidArchitecture(
                "Image dimensions must be divisible by patch dimensions".to_string(),
            ));
        }

        // Calculate number of patches
        let n_h = image_size.0 / patch_size.0;
        let n_w = image_size.1 / patch_size.1;
        let _num_patches = n_h * n_w;

        // Initialize weights and bias
        // Weight shape: [embedding_dim, in_channels * patch_size.0 * patch_size.1]
        let weight_shape = IxDyn(&[embedding_dim, in_channels * patch_size.0 * patch_size.1]);
        let weight = initializers::xavier_uniform::<F>(weight_shape.clone())?;
        let weight_grad = Arc::new(RwLock::new(Array::zeros(weight_shape)));

        // Bias and its gradient
        let (bias, bias_grad) = if use_bias {
            let bias = Some(Array::zeros(IxDyn(&[embedding_dim])));
            let bias_grad = Some(Arc::new(RwLock::new(Array::zeros(IxDyn(&[embedding_dim])))));
            (bias, bias_grad)
        } else {
            (None, None)
        };

        Ok(Self {
            image_size,
            patch_size,
            in_channels,
            embedding_dim,
            weight,
            bias,
            weight_grad,
            bias_grad,
            input_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Calculate the number of patches
    pub fn num_patches(&self) -> usize {
        let n_h = self.image_size.0 / self.patch_size.0;
        let n_w = self.image_size.1 / self.patch_size.1;
        n_h * n_w
    }

    /// Extract patches from input images
    fn extract_patches(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Validate input shape
        if input.ndim() != 4 {
            return Err(Error::InvalidArchitecture(
                "Input to PatchEmbedding must be 4D [batch_size, channels, height, width]"
                    .to_string(),
            ));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        if channels != self.in_channels {
            return Err(Error::InvalidArchitecture(format!(
                "Input has {} channels, but expected {}",
                channels, self.in_channels
            )));
        }

        if height != self.image_size.0 || width != self.image_size.1 {
            return Err(Error::InvalidArchitecture(format!(
                "Input has shape [{}x{}], but expected [{}x{}]",
                height, width, self.image_size.0, self.image_size.1
            )));
        }

        // Calculate patch grid dimensions
        let n_h = height / self.patch_size.0;
        let n_w = width / self.patch_size.1;
        let num_patches = n_h * n_w;

        // Extract patches and flatten them
        let patch_dim = channels * self.patch_size.0 * self.patch_size.1;
        let mut patches = Array::zeros(IxDyn(&[batch_size, num_patches, patch_dim]));

        for b in 0..batch_size {
            for i in 0..n_h {
                for j in 0..n_w {
                    let patch_idx = i * n_w + j;
                    let h_start = i * self.patch_size.0;
                    let w_start = j * self.patch_size.1;

                    // Flatten the patch
                    let mut flat_idx = 0;
                    for c in 0..channels {
                        for ph in 0..self.patch_size.0 {
                            for pw in 0..self.patch_size.1 {
                                let h_idx = h_start + ph;
                                let w_idx = w_start + pw;
                                patches[[b, patch_idx, flat_idx]] = input[[b, c, h_idx, w_idx]];
                                flat_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(patches)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for PatchEmbedding<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Extract patches
        let patches = self.extract_patches(input)?;

        // Cache input for backpropagation
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(patches.clone());
        } else {
            return Err(Error::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        let batch_size = patches.shape()[0];
        let num_patches = patches.shape()[1];

        // Linear projection of patches to embedding dimension
        let mut embeddings = Array::zeros(IxDyn(&[batch_size, num_patches, self.embedding_dim]));

        for b in 0..batch_size {
            for p in 0..num_patches {
                // Matrix multiplication for each patch
                for e in 0..self.embedding_dim {
                    let mut val = F::zero();
                    for i in 0..patches.shape()[2] {
                        val = val + self.weight[[e, i]] * patches[[b, p, i]];
                    }

                    // Add bias if present
                    if let Some(ref bias) = self.bias {
                        val = val + bias[[e]];
                    }

                    embeddings[[b, p, e]] = val;
                }
            }
        }

        Ok(embeddings)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Get cached input from RwLock
        let input_cache_guard = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(Error::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };

        if input_cache_guard.is_none() {
            return Err(Error::InferenceError(
                "Cannot perform backward pass before forward pass".to_string(),
            ));
        }

        let patches = input_cache_guard.as_ref().unwrap();
        let batch_size = patches.shape()[0];
        let num_patches = patches.shape()[1];
        let patch_dim = patches.shape()[2];

        // Validate grad_output shape
        if grad_output.shape() != &[batch_size, num_patches, self.embedding_dim] {
            return Err(Error::InvalidArchitecture(format!(
                "Expected grad_output shape [{}, {}, {}], but got {:?}",
                batch_size,
                num_patches,
                self.embedding_dim,
                grad_output.shape()
            )));
        }

        // Compute gradients with respect to weights and bias
        let mut weight_grad = Array::zeros(self.weight.dim());
        let mut bias_grad = if self.bias.is_some() {
            Some(Array::zeros(IxDyn(&[self.embedding_dim])))
        } else {
            None
        };

        for b in 0..batch_size {
            for p in 0..num_patches {
                for e in 0..self.embedding_dim {
                    let grad = grad_output[[b, p, e]];

                    // Gradient for bias
                    if let Some(ref mut bg) = bias_grad {
                        bg[[e]] = bg[[e]] + grad;
                    }

                    // Gradient for weights
                    for i in 0..patch_dim {
                        weight_grad[[e, i]] = weight_grad[[e, i]] + grad * patches[[b, p, i]];
                    }
                }
            }
        }

        // Update accumulated gradients
        if let Ok(mut weight_grad_guard) = self.weight_grad.write() {
            for e in 0..self.embedding_dim {
                for i in 0..patch_dim {
                    weight_grad_guard[[e, i]] = weight_grad_guard[[e, i]] + weight_grad[[e, i]];
                }
            }
        } else {
            return Err(Error::InferenceError(
                "Failed to acquire write lock on weight gradients".to_string(),
            ));
        }

        if let (Some(ref bg_acc_lock), Some(ref bg)) = (&self.bias_grad, &bias_grad) {
            if let Ok(mut bg_acc) = bg_acc_lock.write() {
                for e in 0..self.embedding_dim {
                    bg_acc[[e]] = bg_acc[[e]] + bg[[e]];
                }
            } else {
                return Err(Error::InferenceError(
                    "Failed to acquire write lock on bias gradients".to_string(),
                ));
            }
        }

        // Compute gradient with respect to input
        let mut input_grad = Array::zeros(IxDyn(&[
            batch_size,
            self.in_channels,
            self.image_size.0,
            self.image_size.1,
        ]));

        // Calculate gradient for each patch
        let mut patches_grad = Array::zeros(patches.dim());

        for b in 0..batch_size {
            for p in 0..num_patches {
                for i in 0..patch_dim {
                    let mut grad = F::zero();
                    for e in 0..self.embedding_dim {
                        grad = grad + grad_output[[b, p, e]] * self.weight[[e, i]];
                    }
                    patches_grad[[b, p, i]] = grad;
                }
            }
        }

        // Reshape patches gradient back to image space
        let n_h = self.image_size.0 / self.patch_size.0;
        let n_w = self.image_size.1 / self.patch_size.1;

        for b in 0..batch_size {
            for i in 0..n_h {
                for j in 0..n_w {
                    let patch_idx = i * n_w + j;
                    let h_start = i * self.patch_size.0;
                    let w_start = j * self.patch_size.1;

                    // Unflatten the patch gradient
                    let mut flat_idx = 0;
                    for c in 0..self.in_channels {
                        for ph in 0..self.patch_size.0 {
                            for pw in 0..self.patch_size.1 {
                                let h_idx = h_start + ph;
                                let w_idx = w_start + pw;
                                input_grad[[b, c, h_idx, w_idx]] =
                                    patches_grad[[b, patch_idx, flat_idx]];
                                flat_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(input_grad)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights
        let lr = learning_rate;
        let patch_dim = self.weight.shape()[1];

        // Get the weight gradients inside RwLock
        if let Ok(weight_grad_guard) = self.weight_grad.read() {
            for e in 0..self.embedding_dim {
                for i in 0..patch_dim {
                    self.weight[[e, i]] = self.weight[[e, i]] - lr * weight_grad_guard[[e, i]];
                }
            }
        } else {
            return Err(Error::InferenceError(
                "Failed to acquire read lock on weight gradients".to_string(),
            ));
        }

        // Update bias if present
        if let Some(ref mut bias) = &mut self.bias {
            if let Some(ref bias_grad_lock) = &self.bias_grad {
                if let Ok(bias_grad_guard) = bias_grad_lock.read() {
                    for e in 0..self.embedding_dim {
                        bias[[e]] = bias[[e]] - lr * bias_grad_guard[[e]];
                    }
                } else {
                    return Err(Error::InferenceError(
                        "Failed to acquire read lock on bias gradients".to_string(),
                    ));
                }
            }
        }

        // Reset gradients
        if let Ok(mut weight_grad_guard) = self.weight_grad.write() {
            weight_grad_guard.fill(F::zero());
        } else {
            return Err(Error::InferenceError(
                "Failed to acquire write lock on weight gradients".to_string(),
            ));
        }

        if let Some(ref bias_grad_lock) = &self.bias_grad {
            if let Ok(mut bias_grad_guard) = bias_grad_lock.write() {
                bias_grad_guard.fill(F::zero());
            } else {
                return Err(Error::InferenceError(
                    "Failed to acquire write lock on bias gradients".to_string(),
                ));
            }
        }

        // Clean input cache to free memory
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = None;
        } else {
            return Err(Error::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::Rng;

    #[test]
    fn test_embedding_creation() {
        // Create embedding layer
        let config = EmbeddingConfig {
            num_embeddings: 10,
            embedding_dim: 5,
            padding_idx: Some(0),
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };

        let embedding = Embedding::<f32>::new(config).unwrap();

        // Check dimensions
        assert_eq!(embedding.weight.shape(), &[10, 5]);

        // Check that padding index is zero
        for i in 0..5 {
            assert_eq!(embedding.weight[[0, i]], 0.0);
        }
    }

    #[test]
    fn test_embedding_forward() {
        // Create embedding layer
        let config = EmbeddingConfig {
            num_embeddings: 10,
            embedding_dim: 5,
            padding_idx: Some(0),
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };

        let mut embedding = Embedding::<f32>::new(config).unwrap();

        // Set weight values for testing
        for i in 0..10 {
            for j in 0..5 {
                embedding.weight[[i, j]] = (i * 10 + j) as f32 / 10.0;
            }
        }

        // Zero out padding index
        for j in 0..5 {
            embedding.weight[[0, j]] = 0.0;
        }

        // Create input indices
        let indices = Array2::from_shape_vec((2, 3), vec![1, 2, 0, 3, 0, 4]).unwrap();
        let indices_dyn = indices.into_dimensionality::<IxDyn>().unwrap();

        // Forward pass
        let output = embedding.forward_impl(&indices_dyn).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[2, 3, 5]);

        // Check values
        // First batch, first token (index 1)
        for j in 0..5 {
            assert_eq!(output[[0, 0, j]], (10 + j) as f32 / 10.0);
        }

        // First batch, second token (index 2)
        for j in 0..5 {
            assert_eq!(output[[0, 1, j]], (20 + j) as f32 / 10.0);
        }

        // First batch, third token (index 0, padding)
        for j in 0..5 {
            assert_eq!(output[[0, 2, j]], 0.0);
        }
    }

    #[test]
    fn test_positional_embedding() {
        // Test learned positional embeddings
        let pos_emb_learned = PositionalEmbedding::<f32>::new(10, 8, true).unwrap();

        // Check dimensions
        assert!(pos_emb_learned.weight.is_some());
        assert_eq!(pos_emb_learned.weight.as_ref().unwrap().shape(), &[10, 8]);

        // Create dummy input
        let input = Array::from_shape_fn(IxDyn(&[2, 5, 8]), |_| 1.0f32);

        // Forward pass
        let output = pos_emb_learned.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[2, 5, 8]);

        // Test fixed sinusoidal positional embeddings
        let pos_emb_fixed = PositionalEmbedding::<f32>::new(10, 8, false).unwrap();

        // Check that weight is None for fixed embeddings
        assert!(pos_emb_fixed.weight.is_none());

        // Forward pass
        let output = pos_emb_fixed.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_patch_embedding() {
        // Create patch embedding layer
        let patch_emb = PatchEmbedding::<f32>::new((32, 32), (8, 8), 3, 96, true).unwrap();

        // Check dimensions
        assert_eq!(patch_emb.weight.shape(), &[96, 3 * 8 * 8]);
        assert!(patch_emb.bias.is_some());
        assert_eq!(patch_emb.bias.as_ref().unwrap().shape(), &[96]);

        // Check number of patches
        assert_eq!(patch_emb.num_patches(), 16); // 4x4 patches of 8x8 in a 32x32 image

        // Create random input
        let mut rand_gen = rand::rng();
        let input = Array::from_shape_fn(IxDyn(&[2, 3, 32, 32]), |_| rand_gen.random::<f32>());

        // Forward pass
        let output = patch_emb.forward(&input).unwrap();

        // Check output shape [batch_size, num_patches, embedding_dim]
        assert_eq!(output.shape(), &[2, 16, 96]);
    }
}
