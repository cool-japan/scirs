//! Feature Fusion Model Architectures
//!
//! This module implements various feature fusion approaches for multi-modal learning,
//! allowing models to combine features from different modalities (e.g., vision, text, audio).

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm, Sequential};

use ndarray::{Array, Axis, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Fusion methods for multi-modal inputs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Concatenate features from different modalities
    Concatenation,
    /// Element-wise sum of features (requires same dimensions)
    Sum,
    /// Element-wise product of features (requires same dimensions)
    Product,
    /// Gated attention mechanism between modalities
    Attention,
    /// Bilinear fusion (outer product)
    Bilinear,
    /// FiLM conditioning (Feature-wise Linear Modulation)
    FiLM,
}

/// Configuration for the Feature Fusion model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFusionConfig {
    /// Dimensions of each input modality
    pub input_dims: Vec<usize>,
    /// Hidden dimension for alignment (if needed)
    pub hidden_dim: usize,
    /// Fusion method to use
    pub fusion_method: FusionMethod,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Number of output classes (if applicable)
    pub num_classes: usize,
    /// Whether to include the classifier head
    pub include_head: bool,
}

/// Feature alignment module
#[derive(Debug, Clone)]
pub struct FeatureAlignment<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension for alignment
    pub output_dim: usize,
    /// Linear projection layer
    pub projection: Dense<F>,
    /// Normalization layer
    pub norm: LayerNorm<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> FeatureAlignment<F> {
    /// Create a new FeatureAlignment module
    pub fn new(input_dim: usize, output_dim: usize, _name: Option<&str>) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);
        let projection = Dense::<F>::new(input_dim, output_dim, None, &mut rng)?;

        let mut rng = SmallRng::seed_from_u64(42);
        let norm = LayerNorm::<F>::new(output_dim, 1e-6, &mut rng)?;

        Ok(Self {
            input_dim,
            output_dim,
            projection,
            norm,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for FeatureAlignment<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let x = self.projection.forward(input)?;
        let x = self.norm.forward(&x)?;
        Ok(x)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.projection.params());
        params.extend(self.norm.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.projection.set_training(training);
        self.norm.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.projection.is_training()
    }
}

/// Cross-Modal Attention module
#[derive(Debug, Clone)]
pub struct CrossModalAttention<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Query projection
    pub query_proj: Dense<F>,
    /// Key projection
    pub key_proj: Dense<F>,
    /// Value projection
    pub value_proj: Dense<F>,
    /// Output projection
    pub output_proj: Dense<F>,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Scale factor for attention
    pub scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> CrossModalAttention<F> {
    /// Create a new CrossModalAttention module
    pub fn new(query_dim: usize, key_dim: usize, hidden_dim: usize) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);

        let query_proj = Dense::<F>::new(query_dim, hidden_dim, None, &mut rng)?;

        let key_proj = Dense::<F>::new(key_dim, hidden_dim, None, &mut rng)?;

        let value_proj = Dense::<F>::new(key_dim, hidden_dim, None, &mut rng)?;

        let output_proj = Dense::<F>::new(hidden_dim, query_dim, None, &mut rng)?;

        // Scale factor for dot product attention
        let scale = F::from(1.0 / (hidden_dim as f64).sqrt()).unwrap();

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            hidden_dim,
            scale,
        })
    }

    /// Forward pass for cross-modal attention
    pub fn forward(
        &self,
        query: &Array<F, IxDyn>,
        context: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Project query, key, and value
        let q = self.query_proj.forward(query)?;
        let k = self.key_proj.forward(context)?;
        let v = self.value_proj.forward(context)?;

        // Reshape for easier computation
        let batch_size = q.shape()[0];
        let query_len = q.shape()[1];
        let context_len = k.shape()[1];

        let q_2d = q
            .clone()
            .into_shape_with_order((batch_size * query_len, self.hidden_dim))?;
        let k_2d = k
            .clone()
            .into_shape_with_order((batch_size * context_len, self.hidden_dim))?;
        let v_2d = v
            .clone()
            .into_shape_with_order((batch_size * context_len, self.hidden_dim))?;

        // Compute attention scores
        let scores = q_2d.dot(&k_2d.t()) * self.scale;

        // Reshape scores to (batch_size, query_len, context_len)
        let scores_3d = scores.into_shape_with_order((batch_size, query_len, context_len))?;

        // Apply softmax along the context dimension
        let mut attention_weights = Array::<F, _>::zeros(scores_3d.raw_dim());

        for b in 0..batch_size {
            for q in 0..query_len {
                let mut row = scores_3d.slice(ndarray::s![b, q, ..]).to_owned();

                // Find max for numerical stability
                let max_val = row.fold(F::neg_infinity(), |m, &v| m.max(v));

                // Compute exp and sum
                let mut exp_sum = F::zero();
                for i in 0..context_len {
                    let exp_val = (row[i] - max_val).exp();
                    row[i] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }

                // Normalize
                if exp_sum > F::zero() {
                    for i in 0..context_len {
                        row[i] = row[i] / exp_sum;
                    }
                }

                // Copy normalized weights
                for i in 0..context_len {
                    attention_weights[[b, q, i]] = row[i];
                }
            }
        }

        // Reshape attention weights for matrix multiplication
        let attn_weights_2d = attention_weights
            .into_shape_with_order((batch_size * query_len, batch_size * context_len))?;

        // Apply attention weights to values
        let context_vec = attn_weights_2d.dot(&v_2d);

        // Reshape and project output
        let context_vec_reshaped =
            context_vec.into_shape_with_order((batch_size, query_len, self.hidden_dim))?;

        // Final projection
        let output = self.output_proj.forward(&context_vec_reshaped.into_dyn())?;

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for CrossModalAttention<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This assumes the input contains both query and context packed together
        // In practical use, use the dedicated forward method with separate inputs
        Err(NeuralError::ValidationError("CrossModalAttention requires separate query and context inputs. Use the dedicated forward method.".to_string()))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.query_proj.params());
        params.extend(self.key_proj.params());
        params.extend(self.value_proj.params());
        params.extend(self.output_proj.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.query_proj.set_training(training);
        self.key_proj.set_training(training);
        self.value_proj.set_training(training);
        self.output_proj.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.query_proj.is_training()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// FiLM (Feature-wise Linear Modulation) conditioning module
#[derive(Debug, Clone)]
pub struct FiLMModule<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Feature dimension to be modulated
    pub feature_dim: usize,
    /// Conditioning input dimension
    pub cond_dim: usize,
    /// Gamma (scale) projection
    pub gamma_proj: Dense<F>,
    /// Beta (shift) projection
    pub beta_proj: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> FiLMModule<F> {
    /// Create a new FiLMModule
    pub fn new(feature_dim: usize, cond_dim: usize) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);
        let gamma_proj = Dense::<F>::new(cond_dim, feature_dim, None, &mut rng)?;

        let beta_proj = Dense::<F>::new(cond_dim, feature_dim, None, &mut rng)?;

        Ok(Self {
            feature_dim,
            cond_dim,
            gamma_proj,
            beta_proj,
        })
    }

    /// Forward pass with separate feature and conditioning inputs
    pub fn forward(
        &self,
        features: &Array<F, IxDyn>,
        conditioning: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Generate gamma and beta for modulation
        let gamma = self.gamma_proj.forward(conditioning)?;
        let beta = self.beta_proj.forward(conditioning)?;

        // Apply FiLM: gamma * features + beta
        let modulated = &gamma * features + &beta;

        Ok(modulated)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for FiLMModule<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This assumes the input contains both features and conditioning packed together
        // In practical use, use the dedicated forward method with separate inputs
        Err(NeuralError::ValidationError("FiLMModule requires separate feature and conditioning inputs. Use the dedicated forward method.".to_string()))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.gamma_proj.params());
        params.extend(self.beta_proj.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.gamma_proj.set_training(training);
        self.beta_proj.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.gamma_proj.is_training()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Bilinear Fusion module for pairwise interactions between modalities
#[derive(Debug, Clone)]
pub struct BilinearFusion<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// First modality dimension
    pub dim_a: usize,
    /// Second modality dimension
    pub dim_b: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Projection from A
    pub proj_a: Dense<F>,
    /// Projection from B
    pub proj_b: Dense<F>,
    /// Low-rank projection to output
    pub low_rank_proj: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> BilinearFusion<F> {
    /// Create a new BilinearFusion module
    pub fn new(dim_a: usize, dim_b: usize, output_dim: usize, rank: usize) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);
        let proj_a = Dense::<F>::new(dim_a, rank, None, &mut rng)?;

        let proj_b = Dense::<F>::new(dim_b, rank, None, &mut rng)?;

        let low_rank_proj = Dense::<F>::new(rank, output_dim, None, &mut rng)?;

        Ok(Self {
            dim_a,
            dim_b,
            output_dim,
            proj_a,
            proj_b,
            low_rank_proj,
        })
    }

    /// Forward pass with separate modality inputs
    pub fn forward(
        &self,
        features_a: &Array<F, IxDyn>,
        features_b: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Project inputs to a common low-rank space
        let a_proj = self.proj_a.forward(features_a)?;
        let b_proj = self.proj_b.forward(features_b)?;

        // Element-wise product for bilinear interaction
        let bilinear = &a_proj * &b_proj;

        // Final projection
        let output = self.low_rank_proj.forward(&bilinear)?;

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for BilinearFusion<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This assumes the input contains both feature sets packed together
        // In practical use, use the dedicated forward method with separate inputs
        Err(NeuralError::ValidationError(
            "BilinearFusion requires separate feature inputs. Use the dedicated forward method."
                .to_string(),
        ))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.proj_a.params());
        params.extend(self.proj_b.params());
        params.extend(self.low_rank_proj.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.proj_a.set_training(training);
        self.proj_b.set_training(training);
        self.low_rank_proj.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.proj_a.is_training()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Feature Fusion model
pub struct FeatureFusion<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Feature aligners for each input modality
    pub aligners: Vec<FeatureAlignment<F>>,
    /// Fusion-specific modules
    pub fusion_module: Option<Box<dyn Layer<F> + Send + Sync>>,
    /// Post-fusion MLP
    pub post_fusion: Sequential<F>,
    /// Classifier head
    pub classifier: Option<Dense<F>>,
    /// Model configuration
    pub config: FeatureFusionConfig,
}

// Manual implementation of Debug for FeatureFusion to handle dyn Layer trait objects
impl<F: Float + Debug + ScalarOperand + Send + Sync> Debug for FeatureFusion<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureFusion")
            .field("aligners", &self.aligners)
            .field(
                "fusion_module",
                &format!("<Box<dyn Layer<F> + Send + Sync>>"),
            )
            .field("post_fusion", &self.post_fusion)
            .field("classifier", &self.classifier)
            .field("config", &self.config)
            .finish()
    }
}

// Manual implementation of Clone for FeatureFusion
impl<F: Float + Debug + ScalarOperand + Send + Sync> Clone for FeatureFusion<F> {
    fn clone(&self) -> Self {
        // We can't clone the dyn Layer directly, so we create a new FeatureFusion
        // without the fusion_module
        let new_fusion = Self {
            aligners: self.aligners.clone(),
            fusion_module: None, // Can't clone the trait object
            post_fusion: self.post_fusion.clone(),
            classifier: self.classifier.clone(),
            config: self.config.clone(),
        };

        // We would need to implement custom clone logic for fusion_module
        // based on its actual type if needed, but for now we leave it as None

        new_fusion
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> FeatureFusion<F> {
    /// Create a new FeatureFusion model
    pub fn new(config: FeatureFusionConfig) -> Result<Self> {
        // Create feature aligners
        let mut aligners = Vec::with_capacity(config.input_dims.len());

        for (i, &dim) in config.input_dims.iter().enumerate() {
            aligners.push(FeatureAlignment::<F>::new(
                dim,
                config.hidden_dim,
                Some(&format!("aligner_{}", i)),
            )?);
        }

        // Create fusion-specific module based on method
        let fusion_module: Option<Box<dyn Layer<F> + Send + Sync>> = match config.fusion_method {
            FusionMethod::Attention => {
                if config.input_dims.len() < 2 {
                    return Err(NeuralError::ValidationError(
                        "Attention fusion requires at least two modalities".to_string(),
                    ));
                }

                let attn = CrossModalAttention::<F>::new(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.hidden_dim,
                )?;

                Some(Box::new(attn))
            }
            FusionMethod::Bilinear => {
                if config.input_dims.len() != 2 {
                    return Err(NeuralError::ValidationError(
                        "Bilinear fusion requires exactly two modalities".to_string(),
                    ));
                }

                let bilinear = BilinearFusion::<F>::new(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.hidden_dim,
                    config.hidden_dim / 4, // Low-rank approximation
                )?;

                Some(Box::new(bilinear))
            }
            FusionMethod::FiLM => {
                if config.input_dims.len() != 2 {
                    return Err(NeuralError::ValidationError(
                        "FiLM fusion requires exactly two modalities".to_string(),
                    ));
                }

                let film = FiLMModule::<F>::new(config.hidden_dim, config.hidden_dim)?;

                Some(Box::new(film))
            }
            // For simpler methods (concat, sum, product), we don't need special modules
            _ => None,
        };

        // Create post-fusion MLP
        let mut post_fusion = Sequential::new();

        // Determine input dimension for the post-fusion network
        let post_fusion_input_dim = match config.fusion_method {
            FusionMethod::Concatenation => config.hidden_dim * config.input_dims.len(),
            _ => config.hidden_dim,
        };

        let mut rng = SmallRng::seed_from_u64(42);
        post_fusion.add(Dense::<F>::new(
            post_fusion_input_dim,
            config.hidden_dim * 2,
            Some("gelu"),
            &mut rng,
        )?);

        if config.dropout_rate > 0.0 {
            let mut rng = SmallRng::seed_from_u64(42);
            post_fusion.add(Dropout::<F>::new(config.dropout_rate, &mut rng)?);
        }

        let mut rng = SmallRng::seed_from_u64(42);
        post_fusion.add(Dense::<F>::new(
            config.hidden_dim * 2,
            config.hidden_dim,
            Some("gelu"),
            &mut rng,
        )?);

        // Create classifier if needed
        let classifier = if config.include_head {
            let mut rng = SmallRng::seed_from_u64(42);
            Some(Dense::<F>::new(
                config.hidden_dim,
                config.num_classes,
                None,
                &mut rng,
            )?)
        } else {
            None
        };

        Ok(Self {
            aligners,
            fusion_module,
            post_fusion,
            classifier,
            config,
        })
    }

    /// Forward pass with multiple input modalities
    pub fn forward_multi(&self, inputs: &[Array<F, IxDyn>]) -> Result<Array<F, IxDyn>> {
        if inputs.len() != self.config.input_dims.len() {
            return Err(NeuralError::ValidationError(format!(
                "Expected {} inputs, got {}",
                self.config.input_dims.len(),
                inputs.len()
            )));
        }

        // Align features from each modality
        let mut aligned_features = Vec::with_capacity(inputs.len());

        for (i, input) in inputs.iter().enumerate() {
            aligned_features.push(self.aligners[i].forward(input)?);
        }

        // Apply fusion based on method
        let fused = match self.config.fusion_method {
            FusionMethod::Concatenation => {
                // Concatenate along feature dimension
                let batch_size = aligned_features[0].shape()[0];
                let mut concatenated = Vec::new();

                for batch_idx in 0..batch_size {
                    for features in &aligned_features {
                        let batch_features = features
                            .slice_axis(Axis(0), ndarray::Slice::from(batch_idx..batch_idx + 1));
                        concatenated.extend(batch_features.iter().cloned());
                    }
                }

                Array::from_shape_vec(
                    [batch_size, self.config.hidden_dim * aligned_features.len()],
                    concatenated,
                )?
                .into_dyn()
            }
            FusionMethod::Sum => {
                // Element-wise sum
                let mut result = aligned_features[0].clone();

                for features in &aligned_features[1..] {
                    result = result + features;
                }

                result
            }
            FusionMethod::Product => {
                // Element-wise product
                let mut result = aligned_features[0].clone();

                for features in &aligned_features[1..] {
                    result = result * features;
                }

                result
            }
            FusionMethod::Attention => {
                // Use attention module (modality 0 attends to modality 1)
                if let Some(ref module) = self.fusion_module {
                    // We need to cast the module as CrossModalAttention
                    if let Some(attn) = module.as_any().downcast_ref::<CrossModalAttention<F>>() {
                        attn.forward(&aligned_features[0], &aligned_features[1])?
                    } else {
                        return Err(NeuralError::InferenceError(
                            "Failed to cast fusion module to CrossModalAttention".to_string(),
                        ));
                    }
                } else {
                    return Err(NeuralError::InferenceError(
                        "Attention fusion module not initialized".to_string(),
                    ));
                }
            }
            FusionMethod::Bilinear => {
                // Use bilinear module
                if let Some(ref module) = self.fusion_module {
                    // We need to cast the module as BilinearFusion
                    if let Some(bilinear) = module.as_any().downcast_ref::<BilinearFusion<F>>() {
                        bilinear.forward(&aligned_features[0], &aligned_features[1])?
                    } else {
                        return Err(NeuralError::InferenceError(
                            "Failed to cast fusion module to BilinearFusion".to_string(),
                        ));
                    }
                } else {
                    return Err(NeuralError::InferenceError(
                        "Bilinear fusion module not initialized".to_string(),
                    ));
                }
            }
            FusionMethod::FiLM => {
                // Use FiLM module (modality 1 conditions modality 0)
                if let Some(ref module) = self.fusion_module {
                    // We need to cast the module as FiLMModule
                    if let Some(film) = module.as_any().downcast_ref::<FiLMModule<F>>() {
                        film.forward(&aligned_features[0], &aligned_features[1])?
                    } else {
                        return Err(NeuralError::InferenceError(
                            "Failed to cast fusion module to FiLMModule".to_string(),
                        ));
                    }
                } else {
                    return Err(NeuralError::InferenceError(
                        "FiLM fusion module not initialized".to_string(),
                    ));
                }
            }
        };

        // Apply post-fusion network
        let features = self.post_fusion.forward(&fused)?;

        // Apply classifier if available
        if let Some(ref classifier) = self.classifier {
            classifier.forward(&features)
        } else {
            Ok(features)
        }
    }

    /// Create a simple early fusion model for two modalities
    pub fn create_early_fusion(
        dim_a: usize,
        dim_b: usize,
        hidden_dim: usize,
        num_classes: usize,
        include_head: bool,
    ) -> Result<Self> {
        let config = FeatureFusionConfig {
            input_dims: vec![dim_a, dim_b],
            hidden_dim,
            fusion_method: FusionMethod::Concatenation,
            dropout_rate: 0.1,
            num_classes,
            include_head,
        };

        Self::new(config)
    }

    /// Create an attention-based fusion model for two modalities
    pub fn create_attention_fusion(
        dim_a: usize,
        dim_b: usize,
        hidden_dim: usize,
        num_classes: usize,
        include_head: bool,
    ) -> Result<Self> {
        let config = FeatureFusionConfig {
            input_dims: vec![dim_a, dim_b],
            hidden_dim,
            fusion_method: FusionMethod::Attention,
            dropout_rate: 0.1,
            num_classes,
            include_head,
        };

        Self::new(config)
    }

    /// Create a FiLM conditioning fusion model (B conditions A)
    pub fn create_film_fusion(
        dim_a: usize,
        dim_b: usize,
        hidden_dim: usize,
        num_classes: usize,
        include_head: bool,
    ) -> Result<Self> {
        let config = FeatureFusionConfig {
            input_dims: vec![dim_a, dim_b],
            hidden_dim,
            fusion_method: FusionMethod::FiLM,
            dropout_rate: 0.1,
            num_classes,
            include_head,
        };

        Self::new(config)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for FeatureFusion<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // For a single packed input, we need to split it into modalities
        // This is mainly for the Layer trait compatibility
        // In practice, use forward_multi with separate inputs
        Err(NeuralError::ValidationError(
            "FeatureFusion requires multiple inputs. Use forward_multi method instead.".to_string(),
        ))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Fusion operation not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();

        for aligner in &self.aligners {
            params.extend(aligner.params());
        }

        if let Some(ref module) = self.fusion_module {
            params.extend(module.params());
        }

        params.extend(self.post_fusion.params());

        if let Some(ref classifier) = self.classifier {
            params.extend(classifier.params());
        }

        params
    }

    fn set_training(&mut self, training: bool) {
        for aligner in &mut self.aligners {
            aligner.set_training(training);
        }

        if let Some(ref mut module) = self.fusion_module {
            module.set_training(training);
        }

        self.post_fusion.set_training(training);

        if let Some(ref mut classifier) = self.classifier {
            classifier.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.aligners[0].is_training()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
