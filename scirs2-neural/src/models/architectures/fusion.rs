//! Feature Fusion Model Architectures
//!
//! This module implements various feature fusion approaches for multi-modal learning,
//! allowing models to combine features from different modalities (e.g., vision, text, audio).

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm, Sequential};
use scirs2_core::ndarray::{Array, Axis, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::SeedableRng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
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
pub struct FeatureAlignment<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign>
where
    F: SimdUnifiedOps,
{
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension for alignment
    pub output_dim: usize,
    /// Linear projection layer
    pub projection: Dense<F>,
    /// Normalization layer
    pub norm: LayerNorm<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> FeatureAlignment<F>
where
    F: SimdUnifiedOps,
{
    /// Create a new FeatureAlignment module
    pub fn new(input_dim: usize, output_dim: usize, _name: Option<&str>) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let projection = Dense::<F>::new(input_dim, output_dim, None, &mut rng)?;
        let norm = LayerNorm::<F>::new(output_dim, 1e-6, &mut rng)?;
        Ok(Self {
            input_dim,
            output_dim,
            projection,
            norm,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> FeatureAlignment<F>
where
    F: SimdUnifiedOps,
{
    /// Reshape `input`'s leading dimensions into a single "batch" axis so
    /// the (strictly 2D) [`Dense`] projection can be applied to inputs with
    /// any number of leading dimensions (e.g. `[batch, seq, features]` for
    /// sequence modalities, not just `[batch, features]`), returning
    /// `(flattened_2d, original_shape, outer_size)`.
    fn flatten_leading(
        &self,
        input: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Vec<usize>, usize)> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        if ndim < 1 {
            return Err(NeuralError::InvalidArchitecture(
                "FeatureAlignment requires at least a 1D input".to_string(),
            ));
        }
        let last = shape[ndim - 1];
        if last != self.input_dim {
            return Err(NeuralError::InvalidArchitecture(format!(
                "FeatureAlignment: expected last dimension {}, got {}",
                self.input_dim, last
            )));
        }
        let outer: usize = shape[..ndim - 1].iter().product();
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[outer, self.input_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?;
        Ok((flat, shape, outer))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Layer<F> for FeatureAlignment<F>
where
    F: SimdUnifiedOps,
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let (flat_in, mut outshape, _outer) = self.flatten_leading(input)?;
        let projected = self.projection.forward(&flat_in)?;
        outshape.pop();
        outshape.push(self.output_dim);
        let projected = projected
            .into_shape_with_order(IxDyn(&outshape))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {e}")))?;
        let x = self.norm.forward(&projected)?;
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
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Backward pass through the alignment layer (Dense -> LayerNorm),
        // supporting the same arbitrary leading dimensions as `forward`.
        let (flat_in, inshape, outer) = self.flatten_leading(input)?;
        let mut projshape = inshape[..inshape.len() - 1].to_vec();
        projshape.push(self.output_dim);

        // Recompute the intermediate (pre-LayerNorm) projection output.
        let proj_output_flat = self.projection.forward(&flat_in)?;
        let proj_output = proj_output_flat
            .into_shape_with_order(IxDyn(&projshape))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape: {e}")))?;

        // Backward through LayerNorm, then flatten again for Dense.
        let grad_proj = self.norm.backward(&proj_output, grad_output)?;
        let grad_proj_flat = grad_proj
            .into_shape_with_order(IxDyn(&[outer, self.output_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape gradient: {e}")))?;

        // Backward through the Dense projection, then restore the original shape.
        let grad_input_flat = self.projection.backward(&flat_in, &grad_proj_flat)?;
        grad_input_flat
            .into_shape_with_order(IxDyn(&inshape))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape input gradient: {e}"))
            })
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update the Dense projection layer
        self.projection.update(learning_rate)?;
        // Update the LayerNorm layer
        self.norm.update(learning_rate)?;
        Ok(())
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
#[derive(Debug)]
pub struct CrossModalAttention<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
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
    /// Forward-pass cache required by [`CrossModalAttention::backward_with_context`]
    cache: Arc<RwLock<Option<CrossModalCache<F>>>>,
}

/// Intermediates of one [`CrossModalAttention::forward`] call
#[derive(Debug, Clone)]
struct CrossModalCache<F> {
    /// Batch size
    batch: usize,
    /// Query sequence length
    query_len: usize,
    /// Context sequence length
    context_len: usize,
    /// Query source flattened to `[batch * query_len, query_dim]`
    query_flat: Array<F, IxDyn>,
    /// Context source flattened to `[batch * context_len, key_dim]`
    context_flat: Array<F, IxDyn>,
    /// Projected queries `[batch * query_len, hidden_dim]`
    q: Array<F, IxDyn>,
    /// Projected keys `[batch * context_len, hidden_dim]`
    k: Array<F, IxDyn>,
    /// Projected values `[batch * context_len, hidden_dim]`
    v: Array<F, IxDyn>,
    /// Attention weights `[batch, query_len, context_len]`
    attention: Array<F, IxDyn>,
    /// Attention output before the output projection
    context_vec: Array<F, IxDyn>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Clone for CrossModalAttention<F> {
    fn clone(&self) -> Self {
        Self {
            query_proj: self.query_proj.clone(),
            key_proj: self.key_proj.clone(),
            value_proj: self.value_proj.clone(),
            output_proj: self.output_proj.clone(),
            hidden_dim: self.hidden_dim,
            scale: self.scale,
            cache: Arc::new(RwLock::new(match self.cache.read() {
                Ok(guard) => guard.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            })),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> CrossModalAttention<F> {
    /// Create a new CrossModalAttention module
    pub fn new(query_dim: usize, key_dim: usize, hidden_dim: usize) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let query_proj = Dense::<F>::new(query_dim, hidden_dim, None, &mut rng)?;
        let key_proj = Dense::<F>::new(key_dim, hidden_dim, None, &mut rng)?;
        let value_proj = Dense::<F>::new(key_dim, hidden_dim, None, &mut rng)?;
        let output_proj = Dense::<F>::new(hidden_dim, query_dim, None, &mut rng)?;
        // Scale factor for dot product attention
        let scale = F::from(1.0 / (hidden_dim as f64).sqrt()).expect("Operation failed");
        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            hidden_dim,
            scale,
            cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Forward pass for cross-modal attention
    ///
    /// # Arguments
    /// * `query` - Query modality `[batch, query_len, query_dim]`
    /// * `context` - Context modality `[batch, context_len, key_dim]`
    ///
    /// # Returns
    /// Attended query representation `[batch, query_len, query_dim]`
    ///
    /// Attention is computed independently per batch element: a query never
    /// attends to another sample's context.
    pub fn forward(
        &self,
        query: &Array<F, IxDyn>,
        context: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        if query.ndim() != 3 || context.ndim() != 3 {
            return Err(NeuralError::ValidationError(format!(
                "CrossModalAttention expects 3D tensors [batch, seq, features], got {}D and {}D",
                query.ndim(),
                context.ndim()
            )));
        }
        let batch = query.shape()[0];
        if context.shape()[0] != batch {
            return Err(NeuralError::ShapeMismatch(format!(
                "Batch size mismatch between query ({}) and context ({})",
                batch,
                context.shape()[0]
            )));
        }
        let query_len = query.shape()[1];
        let context_len = context.shape()[1];
        let query_dim = query.shape()[2];
        let key_dim = context.shape()[2];

        // Dense layers operate on 2D `[rows, features]`, so flatten the batch
        // and sequence axes before projecting.
        let query_flat = query
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch * query_len, query_dim]))?;
        let context_flat = context
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch * context_len, key_dim]))?;

        let q = self.query_proj.forward(&query_flat)?;
        let k = self.key_proj.forward(&context_flat)?;
        let v = self.value_proj.forward(&context_flat)?;

        // Scaled dot-product attention, per batch element.
        let mut attention = Array::<F, IxDyn>::zeros(IxDyn(&[batch, query_len, context_len]));
        for b in 0..batch {
            for i in 0..query_len {
                let qi = b * query_len + i;
                let mut scores = vec![F::zero(); context_len];
                let mut max_val = F::neg_infinity();
                for (j, score) in scores.iter_mut().enumerate() {
                    let kj = b * context_len + j;
                    let mut dot = F::zero();
                    for h in 0..self.hidden_dim {
                        dot += q[[qi, h]] * k[[kj, h]];
                    }
                    *score = dot * self.scale;
                    if *score > max_val {
                        max_val = *score;
                    }
                }
                let mut exp_sum = F::zero();
                for score in scores.iter_mut() {
                    *score = (*score - max_val).exp();
                    exp_sum += *score;
                }
                if exp_sum > F::zero() {
                    for (j, score) in scores.iter().enumerate() {
                        attention[[b, i, j]] = *score / exp_sum;
                    }
                }
            }
        }

        // Weighted sum of the values.
        let mut context_vec =
            Array::<F, IxDyn>::zeros(IxDyn(&[batch * query_len, self.hidden_dim]));
        for b in 0..batch {
            for i in 0..query_len {
                let qi = b * query_len + i;
                for h in 0..self.hidden_dim {
                    let mut sum = F::zero();
                    for j in 0..context_len {
                        sum += attention[[b, i, j]] * v[[b * context_len + j, h]];
                    }
                    context_vec[[qi, h]] = sum;
                }
            }
        }

        let projected = self.output_proj.forward(&context_vec)?;

        if let Ok(mut guard) = self.cache.write() {
            *guard = Some(CrossModalCache {
                batch,
                query_len,
                context_len,
                query_flat,
                context_flat,
                q,
                k,
                v,
                attention,
                context_vec,
            });
        }

        Ok(projected.into_shape_with_order(IxDyn(&[batch, query_len, query_dim]))?)
    }

    /// Exact gradient of [`CrossModalAttention::forward`].
    ///
    /// # Arguments
    /// * `grad_output` - Gradient with respect to the attended output
    ///
    /// # Returns
    /// `(grad_query, grad_context)` with the shapes of the two forward inputs.
    /// The projection layers additionally record their own weight gradients, so
    /// a subsequent [`Layer::update`] trains them.
    pub fn backward_with_context(
        &self,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let cache = self
            .cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on the attention cache".to_string(),
                )
            })?
            .clone()
            .ok_or_else(|| {
                NeuralError::InferenceError(
                    "No cached values for backward pass. Call forward() first.".to_string(),
                )
            })?;

        let (batch, query_len, context_len) = (cache.batch, cache.query_len, cache.context_len);
        let query_dim = cache.query_flat.shape()[1];
        let key_dim = cache.context_flat.shape()[1];
        if grad_output.shape() != [batch, query_len, query_dim] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected an output gradient of shape [{batch}, {query_len}, {query_dim}], got {:?}",
                grad_output.shape()
            )));
        }

        // Output projection.
        let grad_flat = grad_output
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch * query_len, query_dim]))?;
        let d_context_vec = self.output_proj.backward(&cache.context_vec, &grad_flat)?;

        // Attention weights and values.
        let mut d_attention = Array::<F, IxDyn>::zeros(IxDyn(&[batch, query_len, context_len]));
        let mut d_v = Array::<F, IxDyn>::zeros(cache.v.dim());
        for b in 0..batch {
            for i in 0..query_len {
                let qi = b * query_len + i;
                for j in 0..context_len {
                    let kj = b * context_len + j;
                    let a = cache.attention[[b, i, j]];
                    let mut sum = F::zero();
                    for h in 0..self.hidden_dim {
                        let g = d_context_vec[[qi, h]];
                        sum += g * cache.v[[kj, h]];
                        d_v[[kj, h]] += a * g;
                    }
                    d_attention[[b, i, j]] = sum;
                }
            }
        }

        // Softmax Jacobian, then the scaled dot product.
        let mut d_q = Array::<F, IxDyn>::zeros(cache.q.dim());
        let mut d_k = Array::<F, IxDyn>::zeros(cache.k.dim());
        for b in 0..batch {
            for i in 0..query_len {
                let qi = b * query_len + i;
                let mut dot = F::zero();
                for j in 0..context_len {
                    dot += cache.attention[[b, i, j]] * d_attention[[b, i, j]];
                }
                for j in 0..context_len {
                    let kj = b * context_len + j;
                    let d_score =
                        cache.attention[[b, i, j]] * (d_attention[[b, i, j]] - dot) * self.scale;
                    if d_score == F::zero() {
                        continue;
                    }
                    for h in 0..self.hidden_dim {
                        d_q[[qi, h]] += d_score * cache.k[[kj, h]];
                        d_k[[kj, h]] += d_score * cache.q[[qi, h]];
                    }
                }
            }
        }

        let grad_query_flat = self.query_proj.backward(&cache.query_flat, &d_q)?;
        let grad_context_k = self.key_proj.backward(&cache.context_flat, &d_k)?;
        let grad_context_v = self.value_proj.backward(&cache.context_flat, &d_v)?;
        let grad_context_flat = grad_context_k + grad_context_v;

        Ok((
            grad_query_flat.into_shape_with_order(IxDyn(&[batch, query_len, query_dim]))?,
            grad_context_flat.into_shape_with_order(IxDyn(&[batch, context_len, key_dim]))?,
        ))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Layer<F>
    for CrossModalAttention<F>
{
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This assumes the input contains both query and context packed together
        // In practical use, use the dedicated forward method with separate inputs
        Err(NeuralError::ValidationError(
            "CrossModalAttention requires separate query and context inputs. Use the dedicated forward method."
                .to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Gradient with respect to the query modality.
    ///
    /// The single-tensor `Layer` interface cannot express the context-side
    /// gradient, so this delegates to
    /// [`CrossModalAttention::backward_with_context`] and returns only the query
    /// half; use that method directly when both gradients are needed.
    ///
    /// # Errors
    /// Reports [`NeuralError::InferenceError`] when no cached forward pass
    /// exists, because [`Layer::forward`] alone cannot produce one.
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(self.backward_with_context(grad_output)?.0)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all projection layers
        self.query_proj.update(learning_rate)?;
        self.key_proj.update(learning_rate)?;
        self.value_proj.update(learning_rate)?;
        self.output_proj.update(learning_rate)?;
        Ok(())
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
}

/// FiLM (Feature-wise Linear Modulation) conditioning module
#[derive(Debug)]
pub struct FiLMModule<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Feature dimension to be modulated
    pub feature_dim: usize,
    /// Conditioning input dimension
    pub cond_dim: usize,
    /// Gamma (scale) projection
    pub gamma_proj: Dense<F>,
    /// Beta (shift) projection
    pub beta_proj: Dense<F>,
    /// `(features, conditioning, gamma)` recorded by `forward` for the
    /// backward pass
    #[allow(clippy::type_complexity)]
    cache: Arc<RwLock<Option<(Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>)>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Clone for FiLMModule<F> {
    fn clone(&self) -> Self {
        Self {
            feature_dim: self.feature_dim,
            cond_dim: self.cond_dim,
            gamma_proj: self.gamma_proj.clone(),
            beta_proj: self.beta_proj.clone(),
            cache: Arc::new(RwLock::new(match self.cache.read() {
                Ok(guard) => guard.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            })),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> FiLMModule<F> {
    /// Create a new FiLMModule
    pub fn new(feature_dim: usize, cond_dim: usize) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let gamma_proj = Dense::<F>::new(cond_dim, feature_dim, None, &mut rng)?;
        let beta_proj = Dense::<F>::new(cond_dim, feature_dim, None, &mut rng)?;
        Ok(Self {
            feature_dim,
            cond_dim,
            gamma_proj,
            beta_proj,
            cache: Arc::new(RwLock::new(None)),
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
        if let Ok(mut guard) = self.cache.write() {
            *guard = Some((features.to_owned(), conditioning.to_owned(), gamma));
        }
        Ok(modulated)
    }

    /// Exact gradient of `gamma(c) * x + beta(c)`.
    ///
    /// # Returns
    /// `(grad_features, grad_conditioning)`. The two projection layers also
    /// record their own weight gradients for a subsequent [`Layer::update`].
    pub fn backward_with_conditioning(
        &self,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let (features, conditioning, gamma) = self
            .cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on the cache".to_string())
            })?
            .clone()
            .ok_or_else(|| {
                NeuralError::InferenceError(
                    "No cached values for backward pass. Call forward() first.".to_string(),
                )
            })?;

        if grad_output.shape() != features.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Output gradient shape {:?} must match the feature shape {:?}",
                grad_output.shape(),
                features.shape()
            )));
        }

        let grad_features = grad_output * &gamma;
        let grad_gamma = grad_output * &features;
        // beta enters additively, so its gradient is the output gradient.
        let grad_cond_from_gamma = self.gamma_proj.backward(&conditioning, &grad_gamma)?;
        let grad_cond_from_beta = self.beta_proj.backward(&conditioning, grad_output)?;
        Ok((grad_features, grad_cond_from_gamma + grad_cond_from_beta))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Layer<F> for FiLMModule<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This assumes the input contains both features and conditioning packed together
        Err(NeuralError::ValidationError(
            "FiLMModule requires separate feature and conditioning inputs. Use the dedicated forward method."
                .to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Gradient with respect to the modulated features.
    ///
    /// The single-tensor `Layer` interface cannot express the conditioning-side
    /// gradient, so this delegates to
    /// [`FiLMModule::backward_with_conditioning`] and returns only the feature
    /// half; use that method directly when both gradients are needed.
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(self.backward_with_conditioning(grad_output)?.0)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update gamma and beta projection layers
        self.gamma_proj.update(learning_rate)?;
        self.beta_proj.update(learning_rate)?;
        Ok(())
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
}

/// Bilinear Fusion module for pairwise interactions between modalities
#[derive(Debug)]
pub struct BilinearFusion<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
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
    /// `(features_a, features_b, a_proj, b_proj, bilinear)` recorded by
    /// `forward` for the backward pass
    #[allow(clippy::type_complexity)]
    cache: Arc<RwLock<Option<BilinearCache<F>>>>,
}

/// Intermediates of one [`BilinearFusion::forward`] call
#[derive(Debug, Clone)]
struct BilinearCache<F> {
    /// First modality input
    features_a: Array<F, IxDyn>,
    /// Second modality input
    features_b: Array<F, IxDyn>,
    /// `proj_a(features_a)`
    a_proj: Array<F, IxDyn>,
    /// `proj_b(features_b)`
    b_proj: Array<F, IxDyn>,
    /// Element-wise product `a_proj * b_proj`
    bilinear: Array<F, IxDyn>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Clone for BilinearFusion<F> {
    fn clone(&self) -> Self {
        Self {
            dim_a: self.dim_a,
            dim_b: self.dim_b,
            output_dim: self.output_dim,
            proj_a: self.proj_a.clone(),
            proj_b: self.proj_b.clone(),
            low_rank_proj: self.low_rank_proj.clone(),
            cache: Arc::new(RwLock::new(match self.cache.read() {
                Ok(guard) => guard.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            })),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> BilinearFusion<F> {
    /// Create a new BilinearFusion module
    pub fn new(dim_a: usize, dim_b: usize, output_dim: usize, rank: usize) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
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
            cache: Arc::new(RwLock::new(None)),
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
        let output = self.low_rank_proj.forward(&bilinear)?;
        if let Ok(mut guard) = self.cache.write() {
            *guard = Some(BilinearCache {
                features_a: features_a.to_owned(),
                features_b: features_b.to_owned(),
                a_proj,
                b_proj,
                bilinear: bilinear.clone(),
            });
        }
        Ok(output)
    }

    /// Exact gradient of `low_rank(proj_a(a) * proj_b(b))`.
    ///
    /// # Returns
    /// `(grad_features_a, grad_features_b)`. All three projection layers also
    /// record their own weight gradients for a subsequent [`Layer::update`].
    pub fn backward_with_features(
        &self,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let cache = self
            .cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on the cache".to_string())
            })?
            .clone()
            .ok_or_else(|| {
                NeuralError::InferenceError(
                    "No cached values for backward pass. Call forward() first.".to_string(),
                )
            })?;

        let grad_bilinear = self.low_rank_proj.backward(&cache.bilinear, grad_output)?;
        let grad_a_proj = &grad_bilinear * &cache.b_proj;
        let grad_b_proj = &grad_bilinear * &cache.a_proj;
        let grad_a = self.proj_a.backward(&cache.features_a, &grad_a_proj)?;
        let grad_b = self.proj_b.backward(&cache.features_b, &grad_b_proj)?;
        Ok((grad_a, grad_b))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Layer<F> for BilinearFusion<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This assumes the input contains both feature sets packed together
        Err(NeuralError::ValidationError(
            "BilinearFusion requires separate feature inputs. Use the dedicated forward method."
                .to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Gradient with respect to the first modality.
    ///
    /// The single-tensor `Layer` interface cannot express the second modality's
    /// gradient, so this delegates to
    /// [`BilinearFusion::backward_with_features`] and returns only the first
    /// half; use that method directly when both gradients are needed.
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(self.backward_with_features(grad_output)?.0)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.proj_a.update(learning_rate)?;
        self.proj_b.update(learning_rate)?;
        self.low_rank_proj.update(learning_rate)?;
        Ok(())
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
}

/// Mean-pool a 3D `[batch, seq, features]` tensor over the sequence axis
/// (axis 1), producing `[batch, features]`.
///
/// Used to bridge [`CrossModalAttention`]'s inherently sequence-shaped
/// output into the 2D-only (`Dense`-based) `post_fusion`/`classifier` stages
/// of [`FeatureFusion`].
fn mean_pool_axis1<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign>(
    x: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    let shape = x.shape();
    if shape.len() != 3 {
        return Err(NeuralError::InferenceError(format!(
            "mean_pool_axis1 expects a 3D tensor, got {shape:?}"
        )));
    }
    let (batch, seq, feat) = (shape[0], shape[1], shape[2]);
    let n = F::from(seq).ok_or_else(|| {
        NeuralError::InferenceError("Failed to convert sequence length".to_string())
    })?;
    let mut out = Array::<F, IxDyn>::zeros(IxDyn(&[batch, feat]));
    for b in 0..batch {
        for f in 0..feat {
            let mut sum = F::zero();
            for s in 0..seq {
                sum += x[[b, s, f]];
            }
            out[[b, f]] = sum / n;
        }
    }
    Ok(out)
}

/// Gradient of [`mean_pool_axis1`]: broadcasts a `[batch, features]` gradient
/// evenly back across `seq` sequence positions, producing `[batch, seq,
/// features]` (each position receives `grad / seq`, matching the forward
/// average).
fn mean_pool_axis1_backward<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign>(
    grad: &Array<F, IxDyn>,
    seq: usize,
) -> Result<Array<F, IxDyn>> {
    let shape = grad.shape();
    if shape.len() != 2 {
        return Err(NeuralError::InferenceError(format!(
            "mean_pool_axis1_backward expects a 2D gradient, got {shape:?}"
        )));
    }
    let (batch, feat) = (shape[0], shape[1]);
    let n = F::from(seq).ok_or_else(|| {
        NeuralError::InferenceError("Failed to convert sequence length".to_string())
    })?;
    let mut out = Array::<F, IxDyn>::zeros(IxDyn(&[batch, seq, feat]));
    for b in 0..batch {
        for s in 0..seq {
            for f in 0..feat {
                out[[b, s, f]] = grad[[b, f]] / n;
            }
        }
    }
    Ok(out)
}

/// Intermediates of one [`FeatureFusion::forward_multi`] call, required by
/// [`FeatureFusion::backward_multi`].
#[derive(Debug, Clone)]
struct FusionForwardCache<F> {
    /// Raw per-modality inputs passed to `forward_multi`
    inputs: Vec<Array<F, IxDyn>>,
    /// Per-modality outputs of the feature aligners, pre-fusion
    aligned_features: Vec<Array<F, IxDyn>>,
    /// Output of the fusion step, pre-`post_fusion` (always 2D `[batch,
    /// hidden_dim]`; [`FusionMethod::Attention`] mean-pools its 3D
    /// `[batch, query_len, hidden_dim]` attention output down to this shape)
    fused: Array<F, IxDyn>,
    /// Output of `post_fusion`, pre-classifier
    post_fusion_output: Array<F, IxDyn>,
    /// Query sequence length of the last [`FusionMethod::Attention`] forward
    /// pass, needed to un-pool `fused`'s gradient back to 3D for
    /// [`CrossModalAttention::backward_with_context`]. `None` for every
    /// other fusion method.
    attention_query_len: Option<usize>,
}

/// Feature Fusion model
pub struct FeatureFusion<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign>
where
    F: SimdUnifiedOps,
{
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
    /// Forward-pass cache required by [`FeatureFusion::backward_multi`]
    cache: Arc<RwLock<Option<FusionForwardCache<F>>>>,
}

// Manual implementation of Debug for FeatureFusion to handle dyn Layer trait objects
impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Debug for FeatureFusion<F>
where
    F: SimdUnifiedOps,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureFusion")
            .field("aligners", &self.aligners)
            .field(
                "fusion_module",
                &"<Box<dyn Layer<F> + Send + Sync>>".to_string(),
            )
            .field("post_fusion", &self.post_fusion)
            .field("classifier", &self.classifier)
            .field("config", &self.config)
            .finish()
    }
}

// Manual implementation of Clone for FeatureFusion
impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Clone for FeatureFusion<F>
where
    F: SimdUnifiedOps,
{
    fn clone(&self) -> Self {
        // We can't clone the dyn Layer directly, so we create a new FeatureFusion
        // without the fusion_module
        // We would need to implement custom clone logic for fusion_module
        // based on its actual type if needed, but for now we leave it as None
        Self {
            aligners: self.aligners.clone(),
            fusion_module: None, // Can't clone the trait object
            post_fusion: self.post_fusion.clone(),
            classifier: self.classifier.clone(),
            config: self.config.clone(),
            cache: Arc::new(RwLock::new(match self.cache.read() {
                Ok(guard) => guard.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            })),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> FeatureFusion<F>
where
    F: SimdUnifiedOps,
{
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

        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        post_fusion.add(Dense::<F>::new(
            post_fusion_input_dim,
            config.hidden_dim * 2,
            Some("gelu"),
            &mut rng,
        )?);
        if config.dropout_rate > 0.0 {
            post_fusion.add(Dropout::<F>::new(config.dropout_rate, &mut rng)?);
        }
        post_fusion.add(Dense::<F>::new(
            config.hidden_dim * 2,
            config.hidden_dim,
            Some("gelu"),
            &mut rng,
        )?);

        // Create classifier if needed
        let classifier = if config.include_head {
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
            cache: Arc::new(RwLock::new(None)),
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

        // Apply fusion based on method. `attention_query_len` records the
        // pre-pooling query sequence length for `FusionMethod::Attention` so
        // `backward_multi` can un-pool the gradient; every other method
        // leaves it `None`.
        let mut attention_query_len: Option<usize> = None;
        let fused = match self.config.fusion_method {
            FusionMethod::Concatenation => {
                // Concatenate along feature dimension
                let batch_size = aligned_features[0].shape()[0];
                let mut concatenated = Vec::new();
                for batch_idx in 0..batch_size {
                    for features in &aligned_features {
                        let batch_features = features.slice_axis(
                            Axis(0),
                            scirs2_core::ndarray::Slice::from(batch_idx..batch_idx + 1),
                        );
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
                    result += features;
                }
                result
            }
            FusionMethod::Product => {
                // Element-wise product
                let mut result = aligned_features[0].clone();
                for features in &aligned_features[1..] {
                    result *= features;
                }
                result
            }
            FusionMethod::Attention => {
                // Use attention module (modality 0 attends to modality 1).
                // `CrossModalAttention` is inherently sequence-shaped
                // (`[batch, seq, features]`), but `post_fusion`/`classifier`
                // are `Dense`-based and 2D-only, so its attended sequence is
                // mean-pooled over the query positions into a single
                // `[batch, hidden_dim]` vector per sample -- a standard
                // sequence-to-vector reduction (cf. mean-pooled sentence
                // embeddings).
                if let Some(ref module) = self.fusion_module {
                    // We need to cast the module as CrossModalAttention
                    if let Some(attn) = module.as_any().downcast_ref::<CrossModalAttention<F>>() {
                        let attended = attn.forward(&aligned_features[0], &aligned_features[1])?;
                        if attended.ndim() == 3 {
                            let query_len = attended.shape()[1];
                            attention_query_len = Some(query_len);
                            mean_pool_axis1(&attended)?
                        } else {
                            attended
                        }
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
        let post_fusion_output = self.post_fusion.forward(&fused)?;

        if let Ok(mut guard) = self.cache.write() {
            *guard = Some(FusionForwardCache {
                inputs: inputs.to_vec(),
                aligned_features,
                fused,
                post_fusion_output: post_fusion_output.clone(),
                attention_query_len,
            });
        }

        // Apply classifier if available
        if let Some(ref classifier) = self.classifier {
            classifier.forward(&post_fusion_output)
        } else {
            Ok(post_fusion_output)
        }
    }

    /// Exact gradient of [`FeatureFusion::forward_multi`].
    ///
    /// Backpropagates through the classifier (if present), the post-fusion
    /// network, the fusion operation itself (dispatching on
    /// [`FusionMethod`]), and finally each modality's [`FeatureAlignment`],
    /// returning one gradient array per input modality in the same order
    /// they were passed to `forward_multi`. All parameterized sub-modules
    /// also record their own weight gradients, so a subsequent
    /// [`Layer::update`] trains the whole pipeline.
    ///
    /// # Errors
    /// Returns [`NeuralError::InferenceError`] if `forward_multi` has not
    /// been called yet.
    pub fn backward_multi(&self, grad_output: &Array<F, IxDyn>) -> Result<Vec<Array<F, IxDyn>>> {
        let cache = self
            .cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on the fusion cache".to_string(),
                )
            })?
            .clone()
            .ok_or_else(|| {
                NeuralError::InferenceError(
                    "No cached values for backward pass. Call forward_multi() first.".to_string(),
                )
            })?;

        // 1. Backward through the classifier head, if present.
        let grad_post_fusion = if let Some(ref classifier) = self.classifier {
            classifier.backward(&cache.post_fusion_output, grad_output)?
        } else {
            grad_output.clone()
        };

        // 2. Backward through the post-fusion MLP.
        let grad_fused = self.post_fusion.backward(&cache.fused, &grad_post_fusion)?;

        // 3. Backward through the fusion operation itself.
        let num_modalities = cache.aligned_features.len();
        let mut grad_aligned: Vec<Array<F, IxDyn>> = Vec::with_capacity(num_modalities);

        match self.config.fusion_method {
            FusionMethod::Concatenation => {
                let hidden_dim = self.config.hidden_dim;
                let batch_size = cache.aligned_features[0].shape()[0];
                for (i, aligned) in cache.aligned_features.iter().enumerate() {
                    let mut grad_i = Array::<F, IxDyn>::zeros(aligned.dim());
                    for b in 0..batch_size {
                        for h in 0..hidden_dim {
                            grad_i[[b, h]] = grad_fused[[b, i * hidden_dim + h]];
                        }
                    }
                    grad_aligned.push(grad_i);
                }
            }
            FusionMethod::Sum => {
                // d(a_0 + a_1 + ...)/d(a_i) = identity for every modality.
                for _ in 0..num_modalities {
                    grad_aligned.push(grad_fused.clone());
                }
            }
            FusionMethod::Product => {
                // d(prod_i a_i)/d(a_k) = grad_fused * prod_{i != k} a_i
                for k in 0..num_modalities {
                    let mut grad_k = grad_fused.clone();
                    for (i, aligned) in cache.aligned_features.iter().enumerate() {
                        if i != k {
                            grad_k *= aligned;
                        }
                    }
                    grad_aligned.push(grad_k);
                }
            }
            FusionMethod::Attention => {
                let module = self.fusion_module.as_ref().ok_or_else(|| {
                    NeuralError::InferenceError(
                        "Attention fusion module not initialized".to_string(),
                    )
                })?;
                let attn = module
                    .as_any()
                    .downcast_ref::<CrossModalAttention<F>>()
                    .ok_or_else(|| {
                        NeuralError::InferenceError(
                            "Failed to cast fusion module to CrossModalAttention".to_string(),
                        )
                    })?;
                // Undo the mean-pool from `forward_multi` before handing the
                // gradient to the (3D-shaped) attention backward.
                let grad_fused_3d = match cache.attention_query_len {
                    Some(query_len) => mean_pool_axis1_backward(&grad_fused, query_len)?,
                    None => grad_fused.clone(),
                };
                let (grad_query, grad_context) = attn.backward_with_context(&grad_fused_3d)?;
                grad_aligned.push(grad_query);
                grad_aligned.push(grad_context);
            }
            FusionMethod::Bilinear => {
                let module = self.fusion_module.as_ref().ok_or_else(|| {
                    NeuralError::InferenceError(
                        "Bilinear fusion module not initialized".to_string(),
                    )
                })?;
                let bilinear = module
                    .as_any()
                    .downcast_ref::<BilinearFusion<F>>()
                    .ok_or_else(|| {
                        NeuralError::InferenceError(
                            "Failed to cast fusion module to BilinearFusion".to_string(),
                        )
                    })?;
                let (grad_a, grad_b) = bilinear.backward_with_features(&grad_fused)?;
                grad_aligned.push(grad_a);
                grad_aligned.push(grad_b);
            }
            FusionMethod::FiLM => {
                let module = self.fusion_module.as_ref().ok_or_else(|| {
                    NeuralError::InferenceError("FiLM fusion module not initialized".to_string())
                })?;
                let film = module
                    .as_any()
                    .downcast_ref::<FiLMModule<F>>()
                    .ok_or_else(|| {
                        NeuralError::InferenceError(
                            "Failed to cast fusion module to FiLMModule".to_string(),
                        )
                    })?;
                let (grad_features, grad_cond) = film.backward_with_conditioning(&grad_fused)?;
                grad_aligned.push(grad_features);
                grad_aligned.push(grad_cond);
            }
        }

        // 4. Backward through each modality's aligner to reach the raw inputs.
        let mut grad_inputs = Vec::with_capacity(num_modalities);
        for ((aligner, input), grad) in self
            .aligners
            .iter()
            .zip(cache.inputs.iter())
            .zip(grad_aligned.iter())
        {
            grad_inputs.push(aligner.backward(input, grad)?);
        }

        Ok(grad_inputs)
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

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> Layer<F> for FeatureFusion<F>
where
    F: SimdUnifiedOps,
{
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // For a single packed input, we need to split it into modalities
        // This is mainly for the Layer trait compatibility
        // In practice, use forward_multi with separate inputs
        Err(NeuralError::ValidationError(
            "FeatureFusion requires multiple inputs. Use forward_multi method instead.".to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Gradient with respect to the first modality's raw input.
    ///
    /// The single-tensor `Layer` interface cannot express the per-modality
    /// gradients that a multi-input fusion model produces, so this delegates
    /// to [`FeatureFusion::backward_multi`] and returns only the first
    /// modality's half (consistent with how [`CrossModalAttention`],
    /// [`FiLMModule`], and [`BilinearFusion`] handle the same limitation).
    /// Call [`FeatureFusion::backward_multi`] directly to get every
    /// modality's gradient.
    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let mut grads = self.backward_multi(grad_output)?;
        if grads.is_empty() {
            return Err(NeuralError::InferenceError(
                "FeatureFusion::backward_multi returned no gradients".to_string(),
            ));
        }
        Ok(grads.swap_remove(0))
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all aligners
        for aligner in &mut self.aligners {
            aligner.update(learning_rate)?;
        }
        // Update fusion module if present
        if let Some(ref mut module) = self.fusion_module {
            module.update(learning_rate)?;
        }
        // Update post-fusion network
        self.post_fusion.update(learning_rate)?;
        // Update classifier if present
        if let Some(ref mut classifier) = self.classifier {
            classifier.update(learning_rate)?;
        }
        Ok(())
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
}
