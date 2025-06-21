//! Knowledge distillation utilities for neural networks
//!
//! This module provides tools for knowledge distillation including:
//! - Teacher-student training frameworks
//! - Various distillation loss functions
//! - Feature-based distillation
//! - Self-distillation techniques

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use ndarray::{Array, ArrayD, Axis};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Knowledge distillation method
#[derive(Debug, Clone, PartialEq)]
pub enum DistillationMethod {
    /// Response-based distillation (output matching)
    ResponseBased {
        /// Temperature for softmax scaling
        temperature: f64,
        /// Weight for distillation loss
        alpha: f64,
        /// Weight for ground truth loss
        beta: f64,
    },
    /// Feature-based distillation (intermediate layer matching)
    FeatureBased {
        /// Names of feature layers to match
        feature_layers: Vec<String>,
        /// Method for adapting feature dimensions
        adaptation_method: FeatureAdaptation,
    },
    /// Attention-based distillation
    AttentionBased {
        /// Names of attention layers to match
        attention_layers: Vec<String>,
        /// Type of attention mechanism
        attention_type: AttentionType,
    },
    /// Relation-based distillation
    RelationBased {
        /// Type of relational information to distill
        relation_type: RelationType,
        /// Distance metric for comparing relations
        distance_metric: DistanceMetric,
    },
    /// Self-distillation
    SelfDistillation {
        /// Number of models in the ensemble
        ensemble_size: usize,
        /// Method for aggregating ensemble outputs
        aggregation: EnsembleAggregation,
    },
}

/// Feature adaptation methods for different sized features
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureAdaptation {
    /// Linear transformation
    Linear,
    /// Convolutional adaptation
    Convolutional {
        /// Convolution kernel size (height, width)
        kernel_size: (usize, usize),
        /// Convolution stride (height, width)
        stride: (usize, usize),
    },
    /// Attention-based adaptation
    Attention,
    /// Average pooling adaptation
    AvgPool {
        /// Average pooling size (height, width)
        pool_size: (usize, usize),
    },
}

/// Attention types for distillation
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionType {
    /// Spatial attention maps
    Spatial,
    /// Channel attention
    Channel,
    /// Self-attention matrices
    SelfAttention,
}

/// Relation types for relation-based distillation
#[derive(Debug, Clone, PartialEq)]
pub enum RelationType {
    /// Pairwise relationships between samples
    SampleWise,
    /// Channel-wise relationships
    ChannelWise,
    /// Spatial relationships
    SpatialWise,
}

/// Distance metrics for relation computation
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine similarity
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// KL divergence
    KLDivergence,
}

/// Ensemble aggregation methods
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleAggregation {
    /// Simple averaging
    Average,
    /// Weighted averaging
    Weighted {
        /// Weights for each ensemble member
        weights: Vec<f64>,
    },
    /// Soft voting
    SoftVoting,
}

/// Knowledge distillation trainer
pub struct DistillationTrainer<F: Float + Debug> {
    /// Distillation method
    method: DistillationMethod,
    /// Feature extractors for intermediate layers
    feature_extractors: HashMap<String, Box<dyn Layer<F> + Send + Sync>>,
    /// Adaptation layers for feature matching
    adaptation_layers: HashMap<String, Box<dyn Layer<F> + Send + Sync>>,
    /// Training statistics
    training_stats: DistillationStatistics<F>,
}

/// Statistics tracking for distillation training
#[derive(Debug, Clone)]
pub struct DistillationStatistics<F: Float + Debug> {
    /// Total distillation loss over time
    pub distillation_loss_history: Vec<F>,
    /// Ground truth loss over time
    pub ground_truth_loss_history: Vec<F>,
    /// Feature matching losses per layer
    pub feature_losses: HashMap<String, Vec<F>>,
    /// Teacher-student similarity metrics
    pub similarity_metrics: HashMap<String, F>,
    /// Training step
    pub current_step: usize,
}

impl<F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand>
    DistillationTrainer<F>
{
    /// Create a new distillation trainer
    pub fn new(method: DistillationMethod) -> Self {
        Self {
            method,
            feature_extractors: HashMap::new(),
            adaptation_layers: HashMap::new(),
            training_stats: DistillationStatistics {
                distillation_loss_history: Vec::new(),
                ground_truth_loss_history: Vec::new(),
                feature_losses: HashMap::new(),
                similarity_metrics: HashMap::new(),
                current_step: 0,
            },
        }
    }

    /// Add feature extractor for a specific layer
    pub fn add_feature_extractor(
        &mut self,
        layer_name: String,
        extractor: Box<dyn Layer<F> + Send + Sync>,
    ) {
        self.feature_extractors.insert(layer_name, extractor);
    }

    /// Add adaptation layer for feature matching
    pub fn add_adaptation_layer(
        &mut self,
        layer_name: String,
        adapter: Box<dyn Layer<F> + Send + Sync>,
    ) {
        self.adaptation_layers.insert(layer_name, adapter);
    }

    /// Compute distillation loss
    pub fn compute_distillation_loss(
        &mut self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        ground_truth: Option<&ArrayD<F>>,
    ) -> Result<F> {
        match &self.method {
            DistillationMethod::ResponseBased {
                temperature,
                alpha,
                beta,
            } => self.compute_response_loss(
                teacher_outputs,
                student_outputs,
                ground_truth,
                *temperature,
                *alpha,
                *beta,
            ),
            DistillationMethod::FeatureBased {
                feature_layers,
                adaptation_method,
            } => self.compute_feature_loss(
                teacher_outputs,
                student_outputs,
                feature_layers,
                adaptation_method,
            ),
            DistillationMethod::AttentionBased {
                attention_layers,
                attention_type,
            } => self.compute_attention_loss(
                teacher_outputs,
                student_outputs,
                attention_layers,
                attention_type,
            ),
            DistillationMethod::RelationBased {
                relation_type,
                distance_metric,
            } => self.compute_relation_loss(
                teacher_outputs,
                student_outputs,
                relation_type,
                distance_metric,
            ),
            DistillationMethod::SelfDistillation {
                ensemble_size,
                aggregation,
            } => self.compute_self_distillation_loss(student_outputs, *ensemble_size, aggregation),
        }
    }

    fn compute_response_loss(
        &mut self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        ground_truth: Option<&ArrayD<F>>,
        temperature: f64,
        alpha: f64,
        beta: f64,
    ) -> Result<F> {
        let temp = F::from(temperature)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid temperature".to_string()))?;
        let alpha_f = F::from(alpha)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid alpha".to_string()))?;
        let beta_f = F::from(beta)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid beta".to_string()))?;

        // Get final outputs (assuming "output" key)
        let teacher_logits = teacher_outputs
            .get("output")
            .ok_or_else(|| NeuralError::ComputationError("Teacher output not found".to_string()))?;
        let student_logits = student_outputs
            .get("output")
            .ok_or_else(|| NeuralError::ComputationError("Student output not found".to_string()))?;

        if teacher_logits.shape() != student_logits.shape() {
            return Err(NeuralError::DimensionMismatch(
                "Teacher and student output shapes don't match".to_string(),
            ));
        }

        // Compute soft targets using temperature scaling
        let teacher_soft = self.softmax_with_temperature(teacher_logits, temp)?;
        let student_soft = self.softmax_with_temperature(student_logits, temp)?;

        // KL divergence loss between soft targets
        let kl_loss = self.kl_divergence_loss(&teacher_soft, &student_soft)?;
        let distillation_loss = kl_loss * temp * temp; // Scale by T^2

        let mut total_loss = alpha_f * distillation_loss;

        // Add ground truth loss if available
        if let Some(gt) = ground_truth {
            let ce_loss = self.cross_entropy_loss(student_logits, gt)?;
            total_loss = total_loss + beta_f * ce_loss;
            self.training_stats.ground_truth_loss_history.push(ce_loss);
        }

        self.training_stats
            .distillation_loss_history
            .push(distillation_loss);
        self.training_stats.current_step += 1;

        Ok(total_loss)
    }

    fn compute_feature_loss(
        &self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        feature_layers: &[String],
        adaptation_method: &FeatureAdaptation,
    ) -> Result<F> {
        let mut total_loss = F::zero();
        let mut layer_count = 0;

        for layer_name in feature_layers {
            if let (Some(teacher_feat), Some(student_feat)) = (
                teacher_outputs.get(layer_name),
                student_outputs.get(layer_name),
            ) {
                // Adapt student features to match teacher dimensions if needed
                let adapted_student =
                    self.adapt_features(student_feat, teacher_feat, adaptation_method)?;

                // Compute L2 loss between features
                let diff = teacher_feat - &adapted_student;
                let layer_loss = diff.mapv(|x| x * x).mean().unwrap_or(F::zero());

                total_loss = total_loss + layer_loss;
                layer_count += 1;

                // Note: per-layer loss tracking removed for immutable methods
            }
        }

        if layer_count > 0 {
            total_loss = total_loss / F::from(layer_count).unwrap();
        }

        Ok(total_loss)
    }

    fn compute_attention_loss(
        &self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        attention_layers: &[String],
        attention_type: &AttentionType,
    ) -> Result<F> {
        let mut total_loss = F::zero();
        let mut layer_count = 0;

        for layer_name in attention_layers {
            if let (Some(teacher_feat), Some(student_feat)) = (
                teacher_outputs.get(layer_name),
                student_outputs.get(layer_name),
            ) {
                let teacher_attention =
                    self.compute_attention_maps(teacher_feat, attention_type)?;
                let student_attention =
                    self.compute_attention_maps(student_feat, attention_type)?;

                // Normalize attention maps
                let teacher_norm = self.normalize_attention(&teacher_attention)?;
                let student_norm = self.normalize_attention(&student_attention)?;

                // Compute L2 loss between attention maps
                let diff = teacher_norm - student_norm;
                let layer_loss = diff.mapv(|x| x * x).mean().unwrap_or(F::zero());

                total_loss = total_loss + layer_loss;
                layer_count += 1;
            }
        }

        if layer_count > 0 {
            total_loss = total_loss / F::from(layer_count).unwrap();
        }

        Ok(total_loss)
    }

    fn compute_relation_loss(
        &self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        relation_type: &RelationType,
        distance_metric: &DistanceMetric,
    ) -> Result<F> {
        // Get features for relation computation
        let teacher_feat = teacher_outputs
            .get("features")
            .or_else(|| teacher_outputs.get("output"))
            .ok_or_else(|| {
                NeuralError::ComputationError("Teacher features not found".to_string())
            })?;
        let student_feat = student_outputs
            .get("features")
            .or_else(|| student_outputs.get("output"))
            .ok_or_else(|| {
                NeuralError::ComputationError("Student features not found".to_string())
            })?;

        let teacher_relations =
            self.compute_relations(teacher_feat, relation_type, distance_metric)?;
        let student_relations =
            self.compute_relations(student_feat, relation_type, distance_metric)?;

        // Compute loss between relation matrices
        let diff = teacher_relations - student_relations;
        let loss = diff.mapv(|x| x * x).mean().unwrap_or(F::zero());

        Ok(loss)
    }

    fn compute_self_distillation_loss(
        &self,
        student_outputs: &HashMap<String, ArrayD<F>>,
        ensemble_size: usize,
        aggregation: &EnsembleAggregation,
    ) -> Result<F> {
        // Self-distillation: use multiple predictions from the same model
        if ensemble_size < 2 {
            return Ok(F::zero());
        }

        // For simplicity, assume we have multiple outputs stored with different keys
        let mut ensemble_outputs = Vec::new();
        for i in 0..ensemble_size {
            let key = format!("output_{}", i);
            if let Some(output) = student_outputs.get(&key) {
                ensemble_outputs.push(output);
            }
        }

        if ensemble_outputs.len() < 2 {
            return Ok(F::zero());
        }

        // Compute ensemble prediction
        let ensemble_pred = self.aggregate_ensemble(&ensemble_outputs, aggregation)?;

        // Compute KL divergence between individual predictions and ensemble
        let mut total_loss = F::zero();
        for output in &ensemble_outputs {
            let kl_loss = self.kl_divergence_loss(&ensemble_pred, output)?;
            total_loss = total_loss + kl_loss;
        }

        total_loss = total_loss / F::from(ensemble_outputs.len()).unwrap();
        Ok(total_loss)
    }

    fn softmax_with_temperature(&self, logits: &ArrayD<F>, temperature: F) -> Result<ArrayD<F>> {
        let scaled_logits = logits / temperature;
        self.softmax(&scaled_logits)
    }

    fn softmax(&self, x: &ArrayD<F>) -> Result<ArrayD<F>> {
        // Compute softmax along the last axis
        let last_axis = x.ndim() - 1;
        let axis = Axis(last_axis);

        // Subtract max for numerical stability
        let max_vals = x.map_axis(axis, |view| {
            view.iter().cloned().fold(F::neg_infinity(), F::max)
        });

        let shifted = x - &max_vals.insert_axis(axis);
        let exp_vals = shifted.mapv(|x| x.exp());

        let sum_exp = exp_vals.sum_axis(axis);
        let result = exp_vals / &sum_exp.insert_axis(axis);

        Ok(result)
    }

    fn kl_divergence_loss(&self, target: &ArrayD<F>, prediction: &ArrayD<F>) -> Result<F> {
        let eps = F::from(1e-8).unwrap();

        let log_target = target.mapv(|x| (x + eps).ln());
        let log_pred = prediction.mapv(|x| (x + eps).ln());

        let kl = target * (log_target - log_pred);
        let loss = kl.sum() / F::from(target.len()).unwrap();

        Ok(loss)
    }

    fn cross_entropy_loss(&self, logits: &ArrayD<F>, targets: &ArrayD<F>) -> Result<F> {
        let probs = self.softmax(logits)?;
        let eps = F::from(1e-8).unwrap();

        let log_probs = probs.mapv(|x| (x + eps).ln());
        let ce = -(targets * log_probs).sum() / F::from(targets.shape()[0]).unwrap();

        Ok(ce)
    }

    fn adapt_features(
        &self,
        student_feat: &ArrayD<F>,
        teacher_feat: &ArrayD<F>,
        method: &FeatureAdaptation,
    ) -> Result<ArrayD<F>> {
        if student_feat.shape() == teacher_feat.shape() {
            return Ok(student_feat.clone());
        }

        match method {
            FeatureAdaptation::Linear => {
                // Simple linear interpolation/projection (simplified)
                // In practice, this would use a learned linear layer
                if student_feat.len() == teacher_feat.len() {
                    Ok(student_feat
                        .clone()
                        .to_shape(teacher_feat.raw_dim())?
                        .to_owned())
                } else {
                    // Pad or truncate
                    let target_shape = teacher_feat.raw_dim();
                    let mut adapted = Array::zeros(target_shape);

                    // Copy available data
                    let min_size = student_feat.len().min(teacher_feat.len());
                    for (i, &val) in student_feat.iter().take(min_size).enumerate() {
                        if i < adapted.len() {
                            adapted[i] = val;
                        }
                    }

                    Ok(adapted)
                }
            }
            _ => {
                // For other adaptation methods, return student features as-is
                // In practice, these would involve learned adaptation layers
                Ok(student_feat.clone())
            }
        }
    }

    fn compute_attention_maps(
        &self,
        features: &ArrayD<F>,
        attention_type: &AttentionType,
    ) -> Result<ArrayD<F>> {
        match attention_type {
            AttentionType::Spatial => {
                // Compute spatial attention by summing across channels
                if features.ndim() >= 3 {
                    let spatial_map = features.sum_axis(Axis(1)); // Sum across channel dimension
                    Ok(spatial_map)
                } else {
                    Ok(features.clone())
                }
            }
            AttentionType::Channel => {
                // Compute channel attention by averaging across spatial dimensions
                if features.ndim() >= 3 {
                    let mut channel_map = features.clone();
                    for _ in 2..features.ndim() {
                        channel_map = channel_map
                            .mean_axis(Axis(channel_map.ndim() - 1))
                            .ok_or_else(|| {
                                NeuralError::ComputationError("Failed to compute mean".to_string())
                            })?;
                    }
                    Ok(channel_map)
                } else {
                    Ok(features.clone())
                }
            }
            AttentionType::SelfAttention => {
                // Simplified self-attention computation
                Ok(features.clone()) // Placeholder
            }
        }
    }

    fn normalize_attention(&self, attention: &ArrayD<F>) -> Result<ArrayD<F>> {
        let sum = attention.sum();
        if sum > F::zero() {
            Ok(attention / sum)
        } else {
            Ok(attention.clone())
        }
    }

    fn compute_relations(
        &self,
        features: &ArrayD<F>,
        relation_type: &RelationType,
        distance_metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        match relation_type {
            RelationType::SampleWise => self.compute_sample_relations(features, distance_metric),
            RelationType::ChannelWise => self.compute_channel_relations(features, distance_metric),
            RelationType::SpatialWise => self.compute_spatial_relations(features, distance_metric),
        }
    }

    fn compute_sample_relations(
        &self,
        features: &ArrayD<F>,
        metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        let batch_size = features.shape()[0];
        let mut relations = Array::zeros((batch_size, batch_size));

        for i in 0..batch_size {
            for j in 0..batch_size {
                let feat_i = features.slice(ndarray::s![i, ..]);
                let feat_j = features.slice(ndarray::s![j, ..]);

                let distance = match metric {
                    DistanceMetric::Euclidean => {
                        let diff = &feat_i - &feat_j;
                        diff.mapv(|x| x * x).sum().sqrt()
                    }
                    DistanceMetric::Cosine => {
                        let dot = (&feat_i * &feat_j).sum();
                        let norm_i = feat_i.mapv(|x| x * x).sum().sqrt();
                        let norm_j = feat_j.mapv(|x| x * x).sum().sqrt();
                        dot / (norm_i * norm_j)
                    }
                    DistanceMetric::Manhattan => {
                        let diff = &feat_i - &feat_j;
                        diff.mapv(|x| x.abs()).sum()
                    }
                    DistanceMetric::KLDivergence => {
                        // Simplified KL divergence
                        let eps = F::from(1e-8).unwrap();
                        let p = feat_i.mapv(|x| x + eps);
                        let q = feat_j.mapv(|x| x + eps);
                        let p_norm = &p / p.sum();
                        let q_norm = &q / q.sum();
                        (p_norm.clone() * (p_norm.mapv(|x| x.ln()) - q_norm.mapv(|x| x.ln()))).sum()
                    }
                };

                relations[[i, j]] = distance;
            }
        }

        Ok(relations.into_dyn())
    }

    fn compute_channel_relations(
        &self,
        features: &ArrayD<F>,
        _metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        // Simplified channel relation computation
        if features.ndim() < 2 {
            return Ok(features.clone());
        }

        let channels = features.shape()[1];
        let relations = Array::eye(channels);
        Ok(relations.into_dyn())
    }

    fn compute_spatial_relations(
        &self,
        features: &ArrayD<F>,
        _metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        // Simplified spatial relation computation
        Ok(features.clone())
    }

    fn aggregate_ensemble(
        &self,
        outputs: &[&ArrayD<F>],
        method: &EnsembleAggregation,
    ) -> Result<ArrayD<F>> {
        if outputs.is_empty() {
            return Err(NeuralError::ComputationError("Empty ensemble".to_string()));
        }

        match method {
            EnsembleAggregation::Average => {
                let mut sum = outputs[0].clone();
                for output in outputs.iter().skip(1) {
                    sum = sum + *output;
                }
                Ok(sum / F::from(outputs.len()).unwrap())
            }
            EnsembleAggregation::Weighted { weights } => {
                if weights.len() != outputs.len() {
                    return Err(NeuralError::InvalidArchitecture(
                        "Weight count doesn't match ensemble size".to_string(),
                    ));
                }

                let mut result = outputs[0] * F::from(weights[0]).unwrap();
                for (output, &weight) in outputs.iter().zip(weights.iter()).skip(1) {
                    result = result + (*output * F::from(weight).unwrap());
                }
                Ok(result)
            }
            EnsembleAggregation::SoftVoting => {
                // Same as average for regression tasks - compute average directly
                let mut result = outputs[0].clone().to_owned();
                result.fill(F::zero());

                for output in outputs {
                    result = result + *output;
                }
                let n = F::from(outputs.len()).unwrap();
                Ok(result / n)
            }
        }
    }

    /// Get training statistics
    pub fn get_statistics(&self) -> &DistillationStatistics<F> {
        &self.training_stats
    }

    /// Reset training statistics
    pub fn reset_statistics(&mut self) {
        self.training_stats = DistillationStatistics {
            distillation_loss_history: Vec::new(),
            ground_truth_loss_history: Vec::new(),
            feature_losses: HashMap::new(),
            similarity_metrics: HashMap::new(),
            current_step: 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_response_based_distillation() {
        let method = DistillationMethod::ResponseBased {
            temperature: 3.0,
            alpha: 0.7,
            beta: 0.3,
        };

        let mut trainer = DistillationTrainer::<f64>::new(method);

        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();

        teacher_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.5, 1.5, 2.5, 1.0])
                .unwrap()
                .into_dyn(),
        );
        student_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![1.8, 1.2, 0.6, 1.4, 2.3, 1.1])
                .unwrap()
                .into_dyn(),
        );

        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
        assert!(loss.unwrap() > 0.0);
    }

    #[test]
    fn test_feature_based_distillation() {
        let method = DistillationMethod::FeatureBased {
            feature_layers: vec!["layer1".to_string(), "layer2".to_string()],
            adaptation_method: FeatureAdaptation::Linear,
        };

        let mut trainer = DistillationTrainer::<f64>::new(method);

        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();

        teacher_outputs.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap()
                .into_dyn(),
        );
        student_outputs.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 4), vec![1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9])
                .unwrap()
                .into_dyn(),
        );

        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
        assert!(loss.unwrap() > 0.0);
    }

    #[test]
    fn test_attention_based_distillation() {
        let method = DistillationMethod::AttentionBased {
            attention_layers: vec!["conv1".to_string()],
            attention_type: AttentionType::Spatial,
        };

        let mut trainer = DistillationTrainer::<f64>::new(method);

        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();

        // 3D tensor: batch_size=1, channels=2, spatial=4
        teacher_outputs.insert(
            "conv1".to_string(),
            Array::from_shape_vec((1, 2, 4), (0..8).map(|x| x as f64).collect())
                .unwrap()
                .into_dyn(),
        );
        student_outputs.insert(
            "conv1".to_string(),
            Array::from_shape_vec((1, 2, 4), (0..8).map(|x| x as f64 + 0.1).collect())
                .unwrap()
                .into_dyn(),
        );

        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_softmax_with_temperature() {
        let trainer = DistillationTrainer::<f64>::new(DistillationMethod::ResponseBased {
            temperature: 1.0,
            alpha: 1.0,
            beta: 0.0,
        });

        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5])
            .unwrap()
            .into_dyn();
        let temperature = 2.0;

        let result = trainer.softmax_with_temperature(&logits, temperature);
        assert!(result.is_ok());

        let softmax_output = result.unwrap();

        // Check that probabilities sum to 1 for each sample
        for i in 0..2 {
            let row_sum: f64 = (0..3).map(|j| softmax_output[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kl_divergence_loss() {
        let trainer = DistillationTrainer::<f64>::new(DistillationMethod::ResponseBased {
            temperature: 1.0,
            alpha: 1.0,
            beta: 0.0,
        });

        let target = Array2::from_shape_vec((2, 3), vec![0.7, 0.2, 0.1, 0.3, 0.4, 0.3])
            .unwrap()
            .into_dyn();
        let prediction = Array2::from_shape_vec((2, 3), vec![0.6, 0.3, 0.1, 0.2, 0.5, 0.3])
            .unwrap()
            .into_dyn();

        let loss = trainer.kl_divergence_loss(&target, &prediction);
        assert!(loss.is_ok());
        assert!(loss.unwrap() >= 0.0); // KL divergence is non-negative
    }

    #[test]
    fn test_ensemble_aggregation() {
        let trainer = DistillationTrainer::<f64>::new(DistillationMethod::SelfDistillation {
            ensemble_size: 3,
            aggregation: EnsembleAggregation::Average,
        });

        let output1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();
        let output2 = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
            .unwrap()
            .into_dyn();
        let output3 = Array2::from_shape_vec((2, 3), vec![0.9, 1.9, 2.9, 3.9, 4.9, 5.9])
            .unwrap()
            .into_dyn();

        let outputs = vec![&output1, &output2, &output3];

        let result = trainer.aggregate_ensemble(&outputs, &EnsembleAggregation::Average);
        assert!(result.is_ok());

        let avg_output = result.unwrap();
        assert_eq!(avg_output.shape(), output1.shape());

        // Check that it's the average
        assert!((avg_output[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((avg_output[[1, 2]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_distillation_statistics() {
        let mut trainer = DistillationTrainer::<f64>::new(DistillationMethod::ResponseBased {
            temperature: 3.0,
            alpha: 0.7,
            beta: 0.3,
        });

        // Simulate some training steps
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();

        teacher_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.5, 1.5, 2.5, 1.0])
                .unwrap()
                .into_dyn(),
        );
        student_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![1.8, 1.2, 0.6, 1.4, 2.3, 1.1])
                .unwrap()
                .into_dyn(),
        );

        for _ in 0..3 {
            let _ = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        }

        let stats = trainer.get_statistics();
        assert_eq!(stats.distillation_loss_history.len(), 3);
        assert_eq!(stats.current_step, 3);
    }
}
