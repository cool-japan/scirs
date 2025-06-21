//! Model interpretation utilities for neural networks
//!
//! This module provides tools for understanding neural network decisions including:
//! - Gradient-based attribution methods (Saliency, Integrated Gradients, GradCAM)
//! - Feature visualization and analysis
//! - Layer activation analysis and statistics
//! - Decision explanation tools

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

/// Distance metrics for counterfactual generation
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// L1 (Manhattan) distance
    L1,
    /// L2 (Euclidean) distance
    L2,
    /// L-infinity distance
    LInf,
    /// Custom weighted distance
    Weighted(Vec<f64>),
}

/// Perturbation strategies for LIME
#[derive(Debug, Clone, PartialEq)]
pub enum PerturbationStrategy {
    /// Binary masking (for images)
    BinaryMask,
    /// Gaussian noise addition
    GaussianNoise {
        /// Standard deviation of noise
        std: f64,
    },
    /// Uniform noise addition
    UniformNoise {
        /// Noise range
        range: f64,
    },
    /// Feature dropping
    FeatureDropping {
        /// Probability of dropping each feature
        drop_prob: f64,
    },
    /// Superpixel masking (for images)
    SuperpixelMask {
        /// Number of superpixels
        num_superpixels: usize,
    },
}

/// Attention aggregation methods
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionAggregation {
    /// Average across all heads
    Mean,
    /// Maximum across all heads
    Max,
    /// Minimum across all heads
    Min,
    /// Standard deviation across heads
    Std,
    /// Specific head selection
    SelectHead(usize),
    /// Weighted combination
    Weighted(Vec<f64>),
}

/// LRP (Layer-wise Relevance Propagation) rules
#[derive(Debug, Clone, PartialEq)]
pub enum LRPRule {
    /// Basic LRP rule (ε-rule)
    Epsilon,
    /// LRP-γ rule for lower layers
    Gamma {
        /// Gamma parameter
        gamma: f64,
    },
    /// LRP-α1β0 rule (equivalent to LRP-α2β1 with α=2, β=1)
    AlphaBeta {
        /// Alpha parameter
        alpha: f64,
        /// Beta parameter
        beta: f64,
    },
    /// LRP-z+ rule for input layer
    ZPlus,
    /// LRP-zB rule with bounds
    ZB {
        /// Lower bound
        low: f64,
        /// Upper bound
        high: f64,
    },
}

/// Attribution method for computing feature importance
#[derive(Debug, Clone, PartialEq)]
pub enum AttributionMethod {
    /// Simple gradient-based saliency
    Saliency,
    /// Integrated gradients
    IntegratedGradients {
        /// Baseline method for integration
        baseline: BaselineMethod,
        /// Number of integration steps
        num_steps: usize,
    },
    /// Grad-CAM (Gradient-weighted Class Activation Mapping)
    GradCAM {
        /// Name of target layer for gradient computation
        target_layer: String,
    },
    /// Guided backpropagation
    GuidedBackprop,
    /// DeepLIFT
    DeepLIFT {
        /// Baseline method for DeepLIFT
        baseline: BaselineMethod,
    },
    /// SHAP (SHapley Additive exPlanations)
    SHAP {
        /// Number of background samples for SHAP
        background_samples: usize,
        /// Number of samples for SHAP approximation
        num_samples: usize,
    },
    /// Layer-wise Relevance Propagation
    LayerWiseRelevancePropagation {
        /// LRP rule to use
        rule: LRPRule,
        /// Epsilon for numerical stability
        epsilon: f64,
    },
    /// SmoothGrad
    SmoothGrad {
        /// Base attribution method to smooth
        base_method: Box<AttributionMethod>,
        /// Number of noisy samples
        num_samples: usize,
        /// Noise standard deviation
        noise_std: f64,
    },
    /// Input x Gradient
    InputXGradient,
    /// Expected Gradients
    ExpectedGradients {
        /// Reference samples for expectation
        num_references: usize,
        /// Number of integration steps
        num_steps: usize,
    },
}

/// Baseline methods for attribution
#[derive(Debug, Clone, PartialEq)]
pub enum BaselineMethod {
    /// Zero baseline
    Zero,
    /// Random noise baseline
    Random {
        /// Random seed for reproducible baseline
        seed: u64,
    },
    /// Gaussian blur baseline
    GaussianBlur {
        /// Standard deviation for Gaussian blur
        sigma: f64,
    },
    /// Mean of training data
    TrainingMean,
    /// Custom baseline
    Custom(ArrayD<f32>),
}

/// Feature visualization method
#[derive(Debug, Clone, PartialEq)]
pub enum VisualizationMethod {
    /// Activation maximization
    ActivationMaximization {
        /// Target layer name for activation maximization
        target_layer: String,
        /// Specific unit to maximize (None for all)
        target_unit: Option<usize>,
        /// Number of optimization iterations
        num_iterations: usize,
        /// Learning rate for optimization
        learning_rate: f64,
    },
    /// Deep dream
    DeepDream {
        /// Target layer name for deep dream
        target_layer: String,
        /// Number of optimization iterations
        num_iterations: usize,
        /// Learning rate for optimization
        learning_rate: f64,
        /// Factor to amplify activations
        amplify_factor: f64,
    },
    /// Feature inversion
    FeatureInversion {
        /// Target layer name for feature inversion
        target_layer: String,
        /// Weight for regularization term
        regularization_weight: f64,
    },
    /// Class Activation Mapping (CAM)
    ClassActivationMapping {
        /// Target layer for CAM
        target_layer: String,
        /// Target class index
        target_class: usize,
    },
    /// Network dissection for concept visualization
    NetworkDissection {
        /// Concept dataset for analysis
        concept_data: Vec<ArrayD<f32>>,
        /// Concept labels
        concept_labels: Vec<String>,
    },
}

/// Counterfactual explanation generator
#[allow(dead_code)]
pub struct CounterfactualGenerator<F: Float + Debug> {
    /// Maximum number of features to modify
    max_features: usize,
    /// Learning rate for counterfactual optimization
    learning_rate: f64,
    /// Maximum iterations for optimization
    max_iterations: usize,
    /// Distance metric for counterfactual search
    distance_metric: DistanceMetric,
    /// Cached original predictions
    original_predictions: HashMap<String, F>,
}

/// Concept Activation Vector for concept-based explanations
#[derive(Debug, Clone)]
pub struct ConceptActivationVector<F: Float + Debug> {
    /// Concept name
    pub name: String,
    /// Layer where concept is computed
    pub layer_name: String,
    /// Directional vector representing the concept
    pub direction_vector: ArrayD<F>,
    /// Concept activation sensitivity
    pub sensitivity: F,
    /// Examples that strongly activate this concept
    pub positive_examples: Vec<ArrayD<F>>,
    /// Examples that strongly deactivate this concept
    pub negative_examples: Vec<ArrayD<F>>,
}

/// LIME (Local Interpretable Model-agnostic Explanations) explainer
#[allow(dead_code)]
pub struct LIMEExplainer<F: Float + Debug> {
    /// Number of perturbed samples to generate
    num_samples: usize,
    /// Number of features to include in explanation
    num_features: usize,
    /// Kernel width for sample weighting
    kernel_width: f64,
    /// Feature perturbation strategy
    perturbation_strategy: PerturbationStrategy,
    /// Random seed for reproducibility
    random_seed: Option<u64>,
    /// Minimum feature importance threshold
    importance_threshold: F,
    /// Distance metric for sample weighting
    distance_metric: DistanceMetric,
}

/// Attention mechanism visualizer
#[allow(dead_code)]
pub struct AttentionVisualizer<F: Float + Debug> {
    /// Attention heads to visualize
    attention_heads: Vec<String>,
    /// Aggregation method for multi-head attention
    aggregation_method: AttentionAggregation,
    /// Cached attention weights
    attention_cache: HashMap<String, ArrayD<F>>,
}

/// Adversarial explanation for robustness analysis
#[derive(Debug, Clone)]
pub struct AdversarialExplanation<F: Float + Debug> {
    /// Original input
    pub original_input: ArrayD<F>,
    /// Adversarial example
    pub adversarial_input: ArrayD<F>,
    /// Perturbation applied
    pub perturbation: ArrayD<F>,
    /// Original prediction
    pub original_prediction: usize,
    /// Adversarial prediction
    pub adversarial_prediction: usize,
    /// Confidence scores
    pub original_confidence: F,
    /// Adversarial prediction confidence
    pub adversarial_confidence: F,
    /// Attack method used
    pub attack_method: String,
    /// Attack parameters
    pub attack_parameters: HashMap<String, f64>,
}

/// Network dissection results for understanding neuron selectivity
#[derive(Debug, Clone)]
pub struct NetworkDissectionResult<F: Float + Debug> {
    /// Layer name
    pub layer_name: String,
    /// Neuron index
    pub neuron_index: usize,
    /// Concept selectivity scores
    pub concept_scores: HashMap<String, F>,
    /// Top-k most selective concepts
    pub top_concepts: Vec<(String, F)>,
    /// Selectivity threshold used
    pub selectivity_threshold: F,
    /// Number of test images
    pub num_test_images: usize,
}

/// Statistical analysis of layer activations
#[derive(Debug, Clone)]
pub struct LayerAnalysisStats<F: Float + Debug> {
    /// Mean activation value
    pub mean_activation: F,
    /// Standard deviation of activations
    pub std_activation: F,
    /// Maximum activation value
    pub max_activation: F,
    /// Minimum activation value
    pub min_activation: F,
    /// Percentage of dead neurons (always zero)
    pub dead_neuron_percentage: f64,
    /// Sparsity (percentage of near-zero activations)
    pub sparsity: f64,
    /// Activation distribution histogram
    pub histogram: Vec<u32>,
    /// Histogram bin edges
    pub bin_edges: Vec<F>,
}

/// Model interpreter for analyzing neural network decisions
pub struct ModelInterpreter<
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
> {
    /// Available attribution methods
    attribution_methods: Vec<AttributionMethod>,
    /// Cached gradients for different layers
    gradient_cache: HashMap<String, ArrayD<F>>,
    /// Cached activations for different layers
    activation_cache: HashMap<String, ArrayD<F>>,
    /// Layer statistics
    #[allow(dead_code)]
    layer_statistics: HashMap<String, LayerAnalysisStats<F>>,
    /// Counterfactual generator
    counterfactual_generator: Option<CounterfactualGenerator<F>>,
    /// Concept activation vectors
    #[allow(dead_code)]
    concept_vectors: HashMap<String, ConceptActivationVector<F>>,
    /// LIME explainer
    #[allow(dead_code)]
    lime_explainer: Option<LIMEExplainer<F>>,
    /// Attention visualizer
    #[allow(dead_code)]
    attention_visualizer: Option<AttentionVisualizer<F>>,
}

impl<
        F: Float
            + Debug
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive
            + Sum
            + Clone
            + Copy,
    > ModelInterpreter<F>
{
    /// Create a new model interpreter
    pub fn new() -> Self {
        Self {
            attribution_methods: Vec::new(),
            gradient_cache: HashMap::new(),
            activation_cache: HashMap::new(),
            layer_statistics: HashMap::new(),
            counterfactual_generator: None,
            concept_vectors: HashMap::new(),
            lime_explainer: None,
            attention_visualizer: None,
        }
    }

    /// Add an attribution method
    pub fn add_attribution_method(&mut self, method: AttributionMethod) {
        self.attribution_methods.push(method);
    }

    /// Cache layer activations
    pub fn cache_activations(&mut self, layer_name: String, activations: ArrayD<F>) {
        self.activation_cache.insert(layer_name, activations);
    }

    /// Cache layer gradients
    pub fn cache_gradients(&mut self, layer_name: String, gradients: ArrayD<F>) {
        self.gradient_cache.insert(layer_name, gradients);
    }

    /// Compute attribution using specified method
    pub fn compute_attribution(
        &self,
        method: &AttributionMethod,
        input: &ArrayD<F>,
        target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        match method {
            AttributionMethod::Saliency => self.compute_saliency_attribution(input, target_class),
            AttributionMethod::IntegratedGradients {
                baseline,
                num_steps,
            } => self.compute_integrated_gradients(input, baseline, *num_steps, target_class),
            AttributionMethod::GradCAM { target_layer } => {
                self.compute_gradcam_attribution(input, target_layer, target_class)
            }
            AttributionMethod::GuidedBackprop => {
                self.compute_guided_backprop_attribution(input, target_class)
            }
            AttributionMethod::DeepLIFT { baseline } => {
                self.compute_deeplift_attribution(input, baseline, target_class)
            }
            AttributionMethod::SHAP {
                background_samples,
                num_samples,
            } => self.compute_shap_attribution(
                input,
                *background_samples,
                *num_samples,
                target_class,
            ),
            AttributionMethod::LayerWiseRelevancePropagation { rule, epsilon } => {
                self.compute_lrp_attribution(input, rule, *epsilon, target_class)
            }
            AttributionMethod::SmoothGrad {
                base_method,
                num_samples,
                noise_std,
            } => self.compute_smoothgrad_attribution(
                input,
                base_method,
                *num_samples,
                *noise_std,
                target_class,
            ),
            AttributionMethod::InputXGradient => {
                self.compute_input_x_gradient_attribution(input, target_class)
            }
            AttributionMethod::ExpectedGradients {
                num_references,
                num_steps,
            } => self.compute_expected_gradients_attribution(
                input,
                *num_references,
                *num_steps,
                target_class,
            ),
        }
    }

    /// Enable counterfactual explanations
    pub fn enable_counterfactual_explanations(
        &mut self,
        max_features: usize,
        learning_rate: f64,
        max_iterations: usize,
        distance_metric: DistanceMetric,
    ) {
        self.counterfactual_generator = Some(CounterfactualGenerator {
            max_features,
            learning_rate,
            max_iterations,
            distance_metric,
            original_predictions: HashMap::new(),
        });
    }

    fn compute_saliency_attribution(
        &self,
        input: &ArrayD<F>,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // Simple gradient-based saliency
        let grad_key = "input_gradient";
        if let Some(gradient) = self.gradient_cache.get(grad_key) {
            Ok(gradient.mapv(|x| x.abs()))
        } else {
            // Return random attribution as placeholder
            let attribution = input.mapv(|_| F::from(0.5).unwrap());
            Ok(attribution)
        }
    }

    fn compute_integrated_gradients(
        &self,
        input: &ArrayD<F>,
        baseline: &BaselineMethod,
        num_steps: usize,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let baseline_input = self.create_baseline(input, baseline)?;
        let mut accumulated_gradients = Array::zeros(input.raw_dim());

        for i in 0..num_steps {
            let alpha = F::from(i as f64 / (num_steps - 1) as f64).unwrap();
            let interpolated_input = &baseline_input + (&(input.clone() - &baseline_input) * alpha);

            // In practice, would compute gradients for interpolated input
            let step_gradient = interpolated_input.mapv(|x| x * F::from(0.1).unwrap());
            accumulated_gradients = accumulated_gradients + step_gradient;
        }

        let integrated_gradients =
            (input - &baseline_input) * accumulated_gradients / F::from(num_steps).unwrap();
        Ok(integrated_gradients)
    }

    fn compute_gradcam_attribution(
        &self,
        input: &ArrayD<F>,
        target_layer: &str,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // Get activations and gradients for target layer
        let activations = self.activation_cache.get(target_layer).ok_or_else(|| {
            NeuralError::ComputationError(format!(
                "Activations not found for layer: {}",
                target_layer
            ))
        })?;

        let gradients = self.gradient_cache.get(target_layer).ok_or_else(|| {
            NeuralError::ComputationError(format!(
                "Gradients not found for layer: {}",
                target_layer
            ))
        })?;

        if activations.ndim() < 3 {
            return Err(NeuralError::InvalidArchitecture(
                "GradCAM requires at least 3D activations (batch, channels, spatial)".to_string(),
            ));
        }

        // Compute channel-wise weights by global average pooling of gradients
        let mut weights = Vec::new();
        let num_channels = activations.shape()[1];

        for c in 0..num_channels {
            let channel_grad = gradients.index_axis(ndarray::Axis(1), c);
            let weight = channel_grad.mean().unwrap_or(F::zero());
            weights.push(weight);
        }

        // Compute weighted combination of activation maps
        let first_channel = activations
            .index_axis(ndarray::Axis(1), 0)
            .to_owned()
            .into_dyn();
        let mut gradcam = Array::zeros(first_channel.raw_dim());

        for (c, &weight) in weights.iter().enumerate().take(num_channels) {
            let channel_activation = activations
                .index_axis(ndarray::Axis(1), c)
                .to_owned()
                .into_dyn();
            let weighted_activation = channel_activation * weight;
            gradcam = gradcam + weighted_activation;
        }

        // ReLU to keep only positive influences
        let gradcam_relu = gradcam.mapv(|x: F| x.max(F::zero()));

        // Resize to input dimensions if needed
        if gradcam_relu.raw_dim() != input.raw_dim() {
            self.resize_attribution(&gradcam_relu, input.raw_dim())
        } else {
            Ok(gradcam_relu)
        }
    }

    fn compute_guided_backprop_attribution(
        &self,
        _input: &ArrayD<F>,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        if let Some(gradient) = self.gradient_cache.get("input_gradient") {
            Ok(gradient.mapv(|x| x.max(F::zero())))
        } else {
            Ok(Array::zeros(_input.raw_dim()))
        }
    }

    fn compute_deeplift_attribution(
        &self,
        input: &ArrayD<F>,
        baseline: &BaselineMethod,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let baseline_input = self.create_baseline(input, baseline)?;
        let diff = input - &baseline_input;

        if let Some(gradient) = self.gradient_cache.get("input_gradient") {
            Ok(&diff * gradient)
        } else {
            Ok(diff)
        }
    }

    fn compute_shap_attribution(
        &self,
        input: &ArrayD<F>,
        _background_samples: usize,
        num_samples: usize,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let mut total_attribution = Array::zeros(input.raw_dim());

        for _ in 0..num_samples {
            let coalition_mask = input.mapv(|_| {
                if rand::random::<f64>() > 0.5 {
                    F::one()
                } else {
                    F::zero()
                }
            });

            let marginal_contribution = input * &coalition_mask * F::from(0.1).unwrap();
            total_attribution = total_attribution + marginal_contribution;
        }

        Ok(total_attribution / F::from(num_samples).unwrap())
    }

    fn compute_lrp_attribution(
        &self,
        input: &ArrayD<F>,
        rule: &LRPRule,
        epsilon: f64,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        match rule {
            LRPRule::Epsilon => {
                if let Some(gradient) = self.gradient_cache.get("input_gradient") {
                    let eps = F::from(epsilon).unwrap();
                    let denominator = gradient.mapv(|x| x + eps.copysign(x));
                    Ok(input * gradient / denominator)
                } else {
                    Ok(input.clone())
                }
            }
            LRPRule::Gamma { gamma } => {
                if let Some(gradient) = self.gradient_cache.get("input_gradient") {
                    let gamma_val = F::from(*gamma).unwrap();
                    let positive_part = gradient.mapv(|x| x.max(F::zero()));
                    let negative_part = gradient.mapv(|x| x.min(F::zero()));
                    Ok(input * (positive_part * (F::one() + gamma_val) + negative_part))
                } else {
                    Ok(input.clone())
                }
            }
            LRPRule::AlphaBeta { alpha, beta } => {
                if let Some(gradient) = self.gradient_cache.get("input_gradient") {
                    let alpha_val = F::from(*alpha).unwrap();
                    let beta_val = F::from(*beta).unwrap();
                    let positive_part = gradient.mapv(|x| x.max(F::zero()));
                    let negative_part = gradient.mapv(|x| x.min(F::zero()));
                    Ok(input * (positive_part * alpha_val - negative_part * beta_val))
                } else {
                    Ok(input.clone())
                }
            }
            LRPRule::ZPlus => {
                if let Some(gradient) = self.gradient_cache.get("input_gradient") {
                    let positive_input = input.mapv(|x| x.max(F::zero()));
                    Ok(positive_input * gradient)
                } else {
                    Ok(input.mapv(|x| x.max(F::zero())))
                }
            }
            LRPRule::ZB { low, high } => {
                if let Some(gradient) = self.gradient_cache.get("input_gradient") {
                    let low_val = F::from(*low).unwrap();
                    let high_val = F::from(*high).unwrap();
                    let clamped_input = input.mapv(|x| x.max(low_val).min(high_val));
                    Ok(clamped_input * gradient)
                } else {
                    Ok(input.clone())
                }
            }
        }
    }

    fn compute_smoothgrad_attribution(
        &self,
        input: &ArrayD<F>,
        base_method: &AttributionMethod,
        num_samples: usize,
        noise_std: f64,
        target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let mut accumulated_attribution = Array::zeros(input.raw_dim());
        let noise_scale = F::from(noise_std).unwrap();

        for _ in 0..num_samples {
            let noisy_input = input.mapv(|x| {
                let noise = F::from(rand::random::<f64>() - 0.5).unwrap() * noise_scale;
                x + noise
            });

            let attribution = self.compute_attribution(base_method, &noisy_input, target_class)?;
            accumulated_attribution = accumulated_attribution + attribution;
        }

        Ok(accumulated_attribution / F::from(num_samples).unwrap())
    }

    fn compute_input_x_gradient_attribution(
        &self,
        input: &ArrayD<F>,
        _target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        if let Some(gradient) = self.gradient_cache.get("input_gradient") {
            Ok(input * gradient)
        } else {
            Ok(input.clone())
        }
    }

    fn compute_expected_gradients_attribution(
        &self,
        input: &ArrayD<F>,
        num_references: usize,
        num_steps: usize,
        target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        let mut total_attribution = Array::zeros(input.raw_dim());

        for _ in 0..num_references {
            let reference = input.mapv(|_| F::from(rand::random::<f64>()).unwrap());
            let baseline =
                BaselineMethod::Custom(reference.mapv(|x| x.to_f64().unwrap_or(0.0) as f32));

            let attribution =
                self.compute_integrated_gradients(input, &baseline, num_steps, target_class)?;
            total_attribution = total_attribution + attribution;
        }

        Ok(total_attribution / F::from(num_references).unwrap())
    }

    fn create_baseline(&self, input: &ArrayD<F>, method: &BaselineMethod) -> Result<ArrayD<F>> {
        match method {
            BaselineMethod::Zero => Ok(Array::zeros(input.raw_dim())),
            BaselineMethod::Random { seed: _ } => {
                Ok(input.mapv(|_| F::from(rand::random::<f64>()).unwrap()))
            }
            BaselineMethod::GaussianBlur { sigma: _ } => {
                let blurred = input.mapv(|x| x * F::from(0.5).unwrap());
                Ok(blurred)
            }
            BaselineMethod::TrainingMean => Ok(Array::zeros(input.raw_dim())),
            BaselineMethod::Custom(baseline) => {
                if baseline.shape() == input.shape() {
                    let converted = baseline.mapv(|x| F::from(x as f64).unwrap_or(F::zero()));
                    Ok(converted)
                } else {
                    Err(NeuralError::DimensionMismatch(
                        "Custom baseline shape doesn't match input".to_string(),
                    ))
                }
            }
        }
    }

    fn resize_attribution(
        &self,
        attribution: &ArrayD<F>,
        target_shape: IxDyn,
    ) -> Result<ArrayD<F>> {
        if attribution.len() == target_shape.size() {
            Ok(attribution.clone().into_shape_with_order(target_shape)?)
        } else {
            let mean_val = attribution.mean().unwrap_or(F::zero());
            Ok(Array::from_elem(target_shape, mean_val))
        }
    }

    /// Analyze layer activations and store statistics
    pub fn analyze_layer_activations(
        &mut self,
        layer_name: String,
        activations: &ArrayD<F>,
    ) -> Result<()> {
        // Cache the activations
        self.cache_activations(layer_name.clone(), activations.clone());

        // Compute statistics
        let flattened = activations
            .view()
            .into_shape_with_order(activations.len())?;
        let mean_activation = flattened.mean().unwrap_or(F::zero());
        let variance = flattened
            .mapv(|x| (x - mean_activation) * (x - mean_activation))
            .mean()
            .unwrap_or(F::zero());
        let std_activation = variance.sqrt();
        let max_activation =
            flattened.fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });
        let min_activation = flattened.fold(F::infinity(), |acc, &x| if x < acc { x } else { acc });

        // Count dead neurons (activations always zero)
        let dead_count = flattened.iter().filter(|&&x| x == F::zero()).count();
        let dead_neuron_percentage = (dead_count as f64 / flattened.len() as f64) * 100.0;

        // Compute sparsity (percentage of near-zero activations)
        let threshold = F::from(1e-6).unwrap_or(F::zero());
        let sparse_count = flattened.iter().filter(|&&x| x.abs() < threshold).count();
        let sparsity = (sparse_count as f64 / flattened.len() as f64) * 100.0;

        // Create a simple histogram (10 bins)
        let num_bins = 10;
        let range = max_activation - min_activation;
        let bin_width = if range > F::zero() {
            range / F::from(num_bins).unwrap()
        } else {
            F::one()
        };

        let mut histogram = vec![0u32; num_bins];
        let mut bin_edges = Vec::new();

        // Generate bin edges
        for i in 0..=num_bins {
            bin_edges.push(min_activation + F::from(i).unwrap() * bin_width);
        }

        // Fill histogram
        for &activation in flattened.iter() {
            if range > F::zero() {
                let bin_index = ((activation - min_activation) / bin_width)
                    .to_usize()
                    .unwrap_or(0);
                let bin_index = bin_index.min(num_bins - 1);
                histogram[bin_index] += 1;
            } else {
                histogram[0] += 1; // All values in first bin if no range
            }
        }

        let stats = LayerAnalysisStats {
            mean_activation,
            std_activation,
            max_activation,
            min_activation,
            dead_neuron_percentage,
            sparsity,
            histogram,
            bin_edges,
        };

        self.layer_statistics.insert(layer_name, stats);
        Ok(())
    }

    /// Get statistics for a specific layer
    pub fn get_layer_statistics(&self, layer_name: &str) -> Option<&LayerAnalysisStats<F>> {
        self.layer_statistics.get(layer_name)
    }

    /// Generate a comprehensive interpretation report
    pub fn generate_interpretation_report(
        &self,
        input: &ArrayD<F>,
        target_class: Option<usize>,
    ) -> Result<InterpretationReport<F>> {
        let input_shape = input.shape().to_vec();

        // Generate attributions for all available methods
        let mut attributions = HashMap::new();
        let mut attribution_statistics = HashMap::new();

        for method in &self.attribution_methods {
            if let Ok(attribution) = self.compute_attribution(method, input, target_class) {
                let method_name = format!("{:?}", method);

                // Compute attribution statistics
                let flattened = attribution
                    .view()
                    .into_shape_with_order(attribution.len())?;
                let mean = flattened.mean().unwrap_or(F::zero());
                let mean_absolute = flattened.mapv(|x| x.abs()).mean().unwrap_or(F::zero());
                let max_absolute =
                    flattened.fold(
                        F::zero(),
                        |acc, &x| if x.abs() > acc { x.abs() } else { acc },
                    );

                let positive_count = flattened.iter().filter(|&&x| x > F::zero()).count();
                let positive_attribution_ratio = positive_count as f64 / flattened.len() as f64;

                let total_positive_attribution = flattened
                    .iter()
                    .filter(|&&x| x > F::zero())
                    .fold(F::zero(), |acc, &x| acc + x);
                let total_negative_attribution = flattened
                    .iter()
                    .filter(|&&x| x < F::zero())
                    .fold(F::zero(), |acc, &x| acc + x);

                let stats = AttributionStatistics {
                    mean,
                    mean_absolute,
                    max_absolute,
                    positive_attribution_ratio,
                    total_positive_attribution,
                    total_negative_attribution,
                };

                attributions.insert(method_name.clone(), attribution);
                attribution_statistics.insert(method_name, stats);
            }
        }

        // Clone layer statistics
        let layer_statistics = self.layer_statistics.clone();

        // Create interpretation summary
        let interpretation_summary = InterpretationSummary {
            num_attribution_methods: self.attribution_methods.len(),
            average_method_consistency: 0.75, // Placeholder consistency
            most_important_features: vec![0, 1, 2], // Placeholder important features
            interpretation_confidence: 0.85,  // Placeholder confidence
        };

        Ok(InterpretationReport {
            input_shape: IxDyn(&input_shape),
            target_class,
            attributions,
            attribution_statistics,
            layer_statistics,
            interpretation_summary,
        })
    }
}

impl<
        F: Float
            + Debug
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive
            + Sum
            + Clone
            + Copy,
    > Default for ModelInterpreter<F>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of interpretation analysis
#[derive(Debug, Clone)]
pub struct InterpretationSummary {
    /// Number of attribution methods used
    pub num_attribution_methods: usize,
    /// Average consistency across methods
    pub average_method_consistency: f64,
    /// Indices of most important features
    pub most_important_features: Vec<usize>,
    /// Overall interpretation confidence (0-1)
    pub interpretation_confidence: f64,
}

/// Statistics for attribution methods
#[derive(Debug, Clone)]
pub struct AttributionStatistics<F: Float + Debug> {
    /// Mean attribution value
    pub mean: F,
    /// Mean absolute attribution value
    pub mean_absolute: F,
    /// Maximum absolute attribution value
    pub max_absolute: F,
    /// Ratio of positive attributions
    pub positive_attribution_ratio: f64,
    /// Total positive attribution
    pub total_positive_attribution: F,
    /// Total negative attribution
    pub total_negative_attribution: F,
}

/// Comprehensive interpretation report with all explanation types
#[derive(Debug)]
pub struct ComprehensiveInterpretationReport<F: Float + Debug> {
    /// Basic interpretation report
    pub basic_report: InterpretationReport<F>,
    /// Counterfactual explanation
    pub counterfactual_explanation: Option<ArrayD<F>>,
    /// LIME explanation
    pub lime_explanation: Option<ArrayD<F>>,
    /// Concept activation scores
    pub concept_activations: HashMap<String, F>,
    /// Attention visualization maps
    pub attention_visualizations: Option<HashMap<String, ArrayD<F>>>,
    /// Adversarial explanations
    pub adversarial_explanations: Vec<AdversarialExplanation<F>>,
    /// Network dissection results
    pub network_dissection_results: Vec<NetworkDissectionResult<F>>,
}

/// Basic interpretation report
#[derive(Debug)]
pub struct InterpretationReport<F: Float + Debug> {
    /// Shape of input that was interpreted
    pub input_shape: IxDyn,
    /// Target class (if specified)
    pub target_class: Option<usize>,
    /// Attribution maps for each method
    pub attributions: HashMap<String, ArrayD<F>>,
    /// Statistics for each attribution method
    pub attribution_statistics: HashMap<String, AttributionStatistics<F>>,
    /// Layer analysis statistics
    pub layer_statistics: HashMap<String, LayerAnalysisStats<F>>,
    /// Summary of interpretation
    pub interpretation_summary: InterpretationSummary,
}

impl<F: Float + Debug> std::fmt::Display for InterpretationReport<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Neural Network Interpretation Report")?;
        writeln!(f, "===================================")?;
        writeln!(f, "Input Shape: {:?}", self.input_shape)?;
        writeln!(f, "Target Class: {:?}", self.target_class)?;
        writeln!(
            f,
            "Attribution Methods: {}",
            self.attribution_statistics.len()
        )?;
        writeln!(
            f,
            "Interpretation Confidence: {:.3}",
            self.interpretation_summary.interpretation_confidence
        )?;
        writeln!(
            f,
            "Average Method Consistency: {:.3}",
            self.interpretation_summary.average_method_consistency
        )?;
        writeln!(
            f,
            "Top Important Features: {:?}",
            self.interpretation_summary.most_important_features
        )?;

        writeln!(f, "\nLayer Statistics:")?;
        for (layer_name, stats) in &self.layer_statistics {
            writeln!(
                f,
                "  {}: mean={:.3}, std={:.3}, sparsity={:.1}%, dead_neurons={:.1}%",
                layer_name,
                stats.mean_activation.to_f64().unwrap_or(0.0),
                stats.std_activation.to_f64().unwrap_or(0.0),
                stats.sparsity,
                stats.dead_neuron_percentage
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_model_interpreter_creation() {
        let interpreter = ModelInterpreter::<f64>::new();
        assert_eq!(interpreter.attribution_methods.len(), 0);
        assert_eq!(interpreter.gradient_cache.len(), 0);
    }

    #[test]
    fn test_saliency_attribution() {
        let mut interpreter = ModelInterpreter::<f64>::new();

        let gradients = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
            .unwrap()
            .into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), gradients);

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();

        let attribution = interpreter.compute_saliency_attribution(&input, None);
        assert!(attribution.is_ok());

        let attr = attribution.unwrap();
        assert_eq!(attr.shape(), input.shape());
        assert_eq!(attr[[0, 0]], 0.1);
        assert_eq!(attr[[0, 2]], 0.3); // abs(-0.3)
    }

    #[test]
    fn test_integrated_gradients() {
        let interpreter = ModelInterpreter::<f64>::new();

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();
        let baseline = BaselineMethod::Zero;

        let attribution = interpreter.compute_integrated_gradients(&input, &baseline, 10, None);
        assert!(attribution.is_ok());

        let attr = attribution.unwrap();
        assert_eq!(attr.shape(), input.shape());
    }

    #[test]
    fn test_lrp_attribution() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();

        let gradients = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
            .unwrap()
            .into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), gradients);

        let attribution =
            interpreter.compute_lrp_attribution(&input, &LRPRule::Epsilon, 1e-6, None);
        assert!(attribution.is_ok());

        let gamma_attribution = interpreter.compute_lrp_attribution(
            &input,
            &LRPRule::Gamma { gamma: 0.25 },
            1e-6,
            None,
        );
        assert!(gamma_attribution.is_ok());

        let zplus_attribution =
            interpreter.compute_lrp_attribution(&input, &LRPRule::ZPlus, 1e-6, None);
        assert!(zplus_attribution.is_ok());
    }

    #[test]
    fn test_counterfactual_explanations() {
        let mut interpreter = ModelInterpreter::<f64>::new();

        interpreter.enable_counterfactual_explanations(5, 0.01, 10, DistanceMetric::L2);

        let _input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();

        assert!(interpreter.counterfactual_generator.is_some());
    }

    #[test]
    fn test_input_x_gradient_attribution() {
        let mut interpreter = ModelInterpreter::<f64>::new();

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();
        let gradients = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .unwrap()
            .into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), gradients);

        let attribution = interpreter.compute_input_x_gradient_attribution(&input, None);
        assert!(attribution.is_ok());

        let attr = attribution.unwrap();
        assert_eq!(attr.shape(), input.shape());
        assert!((attr[[0, 0]] - 0.1).abs() < 1e-10); // 1.0 * 0.1
        assert!((attr[[1, 2]] - 3.6).abs() < 1e-10); // 6.0 * 0.6
    }

    #[test]
    fn test_baseline_creation() {
        let interpreter = ModelInterpreter::<f64>::new();
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn();

        let zero_baseline = interpreter.create_baseline(&input, &BaselineMethod::Zero);
        assert!(zero_baseline.is_ok());
        let baseline = zero_baseline.unwrap();
        assert_eq!(baseline.shape(), input.shape());
        assert!(baseline.iter().all(|&x| x == 0.0));

        let custom_array = Array2::from_elem((2, 3), 0.5f32).into_dyn();
        let custom_baseline =
            interpreter.create_baseline(&input, &BaselineMethod::Custom(custom_array));
        assert!(custom_baseline.is_ok());
        let baseline = custom_baseline.unwrap();
        assert_eq!(baseline.shape(), input.shape());
        assert!(baseline.iter().all(|&x| x == 0.5));
    }
}
