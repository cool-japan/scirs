//! Core model interpretation orchestrator and shared utilities
//!
//! This module provides the main ModelInterpreter struct that serves as the central
//! orchestrator for all interpretation methods, managing caches, configurations,
//! and coordinating between different interpretation techniques.

use crate::error::Result;
use scirs2_core::ndarray::ArrayD;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;
// Re-export types from other modules
pub use super::analysis::LayerAnalysisStats;
pub use super::attribution::{AttributionMethod, BaselineMethod, LRPRule};
pub use super::explanations::{ConceptActivationVector, CounterfactualGenerator, LIMEExplainer};
pub use super::reporting::{ComprehensiveInterpretationReport, InterpretationReport};
pub use super::visualization::{AttentionVisualizer, VisualizationMethod};
/// Model interpreter for analyzing neural network decisions
///
/// This is the main orchestrator for all interpretation methods. It manages:
/// - Attribution method registration and dispatch
/// - Layer activation and gradient caching
/// - Integration between different interpretation techniques
/// - Unified interface for model analysis
pub struct ModelInterpreter<
    F: Float
        + Debug
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + scirs2_core::numeric::FromPrimitive
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
    layer_statistics: HashMap<String, LayerAnalysisStats<F>>,
    /// Counterfactual generator
    counterfactual_generator: Option<CounterfactualGenerator<F>>,
    /// Concept activation vectors
    concept_vectors: HashMap<String, ConceptActivationVector<F>>,
    /// LIME explainer
    lime_explainer: Option<LIMEExplainer<F>>,
    /// Attention visualizer
    attention_visualizer: Option<AttentionVisualizer<F>>,
}
impl<
        F: Float
            + Debug
            + 'static
            + scirs2_core::ndarray::ScalarOperand
            + scirs2_core::numeric::FromPrimitive
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
    /// Add an attribution method to the interpreter
    pub fn add_attribution_method(&mut self, method: AttributionMethod) {
        self.attribution_methods.push(method);
    /// Cache layer activations for later analysis
    pub fn cache_activations(&mut self, layername: String, activations: ArrayD<F>) {
        self.activation_cache.insert(layer_name, activations);
    /// Cache layer gradients for attribution computation
    pub fn cache_gradients(&mut self, layername: String, gradients: ArrayD<F>) {
        self.gradient_cache.insert(layer_name, gradients);
    /// Get cached activations for a layer
    pub fn get_cached_activations(&self, layername: &str) -> Option<&ArrayD<F>> {
        self.activation_cache.get(layer_name)
    /// Get cached gradients for a layer
    pub fn get_cached_gradients(&self, layername: &str) -> Option<&ArrayD<F>> {
        self.gradient_cache.get(layer_name)
    /// Clear all caches
    pub fn clear_caches(&mut self) {
        self.gradient_cache.clear();
        self.activation_cache.clear();
        self.layer_statistics.clear();
    /// Get available attribution methods
    pub fn attribution_methods(&self) -> &[AttributionMethod] {
        &self.attribution_methods
    /// Check if a specific layer has cached data
    pub fn has_layer_data(&self, layername: &str) -> bool {
        self.activation_cache.contains_key(layer_name)
            || self.gradient_cache.contains_key(layer_name)
    /// Get all cached layer names
    pub fn cached_layers(&self) -> Vec<String> {
        let mut layers: std::collections::HashSet<String> = std::collections::HashSet::new();
        layers.extend(self.activation_cache.keys().cloned());
        layers.extend(self.gradient_cache.keys().cloned());
        layers.into_iter().collect()
    /// Set the counterfactual generator
    pub fn set_counterfactual_generator(&mut self, generator: CounterfactualGenerator<F>) {
        self.counterfactual_generator = Some(generator);
    /// Get the counterfactual generator
    pub fn counterfactual_generator(&self) -> Option<&CounterfactualGenerator<F>> {
        self.counterfactual_generator.as_ref()
    /// Set the LIME explainer
    pub fn set_lime_explainer(&mut self, explainer: LIMEExplainer<F>) {
        self.lime_explainer = Some(explainer);
    /// Get the LIME explainer
    pub fn lime_explainer(&self) -> Option<&LIMEExplainer<F>> {
        self.lime_explainer.as_ref()
    /// Set the attention visualizer
    pub fn set_attention_visualizer(&mut self, visualizer: AttentionVisualizer<F>) {
        self.attention_visualizer = Some(visualizer);
    /// Get the attention visualizer
    pub fn attention_visualizer(&self) -> Option<&AttentionVisualizer<F>> {
        self.attention_visualizer.as_ref()
    /// Add concept activation vector
    pub fn add_concept_vector(&mut self, name: String, vector: ConceptActivationVector<F>) {
        self.concept_vectors.insert(name, vector);
    /// Get concept activation vector
    pub fn get_concept_vector(&self, name: &str) -> Option<&ConceptActivationVector<F>> {
        self.concept_vectors.get(name)
    /// Get layer statistics
    pub fn layer_statistics(&self) -> &HashMap<String, LayerAnalysisStats<F>> {
        &self.layer_statistics
    /// Cache layer statistics
    pub fn cache_layer_statistics(&mut self, layername: String, stats: LayerAnalysisStats<F>) {
        self.layer_statistics.insert(layer_name, stats);
    /// Compute attribution using specified method
    ///
    /// This is the main dispatch method that delegates to specific attribution
    /// implementations in the attribution module.
    pub fn compute_attribution(
        &self,
        method: &AttributionMethod,
        input: &ArrayD<F>,
        target_class: Option<usize>,
    ) -> Result<ArrayD<F>> {
        // Import attribution computation functions
        use super::attribution::{
            compute_deeplift_attribution, compute_expected_gradients_attribution,
            compute_gradcam_attribution, compute_guided_backprop_attribution,
            compute_input_x_gradient_attribution, compute_integrated_gradients,
            compute_lrp_attribution, compute_saliency_attribution,
            compute_shap_attribution, compute_smoothgrad_attribution,
        };
        match method {
            AttributionMethod::Saliency => compute_saliency_attribution(self, input, target_class),
            AttributionMethod::IntegratedGradients {
                baseline,
                num_steps,
            } => compute_integrated_gradients(self, input, baseline, *num_steps, target_class),
            AttributionMethod::GradCAM { target_layer } => {
                compute_gradcam_attribution(self, input, target_layer, target_class)
            }
            AttributionMethod::GuidedBackprop => {
                compute_guided_backprop_attribution(self, input, target_class)
            AttributionMethod::DeepLIFT { baseline } => {
                compute_deeplift_attribution(self, input, baseline, target_class)
            AttributionMethod::SHAP {
                background_samples,
                num_samples,
            } => compute_shap_attribution(
                self,
                input,
                *background_samples,
                *num_samples,
                target_class,
            ),
            AttributionMethod::LayerWiseRelevancePropagation { rule, epsilon } => {
                compute_lrp_attribution(self, input, rule, *epsilon, target_class)
            AttributionMethod::SmoothGrad {
                base_method,
                noise_std,
            } => compute_smoothgrad_attribution(
                *noise_std,
            AttributionMethod::InputXGradient => {
                compute_input_x_gradient_attribution(self, input, target_class)
            AttributionMethod::ExpectedGradients {
                num_references,
            } => compute_expected_gradients_attribution(
                *num_references,
                *num_steps,
    /// Analyze layer activations
    /// Delegates to the analysis module for detailed layer analysis.
    pub fn analyze_layer_activations(&mut self, layername: &str) -> Result<LayerAnalysisStats<F>> {
        use super::analysis::analyze_layer_activations;
        analyze_layer_activations(self, layer_name)
    /// Generate comprehensive interpretation report
    /// Delegates to the reporting module for unified reporting.
    pub fn generate_report(
    ) -> Result<ComprehensiveInterpretationReport<F>> {
        use super::reporting::generate_comprehensive_report;
        generate_comprehensive_report(self, input)
    > Default for ModelInterpreter<F>
    fn default() -> Self {
        Self::new()
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;
    #[test]
    fn test_model_interpreter_creation() {
        let interpreter: ModelInterpreter<f64> = ModelInterpreter::new();
        assert_eq!(interpreter.attribution_methods().len(), 0);
        assert_eq!(interpreter.cached_layers().len(), 0);
    fn test_cache_management() {
        let mut interpreter: ModelInterpreter<f64> = ModelInterpreter::new();
        let activations = Array::zeros((2, 3, 4)).into_dyn();
        let gradients = Array::ones((2, 3, 4)).into_dyn();
        interpreter.cache_activations("conv1".to_string(), activations.clone());
        interpreter.cache_gradients("conv1".to_string(), gradients.clone());
        assert!(interpreter.has_layer_data("conv1"));
        assert!(!interpreter.has_layer_data("conv2"));
        let cached_layers = interpreter.cached_layers();
        assert_eq!(cached_layers.len(), 1);
        assert!(cached_layers.contains(&"conv1".to_string()));
        interpreter.clear_caches();
    fn test_attribution_method_management() {
        let method = AttributionMethod::Saliency;
        interpreter.add_attribution_method(method);
        assert_eq!(interpreter.attribution_methods().len(), 1);
        assert_eq!(
            interpreter.attribution_methods()[0],
            AttributionMethod::Saliency
        );
    fn test_concept_vector_management() {
        // This would normally be a real ConceptActivationVector, but we'll use a placeholder
        // The actual struct will be defined in the explanations module
        // For now, just test the interface exists
        assert!(interpreter.get_concept_vector("test").is_none());
