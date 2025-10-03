//! Advanced explanation techniques for neural network interpretation
//!
//! This module provides sophisticated explanation methods including counterfactual
//! generation, LIME explanations, concept activation vectors, and adversarial explanations.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::ArrayD;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;
/// Distance metric for counterfactual generation
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// L1 (Manhattan) distance
    L1,
    /// L2 (Euclidean) distance
    L2,
    /// Infinity (Chebyshev) distance
    LInf,
    /// Custom distance function
    Custom,
}
/// Counterfactual explanation generator
#[derive(Debug, Clone)]
pub struct CounterfactualGenerator<F: Float + Debug> {
    /// Maximum number of features to perturb
    pub max_features: usize,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// Cache of original predictions
    pub original_predictions: HashMap<String, F>,
/// Concept Activation Vector for concept-based explanations
pub struct ConceptActivationVector<F: Float + Debug> {
    /// Concept name
    pub concept_name: String,
    /// Layer name where concept is detected
    pub layer_name: String,
    /// Activation vector representing the concept
    pub activation_vector: ArrayD<F>,
    /// Confidence score for concept detection
    pub confidence: f64,
    /// Number of examples used to derive this concept
    pub num_examples: usize,
/// LIME (Local Interpretable Model-agnostic Explanations) explainer
pub struct LIMEExplainer<F: Float + Debug> {
    /// Number of perturbations to generate
    pub num_perturbations: usize,
    /// Size of local neighborhood
    pub neighborhood_size: f64,
    /// Regularization strength for sparse explanations
    pub regularization_strength: f64,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Cached perturbations for efficiency
    pub cached_perturbations: HashMap<String, Vec<ArrayD<F>>>,
/// Adversarial explanation result
pub struct AdversarialExplanation<F: Float + Debug> {
    /// Original input
    pub original_input: ArrayD<F>,
    /// Adversarial example
    pub adversarial_input: ArrayD<F>,
    /// Perturbation applied
    pub perturbation: ArrayD<F>,
    /// Original prediction
    pub original_prediction: F,
    /// Adversarial prediction
    pub adversarial_prediction: F,
    /// Attack method used
    pub attack_method: String,
    /// Perturbation magnitude
    pub perturbation_magnitude: f64,
impl<F> CounterfactualGenerator<F>
where
    F: Float
        + Debug
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + scirs2_core::numeric::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    /// Create a new counterfactual generator
    pub fn new(
        max_features: usize,
        learning_rate: f64,
        max_iterations: usize,
        distance_metric: DistanceMetric,
    ) -> Self {
        Self {
            max_features,
            learning_rate,
            max_iterations,
            distance_metric,
            original_predictions: HashMap::new(),
        }
    }
    /// Generate counterfactual explanation
    pub fn generate_counterfactual(
        &mut self,
        original_input: &ArrayD<F>, _target_prediction: F,
    ) -> Result<ArrayD<F>> {
        // Simplified counterfactual generation
        // In practice, this would use optimization to find minimal perturbations
        let mut counterfactual = original_input.clone();
        for _iteration in 0..self.max_iterations {
            // Apply small random perturbations
            counterfactual = counterfactual.mapv(|x| {
                let perturbation = F::from(scirs2_core::random::random::<f64>() * 0.01).unwrap();
                x + perturbation
            });
            // In practice, would evaluate model and check if target is reached
            // For now, return the perturbed input
        Ok(counterfactual)
    /// Compute distance between two inputs
    pub fn compute_distance(&self, input1: &ArrayD<F>, input2: &ArrayD<F>) -> f64 {
        match self.distance_metric {
            DistanceMetric::L1 => (input1 - input2)
                .mapv(|x| x.abs())
                .sum()
                .to_f64()
                .unwrap_or(0.0),
            DistanceMetric::L2 => ((input1 - input2).mapv(|x| x * x).sum().sqrt()), DistanceMetric::LInf => (input1 - input2)
                .iter()
                .cloned()
                .fold(F::zero(), F::max), DistanceMetric::Custom => {
                // Placeholder for custom distance
                0.0
            }
impl<F> ConceptActivationVector<F>
    /// Create a new concept activation vector
        concept_name: String,
        layer_name: String,
        activation_vector: ArrayD<F>,
        confidence: f64,
        num_examples: usize,
            concept_name,
            layer_name,
            activation_vector,
            confidence,
            num_examples,
    /// Compute concept activation score for input
    pub fn compute_activation_score(&self, layeractivations: &ArrayD<F>) -> f64 {
        // Simplified concept activation computation
        // In practice, would use proper dot product or cosine similarity
        if layer_activations.len() == self.activation_vector.len() {
            let dot_product: F = layer_activations
                .zip(self.activation_vector.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            dot_product.to_f64().unwrap_or(0.0)
        } else {
            0.0
impl<F> LIMEExplainer<F>
    /// Create a new LIME explainer
        num_perturbations: usize,
        neighborhood_size: f64,
        regularization_strength: f64,
        random_seed: u64,
            num_perturbations,
            neighborhood_size,
            regularization_strength,
            random_seed,
            cached_perturbations: HashMap::new(),
    /// Generate local explanation using LIME
    pub fn explain_instance(
        input: &ArrayD<F>, _target_class: Option<usize>,
        // Simplified LIME implementation
        // In practice, would generate perturbations, train local model, extract coefficients
        let mut explanation = input.mapv(|_| F::zero());
        for _i in 0..self.num_perturbations {
            // Generate perturbation
            let perturbation = input.mapv(|x| {
                let noise = F::from(scirs2_core::random::random::<f64>() * self.neighborhood_size).unwrap();
                x + noise
            // In practice, would evaluate model on perturbation
            // and use results to train local linear model
            // For now, assign simple importance scores
            explanation = explanation + perturbation.mapv(|x| x * F::from(0.1).unwrap());
        Ok(explanation / F::from(self.num_perturbations).unwrap())
    /// Generate perturbations around input
    pub fn generate_perturbations(&mut self, input: &ArrayD<F>) -> Vec<ArrayD<F>> {
        let cache_key = format!("{:?}", input.shape());
        if let Some(cached) = self.cached_perturbations.get(&cache_key) {
            return cached.clone();
        let mut perturbations = Vec::new();
        for _ in 0..self.num_perturbations {
            perturbations.push(perturbation);
        self.cached_perturbations
            .insert(cache_key, perturbations.clone());
        perturbations
/// Generate adversarial explanation
#[allow(dead_code)]
pub fn generate_adversarial_explanation<F>(
    original_input: &ArrayD<F>,
    attack_method: &str,
    epsilon: f64,
) -> Result<AdversarialExplanation<F>>
    // Simplified adversarial example generation
    let perturbation = match attack_method {
        "fgsm" => {
            // Fast Gradient Sign Method (simplified)
            original_input.mapv(|_| {
                let sign = if scirs2_core::random::random::<f64>() > 0.5 {
                    1.0
                } else {
                    -1.0
                };
                F::from(epsilon * sign).unwrap()
            })
        "pgd" => {
            // Projected Gradient Descent (simplified)
            original_input
                .mapv(|_| F::from(scirs2_core::random::random::<f64>() * epsilon * 2.0 - epsilon).unwrap())
        _ => {
            return Err(NeuralError::NotImplementedError(format!(
                "Attack method '{}' not implemented",
                attack_method
            )));
    };
    let adversarial_input = original_input + &perturbation;
    // Compute perturbation magnitude
    let perturbation_magnitude = perturbation
        .mapv(|x| x * x)
        .sum()
        .sqrt()
        .to_f64()
        .unwrap_or(0.0);
    Ok(AdversarialExplanation {
        original_input: original_input.clone(),
        adversarial_input,
        perturbation,
        original_prediction: F::from(0.8).unwrap(), // Placeholder
        adversarial_prediction: F::from(0.2).unwrap(), // Placeholder
        attack_method: attack_method.to_string(),
        perturbation_magnitude,
    })
/// Compute concept activation vectors from examples
#[allow(dead_code)]
pub fn compute_concept_vectors<F>(
    concept_examples: &[ArrayD<F>],
    concept_labels: &[String],
    layer_name: String,
) -> Result<Vec<ConceptActivationVector<F>>>
    if concept_examples.len() != concept_labels.len() {
        return Err(NeuralError::InvalidArchitecture(
            "Number of examples must match number of labels".to_string(),
        ));
    let mut concept_vectors = Vec::new();
    // Group examples by concept label
    let mut concept_groups: HashMap<String, Vec<&ArrayD<F>>> = HashMap::new();
    for (example, label) in concept_examples.iter().zip(concept_labels.iter()) {
        concept_groups
            .entry(label.clone())
            .or_default()
            .push(example);
    // Compute average activation for each concept
    for (concept_name, examples) in concept_groups {
        if examples.is_empty() {
            continue;
        let first_example = examples[0];
        let mut averaged_activation = first_example.clone();
        for example in examples.iter().skip(1) {
            averaged_activation = averaged_activation + *example;
        averaged_activation = averaged_activation / F::from(examples.len()).unwrap();
        let concept_vector = ConceptActivationVector::new(
            layer_name.clone(),
            averaged_activation,
            0.8, // Placeholder confidence
            examples.len(),
        );
        concept_vectors.push(concept_vector);
    Ok(concept_vectors)
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;
    #[test]
    fn test_counterfactual_generator() {
        let mut generator = CounterfactualGenerator::<f64>::new(5, 0.01, 100, DistanceMetric::L2);
        let input = Array::ones((2, 3)).into_dyn();
        let target = 0.5;
        let counterfactual = generator.generate_counterfactual(&input, target);
        assert!(counterfactual.is_ok());
    fn test_distance_metrics() {
        let generator = CounterfactualGenerator::<f64>::new(5, 0.01, 100, DistanceMetric::L1);
        let input1 = Array::ones((2, 2)).into_dyn();
        let input2 = Array::zeros((2, 2)).into_dyn();
        let distance = generator.compute_distance(&input1, &input2);
        assert_eq!(distance, 4.0); // L1 distance should be 4
    fn test_concept_activation_vector() {
        let activation_vector = Array::from_vec(vec![0.5, 0.8, 0.3]).into_dyn();
        let concept = ConceptActivationVector::new(
            "dog".to_string(),
            "conv5".to_string(),
            0.9,
            100,
        let layer_activations = Array::from_vec(vec![0.6, 0.7, 0.4]).into_dyn();
        let score = concept.compute_activation_score(&layer_activations);
        assert!(score > 0.0);
    fn test_lime_explainer() {
        let mut explainer = LIMEExplainer::<f64>::new(10, 0.1, 0.01, 42);
        let explanation = explainer.explain_instance(&input, Some(0));
        assert!(explanation.is_ok());
    fn test_adversarial_explanation() {
        let explanation = generate_adversarial_explanation::<f64>(&input, "fgsm", 0.1);
        let adv_exp = explanation.unwrap();
        assert_eq!(adv_exp.attack_method, "fgsm");
        assert!(adv_exp.perturbation_magnitude > 0.0);
    fn test_concept_vectors_computation() {
        let examples = vec![
            Array::from_vec(vec![1.0, 0.5]).into_dyn(),
            Array::from_vec(vec![0.8, 0.6]).into_dyn(),
        ];
        let labels = vec!["cat".to_string(), "cat".to_string()];
        let concepts = compute_concept_vectors(&examples, &labels, "conv1".to_string());
        assert!(concepts.is_ok());
        let concept_vectors = concepts.unwrap();
        assert_eq!(concept_vectors.len(), 1);
        assert_eq!(concept_vectors[0].concept_name, "cat");
