//! Local explanation methods for individual predictions

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Local explanation generator
pub struct LocalExplainer<F: Float> {
    /// Number of samples for perturbation-based methods
    pub n_samples: usize,
    /// Perturbation standard deviation
    pub perturbation_std: F,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum> Default
    for LocalExplainer<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum> LocalExplainer<F> {
    /// Create new local explainer
    pub fn new() -> Self {
        Self {
            n_samples: 1000,
            perturbation_std: F::from(0.1).unwrap(),
            random_seed: None,
        }
    }

    /// Set number of samples
    pub fn with_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Set perturbation standard deviation
    pub fn with_perturbation_std(mut self, std: F) -> Self {
        self.perturbation_std = std;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Generate LIME explanation for a single instance
    pub fn explain_lime<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_names: &[String],
    ) -> Result<LocalExplanation<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Generate perturbed samples around the instance
        let (perturbed_samples, distances) = self.generate_perturbed_samples(instance)?;

        // Get model predictions for perturbed samples
        let predictions = model(&perturbed_samples.view());

        // Fit local linear model with distance-based weights
        let weights = self.compute_sample_weights(&distances)?;
        let coefficients =
            self.fit_weighted_linear_model(&perturbed_samples, &predictions, &weights)?;

        // Create explanation
        let mut feature_importance = HashMap::new();
        for (i, feature_name) in feature_names.iter().enumerate() {
            if i < coefficients.len() {
                feature_importance.insert(feature_name.clone(), coefficients[i]);
            }
        }

        Ok(LocalExplanation {
            instance: instance.to_owned(),
            feature_importance,
            prediction: model(&instance.insert_axis(Axis(0)).view())[0],
            confidence: self.compute_explanation_confidence(&coefficients)?,
            method: "LIME".to_string(),
        })
    }

    /// Generate gradient-based explanation
    pub fn explain_gradient<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_names: &[String],
    ) -> Result<LocalExplanation<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Compute numerical gradients
        let gradients = self.compute_numerical_gradients(model, instance)?;

        let mut feature_importance = HashMap::new();
        for (i, feature_name) in feature_names.iter().enumerate() {
            if i < gradients.len() {
                // Gradient * input value for integrated gradients approximation
                feature_importance.insert(feature_name.clone(), gradients[i] * instance[i]);
            }
        }

        Ok(LocalExplanation {
            instance: instance.to_owned(),
            feature_importance,
            prediction: model(&instance.insert_axis(Axis(0)).view())[0],
            confidence: self.compute_explanation_confidence(&gradients)?,
            method: "Gradient".to_string(),
        })
    }

    /// Generate SHAP explanation (simplified approximation)
    pub fn explain_shap<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_names: &[String],
        background: &ArrayView1<F>,
    ) -> Result<LocalExplanation<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();
        let mut shap_values = Array1::zeros(n_features);

        // Simplified SHAP computation using subset sampling
        let _baseline_pred = model(&background.insert_axis(Axis(0)).view())[0];
        let full_pred = model(&instance.insert_axis(Axis(0)).view())[0];

        for i in 0..n_features {
            let mut marginal_contributions = Vec::new();

            // Sample different coalitions
            for _ in 0..50 {
                let coalition = self.sample_coalition(n_features, i)?;

                // Prediction with coalition including feature i
                let mut with_feature = background.to_owned();
                for &j in &coalition {
                    with_feature[j] = instance[j];
                }
                with_feature[i] = instance[i];
                let pred_with = model(&with_feature.insert_axis(Axis(0)).view())[0];

                // Prediction with coalition excluding feature i
                let mut without_feature = background.to_owned();
                for &j in &coalition {
                    if j != i {
                        without_feature[j] = instance[j];
                    }
                }
                let pred_without = model(&without_feature.insert_axis(Axis(0)).view())[0];

                marginal_contributions.push(pred_with - pred_without);
            }

            shap_values[i] = marginal_contributions.iter().cloned().sum::<F>()
                / F::from(marginal_contributions.len()).unwrap();
        }

        let mut feature_importance = HashMap::new();
        for (i, feature_name) in feature_names.iter().enumerate() {
            if i < shap_values.len() {
                feature_importance.insert(feature_name.clone(), shap_values[i]);
            }
        }

        Ok(LocalExplanation {
            instance: instance.to_owned(),
            feature_importance,
            prediction: full_pred,
            confidence: self.compute_explanation_confidence(&shap_values)?,
            method: "SHAP".to_string(),
        })
    }

    /// Generate anchors explanation (rule-based)
    pub fn explain_anchors<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_names: &[String],
    ) -> Result<AnchorExplanation<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let original_prediction = model(&instance.insert_axis(Axis(0)).view())[0];
        let mut anchor_rules = Vec::new();

        // Find features that strongly determine the prediction
        for (i, feature_name) in feature_names.iter().enumerate() {
            if i >= instance.len() {
                continue;
            }

            let feature_value = instance[i];
            let precision = self.test_anchor_precision(model, instance, i, original_prediction)?;

            if precision > F::from(0.8).unwrap() {
                let rule = AnchorRule {
                    feature_name: feature_name.clone(),
                    feature_index: i,
                    condition: format!("{} = {:.3}", feature_name, feature_value.to_f64().unwrap()),
                    precision,
                    coverage: self.compute_anchor_coverage(instance, i)?,
                };
                anchor_rules.push(rule);
            }
        }

        let overall_precision = self.compute_overall_precision(&anchor_rules)?;
        let overall_coverage = self.compute_overall_coverage(&anchor_rules)?;

        Ok(AnchorExplanation {
            instance: instance.to_owned(),
            prediction: original_prediction,
            anchor_rules,
            overall_precision,
            overall_coverage,
        })
    }

    // Helper methods

    fn generate_perturbed_samples(
        &self,
        instance: &ArrayView1<F>,
    ) -> Result<(Array2<F>, Array1<F>)> {
        let n_features = instance.len();
        let mut samples = Array2::zeros((self.n_samples, n_features));
        let mut distances = Array1::zeros(self.n_samples);

        for i in 0..self.n_samples {
            let mut distance = F::zero();
            for j in 0..n_features {
                // Add Gaussian noise
                let noise = self.perturbation_std * self.generate_gaussian_noise(i, j)?;
                samples[[i, j]] = instance[j] + noise;
                distance = distance + noise * noise;
            }
            distances[i] = distance.sqrt();
        }

        Ok((samples, distances))
    }

    fn generate_gaussian_noise(&self, sample_idx: usize, featureidx: usize) -> Result<F> {
        // Simplified Gaussian noise generation using Box-Muller transform
        let seed = self.random_seed.unwrap_or(0) + sample_idx as u64 + featureidx as u64;
        let u1 = F::from((seed % 1000) as f64 / 1000.0).unwrap();
        let u2 = F::from(((seed / 1000) % 1000) as f64 / 1000.0).unwrap();

        let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
            * (F::from(2.0).unwrap() * F::from(std::f64::consts::PI).unwrap() * u2).cos();

        Ok(z)
    }

    fn compute_sample_weights(&self, distances: &Array1<F>) -> Result<Array1<F>> {
        let mut weights = Array1::zeros(distances.len());
        let kernel_width = self.perturbation_std * F::from(2.0).unwrap();

        for (i, &distance) in distances.iter().enumerate() {
            // RBF kernel weights
            weights[i] = (-(distance * distance) / (kernel_width * kernel_width)).exp();
        }

        Ok(weights)
    }

    fn fit_weighted_linear_model(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        weights: &Array1<F>,
    ) -> Result<Array1<F>> {
        let n_features = x.ncols();

        // Simplified weighted least squares (in practice, would use proper linear algebra)
        let mut coefficients = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut numerator = F::zero();
            let mut denominator = F::zero();

            for i in 0..x.nrows() {
                let weight = weights[i];
                numerator = numerator + weight * x[[i, j]] * y[i];
                denominator = denominator + weight * x[[i, j]] * x[[i, j]];
            }

            if denominator != F::zero() {
                coefficients[j] = numerator / denominator;
            }
        }

        Ok(coefficients)
    }

    fn compute_numerical_gradients<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
    ) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();
        let mut gradients = Array1::zeros(n_features);
        let epsilon = F::from(1e-6).unwrap();

        let baseline_pred = model(&instance.insert_axis(Axis(0)).view())[0];

        for i in 0..n_features {
            let mut perturbed = instance.to_owned();
            perturbed[i] = perturbed[i] + epsilon;

            let perturbed_pred = model(&perturbed.insert_axis(Axis(0)).view())[0];
            gradients[i] = (perturbed_pred - baseline_pred) / epsilon;
        }

        Ok(gradients)
    }

    fn sample_coalition(&self, n_features: usize, targetfeature: usize) -> Result<Vec<usize>> {
        let mut coalition = Vec::new();

        // Simple coalition sampling (in practice, would use proper random sampling)
        let seed = self.random_seed.unwrap_or(0) + targetfeature as u64;
        let coalition_size = (seed % n_features as u64) as usize;

        for i in 0..coalition_size {
            if i != targetfeature {
                coalition.push(i);
            }
        }

        Ok(coalition)
    }

    fn compute_explanation_confidence(&self, values: &Array1<F>) -> Result<F> {
        let variance = values.iter().map(|&x| x * x).sum::<F>() / F::from(values.len()).unwrap();

        // Confidence based on magnitude of explanation
        Ok(F::one() / (F::one() + variance))
    }

    fn test_anchor_precision<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_idx: usize,
        original_prediction: F,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut correct_predictions = 0;
        let n_tests = 100;

        for i in 0..n_tests {
            let mut test_instance = self.generate_test_instance(instance, feature_idx, i)?;
            test_instance[feature_idx] = instance[feature_idx]; // Keep anchor feature fixed

            let _prediction = model(&test_instance.insert_axis(Axis(0)).view())[0];

            // Check if _prediction is similar to original (within some tolerance)
            if (_prediction - original_prediction).abs() < F::from(0.1).unwrap() {
                correct_predictions += 1;
            }
        }

        Ok(F::from(correct_predictions).unwrap() / F::from(n_tests).unwrap())
    }

    fn generate_test_instance(
        &self,
        instance: &ArrayView1<F>,
        fixed_feature: usize,
        seed: usize,
    ) -> Result<Array1<F>> {
        let mut test_instance = Array1::zeros(instance.len());

        for i in 0..instance.len() {
            if i == fixed_feature {
                test_instance[i] = instance[i];
            } else {
                // Generate random value (simplified)
                let noise = F::from((seed + i) as f64 * 0.01).unwrap();
                test_instance[i] = instance[i] + noise;
            }
        }

        Ok(test_instance)
    }

    fn compute_anchor_coverage(&self, instance: &ArrayView1<F>, featureidx: usize) -> Result<F> {
        // Simplified coverage computation
        Ok(F::from(0.5).unwrap())
    }

    fn compute_overall_precision(&self, rules: &[AnchorRule<F>]) -> Result<F> {
        if rules.is_empty() {
            return Ok(F::zero());
        }

        let sum_precision: F = rules.iter().map(|r| r.precision).sum();
        Ok(sum_precision / F::from(rules.len()).unwrap())
    }

    fn compute_overall_coverage(&self, rules: &[AnchorRule<F>]) -> Result<F> {
        if rules.is_empty() {
            return Ok(F::zero());
        }

        let sum_coverage: F = rules.iter().map(|r| r.coverage).sum();
        Ok(sum_coverage / F::from(rules.len()).unwrap())
    }
}

/// Local explanation result
#[derive(Debug, Clone)]
pub struct LocalExplanation<F: Float> {
    /// The instance being explained
    pub instance: Array1<F>,
    /// Feature importance scores
    pub feature_importance: HashMap<String, F>,
    /// Model prediction for this instance
    pub prediction: F,
    /// Confidence in the explanation
    pub confidence: F,
    /// Method used to generate explanation
    pub method: String,
}

impl<F: Float> LocalExplanation<F> {
    /// Get top-k most important features
    pub fn get_top_features(&self, k: usize) -> Vec<(String, F)> {
        let mut sorted_features: Vec<(String, F)> = self
            .feature_importance
            .iter()
            .map(|(name, &score)| (name.clone(), score))
            .collect();

        sorted_features.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_features.truncate(k);
        sorted_features
    }

    /// Get features contributing positively to the prediction
    pub fn get_positive_features(&self) -> Vec<(String, F)> {
        self.feature_importance
            .iter()
            .filter(|(_, &score)| score > F::zero())
            .map(|(name, &score)| (name.clone(), score))
            .collect()
    }

    /// Get features contributing negatively to the prediction
    pub fn get_negative_features(&self) -> Vec<(String, F)> {
        self.feature_importance
            .iter()
            .filter(|(_, &score)| score < F::zero())
            .map(|(name, &score)| (name.clone(), score))
            .collect()
    }
}

/// Anchor explanation result
#[derive(Debug, Clone)]
pub struct AnchorExplanation<F: Float> {
    /// The instance being explained
    pub instance: Array1<F>,
    /// Model prediction for this instance
    pub prediction: F,
    /// Set of anchor rules
    pub anchor_rules: Vec<AnchorRule<F>>,
    /// Overall precision of the anchor
    pub overall_precision: F,
    /// Overall coverage of the anchor
    pub overall_coverage: F,
}

/// Individual anchor rule
#[derive(Debug, Clone)]
pub struct AnchorRule<F: Float> {
    /// Name of the feature
    pub feature_name: String,
    /// Index of the feature
    pub feature_index: usize,
    /// Human-readable condition
    pub condition: String,
    /// Precision of this rule
    pub precision: F,
    /// Coverage of this rule
    pub coverage: F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock model for testing
    fn mock_model(x: &ArrayView2<f64>) -> Array1<f64> {
        // Simple linear model: prediction = sum of features
        x.map_axis(Axis(1), |row| row.sum())
    }

    #[test]
    fn test_local_explainer_creation() {
        let explainer = LocalExplainer::<f64>::new()
            .with_samples(500)
            .with_perturbation_std(0.05)
            .with_seed(42);

        assert_eq!(explainer.n_samples, 500);
        assert_eq!(explainer.perturbation_std, 0.05);
        assert_eq!(explainer.random_seed, Some(42));
    }

    #[test]
    fn test_gradient_explanation() {
        let explainer = LocalExplainer::<f64>::new().with_seed(42);
        let instance = array![1.0, 2.0, 3.0];
        let feature_names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];

        let explanation = explainer
            .explain_gradient(&mock_model, &instance.view(), &feature_names)
            .unwrap();

        assert_eq!(explanation.feature_importance.len(), 3);
        assert_eq!(explanation.method, "Gradient");
        assert!(explanation.confidence > 0.0);
    }

    #[test]
    fn test_local_explanation_methods() {
        let mut feature_importance = HashMap::new();
        feature_importance.insert("feature1".to_string(), 0.5);
        feature_importance.insert("feature2".to_string(), -0.3);
        feature_importance.insert("feature3".to_string(), 0.8);

        let explanation = LocalExplanation {
            instance: array![1.0, 2.0, 3.0],
            feature_importance,
            prediction: 2.0,
            confidence: 0.9,
            method: "LIME".to_string(),
        };

        let top_features = explanation.get_top_features(2);
        assert_eq!(top_features.len(), 2);
        assert_eq!(top_features[0].0, "feature3");

        let positive_features = explanation.get_positive_features();
        assert_eq!(positive_features.len(), 2);

        let negative_features = explanation.get_negative_features();
        assert_eq!(negative_features.len(), 1);
        assert_eq!(negative_features[0].0, "feature2");
    }

    #[test]
    fn test_anchor_explanation() {
        let explainer = LocalExplainer::<f64>::new().with_seed(42);
        let instance = array![1.0, 2.0, 3.0];
        let feature_names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];

        let explanation = explainer
            .explain_anchors(&mock_model, &instance.view(), &feature_names)
            .unwrap();

        assert_eq!(explanation.instance, instance);
        assert!(explanation.overall_precision >= 0.0);
        assert!(explanation.overall_coverage >= 0.0);
    }

    #[test]
    fn test_numerical_gradients() {
        let explainer = LocalExplainer::<f64>::new();
        let instance = array![1.0, 2.0, 3.0];

        let gradients = explainer
            .compute_numerical_gradients(&mock_model, &instance.view())
            .unwrap();

        assert_eq!(gradients.len(), 3);
        // For our mock model (sum), all gradients should be approximately 1.0
        for &grad in gradients.iter() {
            assert!((grad - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_perturbed_samples_generation() {
        let explainer = LocalExplainer::<f64>::new()
            .with_samples(10)
            .with_perturbation_std(0.1)
            .with_seed(42);

        let instance = array![1.0, 2.0, 3.0];
        let (samples, distances) = explainer
            .generate_perturbed_samples(&instance.view())
            .unwrap();

        assert_eq!(samples.shape(), &[10, 3]);
        assert_eq!(distances.len(), 10);

        // All distances should be non-negative
        assert!(distances.iter().all(|&d| d >= 0.0));
    }
}
