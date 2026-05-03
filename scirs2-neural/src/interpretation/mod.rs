//! Model interpretability utilities for neural networks
//!
//! This module provides tools for understanding and explaining neural network decisions:
//!
//! - Gradient-based saliency maps (input × gradient, integrated gradients)
//! - GradCAM (Gradient-weighted Class Activation Mapping)
//! - LIME (Local Interpretable Model-agnostic Explanations)
//! - SHAP-style feature importance estimates
//! - Attention weight visualization
//!
//! # Design
//!
//! Interpretability methods operate on **model functions** rather than on
//! parameterized layer structs.  This avoids tight coupling and lets you plug in
//! any callable `Fn(&[f64]) -> Vec<f64>` as the model under test.
//!
//! # Examples
//!
//! ```
//! use scirs2_neural::interpretation::{
//!     InterpretabilityMethod, InterpretabilityExplainer, ExplainerConfig,
//! };
//!
//! // A trivial linear model: output = sum of inputs
//! let model_fn = |inputs: &[f64]| -> Vec<f64> { vec![inputs.iter().sum()] };
//!
//! let config = ExplainerConfig::default();
//! let explainer = InterpretabilityExplainer::new(config);
//!
//! let input = vec![0.5_f64, -0.2, 1.0];
//! let explanation = explainer
//!     .explain(&model_fn, &input, InterpretabilityMethod::Saliency)
//!     .expect("explain ok");
//! assert_eq!(explanation.feature_importances.len(), input.len());
//! ```

use crate::error::{NeuralError, Result};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────────────────

/// Interpretability method to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpretabilityMethod {
    /// Simple gradient-based saliency: `|∂y/∂x|`
    Saliency,
    /// Input × gradient: `x · ∂y/∂x`
    InputXGradient,
    /// Integrated gradients from a zero baseline.
    ///
    /// Numerically integrates the gradient along a straight-line path from the
    /// baseline (zeros) to the input, using `steps` intervals.
    IntegratedGradients {
        /// Number of integration steps (default 50)
        steps: usize,
    },
    /// GradCAM-style class activation map.
    ///
    /// For classification tasks, computes a weighted sum of feature maps
    /// guided by the gradient of the target class score.
    GradCAM {
        /// Index of the target class
        target_class: usize,
    },
    /// LIME-style local linear approximation.
    ///
    /// Perturbs the input `num_samples` times, fits a linear surrogate model,
    /// and returns the surrogate weights as feature importances.
    LIME {
        /// Number of perturbed samples
        num_samples: usize,
        /// Random seed for reproducibility
        seed: u64,
    },
    /// SHAP-style Shapley values via random feature permutations.
    SHAP {
        /// Number of permutation samples
        num_samples: usize,
        /// Random seed
        seed: u64,
    },
    /// Attention weight visualization for transformer-style models.
    ///
    /// The model function must return a flat attention weight vector of the
    /// same length as the input.
    AttentionViz,
}

/// Baseline input for gradient integration methods.
#[derive(Debug, Clone, Default)]
pub enum BaselineMethod {
    /// All-zeros baseline
    #[default]
    Zero,
    /// Gaussian noise with given standard deviation
    GaussianNoise { std_dev: f64 },
    /// Constant value baseline
    Constant(f64),
}

/// Visualization output type.
#[derive(Debug, Clone, PartialEq)]
pub enum VisualizationMethod {
    /// Heatmap of feature importances
    Heatmap,
    /// Bar chart ranking
    BarChart,
    /// Raw numeric values
    Raw,
}

// ─────────────────────────────────────────────────────────────────────────────
// Explanation output
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an interpretability explanation.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// Feature importance scores, one per input feature.
    pub feature_importances: Vec<f64>,
    /// Method used to produce this explanation.
    pub method: String,
    /// Optional auxiliary data (e.g. attention weights, surrogate model coefficients).
    pub metadata: HashMap<String, Vec<f64>>,
    /// Model output at the explained input point.
    pub model_output: Vec<f64>,
}

impl Explanation {
    /// Returns the index of the most important feature.
    pub fn top_feature(&self) -> Option<usize> {
        self.feature_importances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Returns feature indices sorted by absolute importance (descending).
    pub fn ranked_features(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.feature_importances.len()).collect();
        indices.sort_by(|&a, &b| {
            self.feature_importances[b]
                .abs()
                .partial_cmp(&self.feature_importances[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Normalise feature importances to sum to 1.0 (taking absolute values first).
    pub fn normalized_importances(&self) -> Vec<f64> {
        let sum: f64 = self.feature_importances.iter().map(|v| v.abs()).sum();
        if sum < f64::EPSILON {
            return vec![0.0; self.feature_importances.len()];
        }
        self.feature_importances
            .iter()
            .map(|v| v.abs() / sum)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explainer configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`InterpretabilityExplainer`].
#[derive(Debug, Clone)]
pub struct ExplainerConfig {
    /// Numerical finite-difference step for gradient estimation
    pub gradient_eps: f64,
    /// Baseline method for integration-based methods
    pub baseline: BaselineMethod,
    /// Preferred visualization output format
    pub visualization: VisualizationMethod,
    /// Whether to include absolute values in importance scores
    pub use_absolute: bool,
}

impl Default for ExplainerConfig {
    fn default() -> Self {
        Self {
            gradient_eps: 1e-4,
            baseline: BaselineMethod::Zero,
            visualization: VisualizationMethod::Raw,
            use_absolute: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core explainer
// ─────────────────────────────────────────────────────────────────────────────

/// Computes interpretability explanations for a model function.
///
/// The model is represented as a closure `F: Fn(&[f64]) -> Vec<f64>`.
/// No layer-level coupling is required; any differentiable (or black-box)
/// function can be explained.
pub struct InterpretabilityExplainer {
    config: ExplainerConfig,
}

impl InterpretabilityExplainer {
    /// Create a new explainer with the given config.
    pub fn new(config: ExplainerConfig) -> Self {
        Self { config }
    }

    /// Explain a single `input` point using the specified method.
    pub fn explain<F>(
        &self,
        model: &F,
        input: &[f64],
        method: InterpretabilityMethod,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if input.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "input must not be empty".to_string(),
            ));
        }

        let model_output = model(input);

        match method {
            InterpretabilityMethod::Saliency => self.saliency(model, input, model_output),
            InterpretabilityMethod::InputXGradient => {
                self.input_x_gradient(model, input, model_output)
            }
            InterpretabilityMethod::IntegratedGradients { steps } => {
                self.integrated_gradients(model, input, steps, model_output)
            }
            InterpretabilityMethod::GradCAM { target_class } => {
                self.grad_cam(model, input, target_class, model_output)
            }
            InterpretabilityMethod::LIME { num_samples, seed } => {
                self.lime(model, input, num_samples, seed, model_output)
            }
            InterpretabilityMethod::SHAP { num_samples, seed } => {
                self.shap(model, input, num_samples, seed, model_output)
            }
            InterpretabilityMethod::AttentionViz => self.attention_viz(model, input, model_output),
        }
    }

    // ── Saliency ──────────────────────────────────────────────────────────

    fn saliency<F>(&self, model: &F, input: &[f64], model_output: Vec<f64>) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let grads = self.finite_diff_gradient(model, input)?;
        let importances: Vec<f64> = if self.config.use_absolute {
            grads.iter().map(|g| g.abs()).collect()
        } else {
            grads
        };
        Ok(Explanation {
            feature_importances: importances,
            method: "Saliency".to_string(),
            metadata: HashMap::new(),
            model_output,
        })
    }

    // ── Input × Gradient ─────────────────────────────────────────────────

    fn input_x_gradient<F>(
        &self,
        model: &F,
        input: &[f64],
        model_output: Vec<f64>,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let grads = self.finite_diff_gradient(model, input)?;
        let importances: Vec<f64> = input
            .iter()
            .zip(grads.iter())
            .map(|(&x, &g)| x * g)
            .collect();
        Ok(Explanation {
            feature_importances: importances,
            method: "InputXGradient".to_string(),
            metadata: HashMap::new(),
            model_output,
        })
    }

    // ── Integrated Gradients ──────────────────────────────────────────────

    fn integrated_gradients<F>(
        &self,
        model: &F,
        input: &[f64],
        steps: usize,
        model_output: Vec<f64>,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n_steps = steps.max(1);
        let baseline = self.make_baseline(input);

        let mut accumulated: Vec<f64> = vec![0.0; input.len()];
        for step in 0..=n_steps {
            let alpha = step as f64 / n_steps as f64;
            let interpolated: Vec<f64> = input
                .iter()
                .zip(baseline.iter())
                .map(|(&x, &b)| b + alpha * (x - b))
                .collect();
            let grads = self.finite_diff_gradient(model, &interpolated)?;
            for (acc, g) in accumulated.iter_mut().zip(grads.iter()) {
                *acc += g;
            }
        }

        // Trapezoidal rule: multiply by (input - baseline) / n_steps
        let importances: Vec<f64> = accumulated
            .iter()
            .zip(input.iter().zip(baseline.iter()))
            .map(|(&acc, (&x, &b))| acc * (x - b) / (n_steps as f64 + 1.0))
            .collect();

        Ok(Explanation {
            feature_importances: importances,
            method: "IntegratedGradients".to_string(),
            metadata: HashMap::new(),
            model_output,
        })
    }

    // ── GradCAM ───────────────────────────────────────────────────────────

    fn grad_cam<F>(
        &self,
        model: &F,
        input: &[f64],
        target_class: usize,
        model_output: Vec<f64>,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        // For a generic 1-D function, GradCAM reduces to the gradient of
        // the target class score w.r.t. the input features.
        let grads = self.finite_diff_gradient_class(model, input, target_class)?;
        // ReLU on importance scores (as in original GradCAM paper)
        let importances: Vec<f64> = grads.iter().map(|&g| g.max(0.0)).collect();
        let mut meta = HashMap::new();
        meta.insert("target_class".to_string(), vec![target_class as f64]);
        Ok(Explanation {
            feature_importances: importances,
            method: "GradCAM".to_string(),
            metadata: meta,
            model_output,
        })
    }

    // ── LIME ──────────────────────────────────────────────────────────────

    fn lime<F>(
        &self,
        model: &F,
        input: &[f64],
        num_samples: usize,
        seed: u64,
        model_output: Vec<f64>,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if num_samples == 0 {
            return Err(NeuralError::InvalidArgument(
                "LIME: num_samples must be > 0".to_string(),
            ));
        }
        let n = input.len();

        // Simple LCG PRNG seeded by `seed` to avoid external dependencies
        let mut rng_state = seed.wrapping_add(12345);
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // map to [-1, 1]
            ((*state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
        };

        // Build Gaussian perturbations and collect labels
        let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(num_samples);
        let mut y_vec: Vec<f64> = Vec::with_capacity(num_samples);
        let mut w_vec: Vec<f64> = Vec::with_capacity(num_samples);

        // kernel width for exponential kernel: sqrt of n features
        let kernel_width = (n as f64).sqrt();

        for _ in 0..num_samples {
            let perturb: Vec<f64> = (0..n)
                .map(|j| input[j] + lcg_next(&mut rng_state) * 0.1)
                .collect();
            let dist_sq: f64 = perturb
                .iter()
                .zip(input.iter())
                .map(|(p, x)| (p - x).powi(2))
                .sum();
            let weight = (-dist_sq / (2.0 * kernel_width * kernel_width)).exp();
            let y = model(&perturb);
            let y_scalar = y.first().copied().unwrap_or(0.0);
            x_mat.push(perturb);
            y_vec.push(y_scalar);
            w_vec.push(weight);
        }

        // Fit weighted linear regression: minimize Σ w_i (y_i - x_i·β)²
        // Normal equations: (X'WX)β = X'Wy
        // We use a simplified version with diagonal XtWX
        let mut xtwy = vec![0.0_f64; n];
        let mut xtwx_diag = vec![0.0_f64; n];
        for (i, row) in x_mat.iter().enumerate() {
            let wi = w_vec[i];
            let yi = y_vec[i];
            for j in 0..n {
                xtwy[j] += wi * row[j] * yi;
                xtwx_diag[j] += wi * row[j] * row[j];
            }
        }
        let coefficients: Vec<f64> = xtwy
            .iter()
            .zip(xtwx_diag.iter())
            .map(|(&num, &den)| {
                if den.abs() > f64::EPSILON {
                    num / den
                } else {
                    0.0
                }
            })
            .collect();

        let mut meta = HashMap::new();
        meta.insert("num_samples".to_string(), vec![num_samples as f64]);
        meta.insert("surrogate_coefficients".to_string(), coefficients.clone());

        Ok(Explanation {
            feature_importances: coefficients,
            method: "LIME".to_string(),
            metadata: meta,
            model_output,
        })
    }

    // ── SHAP ──────────────────────────────────────────────────────────────

    fn shap<F>(
        &self,
        model: &F,
        input: &[f64],
        num_samples: usize,
        seed: u64,
        model_output: Vec<f64>,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if num_samples == 0 {
            return Err(NeuralError::InvalidArgument(
                "SHAP: num_samples must be > 0".to_string(),
            ));
        }
        let n = input.len();
        let baseline = self.make_baseline(input);

        // Permutation SHAP: repeatedly shuffle features and measure marginal contribution
        let mut shap_values = vec![0.0_f64; n];

        let mut rng_state = seed.wrapping_add(99991);
        let lcg_next_usize = |state: &mut u64, limit: usize| -> usize {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 33) as usize) % limit
        };

        for _ in 0..num_samples {
            // Fisher-Yates shuffle of feature indices
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = lcg_next_usize(&mut rng_state, i + 1);
                order.swap(i, j);
            }

            // Build baseline with features added one by one
            let mut current: Vec<f64> = baseline.clone();
            let mut prev_output = model(&current).first().copied().unwrap_or(0.0);

            for &feat in &order {
                current[feat] = input[feat];
                let new_output = model(&current).first().copied().unwrap_or(0.0);
                shap_values[feat] += new_output - prev_output;
                prev_output = new_output;
            }
        }

        // Average over permutations
        let n_samp = num_samples as f64;
        let importances: Vec<f64> = shap_values.iter().map(|v| v / n_samp).collect();

        let mut meta = HashMap::new();
        meta.insert("num_samples".to_string(), vec![num_samples as f64]);

        Ok(Explanation {
            feature_importances: importances,
            method: "SHAP".to_string(),
            metadata: meta,
            model_output,
        })
    }

    // ── AttentionViz ─────────────────────────────────────────────────────

    fn attention_viz<F>(
        &self,
        model: &F,
        input: &[f64],
        model_output: Vec<f64>,
    ) -> Result<Explanation>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let raw = model(input);
        // The attention weights are the model outputs, re-normalised to sum to 1
        let n = input.len();
        let weights_raw: Vec<f64> = if raw.len() >= n {
            raw[..n].to_vec()
        } else {
            // Pad with zeros if model output is shorter than input
            let mut v = raw.clone();
            v.resize(n, 0.0);
            v
        };
        let sum: f64 = weights_raw.iter().map(|v| v.abs()).sum();
        let importances: Vec<f64> = if sum > f64::EPSILON {
            weights_raw.iter().map(|v| v / sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
        let mut meta = HashMap::new();
        meta.insert("raw_attention".to_string(), weights_raw);
        Ok(Explanation {
            feature_importances: importances,
            method: "AttentionViz".to_string(),
            metadata: meta,
            model_output,
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Compute finite-difference gradient of the scalar model output sum.
    fn finite_diff_gradient<F>(&self, model: &F, input: &[f64]) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let eps = self.config.gradient_eps;
        let n = input.len();
        let base: f64 = model(input).iter().sum();
        let mut grads = Vec::with_capacity(n);
        let mut perturbed = input.to_vec();
        for j in 0..n {
            perturbed[j] += eps;
            let up: f64 = model(&perturbed).iter().sum();
            perturbed[j] = input[j];
            grads.push((up - base) / eps);
        }
        Ok(grads)
    }

    /// Compute finite-difference gradient of a specific output class.
    fn finite_diff_gradient_class<F>(
        &self,
        model: &F,
        input: &[f64],
        class_idx: usize,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let eps = self.config.gradient_eps;
        let n = input.len();
        let base_out = model(input);
        let base_val = base_out.get(class_idx).copied().unwrap_or(0.0);
        let mut grads = Vec::with_capacity(n);
        let mut perturbed = input.to_vec();
        for j in 0..n {
            perturbed[j] += eps;
            let up_out = model(&perturbed);
            let up_val = up_out.get(class_idx).copied().unwrap_or(0.0);
            perturbed[j] = input[j];
            grads.push((up_val - base_val) / eps);
        }
        Ok(grads)
    }

    /// Build the baseline vector based on `config.baseline`.
    fn make_baseline(&self, input: &[f64]) -> Vec<f64> {
        match &self.config.baseline {
            BaselineMethod::Zero => vec![0.0; input.len()],
            BaselineMethod::Constant(c) => vec![*c; input.len()],
            BaselineMethod::GaussianNoise { std_dev } => {
                // Deterministic pseudo-noise seeded by input values
                input
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i as f64 * 1.1 + x).sin() * std_dev)
                    .collect()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelInterpreter convenience wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// High-level interpreter that stores the model function and config.
///
/// Compared to [`InterpretabilityExplainer`], this type owns the model
/// closure so you don't have to pass it on every call.
pub struct ModelInterpreter<F>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    model: F,
    explainer: InterpretabilityExplainer,
}

impl<F> ModelInterpreter<F>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    /// Create a new interpreter wrapping `model`.
    pub fn new(model: F, config: ExplainerConfig) -> Self {
        Self {
            model,
            explainer: InterpretabilityExplainer::new(config),
        }
    }

    /// Explain an input with the given method.
    pub fn explain(&self, input: &[f64], method: InterpretabilityMethod) -> Result<Explanation> {
        self.explainer.explain(&self.model, input, method)
    }

    /// Return the model output for `input`.
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        (self.model)(input)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_model(input: &[f64]) -> Vec<f64> {
        vec![input.iter().sum()]
    }

    fn quadratic_model(input: &[f64]) -> Vec<f64> {
        vec![input.iter().map(|x| x * x).sum()]
    }

    #[test]
    fn test_saliency_linear_model() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let input = vec![1.0_f64, 2.0, 3.0];
        let exp = explainer
            .explain(&linear_model, &input, InterpretabilityMethod::Saliency)
            .expect("saliency ok");
        assert_eq!(exp.feature_importances.len(), 3);
        // For a linear sum model, all gradients should be ~1
        for imp in &exp.feature_importances {
            assert!((imp - 1.0).abs() < 1e-3, "imp={imp}");
        }
    }

    #[test]
    fn test_input_x_gradient_linear() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let input = vec![0.5_f64, -0.2, 1.0];
        let exp = explainer
            .explain(
                &linear_model,
                &input,
                InterpretabilityMethod::InputXGradient,
            )
            .expect("ok");
        // input × gradient ≈ x_i * 1 = x_i
        for (i, &imp) in exp.feature_importances.iter().enumerate() {
            assert!((imp - input[i]).abs() < 1e-3, "i={i} imp={imp}");
        }
    }

    #[test]
    fn test_integrated_gradients_linear() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let input = vec![1.0_f64, 2.0, 3.0];
        let exp = explainer
            .explain(
                &linear_model,
                &input,
                InterpretabilityMethod::IntegratedGradients { steps: 50 },
            )
            .expect("ok");
        assert_eq!(exp.feature_importances.len(), 3);
        // For a linear model the IG attributions should match input values
        for (i, &imp) in exp.feature_importances.iter().enumerate() {
            assert!((imp - input[i]).abs() < 0.1, "i={i} imp={imp}");
        }
    }

    #[test]
    fn test_gradcam_quadratic() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let input = vec![1.0_f64, 2.0, 3.0];
        let exp = explainer
            .explain(
                &quadratic_model,
                &input,
                InterpretabilityMethod::GradCAM { target_class: 0 },
            )
            .expect("ok");
        assert_eq!(exp.feature_importances.len(), 3);
        // All importances ≥ 0 (ReLU applied in GradCAM)
        for imp in &exp.feature_importances {
            assert!(*imp >= 0.0, "imp={imp}");
        }
        // Larger input → larger gradient → higher importance
        // x3=3 > x2=2 > x1=1 so importance[2] > importance[0]
        assert!(
            exp.feature_importances[2] > exp.feature_importances[0],
            "expected imp[2]>imp[0]"
        );
    }

    #[test]
    fn test_lime_linear() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let input = vec![1.0_f64, 2.0, 3.0];
        let exp = explainer
            .explain(
                &linear_model,
                &input,
                InterpretabilityMethod::LIME {
                    num_samples: 200,
                    seed: 42,
                },
            )
            .expect("ok");
        assert_eq!(exp.feature_importances.len(), 3);
    }

    #[test]
    fn test_shap_linear() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let input = vec![0.5_f64, 1.0, 1.5];
        let exp = explainer
            .explain(
                &linear_model,
                &input,
                InterpretabilityMethod::SHAP {
                    num_samples: 50,
                    seed: 7,
                },
            )
            .expect("ok");
        assert_eq!(exp.feature_importances.len(), 3);
        // For a linear sum model, SHAP values should closely match the input values
        // (contribution of x_i = x_i - 0 since baseline is zeros)
        for (i, (&imp, &xi)) in exp.feature_importances.iter().zip(input.iter()).enumerate() {
            assert!((imp - xi).abs() < 0.1, "i={i} imp={imp} xi={xi}");
        }
    }

    #[test]
    fn test_attention_viz_normalises_to_one() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        // Model returns its inputs as attention weights
        let model = |inp: &[f64]| -> Vec<f64> { inp.to_vec() };
        let input = vec![0.3_f64, 0.5, 0.2];
        let exp = explainer
            .explain(&model, &input, InterpretabilityMethod::AttentionViz)
            .expect("ok");
        let sum: f64 = exp.feature_importances.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn test_empty_input_returns_err() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let result = explainer.explain(&linear_model, &[], InterpretabilityMethod::Saliency);
        assert!(result.is_err());
    }

    #[test]
    fn test_explanation_top_feature() {
        let imp = vec![0.1_f64, 0.5, 0.3];
        let exp = Explanation {
            feature_importances: imp,
            method: "test".to_string(),
            metadata: HashMap::new(),
            model_output: vec![0.0],
        };
        assert_eq!(exp.top_feature(), Some(1));
    }

    #[test]
    fn test_explanation_ranked_features() {
        let imp = vec![0.1_f64, 0.5, 0.3];
        let exp = Explanation {
            feature_importances: imp,
            method: "test".to_string(),
            metadata: HashMap::new(),
            model_output: vec![0.0],
        };
        let ranked = exp.ranked_features();
        assert_eq!(ranked[0], 1); // largest importance
    }

    #[test]
    fn test_normalized_importances_sums_to_one() {
        let imp = vec![1.0_f64, 2.0, 3.0];
        let exp = Explanation {
            feature_importances: imp,
            method: "test".to_string(),
            metadata: HashMap::new(),
            model_output: vec![0.0],
        };
        let normed = exp.normalized_importances();
        let sum: f64 = normed.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_interpreter_predict() {
        let interpreter = ModelInterpreter::new(linear_model, ExplainerConfig::default());
        let output = interpreter.predict(&[1.0, 2.0, 3.0]);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_lime_zero_samples_err() {
        let explainer = InterpretabilityExplainer::new(ExplainerConfig::default());
        let result = explainer.explain(
            &linear_model,
            &[1.0, 2.0],
            InterpretabilityMethod::LIME {
                num_samples: 0,
                seed: 0,
            },
        );
        assert!(result.is_err());
    }
}
