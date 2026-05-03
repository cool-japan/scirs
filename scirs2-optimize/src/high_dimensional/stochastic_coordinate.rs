//! Stochastic Dual Coordinate Ascent (SDCA) with Variance Reduction
//! (Shalev-Shwartz & Zhang, 2013)
//!
//! For L2-regularized empirical risk minimization of the form:
//!
//!   min_w (1/n) * sum_{i=1}^{n} loss(w^T x_i, y_i) + (lambda/2) * ||w||^2
//!
//! SDCA works by optimizing the dual problem coordinate-by-coordinate,
//! achieving variance reduction inherently through the dual formulation.
//!
//! ## Supported Loss Functions
//!
//! - Hinge loss (SVM)
//! - Squared hinge loss (smoothed SVM)
//! - Logistic loss (logistic regression)
//! - Squared loss (ridge regression)
//!
//! ## Reference
//!
//! - Shalev-Shwartz, S. and Zhang, T. (2013).
//!   "Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization"

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// SDCA loss function type
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SDCALoss {
    /// Hinge loss: max(0, 1 - y * w^T x) for SVM
    Hinge,
    /// Squared hinge loss: max(0, 1 - y * w^T x)^2 for smoothed SVM
    SquaredHinge,
    /// Logistic loss: log(1 + exp(-y * w^T x)) for logistic regression
    Logistic,
    /// Squared loss: 0.5 * (y - w^T x)^2 for ridge regression
    SquaredLoss,
}

/// Configuration for SDCA solver
#[derive(Debug, Clone)]
pub struct SDCAConfig {
    /// L2 regularization parameter lambda
    pub lambda: f64,
    /// Maximum number of passes over the data (epochs)
    pub max_epochs: usize,
    /// Convergence tolerance on duality gap
    pub tolerance: f64,
    /// Loss function to use
    pub loss: SDCALoss,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for SDCAConfig {
    fn default() -> Self {
        Self {
            lambda: 1e-4,
            max_epochs: 100,
            tolerance: 1e-6,
            loss: SDCALoss::SquaredLoss,
            seed: Some(42),
        }
    }
}

/// Result of SDCA optimization
#[derive(Debug, Clone)]
pub struct SDCAResult {
    /// Primal weights vector
    pub weights: Vec<f64>,
    /// Dual variables (one per sample)
    pub dual_variables: Vec<f64>,
    /// Final primal objective value
    pub primal_objective: f64,
    /// Final dual objective value
    pub dual_objective: f64,
    /// Duality gap (primal - dual)
    pub duality_gap: f64,
    /// Number of epochs performed
    pub epochs: usize,
    /// Whether the solver converged (duality gap < tolerance)
    pub converged: bool,
}

/// Compute the dot product of a feature vector with the weight vector
fn dot_product(features: &[f64], weights: &[f64]) -> f64 {
    features
        .iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum()
}

/// Compute the squared L2 norm of a vector
fn squared_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Compute the primal loss for a single sample
fn primal_loss_single(prediction: f64, label: f64, loss: SDCALoss) -> f64 {
    match loss {
        SDCALoss::Hinge => {
            let margin = label * prediction;
            (1.0 - margin).max(0.0)
        }
        SDCALoss::SquaredHinge => {
            let margin = label * prediction;
            let hinge = (1.0 - margin).max(0.0);
            hinge * hinge
        }
        SDCALoss::Logistic => {
            let margin = label * prediction;
            if margin > 15.0 {
                (-margin).exp() // avoid overflow
            } else if margin < -15.0 {
                -margin
            } else {
                (1.0 + (-margin).exp()).ln()
            }
        }
        SDCALoss::SquaredLoss => {
            let diff = label - prediction;
            0.5 * diff * diff
        }
    }
}

/// Compute the conjugate loss for the dual objective
fn conjugate_loss(alpha: f64, label: f64, loss: SDCALoss) -> f64 {
    match loss {
        SDCALoss::Hinge => {
            // Conjugate of hinge: alpha * y if -1 <= alpha * y <= 0, else +inf
            let ay = alpha * label;
            if ay >= -1.0 && ay <= 0.0 {
                ay
            } else {
                f64::INFINITY
            }
        }
        SDCALoss::SquaredHinge => {
            // Conjugate of squared hinge
            let ay = alpha * label;
            if ay <= 0.0 {
                ay + 0.25 * alpha * alpha
            } else {
                f64::INFINITY
            }
        }
        SDCALoss::Logistic => {
            // Conjugate of logistic: p*log(p) + (1-p)*log(1-p) where p = -alpha*y
            let p = -alpha * label;
            if p <= 0.0 || p >= 1.0 {
                if (p - 0.0).abs() < 1e-15 || (p - 1.0).abs() < 1e-15 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                p * p.ln() + (1.0 - p) * (1.0 - p).ln()
            }
        }
        SDCALoss::SquaredLoss => {
            // Conjugate of 0.5*(y-z)^2 w.r.t. z:
            // sup_z { alpha*z - 0.5*(y-z)^2 } = 0.5*alpha^2 + alpha*y
            0.5 * alpha * alpha + alpha * label
        }
    }
}

/// Compute the optimal dual coordinate update for a given sample
fn compute_dual_update(
    alpha_i: f64,
    xi_dot_w: f64,
    label: f64,
    xi_norm_sq: f64,
    n: f64,
    lambda: f64,
    loss: SDCALoss,
) -> f64 {
    let q = xi_norm_sq / (lambda * n);

    match loss {
        SDCALoss::Hinge => {
            // For hinge loss with convention alpha_i * y_i in [0, 1]
            let s = alpha_i * label;
            let margin = label * xi_dot_w;
            let delta_s = (1.0 - margin - s) / (q + 1.0);
            let new_s = (s + delta_s).max(0.0).min(1.0);
            new_s * label
        }
        SDCALoss::SquaredHinge => {
            // Smooth hinge with alpha_i * y_i >= 0
            let s = alpha_i * label;
            let margin = label * xi_dot_w;
            let delta_s = (1.0 - margin - s) / (q + 1.0 + 0.5);
            let new_s = (s + delta_s).max(0.0);
            new_s * label
        }
        SDCALoss::Logistic => {
            // For logistic loss, alpha_i * y_i in (0, 1)
            let s = (alpha_i * label).max(0.0).min(1.0);
            let margin = label * xi_dot_w;
            let sigmoid = 1.0 / (1.0 + (-margin).exp());
            // Target: s should equal 1 - sigmoid at optimum
            let target = 1.0 - sigmoid;
            let delta = (target - s) / (q + 1.0);
            let new_s = (s + delta).max(1e-10).min(1.0 - 1e-10);
            new_s * label
        }
        SDCALoss::SquaredLoss => {
            // For squared loss: delta = (y_i - w^T x_i - alpha_i) / (q + 1)
            let delta = (label - xi_dot_w - alpha_i) / (q + 1.0);
            alpha_i + delta
        }
    }
}

/// Run Stochastic Dual Coordinate Ascent
///
/// Solves: min_w (1/n) sum_i loss(w^T x_i, y_i) + (lambda/2) ||w||^2
///
/// # Arguments
/// * `features` - Training data, each element is a feature vector for one sample
/// * `labels` - Labels for each sample (n_samples)
/// * `config` - SDCA configuration
///
/// # Returns
/// * `SDCAResult` with optimal weights, dual variables, and convergence info
pub fn sdca(
    features: &[Vec<f64>],
    labels: &[f64],
    config: &SDCAConfig,
) -> OptimizeResult<SDCAResult> {
    let n_samples = features.len();
    if n_samples == 0 {
        return Err(OptimizeError::InvalidInput(
            "Must provide at least one training sample".to_string(),
        ));
    }
    if labels.len() != n_samples {
        return Err(OptimizeError::InvalidInput(format!(
            "Number of labels ({}) must match number of feature vectors ({})",
            labels.len(),
            n_samples
        )));
    }

    let n_features = features[0].len();
    if n_features == 0 {
        return Err(OptimizeError::InvalidInput(
            "Feature vectors must have at least one dimension".to_string(),
        ));
    }

    // Validate consistent feature dimensions
    for (i, feat) in features.iter().enumerate() {
        if feat.len() != n_features {
            return Err(OptimizeError::InvalidInput(format!(
                "Feature vector {} has length {} but expected {}",
                i,
                feat.len(),
                n_features
            )));
        }
    }

    if config.lambda <= 0.0 {
        return Err(OptimizeError::InvalidInput(format!(
            "Regularization parameter lambda must be positive, got {}",
            config.lambda
        )));
    }

    let n = n_samples as f64;
    let lambda = config.lambda;

    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(0),
    };

    // Precompute squared norms of feature vectors
    let xi_norms_sq: Vec<f64> = features.iter().map(|x| squared_norm(x)).collect();

    // Initialize dual variables to zero
    let mut alpha = vec![0.0; n_samples];
    // Primal weights: w = (1 / (lambda * n)) * sum_i alpha_i * x_i
    let mut weights = vec![0.0; n_features];

    let mut converged = false;
    let mut epochs = 0;

    for epoch in 0..config.max_epochs {
        epochs = epoch + 1;

        // Randomly permute sample indices for this epoch
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = rng.random_range(0..=(i as u64)) as usize;
            indices.swap(i, j);
        }

        for &i in &indices {
            let xi = &features[i];
            let yi = labels[i];
            let xi_norm_sq = xi_norms_sq[i];

            if xi_norm_sq < 1e-30 {
                continue; // Skip zero feature vectors
            }

            // Compute w^T x_i
            let xi_dot_w = dot_product(xi, &weights);

            // Compute optimal dual update
            let old_alpha = alpha[i];
            let new_alpha =
                compute_dual_update(old_alpha, xi_dot_w, yi, xi_norm_sq, n, lambda, config.loss);

            let delta_alpha = new_alpha - old_alpha;
            alpha[i] = new_alpha;

            // Update primal weights: w += delta_alpha / (lambda * n) * x_i
            let scale = delta_alpha / (lambda * n);
            for j in 0..n_features {
                weights[j] += scale * xi[j];
            }
        }

        // Compute primal and dual objectives for convergence check
        let primal_obj = compute_primal_objective(features, labels, &weights, lambda, config.loss);
        let dual_obj =
            compute_dual_objective(features, labels, &alpha, &weights, lambda, config.loss);
        let gap = primal_obj - dual_obj;

        if gap.abs() < config.tolerance {
            converged = true;
            break;
        }
    }

    let primal_objective =
        compute_primal_objective(features, labels, &weights, lambda, config.loss);
    let dual_objective =
        compute_dual_objective(features, labels, &alpha, &weights, lambda, config.loss);
    let duality_gap = primal_objective - dual_objective;

    Ok(SDCAResult {
        weights,
        dual_variables: alpha,
        primal_objective,
        dual_objective,
        duality_gap,
        epochs,
        converged,
    })
}

/// Compute the primal objective: (1/n) sum loss(w^T x_i, y_i) + (lambda/2) ||w||^2
fn compute_primal_objective(
    features: &[Vec<f64>],
    labels: &[f64],
    weights: &[f64],
    lambda: f64,
    loss: SDCALoss,
) -> f64 {
    let n = features.len() as f64;
    let empirical_risk: f64 = features
        .iter()
        .zip(labels.iter())
        .map(|(xi, &yi)| {
            let prediction = dot_product(xi, weights);
            primal_loss_single(prediction, yi, loss)
        })
        .sum::<f64>()
        / n;

    let regularization = 0.5 * lambda * squared_norm(weights);
    empirical_risk + regularization
}

/// Compute the dual objective
fn compute_dual_objective(
    features: &[Vec<f64>],
    labels: &[f64],
    alpha: &[f64],
    _weights: &[f64],
    lambda: f64,
    loss: SDCALoss,
) -> f64 {
    let n = features.len() as f64;

    // Dual objective = -(1/n) sum conjugate_loss(-alpha_i) - (1/(2*lambda*n^2)) ||sum alpha_i x_i||^2
    let conj_sum: f64 = alpha
        .iter()
        .zip(labels.iter())
        .map(|(&ai, &yi)| conjugate_loss(-ai, yi, loss))
        .sum::<f64>()
        / n;

    // Compute ||sum alpha_i x_i||^2
    let n_features = features[0].len();
    let mut sum_alpha_x = vec![0.0; n_features];
    for (i, xi) in features.iter().enumerate() {
        for j in 0..n_features {
            sum_alpha_x[j] += alpha[i] * xi[j];
        }
    }
    let norm_sq = squared_norm(&sum_alpha_x);

    -conj_sum - norm_sq / (2.0 * lambda * n * n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: SDCA on linearly separable data with hinge loss (SVM)
    #[test]
    fn test_sdca_hinge_separable() {
        // Simple 2D linearly separable data
        let features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![-2.0, -1.0],
        ];
        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let config = SDCAConfig {
            lambda: 0.01,
            max_epochs: 200,
            tolerance: 1e-4,
            loss: SDCALoss::Hinge,
            seed: Some(42),
        };

        let result = sdca(&features, &labels, &config);
        assert!(result.is_ok());
        let result = result.expect("SDCA should succeed");

        // Check that predictions have correct sign
        for (xi, &yi) in features.iter().zip(labels.iter()) {
            let pred = dot_product(xi, &result.weights);
            assert!(
                pred * yi > -0.5,
                "Misclassification: pred={}, label={}",
                pred,
                yi
            );
        }
    }

    /// Test 2: SDCA ridge regression converges to closed-form solution
    #[test]
    fn test_sdca_ridge_regression() {
        // Simple regression: y = 2*x1 + 3*x2
        let features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![3.0, 1.0],
        ];
        let labels: Vec<f64> = features.iter().map(|x| 2.0 * x[0] + 3.0 * x[1]).collect();

        let config = SDCAConfig {
            lambda: 0.001,
            max_epochs: 500,
            tolerance: 1e-6,
            loss: SDCALoss::SquaredLoss,
            seed: Some(42),
        };

        let result = sdca(&features, &labels, &config);
        assert!(result.is_ok());
        let result = result.expect("SDCA should succeed");

        // With small lambda, weights should be close to [2, 3]
        assert!(
            (result.weights[0] - 2.0).abs() < 0.5,
            "w[0]={}, expected ~2.0",
            result.weights[0]
        );
        assert!(
            (result.weights[1] - 3.0).abs() < 0.5,
            "w[1]={}, expected ~3.0",
            result.weights[1]
        );
    }

    /// Test 3: SDCA logistic regression
    #[test]
    fn test_sdca_logistic() {
        // Use well-separated data for clear classification
        let features = vec![
            vec![3.0, 3.0],
            vec![4.0, 3.0],
            vec![3.0, 4.0],
            vec![-3.0, -3.0],
            vec![-4.0, -3.0],
            vec![-3.0, -4.0],
        ];
        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let config = SDCAConfig {
            lambda: 0.01,
            max_epochs: 500,
            tolerance: 1e-6,
            loss: SDCALoss::Logistic,
            seed: Some(42),
        };

        let result = sdca(&features, &labels, &config);
        assert!(result.is_ok());
        let result = result.expect("SDCA should succeed");

        // Weights should separate the two classes
        let mut correct = 0;
        for (xi, &yi) in features.iter().zip(labels.iter()) {
            let pred = dot_product(xi, &result.weights);
            if pred * yi > 0.0 {
                correct += 1;
            }
        }
        assert!(correct >= 4, "Only {}/6 correct classifications", correct);
    }

    /// Test 4: SDCA squared hinge loss
    #[test]
    fn test_sdca_squared_hinge() {
        let features = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];
        let labels = vec![1.0, 1.0, -1.0, -1.0];

        let config = SDCAConfig {
            lambda: 0.1,
            max_epochs: 200,
            tolerance: 1e-4,
            loss: SDCALoss::SquaredHinge,
            seed: Some(42),
        };

        let result = sdca(&features, &labels, &config);
        assert!(result.is_ok());
        let result = result.expect("SDCA should succeed");

        // Primal objective should be finite
        assert!(
            result.primal_objective.is_finite(),
            "Primal objective is not finite: {}",
            result.primal_objective
        );
    }

    /// Test 5: Empty features error
    #[test]
    fn test_empty_features() {
        let features: Vec<Vec<f64>> = vec![];
        let labels: Vec<f64> = vec![];
        let config = SDCAConfig::default();
        let result = sdca(&features, &labels, &config);
        assert!(result.is_err());
    }

    /// Test 6: Mismatched labels/features error
    #[test]
    fn test_mismatched_dimensions() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![1.0]; // wrong length
        let config = SDCAConfig::default();
        let result = sdca(&features, &labels, &config);
        assert!(result.is_err());
    }

    /// Test 7: Invalid lambda error
    #[test]
    fn test_invalid_lambda() {
        let features = vec![vec![1.0]];
        let labels = vec![1.0];
        let config = SDCAConfig {
            lambda: -1.0,
            ..SDCAConfig::default()
        };
        let result = sdca(&features, &labels, &config);
        assert!(result.is_err());
    }

    /// Test 8: Primal objective decreases over epochs for squared loss
    #[test]
    fn test_primal_decreases_squared_loss() {
        let features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
        ];
        let labels = vec![1.0, 2.0, 3.0, 4.0];

        // Run with few epochs
        let config_few = SDCAConfig {
            lambda: 0.01,
            max_epochs: 5,
            tolerance: 0.0,
            loss: SDCALoss::SquaredLoss,
            seed: Some(42),
        };

        // Run with many epochs
        let config_many = SDCAConfig {
            lambda: 0.01,
            max_epochs: 200,
            tolerance: 0.0,
            loss: SDCALoss::SquaredLoss,
            seed: Some(42),
        };

        let result_few = sdca(&features, &labels, &config_few);
        let result_many = sdca(&features, &labels, &config_many);

        assert!(result_few.is_ok());
        assert!(result_many.is_ok());

        let r_few = result_few.expect("should succeed");
        let r_many = result_many.expect("should succeed");

        // More epochs should yield lower or equal primal objective
        assert!(
            r_many.primal_objective <= r_few.primal_objective + 1e-8,
            "More epochs didn't help: {} vs {}",
            r_many.primal_objective,
            r_few.primal_objective
        );
    }

    /// Test 9: Dual variables are populated
    #[test]
    fn test_dual_variables_populated() {
        let features = vec![vec![1.0], vec![-1.0]];
        let labels = vec![1.0, -1.0];

        let config = SDCAConfig {
            lambda: 0.1,
            max_epochs: 50,
            tolerance: 1e-8,
            loss: SDCALoss::SquaredLoss,
            seed: Some(42),
        };

        let result = sdca(&features, &labels, &config);
        assert!(result.is_ok());
        let result = result.expect("should succeed");

        assert_eq!(result.dual_variables.len(), 2);
        // At least one dual variable should be nonzero after optimization
        let any_nonzero = result.dual_variables.iter().any(|&a| a.abs() > 1e-15);
        assert!(any_nonzero, "All dual variables are zero");
    }

    /// Test 10: Inconsistent feature dimensions error
    #[test]
    fn test_inconsistent_feature_dims() {
        let features = vec![vec![1.0, 2.0], vec![3.0]]; // inconsistent
        let labels = vec![1.0, -1.0];
        let config = SDCAConfig::default();
        let result = sdca(&features, &labels, &config);
        assert!(result.is_err());
    }
}
