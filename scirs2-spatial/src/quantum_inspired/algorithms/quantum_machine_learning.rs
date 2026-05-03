//! Quantum-Inspired Machine Learning Algorithms
//!
//! This module provides quantum-inspired machine learning algorithms for spatial
//! classification and pattern recognition tasks. The algorithms leverage quantum
//! computing principles — specifically quantum feature maps and kernel methods —
//! to achieve enhanced classification performance on complex spatial data.
//!
//! # Algorithms
//!
//! - [`QuantumSVMModel`]: Quantum-enhanced Support Vector Machine using quantum
//!   kernel functions computed via random Fourier features (Rahimi-Recht method)
//! - [`QuantumClassifier`]: Hybrid quantum-classical classifier wrapping the
//!   quantum SVM with optional preprocessing
//!
//! # Theoretical Foundation
//!
//! The quantum feature map φ: ℝᵈ → ℝᴰ (D >> d) approximates a quantum kernel
//! `k(x, z) = ⟨φ(x), φ(z)⟩ ≈ exp(-‖x - z‖² / (2σ²))` by drawing random
//! frequencies ω from a distribution whose Fourier transform is the kernel's
//! spectral density. This is the Rahimi-Recht random Fourier feature approach.
//!
//! The SVM dual problem is solved via Sequential Minimal Optimization (SMO)
//! to find support vectors and dual coefficients α.

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::PI;

/// Quantum-Enhanced Support Vector Machine
///
/// Implements a kernel SVM where the kernel is computed via a quantum-inspired
/// random Fourier feature map (Rahimi-Recht approximation of the RBF kernel).
/// The feature map φ(x) = √(2/D) · [cos(ω₁ᵀx + b₁), …, cos(ω_Dᵀx + b_D)]ᵀ
/// approximates `exp(-‖x - z‖² / (2σ²))` for random ωᵢ ~ N(0, I/σ²) and
/// bᵢ ~ Uniform[0, 2π].
///
/// # Training Algorithm
///
/// Binary classification is performed via a coordinate-descent dual SVM solver
/// (simplified SMO) on the quantum kernel matrix, respecting the box constraint
/// 0 ≤ αᵢ ≤ C for all support vectors.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_spatial::quantum_inspired::algorithms::quantum_machine_learning::QuantumSVMModel;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Two-class separable dataset (4 points, 2 features)
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     5.0, 5.0,
/// ])?;
/// let y = Array1::from_vec(vec![1.0, 1.0, 1.0, -1.0]);
///
/// let mut model = QuantumSVMModel::new(4, 1.0);
/// model.fit(&x, &y)?;
///
/// let x_test = Array2::from_shape_vec((1, 2), vec![0.5, 0.5])?;
/// let preds = model.predict(&x_test)?;
/// assert_eq!(preds.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct QuantumSVMModel {
    /// Number of qubits (determines random Fourier feature dimension D = 2^n_qubits)
    n_qubits: usize,
    /// Regularisation parameter C (box constraint for dual variables)
    regularization: f64,
    /// Stored support vectors (a subset of training points)
    support_vectors: Vec<Array1<f64>>,
    /// Dual coefficients αᵢ · yᵢ for each support vector
    alphas: Vec<f64>,
    /// Bias term learned during training
    bias: f64,
    /// Random frequency matrix Ω ∈ ℝ^{D × d}, rows are ωᵢ
    random_weights: Option<Array2<f64>>,
    /// Random phase offsets bᵢ ~ Uniform[0, 2π] (length D)
    random_offsets: Option<Array1<f64>>,
    /// RBF bandwidth parameter σ
    bandwidth: f64,
}

impl QuantumSVMModel {
    /// Construct a new `QuantumSVMModel`.
    ///
    /// # Arguments
    /// * `n_qubits` - Number of qubits; the random Fourier feature dimension
    ///   is `D = 2^n_qubits` (clamped to [4, 256] for practical use).
    /// * `regularization` - The SVM regularisation constant C > 0.
    ///
    /// # Panics
    /// Does not panic — invalid parameters are caught in [`fit`].
    pub fn new(n_qubits: usize, regularization: f64) -> Self {
        Self {
            n_qubits,
            regularization,
            support_vectors: Vec::new(),
            alphas: Vec::new(),
            bias: 0.0,
            random_weights: None,
            random_offsets: None,
            bandwidth: 1.0,
        }
    }

    /// Return the number of qubits controlling the feature map dimension.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Return the regularisation parameter C.
    pub fn regularization(&self) -> f64 {
        self.regularization
    }

    /// Return the number of support vectors (0 before training).
    pub fn num_support_vectors(&self) -> usize {
        self.support_vectors.len()
    }

    /// Compute the quantum feature dimension D = 2^n_qubits, clamped to [4, 256].
    fn feature_dim(&self) -> usize {
        let raw = 1usize << self.n_qubits;
        raw.clamp(4, 256)
    }

    /// Initialise random Fourier feature parameters for input dimension `d`.
    ///
    /// Draws ωᵢ from N(0, I / σ²) and bᵢ from Uniform[0, 2π].
    fn init_random_features(&mut self, d: usize) {
        let big_d = self.feature_dim();
        let mut rng = scirs2_core::random::rng();

        // Box-Muller transform to produce Gaussian samples (σ² = bandwidth² so
        // 1/σ² = 1/bandwidth²)
        let scale = 1.0 / self.bandwidth;
        let mut weights = Array2::<f64>::zeros((big_d, d));
        let mut offsets = Array1::<f64>::zeros(big_d);

        for i in 0..big_d {
            for j in 0..d {
                // Box-Muller: u1, u2 ~ Uniform(0,1) → z ~ N(0,1)
                let u1: f64 = rng.random_range(1e-10_f64..1.0_f64);
                let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                weights[[i, j]] = z * scale;
            }
            offsets[i] = rng.random_range(0.0_f64..(2.0 * PI));
        }

        self.random_weights = Some(weights);
        self.random_offsets = Some(offsets);
    }

    /// Map a single sample `x ∈ ℝᵈ` to its quantum feature vector `φ(x) ∈ ℝᴰ`.
    fn quantum_feature_map(&self, x: &ArrayView1<'_, f64>) -> SpatialResult<Array1<f64>> {
        let weights = self.random_weights.as_ref().ok_or_else(|| {
            SpatialError::InvalidInput("Model not fitted: call fit() first".to_string())
        })?;
        let offsets = self.random_offsets.as_ref().ok_or_else(|| {
            SpatialError::InvalidInput("Model not fitted: call fit() first".to_string())
        })?;

        let big_d = weights.nrows();
        let scale = (2.0 / big_d as f64).sqrt();

        let mut phi = Array1::<f64>::zeros(big_d);
        for i in 0..big_d {
            let row = weights.row(i);
            // dot product ωᵢᵀ x
            let dot: f64 = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            phi[i] = scale * (dot + offsets[i]).cos();
        }
        Ok(phi)
    }

    /// Compute the quantum kernel k(a, b) = φ(a)ᵀ φ(b).
    fn quantum_kernel(
        &self,
        a: &ArrayView1<'_, f64>,
        b: &ArrayView1<'_, f64>,
    ) -> SpatialResult<f64> {
        let phi_a = self.quantum_feature_map(a)?;
        let phi_b = self.quantum_feature_map(b)?;
        Ok(phi_a.iter().zip(phi_b.iter()).map(|(ai, bi)| ai * bi).sum())
    }

    /// Train the model on labelled data.
    ///
    /// # Arguments
    /// * `x` - Training features matrix of shape `(n_samples, n_features)`.
    /// * `y` - Binary labels `{+1, -1}` of shape `(n_samples,)`.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidInput`] for empty inputs, shape mismatch,
    /// invalid label values, or non-positive `regularization`.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> SpatialResult<()> {
        let (n, d) = x.dim();

        if n == 0 {
            return Err(SpatialError::InvalidInput(
                "Training set must be non-empty".to_string(),
            ));
        }
        if y.len() != n {
            return Err(SpatialError::InvalidInput(format!(
                "x has {} rows but y has {} elements",
                n,
                y.len()
            )));
        }
        if self.regularization <= 0.0 {
            return Err(SpatialError::InvalidInput(
                "regularization (C) must be positive".to_string(),
            ));
        }
        // Validate labels are ±1
        for (i, &yi) in y.iter().enumerate() {
            if (yi - 1.0).abs() > 1e-9 && (yi + 1.0).abs() > 1e-9 {
                return Err(SpatialError::InvalidInput(format!(
                    "Label y[{}] = {} is not in {{-1, +1}}",
                    i, yi
                )));
            }
        }

        // Auto-tune bandwidth via median heuristic on pairwise distances
        let mut sq_dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let sq: f64 = (0..d).map(|k| (x[[i, k]] - x[[j, k]]).powi(2)).sum();
                sq_dists.push(sq);
            }
        }
        if !sq_dists.is_empty() {
            sq_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sq_dists[sq_dists.len() / 2];
            self.bandwidth = median.sqrt().max(1e-6);
        }

        // Initialise quantum random Fourier features
        self.init_random_features(d);

        // Build kernel matrix K[i,j] = k(xᵢ, xⱼ)
        let mut kernel_matrix = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let kij = self.quantum_kernel(&x.row(i), &x.row(j))?;
                kernel_matrix[[i, j]] = kij;
                kernel_matrix[[j, i]] = kij;
            }
        }

        // Simplified SMO: coordinate ascent on the dual
        // max  Σᵢ αᵢ - ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K[i,j]
        // s.t. 0 ≤ αᵢ ≤ C
        let mut alpha = vec![0.0f64; n];
        let max_smo_iter = 200;
        let tol = 1e-4;

        for _ in 0..max_smo_iter {
            let mut changed = false;

            for i in 0..n {
                // Compute decision function at xᵢ
                let fi: f64 = alpha
                    .iter()
                    .enumerate()
                    .map(|(j, &aj)| aj * y[j] * kernel_matrix[[i, j]])
                    .sum::<f64>()
                    + self.bias;

                let ri = fi * y[i] - 1.0;

                // Check KKT violation
                let kkt_violated = (ri < -tol && alpha[i] < self.regularization - tol)
                    || (ri > tol && alpha[i] > tol);

                if !kkt_violated {
                    continue;
                }

                // Pick second variable heuristically: most violating
                let j = (0..n)
                    .filter(|&k| k != i)
                    .max_by(|&a, &b| {
                        let fa: f64 = alpha
                            .iter()
                            .enumerate()
                            .map(|(l, &al)| al * y[l] * kernel_matrix[[a, l]])
                            .sum::<f64>()
                            + self.bias;
                        let fb: f64 = alpha
                            .iter()
                            .enumerate()
                            .map(|(l, &al)| al * y[l] * kernel_matrix[[b, l]])
                            .sum::<f64>()
                            + self.bias;
                        (fa * y[a] - 1.0)
                            .abs()
                            .partial_cmp(&(fb * y[b] - 1.0).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((i + 1) % n);

                // Analytic step
                let eta =
                    kernel_matrix[[i, i]] + kernel_matrix[[j, j]] - 2.0 * kernel_matrix[[i, j]];

                if eta <= 1e-12 {
                    continue;
                }

                let fj: f64 = alpha
                    .iter()
                    .enumerate()
                    .map(|(l, &al)| al * y[l] * kernel_matrix[[j, l]])
                    .sum::<f64>()
                    + self.bias;

                // Use SMO error terms E_i = f(x_i) - y_i, E_j = f(x_j) - y_j
                // to correctly kick alpha off zero on the first iteration
                let e_i = fi - y[i];
                let e_j = fj - y[j];
                let alpha_j_new =
                    (alpha[j] + y[j] * (e_i - e_j) / eta).clamp(0.0, self.regularization);
                let alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new);
                let alpha_i_new = alpha_i_new.clamp(0.0, self.regularization);

                if (alpha_i_new - alpha[i]).abs() > 1e-8 {
                    let delta_i = alpha_i_new - alpha[i];
                    let delta_j = alpha_j_new - alpha[j];
                    alpha[i] = alpha_i_new;
                    alpha[j] = alpha_j_new;

                    // Update bias
                    let b_i = -fi
                        - y[i] * delta_i * kernel_matrix[[i, i]]
                        - y[j] * delta_j * kernel_matrix[[i, j]];
                    let b_j = -fj
                        - y[i] * delta_i * kernel_matrix[[i, j]]
                        - y[j] * delta_j * kernel_matrix[[j, j]];

                    if alpha[i] > tol && alpha[i] < self.regularization - tol {
                        self.bias += b_i;
                    } else if alpha[j] > tol && alpha[j] < self.regularization - tol {
                        self.bias += b_j;
                    } else {
                        self.bias += (b_i + b_j) * 0.5;
                    }

                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Store support vectors (αᵢ > threshold)
        let sv_threshold = 1e-6;
        self.support_vectors.clear();
        self.alphas.clear();
        for i in 0..n {
            if alpha[i] > sv_threshold {
                self.support_vectors.push(x.row(i).to_owned());
                self.alphas.push(alpha[i] * y[i]);
            }
        }

        Ok(())
    }

    /// Predict class labels for new samples.
    ///
    /// # Arguments
    /// * `x` - Test features matrix of shape `(n_test, n_features)`.
    ///
    /// # Returns
    /// Binary predictions `{+1, -1}` of shape `(n_test,)`.
    ///
    /// # Errors
    /// Returns error if the model has not been fitted or if `x` has a different
    /// number of features than the training data.
    pub fn predict(&self, x: &Array2<f64>) -> SpatialResult<Array1<f64>> {
        if self.support_vectors.is_empty() {
            return Err(SpatialError::InvalidInput(
                "Model not fitted: call fit() first".to_string(),
            ));
        }

        let n_test = x.nrows();
        let mut preds = Array1::<f64>::zeros(n_test);

        for (idx, row) in x.outer_iter().enumerate() {
            let mut decision: f64 = self.bias;
            for (sv, &alpha) in self.support_vectors.iter().zip(self.alphas.iter()) {
                let kval = self.quantum_kernel(&row, &sv.view())?;
                decision += alpha * kval;
            }
            preds[idx] = if decision >= 0.0 { 1.0 } else { -1.0 };
        }

        Ok(preds)
    }
}

/// Quantum-Classical Hybrid Classifier
///
/// A high-level classifier that wraps [`QuantumSVMModel`] with optional
/// per-feature standardisation preprocessing. The preprocessing computes
/// z-score normalisation (zero mean, unit variance) from training data and
/// applies it consistently at test time.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_spatial::quantum_inspired::algorithms::quantum_machine_learning::QuantumClassifier;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     5.0, 5.0,
/// ])?;
/// let y = Array1::from_vec(vec![1.0, 1.0, 1.0, -1.0]);
///
/// let mut clf = QuantumClassifier::new(4, 1.0, true);
/// clf.fit(&x, &y)?;
///
/// let x_test = Array2::from_shape_vec((1, 2), vec![0.5, 0.5])?;
/// let preds = clf.predict(&x_test)?;
/// assert_eq!(preds.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct QuantumClassifier {
    /// The underlying quantum SVM
    svm: QuantumSVMModel,
    /// Whether to apply z-score normalisation before passing data to the SVM
    standardise: bool,
    /// Per-feature means (set during fit)
    feature_means: Option<Array1<f64>>,
    /// Per-feature standard deviations (set during fit)
    feature_stds: Option<Array1<f64>>,
}

impl QuantumClassifier {
    /// Construct a new `QuantumClassifier`.
    ///
    /// # Arguments
    /// * `n_qubits` - Passed through to [`QuantumSVMModel`].
    /// * `regularization` - SVM regularisation constant C.
    /// * `standardise` - If `true`, z-score normalise features before fitting.
    pub fn new(n_qubits: usize, regularization: f64, standardise: bool) -> Self {
        Self {
            svm: QuantumSVMModel::new(n_qubits, regularization),
            standardise,
            feature_means: None,
            feature_stds: None,
        }
    }

    /// Train the classifier.
    ///
    /// If `standardise` was set to `true`, this computes feature statistics
    /// from `x` and stores them for consistent use at predict time.
    ///
    /// # Errors
    /// Propagates errors from [`QuantumSVMModel::fit`].
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> SpatialResult<()> {
        let x_proc = if self.standardise {
            self.compute_and_store_stats(x)?
        } else {
            x.clone()
        };
        self.svm.fit(&x_proc, y)
    }

    /// Predict labels for new samples.
    ///
    /// Applies the same normalisation (if any) that was used during training.
    ///
    /// # Errors
    /// Propagates errors from [`QuantumSVMModel::predict`].
    pub fn predict(&self, x: &Array2<f64>) -> SpatialResult<Array1<f64>> {
        let x_proc = if self.standardise {
            self.apply_stored_stats(x)?
        } else {
            x.clone()
        };
        self.svm.predict(&x_proc)
    }

    /// Compute per-feature mean and std from `x`, store them, and return
    /// the standardised copy.
    fn compute_and_store_stats(&mut self, x: &Array2<f64>) -> SpatialResult<Array2<f64>> {
        let (n, d) = x.dim();
        if n == 0 {
            return Err(SpatialError::InvalidInput(
                "Cannot standardise an empty matrix".to_string(),
            ));
        }

        let mut means = Array1::<f64>::zeros(d);
        let mut stds = Array1::<f64>::zeros(d);

        for j in 0..d {
            let col = x.column(j);
            let mean = col.iter().sum::<f64>() / n as f64;
            let variance = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            means[j] = mean;
            stds[j] = variance.sqrt().max(1e-8);
        }

        self.feature_means = Some(means.clone());
        self.feature_stds = Some(stds.clone());

        let mut x_std = x.clone();
        for j in 0..d {
            for i in 0..n {
                x_std[[i, j]] = (x_std[[i, j]] - means[j]) / stds[j];
            }
        }
        Ok(x_std)
    }

    /// Apply previously stored statistics to standardise `x`.
    fn apply_stored_stats(&self, x: &Array2<f64>) -> SpatialResult<Array2<f64>> {
        let means = self.feature_means.as_ref().ok_or_else(|| {
            SpatialError::InvalidInput("Model not fitted: call fit() first".to_string())
        })?;
        let stds = self.feature_stds.as_ref().ok_or_else(|| {
            SpatialError::InvalidInput("Model not fitted: call fit() first".to_string())
        })?;

        let (n, d) = x.dim();
        if d != means.len() {
            return Err(SpatialError::InvalidInput(format!(
                "Expected {} features but got {}",
                means.len(),
                d
            )));
        }

        let mut x_std = x.clone();
        for j in 0..d {
            for i in 0..n {
                x_std[[i, j]] = (x_std[[i, j]] - means[j]) / stds[j];
            }
        }
        Ok(x_std)
    }

    /// Return the number of support vectors in the underlying SVM.
    pub fn num_support_vectors(&self) -> usize {
        self.svm.num_support_vectors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    /// Build a simple two-class dataset with two well-separated clusters.
    fn two_class_data() -> (Array2<f64>, Array1<f64>) {
        // 3 points near origin (+1) and 3 points far from origin (-1)
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.1, 0.1, -0.1, 0.2, 0.2, -0.1, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .expect("shape is valid");
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
        (x, y)
    }

    #[test]
    fn test_quantum_svm_fit_and_predict() {
        let (x, y) = two_class_data();
        let mut model = QuantumSVMModel::new(3, 1.0);
        model.fit(&x, &y).expect("fit should succeed");
        assert!(
            model.num_support_vectors() > 0,
            "model must produce support vectors"
        );

        let preds = model.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), x.nrows());

        // All predicted labels must be ±1
        for &p in preds.iter() {
            assert!(
                (p - 1.0).abs() < 1e-9 || (p + 1.0).abs() < 1e-9,
                "prediction {p} is not ±1"
            );
        }
    }

    #[test]
    fn test_quantum_classifier_with_standardisation() {
        let (x, y) = two_class_data();
        let mut clf = QuantumClassifier::new(3, 1.0, true);
        clf.fit(&x, &y).expect("fit should succeed");

        let preds = clf.predict(&x).expect("predict should succeed");
        assert_eq!(preds.len(), x.nrows());

        // Verify all outputs are binary
        for &p in preds.iter() {
            assert!(
                (p - 1.0).abs() < 1e-9 || (p + 1.0).abs() < 1e-9,
                "prediction {p} is not ±1"
            );
        }
    }

    #[test]
    fn test_svm_rejects_bad_labels() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("shape is valid");
        let y_bad = Array1::from_vec(vec![0.0, 1.0]); // 0 is not a valid label
        let mut model = QuantumSVMModel::new(2, 1.0);
        assert!(model.fit(&x, &y_bad).is_err());
    }

    #[test]
    fn test_predict_before_fit_errors() {
        let model = QuantumSVMModel::new(2, 1.0);
        let x_test = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("shape is valid");
        assert!(model.predict(&x_test).is_err());
    }
}
