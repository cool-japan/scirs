//! Non-negative Matrix Factorization (NMF) for decomposition and feature extraction
//!
//! NMF decomposes a non-negative matrix V into two non-negative matrices W and H
//! such that V ≈ WH. This is useful for parts-based representation and interpretable
//! feature extraction.

use scirs2_core::ndarray::{Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::Rng;

use crate::error::{Result, TransformError};
// use statrs::statistics::Statistics; // TODO: Add statrs dependency - needs generic type fixes

/// Non-negative Matrix Factorization (NMF)
///
/// Finds two non-negative matrices W and H whose product approximates the
/// non-negative input matrix V. The objective function is minimized with
/// multiplicative update rules.
#[derive(Debug, Clone)]
pub struct NMF {
    /// Number of components (latent features)
    n_components: usize,
    /// Initialization method: 'random', 'nndsvd', 'nndsvda', 'nndsvdar'
    init: String,
    /// Solver: 'mu' (multiplicative update), 'cd' (coordinate descent)
    solver: String,
    /// Beta divergence parameter (0: Euclidean, 1: KL divergence, 2: Frobenius)
    beta_loss: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Tolerance for stopping criteria
    tol: f64,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Regularization parameter for components
    alpha: f64,
    /// L1 ratio for regularization (0: L2, 1: L1)
    l1_ratio: f64,
    /// The basis matrix W
    components: Option<Array2<f64>>,
    /// The coefficient matrix H
    coefficients: Option<Array2<f64>>,
    /// Reconstruction error
    reconstruction_err: Option<f64>,
    /// Number of iterations run
    n_iter: Option<usize>,
}

impl NMF {
    /// Creates a new NMF instance
    ///
    /// # Arguments
    /// * `n_components` - Number of components to extract
    pub fn new(ncomponents: usize) -> Self {
        NMF {
            n_components: ncomponents,
            init: "random".to_string(),
            solver: "mu".to_string(),
            beta_loss: 2.0, // Frobenius norm
            max_iter: 200,
            tol: 1e-4,
            random_state: None,
            alpha: 0.0,
            l1_ratio: 0.0,
            components: None,
            coefficients: None,
            reconstruction_err: None,
            n_iter: None,
        }
    }

    /// Set the initialization method
    pub fn with_init(mut self, init: &str) -> Self {
        self.init = init.to_string();
        self
    }

    /// Set the solver
    pub fn with_solver(mut self, solver: &str) -> Self {
        self.solver = solver.to_string();
        self
    }

    /// Set the beta divergence parameter
    pub fn with_beta_loss(mut self, beta: f64) -> Self {
        self.beta_loss = beta;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, maxiter: usize) -> Self {
        self.max_iter = maxiter;
        self
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set regularization parameters
    pub fn with_regularization(mut self, alpha: f64, l1ratio: f64) -> Self {
        self.alpha = alpha;
        self.l1_ratio = l1ratio;
        self
    }

    /// Initialize matrices with random non-negative values
    fn random_initialization(&self, v: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (n_samples, n_features) = (v.shape()[0], v.shape()[1]);
        let mut rng = scirs2_core::random::rng();

        let scale = (v.mean().unwrap() / self.n_components as f64).sqrt();

        let mut w = Array2::zeros((n_samples, self.n_components));
        let mut h = Array2::zeros((self.n_components, n_features));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                w[[i, j]] = rng.random::<f64>() * scale;
            }
        }

        for i in 0..self.n_components {
            for j in 0..n_features {
                h[[i, j]] = rng.random::<f64>() * scale;
            }
        }

        (w, h)
    }

    /// NNDSVD initialization (Non-negative Double Singular Value Decomposition)
    fn nndsvd_initialization(&self, v: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let (n_samples, n_features) = (v.shape()[0], v.shape()[1]);

        // Perform SVD
        let (u, s, vt) = match scirs2_linalg::svd::<f64>(&v.view(), true, None) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };

        let mut w = Array2::zeros((n_samples, self.n_components));
        let mut h = Array2::zeros((self.n_components, n_features));

        // Use the first n_components singular vectors
        for j in 0..self.n_components {
            let x = u.column(j);
            let y = vt.row(j);

            // Make non-negative
            let x_pos = x.mapv(|v| v.max(0.0));
            let x_neg = x.mapv(|v| (-v).max(0.0));
            let y_pos = y.mapv(|v| v.max(0.0));
            let y_neg = y.mapv(|v| (-v).max(0.0));

            let x_pos_norm = x_pos.dot(&x_pos).sqrt();
            let x_neg_norm = x_neg.dot(&x_neg).sqrt();
            let y_pos_norm = y_pos.dot(&y_pos).sqrt();
            let y_neg_norm = y_neg.dot(&y_neg).sqrt();

            let m_pos = x_pos_norm * y_pos_norm;
            let m_neg = x_neg_norm * y_neg_norm;

            if m_pos > m_neg {
                for i in 0..n_samples {
                    w[[i, j]] = (s[j].sqrt() * x_pos[i] / x_pos_norm).max(0.0);
                }
                for i in 0..n_features {
                    h[[j, i]] = (s[j].sqrt() * y_pos[i] / y_pos_norm).max(0.0);
                }
            } else {
                for i in 0..n_samples {
                    w[[i, j]] = (s[j].sqrt() * x_neg[i] / x_neg_norm).max(0.0);
                }
                for i in 0..n_features {
                    h[[j, i]] = (s[j].sqrt() * y_neg[i] / y_neg_norm).max(0.0);
                }
            }
        }

        Ok((w, h))
    }

    /// Initialize W and H matrices
    fn initialize_matrices(&self, v: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        match self.init.as_str() {
            "random" => Ok(self.random_initialization(v)),
            "nndsvd" => self.nndsvd_initialization(v),
            _ => Ok(self.random_initialization(v)),
        }
    }

    /// Compute Frobenius norm loss
    fn frobenius_loss(&self, v: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> f64 {
        let wh = w.dot(h);
        let diff = v - &wh;
        diff.mapv(|x| x * x).sum().sqrt()
    }

    /// Multiplicative update for W
    fn update_w(&self, v: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
        let eps = 1e-10;
        let wh = w.dot(h);

        // Numerator: V * H^T
        let numerator = v.dot(&h.t());

        // Denominator: W * H * H^T + regularization
        let mut denominator = wh.dot(&h.t());

        // Add L2 regularization
        if self.alpha > 0.0 && self.l1_ratio < 1.0 {
            let l2_reg = self.alpha * (1.0 - self.l1_ratio);
            denominator = &denominator + &(w * l2_reg);
        }

        // Add L1 regularization
        if self.alpha > 0.0 && self.l1_ratio > 0.0 {
            let l1_reg = self.alpha * self.l1_ratio;
            denominator = denominator.mapv(|x| x + l1_reg);
        }

        // Multiplicative update
        let mut w_new = w * &(numerator / (denominator + eps));

        // Ensure non-negativity
        w_new.mapv_inplace(|x| x.max(eps));

        w_new
    }

    /// Multiplicative update for H
    fn update_h(&self, v: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
        let eps = 1e-10;
        let wh = w.dot(h);

        // Numerator: W^T * V
        let numerator = w.t().dot(v);

        // Denominator: W^T * W * H + regularization
        let mut denominator = w.t().dot(&wh);

        // Add L2 regularization
        if self.alpha > 0.0 && self.l1_ratio < 1.0 {
            let l2_reg = self.alpha * (1.0 - self.l1_ratio);
            denominator = &denominator + &(h * l2_reg);
        }

        // Add L1 regularization
        if self.alpha > 0.0 && self.l1_ratio > 0.0 {
            let l1_reg = self.alpha * self.l1_ratio;
            denominator = denominator.mapv(|x| x + l1_reg);
        }

        // Multiplicative update
        let mut h_new = h * &(numerator / (denominator + eps));

        // Ensure non-negativity
        h_new.mapv_inplace(|x| x.max(eps));

        h_new
    }

    /// Coordinate descent update for W
    fn update_w_cd(&self, v: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
        let eps = 1e-10;
        let (n_samples, n_components) = w.dim();
        let mut w_new = w.clone();

        // Precompute H * H^T for efficiency
        let hht = h.dot(&h.t());

        for i in 0..n_samples {
            for j in 0..n_components {
                // Compute residual without contribution from w[i,j]
                let mut numerator = 0.0;
                let mut denominator = hht[[j, j]];

                // Compute v[i,:] * h[j,:] (numerator)
                for k in 0..h.ncols() {
                    numerator += v[[i, k]] * h[[j, k]];
                }

                // Compute w[i,:] * (H * H^T)[j,:] excluding w[i,j]
                for k in 0..n_components {
                    if k != j {
                        numerator -= w_new[[i, k]] * hht[[k, j]];
                    }
                }

                // Add regularization terms
                if self.alpha > 0.0 {
                    if self.l1_ratio > 0.0 {
                        // L1 regularization (soft thresholding)
                        let l1_penalty = self.alpha * self.l1_ratio;
                        numerator -= l1_penalty;
                    }
                    if self.l1_ratio < 1.0 {
                        // L2 regularization
                        let l2_penalty = self.alpha * (1.0 - self.l1_ratio);
                        denominator += l2_penalty;
                        numerator -= l2_penalty * w_new[[i, j]];
                    }
                }

                // Update w[i,j]
                let new_val = if denominator > eps {
                    (numerator / denominator).max(eps)
                } else {
                    eps
                };

                w_new[[i, j]] = new_val;
            }
        }

        w_new
    }

    /// Coordinate descent update for H
    fn update_h_cd(&self, v: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
        let eps = 1e-10;
        let (n_components, n_features) = h.dim();
        let mut h_new = h.clone();

        // Precompute W^T * W for efficiency
        let wtw = w.t().dot(w);

        for i in 0..n_components {
            for j in 0..n_features {
                // Compute residual without contribution from h[i,j]
                let mut numerator = 0.0;
                let mut denominator = wtw[[i, i]];

                // Compute w[:,i]^T * v[:,j] (numerator)
                for k in 0..w.nrows() {
                    numerator += w[[k, i]] * v[[k, j]];
                }

                // Compute (W^T * W)[i,:] * h[:,j] excluding h[i,j]
                for k in 0..n_components {
                    if k != i {
                        numerator -= wtw[[i, k]] * h_new[[k, j]];
                    }
                }

                // Add regularization terms
                if self.alpha > 0.0 {
                    if self.l1_ratio > 0.0 {
                        // L1 regularization (soft thresholding)
                        let l1_penalty = self.alpha * self.l1_ratio;
                        numerator -= l1_penalty;
                    }
                    if self.l1_ratio < 1.0 {
                        // L2 regularization
                        let l2_penalty = self.alpha * (1.0 - self.l1_ratio);
                        denominator += l2_penalty;
                        numerator -= l2_penalty * h_new[[i, j]];
                    }
                }

                // Update h[i,j]
                let new_val = if denominator > eps {
                    (numerator / denominator).max(eps)
                } else {
                    eps
                };

                h_new[[i, j]] = new_val;
            }
        }

        h_new
    }

    /// Fit NMF model to data
    ///
    /// # Arguments
    /// * `x` - Input data matrix (must be non-negative)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        // Validate non-negativity before conversion
        for elem in x.iter() {
            let val = NumCast::from(*elem).unwrap_or(0.0);
            if val < 0.0 {
                return Err(TransformError::InvalidInput(
                    "NMF requires non-negative input data".to_string(),
                ));
            }
        }

        // Convert to f64
        let v = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let (n_samples, n_features) = (v.shape()[0], v.shape()[1]);

        if self.n_components > n_features.min(n_samples) {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= min(n_samples={}, n_features={})",
                self.n_components, n_samples, n_features
            )));
        }

        // Initialize W and H
        let (mut w, mut h) = self.initialize_matrices(&v)?;

        let mut prev_error = self.frobenius_loss(&v, &w, &h);
        let mut n_iter = 0;

        // Optimization loop
        for iter in 0..self.max_iter {
            // Update W and H
            if self.solver == "mu" {
                h = self.update_h(&v, &w, &h);
                w = self.update_w(&v, &w, &h);
            } else if self.solver == "cd" {
                h = self.update_h_cd(&v, &w, &h);
                w = self.update_w_cd(&v, &w, &h);
            } else {
                return Err(TransformError::InvalidInput(format!(
                    "Unknown solver '{}'. Supported solvers: 'mu', 'cd'",
                    self.solver
                )));
            }

            // Compute error
            let error = self.frobenius_loss(&v, &w, &h);

            // Check convergence
            if (prev_error - error).abs() / prev_error.max(1e-10) < self.tol {
                n_iter = iter + 1;
                break;
            }

            prev_error = error;
            n_iter = iter + 1;
        }

        self.components = Some(h);
        self.coefficients = Some(w);
        self.reconstruction_err = Some(prev_error);
        self.n_iter = Some(n_iter);

        Ok(())
    }

    /// Transform data using fitted NMF model
    ///
    /// # Arguments
    /// * `x` - Input data matrix (must be non-negative)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Transformed data (W matrix)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "NMF model has not been fitted".to_string(),
            ));
        }

        // Validate non-negativity before conversion
        for elem in x.iter() {
            let val = NumCast::from(*elem).unwrap_or(0.0);
            if val < 0.0 {
                return Err(TransformError::InvalidInput(
                    "NMF requires non-negative input data".to_string(),
                ));
            }
        }

        // Convert to f64
        let v = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

        let h = self.components.as_ref().unwrap();
        let n_samples = v.shape()[0];

        // Initialize W randomly
        let mut rng = scirs2_core::random::rng();

        let scale = (v.mean().unwrap() / self.n_components as f64).sqrt();
        let mut w = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            for j in 0..self.n_components {
                w[[i, j]] = rng.random::<f64>() * scale;
            }
        }

        // Update W while keeping H fixed
        for _ in 0..self.max_iter {
            w = self.update_w(&v, &w, h);
        }

        Ok(w)
    }

    /// Fit and transform in one step
    ///
    /// # Arguments
    /// * `x` - Input data matrix (must be non-negative)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Transformed data (W matrix)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        Ok(self.coefficients.as_ref().unwrap().clone())
    }

    /// Get the components (H matrix)
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Get the coefficients (W matrix)
    pub fn coefficients(&self) -> Option<&Array2<f64>> {
        self.coefficients.as_ref()
    }

    /// Get reconstruction error
    pub fn reconstruction_error(&self) -> Option<f64> {
        self.reconstruction_err
    }

    /// Get number of iterations run
    pub fn n_iterations(&self) -> Option<usize> {
        self.n_iter
    }

    /// Inverse transform - reconstruct data from transformed representation
    pub fn inverse_transform(&self, w: &Array2<f64>) -> Result<Array2<f64>> {
        if self.components.is_none() {
            return Err(TransformError::TransformationError(
                "NMF model has not been fitted".to_string(),
            ));
        }

        let h = self.components.as_ref().unwrap();
        Ok(w.dot(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_nmf_basic() {
        // Create non-negative data
        let x = Array::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0,
                5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0,
            ],
        )
        .unwrap();

        let mut nmf = NMF::new(2).with_max_iter(100).with_random_state(42);

        let w = nmf.fit_transform(&x).unwrap();

        // Check dimensions
        assert_eq!(w.shape(), &[6, 2]);

        // Check non-negativity
        for val in w.iter() {
            assert!(*val >= 0.0);
        }

        // Check components
        let h = nmf.components().unwrap();
        assert_eq!(h.shape(), &[2, 4]);

        for val in h.iter() {
            assert!(*val >= 0.0);
        }

        // Check reconstruction
        let x_reconstructed = nmf.inverse_transform(&w).unwrap();
        assert_eq!(x_reconstructed.shape(), x.shape());
    }

    #[test]
    fn test_nmf_regularization() {
        let x = Array2::<f64>::eye(10) + 0.1; // Add small value to ensure positivity

        let mut nmf = NMF::new(3).with_regularization(0.1, 0.5).with_max_iter(50);

        let result = nmf.fit_transform(&x);
        assert!(result.is_ok());

        let w = result.unwrap();
        assert_eq!(w.shape(), &[10, 3]);
    }

    #[test]
    fn test_nmf_negative_input() {
        let x = Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, -1.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        let mut nmf = NMF::new(2);
        let result = nmf.fit(&x);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("NMF requires non-negative input data"));
        }
    }

    #[test]
    fn test_nmf_coordinate_descent() {
        // Create non-negative data
        let x = Array::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0,
                5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0,
            ],
        )
        .unwrap();

        let mut nmf_cd = NMF::new(2)
            .with_solver("cd")
            .with_max_iter(100)
            .with_random_state(42);

        let w_cd = nmf_cd.fit_transform(&x).unwrap();

        // Check dimensions
        assert_eq!(w_cd.shape(), &[6, 2]);

        // Check non-negativity
        for val in w_cd.iter() {
            assert!(*val >= 0.0);
        }

        // Check components
        let h_cd = nmf_cd.components().unwrap();
        assert_eq!(h_cd.shape(), &[2, 4]);

        for val in h_cd.iter() {
            assert!(*val >= 0.0);
        }

        // Check reconstruction
        let x_reconstructed = nmf_cd.inverse_transform(&w_cd).unwrap();
        assert_eq!(x_reconstructed.shape(), x.shape());

        // Compare with multiplicative update solver
        let mut nmf_mu = NMF::new(2)
            .with_solver("mu")
            .with_max_iter(100)
            .with_random_state(42);

        let _w_mu = nmf_mu.fit_transform(&x).unwrap();

        // Both should converge and produce valid decompositions
        assert!(nmf_cd.reconstruction_error().unwrap() >= 0.0);
        assert!(nmf_mu.reconstruction_error().unwrap() >= 0.0);
    }

    #[test]
    fn test_nmf_invalid_solver() {
        let x = Array2::<f64>::eye(3) + 0.1;
        let mut nmf = NMF::new(2).with_solver("invalid");

        let result = nmf.fit(&x);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown solver"));
    }
}
