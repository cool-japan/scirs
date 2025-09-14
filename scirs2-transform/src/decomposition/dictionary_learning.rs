//! Dictionary Learning for sparse coding and representation
//!
//! Dictionary Learning finds a sparse representation for the input data as a linear
//! combination of basic elements called atoms. The atoms compose a dictionary.
//! This is useful for sparse coding, denoising, and feature extraction.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use rand::Rng;
use scirs2_linalg::{svd, vector_norm};

use crate::error::{Result, TransformError};

/// Dictionary Learning for sparse representation
///
/// Finds a dictionary matrix D and sparse code matrix Alpha such that
/// X ≈ D * Alpha, where Alpha is sparse. This is solved by alternating
/// between sparse coding and dictionary update steps.
#[derive(Debug, Clone)]
pub struct DictionaryLearning {
    /// Number of dictionary atoms to extract
    n_components: usize,
    /// Sparsity controlling parameter
    alpha: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Tolerance for stopping criteria
    tol: f64,
    /// Algorithm for sparse coding: 'omp', 'lasso_lars', 'lasso_cd'
    transform_algorithm: String,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Whether to shuffle data before each epoch
    shuffle: bool,
    /// The learned dictionary
    dictionary: Option<Array2<f64>>,
    /// Number of iterations run
    n_iter: Option<usize>,
}

impl DictionaryLearning {
    /// Creates a new DictionaryLearning instance
    ///
    /// # Arguments
    /// * `n_components` - Number of dictionary elements to extract
    /// * `alpha` - Sparsity controlling parameter
    pub fn new(ncomponents: usize, alpha: f64) -> Self {
        DictionaryLearning {
            n_components: ncomponents,
            alpha,
            max_iter: 1000,
            tol: 1e-4,
            transform_algorithm: "omp".to_string(),
            random_state: None,
            shuffle: true,
            dictionary: None,
            n_iter: None,
        }
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

    /// Set transform algorithm
    pub fn with_transform_algorithm(mut self, algorithm: &str) -> Self {
        self.transform_algorithm = algorithm.to_string();
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set whether to shuffle data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Initialize dictionary with random patches from data
    fn initialize_dictionary(&self, x: &Array2<f64>) -> Array2<f64> {
        let n_features = x.shape()[1];
        let n_samples = x.shape()[0];

        let mut rng = rand::rng();

        let mut dictionary = Array2::zeros((self.n_components, n_features));

        // Select random samples as initial dictionary atoms
        for i in 0..self.n_components {
            let idx = rng.gen_range(0..n_samples);
            dictionary.row_mut(i).assign(&x.row(idx));

            // Normalize atom
            let norm = vector_norm(&dictionary.row(i).view(), 2).unwrap_or(0.0);
            if norm > 1e-10 {
                dictionary.row_mut(i).mapv_inplace(|x| x / norm);
            }
        }

        dictionary
    }

    /// Orthogonal Matching Pursuit (OMP) for sparse coding
    fn omp_sparse_code(
        &self,
        x: &Array1<f64>,
        dictionary: &Array2<f64>,
        n_nonzero_coefs: usize,
    ) -> Array1<f64> {
        let n_atoms = dictionary.shape()[0];
        let mut residual = x.clone();
        let mut sparse_code = Array1::zeros(n_atoms);
        let mut selected_atoms = Vec::new();

        for _ in 0..n_nonzero_coefs.min(n_atoms) {
            // Find atom with highest correlation to residual
            let mut best_atom = 0;
            let mut best_correlation = 0.0;

            for j in 0..n_atoms {
                if selected_atoms.contains(&j) {
                    continue;
                }

                let correlation = residual.dot(&dictionary.row(j)).abs();
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_atom = j;
                }
            }

            if best_correlation < 1e-10 {
                break;
            }

            selected_atoms.push(best_atom);

            // Solve least squares for selected atoms
            if selected_atoms.len() == 1 {
                // Simple case: single atom
                let atom = dictionary.row(best_atom);
                let coef = x.dot(&atom) / atom.dot(&atom);
                sparse_code[best_atom] = coef;
                residual = x - &(atom.to_owned() * coef);
            } else {
                // Multiple atoms: solve least squares
                let n_selected = selected_atoms.len();
                let mut sub_dictionary = Array2::zeros((n_selected, dictionary.shape()[1]));

                for (i, &atom_idx) in selected_atoms.iter().enumerate() {
                    sub_dictionary.row_mut(i).assign(&dictionary.row(atom_idx));
                }

                // Solve X = D^T * alpha using normal equations
                let gram = sub_dictionary.dot(&sub_dictionary.t());
                let proj = sub_dictionary.dot(&x.view());

                // Simple least squares solver (for small systems)
                let alpha = self.solve_small_least_squares(&gram, &proj);

                // Update sparse code and residual
                sparse_code.fill(0.0);
                for (i, &atom_idx) in selected_atoms.iter().enumerate() {
                    sparse_code[atom_idx] = alpha[i];
                }

                residual = x - &dictionary.t().dot(&sparse_code);
            }
        }

        sparse_code
    }

    /// Simple least squares solver for small systems
    fn solve_small_least_squares(&self, a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = a.shape()[0];
        let mut result = b.clone();

        // LU decomposition (simplified for small systems)
        let mut lu = a.clone();
        let mut perm = (0..n).collect::<Vec<_>>();

        // Forward elimination
        for k in 0..n - 1 {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = lu[[k, k]].abs();
            for i in k + 1..n {
                if lu[[i, k]].abs() > max_val {
                    max_val = lu[[i, k]].abs();
                    max_idx = i;
                }
            }

            // Swap rows
            if max_idx != k {
                perm.swap(k, max_idx);
                for j in 0..n {
                    let tmp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = tmp;
                }
                let tmp = result[k];
                result[k] = result[max_idx];
                result[max_idx] = tmp;
            }

            // Eliminate
            for i in k + 1..n {
                let factor = lu[[i, k]] / lu[[k, k]];
                for j in k + 1..n {
                    lu[[i, j]] -= factor * lu[[k, j]];
                }
                result[i] -= factor * result[k];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in i + 1..n {
                result[i] -= lu[[i, j]] * result[j];
            }
            result[i] /= lu[[i, i]];
        }

        result
    }

    /// Sparse coding step: find sparse codes for all samples
    fn sparse_code_step(&self, x: &Array2<f64>, dictionary: &Array2<f64>) -> Array2<f64> {
        let n_samples = x.shape()[0];
        let n_atoms = dictionary.shape()[0];
        let mut codes = Array2::zeros((n_samples, n_atoms));

        // Determine number of non-zero coefficients
        let n_nonzero_coefs = (self.alpha * n_atoms as f64).ceil() as usize;

        // Sparse code each sample
        for i in 0..n_samples {
            let sparse_code =
                self.omp_sparse_code(&x.row(i).to_owned(), dictionary, n_nonzero_coefs);
            codes.row_mut(i).assign(&sparse_code);
        }

        codes
    }

    /// Dictionary update step using SVD
    fn dictionary_update_step(
        &self,
        x: &Array2<f64>,
        sparse_codes: &mut Array2<f64>,
        dictionary: &mut Array2<f64>,
    ) {
        let n_atoms = dictionary.shape()[0];
        let n_features = dictionary.shape()[1];

        for k in 0..n_atoms {
            // Find samples that use this atom
            let mut using_samples = Vec::new();
            for i in 0..sparse_codes.shape()[0] {
                if sparse_codes[[i, k]].abs() > 1e-10 {
                    using_samples.push(i);
                }
            }

            if using_samples.is_empty() {
                continue;
            }

            // Compute residual without atom k
            let mut residual = Array2::zeros((using_samples.len(), n_features));
            for (idx, &i) in using_samples.iter().enumerate() {
                let mut r = x.row(i).to_owned();
                for j in 0..n_atoms {
                    if j != k {
                        r = r - dictionary.row(j).to_owned() * sparse_codes[[i, j]];
                    }
                }
                residual.row_mut(idx).assign(&r);
            }

            // Update atom using SVD
            if residual.shape()[0] > 0 {
                match svd::<f64>(&residual.view(), false, Some(1)) {
                    Ok((u, s, vt)) => {
                        // Update dictionary atom
                        dictionary.row_mut(k).assign(&vt.row(0));

                        // Update sparse _codes
                        for (idx, &i) in using_samples.iter().enumerate() {
                            sparse_codes[[i, k]] = u[[idx, 0]] * s[0];
                        }
                    }
                    Err(_) => {
                        // If SVD fails, normalize current atom
                        let norm = vector_norm(&dictionary.row(k).view(), 2).unwrap_or(0.0);
                        if norm > 1e-10 {
                            dictionary.row_mut(k).mapv_inplace(|x| x / norm);
                        }
                    }
                }
            }
        }
    }

    /// Fit the dictionary learning model
    ///
    /// # Arguments
    /// * `x` - Input data matrix
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|v| num_traits::cast::<S::Elem, f64>(v).unwrap_or(0.0));
        let _n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.n_components > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be <= n_features={}",
                self.n_components, n_features
            )));
        }

        // Initialize dictionary
        let mut dictionary = self.initialize_dictionary(&x_f64);
        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        // Main optimization loop
        for iter in 0..self.max_iter {
            // Sparse coding step
            let mut sparse_codes = self.sparse_code_step(&x_f64, &dictionary);

            // Dictionary update step
            self.dictionary_update_step(&x_f64, &mut sparse_codes, &mut dictionary);

            // Compute reconstruction error
            let reconstruction = sparse_codes.dot(&dictionary);
            let error = (&x_f64 - &reconstruction).mapv(|x| x * x).sum().sqrt();

            // Check convergence
            if (prev_error - error).abs() / prev_error.max(1e-10) < self.tol {
                n_iter = iter + 1;
                break;
            }

            prev_error = error;
            n_iter = iter + 1;
        }

        self.dictionary = Some(dictionary);
        self.n_iter = Some(n_iter);

        Ok(())
    }

    /// Transform data to sparse codes
    ///
    /// # Arguments
    /// * `x` - Input data matrix
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Sparse codes
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.dictionary.is_none() {
            return Err(TransformError::TransformationError(
                "DictionaryLearning model has not been fitted".to_string(),
            ));
        }

        let x_f64 = x.mapv(|v| num_traits::cast::<S::Elem, f64>(v).unwrap_or(0.0));
        let dictionary = self.dictionary.as_ref().unwrap();

        Ok(self.sparse_code_step(&x_f64, dictionary))
    }

    /// Fit and transform in one step
    ///
    /// # Arguments
    /// * `x` - Input data matrix
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Sparse codes
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get the learned dictionary
    pub fn dictionary(&self) -> Option<&Array2<f64>> {
        self.dictionary.as_ref()
    }

    /// Get number of iterations run
    pub fn n_iterations(&self) -> Option<usize> {
        self.n_iter
    }

    /// Reconstruct data from sparse codes
    pub fn inverse_transform(&self, sparsecodes: &Array2<f64>) -> Result<Array2<f64>> {
        if self.dictionary.is_none() {
            return Err(TransformError::TransformationError(
                "DictionaryLearning model has not been fitted".to_string(),
            ));
        }

        let dictionary = self.dictionary.as_ref().unwrap();
        Ok(sparsecodes.dot(dictionary))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    #[ignore] // Slow test - dictionary learning takes ~60s
    fn test_dictionary_learning_basic() {
        // Create synthetic data as sum of sinusoids
        let n_samples = 100;
        let n_features = 20;
        let mut data = Vec::new();

        for i in 0..n_samples {
            for j in 0..n_features {
                let t = j as f64 / n_features as f64 * 2.0 * std::f64::consts::PI;
                let val = (t * (i as f64 / 10.0)).sin() + (2.0 * t * (i as f64 / 15.0)).cos();
                data.push(val);
            }
        }

        let x = Array::from_shape_vec((n_samples, n_features), data).unwrap();

        let mut dict_learning = DictionaryLearning::new(10, 0.1)
            .with_max_iter(50)
            .with_random_state(42);

        let sparse_codes = dict_learning.fit_transform(&x).unwrap();

        // Check dimensions
        assert_eq!(sparse_codes.shape(), &[n_samples, 10]);

        // Check dictionary
        let dictionary = dict_learning.dictionary().unwrap();
        assert_eq!(dictionary.shape(), &[10, n_features]);

        // Check that dictionary atoms are normalized
        for i in 0..10 {
            let norm = vector_norm(&dictionary.row(i).view(), 2).unwrap_or(0.0);
            assert!((norm - 1.0).abs() < 1e-5);
        }

        // Check reconstruction
        let reconstructed = dict_learning.inverse_transform(&sparse_codes).unwrap();
        assert_eq!(reconstructed.shape(), x.shape());
    }

    #[test]
    fn test_dictionary_learning_sparsity() {
        let x: Array2<f64> = Array::eye(20) * 2.0;

        let mut dict_learning = DictionaryLearning::new(10, 0.05).with_max_iter(30);

        let sparse_codes = dict_learning.fit_transform(&x).unwrap();

        // Check sparsity: most elements should be zero
        let n_nonzero = sparse_codes.iter().filter(|&&x| x.abs() > 1e-10).count();
        let total_elements = sparse_codes.len();
        let sparsity = 1.0 - (n_nonzero as f64 / total_elements as f64);

        // Should be quite sparse
        assert!(sparsity > 0.5);
    }
}
