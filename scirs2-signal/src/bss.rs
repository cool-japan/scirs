// Blind source separation module
//
// This module implements various blind source separation (BSS) techniques for signal processing,
// including Independent Component Analysis (ICA), Principal Component Analysis (PCA),
// Non-negative Matrix Factorization (NMF), and related methods.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2, Axis};
// Use scirs2-linalg for linear algebra operations
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use scirs2_linalg::{eigh, solve, solve_multiple, svd};
use std::f64::consts::PI;

/// Type alias for multi-dataset JADE result
type JadeMultiResult = (Vec<Array2<f64>>, Vec<Array2<f64>>);

/// Configuration for blind source separation
#[derive(Debug, Clone)]
pub struct BssConfig {
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence threshold for iterative methods
    pub convergence_threshold: f64,
    /// Whether to use whitening preprocessing
    pub apply_whitening: bool,
    /// Whether to perform dimensionality reduction
    pub dimension_reduction: bool,
    /// Target dimensionality (if None, determined automatically)
    pub target_dimension: Option<usize>,
    /// Variance to preserve in dimensionality reduction (0.0-1.0)
    pub variance_threshold: f64,
    /// Learning rate for gradient-based methods
    pub learning_rate: f64,
    /// Whether to apply non-negativity constraint
    pub non_negative: bool,
    /// Regularization parameter
    pub regularization: f64,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Whether to use fixed-point algorithm for ICA
    pub use_fixed_point: bool,
    /// Whether to use parallel implementation for efficiency
    pub parallel: bool,
}

impl Default for BssConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            apply_whitening: true,
            dimension_reduction: false,
            target_dimension: None,
            variance_threshold: 0.95,
            learning_rate: 0.1,
            non_negative: false,
            regularization: 1e-4,
            random_seed: None,
            use_fixed_point: true,
            parallel: false,
        }
    }
}

/// ICA algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IcaMethod {
    /// FastICA algorithm
    FastICA,
    /// Infomax algorithm
    Infomax,
    /// JADE algorithm (Joint Approximate Diagonalization of Eigen-matrices)
    JADE,
    /// Extended Infomax algorithm (works with sub and super-Gaussian sources)
    ExtendedInfomax,
}

/// ICA nonlinearity functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonlinearityFunction {
    /// Logistic function (G(x) = tanh(x))
    Logistic,
    /// Exponential function (G(x) = x*exp(-x²/2))
    Exponential,
    /// Cubic function (G(x) = x³)
    Cubic,
    /// Hyperbolic tangent
    Tanh,
    /// Hyperbolic cosine
    Cosh,
}

/// Apply Principal Component Analysis (PCA) to separate mixed signals
///
/// PCA finds uncorrelated components that maximize variance.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn pca(signals: &Array2<f64>, config: &BssConfig) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Compute covariance matrix
    let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);

    // Perform eigendecomposition
    let (eigvals, eigvecs) = match eigh(&cov.view()) {
        Ok((vals, vecs)) => (vals, vecs),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition".to_string(),
            ));
        }
    };

    // Sort eigenvectors by eigenvalues in descending order
    let mut indices: Vec<usize> = (0..n_signals).collect();
    indices.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());

    let mut sorted_eigvecs = Array2::<f64>::zeros((n_signals, n_signals));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_signals {
            sorted_eigvecs[[j, i]] = eigvecs[[j, idx]];
        }
    }

    // Determine how many components to keep
    let n_components = if let Some(dim) = config.target_dimension {
        dim.min(n_signals)
    } else if config.dimension_reduction {
        // Calculate the number of components to keep based on variance threshold
        let total_var = eigvals.sum();
        let mut cum_var = 0.0;
        let mut k = 0;

        for i in 0..indices.len() {
            cum_var += eigvals[indices[i]];
            k += 1;
            if cum_var / total_var >= config.variance_threshold {
                break;
            }
        }

        k
    } else {
        n_signals
    };

    // Extract principal components
    let transform = sorted_eigvecs.slice(s![.., 0..n_components]);
    let sources = transform.t().dot(&centered);

    // Calculate the mixing matrix (transform)
    let mixing = transform.to_owned();

    Ok((sources, mixing))
}

/// Apply Independent Component Analysis (ICA) to separate mixed signals
///
/// ICA finds statistically independent components that generated the mixed signals.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract (default: same as signals)
/// * `method` - ICA algorithm to use
/// * `nonlinearity` - Nonlinearity function for FastICA
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn ica(
    signals: &Array2<f64>,
    n_components: Option<usize>,
    method: IcaMethod,
    nonlinearity: NonlinearityFunction,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Determine number of components
    let n_comp = n_components.unwrap_or(n_signals);
    if n_comp > n_signals {
        return Err(SignalError::ValueError(format!(
            "Number of components ({}) cannot exceed number of signals ({})",
            n_comp, n_signals
        )));
    }

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Whitening (decorrelation + scaling)
    let (whitened, whitening_matrix) = if config.apply_whitening {
        whiten_signals(&centered)?
    } else {
        (centered.clone(), Array2::<f64>::eye(n_signals))
    };

    // Apply the requested ICA method
    let (sources, unmixing) = match method {
        IcaMethod::FastICA => fast_ica(&whitened, n_comp, nonlinearity, config)?,
        IcaMethod::Infomax => infomax_ica(&whitened, n_comp, config)?,
        IcaMethod::JADE => jade_ica(&whitened, n_comp, config)?,
        IcaMethod::ExtendedInfomax => extended_infomax_ica(&whitened, n_comp, config)?,
    };

    // Calculate mixing matrix
    // A = W^-1 * whitening_matrix^-1
    let ica_mixing = match solve_multiple(
        &unmixing.view(),
        &Array2::<f64>::eye(unmixing.dim().0).view(),
    ) {
        Ok(inv) => inv,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute ICA mixing matrix".to_string(),
            ));
        }
    };

    let whitening_inv = match solve_multiple(
        &whitening_matrix.view(),
        &Array2::<f64>::eye(whitening_matrix.dim().0).view(),
    ) {
        Ok(inv) => inv,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to invert whitening matrix".to_string(),
            ));
        }
    };

    let mixing = ica_mixing.dot(&whitening_inv);

    Ok((sources, mixing))
}

/// Perform whitening (sphering) of signals
///
/// Whitening decorrelates the signals and scales them to unit variance.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
///
/// # Returns
///
/// * Tuple containing (whitened signals, whitening matrix)
fn whiten_signals(signals: &Array2<f64>) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Compute covariance matrix
    let cov = signals.dot(&signals.t()) / (n_samples as f64 - 1.0);

    // Perform eigendecomposition
    let (eigvals, eigvecs) = match eigh(&cov.view()) {
        Ok((vals, vecs)) => (vals, vecs),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition".to_string(),
            ));
        }
    };

    // Create diagonal matrix of scaled eigenvalues
    let mut d_inv_sqrt = Array2::<f64>::zeros((n_signals, n_signals));
    for i in 0..n_signals {
        if eigvals[i] > 1e-10 {
            d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();
        }
    }

    // Compute whitening matrix
    let whitening_matrix = d_inv_sqrt.dot(&eigvecs.t());

    // Apply whitening
    let whitened = whitening_matrix.dot(signals);

    Ok((whitened, whitening_matrix))
}

/// Implement FastICA algorithm for ICA
///
/// FastICA is a computationally efficient method that uses fixed-point iteration.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract
/// * `nonlinearity` - Nonlinearity function for FastICA
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, unmixing matrix)
fn fast_ica(
    signals: &Array2<f64>,
    n_components: usize,
    nonlinearity: NonlinearityFunction,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Initialize random unmixing matrix
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut w = Array2::<f64>::zeros((n_components, n_signals));

    for i in 0..n_components {
        for j in 0..n_signals {
            w[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Function to apply nonlinearity and its derivative
    type NonlinearityFn = Box<dyn Fn(f64) -> f64>;
    let (g, g_prime): (NonlinearityFn, NonlinearityFn) = match nonlinearity {
        NonlinearityFunction::Logistic => (
            Box::new(|x: f64| x.tanh()),
            Box::new(|x: f64| 1.0 - x.tanh().powi(2)),
        ),
        NonlinearityFunction::Exponential => (
            Box::new(|x: f64| x * (-x * x / 2.0).exp()),
            Box::new(|x: f64| (-x * x / 2.0).exp() * (1.0 - x * x)),
        ),
        NonlinearityFunction::Cubic => (
            Box::new(|x: f64| x.powi(3)),
            Box::new(|x: f64| 3.0 * x.powi(2)),
        ),
        NonlinearityFunction::Tanh => (
            Box::new(|x: f64| x.tanh()),
            Box::new(|x: f64| 1.0 - x.tanh().powi(2)),
        ),
        NonlinearityFunction::Cosh => (
            Box::new(|x: f64| x.tanh()),
            Box::new(|x: f64| 1.0 - x.tanh().powi(2)),
        ),
    };

    // Apply FastICA algorithm (deflation approach)
    if config.use_fixed_point {
        // Apply fixed-point algorithm
        for p in 0..n_components {
            // Initialize the pth row of the unmixing matrix
            let mut wp = w.slice_mut(s![p, ..]).to_owned();

            // Normalize the initial vector
            let dot_product = wp.dot(&wp);
            let norm = f64::sqrt(dot_product);
            if norm > 0.0 {
                wp /= norm;
            }

            let mut w_old = Array1::<f64>::zeros(n_signals);
            let mut iteration = 0;

            // Fixed-point iteration
            loop {
                // Store previous weight vector
                w_old.assign(&wp);

                // Compute projections of signals onto current weight vector
                let mut projected = Array1::<f64>::zeros(n_samples);
                for j in 0..n_samples {
                    for i in 0..n_signals {
                        projected[j] += wp[i] * signals[[i, j]];
                    }
                }

                // Apply nonlinearity
                let mut gx = Array1::<f64>::zeros(n_samples);
                let mut g_prime_sum = 0.0;
                for j in 0..n_samples {
                    gx[j] = g(projected[j]);
                    g_prime_sum += g_prime(projected[j]);
                }
                g_prime_sum /= n_samples as f64;

                // Update weight vector
                let mut new_wp = Array1::<f64>::zeros(n_signals);
                for i in 0..n_signals {
                    let mut sum_gx_x = 0.0;
                    for j in 0..n_samples {
                        sum_gx_x += gx[j] * signals[[i, j]];
                    }
                    new_wp[i] = sum_gx_x / (n_samples as f64) - g_prime_sum * wp[i];
                }

                // Decorrelate from existing components (Gram-Schmidt orthogonalization)
                for i in 0..p {
                    let wi = w.slice(s![i, ..]);
                    let proj = new_wp.dot(&wi);
                    for j in 0..n_signals {
                        new_wp[j] -= proj * wi[j];
                    }
                }

                // Normalize
                let norm = (new_wp.dot(&new_wp)).sqrt();
                if norm > 0.0 {
                    new_wp /= norm;
                }

                wp = new_wp;

                // Copy back to unmixing matrix
                let mut wp_slice = w.slice_mut(s![p, ..]);
                wp_slice.assign(&wp);

                // Check for convergence
                let dot_product = w_old.dot(&wp).abs();
                if (1.0 - dot_product) < config.convergence_threshold {
                    break;
                }

                iteration += 1;
                if iteration >= config.max_iterations {
                    break;
                }
            }
        }
    } else {
        // Apply simple gradient algorithm (less efficient but more robust)
        let mut w_old = Array2::<f64>::zeros((n_components, n_signals));

        for _iteration in 0..config.max_iterations {
            // Store previous weight matrix
            w_old.assign(&w);

            // Compute projections
            let projected = w.dot(signals);

            // Apply nonlinearity
            let mut gx = Array2::<f64>::zeros(projected.dim());
            for i in 0..n_components {
                for j in 0..n_samples {
                    gx[[i, j]] = g(projected[[i, j]]);
                }
            }

            // Compute gradient
            let gradient = gx.dot(&signals.t()) / (n_samples as f64)
                - Array2::<f64>::eye(n_components) * w.mapv(|x: f64| g_prime(x)).mean().unwrap();

            // Update weight matrix
            w = &w + &(&gradient * config.learning_rate);

            // Decorrelate weights (symmetric decorrelation)
            let ww_t = w.dot(&w.t());
            let (eigvals, eigvecs) = match eigh(&ww_t.view()) {
                Ok((vals, vecs)) => (vals, vecs),
                Err(_) => {
                    return Err(SignalError::Compute(
                        "Failed to compute eigendecomposition in FastICA".to_string(),
                    ));
                }
            };

            let mut d_inv_sqrt = Array2::<f64>::zeros((n_components, n_components));
            for i in 0..n_components {
                if eigvals[i] > 1e-10 {
                    d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();
                }
            }

            w = eigvecs.dot(&d_inv_sqrt).dot(&eigvecs.t()).dot(&w);

            // Check for convergence
            let mut converged = true;
            for i in 0..n_components {
                let mut max_dot = 0.0;
                for j in 0..n_components {
                    let dot = w.slice(s![i, ..]).dot(&w_old.slice(s![j, ..])).abs();
                    if dot > max_dot {
                        max_dot = dot;
                    }
                }
                if (1.0 - max_dot) > config.convergence_threshold {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }
        }
    }

    // Extract the independent components
    let sources = w.dot(signals);

    Ok((sources, w))
}

/// Implement Infomax algorithm for ICA
///
/// Infomax is a neural network-based approach to ICA that maximizes information flow.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, unmixing matrix)
fn infomax_ica(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Initialize random unmixing matrix
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut w = Array2::<f64>::zeros((n_components, n_signals));

    for i in 0..n_components {
        for j in 0..n_signals {
            w[[i, j]] = normal.sample(&mut rng) * 0.1;
        }
    }

    // Use identity matrix as initial unmixing matrix
    let eye = Array2::<f64>::eye(n_components);
    for i in 0..n_components.min(n_signals) {
        w[[i, i]] = 1.0;
    }

    // Learning rate schedule
    let mut learning_rate = 0.01;
    let min_learning_rate = 0.0001;
    let decay_rate = 0.9;

    // Batch size for stochastic gradient descent
    let batch_size = 128.min(n_samples);
    let n_batches = n_samples / batch_size;

    // Apply Infomax algorithm
    for _iteration in 0..config.max_iterations {
        let mut delta_w_sum = Array2::<f64>::zeros((n_components, n_signals));

        // Process in batches
        for batch in 0..n_batches {
            let start = batch * batch_size;
            let end = (batch + 1) * batch_size;

            let x_batch = signals.slice(s![.., start..end]);
            let y = w.dot(&x_batch);

            // Apply logistic nonlinearity
            let mut y_sigmoid = Array2::<f64>::zeros(y.dim());
            for i in 0..n_components {
                for j in 0..batch_size {
                    y_sigmoid[[i, j]] = 1.0 / (1.0 + (-y[[i, j]]).exp());
                }
            }

            // Compute gradient
            let block = &eye - &(&y_sigmoid * 2.0).dot(&Array2::ones((batch_size, n_components)));
            let delta_w =
                &block.dot(&y_sigmoid.dot(&x_batch.t())) * (learning_rate / batch_size as f64);

            delta_w_sum += &delta_w;
        }

        // Update unmixing matrix
        let delta_w_avg = delta_w_sum / n_batches as f64;
        w = &w + &delta_w_avg;

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check for convergence (simplified)
        if delta_w_avg.mapv(|x: f64| x.abs()).mean().unwrap() < config.convergence_threshold {
            break;
        }
    }

    // Extract the independent components
    let sources = w.dot(signals);

    Ok((sources, w))
}

/// Implement JADE algorithm for ICA
///
/// JADE uses Joint Approximate Diagonalization of Eigenmatrices.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, unmixing matrix)
fn jade_ica(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Calculate covariance matrices
    let mut cumulants = Vec::new();

    // Use PCA as initial guess
    let (pca_sources, pca_mixing) = pca(signals, config)?;
    let pca_unmixing =
        match solve_multiple(&pca_mixing.view(), &Array2::<f64>::eye(n_signals).view()) {
            Ok(inv) => inv.slice(s![0..n_components, ..]).to_owned(),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to compute PCA unmixing matrix".to_string(),
                ));
            }
        };

    // Calculate cumulant matrices
    for k in 0..n_components {
        for l in k..n_components {
            let mut q = Array2::<f64>::zeros((n_components, n_components));

            // Compute fourth-order cross-cumulants
            for i in 0..n_components {
                for j in 0..n_components {
                    let mut cum = 0.0;

                    for t in 0..n_samples {
                        cum += pca_sources[[i, t]]
                            * pca_sources[[j, t]]
                            * pca_sources[[k, t]]
                            * pca_sources[[l, t]];
                    }

                    cum /= n_samples as f64;

                    // Remove Gaussian contribution
                    if i == j && k == l {
                        cum -= 1.0;
                    }
                    if i == k && j == l {
                        cum -= 1.0;
                    }
                    if i == l && j == k {
                        cum -= 1.0;
                    }

                    q[[i, j]] = cum;
                }
            }

            cumulants.push(q);
        }
    }

    // Joint diagonalization
    let mut v = Array2::<f64>::eye(n_components);

    for _ in 0..config.max_iterations {
        let mut diagonalized = true;

        for i in 0..n_components - 1 {
            for j in i + 1..n_components {
                // Givens rotation parameters
                let mut g11 = 0.0;
                let mut g12 = 0.0;
                let mut g21 = 0.0;
                let mut g22 = 0.0;

                for q in &cumulants {
                    g11 += q[[i, i]].powi(2) + q[[j, j]].powi(2);
                    g12 += q[[i, j]].powi(2) + q[[j, i]].powi(2);
                    g21 += q[[i, i]] * q[[i, j]] + q[[j, j]] * q[[j, i]];
                    g22 += q[[i, j]] * q[[j, i]] - q[[i, i]] * q[[j, j]];
                }

                // Calculate rotation angle
                let gamma = g12 - g11;
                let theta = if g22.abs() < 1e-10 {
                    PI / 4.0 * (g21 >= 0.0) as i32 as f64
                } else {
                    0.5 * (g21 / g22).atan()
                };

                // Check if rotation is needed
                if gamma.abs() > config.convergence_threshold
                    || theta.abs() > config.convergence_threshold
                {
                    diagonalized = false;

                    // Givens rotation matrix
                    let c = theta.cos();
                    let s = theta.sin();
                    let mut g = Array2::<f64>::eye(n_components);
                    g[[i, i]] = c;
                    g[[i, j]] = -s;
                    g[[j, i]] = s;
                    g[[j, j]] = c;

                    // Update V
                    v = v.dot(&g);

                    // Update cumulant matrices
                    for q in &mut cumulants {
                        *q = g.t().dot(q).dot(&g);
                    }
                }
            }
        }

        // Check if all matrices are jointly diagonalized
        if diagonalized {
            break;
        }
    }

    // Unmixing matrix is V * Wpca
    let w = v.dot(&pca_unmixing);

    // Extract the independent components
    let sources = w.dot(signals);

    Ok((sources, w))
}

/// Implement Extended Infomax algorithm for ICA
///
/// Extended Infomax works with both sub-Gaussian and super-Gaussian sources.
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_components` - Number of independent components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, unmixing matrix)
fn extended_infomax_ica(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Initialize random unmixing matrix
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut w = Array2::<f64>::zeros((n_components, n_signals));

    for i in 0..n_components {
        for j in 0..n_signals {
            w[[i, j]] = normal.sample(&mut rng) * 0.1;
        }
    }

    // Use identity matrix as initial unmixing matrix
    let eye = Array2::<f64>::eye(n_components);
    for i in 0..n_components.min(n_signals) {
        w[[i, i]] = 1.0;
    }

    // Learning rate schedule
    let mut learning_rate = 0.01;
    let min_learning_rate = 0.0001;
    let decay_rate = 0.9;

    // Batch size for stochastic gradient descent
    let batch_size = 128.min(n_samples);
    let n_batches = n_samples / batch_size;

    // Sub-Gaussian or super-Gaussian detection
    let mut is_super_gaussian = vec![true; n_components];

    // Apply Extended Infomax algorithm
    for iteration in 0..config.max_iterations {
        let mut delta_w_sum = Array2::<f64>::zeros((n_components, n_signals));

        // Process in batches
        for batch in 0..n_batches {
            let start = batch * batch_size;
            let end = (batch + 1) * batch_size;

            let x_batch = signals.slice(s![.., start..end]);
            let y = w.dot(&x_batch);

            // Determine if signals are sub or super-Gaussian
            if iteration % 10 == 0 {
                for i in 0..n_components {
                    let mut kurtosis = 0.0;
                    for j in 0..batch_size {
                        kurtosis += y[[i, j]].powi(4);
                    }
                    kurtosis = kurtosis / batch_size as f64 - 3.0;
                    is_super_gaussian[i] = kurtosis > 0.0;
                }
            }

            // Compute nonlinearity based on sub/super-Gaussian nature
            let mut k = Array2::<f64>::zeros(y.dim());
            let mut k_prime = Array2::<f64>::zeros((n_components, n_components));

            for i in 0..n_components {
                if is_super_gaussian[i] {
                    // Super-Gaussian: tanh nonlinearity
                    for j in 0..batch_size {
                        k[[i, j]] = y[[i, j]].tanh();
                    }
                    k_prime[[i, i]] =
                        1.0 - k.slice(s![i, ..]).mapv(|x: f64| x.powi(2)).mean().unwrap();
                } else {
                    // Sub-Gaussian: cubic nonlinearity
                    for j in 0..batch_size {
                        k[[i, j]] = y[[i, j]].powi(3);
                    }
                    k_prime[[i, i]] = 3.0;
                }
            }

            // Compute gradient
            let block = &eye - &k.dot(&y.t()) / batch_size as f64 + &k_prime;
            let delta_w = &block.dot(&w) * learning_rate;

            delta_w_sum += &delta_w;
        }

        // Update unmixing matrix
        let delta_w_avg = delta_w_sum / n_batches as f64;
        w = &w + &delta_w_avg;

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check for convergence (simplified)
        if delta_w_avg.mapv(|x: f64| x.abs()).mean().unwrap() < config.convergence_threshold {
            break;
        }
    }

    // Extract the independent components
    let sources = w.dot(signals);

    Ok((sources, w))
}

/// Apply Non-negative Matrix Factorization (NMF) to separate mixed signals
///
/// NMF decomposes non-negative data into non-negative factors.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (sources matrix H, mixing matrix W)
pub fn nmf(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Check that the signals are non-negative
    if signals.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "NMF requires non-negative signals".to_string(),
        ));
    }

    // Initialize random W and H matrices
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let mut w = Array2::<f64>::zeros((n_signals, n_components));
    let mut h = Array2::<f64>::zeros((n_components, n_samples));

    for i in 0..n_signals {
        for j in 0..n_components {
            w[[i, j]] = rng.random_range(0.0..1.0);
        }
    }

    for i in 0..n_components {
        for j in 0..n_samples {
            h[[i, j]] = rng.random_range(0.0..1.0);
        }
    }

    // Normalize columns of W
    for j in 0..n_components {
        let norm = w.slice(s![.., j]).mapv(|x: f64| x.powi(2)).sum().sqrt();
        if norm > 0.0 {
            for i in 0..n_signals {
                w[[i, j]] /= norm;
            }
        }
    }
    // Perform NMF using multiplicative update rules
    for _ in 0..config.max_iterations {
        // Update H (sources)
        let w_t = w.t();
        let w_t_v = w_t.dot(signals);
        let w_t_w_h = w_t.dot(&w).dot(&h);

        for i in 0..n_components {
            for j in 0..n_samples {
                if w_t_w_h[[i, j]] > 1e-10 {
                    h[[i, j]] *= w_t_v[[i, j]] / w_t_w_h[[i, j]];
                }
            }
        }

        // Update W (mixing matrix)
        let v_h_t = signals.dot(&h.t());
        let w_h_h_t = w.dot(&h).dot(&h.t());

        for i in 0..n_signals {
            for j in 0..n_components {
                if w_h_h_t[[i, j]] > 1e-10 {
                    w[[i, j]] *= v_h_t[[i, j]] / w_h_h_t[[i, j]];
                }
            }
        }

        // Normalize columns of W
        for j in 0..n_components {
            let norm = w.slice(s![.., j]).mapv(|x: f64| x.powi(2)).sum().sqrt();
            if norm > 0.0 {
                for i in 0..n_signals {
                    w[[i, j]] /= norm;
                }

                // Scale H accordingly
                for i in 0..n_samples {
                    h[[j, i]] *= norm;
                }
            }
        }
    }

    // H contains the separated sources, W contains the mixing matrix
    Ok((h, w))
}

/// Apply Sparse Component Analysis (SCA) to separate mixed signals
///
/// SCA is useful when the source signals are known to be sparse.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of components to extract
/// * `sparsity_param` - Sparsity parameter (L1 regularization strength)
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn sparse_component_analysis(
    signals: &Array2<f64>,
    n_components: usize,
    sparsity_param: f64,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Use ICA as initial approximation
    let (initial_sources, initial_mixing) = ica(
        signals,
        Some(n_components),
        IcaMethod::FastICA,
        NonlinearityFunction::Tanh,
        config,
    )?;

    // Initialize sources and mixing matrix
    let mut sources = initial_sources.clone();
    let mut mixing = initial_mixing.clone();

    // Learning rate for gradient descent
    let mut learning_rate = config.learning_rate;
    let min_learning_rate = 0.001;
    let decay_rate = 0.95;

    // Perform sparse component analysis using alternating minimization
    for iteration in 0..config.max_iterations {
        // Update sources using gradient descent with L1 regularization
        let residual = signals - &mixing.dot(&sources);
        let gradient = mixing.t().dot(&residual);

        // Apply soft thresholding (proximal operator for L1 norm)
        for i in 0..n_components {
            for j in 0..n_samples {
                let update = sources[[i, j]] + learning_rate * gradient[[i, j]];
                let magnitude = update.abs();

                if magnitude <= sparsity_param * learning_rate {
                    sources[[i, j]] = 0.0;
                } else {
                    let sign = if update >= 0.0 { 1.0 } else { -1.0 };
                    sources[[i, j]] = sign * (magnitude - sparsity_param * learning_rate);
                }
            }
        }

        // Update mixing matrix using least squares
        for i in 0..n_signals {
            let target = signals.slice(s![i, ..]);

            // Solve least squares problem
            let sourcest_sources = sources.dot(&sources.t());
            let sourcest_target = sources.dot(&target.t());

            // Regularized least squares
            let regularized =
                &sourcest_sources + &(Array2::<f64>::eye(n_components) * config.regularization);

            match solve(&regularized.view(), &sourcest_target.view()) {
                Ok(solution) => {
                    for j in 0..n_components {
                        mixing[[i, j]] = solution[j];
                    }
                }
                Err(_) => {
                    return Err(SignalError::Compute(
                        "Failed to solve least squares in SCA".to_string(),
                    ));
                }
            }
        }

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check convergence (simplified)
        if iteration > 10 && learning_rate <= min_learning_rate {
            break;
        }
    }

    Ok((sources, mixing))
}

/// Apply Joint Blind Source Separation (JBSS) to separate mixed signals from multiple datasets
///
/// JBSS extends BSS to multiple datasets with shared sources but different mixing matrices.
///
/// # Arguments
///
/// * `datasets` - Vector of mixed signal matrices (each with rows as signals, columns as samples)
/// * `n_components` - Number of components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (vector of extracted sources, vector of mixing matrices)
pub fn joint_bss(
    datasets: &[Array2<f64>],
    n_components: usize,
    _config: &BssConfig,
) -> SignalResult<JadeMultiResult> {
    if datasets.is_empty() {
        return Err(SignalError::ValueError("No datasets provided".to_string()));
    }

    // Number of datasets
    let n_datasets = datasets.len();

    // Build joint covariance matrices
    let mut joint_cov = Array2::<f64>::zeros((0, 0));
    let mut dataset_dims = Vec::with_capacity(n_datasets);

    for dataset in datasets {
        let (n_signals, n_samples) = dataset.dim();
        dataset_dims.push(n_signals);

        // Center the dataset
        let means = dataset.mean_axis(Axis(1)).unwrap();
        let mut centered = dataset.clone();

        for i in 0..n_signals {
            for j in 0..n_samples {
                centered[[i, j]] -= means[i];
            }
        }

        // Calculate covariance matrix
        let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);

        // Extend joint covariance matrix
        if joint_cov.dim().0 == 0 {
            joint_cov = cov;
        } else {
            // Create block diagonal matrix
            let current_dim = joint_cov.dim().0;
            let new_dim = current_dim + n_signals;
            let mut new_joint_cov = Array2::<f64>::zeros((new_dim, new_dim));

            // Copy existing joint covariance
            for i in 0..current_dim {
                for j in 0..current_dim {
                    new_joint_cov[[i, j]] = joint_cov[[i, j]];
                }
            }

            // Add new covariance block
            for i in 0..n_signals {
                for j in 0..n_signals {
                    new_joint_cov[[current_dim + i, current_dim + j]] = cov[[i, j]];
                }
            }

            joint_cov = new_joint_cov;
        }
    }

    // Perform joint diagonalization on the covariance matrices
    let (_eigvals, eigvecs) = match eigh(&joint_cov.view()) {
        Ok((vals, vecs)) => (vals, vecs),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition of joint covariance".to_string(),
            ));
        }
    };

    // Extract the unmixing matrices for each dataset
    let mut unmixing_matrices = Vec::with_capacity(n_datasets);
    let mut offset = 0;

    for &dim in &dataset_dims {
        let unmixing = eigvecs.slice(s![offset..offset + dim, ..]).to_owned();
        unmixing_matrices.push(unmixing);
        offset += dim;
    }

    // Apply the unmixing matrices to each dataset
    let mut extracted_sources = Vec::with_capacity(n_datasets);
    let mut mixing_matrices = Vec::with_capacity(n_datasets);

    for (i, dataset) in datasets.iter().enumerate() {
        let unmixing = &unmixing_matrices[i];
        let sources = unmixing.t().slice(s![0..n_components, ..]).dot(dataset);
        extracted_sources.push(sources);

        // Calculate mixing matrix (pseudoinverse of unmixing)
        let (u, s, vt) = match svd(&unmixing.view(), false) {
            Ok((u, s, vt)) => (u, s, vt),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to compute SVD of unmixing matrix".to_string(),
                ));
            }
        };

        let mut s_inv = Array2::<f64>::zeros((vt.dim().0, u.dim().0));
        for i in 0..s.len() {
            if s[i] > 1e-10 {
                s_inv[[i, i]] = 1.0 / s[i];
            }
        }

        let mixing = vt.t().dot(&s_inv).dot(&u.t());
        mixing_matrices.push(mixing.slice(s![.., 0..n_components]).to_owned());
    }

    Ok((extracted_sources, mixing_matrices))
}

/// Apply Joint Approximate Diagonalization (JAD) for blind source separation
///
/// JAD simultaneously diagonalizes multiple matrices for BSS.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `n_components` - Number of components to extract
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn joint_diagonalization(
    signals: &Array2<f64>,
    n_components: usize,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Create a set of time-delayed covariance matrices
    let mut cov_matrices = Vec::new();

    // Add the standard covariance matrix
    let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);
    cov_matrices.push(cov);

    // Add time-delayed covariance matrices
    let max_lag = 10.min(n_samples / 4);

    for lag in 1..=max_lag {
        let mut cov_lagged = Array2::<f64>::zeros((n_signals, n_signals));

        for i in 0..n_signals {
            for j in 0..n_signals {
                let mut sum = 0.0;

                for t in 0..n_samples - lag {
                    sum += centered[[i, t]] * centered[[j, t + lag]];
                }

                cov_lagged[[i, j]] = sum / (n_samples as f64 - lag as f64);
            }
        }

        cov_matrices.push(cov_lagged);
    }

    // Perform approximate joint diagonalization
    let mut v = Array2::<f64>::eye(n_signals);

    for _ in 0..config.max_iterations {
        let mut diagonalized = true;

        for i in 0..n_signals - 1 {
            for j in i + 1..n_signals {
                // Calculate rotation angle to jointly diagonalize matrices
                let mut num = 0.0;
                let mut denom = 0.0;

                for cov_matrix in &cov_matrices {
                    let cij = cov_matrix[[i, j]];
                    let cji = cov_matrix[[j, i]];
                    let cii = cov_matrix[[i, i]];
                    let cjj = cov_matrix[[j, j]];

                    num += 2.0 * (cij + cji);
                    denom += cii - cjj;
                }

                // Avoid division by zero
                if denom.abs() < 1e-10 {
                    continue;
                }

                let theta = 0.5 * (num / denom).atan();

                // Check if rotation is needed
                if theta.abs() > config.convergence_threshold {
                    diagonalized = false;

                    // Apply Givens rotation
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    // Create Givens rotation matrix
                    let mut g = Array2::<f64>::eye(n_signals);
                    g[[i, i]] = cos_t;
                    g[[i, j]] = -sin_t;
                    g[[j, i]] = sin_t;
                    g[[j, j]] = cos_t;

                    // Update V
                    v = v.dot(&g);

                    // Update covariance matrices
                    for matrix in &mut cov_matrices {
                        *matrix = g.t().dot(matrix).dot(&g);
                    }
                }
            }
        }

        if diagonalized {
            break;
        }
    }

    // Extract unmixing matrix (take the first n_components rows)
    let w = v.slice(s![0..n_components, ..]).to_owned();

    // Extract sources
    let sources = w.dot(&centered);

    // Calculate mixing matrix (pseudoinverse of w)
    let (u, s, vt) = match svd(&w.view(), false) {
        Ok((u, s, vt)) => (u, s, vt),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute SVD of unmixing matrix".to_string(),
            ));
        }
    };

    let mut s_inv = Array2::<f64>::zeros((vt.dim().0, u.dim().0));
    for i in 0..s.len().min(s_inv.dim().0).min(s_inv.dim().1) {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    let mixing = vt.t().dot(&s_inv).dot(&u.t());

    Ok((sources, mixing))
}

/// Sort decomposition components by explained variance
///
/// # Arguments
///
/// * `sources` - Source signals from decomposition
/// * `mixing` - Mixing matrix from decomposition
///
/// # Returns
///
/// * Tuple containing (sorted sources, sorted mixing matrix)
pub fn sort_components(
    sources: &Array2<f64>,
    mixing: &Array2<f64>,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_components, n_samples) = sources.dim();

    // Calculate variance of each component
    let mut variances = Vec::with_capacity(n_components);

    for i in 0..n_components {
        let component = sources.slice(s![i, ..]);
        let mean = component.mean().unwrap();
        let var = component.mapv(|x: f64| (x - mean).powi(2)).sum() / (n_samples as f64 - 1.0);
        variances.push((i, var));
    }

    // Sort components by variance
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Reorder components
    let mut sorted_sources = Array2::<f64>::zeros(sources.dim());
    let mut sorted_mixing = Array2::<f64>::zeros(mixing.dim());

    for (new_idx, (old_idx, _)) in variances.into_iter().enumerate() {
        sorted_sources
            .slice_mut(s![new_idx, ..])
            .assign(&sources.slice(s![old_idx, ..]));
        sorted_mixing
            .slice_mut(s![.., new_idx])
            .assign(&mixing.slice(s![.., old_idx]));
    }

    Ok((sorted_sources, sorted_mixing))
}

/// Calculate correlation matrix between signals
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
///
/// # Returns
///
/// * Correlation matrix
pub fn calculate_correlation_matrix(signals: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (n_signals, n_samples) = signals.dim();

    // Center and normalize signals
    let mut normalized = Array2::<f64>::zeros(signals.dim());

    for i in 0..n_signals {
        let signal = signals.slice(s![i, ..]);
        let mean = signal.mean().unwrap();
        let std_dev = (signal.mapv(|x: f64| (x - mean).powi(2)).sum() / n_samples as f64).sqrt();

        if std_dev > 1e-10 {
            for j in 0..n_samples {
                normalized[[i, j]] = (signals[[i, j]] - mean) / std_dev;
            }
        }
    }

    // Calculate correlation matrix
    let corr = normalized.dot(&normalized.t()) / (n_samples as f64 - 1.0);

    Ok(corr)
}

/// Calculate mutual information between signals
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_bins` - Number of bins for histogram estimation
///
/// # Returns
///
/// * Matrix of pairwise mutual information values
pub fn calculate_mutual_information(
    signals: &Array2<f64>,
    n_bins: usize,
) -> SignalResult<Array2<f64>> {
    let (n_signals, n_samples) = signals.dim();
    let mut mi_matrix = Array2::<f64>::zeros((n_signals, n_signals));

    // For each pair of signals
    for i in 0..n_signals {
        for j in 0..n_signals {
            if i == j {
                continue;
            }

            let x = signals.slice(s![i, ..]);
            let y = signals.slice(s![j, ..]);

            // Find min and max for each signal to define histogram bins
            let x_min = x.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
            let x_max = x.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
            let y_min = y.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
            let y_max = y.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

            let x_bin_width = (x_max - x_min) / n_bins as f64;
            let y_bin_width = (y_max - y_min) / n_bins as f64;

            // Create joint histogram
            let mut joint_hist = Array2::<f64>::zeros((n_bins, n_bins));
            let mut x_hist = Array1::<f64>::zeros(n_bins);
            let mut y_hist = Array1::<f64>::zeros(n_bins);

            // Fill histograms
            for s in 0..n_samples {
                let x_bin = ((x[s] - x_min) / x_bin_width).floor() as usize;
                let y_bin = ((y[s] - y_min) / y_bin_width).floor() as usize;

                // Handle edge cases
                let x_idx = x_bin.min(n_bins - 1);
                let y_idx = y_bin.min(n_bins - 1);

                joint_hist[[x_idx, y_idx]] += 1.0;
                x_hist[x_idx] += 1.0;
                y_hist[y_idx] += 1.0;
            }

            // Normalize histograms
            joint_hist /= n_samples as f64;
            x_hist /= n_samples as f64;
            y_hist /= n_samples as f64;

            // Calculate mutual information
            let mut mi = 0.0;

            for xi in 0..n_bins {
                for yi in 0..n_bins {
                    if joint_hist[[xi, yi]] > 1e-10 && x_hist[xi] > 1e-10 && y_hist[yi] > 1e-10 {
                        mi += joint_hist[[xi, yi]]
                            * (joint_hist[[xi, yi]] / (x_hist[xi] * y_hist[yi])).ln();
                    }
                }
            }

            mi_matrix[[i, j]] = mi;
        }
    }

    Ok(mi_matrix)
}

/// Estimate number of sources using eigenvalue analysis
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns are samples)
/// * `threshold` - Eigenvalue ratio threshold
///
/// # Returns
///
/// * Estimated number of sources
pub fn estimate_source_count(signals: &Array2<f64>, threshold: f64) -> SignalResult<usize> {
    let (n_signals, n_samples) = signals.dim();

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Compute covariance matrix
    let cov = centered.dot(&centered.t()) / (n_samples as f64 - 1.0);

    // Perform eigendecomposition
    let eigvals = match eigh(&cov.view()) {
        Ok((vals, _)) => vals,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute eigendecomposition".to_string(),
            ));
        }
    };

    // Calculate eigenvalue ratios
    let mut ratios = Vec::with_capacity(n_signals - 1);

    for i in 0..n_signals - 1 {
        if eigvals[i + 1] > 1e-10 {
            ratios.push(eigvals[i] / eigvals[i + 1]);
        }
    }

    // Find the largest gap
    let mut max_ratio_idx = 0;
    let mut max_ratio = 0.0;

    for (i, &ratio) in ratios.iter().enumerate() {
        if ratio > max_ratio {
            max_ratio = ratio;
            max_ratio_idx = i;
        }
    }

    // Check if max ratio exceeds threshold
    if max_ratio > threshold {
        Ok(max_ratio_idx + 1)
    } else {
        // If no clear gap, return a conservative estimate
        Ok(n_signals / 2)
    }
}

/// Apply Kernel ICA for nonlinear blind source separation
///
/// Kernel ICA uses kernel methods to handle nonlinearities in the data.
///
/// # Arguments
///
/// * `signals` - Matrix of mixed signals (rows are signals, columns as samples)
/// * `n_components` - Number of independent components to extract
/// * `kernel_width` - Width parameter for Gaussian kernel
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Tuple containing (extracted sources, mixing matrix)
pub fn kernel_ica(
    signals: &Array2<f64>,
    n_components: usize,
    kernel_width: f64,
    config: &BssConfig,
) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (n_signals, n_samples) = signals.dim();

    // Center the signals
    let means = signals.mean_axis(Axis(1)).unwrap();
    let mut centered = signals.clone();

    for i in 0..n_signals {
        for j in 0..n_samples {
            centered[[i, j]] -= means[i];
        }
    }

    // Use PCA as prewhitening
    let (pca_sources, pca_mixing) = pca(&centered, config)?;

    // Initialize random unmixing matrix
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut w = Array2::<f64>::zeros((n_components, n_components));

    for i in 0..n_components {
        for j in 0..n_components {
            w[[i, j]] = normal.sample(&mut rng) * 0.1;
        }
    }

    // Start with identity matrix
    for i in 0..n_components {
        w[[i, i]] = 1.0;
    }

    // Compute whitened data
    let whitened = pca_sources.slice(s![0..n_components, ..]).to_owned();

    // Compute Gram matrices for each component
    let compute_gram_matrix = |component: &Array1<f64>| -> Array2<f64> {
        let mut gram = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = component[i] - component[j];
                gram[[i, j]] = (-diff * diff / (2.0 * kernel_width * kernel_width)).exp();
            }
        }

        // Center Gram matrix
        let row_means = gram.mean_axis(Axis(0)).unwrap();
        let col_means = gram.mean_axis(Axis(1)).unwrap();
        let total_mean = gram.mean().unwrap();

        for i in 0..n_samples {
            for j in 0..n_samples {
                gram[[i, j]] -= row_means[j] + col_means[i] - total_mean;
            }
        }

        gram
    };

    // Optimize using gradient descent
    let mut learning_rate = config.learning_rate;
    let min_learning_rate = 0.001;
    let decay_rate = 0.95;

    for iteration in 0..config.max_iterations {
        // Apply current unmixing matrix
        let mut y = Array2::<f64>::zeros((n_components, n_samples));
        for i in 0..n_components {
            for j in 0..n_samples {
                for k in 0..n_components {
                    y[[i, j]] += w[[i, k]] * whitened[[k, j]];
                }
            }
        }

        // Compute Gram matrices
        let mut gram_matrices = Vec::with_capacity(n_components);
        for i in 0..n_components {
            gram_matrices.push(compute_gram_matrix(&y.slice(s![i, ..]).to_owned()));
        }

        // Compute HSIC (Hilbert-Schmidt Independence Criterion)
        let mut hsic = 0.0;
        for i in 0..n_components {
            for j in 0..n_components {
                if i != j {
                    let gram_i = &gram_matrices[i];
                    let gram_j = &gram_matrices[j];

                    // Calculate Frobenius inner product tr(gram_i * gram_j)
                    for k in 0..n_samples {
                        for l in 0..n_samples {
                            hsic += gram_i[[k, l]] * gram_j[[k, l]];
                        }
                    }
                }
            }
        }
        hsic /= (n_samples * n_samples) as f64;

        // Compute gradient
        let mut gradient = Array2::<f64>::zeros((n_components, n_components));

        for c in 0..n_components {
            for d in 0..n_components {
                let mut grad_cd = 0.0;

                for i in 0..n_components {
                    if i != c {
                        let gram_i = &gram_matrices[i];
                        let gram_c = &gram_matrices[c];

                        for s in 0..n_samples {
                            for t in 0..n_samples {
                                // Compute partial derivative
                                let mut kernel_deriv_sum = 0.0;

                                for u in 0..n_samples {
                                    let diff_su = y[[c, s]] - y[[c, u]];
                                    let kernel_su = gram_c[[s, u]];
                                    kernel_deriv_sum +=
                                        kernel_su * (whitened[[d, s]] - whitened[[d, u]]) * diff_su;
                                }

                                kernel_deriv_sum /= -kernel_width * kernel_width;
                                grad_cd += kernel_deriv_sum * gram_i[[s, t]];
                            }
                        }
                    }
                }

                gradient[[c, d]] = grad_cd / (n_samples * n_samples) as f64;
            }
        }

        // Update unmixing matrix
        w = &w - &(&gradient * learning_rate);

        // Decorrelate using symmetric decorrelation
        let ww_t = w.dot(&w.t());
        let (eigvals, eigvecs) = match eigh(&ww_t.view()) {
            Ok((vals, vecs)) => (vals, vecs),
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to compute eigendecomposition in KernelICA".to_string(),
                ));
            }
        };

        let mut d_inv_sqrt = Array2::<f64>::zeros((n_components, n_components));
        for i in 0..n_components {
            if eigvals[i] > 1e-10 {
                d_inv_sqrt[[i, i]] = 1.0 / eigvals[i].sqrt();
            }
        }

        w = eigvecs.dot(&d_inv_sqrt).dot(&eigvecs.t()).dot(&w);

        // Reduce learning rate
        learning_rate = (learning_rate * decay_rate).max(min_learning_rate);

        // Check convergence criterion
        if iteration > 10 && hsic < config.convergence_threshold {
            break;
        }
    }

    // Apply final unmixing matrix to whitened data
    let sources = w.dot(&whitened);

    // Compute the total unmixing matrix
    let unmixing = w.dot(&pca_mixing.slice(s![.., 0..n_components]).t());

    // Calculate mixing matrix (pseudoinverse of unmixing)
    let (u, s, vt) = match svd(&unmixing.view(), false) {
        Ok((u, s, vt)) => (u, s, vt),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute SVD of unmixing matrix".to_string(),
            ));
        }
    };

    let mut s_inv = Array2::<f64>::zeros((vt.dim().0, u.dim().0));
    for i in 0..s.len().min(s_inv.dim().0).min(s_inv.dim().1) {
        if s[i] > 1e-10 {
            s_inv[[i, i]] = 1.0 / s[i];
        }
    }

    let mixing = vt.t().dot(&s_inv).dot(&u.t());

    Ok((sources, mixing))
}

/// Apply Multivariate Empirical Mode Decomposition (MEMD) to separate components
///
/// MEMD decomposes signals into a set of Intrinsic Mode Functions (IMFs).
///
/// # Arguments
///
/// * `signals` - Matrix of signals (rows are signals, columns are samples)
/// * `n_directions` - Number of projection directions for MEMD
/// * `max_imfs` - Maximum number of IMFs to extract (or None for automatic)
/// * `config` - BSS configuration
///
/// # Returns
///
/// * Array of decomposed IMFs for each signal
pub fn multivariate_emd(
    signals: &Array2<f64>,
    n_directions: usize,
    max_imfs: Option<usize>,
    config: &BssConfig,
) -> SignalResult<Vec<Array2<f64>>> {
    let (n_signals, n_samples) = signals.dim();

    // Generate direction vectors on a hypersphere
    let mut directions = Vec::with_capacity(n_directions);
    let mut rng = if let Some(seed) = config.random_seed {
        rand::rngs::StdRng::from_seed([seed as u8; 32])
    } else {
        {
            // In rand 0.9, from_rng doesn't return Result but directly returns the PRNG
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        }
    };

    for _ in 0..n_directions {
        let mut v = Vec::with_capacity(n_signals);

        // Generate random normal vector
        for _ in 0..n_signals {
            v.push(rng.random_range(-1.0..1.0));
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }

        directions.push(v);
    }

    // Initialize IMF arrays
    let mut all_imfs = Vec::new();

    // Determine maximum number of IMFs
    let max_possible_imfs = (n_samples as f64).log2().floor() as usize;
    let max_imf_count = max_imfs.unwrap_or(max_possible_imfs);

    // Initialize with input signals
    let mut residuals = signals.clone();

    // Extract IMFs
    for _imf_idx in 0..max_imf_count {
        // Current IMF
        let mut imf = residuals.clone();
        let mut prev_imf = Array2::<f64>::zeros(imf.dim());

        // Apply sifting process
        for _iteration in 0..config.max_iterations {
            // Save previous IMF
            prev_imf.assign(&imf);

            // For each direction
            let mut envelopes = Vec::with_capacity(n_directions);

            for dir in &directions {
                // Project signals onto direction
                let mut projection = Array1::<f64>::zeros(n_samples);

                for j in 0..n_samples {
                    for i in 0..n_signals {
                        projection[j] += imf[[i, j]] * dir[i];
                    }
                }

                // Find extrema (maxima and minima)
                let mut extrema_indices = Vec::new();

                for j in 1..n_samples - 1 {
                    if (projection[j] > projection[j - 1] && projection[j] > projection[j + 1])
                        || (projection[j] < projection[j - 1] && projection[j] < projection[j + 1])
                    {
                        extrema_indices.push(j);
                    }
                }

                // Add first and last points
                extrema_indices.insert(0, 0);
                extrema_indices.push(n_samples - 1);

                // Compute envelope for this direction
                let mut envelope = Array2::<f64>::zeros((n_signals, n_samples));

                // Simple linear interpolation for envelope
                for i in 0..extrema_indices.len() - 1 {
                    let idx1 = extrema_indices[i];
                    let idx2 = extrema_indices[i + 1];

                    if idx2 > idx1 {
                        let step = 1.0 / (idx2 - idx1) as f64;

                        for j in idx1..=idx2 {
                            let t = (j - idx1) as f64 * step;

                            for k in 0..n_signals {
                                envelope[[k, j]] = (1.0 - t) * imf[[k, idx1]] + t * imf[[k, idx2]];
                            }
                        }
                    }
                }

                envelopes.push(envelope);
            }

            // Compute mean of envelopes
            let mut mean_envelope = Array2::<f64>::zeros((n_signals, n_samples));

            for envelope in &envelopes {
                mean_envelope = &mean_envelope + envelope;
            }

            mean_envelope /= n_directions as f64;

            // Subtract envelope mean from current IMF
            imf = &imf - &mean_envelope;

            // Check if IMF criteria are met
            let diff = (&imf - &prev_imf).mapv(|x: f64| x.powi(2)).sum().sqrt();
            let norm = imf.mapv(|x: f64| x.powi(2)).sum().sqrt();

            if diff / norm < config.convergence_threshold {
                break;
            }
        }

        // Store extracted IMF
        all_imfs.push(imf.clone());

        // Update residuals
        residuals = &residuals - &imf;

        // Check if residual is monotonic
        let mut is_monotonic = true;
        for i in 0..n_signals {
            let mut increasing_count = 0;
            let mut decreasing_count = 0;

            for j in 1..n_samples {
                if residuals[[i, j]] > residuals[[i, j - 1]] {
                    increasing_count += 1;
                } else if residuals[[i, j]] < residuals[[i, j - 1]] {
                    decreasing_count += 1;
                }
            }

            if increasing_count > 0 && decreasing_count > 0 {
                is_monotonic = false;
                break;
            }
        }

        if is_monotonic {
            break;
        }
    }

    // Add final residual as last IMF
    all_imfs.push(residuals);

    Ok(all_imfs)
}
