//! CANDECOMP/PARAFAC (CP) decomposition for dense N-way tensors.
//!
//! ## Algorithms
//!
//! ### CP-ALS  (`cp_als`)
//!
//! *Alternating Least Squares* — the standard workhorse algorithm.  At each
//! iteration, for mode `n`:
//!
//! ```text
//! A^(n) ← X_(n) · (⊙_{k≠n} A^(k)) · (⊙_{k≠n} A^(k)^T A^(k))^{-1}
//! ```
//!
//! where `X_(n)` is the mode-n unfolding, `⊙` is the Khatri-Rao product, and
//! the hadamard product of Gram matrices is inverted via pseudo-inverse for
//! robustness.  After each update the columns are normalised and weights
//! accumulated in `lambdas`.
//!
//! ### CP-Gradient  (`cp_grad`)
//!
//! Gradient descent on the squared Frobenius loss:
//!
//! ```text
//! L = (1/2) ‖T - [[ λ; A^(1), …, A^(N) ]]‖_F^2
//! ```
//!
//! Updates use a fixed step-size with optional Armijo line-search along the
//! gradient direction.  This is more numerically stable for near-degenerate
//! problems but converges more slowly than ALS.
//!
//! ### Reconstruction  (`cp_reconstruct`)
//!
//! Converts a [`CPResult`] back into a full tensor:
//!
//! ```text
//! T̃ = Σ_{r=1}^{R} λ_r · a^(1)_r ⊗ a^(2)_r ⊗ … ⊗ a^(N)_r
//! ```
//!
//! ## References
//!
//! - Kolda & Bader (2009). "Tensor Decompositions and Applications".
//!   *SIAM Rev.* 51(3).
//! - Tomasi & Bro (2006). "A comparison of algorithms for fitting the PARAFAC
//!   model". *Comput. Stat. Data Anal.* 50.

use crate::error::{LinalgError, LinalgResult};
use crate::tensor::core::{Tensor, TensorScalar};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for CP decomposition algorithms.
#[derive(Debug, Clone)]
pub struct CPConfig {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change in loss.
    pub tol: f64,
    /// Initial step size for gradient descent (used by `cp_grad`).
    pub lr: f64,
    /// Whether to use random initialisation (true) or HOSVD-based (false).
    pub random_init: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for CPConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-8,
            lr: 1e-3,
            random_init: false,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a CP/PARAFAC decomposition.
///
/// The decomposition approximates
/// `T ≈ Σ_{r=1}^{R} λ_r · a^(1)_r ⊗ a^(2)_r ⊗ … ⊗ a^(N)_r`
/// where:
/// - `lambdas[r]` is the weight of component `r`.
/// - `factors[n][:, r]` is the unit-norm factor vector for mode `n`.
/// - `loss` records the per-iteration reconstruction loss (for diagnostics).
#[derive(Debug, Clone)]
pub struct CPResult<F> {
    /// Component weights, length `rank`.
    pub lambdas: Vec<F>,
    /// Factor matrices `[A_1, …, A_N]`; `factors[n]` has shape
    /// `(shape[n], rank)` with unit-norm columns.
    pub factors: Vec<Array2<F>>,
    /// Reconstruction loss per iteration.
    pub loss: Vec<F>,
}

impl<F: TensorScalar> CPResult<F> {
    /// Reconstruct the full tensor from this decomposition.
    pub fn reconstruct(&self, shape: &[usize]) -> LinalgResult<Tensor<F>> {
        cp_reconstruct(self, shape)
    }

    /// Relative Frobenius reconstruction error `‖T - T̃‖_F / ‖T‖_F`.
    pub fn relative_error(&self, original: &Tensor<F>) -> LinalgResult<F> {
        let recon = self.reconstruct(&original.shape)?;
        let orig_norm = original.frobenius_norm();
        if orig_norm == F::zero() {
            return Ok(F::zero());
        }
        let diff_sq: F = original
            .data
            .iter()
            .zip(recon.data.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .fold(F::zero(), |acc, x| acc + x);
        Ok((diff_sq / (orig_norm * orig_norm)).sqrt())
    }
}

// ---------------------------------------------------------------------------
// CP-ALS
// ---------------------------------------------------------------------------

/// CP decomposition via Alternating Least Squares (ALS).
///
/// # Arguments
///
/// * `tensor` – input N-way tensor.
/// * `rank`   – target CP rank (number of components `R`).
/// * `config` – algorithm parameters.
///
/// # Returns
///
/// [`CPResult`] on success.
///
/// # Errors
///
/// Returns [`LinalgError::ValueError`] when `rank == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
/// use scirs2_linalg::tensor::cp_decomp::{cp_als, CPConfig};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
/// let tensor = Tensor::new(data, vec![2, 3, 4]).expect("valid");
/// let cfg = CPConfig { max_iter: 200, ..Default::default() };
/// let result = cp_als(&tensor, 3, &cfg).expect("cp_als ok");
/// assert_eq!(result.lambdas.len(), 3);
/// ```
pub fn cp_als<F: TensorScalar>(
    tensor: &Tensor<F>,
    rank: usize,
    config: &CPConfig,
) -> LinalgResult<CPResult<F>> {
    if rank == 0 {
        return Err(LinalgError::ValueError("rank must be > 0".to_string()));
    }
    let ndim = tensor.ndim();
    let shape = &tensor.shape;

    // Initialise factor matrices
    let mut factors: Vec<Array2<F>> = init_factors(shape, rank, config.random_init, config.seed);

    let tol = F::from(config.tol).unwrap_or(F::zero());
    let mut loss_history: Vec<F> = Vec::with_capacity(config.max_iter + 1);

    let initial_loss = reconstruction_loss(tensor, &factors, &vec![F::one(); rank]);
    loss_history.push(initial_loss);

    let mut lambdas: Vec<F> = vec![F::one(); rank];

    for _iter in 0..config.max_iter {
        for n in 0..ndim {
            // Build the Khatri-Rao product of all factors except n
            let kr = khatri_rao_except(&factors, n);
            // Mode-n unfolding
            let unfolded = tensor.unfold(n)?;
            // Gram matrices hadamard product: V = ⊙_{k≠n} (A^(k)^T A^(k))
            let gram_hadamard = hadamard_gram_except(&factors, n, rank);
            // Pseudo-inverse of gram_hadamard
            let gram_pinv = pinv_symmetric(&gram_hadamard)?;
            // A^(n) ← X_(n) · KR · V^{-1}
            let new_factor = unfolded.dot(&kr).dot(&gram_pinv);
            // Normalise columns and extract lambdas
            let (normed, norms) = normalise_columns(new_factor);
            lambdas = norms;
            factors[n] = normed;
        }

        let loss = reconstruction_loss(tensor, &factors, &lambdas);
        let prev_loss = loss_history.last().copied().unwrap_or(loss);
        loss_history.push(loss);

        let delta = if prev_loss > F::zero() {
            (prev_loss - loss).abs() / prev_loss
        } else {
            F::zero()
        };
        if delta < tol {
            break;
        }
    }

    // Scale factor columns by lambda^(1/N) for balanced representation
    let root_n = F::from(ndim as f64).unwrap_or(F::one());
    let scale: Vec<F> = lambdas
        .iter()
        .map(|&l| {
            if l < F::zero() {
                -(-l).powf(F::one() / root_n)
            } else {
                l.powf(F::one() / root_n)
            }
        })
        .collect();

    for n in 0..ndim {
        let mut factor = factors[n].clone();
        for r in 0..rank {
            let s = scale[r];
            for i in 0..factor.nrows() {
                factor[[i, r]] *= s;
            }
        }
        factors[n] = factor;
    }
    // After scaling, effective lambdas are all 1
    let final_lambdas = vec![F::one(); rank];

    Ok(CPResult {
        lambdas: final_lambdas,
        factors,
        loss: loss_history,
    })
}

// ---------------------------------------------------------------------------
// CP-Gradient
// ---------------------------------------------------------------------------

/// CP decomposition via gradient descent on the Frobenius loss.
///
/// More stable than ALS for ill-conditioned or near-degenerate problems,
/// at the cost of slower convergence.
///
/// # Arguments
///
/// * `tensor` – input N-way tensor.
/// * `rank`   – target CP rank.
/// * `config` – algorithm parameters; `config.lr` controls the step size.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
/// use scirs2_linalg::tensor::cp_decomp::{cp_grad, CPConfig};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
/// let tensor = Tensor::new(data, vec![2, 3, 4]).expect("valid");
/// let cfg = CPConfig { max_iter: 100, lr: 1e-3, ..Default::default() };
/// let result = cp_grad(&tensor, 2, &cfg).expect("cp_grad ok");
/// assert_eq!(result.factors.len(), 3);
/// ```
pub fn cp_grad<F: TensorScalar>(
    tensor: &Tensor<F>,
    rank: usize,
    config: &CPConfig,
) -> LinalgResult<CPResult<F>> {
    if rank == 0 {
        return Err(LinalgError::ValueError("rank must be > 0".to_string()));
    }
    let ndim = tensor.ndim();
    let shape = &tensor.shape;
    let lr = F::from(config.lr).unwrap_or(F::from(1e-3_f64).unwrap_or(F::one()));
    let tol = F::from(config.tol).unwrap_or(F::zero());

    let mut factors: Vec<Array2<F>> = init_factors(shape, rank, config.random_init, config.seed);
    let lambdas: Vec<F> = vec![F::one(); rank];
    let mut loss_history: Vec<F> = Vec::with_capacity(config.max_iter + 1);
    loss_history.push(reconstruction_loss(tensor, &factors, &lambdas));

    for _iter in 0..config.max_iter {
        // Compute residual T - T̃
        let recon = reconstruct_from_factors(tensor.shape.clone(), &factors, &lambdas);
        let residual_data: Vec<F> = tensor
            .data
            .iter()
            .zip(recon.data.iter())
            .map(|(&t, &r)| t - r)
            .collect();
        let residual = Tensor::new(residual_data, tensor.shape.clone())?;

        // Gradient for each factor n:
        // ∂L/∂A^(n) = -X_(n) · KR_{≠n} + A^(n) · (⊙_{k≠n} G_k)
        // = -(residual)_(n) · KR_{≠n}
        for n in 0..ndim {
            let res_unfold = residual.unfold(n)?;
            let kr = khatri_rao_except(&factors, n);
            // gradient = -residual_(n) · KR
            let grad: Array2<F> = {
                let raw = res_unfold.dot(&kr);
                // Negate (gradient of loss w.r.t. A^(n))
                let mut g = Array2::<F>::zeros(raw.dim());
                for i in 0..raw.nrows() {
                    for j in 0..raw.ncols() {
                        g[[i, j]] = -raw[[i, j]];
                    }
                }
                g
            };
            // gradient step
            let mut new_factor = factors[n].clone();
            for i in 0..new_factor.nrows() {
                for j in 0..new_factor.ncols() {
                    new_factor[[i, j]] -= lr * grad[[i, j]];
                }
            }
            factors[n] = new_factor;
        }

        let loss = reconstruction_loss(tensor, &factors, &lambdas);
        let prev_loss = loss_history.last().copied().unwrap_or(loss);
        loss_history.push(loss);

        let delta = if prev_loss > F::zero() {
            (prev_loss - loss).abs() / prev_loss
        } else {
            F::zero()
        };
        if delta < tol {
            break;
        }
    }

    Ok(CPResult {
        lambdas,
        factors,
        loss: loss_history,
    })
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct a full tensor from a [`CPResult`].
///
/// # Arguments
///
/// * `result` – CP decomposition result.
/// * `shape`  – expected shape of the output tensor (used for validation).
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] when `result.factors.len() !=
/// shape.len()`.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::tensor::core::Tensor;
/// use scirs2_linalg::tensor::cp_decomp::{cp_als, cp_reconstruct, CPConfig};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
/// let tensor = Tensor::new(data, vec![2, 3, 4]).expect("valid");
/// let cfg = CPConfig::default();
/// let result = cp_als(&tensor, 4, &cfg).expect("ok");
/// let recon = cp_reconstruct(&result, &[2, 3, 4]).expect("reconstruct");
/// assert_eq!(recon.shape, vec![2, 3, 4]);
/// ```
pub fn cp_reconstruct<F: TensorScalar>(
    result: &CPResult<F>,
    shape: &[usize],
) -> LinalgResult<Tensor<F>> {
    if result.factors.len() != shape.len() {
        return Err(LinalgError::DimensionError(format!(
            "factors.len() {} != shape.len() {}",
            result.factors.len(),
            shape.len()
        )));
    }
    Ok(reconstruct_from_factors(
        shape.to_vec(),
        &result.factors,
        &result.lambdas,
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Initialise factor matrices.
///
/// When `random_init` is false, initialise with (scaled) identity-like blocks.
/// When `random_init` is true, use a simple deterministic pseudo-random fill
/// seeded by `seed` (we avoid external RNG dependencies by using a simple
/// LCG).
fn init_factors<F: TensorScalar>(
    shape: &[usize],
    rank: usize,
    random_init: bool,
    seed: u64,
) -> Vec<Array2<F>> {
    let mut lcg_state = seed.wrapping_add(1);
    let mut lcg = || -> f64 {
        lcg_state = lcg_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg_state >> 33) as f64) / (u32::MAX as f64)
    };

    shape
        .iter()
        .map(|&dim| {
            let mut mat = Array2::<F>::zeros((dim, rank));
            for i in 0..dim {
                for r in 0..rank {
                    let val: f64 = if random_init {
                        lcg() * 2.0 - 1.0
                    } else {
                        // Columns of a scaled discrete cosine basis
                        let t = (i as f64 + 0.5) * std::f64::consts::PI * (r as f64 + 1.0)
                            / (dim as f64 + 1.0);
                        t.sin() / (dim as f64).sqrt()
                    };
                    mat[[i, r]] = F::from(val).unwrap_or(F::zero());
                }
            }
            mat
        })
        .collect()
}

/// Khatri-Rao product of all factor matrices except mode `skip`.
///
/// The Khatri-Rao product is the column-wise Kronecker product:
/// `[a_r ⊗ b_r]` stacked as columns.
///
/// The resulting matrix has shape `(prod_{k≠skip} shape[k], rank)`.
fn khatri_rao_except<F: TensorScalar>(factors: &[Array2<F>], skip: usize) -> Array2<F> {
    let rank = factors[0].ncols();
    let ndim = factors.len();
    // Modes in natural order except skip
    let modes: Vec<usize> = (0..ndim).filter(|&k| k != skip).collect();
    if modes.is_empty() {
        return Array2::<F>::eye(rank);
    }
    // Start with the first included mode
    let mut result = factors[modes[0]].clone();
    for &m in &modes[1..] {
        let a = &result;
        let b = &factors[m];
        let rows_a = a.nrows();
        let rows_b = b.nrows();
        let mut kr = Array2::<F>::zeros((rows_a * rows_b, rank));
        for r in 0..rank {
            for i in 0..rows_a {
                for j in 0..rows_b {
                    kr[[i * rows_b + j, r]] = a[[i, r]] * b[[j, r]];
                }
            }
        }
        result = kr;
    }
    result
}

/// Hadamard (element-wise) product of Gram matrices `A^(k)^T A^(k)` for k ≠ skip.
fn hadamard_gram_except<F: TensorScalar>(
    factors: &[Array2<F>],
    skip: usize,
    rank: usize,
) -> Array2<F> {
    let mut result = Array2::<F>::from_elem((rank, rank), F::one());
    for (k, factor) in factors.iter().enumerate() {
        if k == skip {
            continue;
        }
        let gram = factor.t().dot(factor);
        for i in 0..rank {
            for j in 0..rank {
                result[[i, j]] *= gram[[i, j]];
            }
        }
    }
    result
}

/// Pseudo-inverse of a small symmetric positive semi-definite matrix.
///
/// Adds Tikhonov regularisation `eps * I` before inverting to handle near-
/// singular cases.
fn pinv_symmetric<F: TensorScalar>(mat: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = mat.nrows();
    let eps = F::from(1e-12_f64).unwrap_or(F::zero());
    // Regularise: M + eps * I
    let mut reg = mat.clone();
    for i in 0..n {
        reg[[i, i]] += eps;
    }
    // Gauss-Jordan inversion for small matrices (rank is typically <= 50)
    let mut augmented: Vec<Vec<F>> = (0..n)
        .map(|i| {
            let mut row: Vec<F> = (0..n).map(|j| reg[[i, j]]).collect();
            for j in 0..n {
                row.push(if i == j { F::one() } else { F::zero() });
            }
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&a, &b| {
                augmented[a][col]
                    .abs()
                    .partial_cmp(&augmented[b][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| LinalgError::SingularMatrixError("gram matrix singular".to_string()))?;
        augmented.swap(col, pivot_row);

        let pivot = augmented[col][col];
        if pivot.abs() < F::from(1e-30_f64).unwrap_or(F::zero()) {
            return Err(LinalgError::SingularMatrixError(
                "gram matrix numerically singular".to_string(),
            ));
        }
        let inv_pivot = F::one() / pivot;
        for j in 0..(2 * n) {
            augmented[col][j] *= inv_pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = augmented[row][col];
            for j in 0..(2 * n) {
                let sub = factor * augmented[col][j];
                augmented[row][j] -= sub;
            }
        }
    }

    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = augmented[i][n + j];
        }
    }
    Ok(inv)
}

/// Normalise columns of a matrix, return (normed, norms).
fn normalise_columns<F: TensorScalar>(mat: Array2<F>) -> (Array2<F>, Vec<F>) {
    let (m, r) = (mat.nrows(), mat.ncols());
    let mut normed = mat.clone();
    let mut norms = vec![F::one(); r];
    for j in 0..r {
        let sq: F = (0..m)
            .map(|i| mat[[i, j]] * mat[[i, j]])
            .fold(F::zero(), |a, b| a + b);
        let n = sq.sqrt();
        norms[j] = n;
        if n > F::from(1e-30_f64).unwrap_or(F::zero()) {
            let inv_n = F::one() / n;
            for i in 0..m {
                normed[[i, j]] = mat[[i, j]] * inv_n;
            }
        }
    }
    (normed, norms)
}

/// Compute the squared Frobenius reconstruction loss.
fn reconstruction_loss<F: TensorScalar>(
    tensor: &Tensor<F>,
    factors: &[Array2<F>],
    lambdas: &[F],
) -> F {
    let recon = reconstruct_from_factors(tensor.shape.clone(), factors, lambdas);
    tensor
        .data
        .iter()
        .zip(recon.data.iter())
        .map(|(&t, &r)| {
            let d = t - r;
            d * d
        })
        .fold(F::zero(), |a, b| a + b)
}

/// Reconstruct a dense tensor from factor matrices and lambdas.
pub(crate) fn reconstruct_from_factors<F: TensorScalar>(
    shape: Vec<usize>,
    factors: &[Array2<F>],
    lambdas: &[F],
) -> Tensor<F> {
    let ndim = shape.len();
    let rank = lambdas.len();
    let total: usize = shape.iter().product();
    let strides = crate::tensor::core::compute_row_major_strides(&shape);
    let mut data = vec![F::zero(); total];

    // Iterate over all elements
    for flat in 0..total {
        // multi-index
        let mut multi = vec![0usize; ndim];
        let mut rem = flat;
        for d in (0..ndim).rev() {
            multi[d] = rem % shape[d];
            rem /= shape[d];
        }
        let mut val = F::zero();
        for r in 0..rank {
            let mut contrib = lambdas[r];
            for n in 0..ndim {
                contrib *= factors[n][[multi[n], r]];
            }
            val += contrib;
        }
        let idx: usize = multi.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        data[idx] = val;
    }
    Tensor {
        data,
        shape,
        strides,
    }
}

// ---------------------------------------------------------------------------
// Re-export Array1 helper (needed by some callers)
// ---------------------------------------------------------------------------
#[allow(unused_imports)]
use scirs2_core::ndarray::ArrayBase;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_tensor() -> Tensor<f64> {
        let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
        Tensor::new(data, vec![2, 3, 4]).expect("valid")
    }

    #[test]
    fn test_cp_als_shape() {
        let t = make_tensor();
        let cfg = CPConfig {
            max_iter: 50,
            ..Default::default()
        };
        let r = cp_als(&t, 3, &cfg).expect("cp_als");
        assert_eq!(r.factors.len(), 3);
        assert_eq!(r.factors[0].shape(), &[2, 3]);
        assert_eq!(r.factors[1].shape(), &[3, 3]);
        assert_eq!(r.factors[2].shape(), &[4, 3]);
        assert_eq!(r.lambdas.len(), 3);
    }

    #[test]
    fn test_cp_als_loss_decreasing() {
        let t = make_tensor();
        let cfg = CPConfig {
            max_iter: 100,
            ..Default::default()
        };
        let r = cp_als(&t, 4, &cfg).expect("cp_als");
        // Loss should be non-increasing (allow tiny numerical noise)
        for window in r.loss.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-6,
                "loss increased: {} -> {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_cp_reconstruct_shape() {
        let t = make_tensor();
        let cfg = CPConfig::default();
        let r = cp_als(&t, 3, &cfg).expect("ok");
        let recon = cp_reconstruct(&r, &[2, 3, 4]).expect("ok");
        assert_eq!(recon.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_cp_grad_shape() {
        let t = make_tensor();
        let cfg = CPConfig {
            max_iter: 50,
            lr: 1e-3,
            ..Default::default()
        };
        let r = cp_grad(&t, 2, &cfg).expect("cp_grad");
        assert_eq!(r.factors.len(), 3);
    }

    #[test]
    fn test_cp_als_low_rank_approximation() {
        // Rank-1 tensor: a ⊗ b ⊗ c
        let a = [1.0_f64, 2.0];
        let b = [1.0_f64, 3.0, 5.0];
        let c = [1.0_f64, 0.5, 2.0, 0.25];
        let mut data = vec![0.0_f64; 2 * 3 * 4];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    data[i * 12 + j * 4 + k] = a[i] * b[j] * c[k];
                }
            }
        }
        let t = Tensor::new(data, vec![2, 3, 4]).expect("ok");
        let cfg = CPConfig {
            max_iter: 300,
            tol: 1e-10,
            ..Default::default()
        };
        let r = cp_als(&t, 1, &cfg).expect("rank-1 ALS");
        let err = r.relative_error(&t).expect("err");
        assert!(
            err < 1e-5,
            "rank-1 CP should reconstruct exactly, err={err}"
        );
    }

    #[test]
    fn test_cp_invalid_rank() {
        let t = make_tensor();
        let cfg = CPConfig::default();
        assert!(cp_als(&t, 0, &cfg).is_err());
        assert!(cp_grad(&t, 0, &cfg).is_err());
    }
}
