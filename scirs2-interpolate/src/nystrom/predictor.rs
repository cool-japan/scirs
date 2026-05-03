//! Core Nyström Kriging predictor.
//!
//! Implements:
//!
//! * `NystromKriging` — the fitted model struct.
//! * `fit` — builds K_{m,m} (landmark-landmark), K_{n,m} (training-landmark),
//!   then solves (K̃ + σ²I) α = y via conjugate gradient where K̃ = K_{n,m}
//!   K_{m,m}^{-1} K_{n,m}^T is the Nyström approximation.
//! * `predict` — computes the posterior mean at test points.
//! * `predict_variance` — computes the diagonal posterior variance.
//!
//! The Cholesky factorisation of K_{m,m} is used for stable inversion of the
//! landmark kernel matrix.  All linear algebra is performed in-house (the m×m
//! systems are small) so there is no dependency on scirs2-linalg here; the
//! conjugate-gradient solver handles the n-dimensional system matrix-free.

use crate::error::InterpolateError;
use crate::random_features::KernelType;
use scirs2_core::ndarray::{Array1, Array2};

// ─── Public re-exports ────────────────────────────────────────────────────────

pub use crate::nystrom::landmarks::LandmarkStrategy;

// ─── Kernel evaluation ───────────────────────────────────────────────────────

/// Evaluate the kernel between two slices: k(x1, x2).
#[inline]
pub fn kernel_eval(kernel: &KernelType, bw: f64, x1: &[f64], x2: &[f64]) -> f64 {
    match kernel {
        KernelType::Gaussian => {
            let sq: f64 = x1.iter().zip(x2).map(|(a, b)| (a - b).powi(2)).sum();
            (-sq / (2.0 * bw * bw)).exp()
        }
        KernelType::Laplacian => {
            let l1: f64 = x1.iter().zip(x2).map(|(a, b)| (a - b).abs()).sum();
            (-l1 / bw).exp()
        }
        KernelType::Cauchy => {
            let sq: f64 = x1.iter().zip(x2).map(|(a, b)| (a - b).powi(2)).sum();
            1.0 / (1.0 + sq / (bw * bw))
        }
        KernelType::Matern32 => {
            let r: f64 = x1
                .iter()
                .zip(x2)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let v = 3.0_f64.sqrt() * r / bw;
            (1.0 + v) * (-v).exp()
        }
        KernelType::Matern52 => {
            let r: f64 = x1
                .iter()
                .zip(x2)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let v = 5.0_f64.sqrt() * r / bw;
            (1.0 + v + v * v / 3.0) * (-v).exp()
        }
        #[allow(unreachable_patterns)]
        _ => {
            // Forward-compatible fallback for any future `#[non_exhaustive]` variants
            let sq: f64 = x1.iter().zip(x2).map(|(a, b)| (a - b).powi(2)).sum();
            (-sq / (2.0 * bw * bw)).exp()
        }
    }
}

// ─── NystromKriging ──────────────────────────────────────────────────────────

/// Nyström-approximated Gaussian process / Kriging model.
///
/// # Algorithm
///
/// Given n training points x and m landmarks x*, the Nyström approximation
/// replaces the n×n kernel matrix K with:
///
/// ```text
/// K̃ ≈ K_{nm} · K_{mm}^{-1} · K_{nm}^T
/// ```
///
/// Prediction at test point x**:
///
/// ```text
/// mean  = k_{m}(x**) · K_{mm}^{-1} · K_{nm}^T · α
/// var   = k(x**,x**) - k_{m}(x**)^T · K_{mm}^{-1} · k_{m}(x**)
/// ```
///
/// where α solves (K̃ + σ²I) α = y via conjugate gradient.
#[derive(Debug, Clone)]
pub struct NystromKriging {
    /// Selected landmark points, shape `m × d`.
    pub landmarks: Array2<f64>,
    /// Kernel type.
    pub kernel: KernelType,
    /// Kernel bandwidth / length-scale.
    pub bandwidth: f64,
    /// Observation noise variance σ².
    pub noise_variance: f64,
    /// Landmark strategy used.
    pub strategy: LandmarkStrategy,
    /// Random seed for landmark selection.
    pub seed: u64,
    /// Number of landmarks requested.
    n_landmarks: usize,

    // ── Fitted quantities ────────────────────────────────────────────────────
    /// K_{nm}: training-landmark kernel matrix (n × m).
    k_nm: Vec<Vec<f64>>,
    /// K_{mm}^{-1}: inverse of landmark kernel matrix (m × m).
    k_mm_inv: Vec<Vec<f64>>,
    /// Dual coefficients α (n-vector).
    alpha: Vec<f64>,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl NystromKriging {
    /// Create a new unfitted Nyström Kriging model.
    ///
    /// # Arguments
    /// * `kernel` — kernel type (Gaussian, Laplacian, Cauchy, Matérn 3/2, Matérn 5/2).
    /// * `m` — number of landmark points.  Must be ≥ 1.
    /// * `strategy` — landmark selection strategy.
    /// * `noise_variance` — observation noise σ² added to the diagonal.
    /// * `bandwidth` — kernel length-scale / bandwidth.
    /// * `seed` — random seed for landmark selection.
    pub fn new(
        kernel: KernelType,
        m: usize,
        strategy: LandmarkStrategy,
        noise_variance: f64,
        bandwidth: f64,
        seed: u64,
    ) -> Self {
        Self {
            landmarks: Array2::zeros((0, 0)),
            kernel,
            bandwidth,
            noise_variance,
            strategy,
            seed,
            n_landmarks: m,
            k_nm: Vec::new(),
            k_mm_inv: Vec::new(),
            alpha: Vec::new(),
            fitted: false,
        }
    }

    /// Number of landmarks requested at construction.
    pub fn n_landmarks(&self) -> usize {
        self.n_landmarks
    }

    /// Fit the model to training data.
    ///
    /// * `x` — training input array of shape (n, d).
    /// * `y` — training target array of shape (n,).
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), InterpolateError> {
        let n = x.nrows();
        let d = x.ncols();

        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Training set is empty".to_string(),
            ));
        }
        if n != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x has {n} rows but y has {} elements",
                y.len()
            )));
        }
        if d == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "Input dimension d must be ≥ 1".to_string(),
            });
        }

        // ── Select landmarks ──────────────────────────────────────────────────
        use crate::nystrom::landmarks::select_landmarks;
        self.landmarks = select_landmarks(
            x,
            self.n_landmarks,
            &self.strategy,
            self.seed,
            self.bandwidth,
        )?;

        let m = self.landmarks.nrows();

        // ── Build K_{mm} (m × m) and invert it ───────────────────────────────
        let k_mm_raw = build_kernel_matrix_2d(
            &self.landmarks,
            &self.landmarks,
            &self.kernel,
            self.bandwidth,
        );
        let mut k_mm_reg = k_mm_raw;
        for i in 0..m {
            k_mm_reg[i][i] += self.noise_variance.max(1e-10);
        }
        self.k_mm_inv = cholesky_inv(&k_mm_reg)?;

        // ── Build K_{nm} (n × m) ─────────────────────────────────────────────
        let landmarks_vv: Vec<Vec<f64>> = (0..m)
            .map(|r| (0..d).map(|c| self.landmarks[[r, c]]).collect())
            .collect();
        let x_vv: Vec<Vec<f64>> = (0..n)
            .map(|r| (0..d).map(|c| x[[r, c]]).collect())
            .collect();
        self.k_nm = build_kernel_matrix_vv(&x_vv, &landmarks_vv, &self.kernel, self.bandwidth);

        // ── Solve (K̃ + σ²I) α = y via conjugate gradient ───────────────────
        let rhs: Vec<f64> = y.iter().copied().collect();
        let sigma2 = self.noise_variance;
        let k_nm = self.k_nm.clone();
        let k_mm_inv = self.k_mm_inv.clone();

        let mv = move |v: &[f64]| -> Vec<f64> {
            // K̃ v = K_{nm} K_{mm}^{-1} (K_{nm}^T v)
            let t1 = mat_vec_t(&k_nm, v); // m-vector
            let t2 = mat_vec(&k_mm_inv, &t1); // m-vector
            let mut res = mat_vec(&k_nm, &t2); // n-vector
            for (ri, vi) in res.iter_mut().zip(v.iter()) {
                *ri += sigma2 * vi;
            }
            res
        };

        self.alpha = conjugate_gradient(mv, &rhs, 500)?;
        self.fitted = true;
        Ok(())
    }

    /// Predict the posterior mean at `x_star` (shape n_test × d).
    pub fn predict(&self, x_star: &Array2<f64>) -> Result<Array1<f64>, InterpolateError> {
        self.check_fitted()?;
        let k_test = self.compute_k_test_m(x_star);

        // mean = k_test_m · K_{mm}^{-1} · K_{nm}^T · α
        let knt_alpha = mat_vec_t(&self.k_nm, &self.alpha); // m-vector
        let km_inv_v = mat_vec(&self.k_mm_inv, &knt_alpha); // m-vector
        let means: Vec<f64> = k_test.iter().map(|row| dot(row, &km_inv_v)).collect();

        Ok(Array1::from_vec(means))
    }

    /// Predict the posterior variance at `x_star` (shape n_test × d).
    ///
    /// Variance: `k(x*,x*) - k_{m}(x*)^T K_{mm}^{-1} k_{m}(x*)` clamped to 0.
    pub fn predict_variance(&self, x_star: &Array2<f64>) -> Result<Array1<f64>, InterpolateError> {
        self.check_fitted()?;
        let n_test = x_star.nrows();
        let d_star = x_star.ncols();
        let m = self.landmarks.nrows();
        let d_lm = self.landmarks.ncols();

        let lm_vv: Vec<Vec<f64>> = (0..m)
            .map(|r| (0..d_lm).map(|c| self.landmarks[[r, c]]).collect())
            .collect();

        let mut vars = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let xi: Vec<f64> = (0..d_star).map(|c| x_star[[i, c]]).collect();
            let k_self = kernel_eval(&self.kernel, self.bandwidth, &xi, &xi);

            let k_m: Vec<f64> = lm_vv
                .iter()
                .map(|lm| kernel_eval(&self.kernel, self.bandwidth, &xi, lm))
                .collect();

            let km_inv_km = mat_vec(&self.k_mm_inv, &k_m);
            let quad: f64 = k_m.iter().zip(km_inv_km.iter()).map(|(a, b)| a * b).sum();

            vars.push((k_self - quad).max(0.0));
        }
        Ok(Array1::from_vec(vars))
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn check_fitted(&self) -> Result<(), InterpolateError> {
        if !self.fitted {
            Err(InterpolateError::InvalidState(
                "NystromKriging: call fit() before predict()".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn compute_k_test_m(&self, x_star: &Array2<f64>) -> Vec<Vec<f64>> {
        let n_test = x_star.nrows();
        let d = x_star.ncols();
        let m = self.landmarks.nrows();
        let d_lm = self.landmarks.ncols();
        let lm: Vec<Vec<f64>> = (0..m)
            .map(|r| (0..d_lm).map(|c| self.landmarks[[r, c]]).collect())
            .collect();

        (0..n_test)
            .map(|i| {
                let xi: Vec<f64> = (0..d).map(|c| x_star[[i, c]]).collect();
                lm.iter()
                    .map(|l| kernel_eval(&self.kernel, self.bandwidth, &xi, l))
                    .collect()
            })
            .collect()
    }
}

// ─── Kernel matrix builders ──────────────────────────────────────────────────

fn build_kernel_matrix_2d(
    a: &Array2<f64>,
    b: &Array2<f64>,
    kernel: &KernelType,
    bw: f64,
) -> Vec<Vec<f64>> {
    let na = a.nrows();
    let nb = b.nrows();
    let d = a.ncols();
    (0..na)
        .map(|i| {
            let ai: Vec<f64> = (0..d).map(|c| a[[i, c]]).collect();
            (0..nb)
                .map(|j| {
                    let bj: Vec<f64> = (0..d).map(|c| b[[j, c]]).collect();
                    kernel_eval(kernel, bw, &ai, &bj)
                })
                .collect()
        })
        .collect()
}

fn build_kernel_matrix_vv(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    kernel: &KernelType,
    bw: f64,
) -> Vec<Vec<f64>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| kernel_eval(kernel, bw, ai, bj)).collect())
        .collect()
}

// ─── Matrix-vector routines ──────────────────────────────────────────────────

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn mat_vec_t(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a[0].len();
    let mut result = vec![0.0f64; m];
    for (i, row) in a.iter().enumerate() {
        if i < x.len() {
            for (j, &aij) in row.iter().enumerate() {
                result[j] += aij * x[i];
            }
        }
    }
    result
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Cholesky inversion ──────────────────────────────────────────────────────

/// Invert a symmetric positive-definite matrix via Cholesky decomposition.
///
/// Returns K^{-1} stored as a dense m×m Vec-of-Vec.
fn cholesky_inv(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, InterpolateError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Lower-triangular Cholesky factor L  (A = L L^T)
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                let s = s.max(1e-14);
                l[i][j] = s.sqrt();
            } else {
                let ljj = l[j][j];
                if ljj.abs() < 1e-300 {
                    return Err(InterpolateError::LinalgError(
                        "Cholesky: zero diagonal element; matrix is singular".to_string(),
                    ));
                }
                l[i][j] = s / ljj;
            }
        }
    }

    // Compute L^{-1} by forward substitution on identity columns
    let mut l_inv = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        l_inv[j][j] = 1.0 / l[j][j];
        for i in (j + 1)..n {
            let mut s = 0.0f64;
            for k in j..i {
                s += l[i][k] * l_inv[k][j];
            }
            l_inv[i][j] = -s / l[i][i];
        }
    }

    // A^{-1} = (L^{-1})^T L^{-1}
    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let val: f64 = (0..n).map(|k| l_inv[k][i] * l_inv[k][j]).sum();
            inv[i][j] = val;
            inv[j][i] = val;
        }
    }
    Ok(inv)
}

// ─── Conjugate gradient ──────────────────────────────────────────────────────

/// Matrix-free CG solver for A x = b where A is SPD (applied via closure).
fn conjugate_gradient(
    a_mv: impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    max_iter: usize,
) -> Result<Vec<f64>, InterpolateError> {
    let n = b.len();
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = dot(&r, &r);

    if rs_old.sqrt() < 1e-14 {
        return Ok(x);
    }

    for _ in 0..max_iter {
        let ap = a_mv(&p);
        let pap = dot(&p, &ap);
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() < 1e-10 {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(x)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_features::KernelType;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_1d(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x: Array2<f64> =
            Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64 / n as f64).collect())
                .expect("shape");
        let y = x.column(0).mapv(|v| (2.0 * std::f64::consts::PI * v).sin());
        (x, y)
    }

    #[test]
    fn predictor_fit_predict_1d() {
        let (x, y) = make_1d(60);
        let mut model = NystromKriging::new(
            KernelType::Gaussian,
            15,
            LandmarkStrategy::UniformRandom,
            1e-4,
            0.3,
            42,
        );
        model.fit(&x, &y).expect("fit");
        let preds = model.predict(&x).expect("predict");
        assert_eq!(preds.len(), 60);
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, yi)| (p - yi).powi(2))
            .sum::<f64>()
            / 60.0;
        assert!(mse < 1.0, "MSE too high: {mse}");
    }

    #[test]
    fn predictor_variance_non_negative() {
        let (x, y) = make_1d(30);
        let mut model = NystromKriging::new(
            KernelType::Gaussian,
            10,
            LandmarkStrategy::UniformRandom,
            1e-4,
            0.5,
            7,
        );
        model.fit(&x, &y).expect("fit");
        let x_test: Array2<f64> =
            Array2::from_shape_vec((5, 1), vec![0.1, 0.3, 0.5, 0.7, 0.9]).expect("shape");
        let vars = model.predict_variance(&x_test).expect("variance");
        for v in vars.iter() {
            assert!(*v >= 0.0, "negative variance: {v}");
        }
    }

    #[test]
    fn predict_before_fit_returns_error() {
        let model = NystromKriging::new(
            KernelType::Gaussian,
            5,
            LandmarkStrategy::UniformRandom,
            1e-4,
            1.0,
            0,
        );
        let x = Array2::zeros((3, 1));
        assert!(model.predict(&x).is_err());
    }
}
