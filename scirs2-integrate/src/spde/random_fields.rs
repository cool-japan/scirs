//! Random field generation for Stochastic PDEs
//!
//! Provides three methods for sampling Gaussian random fields over 2D grids:
//!
//! 1. **Circulant Embedding**: Extends the covariance matrix to a circulant structure,
//!    enabling exact (in distribution) sampling via FFT. O(N log N) per sample.
//!
//! 2. **Karhunen-Loève (KL) Expansion**: Truncated eigendecomposition of the covariance
//!    operator. Provides a low-rank representation and error control via n_terms.
//!
//! 3. **Fourier Spectral Sampling**: Samples in the frequency domain using the spectral
//!    density (power spectrum) of the covariance function. Fast but uses periodic BCs.
//!
//! All methods produce zero-mean Gaussian fields; callers can scale by σ as needed.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;
use scirs2_fft::{fft, ifft};

/// Covariance/correlation function parameterisation for random fields.
#[derive(Debug, Clone)]
pub enum CorrelationFunction {
    /// C(r) = exp(-r / ℓ)
    Exponential { length_scale: f64 },
    /// C(r) = exp(-r² / (2ℓ²))
    Gaussian { length_scale: f64 },
    /// Matérn covariance with smoothness ν.
    /// ν = 0.5 → Exponential; ν = ∞ → Squared-Exponential.
    Matern { nu: f64, length_scale: f64 },
    /// C(r) = exp(-(r/ℓ)^α)
    Powered { exponent: f64, length_scale: f64 },
}

impl CorrelationFunction {
    /// Evaluate the correlation at (scalar) distance `r ≥ 0`.
    pub fn evaluate(&self, r: f64) -> f64 {
        match self {
            CorrelationFunction::Exponential { length_scale } => (-r / length_scale).exp(),
            CorrelationFunction::Gaussian { length_scale } => {
                let z = r / length_scale;
                (-0.5 * z * z).exp()
            }
            CorrelationFunction::Matern { nu, length_scale } => {
                evaluate_matern(r, *nu, *length_scale)
            }
            CorrelationFunction::Powered {
                exponent,
                length_scale,
            } => {
                let z = r / length_scale;
                (-(z.powf(*exponent))).exp()
            }
        }
    }

    /// Return the 1-D spectral density S(ω) (Fourier transform of the covariance).
    pub fn spectral_density_1d(&self, omega: f64) -> f64 {
        let omega2 = omega * omega;
        match self {
            CorrelationFunction::Exponential { length_scale } => {
                let ell = length_scale;
                2.0 * ell / (1.0 + ell * ell * omega2)
            }
            CorrelationFunction::Gaussian { length_scale } => {
                let ell = length_scale;
                ell * std::f64::consts::TAU.sqrt() * (-0.5 * ell * ell * omega2).exp()
            }
            CorrelationFunction::Matern { nu, length_scale } => {
                let ell = length_scale;
                let lambda = (2.0 * nu).sqrt() / ell;
                match (nu * 2.0).round() as i32 {
                    1 => 2.0 / (lambda * lambda + omega2),
                    3 => {
                        let d = lambda * lambda + omega2;
                        4.0 * lambda * lambda / (d * d)
                    }
                    5 => {
                        let d = lambda * lambda + omega2;
                        (8.0 / 3.0) * lambda.powi(4) / d.powi(3)
                    }
                    _ => {
                        let eff_ell = ell * (*nu).sqrt();
                        eff_ell
                            * std::f64::consts::TAU.sqrt()
                            * (-0.5 * eff_ell * eff_ell * omega2).exp()
                    }
                }
            }
            CorrelationFunction::Powered {
                exponent: _,
                length_scale,
            } => {
                let ell = length_scale;
                ell * std::f64::consts::TAU.sqrt() * (-0.5 * ell * ell * omega2).exp()
            }
        }
    }
}

/// A realised random field on a 2-D grid.
#[derive(Debug, Clone)]
pub struct RandomField {
    /// Grid values, shape `[nx, ny]`.
    pub grid: Array2<f64>,
    /// Covariance structure used to generate this field.
    pub covariance: CorrelationFunction,
}

impl RandomField {
    /// Sample a Gaussian random field using the **circulant embedding** method.
    ///
    /// Embeds the covariance into a `[2nx × 2ny]` circulant block matrix,
    /// computes eigenvalues via 2-D FFT, and multiplies complex Gaussian noise
    /// by the square-root eigenvalues in the frequency domain.
    ///
    /// # Errors
    /// Returns an error if `nx` or `ny` is zero.
    pub fn sample_circulant_embedding(
        grid_x: ArrayView1<f64>,
        grid_y: ArrayView1<f64>,
        cov: CorrelationFunction,
        rng: &mut StdRng,
    ) -> IntegrateResult<Array2<f64>> {
        let nx = grid_x.len();
        let ny = grid_y.len();
        if nx == 0 || ny == 0 {
            return Err(IntegrateError::InvalidInput(
                "Grid dimensions must be positive".to_string(),
            ));
        }

        let m = 2 * nx;
        let n = 2 * ny;
        let normal = Normal::new(0.0_f64, 1.0).map_err(|e| {
            IntegrateError::ComputationError(format!("Normal distribution error: {e}"))
        })?;

        // Build first row of embedded circulant covariance matrix
        let mut cov_flat = vec![0.0_f64; m * n];
        for i in 0..m {
            let xi = if i < nx {
                grid_x[i] - grid_x[0]
            } else if i == nx {
                // midpoint of circulant embedding: use max distance
                grid_x[nx - 1] - grid_x[0]
            } else {
                // i in (nx, 2*nx): reflect, m - i < nx
                grid_x[m - i] - grid_x[0]
            };
            for j in 0..n {
                let yj = if j < ny {
                    grid_y[j] - grid_y[0]
                } else if j == ny {
                    // midpoint: use max distance
                    grid_y[ny - 1] - grid_y[0]
                } else {
                    // j in (ny, 2*ny): reflect, n - j < ny
                    grid_y[n - j] - grid_y[0]
                };
                let r = (xi * xi + yj * yj).sqrt();
                cov_flat[i * n + j] = cov.evaluate(r);
            }
        }

        // 2-D FFT of covariance row to obtain eigenvalues (real part)
        let eigenvalues = fft2d_real_from_flat(&cov_flat, m, n)?;

        // Generate complex Gaussian noise scaled by sqrt(eigenvalue / (m*n))
        let scale = 1.0 / ((m * n) as f64).sqrt();
        let mut noise_complex: Vec<Complex64> = (0..m * n)
            .map(|_| {
                let re = rng.sample(normal);
                let im = rng.sample(normal);
                Complex64::new(re, im)
            })
            .collect();

        for (idx, c) in noise_complex.iter_mut().enumerate() {
            let lambda = eigenvalues[idx].max(0.0).sqrt() * scale;
            c.re *= lambda;
            c.im *= lambda;
        }

        // Inverse 2-D FFT → real part, crop to [nx, ny]
        let field_full = ifft2d_complex_flat(&noise_complex, m, n)?;

        let mut result = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                result[[i, j]] = field_full[i * n + j];
            }
        }
        Ok(result)
    }

    /// Sample a Gaussian random field using the **Karhunen-Loève expansion**.
    ///
    /// Builds the discrete covariance matrix over a flattened `[nx × ny]` grid,
    /// extracts leading `n_terms` eigenpairs via power iteration, then assembles:
    ///
    /// ```text
    /// u(x) = Σ_{k=1}^{n_terms} sqrt(λ_k) * ξ_k * φ_k(x)
    /// ```
    ///
    /// # Errors
    /// Returns an error if the grid or `n_terms` is invalid.
    pub fn sample_kl_expansion(
        grid_x: ArrayView1<f64>,
        grid_y: ArrayView1<f64>,
        cov: CorrelationFunction,
        n_terms: usize,
        rng: &mut StdRng,
    ) -> IntegrateResult<Array2<f64>> {
        let nx = grid_x.len();
        let ny = grid_y.len();
        let n_pts = nx * ny;

        if n_pts == 0 {
            return Err(IntegrateError::InvalidInput(
                "Grid must be non-empty".to_string(),
            ));
        }
        if n_terms == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_terms must be at least 1".to_string(),
            ));
        }
        let n_terms = n_terms.min(n_pts);

        // Build flat coordinate arrays
        let mut coords_x = vec![0.0_f64; n_pts];
        let mut coords_y = vec![0.0_f64; n_pts];
        for i in 0..nx {
            for j in 0..ny {
                coords_x[i * ny + j] = grid_x[i];
                coords_y[i * ny + j] = grid_y[j];
            }
        }

        // Build covariance matrix (symmetric)
        let mut cov_mat = vec![0.0_f64; n_pts * n_pts];
        for p in 0..n_pts {
            for q in p..n_pts {
                let dx = coords_x[p] - coords_x[q];
                let dy = coords_y[p] - coords_y[q];
                let r = (dx * dx + dy * dy).sqrt();
                let c = cov.evaluate(r);
                cov_mat[p * n_pts + q] = c;
                cov_mat[q * n_pts + p] = c;
            }
        }

        // Extract leading eigenpairs via power iteration with deflation
        let (eigenvalues, eigenvectors) = power_iteration_eigenpairs(&cov_mat, n_pts, n_terms)?;

        let normal = Normal::new(0.0_f64, 1.0).map_err(|e| {
            IntegrateError::ComputationError(format!("Normal distribution error: {e}"))
        })?;

        let xi: Vec<f64> = (0..n_terms).map(|_| rng.sample(normal)).collect();

        // Assemble u = Σ sqrt(λ_k) * ξ_k * φ_k
        let mut u_flat = vec![0.0_f64; n_pts];
        for k in 0..n_terms {
            let lambda_k = eigenvalues[k].max(0.0).sqrt() * xi[k];
            for p in 0..n_pts {
                u_flat[p] += lambda_k * eigenvectors[k][p];
            }
        }

        let mut result = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                result[[i, j]] = u_flat[i * ny + j];
            }
        }
        Ok(result)
    }

    /// Sample a Gaussian random field using the **Fourier spectral method**.
    ///
    /// Assumes periodic BCs and separable (isotropic) covariance.
    ///
    /// # Errors
    /// Returns an error if `nx` or `ny` is zero.
    pub fn sample_fourier(
        nx: usize,
        ny: usize,
        lx: f64,
        ly: f64,
        cov: CorrelationFunction,
        rng: &mut StdRng,
    ) -> IntegrateResult<Array2<f64>> {
        if nx == 0 || ny == 0 {
            return Err(IntegrateError::InvalidInput(
                "Grid dimensions must be positive".to_string(),
            ));
        }

        let normal = Normal::new(0.0_f64, 1.0).map_err(|e| {
            IntegrateError::ComputationError(format!("Normal distribution error: {e}"))
        })?;

        let dx = lx / nx as f64;
        let dy = ly / ny as f64;
        let two_pi_over_lx = std::f64::consts::TAU / lx;
        let two_pi_over_ly = std::f64::consts::TAU / ly;

        // Build spectral coefficients
        let mut z_complex: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nx * ny];
        for k in 0..nx {
            let omega_x = (if k <= nx / 2 {
                k as f64
            } else {
                k as f64 - nx as f64
            }) * two_pi_over_lx;
            for l in 0..ny {
                let omega_y = (if l <= ny / 2 {
                    l as f64
                } else {
                    l as f64 - ny as f64
                }) * two_pi_over_ly;
                let s_x = cov.spectral_density_1d(omega_x);
                let s_y = cov.spectral_density_1d(omega_y);
                let amplitude = (s_x * s_y / (dx * dy)).max(0.0).sqrt();
                let re = rng.sample(normal);
                let im = rng.sample(normal);
                z_complex[k * ny + l] = Complex64::new(amplitude * re, amplitude * im);
            }
        }

        // Inverse 2-D FFT → real part is the field
        let field_full = ifft2d_complex_flat(&z_complex, nx, ny)?;

        let mut result = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                result[[i, j]] = field_full[i * ny + j];
            }
        }
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate Matérn covariance at distance `r`.
fn evaluate_matern(r: f64, nu: f64, length_scale: f64) -> f64 {
    if r < 1e-14 {
        return 1.0;
    }
    let sqrt2nu_r_over_ell = (2.0 * nu).sqrt() * r / length_scale;
    let nu2 = (nu * 2.0).round() as i32;
    match nu2 {
        1 => (-sqrt2nu_r_over_ell).exp(),
        3 => {
            let x = sqrt2nu_r_over_ell;
            (1.0 + x) * (-x).exp()
        }
        5 => {
            let x = sqrt2nu_r_over_ell;
            (1.0 + x + x * x / 3.0) * (-x).exp()
        }
        _ => {
            if nu > 50.0 {
                let z = r / length_scale;
                return (-0.5 * z * z).exp();
            }
            let x = sqrt2nu_r_over_ell;
            let bk = bessel_k_approx(nu, x);
            if bk <= 0.0 || !bk.is_finite() {
                return 0.0;
            }
            let log_val = nu * x.ln() - log_gamma(nu) + (1.0 - nu) * 2.0_f64.ln() + bk.ln();
            log_val.exp().min(1.0).max(0.0)
        }
    }
}

/// Log Gamma function via Lanczos approximation.
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_810,
        676.520_368_121_885,
        -1_259.139_216_722_403,
        771.323_428_777_653,
        -176.615_029_162_141,
        12.507_343_278_686_9,
        -0.138_571_095_265_720,
        9.984_369_578_019_57e-6,
        1.505_632_735_149_31e-7,
    ];
    let x = x - 1.0;
    let t = x + g + 0.5;
    let mut sum = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        sum += ci / (x + i as f64 + 1.0);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Temme asymptotic approximation of modified Bessel function K_ν(x).
fn bessel_k_approx(nu: f64, x: f64) -> f64 {
    if x < 1e-10 {
        return 1e10;
    }
    let sqrt_pi_over_2x = (std::f64::consts::PI / (2.0 * x)).sqrt();
    let exp_neg_x = (-x).exp();
    let correction = 1.0 + (4.0 * nu * nu - 1.0) / (8.0 * x);
    sqrt_pi_over_2x * exp_neg_x * correction
}

/// Row-major flat-array 2-D FFT returning the real part of the spectrum.
fn fft2d_real_from_flat(a: &[f64], m: usize, n: usize) -> IntegrateResult<Vec<f64>> {
    let complex_in: Vec<Complex64> = a.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let (real_out, _) = fft2d_complex_transform_flat(&complex_in, m, n, false)?;
    Ok(real_out)
}

/// 2-D inverse FFT of complex data; returns real part in flat row-major order.
fn ifft2d_complex_flat(z: &[Complex64], m: usize, n: usize) -> IntegrateResult<Vec<f64>> {
    let (real_out, _) = fft2d_complex_transform_flat(z, m, n, true)?;
    Ok(real_out)
}

/// 2-D forward or inverse FFT of complex data.
/// Applies 1-D FFT along columns then rows.
fn fft2d_complex_transform_flat(
    z_in: &[Complex64],
    m: usize,
    n: usize,
    inverse: bool,
) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    let total = m * n;
    let mut buf: Vec<Complex64> = z_in.to_vec();

    // FFT along columns (axis 0): for each column j, transform the m values
    for j in 0..n {
        let col: Vec<Complex64> = (0..m).map(|i| buf[i * n + j]).collect();
        let transformed = fft1d_complex_transform(&col, inverse)?;
        for i in 0..m {
            buf[i * n + j] = transformed[i];
        }
    }

    // FFT along rows (axis 1): for each row i, transform the n values
    for i in 0..m {
        let row: Vec<Complex64> = (0..n).map(|j| buf[i * n + j]).collect();
        let transformed = fft1d_complex_transform(&row, inverse)?;
        for j in 0..n {
            buf[i * n + j] = transformed[j];
        }
    }

    if inverse {
        let scale = 1.0 / total as f64;
        for c in buf.iter_mut() {
            c.re *= scale;
            c.im *= scale;
        }
    }

    let real_out: Vec<f64> = buf.iter().map(|c| c.re).collect();
    let imag_out: Vec<f64> = buf.iter().map(|c| c.im).collect();
    Ok((real_out, imag_out))
}

/// 1-D complex FFT using scirs2_fft.
fn fft1d_complex_transform(input: &[Complex64], inverse: bool) -> IntegrateResult<Vec<Complex64>> {
    if inverse {
        ifft(input, None).map_err(|e| IntegrateError::ComputationError(format!("IFFT error: {e}")))
    } else {
        fft(input, None).map_err(|e| IntegrateError::ComputationError(format!("FFT error: {e}")))
    }
}

/// Power-iteration method to extract `n_terms` leading eigenpairs of a symmetric
/// positive semi-definite dense matrix (row-major, size `n×n`).
fn power_iteration_eigenpairs(
    a: &[f64],
    n: usize,
    n_terms: usize,
) -> IntegrateResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut eigenvalues = Vec::with_capacity(n_terms);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(n_terms);

    let mut mat = a.to_vec();
    let max_iter = 1000;
    let tol = 1e-10;

    for _k in 0..n_terms {
        // Initialise with a deterministic starting vector
        let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
        normalize_vec(&mut v);

        let mut lambda_prev = 0.0_f64;

        for _iter in 0..max_iter {
            // w = A * v
            let mut w = vec![0.0_f64; n];
            for i in 0..n {
                let mut s = 0.0_f64;
                for j in 0..n {
                    s += mat[i * n + j] * v[j];
                }
                w[i] = s;
            }

            let lambda: f64 = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum();
            normalize_vec(&mut w);
            v = w;

            if (lambda - lambda_prev).abs() < tol {
                break;
            }
            lambda_prev = lambda;
        }

        // Final Rayleigh quotient
        let mut av = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..n {
                s += mat[i * n + j] * v[j];
            }
            av[i] = s;
        }
        let lambda: f64 = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();

        eigenvalues.push(lambda.max(0.0));
        eigenvectors.push(v.clone());

        // Deflation: A ← A - λ v vᵀ
        for i in 0..n {
            for j in 0..n {
                mat[i * n + j] -= lambda * v[i] * v[j];
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Normalise a vector in-place (Euclidean norm).
fn normalize_vec(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::random::prelude::*;

    fn make_rng() -> StdRng {
        seeded_rng(42)
    }

    #[test]
    fn test_correlation_function_at_zero() {
        let covs = [
            CorrelationFunction::Exponential { length_scale: 1.0 },
            CorrelationFunction::Gaussian { length_scale: 1.0 },
            CorrelationFunction::Matern {
                nu: 1.5,
                length_scale: 1.0,
            },
            CorrelationFunction::Powered {
                exponent: 1.5,
                length_scale: 1.0,
            },
        ];
        for cov in &covs {
            let c0 = cov.evaluate(0.0);
            assert!((c0 - 1.0).abs() < 1e-10, "C(0) must be 1, got {c0}");
        }
    }

    #[test]
    fn test_correlation_function_decreasing() {
        let cov = CorrelationFunction::Gaussian { length_scale: 1.0 };
        let c1 = cov.evaluate(0.5);
        let c2 = cov.evaluate(1.0);
        let c3 = cov.evaluate(2.0);
        assert!(
            c1 > c2 && c2 > c3,
            "Correlation should decrease with distance"
        );
    }

    #[test]
    fn test_circulant_embedding_shape() {
        let mut rng = make_rng();
        let gx = Array1::linspace(0.0, 1.0, 8);
        let gy = Array1::linspace(0.0, 1.0, 8);
        let cov = CorrelationFunction::Exponential { length_scale: 0.3 };
        let field = RandomField::sample_circulant_embedding(gx.view(), gy.view(), cov, &mut rng)
            .expect("Circulant embedding failed");
        assert_eq!(field.dim(), (8, 8));
    }

    #[test]
    fn test_kl_expansion_shape() {
        let mut rng = make_rng();
        let gx = Array1::linspace(0.0, 1.0, 6);
        let gy = Array1::linspace(0.0, 1.0, 6);
        let cov = CorrelationFunction::Gaussian { length_scale: 0.3 };
        let field = RandomField::sample_kl_expansion(gx.view(), gy.view(), cov, 10, &mut rng)
            .expect("KL expansion failed");
        assert_eq!(field.dim(), (6, 6));
    }

    #[test]
    fn test_fourier_sampling_shape() {
        let mut rng = make_rng();
        let cov = CorrelationFunction::Gaussian { length_scale: 0.3 };
        let field = RandomField::sample_fourier(8, 8, 1.0, 1.0, cov, &mut rng)
            .expect("Fourier sampling failed");
        assert_eq!(field.dim(), (8, 8));
    }

    #[test]
    fn test_matern_various_nu() {
        for nu in [0.5, 1.5, 2.5, 5.0] {
            let cov = CorrelationFunction::Matern {
                nu,
                length_scale: 1.0,
            };
            let c = cov.evaluate(1.0);
            assert!(
                c > 0.0 && c < 1.0,
                "Matérn({nu}) at r=1 should be in (0,1): got {c}"
            );
        }
    }

    #[test]
    fn test_fourier_field_finite() {
        let mut rng = make_rng();
        let cov = CorrelationFunction::Exponential { length_scale: 0.5 };
        let field = RandomField::sample_fourier(16, 16, 2.0, 2.0, cov, &mut rng)
            .expect("sample_fourier should succeed with valid params");
        assert!(
            field.iter().all(|v| v.is_finite()),
            "Fourier field contains non-finite values"
        );
    }
}
