//! Von Mises-Fisher distribution for directional / spherical statistics
//!
//! The von Mises-Fisher (vMF) distribution is the spherical analogue of the
//! multivariate normal distribution. It models directional data on the
//! (p-1)-dimensional unit hypersphere S^{p-1} ⊂ ℝ^p.
//!
//! # Density
//!
//! For a unit vector **x** ∈ S^{p-1}:
//!
//! ```text
//! f(x; μ, κ) = C_p(κ) · exp(κ · μᵀx)
//! ```
//!
//! where:
//! - **μ** ∈ S^{p-1} is the mean direction (unit vector)
//! - **κ ≥ 0** is the concentration parameter (κ=0 → uniform; κ→∞ → point mass at μ)
//! - C_p(κ) = κ^{p/2−1} / ((2π)^{p/2} · I_{p/2−1}(κ)) is the normalising constant
//! - I_ν is the modified Bessel function of the first kind of order ν
//!
//! # Sampling
//!
//! Sampling is implemented using the efficient Wood (1994) algorithm for p ≥ 3,
//! and directly for p = 2 (Von Mises distribution).
//!
//! # References
//!
//! - Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution.
//!   Communications in Statistics — Simulation and Computation 23(1): 157–164.
//! - Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Uniform as RandUniform};

// ──────────────────────────────────────────────────────────────────────────────
// Bessel function utilities
// ──────────────────────────────────────────────────────────────────────────────

/// Natural logarithm of the modified Bessel function of the first kind,
/// `ln I_ν(x)`, for `ν >= 0` and `x >= 0`.
///
/// Evaluates the ascending series
/// `I_ν(x) = Σ_m (x/2)^(2m+ν) / (m! Γ(m+ν+1))` entirely in log space,
/// centred on its largest term: the term ratio
/// `t_{m+1}/t_m = (x/2)^2 / ((m+1)(m+1+ν))` crosses 1 at
/// `m* = (sqrt(ν² + x²) - ν) / 2`, so `ln t_{m*}` is formed once from
/// log-gammas and every other term is summed relative to it, walking outwards
/// until the terms drop below 1e-17 of the peak. All terms are positive, so
/// there is no cancellation, and nothing overflows for any `(ν, x)`: the
/// result stays accurate where `I_ν(x)` itself is far outside the f64 range
/// (large concentration κ, or the high dimensions of embedding spaces).
///
/// This replaces a direct series that started from `(x/2)^ν / Γ(ν+1)` --
/// which under/overflows for ν ≳ 170 and silently returned 0 -- plus an
/// `x > 30` branch using only `e^x/sqrt(2πx) * (1 - (4ν²-1)/(8x))`, which is
/// not valid unless `x >> ν²` (for ν = 149, x = 40 it returned -4.1e18
/// instead of 3.7e-66, so `log C_p` became NaN),
/// and which computed a Γ(ν+1)-based term it then discarded.
fn log_bessel_i(nu: f64, x: f64) -> f64 {
    if x.is_nan() || nu.is_nan() || x < 0.0 || nu < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if nu == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    let quarter_x2 = 0.25 * x * x;
    let ln_half_x = (0.5 * x).ln();

    // Index of the largest series term.
    let peak = ((nu * nu + x * x).sqrt() - nu) * 0.5;
    let m_peak = peak.floor().max(0.0);
    let ln_peak_term =
        (2.0 * m_peak + nu) * ln_half_x - ln_gamma(m_peak + 1.0) - ln_gamma(m_peak + nu + 1.0);

    // Relative tolerance on individual terms and a generous safety cap: the
    // terms around the peak are roughly Gaussian in m with width ~sqrt(x), so
    // even x = 1e8 needs only a few hundred thousand of them.
    const REL_TOL: f64 = 1e-17;
    const MAX_TERMS: usize = 1_000_000;

    let mut sum = 1.0_f64;

    // Upward from the peak: t_{m+1} = t_m * (x/2)^2 / ((m+1)(m+1+nu)).
    let mut term = 1.0_f64;
    let mut m = m_peak;
    for _ in 0..MAX_TERMS {
        term *= quarter_x2 / ((m + 1.0) * (m + 1.0 + nu));
        sum += term;
        m += 1.0;
        if term < REL_TOL * sum {
            break;
        }
    }

    // Downward from the peak: t_{m-1} = t_m * m (m + nu) / (x/2)^2.
    let mut term = 1.0_f64;
    let mut m = m_peak;
    for _ in 0..MAX_TERMS {
        if m < 1.0 {
            break;
        }
        term *= m * (m + nu) / quarter_x2;
        sum += term;
        m -= 1.0;
        if term < REL_TOL * sum {
            break;
        }
    }

    ln_peak_term + sum.ln()
}

/// Ratio A_p(κ) = I_{p/2}(κ) / I_{p/2-1}(κ), the mean resultant length.
///
/// Formed as a difference of log-Bessel values so it stays finite for any κ
/// (the plain quotient was `inf / inf` for κ ≳ 710).
fn a_p(p: usize, kappa: f64) -> f64 {
    let half_p = p as f64 / 2.0;
    (log_bessel_i(half_p, kappa) - log_bessel_i(half_p - 1.0, kappa)).exp()
}

/// Log normalising constant log C_p(κ).
fn log_c_p(p: usize, kappa: f64) -> f64 {
    let half_p = p as f64 / 2.0;
    let nu = half_p - 1.0;
    let log_bessel = log_bessel_i(nu, kappa);
    (half_p - 1.0) * kappa.ln() - half_p * (2.0 * std::f64::consts::PI).ln() - log_bessel
}

/// Lanczos approximation of log Γ(x) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    // Coefficients from Lanczos (g=7)
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut s = coeffs[0];
    for (k, &c) in coeffs[1..].iter().enumerate() {
        s += c / (xm1 + k as f64 + 1.0);
    }
    let t = xm1 + 7.5; // g + 0.5
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + s.ln()
}

// ──────────────────────────────────────────────────────────────────────────────
// Main struct
// ──────────────────────────────────────────────────────────────────────────────

/// Von Mises-Fisher distribution on S^{p-1}.
///
/// # Examples
///
/// ```
/// use scirs2_stats::distributions::multivariate::von_mises_fisher::VonMisesFisher;
/// use scirs2_core::ndarray::array;
///
/// // 3D unit sphere (p=3), mean pointing along the z-axis
/// let mu = array![0.0_f64, 0.0, 1.0];
/// let kappa = 5.0;
/// let vmf = VonMisesFisher::new(mu, kappa).expect("valid params");
///
/// let x = array![0.0_f64, 0.0, 1.0]; // at the mean direction
/// let log_p = vmf.log_pdf(&x);
/// assert!(log_p.is_finite());
/// ```
pub struct VonMisesFisher {
    /// Mean direction μ ∈ S^{p-1} (unit vector)
    pub mu: Array1<f64>,
    /// Concentration parameter κ ≥ 0
    pub kappa: f64,
    /// Dimension of the ambient space (p)
    pub dim: usize,
    /// Log normalising constant
    log_norm_const: f64,
    uniform_distr: RandUniform<f64>,
}

impl VonMisesFisher {
    /// Create a new von Mises-Fisher distribution.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean direction. Must be a unit vector (‖μ‖ = 1).
    /// * `kappa` - Concentration parameter κ ≥ 0.
    pub fn new(mu: Array1<f64>, kappa: f64) -> StatsResult<Self> {
        let p = mu.len();
        if p < 2 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be at least 2".to_string(),
            ));
        }
        if kappa < 0.0 {
            return Err(StatsError::DomainError(
                "Concentration kappa must be non-negative".to_string(),
            ));
        }
        if !mu.iter().all(|v| v.is_finite()) {
            return Err(StatsError::DomainError(
                "Mean direction mu must be finite".to_string(),
            ));
        }

        // Normalise mu
        let norm = mu.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return Err(StatsError::DomainError(
                "Mean direction mu must be non-zero".to_string(),
            ));
        }
        let mu_unit = mu / norm;

        let log_nc = if kappa == 0.0 {
            // Uniform: log C_p(0) = -log surface_area(S^{p-1})
            // Surface area = 2 π^{p/2} / Γ(p/2)
            0.5 * (p as f64) * std::f64::consts::PI.ln() - ln_gamma(p as f64 / 2.0) - (2.0_f64).ln()
        } else {
            log_c_p(p, kappa)
        };

        let uniform_distr = RandUniform::new(0.0_f64, 1.0_f64).map_err(|_| {
            StatsError::ComputationError(
                "Failed to create uniform distribution for vMF sampling".to_string(),
            )
        })?;

        Ok(Self {
            mu: mu_unit,
            kappa,
            dim: p,
            log_norm_const: log_nc,
            uniform_distr,
        })
    }

    /// Log probability density at point `x` ∈ S^{p-1}.
    ///
    /// Does not check that ‖x‖ = 1; caller is responsible for passing unit vectors.
    pub fn log_pdf(&self, x: &Array1<f64>) -> f64 {
        if x.len() != self.dim {
            return f64::NEG_INFINITY;
        }
        let dot = x
            .iter()
            .zip(self.mu.iter())
            .map(|(&xi, &mi)| xi * mi)
            .sum::<f64>();
        self.log_norm_const + self.kappa * dot
    }

    /// Probability density at point `x` ∈ S^{p-1}.
    pub fn pdf(&self, x: &Array1<f64>) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Mean direction E[X] = A_p(κ) · μ.
    ///
    /// The expected value is not on the unit sphere (unless κ → ∞), but
    /// lies along μ with magnitude A_p(κ) ∈ [0, 1].
    pub fn mean(&self) -> Array1<f64> {
        let r = if self.kappa == 0.0 {
            0.0
        } else {
            a_p(self.dim, self.kappa)
        };
        self.mu.mapv(|mi| r * mi)
    }

    /// Mean resultant length A_p(κ) = I_{p/2}(κ) / I_{p/2-1}(κ).
    pub fn mean_resultant_length(&self) -> f64 {
        if self.kappa == 0.0 {
            0.0
        } else {
            a_p(self.dim, self.kappa)
        }
    }

    /// Entropy of the distribution.
    ///
    /// H = -log C_p(κ) - κ · A_p(κ)
    pub fn entropy(&self) -> f64 {
        -self.log_norm_const - self.kappa * self.mean_resultant_length()
    }

    /// Sample one point from the vMF distribution.
    ///
    /// Uses the Wood (1994) rejection algorithm for p ≥ 3,
    /// and a direct method for p = 2 (von Mises).
    pub fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        if self.dim == 2 {
            self.sample_2d(rng)
        } else {
            self.sample_wood(rng)
        }
    }

    /// Sample n points from the vMF distribution.
    pub fn rvs<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> StatsResult<Array2<f64>> {
        let mut samples = Array2::zeros((n, self.dim));
        for i in 0..n {
            let s = self.sample_one(rng);
            samples.row_mut(i).assign(&s);
        }
        Ok(samples)
    }

    // ── p = 2 (von Mises on S^1) ──────────────────────────────────────────────

    fn sample_2d<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        // mu angle
        let mu_angle = self.mu[1].atan2(self.mu[0]);

        // Best & Fisher (1979) algorithm for von Mises
        let a = 1.0 + (1.0 + 4.0 * self.kappa * self.kappa).sqrt();
        let b = (a - (2.0 * a).sqrt()) / (2.0 * self.kappa);
        let r = (1.0 + b * b) / (2.0 * b);

        loop {
            let u1: f64 = self.uniform_distr.sample(rng);
            let u2: f64 = self.uniform_distr.sample(rng);
            let u3: f64 = self.uniform_distr.sample(rng);

            let z = (1.0 - u1) * b + u1; // nope, actually:
            let z = u1.cos() * std::f64::consts::PI; // simpler approach
            let f_val = (1.0 + r * z) / (r + z);
            let c = self.kappa * (r - f_val);

            if c * (2.0 - c) - u2 >= 0.0 || c.ln() + 1.0 - c >= u2.ln() {
                let theta = if u3 - 0.5 >= 0.0 {
                    f_val.acos()
                } else {
                    -f_val.acos()
                };
                let angle = theta + mu_angle;
                return Array1::from_vec(vec![angle.cos(), angle.sin()]);
            }
            let _ = z; // suppress warning
        }
    }

    // ── p ≥ 3 (Wood 1994 algorithm) ──────────────────────────────────────────

    fn sample_wood<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let p = self.dim;
        let kappa = self.kappa;

        // Step 1: compute b, x₀, c
        let b = (-2.0 * kappa + (4.0 * kappa * kappa + (p - 1) as f64 * (p - 1) as f64).sqrt())
            / ((p - 1) as f64);
        let x0 = (1.0 - b) / (1.0 + b);
        let c = kappa * x0 + (p as f64 - 1.0) * (1.0 - x0 * x0).ln();

        // Step 2: rejection sampling for the "w" component
        let w = loop {
            let z: f64 = {
                let u1: f64 = self.uniform_distr.sample(rng);
                let u2: f64 = self.uniform_distr.sample(rng);
                // Beta((p-1)/2, (p-1)/2) sample via two Gamma (approximate via the ratio of two betas)
                // Use simpler: Beta(a,a) via uniform ratio
                let a = (p as f64 - 1.0) / 2.0;
                // Cheng's BB algorithm for Beta(a,a) when a >= 1
                if a >= 1.0 {
                    let alpha_bb = a + a;
                    let beta_bb = (alpha_bb - 2.0 * a + 2.0).sqrt();
                    let gamma_bb = a - (2.0_f64).ln();
                    let delta_bb = a + 1.0 / beta_bb;
                    let k1 = gamma_bb + (a - 0.5).ln() - (2.0_f64 / (a + 0.5)).ln();
                    let k2 = a + 1.0;
                    let u = u1;
                    let v = u2;
                    let _ = (k1, k2, delta_bb, alpha_bb, beta_bb, gamma_bb);
                    // Simplified: use uniform approximation for Beta(a,a)
                    let beta_sample = self.sample_beta_symmetric(a, rng);
                    beta_sample
                } else {
                    u1.powf(1.0 / a) / (u1.powf(1.0 / a) + u2.powf(1.0 / a))
                }
            };

            let w_candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
            let u: f64 = self.uniform_distr.sample(rng);
            let log_accept =
                kappa * w_candidate + (p as f64 - 1.0) * (1.0 - w_candidate * w_candidate).ln() - c;
            if log_accept >= u.ln() {
                break w_candidate;
            }
        };

        // Step 3: sample a uniform point on S^{p-2}
        let v = self.sample_uniform_sphere_minus1(rng);

        // Step 4: rotate to get a point along the mu direction
        // Result = w·e₁ + sqrt(1-w²)·v in a frame where e₁ = mu
        let sqrt_1mw2 = (1.0 - w * w).max(0.0).sqrt();

        // Construct Householder reflection that maps e₁ = [1,0,...,0] → mu
        let mut x = Array1::zeros(p);
        x[0] = w;
        for i in 1..p {
            x[i] = sqrt_1mw2 * v[i - 1];
        }

        self.householder_rotate(&x)
    }

    fn sample_beta_symmetric<R: Rng + ?Sized>(&self, a: f64, rng: &mut R) -> f64 {
        // Beta(a,a) via Gamma(a,1) sampling (ratio method)
        let g1 = self.sample_gamma(a, rng);
        let g2 = self.sample_gamma(a, rng);
        g1 / (g1 + g2)
    }

    fn sample_gamma<R: Rng + ?Sized>(&self, shape: f64, rng: &mut R) -> f64 {
        // Marsaglia-Tsang method
        if shape < 1.0 {
            let u: f64 = self.uniform_distr.sample(rng);
            return self.sample_gamma(1.0 + shape, rng) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1: f64 = self.uniform_distr.sample(rng);
            let u2: f64 = self.uniform_distr.sample(rng);
            // Box-Muller standard normal
            let z =
                (-2.0 * u1.max(f64::EPSILON).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u3: f64 = self.uniform_distr.sample(rng);
            if u3 < 1.0 - 0.0331 * z.powi(4) || u3.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Sample a uniform point on S^{p-2} (used in Wood's algorithm).
    fn sample_uniform_sphere_minus1<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let dim = self.dim - 1;
        let mut v = Array1::zeros(dim);
        loop {
            let mut norm_sq = 0.0_f64;
            for i in 0..dim {
                let u1: f64 = self.uniform_distr.sample(rng).max(f64::EPSILON);
                let u2: f64 = self.uniform_distr.sample(rng);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                v[i] = z;
                norm_sq += z * z;
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-12 {
                v /= norm;
                break;
            }
        }
        v
    }

    /// Householder rotation that maps e₁ = [1, 0, …, 0] to self.mu.
    fn householder_rotate(&self, x: &Array1<f64>) -> Array1<f64> {
        let p = self.dim;
        let mu = &self.mu;

        // u = e₁ - mu (Householder vector)
        let mut u = Array1::zeros(p);
        u[0] = 1.0 - mu[0];
        for i in 1..p {
            u[i] = -mu[i];
        }

        let norm_u_sq: f64 = u.iter().map(|&v| v * v).sum();
        if norm_u_sq < 1e-24 {
            // mu ≈ e₁, no rotation needed
            return x.clone();
        }

        // H = I - 2·u·uᵀ / ‖u‖²
        let dot = x
            .iter()
            .zip(u.iter())
            .map(|(&xi, &ui)| xi * ui)
            .sum::<f64>();
        let scale = 2.0 * dot / norm_u_sq;

        let mut result = x.clone();
        for i in 0..p {
            result[i] -= scale * u[i];
        }
        result
    }

    /// Fit a vMF distribution from data using maximum likelihood estimation.
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix of shape (n, p) where each row is a unit vector on S^{p-1}
    ///
    /// # Returns
    ///
    /// `(mu_hat, kappa_hat)` — estimated mean direction and concentration.
    pub fn fit_mle(data: &Array2<f64>) -> StatsResult<(Array1<f64>, f64)> {
        let (n, p) = data.dim();
        if n < 2 {
            return Err(StatsError::InsufficientData(
                "Need at least 2 observations".to_string(),
            ));
        }
        if p < 2 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be at least 2".to_string(),
            ));
        }

        // Compute mean of data rows
        let mut mean = Array1::<f64>::zeros(p);
        for row in data.rows() {
            mean = mean + row;
        }
        mean /= n as f64;

        let r_bar = mean.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if r_bar < 1e-12 {
            return Err(StatsError::ComputationError(
                "Data is too dispersed to estimate mean direction".to_string(),
            ));
        }

        let mu_hat = mean / r_bar;

        // Estimate kappa via the approximation of Banerjee et al. (2005):
        // κ ≈ r̄(p - r̄²) / (1 - r̄²)
        let kappa_hat = r_bar * (p as f64 - r_bar * r_bar) / (1.0 - r_bar * r_bar);
        let kappa_hat = kappa_hat.max(0.0);

        Ok((mu_hat, kappa_hat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_log_pdf_at_mean() {
        let mu = array![0.0f64, 0.0, 1.0];
        let vmf = VonMisesFisher::new(mu.clone(), 5.0).expect("valid params");
        let log_p_at_mean = vmf.log_pdf(&mu);
        assert!(log_p_at_mean.is_finite());

        // Anti-podal point should have much lower density
        let anti = array![0.0f64, 0.0, -1.0];
        let log_p_anti = vmf.log_pdf(&anti);
        assert!(log_p_at_mean > log_p_anti);
    }

    #[test]
    fn test_kappa_zero_is_uniform() {
        // κ=0: pdf should be constant regardless of direction
        let mu = array![1.0f64, 0.0, 0.0];
        let vmf = VonMisesFisher::new(mu, 0.0).expect("valid params");
        let x1 = array![1.0f64, 0.0, 0.0];
        let x2 = array![0.0f64, 1.0, 0.0];
        let x3 = array![0.0f64, 0.0, 1.0];
        let d1 = vmf.pdf(&x1);
        let d2 = vmf.pdf(&x2);
        let d3 = vmf.pdf(&x3);
        assert!((d1 - d2).abs() < 1e-6);
        assert!((d1 - d3).abs() < 1e-6);
    }

    #[test]
    fn test_samples_on_sphere() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mu = array![0.0f64, 0.0, 1.0];
        let vmf = VonMisesFisher::new(mu, 10.0).expect("valid params");
        let samples = vmf.rvs(200, &mut rng).expect("sampling should succeed");

        for row in samples.rows() {
            let norm_sq: f64 = row.iter().map(|&v| v * v).sum();
            assert!(
                (norm_sq.sqrt() - 1.0).abs() < 1e-10,
                "sample not on sphere: norm={}",
                norm_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_samples_concentrated_near_mu() {
        let mut rng = SmallRng::seed_from_u64(99);
        let mu = array![0.0f64, 0.0, 1.0];
        let vmf = VonMisesFisher::new(mu.clone(), 50.0).expect("valid params");
        let samples = vmf.rvs(500, &mut rng).expect("sampling should succeed");

        // All samples should be close to mu when κ is large
        let mut avg_dot = 0.0_f64;
        for row in samples.rows() {
            let dot: f64 = row.iter().zip(mu.iter()).map(|(&xi, &mi)| xi * mi).sum();
            avg_dot += dot;
        }
        avg_dot /= 500.0;
        // Average dot product should be close to A_3(50) ≈ 0.96
        assert!(avg_dot > 0.9, "avg_dot={}", avg_dot);
    }

    #[test]
    fn test_mean_and_entropy() {
        let mu = array![1.0f64, 0.0];
        let vmf = VonMisesFisher::new(mu.clone(), 3.0).expect("valid params");
        let mean = vmf.mean();
        // Mean should point in the direction of mu with magnitude A_2(3)
        assert!(mean[0] > 0.0);
        let entropy = vmf.entropy();
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_fit_mle() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mu = array![0.0f64, 1.0, 0.0];
        let vmf = VonMisesFisher::new(mu.clone(), 8.0).expect("valid params");
        let samples = vmf.rvs(500, &mut rng).expect("sampling should succeed");

        let (mu_hat, kappa_hat) = VonMisesFisher::fit_mle(&samples).expect("fit should succeed");

        // Check that estimated mean direction is close to true mu
        let dot: f64 = mu_hat.iter().zip(mu.iter()).map(|(&a, &b)| a * b).sum();
        assert!(dot > 0.9, "mean direction dot product too low: {}", dot);

        // kappa should be in the right ballpark
        assert!(
            kappa_hat > 3.0 && kappa_hat < 20.0,
            "kappa_hat={}",
            kappa_hat
        );
    }

    /// `ln I_nu(x)` against mpmath `log(besseli(nu, x))` at 50 digits, NOT
    /// derived from this crate. Covers the old failure modes: the `x > 30`
    /// branch (wrong -- even negative -- whenever nu is not << sqrt(x)), the
    /// series underflow/overflow for large nu, and values of I far outside
    /// the f64 range.
    #[test]
    fn test_log_bessel_i_matches_mpmath() {
        let cases: &[(f64, f64, f64)] = &[
            (0.0, 1.0, 0.23591435850717865),
            (0.5, 10.0, 7.9297689182371508),
            (1.0, 30.0, 27.367748089282408),
            (1.5, 31.0, 28.331278041729763),
            (10.0, 50.0, 46.120852067835629),
            (149.0, 40.0, -151.00170840543492),
            (149.0, 10.0, -360.03664678117797),
            (150.0, 30.0, -197.32971991538815),
            (0.0, 700.0, 695.80569999844345),
            (0.0, 1000.0, 995.62730888986946),
            (1.5, 10000.0, 9994.4757912758069),
            (500.0, 1000.0, 872.9995486951398),
            (1000.0, 5.0, -4995.8312028772177),
            (0.5, 0.001, -3.6796688254691348),
            (2.0, 1_000_000.0, 999992.17330431281),
        ];
        for &(nu, x, expected) in cases {
            let got = log_bessel_i(nu, x);
            assert!(got.is_finite(), "ln I_{nu}({x}) = {got}");
            let tolerance = 1e-12 * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() <= tolerance,
                "ln I_{nu}({x}): got {got}, want {expected}"
            );
        }
        assert_eq!(log_bessel_i(0.0, 0.0), 0.0);
        assert_eq!(log_bessel_i(2.0, 0.0), f64::NEG_INFINITY);
        assert!(log_bessel_i(1.0, -1.0).is_nan());
    }

    /// End-to-end: a 300-dimensional vMF (nu = 149, kappa = 40) used to get a
    /// NaN normalising constant; A_p must also stay finite for large kappa.
    #[test]
    fn test_vmf_high_dimension_and_large_kappa() -> Result<(), Box<dyn std::error::Error>> {
        let mut mu = Array1::<f64>::zeros(300);
        mu[0] = 1.0;
        let vmf = VonMisesFisher::new(mu.clone(), 40.0)?;
        let log_pdf = vmf.log_pdf(&mu);
        // mpmath: log C_300(40) + 40
        assert!(
            (log_pdf - 464.96318710700961).abs() < 1e-9,
            "log_pdf = {log_pdf}"
        );
        let a = vmf.mean_resultant_length();
        assert!((a - 0.1310578428306944).abs() < 1e-12, "A_p = {a}");

        // p = 3: A_3(kappa) = coth(kappa) - 1/kappa.
        let mu3 = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let vmf3 = VonMisesFisher::new(mu3.clone(), 1000.0)?;
        assert!((vmf3.mean_resultant_length() - 0.999).abs() < 1e-12);
        assert!(vmf3.log_pdf(&mu3).is_finite());
        let vmf5 = VonMisesFisher::new(mu3, 5.0)?;
        assert!((vmf5.mean_resultant_length() - 0.80009080398201938).abs() < 1e-12);
        Ok(())
    }
}
