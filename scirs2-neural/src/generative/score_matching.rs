//! Score Matching for density estimation
//!
//! Provides three complementary score matching objectives:
//!
//! ## Explicit Score Matching (ESM)
//! Matches the model score to the data score directly:
//! ```text
//! L_ESM = E_x [ ½ ||s_θ(x)||² + tr(∇_x s_θ(x)) ]
//! ```
//! Hessian trace is approximated via the Hutchinson estimator (Rademacher vectors).
//!
//! ## Denoising Score Matching (DSM)  (Vincent 2011)
//! ```text
//! L_DSM = E_{x,x̃} [ ||s_θ(x̃) + (x̃−x)/σ²||² ]
//! ```
//! where `x̃ = x + σε`, `ε ~ N(0,I)`.
//!
//! ## Sliced Score Matching (SSM) (Song et al. 2019)
//! ```text
//! L_SSM = E_{x,v} [ v^T ∇_x s_θ(x) v + ½ ||s_θ(x)||² ]
//! ```
//! with random projections `v` (Rademacher or Gaussian).
//!
//! # References
//! - Hyvärinen, A. (2005). Estimation of non-normalized statistical models.
//!   *J. Machine Learning Research*, 6, 695–709.
//! - Vincent, P. (2011). A connection between score matching and denoising autoencoders.
//!   *Neural Computation*, 23(7), 1661–1674.
//! - Song, Y., Garg, S., Shi, J. & Ermon, S. (2019). Sliced Score Matching.
//!   *UAI 2019*. <https://arxiv.org/abs/1905.07088>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// LCG helpers
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn lcg_uniform(state: &mut u64) -> f32 {
    (lcg_next(state) >> 11) as f32 / (1u64 << 53) as f32
}

fn box_muller(state: &mut u64) -> f32 {
    let u1 = (lcg_uniform(state) as f64 + 1e-12).min(1.0 - 1e-12);
    let u2 = lcg_uniform(state) as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos()) as f32
}

fn rademacher(state: &mut u64) -> f32 {
    if lcg_next(state) >> 63 == 0 {
        1.0
    } else {
        -1.0
    }
}

// ---------------------------------------------------------------------------
// ScoreNetwork — lightweight MLP score estimator
// ---------------------------------------------------------------------------

/// Configuration for [`ScoreNetwork`].
#[derive(Debug, Clone)]
pub struct ScoreNetworkConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of hidden layers.
    pub num_layers: usize,
    /// Random seed for weight initialisation.
    pub seed: u64,
}

impl ScoreNetworkConfig {
    /// Default config for low-dimensional data.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 64,
            num_layers: 3,
            seed: 42,
        }
    }

    /// Tiny config for unit tests.
    pub fn tiny(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim: 16,
            num_layers: 2,
            seed: 0,
        }
    }
}

/// Lightweight MLP that estimates the score function `s_θ(x) ≈ ∇_x log p(x)`.
///
/// Architecture:
/// - Input: `x` of length `input_dim`
/// - `num_layers` hidden layers with Softplus activation
/// - Output: `input_dim` (same shape as input)
///
/// Weights are initialised deterministically from an LCG PRNG so no external
/// random crate is needed.
#[derive(Debug, Clone)]
pub struct ScoreNetwork {
    /// Config used to build this network.
    pub config: ScoreNetworkConfig,
    /// Layer (weight_flat, bias): weight stored row-major W[j,i] = w[j*in+i].
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl ScoreNetwork {
    /// Build a [`ScoreNetwork`] from the given config.
    pub fn new(config: ScoreNetworkConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ScoreNetwork: input_dim must be > 0".to_string(),
            ));
        }
        if config.hidden_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ScoreNetwork: hidden_dim must be > 0".to_string(),
            ));
        }
        if config.num_layers == 0 {
            return Err(NeuralError::InvalidArgument(
                "ScoreNetwork: num_layers must be > 0".to_string(),
            ));
        }

        let mut rng = config.seed.wrapping_add(0xdeadbeef);
        let mut layers: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

        let xavier = |fan_in: usize, fan_out: usize, rng: &mut u64| -> Vec<f32> {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;
            (0..fan_in * fan_out)
                .map(|_| {
                    let bits = lcg_next(rng) >> 11;
                    let u = bits as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
                    u * limit
                })
                .collect()
        };

        // Input → hidden
        layers.push((
            xavier(config.input_dim, config.hidden_dim, &mut rng),
            vec![0.0f32; config.hidden_dim],
        ));

        // Hidden → hidden
        for _ in 1..config.num_layers {
            layers.push((
                xavier(config.hidden_dim, config.hidden_dim, &mut rng),
                vec![0.0f32; config.hidden_dim],
            ));
        }

        // Hidden → output
        layers.push((
            xavier(config.hidden_dim, config.input_dim, &mut rng),
            vec![0.0f32; config.input_dim],
        ));

        Ok(Self { config, layers })
    }

    /// Softplus activation: log(1 + exp(x)).
    #[inline]
    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    /// Forward pass through the MLP.
    fn mlp_forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h: Vec<f32> = x.to_vec();
        let n = self.layers.len();
        for (idx, (w, b)) in self.layers.iter().enumerate() {
            let out_dim = b.len();
            let in_dim = h.len();
            let mut next: Vec<f32> = (0..out_dim)
                .map(|j| {
                    let row = j * in_dim;
                    let dot: f32 = h
                        .iter()
                        .enumerate()
                        .map(|(i, &hi)| w.get(row + i).copied().unwrap_or(0.0) * hi)
                        .sum();
                    dot + b[j]
                })
                .collect();
            if idx < n - 1 {
                for v in &mut next {
                    *v = Self::softplus(*v);
                }
            }
            h = next;
        }
        h
    }

    /// Evaluate the score `s_θ(x)`.
    pub fn score(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.config.input_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "ScoreNetwork: input len {} != input_dim {}",
                x.len(),
                self.config.input_dim
            )));
        }
        Ok(self.mlp_forward(x))
    }

    /// Finite-difference Jacobian-vector product: `∇_x s(x) · v`.
    ///
    /// Uses symmetric finite differences with step ε = 1e-4.
    pub fn jvp(&self, x: &[f32], v: &[f32]) -> Result<Vec<f32>> {
        const EPS: f32 = 1e-4;
        let d = x.len();
        if v.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ScoreNetwork jvp: v len {} != input_dim {}",
                v.len(),
                d
            )));
        }
        let x_plus: Vec<f32> = x.iter().zip(v).map(|(&xi, &vi)| xi + EPS * vi).collect();
        let x_minus: Vec<f32> = x.iter().zip(v).map(|(&xi, &vi)| xi - EPS * vi).collect();
        let s_plus = self.score(&x_plus)?;
        let s_minus = self.score(&x_minus)?;
        Ok(s_plus
            .iter()
            .zip(&s_minus)
            .map(|(&sp, &sm)| (sp - sm) / (2.0 * EPS))
            .collect())
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|(w, b)| w.len() + b.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Explicit Score Matching (ESM / Hyvärinen)
// ---------------------------------------------------------------------------

/// Compute the **Explicit Score Matching** loss for a single data point.
///
/// ```text
/// L_ESM(x) = ½ ||s_θ(x)||² + tr(∇_x s_θ(x))
/// ```
///
/// The Hessian trace `tr(∇_x s_θ(x))` is estimated via the Hutchinson estimator:
/// ```text
/// tr(J) ≈ E_v [ v^T (J v) ],   v ~ Rademacher{±1}
/// ```
///
/// # Arguments
/// * `score_fn`    — Score network.
/// * `x`           — Data point.
/// * `n_hutchinson` — Number of Rademacher vectors for trace estimation.
/// * `rng`         — Mutable LCG state.
pub fn explicit_score_matching_loss(
    score_fn: &ScoreNetwork,
    x: &[f32],
    n_hutchinson: usize,
    rng: &mut u64,
) -> Result<f32> {
    let d = x.len();
    let s = score_fn.score(x)?;

    // ½ ||s_θ||²
    let half_sq_norm: f32 = s.iter().map(|&si| si * si).sum::<f32>() * 0.5 / d as f32;

    // Hutchinson trace estimate of tr(∇_x s_θ(x))
    let n = n_hutchinson.max(1);
    let mut trace_est = 0.0f32;
    for _ in 0..n {
        let v: Vec<f32> = (0..d).map(|_| rademacher(rng)).collect();
        // J v via finite-difference JVP
        let jvp = score_fn.jvp(x, &v)?;
        // v^T (J v)
        let vt_jv: f32 = v.iter().zip(&jvp).map(|(&vi, &ji)| vi * ji).sum();
        trace_est += vt_jv;
    }
    trace_est /= n as f32;

    Ok(half_sq_norm + trace_est)
}

// ---------------------------------------------------------------------------
// Denoising Score Matching (DSM)
// ---------------------------------------------------------------------------

/// Compute the **Denoising Score Matching** loss for a single data point.
///
/// ```text
/// L_DSM(x) = ||s_θ(x̃) − (−ε/σ)||²  =  ||s_θ(x̃) + ε/σ||²
/// ```
///
/// where `x̃ = x + σε`, `ε ~ N(0, I)`.
///
/// The optimal score of the noisy distribution is `−(x̃−x)/σ² = −ε/σ`.
///
/// # Arguments
/// * `score_fn` — Score network.
/// * `x`        — Clean data point.
/// * `sigma`    — Noise standard deviation (must be > 0).
/// * `rng`      — Mutable LCG state.
///
/// # Returns
/// `(loss, x_noisy)` — scalar DSM loss and the noisy sample used.
pub fn denoising_score_matching_loss(
    score_fn: &ScoreNetwork,
    x: &[f32],
    sigma: f32,
    rng: &mut u64,
) -> Result<(f32, Vec<f32>)> {
    if sigma <= 0.0 {
        return Err(NeuralError::InvalidArgument(format!(
            "denoising_score_matching_loss: sigma must be > 0, got {sigma}"
        )));
    }
    let d = x.len();

    // Sample ε ~ N(0,I) and construct x̃
    let eps: Vec<f32> = (0..d).map(|_| box_muller(rng)).collect();
    let x_noisy: Vec<f32> = x
        .iter()
        .zip(&eps)
        .map(|(&xi, &ei)| xi + sigma * ei)
        .collect();

    // Network score s_θ(x̃)
    let s_pred = score_fn.score(&x_noisy)?;

    // Target score: −ε/σ
    // L_DSM = ||s_θ(x̃) + ε/σ||²
    let loss: f32 = s_pred
        .iter()
        .zip(&eps)
        .map(|(&s, &e)| {
            let residual = s + e / sigma;
            residual * residual
        })
        .sum::<f32>()
        / d as f32;

    Ok((loss, x_noisy))
}

// ---------------------------------------------------------------------------
// Sliced Score Matching (SSM)
// ---------------------------------------------------------------------------

/// Projection distribution for Sliced Score Matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionDist {
    /// Rademacher ±1 (variance-optimal).
    Rademacher,
    /// Standard Gaussian.
    Gaussian,
}

/// Compute the **Sliced Score Matching** loss for a single data point.
///
/// ```text
/// L_SSM(x) = E_v [ v^T ∇_x s_θ(x) v + ½ ||s_θ(x)||² ]
/// ```
///
/// The divergence term `v^T ∇_x s_θ(x) v` is estimated via finite-difference JVP.
///
/// # Arguments
/// * `score_fn` — Score network.
/// * `x`        — Data point.
/// * `n_proj`   — Number of random projections.
/// * `dist`     — Projection distribution.
/// * `rng`      — Mutable LCG state.
pub fn sliced_score_matching_loss(
    score_fn: &ScoreNetwork,
    x: &[f32],
    n_proj: usize,
    dist: ProjectionDist,
    rng: &mut u64,
) -> Result<f32> {
    if n_proj == 0 {
        return Err(NeuralError::InvalidArgument(
            "sliced_score_matching_loss: n_proj must be > 0".to_string(),
        ));
    }
    let d = x.len();
    let s = score_fn.score(x)?;

    // ½ ||s_θ||²
    let half_sq_norm: f32 = s.iter().map(|&si| si * si).sum::<f32>() * 0.5 / d as f32;

    let mut div_term = 0.0f32;
    for _ in 0..n_proj {
        let v: Vec<f32> = match dist {
            ProjectionDist::Rademacher => (0..d).map(|_| rademacher(rng)).collect(),
            ProjectionDist::Gaussian => (0..d).map(|_| box_muller(rng)).collect(),
        };
        let jvp = score_fn.jvp(x, &v)?;
        let vt_jv: f32 = v.iter().zip(&jvp).map(|(&vi, &ji)| vi * ji).sum();
        div_term += vt_jv;
    }
    div_term /= n_proj as f32;

    Ok(div_term + half_sq_norm)
}

// ---------------------------------------------------------------------------
// Multi-scale (annealed) DSM
// ---------------------------------------------------------------------------

/// Compute the annealed (multi-scale) DSM loss over a geometric noise schedule.
///
/// Samples one noise level per call, weighted by σ²:
/// ```text
/// L_annealed = σ_i² · L_DSM(x, σ_i)
/// ```
///
/// # Arguments
/// * `score_fn`    — Score network.
/// * `x`           — Clean data point.
/// * `sigma_min`   — Minimum noise level.
/// * `sigma_max`   — Maximum noise level.
/// * `n_levels`    — Number of noise levels.
/// * `rng`         — Mutable LCG state.
pub fn annealed_dsm_loss(
    score_fn: &ScoreNetwork,
    x: &[f32],
    sigma_min: f32,
    sigma_max: f32,
    n_levels: usize,
    rng: &mut u64,
) -> Result<f32> {
    if sigma_min <= 0.0 || sigma_max <= sigma_min {
        return Err(NeuralError::InvalidArgument(format!(
            "annealed_dsm_loss: require 0 < sigma_min ({sigma_min}) < sigma_max ({sigma_max})"
        )));
    }
    let l = n_levels.max(1);
    // Pick random level
    let idx = (lcg_next(rng) >> 33) as usize % l;
    let sigma = if l == 1 {
        sigma_min
    } else {
        let ratio = sigma_max / sigma_min;
        sigma_min * ratio.powf(idx as f32 / (l - 1) as f32)
    };

    let (loss, _) = denoising_score_matching_loss(score_fn, x, sigma, rng)?;
    Ok(loss * sigma * sigma)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_net(d: usize) -> ScoreNetwork {
        ScoreNetwork::new(ScoreNetworkConfig::tiny(d)).expect("score net")
    }

    #[test]
    fn test_score_network_shape() {
        let net = tiny_net(4);
        let x = vec![0.1f32, -0.2, 0.3, -0.4];
        let s = net.score(&x).expect("score");
        assert_eq!(s.len(), 4);
        for &v in &s {
            assert!(v.is_finite(), "score not finite: {v}");
        }
    }

    #[test]
    fn test_score_network_params() {
        let net = tiny_net(4);
        assert!(net.parameter_count() > 0);
    }

    #[test]
    fn test_jvp_shape() {
        let net = tiny_net(4);
        let x = vec![0.1f32; 4];
        let v = vec![1.0f32; 4];
        let jvp = net.jvp(&x, &v).expect("jvp");
        assert_eq!(jvp.len(), 4);
        for &v in &jvp {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_explicit_score_matching_loss() {
        let net = tiny_net(4);
        let x = vec![0.5f32, -0.3, 0.2, 0.8];
        let mut rng: u64 = 42;
        let loss = explicit_score_matching_loss(&net, &x, 4, &mut rng).expect("ESM loss");
        assert!(loss.is_finite(), "ESM loss not finite: {loss}");
    }

    #[test]
    fn test_denoising_score_matching_loss() {
        let net = tiny_net(4);
        let x = vec![0.5f32, -0.3, 0.2, 0.8];
        let mut rng: u64 = 0;
        let (loss, x_noisy) =
            denoising_score_matching_loss(&net, &x, 0.1, &mut rng).expect("DSM loss");
        assert!(loss >= 0.0 && loss.is_finite(), "DSM loss invalid: {loss}");
        assert_eq!(x_noisy.len(), 4);
    }

    #[test]
    fn test_dsm_invalid_sigma() {
        let net = tiny_net(4);
        let x = vec![0.0f32; 4];
        let mut rng: u64 = 0;
        assert!(denoising_score_matching_loss(&net, &x, -0.1, &mut rng).is_err());
        assert!(denoising_score_matching_loss(&net, &x, 0.0, &mut rng).is_err());
    }

    #[test]
    fn test_sliced_score_matching_rademacher() {
        let net = tiny_net(4);
        let x = vec![0.5f32, -0.3, 0.2, 0.8];
        let mut rng: u64 = 99;
        let loss = sliced_score_matching_loss(&net, &x, 4, ProjectionDist::Rademacher, &mut rng)
            .expect("SSM loss");
        assert!(loss.is_finite(), "SSM Rademacher loss not finite: {loss}");
    }

    #[test]
    fn test_sliced_score_matching_gaussian() {
        let net = tiny_net(4);
        let x = vec![0.5f32, -0.3, 0.2, 0.8];
        let mut rng: u64 = 7;
        let loss = sliced_score_matching_loss(&net, &x, 2, ProjectionDist::Gaussian, &mut rng)
            .expect("SSM Gaussian loss");
        assert!(loss.is_finite(), "SSM Gaussian loss not finite: {loss}");
    }

    #[test]
    fn test_ssm_zero_projections_error() {
        let net = tiny_net(4);
        let x = vec![0.0f32; 4];
        let mut rng: u64 = 0;
        assert!(
            sliced_score_matching_loss(&net, &x, 0, ProjectionDist::Rademacher, &mut rng).is_err()
        );
    }

    #[test]
    fn test_annealed_dsm() {
        let net = tiny_net(4);
        let x = vec![0.3f32; 4];
        let mut rng: u64 = 123;
        let loss = annealed_dsm_loss(&net, &x, 0.01, 1.0, 5, &mut rng).expect("annealed DSM");
        assert!(loss.is_finite(), "annealed DSM not finite: {loss}");
    }

    #[test]
    fn test_score_network_dim_mismatch() {
        let net = tiny_net(4);
        let x = vec![0.0f32; 5]; // wrong dim
        assert!(net.score(&x).is_err());
    }
}
