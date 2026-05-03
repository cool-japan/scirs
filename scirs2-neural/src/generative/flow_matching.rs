//! Flow Matching for generative modelling
//!
//! Implements Conditional Flow Matching (CFM, Lipman et al. 2022):
//!
//! ## Core ODE
//! ```text
//! dx/dt = v_θ(x, t),   t ∈ [0, 1]
//! x(0) ~ N(0, I),  x(1) ~ p_data
//! ```
//!
//! ## Training objective
//! Given source sample x₀ and target sample x₁, the conditional vector field is:
//! ```text
//! u_t(x₀, x₁) = x₁ − x₀
//! x_t = (1−t)·x₀ + t·x₁       (linear interpolation)
//! ```
//!
//! The flow matching loss regresses v_θ onto this conditional field:
//! ```text
//! L_FM = E_{t~U[0,1], x₀~N(0,I), x₁~p_data} [ ||v_θ(x_t, t) − (x₁ − x₀)||² ]
//! ```
//!
//! ## Generation
//! Starting from x(0) ~ N(0, I), integrate the ODE forward to t=1:
//! ```text
//! x(t+Δt) ≈ x(t) + Δt · v_θ(x(t), t)    (Euler method)
//! ```
//!
//! # References
//! - Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M. & Le, M. (2022).
//!   *Flow Matching for Generative Modeling*. ICLR 2023.
//!   <https://arxiv.org/abs/2210.02747>
//! - Tong, A. et al. (2023). *Improving and Generalizing Flow-Matching*.
//!   <https://arxiv.org/abs/2302.00482>

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

// ---------------------------------------------------------------------------
// FlowMatchingConfig
// ---------------------------------------------------------------------------

/// Configuration for the flow matching objective and ODE sampler.
#[derive(Debug, Clone)]
pub struct FlowMatchingConfig {
    /// Number of Euler integration steps for generation.
    pub timesteps: usize,
    /// Minimum sigma for the noisy interpolant (σ_min in OT-CFM).
    /// Set to 0.0 for exact CFM (no sigma perturbation).
    pub sigma_min: f32,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            timesteps: 100,
            sigma_min: 0.0,
        }
    }
}

impl FlowMatchingConfig {
    /// Standard OT-CFM config.
    pub fn ot_cfm() -> Self {
        Self {
            timesteps: 100,
            sigma_min: 1e-4,
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional flow path
// ---------------------------------------------------------------------------

/// Compute the **linear interpolation path** between source `x0` and target `x1` at time `t`.
///
/// ```text
/// x_t = (1−t)·x₀ + t·x₁
/// ```
///
/// # Arguments
/// * `x0` — Source sample (e.g. from N(0,I)), length D.
/// * `x1` — Target data sample, same length.
/// * `t`  — Interpolation parameter `t ∈ [0, 1]`.
///
/// # Returns
/// Interpolated sample `x_t` of the same length.
pub fn linear_interpolation_path(x0: &[f32], x1: &[f32], t: f32) -> Result<Vec<f32>> {
    if x0.len() != x1.len() {
        return Err(NeuralError::ShapeMismatch(format!(
            "linear_interpolation_path: x0 len {} != x1 len {}",
            x0.len(),
            x1.len()
        )));
    }
    let t = t.clamp(0.0, 1.0);
    Ok(x0
        .iter()
        .zip(x1)
        .map(|(&a, &b)| (1.0 - t) * a + t * b)
        .collect())
}

/// Compute the **conditional vector field** `u_t(x₀, x₁) = x₁ − x₀`.
///
/// This is the target that the velocity network is trained to match.
pub fn conditional_vector_field(x0: &[f32], x1: &[f32]) -> Result<Vec<f32>> {
    if x0.len() != x1.len() {
        return Err(NeuralError::ShapeMismatch(format!(
            "conditional_vector_field: x0 len {} != x1 len {}",
            x0.len(),
            x1.len()
        )));
    }
    Ok(x0.iter().zip(x1).map(|(&a, &b)| b - a).collect())
}

// ---------------------------------------------------------------------------
// VelocityField — MLP velocity network
// ---------------------------------------------------------------------------

/// MLP-based time-conditioned velocity field `v_θ(x, t)`.
///
/// Architecture:
/// - Input: `[x; sin(2π t); cos(2π t)]` — Fourier time embedding improves coverage.
/// - `n_layers` hidden layers with GELU activation.
/// - Output: `data_dim`.
#[derive(Debug, Clone)]
pub struct VelocityField {
    pub data_dim: usize,
    pub time_embed_dim: usize,
    /// Layer weights stored row-major: w[j*in + i] = W[j,i].
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl VelocityField {
    /// Create a new [`VelocityField`].
    ///
    /// # Arguments
    /// * `data_dim`       — Dimensionality of the data.
    /// * `hidden_dim`     — Hidden layer width.
    /// * `n_layers`       — Number of hidden layers.
    /// * `time_embed_dim` — Dimensionality of the Fourier time embedding
    ///   (must be even; if 0, raw [sin(2πt), cos(2πt)] are used).
    /// * `seed`           — LCG seed for weight init.
    pub fn new(
        data_dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        time_embed_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if data_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "VelocityField: data_dim must be > 0".to_string(),
            ));
        }
        if hidden_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "VelocityField: hidden_dim must be > 0".to_string(),
            ));
        }
        if n_layers == 0 {
            return Err(NeuralError::InvalidArgument(
                "VelocityField: n_layers must be > 0".to_string(),
            ));
        }

        // time_embed_dim = 0 → use 2D Fourier (sin, cos)
        let actual_time_dim = if time_embed_dim == 0 {
            2
        } else {
            time_embed_dim
        };
        // Must be even for paired sin/cos
        let actual_time_dim = actual_time_dim + actual_time_dim % 2;

        let in_dim = data_dim + actual_time_dim;

        let mut rng = seed.wrapping_add(0xcafe_babe);
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

        let mut layers: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

        // Input → first hidden
        layers.push((
            xavier(in_dim, hidden_dim, &mut rng),
            vec![0.0f32; hidden_dim],
        ));

        // Hidden → hidden
        for _ in 1..n_layers {
            layers.push((
                xavier(hidden_dim, hidden_dim, &mut rng),
                vec![0.0f32; hidden_dim],
            ));
        }

        // Hidden → output
        layers.push((
            xavier(hidden_dim, data_dim, &mut rng),
            vec![0.0f32; data_dim],
        ));

        Ok(Self {
            data_dim,
            time_embed_dim: actual_time_dim,
            layers,
        })
    }

    /// Compute the Fourier time embedding for scalar `t ∈ [0,1]`.
    fn time_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.time_embed_dim / 2;
        let mut emb = vec![0.0f32; self.time_embed_dim];
        for i in 0..half {
            let freq = (i + 1) as f32;
            let angle = 2.0 * std::f32::consts::PI * freq * t;
            emb[2 * i] = angle.sin();
            if 2 * i + 1 < self.time_embed_dim {
                emb[2 * i + 1] = angle.cos();
            }
        }
        emb
    }

    /// GELU activation.
    #[inline]
    fn gelu(x: f32) -> f32 {
        // Approximate GELU: x * Φ(x) ≈ 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
        const C: f32 = 0.797_884_6_f32; // sqrt(2/pi)
        const A: f32 = 0.044715;
        let cdf = 0.5 * (1.0 + (C * (x + A * x * x * x)).tanh());
        x * cdf
    }

    /// Predict velocity `v_θ(x, t)`.
    ///
    /// # Arguments
    /// * `x` — Current sample, length `data_dim`.
    /// * `t` — Time `t ∈ [0, 1]`.
    pub fn forward(&self, x: &[f32], t: f32) -> Result<Vec<f32>> {
        if x.len() != self.data_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "VelocityField forward: x len {} != data_dim {}",
                x.len(),
                self.data_dim
            )));
        }

        // Build input: [x; time_embedding(t)]
        let t_emb = self.time_embedding(t);
        let mut inp: Vec<f32> = x.to_vec();
        inp.extend_from_slice(&t_emb);

        // MLP forward pass
        let n = self.layers.len();
        let mut h = inp;
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
                    *v = Self::gelu(*v);
                }
            }
            h = next;
        }
        Ok(h)
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|(w, b)| w.len() + b.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Flow matching loss
// ---------------------------------------------------------------------------

/// Compute the **flow matching loss** for a pair (x₀, x₁).
///
/// ```text
/// t ~ U[0,1]
/// x_t = (1−t) x₀ + t x₁
/// u_t = x₁ − x₀
/// L = ||v_θ(x_t, t) − u_t||²
/// ```
///
/// # Arguments
/// * `field` — Velocity network.
/// * `x0`   — Source sample (noise), length D.
/// * `x1`   — Target data sample, same length.
/// * `rng`  — Mutable LCG state (for sampling t).
pub fn flow_matching_loss(
    field: &VelocityField,
    x0: &[f32],
    x1: &[f32],
    rng: &mut u64,
) -> Result<f32> {
    let d = x0.len();
    if x1.len() != d {
        return Err(NeuralError::ShapeMismatch(format!(
            "flow_matching_loss: x0 len {d} != x1 len {}",
            x1.len()
        )));
    }

    // Sample t ~ Uniform[0, 1]
    let t = lcg_uniform(rng);

    // Interpolated sample x_t
    let x_t = linear_interpolation_path(x0, x1, t)?;

    // Target velocity u_t = x1 − x0
    let u_t = conditional_vector_field(x0, x1)?;

    // Predicted velocity
    let v_pred = field.forward(&x_t, t)?;

    // MSE
    let mse: f32 = v_pred
        .iter()
        .zip(&u_t)
        .map(|(&vp, &ut)| {
            let diff = vp - ut;
            diff * diff
        })
        .sum::<f32>()
        / d as f32;

    Ok(mse)
}

// ---------------------------------------------------------------------------
// Euler ODE integration (generation)
// ---------------------------------------------------------------------------

/// Generate a sample by integrating the learned vector field from t=0 to t=1.
///
/// Uses the simple Euler method:
/// ```text
/// x(t + Δt) = x(t) + Δt · v_θ(x(t), t)
/// ```
///
/// # Arguments
/// * `field`    — Trained velocity field.
/// * `x0`       — Starting point (noise sample from N(0,I)), length D.
/// * `n_steps`  — Number of Euler steps. More steps = higher quality.
///
/// # Returns
/// Generated sample `x(1)` of the same length as `x0`.
pub fn euler_integration(field: &VelocityField, x0: &[f32], n_steps: usize) -> Result<Vec<f32>> {
    if n_steps == 0 {
        return Err(NeuralError::InvalidArgument(
            "euler_integration: n_steps must be > 0".to_string(),
        ));
    }
    let dt = 1.0f32 / n_steps as f32;
    let mut x = x0.to_vec();

    for step in 0..n_steps {
        let t = step as f32 * dt;
        let v = field.forward(&x, t)?;
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * vi;
        }
    }
    Ok(x)
}

/// Generate a sample from Gaussian noise using [`euler_integration`].
///
/// Convenience wrapper that samples `x₀ ~ N(0,I)` internally.
///
/// # Arguments
/// * `field`   — Trained velocity field.
/// * `n_dims`  — Dimensionality of the sample to generate.
/// * `n_steps` — Euler steps.
/// * `rng`     — Mutable LCG state for sampling Gaussian noise.
pub fn generate_sample(
    field: &VelocityField,
    n_dims: usize,
    n_steps: usize,
    rng: &mut u64,
) -> Result<Vec<f32>> {
    if n_dims == 0 {
        return Err(NeuralError::InvalidArgument(
            "generate_sample: n_dims must be > 0".to_string(),
        ));
    }
    if n_dims != field.data_dim {
        return Err(NeuralError::ShapeMismatch(format!(
            "generate_sample: n_dims {n_dims} != field.data_dim {}",
            field.data_dim
        )));
    }
    let x0: Vec<f32> = (0..n_dims).map(|_| box_muller(rng)).collect();
    euler_integration(field, &x0, n_steps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation_endpoints() {
        let x0 = vec![0.0f32; 4];
        let x1 = vec![1.0f32; 4];
        let xt_0 = linear_interpolation_path(&x0, &x1, 0.0).expect("t=0");
        let xt_1 = linear_interpolation_path(&x0, &x1, 1.0).expect("t=1");
        for &v in &xt_0 {
            assert!((v - 0.0).abs() < 1e-6);
        }
        for &v in &xt_1 {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_linear_interpolation_midpoint() {
        let x0 = vec![0.0f32; 4];
        let x1 = vec![2.0f32; 4];
        let xt = linear_interpolation_path(&x0, &x1, 0.5).expect("t=0.5");
        for &v in &xt {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0 got {v}");
        }
    }

    #[test]
    fn test_conditional_vector_field() {
        let x0 = vec![1.0f32, 2.0, 3.0];
        let x1 = vec![4.0f32, 6.0, 9.0];
        let u = conditional_vector_field(&x0, &x1).expect("cvf");
        assert_eq!(u, vec![3.0f32, 4.0, 6.0]);
    }

    #[test]
    fn test_velocity_field_shape() {
        let field = VelocityField::new(4, 16, 2, 4, 42).expect("velocity field");
        let x = vec![0.1f32, -0.2, 0.3, -0.4];
        let v = field.forward(&x, 0.5).expect("forward");
        assert_eq!(v.len(), 4);
        for &vi in &v {
            assert!(vi.is_finite(), "velocity not finite: {vi}");
        }
    }

    #[test]
    fn test_velocity_field_zero_time_embed() {
        // time_embed_dim=0 should fall back to 2D
        let field = VelocityField::new(4, 8, 1, 0, 0).expect("vf with 0 time_embed");
        let x = vec![0.0f32; 4];
        let v = field.forward(&x, 0.0).expect("forward");
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_flow_matching_loss_positive() {
        let field = VelocityField::new(4, 8, 2, 4, 7).expect("vf");
        let x0 = vec![0.0f32; 4];
        let x1 = vec![1.0f32; 4];
        let mut rng: u64 = 42;
        let loss = flow_matching_loss(&field, &x0, &x1, &mut rng).expect("fm loss");
        assert!(loss >= 0.0 && loss.is_finite(), "loss invalid: {loss}");
    }

    #[test]
    fn test_euler_integration_shape() {
        let field = VelocityField::new(4, 8, 1, 2, 0).expect("vf");
        let x0 = vec![0.1f32; 4];
        let out = euler_integration(&field, &x0, 5).expect("euler");
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite(), "euler output not finite: {v}");
        }
    }

    #[test]
    fn test_generate_sample_shape() {
        let field = VelocityField::new(4, 8, 1, 2, 0).expect("vf");
        let mut rng: u64 = 1337;
        let sample = generate_sample(&field, 4, 10, &mut rng).expect("generate");
        assert_eq!(sample.len(), 4);
        for &v in &sample {
            assert!(v.is_finite(), "sample not finite: {v}");
        }
    }

    #[test]
    fn test_euler_integration_zero_steps_error() {
        let field = VelocityField::new(4, 8, 1, 2, 0).expect("vf");
        let x0 = vec![0.0f32; 4];
        assert!(euler_integration(&field, &x0, 0).is_err());
    }

    #[test]
    fn test_velocity_field_dim_mismatch() {
        let field = VelocityField::new(4, 8, 1, 2, 0).expect("vf");
        let x = vec![0.0f32; 5]; // wrong dim
        assert!(field.forward(&x, 0.0).is_err());
    }

    #[test]
    fn test_velocity_field_params() {
        let field = VelocityField::new(4, 8, 2, 4, 0).expect("vf");
        assert!(field.parameter_count() > 0);
    }
}
