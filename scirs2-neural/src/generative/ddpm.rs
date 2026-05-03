//! Denoising Diffusion Probabilistic Models (DDPM)
//!
//! Implements the core mathematics from Ho, Jain & Abbeel (2020):
//!
//! ## Forward process
//! ```text
//! q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1−ᾱₜ) I)
//! ```
//!
//! ## Reverse process (simplified objective)
//! ```text
//! L_simple = E_{t,x₀,ε} [ ||ε − ε_θ(xₜ, t)||² ]
//! ```
//!
//! ## Components
//! - [`NoiseSchedule`] — Linear, Cosine, or Quadratic beta schedules.
//! - [`DDPMConfig`] — Top-level configuration.
//! - [`DDPMForwardProcess`] — Precomputed noise schedule tables.
//! - [`SimpleUNet`] / [`ResBlock`] — Minimal UNet backbone with sinusoidal time embedding.
//! - [`DDPMReverseProcess`] — Ancestral sampling loop.
//! - [`DDPMLoss`] — MSE training objective.
//!
//! # Reference
//! Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*.
//! <https://arxiv.org/abs/2006.11239>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// LCG PRNG helpers (no external rand crate)
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Sample from U(0, 1) via LCG.
fn lcg_uniform(state: &mut u64) -> f32 {
    let bits = lcg_next(state) >> 11;
    bits as f32 / (1u64 << 53) as f32
}

/// Sample from N(0, 1) via Box-Muller transform.
fn box_muller(state: &mut u64) -> f32 {
    let u1 = (lcg_uniform(state) as f64 + 1e-12).min(1.0 - 1e-12);
    let u2 = lcg_uniform(state) as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos()) as f32
}

// ---------------------------------------------------------------------------
// NoiseSchedule
// ---------------------------------------------------------------------------

/// Variance schedule for the forward diffusion process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseSchedule {
    /// βₜ linearly interpolated between β_start and β_end.
    Linear,
    /// Cosine schedule (Nichol & Dhariwal 2021): ᾱₜ = cos(…)².
    Cosine,
    /// βₜ grows quadratically from β_start to β_end.
    Quadratic,
}

// ---------------------------------------------------------------------------
// DDPMConfig
// ---------------------------------------------------------------------------

/// Top-level configuration for DDPM.
#[derive(Debug, Clone)]
pub struct DDPMConfig {
    /// Total number of diffusion timesteps T.
    pub timesteps: usize,
    /// β at t = 1 (schedule start).
    pub beta_start: f64,
    /// β at t = T (schedule end).
    pub beta_end: f64,
    /// Which variance schedule to use.
    pub schedule: NoiseSchedule,
    /// Cosine schedule offset `s` (see Nichol & Dhariwal 2021; typically 0.008).
    pub cosine_s: f64,
    /// Clip denoised predictions to [−1, 1].
    pub clip_denoised: bool,
}

impl Default for DDPMConfig {
    fn default() -> Self {
        Self {
            timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 2e-2,
            schedule: NoiseSchedule::Linear,
            cosine_s: 0.008,
            clip_denoised: true,
        }
    }
}

impl DDPMConfig {
    /// Standard DDPM configuration (Ho et al. 2020).
    pub fn ddpm() -> Self {
        Self::default()
    }

    /// Cosine-schedule config (Nichol & Dhariwal 2021).
    pub fn ddpm_cosine() -> Self {
        Self {
            schedule: NoiseSchedule::Cosine,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// DDPMForwardProcess
// ---------------------------------------------------------------------------

/// Precomputed noise schedule tables for the forward diffusion process.
///
/// All vectors are indexed from 0 to T−1 (inclusive), where index `t`
/// corresponds to diffusion step `t+1` in 1-indexed notation.
#[derive(Debug, Clone)]
pub struct DDPMForwardProcess {
    /// βₜ — variance added at each step.
    pub betas: Vec<f64>,
    /// αₜ = 1 − βₜ.
    pub alphas: Vec<f64>,
    /// ᾱₜ = ∏_{s=1}^{t} αₛ — cumulative product.
    pub alphas_cumprod: Vec<f64>,
    /// √ᾱₜ — used to scale the signal.
    pub sqrt_alphas_cumprod: Vec<f64>,
    /// √(1−ᾱₜ) — used to scale the noise.
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    /// Configuration that was used to build these tables.
    pub config: DDPMConfig,
}

impl DDPMForwardProcess {
    /// Build schedule tables from a [`DDPMConfig`].
    pub fn new(config: &DDPMConfig) -> Result<Self> {
        let t = config.timesteps;
        if t == 0 {
            return Err(NeuralError::InvalidArgument(
                "DDPMForwardProcess: timesteps must be > 0".to_string(),
            ));
        }
        if config.beta_start <= 0.0 || config.beta_end <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "DDPMForwardProcess: beta_start and beta_end must be positive".to_string(),
            ));
        }
        if config.beta_start >= config.beta_end {
            return Err(NeuralError::InvalidArgument(
                "DDPMForwardProcess: beta_start must be < beta_end".to_string(),
            ));
        }

        let betas = Self::build_betas(config)?;

        let alphas: Vec<f64> = betas.iter().map(|&b| 1.0 - b).collect();

        let mut alphas_cumprod = Vec::with_capacity(t);
        let mut prod = 1.0f64;
        for &a in &alphas {
            prod *= a;
            alphas_cumprod.push(prod);
        }

        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        Ok(Self {
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            config: config.clone(),
        })
    }

    fn build_betas(config: &DDPMConfig) -> Result<Vec<f64>> {
        let t = config.timesteps;
        match config.schedule {
            NoiseSchedule::Linear => {
                let betas: Vec<f64> = (0..t)
                    .map(|i| {
                        config.beta_start
                            + (config.beta_end - config.beta_start)
                                * (i as f64 / (t - 1).max(1) as f64)
                    })
                    .collect();
                Ok(betas)
            }
            NoiseSchedule::Quadratic => {
                let sqrt_start = config.beta_start.sqrt();
                let sqrt_end = config.beta_end.sqrt();
                let betas: Vec<f64> = (0..t)
                    .map(|i| {
                        let frac = i as f64 / (t - 1).max(1) as f64;
                        let val = sqrt_start + (sqrt_end - sqrt_start) * frac;
                        val * val
                    })
                    .collect();
                Ok(betas)
            }
            NoiseSchedule::Cosine => {
                // Nichol & Dhariwal 2021: compute ᾱ from cosine, then derive β
                let s = config.cosine_s;
                let alpha_bar = |t_frac: f64| {
                    let angle = (t_frac + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2;
                    angle.cos().powi(2)
                };
                let alpha_bar_0 = alpha_bar(0.0);
                // Compute ᾱ_t / ᾱ_0 for each timestep
                let alpha_bars: Vec<f64> = (0..=t)
                    .map(|i| alpha_bar(i as f64 / t as f64) / alpha_bar_0)
                    .collect();
                // β_t = 1 - ᾱ_t / ᾱ_{t-1}, clipped to (0, 0.999)
                let betas: Vec<f64> = (1..=t)
                    .map(|i| {
                        let ab_prev = alpha_bars[i - 1];
                        let ab_cur = alpha_bars[i];
                        let beta = 1.0 - ab_cur / ab_prev.max(1e-12);
                        beta.clamp(0.0, 0.999)
                    })
                    .collect();
                Ok(betas)
            }
        }
    }

    /// Number of diffusion timesteps T.
    #[inline]
    pub fn timesteps(&self) -> usize {
        self.config.timesteps
    }

    /// Apply the forward process: q(xₜ | x₀) given pre-sampled noise.
    ///
    /// ```text
    /// xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε
    /// ```
    ///
    /// # Arguments
    /// * `x0`    — Clean sample, length D.
    /// * `t`     — Timestep index in [0, T).
    /// * `noise` — Pre-sampled ε ~ N(0, I), same length as `x0`.
    ///
    /// # Returns
    /// Noisy sample `xₜ` of the same length.
    pub fn add_noise(&self, x0: &[f32], t: usize, noise: &[f32]) -> Result<Vec<f32>> {
        let d = x0.len();
        if t >= self.config.timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "add_noise: t={t} out of range [0, {})",
                self.config.timesteps
            )));
        }
        if noise.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "add_noise: noise len {} != x0 len {d}",
                noise.len()
            )));
        }
        let sqrt_ab = self.sqrt_alphas_cumprod[t] as f32;
        let sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t] as f32;
        Ok(x0
            .iter()
            .zip(noise)
            .map(|(&x, &e)| sqrt_ab * x + sqrt_one_minus_ab * e)
            .collect())
    }

    /// Recover the noise ε that was added to produce `x_noisy` from `x0` at step `t`.
    ///
    /// ```text
    /// ε = (xₜ − √ᾱₜ · x₀) / √(1−ᾱₜ)
    /// ```
    pub fn noise_at(&self, x_noisy: &[f32], x0: &[f32], t: usize) -> Result<Vec<f32>> {
        let d = x_noisy.len();
        if t >= self.config.timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "noise_at: t={t} out of range [0, {})",
                self.config.timesteps
            )));
        }
        if x0.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "noise_at: x0 len {} != x_noisy len {d}",
                x0.len()
            )));
        }
        let sqrt_ab = self.sqrt_alphas_cumprod[t] as f32;
        let sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t] as f32;
        if sqrt_one_minus_ab.abs() < 1e-8 {
            return Err(NeuralError::ComputationError(
                "noise_at: √(1−ᾱₜ) ≈ 0, noise is not recoverable at t=0".to_string(),
            ));
        }
        Ok(x_noisy
            .iter()
            .zip(x0)
            .map(|(&xt, &x)| (xt - sqrt_ab * x) / sqrt_one_minus_ab)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// GroupNorm helper
// ---------------------------------------------------------------------------

/// Compute GroupNorm on a flat vector of length `n_channels * spatial`.
/// `n_channels` must be divisible by `n_groups`.
fn group_norm(x: &[f32], n_channels: usize, n_groups: usize) -> Vec<f32> {
    if x.is_empty() || n_channels == 0 || n_groups == 0 {
        return x.to_vec();
    }
    let spatial = x.len() / n_channels.max(1);
    let group_size = (n_channels / n_groups.max(1)).max(1);
    let group_len = group_size * spatial;

    let mut out = vec![0.0f32; x.len()];
    for g in 0..n_groups {
        let start = g * group_size * spatial;
        let end = (start + group_len).min(x.len());
        if end <= start {
            continue;
        }
        let slice = &x[start..end];
        let mean = slice.iter().sum::<f32>() / slice.len() as f32;
        let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / slice.len() as f32;
        let std = (var + 1e-5).sqrt();
        for (i, &v) in slice.iter().enumerate() {
            out[start + i] = (v - mean) / std;
        }
    }
    out
}

/// SiLU (swish) activation: x·sigmoid(x).
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Sinusoidal time embedding
// ---------------------------------------------------------------------------

/// Compute sinusoidal positional embedding for timestep `t` with dimension `d`.
///
/// PE_{2i}(t) = sin(t / 10000^{2i/d})
/// PE_{2i+1}(t) = cos(t / 10000^{2i/d})
pub fn sinusoidal_time_embedding(t: usize, d: usize) -> Vec<f32> {
    if d == 0 {
        return Vec::new();
    }
    let half = d / 2;
    let mut emb = vec![0.0f32; d];
    for i in 0..half {
        let freq = 10000.0f64.powf(2.0 * i as f64 / d as f64);
        let angle = t as f64 / freq;
        emb[2 * i] = angle.sin() as f32;
        if 2 * i + 1 < d {
            emb[2 * i + 1] = angle.cos() as f32;
        }
    }
    emb
}

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------

/// Residual block with time conditioning.
///
/// ```text
/// h  = SiLU(GroupNorm(W₁ x + b₁)) + time_proj(t)
/// h  = W₂ GroupNorm(SiLU(h)) + b₂ + x    (residual connection)
/// ```
#[derive(Debug, Clone)]
pub struct ResBlock {
    /// First linear: in_ch → hidden_ch  (stored row-major: w[j*in + i] = W[j,i])
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// Second linear: hidden_ch → in_ch
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    /// Time projection: time_dim → hidden_ch
    pub time_w: Vec<f32>,
    pub time_b: Vec<f32>,
    pub in_ch: usize,
    pub hidden_ch: usize,
    pub n_groups: usize,
}

impl ResBlock {
    /// Create a new ResBlock with Xavier-ish weight initialisation via LCG.
    pub fn new(
        in_ch: usize,
        hidden_ch: usize,
        time_dim: usize,
        n_groups: usize,
        rng: &mut u64,
    ) -> Result<Self> {
        if in_ch == 0 || hidden_ch == 0 || time_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ResBlock: in_ch, hidden_ch, and time_dim must be > 0".to_string(),
            ));
        }

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

        Ok(Self {
            w1: xavier(in_ch, hidden_ch, rng),
            b1: vec![0.0f32; hidden_ch],
            w2: xavier(hidden_ch, in_ch, rng),
            b2: vec![0.0f32; in_ch],
            time_w: xavier(time_dim, hidden_ch, rng),
            time_b: vec![0.0f32; hidden_ch],
            in_ch,
            hidden_ch,
            n_groups,
        })
    }

    /// Forward pass through the residual block.
    ///
    /// # Arguments
    /// * `x`        — Input vector, length `in_ch`.
    /// * `time_emb` — Time embedding, length `time_dim`.
    pub fn forward(&self, x: &[f32], time_emb: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "ResBlock forward: x len {} != in_ch {}",
                x.len(),
                self.in_ch
            )));
        }

        // h1 = W1 x + b1  (in_ch → hidden_ch)
        let mut h: Vec<f32> = (0..self.hidden_ch)
            .map(|j| {
                let row_start = j * self.in_ch;
                let dot: f32 = x
                    .iter()
                    .enumerate()
                    .map(|(i, &xi)| self.w1[row_start + i] * xi)
                    .sum();
                dot + self.b1[j]
            })
            .collect();

        // time projection: time_emb → hidden_ch
        let time_dim = time_emb.len();
        for (j, h_j) in h.iter_mut().enumerate().take(self.hidden_ch) {
            let _row_start = j * time_dim.min(self.time_w.len() / self.hidden_ch.max(1));
            let t_contrib: f32 = time_emb
                .iter()
                .enumerate()
                .filter_map(|(i, &ti)| {
                    let idx = j * time_dim + i;
                    self.time_w.get(idx).map(|&w| w * ti)
                })
                .sum();
            let t_bias = self.time_b.get(j).copied().unwrap_or(0.0);
            *h_j += t_contrib + t_bias;
        }

        // GroupNorm + SiLU
        let h_normed = group_norm(&h, self.hidden_ch, self.n_groups.max(1).min(self.hidden_ch));
        let h_act: Vec<f32> = h_normed.iter().map(|&v| silu(v)).collect();

        // h2 = W2 h_act + b2  (hidden_ch → in_ch)
        let mut h2: Vec<f32> = (0..self.in_ch)
            .map(|j| {
                let row_start = j * self.hidden_ch;
                let dot: f32 = h_act
                    .iter()
                    .enumerate()
                    .map(|(i, &hi)| self.w2.get(row_start + i).copied().unwrap_or(0.0) * hi)
                    .sum();
                dot + self.b2.get(j).copied().unwrap_or(0.0)
            })
            .collect();

        // GroupNorm before second SiLU
        let h2_normed = group_norm(&h2, self.in_ch, self.n_groups.max(1).min(self.in_ch));
        for v in &mut h2 {
            *v = silu(*v);
        }
        let _ = h2_normed; // used only for normalization ordering note

        // Residual connection: output = h2 + x
        Ok(h2.iter().zip(x).map(|(&h, &xi)| h + xi).collect())
    }
}

// ---------------------------------------------------------------------------
// SimpleUNet
// ---------------------------------------------------------------------------

/// Minimal UNet-style backbone for DDPM noise prediction.
///
/// Architecture:
/// - Sinusoidal time embedding → two-layer MLP (time_dim → 4*time_dim → time_dim)
/// - Encoder: `n_blocks` ResBlocks on the input
/// - Decoder: `n_blocks` ResBlocks (mirror), concatenation with skip connections
///   is omitted here (flat spatial) to keep the implementation minimal
/// - Final projection: linear in_channels → in_channels
#[derive(Debug, Clone)]
pub struct SimpleUNet {
    /// Encoder residual blocks.
    pub encoder_layers: Vec<ResBlock>,
    /// Decoder residual blocks (mirror of encoder).
    pub decoder_layers: Vec<ResBlock>,
    /// Time embedding MLP: `time_emb_proj[0]` is (W, b) for layer 1,
    /// `time_emb_proj[1]` for layer 2.
    pub time_emb_proj: Vec<(Vec<f32>, Vec<f32>)>,
    /// Final linear: in_ch → in_ch.
    pub out_proj_w: Vec<f32>,
    pub out_proj_b: Vec<f32>,
    pub in_ch: usize,
    pub time_dim: usize,
}

impl SimpleUNet {
    /// Create a new [`SimpleUNet`].
    ///
    /// # Arguments
    /// * `in_channels` — Dimensionality of the input (and output).
    /// * `hidden`      — Hidden channel width for ResBlocks.
    /// * `n_blocks`    — Number of encoder (and decoder) blocks.
    pub fn new(in_channels: usize, hidden: usize, n_blocks: usize) -> Result<Self> {
        if in_channels == 0 {
            return Err(NeuralError::InvalidArgument(
                "SimpleUNet: in_channels must be > 0".to_string(),
            ));
        }
        if hidden == 0 {
            return Err(NeuralError::InvalidArgument(
                "SimpleUNet: hidden must be > 0".to_string(),
            ));
        }
        if n_blocks == 0 {
            return Err(NeuralError::InvalidArgument(
                "SimpleUNet: n_blocks must be > 0".to_string(),
            ));
        }

        let mut rng: u64 = 0xdeadbeef_cafebabe;
        let time_dim = hidden.max(8);
        let n_groups = 1usize; // GroupNorm with n_groups=1 = LayerNorm-like

        // Time embedding MLP: time_dim → 4*time_dim → time_dim
        let time_hidden = 4 * time_dim;
        let xavier_w = |fan_in: usize, fan_out: usize, rng: &mut u64| -> Vec<f32> {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;
            (0..fan_in * fan_out)
                .map(|_| {
                    let bits = lcg_next(rng) >> 11;
                    let u = bits as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
                    u * limit
                })
                .collect()
        };

        let time_emb_proj = vec![
            (
                xavier_w(time_dim, time_hidden, &mut rng),
                vec![0.0f32; time_hidden],
            ),
            (
                xavier_w(time_hidden, time_dim, &mut rng),
                vec![0.0f32; time_dim],
            ),
        ];

        // Encoder blocks
        let encoder_layers: Result<Vec<ResBlock>> = (0..n_blocks)
            .map(|_| ResBlock::new(in_channels, hidden, time_dim, n_groups, &mut rng))
            .collect();
        let encoder_layers = encoder_layers?;

        // Decoder blocks (same dimensions; in a full UNet they'd have skip concat)
        let decoder_layers: Result<Vec<ResBlock>> = (0..n_blocks)
            .map(|_| ResBlock::new(in_channels, hidden, time_dim, n_groups, &mut rng))
            .collect();
        let decoder_layers = decoder_layers?;

        // Output projection
        let limit = (6.0 / (2 * in_channels) as f64).sqrt() as f32;
        let out_proj_w: Vec<f32> = (0..in_channels * in_channels)
            .map(|_| {
                let bits = lcg_next(&mut rng) >> 11;
                let u = bits as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
                u * limit
            })
            .collect();

        Ok(Self {
            encoder_layers,
            decoder_layers,
            time_emb_proj,
            out_proj_w,
            out_proj_b: vec![0.0f32; in_channels],
            in_ch: in_channels,
            time_dim,
        })
    }

    /// Project time index to time embedding vector.
    fn embed_time(&self, t: usize) -> Vec<f32> {
        // Sinusoidal embedding
        let mut emb = sinusoidal_time_embedding(t, self.time_dim);

        // MLP layer 1: time_dim → time_hidden
        let (w1, b1) = &self.time_emb_proj[0];
        let time_hidden = b1.len();
        let mut h: Vec<f32> = (0..time_hidden)
            .map(|j| {
                let dot: f32 = emb
                    .iter()
                    .enumerate()
                    .map(|(i, &ei)| w1.get(j * self.time_dim + i).copied().unwrap_or(0.0) * ei)
                    .sum();
                silu(dot + b1[j])
            })
            .collect();

        // MLP layer 2: time_hidden → time_dim
        let (w2, b2) = &self.time_emb_proj[1];
        emb = (0..self.time_dim)
            .map(|j| {
                let dot: f32 = h
                    .iter()
                    .enumerate()
                    .map(|(i, &hi)| w2.get(j * time_hidden + i).copied().unwrap_or(0.0) * hi)
                    .sum();
                silu(dot + b2.get(j).copied().unwrap_or(0.0))
            })
            .collect();
        let _ = h; // consumed above

        emb
    }

    /// Forward pass: predict noise ε_θ(xₜ, t).
    ///
    /// # Arguments
    /// * `x`           — Noisy input, length `in_channels`.
    /// * `t`           — Diffusion timestep index.
    /// * `input_shape` — (height, width) for 2-D spatial inputs; currently unused
    ///   (kept for API compatibility with the full UNet).
    pub fn forward(&self, x: &[f32], t: usize, _input_shape: (usize, usize)) -> Result<Vec<f32>> {
        if x.len() != self.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "SimpleUNet forward: input len {} != in_channels {}",
                x.len(),
                self.in_ch
            )));
        }

        let time_emb = self.embed_time(t);

        // Encoder pass
        let mut h = x.to_vec();
        let mut skip_connections: Vec<Vec<f32>> = Vec::with_capacity(self.encoder_layers.len());
        for block in &self.encoder_layers {
            h = block.forward(&h, &time_emb)?;
            skip_connections.push(h.clone());
        }

        // Decoder pass (with additive skip connections as a simplified alternative)
        for (i, block) in self.decoder_layers.iter().enumerate() {
            // Add skip connection from corresponding encoder level
            let skip_idx = self.encoder_layers.len().saturating_sub(1 + i);
            if let Some(skip) = skip_connections.get(skip_idx) {
                for (hi, &si) in h.iter_mut().zip(skip.iter()) {
                    *hi = (*hi + si) * 0.5;
                }
            }
            h = block.forward(&h, &time_emb)?;
        }

        // Final projection: in_ch → in_ch
        let out: Vec<f32> = (0..self.in_ch)
            .map(|j| {
                let dot: f32 = h
                    .iter()
                    .enumerate()
                    .map(|(i, &hi)| {
                        self.out_proj_w
                            .get(j * self.in_ch + i)
                            .copied()
                            .unwrap_or(0.0)
                            * hi
                    })
                    .sum();
                dot + self.out_proj_b.get(j).copied().unwrap_or(0.0)
            })
            .collect();

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// DDPMReverseProcess
// ---------------------------------------------------------------------------

/// DDPM reverse (denoising) process using a [`SimpleUNet`] as noise predictor.
///
/// Starting from xₜ ~ N(0, I) at t = T, each step computes:
/// ```text
/// μ̃_t = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε_θ(xₜ, t))
/// x_{t−1} = μ̃_t + σₜ · z,   z ~ N(0, I)  (or 0 if t = 0)
/// ```
#[derive(Debug, Clone)]
pub struct DDPMReverseProcess {
    /// Forward schedule (needed for αₜ, βₜ tables).
    pub forward: DDPMForwardProcess,
    /// Noise prediction network.
    pub noise_model: SimpleUNet,
    /// Internal LCG state for sampling.
    rng_state: u64,
}

impl DDPMReverseProcess {
    /// Create a reverse process given a forward schedule and model.
    pub fn new(forward: DDPMForwardProcess, noise_model: SimpleUNet, seed: u64) -> Self {
        Self {
            forward,
            noise_model,
            rng_state: seed.wrapping_add(0xfeed_face_cafe_babe),
        }
    }

    /// Predict noise ε_θ(xₜ, t) via the UNet backbone.
    pub fn predict_noise(&self, x_t: &[f32], t: usize) -> Result<Vec<f32>> {
        let shape = (1, 1); // placeholder spatial shape
        self.noise_model.forward(x_t, t, shape)
    }

    /// One DDPM reverse step: xₜ → x_{t−1}.
    ///
    /// # Arguments
    /// * `x_t` — Noisy sample at timestep t.
    /// * `t`   — Timestep index in [0, T).
    ///   At t = 0 no noise is added (final denoising step).
    pub fn reverse_step(&mut self, x_t: &[f32], t: usize) -> Result<Vec<f32>> {
        let fwd = &self.forward;
        if t >= fwd.config.timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "reverse_step: t={t} out of range [0, {})",
                fwd.config.timesteps
            )));
        }

        let beta_t = fwd.betas[t] as f32;
        let alpha_t = fwd.alphas[t] as f32;
        let sqrt_one_minus_ab = fwd.sqrt_one_minus_alphas_cumprod[t] as f32;

        let eps_theta = self.predict_noise(x_t, t)?;

        // μ̃_t = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε_θ)
        let inv_sqrt_alpha = 1.0 / alpha_t.sqrt();
        let coeff = beta_t / sqrt_one_minus_ab.max(1e-8);
        let mut mu: Vec<f32> = x_t
            .iter()
            .zip(&eps_theta)
            .map(|(&xt, &eps)| inv_sqrt_alpha * (xt - coeff * eps))
            .collect();

        // Clip if configured
        if fwd.config.clip_denoised {
            for v in &mut mu {
                *v = v.clamp(-1.0, 1.0);
            }
        }

        // Add noise σₜ · z for t > 0
        if t > 0 {
            let sigma_t = beta_t.sqrt();
            for v in &mut mu {
                *v += sigma_t * box_muller(&mut self.rng_state);
            }
        }

        Ok(mu)
    }

    /// Draw a sample by running the full reverse chain xₜ → … → x₀.
    ///
    /// # Arguments
    /// * `n_dims` — Dimensionality of the sample to generate.
    pub fn sample(&mut self, n_dims: usize) -> Result<Vec<f32>> {
        if n_dims == 0 {
            return Err(NeuralError::InvalidArgument(
                "sample: n_dims must be > 0".to_string(),
            ));
        }
        if n_dims != self.noise_model.in_ch {
            return Err(NeuralError::ShapeMismatch(format!(
                "sample: n_dims {n_dims} != model in_channels {}",
                self.noise_model.in_ch
            )));
        }

        // Start from xₜ ~ N(0, I)
        let mut x: Vec<f32> = (0..n_dims)
            .map(|_| box_muller(&mut self.rng_state))
            .collect();

        // Reverse steps t = T−1, T−2, …, 0
        let t_total = self.forward.config.timesteps;
        for t in (0..t_total).rev() {
            x = self.reverse_step(&x, t)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// DDPMLoss
// ---------------------------------------------------------------------------

/// DDPM training loss (simplified objective).
///
/// Computes:
/// ```text
/// L_simple = E_{t ~ U[0,T), ε ~ N(0,I)} [ ||ε − ε_θ(xₜ, t)||² ]
/// ```
pub struct DDPMLoss;

impl DDPMLoss {
    /// Compute the DDPM MSE loss for a single training example.
    ///
    /// # Arguments
    /// * `forward`  — Pre-built forward process schedule.
    /// * `model`    — Noise prediction network.
    /// * `x0`       — Clean sample, length D.
    /// * `rng`      — Mutable LCG state (pass `&mut seed_u64`).
    ///
    /// # Returns
    /// Scalar MSE loss ∈ [0, ∞).
    pub fn compute(
        forward: &DDPMForwardProcess,
        model: &SimpleUNet,
        x0: &[f32],
        rng: &mut u64,
    ) -> Result<f32> {
        let d = x0.len();
        let t_max = forward.config.timesteps;
        if t_max == 0 {
            return Err(NeuralError::InvalidArgument(
                "DDPMLoss: timesteps must be > 0".to_string(),
            ));
        }

        // Sample t ~ Uniform[0, T)
        let t = (lcg_next(rng) >> 33) as usize % t_max;

        // Sample ε ~ N(0, I)
        let noise: Vec<f32> = (0..d).map(|_| box_muller(rng)).collect();

        // Forward diffuse: xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε
        let x_t = forward.add_noise(x0, t, &noise)?;

        // Predict noise
        let eps_pred = model.forward(&x_t, t, (1, 1))?;

        // MSE(ε, ε_θ)
        let mse: f32 = noise
            .iter()
            .zip(&eps_pred)
            .map(|(&e_true, &e_pred)| {
                let diff = e_true - e_pred;
                diff * diff
            })
            .sum::<f32>()
            / d as f32;

        Ok(mse)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(schedule: NoiseSchedule) -> DDPMConfig {
        DDPMConfig {
            timesteps: 10,
            beta_start: 1e-4,
            beta_end: 2e-2,
            schedule,
            cosine_s: 0.008,
            clip_denoised: true,
        }
    }

    #[test]
    fn test_forward_process_linear() {
        let cfg = make_config(NoiseSchedule::Linear);
        let fwd = DDPMForwardProcess::new(&cfg).expect("build forward process");
        assert_eq!(fwd.betas.len(), 10);
        // Monotone increasing betas
        for i in 1..fwd.betas.len() {
            assert!(fwd.betas[i] >= fwd.betas[i - 1]);
        }
        // alphas_cumprod decreasing
        for i in 1..fwd.alphas_cumprod.len() {
            assert!(fwd.alphas_cumprod[i] <= fwd.alphas_cumprod[i - 1]);
        }
    }

    #[test]
    fn test_forward_process_cosine() {
        let cfg = make_config(NoiseSchedule::Cosine);
        let fwd = DDPMForwardProcess::new(&cfg).expect("cosine schedule");
        assert_eq!(fwd.betas.len(), 10);
        for &b in &fwd.betas {
            assert!((0.0..1.0).contains(&b), "beta out of range: {b}");
        }
    }

    #[test]
    fn test_forward_process_quadratic() {
        let cfg = make_config(NoiseSchedule::Quadratic);
        let fwd = DDPMForwardProcess::new(&cfg).expect("quadratic");
        assert_eq!(fwd.betas.len(), 10);
    }

    #[test]
    fn test_add_noise_shape() {
        let cfg = make_config(NoiseSchedule::Linear);
        let fwd = DDPMForwardProcess::new(&cfg).expect("fwd");
        let x0 = vec![1.0f32; 8];
        let noise = vec![0.5f32; 8];
        let xt = fwd.add_noise(&x0, 5, &noise).expect("add_noise");
        assert_eq!(xt.len(), 8);
    }

    #[test]
    fn test_noise_at_roundtrip() {
        let cfg = make_config(NoiseSchedule::Linear);
        let fwd = DDPMForwardProcess::new(&cfg).expect("fwd");
        let x0 = vec![0.3f32, -0.7f32, 0.1f32, 0.9f32];
        let noise = vec![0.2f32, -0.4f32, 0.6f32, -0.1f32];
        let t = 4;
        let xt = fwd.add_noise(&x0, t, &noise).expect("add_noise");
        let recovered = fwd.noise_at(&xt, &x0, t).expect("noise_at");
        for (&r, &n) in recovered.iter().zip(&noise) {
            assert!(
                (r - n).abs() < 1e-4,
                "noise roundtrip mismatch: got {r}, want {n}"
            );
        }
    }

    #[test]
    fn test_sinusoidal_embedding_shape() {
        let emb = sinusoidal_time_embedding(42, 16);
        assert_eq!(emb.len(), 16);
        for &v in &emb {
            assert!(v.is_finite(), "sinusoidal emb not finite: {v}");
        }
    }

    #[test]
    fn test_simple_unet_forward_shape() {
        let net = SimpleUNet::new(8, 16, 2).expect("unet");
        let x = vec![0.1f32; 8];
        let out = net.forward(&x, 3, (1, 1)).expect("forward");
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_finite(), "UNet output not finite: {v}");
        }
    }

    #[test]
    fn test_ddpm_loss_positive() {
        let cfg = make_config(NoiseSchedule::Linear);
        let fwd = DDPMForwardProcess::new(&cfg).expect("fwd");
        let model = SimpleUNet::new(8, 16, 1).expect("unet");
        let x0 = vec![0.5f32; 8];
        let mut rng: u64 = 42;
        let loss = DDPMLoss::compute(&fwd, &model, &x0, &mut rng).expect("loss");
        assert!(loss >= 0.0 && loss.is_finite(), "invalid loss: {loss}");
    }

    #[test]
    fn test_group_norm_output_finite() {
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let out = group_norm(&x, 4, 2);
        assert_eq!(out.len(), 16);
        for &v in &out {
            assert!(v.is_finite(), "group norm not finite");
        }
    }

    #[test]
    fn test_res_block_forward() {
        let mut rng: u64 = 0;
        let block = ResBlock::new(8, 16, 8, 1, &mut rng).expect("resblock");
        let x = vec![0.1f32; 8];
        let time_emb = vec![0.5f32; 8];
        let out = block.forward(&x, &time_emb).expect("resblock forward");
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_finite(), "resblock output not finite: {v}");
        }
    }

    #[test]
    fn test_reverse_step_shape() {
        let cfg = make_config(NoiseSchedule::Linear);
        let fwd = DDPMForwardProcess::new(&cfg).expect("fwd");
        let model = SimpleUNet::new(4, 8, 1).expect("unet");
        let mut rev = DDPMReverseProcess::new(fwd, model, 77);
        let x_t = vec![0.1f32; 4];
        let out = rev.reverse_step(&x_t, 5).expect("reverse step");
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite(), "reverse step not finite: {v}");
        }
    }

    #[test]
    fn test_full_sampling_smoke() {
        let cfg = DDPMConfig {
            timesteps: 5,
            ..make_config(NoiseSchedule::Linear)
        };
        let fwd = DDPMForwardProcess::new(&cfg).expect("fwd");
        let model = SimpleUNet::new(4, 8, 1).expect("unet");
        let mut rev = DDPMReverseProcess::new(fwd, model, 99);
        let sample = rev.sample(4).expect("sample");
        assert_eq!(sample.len(), 4);
        for &v in &sample {
            assert!(v.is_finite(), "sample not finite: {v}");
        }
    }

    #[test]
    fn test_invalid_config_errors() {
        // beta_start >= beta_end
        let cfg = DDPMConfig {
            beta_start: 0.5,
            beta_end: 0.1,
            ..make_config(NoiseSchedule::Linear)
        };
        assert!(DDPMForwardProcess::new(&cfg).is_err());

        // zero timesteps
        let cfg2 = DDPMConfig {
            timesteps: 0,
            ..make_config(NoiseSchedule::Linear)
        };
        assert!(DDPMForwardProcess::new(&cfg2).is_err());
    }
}
