//! Vector Quantized VAE (VQ-VAE)
//!
//! Implements the discrete latent space components from van den Oord et al. 2017:
//!
//! ## Vector Quantization
//! Each encoder output `z_e` is replaced by the nearest codebook entry `e_k`:
//! ```text
//! k* = argmin_k ||z_e − e_k||²
//! z_q = e_{k*}
//! ```
//!
//! ## Straight-Through Gradient Estimator
//! Because the argmin is non-differentiable, gradients are passed through:
//! ```text
//! z_q_sg = z_e + (z_q − z_e).detach()   →   ∂z_q_sg/∂z_e = I
//! ```
//!
//! ## VQ Loss
//! ```text
//! L_VQ   = ||sg[z_e] − z_q||²     (codebook loss — moves codebook towards encoder)
//! L_commit = β · ||z_e − sg[z_q]||²  (commitment loss — moves encoder towards codebook)
//! ```
//!
//! ## EMA Codebook Update
//! Exponential moving average of cluster statistics (no grad needed for codebook):
//! ```text
//! N_k ← γ N_k + (1−γ) n_k
//! m_k ← γ m_k + (1−γ) Σ_{z_e in cluster k} z_e
//! e_k ← m_k / N_k
//! ```
//!
//! # Reference
//! van den Oord, A., Vinyals, O. & Kavukcuoglu, K. (2017).
//! *Neural Discrete Representation Learning*.
//! <https://arxiv.org/abs/1711.00937>

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

// ---------------------------------------------------------------------------
// VQConfig
// ---------------------------------------------------------------------------

/// Configuration for the VQ-VAE discrete bottleneck.
#[derive(Debug, Clone)]
pub struct VQConfig {
    /// Number of codebook entries (K).
    pub n_embeddings: usize,
    /// Dimensionality of each codebook vector.
    pub embedding_dim: usize,
    /// Commitment loss weight β.
    pub commitment_cost: f64,
    /// EMA decay factor γ for codebook updates (0 < γ < 1).
    pub decay: f64,
    /// Laplace smoothing factor for EMA cluster counts.
    pub laplace_eps: f64,
    /// Random seed for codebook initialisation.
    pub seed: u64,
}

impl Default for VQConfig {
    fn default() -> Self {
        Self {
            n_embeddings: 512,
            embedding_dim: 64,
            commitment_cost: 0.25,
            decay: 0.99,
            laplace_eps: 1e-5,
            seed: 42,
        }
    }
}

impl VQConfig {
    /// Construct a VQConfig with common defaults.
    pub fn new(n_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            n_embeddings,
            embedding_dim,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.n_embeddings == 0 {
            return Err(NeuralError::InvalidArgument(
                "VQConfig: n_embeddings must be > 0".to_string(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "VQConfig: embedding_dim must be > 0".to_string(),
            ));
        }
        if self.commitment_cost < 0.0 {
            return Err(NeuralError::InvalidArgument(
                "VQConfig: commitment_cost must be >= 0".to_string(),
            ));
        }
        if self.decay <= 0.0 || self.decay >= 1.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "VQConfig: decay must be in (0, 1), got {}",
                self.decay
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VQEmbedding — the codebook
// ---------------------------------------------------------------------------

/// The discrete codebook for vector quantization.
///
/// Stores `n_embeddings` vectors of dimension `embedding_dim`.
///
/// Codebook updates use exponential moving averages (EMA) which are more
/// stable than straight gradient updates of the codebook.
#[derive(Debug, Clone)]
pub struct VQEmbedding {
    /// Configuration
    pub config: VQConfig,
    /// Codebook entries — shape (n_embeddings, embedding_dim), stored row-major.
    pub codebook: Vec<Vec<f32>>,
    /// EMA cluster usage counts N_k.
    pub usage_counts: Vec<f64>,
    /// EMA cluster sum statistics m_k (same shape as codebook).
    ema_cluster_sum: Vec<Vec<f64>>,
}

impl VQEmbedding {
    /// Initialise a new [`VQEmbedding`] with random codebook vectors.
    ///
    /// Entries are drawn from N(0, 1) scaled by `1/sqrt(embedding_dim)`.
    pub fn new(config: VQConfig) -> Result<Self> {
        config.validate()?;

        let k = config.n_embeddings;
        let d = config.embedding_dim;
        let scale = (d as f64).sqrt().recip() as f32;
        let mut rng = config.seed.wrapping_add(0xfeed_cafe);

        let codebook: Vec<Vec<f32>> = (0..k)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        // Sample from N(0, 1) * scale via LCG + Box-Muller
                        let bits1 = lcg_next(&mut rng) >> 11;
                        let bits2 = lcg_next(&mut rng) >> 11;
                        let u1 = (bits1 as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
                        let u2 = bits2 as f64 / (1u64 << 53) as f64;
                        let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        (n as f32) * scale
                    })
                    .collect()
            })
            .collect();

        // EMA accumulators
        let usage_counts: Vec<f64> = vec![1.0; k]; // start at 1 for Laplace smoothing
        let ema_cluster_sum: Vec<Vec<f64>> = codebook
            .iter()
            .map(|e| e.iter().map(|&v| v as f64).collect())
            .collect();

        Ok(Self {
            config,
            codebook,
            usage_counts,
            ema_cluster_sum,
        })
    }

    /// Squared Euclidean distance between vectors `a` and `b`.
    #[inline]
    fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&ai, &bi)| {
                let d = ai - bi;
                d * d
            })
            .sum()
    }

    /// Find the index of the nearest codebook entry to `z`.
    fn nearest_entry(&self, z: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        for (k, entry) in self.codebook.iter().enumerate() {
            let dist = Self::sq_dist(z, entry);
            if dist < best_dist {
                best_dist = dist;
                best_idx = k;
            }
        }
        best_idx
    }

    /// Quantize a batch of encoder outputs.
    ///
    /// For each vector `z_e` in `z`:
    /// 1. Find nearest codebook entry `k* = argmin_k ||z_e − e_k||²`.
    /// 2. Return `z_q = e_{k*}` and `k*`.
    ///
    /// The returned `z_q_st` applies the **straight-through estimator** convention:
    /// in a differentiable framework, `z_q_st = z_e + (z_q − z_e).stop_gradient()`
    /// so that gradients flow into the encoder unchanged.
    /// Since this module uses plain `Vec<f32>` (no autograd), `z_q_st == z_q`
    /// and the caller is responsible for adding the straight-through correction
    /// when computing gradients.
    ///
    /// # Arguments
    /// * `z` — Slice of encoder outputs, each of length `embedding_dim`.
    ///
    /// # Returns
    /// `(z_q, indices)` where `z_q[i]` is the quantized vector and `indices[i]` is k*.
    pub fn quantize(&self, z: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        if z.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        for (i, zi) in z.iter().enumerate() {
            if zi.len() != self.config.embedding_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "VQEmbedding quantize: z[{i}] len {} != embedding_dim {}",
                    zi.len(),
                    self.config.embedding_dim
                )));
            }
        }

        let mut z_q: Vec<Vec<f32>> = Vec::with_capacity(z.len());
        let mut indices: Vec<usize> = Vec::with_capacity(z.len());

        for z_e in z {
            let k = self.nearest_entry(z_e);
            indices.push(k);
            // Straight-through: z_q_st = z_e + (codebook[k] - z_e) = codebook[k]
            // but semantically z_q_st carries gradient into z_e
            let z_quantized: Vec<f32> = self.codebook[k]
                .iter()
                .zip(z_e)
                .map(|(&ek, &ze)| ze + (ek - ze)) // = ek (with symbolic STE)
                .collect();
            z_q.push(z_quantized);
        }
        Ok((z_q, indices))
    }

    /// Update the codebook via **Exponential Moving Average** (EMA).
    ///
    /// ```text
    /// N_k ← γ N_k + (1−γ) n_k
    /// m_k ← γ m_k + (1−γ) Σ_{z_e in cluster k} z_e
    /// e_k ← m_k / N_k
    /// ```
    ///
    /// # Arguments
    /// * `z_e`     — Encoder outputs used in the forward pass.
    /// * `indices` — Corresponding nearest-neighbour indices from [`quantize`].
    pub fn update_ema(&mut self, z_e: &[Vec<f32>], indices: &[usize]) -> Result<()> {
        if z_e.len() != indices.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "update_ema: z_e len {} != indices len {}",
                z_e.len(),
                indices.len()
            )));
        }
        if z_e.is_empty() {
            return Ok(());
        }

        let gamma = self.config.decay;
        let d = self.config.embedding_dim;
        let k = self.config.n_embeddings;

        // Accumulate per-cluster counts and sums for this mini-batch
        let mut batch_counts: Vec<f64> = vec![0.0; k];
        let mut batch_sums: Vec<Vec<f64>> = vec![vec![0.0; d]; k];

        for (ze, &idx) in z_e.iter().zip(indices) {
            if idx >= k {
                return Err(NeuralError::InvalidArgument(format!(
                    "update_ema: index {idx} out of range [0, {k})"
                )));
            }
            if ze.len() != d {
                return Err(NeuralError::ShapeMismatch(format!(
                    "update_ema: z_e entry len {} != embedding_dim {d}",
                    ze.len()
                )));
            }
            batch_counts[idx] += 1.0;
            for (sum_j, &ze_j) in batch_sums[idx].iter_mut().zip(ze.iter()) {
                *sum_j += ze_j as f64;
            }
        }

        // EMA update
        let eps = self.config.laplace_eps;
        for i in 0..k {
            self.usage_counts[i] = gamma * self.usage_counts[i] + (1.0 - gamma) * batch_counts[i];
            for (j, bsij) in batch_sums[i].iter().enumerate().take(d) {
                self.ema_cluster_sum[i][j] =
                    gamma * self.ema_cluster_sum[i][j] + (1.0 - gamma) * bsij;
            }
            // Update codebook: e_i = m_i / N_i  (with Laplace smoothing)
            let n = self.usage_counts[i] + eps;
            for (j, cb) in self.codebook[i].iter_mut().enumerate().take(d) {
                *cb = (self.ema_cluster_sum[i][j] / n) as f32;
            }
        }
        Ok(())
    }

    /// Codebook perplexity: exp(H), where H is the entropy of the cluster usage distribution.
    ///
    /// High perplexity means the codebook is well-utilised (all clusters used equally).
    /// Low perplexity indicates codebook collapse.
    pub fn perplexity(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 1.0;
        }
        let k = self.config.n_embeddings;
        let mut counts = vec![0.0f64; k];
        for &idx in indices {
            if idx < k {
                counts[idx] += 1.0;
            }
        }
        let n = indices.len() as f64;
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0.0)
            .map(|&c| {
                let p = c / n;
                -p * p.ln()
            })
            .sum();
        entropy.exp()
    }
}

// ---------------------------------------------------------------------------
// VQVAELoss
// ---------------------------------------------------------------------------

/// Loss components for VQ-VAE training.
pub struct VQVAELoss;

impl VQVAELoss {
    /// Compute all VQ-VAE loss terms.
    ///
    /// # Arguments
    /// * `x_recon`     — Reconstructed sample (decoder output), flat slice.
    /// * `x`           — Original input, same length.
    /// * `z_e`         — Encoder outputs (batch), each of length `embedding_dim`.
    /// * `z_q`         — Quantized outputs (batch), same shape as `z_e`.
    /// * `commit_cost` — Commitment loss weight β.
    ///
    /// # Returns
    /// `(recon_loss, vq_loss, commitment_loss)` as `(f32, f32, f32)`.
    ///
    /// - `recon_loss`      = MSE(x_recon, x) — reconstruction quality.
    /// - `vq_loss`         = ||sg[z_e] − z_q||² — moves codebook towards encoder.
    /// - `commitment_loss` = β · ||z_e − sg[z_q]||² — moves encoder towards codebook.
    ///
    /// The total loss is `recon_loss + vq_loss + commitment_loss`.
    /// When using EMA for codebook updates, `vq_loss` is not backpropagated through
    /// the codebook (it has no gradient); only `commitment_loss` trains the encoder.
    pub fn compute(
        x_recon: &[f32],
        x: &[f32],
        z_e: &[Vec<f32>],
        z_q: &[Vec<f32>],
        commit_cost: f32,
    ) -> Result<(f32, f32, f32)> {
        // Reconstruction loss
        if x_recon.len() != x.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "VQVAELoss: x_recon len {} != x len {}",
                x_recon.len(),
                x.len()
            )));
        }
        if z_e.len() != z_q.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "VQVAELoss: z_e batch {} != z_q batch {}",
                z_e.len(),
                z_q.len()
            )));
        }

        let recon_loss = if x.is_empty() {
            0.0f32
        } else {
            x_recon
                .iter()
                .zip(x)
                .map(|(&r, &xi)| {
                    let d = r - xi;
                    d * d
                })
                .sum::<f32>()
                / x.len() as f32
        };

        let n_latents = z_e.len();
        if n_latents == 0 {
            return Ok((recon_loss, 0.0, 0.0));
        }

        let mut vq_sum = 0.0f32;
        let mut commit_sum = 0.0f32;
        let mut total_dims = 0usize;

        for (ze, zq) in z_e.iter().zip(z_q.iter()) {
            if ze.len() != zq.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "VQVAELoss: z_e[i] len {} != z_q[i] len {}",
                    ze.len(),
                    zq.len()
                )));
            }
            for (&ze_j, &zq_j) in ze.iter().zip(zq.iter()) {
                // vq_loss: ||sg[z_e] - z_q||² (codebook updates towards encoder output)
                // stop_gradient on z_e means we treat ze as constant here
                let diff_vq = ze_j - zq_j; // sg[z_e] − z_q
                vq_sum += diff_vq * diff_vq;

                // commitment: ||z_e - sg[z_q]||² (encoder updates towards codebook)
                // stop_gradient on z_q means we treat zq as constant here
                let diff_commit = ze_j - zq_j; // z_e − sg[z_q] = same diff numerically
                commit_sum += diff_commit * diff_commit;
            }
            total_dims += ze.len();
        }

        let scale = if total_dims > 0 {
            1.0 / total_dims as f32
        } else {
            0.0
        };

        let vq_loss = vq_sum * scale;
        let commitment_loss = commit_cost * commit_sum * scale;

        Ok((recon_loss, vq_loss, commitment_loss))
    }
}

// ---------------------------------------------------------------------------
// Utility: codebook lookup
// ---------------------------------------------------------------------------

/// Decode a sequence of codebook indices back into dense vectors.
///
/// This is the decoder half of the quantization: given a sequence of token
/// indices, retrieve the corresponding codebook entries.
///
/// # Arguments
/// * `embedding` — The VQ codebook.
/// * `indices`   — Token indices, each in [0, n_embeddings).
///
/// # Returns
/// Vec of codebook vectors, one per index.
pub fn lookup_codebook(embedding: &VQEmbedding, indices: &[usize]) -> Result<Vec<Vec<f32>>> {
    let k = embedding.config.n_embeddings;
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= k {
            return Err(NeuralError::InvalidArgument(format!(
                "lookup_codebook: index {idx} out of range [0, {k})"
            )));
        }
        out.push(embedding.codebook[idx].clone());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> VQConfig {
        VQConfig {
            n_embeddings: 8,
            embedding_dim: 4,
            commitment_cost: 0.25,
            decay: 0.99,
            laplace_eps: 1e-5,
            seed: 0,
        }
    }

    #[test]
    fn test_vq_embedding_creation() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq embedding");
        assert_eq!(emb.codebook.len(), 8);
        assert_eq!(emb.codebook[0].len(), 4);
        assert_eq!(emb.usage_counts.len(), 8);
    }

    #[test]
    fn test_quantize_shape() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq embedding");
        let z: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let (z_q, indices) = emb.quantize(&z).expect("quantize");
        assert_eq!(z_q.len(), 3);
        assert_eq!(indices.len(), 3);
        for q in &z_q {
            assert_eq!(q.len(), 4);
        }
        for &idx in &indices {
            assert!(idx < 8, "index out of range: {idx}");
        }
    }

    #[test]
    fn test_quantize_empty() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq");
        let (z_q, indices) = emb.quantize(&[]).expect("empty quantize");
        assert!(z_q.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_update_ema_runs() {
        let mut emb = VQEmbedding::new(tiny_config()).expect("vq");
        let z_e: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let (_, indices) = emb.quantize(&z_e).expect("quantize");
        emb.update_ema(&z_e, &indices).expect("ema update");
        // Codebook should still be valid
        for entry in &emb.codebook {
            for &v in entry {
                assert!(v.is_finite(), "codebook entry not finite after EMA update");
            }
        }
    }

    #[test]
    fn test_vqvae_loss_shapes() {
        let d = 4;
        let batch = 3;
        let x_recon = vec![0.1f32; 8];
        let x = vec![0.0f32; 8];
        let z_e: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.5f32; d]).collect();
        let z_q: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.4f32; d]).collect();

        let (recon, vq, commit) =
            VQVAELoss::compute(&x_recon, &x, &z_e, &z_q, 0.25).expect("vqvae loss");
        assert!(
            recon >= 0.0 && recon.is_finite(),
            "recon loss invalid: {recon}"
        );
        assert!(vq >= 0.0 && vq.is_finite(), "vq loss invalid: {vq}");
        assert!(
            commit >= 0.0 && commit.is_finite(),
            "commit loss invalid: {commit}"
        );
    }

    #[test]
    fn test_vqvae_loss_perfect_reconstruction() {
        let x = vec![0.5f32, -0.3, 0.2, 0.8];
        let z_e: Vec<Vec<f32>> = vec![vec![0.1f32; 4]];
        let z_q: Vec<Vec<f32>> = vec![vec![0.1f32; 4]]; // identical → zero quantization loss

        let (recon, vq, commit) =
            VQVAELoss::compute(&x, &x, &z_e, &z_q, 0.25).expect("perfect recon");
        assert!(
            recon.abs() < 1e-6,
            "recon not 0 for perfect reconstruction: {recon}"
        );
        assert!(vq.abs() < 1e-6, "vq not 0 for identical z_e/z_q: {vq}");
        assert!(
            commit.abs() < 1e-6,
            "commit not 0 for identical z_e/z_q: {commit}"
        );
    }

    #[test]
    fn test_lookup_codebook() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq");
        let indices = vec![0usize, 3, 7];
        let vecs = lookup_codebook(&emb, &indices).expect("lookup");
        assert_eq!(vecs.len(), 3);
        for (i, v) in vecs.iter().enumerate() {
            assert_eq!(v.len(), 4, "codebook vec {i} wrong len");
            for (&a, &b) in v.iter().zip(emb.codebook[indices[i]].iter()) {
                assert!((a - b).abs() < 1e-7, "lookup mismatch");
            }
        }
    }

    #[test]
    fn test_lookup_codebook_out_of_range() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq");
        assert!(lookup_codebook(&emb, &[100]).is_err());
    }

    #[test]
    fn test_perplexity_uniform() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq");
        // All 8 entries used once → perplexity = 8
        let indices: Vec<usize> = (0..8).collect();
        let p = emb.perplexity(&indices);
        assert!((p - 8.0).abs() < 1e-4, "expected perplexity 8, got {p}");
    }

    #[test]
    fn test_perplexity_single_cluster() {
        let emb = VQEmbedding::new(tiny_config()).expect("vq");
        // All entries in cluster 0 → perplexity = 1
        let indices: Vec<usize> = vec![0; 8];
        let p = emb.perplexity(&indices);
        assert!((p - 1.0).abs() < 1e-4, "expected perplexity 1, got {p}");
    }

    #[test]
    fn test_vq_config_validation() {
        let mut cfg = tiny_config();
        cfg.n_embeddings = 0;
        assert!(VQEmbedding::new(cfg).is_err());

        let mut cfg = tiny_config();
        cfg.embedding_dim = 0;
        assert!(VQEmbedding::new(cfg).is_err());

        let mut cfg = tiny_config();
        cfg.decay = 1.5;
        assert!(VQEmbedding::new(cfg).is_err());
    }

    #[test]
    fn test_ema_codebook_moves_toward_data() {
        // With decay very small, codebook should move strongly toward batch data
        let cfg = VQConfig {
            n_embeddings: 2,
            embedding_dim: 2,
            decay: 0.01, // nearly instant update
            ..VQConfig::default()
        };
        let mut emb = VQEmbedding::new(cfg).expect("vq");

        // Force all samples into cluster 0 by making its codebook match well
        emb.codebook[0] = vec![10.0f32, 10.0];
        emb.codebook[1] = vec![-10.0f32, -10.0];
        // Update cluster 0's EMA sums
        emb.ema_cluster_sum[0] = vec![10.0f64, 10.0];
        emb.ema_cluster_sum[1] = vec![-10.0f64, -10.0];

        let z_e = vec![vec![10.0f32, 10.0]; 4]; // all similar to cluster 0
        let (_, indices) = emb.quantize(&z_e).expect("quantize");
        for &i in &indices {
            assert_eq!(i, 0, "expected all indices to be 0");
        }
        emb.update_ema(&z_e, &indices).expect("ema");
        // Codebook cluster 0 should now be close to [10, 10]
        assert!(
            (emb.codebook[0][0] - 10.0).abs() < 1.0,
            "codebook[0] should be near 10"
        );
    }
}
