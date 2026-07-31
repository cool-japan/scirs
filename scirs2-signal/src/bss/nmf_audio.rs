// Non-negative Matrix Factorization for Audio Source Separation
//
// Implements NMF with β-divergence, KL-NMF (IS-NMF), supervised NMF,
// temporal continuity regularisation, and harmonically-constrained NMF.
//
// References:
//   Févotte et al. (2009). "Nonnegative Matrix Factorization with the
//   Itakura-Saito Divergence." Neural Computation, 21(3), 793-830.
//
//   Virtanen (2007). "Monaural Sound Source Separation by Nonnegative
//   Matrix Factorization with Temporal Continuity and Sparseness Criteria."
//   IEEE Trans. Audio Speech Lang. Process., 15(3), 1066-1074.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::{RngExt, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Small epsilon to avoid division-by-zero or log(0).
const EPS: f64 = 1e-10;

/// Clip all entries of a matrix to be >= EPS in-place.
fn clip_matrix(m: &mut Array2<f64>) {
    m.mapv_inplace(|v| v.max(EPS));
}

/// Compute the β-divergence d_β(x || y).
///
/// β = 2 : squared Euclidean, β = 1 : KL, β = 0 : Itakura-Saito
fn beta_divergence(x: f64, y: f64, beta: f64) -> f64 {
    let y_safe = y.max(EPS);
    if (beta - 2.0).abs() < 1e-9 {
        // Euclidean
        0.5 * (x - y_safe).powi(2)
    } else if (beta - 1.0).abs() < 1e-9 {
        // KL
        if x < EPS {
            y_safe
        } else {
            x * (x / y_safe).ln() - x + y_safe
        }
    } else if beta.abs() < 1e-9 {
        // IS
        let r = x / y_safe;
        r - r.ln() - 1.0
    } else {
        // General
        let num = x.powf(beta) / (beta * (beta - 1.0));
        let t2 = y_safe.powf(beta) / beta;
        let t3 = x * y_safe.powf(beta - 1.0) / (beta - 1.0);
        num - t3 + t2
    }
}

/// Total β-divergence between two non-negative matrices V and W*H.
pub fn total_beta_divergence(v: &Array2<f64>, approx: &Array2<f64>, beta: f64) -> f64 {
    let (r, c) = v.dim();
    let mut total = 0.0f64;
    for i in 0..r {
        for j in 0..c {
            total += beta_divergence(v[[i, j]], approx[[i, j]], beta);
        }
    }
    total
}

// ---------------------------------------------------------------------------
// AudioNMF: NMF with β-divergence
// ---------------------------------------------------------------------------

/// Configuration for AudioNMF
#[derive(Debug, Clone)]
pub struct AudioNMFConfig {
    /// β parameter for the divergence (2=Euclidean, 1=KL, 0=IS)
    pub beta: f64,
    /// Number of factorisation components (audio templates)
    pub n_components: usize,
    /// Maximum multiplicative update iterations
    pub max_iterations: usize,
    /// Convergence threshold on relative cost change
    pub tolerance: f64,
    /// L1 sparsity weight on H (activations)
    pub sparsity_h: f64,
    /// L2 regularisation weight on H
    pub l2_h: f64,
    /// L1 sparsity weight on W (templates)
    pub sparsity_w: f64,
    /// Random seed for initialisation
    pub random_seed: Option<u64>,
}

impl Default for AudioNMFConfig {
    fn default() -> Self {
        Self {
            beta: 1.0, // KL divergence by default (good for spectrograms)
            n_components: 8,
            max_iterations: 200,
            tolerance: 1e-4,
            sparsity_h: 0.0,
            l2_h: 0.0,
            sparsity_w: 0.0,
            random_seed: None,
        }
    }
}

/// Result from AudioNMF
#[derive(Debug, Clone)]
pub struct AudioNMFResult {
    /// Template (dictionary) matrix W (n_bins x n_components)
    pub w: Array2<f64>,
    /// Activation matrix H (n_components x n_frames)
    pub h: Array2<f64>,
    /// Approximation W*H (n_bins x n_frames)
    pub approx: Array2<f64>,
    /// Per-iteration cost (β-divergence)
    pub cost_history: Vec<f64>,
    /// Whether convergence criterion was met
    pub converged: bool,
}

/// NMF with β-divergence for audio source separation.
///
/// Factorises a non-negative spectrogram V ≈ W H using multiplicative
/// update rules derived from the β-divergence.
///
/// ## Typical use
///
/// For a magnitude/power spectrogram: use β=0 (Itakura-Saito) for perceptually
/// motivated separation, β=1 (KL) for a good default, or β=2 (Euclidean) for
/// reconstruction MSE minimisation.
///
/// # Arguments
///
/// * `v`      - Non-negative input matrix (n_bins x n_frames), e.g. spectrogram.
/// * `config` - AudioNMF configuration.
///
/// # Returns
///
/// An [`AudioNMFResult`] containing W, H, and diagnostics.
pub fn audio_nmf(v: &Array2<f64>, config: &AudioNMFConfig) -> SignalResult<AudioNMFResult> {
    let (n_bins, n_frames) = v.dim();

    if n_bins == 0 || n_frames == 0 {
        return Err(SignalError::ValueError(
            "Input matrix V must be non-empty".to_string(),
        ));
    }
    if config.n_components == 0 {
        return Err(SignalError::ValueError(
            "n_components must be at least 1".to_string(),
        ));
    }
    if v.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "AudioNMF requires a non-negative input matrix".to_string(),
        ));
    }

    let k = config.n_components;
    let beta = config.beta;

    // Initialise W and H randomly
    let mut rng: scirs2_core::random::rngs::StdRng = match config.random_seed {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed as u64),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::rng()),
    };

    let mut w = Array2::<f64>::zeros((n_bins, k));
    let mut h = Array2::<f64>::zeros((k, n_frames));

    for i in 0..n_bins {
        for j in 0..k {
            w[[i, j]] = rng.random_range(0.01..1.0);
        }
    }
    for i in 0..k {
        for j in 0..n_frames {
            h[[i, j]] = rng.random_range(0.01..1.0);
        }
    }

    // Scale initialisation to match magnitude of V
    let v_mean = v.sum() / (n_bins * n_frames) as f64;
    let wh_mean = w.dot(&h).sum() / (n_bins * n_frames) as f64;
    if wh_mean > EPS {
        let scale = (v_mean / wh_mean).sqrt();
        w.mapv_inplace(|x| x * scale);
        h.mapv_inplace(|x| x * scale);
    }

    let mut cost_history: Vec<f64> = Vec::with_capacity(config.max_iterations);
    let mut converged = false;
    let mut prev_cost = f64::INFINITY;

    for _iter in 0..config.max_iterations {
        let approx = w.dot(&h);

        // Compute cost
        let cost = total_beta_divergence(v, &approx, beta);
        cost_history.push(cost);

        // Convergence check
        let rel = (prev_cost - cost).abs() / (prev_cost.abs() + EPS);
        if rel < config.tolerance && _iter > 0 {
            converged = true;
            break;
        }
        prev_cost = cost;

        // -------- Multiplicative update for H --------
        // Numerator:   W^T (V / (WH)^{2-β})
        // Denominator: W^T (WH)^{β-1}
        {
            let mut num_h = Array2::<f64>::zeros((k, n_frames));
            let mut den_h = Array2::<f64>::zeros((k, n_frames));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx[[f, t]].max(EPS);
                    let num_val = v_ft / a_ft.powf(2.0 - beta);
                    let den_val = a_ft.powf(beta - 1.0);
                    for c in 0..k {
                        num_h[[c, t]] += w[[f, c]] * num_val;
                        den_h[[c, t]] += w[[f, c]] * den_val;
                    }
                }
            }

            // Sparsity and regularisation on H
            for c in 0..k {
                for t in 0..n_frames {
                    let d = den_h[[c, t]] + config.sparsity_h + config.l2_h * h[[c, t]] + EPS;
                    h[[c, t]] *= num_h[[c, t]] / d;
                }
            }
        }
        clip_matrix(&mut h);

        // Recompute approximation after H update
        let approx2 = w.dot(&h);

        // -------- Multiplicative update for W --------
        {
            let mut num_w = Array2::<f64>::zeros((n_bins, k));
            let mut den_w = Array2::<f64>::zeros((n_bins, k));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx2[[f, t]].max(EPS);
                    let num_val = v_ft / a_ft.powf(2.0 - beta);
                    let den_val = a_ft.powf(beta - 1.0);
                    for c in 0..k {
                        num_w[[f, c]] += h[[c, t]] * num_val;
                        den_w[[f, c]] += h[[c, t]] * den_val;
                    }
                }
            }

            for f in 0..n_bins {
                for c in 0..k {
                    let d = den_w[[f, c]] + config.sparsity_w + EPS;
                    w[[f, c]] *= num_w[[f, c]] / d;
                }
            }
        }
        clip_matrix(&mut w);
    }

    let approx = w.dot(&h);

    Ok(AudioNMFResult {
        w,
        h,
        approx,
        cost_history,
        converged,
    })
}

// ---------------------------------------------------------------------------
// KL_NMF: NMF with Kullback-Leibler divergence (IS-NMF variant)
// ---------------------------------------------------------------------------

/// Configuration for KL-NMF (also covers Itakura-Saito NMF with β=0).
#[derive(Debug, Clone)]
pub struct KLNMFConfig {
    /// Number of components
    pub n_components: usize,
    /// True = IS divergence (β=0); False = KL divergence (β=1)
    pub use_is_divergence: bool,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Sparsity weight on activations H
    pub sparsity: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for KLNMFConfig {
    fn default() -> Self {
        Self {
            n_components: 8,
            use_is_divergence: false,
            max_iterations: 300,
            tolerance: 1e-5,
            sparsity: 0.0,
            random_seed: None,
        }
    }
}

/// KL-NMF (or IS-NMF): NMF with Kullback-Leibler or Itakura-Saito divergence.
///
/// Delegates to [`audio_nmf`] with the appropriate β parameter and a clean
/// dedicated API.
///
/// # Arguments
///
/// * `v`      - Non-negative spectrogram (n_bins x n_frames).
/// * `config` - KL-NMF configuration.
///
/// # Returns
///
/// [`AudioNMFResult`] with W, H, and diagnostics.
pub fn kl_nmf(v: &Array2<f64>, config: &KLNMFConfig) -> SignalResult<AudioNMFResult> {
    let beta = if config.use_is_divergence { 0.0 } else { 1.0 };

    let audio_cfg = AudioNMFConfig {
        beta,
        n_components: config.n_components,
        max_iterations: config.max_iterations,
        tolerance: config.tolerance,
        sparsity_h: config.sparsity,
        l2_h: 0.0,
        sparsity_w: 0.0,
        random_seed: config.random_seed,
    };

    audio_nmf(v, &audio_cfg)
}

// ---------------------------------------------------------------------------
// SNMF: Supervised NMF with dictionary learning
// ---------------------------------------------------------------------------

/// Configuration for supervised NMF
#[derive(Debug, Clone)]
pub struct SNMFConfig {
    /// Maximum update iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// β divergence parameter
    pub beta: f64,
    /// Whether to update the dictionary W during inference
    pub update_w: bool,
    /// Sparsity penalty on activations
    pub sparsity_h: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for SNMFConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-4,
            beta: 1.0,
            update_w: false,
            sparsity_h: 0.01,
            random_seed: None,
        }
    }
}

/// Supervised NMF result
#[derive(Debug, Clone)]
pub struct SNMFResult {
    /// Updated (or fixed) dictionary (n_bins x n_components)
    pub dictionary: Array2<f64>,
    /// Activation matrix (n_components x n_frames)
    pub activations: Array2<f64>,
    /// Reconstruction V ≈ dictionary * activations
    pub reconstruction: Array2<f64>,
    /// Cost history
    pub cost_history: Vec<f64>,
    /// Convergence flag
    pub converged: bool,
}

/// Supervised NMF: activate a pre-trained dictionary on new observations.
///
/// Given a fixed (or semi-fixed) dictionary W of shape (n_bins, n_components),
/// finds activations H such that V ≈ W H in the β-divergence sense.
///
/// If `config.update_w = true`, the dictionary is also refined.
///
/// # Arguments
///
/// * `v`          - Observed spectrogram (n_bins x n_frames).
/// * `dictionary` - Pre-trained dictionary W (n_bins x n_components).
/// * `config`     - SNMF configuration.
///
/// # Returns
///
/// An [`SNMFResult`] with activations and optional updated dictionary.
pub fn supervised_nmf(
    v: &Array2<f64>,
    dictionary: &Array2<f64>,
    config: &SNMFConfig,
) -> SignalResult<SNMFResult> {
    let (n_bins, n_frames) = v.dim();
    let (dict_bins, n_components) = dictionary.dim();

    if n_bins == 0 || n_frames == 0 {
        return Err(SignalError::ValueError(
            "Input spectrogram V must be non-empty".to_string(),
        ));
    }
    if dict_bins != n_bins {
        return Err(SignalError::DimensionMismatch(format!(
            "Dictionary has {} frequency bins but V has {}",
            dict_bins, n_bins
        )));
    }
    if n_components == 0 {
        return Err(SignalError::ValueError(
            "Dictionary must have at least one component".to_string(),
        ));
    }
    if v.iter().any(|&x| x < 0.0) || dictionary.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "Supervised NMF requires non-negative inputs".to_string(),
        ));
    }

    let beta = config.beta;

    // Initialise H randomly
    let mut rng: scirs2_core::random::rngs::StdRng = match config.random_seed {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed as u64),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::rng()),
    };

    let mut h = Array2::<f64>::zeros((n_components, n_frames));
    for i in 0..n_components {
        for j in 0..n_frames {
            h[[i, j]] = rng.random_range(0.01..1.0);
        }
    }

    let mut w = dictionary.clone();
    clip_matrix(&mut w);

    let mut cost_history: Vec<f64> = Vec::with_capacity(config.max_iterations);
    let mut converged = false;
    let mut prev_cost = f64::INFINITY;

    for _iter in 0..config.max_iterations {
        let approx = w.dot(&h);
        let cost = total_beta_divergence(v, &approx, beta);
        cost_history.push(cost);

        let rel = (prev_cost - cost).abs() / (prev_cost.abs() + EPS);
        if rel < config.tolerance && _iter > 0 {
            converged = true;
            break;
        }
        prev_cost = cost;

        // Update H
        {
            let mut num_h = Array2::<f64>::zeros((n_components, n_frames));
            let mut den_h = Array2::<f64>::zeros((n_components, n_frames));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx[[f, t]].max(EPS);
                    let nv = v_ft / a_ft.powf(2.0 - beta);
                    let dv = a_ft.powf(beta - 1.0);
                    for c in 0..n_components {
                        num_h[[c, t]] += w[[f, c]] * nv;
                        den_h[[c, t]] += w[[f, c]] * dv;
                    }
                }
            }

            for c in 0..n_components {
                for t in 0..n_frames {
                    let d = den_h[[c, t]] + config.sparsity_h + EPS;
                    h[[c, t]] *= num_h[[c, t]] / d;
                }
            }
        }
        clip_matrix(&mut h);

        // Optionally update W
        if config.update_w {
            let approx2 = w.dot(&h);
            let mut num_w = Array2::<f64>::zeros((n_bins, n_components));
            let mut den_w = Array2::<f64>::zeros((n_bins, n_components));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx2[[f, t]].max(EPS);
                    let nv = v_ft / a_ft.powf(2.0 - beta);
                    let dv = a_ft.powf(beta - 1.0);
                    for c in 0..n_components {
                        num_w[[f, c]] += h[[c, t]] * nv;
                        den_w[[f, c]] += h[[c, t]] * dv;
                    }
                }
            }

            for f in 0..n_bins {
                for c in 0..n_components {
                    w[[f, c]] *= num_w[[f, c]] / (den_w[[f, c]] + EPS);
                }
            }
            clip_matrix(&mut w);
        }
    }

    let reconstruction = w.dot(&h);

    Ok(SNMFResult {
        dictionary: w,
        activations: h,
        reconstruction,
        cost_history,
        converged,
    })
}

// ---------------------------------------------------------------------------
// TemporalContinuity: NMF with temporal smoothness regularisation
// ---------------------------------------------------------------------------

/// Configuration for NMF with temporal continuity regularisation
#[derive(Debug, Clone)]
pub struct TemporalContinuityConfig {
    /// Number of components
    pub n_components: usize,
    /// β divergence parameter
    pub beta: f64,
    /// Temporal continuity weight λ_t
    pub lambda_temporal: f64,
    /// Sparsity weight λ_s
    pub lambda_sparse: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for TemporalContinuityConfig {
    fn default() -> Self {
        Self {
            n_components: 8,
            beta: 1.0,
            lambda_temporal: 1.0,
            lambda_sparse: 0.1,
            max_iterations: 300,
            tolerance: 1e-4,
            random_seed: None,
        }
    }
}

/// Result from temporal-continuity NMF
#[derive(Debug, Clone)]
pub struct TemporalContinuityResult {
    /// Dictionary (n_bins x n_components)
    pub w: Array2<f64>,
    /// Activations (n_components x n_frames)
    pub h: Array2<f64>,
    /// Reconstruction
    pub reconstruction: Array2<f64>,
    /// Cost history
    pub cost_history: Vec<f64>,
    /// Convergence flag
    pub converged: bool,
}

/// NMF with temporal continuity and sparsity regularisation.
///
/// Adds a temporal smoothness penalty to the NMF cost function:
///   `J = D_β(V || WH) + λ_t Σ_{k,t} (H[k,t] - H[k,t-1])^2 + λ_s Σ H[k,t]`
///
/// This encourages activations that change smoothly over time, which is
/// particularly useful for separating sustained musical notes.
///
/// # Arguments
///
/// * `v`      - Non-negative spectrogram (n_bins x n_frames).
/// * `config` - Temporal continuity NMF configuration.
///
/// # Returns
///
/// A [`TemporalContinuityResult`] with W, H, and diagnostics.
pub fn temporal_continuity_nmf(
    v: &Array2<f64>,
    config: &TemporalContinuityConfig,
) -> SignalResult<TemporalContinuityResult> {
    let (n_bins, n_frames) = v.dim();

    if n_bins == 0 || n_frames == 0 {
        return Err(SignalError::ValueError(
            "Input V must be non-empty".to_string(),
        ));
    }
    if config.n_components == 0 {
        return Err(SignalError::ValueError(
            "n_components must be at least 1".to_string(),
        ));
    }
    if v.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "Temporal continuity NMF requires a non-negative input matrix".to_string(),
        ));
    }

    let k = config.n_components;
    let beta = config.beta;
    let lambda_t = config.lambda_temporal;
    let lambda_s = config.lambda_sparse;

    let mut rng: scirs2_core::random::rngs::StdRng = match config.random_seed {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed as u64),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::rng()),
    };

    let mut w = Array2::<f64>::zeros((n_bins, k));
    let mut h = Array2::<f64>::zeros((k, n_frames));

    for i in 0..n_bins {
        for j in 0..k {
            w[[i, j]] = rng.random_range(0.01..1.0);
        }
    }
    for i in 0..k {
        for j in 0..n_frames {
            h[[i, j]] = rng.random_range(0.01..1.0);
        }
    }

    let mut cost_history: Vec<f64> = Vec::with_capacity(config.max_iterations);
    let mut converged = false;
    let mut prev_cost = f64::INFINITY;

    for _iter in 0..config.max_iterations {
        let approx = w.dot(&h);

        // Reconstruction cost
        let rec_cost = total_beta_divergence(v, &approx, beta);

        // Temporal continuity cost
        let mut tc_cost = 0.0f64;
        for c in 0..k {
            for t in 1..n_frames {
                let diff = h[[c, t]] - h[[c, t - 1]];
                tc_cost += diff * diff;
            }
        }

        // Sparsity cost
        let sp_cost: f64 = h.sum();

        let cost = rec_cost + lambda_t * tc_cost + lambda_s * sp_cost;
        cost_history.push(cost);

        let rel = (prev_cost - cost).abs() / (prev_cost.abs() + EPS);
        if rel < config.tolerance && _iter > 0 {
            converged = true;
            break;
        }
        prev_cost = cost;

        // Update H with temporal continuity penalty
        // Standard update + correction for temporal difference penalty:
        //   ∂(temporal)/∂H[k,t] = 2λ_t (H[k,t] - H[k,t-1]) - 2λ_t (H[k,t+1] - H[k,t])
        //                        = 2λ_t (2H[k,t] - H[k,t-1] - H[k,t+1])
        // This is a second-order difference (discrete Laplacian).
        // We split into multiplicative num/den form using the auxiliary variable trick.
        {
            let mut num_h = Array2::<f64>::zeros((k, n_frames));
            let mut den_h = Array2::<f64>::zeros((k, n_frames));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx[[f, t]].max(EPS);
                    let nv = v_ft / a_ft.powf(2.0 - beta);
                    let dv = a_ft.powf(beta - 1.0);
                    for c in 0..k {
                        num_h[[c, t]] += w[[f, c]] * nv;
                        den_h[[c, t]] += w[[f, c]] * dv;
                    }
                }
            }

            // Add temporal continuity gradient split into positive (num) and negative (den) parts
            for c in 0..k {
                for t in 0..n_frames {
                    // Neighbour terms (positive contributions to numerator)
                    let prev = if t > 0 { h[[c, t - 1]] } else { 0.0 };
                    let next = if t + 1 < n_frames { h[[c, t + 1]] } else { 0.0 };

                    // Positive gradient: λ_t * (prev + next) → add to numerator
                    num_h[[c, t]] += lambda_t * 2.0 * (prev + next);
                    // Negative gradient: λ_t * 4 * h[c,t] → add to denominator
                    den_h[[c, t]] += lambda_t * 4.0 * h[[c, t]];

                    // Sparsity (L1)
                    den_h[[c, t]] += lambda_s;

                    let d = den_h[[c, t]].max(EPS);
                    h[[c, t]] *= num_h[[c, t]] / d;
                }
            }
        }
        clip_matrix(&mut h);

        // Update W (standard update)
        let approx2 = w.dot(&h);
        {
            let mut num_w = Array2::<f64>::zeros((n_bins, k));
            let mut den_w = Array2::<f64>::zeros((n_bins, k));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx2[[f, t]].max(EPS);
                    let nv = v_ft / a_ft.powf(2.0 - beta);
                    let dv = a_ft.powf(beta - 1.0);
                    for c in 0..k {
                        num_w[[f, c]] += h[[c, t]] * nv;
                        den_w[[f, c]] += h[[c, t]] * dv;
                    }
                }
            }

            for f in 0..n_bins {
                for c in 0..k {
                    w[[f, c]] *= num_w[[f, c]] / (den_w[[f, c]] + EPS);
                }
            }
        }
        clip_matrix(&mut w);
    }

    let reconstruction = w.dot(&h);

    Ok(TemporalContinuityResult {
        w,
        h,
        reconstruction,
        cost_history,
        converged,
    })
}

// ---------------------------------------------------------------------------
// HarmonicNMF: harmonically-constrained NMF for music
// ---------------------------------------------------------------------------

/// Configuration for harmonically-constrained NMF
#[derive(Debug, Clone)]
pub struct HarmonicNMFConfig {
    /// Number of pitched instruments (harmonic components)
    pub n_harmonic: usize,
    /// Number of percussive (non-harmonic) components
    pub n_percussive: usize,
    /// Number of harmonics per pitch
    pub n_harmonics_per_pitch: usize,
    /// Frequency resolution: Hz per bin
    pub freq_resolution: f64,
    /// Fundamental frequency candidates (Hz). If empty, use MIDI 21-108.
    pub f0_candidates: Vec<f64>,
    /// β divergence parameter
    pub beta: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Sparsity weight on activations
    pub sparsity: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for HarmonicNMFConfig {
    fn default() -> Self {
        Self {
            n_harmonic: 4,
            n_percussive: 2,
            n_harmonics_per_pitch: 6,
            freq_resolution: 10.0, // Hz per bin
            f0_candidates: Vec::new(),
            beta: 1.0,
            max_iterations: 200,
            tolerance: 1e-4,
            sparsity: 0.01,
            random_seed: None,
        }
    }
}

/// Result from harmonic NMF
#[derive(Debug, Clone)]
pub struct HarmonicNMFResult {
    /// Full dictionary W (n_bins x n_components)
    pub w: Array2<f64>,
    /// Activations H (n_components x n_frames)
    pub h: Array2<f64>,
    /// Harmonic component indices in the dictionary
    pub harmonic_indices: Vec<usize>,
    /// Percussive component indices
    pub percussive_indices: Vec<usize>,
    /// Reconstruction
    pub reconstruction: Array2<f64>,
    /// Cost history
    pub cost_history: Vec<f64>,
    /// Convergence flag
    pub converged: bool,
}

/// Build a harmonic template for a given fundamental frequency f0.
///
/// Returns a template vector of length `n_bins` with peaks at harmonics of f0.
/// Each harmonic is a Gaussian bump of width `width_bins`.
fn build_harmonic_template(
    n_bins: usize,
    f0: f64,
    freq_resolution: f64,
    n_harmonics: usize,
    width_bins: f64,
) -> Array1<f64> {
    let mut template = Array1::<f64>::zeros(n_bins);
    for h in 1..=n_harmonics {
        let center_bin = (h as f64 * f0 / freq_resolution) as f64;
        let center = center_bin.round() as usize;
        if center >= n_bins {
            break;
        }
        // Gaussian bump around harmonic
        let half_win = (4.0 * width_bins) as usize;
        let lo = center.saturating_sub(half_win);
        let hi = (center + half_win + 1).min(n_bins);
        for b in lo..hi {
            let diff = b as f64 - center_bin;
            let amplitude = (-0.5 * (diff / width_bins).powi(2)).exp();
            // Amplitude decays with harmonic number
            template[b] += amplitude / (h as f64).sqrt();
        }
    }

    // Normalise to unit L1 norm
    let norm = template.sum();
    if norm > EPS {
        template.mapv_inplace(|x| x / norm);
    }
    template
}

/// Build a percussive template: flat / broadband noise-like shape.
fn build_percussive_template(n_bins: usize, index: usize, n_components: usize) -> Array1<f64> {
    let mut template = Array1::<f64>::ones(n_bins);
    // Add slight variation to differentiate components
    for b in 0..n_bins {
        let phase = PI * (index + 1) as f64 * b as f64 / n_bins as f64;
        template[b] = 1.0 + 0.1 * phase.sin();
    }
    let norm = template.sum();
    if norm > EPS {
        template.mapv_inplace(|x| x / norm);
    }
    template
}

/// Harmonically-constrained NMF for music source separation.
///
/// Constructs a structured dictionary where each harmonic component is
/// parameterised by a fundamental frequency candidate. The harmonic
/// templates are initialised from the f0 candidates and optionally refined
/// while keeping the harmonic structure.
///
/// # Arguments
///
/// * `v`      - Non-negative spectrogram (n_bins x n_frames).
/// * `config` - Harmonic NMF configuration.
///
/// # Returns
///
/// A [`HarmonicNMFResult`] with the full dictionary, activations, and indices.
pub fn harmonic_nmf(
    v: &Array2<f64>,
    config: &HarmonicNMFConfig,
) -> SignalResult<HarmonicNMFResult> {
    let (n_bins, n_frames) = v.dim();
    let n_harmonic = config.n_harmonic;
    let n_percussive = config.n_percussive;
    let n_components = n_harmonic + n_percussive;

    if n_bins == 0 || n_frames == 0 {
        return Err(SignalError::ValueError(
            "Input V must be non-empty".to_string(),
        ));
    }
    if n_components == 0 {
        return Err(SignalError::ValueError(
            "Must have at least one component (harmonic or percussive)".to_string(),
        ));
    }
    if v.iter().any(|&x| x < 0.0) {
        return Err(SignalError::ValueError(
            "Harmonic NMF requires a non-negative input matrix".to_string(),
        ));
    }

    // Build f0 candidates
    let f0_cands: Vec<f64> = if config.f0_candidates.is_empty() {
        // MIDI notes 21 (A0, ~27.5 Hz) to 108 (C8, ~4186 Hz)
        (21..=108u32)
            .map(|midi| 440.0 * 2.0_f64.powf((midi as f64 - 69.0) / 12.0))
            .collect()
    } else {
        config.f0_candidates.clone()
    };

    // Select n_harmonic evenly spaced candidates
    let step = if f0_cands.len() > n_harmonic {
        f0_cands.len() / n_harmonic
    } else {
        1
    };

    let mut w = Array2::<f64>::zeros((n_bins, n_components));
    let harmonic_indices: Vec<usize> = (0..n_harmonic).collect();
    let percussive_indices: Vec<usize> = (n_harmonic..n_components).collect();

    // Fill harmonic templates
    for (col, cand_idx) in harmonic_indices.iter().enumerate() {
        let f0_idx = ((*cand_idx) * step).min(f0_cands.len() - 1);
        let f0 = f0_cands[f0_idx];
        let template = build_harmonic_template(
            n_bins,
            f0,
            config.freq_resolution,
            config.n_harmonics_per_pitch,
            1.5, // width in bins
        );
        for b in 0..n_bins {
            w[[b, col]] = template[b].max(EPS);
        }
    }

    // Fill percussive templates
    for (idx, col) in percussive_indices.iter().enumerate() {
        let template = build_percussive_template(n_bins, idx, n_percussive);
        for b in 0..n_bins {
            w[[b, *col]] = template[b].max(EPS);
        }
    }

    // Random initialise H
    let mut rng: scirs2_core::random::rngs::StdRng = match config.random_seed {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed as u64),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::rng()),
    };

    let mut h = Array2::<f64>::zeros((n_components, n_frames));
    for i in 0..n_components {
        for j in 0..n_frames {
            h[[i, j]] = rng.random_range(0.01..1.0);
        }
    }

    let beta = config.beta;
    let lambda_s = config.sparsity;
    let mut cost_history: Vec<f64> = Vec::with_capacity(config.max_iterations);
    let mut converged = false;
    let mut prev_cost = f64::INFINITY;

    for _iter in 0..config.max_iterations {
        let approx = w.dot(&h);
        let cost = total_beta_divergence(v, &approx, beta) + lambda_s * h.sum();
        cost_history.push(cost);

        let rel = (prev_cost - cost).abs() / (prev_cost.abs() + EPS);
        if rel < config.tolerance && _iter > 0 {
            converged = true;
            break;
        }
        prev_cost = cost;

        // Update H
        {
            let mut num_h = Array2::<f64>::zeros((n_components, n_frames));
            let mut den_h = Array2::<f64>::zeros((n_components, n_frames));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx[[f, t]].max(EPS);
                    let nv = v_ft / a_ft.powf(2.0 - beta);
                    let dv = a_ft.powf(beta - 1.0);
                    for c in 0..n_components {
                        num_h[[c, t]] += w[[f, c]] * nv;
                        den_h[[c, t]] += w[[f, c]] * dv;
                    }
                }
            }

            for c in 0..n_components {
                for t in 0..n_frames {
                    h[[c, t]] *= num_h[[c, t]] / (den_h[[c, t]] + lambda_s + EPS);
                }
            }
        }
        clip_matrix(&mut h);

        // Update W (templates) with normalisation
        let approx2 = w.dot(&h);
        {
            let mut num_w = Array2::<f64>::zeros((n_bins, n_components));
            let mut den_w = Array2::<f64>::zeros((n_bins, n_components));

            for f in 0..n_bins {
                for t in 0..n_frames {
                    let v_ft = v[[f, t]].max(EPS);
                    let a_ft = approx2[[f, t]].max(EPS);
                    let nv = v_ft / a_ft.powf(2.0 - beta);
                    let dv = a_ft.powf(beta - 1.0);
                    for c in 0..n_components {
                        num_w[[f, c]] += h[[c, t]] * nv;
                        den_w[[f, c]] += h[[c, t]] * dv;
                    }
                }
            }

            // Only update percussive components freely;
            // harmonic templates get a soft update with L2 penalty toward initial shape
            for f in 0..n_bins {
                for c in 0..n_harmonic {
                    // Mild update
                    let alpha = 0.1f64;
                    let update = num_w[[f, c]] / (den_w[[f, c]] + EPS);
                    w[[f, c]] = (1.0 - alpha) * w[[f, c]] + alpha * w[[f, c]] * update;
                }
                for c in n_harmonic..n_components {
                    w[[f, c]] *= num_w[[f, c]] / (den_w[[f, c]] + EPS);
                }
            }
        }
        clip_matrix(&mut w);

        // Normalise columns of W and rescale H accordingly
        for c in 0..n_components {
            let col_sum: f64 = (0..n_bins).map(|f| w[[f, c]]).sum();
            if col_sum > EPS {
                for f in 0..n_bins {
                    w[[f, c]] /= col_sum;
                }
                for t in 0..n_frames {
                    h[[c, t]] *= col_sum;
                }
            }
        }
    }

    let reconstruction = w.dot(&h);

    Ok(HarmonicNMFResult {
        w,
        h,
        harmonic_indices,
        percussive_indices,
        reconstruction,
        cost_history,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_spectrogram(n_bins: usize, n_frames: usize) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((n_bins, n_frames));
        for f in 0..n_bins {
            for t in 0..n_frames {
                v[[f, t]] = ((f as f64 * t as f64 / (n_bins * n_frames) as f64) + 0.1).abs();
            }
        }
        v
    }

    #[test]
    fn test_audio_nmf_kl() {
        let v = make_spectrogram(32, 64);
        let config = AudioNMFConfig {
            beta: 1.0,
            n_components: 4,
            max_iterations: 50,
            ..Default::default()
        };
        let result = audio_nmf(&v, &config).expect("failed to create result");
        assert_eq!(result.w.dim(), (32, 4));
        assert_eq!(result.h.dim(), (4, 64));
        assert!(!result.cost_history.is_empty());
        // Cost should not increase dramatically
        let first = result.cost_history[0];
        let last = *result.cost_history.last().expect("failed to create last");
        assert!(
            last <= first * 1.1,
            "Cost should not increase: {} -> {}",
            first,
            last
        );
    }

    #[test]
    fn test_kl_nmf_basic() {
        let v = make_spectrogram(16, 32);
        let config = KLNMFConfig {
            n_components: 3,
            max_iterations: 30,
            ..Default::default()
        };
        let result = kl_nmf(&v, &config).expect("failed to create result");
        assert_eq!(result.w.nrows(), 16);
        assert_eq!(result.h.ncols(), 32);
    }

    #[test]
    fn test_supervised_nmf() {
        let n_bins = 16;
        let n_frames = 32;
        let n_components = 3;

        let v = make_spectrogram(n_bins, n_frames);
        let dict = make_spectrogram(n_bins, n_components);
        let config = SNMFConfig {
            max_iterations: 50,
            ..Default::default()
        };

        let result = supervised_nmf(&v, &dict, &config).expect("failed to create result");
        assert_eq!(result.activations.dim(), (n_components, n_frames));
    }

    #[test]
    fn test_temporal_continuity_nmf() {
        let v = make_spectrogram(16, 64);
        let config = TemporalContinuityConfig {
            n_components: 3,
            lambda_temporal: 0.5,
            lambda_sparse: 0.05,
            max_iterations: 50,
            ..Default::default()
        };
        let result = temporal_continuity_nmf(&v, &config).expect("failed to create result");
        assert_eq!(result.w.dim(), (16, 3));
        assert_eq!(result.h.dim(), (3, 64));
    }

    #[test]
    fn test_harmonic_nmf() {
        let n_bins = 64;
        let n_frames = 32;
        let v = make_spectrogram(n_bins, n_frames);
        let config = HarmonicNMFConfig {
            n_harmonic: 2,
            n_percussive: 1,
            n_harmonics_per_pitch: 4,
            freq_resolution: 50.0,
            max_iterations: 30,
            ..Default::default()
        };
        let result = harmonic_nmf(&v, &config).expect("failed to create result");
        assert_eq!(result.w.dim(), (n_bins, 3));
        assert_eq!(result.h.dim(), (3, n_frames));
        assert_eq!(result.harmonic_indices.len(), 2);
        assert_eq!(result.percussive_indices.len(), 1);
    }

    #[test]
    fn test_beta_divergence_values() {
        // KL: d_1(x || y) = x ln(x/y) - x + y  >= 0
        let d_kl = beta_divergence(2.0, 1.0, 1.0);
        assert!(d_kl >= 0.0);

        // Euclidean: d_2(x || y) = 0.5(x-y)^2
        let d_eu = beta_divergence(3.0, 2.0, 2.0);
        assert!((d_eu - 0.5).abs() < 1e-10);

        // IS: d_0(x || y) = x/y - ln(x/y) - 1 >= 0
        let d_is = beta_divergence(1.0, 1.0, 0.0);
        assert!(d_is.abs() < 1e-10); // exact match => 0
    }
}
