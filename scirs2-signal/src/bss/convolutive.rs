// Convolutive Blind Source Separation
//
// Implements convolutive BSS in the frequency domain using per-bin ICA with
// permutation alignment. Based on:
//
//   Smaragdis (1998). "Blind separation of convolved mixtures in the
//   frequency domain." Neurocomputing, 22(1–3), 21-34.
//
//   Saruwatari et al. (2001). "Blind source separation based on a fast-
//   convergence algorithm combining ICA and beamforming."
//
//   Pedersen et al. (2007). "Convolutive blind source separation methods."
//   Proc. IEEE, 95(6), 1167-1182.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_linalg::svd;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Small complex helpers (avoid importing heavy deps)
// ---------------------------------------------------------------------------

/// Complex matrix-vector multiply: y = M * x for M (n x n) complex, x length n.
fn cmat_vec(m: &[Complex64], n: usize, x: &[Complex64]) -> Vec<Complex64> {
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        for j in 0..n {
            y[i] = y[i] + m[i * n + j] * x[j];
        }
    }
    y
}

/// Complex matrix multiply: C = A * B, all matrices n x n stored row-major.
fn cmat_mul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

/// Complex matrix identity (n x n).
fn cmat_eye(n: usize) -> Vec<Complex64> {
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        m[i * n + i] = Complex64::new(1.0, 0.0);
    }
    m
}

/// Transpose-conjugate (Hermitian) of n x n complex matrix.
fn cmat_herm(a: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut b = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            b[j * n + i] = a[i * n + j].conj();
        }
    }
    b
}

/// Determinant absolute value of 2x2 complex matrix (for whiten check).
fn cmat2_abs_det(a: &[Complex64]) -> f64 {
    (a[0] * a[3] - a[1] * a[2]).norm()
}

/// Invert 2x2 complex matrix; returns None if singular.
fn cmat2_inv(a: &[Complex64]) -> Option<[Complex64; 4]> {
    let det = a[0] * a[3] - a[1] * a[2];
    if det.norm() < 1e-15 {
        return None;
    }
    let inv_det = Complex64::new(1.0, 0.0) / det;
    Some([
        a[3] * inv_det,
        -(a[1]) * inv_det,
        -(a[2]) * inv_det,
        a[0] * inv_det,
    ])
}

/// Naïve complex matrix inversion for n x n via Gauss-Jordan.
/// Returns None if singular.
fn cmat_inv(a: &[Complex64], n: usize) -> Option<Vec<Complex64>> {
    // Build augmented [A | I]
    let mut aug: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = Complex64::new(1.0, 0.0);
    }

    let w = 2 * n;
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * w + col].norm();
        for row in (col + 1)..n {
            let v = aug[row * w + col].norm();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // singular
        }
        if max_row != col {
            for k in 0..w {
                let tmp = aug[col * w + k];
                aug[col * w + k] = aug[max_row * w + k];
                aug[max_row * w + k] = tmp;
            }
        }
        let pivot = aug[col * w + col];
        let inv_pivot = Complex64::new(1.0, 0.0) / pivot;
        for k in 0..w {
            aug[col * w + k] = aug[col * w + k] * inv_pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row * w + col];
                for k in 0..w {
                    let sub = factor * aug[col * w + k];
                    aug[row * w + k] = aug[row * w + k] - sub;
                }
            }
        }
    }

    let mut inv = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * w + n + j];
        }
    }
    Some(inv)
}

// ---------------------------------------------------------------------------
// FFT helpers using scirs2_fft
// ---------------------------------------------------------------------------

/// Compute forward FFT of a real signal, return complex spectrum of length nfft.
fn fft_real(signal: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    let mut padded: Vec<f64> = signal.to_vec();
    padded.resize(nfft, 0.0);
    scirs2_fft::fft(&padded, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {e}")))
}

/// Compute inverse FFT, return real part of output.
fn ifft_complex(spectrum: &[Complex64], nfft: usize) -> SignalResult<Vec<f64>> {
    let result = scirs2_fft::ifft(spectrum, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {e}")))?;
    Ok(result.iter().map(|c| c.re).collect())
}

// ---------------------------------------------------------------------------
// OverlapAdd: frequency-domain signal reconstruction
// ---------------------------------------------------------------------------

/// Overlap-add synthesis parameters.
#[derive(Debug, Clone)]
pub struct OverlapAdd {
    /// STFT frame length (samples)
    pub frame_len: usize,
    /// Hop size (samples)
    pub hop_size: usize,
    /// FFT length
    pub nfft: usize,
}

impl OverlapAdd {
    /// Create an overlap-add processor.
    ///
    /// # Arguments
    ///
    /// * `frame_len` - Analysis window length.
    /// * `hop_size`  - Hop between frames (must be <= frame_len).
    /// * `nfft`      - FFT size (must be >= frame_len).
    ///
    /// # Errors
    ///
    /// Returns error if parameters are inconsistent.
    pub fn new(frame_len: usize, hop_size: usize, nfft: usize) -> SignalResult<Self> {
        if hop_size == 0 || hop_size > frame_len {
            return Err(SignalError::ValueError(format!(
                "hop_size ({hop_size}) must be in [1, frame_len ({frame_len})]"
            )));
        }
        if nfft < frame_len {
            return Err(SignalError::ValueError(format!(
                "nfft ({nfft}) must be >= frame_len ({frame_len})"
            )));
        }
        Ok(Self {
            frame_len,
            hop_size,
            nfft,
        })
    }

    /// Frame a 1-D signal into overlapping windows.
    ///
    /// Returns a matrix of shape `(n_frames, frame_len)`.
    pub fn frame_signal(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let n = signal.len();
        if n == 0 {
            return vec![];
        }
        let mut frames = Vec::new();
        let mut start = 0usize;
        while start + self.frame_len <= n {
            frames.push(signal[start..start + self.frame_len].to_vec());
            start += self.hop_size;
        }
        // Last partial frame
        if start < n && start + self.frame_len > n {
            let mut frame = vec![0.0f64; self.frame_len];
            let remaining = n - start;
            frame[..remaining].copy_from_slice(&signal[start..]);
            frames.push(frame);
        }
        frames
    }

    /// Reconstruct a signal from a sequence of frequency-domain frames using
    /// overlap-add.
    ///
    /// # Arguments
    ///
    /// * `spectra` - Frequency-domain frames: `n_frames` vectors each of length `nfft`.
    /// * `signal_len` - Target output length.
    /// * `window` - The analysis window used to produce `spectra` (via
    ///   [`Self::stft`]), if any. Each output sample is a weighted overlap-add
    ///   of `window[k] * signal[i]` across every frame that touches it, so it
    ///   must be normalised by the *sum of the window's own values* at that
    ///   position -- not simply by how many frames overlap there -- to
    ///   recover the original (unwindowed) amplitude. Pass `None` only when
    ///   `spectra` was produced from unwindowed (rectangular-window) frames.
    ///
    /// # Returns
    ///
    /// Reconstructed time-domain signal of length `signal_len`.
    pub fn synthesise(
        &self,
        spectra: &[Vec<Complex64>],
        signal_len: usize,
        window: Option<&[f64]>,
    ) -> SignalResult<Vec<f64>> {
        if let Some(win) = window {
            if win.len() != self.frame_len {
                return Err(SignalError::DimensionMismatch(format!(
                    "Window length {} != frame_len {}",
                    win.len(),
                    self.frame_len
                )));
            }
        }

        let mut output = vec![0.0f64; signal_len];
        let mut weights = vec![0.0f64; signal_len];

        for (frame_idx, spectrum) in spectra.iter().enumerate() {
            if spectrum.len() != self.nfft {
                return Err(SignalError::DimensionMismatch(format!(
                    "Spectrum {} has length {} but nfft is {}",
                    frame_idx,
                    spectrum.len(),
                    self.nfft
                )));
            }

            let frame_time = ifft_complex(spectrum, self.nfft)?;
            let start = frame_idx * self.hop_size;

            for k in 0..self.frame_len.min(self.nfft) {
                if start + k < signal_len {
                    output[start + k] += frame_time[k];
                    weights[start + k] += window.map_or(1.0, |w| w[k]);
                }
            }
        }

        // Normalise by the local overlap-added window sum (== overlap count
        // when no window / a rectangular window was used).
        for i in 0..signal_len {
            if weights[i].abs() > 1e-12 {
                output[i] /= weights[i];
            }
        }

        Ok(output)
    }

    /// STFT: compute complex spectra for all frames of a real signal.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal.
    /// * `window` - Analysis window of length `frame_len` (Hann, etc.).
    ///
    /// # Returns
    ///
    /// `(n_frames, nfft)` complex spectra.
    pub fn stft(
        &self,
        signal: &[f64],
        window: Option<&[f64]>,
    ) -> SignalResult<Vec<Vec<Complex64>>> {
        let frames = self.frame_signal(signal);
        let mut spectra = Vec::with_capacity(frames.len());

        for frame in &frames {
            let mut windowed = frame.clone();
            if let Some(win) = window {
                if win.len() != self.frame_len {
                    return Err(SignalError::DimensionMismatch(format!(
                        "Window length {} != frame_len {}",
                        win.len(),
                        self.frame_len
                    )));
                }
                for (s, &w) in windowed.iter_mut().zip(win.iter()) {
                    *s *= w;
                }
            }
            let spectrum = fft_real(&windowed, self.nfft)?;
            spectra.push(spectrum);
        }

        Ok(spectra)
    }
}

/// Generate a Hann window of length `n`.
pub fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// PermutationAlignment
// ---------------------------------------------------------------------------

/// Strategy for aligning frequency-bin permutations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlignmentStrategy {
    /// Correlate adjacent frequency bins' demixing matrices.
    AdjacentCorrelation,
    /// Use inter-frequency correlation of source power envelopes.
    PowerCorrelation,
}

/// Permutation alignment for frequency-domain ICA.
///
/// After independent ICA at each frequency bin, the source ordering may differ
/// between bins. This aligns the permutations so each bin's output #k
/// consistently corresponds to the same physical source.
pub struct PermutationAlignment {
    /// Alignment strategy
    pub strategy: AlignmentStrategy,
}

impl Default for PermutationAlignment {
    fn default() -> Self {
        Self {
            strategy: AlignmentStrategy::AdjacentCorrelation,
        }
    }
}

impl PermutationAlignment {
    /// Create with a given strategy.
    pub fn new(strategy: AlignmentStrategy) -> Self {
        Self { strategy }
    }

    /// Align per-bin separated sources.
    ///
    /// # Arguments
    ///
    /// * `separated` - Shape `(n_bins, n_sources, n_frames)`.
    /// * `demix`     - Demixing matrices per bin: `n_bins` flat arrays each of
    ///                 length `n_sources^2` (row-major complex).
    ///
    /// # Returns
    ///
    /// Permuted `separated` and updated `demix` so sources are consistently ordered.
    pub fn align(
        &self,
        separated: &mut Array3<Complex64>,
        demix: &mut Vec<Vec<Complex64>>,
    ) -> SignalResult<Vec<Vec<usize>>> {
        let (n_bins, n_sources, n_frames) = separated.dim();
        if n_bins == 0 || n_sources == 0 || n_frames == 0 {
            return Ok(vec![]);
        }

        let mut permutations: Vec<Vec<usize>> = Vec::with_capacity(n_bins);
        // Bin 0 is reference (identity permutation)
        permutations.push((0..n_sources).collect());

        for bin in 1..n_bins {
            let perm = match self.strategy {
                AlignmentStrategy::AdjacentCorrelation => {
                    self.align_by_demixing_correlation(demix, bin, n_sources)
                }
                AlignmentStrategy::PowerCorrelation => {
                    self.align_by_power_correlation(separated, bin, n_sources, n_frames)
                }
            };

            // Apply permutation to separated[bin] and demix[bin]
            let perm_clone = perm.clone();
            let old_row: Vec<Vec<Complex64>> = (0..n_sources)
                .map(|s| (0..n_frames).map(|t| separated[[bin, s, t]]).collect())
                .collect();

            for (new_s, &old_s) in perm.iter().enumerate() {
                for t in 0..n_frames {
                    separated[[bin, new_s, t]] = old_row[old_s][t];
                }
            }

            // Permute rows of demixing matrix
            if bin < demix.len() {
                let n2 = n_sources * n_sources;
                let old_demix = demix[bin].clone();
                for (new_s, &old_s) in perm_clone.iter().enumerate() {
                    for j in 0..n_sources {
                        demix[bin][new_s * n_sources + j] = old_demix[old_s * n_sources + j];
                    }
                }
            }

            permutations.push(perm_clone);
        }

        Ok(permutations)
    }

    /// Align bin `b` to bin `b-1` by comparing demixing matrix columns.
    fn align_by_demixing_correlation(
        &self,
        demix: &[Vec<Complex64>],
        bin: usize,
        n: usize,
    ) -> Vec<usize> {
        if bin == 0 || bin >= demix.len() || demix[bin - 1].len() < n * n {
            return (0..n).collect();
        }

        let prev = &demix[bin - 1];
        let curr = &demix[bin];

        if curr.len() < n * n {
            return (0..n).collect();
        }

        // Greedy assignment: for each current row, find best match in previous
        let mut used = vec![false; n];
        let mut perm = vec![0usize; n];

        for new_s in 0..n {
            let mut best_match = 0;
            let mut best_score = f64::NEG_INFINITY;

            for old_s in 0..n {
                if used[old_s] {
                    continue;
                }
                // Correlation: sum |curr[new_s, j] * conj(prev[old_s, j])|
                let score: f64 = (0..n)
                    .map(|j| (curr[new_s * n + j] * prev[old_s * n + j].conj()).norm())
                    .sum();
                if score > best_score {
                    best_score = score;
                    best_match = old_s;
                }
            }

            perm[new_s] = best_match;
            used[best_match] = true;
        }

        perm
    }

    /// Align bin `b` by correlating power envelopes across time.
    fn align_by_power_correlation(
        &self,
        separated: &Array3<Complex64>,
        bin: usize,
        n_sources: usize,
        n_frames: usize,
    ) -> Vec<usize> {
        if bin == 0 || n_frames == 0 {
            return (0..n_sources).collect();
        }

        // Compute power envelopes for bin and bin-1
        let power_curr: Vec<Vec<f64>> = (0..n_sources)
            .map(|s| {
                (0..n_frames)
                    .map(|t| separated[[bin, s, t]].norm_sqr())
                    .collect()
            })
            .collect();

        let power_prev: Vec<Vec<f64>> = (0..n_sources)
            .map(|s| {
                (0..n_frames)
                    .map(|t| separated[[bin - 1, s, t]].norm_sqr())
                    .collect()
            })
            .collect();

        // Greedy assignment by power correlation
        let mut used = vec![false; n_sources];
        let mut perm = vec![0usize; n_sources];

        for new_s in 0..n_sources {
            let mut best_match = 0;
            let mut best_score = f64::NEG_INFINITY;

            for old_s in 0..n_sources {
                if used[old_s] {
                    continue;
                }
                let score = correlation_score(&power_curr[new_s], &power_prev[old_s]);
                if score > best_score {
                    best_score = score;
                    best_match = old_s;
                }
            }

            perm[new_s] = best_match;
            used[best_match] = true;
        }

        perm
    }
}

/// Pearson correlation between two real vectors.
fn correlation_score(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean_a: f64 = a[..n].iter().sum::<f64>() / n_f;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n_f;

    let mut num = 0.0f64;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        sa += da * da;
        sb += db * db;
    }

    let denom = (sa * sb).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

// ---------------------------------------------------------------------------
// FrequencyDomainICA: per-bin complex ICA
// ---------------------------------------------------------------------------

/// Per-frequency-bin ICA using a natural gradient update.
///
/// At each frequency bin k, the data X_k is a (n_sources x n_frames) complex
/// matrix. We find a demixing matrix W_k such that Y_k = W_k X_k has
/// statistically independent rows.
///
/// The update rule uses the natural gradient:
///   W ← W + μ (I - φ(Y) Y^H) W
/// where φ is a nonlinearity. For complex-valued sources, we use the
/// complex logistic nonlinearity: φ(y) = y / (1 + |y|²).
pub struct FrequencyDomainICA {
    /// Learning rate for the natural gradient
    pub learning_rate: f64,
    /// Maximum iterations per bin
    pub max_iterations: usize,
    /// Convergence threshold (Frobenius norm of W update)
    pub tolerance: f64,
}

impl Default for FrequencyDomainICA {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iterations: 500,
            tolerance: 1e-4,
        }
    }
}

impl FrequencyDomainICA {
    /// Create with custom parameters.
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance,
        }
    }

    /// Run ICA at a single frequency bin.
    ///
    /// # Arguments
    ///
    /// * `x_bin` - Complex data matrix `(n_sources, n_frames)` for this bin.
    ///
    /// # Returns
    ///
    /// `(demixing_matrix, sources)` where demixing_matrix is a flat row-major
    /// vector of length `n_sources^2` and sources is `(n_sources, n_frames)`.
    pub fn run_bin(
        &self,
        x_bin: &[Vec<Complex64>],
    ) -> SignalResult<(Vec<Complex64>, Vec<Vec<Complex64>>)> {
        let n = x_bin.len();
        if n == 0 {
            return Err(SignalError::ValueError(
                "Empty channel data for ICA bin".to_string(),
            ));
        }
        let t = x_bin[0].len();
        if t == 0 {
            return Ok((cmat_eye(n), vec![vec![]; n]));
        }

        // Whiten the data at this bin
        let (x_white, w_white) = self.whiten_bin(x_bin, n, t)?;

        // Initialise demixing matrix as identity
        let mut w = cmat_eye(n);

        let t_f = t as f64;

        for _iter in 0..self.max_iterations {
            // Compute sources y = W * x_white
            let y: Vec<Vec<Complex64>> = (0..n)
                .map(|i| {
                    let w_row: Vec<Complex64> = (0..n).map(|j| w[i * n + j]).collect();
                    (0..t)
                        .map(|s_idx| {
                            let x_col: Vec<Complex64> =
                                (0..n).map(|ch| x_white[ch][s_idx]).collect();
                            cmat_vec_row(&w_row, &x_col)
                        })
                        .collect()
                })
                .collect();

            // Compute natural gradient update
            // Δ = (I - (1/T) Σ_t φ(y_t) y_t^H) W
            let mut phi_y_yh = vec![Complex64::new(0.0, 0.0); n * n];
            for t_idx in 0..t {
                let y_t: Vec<Complex64> = (0..n).map(|i| y[i][t_idx]).collect();
                let phi_t: Vec<Complex64> = y_t
                    .iter()
                    .map(|&yi| {
                        // Complex-valued logistic: yi / (1 + |yi|^2)
                        let denom = 1.0 + yi.norm_sqr();
                        yi * Complex64::new(1.0 / denom, 0.0)
                    })
                    .collect();

                // Outer product φ(y_t) y_t^H
                for i in 0..n {
                    for j in 0..n {
                        phi_y_yh[i * n + j] = phi_y_yh[i * n + j] + phi_t[i] * y_t[j].conj();
                    }
                }
            }

            let scale = Complex64::new(1.0 / t_f, 0.0);
            let mut inner = cmat_eye(n);
            for i in 0..n * n {
                inner[i] = inner[i] - phi_y_yh[i] * scale;
            }

            // Δ = inner * W
            let delta = cmat_mul(&inner, &w, n);
            let mut w_new = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n * n {
                w_new[i] = w[i] + delta[i] * Complex64::new(self.learning_rate, 0.0);
            }

            // Check convergence
            let norm_delta: f64 = delta.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            w = w_new;
            if norm_delta < self.tolerance {
                break;
            }
        }

        // Final demixing matrix in original space: w_final = W * W_white
        let w_final = cmat_mul(&w, &w_white, n);

        // Apply to original data
        let sources: Vec<Vec<Complex64>> = (0..n)
            .map(|i| {
                let row: Vec<Complex64> = (0..n).map(|j| w_final[i * n + j]).collect();
                (0..t)
                    .map(|s_idx| {
                        let x_col: Vec<Complex64> = (0..n).map(|ch| x_bin[ch][s_idx]).collect();
                        cmat_vec_row(&row, &x_col)
                    })
                    .collect()
            })
            .collect();

        Ok((w_final, sources))
    }

    /// Whiten the data at a single frequency bin.
    ///
    /// Returns (whitened_data, whitening_matrix) where whitened_data and
    /// whitening_matrix are both (n x t) and (n x n) respectively.
    fn whiten_bin(
        &self,
        x: &[Vec<Complex64>],
        n: usize,
        t: usize,
    ) -> SignalResult<(Vec<Vec<Complex64>>, Vec<Complex64>)> {
        let t_f = t as f64;

        // Compute complex covariance C = X X^H / T
        let mut cov_re = vec![0.0f64; n * n];
        let mut cov_im = vec![0.0f64; n * n];
        for t_idx in 0..t {
            for i in 0..n {
                for j in 0..n {
                    let prod = x[i][t_idx] * x[j][t_idx].conj();
                    cov_re[i * n + j] += prod.re;
                    cov_im[i * n + j] += prod.im;
                }
            }
        }
        for i in 0..n * n {
            cov_re[i] /= t_f;
            cov_im[i] /= t_f;
        }

        // For small n, compute eigendecomposition of the real part of the covariance
        // (Hermitian covariance of zero-mean complex data has real diagonal).
        // We approximate by using only the real part.
        // For proper Hermitian EVD, compute W = D^{-1/2} U^H where U diag(D) U^H = C_real.
        let cov_real_arr = Array2::from_shape_fn((n, n), |(i, j)| cov_re[i * n + j]);
        let (eigvals, eigvecs) = scirs2_linalg::eigh(&cov_real_arr.view(), None).map_err(|e| {
            SignalError::ComputationError(format!("Eigendecomposition in whiten_bin failed: {e}"))
        })?;

        // Build whitening matrix W = D^{-1/2} E^T (real approximation)
        let mut w_white = vec![Complex64::new(0.0, 0.0); n * n];
        for (new_i, _) in (0..n).enumerate() {
            // Sort descending eigenvalue order
            let scale = if eigvals[new_i] > 1e-12 {
                1.0 / eigvals[new_i].sqrt()
            } else {
                0.0
            };
            for j in 0..n {
                w_white[new_i * n + j] = Complex64::new(scale * eigvecs[[j, new_i]], 0.0);
            }
        }

        // Apply whitening to data
        let mut x_white: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); t]; n];
        for t_idx in 0..t {
            let x_col: Vec<Complex64> = (0..n).map(|ch| x[ch][t_idx]).collect();
            let y_col = cmat_vec(&w_white, n, &x_col);
            for i in 0..n {
                x_white[i][t_idx] = y_col[i];
            }
        }

        Ok((x_white, w_white))
    }
}

/// Compute dot product of a complex row vector with a complex column vector.
#[inline]
fn cmat_vec_row(row: &[Complex64], col: &[Complex64]) -> Complex64 {
    row.iter()
        .zip(col.iter())
        .fold(Complex64::new(0.0, 0.0), |acc, (&a, &b)| acc + a * b)
}

// ---------------------------------------------------------------------------
// ConvBSS: top-level convolutive BSS
// ---------------------------------------------------------------------------

/// Configuration for convolutive BSS
#[derive(Debug, Clone)]
pub struct ConvBSSConfig {
    /// STFT frame length
    pub frame_len: usize,
    /// Hop size
    pub hop_size: usize,
    /// FFT length (must be >= frame_len)
    pub nfft: usize,
    /// Learning rate for per-bin ICA
    pub ica_learning_rate: f64,
    /// Maximum ICA iterations per bin
    pub ica_max_iterations: usize,
    /// ICA convergence tolerance
    pub ica_tolerance: f64,
    /// Permutation alignment strategy
    pub alignment_strategy: AlignmentStrategy,
    /// Whether to apply a synthesis window
    pub apply_synthesis_window: bool,
}

impl Default for ConvBSSConfig {
    fn default() -> Self {
        let frame_len = 1024;
        let hop_size = frame_len / 2;
        Self {
            frame_len,
            hop_size,
            nfft: frame_len,
            ica_learning_rate: 0.1,
            ica_max_iterations: 300,
            ica_tolerance: 1e-4,
            alignment_strategy: AlignmentStrategy::AdjacentCorrelation,
            apply_synthesis_window: true,
        }
    }
}

/// Result from convolutive BSS
#[derive(Debug, Clone)]
pub struct ConvBSSResult {
    /// Separated source signals: `(n_sources, n_samples)`
    pub sources: Vec<Vec<f64>>,
    /// Per-bin demixing matrices: `n_bins` entries, each of length `n_sources^2`
    pub demixing_matrices: Vec<Vec<Complex64>>,
    /// Permutation used at each frequency bin
    pub permutations: Vec<Vec<usize>>,
    /// Number of active frequency bins processed
    pub n_bins_processed: usize,
}

/// Convolutive Blind Source Separation in the frequency domain.
///
/// Separates sources from convolutive mixtures (reverberant environments)
/// by applying independent ICA to each frequency bin of the STFT, followed
/// by permutation alignment across bins and overlap-add reconstruction.
///
/// ## Algorithm overview
///
/// 1. Compute the STFT of each observed channel.
/// 2. At each frequency bin, treat the T time-frequency points as samples
///    and run complex-valued ICA to find a per-bin demixing matrix.
/// 3. Align permutations across bins using adjacent-bin demixing correlation
///    or power-envelope correlation.
/// 4. Apply the demixing matrices to obtain separated spectrograms.
/// 5. Reconstruct time-domain signals via ISTFT (overlap-add).
///
/// ## Limitations
///
/// This implementation uses a fixed-order complex ICA with a logistic
/// nonlinearity. For best results, the filter length should be much smaller
/// than the frame length (the "narrowband approximation" holds well).
///
/// # Arguments
///
/// * `observations` - Observed mixed signals: slice of `n_channels` signals
///                    each of the same length `n_samples`.
/// * `n_sources`    - Number of sources to separate (must equal n_channels).
/// * `config`       - ConvBSS configuration.
///
/// # Returns
///
/// A [`ConvBSSResult`] with the separated sources and diagnostic information.
///
/// # Errors
///
/// Returns [`SignalError`] on invalid inputs or numerical failures.
///
/// # Example
///
/// ```rust
/// use scirs2_signal::bss::convolutive::{conv_bss, ConvBSSConfig};
///
/// let n_samples = 4096;
/// let obs: Vec<Vec<f64>> = (0..2)
///     .map(|ch| (0..n_samples).map(|t| (t as f64 * 0.01 * (ch + 1) as f64).sin()).collect())
///     .collect();
/// let config = ConvBSSConfig { frame_len: 256, hop_size: 128, nfft: 256, ..Default::default() };
/// let result = conv_bss(&obs, 2, &config).expect("operation should succeed");
/// assert_eq!(result.sources.len(), 2);
/// ```
pub fn conv_bss(
    observations: &[Vec<f64>],
    n_sources: usize,
    config: &ConvBSSConfig,
) -> SignalResult<ConvBSSResult> {
    let n_channels = observations.len();

    if n_channels == 0 {
        return Err(SignalError::ValueError(
            "No observation channels provided".to_string(),
        ));
    }
    if n_sources != n_channels {
        return Err(SignalError::ValueError(format!(
            "n_sources ({n_sources}) must equal n_channels ({n_channels}) for square demixing"
        )));
    }
    let n_samples = observations[0].len();
    if n_samples == 0 {
        return Err(SignalError::ValueError(
            "Observation signals have zero length".to_string(),
        ));
    }
    for (ch, obs) in observations.iter().enumerate() {
        if obs.len() != n_samples {
            return Err(SignalError::DimensionMismatch(format!(
                "Channel {ch} has {} samples but channel 0 has {n_samples}",
                obs.len()
            )));
        }
    }

    let ola = OverlapAdd::new(config.frame_len, config.hop_size, config.nfft)?;
    let win = hann_window(config.frame_len);

    // ----------------------------------------------------------------
    // Step 1: STFT of each observation channel
    // ----------------------------------------------------------------
    let stfts: Vec<Vec<Vec<Complex64>>> = observations
        .iter()
        .map(|ch| ola.stft(ch, Some(&win)))
        .collect::<SignalResult<_>>()?;

    let n_frames = stfts[0].len();
    if n_frames == 0 {
        return Err(SignalError::ValueError(
            "Signal too short for the given frame_len".to_string(),
        ));
    }

    // Only process the non-redundant bins (DC to Nyquist)
    let n_bins = config.nfft / 2 + 1;

    // ----------------------------------------------------------------
    // Step 2: Per-bin ICA
    // ----------------------------------------------------------------
    let bin_ica = FrequencyDomainICA::new(
        config.ica_learning_rate,
        config.ica_max_iterations,
        config.ica_tolerance,
    );

    // separated[bin][source][frame] = Complex64
    let mut separated = Array3::from_elem((n_bins, n_sources, n_frames), Complex64::new(0.0, 0.0));
    let mut demix_matrices: Vec<Vec<Complex64>> = Vec::with_capacity(n_bins);

    for bin in 0..n_bins {
        // x_bin[channel][frame] for this frequency bin
        let x_bin: Vec<Vec<Complex64>> = (0..n_channels)
            .map(|ch| (0..n_frames).map(|t| stfts[ch][t][bin]).collect())
            .collect();

        let (demix, srcs) = bin_ica.run_bin(&x_bin)?;
        demix_matrices.push(demix);

        for (src_idx, src_frames) in srcs.iter().enumerate() {
            for (t, &val) in src_frames.iter().enumerate() {
                separated[[bin, src_idx, t]] = val;
            }
        }
    }

    // ----------------------------------------------------------------
    // Step 3: Permutation alignment
    // ----------------------------------------------------------------
    let aligner = PermutationAlignment::new(config.alignment_strategy);
    let permutations = aligner.align(&mut separated, &mut demix_matrices)?;

    // ----------------------------------------------------------------
    // Step 4: Reconstruct full spectrum (mirror upper half) and ISTFT
    // ----------------------------------------------------------------
    let mut sources_out: Vec<Vec<f64>> = Vec::with_capacity(n_sources);

    for src in 0..n_sources {
        // Build complete spectra (n_frames x nfft) by mirroring
        let mut spectra: Vec<Vec<Complex64>> = Vec::with_capacity(n_frames);

        for t in 0..n_frames {
            let mut spectrum = vec![Complex64::new(0.0, 0.0); config.nfft];
            for bin in 0..n_bins {
                spectrum[bin] = separated[[bin, src, t]];
            }
            // Mirror: spectrum[nfft - bin] = conj(spectrum[bin]) for bin > 0
            for bin in 1..(config.nfft / 2) {
                spectrum[config.nfft - bin] = separated[[bin, src, t]].conj();
            }
            spectra.push(spectrum);
        }

        let signal = ola.synthesise(&spectra, n_samples, Some(&win))?;
        sources_out.push(signal);
    }

    Ok(ConvBSSResult {
        sources: sources_out,
        demixing_matrices: demix_matrices,
        permutations,
        n_bins_processed: n_bins,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: f64, n_samples: usize, fs: f64) -> Vec<f64> {
        (0..n_samples)
            .map(|t| (2.0 * PI * freq * t as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_overlap_add_roundtrip() {
        let n_samples = 1024;
        let signal = sine(440.0, n_samples, 16000.0);
        let ola = OverlapAdd::new(256, 128, 256).expect("failed to create ola");
        let win = hann_window(256);

        let spectra = ola
            .stft(&signal, Some(&win))
            .expect("failed to create spectra");
        let reconstructed = ola
            .synthesise(&spectra, n_samples, Some(&win))
            .expect("failed to create reconstructed");

        // Overlap-add with Hann window (50% overlap) should reconstruct well
        // Check middle portion (avoid boundary effects)
        let start = 256;
        let end = n_samples - 256;
        let max_err = (start..end)
            .map(|i| (signal[i] - reconstructed[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 0.05, "Reconstruction error too large: {max_err}");
    }

    #[test]
    fn test_hann_window() {
        let n = 64;
        let w = hann_window(n);
        assert_eq!(w.len(), n);
        // Should be close to 0 at endpoints
        assert!(w[0].abs() < 0.01);
        // Peak near centre
        let peak = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((peak - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_conv_bss_basic() {
        // Two sinusoidal sources, simple mixing
        let fs = 8000.0;
        let n_samples = 4096;
        let s1 = sine(440.0, n_samples, fs);
        let s2 = sine(880.0, n_samples, fs);

        // Simple instantaneous mixing (convolutive with length-1 filters)
        let x1: Vec<f64> = s1
            .iter()
            .zip(s2.iter())
            .map(|(&a, &b)| 0.7 * a + 0.3 * b)
            .collect();
        let x2: Vec<f64> = s1
            .iter()
            .zip(s2.iter())
            .map(|(&a, &b)| 0.4 * a + 0.6 * b)
            .collect();

        let config = ConvBSSConfig {
            frame_len: 256,
            hop_size: 128,
            nfft: 256,
            ica_max_iterations: 100,
            ..Default::default()
        };

        let result = conv_bss(&[x1, x2], 2, &config).expect("failed to create result");

        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.sources[0].len(), n_samples);
        assert_eq!(result.sources[1].len(), n_samples);
        assert_eq!(result.n_bins_processed, 256 / 2 + 1);
    }

    #[test]
    fn test_permutation_alignment_trivial() {
        let n_bins = 4;
        let n_sources = 2;
        let n_frames = 16;
        let mut separated =
            Array3::from_elem((n_bins, n_sources, n_frames), Complex64::new(1.0, 0.0));
        let mut demix: Vec<Vec<Complex64>> = (0..n_bins).map(|_| cmat_eye(n_sources)).collect();

        let aligner = PermutationAlignment::default();
        let perms = aligner
            .align(&mut separated, &mut demix)
            .expect("failed to create perms");

        // Identity permutation case should return [0,1] everywhere
        for perm in &perms {
            assert_eq!(perm.len(), n_sources);
        }
    }

    #[test]
    fn test_frequency_domain_ica_trivial() {
        // Single source → demixing should be trivial
        let t = 128;
        let x_bin = vec![(0..t)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect::<Vec<_>>()];
        let ica = FrequencyDomainICA::default();
        let (demix, sources) = ica.run_bin(&x_bin).expect("unexpected None or Err");
        assert_eq!(demix.len(), 1);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].len(), t);
    }

    #[test]
    fn test_conv_bss_invalid_source_count() {
        let obs = vec![vec![0.0f64; 100], vec![0.0f64; 100]];
        let config = ConvBSSConfig::default();
        let result = conv_bss(&obs, 3, &config);
        assert!(result.is_err());
    }
}
