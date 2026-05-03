//! Cross-lingual sentence embedding alignment via orthogonal Procrustes.
//!
//! Provides:
//! - [`procrustes_align`]: compute the optimal orthogonal alignment matrix `W`
//!   such that `X·W ≈ Y` in the least-squares / Frobenius sense.
//! - [`CrossLingualAligner`]: learned alignment that wraps the `W` matrix and
//!   exposes single / batch projection.
//! - [`AlignedEncoder`]: thin wrapper combining a base token-level encoder with
//!   a [`CrossLingualAligner`] to produce target-space embeddings.
//!
//! # Algorithm
//!
//! The orthogonal Procrustes problem is:
//! ```text
//! min_W  ||X·W − Y||_F   s.t.  Wᵀ·W = I
//! ```
//! Solution (Schönemann 1966):
//! 1. Compute `M = Xᵀ · Y`  (shape `[d_src, d_tgt]`)
//! 2. Compute the SVD: `M = U · Σ · Vᵀ`
//! 3. The minimiser is `W = U · Vᵀ`  (shape `[d_src, d_tgt]`)
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::ndarray::{Array2, arr1};
//! use scirs2_text::sentence_embeddings::{CrossLingualAligner, procrustes_align};
//!
//! // Build a trivial 2-D rotation by 90°
//! let x = Array2::<f32>::from_shape_vec((3, 2), vec![
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]).unwrap();
//! // Rotate 90° CCW: (x, y) → (−y, x)
//! let y = Array2::<f32>::from_shape_vec((3, 2), vec![
//!     0.0,  1.0,
//!     -1.0, 0.0,
//!     -1.0, 1.0,
//! ]).unwrap();
//! let aligner = CrossLingualAligner::fit(&x, &y).unwrap();
//! let projected = aligner.transform_batch(&x);
//! // projected ≈ y  (up to floating-point)
//! for i in 0..3 {
//!     for j in 0..2 {
//!         assert!((projected[[i, j]] - y[[i, j]]).abs() < 1e-5);
//!     }
//! }
//! ```

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::TextError;

/// Convenience alias used throughout this module.
pub type AlignResult<T> = Result<T, TextError>;

// ── procrustes_align ──────────────────────────────────────────────────────────

/// Compute the optimal orthogonal alignment matrix `W` such that `X·W ≈ Y`.
///
/// Uses the Schönemann (1966) closed-form solution via SVD:
/// 1. `M = Xᵀ·Y`
/// 2. `M = U·Σ·Vᵀ`  (SVD)
/// 3. `W = U·Vᵀ`
///
/// # Parameters
/// - `x`: source embedding matrix of shape `[n, d_src]`.
/// - `y`: target embedding matrix of shape `[n, d_tgt]`.
///
/// Both matrices must have the same number of rows `n` (parallel samples).
///
/// # Errors
/// Returns [`TextError::InvalidInput`] if shapes are incompatible or the SVD
/// fails.
pub fn procrustes_align(x: &Array2<f32>, y: &Array2<f32>) -> AlignResult<Array2<f32>> {
    let n_x = x.nrows();
    let n_y = y.nrows();
    if n_x != n_y {
        return Err(TextError::InvalidInput(format!(
            "procrustes_align: row counts must match (x={n_x}, y={n_y})"
        )));
    }
    if n_x == 0 {
        return Err(TextError::InvalidInput(
            "procrustes_align: input matrices must not be empty".to_string(),
        ));
    }

    let d_src = x.ncols();
    let d_tgt = y.ncols();

    // M = Xᵀ · Y  →  shape [d_src, d_tgt]
    let xt = x.t(); // [d_src, n]
    let m_f64: Array2<f64> = {
        let mut m = Array2::<f64>::zeros((d_src, d_tgt));
        for i in 0..d_src {
            for j in 0..d_tgt {
                let mut acc = 0.0f64;
                for k in 0..n_x {
                    acc += xt[[i, k]] as f64 * y[[k, j]] as f64;
                }
                m[[i, j]] = acc;
            }
        }
        m
    };

    // SVD(M) = U · Σ · Vᵀ  (using scirs2-linalg)
    let (u, _s, vt) = scirs2_linalg::svd(&m_f64.view(), false, None)
        .map_err(|e| TextError::EmbeddingError(format!("procrustes_align: SVD failed: {e}")))?;

    // W = U · Vᵀ  →  shape [d_src, d_tgt]
    // u: [d_src, k]   vt: [k, d_tgt]   product: [d_src, d_tgt]
    let w_f64: Array2<f64> = u.dot(&vt);

    // Convert back to f32
    let w = w_f64.mapv(|v| v as f32);
    Ok(w)
}

// ── CrossLingualAligner ───────────────────────────────────────────────────────

/// Learned orthogonal alignment between two embedding spaces.
///
/// Fit once from parallel samples (one embedding per language per sentence),
/// then use [`transform`][Self::transform] or [`transform_batch`][Self::transform_batch]
/// to project source-language embeddings into the target space.
#[derive(Debug, Clone)]
pub struct CrossLingualAligner {
    /// Alignment matrix `W` of shape `[d_src, d_tgt]`.
    pub alignment_matrix: Array2<f32>,
    /// Dimensionality of the source embedding space.
    pub d_src: usize,
    /// Dimensionality of the target embedding space.
    pub d_tgt: usize,
}

impl CrossLingualAligner {
    /// Fit an alignment from parallel embedding matrices.
    ///
    /// `src_embeddings[i]` must correspond to the same sentence as
    /// `tgt_embeddings[i]`.  Both matrices have shape `[n, d_x]`.
    pub fn fit(src_embeddings: &Array2<f32>, tgt_embeddings: &Array2<f32>) -> AlignResult<Self> {
        let w = procrustes_align(src_embeddings, tgt_embeddings)?;
        let d_src = src_embeddings.ncols();
        let d_tgt = tgt_embeddings.ncols();
        Ok(CrossLingualAligner {
            alignment_matrix: w,
            d_src,
            d_tgt,
        })
    }

    /// Project a single source-language embedding into the target space.
    ///
    /// `src_embedding` must have length `d_src`.
    pub fn transform(&self, src_embedding: &Array1<f32>) -> Array1<f32> {
        // Promote to [1, d_src], multiply, squeeze back
        let src_2d = src_embedding
            .view()
            .to_shape((1, self.d_src))
            .expect("reshape cannot fail here")
            .to_owned();
        src_2d
            .dot(&self.alignment_matrix)
            .index_axis(Axis(0), 0)
            .to_owned()
    }

    /// Project multiple source embeddings — one per row of the input.
    ///
    /// `src_embeddings` has shape `[m, d_src]`; returns `[m, d_tgt]`.
    pub fn transform_batch(&self, src_embeddings: &Array2<f32>) -> Array2<f32> {
        src_embeddings.dot(&self.alignment_matrix)
    }
}

// ── AlignedEncoder ────────────────────────────────────────────────────────────

/// Wraps a source-language token encoder together with a cross-lingual
/// alignment projection.
///
/// The generic parameter `F` is a callable `Fn(&[usize]) -> Array1<f32>`,
/// which avoids `dyn` dispatch overhead.
pub struct AlignedEncoder<'a, F>
where
    F: Fn(&[usize]) -> Array1<f32>,
{
    /// The base encoder function: maps token-ID slices to source-space vectors.
    pub base_encoder: &'a F,
    /// The learned alignment.
    pub aligner: &'a CrossLingualAligner,
    /// When `true`, L2-normalise the projected vector to unit length.
    pub normalize_output: bool,
}

impl<'a, F> AlignedEncoder<'a, F>
where
    F: Fn(&[usize]) -> Array1<f32>,
{
    /// Construct a new `AlignedEncoder`.
    pub fn new(
        base_encoder: &'a F,
        aligner: &'a CrossLingualAligner,
        normalize_output: bool,
    ) -> Self {
        AlignedEncoder {
            base_encoder,
            aligner,
            normalize_output,
        }
    }

    /// Encode `tokens` in the source language and project into the target
    /// embedding space.
    pub fn encode(&self, tokens: &[usize]) -> Array1<f32> {
        let base = (self.base_encoder)(tokens);
        let aligned = self.aligner.transform(&base);
        if self.normalize_output {
            let norm: f32 = aligned.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                aligned.mapv(|x| x / norm)
            } else {
                aligned
            }
        } else {
            aligned
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn lcg_f32(seed: u64, offset: u64) -> f32 {
        const A: u64 = 6_364_136_223_846_793_005;
        const C: u64 = 1_442_695_040_888_963_407;
        let state = A.wrapping_mul(seed.wrapping_add(offset)).wrapping_add(C);
        (((state >> 12) as f64) / ((1u64 << 52) as f64)) as f32 * 2.0 - 1.0
    }

    fn rand_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| lcg_f32(seed, (i * cols + j) as u64))
    }

    // Apply a 2-D rotation of `angle` radians to the columns of `x`.
    // `x` must have exactly 2 columns.
    fn rotate_2d(x: &Array2<f32>, angle: f32) -> Array2<f32> {
        let (cos, sin) = (angle.cos(), angle.sin());
        Array2::from_shape_fn((x.nrows(), 2), |(i, j)| {
            if j == 0 {
                x[[i, 0]] * cos - x[[i, 1]] * sin
            } else {
                x[[i, 0]] * sin + x[[i, 1]] * cos
            }
        })
    }

    // ── procrustes tests ──────────────────────────────────────────────────────

    #[test]
    fn procrustes_aligns_rotated_copies_exactly() {
        // Build X of shape [6, 2], create Y = X rotated by 90°.
        // Procrustes should recover a rotation so X·W ≈ Y.
        let x = rand_matrix(6, 2, 1);
        let angle = std::f32::consts::FRAC_PI_2; // 90°
        let y = rotate_2d(&x, angle);

        let w = procrustes_align(&x, &y).expect("procrustes should succeed");

        let xw = x.dot(&w);
        let mut max_err = 0.0f32;
        for i in 0..6 {
            for j in 0..2 {
                let err = (xw[[i, j]] - y[[i, j]]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 1e-4,
            "max element-wise error = {max_err}, expected < 1e-4"
        );
    }

    #[test]
    fn procrustes_identity_when_src_equals_tgt() {
        // If X == Y the optimal W is the identity matrix (up to reflections).
        // We verify X·W ≈ X rather than checking W == I directly.
        let x = rand_matrix(5, 3, 99);
        let w = procrustes_align(&x, &x).expect("procrustes should succeed");
        let xw = x.dot(&w);

        for i in 0..5 {
            for j in 0..3 {
                assert!(
                    (xw[[i, j]] - x[[i, j]]).abs() < 1e-4,
                    "xw[{i},{j}] = {} ≠ x[{i},{j}] = {}",
                    xw[[i, j]],
                    x[[i, j]]
                );
            }
        }
    }

    #[test]
    fn procrustes_fit_reduces_frobenius_distance() {
        // After alignment, ||X·W - Y||_F should be ≤ ||X - Y||_F.
        let x = rand_matrix(8, 3, 42);
        let y = rand_matrix(8, 3, 77);

        let frobenius = |a: &Array2<f32>, b: &Array2<f32>| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).powi(2))
                .sum::<f32>()
                .sqrt()
        };

        let before = frobenius(&x, &y);
        let aligner = CrossLingualAligner::fit(&x, &y).expect("fit should succeed");
        let xw = aligner.transform_batch(&x);
        let after = frobenius(&xw, &y);

        assert!(
            after <= before + 1e-4,
            "||X·W - Y||_F = {after} should be ≤ ||X - Y||_F = {before}"
        );
    }

    // ── aligned_encoder tests ─────────────────────────────────────────────────

    #[test]
    fn aligned_encoder_preserves_approximate_norm() {
        // An orthogonal transform preserves the Euclidean norm.
        let x = rand_matrix(5, 2, 10);
        let y = rotate_2d(&x, 0.5);

        let aligner = CrossLingualAligner::fit(&x, &y).expect("fit should succeed");

        let encoder = |tokens: &[usize]| -> scirs2_core::ndarray::Array1<f32> {
            // encode each token set as a 2-D sum vector
            let mut v = scirs2_core::ndarray::Array1::<f32>::zeros(2);
            for &t in tokens {
                let row = t % 5;
                v[0] += x[[row, 0]];
                v[1] += x[[row, 1]];
            }
            v
        };

        let enc = AlignedEncoder::new(&encoder, &aligner, false);

        for seed in 0..4usize {
            let tokens: Vec<usize> = vec![seed, seed + 1];
            let base = encoder(&tokens);
            let aligned_out = enc.encode(&tokens);

            let norm_base: f32 = base.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_aligned: f32 = aligned_out.iter().map(|x| x * x).sum::<f32>().sqrt();

            assert!(
                (norm_base - norm_aligned).abs() < 1e-4,
                "norms differ: base={norm_base}, aligned={norm_aligned}"
            );
        }
    }

    #[test]
    fn cross_lingual_transform_batch_equals_individual() {
        let src = rand_matrix(6, 3, 55);
        let tgt = rand_matrix(6, 3, 66);
        let aligner = CrossLingualAligner::fit(&src, &tgt).expect("fit");

        let batch_out = aligner.transform_batch(&src);

        for i in 0..6 {
            let row = src.index_axis(Axis(0), i).to_owned();
            let individual = aligner.transform(&row);
            let batch_row = batch_out.index_axis(Axis(0), i);
            for j in 0..3 {
                assert!(
                    (individual[j] - batch_row[j]).abs() < 1e-6,
                    "row {i} col {j}: individual={} batch={}",
                    individual[j],
                    batch_row[j]
                );
            }
        }
    }

    #[test]
    fn procrustes_mismatched_rows_returns_error() {
        let x = rand_matrix(4, 2, 1);
        let y = rand_matrix(3, 2, 2);
        let result = procrustes_align(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn aligned_encoder_normalise_output_unit_norm() {
        let x = rand_matrix(4, 2, 7);
        let y = rotate_2d(&x, 0.3);
        let aligner = CrossLingualAligner::fit(&x, &y).expect("fit");

        let encoder = |tokens: &[usize]| -> scirs2_core::ndarray::Array1<f32> {
            let mut v = scirs2_core::ndarray::Array1::<f32>::zeros(2);
            for &t in tokens {
                let row = t % 4;
                v[0] += x[[row, 0]];
                v[1] += x[[row, 1]];
            }
            v
        };

        let enc = AlignedEncoder::new(&encoder, &aligner, true);
        let out = enc.encode(&[0, 1, 2]);
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }
}
