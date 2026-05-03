//! Integration tests for cross-lingual sentence embedding alignment.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_text::sentence_embeddings::{procrustes_align, AlignedEncoder, CrossLingualAligner};

// ── helpers ───────────────────────────────────────────────────────────────────

fn lcg_f32(seed: u64, offset: u64) -> f32 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let state = A.wrapping_mul(seed.wrapping_add(offset)).wrapping_add(C);
    (((state >> 12) as f64) / ((1u64 << 52) as f64)) as f32 * 2.0 - 1.0
}

fn rand_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| lcg_f32(seed, (i * cols + j) as u64))
}

/// Apply a 2-D rotation of `angle` radians to the rows of `x` (which must
/// have exactly 2 columns).
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

fn frobenius_dist(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ── procrustes_align tests ────────────────────────────────────────────────────

#[test]
fn procrustes_aligns_rotated_copies_exactly() {
    // X of shape [8, 2], Y = X rotated 90°; W should ≈ R(90°).
    let x = rand_matrix(8, 2, 1);
    let y = rotate_2d(&x, std::f32::consts::FRAC_PI_2);

    let w = procrustes_align(&x, &y).expect("procrustes should succeed");
    let xw = x.dot(&w);

    let max_err = xw
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-4,
        "max element-wise error = {max_err} (expected < 1e-4)"
    );
}

#[test]
fn procrustes_identity_when_src_equals_tgt() {
    // If X == Y the minimiser W must satisfy X·W ≈ X.
    let x = rand_matrix(5, 3, 99);
    let w = procrustes_align(&x, &x).expect("procrustes should succeed");
    let xw = x.dot(&w);

    let max_err = xw
        .iter()
        .zip(x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-4,
        "max error = {max_err} (expected < 1e-4 for identity case)"
    );
}

#[test]
fn procrustes_fit_reduces_frobenius_distance() {
    let x = rand_matrix(10, 4, 42);
    let y = rand_matrix(10, 4, 77);

    let before = frobenius_dist(&x, &y);
    let aligner = CrossLingualAligner::fit(&x, &y).expect("fit should succeed");
    let xw = aligner.transform_batch(&x);
    let after = frobenius_dist(&xw, &y);

    assert!(
        after <= before + 1e-4,
        "||X·W - Y||_F = {after} > ||X - Y||_F = {before}"
    );
}

#[test]
fn procrustes_mismatched_rows_returns_error() {
    let x = rand_matrix(4, 2, 1);
    let y = rand_matrix(3, 2, 2);
    assert!(procrustes_align(&x, &y).is_err());
}

// ── CrossLingualAligner tests ─────────────────────────────────────────────────

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

// ── AlignedEncoder tests ──────────────────────────────────────────────────────

#[test]
fn aligned_encoder_preserves_approximate_norm() {
    // Orthogonal transform preserves the Euclidean norm.
    let x = rand_matrix(5, 2, 10);
    let y = rotate_2d(&x, 0.5);

    let aligner = CrossLingualAligner::fit(&x, &y).expect("fit");

    let encoder = |tokens: &[usize]| -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(2);
        for &t in tokens {
            let row = t % 5;
            v[0] += x[[row, 0]];
            v[1] += x[[row, 1]];
        }
        v
    };

    let enc = AlignedEncoder::new(&encoder, &aligner, false);

    for seed in 0..4usize {
        let tokens = vec![seed, seed + 1];
        let base = encoder(&tokens);
        let aligned_out = enc.encode(&tokens);

        let norm_base: f32 = base.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_aligned: f32 = aligned_out.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            (norm_base - norm_aligned).abs() < 1e-4,
            "seed={seed}: base_norm={norm_base}, aligned_norm={norm_aligned}"
        );
    }
}

#[test]
fn aligned_encoder_normalise_output_unit_norm() {
    let x = rand_matrix(4, 2, 7);
    let y = rotate_2d(&x, 0.3);
    let aligner = CrossLingualAligner::fit(&x, &y).expect("fit");

    let encoder = |tokens: &[usize]| -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(2);
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
