//! Co-occurrence Matrix Features and Run-Length Matrix
//!
//! This module provides:
//!
//! - **GLCM** (Gray-Level Co-occurrence Matrix) computation from `u8` images.
//! - Statistical features derived from a GLCM:
//!   contrast, correlation, energy, homogeneity, entropy.
//! - **Run-Length Matrix** (RLRM) for texture run-length statistics.
//!
//! # References
//! - Haralick, R.M., Shanmugam, K. & Dinstein, I. (1973). "Textural Features for Image
//!   Classification." IEEE Transactions on Systems, Man, and Cybernetics.
//! - Galloway, M.M. (1975). "Texture analysis using gray level run lengths." Computer
//!   Graphics and Image Processing.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3};

// ---------------------------------------------------------------------------
// GLCM computation
// ---------------------------------------------------------------------------

/// Compute the normalized Gray-Level Co-occurrence Matrix (GLCM).
///
/// For each offset in `offsets` a separate GLCM of size `n_levels × n_levels`
/// is computed and stored in the output as a 3-D array of shape
/// `(n_offsets, n_levels, n_levels)`.
///
/// Each GLCM is symmetric (both `(i,j)` and `(j,i)` transitions are counted)
/// and normalized so that the entries sum to 1.
///
/// # Parameters
/// - `image`    – input 8-bit grayscale image.  Pixel values must lie in
///               `[0, n_levels - 1]`; values ≥ `n_levels` are **ignored**.
/// - `offsets`  – spatial displacement vectors `(dy, dx)`.
/// - `n_levels` – number of gray levels (must be ≥ 2).
///
/// # Returns
/// Array of shape `(n_offsets, n_levels, n_levels)` with normalized
/// co-occurrence probabilities.
///
/// # Errors
/// - `NdimageError::InvalidInput` when `n_levels < 2`, `offsets` is empty,
///   or the image is empty.
pub fn glcm(
    image: &Array2<u8>,
    offsets: &[(i32, i32)],
    n_levels: usize,
) -> NdimageResult<Array3<f64>> {
    if n_levels < 2 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be at least 2".into(),
        ));
    }
    if offsets.is_empty() {
        return Err(NdimageError::InvalidInput(
            "offsets must be non-empty".into(),
        ));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    let n_offsets = offsets.len();
    let mut out = Array3::<f64>::zeros((n_offsets, n_levels, n_levels));

    for (k, &(dy, dx)) in offsets.iter().enumerate() {
        let mut count = 0.0f64;

        for r in 0..rows {
            for c in 0..cols {
                let nr = r as i64 + dy as i64;
                let nc = c as i64 + dx as i64;
                if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 {
                    continue;
                }
                let g1 = image[[r, c]] as usize;
                let g2 = image[[nr as usize, nc as usize]] as usize;
                if g1 < n_levels && g2 < n_levels {
                    out[[k, g1, g2]] += 1.0;
                    out[[k, g2, g1]] += 1.0;
                    count += 2.0;
                }
            }
        }

        // Normalize
        if count > 0.0 {
            for g1 in 0..n_levels {
                for g2 in 0..n_levels {
                    out[[k, g1, g2]] /= count;
                }
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Helper: extract a 2-D view for one offset from the GLCM volume
// ---------------------------------------------------------------------------

fn check_glcm_slice(glcm_slice: &Array2<f64>) -> NdimageResult<usize> {
    let (r, c) = glcm_slice.dim();
    if r != c || r < 2 {
        return Err(NdimageError::InvalidInput(
            "GLCM slice must be a square matrix with at least 2 levels".into(),
        ));
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// GLCM scalar features
// ---------------------------------------------------------------------------

/// Compute the **contrast** feature from a normalized GLCM.
///
/// `contrast = sum_{i,j} (i - j)^2 * P(i,j)`
///
/// A high value means large local intensity variations; zero means the
/// image is uniform within the spatial offset.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` when the matrix is not square or
/// has fewer than 2 levels.
pub fn glcm_contrast(glcm_slice: &Array2<f64>) -> NdimageResult<f64> {
    let n = check_glcm_slice(glcm_slice)?;
    let mut contrast = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let diff = i as f64 - j as f64;
            contrast += diff * diff * glcm_slice[[i, j]];
        }
    }
    Ok(contrast)
}

/// Compute the **correlation** feature from a normalized GLCM.
///
/// `correlation = [sum_{i,j} (i - mu_i)(j - mu_j) P(i,j)] / (sigma_i * sigma_j)`
///
/// Returns 0 when either marginal standard deviation is zero.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for a non-square or degenerate matrix.
pub fn glcm_correlation(glcm_slice: &Array2<f64>) -> NdimageResult<f64> {
    let n = check_glcm_slice(glcm_slice)?;

    // Marginal means
    let mut mu_i = 0.0f64;
    let mut mu_j = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let p = glcm_slice[[i, j]];
            mu_i += i as f64 * p;
            mu_j += j as f64 * p;
        }
    }

    // Marginal standard deviations
    let mut var_i = 0.0f64;
    let mut var_j = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let p = glcm_slice[[i, j]];
            var_i += (i as f64 - mu_i).powi(2) * p;
            var_j += (j as f64 - mu_j).powi(2) * p;
        }
    }

    let sigma_i = var_i.sqrt();
    let sigma_j = var_j.sqrt();

    if sigma_i < 1e-12 || sigma_j < 1e-12 {
        return Ok(0.0);
    }

    let mut corr = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let p = glcm_slice[[i, j]];
            corr += (i as f64 - mu_i) * (j as f64 - mu_j) * p;
        }
    }
    Ok(corr / (sigma_i * sigma_j))
}

/// Compute the **energy** (Angular Second Moment) from a normalized GLCM.
///
/// `energy = sqrt(sum_{i,j} P(i,j)^2)`
///
/// Values in [0, 1]; 1 means the image has a constant texture.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for a non-square or degenerate matrix.
pub fn glcm_energy(glcm_slice: &Array2<f64>) -> NdimageResult<f64> {
    let n = check_glcm_slice(glcm_slice)?;
    let asm: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| glcm_slice[[i, j]].powi(2))
        .sum();
    Ok(asm.sqrt())
}

/// Compute the **homogeneity** (Inverse Difference Moment) from a normalized GLCM.
///
/// `homogeneity = sum_{i,j} P(i,j) / (1 + (i - j)^2)`
///
/// Values in [0, 1]; 1 when all co-occurrences are on the diagonal (no
/// gray-level transitions at the given offset).
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for a non-square or degenerate matrix.
pub fn glcm_homogeneity(glcm_slice: &Array2<f64>) -> NdimageResult<f64> {
    let n = check_glcm_slice(glcm_slice)?;
    let mut homogeneity = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let diff = i as f64 - j as f64;
            homogeneity += glcm_slice[[i, j]] / (1.0 + diff * diff);
        }
    }
    Ok(homogeneity)
}

/// Compute the **entropy** from a normalized GLCM.
///
/// `entropy = -sum_{i,j} P(i,j) * log(P(i,j))`
///
/// (zero-probability entries contribute 0 by convention).
///
/// # Errors
/// Returns `NdimageError::InvalidInput` for a non-square or degenerate matrix.
pub fn glcm_entropy(glcm_slice: &Array2<f64>) -> NdimageResult<f64> {
    let n = check_glcm_slice(glcm_slice)?;
    let mut entropy = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let p = glcm_slice[[i, j]];
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
    }
    Ok(entropy)
}

// ---------------------------------------------------------------------------
// Run-Length Matrix (RLRM)
// ---------------------------------------------------------------------------

/// Scan direction for the run-length matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlrmDirection {
    /// Horizontal (left-to-right) runs
    Horizontal,
    /// Vertical (top-to-bottom) runs
    Vertical,
    /// Diagonal 45° (upper-left to lower-right)
    Diagonal45,
    /// Anti-diagonal 135° (upper-right to lower-left)
    Diagonal135,
}

/// Compute the Gray-Level Run-Length Matrix (GLRLM).
///
/// The matrix `M[g, l]` counts the number of runs of consecutive pixels all
/// having gray level `g` and run length `l+1` in the given direction.
///
/// The output shape is `(n_levels, max_run_length)` where `max_run_length`
/// is the longest run found in the image.  If the image is uniform, this
/// equals `rows` or `cols` depending on direction.
///
/// # Parameters
/// - `image`     – 8-bit grayscale image.  Pixel values ≥ `n_levels` are
///                treated as gray level `n_levels - 1`.
/// - `direction` – scan direction for run enumeration.
/// - `n_levels`  – number of distinct gray levels (≥ 2).
///
/// # Returns
/// `Array2<f64>` of shape `(n_levels, max_run_length)` (un-normalized).
///
/// # Errors
/// Returns `NdimageError::InvalidInput` when `n_levels < 2` or the image
/// is empty.
pub fn run_length_matrix(
    image: &Array2<u8>,
    direction: RlrmDirection,
    n_levels: usize,
) -> NdimageResult<Array2<f64>> {
    if n_levels < 2 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be at least 2".into(),
        ));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    // Collect all runs for the chosen direction
    let runs = collect_runs(image, direction, rows, cols, n_levels);

    // Determine maximum run length
    let max_run = runs.iter().map(|&(_, l)| l).max().unwrap_or(1);

    let mut matrix = Array2::<f64>::zeros((n_levels, max_run));

    for (g, l) in runs {
        matrix[[g, l - 1]] += 1.0;
    }

    Ok(matrix)
}

/// Collect `(gray_level, run_length)` pairs for all runs in the image.
fn collect_runs(
    image: &Array2<u8>,
    direction: RlrmDirection,
    rows: usize,
    cols: usize,
    n_levels: usize,
) -> Vec<(usize, usize)> {
    match direction {
        RlrmDirection::Horizontal => {
            let mut runs = Vec::new();
            for r in 0..rows {
                let mut c = 0;
                while c < cols {
                    let g = (image[[r, c]] as usize).min(n_levels - 1);
                    let start = c;
                    while c < cols && (image[[r, c]] as usize).min(n_levels - 1) == g {
                        c += 1;
                    }
                    runs.push((g, c - start));
                }
            }
            runs
        }
        RlrmDirection::Vertical => {
            let mut runs = Vec::new();
            for c in 0..cols {
                let mut r = 0;
                while r < rows {
                    let g = (image[[r, c]] as usize).min(n_levels - 1);
                    let start = r;
                    while r < rows && (image[[r, c]] as usize).min(n_levels - 1) == g {
                        r += 1;
                    }
                    runs.push((g, r - start));
                }
            }
            runs
        }
        RlrmDirection::Diagonal45 => {
            // Diagonals from top-right corner downward (r increases, c increases)
            let mut runs = Vec::new();
            // Starting points: first row and first column
            let starts: Vec<(i64, i64)> = (0..cols as i64)
                .map(|c| (0i64, c))
                .chain((1..rows as i64).map(|r| (r, 0i64)))
                .collect();

            for (r0, c0) in starts {
                // Walk diagonal
                let diag: Vec<usize> = (0..)
                    .map(|k| (r0 + k as i64, c0 + k as i64))
                    .take_while(|&(r, c)| r >= 0 && r < rows as i64 && c >= 0 && c < cols as i64)
                    .map(|(r, c)| (image[[r as usize, c as usize]] as usize).min(n_levels - 1))
                    .collect();

                let mut i = 0;
                while i < diag.len() {
                    let g = diag[i];
                    let start = i;
                    while i < diag.len() && diag[i] == g {
                        i += 1;
                    }
                    runs.push((g, i - start));
                }
            }
            runs
        }
        RlrmDirection::Diagonal135 => {
            // Anti-diagonal: r increases, c decreases
            let mut runs = Vec::new();
            let starts: Vec<(i64, i64)> = (0..cols as i64)
                .map(|c| (0i64, c))
                .chain((1..rows as i64).map(|r| (r, cols as i64 - 1)))
                .collect();

            for (r0, c0) in starts {
                let diag: Vec<usize> = (0..)
                    .map(|k| (r0 + k as i64, c0 - k as i64))
                    .take_while(|&(r, c)| r >= 0 && r < rows as i64 && c >= 0 && c < cols as i64)
                    .map(|(r, c)| (image[[r as usize, c as usize]] as usize).min(n_levels - 1))
                    .collect();

                let mut i = 0;
                while i < diag.len() {
                    let g = diag[i];
                    let start = i;
                    while i < diag.len() && diag[i] == g {
                        i += 1;
                    }
                    runs.push((g, i - start));
                }
            }
            runs
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn simple_image_u8() -> Array2<u8> {
        Array2::from_shape_vec((3, 3), vec![0u8, 1, 0, 1, 0, 1, 0, 1, 0]).expect("shape ok")
    }

    #[test]
    fn test_glcm_shape() {
        let img = simple_image_u8();
        let offsets = vec![(0i32, 1i32), (1, 0)];
        let g = glcm(&img, &offsets, 4).expect("glcm ok");
        assert_eq!(g.dim(), (2, 4, 4));
    }

    #[test]
    fn test_glcm_normalized() {
        let img = simple_image_u8();
        let g = glcm(&img, &[(0i32, 1i32)], 2).expect("glcm ok");
        // Sum of the (0, 0) GLCM slice should be ~1
        let sum: f64 = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| g[[0, i, j]])
            .sum();
        assert!(
            (sum - 1.0).abs() < 1e-12 || sum == 0.0,
            "GLCM should be normalized, sum={sum}"
        );
    }

    #[test]
    fn test_glcm_contrast_uniform() {
        // Uniform image → all transitions are 0→0 → contrast = 0
        let img = Array2::from_elem((4, 4), 1u8);
        let g = glcm(&img, &[(0i32, 1i32)], 4).expect("glcm ok");
        let slice = g.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let c = glcm_contrast(&slice).expect("contrast ok");
        assert!(c < 1e-12, "Contrast of uniform image should be 0, got {c}");
    }

    #[test]
    fn test_glcm_energy_uniform() {
        // Uniform image → GLCM has a single non-zero entry → high energy
        let img = Array2::from_elem((4, 4), 0u8);
        let g = glcm(&img, &[(0i32, 1i32)], 2).expect("glcm ok");
        let slice = g.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let e = glcm_energy(&slice).expect("energy ok");
        // Should be 1.0 (all mass in one cell)
        assert!(
            (e - 1.0).abs() < 1e-10,
            "Energy of uniform image should be 1, got {e}"
        );
    }

    #[test]
    fn test_glcm_homogeneity_uniform() {
        let img = Array2::from_elem((4, 4), 0u8);
        let g = glcm(&img, &[(0i32, 1i32)], 2).expect("glcm ok");
        let slice = g.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let h = glcm_homogeneity(&slice).expect("homogeneity ok");
        // All on diagonal → 1.0
        assert!(
            (h - 1.0).abs() < 1e-10,
            "Homogeneity of uniform image should be 1, got {h}"
        );
    }

    #[test]
    fn test_glcm_entropy_uniform() {
        let img = Array2::from_elem((4, 4), 0u8);
        let g = glcm(&img, &[(0i32, 1i32)], 2).expect("glcm ok");
        let slice = g.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let ent = glcm_entropy(&slice).expect("entropy ok");
        // P(0,0) = 1 → entropy = 0
        assert!(
            ent < 1e-10,
            "Entropy of uniform image should be 0, got {ent}"
        );
    }

    #[test]
    fn test_glcm_correlation_bounds() {
        let img = simple_image_u8();
        let g = glcm(&img, &[(0i32, 1i32)], 2).expect("glcm ok");
        let slice = g.slice(scirs2_core::ndarray::s![0, .., ..]).to_owned();
        let corr = glcm_correlation(&slice).expect("correlation ok");
        assert!(
            corr >= -1.0 - 1e-9 && corr <= 1.0 + 1e-9,
            "Correlation must be in [-1, 1], got {corr}"
        );
    }

    #[test]
    fn test_run_length_matrix_horizontal() {
        // Image with long horizontal runs at level 0 and 1
        let img = Array2::from_shape_fn((4, 8), |(_, c)| if c < 4 { 0u8 } else { 1u8 });
        let rlm = run_length_matrix(&img, RlrmDirection::Horizontal, 4).expect("rlm ok");
        // Shape: (4 levels, max_run_length)
        // There should be 4 runs of length 4 for level 0 and 4 runs for level 1
        assert!(rlm.dim().0 == 4, "n_levels mismatch");
        // rlm[[0, 3]] counts runs of gray=0, length=4 → should be 4
        assert!(
            rlm[[0, 3]] >= 3.0,
            "Expected >=3 runs of length 4 at gray=0, got {}",
            rlm[[0, 3]]
        );
    }

    #[test]
    fn test_run_length_matrix_vertical() {
        let img = Array2::from_shape_fn((8, 4), |(r, _)| if r < 4 { 0u8 } else { 1u8 });
        let rlm = run_length_matrix(&img, RlrmDirection::Vertical, 4).expect("rlm ok");
        // 4 vertical runs of length 4 for gray=0
        assert!(
            rlm[[0, 3]] >= 3.0,
            "Expected >=3 vertical runs, got {}",
            rlm[[0, 3]]
        );
    }

    #[test]
    fn test_run_length_matrix_diagonal() {
        let img: Array2<u8> = Array2::from_elem((6, 6), 0u8);
        let rlm = run_length_matrix(&img, RlrmDirection::Diagonal45, 2).expect("rlm ok");
        // All pixels are 0, all diagonals are uniform
        let total: f64 = rlm.iter().sum();
        assert!(total > 0.0);
    }

    #[test]
    fn test_glcm_too_few_levels_errors() {
        let img: Array2<u8> = Array2::from_elem((4, 4), 0u8);
        let err = glcm(&img, &[(0i32, 1i32)], 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_rlm_too_few_levels_errors() {
        let img: Array2<u8> = Array2::from_elem((4, 4), 0u8);
        let err = run_length_matrix(&img, RlrmDirection::Horizontal, 1);
        assert!(err.is_err());
    }
}
