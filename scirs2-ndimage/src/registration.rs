//! Image Registration Module
//!
//! Provides algorithms for aligning images and point sets:
//!
//! - **Phase correlation**: FFT-based sub-pixel translation estimation
//! - **ICP (Iterative Closest Point)**: Point set registration
//! - **Affine registration**: Full affine transform fitting (6 DOF in 2D)
//! - **Rigid registration**: Rotation + translation (3 DOF in 2D)
//! - **Multi-resolution pyramid**: Coarse-to-fine registration
//! - **Quality metrics**: TRE, mutual information estimate
//! - **Deformable registration**: diffeomorphic demons, fluid model, B-spline FFD

/// Deformable image registration (demons, fluid, B-spline FFD).
pub mod deformable;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft2, fftfreq, ifft2};
use std::f64::consts::PI;

use crate::error::{NdimageError, NdimageResult};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Result of a translation registration (phase correlation)
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Estimated shift along row (y) axis
    pub shift_y: f64,
    /// Estimated shift along column (x) axis
    pub shift_x: f64,
    /// Peak correlation value (confidence indicator, 0..1 range)
    pub peak_value: f64,
}

/// A 2-D affine transform represented as a 3x3 homogeneous matrix.
///
/// The matrix maps source coordinates to target coordinates:
///   \[x'\]   \[a00 a01 a02\] \[x\]
///   \[y'\] = \[a10 a11 a12\] \[y\]
///   \[ 1\]   \[ 0   0   1 \] \[1\]
#[derive(Debug, Clone)]
pub struct AffineTransform2D {
    /// 3x3 homogeneous matrix (last row is \[0,0,1\])
    pub matrix: Array2<f64>,
    /// Residual (mean squared error) of the fit
    pub residual: f64,
}

/// Result of rigid registration (rotation + translation)
#[derive(Debug, Clone)]
pub struct RigidTransform2D {
    /// Rotation angle in radians (counter-clockwise)
    pub angle: f64,
    /// Translation along x
    pub tx: f64,
    /// Translation along y
    pub ty: f64,
    /// Residual (mean squared error)
    pub residual: f64,
}

/// Result of ICP registration
#[derive(Debug, Clone)]
pub struct IcpResult {
    /// Final rigid transform
    pub transform: RigidTransform2D,
    /// Number of iterations performed
    pub iterations: usize,
    /// History of mean squared errors per iteration
    pub mse_history: Vec<f64>,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Configuration for ICP
#[derive(Debug, Clone)]
pub struct IcpConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance on MSE change
    pub tolerance: f64,
    /// Maximum correspondence distance (points farther than this are rejected)
    pub max_distance: Option<f64>,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            max_distance: None,
        }
    }
}

/// Configuration for multi-resolution pyramid registration
#[derive(Debug, Clone)]
pub struct PyramidConfig {
    /// Number of pyramid levels (including the original resolution)
    pub levels: usize,
    /// Down-sampling factor between successive levels
    pub scale_factor: f64,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            scale_factor: 2.0,
        }
    }
}

/// Registration quality metrics
#[derive(Debug, Clone)]
pub struct RegistrationMetrics {
    /// Target Registration Error -- RMS distance between transformed source
    /// landmarks and corresponding target landmarks
    pub tre: f64,
    /// Estimated mutual information (histogram-based, discrete)
    pub mutual_information: f64,
    /// Normalized Cross-Correlation
    pub ncc: f64,
}

// ---------------------------------------------------------------------------
// Phase correlation
// ---------------------------------------------------------------------------

/// Estimate translation between two images using phase correlation.
///
/// Computes the cross-power spectrum of the two images, then finds the peak
/// of its inverse FFT.  The location of the peak gives the integer shift;
/// sub-pixel refinement is performed via parabolic interpolation.
///
/// Both images must have the same shape.
///
/// # Arguments
/// * `reference` - Reference (fixed) image
/// * `moving`    - Moving (to-be-registered) image
///
/// # Returns
/// A `TranslationResult` with the estimated shift and confidence.
pub fn phase_correlation(
    reference: &Array2<f64>,
    moving: &Array2<f64>,
) -> NdimageResult<TranslationResult> {
    let (ny, nx) = reference.dim();
    if moving.dim() != (ny, nx) {
        return Err(NdimageError::DimensionError(format!(
            "Image shapes must match: reference ({},{}) vs moving ({},{})",
            ny,
            nx,
            moving.nrows(),
            moving.ncols()
        )));
    }
    if ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput(
            "Images must be non-empty".into(),
        ));
    }

    // Forward FFT of both images
    let spec_ref = fft2(reference, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("FFT of reference failed: {}", e)))?;
    let spec_mov = fft2(moving, None, None, None).map_err(|e| {
        NdimageError::ComputationError(format!("FFT of moving image failed: {}", e))
    })?;

    // Cross-power spectrum:  R = F1* . F2 / |F1* . F2|
    let mut cross_power = Array2::<Complex64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            let prod = spec_ref[[i, j]].conj() * spec_mov[[i, j]];
            let mag = prod.norm();
            cross_power[[i, j]] = if mag > 1e-15 {
                prod / mag
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }

    // Inverse FFT to get the correlation surface
    let corr_complex = ifft2(&cross_power, None, None, None).map_err(|e| {
        NdimageError::ComputationError(format!("IFFT of cross-power failed: {}", e))
    })?;

    // Find peak in the real part
    let mut best_val = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    let mut best_j = 0usize;
    for i in 0..ny {
        for j in 0..nx {
            let v = corr_complex[[i, j]].re;
            if v > best_val {
                best_val = v;
                best_i = i;
                best_j = j;
            }
        }
    }

    // Sub-pixel refinement via parabolic interpolation along each axis
    let sub_y = subpixel_1d(
        corr_complex[[(best_i + ny - 1) % ny, best_j]].re,
        best_val,
        corr_complex[[(best_i + 1) % ny, best_j]].re,
    );
    let sub_x = subpixel_1d(
        corr_complex[[best_i, (best_j + nx - 1) % nx]].re,
        best_val,
        corr_complex[[best_i, (best_j + 1) % nx]].re,
    );

    // Convert from FFT index to shift (wrap around center)
    let shift_y = if best_i as f64 + sub_y > ny as f64 / 2.0 {
        best_i as f64 + sub_y - ny as f64
    } else {
        best_i as f64 + sub_y
    };
    let shift_x = if best_j as f64 + sub_x > nx as f64 / 2.0 {
        best_j as f64 + sub_x - nx as f64
    } else {
        best_j as f64 + sub_x
    };

    Ok(TranslationResult {
        shift_y,
        shift_x,
        peak_value: best_val,
    })
}

/// Parabolic sub-pixel refinement: given three consecutive samples
/// `(y_minus, y_center, y_plus)` around a peak, returns the fractional offset.
fn subpixel_1d(y_minus: f64, y_center: f64, y_plus: f64) -> f64 {
    let denom = 2.0 * (2.0 * y_center - y_minus - y_plus);
    if denom.abs() < 1e-15 {
        0.0
    } else {
        (y_minus - y_plus) / denom
    }
}

// ---------------------------------------------------------------------------
// Affine registration (least-squares)
// ---------------------------------------------------------------------------

/// Compute a 2-D affine transform that maps `source` points to `target` points
/// in the least-squares sense.
///
/// Each row of `source` / `target` is a point `[x, y]`.
/// At least 3 non-collinear point pairs are required.
///
/// The affine transform is:
///   x' = a00*x + a01*y + a02
///   y' = a10*x + a11*y + a12
pub fn affine_registration(
    source: &Array2<f64>,
    target: &Array2<f64>,
) -> NdimageResult<AffineTransform2D> {
    let n = source.nrows();
    if n < 3 {
        return Err(NdimageError::InvalidInput(
            "Need at least 3 point pairs for affine registration".into(),
        ));
    }
    if source.ncols() != 2 || target.ncols() != 2 {
        return Err(NdimageError::InvalidInput(
            "Point arrays must have 2 columns (x, y)".into(),
        ));
    }
    if target.nrows() != n {
        return Err(NdimageError::DimensionError(
            "source and target must have the same number of rows".into(),
        ));
    }

    // Build the design matrix A (n*2  x  6) and observation vector b (n*2)
    // For each point pair (sx, sy) -> (tx, ty):
    //   tx = a00*sx + a01*sy + a02
    //   ty = a10*sx + a11*sy + a12
    //
    // We solve  A * p = b  with p = [a00 a01 a02 a10 a11 a12]^T
    let m = 2 * n;
    let mut a_mat = Array2::<f64>::zeros((m, 6));
    let mut b_vec = Array1::<f64>::zeros(m);

    for k in 0..n {
        let sx = source[[k, 0]];
        let sy = source[[k, 1]];
        // row for x'
        let r0 = 2 * k;
        a_mat[[r0, 0]] = sx;
        a_mat[[r0, 1]] = sy;
        a_mat[[r0, 2]] = 1.0;
        b_vec[r0] = target[[k, 0]];
        // row for y'
        let r1 = 2 * k + 1;
        a_mat[[r1, 3]] = sx;
        a_mat[[r1, 4]] = sy;
        a_mat[[r1, 5]] = 1.0;
        b_vec[r1] = target[[k, 1]];
    }

    // Solve via normal equations:  A^T A p = A^T b
    let ata = a_mat.t().dot(&a_mat);
    let atb = a_mat.t().dot(&b_vec);

    let params = solve_6x6(&ata, &atb)?;

    // Build homogeneous 3x3 matrix
    let mut matrix = Array2::<f64>::zeros((3, 3));
    matrix[[0, 0]] = params[0];
    matrix[[0, 1]] = params[1];
    matrix[[0, 2]] = params[2];
    matrix[[1, 0]] = params[3];
    matrix[[1, 1]] = params[4];
    matrix[[1, 2]] = params[5];
    matrix[[2, 2]] = 1.0;

    // Compute residual
    let predicted = a_mat.dot(&params);
    let diff = &predicted - &b_vec;
    let residual = diff.dot(&diff) / n as f64;

    Ok(AffineTransform2D { matrix, residual })
}

/// Solve a 6x6 symmetric positive-definite system via Cholesky decomposition.
fn solve_6x6(ata: &Array2<f64>, atb: &Array1<f64>) -> NdimageResult<Array1<f64>> {
    let n = 6;
    // Cholesky L such that ata = L * L^T
    let mut l_mat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l_mat[[i, k]] * l_mat[[j, k]];
            }
            if i == j {
                let diag = ata[[i, i]] - s;
                if diag <= 0.0 {
                    return Err(NdimageError::ComputationError(
                        "Matrix is not positive-definite (collinear points?)".into(),
                    ));
                }
                l_mat[[i, j]] = diag.sqrt();
            } else {
                l_mat[[i, j]] = (ata[[i, j]] - s) / l_mat[[j, j]];
            }
        }
    }

    // Forward substitution: L y = atb
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for k in 0..i {
            s += l_mat[[i, k]] * y[k];
        }
        y[i] = (atb[i] - s) / l_mat[[i, i]];
    }

    // Back substitution: L^T x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = 0.0;
        for k in (i + 1)..n {
            s += l_mat[[k, i]] * x[k];
        }
        x[i] = (y[i] - s) / l_mat[[i, i]];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Rigid registration (SVD-based, Umeyama / Procrustes)
// ---------------------------------------------------------------------------

/// Compute the rigid (rotation + translation) transform that best maps `source`
/// to `target` in the least-squares sense.
///
/// Uses the SVD-based method (Umeyama 1991).
/// Each row is a 2-D point `[x, y]`.  At least 2 non-coincident point pairs
/// are needed.
pub fn rigid_registration(
    source: &Array2<f64>,
    target: &Array2<f64>,
) -> NdimageResult<RigidTransform2D> {
    let n = source.nrows();
    if n < 2 {
        return Err(NdimageError::InvalidInput(
            "Need at least 2 point pairs for rigid registration".into(),
        ));
    }
    if source.ncols() != 2 || target.ncols() != 2 {
        return Err(NdimageError::InvalidInput(
            "Point arrays must have 2 columns (x, y)".into(),
        ));
    }
    if target.nrows() != n {
        return Err(NdimageError::DimensionError(
            "source and target must have the same number of rows".into(),
        ));
    }

    // Centroids
    let src_mean = source.mean_axis(Axis(0)).ok_or_else(|| {
        NdimageError::ComputationError("Failed to compute source centroid".into())
    })?;
    let tgt_mean = target.mean_axis(Axis(0)).ok_or_else(|| {
        NdimageError::ComputationError("Failed to compute target centroid".into())
    })?;

    // Center the points
    let src_centered = source - &src_mean.view().insert_axis(Axis(0));
    let tgt_centered = target - &tgt_mean.view().insert_axis(Axis(0));

    // Cross-covariance matrix H = src_centered^T * tgt_centered  (2x2)
    let h = src_centered.t().dot(&tgt_centered);

    // SVD of H via closed-form for 2x2
    let (u, _s, vt) = svd_2x2(h[[0, 0]], h[[0, 1]], h[[1, 0]], h[[1, 1]]);

    // Rotation matrix R = V * U^T
    // Ensure proper rotation (det > 0)
    let det = (u[[0, 0]] * u[[1, 1]] - u[[0, 1]] * u[[1, 0]])
        * (vt[[0, 0]] * vt[[1, 1]] - vt[[0, 1]] * vt[[1, 0]]);
    let sign = if det < 0.0 { -1.0 } else { 1.0 };

    let mut d_mat = Array2::<f64>::zeros((2, 2));
    d_mat[[0, 0]] = 1.0;
    d_mat[[1, 1]] = sign;

    let rot = vt.t().dot(&d_mat).dot(&u.t());
    let angle = rot[[1, 0]].atan2(rot[[0, 0]]);

    // Translation  t = tgt_mean - R * src_mean
    let rotated_mean = rot.dot(&src_mean);
    let tx = tgt_mean[0] - rotated_mean[0];
    let ty = tgt_mean[1] - rotated_mean[1];

    // Residual
    let transformed = src_centered.dot(&rot.t());
    let diff = &transformed - &tgt_centered;
    let mse = diff.mapv(|v| v * v).sum() / n as f64;

    Ok(RigidTransform2D {
        angle,
        tx,
        ty,
        residual: mse,
    })
}

/// Closed-form 2x2 SVD.
/// Returns (U, [s1, s2], V^T) such that A = U diag(s) V^T.
fn svd_2x2(a: f64, b: f64, c: f64, d: f64) -> (Array2<f64>, [f64; 2], Array2<f64>) {
    // Using the analytical formula for 2x2 SVD
    let s1_sq = (a * a + b * b + c * c + d * d) / 2.0;
    let det = a * d - b * c;
    let tmp =
        ((a * a + b * b - c * c - d * d).powi(2) + 4.0 * (a * c + b * d).powi(2)).sqrt() / 2.0;

    let sigma1 = (s1_sq + tmp).sqrt();
    let sigma2 = (s1_sq - tmp).max(0.0).sqrt();

    // A^T A eigenvalues are sigma^2
    let ata_00 = a * a + c * c;
    let ata_01 = a * b + c * d;
    let ata_11 = b * b + d * d;

    // Eigenvectors of A^T A -> columns of V
    let theta_v = if ata_01.abs() < 1e-15 {
        0.0
    } else {
        0.5 * (2.0 * ata_01).atan2(ata_00 - ata_11)
    };

    let mut vt = Array2::<f64>::zeros((2, 2));
    vt[[0, 0]] = theta_v.cos();
    vt[[0, 1]] = theta_v.sin();
    vt[[1, 0]] = -theta_v.sin();
    vt[[1, 1]] = theta_v.cos();

    // U columns from A V / sigma
    let mut u = Array2::<f64>::zeros((2, 2));
    if sigma1 > 1e-15 {
        u[[0, 0]] = (a * vt[[0, 0]] + b * vt[[0, 1]]) / sigma1;
        u[[1, 0]] = (c * vt[[0, 0]] + d * vt[[0, 1]]) / sigma1;
    } else {
        u[[0, 0]] = 1.0;
    }
    if sigma2 > 1e-15 {
        u[[0, 1]] = (a * vt[[1, 0]] + b * vt[[1, 1]]) / sigma2;
        u[[1, 1]] = (c * vt[[1, 0]] + d * vt[[1, 1]]) / sigma2;
    } else {
        // Choose orthogonal column
        u[[0, 1]] = -u[[1, 0]];
        u[[1, 1]] = u[[0, 0]];
    }

    (u, [sigma1, sigma2], vt)
}

// ---------------------------------------------------------------------------
// Iterative Closest Point (ICP)
// ---------------------------------------------------------------------------

/// Register `source` point set to `target` point set using ICP.
///
/// Both arrays have shape (N, 2) where each row is `[x, y]`.
/// The algorithm iteratively:
///   1. Finds closest target point for each source point
///   2. Computes the best rigid transform
///   3. Applies the transform
///   4. Checks convergence
pub fn icp_registration(
    source: &Array2<f64>,
    target: &Array2<f64>,
    config: Option<IcpConfig>,
) -> NdimageResult<IcpResult> {
    let cfg = config.unwrap_or_default();

    if source.ncols() != 2 || target.ncols() != 2 {
        return Err(NdimageError::InvalidInput(
            "Point arrays must have 2 columns".into(),
        ));
    }
    if source.nrows() < 2 || target.nrows() < 2 {
        return Err(NdimageError::InvalidInput(
            "Need at least 2 points in each set".into(),
        ));
    }

    let n_src = source.nrows();
    let mut current = source.to_owned();
    let mut cum_angle: f64 = 0.0;
    let mut cum_tx: f64 = 0.0;
    let mut cum_ty: f64 = 0.0;
    let mut mse_history = Vec::new();
    let mut converged = false;

    for iter in 0..cfg.max_iterations {
        // 1. Find correspondences (nearest target for each source)
        let (correspondences, mse) = find_correspondences(&current, target, cfg.max_distance)?;

        mse_history.push(mse);

        // Check convergence
        if iter > 0 {
            let prev = mse_history[iter - 1];
            if (prev - mse).abs() < cfg.tolerance {
                converged = true;
                break;
            }
        }

        if correspondences.is_empty() {
            return Err(NdimageError::ComputationError(
                "No valid correspondences found".into(),
            ));
        }

        // 2. Build matched point sets
        let n_match = correspondences.len();
        let mut src_matched = Array2::<f64>::zeros((n_match, 2));
        let mut tgt_matched = Array2::<f64>::zeros((n_match, 2));
        for (k, &(si, ti)) in correspondences.iter().enumerate() {
            src_matched[[k, 0]] = current[[si, 0]];
            src_matched[[k, 1]] = current[[si, 1]];
            tgt_matched[[k, 0]] = target[[ti, 0]];
            tgt_matched[[k, 1]] = target[[ti, 1]];
        }

        // 3. Compute best rigid transform
        let rigid = rigid_registration(&src_matched, &tgt_matched)?;

        // 4. Apply transform to all source points
        let cos_a = rigid.angle.cos();
        let sin_a = rigid.angle.sin();
        for k in 0..n_src {
            let x = current[[k, 0]];
            let y = current[[k, 1]];
            current[[k, 0]] = cos_a * x - sin_a * y + rigid.tx;
            current[[k, 1]] = sin_a * x + cos_a * y + rigid.ty;
        }

        // Accumulate transform
        let old_tx = cum_tx;
        let old_ty = cum_ty;
        let old_cos = cum_angle.cos();
        let old_sin = cum_angle.sin();
        cum_tx = cos_a * old_tx - sin_a * old_ty + rigid.tx;
        cum_ty = sin_a * old_tx + cos_a * old_ty + rigid.ty;
        cum_angle += rigid.angle;
    }

    let final_iters = mse_history.len();

    Ok(IcpResult {
        transform: RigidTransform2D {
            angle: cum_angle,
            tx: cum_tx,
            ty: cum_ty,
            residual: mse_history.last().copied().unwrap_or(f64::INFINITY),
        },
        iterations: final_iters,
        mse_history,
        converged,
    })
}

/// Find nearest-neighbor correspondences from `source` to `target`.
/// Returns pairs of (source_idx, target_idx) and the mean squared distance.
fn find_correspondences(
    source: &Array2<f64>,
    target: &Array2<f64>,
    max_dist: Option<f64>,
) -> NdimageResult<(Vec<(usize, usize)>, f64)> {
    let n_src = source.nrows();
    let n_tgt = target.nrows();
    let max_dist_sq = max_dist.map(|d| d * d);

    let mut pairs = Vec::with_capacity(n_src);
    let mut total_dist_sq = 0.0;

    for si in 0..n_src {
        let sx = source[[si, 0]];
        let sy = source[[si, 1]];

        let mut best_dist_sq = f64::INFINITY;
        let mut best_ti = 0usize;

        for ti in 0..n_tgt {
            let dx = sx - target[[ti, 0]];
            let dy = sy - target[[ti, 1]];
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_ti = ti;
            }
        }

        let accept = match max_dist_sq {
            Some(md2) => best_dist_sq <= md2,
            None => true,
        };

        if accept {
            pairs.push((si, best_ti));
            total_dist_sq += best_dist_sq;
        }
    }

    let mse = if pairs.is_empty() {
        f64::INFINITY
    } else {
        total_dist_sq / pairs.len() as f64
    };

    Ok((pairs, mse))
}

// ---------------------------------------------------------------------------
// Multi-resolution pyramid registration
// ---------------------------------------------------------------------------

/// Perform multi-resolution pyramid registration using phase correlation at
/// each level, refining from coarse to fine.
///
/// At the coarsest level the shift is estimated on heavily down-sampled images;
/// that estimate is propagated to the next finer level as an initial guess.
///
/// Returns the final sub-pixel translation estimate.
pub fn pyramid_registration(
    reference: &Array2<f64>,
    moving: &Array2<f64>,
    config: Option<PyramidConfig>,
) -> NdimageResult<TranslationResult> {
    let cfg = config.unwrap_or_default();
    let (ny, nx) = reference.dim();
    if moving.dim() != (ny, nx) {
        return Err(NdimageError::DimensionError(
            "Images must have the same shape for pyramid registration".into(),
        ));
    }
    if cfg.levels == 0 {
        return Err(NdimageError::InvalidInput(
            "Number of pyramid levels must be >= 1".into(),
        ));
    }
    if cfg.scale_factor <= 1.0 {
        return Err(NdimageError::InvalidInput(
            "Scale factor must be > 1.0".into(),
        ));
    }

    // Build pyramid by successive down-sampling
    let mut ref_pyramid = vec![reference.clone()];
    let mut mov_pyramid = vec![moving.clone()];
    for _ in 1..cfg.levels {
        let ref_prev = ref_pyramid
            .last()
            .ok_or_else(|| NdimageError::ComputationError("Empty pyramid".into()))?;
        let mov_prev = mov_pyramid
            .last()
            .ok_or_else(|| NdimageError::ComputationError("Empty pyramid".into()))?;
        ref_pyramid.push(downsample_2x(ref_prev));
        mov_pyramid.push(downsample_2x(mov_prev));
    }

    // Register coarse-to-fine (last element = coarsest)
    let mut cum_shift_y = 0.0;
    let mut cum_shift_x = 0.0;
    let mut best_peak = 0.0;

    for level in (0..cfg.levels).rev() {
        let ref_level = &ref_pyramid[level];
        let mov_level = &mov_pyramid[level];

        // If the image is too small, skip
        if ref_level.nrows() < 4 || ref_level.ncols() < 4 {
            continue;
        }

        let result = phase_correlation(ref_level, mov_level)?;

        if level == cfg.levels - 1 {
            // Coarsest level: use directly
            cum_shift_y = result.shift_y;
            cum_shift_x = result.shift_x;
        } else {
            // Refine: the coarser estimate is scaled up by 2
            cum_shift_y = cum_shift_y * 2.0 + result.shift_y;
            cum_shift_x = cum_shift_x * 2.0 + result.shift_x;
        }
        best_peak = result.peak_value;
    }

    Ok(TranslationResult {
        shift_y: cum_shift_y,
        shift_x: cum_shift_x,
        peak_value: best_peak,
    })
}

/// Simple 2x down-sampling by averaging 2x2 blocks.
fn downsample_2x(image: &Array2<f64>) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let out_ny = ny / 2;
    let out_nx = nx / 2;
    if out_ny == 0 || out_nx == 0 {
        return Array2::zeros((1.max(out_ny), 1.max(out_nx)));
    }

    let mut out = Array2::zeros((out_ny, out_nx));
    for i in 0..out_ny {
        for j in 0..out_nx {
            let ii = 2 * i;
            let jj = 2 * j;
            out[[i, j]] = (image[[ii, jj]]
                + image[[ii + 1, jj]]
                + image[[ii, jj + 1]]
                + image[[ii + 1, jj + 1]])
                / 4.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Registration quality metrics
// ---------------------------------------------------------------------------

/// Compute registration quality metrics.
///
/// * `source_landmarks` / `target_landmarks` are Nx2 arrays of corresponding
///   landmark points *before* and *after* registration of the source image.
/// * `reference` / `registered` are the reference and the source-after-
///   registration images (used for NCC and MI).
///
/// If landmark arrays are empty, TRE is returned as 0.
/// If image arrays are empty, NCC and MI are returned as 0.
pub fn registration_metrics(
    source_landmarks: Option<&Array2<f64>>,
    target_landmarks: Option<&Array2<f64>>,
    reference: Option<&Array2<f64>>,
    registered: Option<&Array2<f64>>,
) -> NdimageResult<RegistrationMetrics> {
    // TRE
    let tre = match (source_landmarks, target_landmarks) {
        (Some(src), Some(tgt)) => {
            if src.nrows() != tgt.nrows() {
                return Err(NdimageError::DimensionError(
                    "Landmark arrays must have the same number of rows".into(),
                ));
            }
            compute_tre(src, tgt)
        }
        _ => 0.0,
    };

    // NCC and MI
    let (ncc, mi) = match (reference, registered) {
        (Some(ref_img), Some(reg_img)) => {
            if ref_img.dim() != reg_img.dim() {
                return Err(NdimageError::DimensionError(
                    "Images must have the same shape for metric computation".into(),
                ));
            }
            let n = compute_ncc(ref_img, reg_img);
            let m = compute_mutual_information(ref_img, reg_img);
            (n, m)
        }
        _ => (0.0, 0.0),
    };

    Ok(RegistrationMetrics {
        tre,
        mutual_information: mi,
        ncc,
    })
}

/// Target Registration Error: RMS distance between corresponding landmarks.
fn compute_tre(transformed_src: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let n = transformed_src.nrows();
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for i in 0..n {
        let dx = transformed_src[[i, 0]] - target[[i, 0]];
        let dy = transformed_src[[i, 1]] - target[[i, 1]];
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / n as f64).sqrt()
}

/// Normalized Cross-Correlation between two images.
fn compute_ncc(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.len() as f64;
    if n < 1.0 {
        return 0.0;
    }
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;

    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;

    for (va, vb) in a.iter().zip(b.iter()) {
        let da = va - mean_a;
        let db = vb - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }

    let denom = (denom_a * denom_b).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

/// Estimate mutual information using a joint histogram with 64 bins.
fn compute_mutual_information(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n_bins = 64usize;

    // Find intensity ranges
    let (mut a_min, mut a_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut b_min, mut b_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&va, &vb) in a.iter().zip(b.iter()) {
        if va < a_min {
            a_min = va;
        }
        if va > a_max {
            a_max = va;
        }
        if vb < b_min {
            b_min = vb;
        }
        if vb > b_max {
            b_max = vb;
        }
    }

    let a_range = a_max - a_min;
    let b_range = b_max - b_min;
    if a_range < 1e-15 || b_range < 1e-15 {
        return 0.0;
    }

    // Build joint histogram
    let mut joint = vec![0usize; n_bins * n_bins];
    let n_total = a.len();
    let a_scale = (n_bins as f64 - 1e-10) / a_range;
    let b_scale = (n_bins as f64 - 1e-10) / b_range;

    for (&va, &vb) in a.iter().zip(b.iter()) {
        let ai = ((va - a_min) * a_scale) as usize;
        let bi = ((vb - b_min) * b_scale) as usize;
        let ai = ai.min(n_bins - 1);
        let bi = bi.min(n_bins - 1);
        joint[ai * n_bins + bi] += 1;
    }

    // Marginal histograms
    let mut hist_a = vec![0usize; n_bins];
    let mut hist_b = vec![0usize; n_bins];
    for ai in 0..n_bins {
        for bi in 0..n_bins {
            let c = joint[ai * n_bins + bi];
            hist_a[ai] += c;
            hist_b[bi] += c;
        }
    }

    // MI = sum p(a,b) * log(p(a,b) / (p(a)*p(b)))
    let n_f = n_total as f64;
    let mut mi = 0.0;
    for ai in 0..n_bins {
        for bi in 0..n_bins {
            let pab = joint[ai * n_bins + bi] as f64 / n_f;
            let pa = hist_a[ai] as f64 / n_f;
            let pb = hist_b[bi] as f64 / n_f;
            if pab > 1e-15 && pa > 1e-15 && pb > 1e-15 {
                mi += pab * (pab / (pa * pb)).ln();
            }
        }
    }
    mi
}

// ---------------------------------------------------------------------------
// Apply transform helpers
// ---------------------------------------------------------------------------

/// Apply an affine transform to a set of 2-D points.
/// Each row of `points` is [x, y].
pub fn apply_affine_to_points(
    points: &Array2<f64>,
    transform: &AffineTransform2D,
) -> NdimageResult<Array2<f64>> {
    if points.ncols() != 2 {
        return Err(NdimageError::InvalidInput(
            "Points must have 2 columns".into(),
        ));
    }
    let n = points.nrows();
    let m = &transform.matrix;
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        out[[i, 0]] = m[[0, 0]] * x + m[[0, 1]] * y + m[[0, 2]];
        out[[i, 1]] = m[[1, 0]] * x + m[[1, 1]] * y + m[[1, 2]];
    }
    Ok(out)
}

/// Apply a rigid transform to a set of 2-D points.
pub fn apply_rigid_to_points(
    points: &Array2<f64>,
    transform: &RigidTransform2D,
) -> NdimageResult<Array2<f64>> {
    if points.ncols() != 2 {
        return Err(NdimageError::InvalidInput(
            "Points must have 2 columns".into(),
        ));
    }
    let n = points.nrows();
    let cos_a = transform.angle.cos();
    let sin_a = transform.angle.sin();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        out[[i, 0]] = cos_a * x - sin_a * y + transform.tx;
        out[[i, 1]] = sin_a * x + cos_a * y + transform.ty;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_phase_correlation_no_shift() {
        let img = Array2::from_shape_fn((32, 32), |(i, j)| {
            ((i as f64 * 0.3).sin() + (j as f64 * 0.5).cos()) * 10.0
        });
        let result = phase_correlation(&img, &img).expect("phase_correlation failed");
        assert!(
            result.shift_y.abs() < 1.0,
            "shift_y should be ~0, got {}",
            result.shift_y
        );
        assert!(
            result.shift_x.abs() < 1.0,
            "shift_x should be ~0, got {}",
            result.shift_x
        );
    }

    #[test]
    fn test_phase_correlation_known_shift() {
        // Create reference and shifted version
        let ny = 64;
        let nx = 64;
        let reference = Array2::from_shape_fn((ny, nx), |(i, j)| {
            ((i as f64 / 8.0).sin() * (j as f64 / 8.0).cos()) * 100.0
        });
        // Shift by (3, 5) via circular shift
        let mut moved = Array2::zeros((ny, nx));
        for i in 0..ny {
            for j in 0..nx {
                moved[[(i + 3) % ny, (j + 5) % nx]] = reference[[i, j]];
            }
        }
        let result = phase_correlation(&reference, &moved).expect("phase_correlation failed");
        assert!(
            (result.shift_y - 3.0).abs() < 1.5,
            "shift_y ~ 3, got {}",
            result.shift_y
        );
        assert!(
            (result.shift_x - 5.0).abs() < 1.5,
            "shift_x ~ 5, got {}",
            result.shift_x
        );
    }

    #[test]
    fn test_affine_registration_identity() {
        let pts = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("shape error");
        let result = affine_registration(&pts, &pts).expect("affine_registration failed");
        // Should be close to identity
        assert!((result.matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result.matrix[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(result.residual < 1e-10);
    }

    #[test]
    fn test_affine_registration_translation() {
        let src = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("shape error");
        let tgt = Array2::from_shape_vec((4, 2), vec![3.0, 2.0, 4.0, 2.0, 3.0, 3.0, 4.0, 3.0])
            .expect("shape error");
        let result = affine_registration(&src, &tgt).expect("affine_registration failed");
        assert!((result.matrix[[0, 2]] - 3.0).abs() < 1e-8, "tx ~ 3");
        assert!((result.matrix[[1, 2]] - 2.0).abs() < 1e-8, "ty ~ 2");
    }

    #[test]
    fn test_rigid_registration_identity() {
        let pts = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("shape error");
        let result = rigid_registration(&pts, &pts).expect("rigid_registration failed");
        assert!(result.angle.abs() < 1e-8);
        assert!(result.tx.abs() < 1e-8);
        assert!(result.ty.abs() < 1e-8);
    }

    #[test]
    fn test_rigid_registration_translation() {
        let src = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("shape error");
        let tgt = Array2::from_shape_vec((4, 2), vec![5.0, 3.0, 6.0, 3.0, 5.0, 4.0, 6.0, 4.0])
            .expect("shape error");
        let result = rigid_registration(&src, &tgt).expect("rigid_registration failed");
        assert!(
            result.angle.abs() < 1e-8,
            "no rotation expected, got {}",
            result.angle
        );
        assert!((result.tx - 5.0).abs() < 1e-6, "tx ~ 5, got {}", result.tx);
        assert!((result.ty - 3.0).abs() < 1e-6, "ty ~ 3, got {}", result.ty);
    }

    #[test]
    fn test_rigid_registration_rotation() {
        let angle = PI / 6.0; // 30 degrees
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let src = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
            .expect("shape error");
        // Rotate source by 30 degrees around origin
        let mut tgt = Array2::zeros((4, 2));
        for i in 0..4 {
            let x = src[[i, 0]];
            let y = src[[i, 1]];
            tgt[[i, 0]] = cos_a * x - sin_a * y;
            tgt[[i, 1]] = sin_a * x + cos_a * y;
        }
        let result = rigid_registration(&src, &tgt).expect("rigid_registration failed");
        assert!(
            (result.angle - angle).abs() < 1e-6,
            "angle ~ pi/6, got {}",
            result.angle
        );
    }

    #[test]
    fn test_icp_registration() {
        // Use well-spaced points with a SMALL shift relative to inter-point distance
        // so that nearest-neighbor correspondences are correct from the start.
        let src = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 10.0, 0.0, 20.0, 10.0,
                20.0, 20.0, 20.0,
            ],
        )
        .expect("shape error");
        let mut tgt = src.clone();
        // Small translation (well below half the inter-point distance of 10)
        let shift_x = 1.5;
        let shift_y = 2.0;
        for i in 0..tgt.nrows() {
            tgt[[i, 0]] += shift_x;
            tgt[[i, 1]] += shift_y;
        }

        let result = icp_registration(&src, &tgt, None).expect("icp failed");
        assert!(
            (result.transform.tx - shift_x).abs() < 0.5,
            "tx ~ {}, got {}",
            shift_x,
            result.transform.tx
        );
        assert!(
            (result.transform.ty - shift_y).abs() < 0.5,
            "ty ~ {}, got {}",
            shift_y,
            result.transform.ty
        );
        assert!(result.converged, "ICP should converge");
    }

    #[test]
    fn test_pyramid_registration_no_shift() {
        let img = Array2::from_shape_fn((64, 64), |(i, j)| {
            ((i as f64 / 10.0).sin() + (j as f64 / 10.0).cos()) * 50.0
        });
        let result = pyramid_registration(&img, &img, None).expect("pyramid failed");
        assert!(
            result.shift_y.abs() < 2.0,
            "shift_y ~ 0, got {}",
            result.shift_y
        );
        assert!(
            result.shift_x.abs() < 2.0,
            "shift_x ~ 0, got {}",
            result.shift_x
        );
    }

    #[test]
    fn test_registration_metrics_perfect() {
        let pts = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape error");
        let metrics =
            registration_metrics(Some(&pts), Some(&pts), None, None).expect("metrics failed");
        assert!(
            metrics.tre < 1e-10,
            "TRE should be 0 for identical landmarks"
        );
    }

    #[test]
    fn test_registration_metrics_ncc() {
        let img = Array2::from_shape_fn((16, 16), |(i, j)| (i + j) as f64);
        let metrics =
            registration_metrics(None, None, Some(&img), Some(&img)).expect("metrics failed");
        assert!(
            (metrics.ncc - 1.0).abs() < 1e-10,
            "NCC should be 1 for identical images"
        );
    }

    #[test]
    fn test_registration_metrics_mi() {
        let img = Array2::from_shape_fn((32, 32), |(i, j)| (i * j) as f64);
        let metrics =
            registration_metrics(None, None, Some(&img), Some(&img)).expect("metrics failed");
        // MI should be positive for identical images
        assert!(metrics.mutual_information > 0.0, "MI should be positive");
    }

    #[test]
    fn test_apply_affine_to_points() {
        let pts = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape error");
        let mut mat = Array2::<f64>::zeros((3, 3));
        mat[[0, 0]] = 1.0;
        mat[[1, 1]] = 1.0;
        mat[[0, 2]] = 10.0; // translate x by 10
        mat[[1, 2]] = 20.0; // translate y by 20
        mat[[2, 2]] = 1.0;
        let tf = AffineTransform2D {
            matrix: mat,
            residual: 0.0,
        };
        let result = apply_affine_to_points(&pts, &tf).expect("apply affine failed");
        assert!((result[[0, 0]] - 11.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 20.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 10.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_rigid_to_points() {
        let pts = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).expect("shape error");
        let tf = RigidTransform2D {
            angle: PI / 2.0,
            tx: 0.0,
            ty: 0.0,
            residual: 0.0,
        };
        let result = apply_rigid_to_points(&pts, &tf).expect("apply rigid failed");
        assert!(result[[0, 0]].abs() < 1e-10, "x ~ 0 after 90-deg rotation");
        assert!(
            (result[[0, 1]] - 1.0).abs() < 1e-10,
            "y ~ 1 after 90-deg rotation"
        );
    }

    #[test]
    fn test_downsample_2x() {
        let img = Array2::from_shape_fn((8, 8), |(i, j)| (i * 8 + j) as f64);
        let ds = downsample_2x(&img);
        assert_eq!(ds.dim(), (4, 4));
        // Top-left 2x2 block: 0, 1, 8, 9 -> avg = 4.5
        assert!((ds[[0, 0]] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_phase_correlation_dimension_mismatch() {
        let a = Array2::zeros((10, 10));
        let b = Array2::zeros((10, 12));
        assert!(phase_correlation(&a, &b).is_err());
    }

    #[test]
    fn test_affine_too_few_points() {
        let src = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("shape");
        let tgt = src.clone();
        assert!(affine_registration(&src, &tgt).is_err());
    }

    #[test]
    fn test_rigid_too_few_points() {
        let src = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("shape");
        let tgt = src.clone();
        assert!(rigid_registration(&src, &tgt).is_err());
    }
}
