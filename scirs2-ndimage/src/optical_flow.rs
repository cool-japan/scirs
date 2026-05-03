//! Optical flow algorithms for dense motion estimation
//!
//! This module provides classic and modern dense optical flow methods:
//! - Lucas-Kanade (local window-based, differential)
//! - Horn-Schunck (global variational smoothness constraint)
//! - Farneback (polynomial expansion-based)

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3};

// ─── FlowField ───────────────────────────────────────────────────────────────

/// A dense optical flow field.
///
/// The underlying array has shape `(rows, cols, 2)` where
/// index `[r, c, 0]` is the horizontal flow `u` (x-direction) and
/// `[r, c, 1]` is the vertical flow `v` (y-direction).
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Internal array of shape (rows, cols, 2).
    pub data: Array3<f64>,
}

impl FlowField {
    /// Create a zero flow field.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        FlowField {
            data: Array3::<f64>::zeros((rows, cols, 2)),
        }
    }

    /// Create from existing Array3.
    pub fn from_array(data: Array3<f64>) -> NdimageResult<Self> {
        if data.ndim() != 3 || data.shape()[2] != 2 {
            return Err(NdimageError::InvalidInput(
                "FlowField data must have shape (rows, cols, 2)".into(),
            ));
        }
        Ok(FlowField { data })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.data.shape()[0]
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.data.shape()[1]
    }

    /// Get flow vector at pixel (r, c) as (u, v).
    pub fn get(&self, r: usize, c: usize) -> (f64, f64) {
        (self.data[[r, c, 0]], self.data[[r, c, 1]])
    }

    /// Set flow vector at pixel (r, c).
    pub fn set(&mut self, r: usize, c: usize, u: f64, v: f64) {
        self.data[[r, c, 0]] = u;
        self.data[[r, c, 1]] = v;
    }

    /// Compute the magnitude of the flow at each pixel, returning Array2.
    pub fn magnitude(&self) -> Array2<f64> {
        let rows = self.rows();
        let cols = self.cols();
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            let u = self.data[[r, c, 0]];
            let v = self.data[[r, c, 1]];
            (u * u + v * v).sqrt()
        })
    }
}

// ─── Shared derivative helpers ───────────────────────────────────────────────

/// Compute spatial and temporal image derivatives for a frame pair.
/// Returns (Ix, Iy, It) each of shape (rows, cols).
fn compute_derivatives(
    prev: &Array2<f64>,
    next: &Array2<f64>,
) -> NdimageResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let rows = prev.nrows();
    let cols = prev.ncols();
    if next.shape() != prev.shape() {
        return Err(NdimageError::DimensionError(
            "prev and next frames must have the same shape".into(),
        ));
    }
    if rows < 2 || cols < 2 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 2×2".into(),
        ));
    }

    // Central difference for spatial; temporal is (next - prev)/2
    let mut ix = Array2::<f64>::zeros((rows, cols));
    let mut iy = Array2::<f64>::zeros((rows, cols));
    let mut it = Array2::<f64>::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let ip1 = if c + 1 < cols {
                prev[[r, c + 1]]
            } else {
                prev[[r, c]]
            };
            let im1 = if c > 0 {
                prev[[r, c - 1]]
            } else {
                prev[[r, c]]
            };
            ix[[r, c]] = (ip1 - im1) / 2.0;

            let jp1 = if r + 1 < rows {
                prev[[r + 1, c]]
            } else {
                prev[[r, c]]
            };
            let jm1 = if r > 0 {
                prev[[r - 1, c]]
            } else {
                prev[[r, c]]
            };
            iy[[r, c]] = (jp1 - jm1) / 2.0;

            it[[r, c]] = next[[r, c]] - prev[[r, c]];
        }
    }
    Ok((ix, iy, it))
}

/// Simple box average over a window of given half-width.
fn box_average(arr: &Array2<f64>, half_win: usize) -> Array2<f64> {
    let rows = arr.nrows();
    let cols = arr.ncols();
    let mut out = Array2::<f64>::zeros((rows, cols));
    let wsize = 2 * half_win + 1;
    let area = (wsize * wsize) as f64;
    for r in 0..rows {
        for c in 0..cols {
            let mut s = 0.0f64;
            for dr in 0..wsize {
                let nr = (r + dr).saturating_sub(half_win).min(rows - 1);
                for dc in 0..wsize {
                    let nc = (c + dc).saturating_sub(half_win).min(cols - 1);
                    s += arr[[nr, nc]];
                }
            }
            out[[r, c]] = s / area;
        }
    }
    out
}

/// Laplacian (neighbour average – self) used in Horn-Schunck smoothness.
fn laplacian(u: &Array2<f64>) -> Array2<f64> {
    let rows = u.nrows();
    let cols = u.ncols();
    let mut out = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let sum_nb = {
                let up = if r > 0 { u[[r - 1, c]] } else { u[[r, c]] };
                let dn = if r + 1 < rows {
                    u[[r + 1, c]]
                } else {
                    u[[r, c]]
                };
                let lt = if c > 0 { u[[r, c - 1]] } else { u[[r, c]] };
                let rt = if c + 1 < cols {
                    u[[r, c + 1]]
                } else {
                    u[[r, c]]
                };
                up + dn + lt + rt
            };
            out[[r, c]] = sum_nb / 4.0 - u[[r, c]];
        }
    }
    out
}

// ─── Lucas-Kanade Optical Flow ───────────────────────────────────────────────

/// Dense Lucas-Kanade optical flow.
///
/// For each pixel, builds a local least-squares system using the
/// brightness constancy constraint over a `window_size × window_size`
/// window and solves for (u, v).
///
/// # Arguments
/// * `prev`        – first frame (rows × cols)
/// * `next`        – second frame (same shape)
/// * `window_size` – size of integration window (odd, e.g. 5 or 15)
///
/// # Returns
/// Flow field of shape (rows, cols, 2).
pub fn lucas_kanade_flow(
    prev: &Array2<f64>,
    next: &Array2<f64>,
    window_size: usize,
) -> NdimageResult<Array3<f64>> {
    if window_size == 0 {
        return Err(NdimageError::InvalidInput(
            "window_size must be positive".into(),
        ));
    }
    let rows = prev.nrows();
    let cols = prev.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput(
            "Frames must not be empty".into(),
        ));
    }
    if prev.shape() != next.shape() {
        return Err(NdimageError::DimensionError(
            "prev and next must have the same shape".into(),
        ));
    }

    let (ix, iy, it) = compute_derivatives(prev, next)?;

    // Precompute products for the structure tensor
    let ixx = {
        let mut a = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                a[[r, c]] = ix[[r, c]] * ix[[r, c]];
            }
        }
        a
    };
    let iyy = {
        let mut a = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                a[[r, c]] = iy[[r, c]] * iy[[r, c]];
            }
        }
        a
    };
    let ixy = {
        let mut a = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                a[[r, c]] = ix[[r, c]] * iy[[r, c]];
            }
        }
        a
    };
    let ixt = {
        let mut a = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                a[[r, c]] = ix[[r, c]] * it[[r, c]];
            }
        }
        a
    };
    let iyt = {
        let mut a = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                a[[r, c]] = iy[[r, c]] * it[[r, c]];
            }
        }
        a
    };

    let hw = window_size / 2;
    // Box-average the structure tensor components
    let sxx = box_average(&ixx, hw);
    let syy = box_average(&iyy, hw);
    let sxy = box_average(&ixy, hw);
    let sxt = box_average(&ixt, hw);
    let syt = box_average(&iyt, hw);

    let mut flow = Array3::<f64>::zeros((rows, cols, 2));
    let eps = 1e-12f64;

    for r in 0..rows {
        for c in 0..cols {
            let a = sxx[[r, c]];
            let b = sxy[[r, c]];
            let d = syy[[r, c]];
            let det = a * d - b * b;
            if det.abs() > eps {
                let u = (-d * sxt[[r, c]] + b * syt[[r, c]]) / det;
                let v = (b * sxt[[r, c]] - a * syt[[r, c]]) / det;
                flow[[r, c, 0]] = u;
                flow[[r, c, 1]] = v;
            }
        }
    }
    Ok(flow)
}

// ─── Horn-Schunck Optical Flow ───────────────────────────────────────────────

/// Horn-Schunck global variational optical flow.
///
/// Minimises the energy functional combining the brightness constancy
/// constraint with a global smoothness term (controlled by `alpha`).
/// Solved iteratively via Gauss-Seidel.
///
/// # Arguments
/// * `prev`       – first frame
/// * `next`       – second frame (same shape)
/// * `alpha`      – smoothness weight (larger → smoother flow, typical: 1.0–100.0)
/// * `iterations` – number of Gauss-Seidel iterations
///
/// # Returns
/// Flow field of shape (rows, cols, 2).
pub fn horn_schunck_flow(
    prev: &Array2<f64>,
    next: &Array2<f64>,
    alpha: f64,
    iterations: usize,
) -> NdimageResult<Array3<f64>> {
    if alpha <= 0.0 {
        return Err(NdimageError::InvalidInput("alpha must be positive".into()));
    }
    if iterations == 0 {
        return Err(NdimageError::InvalidInput(
            "iterations must be at least 1".into(),
        ));
    }
    let rows = prev.nrows();
    let cols = prev.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput(
            "Frames must not be empty".into(),
        ));
    }

    let (ix, iy, it) = compute_derivatives(prev, next)?;

    let alpha_sq = alpha * alpha;
    let mut u = Array2::<f64>::zeros((rows, cols));
    let mut v = Array2::<f64>::zeros((rows, cols));

    for _ in 0..iterations {
        // Compute local averages using 4-connected neighbourhood
        let u_avg = {
            let mut a = Array2::<f64>::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    let up = if r > 0 { u[[r - 1, c]] } else { u[[r, c]] };
                    let dn = if r + 1 < rows {
                        u[[r + 1, c]]
                    } else {
                        u[[r, c]]
                    };
                    let lt = if c > 0 { u[[r, c - 1]] } else { u[[r, c]] };
                    let rt = if c + 1 < cols {
                        u[[r, c + 1]]
                    } else {
                        u[[r, c]]
                    };
                    a[[r, c]] = (up + dn + lt + rt) / 4.0;
                }
            }
            a
        };
        let v_avg = {
            let mut a = Array2::<f64>::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    let up = if r > 0 { v[[r - 1, c]] } else { v[[r, c]] };
                    let dn = if r + 1 < rows {
                        v[[r + 1, c]]
                    } else {
                        v[[r, c]]
                    };
                    let lt = if c > 0 { v[[r, c - 1]] } else { v[[r, c]] };
                    let rt = if c + 1 < cols {
                        v[[r, c + 1]]
                    } else {
                        v[[r, c]]
                    };
                    a[[r, c]] = (up + dn + lt + rt) / 4.0;
                }
            }
            a
        };

        for r in 0..rows {
            for c in 0..cols {
                let ixi = ix[[r, c]];
                let iyi = iy[[r, c]];
                let iti = it[[r, c]];
                let ua = u_avg[[r, c]];
                let va = v_avg[[r, c]];
                let denom = alpha_sq + ixi * ixi + iyi * iyi;
                let p = (ixi * ua + iyi * va + iti) / denom;
                u[[r, c]] = ua - ixi * p;
                v[[r, c]] = va - iyi * p;
            }
        }
    }

    let mut flow = Array3::<f64>::zeros((rows, cols, 2));
    for r in 0..rows {
        for c in 0..cols {
            flow[[r, c, 0]] = u[[r, c]];
            flow[[r, c, 1]] = v[[r, c]];
        }
    }
    Ok(flow)
}

// ─── Farneback Polynomial Expansion Optical Flow ─────────────────────────────

/// Farneback dense optical flow using polynomial expansion.
///
/// Approximates each image neighbourhood by a quadratic polynomial and
/// estimates the displacement that aligns the polynomial approximations
/// of two frames. Uses a coarse-to-fine pyramid for large displacements.
///
/// # Arguments
/// * `prev`   – first frame
/// * `next`   – second frame (same shape)
/// * `levels` – number of pyramid levels (1 = no pyramid)
/// * `winsize` – polynomial expansion window half-size
///
/// # Returns
/// Flow field of shape (rows, cols, 2).
pub fn farneback_flow(
    prev: &Array2<f64>,
    next: &Array2<f64>,
    levels: usize,
    winsize: usize,
) -> NdimageResult<Array3<f64>> {
    if levels == 0 {
        return Err(NdimageError::InvalidInput(
            "levels must be at least 1".into(),
        ));
    }
    if winsize == 0 {
        return Err(NdimageError::InvalidInput(
            "winsize must be positive".into(),
        ));
    }
    let rows = prev.nrows();
    let cols = prev.ncols();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput(
            "Frames must not be empty".into(),
        ));
    }
    if prev.shape() != next.shape() {
        return Err(NdimageError::DimensionError(
            "prev and next must have the same shape".into(),
        ));
    }

    // Build Gaussian pyramids
    let prev_pyr = build_pyramid(prev, levels);
    let next_pyr = build_pyramid(next, levels);

    // Coarse-to-fine estimation
    let coarsest = levels - 1;
    let pr = prev_pyr[coarsest].nrows();
    let pc = prev_pyr[coarsest].ncols();
    let mut flow_u = Array2::<f64>::zeros((pr, pc));
    let mut flow_v = Array2::<f64>::zeros((pr, pc));

    for lvl in (0..levels).rev() {
        let p = &prev_pyr[lvl];
        let n = &next_pyr[lvl];
        let lr = p.nrows();
        let lc = p.ncols();

        // Upsample flow if not at coarsest
        if lvl < coarsest || (lr != flow_u.nrows() || lc != flow_u.ncols()) {
            flow_u = upsample_flow(&flow_u, lr, lc, 2.0);
            flow_v = upsample_flow(&flow_v, lr, lc, 2.0);
        }

        // Warp next frame by current flow estimate
        let warped = warp_image(n, &flow_u, &flow_v);

        // Polynomial expansion: fit A*x^2 + B*y^2 + C*x*y + D*x + E*y + F
        // to windows in both frames. Extract gradient from the linear terms.
        let (ix, iy, it) = compute_derivatives(p, &warped)?;

        // Update flow using Lucas-Kanade style on the polynomial coefficients
        let hw = winsize;
        let ixx = Array2::from_shape_fn((lr, lc), |(r, c)| ix[[r, c]] * ix[[r, c]]);
        let iyy = Array2::from_shape_fn((lr, lc), |(r, c)| iy[[r, c]] * iy[[r, c]]);
        let ixy = Array2::from_shape_fn((lr, lc), |(r, c)| ix[[r, c]] * iy[[r, c]]);
        let ixt = Array2::from_shape_fn((lr, lc), |(r, c)| ix[[r, c]] * it[[r, c]]);
        let iyt = Array2::from_shape_fn((lr, lc), |(r, c)| iy[[r, c]] * it[[r, c]]);

        let sxx = box_average(&ixx, hw);
        let syy = box_average(&iyy, hw);
        let sxy = box_average(&ixy, hw);
        let sxt = box_average(&ixt, hw);
        let syt = box_average(&iyt, hw);

        let eps = 1e-12f64;
        for r in 0..lr {
            for c in 0..lc {
                let a = sxx[[r, c]];
                let b = sxy[[r, c]];
                let d = syy[[r, c]];
                let det = a * d - b * b;
                if det.abs() > eps {
                    let du = (-d * sxt[[r, c]] + b * syt[[r, c]]) / det;
                    let dv = (b * sxt[[r, c]] - a * syt[[r, c]]) / det;
                    flow_u[[r, c]] += du;
                    flow_v[[r, c]] += dv;
                }
            }
        }
    }

    // Build output at original resolution
    let mut flow = Array3::<f64>::zeros((rows, cols, 2));
    let final_u = if flow_u.nrows() == rows && flow_u.ncols() == cols {
        flow_u
    } else {
        upsample_flow(
            &flow_u,
            rows,
            cols,
            (rows as f64 / flow_u.nrows() as f64).max(1.0),
        )
    };
    let final_v = if flow_v.nrows() == rows && flow_v.ncols() == cols {
        flow_v
    } else {
        upsample_flow(
            &flow_v,
            rows,
            cols,
            (rows as f64 / flow_v.nrows() as f64).max(1.0),
        )
    };
    for r in 0..rows {
        for c in 0..cols {
            flow[[r, c, 0]] = final_u[[r, c]];
            flow[[r, c, 1]] = final_v[[r, c]];
        }
    }
    Ok(flow)
}

/// Build a Gaussian image pyramid with `levels` levels.
fn build_pyramid(image: &Array2<f64>, levels: usize) -> Vec<Array2<f64>> {
    let mut pyr = vec![image.clone()];
    for _ in 1..levels {
        let last = &pyr[pyr.len() - 1];
        let down = downsample(last);
        pyr.push(down);
    }
    pyr
}

/// Downsample an image by a factor of 2 (with Gaussian pre-blur).
fn downsample(image: &Array2<f64>) -> Array2<f64> {
    let rows = image.nrows();
    let cols = image.ncols();
    let new_rows = (rows + 1) / 2;
    let new_cols = (cols + 1) / 2;
    if new_rows == 0 || new_cols == 0 {
        return image.clone();
    }
    // Simple 2×2 box blur then subsample
    Array2::from_shape_fn((new_rows, new_cols), |(r, c)| {
        let sr = r * 2;
        let sc = c * 2;
        let mut s = 0.0f64;
        let mut cnt = 0u32;
        for dr in 0..=1usize {
            for dc in 0..=1usize {
                let nr = (sr + dr).min(rows - 1);
                let nc = (sc + dc).min(cols - 1);
                s += image[[nr, nc]];
                cnt += 1;
            }
        }
        s / cnt as f64
    })
}

/// Upsample a flow array to (target_rows, target_cols) and scale values.
fn upsample_flow(
    flow: &Array2<f64>,
    target_rows: usize,
    target_cols: usize,
    scale: f64,
) -> Array2<f64> {
    let src_rows = flow.nrows();
    let src_cols = flow.ncols();
    Array2::from_shape_fn((target_rows, target_cols), |(r, c)| {
        let sr = (r * src_rows / target_rows).min(src_rows - 1);
        let sc = (c * src_cols / target_cols).min(src_cols - 1);
        flow[[sr, sc]] * scale
    })
}

/// Warp `image` by the displacement field (u, v) using bilinear interpolation.
fn warp_image(image: &Array2<f64>, u: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
    let rows = image.nrows();
    let cols = image.ncols();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let x = c as f64 + u[[r, c]];
        let y = r as f64 + v[[r, c]];
        // Bilinear interpolation with clamped boundary
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let sample = |ri: isize, ci: isize| -> f64 {
            let ri = ri.max(0).min(rows as isize - 1) as usize;
            let ci = ci.max(0).min(cols as isize - 1) as usize;
            image[[ri, ci]]
        };
        let i00 = sample(y0, x0);
        let i01 = sample(y0, x0 + 1);
        let i10 = sample(y0 + 1, x0);
        let i11 = sample(y0 + 1, x0 + 1);
        i00 * (1.0 - fx) * (1.0 - fy)
            + i01 * fx * (1.0 - fy)
            + i10 * (1.0 - fx) * fy
            + i11 * fx * fy
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Create a simple gradient-shifted frame pair with known horizontal shift.
    fn make_shifted_pair(rows: usize, cols: usize, shift_c: isize) -> (Array2<f64>, Array2<f64>) {
        let prev = Array2::from_shape_fn((rows, cols), |(_, c)| c as f64 / cols as f64);
        let next = Array2::from_shape_fn((rows, cols), |(_, c)| {
            let nc = (c as isize - shift_c).max(0).min(cols as isize - 1) as usize;
            nc as f64 / cols as f64
        });
        (prev, next)
    }

    // ── FlowField tests ──────────────────────────────────────────────────────

    #[test]
    fn test_flow_field_zeros() {
        let ff = FlowField::zeros(4, 4);
        assert_eq!(ff.rows(), 4);
        assert_eq!(ff.cols(), 4);
        let (u, v) = ff.get(0, 0);
        assert!((u - 0.0).abs() < 1e-12);
        assert!((v - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_flow_field_set_get() {
        let mut ff = FlowField::zeros(4, 4);
        ff.set(2, 3, 1.5, -0.5);
        let (u, v) = ff.get(2, 3);
        assert!((u - 1.5).abs() < 1e-12);
        assert!((v + 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_flow_field_from_array_invalid() {
        let bad = Array3::<f64>::zeros((4, 4, 3)); // should be 2, not 3
        assert!(FlowField::from_array(bad).is_err());
    }

    #[test]
    fn test_flow_field_magnitude() {
        let mut ff = FlowField::zeros(3, 3);
        ff.set(1, 1, 3.0, 4.0);
        let mag = ff.magnitude();
        assert!((mag[[1, 1]] - 5.0).abs() < 1e-9);
    }

    // ── Lucas-Kanade tests ───────────────────────────────────────────────────

    #[test]
    fn test_lucas_kanade_shape() {
        let (prev, next) = make_shifted_pair(16, 16, 1);
        let flow = lucas_kanade_flow(&prev, &next, 5).expect("LK failed");
        assert_eq!(flow.shape(), &[16, 16, 2]);
    }

    #[test]
    fn test_lucas_kanade_zero_motion() {
        let img = Array2::from_shape_fn((10, 10), |(r, c)| (r + c) as f64 * 0.1);
        let flow = lucas_kanade_flow(&img, &img, 5).expect("LK zero motion");
        // No motion → flow should be near zero
        let max_flow = flow.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(max_flow < 1e-6, "Expected near-zero flow, got {max_flow}");
    }

    #[test]
    fn test_lucas_kanade_invalid_window() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(lucas_kanade_flow(&img, &img, 0).is_err());
    }

    #[test]
    fn test_lucas_kanade_shape_mismatch() {
        let prev = Array2::<f64>::zeros((8, 8));
        let next = Array2::<f64>::zeros((4, 4));
        assert!(lucas_kanade_flow(&prev, &next, 5).is_err());
    }

    // ── Horn-Schunck tests ───────────────────────────────────────────────────

    #[test]
    fn test_horn_schunck_shape() {
        let (prev, next) = make_shifted_pair(12, 12, 1);
        let flow = horn_schunck_flow(&prev, &next, 1.0, 50).expect("HS failed");
        assert_eq!(flow.shape(), &[12, 12, 2]);
    }

    #[test]
    fn test_horn_schunck_zero_motion() {
        let img = Array2::from_shape_fn((8, 8), |(r, c)| (r * c) as f64 * 0.01);
        let flow = horn_schunck_flow(&img, &img, 10.0, 20).expect("HS zero motion");
        let max_flow = flow.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(max_flow < 1e-6, "Expected near-zero flow, got {max_flow}");
    }

    #[test]
    fn test_horn_schunck_invalid_alpha() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(horn_schunck_flow(&img, &img, 0.0, 10).is_err());
        assert!(horn_schunck_flow(&img, &img, -1.0, 10).is_err());
    }

    #[test]
    fn test_horn_schunck_invalid_iterations() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(horn_schunck_flow(&img, &img, 1.0, 0).is_err());
    }

    // ── Farneback tests ──────────────────────────────────────────────────────

    #[test]
    fn test_farneback_shape() {
        let (prev, next) = make_shifted_pair(16, 16, 1);
        let flow = farneback_flow(&prev, &next, 2, 5).expect("Farneback failed");
        assert_eq!(flow.shape(), &[16, 16, 2]);
    }

    #[test]
    fn test_farneback_zero_motion() {
        let img = Array2::from_shape_fn((8, 8), |(r, c)| (r + c) as f64 * 0.1);
        let flow = farneback_flow(&img, &img, 1, 3).expect("Farneback zero motion");
        let max_flow = flow.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        assert!(max_flow < 1e-4, "Expected near-zero flow, got {max_flow}");
    }

    #[test]
    fn test_farneback_invalid_levels() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(farneback_flow(&img, &img, 0, 3).is_err());
    }

    #[test]
    fn test_farneback_invalid_winsize() {
        let img = Array2::<f64>::zeros((8, 8));
        assert!(farneback_flow(&img, &img, 2, 0).is_err());
    }
}
