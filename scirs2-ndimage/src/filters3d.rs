//! 3D Image Filters
//!
//! This module provides a collection of three-dimensional image filters for
//! volumetric data, including linear filters (Gaussian, uniform, Sobel,
//! Laplacian) and non-linear filters (median, bilateral, non-local means,
//! anisotropic diffusion).
//!
//! # Design Notes
//!
//! * All filters operate on `f64` arrays for precision.
//! * Border handling uses "clamp to edge" (nearest neighbor) by default.
//! * The 3D Gaussian filter is implemented as separable 1D convolutions
//!   applied successively along z, y, and x axes.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array1, Array3, ArrayView3};

// ---------------------------------------------------------------------------
// Border handling helpers
// ---------------------------------------------------------------------------

/// Clamp index to `[0, max_idx]`.
#[inline]
fn clamp_idx(i: isize, max_idx: usize) -> usize {
    i.clamp(0, max_idx as isize) as usize
}

/// Read a voxel with clamp-to-edge border handling.
#[inline]
fn get_clamped(img: &Array3<f64>, z: isize, y: isize, x: isize) -> f64 {
    let shape = img.shape();
    let sz = clamp_idx(z, shape[0].saturating_sub(1));
    let sy = clamp_idx(y, shape[1].saturating_sub(1));
    let sx = clamp_idx(x, shape[2].saturating_sub(1));
    img[[sz, sy, sx]]
}

// ---------------------------------------------------------------------------
// 1D Gaussian kernel
// ---------------------------------------------------------------------------

/// Build a 1D Gaussian kernel of half-width `radius = (truncate * sigma).ceil()`.
fn gaussian_kernel1d(sigma: f64, truncate: f64) -> Array1<f64> {
    let radius = (truncate * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut k = Array1::zeros(size);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..size {
        let x = i as f64 - radius as f64;
        let v = (-x * x / two_sigma_sq).exp();
        k[i] = v;
        sum += v;
    }
    k.mapv_inplace(|v| v / sum);
    k
}

// ---------------------------------------------------------------------------
// Separable 1D convolution helpers
// ---------------------------------------------------------------------------

/// Apply a 1D kernel along the Z axis of a 3D array (clamp border).
fn convolve_z(src: &Array3<f64>, kernel: &Array1<f64>) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let half = (kernel.len() / 2) as isize;
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nz = iz as isize + ki as isize - half;
                    let cz = clamp_idx(nz, sz.saturating_sub(1));
                    acc += kv * src[[cz, iy, ix]];
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    out
}

/// Apply a 1D kernel along the Y axis of a 3D array (clamp border).
fn convolve_y(src: &Array3<f64>, kernel: &Array1<f64>) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let half = (kernel.len() / 2) as isize;
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ny = iy as isize + ki as isize - half;
                    let cy = clamp_idx(ny, sy.saturating_sub(1));
                    acc += kv * src[[iz, cy, ix]];
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    out
}

/// Apply a 1D kernel along the X axis of a 3D array (clamp border).
fn convolve_x(src: &Array3<f64>, kernel: &Array1<f64>) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let half = (kernel.len() / 2) as isize;
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nx = ix as isize + ki as isize - half;
                    let cx = clamp_idx(nx, sx.saturating_sub(1));
                    acc += kv * src[[iz, iy, cx]];
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public filter API
// ---------------------------------------------------------------------------

/// 3D Gaussian filter (separable).
///
/// Applies a Gaussian blur by convolving the image with a 1D Gaussian kernel
/// independently along each of the three spatial axes (z, y, x).  This is
/// mathematically equivalent to a full 3D Gaussian convolution but runs in
/// O(n × k) time instead of O(n × k³).
///
/// # Arguments
///
/// * `image`    - Input volumetric image.
/// * `sigma`    - Standard deviation of the Gaussian in voxels (must be > 0).
/// * `truncate` - Kernel half-width in units of sigma (typical: 4.0).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if `sigma ≤ 0` or `truncate ≤ 0`.
pub fn gaussian_filter3d(
    image: ArrayView3<f64>,
    sigma: f64,
    truncate: f64,
) -> NdimageResult<Array3<f64>> {
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "sigma must be positive".to_string(),
        ));
    }
    if truncate <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "truncate must be positive".to_string(),
        ));
    }
    let kernel = gaussian_kernel1d(sigma, truncate);
    let owned = image.to_owned();
    let tmp1 = convolve_z(&owned, &kernel);
    let tmp2 = convolve_y(&tmp1, &kernel);
    Ok(convolve_x(&tmp2, &kernel))
}

/// 3D median filter.
///
/// Replaces each voxel with the median value in a cubic window of side `size`
/// (must be odd and ≥ 1).  Uses clamp-to-edge border handling.
///
/// # Arguments
///
/// * `image` - Input volumetric image.
/// * `size`  - Cubic window side length (must be odd and ≥ 1).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if `size == 0` or `size` is even.
pub fn median_filter3d(image: ArrayView3<f64>, size: usize) -> NdimageResult<Array3<f64>> {
    if size == 0 {
        return Err(NdimageError::InvalidInput(
            "size must be at least 1".to_string(),
        ));
    }
    if size % 2 == 0 {
        return Err(NdimageError::InvalidInput("size must be odd".to_string()));
    }
    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let half = (size / 2) as isize;
    let owned = image.to_owned();
    let mut out = Array3::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut window: Vec<f64> = Vec::with_capacity(size * size * size);
                for dz in -half..=half {
                    for dy in -half..=half {
                        for dx in -half..=half {
                            window.push(get_clamped(
                                &owned,
                                iz as isize + dz,
                                iy as isize + dy,
                                ix as isize + dx,
                            ));
                        }
                    }
                }
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                out[[iz, iy, ix]] = window[window.len() / 2];
            }
        }
    }
    Ok(out)
}

/// 3D uniform (box) filter.
///
/// Replaces each voxel with the mean value in a cubic window of side `size`
/// (must be odd and ≥ 1).  Uses clamp-to-edge border handling.
///
/// # Arguments
///
/// * `image` - Input volumetric image.
/// * `size`  - Cubic window side length (must be odd and ≥ 1).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if `size == 0` or `size` is even.
pub fn uniform_filter3d(image: ArrayView3<f64>, size: usize) -> NdimageResult<Array3<f64>> {
    if size == 0 {
        return Err(NdimageError::InvalidInput(
            "size must be at least 1".to_string(),
        ));
    }
    if size % 2 == 0 {
        return Err(NdimageError::InvalidInput("size must be odd".to_string()));
    }
    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let half = (size / 2) as isize;
    let owned = image.to_owned();
    let n = (size * size * size) as f64;
    let mut out = Array3::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut sum = 0.0;
                for dz in -half..=half {
                    for dy in -half..=half {
                        for dx in -half..=half {
                            sum += get_clamped(
                                &owned,
                                iz as isize + dz,
                                iy as isize + dy,
                                ix as isize + dx,
                            );
                        }
                    }
                }
                out[[iz, iy, ix]] = sum / n;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Gradient and Laplacian
// ---------------------------------------------------------------------------

/// 3D Sobel gradient magnitude.
///
/// Computes the gradient magnitude using Sobel kernels applied independently
/// along x, y, and z axes, then combines them as
/// `sqrt(Gx² + Gy² + Gz²)`.
///
/// # Arguments
///
/// * `image` - Input volumetric image.
pub fn sobel_gradient3d(image: ArrayView3<f64>) -> NdimageResult<Array3<f64>> {
    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let owned = image.to_owned();
    let mut out = Array3::zeros((sz, sy, sx));

    // Sobel kernel: smooth × diff in 3D
    // Gx = sobel along x, Gy along y, Gz along z
    // Each is the outer product of:
    //   smooth = [1, 2, 1] / 4
    //   diff   = [-1, 0, 1]
    // We compute them as separable operations for efficiency.

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let z = iz as isize;
                let y = iy as isize;
                let x = ix as isize;

                // Gx: diff along x, smooth along y and z
                let gx = {
                    let s = |dz: isize, dy: isize, dx: isize| {
                        get_clamped(&owned, z + dz, y + dy, x + dx)
                    };
                    // smooth(z) × smooth(y) × diff(x)
                    let sz_weights = [1.0f64, 2.0, 1.0];
                    let sy_weights = [1.0f64, 2.0, 1.0];
                    let dx_weights = [-1.0f64, 0.0, 1.0];
                    let mut g = 0.0f64;
                    for (dz, &wz) in sz_weights.iter().enumerate() {
                        for (dy, &wy) in sy_weights.iter().enumerate() {
                            for (dx, &wx) in dx_weights.iter().enumerate() {
                                let ndz = dz as isize - 1;
                                let ndy = dy as isize - 1;
                                let ndx = dx as isize - 1;
                                g += wz * wy * wx * s(ndz, ndy, ndx);
                            }
                        }
                    }
                    g / 32.0
                };

                // Gy: smooth along x and z, diff along y
                let gy = {
                    let s = |dz: isize, dy: isize, dx: isize| {
                        get_clamped(&owned, z + dz, y + dy, x + dx)
                    };
                    let sz_weights = [1.0f64, 2.0, 1.0];
                    let dy_weights = [-1.0f64, 0.0, 1.0];
                    let sx_weights = [1.0f64, 2.0, 1.0];
                    let mut g = 0.0f64;
                    for (dz, &wz) in sz_weights.iter().enumerate() {
                        for (dy, &wy) in dy_weights.iter().enumerate() {
                            for (dx, &wx) in sx_weights.iter().enumerate() {
                                let ndz = dz as isize - 1;
                                let ndy = dy as isize - 1;
                                let ndx = dx as isize - 1;
                                g += wz * wy * wx * s(ndz, ndy, ndx);
                            }
                        }
                    }
                    g / 32.0
                };

                // Gz: smooth along x and y, diff along z
                let gz = {
                    let s = |dz: isize, dy: isize, dx: isize| {
                        get_clamped(&owned, z + dz, y + dy, x + dx)
                    };
                    let dz_weights = [-1.0f64, 0.0, 1.0];
                    let sy_weights = [1.0f64, 2.0, 1.0];
                    let sx_weights = [1.0f64, 2.0, 1.0];
                    let mut g = 0.0f64;
                    for (dz, &wz) in dz_weights.iter().enumerate() {
                        for (dy, &wy) in sy_weights.iter().enumerate() {
                            for (dx, &wx) in sx_weights.iter().enumerate() {
                                let ndz = dz as isize - 1;
                                let ndy = dy as isize - 1;
                                let ndx = dx as isize - 1;
                                g += wz * wy * wx * s(ndz, ndy, ndx);
                            }
                        }
                    }
                    g / 32.0
                };

                out[[iz, iy, ix]] = (gx * gx + gy * gy + gz * gz).sqrt();
            }
        }
    }
    Ok(out)
}

/// 3D Laplacian (discrete second-order derivative).
///
/// Uses the 7-point stencil (6-connected neighbors):
/// `L[z,y,x] = sum_neighbors(v) - 6 * v[z,y,x]`
///
/// # Arguments
///
/// * `image` - Input volumetric image.
pub fn laplace3d(image: ArrayView3<f64>) -> NdimageResult<Array3<f64>> {
    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let owned = image.to_owned();
    let mut out = Array3::zeros((sz, sy, sx));

    let neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let center = owned[[iz, iy, ix]];
                let mut laplacian = -6.0 * center;
                for &(dz, dy, dx) in &neighbors {
                    laplacian +=
                        get_clamped(&owned, iz as isize + dz, iy as isize + dy, ix as isize + dx);
                }
                out[[iz, iy, ix]] = laplacian;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Non-local means denoising
// ---------------------------------------------------------------------------

/// 3D non-local means denoising.
///
/// Each voxel is replaced by a weighted average of voxels with similar patch
/// neighborhoods within a search window.  The weights decay exponentially with
/// the patch dissimilarity.
///
/// # Arguments
///
/// * `image`         - Input volumetric image.
/// * `patch_size`    - Side length of comparison patch cube (must be odd and ≥ 1).
/// * `search_radius` - Half-side of the search window (search cube side = `2 * r + 1`).
/// * `h`             - Filter strength parameter (larger ⇒ smoother).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for invalid parameter values.
pub fn non_local_means3d(
    image: ArrayView3<f64>,
    patch_size: usize,
    search_radius: usize,
    h: f64,
) -> NdimageResult<Array3<f64>> {
    if patch_size == 0 || patch_size % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "patch_size must be a positive odd integer".to_string(),
        ));
    }
    if h <= 0.0 {
        return Err(NdimageError::InvalidInput("h must be positive".to_string()));
    }
    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let owned = image.to_owned();
    let ph = (patch_size / 2) as isize;
    let sr = search_radius as isize;
    let h2 = h * h;
    let patch_vol = patch_size * patch_size * patch_size;

    let mut out = Array3::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut weight_sum = 0.0f64;
                let mut value_sum = 0.0f64;

                // Iterate over search window
                let z0 = (iz as isize - sr).max(0) as usize;
                let z1 = (iz as isize + sr).min(sz as isize - 1) as usize;
                let y0 = (iy as isize - sr).max(0) as usize;
                let y1 = (iy as isize + sr).min(sy as isize - 1) as usize;
                let x0 = (ix as isize - sr).max(0) as usize;
                let x1 = (ix as isize + sr).min(sx as isize - 1) as usize;

                for jz in z0..=z1 {
                    for jy in y0..=y1 {
                        for jx in x0..=x1 {
                            // Compute squared patch distance
                            let mut dist_sq = 0.0f64;
                            for dz in -ph..=ph {
                                for dy in -ph..=ph {
                                    for dx in -ph..=ph {
                                        let pz_i = iz as isize + dz;
                                        let py_i = iy as isize + dy;
                                        let px_i = ix as isize + dx;
                                        let pz_j = jz as isize + dz;
                                        let py_j = jy as isize + dy;
                                        let px_j = jx as isize + dx;
                                        let vi = get_clamped(&owned, pz_i, py_i, px_i);
                                        let vj = get_clamped(&owned, pz_j, py_j, px_j);
                                        let diff = vi - vj;
                                        dist_sq += diff * diff;
                                    }
                                }
                            }
                            dist_sq /= patch_vol as f64;
                            let w = (-dist_sq / h2).exp();
                            weight_sum += w;
                            value_sum += w * owned[[jz, jy, jx]];
                        }
                    }
                }

                out[[iz, iy, ix]] = if weight_sum > 0.0 {
                    value_sum / weight_sum
                } else {
                    owned[[iz, iy, ix]]
                };
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Anisotropic diffusion (Perona-Malik)
// ---------------------------------------------------------------------------

/// Perona-Malik conductance function variant.
#[inline]
fn pm_conductance(gradient_mag: f64, kappa: f64) -> f64 {
    // Exponential: c = exp(-(|∇I|/kappa)²)
    let k = gradient_mag / kappa;
    (-k * k).exp()
}

/// 3D anisotropic diffusion (Perona-Malik model).
///
/// Iteratively diffuses the image while reducing diffusion across high-gradient
/// regions (edges), thus denoising flat areas while preserving boundaries.
///
/// # Stability
///
/// For 3D stability the time step `gamma` should satisfy `gamma < 1/6`.
/// Values larger than this may produce oscillations.
///
/// # Arguments
///
/// * `image`      - Input volumetric image.
/// * `iterations` - Number of diffusion steps.
/// * `kappa`      - Edge-stopping threshold.  Larger values preserve fewer edges.
/// * `gamma`      - Time step (`< 1/6` for stable 3D diffusion).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for non-positive parameters.
pub fn anisotropic_diffusion3d(
    image: ArrayView3<f64>,
    iterations: usize,
    kappa: f64,
    gamma: f64,
) -> NdimageResult<Array3<f64>> {
    if kappa <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "kappa must be positive".to_string(),
        ));
    }
    if gamma <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "gamma must be positive".to_string(),
        ));
    }
    if iterations == 0 {
        return Ok(image.to_owned());
    }

    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut current = image.to_owned();

    for _ in 0..iterations {
        let prev = current.clone();
        for iz in 0..sz {
            for iy in 0..sy {
                for ix in 0..sx {
                    let z = iz as isize;
                    let y = iy as isize;
                    let x = ix as isize;
                    let center = prev[[iz, iy, ix]];

                    // 6-connected gradient differences
                    let gz_n = get_clamped(&prev, z - 1, y, x) - center;
                    let gz_p = get_clamped(&prev, z + 1, y, x) - center;
                    let gy_n = get_clamped(&prev, z, y - 1, x) - center;
                    let gy_p = get_clamped(&prev, z, y + 1, x) - center;
                    let gx_n = get_clamped(&prev, z, y, x - 1) - center;
                    let gx_p = get_clamped(&prev, z, y, x + 1) - center;

                    // Conductance coefficients
                    let cn = pm_conductance(gz_n.abs(), kappa);
                    let cp = pm_conductance(gz_p.abs(), kappa);
                    let cyn = pm_conductance(gy_n.abs(), kappa);
                    let cyp = pm_conductance(gy_p.abs(), kappa);
                    let cxn = pm_conductance(gx_n.abs(), kappa);
                    let cxp = pm_conductance(gx_p.abs(), kappa);

                    let flux =
                        cn * gz_n + cp * gz_p + cyn * gy_n + cyp * gy_p + cxn * gx_n + cxp * gx_p;

                    current[[iz, iy, ix]] = center + gamma * flux;
                }
            }
        }
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
// Bilateral filter
// ---------------------------------------------------------------------------

/// 3D bilateral filter.
///
/// Smooths the image while preserving edges by weighting contributions from
/// neighboring voxels by both spatial proximity and intensity similarity.
///
/// The window radius is derived from `sigma_spatial` as
/// `radius = ceil(3 * sigma_spatial)`.
///
/// # Arguments
///
/// * `image`            - Input volumetric image.
/// * `sigma_spatial`    - Spatial Gaussian standard deviation in voxels.
/// * `sigma_intensity`  - Intensity Gaussian standard deviation.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if either sigma is non-positive.
pub fn bilateral_filter3d(
    image: ArrayView3<f64>,
    sigma_spatial: f64,
    sigma_intensity: f64,
) -> NdimageResult<Array3<f64>> {
    if sigma_spatial <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "sigma_spatial must be positive".to_string(),
        ));
    }
    if sigma_intensity <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "sigma_intensity must be positive".to_string(),
        ));
    }
    let (sz, sy, sx) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let owned = image.to_owned();
    let radius = (3.0 * sigma_spatial).ceil() as isize;
    let two_ss_sq = 2.0 * sigma_spatial * sigma_spatial;
    let two_si_sq = 2.0 * sigma_intensity * sigma_intensity;
    let mut out = Array3::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let center_val = owned[[iz, iy, ix]];
                let mut w_sum = 0.0f64;
                let mut v_sum = 0.0f64;

                for dz in -radius..=radius {
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let nz = iz as isize + dz;
                            let ny = iy as isize + dy;
                            let nx = ix as isize + dx;
                            let v = get_clamped(&owned, nz, ny, nx);
                            let spatial_dist_sq = (dz * dz + dy * dy + dx * dx) as f64;
                            let intensity_diff = v - center_val;
                            let w = (-spatial_dist_sq / two_ss_sq
                                - intensity_diff * intensity_diff / two_si_sq)
                                .exp();
                            w_sum += w;
                            v_sum += w * v;
                        }
                    }
                }

                out[[iz, iy, ix]] = if w_sum > 0.0 {
                    v_sum / w_sum
                } else {
                    center_val
                };
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    /// Create a uniform 5×5×5 volume filled with `value`.
    fn uniform_volume(value: f64) -> Array3<f64> {
        Array3::from_elem((5, 5, 5), value)
    }

    /// Create a ramp volume where `v[z,y,x] = (z + y + x) as f64`.
    fn ramp_volume() -> Array3<f64> {
        Array3::from_shape_fn((5, 5, 5), |(z, y, x)| (z + y + x) as f64)
    }

    /// Create a noisy step-edge volume: left half ≈ 0, right half ≈ 1.
    fn step_volume() -> Array3<f64> {
        Array3::from_shape_fn((8, 8, 8), |(_, _, x)| if x < 4 { 0.0 } else { 1.0 })
    }

    // ----- Gaussian filter -----

    #[test]
    fn test_gaussian_uniform_preserves_value() {
        let img = uniform_volume(3.0);
        let result = gaussian_filter3d(img.view(), 1.0, 4.0).expect("gaussian failed");
        for &v in result.iter() {
            assert!((v - 3.0).abs() < 1e-9, "expected 3.0, got {}", v);
        }
    }

    #[test]
    fn test_gaussian_sigma_zero_error() {
        let img = uniform_volume(1.0);
        assert!(gaussian_filter3d(img.view(), 0.0, 4.0).is_err());
    }

    #[test]
    fn test_gaussian_truncate_zero_error() {
        let img = uniform_volume(1.0);
        assert!(gaussian_filter3d(img.view(), 1.0, 0.0).is_err());
    }

    #[test]
    fn test_gaussian_smooths_step() {
        let img = step_volume();
        let smoothed = gaussian_filter3d(img.view(), 1.5, 4.0).expect("gaussian failed");
        // The sharp edge at x=4 should be blurred: pixels near edge have intermediate values
        let edge_val = smoothed[[4, 4, 4]]; // near the edge
        assert!(
            edge_val > 0.01 && edge_val < 0.99,
            "expected blurred edge, got {}",
            edge_val
        );
    }

    #[test]
    fn test_gaussian_output_shape() {
        let img = ramp_volume();
        let result = gaussian_filter3d(img.view(), 1.0, 3.0).expect("gaussian failed");
        assert_eq!(result.shape(), img.shape());
    }

    // ----- Median filter -----

    #[test]
    fn test_median_uniform_preserves_value() {
        let img = uniform_volume(7.0);
        let result = median_filter3d(img.view(), 3).expect("median failed");
        for &v in result.iter() {
            assert!((v - 7.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_median_removes_spike() {
        let mut img = uniform_volume(0.0);
        img[[2, 2, 2]] = 100.0; // impulse noise
        let result = median_filter3d(img.view(), 3).expect("median failed");
        // Center voxel should be suppressed
        assert!(
            result[[2, 2, 2]] < 10.0,
            "spike not removed: {}",
            result[[2, 2, 2]]
        );
    }

    #[test]
    fn test_median_even_size_error() {
        let img = uniform_volume(1.0);
        assert!(median_filter3d(img.view(), 2).is_err());
    }

    #[test]
    fn test_median_zero_size_error() {
        let img = uniform_volume(1.0);
        assert!(median_filter3d(img.view(), 0).is_err());
    }

    #[test]
    fn test_median_output_shape() {
        let img = ramp_volume();
        let result = median_filter3d(img.view(), 3).expect("median failed");
        assert_eq!(result.shape(), img.shape());
    }

    // ----- Uniform filter -----

    #[test]
    fn test_uniform_constant_image() {
        let img = uniform_volume(5.0);
        let result = uniform_filter3d(img.view(), 3).expect("uniform failed");
        for &v in result.iter() {
            assert!((v - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_uniform_output_shape() {
        let img = ramp_volume();
        let result = uniform_filter3d(img.view(), 3).expect("uniform failed");
        assert_eq!(result.shape(), img.shape());
    }

    #[test]
    fn test_uniform_even_size_error() {
        let img = uniform_volume(1.0);
        assert!(uniform_filter3d(img.view(), 4).is_err());
    }

    // ----- Sobel gradient -----

    #[test]
    fn test_sobel_uniform_near_zero() {
        let img = uniform_volume(2.0);
        let result = sobel_gradient3d(img.view()).expect("sobel failed");
        for &v in result.iter() {
            assert!(v.abs() < 1e-9, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_sobel_step_detects_edge() {
        let img = step_volume();
        let result = sobel_gradient3d(img.view()).expect("sobel failed");
        // Voxels at the edge (x around 3-4) should have high gradient
        let edge_mag = result[[4, 4, 3]];
        assert!(
            edge_mag > 0.01,
            "expected nonzero gradient at edge, got {}",
            edge_mag
        );
    }

    #[test]
    fn test_sobel_output_shape() {
        let img = ramp_volume();
        let result = sobel_gradient3d(img.view()).expect("sobel failed");
        assert_eq!(result.shape(), img.shape());
    }

    // ----- Laplacian -----

    #[test]
    fn test_laplace_uniform_zero() {
        let img = uniform_volume(3.0);
        let result = laplace3d(img.view()).expect("laplace failed");
        // Interior voxels should have zero Laplacian for constant image
        // (border voxels may differ due to clamping)
        for iz in 1..4usize {
            for iy in 1..4usize {
                for ix in 1..4usize {
                    assert!(
                        result[[iz, iy, ix]].abs() < 1e-9,
                        "expected 0, got {}",
                        result[[iz, iy, ix]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_laplace_output_shape() {
        let img = ramp_volume();
        let result = laplace3d(img.view()).expect("laplace failed");
        assert_eq!(result.shape(), img.shape());
    }

    // ----- Non-local means -----

    #[test]
    fn test_nlm_uniform_preserves_value() {
        let img = uniform_volume(4.0);
        let result = non_local_means3d(img.view(), 3, 2, 0.1).expect("NLM failed");
        for &v in result.iter() {
            assert!((v - 4.0).abs() < 1e-6, "expected 4.0, got {}", v);
        }
    }

    #[test]
    fn test_nlm_invalid_patch_size() {
        let img = uniform_volume(1.0);
        assert!(non_local_means3d(img.view(), 2, 2, 0.1).is_err()); // even size
        assert!(non_local_means3d(img.view(), 0, 2, 0.1).is_err()); // zero size
    }

    #[test]
    fn test_nlm_output_shape() {
        let img = ramp_volume();
        let result = non_local_means3d(img.view(), 3, 1, 0.5).expect("NLM failed");
        assert_eq!(result.shape(), img.shape());
    }

    // ----- Anisotropic diffusion -----

    #[test]
    fn test_aniso_diff_uniform_preserves_value() {
        let img = uniform_volume(5.0);
        let result = anisotropic_diffusion3d(img.view(), 5, 10.0, 0.1)
            .expect("anisotropic diffusion failed");
        for &v in result.iter() {
            assert!((v - 5.0).abs() < 1e-8, "expected 5.0, got {}", v);
        }
    }

    #[test]
    fn test_aniso_diff_zero_iterations_identity() {
        let img = ramp_volume();
        let result = anisotropic_diffusion3d(img.view(), 0, 10.0, 0.1)
            .expect("anisotropic diffusion failed");
        for ((z, y, x), &v) in result.indexed_iter() {
            assert!((v - img[[z, y, x]]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_aniso_diff_invalid_kappa_error() {
        let img = uniform_volume(1.0);
        assert!(anisotropic_diffusion3d(img.view(), 5, 0.0, 0.1).is_err());
    }

    #[test]
    fn test_aniso_diff_output_shape() {
        let img = ramp_volume();
        let result = anisotropic_diffusion3d(img.view(), 3, 10.0, 0.1)
            .expect("anisotropic diffusion failed");
        assert_eq!(result.shape(), img.shape());
    }

    // ----- Bilateral filter -----

    #[test]
    fn test_bilateral_uniform_preserves_value() {
        let img = uniform_volume(6.0);
        let result = bilateral_filter3d(img.view(), 1.5, 1.0).expect("bilateral filter failed");
        for &v in result.iter() {
            assert!((v - 6.0).abs() < 1e-8, "expected 6.0, got {}", v);
        }
    }

    #[test]
    fn test_bilateral_output_shape() {
        let img = ramp_volume();
        let result = bilateral_filter3d(img.view(), 1.5, 1.0).expect("bilateral filter failed");
        assert_eq!(result.shape(), img.shape());
    }

    #[test]
    fn test_bilateral_invalid_sigma_error() {
        let img = uniform_volume(1.0);
        assert!(bilateral_filter3d(img.view(), 0.0, 1.0).is_err());
        assert!(bilateral_filter3d(img.view(), 1.0, 0.0).is_err());
    }

    #[test]
    fn test_bilateral_step_edge_preservation() {
        // Bilateral should preserve the step edge better than Gaussian
        let img = step_volume();
        let bilateral = bilateral_filter3d(img.view(), 1.5, 0.1).expect("bilateral filter failed");
        let gaussian = gaussian_filter3d(img.view(), 1.5, 4.0).expect("gaussian failed");
        // The far interior of the left half should stay close to 0 with bilateral
        let bi_left = bilateral[[4, 4, 0]]; // deep in the left (0) half
        let ga_left = gaussian[[4, 4, 0]];
        // Bilateral should be at least as close to 0 as Gaussian
        assert!(
            bi_left.abs() <= ga_left.abs() + 0.3,
            "bilateral({}) should preserve flat region better than gaussian({})",
            bi_left,
            ga_left
        );
    }
}
