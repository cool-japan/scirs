//! 3D Convolution and Volumetric Filtering
//!
//! This module provides 3D convolution and cross-correlation, separable Gaussian
//! filtering, box filtering, non-separable median filtering, and Laplacian-of-Gaussian
//! (LoG) blob detection for volumetric (`Array3<f64>`) data.
//!
//! # Border / Padding Modes
//!
//! Three padding strategies are available via [`Padding3D`]:
//!
//! | Variant     | Behaviour at the boundary                        |
//! |-------------|--------------------------------------------------|
//! | `Zero`      | Out-of-bounds indices map to 0.0                 |
//! | `Reflect`   | Mirror reflection around the edge voxel          |
//! | `Replicate` | Clamp-to-edge (nearest-neighbour extrapolation)  |
//!
//! # Separable Filter Optimization
//!
//! Filters that are separable (Gaussian, uniform/box) are implemented as three
//! successive 1D convolutions along z, y, and x.  This reduces the per-voxel
//! work from O(k³) to O(3k) where k is the 1D kernel half-width.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array1, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// Padding type
// ─────────────────────────────────────────────────────────────────────────────

/// Padding strategy for 3D convolution / filtering operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding3D {
    /// Zero-pad: out-of-bounds voxels are treated as 0.0.
    Zero,
    /// Reflect: mirror the volume around the boundary.
    ///
    /// For an axis of length N the index `i` outside `[0, N-1]` is mapped to
    /// the reflected position using the rule `reflect(i, N)`.
    Reflect,
    /// Replicate (clamp-to-edge): clamp the index to `[0, N-1]`.
    Replicate,
}

// ─────────────────────────────────────────────────────────────────────────────
// Index resolution helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Reflect an out-of-range index `i` into `[0, n-1]`.
///
/// Uses the "edge-inclusive" formula:  ...3 2 1 0 1 2 3...
#[inline]
fn reflect_index(i: isize, n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let n = n as isize;
    // Fold the index into the periodic double-length domain.
    let period = 2 * n - 2;
    if period <= 0 {
        return Some(0); // single-element axis
    }
    // Reduce to [0, period)
    let mut r = i % period;
    if r < 0 {
        r += period;
    }
    // Unfold second half
    if r >= n {
        r = period - r;
    }
    Some(r as usize)
}

/// Resolve a raw (possibly out-of-range) index according to the padding mode.
///
/// Returns `None` when the mode is `Zero` and the index is out of bounds.
#[inline]
fn resolve_index(i: isize, n: usize, padding: Padding3D) -> Option<usize> {
    if i >= 0 && (i as usize) < n {
        return Some(i as usize);
    }
    match padding {
        Padding3D::Zero => None,
        Padding3D::Replicate => {
            if n == 0 {
                None
            } else {
                Some(i.clamp(0, n as isize - 1) as usize)
            }
        }
        Padding3D::Reflect => reflect_index(i, n),
    }
}

/// Read a voxel from `volume`, applying `padding` for out-of-bounds coordinates.
#[inline]
fn get_padded(volume: &Array3<f64>, z: isize, y: isize, x: isize, padding: Padding3D) -> f64 {
    let shape = volume.shape();
    let oz = resolve_index(z, shape[0], padding);
    let oy = resolve_index(y, shape[1], padding);
    let ox = resolve_index(x, shape[2], padding);
    match (oz, oy, ox) {
        (Some(rz), Some(ry), Some(rx)) => volume[[rz, ry, rx]],
        _ => 0.0, // Zero mode, or zero-volume axis
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Full 3D convolution / cross-correlation
// ─────────────────────────────────────────────────────────────────────────────

/// 3D discrete convolution.
///
/// Convolves `volume` with `kernel` using the chosen `padding` strategy.
/// The kernel is flipped (mirrored) along all three axes before sliding, which
/// is the definition of mathematical convolution.
///
/// # Arguments
///
/// * `volume`  - Input 3D array.
/// * `kernel`  - Convolution kernel (any odd-or-even size is accepted).
/// * `padding` - Border handling strategy.
///
/// # Returns
///
/// An `Array3<f64>` with the same shape as `volume`.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if the kernel has zero extent on any axis.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::convolution3d::{convolve3d, Padding3D};
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((8, 8, 8));
/// let kernel = Array3::<f64>::ones((3, 3, 3));
/// let out = convolve3d(&vol, &kernel, Padding3D::Zero).unwrap();
/// assert_eq!(out.shape(), [8, 8, 8]);
/// ```
pub fn convolve3d(
    volume: &Array3<f64>,
    kernel: &Array3<f64>,
    padding: Padding3D,
) -> NdimageResult<Array3<f64>> {
    let ks = kernel.shape();
    if ks[0] == 0 || ks[1] == 0 || ks[2] == 0 {
        return Err(NdimageError::InvalidInput(
            "kernel must have non-zero extent on all axes".to_string(),
        ));
    }
    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);
    let (kz, ky, kx) = (ks[0], ks[1], ks[2]);
    let hz = (kz / 2) as isize;
    let hy = (ky / 2) as isize;
    let hx = (kx / 2) as isize;

    let mut out = Array3::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0f64;
                for dkz in 0..kz {
                    for dky in 0..ky {
                        for dkx in 0..kx {
                            // Flipped kernel indices (convolution vs correlation)
                            let fkz = kz - 1 - dkz;
                            let fky = ky - 1 - dky;
                            let fkx = kx - 1 - dkx;
                            let kval = kernel[[fkz, fky, fkx]];
                            let vz = iz as isize + dkz as isize - hz;
                            let vy = iy as isize + dky as isize - hy;
                            let vx = ix as isize + dkx as isize - hx;
                            acc += kval * get_padded(volume, vz, vy, vx, padding);
                        }
                    }
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    Ok(out)
}

/// 3D cross-correlation.
///
/// Slides `kernel` over `volume` **without** flipping it (unlike [`convolve3d`]).
/// This is the operation commonly called "convolution" in machine-learning
/// frameworks.
///
/// # Arguments
///
/// * `volume`  - Input 3D array.
/// * `kernel`  - Correlation template.
/// * `padding` - Border handling strategy.
///
/// # Returns
///
/// An `Array3<f64>` with the same shape as `volume`.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if the kernel has zero extent on any axis.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::convolution3d::{correlate3d, Padding3D};
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((6, 6, 6));
/// let kernel = Array3::<f64>::ones((3, 3, 3));
/// let out = correlate3d(&vol, &kernel, Padding3D::Replicate).unwrap();
/// assert_eq!(out.shape(), [6, 6, 6]);
/// ```
pub fn correlate3d(
    volume: &Array3<f64>,
    kernel: &Array3<f64>,
    padding: Padding3D,
) -> NdimageResult<Array3<f64>> {
    let ks = kernel.shape();
    if ks[0] == 0 || ks[1] == 0 || ks[2] == 0 {
        return Err(NdimageError::InvalidInput(
            "kernel must have non-zero extent on all axes".to_string(),
        ));
    }
    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);
    let (kz, ky, kx) = (ks[0], ks[1], ks[2]);
    let hz = (kz / 2) as isize;
    let hy = (ky / 2) as isize;
    let hx = (kx / 2) as isize;

    let mut out = Array3::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0f64;
                for dkz in 0..kz {
                    for dky in 0..ky {
                        for dkx in 0..kx {
                            let kval = kernel[[dkz, dky, dkx]];
                            let vz = iz as isize + dkz as isize - hz;
                            let vy = iy as isize + dky as isize - hy;
                            let vx = ix as isize + dkx as isize - hx;
                            acc += kval * get_padded(volume, vz, vy, vx, padding);
                        }
                    }
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Separable 1D kernel helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a 1D kernel along the Z axis with the given padding mode.
fn convolve1d_z(src: &Array3<f64>, kernel: &Array1<f64>, padding: Padding3D) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let klen = kernel.len();
    let half = (klen / 2) as isize;
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nz = iz as isize + ki as isize - half;
                    acc += kv * get_padded(src, nz, iy as isize, ix as isize, padding);
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    out
}

/// Apply a 1D kernel along the Y axis with the given padding mode.
fn convolve1d_y(src: &Array3<f64>, kernel: &Array1<f64>, padding: Padding3D) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let klen = kernel.len();
    let half = (klen / 2) as isize;
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ny = iy as isize + ki as isize - half;
                    acc += kv * get_padded(src, iz as isize, ny, ix as isize, padding);
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    out
}

/// Apply a 1D kernel along the X axis with the given padding mode.
fn convolve1d_x(src: &Array3<f64>, kernel: &Array1<f64>, padding: Padding3D) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let klen = kernel.len();
    let half = (klen / 2) as isize;
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut acc = 0.0f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nx = ix as isize + ki as isize - half;
                    acc += kv * get_padded(src, iz as isize, iy as isize, nx, padding);
                }
                out[[iz, iy, ix]] = acc;
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian filter
// ─────────────────────────────────────────────────────────────────────────────

/// Build a normalised 1D Gaussian kernel.
///
/// The kernel spans `2 * ceil(truncate * sigma) + 1` elements.
fn gaussian_kernel1d(sigma: f64, truncate: f64) -> Array1<f64> {
    let radius = (truncate * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut k = Array1::zeros(size);
    let two_s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f64;
    for i in 0..size {
        let x = i as f64 - radius as f64;
        let v = (-x * x / two_s2).exp();
        k[i] = v;
        sum += v;
    }
    if sum != 0.0 {
        k.mapv_inplace(|v| v / sum);
    }
    k
}

/// 3D separable Gaussian filter with per-axis sigma values.
///
/// The filter is applied as three independent 1D convolutions along z, y, and x.
/// This is mathematically equivalent to a full 3D Gaussian convolution but runs
/// in O(N · k) rather than O(N · k³) time.
///
/// # Arguments
///
/// * `volume`  - Input volumetric array.
/// * `sigma`   - Standard deviations `[sz, sy, sx]` (all must be > 0).
/// * `padding` - Border handling.  Defaults to `Replicate` if not specified;
///               this function always requires an explicit choice.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if any sigma component is ≤ 0.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::convolution3d::{gaussian_filter3d, Padding3D};
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((10, 10, 10));
/// let out = gaussian_filter3d(&vol, [1.0, 1.0, 1.0], Padding3D::Reflect).unwrap();
/// assert_eq!(out.shape(), [10, 10, 10]);
/// // Interior of a uniform volume is unchanged by Gaussian filtering
/// assert!((out[[5, 5, 5]] - 1.0).abs() < 1e-10);
/// ```
pub fn gaussian_filter3d(
    volume: &Array3<f64>,
    sigma: [f64; 3],
    padding: Padding3D,
) -> NdimageResult<Array3<f64>> {
    for (axis, &s) in sigma.iter().enumerate() {
        if s <= 0.0 {
            return Err(NdimageError::InvalidInput(format!(
                "sigma[{}] must be positive, got {}",
                axis, s
            )));
        }
    }
    const TRUNCATE: f64 = 4.0;
    let kz = gaussian_kernel1d(sigma[0], TRUNCATE);
    let ky = gaussian_kernel1d(sigma[1], TRUNCATE);
    let kx = gaussian_kernel1d(sigma[2], TRUNCATE);

    let tmp1 = convolve1d_z(volume, &kz, padding);
    let tmp2 = convolve1d_y(&tmp1, &ky, padding);
    Ok(convolve1d_x(&tmp2, &kx, padding))
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform (box) filter
// ─────────────────────────────────────────────────────────────────────────────

/// 3D separable uniform (box) filter with per-axis window sizes.
///
/// Each voxel in the output is the mean of the rectangular neighbourhood of
/// shape `size[0] × size[1] × size[2]`.  The filter is applied as three
/// successive 1D averaging kernels for efficiency.
///
/// # Arguments
///
/// * `volume`  - Input volumetric array.
/// * `size`    - Window sizes `[sz, sy, sx]`.  Each component must be ≥ 1.
/// * `padding` - Border handling strategy.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if any size component is 0.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::convolution3d::{uniform_filter3d, Padding3D};
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((8, 8, 8));
/// let out = uniform_filter3d(&vol, [3, 3, 3], Padding3D::Replicate).unwrap();
/// assert_eq!(out.shape(), [8, 8, 8]);
/// ```
pub fn uniform_filter3d(
    volume: &Array3<f64>,
    size: [usize; 3],
    padding: Padding3D,
) -> NdimageResult<Array3<f64>> {
    for (axis, &s) in size.iter().enumerate() {
        if s == 0 {
            return Err(NdimageError::InvalidInput(format!(
                "size[{}] must be at least 1",
                axis
            )));
        }
    }
    // Build normalised box kernels for each axis.
    let make_box = |s: usize| -> Array1<f64> { Array1::from_elem(s, 1.0 / s as f64) };
    let kz = make_box(size[0]);
    let ky = make_box(size[1]);
    let kx = make_box(size[2]);

    let tmp1 = convolve1d_z(volume, &kz, padding);
    let tmp2 = convolve1d_y(&tmp1, &ky, padding);
    Ok(convolve1d_x(&tmp2, &kx, padding))
}

// ─────────────────────────────────────────────────────────────────────────────
// Median filter
// ─────────────────────────────────────────────────────────────────────────────

/// 3D median filter with a rectangular window.
///
/// Replaces each voxel with the median value in the neighbourhood of shape
/// `size[0] × size[1] × size[2]`.
///
/// # Arguments
///
/// * `volume`  - Input volumetric array.
/// * `size`    - Window sizes `[sz, sy, sx]`.  Each component must be ≥ 1.
/// * `padding` - Border handling strategy.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if any size component is 0.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::convolution3d::{median_filter3d, Padding3D};
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((6, 6, 6));
/// let out = median_filter3d(&vol, [3, 3, 3], Padding3D::Replicate).unwrap();
/// assert_eq!(out.shape(), [6, 6, 6]);
/// ```
pub fn median_filter3d(
    volume: &Array3<f64>,
    size: [usize; 3],
    padding: Padding3D,
) -> NdimageResult<Array3<f64>> {
    for (axis, &s) in size.iter().enumerate() {
        if s == 0 {
            return Err(NdimageError::InvalidInput(format!(
                "size[{}] must be at least 1",
                axis
            )));
        }
    }
    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);
    let hz = (size[0] / 2) as isize;
    let hy = (size[1] / 2) as isize;
    let hx = (size[2] / 2) as isize;
    let window_vol = size[0] * size[1] * size[2];

    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut window: Vec<f64> = Vec::with_capacity(window_vol);
                for dz in -hz..=hz {
                    for dy in -hy..=hy {
                        for dx in -hx..=hx {
                            window.push(get_padded(
                                volume,
                                iz as isize + dz,
                                iy as isize + dy,
                                ix as isize + dx,
                                padding,
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

// ─────────────────────────────────────────────────────────────────────────────
// Laplacian of Gaussian (LoG)
// ─────────────────────────────────────────────────────────────────────────────

/// 3D Laplacian of Gaussian (LoG) filter for blob detection.
///
/// The LoG is computed as the Laplacian of a Gaussian-blurred volume:
///
/// ```text
/// LoG(f) = ∇²(G_σ * f)
/// ```
///
/// Implementation uses the approximation:
///
/// ```text
/// LoG ≈ G_σ * L(f)
/// ```
///
/// where `L` is the 6-neighbour discrete Laplacian.  This is faster than
/// building a full 3D LoG kernel and numerically equivalent for smooth signals.
///
/// Blobs appear as local minima (dark blobs on bright background) or local
/// maxima (bright blobs on dark background) in the output.
///
/// # Arguments
///
/// * `volume`  - Input volumetric array.
/// * `sigma`   - Gaussian sigma `[sz, sy, sx]` (all must be > 0).
///               Larger sigma detects larger blobs.
/// * `padding` - Border handling strategy.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if any sigma component is ≤ 0.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::convolution3d::{laplacian_of_gaussian3d, Padding3D};
/// use scirs2_core::ndarray::Array3;
///
/// let mut vol = Array3::<f64>::zeros((12, 12, 12));
/// vol[[6, 6, 6]] = 1.0;
/// let log = laplacian_of_gaussian3d(&vol, [1.5, 1.5, 1.5], Padding3D::Zero).unwrap();
/// assert_eq!(log.shape(), [12, 12, 12]);
/// ```
pub fn laplacian_of_gaussian3d(
    volume: &Array3<f64>,
    sigma: [f64; 3],
    padding: Padding3D,
) -> NdimageResult<Array3<f64>> {
    // 1. Gaussian-smooth the volume.
    let smoothed = gaussian_filter3d(volume, sigma, padding)?;
    // 2. Apply discrete 3D Laplacian (6-neighbour stencil).
    laplacian3d_internal(&smoothed, padding)
}

/// 3D discrete Laplacian using the 6-neighbour stencil.
///
/// The stencil is:  L[i,j,k] = f[i+1,j,k] + f[i-1,j,k]
///                            + f[i,j+1,k] + f[i,j-1,k]
///                            + f[i,j,k+1] + f[i,j,k-1]
///                            - 6 · f[i,j,k]
fn laplacian3d_internal(volume: &Array3<f64>, padding: Padding3D) -> NdimageResult<Array3<f64>> {
    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);
    let mut out = Array3::zeros((sz, sy, sx));
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let z = iz as isize;
                let y = iy as isize;
                let x = ix as isize;
                let center = volume[[iz, iy, ix]];
                let lp = get_padded(volume, z - 1, y, x, padding)
                    + get_padded(volume, z + 1, y, x, padding)
                    + get_padded(volume, z, y - 1, x, padding)
                    + get_padded(volume, z, y + 1, x, padding)
                    + get_padded(volume, z, y, x - 1, padding)
                    + get_padded(volume, z, y, x + 1, padding)
                    - 6.0 * center;
                out[[iz, iy, ix]] = lp;
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn make_impulse(sz: usize, sy: usize, sx: usize) -> Array3<f64> {
        let mut v = Array3::zeros((sz, sy, sx));
        v[[sz / 2, sy / 2, sx / 2]] = 1.0;
        v
    }

    #[test]
    fn test_convolve3d_identity_kernel() {
        let vol = Array3::<f64>::from_shape_fn((4, 5, 6), |(z, y, x)| (z * 30 + y * 6 + x) as f64);
        let mut identity = Array3::zeros((3, 3, 3));
        identity[[1, 1, 1]] = 1.0;
        // Convolution with a delta kernel (flipped, but delta is symmetric)
        let out = convolve3d(&vol, &identity, Padding3D::Zero).expect("convolve3d failed");
        // Interior should match the original
        for iz in 1..3usize {
            for iy in 1..4usize {
                for ix in 1..5usize {
                    assert!(
                        (out[[iz, iy, ix]] - vol[[iz, iy, ix]]).abs() < 1e-12,
                        "mismatch at [{}, {}, {}]: got {}, expected {}",
                        iz,
                        iy,
                        ix,
                        out[[iz, iy, ix]],
                        vol[[iz, iy, ix]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_correlate3d_vs_convolve3d_symmetric() {
        // For a symmetric kernel, correlate == convolve
        let vol = Array3::<f64>::ones((6, 6, 6));
        let kernel = Array3::from_shape_fn((3, 3, 3), |(z, y, x)| {
            // symmetric kernel: distance-weighted
            let dz = z as f64 - 1.0;
            let dy = y as f64 - 1.0;
            let dx = x as f64 - 1.0;
            1.0 / (1.0 + dz * dz + dy * dy + dx * dx)
        });
        let c1 = convolve3d(&vol, &kernel, Padding3D::Replicate).expect("convolve3d");
        let c2 = correlate3d(&vol, &kernel, Padding3D::Replicate).expect("correlate3d");
        for iz in 0..6 {
            for iy in 0..6 {
                for ix in 0..6 {
                    assert!(
                        (c1[[iz, iy, ix]] - c2[[iz, iy, ix]]).abs() < 1e-12,
                        "asymmetry at [{},{},{}]",
                        iz,
                        iy,
                        ix
                    );
                }
            }
        }
    }

    #[test]
    fn test_gaussian_filter3d_uniform_volume() {
        let vol = Array3::<f64>::ones((10, 10, 10));
        let out = gaussian_filter3d(&vol, [1.0, 1.0, 1.0], Padding3D::Reflect)
            .expect("gaussian_filter3d");
        // Interior of uniform volume should be preserved exactly.
        for iz in 3..7 {
            for iy in 3..7 {
                for ix in 3..7 {
                    assert!(
                        (out[[iz, iy, ix]] - 1.0).abs() < 1e-10,
                        "interior mismatch at [{},{},{}]: {}",
                        iz,
                        iy,
                        ix,
                        out[[iz, iy, ix]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gaussian_filter3d_invalid_sigma() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        assert!(gaussian_filter3d(&vol, [0.0, 1.0, 1.0], Padding3D::Zero).is_err());
        assert!(gaussian_filter3d(&vol, [1.0, -1.0, 1.0], Padding3D::Zero).is_err());
    }

    #[test]
    fn test_uniform_filter3d_mean() {
        let vol = Array3::<f64>::ones((8, 8, 8));
        let out =
            uniform_filter3d(&vol, [3, 3, 3], Padding3D::Replicate).expect("uniform_filter3d");
        // Mean of a uniform volume is the same value.
        for iz in 1..7 {
            for iy in 1..7 {
                for ix in 1..7 {
                    assert!((out[[iz, iy, ix]] - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_median_filter3d_constant() {
        let vol = Array3::<f64>::from_elem((5, 5, 5), 7.0);
        let out = median_filter3d(&vol, [3, 3, 3], Padding3D::Replicate).expect("median_filter3d");
        for &v in out.iter() {
            assert!((v - 7.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_median_filter3d_invalid_size() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        assert!(median_filter3d(&vol, [0, 3, 3], Padding3D::Zero).is_err());
    }

    #[test]
    fn test_log3d_impulse() {
        // LoG of an impulse should resemble the Mexican-hat shape.
        let vol = make_impulse(12, 12, 12);
        let log = laplacian_of_gaussian3d(&vol, [1.0, 1.0, 1.0], Padding3D::Zero)
            .expect("laplacian_of_gaussian3d");
        // The center should be a local extremum (negative for bright-on-dark with LoG convention)
        let center = log[[6, 6, 6]];
        let neighbour = log[[7, 6, 6]];
        // Center should be more negative than a neighbour
        assert!(
            center < neighbour,
            "center={}, neighbour={}",
            center,
            neighbour
        );
    }

    #[test]
    fn test_padding_modes_shape() {
        let vol = Array3::<f64>::ones((4, 5, 6));
        let k = Array3::<f64>::ones((3, 3, 3));
        for &mode in &[Padding3D::Zero, Padding3D::Reflect, Padding3D::Replicate] {
            let out = convolve3d(&vol, &k, mode).expect("convolve3d");
            assert_eq!(out.shape(), [4, 5, 6]);
        }
    }

    #[test]
    fn test_reflect_index_symmetry() {
        // reflect_index should map -1 -> 1, n -> n-2 for n > 1
        assert_eq!(reflect_index(-1, 5), Some(1));
        assert_eq!(reflect_index(5, 5), Some(3));
        assert_eq!(reflect_index(0, 5), Some(0));
        assert_eq!(reflect_index(4, 5), Some(4));
    }
}
