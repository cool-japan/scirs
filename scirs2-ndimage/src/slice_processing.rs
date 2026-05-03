//! Multi-Planar Reconstruction and Volume Rendering
//!
//! This module provides functions for extracting 2D slices from 3D volumetric
//! data at arbitrary orientations (axial, coronal, sagittal, and oblique), plus
//! intensity projection techniques (MIP and AIP) and a simple pre-integrated
//! transfer function for direct volume rendering.
//!
//! # Coordinate Convention
//!
//! All arrays follow the (depth/z, height/y, width/x) axis convention used
//! throughout `scirs2-ndimage`.  Indices in the slice extraction functions
//! therefore correspond to:
//!
//! * **axial** – constant z (depth) slice → output shape `(height, width)`
//! * **coronal** – constant y (height) slice → output shape `(depth, width)`
//! * **sagittal** – constant x (width) slice → output shape `(depth, height)`
//!
//! # References
//!
//! - Engel, Hadwiger, Kniss & Rezk-Salama (2006), "Real-Time Volume Graphics",
//!   AK Peters.
//! - Levoy (1988), "Display of Surfaces from Volume Data", IEEE CG&A 8(3):29–37.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3};

// ---------------------------------------------------------------------------
// Trilinear interpolation helper
// ---------------------------------------------------------------------------

/// Sample `volume` at a continuous position `(z, y, x)` using trilinear
/// interpolation with clamp-to-edge border handling.
fn trilinear_sample(volume: &Array3<f64>, z: f64, y: f64, x: f64) -> f64 {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0] as f64, shape[1] as f64, shape[2] as f64);

    // Clamp to valid range.
    let z = z.clamp(0.0, sz - 1.0);
    let y = y.clamp(0.0, sy - 1.0);
    let x = x.clamp(0.0, sx - 1.0);

    let z0 = z.floor() as usize;
    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let z1 = (z0 + 1).min(shape[0] - 1);
    let y1 = (y0 + 1).min(shape[1] - 1);
    let x1 = (x0 + 1).min(shape[2] - 1);

    let tz = z - z.floor();
    let ty = y - y.floor();
    let tx = x - x.floor();

    let v000 = volume[[z0, y0, x0]];
    let v001 = volume[[z0, y0, x1]];
    let v010 = volume[[z0, y1, x0]];
    let v011 = volume[[z0, y1, x1]];
    let v100 = volume[[z1, y0, x0]];
    let v101 = volume[[z1, y0, x1]];
    let v110 = volume[[z1, y1, x0]];
    let v111 = volume[[z1, y1, x1]];

    let c00 = v000 * (1.0 - tx) + v001 * tx;
    let c01 = v010 * (1.0 - tx) + v011 * tx;
    let c10 = v100 * (1.0 - tx) + v101 * tx;
    let c11 = v110 * (1.0 - tx) + v111 * tx;

    let c0 = c00 * (1.0 - ty) + c01 * ty;
    let c1 = c10 * (1.0 - ty) + c11 * ty;

    c0 * (1.0 - tz) + c1 * tz
}

// ---------------------------------------------------------------------------
// Axial / Coronal / Sagittal reformats
// ---------------------------------------------------------------------------

/// Extract an axial (constant-z) slice from a 3D volume.
///
/// Returns a 2D array of shape `(height, width)`.
///
/// # Arguments
///
/// * `volume` – 3D array `(depth, height, width)`.
/// * `slice_idx` – Depth index to extract (0 ≤ `slice_idx` < depth).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` when `slice_idx` is out of range.
pub fn reformat_axial(volume: &Array3<f64>, slice_idx: usize) -> NdimageResult<Array2<f64>> {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if slice_idx >= sz {
        return Err(NdimageError::InvalidInput(format!(
            "slice_idx {slice_idx} is out of range (depth = {sz})"
        )));
    }
    let mut out = Array2::zeros((sy, sx));
    for iy in 0..sy {
        for ix in 0..sx {
            out[[iy, ix]] = volume[[slice_idx, iy, ix]];
        }
    }
    Ok(out)
}

/// Extract a coronal (constant-y) slice from a 3D volume.
///
/// Returns a 2D array of shape `(depth, width)`.
///
/// # Arguments
///
/// * `volume` – 3D array `(depth, height, width)`.
/// * `slice_idx` – Height index to extract (0 ≤ `slice_idx` < height).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` when `slice_idx` is out of range.
pub fn reformat_coronal(volume: &Array3<f64>, slice_idx: usize) -> NdimageResult<Array2<f64>> {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if slice_idx >= sy {
        return Err(NdimageError::InvalidInput(format!(
            "slice_idx {slice_idx} is out of range (height = {sy})"
        )));
    }
    let mut out = Array2::zeros((sz, sx));
    for iz in 0..sz {
        for ix in 0..sx {
            out[[iz, ix]] = volume[[iz, slice_idx, ix]];
        }
    }
    Ok(out)
}

/// Extract a sagittal (constant-x) slice from a 3D volume.
///
/// Returns a 2D array of shape `(depth, height)`.
///
/// # Arguments
///
/// * `volume` – 3D array `(depth, height, width)`.
/// * `slice_idx` – Width index to extract (0 ≤ `slice_idx` < width).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` when `slice_idx` is out of range.
pub fn reformat_sagittal(volume: &Array3<f64>, slice_idx: usize) -> NdimageResult<Array2<f64>> {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if slice_idx >= sx {
        return Err(NdimageError::InvalidInput(format!(
            "slice_idx {slice_idx} is out of range (width = {sx})"
        )));
    }
    let mut out = Array2::zeros((sz, sy));
    for iz in 0..sz {
        for iy in 0..sy {
            out[[iz, iy]] = volume[[iz, iy, slice_idx]];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Oblique slice
// ---------------------------------------------------------------------------

/// Sample an oblique plane defined by a centre point and a normal vector.
///
/// Two orthogonal basis vectors `u` and `v` are constructed from the normal
/// using a stable algorithm (Frisvad 2012).  The output image is of shape
/// `(size, size)` and the sample grid is centred at `point` with step
/// `spacing` in both `u` and `v` directions.  Trilinear interpolation is
/// used to sub-voxel precision.
///
/// # Arguments
///
/// * `volume` – Source 3D volume `(depth, height, width)`.
/// * `point` – Centre of the output plane in voxel coordinates `(z, y, x)`.
/// * `normal` – Normal vector of the plane (need not be normalised).
/// * `size` – Number of pixels along each axis of the output image.
/// * `spacing` – Physical spacing between adjacent output pixels in voxels.
///
/// # Errors
///
/// * `NdimageError::InvalidInput` if `size` is 0 or `normal` is the zero
///   vector.
pub fn oblique_slice(
    volume: &Array3<f64>,
    point: (f64, f64, f64),
    normal: (f64, f64, f64),
    size: usize,
    spacing: f64,
) -> NdimageResult<Array2<f64>> {
    if size == 0 {
        return Err(NdimageError::InvalidInput(
            "size must be greater than 0".to_string(),
        ));
    }
    let (nz, ny, nx) = normal;
    let len = (nz * nz + ny * ny + nx * nx).sqrt();
    if len < 1e-12 {
        return Err(NdimageError::InvalidInput(
            "normal vector must be non-zero".to_string(),
        ));
    }
    // Normalise.
    let (nz, ny, nx) = (nz / len, ny / len, nx / len);

    // Frisvad 2012 ONB construction — avoids singularity at (0,0,-1).
    let (uz, uy, ux, vz, vy, vx) = if nz < -1.0 + 1e-9 {
        // Special case: normal ≈ (0,0,-1).
        (0.0_f64, -1.0_f64, 0.0_f64, -1.0_f64, 0.0_f64, 0.0_f64)
    } else {
        let a = 1.0 / (1.0 + nz);
        let b = -nx * ny * a;
        // u = (1 - nx²·a, b, -nx)
        let ux = 1.0 - nx * nx * a;
        let uy = b;
        let uz = -nx;
        // v = (b, 1 - ny²·a, -ny)
        let vx = b;
        let vy = 1.0 - ny * ny * a;
        let vz = -ny;
        (uz, uy, ux, vz, vy, vx)
    };

    let half = (size as f64 - 1.0) * 0.5;
    let mut out = Array2::zeros((size, size));

    for row in 0..size {
        let sv = (row as f64 - half) * spacing;
        for col in 0..size {
            let su = (col as f64 - half) * spacing;
            let z = point.0 + su * uz + sv * vz;
            let y = point.1 + su * uy + sv * vy;
            let x = point.2 + su * ux + sv * vx;
            out[[row, col]] = trilinear_sample(volume, z, y, x);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Intensity projections
// ---------------------------------------------------------------------------

/// Compute the Maximum Intensity Projection (MIP) along a given axis.
///
/// For each ray parallel to `axis`, the output pixel is the maximum value
/// encountered.
///
/// # Arguments
///
/// * `volume` – 3D array `(depth, height, width)`.
/// * `axis` – Projection axis: 0 = depth, 1 = height, 2 = width.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for an unknown axis or empty volume.
pub fn maximum_intensity_projection(
    volume: &Array3<f64>,
    axis: usize,
) -> NdimageResult<Array2<f64>> {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }
    match axis {
        0 => {
            // Project along z → output (y, x)
            let mut out = Array2::from_elem((sy, sx), f64::NEG_INFINITY);
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        let v = volume[[iz, iy, ix]];
                        if v > out[[iy, ix]] {
                            out[[iy, ix]] = v;
                        }
                    }
                }
            }
            Ok(out)
        }
        1 => {
            // Project along y → output (z, x)
            let mut out = Array2::from_elem((sz, sx), f64::NEG_INFINITY);
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        let v = volume[[iz, iy, ix]];
                        if v > out[[iz, ix]] {
                            out[[iz, ix]] = v;
                        }
                    }
                }
            }
            Ok(out)
        }
        2 => {
            // Project along x → output (z, y)
            let mut out = Array2::from_elem((sz, sy), f64::NEG_INFINITY);
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        let v = volume[[iz, iy, ix]];
                        if v > out[[iz, iy]] {
                            out[[iz, iy]] = v;
                        }
                    }
                }
            }
            Ok(out)
        }
        other => Err(NdimageError::InvalidInput(format!(
            "axis {other} is out of range; must be 0, 1, or 2"
        ))),
    }
}

/// Compute the Average Intensity Projection (AIP) along a given axis.
///
/// For each ray parallel to `axis`, the output pixel is the arithmetic mean
/// of all values along the ray.
///
/// # Arguments
///
/// * `volume` – 3D array `(depth, height, width)`.
/// * `axis` – Projection axis: 0 = depth, 1 = height, 2 = width.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for an unknown axis or empty volume.
pub fn average_intensity_projection(
    volume: &Array3<f64>,
    axis: usize,
) -> NdimageResult<Array2<f64>> {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }
    match axis {
        0 => {
            let mut out = Array2::zeros((sy, sx));
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        out[[iy, ix]] += volume[[iz, iy, ix]];
                    }
                }
            }
            out.mapv_inplace(|v| v / sz as f64);
            Ok(out)
        }
        1 => {
            let mut out = Array2::zeros((sz, sx));
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        out[[iz, ix]] += volume[[iz, iy, ix]];
                    }
                }
            }
            out.mapv_inplace(|v| v / sy as f64);
            Ok(out)
        }
        2 => {
            let mut out = Array2::zeros((sz, sy));
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        out[[iz, iy]] += volume[[iz, iy, ix]];
                    }
                }
            }
            out.mapv_inplace(|v| v / sx as f64);
            Ok(out)
        }
        other => Err(NdimageError::InvalidInput(format!(
            "axis {other} is out of range; must be 0, 1, or 2"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Volume rendering transfer function
// ---------------------------------------------------------------------------

/// A simple pre-integrated transfer function for direct volume rendering.
///
/// Maps scalar density values to RGBA colour.  The mapping is designed to
/// mimic a typical medical volume rendering preset:
///
/// | Density range | Colour              | Interpretation         |
/// |---------------|---------------------|------------------------|
/// | 0–0.1        | transparent black   | air / background       |
/// | 0.1–0.3      | semi-transparent blue | soft tissue          |
/// | 0.3–0.6      | semi-opaque orange  | dense soft tissue      |
/// | 0.6–0.9      | opaque white-yellow | bone                   |
/// | 0.9–1.0      | fully opaque white  | very dense material    |
///
/// Density values outside `[0, 1]` are clamped to the nearest boundary.
///
/// # Returns
///
/// A tuple `(R, G, B, A)` where each component is in `[0, 1]`.
pub fn volume_rendering_transfer_function(density: f64) -> (f64, f64, f64, f64) {
    let d = density.clamp(0.0, 1.0);

    // Piecewise linear segments over density.
    if d < 0.1 {
        // Air: fully transparent.
        let t = d / 0.1;
        (0.0, 0.0, t * 0.1, t * 0.05)
    } else if d < 0.3 {
        // Soft tissue: blue-ish, semi-transparent.
        let t = (d - 0.1) / 0.2;
        let alpha = 0.05 + t * 0.25; // 0.05 → 0.30
        let r = 0.1 * t;
        let g = 0.2 * t;
        let b = 0.5 + t * 0.3; // 0.5 → 0.8
        (r, g, b, alpha)
    } else if d < 0.6 {
        // Dense soft tissue: transitions from blue to orange.
        let t = (d - 0.3) / 0.3;
        let alpha = 0.30 + t * 0.40; // 0.30 → 0.70
        let r = 0.1 + t * 0.8; // 0.1 → 0.9
        let g = 0.2 + t * 0.4; // 0.2 → 0.6
        let b = 0.8 - t * 0.7; // 0.8 → 0.1
        (r, g, b, alpha)
    } else if d < 0.9 {
        // Bone: opaque white-yellow.
        let t = (d - 0.6) / 0.3;
        let alpha = 0.70 + t * 0.25; // 0.70 → 0.95
        let r = 0.9 + t * 0.1; // 0.9 → 1.0
        let g = 0.6 + t * 0.35; // 0.6 → 0.95
        let b = 0.1 + t * 0.85; // 0.1 → 0.95
        (r, g, b, alpha)
    } else {
        // Very dense: fully opaque white.
        let t = (d - 0.9) / 0.1;
        let alpha = 0.95 + t * 0.05;
        (1.0, 0.95 + t * 0.05, 0.95 + t * 0.05, alpha)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn small_vol() -> Array3<f64> {
        // 4 × 3 × 5 volume filled with voxel index as value.
        Array3::from_shape_fn((4, 3, 5), |(z, y, x)| (z * 100 + y * 10 + x) as f64)
    }

    #[test]
    fn test_reformat_axial_shape() {
        let vol = small_vol();
        let slice = reformat_axial(&vol, 2).expect("reformat_axial failed");
        assert_eq!(slice.shape(), &[3, 5]);
        // Check a known value: z=2, y=1, x=3 → 213.
        assert_eq!(slice[[1, 3]], 213.0);
    }

    #[test]
    fn test_reformat_axial_oob() {
        let vol = small_vol();
        assert!(reformat_axial(&vol, 4).is_err());
    }

    #[test]
    fn test_reformat_coronal_shape() {
        let vol = small_vol();
        let slice = reformat_coronal(&vol, 1).expect("reformat_coronal failed");
        assert_eq!(slice.shape(), &[4, 5]);
        // z=0, y=1 → 10 + x; at x=2 → 12.
        assert_eq!(slice[[0, 2]], 12.0);
    }

    #[test]
    fn test_reformat_sagittal_shape() {
        let vol = small_vol();
        let slice = reformat_sagittal(&vol, 3).expect("reformat_sagittal failed");
        assert_eq!(slice.shape(), &[4, 3]);
        // z=1, y=2, x=3 → 123.
        assert_eq!(slice[[1, 2]], 123.0);
    }

    #[test]
    fn test_oblique_slice_axial_equivalent() {
        // An oblique slice with normal (1,0,0) through the centre of a uniform
        // volume should produce a constant-value image.
        let vol = Array3::from_elem((8, 8, 8), 42.0_f64);
        let centre = (4.0, 4.0, 4.0);
        let normal = (1.0, 0.0, 0.0);
        let slice = oblique_slice(&vol, centre, normal, 4, 1.0).expect("oblique_slice failed");
        for v in slice.iter() {
            assert!((*v - 42.0).abs() < 1e-9, "Expected 42.0 got {v}");
        }
    }

    #[test]
    fn test_mip_axis0_shape() {
        let vol = small_vol(); // 4×3×5
        let mip = maximum_intensity_projection(&vol, 0).expect("MIP failed");
        assert_eq!(mip.shape(), &[3, 5]);
        // Maximum along z for (y=0, x=0) should be z=3 → 300.
        assert_eq!(mip[[0, 0]], 300.0);
    }

    #[test]
    fn test_aip_axis1_shape() {
        let vol = small_vol(); // 4×3×5
        let aip = average_intensity_projection(&vol, 1).expect("AIP failed");
        assert_eq!(aip.shape(), &[4, 5]);
    }

    #[test]
    fn test_mip_invalid_axis() {
        let vol = small_vol();
        assert!(maximum_intensity_projection(&vol, 3).is_err());
    }

    #[test]
    fn test_transfer_function_ranges() {
        // All outputs must be in [0,1].
        for i in 0..=100 {
            let d = i as f64 / 100.0;
            let (r, g, b, a) = volume_rendering_transfer_function(d);
            for &c in &[r, g, b, a] {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "component {c} out of [0,1] at density {d}"
                );
            }
        }
        // Boundary clamping.
        let (_, _, _, a_neg) = volume_rendering_transfer_function(-1.0);
        let (_, _, _, a_over) = volume_rendering_transfer_function(2.0);
        assert!(a_neg.is_finite());
        assert!((a_over - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_trilinear_sample_exact() {
        let vol = small_vol();
        // Exact integer position should equal the array value.
        for z in 0_usize..4 {
            for y in 0_usize..3 {
                for x in 0_usize..5 {
                    let expected = vol[[z, y, x]];
                    let got = trilinear_sample(&vol, z as f64, y as f64, x as f64);
                    assert!(
                        (got - expected).abs() < 1e-9,
                        "trilinear mismatch at ({z},{y},{x}): expected {expected}, got {got}"
                    );
                }
            }
        }
    }
}
