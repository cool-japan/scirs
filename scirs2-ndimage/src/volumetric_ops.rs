//! Volumetric Image Operations
//!
//! This module provides geometric and spatial transformations for 3D
//! volumetric (`Array3<f64>`) data:
//!
//! | Function                | Description                                   |
//! |-------------------------|-----------------------------------------------|
//! | [`zoom3d`]              | Resize / zoom with bilinear or nearest-neighbour interpolation |
//! | [`rotate3d`]            | Euler-angle rotation (ZYX convention)         |
//! | [`affine_transform3d`]  | Arbitrary 3×3 or 4×4 affine transform         |
//! | [`shift3d`]             | Sub-voxel translation                         |
//! | [`flip3d`]              | Flip (mirror) along a single axis             |
//! | [`pad3d`]               | Extend volume with zero, edge, or reflect fill |
//!
//! # Interpolation
//!
//! All resampling operations (`zoom3d`, `rotate3d`, `affine_transform3d`,
//! `shift3d`) support two interpolation orders:
//!
//! * **Order 0** – nearest-neighbour (fast, introduces aliasing)
//! * **Order 1** – trilinear interpolation (smooth, default)
//!
//! Higher-order methods (cubic, etc.) are not yet implemented and will return
//! `NdimageError::NotImplementedError`.
//!
//! # Padding for Resampling
//!
//! Voxels that map outside the source volume during resampling are filled with
//! `0.0` (zero-boundary).
//!
//! # References
//!
//! * Foley et al., "Computer Graphics: Principles and Practice", 2nd ed.
//! * Gonzalez & Woods, "Digital Image Processing", 3rd ed.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// Interpolation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Trilinear interpolation inside a volume.
///
/// Coordinates outside `[0, dim-1]` are clamped to the edge.
#[inline]
fn trilinear(volume: &Array3<f64>, z: f64, y: f64, x: f64) -> f64 {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);

    if sz == 0 || sy == 0 || sx == 0 {
        return 0.0;
    }

    // Clamp into valid range
    let z = z.clamp(0.0, (sz as f64) - 1.0);
    let y = y.clamp(0.0, (sy as f64) - 1.0);
    let x = x.clamp(0.0, (sx as f64) - 1.0);

    let iz0 = z.floor() as usize;
    let iy0 = y.floor() as usize;
    let ix0 = x.floor() as usize;
    let iz1 = (iz0 + 1).min(sz - 1);
    let iy1 = (iy0 + 1).min(sy - 1);
    let ix1 = (ix0 + 1).min(sx - 1);

    let dz = z - iz0 as f64;
    let dy = y - iy0 as f64;
    let dx = x - ix0 as f64;

    // Trilinear interpolation over 8 corners
    let v000 = volume[[iz0, iy0, ix0]];
    let v001 = volume[[iz0, iy0, ix1]];
    let v010 = volume[[iz0, iy1, ix0]];
    let v011 = volume[[iz0, iy1, ix1]];
    let v100 = volume[[iz1, iy0, ix0]];
    let v101 = volume[[iz1, iy0, ix1]];
    let v110 = volume[[iz1, iy1, ix0]];
    let v111 = volume[[iz1, iy1, ix1]];

    let c00 = v000 * (1.0 - dx) + v001 * dx;
    let c01 = v010 * (1.0 - dx) + v011 * dx;
    let c10 = v100 * (1.0 - dx) + v101 * dx;
    let c11 = v110 * (1.0 - dx) + v111 * dx;

    let c0 = c00 * (1.0 - dy) + c01 * dy;
    let c1 = c10 * (1.0 - dy) + c11 * dy;

    c0 * (1.0 - dz) + c1 * dz
}

/// Nearest-neighbour lookup inside a volume with zero-boundary.
#[inline]
fn nearest_neighbor(volume: &Array3<f64>, z: f64, y: f64, x: f64) -> f64 {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    let iz = z.round() as isize;
    let iy = y.round() as isize;
    let ix = x.round() as isize;
    if iz < 0 || iy < 0 || ix < 0 || iz as usize >= sz || iy as usize >= sy || ix as usize >= sx {
        return 0.0;
    }
    volume[[iz as usize, iy as usize, ix as usize]]
}

/// Sample the volume at a continuous coordinate using the specified interpolation order.
///
/// * order 0 → nearest-neighbour (zero-boundary)
/// * order 1 → trilinear (clamp-to-edge)
fn sample(volume: &Array3<f64>, z: f64, y: f64, x: f64, order: usize) -> f64 {
    match order {
        0 => nearest_neighbor(volume, z, y, x),
        _ => trilinear(volume, z, y, x),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// zoom3d
// ─────────────────────────────────────────────────────────────────────────────

/// Zoom / resize a 3D volume by the given scale factors.
///
/// The output shape is `ceil(input_shape[i] * factors[i])` for each axis.
/// Fractional coordinates in the source volume are sampled using `order`-th
/// order interpolation.
///
/// # Arguments
///
/// * `volume`  - Input volumetric array.
/// * `factors` - Scale factors `[fz, fy, fx]` (each must be > 0).
/// * `order`   - Interpolation order: 0 = nearest-neighbour, 1 = trilinear.
///               Higher orders are not yet supported.
///
/// # Errors
///
/// * `NdimageError::InvalidInput` if any factor is ≤ 0.
/// * `NdimageError::NotImplementedError` if order > 1.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::volumetric_ops::zoom3d;
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((4, 6, 8));
/// let out = zoom3d(&vol, [2.0, 2.0, 2.0], 1).unwrap();
/// assert_eq!(out.shape(), [8, 12, 16]);
/// ```
pub fn zoom3d(volume: &Array3<f64>, factors: [f64; 3], order: usize) -> NdimageResult<Array3<f64>> {
    for (axis, &f) in factors.iter().enumerate() {
        if f <= 0.0 {
            return Err(NdimageError::InvalidInput(format!(
                "factors[{}] must be positive, got {}",
                axis, f
            )));
        }
    }
    if order > 1 {
        return Err(NdimageError::NotImplementedError(format!(
            "zoom3d only supports order 0 or 1, got {}",
            order
        )));
    }

    let vs = volume.shape();
    let out_z = ((vs[0] as f64) * factors[0]).ceil() as usize;
    let out_y = ((vs[1] as f64) * factors[1]).ceil() as usize;
    let out_x = ((vs[2] as f64) * factors[2]).ceil() as usize;

    // Guard against zero-size output
    let out_z = out_z.max(1);
    let out_y = out_y.max(1);
    let out_x = out_x.max(1);

    // Scale factors mapping output → input coordinates
    let sz = if out_z > 1 {
        (vs[0] as f64 - 1.0) / (out_z as f64 - 1.0)
    } else {
        0.0
    };
    let sy = if out_y > 1 {
        (vs[1] as f64 - 1.0) / (out_y as f64 - 1.0)
    } else {
        0.0
    };
    let sxf = if out_x > 1 {
        (vs[2] as f64 - 1.0) / (out_x as f64 - 1.0)
    } else {
        0.0
    };

    let out = Array3::from_shape_fn((out_z, out_y, out_x), |(oz, oy, ox)| {
        let src_z = oz as f64 * sz;
        let src_y = oy as f64 * sy;
        let src_x = ox as f64 * sxf;
        sample(volume, src_z, src_y, src_x, order)
    });
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// rotate3d
// ─────────────────────────────────────────────────────────────────────────────

/// 3D rotation using ZYX Euler angles.
///
/// Rotates the volume around the geometric centre using the intrinsic ZYX
/// (yaw-pitch-roll) convention:  first rotate by `angles[2]` around Z, then
/// `angles[1]` around Y, then `angles[0]` around X.  All angles are in
/// **radians**.
///
/// The output has the same shape as the input.  Voxels that rotate outside
/// the boundary of the source volume are set to 0.
///
/// # Arguments
///
/// * `volume` - Input volumetric array.
/// * `angles` - Euler angles `[rx, ry, rz]` in radians (X, Y, Z rotations).
/// * `order`  - Interpolation order: 0 = nearest-neighbour, 1 = trilinear.
///
/// # Errors
///
/// * `NdimageError::NotImplementedError` if order > 1.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::volumetric_ops::rotate3d;
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((8, 8, 8));
/// // 90-degree rotation around Z
/// let out = rotate3d(&vol, [0.0, 0.0, std::f64::consts::FRAC_PI_2], 1).unwrap();
/// assert_eq!(out.shape(), [8, 8, 8]);
/// ```
pub fn rotate3d(
    volume: &Array3<f64>,
    angles: [f64; 3],
    order: usize,
) -> NdimageResult<Array3<f64>> {
    if order > 1 {
        return Err(NdimageError::NotImplementedError(format!(
            "rotate3d only supports order 0 or 1, got {}",
            order
        )));
    }

    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);

    // Volume center in voxel coordinates
    let cz = (sz as f64 - 1.0) / 2.0;
    let cy = (sy as f64 - 1.0) / 2.0;
    let cx = (sx as f64 - 1.0) / 2.0;

    // Build the combined rotation matrix R = Rx * Ry * Rz (applied right-to-left)
    let (rx, ry, rz) = (angles[0], angles[1], angles[2]);

    let cos_x = rx.cos();
    let sin_x = rx.sin();
    let cos_y = ry.cos();
    let sin_y = ry.sin();
    let cos_z = rz.cos();
    let sin_z = rz.sin();

    // Rx
    let rx_mat = [
        [1.0f64, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x],
    ];
    // Ry
    let ry_mat = [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]];
    // Rz
    let rz_mat = [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]];

    // R = Rx * Ry * Rz
    let ryx = mat3x3_mul(&rx_mat, &ry_mat);
    let r = mat3x3_mul(&ryx, &rz_mat);

    // Inverse rotation (transpose for orthogonal matrix)
    let r_inv = mat3x3_transpose(&r);

    let out = Array3::from_shape_fn((sz, sy, sx), |(oz, oy, ox)| {
        // Translate to centre, rotate backwards, translate back
        let dz = oz as f64 - cz;
        let dy = oy as f64 - cy;
        let dx = ox as f64 - cx;

        let src_z = r_inv[0][0] * dz + r_inv[0][1] * dy + r_inv[0][2] * dx + cz;
        let src_y = r_inv[1][0] * dz + r_inv[1][1] * dy + r_inv[1][2] * dx + cy;
        let src_x = r_inv[2][0] * dz + r_inv[2][1] * dy + r_inv[2][2] * dx + cx;

        // Out-of-bounds → 0
        if src_z < 0.0
            || src_y < 0.0
            || src_x < 0.0
            || src_z > (sz as f64 - 1.0)
            || src_y > (sy as f64 - 1.0)
            || src_x > (sx as f64 - 1.0)
        {
            return 0.0;
        }
        sample(volume, src_z, src_y, src_x, order)
    });
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// affine_transform3d
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a 3D affine transform to a volume.
///
/// The `matrix` can be either 3×3 (linear part only) or 4×4 (with translation
/// in homogeneous coordinates, last row `[0 0 0 1]`).
///
/// The transform maps **output** voxel coordinates to **input** voxel
/// coordinates (pull-back / inverse-mapping convention), which is standard
/// for image resampling.  Pass the *inverse* of your desired forward transform.
///
/// # Arguments
///
/// * `volume` - Input volumetric array.
/// * `matrix` - 3×3 or 4×4 affine matrix (output→input mapping).
/// * `order`  - Interpolation order: 0 = nearest-neighbour, 1 = trilinear.
///
/// # Errors
///
/// * `NdimageError::InvalidInput` if the matrix is not 3×3 or 4×4.
/// * `NdimageError::NotImplementedError` if order > 1.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::volumetric_ops::affine_transform3d;
/// use scirs2_core::ndarray::{Array2, Array3};
///
/// let vol = Array3::<f64>::ones((8, 8, 8));
/// // Identity transform (no change)
/// let identity = Array2::eye(4);
/// let out = affine_transform3d(&vol, &identity, 1).unwrap();
/// assert_eq!(out.shape(), [8, 8, 8]);
/// ```
pub fn affine_transform3d(
    volume: &Array3<f64>,
    matrix: &Array2<f64>,
    order: usize,
) -> NdimageResult<Array3<f64>> {
    let mshape = matrix.shape();
    if !(mshape == [3, 3] || mshape == [4, 4]) {
        return Err(NdimageError::InvalidInput(format!(
            "matrix must be 3×3 or 4×4, got {}×{}",
            mshape[0], mshape[1]
        )));
    }
    if order > 1 {
        return Err(NdimageError::NotImplementedError(format!(
            "affine_transform3d only supports order 0 or 1, got {}",
            order
        )));
    }

    // Extract linear part and translation
    let a00 = matrix[[0, 0]];
    let a01 = matrix[[0, 1]];
    let a02 = matrix[[0, 2]];
    let a10 = matrix[[1, 0]];
    let a11 = matrix[[1, 1]];
    let a12 = matrix[[1, 2]];
    let a20 = matrix[[2, 0]];
    let a21 = matrix[[2, 1]];
    let a22 = matrix[[2, 2]];

    let (tz, ty, tx) = if mshape[0] == 4 {
        (matrix[[0, 3]], matrix[[1, 3]], matrix[[2, 3]])
    } else {
        (0.0, 0.0, 0.0)
    };

    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);

    let out = Array3::from_shape_fn((sz, sy, sx), |(oz, oy, ox)| {
        let oz = oz as f64;
        let oy = oy as f64;
        let ox = ox as f64;

        let src_z = a00 * oz + a01 * oy + a02 * ox + tz;
        let src_y = a10 * oz + a11 * oy + a12 * ox + ty;
        let src_x = a20 * oz + a21 * oy + a22 * ox + tx;

        if src_z < 0.0
            || src_y < 0.0
            || src_x < 0.0
            || src_z > (sz as f64 - 1.0)
            || src_y > (sy as f64 - 1.0)
            || src_x > (sx as f64 - 1.0)
        {
            return 0.0;
        }
        sample(volume, src_z, src_y, src_x, order)
    });
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// shift3d
// ─────────────────────────────────────────────────────────────────────────────

/// Translate a volume by a sub-voxel shift.
///
/// Each output voxel at `(z, y, x)` is sampled from the source at
/// `(z + shifts[0], y + shifts[1], x + shifts[2])`.  Fractional shifts
/// are handled by trilinear interpolation (order = 1) or nearest-neighbour
/// (order = 0).  Voxels shifted outside the volume boundary are set to 0.
///
/// # Arguments
///
/// * `volume` - Input volumetric array.
/// * `shifts` - Sub-voxel shifts `[dz, dy, dx]` (positive = shift right/down/forward).
/// * `order`  - Interpolation order: 0 = nearest-neighbour, 1 = trilinear.
///
/// # Errors
///
/// * `NdimageError::NotImplementedError` if order > 1.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::volumetric_ops::shift3d;
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((6, 6, 6));
/// let out = shift3d(&vol, [0.5, 0.5, 0.5], 1).unwrap();
/// assert_eq!(out.shape(), [6, 6, 6]);
/// ```
pub fn shift3d(volume: &Array3<f64>, shifts: [f64; 3], order: usize) -> NdimageResult<Array3<f64>> {
    if order > 1 {
        return Err(NdimageError::NotImplementedError(format!(
            "shift3d only supports order 0 or 1, got {}",
            order
        )));
    }

    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);

    let out = Array3::from_shape_fn((sz, sy, sx), |(oz, oy, ox)| {
        let src_z = oz as f64 + shifts[0];
        let src_y = oy as f64 + shifts[1];
        let src_x = ox as f64 + shifts[2];

        if src_z < 0.0
            || src_y < 0.0
            || src_x < 0.0
            || src_z > (sz as f64 - 1.0)
            || src_y > (sy as f64 - 1.0)
            || src_x > (sx as f64 - 1.0)
        {
            return 0.0;
        }
        sample(volume, src_z, src_y, src_x, order)
    });
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// flip3d
// ─────────────────────────────────────────────────────────────────────────────

/// Flip (mirror) a volume along one axis.
///
/// # Arguments
///
/// * `volume` - Input volumetric array.
/// * `axis`   - Axis to flip: 0 = Z, 1 = Y, 2 = X.
///
/// # Errors
///
/// * `NdimageError::InvalidInput` if axis > 2.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::volumetric_ops::flip3d;
/// use scirs2_core::ndarray::Array3;
///
/// let mut vol = Array3::<f64>::zeros((4, 4, 4));
/// vol[[0, 0, 0]] = 1.0;
/// let out = flip3d(&vol, 0).unwrap();
/// assert_eq!(out[[3, 0, 0]], 1.0);
/// assert_eq!(out[[0, 0, 0]], 0.0);
/// ```
pub fn flip3d(volume: &Array3<f64>, axis: usize) -> NdimageResult<Array3<f64>> {
    if axis > 2 {
        return Err(NdimageError::InvalidInput(format!(
            "axis must be 0, 1, or 2, got {}",
            axis
        )));
    }
    let vs = volume.shape();
    let (sz, sy, sx) = (vs[0], vs[1], vs[2]);

    let out = Array3::from_shape_fn((sz, sy, sx), |(iz, iy, ix)| {
        let src_z = if axis == 0 { sz - 1 - iz } else { iz };
        let src_y = if axis == 1 { sy - 1 - iy } else { iy };
        let src_x = if axis == 2 { sx - 1 - ix } else { ix };
        volume[[src_z, src_y, src_x]]
    });
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// pad3d
// ─────────────────────────────────────────────────────────────────────────────

/// Padding mode used by [`pad3d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode3D {
    /// Fill with 0.0.
    Zero,
    /// Replicate the edge voxel (clamp-to-edge).
    Edge,
    /// Mirror-reflect around the boundary.
    Reflect,
    /// Fill with a specified constant (use the `Zero` variant and a pre-filled
    /// array, or use the helper [`pad3d_constant`] for an arbitrary constant).
    Constant,
}

/// Pad a 3D volume with a given mode and pad widths.
///
/// # Arguments
///
/// * `volume`    - Input volumetric array.
/// * `pad_width` - Per-axis pad widths `[(before_z, after_z), (before_y, after_y), (before_x, after_x)]`.
/// * `mode`      - Padding mode.
/// * `constant`  - Value to use when `mode == PadMode3D::Constant` or `PadMode3D::Zero`.
///
/// # Errors
///
/// None (always succeeds for valid inputs).
///
/// # Example
///
/// ```
/// use scirs2_ndimage::volumetric_ops::{pad3d, PadMode3D};
/// use scirs2_core::ndarray::Array3;
///
/// let vol = Array3::<f64>::ones((4, 5, 6));
/// let out = pad3d(&vol, [(1, 2), (0, 1), (2, 2)], PadMode3D::Zero, 0.0).unwrap();
/// assert_eq!(out.shape(), [7, 6, 10]);
/// ```
pub fn pad3d(
    volume: &Array3<f64>,
    pad_width: [(usize, usize); 3],
    mode: PadMode3D,
    constant: f64,
) -> NdimageResult<Array3<f64>> {
    let vs = volume.shape();
    let (in_z, in_y, in_x) = (vs[0], vs[1], vs[2]);

    let out_z = in_z + pad_width[0].0 + pad_width[0].1;
    let out_y = in_y + pad_width[1].0 + pad_width[1].1;
    let out_x = in_x + pad_width[2].0 + pad_width[2].1;

    let pz0 = pad_width[0].0;
    let py0 = pad_width[1].0;
    let px0 = pad_width[2].0;

    let out = Array3::from_shape_fn((out_z, out_y, out_x), |(oz, oy, ox)| {
        let src_z = oz as isize - pz0 as isize;
        let src_y = oy as isize - py0 as isize;
        let src_x = ox as isize - px0 as isize;

        let in_bounds_z = src_z >= 0 && (src_z as usize) < in_z;
        let in_bounds_y = src_y >= 0 && (src_y as usize) < in_y;
        let in_bounds_x = src_x >= 0 && (src_x as usize) < in_x;

        if in_bounds_z && in_bounds_y && in_bounds_x {
            return volume[[src_z as usize, src_y as usize, src_x as usize]];
        }

        match mode {
            PadMode3D::Zero | PadMode3D::Constant => constant,
            PadMode3D::Edge => {
                let cz = src_z.clamp(0, in_z as isize - 1) as usize;
                let cy = src_y.clamp(0, in_y as isize - 1) as usize;
                let cx = src_x.clamp(0, in_x as isize - 1) as usize;
                volume[[cz, cy, cx]]
            }
            PadMode3D::Reflect => {
                let rz = reflect_pad_index(src_z, in_z);
                let ry = reflect_pad_index(src_y, in_y);
                let rx = reflect_pad_index(src_x, in_x);
                volume[[rz, ry, rx]]
            }
        }
    });
    Ok(out)
}

/// Reflect an out-of-range padded index into `[0, n-1]` using edge-inclusive
/// reflection.
#[inline]
fn reflect_pad_index(i: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }
    let n = n as isize;
    let period = 2 * n - 2;
    let mut r = i % period;
    if r < 0 {
        r += period;
    }
    if r >= n {
        r = period - r;
    }
    r as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Small 3×3 matrix helpers (no heap allocation)
// ─────────────────────────────────────────────────────────────────────────────

type Mat3 = [[f64; 3]; 3];

#[inline]
fn mat3x3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

#[inline]
fn mat3x3_transpose(a: &Mat3) -> Mat3 {
    let mut t = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    // ── zoom3d ────────────────────────────────────────────────────────────────

    #[test]
    fn test_zoom3d_double() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        let out = zoom3d(&vol, [2.0, 2.0, 2.0], 1).expect("zoom3d failed");
        assert_eq!(out.shape(), [8, 8, 8]);
        for &v in out.iter() {
            assert!((v - 1.0).abs() < 1e-10, "unexpected value {}", v);
        }
    }

    #[test]
    fn test_zoom3d_half() {
        let vol = Array3::<f64>::ones((8, 8, 8));
        let out = zoom3d(&vol, [0.5, 0.5, 0.5], 1).expect("zoom3d failed");
        // ceil(8 * 0.5) = 4
        assert_eq!(out.shape(), [4, 4, 4]);
    }

    #[test]
    fn test_zoom3d_anisotropic() {
        let vol = Array3::<f64>::ones((3, 4, 5));
        let out = zoom3d(&vol, [2.0, 1.0, 0.5], 0).expect("zoom3d anisotropic");
        assert_eq!(out.shape(), [6, 4, 3]);
    }

    #[test]
    fn test_zoom3d_invalid_factor() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        assert!(zoom3d(&vol, [0.0, 1.0, 1.0], 1).is_err());
        assert!(zoom3d(&vol, [1.0, -1.0, 1.0], 1).is_err());
    }

    #[test]
    fn test_zoom3d_unsupported_order() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        assert!(zoom3d(&vol, [1.0, 1.0, 1.0], 3).is_err());
    }

    // ── rotate3d ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rotate3d_identity_zero_angle() {
        let vol = Array3::<f64>::from_shape_fn((8, 8, 8), |(z, y, x)| (z + y + x) as f64);
        let out = rotate3d(&vol, [0.0, 0.0, 0.0], 1).expect("rotate3d identity");
        for iz in 1..7 {
            for iy in 1..7 {
                for ix in 1..7 {
                    assert!(
                        (out[[iz, iy, ix]] - vol[[iz, iy, ix]]).abs() < 1e-9,
                        "mismatch at [{},{},{}]",
                        iz,
                        iy,
                        ix
                    );
                }
            }
        }
    }

    #[test]
    fn test_rotate3d_360_deg() {
        let vol = Array3::<f64>::from_shape_fn((8, 8, 8), |(z, y, x)| (z + y + x) as f64);
        // Full rotation should nearly restore the original
        let out = rotate3d(&vol, [0.0, 0.0, 2.0 * std::f64::consts::PI], 1).expect("rotate3d 360");
        for iz in 2..6 {
            for iy in 2..6 {
                for ix in 2..6 {
                    assert!(
                        (out[[iz, iy, ix]] - vol[[iz, iy, ix]]).abs() < 1e-8,
                        "mismatch at [{},{},{}]: {} vs {}",
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
    fn test_rotate3d_shape_preserved() {
        let vol = Array3::<f64>::ones((7, 9, 5));
        let out = rotate3d(&vol, [0.3, 0.1, 0.5], 0).expect("rotate3d shape");
        assert_eq!(out.shape(), [7, 9, 5]);
    }

    // ── affine_transform3d ────────────────────────────────────────────────────

    #[test]
    fn test_affine_identity_4x4() {
        let vol = Array3::<f64>::from_shape_fn((6, 6, 6), |(z, y, x)| (z * 36 + y * 6 + x) as f64);
        let identity = Array2::eye(4);
        let out = affine_transform3d(&vol, &identity, 1).expect("affine identity");
        for iz in 0..6 {
            for iy in 0..6 {
                for ix in 0..6 {
                    assert!(
                        (out[[iz, iy, ix]] - vol[[iz, iy, ix]]).abs() < 1e-10,
                        "identity mismatch at [{},{},{}]",
                        iz,
                        iy,
                        ix
                    );
                }
            }
        }
    }

    #[test]
    fn test_affine_identity_3x3() {
        let vol = Array3::<f64>::ones((5, 5, 5));
        let identity = Array2::eye(3);
        let out = affine_transform3d(&vol, &identity, 1).expect("affine identity 3x3");
        assert_eq!(out.shape(), [5, 5, 5]);
        for &v in out.iter() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_affine_invalid_matrix() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        let bad = Array2::<f64>::zeros((2, 2));
        assert!(affine_transform3d(&vol, &bad, 1).is_err());
    }

    // ── shift3d ───────────────────────────────────────────────────────────────

    #[test]
    fn test_shift3d_zero_shift() {
        let vol = Array3::<f64>::from_shape_fn((6, 6, 6), |(z, y, x)| (z + y + x) as f64);
        let out = shift3d(&vol, [0.0, 0.0, 0.0], 1).expect("shift3d zero");
        for iz in 0..6 {
            for iy in 0..6 {
                for ix in 0..6 {
                    assert!(
                        (out[[iz, iy, ix]] - vol[[iz, iy, ix]]).abs() < 1e-10,
                        "shift zero mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn test_shift3d_integer_shift() {
        let mut vol = Array3::<f64>::zeros((8, 8, 8));
        vol[[3, 3, 3]] = 1.0;
        // Shift by (1, 0, 0) => the 1.0 should appear at (4, 3, 3) in output
        let out = shift3d(&vol, [-1.0, 0.0, 0.0], 0).expect("shift3d integer");
        assert_eq!(out.shape(), [8, 8, 8]);
        assert!(
            (out[[4, 3, 3]] - 1.0).abs() < 1e-10,
            "shifted voxel not found"
        );
    }

    #[test]
    fn test_shift3d_boundary() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        // Large shift pushes all source voxels out of bounds → all zeros
        let out = shift3d(&vol, [100.0, 0.0, 0.0], 1).expect("shift3d oob");
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }

    // ── flip3d ────────────────────────────────────────────────────────────────

    #[test]
    fn test_flip3d_axis0() {
        let mut vol = Array3::<f64>::zeros((4, 4, 4));
        vol[[0, 0, 0]] = 1.0;
        let out = flip3d(&vol, 0).expect("flip3d axis0");
        assert_eq!(out[[3, 0, 0]], 1.0);
        assert_eq!(out[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_flip3d_axis1() {
        let mut vol = Array3::<f64>::zeros((4, 4, 4));
        vol[[0, 0, 0]] = 2.0;
        let out = flip3d(&vol, 1).expect("flip3d axis1");
        assert_eq!(out[[0, 3, 0]], 2.0);
    }

    #[test]
    fn test_flip3d_axis2() {
        let mut vol = Array3::<f64>::zeros((4, 4, 4));
        vol[[0, 0, 0]] = 3.0;
        let out = flip3d(&vol, 2).expect("flip3d axis2");
        assert_eq!(out[[0, 0, 3]], 3.0);
    }

    #[test]
    fn test_flip3d_double_flip_identity() {
        let vol = Array3::<f64>::from_shape_fn((5, 5, 5), |(z, y, x)| (z * 25 + y * 5 + x) as f64);
        for axis in 0..3 {
            let flipped = flip3d(&vol, axis).expect("first flip");
            let restored = flip3d(&flipped, axis).expect("second flip");
            for iz in 0..5 {
                for iy in 0..5 {
                    for ix in 0..5 {
                        assert_eq!(
                            restored[[iz, iy, ix]],
                            vol[[iz, iy, ix]],
                            "double flip mismatch on axis {}",
                            axis
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_flip3d_invalid_axis() {
        let vol = Array3::<f64>::ones((4, 4, 4));
        assert!(flip3d(&vol, 3).is_err());
    }

    // ── pad3d ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_pad3d_zero_shape() {
        let vol = Array3::<f64>::ones((3, 4, 5));
        let out = pad3d(&vol, [(1, 2), (0, 1), (2, 2)], PadMode3D::Zero, 0.0).expect("pad3d zero");
        assert_eq!(out.shape(), [6, 5, 9]);
    }

    #[test]
    fn test_pad3d_zero_values() {
        let vol = Array3::<f64>::ones((3, 3, 3));
        let out =
            pad3d(&vol, [(1, 1), (1, 1), (1, 1)], PadMode3D::Zero, 0.0).expect("pad3d zero values");
        // Corners should be 0
        assert_eq!(out[[0, 0, 0]], 0.0);
        // Interior (shifted by 1) should be 1
        assert_eq!(out[[1, 1, 1]], 1.0);
        assert_eq!(out[[3, 3, 3]], 1.0);
        // After interior
        assert_eq!(out[[4, 4, 4]], 0.0);
    }

    #[test]
    fn test_pad3d_edge_mode() {
        let mut vol = Array3::<f64>::zeros((3, 3, 3));
        vol[[0, 0, 0]] = 5.0;
        let out = pad3d(&vol, [(1, 0), (1, 0), (1, 0)], PadMode3D::Edge, 0.0).expect("pad3d edge");
        // The padded corner should replicate vol[[0,0,0]] = 5.0
        assert_eq!(out[[0, 0, 0]], 5.0);
    }

    #[test]
    fn test_pad3d_reflect_mode() {
        let vol = Array3::<f64>::from_shape_fn((4, 4, 4), |(z, y, x)| (z + y + x) as f64);
        let out =
            pad3d(&vol, [(2, 2), (2, 2), (2, 2)], PadMode3D::Reflect, 0.0).expect("pad3d reflect");
        assert_eq!(out.shape(), [8, 8, 8]);
    }

    #[test]
    fn test_pad3d_no_padding() {
        let vol = Array3::<f64>::from_shape_fn((3, 4, 5), |(z, y, x)| (z * 20 + y * 5 + x) as f64);
        let out =
            pad3d(&vol, [(0, 0), (0, 0), (0, 0)], PadMode3D::Zero, 0.0).expect("pad3d no padding");
        assert_eq!(out.shape(), [3, 4, 5]);
        for iz in 0..3 {
            for iy in 0..4 {
                for ix in 0..5 {
                    assert_eq!(out[[iz, iy, ix]], vol[[iz, iy, ix]]);
                }
            }
        }
    }

    // ── mat helpers ───────────────────────────────────────────────────────────

    #[test]
    fn test_mat3x3_transpose_identity() {
        let i: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t = mat3x3_transpose(&i);
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(t[row][col], i[row][col]);
            }
        }
    }

    #[test]
    fn test_reflect_pad_index_boundary() {
        // index 0 should stay 0
        assert_eq!(reflect_pad_index(0, 5), 0);
        // index n-1 should stay n-1
        assert_eq!(reflect_pad_index(4, 5), 4);
        // index -1 should reflect to 1
        assert_eq!(reflect_pad_index(-1, 5), 1);
        // index n should reflect to n-2
        assert_eq!(reflect_pad_index(5, 5), 3);
    }
}
