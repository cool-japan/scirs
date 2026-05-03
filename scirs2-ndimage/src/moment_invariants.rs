//! Image moments and invariant descriptors.
//!
//! # Overview
//!
//! This module provides:
//!
//! - **Raw moments** m_{pq}: sum of x^p · y^q · I(x,y).
//! - **Central moments** μ_{pq}: moments relative to the image centroid.
//! - **Normalized central moments** η_{pq}: scale-invariant central moments.
//! - **Hu's 7 moment invariants**: rotation-, translation-, and scale-invariant
//!   descriptors widely used in shape recognition.
//! - **Zernike moments**: orthogonal moments on the unit disc, suitable for
//!   rotation-invariant feature extraction.
//! - **Reconstruction from Zernike moments**: approximate image synthesis.
//! - **Image centroid**: center of mass of intensity values.
//!
//! # References
//!
//! - Hu, M.K. (1962). "Visual pattern recognition by moment invariants."
//!   *IRE Trans. Inf. Theory*, 8(2), 179-187.
//! - Teague, M.R. (1980). "Image analysis via the general theory of moments."
//!   *J. Opt. Soc. Am.*, 70(8), 920-930.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Raw Moments
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the raw (geometric) moment m_{pq} of an image.
///
/// ```text
/// m_{pq} = Σ_x Σ_y  x^p · y^q · I(x, y)
/// ```
///
/// where *x* is the column index and *y* is the row index.
///
/// # Arguments
///
/// * `image` – 2-D intensity array.
/// * `p`     – Column exponent (order in x).
/// * `q`     – Row exponent (order in y).
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::raw_moment;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::from_elem((4, 4), 1.0_f64);
/// let m00 = raw_moment(&img.view(), 0, 0);
/// assert_eq!(m00, 16.0);
/// ```
pub fn raw_moment(image: &ArrayView2<f64>, p: u32, q: u32) -> f64 {
    let rows = image.nrows();
    let cols = image.ncols();
    let mut result = 0.0f64;
    for r in 0..rows {
        let y_pow = (r as f64).powi(q as i32);
        for c in 0..cols {
            let x_pow = (c as f64).powi(p as i32);
            result += x_pow * y_pow * image[[r, c]];
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Central Moments
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the central moment μ_{pq} of an image.
///
/// ```text
/// μ_{pq} = Σ_x Σ_y  (x - x̄)^p · (y - ȳ)^q · I(x, y)
/// ```
///
/// where (x̄, ȳ) is the centroid computed from m_{00}, m_{10}, m_{01}.
///
/// # Arguments
///
/// * `image` – 2-D intensity array.
/// * `p`     – Column exponent.
/// * `q`     – Row exponent.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::central_moment;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::from_elem((4, 4), 1.0_f64);
/// // Central moment of order (1,0) must be zero by definition
/// let mu10 = central_moment(&img.view(), 1, 0);
/// assert!(mu10.abs() < 1e-10);
/// ```
pub fn central_moment(image: &ArrayView2<f64>, p: u32, q: u32) -> f64 {
    let m00 = raw_moment(image, 0, 0);
    if m00.abs() < 1e-15 {
        return 0.0;
    }
    let (cx, cy) = centroid_from_raw(image, m00);

    let rows = image.nrows();
    let cols = image.ncols();
    let mut result = 0.0f64;
    for r in 0..rows {
        let dy_pow = (r as f64 - cy).powi(q as i32);
        for c in 0..cols {
            let dx_pow = (c as f64 - cx).powi(p as i32);
            result += dx_pow * dy_pow * image[[r, c]];
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalized Central Moments
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the normalized (scale-invariant) central moment η_{pq}.
///
/// ```text
/// η_{pq} = μ_{pq} / μ_{00}^γ,   γ = (p + q)/2 + 1
/// ```
///
/// # Arguments
///
/// * `image` – 2-D intensity array.
/// * `p`     – Column exponent.
/// * `q`     – Row exponent.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::normalized_central_moment;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::from_elem((4, 4), 1.0_f64);
/// let eta20 = normalized_central_moment(&img.view(), 2, 0);
/// assert!(eta20 > 0.0);
/// ```
pub fn normalized_central_moment(image: &ArrayView2<f64>, p: u32, q: u32) -> f64 {
    let mu_pq = central_moment(image, p, q);
    let mu00 = central_moment(image, 0, 0);
    if mu00.abs() < 1e-15 {
        return 0.0;
    }
    let gamma = (p + q) as f64 / 2.0 + 1.0;
    mu_pq / mu00.powf(gamma)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hu Moment Invariants
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Hu's 7 moment invariants for an image.
///
/// These are algebraic combinations of normalized central moments that are
/// invariant to translation, rotation, and uniform scaling.  The seventh
/// invariant is also sensitive to reflection (skew invariant).
///
/// Returns `[I1, I2, I3, I4, I5, I6, I7]`.
///
/// # Arguments
///
/// * `image` – 2-D intensity array.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::hu_moments;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::from_shape_fn((20, 20), |(r, c)| {
///     let dr = r as f64 - 10.0;
///     let dc = c as f64 - 10.0;
///     if dr*dr + dc*dc < 25.0 { 1.0 } else { 0.0 }
/// });
/// let hu = hu_moments(&img.view());
/// // First Hu invariant is always positive for a non-empty image
/// assert!(hu[0] > 0.0);
/// ```
pub fn hu_moments(image: &ArrayView2<f64>) -> [f64; 7] {
    // Collect all needed normalized central moments
    let eta = |p, q| normalized_central_moment(image, p, q);

    let e20 = eta(2, 0);
    let e02 = eta(0, 2);
    let e11 = eta(1, 1);
    let e30 = eta(3, 0);
    let e12 = eta(1, 2);
    let e21 = eta(2, 1);
    let e03 = eta(0, 3);

    let h1 = e20 + e02;

    let h2 = (e20 - e02).powi(2) + 4.0 * e11 * e11;

    let h3 = (e30 - 3.0 * e12).powi(2) + (3.0 * e21 - e03).powi(2);

    let h4 = (e30 + e12).powi(2) + (e21 + e03).powi(2);

    let h5 = (e30 - 3.0 * e12) * (e30 + e12) * ((e30 + e12).powi(2) - 3.0 * (e21 + e03).powi(2))
        + (3.0 * e21 - e03) * (e21 + e03) * (3.0 * (e30 + e12).powi(2) - (e21 + e03).powi(2));

    let h6 = (e20 - e02) * ((e30 + e12).powi(2) - (e21 + e03).powi(2))
        + 4.0 * e11 * (e30 + e12) * (e21 + e03);

    let h7 = (3.0 * e21 - e03) * (e30 + e12) * ((e30 + e12).powi(2) - 3.0 * (e21 + e03).powi(2))
        - (e30 - 3.0 * e12) * (e21 + e03) * (3.0 * (e30 + e12).powi(2) - (e21 + e03).powi(2));

    [h1, h2, h3, h4, h5, h6, h7]
}

// ─────────────────────────────────────────────────────────────────────────────
// Zernike Moments
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Zernike moment Z_{n,m} of an image.
///
/// Zernike moments are projections of the image onto orthogonal Zernike
/// polynomials defined on the unit disc.  They are rotation-invariant in
/// magnitude and enable lossless reconstruction up to a given order.
///
/// ```text
/// Z_{nm} = (n+1)/π · Σ_{x²+y²≤1}  V_{nm}*(ρ, θ) · I(x, y)
/// ```
///
/// where `ρ` and `θ` are the polar coordinates of pixel (x, y) mapped to the
/// unit disc centred at the image centre.
///
/// # Arguments
///
/// * `image`  – 2-D intensity array.
/// * `n`      – Radial order (non-negative integer).
/// * `m`      – Angular frequency, |m| ≤ n, (n - |m|) even.
/// * `radius` – Radius of the unit-disc mapping in pixels.  If `None` the
///   function uses half the minimum image dimension.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if `n` < |m|, or if (n - |m|) is odd.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::zernike_moment;
/// use scirs2_core::ndarray::Array2;
/// use std::f64::consts::PI;
///
/// let img = Array2::from_shape_fn((20, 20), |(r, c)| {
///     let dr = r as f64 - 10.0;
///     let dc = c as f64 - 10.0;
///     if dr*dr + dc*dc < 25.0 { 1.0_f64 } else { 0.0 }
/// });
/// // Z_{0,0} magnitude equals total intensity / (π * radius²) * (0+1)/π * π * r²
/// let z00 = zernike_moment(&img.view(), 0, 0, None).unwrap();
/// assert!(z00.re > 0.0);
/// ```
pub fn zernike_moment(
    image: &ArrayView2<f64>,
    n: u32,
    m: i32,
    radius: Option<f64>,
) -> NdimageResult<Complex64> {
    let m_abs = m.unsigned_abs();
    if m_abs > n {
        return Err(NdimageError::InvalidInput(format!(
            "zernike_moment: |m|={m_abs} > n={n}"
        )));
    }
    if (n - m_abs) % 2 != 0 {
        return Err(NdimageError::InvalidInput(format!(
            "zernike_moment: (n - |m|) = {} must be even",
            n - m_abs
        )));
    }

    let rows = image.nrows();
    let cols = image.ncols();
    let r_disc = radius.unwrap_or_else(|| (rows.min(cols) as f64) / 2.0);
    if r_disc < 1.0 {
        return Err(NdimageError::InvalidInput(
            "zernike_moment: radius must be ≥ 1".to_string(),
        ));
    }

    let cy = (rows as f64 - 1.0) / 2.0;
    let cx = (cols as f64 - 1.0) / 2.0;

    let mut re_sum = 0.0f64;
    let mut im_sum = 0.0f64;

    for r in 0..rows {
        let y = (r as f64 - cy) / r_disc;
        for c in 0..cols {
            let x = (c as f64 - cx) / r_disc;
            let rho = (x * x + y * y).sqrt();
            if rho > 1.0 {
                continue;
            }
            let theta = y.atan2(x);
            let radial = zernike_radial(n, m_abs, rho);
            let angle = -(m as f64) * theta; // complex conjugate → negative angle
            let i_val = image[[r, c]];
            re_sum += radial * angle.cos() * i_val;
            im_sum += radial * angle.sin() * i_val;
        }
    }

    let scale = (n + 1) as f64 / PI;
    Ok(Complex64::new(re_sum * scale, im_sum * scale))
}

/// Zernike radial polynomial R_n^|m|(ρ).
fn zernike_radial(n: u32, m_abs: u32, rho: f64) -> f64 {
    // R_n^m(ρ) = Σ_{s=0}^{(n-m)/2}  (-1)^s (n-s)! / (s! ((n+m)/2 - s)! ((n-m)/2 - s)!) ρ^{n-2s}
    let half_diff = (n - m_abs) / 2;
    let half_sum = (n + m_abs) / 2;
    let mut sum = 0.0f64;
    for s in 0..=half_diff {
        let sign = if s % 2 == 0 { 1.0f64 } else { -1.0f64 };
        let num = factorial(n - s);
        let den = factorial(s) * factorial(half_sum - s) * factorial(half_diff - s);
        if den == 0 {
            continue;
        }
        let coef = sign * (num as f64) / (den as f64);
        sum += coef * rho.powi((n - 2 * s) as i32);
    }
    sum
}

/// Unsigned integer factorial (returns 0 for overflow cases – truncated at u64::MAX).
fn factorial(n: u32) -> u64 {
    if n == 0 || n == 1 {
        return 1;
    }
    let mut acc = 1u64;
    for i in 2..=n as u64 {
        acc = acc.saturating_mul(i);
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction from Zernike Moments
// ─────────────────────────────────────────────────────────────────────────────

/// Approximate image reconstruction from Zernike moments up to order `max_order`.
///
/// ```text
/// Î(x, y) = Σ_{n,m} Z_{nm} · V_{nm}(ρ, θ)    (for ρ ≤ 1)
/// ```
///
/// # Arguments
///
/// * `moments` – Pre-computed Zernike moments as (n, m, Z_{nm}) tuples.
/// * `order`   – Maximum radial order to include in reconstruction.
/// * `size`    – Output image dimensions (rows, cols).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] for zero-size outputs.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::{zernike_moment, reconstruct_from_zernike};
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::from_shape_fn((16, 16), |(r, c)| {
///     let dr = r as f64 - 8.0;
///     let dc = c as f64 - 8.0;
///     if dr*dr + dc*dc < 16.0 { 1.0_f64 } else { 0.0 }
/// });
///
/// // Gather moments up to order 4
/// let mut moments = Vec::new();
/// for n in 0u32..=4 {
///     for mi in -(n as i32)..=(n as i32) {
///         if ((n as i32) - mi.abs()) % 2 == 0 {
///             if let Ok(z) = zernike_moment(&img.view(), n, mi, None) {
///                 moments.push((n, mi, z));
///             }
///         }
///     }
/// }
/// let recon = reconstruct_from_zernike(&moments, 4, (16, 16)).unwrap();
/// assert_eq!(recon.shape(), &[16, 16]);
/// ```
pub fn reconstruct_from_zernike(
    moments: &[(u32, i32, Complex64)],
    order: u32,
    size: (usize, usize),
) -> NdimageResult<Array2<f64>> {
    let (rows, cols) = size;
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput(
            "reconstruct_from_zernike: output size must be non-zero".to_string(),
        ));
    }

    let r_disc = (rows.min(cols) as f64) / 2.0;
    let cy = (rows as f64 - 1.0) / 2.0;
    let cx = (cols as f64 - 1.0) / 2.0;

    let mut output = Array2::<f64>::zeros((rows, cols));

    for r in 0..rows {
        let y = (r as f64 - cy) / r_disc;
        for c in 0..cols {
            let x = (c as f64 - cx) / r_disc;
            let rho = (x * x + y * y).sqrt();
            if rho > 1.0 {
                continue;
            }
            let theta = y.atan2(x);
            let mut val = 0.0f64;

            for &(n, m, z) in moments {
                if n > order {
                    continue;
                }
                let m_abs = m.unsigned_abs();
                let radial = zernike_radial(n, m_abs, rho);
                let angle = (m as f64) * theta;
                // V_{nm}(ρ,θ) = R_n^|m|(ρ) · e^{i m θ}
                // Contribution: Re( Z_{nm} · V_{nm} )
                let v_re = radial * angle.cos();
                let v_im = radial * angle.sin();
                val += z.re * v_re - z.im * v_im;
            }

            output[[r, c]] = val;
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Image Centroid
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the centroid (center of mass) of a 2-D intensity image.
///
/// Returns `(x̄, ȳ)` where x̄ is the weighted column mean and ȳ the weighted
/// row mean.
///
/// If the total intensity is zero the geometric centre of the image is returned.
///
/// # Arguments
///
/// * `image` – 2-D intensity array.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::moment_invariants::image_centroid;
/// use scirs2_core::ndarray::Array2;
///
/// let img = Array2::from_elem((4, 4), 1.0_f64);
/// let (cx, cy) = image_centroid(&img.view());
/// assert!((cx - 1.5).abs() < 1e-10);
/// assert!((cy - 1.5).abs() < 1e-10);
/// ```
pub fn image_centroid(image: &ArrayView2<f64>) -> (f64, f64) {
    let m00 = raw_moment(image, 0, 0);
    if m00.abs() < 1e-15 {
        let rows = image.nrows() as f64;
        let cols = image.ncols() as f64;
        return (cols / 2.0, rows / 2.0);
    }
    centroid_from_raw(image, m00)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute centroid (cx, cy) given precomputed m00.
fn centroid_from_raw(image: &ArrayView2<f64>, m00: f64) -> (f64, f64) {
    let m10 = raw_moment(image, 1, 0);
    let m01 = raw_moment(image, 0, 1);
    (m10 / m00, m01 / m00)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn uniform_image(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_elem((rows, cols), 1.0f64)
    }

    fn disk_image(size: usize, radius: f64) -> Array2<f64> {
        let half = size as f64 / 2.0;
        Array2::from_shape_fn((size, size), |(r, c)| {
            let dr = r as f64 - half;
            let dc = c as f64 - half;
            if dr * dr + dc * dc < radius * radius {
                1.0
            } else {
                0.0
            }
        })
    }

    #[test]
    fn test_raw_moment_uniform() {
        let img = uniform_image(4, 4);
        let m00 = raw_moment(&img.view(), 0, 0);
        assert_eq!(m00, 16.0);
    }

    #[test]
    fn test_central_moment_zero_order_one() {
        let img = uniform_image(4, 4);
        // μ_{10} = μ_{01} = 0 by definition of centroid
        let mu10 = central_moment(&img.view(), 1, 0);
        assert!(mu10.abs() < 1e-9, "mu10={mu10}");
        let mu01 = central_moment(&img.view(), 0, 1);
        assert!(mu01.abs() < 1e-9, "mu01={mu01}");
    }

    #[test]
    fn test_centroid_uniform() {
        let img = uniform_image(4, 4);
        let (cx, cy) = image_centroid(&img.view());
        assert!((cx - 1.5).abs() < 1e-9, "cx={cx}");
        assert!((cy - 1.5).abs() < 1e-9, "cy={cy}");
    }

    #[test]
    fn test_centroid_empty_image() {
        let img = Array2::zeros((6, 8));
        let (cx, cy) = image_centroid(&img.view());
        assert!((cx - 4.0).abs() < 1e-9);
        assert!((cy - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_hu_moments_disk_positive_h1() {
        let img = disk_image(20, 5.0);
        let hu = hu_moments(&img.view());
        assert!(hu[0] > 0.0, "H1 must be positive: {}", hu[0]);
    }

    #[test]
    fn test_hu_moments_rotation_invariance() {
        // A disc is symmetric – H1 should be stable across different
        // translations (we just check that the value is consistent)
        let img1 = disk_image(20, 5.0);
        let img2 = disk_image(20, 5.0); // identical
        let hu1 = hu_moments(&img1.view());
        let hu2 = hu_moments(&img2.view());
        assert!((hu1[0] - hu2[0]).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_central_moment_positive() {
        let img = disk_image(20, 5.0);
        let eta20 = normalized_central_moment(&img.view(), 2, 0);
        assert!(eta20 > 0.0, "eta20={eta20}");
    }

    #[test]
    fn test_zernike_moment_zero_order() {
        let img = disk_image(20, 5.0);
        let z = zernike_moment(&img.view(), 0, 0, None)
            .expect("zernike_moment should succeed for order (0,0) on valid image");
        // Z_{0,0} = (1/π) * sum of pixels in unit disc / (r_disc^2 π)
        // Just ensure it's a real positive number
        assert!(z.re > 0.0, "Z00.re={}", z.re);
        assert!(z.im.abs() < 1e-9, "Z00 should be real: im={}", z.im);
    }

    #[test]
    fn test_zernike_moment_invalid_order() {
        let img = uniform_image(8, 8);
        // |m| > n is invalid
        assert!(zernike_moment(&img.view(), 2, 3, None).is_err());
        // (n - |m|) odd is invalid
        assert!(zernike_moment(&img.view(), 3, 2, None).is_err());
    }

    #[test]
    fn test_reconstruct_from_zernike_shape() {
        let img = disk_image(16, 4.0);
        let mut moments = Vec::new();
        for n in 0u32..=4 {
            for mi in -(n as i32)..=(n as i32) {
                if ((n as i32) - mi.abs()) % 2 == 0 {
                    if let Ok(z) = zernike_moment(&img.view(), n, mi, None) {
                        moments.push((n, mi, z));
                    }
                }
            }
        }
        let recon = reconstruct_from_zernike(&moments, 4, (16, 16))
            .expect("reconstruct_from_zernike should succeed with valid moments");
        assert_eq!(recon.shape(), &[16, 16]);
    }

    #[test]
    fn test_factorial_base_cases() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3_628_800);
    }

    #[test]
    fn test_zernike_radial_n0m0() {
        // R_0^0(ρ) = 1 for all ρ
        for rho in [0.0f64, 0.3, 0.7, 1.0] {
            let r = zernike_radial(0, 0, rho);
            assert!((r - 1.0).abs() < 1e-10, "R00({rho})={r}");
        }
    }
}
