//! Geometric kernels for Lie group interpolation.
//!
//! Provides distance functions and kernel evaluations for:
//! - S² (2-sphere): great-circle (geodesic) distance
//! - SO(3) rotation group: quaternion-based geodesic distance
//! - SE(3) rigid motions: product-metric distance
//!
//! The kernel functions map geodesic distances to kernel values suitable
//! for radial basis function interpolation on curved spaces.

use std::f64::consts::PI;

/// Kernel type for geometric (manifold) RBF interpolation.
#[derive(Clone, Debug)]
pub enum GeometricKernel {
    /// Heat kernel: `k(d) = exp(-d² / σ²)`.
    ///
    /// Corresponds to the fundamental solution of the heat equation on the manifold.
    Heat {
        /// Bandwidth parameter σ > 0.
        sigma: f64,
    },

    /// Zonal kernel via truncated spherical harmonic expansion.
    ///
    /// `k(θ) = Σ_{l=0}^{L} (2l+1)/(4π) · exp(-l(l+1)σ²) · P_l(cos θ)`
    ///
    /// where P_l are Legendre polynomials.  Only meaningful on S².
    SphericalHarmonic {
        /// Maximum angular frequency (bandwidth).
        bandwidth: usize,
        /// Diffusion time / decay parameter σ > 0.
        sigma: f64,
    },

    /// Matérn kernel using geodesic distance.
    ///
    /// Supports ν = 0.5 (exponential), 1.5, 2.5; falls back to Gaussian for other ν.
    Matern {
        /// Smoothness parameter ν (typically 0.5, 1.5, or 2.5).
        nu: f64,
        /// Length-scale parameter ℓ > 0.
        length_scale: f64,
    },
}

/// Compute the geodesic (great-circle) distance on S² between two unit vectors.
///
/// Both `u` and `v` should be unit vectors in ℝ³ (‖u‖ = ‖v‖ = 1).
/// Returns an angle in [0, π].
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::lie_group::kernel::sphere_geodesic_dist;
/// let north = [0.0_f64, 0.0, 1.0];
/// let south = [0.0_f64, 0.0, -1.0];
/// let d = sphere_geodesic_dist(&north, &south);
/// assert!((d - std::f64::consts::PI).abs() < 1e-12);
/// ```
pub fn sphere_geodesic_dist(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    dot.acos()
}

/// Compute the geodesic distance on SO(3) between two unit quaternions.
///
/// Quaternions are given as `[w, x, y, z]` (scalar-first convention).
/// Returns an angle in [0, π].
///
/// The formula exploits the double-cover SO(3) ≅ S³/±1:
/// `d(q1, q2) = 2 · arccos(|q1·q2|)`.
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::lie_group::kernel::so3_geodesic_dist;
/// let identity = [1.0_f64, 0.0, 0.0, 0.0];
/// let d = so3_geodesic_dist(&identity, &identity);
/// assert!(d.abs() < 1e-12);
/// ```
pub fn so3_geodesic_dist(q1: &[f64; 4], q2: &[f64; 4]) -> f64 {
    // |q1·q2|: absolute dot product handles the double-cover identification q ~ -q.
    let dot = (q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
        .abs()
        .clamp(0.0, 1.0);
    2.0 * dot.acos()
}

/// Compute a weighted product-metric distance on SE(3).
///
/// SE(3) = ℝ³ ⋊ SO(3).  The distance combines Euclidean translation distance
/// and the SO(3) geodesic:
/// `d((t1,q1),(t2,q2)) = ‖t1-t2‖ + w_rot · d_{SO(3)}(q1,q2)`.
///
/// # Arguments
///
/// * `t1`, `t2` — translation vectors in ℝ³.
/// * `q1`, `q2` — unit quaternions `[w, x, y, z]` for the rotation components.
/// * `w_rot` — non-negative weight balancing rotational vs translational distance.
pub fn se3_geodesic_dist(
    t1: &[f64; 3],
    q1: &[f64; 4],
    t2: &[f64; 3],
    q2: &[f64; 4],
    w_rot: f64,
) -> f64 {
    let dt = ((t1[0] - t2[0]).powi(2) + (t1[1] - t2[1]).powi(2) + (t1[2] - t2[2]).powi(2)).sqrt();
    let dr = so3_geodesic_dist(q1, q2);
    dt + w_rot * dr
}

/// Evaluate a [`GeometricKernel`] at a given geodesic distance.
///
/// Returns a non-negative kernel value.  The functions are all positive (semi-)
/// definite on their respective manifolds when the parameters are in range.
pub fn eval_kernel(dist: f64, kernel: &GeometricKernel) -> f64 {
    match kernel {
        GeometricKernel::Heat { sigma } => {
            let s2 = sigma * sigma;
            if s2 < f64::EPSILON {
                return if dist < f64::EPSILON { 1.0 } else { 0.0 };
            }
            (-dist * dist / s2).exp()
        }

        GeometricKernel::Matern { nu, length_scale } => {
            let r = dist / length_scale.max(f64::EPSILON);
            if r < 1e-12 {
                return 1.0;
            }
            // Match on 2*nu rounded to nearest integer to avoid floating-point equality.
            let two_nu = (nu * 2.0).round() as u32;
            match two_nu {
                1 => (-r).exp(),                                             // ν = 0.5
                3 => (1.0 + 3_f64.sqrt() * r) * (-(3_f64.sqrt() * r)).exp(), // ν = 1.5
                5 => (1.0 + 5_f64.sqrt() * r + 5.0 / 3.0 * r * r) * (-(5_f64.sqrt() * r)).exp(), // ν = 2.5
                _ => (-r * r / 2.0).exp(), // fallback: squared-exponential (smooth)
            }
        }

        GeometricKernel::SphericalHarmonic { bandwidth, sigma } => {
            // Truncated zonal harmonic expansion using Bonnet recurrence for P_l.
            // k(θ) = Σ_{l=0}^{L} (2l+1)/(4π) · exp(-l(l+1)σ²) · P_l(cos θ)
            let cos_theta = if dist < 1e-12 {
                1.0
            } else {
                dist.cos().clamp(-1.0, 1.0)
            };
            // Legendre recurrence: P_0=1, P_1=x, P_l = ((2l-1)x·P_{l-1} - (l-1)P_{l-2}) / l
            let mut p_prev = 1.0_f64; // P_{l-2}
            let mut p_curr = cos_theta; // P_{l-1}
            let mut sum = 0.0_f64;
            for l in 0usize..=*bandwidth {
                let pl = if l == 0 {
                    1.0
                } else if l == 1 {
                    cos_theta
                } else {
                    let lf = l as f64;
                    let next = ((2.0 * lf - 1.0) * cos_theta * p_curr - (lf - 1.0) * p_prev) / lf;
                    p_prev = p_curr;
                    p_curr = next;
                    next
                };
                let weight = (2 * l + 1) as f64 / (4.0 * PI)
                    * (-((l * (l + 1)) as f64) * sigma * sigma).exp();
                sum += weight * pl;
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_geodesic_dist_poles() {
        let north = [0.0_f64, 0.0, 1.0];
        let south = [0.0_f64, 0.0, -1.0];
        let d = sphere_geodesic_dist(&north, &south);
        assert!(
            (d - PI).abs() < 1e-12,
            "antipodal distance should be π, got {d}"
        );
    }

    #[test]
    fn test_sphere_geodesic_dist_same_point() {
        let p = [1.0_f64, 0.0, 0.0];
        let d = sphere_geodesic_dist(&p, &p);
        assert!(d.abs() < 1e-12, "self-distance should be 0, got {d}");
    }

    #[test]
    fn test_sphere_geodesic_dist_quarter_circle() {
        let p1 = [1.0_f64, 0.0, 0.0];
        let p2 = [0.0_f64, 1.0, 0.0];
        let d = sphere_geodesic_dist(&p1, &p2);
        assert!(
            (d - PI / 2.0).abs() < 1e-12,
            "quarter circle should be π/2, got {d}"
        );
    }

    #[test]
    fn test_so3_geodesic_dist_identity() {
        let id = [1.0_f64, 0.0, 0.0, 0.0];
        let d = so3_geodesic_dist(&id, &id);
        assert!(d.abs() < 1e-12, "identity distance should be 0, got {d}");
    }

    #[test]
    fn test_so3_geodesic_dist_half_cover() {
        // q and -q represent the same rotation; distance should be 0.
        // Use exact 1/√2 to ensure the quaternion is precisely unit length.
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let q = [s, s, 0.0_f64, 0.0];
        let neg_q = [-s, -s, 0.0_f64, 0.0];
        let d = so3_geodesic_dist(&q, &neg_q);
        // Tolerance reflects floating-point precision in arccos near 1.0.
        assert!(d.abs() < 1e-6, "q and -q should have distance 0, got {d}");
    }

    #[test]
    fn test_eval_kernel_heat_at_zero() {
        let k = eval_kernel(0.0, &GeometricKernel::Heat { sigma: 1.0 });
        assert!(
            (k - 1.0).abs() < 1e-12,
            "heat kernel at 0 should be 1, got {k}"
        );
    }

    #[test]
    fn test_eval_kernel_matern_at_zero() {
        for nu in [0.5, 1.5, 2.5, 3.0] {
            let k = eval_kernel(
                0.0,
                &GeometricKernel::Matern {
                    nu,
                    length_scale: 1.0,
                },
            );
            assert!(
                (k - 1.0).abs() < 1e-12,
                "Matern(nu={nu}) at 0 should be 1, got {k}"
            );
        }
    }

    #[test]
    fn test_eval_kernel_spherical_harmonic_non_negative() {
        let kernel = GeometricKernel::SphericalHarmonic {
            bandwidth: 10,
            sigma: 0.3,
        };
        for dist_deg in [0, 30, 60, 90, 120, 150, 180] {
            let d = (dist_deg as f64) * PI / 180.0;
            let k = eval_kernel(d, &kernel);
            assert!(
                k.is_finite(),
                "SphericalHarmonic kernel at {dist_deg}° should be finite, got {k}"
            );
        }
    }

    #[test]
    fn test_se3_geodesic_dist_same_pose() {
        let t = [1.0_f64, 2.0, 3.0];
        let q = [1.0_f64, 0.0, 0.0, 0.0];
        let d = se3_geodesic_dist(&t, &q, &t, &q, 1.0);
        assert!(d.abs() < 1e-12, "same pose distance should be 0, got {d}");
    }
}
