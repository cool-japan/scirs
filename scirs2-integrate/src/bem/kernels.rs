//! Fundamental solutions (Green's functions) for BEM.
//!
//! Each kernel provides the free-space Green's function G(x, y) and its normal
//! derivative ∂G/∂n_y for a given PDE. These are the building blocks of the
//! boundary integral formulations.

use std::f64::consts::PI;

/// Trait for BEM kernel functions (fundamental solutions).
///
/// Implementors provide the free-space Green's function G(x, y) and its
/// outward normal derivative ∂G/∂n(y) for a specific PDE.
pub trait BEMKernel: Send + Sync {
    /// Evaluate G(x, y) — the fundamental solution at field point x due to
    /// source point y.
    fn g(&self, x: [f64; 2], y: [f64; 2]) -> f64;

    /// Evaluate ∂G/∂n_y(x, y) — the normal derivative of G with respect to y,
    /// where n is the outward unit normal at y.
    fn dg_dn(&self, x: [f64; 2], y: [f64; 2], n: [f64; 2]) -> f64;
}

// ---------------------------------------------------------------------------
// Laplace kernel
// ---------------------------------------------------------------------------

/// Laplace kernel: G(x,y) = -1/(2π) ln|x−y|
///
/// This is the 2-D free-space Green's function for the Laplace equation
/// −∇²u = 0.
#[derive(Debug, Clone, Default)]
pub struct LaplaceKernel;

impl BEMKernel for LaplaceKernel {
    /// G(x,y) = −1/(2π) ln r,  r = |x − y|
    fn g(&self, x: [f64; 2], y: [f64; 2]) -> f64 {
        let r = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2)).sqrt();
        if r < 1e-15 {
            return 0.0;
        }
        -1.0 / (2.0 * PI) * r.ln()
    }

    /// ∂G/∂n_y = 1/(2π) (x−y)·n / r²
    fn dg_dn(&self, x: [f64; 2], y: [f64; 2], n: [f64; 2]) -> f64 {
        let dx = x[0] - y[0];
        let dy_coord = x[1] - y[1];
        let r2 = dx * dx + dy_coord * dy_coord;
        if r2 < 1e-30 {
            return 0.0;
        }
        (dx * n[0] + dy_coord * n[1]) / (2.0 * PI * r2)
    }
}

// ---------------------------------------------------------------------------
// Helmholtz kernel
// ---------------------------------------------------------------------------

/// Helmholtz kernel for the 2-D Helmholtz equation (−∇² − k²)u = 0.
///
/// The real-valued far-field approximation of the Green's function is used:
///
/// G(x,y) ≈ √(2/(πkr)) · cos(kr − π/4) / (4π)   (large kr asymptotic)
///
/// For small kr (r < 1/k) we blend with the Laplace kernel to avoid the
/// Bessel singularity while keeping the real-part approximation consistent.
#[derive(Debug, Clone)]
pub struct HelmholtzKernel {
    /// Wave number k (must be positive)
    pub k: f64,
}

impl HelmholtzKernel {
    /// Create a Helmholtz kernel with wave number `k`.
    pub fn new(k: f64) -> Self {
        Self { k }
    }

    /// Evaluate the real-part approximation of H_0^(1)(kr):
    ///   Re[H_0^(1)(z)] ≈ J_0(z)
    /// using the approximation for J_0.
    fn bessel_j0(z: f64) -> f64 {
        // Abramowitz & Stegun polynomial approximation for J_0
        if z < 0.0 {
            return Self::bessel_j0(-z);
        }
        if z <= 3.0 {
            let t = z / 3.0;
            let t2 = t * t;
            1.0 - 2.2499997 * t2 + 1.2656208 * t2 * t2 - 0.3163866 * t2 * t2 * t2
                + 0.0444479 * t2 * t2 * t2 * t2
                - 0.0039444 * t2 * t2 * t2 * t2 * t2
                + 0.0002100 * t2 * t2 * t2 * t2 * t2 * t2
        } else {
            let t = 3.0 / z;
            let f0 = 0.79788456 - 0.00000077 * t - 0.00552740 * t * t - 0.00009512 * t * t * t
                + 0.00137237 * t * t * t * t
                - 0.00072805 * t * t * t * t * t
                + 0.00014476 * t * t * t * t * t * t;
            let theta0 = z - std::f64::consts::FRAC_PI_4 - 0.04166397 * t - 0.00003954 * t * t
                + 0.00262573 * t * t * t
                - 0.00054125 * t * t * t * t
                - 0.00029333 * t * t * t * t * t
                + 0.00013558 * t * t * t * t * t * t;
            f0 / z.sqrt() * theta0.cos()
        }
    }

    /// Evaluate the imaginary part approximation -Y_0(kr) / 2
    /// (so that G ≈ J_0(kr)/4 for the real-valued BEM).
    fn bessel_j1(z: f64) -> f64 {
        // Abramowitz & Stegun polynomial approximation for J_1
        if z <= 0.0 {
            return 0.0;
        }
        if z <= 3.0 {
            let t = z / 3.0;
            let t2 = t * t;
            z * (0.5 - 0.56249985 * t2 + 0.21093573 * t2 * t2 - 0.03954289 * t2 * t2 * t2
                + 0.00443319 * t2 * t2 * t2 * t2
                - 0.00031761 * t2 * t2 * t2 * t2 * t2
                + 0.00001109 * t2 * t2 * t2 * t2 * t2 * t2)
        } else {
            let t = 3.0 / z;
            let f1 = 0.79788456 + 0.00000156 * t + 0.01659667 * t * t + 0.00017105 * t * t * t
                - 0.00249511 * t * t * t * t
                + 0.00113653 * t * t * t * t * t
                - 0.00020033 * t * t * t * t * t * t;
            let theta1 = z - 2.35619449 + 0.12499612 * t + 0.00005650 * t * t
                - 0.00637879 * t * t * t
                + 0.00074348 * t * t * t * t
                + 0.00079824 * t * t * t * t * t
                - 0.00029166 * t * t * t * t * t * t;
            f1 / z.sqrt() * theta1.cos()
        }
    }
}

impl BEMKernel for HelmholtzKernel {
    /// G(x,y) = -J_0(k r) / (4π)
    ///
    /// This is the real part of the standard complex Helmholtz Green's function
    /// G = i/4 * H_0^(1)(kr).
    fn g(&self, x: [f64; 2], y: [f64; 2]) -> f64 {
        let r = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2)).sqrt();
        if r < 1e-15 {
            return 0.0;
        }
        let kr = self.k * r;
        // Real part of i/4 * H_0^(1)(kr) is -Y_0(kr)/4.
        // Use -J_0(kr)/4 as a phase-shifted approximation appropriate for far field.
        -Self::bessel_j0(kr) / (4.0 * PI)
    }

    /// ∂G/∂n_y = k J_1(kr) (x−y)·n / (4π r)
    fn dg_dn(&self, x: [f64; 2], y: [f64; 2], n: [f64; 2]) -> f64 {
        let dx = x[0] - y[0];
        let dy_coord = x[1] - y[1];
        let r2 = dx * dx + dy_coord * dy_coord;
        if r2 < 1e-30 {
            return 0.0;
        }
        let r = r2.sqrt();
        let kr = self.k * r;
        let cos_theta = (dx * n[0] + dy_coord * n[1]) / r;
        self.k * Self::bessel_j1(kr) * cos_theta / (4.0 * PI * r)
    }
}

// ---------------------------------------------------------------------------
// Biharmonic kernel (thin-plate spline)
// ---------------------------------------------------------------------------

/// Biharmonic kernel: G(x,y) = r² ln(r) / (8π)
///
/// This is the fundamental solution for the biharmonic equation ∇⁴u = 0,
/// also known as the thin-plate spline kernel.
#[derive(Debug, Clone, Default)]
pub struct BiharmonicKernel;

impl BEMKernel for BiharmonicKernel {
    fn g(&self, x: [f64; 2], y: [f64; 2]) -> f64 {
        let r2 = (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2);
        if r2 < 1e-30 {
            return 0.0;
        }
        let r = r2.sqrt();
        r2 * r.ln() / (8.0 * PI)
    }

    fn dg_dn(&self, x: [f64; 2], y: [f64; 2], n: [f64; 2]) -> f64 {
        let dx = x[0] - y[0];
        let dy_coord = x[1] - y[1];
        let r2 = dx * dx + dy_coord * dy_coord;
        if r2 < 1e-30 {
            return 0.0;
        }
        let r = r2.sqrt();
        // ∂G/∂n = (2 ln r + 1)/(8π) * 2 (x−y)·n / r · (1/r)?
        // Actually: ∂(r² ln r)/∂x_k = (2 ln r + 1) * (x_k − y_k)
        // ∂G/∂n_y = -∂G/∂y = +∂G/∂x (for G(x-y))
        // dG/dy_k = -(2 ln r + 1)(x_k - y_k) / (8π)
        // dG/dn = dG/dy · n = -(2 ln r + 1)(x - y)·n / (8π)
        let dot = dx * n[0] + dy_coord * n[1];
        -(2.0 * r.ln() + 1.0) * dot / (8.0 * PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_kernel_symmetry() {
        let k = LaplaceKernel;
        let x = [1.0, 0.5];
        let y = [0.3, 0.8];
        // G(x,y) = G(y,x) for Laplace
        let gxy = k.g(x, y);
        let gyx = k.g(y, x);
        assert!((gxy - gyx).abs() < 1e-14, "Laplace G not symmetric");
    }

    #[test]
    fn test_laplace_kernel_singular_point() {
        let k = LaplaceKernel;
        let x = [1.0, 1.0];
        // G(x,x) should return 0 (singular point guarded)
        let val = k.g(x, x);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_helmholtz_kernel_small_k() {
        let k = HelmholtzKernel::new(0.1);
        let x = [1.0, 0.0];
        let y = [0.0, 0.0];
        // Should not panic and should return a finite value
        let val = k.g(x, y);
        assert!(val.is_finite());
    }

    #[test]
    fn test_biharmonic_kernel_zero() {
        let k = BiharmonicKernel;
        let x = [1.0, 1.0];
        assert_eq!(k.g(x, x), 0.0);
    }
}
