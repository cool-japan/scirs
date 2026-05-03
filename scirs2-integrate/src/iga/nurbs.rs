//! NURBS (Non-Uniform Rational B-Spline) curves and surfaces.
//!
//! NURBS generalize B-splines by assigning a positive weight w_i to each
//! control point. The rational basis functions are:
//!
//! ```text
//! R_{i,p}(t) = N_{i,p}(t) w_i / W(t),   W(t) = Σ_j N_{j,p}(t) w_j
//! ```
//!
//! NURBS can exactly represent conic sections (circles, ellipses) in addition
//! to polynomial curves.
//!
//! ## References
//!
//! - Piegl & Tiller (1997), "The NURBS Book", Chapter 4

use super::bspline::BSplineBasis;
use crate::error::{IntegrateError, IntegrateResult};

// ---------------------------------------------------------------------------
// NurbsCurve
// ---------------------------------------------------------------------------

/// NURBS curve in 2-D.
///
/// C(t) = Σ_i N_{i,p}(t) w_i P_i / Σ_i N_{i,p}(t) w_i
#[derive(Debug, Clone)]
pub struct NurbsCurve {
    /// Underlying B-spline basis.
    pub basis: BSplineBasis,
    /// Control points in 2-D homogeneous coordinates [x*w, y*w, w].
    control_points_hw: Vec<[f64; 3]>,
    /// Weights (stored separately for convenience).
    pub weights: Vec<f64>,
}

impl NurbsCurve {
    /// Create a NURBS curve from control points and weights.
    ///
    /// # Arguments
    ///
    /// * `degree` — Polynomial degree p.
    /// * `knots` — Knot vector.
    /// * `control_points` — Euclidean 2-D control points.
    /// * `weights` — Positive weights for each control point.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        control_points: Vec<[f64; 2]>,
        weights: Vec<f64>,
    ) -> IntegrateResult<Self> {
        let basis = BSplineBasis::new(degree, knots)?;
        if control_points.len() != basis.n_basis {
            return Err(IntegrateError::DimensionMismatch(format!(
                "control_points.len()={} != basis.n_basis={}",
                control_points.len(),
                basis.n_basis
            )));
        }
        if weights.len() != control_points.len() {
            return Err(IntegrateError::DimensionMismatch(
                "weights.len() != control_points.len()".to_string(),
            ));
        }
        for (i, &w) in weights.iter().enumerate() {
            if w <= 0.0 {
                return Err(IntegrateError::InvalidInput(format!(
                    "Weight {w} at index {i} must be positive"
                )));
            }
        }

        // Pre-compute homogeneous coordinates
        let control_points_hw: Vec<[f64; 3]> = control_points
            .iter()
            .zip(weights.iter())
            .map(|(p, &w)| [p[0] * w, p[1] * w, w])
            .collect();

        Ok(Self {
            basis,
            control_points_hw,
            weights,
        })
    }

    /// Create a circular arc NURBS curve.
    ///
    /// The full circle of radius `r` centred at `centre` using the standard
    /// 9-control-point quadratic NURBS representation.
    pub fn circle(centre: [f64; 2], radius: f64) -> IntegrateResult<Self> {
        use std::f64::consts::{FRAC_1_SQRT_2, PI};
        let r = radius;
        let cx = centre[0];
        let cy = centre[1];

        // Standard circle NURBS: degree 2, 9 control points, 12-element knot vector
        // Control points at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°, 360°
        // (angles: every 45°, alternating on- and off-circle).
        let sq = FRAC_1_SQRT_2; // 1/√2
        let control_points = vec![
            [cx + r, cy],
            [cx + r, cy + r],
            [cx, cy + r],
            [cx - r, cy + r],
            [cx - r, cy],
            [cx - r, cy - r],
            [cx, cy - r],
            [cx + r, cy - r],
            [cx + r, cy],
        ];
        let weights = vec![1.0, sq, 1.0, sq, 1.0, sq, 1.0, sq, 1.0];
        let knots = vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ];

        let _ = PI; // suppress unused warning
        Self::new(2, knots, control_points, weights)
    }

    /// Evaluate the NURBS curve at parameter t.
    pub fn eval(&self, t: f64) -> [f64; 2] {
        let (span, n_vals) = self.basis.eval_basis_functions(t);
        let p = self.basis.degree;
        let start = span.saturating_sub(p);

        let mut hw = [0.0_f64; 3];
        for (k, &n_k) in n_vals.iter().enumerate() {
            let idx = start + k;
            if idx < self.control_points_hw.len() {
                let cp = self.control_points_hw[idx];
                hw[0] += n_k * cp[0];
                hw[1] += n_k * cp[1];
                hw[2] += n_k * cp[2];
            }
        }

        if hw[2].abs() < 1e-300 {
            [0.0, 0.0]
        } else {
            [hw[0] / hw[2], hw[1] / hw[2]]
        }
    }

    /// Evaluate the first derivative C'(t) using the quotient rule.
    pub fn eval_deriv(&self, t: f64) -> [f64; 2] {
        let (span, n_vals) = self.basis.eval_basis_functions(t);
        let (_, dn_vals) = self.basis.eval_basis_derivatives(t);
        let p = self.basis.degree;
        let start = span.saturating_sub(p);

        // A(t) = Σ N_i * w_i * P_i,  W(t) = Σ N_i * w_i
        let mut a = [0.0_f64; 2];
        let mut w = 0.0_f64;
        let mut da = [0.0_f64; 2];
        let mut dw = 0.0_f64;

        for k in 0..n_vals.len() {
            let idx = start + k;
            if idx >= self.control_points_hw.len() {
                continue;
            }
            let cp = self.control_points_hw[idx];
            let n_k = n_vals[k];
            let dn_k = dn_vals.get(k).copied().unwrap_or(0.0);

            a[0] += n_k * cp[0];
            a[1] += n_k * cp[1];
            w += n_k * cp[2];

            da[0] += dn_k * cp[0];
            da[1] += dn_k * cp[1];
            dw += dn_k * cp[2];
        }

        if w.abs() < 1e-300 {
            [0.0, 0.0]
        } else {
            // C'(t) = (A'(t) W(t) - A(t) W'(t)) / W(t)²
            [
                (da[0] * w - a[0] * dw) / (w * w),
                (da[1] * w - a[1] * dw) / (w * w),
            ]
        }
    }

    /// Return the parameter domain.
    pub fn domain(&self) -> (f64, f64) {
        self.basis.domain()
    }

    /// Compute arc length by adaptive quadrature with n_samples points.
    pub fn arc_length(&self, n_samples: usize) -> f64 {
        let (t0, t1) = self.domain();
        let dt = (t1 - t0) / n_samples as f64;
        let mut length = 0.0_f64;
        let mut prev = self.eval(t0);
        for i in 1..=n_samples {
            let t = t0 + i as f64 * dt;
            let curr = self.eval(t.min(t1));
            let dx = curr[0] - prev[0];
            let dy = curr[1] - prev[1];
            length += (dx * dx + dy * dy).sqrt();
            prev = curr;
        }
        length
    }
}

// ---------------------------------------------------------------------------
// NurbsSurface
// ---------------------------------------------------------------------------

/// NURBS surface in 3-D (tensor product).
///
/// S(u,v) = Σ_i Σ_j R_{i,j}(u,v) P_{ij}
/// where R_{i,j} = N_{i,p}(u) N_{j,q}(v) w_{ij} / W(u,v)
#[derive(Debug, Clone)]
pub struct NurbsSurface {
    /// B-spline basis in u direction.
    pub basis_u: BSplineBasis,
    /// B-spline basis in v direction.
    pub basis_v: BSplineBasis,
    /// Homogeneous control points [x*w, y*w, z*w, w] indexed [i][j].
    control_points_hw: Vec<Vec<[f64; 4]>>,
    /// Weights w_{ij} (stored separately for convenience), indexed [i][j].
    pub weights: Vec<Vec<f64>>,
}

impl NurbsSurface {
    /// Create a NURBS surface.
    ///
    /// # Arguments
    ///
    /// * `degree_u`, `degree_v` — Polynomial degrees.
    /// * `knots_u`, `knots_v` — Knot vectors.
    /// * `control_points` — 3-D Euclidean control points [n_u][n_v].
    /// * `weights` — Positive weights [n_u][n_v].
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        control_points: Vec<Vec<[f64; 3]>>,
        weights: Vec<Vec<f64>>,
    ) -> IntegrateResult<Self> {
        let basis_u = BSplineBasis::new(degree_u, knots_u)?;
        let basis_v = BSplineBasis::new(degree_v, knots_v)?;

        if control_points.len() != basis_u.n_basis {
            return Err(IntegrateError::DimensionMismatch(format!(
                "control_points rows {} != basis_u.n_basis {}",
                control_points.len(),
                basis_u.n_basis
            )));
        }

        let mut control_points_hw = Vec::with_capacity(control_points.len());
        for (i, (cp_row, w_row)) in control_points.iter().zip(weights.iter()).enumerate() {
            if cp_row.len() != basis_v.n_basis {
                return Err(IntegrateError::DimensionMismatch(format!(
                    "control_points[{i}] len {} != basis_v.n_basis {}",
                    cp_row.len(),
                    basis_v.n_basis
                )));
            }
            if w_row.len() != cp_row.len() {
                return Err(IntegrateError::DimensionMismatch(format!(
                    "weights[{i}] len {} != control_points[{i}] len {}",
                    w_row.len(),
                    cp_row.len()
                )));
            }
            let hw_row: Vec<[f64; 4]> = cp_row
                .iter()
                .zip(w_row.iter())
                .map(|(p, &w)| [p[0] * w, p[1] * w, p[2] * w, w])
                .collect();
            control_points_hw.push(hw_row);
        }

        Ok(Self {
            basis_u,
            basis_v,
            control_points_hw,
            weights,
        })
    }

    /// Create a toroidal NURBS surface.
    ///
    /// Major radius R (from origin to tube centre), minor radius r (tube radius).
    pub fn torus(major_radius: f64, minor_radius: f64) -> IntegrateResult<Self> {
        // Build using 9×9 quadratic NURBS representation.
        // This is a standard construction; we use a simplified 4×4 bilinear
        // tensor product approximation here for illustration.
        // For exactness, one would use degree-2 NURBS in both directions.
        let r_maj = major_radius;
        let r_min = minor_radius;
        let sq = 1.0_f64 / 2.0_f64.sqrt();

        // Use 3×3 quadratic NURBS: 3 control points in u (covers half-circle),
        // and the same in v. For simplicity use a patch.
        // Degree 2, 3 CPs per direction.
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // v-direction: one quarter of the tube cross-section
        let cp = vec![
            vec![
                [r_maj + r_min, 0.0, 0.0],
                [r_maj + r_min, 0.0, r_min],
                [r_maj, 0.0, r_min],
            ],
            vec![
                [(r_maj + r_min) * sq, (r_maj + r_min) * sq, 0.0],
                [(r_maj + r_min) * sq, (r_maj + r_min) * sq, r_min * sq],
                [r_maj * sq, r_maj * sq, r_min],
            ],
            vec![
                [0.0, r_maj + r_min, 0.0],
                [0.0, r_maj + r_min, r_min],
                [0.0, r_maj, r_min],
            ],
        ];
        let w = vec![
            vec![1.0, sq, 1.0],
            vec![sq, sq * sq, sq],
            vec![1.0, sq, 1.0],
        ];

        Self::new(2, 2, knots.clone(), knots, cp, w)
    }

    /// Evaluate the NURBS surface at (u, v).
    pub fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        let (span_u, n_u) = self.basis_u.eval_basis_functions(u);
        let (span_v, n_v) = self.basis_v.eval_basis_functions(v);
        let pu = self.basis_u.degree;
        let pv = self.basis_v.degree;
        let start_u = span_u.saturating_sub(pu);
        let start_v = span_v.saturating_sub(pv);

        let mut hw = [0.0_f64; 4];
        for (ki, &n_ui) in n_u.iter().enumerate() {
            let i = start_u + ki;
            if i >= self.control_points_hw.len() {
                continue;
            }
            for (kj, &n_vj) in n_v.iter().enumerate() {
                let j = start_v + kj;
                if j >= self.control_points_hw[i].len() {
                    continue;
                }
                let cp = self.control_points_hw[i][j];
                let scale = n_ui * n_vj;
                hw[0] += scale * cp[0];
                hw[1] += scale * cp[1];
                hw[2] += scale * cp[2];
                hw[3] += scale * cp[3];
            }
        }

        if hw[3].abs() < 1e-300 {
            [0.0, 0.0, 0.0]
        } else {
            [hw[0] / hw[3], hw[1] / hw[3], hw[2] / hw[3]]
        }
    }

    /// Evaluate the surface normal at (u, v) using finite differences.
    pub fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        let eps = 1e-6;
        let (t0u, t1u) = self.basis_u.domain();
        let (t0v, t1v) = self.basis_v.domain();

        let du = (t1u - t0u) * eps;
        let dv = (t1v - t0v) * eps;

        let p1 = self.eval((u + du).min(t1u), v);
        let p0 = self.eval((u - du).max(t0u), v);
        let q1 = self.eval(u, (v + dv).min(t1v));
        let q0 = self.eval(u, (v - dv).max(t0v));

        let su = [
            (p1[0] - p0[0]) / (2.0 * du),
            (p1[1] - p0[1]) / (2.0 * du),
            (p1[2] - p0[2]) / (2.0 * du),
        ];
        let sv = [
            (q1[0] - q0[0]) / (2.0 * dv),
            (q1[1] - q0[1]) / (2.0 * dv),
            (q1[2] - q0[2]) / (2.0 * dv),
        ];

        [
            su[1] * sv[2] - su[2] * sv[1],
            su[2] * sv[0] - su[0] * sv[2],
            su[0] * sv[1] - su[1] * sv[0],
        ]
    }

    /// Return the parameter domains ([u_min, u_max], [v_min, v_max]).
    pub fn domain(&self) -> ((f64, f64), (f64, f64)) {
        (self.basis_u.domain(), self.basis_v.domain())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_nurbs_circle_radius() {
        // NURBS circle should have radius 1 at all evaluated points.
        let circle = NurbsCurve::circle([0.0, 0.0], 1.0).expect("circle creation");
        let (t0, t1) = circle.domain();
        for k in 0..20 {
            let t = t0 + (t1 - t0) * k as f64 / 20.0 * 0.9999;
            let p = circle.eval(t);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-6,
                "Circle radius at t={t:.4}: r={r:.8} (expected 1.0)"
            );
        }
    }

    #[test]
    fn test_nurbs_circle_circumference() {
        // NURBS circle of radius 1 should have circumference ≈ 2π.
        let circle = NurbsCurve::circle([0.0, 0.0], 1.0).expect("circle");
        let length = circle.arc_length(200);
        let expected = 2.0 * PI;
        assert!(
            (length - expected).abs() / expected < 0.01,
            "Circumference {length:.6} != 2π={expected:.6}"
        );
    }

    #[test]
    fn test_nurbs_surface_eval_finite() {
        // Basic check that the surface evaluates to finite values.
        let torus = NurbsSurface::torus(2.0, 0.5).expect("torus");
        let ((u0, u1), (v0, v1)) = torus.domain();
        for i in 0..5 {
            for j in 0..5 {
                let u = u0 + (u1 - u0) * i as f64 / 4.0 * 0.999;
                let v = v0 + (v1 - v0) * j as f64 / 4.0 * 0.999;
                let p = torus.eval(u, v);
                for &coord in &p {
                    assert!(coord.is_finite(), "Surface coord not finite at ({u},{v})");
                }
            }
        }
    }
}
