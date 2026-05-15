//! B-spline basis functions and geometric objects.
//!
//! Implements the Cox-de Boor recursion for B-spline basis functions and
//! provides curve and surface geometric objects for use in IGA.
//!
//! ## B-spline basis functions
//!
//! Given a degree p and a non-decreasing knot vector
//! T = {t_0, t_1, …, t_{n+p+1}} the B-spline basis functions N_{i,p}(t) are
//! defined by the Cox-de Boor recursion:
//!
//! ```text
//! N_{i,0}(t) = 1  if t_i ≤ t < t_{i+1}, else 0
//! N_{i,p}(t) = (t − t_i)/(t_{i+p} − t_i) N_{i,p−1}(t)
//!            + (t_{i+p+1} − t)/(t_{i+p+1} − t_{i+1}) N_{i+1,p−1}(t)
//! ```
//!
//! ## References
//!
//! - Piegl & Tiller (1997), "The NURBS Book"
//! - Hughes, Cottrell & Bazilevs (2005), "Isogeometric Analysis"

use crate::error::{IntegrateError, IntegrateResult};

// ---------------------------------------------------------------------------
// BSplineBasis
// ---------------------------------------------------------------------------

/// B-spline basis defined by a polynomial degree and knot vector.
///
/// The number of basis functions is `n_basis = n_knots − degree − 1`.
#[derive(Debug, Clone)]
pub struct BSplineBasis {
    /// Polynomial degree p.
    pub degree: usize,
    /// Non-decreasing knot vector T of length m = n + p + 1.
    pub knots: Vec<f64>,
    /// Number of basis functions n = m − p − 1.
    pub n_basis: usize,
}

impl BSplineBasis {
    /// Create a B-spline basis from a degree and knot vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the knot vector is too short for the given degree.
    pub fn new(degree: usize, knots: Vec<f64>) -> IntegrateResult<Self> {
        let m = knots.len();
        if m < degree + 2 {
            return Err(IntegrateError::InvalidInput(format!(
                "Knot vector length {m} too short for degree {degree}: need at least {}",
                degree + 2
            )));
        }
        let n_basis = m - degree - 1;
        Ok(Self {
            degree,
            knots,
            n_basis,
        })
    }

    /// Create a **uniform open** (clamped) B-spline basis.
    ///
    /// An open knot vector has `p+1` repeated knots at both ends and uniformly
    /// spaced interior knots, giving interpolation at the first and last control
    /// points.
    ///
    /// # Arguments
    ///
    /// * `degree` — Polynomial degree p.
    /// * `n_control_points` — Number of control points n.
    ///
    /// The knot vector has length n + p + 1.
    pub fn uniform_open(degree: usize, n_control_points: usize) -> IntegrateResult<Self> {
        let n = n_control_points;
        if n < degree + 1 {
            return Err(IntegrateError::InvalidInput(format!(
                "n_control_points={n} must be ≥ degree+1 = {}",
                degree + 1
            )));
        }
        let m = n + degree + 1;
        let mut knots = vec![0.0_f64; m];

        // First p+1 knots are 0
        for k in 0..=degree {
            knots[k] = 0.0;
        }
        // Last p+1 knots are 1
        for k in (m - degree - 1)..m {
            knots[k] = 1.0;
        }
        // Interior knots uniformly spaced
        let n_interior = m - 2 * (degree + 1);
        for i in 0..n_interior {
            knots[degree + 1 + i] = (i + 1) as f64 / (n_interior + 1) as f64;
        }

        Ok(Self {
            degree,
            knots,
            n_basis: n,
        })
    }

    /// Find the knot span index i such that T\[i\] ≤ t < T\[i+1\].
    ///
    /// Uses binary search. For t at the end of the domain (t = T[m-1]), returns
    /// the index of the last non-empty span.
    pub fn find_span(&self, t: f64) -> usize {
        let p = self.degree;
        let n = self.n_basis - 1; // last basis index
        let m = self.knots.len() - 1;

        // Special case: t at the right end of the domain
        if t >= self.knots[n + 1] {
            return n;
        }
        if t <= self.knots[p] {
            return p;
        }

        // Binary search
        let mut low = p;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while t < self.knots[mid] || t >= self.knots[mid + 1] {
            if t < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
            if mid + 1 >= m {
                break;
            }
        }
        // Clamp to valid span range
        mid.min(n)
    }

    /// Evaluate all non-zero basis functions at parameter t.
    ///
    /// Returns `(span, values)` where `span` is the knot span index and `values`
    /// is a vector of length `degree + 1` with N_{span−p,p}(t), …, N_{span,p}(t).
    ///
    /// Uses the efficient de Boor triangular table algorithm from Piegl & Tiller.
    pub fn eval_basis_functions(&self, t: f64) -> (usize, Vec<f64>) {
        let p = self.degree;
        let i = self.find_span(t);

        let mut n_vals = vec![0.0_f64; p + 1];
        let mut left = vec![0.0_f64; p + 1];
        let mut right = vec![0.0_f64; p + 1];

        n_vals[0] = 1.0;
        for j in 1..=p {
            left[j] = t - self.knots[i + 1 - j];
            right[j] = self.knots[i + j] - t;
            let mut saved = 0.0_f64;
            for r in 0..j {
                let denom = right[r + 1] + left[j - r];
                let temp = if denom.abs() < 1e-300 {
                    0.0
                } else {
                    n_vals[r] / denom
                };
                n_vals[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n_vals[j] = saved;
        }
        (i, n_vals)
    }

    /// Evaluate a single basis function N_{i,p}(t).
    pub fn eval(&self, i: usize, t: f64) -> f64 {
        // Use the de Boor triangular table via a local computation.
        let p = self.degree;
        let (span, vals) = self.eval_basis_functions(t);

        // span - p is the start index of the non-zero basis functions.
        // The non-zero N_{j,p} for j = span-p, ..., span.
        let start = span.saturating_sub(p);
        if i < start || i > span {
            return 0.0;
        }
        let local = i - start;
        if local < vals.len() {
            vals[local]
        } else {
            0.0
        }
    }

    /// Evaluate the first derivative N'_{i,p}(t) of basis function i.
    ///
    /// Uses the recurrence: N'_{i,p}(t) = p * [N_{i,p−1}(t)/(t_{i+p}−t_i) − N_{i+1,p−1}(t)/(t_{i+p+1}−t_{i+1})]
    pub fn eval_deriv(&self, i: usize, t: f64) -> f64 {
        let p = self.degree;
        if p == 0 {
            return 0.0;
        }
        let n_basis = self.n_basis;
        let p_f = p as f64;

        // N'_{i,p} = p * N_{i,p-1} / (t_{i+p} - t_i)  -  p * N_{i+1,p-1} / (t_{i+p+1} - t_{i+1})
        let left_denom = self.knots.get(i + p).copied().unwrap_or(0.0)
            - self.knots.get(i).copied().unwrap_or(0.0);
        let left = if left_denom.abs() < 1e-300 || i >= n_basis {
            0.0
        } else {
            p_f * self.eval_lower(i, p - 1, t) / left_denom
        };

        let right_denom = self.knots.get(i + p + 1).copied().unwrap_or(0.0)
            - self.knots.get(i + 1).copied().unwrap_or(0.0);
        let right = if right_denom.abs() < 1e-300 || i + 1 > n_basis {
            0.0
        } else {
            p_f * self.eval_lower(i + 1, p - 1, t) / right_denom
        };

        left - right
    }

    /// Evaluate the basis function N_{i,q}(t) for lower degree q (helper for derivatives).
    fn eval_lower(&self, i: usize, q: usize, t: f64) -> f64 {
        // Recursively compute using Cox-de Boor
        self.cox_de_boor(i, q, t)
    }

    /// Cox-de Boor recursion for N_{i,p}(t).
    fn cox_de_boor(&self, i: usize, p: usize, t: f64) -> f64 {
        if p == 0 {
            let t_i = self.knots.get(i).copied().unwrap_or(f64::NEG_INFINITY);
            let t_i1 = self.knots.get(i + 1).copied().unwrap_or(f64::NEG_INFINITY);
            if t >= t_i && t < t_i1 {
                1.0
            } else if t == t_i1 && (i + 1) >= self.knots.len() - 1 {
                // Special case: right end of domain
                1.0
            } else {
                0.0
            }
        } else {
            let t_i = self.knots.get(i).copied().unwrap_or(0.0);
            let t_ip = self.knots.get(i + p).copied().unwrap_or(0.0);
            let t_i1 = self.knots.get(i + 1).copied().unwrap_or(0.0);
            let t_ip1 = self.knots.get(i + p + 1).copied().unwrap_or(0.0);

            let left = {
                let denom = t_ip - t_i;
                if denom.abs() < 1e-300 {
                    0.0
                } else {
                    (t - t_i) / denom * self.cox_de_boor(i, p - 1, t)
                }
            };
            let right = {
                let denom = t_ip1 - t_i1;
                if denom.abs() < 1e-300 {
                    0.0
                } else {
                    (t_ip1 - t) / denom * self.cox_de_boor(i + 1, p - 1, t)
                }
            };
            left + right
        }
    }

    /// Evaluate all basis function first derivatives at t.
    ///
    /// Returns `(span, derivs)` where `derivs[k]` = N'_{span-p+k,p}(t).
    pub fn eval_basis_derivatives(&self, t: f64) -> (usize, Vec<f64>) {
        let p = self.degree;
        let (span, _n0) = self.eval_basis_functions(t);

        // Use the formula:
        //   N'_{i,p}(t) = p [ N_{i,p-1}/(t_{i+p}-t_i) − N_{i+1,p-1}/(t_{i+p+1}-t_{i+1}) ]
        // Evaluate all order-(p-1) basis functions in the span.
        let mut dn = vec![0.0_f64; p + 1];
        if p == 0 {
            return (span, dn);
        }

        // Order p-1 basis functions non-zero for j = span-(p-1), ..., span
        let basis_p1 = self.eval_basis_functions_order(t, p - 1);
        // basis_p1[k] = N_{span-(p-1)+k, p-1}(t) for k = 0..p

        for k in 0..=p {
            // Basis index for N_{i,p} is i = span - p + k
            let i = if span >= p { span - p + k } else { k };

            // Left term: N_{i,p-1} / (t_{i+p} - t_i)
            // N_{i,p-1} is non-zero only for i in [span-(p-1), span].
            // When k=0 and span>=p, i = span-p which is below span-(p-1), so N_{i,p-1}=0.
            let left = if k == 0 && span >= p {
                // i = span - p is outside the support of the order-(p-1) basis
                0.0
            } else {
                // local_idx = i - (span - (p-1)) = i + p - 1 - span (when span >= p-1)
                // Use checked subtraction to avoid overflow
                let local_idx_signed = (i as isize) + (p as isize) - 1 - (span as isize);
                let local_idx = if span >= p.saturating_sub(1) && local_idx_signed >= 0 {
                    local_idx_signed as usize
                } else {
                    k
                };
                let n_im1 = if local_idx < basis_p1.len() {
                    basis_p1[local_idx]
                } else {
                    0.0
                };
                let t_ip = self.knots.get(i + p).copied().unwrap_or(0.0);
                let t_i = self.knots.get(i).copied().unwrap_or(0.0);
                let denom = t_ip - t_i;
                if denom.abs() < 1e-300 {
                    0.0
                } else {
                    n_im1 / denom
                }
            };

            // Right term: N_{i+1,p-1} / (t_{i+p+1} - t_{i+1})
            // N_{i+1,p-1} is non-zero only for (i+1) in [span-(p-1), span].
            // When k=p and span>=p, i+1 = span+1 which is above span, so N_{i+1,p-1}=0.
            let right = if k == p {
                0.0
            } else {
                let local_idx_signed = (i as isize) + (p as isize) - (span as isize);
                let local_idx = if span >= p.saturating_sub(1) && local_idx_signed >= 0 {
                    local_idx_signed as usize
                } else {
                    k + 1
                };
                let n_i1p1 = if local_idx < basis_p1.len() {
                    basis_p1[local_idx]
                } else {
                    0.0
                };
                let t_ip1 = self.knots.get(i + p + 1).copied().unwrap_or(0.0);
                let t_i1 = self.knots.get(i + 1).copied().unwrap_or(0.0);
                let denom = t_ip1 - t_i1;
                if denom.abs() < 1e-300 {
                    0.0
                } else {
                    n_i1p1 / denom
                }
            };

            dn[k] = p as f64 * (left - right);
        }
        (span, dn)
    }

    /// Evaluate all non-zero basis functions of order `q` at t.
    fn eval_basis_functions_order(&self, t: f64, q: usize) -> Vec<f64> {
        if q == 0 {
            let i = self.find_span(t);
            let mut vals = vec![0.0_f64; 2];
            // For order 0: N_{i,0} = 1 in span i
            vals[0] = 1.0;
            return vals;
        }
        let i = self.find_span_for_degree(t, q);
        let mut n_vals = vec![0.0_f64; q + 1];
        let mut left = vec![0.0_f64; q + 1];
        let mut right = vec![0.0_f64; q + 1];

        n_vals[0] = 1.0;
        for j in 1..=q {
            left[j] = t - self.knots.get(i + 1 - j).copied().unwrap_or(0.0);
            right[j] = self.knots.get(i + j).copied().unwrap_or(0.0) - t;
            let mut saved = 0.0_f64;
            for r in 0..j {
                let denom = right[r + 1] + left[j - r];
                let temp = if denom.abs() < 1e-300 {
                    0.0
                } else {
                    n_vals[r] / denom
                };
                n_vals[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n_vals[j] = saved;
        }
        n_vals
    }

    /// Find the knot span for parameter t treating degree as `q`.
    fn find_span_for_degree(&self, t: f64, q: usize) -> usize {
        let n = if self.knots.len() > q + 1 {
            self.knots.len() - q - 2
        } else {
            0
        };
        if t >= self.knots.last().copied().unwrap_or(1.0) {
            return n;
        }
        if t <= self.knots.get(q).copied().unwrap_or(0.0) {
            return q;
        }
        let mut low = q;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while t < self.knots[mid] || t >= self.knots[mid + 1] {
            if t < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
            if mid + 1 >= self.knots.len() {
                break;
            }
        }
        mid.min(n)
    }

    /// Return the parameter domain [t_min, t_max].
    pub fn domain(&self) -> (f64, f64) {
        let p = self.degree;
        let t_min = self.knots.get(p).copied().unwrap_or(0.0);
        let t_max = self
            .knots
            .get(self.knots.len() - p - 1)
            .copied()
            .unwrap_or(1.0);
        (t_min, t_max)
    }
}

// ---------------------------------------------------------------------------
// BSplineCurve
// ---------------------------------------------------------------------------

/// B-spline curve in 2-D.
///
/// Maps a parameter t ∈ [t_0, t_m] to a 2-D point via:
/// C(t) = Σ_i N_{i,p}(t) P_i
#[derive(Debug, Clone)]
pub struct BSplineCurve {
    /// B-spline basis.
    pub basis: BSplineBasis,
    /// Control points P_i ∈ ℝ².
    pub control_points: Vec<[f64; 2]>,
}

impl BSplineCurve {
    /// Create a B-spline curve.
    ///
    /// # Errors
    ///
    /// Returns an error if `control_points.len() != basis.n_basis`.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        control_points: Vec<[f64; 2]>,
    ) -> IntegrateResult<Self> {
        let basis = BSplineBasis::new(degree, knots)?;
        if control_points.len() != basis.n_basis {
            return Err(IntegrateError::DimensionMismatch(format!(
                "control_points.len()={} != basis.n_basis={}",
                control_points.len(),
                basis.n_basis
            )));
        }
        Ok(Self {
            basis,
            control_points,
        })
    }

    /// Evaluate the curve at parameter t.
    pub fn eval(&self, t: f64) -> [f64; 2] {
        let (span, n_vals) = self.basis.eval_basis_functions(t);
        let p = self.basis.degree;
        let start = span.saturating_sub(p);

        let mut point = [0.0_f64; 2];
        for (k, &n_k) in n_vals.iter().enumerate() {
            let idx = start + k;
            if idx < self.control_points.len() {
                let cp = self.control_points[idx];
                point[0] += n_k * cp[0];
                point[1] += n_k * cp[1];
            }
        }
        point
    }

    /// Evaluate the first derivative C'(t).
    pub fn eval_deriv(&self, t: f64) -> [f64; 2] {
        let (span, dn_vals) = self.basis.eval_basis_derivatives(t);
        let p = self.basis.degree;
        let start = span.saturating_sub(p);

        let mut deriv = [0.0_f64; 2];
        for (k, &dn_k) in dn_vals.iter().enumerate() {
            let idx = start + k;
            if idx < self.control_points.len() {
                let cp = self.control_points[idx];
                deriv[0] += dn_k * cp[0];
                deriv[1] += dn_k * cp[1];
            }
        }
        deriv
    }

    /// Compute the arc length by numerical integration using n samples.
    pub fn arc_length(&self, n_samples: usize) -> f64 {
        let (t0, t1) = self.basis.domain();
        let dt = (t1 - t0) / n_samples as f64;
        let mut length = 0.0_f64;
        let mut prev = self.eval(t0);
        for i in 1..=n_samples {
            let t = t0 + i as f64 * dt;
            let curr = self.eval(t);
            let dx = curr[0] - prev[0];
            let dy = curr[1] - prev[1];
            length += (dx * dx + dy * dy).sqrt();
            prev = curr;
        }
        length
    }

    /// Return the parameter domain [t_min, t_max].
    pub fn domain(&self) -> (f64, f64) {
        self.basis.domain()
    }
}

// ---------------------------------------------------------------------------
// BSplineSurface
// ---------------------------------------------------------------------------

/// B-spline surface (tensor product) in 3-D.
///
/// Maps (u, v) ∈ \[0,1\]² to a 3-D point via:
/// S(u,v) = Σ_i Σ_j N_{i,p}(u) N_{j,q}(v) P_{ij}
#[derive(Debug, Clone)]
pub struct BSplineSurface {
    /// B-spline basis in the u direction.
    pub basis_u: BSplineBasis,
    /// B-spline basis in the v direction.
    pub basis_v: BSplineBasis,
    /// Control points P_{ij} ∈ ℝ³, indexed as \[i\]\[j\].
    pub control_points: Vec<Vec<[f64; 3]>>,
}

impl BSplineSurface {
    /// Create a B-spline surface.
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        control_points: Vec<Vec<[f64; 3]>>,
    ) -> IntegrateResult<Self> {
        let basis_u = BSplineBasis::new(degree_u, knots_u)?;
        let basis_v = BSplineBasis::new(degree_v, knots_v)?;

        if control_points.len() != basis_u.n_basis {
            return Err(IntegrateError::DimensionMismatch(format!(
                "control_points.len()={} != basis_u.n_basis={}",
                control_points.len(),
                basis_u.n_basis
            )));
        }
        for (i, row) in control_points.iter().enumerate() {
            if row.len() != basis_v.n_basis {
                return Err(IntegrateError::DimensionMismatch(format!(
                    "control_points[{i}].len()={} != basis_v.n_basis={}",
                    row.len(),
                    basis_v.n_basis
                )));
            }
        }

        Ok(Self {
            basis_u,
            basis_v,
            control_points,
        })
    }

    /// Evaluate the surface at (u, v).
    pub fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        let (span_u, n_u) = self.basis_u.eval_basis_functions(u);
        let (span_v, n_v) = self.basis_v.eval_basis_functions(v);
        let pu = self.basis_u.degree;
        let pv = self.basis_v.degree;
        let start_u = span_u.saturating_sub(pu);
        let start_v = span_v.saturating_sub(pv);

        let mut point = [0.0_f64; 3];
        for (ki, &n_ui) in n_u.iter().enumerate() {
            let i = start_u + ki;
            if i >= self.control_points.len() {
                continue;
            }
            for (kj, &n_vj) in n_v.iter().enumerate() {
                let j = start_v + kj;
                if j >= self.control_points[i].len() {
                    continue;
                }
                let cp = self.control_points[i][j];
                point[0] += n_ui * n_vj * cp[0];
                point[1] += n_ui * n_vj * cp[1];
                point[2] += n_ui * n_vj * cp[2];
            }
        }
        point
    }

    /// Evaluate the partial derivative ∂S/∂u at (u, v).
    pub fn eval_partial_u(&self, u: f64, v: f64) -> [f64; 3] {
        let (span_u, dn_u) = self.basis_u.eval_basis_derivatives(u);
        let (span_v, n_v) = self.basis_v.eval_basis_functions(v);
        let pu = self.basis_u.degree;
        let pv = self.basis_v.degree;
        let start_u = span_u.saturating_sub(pu);
        let start_v = span_v.saturating_sub(pv);

        let mut deriv = [0.0_f64; 3];
        for (ki, &dn_ui) in dn_u.iter().enumerate() {
            let i = start_u + ki;
            if i >= self.control_points.len() {
                continue;
            }
            for (kj, &n_vj) in n_v.iter().enumerate() {
                let j = start_v + kj;
                if j >= self.control_points[i].len() {
                    continue;
                }
                let cp = self.control_points[i][j];
                deriv[0] += dn_ui * n_vj * cp[0];
                deriv[1] += dn_ui * n_vj * cp[1];
                deriv[2] += dn_ui * n_vj * cp[2];
            }
        }
        deriv
    }

    /// Evaluate the partial derivative ∂S/∂v at (u, v).
    pub fn eval_partial_v(&self, u: f64, v: f64) -> [f64; 3] {
        let (span_u, n_u) = self.basis_u.eval_basis_functions(u);
        let (span_v, dn_v) = self.basis_v.eval_basis_derivatives(v);
        let pu = self.basis_u.degree;
        let pv = self.basis_v.degree;
        let start_u = span_u.saturating_sub(pu);
        let start_v = span_v.saturating_sub(pv);

        let mut deriv = [0.0_f64; 3];
        for (ki, &n_ui) in n_u.iter().enumerate() {
            let i = start_u + ki;
            if i >= self.control_points.len() {
                continue;
            }
            for (kj, &dn_vj) in dn_v.iter().enumerate() {
                let j = start_v + kj;
                if j >= self.control_points[i].len() {
                    continue;
                }
                let cp = self.control_points[i][j];
                deriv[0] += n_ui * dn_vj * cp[0];
                deriv[1] += n_ui * dn_vj * cp[1];
                deriv[2] += n_ui * dn_vj * cp[2];
            }
        }
        deriv
    }

    /// Compute the surface normal vector (cross product of partials) at (u, v).
    ///
    /// Returns the un-normalized normal vector S_u × S_v.
    pub fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        let su = self.eval_partial_u(u, v);
        let sv = self.eval_partial_v(u, v);
        // Cross product su × sv
        [
            su[1] * sv[2] - su[2] * sv[1],
            su[2] * sv[0] - su[0] * sv[2],
            su[0] * sv[1] - su[1] * sv[0],
        ]
    }

    /// Compute the unit normal at (u, v).
    pub fn unit_normal(&self, u: f64, v: f64) -> [f64; 3] {
        let n = self.normal(u, v);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-300 {
            [0.0, 0.0, 1.0]
        } else {
            [n[0] / len, n[1] / len, n[2] / len]
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_basis_partition_of_unity() {
        // B-spline basis functions must sum to 1 at every point.
        let basis = BSplineBasis::uniform_open(3, 6).expect("basis creation failed");
        let (t0, t1) = basis.domain();
        let n_test = 50;
        for k in 0..=n_test {
            let t = t0 + (t1 - t0) * k as f64 / n_test as f64 * 0.9999;
            let (_span, vals) = basis.eval_basis_functions(t);
            let sum: f64 = vals.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Partition of unity violated at t={t:.4}: sum={sum:.14}"
            );
        }
    }

    #[test]
    fn test_bspline_basis_non_negativity() {
        let basis = BSplineBasis::uniform_open(2, 5).expect("basis creation");
        let (t0, t1) = basis.domain();
        for k in 0..100 {
            let t = t0 + (t1 - t0) * k as f64 / 99.0 * 0.9999;
            let (_span, vals) = basis.eval_basis_functions(t);
            for &v in &vals {
                assert!(v >= -1e-12, "Basis function negative: {v}");
            }
        }
    }

    #[test]
    fn test_bspline_curve_endpoints_interpolation() {
        // Open B-spline curve should pass through its first and last control points.
        let control_points = vec![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 0.0]];
        let n = control_points.len();
        let degree = 3;
        let basis = BSplineBasis::uniform_open(degree, n).expect("basis");
        let (t0, t1) = basis.domain();
        let knots = basis.knots.clone();

        let curve =
            BSplineCurve::new(degree, knots, control_points.clone()).expect("curve creation");

        // Check start endpoint
        let p_start = curve.eval(t0 + 1e-12);
        assert!(
            (p_start[0] - control_points[0][0]).abs() < 1e-6
                && (p_start[1] - control_points[0][1]).abs() < 1e-6,
            "Curve start {p_start:?} != control_points[0] {:?}",
            control_points[0]
        );

        // Check end endpoint
        let p_end = curve.eval(t1 - 1e-12);
        let last = control_points[n - 1];
        assert!(
            (p_end[0] - last[0]).abs() < 1e-6 && (p_end[1] - last[1]).abs() < 1e-6,
            "Curve end {p_end:?} != control_points[last] {last:?}"
        );
    }

    #[test]
    fn test_bspline_curve_arc_length_straight() {
        // A straight-line B-spline from (0,0) to (1,0) should have arc length 1.
        let control_points = vec![[0.0, 0.0], [1.0, 0.0]];
        let degree = 1;
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let curve = BSplineCurve::new(degree, knots, control_points).expect("straight line curve");
        let length = curve.arc_length(100);
        assert!((length - 1.0).abs() < 0.01, "Arc length = {length}");
    }

    #[test]
    fn test_bspline_surface_eval_corners() {
        // Bilinear patch: degree 1 in both directions, 2×2 control points.
        // C(0,0)=[0,0,0], C(1,0)=[1,0,0], C(0,1)=[0,1,0], C(1,1)=[1,1,0]
        let cp = vec![
            vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let surf = BSplineSurface::new(1, 1, knots.clone(), knots, cp).expect("surface");

        let p00 = surf.eval(0.0, 0.0);
        let p10 = surf.eval(1.0 - 1e-12, 0.0);
        let p01 = surf.eval(0.0, 1.0 - 1e-12);
        let p11 = surf.eval(1.0 - 1e-12, 1.0 - 1e-12);

        assert!((p00[0]).abs() < 1e-10 && (p00[1]).abs() < 1e-10);
        assert!((p10[0] - 1.0).abs() < 1e-6 && (p10[1]).abs() < 1e-6);
        assert!((p01[0]).abs() < 1e-6 && (p01[1] - 1.0).abs() < 1e-6);
        assert!((p11[0] - 1.0).abs() < 1e-6 && (p11[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bspline_surface_normal_flat_patch() {
        // Flat patch in z=0 plane should have normal (0,0,1).
        let cp = vec![
            vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let surf = BSplineSurface::new(1, 1, knots.clone(), knots, cp).expect("surface");
        let n = surf.unit_normal(0.5, 0.5);
        assert!(
            (n[2].abs() - 1.0).abs() < 1e-10,
            "Normal z-component should be 1"
        );
    }
}
