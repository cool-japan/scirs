//! PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) and Akima interpolation
//!
//! This module exposes two shape-preserving 1-D interpolators as simple,
//! allocation-friendly types that work on plain `&[f64]` slices:
//!
//! - [`PchipInterpolator`] — monotonicity-preserving cubic Hermite interpolation
//!   using the Fritsch–Carlson derivative formula.
//! - [`AkimaInterpolator`] — Akima's method; more local and less sensitive to
//!   outlier data values.
//!
//! Both types support evaluation, batch evaluation, and definite integration
//! (antiderivative).  `PchipInterpolator` additionally exposes a derivative
//! method and a monotonicity check.
//!
//! # References
//!
//! - Fritsch, F. N. & Carlson, R. E. (1980).  "Monotone piecewise cubic
//!   interpolation."  *SIAM J. Numer. Anal.* **17**(2), 238–246.
//! - Fritsch, F. N. & Butland, J. (1984).  "A method for constructing local
//!   monotone piecewise cubic interpolants."  *SIAM J. Sci. Stat. Comput.*
//!   **5**(2), 300–304.
//! - Akima, H. (1970).  "A new method of interpolation and smooth curve fitting
//!   based on local procedures."  *J. ACM* **17**(4), 589–602.

use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// Helper: binary search for interval
// ---------------------------------------------------------------------------

/// Return index `i` such that `x[i] <= xi < x[i+1]`, clamped to `[0, n-2]`.
fn find_interval(xs: &[f64], xi: f64) -> usize {
    let n = xs.len();
    // clamp outside range
    if xi <= xs[0] {
        return 0;
    }
    if xi >= xs[n - 1] {
        return n - 2;
    }
    // binary search
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// Hermite basis evaluation helpers
// ---------------------------------------------------------------------------

/// Evaluate the Hermite cubic on [0,1] at local parameter `t`.
///
/// h_00(t) = 2t³ - 3t² + 1
/// h_10(t) = t³ - 2t² + t
/// h_01(t) = -2t³ + 3t²
/// h_11(t) = t³ - t²
#[inline]
fn hermite_eval(y0: f64, y1: f64, m0: f64, m1: f64, h: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
}

/// Derivative of the Hermite cubic at local `t`.
#[inline]
fn hermite_deriv(y0: f64, y1: f64, m0: f64, m1: f64, h: f64, t: f64) -> f64 {
    let t2 = t * t;
    // d/dt: (6t² - 6t)/h * (y0-y1) + (3t²-4t+1)*m0 + (3t²-2t)*m1
    let dh00 = 6.0 * t2 - 6.0 * t;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 = 3.0 * t2 - 2.0 * t;
    (dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1) / h
}

/// Definite integral of the Hermite cubic from t=0 to t=1, scaled by h.
///
/// ∫₀¹ p(t) dt  * h
#[inline]
fn hermite_integral(y0: f64, y1: f64, m0: f64, m1: f64, h: f64) -> f64 {
    // anti-derivative of Hermite: F(1) - F(0)
    // ∫ h00 dt = [t²/2 - t³/3 + t] from which integral over [0,1] = 1/2
    // ∫ h10 dt = [t³/4 - 2t³/3 + t²/2] => 1/4-2/3+1/2 = 3/12-8/12+6/12 = 1/12
    // ∫ h01 dt = 1/2
    // ∫ h11 dt = 1/4 - 1/2 = -1/4 => -1/12
    // Actually recompute analytically:
    // ∫₀¹ (2t³-3t²+1) dt = [t⁴/2 - t³ + t]₀¹ = 1/2-1+1 = 1/2
    // ∫₀¹ (t³-2t²+t)  dt = [t⁴/4 - 2t³/3 + t²/2]₀¹ = 1/4-2/3+1/2 = 1/12
    // ∫₀¹ (-2t³+3t²)  dt = [-t⁴/2+t³]₀¹ = -1/2+1 = 1/2
    // ∫₀¹ (t³-t²)     dt = [t⁴/4-t³/3]₀¹ = 1/4-1/3 = -1/12
    h * (0.5 * y0 + (1.0 / 12.0) * h * m0 + 0.5 * y1 + (-1.0 / 12.0) * h * m1)
}

// ---------------------------------------------------------------------------
// PchipInterpolator
// ---------------------------------------------------------------------------

/// Shape-preserving piecewise cubic Hermite interpolator (PCHIP).
///
/// Uses the Fritsch–Carlson algorithm to select derivatives at each knot so
/// that the interpolant is monotone on each interval where the data is monotone.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::pchip::PchipInterpolator;
///
/// let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
/// let interp = PchipInterpolator::new(&x, &y).expect("construction failed");
///
/// // Exact at knots
/// assert!((interp.evaluate(2.0) - 4.0).abs() < 1e-12);
///
/// // Monotone interpolation between knots
/// let v = interp.evaluate(2.5);
/// assert!(v > 4.0 && v < 9.0);
/// ```
#[derive(Debug, Clone)]
pub struct PchipInterpolator {
    /// Knot x-coordinates (strictly increasing).
    x: Vec<f64>,
    /// Knot y-values.
    y: Vec<f64>,
    /// Hermite derivatives at each knot.
    derivatives: Vec<f64>,
}

impl PchipInterpolator {
    /// Construct a PCHIP interpolator.
    ///
    /// # Arguments
    ///
    /// * `x` — knot positions (must be strictly increasing, at least 2 points).
    /// * `y` — data values at `x`.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError`] when input validation fails.
    pub fn new(x: &[f64], y: &[f64]) -> InterpolateResult<Self> {
        let n = x.len();
        if n < 2 {
            return Err(InterpolateError::insufficient_points(2, n, "PCHIP"));
        }
        if y.len() != n {
            return Err(InterpolateError::invalid_input(
                "x and y must have the same length".to_string(),
            ));
        }
        for i in 1..n {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x must be strictly increasing".to_string(),
                ));
            }
        }

        let derivatives = Self::compute_derivatives(x, y);
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            derivatives,
        })
    }

    /// Fritsch–Carlson derivative computation.
    ///
    /// For each interior knot the derivative is set using the weighted harmonic
    /// mean of the adjacent chord slopes, ensuring monotonicity.  Endpoint
    /// derivatives use a one-sided finite difference.
    fn compute_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut d = vec![0.0f64; n];

        // Chord slopes
        let mut delta = vec![0.0f64; n - 1];
        let mut h_arr = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            h_arr[i] = x[i + 1] - x[i];
            delta[i] = (y[i + 1] - y[i]) / h_arr[i];
        }

        // Endpoint derivatives (one-sided, Fritsch–Butland formula)
        d[0] = Self::endpoint_derivative(delta[0], delta[1], h_arr[0], h_arr[1]);
        d[n - 1] = Self::endpoint_derivative(
            delta[n - 2],
            delta[n - 3].max(delta[n - 2]),
            h_arr[n - 2],
            h_arr[n - 3].max(h_arr[n - 2]),
        );
        // Simpler: classic one-sided
        d[0] = ((2.0 * h_arr[0] + h_arr[1]) * delta[0] - h_arr[0] * delta[1])
            / (h_arr[0] + h_arr[1]);
        d[n - 1] = ((2.0 * h_arr[n - 2] + h_arr[n - 3]) * delta[n - 2]
            - h_arr[n - 2] * delta[n - 3])
            / (h_arr[n - 2] + h_arr[n - 3]);

        // Clamp endpoints so they don't overshoot
        if d[0] * delta[0] < 0.0 {
            d[0] = 0.0;
        } else if delta[0] != 0.0 && (d[0] / delta[0]).abs() > 3.0 {
            d[0] = 3.0 * delta[0];
        }
        if d[n - 1] * delta[n - 2] < 0.0 {
            d[n - 1] = 0.0;
        } else if delta[n - 2] != 0.0 && (d[n - 1] / delta[n - 2]).abs() > 3.0 {
            d[n - 1] = 3.0 * delta[n - 2];
        }

        // Interior knots — weighted harmonic mean
        for i in 1..n - 1 {
            if delta[i - 1] * delta[i] <= 0.0 {
                // sign change → local extremum; set derivative to 0
                d[i] = 0.0;
            } else {
                let w1 = 2.0 * h_arr[i] + h_arr[i - 1];
                let w2 = h_arr[i] + 2.0 * h_arr[i - 1];
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }

        d
    }

    /// One-sided endpoint derivative (Fritsch–Butland).
    #[inline]
    fn endpoint_derivative(d0: f64, d1: f64, h0: f64, h1: f64) -> f64 {
        let v = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        // ensure same sign as d0 and not steeper than 3*d0
        if v * d0 <= 0.0 {
            return 0.0;
        }
        if d0 != 0.0 && (v / d0).abs() > 3.0 {
            return 3.0 * d0;
        }
        v
    }

    /// Evaluate the interpolant at `xi`.
    ///
    /// Evaluates via cubic Hermite polynomial on the containing interval.
    /// Outside the data range the result is extrapolated using the cubic at
    /// the nearest boundary interval.
    pub fn evaluate(&self, xi: f64) -> f64 {
        let i = find_interval(&self.x, xi);
        let h = self.x[i + 1] - self.x[i];
        let t = (xi - self.x[i]) / h;
        hermite_eval(
            self.y[i],
            self.y[i + 1],
            self.derivatives[i],
            self.derivatives[i + 1],
            h,
            t,
        )
    }

    /// Evaluate the interpolant at each point in `xi`.
    pub fn evaluate_batch(&self, xi: &[f64]) -> Vec<f64> {
        xi.iter().map(|&v| self.evaluate(v)).collect()
    }

    /// First derivative of the interpolant at `xi`.
    pub fn derivative(&self, xi: f64) -> f64 {
        let i = find_interval(&self.x, xi);
        let h = self.x[i + 1] - self.x[i];
        let t = (xi - self.x[i]) / h;
        hermite_deriv(
            self.y[i],
            self.y[i + 1],
            self.derivatives[i],
            self.derivatives[i + 1],
            h,
            t,
        )
    }

    /// Definite integral ∫ₐᵇ p(x) dx.
    ///
    /// Integrates the piecewise-cubic exactly by summing over each interval
    /// that lies within `[a, b]`.
    pub fn antiderivative(&self, a: f64, b: f64) -> f64 {
        if a == b {
            return 0.0;
        }
        let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
        let n = self.x.len();
        let mut total = 0.0f64;
        for i in 0..n - 1 {
            let x0 = self.x[i];
            let x1 = self.x[i + 1];
            // Does this interval overlap [lo, hi]?
            if x1 <= lo || x0 >= hi {
                continue;
            }
            let a_local = lo.max(x0);
            let b_local = hi.min(x1);
            let h = x1 - x0;
            let ta = (a_local - x0) / h;
            let tb = (b_local - x0) / h;
            total += self.integrate_segment(i, ta, tb, h);
        }
        sign * total
    }

    /// Integrate segment `i` from local parameter `ta` to `tb`.
    fn integrate_segment(&self, i: usize, ta: f64, tb: f64, h: f64) -> f64 {
        // Antiderivative of the Hermite cubic at parameter t:
        //   F(t) = h * [ h00_int*y0 + h10_int*h*m0 + h01_int*y1 + h11_int*h*m1 ]
        // where h00_int(t) = t^4/2 - t^3 + t, etc.
        let antideriv = |t: f64| -> f64 {
            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t3 * t;
            let int_h00 = 0.5 * t4 - t3 + t;
            let int_h10 = 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2;
            let int_h01 = -0.5 * t4 + t3;
            let int_h11 = 0.25 * t4 - (1.0 / 3.0) * t3;
            h * (int_h00 * self.y[i]
                + int_h10 * h * self.derivatives[i]
                + int_h01 * self.y[i + 1]
                + int_h11 * h * self.derivatives[i + 1])
        };
        antideriv(tb) - antideriv(ta)
    }

    /// Check whether the interpolant preserves monotonicity of the input data.
    ///
    /// Returns `true` when every interval where the data is monotone is also
    /// monotone in the interpolant.
    pub fn is_monotone(&self) -> bool {
        let n = self.x.len();
        for i in 0..n - 1 {
            let h = self.x[i + 1] - self.x[i];
            let delta = (self.y[i + 1] - self.y[i]) / h;
            let m0 = self.derivatives[i];
            let m1 = self.derivatives[i + 1];
            // Fritsch–Carlson condition: alpha^2 + beta^2 <= 9
            if delta.abs() < f64::EPSILON {
                if m0.abs() > f64::EPSILON || m1.abs() > f64::EPSILON {
                    return false;
                }
                continue;
            }
            let alpha = m0 / delta;
            let beta = m1 / delta;
            if alpha < 0.0 || beta < 0.0 {
                return false;
            }
            if alpha * alpha + beta * beta > 9.0 + 1e-10 {
                return false;
            }
        }
        true
    }

    /// Number of knots.
    pub fn n_knots(&self) -> usize {
        self.x.len()
    }

    /// Read-only access to the computed Hermite derivatives.
    pub fn derivatives(&self) -> &[f64] {
        &self.derivatives
    }

    /// Convert the Hermite representation to power-basis piecewise-cubic coefficients.
    ///
    /// Returns `(breakpoints, coeffs)` where `coeffs` has shape `(n_seg, 4)` and
    /// row `i` holds `[a, b, c, d]` such that
    ///
    /// ```text
    /// p_i(x) = a + b*(x - x_i) + c*(x - x_i)^2 + d*(x - x_i)^3
    /// ```
    ///
    /// Used by `symbolic_derivative::Differentiable`.
    pub(crate) fn to_power_basis_coeffs(
        &self,
    ) -> crate::error::InterpolateResult<(
        scirs2_core::ndarray::Array1<f64>,
        scirs2_core::ndarray::Array2<f64>,
    )> {
        let n = self.x.len();
        if n < 2 {
            return Err(crate::error::InterpolateError::invalid_input(
                "PCHIP requires at least 2 knots".to_string(),
            ));
        }
        let n_seg = n - 1;
        let mut coeffs = scirs2_core::ndarray::Array2::<f64>::zeros((n_seg, 4));
        for i in 0..n_seg {
            let h = self.x[i + 1] - self.x[i];
            let y0 = self.y[i];
            let y1 = self.y[i + 1];
            let m0 = self.derivatives[i];
            let m1 = self.derivatives[i + 1];
            // Power-basis: a + b*t + c*t^2 + d*t^3, t = x - x_i
            let a = y0;
            let b = m0;
            let c = (3.0 * (y1 - y0) / h - 2.0 * m0 - m1) / h;
            let d = (2.0 * (y0 - y1) / h + m0 + m1) / (h * h);
            coeffs[[i, 0]] = a;
            coeffs[[i, 1]] = b;
            coeffs[[i, 2]] = c;
            coeffs[[i, 3]] = d;
        }
        let bp = scirs2_core::ndarray::Array1::from(self.x.clone());
        Ok((bp, coeffs))
    }
}

// ---------------------------------------------------------------------------
// AkimaInterpolator
// ---------------------------------------------------------------------------

/// Akima spline interpolator.
///
/// Akima's method computes derivatives at each knot by looking at the local
/// slope pattern over a five-point stencil.  This makes the interpolant
/// insensitive to outlier data values and avoids oscillation.
///
/// Unlike cubic splines, Akima splines do **not** satisfy global smoothness
/// conditions; they are only C¹ (not C²).
///
/// # Minimum data requirement
///
/// At least **5** data points are required so that the endpoint ghost slopes
/// can be computed.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::pchip::AkimaInterpolator;
///
/// let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
/// let interp = AkimaInterpolator::new(&x, &y).expect("construction failed");
/// let v = interp.evaluate(2.5);
/// assert!((v - 6.25).abs() < 0.5); // near y=x^2 at x=2.5
/// ```
#[derive(Debug, Clone)]
pub struct AkimaInterpolator {
    /// Knot x-coordinates (strictly increasing).
    x: Vec<f64>,
    /// Knot y-values.
    y: Vec<f64>,
    /// Cubic coefficients [a, b, c, d] per interval.
    ///
    /// On interval i: p_i(x) = a + b·(x-x_i) + c·(x-x_i)² + d·(x-x_i)³
    spline_coeffs: Vec<[f64; 4]>,
}

impl AkimaInterpolator {
    /// Construct an Akima interpolator.
    ///
    /// # Arguments
    ///
    /// * `x` — knot positions (strictly increasing, at least 5 points).
    /// * `y` — data values at `x`.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError`] when input validation fails.
    pub fn new(x: &[f64], y: &[f64]) -> InterpolateResult<Self> {
        let n = x.len();
        if n < 5 {
            return Err(InterpolateError::insufficient_points(5, n, "Akima"));
        }
        if y.len() != n {
            return Err(InterpolateError::invalid_input(
                "x and y must have the same length".to_string(),
            ));
        }
        for i in 1..n {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x must be strictly increasing".to_string(),
                ));
            }
        }

        let spline_coeffs = Self::compute_coefficients(x, y);
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            spline_coeffs,
        })
    }

    /// Compute cubic spline coefficients using Akima's formula.
    fn compute_coefficients(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
        let n = x.len();

        // Chord slopes between consecutive knots.
        let mut m = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            m[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        }

        // Extend slopes with two ghost entries at each end.
        // m_ext[0..=1] are ghosts on the left,
        // m_ext[2..n+1] correspond to m[0..n-1],
        // m_ext[n+1..=n+2] are ghosts on the right.
        let mut m_ext = vec![0.0f64; n + 3];
        m_ext[2..n + 1].copy_from_slice(&m);
        m_ext[0] = 3.0 * m_ext[2] - 2.0 * m_ext[3];
        m_ext[1] = 2.0 * m_ext[2] - m_ext[3];
        m_ext[n + 1] = 2.0 * m_ext[n] - m_ext[n - 1];
        m_ext[n + 2] = 3.0 * m_ext[n] - 2.0 * m_ext[n - 1];

        // Compute derivative at each knot using Akima's weighted average.
        let mut t = vec![0.0f64; n];
        for k in 0..n {
            // Using extended index: knot k corresponds to m_ext[k+2]
            let w1 = (m_ext[k + 3] - m_ext[k + 2]).abs();
            let w2 = (m_ext[k + 1] - m_ext[k]).abs();
            if w1 + w2 < f64::EPSILON {
                // Both weights zero: simple average
                t[k] = 0.5 * (m_ext[k + 1] + m_ext[k + 2]);
            } else {
                t[k] = (w1 * m_ext[k + 1] + w2 * m_ext[k + 2]) / (w1 + w2);
            }
        }

        // Build cubic coefficients on each interval [x_i, x_{i+1}].
        let mut coeffs = vec![[0.0f64; 4]; n - 1];
        for i in 0..n - 1 {
            let h = x[i + 1] - x[i];
            let a = y[i];
            let b = t[i];
            let c = (3.0 * m[i] - 2.0 * t[i] - t[i + 1]) / h;
            let d = (t[i] + t[i + 1] - 2.0 * m[i]) / (h * h);
            coeffs[i] = [a, b, c, d];
        }
        coeffs
    }

    /// Evaluate the Akima spline at `xi`.
    pub fn evaluate(&self, xi: f64) -> f64 {
        let i = find_interval(&self.x, xi);
        let dx = xi - self.x[i];
        let [a, b, c, d] = self.spline_coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate the Akima spline at each point in `xi`.
    pub fn evaluate_batch(&self, xi: &[f64]) -> Vec<f64> {
        xi.iter().map(|&v| self.evaluate(v)).collect()
    }

    /// First derivative of the Akima spline at `xi`.
    pub fn derivative(&self, xi: f64) -> f64 {
        let i = find_interval(&self.x, xi);
        let dx = xi - self.x[i];
        let [_a, b, c, d] = self.spline_coeffs[i];
        b + dx * (2.0 * c + 3.0 * dx * d)
    }

    /// Definite integral ∫ₐᵇ p(x) dx (exact over the piecewise cubic).
    pub fn antiderivative(&self, a: f64, b: f64) -> f64 {
        if a == b {
            return 0.0;
        }
        let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
        let n = self.x.len();
        let mut total = 0.0f64;
        for i in 0..n - 1 {
            let x0 = self.x[i];
            let x1 = self.x[i + 1];
            if x1 <= lo || x0 >= hi {
                continue;
            }
            let a_local = lo.max(x0);
            let b_local = hi.min(x1);
            let [a_c, b_c, c_c, d_c] = self.spline_coeffs[i];
            // Antiderivative: A + B·u + C·u²/2 + D·u³/3 + E·u⁴/4 where u = x - x_i
            let antideriv = |u: f64| -> f64 {
                a_c * u + b_c * u * u * 0.5 + c_c * u * u * u / 3.0
                    + d_c * u * u * u * u * 0.25
            };
            total += antideriv(b_local - x0) - antideriv(a_local - x0);
        }
        sign * total
    }

    /// Number of knots.
    pub fn n_knots(&self) -> usize {
        self.x.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // --- PchipInterpolator tests ---

    #[test]
    fn test_pchip_exact_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        for (xi, yi) in x.iter().zip(y.iter()) {
            let v = p.evaluate(*xi);
            assert!(
                (v - yi).abs() < 1e-10,
                "PCHIP should be exact at knots: xi={xi} expected={yi} got={v}"
            );
        }
    }

    #[test]
    fn test_pchip_monotone_data() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        assert!(p.is_monotone(), "Expected monotone interpolant for monotone data");
        // All interpolated values should be in [0,10]
        for i in 0..40 {
            let xi = i as f64 / 10.0;
            let v = p.evaluate(xi);
            assert!(v >= -1e-10 && v <= 10.0 + 1e-10, "value {v} out of range at xi={xi}");
        }
    }

    #[test]
    fn test_pchip_non_monotone_data() {
        // Data with a local maximum
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 3.0, 5.0, 3.0, 0.0];
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        // is_monotone should be false overall, but each sub-interval can be checked
        // Just verify exact at knots
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!((p.evaluate(*xi) - yi).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pchip_batch_evaluation() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        let xi = vec![0.5, 1.5, 2.5];
        let vals = p.evaluate_batch(&xi);
        assert_eq!(vals.len(), 3);
        for v in &vals {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_pchip_derivative() {
        // For linear data the derivative should equal the slope
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 4.0, 6.0, 8.0]; // slope = 2
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        // Interior derivative must be 2 for linear data
        let d = p.derivative(2.0);
        assert!((d - 2.0).abs() < 1e-8, "derivative={d} expected ~2.0");
    }

    #[test]
    fn test_pchip_antiderivative_linear() {
        // For f(x)=2x on [0,4], antiderivative should be 16
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        let integral = p.antiderivative(0.0, 4.0);
        assert!((integral - 16.0).abs() < 1e-8, "integral={integral} expected=16");
    }

    #[test]
    fn test_pchip_antiderivative_reversed() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let p = PchipInterpolator::new(&x, &y).expect("construction failed");
        let fwd = p.antiderivative(0.0, 4.0);
        let rev = p.antiderivative(4.0, 0.0);
        assert!((fwd + rev).abs() < 1e-10);
    }

    #[test]
    fn test_pchip_error_too_few_points() {
        let x = vec![0.0];
        let y = vec![1.0];
        assert!(PchipInterpolator::new(&x, &y).is_err());
    }

    #[test]
    fn test_pchip_error_length_mismatch() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        assert!(PchipInterpolator::new(&x, &y).is_err());
    }

    #[test]
    fn test_pchip_error_unsorted() {
        let x = vec![0.0, 2.0, 1.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        assert!(PchipInterpolator::new(&x, &y).is_err());
    }

    // --- AkimaInterpolator tests ---

    #[test]
    fn test_akima_exact_at_knots() {
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let a = AkimaInterpolator::new(&x, &y).expect("construction failed");
        for (xi, yi) in x.iter().zip(y.iter()) {
            let v = a.evaluate(*xi);
            assert!(
                (v - yi).abs() < 1e-8,
                "Akima should be exact at knots: xi={xi} expected={yi} got={v}"
            );
        }
    }

    #[test]
    fn test_akima_smooth_sine() {
        let n = 20;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 2.0 * PI / (n as f64)).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        let a = AkimaInterpolator::new(&x, &y).expect("construction failed");

        // Check interpolated values at midpoints are close to sin
        for i in 0..n - 1 {
            let xi = (x[i] + x[i + 1]) * 0.5;
            let approx = a.evaluate(xi);
            let exact = xi.sin();
            assert!(
                (approx - exact).abs() < 0.01,
                "Akima sine error at xi={xi}: approx={approx} exact={exact}"
            );
        }
    }

    #[test]
    fn test_akima_batch() {
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        let a = AkimaInterpolator::new(&x, &y).expect("construction failed");
        let xi = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let vals = a.evaluate_batch(&xi);
        assert_eq!(vals.len(), xi.len());
        for v in &vals {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_akima_antiderivative_linear() {
        // f(x)=x, ∫₀⁵ x dx = 12.5
        let x: Vec<f64> = (0..7).map(|i| i as f64).collect();
        let y: Vec<f64> = x.clone();
        let a = AkimaInterpolator::new(&x, &y).expect("construction failed");
        let integral = a.antiderivative(0.0, 5.0);
        assert!((integral - 12.5).abs() < 1e-6, "integral={integral} expected=12.5");
    }

    #[test]
    fn test_akima_min_points_error() {
        let x = vec![0.0, 1.0, 2.0, 3.0]; // only 4 points
        let y = vec![0.0, 1.0, 0.0, 1.0];
        assert!(AkimaInterpolator::new(&x, &y).is_err());
    }

    #[test]
    fn test_akima_derivative() {
        // f(x)=x^2, f'(x)=2x; test at interior knot
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let a = AkimaInterpolator::new(&x, &y).expect("construction failed");
        let d = a.derivative(3.0);
        assert!((d - 6.0).abs() < 0.5, "derivative at x=3 should be ~6, got {d}");
    }
}
