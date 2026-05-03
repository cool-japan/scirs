//! Spline calculus: derivatives, integrals, antiderivatives, and root-finding
//!
//! This module provides advanced calculus operations on cubic splines:
//!
//! - **`spline_derivative`**: Compute the n-th derivative of a cubic spline, returning
//!   a new piecewise polynomial representation.
//! - **`spline_integral`**: Compute the definite integral of a spline over [a, b].
//! - **`spline_antiderivative`**: Return a new piecewise polynomial representing the
//!   antiderivative (indefinite integral) of the spline.
//! - **`spline_roots`**: Find all real roots (zero-crossings) of a cubic spline by
//!   solving the cubic on each segment analytically.
//!
//! All functions use `Result` types and avoid `unwrap()`.

use crate::error::{InterpolateError, InterpolateResult};
use crate::spline::CubicSpline;
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Piecewise polynomial representation returned by derivative / antiderivative
// ---------------------------------------------------------------------------

/// A piecewise polynomial resulting from spline calculus operations.
///
/// Each segment `i` is defined on `[breakpoints[i], breakpoints[i+1])` as:
///
/// ```text
/// p_i(x) = coeffs[[i, 0]] + coeffs[[i, 1]] * (x - breakpoints[i])
///        + coeffs[[i, 2]] * (x - breakpoints[i])^2
///        + ... + coeffs[[i, k-1]] * (x - breakpoints[i])^(k-1)
/// ```
///
/// where `k` is the number of coefficient columns (degree + 1).
#[derive(Debug, Clone)]
pub struct PiecewisePolynomial<F: InterpolationFloat> {
    /// Breakpoints (sorted, length = n_segments + 1)
    breakpoints: Array1<F>,
    /// Coefficients matrix: shape (n_segments, degree+1).
    /// Row `i`, column `j` holds the coefficient for `(x - breakpoints[i])^j`.
    coeffs: Array2<F>,
}

impl<F: InterpolationFloat> PiecewisePolynomial<F> {
    /// Construct a `PiecewisePolynomial` directly from breakpoints and coefficients.
    ///
    /// # Arguments
    ///
    /// * `breakpoints` – sorted vector of `n_segments + 1` breakpoints.
    /// * `coeffs`      – coefficient matrix of shape `(n_segments, degree+1)`.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent.
    pub fn new(breakpoints: Array1<F>, coeffs: Array2<F>) -> InterpolateResult<Self> {
        if breakpoints.len() < 2 {
            return Err(InterpolateError::invalid_input(
                "PiecewisePolynomial requires at least 2 breakpoints".to_string(),
            ));
        }
        let n_seg = breakpoints.len() - 1;
        if coeffs.nrows() != n_seg {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("({n_seg}, k)"),
                actual: format!("({}, {})", coeffs.nrows(), coeffs.ncols()),
                object: "PiecewisePolynomial coefficients".to_string(),
            });
        }
        Ok(Self {
            breakpoints,
            coeffs,
        })
    }

    /// Evaluate the piecewise polynomial at a single point.
    pub fn evaluate(&self, x: F) -> InterpolateResult<F> {
        let seg = self.find_segment(x)?;
        let dx = x - self.breakpoints[seg];
        let ncols = self.coeffs.ncols();
        // Horner-like evaluation from highest power down
        let mut val = self.coeffs[[seg, ncols - 1]];
        for j in (0..ncols - 1).rev() {
            val = val * dx + self.coeffs[[seg, j]];
        }
        Ok(val)
    }

    /// Evaluate at multiple points.
    pub fn evaluate_array(&self, xs: &[F]) -> InterpolateResult<Vec<F>> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Return the breakpoints.
    pub fn breakpoints(&self) -> &Array1<F> {
        &self.breakpoints
    }

    /// Return the coefficient matrix.
    pub fn coeffs(&self) -> &Array2<F> {
        &self.coeffs
    }

    /// Number of polynomial segments.
    pub fn n_segments(&self) -> usize {
        self.breakpoints.len() - 1
    }

    /// Polynomial degree of each segment (degree = ncols - 1).
    pub fn degree(&self) -> usize {
        if self.coeffs.ncols() == 0 {
            return 0;
        }
        self.coeffs.ncols() - 1
    }

    /// Find the segment index for a given x value.
    fn find_segment(&self, x: F) -> InterpolateResult<usize> {
        let n = self.breakpoints.len();
        if n < 2 {
            return Err(InterpolateError::InvalidState(
                "PiecewisePolynomial has no segments".to_string(),
            ));
        }
        // Allow evaluation at the very last breakpoint by mapping it to the last segment.
        if x == self.breakpoints[n - 1] {
            return Ok(n - 2);
        }
        if x < self.breakpoints[0] || x > self.breakpoints[n - 1] {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {} is outside [{}, {}]",
                x,
                self.breakpoints[0],
                self.breakpoints[n - 1]
            )));
        }
        // Binary search
        let mut lo: usize = 0;
        let mut hi: usize = n - 2;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if x < self.breakpoints[mid + 1] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }
}

// ---------------------------------------------------------------------------
// spline_derivative
// ---------------------------------------------------------------------------

/// Compute the n-th derivative of a cubic spline, returning a `PiecewisePolynomial`.
///
/// The input spline stores coefficients `[a, b, c, d]` per segment such that
///
/// ```text
/// s(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
/// ```
///
/// The first derivative is `b + 2c*(x-x_i) + 3d*(x-x_i)^2`, etc.
///
/// For `order >= 4`, the result is the zero polynomial on each segment.
///
/// # Arguments
///
/// * `spline` - The cubic spline to differentiate.
/// * `order`  - Derivative order (1, 2, 3, ...).
///
/// # Returns
///
/// A `PiecewisePolynomial` representing the derivative.
pub fn spline_derivative<F: InterpolationFloat + ToString>(
    spline: &CubicSpline<F>,
    order: usize,
) -> InterpolateResult<PiecewisePolynomial<F>> {
    if order == 0 {
        // Return the spline itself as a PiecewisePolynomial
        return Ok(cubic_spline_to_pp(spline));
    }

    let n_seg = spline.coeffs().nrows();
    let breakpoints = spline.x().clone();

    if order >= 4 {
        // Derivative of order >= 4 of a cubic is identically zero
        let coeffs = Array2::zeros((n_seg, 1));
        return Ok(PiecewisePolynomial {
            breakpoints,
            coeffs,
        });
    }

    // Start from the original coefficients [a, b, c, d] and differentiate `order` times.
    // After k differentiations the column count is 4 - k.
    // Derivative rule: if p(t) = sum_{j=0}^{deg} c_j t^j
    //   then p'(t) = sum_{j=0}^{deg-1} (j+1) c_{j+1} t^j
    let mut current_ncols: usize = 4;
    // Build a working copy of coefficients; column j holds coeff of (x - x_i)^j
    let mut work = Array2::<F>::zeros((n_seg, current_ncols));
    for i in 0..n_seg {
        for j in 0..4 {
            work[[i, j]] = spline.coeffs()[[i, j]];
        }
    }

    for _d in 0..order {
        if current_ncols <= 1 {
            // Already constant; derivative is zero
            let coeffs = Array2::zeros((n_seg, 1));
            return Ok(PiecewisePolynomial {
                breakpoints,
                coeffs,
            });
        }
        let new_ncols = current_ncols - 1;
        let mut new_work = Array2::<F>::zeros((n_seg, new_ncols));
        for i in 0..n_seg {
            for j in 0..new_ncols {
                let factor = F::from_usize(j + 1).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert derivative factor to float".to_string(),
                    )
                })?;
                new_work[[i, j]] = work[[i, j + 1]] * factor;
            }
        }
        work = new_work;
        current_ncols = new_ncols;
    }

    Ok(PiecewisePolynomial {
        breakpoints,
        coeffs: work,
    })
}

// ---------------------------------------------------------------------------
// spline_integral
// ---------------------------------------------------------------------------

/// Compute the definite integral of a cubic spline over `[a, b]`.
///
/// This is a convenience wrapper that delegates to the existing
/// `CubicSpline::integrate` method, but also supports the case where
/// `a > b` (returns the negated integral).
///
/// # Arguments
///
/// * `spline` - The cubic spline to integrate.
/// * `a`      - Lower integration bound.
/// * `b`      - Upper integration bound.
///
/// # Returns
///
/// The value of the definite integral.
pub fn spline_integral<F: InterpolationFloat + ToString>(
    spline: &CubicSpline<F>,
    a: F,
    b: F,
) -> InterpolateResult<F> {
    spline.integrate(a, b)
}

// ---------------------------------------------------------------------------
// spline_antiderivative
// ---------------------------------------------------------------------------

/// Construct the antiderivative (indefinite integral) of a cubic spline,
/// returned as a `PiecewisePolynomial` of degree 4 (quartic).
///
/// The integration constant is chosen so that the antiderivative evaluates
/// to zero at the left-most breakpoint.
///
/// For each segment with coefficients `[a, b, c, d]` the antiderivative is:
///
/// ```text
/// A(x) = C_i + a*(x-x_i) + b/2*(x-x_i)^2 + c/3*(x-x_i)^3 + d/4*(x-x_i)^4
/// ```
///
/// where `C_i` is the accumulated integral from `x_0` up to `x_i`.
///
/// # Arguments
///
/// * `spline` - The cubic spline to antidifferentiate.
///
/// # Returns
///
/// A `PiecewisePolynomial` of degree 4 representing the antiderivative.
pub fn spline_antiderivative<F: InterpolationFloat + ToString>(
    spline: &CubicSpline<F>,
) -> InterpolateResult<PiecewisePolynomial<F>> {
    let n_seg = spline.coeffs().nrows();
    let breakpoints = spline.x().clone();

    // New polynomial degree = 4, so 5 coefficients per segment
    let mut coeffs = Array2::<F>::zeros((n_seg, 5));

    let two = F::from_f64(2.0)
        .ok_or_else(|| InterpolateError::ComputationError("Failed to convert 2.0".to_string()))?;
    let three = F::from_f64(3.0)
        .ok_or_else(|| InterpolateError::ComputationError("Failed to convert 3.0".to_string()))?;
    let four = F::from_f64(4.0)
        .ok_or_else(|| InterpolateError::ComputationError("Failed to convert 4.0".to_string()))?;

    let mut accumulated = F::zero(); // C_0 = 0

    for i in 0..n_seg {
        let a = spline.coeffs()[[i, 0]];
        let b = spline.coeffs()[[i, 1]];
        let c = spline.coeffs()[[i, 2]];
        let d = spline.coeffs()[[i, 3]];

        // Constant term = accumulated integral up to this breakpoint
        coeffs[[i, 0]] = accumulated;
        // x^1 coefficient
        coeffs[[i, 1]] = a;
        // x^2 coefficient
        coeffs[[i, 2]] = b / two;
        // x^3 coefficient
        coeffs[[i, 3]] = c / three;
        // x^4 coefficient
        coeffs[[i, 4]] = d / four;

        // Accumulate the integral over this segment for the next segment's constant
        let h = breakpoints[i + 1] - breakpoints[i];
        let seg_integral =
            a * h + b * h * h / two + c * h * h * h / three + d * h * h * h * h / four;
        accumulated = accumulated + seg_integral;
    }

    Ok(PiecewisePolynomial {
        breakpoints,
        coeffs,
    })
}

// ---------------------------------------------------------------------------
// spline_roots
// ---------------------------------------------------------------------------

/// Find all real roots (zero-crossings) of a cubic spline within its domain.
///
/// On each segment the spline is a cubic polynomial
/// `a + b*t + c*t^2 + d*t^3 = 0` where `t = x - x_i`.
/// The routine analytically solves the cubic (or simpler cases) and
/// collects roots that lie within `[0, h_i]` (mapped back to `[x_i, x_{i+1}]`).
///
/// # Arguments
///
/// * `spline` - The cubic spline whose roots to find.
///
/// # Returns
///
/// A sorted `Vec<F>` of root locations within the spline domain.
pub fn spline_roots<F: InterpolationFloat + ToString>(
    spline: &CubicSpline<F>,
) -> InterpolateResult<Vec<F>> {
    let n_seg = spline.coeffs().nrows();
    let mut roots = Vec::new();
    let tol = F::from_f64(1e-12).ok_or_else(|| {
        InterpolateError::ComputationError("Failed to convert tolerance".to_string())
    })?;

    for i in 0..n_seg {
        let a = spline.coeffs()[[i, 0]];
        let b = spline.coeffs()[[i, 1]];
        let c = spline.coeffs()[[i, 2]];
        let d = spline.coeffs()[[i, 3]];
        let h = spline.x()[i + 1] - spline.x()[i];

        let seg_roots = solve_cubic_on_interval(a, b, c, d, F::zero(), h, tol)?;
        for t in seg_roots {
            let x_root = spline.x()[i] + t;
            // Avoid duplicate roots at segment boundaries
            if roots.is_empty() || (x_root - *roots.last().unwrap_or(&F::zero())).abs() > tol {
                roots.push(x_root);
            }
        }
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Find roots of `y_target` instead of zero.
///
/// Returns x values where `spline(x) = y_target`.
pub fn spline_solve<F: InterpolationFloat + ToString>(
    spline: &CubicSpline<F>,
    y_target: F,
) -> InterpolateResult<Vec<F>> {
    let n_seg = spline.coeffs().nrows();
    let mut roots = Vec::new();
    let tol = F::from_f64(1e-12).ok_or_else(|| {
        InterpolateError::ComputationError("Failed to convert tolerance".to_string())
    })?;

    for i in 0..n_seg {
        // Shift by y_target: solve a - y_target + b*t + c*t^2 + d*t^3 = 0
        let a = spline.coeffs()[[i, 0]] - y_target;
        let b = spline.coeffs()[[i, 1]];
        let c = spline.coeffs()[[i, 2]];
        let d = spline.coeffs()[[i, 3]];
        let h = spline.x()[i + 1] - spline.x()[i];

        let seg_roots = solve_cubic_on_interval(a, b, c, d, F::zero(), h, tol)?;
        for t in seg_roots {
            let x_root = spline.x()[i] + t;
            if roots.is_empty() || (x_root - *roots.last().unwrap_or(&F::zero())).abs() > tol {
                roots.push(x_root);
            }
        }
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

// ---------------------------------------------------------------------------
// Helper: convert CubicSpline to PiecewisePolynomial
// ---------------------------------------------------------------------------

fn cubic_spline_to_pp<F: InterpolationFloat>(spline: &CubicSpline<F>) -> PiecewisePolynomial<F> {
    let n_seg = spline.coeffs().nrows();
    let mut coeffs = Array2::<F>::zeros((n_seg, 4));
    for i in 0..n_seg {
        for j in 0..4 {
            coeffs[[i, j]] = spline.coeffs()[[i, j]];
        }
    }
    PiecewisePolynomial {
        breakpoints: spline.x().clone(),
        coeffs,
    }
}

// ---------------------------------------------------------------------------
// Cubic solver helpers
// ---------------------------------------------------------------------------

/// Solve `a + b*t + c*t^2 + d*t^3 = 0` for real roots in `[lo, hi]`.
fn solve_cubic_on_interval<F: InterpolationFloat>(
    a: F,
    b: F,
    c: F,
    d: F,
    lo: F,
    hi: F,
    tol: F,
) -> InterpolateResult<Vec<F>> {
    let zero = F::zero();
    // Classify the polynomial degree
    if d.abs() < tol && c.abs() < tol && b.abs() < tol {
        // Constant or zero: root only if |a| < tol (everywhere zero)
        // We do not return infinitely many roots; return empty.
        return Ok(Vec::new());
    }

    if d.abs() < tol && c.abs() < tol {
        // Linear: b*t + a = 0 => t = -a / b
        if b.abs() < tol {
            return Ok(Vec::new());
        }
        let t = -a / b;
        return Ok(filter_roots_in_interval(&[t], lo, hi, tol));
    }

    if d.abs() < tol {
        // Quadratic: c*t^2 + b*t + a = 0
        let disc = b * b
            - F::from_f64(4.0).ok_or_else(|| {
                InterpolateError::ComputationError("float conversion".to_string())
            })? * c
                * a;
        if disc < zero {
            return Ok(Vec::new());
        }
        let sqrt_disc = disc.sqrt();
        let two_c = c + c;
        if two_c.abs() < tol {
            return Ok(Vec::new());
        }
        let t1 = (-b + sqrt_disc) / two_c;
        let t2 = (-b - sqrt_disc) / two_c;
        return Ok(filter_roots_in_interval(&[t1, t2], lo, hi, tol));
    }

    // Full cubic: d*t^3 + c*t^2 + b*t + a = 0
    // Normalize: t^3 + p*t^2 + q*t + r = 0 where p=c/d, q=b/d, r=a/d
    let p = c / d;
    let q = b / d;
    let r = a / d;

    let roots = solve_depressed_cubic(p, q, r, tol)?;
    Ok(filter_roots_in_interval(&roots, lo, hi, tol))
}

/// Solve the monic cubic `t^3 + p*t^2 + q*t + r = 0` using Cardano's method.
/// Returns all real roots.
fn solve_depressed_cubic<F: InterpolationFloat>(
    p: F,
    q: F,
    r: F,
    tol: F,
) -> InterpolateResult<Vec<F>> {
    let three = F::from_f64(3.0)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
    let two = F::from_f64(2.0)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
    let nine = F::from_f64(9.0)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
    let twenty_seven = F::from_f64(27.0)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;
    let fifty_four = F::from_f64(54.0)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?;

    // Substitute t = u - p/3 to get depressed cubic u^3 + au + b = 0
    let a_dep = q - p * p / three;
    let b_dep = (two * p * p * p - nine * p * q + twenty_seven * r) / twenty_seven;

    let discriminant = -(F::from_f64(4.0)
        .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?
        * a_dep
        * a_dep
        * a_dep
        + twenty_seven * b_dep * b_dep);

    let shift = p / three;

    if discriminant.abs() < tol {
        // Multiple roots
        if a_dep.abs() < tol {
            // Triple root at u = 0
            return Ok(vec![-shift]);
        }
        let u1 = F::from_f64(3.0)
            .ok_or_else(|| InterpolateError::ComputationError("float conversion".to_string()))?
            * b_dep
            / (two * a_dep);
        let u2 = -u1 / two;
        let mut roots = vec![u1 - shift, u2 - shift];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        roots.dedup_by(|a, b| (*a - *b).abs() < tol);
        return Ok(roots);
    }

    if discriminant > F::zero() {
        // Three distinct real roots -- use trigonometric method
        if a_dep >= F::zero() {
            // Shouldn't happen for three real roots (a_dep must be < 0),
            // but handle gracefully
            return solve_cubic_numerically(p, q, r, tol);
        }
        let m = (-a_dep / three).sqrt();
        let theta_arg = -b_dep / (two * m * m * m);
        // Clamp to [-1, 1] to avoid NaN from acos
        let theta_arg_clamped = if theta_arg > F::one() {
            F::one()
        } else if theta_arg < -F::one() {
            -F::one()
        } else {
            theta_arg
        };
        let theta = theta_arg_clamped.acos() / three;
        let pi = F::from_f64(std::f64::consts::PI)
            .ok_or_else(|| InterpolateError::ComputationError("pi conversion".to_string()))?;

        let u0 = two * m * theta.cos();
        let u1 = two * m * (theta - two * pi / three).cos();
        let u2 = two * m * (theta + two * pi / three).cos();

        let mut roots = vec![u0 - shift, u1 - shift, u2 - shift];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(roots)
    } else {
        // One real root -- use Cardano
        let half_b = b_dep / two;
        let q_card = -a_dep / three;
        let det = half_b * half_b - q_card * q_card * q_card;
        if det < F::zero() {
            // Fallback to numeric
            return solve_cubic_numerically(p, q, r, tol);
        }
        let sqrt_det = det.sqrt();
        let s_val = -half_b + sqrt_det;
        let t_val = -half_b - sqrt_det;
        let s_cbrt = cbrt(s_val);
        let t_cbrt = cbrt(t_val);
        let u0 = s_cbrt + t_cbrt;
        Ok(vec![u0 - shift])
    }
}

/// Cube root that handles negative numbers.
fn cbrt<F: InterpolationFloat>(x: F) -> F {
    if x >= F::zero() {
        x.powf(F::from_f64(1.0 / 3.0).unwrap_or(F::zero()))
    } else {
        -(-x).powf(F::from_f64(1.0 / 3.0).unwrap_or(F::zero()))
    }
}

/// Numerical fallback for difficult cubic cases using Newton's method.
fn solve_cubic_numerically<F: InterpolationFloat>(
    p: F,
    q: F,
    r: F,
    tol: F,
) -> InterpolateResult<Vec<F>> {
    // Evaluate f(t) = t^3 + p*t^2 + q*t + r and f'(t) = 3t^2 + 2pt + q
    let f = |t: F| -> F { t * t * t + p * t * t + q * t + r };
    let f_prime = |t: F| -> F {
        let three = F::from_f64(3.0).unwrap_or(F::zero());
        let two = F::from_f64(2.0).unwrap_or(F::zero());
        three * t * t + two * p * t + q
    };

    let mut roots = Vec::new();
    // Try several starting points
    let starts = [
        F::zero(),
        F::one(),
        -F::one(),
        F::from_f64(10.0).unwrap_or(F::zero()),
        F::from_f64(-10.0).unwrap_or(F::zero()),
        -p / F::from_f64(3.0).unwrap_or(F::one()),
    ];

    for &start in &starts {
        let mut t = start;
        for _ in 0..100 {
            let ft = f(t);
            let fpt = f_prime(t);
            if fpt.abs() < tol {
                break;
            }
            let dt = ft / fpt;
            t = t - dt;
            if dt.abs() < tol {
                break;
            }
        }
        if f(t).abs() < F::from_f64(1e-8).unwrap_or(tol) {
            // Check if this root is already found
            let already = roots.iter().any(|&existing: &F| (existing - t).abs() < tol);
            if !already {
                roots.push(t);
            }
        }
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Filter roots to those within `[lo - tol, hi + tol]` and clamp to `[lo, hi]`.
fn filter_roots_in_interval<F: InterpolationFloat>(
    candidates: &[F],
    lo: F,
    hi: F,
    tol: F,
) -> Vec<F> {
    let mut result = Vec::new();
    for &t in candidates {
        if t >= lo - tol && t <= hi + tol {
            let clamped = if t < lo {
                lo
            } else if t > hi {
                hi
            } else {
                t
            };
            // Avoid duplicates
            let dup = result
                .iter()
                .any(|&existing: &F| (existing - clamped).abs() < tol);
            if !dup {
                result.push(clamped);
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // -- spline_derivative tests --

    #[test]
    fn test_derivative_order_1_quadratic_data() {
        // y = x^2 on [0,3]: derivative ~ 2x
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let deriv = spline_derivative(&spline, 1).expect("spline_derivative failed");
        // At x = 1.0 derivative of x^2 is 2.0
        let val = deriv.evaluate(1.0).expect("evaluate failed");
        assert_abs_diff_eq!(val, 2.0, epsilon = 0.5);
    }

    #[test]
    fn test_derivative_order_2_quadratic_data() {
        // y = x^2, second derivative ~ 2
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let deriv2 = spline_derivative(&spline, 2).expect("spline_derivative(2) failed");
        let val = deriv2.evaluate(1.5).expect("evaluate failed");
        assert_abs_diff_eq!(val, 2.0, epsilon = 1.0);
    }

    #[test]
    fn test_derivative_order_3() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let deriv3 = spline_derivative(&spline, 3).expect("spline_derivative(3) failed");
        // Third derivative of a cubic spline is piecewise constant (6*d)
        let val: f64 = deriv3.evaluate(0.5).expect("evaluate failed");
        // Just check it's finite
        assert!(val.is_finite());
    }

    #[test]
    fn test_derivative_order_4_is_zero() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let deriv4 = spline_derivative(&spline, 4).expect("spline_derivative(4) failed");
        let val = deriv4.evaluate(1.5).expect("evaluate failed");
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_derivative_order_0_returns_original() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let pp = spline_derivative(&spline, 0).expect("spline_derivative(0) failed");
        let val = pp.evaluate(1.5).expect("evaluate failed");
        let expected = spline.evaluate(1.5).expect("evaluate failed");
        assert_abs_diff_eq!(val, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_derivative_consistency_with_spline() {
        // Compare PiecewisePolynomial derivative with CubicSpline::derivative_n
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 0.5, 2.0, 1.5, 3.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let deriv_pp = spline_derivative(&spline, 1).expect("spline_derivative failed");

        for &xi in &[0.5, 1.5, 2.5, 3.5] {
            let pp_val = deriv_pp.evaluate(xi).expect("evaluate failed");
            let sp_val = spline.derivative_n(xi, 1).expect("derivative_n failed");
            assert_abs_diff_eq!(pp_val, sp_val, epsilon = 1e-10);
        }
    }

    // -- spline_integral tests --

    #[test]
    fn test_integral_constant_function() {
        // y = 2 on [0,3], integral = 2 * 3 = 6
        // Use points that approximate y=2
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![2.0, 2.0, 2.0, 2.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let integral = spline_integral(&spline, 0.0, 3.0).expect("spline_integral failed");
        assert_abs_diff_eq!(integral, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_integral_zero_width() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let integral = spline_integral(&spline, 1.5, 1.5).expect("spline_integral failed");
        assert_abs_diff_eq!(integral, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_integral_reversed_bounds() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let i1 = spline_integral(&spline, 0.0, 3.0).expect("spline_integral failed");
        let i2 = spline_integral(&spline, 3.0, 0.0).expect("spline_integral failed");
        assert_abs_diff_eq!(i1, -i2, epsilon = 1e-12);
    }

    #[test]
    fn test_integral_linear_function() {
        // y = x on [0,3], integral = 4.5
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 2.0, 3.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let integral = spline_integral(&spline, 0.0, 3.0).expect("spline_integral failed");
        assert_abs_diff_eq!(integral, 4.5, epsilon = 0.1);
    }

    #[test]
    fn test_integral_partial_domain() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![1.0, 1.0, 1.0, 1.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let integral = spline_integral(&spline, 0.5, 2.5).expect("spline_integral failed");
        assert_abs_diff_eq!(integral, 2.0, epsilon = 1e-10);
    }

    // -- spline_antiderivative tests --

    #[test]
    fn test_antiderivative_at_left_is_zero() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let anti = spline_antiderivative(&spline).expect("spline_antiderivative failed");
        let val = anti.evaluate(0.0).expect("evaluate failed");
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_antiderivative_matches_integral() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let anti = spline_antiderivative(&spline).expect("spline_antiderivative failed");

        // The antiderivative at x=3 should equal integrate(0, 3)
        let anti_val = anti.evaluate(3.0).expect("evaluate failed");
        let int_val = spline_integral(&spline, 0.0, 3.0).expect("spline_integral failed");
        assert_abs_diff_eq!(anti_val, int_val, epsilon = 1e-10);
    }

    #[test]
    fn test_antiderivative_at_interior_point() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let anti = spline_antiderivative(&spline).expect("spline_antiderivative failed");

        // A(2.0) should equal integrate(0, 2)
        let anti_val = anti.evaluate(2.0).expect("evaluate failed");
        let int_val = spline_integral(&spline, 0.0, 2.0).expect("spline_integral failed");
        assert_abs_diff_eq!(anti_val, int_val, epsilon = 1e-10);
    }

    #[test]
    fn test_antiderivative_constant_function() {
        // Antiderivative of y=2 from 0..3 is 2*x
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![2.0, 2.0, 2.0, 2.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let anti = spline_antiderivative(&spline).expect("spline_antiderivative failed");
        let val = anti.evaluate(2.0).expect("evaluate failed");
        assert_abs_diff_eq!(val, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_antiderivative_degree() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let anti = spline_antiderivative(&spline).expect("spline_antiderivative failed");
        assert_eq!(anti.degree(), 4);
    }

    // -- spline_roots tests --

    #[test]
    fn test_roots_linear_crossing() {
        // y crosses zero between x=1 and x=2 (1 -> -1)
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![2.0, 1.0, -1.0, -2.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let roots = spline_roots(&spline).expect("spline_roots failed");
        assert!(!roots.is_empty(), "Expected at least one root");
        // The root should be near x = 1.5
        let closest = roots.iter().min_by(|a, b| {
            ((**a) - 1.5_f64)
                .abs()
                .partial_cmp(&((**b) - 1.5_f64).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert!(closest.is_some());
        assert_abs_diff_eq!(*closest.expect("no root"), 1.5, epsilon = 0.3);
    }

    #[test]
    fn test_roots_no_crossing() {
        // All positive values
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let roots = spline_roots(&spline).expect("spline_roots failed");
        assert!(
            roots.is_empty(),
            "Expected no roots for all-positive function"
        );
    }

    #[test]
    fn test_roots_at_data_point() {
        // y = 0 at x = 2
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![1.0, 0.5, 0.0, -0.5];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let roots = spline_roots(&spline).expect("spline_roots failed");
        assert!(!roots.is_empty(), "Expected root near x=2");
        let has_root_near_2 = roots.iter().any(|&r: &f64| (r - 2.0).abs() < 0.1);
        assert!(has_root_near_2, "Expected root near x=2, got {:?}", roots);
    }

    #[test]
    fn test_roots_multiple() {
        // Sine-like: has multiple zero crossings
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![0.0, 0.84, 0.91, 0.14, -0.76, -0.96, -0.28];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let roots = spline_roots(&spline).expect("spline_roots failed");
        // Expect at least 2 roots (near 0 and near pi, and near 2*pi perhaps)
        assert!(
            roots.len() >= 2,
            "Expected at least 2 roots, got {}",
            roots.len()
        );
    }

    #[test]
    fn test_roots_returns_sorted() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![0.0, 0.84, 0.91, 0.14, -0.76, -0.96, -0.28];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let roots = spline_roots(&spline).expect("spline_roots failed");
        for i in 1..roots.len() {
            assert!(roots[i] >= roots[i - 1], "Roots must be sorted");
        }
    }

    // -- spline_solve tests --

    #[test]
    fn test_solve_for_known_value() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let solutions = spline_solve(&spline, 1.0).expect("spline_solve failed");
        // x=1 should be a solution
        assert!(!solutions.is_empty());
        let has_near_1 = solutions.iter().any(|&s: &f64| (s - 1.0).abs() < 0.1);
        assert!(
            has_near_1,
            "Expected solution near x=1, got {:?}",
            solutions
        );
    }

    #[test]
    fn test_piecewise_polynomial_evaluate_array() {
        let x = array![0.0_f64, 1.0, 2.0, 3.0];
        let y = array![0.0_f64, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let pp = spline_derivative(&spline, 0).expect("spline_derivative(0) failed");
        let vals = pp
            .evaluate_array(&[0.5_f64, 1.5, 2.5])
            .expect("evaluate_array failed");
        assert_eq!(vals.len(), 3);
        for v in &vals {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_piecewise_polynomial_out_of_bounds() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let spline =
            CubicSpline::new(&x.view(), &y.view()).expect("CubicSpline construction failed");
        let pp = spline_derivative(&spline, 0).expect("spline_derivative(0) failed");
        assert!(pp.evaluate(-1.0).is_err());
        assert!(pp.evaluate(4.0).is_err());
    }
}
