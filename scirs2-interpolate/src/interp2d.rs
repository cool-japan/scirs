//! 2D interpolation - SciPy-compatible interp2d implementation
//!
//! This module provides 2D interpolation functionality compatible with
//! SciPy's interp2d function for interpolating data on regular grids.

use crate::error::{InterpolateError, InterpolateResult};
use crate::interp1d::linear_interpolate;
use crate::numerical_stability::solve_with_stability_monitoring;
use crate::spline::CubicSpline;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// 2D interpolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interp2dKind {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation using splines
    Cubic,
    /// Quintic interpolation: a natural, C4-continuous, degree-5 piecewise
    /// polynomial spline (the direct higher-order analogue of `Cubic`'s
    /// natural cubic spline), applied separably along each axis.
    Quintic,
}

/// 2D interpolator for data on regular grids
///
/// This struct provides functionality similar to SciPy's interp2d for
/// interpolating 2D data defined on regular grids.
#[derive(Debug, Clone)]
pub struct Interp2d<F> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates (must be sorted)
    y: Array1<F>,
    /// Z values with shape (len(y), len(x))
    z: Array2<F>,
    /// Interpolation method
    kind: Interp2dKind,
}

impl<F> Interp2d<F>
where
    F: Float + FromPrimitive + Debug + Clone + crate::traits::InterpolationFloat,
{
    /// Create a new 2D interpolator
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinates (must be sorted), length n_x
    /// * `y` - Y coordinates (must be sorted), length n_y  
    /// * `z` - Z values with shape (n_y, n_x)
    /// * `kind` - Interpolation method
    ///
    /// # Returns
    ///
    /// New 2D interpolator
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - If z.shape() != (y.len(), x.len())
    /// * `InvalidInput` - If x or y are not sorted
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::{array, Array2};
    /// use scirs2_interpolate::interp2d::{Interp2d, Interp2dKind};
    ///
    /// // Define grid
    /// let x = array![0.0, 1.0, 2.0];
    /// let y = array![0.0, 1.0];
    ///
    /// // Define function z = x + y on the grid
    /// let z = Array2::from_shape_fn((2, 3), |(i, j)| {
    ///     y[i] + x[j]
    /// });
    ///
    /// let interp = Interp2d::new(&x.view(), &y.view(), &z.view(),
    ///                           Interp2dKind::Linear)?;
    ///
    /// // Interpolate at a point
    /// let result = interp.evaluate(0.5, 0.5)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        z: &ArrayView2<F>,
        kind: Interp2dKind,
    ) -> InterpolateResult<Self> {
        // Validate shapes
        if z.nrows() != y.len() || z.ncols() != x.len() {
            return Err(InterpolateError::shape_mismatch(
                format!("({}, {})", y.len(), x.len()),
                format!("({}, {})", z.nrows(), z.ncols()),
                "interp2d z array shape",
            ));
        }

        // Check that x and y are sorted
        if !is_sorted(x) {
            return Err(InterpolateError::invalid_input(
                "x coordinates must be sorted in ascending order",
            ));
        }

        if !is_sorted(y) {
            return Err(InterpolateError::invalid_input(
                "y coordinates must be sorted in ascending order",
            ));
        }

        // Check for minimum grid size
        if x.len() < 2 || y.len() < 2 {
            return Err(InterpolateError::invalid_input(
                "need at least 2 points in each dimension",
            ));
        }

        Ok(Self {
            x: x.to_owned(),
            y: y.to_owned(),
            z: z.to_owned(),
            kind,
        })
    }

    /// Evaluate the interpolator at a single point
    ///
    /// # Arguments
    ///
    /// * `x_new` - X coordinate for evaluation
    /// * `ynew` - Y coordinate for evaluation
    ///
    /// # Returns
    ///
    /// Interpolated value at (x_new, ynew)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::{array, Array2};
    /// use scirs2_interpolate::interp2d::{Interp2d, Interp2dKind};
    ///
    /// let x = array![0.0, 1.0, 2.0];
    /// let y = array![0.0, 1.0];
    /// let z = Array2::from_shape_fn((2, 3), |(i, j)| {
    ///     y[i] + x[j] // z = x + y
    /// });
    ///
    /// let interp = Interp2d::new(&x.view(), &y.view(), &z.view(),
    ///                           Interp2dKind::Linear)?;
    ///
    /// let result = interp.evaluate(0.5, 0.5)?;
    /// // Should be approximately 1.0 (0.5 + 0.5)
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn evaluate(&self, x_new: F, ynew: F) -> InterpolateResult<F> {
        match self.kind {
            Interp2dKind::Linear => self.evaluate_linear(x_new, ynew),
            Interp2dKind::Cubic => self.evaluate_cubic(x_new, ynew),
            Interp2dKind::Quintic => self.evaluate_quintic(x_new, ynew),
        }
    }

    /// Evaluate at multiple points
    ///
    /// # Arguments
    ///
    /// * `x_new` - X coordinates for evaluation
    /// * `ynew` - Y coordinates for evaluation (must have same length as x_new)
    ///
    /// # Returns
    ///
    /// Array of interpolated values
    pub fn evaluate_array(
        &self,
        x_new: &ArrayView1<F>,
        ynew: &ArrayView1<F>,
    ) -> InterpolateResult<Array1<F>> {
        if x_new.len() != ynew.len() {
            return Err(InterpolateError::shape_mismatch(
                format!("x_new.len() = {}", x_new.len()),
                format!("ynew.len() = {}", ynew.len()),
                "interp2d coordinate arrays",
            ));
        }

        let mut result = Array1::zeros(x_new.len());
        for i in 0..x_new.len() {
            result[i] = self.evaluate(x_new[i], ynew[i])?;
        }
        Ok(result)
    }

    /// Evaluate on a regular grid
    ///
    /// # Arguments
    ///
    /// * `x_new` - X coordinates for output grid
    /// * `ynew` - Y coordinates for output grid
    ///
    /// # Returns
    ///
    /// 2D array with shape (len(ynew), len(x_new))
    pub fn evaluate_grid(
        &self,
        x_new: &ArrayView1<F>,
        ynew: &ArrayView1<F>,
    ) -> InterpolateResult<Array2<F>> {
        let mut result = Array2::zeros((ynew.len(), x_new.len()));

        for (i, &y_val) in ynew.iter().enumerate() {
            for (j, &x_val) in x_new.iter().enumerate() {
                result[[i, j]] = self.evaluate(x_val, y_val)?;
            }
        }

        Ok(result)
    }

    /// Linear interpolation implementation
    fn evaluate_linear(&self, x_new: F, ynew: F) -> InterpolateResult<F> {
        // Find y index and interpolate along x for neighboring y values
        let y_idx = find_interval(&self.y.view(), ynew);

        let result = if y_idx == 0 && ynew < self.y[0] {
            // Extrapolate below
            let row = self.z.slice(scirs2_core::ndarray::s![0, ..]);
            linear_interpolate(&self.x.view(), &row, &Array1::from_vec(vec![x_new]).view())?[0]
        } else if y_idx >= self.y.len() - 1 && ynew > self.y[self.y.len() - 1] {
            // Extrapolate above
            let row = self.z.slice(scirs2_core::ndarray::s![self.y.len() - 1, ..]);
            linear_interpolate(&self.x.view(), &row, &Array1::from_vec(vec![x_new]).view())?[0]
        } else {
            // Interpolate between two y values
            let y_idx = y_idx.min(self.y.len() - 2);

            // Interpolate along x for both y levels
            let row0 = self.z.slice(scirs2_core::ndarray::s![y_idx, ..]);
            let row1 = self.z.slice(scirs2_core::ndarray::s![y_idx + 1, ..]);

            let val0 =
                linear_interpolate(&self.x.view(), &row0, &Array1::from_vec(vec![x_new]).view())?
                    [0];
            let val1 =
                linear_interpolate(&self.x.view(), &row1, &Array1::from_vec(vec![x_new]).view())?
                    [0];

            // Interpolate along y
            let y0 = self.y[y_idx];
            let y1 = self.y[y_idx + 1];

            if (y1 - y0).abs() < F::epsilon() {
                val0
            } else {
                let t = (ynew - y0) / (y1 - y0);
                val0 + t * (val1 - val0)
            }
        };

        Ok(result)
    }

    /// Cubic interpolation implementation
    fn evaluate_cubic(&self, x_new: F, ynew: F) -> InterpolateResult<F> {
        // Create cubic splines for each x value across y
        let mut values_at_x = Array1::zeros(self.y.len());

        for (i, &_y_val) in self.y.iter().enumerate() {
            let row = self.z.slice(scirs2_core::ndarray::s![i, ..]);
            let spline = CubicSpline::new(&self.x.view(), &row)?;
            values_at_x[i] = spline.evaluate(x_new)?;
        }

        // Create cubic spline along y direction
        let y_spline = CubicSpline::new(&self.y.view(), &values_at_x.view())?;
        y_spline.evaluate(ynew)
    }

    /// Quintic interpolation implementation
    ///
    /// Mirrors [`Self::evaluate_cubic`]'s separable (tensor-product)
    /// construction: a 1D quintic spline is built along `x` for every `y`
    /// row to get the value at `x_new` on each row, and a second 1D quintic
    /// spline is then built along `y` through those values and evaluated at
    /// `ynew`. Each 1D quintic spline is a true C4-continuous, degree-5
    /// piecewise polynomial (see [`QuinticSpline1D`]), not a cubic spline in
    /// disguise.
    fn evaluate_quintic(&self, x_new: F, ynew: F) -> InterpolateResult<F> {
        let n_x = self.x.len();
        let n_y = self.y.len();

        if n_x < 3 || n_y < 3 {
            return Err(InterpolateError::invalid_input(
                "quintic interpolation requires at least 3 points in each dimension",
            ));
        }

        // Build a quintic spline along x for each y row and evaluate at x_new.
        let mut values_at_x = Array1::zeros(n_y);
        for i in 0..n_y {
            let row = self.z.slice(scirs2_core::ndarray::s![i, ..]);
            let spline = QuinticSpline1D::new(&self.x.view(), &row)?;
            values_at_x[i] = spline.evaluate(x_new);
        }

        // Build a quintic spline along y through those values and evaluate at ynew.
        let y_spline = QuinticSpline1D::new(&self.y.view(), &values_at_x.view())?;
        Ok(y_spline.evaluate(ynew))
    }
}

/// Build a small non-negative integer constant for a generic float type
/// without any fallible conversion: `FromPrimitive::from_u32` is tried
/// first, falling back to repeated addition of `F::one()` (which can never
/// fail) if that conversion is somehow unavailable.
fn small_const<F: Float + FromPrimitive>(value: u32) -> F {
    F::from_u32(value).unwrap_or_else(|| {
        let mut acc = F::zero();
        for _ in 0..value {
            acc = acc + F::one();
        }
        acc
    })
}

/// A single degree-5 polynomial segment of a [`QuinticSpline1D`], expressed
/// in the local variable `t = x - x_i` and valid on `t in [0, h_i]`.
#[derive(Debug, Clone)]
struct QuinticSegment<F> {
    /// Coefficients `[a0, a1, a2, a3, a4, a5]` such that
    /// `p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5`.
    coeffs: [F; 6],
}

impl<F: Float> QuinticSegment<F> {
    /// Evaluate the segment polynomial at local coordinate `t` via Horner's
    /// method.
    fn evaluate(&self, t: F) -> F {
        let mut result = self.coeffs[5];
        for k in (0..5).rev() {
            result = result * t + self.coeffs[k];
        }
        result
    }

    /// Build the segment from Hermite-style endpoint data: values `y0`,
    /// `y1`, first derivatives `m0`, `m1`, and second derivatives `mm0`,
    /// `mm1` at the two ends of an interval of width `h`.
    #[allow(clippy::too_many_arguments)]
    fn from_hermite_quintic(y0: F, y1: F, m0: F, m1: F, mm0: F, mm1: F, h: F) -> Self
    where
        F: FromPrimitive,
    {
        let two = small_const::<F>(2);
        let three = small_const::<F>(3);
        let six = small_const::<F>(6);
        let seven = small_const::<F>(7);
        let eight = small_const::<F>(8);
        let twelve = small_const::<F>(12);
        let fifteen = small_const::<F>(15);
        let twenty = small_const::<F>(20);

        let h2 = h * h;
        let h3 = h2 * h;
        let h4 = h2 * h2;
        let h5 = h4 * h;

        let a0 = y0;
        let a1 = m0;
        let a2 = mm0 / two;
        let a3 = (-three * mm0 * h2 + mm1 * h2 - twelve * h * m0 - eight * h * m1 - twenty * y0
            + twenty * y1)
            / (two * h3);
        let a4 =
            (three / two * mm0 * h2 - mm1 * h2 + eight * h * m0 + seven * h * m1 + fifteen * y0
                - fifteen * y1)
                / h4;
        let a5 = (-mm0 * h2 + mm1 * h2 - six * h * m0 - six * h * m1 - twelve * y0 + twelve * y1)
            / (two * h5);

        Self {
            coeffs: [a0, a1, a2, a3, a4, a5],
        }
    }
}

/// A natural quintic spline: a C4-continuous, degree-5 piecewise polynomial
/// interpolant through 1D data.
///
/// This is the direct higher-order generalization of the classical
/// "natural" cubic spline (which enforces continuity of the function value,
/// first, and second derivatives, and sets the second derivative to zero at
/// the two endpoints). Here, continuity is additionally enforced for the
/// third and fourth derivatives at every interior knot, and the "natural"
/// boundary condition sets the third *and* fourth derivatives to zero at
/// the two endpoints (the two missing degrees of freedom needed to close
/// the system).
///
/// Internally this is built by solving a single linear system for the
/// first and second derivatives at every knot (`m_i`, `M_i`), then
/// constructing each segment as a quintic Hermite polynomial matching
/// `y`, `m`, and `M` at both of its endpoints -- which automatically
/// guarantees continuity of the function value and its first two
/// derivatives, while the linear system enforces continuity of the third
/// and fourth derivatives as well.
struct QuinticSpline1D<F> {
    x: Array1<F>,
    segments: Vec<QuinticSegment<F>>,
}

impl<F> QuinticSpline1D<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Clone
        + 'static,
{
    fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        let n = x.len();
        if n != y.len() {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("{n} elements"),
                actual: format!("{} elements", y.len()),
                object: "quintic spline y values".to_string(),
            });
        }
        if n < 3 {
            return Err(InterpolateError::invalid_input(
                "quintic spline construction requires at least 3 points",
            ));
        }

        let h: Vec<F> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        for (i, &hi) in h.iter().enumerate() {
            if hi <= F::zero() {
                return Err(InterpolateError::invalid_input(format!(
                    "quintic spline requires strictly increasing x values \
                     (non-increasing step between indices {i} and {})",
                    i + 1
                )));
            }
        }

        let two = small_const::<F>(2);
        let three = small_const::<F>(3);
        let eight = small_const::<F>(8);
        let twelve = small_const::<F>(12);
        let fourteen = small_const::<F>(14);
        let sixteen = small_const::<F>(16);
        let twenty = small_const::<F>(20);
        let thirty = small_const::<F>(30);

        // Unknowns, in order: [m_0, M_0, m_1, M_1, ..., m_{n-1}, M_{n-1}]
        // (first and second derivatives at every knot).
        let dim = 2 * n;
        let mut a = Array2::<F>::zeros((dim, dim));
        let mut rhs = Array1::<F>::zeros(dim);
        let idx_m = |i: usize| 2 * i;
        let idx_mm = |i: usize| 2 * i + 1;

        let mut row = 0usize;

        // Left natural boundary condition: third and fourth derivatives of
        // the first segment vanish at its left end (t = 0).
        {
            let h0 = h[0];
            let h0_2 = h0 * h0;

            a[(row, idx_m(0))] = -twelve * h0;
            a[(row, idx_m(1))] = -eight * h0;
            a[(row, idx_mm(0))] = -three * h0_2;
            a[(row, idx_mm(1))] = h0_2;
            rhs[row] = twenty * (y[0] - y[1]);
            row += 1;

            a[(row, idx_m(0))] = sixteen * h0;
            a[(row, idx_m(1))] = fourteen * h0;
            a[(row, idx_mm(0))] = three * h0_2;
            a[(row, idx_mm(1))] = -two * h0_2;
            rhs[row] = -thirty * (y[0] - y[1]);
            row += 1;
        }

        // Interior continuity: third and fourth derivative continuity at
        // every interior knot i = 1 ..= n-2, linking segment (i-1) and
        // segment i.
        for i in 1..n - 1 {
            let h_prev = h[i - 1];
            let h_next = h[i];
            let a_coef = F::one() / (h_prev * h_prev * h_prev);
            let b_coef = F::one() / (h_next * h_next * h_next);
            let c_coef = a_coef / h_prev;
            let d_coef = b_coef / h_next;

            // Third-derivative continuity.
            a[(row, idx_m(i - 1))] += -eight * a_coef * h_prev;
            a[(row, idx_m(i))] += -twelve * a_coef * h_prev + twelve * b_coef * h_next;
            a[(row, idx_m(i + 1))] += eight * b_coef * h_next;
            a[(row, idx_mm(i - 1))] += -a_coef * h_prev * h_prev;
            a[(row, idx_mm(i))] +=
                three * a_coef * h_prev * h_prev + three * b_coef * h_next * h_next;
            a[(row, idx_mm(i + 1))] += -b_coef * h_next * h_next;
            rhs[row] = twenty * a_coef * y[i - 1] - twenty * (a_coef + b_coef) * y[i]
                + twenty * b_coef * y[i + 1];
            row += 1;

            // Fourth-derivative continuity.
            a[(row, idx_m(i - 1))] += -fourteen * c_coef * h_prev;
            a[(row, idx_m(i))] += -sixteen * c_coef * h_prev - sixteen * d_coef * h_next;
            a[(row, idx_m(i + 1))] += -fourteen * d_coef * h_next;
            a[(row, idx_mm(i - 1))] += -two * c_coef * h_prev * h_prev;
            a[(row, idx_mm(i))] +=
                three * c_coef * h_prev * h_prev - three * d_coef * h_next * h_next;
            a[(row, idx_mm(i + 1))] += two * d_coef * h_next * h_next;
            rhs[row] = thirty * c_coef * (y[i - 1] - y[i]) + thirty * d_coef * (y[i] - y[i + 1]);
            row += 1;
        }

        // Right natural boundary condition: third and fourth derivatives of
        // the last segment vanish at its right end (t = h_last).
        {
            let h_last = h[n - 2];
            let h_last_2 = h_last * h_last;

            a[(row, idx_m(n - 2))] = -eight * h_last;
            a[(row, idx_m(n - 1))] = -twelve * h_last;
            a[(row, idx_mm(n - 2))] = -h_last_2;
            a[(row, idx_mm(n - 1))] = three * h_last_2;
            rhs[row] = twenty * (y[n - 2] - y[n - 1]);
            row += 1;

            a[(row, idx_m(n - 2))] = -fourteen * h_last;
            a[(row, idx_m(n - 1))] = -sixteen * h_last;
            a[(row, idx_mm(n - 2))] = -two * h_last_2;
            a[(row, idx_mm(n - 1))] = three * h_last_2;
            rhs[row] = thirty * (y[n - 2] - y[n - 1]);
            row += 1;
        }

        debug_assert_eq!(row, dim);

        let solution = solve_with_stability_monitoring(&a.view(), &rhs.view()).map_err(|e| {
            InterpolateError::NumericalInstability {
                message: format!("failed to solve quintic spline derivative system: {e}"),
            }
        })?;

        let mut segments = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            segments.push(QuinticSegment::from_hermite_quintic(
                y[i],
                y[i + 1],
                solution[idx_m(i)],
                solution[idx_m(i + 1)],
                solution[idx_mm(i)],
                solution[idx_mm(i + 1)],
                h[i],
            ));
        }

        Ok(Self {
            x: x.to_owned(),
            segments,
        })
    }

    /// Evaluate the spline at `x_new`, clamping to the nearest valid segment
    /// (and thus extrapolating via that segment's polynomial) if `x_new`
    /// falls outside `[x[0], x[n-1]]`.
    fn evaluate(&self, x_new: F) -> F {
        let n = self.x.len();
        let idx = find_interval(&self.x.view(), x_new).min(n - 2);
        let t = x_new - self.x[idx];
        self.segments[idx].evaluate(t)
    }
}

/// Check if array is sorted in ascending order
#[allow(dead_code)]
fn is_sorted<F: PartialOrd>(arr: &ArrayView1<F>) -> bool {
    for window in arr.windows(2) {
        if window[0] > window[1] {
            return false;
        }
    }
    true
}

/// Find interval containing the value using binary search
#[allow(dead_code)]
fn find_interval<F: PartialOrd>(arr: &ArrayView1<F>, value: F) -> usize {
    // Convert to slice to use binary_search_by
    let slice: &[F] = arr.as_slice().expect("Operation failed");
    match slice.binary_search_by(|x| x.partial_cmp(&value).expect("Operation failed")) {
        Ok(idx) => idx,
        Err(idx) => {
            if idx == 0 {
                0
            } else if idx >= arr.len() {
                arr.len() - 1
            } else {
                idx - 1
            }
        }
    }
}

/// Create a 2D interpolator (convenience function)
///
/// This function provides a simple interface similar to SciPy's interp2d.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_interpolate::interp2d::{interp2d, Interp2dKind};
///
/// let x = array![0.0, 1.0, 2.0];
/// let y = array![0.0, 1.0];
/// let z = Array2::from_shape_fn((2, 3), |(i, j)| {
///     y[i] * x[j] // z = x * y
/// });
///
/// let interp = interp2d(&x.view(), &y.view(), &z.view(), Interp2dKind::Linear)?;
/// let result = interp.evaluate(1.5, 0.5)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn interp2d<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    z: &ArrayView2<F>,
    kind: Interp2dKind,
) -> InterpolateResult<Interp2d<F>>
where
    F: Float + FromPrimitive + Debug + Clone + crate::traits::InterpolationFloat,
{
    Interp2d::new(x, y, z, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_linear_interpolation() -> InterpolateResult<()> {
        // Create a simple 2x3 grid where z = x + y
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0];
        let z = Array2::from_shape_fn((2, 3), |(i, j)| y[i] + x[j]);

        let interp = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Linear)?;

        // Test exact grid points
        assert_abs_diff_eq!(interp.evaluate(0.0, 0.0)?, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.evaluate(1.0, 0.0)?, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.evaluate(0.0, 1.0)?, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.evaluate(2.0, 1.0)?, 3.0, epsilon = 1e-10);

        // Test interpolated point
        assert_abs_diff_eq!(interp.evaluate(0.5, 0.5)?, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.evaluate(1.5, 0.5)?, 2.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_cubic_interpolation() -> InterpolateResult<()> {
        // Create a 4x4 grid for cubic interpolation
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 2.0, 3.0];
        let z = Array2::from_shape_fn((4, 4), |(i, j)| {
            let x_val = x[j];
            let y_val = y[i];
            x_val * x_val + y_val * y_val // z = x² + y²
        });

        let interp = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Cubic)?;

        // Test exact grid points
        assert_abs_diff_eq!(interp.evaluate(0.0, 0.0)?, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.evaluate(1.0, 1.0)?, 2.0, epsilon = 1e-10);

        // Test interpolated point (should be close to the function value)
        let result = interp.evaluate(1.5, 1.5)?;
        let expected = 1.5 * 1.5 + 1.5 * 1.5; // 4.5
        assert!((result - expected).abs() < 0.5); // Reasonable tolerance for cubic

        Ok(())
    }

    #[test]
    fn test_grid_evaluation() -> InterpolateResult<()> {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];
        let z = Array2::from_shape_fn((2, 2), |(i, j)| y[i] + x[j]);

        let interp = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Linear)?;

        let x_new = array![0.0, 0.5, 1.0];
        let ynew = array![0.0, 0.5, 1.0];

        let result = interp.evaluate_grid(&x_new.view(), &ynew.view())?;

        assert_eq!(result.shape(), &[3, 3]);
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10); // (0,0)
        assert_abs_diff_eq!(result[[1, 1]], 1.0, epsilon = 1e-10); // (0.5,0.5)
        assert_abs_diff_eq!(result[[2, 2]], 2.0, epsilon = 1e-10); // (1,1)

        Ok(())
    }

    #[test]
    fn test_validation() {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];
        let z = Array2::zeros((3, 2)); // Wrong shape

        let result = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsorted_coordinates() {
        let x = array![1.0, 0.0]; // Not sorted
        let y = array![0.0, 1.0];
        let z = Array2::zeros((2, 2));

        let result = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_quintic_spline_1d_exact_quadratic_reproduction() -> InterpolateResult<()> {
        // A natural quintic spline (third and fourth derivatives zero at
        // the two endpoints) exactly reproduces any polynomial of degree
        // <= 2, exactly as a natural cubic spline exactly reproduces any
        // polynomial of degree <= 1 -- both because such low-degree
        // polynomials trivially satisfy the "natural" boundary condition
        // everywhere. Non-uniform grid, non-constant data.
        let x = array![0.0, 0.3, 0.9, 1.5, 2.2, 3.0, 3.7];
        let poly = |v: f64| 3.0 - 2.0 * v + 0.5 * v * v;
        let y = x.mapv(poly);

        let spline = QuinticSpline1D::new(&x.view(), &y.view())?;

        for &xq in &[0.05, 0.6, 1.1, 1.9, 2.6, 3.5] {
            let got = spline.evaluate(xq);
            let expected = poly(xq);
            assert!(
                (got - expected).abs() < 1e-9,
                "quintic spline should exactly reproduce a quadratic: got {got}, \
                 expected {expected} at x={xq}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_quintic_spline_1d_converges_faster_than_cubic_order() -> InterpolateResult<()> {
        // Genuine quintic-order accuracy on a smooth, non-polynomial
        // function (sin) must converge dramatically faster than a cubic
        // spline's well-known O(h^4) rate as the grid is refined: doubling
        // the resolution should shrink the error by far more than cubic's
        // ~16x (2^4) per halving. A silent fallback to cubic would only
        // ever show ~16x here, never the >50x required below.
        fn error_at(n: usize) -> InterpolateResult<f64> {
            let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, n);
            let y = x.mapv(|v: f64| v.sin());
            let spline = QuinticSpline1D::new(&x.view(), &y.view())?;
            let h = x[1] - x[0];
            let xq = std::f64::consts::PI + 0.31 * h; // off-node, near domain center
            Ok((spline.evaluate(xq) - xq.sin()).abs())
        }

        let e11 = error_at(11)?;
        let e21 = error_at(21)?;
        let e41 = error_at(41)?;

        assert!(e11 > 0.0 && e21 > 0.0 && e41 > 0.0);
        assert!(
            e11 / e21 > 50.0,
            "expected quintic-order convergence (>50x per doubling), got {}x \
             (e11={e11}, e21={e21})",
            e11 / e21
        );
        assert!(
            e21 / e41 > 50.0,
            "expected quintic-order convergence (>50x per doubling), got {}x \
             (e21={e21}, e41={e41})",
            e21 / e41
        );

        Ok(())
    }

    #[test]
    fn test_quintic_spline_1d_rejects_degenerate_input() {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];
        // Fewer than 3 points: cannot build the natural quintic system.
        assert!(QuinticSpline1D::new(&x.view(), &y.view()).is_err());

        let x2 = array![0.0, 1.0, 1.0];
        let y2 = array![0.0, 1.0, 2.0];
        // Non-strictly-increasing x.
        assert!(QuinticSpline1D::new(&x2.view(), &y2.view()).is_err());
    }

    #[test]
    fn test_interp2d_quintic_reproduces_separable_quadratic_exactly() -> InterpolateResult<()> {
        // z = x^2 + y^2 is additively separable into two degree-2 pieces,
        // so the tensor-product quintic construction (a quintic spline
        // along x for every row, then a quintic spline along y) must
        // reproduce it to near machine precision -- a stronger, more
        // reliable check than merely comparing against a cubic spline
        // (whose own accuracy is data- and resolution-dependent).
        let x = array![0.0, 0.4, 1.1, 1.8, 2.6, 3.5];
        let y = array![0.0, 0.5, 1.3, 2.1, 2.9];
        let z = Array2::from_shape_fn((y.len(), x.len()), |(i, j)| x[j] * x[j] + y[i] * y[i]);

        let interp = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Quintic)?;

        for &(xq, yq) in &[(0.2, 0.3), (1.5, 1.0), (3.2, 2.5), (0.05, 2.85)] {
            let got = interp.evaluate(xq, yq)?;
            let expected = xq * xq + yq * yq;
            assert!(
                (got - expected).abs() < 1e-8,
                "quintic Interp2d should exactly reproduce x^2+y^2: got {got}, \
                 expected {expected} at ({xq}, {yq})"
            );
        }

        Ok(())
    }

    #[test]
    fn test_interp2d_quintic_is_not_a_silent_cubic_fallback() -> InterpolateResult<()> {
        // On identical, smooth (non-polynomial) data, Quintic must produce
        // a genuinely different result from Cubic. Under the previous
        // implementation, `Interp2dKind::Quintic` silently called
        // `evaluate_cubic`, so the two would have been bit-for-bit
        // identical; a real quintic implementation must differ by far more
        // than any floating-point rounding noise (~1e-13 here).
        let n = 9;
        let x = Array1::linspace(0.0, 3.0, n);
        let y = Array1::linspace(0.0, 2.0, n);
        let z = Array2::from_shape_fn((n, n), |(i, j)| x[j].sin() + y[i].cos());

        let quintic = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Quintic)?;
        let cubic = Interp2d::new(&x.view(), &y.view(), &z.view(), Interp2dKind::Cubic)?;

        let (xq, yq) = (1.35, 0.83);
        let quintic_val = quintic.evaluate(xq, yq)?;
        let cubic_val = cubic.evaluate(xq, yq)?;

        assert!(
            (quintic_val - cubic_val).abs() > 1e-8,
            "Quintic ({quintic_val}) must not silently match Cubic ({cubic_val})"
        );

        Ok(())
    }
}
