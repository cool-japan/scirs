//! 2D interpolation - SciPy-compatible interp2d implementation
//!
//! This module provides 2D interpolation functionality compatible with
//! SciPy's interp2d function for interpolating data on regular grids.

use crate::error::{InterpolateError, InterpolateResult};
use crate::interp1d::linear_interpolate;
use crate::spline::CubicSpline;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// 2D interpolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interp2dKind {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation using splines
    Cubic,
    /// Quintic interpolation (not implemented yet)
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
    F: Float + FromPrimitive + Debug + Clone,
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
    /// use ndarray::{array, Array2};
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
    /// * `y_new` - Y coordinate for evaluation
    ///
    /// # Returns
    ///
    /// Interpolated value at (x_new, y_new)
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, Array2};
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
    pub fn evaluate(&self, x_new: F, y_new: F) -> InterpolateResult<F> {
        match self.kind {
            Interp2dKind::Linear => self.evaluate_linear(x_new, y_new),
            Interp2dKind::Cubic => self.evaluate_cubic(x_new, y_new),
            Interp2dKind::Quintic => Err(InterpolateError::invalid_input(
                "quintic interpolation not implemented yet",
            )),
        }
    }

    /// Evaluate at multiple points
    ///
    /// # Arguments
    ///
    /// * `x_new` - X coordinates for evaluation
    /// * `y_new` - Y coordinates for evaluation (must have same length as x_new)
    ///
    /// # Returns
    ///
    /// Array of interpolated values
    pub fn evaluate_array(
        &self,
        x_new: &ArrayView1<F>,
        y_new: &ArrayView1<F>,
    ) -> InterpolateResult<Array1<F>> {
        if x_new.len() != y_new.len() {
            return Err(InterpolateError::shape_mismatch(
                format!("x_new.len() = {}", x_new.len()),
                format!("y_new.len() = {}", y_new.len()),
                "interp2d coordinate arrays",
            ));
        }

        let mut result = Array1::zeros(x_new.len());
        for i in 0..x_new.len() {
            result[i] = self.evaluate(x_new[i], y_new[i])?;
        }
        Ok(result)
    }

    /// Evaluate on a regular grid
    ///
    /// # Arguments
    ///
    /// * `x_new` - X coordinates for output grid
    /// * `y_new` - Y coordinates for output grid
    ///
    /// # Returns
    ///
    /// 2D array with shape (len(y_new), len(x_new))
    pub fn evaluate_grid(
        &self,
        x_new: &ArrayView1<F>,
        y_new: &ArrayView1<F>,
    ) -> InterpolateResult<Array2<F>> {
        let mut result = Array2::zeros((y_new.len(), x_new.len()));

        for (i, &y_val) in y_new.iter().enumerate() {
            for (j, &x_val) in x_new.iter().enumerate() {
                result[[i, j]] = self.evaluate(x_val, y_val)?;
            }
        }

        Ok(result)
    }

    /// Linear interpolation implementation
    fn evaluate_linear(&self, x_new: F, y_new: F) -> InterpolateResult<F> {
        // Find y index and interpolate along x for neighboring y values
        let y_idx = find_interval(&self.y.view(), y_new);

        let result = if y_idx == 0 && y_new < self.y[0] {
            // Extrapolate below
            let row = self.z.slice(ndarray::s![0, ..]);
            linear_interpolate(&self.x.view(), &row, &Array1::from_vec(vec![x_new]).view())?[0]
        } else if y_idx >= self.y.len() - 1 && y_new > self.y[self.y.len() - 1] {
            // Extrapolate above
            let row = self.z.slice(ndarray::s![self.y.len() - 1, ..]);
            linear_interpolate(&self.x.view(), &row, &Array1::from_vec(vec![x_new]).view())?[0]
        } else {
            // Interpolate between two y values
            let y_idx = y_idx.min(self.y.len() - 2);

            // Interpolate along x for both y levels
            let row0 = self.z.slice(ndarray::s![y_idx, ..]);
            let row1 = self.z.slice(ndarray::s![y_idx + 1, ..]);

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
                let t = (y_new - y0) / (y1 - y0);
                val0 + t * (val1 - val0)
            }
        };

        Ok(result)
    }

    /// Cubic interpolation implementation
    fn evaluate_cubic(&self, x_new: F, y_new: F) -> InterpolateResult<F> {
        // Create cubic splines for each x value across y
        let mut values_at_x = Array1::zeros(self.y.len());

        for (i, &_y_val) in self.y.iter().enumerate() {
            let row = self.z.slice(ndarray::s![i, ..]);
            let spline = CubicSpline::new(&self.x.view(), &row)?;
            values_at_x[i] = spline.evaluate(x_new)?;
        }

        // Create cubic spline along y direction
        let y_spline = CubicSpline::new(&self.y.view(), &values_at_x.view())?;
        y_spline.evaluate(y_new)
    }
}

/// Check if array is sorted in ascending order
fn is_sorted<F: PartialOrd>(arr: &ArrayView1<F>) -> bool {
    for window in arr.windows(2) {
        if window[0] > window[1] {
            return false;
        }
    }
    true
}

/// Find interval containing the value using binary search
fn find_interval<F: PartialOrd>(arr: &ArrayView1<F>, value: F) -> usize {
    // Convert to slice to use binary_search_by
    let slice: &[F] = arr.as_slice().unwrap();
    match slice.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
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
/// use ndarray::{array, Array2};
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
pub fn interp2d<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    z: &ArrayView2<F>,
    kind: Interp2dKind,
) -> InterpolateResult<Interp2d<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    Interp2d::new(x, y, z, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

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
        let y_new = array![0.0, 0.5, 1.0];

        let result = interp.evaluate_grid(&x_new.view(), &y_new.view())?;

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
}
