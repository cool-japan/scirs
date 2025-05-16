use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::collections::HashSet;
use std::fmt::Debug;

use crate::bspline::{make_lsq_bspline, BSpline};
use crate::error::{InterpolateError, InterpolateResult};
use crate::ExtrapolateMode;

/// Multiscale B-spline interpolation with adaptive refinement.
///
/// This implementation uses a hierarchical approach to B-spline interpolation,
/// starting with a coarse representation and progressively refining it by adding
/// knots where needed based on error metrics. This allows for more efficient
/// representation of functions with varying complexity in different regions.
///
/// Features:
/// - Starts with a coarse approximation and refines adaptively
/// - Supports different refinement criteria (error-based, curvature-based)
/// - Allows for local refinement without affecting the entire domain
/// - Maintains specified continuity across refinement levels
/// - Supports different error metrics and thresholds
#[derive(Debug, Clone)]
pub struct MultiscaleBSpline<T: Float + FromPrimitive + Debug + std::fmt::Display + 
                               std::ops::AddAssign + std::ops::SubAssign + 
                               std::ops::MulAssign + std::ops::DivAssign + 
                               std::ops::RemAssign> {
    /// Original x coordinates
    x: Array1<T>,
    /// Original y coordinates
    y: Array1<T>,
    /// Array of B-spline models at different scales
    levels: Vec<BSpline<T>>,
    /// The active (finest) level B-spline
    active_level: usize,
    /// Order (degree + 1) of the B-spline basis
    order: usize,
    /// Extrapolation mode for points outside the domain
    extrapolate: ExtrapolateMode,
    /// Maximum number of refinement levels
    max_levels: usize,
    /// Error threshold for refinement decisions
    error_threshold: T,
}

/// Criteria used for adaptive refinement decisions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RefinementCriterion {
    /// Refine based on absolute error exceeding threshold
    AbsoluteError,
    /// Refine based on relative error exceeding threshold
    RelativeError,
    /// Refine based on local curvature exceeding threshold
    Curvature,
    /// Refine based on both error and curvature
    Combined,
}

impl<T: Float + FromPrimitive + Debug + std::fmt::Display + 
           std::ops::AddAssign + std::ops::SubAssign + 
           std::ops::MulAssign + std::ops::DivAssign + 
           std::ops::RemAssign> MultiscaleBSpline<T> {
    /// Creates a new Multiscale B-spline interpolator with adaptive refinement.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinates of the data points, must be strictly increasing
    /// * `y` - The y-coordinates of the data points
    /// * `initial_knots` - Number of knots for the initial coarse approximation
    /// * `degree` - Degree of the B-spline (cubic = 3)
    /// * `max_levels` - Maximum number of refinement levels allowed
    /// * `error_threshold` - Error threshold for refinement decisions
    /// * `extrapolate` - How to handle points outside the domain of the data
    ///
    /// # Returns
    ///
    /// A `MultiscaleBSpline` object initialized with a coarse approximation.
    pub fn new(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        initial_knots: usize,
        degree: usize,
        max_levels: usize,
        error_threshold: T,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Input arrays must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }

        if x.len() < degree + 1 {
            return Err(InterpolateError::InvalidValue(format!(
                "At least {} data points are required for degree {}",
                degree + 1,
                degree
            )));
        }

        // Check if x is strictly increasing
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::InvalidValue(
                    "x values must be strictly increasing".to_string(),
                ));
            }
        }

        // Create the initial coarse B-spline approximation
        let order = degree + 1;
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();

        // Generate initial knots
        let initial_knots = std::cmp::min(initial_knots, x.len());
        let initial_spline =
            make_lsq_bspline(&x.view(), &y.view(), initial_knots, degree, extrapolate)?;

        let levels = vec![initial_spline];

        Ok(Self {
            x: x_owned,
            y: y_owned,
            levels,
            active_level: 0,
            order,
            extrapolate,
            max_levels,
            error_threshold,
        })
    }

    /// Refines the B-spline approximation by adding knots where needed.
    ///
    /// # Arguments
    ///
    /// * `criterion` - The criterion to use for refinement decisions
    /// * `max_new_knots` - Maximum number of new knots to add in this refinement step
    ///
    /// # Returns
    ///
    /// `true` if refinement was performed, `false` if no refinement was needed
    /// or the maximum level has been reached.
    pub fn refine(
        &mut self,
        criterion: RefinementCriterion,
        max_new_knots: usize,
    ) -> InterpolateResult<bool> {
        // Check if we've reached the maximum refinement level
        if self.active_level >= self.max_levels - 1 {
            return Ok(false);
        }

        // Get the current active B-spline
        let current_spline = &self.levels[self.active_level];

        // Compute errors at the original data points
        let y_approx = current_spline.evaluate(&self.x.view())?;
        let errors = &self.y - &y_approx;

        // Find candidate regions for refinement based on the criterion
        let candidates = self.find_refinement_candidates(&errors, criterion)?;

        if candidates.is_empty() {
            return Ok(false); // No refinement needed
        }

        // Limit the number of new knots to add
        let n_add = std::cmp::min(candidates.len(), max_new_knots);
        let candidates = candidates.into_iter().take(n_add).collect::<Vec<_>>();

        // Get current knots and add new ones
        let current_knots = current_spline.knot_vector();
        let degree = self.order - 1;

        // Generate new knots by adding to the current knot vector
        let mut new_knots = current_knots.clone();

        for &idx in &candidates {
            // Add a knot between data points where error is high
            if idx < self.x.len() - 1 {
                let new_knot = (self.x[idx] + self.x[idx + 1]) / T::from(2.0).unwrap();

                // Only add if it's not already in the knot vector
                if !new_knots
                    .iter()
                    .any(|&k| (k - new_knot).abs() < T::epsilon())
                {
                    new_knots.push(new_knot);
                }
            }
        }

        // Sort the new knots (required for B-splines)
        new_knots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Create a new refined B-spline with the expanded knot vector
        let refined_spline = BSpline::new(
            current_spline.get_coefficients().to_owned(),
            new_knots,
            self.order,
            self.extrapolate,
        )?;

        // Add the refined spline to the levels and make it active
        self.levels.push(refined_spline);
        self.active_level += 1;

        Ok(true)
    }

    /// Finds candidate regions for refinement based on the specified criterion.
    ///
    /// # Arguments
    ///
    /// * `errors` - Error values at data points
    /// * `criterion` - The criterion to use for refinement decisions
    ///
    /// # Returns
    ///
    /// Indices of data points where refinement should occur
    fn find_refinement_candidates(
        &self,
        errors: &Array1<T>,
        criterion: RefinementCriterion,
    ) -> InterpolateResult<Vec<usize>> {
        let mut candidates = HashSet::new();

        match criterion {
            RefinementCriterion::AbsoluteError => {
                // Find regions where absolute error exceeds threshold
                for (i, &err) in errors.iter().enumerate() {
                    if err.abs() > self.error_threshold {
                        candidates.insert(i);
                    }
                }
            }
            RefinementCriterion::RelativeError => {
                // Find regions where relative error exceeds threshold
                for (i, (&err, &y)) in errors.iter().zip(self.y.iter()).enumerate() {
                    let rel_err = if y.abs() > T::epsilon() {
                        err.abs() / y.abs()
                    } else {
                        err.abs()
                    };

                    if rel_err > self.error_threshold {
                        candidates.insert(i);
                    }
                }
            }
            RefinementCriterion::Curvature => {
                // Use the current active spline
                let spline = &self.levels[self.active_level];

                // Compute second derivatives to estimate curvature
                let second_derivs = spline.derivative(2, &self.x.view())?;

                // Find regions of high curvature
                for (i, &d2) in second_derivs.iter().enumerate() {
                    if d2.abs() > self.error_threshold {
                        candidates.insert(i);
                    }
                }
            }
            RefinementCriterion::Combined => {
                // Combine both error and curvature criteria
                let spline = &self.levels[self.active_level];
                let second_derivs = spline.derivative(2, &self.x.view())?;

                for (i, (&err, &d2)) in errors.iter().zip(second_derivs.iter()).enumerate() {
                    let combined_metric = err.abs() * (T::one() + d2.abs());

                    if combined_metric > self.error_threshold {
                        candidates.insert(i);
                    }
                }
            }
        }

        // Convert the set to a sorted vector
        let mut result: Vec<_> = candidates.into_iter().collect();
        result.sort();

        Ok(result)
    }

    /// Automatically refines the B-spline until the error threshold is met
    /// or the maximum number of levels is reached.
    ///
    /// # Arguments
    ///
    /// * `criterion` - The criterion to use for refinement decisions
    /// * `max_knots_per_level` - Maximum number of new knots to add per refinement level
    ///
    /// # Returns
    ///
    /// The number of refinement levels that were added
    pub fn auto_refine(
        &mut self,
        criterion: RefinementCriterion,
        max_knots_per_level: usize,
    ) -> InterpolateResult<usize> {
        let initial_level = self.active_level;

        loop {
            let refined = self.refine(criterion, max_knots_per_level)?;

            if !refined || self.active_level >= self.max_levels - 1 {
                break;
            }

            // Check if error is now below threshold
            let current_spline = &self.levels[self.active_level];
            let y_approx = current_spline.evaluate(&self.x.view())?;
            let errors = &self.y - &y_approx;

            // Calculate maximum error
            let max_error =
                errors.iter().fold(
                    T::zero(),
                    |max, &err| {
                        if err.abs() > max {
                            err.abs()
                        } else {
                            max
                        }
                    },
                );

            if max_error < self.error_threshold {
                break;
            }
        }

        Ok(self.active_level - initial_level)
    }

    /// Evaluate the multiscale B-spline at the given points.
    ///
    /// # Arguments
    ///
    /// * `x_new` - The points at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// A `Result` containing the interpolated values at the given points.
    pub fn evaluate(&self, x_new: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        if self.levels.is_empty() {
            return Err(InterpolateError::InvalidState(
                "No B-spline models available".to_string(),
            ));
        }

        // Use the current active (finest) level B-spline
        self.levels[self.active_level].evaluate(x_new)
    }

    /// Calculate derivative of the multiscale B-spline at the given points.
    ///
    /// # Arguments
    ///
    /// * `deriv_order` - The order of the derivative (1 for first derivative, 2 for second, etc.)
    /// * `x_new` - The points at which to evaluate the derivative
    ///
    /// # Returns
    ///
    /// A `Result` containing the derivative values at the given points.
    pub fn derivative(
        &self,
        deriv_order: usize,
        x_new: &ArrayView1<T>,
    ) -> InterpolateResult<Array1<T>> {
        if self.levels.is_empty() {
            return Err(InterpolateError::InvalidState(
                "No B-spline models available".to_string(),
            ));
        }

        // Use the current active (finest) level B-spline
        self.levels[self.active_level].derivative(deriv_order, x_new)
    }

    /// Get the number of knots at each level of refinement.
    pub fn get_knots_per_level(&self) -> Vec<usize> {
        self.levels
            .iter()
            .map(|spline| spline.get_knots().len())
            .collect()
    }

    /// Get the current active refinement level.
    pub fn get_active_level(&self) -> usize {
        self.active_level
    }

    /// Get the total number of refinement levels.
    pub fn get_num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get the current error threshold.
    pub fn get_error_threshold(&self) -> T {
        self.error_threshold
    }

    /// Set a new error threshold.
    pub fn set_error_threshold(&mut self, threshold: T) {
        self.error_threshold = threshold;
    }

    /// Get the maximum allowed refinement levels.
    pub fn get_max_levels(&self) -> usize {
        self.max_levels
    }

    /// Get a reference to the B-spline at a specific level.
    pub fn get_level_spline(&self, level: usize) -> Option<&BSpline<T>> {
        self.levels.get(level)
    }

    /// Switch to a different refinement level (coarser or finer).
    ///
    /// # Arguments
    ///
    /// * `level` - The refinement level to switch to (0 = coarsest)
    ///
    /// # Returns
    ///
    /// `true` if switch was successful, `false` if level is out of range
    pub fn switch_level(&mut self, level: usize) -> bool {
        if level < self.levels.len() {
            self.active_level = level;
            true
        } else {
            false
        }
    }
}

/// Creates a new multiscale B-spline with automatic adaptive refinement.
///
/// # Arguments
///
/// * `x` - The x-coordinates of the data points
/// * `y` - The y-coordinates of the data points
/// * `initial_knots` - Number of knots for the initial coarse approximation
/// * `degree` - Degree of the B-spline (cubic = 3)
/// * `error_threshold` - Error threshold for refinement decisions
/// * `criterion` - The criterion to use for refinement decisions
/// * `max_levels` - Maximum number of refinement levels allowed
/// * `extrapolate` - How to handle points outside the domain of the data
///
/// # Returns
///
/// A `Result` containing the adaptively refined multiscale B-spline.
pub fn make_adaptive_bspline<T: Float + FromPrimitive + Debug + std::fmt::Display + 
                              std::ops::AddAssign + std::ops::SubAssign + 
                              std::ops::MulAssign + std::ops::DivAssign + 
                              std::ops::RemAssign>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    initial_knots: usize,
    degree: usize,
    error_threshold: T,
    criterion: RefinementCriterion,
    max_levels: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<MultiscaleBSpline<T>> {
    // Create the initial multiscale B-spline
    let mut spline = MultiscaleBSpline::new(
        x,
        y,
        initial_knots,
        degree,
        max_levels,
        error_threshold,
        extrapolate,
    )?;

    // Automatically refine it
    spline.auto_refine(criterion, initial_knots)?;

    Ok(spline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_multiscale_bspline_creation() {
        let x = Array::linspace(0.0, 10.0, 101);
        let y = x.mapv(|v| v.sin());

        let spline =
            MultiscaleBSpline::new(&x.view(), &y.view(), 5, 3, 5, 0.01, ExtrapolateMode::Error)
                .unwrap();

        // Check that initial level is created
        assert_eq!(spline.get_num_levels(), 1);
        assert_eq!(spline.get_active_level(), 0);
    }

    #[test]
    fn test_multiscale_bspline_refinement() {
        let x = Array::linspace(0.0, 10.0, 101);

        // Create a function with a sharp feature in the middle
        let y = x.mapv(|v| {
            if (v - 5.0).abs() < 1.0 {
                (v - 5.0).powi(2)
            } else {
                v.sin()
            }
        });

        let mut spline =
            MultiscaleBSpline::new(&x.view(), &y.view(), 5, 3, 5, 0.05, ExtrapolateMode::Error)
                .unwrap();

        // Perform one refinement step
        let refined = spline
            .refine(RefinementCriterion::AbsoluteError, 3)
            .unwrap();

        // Refinement should have occurred
        assert!(refined);
        assert_eq!(spline.get_num_levels(), 2);
        assert_eq!(spline.get_active_level(), 1);
    }

    #[test]
    fn test_adaptive_bspline_auto_refinement() {
        let x = Array::linspace(0.0, 10.0, 101);

        // Create a function with multiple sharp features
        let y = x.mapv(|v| v.sin() + 0.5 * (v * 2.0).sin());

        // Create and auto-refine a multiscale B-spline
        let spline = make_adaptive_bspline(
            &x.view(),
            &y.view(),
            5,
            3,
            0.01,
            RefinementCriterion::AbsoluteError,
            5,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Should have refined to multiple levels
        assert!(spline.get_num_levels() > 1);

        // Evaluate at the original points
        let y_approx = spline.evaluate(&x.view()).unwrap();

        // Calculate maximum error
        let max_error = y
            .iter()
            .zip(y_approx.iter())
            .map(|(&y_true, &y_pred)| (y_true - y_pred).abs())
            .fold(0.0, |max, err| if err > max { err } else { max });

        // Error should be below threshold
        assert!(
            max_error <= spline.get_error_threshold()
                || (max_error - spline.get_error_threshold()).abs() < 1e-6
        );
    }

    #[test]
    fn test_multiscale_bspline_derivatives() {
        let x = Array::linspace(0.0, 10.0, 101);
        let y = x.mapv(|v| v.powi(2));

        // Create and auto-refine a multiscale B-spline
        let spline = make_adaptive_bspline(
            &x.view(),
            &y.view(),
            5,
            3,
            0.01,
            RefinementCriterion::AbsoluteError,
            3,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Calculate first derivative at several points
        let x_test = Array::from_vec(vec![2.5, 5.0, 7.5]);
        let deriv1 = spline.derivative(1, &x_test.view()).unwrap();

        // For y = x^2, the first derivative should be approximately 2*x
        for i in 0..x_test.len() {
            let expected = 2.0 * x_test[i];
            assert_abs_diff_eq!(deriv1[i], expected, epsilon = 0.2);
        }
    }

    #[test]
    fn test_multiscale_bspline_level_switching() {
        let x = Array::linspace(0.0, 10.0, 101);
        let y = x.mapv(|v| v.sin());

        // Create and auto-refine a multiscale B-spline
        let mut spline = make_adaptive_bspline(
            &x.view(),
            &y.view(),
            5,
            3,
            0.01,
            RefinementCriterion::AbsoluteError,
            3,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Ensure multiple levels exist
        assert!(spline.get_num_levels() > 1);

        // Get the initial active level
        let initial_level = spline.get_active_level();

        // Switch to the coarsest level
        let result = spline.switch_level(0);
        assert!(result);
        assert_eq!(spline.get_active_level(), 0);

        // Evaluate at the original points using the coarse level
        let y_coarse = spline.evaluate(&x.view()).unwrap();

        // Switch back to the finest level
        let result = spline.switch_level(initial_level);
        assert!(result);
        assert_eq!(spline.get_active_level(), initial_level);

        // Evaluate at the original points using the fine level
        let y_fine = spline.evaluate(&x.view()).unwrap();

        // Fine approximation should be more accurate
        let err_coarse = y
            .iter()
            .zip(y_coarse.iter())
            .map(|(&y_true, &y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        let err_fine = y
            .iter()
            .zip(y_fine.iter())
            .map(|(&y_true, &y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        assert!(err_fine < err_coarse);
    }

    #[test]
    fn test_different_refinement_criteria() {
        let x = Array::linspace(0.0, 10.0, 101);

        // Create a function with sharp features and varying curvature
        let y = x.mapv(|v| v.sin() + 0.2 * (v * 3.0).sin() + 0.1 * (v - 5.0).powi(2));

        // Create splines with different refinement criteria
        let spline_abs = make_adaptive_bspline(
            &x.view(),
            &y.view(),
            5,
            3,
            0.01,
            RefinementCriterion::AbsoluteError,
            3,
            ExtrapolateMode::Error,
        )
        .unwrap();

        let spline_curv = make_adaptive_bspline(
            &x.view(),
            &y.view(),
            5,
            3,
            0.5,
            RefinementCriterion::Curvature,
            3,
            ExtrapolateMode::Error,
        )
        .unwrap();

        let spline_comb = make_adaptive_bspline(
            &x.view(),
            &y.view(),
            5,
            3,
            0.01,
            RefinementCriterion::Combined,
            3,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // All should have refined to multiple levels
        assert!(spline_abs.get_num_levels() > 1);
        assert!(spline_curv.get_num_levels() > 1);
        assert!(spline_comb.get_num_levels() > 1);

        // Get knots per level to verify refinement patterns
        let knots_abs = spline_abs.get_knots_per_level();
        let knots_curv = spline_curv.get_knots_per_level();
        let knots_comb = spline_comb.get_knots_per_level();

        // Each refinement should add knots
        for i in 1..knots_abs.len() {
            assert!(knots_abs[i] > knots_abs[i - 1]);
        }

        for i in 1..knots_curv.len() {
            assert!(knots_curv[i] > knots_curv[i - 1]);
        }

        for i in 1..knots_comb.len() {
            assert!(knots_comb[i] > knots_comb[i - 1]);
        }
    }
}
