//! Core NURBS constructors and basic operations
//!
//! This module contains the fundamental constructors and accessor methods
//! for NURBS curves and surfaces, including parameter validation and
//! basic geometric queries.

use crate::bspline::{BSpline, ExtrapolateMode};
use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use super::types::{NurbsCurve, NurbsSurface, NurbsFloat, NurbsValidator, ValidationResult};

impl<T: NurbsFloat> NurbsCurve<T> {
    /// Create a new NURBS curve from control points, weights, knots, and degree
    ///
    /// # Arguments
    ///
    /// * `control_points` - Control points in n-dimensional space
    /// * `weights` - Weights for each control point (must have the same length as control_points.shape()[0])
    /// * `knots` - Knot vector
    /// * `degree` - Degree of the NURBS curve
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new `NurbsCurve` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Control points and weights have different lengths
    /// - Any weight is non-positive
    /// - Knot vector is invalid
    /// - Degree is too high for the number of control points
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_core::ndarray::{array, Array1, Array2};
    /// use scirs2_interpolate::nurbs::NurbsCurve;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// // Create a simple NURBS curve
    /// let control_points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 1.0],
    ///     [2.0, 0.0]
    /// ];
    /// let weights = array![1.0, 1.0, 1.0];
    /// let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    /// let degree = 2;
    ///
    /// let curve = NurbsCurve::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     &knots.view(),
    ///     degree,
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    /// ```
    pub fn new(
        control_points: &ArrayView2<T>,
        weights: &ArrayView1<T>,
        knots: &ArrayView1<T>,
        degree: usize,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        // Validate input parameters
        let validation = NurbsValidator::validate_curve_inputs(&control_points.to_owned(), &weights.to_owned(), &knots.to_owned(), degree);
        if !validation.is_valid() {
            return Err(InterpolateError::invalid_input(
                validation.to_error_message().to_string(),
            ));
        }

        // Create homogeneous coordinates by multiplying control points by weights
        let n = control_points.shape()[0];
        let dim = control_points.shape()[1];
        let mut homogeneous_coords = Array1::zeros(n);

        // We set the coefficient array to just the weights for now
        // Later in evaluate() we'll compute the full homogeneous coordinates
        for i in 0..n {
            homogeneous_coords[i] = weights[i];
        }

        // Create the underlying B-spline
        let bspline = BSpline::new(knots, &homogeneous_coords.view(), degree, extrapolate)?;

        Ok(NurbsCurve {
            control_points: control_points.to_owned(),
            weights: weights.to_owned(),
            bspline,
            dimension: dim,
        })
    }

    /// Create a NURBS curve from arrays (convenience constructor)
    ///
    /// This is a convenience method that takes owned arrays instead of views.
    ///
    /// # Arguments
    ///
    /// * `control_points` - Control points in n-dimensional space
    /// * `weights` - Weights for each control point
    /// * `knots` - Knot vector
    /// * `degree` - Degree of the NURBS curve
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new `NurbsCurve` object
    pub fn from_arrays(
        control_points: Array2<T>,
        weights: Array1<T>,
        knots: Array1<T>,
        degree: usize,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        Self::new(
            &control_points.view(),
            &weights.view(),
            &knots.view(),
            degree,
            extrapolate,
        )
    }

    /// Get the control points of the NURBS curve
    ///
    /// # Returns
    ///
    /// Reference to the 2D array of control points
    pub fn control_points(&self) -> &Array2<T> {
        &self.control_points
    }

    /// Get the weights of the NURBS curve
    ///
    /// # Returns
    ///
    /// Reference to the 1D array of weights
    pub fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    /// Get the knot vector of the NURBS curve
    ///
    /// # Returns
    ///
    /// Reference to the knot vector
    pub fn knots(&self) -> &Array1<T> {
        self.bspline.knot_vector()
    }

    /// Get the degree of the NURBS curve
    ///
    /// # Returns
    ///
    /// The degree of the curve
    pub fn degree(&self) -> usize {
        self.bspline.degree()
    }

    /// Get the dimension of the control points
    ///
    /// # Returns
    ///
    /// The spatial dimension of the control points
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of control points
    ///
    /// # Returns
    ///
    /// The number of control points
    pub fn len(&self) -> usize {
        self.control_points.nrows()
    }

    /// Check if the curve has no control points
    ///
    /// # Returns
    ///
    /// True if the curve has no control points
    pub fn is_empty(&self) -> bool {
        self.control_points.is_empty()
    }

    /// Get the parameter domain of the curve
    ///
    /// # Returns
    ///
    /// A tuple (t_min, t_max) representing the parameter domain
    pub fn domain(&self) -> (T, T) {
        let knots = self.bspline.knot_vector();
        let degree = self.bspline.degree();
        let t_min = knots[degree];
        let t_max = knots[knots.len() - degree - 1];
        (t_min, t_max)
    }

    /// Clone the curve with modified weights
    ///
    /// Creates a new curve with the same control points and knots but different weights.
    ///
    /// # Arguments
    ///
    /// * `new_weights` - New weights for the control points
    ///
    /// # Returns
    ///
    /// A new `NurbsCurve` with the modified weights
    ///
    /// # Errors
    ///
    /// Returns an error if the new weights array has incorrect length or contains non-positive values.
    pub fn with_weights(&self, new_weights: &ArrayView1<T>) -> InterpolateResult<Self> {
        Self::new(
            &self.control_points.view(),
            new_weights,
            &self.knots().view(),
            self.degree(),
            self.bspline.extrapolate_mode(),
        )
    }

    /// Clone the curve with modified control points
    ///
    /// Creates a new curve with the same weights and knots but different control points.
    ///
    /// # Arguments
    ///
    /// * `new_control_points` - New control points
    ///
    /// # Returns
    ///
    /// A new `NurbsCurve` with the modified control points
    ///
    /// # Errors
    ///
    /// Returns an error if the new control points have incorrect dimensions.
    pub fn with_control_points(&self, new_control_points: &ArrayView2<T>) -> InterpolateResult<Self> {
        Self::new(
            new_control_points,
            &self.weights.view(),
            &self.knots().view(),
            self.degree(),
            self.bspline.extrapolate_mode(),
        )
    }
}

impl<T: NurbsFloat> NurbsSurface<T> {
    /// Create a new NURBS surface from control points, weights, knot vectors, and degrees
    ///
    /// # Arguments
    ///
    /// * `control_points` - Control points arranged as (nu * nv x dim)
    /// * `weights` - Weights for each control point (nu * nv)
    /// * `nu` - Number of control points in the u direction
    /// * `nv` - Number of control points in the v direction
    /// * `knotsu` - Knot vector in the u direction
    /// * `knotsv` - Knot vector in the v direction
    /// * `degreeu` - Degree in the u direction
    /// * `degreev` - Degree in the v direction
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new `NurbsSurface` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Control points and weights have inconsistent dimensions
    /// - Any weight is non-positive
    /// - Knot vectors are invalid
    /// - Degrees are too high for the number of control points
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_core::ndarray::{array, Array1, Array2};
    /// use scirs2_interpolate::nurbs::NurbsSurface;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// // Create a simple NURBS surface
    /// let control_points = array![
    ///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]
    /// ];
    /// let weights = array![1.0, 1.0, 1.0, 1.0];
    /// let knotsu = array![0.0, 0.0, 1.0, 1.0];
    /// let knotsv = array![0.0, 0.0, 1.0, 1.0];
    ///
    /// let surface = NurbsSurface::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     2, 2,  // nu, nv
    ///     &knotsu.view(),
    ///     &knotsv.view(),
    ///     1, 1,  // degreeu, degreev
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    /// ```
    pub fn new(
        control_points: &ArrayView2<T>,
        weights: &ArrayView1<T>,
        nu: usize,
        nv: usize,
        knotsu: &ArrayView1<T>,
        knotsv: &ArrayView1<T>,
        degreeu: usize,
        degreev: usize,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        // Validate input parameters
        let validation = NurbsValidator::validate_surface_inputs(
            &control_points.to_owned(), &weights.to_owned(), nu, nv, &knotsu.to_owned(), &knotsv.to_owned(), degreeu, degreev,
        );
        if !validation.is_valid() {
            return Err(InterpolateError::invalid_input(
                validation.to_error_message().to_string(),
            ));
        }

        let dimension = control_points.shape()[1];

        Ok(NurbsSurface {
            control_points: control_points.to_owned(),
            weights: weights.to_owned(),
            nu,
            nv,
            knotsu: knotsu.to_owned(),
            knotsv: knotsv.to_owned(),
            degreeu,
            degreev,
            dimension,
            extrapolate,
        })
    }

    /// Create a NURBS surface from arrays (convenience constructor)
    ///
    /// This is a convenience method that takes owned arrays instead of views.
    ///
    /// # Arguments
    ///
    /// * `control_points` - Control points arranged as (nu * nv x dim)
    /// * `weights` - Weights for each control point
    /// * `nu` - Number of control points in the u direction
    /// * `nv` - Number of control points in the v direction
    /// * `knotsu` - Knot vector in the u direction
    /// * `knotsv` - Knot vector in the v direction
    /// * `degreeu` - Degree in the u direction
    /// * `degreev` - Degree in the v direction
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new `NurbsSurface` object
    pub fn from_arrays(
        control_points: Array2<T>,
        weights: Array1<T>,
        nu: usize,
        nv: usize,
        knotsu: Array1<T>,
        knotsv: Array1<T>,
        degreeu: usize,
        degreev: usize,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        Self::new(
            &control_points.view(),
            &weights.view(),
            nu,
            nv,
            &knotsu.view(),
            &knotsv.view(),
            degreeu,
            degreev,
            extrapolate,
        )
    }

    /// Get the control points of the NURBS surface
    ///
    /// # Returns
    ///
    /// Reference to the 2D array of control points
    pub fn control_points(&self) -> &Array2<T> {
        &self.control_points
    }

    /// Get the weights of the NURBS surface
    ///
    /// # Returns
    ///
    /// Reference to the 1D array of weights
    pub fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    /// Get the knot vector in the u direction
    ///
    /// # Returns
    ///
    /// Reference to the u-direction knot vector
    pub fn knotsu(&self) -> &Array1<T> {
        &self.knotsu
    }

    /// Get the knot vector in the v direction
    ///
    /// # Returns
    ///
    /// Reference to the v-direction knot vector
    pub fn knotsv(&self) -> &Array1<T> {
        &self.knotsv
    }

    /// Get the degree in the u direction
    ///
    /// # Returns
    ///
    /// The degree in the u direction
    pub fn degreeu(&self) -> usize {
        self.degreeu
    }

    /// Get the degree in the v direction
    ///
    /// # Returns
    ///
    /// The degree in the v direction
    pub fn degreev(&self) -> usize {
        self.degreev
    }

    /// Get the number of control points in the u direction
    ///
    /// # Returns
    ///
    /// Number of control points in u direction
    pub fn nu(&self) -> usize {
        self.nu
    }

    /// Get the number of control points in the v direction
    ///
    /// # Returns
    ///
    /// Number of control points in v direction
    pub fn nv(&self) -> usize {
        self.nv
    }

    /// Get the dimension of the control points
    ///
    /// # Returns
    ///
    /// The spatial dimension of the control points
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the total number of control points
    ///
    /// # Returns
    ///
    /// The total number of control points (nu * nv)
    pub fn len(&self) -> usize {
        self.nu * self.nv
    }

    /// Check if the surface has no control points
    ///
    /// # Returns
    ///
    /// True if the surface has no control points
    pub fn is_empty(&self) -> bool {
        self.nu == 0 || self.nv == 0
    }

    /// Get the parameter domain of the surface
    ///
    /// # Returns
    ///
    /// A tuple ((u_min, u_max), (v_min, v_max)) representing the parameter domain
    pub fn domain(&self) -> ((T, T), (T, T)) {
        let u_min = self.knotsu[self.degreeu];
        let u_max = self.knotsu[self.nu];
        let v_min = self.knotsv[self.degreev];
        let v_max = self.knotsv[self.nv];

        ((u_min, u_max), (v_min, v_max))
    }

    /// Get the extrapolation mode
    ///
    /// # Returns
    ///
    /// The extrapolation mode used by the surface
    pub fn extrapolate_mode(&self) -> ExtrapolateMode {
        self.extrapolate
    }
}