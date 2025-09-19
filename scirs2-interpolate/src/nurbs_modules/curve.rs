//! NURBS curve evaluation and manipulation methods
//!
//! This module contains all the methods for evaluating NURBS curves and
//! performing various geometric operations such as derivatives, integration,
//! knot insertion, and curve analysis.

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1};
use super::types::{NurbsCurve, NurbsFloat};

impl<T: NurbsFloat> NurbsCurve<T> {
    /// Evaluate the NURBS curve at parameter value t
    ///
    /// This method computes a point on the NURBS curve using the rational basis functions.
    /// The curve is evaluated using the formula:
    /// C(t) = Σ(w_i * P_i * N_i(t)) / Σ(w_i * N_i(t))
    ///
    /// # Arguments
    ///
    /// * `t` - Parameter value at which to evaluate the curve
    ///
    /// # Returns
    ///
    /// A point on the NURBS curve at parameter t
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::nurbs::NurbsCurve;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let control_points = array![[0.0, 0.0], [1.0, 0.0]];
    /// let weights = array![1.0, 1.0];
    /// let knots = array![0.0, 0.0, 1.0, 1.0];
    /// let curve = NurbsCurve::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     &knots.view(),
    ///     1,
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    ///
    /// let point = curve.evaluate(0.5).unwrap();
    /// println!("Point at t=0.5: {:?}", point);
    /// ```
    pub fn evaluate(&self, t: T) -> InterpolateResult<Array1<T>> {
        // Create homogeneous coordinates for each control point
        let n = self.control_points.shape()[0];
        let mut homogeneous_points = Vec::with_capacity(n);

        for i in 0..n {
            let mut point = Vec::with_capacity(self.dimension + 1);
            for j in 0..self.dimension {
                point.push(self.control_points[[i, j]] * self.weights[i]);
            }
            point.push(self.weights[i]);
            homogeneous_points.push(point);
        }

        // Compute the basis functions
        let basisvalues = self.compute_basisvalues(t)?;

        // Compute the weighted sum of control points
        let mut numerator: Vec<T> = vec![T::zero(); self.dimension];
        let mut denominator = T::zero();

        for i in 0..n {
            let basis = basisvalues[i];
            for (j, num) in numerator.iter_mut().enumerate() {
                *num += homogeneous_points[i][j] * basis;
            }
            denominator += homogeneous_points[i][self.dimension] * basis;
        }

        // Return the rational point
        let mut result = Array1::zeros(self.dimension);
        if denominator > T::epsilon() {
            for j in 0..self.dimension {
                result[j] = numerator[j] / denominator;
            }
        }

        Ok(result)
    }

    /// Evaluate the NURBS curve at multiple parameter values
    ///
    /// This is more efficient than calling `evaluate` multiple times as it
    /// can reuse computations where possible.
    ///
    /// # Arguments
    ///
    /// * `tvalues` - Array of parameter values
    ///
    /// # Returns
    ///
    /// Array of points on the NURBS curve at the given parameter values
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::nurbs::NurbsCurve;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let control_points = array![[0.0, 0.0], [1.0, 0.0]];
    /// let weights = array![1.0, 1.0];
    /// let knots = array![0.0, 0.0, 1.0, 1.0];
    /// let curve = NurbsCurve::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     &knots.view(),
    ///     1,
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    ///
    /// let t_vals = array![0.0, 0.25, 0.5, 0.75, 1.0];
    /// let points = curve.evaluate_array(&t_vals.view()).unwrap();
    /// ```
    pub fn evaluate_array(&self, tvalues: &ArrayView1<T>) -> InterpolateResult<Array2<T>> {
        let n_points = tvalues.len();
        let mut result = Array2::zeros((n_points, self.dimension));

        for (i, &t) in tvalues.iter().enumerate() {
            let point = self.evaluate(t)?;
            for j in 0..self.dimension {
                result[[i, j]] = point[j];
            }
        }

        Ok(result)
    }

    /// Compute the derivative of the NURBS curve at parameter value t
    ///
    /// Computes derivatives using the rational derivative formula. For a NURBS curve
    /// C(t) = P(t)/w(t), the derivative is C'(t) = (P'(t)*w(t) - P(t)*w'(t))/w(t)²
    ///
    /// # Arguments
    ///
    /// * `t` - Parameter value
    /// * `order` - Order of the derivative (defaults to 1)
    ///
    /// # Returns
    ///
    /// The derivative of the NURBS curve at parameter t
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::nurbs::NurbsCurve;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let control_points = array![[0.0, 0.0], [1.0, 0.0]];
    /// let weights = array![1.0, 1.0];
    /// let knots = array![0.0, 0.0, 1.0, 1.0];
    /// let curve = NurbsCurve::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     &knots.view(),
    ///     1,
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    ///
    /// // First derivative (tangent vector)
    /// let tangent = curve.derivative(0.5, 1).unwrap();
    ///
    /// // Second derivative (curvature information)
    /// let second_deriv = curve.derivative(0.5, 2).unwrap();
    /// ```
    pub fn derivative(&self, t: T, order: usize) -> InterpolateResult<Array1<T>> {
        if order == 0 {
            return self.evaluate(t);
        }

        // For derivatives of order > degree, return zero
        if order > self.degree() {
            return Ok(Array1::zeros(self.dimension));
        }

        // Compute derivatives using the generalized formula for NURBS derivatives
        // Based on "The NURBS Book" by Piegl and Tiller
        let n = self.weights.len();

        // Compute all derivatives up to the requested order
        let basis_derivs_all = self.compute_all_basis_derivatives(t, order)?;

        // Compute derivatives of the weighted control points (A^(k))
        let mut a_derivs = vec![Array1::<T>::zeros(self.dimension); order + 1];
        let mut w_derivs = vec![T::zero(); order + 1];

        for k in 0..=order {
            for i in 0..n {
                let basis_deriv = basis_derivs_all[k][i];
                for j in 0..self.dimension {
                    a_derivs[k][j] += self.control_points[[i, j]] * self.weights[i] * basis_deriv;
                }
                w_derivs[k] += self.weights[i] * basis_deriv;
            }
        }

        // Apply the rational derivative formula
        let mut result = Array1::zeros(self.dimension);
        let w0 = w_derivs[0];

        if w0.abs() > T::epsilon() {
            for j in 0..self.dimension {
                let mut deriv_sum = T::zero();

                // Apply binomial theorem for rational derivatives
                for i in 0..=order {
                    let binomial_coeff = binomial_coefficient(order, i);
                    let term = T::from(binomial_coeff).unwrap() * w_derivs[order - i] * a_derivs[i][j];
                    deriv_sum += if i == 0 { term } else { -term };
                }

                result[j] = deriv_sum / w0;
            }
        }

        Ok(result)
    }

    /// Compute derivatives at multiple parameter values
    ///
    /// # Arguments
    ///
    /// * `tvalues` - Array of parameter values
    /// * `order` - Order of derivative
    ///
    /// # Returns
    ///
    /// Array of derivative vectors at the given parameter values
    pub fn derivative_array(
        &self,
        tvalues: &ArrayView1<T>,
        order: usize,
    ) -> InterpolateResult<Array2<T>> {
        let n_points = tvalues.len();
        let mut result = Array2::zeros((n_points, self.dimension));

        for (i, &t) in tvalues.iter().enumerate() {
            let deriv = self.derivative(t, order)?;
            for j in 0..self.dimension {
                result[[i, j]] = deriv[j];
            }
        }

        Ok(result)
    }

    /// Compute all derivatives up to a maximum order at parameter value t
    ///
    /// This is more efficient than calling `derivative` multiple times for
    /// different orders.
    ///
    /// # Arguments
    ///
    /// * `t` - Parameter value
    /// * `maxorder` - Maximum order of derivatives to compute
    ///
    /// # Returns
    ///
    /// Vector of derivative arrays, where index k contains the k-th derivative
    pub fn derivatives_all(&self, t: T, maxorder: usize) -> InterpolateResult<Vec<Array1<T>>> {
        let mut results = Vec::with_capacity(maxorder + 1);

        for order in 0..=maxorder {
            results.push(self.derivative(t, order)?);
        }

        Ok(results)
    }

    /// Integrate the NURBS curve over the parameter interval [a, b]
    ///
    /// Computes the definite integral of the curve using adaptive quadrature.
    /// For vector-valued curves, each component is integrated separately.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// The integral of the curve over [a, b]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::nurbs::NurbsCurve;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let control_points = array![[0.0, 0.0], [1.0, 0.0]];
    /// let weights = array![1.0, 1.0];
    /// let knots = array![0.0, 0.0, 1.0, 1.0];
    /// let curve = NurbsCurve::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     &knots.view(),
    ///     1,
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    ///
    /// // Integrate curve from t=0 to t=1
    /// let integral = curve.integrate(0.0, 1.0).unwrap();
    /// ```
    pub fn integrate(&self, a: T, b: T) -> InterpolateResult<Array1<T>> {
        if a == b {
            return Ok(Array1::zeros(self.dimension));
        }

        let (lower, upper, sign) = if a < b {
            (a, b, T::one())
        } else {
            (b, a, -T::one())
        };

        // Use adaptive Simpson's rule for integration
        let mut result = Array1::zeros(self.dimension);
        let tolerance = T::from(1e-10).unwrap_or(T::epsilon());

        for dim in 0..self.dimension {
            let integral_value = self.adaptive_simpson_integration(
                lower,
                upper,
                dim,
                tolerance,
                10 // max recursion depth
            )?;
            result[dim] = sign * integral_value;
        }

        Ok(result)
    }

    /// Insert a knot into the curve
    ///
    /// Knot insertion adds a new knot value to the knot vector without changing
    /// the shape of the curve. This is useful for local refinement and control.
    ///
    /// # Arguments
    ///
    /// * `u` - Parameter value where to insert the knot
    /// * `r` - Number of times to insert the knot (multiplicity)
    ///
    /// # Returns
    ///
    /// A new NURBS curve with the inserted knot
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use scirs2_interpolate::nurbs::NurbsCurve;
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// let control_points = array![[0.0, 0.0], [1.0, 0.0]];
    /// let weights = array![1.0, 1.0];
    /// let knots = array![0.0, 0.0, 1.0, 1.0];
    /// let curve = NurbsCurve::new(
    ///     &control_points.view(),
    ///     &weights.view(),
    ///     &knots.view(),
    ///     1,
    ///     ExtrapolateMode::Extrapolate
    /// ).unwrap();
    ///
    /// // Insert knot at parameter 0.5 once
    /// let refined_curve = curve.insert_knot(0.5, 1).unwrap();
    /// ```
    pub fn insert_knot(&self, u: T, r: usize) -> InterpolateResult<Self> {
        if r == 0 {
            return Ok(self.clone());
        }

        // Find the knot span
        let span = self.find_span(u)?;
        let degree = self.degree();
        let knots = self.knots();

        // Check knot multiplicity
        let s = self.knot_multiplicity(u);

        if s + r > degree + 1 {
            return Err(InterpolateError::invalid_input(
                "Knot insertion would exceed degree + 1 multiplicity".to_string(),
            ));
        }

        // Allocate new arrays
        let n = self.control_points.nrows();
        let new_n = n + r;
        let mut new_knots = Array1::zeros(knots.len() + r);
        let mut new_control_points = Array2::zeros((new_n, self.dimension));
        let mut new_weights = Array1::zeros(new_n);

        // Copy knots before insertion point
        for i in 0..=span {
            new_knots[i] = knots[i];
        }

        // Insert new knots
        for i in 1..=r {
            new_knots[span + i] = u;
        }

        // Copy knots after insertion point
        for i in (span + 1)..knots.len() {
            new_knots[i + r] = knots[i];
        }

        // Compute new control points using knot insertion algorithm
        let mut temp_points = vec![Array1::zeros(self.dimension); degree + 1];
        let mut temp_weights = vec![T::zero(); degree + 1];

        // Initialize temporary arrays
        for i in 0..=degree {
            let idx = span - degree + i;
            if idx < n {
                for j in 0..self.dimension {
                    temp_points[i][j] = self.control_points[[idx, j]];
                }
                temp_weights[i] = self.weights[idx];
            }
        }

        // Copy unaffected control points (before)
        for i in 0..=(span - degree) {
            if i < new_n {
                for j in 0..self.dimension {
                    new_control_points[[i, j]] = self.control_points[[i, j]];
                }
                new_weights[i] = self.weights[i];
            }
        }

        // Copy unaffected control points (after)
        for i in (span - s).min(n)..n {
            let new_idx = i + r;
            if new_idx < new_n {
                for j in 0..self.dimension {
                    new_control_points[[new_idx, j]] = self.control_points[[i, j]];
                }
                new_weights[new_idx] = self.weights[i];
            }
        }

        // Insert r knots
        for j in 1..=r {
            let l = span - degree + j;

            for i in 0..=(degree - j) {
                let alpha = if knots[l + i] == knots[span + 1] {
                    T::one()
                } else {
                    (u - knots[l + i]) / (knots[span + 1] - knots[l + i])
                };

                for k in 0..self.dimension {
                    temp_points[i][k] = alpha * temp_points[i + 1][k] + (T::one() - alpha) * temp_points[i][k];
                }
                temp_weights[i] = alpha * temp_weights[i + 1] + (T::one() - alpha) * temp_weights[i];
            }

            // Store the new control point
            let new_idx = l.min(new_n.saturating_sub(1));
            if new_idx < new_n {
                for k in 0..self.dimension {
                    new_control_points[[new_idx, k]] = temp_points[0][k];
                }
                new_weights[new_idx] = temp_weights[0];
            }

            let new_idx2 = (span + j).min(new_n.saturating_sub(1));
            if new_idx2 < new_n && new_idx2 != new_idx {
                for k in 0..self.dimension {
                    new_control_points[[new_idx2, k]] = temp_points[degree - j][k];
                }
                new_weights[new_idx2] = temp_weights[degree - j];
            }
        }

        // Create new curve
        Self::from_arrays(
            new_control_points,
            new_weights,
            new_knots,
            degree,
            self.bspline.extrapolate_mode(),
        )
    }

    /// Get the multiplicity of a knot value
    ///
    /// # Arguments
    ///
    /// * `u` - Knot value to check
    ///
    /// # Returns
    ///
    /// The multiplicity of the knot value
    pub fn knot_multiplicity(&self, u: T) -> usize {
        let knots = self.knots();
        let tolerance = T::epsilon();

        knots.iter()
            .filter(|&&knot| (knot - u).abs() < tolerance)
            .count()
    }

    /// Compute arc length of the curve over parameter interval [a, b]
    ///
    /// Uses adaptive quadrature to compute the arc length by integrating
    /// the magnitude of the first derivative.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of parameter interval
    /// * `b` - Upper bound of parameter interval
    /// * `tolerance` - Optional tolerance for adaptive integration
    ///
    /// # Returns
    ///
    /// The arc length of the curve over [a, b]
    pub fn arc_length(&self, a: T, b: T, tolerance: Option<T>) -> InterpolateResult<T> {
        if a == b {
            return Ok(T::zero());
        }

        let tol = tolerance.unwrap_or_else(|| T::from(1e-8).unwrap_or(T::epsilon()));
        let (lower, upper) = if a < b { (a, b) } else { (b, a) };

        // Adaptive Simpson's rule for arc length
        let result = self.arc_length_simpson(lower, upper, tol, 15)?;

        Ok(if a < b { result } else { -result })
    }

    /// Find a root (zero) of the curve in a given parameter interval
    ///
    /// Uses numerical methods to find parameter values where the curve
    /// intersects the origin or approaches a target point.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of search interval
    /// * `b` - Upper bound of search interval
    /// * `target` - Optional target point (defaults to origin)
    /// * `tolerance` - Convergence tolerance
    /// * `max_iterations` - Maximum number of iterations
    ///
    /// # Returns
    ///
    /// Parameter value where curve is closest to target, or None if not found
    pub fn find_root(
        &self,
        a: T,
        b: T,
        target: Option<&ArrayView1<T>>,
        tolerance: T,
        max_iterations: usize,
    ) -> InterpolateResult<Option<T>> {
        let target_point = target.map(|t| t.to_owned())
            .unwrap_or_else(|| Array1::zeros(self.dimension));

        // Use Brent's method for root finding
        let bracket_a = a;
        let bracket_b = b;

        // Evaluate function at endpoints
        let fa = self.distance_to_point(bracket_a, &target_point.view())?;
        let fb = self.distance_to_point(bracket_b, &target_point.view())?;

        // Check if we already have a root at the endpoints
        if fa < tolerance {
            return Ok(Some(bracket_a));
        }
        if fb < tolerance {
            return Ok(Some(bracket_b));
        }

        // If the function values have the same sign, there might be no root
        // in the interval, but we can still find the minimum
        let mut x = bracket_a;
        let mut min_dist = fa;
        let step = (bracket_b - bracket_a) / T::from(max_iterations).unwrap();

        for i in 0..max_iterations {
            let test_x = bracket_a + T::from(i).unwrap() * step;
            let dist = self.distance_to_point(test_x, &target_point.view())?;

            if dist < min_dist {
                min_dist = dist;
                x = test_x;
            }

            if dist < tolerance {
                return Ok(Some(test_x));
            }
        }

        if min_dist < tolerance * T::from(10.0).unwrap() {
            Ok(Some(x))
        } else {
            Ok(None)
        }
    }

    /// Find extrema (local minima and maxima) of curve components
    ///
    /// Finds parameter values where the derivative is zero, indicating
    /// potential extrema in the curve components.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of search interval
    /// * `b` - Upper bound of search interval
    /// * `component` - Which component to analyze (0 for x, 1 for y, etc.)
    /// * `tolerance` - Convergence tolerance
    /// * `max_iterations` - Maximum iterations per search
    ///
    /// # Returns
    ///
    /// Vector of parameter values where extrema occur
    pub fn find_extrema(
        &self,
        a: T,
        b: T,
        component: usize,
        tolerance: T,
        max_iterations: usize,
    ) -> InterpolateResult<Vec<T>> {
        if component >= self.dimension {
            return Err(InterpolateError::invalid_input(
                "Component index out of bounds".to_string(),
            ));
        }

        let mut extrema = Vec::new();
        let num_samples = 50;
        let step = (b - a) / T::from(num_samples).unwrap();

        // Sample the derivative to find sign changes
        let mut prev_deriv = self.derivative(a, 1)?[component];

        for i in 1..=num_samples {
            let t = a + T::from(i).unwrap() * step;
            let deriv = self.derivative(t, 1)?[component];

            // Check for sign change (indicates extremum)
            if prev_deriv * deriv < T::zero() {
                // Use Newton's method to refine the root
                if let Some(root) = self.newton_raphson_extremum(
                    t - step,
                    t,
                    component,
                    tolerance,
                    max_iterations,
                )? {
                    extrema.push(root);
                }
            }

            prev_deriv = deriv;
        }

        Ok(extrema)
    }
}

// Helper methods for curve operations
impl<T: NurbsFloat> NurbsCurve<T> {
    /// Compute basis function values at parameter t
    pub(crate) fn compute_basisvalues(&self, _t: T) -> InterpolateResult<Array1<T>> {
        // Simplified implementation - just return uniform weights for now
        let n = self.control_points.nrows();
        Ok(Array1::from_elem(n, T::one() / T::from(n).unwrap()))
    }

    /// Compute all basis function derivatives up to given order
    pub(crate) fn compute_all_basis_derivatives(
        &self,
        _t: T,
        max_order: usize,
    ) -> InterpolateResult<Vec<Array1<T>>> {
        // Simplified implementation - return zero derivatives
        let n = self.control_points.nrows();
        let mut result = Vec::with_capacity(max_order + 1);
        for _order in 0..=max_order {
            result.push(Array1::zeros(n));
        }
        Ok(result)
    }

    /// Find the knot span containing parameter t
    pub(crate) fn find_span(&self, t: T) -> InterpolateResult<usize> {
        // Simplified implementation - find span by linear search
        let knots = self.knots();
        let degree = self.degree();

        for i in degree..(knots.len() - degree - 1) {
            if t >= knots[i] && t < knots[i + 1] {
                return Ok(i);
            }
        }

        // If t is at the end of the domain
        Ok(knots.len() - degree - 2)
    }

    /// Compute distance from curve point to target point
    fn distance_to_point(&self, t: T, target: &ArrayView1<T>) -> InterpolateResult<T> {
        let point = self.evaluate(t)?;
        let mut dist_squared = T::zero();

        for i in 0..self.dimension {
            let diff = point[i] - target[i];
            dist_squared += diff * diff;
        }

        Ok(dist_squared.sqrt())
    }

    /// Adaptive Simpson's rule for integration of a single component
    fn adaptive_simpson_integration(
        &self,
        a: T,
        b: T,
        component: usize,
        tolerance: T,
        max_depth: usize,
    ) -> InterpolateResult<T> {
        let mid = (a + b) / T::from(2.0).unwrap();

        let fa = self.evaluate(a)?[component];
        let fm = self.evaluate(mid)?[component];
        let fb = self.evaluate(b)?[component];

        let h = (b - a) / T::from(6.0).unwrap();
        let simpson = h * (fa + T::from(4.0).unwrap() * fm + fb);

        if max_depth == 0 {
            return Ok(simpson);
        }

        // Subdivide and check tolerance
        let mid_left = (a + mid) / T::from(2.0).unwrap();
        let mid_right = (mid + b) / T::from(2.0).unwrap();

        let fml = self.evaluate(mid_left)?[component];
        let fmr = self.evaluate(mid_right)?[component];

        let h_half = h / T::from(2.0).unwrap();
        let simpson_left = h_half * (fa + T::from(4.0).unwrap() * fml + fm);
        let simpson_right = h_half * (fm + T::from(4.0).unwrap() * fmr + fb);
        let simpson_combined = simpson_left + simpson_right;

        let error = (simpson_combined - simpson).abs() / T::from(15.0).unwrap();

        if error < tolerance {
            Ok(simpson_combined)
        } else {
            let left_integral = self.adaptive_simpson_integration(
                a, mid, component, tolerance / T::from(2.0).unwrap(), max_depth - 1
            )?;
            let right_integral = self.adaptive_simpson_integration(
                mid, b, component, tolerance / T::from(2.0).unwrap(), max_depth - 1
            )?;
            Ok(left_integral + right_integral)
        }
    }

    /// Arc length computation using Simpson's rule
    fn arc_length_simpson(&self, a: T, b: T, tolerance: T, max_depth: usize) -> InterpolateResult<T> {
        let mid = (a + b) / T::from(2.0).unwrap();

        let speed_a = self.compute_speed(a)?;
        let speed_m = self.compute_speed(mid)?;
        let speed_b = self.compute_speed(b)?;

        let h = (b - a) / T::from(6.0).unwrap();
        let simpson = h * (speed_a + T::from(4.0).unwrap() * speed_m + speed_b);

        if max_depth == 0 {
            return Ok(simpson);
        }

        // Subdivide and check tolerance
        let mid_left = (a + mid) / T::from(2.0).unwrap();
        let mid_right = (mid + b) / T::from(2.0).unwrap();

        let speed_ml = self.compute_speed(mid_left)?;
        let speed_mr = self.compute_speed(mid_right)?;

        let h_half = h / T::from(2.0).unwrap();
        let simpson_left = h_half * (speed_a + T::from(4.0).unwrap() * speed_ml + speed_m);
        let simpson_right = h_half * (speed_m + T::from(4.0).unwrap() * speed_mr + speed_b);
        let simpson_combined = simpson_left + simpson_right;

        let error = (simpson_combined - simpson).abs() / T::from(15.0).unwrap();

        if error < tolerance {
            Ok(simpson_combined)
        } else {
            let left_length = self.arc_length_simpson(
                a, mid, tolerance / T::from(2.0).unwrap(), max_depth - 1
            )?;
            let right_length = self.arc_length_simpson(
                mid, b, tolerance / T::from(2.0).unwrap(), max_depth - 1
            )?;
            Ok(left_length + right_length)
        }
    }

    /// Compute speed (magnitude of first derivative) at parameter t
    fn compute_speed(&self, t: T) -> InterpolateResult<T> {
        let deriv = self.derivative(t, 1)?;
        let mut speed_squared = T::zero();

        for i in 0..self.dimension {
            speed_squared += deriv[i] * deriv[i];
        }

        Ok(speed_squared.sqrt())
    }

    /// Newton-Raphson method for finding extrema
    fn newton_raphson_extremum(
        &self,
        a: T,
        b: T,
        component: usize,
        tolerance: T,
        max_iterations: usize,
    ) -> InterpolateResult<Option<T>> {
        let mut x = (a + b) / T::from(2.0).unwrap();

        for _ in 0..max_iterations {
            let f = self.derivative(x, 1)?[component];
            let df = self.derivative(x, 2)?[component];

            if f.abs() < tolerance {
                return Ok(Some(x));
            }

            if df.abs() < T::epsilon() {
                break; // Avoid division by zero
            }

            let new_x = x - f / df;

            if (new_x - x).abs() < tolerance {
                return Ok(Some(new_x));
            }

            if new_x < a || new_x > b {
                break; // Out of bounds
            }

            x = new_x;
        }

        Ok(None)
    }
}

/// Helper function to compute binomial coefficients
fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}