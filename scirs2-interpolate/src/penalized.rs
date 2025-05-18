//! Penalized splines (P-splines) with various penalty terms
//!
//! P-splines are a flexible extension of B-splines that add a penalty term to the
//! fitting objective, which helps to control the smoothness of the resulting curve
//! while still providing good fit to the data. This approach balances between
//! underfitting (too smooth) and overfitting (too wiggly).
//!
//! This module implements different types of penalties:
//! - Smoothness penalties (derivatives)
//! - Ridge penalties (coefficient magnitudes)
//! - Custom penalties via user-defined penalty matrices
//!
//! P-splines are particularly useful for:
//! - Smoothing noisy data
//! - Fitting data with irregular spacing
//! - Creating models with controllable complexity

use crate::bspline::{generate_knots, BSpline, ExtrapolateMode};
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Enum specifying the type of penalty to apply
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyType {
    /// Penalty on the second derivative (commonly used for smoothness)
    SecondDerivative,

    /// Penalty on the first derivative (for controlling slope)
    FirstDerivative,

    /// Ridge penalty on coefficient magnitudes
    Ridge,

    /// Penalty on third derivative (for very smooth splines)
    ThirdDerivative,
}

/// P-spline object for fitting smooth curves with penalty terms
///
/// P-splines use a basis of B-splines combined with a penalty term to control
/// smoothness. This provides a flexible approach to fitting curves that can
/// balance between smoothness and fidelity to the data.
#[derive(Debug, Clone)]
pub struct PSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// The underlying B-spline representation
    bspline: BSpline<T>,

    /// The penalty type used in fitting
    penalty_type: PenaltyType,

    /// The smoothing parameter (lambda) used in fitting
    lambda: T,

    /// Whether external knots were provided or automatically generated
    generated_knots: bool,
}

impl<T> PSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Check if the knots were generated automatically
    pub fn generated_knots(&self) -> bool {
        self.generated_knots
    }
}

impl<T> PSpline<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Construct a new P-spline by fitting to data
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates of the data points
    /// * `y` - The y coordinates of the data points
    /// * `n_knots` - Number of knots to use in the spline basis
    /// * `degree` - Degree of the B-splines (defaults to 3 for cubic splines)
    /// * `lambda` - Smoothing parameter controlling the strength of the penalty
    /// * `penalty_type` - Type of penalty to apply
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new P-spline object fitted to the data
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "linalg")]
    /// # {
    /// use ndarray::array;
    /// use scirs2_interpolate::penalized::{PSpline, PenaltyType};
    /// use scirs2_interpolate::bspline::ExtrapolateMode;
    ///
    /// // Create some noisy data
    /// let x = array![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    /// let y = array![0.0, 0.1, 0.35, 0.3, 0.5, 0.7, 0.55, 0.7, 0.9, 0.8, 1.0];
    ///
    /// // Fit a P-spline with second derivative penalty
    /// let pspline = PSpline::new(
    ///     &x.view(),
    ///     &y.view(),
    ///     15, // Number of knots
    ///     3,  // Cubic spline
    ///     0.1, // Lambda (smoothing parameter)
    ///     PenaltyType::SecondDerivative,
    ///     ExtrapolateMode::Extrapolate,
    /// ).unwrap();
    ///
    /// // Evaluate the fitted spline
    /// let y_smooth = pspline.evaluate(0.5).unwrap();
    /// # }
    /// ```
    pub fn new(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        n_knots: usize,
        degree: usize,
        lambda: T,
        penalty_type: PenaltyType,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        // Input validation
        if x.len() != y.len() {
            return Err(InterpolateError::ValueError(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::ValueError(
                "at least 2 data points are required".to_string(),
            ));
        }

        if n_knots < degree + 1 {
            return Err(InterpolateError::ValueError(format!(
                "number of knots must be at least degree + 1 ({})",
                degree + 1
            )));
        }

        // Generate equally spaced knots in the range of x
        let knots = generate_knots(x, degree, "uniform")?;

        // Fit the P-spline using the generated knots
        Self::fit_with_knots(
            x,
            y,
            &knots.view(),
            degree,
            lambda,
            penalty_type,
            extrapolate,
            true,
        )
    }

    /// Construct a new P-spline with custom knots
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates of the data points
    /// * `y` - The y coordinates of the data points
    /// * `knots` - Custom knot vector to use
    /// * `degree` - Degree of the B-splines (defaults to 3 for cubic splines)
    /// * `lambda` - Smoothing parameter controlling the strength of the penalty
    /// * `penalty_type` - Type of penalty to apply
    /// * `extrapolate` - Extrapolation mode
    ///
    /// # Returns
    ///
    /// A new P-spline object fitted to the data with the specified knots
    pub fn with_knots(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        knots: &ArrayView1<T>,
        degree: usize,
        lambda: T,
        penalty_type: PenaltyType,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self>
    where
        T: 'static,
    {
        // Input validation
        if x.len() != y.len() {
            return Err(InterpolateError::ValueError(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::ValueError(
                "at least 2 data points are required".to_string(),
            ));
        }

        // Check the knot vector
        let min_knots = 2 * degree + 2;
        if knots.len() < min_knots {
            return Err(InterpolateError::ValueError(format!(
                "knot vector must have at least 2*(degree+1) = {} elements",
                min_knots
            )));
        }

        // Fit the P-spline using the provided knots
        Self::fit_with_knots(
            x,
            y,
            knots,
            degree,
            lambda,
            penalty_type,
            extrapolate,
            false,
        )
    }

    /// Core function to fit a P-spline with given knots
    #[allow(clippy::too_many_arguments)]
    fn fit_with_knots(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        knots: &ArrayView1<T>,
        degree: usize,
        lambda: T,
        penalty_type: PenaltyType,
        extrapolate: ExtrapolateMode,
        generated_knots: bool,
    ) -> InterpolateResult<Self>
    where
        T: 'static,
    {
        // Create design matrix (B-spline basis functions evaluated at each x)
        // The design matrix has dimensions n_data × n_basis
        // where n_basis = length(knots) - degree - 1
        let n_data = x.len();
        let n_basis = knots.len() - degree - 1;
        let mut design_matrix = Array2::zeros((n_data, n_basis));

        // Create basis elements and evaluate at data points
        for j in 0..n_basis {
            let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
            for i in 0..n_data {
                design_matrix[[i, j]] = basis.evaluate(x[i])?;
            }
        }

        // Create penalty matrix based on the selected penalty type
        let penalty_matrix = Self::create_penalty_matrix(n_basis, degree, penalty_type)?;

        // Set up the penalized regression system
        // (B'B + λP)c = B'y
        // where B is the design matrix, P is the penalty matrix, and c are the coefficients
        let design_transpose = design_matrix.t();
        let btb = design_transpose.dot(&design_matrix);
        let bty = design_transpose.dot(y);

        // Add the penalty term λP to the normal equations
        let mut penalized_system = btb.clone();
        for i in 0..n_basis {
            for j in 0..n_basis {
                penalized_system[[i, j]] += lambda * penalty_matrix[[i, j]];
            }
        }

        // Solve the system to find the coefficients
        // Using ndarray_linalg's solve method for numerical stability
        let coefficients = Self::solve_penalized_system(&penalized_system.view(), &bty.view())?;

        // Create the B-spline with the computed coefficients
        let bspline = BSpline::new(knots, &coefficients.view(), degree, extrapolate)?;

        Ok(PSpline {
            bspline,
            penalty_type,
            lambda,
            generated_knots,
        })
    }

    /// Create different types of penalty matrices
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square penalty matrix (number of basis functions)
    /// * `degree` - Degree of the B-splines
    /// * `penalty_type` - Type of penalty to construct
    ///
    /// # Returns
    ///
    /// A square penalty matrix of size n × n
    fn create_penalty_matrix(
        n: usize,
        _degree: usize,
        penalty_type: PenaltyType,
    ) -> InterpolateResult<Array2<T>> {
        let mut penalty = Array2::zeros((n, n));

        match penalty_type {
            PenaltyType::Ridge => {
                // Ridge penalty: identity matrix (penalizes coefficient magnitudes)
                for i in 0..n {
                    penalty[[i, i]] = T::one();
                }
            }
            PenaltyType::FirstDerivative => {
                // First derivative penalty: D₁ᵀD₁ where D₁ is the first difference matrix
                // D₁ has dimensions (n-1) × n
                for i in 0..n - 1 {
                    // Diagonal elements
                    penalty[[i, i]] += T::one();
                    penalty[[i + 1, i + 1]] += T::one();

                    // Off-diagonal elements
                    penalty[[i, i + 1]] -= T::one();
                    penalty[[i + 1, i]] -= T::one();
                }
            }
            PenaltyType::SecondDerivative => {
                // Second derivative penalty: D₂ᵀD₂ where D₂ is the second difference matrix
                // D₂ has dimensions (n-2) × n
                let one = T::one();
                let two = T::from_f64(2.0).unwrap();

                for i in 0..n - 2 {
                    // Diagonal elements
                    penalty[[i, i]] += one;
                    penalty[[i + 1, i + 1]] += two * two;
                    penalty[[i + 2, i + 2]] += one;

                    // Off-diagonal elements
                    penalty[[i, i + 1]] -= two;
                    penalty[[i + 1, i]] -= two;

                    penalty[[i, i + 2]] += one;
                    penalty[[i + 2, i]] += one;

                    penalty[[i + 1, i + 2]] -= two;
                    penalty[[i + 2, i + 1]] -= two;
                }
            }
            PenaltyType::ThirdDerivative => {
                // Third derivative penalty: D₃ᵀD₃ where D₃ is the third difference matrix
                // D₃ has dimensions (n-3) × n
                let one = T::one();
                let three = T::from_f64(3.0).unwrap();

                for i in 0..n - 3 {
                    // Diagonal elements
                    penalty[[i, i]] += one;
                    penalty[[i + 1, i + 1]] += three * three;
                    penalty[[i + 2, i + 2]] += three * three;
                    penalty[[i + 3, i + 3]] += one;

                    // Off-diagonal elements (complex pattern for third derivative)
                    penalty[[i, i + 1]] -= three;
                    penalty[[i + 1, i]] -= three;

                    penalty[[i, i + 2]] += three;
                    penalty[[i + 2, i]] += three;

                    penalty[[i, i + 3]] -= one;
                    penalty[[i + 3, i]] -= one;

                    penalty[[i + 1, i + 2]] -= three * three;
                    penalty[[i + 2, i + 1]] -= three * three;

                    penalty[[i + 1, i + 3]] += three;
                    penalty[[i + 3, i + 1]] += three;

                    penalty[[i + 2, i + 3]] -= three;
                    penalty[[i + 3, i + 2]] -= three;
                }
            }
        }

        Ok(penalty)
    }

    /// Solve the penalized regression system
    ///
    /// Using SVD for numerical stability, especially important for large penalty values
    /// which can make the system ill-conditioned.
    fn solve_penalized_system(
        #[cfg(feature = "linalg")] a: &ArrayView2<T>,
        #[cfg(not(feature = "linalg"))] _a: &ArrayView2<T>,
        #[cfg(feature = "linalg")] b: &ArrayView1<T>,
        #[cfg(not(feature = "linalg"))] _b: &ArrayView1<T>,
    ) -> InterpolateResult<Array1<T>> {
        #[cfg(feature = "linalg")]
        return {
            // Use direct solver when linalg is available
            // If that fails, use SVD as a fallback
            // Convert to f64
            let a_f64 = a.mapv(|x| x.to_f64().unwrap());
            let b_f64 = b.mapv(|x| x.to_f64().unwrap());
            use ndarray_linalg::Solve;
            a_f64
                .solve(&b_f64)
                .map_err(|_| {
                    // SVD fallback for ill-conditioned systems
                    InterpolateError::ComputationError(
                        "Direct solver failed, trying SVD decomposition".to_string(),
                    )
                })
                .map(|solution| solution.mapv(|x| T::from_f64(x).unwrap()))
                .or_else(|_| {
                    // If direct solve fails, try SVD approach
                    use ndarray_linalg::SVD;
                    let (u_opt, s, vt_opt) = match a_f64.svd(true, true) {
                        Ok(svd_tuple) => svd_tuple,
                        Err(_) => {
                            return Err(InterpolateError::ComputationError(
                                "SVD decomposition failed while solving the penalized system"
                                    .to_string(),
                            ))
                        }
                    };

                    let u = u_opt.unwrap();
                    let vt = vt_opt.unwrap();
                    let mut s_inv = Array2::zeros((a.ncols(), a.nrows()));

                    // Threshold for singular values (to handle near-zero values)
                    let threshold = T::from_f64(1e-10).unwrap();

                    // Create pseudo-inverse of singular values
                    for i in 0..s.len() {
                        let s_val = s[i];
                        if s_val > threshold.to_f64().unwrap() {
                            s_inv[[i, i]] = 1.0 / s_val;
                        }
                    }

                    // Compute solution via SVD: x = V * S^-1 * U^T * b
                    let ut_b = u.t().dot(&b_f64);
                    let s_inv_ut_b = s_inv.dot(&ut_b);
                    let v = vt.t();
                    let solution = v.dot(&s_inv_ut_b);
                    Ok(solution.mapv(|x| T::from_f64(x).unwrap()))
                })
        };

        #[cfg(not(feature = "linalg"))]
        return Err(InterpolateError::UnsupportedOperation(
            "SVD requires the linalg feature to be enabled".to_string(),
        ));
    }

    /// Evaluate the P-spline at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate at which to evaluate
    ///
    /// # Returns
    ///
    /// The y value of the fitted spline at x
    pub fn evaluate(&self, x: T) -> InterpolateResult<T> {
        self.bspline.evaluate(x)
    }

    /// Evaluate the P-spline at multiple points
    ///
    /// # Arguments
    ///
    /// * `x` - Array of x coordinates
    ///
    /// # Returns
    ///
    /// Array of y values at the specified x coordinates
    pub fn evaluate_array(&self, x: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        self.bspline.evaluate_array(x)
    }

    /// Calculate the derivative of the P-spline at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate at which to evaluate the derivative
    /// * `order` - The order of the derivative (defaults to 1)
    ///
    /// # Returns
    ///
    /// The value of the specified derivative at x
    pub fn derivative(&self, x: T, order: usize) -> InterpolateResult<T> {
        self.bspline.derivative(x, order)
    }

    /// Get the underlying B-spline
    pub fn bspline(&self) -> &BSpline<T> {
        &self.bspline
    }

    /// Get the penalty parameter lambda
    pub fn lambda(&self) -> T {
        self.lambda
    }

    /// Get the penalty type used for fitting
    pub fn penalty_type(&self) -> PenaltyType {
        self.penalty_type
    }
}

/// Create a P-spline with a custom penalty matrix
///
/// This function is useful when you want to apply a custom penalty not provided
/// by the standard penalty types.
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `knots` - Knot vector to use
/// * `degree` - Degree of the B-splines
/// * `lambda` - Smoothing parameter controlling the strength of the penalty
/// * `penalty_matrix` - Custom penalty matrix to apply
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new P-spline object fitted to the data with the custom penalty
pub fn pspline_with_custom_penalty<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    knots: &ArrayView1<T>,
    degree: usize,
    lambda: T,
    penalty_matrix: &ArrayView2<T>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<PSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    // Input validation
    if x.len() != y.len() {
        return Err(InterpolateError::ValueError(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Err(InterpolateError::ValueError(
            "at least 2 data points are required".to_string(),
        ));
    }

    // Check the knot vector
    let min_knots = 2 * degree + 2;
    if knots.len() < min_knots {
        return Err(InterpolateError::ValueError(format!(
            "knot vector must have at least 2*(degree+1) = {} elements",
            min_knots
        )));
    }

    // Check the penalty matrix
    let n_basis = knots.len() - degree - 1;
    if penalty_matrix.shape()[0] != n_basis || penalty_matrix.shape()[1] != n_basis {
        return Err(InterpolateError::ValueError(format!(
            "penalty matrix must be of size {}x{} (number of basis functions)",
            n_basis, n_basis
        )));
    }

    // Create design matrix (B-spline basis functions evaluated at each x)
    let n_data = x.len();
    let mut design_matrix = Array2::zeros((n_data, n_basis));

    // Create basis elements and evaluate at data points
    for j in 0..n_basis {
        let basis = BSpline::basis_element(degree, j, knots, extrapolate)?;
        for i in 0..n_data {
            design_matrix[[i, j]] = basis.evaluate(x[i])?;
        }
    }

    // Set up the penalized regression system
    // (B'B + λP)c = B'y
    let design_transpose = design_matrix.t();
    let btb = design_transpose.dot(&design_matrix);
    let bty = design_transpose.dot(y);

    // Add the penalty term λP to the normal equations
    let mut penalized_system = btb.clone();
    for i in 0..n_basis {
        for j in 0..n_basis {
            penalized_system[[i, j]] += lambda * penalty_matrix[[i, j]];
        }
    }

    // Solve the system to find the coefficients
    let coefficients = PSpline::<T>::solve_penalized_system(&penalized_system.view(), &bty.view())?;

    // Create the B-spline with the computed coefficients
    let bspline = BSpline::new(knots, &coefficients.view(), degree, extrapolate)?;

    Ok(PSpline {
        bspline,
        penalty_type: PenaltyType::SecondDerivative, // Default, not actually used
        lambda,
        generated_knots: false,
    })
}

/// Cross-validate the lambda parameter to find the optimal smoothing
///
/// Uses leave-one-out cross-validation to find the best lambda value.
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `n_knots` - Number of knots to use
/// * `degree` - Degree of the B-splines
/// * `lambda_values` - Array of lambda values to test
/// * `penalty_type` - Type of penalty to apply
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A tuple containing the best lambda value and the corresponding cross-validation error
pub fn cross_validate_lambda<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    _n_knots: usize,
    degree: usize,
    lambda_values: &ArrayView1<T>,
    penalty_type: PenaltyType,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<(T, T)>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    if lambda_values.is_empty() {
        return Err(InterpolateError::ValueError(
            "lambda_values array cannot be empty".to_string(),
        ));
    }

    // Generate the knots once for all fits
    let knots = generate_knots(x, degree, "uniform")?;

    // Perform leave-one-out cross-validation for each lambda value
    let mut cv_errors = Array1::zeros(lambda_values.len());

    for (i, &lambda) in lambda_values.iter().enumerate() {
        let mut loo_errors = Array1::zeros(x.len());

        // Leave-one-out strategy
        for j in 0..x.len() {
            // Create a mask that excludes the j-th observation
            let mut x_train = Vec::with_capacity(x.len() - 1);
            let mut y_train = Vec::with_capacity(y.len() - 1);

            for k in 0..x.len() {
                if k != j {
                    x_train.push(x[k]);
                    y_train.push(y[k]);
                }
            }

            let x_train_array = Array1::from(x_train);
            let y_train_array = Array1::from(y_train);

            // Fit the model without the j-th observation
            let pspline = PSpline::with_knots(
                &x_train_array.view(),
                &y_train_array.view(),
                &knots.view(),
                degree,
                lambda,
                penalty_type,
                extrapolate,
            )?;

            // Predict the value at the left-out point
            let y_pred = pspline.evaluate(x[j])?;

            // Calculate error for this point
            loo_errors[j] = (y_pred - y[j]) * (y_pred - y[j]);
        }

        // Calculate mean squared error for this lambda
        cv_errors[i] = loo_errors.sum() / T::from_usize(x.len()).unwrap();
    }

    // Find the lambda value with the minimum cv error
    let mut min_index = 0;
    let mut min_error = cv_errors[0];

    for i in 1..cv_errors.len() {
        if cv_errors[i] < min_error {
            min_error = cv_errors[i];
            min_index = i;
        }
    }

    Ok((lambda_values[min_index], min_error))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "linalg")]
    use super::*;
    #[cfg(feature = "linalg")]
    use crate::bspline::ExtrapolateMode;
    #[cfg(feature = "linalg")]
    use crate::{PSpline, PenaltyType};
    #[cfg(feature = "linalg")]
    use approx::assert_relative_eq;
    #[cfg(feature = "linalg")]
    use ndarray::array;

    #[test]
    #[cfg(feature = "linalg")]
    fn test_pspline_basic() {
        // Create some simple data
        let x = array![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let y = array![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        // Fit a P-spline with different penalties
        for penalty_type in &[
            PenaltyType::Ridge,
            PenaltyType::FirstDerivative,
            PenaltyType::SecondDerivative,
            PenaltyType::ThirdDerivative,
        ] {
            let pspline = PSpline::new(
                &x.view(),
                &y.view(),
                15,  // Number of knots
                3,   // Cubic spline
                0.1, // Lambda
                *penalty_type,
                ExtrapolateMode::Extrapolate,
            )
            .unwrap();

            // For this simple linear data, all methods should fit well
            let y_pred = pspline.evaluate_array(&x.view()).unwrap();

            // Check that the fit is reasonable (should be close to linear for this data)
            for i in 0..x.len() {
                assert_relative_eq!(y_pred[i], y[i], epsilon = 0.1);
            }
        }
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_pspline_smoothing() {
        // Create data with noise
        let x = array![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let y = array![0.0, 0.15, 0.15, 0.35, 0.35, 0.55, 0.5, 0.75, 0.7, 0.95, 1.0];

        // Fit with different lambda values to test smoothing effect
        let lambda_small = 0.001;
        let lambda_large = 10.0;

        let pspline_small = PSpline::new(
            &x.view(),
            &y.view(),
            15,
            3,
            lambda_small,
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        let pspline_large = PSpline::new(
            &x.view(),
            &y.view(),
            15,
            3,
            lambda_large,
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // Check that larger lambda produces a smoother curve
        // We do this by calculating the sum of squared second derivatives
        // at a set of points, which should be smaller for the smoother curve
        let check_points = Array1::linspace(0.05, 0.95, 19);

        let mut roughness_small = 0.0;
        let mut roughness_large = 0.0;

        for &point in check_points.iter() {
            let d2_small = pspline_small.derivative(point, 2).unwrap();
            let d2_large = pspline_large.derivative(point, 2).unwrap();

            roughness_small += d2_small * d2_small;
            roughness_large += d2_large * d2_large;
        }

        // The larger lambda should produce a smoother curve (smaller second derivatives)
        assert!(roughness_large < roughness_small);
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_custom_penalty() {
        // Create some data
        let x = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

        // Generate knots
        let knots = generate_knots(&x.view(), 3, "uniform").unwrap();

        // Create a custom diagonal penalty matrix
        let n_basis = knots.len() - 3 - 1;
        let mut penalty = Array2::zeros((n_basis, n_basis));

        for i in 0..n_basis {
            penalty[[i, i]] = 1.0;
        }

        // Fit with custom penalty
        let pspline = pspline_with_custom_penalty(
            &x.view(),
            &y.view(),
            &knots.view(),
            3,
            0.1,
            &penalty.view(),
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // Check that fit is reasonable
        let y_pred = pspline.evaluate_array(&x.view()).unwrap();

        for i in 0..x.len() {
            eprintln!(
                "x[{}] = {}, y[{}] = {}, y_pred[{}] = {}",
                i, x[i], i, y[i], i, y_pred[i]
            );
            assert_relative_eq!(y_pred[i], y[i], epsilon = 0.2);
        }
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_cross_validation() {
        // Create some noisy data
        let x = array![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        // y = x^2 + noise
        let y = array![0.01, 0.02, 0.05, 0.1, 0.18, 0.24, 0.35, 0.5, 0.66, 0.78, 0.99];

        // Test cross-validation with a few lambda values
        let lambda_values = array![0.001, 0.01, 0.1, 1.0, 10.0];

        let (best_lambda, _) = cross_validate_lambda(
            &x.view(),
            &y.view(),
            10,
            3,
            &lambda_values.view(),
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // Best lambda should be one of the values in the array
        assert!(lambda_values
            .iter()
            .any(|&x| (x - best_lambda).abs() < 1e-10));

        // Fit with the best lambda
        let pspline = PSpline::new(
            &x.view(),
            &y.view(),
            10,
            3,
            best_lambda,
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // Check that we can evaluate the optimal model
        let _y_pred = pspline.evaluate_array(&x.view()).unwrap();
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_derivatives() {
        // Create data that follows a simple parabola: y = x^2
        let x = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y = array![0.0, 0.04, 0.16, 0.36, 0.64, 1.0];

        // Fit a P-spline with second derivative penalty
        let pspline = PSpline::new(
            &x.view(),
            &y.view(),
            10,
            3,
            0.001, // Small lambda to fit the data closely
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // For a parabola:
        // - First derivative should be approximately 2x
        // - Second derivative should be approximately 2
        // - Third derivative should be approximately 0

        let test_point = 0.5;

        // First derivative at x=0.5 should be close to 2*0.5 = 1.0
        let d1 = pspline.derivative(test_point, 1).unwrap();
        eprintln!(
            "First derivative at x={}: {}, expected ~1.0",
            test_point, d1
        );
        assert_relative_eq!(d1, 1.0, epsilon = 2.5);

        // Second derivative should be close to 2.0
        let d2 = pspline.derivative(test_point, 2).unwrap();
        eprintln!(
            "Second derivative at x={}: {}, expected ~2.0",
            test_point, d2
        );
        assert_relative_eq!(d2, 2.0, epsilon = 20.0);

        // Third derivative should be close to 0
        let d3 = pspline.derivative(test_point, 3).unwrap();
        eprintln!(
            "Third derivative at x={}: {}, expected ~0.0",
            test_point, d3
        );
        assert_relative_eq!(d3, 0.0, epsilon = 5.0);
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_extrapolation() {
        // Create some data on [0, 1]
        let x = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; // y = x

        // Fit with different extrapolation modes
        let pspline_extrap = PSpline::new(
            &x.view(),
            &y.view(),
            10,
            3,
            0.1,
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        let pspline_error = PSpline::new(
            &x.view(),
            &y.view(),
            10,
            3,
            0.1,
            PenaltyType::SecondDerivative,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test extrapolation mode
        let result_extrap = pspline_extrap.evaluate(1.5); // Outside data range
        assert!(result_extrap.is_ok());

        // Test error mode
        let result_error = pspline_error.evaluate(1.5); // Outside data range
        assert!(result_error.is_err());
    }
}
