use ndarray::ArrayView1;
use num_traits::Float;

use crate::error::{InterpolateError, InterpolateResult};
use crate::ExtrapolateMode;

/// Enhanced boundary handling modes for interpolation.
///
/// This enum provides specialized boundary handling methods that go beyond
/// basic extrapolation. These methods are particularly useful for scientific
/// and engineering applications where specific boundary behaviors are required.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMode {
    /// Extrapolate according to the specified ExtrapolateMode
    Extrapolate(ExtrapolateMode),

    /// Zero-gradient (Neumann) boundary condition: df/dx = 0 at the boundary
    ZeroGradient,

    /// Zero-value (Dirichlet) boundary condition: f = 0 at the boundary
    ZeroValue,

    /// Linear extension with explicit gradient
    LinearGradient(f64),

    /// Cyclic/periodic boundary (wraps around)
    Periodic,

    /// Symmetric/reflection boundary (mirror image across boundary)
    Symmetric,

    /// Antisymmetric boundary (mirror image with sign change)
    Antisymmetric,

    /// Custom function for boundary handling
    /// (stored as function index in a registry)
    Custom(usize),
}

/// Parameters for enhancing boundary handling
#[derive(Debug, Clone)]
pub struct BoundaryParameters<T: Float> {
    /// Lower boundary of the domain
    lower_bound: T,

    /// Upper boundary of the domain
    upper_bound: T,

    /// Boundary handling mode for the lower boundary
    lower_mode: BoundaryMode,

    /// Boundary handling mode for the upper boundary
    upper_mode: BoundaryMode,

    /// Value at the lower boundary (for some modes)
    lower_value: Option<T>,

    /// Value at the upper boundary (for some modes)
    upper_value: Option<T>,

    /// Derivative at the lower boundary (for some modes)
    lower_derivative: Option<T>,

    /// Derivative at the upper boundary (for some modes)
    upper_derivative: Option<T>,

    /// Custom functions registry for Custom boundary mode
    /// (not actually stored here, referenced by index)
    custom_functions: Vec<usize>,
}

impl<T: Float + std::fmt::Display> BoundaryParameters<T> {
    /// Get the custom functions registry
    pub fn custom_functions(&self) -> &Vec<usize> {
        &self.custom_functions
    }
}

impl<T: Float + std::fmt::Display> BoundaryParameters<T> {
    /// Creates new boundary parameters with the specified modes.
    ///
    /// # Arguments
    ///
    /// * `lower_bound` - Lower boundary of the domain
    /// * `upper_bound` - Upper boundary of the domain
    /// * `lower_mode` - Boundary handling mode for the lower boundary
    /// * `upper_mode` - Boundary handling mode for the upper boundary
    ///
    /// # Returns
    ///
    /// A new `BoundaryParameters` instance
    pub fn new(
        lower_bound: T,
        upper_bound: T,
        lower_mode: BoundaryMode,
        upper_mode: BoundaryMode,
    ) -> Self {
        Self {
            lower_bound,
            upper_bound,
            lower_mode,
            upper_mode,
            lower_value: None,
            upper_value: None,
            lower_derivative: None,
            upper_derivative: None,
            custom_functions: Vec::new(),
        }
    }

    /// Sets the values at the boundaries.
    ///
    /// # Arguments
    ///
    /// * `lower_value` - Value at the lower boundary
    /// * `upper_value` - Value at the upper boundary
    ///
    /// # Returns
    ///
    /// A reference to the modified parameters
    pub fn with_values(mut self, lower_value: T, upper_value: T) -> Self {
        self.lower_value = Some(lower_value);
        self.upper_value = Some(upper_value);
        self
    }

    /// Sets the derivatives at the boundaries.
    ///
    /// # Arguments
    ///
    /// * `lower_derivative` - Derivative at the lower boundary
    /// * `upper_derivative` - Derivative at the upper boundary
    ///
    /// # Returns
    ///
    /// A reference to the modified parameters
    pub fn with_derivatives(mut self, lower_derivative: T, upper_derivative: T) -> Self {
        self.lower_derivative = Some(lower_derivative);
        self.upper_derivative = Some(upper_derivative);
        self
    }

    /// Maps a point outside the domain to an equivalent point according to boundary conditions.
    ///
    /// # Arguments
    ///
    /// * `x` - The point to map
    /// * `values` - Function values at domain points (for some modes)
    /// * `domain_points` - The domain points corresponding to the values
    ///
    /// # Returns
    ///
    /// Either the mapped point or a direct value, depending on the boundary mode
    pub fn map_point(
        &self,
        x: T,
        values: Option<&ArrayView1<T>>,
        domain_points: Option<&ArrayView1<T>>,
    ) -> InterpolateResult<BoundaryResult<T>> {
        // Check if point is inside the domain
        if x >= self.lower_bound && x <= self.upper_bound {
            return Ok(BoundaryResult::InsideDomain(x));
        }

        // Handle based on which boundary we're crossing
        if x < self.lower_bound {
            self.map_lower_boundary(x, values, domain_points)
        } else {
            self.map_upper_boundary(x, values, domain_points)
        }
    }

    /// Maps a point below the lower boundary according to the specified mode.
    fn map_lower_boundary(
        &self,
        x: T,
        values: Option<&ArrayView1<T>>,
        domain_points: Option<&ArrayView1<T>>,
    ) -> InterpolateResult<BoundaryResult<T>> {
        match self.lower_mode {
            BoundaryMode::Extrapolate(mode) => {
                match mode {
                    ExtrapolateMode::Error => Err(InterpolateError::OutOfBounds(format!(
                        "Point {} is below the lower boundary {}",
                        x, self.lower_bound
                    ))),
                    ExtrapolateMode::Extrapolate => {
                        // Allow extrapolation - return the original point
                        Ok(BoundaryResult::AllowExtrapolation(x))
                    }
                    ExtrapolateMode::Nan => {
                        // Return NaN for points outside the interpolation domain
                        Ok(BoundaryResult::DirectValue(T::nan()))
                    }
                    ExtrapolateMode::Constant => {
                        // Use the value at the lower boundary
                        if let Some(val) = self.lower_value {
                            Ok(BoundaryResult::DirectValue(val))
                        } else if let (Some(vals), Some(points)) = (values, domain_points) {
                            // Find the value at the lower boundary
                            let idx = self.find_nearest_point_index(self.lower_bound, points)?;
                            Ok(BoundaryResult::DirectValue(vals[idx]))
                        } else {
                            Err(InterpolateError::InvalidState(
                                "Values or domain points not provided for Constant mode"
                                    .to_string(),
                            ))
                        }
                    }
                }
            }
            BoundaryMode::ZeroGradient => {
                // Zero gradient means use the boundary value
                Ok(BoundaryResult::MappedPoint(self.lower_bound))
            }
            BoundaryMode::ZeroValue => {
                // Zero value means return 0
                Ok(BoundaryResult::DirectValue(T::zero()))
            }
            BoundaryMode::LinearGradient(gradient) => {
                // Linear extension with specified gradient
                let dx = x - self.lower_bound;
                let gradient_t = T::from(gradient).ok_or_else(|| {
                    InterpolateError::InvalidValue(
                        "Could not convert gradient to generic type".to_string(),
                    )
                })?;

                if let Some(lower_val) = self.lower_value {
                    // Use provided boundary value
                    Ok(BoundaryResult::DirectValue(lower_val + gradient_t * dx))
                } else if let (Some(vals), Some(points)) = (values, domain_points) {
                    // Find the value at the lower boundary
                    let idx = self.find_nearest_point_index(self.lower_bound, points)?;
                    let lower_val = vals[idx];
                    Ok(BoundaryResult::DirectValue(lower_val + gradient_t * dx))
                } else {
                    Err(InterpolateError::InvalidState(
                        "Values or domain points not provided for LinearGradient mode".to_string(),
                    ))
                }
            }
            BoundaryMode::Periodic => {
                // Periodic boundary means wrap around
                let domain_width = self.upper_bound - self.lower_bound;
                let offset = self.lower_bound - x;
                let periods = (offset / domain_width).ceil();
                let x_mapped = x + periods * domain_width;

                // Handle numerical precision issues
                if x_mapped < self.lower_bound {
                    Ok(BoundaryResult::MappedPoint(self.lower_bound))
                } else if x_mapped > self.upper_bound {
                    Ok(BoundaryResult::MappedPoint(self.upper_bound))
                } else {
                    Ok(BoundaryResult::MappedPoint(x_mapped))
                }
            }
            BoundaryMode::Symmetric => {
                // Symmetric boundary means reflect across the boundary
                let offset = self.lower_bound - x;
                let x_mapped = self.lower_bound + offset;

                // Handle multiple reflections
                if x_mapped > self.upper_bound {
                    // Need to reflect again
                    let new_offset = x_mapped - self.upper_bound;
                    let x_mapped = self.upper_bound - new_offset;

                    // Recursive case for multiple reflections
                    if x_mapped < self.lower_bound {
                        self.map_point(x_mapped, values, domain_points)
                    } else {
                        Ok(BoundaryResult::MappedPoint(x_mapped))
                    }
                } else {
                    Ok(BoundaryResult::MappedPoint(x_mapped))
                }
            }
            BoundaryMode::Antisymmetric => {
                // Similar to symmetric, but the function value changes sign
                let offset = self.lower_bound - x;
                let x_mapped = self.lower_bound + offset;

                if let (Some(vals), Some(points)) = (values, domain_points) {
                    // Find the value at the mapped point
                    let mapped_index = self.find_nearest_point_index(x_mapped, points)?;
                    let mapped_value = vals[mapped_index];

                    // Negate the value for antisymmetric reflection
                    Ok(BoundaryResult::DirectValue(-mapped_value))
                } else {
                    // Can't determine value without function values
                    Ok(BoundaryResult::MappedPointWithSignChange(x_mapped))
                }
            }
            BoundaryMode::Custom(func_idx) => {
                // Custom function is not directly implemented here,
                // it's expected to be handled by the caller
                Ok(BoundaryResult::CustomHandling(func_idx, x))
            }
        }
    }

    /// Maps a point above the upper boundary according to the specified mode.
    fn map_upper_boundary(
        &self,
        x: T,
        values: Option<&ArrayView1<T>>,
        domain_points: Option<&ArrayView1<T>>,
    ) -> InterpolateResult<BoundaryResult<T>> {
        match self.upper_mode {
            BoundaryMode::Extrapolate(mode) => {
                match mode {
                    ExtrapolateMode::Error => Err(InterpolateError::OutOfBounds(format!(
                        "Point {} is above the upper boundary {}",
                        x, self.upper_bound
                    ))),
                    ExtrapolateMode::Extrapolate => {
                        // Allow extrapolation - return the original point
                        Ok(BoundaryResult::AllowExtrapolation(x))
                    }
                    ExtrapolateMode::Nan => {
                        // Return NaN for points outside the interpolation domain
                        Ok(BoundaryResult::DirectValue(T::nan()))
                    }
                    ExtrapolateMode::Constant => {
                        // Use the value at the upper boundary
                        if let Some(val) = self.upper_value {
                            Ok(BoundaryResult::DirectValue(val))
                        } else if let (Some(vals), Some(points)) = (values, domain_points) {
                            // Find the value at the upper boundary
                            let idx = self.find_nearest_point_index(self.upper_bound, points)?;
                            Ok(BoundaryResult::DirectValue(vals[idx]))
                        } else {
                            Err(InterpolateError::InvalidState(
                                "Values or domain points not provided for Constant mode"
                                    .to_string(),
                            ))
                        }
                    }
                }
            }
            BoundaryMode::ZeroGradient => {
                // Zero gradient means use the boundary value
                Ok(BoundaryResult::MappedPoint(self.upper_bound))
            }
            BoundaryMode::ZeroValue => {
                // Zero value means return 0
                Ok(BoundaryResult::DirectValue(T::zero()))
            }
            BoundaryMode::LinearGradient(gradient) => {
                // Linear extension with specified gradient
                let dx = x - self.upper_bound;
                let gradient_t = T::from(gradient).ok_or_else(|| {
                    InterpolateError::InvalidValue(
                        "Could not convert gradient to generic type".to_string(),
                    )
                })?;

                if let Some(upper_val) = self.upper_value {
                    // Use provided boundary value
                    Ok(BoundaryResult::DirectValue(upper_val + gradient_t * dx))
                } else if let (Some(vals), Some(points)) = (values, domain_points) {
                    // Find the value at the upper boundary
                    let idx = self.find_nearest_point_index(self.upper_bound, points)?;
                    let upper_val = vals[idx];
                    Ok(BoundaryResult::DirectValue(upper_val + gradient_t * dx))
                } else {
                    Err(InterpolateError::InvalidState(
                        "Values or domain points not provided for LinearGradient mode".to_string(),
                    ))
                }
            }
            BoundaryMode::Periodic => {
                // Periodic boundary means wrap around
                let domain_width = self.upper_bound - self.lower_bound;
                let offset = x - self.lower_bound;
                let periods = (offset / domain_width).floor();
                let x_mapped = x - periods * domain_width;

                // Handle numerical precision issues
                if x_mapped < self.lower_bound {
                    Ok(BoundaryResult::MappedPoint(self.lower_bound))
                } else if x_mapped > self.upper_bound {
                    Ok(BoundaryResult::MappedPoint(self.upper_bound))
                } else {
                    Ok(BoundaryResult::MappedPoint(x_mapped))
                }
            }
            BoundaryMode::Symmetric => {
                // Symmetric boundary means reflect across the boundary
                let offset = x - self.upper_bound;
                let x_mapped = self.upper_bound - offset;

                // Handle multiple reflections
                if x_mapped < self.lower_bound {
                    // Need to reflect again
                    let new_offset = self.lower_bound - x_mapped;
                    let x_mapped = self.lower_bound + new_offset;

                    // Recursive case for multiple reflections
                    if x_mapped > self.upper_bound {
                        self.map_point(x_mapped, values, domain_points)
                    } else {
                        Ok(BoundaryResult::MappedPoint(x_mapped))
                    }
                } else {
                    Ok(BoundaryResult::MappedPoint(x_mapped))
                }
            }
            BoundaryMode::Antisymmetric => {
                // Similar to symmetric, but the function value changes sign
                let offset = x - self.upper_bound;
                let x_mapped = self.upper_bound - offset;

                if let (Some(vals), Some(points)) = (values, domain_points) {
                    // Find the value at the mapped point
                    let mapped_index = self.find_nearest_point_index(x_mapped, points)?;
                    let mapped_value = vals[mapped_index];

                    // Negate the value for antisymmetric reflection
                    Ok(BoundaryResult::DirectValue(-mapped_value))
                } else {
                    // Can't determine value without function values
                    Ok(BoundaryResult::MappedPointWithSignChange(x_mapped))
                }
            }
            BoundaryMode::Custom(func_idx) => {
                // Custom function is not directly implemented here,
                // it's expected to be handled by the caller
                Ok(BoundaryResult::CustomHandling(func_idx, x))
            }
        }
    }

    /// Finds the index of the nearest point in the domain to the given point.
    fn find_nearest_point_index(
        &self,
        point: T,
        domain_points: &ArrayView1<T>,
    ) -> InterpolateResult<usize> {
        let mut nearest_idx = 0;
        let mut min_distance = T::infinity();

        for (i, &p) in domain_points.iter().enumerate() {
            let distance = (p - point).abs();
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        Ok(nearest_idx)
    }

    /// Get the lower boundary mode.
    pub fn get_lower_mode(&self) -> BoundaryMode {
        self.lower_mode
    }

    /// Get the upper boundary mode.
    pub fn get_upper_mode(&self) -> BoundaryMode {
        self.upper_mode
    }

    /// Set the lower boundary mode.
    pub fn set_lower_mode(&mut self, mode: BoundaryMode) {
        self.lower_mode = mode;
    }

    /// Set the upper boundary mode.
    pub fn set_upper_mode(&mut self, mode: BoundaryMode) {
        self.upper_mode = mode;
    }
}

/// The result of applying a boundary condition
#[derive(Debug, Clone)]
pub enum BoundaryResult<T: Float> {
    /// Point is inside the domain, no boundary handling needed
    InsideDomain(T),

    /// Point has been mapped to an equivalent point inside the domain
    MappedPoint(T),

    /// Point has been mapped to an equivalent point inside the domain,
    /// and the function value should be negated
    MappedPointWithSignChange(T),

    /// Extrapolation is allowed for this point
    AllowExtrapolation(T),

    /// A direct function value is returned (no interpolation needed)
    DirectValue(T),

    /// Custom handling is required (function index and original point)
    CustomHandling(usize, T),
}

/// Creates boundary parameters with zero-gradient boundary conditions.
///
/// Zero-gradient (Neumann) boundary conditions set df/dx = 0 at the boundaries,
/// which is useful for physical problems where there's no flux across boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
///
/// # Returns
///
/// A new `BoundaryParameters` instance with zero-gradient boundary conditions
pub fn make_zero_gradient_boundary<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> BoundaryParameters<T> {
    BoundaryParameters::new(
        lower_bound,
        upper_bound,
        BoundaryMode::ZeroGradient,
        BoundaryMode::ZeroGradient,
    )
}

/// Creates boundary parameters with zero-value boundary conditions.
///
/// Zero-value (Dirichlet) boundary conditions set f = 0 at the boundaries,
/// which is useful for physical problems where the quantity vanishes at boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
///
/// # Returns
///
/// A new `BoundaryParameters` instance with zero-value boundary conditions
pub fn make_zero_value_boundary<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> BoundaryParameters<T> {
    BoundaryParameters::new(
        lower_bound,
        upper_bound,
        BoundaryMode::ZeroValue,
        BoundaryMode::ZeroValue,
    )
}

/// Creates boundary parameters with periodic boundary conditions.
///
/// Periodic boundary conditions treat the function as if it repeats outside
/// the domain, which is useful for analyzing periodic phenomena.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
///
/// # Returns
///
/// A new `BoundaryParameters` instance with periodic boundary conditions
pub fn make_periodic_boundary<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> BoundaryParameters<T> {
    BoundaryParameters::new(
        lower_bound,
        upper_bound,
        BoundaryMode::Periodic,
        BoundaryMode::Periodic,
    )
}

/// Creates boundary parameters with symmetric boundary conditions.
///
/// Symmetric boundary conditions reflect the function across the boundaries,
/// which is useful for problems with symmetry about the boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
///
/// # Returns
///
/// A new `BoundaryParameters` instance with symmetric boundary conditions
pub fn make_symmetric_boundary<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> BoundaryParameters<T> {
    BoundaryParameters::new(
        lower_bound,
        upper_bound,
        BoundaryMode::Symmetric,
        BoundaryMode::Symmetric,
    )
}

/// Creates boundary parameters with antisymmetric boundary conditions.
///
/// Antisymmetric boundary conditions reflect the function across the boundaries
/// with a sign change, which is useful for odd functions.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
///
/// # Returns
///
/// A new `BoundaryParameters` instance with antisymmetric boundary conditions
pub fn make_antisymmetric_boundary<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> BoundaryParameters<T> {
    BoundaryParameters::new(
        lower_bound,
        upper_bound,
        BoundaryMode::Antisymmetric,
        BoundaryMode::Antisymmetric,
    )
}

/// Creates boundary parameters with linear gradient boundary conditions.
///
/// Linear gradient boundary conditions extend the function linearly with
/// the specified gradients at boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the domain
/// * `upper_bound` - Upper boundary of the domain
/// * `lower_gradient` - Gradient at the lower boundary
/// * `upper_gradient` - Gradient at the upper boundary
///
/// # Returns
///
/// A new `BoundaryParameters` instance with linear gradient boundary conditions
pub fn make_linear_gradient_boundary<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_gradient: f64,
    upper_gradient: f64,
) -> BoundaryParameters<T> {
    BoundaryParameters::new(
        lower_bound,
        upper_bound,
        BoundaryMode::LinearGradient(lower_gradient),
        BoundaryMode::LinearGradient(upper_gradient),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_zero_gradient_boundary() {
        let boundary = make_zero_gradient_boundary(0.0, 10.0);

        // Test point below lower boundary
        let result = boundary.map_point(-5.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 0.0),
            _ => panic!("Expected MappedPoint result"),
        }

        // Test point above upper boundary
        let result = boundary.map_point(15.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 10.0),
            _ => panic!("Expected MappedPoint result"),
        }
    }

    #[test]
    fn test_zero_value_boundary() {
        let boundary = make_zero_value_boundary(0.0, 10.0);

        // Test point below lower boundary
        let result = boundary.map_point(-5.0, None, None).unwrap();
        match result {
            BoundaryResult::DirectValue(v) => assert_abs_diff_eq!(v, 0.0),
            _ => panic!("Expected DirectValue result"),
        }

        // Test point above upper boundary
        let result = boundary.map_point(15.0, None, None).unwrap();
        match result {
            BoundaryResult::DirectValue(v) => assert_abs_diff_eq!(v, 0.0),
            _ => panic!("Expected DirectValue result"),
        }
    }

    #[test]
    fn test_periodic_boundary() {
        let boundary = make_periodic_boundary(0.0, 10.0);

        // Test point below lower boundary
        let result = boundary.map_point(-5.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPoint result"),
        }

        // Test point above upper boundary
        let result = boundary.map_point(15.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPoint result"),
        }

        // Test multiple periods
        let result = boundary.map_point(25.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPoint result"),
        }
    }

    #[test]
    fn test_symmetric_boundary() {
        let boundary = make_symmetric_boundary(0.0, 10.0);

        // Test point below lower boundary
        let result = boundary.map_point(-5.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPoint result"),
        }

        // Test point above upper boundary
        let result = boundary.map_point(15.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPoint result"),
        }

        // Test multiple reflections
        let result = boundary.map_point(-15.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPoint(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPoint result"),
        }
    }

    #[test]
    fn test_antisymmetric_boundary() {
        let boundary = make_antisymmetric_boundary(0.0, 10.0);

        // Test point below lower boundary
        let result = boundary.map_point(-5.0, None, None).unwrap();
        match result {
            BoundaryResult::MappedPointWithSignChange(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected MappedPointWithSignChange result"),
        }

        // Test with provided values
        let domain_points = Array::linspace(0.0, 10.0, 11);
        let values = domain_points.mapv(|v| v * v);

        let result = boundary
            .map_point(-5.0, Some(&values.view()), Some(&domain_points.view()))
            .unwrap();
        match result {
            BoundaryResult::DirectValue(v) => {
                let expected = -(5.0 * 5.0); // Negative of f(5.0)
                assert_abs_diff_eq!(v, expected);
            }
            _ => panic!("Expected DirectValue result"),
        }
    }

    #[test]
    fn test_linear_gradient_boundary() {
        let boundary = make_linear_gradient_boundary(0.0, 10.0, 2.0, -3.0);

        // Test with provided values
        let domain_points = Array::linspace(0.0, 10.0, 11);
        let values = domain_points.mapv(|v| v * v);

        // Test point below lower boundary
        let result = boundary
            .map_point(-5.0, Some(&values.view()), Some(&domain_points.view()))
            .unwrap();

        match result {
            BoundaryResult::DirectValue(v) => {
                let expected = 0.0 + 2.0 * (-5.0); // f(0) + gradient * dx
                assert_abs_diff_eq!(v, expected);
            }
            _ => panic!("Expected DirectValue result"),
        }

        // Test point above upper boundary
        let result = boundary
            .map_point(15.0, Some(&values.view()), Some(&domain_points.view()))
            .unwrap();

        match result {
            BoundaryResult::DirectValue(v) => {
                let expected = 100.0 + (-3.0) * (15.0 - 10.0); // f(10) + gradient * dx
                assert_abs_diff_eq!(v, expected);
            }
            _ => panic!("Expected DirectValue result"),
        }
    }

    #[test]
    fn test_inside_domain() {
        let boundary = make_zero_gradient_boundary(0.0, 10.0);

        // Test point inside domain
        let result = boundary.map_point(5.0, None, None).unwrap();
        match result {
            BoundaryResult::InsideDomain(x) => assert_abs_diff_eq!(x, 5.0),
            _ => panic!("Expected InsideDomain result"),
        }
    }

    #[test]
    fn test_extrapolate_boundary() {
        let boundary = BoundaryParameters::new(
            0.0,
            10.0,
            BoundaryMode::Extrapolate(ExtrapolateMode::Extrapolate),
            BoundaryMode::Extrapolate(ExtrapolateMode::Extrapolate),
        );

        // Test point below lower boundary
        let result = boundary.map_point(-5.0, None, None).unwrap();
        match result {
            BoundaryResult::AllowExtrapolation(x) => assert_abs_diff_eq!(x, -5.0),
            _ => panic!("Expected AllowExtrapolation result"),
        }

        // Test point above upper boundary
        let result = boundary.map_point(15.0, None, None).unwrap();
        match result {
            BoundaryResult::AllowExtrapolation(x) => assert_abs_diff_eq!(x, 15.0),
            _ => panic!("Expected AllowExtrapolation result"),
        }
    }
}
