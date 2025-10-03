//! Core extrapolation functionality
//!
//! This module contains the main Extrapolator struct and its core implementation
//! for performing extrapolation using various methods.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};

use crate::error::{InterpolateError, InterpolateResult};

use super::config::ExtrapolationParameters;
use super::types::{ExtrapolationDirection, ExtrapolationMethod};

/// Extrapolator for extending interpolation methods beyond their domain.
///
/// This class provides a flexible way to extrapolate values outside the
/// original domain of interpolation, using a variety of methods that can be
/// customized separately for the lower and upper boundaries.
#[derive(Debug, Clone)]
pub struct Extrapolator<T: Float> {
    /// Lower boundary of the original domain
    pub(crate) lower_bound: T,

    /// Upper boundary of the original domain
    pub(crate) upper_bound: T,

    /// Extrapolation method for below the lower boundary
    pub(crate) lower_method: ExtrapolationMethod,

    /// Extrapolation method for above the upper boundary
    pub(crate) upper_method: ExtrapolationMethod,

    /// Value at the lower boundary
    pub(crate) lower_value: T,

    /// Value at the upper boundary
    pub(crate) upper_value: T,

    /// Derivative at the lower boundary
    pub(crate) lower_derivative: T,

    /// Derivative at the upper boundary
    pub(crate) upper_derivative: T,

    /// Second derivative at the lower boundary (for higher-order methods)
    pub(crate) lower_second_derivative: Option<T>,

    /// Second derivative at the upper boundary (for higher-order methods)
    pub(crate) upper_second_derivative: Option<T>,

    /// Parameters for specialized extrapolation models
    pub(crate) parameters: ExtrapolationParameters<T>,
}

impl<T: Float + std::fmt::Display> Extrapolator<T> {
    /// Creates a new extrapolator with the specified methods and boundary values.
    ///
    /// # Arguments
    ///
    /// * `lower_bound` - Lower boundary of the original domain
    /// * `upper_bound` - Upper boundary of the original domain
    /// * `lower_value` - Function value at the lower boundary
    /// * `upper_value` - Function value at the upper boundary
    /// * `lower_method` - Extrapolation method for below the lower boundary
    /// * `upper_method` - Extrapolation method for above the upper boundary
    ///
    /// # Returns
    ///
    /// A new `Extrapolator` instance
    pub fn new(
        lower_bound: T,
        upper_bound: T,
        lower_value: T,
        upper_value: T,
        lower_method: ExtrapolationMethod,
        upper_method: ExtrapolationMethod,
    ) -> Self {
        // For linear methods, calculate derivatives based on boundary values
        let slope = if upper_bound != lower_bound {
            (upper_value - lower_value) / (upper_bound - lower_bound)
        } else {
            T::zero()
        };

        let lower_derivative = match lower_method {
            ExtrapolationMethod::Linear => slope,
            _ => T::zero(),
        };

        let upper_derivative = match upper_method {
            ExtrapolationMethod::Linear => slope,
            _ => T::zero(),
        };

        Self {
            lower_bound,
            upper_bound,
            lower_method,
            upper_method,
            lower_value,
            upper_value,
            lower_derivative,
            upper_derivative,
            lower_second_derivative: None,
            upper_second_derivative: None,
            parameters: ExtrapolationParameters::default(),
        }
    }

    /// Sets the derivatives at the boundaries for gradient-aware extrapolation.
    ///
    /// # Arguments
    ///
    /// * `lower_derivative` - Derivative at the lower boundary
    /// * `upper_derivative` - Derivative at the upper boundary
    ///
    /// # Returns
    ///
    /// A reference to the modified extrapolator
    pub fn with_derivatives(mut self, lower_derivative: T, upper_derivative: T) -> Self {
        self.lower_derivative = lower_derivative;
        self.upper_derivative = upper_derivative;
        self
    }

    /// Sets the second derivatives at the boundaries for higher-order extrapolation.
    ///
    /// # Arguments
    ///
    /// * `lower_second_derivative` - Second derivative at the lower boundary
    /// * `upper_second_derivative` - Second derivative at the upper boundary
    ///
    /// # Returns
    ///
    /// A reference to the modified extrapolator
    pub fn with_second_derivatives(
        mut self,
        lower_second_derivative: T,
        upper_second_derivative: T,
    ) -> Self {
        self.lower_second_derivative = Some(lower_second_derivative);
        self.upper_second_derivative = Some(upper_second_derivative);
        self
    }

    /// Sets custom parameters for specialized extrapolation methods.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Custom parameters for extrapolation methods
    ///
    /// # Returns
    ///
    /// A reference to the modified extrapolator
    pub fn with_parameters(mut self, parameters: ExtrapolationParameters<T>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Extrapolates the function value at the given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the function
    ///
    /// # Returns
    ///
    /// The extrapolated function value
    pub fn extrapolate(&self, x: T) -> InterpolateResult<T> {
        if x < self.lower_bound {
            self.extrapolate_direction(x, ExtrapolationDirection::Lower)
        } else if x > self.upper_bound {
            self.extrapolate_direction(x, ExtrapolationDirection::Upper)
        } else {
            // Point is inside the domain, shouldn't be extrapolating
            Err(InterpolateError::InvalidValue(format!(
                "Point {} is inside the domain [{}, {}], use interpolation instead",
                x, self.lower_bound, self.upper_bound
            )))
        }
    }

    /// Extrapolates the function value in the specified direction.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the function
    /// * `direction` - Direction of extrapolation (lower or upper)
    ///
    /// # Returns
    ///
    /// The extrapolated function value
    fn extrapolate_direction(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let method = match direction {
            ExtrapolationDirection::Lower => self.lower_method,
            ExtrapolationDirection::Upper => self.upper_method,
        };

        match method {
            ExtrapolationMethod::Error => Err(InterpolateError::OutOfBounds(format!(
                "Point {} is outside the domain [{}, {}]",
                x, self.lower_bound, self.upper_bound
            ))),
            ExtrapolationMethod::Constant => match direction {
                ExtrapolationDirection::Lower => Ok(self.lower_value),
                ExtrapolationDirection::Upper => Ok(self.upper_value),
            },
            ExtrapolationMethod::Linear => self.linear_extrapolation(x, direction),
            ExtrapolationMethod::Quadratic => self.quadratic_extrapolation(x, direction),
            ExtrapolationMethod::Cubic => self.cubic_extrapolation(x, direction),
            ExtrapolationMethod::Periodic => self.periodic_extrapolation(x),
            ExtrapolationMethod::Reflection => self.reflection_extrapolation(x),
            ExtrapolationMethod::Exponential => self.exponential_extrapolation(x, direction),
            ExtrapolationMethod::PowerLaw => self.power_law_extrapolation(x, direction),
            ExtrapolationMethod::Spline => self.spline_extrapolation(x, direction),
            ExtrapolationMethod::Akima => self.akima_extrapolation(x, direction),
            ExtrapolationMethod::Sinusoidal => self.sinusoidal_extrapolation(x, direction),
            ExtrapolationMethod::Rational => self.rational_extrapolation(x, direction),
            ExtrapolationMethod::Confidence => self.confidence_extrapolation(x, direction),
            ExtrapolationMethod::Ensemble => self.ensemble_extrapolation(x, direction),
            ExtrapolationMethod::Adaptive => self.adaptive_extrapolation(x, direction),
            ExtrapolationMethod::Autoregressive => self.autoregressive_extrapolation(x, direction),
            ExtrapolationMethod::Zeros => Ok(T::zero()),
            ExtrapolationMethod::Nearest => self.nearest_extrapolation(x, direction),
            ExtrapolationMethod::Mirror => self.mirror_extrapolation(x, direction),
            ExtrapolationMethod::Wrap => self.wrap_extrapolation(x),
            ExtrapolationMethod::Clamped => self.clamped_extrapolation(x, direction),
            ExtrapolationMethod::GridMirror => self.grid_mirror_extrapolation(x, direction),
            ExtrapolationMethod::GridConstant => self.grid_constant_extrapolation(x, direction),
            ExtrapolationMethod::GridWrap => self.grid_wrap_extrapolation(x),
        }
    }

    /// Linear extrapolation based on endpoint values and derivatives.
    ///
    /// Uses the formula: f(x) = f(x₀) + f'(x₀) * (x - x₀)
    fn linear_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        match direction {
            ExtrapolationDirection::Lower => {
                let dx = x - self.lower_bound;
                Ok(self.lower_value + self.lower_derivative * dx)
            }
            ExtrapolationDirection::Upper => {
                let dx = x - self.upper_bound;
                Ok(self.upper_value + self.upper_derivative * dx)
            }
        }
    }

    /// Quadratic extrapolation based on endpoint values, derivatives, and curvature.
    ///
    /// Uses the formula: f(x) = f(x₀) + f'(x₀) * (x - x₀) + 0.5 * f''(x₀) * (x - x₀)²
    fn quadratic_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value, deriv, second_deriv) = match direction {
            ExtrapolationDirection::Lower => {
                let second_deriv = self.lower_second_derivative.ok_or_else(|| {
                    InterpolateError::InvalidState(
                        "Second derivative not provided for quadratic extrapolation".to_string(),
                    )
                })?;
                (
                    self.lower_bound,
                    self.lower_value,
                    self.lower_derivative,
                    second_deriv,
                )
            }
            ExtrapolationDirection::Upper => {
                let second_deriv = self.upper_second_derivative.ok_or_else(|| {
                    InterpolateError::InvalidState(
                        "Second derivative not provided for quadratic extrapolation".to_string(),
                    )
                })?;
                (
                    self.upper_bound,
                    self.upper_value,
                    self.upper_derivative,
                    second_deriv,
                )
            }
        };

        let dx = x - bound;
        let half = T::from(0.5).unwrap();

        Ok(value + deriv * dx + half * second_deriv * dx * dx)
    }

    /// Cubic extrapolation preserving both values and derivatives at boundaries.
    ///
    /// For lower boundary:
    /// - f(x_lower) = lower_value
    /// - f'(x_lower) = lower_derivative
    /// - The cubic polynomial is constructed to smoothly match these conditions
    fn cubic_extrapolation(&self, x: T, direction: ExtrapolationDirection) -> InterpolateResult<T> {
        // Cubic extrapolation requires second derivatives to be specified
        if self.lower_second_derivative.is_none() || self.upper_second_derivative.is_none() {
            return Err(InterpolateError::InvalidState(
                "Second derivatives must be provided for cubic extrapolation".to_string(),
            ));
        }

        let (bound, value, deriv, second_deriv) = match direction {
            ExtrapolationDirection::Lower => (
                self.lower_bound,
                self.lower_value,
                self.lower_derivative,
                self.lower_second_derivative.unwrap(),
            ),
            ExtrapolationDirection::Upper => (
                self.upper_bound,
                self.upper_value,
                self.upper_derivative,
                self.upper_second_derivative.unwrap(),
            ),
        };

        let dx = x - bound;
        let dx2 = dx * dx;
        let dx3 = dx2 * dx;

        // Coefficients for cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
        let a = value;
        let b = deriv;
        let c = second_deriv / T::from(2.0).unwrap();

        // The third coefficient (d) depends on the third derivative, which we don't have directly
        // Let's set it to a small value based on the rate of change of the second derivative
        let d = T::from(0.0).unwrap(); // Simplified version sets this to zero

        Ok(a + b * dx + c * dx2 + d * dx3)
    }

    /// Periodic extrapolation extending the domain as if the function repeats.
    ///
    /// Maps the point x to an equivalent point within the domain using modular arithmetic,
    /// effectively treating the function as periodic with period equal to the domain width.
    fn periodic_extrapolation(&self, x: T) -> InterpolateResult<T> {
        let domain_width = self.upper_bound - self.lower_bound;

        // If a custom period is specified, use that instead of the domain width
        let period = if self.parameters.period > T::zero() {
            self.parameters.period
        } else {
            domain_width
        };

        // Compute the equivalent position within the domain
        let mut x_equiv = x;

        // Handle points below the lower bound
        if x < self.lower_bound {
            let offset = self.lower_bound - x;
            let periods = (offset / period).ceil();
            x_equiv = x + periods * period;
        }
        // Handle points above the upper bound
        else if x > self.upper_bound {
            let offset = x - self.upper_bound;
            let periods = (offset / period).ceil();
            x_equiv = x - periods * period;
        }

        // For periodic extrapolation, we need to map the value
        // This is a simplified version - in practice you'd interpolate at the mapped position
        let mapped_pos = (x_equiv - self.lower_bound) / domain_width;
        let linear_interp =
            self.lower_value * (T::one() - mapped_pos) + self.upper_value * mapped_pos;

        Ok(linear_interp)
    }

    /// Reflection extrapolation that mirrors the function at the boundaries.
    fn reflection_extrapolation(&self, x: T) -> InterpolateResult<T> {
        let domain_width = self.upper_bound - self.lower_bound;

        if x < self.lower_bound {
            // Reflect around the lower boundary
            let offset = self.lower_bound - x;
            let x_reflected = self.lower_bound + offset;

            // If the reflected point is still outside the domain, use periodic reflection
            if x_reflected > self.upper_bound {
                let normalized_offset = offset % domain_width;
                let final_x = self.lower_bound + normalized_offset;
                let t = (final_x - self.lower_bound) / domain_width;
                Ok(self.lower_value * (T::one() - t) + self.upper_value * t)
            } else {
                let t = (x_reflected - self.lower_bound) / domain_width;
                Ok(self.lower_value * (T::one() - t) + self.upper_value * t)
            }
        } else if x > self.upper_bound {
            // Reflect around the upper boundary
            let offset = x - self.upper_bound;
            let x_reflected = self.upper_bound - offset;

            // If the reflected point is still outside the domain, use periodic reflection
            if x_reflected < self.lower_bound {
                let normalized_offset = offset % domain_width;
                let final_x = self.upper_bound - normalized_offset;
                let t = (final_x - self.lower_bound) / domain_width;
                Ok(self.lower_value * (T::one() - t) + self.upper_value * t)
            } else {
                let t = (x_reflected - self.lower_bound) / domain_width;
                Ok(self.lower_value * (T::one() - t) + self.upper_value * t)
            }
        } else {
            // Point is inside domain
            let t = (x - self.lower_bound) / domain_width;
            Ok(self.lower_value * (T::one() - t) + self.upper_value * t)
        }
    }

    /// Exponential extrapolation for asymptotic behavior.
    ///
    /// Uses the formula: f(x) = a * exp(r * (x - x₀)) + b
    fn exponential_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value) = match direction {
            ExtrapolationDirection::Lower => (self.lower_bound, self.lower_value),
            ExtrapolationDirection::Upper => (self.upper_bound, self.upper_value),
        };

        let dx = x - bound;
        let rate = self.parameters.exponential_rate;
        let offset = self.parameters.exponential_offset;

        // f(x) = value * exp(rate * dx) + offset
        let exp_term = (rate * dx).exp();
        Ok(value * exp_term + offset)
    }

    /// Power law extrapolation for algebraic decay/growth.
    ///
    /// Uses the formula: f(x) = a * (x - x₀)^p + b
    fn power_law_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value) = match direction {
            ExtrapolationDirection::Lower => (self.lower_bound, self.lower_value),
            ExtrapolationDirection::Upper => (self.upper_bound, self.upper_value),
        };

        let dx = x - bound;
        let exponent = self.parameters.power_exponent;
        let scale = self.parameters.power_scale;

        // Avoid issues with negative bases and fractional exponents
        if dx < T::zero() && exponent != exponent.floor() {
            return Err(InterpolateError::ComputationError(
                "Power law extrapolation with fractional exponent requires positive displacement"
                    .to_string(),
            ));
        }

        // f(x) = scale * dx^exponent + value
        let power_term = if dx == T::zero() {
            T::zero()
        } else {
            scale * dx.powf(exponent)
        };

        Ok(value + power_term)
    }

    // Placeholder implementations for advanced methods
    // These would be implemented in the methods.rs module in a real scenario

    fn spline_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would use spline continuation
        self.linear_extrapolation(_x, _direction)
    }

    fn akima_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would use Akima spline extrapolation
        self.linear_extrapolation(_x, _direction)
    }

    fn sinusoidal_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would use sinusoidal fitting
        self.linear_extrapolation(_x, _direction)
    }

    fn rational_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would use rational function extrapolation
        self.linear_extrapolation(_x, _direction)
    }

    fn confidence_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would include uncertainty estimation
        self.linear_extrapolation(_x, _direction)
    }

    fn ensemble_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would combine multiple methods
        self.linear_extrapolation(_x, _direction)
    }

    fn adaptive_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would adaptively select method
        self.linear_extrapolation(_x, _direction)
    }

    fn autoregressive_extrapolation(
        &self,
        _x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Placeholder - would use AR models
        self.linear_extrapolation(_x, _direction)
    }

    fn nearest_extrapolation(
        &self,
        _x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        match direction {
            ExtrapolationDirection::Lower => Ok(self.lower_value),
            ExtrapolationDirection::Upper => Ok(self.upper_value),
        }
    }

    fn mirror_extrapolation(
        &self,
        x: T,
        _direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        self.reflection_extrapolation(x)
    }

    fn wrap_extrapolation(&self, x: T) -> InterpolateResult<T> {
        self.periodic_extrapolation(x)
    }

    fn clamped_extrapolation(
        &self,
        _x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Clamped means zero derivative at boundaries
        match direction {
            ExtrapolationDirection::Lower => Ok(self.lower_value),
            ExtrapolationDirection::Upper => Ok(self.upper_value),
        }
    }

    fn grid_mirror_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Grid-specific mirror mode - similar to regular mirror
        self.mirror_extrapolation(x, direction)
    }

    fn grid_constant_extrapolation(
        &self,
        _x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Grid-specific constant mode
        self.nearest_extrapolation(_x, direction)
    }

    fn grid_wrap_extrapolation(&self, x: T) -> InterpolateResult<T> {
        // Grid-specific wrap mode
        self.wrap_extrapolation(x)
    }

    /// Get the lower bound of the extrapolator domain
    pub fn lower_bound(&self) -> T {
        self.lower_bound
    }

    /// Get the upper bound of the extrapolator domain
    pub fn upper_bound(&self) -> T {
        self.upper_bound
    }

    /// Get the extrapolation method for the lower boundary
    pub fn lower_method(&self) -> ExtrapolationMethod {
        self.lower_method
    }

    /// Get the extrapolation method for the upper boundary
    pub fn upper_method(&self) -> ExtrapolationMethod {
        self.upper_method
    }

    /// Check if the point is within the extrapolator's domain
    pub fn is_in_domain(&self, x: T) -> bool {
        x >= self.lower_bound && x <= self.upper_bound
    }

    /// Get the domain width
    pub fn domain_width(&self) -> T {
        self.upper_bound - self.lower_bound
    }
}
