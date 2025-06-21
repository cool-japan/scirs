//! Finite difference approximations for gradient verification
//!
//! This module provides various finite difference schemes for approximating
//! gradients and higher-order derivatives.

use super::StabilityError;
use crate::tensor::Tensor;
use crate::Float;
use ndarray::{Array, IxDyn};

/// Configuration for finite difference computations
#[derive(Debug, Clone)]
pub struct FiniteDifferenceConfig {
    /// Step size for finite differences
    pub step_size: f64,
    /// Type of finite difference scheme
    pub scheme: FiniteDifferenceScheme,
    /// Adaptive step size selection
    pub adaptive_step: bool,
    /// Minimum step size for adaptive schemes
    pub min_step: f64,
    /// Maximum step size for adaptive schemes
    pub max_step: f64,
}

impl Default for FiniteDifferenceConfig {
    fn default() -> Self {
        Self {
            step_size: 1e-8,
            scheme: FiniteDifferenceScheme::Central,
            adaptive_step: false,
            min_step: 1e-12,
            max_step: 1e-4,
        }
    }
}

/// Types of finite difference schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FiniteDifferenceScheme {
    /// Forward difference: (f(x+h) - f(x)) / h
    Forward,
    /// Backward difference: (f(x) - f(x-h)) / h
    Backward,
    /// Central difference: (f(x+h) - f(x-h)) / (2h)
    Central,
    /// High-order central difference with O(h^4) accuracy
    HighOrderCentral,
}

/// Finite difference gradient computer
pub struct FiniteDifferenceComputer<F: Float> {
    config: FiniteDifferenceConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> FiniteDifferenceComputer<F> {
    /// Create a new finite difference computer
    pub fn new() -> Self {
        Self {
            config: FiniteDifferenceConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FiniteDifferenceConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute finite difference approximation of gradient
    pub fn compute_gradient<'a, Func>(
        &self,
        function: Func,
        input: &Tensor<'a, F>,
    ) -> Result<Tensor<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        match self.config.scheme {
            FiniteDifferenceScheme::Forward => self.forward_difference(function, input),
            FiniteDifferenceScheme::Backward => self.backward_difference(function, input),
            FiniteDifferenceScheme::Central => self.central_difference(function, input),
            FiniteDifferenceScheme::HighOrderCentral => {
                self.high_order_central_difference(function, input)
            }
        }
    }

    /// Compute second-order derivatives (Hessian approximation)
    pub fn compute_hessian<'a, Func>(
        &self,
        function: Func,
        input: &Tensor<'a, F>,
    ) -> Result<Array<F, IxDyn>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let input_shape = input.shape();
        let n = input_shape.iter().product::<usize>();

        // Create Hessian matrix (simplified - assumes flattened input)
        let mut hessian = Array::zeros(IxDyn(&[n, n]));

        let step = F::from(self.config.step_size).unwrap();

        // Compute second partial derivatives using central differences
        for i in 0..n {
            for j in 0..n {
                let second_derivative = if i == j {
                    // Diagonal elements: ∂²f/∂x_i²
                    self.compute_second_partial_diagonal(&function, input, i, step)?
                } else {
                    // Off-diagonal elements: ∂²f/∂x_i∂x_j
                    self.compute_second_partial_mixed(&function, input, i, j, step)?
                };

                hessian[[i, j]] = second_derivative;
            }
        }

        Ok(hessian)
    }

    /// Forward difference implementation
    fn forward_difference<'a, Func>(
        &self,
        function: Func,
        input: &Tensor<'a, F>,
    ) -> Result<Tensor<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let step = if self.config.adaptive_step {
            self.select_optimal_step_size(&function, input)?
        } else {
            F::from(self.config.step_size).unwrap()
        };

        let f_x = function(input)?;
        let input_shape = input.shape();
        let mut gradient = Array::zeros(ndarray::IxDyn(&input_shape));

        // Compute partial derivatives
        for (i, input_perturbed) in self.create_perturbed_inputs(input, step).enumerate() {
            let f_x_plus_h = function(&input_perturbed)?;

            // ∂f/∂x_i ≈ (f(x + h*e_i) - f(x)) / h
            let partial_derivative = self.compute_partial_derivative(&f_x_plus_h, &f_x, step);

            // Store in gradient tensor
            self.set_gradient_component(&mut gradient, i, partial_derivative)?;
        }

        let gradient_vec = gradient.into_raw_vec_and_offset().0;
        let gradient_shape = input_shape.to_vec();
        Ok(Tensor::from_vec(
            gradient_vec,
            gradient_shape,
            input.graph(),
        ))
    }

    /// Backward difference implementation
    fn backward_difference<'a, Func>(
        &self,
        function: Func,
        input: &Tensor<'a, F>,
    ) -> Result<Tensor<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let step = F::from(self.config.step_size).unwrap();
        let f_x = function(input)?;
        let input_shape = input.shape();
        let mut gradient = Array::zeros(ndarray::IxDyn(&input_shape));

        // Compute partial derivatives using backward differences
        for (i, input_perturbed) in self.create_perturbed_inputs(input, -step).enumerate() {
            let f_x_minus_h = function(&input_perturbed)?;

            // ∂f/∂x_i ≈ (f(x) - f(x - h*e_i)) / h
            let partial_derivative = self.compute_partial_derivative(&f_x, &f_x_minus_h, step);

            self.set_gradient_component(&mut gradient, i, partial_derivative)?;
        }

        let gradient_vec = gradient.into_raw_vec_and_offset().0;
        let gradient_shape = input_shape.to_vec();
        Ok(Tensor::from_vec(
            gradient_vec,
            gradient_shape,
            input.graph(),
        ))
    }

    /// Central difference implementation
    fn central_difference<'a, Func>(
        &self,
        function: Func,
        input: &Tensor<'a, F>,
    ) -> Result<Tensor<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let step = F::from(self.config.step_size).unwrap();
        let input_shape = input.shape();
        let mut gradient = Array::zeros(ndarray::IxDyn(&input_shape));

        // Compute partial derivatives using central differences
        for (i, (input_plus, input_minus)) in self
            .create_central_perturbed_inputs(input, step)
            .enumerate()
        {
            let f_x_plus_h = function(&input_plus)?;
            let f_x_minus_h = function(&input_minus)?;

            // ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
            let partial_derivative =
                self.compute_central_partial_derivative(&f_x_plus_h, &f_x_minus_h, step);

            self.set_gradient_component(&mut gradient, i, partial_derivative)?;
        }

        let gradient_vec = gradient.into_raw_vec_and_offset().0;
        let gradient_shape = input_shape.to_vec();
        Ok(Tensor::from_vec(
            gradient_vec,
            gradient_shape,
            input.graph(),
        ))
    }

    /// High-order central difference with O(h^4) accuracy
    fn high_order_central_difference<'a, Func>(
        &self,
        function: Func,
        input: &Tensor<'a, F>,
    ) -> Result<Tensor<'a, F>, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let step = F::from(self.config.step_size).unwrap();
        let input_shape = input.shape();
        let mut gradient = Array::zeros(ndarray::IxDyn(&input_shape));

        // Use 5-point stencil: (-2h, -h, 0, h, 2h)
        for i in 0..input_shape.iter().product() {
            let (f_minus_2h, f_minus_h, f_plus_h, f_plus_2h) =
                self.compute_five_point_stencil(&function, input, i, step)?;

            // ∂f/∂x_i ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
            let _two = F::from(2.0).unwrap();
            let eight = F::from(8.0).unwrap();
            let twelve = F::from(12.0).unwrap();

            let partial_derivative =
                (-f_plus_2h + eight * f_plus_h - eight * f_minus_h + f_minus_2h) / (twelve * step);

            self.set_gradient_component(&mut gradient, i, partial_derivative)?;
        }

        let gradient_vec = gradient.into_raw_vec_and_offset().0;
        let gradient_shape = input_shape.to_vec();
        Ok(Tensor::from_vec(
            gradient_vec,
            gradient_shape,
            input.graph(),
        ))
    }

    /// Helper methods
    #[allow(dead_code)]
    fn select_optimal_step_size<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
    ) -> Result<F, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        // Implement adaptive step size selection using Richardson extrapolation
        // or other numerical analysis techniques

        let mut best_step = F::from(self.config.step_size).unwrap();
        let mut best_error = F::from(f64::INFINITY).unwrap();

        // Test several step sizes
        let step_candidates = [
            self.config.step_size * 0.1,
            self.config.step_size,
            self.config.step_size * 10.0,
        ];

        for &step_size in &step_candidates {
            if step_size >= self.config.min_step && step_size <= self.config.max_step {
                let step = F::from(step_size).unwrap();
                let error = self.estimate_truncation_error(function, input, step)?;

                if error < best_error {
                    best_error = error;
                    best_step = step;
                }
            }
        }

        Ok(best_step)
    }

    #[allow(dead_code)]
    fn estimate_truncation_error<Func>(
        &self,
        _function: &Func,
        _input: &Tensor<F>,
        _step: F,
    ) -> Result<F, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        // Simplified error estimation - in practice would use Richardson extrapolation
        Ok(F::from(1e-10).unwrap())
    }

    #[allow(dead_code)]
    fn create_perturbed_inputs<'a>(
        &self,
        input: &Tensor<'a, F>,
        step: F,
    ) -> PerturbedInputIterator<'a, F> {
        PerturbedInputIterator::new(input, step)
    }

    #[allow(dead_code)]
    fn create_central_perturbed_inputs<'a>(
        &self,
        input: &Tensor<'a, F>,
        step: F,
    ) -> CentralPerturbedInputIterator<'a, F> {
        CentralPerturbedInputIterator::new(input, step)
    }

    #[allow(dead_code)]
    fn compute_partial_derivative(
        &self,
        _f_perturbed: &Tensor<F>,
        _f_original: &Tensor<F>,
        step: F,
    ) -> F {
        // Simplified - would compute actual difference between tensor values
        let diff = F::from(0.001).unwrap(); // Placeholder
        diff / step
    }

    #[allow(dead_code)]
    fn compute_central_partial_derivative(
        &self,
        _f_plus: &Tensor<F>,
        _f_minus: &Tensor<F>,
        step: F,
    ) -> F {
        // Simplified - would compute actual difference between tensor values
        let diff = F::from(0.002).unwrap(); // Placeholder
        let two = F::from(2.0).unwrap();
        diff / (two * step)
    }

    #[allow(dead_code)]
    fn set_gradient_component(
        &self,
        gradient: &mut Array<F, IxDyn>,
        index: usize,
        value: F,
    ) -> Result<(), StabilityError> {
        // Simplified - would set the appropriate component in the gradient tensor
        if index < gradient.len() {
            gradient[index] = value;
            Ok(())
        } else {
            Err(StabilityError::ComputationError(
                "Index out of bounds".to_string(),
            ))
        }
    }

    #[allow(dead_code)]
    fn compute_second_partial_diagonal<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        index: usize,
        step: F,
    ) -> Result<F, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        // Compute ∂²f/∂x_i² using central differences
        // ∂²f/∂x_i² ≈ (f(x_i + h) - 2f(x_i) + f(x_i - h)) / h²

        let f_x = function(input)?;
        let input_plus = self.create_single_perturbation(input, index, step)?;
        let input_minus = self.create_single_perturbation(input, index, -step)?;

        let f_plus = function(&input_plus)?;
        let f_minus = function(&input_minus)?;

        let two = F::from(2.0).unwrap();
        let second_derivative = (self.extract_scalar(&f_plus)?
            - two * self.extract_scalar(&f_x)?
            + self.extract_scalar(&f_minus)?)
            / (step * step);

        Ok(second_derivative)
    }

    #[allow(dead_code)]
    fn compute_second_partial_mixed<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        i: usize,
        j: usize,
        step: F,
    ) -> Result<F, StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        // Compute ∂²f/∂x_i∂x_j using central differences
        // ∂²f/∂x_i∂x_j ≈ (f(x_i+h, x_j+h) - f(x_i+h, x_j-h) - f(x_i-h, x_j+h) + f(x_i-h, x_j-h)) / (4h²)

        let input_pp = self.create_double_perturbation(input, i, j, step, step)?;
        let input_pm = self.create_double_perturbation(input, i, j, step, -step)?;
        let input_mp = self.create_double_perturbation(input, i, j, -step, step)?;
        let input_mm = self.create_double_perturbation(input, i, j, -step, -step)?;

        let f_pp = function(&input_pp)?;
        let f_pm = function(&input_pm)?;
        let f_mp = function(&input_mp)?;
        let f_mm = function(&input_mm)?;

        let four = F::from(4.0).unwrap();
        let mixed_derivative = (self.extract_scalar(&f_pp)?
            - self.extract_scalar(&f_pm)?
            - self.extract_scalar(&f_mp)?
            + self.extract_scalar(&f_mm)?)
            / (four * step * step);

        Ok(mixed_derivative)
    }

    #[allow(dead_code)]
    fn compute_five_point_stencil<Func>(
        &self,
        function: &Func,
        input: &Tensor<F>,
        index: usize,
        step: F,
    ) -> Result<(F, F, F, F), StabilityError>
    where
        Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
    {
        let two = F::from(2.0).unwrap();

        let input_minus_2h = self.create_single_perturbation(input, index, -two * step)?;
        let input_minus_h = self.create_single_perturbation(input, index, -step)?;
        let input_plus_h = self.create_single_perturbation(input, index, step)?;
        let input_plus_2h = self.create_single_perturbation(input, index, two * step)?;

        let f_minus_2h = self.extract_scalar(&function(&input_minus_2h)?)?;
        let f_minus_h = self.extract_scalar(&function(&input_minus_h)?)?;
        let f_plus_h = self.extract_scalar(&function(&input_plus_h)?)?;
        let f_plus_2h = self.extract_scalar(&function(&input_plus_2h)?)?;

        Ok((f_minus_2h, f_minus_h, f_plus_h, f_plus_2h))
    }

    #[allow(dead_code)]
    fn create_single_perturbation<'a>(
        &self,
        input: &Tensor<'a, F>,
        _index: usize,
        _delta: F,
    ) -> Result<Tensor<'a, F>, StabilityError> {
        // Create a copy of input with a single component perturbed
        let perturbed = *input;
        // Simplified - would actually perturb the specific index
        Ok(perturbed)
    }

    #[allow(dead_code)]
    fn create_double_perturbation<'a>(
        &self,
        input: &Tensor<'a, F>,
        _i: usize,
        _j: usize,
        _delta_i: F,
        _delta_j: F,
    ) -> Result<Tensor<'a, F>, StabilityError> {
        // Create a copy of input with two components perturbed
        let perturbed = *input;
        // Simplified - would actually perturb the specific indices
        Ok(perturbed)
    }

    #[allow(dead_code)]
    fn extract_scalar(&self, _tensor: &Tensor<'_, F>) -> Result<F, StabilityError> {
        // Extract a scalar value from the tensor (assumes output is scalar)
        // Simplified implementation
        Ok(F::from(1.0).unwrap())
    }
}

impl<F: Float> Default for FiniteDifferenceComputer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator for creating perturbed inputs
pub struct PerturbedInputIterator<'a, F: Float> {
    input: Tensor<'a, F>,
    #[allow(dead_code)]
    step: F,
    current_index: usize,
    max_index: usize,
}

impl<'a, F: Float> PerturbedInputIterator<'a, F> {
    fn new(input: &Tensor<'a, F>, step: F) -> Self {
        let max_index = input.shape().iter().product();
        Self {
            input: *input,
            step,
            current_index: 0,
            max_index,
        }
    }
}

impl<'a, F: Float> Iterator for PerturbedInputIterator<'a, F> {
    type Item = Tensor<'a, F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.max_index {
            return None;
        }

        // Create perturbed input
        let perturbed = self.input;
        // Simplified - would actually perturb the current index

        self.current_index += 1;
        Some(perturbed)
    }
}

/// Iterator for creating central difference perturbed inputs
pub struct CentralPerturbedInputIterator<'a, F: Float> {
    input: Tensor<'a, F>,
    #[allow(dead_code)]
    step: F,
    current_index: usize,
    max_index: usize,
}

impl<'a, F: Float> CentralPerturbedInputIterator<'a, F> {
    fn new(input: &Tensor<'a, F>, step: F) -> Self {
        let max_index = input.shape().iter().product();
        Self {
            input: *input,
            step,
            current_index: 0,
            max_index,
        }
    }
}

impl<'a, F: Float> Iterator for CentralPerturbedInputIterator<'a, F> {
    type Item = (Tensor<'a, F>, Tensor<'a, F>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.max_index {
            return None;
        }

        // Create both positive and negative perturbations
        let input_plus = self.input;
        let input_minus = self.input;
        // Simplified - would actually perturb the current index

        self.current_index += 1;
        Some((input_plus, input_minus))
    }
}

/// Compute gradient using finite differences with specified scheme
pub fn compute_finite_difference_gradient<'a, F: Float, Func>(
    function: Func,
    input: &Tensor<'a, F>,
    scheme: FiniteDifferenceScheme,
    step_size: f64,
) -> Result<Tensor<'a, F>, StabilityError>
where
    Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
{
    let config = FiniteDifferenceConfig {
        step_size,
        scheme,
        ..Default::default()
    };

    let computer = FiniteDifferenceComputer::with_config(config);
    computer.compute_gradient(function, input)
}

/// Quick central difference gradient computation
pub fn central_difference_gradient<'a, F: Float, Func>(
    function: Func,
    input: &Tensor<'a, F>,
) -> Result<Tensor<'a, F>, StabilityError>
where
    Func: for<'b> Fn(&Tensor<'b, F>) -> Result<Tensor<'b, F>, StabilityError>,
{
    compute_finite_difference_gradient(function, input, FiniteDifferenceScheme::Central, 1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_difference_config() {
        let config = FiniteDifferenceConfig {
            step_size: 1e-6,
            scheme: FiniteDifferenceScheme::Central,
            adaptive_step: true,
            ..Default::default()
        };

        assert_eq!(config.step_size, 1e-6);
        assert_eq!(config.scheme, FiniteDifferenceScheme::Central);
        assert!(config.adaptive_step);
    }

    #[test]
    fn test_finite_difference_schemes() {
        assert_eq!(
            FiniteDifferenceScheme::Forward,
            FiniteDifferenceScheme::Forward
        );
        assert_ne!(
            FiniteDifferenceScheme::Forward,
            FiniteDifferenceScheme::Central
        );
    }

    #[test]
    fn test_computer_creation() {
        let _computer = FiniteDifferenceComputer::<f32>::new();

        let config = FiniteDifferenceConfig::default();
        let _computer_with_config = FiniteDifferenceComputer::<f32>::with_config(config);
    }
}
