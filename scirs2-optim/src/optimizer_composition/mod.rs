//! Optimizer composition framework
//!
//! This module provides compositions of optimizers to create more sophisticated
//! optimization strategies. It includes three main types of compositions:
//!
//! 1. **Sequential**: Apply multiple optimizers in sequence
//! 2. **Parallel**: Apply different optimizers to different parameter groups
//! 3. **Chained**: Wrap an optimizer with another (similar to Lookahead wrapping other optimizers)

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// A sequential composition of optimizers
///
/// This applies multiple optimizers in sequence to the same parameters.
/// Each optimizer's output becomes the input to the next optimizer.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizer_composition::SequentialOptimizer;
/// use scirs2_optim::optimizers::{SGD, Adam, Optimizer};
///
/// // Create optimizers
/// let sgd = SGD::new(0.1);
/// let adam = Adam::new(0.01);
///
/// // Combine them sequentially
/// let mut seq_optimizer = SequentialOptimizer::new(vec![
///     Box::new(sgd),
///     Box::new(adam),
/// ]);
///
/// // Use the sequential optimizer
/// let params = Array1::zeros(5);
/// let gradients = Array1::ones(5);
/// let updated_params = seq_optimizer.step(&params, &gradients).unwrap();
/// ```
pub struct SequentialOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// List of optimizers to apply in sequence
    optimizers: Vec<Box<dyn Optimizer<A, D>>>,
}

impl<A, D> SequentialOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Create a new sequential optimizer
    ///
    /// # Arguments
    ///
    /// * `optimizers` - List of optimizers to apply in sequence
    pub fn new(optimizers: Vec<Box<dyn Optimizer<A, D>>>) -> Self {
        Self { optimizers }
    }

    /// Add an optimizer to the sequence
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to add
    pub fn add_optimizer(&mut self, optimizer: Box<dyn Optimizer<A, D>>) {
        self.optimizers.push(optimizer);
    }

    /// Get the number of optimizers in the sequence
    pub fn num_optimizers(&self) -> usize {
        self.optimizers.len()
    }

    /// Get a reference to an optimizer by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the optimizer
    ///
    /// # Returns
    ///
    /// A reference to the optimizer at the given index, or None if out of bounds
    pub fn get_optimizer(&self, index: usize) -> Option<&dyn Optimizer<A, D>> {
        if index < self.optimizers.len() {
            Some(self.optimizers[index].as_ref())
        } else {
            None
        }
    }

    /// Get a mutable reference to an optimizer by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the optimizer
    ///
    /// # Returns
    ///
    /// A mutable reference to the optimizer at the given index, or None if out of bounds
    pub fn get_optimizer_mut(&mut self, index: usize) -> Option<&mut dyn Optimizer<A, D>> {
        if index < self.optimizers.len() {
            Some(self.optimizers[index].as_mut())
        } else {
            None
        }
    }
}

impl<A, D> Optimizer<A, D> for SequentialOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Check if we have any optimizers
        if self.optimizers.is_empty() {
            return Err(OptimError::InvalidConfig(
                "SequentialOptimizer has no optimizers".to_string(),
            ));
        }

        // Start with the initial parameters
        let mut current_params = params.clone();

        // Apply each optimizer in sequence
        for optimizer in &mut self.optimizers {
            current_params = optimizer.step(&current_params, gradients)?;
        }

        Ok(current_params)
    }

    fn get_learning_rate(&self) -> A {
        // Return the learning rate of the first optimizer, or a default if empty
        if let Some(optimizer) = self.optimizers.first() {
            optimizer.get_learning_rate()
        } else {
            A::from(0.01).unwrap() // Default learning rate
        }
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        // Set the learning rate for all optimizers
        for optimizer in &mut self.optimizers {
            optimizer.set_learning_rate(learning_rate);
        }
    }
}

/// A struct for assigning parameters to specific groups for parallel optimization
pub struct ParameterGroup<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// The parameters in this group
    pub params: Array<A, D>,
    /// The index of the optimizer to use for this group
    pub optimizer_index: usize,
}

impl<A, D> ParameterGroup<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Create a new parameter group
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters in this group
    /// * `optimizer_index` - The index of the optimizer to use for this group
    pub fn new(params: Array<A, D>, optimizer_index: usize) -> Self {
        Self {
            params,
            optimizer_index,
        }
    }
}

/// A parallel composition of optimizers
///
/// This applies different optimizers to different parameter groups.
/// Each group of parameters is updated using its assigned optimizer.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizer_composition::{ParallelOptimizer, ParameterGroup};
/// use scirs2_optim::optimizers::{SGD, Adam, Optimizer};
///
/// // Create optimizers
/// let sgd = SGD::new(0.1);
/// let adam = Adam::new(0.01);
///
/// // Create parameter groups
/// let params1 = Array1::zeros(3);
/// let params2 = Array1::zeros(5);
///
/// let group1 = ParameterGroup::new(params1, 0); // Use SGD
/// let group2 = ParameterGroup::new(params2, 1); // Use Adam
///
/// // Combine them in parallel
/// let mut parallel_optimizer = ParallelOptimizer::new(
///     vec![Box::new(sgd), Box::new(adam)],
///     vec![group1, group2],
/// );
///
/// // The step method will update all parameter groups using their assigned optimizers
/// // (In a real use case, you'd provide the corresponding gradients)
/// ```
pub struct ParallelOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// List of optimizers to apply to different parameter groups
    optimizers: Vec<Box<dyn Optimizer<A, D>>>,
    /// Groups of parameters with their assigned optimizer indices
    parameter_groups: Vec<ParameterGroup<A, D>>,
}

impl<A, D> ParallelOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Create a new parallel optimizer
    ///
    /// # Arguments
    ///
    /// * `optimizers` - List of optimizers to use
    /// * `parameter_groups` - Groups of parameters with their assigned optimizer indices
    pub fn new(
        optimizers: Vec<Box<dyn Optimizer<A, D>>>,
        parameter_groups: Vec<ParameterGroup<A, D>>,
    ) -> Self {
        Self {
            optimizers,
            parameter_groups,
        }
    }

    /// Add an optimizer
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to add
    ///
    /// # Returns
    ///
    /// The index of the added optimizer
    pub fn add_optimizer(&mut self, optimizer: Box<dyn Optimizer<A, D>>) -> usize {
        let index = self.optimizers.len();
        self.optimizers.push(optimizer);
        index
    }

    /// Add a parameter group
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters in this group
    /// * `optimizer_index` - The index of the optimizer to use for this group
    ///
    /// # Returns
    ///
    /// Result with the index of the added parameter group, or an error if the optimizer index is invalid
    pub fn add_parameter_group(
        &mut self,
        params: Array<A, D>,
        optimizer_index: usize,
    ) -> Result<usize> {
        // Check if the optimizer index is valid
        if optimizer_index >= self.optimizers.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Invalid optimizer index: {}. Only {} optimizers available.",
                optimizer_index,
                self.optimizers.len()
            )));
        }

        let index = self.parameter_groups.len();
        self.parameter_groups
            .push(ParameterGroup::new(params, optimizer_index));
        Ok(index)
    }

    /// Get the number of optimizers
    pub fn num_optimizers(&self) -> usize {
        self.optimizers.len()
    }

    /// Get the number of parameter groups
    pub fn num_parameter_groups(&self) -> usize {
        self.parameter_groups.len()
    }

    /// Get a reference to an optimizer by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the optimizer
    ///
    /// # Returns
    ///
    /// A reference to the optimizer at the given index, or None if out of bounds
    pub fn get_optimizer(&self, index: usize) -> Option<&dyn Optimizer<A, D>> {
        if index < self.optimizers.len() {
            Some(self.optimizers[index].as_ref())
        } else {
            None
        }
    }

    /// Get a mutable reference to an optimizer by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the optimizer
    ///
    /// # Returns
    ///
    /// A mutable reference to the optimizer at the given index, or None if out of bounds
    pub fn get_optimizer_mut(&mut self, index: usize) -> Option<&mut dyn Optimizer<A, D>> {
        if index < self.optimizers.len() {
            Some(self.optimizers[index].as_mut())
        } else {
            None
        }
    }

    /// Get a reference to a parameter group by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the parameter group
    ///
    /// # Returns
    ///
    /// A reference to the parameter group at the given index, or None if out of bounds
    pub fn get_parameter_group(&self, index: usize) -> Option<&ParameterGroup<A, D>> {
        self.parameter_groups.get(index)
    }

    /// Get a mutable reference to a parameter group by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the parameter group
    ///
    /// # Returns
    ///
    /// A mutable reference to the parameter group at the given index, or None if out of bounds
    pub fn get_parameter_group_mut(&mut self, index: usize) -> Option<&mut ParameterGroup<A, D>> {
        self.parameter_groups.get_mut(index)
    }

    /// Get all current parameter values as a single array
    ///
    /// # Returns
    ///
    /// A result containing all parameter values concatenated into a single array
    pub fn get_all_parameters(&self) -> Result<Vec<Array<A, D>>> {
        Ok(self
            .parameter_groups
            .iter()
            .map(|group| group.params.clone())
            .collect())
    }

    /// Update all parameter groups using their assigned optimizers
    ///
    /// # Arguments
    ///
    /// * `gradients` - List of gradient arrays corresponding to parameter groups
    ///
    /// # Returns
    ///
    /// Result with the updated parameter values, or an error
    pub fn update_all_parameters(&mut self, gradients: &[Array<A, D>]) -> Result<Vec<Array<A, D>>> {
        // Check if the number of gradients matches the number of parameter groups
        if gradients.len() != self.parameter_groups.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of gradients ({}) does not match number of parameter groups ({})",
                gradients.len(),
                self.parameter_groups.len()
            )));
        }

        let mut updated_params = Vec::with_capacity(self.parameter_groups.len());

        // Update each parameter group using its assigned optimizer
        for (i, group) in self.parameter_groups.iter_mut().enumerate() {
            let optimizer_index = group.optimizer_index;

            // Check if the optimizer index is valid
            if optimizer_index >= self.optimizers.len() {
                return Err(OptimError::InvalidConfig(format!(
                    "Invalid optimizer index: {}. Only {} optimizers available.",
                    optimizer_index,
                    self.optimizers.len()
                )));
            }

            // Get the optimizer and update the parameters
            let optimizer = &mut self.optimizers[optimizer_index];
            let params = &group.params;
            let gradient = &gradients[i];

            // Update the parameters
            let updated = optimizer.step(params, gradient)?;
            group.params = updated.clone();
            updated_params.push(updated);
        }

        Ok(updated_params)
    }
}

impl<A, D> Optimizer<A, D> for ParallelOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, _params: &Array<A, D>, _gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // This implementation is a bit tricky since we have multiple parameter groups
        // We'll return an error message directing users to use update_all_parameters instead
        Err(OptimError::InvalidConfig(
            "ParallelOptimizer doesn't support the standard step method. Use update_all_parameters instead."
                .to_string(),
        ))
    }

    fn step_list(
        &mut self,
        params_list: &[&Array<A, D>],
        gradients_list: &[&Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        // Convert params_list to owned arrays
        let params_vec: Vec<Array<A, D>> = params_list.iter().map(|&p| p.clone()).collect();

        // Set parameter groups based on the input params
        self.parameter_groups = params_vec
            .into_iter()
            .enumerate()
            .map(|(i, params)| {
                // Use the first optimizer for all if there are more params than optimizers
                let optimizer_index = i.min(self.optimizers.len() - 1);
                ParameterGroup::new(params, optimizer_index)
            })
            .collect();

        // Convert gradients_list to owned arrays
        let gradients_vec: Vec<Array<A, D>> = gradients_list.iter().map(|&g| g.clone()).collect();

        // Update parameter groups using their assigned optimizers
        self.update_all_parameters(&gradients_vec)
    }

    fn get_learning_rate(&self) -> A {
        // Return the learning rate of the first optimizer, or a default if empty
        if let Some(optimizer) = self.optimizers.first() {
            optimizer.get_learning_rate()
        } else {
            A::from(0.01).unwrap() // Default learning rate
        }
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        // Set the learning rate for all optimizers
        for optimizer in &mut self.optimizers {
            optimizer.set_learning_rate(learning_rate);
        }
    }
}

/// A chained composition of optimizers
///
/// This wraps one optimizer with another, similar to how Lookahead wraps
/// another optimizer. The inner optimizer is applied first, and then the
/// outer optimizer is applied to the result.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizer_composition::ChainedOptimizer;
/// use scirs2_optim::optimizers::{SGD, Adam, Optimizer};
///
/// // Create optimizers
/// let inner = SGD::new(0.1);
/// let outer = Adam::new(0.01);
///
/// // Chain them together
/// let mut chained_optimizer = ChainedOptimizer::new(Box::new(inner), Box::new(outer));
///
/// // Use the chained optimizer
/// let params = Array1::zeros(5);
/// let gradients = Array1::ones(5);
/// let updated_params = chained_optimizer.step(&params, &gradients).unwrap();
/// ```
pub struct ChainedOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// The inner optimizer, applied first
    inner: Box<dyn Optimizer<A, D>>,
    /// The outer optimizer, applied to the result of the inner optimizer
    outer: Box<dyn Optimizer<A, D>>,
}

impl<A, D> ChainedOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    /// Create a new chained optimizer
    ///
    /// # Arguments
    ///
    /// * `inner` - The inner optimizer, applied first
    /// * `outer` - The outer optimizer, applied to the result of the inner optimizer
    pub fn new(inner: Box<dyn Optimizer<A, D>>, outer: Box<dyn Optimizer<A, D>>) -> Self {
        Self { inner, outer }
    }

    /// Get a reference to the inner optimizer
    pub fn inner(&self) -> &dyn Optimizer<A, D> {
        self.inner.as_ref()
    }

    /// Get a mutable reference to the inner optimizer
    pub fn inner_mut(&mut self) -> &mut dyn Optimizer<A, D> {
        self.inner.as_mut()
    }

    /// Get a reference to the outer optimizer
    pub fn outer(&self) -> &dyn Optimizer<A, D> {
        self.outer.as_ref()
    }

    /// Get a mutable reference to the outer optimizer
    pub fn outer_mut(&mut self) -> &mut dyn Optimizer<A, D> {
        self.outer.as_mut()
    }
}

impl<A, D> Optimizer<A, D> for ChainedOptimizer<A, D>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Apply the inner optimizer first
        let intermediate_params = self.inner.step(params, gradients)?;

        // Then apply the outer optimizer to the result
        self.outer.step(&intermediate_params, gradients)
    }

    fn get_learning_rate(&self) -> A {
        // Return the learning rate of the inner optimizer
        self.inner.get_learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        // Set the learning rate for both optimizers
        self.inner.set_learning_rate(learning_rate);
        self.outer.set_learning_rate(learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{Adam, SGD};
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_sequential_optimizer() {
        // Create a sequential optimizer with SGD followed by Adam
        let sgd = SGD::new(0.1);
        let adam = Adam::new(0.01);

        let mut seq_optimizer: SequentialOptimizer<f64, ndarray::Ix1> =
            SequentialOptimizer::new(vec![Box::new(sgd), Box::new(adam)]);

        // Create test parameters and gradients
        let params = Array1::zeros(3);
        let gradients = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Apply the sequential optimizer
        let updated_params = seq_optimizer.step(&params, &gradients).unwrap();

        // Verify the result
        // First SGD updates: params - 0.1 * gradients = [0, 0, 0] - 0.1 * [1, 2, 3] = [-0.1, -0.2, -0.3]
        // Then Adam makes additional updates
        assert!(updated_params[0] < -0.1);
        assert!(updated_params[1] < -0.2);
        assert!(updated_params[2] < -0.3);
    }

    #[test]
    fn test_parallel_optimizer() {
        // Create a parallel optimizer with SGD and Adam
        let sgd = SGD::new(0.1);
        let adam = Adam::new(0.01);

        let params1 = Array1::zeros(2);
        let params2 = Array1::zeros(3);

        let group1 = ParameterGroup::new(params1.clone(), 0); // Use SGD
        let group2 = ParameterGroup::new(params2.clone(), 1); // Use Adam

        let mut parallel_optimizer: ParallelOptimizer<f64, ndarray::Ix1> =
            ParallelOptimizer::new(vec![Box::new(sgd), Box::new(adam)], vec![group1, group2]);

        // Create test gradients
        let gradients1 = Array1::from_vec(vec![1.0, 2.0]);
        let gradients2 = Array1::from_vec(vec![3.0, 4.0, 5.0]);

        // Update the parameters
        let updated_params = parallel_optimizer
            .update_all_parameters(&[gradients1, gradients2])
            .unwrap();

        // Verify the results
        // Group 1 (SGD): params - 0.1 * gradients = [0, 0] - 0.1 * [1, 2] = [-0.1, -0.2]
        assert_abs_diff_eq!(updated_params[0][0], -0.1);
        assert_abs_diff_eq!(updated_params[0][1], -0.2);

        // Group 2 (Adam): The update will be different due to Adam's adaptive nature
        // Just verify it's different from the original params
        assert!(updated_params[1][0] != 0.0);
        assert!(updated_params[1][1] != 0.0);
        assert!(updated_params[1][2] != 0.0);
    }

    #[test]
    fn test_chained_optimizer() {
        // Create a chained optimizer with SGD as inner and Adam as outer
        let inner = SGD::new(0.1);
        let outer = Adam::new(0.01);

        let mut chained_optimizer: ChainedOptimizer<f64, ndarray::Ix1> =
            ChainedOptimizer::new(Box::new(inner), Box::new(outer));

        // Create test parameters and gradients
        let params = Array1::zeros(3);
        let gradients = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Apply the chained optimizer
        let updated_params = chained_optimizer.step(&params, &gradients).unwrap();

        // Verify the result
        // Inner (SGD): params - 0.1 * gradients = [0, 0, 0] - 0.1 * [1, 2, 3] = [-0.1, -0.2, -0.3]
        // Then outer (Adam) applies another update
        assert!(updated_params[0] < -0.1);
        assert!(updated_params[1] < -0.2);
        assert!(updated_params[2] < -0.3);
    }

    #[test]
    fn test_sequential_learning_rate() {
        // Create a sequential optimizer with SGD followed by Adam
        let sgd = SGD::new(0.1);
        let adam = Adam::new(0.01);

        let mut seq_optimizer: SequentialOptimizer<f64, ndarray::Ix1> =
            SequentialOptimizer::new(vec![Box::new(sgd), Box::new(adam)]);

        // Test getting the learning rate (should be from the first optimizer)
        assert_abs_diff_eq!(seq_optimizer.get_learning_rate(), 0.1);

        // Test setting the learning rate for all optimizers
        seq_optimizer.set_learning_rate(0.05);

        // Verify the learning rate has been set for both optimizers
        assert_abs_diff_eq!(seq_optimizer.get_learning_rate(), 0.05);
        assert_abs_diff_eq!(
            seq_optimizer.get_optimizer(0).unwrap().get_learning_rate(),
            0.05
        );
        assert_abs_diff_eq!(
            seq_optimizer.get_optimizer(1).unwrap().get_learning_rate(),
            0.05
        );
    }

    #[test]
    fn test_parallel_optimizer_step_list() {
        // Create a parallel optimizer with SGD and Adam
        let sgd = SGD::new(0.1);
        let adam = Adam::new(0.01);

        let mut parallel_optimizer: ParallelOptimizer<f64, ndarray::Ix1> =
            ParallelOptimizer::new(vec![Box::new(sgd), Box::new(adam)], vec![]);

        // Create test parameters and gradients
        let params1 = Array1::zeros(2);
        let params2 = Array1::zeros(3);
        let params3 = Array1::zeros(4);

        let gradients1 = Array1::from_vec(vec![1.0, 2.0]);
        let gradients2 = Array1::from_vec(vec![3.0, 4.0, 5.0]);
        let gradients3 = Array1::from_vec(vec![6.0, 7.0, 8.0, 9.0]);

        // Use step_list to update all parameters
        let params_refs = vec![&params1, &params2, &params3];
        let gradients_refs = vec![&gradients1, &gradients2, &gradients3];

        let updated_params = parallel_optimizer
            .step_list(&params_refs, &gradients_refs)
            .unwrap();

        // Verify the results
        // Group 1 (SGD): params - 0.1 * gradients = [0, 0] - 0.1 * [1, 2] = [-0.1, -0.2]
        assert_abs_diff_eq!(updated_params[0][0], -0.1);
        assert_abs_diff_eq!(updated_params[0][1], -0.2);

        // Group 2 will use SGD since we only have 2 optimizers and index 1 % 2 = 1 (Adam)
        // Adam: The update will be different than SGD
        assert!(updated_params[1][0] != -0.3);

        // Group 3 will wrap around to optimize with Adam
        // Just check that it's been updated from zero
        assert!(updated_params[2][0] < 0.0);
    }

    #[test]
    fn test_chained_optimizer_learning_rate() {
        // Create a chained optimizer with SGD as inner and Adam as outer
        let inner = SGD::new(0.1);
        let outer = Adam::new(0.01);

        let mut chained_optimizer: ChainedOptimizer<f64, ndarray::Ix1> =
            ChainedOptimizer::new(Box::new(inner), Box::new(outer));

        // Test getting the learning rate (should be from the inner optimizer)
        assert_abs_diff_eq!(chained_optimizer.get_learning_rate(), 0.1);

        // Test setting the learning rate for both optimizers
        chained_optimizer.set_learning_rate(0.05);

        // Verify the learning rate has been set for both optimizers
        assert_abs_diff_eq!(chained_optimizer.get_learning_rate(), 0.05);
        assert_abs_diff_eq!(chained_optimizer.inner().get_learning_rate(), 0.05);
        assert_abs_diff_eq!(chained_optimizer.outer().get_learning_rate(), 0.05);
    }
}
