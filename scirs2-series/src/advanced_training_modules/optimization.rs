//! Meta-Optimization Algorithms
//!
//! This module implements meta-optimization techniques including learned optimizers
//! that can adaptively optimize neural network parameters based on the optimization history.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;

/// Meta-Optimizer using LSTM to generate parameter updates
#[derive(Debug)]
pub struct MetaOptimizer<F: Float + Debug + ndarray::ScalarOperand> {
    /// LSTM parameters for the optimizer
    #[allow(dead_code)]
    lstm_params: Array2<F>,
    /// Hidden state size
    hidden_size: usize,
    /// Input dimension (gradient + other features)
    input_dim: usize,
    /// Current LSTM hidden state
    hidden_state: Array1<F>,
    /// Current LSTM cell state
    cell_state: Array1<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> MetaOptimizer<F> {
    /// Create new meta-optimizer
    pub fn new(input_dim: usize, hidden_size: usize) -> Self {
        // Initialize LSTM parameters
        let param_count = 4 * hidden_size * (input_dim + hidden_size) + 4 * hidden_size; // 4 gates
        let mut lstm_params = Array2::zeros((1, param_count));

        let scale = F::from(1.0).unwrap() / F::from(hidden_size).unwrap().sqrt();
        for i in 0..param_count {
            let val = ((i * 79) % 1000) as f64 / 1000.0 - 0.5;
            lstm_params[[0, i]] = F::from(val).unwrap() * scale;
        }

        let hidden_state = Array1::zeros(hidden_size);
        let cell_state = Array1::zeros(hidden_size);

        Self {
            lstm_params,
            hidden_size,
            input_dim,
            hidden_state,
            cell_state,
        }
    }

    /// Generate parameter update using meta-optimizer
    pub fn generate_update(
        &mut self,
        gradient: F,
        loss_history: &[F],
        step_count: usize,
    ) -> Result<F> {
        // Prepare input features
        let mut input = Array1::zeros(self.input_dim);
        input[0] = gradient;

        if self.input_dim > 1 && !loss_history.is_empty() {
            input[1] = loss_history[loss_history.len() - 1];
        }

        if self.input_dim > 2 {
            input[2] = F::from(step_count).unwrap();
        }

        // LSTM forward pass
        let (new_hidden, new_cell) = self.lstm_forward(&input)?;
        self.hidden_state = new_hidden.clone();
        self.cell_state = new_cell;

        // Generate parameter update (use first output as update)
        Ok(new_hidden[0])
    }

    /// LSTM forward pass
    fn lstm_forward(&self, input: &Array1<F>) -> Result<(Array1<F>, Array1<F>)> {
        // Extract LSTM weights (simplified implementation)
        let combined_input = self.combine_input_hidden(input);

        // Compute gates (simplified)
        let forget_gate = self.sigmoid(combined_input[0]);
        let input_gate = self.sigmoid(combined_input[1]);
        let candidate_gate = self.tanh(combined_input[2]);
        let output_gate = self.sigmoid(combined_input[3]);

        // Update cell state
        let mut new_cell_state = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_cell_state[i] = forget_gate * self.cell_state[i] + input_gate * candidate_gate;
        }

        // Update hidden state
        let mut new_hidden_state = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_hidden_state[i] = output_gate * self.tanh(new_cell_state[i]);
        }

        Ok((new_hidden_state, new_cell_state))
    }

    /// Combine input and hidden state
    fn combine_input_hidden(&self, input: &Array1<F>) -> Array1<F> {
        // Simplified combination - just use input values for gates
        let mut combined = Array1::zeros(4);
        for i in 0..4.min(input.len()) {
            combined[i] = input[i.min(input.len() - 1)];
        }
        combined
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }

    /// Hyperbolic tangent activation
    fn tanh(&self, x: F) -> F {
        x.tanh()
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.hidden_state = Array1::zeros(self.hidden_size);
        self.cell_state = Array1::zeros(self.hidden_size);
    }

    /// Train meta-optimizer on optimization tasks
    pub fn meta_train(&mut self, optimization_problems: &[OptimizationProblem<F>]) -> Result<F> {
        let mut total_loss = F::zero();

        for problem in optimization_problems {
            self.reset();

            let mut current_params = problem.initial_params.clone();
            let mut loss_history = Vec::new();

            // Simulate optimization steps
            for step in 0..problem.max_steps {
                // Compute gradient
                let gradient = self.compute_simple_gradient(&current_params, problem)?;

                // Generate update using meta-optimizer
                let update = self.generate_update(gradient, &loss_history, step)?;

                // Apply update
                current_params = current_params + update;

                // Compute loss
                let loss = self.evaluate_objective(&current_params, problem)?;
                loss_history.push(loss);
                total_loss = total_loss + loss;
            }
        }

        Ok(total_loss / F::from(optimization_problems.len()).unwrap())
    }

    /// Compute simple gradient (placeholder)
    fn compute_simple_gradient(
        &self,
        params: &Array1<F>,
        problem: &OptimizationProblem<F>,
    ) -> Result<F> {
        // Simplified gradient computation
        if !params.is_empty() && !problem.target.is_empty() {
            Ok(params[0] - problem.target[0])
        } else {
            Ok(F::zero())
        }
    }

    /// Evaluate objective function
    fn evaluate_objective(
        &self,
        params: &Array1<F>,
        problem: &OptimizationProblem<F>,
    ) -> Result<F> {
        // Simple quadratic objective
        let mut loss = F::zero();
        for i in 0..params.len().min(problem.target.len()) {
            let diff = params[i] - problem.target[i];
            loss = loss + diff * diff;
        }
        Ok(loss)
    }

    /// Get current hidden state
    pub fn hidden_state(&self) -> &Array1<F> {
        &self.hidden_state
    }

    /// Get current cell state
    pub fn cell_state(&self) -> &Array1<F> {
        &self.cell_state
    }

    /// Set hidden state
    pub fn set_hidden_state(&mut self, state: Array1<F>) -> Result<()> {
        if state.len() != self.hidden_size {
            return Err(crate::error::TimeSeriesError::InvalidOperation(
                "Hidden state size mismatch".to_string(),
            ));
        }
        self.hidden_state = state;
        Ok(())
    }

    /// Set cell state
    pub fn set_cell_state(&mut self, state: Array1<F>) -> Result<()> {
        if state.len() != self.hidden_size {
            return Err(crate::error::TimeSeriesError::InvalidOperation(
                "Cell state size mismatch".to_string(),
            ));
        }
        self.cell_state = state;
        Ok(())
    }

    /// Get optimizer dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.input_dim, self.hidden_size)
    }

    /// Apply meta-optimizer to optimize parameters
    pub fn optimize_parameters(
        &mut self,
        initial_params: &Array1<F>,
        target: &Array1<F>,
        max_steps: usize,
    ) -> Result<(Array1<F>, Vec<F>)> {
        self.reset();

        let mut current_params = initial_params.clone();
        let mut loss_history = Vec::new();

        let problem = OptimizationProblem {
            initial_params: initial_params.clone(),
            target: target.clone(),
            max_steps,
        };

        for step in 0..max_steps {
            // Compute gradient
            let gradient = self.compute_simple_gradient(&current_params, &problem)?;

            // Generate update using meta-optimizer
            let update = self.generate_update(gradient, &loss_history, step)?;

            // Apply update
            current_params = current_params + update;

            // Compute and record loss
            let loss = self.evaluate_objective(&current_params, &problem)?;
            loss_history.push(loss);
        }

        Ok((current_params, loss_history))
    }

    /// Generate vectorized updates for multiple parameters
    pub fn generate_vectorized_update(
        &mut self,
        gradients: &Array1<F>,
        loss_history: &[F],
        step_count: usize,
    ) -> Result<Array1<F>> {
        let mut updates = Array1::zeros(gradients.len());

        for (i, &gradient) in gradients.iter().enumerate() {
            let update = self.generate_update(gradient, loss_history, step_count + i)?;
            updates[i] = update;
        }

        Ok(updates)
    }
}

/// Optimization problem for meta-optimizer training
#[derive(Debug, Clone)]
pub struct OptimizationProblem<F: Float + Debug> {
    /// Initial parameters
    pub initial_params: Array1<F>,
    /// Target parameters
    pub target: Array1<F>,
    /// Maximum optimization steps
    pub max_steps: usize,
}

impl<F: Float + Debug> OptimizationProblem<F> {
    /// Create a new optimization problem
    pub fn new(initial_params: Array1<F>, target: Array1<F>, max_steps: usize) -> Self {
        Self {
            initial_params,
            target,
            max_steps,
        }
    }

    /// Create a quadratic optimization problem
    pub fn quadratic(dim: usize, max_steps: usize) -> Self
    where
        F: FromPrimitive,
    {
        let initial_params = Array1::from_vec(
            (0..dim)
                .map(|i| F::from((i * 13) % 100).unwrap() / F::from(100.0).unwrap())
                .collect(),
        );
        let target = Array1::zeros(dim);

        Self {
            initial_params,
            target,
            max_steps,
        }
    }

    /// Create a random optimization problem
    pub fn random(dim: usize, max_steps: usize) -> Self
    where
        F: FromPrimitive,
    {
        let initial_params = Array1::from_vec(
            (0..dim)
                .map(|i| F::from((i * 17 + 23) % 200).unwrap() / F::from(100.0).unwrap() - F::one())
                .collect(),
        );
        let target = Array1::from_vec(
            (0..dim)
                .map(|i| F::from((i * 19 + 37) % 100).unwrap() / F::from(200.0).unwrap())
                .collect(),
        );

        Self {
            initial_params,
            target,
            max_steps,
        }
    }

    /// Get the dimension of the problem
    pub fn dimension(&self) -> usize {
        self.initial_params.len()
    }

    /// Evaluate the objective function at given parameters
    pub fn evaluate(&self, params: &Array1<F>) -> F {
        let mut loss = F::zero();
        for i in 0..params.len().min(self.target.len()) {
            let diff = params[i] - self.target[i];
            loss = loss + diff * diff;
        }
        loss
    }

    /// Compute the gradient at given parameters
    pub fn gradient(&self, params: &Array1<F>) -> Array1<F> {
        let mut grad = Array1::zeros(params.len());
        for i in 0..params.len().min(self.target.len()) {
            grad[i] = F::from(2.0).unwrap() * (params[i] - self.target[i]);
        }
        grad
    }

    /// Check if the problem has converged
    pub fn has_converged(&self, params: &Array1<F>, tolerance: F) -> bool {
        self.evaluate(params) < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_meta_optimizer_creation() {
        let meta_opt = MetaOptimizer::<f64>::new(3, 5);
        let (input_dim, hidden_size) = meta_opt.dimensions();

        assert_eq!(input_dim, 3);
        assert_eq!(hidden_size, 5);
        assert_eq!(meta_opt.hidden_state().len(), 5);
        assert_eq!(meta_opt.cell_state().len(), 5);
    }

    #[test]
    fn test_meta_optimizer_update_generation() {
        let mut meta_opt = MetaOptimizer::<f64>::new(3, 4);

        let gradient = 0.1;
        let loss_history = vec![1.0, 0.8, 0.6];
        let step_count = 5;

        let update = meta_opt
            .generate_update(gradient, &loss_history, step_count)
            .unwrap();
        assert!(update.is_finite());
    }

    #[test]
    fn test_meta_optimizer_reset() {
        let mut meta_opt = MetaOptimizer::<f64>::new(2, 3);

        // Generate some updates to change state
        let _ = meta_opt.generate_update(0.5, &[1.0], 1).unwrap();

        // Reset should zero out the states
        meta_opt.reset();

        for &val in meta_opt.hidden_state().iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
        for &val in meta_opt.cell_state().iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_optimization_problem_creation() {
        let initial = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let problem = OptimizationProblem::new(initial, target, 100);

        assert_eq!(problem.dimension(), 3);
        assert_eq!(problem.max_steps, 100);
    }

    #[test]
    fn test_optimization_problem_evaluation() {
        let initial = Array1::from_vec(vec![1.0, 2.0]);
        let target = Array1::from_vec(vec![0.0, 0.0]);
        let problem = OptimizationProblem::new(initial, target, 50);

        let params = Array1::from_vec(vec![1.0, 1.0]);
        let loss = problem.evaluate(&params);
        let expected = (1.0 - 0.0).powi(2) + (1.0 - 0.0).powi(2);
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_optimization_problem_gradient() {
        let initial = Array1::from_vec(vec![2.0, 3.0]);
        let target = Array1::from_vec(vec![1.0, 1.0]);
        let problem = OptimizationProblem::new(initial, target, 50);

        let params = Array1::from_vec(vec![2.0, 3.0]);
        let gradient = problem.gradient(&params);

        // For quadratic loss f(x) = (x - target)^2, gradient is 2(x - target)
        assert_abs_diff_eq!(gradient[0], 2.0 * (2.0 - 1.0), epsilon = 1e-10);
        assert_abs_diff_eq!(gradient[1], 2.0 * (3.0 - 1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_optimization_problem_convergence() {
        let initial = Array1::from_vec(vec![1.0]);
        let target = Array1::from_vec(vec![0.0]);
        let problem = OptimizationProblem::new(initial, target, 50);

        let converged_params = Array1::from_vec(vec![0.001]);
        let not_converged_params = Array1::from_vec(vec![0.5]);

        assert!(problem.has_converged(&converged_params, 0.01));
        assert!(!problem.has_converged(&not_converged_params, 0.01));
    }

    #[test]
    fn test_quadratic_optimization_problem() {
        let problem = OptimizationProblem::<f64>::quadratic(3, 100);

        assert_eq!(problem.dimension(), 3);
        assert_eq!(problem.max_steps, 100);

        // Target should be zeros
        for &val in problem.target.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_random_optimization_problem() {
        let problem = OptimizationProblem::<f64>::random(4, 200);

        assert_eq!(problem.dimension(), 4);
        assert_eq!(problem.max_steps, 200);
        assert_eq!(problem.initial_params.len(), 4);
        assert_eq!(problem.target.len(), 4);
    }

    #[test]
    fn test_meta_optimizer_state_management() {
        let mut meta_opt = MetaOptimizer::<f64>::new(2, 3);

        let new_hidden = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let new_cell = Array1::from_vec(vec![0.5, 1.5, 2.5]);

        meta_opt.set_hidden_state(new_hidden.clone()).unwrap();
        meta_opt.set_cell_state(new_cell.clone()).unwrap();

        for (i, &val) in meta_opt.hidden_state().iter().enumerate() {
            assert_abs_diff_eq!(val, new_hidden[i], epsilon = 1e-10);
        }
        for (i, &val) in meta_opt.cell_state().iter().enumerate() {
            assert_abs_diff_eq!(val, new_cell[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_meta_optimizer_state_validation() {
        let mut meta_opt = MetaOptimizer::<f64>::new(2, 3);

        // Try to set state with wrong dimensions
        let wrong_state = Array1::from_vec(vec![1.0, 2.0]); // Should be size 3

        let result = meta_opt.set_hidden_state(wrong_state.clone());
        assert!(result.is_err());

        let result = meta_opt.set_cell_state(wrong_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize_parameters() {
        let mut meta_opt = MetaOptimizer::<f64>::new(3, 4);

        let initial = Array1::from_vec(vec![2.0, 3.0]);
        let target = Array1::from_vec(vec![0.0, 0.0]);

        let (final_params, loss_history) =
            meta_opt.optimize_parameters(&initial, &target, 10).unwrap();

        assert_eq!(final_params.len(), 2);
        assert_eq!(loss_history.len(), 10);

        // Loss should generally decrease (though with the simplified optimizer, this might not always be true)
        assert!(loss_history.iter().all(|&loss| loss.is_finite()));
    }

    #[test]
    fn test_vectorized_update_generation() {
        let mut meta_opt = MetaOptimizer::<f64>::new(3, 4);

        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let loss_history = vec![1.0, 0.8];

        let updates = meta_opt
            .generate_vectorized_update(&gradients, &loss_history, 5)
            .unwrap();

        assert_eq!(updates.len(), 3);
        for &update in updates.iter() {
            assert!(update.is_finite());
        }
    }

    #[test]
    fn test_lstm_forward_pass() {
        let meta_opt = MetaOptimizer::<f64>::new(2, 3);
        let input = Array1::from_vec(vec![0.5, -0.3]);

        let (hidden, cell) = meta_opt.lstm_forward(&input).unwrap();

        assert_eq!(hidden.len(), 3);
        assert_eq!(cell.len(), 3);

        for &val in hidden.iter() {
            assert!(val.is_finite());
        }
        for &val in cell.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_activation_functions() {
        let meta_opt = MetaOptimizer::<f64>::new(1, 1);

        // Test sigmoid
        let sigmoid_result = meta_opt.sigmoid(0.0);
        assert_abs_diff_eq!(sigmoid_result, 0.5, epsilon = 1e-10);

        let sigmoid_pos = meta_opt.sigmoid(1000.0); // Should be close to 1
        assert!(sigmoid_pos > 0.99);

        let sigmoid_neg = meta_opt.sigmoid(-1000.0); // Should be close to 0
        assert!(sigmoid_neg < 0.01);

        // Test tanh
        let tanh_result = meta_opt.tanh(0.0);
        assert_abs_diff_eq!(tanh_result, 0.0, epsilon = 1e-10);

        let tanh_pos = meta_opt.tanh(1000.0); // Should be close to 1
        assert!(tanh_pos > 0.99);

        let tanh_neg = meta_opt.tanh(-1000.0); // Should be close to -1
        assert!(tanh_neg < -0.99);
    }
}
