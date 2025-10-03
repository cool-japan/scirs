//! Meta-learning algorithms for few-shot time series forecasting
//!
//! This module implements Model-Agnostic Meta-Learning (MAML) and related
//! meta-learning algorithms for rapid adaptation to new time series tasks.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::TaskData;
use crate::error::Result;

/// Model-Agnostic Meta-Learning (MAML) for few-shot time series forecasting
#[derive(Debug)]
pub struct MAML<F: Float + Debug + scirs2_core::ndarray::ScalarOperand> {
    /// Base model parameters
    parameters: Array2<F>,
    /// Meta-learning rate
    meta_lr: F,
    /// Inner loop learning rate
    inner_lr: F,
    /// Number of inner gradient steps
    inner_steps: usize,
    /// Model dimensions
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive + scirs2_core::ndarray::ScalarOperand> MAML<F> {
    /// Create new MAML instance
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        meta_lr: F,
        inner_lr: F,
        inner_steps: usize,
    ) -> Self {
        // Initialize parameters using Xavier initialization
        let total_params =
            input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim;
        let scale = F::from(2.0).unwrap() / F::from(input_dim + output_dim).unwrap();
        let std_dev = scale.sqrt();

        let mut parameters = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 17) % 1000) as f64 / 1000.0 - 0.5;
            parameters[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        Self {
            parameters,
            meta_lr,
            inner_lr,
            inner_steps,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Meta-training step with multiple tasks
    pub fn meta_train(&mut self, tasks: &[TaskData<F>]) -> Result<F> {
        let mut meta_gradients = Array2::zeros(self.parameters.dim());
        let mut total_loss = F::zero();

        for task in tasks {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(task)?;

            // Compute meta-gradient
            let task_loss = self.compute_meta_loss(&adapted_params, task)?;
            let task_gradient = self.compute_meta_gradient(&adapted_params, task)?;

            meta_gradients = meta_gradients + task_gradient;
            total_loss = total_loss + task_loss;
        }

        // Meta-update
        let num_tasks = F::from(tasks.len()).unwrap();
        meta_gradients = meta_gradients / num_tasks;
        total_loss = total_loss / num_tasks;

        // Update meta-parameters
        self.parameters = self.parameters.clone() - meta_gradients * self.meta_lr;

        Ok(total_loss)
    }

    /// Inner loop adaptation for a single task
    fn inner_loop_adaptation(&self, task: &TaskData<F>) -> Result<Array2<F>> {
        let mut adapted_params = self.parameters.clone();

        for _ in 0..self.inner_steps {
            let _loss = self.forward(&adapted_params, &task.support_x, &task.support_y)?;
            let gradients = self.compute_gradients(&adapted_params, task)?;
            adapted_params = adapted_params - gradients * self.inner_lr;
        }

        Ok(adapted_params)
    }

    /// Forward pass through neural network
    fn forward(&self, params: &Array2<F>, inputs: &Array2<F>, targets: &Array2<F>) -> Result<F> {
        let predictions = self.predict(params, inputs)?;

        // Mean squared error loss
        let mut loss = F::zero();
        let (batch_size, _) = predictions.dim();

        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let diff = predictions[[i, j]] - targets[[i, j]];
                loss = loss + diff * diff;
            }
        }

        Ok(loss / F::from(batch_size).unwrap())
    }

    /// Make predictions using current parameters
    fn predict(&self, params: &Array2<F>, inputs: &Array2<F>) -> Result<Array2<F>> {
        let (batch_size, _) = inputs.dim();

        // Extract weight matrices from flattened parameters
        let (w1, b1, w2, b2) = self.extract_weights(params);

        // Forward pass: input -> hidden -> output
        let mut hidden = Array2::zeros((batch_size, self.hidden_dim));

        // Input to hidden layer
        for i in 0..batch_size {
            for j in 0..self.hidden_dim {
                let mut sum = b1[j];
                for k in 0..self.input_dim {
                    sum = sum + inputs[[i, k]] * w1[[j, k]];
                }
                hidden[[i, j]] = self.relu(sum); // ReLU activation
            }
        }

        // Hidden to output layer
        let mut output = Array2::zeros((batch_size, self.output_dim));
        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let mut sum = b2[j];
                for k in 0..self.hidden_dim {
                    sum = sum + hidden[[i, k]] * w2[[j, k]];
                }
                output[[i, j]] = sum; // Linear output
            }
        }

        Ok(output)
    }

    /// Extract weight matrices from flattened parameter vector
    fn extract_weights(&self, params: &Array2<F>) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = params.row(0);
        let mut idx = 0;

        // W1: input_dim x hidden_dim
        let mut w1 = Array2::zeros((self.hidden_dim, self.input_dim));
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: hidden_dim
        let mut b1 = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W2: hidden_dim x output_dim
        let mut w2 = Array2::zeros((self.output_dim, self.hidden_dim));
        for i in 0..self.output_dim {
            for j in 0..self.hidden_dim {
                w2[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b2: output_dim
        let mut b2 = Array1::zeros(self.output_dim);
        for i in 0..self.output_dim {
            b2[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w2, b2)
    }

    /// ReLU activation function
    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }

    /// Compute gradients (simplified numerical differentiation)
    fn compute_gradients(&self, params: &Array2<F>, task: &TaskData<F>) -> Result<Array2<F>> {
        let epsilon = F::from(1e-5).unwrap();
        let mut gradients = Array2::zeros(params.dim());

        let base_loss = self.forward(params, &task.support_x, &task.support_y)?;

        for i in 0..params.ncols() {
            let mut perturbed_params = params.clone();
            perturbed_params[[0, i]] = perturbed_params[[0, i]] + epsilon;

            let perturbed_loss =
                self.forward(&perturbed_params, &task.support_x, &task.support_y)?;
            gradients[[0, i]] = (perturbed_loss - base_loss) / epsilon;
        }

        Ok(gradients)
    }

    /// Compute meta-gradient for meta-learning update
    fn compute_meta_gradient(
        &self,
        adapted_params: &Array2<F>,
        task: &TaskData<F>,
    ) -> Result<Array2<F>> {
        // Simplified meta-gradient computation
        let _meta_loss = self.forward(adapted_params, &task.query_x, &task.query_y)?;
        self.compute_gradients(
            adapted_params,
            &TaskData {
                support_x: task.query_x.clone(),
                support_y: task.query_y.clone(),
                query_x: task.query_x.clone(),
                query_y: task.query_y.clone(),
            },
        )
    }

    /// Compute meta-loss on query set
    fn compute_meta_loss(&self, adapted_params: &Array2<F>, task: &TaskData<F>) -> Result<F> {
        self.forward(adapted_params, &task.query_x, &task.query_y)
    }

    /// Fast adaptation for new task (few-shot learning)
    pub fn fast_adapt(&self, support_x: &Array2<F>, support_y: &Array2<F>) -> Result<Array2<F>> {
        let task = TaskData {
            support_x: support_x.clone(),
            support_y: support_y.clone(),
            query_x: support_x.clone(),
            query_y: support_y.clone(),
        };

        self.inner_loop_adaptation(&task)
    }
}
