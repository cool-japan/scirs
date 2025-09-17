//! Few-Shot Learning Algorithms
//!
//! This module implements advanced few-shot learning techniques including
//! Prototypical Networks and REPTILE for rapid adaptation to new tasks
//! with minimal training data.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::TaskData;
use crate::error::Result;

/// Prototypical Networks for Few-Shot Learning
#[derive(Debug)]
pub struct PrototypicalNetworks<F: Float + Debug + ndarray::ScalarOperand> {
    /// Feature extraction network parameters
    feature_extractor: Array2<F>,
    /// Input dimension
    input_dim: usize,
    /// Feature dimension
    feature_dim: usize,
    /// Hidden dimensions for feature extractor
    hidden_dims: Vec<usize>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> PrototypicalNetworks<F> {
    /// Create new Prototypical Networks model
    pub fn new(input_dim: usize, feature_dim: usize, hidden_dims: Vec<usize>) -> Self {
        // Calculate total parameters for feature extractor
        let mut total_params = 0;
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(&hidden_dims);
        layer_sizes.push(feature_dim);

        for i in 0..layer_sizes.len() - 1 {
            total_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1];
            // weights + biases
        }

        // Initialize feature extractor parameters
        let scale = F::from(2.0).unwrap() / F::from(input_dim + feature_dim).unwrap();
        let std_dev = scale.sqrt();

        let mut feature_extractor = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 43) % 1000) as f64 / 1000.0 - 0.5;
            feature_extractor[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        Self {
            feature_extractor,
            input_dim,
            feature_dim,
            hidden_dims,
        }
    }

    /// Extract features from input data
    pub fn extract_features(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let batch_size_ = input.nrows();
        let mut current_input = input.clone();

        // Extract layer weights and biases
        let layer_params = self.extract_layer_parameters();

        // Forward pass through feature extractor
        for (weights, biases) in layer_params {
            let mut layer_output = Array2::zeros((batch_size_, biases.len()));

            // Apply linear transformation
            for i in 0..batch_size_ {
                for j in 0..biases.len() {
                    let mut sum = biases[j];
                    for k in 0..current_input.ncols() {
                        if k < weights.ncols() {
                            sum = sum + current_input[[i, k]] * weights[[j, k]];
                        }
                    }
                    layer_output[[i, j]] = self.relu(sum);
                }
            }

            current_input = layer_output;
        }

        Ok(current_input)
    }

    /// Compute prototypes for each class
    pub fn compute_prototypes(
        &self,
        support_features: &Array2<F>,
        support_labels: &Array1<usize>,
    ) -> Result<Array2<F>> {
        // Find unique classes
        let mut unique_classes = Vec::new();
        for &label in support_labels {
            if !unique_classes.contains(&label) {
                unique_classes.push(label);
            }
        }
        unique_classes.sort();

        let num_classes = unique_classes.len();
        let mut prototypes = Array2::zeros((num_classes, self.feature_dim));

        // Compute prototype for each class
        for (class_idx, &class_label) in unique_classes.iter().enumerate() {
            let mut class_features = Vec::new();
            for (i, &label) in support_labels.iter().enumerate() {
                if label == class_label {
                    class_features.push(support_features.row(i).to_owned());
                }
            }

            if !class_features.is_empty() {
                // Compute mean of class _features
                for j in 0..self.feature_dim {
                    let mut sum = F::zero();
                    for features in &class_features {
                        sum = sum + features[j];
                    }
                    prototypes[[class_idx, j]] = sum / F::from(class_features.len()).unwrap();
                }
            }
        }

        Ok(prototypes)
    }

    /// Classify query samples using prototypical networks
    pub fn classify_queries(
        &self,
        query_features: &Array2<F>,
        prototypes: &Array2<F>,
    ) -> Result<Array1<usize>> {
        let num_queries = query_features.nrows();
        let num_classes = prototypes.nrows();
        let mut predictions = Array1::zeros(num_queries);

        for i in 0..num_queries {
            let mut min_distance = F::infinity();
            let mut predicted_class = 0;

            // Find closest prototype
            for j in 0..num_classes {
                let distance = self.euclidean_distance(
                    &query_features.row(i).to_owned(),
                    &prototypes.row(j).to_owned(),
                )?;

                if distance < min_distance {
                    min_distance = distance;
                    predicted_class = j;
                }
            }

            predictions[i] = predicted_class;
        }

        Ok(predictions)
    }

    /// Few-shot learning episode
    pub fn few_shot_episode(
        &self,
        support_x: &Array2<F>,
        support_y: &Array1<usize>,
        query_x: &Array2<F>,
    ) -> Result<Array1<usize>> {
        // Extract features
        let support_features = self.extract_features(support_x)?;
        let query_features = self.extract_features(query_x)?;

        // Compute prototypes
        let prototypes = self.compute_prototypes(&support_features, support_y)?;

        // Classify queries
        self.classify_queries(&query_features, &prototypes)
    }

    /// Train the feature extractor on a batch of few-shot tasks
    pub fn meta_train(&mut self, episodes: &[FewShotEpisode<F>]) -> Result<F> {
        let mut total_loss = F::zero();
        let mut total_gradients = Array2::zeros(self.feature_extractor.dim());

        for episode in episodes {
            // Forward pass
            let predictions =
                self.few_shot_episode(&episode.support_x, &episode.support_y, &episode.query_x)?;

            // Compute loss (cross-entropy approximation)
            let mut episode_loss = F::zero();
            for (i, &pred) in predictions.iter().enumerate() {
                if i < episode.query_y.len() {
                    let target = episode.query_y[i];
                    if pred != target {
                        episode_loss = episode_loss + F::one();
                    }
                }
            }
            episode_loss = episode_loss / F::from(predictions.len()).unwrap();

            // Compute gradients (simplified numerical differentiation)
            let gradients = self.compute_gradients(episode)?;
            total_gradients = total_gradients + gradients;
            total_loss = total_loss + episode_loss;
        }

        // Update parameters
        let learning_rate = F::from(0.001).unwrap();
        let num_episodes = F::from(episodes.len()).unwrap();
        total_gradients = total_gradients / num_episodes;

        self.feature_extractor = self.feature_extractor.clone() - total_gradients * learning_rate;

        Ok(total_loss / num_episodes)
    }

    // Helper methods
    fn extract_layer_parameters(&self) -> Vec<(Array2<F>, Array1<F>)> {
        let param_vec = self.feature_extractor.row(0);
        let mut layer_params = Vec::new();
        let mut param_idx = 0;

        let mut layer_sizes = vec![self.input_dim];
        layer_sizes.extend(&self.hidden_dims);
        layer_sizes.push(self.feature_dim);

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Extract weights
            let mut weights = Array2::zeros((output_size, input_size));
            for j in 0..output_size {
                for k in 0..input_size {
                    if param_idx < param_vec.len() {
                        weights[[j, k]] = param_vec[param_idx];
                        param_idx += 1;
                    }
                }
            }

            // Extract biases
            let mut biases = Array1::zeros(output_size);
            for j in 0..output_size {
                if param_idx < param_vec.len() {
                    biases[j] = param_vec[param_idx];
                    param_idx += 1;
                }
            }

            layer_params.push((weights, biases));
        }

        layer_params
    }

    fn euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        let mut sum = F::zero();
        for i in 0..a.len().min(b.len()) {
            let diff = a[i] - b[i];
            sum = sum + diff * diff;
        }
        Ok(sum.sqrt())
    }

    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }

    fn compute_gradients(&self, episode: &FewShotEpisode<F>) -> Result<Array2<F>> {
        // Simplified gradient computation
        let epsilon = F::from(1e-5).unwrap();
        let mut gradients = Array2::zeros(self.feature_extractor.dim());

        let base_predictions =
            self.few_shot_episode(&episode.support_x, &episode.support_y, &episode.query_x)?;
        let mut base_loss = F::zero();
        for (i, &pred) in base_predictions.iter().enumerate() {
            if i < episode.query_y.len() && pred != episode.query_y[i] {
                base_loss = base_loss + F::one();
            }
        }

        // Numerical differentiation for each parameter
        for i in 0..self.feature_extractor.ncols() {
            let mut perturbed_extractor = self.feature_extractor.clone();
            perturbed_extractor[[0, i]] = perturbed_extractor[[0, i]] + epsilon;

            // Create temporary network with perturbed parameters
            let mut temp_network = self.clone();
            temp_network.feature_extractor = perturbed_extractor;

            let perturbed_predictions = temp_network.few_shot_episode(
                &episode.support_x,
                &episode.support_y,
                &episode.query_x,
            )?;
            let mut perturbed_loss = F::zero();
            for (j, &pred) in perturbed_predictions.iter().enumerate() {
                if j < episode.query_y.len() && pred != episode.query_y[j] {
                    perturbed_loss = perturbed_loss + F::one();
                }
            }

            gradients[[0, i]] = (perturbed_loss - base_loss) / epsilon;
        }

        Ok(gradients)
    }
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> Clone
    for PrototypicalNetworks<F>
{
    fn clone(&self) -> Self {
        Self {
            feature_extractor: self.feature_extractor.clone(),
            input_dim: self.input_dim,
            feature_dim: self.feature_dim,
            hidden_dims: self.hidden_dims.clone(),
        }
    }
}

/// Few-shot learning episode data structure
#[derive(Debug, Clone)]
pub struct FewShotEpisode<F: Float + Debug> {
    /// Support set inputs
    pub support_x: Array2<F>,
    /// Support set labels
    pub support_y: Array1<usize>,
    /// Query set inputs
    pub query_x: Array2<F>,
    /// Query set labels
    pub query_y: Array1<usize>,
}

impl<F: Float + Debug> FewShotEpisode<F> {
    /// Create a new few-shot episode
    pub fn new(
        support_x: Array2<F>,
        support_y: Array1<usize>,
        query_x: Array2<F>,
        query_y: Array1<usize>,
    ) -> Self {
        Self {
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    /// Get the number of support samples
    pub fn support_size(&self) -> usize {
        self.support_x.nrows()
    }

    /// Get the number of query samples
    pub fn query_size(&self) -> usize {
        self.query_x.nrows()
    }

    /// Get unique classes in the episode
    pub fn unique_classes(&self) -> Vec<usize> {
        let mut classes = Vec::new();
        for &label in &self.support_y {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        for &label in &self.query_y {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort();
        classes
    }
}

/// REPTILE Algorithm for Meta-Learning
#[derive(Debug)]
pub struct REPTILE<F: Float + Debug + ndarray::ScalarOperand> {
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

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> REPTILE<F> {
    /// Create new REPTILE instance
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
            let val = ((i * 59) % 1000) as f64 / 1000.0 - 0.5;
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

    /// REPTILE meta-training step
    pub fn meta_train(&mut self, tasks: &[TaskData<F>]) -> Result<F> {
        let mut total_loss = F::zero();
        let mut parameter_updates = Array2::zeros(self.parameters.dim());

        for task in tasks {
            // Store initial parameters
            let initial_params = self.parameters.clone();

            // Inner loop training on task
            let mut task_params = initial_params.clone();
            for _ in 0..self.inner_steps {
                let gradients = self.compute_task_gradients(&task_params, task)?;
                task_params = task_params - gradients * self.inner_lr;
            }

            // Compute task loss
            let task_loss = self.forward(&task_params, &task.support_x, &task.support_y)?;
            total_loss = total_loss + task_loss;

            // REPTILE update: move towards task-adapted parameters
            let update = task_params - initial_params;
            parameter_updates = parameter_updates + update;
        }

        // Meta-update: average parameter updates across tasks
        let num_tasks = F::from(tasks.len()).unwrap();
        parameter_updates = parameter_updates / num_tasks;
        total_loss = total_loss / num_tasks;

        // Update meta-parameters
        self.parameters = self.parameters.clone() + parameter_updates * self.meta_lr;

        Ok(total_loss)
    }

    /// Fast adaptation for new task (few-shot learning)
    pub fn fast_adapt(&self, support_x: &Array2<F>, support_y: &Array2<F>) -> Result<Array2<F>> {
        let task = TaskData {
            support_x: support_x.clone(),
            support_y: support_y.clone(),
            query_x: support_x.clone(),
            query_y: support_y.clone(),
        };

        // Inner loop adaptation
        let mut adapted_params = self.parameters.clone();
        for _ in 0..self.inner_steps {
            let gradients = self.compute_task_gradients(&adapted_params, &task)?;
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
    pub fn predict(&self, params: &Array2<F>, inputs: &Array2<F>) -> Result<Array2<F>> {
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

    /// Compute task-specific gradients
    fn compute_task_gradients(&self, params: &Array2<F>, task: &TaskData<F>) -> Result<Array2<F>> {
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

    /// Get current parameters
    pub fn parameters(&self) -> &Array2<F> {
        &self.parameters
    }

    /// Set parameters
    pub fn set_parameters(&mut self, parameters: Array2<F>) {
        self.parameters = parameters;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_prototypical_networks_creation() {
        let hidden_dims = vec![16, 32];
        let model = PrototypicalNetworks::<f64>::new(10, 8, hidden_dims.clone());

        assert_eq!(model.input_dim, 10);
        assert_eq!(model.feature_dim, 8);
        assert_eq!(model.hidden_dims, hidden_dims);
    }

    #[test]
    fn test_few_shot_episode() {
        let support_x =
            Array2::from_shape_vec((4, 3), (0..12).map(|i| i as f64).collect()).unwrap();
        let support_y = Array1::from_vec(vec![0, 0, 1, 1]);
        let query_x = Array2::from_shape_vec((2, 3), (12..18).map(|i| i as f64).collect()).unwrap();
        let query_y = Array1::from_vec(vec![0, 1]);

        let episode = FewShotEpisode::new(support_x, support_y, query_x, query_y);

        assert_eq!(episode.support_size(), 4);
        assert_eq!(episode.query_size(), 2);

        let classes = episode.unique_classes();
        assert_eq!(classes, vec![0, 1]);
    }

    #[test]
    fn test_prototypical_networks_features() {
        let model = PrototypicalNetworks::<f64>::new(5, 4, vec![8]);
        let input =
            Array2::from_shape_vec((3, 5), (0..15).map(|i| i as f64 * 0.1).collect()).unwrap();

        let features = model.extract_features(&input).unwrap();
        assert_eq!(features.dim(), (3, 4));

        // Check that features are finite
        for &val in features.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_prototypical_networks_classification() {
        let model = PrototypicalNetworks::<f64>::new(4, 6, vec![8]);

        let support_x =
            Array2::from_shape_vec((6, 4), (0..24).map(|i| i as f64 * 0.1).collect()).unwrap();
        let support_y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let query_x =
            Array2::from_shape_vec((2, 4), (24..32).map(|i| i as f64 * 0.1).collect()).unwrap();

        let predictions = model
            .few_shot_episode(&support_x, &support_y, &query_x)
            .unwrap();
        assert_eq!(predictions.len(), 2);

        // Predictions should be within valid class range
        for &pred in predictions.iter() {
            assert!(pred <= 1);
        }
    }

    #[test]
    fn test_reptile_creation() {
        let reptile = REPTILE::<f64>::new(5, 10, 3, 0.01, 0.1, 5);

        assert_eq!(reptile.input_dim, 5);
        assert_eq!(reptile.hidden_dim, 10);
        assert_eq!(reptile.output_dim, 3);
    }

    #[test]
    fn test_reptile_prediction() {
        let reptile = REPTILE::<f64>::new(4, 8, 2, 0.01, 0.1, 3);
        let input =
            Array2::from_shape_vec((3, 4), (0..12).map(|i| i as f64 * 0.1).collect()).unwrap();

        let output = reptile.predict(&reptile.parameters, &input).unwrap();
        assert_eq!(output.dim(), (3, 2));

        // Check that output is finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_reptile_fast_adapt() {
        let reptile = REPTILE::<f64>::new(3, 6, 2, 0.01, 0.1, 2);
        let support_x =
            Array2::from_shape_vec((4, 3), (0..12).map(|i| i as f64 * 0.2).collect()).unwrap();
        let support_y =
            Array2::from_shape_vec((4, 2), (0..8).map(|i| i as f64 * 0.1).collect()).unwrap();

        let adapted_params = reptile.fast_adapt(&support_x, &support_y).unwrap();
        assert_eq!(adapted_params.dim(), reptile.parameters.dim());

        // Adapted parameters should be different from original
        let params_changed = adapted_params
            .iter()
            .zip(reptile.parameters.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(params_changed);
    }

    #[test]
    fn test_euclidean_distance() {
        let model = PrototypicalNetworks::<f64>::new(3, 4, vec![]);
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let distance = model.euclidean_distance(&a, &b).unwrap();
        let expected = ((3.0_f64).powi(2) + (3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_prototypes() {
        let model = PrototypicalNetworks::<f64>::new(4, 3, vec![]);

        // Create simple features where we know the expected prototypes
        let features = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 1.0, // Class 0
                2.0, 2.0, 2.0, // Class 0
                3.0, 3.0, 3.0, // Class 1
                4.0, 4.0, 4.0, // Class 1
                5.0, 5.0, 5.0, // Class 1
                6.0, 6.0, 6.0, // Class 2
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);

        let prototypes = model.compute_prototypes(&features, &labels).unwrap();
        assert_eq!(prototypes.dim(), (3, 3)); // 3 classes, 3 features

        // Check class 0 prototype (mean of [1,1,1] and [2,2,2])
        assert_abs_diff_eq!(prototypes[[0, 0]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(prototypes[[0, 1]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(prototypes[[0, 2]], 1.5, epsilon = 1e-10);

        // Check class 1 prototype (mean of [3,3,3], [4,4,4], [5,5,5])
        assert_abs_diff_eq!(prototypes[[1, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(prototypes[[1, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(prototypes[[1, 2]], 4.0, epsilon = 1e-10);

        // Check class 2 prototype ([6,6,6])
        assert_abs_diff_eq!(prototypes[[2, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(prototypes[[2, 1]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(prototypes[[2, 2]], 6.0, epsilon = 1e-10);
    }
}
