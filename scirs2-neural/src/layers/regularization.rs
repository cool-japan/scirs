//! Activity regularization layers implementation
//!
//! This module provides implementations of activity regularization techniques
//! such as L1 and L2 activity regularization.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Activity regularization layer
///
/// Applies L1 and/or L2 regularization to the activations.
/// This encourages sparse or small activations in the network.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{ActivityRegularization, Layer};
/// use ndarray::{Array, Array2};
///
/// // Create an activity regularization layer with L1=0.01 and L2=0.01
/// let regularizer = ActivityRegularization::new(Some(0.01), Some(0.01), Some("activity_reg")).unwrap();
///
/// // Forward pass with a batch of 2 samples, 10 features
/// let batch_size = 2;
/// let features = 10;
/// let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();
///
/// // Forward pass
/// let output = regularizer.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
#[derive(Debug, Clone)]
pub struct ActivityRegularization<F: Float + Debug + Send + Sync> {
    /// L1 regularization factor (None to disable)
    l1_factor: Option<F>,
    /// L2 regularization factor (None to disable)
    l2_factor: Option<F>,
    /// Name of the layer
    name: Option<String>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Activity loss cache
    activity_loss: Arc<RwLock<F>>,
    /// Phantom data for type parameter
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> ActivityRegularization<F> {
    /// Create a new activity regularization layer
    ///
    /// # Arguments
    ///
    /// * `l1_factor` - L1 regularization factor (None to disable L1)
    /// * `l2_factor` - L2 regularization factor (None to disable L2)
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    ///
    /// * A new activity regularization layer
    pub fn new(l1_factor: Option<f64>, l2_factor: Option<f64>, name: Option<&str>) -> Result<Self> {
        // Validate that at least one regularization factor is provided
        if l1_factor.is_none() && l2_factor.is_none() {
            return Err(NeuralError::InvalidArchitecture(
                "At least one of L1 or L2 regularization factor must be provided".to_string(),
            ));
        }

        // Validate that factors are non-negative
        if let Some(l1) = l1_factor {
            if l1 < 0.0 {
                return Err(NeuralError::InvalidArchitecture(
                    "L1 regularization factor must be non-negative".to_string(),
                ));
            }
        }

        if let Some(l2) = l2_factor {
            if l2 < 0.0 {
                return Err(NeuralError::InvalidArchitecture(
                    "L2 regularization factor must be non-negative".to_string(),
                ));
            }
        }

        Ok(Self {
            l1_factor: l1_factor.map(|x| F::from(x).unwrap()),
            l2_factor: l2_factor.map(|x| F::from(x).unwrap()),
            name: name.map(String::from),
            input_cache: Arc::new(RwLock::new(None)),
            activity_loss: Arc::new(RwLock::new(F::zero())),
            _phantom: PhantomData,
        })
    }

    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the current activity loss
    pub fn get_activity_loss(&self) -> Result<F> {
        match self.activity_loss.read() {
            Ok(loss) => Ok(*loss),
            Err(_) => Err(NeuralError::InferenceError(
                "Failed to acquire read lock on activity loss".to_string(),
            )),
        }
    }

    /// Calculate activity regularization loss
    fn calculate_activity_loss(&self, input: &Array<F, IxDyn>) -> F {
        let mut total_loss = F::zero();

        // L1 regularization (sum of absolute values)
        if let Some(l1_factor) = self.l1_factor {
            let l1_loss = input.mapv(|x| x.abs()).sum();
            total_loss = total_loss + l1_factor * l1_loss;
        }

        // L2 regularization (sum of squared values)
        if let Some(l2_factor) = self.l2_factor {
            let l2_loss = input.mapv(|x| x * x).sum();
            total_loss = total_loss + l2_factor * l2_loss;
        }

        total_loss
    }

    /// Calculate gradients for activity regularization
    fn calculate_activity_gradients(&self, input: &Array<F, IxDyn>) -> Array<F, IxDyn> {
        let mut grad = Array::<F, IxDyn>::zeros(input.raw_dim());

        // L1 regularization gradient (sign of activations)
        if let Some(l1_factor) = self.l1_factor {
            let l1_grad = input.mapv(|x| {
                if x > F::zero() {
                    l1_factor
                } else if x < F::zero() {
                    -l1_factor
                } else {
                    F::zero()
                }
            });
            grad = grad + l1_grad;
        }

        // L2 regularization gradient (2 * activations)
        if let Some(l2_factor) = self.l2_factor {
            let two = F::from(2.0).unwrap();
            let l2_grad = input.mapv(|x| two * l2_factor * x);
            grad = grad + l2_grad;
        }

        grad
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F>
    for ActivityRegularization<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        // Calculate and cache activity loss
        let loss = self.calculate_activity_loss(input);
        if let Ok(mut loss_cache) = self.activity_loss.write() {
            *loss_cache = loss;
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on activity loss cache".to_string(),
            ));
        }

        // Activity regularization doesn't modify the activations during forward pass
        // The regularization is applied as an additional loss term
        Ok(input.clone())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached input
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };
        if input_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            ));
        }
        let cached_input = input_ref.as_ref().unwrap();

        // Check shapes match
        if cached_input.shape() != grad_output.shape() {
            return Err(NeuralError::InferenceError(
                "Input and gradient output shapes must match".to_string(),
            ));
        }

        // Calculate activity regularization gradients
        let activity_grad = self.calculate_activity_gradients(cached_input);

        // Add activity regularization gradients to the incoming gradients
        Ok(grad_output + &activity_grad)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // ActivityRegularization has no learnable parameters
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "ActivityRegularization"
    }

    fn parameter_count(&self) -> usize {
        // ActivityRegularization has no parameters
        0
    }

    fn layer_description(&self) -> String {
        let l1_str = match self.l1_factor {
            Some(l1) => format!("{:?}", l1),
            None => "None".to_string(),
        };
        let l2_str = match self.l2_factor {
            Some(l2) => format!("{:?}", l2),
            None => "None".to_string(),
        };

        format!(
            "type:ActivityRegularization, l1:{}, l2:{}, name:{}",
            l1_str,
            l2_str,
            self.name.as_ref().map_or("None", |s| s)
        )
    }
}

/// L1 Activity Regularization layer
///
/// A convenience layer that applies only L1 regularization to activations.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{L1ActivityRegularization, Layer};
/// use ndarray::{Array, Array2};
///
/// // Create an L1 activity regularization layer with factor 0.01
/// let regularizer = L1ActivityRegularization::new(0.01, Some("l1_reg")).unwrap();
///
/// // Forward pass with a batch of 2 samples, 10 features
/// let batch_size = 2;
/// let features = 10;
/// let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();
///
/// // Forward pass
/// let output = regularizer.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
#[derive(Debug, Clone)]
pub struct L1ActivityRegularization<F: Float + Debug + Send + Sync> {
    inner: ActivityRegularization<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> L1ActivityRegularization<F> {
    /// Create a new L1 activity regularization layer
    ///
    /// # Arguments
    ///
    /// * `factor` - L1 regularization factor
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    ///
    /// * A new L1 activity regularization layer
    pub fn new(factor: f64, name: Option<&str>) -> Result<Self> {
        Ok(Self {
            inner: ActivityRegularization::new(Some(factor), None, name)?,
        })
    }

    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    /// Get the current activity loss
    pub fn get_activity_loss(&self) -> Result<F> {
        self.inner.get_activity_loss()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F>
    for L1ActivityRegularization<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.inner.forward(input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        self.inner.backward(input, grad_output)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.inner.update(learning_rate)
    }

    fn layer_type(&self) -> &str {
        "L1ActivityRegularization"
    }

    fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    fn layer_description(&self) -> String {
        self.inner
            .layer_description()
            .replace("ActivityRegularization", "L1ActivityRegularization")
    }
}

/// L2 Activity Regularization layer
///
/// A convenience layer that applies only L2 regularization to activations.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{L2ActivityRegularization, Layer};
/// use ndarray::{Array, Array2};
///
/// // Create an L2 activity regularization layer with factor 0.01
/// let regularizer = L2ActivityRegularization::new(0.01, Some("l2_reg")).unwrap();
///
/// // Forward pass with a batch of 2 samples, 10 features
/// let batch_size = 2;
/// let features = 10;
/// let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();
///
/// // Forward pass
/// let output = regularizer.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
#[derive(Debug, Clone)]
pub struct L2ActivityRegularization<F: Float + Debug + Send + Sync> {
    inner: ActivityRegularization<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> L2ActivityRegularization<F> {
    /// Create a new L2 activity regularization layer
    ///
    /// # Arguments
    ///
    /// * `factor` - L2 regularization factor
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    ///
    /// * A new L2 activity regularization layer
    pub fn new(factor: f64, name: Option<&str>) -> Result<Self> {
        Ok(Self {
            inner: ActivityRegularization::new(None, Some(factor), name)?,
        })
    }

    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    /// Get the current activity loss
    pub fn get_activity_loss(&self) -> Result<F> {
        self.inner.get_activity_loss()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F>
    for L2ActivityRegularization<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.inner.forward(input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        self.inner.backward(input, grad_output)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.inner.update(learning_rate)
    }

    fn layer_type(&self) -> &str {
        "L2ActivityRegularization"
    }

    fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    fn layer_description(&self) -> String {
        self.inner
            .layer_description()
            .replace("ActivityRegularization", "L2ActivityRegularization")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_activity_regularization_creation() {
        // Test L1 only
        let l1_reg = ActivityRegularization::<f64>::new(Some(0.01), None, Some("l1")).unwrap();
        assert!(l1_reg.l1_factor.is_some());
        assert!(l1_reg.l2_factor.is_none());

        // Test L2 only
        let l2_reg = ActivityRegularization::<f64>::new(None, Some(0.02), Some("l2")).unwrap();
        assert!(l2_reg.l1_factor.is_none());
        assert!(l2_reg.l2_factor.is_some());

        // Test both L1 and L2
        let both_reg =
            ActivityRegularization::<f64>::new(Some(0.01), Some(0.02), Some("both")).unwrap();
        assert!(both_reg.l1_factor.is_some());
        assert!(both_reg.l2_factor.is_some());

        // Test error when neither provided
        assert!(ActivityRegularization::<f64>::new(None, None, Some("none")).is_err());
    }

    #[test]
    fn test_activity_regularization_forward() {
        let reg = ActivityRegularization::<f64>::new(Some(0.01), Some(0.02), Some("test")).unwrap();

        let input = Array2::<f64>::from_elem((2, 3), 1.0);
        let input_dyn = input.clone().into_dyn();
        let output = reg.forward(&input_dyn).unwrap();

        // Forward pass should not modify the input
        assert_eq!(input.into_dyn().shape(), output.shape());
        for (a, b) in input_dyn.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_activity_regularization_backward() {
        let reg = ActivityRegularization::<f64>::new(Some(0.1), Some(0.1), Some("test")).unwrap();

        let input = array![[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]].into_dyn();
        let grad_output = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn();

        // First do forward pass to cache input
        let _output = reg.forward(&input).unwrap();

        // Then do backward pass
        let grad_input = reg.backward(&input, &grad_output).unwrap();

        // Gradient should include regularization terms
        assert_eq!(grad_input.shape(), input.shape());
        // For positive values, L1 contributes +0.1, L2 contributes +0.2*value
        // For negative values, L1 contributes -0.1, L2 contributes +0.2*value
    }

    #[test]
    fn test_l1_activity_regularization() {
        let reg = L1ActivityRegularization::<f64>::new(0.01, Some("l1_test")).unwrap();

        let input = Array2::<f64>::from_elem((2, 3), 2.0);
        let input_dyn = input.clone().into_dyn();
        let output = reg.forward(&input_dyn).unwrap();

        // Forward pass should not modify the input
        assert_eq!(input.into_dyn().shape(), output.shape());

        // Check that activity loss is calculated
        let loss = reg.get_activity_loss().unwrap();
        assert!(loss > 0.0); // Should have positive loss for positive activations
    }

    #[test]
    fn test_l2_activity_regularization() {
        let reg = L2ActivityRegularization::<f64>::new(0.01, Some("l2_test")).unwrap();

        let input = Array2::<f64>::from_elem((2, 3), 2.0);
        let input_dyn = input.clone().into_dyn();
        let output = reg.forward(&input_dyn).unwrap();

        // Forward pass should not modify the input
        assert_eq!(input.into_dyn().shape(), output.shape());

        // Check that activity loss is calculated
        let loss = reg.get_activity_loss().unwrap();
        assert!(loss > 0.0); // Should have positive loss for positive activations
    }

    #[test]
    fn test_activity_loss_calculation() {
        let reg = ActivityRegularization::<f64>::new(Some(0.1), Some(0.1), Some("test")).unwrap();

        // Test with known values
        let input = array![[1.0, -1.0], [2.0, 0.0]].into_dyn();
        let _output = reg.forward(&input).unwrap();

        let loss = reg.get_activity_loss().unwrap();

        // L1 loss: |1| + |-1| + |2| + |0| = 4
        // L2 loss: 1^2 + (-1)^2 + 2^2 + 0^2 = 6
        // Total: 0.1 * 4 + 0.1 * 6 = 1.0
        assert!((loss - 1.0).abs() < 1e-10);
    }
}
