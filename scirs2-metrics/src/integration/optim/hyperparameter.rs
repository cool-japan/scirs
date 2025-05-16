//! Hyperparameter tuning utilities
//!
//! This module provides utilities for hyperparameter tuning using metrics.

use crate::error::{MetricsError, Result};
use crate::integration::optim::OptimizationMode;
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// A hyperparameter with its range
#[derive(Debug, Clone)]
pub struct HyperParameter<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Name of the hyperparameter
    name: String,
    /// Current value
    value: F,
    /// Minimum value (inclusive)
    min_value: F,
    /// Maximum value (inclusive)
    max_value: F,
    /// Step size for discrete search
    step: Option<F>,
    /// Is the parameter categorical
    is_categorical: bool,
    /// Possible categorical values
    categorical_values: Option<Vec<F>>,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> HyperParameter<F> {
    /// Create a new continuous hyperparameter
    pub fn new<S: Into<String>>(name: S, value: F, min_value: F, max_value: F) -> Self {
        Self {
            name: name.into(),
            value,
            min_value,
            max_value,
            step: None,
            is_categorical: false,
            categorical_values: None,
        }
    }

    /// Create a new discrete hyperparameter
    pub fn discrete<S: Into<String>>(
        name: S,
        value: F,
        min_value: F,
        max_value: F,
        step: F,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            min_value,
            max_value,
            step: Some(step),
            is_categorical: false,
            categorical_values: None,
        }
    }

    /// Create a new categorical hyperparameter
    pub fn categorical<S: Into<String>>(name: S, value: F, values: Vec<F>) -> Self {
        if values.is_empty() {
            panic!("Categorical values cannot be empty");
        }
        if !values.contains(&value) {
            panic!("Current value must be one of the categorical values");
        }

        Self {
            name: name.into(),
            value,
            min_value: F::zero(),
            max_value: F::from(values.len() - 1).unwrap(),
            step: Some(F::one()),
            is_categorical: true,
            categorical_values: Some(values),
        }
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the current value
    pub fn value(&self) -> F {
        self.value
    }

    /// Set the value
    pub fn set_value(&mut self, value: F) -> Result<()> {
        if self.is_categorical {
            if let Some(values) = &self.categorical_values {
                if !values.contains(&value) {
                    return Err(MetricsError::InvalidArgument(format!(
                        "Value {} is not a valid categorical value for parameter {}",
                        value, self.name
                    )));
                }
            }
        } else if value < self.min_value || value > self.max_value {
            return Err(MetricsError::InvalidArgument(format!(
                "Value {} out of range [{}, {}] for parameter {}",
                value, self.min_value, self.max_value, self.name
            )));
        }

        self.value = value;
        Ok(())
    }

    /// Get a random value within the parameter's range
    pub fn random_value(&self) -> F {
        if self.is_categorical {
            if let Some(values) = &self.categorical_values {
                let mut rng = rand::rng();
                let idx = rng.random_range(0..values.len());
                return values[idx];
            }
        }

        let range = self.max_value - self.min_value;
        let mut rng = rand::rng();
        let rand_val = F::from(rng.random::<f64>()).unwrap() * range + self.min_value;

        if let Some(step) = self.step {
            // For discrete parameters, round to the nearest step
            let steps = ((rand_val - self.min_value) / step).round();
            self.min_value + steps * step
        } else {
            rand_val
        }
    }
}

/// A hyperparameter search result
#[derive(Debug, Clone)]
pub struct HyperParameterSearchResult<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Metric name that was optimized
    #[allow(dead_code)]
    metric_name: String,
    /// Optimization mode used
    mode: OptimizationMode,
    /// Best metric value found
    best_metric: F,
    /// Best hyperparameter values found
    best_params: HashMap<String, F>,
    /// History of all evaluations
    history: Vec<(HashMap<String, F>, F)>,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> HyperParameterSearchResult<F> {
    /// Create a new hyperparameter search result
    pub fn new<S: Into<String>>(
        metric_name: S,
        mode: OptimizationMode,
        best_metric: F,
        best_params: HashMap<String, F>,
    ) -> Self {
        Self {
            metric_name: metric_name.into(),
            mode,
            best_metric,
            best_params,
            history: Vec::new(),
        }
    }

    /// Add an evaluation to the history
    pub fn add_evaluation(&mut self, params: HashMap<String, F>, metric: F) {
        self.history.push((params.clone(), metric));

        // Update best if better
        let is_better = match self.mode {
            OptimizationMode::Maximize => metric > self.best_metric,
            OptimizationMode::Minimize => metric < self.best_metric,
        };

        if is_better {
            self.best_metric = metric;
            self.best_params = params;
        }
    }

    /// Get the best metric value
    pub fn best_metric(&self) -> F {
        self.best_metric
    }

    /// Get the best hyperparameter values
    pub fn best_params(&self) -> &HashMap<String, F> {
        &self.best_params
    }

    /// Get the history of evaluations
    pub fn history(&self) -> &[(HashMap<String, F>, F)] {
        &self.history
    }
}

/// A hyperparameter tuner
#[derive(Debug)]
pub struct HyperParameterTuner<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Hyperparameters to tune
    params: Vec<HyperParameter<F>>,
    /// Metric name to optimize
    metric_name: String,
    /// Optimization mode
    mode: OptimizationMode,
    /// Maximum number of evaluations
    max_evals: usize,
    /// Current best value
    best_value: Option<F>,
    /// Current best parameters
    best_params: HashMap<String, F>,
    /// History of evaluations
    history: Vec<(HashMap<String, F>, F)>,
    /// Phantom data for F type
    _phantom: PhantomData<F>,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> HyperParameterTuner<F> {
    /// Create a new hyperparameter tuner
    pub fn new<S: Into<String>>(
        params: Vec<HyperParameter<F>>,
        metric_name: S,
        maximize: bool,
        max_evals: usize,
    ) -> Self {
        Self {
            params,
            metric_name: metric_name.into(),
            mode: if maximize {
                OptimizationMode::Maximize
            } else {
                OptimizationMode::Minimize
            },
            max_evals,
            best_value: None,
            best_params: HashMap::new(),
            history: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Get the current hyperparameter values
    pub fn get_current_params(&self) -> HashMap<String, F> {
        self.params
            .iter()
            .map(|p| (p.name().to_string(), p.value()))
            .collect()
    }

    /// Set hyperparameter values
    pub fn set_params(&mut self, params: &HashMap<String, F>) -> Result<()> {
        for (name, value) in params {
            if let Some(param) = self.params.iter_mut().find(|p| p.name() == name) {
                param.set_value(*value)?;
            }
        }
        Ok(())
    }

    /// Update the tuner with an evaluation result
    pub fn update(&mut self, metric_value: F) -> Result<bool> {
        let current_params = self.get_current_params();

        // Check if this is the best value so far
        let is_best = match (self.best_value, self.mode) {
            (None, _) => true,
            (Some(best), OptimizationMode::Maximize) => metric_value > best,
            (Some(best), OptimizationMode::Minimize) => metric_value < best,
        };

        // Update history
        self.history.push((current_params.clone(), metric_value));

        // Update best if this is the best so far
        if is_best {
            self.best_value = Some(metric_value);
            self.best_params = current_params;
        }

        Ok(is_best)
    }

    /// Get a random set of hyperparameters
    pub fn random_params(&self) -> HashMap<String, F> {
        self.params
            .iter()
            .map(|p| (p.name().to_string(), p.random_value()))
            .collect()
    }

    /// Run random search for hyperparameter tuning
    pub fn random_search<FnEval>(
        &mut self,
        eval_fn: FnEval,
    ) -> Result<HyperParameterSearchResult<F>>
    where
        FnEval: Fn(&HashMap<String, F>) -> Result<F>,
    {
        // Reset history
        self.history.clear();
        self.best_value = None;

        for _ in 0..self.max_evals {
            // Get random parameters
            let params = self.random_params();

            // Set parameters
            self.set_params(&params)?;

            // Evaluate
            let metric = eval_fn(&params)?;

            // Update
            self.update(metric)?;
        }

        // Create result
        let result = HyperParameterSearchResult::new(
            self.metric_name.clone(),
            self.mode,
            self.best_value.unwrap_or_else(|| match self.mode {
                OptimizationMode::Maximize => F::neg_infinity(),
                OptimizationMode::Minimize => F::infinity(),
            }),
            self.best_params.clone(),
        );

        Ok(result)
    }

    /// Get the best parameters found so far
    pub fn best_params(&self) -> &HashMap<String, F> {
        &self.best_params
    }

    /// Get the best metric value found so far
    pub fn best_value(&self) -> Option<F> {
        self.best_value
    }

    /// Get the history of evaluations
    pub fn history(&self) -> &[(HashMap<String, F>, F)] {
        &self.history
    }
}
