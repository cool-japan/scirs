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
    maxvalue: F,
    /// Step size for discrete search
    step: Option<F>,
    /// Is the parameter categorical
    is_categorical: bool,
    /// Possible categorical values
    categorical_values: Option<Vec<F>>,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> HyperParameter<F> {
    /// Create a new continuous hyperparameter
    pub fn new<S: Into<String>>(name: S, value: F, min_value: F, maxvalue: F) -> Self {
        Self {
            name: name.into(),
            value,
            min_value,
            maxvalue,
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
        maxvalue: F,
        step: F,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            min_value,
            maxvalue,
            step: Some(step),
            is_categorical: false,
            categorical_values: None,
        }
    }

    /// Create a new categorical hyperparameter
    pub fn categorical<S: Into<String>>(name: S, value: F, values: Vec<F>) -> Result<Self> {
        if values.is_empty() {
            return Err(MetricsError::InvalidArgument(
                "Categorical values cannot be empty".to_string(),
            ));
        }
        if !values.contains(&value) {
            return Err(MetricsError::InvalidArgument(format!(
                "Current value {} must be one of the categorical values",
                value
            )));
        }

        Ok(Self {
            name: name.into(),
            value,
            min_value: F::zero(),
            maxvalue: F::from(values.len() - 1).unwrap(),
            step: Some(F::one()),
            is_categorical: true,
            categorical_values: Some(values),
        })
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
        } else if value < self.min_value || value > self.maxvalue {
            return Err(MetricsError::InvalidArgument(format!(
                "Value {} out of range [{}, {}] for parameter {}",
                value, self.min_value, self.maxvalue, self.name
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

        let range = self.maxvalue - self.min_value;
        let mut rng = rand::rng();
        let rand_val = F::from(rng.random::<f64>()).unwrap() * range + self.min_value;

        if let Some(step) = self.step {
            // For discrete parameters..round to the nearest step
            let steps = ((rand_val - self.min_value) / step).round();
            self.min_value + steps * step
        } else {
            rand_val
        }
    }

    /// Validate that the current parameter configuration is valid
    pub fn validate(&self) -> Result<()> {
        if self.is_categorical {
            if let Some(values) = &self.categorical_values {
                if values.is_empty() {
                    return Err(MetricsError::InvalidArgument(
                        "Categorical values cannot be empty".to_string(),
                    ));
                }
                if !values.contains(&self.value) {
                    return Err(MetricsError::InvalidArgument(format!(
                        "Current value {} is not in categorical values for parameter {}",
                        self.value, self.name
                    )));
                }
            } else {
                return Err(MetricsError::InvalidArgument(format!(
                    "Categorical parameter {} missing values",
                    self.name
                )));
            }
        } else {
            if self.min_value > self.maxvalue {
                return Err(MetricsError::InvalidArgument(format!(
                    "Min value {} cannot be greater than max value {} for parameter {}",
                    self.min_value, self.maxvalue, self.name
                )));
            }
            if self.value < self.min_value || self.value > self.maxvalue {
                return Err(MetricsError::InvalidArgument(format!(
                    "Current value {} is out of range [{}, {}] for parameter {}",
                    self.value, self.min_value, self.maxvalue, self.name
                )));
            }
            if let Some(step) = self.step {
                if step <= F::zero() {
                    return Err(MetricsError::InvalidArgument(format!(
                        "Step size must be positive for parameter {}",
                        self.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Get the valid range for this parameter
    pub fn get_range(&self) -> (F, F) {
        (self.min_value, self.maxvalue)
    }

    /// Get the step size (if discrete)
    pub fn get_step(&self) -> Option<F> {
        self.step
    }

    /// Check if parameter is categorical
    pub fn is_categorical(&self) -> bool {
        self.is_categorical
    }

    /// Get categorical values (if categorical)
    pub fn get_categorical_values(&self) -> Option<&Vec<F>> {
        self.categorical_values.as_ref()
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
    ) -> Result<Self> {
        if params.is_empty() {
            return Err(MetricsError::InvalidArgument(
                "Cannot create tuner with empty parameter list".to_string(),
            ));
        }

        if max_evals == 0 {
            return Err(MetricsError::InvalidArgument(
                "Maximum evaluations must be greater than 0".to_string(),
            ));
        }

        // Validate all parameters
        for param in &params {
            param.validate()?;
        }

        // Check for duplicate parameter names
        let mut names = std::collections::HashSet::new();
        for param in &params {
            if !names.insert(param.name()) {
                return Err(MetricsError::InvalidArgument(format!(
                    "Duplicate parameter name: {}",
                    param.name()
                )));
            }
        }

        Ok(Self {
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
        })
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
    pub fn update(&mut self, metricvalue: F) -> Result<bool> {
        let current_params = self.get_current_params();

        // Check if this is the best _value so far
        let is_best = match (self.best_value, self.mode) {
            (None, _) => true,
            (Some(best), OptimizationMode::Maximize) => metricvalue > best,
            (Some(best), OptimizationMode::Minimize) => metricvalue < best,
        };

        // Update history
        self.history.push((current_params.clone(), metricvalue));

        // Update best if this is the best so far
        if is_best {
            self.best_value = Some(metricvalue);
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
