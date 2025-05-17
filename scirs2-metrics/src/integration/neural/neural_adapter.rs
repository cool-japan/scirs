//! Neural metric adapter implementation
//!
//! This module provides an implementation of the neural interface
//! for metrics when the neural_common feature is enabled.

use crate::error::MetricsError;
use crate::integration::traits::MetricComputation;
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

/// Type alias for metric function
type MetricFn<F> = Box<dyn Fn(&Array<F, IxDyn>, &Array<F, IxDyn>) -> Result<F, MetricsError> + Send + Sync>;

/// Adapter for using metrics with neural network models
pub struct NeuralMetricAdapter<F: Float + Debug + Display + FromPrimitive> {
    /// The name of the metric
    pub name: String,
    metric_fn: MetricFn<F>,
    #[cfg(feature = "neural_common")]
    predictions: Option<Array<F, IxDyn>>,
    #[cfg(feature = "neural_common")]
    targets: Option<Array<F, IxDyn>>,
}

impl<F: Float + Debug + Display + FromPrimitive> std::fmt::Debug for NeuralMetricAdapter<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = f.debug_struct("NeuralMetricAdapter");
        builder.field("name", &self.name);
        builder.field("metric_fn", &"<function>");

        #[cfg(feature = "neural_common")]
        {
            builder.field("predictions", &self.predictions);
            builder.field("targets", &self.targets);
        }

        builder.finish()
    }
}

impl<F: Float + Debug + Display + FromPrimitive> NeuralMetricAdapter<F> {
    /// Create a new neural metric adapter
    pub fn new(
        name: &str,
        metric_fn: MetricFn<F>,
    ) -> Self {
        #[cfg(feature = "neural_common")]
        {
            Self {
                name: name.to_string(),
                metric_fn,
                predictions: None,
                targets: None,
            }
        }

        #[cfg(not(feature = "neural_common"))]
        {
            Self {
                name: name.to_string(),
                metric_fn,
            }
        }
    }

    /// Create an accuracy metric adapter
    pub fn accuracy() -> Self {
        Self::new(
            "accuracy",
            Box::new(|preds, targets| {
                // Convert to f64, calculate, and then convert back to F
                let preds_f64 = preds.mapv(|x| x.to_f64().unwrap_or(0.0));
                let targets_f64 = targets.mapv(|x| x.to_f64().unwrap_or(0.0));
                let result = crate::classification::accuracy_score(&targets_f64, &preds_f64)?;
                Ok(F::from(result).unwrap())
            }),
        )
    }

    /// Create a precision metric adapter
    pub fn precision() -> Self {
        Self::new(
            "precision",
            Box::new(|preds, targets| {
                // Convert to f64, calculate, and then convert back to F
                let preds_f64 = preds.mapv(|x| x.to_f64().unwrap_or(0.0));
                let targets_f64 = targets.mapv(|x| x.to_f64().unwrap_or(0.0));
                let pos_label = 1.0;
                let result =
                    crate::classification::precision_score(&preds_f64, &targets_f64, pos_label)?;
                Ok(F::from(result).unwrap())
            }),
        )
    }

    /// Create a recall metric adapter
    pub fn recall() -> Self {
        Self::new(
            "recall",
            Box::new(|preds, targets| {
                // Convert to f64, calculate, and then convert back to F
                let preds_f64 = preds.mapv(|x| x.to_f64().unwrap_or(0.0));
                let targets_f64 = targets.mapv(|x| x.to_f64().unwrap_or(0.0));
                let pos_label = 1.0;
                let result =
                    crate::classification::recall_score(&preds_f64, &targets_f64, pos_label)?;
                Ok(F::from(result).unwrap())
            }),
        )
    }

    /// Create an F1 score metric adapter
    pub fn f1_score() -> Self {
        Self::new(
            "f1_score",
            Box::new(|preds, targets| {
                // Convert to f64, calculate, and then convert back to F
                let preds_f64 = preds.mapv(|x| x.to_f64().unwrap_or(0.0));
                let targets_f64 = targets.mapv(|x| x.to_f64().unwrap_or(0.0));
                let pos_label = 1.0;
                let result = crate::classification::f1_score(&preds_f64, &targets_f64, pos_label)?;
                Ok(F::from(result).unwrap())
            }),
        )
    }

    /// Create a ROC AUC metric adapter
    pub fn roc_auc() -> Self {
        Self::new(
            "roc_auc",
            Box::new(|preds, targets| {
                // This is a special case, as roc_auc_score has specific type requirements
                // We need to convert targets to u32 and preds to f64
                let targets_u32 = targets.mapv(|x| x.to_f64().unwrap_or(0.0).round() as u32);
                let preds_f64 = preds.mapv(|x| x.to_f64().unwrap_or(0.0));
                let result = crate::classification::roc_auc_score(&targets_u32, &preds_f64)?;
                Ok(F::from(result).unwrap())
            }),
        )
    }

    /// Create a mean squared error metric adapter
    pub fn mse() -> Self {
        Self::new(
            "mse",
            Box::new(|preds, targets| {
                // Call to appropriate regression metric
                crate::regression::mean_squared_error(targets, preds)
            }),
        )
    }

    /// Create a mean absolute error metric adapter
    pub fn mae() -> Self {
        Self::new(
            "mae",
            Box::new(|preds, targets| {
                // Call to appropriate regression metric
                crate::regression::mean_absolute_error(targets, preds)
            }),
        )
    }

    /// Create an R-squared metric adapter
    pub fn r2() -> Self {
        Self::new(
            "r2",
            Box::new(|preds, targets| {
                // Call to appropriate regression metric
                crate::regression::r2_score(targets, preds)
            }),
        )
    }

    /// Create an explained variance metric adapter
    pub fn explained_variance() -> Self {
        Self::new(
            "explained_variance",
            Box::new(|preds, targets| {
                // Call to appropriate regression metric
                crate::regression::explained_variance_score(targets, preds)
            }),
        )
    }
}

impl<F: Float + Debug + Display + FromPrimitive> MetricComputation<F> for NeuralMetricAdapter<F> {
    fn compute(
        &self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<F, MetricsError> {
        (self.metric_fn)(predictions, targets)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[allow(unexpected_cfgs)]
#[cfg(all(feature = "neural_common", feature = "neural_integration"))]
mod neural_trait_impl {
    use super::*;

    impl<F: Float + Debug + Display + FromPrimitive + Send + Sync + 'static>
        scirs2_neural::evaluation::Metric<F> for NeuralMetricAdapter<F>
    {
        fn update(
            &mut self,
            predictions: &Array<F, IxDyn>,
            targets: &Array<F, IxDyn>,
            _loss: Option<F>,
        ) {
            // Store predictions and targets for later computation
            self.predictions = Some(predictions.clone());
            self.targets = Some(targets.clone());
        }

        fn reset(&mut self) {
            self.predictions = None;
            self.targets = None;
        }

        fn result(&self) -> F {
            if let (Some(preds), Some(targets)) = (&self.predictions, &self.targets) {
                self.compute(preds, targets).unwrap_or(F::zero())
            } else {
                F::zero()
            }
        }

        fn name(&self) -> &str {
            &self.name
        }
    }
}
