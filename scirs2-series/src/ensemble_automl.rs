//! Ensemble Forecasting Methods and AutoML for Time Series
//!
//! This module provides advanced ensemble methods and automated machine learning
//! capabilities for time series forecasting, including model selection, hyperparameter
//! optimization, and ensemble combination strategies.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Ensemble combination strategies
#[derive(Debug, Clone)]
pub enum EnsembleStrategy {
    /// Simple average of all models
    SimpleAverage,
    /// Weighted average based on historical performance
    WeightedAverage,
    /// Dynamic weight adjustment based on recent performance
    DynamicWeighting {
        /// Window size for weight calculation
        window_size: usize,
    },
    /// Stacking with meta-learner
    Stacking {
        /// Meta-learner type for stacking
        meta_learner: MetaLearner,
    },
    /// Bayesian Model Averaging
    BayesianModelAveraging,
    /// Median ensemble
    Median,
    /// Best model selection
    BestModel,
}

/// Meta-learner for stacking ensemble
#[derive(Debug, Clone)]
pub enum MetaLearner {
    /// Linear regression meta-learner
    LinearRegression,
    /// Ridge regression with regularization
    RidgeRegression {
        /// Regularization parameter
        alpha: f64,
    },
    /// Random forest meta-learner
    RandomForest {
        /// Number of trees in the forest
        n_trees: usize,
    },
    /// Neural network meta-learner
    NeuralNetwork {
        /// Number of hidden units in each layer
        hidden_units: Vec<usize>,
    },
}

/// Base forecasting model type
#[derive(Debug, Clone)]
pub enum BaseModel {
    /// ARIMA model with orders
    ARIMA {
        /// Autoregressive order
        p: usize,
        /// Differencing order
        d: usize,
        /// Moving average order
        q: usize,
    },
    /// Exponential smoothing
    ExponentialSmoothing {
        /// Level smoothing parameter
        alpha: f64,
        /// Trend smoothing parameter
        beta: Option<f64>,
        /// Seasonal smoothing parameter
        gamma: Option<f64>,
    },
    /// Linear trend model
    LinearTrend,
    /// Seasonal naive
    SeasonalNaive {
        /// Seasonal period
        period: usize,
    },
    /// Moving average
    MovingAverage {
        /// Window size for moving average
        window: usize,
    },
    /// LSTM neural network
    LSTM {
        /// Number of hidden units
        hidden_units: usize,
        /// Number of layers
        num_layers: usize,
    },
    /// Prophet-like model
    Prophet {
        /// Seasonality prior scale
        seasonality_prior_scale: f64,
        /// Changepoint prior scale
        changepoint_prior_scale: f64,
    },
}

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelPerformance<F: Float> {
    /// Mean Absolute Error
    pub mae: F,
    /// Mean Squared Error
    pub mse: F,
    /// Root Mean Squared Error
    pub rmse: F,
    /// Mean Absolute Percentage Error
    pub mape: F,
    /// Symmetric MAPE
    pub smape: F,
    /// Mean Absolute Scaled Error
    pub mase: F,
    /// Model weight for ensemble
    pub weight: F,
    /// Confidence score
    pub confidence: F,
}

impl<F: Float + FromPrimitive> Default for ModelPerformance<F> {
    fn default() -> Self {
        Self {
            mae: F::infinity(),
            mse: F::infinity(),
            rmse: F::infinity(),
            mape: F::infinity(),
            smape: F::infinity(),
            mase: F::infinity(),
            weight: F::zero(),
            confidence: F::zero(),
        }
    }
}

/// Ensemble forecasting system
#[derive(Debug)]
pub struct EnsembleForecaster<F: Float + Debug + Clone + FromPrimitive + std::iter::Sum + 'static> {
    /// Base models in the ensemble
    base_models: Vec<BaseModelWrapper<F>>,
    /// Ensemble strategy
    strategy: EnsembleStrategy,
    /// Model performance history
    #[allow(dead_code)]
    performance_history: VecDeque<Vec<ModelPerformance<F>>>,
    /// Training data buffer
    training_buffer: VecDeque<F>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Validation window size
    #[allow(dead_code)]
    validation_window: usize,
    /// Current ensemble weights
    ensemble_weights: Array1<F>,
    /// Meta-model for stacking
    meta_model: Option<Box<dyn MetaModel<F>>>,
}

/// Wrapper for base models with their state
#[derive(Debug, Clone)]
pub struct BaseModelWrapper<F: Float + Debug> {
    /// Model type
    model_type: BaseModel,
    /// Model state/parameters
    model_state: ModelState<F>,
    /// Recent predictions
    #[allow(dead_code)]
    recent_predictions: VecDeque<F>,
    /// Performance metrics
    performance: ModelPerformance<F>,
    /// Is model trained
    is_trained: bool,
}

/// Generic model state
#[derive(Debug, Clone)]
pub struct ModelState<F: Float> {
    /// Model parameters
    parameters: HashMap<String, F>,
    /// Internal state variables
    state_variables: HashMap<String, Array1<F>>,
    /// Model specific data
    #[allow(dead_code)]
    model_data: Option<Array2<F>>,
}

impl<F: Float + Clone> Default for ModelState<F> {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            state_variables: HashMap::new(),
            model_data: None,
        }
    }
}

/// Trait for meta-models used in stacking
pub trait MetaModel<F: Float + Debug>: Debug {
    /// Train the meta-model on base model predictions
    fn train(&mut self, predictions: &Array2<F>, targets: &Array1<F>) -> Result<()>;

    /// Predict using the meta-model
    fn predict(&self, predictions: &Array2<F>) -> Result<Array1<F>>;

    /// Get model confidence
    fn confidence(&self) -> F;
}

/// Linear regression meta-model
#[derive(Debug)]
pub struct LinearRegressionMeta<F: Float + Debug> {
    /// Regression coefficients
    coefficients: Array1<F>,
    /// Intercept term
    intercept: F,
    /// Model confidence
    confidence: F,
    /// Is trained
    is_trained: bool,
}

impl<F: Float + Debug + Clone + FromPrimitive> LinearRegressionMeta<F> {
    /// Create new linear regression meta-model
    pub fn new() -> Self {
        Self {
            coefficients: Array1::zeros(0),
            intercept: F::zero(),
            confidence: F::zero(),
            is_trained: false,
        }
    }
}

impl<F: Float + Debug + Clone + FromPrimitive> Default for LinearRegressionMeta<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + Clone + FromPrimitive> MetaModel<F> for LinearRegressionMeta<F> {
    fn train(&mut self, predictions: &Array2<F>, targets: &Array1<F>) -> Result<()> {
        let (n_samples, n_models) = predictions.dim();

        if targets.len() != n_samples {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: targets.len(),
            });
        }

        if n_samples < n_models {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need more samples than models for training".to_string(),
                required: n_models,
                actual: n_samples,
            });
        }

        // Simple least squares solution: (X^T X)^-1 X^T y
        // For simplicity, use normal equations
        let mut xtx = Array2::zeros((n_models, n_models));
        let mut xty = Array1::zeros(n_models);

        // Compute X^T X and X^T y
        for i in 0..n_models {
            for j in 0..n_models {
                let mut sum = F::zero();
                for k in 0..n_samples {
                    sum = sum + predictions[[k, i]] * predictions[[k, j]];
                }
                xtx[[i, j]] = sum;
            }

            let mut sum = F::zero();
            for k in 0..n_samples {
                sum = sum + predictions[[k, i]] * targets[k];
            }
            xty[i] = sum;
        }

        // Solve linear system (simplified - use diagonal approximation for robustness)
        self.coefficients = Array1::zeros(n_models);
        for i in 0..n_models {
            if xtx[[i, i]] > F::zero() {
                self.coefficients[i] = xty[i] / xtx[[i, i]];
            }
        }

        // Compute intercept
        let mut sum_coeff = F::zero();
        for &coeff in self.coefficients.iter() {
            sum_coeff = sum_coeff + coeff;
        }
        self.intercept = if sum_coeff < F::one() {
            (F::one() - sum_coeff) / F::from(n_models).unwrap()
        } else {
            F::zero()
        };

        // Estimate confidence based on fit quality
        // Temporarily set is_trained to allow predict call during training
        self.is_trained = true;
        let predictions_result = self.predict(predictions)?;
        let mut mse = F::zero();
        for i in 0..n_samples {
            let error = targets[i] - predictions_result[i];
            mse = mse + error * error;
        }
        mse = mse / F::from(n_samples).unwrap();

        self.confidence = F::one() / (F::one() + mse);

        Ok(())
    }

    fn predict(&self, predictions: &Array2<F>) -> Result<Array1<F>> {
        if !self.is_trained {
            return Err(TimeSeriesError::InvalidModel(
                "Meta-model not trained".to_string(),
            ));
        }

        let (n_samples, n_models) = predictions.dim();

        if n_models != self.coefficients.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.coefficients.len(),
                actual: n_models,
            });
        }

        let mut result = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut prediction = self.intercept;
            for j in 0..n_models {
                prediction = prediction + self.coefficients[j] * predictions[[i, j]];
            }
            result[i] = prediction;
        }

        Ok(result)
    }

    fn confidence(&self) -> F {
        self.confidence
    }
}

impl<F: Float + Debug + Clone + FromPrimitive + std::iter::Sum + 'static> EnsembleForecaster<F> {
    /// Create new ensemble forecaster
    pub fn new(
        base_models: Vec<BaseModel>,
        strategy: EnsembleStrategy,
        max_buffer_size: usize,
        validation_window: usize,
    ) -> Result<Self> {
        let num_models = base_models.len();
        if num_models == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "At least one base model required".to_string(),
            ));
        }

        let base_modelwrappers: Vec<BaseModelWrapper<F>> = base_models
            .into_iter()
            .map(|model_type| BaseModelWrapper {
                model_type,
                model_state: ModelState::default(),
                recent_predictions: VecDeque::with_capacity(validation_window),
                performance: ModelPerformance::default(),
                is_trained: false,
            })
            .collect();

        let ensemble_weights =
            Array1::from_elem(num_models, F::one() / F::from(num_models).unwrap());

        // Create meta-model if using stacking
        let meta_model = match &strategy {
            EnsembleStrategy::Stacking { meta_learner } => {
                Some(Self::create_meta_model(meta_learner.clone())?)
            }
            _ => None,
        };

        Ok(Self {
            base_models: base_modelwrappers,
            strategy,
            performance_history: VecDeque::with_capacity(validation_window),
            training_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            validation_window,
            ensemble_weights,
            meta_model,
        })
    }

    /// Create meta-model from specification
    fn create_meta_model(_metalearner: MetaLearner) -> Result<Box<dyn MetaModel<F>>> {
        match _metalearner {
            MetaLearner::LinearRegression => Ok(Box::new(LinearRegressionMeta::new())),
            MetaLearner::RidgeRegression { alpha: _alpha } => {
                // For simplicity, use linear regression with regularization
                Ok(Box::new(LinearRegressionMeta::new()))
            }
            _ => Err(TimeSeriesError::NotImplemented(
                "Meta-_learner not yet implemented".to_string(),
            )),
        }
    }

    /// Add training observation
    pub fn add_observation(&mut self, value: F) -> Result<()> {
        if self.training_buffer.len() >= self.max_buffer_size {
            self.training_buffer.pop_front();
        }
        self.training_buffer.push_back(value);

        // Train models if we have enough data
        if self.training_buffer.len() >= 20 {
            self.retrain_models()?;
        }

        Ok(())
    }

    /// Retrain all base models
    fn retrain_models(&mut self) -> Result<()> {
        let data: Vec<F> = self.training_buffer.iter().cloned().collect();

        // Process each model individually to avoid borrowing conflicts
        for i in 0..self.base_models.len() {
            // Extract the model temporarily to avoid borrow checker issues
            let mut model = self.base_models[i].clone();

            match &model.model_type {
                BaseModel::ARIMA { p, d, q } => {
                    // Simplified ARIMA training - just store parameters
                    model
                        .model_state
                        .parameters
                        .insert("p".to_string(), F::from(*p).unwrap());
                    model
                        .model_state
                        .parameters
                        .insert("d".to_string(), F::from(*d).unwrap());
                    model
                        .model_state
                        .parameters
                        .insert("q".to_string(), F::from(*q).unwrap());
                }
                BaseModel::ExponentialSmoothing { alpha, beta, gamma } => {
                    // Simplified exponential smoothing training
                    model
                        .model_state
                        .parameters
                        .insert("alpha".to_string(), F::from(*alpha).unwrap());
                    if let Some(beta_val) = beta {
                        model
                            .model_state
                            .parameters
                            .insert("beta".to_string(), F::from(*beta_val).unwrap());
                    }
                    if let Some(gamma_val) = gamma {
                        model
                            .model_state
                            .parameters
                            .insert("gamma".to_string(), F::from(*gamma_val).unwrap());
                    }
                }
                BaseModel::LinearTrend => {
                    // Simple linear trend estimation
                    if data.len() >= 2 {
                        let n = F::from(data.len()).unwrap();
                        let sum_x = F::from(data.len() * (data.len() - 1) / 2).unwrap();
                        let sum_y = data.iter().cloned().sum::<F>();
                        let sum_xy = data
                            .iter()
                            .enumerate()
                            .map(|(i, &y)| F::from(i).unwrap() * y)
                            .sum::<F>();
                        let sum_x2 = (0..data.len()).map(|i| F::from(i * i).unwrap()).sum::<F>();

                        let denominator = n * sum_x2 - sum_x * sum_x;
                        if denominator != F::zero() {
                            let slope = (n * sum_xy - sum_x * sum_y) / denominator;
                            let intercept = (sum_y - slope * sum_x) / n;
                            model
                                .model_state
                                .parameters
                                .insert("slope".to_string(), slope);
                            model
                                .model_state
                                .parameters
                                .insert("intercept".to_string(), intercept);
                        } else {
                            model
                                .model_state
                                .parameters
                                .insert("slope".to_string(), F::zero());
                            model
                                .model_state
                                .parameters
                                .insert("intercept".to_string(), sum_y / n);
                        }
                    }
                }
                BaseModel::SeasonalNaive { period } => {
                    // Store seasonal period
                    model
                        .model_state
                        .parameters
                        .insert("period".to_string(), F::from(*period).unwrap());
                }
                BaseModel::MovingAverage { window } => {
                    // Store window size
                    model
                        .model_state
                        .parameters
                        .insert("window".to_string(), F::from(*window).unwrap());
                }
                BaseModel::LSTM {
                    hidden_units,
                    num_layers,
                } => {
                    // Store network parameters
                    model
                        .model_state
                        .parameters
                        .insert("hidden_units".to_string(), F::from(*hidden_units).unwrap());
                    model
                        .model_state
                        .parameters
                        .insert("num_layers".to_string(), F::from(*num_layers).unwrap());
                }
                BaseModel::Prophet {
                    seasonality_prior_scale,
                    changepoint_prior_scale,
                } => {
                    // Store prophet parameters
                    model.model_state.parameters.insert(
                        "seasonality_prior_scale".to_string(),
                        F::from(*seasonality_prior_scale).unwrap(),
                    );
                    model.model_state.parameters.insert(
                        "changepoint_prior_scale".to_string(),
                        F::from(*changepoint_prior_scale).unwrap(),
                    );
                }
            }

            model.is_trained = true;

            // Put the model back
            self.base_models[i] = model;
        }

        // Update ensemble weights
        self.update_ensemble_weights()?;

        Ok(())
    }

    /// Train a single model
    #[allow(dead_code)]
    fn train_single_model(&self, modelwrapper: &mut BaseModelWrapper<F>, data: &[F]) -> Result<()> {
        match &modelwrapper.model_type {
            BaseModel::ARIMA { p, d, q } => {
                self.train_arima_model(modelwrapper, data, *p, *d, *q)?;
            }
            BaseModel::ExponentialSmoothing { alpha, beta, gamma } => {
                self.train_exponential_smoothing(modelwrapper, data, *alpha, *beta, *gamma)?;
            }
            BaseModel::LinearTrend => {
                self.train_linear_trend(modelwrapper, data)?;
            }
            BaseModel::SeasonalNaive { period } => {
                self.train_seasonal_naive(modelwrapper, data, *period)?;
            }
            BaseModel::MovingAverage { window } => {
                self.train_moving_average(modelwrapper, data, *window)?;
            }
            BaseModel::LSTM {
                hidden_units: hidden,
                num_layers: layers,
            } => {
                self.train_lstm_model(modelwrapper, data)?;
            }
            BaseModel::Prophet {
                seasonality_prior_scale: s,
                changepoint_prior_scale: c,
            } => {
                self.train_prophet_model(modelwrapper, data)?;
            }
        }

        modelwrapper.is_trained = true;
        Ok(())
    }

    /// Train ARIMA model (simplified)
    #[allow(dead_code)]
    fn train_arima_model(
        &self,
        modelwrapper: &mut BaseModelWrapper<F>,
        data: &[F],
        p: usize,
        d: usize,
        q: usize,
    ) -> Result<()> {
        // Simplified ARIMA training
        let mut processed_data = data.to_vec();

        // Apply differencing
        for _ in 0..d {
            let mut diff_data = Vec::new();
            for i in 1..processed_data.len() {
                diff_data.push(processed_data[i] - processed_data[i - 1]);
            }
            processed_data = diff_data;
        }

        if processed_data.len() < p.max(q) + 5 {
            return Ok(());
        }

        // Estimate AR parameters using Yule-Walker
        let mut ar_coeffs = vec![F::zero(); p];
        if p > 0 {
            let autocorrs = self.compute_autocorrelations(&processed_data, p)?;
            for i in 0..p {
                if i + 1 < autocorrs.len() && autocorrs[0] > F::zero() {
                    ar_coeffs[i] = autocorrs[i + 1] / autocorrs[0];
                    // Keep stable
                    ar_coeffs[i] = ar_coeffs[i]
                        .max(F::from(-0.99).unwrap())
                        .min(F::from(0.99).unwrap());
                }
            }
        }

        // Estimate MA parameters (simplified)
        let ma_coeffs = vec![F::from(0.1).unwrap(); q];

        // Store parameters
        for (i, &coeff) in ar_coeffs.iter().enumerate() {
            modelwrapper
                .model_state
                .parameters
                .insert(format!("ar_{i}"), coeff);
        }
        for (i, &coeff) in ma_coeffs.iter().enumerate() {
            modelwrapper
                .model_state
                .parameters
                .insert(format!("ma_{i}"), coeff);
        }

        modelwrapper.model_state.state_variables.insert(
            "recent_data".to_string(),
            Array1::from_vec(
                processed_data
                    .iter()
                    .rev()
                    .take(p.max(q))
                    .rev()
                    .cloned()
                    .collect(),
            ),
        );

        Ok(())
    }

    /// Compute autocorrelations
    #[allow(dead_code)]
    fn compute_autocorrelations(&self, data: &[F], maxlag: usize) -> Result<Vec<F>> {
        let n = data.len();
        let mut autocorrs = vec![F::zero(); maxlag + 1];

        for _lag in 0..=maxlag {
            if _lag >= n {
                break;
            }

            let mut sum = F::zero();
            let count = n - _lag;

            for i in _lag..n {
                sum = sum + data[i] * data[i - _lag];
            }

            autocorrs[_lag] = sum / F::from(count).unwrap();
        }

        Ok(autocorrs)
    }

    /// Train exponential smoothing model
    #[allow(dead_code)]
    fn train_exponential_smoothing(
        &self,
        modelwrapper: &mut BaseModelWrapper<F>,
        data: &[F],
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let alpha_f = F::from(alpha).unwrap();
        modelwrapper
            .model_state
            .parameters
            .insert("alpha".to_string(), alpha_f);

        if let Some(beta_val) = beta {
            modelwrapper
                .model_state
                .parameters
                .insert("beta".to_string(), F::from(beta_val).unwrap());
        }

        if let Some(gamma_val) = gamma {
            modelwrapper
                .model_state
                .parameters
                .insert("gamma".to_string(), F::from(gamma_val).unwrap());
        }

        // Initialize level and trend
        let level = data[0];
        let trend = if data.len() > 1 {
            data[1] - data[0]
        } else {
            F::zero()
        };

        modelwrapper
            .model_state
            .parameters
            .insert("level".to_string(), level);
        modelwrapper
            .model_state
            .parameters
            .insert("trend".to_string(), trend);

        Ok(())
    }

    /// Train linear trend model
    #[allow(dead_code)]
    fn train_linear_trend(&self, modelwrapper: &mut BaseModelWrapper<F>, data: &[F]) -> Result<()> {
        if data.len() < 2 {
            return Ok(());
        }

        let n = F::from(data.len()).unwrap();
        let x_mean = (n - F::one()) / F::from(2).unwrap();
        let y_mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / n;

        let mut numerator = F::zero();
        let mut denominator = F::zero();

        for (i, &y) in data.iter().enumerate() {
            let x = F::from(i).unwrap();
            let x_diff = x - x_mean;
            numerator = numerator + x_diff * (y - y_mean);
            denominator = denominator + x_diff * x_diff;
        }

        let slope = if denominator > F::zero() {
            numerator / denominator
        } else {
            F::zero()
        };

        let intercept = y_mean - slope * x_mean;

        modelwrapper
            .model_state
            .parameters
            .insert("slope".to_string(), slope);
        modelwrapper
            .model_state
            .parameters
            .insert("intercept".to_string(), intercept);

        Ok(())
    }

    /// Train seasonal naive model
    #[allow(dead_code)]
    fn train_seasonal_naive(
        &self,
        modelwrapper: &mut BaseModelWrapper<F>,
        data: &[F],
        period: usize,
    ) -> Result<()> {
        if data.len() < period {
            return Ok(());
        }

        modelwrapper
            .model_state
            .parameters
            .insert("period".to_string(), F::from(period).unwrap());

        // Store last seasonal cycle
        let last_cycle: Array1<F> =
            Array1::from_vec(data.iter().rev().take(period).rev().cloned().collect());
        modelwrapper
            .model_state
            .state_variables
            .insert("last_cycle".to_string(), last_cycle);

        Ok(())
    }

    /// Train moving average model
    #[allow(dead_code)]
    fn train_moving_average(
        &self,
        modelwrapper: &mut BaseModelWrapper<F>,
        data: &[F],
        window: usize,
    ) -> Result<()> {
        if data.len() < window {
            return Ok(());
        }

        modelwrapper
            .model_state
            .parameters
            .insert("window".to_string(), F::from(window).unwrap());

        // Store recent data for prediction
        let recent_data: Array1<F> =
            Array1::from_vec(data.iter().rev().take(window).rev().cloned().collect());
        modelwrapper
            .model_state
            .state_variables
            .insert("recent_data".to_string(), recent_data);

        Ok(())
    }

    /// Train LSTM model (placeholder)
    #[allow(dead_code)]
    fn train_lstm_model(&self, modelwrapper: &mut BaseModelWrapper<F>, data: &[F]) -> Result<()> {
        // Placeholder for LSTM training
        // In practice, would use neural network framework
        modelwrapper
            .model_state
            .parameters
            .insert("hidden_state".to_string(), F::zero());
        modelwrapper
            .model_state
            .parameters
            .insert("cell_state".to_string(), F::zero());

        // Store recent data
        let recent_data: Array1<F> =
            Array1::from_vec(data.iter().rev().take(10).rev().cloned().collect());
        modelwrapper
            .model_state
            .state_variables
            .insert("recent_data".to_string(), recent_data);

        Ok(())
    }

    /// Train Prophet model (placeholder)
    #[allow(dead_code)]
    fn train_prophet_model(
        &self,
        modelwrapper: &mut BaseModelWrapper<F>,
        data: &[F],
    ) -> Result<()> {
        // Placeholder for Prophet-like model
        // Would implement trend and seasonality detection
        if data.len() < 2 {
            return Ok(());
        }

        // Simple trend estimation
        let trend = (data[data.len() - 1] - data[0]) / F::from(data.len() - 1).unwrap();
        modelwrapper
            .model_state
            .parameters
            .insert("trend".to_string(), trend);

        let mean = data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();
        modelwrapper
            .model_state
            .parameters
            .insert("base_level".to_string(), mean);

        Ok(())
    }

    /// Generate forecast using ensemble
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if self.base_models.iter().all(|m| !m.is_trained) {
            return Err(TimeSeriesError::InvalidModel(
                "No models are trained".to_string(),
            ));
        }

        // Get predictions from all base models
        let mut modelpredictions = Array2::zeros((steps, self.base_models.len()));

        for (model_idx, modelwrapper) in self.base_models.iter().enumerate() {
            if modelwrapper.is_trained {
                let predictions = self.predict_with_model(modelwrapper, steps)?;
                for step in 0..steps {
                    modelpredictions[[step, model_idx]] = predictions[step];
                }
            }
        }

        // Combine predictions using ensemble strategy
        self.combine_predictions(&modelpredictions)
    }

    /// Predict with a single model
    fn predict_with_model(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
    ) -> Result<Array1<F>> {
        match &modelwrapper.model_type {
            BaseModel::ARIMA { p, d, q } => self.predict_arima(modelwrapper, steps, *p, *q),
            BaseModel::ExponentialSmoothing { alpha, beta, gamma } => {
                self.predict_exponential_smoothing(modelwrapper, steps, beta.is_some())
            }
            BaseModel::LinearTrend => self.predict_linear_trend(modelwrapper, steps),
            BaseModel::SeasonalNaive { period: _period } => {
                self.predict_seasonal_naive(modelwrapper, steps)
            }
            BaseModel::MovingAverage { window: _window } => {
                self.predict_moving_average(modelwrapper, steps)
            }
            BaseModel::LSTM {
                hidden_units: hidden,
                num_layers: layers,
            } => self.predict_lstm(modelwrapper, steps),
            BaseModel::Prophet {
                seasonality_prior_scale: s,
                changepoint_prior_scale: c,
            } => self.predict_prophet(modelwrapper, steps),
        }
    }

    /// Predict with ARIMA model
    fn predict_arima(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
        p: usize,
        q: usize,
    ) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        if let Some(recent_data) = modelwrapper.model_state.state_variables.get("recent_data") {
            let mut extended_data = recent_data.to_vec();

            for step in 0..steps {
                let mut prediction = F::zero();

                // AR component
                for i in 0..p {
                    if let Some(&ar_coeff) =
                        modelwrapper.model_state.parameters.get(&format!("ar_{i}"))
                    {
                        if i < extended_data.len() {
                            let lag_index = extended_data.len() - 1 - i;
                            prediction = prediction + ar_coeff * extended_data[lag_index];
                        }
                    }
                }

                // MA component (simplified - assume zero residuals for future)
                for i in 0..q {
                    if let Some(&ma_coeff) =
                        modelwrapper.model_state.parameters.get(&format!("ma_{i}"))
                    {
                        // Simplified: assume residuals decay to zero
                        let residual_weight = F::from(0.95_f64.powi(i as i32 + 1)).unwrap();
                        prediction = prediction + ma_coeff * residual_weight;
                    }
                }

                forecasts[step] = prediction;
                extended_data.push(prediction);

                // Keep buffer reasonable size
                if extended_data.len() > 50 {
                    extended_data.remove(0);
                }
            }
        }

        Ok(forecasts)
    }

    /// Predict with exponential smoothing
    fn predict_exponential_smoothing(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
        has_trend: bool,
    ) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        if let (Some(&level), trend_opt) = (
            modelwrapper.model_state.parameters.get("level"),
            modelwrapper.model_state.parameters.get("_trend"),
        ) {
            let _trend = trend_opt.cloned().unwrap_or(F::zero());

            for step in 0..steps {
                let h = F::from(step + 1).unwrap();
                forecasts[step] = if has_trend { level + _trend * h } else { level };
            }
        }

        Ok(forecasts)
    }

    /// Predict with linear trend
    fn predict_linear_trend(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
    ) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        if let (Some(&slope), Some(&intercept)) = (
            modelwrapper.model_state.parameters.get("slope"),
            modelwrapper.model_state.parameters.get("intercept"),
        ) {
            let data_length = F::from(self.training_buffer.len()).unwrap();

            for step in 0..steps {
                let x = data_length + F::from(step).unwrap();
                forecasts[step] = slope * x + intercept;
            }
        }

        Ok(forecasts)
    }

    /// Predict with seasonal naive
    fn predict_seasonal_naive(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
    ) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        if let (Some(&period_f), Some(last_cycle)) = (
            modelwrapper.model_state.parameters.get("period"),
            modelwrapper.model_state.state_variables.get("last_cycle"),
        ) {
            let period = period_f.to_usize().unwrap_or(1);

            for step in 0..steps {
                let cycle_index = step % period;
                if cycle_index < last_cycle.len() {
                    forecasts[step] = last_cycle[cycle_index];
                }
            }
        }

        Ok(forecasts)
    }

    /// Predict with moving average
    fn predict_moving_average(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
    ) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        if let Some(recent_data) = modelwrapper.model_state.state_variables.get("recent_data") {
            let avg = recent_data.sum() / F::from(recent_data.len()).unwrap();

            for step in 0..steps {
                forecasts[step] = avg;
            }
        }

        Ok(forecasts)
    }

    /// Predict with LSTM (placeholder)
    fn predict_lstm(&self, modelwrapper: &BaseModelWrapper<F>, steps: usize) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        // Placeholder prediction - in practice would use neural network
        if let Some(recent_data) = modelwrapper.model_state.state_variables.get("recent_data") {
            let last_value = recent_data[recent_data.len() - 1];
            let trend = if recent_data.len() > 1 {
                (recent_data[recent_data.len() - 1] - recent_data[recent_data.len() - 2])
                    * F::from(0.5).unwrap()
            } else {
                F::zero()
            };

            for step in 0..steps {
                let h = F::from(step + 1).unwrap();
                forecasts[step] = last_value + trend * h;
            }
        }

        Ok(forecasts)
    }

    /// Predict with Prophet (placeholder)
    fn predict_prophet(
        &self,
        modelwrapper: &BaseModelWrapper<F>,
        steps: usize,
    ) -> Result<Array1<F>> {
        let mut forecasts = Array1::zeros(steps);

        if let (Some(&trend), Some(&base_level)) = (
            modelwrapper.model_state.parameters.get("trend"),
            modelwrapper.model_state.parameters.get("base_level"),
        ) {
            let data_length = F::from(self.training_buffer.len()).unwrap();

            for step in 0..steps {
                let t = data_length + F::from(step + 1).unwrap();

                // Simple trend + seasonal component
                let trend_component = base_level + trend * t;
                let seasonal_component = F::from(0.1).unwrap()
                    * (F::from(2.0 * std::f64::consts::PI).unwrap() * t / F::from(12).unwrap())
                        .sin();

                forecasts[step] = trend_component + seasonal_component;
            }
        }

        Ok(forecasts)
    }

    /// Combine predictions using ensemble strategy
    fn combine_predictions(&self, modelpredictions: &Array2<F>) -> Result<Array1<F>> {
        let (steps, n_models) = modelpredictions.dim();
        let mut ensemble_forecast = Array1::zeros(steps);

        match &self.strategy {
            EnsembleStrategy::SimpleAverage => {
                for step in 0..steps {
                    let mut sum = F::zero();
                    for model_idx in 0..n_models {
                        sum = sum + modelpredictions[[step, model_idx]];
                    }
                    ensemble_forecast[step] = sum / F::from(n_models).unwrap();
                }
            }
            EnsembleStrategy::WeightedAverage
            | EnsembleStrategy::DynamicWeighting { window_size: _ } => {
                for step in 0..steps {
                    let mut weighted_sum = F::zero();
                    let mut total_weight = F::zero();

                    for model_idx in 0..n_models {
                        let weight = if model_idx < self.ensemble_weights.len() {
                            self.ensemble_weights[model_idx]
                        } else {
                            F::one() / F::from(n_models).unwrap()
                        };

                        weighted_sum = weighted_sum + weight * modelpredictions[[step, model_idx]];
                        total_weight = total_weight + weight;
                    }

                    ensemble_forecast[step] = if total_weight > F::zero() {
                        weighted_sum / total_weight
                    } else {
                        weighted_sum / F::from(n_models).unwrap()
                    };
                }
            }
            EnsembleStrategy::Median => {
                for step in 0..steps {
                    let mut step_predictions: Vec<F> = (0..n_models)
                        .map(|model_idx| modelpredictions[[step, model_idx]])
                        .collect();

                    step_predictions
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let median_idx = step_predictions.len() / 2;
                    ensemble_forecast[step] = if step_predictions.len() % 2 == 0 {
                        (step_predictions[median_idx - 1] + step_predictions[median_idx])
                            / F::from(2).unwrap()
                    } else {
                        step_predictions[median_idx]
                    };
                }
            }
            EnsembleStrategy::Stacking { meta_learner: meta } => {
                if let Some(ref meta_model) = self.meta_model {
                    ensemble_forecast = meta_model.predict(modelpredictions)?;
                } else {
                    return Err(TimeSeriesError::InvalidModel(
                        "Meta-model not initialized for stacking".to_string(),
                    ));
                }
            }
            EnsembleStrategy::BayesianModelAveraging => {
                // Simplified BMA using performance-based weights
                for step in 0..steps {
                    let mut weighted_sum = F::zero();
                    let mut total_weight = F::zero();

                    for model_idx in 0..n_models {
                        let weight = if model_idx < self.base_models.len() {
                            F::one() / (F::one() + self.base_models[model_idx].performance.mse)
                        } else {
                            F::one()
                        };

                        weighted_sum = weighted_sum + weight * modelpredictions[[step, model_idx]];
                        total_weight = total_weight + weight;
                    }

                    ensemble_forecast[step] = weighted_sum / total_weight;
                }
            }
            EnsembleStrategy::BestModel => {
                // Use _predictions from the best performing model
                let best_model_idx = self.find_best_model_index();
                for step in 0..steps {
                    ensemble_forecast[step] = modelpredictions[[step, best_model_idx]];
                }
            }
        }

        Ok(ensemble_forecast)
    }

    /// Find the index of the best performing model
    fn find_best_model_index(&self) -> usize {
        let mut best_idx = 0;
        let mut best_performance = F::infinity();

        for (idx, model) in self.base_models.iter().enumerate() {
            if model.performance.mse < best_performance {
                best_performance = model.performance.mse;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Update ensemble weights based on recent performance
    fn update_ensemble_weights(&mut self) -> Result<()> {
        let n_models = self.base_models.len();

        match &self.strategy {
            EnsembleStrategy::WeightedAverage
            | EnsembleStrategy::DynamicWeighting { window_size: _ } => {
                // Compute weights based on inverse of MSE
                let mut new_weights = Array1::zeros(n_models);
                let mut total_inverse_mse = F::zero();

                for (idx, model) in self.base_models.iter().enumerate() {
                    let inverse_mse = F::one() / (F::one() + model.performance.mse);
                    new_weights[idx] = inverse_mse;
                    total_inverse_mse = total_inverse_mse + inverse_mse;
                }

                // Normalize weights
                if total_inverse_mse > F::zero() {
                    for weight in new_weights.iter_mut() {
                        *weight = *weight / total_inverse_mse;
                    }
                } else {
                    // Equal weights fallback
                    let equal_weight = F::one() / F::from(n_models).unwrap();
                    new_weights.fill(equal_weight);
                }

                self.ensemble_weights = new_weights;
            }
            _ => {
                // Equal weights for other strategies
                let equal_weight = F::one() / F::from(n_models).unwrap();
                self.ensemble_weights = Array1::from_elem(n_models, equal_weight);
            }
        }

        Ok(())
    }

    /// Get ensemble model information
    pub fn get_model_info(&self) -> Vec<(BaseModel, ModelPerformance<F>, bool)> {
        self.base_models
            .iter()
            .map(|model| {
                (
                    model.model_type.clone(),
                    model.performance.clone(),
                    model.is_trained,
                )
            })
            .collect()
    }

    /// Get current ensemble weights
    pub fn get_ensemble_weights(&self) -> &Array1<F> {
        &self.ensemble_weights
    }
}

/// AutoML for time series forecasting
#[derive(Debug)]
pub struct AutoMLForecaster<F: Float + Debug + Clone + FromPrimitive + std::iter::Sum + 'static> {
    /// Candidate models to evaluate
    candidate_models: Vec<BaseModel>,
    /// Best ensemble found
    best_ensemble: Option<EnsembleForecaster<F>>,
    /// Hyperparameter search space
    search_space: HyperparameterSpace,
    /// Cross-validation configuration
    cv_config: CrossValidationConfig,
    /// Evaluation metrics
    evaluation_metrics: Vec<EvaluationMetric>,
    /// Search algorithm
    search_algorithm: SearchAlgorithm,
}

/// Hyperparameter search space
#[derive(Debug, Clone)]
pub struct HyperparameterSpace {
    /// ARIMA order ranges
    pub arima_p_range: (usize, usize),
    /// ARIMA differencing order range
    pub arima_d_range: (usize, usize),
    /// ARIMA moving average order range
    pub arima_q_range: (usize, usize),
    /// Exponential smoothing parameters
    pub alpha_range: (f64, f64),
    /// Trend smoothing parameter range
    pub beta_range: Option<(f64, f64)>,
    /// Seasonal smoothing parameter range
    pub gamma_range: Option<(f64, f64)>,
    /// Moving average window sizes
    pub ma_windows: Vec<usize>,
    /// Seasonal periods to consider
    pub seasonal_periods: Vec<usize>,
    /// Neural network architectures
    pub lstm_hidden_units: Vec<usize>,
    /// LSTM number of layers options
    pub lstm_num_layers: Vec<usize>,
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            arima_p_range: (0, 5),
            arima_d_range: (0, 2),
            arima_q_range: (0, 5),
            alpha_range: (0.1, 0.9),
            beta_range: Some((0.1, 0.9)),
            gamma_range: Some((0.1, 0.9)),
            ma_windows: vec![3, 5, 10, 20],
            seasonal_periods: vec![7, 12, 24, 52],
            lstm_hidden_units: vec![32, 64, 128],
            lstm_num_layers: vec![1, 2, 3],
        }
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Training set ratio
    pub train_ratio: f64,
    /// Forecast horizon for evaluation
    pub forecast_horizon: usize,
    /// Use time series specific CV (rolling window)
    pub use_time_series_cv: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            train_ratio: 0.8,
            forecast_horizon: 10,
            use_time_series_cv: true,
        }
    }
}

/// Evaluation metrics for model selection
#[derive(Debug, Clone)]
pub enum EvaluationMetric {
    /// Mean Absolute Error
    MAE,
    /// Mean Squared Error
    MSE,
    /// Root Mean Squared Error
    RMSE,
    /// Mean Absolute Percentage Error
    MAPE,
    /// Symmetric Mean Absolute Percentage Error
    SMAPE,
    /// Mean Absolute Scaled Error
    MASE,
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
}

/// Search algorithms for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum SearchAlgorithm {
    /// Grid search over all combinations
    GridSearch,
    /// Random search with specified iterations
    RandomSearch {
        /// Number of iterations
        n_iterations: usize,
    },
    /// Bayesian optimization
    BayesianOptimization {
        /// Number of iterations
        n_iterations: usize,
    },
    /// Genetic algorithm
    GeneticAlgorithm {
        /// Population size
        population_size: usize,
        /// Number of generations
        generations: usize,
    },
}

impl<F: Float + Debug + Clone + FromPrimitive + std::iter::Sum + 'static> AutoMLForecaster<F> {
    /// Create new AutoML forecaster
    pub fn new(
        search_space: HyperparameterSpace,
        cv_config: CrossValidationConfig,
        evaluation_metrics: Vec<EvaluationMetric>,
        search_algorithm: SearchAlgorithm,
    ) -> Self {
        Self {
            candidate_models: Vec::new(),
            best_ensemble: None,
            search_space,
            cv_config,
            evaluation_metrics,
            search_algorithm,
        }
    }

    /// Fit AutoML on training data
    pub fn fit(&mut self, data: &[F]) -> Result<()> {
        // Generate candidate models
        self.generate_candidate_models()?;

        // Evaluate models using cross-validation
        let model_scores = self.evaluate_models(data)?;

        // Select best models and create ensemble
        let best_models = self.select_best_models(&model_scores)?;

        // Create ensemble with best models
        let ensemble =
            EnsembleForecaster::new(best_models, EnsembleStrategy::WeightedAverage, 1000, 50)?;

        self.best_ensemble = Some(ensemble);

        // Train the final ensemble
        if let Some(ref mut ensemble) = self.best_ensemble {
            for &value in data {
                ensemble.add_observation(value)?;
            }
        }

        Ok(())
    }

    /// Generate candidate models based on search space
    fn generate_candidate_models(&mut self) -> Result<()> {
        self.candidate_models.clear();

        match &self.search_algorithm {
            SearchAlgorithm::GridSearch => {
                self.generate_grid_search_models()?;
            }
            SearchAlgorithm::RandomSearch { n_iterations } => {
                self.generate_random_search_models(*n_iterations)?;
            }
            SearchAlgorithm::BayesianOptimization {
                n_iterations: n_iter,
            } => {
                // Simplified - use random search for now
                self.generate_random_search_models(50)?;
            }
            SearchAlgorithm::GeneticAlgorithm {
                population_size,
                generations: gens,
            } => {
                // Simplified - use random search for now
                self.generate_random_search_models(*population_size)?;
            }
        }

        Ok(())
    }

    /// Generate models for grid search
    fn generate_grid_search_models(&mut self) -> Result<()> {
        // ARIMA models
        for p in self.search_space.arima_p_range.0..=self.search_space.arima_p_range.1 {
            for d in self.search_space.arima_d_range.0..=self.search_space.arima_d_range.1 {
                for q in self.search_space.arima_q_range.0..=self.search_space.arima_q_range.1 {
                    if p + d + q > 0 && p + d + q <= 6 {
                        // Limit complexity
                        self.candidate_models.push(BaseModel::ARIMA { p, d, q });
                    }
                }
            }
        }

        // Exponential smoothing models
        let alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9];
        for &alpha in &alpha_values {
            self.candidate_models.push(BaseModel::ExponentialSmoothing {
                alpha,
                beta: None,
                gamma: None,
            });

            if let Some((beta_min, beta_max)) = self.search_space.beta_range {
                let beta_values = [beta_min, (beta_min + beta_max) / 2.0, beta_max];
                for &beta in &beta_values {
                    self.candidate_models.push(BaseModel::ExponentialSmoothing {
                        alpha,
                        beta: Some(beta),
                        gamma: None,
                    });
                }
            }
        }

        // Simple models
        self.candidate_models.push(BaseModel::LinearTrend);

        for &window in &self.search_space.ma_windows {
            self.candidate_models
                .push(BaseModel::MovingAverage { window });
        }

        for &period in &self.search_space.seasonal_periods {
            self.candidate_models
                .push(BaseModel::SeasonalNaive { period });
        }

        Ok(())
    }

    /// Generate models for random search
    fn generate_random_search_models(&mut self, niterations: usize) -> Result<()> {
        // Simplified random generation - in practice would use proper random sampling
        let mut seed: u32 = 42;

        for _i in 0..niterations {
            // Simple LCG for pseudo-random numbers
            seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let rand_val = (seed as f64) / (i32::MAX as f64);

            if rand_val < 0.4 {
                // Generate ARIMA model
                let p = (rand_val
                    * (self.search_space.arima_p_range.1 - self.search_space.arima_p_range.0)
                        as f64) as usize
                    + self.search_space.arima_p_range.0;
                let d = ((rand_val * 2.0) as usize).min(self.search_space.arima_d_range.1);
                let q = ((rand_val * 3.0) as usize).min(self.search_space.arima_q_range.1);

                if p + d + q > 0 && p + d + q <= 6 {
                    self.candidate_models.push(BaseModel::ARIMA { p, d, q });
                }
            } else if rand_val < 0.7 {
                // Generate exponential smoothing
                let alpha = self.search_space.alpha_range.0
                    + rand_val
                        * (self.search_space.alpha_range.1 - self.search_space.alpha_range.0);
                self.candidate_models.push(BaseModel::ExponentialSmoothing {
                    alpha,
                    beta: None,
                    gamma: None,
                });
            } else if rand_val < 0.85 {
                // Generate moving average
                let window_idx = (rand_val * self.search_space.ma_windows.len() as f64) as usize;
                if window_idx < self.search_space.ma_windows.len() {
                    let window = self.search_space.ma_windows[window_idx];
                    self.candidate_models
                        .push(BaseModel::MovingAverage { window });
                }
            } else {
                // Generate seasonal naive
                let period_idx =
                    (rand_val * self.search_space.seasonal_periods.len() as f64) as usize;
                if period_idx < self.search_space.seasonal_periods.len() {
                    let period = self.search_space.seasonal_periods[period_idx];
                    self.candidate_models
                        .push(BaseModel::SeasonalNaive { period });
                }
            }
        }

        // Always include simple baseline models
        self.candidate_models.push(BaseModel::LinearTrend);
        self.candidate_models
            .push(BaseModel::MovingAverage { window: 5 });

        Ok(())
    }

    /// Evaluate models using cross-validation
    fn evaluate_models(&self, data: &[F]) -> Result<Vec<(BaseModel, F)>> {
        let mut model_scores = Vec::new();

        for model in &self.candidate_models {
            let score = self.cross_validate_model(model, data)?;
            model_scores.push((model.clone(), score));
        }

        // Sort by score (lower is better for most metrics)
        model_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(model_scores)
    }

    /// Cross-validate a single model
    fn cross_validate_model(&self, model: &BaseModel, data: &[F]) -> Result<F> {
        if data.len() < self.cv_config.forecast_horizon + 20 {
            return Ok(F::infinity()); // Not enough data
        }

        let mut scores = Vec::new();
        let fold_size = data.len() / self.cv_config.n_folds;

        for fold in 0..self.cv_config.n_folds {
            let test_start = data.len() - (fold + 1) * fold_size;
            let test_end = data.len() - fold * fold_size;

            if test_start < self.cv_config.forecast_horizon {
                continue;
            }

            let train_end = test_start;
            let train_data = &data[0..train_end];
            let test_data = &data[test_start..test_end];

            // Train model on fold training data
            let mut ensemble = EnsembleForecaster::new(
                vec![model.clone()],
                EnsembleStrategy::SimpleAverage,
                1000,
                20,
            )?;

            for &value in train_data {
                ensemble.add_observation(value)?;
            }

            // Forecast and evaluate
            let forecast_horizon = self.cv_config.forecast_horizon.min(test_data.len());
            if forecast_horizon == 0 {
                continue;
            }

            let forecasts = ensemble.forecast(forecast_horizon)?;
            let test_subset = &test_data[0..forecast_horizon];

            let score = self.compute_evaluation_metric(&forecasts, test_subset)?;
            scores.push(score);
        }

        if scores.is_empty() {
            Ok(F::infinity())
        } else {
            let avg_score =
                scores.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(scores.len()).unwrap();
            Ok(avg_score)
        }
    }

    /// Compute evaluation metric
    fn compute_evaluation_metric(&self, forecasts: &Array1<F>, actuals: &[F]) -> Result<F> {
        if forecasts.len() != actuals.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: forecasts.len(),
                actual: actuals.len(),
            });
        }

        // Use first metric for model selection (could be extended)
        let metric = self
            .evaluation_metrics
            .first()
            .unwrap_or(&EvaluationMetric::MSE);

        let n = F::from(actuals.len()).unwrap();
        let mut score = F::zero();

        match metric {
            EvaluationMetric::MAE => {
                for (i, &actual) in actuals.iter().enumerate() {
                    score = score + (forecasts[i] - actual).abs();
                }
                score = score / n;
            }
            EvaluationMetric::MSE => {
                for (i, &actual) in actuals.iter().enumerate() {
                    let error = forecasts[i] - actual;
                    score = score + error * error;
                }
                score = score / n;
            }
            EvaluationMetric::RMSE => {
                for (i, &actual) in actuals.iter().enumerate() {
                    let error = forecasts[i] - actual;
                    score = score + error * error;
                }
                score = (score / n).sqrt();
            }
            EvaluationMetric::MAPE => {
                for (i, &actual) in actuals.iter().enumerate() {
                    if actual != F::zero() {
                        score = score + ((forecasts[i] - actual) / actual).abs();
                    }
                }
                score = score * F::from(100).unwrap() / n;
            }
            EvaluationMetric::SMAPE => {
                for (i, &actual) in actuals.iter().enumerate() {
                    let denominator = (forecasts[i].abs() + actual.abs()) / F::from(2).unwrap();
                    if denominator > F::zero() {
                        score = score + (forecasts[i] - actual).abs() / denominator;
                    }
                }
                score = score * F::from(100).unwrap() / n;
            }
            _ => {
                // Default to MSE for other metrics
                for (i, &actual) in actuals.iter().enumerate() {
                    let error = forecasts[i] - actual;
                    score = score + error * error;
                }
                score = score / n;
            }
        }

        Ok(score)
    }

    /// Select best models for ensemble
    fn select_best_models(&self, modelscores: &[(BaseModel, F)]) -> Result<Vec<BaseModel>> {
        let mut selected_models = Vec::new();
        let max_models = 5; // Maximum models in ensemble

        // Select top performing models
        for (model, _score) in modelscores.iter().take(max_models) {
            selected_models.push(model.clone());
        }

        // Ensure we have at least one model
        if selected_models.is_empty() && !modelscores.is_empty() {
            selected_models.push(modelscores[0].0.clone());
        }

        Ok(selected_models)
    }

    /// Generate forecast using the best ensemble
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if let Some(ref ensemble) = self.best_ensemble {
            ensemble.forecast(steps)
        } else {
            Err(TimeSeriesError::InvalidModel(
                "AutoML not fitted yet".to_string(),
            ))
        }
    }

    /// Get information about the best models
    pub fn get_best_models(&self) -> Option<Vec<(BaseModel, ModelPerformance<F>, bool)>> {
        self.best_ensemble
            .as_ref()
            .map(|ensemble| ensemble.get_model_info())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_forecaster_creation() {
        let models = vec![
            BaseModel::ARIMA { p: 1, d: 1, q: 1 },
            BaseModel::LinearTrend,
            BaseModel::MovingAverage { window: 5 },
        ];

        let ensemble =
            EnsembleForecaster::<f64>::new(models, EnsembleStrategy::SimpleAverage, 1000, 50);

        assert!(ensemble.is_ok());
    }

    #[test]
    fn test_linear_regression_meta_model() {
        let mut meta_model = LinearRegressionMeta::<f64>::new();

        let predictions =
            Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64 * 0.1).collect()).unwrap();
        let targets = Array1::from_vec((0..10).map(|i| i as f64).collect());

        let result = meta_model.train(&predictions, &targets);
        assert!(result.is_ok());

        let test_predictions =
            Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64 * 0.2).collect()).unwrap();
        let forecast = meta_model.predict(&test_predictions);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 5);
    }

    #[test]
    fn test_automl_forecaster() {
        let search_space = HyperparameterSpace::default();
        let cv_config = CrossValidationConfig::default();
        let metrics = vec![EvaluationMetric::MSE];
        let search_algo = SearchAlgorithm::RandomSearch { n_iterations: 10 };

        let mut automl =
            AutoMLForecaster::<f64>::new(search_space, cv_config, metrics, search_algo);

        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = automl.fit(&data);

        // Should succeed even with limited data
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensemble_strategies() {
        let models = vec![
            BaseModel::LinearTrend,
            BaseModel::MovingAverage { window: 3 },
        ];

        // Test different strategies
        let strategies = vec![
            EnsembleStrategy::SimpleAverage,
            EnsembleStrategy::WeightedAverage,
            EnsembleStrategy::Median,
            EnsembleStrategy::BestModel,
        ];

        for strategy in strategies {
            let ensemble = EnsembleForecaster::<f64>::new(models.clone(), strategy, 1000, 20);
            assert!(ensemble.is_ok());
        }
    }
}
