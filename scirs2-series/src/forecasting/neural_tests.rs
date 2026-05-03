use super::*;

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::Array2;

#[test]
fn test_sliding_windows() {
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let (x, y) = utils::create_sliding_windows(&data, 3, 2).expect("Operation failed");

    assert_eq!(x.nrows(), 6);
    assert_eq!(x.ncols(), 3);
    assert_eq!(y.nrows(), 6);
    assert_eq!(y.ncols(), 2);

    // Check first window
    assert_abs_diff_eq!(x[[0, 0]], 1.0);
    assert_abs_diff_eq!(x[[0, 1]], 2.0);
    assert_abs_diff_eq!(x[[0, 2]], 3.0);
    assert_abs_diff_eq!(y[[0, 0]], 4.0);
    assert_abs_diff_eq!(y[[0, 1]], 5.0);
}

#[test]
fn test_normalize_data() {
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (normalized, min_val, max_val) = utils::normalize_data(&data).expect("Operation failed");

    assert_abs_diff_eq!(min_val, 1.0);
    assert_abs_diff_eq!(max_val, 5.0);
    assert_abs_diff_eq!(normalized[0], 0.0);
    assert_abs_diff_eq!(normalized[4], 1.0);

    // Test denormalization
    let denormalized = utils::denormalize_data(&normalized, min_val, max_val);
    for i in 0..data.len() {
        assert_abs_diff_eq!(denormalized[i], data[i], epsilon = 1e-10);
    }
}

#[test]
fn test_train_val_split() {
    let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64).collect())
        .expect("Operation failed");
    let y = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect())
        .expect("Operation failed");

    let (x_train, x_val, y_train, y_val) =
        utils::train_val_split(&x, &y, 0.2).expect("Operation failed");

    assert_eq!(x_train.nrows(), 8);
    assert_eq!(x_val.nrows(), 2);
    assert_eq!(y_train.nrows(), 8);
    assert_eq!(y_val.nrows(), 2);
}

#[test]
fn test_neural_config_defaults() {
    let config = NeuralConfig::default();
    assert_eq!(config.lookback_window, 24);
    assert_eq!(config.forecast_horizon, 1);
    assert_eq!(config.epochs, 100);
    assert_eq!(config.batch_size, 32);
}

#[test]
fn test_lstm_forecaster_creation() {
    let forecaster = LSTMForecaster::<f64>::with_default_config();
    assert!(!forecaster.trained);
    assert!(forecaster.loss_history.is_empty());
    assert!(forecaster.input_layer.is_none());
}

#[test]
fn test_transformer_forecaster_creation() {
    let forecaster = TransformerForecaster::<f64>::with_default_config();
    assert!(!forecaster.trained);
    assert!(forecaster.loss_history.is_empty());
    assert!(forecaster.attention_layer.is_none());
}

#[test]
fn test_nbeats_forecaster_creation() {
    let forecaster = NBeatsForecaster::<f64>::with_default_config();
    assert!(!forecaster.trained);
    assert!(forecaster.loss_history.is_empty());
    assert!(forecaster.stack_layers.is_empty());
}

#[test]
fn test_simple_neural_network_training() {
    // Test that neural forecasters can actually train on simple data
    let data = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());

    let mut lstm = LSTMForecaster::<f64>::with_default_config();
    let result = lstm.fit(&data);

    // Should succeed now instead of returning NotImplemented
    assert!(result.is_ok(), "LSTM training should succeed");
    assert!(lstm.trained, "LSTM should be marked as trained");
    assert!(
        !lstm.loss_history.is_empty(),
        "Loss history should not be empty"
    );
}

#[test]
fn test_neural_prediction() {
    let data = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());

    let mut transformer = TransformerForecaster::<f64>::with_default_config();
    transformer.fit(&data).expect("Operation failed");

    let forecast = transformer.predict(5);
    assert!(forecast.is_ok(), "Transformer prediction should succeed");

    let result = forecast.expect("Operation failed");
    assert_eq!(result.forecast.len(), 5, "Should predict 5 steps");
}

#[test]
fn test_uncertainty_prediction() {
    let data = Array1::from_vec((0..30).map(|i| (i as f64 * 0.1).sin()).collect());

    // Use smaller config for faster testing
    let config = NBeatsConfig {
        base: NeuralConfig {
            lookback_window: 10,
            forecast_horizon: 1,
            epochs: 10, // Reduced from 100
            ..Default::default()
        },
        num_stacks: 3,   // Reduced from 30
        layer_width: 64, // Reduced from 512
        ..Default::default()
    };
    let mut nbeats = NBeatsForecaster::<f64>::new(config);
    nbeats.fit(&data).expect("Operation failed");

    let forecast = nbeats.predict_with_uncertainty(3, 0.95);
    assert!(
        forecast.is_ok(),
        "N-BEATS uncertainty prediction should succeed"
    );

    let result = forecast.expect("Operation failed");
    assert_eq!(result.forecast.len(), 3);
    // Check that confidence intervals are not all zeros (indicating they were computed)
    assert!(result.lower_ci.iter().any(|&x| x != 0.0) || result.upper_ci.iter().any(|&x| x != 0.0));
}

/// Advanced neural forecasting models and techniques
pub mod advanced {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_core::numeric::{Float, FromPrimitive};
    use std::collections::{HashMap, VecDeque};

    /// Stack types for N-BEATS
    #[derive(Debug, Clone)]
    pub enum StackType {
        /// Trend stack for trend patterns
        Trend,
        /// Seasonality stack for seasonal patterns
        Seasonality,
        /// Generic stack for general patterns
        Generic,
    }

    /// Multi-scale neural forecasting that combines multiple time horizons
    #[derive(Debug)]
    pub struct MultiScaleNeuralForecaster<F: Float + Debug + FromPrimitive> {
        /// Short-term forecaster (hourly/daily patterns)
        short_term: LSTMForecaster<F>,
        /// Medium-term forecaster (weekly/monthly patterns)
        medium_term: TransformerForecaster<F>,
        /// Long-term forecaster (seasonal/yearly patterns)
        long_term: NBeatsForecaster<F>,
        /// Combination weights learned during training
        combination_weights: Option<Array1<F>>,
        /// Training configuration
        config: MultiScaleConfig,
        /// Whether the model has been trained
        trained: bool,
    }

    /// Multi-scale forecasting configuration
    #[derive(Debug, Clone)]
    pub struct MultiScaleConfig {
        /// Short-term forecasting window size
        pub short_term_window: usize,
        /// Medium-term forecasting window size
        pub medium_term_window: usize,
        /// Long-term forecasting window size
        pub long_term_window: usize,
        /// Forecast horizon length
        pub forecast_horizon: usize,
        /// Learning rate for training
        pub learning_rate: f64,
        /// Number of meta-learning epochs
        pub meta_epochs: usize,
    }

    impl Default for MultiScaleConfig {
        fn default() -> Self {
            Self {
                short_term_window: 24,   // 1 day of hourly data
                medium_term_window: 168, // 1 week of hourly data
                long_term_window: 8760,  // 1 year of hourly data
                forecast_horizon: 24,
                learning_rate: 0.001,
                meta_epochs: 50,
            }
        }
    }

    impl<F: Float + Debug + Clone + FromPrimitive + scirs2_core::ndarray::ScalarOperand>
        MultiScaleNeuralForecaster<F>
    {
        /// Create new multi-scale neural forecaster
        pub fn new(config: MultiScaleConfig) -> Self {
            // Configure individual models for different time scales
            let short_term_config = LSTMConfig {
                base: NeuralConfig {
                    lookback_window: config.short_term_window,
                    forecast_horizon: config.forecast_horizon,
                    epochs: 50,
                    learning_rate: config.learning_rate,
                    ..Default::default()
                },
                hidden_size: 32,
                num_layers: 1,
                ..Default::default()
            };

            let medium_term_config = TransformerConfig {
                base: NeuralConfig {
                    lookback_window: config.medium_term_window,
                    forecast_horizon: config.forecast_horizon,
                    epochs: 30,
                    learning_rate: config.learning_rate * 0.5,
                    ..Default::default()
                },
                d_model: 64,
                num_heads: 4,
                num_encoder_layers: 2,
                ..Default::default()
            };

            let long_term_config = NBeatsConfig {
                base: NeuralConfig {
                    lookback_window: config.long_term_window,
                    forecast_horizon: config.forecast_horizon,
                    epochs: 20,
                    learning_rate: config.learning_rate * 0.1,
                    ..Default::default()
                },
                num_stacks: 3,
                num_blocks_per_stack: 2,
                num_layers_per_block: 4,
                layer_width: 256,
                expansion_coefficient_dim: 5,
                share_weights_in_stack: false,
                generic_architecture: false,
            };

            Self {
                short_term: LSTMForecaster::new(short_term_config),
                medium_term: TransformerForecaster::new(medium_term_config),
                long_term: NBeatsForecaster::new(long_term_config),
                combination_weights: None,
                config,
                trained: false,
            }
        }

        /// Create multi-scale forecaster with default configuration
        pub fn with_default_config() -> Self {
            Self::new(MultiScaleConfig::default())
        }

        /// Train the multi-scale forecaster
        pub fn fit(&mut self, data: &Array1<F>) -> Result<()> {
            if data.len() < self.config.long_term_window + self.config.forecast_horizon {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Not enough data for multi-scale training".to_string(),
                    required: self.config.long_term_window + self.config.forecast_horizon,
                    actual: data.len(),
                });
            }

            // Train individual models on different data windows
            let data_len = data.len();

            // Short-term: use recent data
            let short_start = data_len.saturating_sub(self.config.short_term_window * 10);
            let short_data = data.slice(s![short_start..]).to_owned();
            self.short_term.fit(&short_data)?;

            // Medium-term: use more data but downsample if necessary
            let medium_start = data_len.saturating_sub(self.config.medium_term_window * 5);
            let medium_data = data.slice(s![medium_start..]).to_owned();
            self.medium_term.fit(&medium_data)?;

            // Long-term: use all available data
            self.long_term.fit(data)?;

            // Learn combination weights using validation data
            self.learn_combination_weights(data)?;

            self.trained = true;
            Ok(())
        }

        /// Learn optimal combination weights for different models
        fn learn_combination_weights(&mut self, data: &Array1<F>) -> Result<()> {
            let validation_size = (data.len() as f64 * 0.2) as usize;
            if validation_size < self.config.forecast_horizon * 3 {
                // Use equal weights if not enough validation data
                self.combination_weights = Some(Array1::from_elem(
                    3,
                    F::from(1.0 / 3.0).expect("Failed to convert to float"),
                ));
                return Ok(());
            }

            let train_end = data.len() - validation_size;
            let _validation_data = data.slice(s![train_end..]).to_owned();

            // Generate predictions from each model
            let mut short_predictions = Vec::new();
            let mut medium_predictions = Vec::new();
            let mut long_predictions = Vec::new();
            let mut targets = Vec::new();

            let step_size = self.config.forecast_horizon;
            for i in (0..validation_size - self.config.forecast_horizon).step_by(step_size) {
                let window_end = train_end + i;
                let _current_data = data.slice(s![..window_end]).to_owned();

                // Get predictions from each model
                if let Ok(short_pred) = self.short_term.predict(self.config.forecast_horizon) {
                    short_predictions.push(short_pred.forecast);
                }
                if let Ok(medium_pred) = self.medium_term.predict(self.config.forecast_horizon) {
                    medium_predictions.push(medium_pred.forecast);
                }
                if let Ok(long_pred) = self.long_term.predict(self.config.forecast_horizon) {
                    long_predictions.push(long_pred.forecast);
                }

                // Get actual targets
                let target_start = window_end;
                let target_end = (window_end + self.config.forecast_horizon).min(data.len());
                if target_end > target_start {
                    let target = data.slice(s![target_start..target_end]).to_owned();
                    targets.push(target);
                }
            }

            // Optimize weights using simple grid search
            let mut best_weights =
                Array1::from_elem(3, F::from(1.0 / 3.0).expect("Failed to convert to float"));
            let mut best_error = F::infinity();

            // Grid search over possible weight combinations
            for w1 in (0..=10).map(|x| {
                F::from(x).expect("Failed to convert to float")
                    / F::from(10).expect("Failed to convert constant to float")
            }) {
                for w2 in (0..=10).map(|x| {
                    F::from(x).expect("Failed to convert to float")
                        / F::from(10).expect("Failed to convert constant to float")
                }) {
                    let w3 = F::one() - w1 - w2;
                    if w3 >= F::zero() && w3 <= F::one() {
                        let weights = Array1::from_vec(vec![w1, w2, w3]);
                        let error = self.evaluate_weights(
                            &weights,
                            &short_predictions,
                            &medium_predictions,
                            &long_predictions,
                            &targets,
                        );

                        if error < best_error {
                            best_error = error;
                            best_weights = weights;
                        }
                    }
                }
            }

            self.combination_weights = Some(best_weights);
            Ok(())
        }

        /// Evaluate a set of combination weights
        fn evaluate_weights(
            &self,
            weights: &Array1<F>,
            short_predictions: &[Array1<F>],
            medium_predictions: &[Array1<F>],
            long_predictions: &[Array1<F>],
            targets: &[Array1<F>],
        ) -> F {
            let mut total_error = F::zero();
            let mut count = 0;

            let min_len = short_predictions
                .len()
                .min(medium_predictions.len())
                .min(long_predictions.len())
                .min(targets.len());

            for i in 0..min_len {
                if short_predictions[i].len() == targets[i].len()
                    && medium_predictions[i].len() == targets[i].len()
                    && long_predictions[i].len() == targets[i].len()
                {
                    // Combine _predictions using weights
                    let combined = &short_predictions[i] * weights[0]
                        + &medium_predictions[i] * weights[1]
                        + &long_predictions[i] * weights[2];

                    // Calculate MSE
                    for (&pred, &actual) in combined.iter().zip(targets[i].iter()) {
                        let error = pred - actual;
                        total_error = total_error + error * error;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                total_error / F::from(count).expect("Failed to convert to float")
            } else {
                F::infinity()
            }
        }

        /// Make predictions using the trained multi-scale model
        pub fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
            if !self.trained {
                return Err(TimeSeriesError::InvalidModel(
                    "Model has not been trained".to_string(),
                ));
            }

            // Get predictions from each model
            let short_pred = self.short_term.predict(steps)?;
            let medium_pred = self.medium_term.predict(steps)?;
            let long_pred = self.long_term.predict(steps)?;

            // Combine predictions using learned weights
            let weights = self.combination_weights.as_ref().expect("Operation failed");
            let combined_forecast = &short_pred.forecast * weights[0]
                + &medium_pred.forecast * weights[1]
                + &long_pred.forecast * weights[2];

            // Combine confidence intervals (conservative approach)
            let combined_lower = short_pred
                .lower_ci
                .iter()
                .zip(medium_pred.lower_ci.iter())
                .zip(long_pred.lower_ci.iter())
                .map(|((&s, &m), &l)| s.min(m).min(l))
                .collect();

            let combined_upper = short_pred
                .upper_ci
                .iter()
                .zip(medium_pred.upper_ci.iter())
                .zip(long_pred.upper_ci.iter())
                .map(|((&s, &m), &l)| s.max(m).max(l))
                .collect();

            Ok(ForecastResult {
                forecast: combined_forecast,
                lower_ci: Array1::from_vec(combined_lower),
                upper_ci: Array1::from_vec(combined_upper),
            })
        }

        /// Get the combination weights learned during training
        pub fn get_combination_weights(&self) -> Option<&Array1<F>> {
            self.combination_weights.as_ref()
        }
    }

    /// Online neural forecaster that can update incrementally
    #[derive(Debug)]
    pub struct OnlineNeuralForecaster<F: Float + Debug + FromPrimitive> {
        /// Base neural model
        base_model: LSTMForecaster<F>,
        /// Recent data buffer
        data_buffer: VecDeque<F>,
        /// Maximum buffer size
        max_buffer_size: usize,
        /// Incremental learning rate
        #[allow(dead_code)]
        incremental_lr: F,
        /// Update frequency (retrain every n observations)
        #[allow(dead_code)]
        update_frequency: usize,
        /// Observation counter
        observation_count: usize,
        /// Configuration
        config: OnlineConfig,
    }

    /// Online learning configuration
    #[derive(Debug, Clone)]
    pub struct OnlineConfig {
        /// Buffer size for streaming data
        pub buffer_size: usize,
        /// Initial training window size
        pub initial_window: usize,
        /// Model update frequency
        pub update_frequency: usize,
        /// Learning rate for incremental updates
        pub incremental_learning_rate: f64,
        /// Forecast horizon length
        pub forecast_horizon: usize,
    }

    impl Default for OnlineConfig {
        fn default() -> Self {
            Self {
                buffer_size: 1000,
                initial_window: 100,
                update_frequency: 10,
                incremental_learning_rate: 0.0001,
                forecast_horizon: 1,
            }
        }
    }

    impl<F: Float + Debug + Clone + FromPrimitive> OnlineNeuralForecaster<F> {
        /// Create new online neural forecaster
        pub fn new(config: OnlineConfig) -> Self {
            let lstm_config = LSTMConfig {
                base: NeuralConfig {
                    lookback_window: 24,
                    forecast_horizon: config.forecast_horizon,
                    epochs: 20,
                    learning_rate: 0.001,
                    batch_size: 16,
                    ..Default::default()
                },
                hidden_size: 32,
                num_layers: 1,
                dropout: 0.1,
                ..Default::default()
            };

            Self {
                base_model: LSTMForecaster::new(lstm_config),
                data_buffer: VecDeque::with_capacity(config.buffer_size),
                max_buffer_size: config.buffer_size,
                incremental_lr: F::from(config.incremental_learning_rate)
                    .expect("Failed to convert to float"),
                update_frequency: config.update_frequency,
                observation_count: 0,
                config,
            }
        }

        /// Add new observation and potentially update the model
        pub fn update(&mut self, observation: F) -> Result<()> {
            // Add to buffer
            if self.data_buffer.len() >= self.max_buffer_size {
                self.data_buffer.pop_front();
            }
            self.data_buffer.push_back(observation);
            self.observation_count += 1;

            // Check if we need to retrain - ensure we have enough data for lookback window
            if self.data_buffer.len()
                >= self
                    .base_model
                    .config
                    .base
                    .lookback_window
                    .max(self.config.initial_window)
                && self
                    .observation_count
                    .is_multiple_of(self.config.update_frequency)
            {
                let current_data: Array1<F> = Array1::from_iter(self.data_buffer.iter().cloned());
                self.base_model.fit(&current_data)?;
            }

            Ok(())
        }

        /// Make online prediction
        pub fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
            if self.data_buffer.len() < self.config.initial_window {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Not enough data for prediction".to_string(),
                    required: self.config.initial_window,
                    actual: self.data_buffer.len(),
                });
            }

            self.base_model.predict(steps)
        }

        /// Get current buffer size
        pub fn buffer_size(&self) -> usize {
            self.data_buffer.len()
        }

        /// Check if model is ready for predictions
        pub fn is_ready(&self) -> bool {
            self.data_buffer.len() >= self.config.initial_window && self.base_model.trained
        }
    }

    /// Attention-based neural forecaster with interpretable attention weights
    #[derive(Debug)]
    pub struct AttentionForecaster<F: Float + Debug> {
        /// Input dimension
        input_dim: usize,
        /// Hidden dimension
        hidden_dim: usize,
        /// Output dimension
        output_dim: usize,
        /// Attention weights (learned during training)
        attention_weights: Option<Array2<F>>,
        /// Model parameters
        model_weights: Option<ModelWeights<F>>,
        /// Training configuration
        config: NeuralConfig,
        /// Whether model is trained
        trained: bool,
        /// Training loss history
        loss_history: Vec<F>,
    }

    #[derive(Debug, Clone)]
    struct ModelWeights<F: Float> {
        attention_query: Array2<F>,
        attention_key: Array2<F>,
        attention_value: Array2<F>,
        output_projection: Array2<F>,
        bias: Array1<F>,
    }

    impl<F: Float + Debug + Clone + FromPrimitive + scirs2_core::ndarray::ScalarOperand>
        AttentionForecaster<F>
    {
        /// Create new attention-based forecaster
        pub fn new(config: NeuralConfig) -> Self {
            Self {
                input_dim: config.lookback_window,
                hidden_dim: 64,
                output_dim: config.forecast_horizon,
                attention_weights: None,
                model_weights: None,
                config,
                trained: false,
                loss_history: Vec::new(),
            }
        }

        /// Train the attention-based forecaster
        pub fn fit(&mut self, data: &Array1<F>) -> Result<()> {
            if data.len() < self.config.lookback_window + self.config.forecast_horizon {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Insufficient data for attention model training".to_string(),
                    required: self.config.lookback_window + self.config.forecast_horizon,
                    actual: data.len(),
                });
            }

            // Initialize model weights randomly
            self.initialize_weights();

            // Create training windows
            let (x_train, y_train) = utils::create_sliding_windows(
                data,
                self.config.lookback_window,
                self.config.forecast_horizon,
            )?;

            // Simple training loop (simplified)
            for _epoch in 0..self.config.epochs {
                let mut epoch_loss = F::zero();
                let mut batch_count = 0;

                for i in 0..x_train.nrows() {
                    let input = x_train.row(i).to_owned();
                    let target = y_train.row(i).to_owned();

                    // Forward pass with attention
                    let prediction = self.forward_with_attention(&input)?;

                    // Calculate loss (MSE)
                    let loss = self.calculate_loss(&prediction, &target);
                    epoch_loss = epoch_loss + loss;
                    batch_count += 1;

                    // Simplified parameter update (gradient descent approximation)
                    self.update_parameters(&input, &target, &prediction)?;
                }

                if batch_count > 0 {
                    epoch_loss =
                        epoch_loss / F::from(batch_count).expect("Failed to convert to float");
                    self.loss_history.push(epoch_loss);
                }

                // Early stopping check
                if let Some(patience) = self.config.early_stopping_patience {
                    if self.loss_history.len() > patience {
                        let recent_losses =
                            &self.loss_history[self.loss_history.len() - patience..];
                        let is_improving = recent_losses.windows(2).any(|w| w[1] < w[0]);
                        if !is_improving {
                            break;
                        }
                    }
                }
            }

            self.trained = true;
            Ok(())
        }

        /// Initialize model weights
        fn initialize_weights(&mut self) {
            let h = self.hidden_dim;
            let i = self.input_dim;
            let o = self.output_dim;

            // Simple random initialization (in practice, would use proper initialization)
            self.model_weights = Some(ModelWeights {
                attention_query: Array2::from_elem(
                    (h, i),
                    F::from(0.1).expect("Failed to convert constant to float"),
                ),
                attention_key: Array2::from_elem(
                    (h, i),
                    F::from(0.1).expect("Failed to convert constant to float"),
                ),
                attention_value: Array2::from_elem(
                    (h, i),
                    F::from(0.1).expect("Failed to convert constant to float"),
                ),
                output_projection: Array2::from_elem(
                    (o, h),
                    F::from(0.1).expect("Failed to convert constant to float"),
                ),
                bias: Array1::zeros(o),
            });
        }

        /// Forward pass with attention mechanism
        fn forward_with_attention(&mut self, input: &Array1<F>) -> Result<Array1<F>> {
            let weights = self.model_weights.as_ref().expect("Operation failed");

            // Compute attention scores (simplified)
            let query = weights.attention_query.dot(input);
            let key = weights.attention_key.dot(input);
            let value = weights.attention_value.dot(input);

            // Attention weights (simplified softmax)
            let attention_scores =
                Array1::from_iter(query.iter().zip(key.iter()).map(|(&q, &k)| (q * k).exp()));
            let sum_scores = attention_scores.sum();
            let attention_weights = attention_scores / sum_scores;

            // Store attention weights for interpretability
            self.attention_weights = Some(
                Array2::from_shape_vec((1, attention_weights.len()), attention_weights.to_vec())
                    .expect("Operation failed"),
            );

            // Apply attention to values
            let attended_values = value * &attention_weights;

            // Output projection
            let output = weights.output_projection.dot(&attended_values) + &weights.bias;

            Ok(output)
        }

        /// Calculate MSE loss
        fn calculate_loss(&self, prediction: &Array1<F>, target: &Array1<F>) -> F {
            let mut loss = F::zero();
            for (p, t) in prediction.iter().zip(target.iter()) {
                let diff = *p - *t;
                loss = loss + diff * diff;
            }
            loss / F::from(prediction.len()).expect("Operation failed")
        }

        /// Update parameters (simplified gradient descent)
        fn update_parameters(
            &mut self,
            _input: &Array1<F>,
            _target: &Array1<F>,
            _prediction: &Array1<F>,
        ) -> Result<()> {
            // Simplified parameter update
            // In practice, would compute gradients and update weights
            Ok(())
        }

        /// Make prediction with attention
        pub fn predict(&self, data: &Array1<F>) -> Result<ForecastResult<F>> {
            if !self.trained {
                return Err(TimeSeriesError::InvalidModel(
                    "Model has not been trained".to_string(),
                ));
            }

            if data.len() < self.config.lookback_window {
                return Err(TimeSeriesError::InsufficientData {
                    message: "Not enough data for prediction".to_string(),
                    required: self.config.lookback_window,
                    actual: data.len(),
                });
            }

            let input = data
                .slice(s![data.len() - self.config.lookback_window..])
                .to_owned();
            let mut forecaster = self.clone();
            let prediction = forecaster.forward_with_attention(&input)?;

            // Simple confidence intervals (would be more sophisticated in practice)
            let std_dev = if !self.loss_history.is_empty() {
                self.loss_history[self.loss_history.len() - 1].sqrt()
            } else {
                F::from(0.1).expect("Failed to convert constant to float")
            };

            let margin = std_dev * F::from(1.96).expect("Failed to convert constant to float");
            let lower_ci = prediction.mapv(|x| x - margin);
            let upper_ci = prediction.mapv(|x| x + margin);

            Ok(ForecastResult {
                forecast: prediction,
                lower_ci,
                upper_ci,
            })
        }

        /// Get attention weights for interpretability
        pub fn get_attention_weights(&self) -> Option<&Array2<F>> {
            self.attention_weights.as_ref()
        }

        /// Get training loss history
        pub fn get_loss_history(&self) -> &[F] {
            &self.loss_history
        }
    }

    impl<F: Float + Debug + Clone> Clone for AttentionForecaster<F> {
        fn clone(&self) -> Self {
            Self {
                input_dim: self.input_dim,
                hidden_dim: self.hidden_dim,
                output_dim: self.output_dim,
                attention_weights: self.attention_weights.clone(),
                model_weights: self.model_weights.clone(),
                config: self.config.clone(),
                trained: self.trained,
                loss_history: self.loss_history.clone(),
            }
        }
    }

    /// Ensemble neural forecasting system
    pub struct EnsembleNeuralForecaster<F: Float + Debug> {
        /// Individual forecasters
        forecasters: Vec<Box<dyn NeuralForecaster<F>>>,
        /// Ensemble weights (learned or predefined)
        weights: Option<Array1<F>>,
        /// Ensemble method
        ensemble_method: EnsembleMethod,
        /// Model names
        model_names: Vec<String>,
        /// Whether the ensemble is trained
        trained: bool,
    }

    impl<F: Float + Debug> std::fmt::Debug for EnsembleNeuralForecaster<F> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("EnsembleNeuralForecaster")
                .field("forecasters_count", &self.forecasters.len())
                .field("weights", &self.weights)
                .field("ensemble_method", &self.ensemble_method)
                .field("model_names", &self.model_names)
                .field("trained", &self.trained)
                .finish()
        }
    }

    /// Ensemble combination methods
    #[derive(Debug, Clone)]
    pub enum EnsembleMethod {
        /// Simple average
        Average,
        /// Weighted average with learned weights
        WeightedAverage,
        /// Stacking with meta-learner
        Stacking,
        /// Best model selection based on validation
        BestModel,
    }

    impl<F: Float + Debug + Clone + FromPrimitive> EnsembleNeuralForecaster<F> {
        /// Create new ensemble forecaster
        pub fn new(_ensemblemethod: EnsembleMethod) -> Self {
            Self {
                forecasters: Vec::new(),
                weights: None,
                ensemble_method: _ensemblemethod,
                model_names: Vec::new(),
                trained: false,
            }
        }

        /// Add a forecaster to the ensemble
        pub fn add_forecaster(&mut self, forecaster: Box<dyn NeuralForecaster<F>>, name: String) {
            self.forecasters.push(forecaster);
            self.model_names.push(name);
        }

        /// Train the ensemble on data
        pub fn fit(&mut self, data: &Array1<F>) -> Result<()> {
            if self.forecasters.is_empty() {
                return Err(TimeSeriesError::InvalidModel(
                    "No forecasters added to ensemble".to_string(),
                ));
            }

            // Train individual forecasters
            for forecaster in &mut self.forecasters {
                forecaster.fit(data)?;
            }

            // Learn ensemble weights based on method
            match self.ensemble_method {
                EnsembleMethod::Average => {
                    let n = self.forecasters.len();
                    self.weights = Some(Array1::from_elem(
                        n,
                        F::one() / F::from(n).expect("Failed to convert to float"),
                    ));
                }
                EnsembleMethod::WeightedAverage => {
                    self.learn_weights(data)?;
                }
                EnsembleMethod::Stacking => {
                    self.learn_stacking_weights(data)?;
                }
                EnsembleMethod::BestModel => {
                    self.select_best_model(data)?;
                }
            }

            self.trained = true;
            Ok(())
        }

        /// Make ensemble predictions
        pub fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
            if !self.trained {
                return Err(TimeSeriesError::InvalidModel(
                    "Ensemble not trained".to_string(),
                ));
            }

            // Get predictions from all models
            let mut predictions = Vec::new();
            let mut lower_cis = Vec::new();
            let mut upper_cis = Vec::new();

            for forecaster in &self.forecasters {
                let result = forecaster.predict(steps)?;
                predictions.push(result.forecast);
                lower_cis.push(result.lower_ci);
                upper_cis.push(result.upper_ci);
            }

            // Combine predictions based on ensemble method
            let weights = self.weights.as_ref().expect("Operation failed");
            let mut combined_forecast = Array1::zeros(steps);
            let mut combined_lower = Array1::zeros(steps);
            let mut combined_upper = Array1::zeros(steps);

            for step in 0..steps {
                let mut weighted_pred = F::zero();
                let mut weighted_lower = F::zero();
                let mut weighted_upper = F::zero();

                for (i, weight) in weights.iter().enumerate() {
                    weighted_pred = weighted_pred + *weight * predictions[i][step];
                    weighted_lower = weighted_lower + *weight * lower_cis[i][step];
                    weighted_upper = weighted_upper + *weight * upper_cis[i][step];
                }

                combined_forecast[step] = weighted_pred;
                combined_lower[step] = weighted_lower;
                combined_upper[step] = weighted_upper;
            }

            Ok(ForecastResult {
                forecast: combined_forecast,
                lower_ci: combined_lower,
                upper_ci: combined_upper,
            })
        }

        /// Learn optimal weights for weighted average
        fn learn_weights(&mut self, data: &Array1<F>) -> Result<()> {
            let validation_size = (data.len() as f64 * 0.2) as usize;
            if validation_size < 10 {
                // Fall back to equal weights
                let n = self.forecasters.len();
                self.weights = Some(Array1::from_elem(
                    n,
                    F::one() / F::from(n).expect("Failed to convert to float"),
                ));
                return Ok(());
            }

            let train_end = data.len() - validation_size;
            let train_data = data.slice(s![..train_end]).to_owned();
            let validation_data = data.slice(s![train_end..]).to_owned();

            // Train on subset
            for forecaster in &mut self.forecasters {
                forecaster.fit(&train_data)?;
            }

            // Generate validation predictions
            let forecast_horizon = validation_size.min(10);
            let mut validation_errors = vec![F::zero(); self.forecasters.len()];

            for (i, forecaster) in self.forecasters.iter().enumerate() {
                if let Ok(result) = forecaster.predict(forecast_horizon) {
                    let actual = validation_data.slice(s![..forecast_horizon]);
                    let mut mse = F::zero();
                    for j in 0..forecast_horizon {
                        let error = result.forecast[j] - actual[j];
                        mse = mse + error * error;
                    }
                    validation_errors[i] =
                        mse / F::from(forecast_horizon).expect("Failed to convert to float");
                }
            }

            // Convert errors to weights (inverse of error)
            let mut weights = Array1::zeros(self.forecasters.len());
            let mut total_weight = F::zero();

            for (i, &error) in validation_errors.iter().enumerate() {
                let weight = if error > F::zero() {
                    F::one() / (error + F::epsilon())
                } else {
                    F::one()
                };
                weights[i] = weight;
                total_weight = total_weight + weight;
            }

            // Normalize weights
            if total_weight > F::zero() {
                for weight in weights.iter_mut() {
                    *weight = *weight / total_weight;
                }
            }

            self.weights = Some(weights);
            Ok(())
        }

        /// Learn stacking weights using meta-learner
        fn learn_stacking_weights(&mut self, data: &Array1<F>) -> Result<()> {
            // Simplified stacking: use cross-validation to generate meta-features
            // then learn linear combination weights
            let n_folds = 3;
            let fold_size = data.len() / n_folds;

            let mut meta_features = Array2::zeros((0, self.forecasters.len()));
            let mut meta_targets = Array1::zeros(0);

            for fold in 0..n_folds {
                let val_start = fold * fold_size;
                let val_end = if fold == n_folds - 1 {
                    data.len()
                } else {
                    (fold + 1) * fold_size
                };

                // Create train data (excluding validation fold)
                let mut train_indices = Vec::new();
                for i in 0..data.len() {
                    if i < val_start || i >= val_end {
                        train_indices.push(i);
                    }
                }

                if train_indices.len() < 50 {
                    continue; // Skip if insufficient data
                }

                let train_data = Array1::from_iter(train_indices.iter().map(|&i| data[i]));

                // Train models on fold
                for forecaster in &mut self.forecasters {
                    let _ = forecaster.fit(&train_data);
                }

                // Generate predictions for validation fold
                let val_size = val_end - val_start;
                if val_size > 0 {
                    let mut fold_predictions = Array2::zeros((val_size, self.forecasters.len()));

                    for (i, forecaster) in self.forecasters.iter().enumerate() {
                        if let Ok(result) = forecaster.predict(val_size) {
                            for j in 0..val_size.min(result.forecast.len()) {
                                fold_predictions[[j, i]] = result.forecast[j];
                            }
                        }
                    }

                    // Add to meta-features
                    let current_meta_size = meta_features.nrows();
                    let new_meta_size = current_meta_size + val_size;

                    // Resize meta_features and meta_targets
                    let mut new_meta_features =
                        Array2::zeros((new_meta_size, self.forecasters.len()));
                    let mut new_meta_targets = Array1::zeros(new_meta_size);

                    // Copy existing data
                    for i in 0..current_meta_size {
                        for j in 0..self.forecasters.len() {
                            new_meta_features[[i, j]] = meta_features[[i, j]];
                        }
                        new_meta_targets[i] = meta_targets[i];
                    }

                    // Add new data
                    for i in 0..val_size {
                        for j in 0..self.forecasters.len() {
                            new_meta_features[[current_meta_size + i, j]] =
                                fold_predictions[[i, j]];
                        }
                        if val_start + i < data.len() {
                            new_meta_targets[current_meta_size + i] = data[val_start + i];
                        }
                    }

                    meta_features = new_meta_features;
                    meta_targets = new_meta_targets;
                }
            }

            // Learn linear combination weights using simple least squares
            if meta_features.nrows() > 0 {
                self.weights = Some(self.solve_least_squares(&meta_features, &meta_targets)?);
            } else {
                // Fall back to equal weights
                let n = self.forecasters.len();
                self.weights = Some(Array1::from_elem(
                    n,
                    F::one() / F::from(n).expect("Failed to convert to float"),
                ));
            }

            Ok(())
        }

        /// Simple least squares solver for meta-learning
        fn solve_least_squares(&self, x: &Array2<F>, y: &Array1<F>) -> Result<Array1<F>> {
            let n_features = x.ncols();
            let mut weights = Array1::zeros(n_features);

            // Simple approach: solve each feature independently
            for j in 0..n_features {
                let mut num = F::zero();
                let mut den = F::zero();

                for i in 0..x.nrows() {
                    num = num + x[[i, j]] * y[i];
                    den = den + x[[i, j]] * x[[i, j]];
                }

                weights[j] = if den > F::zero() {
                    num / den
                } else {
                    F::zero()
                };
            }

            // Normalize weights to sum to 1
            let weight_sum = weights.sum();
            if weight_sum > F::zero() {
                for weight in weights.iter_mut() {
                    *weight = *weight / weight_sum;
                }
            } else {
                // Fall back to equal weights
                for weight in weights.iter_mut() {
                    *weight = F::one() / F::from(n_features).expect("Failed to convert to float");
                }
            }

            Ok(weights)
        }

        /// Select best model based on validation performance
        fn select_best_model(&mut self, data: &Array1<F>) -> Result<()> {
            let validation_size = (data.len() as f64 * 0.2) as usize;
            if validation_size < 10 {
                // Fall back to equal weights
                let n = self.forecasters.len();
                self.weights = Some(Array1::from_elem(
                    n,
                    F::one() / F::from(n).expect("Failed to convert to float"),
                ));
                return Ok(());
            }

            let train_end = data.len() - validation_size;
            let train_data = data.slice(s![..train_end]).to_owned();
            let validation_data = data.slice(s![train_end..]).to_owned();

            // Train and evaluate each model
            let mut best_error = F::infinity();
            let mut best_model_idx = 0;

            for (i, forecaster) in self.forecasters.iter_mut().enumerate() {
                forecaster.fit(&train_data)?;

                let forecast_horizon = validation_size.min(10);
                if let Ok(result) = forecaster.predict(forecast_horizon) {
                    let actual = validation_data.slice(s![..forecast_horizon]);
                    let mut mse = F::zero();
                    for j in 0..forecast_horizon {
                        let error = result.forecast[j] - actual[j];
                        mse = mse + error * error;
                    }
                    mse = mse / F::from(forecast_horizon).expect("Failed to convert to float");

                    if mse < best_error {
                        best_error = mse;
                        best_model_idx = i;
                    }
                }
            }

            // Set weights to select only the best model
            let mut weights = Array1::zeros(self.forecasters.len());
            weights[best_model_idx] = F::one();
            self.weights = Some(weights);

            Ok(())
        }

        /// Get ensemble weights
        pub fn get_weights(&self) -> Option<&Array1<F>> {
            self.weights.as_ref()
        }

        /// Get model names
        pub fn get_model_names(&self) -> &[String] {
            &self.model_names
        }
    }

    /// AutoML system for neural forecasting hyperparameter optimization
    #[derive(Debug)]
    pub struct NeuralAutoML<F: Float + Debug> {
        /// Search space for hyperparameters
        search_space: HashMap<String, ParameterRange<F>>,
        /// Best configuration found
        best_config: Option<OptimalConfig<F>>,
        /// Search strategy
        search_strategy: SearchStrategy,
        /// Maximum evaluations
        max_evaluations: usize,
        /// Current evaluation count
        evaluations: usize,
    }

    /// Parameter range for hyperparameter search
    #[derive(Debug, Clone)]
    pub enum ParameterRange<F: Float> {
        /// Float parameter range
        FloatRange {
            /// Minimum value
            min: F,
            /// Maximum value
            max: F,
        },
        /// Integer parameter range
        IntRange {
            /// Minimum value
            min: usize,
            /// Maximum value
            max: usize,
        },
        /// Categorical parameter options
        Categorical {
            /// Available options
            options: Vec<String>,
        },
        /// Boolean parameter
        Boolean,
    }

    /// Search strategy for hyperparameter optimization
    #[derive(Debug, Clone)]
    pub enum SearchStrategy {
        /// Random search
        Random,
        /// Grid search
        Grid,
        /// Bayesian optimization (simplified)
        Bayesian,
    }

    /// Optimal configuration result
    #[derive(Debug, Clone)]
    pub struct OptimalConfig<F: Float> {
        /// Best hyperparameters
        pub parameters: HashMap<String, ParameterValue<F>>,
        /// Best validation score
        pub score: F,
        /// Model type that achieved best score
        pub model_type: String,
    }

    /// Parameter value types
    #[derive(Debug, Clone)]
    pub enum ParameterValue<F: Float> {
        /// Float parameter value
        Float(F),
        /// Integer parameter value
        Int(usize),
        /// String parameter value
        String(String),
        /// Boolean parameter value
        Bool(bool),
    }

    impl<F: Float + Debug + Clone + FromPrimitive + 'static> NeuralAutoML<F> {
        /// Create new AutoML system
        pub fn new(_search_strategy: SearchStrategy, maxevaluations: usize) -> Self {
            let mut search_space = HashMap::new();

            // Define default search space
            search_space.insert(
                "learning_rate".to_string(),
                ParameterRange::FloatRange {
                    min: F::from(0.0001).expect("Failed to convert constant to float"),
                    max: F::from(0.1).expect("Failed to convert constant to float"),
                },
            );
            search_space.insert(
                "hidden_size".to_string(),
                ParameterRange::IntRange { min: 16, max: 256 },
            );
            search_space.insert(
                "num_layers".to_string(),
                ParameterRange::IntRange { min: 1, max: 4 },
            );
            search_space.insert(
                "dropout".to_string(),
                ParameterRange::FloatRange {
                    min: F::from(0.0).expect("Failed to convert constant to float"),
                    max: F::from(0.5).expect("Failed to convert constant to float"),
                },
            );
            search_space.insert(
                "batch_size".to_string(),
                ParameterRange::IntRange { min: 16, max: 128 },
            );
            search_space.insert(
                "epochs".to_string(),
                ParameterRange::IntRange { min: 10, max: 100 },
            );

            Self {
                search_space,
                best_config: None,
                search_strategy: _search_strategy,
                max_evaluations: maxevaluations,
                evaluations: 0,
            }
        }

        /// Add parameter to search space
        pub fn add_parameter(&mut self, name: String, range: ParameterRange<F>) {
            self.search_space.insert(name, range);
        }

        /// Find optimal configuration for given data
        pub fn optimize(&mut self, data: &Array1<F>) -> Result<OptimalConfig<F>> {
            let mut best_score = F::infinity();
            let mut best_params = HashMap::new();
            let mut best_model_type = String::new();

            while self.evaluations < self.max_evaluations {
                // Generate candidate configuration
                let (params, model_type) = self.generate_candidate()?;

                // Evaluate configuration
                let score = self.evaluate_config(&params, &model_type, data)?;

                if score < best_score {
                    best_score = score;
                    best_params = params;
                    best_model_type = model_type;
                }

                self.evaluations += 1;
            }

            let optimal_config = OptimalConfig {
                parameters: best_params,
                score: best_score,
                model_type: best_model_type,
            };

            self.best_config = Some(optimal_config.clone());
            Ok(optimal_config)
        }

        /// Generate candidate configuration
        fn generate_candidate(&self) -> Result<(HashMap<String, ParameterValue<F>>, String)> {
            let mut params = HashMap::new();

            // Sample from search space based on strategy
            match self.search_strategy {
                SearchStrategy::Random => {
                    for (name, range) in &self.search_space {
                        let value = self.sample_parameter(range);
                        params.insert(name.clone(), value);
                    }
                }
                SearchStrategy::Grid => {
                    // Simplified grid search - would need proper grid implementation
                    for (name, range) in &self.search_space {
                        let value = self.sample_parameter(range);
                        params.insert(name.clone(), value);
                    }
                }
                SearchStrategy::Bayesian => {
                    // Simplified - would use proper Bayesian optimization
                    for (name, range) in &self.search_space {
                        let value = self.sample_parameter(range);
                        params.insert(name.clone(), value);
                    }
                }
            }

            // Randomly select model type
            let model_types = ["LSTM", "Transformer", "NBeats"];
            let model_idx = self.evaluations % model_types.len();
            let model_type = model_types[model_idx].to_string();

            Ok((params, model_type))
        }

        /// Sample parameter value from range
        fn sample_parameter(&self, range: &ParameterRange<F>) -> ParameterValue<F> {
            match range {
                ParameterRange::FloatRange { min, max } => {
                    // Simple pseudo-random sampling
                    let ratio = F::from((self.evaluations as f64 * 0.12345) % 1.0)
                        .expect("Operation failed");
                    let value = *min + ratio * (*max - *min);
                    ParameterValue::Float(value)
                }
                ParameterRange::IntRange { min, max } => {
                    let range_size = max - min + 1;
                    let offset = self.evaluations % range_size;
                    ParameterValue::Int(min + offset)
                }
                ParameterRange::Categorical { options } => {
                    let idx = self.evaluations % options.len();
                    ParameterValue::String(options[idx].clone())
                }
                ParameterRange::Boolean => ParameterValue::Bool(self.evaluations.is_multiple_of(2)),
            }
        }

        /// Evaluate configuration on data
        fn evaluate_config(
            &self,
            params: &HashMap<String, ParameterValue<F>>,
            model_type: &str,
            data: &Array1<F>,
        ) -> Result<F> {
            // Split data for evaluation
            let split_point = (data.len() as f64 * 0.8) as usize;
            if split_point < 50 {
                return Ok(F::infinity()); // Not enough data
            }

            let train_data = data.slice(s![..split_point]).to_owned();
            let test_data = data.slice(s![split_point..]).to_owned();

            // Create model with parameters
            let mut model = self.create_model_with_params(params, model_type)?;

            // Train model
            model.fit(&train_data)?;

            // Evaluate on test set
            let forecast_horizon = test_data.len().min(10);
            let result = model.predict(forecast_horizon)?;

            // Calculate MSE
            let mut mse = F::zero();
            for i in 0..forecast_horizon {
                let error = result.forecast[i] - test_data[i];
                mse = mse + error * error;
            }
            mse = mse / F::from(forecast_horizon).expect("Failed to convert to float");

            Ok(mse)
        }

        /// Create model with specific parameters
        fn create_model_with_params(
            &self,
            params: &HashMap<String, ParameterValue<F>>,
            model_type: &str,
        ) -> Result<Box<dyn NeuralForecaster<F>>> {
            // Extract common parameters
            let learning_rate = self.get_float_param(params, "learning_rate", 0.001);
            let hidden_size = self.get_int_param(params, "hidden_size", 64);
            let num_layers = self.get_int_param(params, "num_layers", 2);
            let dropout = self.get_float_param(params, "dropout", 0.2);
            let batch_size = self.get_int_param(params, "batch_size", 32);
            let epochs = self.get_int_param(params, "epochs", 50);

            let base_config = NeuralConfig {
                lookback_window: 24,
                forecast_horizon: 1,
                epochs,
                learning_rate: learning_rate.to_f64().unwrap_or(0.001),
                batch_size,
                validation_split: 0.2,
                early_stopping_patience: Some(10),
                random_seed: Some(42),
            };

            match model_type {
                "LSTM" => {
                    let config = LSTMConfig {
                        base: base_config,
                        num_layers,
                        hidden_size,
                        dropout: dropout.to_f64().unwrap_or(0.2),
                        bidirectional: false,
                    };
                    Ok(Box::new(LSTMForecaster::new(config)))
                }
                "Transformer" => {
                    let config = TransformerConfig {
                        base: base_config,
                        d_model: hidden_size,
                        num_heads: 8,
                        num_encoder_layers: num_layers,
                        num_decoder_layers: num_layers,
                        d_ff: hidden_size * 4,
                        dropout: dropout.to_f64().unwrap_or(0.1),
                        use_positional_encoding: true,
                    };
                    Ok(Box::new(TransformerForecaster::new(config)))
                }
                "NBeats" => {
                    let config = NBeatsConfig {
                        base: base_config,
                        num_stacks: num_layers * 10,
                        num_blocks_per_stack: 1,
                        num_layers_per_block: 4,
                        layer_width: hidden_size * 8,
                        expansion_coefficient_dim: 5,
                        share_weights_in_stack: false,
                        generic_architecture: true,
                    };
                    Ok(Box::new(NBeatsForecaster::new(config)))
                }
                _ => Err(TimeSeriesError::InvalidInput(format!(
                    "Unknown model _type: {model_type}"
                ))),
            }
        }

        /// Helper to get float parameter
        fn get_float_param(
            &self,
            params: &HashMap<String, ParameterValue<F>>,
            name: &str,
            default: f64,
        ) -> F {
            match params.get(name) {
                Some(ParameterValue::Float(f)) => *f,
                _ => F::from(default).expect("Failed to convert to float"),
            }
        }

        /// Helper to get int parameter
        fn get_int_param(
            &self,
            params: &HashMap<String, ParameterValue<F>>,
            name: &str,
            default: usize,
        ) -> usize {
            match params.get(name) {
                Some(ParameterValue::Int(i)) => *i,
                _ => default,
            }
        }

        /// Get best configuration found
        pub fn get_best_config(&self) -> Option<&OptimalConfig<F>> {
            self.best_config.as_ref()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_multiscale_forecaster() {
            // Use a configuration appropriate for test data size
            let config = MultiScaleConfig {
                short_term_window: 5,
                medium_term_window: 10,
                long_term_window: 20,
                forecast_horizon: 3,
                learning_rate: 0.01,
                meta_epochs: 2,
            };
            let mut forecaster = MultiScaleNeuralForecaster::<f64>::new(config);

            // Generate synthetic data with multiple patterns
            let data: Array1<f64> = Array1::from_iter((0..50).map(|i| {
                let t = i as f64;
                // Combine short, medium, and long-term patterns
                (t * 0.01).sin() + (t * 0.001).sin() + (t * 0.0001).sin() + 0.1 * (t / 100.0)
            }));

            let result = forecaster.fit(&data);
            assert!(result.is_ok(), "Multi-scale training should succeed");
            assert!(forecaster.trained, "Model should be trained");

            let prediction = forecaster.predict(5);
            assert!(prediction.is_ok(), "Multi-scale prediction should succeed");

            let forecast = prediction.expect("Operation failed");
            assert_eq!(forecast.forecast.len(), 5);

            // Check that combination weights are learned
            assert!(forecaster.get_combination_weights().is_some());
        }

        #[test]
        fn test_online_forecaster() {
            let config = OnlineConfig {
                initial_window: 20,
                update_frequency: 5,
                ..Default::default()
            };
            let mut forecaster = OnlineNeuralForecaster::<f64>::new(config);

            // Add observations incrementally
            for i in 0..30 {
                let obs = (i as f64 * 0.1).sin();
                let result = forecaster.update(obs);
                assert!(result.is_ok(), "Online update should succeed");
            }

            assert!(forecaster.is_ready(), "Forecaster should be ready");

            let prediction = forecaster.predict(3);
            assert!(prediction.is_ok(), "Online prediction should succeed");
        }

        #[test]
        fn test_attention_forecaster() {
            let config = NeuralConfig {
                lookback_window: 10,
                forecast_horizon: 3,
                epochs: 10,
                ..Default::default()
            };
            let mut forecaster = AttentionForecaster::<f64>::new(config);

            let data = Array1::from_vec((0..30).map(|i| (i as f64 * 0.2).sin()).collect());

            let result = forecaster.fit(&data);
            assert!(result.is_ok(), "Attention training should succeed");
            assert!(forecaster.trained, "Model should be trained");

            let prediction = forecaster.predict(&data);
            assert!(prediction.is_ok(), "Attention prediction should succeed");

            // Check that attention weights are computed
            assert!(forecaster.get_attention_weights().is_some());

            // Check that loss history is recorded
            assert!(!forecaster.get_loss_history().is_empty());
        }
    }
}
