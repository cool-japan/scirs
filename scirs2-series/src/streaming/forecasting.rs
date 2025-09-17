//! Real-time forecasting capabilities for streaming time series
//!
//! This module provides streaming forecasting algorithms including exponential smoothing
//! with trend and seasonal components for real-time time series forecasting.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Real-time forecasting with online model updates
#[derive(Debug)]
pub struct StreamingForecaster<F: Float + Debug> {
    /// Exponential smoothing parameter
    alpha: F,
    /// Trend parameter
    beta: Option<F>,
    /// Seasonal parameter
    gamma: Option<F>,
    /// Seasonal period
    seasonal_period: Option<usize>,
    /// Current level
    level: Option<F>,
    /// Current trend
    trend: Option<F>,
    /// Seasonal components
    seasonal: VecDeque<F>,
    /// Recent observations buffer
    buffer: VecDeque<F>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Number of observations processed
    observation_count: usize,
}

impl<F: Float + Debug + Clone> StreamingForecaster<F> {
    /// Create new streaming forecaster
    pub fn new(
        alpha: F,
        beta: Option<F>,
        gamma: Option<F>,
        seasonal_period: Option<usize>,
        max_buffer_size: usize,
    ) -> Result<Self> {
        if alpha <= F::zero() || alpha > F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "Alpha must be between 0 and 1".to_string(),
            });
        }

        let seasonal = if let Some(_period) = seasonal_period {
            VecDeque::with_capacity(_period)
        } else {
            VecDeque::new()
        };

        Ok(Self {
            alpha,
            beta,
            gamma,
            seasonal_period,
            level: None,
            trend: None,
            seasonal,
            buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            observation_count: 0,
        })
    }

    /// Add new observation and update model
    pub fn update(&mut self, value: F) -> Result<()> {
        self.observation_count += 1;

        // Add to buffer
        if self.buffer.len() >= self.max_buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);

        // Initialize components
        if self.level.is_none() {
            self.level = Some(value);
            if self.beta.is_some() {
                self.trend = Some(F::zero());
            }
            if let Some(period) = self.seasonal_period {
                for _ in 0..period {
                    self.seasonal.push_back(F::zero());
                }
            }
            return Ok(());
        }

        let current_level = self.level.unwrap();
        let mut new_level = value;

        // Handle seasonality
        let _seasonal_component = if let Some(period) = self.seasonal_period {
            if self.seasonal.len() >= period {
                let seasonal_idx = (self.observation_count - 1) % period;
                let seasonal_val = self.seasonal[seasonal_idx];
                new_level = new_level - seasonal_val;
                seasonal_val
            } else {
                F::zero()
            }
        } else {
            F::zero()
        };

        // Update level
        self.level = Some(self.alpha * new_level + (F::one() - self.alpha) * current_level);

        // Update trend if enabled
        if let Some(beta) = self.beta {
            if let Some(current_trend) = self.trend {
                let new_trend = beta * (self.level.unwrap() - current_level)
                    + (F::one() - beta) * current_trend;
                self.trend = Some(new_trend);
            }
        }

        // Update seasonal component if enabled
        if let (Some(gamma), Some(period)) = (self.gamma, self.seasonal_period) {
            if self.seasonal.len() >= period {
                let seasonal_idx = (self.observation_count - 1) % period;
                let current_seasonal = self.seasonal[seasonal_idx];
                let new_seasonal =
                    gamma * (value - self.level.unwrap()) + (F::one() - gamma) * current_seasonal;
                self.seasonal[seasonal_idx] = new_seasonal;
            }
        }

        Ok(())
    }

    /// Generate forecast for next h steps
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if self.level.is_none() {
            return Err(TimeSeriesError::InvalidModel(
                "Model not initialized with any data".to_string(),
            ));
        }

        let mut forecasts = Array1::zeros(steps);
        let level = self.level.unwrap();
        let trend = self.trend.unwrap_or(F::zero());

        for h in 0..steps {
            let h_f = F::from(h + 1).unwrap();
            let mut forecast = level + trend * h_f;

            // Add seasonal component if available
            if let Some(period) = self.seasonal_period {
                if !self.seasonal.is_empty() {
                    let seasonal_idx = (self.observation_count + h) % period;
                    if seasonal_idx < self.seasonal.len() {
                        forecast = forecast + self.seasonal[seasonal_idx];
                    }
                }
            }

            forecasts[h] = forecast;
        }

        Ok(forecasts)
    }

    /// Get current model state summary
    pub fn get_state(&self) -> ModelState<F> {
        ModelState {
            level: self.level,
            trend: self.trend,
            seasonal_components: self.seasonal.iter().cloned().collect(),
            observation_count: self.observation_count,
            buffer_size: self.buffer.len(),
        }
    }
}

/// Model state summary
#[derive(Debug, Clone)]
pub struct ModelState<F: Float> {
    /// Current level component
    pub level: Option<F>,
    /// Current trend component
    pub trend: Option<F>,
    /// Seasonal components vector
    pub seasonal_components: Vec<F>,
    /// Number of observations processed
    pub observation_count: usize,
    /// Size of the internal buffer
    pub buffer_size: usize,
}
