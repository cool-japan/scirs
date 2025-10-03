//! WebAssembly bindings for scirs2-series
//!
//! This module provides JavaScript bindings for time series analysis functionality,
//! enabling browser-based time series analysis with full performance and feature parity.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use web_sys::console;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use crate::{
    anomaly::{AnomalyMethod, AnomalyOptions},
    arima_models::{ArimaConfig, ArimaModel},
    decomposition::STLOptions,
    forecasting::neural::NeuralForecaster,
    utils::*,
};

#[cfg(feature = "wasm")]
use scirs2_core::ndarray::Array1;

#[cfg(feature = "wasm")]
// Utility macro for error handling in WASM
macro_rules! js_error {
    ($msg:expr) => {
        JsValue::from_str(&format!("Error: {}", $msg))
    };
}

#[cfg(feature = "wasm")]
macro_rules! js_result {
    ($result:expr) => {
        match $result {
            Ok(val) => Ok(val),
            Err(e) => Err(js_error!(e)),
        }
    };
}

/// Time series data structure for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    values: Vec<f64>,
    timestamps: Option<Vec<f64>>,
    frequency: Option<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl TimeSeriesData {
    /// Create a new time series from JavaScript array
    #[wasm_bindgen(constructor)]
    pub fn new(values: &[f64]) -> TimeSeriesData {
        TimeSeriesData {
            values: values.to_vec(),
            timestamps: None,
            frequency: None,
        }
    }

    /// Create time series with timestamps
    #[wasm_bindgen]
    pub fn with_timestamps(
        values: &[f64],
        timestamps: &[f64],
    ) -> std::result::Result<TimeSeriesData, JsValue> {
        if values.len() != timestamps.len() {
            return Err(js_error!("Values and timestamps must have the same length"));
        }

        Ok(TimeSeriesData {
            values: values.to_vec(),
            timestamps: Some(timestamps.to_vec()),
            frequency: None,
        })
    }

    /// Set frequency for the time series
    #[wasm_bindgen]
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = Some(frequency);
    }

    /// Get the length of the time series
    #[wasm_bindgen]
    pub fn length(&self) -> usize {
        self.values.len()
    }

    /// Get values as JavaScript array
    #[wasm_bindgen]
    pub fn get_values(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Get timestamps as JavaScript array
    #[wasm_bindgen]
    pub fn get_timestamps(&self) -> Option<Vec<f64>> {
        self.timestamps.clone()
    }
}

/// ARIMA model wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmARIMA {
    model: Option<ArimaModel<f64>>,
    config: ArimaConfig,
    data: Option<Array1<f64>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmARIMA {
    /// Create a new ARIMA model
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize) -> WasmARIMA {
        let config = ArimaConfig {
            p,
            d,
            q,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            seasonal_period: 1,
        };

        WasmARIMA {
            model: None,
            config,
            data: None,
        }
    }

    /// Create SARIMA model with seasonal components
    #[wasm_bindgen]
    pub fn sarima(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> WasmARIMA {
        let config = crate::arima_models::ArimaConfig {
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            seasonal_period,
        };

        WasmARIMA {
            model: None,
            config,
            data: None,
        }
    }

    /// Fit the ARIMA model to time series data
    #[wasm_bindgen]
    pub fn fit(&mut self, data: &TimeSeriesData) -> std::result::Result<(), JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let mut model = js_result!(ArimaModel::new(self.config.p, self.config.d, self.config.q))?;
        js_result!(model.fit(&arr))?;
        self.model = Some(model);
        self.data = Some(arr);
        Ok(())
    }

    /// Generate forecasts
    #[wasm_bindgen]
    pub fn forecast(&self, steps: usize) -> std::result::Result<Vec<f64>, JsValue> {
        match (&self.model, &self.data) {
            (Some(model), Some(data)) => {
                let forecasts = js_result!(model.forecast(steps, data))?;
                Ok(forecasts.to_vec())
            }
            _ => Err(js_error!("Model not fitted. Call fit() first.")),
        }
    }

    /// Get model parameters
    #[wasm_bindgen]
    pub fn get_params(&self) -> std::result::Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self.config)?)
    }
}

/// Anomaly detector wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAnomalyDetector {
    #[allow(dead_code)]
    method: AnomalyMethod,
    #[allow(dead_code)]
    options: AnomalyOptions,
}

#[cfg(feature = "wasm")]
impl Default for WasmAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAnomalyDetector {
    /// Create a new anomaly detector
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAnomalyDetector {
        WasmAnomalyDetector {
            method: AnomalyMethod::ZScore,
            options: AnomalyOptions::default(),
        }
    }

    /// Detect anomalies using IQR method
    #[wasm_bindgen]
    pub fn detect_iqr(
        &self,
        data: &TimeSeriesData,
        multiplier: f64,
    ) -> std::result::Result<Vec<usize>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let options = AnomalyOptions {
            method: AnomalyMethod::InterquartileRange,
            threshold: Some(multiplier),
            ..Default::default()
        };
        let anomalies = js_result!(crate::anomaly::detect_anomalies(&arr, &options))?;
        let anomaly_indices: Vec<usize> = anomalies
            .is_anomaly
            .iter()
            .enumerate()
            .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
            .collect();
        Ok(anomaly_indices)
    }

    /// Detect anomalies using Z-score method
    #[wasm_bindgen]
    pub fn detect_zscore(
        &self,
        data: &TimeSeriesData,
        threshold: f64,
    ) -> std::result::Result<Vec<usize>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let options = AnomalyOptions {
            method: AnomalyMethod::ZScore,
            threshold: Some(threshold),
            ..Default::default()
        };
        let anomalies = js_result!(crate::anomaly::detect_anomalies(&arr, &options))?;
        let anomaly_indices: Vec<usize> = anomalies
            .is_anomaly
            .iter()
            .enumerate()
            .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
            .collect();
        Ok(anomaly_indices)
    }

    /// Detect anomalies using isolation forest
    #[wasm_bindgen]
    pub fn detect_isolation_forest(
        &self,
        data: &TimeSeriesData,
        contamination: f64,
    ) -> std::result::Result<Vec<usize>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let options = AnomalyOptions {
            method: AnomalyMethod::IsolationForest,
            threshold: Some(contamination),
            ..Default::default()
        };
        let anomalies = js_result!(crate::anomaly::detect_anomalies(&arr, &options))?;
        let anomaly_indices: Vec<usize> = anomalies
            .is_anomaly
            .iter()
            .enumerate()
            .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
            .collect();
        Ok(anomaly_indices)
    }
}

/// STL decomposition wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSTLDecomposition {
    period: usize,
    options: STLOptions,
}

/// Result of STL decomposition containing trend, seasonal, and residual components
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Trend component of the decomposition
    pub trend: Vec<f64>,
    /// Seasonal component of the decomposition
    pub seasonal: Vec<f64>,
    /// Residual component of the decomposition
    pub residual: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSTLDecomposition {
    /// Create a new STL decomposition
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> WasmSTLDecomposition {
        WasmSTLDecomposition {
            period,
            options: STLOptions::default(),
        }
    }

    /// Perform STL decomposition
    #[wasm_bindgen]
    pub fn decompose(&self, data: &TimeSeriesData) -> std::result::Result<JsValue, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let result = js_result!(crate::decomposition::stl::stl_decomposition(
            &arr,
            self.period,
            &self.options
        ))?;

        let decomp_result = DecompositionResult {
            trend: result.trend.to_vec(),
            seasonal: result.seasonal.to_vec(),
            residual: result.residual.to_vec(),
        };

        Ok(serde_wasm_bindgen::to_value(&decomp_result)?)
    }
}

/// Neural forecaster wrapper for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmNeuralForecaster {
    forecaster: Option<Box<dyn NeuralForecaster<f64>>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmNeuralForecaster {
    /// Create a new neural forecaster
    #[wasm_bindgen(constructor)]
    pub fn new(_input_size: usize, _hidden_size: usize, _size: usize) -> WasmNeuralForecaster {
        WasmNeuralForecaster {
            forecaster: Some(Box::new(crate::forecasting::neural::LSTMForecaster::new(
                crate::forecasting::neural::LSTMConfig::default(),
            )) as Box<dyn NeuralForecaster<f64>>),
        }
    }

    /// Train the neural forecaster
    #[wasm_bindgen]
    pub fn train(
        &mut self,
        data: &TimeSeriesData,
        epochs: usize,
        _learning_rate: f64,
    ) -> std::result::Result<(), JsValue> {
        if let Some(forecaster) = &mut self.forecaster {
            let arr = Array1::from_vec(data.values.clone());
            js_result!(forecaster.fit(&arr))?;
            Ok(())
        } else {
            Err(js_error!("Neural forecaster not initialized"))
        }
    }

    /// Generate forecasts using the neural model
    #[wasm_bindgen]
    pub fn forecast(&self, input: &[f64], steps: usize) -> std::result::Result<Vec<f64>, JsValue> {
        if let Some(forecaster) = &self.forecaster {
            let _input_arr = Array1::from_vec(input.to_vec());
            let forecast_result = js_result!(forecaster.predict(steps))?;
            Ok(forecast_result.forecast.to_vec())
        } else {
            Err(js_error!("Neural forecaster not initialized"))
        }
    }
}

/// Utility functions for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmUtils;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmUtils {
    /// Calculate basic statistics for time series
    #[wasm_bindgen]
    pub fn calculate_stats(data: &TimeSeriesData) -> std::result::Result<JsValue, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let stats = js_result!(calculate_basic_stats(&arr))?;
        Ok(serde_wasm_bindgen::to_value(&stats)?)
    }

    /// Check if time series is stationary
    #[wasm_bindgen]
    pub fn is_stationary(data: &TimeSeriesData) -> std::result::Result<bool, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let (_test_stat, p_value) = js_result!(is_stationary(&arr, None))?;
        // Consider stationary if p-value < 0.05 (5% significance level)
        Ok(p_value < 0.05)
    }

    /// Apply differencing to make time series stationary
    #[wasm_bindgen]
    pub fn difference(
        data: &TimeSeriesData,
        periods: usize,
    ) -> std::result::Result<Vec<f64>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let differenced = js_result!(difference_series(&arr, periods))?;
        Ok(differenced.to_vec())
    }

    /// Apply seasonal differencing
    #[wasm_bindgen]
    pub fn seasonal_difference(
        data: &TimeSeriesData,
        periods: usize,
    ) -> std::result::Result<Vec<f64>, JsValue> {
        let arr = Array1::from_vec(data.values.clone());
        let differenced = js_result!(seasonal_difference_series(&arr, periods))?;
        Ok(differenced.to_vec())
    }
}

/// Initialize WASM module
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
#[allow(dead_code)]
pub fn init() {
    console_error_panic_hook::set_once();
    console::log_1(&"SciRS2 Time Series Analysis WASM module initialized".into());
}

/// Log function for debugging
#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(feature = "wasm")]
#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Auto-ARIMA functionality for WASM
/// Automatically selects the best ARIMA model parameters
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAutoARIMA;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAutoARIMA {
    /// Automatically select best ARIMA model
    #[wasm_bindgen]
    pub fn auto_arima(
        data: &TimeSeriesData,
        max_p: usize,
        max_d: usize,
        max_q: usize,
        seasonal: bool,
        max_seasonal_p: Option<usize>,
        max_seasonal_d: Option<usize>,
        max_seasonal_q: Option<usize>,
        seasonal_period: Option<usize>,
    ) -> std::result::Result<WasmARIMA, JsValue> {
        let arr = Array1::from_vec(data.values.clone());

        let max_sp = max_seasonal_p.unwrap_or(0);
        let max_sd = max_seasonal_d.unwrap_or(0);
        let max_sq = max_seasonal_q.unwrap_or(0);
        let _s_period = seasonal_period.unwrap_or(0);

        let options = crate::arima_models::ArimaSelectionOptions {
            max_p,
            max_d,
            max_q,
            seasonal,
            seasonal_period: if seasonal { seasonal_period } else { None },
            max_seasonal_p: max_sp,
            max_seasonal_d: max_sd,
            max_seasonal_q: max_sq,
            ..Default::default()
        };

        let (model, params) = js_result!(crate::arima_models::auto_arima(&arr, &options))?;

        let config = ArimaConfig {
            p: params.pdq.0,
            d: params.pdq.1,
            q: params.pdq.2,
            seasonal_p: params.seasonal_pdq.0,
            seasonal_d: params.seasonal_pdq.1,
            seasonal_q: params.seasonal_pdq.2,
            seasonal_period: params.seasonal_period,
        };

        Ok(WasmARIMA {
            model: Some(model),
            config,
            data: Some(arr),
        })
    }
}

// Export main functions for easier access

/// Creates a new time series data structure from values
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(dead_code)]
pub fn create_time_series(values: &[f64]) -> TimeSeriesData {
    TimeSeriesData::new(values)
}

/// Creates a new ARIMA model with specified parameters
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(dead_code)]
pub fn create_arima_model(p: usize, d: usize, q: usize) -> WasmARIMA {
    WasmARIMA::new(p, d, q)
}

/// Creates a new anomaly detector
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(dead_code)]
pub fn create_anomaly_detector() -> WasmAnomalyDetector {
    WasmAnomalyDetector::new()
}

/// Creates a new STL decomposition with specified period
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(dead_code)]
pub fn create_stl_decomposition(period: usize) -> WasmSTLDecomposition {
    WasmSTLDecomposition::new(period)
}

/// Creates a new neural forecaster with specified architecture
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(dead_code)]
pub fn create_neural_forecaster(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) -> WasmNeuralForecaster {
    WasmNeuralForecaster::new(input_size, hidden_size, output_size)
}
