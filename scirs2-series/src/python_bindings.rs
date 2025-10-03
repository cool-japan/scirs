//! Python bindings for scirs2-series using PyO3
//!
//! This module provides Python bindings for seamless integration with pandas,
//! statsmodels, and other Python time series analysis libraries.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::types::{PyAny, PyDict, PyType};

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

#[cfg(feature = "python")]
use scirs2_core::ndarray::Array1;

#[cfg(feature = "python")]
use std::collections::HashMap;

#[cfg(feature = "python")]
use crate::{
    arima_models::{ArimaConfig, ArimaModel},
    utils::*,
};

#[cfg(feature = "python")]
use statrs::statistics::Statistics;

/// Python wrapper for time series data
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTimeSeries {
    values: Array1<f64>,
    timestamps: Option<Array1<f64>>,
    frequency: Option<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTimeSeries {
    /// Create a new time series from Python list or numpy array
    #[new]
    fn new(
        values: PyReadonlyArray1<f64>,
        timestamps: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let values_array = values.as_array().to_owned();
        let timestamps_array = timestamps.map(|ts| ts.as_array().to_owned());

        Ok(PyTimeSeries {
            values: values_array,
            timestamps: timestamps_array,
            frequency: None,
        })
    }

    /// Set the frequency of the time series
    fn set_frequency(&mut self, frequency: f64) {
        self.frequency = Some(frequency);
    }

    /// Get the length of the time series
    fn __len__(&self) -> usize {
        self.values.len()
    }

    /// Get values as numpy array
    fn get_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.clone().into_pyarray(py)
    }

    /// Get timestamps as numpy array (if available)
    fn get_timestamps<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.timestamps
            .as_ref()
            .map(|ts| ts.clone().into_pyarray(py))
    }

    /// Convert to pandas-compatible dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("values", self.values.clone().into_pyarray(py))?;

        if let Some(ref timestamps) = self.timestamps {
            dict.set_item("timestamps", timestamps.clone().into_pyarray(py))?;
        }

        if let Some(freq) = self.frequency {
            dict.set_item("frequency", freq)?;
        }

        Ok(dict.into())
    }

    /// Create from pandas Series
    #[classmethod]
    fn from_pandas(cls: &Bound<'_, PyType>, series: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Extract values from pandas Series
        let values = series.getattr("values")?;
        let values_array: PyReadonlyArray1<f64> = values.extract()?;

        // Try to extract index (timestamps) if available
        let index = series.getattr("index")?;
        let timestamps = if index.hasattr("values")? {
            let index_values: PyResult<PyReadonlyArray1<f64>> = index.getattr("values")?.extract();
            index_values.ok()
        } else {
            None
        };

        Self::new(values_array, timestamps)
    }

    /// Statistical summary
    fn describe(&self) -> PyResult<HashMap<String, f64>> {
        let mut stats = HashMap::new();
        let values = &self.values;

        let mean = values.mean().unwrap_or(0.0);
        let std = values.std(0.0);
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        stats.insert("count".to_string(), values.len() as f64);
        stats.insert("mean".to_string(), mean);
        stats.insert("std".to_string(), std);
        stats.insert("min".to_string(), min);
        stats.insert("max".to_string(), max);

        // Calculate quantiles
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_values.len();

        stats.insert("25%".to_string(), sorted_values[n / 4]);
        stats.insert("50%".to_string(), sorted_values[n / 2]);
        stats.insert("75%".to_string(), sorted_values[3 * n / 4]);

        Ok(stats)
    }
}

/// Python wrapper for ARIMA models
#[cfg(feature = "python")]
#[pyclass]
pub struct PyARIMA {
    model: Option<ArimaModel<f64>>,
    config: ArimaConfig,
    data: Option<Array1<f64>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyARIMA {
    /// Create a new ARIMA model
    #[new]
    fn new(p: usize, d: usize, q: usize) -> Self {
        let config = ArimaConfig {
            p,
            d,
            q,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            seasonal_period: 0,
        };

        PyARIMA {
            model: None,
            config,
            data: None,
        }
    }

    /// Create SARIMA model
    #[classmethod]
    fn sarima(
        _cls: &Bound<'_, PyType>,
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> Self {
        let config = ArimaConfig {
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            seasonal_period,
        };

        PyARIMA {
            model: None,
            config,
            data: None,
        }
    }

    /// Fit the ARIMA model
    fn fit(&mut self, data: &PyTimeSeries) -> PyResult<()> {
        let mut model = ArimaModel::new(self.config.p, self.config.d, self.config.q)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        model
            .fit(&data.values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        self.model = Some(model);
        self.data = Some(data.values.clone());
        Ok(())
    }

    /// Generate forecasts
    fn forecast<'py>(&self, py: Python<'py>, steps: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match (&self.model, &self.data) {
            (Some(model), Some(data)) => {
                let forecasts = model
                    .forecast(steps, data)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                Ok(forecasts.into_pyarray(py))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            )),
        }
    }

    /// Get model parameters
    fn get_params(&self) -> PyResult<HashMap<String, f64>> {
        let mut params = HashMap::new();
        params.insert("p".to_string(), self.config.p as f64);
        params.insert("d".to_string(), self.config.d as f64);
        params.insert("q".to_string(), self.config.q as f64);
        params.insert("seasonal_p".to_string(), self.config.seasonal_p as f64);
        params.insert("seasonal_d".to_string(), self.config.seasonal_d as f64);
        params.insert("seasonal_q".to_string(), self.config.seasonal_q as f64);
        params.insert(
            "seasonal_period".to_string(),
            self.config.seasonal_period as f64,
        );
        Ok(params)
    }

    /// Get model summary (similar to statsmodels)
    fn summary(&self) -> PyResult<String> {
        match &self.model {
            Some(_model) => {
                // Get parameters from config since model doesn't have get_params method
                let mut params = HashMap::new();
                params.insert("p".to_string(), self.config.p as f64);
                params.insert("d".to_string(), self.config.d as f64);
                params.insert("q".to_string(), self.config.q as f64);

                let mut summary = format!(
                    "ARIMA({},{},{}) Model Results\n",
                    self.config.p, self.config.d, self.config.q
                );
                summary.push_str("=====================================\n");

                for (param, value) in params {
                    summary.push_str(&format!("{param:20}: {value:10.6}\n"));
                }

                Ok(summary)
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            )),
        }
    }
}

/// Utility functions for Python integration
#[cfg(feature = "python")]
#[pyfunction]
#[allow(dead_code)]
fn calculate_statistics(data: &PyTimeSeries) -> PyResult<HashMap<String, f64>> {
    let stats = calculate_basic_stats(&data.values)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(stats)
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(dead_code)]
fn check_stationarity(data: &PyTimeSeries) -> PyResult<bool> {
    let (_test_stat, p_value) = is_stationary(&data.values, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    // Consider stationary if p-value < 0.05 (5% significance level)
    Ok(p_value < 0.05)
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(dead_code)]
fn apply_differencing<'py>(
    py: Python<'py>,
    data: &PyTimeSeries,
    periods: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let differenced = difference_series(&data.values, periods)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(differenced.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(dead_code)]
fn apply_seasonal_differencing<'py>(
    py: Python<'py>,
    data: &PyTimeSeries,
    periods: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let differenced = seasonal_difference_series(&data.values, periods)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(differenced.into_pyarray(py))
}

/// Auto-ARIMA functionality for Python
#[cfg(feature = "python")]
#[pyfunction]
#[allow(dead_code)]
fn auto_arima(
    data: &PyTimeSeries,
    max_p: usize,
    max_d: usize,
    max_q: usize,
    seasonal: bool,
    max_seasonal_p: Option<usize>,
    max_seasonal_d: Option<usize>,
    max_seasonal_q: Option<usize>,
    seasonal_period: Option<usize>,
) -> PyResult<PyARIMA> {
    let max_sp = max_seasonal_p.unwrap_or(0);
    let max_sd = max_seasonal_d.unwrap_or(0);
    let max_sq = max_seasonal_q.unwrap_or(0);
    let s_period = seasonal_period.unwrap_or(0);

    let options = crate::arima_models::ArimaSelectionOptions {
        max_p,
        max_d,
        max_q,
        seasonal,
        max_seasonal_p: max_sp,
        max_seasonal_d: max_sd,
        max_seasonal_q: max_sq,
        seasonal_period: Some(s_period),
        ..Default::default()
    };

    let (_model, params) = crate::arima_models::auto_arima(&data.values, &options)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let config = ArimaConfig {
        p: params.pdq.0,
        d: params.pdq.1,
        q: params.pdq.2,
        seasonal_p: params.seasonal_pdq.0,
        seasonal_d: params.seasonal_pdq.1,
        seasonal_q: params.seasonal_pdq.2,
        seasonal_period: params.seasonal_period,
    };

    Ok(PyARIMA {
        model: None,
        config,
        data: None,
    })
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
#[allow(dead_code)]
fn scirs2_series(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyARIMA>()?;

    m.add_function(wrap_pyfunction!(calculate_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(check_stationarity, m)?)?;
    m.add_function(wrap_pyfunction!(apply_differencing, m)?)?;
    m.add_function(wrap_pyfunction!(apply_seasonal_differencing, m)?)?;
    m.add_function(wrap_pyfunction!(auto_arima, m)?)?;

    Ok(())
}

// Helper functions for pandas integration

/// Creates a pandas DataFrame from a HashMap of Array1<f64> data
///
/// # Arguments
/// * `py` - Python interpreter instance
/// * `data` - HashMap where keys are column names and values are Array1<f64> data
///
/// # Returns
/// A PyResult containing the pandas DataFrame object
#[cfg(feature = "python")]
#[allow(dead_code)]
pub fn create_pandas_dataframe(
    py: Python,
    data: HashMap<String, Array1<f64>>,
) -> PyResult<PyObject> {
    let pandas = py.import("pandas")?;
    let dict = PyDict::new(py);

    for (key, values) in data {
        dict.set_item(key, values.into_pyarray(py))?;
    }

    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}

/// Creates a pandas Series from a Rust Array1<f64>
///
/// # Arguments
/// * `py` - Python interpreter instance
/// * `data` - Array1<f64> containing the data
/// * `name` - Optional name for the series
///
/// # Returns
/// A PyResult containing the pandas Series object
#[cfg(feature = "python")]
#[allow(dead_code)]
pub fn create_pandas_series(
    py: Python,
    data: Array1<f64>,
    name: Option<&str>,
) -> PyResult<PyObject> {
    let pandas = py.import("pandas")?;
    let args = (data.into_pyarray(py),);
    let kwargs = PyDict::new(py);

    if let Some(name) = name {
        kwargs.set_item("name", name)?;
    }

    let series = pandas.call_method("Series", args, Some(&kwargs))?;
    Ok(series.into())
}
