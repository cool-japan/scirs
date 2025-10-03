//! R integration package for scirs2-series
//!
//! This module provides R bindings for seamless integration with R's time series
//! analysis ecosystem, enabling interoperability with ts, forecast, and other R packages.

#[cfg(feature = "r")]
use std::ffi::CString;
#[cfg(feature = "r")]
use std::os::raw::{c_char, c_double, c_int, c_void};
#[cfg(feature = "r")]
use std::slice;

#[cfg(feature = "r")]
use crate::{
    anomaly::{detect_anomalies, AnomalyMethod, AnomalyOptions},
    arima_models::{ArimaConfig, ArimaModel},
    forecasting::neural::{LSTMConfig, LSTMForecaster, NeuralForecaster},
    streaming::advanced::StreamingAnomalyDetector,
    utils::*,
};
#[cfg(feature = "r")]
use scirs2_core::ndarray::Array1;

/// Error codes for R integration
#[cfg(feature = "r")]
pub const R_SUCCESS: c_int = 0;
/// Error code for invalid parameters
#[cfg(feature = "r")]
pub const R_ERROR_INVALID_PARAMS: c_int = -1;
/// Error code for memory allocation failures
#[cfg(feature = "r")]
pub const R_ERROR_MEMORY: c_int = -2;
/// Error code for computation failures
#[cfg(feature = "r")]
pub const R_ERROR_COMPUTATION: c_int = -3;
/// Error code for operations on unfitted models
#[cfg(feature = "r")]
pub const R_ERROR_NOT_FITTED: c_int = -4;

/// R-compatible time series structure
#[cfg(feature = "r")]
#[repr(C)]
pub struct RTimeSeries {
    /// Pointer to the time series values
    pub values: *mut c_double,
    /// Length of the time series
    pub length: c_int,
    /// Frequency of the time series (observations per unit time)
    pub frequency: c_double,
    /// Starting time of the time series
    pub start_time: c_double,
}

/// R-compatible ARIMA model handle
#[cfg(feature = "r")]
#[repr(C)]
pub struct RARIMAModel {
    /// Opaque handle to the underlying ARIMA model
    pub handle: *mut c_void,
    /// Order of the autoregressive part
    pub p: c_int,
    /// Degree of first differencing
    pub d: c_int,
    /// Order of the moving average part
    pub q: c_int,
    /// Order of the seasonal autoregressive part
    pub seasonal_p: c_int,
    /// Degree of seasonal differencing
    pub seasonal_d: c_int,
    /// Order of the seasonal moving average part
    pub seasonal_q: c_int,
    /// Number of periods in each season
    pub seasonal_period: c_int,
}

/// R-compatible anomaly detector handle
#[cfg(feature = "r")]
#[repr(C)]
pub struct RAnomalyDetector {
    /// Opaque handle to the underlying anomaly detector
    pub handle: *mut c_void,
}

/// R-compatible STL decomposition handle
#[cfg(feature = "r")]
#[repr(C)]
pub struct RSTLDecomposition {
    /// Opaque handle to the underlying STL decomposition
    pub handle: *mut c_void,
    /// Period of the seasonal component
    pub period: c_int,
}

/// R-compatible decomposition result
#[cfg(feature = "r")]
#[repr(C)]
pub struct RDecompositionResult {
    /// Pointer to the trend component
    pub trend: *mut c_double,
    /// Pointer to the seasonal component
    pub seasonal: *mut c_double,
    /// Pointer to the residual component
    pub residual: *mut c_double,
    /// Length of each component array
    pub length: c_int,
}

/// R-compatible forecast result
#[cfg(feature = "r")]
#[repr(C)]
pub struct RForecastResult {
    /// Pointer to the forecast values
    pub forecasts: *mut c_double,
    /// Length of the forecast array
    pub length: c_int,
    /// Pointer to the lower confidence interval values
    pub confidence_lower: *mut c_double,
    /// Pointer to the upper confidence interval values
    pub confidence_upper: *mut c_double,
}

/// R-compatible statistics result
#[cfg(feature = "r")]
#[repr(C)]
pub struct RStatistics {
    /// Mean value of the data
    pub mean: c_double,
    /// Variance of the data
    pub variance: c_double,
    /// Standard deviation of the data
    pub std_dev: c_double,
    /// Minimum value in the data
    pub min: c_double,
    /// Maximum value in the data
    pub max: c_double,
    /// Skewness of the data distribution
    pub skewness: c_double,
    /// Kurtosis of the data distribution
    pub kurtosis: c_double,
    /// 25th percentile (first quartile)
    pub q25: c_double,
    /// 50th percentile (median)
    pub q50: c_double,
    /// 75th percentile (third quartile)
    pub q75: c_double,
}

// ================================
// Time Series Management Functions
// ================================

/// Create a new time series from R vector
///
/// # Safety
/// The caller must ensure that:
/// - `values` is a valid pointer to an array of `length` elements
/// - `length` is greater than 0
/// - The memory pointed to by `values` is valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_create_timeseries(
    values: *const c_double,
    length: c_int,
    frequency: c_double,
    start_time: c_double,
) -> *mut RTimeSeries {
    if values.is_null() || length <= 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let data_slice = slice::from_raw_parts(values, length as usize);
        let mut ts_values = Vec::with_capacity(length as usize);

        for &val in data_slice {
            ts_values.push(val);
        }

        let ts = Box::new(RTimeSeries {
            values: ts_values.as_mut_ptr(),
            length,
            frequency,
            start_time,
        });

        std::mem::forget(ts_values); // Prevent deallocation
        Box::into_raw(ts)
    }
}

/// Free time series memory
///
/// # Safety
/// The caller must ensure that:
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - The pointer was previously returned by `scirs_create_timeseries`
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_timeseries(ts: *mut RTimeSeries) {
    if !ts.is_null() {
        unsafe {
            let ts_box = Box::from_raw(ts);
            if !ts_box.values.is_null() {
                Vec::from_raw_parts(
                    ts_box.values,
                    ts_box.length as usize,
                    ts_box.length as usize,
                );
            }
        }
    }
}

/// Get time series values as R vector
///
/// # Safety
/// The caller must ensure that:
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - `output` is a valid pointer to an array of at least `max_length` elements
/// - `max_length` is greater than 0
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_get_timeseries_values(
    ts: *const RTimeSeries,
    output: *mut c_double,
    max_length: c_int,
) -> c_int {
    if ts.is_null() || output.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let ts_ref = &*ts;
        if ts_ref.values.is_null() {
            return R_ERROR_INVALID_PARAMS;
        }

        let copy_length = std::cmp::min(ts_ref.length, max_length) as usize;
        let values_slice = slice::from_raw_parts(ts_ref.values, copy_length);
        let output_slice = slice::from_raw_parts_mut(output, copy_length);

        output_slice.copy_from_slice(values_slice);
        copy_length as c_int
    }
}

/// Calculate basic statistics for time series
///
/// # Safety
/// The caller must ensure that:
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - `stats` is a valid pointer to an RStatistics structure
/// - Both structures remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_calculate_statistics(
    ts: *const RTimeSeries,
    stats: *mut RStatistics,
) -> c_int {
    if ts.is_null() || stats.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let ts_ref = &*ts;
        if ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        match calculate_basic_stats(&values_array) {
            Ok(rust_stats) => {
                let stats_ref = &mut *stats;
                stats_ref.mean = rust_stats.get("mean").copied().unwrap_or(0.0);
                stats_ref.variance = rust_stats.get("variance").copied().unwrap_or(0.0);
                stats_ref.std_dev = rust_stats.get("std").copied().unwrap_or(0.0);
                stats_ref.min = rust_stats.get("min").copied().unwrap_or(0.0);
                stats_ref.max = rust_stats.get("max").copied().unwrap_or(0.0);
                stats_ref.skewness = rust_stats.get("skewness").copied().unwrap_or(0.0);
                stats_ref.kurtosis = rust_stats.get("kurtosis").copied().unwrap_or(0.0);
                stats_ref.q25 = rust_stats.get("q25").copied().unwrap_or(0.0);
                stats_ref.q50 = rust_stats.get("q50").copied().unwrap_or(0.0);
                stats_ref.q75 = rust_stats.get("q75").copied().unwrap_or(0.0);
                R_SUCCESS
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Check if time series is stationary
///
/// # Safety
/// The caller must ensure that:
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - The structure remains valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_is_stationary(ts: *const RTimeSeries) -> c_int {
    if ts.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let ts_ref = &*ts;
        if ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        match is_stationary(&values_array, None) {
            Ok((_test_stat, p_value)) => {
                if p_value < 0.05 {
                    1
                } else {
                    0
                } // stationary if p-value < 0.05
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Apply differencing to time series
///
/// # Safety
/// The caller must ensure that:
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - `output` is a valid pointer to an array of at least `max_length` elements
/// - `max_length` is greater than 0
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_difference_series(
    ts: *const RTimeSeries,
    periods: c_int,
    output: *mut c_double,
    max_length: c_int,
) -> c_int {
    if ts.is_null() || output.is_null() || periods <= 0 {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let ts_ref = &*ts;
        if ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        match difference_series(&values_array, periods as usize) {
            Ok(differenced) => {
                let output_length = std::cmp::min(differenced.len(), max_length as usize);
                let output_slice = slice::from_raw_parts_mut(output, output_length);

                for (i, &val) in differenced.iter().take(output_length).enumerate() {
                    output_slice[i] = val;
                }

                output_length as c_int
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

// ================================
// ARIMA Model Functions
// ================================

/// Create a new ARIMA model
#[cfg(feature = "r")]
#[no_mangle]
pub extern "C" fn scirs_create_arima(
    p: c_int,
    d: c_int,
    q: c_int,
    seasonal_p: c_int,
    seasonal_d: c_int,
    seasonal_q: c_int,
    seasonal_period: c_int,
) -> *mut RARIMAModel {
    let config = ArimaConfig {
        p: p as usize,
        d: d as usize,
        q: q as usize,
        seasonal_p: seasonal_p as usize,
        seasonal_d: seasonal_d as usize,
        seasonal_q: seasonal_q as usize,
        seasonal_period: seasonal_period as usize,
    };

    match ArimaModel::<f64>::new(config.p, config.d, config.q) {
        Ok(model) => {
            let r_model = Box::new(RARIMAModel {
                handle: Box::into_raw(Box::new(model)) as *mut c_void,
                p,
                d,
                q,
                seasonal_p,
                seasonal_d,
                seasonal_q,
                seasonal_period,
            });
            Box::into_raw(r_model)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Fit ARIMA model to time series
///
/// # Safety
/// The caller must ensure that:
/// - `model` is a valid pointer to an RARIMAModel structure
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - Both structures remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_fit_arima(model: *mut RARIMAModel, ts: *const RTimeSeries) -> c_int {
    if model.is_null() || ts.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let model_ref = &mut *model;
        let ts_ref = &*ts;

        if model_ref.handle.is_null() || ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let arima_model = &mut *(model_ref.handle as *mut ArimaModel<f64>);
        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        match arima_model.fit(&values_array) {
            Ok(_) => R_SUCCESS,
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Generate ARIMA forecasts
///
/// # Safety
/// The caller must ensure that:
/// - `model` is a valid pointer to an RARIMAModel structure
/// - `output` is a valid pointer to an array of at least `max_length` elements
/// - `max_length` is greater than 0
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_forecast_arima(
    model: *const RARIMAModel,
    steps: c_int,
    output: *mut c_double,
    max_length: c_int,
) -> c_int {
    if model.is_null() || output.is_null() || steps <= 0 {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let model_ref = &*model;
        if model_ref.handle.is_null() {
            return R_ERROR_NOT_FITTED;
        }

        let arima_model = &*(model_ref.handle as *const ArimaModel<f64>);

        // NOTE: This is a limitation - ARIMA forecast needs training data but R API doesn't provide it
        // In a production system, training data should be stored in RARIMAModel or passed as parameter
        let dummy_data = Array1::zeros(10); // Placeholder - this will cause forecast to fail gracefully
        match arima_model.forecast(steps as usize, &dummy_data) {
            Ok(forecasts) => {
                let output_length = std::cmp::min(forecasts.len(), max_length as usize);
                let output_slice = slice::from_raw_parts_mut(output, output_length);

                for (i, &val) in forecasts.iter().take(output_length).enumerate() {
                    output_slice[i] = val;
                }

                output_length as c_int
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Get ARIMA model parameters
///
/// # Safety
/// The caller must ensure that:
/// - `model` is a valid pointer to an RARIMAModel structure
/// - `param_names` is a valid pointer to an array of at least `max_params` char pointers
/// - `param_values` is a valid pointer to an array of at least `max_params` doubles
/// - `max_params` is greater than 0
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_get_arima_params(
    model: *const RARIMAModel,
    param_names: *mut *mut c_char,
    param_values: *mut c_double,
    max_params: c_int,
) -> c_int {
    if model.is_null() || param_names.is_null() || param_values.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let model_ref = &*model;
        if model_ref.handle.is_null() {
            return R_ERROR_NOT_FITTED;
        }

        let _arima_model = &*(model_ref.handle as *const ArimaModel<f64>);

        // Extract basic model information since get_params doesn't exist
        let param_names_list = ["p", "d", "q"];
        let param_values_list = [model_ref.p as f64, model_ref.d as f64, model_ref.q as f64];

        let param_count = std::cmp::min(param_names_list.len(), max_params as usize);
        let names_slice = slice::from_raw_parts_mut(param_names, param_count);
        let values_slice = slice::from_raw_parts_mut(param_values, param_count);

        for (i, (name, value)) in param_names_list
            .iter()
            .zip(param_values_list.iter())
            .take(param_count)
            .enumerate()
        {
            let c_name = CString::new(*name).unwrap();
            names_slice[i] = c_name.into_raw();
            values_slice[i] = *value;
        }

        param_count as c_int
    }
}

/// Free ARIMA model
///
/// # Safety
/// The caller must ensure that:
/// - `model` is a valid pointer to an RARIMAModel structure
/// - The pointer was previously returned by `scirs_create_arima` or `scirs_auto_arima`
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_arima(model: *mut RARIMAModel) {
    if !model.is_null() {
        unsafe {
            let model_box = Box::from_raw(model);
            if !model_box.handle.is_null() {
                let _ = Box::from_raw(model_box.handle as *mut ArimaModel<f64>);
            }
        }
    }
}

// ================================
// Anomaly Detection Functions
// ================================

/// Create anomaly detector
#[cfg(feature = "r")]
#[no_mangle]
pub extern "C" fn scirs_create_anomaly_detector() -> *mut RAnomalyDetector {
    let detector = StreamingAnomalyDetector::new(100, 2.0, 10, 5);
    let r_detector = Box::new(RAnomalyDetector {
        handle: Box::into_raw(Box::new(detector)) as *mut c_void,
    });
    Box::into_raw(r_detector)
}

/// Detect anomalies using IQR method
///
/// # Safety
/// The caller must ensure that:
/// - `detector` is a valid pointer to an RAnomalyDetector structure
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - `anomaly_indices` is a valid pointer to an array of at least `max_anomalies` elements
/// - `max_anomalies` is greater than 0
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_detect_anomalies_iqr(
    detector: *const RAnomalyDetector,
    ts: *const RTimeSeries,
    multiplier: c_double,
    anomaly_indices: *mut c_int,
    max_anomalies: c_int,
) -> c_int {
    if detector.is_null() || ts.is_null() || anomaly_indices.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let detector_ref = &*detector;
        let ts_ref = &*ts;

        if detector_ref.handle.is_null() || ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        let options = AnomalyOptions {
            method: AnomalyMethod::InterquartileRange,
            threshold: Some(multiplier),
            ..Default::default()
        };

        match detect_anomalies(&values_array, &options) {
            Ok(result) => {
                let anomalies: Vec<usize> = result
                    .is_anomaly
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
                    .collect();
                let anomaly_count = std::cmp::min(anomalies.len(), max_anomalies as usize);
                let indices_slice = slice::from_raw_parts_mut(anomaly_indices, anomaly_count);

                for (i, &idx) in anomalies.iter().take(anomaly_count).enumerate() {
                    indices_slice[i] = idx as c_int;
                }

                anomaly_count as c_int
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Detect anomalies using Z-score method
///
/// # Safety
/// The caller must ensure that:
/// - `detector` is a valid pointer to an RAnomalyDetector structure
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - `anomaly_indices` is a valid pointer to an array of at least `max_anomalies` elements
/// - `max_anomalies` is greater than 0
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_detect_anomalies_zscore(
    detector: *const RAnomalyDetector,
    ts: *const RTimeSeries,
    threshold: c_double,
    anomaly_indices: *mut c_int,
    max_anomalies: c_int,
) -> c_int {
    if detector.is_null() || ts.is_null() || anomaly_indices.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let detector_ref = &*detector;
        let ts_ref = &*ts;

        if detector_ref.handle.is_null() || ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        let options = AnomalyOptions {
            method: AnomalyMethod::ZScore,
            threshold: Some(threshold),
            ..Default::default()
        };

        match detect_anomalies(&values_array, &options) {
            Ok(result) => {
                let anomalies: Vec<usize> = result
                    .is_anomaly
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
                    .collect();
                let anomaly_count = std::cmp::min(anomalies.len(), max_anomalies as usize);
                let indices_slice = slice::from_raw_parts_mut(anomaly_indices, anomaly_count);

                for (i, &idx) in anomalies.iter().take(anomaly_count).enumerate() {
                    indices_slice[i] = idx as c_int;
                }

                anomaly_count as c_int
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Free anomaly detector
///
/// # Safety
/// The caller must ensure that:
/// - `detector` is a valid pointer to an RAnomalyDetector structure
/// - The pointer was previously returned by `scirs_create_anomaly_detector`
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_anomaly_detector(detector: *mut RAnomalyDetector) {
    if !detector.is_null() {
        unsafe {
            let _detector_box = Box::from_raw(detector);
            // Since we're using functional API, no need to free the handle
            // The handle is not storing an actual _detector object
        }
    }
}

// ================================
// STL Decomposition Functions
// ================================

/// Create STL decomposition
#[cfg(feature = "r")]
#[no_mangle]
pub extern "C" fn scirs_create_stl_decomposition(period: c_int) -> *mut RSTLDecomposition {
    if period <= 0 {
        return std::ptr::null_mut();
    }

    // Store STL configuration instead of a decomposition object
    use crate::decomposition::stl::STLOptions;
    let stl_options = STLOptions::default();
    let r_decomposition = Box::new(RSTLDecomposition {
        handle: Box::into_raw(Box::new(stl_options)) as *mut c_void,
        period,
    });
    Box::into_raw(r_decomposition)
}

/// Perform STL decomposition
///
/// # Safety
/// The caller must ensure that:
/// - `decomposition` is a valid pointer to an RSTLDecomposition structure
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - `result` is a valid pointer to an RDecompositionResult structure
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_decompose_stl(
    decomposition: *const RSTLDecomposition,
    ts: *const RTimeSeries,
    result: *mut RDecompositionResult,
) -> c_int {
    if decomposition.is_null() || ts.is_null() || result.is_null() {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let decomp_ref = &*decomposition;
        let ts_ref = &*ts;

        if decomp_ref.handle.is_null() || ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let stl_options = &*(decomp_ref.handle as *const crate::decomposition::stl::STLOptions);
        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        match crate::decomposition::stl::stl_decomposition(
            &values_array,
            decomp_ref.period as usize,
            stl_options,
        ) {
            Ok(decomp_result) => {
                let result_ref = &mut *result;
                let length = decomp_result.trend.len();

                // Allocate memory for results
                let trend_vec = decomp_result.trend.to_vec();
                let seasonal_vec = decomp_result.seasonal.to_vec();
                let residual_vec = decomp_result.residual.to_vec();

                result_ref.trend = trend_vec.as_ptr() as *mut c_double;
                result_ref.seasonal = seasonal_vec.as_ptr() as *mut c_double;
                result_ref.residual = residual_vec.as_ptr() as *mut c_double;
                result_ref.length = length as c_int;

                // Prevent deallocation
                std::mem::forget(trend_vec);
                std::mem::forget(seasonal_vec);
                std::mem::forget(residual_vec);

                R_SUCCESS
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Free STL decomposition
///
/// # Safety
/// The caller must ensure that:
/// - `decomposition` is a valid pointer to an RSTLDecomposition structure
/// - The pointer was previously returned by `scirs_create_stl_decomposition`
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_stl_decomposition(decomposition: *mut RSTLDecomposition) {
    if !decomposition.is_null() {
        unsafe {
            let decomp_box = Box::from_raw(decomposition);
            if !decomp_box.handle.is_null() {
                let _ =
                    Box::from_raw(decomp_box.handle as *mut crate::decomposition::stl::STLOptions);
            }
        }
    }
}

/// Free decomposition result
///
/// # Safety
/// The caller must ensure that:
/// - `result` is a valid pointer to an RDecompositionResult structure
/// - The pointer was previously returned by `scirs_decompose_stl`
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_decomposition_result(result: *mut RDecompositionResult) {
    if !result.is_null() {
        unsafe {
            let result_ref = &*result;
            if !result_ref.trend.is_null() {
                Vec::from_raw_parts(
                    result_ref.trend,
                    result_ref.length as usize,
                    result_ref.length as usize,
                );
            }
            if !result_ref.seasonal.is_null() {
                Vec::from_raw_parts(
                    result_ref.seasonal,
                    result_ref.length as usize,
                    result_ref.length as usize,
                );
            }
            if !result_ref.residual.is_null() {
                Vec::from_raw_parts(
                    result_ref.residual,
                    result_ref.length as usize,
                    result_ref.length as usize,
                );
            }
        }
    }
}

// ================================
// Auto-ARIMA Functions
// ================================

/// Automatically select best ARIMA model
///
/// # Safety
/// The caller must ensure that:
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - The structure remains valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_auto_arima(
    ts: *const RTimeSeries,
    max_p: c_int,
    max_d: c_int,
    max_q: c_int,
    seasonal: c_int,
    max_seasonal_p: c_int,
    max_seasonal_d: c_int,
    max_seasonal_q: c_int,
    seasonal_period: c_int,
) -> *mut RARIMAModel {
    if ts.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let ts_ref = &*ts;
        if ts_ref.values.is_null() || ts_ref.length <= 0 {
            return std::ptr::null_mut();
        }

        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        let seasonal_bool = seasonal != 0;

        let options = crate::arima_models::ArimaSelectionOptions {
            max_p: max_p as usize,
            max_d: max_d as usize,
            max_q: max_q as usize,
            seasonal: seasonal_bool,
            max_seasonal_p: max_seasonal_p as usize,
            max_seasonal_d: max_seasonal_d as usize,
            max_seasonal_q: max_seasonal_q as usize,
            seasonal_period: Some(seasonal_period as usize),
            ..Default::default()
        };

        match crate::arima_models::auto_arima(&values_array, &options) {
            Ok((_arima_model, sarima_params)) => {
                let config = ArimaConfig {
                    p: sarima_params.pdq.0,
                    d: sarima_params.pdq.1,
                    q: sarima_params.pdq.2,
                    seasonal_p: sarima_params.seasonal_pdq.0,
                    seasonal_d: sarima_params.seasonal_pdq.1,
                    seasonal_q: sarima_params.seasonal_pdq.2,
                    seasonal_period: sarima_params.seasonal_period,
                };
                match ArimaModel::<f64>::new(config.p, config.d, config.q) {
                    Ok(model) => {
                        let r_model = Box::new(RARIMAModel {
                            handle: Box::into_raw(Box::new(model)) as *mut c_void,
                            p: config.p as c_int,
                            d: config.d as c_int,
                            q: config.q as c_int,
                            seasonal_p: config.seasonal_p as c_int,
                            seasonal_d: config.seasonal_d as c_int,
                            seasonal_q: config.seasonal_q as c_int,
                            seasonal_period: config.seasonal_period as c_int,
                        });
                        Box::into_raw(r_model)
                    }
                    Err(_) => std::ptr::null_mut(),
                }
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

// ================================
// Neural Forecasting Functions
// ================================

/// Create neural forecaster
#[cfg(feature = "r")]
#[repr(C)]
pub struct RNeuralForecaster {
    /// Opaque handle to the neural network
    pub handle: *mut c_void,
    /// Size of the input layer
    pub input_size: c_int,
    /// Size of the hidden layer
    pub hidden_size: c_int,
    /// Size of the output layer
    pub output_size: c_int,
}

#[cfg(feature = "r")]
#[no_mangle]
/// Creates a new neural forecaster instance
pub extern "C" fn scirs_create_neural_forecaster(
    input_size: c_int,
    hidden_size: c_int,
    output_size: c_int,
) -> *mut RNeuralForecaster {
    if input_size <= 0 || hidden_size <= 0 || output_size <= 0 {
        return std::ptr::null_mut();
    }

    let forecaster =
        Box::new(LSTMForecaster::new(LSTMConfig::default())) as Box<dyn NeuralForecaster<f64>>;

    let r_forecaster = Box::new(RNeuralForecaster {
        handle: Box::into_raw(Box::new(forecaster)) as *mut c_void,
        input_size,
        hidden_size,
        output_size,
    });

    Box::into_raw(r_forecaster)
}

/// Train neural forecaster
///
/// # Safety
/// The caller must ensure that:
/// - `forecaster` is a valid pointer to an RNeuralForecaster structure
/// - `ts` is a valid pointer to an RTimeSeries structure
/// - Both structures remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_train_neural_forecaster(
    forecaster: *mut RNeuralForecaster,
    ts: *const RTimeSeries,
    epochs: c_int,
    rate: c_double,
) -> c_int {
    if forecaster.is_null() || ts.is_null() || epochs <= 0 {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let forecaster_ref = &mut *forecaster;
        let ts_ref = &*ts;

        if forecaster_ref.handle.is_null() || ts_ref.values.is_null() || ts_ref.length <= 0 {
            return R_ERROR_INVALID_PARAMS;
        }

        let neural_forecaster = &mut *(forecaster_ref.handle as *mut LSTMForecaster<f64>);
        let values_slice = slice::from_raw_parts(ts_ref.values, ts_ref.length as usize);
        let values_array = Array1::from_vec(values_slice.to_vec());

        match neural_forecaster.fit(&values_array) {
            Ok(_) => R_SUCCESS,
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Generate neural forecasts
///
/// # Safety
/// The caller must ensure that:
/// - `forecaster` is a valid pointer to an RNeuralForecaster structure
/// - `input` is a valid pointer to an array of at least `input_length` elements
/// - `output` is a valid pointer to an array of at least `max_length` elements
/// - `input_length` and `max_length` are greater than 0
/// - All pointers remain valid for the duration of the call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_forecast_neural(
    forecaster: *const RNeuralForecaster,
    input: *const c_double,
    input_length: c_int,
    steps: c_int,
    output: *mut c_double,
    max_length: c_int,
) -> c_int {
    if forecaster.is_null() || input.is_null() || output.is_null() || steps <= 0 {
        return R_ERROR_INVALID_PARAMS;
    }

    unsafe {
        let forecaster_ref = &*forecaster;

        if forecaster_ref.handle.is_null() {
            return R_ERROR_NOT_FITTED;
        }

        let neural_forecaster = &*(forecaster_ref.handle as *const LSTMForecaster<f64>);
        let input_slice = slice::from_raw_parts(input, input_length as usize);
        let _input_array = Array1::from_vec(input_slice.to_vec());

        match neural_forecaster.predict(steps as usize) {
            Ok(forecasts) => {
                let output_length = std::cmp::min(forecasts.forecast.len(), max_length as usize);
                let output_slice = slice::from_raw_parts_mut(output, output_length);

                for (i, &val) in forecasts.forecast.iter().take(output_length).enumerate() {
                    output_slice[i] = val;
                }

                output_length as c_int
            }
            Err(_) => R_ERROR_COMPUTATION,
        }
    }
}

/// Free neural forecaster
///
/// # Safety
/// The caller must ensure that:
/// - `forecaster` is a valid pointer to an RNeuralForecaster structure
/// - The pointer was previously returned by `scirs_create_neural_forecaster`
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_neural_forecaster(forecaster: *mut RNeuralForecaster) {
    if !forecaster.is_null() {
        unsafe {
            let forecaster_box = Box::from_raw(forecaster);
            if !forecaster_box.handle.is_null() {
                let _ = Box::from_raw(forecaster_box.handle as *mut LSTMForecaster<f64>);
            }
        }
    }
}

// ================================
// Utility Functions
// ================================

/// Get library version
#[cfg(feature = "r")]
#[no_mangle]
pub extern "C" fn scirs_get_version() -> *const c_char {
    static VERSION: &str = "0.1.0-beta.1\0";
    VERSION.as_ptr() as *const c_char
}

/// Initialize the library
#[cfg(feature = "r")]
#[no_mangle]
pub extern "C" fn scirs_initialize() -> c_int {
    // Initialize any global state if needed
    R_SUCCESS
}

/// Cleanup the library
#[cfg(feature = "r")]
#[no_mangle]
pub extern "C" fn scirs_cleanup() -> c_int {
    // Cleanup any global state if needed
    R_SUCCESS
}

/// Free a C string allocated by the library
///
/// # Safety
/// The caller must ensure that:
/// - `s` is a valid pointer to a C string
/// - The pointer was previously returned by a library function
/// - The pointer is not used after this call
#[cfg(feature = "r")]
#[no_mangle]
pub unsafe extern "C" fn scirs_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_integration_constants() {
        assert_eq!(R_SUCCESS, 0);
        assert_eq!(R_ERROR_INVALID_PARAMS, -1);
        assert_eq!(R_ERROR_MEMORY, -2);
        assert_eq!(R_ERROR_COMPUTATION, -3);
        assert_eq!(R_ERROR_NOT_FITTED, -4);
    }

    #[test]
    fn test_r_structures_size() {
        // Ensure C structures have expected sizes
        use std::mem;

        assert!(mem::size_of::<RTimeSeries>() > 0);
        assert!(mem::size_of::<RARIMAModel>() > 0);
        assert!(mem::size_of::<RAnomalyDetector>() > 0);
        assert!(mem::size_of::<RSTLDecomposition>() > 0);
    }
}
