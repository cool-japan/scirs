//! Time series datasets.
//!
//! This module provides access to common time series datasets, including:
//! - Electrocardiogram (ECG) dataset
//! - Stock market dataset
//! - Weather dataset
//!
//! These datasets are designed for testing time series analysis algorithms,
//! signal processing, and forecasting methods.

use crate::cache::{fetch_data, RegistryEntry};
use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

// Registry mapping files to their SHA256 hashes and URLs
lazy_static::lazy_static! {
    static ref REGISTRY: HashMap<&'static str, RegistryEntry> = {
        let mut registry = HashMap::new();

        // ECG dataset
        registry.insert("ecg.dat", RegistryEntry {
            sha256: "f20ad3365fb9b7f845d0e5c48b6fe67081377ee466c3a220b7f69f35c8958baf",
            url: "https://raw.githubusercontent.com/scipy/dataset-ecg/main/ecg.dat",
        });

        // Stock market dataset
        registry.insert("stock_market.csv", RegistryEntry {
            sha256: "e6d5392bd79e82e3f6d7fe171d8c2fafae84b1a4e9e95a532ec252caa3053dc9",
            url: "https://raw.githubusercontent.com/scirs/datasets/main/stock_market.csv",
        });

        // Weather dataset
        registry.insert("weather.csv", RegistryEntry {
            sha256: "f8bdaef6d968c1eddb0c0c7cf9c245b07d60ffe3a7d8e5ed8953f5750ee0f610",
            url: "https://raw.githubusercontent.com/scirs/datasets/main/weather.csv",
        });

        registry
    };
}

/// Load an electrocardiogram as an example for a 1-D signal.
///
/// The returned signal is a 5 minute long electrocardiogram (ECG), a medical
/// recording of the heart's electrical activity, sampled at 360 Hz.
///
/// # Returns
///
/// A Dataset containing:
/// - `data`: The electrocardiogram in millivolt (mV) sampled at 360 Hz (as a column vector)
/// - No target values
/// - Metadata including sampling rate
///
/// # Examples
///
/// ```
/// use scirs2_datasets::time_series::electrocardiogram;
///
/// let ecg = electrocardiogram().unwrap();
/// println!("ECG data shape: ({}, {})", ecg.n_samples(), ecg.n_features());
/// ```
pub fn electrocardiogram() -> Result<Dataset> {
    // Fetch the ECG data file
    let ecg_file = match fetch_data("ecg.dat", REGISTRY.get("ecg.dat")) {
        Ok(path) => path,
        Err(e) => {
            return Err(DatasetsError::LoadingError(format!(
                "Failed to fetch ECG data: {}",
                e
            )))
        }
    };

    // Read the file
    let ecg_data = match fs::read(ecg_file) {
        Ok(data) => data,
        Err(e) => {
            return Err(DatasetsError::LoadingError(format!(
                "Failed to read ECG data: {}",
                e
            )))
        }
    };

    // Parse the binary data - ECG data is 16-bit integers
    let mut ecg_values = Vec::with_capacity(ecg_data.len() / 2);
    let mut i = 0;
    while i < ecg_data.len() {
        if i + 1 < ecg_data.len() {
            let value = (ecg_data[i] as u16) | ((ecg_data[i + 1] as u16) << 8);
            ecg_values.push(value);
        }
        i += 2;
    }

    // Convert raw ADC output to mV: (ecg - adc_zero) / adc_gain
    // Following SciPy's conversion formula
    let ecg_values = ecg_values
        .into_iter()
        .map(|x| (x as f64 - 1024.0) / 200.0)
        .collect::<Vec<f64>>();

    let ecg_array = Array1::from_vec(ecg_values);

    // Get the length before converting to avoid borrow after move
    let len = ecg_array.len();

    // Convert the 1D array to a 2D column vector using reshape which should be safer
    let data = ecg_array.into_shape_with_order((len, 1)).unwrap();

    // Create the dataset
    let mut dataset = Dataset::new(data, None);
    dataset = dataset
        .with_feature_names(vec!["ecg".to_string()])
        .with_description("Electrocardiogram (ECG) data, 5 minutes sampled at 360 Hz".to_string())
        .with_metadata("sampling_rate", "360")
        .with_metadata("units", "mV")
        .with_metadata("duration", "5 minutes");

    Ok(dataset)
}

/// Stock market price data structure for parsing CSV
#[derive(Debug, Deserialize)]
struct StockPrice {
    date: String,
    open: f64,
    #[allow(dead_code)]
    high: f64,
    #[allow(dead_code)]
    low: f64,
    close: f64,
    #[allow(dead_code)]
    volume: f64,
    symbol: String,
}

/// Load stock market data for multiple companies.
///
/// This dataset contains historical daily price data for multiple companies,
/// which can be used for financial time series analysis.
///
/// # Parameters
///
/// * `returns` - If true, returns daily price changes (close - open) instead of absolute prices.
///
/// # Returns
///
/// A Dataset containing:
/// - `data`: Price data for multiple stocks over time
/// - No target values
/// - Feature names corresponding to stock symbols
/// - Metadata including date range and symbols
///
/// # Examples
///
/// ```ignore
/// use scirs2_datasets::time_series::stock_market;
///
/// let stock_data = stock_market(true).unwrap(); // Get price changes
/// println!("Stock data shape: ({}, {})", stock_data.n_samples(), stock_data.n_features());
/// ```
pub fn stock_market(returns: bool) -> Result<Dataset> {
    // Fetch the stock market data file
    let stock_file = match fetch_data("stock_market.csv", REGISTRY.get("stock_market.csv")) {
        Ok(path) => path,
        Err(e) => {
            return Err(DatasetsError::LoadingError(format!(
                "Failed to fetch stock market data: {}",
                e
            )))
        }
    };

    // Read and parse the CSV file
    let file_content = match fs::read_to_string(&stock_file) {
        Ok(content) => content,
        Err(e) => {
            return Err(DatasetsError::LoadingError(format!(
                "Failed to read stock market data: {}",
                e
            )))
        }
    };

    let mut reader = csv::Reader::from_reader(file_content.as_bytes());
    let records: Result<Vec<StockPrice>> = reader
        .deserialize()
        .map(|result| {
            result.map_err(|e| DatasetsError::LoadingError(format!("CSV parsing error: {}", e)))
        })
        .collect();

    let records = records?;
    if records.is_empty() {
        return Err(DatasetsError::LoadingError(
            "Stock market data is empty".to_string(),
        ));
    }

    // Extract unique symbols and dates
    let mut symbols = Vec::new();
    let mut dates = Vec::new();
    for record in &records {
        if !symbols.contains(&record.symbol) {
            symbols.push(record.symbol.clone());
        }
        if !dates.contains(&record.date) {
            dates.push(record.date.clone());
        }
    }

    symbols.sort();
    dates.sort();

    // Create a mapping of (date, symbol) to price data
    let mut date_symbol_map = HashMap::new();
    for record in &records {
        date_symbol_map.insert((record.date.clone(), record.symbol.clone()), record);
    }

    // Create the data matrix (dates x symbols)
    let mut data = Array2::zeros((dates.len(), symbols.len()));

    for (i, date) in dates.iter().enumerate() {
        for (j, symbol) in symbols.iter().enumerate() {
            if let Some(record) = date_symbol_map.get(&(date.clone(), symbol.clone())) {
                data[[i, j]] = if returns {
                    record.close - record.open
                } else {
                    record.close
                };
            }
        }
    }

    // Create the dataset
    let mut dataset = Dataset::new(data, None);
    dataset = dataset
        .with_feature_names(symbols.clone())
        .with_description(format!(
            "Stock market data for {} companies from {} to {}",
            symbols.len(),
            dates.first().unwrap_or(&"unknown".to_string()),
            dates.last().unwrap_or(&"unknown".to_string())
        ))
        .with_metadata("n_symbols", &symbols.len().to_string())
        .with_metadata(
            "start_date",
            dates.first().unwrap_or(&"unknown".to_string()),
        )
        .with_metadata("end_date", dates.last().unwrap_or(&"unknown".to_string()))
        .with_metadata("data_type", if returns { "returns" } else { "prices" });

    Ok(dataset)
}

/// Weather observation data structure for parsing CSV
#[derive(Debug, Deserialize)]
struct WeatherObservation {
    date: String,
    temperature: f64,
    humidity: f64,
    pressure: f64,
    wind_speed: f64,
    precipitation: f64,
    location: String,
}

/// Load weather time series data from multiple locations.
///
/// This dataset contains daily weather measurements (temperature, humidity, pressure,
/// wind speed, and precipitation) for multiple locations.
///
/// # Parameters
///
/// * `feature` - Which weather feature to use as the primary data. Options are:
///   "temperature", "humidity", "pressure", "wind_speed", or "precipitation".
///   If None, returns all features.
///
/// # Returns
///
/// A Dataset containing:
/// - `data`: Weather data over time (dates x locations or dates x features x locations)
/// - No target values
/// - Feature names corresponding to locations or weather measurements
/// - Metadata including date range and locations
///
/// # Examples
///
/// ```ignore
/// use scirs2_datasets::time_series::weather;
///
/// // Get temperature data for all locations
/// let temp_data = weather(Some("temperature")).unwrap();
/// println!("Temperature data shape: ({}, {})", temp_data.n_samples(), temp_data.n_features());
///
/// // Get all weather features
/// let all_weather = weather(None).unwrap();
/// println!("All weather data shape: ({}, {})", all_weather.n_samples(), all_weather.n_features());
/// ```
pub fn weather(feature: Option<&str>) -> Result<Dataset> {
    // Validate feature parameter
    let valid_features = vec![
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "precipitation",
    ];

    if let Some(f) = feature {
        if !valid_features.contains(&f) {
            return Err(DatasetsError::InvalidFormat(format!(
                "Invalid feature: {}. Valid features are: {:?}",
                f, valid_features
            )));
        }
    }

    // Fetch the weather data file
    let weather_file = match fetch_data("weather.csv", REGISTRY.get("weather.csv")) {
        Ok(path) => path,
        Err(e) => {
            return Err(DatasetsError::LoadingError(format!(
                "Failed to fetch weather data: {}",
                e
            )))
        }
    };

    // Read and parse the CSV file
    let file_content = match fs::read_to_string(&weather_file) {
        Ok(content) => content,
        Err(e) => {
            return Err(DatasetsError::LoadingError(format!(
                "Failed to read weather data: {}",
                e
            )))
        }
    };

    let mut reader = csv::Reader::from_reader(file_content.as_bytes());
    let records: Result<Vec<WeatherObservation>> = reader
        .deserialize()
        .map(|result| {
            result.map_err(|e| DatasetsError::LoadingError(format!("CSV parsing error: {}", e)))
        })
        .collect();

    let records = records?;
    if records.is_empty() {
        return Err(DatasetsError::LoadingError(
            "Weather data is empty".to_string(),
        ));
    }

    // Extract unique locations and dates
    let mut locations = Vec::new();
    let mut dates = Vec::new();
    for record in &records {
        if !locations.contains(&record.location) {
            locations.push(record.location.clone());
        }
        if !dates.contains(&record.date) {
            dates.push(record.date.clone());
        }
    }

    locations.sort();
    dates.sort();

    // Create a mapping of (date, location) to weather data
    let mut date_location_map = HashMap::new();
    for record in &records {
        date_location_map.insert((record.date.clone(), record.location.clone()), record);
    }

    let mut dataset = match feature {
        Some(feat) => {
            // Single feature mode - create a 2D matrix (dates x locations)
            let mut data = Array2::zeros((dates.len(), locations.len()));

            for (i, date) in dates.iter().enumerate() {
                for (j, location) in locations.iter().enumerate() {
                    if let Some(record) = date_location_map.get(&(date.clone(), location.clone())) {
                        data[[i, j]] = match feat {
                            "temperature" => record.temperature,
                            "humidity" => record.humidity,
                            "pressure" => record.pressure,
                            "wind_speed" => record.wind_speed,
                            "precipitation" => record.precipitation,
                            _ => 0.0, // Should never happen due to validation above
                        };
                    }
                }
            }

            // Create the dataset
            let mut ds = Dataset::new(data, None);

            // Feature names are location names in this case
            ds = ds
                .with_feature_names(locations.clone())
                .with_description(format!(
                    "Weather {} data for {} locations from {} to {}",
                    feat,
                    locations.len(),
                    dates.first().unwrap_or(&"unknown".to_string()),
                    dates.last().unwrap_or(&"unknown".to_string())
                ))
                .with_metadata("feature", feat)
                .with_metadata("n_locations", &locations.len().to_string())
                .with_metadata(
                    "start_date",
                    dates.first().unwrap_or(&"unknown".to_string()),
                )
                .with_metadata("end_date", dates.last().unwrap_or(&"unknown".to_string()));

            ds
        }
        None => {
            // All features mode - create a 2D matrix (dates x (features*locations))
            // Each location will have multiple columns, one for each feature
            let n_features = valid_features.len();
            let mut data = Array2::zeros((dates.len(), n_features * locations.len()));

            for (i, date) in dates.iter().enumerate() {
                for (j, location) in locations.iter().enumerate() {
                    if let Some(record) = date_location_map.get(&(date.clone(), location.clone())) {
                        // Calculate base column index for this location
                        let base_col = j * n_features;

                        // Fill in all features for this location and date
                        data[[i, base_col]] = record.temperature;
                        data[[i, base_col + 1]] = record.humidity;
                        data[[i, base_col + 2]] = record.pressure;
                        data[[i, base_col + 3]] = record.wind_speed;
                        data[[i, base_col + 4]] = record.precipitation;
                    }
                }
            }

            // Create feature names: for each location, add each feature
            let mut feature_names = Vec::with_capacity(n_features * locations.len());
            for location in &locations {
                for feat in &valid_features {
                    feature_names.push(format!("{}_{}", location, feat));
                }
            }

            // Create the dataset
            let mut ds = Dataset::new(data, None);
            ds = ds
                .with_feature_names(feature_names)
                .with_description(format!(
                    "Weather data (all features) for {} locations from {} to {}",
                    locations.len(),
                    dates.first().unwrap_or(&"unknown".to_string()),
                    dates.last().unwrap_or(&"unknown".to_string())
                ))
                .with_metadata("features", &valid_features.join(","))
                .with_metadata("n_locations", &locations.len().to_string())
                .with_metadata(
                    "start_date",
                    dates.first().unwrap_or(&"unknown".to_string()),
                )
                .with_metadata("end_date", dates.last().unwrap_or(&"unknown".to_string()));

            ds
        }
    };

    // Add locations metadata
    dataset = dataset.with_metadata("locations", &locations.join(","));

    Ok(dataset)
}

// The fetch_data function is now provided by the cache module
