//! IoT sensor data analysis for time series
//!
//! This module provides specialized functionality for analyzing time series data
//! from Internet of Things (IoT) sensors including temperature, humidity, motion,
//! GPS, accelerometer, and other sensor data streams.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2, ArrayStatCompat};
use scirs2_core::validation::check_positive;
use statrs::statistics::Statistics;
use std::collections::HashMap;

/// IoT sensor data types
#[derive(Debug, Clone)]
pub enum SensorType {
    /// Temperature sensor (°C)
    Temperature,
    /// Humidity sensor (%)
    Humidity,
    /// Pressure sensor (hPa)
    Pressure,
    /// Light intensity sensor (lux)
    Light,
    /// Motion/PIR sensor (binary)
    Motion,
    /// Accelerometer (3-axis: x, y, z in m/s²)
    Accelerometer,
    /// GPS coordinates (latitude, longitude)
    GPS,
    /// Sound level sensor (dB)
    Sound,
    /// Air quality sensor (AQI or ppm)
    AirQuality,
    /// Energy consumption (W or kWh)
    Energy,
}

/// Sensor data quality assessment
#[derive(Debug, Clone)]
pub struct DataQuality {
    /// Missing data percentage
    pub missing_percentage: f64,
    /// Outlier percentage
    pub outlier_percentage: f64,
    /// Signal-to-noise ratio
    pub snr: f64,
    /// Data consistency score
    pub consistency_score: f64,
}

/// Environmental sensor data analysis
pub struct EnvironmentalSensorAnalysis {
    /// Temperature readings
    pub temperature: Option<Array1<f64>>,
    /// Humidity readings
    pub humidity: Option<Array1<f64>>,
    /// Pressure readings
    pub pressure: Option<Array1<f64>>,
    /// Light readings
    pub light: Option<Array1<f64>>,
    /// Time stamps
    pub timestamps: Array1<i64>,
    /// Sampling interval in seconds
    pub sampling_interval: f64,
}

impl EnvironmentalSensorAnalysis {
    /// Create new environmental sensor analysis
    pub fn new(_timestamps: Array1<i64>, samplinginterval: f64) -> Result<Self> {
        check_positive(samplinginterval, "sampling_interval")?;

        Ok(Self {
            temperature: None,
            humidity: None,
            pressure: None,
            light: None,
            timestamps: _timestamps,
            sampling_interval: samplinginterval,
        })
    }

    /// Add temperature data
    pub fn with_temperature(mut self, data: Array1<f64>) -> Result<Self> {
        if data.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Temperature data contains non-finite values".to_string(),
            ));
        }
        if data.len() != self.timestamps.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Temperature data length must match timestamps".to_string(),
            ));
        }
        self.temperature = Some(data);
        Ok(self)
    }

    /// Add humidity data
    pub fn with_humidity(mut self, data: Array1<f64>) -> Result<Self> {
        if data.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Humidity data contains non-finite values".to_string(),
            ));
        }

        // Validate humidity range (0-100%)
        if data.iter().any(|&x| !(0.0..=100.0).contains(&x)) {
            return Err(TimeSeriesError::InvalidInput(
                "Humidity values must be between 0 and 100%".to_string(),
            ));
        }

        self.humidity = Some(data);
        Ok(self)
    }

    /// Detect sensor malfunctions using multiple criteria
    pub fn detect_sensor_malfunctions(&self) -> Result<HashMap<String, Vec<usize>>> {
        let mut malfunctions = HashMap::new();

        // Temperature sensor malfunction detection
        if let Some(ref temp_data) = self.temperature {
            let mut temp_issues = Vec::new();

            // Stuck sensor (same value for extended period)
            let mut consecutive_count = 1;
            for i in 1..temp_data.len() {
                if (temp_data[i] - temp_data[i - 1]).abs() < 0.01 {
                    consecutive_count += 1;
                } else {
                    if consecutive_count > 20 {
                        // 20 consecutive identical readings
                        for j in (i - consecutive_count)..i {
                            temp_issues.push(j);
                        }
                    }
                    consecutive_count = 1;
                }
            }

            // Impossible values (temperature outside reasonable range)
            for (i, &temp) in temp_data.iter().enumerate() {
                if !(-50.0..=100.0).contains(&temp) {
                    temp_issues.push(i);
                }
            }

            // Sudden jumps (> 10°C change in one reading)
            for i in 1..temp_data.len() {
                if (temp_data[i] - temp_data[i - 1]).abs() > 10.0 {
                    temp_issues.push(i);
                }
            }

            malfunctions.insert("Temperature".to_string(), temp_issues);
        }

        // Humidity sensor malfunction detection
        if let Some(ref humidity_data) = self.humidity {
            let mut humidity_issues = Vec::new();

            // Stuck at 0% or 100%
            for (i, &humidity) in humidity_data.iter().enumerate() {
                if humidity == 0.0 || humidity == 100.0 {
                    humidity_issues.push(i);
                }
            }

            // Sudden changes > 20%
            for i in 1..humidity_data.len() {
                if (humidity_data[i] - humidity_data[i - 1]).abs() > 20.0 {
                    humidity_issues.push(i);
                }
            }

            malfunctions.insert("Humidity".to_string(), humidity_issues);
        }

        Ok(malfunctions)
    }

    /// Calculate comfort index from temperature and humidity
    pub fn comfort_index(&self) -> Result<Array1<f64>> {
        let temp_data = self.temperature.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidInput("Temperature data required".to_string())
        })?;
        let humidity_data = self
            .humidity
            .as_ref()
            .ok_or_else(|| TimeSeriesError::InvalidInput("Humidity data required".to_string()))?;

        let mut comfort = Array1::zeros(temp_data.len());

        for i in 0..comfort.len() {
            let temp = temp_data[i];
            let rh = humidity_data[i];

            // Heat Index calculation (simplified)
            let heat_index = if temp >= 27.0 && rh >= 40.0 {
                -42.379 + 2.04901523 * temp + 10.14333127 * rh
                    - 0.22475541 * temp * rh
                    - 0.00683783 * temp * temp
                    - 0.05481717 * rh * rh
                    + 0.00122874 * temp * temp * rh
                    + 0.00085282 * temp * rh * rh
                    - 0.00000199 * temp * temp * rh * rh
            } else {
                temp
            };

            // Comfort score (0-100, higher is more comfortable)
            comfort[i] = if heat_index <= 27.0 && (30.0..=60.0).contains(&rh) {
                100.0 - (heat_index - 22.0).abs() * 5.0 - (rh - 45.0).abs() * 0.5
            } else {
                50.0 - (heat_index - 22.0).abs() * 2.0 - (rh - 45.0).abs() * 0.3
            }
            .clamp(0.0, 100.0);
        }

        Ok(comfort)
    }

    /// Energy optimization recommendations based on environmental data
    pub fn energy_optimization_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if let Some(ref temp_data) = self.temperature {
            let avg_temp = temp_data.mean_or(0.0);

            if avg_temp > 25.0 {
                recommendations
                    .push("Consider increasing cooling setpoint during peak hours".to_string());
            } else if avg_temp < 18.0 {
                recommendations
                    .push("Consider decreasing heating setpoint during off-peak hours".to_string());
            }

            // Temperature stability analysis
            let temp_std = temp_data.std(0.0);
            if temp_std > 3.0 {
                recommendations.push(
                    "High temperature variation detected - check HVAC efficiency".to_string(),
                );
            }
        }

        if let Some(ref humidity_data) = self.humidity {
            let avg_humidity = humidity_data.mean_or(0.0);

            if avg_humidity > 70.0 {
                recommendations
                    .push("High humidity detected - consider dehumidification".to_string());
            } else if avg_humidity < 30.0 {
                recommendations.push("Low humidity detected - consider humidification".to_string());
            }
        }

        Ok(recommendations)
    }
}

/// Motion and acceleration sensor analysis
pub struct MotionSensorAnalysis {
    /// Accelerometer data (3-axis)
    pub acceleration: Option<Array2<f64>>,
    /// Motion detection data (binary)
    pub motion: Option<Array1<f64>>,
    /// GPS coordinates [latitude, longitude]
    pub gps: Option<Array2<f64>>,
    /// Time stamps
    pub timestamps: Array1<i64>,
    /// Sampling frequency (Hz)
    pub fs: f64,
}

impl MotionSensorAnalysis {
    /// Create new motion sensor analysis
    pub fn new(timestamps: Array1<i64>, fs: f64) -> Result<Self> {
        check_positive(fs, "sampling_frequency")?;

        Ok(Self {
            acceleration: None,
            motion: None,
            gps: None,
            timestamps,
            fs,
        })
    }

    /// Add accelerometer data (3-axis: x, y, z)
    pub fn with_accelerometer(mut self, data: Array2<f64>) -> Result<Self> {
        if data.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Acceleration data contains non-finite values".to_string(),
            ));
        }

        if data.ncols() != 3 {
            return Err(TimeSeriesError::InvalidInput(
                "Accelerometer data must have 3 columns (x, y, z)".to_string(),
            ));
        }

        self.acceleration = Some(data);
        Ok(self)
    }

    /// Add GPS data
    pub fn with_gps(mut self, data: Array2<f64>) -> Result<Self> {
        if data.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "GPS data contains non-finite values".to_string(),
            ));
        }

        if data.ncols() != 2 {
            return Err(TimeSeriesError::InvalidInput(
                "GPS data must have 2 columns (latitude, longitude)".to_string(),
            ));
        }

        // Validate GPS coordinates
        for row in data.outer_iter() {
            let lat = row[0];
            let lon = row[1];
            if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
                return Err(TimeSeriesError::InvalidInput(
                    "Invalid GPS coordinates".to_string(),
                ));
            }
        }

        self.gps = Some(data);
        Ok(self)
    }

    /// Detect different activity types from accelerometer data
    pub fn activity_recognition(&self) -> Result<Vec<(usize, String)>> {
        let accel_data = self.acceleration.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidInput("Accelerometer data required".to_string())
        })?;

        let window_size = (2.0 * self.fs) as usize; // 2-second windows
        let mut activities = Vec::new();

        for start in (0..accel_data.nrows()).step_by(window_size) {
            let end = (start + window_size).min(accel_data.nrows());
            if end - start < window_size / 2 {
                break; // Skip incomplete windows
            }

            let window = accel_data.slice(scirs2_core::ndarray::s![start..end, ..]);

            // Calculate features
            let magnitude: Array1<f64> = window
                .outer_iter()
                .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
                .collect();

            let mean_magnitude = magnitude.clone().mean();
            let std_magnitude = magnitude.std(0.0);
            let max_magnitude = magnitude.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Simple activity classification based on acceleration patterns
            let activity = if std_magnitude > 3.0 && max_magnitude > 15.0 {
                "Running"
            } else if std_magnitude > 1.5 && mean_magnitude > 10.0 {
                "Walking"
            } else if std_magnitude < 0.5 && mean_magnitude < 10.5 {
                "Stationary"
            } else if max_magnitude > 20.0 {
                "High Impact Activity"
            } else {
                "Light Activity"
            };

            activities.push((start, activity.to_string()));
        }

        Ok(activities)
    }

    /// Calculate distance traveled from GPS data
    pub fn distance_traveled(&self) -> Result<f64> {
        let gps_data = self
            .gps
            .as_ref()
            .ok_or_else(|| TimeSeriesError::InvalidInput("GPS data required".to_string()))?;

        if gps_data.nrows() < 2 {
            return Ok(0.0);
        }

        let mut total_distance = 0.0;

        for i in 1..gps_data.nrows() {
            let lat1 = gps_data[[i - 1, 0]].to_radians();
            let lon1 = gps_data[[i - 1, 1]].to_radians();
            let lat2 = gps_data[[i, 0]].to_radians();
            let lon2 = gps_data[[i, 1]].to_radians();

            // Haversine formula for distance calculation
            let dlat = lat2 - lat1;
            let dlon = lon2 - lon1;

            let a =
                (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
            let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
            let distance = 6371000.0 * c; // Earth radius in meters

            total_distance += distance;
        }

        Ok(total_distance)
    }

    /// Detect falls using accelerometer data
    pub fn fall_detection(&self, threshold: f64) -> Result<Vec<usize>> {
        let accel_data = self.acceleration.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidInput("Accelerometer data required".to_string())
        })?;

        let mut fall_events = Vec::new();

        // Calculate magnitude of acceleration vector
        for (i, row) in accel_data.outer_iter().enumerate() {
            let magnitude = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();

            // Fall detection: sudden increase followed by near-zero acceleration
            if magnitude > threshold {
                // Check for low acceleration in the next few samples (impact followed by stillness)
                let check_samples = ((0.5 * self.fs) as usize).min(accel_data.nrows() - i - 1);
                if check_samples > 0 {
                    let future_window = accel_data
                        .slice(scirs2_core::ndarray::s![i + 1..i + 1 + check_samples, ..]);
                    let future_magnitudes: Array1<f64> = future_window
                        .outer_iter()
                        .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
                        .collect();

                    let mean_future = future_magnitudes.mean();
                    if mean_future < 2.0 {
                        // Low acceleration following high impact
                        fall_events.push(i);
                    }
                }
            }
        }

        Ok(fall_events)
    }

    /// Calculate step count from accelerometer data
    pub fn step_count(&self) -> Result<usize> {
        let accel_data = self.acceleration.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidInput("Accelerometer data required".to_string())
        })?;

        // Calculate magnitude of acceleration
        let magnitude: Array1<f64> = accel_data
            .outer_iter()
            .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
            .collect();

        // Apply high-pass filter to remove gravity component
        let mean_magnitude = magnitude.clone().mean();
        let filtered: Array1<f64> = magnitude.iter().map(|&x| x - mean_magnitude).collect();

        // Simple peak detection for step counting
        let mut steps = 0;
        let min_peak_height = 1.0; // Threshold for step detection
        let min_peak_distance = (0.5 * self.fs) as usize; // Minimum 0.5 seconds between steps
        let mut last_peak = 0;

        for i in 1..filtered.len() - 1 {
            if filtered[i] > filtered[i - 1]
                && filtered[i] > filtered[i + 1]
                && filtered[i] > min_peak_height
                && (i - last_peak) > min_peak_distance
            {
                steps += 1;
                last_peak = i;
            }
        }

        Ok(steps)
    }
}

/// General IoT sensor data quality assessment
pub struct IoTDataQualityAnalysis {
    /// Sensor data array
    pub data: Array1<f64>,
    /// Sensor type
    pub sensor_type: SensorType,
    /// Time stamps
    pub timestamps: Array1<i64>,
}

impl IoTDataQualityAnalysis {
    /// Create new data quality analysis
    pub fn new(
        data: Array1<f64>,
        sensor_type: SensorType,
        timestamps: Array1<i64>,
    ) -> Result<Self> {
        if data.len() != timestamps.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Data and timestamp arrays must have same length".to_string(),
            ));
        }

        Ok(Self {
            data,
            sensor_type,
            timestamps,
        })
    }

    /// Assess overall data quality
    pub fn assess_quality(&self) -> Result<DataQuality> {
        let missing_percentage = self.calculate_missing_percentage()?;
        let outlier_percentage = self.calculate_outlier_percentage()?;
        let snr = self.calculate_snr()?;
        let consistency_score = self.calculate_consistency_score()?;

        Ok(DataQuality {
            missing_percentage,
            outlier_percentage,
            snr,
            consistency_score,
        })
    }

    /// Calculate percentage of missing data points
    fn calculate_missing_percentage(&self) -> Result<f64> {
        let missing_count = self.data.iter().filter(|&&x| x.is_nan()).count();
        Ok((missing_count as f64 / self.data.len() as f64) * 100.0)
    }

    /// Calculate percentage of outliers using IQR method
    fn calculate_outlier_percentage(&self) -> Result<f64> {
        let mut sorted_data: Vec<f64> = self
            .data
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if sorted_data.len() < 4 {
            return Ok(0.0);
        }

        let q1_idx = sorted_data.len() / 4;
        let q3_idx = 3 * sorted_data.len() / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let outlier_count = self
            .data
            .iter()
            .filter(|&&x| !x.is_nan() && (x < lower_bound || x > upper_bound))
            .count();

        Ok((outlier_count as f64 / sorted_data.len() as f64) * 100.0)
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self) -> Result<f64> {
        let valid_data: Vec<f64> = self
            .data
            .iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .collect();

        if valid_data.is_empty() {
            return Ok(0.0);
        }

        let mean = valid_data.iter().sum::<f64>() / valid_data.len() as f64;
        let variance =
            valid_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / valid_data.len() as f64;

        let signal_power = mean.abs();
        let noise_power = variance.sqrt();

        if noise_power == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(20.0 * (signal_power / noise_power).log10())
        }
    }

    /// Calculate data consistency score based on expected patterns
    fn calculate_consistency_score(&self) -> Result<f64> {
        // Check for reasonable sampling rate consistency
        if self.timestamps.len() < 2 {
            return Ok(100.0);
        }

        let mut intervals = Vec::new();
        for i in 1..self.timestamps.len() {
            intervals.push(self.timestamps[i] - self.timestamps[i - 1]);
        }

        if intervals.is_empty() {
            return Ok(100.0);
        }

        let mean_interval = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
        let std_interval = {
            let variance = intervals
                .iter()
                .map(|&x| (x as f64 - mean_interval).powi(2))
                .sum::<f64>()
                / intervals.len() as f64;
            variance.sqrt()
        };

        // Consistency score based on interval regularity (0-100)
        let consistency = if mean_interval == 0.0 {
            0.0
        } else {
            100.0 - (std_interval / mean_interval * 100.0).min(100.0)
        };

        Ok(consistency.max(0.0))
    }
}

/// Comprehensive IoT sensor analysis
pub struct IoTAnalysis {
    /// Environmental sensor analysis
    pub environmental: Option<EnvironmentalSensorAnalysis>,
    /// Motion sensor analysis
    pub motion: Option<MotionSensorAnalysis>,
    /// Data quality assessments
    pub quality_assessments: HashMap<String, DataQuality>,
}

impl Default for IoTAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl IoTAnalysis {
    /// Create new IoT analysis
    pub fn new() -> Self {
        Self {
            environmental: None,
            motion: None,
            quality_assessments: HashMap::new(),
        }
    }

    /// Add environmental sensor analysis
    pub fn with_environmental(mut self, analysis: EnvironmentalSensorAnalysis) -> Self {
        self.environmental = Some(analysis);
        self
    }

    /// Add motion sensor analysis
    pub fn with_motion(mut self, analysis: MotionSensorAnalysis) -> Self {
        self.motion = Some(analysis);
        self
    }

    /// Add data quality assessment for a sensor
    pub fn add_quality_assessment(&mut self, sensorname: String, quality: DataQuality) {
        self.quality_assessments.insert(sensorname, quality);
    }

    /// Generate comprehensive IoT system health report
    pub fn system_health_report(&self) -> Result<HashMap<String, String>> {
        let mut report = HashMap::new();

        // Environmental system health
        if let Some(ref env) = self.environmental {
            let malfunctions = env.detect_sensor_malfunctions()?;
            let total_issues: usize = malfunctions.values().map(|v| v.len()).sum();

            let env_status = if total_issues == 0 {
                "All environmental sensors functioning normally".to_string()
            } else {
                format!("{total_issues} environmental sensor issues detected")
            };
            report.insert("Environmental_Status".to_string(), env_status);

            // Energy recommendations
            let recommendations = env.energy_optimization_recommendations()?;
            if !recommendations.is_empty() {
                report.insert(
                    "Energy_Recommendations".to_string(),
                    recommendations.join("; "),
                );
            }
        }

        // Motion system health
        if let Some(ref motion) = self.motion {
            if motion.acceleration.is_some() {
                let activities = motion.activity_recognition()?;
                let activity_summary = format!("{} activity periods detected", activities.len());
                report.insert("Activity_Status".to_string(), activity_summary);
            }

            if motion.gps.is_some() {
                let distance = motion.distance_traveled()?;
                report.insert(
                    "Distance_Traveled".to_string(),
                    format!("{distance:.2} meters"),
                );
            }
        }

        // Overall data quality
        let mut quality_issues = 0;
        for quality in self.quality_assessments.values() {
            if quality.missing_percentage > 10.0
                || quality.outlier_percentage > 5.0
                || quality.snr < 10.0
            {
                quality_issues += 1;
            }
        }

        let quality_status = if quality_issues == 0 {
            "All sensors showing good data quality".to_string()
        } else {
            format!("{quality_issues} sensors showing data quality issues")
        };
        report.insert("Data_Quality_Status".to_string(), quality_status);

        Ok(report)
    }

    /// Predict maintenance needs based on sensor data patterns
    pub fn predictive_maintenance(&self) -> Result<Vec<String>> {
        let mut maintenance_alerts = Vec::new();

        // Check data quality degradation
        for (sensor_name, quality) in &self.quality_assessments {
            if quality.missing_percentage > 20.0 {
                maintenance_alerts.push(format!(
                    "Sensor '{sensor_name}' may need replacement - high missing data rate"
                ));
            }

            if quality.outlier_percentage > 15.0 {
                maintenance_alerts.push(format!(
                    "Sensor '{sensor_name}' may need calibration - high outlier rate"
                ));
            }

            if quality.snr < 5.0 {
                maintenance_alerts.push(format!(
                    "Sensor '{sensor_name}' may have connectivity issues - low signal quality"
                ));
            }
        }

        // Environmental sensor specific maintenance
        if let Some(ref env) = self.environmental {
            let malfunctions = env.detect_sensor_malfunctions()?;

            for (sensor_type, issues) in malfunctions {
                if issues.len() > 100 {
                    maintenance_alerts.push(format!(
                        "{sensor_type} sensor requires immediate attention - multiple malfunctions detected"
                    ));
                }
            }
        }

        Ok(maintenance_alerts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_environmental_sensor_analysis() {
        let timestamps = arr1(&[1, 2, 3, 4, 5]);
        let temperatures = arr1(&[20.0, 21.0, 22.0, 23.0, 24.0]);
        let humidity = arr1(&[45.0, 46.0, 47.0, 48.0, 49.0]);

        let analysis = EnvironmentalSensorAnalysis::new(timestamps, 1.0)
            .unwrap()
            .with_temperature(temperatures)
            .unwrap()
            .with_humidity(humidity)
            .unwrap();

        let comfort = analysis.comfort_index().unwrap();
        assert_eq!(comfort.len(), 5);
        assert!(comfort.iter().all(|&x| (0.0..=100.0).contains(&x)));

        let malfunctions = analysis.detect_sensor_malfunctions().unwrap();
        assert!(malfunctions.contains_key("Temperature"));
        assert!(malfunctions.contains_key("Humidity"));
    }

    #[test]
    fn test_motion_sensor_analysis() {
        let timestamps = arr1(&[1, 2, 3, 4, 5]);
        let accel_data = arr2(&[
            [1.0, 2.0, 9.8],
            [1.1, 2.1, 9.9],
            [1.2, 2.2, 10.0],
            [1.3, 2.3, 10.1],
            [1.4, 2.4, 10.2],
        ]);

        // Use a lower sampling frequency so the window size fits our data
        let analysis = MotionSensorAnalysis::new(timestamps, 2.0)
            .unwrap()
            .with_accelerometer(accel_data)
            .unwrap();

        let activities = analysis.activity_recognition().unwrap();
        assert!(!activities.is_empty());

        let _steps = analysis.step_count().unwrap();
    }

    #[test]
    fn test_iot_data_quality() {
        let data = arr1(&[1.0, 2.0, 3.0, 100.0, 5.0]); // Contains outlier
        let timestamps = arr1(&[1, 2, 3, 4, 5]);

        let quality_analysis =
            IoTDataQualityAnalysis::new(data, SensorType::Temperature, timestamps).unwrap();
        let quality = quality_analysis.assess_quality().unwrap();

        assert!(quality.outlier_percentage > 0.0);
        assert!(quality.consistency_score >= 0.0 && quality.consistency_score <= 100.0);
    }

    #[test]
    fn test_fall_detection() {
        let timestamps = arr1(&[1, 2, 3, 4, 5]);
        // Simulate fall: normal acceleration followed by high impact then stillness
        let accel_data = arr2(&[
            [1.0, 2.0, 9.8],    // Normal
            [1.0, 2.0, 9.8],    // Normal
            [10.0, 15.0, 20.0], // High impact (fall)
            [0.5, 0.5, 0.5],    // Near stillness after fall
            [0.5, 0.5, 0.5],    // Continued stillness
        ]);

        let analysis = MotionSensorAnalysis::new(timestamps, 100.0)
            .unwrap()
            .with_accelerometer(accel_data)
            .unwrap();

        let falls = analysis.fall_detection(25.0).unwrap();
        assert!(!falls.is_empty());
        assert_eq!(falls[0], 2); // Fall detected at index 2
    }
}
