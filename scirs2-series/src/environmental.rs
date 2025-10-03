//! Environmental and climate data analysis for time series
//!
//! This module provides specialized functionality for analyzing environmental
//! and climate time series data, including temperature records, precipitation,
//! atmospheric measurements, and other environmental indicators.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::check_positive;
use statrs::statistics::Statistics;
use std::collections::HashMap;

/// Climate data anomaly detection methods
#[derive(Debug, Clone)]
pub enum ClimateAnomalyMethod {
    /// Z-score based on long-term climatology
    Climatological {
        /// Z-score threshold for anomaly detection
        threshold: f64,
    },
    /// Percentile-based anomalies
    Percentile {
        /// Lower percentile threshold
        lower: f64,
        /// Upper percentile threshold
        upper: f64,
    },
    /// Temperature-specific extreme event detection
    TemperatureExtreme {
        /// Heat wave threshold temperature
        heat_threshold: f64,
        /// Cold wave threshold temperature
        cold_threshold: f64,
    },
    /// Precipitation anomaly detection
    PrecipitationAnomaly {
        /// Drought threshold for precipitation
        drought_threshold: f64,
        /// Flood threshold for precipitation
        flood_threshold: f64,
    },
}

/// Temperature time series analysis
pub struct TemperatureAnalysis {
    /// Daily temperature data
    pub temperatures: Array1<f64>,
    /// Time stamps (as day of year)
    pub time_stamps: Array1<i64>,
    /// Climatological baseline period
    pub baseline_years: (i32, i32),
}

impl TemperatureAnalysis {
    /// Create new temperature analysis
    pub fn new(
        temperatures: Array1<f64>,
        time_stamps: Array1<i64>,
        baseline_years: (i32, i32),
    ) -> Result<Self> {
        if temperatures.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Temperatures contain non-finite values".to_string(),
            ));
        }
        if temperatures.len() != time_stamps.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Temperature and time arrays must have same length".to_string(),
            ));
        }

        Ok(Self {
            temperatures,
            time_stamps,
            baseline_years,
        })
    }

    /// Calculate heat wave indicators
    pub fn detect_heat_waves(
        &self,
        threshold: f64,
        min_duration: usize,
    ) -> Result<Vec<(usize, usize)>> {
        check_positive(min_duration, "min_duration")?;

        let mut heat_waves = Vec::new();
        let mut current_start = None;

        for (i, &temp) in self.temperatures.iter().enumerate() {
            if temp > threshold {
                if current_start.is_none() {
                    current_start = Some(i);
                }
            } else if let Some(start) = current_start {
                let _duration = i - start;
                if _duration >= min_duration {
                    heat_waves.push((start, i - 1));
                }
                current_start = None;
            }
        }

        // Check for heat wave at end of series
        if let Some(start) = current_start {
            let _duration = self.temperatures.len() - start;
            if _duration >= min_duration {
                heat_waves.push((start, self.temperatures.len() - 1));
            }
        }

        Ok(heat_waves)
    }

    /// Calculate growing degree days
    pub fn growing_degree_days(
        &self,
        base_temp: f64,
        max_temp: Option<f64>,
    ) -> Result<Array1<f64>> {
        let mut gdd = Array1::zeros(self.temperatures.len());

        for (i, &_temp) in self.temperatures.iter().enumerate() {
            let effective_temp = if let Some(max_t) = max_temp {
                _temp.min(max_t)
            } else {
                _temp
            };

            gdd[i] = (effective_temp - base_temp).max(0.0);
        }

        Ok(gdd)
    }

    /// Calculate temperature trends using robust regression
    pub fn temperature_trend(&self) -> Result<(f64, f64, f64)> {
        let n = self.temperatures.len() as f64;
        let x_mean = self.time_stamps.iter().map(|&x| x as f64).sum::<f64>() / n;
        let y_mean = self.temperatures.clone().mean();

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &temp) in self.temperatures.iter().enumerate() {
            let x = self.time_stamps[i] as f64;
            let x_diff = x - x_mean;
            numerator += x_diff * (temp - y_mean);
            denominator += x_diff * x_diff;
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        // Calculate R-squared
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for (i, &temp) in self.temperatures.iter().enumerate() {
            let x = self.time_stamps[i] as f64;
            let predicted = slope * x + intercept;
            ss_res += (temp - predicted).powi(2);
            ss_tot += (temp - y_mean).powi(2);
        }

        let r_squared = 1.0 - (ss_res / ss_tot);

        Ok((slope, intercept, r_squared))
    }

    /// Calculate climate normals (30-year averages)
    pub fn climate_normals(&self, windowsize: usize) -> Result<Array1<f64>> {
        if windowsize > self.temperatures.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Window _size larger than data".to_string(),
            ));
        }

        let mut normals = Array1::zeros(self.temperatures.len() - windowsize + 1);

        for i in 0..normals.len() {
            let window = self
                .temperatures
                .slice(scirs2_core::ndarray::s![i..i + windowsize]);
            normals[i] = window.mean();
        }

        Ok(normals)
    }
}

/// Precipitation analysis for hydrological studies
pub struct PrecipitationAnalysis {
    /// Daily precipitation amounts
    pub precipitation: Array1<f64>,
    /// Time stamps
    pub time_stamps: Array1<i64>,
}

impl PrecipitationAnalysis {
    /// Create new precipitation analysis
    pub fn new(precipitation: Array1<f64>, timestamps: Array1<i64>) -> Result<Self> {
        if precipitation.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Precipitation contains non-finite values".to_string(),
            ));
        }

        // Check for negative _precipitation
        if precipitation.iter().any(|&x| x < 0.0) {
            return Err(TimeSeriesError::InvalidInput(
                "Precipitation values cannot be negative".to_string(),
            ));
        }

        Ok(Self {
            precipitation,
            time_stamps: timestamps,
        })
    }

    /// Detect drought periods using Standardized Precipitation Index
    pub fn drought_detection(
        &self,
        windowsize: usize,
        threshold: f64,
    ) -> Result<Vec<(usize, usize)>> {
        let spi = self.standardized_precipitation_index(windowsize)?;
        let mut droughts = Vec::new();
        let mut current_start = None;

        for (i, &spi_val) in spi.iter().enumerate() {
            if spi_val <= threshold {
                if current_start.is_none() {
                    current_start = Some(i);
                }
            } else if let Some(start) = current_start {
                droughts.push((start, i - 1));
                current_start = None;
            }
        }

        // Check for drought at end of series
        if let Some(start) = current_start {
            droughts.push((start, spi.len() - 1));
        }

        Ok(droughts)
    }

    /// Calculate Standardized Precipitation Index (SPI)
    pub fn standardized_precipitation_index(&self, windowsize: usize) -> Result<Array1<f64>> {
        if windowsize > self.precipitation.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Window _size larger than data".to_string(),
            ));
        }

        let mut spi = Array1::zeros(self.precipitation.len() - windowsize + 1);

        // Calculate rolling sums
        let mut rolling_sums = Array1::zeros(spi.len());
        for i in 0..spi.len() {
            rolling_sums[i] = self
                .precipitation
                .slice(scirs2_core::ndarray::s![i..i + windowsize])
                .sum();
        }

        // Calculate statistics
        let mean = rolling_sums.clone().mean();
        let std_dev = rolling_sums.std(0.0);

        // Calculate SPI
        for i in 0..spi.len() {
            spi[i] = (rolling_sums[i] - mean) / std_dev;
        }

        Ok(spi)
    }

    /// Calculate rainfall intensity categories
    pub fn rainfall_intensity_classification(&self) -> Result<HashMap<String, usize>> {
        let mut classification = HashMap::new();
        classification.insert("No Rain".to_string(), 0);
        classification.insert("Light".to_string(), 0);
        classification.insert("Moderate".to_string(), 0);
        classification.insert("Heavy".to_string(), 0);
        classification.insert("Very Heavy".to_string(), 0);
        classification.insert("Extreme".to_string(), 0);

        for &precip in self.precipitation.iter() {
            let category = match precip {
                0.0 => "No Rain",
                x if x < 2.5 => "Light",
                x if x < 7.6 => "Moderate",
                x if x < 35.0 => "Heavy",
                x if x < 50.0 => "Very Heavy",
                _ => "Extreme",
            };

            *classification.get_mut(category).unwrap() += 1;
        }

        Ok(classification)
    }

    /// Calculate consecutive dry days
    pub fn consecutive_dry_days(&self, drythreshold: f64) -> Result<Array1<usize>> {
        let mut consecutive_days = Array1::zeros(self.precipitation.len());
        let mut current_streak = 0;

        for (i, &precip) in self.precipitation.iter().enumerate() {
            if precip <= drythreshold {
                current_streak += 1;
            } else {
                current_streak = 0;
            }
            consecutive_days[i] = current_streak;
        }

        Ok(consecutive_days)
    }
}

/// Atmospheric pressure and wind analysis
pub struct AtmosphericAnalysis {
    /// Atmospheric pressure measurements
    pub pressure: Array1<f64>,
    /// Wind speed measurements  
    pub wind_speed: Array1<f64>,
    /// Wind direction (degrees)
    pub wind_direction: Option<Array1<f64>>,
    /// Time stamps
    pub time_stamps: Array1<i64>,
}

impl AtmosphericAnalysis {
    /// Create new atmospheric analysis
    pub fn new(
        pressure: Array1<f64>,
        wind_speed: Array1<f64>,
        wind_direction: Option<Array1<f64>>,
        time_stamps: Array1<i64>,
    ) -> Result<Self> {
        if pressure.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Pressure contains non-finite values".to_string(),
            ));
        }
        if wind_speed.iter().any(|x| !x.is_finite()) {
            return Err(TimeSeriesError::InvalidInput(
                "Wind _speed contains non-finite values".to_string(),
            ));
        }

        if let Some(ref dir) = wind_direction {
            if dir.iter().any(|x| !x.is_finite()) {
                return Err(TimeSeriesError::InvalidInput(
                    "Wind _direction contains non-finite values".to_string(),
                ));
            }
            if dir.iter().any(|&x| !(0.0..360.0).contains(&x)) {
                return Err(TimeSeriesError::InvalidInput(
                    "Wind _direction must be between 0 and 360 degrees".to_string(),
                ));
            }
        }

        Ok(Self {
            pressure,
            wind_speed,
            wind_direction,
            time_stamps,
        })
    }

    /// Detect storm systems using pressure drops
    pub fn detect_storm_systems(
        &self,
        pressure_drop_threshold: f64,
        min_duration: usize,
    ) -> Result<Vec<(usize, usize)>> {
        check_positive(min_duration, "min_duration")?;

        // Calculate pressure changes
        let mut pressure_changes = Array1::zeros(self.pressure.len() - 1);
        for i in 0..pressure_changes.len() {
            pressure_changes[i] = self.pressure[i + 1] - self.pressure[i];
        }

        let mut storms = Vec::new();
        let mut current_start = None;

        for (i, &change) in pressure_changes.iter().enumerate() {
            if change <= -pressure_drop_threshold {
                if current_start.is_none() {
                    current_start = Some(i);
                }
            } else if let Some(start) = current_start {
                let _duration = i - start;
                if _duration >= min_duration {
                    storms.push((start, i - 1));
                }
                current_start = None;
            }
        }

        Ok(storms)
    }

    /// Calculate wind power density
    pub fn wind_power_density(&self, airdensity: f64) -> Result<Array1<f64>> {
        check_positive(airdensity, "airdensity")?;

        let mut power_density = Array1::zeros(self.wind_speed.len());

        for (i, &speed) in self.wind_speed.iter().enumerate() {
            // Power _density = 0.5 * ρ * v³
            power_density[i] = 0.5 * airdensity * speed.powi(3);
        }

        Ok(power_density)
    }

    /// Calculate wind rose statistics
    pub fn wind_rose_statistics(&self, directionbins: usize) -> Result<Array2<f64>> {
        let wind_dir = self.wind_direction.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidInput("Wind direction data required".to_string())
        })?;

        let bin_size = 360.0 / directionbins as f64;
        let speed_bins = [0.0, 5.0, 10.0, 15.0, 20.0, f64::INFINITY];

        let mut rose_data = Array2::zeros((directionbins, speed_bins.len() - 1));

        for (&dir, &speed) in wind_dir.iter().zip(self.wind_speed.iter()) {
            let dir_bin = ((dir / bin_size).floor() as usize).min(directionbins - 1);

            for (s_bin, window) in speed_bins.windows(2).enumerate() {
                if speed >= window[0] && speed < window[1] {
                    rose_data[[dir_bin, s_bin]] += 1.0;
                    break;
                }
            }
        }

        // Convert to percentages
        let total = rose_data.sum();
        if total > 0.0 {
            rose_data /= total / 100.0;
        }

        Ok(rose_data)
    }
}

/// Climate index calculations (ENSO, NAO, etc.)
pub struct ClimateIndices;

impl ClimateIndices {
    /// Calculate Southern Oscillation Index (SOI)
    pub fn southern_oscillation_index(
        tahiti_pressure: &Array1<f64>,
        darwin_pressure: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        if tahiti_pressure.len() != darwin_pressure.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Pressure arrays must have same length".to_string(),
            ));
        }

        let tahiti_mean = tahiti_pressure.mean().unwrap();
        let darwin_mean = darwin_pressure.mean().unwrap();
        let tahiti_std = tahiti_pressure.std(0.0);
        let darwin_std = darwin_pressure.std(0.0);

        let mut soi = Array1::zeros(tahiti_pressure.len());

        for i in 0..soi.len() {
            let tahiti_norm = (tahiti_pressure[i] - tahiti_mean) / tahiti_std;
            let darwin_norm = (darwin_pressure[i] - darwin_mean) / darwin_std;
            soi[i] = tahiti_norm - darwin_norm;
        }

        Ok(soi)
    }

    /// Calculate North Atlantic Oscillation (NAO) index
    pub fn north_atlantic_oscillation(
        azores_pressure: &Array1<f64>,
        iceland_pressure: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        if azores_pressure.len() != iceland_pressure.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Pressure arrays must have same length".to_string(),
            ));
        }

        let azores_mean = azores_pressure.mean().unwrap();
        let iceland_mean = iceland_pressure.mean().unwrap();
        let azores_std = azores_pressure.std(0.0);
        let iceland_std = iceland_pressure.std(0.0);

        let mut nao = Array1::zeros(azores_pressure.len());

        for i in 0..nao.len() {
            let azores_norm = (azores_pressure[i] - azores_mean) / azores_std;
            let iceland_norm = (iceland_pressure[i] - iceland_mean) / iceland_std;
            nao[i] = azores_norm - iceland_norm;
        }

        Ok(nao)
    }

    /// Calculate Palmer Drought Severity Index (PDSI)
    pub fn palmer_drought_severity_index(
        precipitation: &Array1<f64>,
        temperature: &Array1<f64>,
        _latitude: f64,
    ) -> Result<Array1<f64>> {
        if precipitation.len() != temperature.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Precipitation and temperature arrays must have same length".to_string(),
            ));
        }

        // Simplified PDSI calculation
        let mut pdsi = Array1::zeros(precipitation.len());
        let mut soil_moisture = 0.0;

        for i in 0..pdsi.len() {
            // Calculate potential evapotranspiration (simplified Thornthwaite)
            let pet = if temperature[i] > 0.0 {
                16.0 * (10.0 * temperature[i] / 100.0).powf(1.514)
            } else {
                0.0
            };

            // Water balance
            soil_moisture += precipitation[i] - pet;
            soil_moisture = soil_moisture.clamp(-100.0, 100.0);

            // Simplified PDSI calculation
            pdsi[i] = soil_moisture / 25.0;
        }

        Ok(pdsi)
    }
}

/// Comprehensive environmental data analysis
pub struct EnvironmentalAnalysis {
    /// Temperature analysis
    pub temperature: Option<TemperatureAnalysis>,
    /// Precipitation analysis
    pub precipitation: Option<PrecipitationAnalysis>,
    /// Atmospheric analysis
    pub atmospheric: Option<AtmosphericAnalysis>,
}

impl EnvironmentalAnalysis {
    /// Create new environmental analysis
    pub fn new() -> Self {
        Self {
            temperature: None,
            precipitation: None,
            atmospheric: None,
        }
    }

    /// Add temperature data
    pub fn with_temperature(mut self, analysis: TemperatureAnalysis) -> Self {
        self.temperature = Some(analysis);
        self
    }

    /// Add precipitation data
    pub fn with_precipitation(mut self, analysis: PrecipitationAnalysis) -> Self {
        self.precipitation = Some(analysis);
        self
    }

    /// Add atmospheric data
    pub fn with_atmospheric(mut self, analysis: AtmosphericAnalysis) -> Self {
        self.atmospheric = Some(analysis);
        self
    }

    /// Comprehensive climate anomaly detection
    pub fn detect_climate_anomalies(
        &self,
        method: ClimateAnomalyMethod,
    ) -> Result<Vec<(String, usize, usize)>> {
        let mut anomalies = Vec::new();

        match method {
            ClimateAnomalyMethod::TemperatureExtreme {
                heat_threshold,
                cold_threshold,
            } => {
                if let Some(ref temp_analysis) = self.temperature {
                    // Detect heat waves
                    let heat_waves = temp_analysis.detect_heat_waves(heat_threshold, 3)?;
                    for (start, end) in heat_waves {
                        anomalies.push(("Heat Wave".to_string(), start, end));
                    }

                    // Detect cold spells
                    for (i, &temp) in temp_analysis.temperatures.iter().enumerate() {
                        if temp < cold_threshold {
                            anomalies.push(("Cold Spell".to_string(), i, i));
                        }
                    }
                }
            }
            ClimateAnomalyMethod::PrecipitationAnomaly {
                drought_threshold,
                flood_threshold,
            } => {
                if let Some(ref precip_analysis) = self.precipitation {
                    let spi = precip_analysis.standardized_precipitation_index(30)?;

                    for (i, &spi_val) in spi.iter().enumerate() {
                        if spi_val <= drought_threshold {
                            anomalies.push(("Drought".to_string(), i, i));
                        } else if spi_val >= flood_threshold {
                            anomalies.push(("Flood Risk".to_string(), i, i));
                        }
                    }
                }
            }
            _ => {
                return Err(TimeSeriesError::InvalidInput(
                    "Anomaly method not yet implemented".to_string(),
                ));
            }
        }

        Ok(anomalies)
    }

    /// Calculate environmental stress index
    pub fn environmental_stress_index(&self) -> Result<Array1<f64>> {
        let mut stress_factors = Vec::new();

        // Temperature stress
        if let Some(ref temp_analysis) = self.temperature {
            let temp_mean = temp_analysis.temperatures.clone().mean();
            let temp_std = temp_analysis.temperatures.std(0.0);

            let temp_stress: Array1<f64> = temp_analysis
                .temperatures
                .iter()
                .map(|&t| ((t - temp_mean) / temp_std).abs())
                .collect();
            stress_factors.push(temp_stress);
        }

        // Precipitation stress
        if let Some(ref precip_analysis) = self.precipitation {
            let spi = precip_analysis.standardized_precipitation_index(30)?;
            let precip_stress: Array1<f64> = spi.iter().map(|&s| s.abs()).collect();
            stress_factors.push(precip_stress);
        }

        if stress_factors.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No environmental data available for stress calculation".to_string(),
            ));
        }

        // Combine stress factors
        let min_len = stress_factors.iter().map(|s| s.len()).min().unwrap();
        let mut combined_stress = Array1::zeros(min_len);

        for i in 0..min_len {
            let mut stress_sum = 0.0;
            for factor in &stress_factors {
                stress_sum += factor[i];
            }
            combined_stress[i] = stress_sum / stress_factors.len() as f64;
        }

        Ok(combined_stress)
    }
}

impl Default for EnvironmentalAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_temperature_analysis() {
        let temps = arr1(&[20.0, 25.0, 30.0, 35.0, 40.0, 38.0, 32.0, 28.0]);
        let times = arr1(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let analysis = TemperatureAnalysis::new(temps, times, (1990, 2020)).unwrap();
        let heat_waves = analysis.detect_heat_waves(35.0, 2).unwrap();

        assert_eq!(heat_waves.len(), 1);
        // Check that we detected a heat wave somewhere in the expected range
        assert!(heat_waves[0].0 >= 3 && heat_waves[0].0 <= 4);
        assert!(heat_waves[0].1 >= 5 && heat_waves[0].1 <= 6);
    }

    #[test]
    fn test_precipitation_analysis() {
        let precip = arr1(&[0.0, 2.0, 0.0, 0.0, 15.0, 25.0, 0.0, 1.0]);
        let times = arr1(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let analysis = PrecipitationAnalysis::new(precip, times).unwrap();
        let dry_days = analysis.consecutive_dry_days(1.0).unwrap();

        assert_eq!(dry_days[2], 1); // Third day (index 2) has 1 consecutive dry day
        assert_eq!(dry_days[3], 2); // Fourth day (index 3) has 2 consecutive dry days
    }

    #[test]
    fn test_climate_indices() {
        let tahiti = arr1(&[1013.0, 1015.0, 1010.0, 1018.0]);
        let darwin = arr1(&[1008.0, 1012.0, 1005.0, 1020.0]);

        let soi = ClimateIndices::southern_oscillation_index(&tahiti, &darwin).unwrap();
        assert_eq!(soi.len(), 4);
    }

    #[test]
    fn test_wind_power_density() {
        let pressure = arr1(&[1013.0, 1015.0, 1010.0, 1018.0]);
        let wind_speed = arr1(&[5.0, 10.0, 15.0, 8.0]);
        let times = arr1(&[1, 2, 3, 4]);

        let analysis = AtmosphericAnalysis::new(pressure, wind_speed, None, times).unwrap();
        let power_density = analysis.wind_power_density(1.225).unwrap();

        // Check that power density increases with wind speed cubed
        assert!(power_density[2] > power_density[1]); // 15 m/s > 10 m/s
    }
}
