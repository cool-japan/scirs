//! Pattern detection for time series
//!
//! Detects trends, seasonality, and cyclic patterns in time series data.
//! Uses linear regression for trend detection, FFT-based periodogram for
//! seasonal detection, and autocorrelation for cyclic pattern detection.

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};

/// Detected pattern in a time series
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Linear trend with the given slope
    Trend(f64),
    /// Seasonal pattern with given period and strength (ratio of peak to total power)
    Seasonal {
        /// Detected seasonal period in samples
        period: usize,
        /// Relative strength: peak power / total power in the periodogram
        strength: f64,
    },
    /// Cyclic pattern with approximate period (autocorrelation-based)
    Cyclic {
        /// Approximate period of the cyclic pattern in samples
        period: usize,
    },
}

/// Detector for trends, seasonality, and cyclic patterns
#[derive(Debug, Clone)]
pub struct PatternDetector {
    /// T-statistic threshold for trend significance
    trend_t_threshold: f64,
    /// Ratio threshold for seasonal significance (peak / median power)
    seasonal_ratio_threshold: f64,
    /// Autocorrelation threshold for cyclic detection
    cyclic_acf_threshold: f64,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    /// Create a new PatternDetector with default parameters
    pub fn new() -> Self {
        PatternDetector {
            trend_t_threshold: 2.0,
            seasonal_ratio_threshold: 2.0,
            cyclic_acf_threshold: 0.3,
        }
    }

    /// Create a PatternDetector with custom parameters
    pub fn with_params(
        trend_t_threshold: f64,
        seasonal_ratio_threshold: f64,
        cyclic_acf_threshold: f64,
    ) -> Self {
        PatternDetector {
            trend_t_threshold,
            seasonal_ratio_threshold,
            cyclic_acf_threshold,
        }
    }

    /// Detect patterns in a time series
    ///
    /// Returns a vector of detected patterns. Multiple patterns may be detected.
    pub fn detect(&self, series: &Array1<f64>) -> Vec<Pattern> {
        let n = series.len();
        if n < 8 {
            return vec![];
        }

        let mut patterns = Vec::new();

        // Detect trend
        if let Some(trend) = self.detect_trend(series) {
            patterns.push(trend);
        }

        // Detect seasonality via periodogram
        if let Some(seasonal) = self.detect_seasonal(series) {
            // Detect cyclic patterns at lags beyond detected seasonal period
            if let Some(cyclic) = self.detect_cyclic(series, Some(seasonal.period_hint())) {
                patterns.push(cyclic);
            }
            patterns.push(seasonal.into_pattern());
        } else {
            // No seasonal: look for cyclic at lags > N/4
            let min_cyclic_lag = n / 4;
            if let Some(cyclic) = self.detect_cyclic(series, Some(min_cyclic_lag)) {
                patterns.push(cyclic);
            }
        }

        patterns
    }

    /// Detect a linear trend using OLS regression and t-statistic
    fn detect_trend(&self, series: &Array1<f64>) -> Option<Pattern> {
        let n = series.len();
        let n_f = n as f64;

        // x = [0, 1, ..., n-1], compute OLS: y = a + b*x
        let x_bar = (n_f - 1.0) / 2.0;
        let y_bar: f64 = series.iter().sum::<f64>() / n_f;

        let sxx: f64 = (0..n)
            .map(|i| {
                let xi = i as f64 - x_bar;
                xi * xi
            })
            .sum();

        if sxx < 1e-12 {
            return None;
        }

        let sxy: f64 = (0..n)
            .map(|i| {
                let xi = i as f64 - x_bar;
                let yi = series[i] - y_bar;
                xi * yi
            })
            .sum();

        let slope = sxy / sxx;
        let intercept = y_bar - slope * x_bar;

        // Residual variance
        let sse: f64 = (0..n)
            .map(|i| {
                let fitted = intercept + slope * (i as f64);
                let res = series[i] - fitted;
                res * res
            })
            .sum();

        if n < 3 {
            return None;
        }

        let s2 = sse / (n_f - 2.0);
        if s2 < 1e-300 {
            // Near-perfect fit
            return Some(Pattern::Trend(slope));
        }

        let se_slope = (s2 / sxx).sqrt();
        if se_slope < 1e-300 {
            return None;
        }

        let t_stat = slope / se_slope;

        if t_stat.abs() > self.trend_t_threshold {
            Some(Pattern::Trend(slope))
        } else {
            None
        }
    }

    /// Detect seasonal patterns via DFT periodogram
    fn detect_seasonal(&self, series: &Array1<f64>) -> Option<SeasonalHint> {
        let n = series.len();
        if n < 8 {
            return None;
        }

        // Demean
        let mean: f64 = series.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = series.iter().map(|&x| x - mean).collect();

        // Compute power spectrum via DFT for k in [2, N/2]
        let max_k = n / 2;
        if max_k < 2 {
            return None;
        }

        let mut powers = Vec::with_capacity(max_k + 1);

        for k in 0..=max_k {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, &x) in centered.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            let power = (re * re + im * im) / n as f64;
            powers.push(power);
        }

        // Find dominant frequency k in [2, N/2] (period N/k in [2, N/2])
        // Skip k=0 (DC) and k=1 (very long period)
        let valid_k_start = 2usize;
        let valid_k_end = max_k;

        if valid_k_start > valid_k_end {
            return None;
        }

        // Find peak power in valid range
        let mut best_k = valid_k_start;
        let mut best_power = powers[valid_k_start];
        for k in (valid_k_start + 1)..=valid_k_end {
            if powers[k] > best_power {
                best_power = powers[k];
                best_k = k;
            }
        }

        // Compute median power (excluding DC and the peak itself)
        let mut sorted_powers: Vec<f64> = powers[valid_k_start..=valid_k_end].to_vec();
        sorted_powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_power = if sorted_powers.is_empty() {
            return None;
        } else if sorted_powers.len() % 2 == 1 {
            sorted_powers[sorted_powers.len() / 2]
        } else {
            (sorted_powers[sorted_powers.len() / 2 - 1] + sorted_powers[sorted_powers.len() / 2])
                / 2.0
        };

        if median_power < 1e-300 {
            // All power is in DC; no seasonal
            return None;
        }

        let ratio = best_power / median_power;

        if ratio > self.seasonal_ratio_threshold {
            let period = n / best_k;
            let total_power: f64 = powers[valid_k_start..=valid_k_end].iter().sum();
            let strength = if total_power > 1e-300 {
                best_power / total_power
            } else {
                1.0
            };
            Some(SeasonalHint {
                seasonal: Pattern::Seasonal { period, strength },
                period_size: period,
            })
        } else {
            None
        }
    }

    /// Detect cyclic patterns via autocorrelation peaks beyond seasonal period
    fn detect_cyclic(&self, series: &Array1<f64>, min_lag: Option<usize>) -> Option<Pattern> {
        let n = series.len();
        let min_lag = min_lag.unwrap_or(n / 4);
        let max_lag = n / 2;

        if min_lag >= max_lag {
            return None;
        }

        let mean: f64 = series.iter().sum::<f64>() / n as f64;
        let variance: f64 = series.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        if variance < 1e-300 {
            return None;
        }

        // Compute autocorrelation at lags from min_lag to max_lag
        let mut best_lag = 0;
        let mut best_acf = 0.0f64;

        for lag in min_lag..=max_lag {
            let mut cov = 0.0;
            let count = n - lag;
            for i in 0..count {
                cov += (series[i] - mean) * (series[i + lag] - mean);
            }
            let acf = cov / (count as f64 * variance);

            // We want peaks (local max) above threshold
            if acf > best_acf {
                best_acf = acf;
                best_lag = lag;
            }
        }

        if best_acf > self.cyclic_acf_threshold && best_lag > 0 {
            Some(Pattern::Cyclic { period: best_lag })
        } else {
            None
        }
    }

    /// Detect the dominant period (convenience method)
    ///
    /// Returns the period of the strongest seasonal pattern, or None if no seasonal
    /// pattern is detected.
    pub fn detect_period(&self, series: &Array1<f64>) -> Result<usize> {
        let n = series.len();
        if n < 8 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 8 data points for period detection".to_string(),
                required: 8,
                actual: n,
            });
        }

        let patterns = self.detect(series);
        for pattern in &patterns {
            if let Pattern::Seasonal { period, .. } = pattern {
                return Ok(*period);
            }
        }

        // Fall back to ACF-based detection
        self.detect_period_acf(series)
    }

    /// ACF-based period detection fallback
    fn detect_period_acf(&self, series: &Array1<f64>) -> Result<usize> {
        let n = series.len();
        let mean: f64 = series.iter().sum::<f64>() / n as f64;
        let variance: f64 = series.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        if variance < 1e-300 {
            return Ok(1);
        }

        let max_lag = n / 2;
        let min_lag = 2usize;

        let mut best_lag = min_lag;
        let mut best_acf = f64::NEG_INFINITY;

        for lag in min_lag..=max_lag {
            let count = n - lag;
            let mut cov = 0.0;
            for i in 0..count {
                cov += (series[i] - mean) * (series[i + lag] - mean);
            }
            let acf = cov / (count as f64 * variance);
            if acf > best_acf {
                best_acf = acf;
                best_lag = lag;
            }
        }

        Ok(best_lag)
    }
}

/// Internal helper struct
struct SeasonalHint {
    seasonal: Pattern,
    period_size: usize,
}

impl SeasonalHint {
    fn period_hint(&self) -> usize {
        self.period_size
    }
}

impl From<SeasonalHint> for Pattern {
    fn from(h: SeasonalHint) -> Self {
        h.seasonal
    }
}

// Make SeasonalHint accessible only in this module
impl SeasonalHint {
    fn into_pattern(self) -> Pattern {
        self.seasonal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_detect_seasonal_sine_period12() {
        // Pure sine wave with period 12 should detect Seasonal{period:12}
        let n = 120;
        let series: Array1<f64> =
            Array1::from_iter((0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()));

        let detector = PatternDetector::new();
        let patterns = detector.detect(&series);

        let has_seasonal_12 = patterns.iter().any(|p| {
            if let Pattern::Seasonal { period, .. } = p {
                // Allow period to be 12 or harmonics (6, 24)
                *period >= 10 && *period <= 14
            } else {
                false
            }
        });

        assert!(
            has_seasonal_12,
            "Expected Seasonal{{period≈12}} but got: {patterns:?}"
        );
    }

    #[test]
    fn test_detect_trend_linear() {
        // y = i: strong positive trend
        let n = 100;
        let series: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));

        let detector = PatternDetector::new();
        let patterns = detector.detect(&series);

        let has_positive_trend = patterns.iter().any(|p| {
            if let Pattern::Trend(slope) = p {
                *slope > 0.0
            } else {
                false
            }
        });

        assert!(
            has_positive_trend,
            "Expected Trend(positive) but got: {patterns:?}"
        );
    }
}
