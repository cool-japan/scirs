//! Anomaly detection for the detection module
//!
//! Provides `AnomalyDetector` with Z-score, MAD, and IQR methods.
//! This detector bridges with the existing `anomaly::AnomalyMethod` enum for
//! compatibility with integration test usage.

use scirs2_core::ndarray::Array1;

use crate::anomaly::AnomalyMethod as LegacyAnomalyMethod;
use crate::error::{Result, TimeSeriesError};

/// Internal method selection for this detector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InternalMethod {
    ZScore,
    MAD,
    IQR,
    Other, // Falls back to ZScore
}

impl From<LegacyAnomalyMethod> for InternalMethod {
    fn from(m: LegacyAnomalyMethod) -> Self {
        match m {
            LegacyAnomalyMethod::ZScore => InternalMethod::ZScore,
            LegacyAnomalyMethod::ModifiedZScore => InternalMethod::MAD,
            LegacyAnomalyMethod::InterquartileRange => InternalMethod::IQR,
            _ => InternalMethod::ZScore,
        }
    }
}

/// Method for anomaly detection (for use when constructing directly)
///
/// See also `scirs2_series::anomaly::AnomalyMethod` for the full enum used
/// with the `with_method()` builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyMethod {
    /// Z-score: `|x_i - μ| / σ > threshold`
    ZScore,
    /// MAD (Median Absolute Deviation): `|x_i - median| / (1.4826 * MAD) > threshold`
    MAD,
    /// IQR: `x_i < Q1 - threshold*IQR || x_i > Q3 + threshold*IQR`
    IQR,
}

impl From<AnomalyMethod> for InternalMethod {
    fn from(m: AnomalyMethod) -> Self {
        match m {
            AnomalyMethod::ZScore => InternalMethod::ZScore,
            AnomalyMethod::MAD => InternalMethod::MAD,
            AnomalyMethod::IQR => InternalMethod::IQR,
        }
    }
}

/// Anomaly detector with configurable method and threshold
///
/// Supports both direct construction (`new(method, threshold)`) and
/// builder-style construction (`new().with_method(...).with_threshold(...)`).
///
/// The `with_method()` builder accepts the existing `scirs2_series::anomaly::AnomalyMethod`
/// enum for integration test compatibility.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    method: InternalMethod,
    threshold: f64,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::default_new()
    }
}

impl AnomalyDetector {
    /// Create an `AnomalyDetector` with default settings (ZScore, threshold 3.0)
    pub fn new() -> Self {
        Self::default_new()
    }

    /// Create an `AnomalyDetector` with explicit method and threshold
    pub fn with_params(method: AnomalyMethod, threshold: f64) -> Self {
        AnomalyDetector {
            method: method.into(),
            threshold,
        }
    }

    fn default_new() -> Self {
        AnomalyDetector {
            method: InternalMethod::ZScore,
            threshold: 3.0,
        }
    }

    /// Set the detection method (builder pattern)
    ///
    /// Accepts `scirs2_series::anomaly::AnomalyMethod` for compatibility.
    pub fn with_method(mut self, method: LegacyAnomalyMethod) -> Self {
        self.method = method.into();
        self
    }

    /// Set the method using the detection module's own `AnomalyMethod` enum
    pub fn with_own_method(mut self, method: AnomalyMethod) -> Self {
        self.method = method.into();
        self
    }

    /// Set the threshold (builder pattern)
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Detect anomalies, returning a score array (1.0 = anomaly, 0.0 = normal)
    ///
    /// This signature matches integration test usage (`anomalies[i] > 0.5`).
    pub fn detect(&self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let n = series.len();
        if n == 0 {
            return Ok(Array1::zeros(0));
        }
        if n < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Anomaly detection requires at least 3 data points".to_string(),
                required: 3,
                actual: n,
            });
        }

        let indices = match self.method {
            InternalMethod::ZScore | InternalMethod::Other => self.detect_zscore(series)?,
            InternalMethod::MAD => self.detect_mad(series)?,
            InternalMethod::IQR => self.detect_iqr(series)?,
        };

        let mut scores = Array1::<f64>::zeros(n);
        for idx in indices {
            scores[idx] = 1.0;
        }
        Ok(scores)
    }

    /// Detect anomalies returning indices
    pub fn detect_indices(&self, series: &Array1<f64>) -> Result<Vec<usize>> {
        let n = series.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Anomaly detection requires at least 3 data points".to_string(),
                required: 3,
                actual: n,
            });
        }

        match self.method {
            InternalMethod::ZScore | InternalMethod::Other => self.detect_zscore(series),
            InternalMethod::MAD => self.detect_mad(series),
            InternalMethod::IQR => self.detect_iqr(series),
        }
    }

    fn detect_zscore(&self, series: &Array1<f64>) -> Result<Vec<usize>> {
        let n = series.len();
        let mean = series.iter().sum::<f64>() / n as f64;
        let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-300 {
            return Ok(vec![]);
        }

        Ok((0..n)
            .filter(|&i| ((series[i] - mean) / std_dev).abs() > self.threshold)
            .collect())
    }

    fn detect_mad(&self, series: &Array1<f64>) -> Result<Vec<usize>> {
        let n = series.len();

        let mut sorted: Vec<f64> = series.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };

        let mut abs_devs: Vec<f64> = series.iter().map(|&x| (x - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = if n % 2 == 1 {
            abs_devs[n / 2]
        } else {
            (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
        };

        // Consistency factor 1.4826 makes MAD a consistent estimator of σ for normal data
        let sigma_hat = 1.4826 * mad;

        if sigma_hat < 1e-300 {
            return Ok(vec![]);
        }

        Ok((0..n)
            .filter(|&i| ((series[i] - median) / sigma_hat).abs() > self.threshold)
            .collect())
    }

    fn detect_iqr(&self, series: &Array1<f64>) -> Result<Vec<usize>> {
        let n = series.len();

        let mut sorted: Vec<f64> = series.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        if iqr < 1e-300 {
            return Ok(vec![]);
        }

        let lower = q1 - self.threshold * iqr;
        let upper = q3 + self.threshold * iqr;

        Ok((0..n)
            .filter(|&i| series[i] < lower || series[i] > upper)
            .collect())
    }
}

/// Compute a percentile via linear interpolation on sorted data
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }

    let rank = pct / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_normal_series(n: usize) -> Array1<f64> {
        // Deterministic pseudo-normal using Box-Muller-like approach (wrapping arithmetic)
        Array1::from_iter((0..n).map(|i| {
            let h1 = (i as u64).wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            let h2 = ((i as u64).wrapping_add(7))
                .wrapping_mul(1103515245)
                .wrapping_add(12345)
                & 0x7fffffff;
            let u1 = (h1 as f64 / 0x7fffffff_u64 as f64).max(1e-12);
            let u2 = h2 as f64 / 0x7fffffff_u64 as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }))
    }

    #[test]
    fn test_anomaly_zscore_detects_outliers() {
        // 97 normal samples near 0, 3 outliers at large deviation
        let data_base = make_normal_series(97);
        let mut data: Vec<f64> = data_base.to_vec();
        data.push(10.0); // ~10σ away
        data.push(-10.0); // ~10σ away
        data.push(8.0); // ~8σ away
        let series = Array1::from_vec(data);

        let detector = AnomalyDetector::with_params(AnomalyMethod::ZScore, 3.0);
        let scores = detector.detect(&series).expect("detect failed");

        // Should score the last 3 as anomalies
        assert!(
            scores[97] > 0.5 && scores[98] > 0.5 && scores[99] > 0.5,
            "Expected indices 97,98,99 to be anomalies: scores[97]={}, scores[98]={}, scores[99]={}",
            scores[97], scores[98], scores[99]
        );
    }

    #[test]
    fn test_anomaly_mad_detects_outliers() {
        let data_base = make_normal_series(97);
        let mut data: Vec<f64> = data_base.to_vec();
        data.push(15.0);
        data.push(-15.0);
        data.push(12.0);
        let series = Array1::from_vec(data);

        let detector = AnomalyDetector::with_params(AnomalyMethod::MAD, 3.5);
        let scores = detector.detect(&series).expect("detect failed");

        // At least one of the outliers should be detected
        let detected_any = scores[97] > 0.5 || scores[98] > 0.5 || scores[99] > 0.5;
        assert!(
            detected_any,
            "Expected at least one outlier detected with MAD: scores[97]={}, [98]={}, [99]={}",
            scores[97], scores[98], scores[99]
        );
    }
}
