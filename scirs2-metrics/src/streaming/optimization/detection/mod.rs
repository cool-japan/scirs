//! Concept drift detection algorithms
//!
//! This module provides comprehensive concept drift detection capabilities:
//! - ADWIN (Adaptive Windowing) algorithm
//! - DDM (Drift Detection Method)
//! - Page-Hinkley test
//! - Statistical drift detection methods
//! - Ensemble-based drift detection

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

pub use super::config::{DriftDetectionMethod, AlertSeverity};

/// Trait for concept drift detection algorithms
pub trait ConceptDriftDetector<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum>:
    Send + Sync + std::fmt::Debug
{
    /// Add a new value and check for drift
    fn add_element(&mut self, value: F) -> Result<DriftDetectionResult>;
    
    /// Reset the detector state
    fn reset(&mut self);
    
    /// Get current drift statistics
    fn get_statistics(&self) -> DriftStatistics<F>;
    
    /// Check if detector detected drift
    fn detected_drift(&self) -> bool;
    
    /// Check if detector is in warning state
    fn detected_warning(&self) -> bool;
    
    /// Get detector name
    fn name(&self) -> &str;
    
    /// Get detector configuration
    fn get_config(&self) -> std::collections::HashMap<String, f64>;
}

/// Result of drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionResult {
    /// Drift status
    pub status: DriftStatus,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Change magnitude
    pub magnitude: f64,
    /// Detection timestamp
    pub timestamp: SystemTime,
}

/// Drift status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DriftStatus {
    /// No drift detected
    Stable,
    /// Warning level drift
    Warning,
    /// Confirmed drift
    Drift,
}

/// Drift detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftStatistics<F: Float + std::fmt::Debug> {
    /// Number of processed elements
    pub n_elements: usize,
    /// Current mean
    pub mean: F,
    /// Current variance
    pub variance: F,
    /// Total detected drifts
    pub total_drifts: usize,
    /// Total warnings
    pub total_warnings: usize,
    /// Last drift time
    pub last_drift_time: Option<SystemTime>,
}

/// ADWIN (Adaptive Windowing) drift detector
#[derive(Debug)]
pub struct AdwinDetector<F: Float + std::fmt::Debug> {
    /// Confidence parameter delta
    delta: f64,
    /// List of buckets
    buckets: VecDeque<Bucket<F>>,
    /// Total sum
    total_sum: F,
    /// Total count
    total_count: usize,
    /// Detected drift flag
    drift_detected: bool,
    /// Statistics
    statistics: DriftStatistics<F>,
    /// Creation time
    created_at: Instant,
}

#[derive(Debug, Clone)]
struct Bucket<F: Float + std::fmt::Debug> {
    sum: F,
    count: usize,
    max_bucket_size: usize,
}

impl<F: Float + std::fmt::Debug + Send + Sync> Bucket<F> {
    fn new() -> Self {
        Self {
            sum: F::zero(),
            count: 0,
            max_bucket_size: 1,
        }
    }
    
    fn insert(&mut self, value: F) {
        self.sum = self.sum + value;
        self.count += 1;
    }
    
    fn can_merge(&self, other: &Bucket<F>) -> bool {
        self.max_bucket_size == other.max_bucket_size
    }
    
    fn merge(&mut self, other: &Bucket<F>) -> Result<()> {
        if !self.can_merge(other) {
            return Err(MetricsError::InvalidInput(
                "Cannot merge buckets with different max sizes".to_string()
            ));
        }
        
        self.sum = self.sum + other.sum;
        self.count += other.count;
        self.max_bucket_size *= 2;
        
        Ok(())
    }
    
    fn mean(&self) -> F {
        if self.count == 0 {
            F::zero()
        } else {
            self.sum / F::from(self.count).unwrap()
        }
    }
}

impl<F: Float + std::iter::Sum + std::fmt::Debug + Send + Sync> AdwinDetector<F> {
    /// Create a new ADWIN detector
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            buckets: VecDeque::new(),
            total_sum: F::zero(),
            total_count: 0,
            drift_detected: false,
            statistics: DriftStatistics {
                n_elements: 0,
                mean: F::zero(),
                variance: F::zero(),
                total_drifts: 0,
                total_warnings: 0,
                last_drift_time: None,
            },
            created_at: Instant::now(),
        }
    }
    
    /// Calculate cut threshold
    fn calculate_cut_threshold(&self, n0: usize, n1: usize) -> f64 {
        let n = (n0 + n1) as f64;
        let delta_prime = self.delta / n;
        let m = 1.0 / (2.0 * n0 as f64) + 1.0 / (2.0 * n1 as f64);
        (2.0 * delta_prime.ln() / m).sqrt()
    }
    
    /// Check for drift between two windows
    fn check_drift(&self, mean0: F, mean1: F, n0: usize, n1: usize) -> bool {
        let threshold = self.calculate_cut_threshold(n0, n1);
        let diff = (mean0 - mean1).abs();
        diff > F::from(threshold).unwrap()
    }
    
    /// Compress buckets to maintain efficiency
    fn compress_buckets(&mut self) {
        let mut i = 0;
        while i < self.buckets.len() - 1 {
            if self.buckets[i].can_merge(&self.buckets[i + 1]) {
                let next_bucket = self.buckets.remove(i + 1).unwrap();
                if let Err(_) = self.buckets[i].merge(&next_bucket) {
                    // If merge fails, put the bucket back
                    self.buckets.insert(i + 1, next_bucket);
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }
}

impl<F: Float + std::iter::Sum + std::fmt::Debug + Send + Sync> ConceptDriftDetector<F>
    for AdwinDetector<F>
{
    fn add_element(&mut self, value: F) -> Result<DriftDetectionResult> {
        self.drift_detected = false;
        
        // Add new bucket if needed
        if self.buckets.is_empty() || self.buckets.back().unwrap().count >= self.buckets.back().unwrap().max_bucket_size {
            self.buckets.push_back(Bucket::new());
        }
        
        // Insert value into the last bucket
        if let Some(bucket) = self.buckets.back_mut() {
            bucket.insert(value);
        }
        
        self.total_sum = self.total_sum + value;
        self.total_count += 1;
        self.statistics.n_elements += 1;
        
        // Update statistics
        self.statistics.mean = self.total_sum / F::from(self.total_count).unwrap();
        
        // Check for drift
        let mut drift_point = None;
        let mut n0 = 0;
        let mut sum0 = F::zero();
        
        for (i, bucket) in self.buckets.iter().enumerate() {
            n0 += bucket.count;
            sum0 = sum0 + bucket.sum;
            
            let n1 = self.total_count - n0;
            if n1 > 0 {
                let sum1 = self.total_sum - sum0;
                let mean0 = sum0 / F::from(n0).unwrap();
                let mean1 = sum1 / F::from(n1).unwrap();
                
                if self.check_drift(mean0, mean1, n0, n1) {
                    drift_point = Some(i);
                    break;
                }
            }
        }
        
        // Handle drift detection
        if let Some(point) = drift_point {
            // Remove buckets before drift point
            for _ in 0..=point {
                if let Some(bucket) = self.buckets.pop_front() {
                    self.total_sum = self.total_sum - bucket.sum;
                    self.total_count -= bucket.count;
                }
            }
            
            self.drift_detected = true;
            self.statistics.total_drifts += 1;
            self.statistics.last_drift_time = Some(SystemTime::now());
            
            // Recalculate mean after drift
            if self.total_count > 0 {
                self.statistics.mean = self.total_sum / F::from(self.total_count).unwrap();
            }
        }
        
        // Compress buckets to maintain efficiency
        self.compress_buckets();
        
        Ok(DriftDetectionResult {
            status: if self.drift_detected { DriftStatus::Drift } else { DriftStatus::Stable },
            confidence: if self.drift_detected { 1.0 - self.delta } else { 0.0 },
            magnitude: if self.drift_detected { 1.0 } else { 0.0 },
            timestamp: SystemTime::now(),
        })
    }
    
    fn reset(&mut self) {
        self.buckets.clear();
        self.total_sum = F::zero();
        self.total_count = 0;
        self.drift_detected = false;
        self.statistics = DriftStatistics {
            n_elements: 0,
            mean: F::zero(),
            variance: F::zero(),
            total_drifts: 0,
            total_warnings: 0,
            last_drift_time: None,
        };
    }
    
    fn get_statistics(&self) -> DriftStatistics<F> {
        self.statistics.clone()
    }
    
    fn detected_drift(&self) -> bool {
        self.drift_detected
    }
    
    fn detected_warning(&self) -> bool {
        false // ADWIN doesn't have warning state
    }
    
    fn name(&self) -> &str {
        "ADWIN"
    }
    
    fn get_config(&self) -> std::collections::HashMap<String, f64> {
        let mut config = std::collections::HashMap::new();
        config.insert("delta".to_string(), self.delta);
        config
    }
}

/// DDM (Drift Detection Method) detector
#[derive(Debug)]
pub struct DdmDetector<F: Float + std::fmt::Debug> {
    /// Warning level parameter
    alpha: f64,
    /// Drift level parameter
    beta: f64,
    /// Current error rate
    error_rate: F,
    /// Standard deviation
    std_dev: F,
    /// Minimum error rate
    min_error_rate: F,
    /// Minimum standard deviation
    min_std_dev: F,
    /// Number of elements
    n_elements: usize,
    /// Drift detected flag
    drift_detected: bool,
    /// Warning detected flag
    warning_detected: bool,
    /// Statistics
    statistics: DriftStatistics<F>,
}

impl<F: Float + std::fmt::Debug + Send + Sync> DdmDetector<F> {
    /// Create a new DDM detector
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            error_rate: F::zero(),
            std_dev: F::zero(),
            min_error_rate: F::infinity(),
            min_std_dev: F::infinity(),
            n_elements: 0,
            drift_detected: false,
            warning_detected: false,
            statistics: DriftStatistics {
                n_elements: 0,
                mean: F::zero(),
                variance: F::zero(),
                total_drifts: 0,
                total_warnings: 0,
                last_drift_time: None,
            },
        }
    }
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum> ConceptDriftDetector<F>
    for DdmDetector<F>
{
    fn add_element(&mut self, error: F) -> Result<DriftDetectionResult> {
        self.n_elements += 1;
        self.statistics.n_elements += 1;
        
        // Update error rate using exponential moving average
        let alpha = F::from(0.1).unwrap(); // Smoothing factor
        if self.n_elements == 1 {
            self.error_rate = error;
        } else {
            self.error_rate = alpha * error + (F::one() - alpha) * self.error_rate;
        }
        
        // Calculate standard deviation
        if self.n_elements > 1 {
            let variance = self.error_rate * (F::one() - self.error_rate) / F::from(self.n_elements).unwrap();
            self.std_dev = variance.sqrt();
        }
        
        // Update statistics
        self.statistics.mean = self.error_rate;
        self.statistics.variance = self.std_dev * self.std_dev;
        
        // Check for new minimum
        if self.error_rate + self.std_dev < self.min_error_rate + self.min_std_dev {
            self.min_error_rate = self.error_rate;
            self.min_std_dev = self.std_dev;
        }
        
        // Reset flags
        self.drift_detected = false;
        self.warning_detected = false;
        
        // Check for drift
        let current_level = self.error_rate + self.std_dev;
        let min_level = self.min_error_rate + self.min_std_dev;
        
        let warning_threshold = min_level + F::from(self.alpha).unwrap() * self.min_std_dev;
        let drift_threshold = min_level + F::from(self.beta).unwrap() * self.min_std_dev;
        
        let status = if current_level > drift_threshold {
            self.drift_detected = true;
            self.statistics.total_drifts += 1;
            self.statistics.last_drift_time = Some(SystemTime::now());
            DriftStatus::Drift
        } else if current_level > warning_threshold {
            self.warning_detected = true;
            self.statistics.total_warnings += 1;
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        };
        
        let confidence = if self.drift_detected {
            1.0 - self.beta
        } else if self.warning_detected {
            1.0 - self.alpha
        } else {
            0.0
        };
        
        Ok(DriftDetectionResult {
            status,
            confidence,
            magnitude: (current_level - min_level).to_f64().unwrap_or(0.0),
            timestamp: SystemTime::now(),
        })
    }
    
    fn reset(&mut self) {
        self.error_rate = F::zero();
        self.std_dev = F::zero();
        self.min_error_rate = F::infinity();
        self.min_std_dev = F::infinity();
        self.n_elements = 0;
        self.drift_detected = false;
        self.warning_detected = false;
        self.statistics = DriftStatistics {
            n_elements: 0,
            mean: F::zero(),
            variance: F::zero(),
            total_drifts: 0,
            total_warnings: 0,
            last_drift_time: None,
        };
    }
    
    fn get_statistics(&self) -> DriftStatistics<F> {
        self.statistics.clone()
    }
    
    fn detected_drift(&self) -> bool {
        self.drift_detected
    }
    
    fn detected_warning(&self) -> bool {
        self.warning_detected
    }
    
    fn name(&self) -> &str {
        "DDM"
    }
    
    fn get_config(&self) -> std::collections::HashMap<String, f64> {
        let mut config = std::collections::HashMap::new();
        config.insert("alpha".to_string(), self.alpha);
        config.insert("beta".to_string(), self.beta);
        config
    }
}

/// Page-Hinkley drift detector
#[derive(Debug)]
pub struct PageHinkleyDetector<F: Float + std::fmt::Debug> {
    /// Minimum number of instances
    min_instances: usize,
    /// Detection threshold
    threshold: f64,
    /// Alpha parameter
    alpha: f64,
    /// Cumulative sum
    cumulative_sum: F,
    /// Minimum cumulative sum
    min_cumulative_sum: F,
    /// Number of elements
    n_elements: usize,
    /// Drift detected flag
    drift_detected: bool,
    /// Statistics
    statistics: DriftStatistics<F>,
}

impl<F: Float + std::fmt::Debug + Send + Sync> PageHinkleyDetector<F> {
    /// Create a new Page-Hinkley detector
    pub fn new(min_instances: usize, threshold: f64, alpha: f64) -> Self {
        Self {
            min_instances,
            threshold,
            alpha,
            cumulative_sum: F::zero(),
            min_cumulative_sum: F::zero(),
            n_elements: 0,
            drift_detected: false,
            statistics: DriftStatistics {
                n_elements: 0,
                mean: F::zero(),
                variance: F::zero(),
                total_drifts: 0,
                total_warnings: 0,
                last_drift_time: None,
            },
        }
    }
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum> ConceptDriftDetector<F>
    for PageHinkleyDetector<F>
{
    fn add_element(&mut self, value: F) -> Result<DriftDetectionResult> {
        self.n_elements += 1;
        self.statistics.n_elements += 1;
        
        // Update cumulative sum
        let alpha_f = F::from(self.alpha).unwrap();
        self.cumulative_sum = self.cumulative_sum + value - alpha_f;
        
        // Update minimum cumulative sum
        if self.cumulative_sum < self.min_cumulative_sum {
            self.min_cumulative_sum = self.cumulative_sum;
        }
        
        // Update statistics
        self.statistics.mean = self.cumulative_sum / F::from(self.n_elements).unwrap();
        
        // Reset drift flag
        self.drift_detected = false;
        
        // Check for drift
        let status = if self.n_elements >= self.min_instances {
            let ph_value = self.cumulative_sum - self.min_cumulative_sum;
            if ph_value > F::from(self.threshold).unwrap() {
                self.drift_detected = true;
                self.statistics.total_drifts += 1;
                self.statistics.last_drift_time = Some(SystemTime::now());
                
                // Reset after drift detection
                self.cumulative_sum = F::zero();
                self.min_cumulative_sum = F::zero();
                
                DriftStatus::Drift
            } else {
                DriftStatus::Stable
            }
        } else {
            DriftStatus::Stable
        };
        
        let confidence = if self.drift_detected { 0.95 } else { 0.0 };
        
        Ok(DriftDetectionResult {
            status,
            confidence,
            magnitude: (self.cumulative_sum - self.min_cumulative_sum).to_f64().unwrap_or(0.0),
            timestamp: SystemTime::now(),
        })
    }
    
    fn reset(&mut self) {
        self.cumulative_sum = F::zero();
        self.min_cumulative_sum = F::zero();
        self.n_elements = 0;
        self.drift_detected = false;
        self.statistics = DriftStatistics {
            n_elements: 0,
            mean: F::zero(),
            variance: F::zero(),
            total_drifts: 0,
            total_warnings: 0,
            last_drift_time: None,
        };
    }
    
    fn get_statistics(&self) -> DriftStatistics<F> {
        self.statistics.clone()
    }
    
    fn detected_drift(&self) -> bool {
        self.drift_detected
    }
    
    fn detected_warning(&self) -> bool {
        false // Page-Hinkley doesn't have warning state
    }
    
    fn name(&self) -> &str {
        "Page-Hinkley"
    }
    
    fn get_config(&self) -> std::collections::HashMap<String, f64> {
        let mut config = std::collections::HashMap::new();
        config.insert("min_instances".to_string(), self.min_instances as f64);
        config.insert("threshold".to_string(), self.threshold);
        config.insert("alpha".to_string(), self.alpha);
        config
    }
}

/// Factory for creating drift detectors
pub struct DriftDetectorFactory;

impl DriftDetectorFactory {
    /// Create a drift detector based on method configuration
    pub fn create_detector<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum + 'static>(
        method: &DriftDetectionMethod,
    ) -> Result<Box<dyn ConceptDriftDetector<F>>> {
        match method {
            DriftDetectionMethod::Adwin { delta } => {
                Ok(Box::new(AdwinDetector::new(*delta)))
            }
            DriftDetectionMethod::Ddm { alpha, beta } => {
                Ok(Box::new(DdmDetector::new(*alpha, *beta)))
            }
            DriftDetectionMethod::PageHinkley { min_instances, threshold, alpha } => {
                Ok(Box::new(PageHinkleyDetector::new(*min_instances, *threshold, *alpha)))
            }
            _ => Err(MetricsError::NotImplemented(
                format!("Drift detection method {:?} not implemented", method)
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adwin_detector() {
        let mut detector = AdwinDetector::<f64>::new(0.002);
        
        // Add stable data
        for i in 0..100 {
            let value = 0.5 + (i as f64) * 0.001; // Slight upward trend
            let result = detector.add_element(value).unwrap();
            assert_eq!(result.status, DriftStatus::Stable);
        }
        
        // Add drift data
        for _ in 0..50 {
            let value = 2.0; // Sudden jump
            detector.add_element(value).unwrap();
        }
        
        // Check statistics
        let stats = detector.get_statistics();
        assert!(stats.n_elements > 0);
    }

    #[test]
    fn test_ddm_detector() {
        let mut detector = DdmDetector::<f64>::new(2.0, 3.0);
        
        // Add stable error rates
        for _ in 0..100 {
            let error = 0.1; // 10% error rate
            let result = detector.add_element(error).unwrap();
            if detector.n_elements > 30 {
                // Should be stable after initial learning
                assert_ne!(result.status, DriftStatus::Drift);
            }
        }
        
        // Add increasing error rates
        for i in 0..20 {
            let error = 0.1 + (i as f64) * 0.05; // Increasing error
            detector.add_element(error).unwrap();
        }
        
        assert_eq!(detector.name(), "DDM");
    }

    #[test]
    fn test_page_hinkley_detector() {
        let mut detector = PageHinkleyDetector::<f64>::new(30, 50.0, 0.005);
        
        // Add stable data
        for _ in 0..50 {
            let value = 0.0;
            let result = detector.add_element(value).unwrap();
            // Early samples might not detect drift due to min_instances
        }
        
        // Add drift data
        for _ in 0..20 {
            let value = 1.0; // Change in mean
            detector.add_element(value).unwrap();
        }
        
        assert_eq!(detector.name(), "Page-Hinkley");
    }

    #[test]
    fn test_drift_detector_factory() {
        let adwin_method = DriftDetectionMethod::Adwin { delta: 0.002 };
        let detector = DriftDetectorFactory::create_detector::<f64>(&adwin_method);
        assert!(detector.is_ok());
        
        let ddm_method = DriftDetectionMethod::Ddm { alpha: 2.0, beta: 3.0 };
        let detector = DriftDetectorFactory::create_detector::<f64>(&ddm_method);
        assert!(detector.is_ok());
        
        let ph_method = DriftDetectionMethod::PageHinkley { 
            min_instances: 30, 
            threshold: 50.0, 
            alpha: 0.005 
        };
        let detector = DriftDetectorFactory::create_detector::<f64>(&ph_method);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_drift_detection_result() {
        let result = DriftDetectionResult {
            status: DriftStatus::Drift,
            confidence: 0.95,
            magnitude: 1.5,
            timestamp: SystemTime::now(),
        };
        
        assert_eq!(result.status, DriftStatus::Drift);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.magnitude, 1.5);
    }

    #[test]
    fn test_drift_statistics() {
        let stats = DriftStatistics::<f64> {
            n_elements: 100,
            mean: 0.5,
            variance: 0.25,
            total_drifts: 2,
            total_warnings: 5,
            last_drift_time: Some(SystemTime::now()),
        };
        
        assert_eq!(stats.n_elements, 100);
        assert_eq!(stats.total_drifts, 2);
        assert_eq!(stats.total_warnings, 5);
    }
}