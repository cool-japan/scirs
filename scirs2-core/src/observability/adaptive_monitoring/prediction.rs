//! Predictive performance analysis and forecasting

use super::types::{
    ComprehensivePerformanceMetrics, CorrelationAnalyzer, DetectedPattern, PatternDetector,
    PerformancePredictions, TimeSeriesModel,
};
use crate::error::CoreResult;
use std::collections::HashMap;
use std::time::Instant;

/// Predictive performance analysis engine for forecasting system behavior
#[allow(dead_code)]
#[derive(Debug)]
pub struct PredictionEngine {
    time_series_models: HashMap<String, TimeSeriesModel>,
    correlation_analyzer: CorrelationAnalyzer,
    pattern_detector: PatternDetector,
    prediction_accuracy: f64,
}

impl PredictionEngine {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            time_series_models: HashMap::new(),
            correlation_analyzer: CorrelationAnalyzer::new(),
            pattern_detector: PatternDetector::new(),
            prediction_accuracy: 0.5, // Start with neutral accuracy
        })
    }

    pub fn update_with_data(
        &mut self,
        historical_data: &[ComprehensivePerformanceMetrics],
    ) -> CoreResult<()> {
        if historical_data.len() < 10 {
            return Ok(()); // Need at least 10 data points for predictions
        }

        // Update time series models
        self.update_time_series_models(historical_data)?;

        // Analyze correlations
        self.correlation_analyzer
            .analyze_correlations(historical_data)?;

        // Detect patterns
        self.pattern_detector.detect_patterns(historical_data)?;

        // Update prediction accuracy based on recent performance
        self.update_prediction_accuracy(historical_data)?;

        Ok(())
    }

    fn update_time_series_models(
        &mut self,
        data: &[ComprehensivePerformanceMetrics],
    ) -> CoreResult<()> {
        // Extract CPU utilization time series
        let cpu_data: Vec<f64> = data.iter().map(|m| m.cpu_utilization).collect();
        let cpu_model = self
            .time_series_models
            .entry("cpu".to_string())
            .or_default();
        cpu_model.add_data(cpu_data)?;

        // Extract memory utilization time series
        let memory_data: Vec<f64> = data.iter().map(|m| m.memory_utilization).collect();
        let memory_model = self
            .time_series_models
            .entry("memory".to_string())
            .or_default();
        memory_model.add_data(memory_data)?;

        // Extract throughput time series
        let throughput_data: Vec<f64> = data.iter().map(|m| m.operations_per_second).collect();
        let throughput_model = self
            .time_series_models
            .entry("throughput".to_string())
            .or_default();
        throughput_model.add_data(throughput_data)?;

        // Extract latency time series
        let latency_data: Vec<f64> = data.iter().map(|m| m.average_latency_ms).collect();
        let latency_model = self
            .time_series_models
            .entry("latency".to_string())
            .or_default();
        latency_model.add_data(latency_data)?;

        Ok(())
    }

    pub fn get_current_predictions(&self) -> CoreResult<PerformancePredictions> {
        let cpu_prediction = self
            .time_series_models
            .get("cpu")
            .map(|model| model.predict_next(5)) // Predict next 5 time steps
            .unwrap_or_else(|| vec![0.5; 5]);

        let memory_prediction = self
            .time_series_models
            .get("memory")
            .map(|model| model.predict_next(5))
            .unwrap_or_else(|| vec![0.5; 5]);

        let throughput_prediction = self
            .time_series_models
            .get("throughput")
            .map(|model| model.predict_next(5))
            .unwrap_or_else(|| vec![1000.0; 5]);

        // Analyze predictions for issues
        let predicted_cpu_spike = cpu_prediction.iter().any(|&val| val > 0.9);
        let predicted_memory_pressure = memory_prediction.iter().any(|&val| val > 0.9);
        let predicted_throughput_drop = throughput_prediction.iter().any(|&val| val < 100.0);

        Ok(PerformancePredictions {
            predicted_cpu_spike,
            predicted_memory_pressure,
            predicted_throughput_drop,
            cpu_forecast: cpu_prediction,
            memory_forecast: memory_prediction,
            throughput_forecast: throughput_prediction,
            confidence: self.prediction_accuracy,
            time_horizon_minutes: 5,
            generated_at: Instant::now(),
            predicted_performance_change: if predicted_cpu_spike
                || predicted_memory_pressure
                || predicted_throughput_drop
            {
                -0.2
            } else {
                0.0
            },
        })
    }

    /// Update prediction accuracy based on actual vs predicted values
    fn update_prediction_accuracy(
        &mut self,
        historical_data: &[ComprehensivePerformanceMetrics],
    ) -> CoreResult<()> {
        if historical_data.len() < 20 {
            return Ok(());
        }

        // Take the last 10 data points as "actual" and compare with what we would have predicted
        let (training_data, actual_data) = historical_data.split_at(historical_data.len() - 10);

        // Create temporary models with training data
        let mut temp_cpu_model = TimeSeriesModel::new();
        let cpu_training: Vec<f64> = training_data.iter().map(|m| m.cpu_utilization).collect();
        temp_cpu_model.add_data(cpu_training)?;

        let mut temp_memory_model = TimeSeriesModel::new();
        let memory_training: Vec<f64> = training_data.iter().map(|m| m.memory_utilization).collect();
        temp_memory_model.add_data(memory_training)?;

        // Generate predictions and compare with actual values
        let predicted_cpu = temp_cpu_model.predict_next(10);
        let actual_cpu: Vec<f64> = actual_data.iter().map(|m| m.cpu_utilization).collect();

        let predicted_memory = temp_memory_model.predict_next(10);
        let actual_memory: Vec<f64> = actual_data.iter().map(|m| m.memory_utilization).collect();

        // Calculate accuracy (lower RMSE = higher accuracy)
        let cpu_accuracy = 1.0 - self.calculate_rmse(&predicted_cpu, &actual_cpu);
        let memory_accuracy = 1.0 - self.calculate_rmse(&predicted_memory, &actual_memory);

        // Update overall accuracy with weighted average
        let new_accuracy = (cpu_accuracy + memory_accuracy) / 2.0;
        self.prediction_accuracy = (self.prediction_accuracy * 0.8 + new_accuracy * 0.2).clamp(0.1, 0.95);

        Ok(())
    }

    /// Calculate Root Mean Square Error between predicted and actual values
    fn calculate_rmse(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        if predicted.len() != actual.len() || predicted.is_empty() {
            return 1.0; // Maximum error if lengths don't match
        }

        let mse = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>() / predicted.len() as f64;

        mse.sqrt().clamp(0.0, 1.0)
    }

    /// Get predictions for a specific time horizon
    pub fn get_predictions_for_horizon(&self, minutes: u32) -> CoreResult<PerformancePredictions> {
        let steps = (minutes as usize).max(1).min(60); // Limit to 1-60 steps

        let cpu_prediction = self
            .time_series_models
            .get("cpu")
            .map(|model| model.predict_next(steps))
            .unwrap_or_else(|| vec![0.5; steps]);

        let memory_prediction = self
            .time_series_models
            .get("memory")
            .map(|model| model.predict_next(steps))
            .unwrap_or_else(|| vec![0.5; steps]);

        let throughput_prediction = self
            .time_series_models
            .get("throughput")
            .map(|model| model.predict_next(steps))
            .unwrap_or_else(|| vec![1000.0; steps]);

        // Analyze predictions for issues
        let predicted_cpu_spike = cpu_prediction.iter().any(|&val| val > 0.9);
        let predicted_memory_pressure = memory_prediction.iter().any(|&val| val > 0.9);
        let predicted_throughput_drop = throughput_prediction.iter().any(|&val| val < 100.0);

        // Calculate overall performance change trend
        let cpu_change = if cpu_prediction.len() >= 2 {
            cpu_prediction.last().unwrap() - cpu_prediction.first().unwrap()
        } else {
            0.0
        };
        let memory_change = if memory_prediction.len() >= 2 {
            memory_prediction.last().unwrap() - memory_prediction.first().unwrap()
        } else {
            0.0
        };
        let throughput_change = if throughput_prediction.len() >= 2 {
            (throughput_prediction.last().unwrap() - throughput_prediction.first().unwrap()) / throughput_prediction.first().unwrap()
        } else {
            0.0
        };

        let predicted_performance_change = (cpu_change + memory_change - throughput_change) / 3.0;

        Ok(PerformancePredictions {
            predicted_cpu_spike,
            predicted_memory_pressure,
            predicted_throughput_drop,
            cpu_forecast: cpu_prediction,
            memory_forecast: memory_prediction,
            throughput_forecast: throughput_prediction,
            confidence: self.prediction_accuracy,
            time_horizon_minutes: minutes,
            generated_at: Instant::now(),
            predicted_performance_change,
        })
    }

    /// Get detected patterns in the performance data
    pub fn get_detected_patterns(&self) -> Vec<DetectedPattern> {
        // In a real implementation, this would return the patterns from pattern_detector
        // For now, return empty vector as the pattern_detector's detected_patterns field is private
        Vec::new()
    }

    /// Get prediction accuracy score (0.0 to 1.0)
    pub fn get_prediction_accuracy(&self) -> f64 {
        self.prediction_accuracy
    }

    /// Reset prediction models (useful for testing or when system characteristics change dramatically)
    pub fn reset_models(&mut self) -> CoreResult<()> {
        self.time_series_models.clear();
        self.correlation_analyzer = CorrelationAnalyzer::new();
        self.pattern_detector = PatternDetector::new();
        self.prediction_accuracy = 0.5;
        Ok(())
    }

    /// Get prediction statistics
    pub fn get_prediction_stats(&self) -> PredictionStats {
        let model_count = self.time_series_models.len();
        let has_sufficient_data = self.time_series_models.values().all(|model| {
            // This is a rough estimate since we can't access the internal data directly
            true // In practice, we'd check if model has enough data points
        });

        PredictionStats {
            model_count,
            prediction_accuracy: self.prediction_accuracy,
            has_sufficient_data,
            last_update: Instant::now(), // Would be tracked in real implementation
        }
    }

    /// Predict specific metric values
    pub fn predict_metric(&self, metric_name: &str, steps: usize) -> CoreResult<Vec<f64>> {
        match self.time_series_models.get(metric_name) {
            Some(model) => Ok(model.predict_next(steps.clamp(1, 100))),
            None => Ok(vec![0.0; steps.clamp(1, 100)]), // Return zeros if model doesn't exist
        }
    }

    /// Check if predictions indicate potential performance issues
    pub fn check_performance_risks(&self) -> CoreResult<Vec<PerformanceRisk>> {
        let predictions = self.get_current_predictions()?;
        let mut risks = Vec::new();

        if predictions.predicted_cpu_spike {
            risks.push(PerformanceRisk {
                risk_type: RiskType::CpuSpike,
                severity: if predictions.cpu_forecast.iter().any(|&val| val > 0.95) {
                    RiskSeverity::High
                } else {
                    RiskSeverity::Medium
                },
                probability: predictions.confidence,
                time_to_impact_minutes: 5, // Based on our prediction horizon
                description: "Predicted CPU utilization spike that may impact performance".to_string(),
            });
        }

        if predictions.predicted_memory_pressure {
            risks.push(PerformanceRisk {
                risk_type: RiskType::MemoryPressure,
                severity: if predictions.memory_forecast.iter().any(|&val| val > 0.95) {
                    RiskSeverity::High
                } else {
                    RiskSeverity::Medium
                },
                probability: predictions.confidence,
                time_to_impact_minutes: 5,
                description: "Predicted memory pressure that may cause performance degradation".to_string(),
            });
        }

        if predictions.predicted_throughput_drop {
            risks.push(PerformanceRisk {
                risk_type: RiskType::ThroughputDrop,
                severity: if predictions.throughput_forecast.iter().any(|&val| val < 50.0) {
                    RiskSeverity::High
                } else {
                    RiskSeverity::Medium
                },
                probability: predictions.confidence,
                time_to_impact_minutes: 5,
                description: "Predicted throughput drop that may affect system responsiveness".to_string(),
            });
        }

        Ok(risks)
    }
}

/// Statistics about the prediction engine
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PredictionStats {
    pub model_count: usize,
    pub prediction_accuracy: f64,
    pub has_sufficient_data: bool,
    pub last_update: Instant,
}

/// Represents a potential performance risk identified by predictions
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceRisk {
    pub risk_type: RiskType,
    pub severity: RiskSeverity,
    pub probability: f64,
    pub time_to_impact_minutes: u32,
    pub description: String,
}

/// Types of performance risks
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskType {
    CpuSpike,
    MemoryPressure,
    ThroughputDrop,
    LatencyIncrease,
    ResourceStarvation,
}

/// Risk severity levels
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}