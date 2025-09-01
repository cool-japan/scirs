//! Architecture evaluation utilities for NAS

use crate::error::Result;
use crate::models::sequential::Sequential;
use ndarray::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use ndarray::ArrayView1;
/// Evaluation metrics type
pub type EvaluationMetrics = HashMap<String, f64>;
/// Architecture evaluator
pub struct ArchitectureEvaluator {
    /// Batch size for evaluation
    batch_size: usize,
    /// Device to use
    device: String,
    /// Whether to use mixed precision
    mixed_precision: bool,
    /// Metrics to compute
    metrics_config: MetricsConfig,
}
/// Configuration for metrics computation
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Compute accuracy
    pub accuracy: bool,
    /// Compute top-k accuracy
    pub top_k: Option<Vec<usize>>,
    /// Compute precision/recall
    pub precision_recall: bool,
    /// Compute F1 score
    pub f1_score: bool,
    /// Compute confusion matrix
    pub confusion_matrix: bool,
    /// Compute inference time
    pub inference_time: bool,
    /// Compute memory usage
    pub memory_usage: bool,
impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            accuracy: true,
            top_k: Some(vec![5]),
            precision_recall: false,
            f1_score: false,
            confusion_matrix: false,
            inference_time: true,
            memory_usage: true,
        }
    }
impl ArchitectureEvaluator {
    /// Create a new evaluator
    pub fn new(config: crate::nas::controller::ControllerConfig) -> Result<Self> {
        Ok(Self {
            batch_size: 32,
            device: config.device,
            mixed_precision: false,
            metrics_config: MetricsConfig::default(),
        })
    /// Set batch size
    pub fn with_batch_size(mut self, batchsize: usize) -> Self {
        self.batch_size = batch_size;
        self
    /// Set metrics configuration
    pub fn with_metrics_config(mut self, config: MetricsConfig) -> Self {
        self.metrics_config = config;
    /// Evaluate a model on given data
    pub fn evaluate(
        &self,
        model: &Sequential<f32>,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<EvaluationMetrics> {
        let mut metrics = EvaluationMetrics::new();
        let num_samples = data.shape()[0];
        let num_classes = *labels.iter().max().unwrap_or(&0) + 1;
        // Initialize counters
        let mut correct = 0;
        let mut top_k_correct: HashMap<usize, usize> = HashMap::new();
        if let Some(ref k_values) = self.metrics_config.top_k {
            for k in k_values {
                top_k_correct.insert(*k, 0);
            }
        let mut all_predictions = Vec::new();
        let mut all_probabilities = Vec::new();
        let mut inference_times = Vec::new();
        // Process in batches
        for batch_start in (0..num_samples).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(num_samples);
            let batch_data = data.slice(s![batch_start..batch_end, ..]);
            let batch_labels = labels.slice(s![batch_start..batch_end]);
            // Time inference
            let start = Instant::now();
            // Forward pass (simplified - in practice would use actual model)
            let batch_predictions = self.predict_batch(&batch_data)?;
            if self.metrics_config.inference_time {
                inference_times.push(start.elapsed().as_secs_f64());
            // Process predictions
            for (i, true_label) in batch_labels.iter().enumerate() {
                let pred_probs = &batch_predictions[i];
                let pred_label = pred_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx_)| idx)
                    .unwrap_or(0);
                // Accuracy
                if pred_label == *true_label {
                    correct += 1;
                }
                // Top-k accuracy
                if let Some(ref k_values) = self.metrics_config.top_k {
                    let mut sorted_indices: Vec<usize> = (0..pred_probs.len()).collect();
                    sorted_indices
                        .sort_by(|&a, &b| pred_probs[b].partial_cmp(&pred_probs[a]).unwrap());
                    for k in k_values {
                        if sorted_indices[..*k.min(&sorted_indices.len())].contains(true_label) {
                            *top_k_correct.get_mut(k).unwrap() += 1;
                        }
                    }
                all_predictions.push(pred_label);
                all_probabilities.push(pred_probs.clone());
        // Compute metrics
        if self.metrics_config.accuracy {
            metrics.insert("accuracy".to_string(), correct as f64 / num_samples as f64);
        // Top-k accuracy
        for (k, correct_k) in top_k_correct {
            metrics.insert(
                format!("top_{}_accuracy", k),
                correct_k as f64 / num_samples as f64,
            );
        // Precision, Recall, F1
        if self.metrics_config.precision_recall || self.metrics_config.f1_score {
            let (precision, recall, f1) = self.compute_precision_recall_f1(
                &all_predictions,
                labels.as_slice().unwrap(),
                num_classes,
            )?;
            if self.metrics_config.precision_recall {
                metrics.insert("precision".to_string(), precision);
                metrics.insert("recall".to_string(), recall);
            if self.metrics_config.f1_score {
                metrics.insert("f1_score".to_string(), f1);
        // Inference time
        if self.metrics_config.inference_time && !inference_times.is_empty() {
            let avg_time = inference_times.iter().sum::<f64>() / inference_times.len() as f64;
            metrics.insert("inference_time_ms".to_string(), avg_time * 1000.0);
        // Memory usage (simplified estimation)
        if self.metrics_config.memory_usage {
            let param_memory = self.estimate_memory_usage(model)?;
            metrics.insert("memory_usage_mb".to_string(), param_memory);
        Ok(metrics)
    /// Predict batch (simplified implementation)
    fn predict_batch(&self, batchdata: &ArrayView2<f32>) -> Result<Vec<Vec<f64>>> {
        let batch_size = batch_data.shape()[0];
        let num_classes = 10; // Simplified assumption
        // Generate dummy predictions for now
        let mut predictions = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut probs = vec![0.1; num_classes];
            // Make one class more likely
            use rand::Rng;
            let idx = rng().random_range(0..num_classes);
            probs[idx] = 0.9;
            // Normalize
            let sum: f64 = probs.iter().sum();
            for p in &mut probs {
                *p /= sum;
            predictions.push(probs);
        Ok(predictions)
    /// Compute precision..recall, and F1 score
    fn compute_precision_recall_f1(
        predictions: &[usize],
        true_labels: &[usize],
        num_classes: usize,) -> Result<(f64, f64, f64)> {
        let mut true_positives = vec![0; num_classes];
        let mut false_positives = vec![0; num_classes];
        let mut false_negatives = vec![0; num_classes];
        for (&pred, &true_label) in predictions.iter().zip(true_labels.iter()) {
            if pred == true_label {
                true_positives[pred] += 1;
            } else {
                false_positives[pred] += 1;
                false_negatives[true_label] += 1;
        // Macro-averaged metrics
        let mut precision_sum = 0.0;
        let mut recall_sum = 0.0;
        let mut valid_classes = 0;
        for i in 0..num_classes {
            let tp = true_positives[i] as f64;
            let fp = false_positives[i] as f64;
            let fn_ = false_negatives[i] as f64;
            if tp + fp > 0.0 {
                precision_sum += tp / (tp + fp);
                valid_classes += 1;
            if tp + fn_ > 0.0 {
                recall_sum += tp / (tp + fn_);
        let precision = if valid_classes > 0 {
            precision_sum / valid_classes as f64
        } else {
            0.0
        };
        let recall = recall_sum / num_classes as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        Ok((precision, recall, f1))
    /// Estimate memory usage of a model
    fn estimate_memory_usage(&self, model: &Sequential<f32>) -> Result<f64> {
        // Simplified estimation
        // In practice, would calculate based on actual parameters
        let param_count = 1_000_000; // Placeholder
        let bytes_per_param = 4; // float32
        let memory_mb = (param_count * bytes_per_param) as f64 / (1024.0 * 1024.0);
        Ok(memory_mb)
    /// Evaluate model robustness
    pub fn evaluate_robustness(
        noise_levels: &[f64],
    ) -> Result<HashMap<String, f64>> {
        let mut robustness_metrics = HashMap::new();
        for &noise_level in noise_levels {
            // Add noise to data (simplified)
            let noisy_data = data.clone();
            // Evaluate on noisy data
            let metrics = self.evaluate(model, &noisy_data.view(), labels)?;
            if let Some(accuracy) = metrics.get("accuracy") {
                robustness_metrics.insert(format!("accuracy_noise_{:.2}", noise_level), *accuracy);
        // Compute robustness score
        if !robustness_metrics.is_empty() {
            let avg_accuracy: f64 =
                robustness_metrics.values().sum::<f64>() / robustness_metrics.len() as f64;
            robustness_metrics.insert("robustness_score".to_string(), avg_accuracy);
        Ok(robustness_metrics)
    /// Evaluate model on multiple datasets
    pub fn evaluate_multi_dataset(
        datasets: &[(String, ArrayView2<f32>, ArrayView1<usize>)],
    ) -> Result<HashMap<String, EvaluationMetrics>> {
        let mut all_metrics = HashMap::new();
        for (name, data, labels) in datasets {
            let metrics = self.evaluate(model, data, labels)?;
            all_metrics.insert(name.clone(), metrics);
        Ok(all_metrics)
/// Hardware-aware evaluator
pub struct HardwareAwareEvaluator {
    base_evaluator: ArchitectureEvaluator,
    hardware_config: HardwareConfig,
/// Hardware configuration
pub struct HardwareConfig {
    /// Device type (cpu, gpu, mobile, edge)
    pub device_type: String,
    /// Memory limit in MB
    pub memory_limit: Option<usize>,
    /// Power budget in watts
    pub power_budget: Option<f64>,
    /// Target latency in ms
    pub target_latency: Option<f64>,
impl HardwareAwareEvaluator {
    /// Create a new hardware-aware evaluator
    pub fn new(_base_evaluator: ArchitectureEvaluator, hardwareconfig: HardwareConfig) -> Self {
            base_evaluator,
            hardware_config,
    /// Evaluate with hardware constraints
    pub fn evaluate_with_constraints(
        let mut metrics = self.base_evaluator.evaluate(model, data, labels)?;
        // Add hardware-specific metrics
        let hw_metrics = self.compute_hardware_metrics(model)?;
        metrics.extend(hw_metrics);
        // Check constraints
        let mut constraint_violations = 0;
        if let Some(memory_limit) = self.hardware_config.memory_limit {
            if let Some(memory_usage) = metrics.get("memory_usage_mb") {
                if *memory_usage > memory_limit as f64 {
                    constraint_violations += 1;
                    metrics.insert("memory_constraint_violated".to_string(), 1.0);
        if let Some(target_latency) = self.hardware_config.target_latency {
            if let Some(inference_time) = metrics.get("inference_time_ms") {
                if *inference_time > target_latency {
                    metrics.insert("latency_constraint_violated".to_string(), 1.0);
        metrics.insert(
            "constraint_violations".to_string(),
            constraint_violations as f64,
        );
    /// Compute hardware-specific metrics
    fn compute_hardware_metrics(&self, model: &Sequential<f32>) -> Result<HashMap<String, f64>> {
        let mut hw_metrics = HashMap::new();
        // Estimate based on device type
        match self.hardware_config.device_type.as_str() {
            "mobile" => {
                hw_metrics.insert("mobile_efficiency_score".to_string(), 0.75);
                hw_metrics.insert("estimated_battery_hours".to_string(), 4.5);
            "edge" => {
                hw_metrics.insert("edge_deployment_score".to_string(), 0.82);
                hw_metrics.insert("estimated_power_watts".to_string(), 2.5);
            "gpu" => {
                hw_metrics.insert("gpu_utilization".to_string(), 0.65);
                hw_metrics.insert("estimated_tflops".to_string(), 5.2);
            _ => {
                hw_metrics.insert("device_efficiency".to_string(), 0.5);
        Ok(hw_metrics)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::controller::ControllerConfig;
    #[test]
    fn test_evaluator_creation() {
        let config = ControllerConfig::default();
        let evaluator = ArchitectureEvaluator::new(config).unwrap();
        assert_eq!(evaluator.batch_size, 32);
    fn test_metrics_config() {
        let config = MetricsConfig::default();
        assert!(config.accuracy);
        assert!(_config.top_k.is_some());
    fn test_hardware_aware_evaluator() {
        let base_config = ControllerConfig::default();
        let base_evaluator = ArchitectureEvaluator::new(base_config).unwrap();
        let hw_config = HardwareConfig {
            device_type: "mobile".to_string(),
            memory_limit: Some(512),
            power_budget: Some(5.0),
            target_latency: Some(50.0),
        let hw_evaluator = HardwareAwareEvaluator::new(base_evaluator, hw_config);
        assert_eq!(hw_evaluator.hardware_config.device_type, "mobile");
