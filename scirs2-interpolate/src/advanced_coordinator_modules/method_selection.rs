//! Intelligent method selection system for interpolation
//!
//! This module provides AI-driven selection of optimal interpolation methods
//! based on data characteristics, performance requirements, and historical data.

use super::types::*;
use crate::error::InterpolateResult;
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Intelligent method selection system
#[derive(Debug)]
pub struct IntelligentMethodSelector<F: Float + Debug> {
    /// Method performance database
    method_db: HashMap<MethodKey, MethodPerformanceData>,
    /// Current data characteristics
    current_data_profile: Option<DataProfile<F>>,
    /// Method selection model
    selection_model: MethodSelectionModel<F>,
    /// Historical performance data
    performance_history: VecDeque<MethodPerformanceRecord>,
}

/// Method selection model for decision making
#[derive(Debug)]
pub struct MethodSelectionModel<F: Float> {
    /// Feature weights for method selection
    feature_weights: HashMap<String, f64>,
    /// Decision tree for method selection
    decision_tree: Vec<MethodSelectionRule>,
    /// Learning rate for weight updates
    learning_rate: f64,
    /// Model confidence
    model_confidence: F,
}

/// Selection rule for decision tree
#[derive(Debug, Clone)]
pub struct MethodSelectionRule {
    /// Condition for rule activation
    pub condition: MethodSelectionCondition,
    /// Recommended method
    pub method: InterpolationMethodType,
    /// Confidence score
    pub confidence: f64,
    /// Expected accuracy
    pub expected_accuracy: f64,
}

/// Condition for method selection
#[derive(Debug, Clone)]
pub enum MethodSelectionCondition {
    /// Data size based condition
    DataSizeRange { min: usize, max: usize },
    /// Smoothness threshold condition
    SmoothnessThreshold { threshold: f64 },
    /// Noise level condition
    NoiseLevel { max_noise: f64 },
    /// Pattern type condition
    PatternTypeMatch { pattern: DataPatternType },
    /// Accuracy requirement condition
    AccuracyRequirement { min_accuracy: f64 },
    /// Performance requirement condition
    PerformanceRequirement { max_time: f64 },
    /// Composite condition (AND)
    And {
        conditions: Vec<MethodSelectionCondition>,
    },
    /// Composite condition (OR)
    Or {
        conditions: Vec<MethodSelectionCondition>,
    },
}

impl<F: Float + Debug> IntelligentMethodSelector<F> {
    /// Create a new intelligent method selector
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            method_db: HashMap::new(),
            current_data_profile: None,
            selection_model: MethodSelectionModel::new()?,
            performance_history: VecDeque::new(),
        })
    }

    /// Select the best method for given data characteristics
    pub fn select_method(
        &mut self,
        data_profile: &DataProfile<F>,
        performance_targets: &PerformanceTargets,
    ) -> InterpolateResult<InterpolationRecommendation<F>> {
        self.current_data_profile = Some(data_profile.clone());

        // Apply decision tree rules
        let method_scores = self.evaluate_methods(data_profile, performance_targets)?;

        // Select best method based on scores
        let best_method = self.select_best_method(&method_scores)?;

        // Generate recommendation
        self.generate_recommendation(best_method, &method_scores, data_profile)
    }

    /// Update method performance data
    pub fn update_performance(
        &mut self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        performance: &MethodPerformanceData,
    ) -> InterpolateResult<()> {
        let key = self.create_method_key(method, data_profile);

        // Update or insert performance data
        let existing = self
            .method_db
            .entry(key)
            .or_insert_with(|| MethodPerformanceData {
                avg_execution_time: 0.0,
                memory_usage: 0,
                accuracy: 0.0,
                noise_robustness: 0.0,
                sample_count: 0,
                last_update: Instant::now(),
            });

        // Exponential moving average update
        let alpha = 0.1; // Learning rate
        existing.avg_execution_time =
            (1.0 - alpha) * existing.avg_execution_time + alpha * performance.avg_execution_time;
        existing.accuracy = (1.0 - alpha) * existing.accuracy + alpha * performance.accuracy;
        existing.noise_robustness =
            (1.0 - alpha) * existing.noise_robustness + alpha * performance.noise_robustness;
        existing.memory_usage = ((1.0 - alpha) * existing.memory_usage as f64
            + alpha * performance.memory_usage as f64) as usize;
        existing.sample_count += 1;
        existing.last_update = Instant::now();

        // Add to performance history
        self.performance_history.push_back(MethodPerformanceRecord {
            timestamp: Instant::now(),
            method,
            execution_time: performance.avg_execution_time,
            memory_usage: performance.memory_usage,
            accuracy: performance.accuracy,
            data_size: data_profile.size,
            success: true,
        });

        // Limit history size
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Update selection model if needed
        self.update_selection_model()?;

        Ok(())
    }

    /// Get performance statistics for a method
    pub fn get_method_performance(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
    ) -> Option<&MethodPerformanceData> {
        let key = self.create_method_key(method, data_profile);
        self.method_db.get(&key)
    }

    /// Get all available methods for given data characteristics
    pub fn get_available_methods(
        &self,
        data_profile: &DataProfile<F>,
    ) -> Vec<InterpolationMethodType> {
        let mut methods = Vec::new();

        // Add methods based on data characteristics
        methods.push(InterpolationMethodType::Linear);
        methods.push(InterpolationMethodType::CubicSpline);

        if data_profile.size < 10000 {
            methods.push(InterpolationMethodType::Polynomial);
            methods.push(InterpolationMethodType::RadialBasisFunction);
        }

        if data_profile.dimensionality <= 3 {
            methods.push(InterpolationMethodType::Kriging);
            methods.push(InterpolationMethodType::ThinPlateSpline);
        }

        methods.push(InterpolationMethodType::BSpline);
        methods.push(InterpolationMethodType::PchipInterpolation);
        methods.push(InterpolationMethodType::AkimaSpline);

        if data_profile.size >= 1000 {
            methods.push(InterpolationMethodType::NaturalNeighbor);
            methods.push(InterpolationMethodType::ShepardsMethod);
        }

        methods
    }

    /// Evaluate all methods for given data characteristics
    fn evaluate_methods(
        &self,
        data_profile: &DataProfile<F>,
        performance_targets: &PerformanceTargets,
    ) -> InterpolateResult<HashMap<InterpolationMethodType, f64>> {
        let mut scores = HashMap::new();
        let available_methods = self.get_available_methods(data_profile);

        for method in available_methods {
            let score = self.calculate_method_score(method, data_profile, performance_targets)?;
            scores.insert(method, score);
        }

        Ok(scores)
    }

    /// Calculate score for a specific method
    fn calculate_method_score(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        performance_targets: &PerformanceTargets,
    ) -> InterpolateResult<f64> {
        let mut score = 0.0;

        // Base scores for different methods
        let base_score = match method {
            InterpolationMethodType::Linear => 0.6,
            InterpolationMethodType::CubicSpline => 0.8,
            InterpolationMethodType::BSpline => 0.85,
            InterpolationMethodType::RadialBasisFunction => 0.75,
            InterpolationMethodType::Kriging => 0.9,
            InterpolationMethodType::Polynomial => 0.7,
            InterpolationMethodType::PchipInterpolation => 0.8,
            InterpolationMethodType::AkimaSpline => 0.82,
            InterpolationMethodType::ThinPlateSpline => 0.78,
            InterpolationMethodType::NaturalNeighbor => 0.76,
            InterpolationMethodType::ShepardsMethod => 0.65,
            InterpolationMethodType::QuantumInspired => 0.95,
        };

        score += base_score * 0.3; // 30% base score

        // Adjust score based on data characteristics
        let smoothness_factor = data_profile.smoothness.to_f64().unwrap_or(0.5);
        let noise_factor = 1.0 - data_profile.noise_level.to_f64().unwrap_or(0.1);

        score += smoothness_factor * 0.2; // 20% smoothness
        score += noise_factor * 0.2; // 20% noise tolerance

        // Historical performance adjustment
        if let Some(perf_data) = self.get_method_performance(method, data_profile) {
            let accuracy_score = perf_data.accuracy;
            let speed_score = 1.0 / (1.0 + perf_data.avg_execution_time / 1000.0); // Normalize speed

            score += accuracy_score * 0.15; // 15% historical accuracy
            score += speed_score * 0.15; // 15% historical speed
        }

        // Apply decision tree rules
        for rule in &self.selection_model.decision_tree {
            if self.evaluate_condition(&rule.condition, data_profile, performance_targets)? {
                if rule.method == method {
                    score += rule.confidence * 0.1; // 10% rule bonus
                }
            }
        }

        Ok(score.min(1.0)) // Cap at 1.0
    }

    /// Evaluate a selection condition
    fn evaluate_condition(
        &self,
        condition: &MethodSelectionCondition,
        data_profile: &DataProfile<F>,
        performance_targets: &PerformanceTargets,
    ) -> InterpolateResult<bool> {
        match condition {
            MethodSelectionCondition::DataSizeRange { min, max } => {
                Ok(data_profile.size >= *min && data_profile.size <= *max)
            }
            MethodSelectionCondition::SmoothnessThreshold { threshold } => {
                Ok(data_profile.smoothness.to_f64().unwrap_or(0.0) >= *threshold)
            }
            MethodSelectionCondition::NoiseLevel { max_noise } => {
                Ok(data_profile.noise_level.to_f64().unwrap_or(1.0) <= *max_noise)
            }
            MethodSelectionCondition::PatternTypeMatch { pattern } => {
                // This would need access to pattern analysis results
                Ok(true) // Simplified for now
            }
            MethodSelectionCondition::AccuracyRequirement { min_accuracy } => {
                Ok(performance_targets.target_accuracy >= *min_accuracy)
            }
            MethodSelectionCondition::PerformanceRequirement { max_time } => {
                Ok(performance_targets.max_time <= *max_time)
            }
            MethodSelectionCondition::And { conditions } => {
                for cond in conditions {
                    if !self.evaluate_condition(cond, data_profile, performance_targets)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            MethodSelectionCondition::Or { conditions } => {
                for cond in conditions {
                    if self.evaluate_condition(cond, data_profile, performance_targets)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    /// Select the best method from scores
    fn select_best_method(
        &self,
        method_scores: &HashMap<InterpolationMethodType, f64>,
    ) -> InterpolateResult<InterpolationMethodType> {
        let best = method_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(method, _)| *method);

        best.ok_or_else(|| {
            crate::error::InterpolateError::invalid_input("No suitable method found".to_string())
        })
    }

    /// Generate method recommendation
    fn generate_recommendation(
        &self,
        best_method: InterpolationMethodType,
        method_scores: &HashMap<InterpolationMethodType, f64>,
        data_profile: &DataProfile<F>,
    ) -> InterpolateResult<InterpolationRecommendation<F>> {
        let primary_score = method_scores.get(&best_method).copied().unwrap_or(0.0);

        // Get alternative methods
        let mut alternatives: Vec<_> = method_scores
            .iter()
            .filter(|(method, _)| **method != best_method)
            .map(|(method, score)| (*method, *score))
            .collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        alternatives.truncate(3); // Top 3 alternatives

        let primary_recommendation = MethodRecommendation {
            method: best_method,
            confidence: 0.8, // Default confidence score
            expected_performance: ExpectedPerformance {
                accuracy_range: (0.8, 0.95),
                time_range: (100.0, 1000.0),
                memory_range: (1024, 10240),
                performance_score: 0.8,
            },
            parameters: self.get_default_parameters(best_method),
            configuration: self.get_method_configuration(best_method),
            benefits: self.get_method_benefits(best_method),
            limitations: self.get_method_limitations(best_method),
        };

        let alternative_recommendations: Vec<_> = alternatives
            .into_iter()
            .map(|(method, _)| MethodRecommendation {
                method,
                confidence: 0.6, // Lower confidence for alternatives
                expected_performance: ExpectedPerformance {
                    accuracy_range: (0.8, 0.95),
                    time_range: (100.0, 1000.0),
                    memory_range: (1024, 10240),
                    performance_score: 0.8,
                },
                parameters: self.get_default_parameters(method),
                configuration: self.get_method_configuration(method),
                benefits: self.get_method_benefits(method),
                limitations: self.get_method_limitations(method),
            })
            .collect();

        let expected_performance = self.estimate_performance(best_method, data_profile);

        Ok(InterpolationRecommendation {
            primary_method: primary_recommendation,
            alternatives: alternative_recommendations,
            expected_performance,
            confidence: F::from(primary_score).unwrap(),
            reasoning: self.generate_reasoning(best_method, data_profile),
        })
    }

    /// Create method key for database storage
    fn create_method_key(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
    ) -> MethodKey {
        let size_class = match data_profile.size {
            0..=1000 => DataSizeClass::Small,
            1001..=100000 => DataSizeClass::Medium,
            100001..=10000000 => DataSizeClass::Large,
            _ => DataSizeClass::Massive,
        };

        let pattern_type = self.classify_pattern(data_profile);

        MethodKey {
            method_type: method,
            size_class,
            pattern_type,
            dimensionality: data_profile.dimensionality as u8,
        }
    }

    /// Classify data pattern based on profile
    fn classify_pattern(&self, data_profile: &DataProfile<F>) -> DataPatternType {
        let smoothness = data_profile.smoothness.to_f64().unwrap_or(0.5);
        let noise = data_profile.noise_level.to_f64().unwrap_or(0.1);
        let sparsity = data_profile.sparsity.to_f64().unwrap_or(0.0);

        if noise > 0.1 {
            DataPatternType::Noisy
        } else if sparsity > 0.5 {
            DataPatternType::Sparse
        } else if smoothness > 0.8 {
            DataPatternType::Smooth
        } else if smoothness < 0.3 {
            DataPatternType::Irregular
        } else {
            DataPatternType::Structured
        }
    }

    /// Update the selection model based on performance history
    fn update_selection_model(&mut self) -> InterpolateResult<()> {
        // Simplified model update - would be more sophisticated in practice
        if self.performance_history.len() >= 10 {
            // Analyze recent performance trends
            let recent_records: Vec<_> = self.performance_history.iter().rev().take(10).collect();

            // Update model confidence based on recent success rate
            let success_rate = recent_records.iter().filter(|r| r.success).count() as f64
                / recent_records.len() as f64;
            self.selection_model.model_confidence = F::from(success_rate * 0.9 + 0.1).unwrap();
        }

        Ok(())
    }

    /// Get default parameters for a method
    fn get_default_parameters(&self, method: InterpolationMethodType) -> Vec<f64> {
        match method {
            InterpolationMethodType::Linear => vec![],
            InterpolationMethodType::CubicSpline => vec![0.0], // smoothing factor
            InterpolationMethodType::BSpline => vec![3.0],     // degree
            InterpolationMethodType::RadialBasisFunction => vec![1.0], // shape parameter
            InterpolationMethodType::Kriging => vec![1.0, 1.0], // nugget, sill
            InterpolationMethodType::Polynomial => vec![3.0],  // degree
            _ => vec![],
        }
    }

    /// Get method configuration string
    fn get_method_configuration(&self, method: InterpolationMethodType) -> String {
        match method {
            InterpolationMethodType::Linear => "Standard linear interpolation".to_string(),
            InterpolationMethodType::CubicSpline => "Natural boundary conditions".to_string(),
            InterpolationMethodType::BSpline => "Cubic B-spline with clamped ends".to_string(),
            InterpolationMethodType::RadialBasisFunction => {
                "Gaussian RBF with automatic scaling".to_string()
            }
            InterpolationMethodType::Kriging => {
                "Ordinary kriging with exponential variogram".to_string()
            }
            _ => "Default configuration".to_string(),
        }
    }

    /// Get method benefits
    fn get_method_benefits(&self, method: InterpolationMethodType) -> Vec<String> {
        match method {
            InterpolationMethodType::Linear => {
                vec!["Fast".to_string(), "Memory efficient".to_string()]
            }
            InterpolationMethodType::CubicSpline => {
                vec!["Smooth".to_string(), "Good for smooth data".to_string()]
            }
            InterpolationMethodType::BSpline => {
                vec!["Flexible".to_string(), "Numerical stability".to_string()]
            }
            InterpolationMethodType::RadialBasisFunction => vec![
                "Handles scattered data".to_string(),
                "High accuracy".to_string(),
            ],
            InterpolationMethodType::Kriging => vec![
                "Uncertainty quantification".to_string(),
                "Optimal for spatial data".to_string(),
            ],
            _ => vec!["General purpose".to_string()],
        }
    }

    /// Get method limitations
    fn get_method_limitations(&self, method: InterpolationMethodType) -> Vec<String> {
        match method {
            InterpolationMethodType::Linear => {
                vec!["Low accuracy".to_string(), "Not smooth".to_string()]
            }
            InterpolationMethodType::CubicSpline => vec![
                "Sensitive to outliers".to_string(),
                "Requires ordered data".to_string(),
            ],
            InterpolationMethodType::RadialBasisFunction => vec![
                "Computationally expensive".to_string(),
                "Memory intensive".to_string(),
            ],
            _ => vec!["None significant".to_string()],
        }
    }

    /// Estimate performance for a method
    fn estimate_performance(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
    ) -> ExpectedPerformance {
        let base_time = match method {
            InterpolationMethodType::Linear => 10.0,
            InterpolationMethodType::CubicSpline => 100.0,
            InterpolationMethodType::BSpline => 150.0,
            InterpolationMethodType::RadialBasisFunction => 1000.0,
            InterpolationMethodType::Kriging => 2000.0,
            _ => 500.0,
        };

        let time_factor = (data_profile.size as f64).log10();
        let estimated_time = base_time * time_factor;

        ExpectedPerformance {
            accuracy_range: (0.95, 0.99),
            time_range: (estimated_time * 0.5, estimated_time * 2.0),
            memory_range: (data_profile.size * 8, data_profile.size * 32),
            performance_score: 0.8,
        }
    }

    /// Generate reasoning for recommendation
    fn generate_reasoning(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
    ) -> String {
        format!(
            "Selected {} for {} points with dimensionality {} based on data characteristics and performance history.",
            format!("{:?}", method),
            data_profile.size,
            data_profile.dimensionality
        )
    }
}

impl<F: Float> MethodSelectionModel<F> {
    /// Create a new method selection model
    pub fn new() -> InterpolateResult<Self> {
        let mut model = Self {
            feature_weights: HashMap::new(),
            decision_tree: Vec::new(),
            learning_rate: 0.01,
            model_confidence: F::from(0.8).unwrap(),
        };

        // Initialize with default rules
        model.initialize_default_rules();

        Ok(model)
    }

    /// Initialize with default selection rules
    fn initialize_default_rules(&mut self) {
        // Rule 1: Small smooth data -> Cubic Spline
        self.decision_tree.push(MethodSelectionRule {
            condition: MethodSelectionCondition::And {
                conditions: vec![
                    MethodSelectionCondition::DataSizeRange { min: 0, max: 1000 },
                    MethodSelectionCondition::SmoothnessThreshold { threshold: 0.8 },
                ],
            },
            method: InterpolationMethodType::CubicSpline,
            confidence: 0.9,
            expected_accuracy: 0.95,
        });

        // Rule 2: Large data -> Linear interpolation for speed
        self.decision_tree.push(MethodSelectionRule {
            condition: MethodSelectionCondition::DataSizeRange {
                min: 100000,
                max: usize::MAX,
            },
            method: InterpolationMethodType::Linear,
            confidence: 0.7,
            expected_accuracy: 0.8,
        });

        // Rule 3: Noisy data -> B-spline for robustness
        self.decision_tree.push(MethodSelectionRule {
            condition: MethodSelectionCondition::NoiseLevel { max_noise: 0.05 },
            method: InterpolationMethodType::BSpline,
            confidence: 0.85,
            expected_accuracy: 0.9,
        });

        // Rule 4: High accuracy requirement -> Kriging
        self.decision_tree.push(MethodSelectionRule {
            condition: MethodSelectionCondition::AccuracyRequirement { min_accuracy: 0.99 },
            method: InterpolationMethodType::Kriging,
            confidence: 0.9,
            expected_accuracy: 0.99,
        });
    }

    /// Get model confidence
    pub fn get_confidence(&self) -> F {
        self.model_confidence
    }

    /// Update model based on feedback
    pub fn update_model(&mut self, feedback: &[MethodPerformanceRecord]) -> InterpolateResult<()> {
        // Simplified model update
        if feedback.len() >= 5 {
            let success_rate =
                feedback.iter().filter(|r| r.success).count() as f64 / feedback.len() as f64;
            self.model_confidence = F::from(success_rate * 0.9 + 0.1).unwrap();
        }
        Ok(())
    }
}
