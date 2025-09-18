//! Convergence analysis and detection for optimization coordinator
//!
//! This module provides comprehensive convergence analysis capabilities including
//! multiple detection methods, convergence evidence tracking, and early stopping
//! mechanisms for optimization processes.

use super::config::*;
use super::performance::PerformanceSnapshot;
use crate::OptimizerError as OptimError;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

/// Result type for convergence operations
type Result<T> = std::result::Result<T, OptimError>;

/// Convergence analyzer for optimization monitoring
#[derive(Debug)]
pub struct ConvergenceAnalyzer<T: Float + Send + Sync + Debug> {
    /// Configuration
    config: ConvergenceAnalysisConfig,

    /// Convergence detectors
    detectors: Vec<ConvergenceDetector<T>>,

    /// Analysis history
    analysis_history: VecDeque<ConvergenceAnalysis<T>>,

    /// Current convergence state
    current_state: ConvergenceState<T>,

    /// Early stopping controller
    early_stopping: EarlyStoppingController<T>,

    /// Detection evidence
    evidence_tracker: ConvergenceEvidenceTracker<T>,
}

/// Individual convergence detector
#[derive(Debug)]
pub struct ConvergenceDetector<T: Float + Send + Sync + Debug> {
    /// Detection method
    method: ConvergenceDetectionMethod,

    /// Method parameters
    parameters: ConvergenceParameters<T>,

    /// Detection history
    detection_history: VecDeque<ConvergenceDetection<T>>,

    /// Current detection state
    current_detection: Option<ConvergenceDetection<T>>,

    /// Method-specific state
    method_state: DetectionMethodState<T>,
}

/// Convergence detection result
#[derive(Debug, Clone)]
pub struct ConvergenceDetection<T: Float> {
    /// Timestamp of detection
    pub timestamp: SystemTime,

    /// Training step
    pub step: usize,

    /// Detection method
    pub method: ConvergenceDetectionMethod,

    /// Convergence detected
    pub converged: bool,

    /// Confidence in detection
    pub confidence: T,

    /// Evidence supporting detection
    pub evidence: ConvergenceEvidence<T>,

    /// Detection metrics
    pub metrics: ConvergenceMetrics<T>,
}

/// Evidence supporting convergence detection
#[derive(Debug, Clone)]
pub struct ConvergenceEvidence<T: Float> {
    /// Loss-based evidence
    pub loss_evidence: LossEvidence<T>,

    /// Gradient-based evidence
    pub gradient_evidence: GradientEvidence<T>,

    /// Parameter-based evidence
    pub parameter_evidence: ParameterEvidence<T>,

    /// Statistical evidence
    pub statistical_evidence: StatisticalEvidence<T>,

    /// Validation-based evidence
    pub validation_evidence: ValidationEvidence<T>,
}

/// Loss-based convergence evidence
#[derive(Debug, Clone)]
pub struct LossEvidence<T: Float> {
    /// Loss change rate
    pub loss_change_rate: T,

    /// Loss stability
    pub loss_stability: T,

    /// Loss plateau duration
    pub plateau_duration: usize,

    /// Relative improvement
    pub relative_improvement: T,

    /// Absolute improvement
    pub absolute_improvement: T,
}

/// Gradient-based convergence evidence
#[derive(Debug, Clone)]
pub struct GradientEvidence<T: Float> {
    /// Gradient norm
    pub gradient_norm: T,

    /// Gradient norm change rate
    pub gradient_norm_change_rate: T,

    /// Gradient stability
    pub gradient_stability: T,

    /// Gradient direction consistency
    pub direction_consistency: T,
}

/// Parameter-based convergence evidence
#[derive(Debug, Clone)]
pub struct ParameterEvidence<T: Float> {
    /// Parameter change magnitude
    pub parameter_change_magnitude: T,

    /// Parameter change rate
    pub parameter_change_rate: T,

    /// Parameter stability
    pub parameter_stability: T,

    /// Parameter convergence ratio
    pub convergence_ratio: T,
}

/// Statistical convergence evidence
#[derive(Debug, Clone)]
pub struct StatisticalEvidence<T: Float> {
    /// Statistical significance
    pub p_value: T,

    /// Effect size
    pub effect_size: T,

    /// Confidence interval
    pub confidence_interval: (T, T),

    /// Test statistic
    pub test_statistic: T,

    /// Test type
    pub test_type: StatisticalTestType,
}

/// Statistical test types
#[derive(Debug, Clone, Copy)]
pub enum StatisticalTestType {
    TTest,
    WilcoxonSignedRank,
    KolmogorovSmirnov,
    AndersonDarling,
    MannWhitney,
}

/// Validation-based convergence evidence
#[derive(Debug, Clone)]
pub struct ValidationEvidence<T: Float> {
    /// Validation loss change
    pub val_loss_change: T,

    /// Validation metric change
    pub val_metric_change: T,

    /// Generalization gap
    pub generalization_gap: T,

    /// Overfitting indicator
    pub overfitting_indicator: T,
}

/// Convergence detection metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics<T: Float> {
    /// Detection sensitivity
    pub sensitivity: T,

    /// Detection specificity
    pub specificity: T,

    /// False positive rate
    pub false_positive_rate: T,

    /// False negative rate
    pub false_negative_rate: T,

    /// Detection latency
    pub detection_latency: usize,
}

/// Overall convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis<T: Float> {
    /// Analysis timestamp
    pub timestamp: SystemTime,

    /// Training step
    pub step: usize,

    /// Individual detection results
    pub detections: Vec<ConvergenceDetection<T>>,

    /// Consensus result
    pub consensus: ConvergenceConsensus<T>,

    /// Analysis summary
    pub summary: ConvergenceAnalysisSummary<T>,

    /// Recommendations
    pub recommendations: Vec<ConvergenceRecommendation>,
}

/// Convergence consensus across methods
#[derive(Debug, Clone)]
pub struct ConvergenceConsensus<T: Float> {
    /// Overall convergence probability
    pub convergence_probability: T,

    /// Consensus confidence
    pub consensus_confidence: T,

    /// Voting results
    pub voting_results: HashMap<ConvergenceDetectionMethod, bool>,

    /// Weighted score
    pub weighted_score: T,

    /// Method agreement
    pub method_agreement: T,
}

/// Convergence analysis summary
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysisSummary<T: Float> {
    /// Time to convergence estimate
    pub estimated_convergence_time: Option<Duration>,

    /// Convergence quality score
    pub convergence_quality: T,

    /// Stability assessment
    pub stability_assessment: StabilityAssessment<T>,

    /// Performance trajectory
    pub performance_trajectory: TrajectoryAnalysis<T>,
}

/// Stability assessment
#[derive(Debug, Clone)]
pub struct StabilityAssessment<T: Float> {
    /// Loss stability score
    pub loss_stability: T,

    /// Gradient stability score
    pub gradient_stability: T,

    /// Parameter stability score
    pub parameter_stability: T,

    /// Overall stability score
    pub overall_stability: T,

    /// Stability trend
    pub stability_trend: StabilityTrend,
}

/// Stability trend indicators
#[derive(Debug, Clone, Copy)]
pub enum StabilityTrend {
    Improving,
    Stable,
    Degrading,
    Oscillating,
    Unknown,
}

/// Performance trajectory analysis
#[derive(Debug, Clone)]
pub struct TrajectoryAnalysis<T: Float> {
    /// Trajectory type
    pub trajectory_type: TrajectoryType,

    /// Convergence rate
    pub convergence_rate: T,

    /// Trajectory smoothness
    pub trajectory_smoothness: T,

    /// Phase transitions
    pub phase_transitions: Vec<PhaseTransition<T>>,
}

/// Performance trajectory types
#[derive(Debug, Clone, Copy)]
pub enum TrajectoryType {
    Monotonic,
    Oscillatory,
    Stepped,
    Exponential,
    Linear,
    Plateau,
    Unknown,
}

/// Phase transition in optimization
#[derive(Debug, Clone)]
pub struct PhaseTransition<T: Float> {
    /// Transition step
    pub step: usize,

    /// Transition type
    pub transition_type: TransitionType,

    /// Performance change
    pub performance_change: T,

    /// Transition confidence
    pub confidence: T,
}

/// Types of phase transitions
#[derive(Debug, Clone, Copy)]
pub enum TransitionType {
    LinearToExponential,
    ExponentialToLinear,
    FastToSlow,
    SlowToFast,
    StableToOscillatory,
    OscillatoryToStable,
    NoiseReduction,
    NoiseIncrease,
}

/// Convergence recommendations
#[derive(Debug, Clone)]
pub enum ConvergenceRecommendation {
    /// Continue training
    ContinueTraining,

    /// Early stopping recommended
    EarlyStop,

    /// Adjust learning rate
    AdjustLearningRate { factor: f64 },

    /// Increase batch size
    IncreaseBatchSize { new_size: usize },

    /// Add regularization
    AddRegularization { l2_lambda: f64 },

    /// Change optimizer
    ChangeOptimizer { optimizer: String },

    /// Reduce model complexity
    ReduceComplexity,

    /// Increase model capacity
    IncreaseCapacity,
}

/// Current convergence state
#[derive(Debug, Clone)]
pub struct ConvergenceState<T: Float> {
    /// Is converged
    pub converged: bool,

    /// Convergence confidence
    pub confidence: T,

    /// Steps since convergence detected
    pub steps_since_detection: usize,

    /// Current phase
    pub current_phase: OptimizationPhase,

    /// State timestamp
    pub timestamp: SystemTime,
}

/// Optimization phases
#[derive(Debug, Clone, Copy)]
pub enum OptimizationPhase {
    Initialization,
    FastLearning,
    SlowLearning,
    FineTuning,
    Converged,
    Diverged,
    Oscillating,
}

/// Early stopping controller
#[derive(Debug)]
pub struct EarlyStoppingController<T: Float + Send + Sync + Debug> {
    /// Configuration
    config: EarlyStoppingConfig<T>,

    /// Monitoring history
    monitoring_history: VecDeque<T>,

    /// Best metric value
    best_metric: Option<T>,

    /// Steps since improvement
    steps_since_improvement: usize,

    /// Early stopping triggered
    triggered: bool,

    /// Best weights (if configured to restore)
    best_weights: Option<Array1<T>>,
}

/// Convergence evidence tracker
#[derive(Debug)]
pub struct ConvergenceEvidenceTracker<T: Float + Send + Sync + Debug> {
    /// Evidence history
    evidence_history: VecDeque<ConvergenceEvidence<T>>,

    /// Evidence weights
    evidence_weights: HashMap<String, T>,

    /// Cumulative evidence score
    cumulative_score: T,

    /// Evidence trends
    evidence_trends: HashMap<String, Vec<T>>,
}

/// Method-specific detection state
#[derive(Debug)]
pub enum DetectionMethodState<T: Float> {
    /// Loss-based state
    LossBased {
        recent_losses: VecDeque<T>,
        loss_window: VecDeque<T>,
        improvement_history: VecDeque<T>,
    },

    /// Gradient norm state
    GradientNorm {
        gradient_norms: VecDeque<T>,
        norm_changes: VecDeque<T>,
    },

    /// Parameter change state
    ParameterChange {
        parameter_changes: VecDeque<T>,
        parameter_history: VecDeque<Array1<T>>,
    },

    /// Validation metric state
    ValidationMetric {
        validation_metrics: VecDeque<T>,
        generalization_gaps: VecDeque<T>,
    },

    /// Statistical test state
    StatisticalTest {
        test_statistics: VecDeque<T>,
        p_values: VecDeque<T>,
    },

    /// ML-based state
    MLBased {
        features: VecDeque<Array1<T>>,
        predictions: VecDeque<T>,
    },
}

impl<T: Float + Send + Sync + Debug + Default + Clone> ConvergenceAnalyzer<T> {
    /// Create new convergence analyzer
    pub fn new(config: ConvergenceAnalysisConfig) -> Result<Self> {
        let mut detectors = Vec::new();

        // Create detectors for each configured method
        for method in &config.detection_methods {
            let detector = ConvergenceDetector::new(*method, config.detection_parameters.clone())?;
            detectors.push(detector);
        }

        Ok(Self {
            config: config.clone(),
            detectors,
            analysis_history: VecDeque::new(),
            current_state: ConvergenceState::default(),
            early_stopping: EarlyStoppingController::new(config.early_stopping_config)?,
            evidence_tracker: ConvergenceEvidenceTracker::new(),
        })
    }

    /// Analyze convergence with new performance data
    pub fn analyze_convergence(&mut self, snapshot: &PerformanceSnapshot<T>) -> Result<ConvergenceAnalysis<T>> {
        // Run individual detection methods
        let mut detections = Vec::new();
        for detector in &mut self.detectors {
            let detection = detector.detect_convergence(snapshot)?;
            if let Some(detection) = detection {
                detections.push(detection);
            }
        }

        // Build consensus
        let consensus = self.build_consensus(&detections)?;

        // Update evidence tracker
        if !detections.is_empty() {
            let evidence = &detections[0].evidence; // Use first detection's evidence
            self.evidence_tracker.update_evidence(evidence.clone())?;
        }

        // Generate analysis summary
        let summary = self.generate_summary(&detections, &consensus)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&detections, &consensus, &summary)?;

        // Create analysis result
        let analysis = ConvergenceAnalysis {
            timestamp: SystemTime::now(),
            step: snapshot.step,
            detections,
            consensus: consensus.clone(),
            summary,
            recommendations,
        };

        // Update convergence state
        self.update_convergence_state(&consensus, snapshot.step)?;

        // Check early stopping
        if self.config.enable_early_stopping {
            self.early_stopping.update(&snapshot.metrics.val_loss.unwrap_or(snapshot.loss))?;
        }

        // Add to history
        if self.analysis_history.len() >= self.config.analysis_window_size {
            self.analysis_history.pop_front();
        }
        self.analysis_history.push_back(analysis.clone());

        Ok(analysis)
    }

    /// Check if early stopping should be triggered
    pub fn should_early_stop(&self) -> bool {
        self.early_stopping.should_stop()
    }

    /// Get current convergence state
    pub fn get_convergence_state(&self) -> &ConvergenceState<T> {
        &self.current_state
    }

    /// Get convergence history
    pub fn get_convergence_history(&self) -> &VecDeque<ConvergenceAnalysis<T>> {
        &self.analysis_history
    }

    /// Build consensus from individual detections
    fn build_consensus(&self, detections: &[ConvergenceDetection<T>]) -> Result<ConvergenceConsensus<T>> {
        if detections.is_empty() {
            return Ok(ConvergenceConsensus::default());
        }

        let mut voting_results = HashMap::new();
        let mut confidence_sum = T::zero();
        let mut converged_count = 0;

        for detection in detections {
            voting_results.insert(detection.method, detection.converged);
            confidence_sum = confidence_sum + detection.confidence;
            if detection.converged {
                converged_count += 1;
            }
        }

        let convergence_probability = num_traits::cast::cast(converged_count).unwrap_or_else(|| T::zero()) / T::from(detections.len()).unwrap();
        let consensus_confidence = confidence_sum / T::from(detections.len()).unwrap();
        let method_agreement = self.calculate_method_agreement(&voting_results)?;

        Ok(ConvergenceConsensus {
            convergence_probability,
            consensus_confidence,
            voting_results,
            weighted_score: convergence_probability * consensus_confidence,
            method_agreement,
        })
    }

    /// Calculate agreement between methods
    fn calculate_method_agreement(&self, voting_results: &HashMap<ConvergenceDetectionMethod, bool>) -> Result<T> {
        if voting_results.len() < 2 {
            return Ok(T::one());
        }

        let values: Vec<bool> = voting_results.values().cloned().collect();
        let agree_count = values.windows(2).filter(|w| w[0] == w[1]).count();
        let total_pairs = values.len() - 1;

        Ok(num_traits::cast::cast(agree_count).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(total_pairs).unwrap_or_else(|| T::zero()))
    }

    /// Generate analysis summary
    fn generate_summary(
        &self,
        detections: &[ConvergenceDetection<T>],
        consensus: &ConvergenceConsensus<T>,
    ) -> Result<ConvergenceAnalysisSummary<T>> {
        // Estimate convergence time
        let estimated_convergence_time = if consensus.convergence_probability > num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()) {
            Some(Duration::from_secs(0)) // Already converged
        } else {
            self.estimate_convergence_time(detections)?
        };

        // Calculate convergence quality
        let convergence_quality = consensus.consensus_confidence * consensus.method_agreement;

        // Assess stability
        let stability_assessment = self.assess_stability(detections)?;

        // Analyze trajectory
        let performance_trajectory = self.analyze_trajectory()?;

        Ok(ConvergenceAnalysisSummary {
            estimated_convergence_time,
            convergence_quality,
            stability_assessment,
            performance_trajectory,
        })
    }

    /// Generate convergence recommendations
    fn generate_recommendations(
        &self,
        detections: &[ConvergenceDetection<T>],
        consensus: &ConvergenceConsensus<T>,
        summary: &ConvergenceAnalysisSummary<T>,
    ) -> Result<Vec<ConvergenceRecommendation>> {
        let mut recommendations = Vec::new();

        // High convergence probability
        if consensus.convergence_probability > num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()) {
            recommendations.push(ConvergenceRecommendation::EarlyStop);
        }
        // Low convergence probability
        else if consensus.convergence_probability < num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()) {
            recommendations.push(ConvergenceRecommendation::ContinueTraining);

            // Check for oscillatory behavior
            if matches!(summary.stability_assessment.stability_trend, StabilityTrend::Oscillating) {
                recommendations.push(ConvergenceRecommendation::AdjustLearningRate { factor: 0.5 });
            }

            // Check for slow convergence
            if summary.performance_trajectory.convergence_rate < num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()) {
                recommendations.push(ConvergenceRecommendation::AdjustLearningRate { factor: 1.5 });
            }
        }
        // Medium convergence probability
        else {
            recommendations.push(ConvergenceRecommendation::ContinueTraining);
        }

        Ok(recommendations)
    }

    /// Update convergence state
    fn update_convergence_state(&mut self, consensus: &ConvergenceConsensus<T>, step: usize) -> Result<()> {
        let converged = consensus.convergence_probability > num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero());

        if converged && !self.current_state.converged {
            // Newly converged
            self.current_state.converged = true;
            self.current_state.steps_since_detection = 0;
        } else if converged {
            // Still converged
            self.current_state.steps_since_detection += 1;
        } else {
            // Not converged
            self.current_state.converged = false;
            self.current_state.steps_since_detection = 0;
        }

        self.current_state.confidence = consensus.consensus_confidence;
        self.current_state.timestamp = SystemTime::now();

        Ok(())
    }

    /// Estimate time to convergence
    fn estimate_convergence_time(&self, _detections: &[ConvergenceDetection<T>]) -> Result<Option<Duration>> {
        // Simplified estimation
        Ok(Some(Duration::from_secs(3600))) // 1 hour estimate
    }

    /// Assess stability
    fn assess_stability(&self, _detections: &[ConvergenceDetection<T>]) -> Result<StabilityAssessment<T>> {
        Ok(StabilityAssessment {
            loss_stability: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            gradient_stability: num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero()),
            parameter_stability: num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()),
            overall_stability: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            stability_trend: StabilityTrend::Stable,
        })
    }

    /// Analyze performance trajectory
    fn analyze_trajectory(&self) -> Result<TrajectoryAnalysis<T>> {
        Ok(TrajectoryAnalysis {
            trajectory_type: TrajectoryType::Monotonic,
            convergence_rate: num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()),
            trajectory_smoothness: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            phase_transitions: Vec::new(),
        })
    }
}

impl<T: Float + Send + Sync + Debug + Default + Clone> ConvergenceDetector<T> {
    /// Create new convergence detector
    pub fn new(method: ConvergenceDetectionMethod, parameters: ConvergenceParameters<T>) -> Result<Self> {
        let method_state = match method {
            ConvergenceDetectionMethod::LossBased => DetectionMethodState::LossBased {
                recent_losses: VecDeque::new(),
                loss_window: VecDeque::new(),
                improvement_history: VecDeque::new(),
            },
            ConvergenceDetectionMethod::GradientNorm => DetectionMethodState::GradientNorm {
                gradient_norms: VecDeque::new(),
                norm_changes: VecDeque::new(),
            },
            ConvergenceDetectionMethod::ParameterChange => DetectionMethodState::ParameterChange {
                parameter_changes: VecDeque::new(),
                parameter_history: VecDeque::new(),
            },
            ConvergenceDetectionMethod::ValidationMetric => DetectionMethodState::ValidationMetric {
                validation_metrics: VecDeque::new(),
                generalization_gaps: VecDeque::new(),
            },
            ConvergenceDetectionMethod::StatisticalTest => DetectionMethodState::StatisticalTest {
                test_statistics: VecDeque::new(),
                p_values: VecDeque::new(),
            },
            ConvergenceDetectionMethod::MLBased => DetectionMethodState::MLBased {
                features: VecDeque::new(),
                predictions: VecDeque::new(),
            },
        };

        Ok(Self {
            method,
            parameters,
            detection_history: VecDeque::new(),
            current_detection: None,
            method_state,
        })
    }

    /// Detect convergence with new snapshot
    pub fn detect_convergence(&mut self, snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        let detection = match self.method {
            ConvergenceDetectionMethod::LossBased => self.detect_loss_based(snapshot)?,
            ConvergenceDetectionMethod::GradientNorm => self.detect_gradient_norm_based(snapshot)?,
            ConvergenceDetectionMethod::ParameterChange => self.detect_parameter_change_based(snapshot)?,
            ConvergenceDetectionMethod::ValidationMetric => self.detect_validation_metric_based(snapshot)?,
            ConvergenceDetectionMethod::StatisticalTest => self.detect_statistical_test_based(snapshot)?,
            ConvergenceDetectionMethod::MLBased => self.detect_ml_based(snapshot)?,
        };

        if let Some(detection) = &detection {
            self.current_detection = Some(detection.clone());
            self.detection_history.push_back(detection.clone());

            // Limit history size
            if self.detection_history.len() > 100 {
                self.detection_history.pop_front();
            }
        }

        Ok(detection)
    }

    /// Loss-based convergence detection
    fn detect_loss_based(&mut self, snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        if let DetectionMethodState::LossBased { recent_losses, loss_window, improvement_history } = &mut self.method_state {
            recent_losses.push_back(snapshot.loss);
            if recent_losses.len() > self.parameters.window_size {
                recent_losses.pop_front();
            }

            if recent_losses.len() < self.parameters.window_size {
                return Ok(None);
            }

            // Calculate loss change rate
            let first_loss = recent_losses.front().unwrap();
            let last_loss = recent_losses.back().unwrap();
            let loss_change = (*first_loss - *last_loss).abs();
            let relative_change = loss_change / *first_loss;

            // Check convergence criteria
            let converged = relative_change < self.parameters.loss_tolerance;
            let confidence = if converged {
                T::one() - relative_change / self.parameters.loss_tolerance
            } else {
                relative_change / self.parameters.loss_tolerance
            };

            let evidence = ConvergenceEvidence {
                loss_evidence: LossEvidence {
                    loss_change_rate: relative_change,
                    loss_stability: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()), // Simplified
                    plateau_duration: 0,
                    relative_improvement: relative_change,
                    absolute_improvement: loss_change,
                },
                gradient_evidence: GradientEvidence::default(),
                parameter_evidence: ParameterEvidence::default(),
                statistical_evidence: StatisticalEvidence::default(),
                validation_evidence: ValidationEvidence::default(),
            };

            Ok(Some(ConvergenceDetection {
                timestamp: SystemTime::now(),
                step: snapshot.step,
                method: self.method,
                converged,
                confidence,
                evidence,
                metrics: ConvergenceMetrics::default(),
            }))
        } else {
            Err(OptimError::InvalidState("Invalid method state for loss-based detection".to_string()))
        }
    }

    /// Gradient norm-based convergence detection
    fn detect_gradient_norm_based(&mut self, snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        if let Some(gradient_norm) = snapshot.gradient_norm {
            if let DetectionMethodState::GradientNorm { gradient_norms, norm_changes } = &mut self.method_state {
                gradient_norms.push_back(gradient_norm);
                if gradient_norms.len() > self.parameters.window_size {
                    gradient_norms.pop_front();
                }

                if gradient_norms.len() < 2 {
                    return Ok(None);
                }

                // Calculate gradient norm change
                let prev_norm = gradient_norms[gradient_norms.len() - 2];
                let norm_change = (gradient_norm - prev_norm).abs();
                norm_changes.push_back(norm_change);

                if norm_changes.len() > self.parameters.window_size {
                    norm_changes.pop_front();
                }

                // Check convergence criteria
                let converged = gradient_norm < self.parameters.gradient_tolerance;
                let confidence = if converged {
                    T::one() - gradient_norm / self.parameters.gradient_tolerance
                } else {
                    self.parameters.gradient_tolerance / gradient_norm
                };

                let evidence = ConvergenceEvidence {
                    loss_evidence: LossEvidence::default(),
                    gradient_evidence: GradientEvidence {
                        gradient_norm,
                        gradient_norm_change_rate: norm_change,
                        gradient_stability: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
                        direction_consistency: num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()),
                    },
                    parameter_evidence: ParameterEvidence::default(),
                    statistical_evidence: StatisticalEvidence::default(),
                    validation_evidence: ValidationEvidence::default(),
                };

                return Ok(Some(ConvergenceDetection {
                    timestamp: SystemTime::now(),
                    step: snapshot.step,
                    method: self.method,
                    converged,
                    confidence,
                    evidence,
                    metrics: ConvergenceMetrics::default(),
                }));
            }
        }

        Ok(None)
    }

    /// Parameter change-based convergence detection
    fn detect_parameter_change_based(&mut self, _snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        // Simplified implementation - would need actual parameter access
        Ok(None)
    }

    /// Validation metric-based convergence detection
    fn detect_validation_metric_based(&mut self, snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        if let Some(val_loss) = snapshot.metrics.val_loss {
            // Simplified validation-based detection
            let converged = val_loss < self.parameters.loss_tolerance;
            let confidence = if converged { num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()) } else { num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()) };

            let evidence = ConvergenceEvidence {
                loss_evidence: LossEvidence::default(),
                gradient_evidence: GradientEvidence::default(),
                parameter_evidence: ParameterEvidence::default(),
                statistical_evidence: StatisticalEvidence::default(),
                validation_evidence: ValidationEvidence {
                    val_loss_change: val_loss - snapshot.loss,
                    val_metric_change: T::zero(),
                    generalization_gap: val_loss - snapshot.loss,
                    overfitting_indicator: if val_loss > snapshot.loss { T::one() } else { T::zero() },
                },
            };

            return Ok(Some(ConvergenceDetection {
                timestamp: SystemTime::now(),
                step: snapshot.step,
                method: self.method,
                converged,
                confidence,
                evidence,
                metrics: ConvergenceMetrics::default(),
            }));
        }

        Ok(None)
    }

    /// Statistical test-based convergence detection
    fn detect_statistical_test_based(&mut self, _snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        // Simplified implementation
        Ok(None)
    }

    /// ML-based convergence detection
    fn detect_ml_based(&mut self, _snapshot: &PerformanceSnapshot<T>) -> Result<Option<ConvergenceDetection<T>>> {
        // Simplified implementation
        Ok(None)
    }
}

impl<T: Float + Send + Sync + Debug + Default + Clone> EarlyStoppingController<T> {
    /// Create new early stopping controller
    pub fn new(config: EarlyStoppingConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            monitoring_history: VecDeque::new(),
            best_metric: None,
            steps_since_improvement: 0,
            triggered: false,
            best_weights: None,
        })
    }

    /// Update with new metric value
    pub fn update(&mut self, metric_value: &T) -> Result<()> {
        self.monitoring_history.push_back(*metric_value);

        // Check for improvement
        let improved = match self.best_metric {
            Some(best) => match self.config.mode {
                EarlyStoppingMode::Min => *metric_value < best - self.config.min_delta,
                EarlyStoppingMode::Max => *metric_value > best + self.config.min_delta,
            },
            None => true,
        };

        if improved {
            self.best_metric = Some(*metric_value);
            self.steps_since_improvement = 0;
        } else {
            self.steps_since_improvement += 1;
        }

        // Check if early stopping should be triggered
        if self.steps_since_improvement >= self.config.patience {
            self.triggered = true;
        }

        Ok(())
    }

    /// Check if early stopping should be triggered
    pub fn should_stop(&self) -> bool {
        self.triggered && self.config.enabled
    }

    /// Reset early stopping state
    pub fn reset(&mut self) {
        self.best_metric = None;
        self.steps_since_improvement = 0;
        self.triggered = false;
        self.monitoring_history.clear();
    }
}

impl<T: Float + Send + Sync + Debug + Default + Clone> ConvergenceEvidenceTracker<T> {
    /// Create new evidence tracker
    pub fn new() -> Self {
        Self {
            evidence_history: VecDeque::new(),
            evidence_weights: HashMap::new(),
            cumulative_score: T::zero(),
            evidence_trends: HashMap::new(),
        }
    }

    /// Update with new evidence
    pub fn update_evidence(&mut self, evidence: ConvergenceEvidence<T>) -> Result<()> {
        self.evidence_history.push_back(evidence.clone());
        if self.evidence_history.len() > 1000 {
            self.evidence_history.pop_front();
        }

        // Update cumulative score (simplified)
        self.cumulative_score = self.cumulative_score + evidence.loss_evidence.loss_stability;

        Ok(())
    }
}

// Default implementations
impl<T: Float> Default for ConvergenceState<T> {
    fn default() -> Self {
        Self {
            converged: false,
            confidence: T::zero(),
            steps_since_detection: 0,
            current_phase: OptimizationPhase::Initialization,
            timestamp: SystemTime::now(),
        }
    }
}

impl<T: Float> Default for ConvergenceConsensus<T> {
    fn default() -> Self {
        Self {
            convergence_probability: T::zero(),
            consensus_confidence: T::zero(),
            voting_results: HashMap::new(),
            weighted_score: T::zero(),
            method_agreement: T::zero(),
        }
    }
}

impl<T: Float> Default for LossEvidence<T> {
    fn default() -> Self {
        Self {
            loss_change_rate: T::zero(),
            loss_stability: T::zero(),
            plateau_duration: 0,
            relative_improvement: T::zero(),
            absolute_improvement: T::zero(),
        }
    }
}

impl<T: Float> Default for GradientEvidence<T> {
    fn default() -> Self {
        Self {
            gradient_norm: T::zero(),
            gradient_norm_change_rate: T::zero(),
            gradient_stability: T::zero(),
            direction_consistency: T::zero(),
        }
    }
}

impl<T: Float> Default for ParameterEvidence<T> {
    fn default() -> Self {
        Self {
            parameter_change_magnitude: T::zero(),
            parameter_change_rate: T::zero(),
            parameter_stability: T::zero(),
            convergence_ratio: T::zero(),
        }
    }
}

impl<T: Float> Default for StatisticalEvidence<T> {
    fn default() -> Self {
        Self {
            p_value: T::one(),
            effect_size: T::zero(),
            confidence_interval: (T::zero(), T::zero()),
            test_statistic: T::zero(),
            test_type: StatisticalTestType::TTest,
        }
    }
}

impl<T: Float> Default for ValidationEvidence<T> {
    fn default() -> Self {
        Self {
            val_loss_change: T::zero(),
            val_metric_change: T::zero(),
            generalization_gap: T::zero(),
            overfitting_indicator: T::zero(),
        }
    }
}

impl<T: Float> Default for ConvergenceMetrics<T> {
    fn default() -> Self {
        Self {
            sensitivity: T::from(0.8).unwrap_or(T::zero()),
            specificity: T::from(0.9).unwrap_or(T::zero()),
            false_positive_rate: T::from(0.1).unwrap_or(T::zero()),
            false_negative_rate: T::from(0.2).unwrap_or(T::zero()),
            detection_latency: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::performance::*;

    #[test]
    fn test_convergence_analyzer_creation() {
        let config = ConvergenceAnalysisConfig::default();
        let analyzer = ConvergenceAnalyzer::<f32>::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_convergence_detector_creation() {
        let params = ConvergenceParameters::<f32>::default();
        let detector = ConvergenceDetector::new(ConvergenceDetectionMethod::LossBased, params);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_early_stopping_controller() {
        let config = EarlyStoppingConfig::<f32>::default();
        let mut controller = EarlyStoppingController::new(config).unwrap();

        // Should not stop initially
        assert!(!controller.should_stop());

        // Add some improving values
        controller.update(&1.0).unwrap();
        controller.update(&0.9).unwrap();
        controller.update(&0.8).unwrap();

        // Should not stop with improvements
        assert!(!controller.should_stop());

        // Add non-improving values
        for _ in 0..15 {
            controller.update(&0.9).unwrap();
        }

        // Should trigger early stopping after patience exceeded
        assert!(controller.should_stop());
    }

    #[test]
    fn test_convergence_evidence_default() {
        let evidence = ConvergenceEvidence::<f32>::default();
        assert_eq!(evidence.loss_evidence.loss_change_rate, 0.0);
        assert_eq!(evidence.gradient_evidence.gradient_norm, 0.0);
    }

    #[test]
    fn test_convergence_state_default() {
        let state = ConvergenceState::<f32>::default();
        assert!(!state.converged);
        assert_eq!(state.confidence, 0.0);
        assert_eq!(state.steps_since_detection, 0);
    }
}