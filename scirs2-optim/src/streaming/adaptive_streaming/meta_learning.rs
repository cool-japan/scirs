//! Meta-learning and experience management for adaptive streaming
//!
//! This module provides sophisticated meta-learning capabilities that learn from
//! optimization experiences to improve future adaptation decisions, including
//! experience replay, transfer learning, and adaptive strategy selection.

use super::config::*;
use super::optimizer::{StreamingDataPoint, Adaptation, AdaptationType, AdaptationPriority};
use super::performance::{PerformanceSnapshot, PerformanceTracker};

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Meta-learning system for streaming optimization
pub struct MetaLearner<A: Float> {
    /// Meta-learning configuration
    config: MetaLearningConfig,
    /// Experience buffer for learning
    experience_buffer: ExperienceBuffer<A>,
    /// Meta-model for decision making
    meta_model: MetaModel<A>,
    /// Strategy selection system
    strategy_selector: StrategySelector<A>,
    /// Transfer learning system
    transfer_learning: TransferLearning<A>,
    /// Meta-learning statistics
    statistics: MetaLearningStatistics<A>,
    /// Learning rate adaptation
    learning_rate_adapter: LearningRateAdapter<A>,
}

/// Experience buffer for storing and managing learning experiences
pub struct ExperienceBuffer<A: Float> {
    /// Buffer configuration
    config: ExperienceReplayConfig,
    /// Stored experiences
    experiences: VecDeque<MetaExperience<A>>,
    /// Priority queue for prioritized replay
    priority_queue: VecDeque<(MetaExperience<A>, A)>,
    /// Experience importance sampling
    importance_weights: HashMap<usize, A>,
    /// Experience diversity tracker
    diversity_tracker: ExperienceDiversityTracker<A>,
}

/// Meta-learning experience representation
#[derive(Debug, Clone)]
pub struct MetaExperience<A: Float> {
    /// Unique experience ID
    pub id: u64,
    /// State when experience occurred
    pub state: MetaState<A>,
    /// Action taken
    pub action: MetaAction<A>,
    /// Reward received
    pub reward: A,
    /// Next state after action
    pub next_state: Option<MetaState<A>>,
    /// Experience timestamp
    pub timestamp: Instant,
    /// Episode context
    pub episode_context: EpisodeContext<A>,
    /// Experience priority for replay
    pub priority: A,
    /// Number of times replayed
    pub replay_count: usize,
}

/// Meta-state representation
#[derive(Debug, Clone)]
pub struct MetaState<A: Float> {
    /// Performance metrics at this state
    pub performance_metrics: Vec<A>,
    /// Resource state
    pub resource_state: Vec<A>,
    /// Drift indicators
    pub drift_indicators: Vec<A>,
    /// Adaptation history length
    pub adaptation_history: usize,
    /// State timestamp
    pub timestamp: Instant,
}

/// Meta-action representation
#[derive(Debug, Clone)]
pub struct MetaAction<A: Float> {
    /// Adaptation magnitudes applied
    pub adaptation_magnitudes: Vec<A>,
    /// Types of adaptations
    pub adaptation_types: Vec<AdaptationType>,
    /// Learning rate change
    pub learning_rate_change: A,
    /// Buffer size change
    pub buffer_size_change: A,
    /// Action timestamp
    pub timestamp: Instant,
}

/// Episode context for meta-learning
#[derive(Debug, Clone)]
pub struct EpisodeContext<A: Float> {
    /// Episode ID
    pub episode_id: u64,
    /// Episode start time
    pub start_time: Instant,
    /// Episode duration
    pub duration: Duration,
    /// Initial performance
    pub initial_performance: A,
    /// Final performance
    pub final_performance: A,
    /// Number of adaptations in episode
    pub adaptation_count: usize,
    /// Episode outcome classification
    pub outcome: EpisodeOutcome,
}

/// Episode outcome classifications
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeOutcome {
    /// Significant improvement
    Success,
    /// Moderate improvement
    PartialSuccess,
    /// No significant change
    Neutral,
    /// Performance degradation
    Failure,
    /// Severe performance degradation
    CriticalFailure,
}

/// Experience diversity tracking
pub struct ExperienceDiversityTracker<A: Float> {
    /// State space clustering
    state_clusters: Vec<StateCluster<A>>,
    /// Action space clustering
    action_clusters: Vec<ActionCluster<A>>,
    /// Diversity metrics
    diversity_metrics: DiversityMetrics<A>,
    /// Novelty detection
    novelty_detector: NoveltyDetector<A>,
}

/// State clustering for diversity
#[derive(Debug, Clone)]
pub struct StateCluster<A: Float> {
    /// Cluster center
    pub center: MetaState<A>,
    /// Cluster members
    pub members: Vec<usize>,
    /// Cluster radius
    pub radius: A,
    /// Last update time
    pub last_update: Instant,
}

/// Action clustering for diversity
#[derive(Debug, Clone)]
pub struct ActionCluster<A: Float> {
    /// Cluster center
    pub center: MetaAction<A>,
    /// Cluster members
    pub members: Vec<usize>,
    /// Cluster effectiveness
    pub effectiveness: A,
    /// Usage frequency
    pub usage_frequency: usize,
}

/// Diversity metrics for experience management
#[derive(Debug, Clone)]
pub struct DiversityMetrics<A: Float> {
    /// State space coverage
    pub state_coverage: A,
    /// Action space coverage
    pub action_coverage: A,
    /// Experience entropy
    pub experience_entropy: A,
    /// Temporal diversity
    pub temporal_diversity: A,
    /// Outcome diversity
    pub outcome_diversity: A,
}

/// Novelty detection for new experiences
pub struct NoveltyDetector<A: Float> {
    /// Reference experiences for comparison
    reference_experiences: VecDeque<MetaExperience<A>>,
    /// Novelty threshold
    novelty_threshold: A,
    /// Feature importance weights
    feature_weights: Vec<A>,
}

/// Meta-model for decision making
pub struct MetaModel<A: Float> {
    /// Model type
    model_type: MetaModelType,
    /// Model parameters
    parameters: MetaModelParameters<A>,
    /// Training history
    training_history: VecDeque<TrainingEpisode<A>>,
    /// Model performance metrics
    performance_metrics: ModelPerformanceMetrics<A>,
    /// Feature importance
    feature_importance: Vec<A>,
}

/// Meta-model parameters
#[derive(Debug, Clone)]
pub struct MetaModelParameters<A: Float> {
    /// Weight matrices for neural networks
    pub weights: Vec<Vec<A>>,
    /// Bias vectors
    pub biases: Vec<A>,
    /// Learning rate
    pub learning_rate: A,
    /// Regularization parameters
    pub regularization: RegularizationParams<A>,
    /// Optimization parameters
    pub optimization: OptimizationParams<A>,
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParams<A: Float> {
    /// L1 regularization strength
    pub l1_lambda: A,
    /// L2 regularization strength
    pub l2_lambda: A,
    /// Dropout rate
    pub dropout_rate: A,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Optimization parameters
#[derive(Debug, Clone)]
pub struct OptimizationParams<A: Float> {
    /// Momentum coefficient
    pub momentum: A,
    /// Adam beta1 parameter
    pub beta1: A,
    /// Adam beta2 parameter
    pub beta2: A,
    /// Epsilon for numerical stability
    pub epsilon: A,
    /// Gradient clipping threshold
    pub grad_clip_threshold: A,
}

/// Training episode for meta-model
#[derive(Debug, Clone)]
pub struct TrainingEpisode<A: Float> {
    /// Episode ID
    pub episode_id: u64,
    /// Training loss
    pub training_loss: A,
    /// Validation loss
    pub validation_loss: A,
    /// Training accuracy
    pub training_accuracy: A,
    /// Validation accuracy
    pub validation_accuracy: A,
    /// Episode duration
    pub duration: Duration,
    /// Timestamp
    pub timestamp: Instant,
}

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics<A: Float> {
    /// Prediction accuracy
    pub prediction_accuracy: A,
    /// Decision quality
    pub decision_quality: A,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: A,
    /// Transfer learning success rate
    pub transfer_success_rate: A,
    /// Generalization performance
    pub generalization_performance: A,
}

/// Strategy selection system
pub struct StrategySelector<A: Float> {
    /// Available strategies
    strategies: HashMap<String, AdaptationStrategy<A>>,
    /// Strategy performance history
    strategy_performance: HashMap<String, StrategyPerformance<A>>,
    /// Strategy selection policy
    selection_policy: SelectionPolicy,
    /// Exploration parameters
    exploration_params: ExplorationParams<A>,
    /// Context-based selection
    context_selector: ContextBasedSelector<A>,
}

/// Adaptation strategy representation
#[derive(Debug, Clone)]
pub struct AdaptationStrategy<A: Float> {
    /// Strategy name
    pub name: String,
    /// Strategy parameters
    pub parameters: HashMap<String, A>,
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Applicability conditions
    pub conditions: Vec<StrategyCondition<A>>,
    /// Expected outcomes
    pub expected_outcomes: Vec<A>,
}

/// Strategy types
#[derive(Debug, Clone)]
pub enum StrategyType {
    /// Conservative strategy (small changes)
    Conservative,
    /// Aggressive strategy (large changes)
    Aggressive,
    /// Balanced strategy
    Balanced,
    /// Reactive strategy (responds to changes)
    Reactive,
    /// Proactive strategy (anticipates changes)
    Proactive,
    /// Custom strategy
    Custom(String),
}

/// Strategy applicability conditions
#[derive(Debug, Clone)]
pub struct StrategyCondition<A: Float> {
    /// Condition type
    pub condition_type: ConditionType,
    /// Threshold value
    pub threshold: A,
    /// Operator for comparison
    pub operator: ComparisonOperator,
    /// Weight in decision making
    pub weight: A,
}

/// Condition types for strategy selection
#[derive(Debug, Clone)]
pub enum ConditionType {
    /// Performance threshold
    Performance,
    /// Resource utilization
    ResourceUtilization,
    /// Data quality
    DataQuality,
    /// Drift detection
    DriftDetection,
    /// Time-based condition
    Temporal,
    /// Custom condition
    Custom(String),
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Equal to
    EqualTo,
    /// Between values
    Between(f64, f64),
    /// In set of values
    InSet(Vec<f64>),
}

/// Strategy performance tracking
#[derive(Debug, Clone)]
pub struct StrategyPerformance<A: Float> {
    /// Number of times used
    pub usage_count: usize,
    /// Success rate
    pub success_rate: A,
    /// Average improvement
    pub avg_improvement: A,
    /// Best improvement achieved
    pub best_improvement: A,
    /// Worst outcome
    pub worst_outcome: A,
    /// Recent performance trend
    pub recent_trend: TrendDirection,
    /// Context-specific performance
    pub context_performance: HashMap<String, A>,
}

/// Trend direction for strategy performance
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Declining trend
    Declining,
    /// Stable trend
    Stable,
    /// Oscillating trend
    Oscillating,
}

/// Strategy selection policies
#[derive(Debug, Clone)]
pub enum SelectionPolicy {
    /// Epsilon-greedy selection
    EpsilonGreedy { epsilon: f64 },
    /// Upper Confidence Bound
    UCB { confidence_parameter: f64 },
    /// Thompson sampling
    ThompsonSampling,
    /// Softmax selection
    Softmax { temperature: f64 },
    /// Context-aware selection
    ContextAware,
    /// Multi-armed bandit
    MultiArmedBandit,
}

/// Exploration parameters
#[derive(Debug, Clone)]
pub struct ExplorationParams<A: Float> {
    /// Exploration rate
    pub exploration_rate: A,
    /// Exploration decay
    pub exploration_decay: A,
    /// Minimum exploration rate
    pub min_exploration_rate: A,
    /// Curiosity bonus weight
    pub curiosity_weight: A,
    /// Novelty bonus weight
    pub novelty_weight: A,
}

/// Context-based strategy selector
pub struct ContextBasedSelector<A: Float> {
    /// Context features
    context_features: Vec<ContextFeature<A>>,
    /// Context clustering
    context_clusters: Vec<ContextCluster<A>>,
    /// Strategy mappings per context
    context_strategies: HashMap<String, Vec<String>>,
    /// Context recognition model
    context_model: ContextModel<A>,
}

/// Context feature for strategy selection
#[derive(Debug, Clone)]
pub struct ContextFeature<A: Float> {
    /// Feature name
    pub name: String,
    /// Feature value
    pub value: A,
    /// Feature importance
    pub importance: A,
    /// Feature stability
    pub stability: A,
}

/// Context clustering for similar situations
#[derive(Debug, Clone)]
pub struct ContextCluster<A: Float> {
    /// Cluster ID
    pub id: String,
    /// Cluster center features
    pub center: Vec<A>,
    /// Cluster radius
    pub radius: A,
    /// Associated strategies
    pub strategies: Vec<String>,
    /// Cluster performance
    pub performance: A,
}

/// Context recognition model
pub struct ContextModel<A: Float> {
    /// Model parameters
    parameters: Vec<A>,
    /// Feature weights
    feature_weights: Vec<A>,
    /// Classification threshold
    threshold: A,
    /// Model accuracy
    accuracy: A,
}

/// Transfer learning system
pub struct TransferLearning<A: Float> {
    /// Source domain experiences
    source_experiences: HashMap<String, Vec<MetaExperience<A>>>,
    /// Transfer learning strategies
    transfer_strategies: Vec<TransferStrategy>,
    /// Domain adaptation methods
    domain_adaptation: DomainAdaptation<A>,
    /// Transfer learning metrics
    transfer_metrics: TransferMetrics<A>,
}

/// Transfer learning strategies
#[derive(Debug, Clone)]
pub enum TransferStrategy {
    /// Direct parameter transfer
    ParameterTransfer,
    /// Feature transfer
    FeatureTransfer,
    /// Instance transfer
    InstanceTransfer,
    /// Relational transfer
    RelationalTransfer,
    /// Meta-transfer learning
    MetaTransfer,
}

/// Domain adaptation methods
pub struct DomainAdaptation<A: Float> {
    /// Source domain characteristics
    source_characteristics: Vec<A>,
    /// Target domain characteristics
    target_characteristics: Vec<A>,
    /// Adaptation weights
    adaptation_weights: Vec<A>,
    /// Domain similarity measure
    domain_similarity: A,
}

/// Transfer learning metrics
#[derive(Debug, Clone)]
pub struct TransferMetrics<A: Float> {
    /// Transfer success rate
    pub success_rate: A,
    /// Improvement from transfer
    pub improvement: A,
    /// Transfer efficiency
    pub efficiency: A,
    /// Negative transfer incidents
    pub negative_transfer_count: usize,
}

/// Learning rate adaptation system
pub struct LearningRateAdapter<A: Float> {
    /// Current learning rate
    current_rate: A,
    /// Learning rate history
    rate_history: VecDeque<A>,
    /// Performance feedback
    performance_feedback: VecDeque<A>,
    /// Adaptation strategy
    adaptation_strategy: LearningRateStrategy,
    /// Rate bounds
    min_rate: A,
    max_rate: A,
}

/// Learning rate adaptation strategies
#[derive(Debug, Clone)]
pub enum LearningRateStrategy {
    /// Fixed learning rate
    Fixed,
    /// Step decay
    StepDecay { decay_factor: f64, step_size: usize },
    /// Exponential decay
    ExponentialDecay { decay_rate: f64 },
    /// Performance-based adaptation
    PerformanceBased,
    /// Cyclical learning rates
    Cyclical { min_lr: f64, max_lr: f64, cycle_length: usize },
    /// Adaptive learning rate (Adam-style)
    Adaptive,
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStatistics<A: Float> {
    /// Total experiences collected
    pub total_experiences: usize,
    /// Model training episodes
    pub training_episodes: usize,
    /// Average reward per episode
    pub avg_reward_per_episode: A,
    /// Best episode reward
    pub best_episode_reward: A,
    /// Learning progress
    pub learning_progress: A,
    /// Strategy selection accuracy
    pub strategy_selection_accuracy: A,
    /// Transfer learning success rate
    pub transfer_success_rate: A,
    /// Experience replay effectiveness
    pub replay_effectiveness: A,
}

impl<A: Float + Default + Clone + std::iter::Sum> MetaLearner<A> {
    /// Creates a new meta-learner
    pub fn new(config: &StreamingConfig) -> Result<Self, String> {
        let meta_config = config.meta_learning_config.clone();

        let experience_buffer = ExperienceBuffer::new(&meta_config.replay_config);
        let meta_model = MetaModel::new(meta_config.model_complexity.clone())?;
        let strategy_selector = StrategySelector::new();
        let transfer_learning = TransferLearning::new();
        let learning_rate_adapter = LearningRateAdapter::new(meta_config.meta_learning_rate);

        let statistics = MetaLearningStatistics {
            total_experiences: 0,
            training_episodes: 0,
            avg_reward_per_episode: A::zero(),
            best_episode_reward: A::zero(),
            learning_progress: A::zero(),
            strategy_selection_accuracy: A::zero(),
            transfer_success_rate: A::zero(),
            replay_effectiveness: A::zero(),
        };

        Ok(Self {
            config: meta_config,
            experience_buffer,
            meta_model,
            strategy_selector,
            transfer_learning,
            statistics,
            learning_rate_adapter,
        })
    }

    /// Updates the meta-learner with new experience
    pub fn update_experience(&mut self, state: MetaState<A>, action: MetaAction<A>, reward: A) -> Result<(), String> {
        let experience = MetaExperience {
            id: self.generate_experience_id(),
            state,
            action,
            reward,
            next_state: None, // Will be filled in next update
            timestamp: Instant::now(),
            episode_context: self.create_episode_context(reward)?,
            priority: self.calculate_experience_priority(reward),
            replay_count: 0,
        };

        // Add to experience buffer
        self.experience_buffer.add_experience(experience)?;

        // Update statistics
        self.statistics.total_experiences += 1;

        // Trigger learning if enough experiences collected
        if self.statistics.total_experiences % self.config.update_frequency == 0 {
            self.trigger_learning()?;
        }

        Ok(())
    }

    /// Generates unique experience ID
    fn generate_experience_id(&self) -> u64 {
        self.statistics.total_experiences as u64 + 1
    }

    /// Creates episode context for experience
    fn create_episode_context(&self, reward: A) -> Result<EpisodeContext<A>, String> {
        let outcome = if reward > A::from(0.8).unwrap() {
            EpisodeOutcome::Success
        } else if reward > A::from(0.5).unwrap() {
            EpisodeOutcome::PartialSuccess
        } else if reward > A::from(0.2).unwrap() {
            EpisodeOutcome::Neutral
        } else if reward > A::from(-0.2).unwrap() {
            EpisodeOutcome::Failure
        } else {
            EpisodeOutcome::CriticalFailure
        };

        Ok(EpisodeContext {
            episode_id: self.statistics.training_episodes as u64,
            start_time: Instant::now(),
            duration: Duration::from_secs(60), // Simplified
            initial_performance: A::from(0.5).unwrap(), // Simplified
            final_performance: reward,
            adaptation_count: 1, // Simplified
            outcome,
        })
    }

    /// Calculates priority for experience replay
    fn calculate_experience_priority(&self, reward: A) -> A {
        // Higher priority for experiences with extreme rewards (positive or negative)
        let abs_reward = reward.abs();
        let surprise = if abs_reward > A::from(0.8).unwrap() {
            A::from(1.0).unwrap()
        } else {
            abs_reward
        };

        surprise
    }

    /// Triggers meta-learning update
    fn trigger_learning(&mut self) -> Result<(), String> {
        // Sample experiences for training
        let training_batch = self.experience_buffer.sample_batch(self.config.replay_config.batch_size)?;

        // Train meta-model
        self.meta_model.train_on_batch(&training_batch)?;

        // Update strategy selection
        self.strategy_selector.update_from_experiences(&training_batch)?;

        // Update statistics
        self.statistics.training_episodes += 1;

        Ok(())
    }

    /// Recommends adaptations based on current state
    pub fn recommend_adaptations(
        &mut self,
        current_data: &[StreamingDataPoint<A>],
        performance_tracker: &PerformanceTracker<A>,
    ) -> Result<Vec<Adaptation<A>>, String> {
        // Extract current meta-state
        let current_state = self.extract_meta_state(current_data, performance_tracker)?;

        // Use meta-model to predict best action
        let predicted_action = self.meta_model.predict_action(&current_state)?;

        // Select appropriate strategy
        let strategy = self.strategy_selector.select_strategy(&current_state)?;

        // Generate adaptations based on prediction and strategy
        let adaptations = self.generate_adaptations_from_prediction(&predicted_action, &strategy)?;

        Ok(adaptations)
    }

    /// Extracts meta-state from current situation
    fn extract_meta_state(
        &self,
        current_data: &[StreamingDataPoint<A>],
        performance_tracker: &PerformanceTracker<A>,
    ) -> Result<MetaState<A>, String> {
        // Get recent performance
        let recent_performance = performance_tracker.get_recent_performance(5);
        let performance_metrics = if !recent_performance.is_empty() {
            vec![
                recent_performance[0].loss,
                recent_performance[0].accuracy.unwrap_or(A::zero()),
                recent_performance[0].convergence_rate.unwrap_or(A::zero()),
            ]
        } else {
            vec![A::zero(), A::zero(), A::zero()]
        };

        // Extract resource state (simplified)
        let resource_state = vec![A::from(0.5).unwrap(), A::from(0.3).unwrap()];

        // Extract drift indicators (simplified)
        let drift_indicators = vec![A::from(0.1).unwrap()];

        Ok(MetaState {
            performance_metrics,
            resource_state,
            drift_indicators,
            adaptation_history: self.statistics.total_experiences,
            timestamp: Instant::now(),
        })
    }

    /// Generates adaptations from model prediction
    fn generate_adaptations_from_prediction(
        &self,
        predicted_action: &MetaAction<A>,
        _strategy: &AdaptationStrategy<A>,
    ) -> Result<Vec<Adaptation<A>>, String> {
        let mut adaptations = Vec::new();

        // Generate adaptations based on predicted action
        for (i, &magnitude) in predicted_action.adaptation_magnitudes.iter().enumerate() {
            if magnitude.abs() > A::from(0.05).unwrap() { // Minimum threshold
                let adaptation_type = if i < predicted_action.adaptation_types.len() {
                    predicted_action.adaptation_types[i].clone()
                } else {
                    AdaptationType::LearningRate // Default
                };

                let adaptation = Adaptation {
                    adaptation_type,
                    magnitude,
                    target_component: "meta_learner".to_string(),
                    parameters: std::collections::HashMap::new(),
                    priority: if magnitude.abs() > A::from(0.3).unwrap() {
                        AdaptationPriority::High
                    } else {
                        AdaptationPriority::Normal
                    },
                    timestamp: Instant::now(),
                };

                adaptations.push(adaptation);
            }
        }

        Ok(adaptations)
    }

    /// Applies adaptation to meta-learning system
    pub fn apply_adaptation(&mut self, adaptation: &Adaptation<A>) -> Result<(), String> {
        match adaptation.adaptation_type {
            AdaptationType::MetaLearning => {
                // Adjust meta-learning parameters
                let new_rate = self.learning_rate_adapter.current_rate + adaptation.magnitude;
                self.learning_rate_adapter.update_rate(new_rate)?;
            },
            _ => {
                // Handle other adaptation types
            }
        }

        Ok(())
    }

    /// Gets meta-learning effectiveness score
    pub fn get_effectiveness_score(&self) -> f32 {
        self.statistics.learning_progress.to_f32().unwrap_or(0.0)
    }

    /// Gets diagnostic information
    pub fn get_diagnostics(&self) -> MetaLearningDiagnostics {
        MetaLearningDiagnostics {
            total_experiences: self.statistics.total_experiences,
            training_episodes: self.statistics.training_episodes,
            current_learning_rate: self.learning_rate_adapter.current_rate.to_f64().unwrap_or(0.0),
            model_accuracy: self.meta_model.performance_metrics.prediction_accuracy.to_f64().unwrap_or(0.0),
            strategy_count: self.strategy_selector.strategies.len(),
            transfer_success_rate: self.statistics.transfer_success_rate.to_f64().unwrap_or(0.0),
        }
    }
}

impl<A: Float + Default + Clone> ExperienceBuffer<A> {
    fn new(config: &ExperienceReplayConfig) -> Self {
        Self {
            config: config.clone(),
            experiences: VecDeque::with_capacity(10000),
            priority_queue: VecDeque::new(),
            importance_weights: HashMap::new(),
            diversity_tracker: ExperienceDiversityTracker::new(),
        }
    }

    fn add_experience(&mut self, experience: MetaExperience<A>) -> Result<(), String> {
        // Add to main buffer
        if self.experiences.len() >= 10000 {
            self.experiences.pop_front();
        }
        self.experiences.push_back(experience.clone());

        // Add to priority queue if using prioritized replay
        if self.config.enable_prioritized_replay {
            let priority = experience.priority;
            self.priority_queue.push_back((experience, priority));

            // Sort by priority
            self.priority_queue.make_contiguous().sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Limit priority queue size
            if self.priority_queue.len() > 1000 {
                self.priority_queue.pop_front();
            }
        }

        Ok(())
    }

    fn sample_batch(&mut self, batch_size: usize) -> Result<Vec<MetaExperience<A>>, String> {
        if self.experiences.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch = Vec::with_capacity(batch_size);

        if self.config.enable_prioritized_replay && !self.priority_queue.is_empty() {
            // Sample from priority queue
            for _ in 0..batch_size.min(self.priority_queue.len()) {
                if let Some((experience, _)) = self.priority_queue.pop_front() {
                    batch.push(experience);
                }
            }
        } else {
            // Random sampling
            for _ in 0..batch_size.min(self.experiences.len()) {
                let idx = fastrand::usize(0..self.experiences.len());
                if let Some(experience) = self.experiences.get(idx) {
                    batch.push(experience.clone());
                }
            }
        }

        Ok(batch)
    }
}

impl<A: Float + Default + Clone> ExperienceDiversityTracker<A> {
    fn new() -> Self {
        Self {
            state_clusters: Vec::new(),
            action_clusters: Vec::new(),
            diversity_metrics: DiversityMetrics {
                state_coverage: A::zero(),
                action_coverage: A::zero(),
                experience_entropy: A::zero(),
                temporal_diversity: A::zero(),
                outcome_diversity: A::zero(),
            },
            novelty_detector: NoveltyDetector::new(),
        }
    }
}

impl<A: Float + Default + Clone> NoveltyDetector<A> {
    fn new() -> Self {
        Self {
            reference_experiences: VecDeque::with_capacity(1000),
            novelty_threshold: A::from(0.5).unwrap(),
            feature_weights: Vec::new(),
        }
    }
}

impl<A: Float + Default + Clone> MetaModel<A> {
    fn new(complexity: MetaModelComplexity) -> Result<Self, String> {
        let parameters = match complexity {
            MetaModelComplexity::Low => MetaModelParameters {
                weights: vec![vec![A::from(0.1).unwrap(); 10]; 2],
                biases: vec![A::zero(); 10],
                learning_rate: A::from(0.01).unwrap(),
                regularization: RegularizationParams {
                    l1_lambda: A::from(0.001).unwrap(),
                    l2_lambda: A::from(0.001).unwrap(),
                    dropout_rate: A::from(0.1).unwrap(),
                    early_stopping_patience: 10,
                },
                optimization: OptimizationParams {
                    momentum: A::from(0.9).unwrap(),
                    beta1: A::from(0.9).unwrap(),
                    beta2: A::from(0.999).unwrap(),
                    epsilon: A::from(1e-8).unwrap(),
                    grad_clip_threshold: A::from(1.0).unwrap(),
                },
            },
            _ => MetaModelParameters {
                weights: vec![vec![A::from(0.1).unwrap(); 50]; 3],
                biases: vec![A::zero(); 50],
                learning_rate: A::from(0.001).unwrap(),
                regularization: RegularizationParams {
                    l1_lambda: A::from(0.0001).unwrap(),
                    l2_lambda: A::from(0.0001).unwrap(),
                    dropout_rate: A::from(0.2).unwrap(),
                    early_stopping_patience: 20,
                },
                optimization: OptimizationParams {
                    momentum: A::from(0.9).unwrap(),
                    beta1: A::from(0.9).unwrap(),
                    beta2: A::from(0.999).unwrap(),
                    epsilon: A::from(1e-8).unwrap(),
                    grad_clip_threshold: A::from(1.0).unwrap(),
                },
            },
        };

        Ok(Self {
            model_type: MetaModelType::NeuralNetwork,
            parameters,
            training_history: VecDeque::with_capacity(1000),
            performance_metrics: ModelPerformanceMetrics {
                prediction_accuracy: A::from(0.5).unwrap(),
                decision_quality: A::from(0.5).unwrap(),
                adaptation_effectiveness: A::from(0.5).unwrap(),
                transfer_success_rate: A::from(0.5).unwrap(),
                generalization_performance: A::from(0.5).unwrap(),
            },
            feature_importance: Vec::new(),
        })
    }

    fn train_on_batch(&mut self, batch: &[MetaExperience<A>]) -> Result<(), String> {
        if batch.is_empty() {
            return Ok(());
        }

        // Simplified training - in practice would implement proper neural network training
        let avg_reward = batch.iter().map(|e| e.reward).sum::<A>() / A::from(batch.len()).unwrap();

        // Update learning rate based on performance
        if avg_reward > A::from(0.5).unwrap() {
            self.parameters.learning_rate = self.parameters.learning_rate * A::from(1.01).unwrap();
        } else {
            self.parameters.learning_rate = self.parameters.learning_rate * A::from(0.99).unwrap();
        }

        // Update performance metrics
        self.performance_metrics.prediction_accuracy = avg_reward;

        Ok(())
    }

    fn predict_action(&self, state: &MetaState<A>) -> Result<MetaAction<A>, String> {
        // Simplified prediction - in practice would use trained neural network
        let action = MetaAction {
            adaptation_magnitudes: vec![A::from(0.1).unwrap(), A::from(-0.05).unwrap()],
            adaptation_types: vec![AdaptationType::LearningRate, AdaptationType::BufferSize],
            learning_rate_change: A::from(0.01).unwrap(),
            buffer_size_change: A::from(5.0).unwrap(),
            timestamp: Instant::now(),
        };

        Ok(action)
    }
}

impl<A: Float + Default + Clone> StrategySelector<A> {
    fn new() -> Self {
        let mut strategies = HashMap::new();

        // Add default strategies
        strategies.insert("conservative".to_string(), AdaptationStrategy {
            name: "conservative".to_string(),
            parameters: HashMap::new(),
            strategy_type: StrategyType::Conservative,
            conditions: Vec::new(),
            expected_outcomes: vec![A::from(0.05).unwrap()],
        });

        strategies.insert("aggressive".to_string(), AdaptationStrategy {
            name: "aggressive".to_string(),
            parameters: HashMap::new(),
            strategy_type: StrategyType::Aggressive,
            conditions: Vec::new(),
            expected_outcomes: vec![A::from(0.2).unwrap()],
        });

        Self {
            strategies,
            strategy_performance: HashMap::new(),
            selection_policy: SelectionPolicy::EpsilonGreedy { epsilon: 0.1 },
            exploration_params: ExplorationParams {
                exploration_rate: A::from(0.1).unwrap(),
                exploration_decay: A::from(0.99).unwrap(),
                min_exploration_rate: A::from(0.01).unwrap(),
                curiosity_weight: A::from(0.1).unwrap(),
                novelty_weight: A::from(0.1).unwrap(),
            },
            context_selector: ContextBasedSelector::new(),
        }
    }

    fn select_strategy(&self, _state: &MetaState<A>) -> Result<AdaptationStrategy<A>, String> {
        // Simple strategy selection - in practice would be more sophisticated
        if let Some(strategy) = self.strategies.get("balanced") {
            Ok(strategy.clone())
        } else if let Some(strategy) = self.strategies.values().next() {
            Ok(strategy.clone())
        } else {
            Err("No strategies available".to_string())
        }
    }

    fn update_from_experiences(&mut self, _experiences: &[MetaExperience<A>]) -> Result<(), String> {
        // Update strategy performance based on experiences
        Ok(())
    }
}

impl<A: Float + Default + Clone> ContextBasedSelector<A> {
    fn new() -> Self {
        Self {
            context_features: Vec::new(),
            context_clusters: Vec::new(),
            context_strategies: HashMap::new(),
            context_model: ContextModel {
                parameters: Vec::new(),
                feature_weights: Vec::new(),
                threshold: A::from(0.5).unwrap(),
                accuracy: A::from(0.5).unwrap(),
            },
        }
    }
}

impl<A: Float + Default + Clone> TransferLearning<A> {
    fn new() -> Self {
        Self {
            source_experiences: HashMap::new(),
            transfer_strategies: vec![TransferStrategy::ParameterTransfer],
            domain_adaptation: DomainAdaptation {
                source_characteristics: Vec::new(),
                target_characteristics: Vec::new(),
                adaptation_weights: Vec::new(),
                domain_similarity: A::from(0.5).unwrap(),
            },
            transfer_metrics: TransferMetrics {
                success_rate: A::from(0.5).unwrap(),
                improvement: A::from(0.1).unwrap(),
                efficiency: A::from(0.7).unwrap(),
                negative_transfer_count: 0,
            },
        }
    }
}

impl<A: Float + Default + Clone> LearningRateAdapter<A> {
    fn new(initial_rate: f64) -> Self {
        Self {
            current_rate: A::from(initial_rate).unwrap(),
            rate_history: VecDeque::with_capacity(100),
            performance_feedback: VecDeque::with_capacity(100),
            adaptation_strategy: LearningRateStrategy::PerformanceBased,
            min_rate: A::from(1e-6).unwrap(),
            max_rate: A::from(0.1).unwrap(),
        }
    }

    fn update_rate(&mut self, new_rate: A) -> Result<(), String> {
        self.current_rate = new_rate.max(self.min_rate).min(self.max_rate);

        if self.rate_history.len() >= 100 {
            self.rate_history.pop_front();
        }
        self.rate_history.push_back(self.current_rate);

        Ok(())
    }
}

/// Diagnostic information for meta-learning
#[derive(Debug, Clone)]
pub struct MetaLearningDiagnostics {
    pub total_experiences: usize,
    pub training_episodes: usize,
    pub current_learning_rate: f64,
    pub model_accuracy: f64,
    pub strategy_count: usize,
    pub transfer_success_rate: f64,
}