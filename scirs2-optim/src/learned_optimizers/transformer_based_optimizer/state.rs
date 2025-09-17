//! State management for transformer-based optimizer

use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use crate::error::Result;
use super::config::TransformerBasedOptimizerConfig;
use super::meta_learning::MetaState;

/// Transformer optimizer state
pub struct TransformerOptimizerState<T: Float> {
    /// Current model parameters
    pub current_parameters: Array1<T>,

    /// Parameter history
    parameter_history: ParameterHistory<T>,

    /// Optimization state
    optimization_state: OptimizationState<T>,

    /// Learning state
    learning_state: LearningState<T>,

    /// Memory state
    memory_state: MemoryState<T>,

    /// Checkpoint manager
    checkpoint_manager: CheckpointManager<T>,

    /// State configuration
    config: StateConfig,

    /// State statistics
    statistics: StateStatistics<T>,

    /// State version for tracking changes
    version: usize,

    /// Creation timestamp
    created_at: std::time::Instant,

    /// Last update timestamp
    last_updated: std::time::Instant,
}

impl<T: Float> TransformerOptimizerState<T> {
    /// Create new optimizer state
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let parameter_count = config.model_dimension * config.num_transformer_layers;
        let current_parameters = Array1::zeros(parameter_count);

        let parameter_history = ParameterHistory::new(1000, parameter_count)?;
        let optimization_state = OptimizationState::new(config)?;
        let learning_state = LearningState::new(config)?;
        let memory_state = MemoryState::new()?;
        let checkpoint_manager = CheckpointManager::new(config)?;
        let state_config = StateConfig::from_optimizer_config(config);
        let statistics = StateStatistics::new();

        let now = std::time::Instant::now();

        Ok(Self {
            current_parameters,
            parameter_history,
            optimization_state,
            learning_state,
            memory_state,
            checkpoint_manager,
            config: state_config,
            statistics,
            version: 0,
            created_at: now,
            last_updated: now,
        })
    }

    /// Update state with optimization step
    pub fn update_with_step(&mut self, update: &Array1<T>, loss: Option<T>) -> Result<()> {
        // Apply parameter update
        self.current_parameters = &self.current_parameters + update;

        // Record parameter history
        self.parameter_history.record_parameters(&self.current_parameters)?;

        // Update optimization state
        self.optimization_state.update_with_step(update, loss)?;

        // Update learning state
        if let Some(loss_val) = loss {
            self.learning_state.update_with_loss(loss_val)?;
        }

        // Update statistics
        self.statistics.record_update(update, loss);

        // Increment version and update timestamp
        self.version += 1;
        self.last_updated = std::time::Instant::now();

        Ok(())
    }

    /// Create state snapshot
    pub fn create_snapshot(&self) -> Result<OptimizerStateSnapshot<T>> {
        Ok(OptimizerStateSnapshot {
            parameters: self.current_parameters.clone(),
            optimization_state: self.optimization_state.clone(),
            learning_state: self.learning_state.clone(),
            memory_state: self.memory_state.clone(),
            version: self.version,
            timestamp: self.last_updated,
            metadata: SnapshotMetadata {
                parameter_count: self.current_parameters.len(),
                total_updates: self.statistics.total_updates,
                session_duration: self.last_updated.duration_since(self.created_at),
            },
        })
    }

    /// Restore from snapshot
    pub fn restore_from_snapshot(&mut self, snapshot: &OptimizerStateSnapshot<T>) -> Result<()> {
        self.current_parameters = snapshot.parameters.clone();
        self.optimization_state = snapshot.optimization_state.clone();
        self.learning_state = snapshot.learning_state.clone();
        self.memory_state = snapshot.memory_state.clone();
        self.version = snapshot.version;
        self.last_updated = snapshot.timestamp;

        Ok(())
    }

    /// Save checkpoint
    pub fn save_checkpoint(&mut self, name: String) -> Result<String> {
        let snapshot = self.create_snapshot()?;
        let checkpoint_id = self.checkpoint_manager.save_checkpoint(name, snapshot)?;
        Ok(checkpoint_id)
    }

    /// Load checkpoint
    pub fn load_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        let snapshot = self.checkpoint_manager.load_checkpoint(checkpoint_id)?;
        self.restore_from_snapshot(&snapshot)?;
        Ok(())
    }

    /// Get parameter statistics
    pub fn get_parameter_stats(&self) -> ParameterStatistics<T> {
        self.parameter_history.get_statistics()
    }

    /// Get optimization progress
    pub fn get_optimization_progress(&self) -> OptimizationProgress<T> {
        self.optimization_state.get_progress()
    }

    /// Get learning statistics
    pub fn get_learning_stats(&self) -> LearningStatistics<T> {
        self.learning_state.get_statistics()
    }

    /// Reset state
    pub fn reset(&mut self) -> Result<()> {
        self.current_parameters.fill(T::zero());
        self.parameter_history.clear();
        self.optimization_state.reset()?;
        self.learning_state.reset()?;
        self.memory_state.reset()?;
        self.statistics.reset();
        self.version = 0;
        self.last_updated = std::time::Instant::now();
        Ok(())
    }

    /// Validate state consistency
    pub fn validate_state(&self) -> Result<StateValidationReport> {
        let mut issues = Vec::new();

        // Check parameter validity
        if self.current_parameters.iter().any(|&x| !x.is_finite()) {
            issues.push("Invalid parameters detected (NaN or infinity)".to_string());
        }

        // Check version consistency
        if self.version == 0 && self.statistics.total_updates > 0 {
            issues.push("Version mismatch with update count".to_string());
        }

        // Check timestamp consistency
        if self.last_updated < self.created_at {
            issues.push("Invalid timestamp ordering".to_string());
        }

        // Validate optimization state
        let opt_validation = self.optimization_state.validate()?;
        issues.extend(opt_validation.issues);

        // Validate learning state
        let learning_validation = self.learning_state.validate()?;
        issues.extend(learning_validation.issues);

        Ok(StateValidationReport {
            is_valid: issues.is_empty(),
            issues,
            validation_timestamp: std::time::Instant::now(),
        })
    }

    /// Get state summary
    pub fn get_state_summary(&self) -> StateSummary<T> {
        StateSummary {
            version: self.version,
            parameter_count: self.current_parameters.len(),
            parameter_norm: self.compute_parameter_norm(),
            total_updates: self.statistics.total_updates,
            session_duration: self.last_updated.duration_since(self.created_at),
            last_update_magnitude: self.statistics.last_update_magnitude,
            average_loss: self.learning_state.get_average_loss(),
            convergence_rate: self.learning_state.get_convergence_rate(),
            memory_usage: self.memory_state.get_total_usage(),
            checkpoint_count: self.checkpoint_manager.get_checkpoint_count(),
        }
    }

    /// Compute parameter norm
    fn compute_parameter_norm(&self) -> T {
        self.current_parameters.iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }

    /// Get state metadata
    pub fn get_metadata(&self) -> StateMetadata {
        StateMetadata {
            version: self.version,
            created_at: SystemTime::now(), // Convert from Instant for serialization
            last_updated: SystemTime::now(), // Convert from Instant for serialization
            total_updates: self.statistics.total_updates,
            configuration: self.config.clone(),
        }
    }

    /// Export state to serializable format
    pub fn export_state(&self) -> Result<SerializableState<T>> {
        Ok(SerializableState {
            parameters: self.current_parameters.to_vec(),
            parameter_shape: self.current_parameters.shape().to_vec(),
            optimization_state: self.optimization_state.to_serializable()?,
            learning_state: self.learning_state.to_serializable()?,
            metadata: self.get_metadata(),
            statistics: self.statistics.clone(),
        })
    }

    /// Import state from serializable format
    pub fn import_state(&mut self, state: SerializableState<T>) -> Result<()> {
        // Reconstruct parameters
        if state.parameter_shape.len() != 1 {
            return Err(crate::error::OptimError::Other(
                "Invalid parameter shape for 1D array".to_string()
            ));
        }

        self.current_parameters = Array1::from_vec(state.parameters);

        // Restore other state components
        self.optimization_state.from_serializable(state.optimization_state)?;
        self.learning_state.from_serializable(state.learning_state)?;
        self.statistics = state.statistics;
        self.version = state.metadata.version;
        self.last_updated = Instant::now(); // Convert from SystemTime for internal use

        Ok(())
    }
}

/// Parameter history management
pub struct ParameterHistory<T: Float> {
    /// Parameter snapshots
    snapshots: VecDeque<ParameterSnapshot<T>>,

    /// Maximum history size
    max_size: usize,

    /// Parameter dimension
    parameter_dimension: usize,

    /// Statistics
    statistics: ParameterStatistics<T>,
}

impl<T: Float> ParameterHistory<T> {
    pub fn new(max_size: usize, parameter_dimension: usize) -> Result<Self> {
        Ok(Self {
            snapshots: VecDeque::new(),
            max_size,
            parameter_dimension,
            statistics: ParameterStatistics::new(),
        })
    }

    pub fn record_parameters(&mut self, parameters: &Array1<T>) -> Result<()> {
        let snapshot = ParameterSnapshot {
            parameters: parameters.clone(),
            timestamp: std::time::Instant::now(),
            norm: parameters.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt(),
        };

        self.snapshots.push_back(snapshot.clone());
        if self.snapshots.len() > self.max_size {
            self.snapshots.pop_front();
        }

        self.statistics.update_with_snapshot(&snapshot);
        Ok(())
    }

    pub fn get_recent_parameters(&self, count: usize) -> Vec<Array1<T>> {
        self.snapshots.iter()
            .rev()
            .take(count)
            .map(|snapshot| snapshot.parameters.clone())
            .collect()
    }

    pub fn get_statistics(&self) -> ParameterStatistics<T> {
        self.statistics.clone()
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.statistics = ParameterStatistics::new();
    }
}

/// Optimization state tracking
#[derive(Debug, Clone)]
pub struct OptimizationState<T: Float> {
    /// Current learning rate
    pub learning_rate: T,

    /// Momentum state
    pub momentum: Option<Array1<T>>,

    /// Adaptive learning rate state (e.g., Adam)
    pub adaptive_state: Option<AdaptiveState<T>>,

    /// Gradient accumulation
    pub gradient_accumulator: GradientAccumulator<T>,

    /// Step count
    pub step_count: usize,

    /// Last update magnitude
    pub last_update_magnitude: T,

    /// Convergence tracking
    pub convergence_tracker: ConvergenceTracker<T>,
}

impl<T: Float> OptimizationState<T> {
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let parameter_count = config.model_dimension * config.num_transformer_layers;

        Ok(Self {
            learning_rate: config.learning_rate,
            momentum: None,
            adaptive_state: Some(AdaptiveState::new(parameter_count)?),
            gradient_accumulator: GradientAccumulator::new(parameter_count)?,
            step_count: 0,
            last_update_magnitude: T::zero(),
            convergence_tracker: ConvergenceTracker::new(),
        })
    }

    pub fn update_with_step(&mut self, update: &Array1<T>, loss: Option<T>) -> Result<()> {
        self.step_count += 1;
        self.last_update_magnitude = update.iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt();

        if let Some(loss_val) = loss {
            self.convergence_tracker.record_loss(loss_val);
        }

        // Update adaptive state
        if let Some(ref mut adaptive) = self.adaptive_state {
            adaptive.update_with_step(update)?;
        }

        Ok(())
    }

    pub fn get_progress(&self) -> OptimizationProgress<T> {
        OptimizationProgress {
            step_count: self.step_count,
            current_learning_rate: self.learning_rate,
            last_update_magnitude: self.last_update_magnitude,
            convergence_rate: self.convergence_tracker.get_convergence_rate(),
            stability_score: self.convergence_tracker.get_stability_score(),
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.step_count = 0;
        self.last_update_magnitude = T::zero();
        self.convergence_tracker.reset();

        if let Some(ref mut adaptive) = self.adaptive_state {
            adaptive.reset()?;
        }

        self.gradient_accumulator.reset()?;
        Ok(())
    }

    pub fn validate(&self) -> Result<ValidationResult> {
        let mut issues = Vec::new();

        if self.learning_rate <= T::zero() {
            issues.push("Invalid learning rate".to_string());
        }

        if !self.last_update_magnitude.is_finite() {
            issues.push("Invalid update magnitude".to_string());
        }

        Ok(ValidationResult { issues })
    }

    pub fn to_serializable(&self) -> Result<SerializableOptimizationState<T>> {
        Ok(SerializableOptimizationState {
            learning_rate: self.learning_rate,
            step_count: self.step_count,
            last_update_magnitude: self.last_update_magnitude,
            momentum: self.momentum.as_ref().map(|m| m.to_vec()),
            convergence_metrics: self.convergence_tracker.to_serializable(),
        })
    }

    pub fn from_serializable(&mut self, state: SerializableOptimizationState<T>) -> Result<()> {
        self.learning_rate = state.learning_rate;
        self.step_count = state.step_count;
        self.last_update_magnitude = state.last_update_magnitude;

        if let Some(momentum_vec) = state.momentum {
            self.momentum = Some(Array1::from_vec(momentum_vec));
        }

        self.convergence_tracker.from_serializable(state.convergence_metrics)?;
        Ok(())
    }
}

/// Learning state tracking
#[derive(Debug, Clone)]
pub struct LearningState<T: Float> {
    /// Loss history
    loss_history: VecDeque<T>,

    /// Meta-learning state
    meta_state: Option<MetaState<T>>,

    /// Task adaptation history
    adaptation_history: VecDeque<TaskAdaptationRecord<T>>,

    /// Learning rate schedule
    learning_schedule: LearningSchedule<T>,

    /// Performance metrics
    performance_metrics: LearningPerformanceMetrics<T>,
}

impl<T: Float> LearningState<T> {
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let meta_state = Some(MetaState::new(config.model_dimension)?);
        let learning_schedule = LearningSchedule::new(config.learning_rate, config.warmup_steps);

        Ok(Self {
            loss_history: VecDeque::new(),
            meta_state,
            adaptation_history: VecDeque::new(),
            learning_schedule,
            performance_metrics: LearningPerformanceMetrics::new(),
        })
    }

    pub fn update_with_loss(&mut self, loss: T) -> Result<()> {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }

        self.performance_metrics.record_loss(loss);

        if let Some(ref mut meta) = self.meta_state {
            meta.update_loss_history(loss);
        }

        Ok(())
    }

    pub fn get_statistics(&self) -> LearningStatistics<T> {
        LearningStatistics {
            total_episodes: self.loss_history.len(),
            average_loss: self.get_average_loss(),
            best_loss: self.get_best_loss(),
            convergence_rate: self.get_convergence_rate(),
            learning_stability: self.performance_metrics.get_stability_score(),
        }
    }

    pub fn get_average_loss(&self) -> T {
        if self.loss_history.is_empty() {
            T::zero()
        } else {
            self.loss_history.iter().fold(T::zero(), |acc, &loss| acc + loss) / T::from(self.loss_history.len()).unwrap()
        }
    }

    pub fn get_best_loss(&self) -> T {
        self.loss_history.iter().fold(T::infinity(), |min, &loss| min.min(loss))
    }

    pub fn get_convergence_rate(&self) -> T {
        if self.loss_history.len() < 2 {
            return T::zero();
        }

        let recent_losses: Vec<_> = self.loss_history.iter().rev().take(10).cloned().collect();
        if recent_losses.len() < 2 {
            return T::zero();
        }

        let initial = recent_losses.last().unwrap();
        let final_loss = recent_losses.first().unwrap();

        if *initial > T::zero() {
            (*initial - *final_loss) / *initial
        } else {
            T::zero()
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.loss_history.clear();
        self.adaptation_history.clear();
        self.performance_metrics.reset();

        if let Some(ref mut meta) = self.meta_state {
            *meta = MetaState::new(meta.get_parameters().len())?;
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<ValidationResult> {
        let mut issues = Vec::new();

        if self.loss_history.iter().any(|&loss| !loss.is_finite()) {
            issues.push("Invalid loss values detected".to_string());
        }

        Ok(ValidationResult { issues })
    }

    pub fn to_serializable(&self) -> Result<SerializableLearningState<T>> {
        Ok(SerializableLearningState {
            loss_history: self.loss_history.iter().cloned().collect(),
            average_loss: self.get_average_loss(),
            best_loss: self.get_best_loss(),
            convergence_rate: self.get_convergence_rate(),
        })
    }

    pub fn from_serializable(&mut self, state: SerializableLearningState<T>) -> Result<()> {
        self.loss_history = VecDeque::from(state.loss_history);
        Ok(())
    }
}

/// Memory state management
#[derive(Debug, Clone)]
pub struct MemoryState<T: Float> {
    /// Attention caches
    attention_caches: HashMap<String, AttentionCache<T>>,

    /// Memory usage tracking
    memory_usage: MemoryUsageTracker,

    /// Cache statistics
    cache_statistics: CacheStatistics,
}

impl<T: Float> MemoryState<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            attention_caches: HashMap::new(),
            memory_usage: MemoryUsageTracker::new(),
            cache_statistics: CacheStatistics::new(),
        })
    }

    pub fn get_total_usage(&self) -> usize {
        self.memory_usage.total_usage
    }

    pub fn reset(&mut self) -> Result<()> {
        self.attention_caches.clear();
        self.memory_usage.reset();
        self.cache_statistics.reset();
        Ok(())
    }
}

/// Checkpoint management
pub struct CheckpointManager<T: Float> {
    /// Stored checkpoints
    checkpoints: HashMap<String, OptimizerStateSnapshot<T>>,

    /// Checkpoint metadata
    metadata: HashMap<String, CheckpointMetadata>,

    /// Maximum checkpoints to keep
    max_checkpoints: usize,

    /// Auto-save configuration
    auto_save_config: AutoSaveConfig,
}

impl<T: Float> CheckpointManager<T> {
    pub fn new(config: &TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        Ok(Self {
            checkpoints: HashMap::new(),
            metadata: HashMap::new(),
            max_checkpoints: 10,
            auto_save_config: AutoSaveConfig::default(),
        })
    }

    pub fn save_checkpoint(&mut self, name: String, snapshot: OptimizerStateSnapshot<T>) -> Result<String> {
        let checkpoint_id = format!("{}_{}", name, snapshot.version);

        // Add metadata
        let metadata = CheckpointMetadata {
            id: checkpoint_id.clone(),
            name: name.clone(),
            created_at: std::time::Instant::now(),
            size_estimate: snapshot.parameters.len() * std::mem::size_of::<T>(),
            description: format!("Checkpoint at version {}", snapshot.version),
        };

        self.checkpoints.insert(checkpoint_id.clone(), snapshot);
        self.metadata.insert(checkpoint_id.clone(), metadata);

        // Cleanup old checkpoints if necessary
        if self.checkpoints.len() > self.max_checkpoints {
            self.cleanup_old_checkpoints()?;
        }

        Ok(checkpoint_id)
    }

    pub fn load_checkpoint(&self, checkpoint_id: &str) -> Result<OptimizerStateSnapshot<T>> {
        self.checkpoints.get(checkpoint_id)
            .cloned()
            .ok_or_else(|| crate::error::OptimError::Other(
                format!("Checkpoint {} not found", checkpoint_id)
            ))
    }

    pub fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        self.metadata.values().cloned().collect()
    }

    pub fn delete_checkpoint(&mut self, checkpoint_id: &str) -> Result<bool> {
        let removed_checkpoint = self.checkpoints.remove(checkpoint_id).is_some();
        let removed_metadata = self.metadata.remove(checkpoint_id).is_some();
        Ok(removed_checkpoint && removed_metadata)
    }

    pub fn get_checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        // Remove oldest checkpoints if over limit
        while self.checkpoints.len() > self.max_checkpoints {
            if let Some((oldest_id, _)) = self.metadata.iter()
                .min_by_key(|(_, metadata)| metadata.created_at)
                .map(|(id, metadata)| (id.clone(), metadata.clone()))
            {
                self.checkpoints.remove(&oldest_id);
                self.metadata.remove(&oldest_id);
            } else {
                break;
            }
        }
        Ok(())
    }
}

/// Supporting data structures and types

#[derive(Debug, Clone)]
pub struct ParameterSnapshot<T: Float> {
    pub parameters: Array1<T>,
    pub timestamp: std::time::Instant,
    pub norm: T,
}

#[derive(Debug, Clone)]
pub struct ParameterStatistics<T: Float> {
    pub total_snapshots: usize,
    pub average_norm: T,
    pub max_norm: T,
    pub min_norm: T,
    pub norm_trend: T,
}

impl<T: Float> ParameterStatistics<T> {
    pub fn new() -> Self {
        Self {
            total_snapshots: 0,
            average_norm: T::zero(),
            max_norm: T::zero(),
            min_norm: T::infinity(),
            norm_trend: T::zero(),
        }
    }

    pub fn update_with_snapshot(&mut self, snapshot: &ParameterSnapshot<T>) {
        self.total_snapshots += 1;
        self.average_norm = (self.average_norm * T::from(self.total_snapshots - 1).unwrap() + snapshot.norm) / T::from(self.total_snapshots).unwrap();
        self.max_norm = self.max_norm.max(snapshot.norm);
        self.min_norm = self.min_norm.min(snapshot.norm);
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveState<T: Float> {
    /// First moment estimates
    pub m: Array1<T>,
    /// Second moment estimates
    pub v: Array1<T>,
    /// Step count for bias correction
    pub step_count: usize,
    /// Beta parameters
    pub beta1: T,
    pub beta2: T,
    /// Epsilon for numerical stability
    pub epsilon: T,
}

impl<T: Float> AdaptiveState<T> {
    pub fn new(parameter_count: usize) -> Result<Self> {
        Ok(Self {
            m: Array1::zeros(parameter_count),
            v: Array1::zeros(parameter_count),
            step_count: 0,
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
        })
    }

    pub fn update_with_step(&mut self, _update: &Array1<T>) -> Result<()> {
        self.step_count += 1;
        // Adam-style update logic would go here
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.m.fill(T::zero());
        self.v.fill(T::zero());
        self.step_count = 0;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GradientAccumulator<T: Float> {
    /// Accumulated gradients
    pub accumulated_gradients: Array1<T>,
    /// Accumulation count
    pub accumulation_count: usize,
}

impl<T: Float> GradientAccumulator<T> {
    pub fn new(parameter_count: usize) -> Result<Self> {
        Ok(Self {
            accumulated_gradients: Array1::zeros(parameter_count),
            accumulation_count: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.accumulated_gradients.fill(T::zero());
        self.accumulation_count = 0;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ConvergenceTracker<T: Float> {
    /// Recent loss values
    recent_losses: VecDeque<T>,
    /// Convergence threshold
    convergence_threshold: T,
    /// Stability window size
    stability_window: usize,
}

impl<T: Float> ConvergenceTracker<T> {
    pub fn new() -> Self {
        Self {
            recent_losses: VecDeque::new(),
            convergence_threshold: T::from(1e-6).unwrap(),
            stability_window: 10,
        }
    }

    pub fn record_loss(&mut self, loss: T) {
        self.recent_losses.push_back(loss);
        if self.recent_losses.len() > self.stability_window {
            self.recent_losses.pop_front();
        }
    }

    pub fn get_convergence_rate(&self) -> T {
        if self.recent_losses.len() < 2 {
            return T::zero();
        }

        let first = self.recent_losses[0];
        let last = *self.recent_losses.back().unwrap();

        if first > T::zero() {
            (first - last) / first
        } else {
            T::zero()
        }
    }

    pub fn get_stability_score(&self) -> T {
        if self.recent_losses.len() < 2 {
            return T::zero();
        }

        let mean = self.recent_losses.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(self.recent_losses.len()).unwrap();
        let variance = self.recent_losses.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(self.recent_losses.len()).unwrap();

        T::one() / (T::one() + variance.sqrt())
    }

    pub fn reset(&mut self) {
        self.recent_losses.clear();
    }

    pub fn to_serializable(&self) -> SerializableConvergenceState<T> {
        SerializableConvergenceState {
            recent_losses: self.recent_losses.iter().cloned().collect(),
            convergence_rate: self.get_convergence_rate(),
            stability_score: self.get_stability_score(),
        }
    }

    pub fn from_serializable(&mut self, state: SerializableConvergenceState<T>) -> Result<()> {
        self.recent_losses = VecDeque::from(state.recent_losses);
        Ok(())
    }
}

/// State snapshots and serialization
#[derive(Debug, Clone)]
pub struct OptimizerStateSnapshot<T: Float> {
    pub parameters: Array1<T>,
    pub optimization_state: OptimizationState<T>,
    pub learning_state: LearningState<T>,
    pub memory_state: MemoryState<T>,
    pub version: usize,
    pub timestamp: std::time::Instant,
    pub metadata: SnapshotMetadata,
}

#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    pub parameter_count: usize,
    pub total_updates: usize,
    pub session_duration: Duration,
}

/// Configuration and metadata structures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateConfig {
    pub max_history_size: usize,
    pub checkpoint_frequency: usize,
    pub auto_save_enabled: bool,
    pub validation_enabled: bool,
}

impl StateConfig {
    pub fn from_optimizer_config<T: Float>(config: &TransformerBasedOptimizerConfig<T>) -> Self {
        Self {
            max_history_size: 1000,
            checkpoint_frequency: 100,
            auto_save_enabled: true,
            validation_enabled: true,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateMetadata {
    pub version: usize,
    pub created_at: std::time::SystemTime,
    pub last_updated: std::time::SystemTime,
    pub total_updates: usize,
    pub configuration: StateConfig,
}

/// Statistics and tracking structures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateStatistics<T: Float> {
    pub total_updates: usize,
    pub last_update_magnitude: T,
    pub average_update_magnitude: T,
    pub parameter_change_rate: T,
    pub update_frequency: f64,
}

impl<T: Float> StateStatistics<T> {
    pub fn new() -> Self {
        Self {
            total_updates: 0,
            last_update_magnitude: T::zero(),
            average_update_magnitude: T::zero(),
            parameter_change_rate: T::zero(),
            update_frequency: 0.0,
        }
    }

    pub fn record_update(&mut self, update: &Array1<T>, _loss: Option<T>) {
        self.total_updates += 1;
        let magnitude = update.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x).sqrt();
        self.last_update_magnitude = magnitude;
        self.average_update_magnitude = (self.average_update_magnitude * T::from(self.total_updates - 1).unwrap() + magnitude) / T::from(self.total_updates).unwrap();
    }

    pub fn reset(&mut self) {
        self.total_updates = 0;
        self.last_update_magnitude = T::zero();
        self.average_update_magnitude = T::zero();
        self.parameter_change_rate = T::zero();
        self.update_frequency = 0.0;
    }
}

/// Validation and reporting structures
#[derive(Debug, Clone)]
pub struct StateValidationReport {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub validation_timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub issues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StateSummary<T: Float> {
    pub version: usize,
    pub parameter_count: usize,
    pub parameter_norm: T,
    pub total_updates: usize,
    pub session_duration: Duration,
    pub last_update_magnitude: T,
    pub average_loss: T,
    pub convergence_rate: T,
    pub memory_usage: usize,
    pub checkpoint_count: usize,
}

/// Serializable state structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableState<T: Float> {
    pub parameters: Vec<T>,
    pub parameter_shape: Vec<usize>,
    pub optimization_state: SerializableOptimizationState<T>,
    pub learning_state: SerializableLearningState<T>,
    pub metadata: StateMetadata,
    pub statistics: StateStatistics<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableOptimizationState<T: Float> {
    pub learning_rate: T,
    pub step_count: usize,
    pub last_update_magnitude: T,
    pub momentum: Option<Vec<T>>,
    pub convergence_metrics: SerializableConvergenceState<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLearningState<T: Float> {
    pub loss_history: Vec<T>,
    pub average_loss: T,
    pub best_loss: T,
    pub convergence_rate: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConvergenceState<T: Float> {
    pub recent_losses: Vec<T>,
    pub convergence_rate: T,
    pub stability_score: T,
}

/// Additional supporting structures
#[derive(Debug, Clone)]
pub struct TaskAdaptationRecord<T: Float> {
    pub task_id: String,
    pub adaptation_steps: usize,
    pub final_loss: T,
    pub adaptation_time: Duration,
}

#[derive(Debug, Clone)]
pub struct LearningSchedule<T: Float> {
    pub initial_rate: T,
    pub current_rate: T,
    pub warmup_steps: usize,
    pub decay_factor: T,
}

impl<T: Float> LearningSchedule<T> {
    pub fn new(initial_rate: T, warmup_steps: usize) -> Self {
        Self {
            initial_rate: initial_rate,
            current_rate: initial_rate,
            warmup_steps,
            decay_factor: T::from(0.95).unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LearningPerformanceMetrics<T: Float> {
    pub loss_trend: T,
    pub convergence_stability: T,
    pub adaptation_efficiency: T,
}

impl<T: Float> LearningPerformanceMetrics<T> {
    pub fn new() -> Self {
        Self {
            loss_trend: T::zero(),
            convergence_stability: T::zero(),
            adaptation_efficiency: T::zero(),
        }
    }

    pub fn record_loss(&mut self, _loss: T) {
        // Update performance metrics logic
    }

    pub fn get_stability_score(&self) -> T {
        self.convergence_stability
    }

    pub fn reset(&mut self) {
        self.loss_trend = T::zero();
        self.convergence_stability = T::zero();
        self.adaptation_efficiency = T::zero();
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationProgress<T: Float> {
    pub step_count: usize,
    pub current_learning_rate: T,
    pub last_update_magnitude: T,
    pub convergence_rate: T,
    pub stability_score: T,
}

#[derive(Debug, Clone)]
pub struct LearningStatistics<T: Float> {
    pub total_episodes: usize,
    pub average_loss: T,
    pub best_loss: T,
    pub convergence_rate: T,
    pub learning_stability: T,
}

#[derive(Debug, Clone)]
pub struct AttentionCache<T: Float> {
    pub cached_keys: Array2<T>,
    pub cached_values: Array2<T>,
    pub cache_size: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryUsageTracker {
    pub total_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
}

impl MemoryUsageTracker {
    pub fn new() -> Self {
        Self {
            total_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.total_usage = 0;
        self.peak_usage = 0;
        self.allocation_count = 0;
    }
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hit_count: usize,
    pub miss_count: usize,
    pub eviction_count: usize,
}

impl CacheStatistics {
    pub fn new() -> Self {
        Self {
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.hit_count = 0;
        self.miss_count = 0;
        self.eviction_count = 0;
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    pub id: String,
    pub name: String,
    pub created_at: Instant,
    pub size_estimate: usize,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct AutoSaveConfig {
    pub enabled: bool,
    pub frequency: usize,
    pub max_auto_saves: usize,
}

impl Default for AutoSaveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: 100,
            max_auto_saves: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_state_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config);
        assert!(state.is_ok());

        let s = state.unwrap();
        assert_eq!(s.version, 0);
        assert!(s.current_parameters.len() > 0);
    }

    #[test]
    fn test_state_update() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let mut state = TransformerOptimizerState::new(&config).unwrap();

        let update = Array1::<f32>::ones(state.current_parameters.len());
        let result = state.update_with_step(&update, Some(1.5));
        assert!(result.is_ok());
        assert_eq!(state.version, 1);
    }

    #[test]
    fn test_snapshot_creation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config).unwrap();

        let snapshot = state.create_snapshot();
        assert!(snapshot.is_ok());

        let snap = snapshot.unwrap();
        assert_eq!(snap.version, 0);
        assert_eq!(snap.parameters.len(), state.current_parameters.len());
    }

    #[test]
    fn test_checkpoint_management() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let mut state = TransformerOptimizerState::new(&config).unwrap();

        let checkpoint_id = state.save_checkpoint("test_checkpoint".to_string());
        assert!(checkpoint_id.is_ok());

        let id = checkpoint_id.unwrap();
        let load_result = state.load_checkpoint(&id);
        assert!(load_result.is_ok());
    }

    #[test]
    fn test_parameter_history() {
        let mut history = ParameterHistory::<f32>::new(10, 5);
        assert!(history.is_ok());

        let mut h = history.unwrap();
        let params = Array1::<f32>::ones(5);
        assert!(h.record_parameters(&params).is_ok());

        let recent = h.get_recent_parameters(1);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_convergence_tracker() {
        let mut tracker = ConvergenceTracker::<f32>::new();

        tracker.record_loss(2.0);
        tracker.record_loss(1.5);
        tracker.record_loss(1.0);

        let convergence = tracker.get_convergence_rate();
        assert!(convergence > 0.0);

        let stability = tracker.get_stability_score();
        assert!(stability > 0.0 && stability <= 1.0);
    }

    #[test]
    fn test_state_validation() {
        let config = super::super::config::TransformerBasedOptimizerConfig::<f32>::default();
        let state = TransformerOptimizerState::new(&config).unwrap();

        let validation = state.validate_state();
        assert!(validation.is_ok());

        let report = validation.unwrap();
        assert!(report.is_valid);
    }
}