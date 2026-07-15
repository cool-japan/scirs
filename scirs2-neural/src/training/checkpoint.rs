//! Training checkpoint support
//!
//! This module provides checkpoint functionality for saving and restoring
//! mid-training state. A checkpoint captures the model weights, optimizer state,
//! current epoch, training metrics, and any other state needed to resume training.
//!
//! ## Overview
//!
//! Checkpoints enable:
//! - **Fault tolerance**: Resume training after crashes or interruptions
//! - **Best model tracking**: Save the best model during training
//! - **Training inspection**: Analyze training state at any point
//! - **Transfer learning**: Start from a checkpoint with different training config
//!
//! ## Format
//!
//! Checkpoints are stored as a directory containing:
//! - `model.safetensors` — Model weights in SafeTensors format
//! - `checkpoint_meta.json` — Epoch, metrics, and optimizer state metadata
//! - `optimizer_state.safetensors` — Optional optimizer moment vectors
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_neural::training::checkpoint::{CheckpointConfig, CheckpointManager};
//! use std::path::PathBuf;
//!
//! let config = CheckpointConfig {
//!     save_dir: PathBuf::from("/tmp/checkpoints"),
//!     save_every: 5,
//!     max_checkpoints: 3,
//!     save_best: true,
//!     monitor_metric: "val_loss".to_string(),
//!     minimize_metric: true,
//! };
//!
//! let manager = CheckpointManager::<f64>::new(config);
//! ```

use crate::error::{NeuralError, Result};
use crate::serialization::safetensors::{SafeTensorsReader, SafeTensorsWriter};
use crate::serialization::traits::{ModelMetadata, NamedParameters};
use scirs2_core::ndarray::IxDyn;
use scirs2_core::numeric::{Float, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

// ============================================================================
// Error type
// ============================================================================

/// Error type for checkpoint operations
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// I/O error during checkpoint save/load
    #[error("Checkpoint I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Checkpoint serialization error: {0}")]
    Serialization(String),

    /// No checkpoint found in the specified directory
    #[error("No checkpoint found in directory: {0}")]
    NotFound(String),

    /// Checkpoint format version mismatch
    #[error("Checkpoint version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },

    /// Invalid checkpoint configuration
    #[error("Invalid checkpoint configuration: {0}")]
    InvalidConfig(String),
}

impl From<CheckpointError> for NeuralError {
    fn from(e: CheckpointError) -> Self {
        NeuralError::IOError(e.to_string())
    }
}

// ============================================================================
// Checkpoint configuration
// ============================================================================

/// Configuration for the checkpoint manager
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints in
    pub save_dir: PathBuf,
    /// Save a checkpoint every N epochs (0 = disabled)
    pub save_every: usize,
    /// Maximum number of checkpoints to keep (0 = keep all)
    pub max_checkpoints: usize,
    /// Save the best checkpoint separately as "best.ckpt/"
    pub save_best: bool,
    /// Metric to monitor for determining "best" checkpoint
    /// (e.g., "val_loss", "val_accuracy")
    pub monitor_metric: String,
    /// If true, lower values of `monitor_metric` are considered better (e.g., loss)
    /// If false, higher values are better (e.g., accuracy)
    pub minimize_metric: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            save_dir: PathBuf::from("checkpoints"),
            save_every: 1,
            max_checkpoints: 5,
            save_best: true,
            monitor_metric: "val_loss".to_string(),
            minimize_metric: true,
        }
    }
}

impl CheckpointConfig {
    /// Validate the configuration
    pub fn validate(&self) -> std::result::Result<(), CheckpointError> {
        if self.monitor_metric.is_empty() {
            return Err(CheckpointError::InvalidConfig(
                "monitor_metric must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Optimizer checkpoint state
// ============================================================================

/// Serializable state for first-moment (m) and second-moment (v) Adam buffers,
/// or SGD momentum buffers.
///
/// The moment vectors are stored as raw f64 data so we can serialize them
/// to JSON. On load they are converted back to the concrete float type F.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerCheckpointState {
    /// Optimizer type name (e.g., "Adam", "SGD", "AdamW")
    pub optimizer_type: String,
    /// Current learning rate
    pub learning_rate: f64,
    /// Step counter (used to correct Adam bias)
    pub step: usize,
    /// Beta1 (Adam-family) or momentum (SGD)
    pub beta1: Option<f64>,
    /// Beta2 (Adam-family)
    pub beta2: Option<f64>,
    /// Epsilon (Adam-family)
    pub epsilon: Option<f64>,
    /// Weight decay
    pub weight_decay: f64,
    /// First moment (m) vectors, keyed by parameter name
    pub first_moments: HashMap<String, Vec<f64>>,
    /// Second moment (v) vectors, keyed by parameter name
    pub second_moments: HashMap<String, Vec<f64>>,
    /// Parameter shapes, keyed by parameter name
    pub param_shapes: HashMap<String, Vec<usize>>,
}

impl Default for OptimizerCheckpointState {
    fn default() -> Self {
        Self {
            optimizer_type: "Unknown".to_string(),
            learning_rate: 0.001,
            step: 0,
            beta1: None,
            beta2: None,
            epsilon: None,
            weight_decay: 0.0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            param_shapes: HashMap::new(),
        }
    }
}

impl OptimizerCheckpointState {
    /// Create optimizer state for an Adam optimizer
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            optimizer_type: "Adam".to_string(),
            learning_rate,
            step: 0,
            beta1: Some(beta1),
            beta2: Some(beta2),
            epsilon: Some(epsilon),
            weight_decay: 0.0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            param_shapes: HashMap::new(),
        }
    }

    /// Create optimizer state for an SGD optimizer
    pub fn sgd(learning_rate: f64, momentum: f64, weight_decay: f64) -> Self {
        Self {
            optimizer_type: "SGD".to_string(),
            learning_rate,
            step: 0,
            beta1: Some(momentum),
            beta2: None,
            epsilon: None,
            weight_decay,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            param_shapes: HashMap::new(),
        }
    }

    /// Returns true if this state has any moment vectors stored
    pub fn has_moments(&self) -> bool {
        !self.first_moments.is_empty() || !self.second_moments.is_empty()
    }
}

// ============================================================================
// Learning rate scheduler state
// ============================================================================

/// Serializable state for learning rate schedulers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrSchedulerState {
    /// Scheduler type name (e.g., "CosineAnnealing", "StepLR", "ReduceOnPlateau")
    pub scheduler_type: String,
    /// Current step within the scheduler
    pub scheduler_step: usize,
    /// Current learning rate
    pub current_lr: f64,
    /// Base (initial) learning rate
    pub base_lr: f64,
    /// Number of epochs without improvement (for ReduceOnPlateau)
    pub patience_counter: usize,
    /// Best monitored metric value seen so far
    pub best_metric: Option<f64>,
    /// Extra scheduler-specific parameters
    pub extra_params: HashMap<String, f64>,
}

impl Default for LrSchedulerState {
    fn default() -> Self {
        Self {
            scheduler_type: "Identity".to_string(),
            scheduler_step: 0,
            current_lr: 0.001,
            base_lr: 0.001,
            patience_counter: 0,
            best_metric: None,
            extra_params: HashMap::new(),
        }
    }
}

impl LrSchedulerState {
    /// Create a state for a cosine annealing scheduler
    pub fn cosine_annealing(base_lr: f64, t_max: usize) -> Self {
        let mut extra = HashMap::new();
        extra.insert("t_max".to_string(), t_max as f64);
        Self {
            scheduler_type: "CosineAnnealing".to_string(),
            current_lr: base_lr,
            base_lr,
            extra_params: extra,
            ..Default::default()
        }
    }

    /// Create a state for a step LR scheduler
    pub fn step_lr(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        let mut extra = HashMap::new();
        extra.insert("step_size".to_string(), step_size as f64);
        extra.insert("gamma".to_string(), gamma);
        Self {
            scheduler_type: "StepLR".to_string(),
            current_lr: base_lr,
            base_lr,
            extra_params: extra,
            ..Default::default()
        }
    }
}

// ============================================================================
// Training checkpoint — full training state snapshot
// ============================================================================

/// A full snapshot of training state, sufficient to resume training identically.
///
/// Stored as a directory:
/// - `checkpoint_meta.json` — all scalar fields and metrics
/// - `model.safetensors` — model parameter tensors
/// - `optimizer_state.json` — optimizer moment data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current epoch number (0-indexed, points to the *completed* epoch)
    pub epoch: usize,
    /// Global step counter (total optimizer updates)
    pub step: usize,
    /// Best monitored metric value seen across all epochs so far
    pub best_metric: Option<f64>,
    /// Metrics history: one `HashMap<String, f64>` per epoch
    pub metrics_history: Vec<HashMap<String, f64>>,
    /// Optimizer state (serialized as JSON-compatible struct)
    pub optimizer_state: OptimizerCheckpointState,
    /// Learning rate scheduler state (if any)
    pub lr_scheduler_state: Option<LrSchedulerState>,
    /// Checkpoint format version for forward compatibility
    pub format_version: String,
    /// Architecture name of the saved model
    pub architecture: String,
    /// Timestamp when this checkpoint was created
    pub timestamp: String,
    /// Total number of epochs planned for training
    pub total_epochs: Option<usize>,
    /// Whether training completed without interruption
    pub training_completed: bool,
    /// Random seed used for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for TrainingCheckpoint {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            best_metric: None,
            metrics_history: Vec::new(),
            optimizer_state: OptimizerCheckpointState::default(),
            lr_scheduler_state: None,
            format_version: "0.3.0".to_string(),
            architecture: "Unknown".to_string(),
            timestamp: simple_timestamp(),
            total_epochs: None,
            training_completed: false,
            random_seed: None,
        }
    }
}

impl TrainingCheckpoint {
    /// Create a new checkpoint for the given epoch
    pub fn new(epoch: usize, step: usize, architecture: &str) -> Self {
        Self {
            epoch,
            step,
            architecture: architecture.to_string(),
            timestamp: simple_timestamp(),
            ..Default::default()
        }
    }

    /// Retrieve the latest value of a metric from metrics_history
    pub fn latest_metric(&self, name: &str) -> Option<f64> {
        self.metrics_history
            .last()
            .and_then(|m| m.get(name).copied())
    }

    /// Mark training as completed
    pub fn mark_completed(mut self) -> Self {
        self.training_completed = true;
        self
    }
}

// ============================================================================
// Checkpoint manager
// ============================================================================

/// Manages saving, loading, and cleanup of training checkpoints.
///
/// The manager tracks saved checkpoint paths and enforces the `max_checkpoints`
/// limit by deleting the oldest checkpoint when a new one is saved.
pub struct CheckpointManager<F: Float + Debug> {
    /// Configuration
    config: CheckpointConfig,
    /// Paths of saved regular (non-best) checkpoints, oldest first
    saved_checkpoints: Vec<PathBuf>,
    /// Current best metric value (for best-model tracking)
    best_metric_value: Option<f64>,
    /// Phantom for the float type F
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ToPrimitive + 'static> CheckpointManager<F> {
    /// Create a new checkpoint manager with the given configuration.
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            saved_checkpoints: Vec::new(),
            best_metric_value: None,
            _phantom: PhantomData,
        }
    }

    /// Save a training checkpoint.
    ///
    /// Creates a directory named `epoch_{epoch:04}.ckpt/` inside `config.save_dir`.
    /// If `config.save_best` is true and the monitored metric improved, also saves
    /// a `best.ckpt/` symlink-style copy.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` — Snapshot of training state (metadata, optimizer, scheduler)
    /// * `model_params` — Named model parameters to persist in safetensors format
    /// * `epoch` — Current epoch number (used to name the directory)
    /// * `metrics` — Current epoch metrics map (e.g., `{"val_loss": 0.35}`)
    ///
    /// # Returns
    ///
    /// The path to the directory where the checkpoint was written.
    pub fn save(
        &mut self,
        checkpoint: &TrainingCheckpoint,
        model_params: &NamedParameters,
        epoch: usize,
        metrics: &HashMap<String, F>,
    ) -> std::result::Result<PathBuf, CheckpointError> {
        self.config.validate()?;

        // Only save if save_every trigger fires
        if self.config.save_every > 0 && !epoch.is_multiple_of(self.config.save_every) {
            // Not a checkpoint epoch; still check if best
            if self.config.save_best {
                let _ = self.maybe_save_best(checkpoint, model_params, metrics)?;
            }
            return Ok(self.config.save_dir.clone());
        }

        fs::create_dir_all(&self.config.save_dir)?;

        let dir_name = format!("epoch_{:04}.ckpt", epoch);
        let ckpt_dir = self.config.save_dir.join(&dir_name);

        self.write_checkpoint_to_dir(checkpoint, model_params, &ckpt_dir)?;

        // Track and enforce max_checkpoints
        self.saved_checkpoints.push(ckpt_dir.clone());
        self.cleanup_old_checkpoints()?;

        // Check if best
        if self.config.save_best {
            let _ = self.maybe_save_best(checkpoint, model_params, metrics)?;
        }

        Ok(ckpt_dir)
    }

    /// Load a training checkpoint from a specific directory path.
    ///
    /// # Returns
    ///
    /// A tuple of `(TrainingCheckpoint, NamedParameters)` where:
    /// - `TrainingCheckpoint` contains all scalar metadata
    /// - `NamedParameters` contains the model parameter tensors
    pub fn load(
        path: &Path,
    ) -> std::result::Result<(TrainingCheckpoint, NamedParameters), CheckpointError> {
        if !path.exists() {
            return Err(CheckpointError::NotFound(path.display().to_string()));
        }

        // Load metadata JSON
        let meta_path = path.join("checkpoint_meta.json");
        if !meta_path.exists() {
            return Err(CheckpointError::NotFound(format!(
                "checkpoint_meta.json not found in {}",
                path.display()
            )));
        }
        let meta_bytes = fs::read(&meta_path)?;
        let checkpoint: TrainingCheckpoint = serde_json::from_slice(&meta_bytes)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        // Load model parameters via safetensors
        let model_path = path.join("model.safetensors");
        if !model_path.exists() {
            return Err(CheckpointError::NotFound(format!(
                "model.safetensors not found in {}",
                path.display()
            )));
        }
        let reader = SafeTensorsReader::from_file(&model_path)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        let model_params = reader
            .to_named_parameters()
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        Ok((checkpoint, model_params))
    }

    /// Load the latest checkpoint from the configured save directory.
    ///
    /// Returns `None` if no checkpoints exist.
    pub fn load_latest(
        save_dir: &Path,
    ) -> std::result::Result<Option<(TrainingCheckpoint, NamedParameters)>, CheckpointError> {
        let checkpoints = Self::list_checkpoints(save_dir)?;
        match checkpoints.last() {
            None => Ok(None),
            Some(path) => {
                let result = Self::load(path)?;
                Ok(Some(result))
            }
        }
    }

    /// Load the best checkpoint from the configured save directory.
    ///
    /// Looks for a `best.ckpt/` directory in `save_dir`.
    /// Returns `None` if no best checkpoint exists.
    pub fn load_best(
        save_dir: &Path,
    ) -> std::result::Result<Option<(TrainingCheckpoint, NamedParameters)>, CheckpointError> {
        let best_dir = save_dir.join("best.ckpt");
        if !best_dir.exists() {
            return Ok(None);
        }
        let result = Self::load(&best_dir)?;
        Ok(Some(result))
    }

    /// List all regular checkpoint directories in a save directory, sorted by epoch.
    ///
    /// Only directories matching `epoch_NNNN.ckpt` pattern are included.
    /// The `best.ckpt` directory is excluded.
    pub fn list_checkpoints(save_dir: &Path) -> std::result::Result<Vec<PathBuf>, CheckpointError> {
        if !save_dir.exists() {
            return Ok(Vec::new());
        }

        let entries = fs::read_dir(save_dir)?;
        let mut checkpoints: Vec<(usize, PathBuf)> = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let file_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_owned(),
                None => continue,
            };
            // Match pattern epoch_NNNN.ckpt
            if file_name.starts_with("epoch_") && file_name.ends_with(".ckpt") {
                let epoch_part = file_name
                    .trim_start_matches("epoch_")
                    .trim_end_matches(".ckpt");
                if let Ok(epoch) = epoch_part.parse::<usize>() {
                    checkpoints.push((epoch, path));
                }
            }
        }

        // Sort by epoch number ascending
        checkpoints.sort_by_key(|(epoch, _)| *epoch);
        Ok(checkpoints.into_iter().map(|(_, p)| p).collect())
    }

    /// Delete old checkpoints, keeping only the `max_checkpoints` most recent.
    fn cleanup_old_checkpoints(&mut self) -> std::result::Result<(), CheckpointError> {
        if self.config.max_checkpoints == 0 {
            return Ok(());
        }

        while self.saved_checkpoints.len() > self.config.max_checkpoints {
            let oldest = self.saved_checkpoints.remove(0);
            if oldest.exists() {
                fs::remove_dir_all(&oldest)?;
            }
        }
        Ok(())
    }

    /// Check if the current metrics improve on the best, and if so, save a `best.ckpt/` copy.
    fn maybe_save_best(
        &mut self,
        checkpoint: &TrainingCheckpoint,
        model_params: &NamedParameters,
        metrics: &HashMap<String, F>,
    ) -> std::result::Result<bool, CheckpointError> {
        let metric_value = match metrics.get(&self.config.monitor_metric) {
            Some(v) => match v.to_f64() {
                Some(f) => f,
                None => return Ok(false),
            },
            None => return Ok(false),
        };

        let is_better = match self.best_metric_value {
            None => true,
            Some(best) => {
                if self.config.minimize_metric {
                    metric_value < best
                } else {
                    metric_value > best
                }
            }
        };

        if is_better {
            self.best_metric_value = Some(metric_value);
            fs::create_dir_all(&self.config.save_dir)?;
            let best_dir = self.config.save_dir.join("best.ckpt");
            self.write_checkpoint_to_dir(checkpoint, model_params, &best_dir)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Internal: write checkpoint to a specific directory path.
    fn write_checkpoint_to_dir(
        &self,
        checkpoint: &TrainingCheckpoint,
        model_params: &NamedParameters,
        dir: &Path,
    ) -> std::result::Result<(), CheckpointError> {
        fs::create_dir_all(dir)?;

        // Write metadata JSON
        let meta_json = serde_json::to_string_pretty(checkpoint)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        fs::write(dir.join("checkpoint_meta.json"), meta_json.as_bytes())?;

        // Write model parameters using SafeTensors
        let model_path = dir.join("model.safetensors");
        let meta = ModelMetadata::new(
            &checkpoint.architecture,
            "f64",
            model_params.total_parameters(),
        )
        .with_extra("epoch", &checkpoint.epoch.to_string())
        .with_extra("format_version", &checkpoint.format_version);

        let mut writer = SafeTensorsWriter::new();
        writer.add_model_metadata(&meta);
        writer
            .add_named_parameters(model_params)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        writer
            .write_to_file(&model_path)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        // Write optimizer state as a separate JSON file for easy inspection
        let opt_json = serde_json::to_string_pretty(&checkpoint.optimizer_state)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        fs::write(dir.join("optimizer_state.json"), opt_json.as_bytes())?;

        Ok(())
    }

    /// Get the current best metric value tracked by this manager.
    pub fn best_metric_value(&self) -> Option<f64> {
        self.best_metric_value
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Get the list of currently tracked checkpoint paths.
    pub fn saved_checkpoint_paths(&self) -> &[PathBuf] {
        &self.saved_checkpoints
    }
}

// ============================================================================
// Legacy / lower-level checkpoint functions (compatible with old API)
// ============================================================================

/// A training checkpoint capturing the full state needed to resume training
///
/// This is the legacy metadata structure — see [`TrainingCheckpoint`] for
/// the richer v0.3.0 version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Current epoch number (0-indexed)
    pub epoch: usize,
    /// Global step counter
    pub global_step: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Training loss at checkpoint time
    pub train_loss: Option<f64>,
    /// Validation loss at checkpoint time
    pub val_loss: Option<f64>,
    /// Best validation loss seen so far
    pub best_val_loss: Option<f64>,
    /// Additional training metrics (e.g., accuracy, F1)
    pub metrics: HashMap<String, f64>,
    /// Optimizer state metadata
    pub optimizer_state: OptimizerStateMetadata,
    /// Architecture name
    pub architecture: String,
    /// Model version string
    pub model_version: String,
    /// Timestamp when checkpoint was created
    pub timestamp: String,
    /// Whether training was completed or interrupted
    pub training_completed: bool,
    /// Total number of epochs planned
    pub total_epochs: Option<usize>,
    /// Random seed used for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            learning_rate: 0.001,
            train_loss: None,
            val_loss: None,
            best_val_loss: None,
            metrics: HashMap::new(),
            optimizer_state: OptimizerStateMetadata::default(),
            architecture: "Unknown".to_string(),
            model_version: "0.1.0".to_string(),
            timestamp: simple_timestamp(),
            training_completed: false,
            total_epochs: None,
            random_seed: None,
        }
    }
}

impl CheckpointMetadata {
    /// Create a new checkpoint metadata with basic info
    pub fn new(architecture: &str, epoch: usize, learning_rate: f64) -> Self {
        Self {
            architecture: architecture.to_string(),
            epoch,
            learning_rate,
            timestamp: simple_timestamp(),
            ..Default::default()
        }
    }

    /// Set training loss
    pub fn with_train_loss(mut self, loss: f64) -> Self {
        self.train_loss = Some(loss);
        self
    }

    /// Set validation loss
    pub fn with_val_loss(mut self, loss: f64) -> Self {
        self.val_loss = Some(loss);
        self
    }

    /// Set best validation loss
    pub fn with_best_val_loss(mut self, loss: f64) -> Self {
        self.best_val_loss = Some(loss);
        self
    }

    /// Add a metric
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    /// Set total epochs
    pub fn with_total_epochs(mut self, total: usize) -> Self {
        self.total_epochs = Some(total);
        self
    }

    /// Set global step
    pub fn with_global_step(mut self, step: usize) -> Self {
        self.global_step = step;
        self
    }

    /// Mark training as completed
    pub fn mark_completed(mut self) -> Self {
        self.training_completed = true;
        self
    }
}

/// Metadata for optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStateMetadata {
    /// Optimizer type name (e.g., "Adam", "SGD", "AdamW")
    pub optimizer_type: String,
    /// Number of parameter groups
    pub num_param_groups: usize,
    /// Per-parameter-group settings
    pub param_groups: Vec<ParamGroupState>,
}

impl Default for OptimizerStateMetadata {
    fn default() -> Self {
        Self {
            optimizer_type: "Unknown".to_string(),
            num_param_groups: 0,
            param_groups: Vec::new(),
        }
    }
}

impl OptimizerStateMetadata {
    /// Create metadata for a simple optimizer
    pub fn new(optimizer_type: &str) -> Self {
        Self {
            optimizer_type: optimizer_type.to_string(),
            num_param_groups: 1,
            param_groups: vec![ParamGroupState::default()],
        }
    }
}

/// State of a single parameter group in the optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamGroupState {
    /// Learning rate for this group
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Momentum (for SGD-like optimizers)
    pub momentum: Option<f64>,
    /// Beta1 (for Adam-like optimizers)
    pub beta1: Option<f64>,
    /// Beta2 (for Adam-like optimizers)
    pub beta2: Option<f64>,
    /// Epsilon (for Adam-like optimizers)
    pub epsilon: Option<f64>,
    /// Step count for this group
    pub step_count: usize,
}

impl Default for ParamGroupState {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 0.0,
            momentum: None,
            beta1: None,
            beta2: None,
            epsilon: None,
            step_count: 0,
        }
    }
}

impl ParamGroupState {
    /// Create state for an Adam optimizer
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            weight_decay: 0.0,
            momentum: None,
            beta1: Some(beta1),
            beta2: Some(beta2),
            epsilon: Some(epsilon),
            step_count: 0,
        }
    }

    /// Create state for an SGD optimizer
    pub fn sgd(learning_rate: f64, momentum: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            weight_decay,
            momentum: Some(momentum),
            beta1: None,
            beta2: None,
            epsilon: None,
            step_count: 0,
        }
    }
}

// ============================================================================
// Lower-level save/load functions (legacy compatibility)
// ============================================================================

/// Save a training checkpoint to a directory.
///
/// Creates a directory containing:
/// - `model.safetensors` — Model weights
/// - `checkpoint_meta.json` — Training metadata
/// - `optimizer_state.safetensors` — Optimizer moment vectors (optional)
pub fn save_checkpoint(
    checkpoint_dir: &Path,
    model_params: &NamedParameters,
    metadata: &CheckpointMetadata,
    optimizer_moments: Option<&NamedParameters>,
) -> Result<()> {
    fs::create_dir_all(checkpoint_dir)
        .map_err(|e| NeuralError::IOError(format!("Cannot create checkpoint directory: {e}")))?;

    // Save model weights
    let model_path = checkpoint_dir.join("model.safetensors");
    let model_metadata = ModelMetadata::new(
        &metadata.architecture,
        "f64",
        model_params.total_parameters(),
    )
    .with_extra("epoch", &metadata.epoch.to_string())
    .with_extra("checkpoint", "true");

    let mut writer = SafeTensorsWriter::new();
    writer.add_model_metadata(&model_metadata);
    writer.add_named_parameters(model_params)?;
    writer.write_to_file(&model_path)?;

    // Save checkpoint metadata
    let meta_path = checkpoint_dir.join("checkpoint_meta.json");
    let meta_json = serde_json::to_string_pretty(metadata)
        .map_err(|e| NeuralError::SerializationError(format!("Cannot serialize metadata: {e}")))?;
    fs::write(&meta_path, meta_json)
        .map_err(|e| NeuralError::IOError(format!("Cannot write metadata: {e}")))?;

    // Save optimizer state if provided
    if let Some(moments) = optimizer_moments {
        if !moments.is_empty() {
            let optimizer_path = checkpoint_dir.join("optimizer_state.safetensors");
            let opt_metadata = ModelMetadata::new("optimizer", "f64", moments.total_parameters());
            let mut opt_writer = SafeTensorsWriter::new();
            opt_writer.add_model_metadata(&opt_metadata);
            opt_writer.add_named_parameters(moments)?;
            opt_writer.write_to_file(&optimizer_path)?;
        }
    }

    Ok(())
}

/// Load a training checkpoint from a directory.
///
/// Returns the model parameters, checkpoint metadata, and optional optimizer moments.
pub fn load_checkpoint(
    checkpoint_dir: &Path,
) -> Result<(NamedParameters, CheckpointMetadata, Option<NamedParameters>)> {
    if !checkpoint_dir.exists() {
        return Err(NeuralError::IOError(format!(
            "Checkpoint directory does not exist: {}",
            checkpoint_dir.display()
        )));
    }

    // Load model weights
    let model_path = checkpoint_dir.join("model.safetensors");
    if !model_path.exists() {
        return Err(NeuralError::IOError(format!(
            "Model weights not found at: {}",
            model_path.display()
        )));
    }
    let reader = SafeTensorsReader::from_file(&model_path)?;
    let model_params = reader.to_named_parameters()?;

    // Load metadata
    let meta_path = checkpoint_dir.join("checkpoint_meta.json");
    if !meta_path.exists() {
        return Err(NeuralError::IOError(format!(
            "Checkpoint metadata not found at: {}",
            meta_path.display()
        )));
    }
    let meta_json = fs::read_to_string(&meta_path)
        .map_err(|e| NeuralError::IOError(format!("Cannot read metadata: {e}")))?;
    let metadata: CheckpointMetadata = serde_json::from_str(&meta_json)
        .map_err(|e| NeuralError::DeserializationError(format!("Invalid metadata: {e}")))?;

    // Load optimizer state if available (safetensors format)
    let optimizer_path = checkpoint_dir.join("optimizer_state.safetensors");
    let optimizer_moments = if optimizer_path.exists() {
        let opt_reader = SafeTensorsReader::from_file(&optimizer_path)?;
        Some(opt_reader.to_named_parameters()?)
    } else {
        None
    };

    Ok((model_params, metadata, optimizer_moments))
}

/// List all checkpoints in a directory, sorted by epoch.
///
/// Expects checkpoint directories to be named like `checkpoint_epoch_NNNN`.
pub fn list_checkpoints(base_dir: &Path) -> Result<Vec<(usize, PathBuf)>> {
    if !base_dir.exists() {
        return Ok(Vec::new());
    }

    let mut checkpoints = Vec::new();

    let entries = fs::read_dir(base_dir)
        .map_err(|e| NeuralError::IOError(format!("Cannot read directory: {e}")))?;

    for entry in entries {
        let entry = entry.map_err(|e| NeuralError::IOError(format!("Cannot read entry: {e}")))?;
        let path = entry.path();

        if path.is_dir() {
            let meta_path = path.join("checkpoint_meta.json");
            if meta_path.exists() {
                if let Ok(meta_json) = fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&meta_json) {
                        checkpoints.push((meta.epoch, path));
                    }
                }
            }
        }
    }

    checkpoints.sort_by_key(|(epoch, _)| *epoch);
    Ok(checkpoints)
}

/// Get the latest checkpoint in a directory
pub fn latest_checkpoint(base_dir: &Path) -> Result<Option<PathBuf>> {
    let checkpoints = list_checkpoints(base_dir)?;
    Ok(checkpoints.last().map(|(_, path)| path.clone()))
}

/// Get the best checkpoint based on validation loss
pub fn best_checkpoint(base_dir: &Path) -> Result<Option<PathBuf>> {
    let checkpoints = list_checkpoints(base_dir)?;

    let mut best: Option<(f64, PathBuf)> = None;

    for (_, path) in &checkpoints {
        let meta_path = path.join("checkpoint_meta.json");
        if let Ok(meta_json) = fs::read_to_string(&meta_path) {
            if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&meta_json) {
                if let Some(val_loss) = meta.val_loss {
                    match &best {
                        None => best = Some((val_loss, path.clone())),
                        Some((best_loss, _)) => {
                            if val_loss < *best_loss {
                                best = Some((val_loss, path.clone()));
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(best.map(|(_, path)| path))
}

/// Create a checkpoint directory name from epoch number
pub fn checkpoint_dir_name(epoch: usize) -> String {
    format!("checkpoint_epoch_{epoch:04}")
}

// ============================================================================
// Helper: simple timestamp without chrono dependency
// ============================================================================

/// Generate a simple ISO-like timestamp string using `SystemTime`
fn simple_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Approximate date calculation (not calendar-accurate but unique)
    let years = 1970 + (days / 365);
    let day_in_year = days % 365;
    let month = (day_in_year / 30) + 1;
    let day = (day_in_year % 30) + 1;

    format!("{years:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata_default() {
        let meta = CheckpointMetadata::default();
        assert_eq!(meta.epoch, 0);
        assert_eq!(meta.global_step, 0);
        assert!(!meta.training_completed);
        assert!(meta.train_loss.is_none());
        assert!(meta.val_loss.is_none());
    }

    #[test]
    fn test_checkpoint_metadata_builder() {
        let meta = CheckpointMetadata::new("ResNet", 5, 0.001)
            .with_train_loss(0.25)
            .with_val_loss(0.30)
            .with_best_val_loss(0.28)
            .with_metric("accuracy", 0.92)
            .with_total_epochs(100)
            .with_global_step(5000);

        assert_eq!(meta.architecture, "ResNet");
        assert_eq!(meta.epoch, 5);
        assert_eq!(meta.learning_rate, 0.001);
        assert_eq!(meta.train_loss, Some(0.25));
        assert_eq!(meta.val_loss, Some(0.30));
        assert_eq!(meta.best_val_loss, Some(0.28));
        assert_eq!(meta.metrics.get("accuracy"), Some(&0.92));
        assert_eq!(meta.total_epochs, Some(100));
        assert_eq!(meta.global_step, 5000);
    }

    #[test]
    fn test_checkpoint_metadata_serialization() -> Result<()> {
        let meta = CheckpointMetadata::new("BERT", 10, 0.0001)
            .with_train_loss(0.15)
            .with_val_loss(0.20);

        let json = serde_json::to_string_pretty(&meta)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        let restored: CheckpointMetadata = serde_json::from_str(&json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;

        assert_eq!(restored.architecture, "BERT");
        assert_eq!(restored.epoch, 10);
        assert_eq!(restored.train_loss, Some(0.15));

        Ok(())
    }

    #[test]
    fn test_optimizer_state_metadata() {
        let state = OptimizerStateMetadata::new("Adam");
        assert_eq!(state.optimizer_type, "Adam");
        assert_eq!(state.num_param_groups, 1);
    }

    #[test]
    fn test_param_group_state_adam() {
        let pg = ParamGroupState::adam(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(pg.learning_rate, 0.001);
        assert_eq!(pg.beta1, Some(0.9));
        assert_eq!(pg.beta2, Some(0.999));
        assert_eq!(pg.epsilon, Some(1e-8));
        assert!(pg.momentum.is_none());
    }

    #[test]
    fn test_param_group_state_sgd() {
        let pg = ParamGroupState::sgd(0.01, 0.9, 0.0001);
        assert_eq!(pg.learning_rate, 0.01);
        assert_eq!(pg.momentum, Some(0.9));
        assert_eq!(pg.weight_decay, 0.0001);
        assert!(pg.beta1.is_none());
    }

    #[test]
    fn test_save_load_checkpoint() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_checkpoint_test");
        let checkpoint_dir = test_dir.join("checkpoint_epoch_0005");

        // Clean up from any previous test runs
        let _ = fs::remove_dir_all(&test_dir);

        // Create model parameters
        let mut params = NamedParameters::new();
        params.add("layer.0.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        params.add("layer.0.bias", vec![0.1, 0.2], vec![2]);
        params.add("layer.1.weight", vec![5.0, 6.0], vec![1, 2]);
        params.add("layer.1.bias", vec![0.3], vec![1]);

        // Create optimizer moments
        let mut moments = NamedParameters::new();
        moments.add("layer.0.weight.m", vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]);
        moments.add(
            "layer.0.weight.v",
            vec![0.001, 0.002, 0.003, 0.004],
            vec![2, 2],
        );

        // Create metadata
        let meta = CheckpointMetadata::new("TestModel", 5, 0.001)
            .with_train_loss(0.25)
            .with_val_loss(0.30)
            .with_total_epochs(100);

        // Save
        save_checkpoint(&checkpoint_dir, &params, &meta, Some(&moments))?;

        // Verify files exist
        assert!(checkpoint_dir.join("model.safetensors").exists());
        assert!(checkpoint_dir.join("checkpoint_meta.json").exists());
        assert!(checkpoint_dir.join("optimizer_state.safetensors").exists());

        // Load
        let (loaded_params, loaded_meta, loaded_moments) = load_checkpoint(&checkpoint_dir)?;

        // Verify model params
        assert_eq!(loaded_params.len(), 4);
        assert_eq!(loaded_params.total_parameters(), 9); // 4+2+2+1

        let (_, values, shape) = loaded_params
            .get("layer.0.weight")
            .ok_or_else(|| NeuralError::DeserializationError("not found".to_string()))?;
        assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[2, 2]);

        // Verify metadata
        assert_eq!(loaded_meta.architecture, "TestModel");
        assert_eq!(loaded_meta.epoch, 5);
        assert_eq!(loaded_meta.learning_rate, 0.001);
        assert_eq!(loaded_meta.train_loss, Some(0.25));
        assert_eq!(loaded_meta.val_loss, Some(0.30));
        assert_eq!(loaded_meta.total_epochs, Some(100));

        // Verify optimizer moments
        assert!(loaded_moments.is_some());
        let moments = loaded_moments.expect("should have moments");
        assert_eq!(moments.len(), 2);

        // Clean up
        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_save_checkpoint_without_optimizer() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_checkpoint_no_opt");
        let checkpoint_dir = test_dir.join("checkpoint_epoch_0001");

        let _ = fs::remove_dir_all(&test_dir);

        let mut params = NamedParameters::new();
        params.add("w", vec![1.0, 2.0], vec![2]);

        let meta = CheckpointMetadata::new("Simple", 1, 0.01);

        save_checkpoint(&checkpoint_dir, &params, &meta, None)?;

        let (loaded_params, loaded_meta, loaded_moments) = load_checkpoint(&checkpoint_dir)?;

        assert_eq!(loaded_params.len(), 1);
        assert_eq!(loaded_meta.epoch, 1);
        assert!(loaded_moments.is_none());

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_list_checkpoints() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_list_checkpoints");
        let _ = fs::remove_dir_all(&test_dir);

        for epoch in [1, 5, 10] {
            let dir_name = checkpoint_dir_name(epoch);
            let dir = test_dir.join(&dir_name);

            let mut params = NamedParameters::new();
            params.add("w", vec![1.0], vec![1]);

            let meta = CheckpointMetadata::new("Test", epoch, 0.001);
            save_checkpoint(&dir, &params, &meta, None)?;
        }

        let checkpoints = list_checkpoints(&test_dir)?;
        assert_eq!(checkpoints.len(), 3);
        assert_eq!(checkpoints[0].0, 1);
        assert_eq!(checkpoints[1].0, 5);
        assert_eq!(checkpoints[2].0, 10);

        // Test latest
        let latest = latest_checkpoint(&test_dir)?;
        assert!(latest.is_some());

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_best_checkpoint() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_best_checkpoint");
        let _ = fs::remove_dir_all(&test_dir);

        let losses = [(1, 0.50), (2, 0.35), (3, 0.30), (4, 0.32), (5, 0.28)];

        for (epoch, val_loss) in &losses {
            let dir = test_dir.join(checkpoint_dir_name(*epoch));
            let mut params = NamedParameters::new();
            params.add("w", vec![1.0], vec![1]);

            let meta = CheckpointMetadata::new("Test", *epoch, 0.001).with_val_loss(*val_loss);

            save_checkpoint(&dir, &params, &meta, None)?;
        }

        let best = best_checkpoint(&test_dir)?;
        assert!(best.is_some());

        // Load the best checkpoint and verify it's epoch 5 (val_loss=0.28)
        let (_, meta, _) = load_checkpoint(&best.expect("best should exist"))?;
        assert_eq!(meta.epoch, 5);
        assert_eq!(meta.val_loss, Some(0.28));

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_checkpoint_dir_name() {
        assert_eq!(checkpoint_dir_name(0), "checkpoint_epoch_0000");
        assert_eq!(checkpoint_dir_name(1), "checkpoint_epoch_0001");
        assert_eq!(checkpoint_dir_name(42), "checkpoint_epoch_0042");
        assert_eq!(checkpoint_dir_name(999), "checkpoint_epoch_0999");
        assert_eq!(checkpoint_dir_name(10000), "checkpoint_epoch_10000");
    }

    #[test]
    fn test_load_nonexistent_checkpoint() {
        let path = std::env::temp_dir().join("nonexistent_checkpoint_xyz");
        let result = load_checkpoint(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_empty_directory() -> Result<()> {
        let path = std::env::temp_dir().join("nonexistent_dir_xyz");
        let result = list_checkpoints(&path)?;
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_timestamp_format() {
        let ts = simple_timestamp();
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
        assert!(ts.len() >= 19);
    }

    // -----------------------------------------------------------------------
    // CheckpointConfig and CheckpointManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.save_every, 1);
        assert_eq!(config.max_checkpoints, 5);
        assert!(config.save_best);
        assert_eq!(config.monitor_metric, "val_loss");
        assert!(config.minimize_metric);
    }

    #[test]
    fn test_checkpoint_config_validate_ok() {
        let config = CheckpointConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_checkpoint_config_validate_empty_metric() {
        let config = CheckpointConfig {
            monitor_metric: String::new(),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_optimizer_checkpoint_state_adam() {
        let state = OptimizerCheckpointState::adam(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(state.optimizer_type, "Adam");
        assert_eq!(state.learning_rate, 0.001);
        assert_eq!(state.beta1, Some(0.9));
        assert_eq!(state.beta2, Some(0.999));
        assert_eq!(state.epsilon, Some(1e-8));
        assert!(!state.has_moments());
    }

    #[test]
    fn test_optimizer_checkpoint_state_sgd() {
        let state = OptimizerCheckpointState::sgd(0.01, 0.9, 0.0001);
        assert_eq!(state.optimizer_type, "SGD");
        assert_eq!(state.learning_rate, 0.01);
        assert_eq!(state.beta1, Some(0.9)); // momentum stored in beta1
    }

    #[test]
    fn test_lr_scheduler_state_cosine() {
        let state = LrSchedulerState::cosine_annealing(0.001, 100);
        assert_eq!(state.scheduler_type, "CosineAnnealing");
        assert_eq!(state.base_lr, 0.001);
        assert_eq!(state.extra_params["t_max"], 100.0);
    }

    #[test]
    fn test_lr_scheduler_state_step() {
        let state = LrSchedulerState::step_lr(0.01, 30, 0.1);
        assert_eq!(state.scheduler_type, "StepLR");
        assert_eq!(state.extra_params["step_size"], 30.0);
        assert_eq!(state.extra_params["gamma"], 0.1);
    }

    #[test]
    fn test_training_checkpoint_new() {
        let ckpt = TrainingCheckpoint::new(5, 500, "ResNet50");
        assert_eq!(ckpt.epoch, 5);
        assert_eq!(ckpt.step, 500);
        assert_eq!(ckpt.architecture, "ResNet50");
        assert!(!ckpt.training_completed);
        assert!(ckpt.best_metric.is_none());
    }

    #[test]
    fn test_training_checkpoint_latest_metric() {
        let mut ckpt = TrainingCheckpoint::new(3, 300, "BERT");
        let mut metrics = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.35);
        metrics.insert("accuracy".to_string(), 0.88);
        ckpt.metrics_history.push(metrics);

        assert_eq!(ckpt.latest_metric("val_loss"), Some(0.35));
        assert_eq!(ckpt.latest_metric("accuracy"), Some(0.88));
        assert!(ckpt.latest_metric("missing").is_none());
    }

    #[test]
    fn test_checkpoint_manager_new() {
        let config = CheckpointConfig {
            save_dir: std::env::temp_dir().join("test_ckpt_mgr"),
            save_every: 5,
            max_checkpoints: 3,
            save_best: true,
            monitor_metric: "val_loss".to_string(),
            minimize_metric: true,
        };
        let manager: CheckpointManager<f64> = CheckpointManager::new(config.clone());
        assert_eq!(manager.config().save_every, 5);
        assert_eq!(manager.config().max_checkpoints, 3);
        assert!(manager.best_metric_value().is_none());
        assert!(manager.saved_checkpoint_paths().is_empty());
    }

    #[test]
    fn test_checkpoint_manager_save_load_roundtrip() -> std::result::Result<(), CheckpointError> {
        let test_dir = std::env::temp_dir().join("scirs2_ckpt_mgr_test");
        let _ = fs::remove_dir_all(&test_dir);

        let config = CheckpointConfig {
            save_dir: test_dir.clone(),
            save_every: 1,
            max_checkpoints: 10,
            save_best: true,
            monitor_metric: "val_loss".to_string(),
            minimize_metric: true,
        };
        let mut manager: CheckpointManager<f64> = CheckpointManager::new(config);

        // Build model params
        let mut params = NamedParameters::new();
        params.add("fc.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        params.add("fc.bias", vec![0.5, 0.5], vec![2]);

        // Build checkpoint
        let mut ckpt = TrainingCheckpoint::new(10, 1000, "TestNet");
        ckpt.best_metric = Some(0.32);
        let mut epoch_metrics_map: HashMap<String, f64> = HashMap::new();
        epoch_metrics_map.insert("val_loss".to_string(), 0.32);
        ckpt.metrics_history.push(epoch_metrics_map);

        // Metrics as F-typed (f64)
        let mut metrics: HashMap<String, f64> = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.32);

        let saved_path = manager.save(&ckpt, &params, 10, &metrics)?;

        // Verify saved directory exists
        assert!(saved_path.exists() || test_dir.exists());

        // Load it back
        let ckpt_path = test_dir.join("epoch_0010.ckpt");
        assert!(
            ckpt_path.exists(),
            "Checkpoint dir should exist: {:?}",
            ckpt_path
        );

        let (loaded_ckpt, loaded_params) = CheckpointManager::<f64>::load(&ckpt_path)?;
        assert_eq!(loaded_ckpt.epoch, 10);
        assert_eq!(loaded_ckpt.step, 1000);
        assert_eq!(loaded_ckpt.architecture, "TestNet");
        assert_eq!(loaded_params.total_parameters(), 6); // 4 + 2

        // Best checkpoint should also be saved
        let best = CheckpointManager::<f64>::load_best(&test_dir)?;
        assert!(best.is_some());
        let (best_ckpt, _) = best.expect("best ckpt");
        assert_eq!(best_ckpt.epoch, 10);

        // list_checkpoints
        let list = CheckpointManager::<f64>::list_checkpoints(&test_dir)?;
        assert_eq!(list.len(), 1);

        // Clean up
        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_checkpoint_manager_max_checkpoints_cleanup() -> std::result::Result<(), CheckpointError>
    {
        let test_dir = std::env::temp_dir().join("scirs2_ckpt_mgr_cleanup");
        let _ = fs::remove_dir_all(&test_dir);

        let config = CheckpointConfig {
            save_dir: test_dir.clone(),
            save_every: 1,
            max_checkpoints: 2, // Keep only 2
            save_best: false,
            monitor_metric: "val_loss".to_string(),
            minimize_metric: true,
        };
        let mut manager: CheckpointManager<f64> = CheckpointManager::new(config);

        let mut params = NamedParameters::new();
        params.add("w", vec![1.0, 2.0], vec![2]);

        let mut metrics: HashMap<String, f64> = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.5);

        // Save 4 checkpoints - only 2 should remain
        for epoch in [0, 1, 2, 3] {
            let ckpt = TrainingCheckpoint::new(epoch, epoch * 100, "TestNet");
            manager.save(&ckpt, &params, epoch, &metrics)?;
        }

        // Only max_checkpoints (2) should remain
        assert_eq!(manager.saved_checkpoint_paths().len(), 2);

        // The saved ones should be the newest
        let list = CheckpointManager::<f64>::list_checkpoints(&test_dir)?;
        assert_eq!(list.len(), 2);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_checkpoint_manager_save_best_tracking() -> std::result::Result<(), CheckpointError> {
        let test_dir = std::env::temp_dir().join("scirs2_ckpt_mgr_best");
        let _ = fs::remove_dir_all(&test_dir);

        let config = CheckpointConfig {
            save_dir: test_dir.clone(),
            save_every: 1,
            max_checkpoints: 10,
            save_best: true,
            monitor_metric: "val_loss".to_string(),
            minimize_metric: true,
        };
        let mut manager: CheckpointManager<f64> = CheckpointManager::new(config);

        let mut params = NamedParameters::new();
        params.add("w", vec![1.0], vec![1]);

        let val_losses: Vec<f64> = vec![0.9, 0.7, 0.5, 0.6, 0.4, 0.45];

        for (i, &val_loss) in val_losses.iter().enumerate() {
            let ckpt = TrainingCheckpoint::new(i, i * 100, "Net");
            let mut metrics = HashMap::new();
            metrics.insert("val_loss".to_string(), val_loss);
            manager.save(&ckpt, &params, i, &metrics)?;
        }

        // Best should be 0.4 (epoch 4)
        assert_eq!(manager.best_metric_value(), Some(0.4));

        let best = CheckpointManager::<f64>::load_best(&test_dir)?;
        assert!(best.is_some());
        let (best_ckpt, _) = best.expect("best ckpt");
        assert_eq!(best_ckpt.epoch, 4);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_checkpoint_manager_load_latest() -> std::result::Result<(), CheckpointError> {
        let test_dir = std::env::temp_dir().join("scirs2_ckpt_mgr_latest");
        let _ = fs::remove_dir_all(&test_dir);

        let config = CheckpointConfig {
            save_dir: test_dir.clone(),
            save_every: 1,
            max_checkpoints: 10,
            save_best: false,
            monitor_metric: "val_loss".to_string(),
            minimize_metric: true,
        };
        let mut manager: CheckpointManager<f64> = CheckpointManager::new(config);

        let mut params = NamedParameters::new();
        params.add("w", vec![1.0], vec![1]);
        let mut metrics = HashMap::new();
        metrics.insert("val_loss".to_string(), 0.3f64);

        for epoch in 0..5 {
            let ckpt = TrainingCheckpoint::new(epoch, epoch * 50, "Net");
            manager.save(&ckpt, &params, epoch, &metrics)?;
        }

        let latest = CheckpointManager::<f64>::load_latest(&test_dir)?;
        assert!(latest.is_some());
        let (latest_ckpt, _) = latest.expect("latest");
        assert_eq!(latest_ckpt.epoch, 4);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_checkpoint_manager_load_best_no_best() -> std::result::Result<(), CheckpointError> {
        let test_dir = std::env::temp_dir().join("scirs2_ckpt_no_best");
        let _ = fs::remove_dir_all(&test_dir);
        let result = CheckpointManager::<f64>::load_best(&test_dir)?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn test_checkpoint_manager_load_latest_empty() -> std::result::Result<(), CheckpointError> {
        let test_dir = std::env::temp_dir().join("scirs2_ckpt_empty_latest");
        let _ = fs::remove_dir_all(&test_dir);
        let result = CheckpointManager::<f64>::load_latest(&test_dir)?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn test_checkpoint_error_display() {
        let err = CheckpointError::NotFound("/tmp/missing".to_string());
        let msg = err.to_string();
        assert!(msg.contains("missing"));

        let err2 = CheckpointError::Serialization("bad json".to_string());
        let msg2 = err2.to_string();
        assert!(msg2.contains("bad json"));
    }

    #[test]
    fn test_training_checkpoint_serialization_roundtrip() {
        let mut ckpt = TrainingCheckpoint::new(7, 700, "GPT");
        ckpt.best_metric = Some(0.28);
        ckpt.total_epochs = Some(50);
        ckpt.optimizer_state = OptimizerCheckpointState::adam(0.001, 0.9, 0.999, 1e-8);
        ckpt.lr_scheduler_state = Some(LrSchedulerState::cosine_annealing(0.001, 50));

        let json = serde_json::to_string_pretty(&ckpt).expect("serialize");
        let restored: TrainingCheckpoint = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.epoch, 7);
        assert_eq!(restored.step, 700);
        assert_eq!(restored.architecture, "GPT");
        assert_eq!(restored.best_metric, Some(0.28));
        assert_eq!(restored.total_epochs, Some(50));
        assert_eq!(restored.optimizer_state.optimizer_type, "Adam");
        assert!(restored.lr_scheduler_state.is_some());
    }
}
