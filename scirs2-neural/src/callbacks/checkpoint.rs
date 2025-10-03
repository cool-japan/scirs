//! Model checkpoint storage manager implementation

use crate::error::Result;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
/// Type alias for checkpoint loading result
type LoadResult<F> = Result<(usize, F, Option<Vec<(String, F)>>)>;
/// Checkpoint storage for model state during training
pub struct ModelCheckpoint<F: Float + Debug + ScalarOperand> {
    /// Directory to save checkpoints
    checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    max_to_keep: usize,
    /// List of saved checkpoint paths, ordered by creation time
    checkpoints: Vec<PathBuf>,
    /// Phantom data for generic type
    _phantom: std::marker::PhantomData<F>,
}
impl<F: Float + Debug + ScalarOperand> ModelCheckpoint<F> {
    /// Create a new model checkpoint storage
    ///
    /// # Arguments
    /// * `checkpoint_dir` - Directory to save checkpoints
    /// * `max_to_keep` - Maximum number of checkpoints to keep (0 means keep all)
    pub fn new<P: AsRef<Path>>(_checkpoint, dir: P, max_tokeep: usize) -> Result<Self> {
        // Create directory if it doesn't exist
        let _dir = checkpoint_dir.as_ref();
        if !_dir.exists() {
            std::fs::create_dir_all(_dir)?;
        }
        Ok(Self {
            checkpoint_dir: dir.to_path_buf(),
            max_to_keep,
            checkpoints: Vec::new(), _phantom: std::marker::PhantomData,
        })
    }
    /// Save a model checkpoint
    /// * `epoch` - Current epoch number
    /// * `model` - Model to save
    /// * `optimizer` - Optimizer to save
    /// * `loss` - Current loss value
    /// * `metrics` - Additional metrics to save
    pub fn save(
        &mut self,
        epoch: usize, model: &impl Debug, _optimizer: &impl Debug,
        loss: F,
        metrics: Option<Vec<(String, F)>>,
    ) -> Result<PathBuf> {
        // Create checkpoint filename
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint_epoch_{}.pth", epoch + 1));
        println!(
            "Saving checkpoint at epoch {} with loss {:.6?} to {}",
            epoch + 1,
            loss,
            checkpoint_path.display()
        );
        if let Some(metrics) = &metrics {
            for (name, value) in metrics {
                println!("  Metric {}: {:.6?}", name, value);
            }
        // In a real implementation, we'd save the model and optimizer state here
        // let state = {
        //     "epoch": epoch,
        //     "model_state_dict": model.state_dict(),
        //     "optimizer_state_dict": optimizer.state_dict(),
        //     "loss": loss,
        //     "metrics": metrics
        // };
        // torch::save(state, checkpoint_path);
        // Add to list of checkpoints
        self.checkpoints.push(checkpoint_path.clone());
        // Remove oldest checkpoints if we have too many
        if self.max_to_keep > 0 && self.checkpoints.len() > self.max_to_keep {
            let to_remove = self.checkpoints.len() - self.max_to_keep;
            for _ in 0..to_remove {
                if let Some(old_checkpoint) = self.checkpoints.first().cloned() {
                    println!("Removing old checkpoint: {}", old_checkpoint.display());
                    // In a real implementation, we'd remove the file here
                    // std::fs::remove_file(&old_checkpoint)?;
                    self.checkpoints.remove(0);
                }
        Ok(checkpoint_path)
    /// Load a model checkpoint
    /// * `checkpoint_path` - Path to the checkpoint to load
    /// * `model` - Model to load state into
    /// * `optimizer` - Optimizer to load state into
    pub fn load<P: AsRef<Path>>(
        &self,
        checkpoint_path: P, _model: &mut impl Debug, optimizer: &mut impl Debug,
    ) -> LoadResult<F> {
        let path = checkpoint_path.as_ref();
        println!("Loading checkpoint from {}", path.display());
        // In a real implementation, we'd load the model and optimizer state here
        // let checkpoint = torch::load(path)?;
        // model.load_state_dict(checkpoint["model_state_dict"]);
        // optimizer.load_state_dict(checkpoint["optimizer_state_dict"]);
        // let epoch = checkpoint["epoch"].item::<i64>() as usize;
        // let loss = checkpoint["loss"].item::<F>();
        // let metrics = checkpoint["metrics"].to_vec();
        // Mock values for example purposes
        let epoch = 0;
        let loss = F::zero();
        let metrics = None;
        Ok((epoch, loss, metrics))
    /// Get the latest checkpoint path
    pub fn latest_checkpoint(&self) -> Option<PathBuf> {
        self.checkpoints.last().cloned()
    /// Get all checkpoint paths
    pub fn all_checkpoints(&self) -> &[PathBuf] {
        &self.checkpoints
