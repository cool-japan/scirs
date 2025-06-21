// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Distributed training support for the array protocol.
//!
//! This module provides utilities for distributed training of neural networks
//! using the array protocol. It includes data-parallel and model-parallel
//! training strategies, parameter synchronization, and distributed optimization.

use std::fmt;
use std::sync::Arc;

use crate::array_protocol::neural::Sequential;
use crate::array_protocol::training::{DataLoader, Dataset, Metrics, Trainer, TrainingCallback};
use crate::array_protocol::ArrayProtocol;
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Distributed training strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedStrategy {
    /// Data parallelism - same model on each worker, different data.
    DataParallel,

    /// Model parallelism - different parts of the model on each worker.
    ModelParallel,

    /// Hybrid parallelism - combination of data and model parallelism.
    HybridParallel,

    /// Pipeline parallelism - model stages executed in a pipeline.
    PipelineParallel,
}

impl fmt::Display for DistributedStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataParallel => write!(f, "DataParallel"),
            Self::ModelParallel => write!(f, "ModelParallel"),
            Self::HybridParallel => write!(f, "HybridParallel"),
            Self::PipelineParallel => write!(f, "PipelineParallel"),
        }
    }
}

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    /// Distributed training strategy.
    pub strategy: DistributedStrategy,

    /// Number of workers.
    pub num_workers: usize,

    /// Rank of the current worker.
    pub rank: usize,

    /// Whether this worker is the master.
    pub is_master: bool,

    /// Synchronization interval (in batches).
    pub sync_interval: usize,

    /// Communication backend.
    pub backend: String,

    /// Whether to use mixed precision training.
    pub mixed_precision: bool,

    /// Gradient accumulation steps.
    pub gradient_accumulation_steps: usize,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            strategy: DistributedStrategy::DataParallel,
            num_workers: 1,
            rank: 0,
            is_master: true,
            sync_interval: 1,
            backend: "threaded".to_string(),
            mixed_precision: false,
            gradient_accumulation_steps: 1,
        }
    }
}

/// A node in a distributed training cluster.
pub struct DistributedNode {
    /// Configuration for the node.
    config: DistributedTrainingConfig,

    /// The model being trained.
    model: Sequential,

    /// Communication channel to other nodes (kept private to avoid warning).
    _channel: CommunicationChannel,
}

impl DistributedNode {
    /// Create a new distributed node.
    pub fn new(
        model: Sequential,
        config: DistributedTrainingConfig,
        channel: Box<dyn DistributedCommunication>,
    ) -> Self {
        Self {
            config,
            model,
            _channel: CommunicationChannel::new(channel),
        }
    }

    /// Synchronize model parameters with other nodes.
    pub fn synchronize_parameters(&mut self) -> CoreResult<()> {
        match self.config.strategy {
            DistributedStrategy::DataParallel => {
                // In data parallelism, we average the gradients across workers
                self.average_gradients()?;
            }
            DistributedStrategy::ModelParallel => {
                // In model parallelism, we exchange activations and gradients
                // between adjacent layers
                self.exchange_activations_and_gradients()?;
            }
            DistributedStrategy::HybridParallel => {
                // In hybrid parallelism, we do a combination of both
                self.average_gradients()?;
                self.exchange_activations_and_gradients()?;
            }
            DistributedStrategy::PipelineParallel => {
                // In pipeline parallelism, we maintain a pipeline of batches
                self.pipeline_forward_backward()?;
            }
        }

        Ok(())
    }

    /// Average gradients across workers.
    fn average_gradients(&self) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would use the DistributedCommunication
        // channel to exchange gradients with other workers.

        // 1. Get model parameters
        let params = self.model.parameters();

        // 2. For each parameter, send gradient to other workers and receive their gradients
        for _param in params {
            // Example: In a real implementation, we would do something like:
            // let gradient = param.grad()?;
            // let averaged_gradient = self.channel.all_reduce(gradient, "mean")?;
            // param.set_grad(averaged_gradient)?;
        }

        Ok(())
    }

    /// Exchange activations and gradients between adjacent layers.
    fn exchange_activations_and_gradients(&self) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would use the DistributedCommunication
        // channel to exchange activations and gradients with adjacent workers.

        // For model parallelism, each worker has a subset of the model's layers.
        // During forward pass:
        //   - Worker i computes activations for its layers
        //   - Worker i sends activations to worker i+1
        //   - Worker i+1 receives activations from worker i
        //
        // During backward pass:
        //   - Worker i+1 computes gradients for its layers
        //   - Worker i+1 sends gradients to worker i
        //   - Worker i receives gradients from worker i+1

        Ok(())
    }

    /// Implement pipeline parallelism.
    fn pipeline_forward_backward(&self) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would maintain a pipeline of mini-batches.

        // In pipeline parallelism:
        // - The model is divided into stages, with each stage on a different worker
        // - Multiple mini-batches are processed concurrently
        // - When worker i finishes processing a mini-batch, it sends the activations
        //   to worker i+1 and starts processing the next mini-batch
        // - This creates a pipeline where different workers are processing different
        //   mini-batches at the same time

        Ok(())
    }
}

/// Trait for distributed communication between nodes.
pub trait DistributedCommunication: Send + Sync {
    /// Send a tensor to another worker.
    fn send(&self, tensor: Box<dyn ArrayProtocol>, destination: usize) -> CoreResult<()>;

    /// Receive a tensor from another worker.
    fn recv(&self, source: usize) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// Broadcast a tensor from the master to all workers.
    fn broadcast(&self, tensor: Box<dyn ArrayProtocol>) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// Gather tensors from all workers to the master.
    fn gather(&self, tensor: Box<dyn ArrayProtocol>) -> CoreResult<Vec<Box<dyn ArrayProtocol>>>;

    /// Scatter tensors from the master to all workers.
    fn scatter(&self, tensors: Vec<Box<dyn ArrayProtocol>>) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// Reduce tensors from all workers to the master.
    fn reduce(
        &self,
        tensor: Box<dyn ArrayProtocol>,
        op: &str,
    ) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// All-reduce tensors across all workers.
    fn all_reduce(
        &self,
        tensor: Box<dyn ArrayProtocol>,
        op: &str,
    ) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// All-gather tensors from all workers to all workers.
    fn all_gather(&self, tensor: Box<dyn ArrayProtocol>)
        -> CoreResult<Vec<Box<dyn ArrayProtocol>>>;

    /// Barrier synchronization.
    fn barrier(&self) -> CoreResult<()>;

    /// Clone this communication channel.
    fn box_clone(&self) -> Box<dyn DistributedCommunication>;
}

/// A wrapper type that makes `Box<dyn DistributedCommunication>` cloneable
#[derive(Clone)]
pub struct CommunicationChannel(Arc<Box<dyn DistributedCommunication>>);

impl CommunicationChannel {
    /// Create a new communication channel from a communication implementation.
    pub fn new(comm: Box<dyn DistributedCommunication>) -> Self {
        Self(Arc::new(comm))
    }

    /// Get the underlying communication implementation.
    pub fn inner(&self) -> &dyn DistributedCommunication {
        self.0.as_ref().as_ref()
    }
}

/// Make the `Box<dyn DistributedCommunication>` cloneable via box_clone
impl Clone for Box<dyn DistributedCommunication> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// A mock implementation of distributed communication for testing.
pub struct MockDistributedCommunication {
    /// Number of workers.
    num_workers: usize,

    /// Rank of the current worker.
    rank: usize,
}

impl MockDistributedCommunication {
    /// Create a new mock distributed communication channel.
    pub fn new(num_workers: usize, rank: usize) -> Self {
        Self { num_workers, rank }
    }
}

impl DistributedCommunication for MockDistributedCommunication {
    fn send(&self, _tensor: Box<dyn ArrayProtocol>, _destination: usize) -> CoreResult<()> {
        // In a real implementation, this would send the tensor to the destination worker
        Ok(())
    }

    fn recv(&self, _source: usize) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would receive a tensor from the source worker
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "recv not implemented for MockDistributedCommunication".to_string(),
        )))
    }

    fn broadcast(&self, tensor: Box<dyn ArrayProtocol>) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would broadcast the tensor to all workers
        Ok(tensor)
    }

    fn gather(&self, tensor: Box<dyn ArrayProtocol>) -> CoreResult<Vec<Box<dyn ArrayProtocol>>> {
        // In a real implementation, this would gather tensors from all workers
        Ok(vec![tensor])
    }

    fn scatter(&self, tensors: Vec<Box<dyn ArrayProtocol>>) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would scatter tensors to all workers
        if tensors.is_empty() {
            return Err(CoreError::InvalidArgument(ErrorContext::new(
                "Empty tensors list for scatter".to_string(),
            )));
        }

        Ok(tensors[0].clone())
    }

    fn reduce(
        &self,
        tensor: Box<dyn ArrayProtocol>,
        op: &str,
    ) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would reduce tensors across all workers
        match op {
            "sum" | "mean" => Ok(tensor),
            _ => Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Unknown reduction operation: {}",
                op
            )))),
        }
    }

    fn all_reduce(
        &self,
        tensor: Box<dyn ArrayProtocol>,
        op: &str,
    ) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would all-reduce tensors across all workers
        match op {
            "sum" | "mean" => Ok(tensor),
            _ => Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Unknown reduction operation: {}",
                op
            )))),
        }
    }

    fn all_gather(
        &self,
        tensor: Box<dyn ArrayProtocol>,
    ) -> CoreResult<Vec<Box<dyn ArrayProtocol>>> {
        // In a real implementation, this would all-gather tensors from all workers
        Ok(vec![tensor])
    }

    fn barrier(&self) -> CoreResult<()> {
        // In a real implementation, this would synchronize all workers
        Ok(())
    }

    fn box_clone(&self) -> Box<dyn DistributedCommunication> {
        Box::new(MockDistributedCommunication {
            num_workers: self.num_workers,
            rank: self.rank,
        })
    }
}

/// Distributed Dataset that partitions data across workers.
pub struct DistributedDataset {
    /// The underlying dataset.
    dataset: Box<dyn Dataset>,

    /// Number of workers (kept private to avoid warning).
    _num_workers: usize,

    /// Rank of the current worker (kept private to avoid warning).
    _rank: usize,

    /// Indices of samples assigned to this worker.
    indices: Vec<usize>,
}

impl DistributedDataset {
    /// Create a new distributed dataset.
    pub fn new(dataset: Box<dyn Dataset>, num_workers: usize, rank: usize) -> Self {
        let num_samples = dataset.len();
        let samples_per_worker = num_samples / num_workers;
        let remainder = num_samples % num_workers;

        let start = if rank < remainder {
            rank * (samples_per_worker + 1)
        } else {
            rank * samples_per_worker + remainder
        };

        let end = if rank < remainder {
            start + samples_per_worker + 1
        } else {
            start + samples_per_worker
        };

        let indices = (start..end).collect();

        Self {
            dataset,
            _num_workers: num_workers,
            _rank: rank,
            indices,
        }
    }
}

impl Dataset for DistributedDataset {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Option<(Box<dyn ArrayProtocol>, Box<dyn ArrayProtocol>)> {
        if index >= self.len() {
            return None;
        }

        let global_index = self.indices[index];
        self.dataset.get(global_index)
    }

    fn input_shape(&self) -> Vec<usize> {
        self.dataset.input_shape()
    }

    fn output_shape(&self) -> Vec<usize> {
        self.dataset.output_shape()
    }
}

/// Distributed Trainer for handling distributed training.
pub struct DistributedTrainer {
    /// The underlying trainer.
    trainer: Trainer,

    /// Configuration for distributed training.
    config: DistributedTrainingConfig,

    /// Communication channel to other nodes.
    channel: CommunicationChannel,

    /// Batch counter for synchronization (kept private to avoid warning).
    _batch_counter: usize,
}

impl DistributedTrainer {
    /// Create a new distributed trainer.
    pub fn new(
        trainer: Trainer,
        config: DistributedTrainingConfig,
        channel: Box<dyn DistributedCommunication>,
    ) -> Self {
        Self {
            trainer,
            config,
            channel: CommunicationChannel::new(channel),
            _batch_counter: 0,
        }
    }

    /// Train the model in a distributed setting.
    pub fn train(
        &mut self,
        train_loader: DataLoader,
        num_epochs: usize,
        val_loader: Option<DataLoader>,
    ) -> CoreResult<()> {
        // Synchronize initial model parameters
        self.synchronize_parameters()?;

        // Train the model
        if self.config.strategy == DistributedStrategy::DataParallel {
            // For data parallelism, we can use the regular trainer
            // but with periodic parameter synchronization
            self.train_data_parallel(train_loader, num_epochs, val_loader)?;
        } else {
            // For other strategies, we need custom training loops
            match self.config.strategy {
                DistributedStrategy::ModelParallel => {
                    self.train_model_parallel(train_loader, num_epochs, val_loader)?;
                }
                DistributedStrategy::HybridParallel => {
                    self.train_hybrid_parallel(train_loader, num_epochs, val_loader)?;
                }
                DistributedStrategy::PipelineParallel => {
                    self.train_pipeline_parallel(train_loader, num_epochs, val_loader)?;
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    /// Synchronize model parameters with other workers.
    fn synchronize_parameters(&self) -> CoreResult<()> {
        // In a real implementation, this would synchronize model parameters
        // across all workers.

        // If this is the master worker, broadcast parameters to all workers
        // Otherwise, receive parameters from the master

        // For simplicity, we'll just call barrier to synchronize all workers
        self.channel.inner().barrier()?;

        Ok(())
    }

    /// Train the model using data parallelism.
    fn train_data_parallel(
        &mut self,
        train_loader: DataLoader,
        num_epochs: usize,
        val_loader: Option<DataLoader>,
    ) -> CoreResult<()> {
        // Create a callback for parameter synchronization
        let _sync_callback = ParameterSyncCallback::new(
            self.config.sync_interval,
            self.channel.0.clone().box_clone(),
        );

        // Add the callback to the trainer
        // self.trainer.add_callback(Box::new(sync_callback));

        // Train the model using the regular trainer
        self.trainer.train(train_loader, num_epochs, val_loader)?;

        Ok(())
    }

    /// Train the model using model parallelism.
    fn train_model_parallel(
        &self,
        _train_loader: DataLoader,
        _num_epochs: usize,
        _val_loader: Option<DataLoader>,
    ) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would implement a custom training loop
        // that exchanges activations and gradients between workers.

        Ok(())
    }

    /// Train the model using hybrid parallelism.
    fn train_hybrid_parallel(
        &self,
        _train_loader: DataLoader,
        _num_epochs: usize,
        _val_loader: Option<DataLoader>,
    ) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would implement a custom training loop
        // that combines data and model parallelism.

        Ok(())
    }

    /// Train the model using pipeline parallelism.
    fn train_pipeline_parallel(
        &self,
        _train_loader: DataLoader,
        _num_epochs: usize,
        _val_loader: Option<DataLoader>,
    ) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would implement a custom training loop
        // that uses pipeline parallelism.

        Ok(())
    }
}

/// Callback for synchronizing parameters between workers.
pub struct ParameterSyncCallback {
    /// Synchronization interval (in batches).
    sync_interval: usize,

    /// Batch counter.
    batch_counter: usize,

    /// Communication channel to other workers.
    channel: CommunicationChannel,
}

impl ParameterSyncCallback {
    /// Create a new parameter synchronization callback.
    pub fn new(sync_interval: usize, channel: Box<dyn DistributedCommunication>) -> Self {
        Self {
            sync_interval,
            batch_counter: 0,
            channel: CommunicationChannel::new(channel),
        }
    }
}

impl TrainingCallback for ParameterSyncCallback {
    fn on_epoch_start(&mut self, _epoch: usize, _num_epochs: usize) {
        // Reset batch counter at the start of each epoch
        self.batch_counter = 0;
    }

    fn on_epoch_end(&mut self, _epoch: usize, _num_epochs: usize, _metrics: &Metrics) {
        // Synchronize parameters at the end of each epoch
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would call channel.all_reduce() for each parameter.

        match self.channel.inner().barrier() {
            Ok(()) => {}
            Err(e) => eprintln!("Error in barrier synchronization: {}", e),
        }
    }

    fn on_batch_start(&mut self, _batch: usize, _num_batches: usize) {
        // No-op for this callback
    }

    fn on_batch_end(&mut self, _batch: usize, _num_batches: usize, _loss: f64) {
        // Increment batch counter
        self.batch_counter += 1;

        // Synchronize parameters if needed
        if self.batch_counter % self.sync_interval == 0 {
            // This is a simplified implementation for demonstration purposes.
            // In a real implementation, this would call channel.all_reduce() for each parameter.

            match self.channel.inner().barrier() {
                Ok(()) => {}
                Err(e) => eprintln!("Error in barrier synchronization: {}", e),
            }
        }
    }

    fn on_train_start(&mut self, _num_epochs: usize) {
        // Synchronize initial parameters
        match self.channel.inner().barrier() {
            Ok(()) => {}
            Err(e) => eprintln!("Error in barrier synchronization: {}", e),
        }
    }

    fn on_train_end(&mut self, _metrics: &Metrics) {
        // Final synchronization
        match self.channel.inner().barrier() {
            Ok(()) => {}
            Err(e) => eprintln!("Error in barrier synchronization: {}", e),
        }
    }
}

/// Factory for creating distributed training components.
pub struct DistributedTrainingFactory;

impl DistributedTrainingFactory {
    /// Create a new distributed dataset.
    pub fn create_dataset(
        dataset: Box<dyn Dataset>,
        config: &DistributedTrainingConfig,
    ) -> Box<dyn Dataset> {
        Box::new(DistributedDataset::new(
            dataset,
            config.num_workers,
            config.rank,
        ))
    }

    /// Create a new distributed trainer.
    pub fn create_trainer(
        trainer: Trainer,
        config: DistributedTrainingConfig,
    ) -> DistributedTrainer {
        // Create communication channel
        let channel: Box<dyn DistributedCommunication> = match config.backend.as_str() {
            "threaded" => Box::new(MockDistributedCommunication::new(
                config.num_workers,
                config.rank,
            )),
            // Other backends would be added here
            _ => Box::new(MockDistributedCommunication::new(
                config.num_workers,
                config.rank,
            )),
        };

        DistributedTrainer::new(trainer, config, channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol::training::InMemoryDataset;
    use crate::array_protocol::NdarrayWrapper;
    use ndarray::Array2;

    #[test]
    fn test_distributed_dataset() {
        // Create a dataset
        let inputs = Array2::<f64>::ones((10, 5));
        let targets = Array2::<f64>::zeros((10, 2));
        let dataset = Box::new(InMemoryDataset::from_arrays(inputs, targets));

        // Create a distributed dataset
        let dist_dataset = DistributedDataset::new(dataset, 2, 0);

        // Check properties
        assert_eq!(dist_dataset.len(), 5);
        assert_eq!(dist_dataset.input_shape(), vec![5]);
        assert_eq!(dist_dataset.output_shape(), vec![2]);

        // Get a sample
        let (input, target) = dist_dataset.get(0).unwrap();
        assert!(input
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, ndarray::IxDyn>>()
            .is_some());
        assert!(target
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, ndarray::IxDyn>>()
            .is_some());
    }

    #[test]
    fn test_mock_distributed_communication() {
        // Create a mock distributed communication channel
        let channel = MockDistributedCommunication::new(2, 0);

        // Create a tensor
        let tensor = NdarrayWrapper::new(Array2::<f64>::ones((2, 2)));
        let boxed_tensor = Box::new(tensor);

        // Test broadcast
        let result = channel.broadcast(boxed_tensor.clone());
        assert!(result.is_ok());

        // Test all_reduce
        let result = channel.all_reduce(boxed_tensor.clone(), "mean");
        assert!(result.is_ok());

        // Test barrier
        let result = channel.barrier();
        assert!(result.is_ok());
    }
}
