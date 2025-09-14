//! Message passing system for distributed clustering coordination
//!
//! This module provides the messaging infrastructure for coordinating
//! distributed clustering operations across multiple worker nodes.

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};

/// Message types for distributed clustering coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringMessage<F: Float> {
    /// Initialize worker with partition data
    InitializeWorker {
        workerid: usize,
        partition_data: Array2<F>,
        initial_centroids: Array2<F>,
    },
    /// Update global centroids
    UpdateCentroids { round: usize, centroids: Array2<F> },
    /// Request local computation
    ComputeLocal { round: usize, max_iterations: usize },
    /// Local computation result
    LocalResult {
        workerid: usize,
        round: usize,
        local_centroids: Array2<F>,
        local_labels: Array1<usize>,
        local_inertia: f64,
        computation_time_ms: u64,
    },
    /// Heartbeat for health monitoring
    Heartbeat {
        workerid: usize,
        timestamp: u64,
        cpu_usage: f64,
        memory_usage: f64,
    },
    /// Synchronization barrier
    SyncBarrier {
        round: usize,
        participant_count: usize,
    },
    /// Convergence check result
    ConvergenceCheck {
        round: usize,
        converged: bool,
        max_centroid_movement: f64,
    },
    /// Terminate worker
    Terminate,
    /// Checkpoint creation request
    CreateCheckpoint { round: usize },
    /// Checkpoint data
    CheckpointData {
        workerid: usize,
        round: usize,
        centroids: Array2<F>,
        labels: Array1<usize>,
    },
    /// Recovery request
    RecoveryRequest {
        failed_workerid: usize,
        recovery_strategy: RecoveryStrategy,
    },
    /// Load balancing request
    LoadBalance {
        target_worker_loads: HashMap<usize, f64>,
    },
    /// Data migration for load balancing
    MigrateData {
        source_worker: usize,
        target_worker: usize,
        data_subset: Array2<F>,
    },
    /// Acknowledgment message
    Acknowledgment { workerid: usize, message_id: u64 },
}

/// Recovery strategies for failed workers
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Redistribute failed worker's data to other workers
    Redistribute,
    /// Replace failed worker with a new one
    Replace,
    /// Restore from checkpoint
    Checkpoint,
    /// Restart entire computation
    Restart,
    /// Continue with degraded performance
    Degrade,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Critical = 0, // Immediate processing required
    High = 1,     // High priority
    Normal = 2,   // Normal processing
    Low = 3,      // Background processing
}

/// Message envelope with metadata
#[derive(Debug, Clone)]
pub struct MessageEnvelope<F: Float> {
    pub message_id: u64,
    pub sender_id: usize,
    pub receiver_id: usize,
    pub priority: MessagePriority,
    pub timestamp: u64,
    pub retry_count: u32,
    pub timeout_ms: u64,
    pub message: ClusteringMessage<F>,
}

/// Message passing coordinator for distributed clustering
#[derive(Debug)]
pub struct MessagePassingCoordinator<F: Float> {
    pub coordinator_id: usize,
    pub worker_channels: HashMap<usize, Sender<MessageEnvelope<F>>>,
    pub coordinator_receiver: Receiver<MessageEnvelope<F>>,
    pub coordinator_sender: Sender<MessageEnvelope<F>>,
    pub message_counter: Arc<Mutex<u64>>,
    pub pending_messages: HashMap<u64, MessageEnvelope<F>>,
    pub message_timeouts: HashMap<u64, Instant>,
    pub worker_status: HashMap<usize, WorkerStatus>,
    pub sync_barriers: HashMap<usize, SynchronizationBarrier>,
    pub config: MessagePassingConfig,
}

/// Worker status for health monitoring
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerStatus {
    Active,
    Inactive,
    Failed,
    Recovering,
}

/// Configuration for message passing system
#[derive(Debug, Clone)]
pub struct MessagePassingConfig {
    pub max_message_queue_size: usize,
    pub message_timeout_ms: u64,
    pub max_retry_attempts: u32,
    pub heartbeat_interval_ms: u64,
    pub sync_timeout_ms: u64,
    pub enable_message_compression: bool,
    pub enable_message_ordering: bool,
    pub batch_size: usize,
}

impl Default for MessagePassingConfig {
    fn default() -> Self {
        Self {
            max_message_queue_size: 1000,
            message_timeout_ms: 30000,
            max_retry_attempts: 3,
            heartbeat_interval_ms: 5000,
            sync_timeout_ms: 60000,
            enable_message_compression: false,
            enable_message_ordering: true,
            batch_size: 10,
        }
    }
}

/// Synchronization barrier for coordinating worker phases
#[derive(Debug)]
pub struct SynchronizationBarrier {
    pub round: usize,
    pub expected_participants: usize,
    pub arrived_participants: HashSet<usize>,
    pub barrier_start_time: Instant,
    pub timeout_ms: u64,
}

impl<F: Float + Debug + Send + Sync + 'static> MessagePassingCoordinator<F> {
    /// Create new message passing coordinator
    pub fn new(coordinatorid: usize, config: MessagePassingConfig) -> Self {
        let (coordinator_sender, coordinator_receiver) = mpsc::channel();

        Self {
            coordinator_id: coordinatorid,
            worker_channels: HashMap::new(),
            coordinator_receiver,
            coordinator_sender,
            message_counter: Arc::new(Mutex::new(0)),
            pending_messages: HashMap::new(),
            message_timeouts: HashMap::new(),
            worker_status: HashMap::new(),
            sync_barriers: HashMap::new(),
            config,
        }
    }

    /// Register a new worker with the coordinator
    pub fn register_worker(&mut self, workerid: usize) -> Receiver<MessageEnvelope<F>> {
        let (sender, receiver) = mpsc::channel();
        self.worker_channels.insert(workerid, sender);
        self.worker_status.insert(workerid, WorkerStatus::Active);
        receiver
    }

    /// Send message to a specific worker
    pub fn send_message_to_worker(
        &mut self,
        workerid: usize,
        message: ClusteringMessage<F>,
        priority: MessagePriority,
    ) -> Result<u64> {
        let message_id = {
            let mut counter = self.message_counter.lock().unwrap();
            *counter += 1;
            *counter
        };

        let envelope = MessageEnvelope {
            message_id,
            sender_id: self.coordinator_id,
            receiver_id: workerid,
            priority,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            retry_count: 0,
            timeout_ms: self.config.message_timeout_ms,
            message,
        };

        if let Some(sender) = self.worker_channels.get(&workerid) {
            sender.send(envelope.clone()).map_err(|_| {
                ClusteringError::InvalidInput(format!("Worker {} unavailable", workerid))
            })?;

            self.pending_messages.insert(message_id, envelope);
            self.message_timeouts.insert(message_id, Instant::now());
            Ok(message_id)
        } else {
            Err(ClusteringError::InvalidInput(format!(
                "Worker {} not registered",
                workerid
            )))
        }
    }

    /// Broadcast message to all workers
    pub fn broadcast_message(
        &mut self,
        message: ClusteringMessage<F>,
        priority: MessagePriority,
    ) -> Result<Vec<u64>> {
        let workerids: Vec<usize> = self.worker_channels.keys().copied().collect();
        let mut message_ids = Vec::new();

        for workerid in workerids {
            let message_id = self.send_message_to_worker(workerid, message.clone(), priority)?;
            message_ids.push(message_id);
        }

        Ok(message_ids)
    }

    /// Process incoming messages from workers
    pub fn process_messages(&mut self, timeout: Duration) -> Result<Vec<MessageEnvelope<F>>> {
        let mut messages = Vec::new();
        let deadline = Instant::now() + timeout;

        while Instant::now() < deadline {
            match self.coordinator_receiver.try_recv() {
                Ok(envelope) => {
                    messages.push(envelope);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No more messages available
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(ClusteringError::InvalidInput(
                        "Coordinator channel disconnected".to_string(),
                    ));
                }
            }
        }

        // Clean up timed-out messages
        self.cleanup_timed_out_messages();

        Ok(messages)
    }

    /// Create synchronization barrier
    pub fn create_sync_barrier(
        &mut self,
        round: usize,
        expected_participants: usize,
    ) -> Result<()> {
        let barrier = SynchronizationBarrier {
            round,
            expected_participants,
            arrived_participants: HashSet::new(),
            barrier_start_time: Instant::now(),
            timeout_ms: self.config.sync_timeout_ms,
        };

        self.sync_barriers.insert(round, barrier);
        Ok(())
    }

    /// Wait for workers to reach synchronization barrier
    pub fn wait_for_barrier(&mut self, round: usize) -> Result<bool> {
        if let Some(barrier) = self.sync_barriers.get_mut(&round) {
            let timeout_reached =
                barrier.barrier_start_time.elapsed().as_millis() as u64 > barrier.timeout_ms;

            if timeout_reached {
                // Remove timed-out barrier
                self.sync_barriers.remove(&round);
                return Ok(false);
            }

            let all_arrived = barrier.arrived_participants.len() >= barrier.expected_participants;
            if all_arrived {
                self.sync_barriers.remove(&round);
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err(ClusteringError::InvalidInput(format!(
                "Sync barrier for round {} not found",
                round
            )))
        }
    }

    /// Register worker arrival at synchronization barrier
    pub fn register_barrier_arrival(&mut self, round: usize, workerid: usize) -> Result<()> {
        if let Some(barrier) = self.sync_barriers.get_mut(&round) {
            barrier.arrived_participants.insert(workerid);
            Ok(())
        } else {
            Err(ClusteringError::InvalidInput(format!(
                "Sync barrier for round {} not found",
                round
            )))
        }
    }

    /// Clean up timed-out messages and retry failed sends
    fn cleanup_timed_out_messages(&mut self) {
        let now = Instant::now();
        let timeout_duration = Duration::from_millis(self.config.message_timeout_ms);

        let mut timed_out_messages = Vec::new();

        for (&message_id, &send_time) in &self.message_timeouts {
            if now.duration_since(send_time) > timeout_duration {
                timed_out_messages.push(message_id);
            }
        }

        for message_id in timed_out_messages {
            if let Some(envelope) = self.pending_messages.remove(&message_id) {
                self.message_timeouts.remove(&message_id);

                // Retry if under retry limit
                if envelope.retry_count < self.config.max_retry_attempts {
                    let mut retry_envelope = envelope;
                    retry_envelope.retry_count += 1;

                    if let Some(sender) = self.worker_channels.get(&retry_envelope.receiver_id) {
                        let _ = sender.send(retry_envelope);
                    }
                } else {
                    // Mark worker as failed after max retries
                    self.worker_status
                        .insert(envelope.receiver_id, WorkerStatus::Failed);
                }
            }
        }
    }

    /// Get worker status
    pub fn get_worker_status(&self, workerid: usize) -> Option<WorkerStatus> {
        self.worker_status.get(&workerid).copied()
    }

    /// Update worker status
    pub fn update_worker_status(&mut self, workerid: usize, status: WorkerStatus) {
        self.worker_status.insert(workerid, status);
    }

    /// Get active workers
    pub fn get_active_workers(&self) -> Vec<usize> {
        self.worker_status
            .iter()
            .filter(|(_, &status)| status == WorkerStatus::Active)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get failed workers
    pub fn get_failed_workers(&self) -> Vec<usize> {
        self.worker_status
            .iter()
            .filter(|(_, &status)| status == WorkerStatus::Failed)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Shutdown coordinator and all worker channels
    pub fn shutdown(&mut self) {
        // Send terminate message to all workers
        let _ = self.broadcast_message(ClusteringMessage::Terminate, MessagePriority::Critical);

        // Clear all state
        self.worker_channels.clear();
        self.pending_messages.clear();
        self.message_timeouts.clear();
        self.worker_status.clear();
        self.sync_barriers.clear();
    }
}

impl SynchronizationBarrier {
    /// Check if barrier is complete
    pub fn is_complete(&self) -> bool {
        self.arrived_participants.len() >= self.expected_participants
    }

    /// Check if barrier has timed out
    pub fn is_timed_out(&self) -> bool {
        self.barrier_start_time.elapsed().as_millis() as u64 > self.timeout_ms
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        self.arrived_participants.len() as f64 / self.expected_participants as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_message_passing_coordinator_creation() {
        let config = MessagePassingConfig::default();
        let coordinator = MessagePassingCoordinator::<f64>::new(0, config);

        assert_eq!(coordinator.coordinator_id, 0);
        assert!(coordinator.worker_channels.is_empty());
        assert!(coordinator.pending_messages.is_empty());
    }

    #[test]
    fn test_worker_registration() {
        let config = MessagePassingConfig::default();
        let mut coordinator = MessagePassingCoordinator::<f64>::new(0, config);

        let _receiver = coordinator.register_worker(1);
        assert!(coordinator.worker_channels.contains_key(&1));
        assert_eq!(coordinator.get_worker_status(1), Some(WorkerStatus::Active));
    }

    #[test]
    fn test_sync_barrier_creation() {
        let config = MessagePassingConfig::default();
        let mut coordinator = MessagePassingCoordinator::<f64>::new(0, config);

        let result = coordinator.create_sync_barrier(1, 3);
        assert!(result.is_ok());
        assert!(coordinator.sync_barriers.contains_key(&1));
    }

    #[test]
    fn test_sync_barrier_completion() {
        let mut barrier = SynchronizationBarrier {
            round: 1,
            expected_participants: 2,
            arrived_participants: HashSet::new(),
            barrier_start_time: Instant::now(),
            timeout_ms: 1000,
        };

        assert!(!barrier.is_complete());
        assert_relative_eq!(barrier.completion_percentage(), 0.0);

        barrier.arrived_participants.insert(1);
        assert_relative_eq!(barrier.completion_percentage(), 0.5);

        barrier.arrived_participants.insert(2);
        assert!(barrier.is_complete());
        assert_relative_eq!(barrier.completion_percentage(), 1.0);
    }
}
