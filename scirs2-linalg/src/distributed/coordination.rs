//! Coordination and synchronization for distributed linear algebra
//!
//! This module provides coordination primitives for managing distributed
//! linear algebra operations, including synchronization barriers,
//! distributed locks, consensus algorithms, and fault tolerance mechanisms.

use crate::error::{LinalgError, LinalgResult};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Distributed coordinator for managing synchronization across nodes
pub struct DistributedCoordinator {
    /// Current node rank
    node_rank: usize,
    /// Total number of nodes
    num_nodes: usize,
    /// Synchronization state
    sync_state: Arc<Mutex<CoordinationState>>,
    /// Communication interface
    communicator: Option<Arc<super::communication::DistributedCommunicator>>,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub fn new(config: &super::DistributedConfig) -> LinalgResult<Self> {
        let sync_state = Arc::new(Mutex::new(CoordinationState::new(config.num_nodes)));
        
        Ok(Self {
            node_rank: config.node_rank,
            num_nodes: config.num_nodes,
            sync_state,
            communicator: None,
        })
    }
    
    /// Set the communicator for this coordinator
    pub fn set_communicator(&mut self, communicator: Arc<super::communication::DistributedCommunicator>) {
        self.communicator = Some(communicator);
    }
    
    /// Execute a global barrier synchronization
    pub fn barrier(&self) -> LinalgResult<()> {
        self.barrier_with_timeout(Duration::from_secs(30))
    }
    
    /// Execute barrier with custom timeout
    pub fn barrier_with_timeout(&self, timeout: Duration) -> LinalgResult<()> {
        let start_time = Instant::now();
        
        if let Some(ref comm) = self.communicator {
            // Use communicator's barrier if available
            comm.barrier()
        } else {
            // Fallback to local simulation
            self.simulate_barrier(timeout)
        }
    }
    
    /// Create a distributed lock with given name
    pub fn create_distributed_lock(&self, lock_name: &str) -> LinalgResult<DistributedLock> {
        DistributedLock::new(lock_name.to_string(), self.node_rank, self.num_nodes)
    }
    
    /// Execute a distributed consensus operation
    pub fn consensus<T>(&self, proposal: T) -> LinalgResult<T>
    where
        T: Clone + PartialEq + Send + Sync + 'static,
    {
        // Simple consensus: use proposal from rank 0
        if self.node_rank == 0 {
            // Broadcast proposal to all nodes
            // In real implementation, would use communicator
            Ok(proposal)
        } else {
            // Receive consensus result from rank 0
            // In real implementation, would receive from communicator
            Ok(proposal) // Simplified
        }
    }
    
    /// Wait for all nodes to reach a checkpoint
    pub fn checkpoint(&self, checkpoint_id: u64) -> LinalgResult<()> {
        let mut state = self.sync_state.lock().unwrap();
        
        // Mark this node as reached checkpoint
        state.checkpoints.insert(self.node_rank, checkpoint_id);
        
        // Check if all nodes have reached this checkpoint
        if state.checkpoints.len() == self.num_nodes {
            let min_checkpoint = *state.checkpoints.values().min().unwrap_or(&0);
            if min_checkpoint >= checkpoint_id {
                // All nodes have reached at least this checkpoint
                state.checkpoints.clear();
                return Ok(());
            }
        }
        
        // Wait for other nodes (simplified implementation)
        drop(state);
        std::thread::sleep(Duration::from_millis(10));
        self.checkpoint(checkpoint_id)
    }
    
    /// Coordinate a distributed reduction operation
    pub fn coordinate_reduction(&self, operation: ReductionOperation) -> LinalgResult<ReductionCoordination> {
        ReductionCoordination::new(operation, self.node_rank, self.num_nodes)
    }
    
    /// Handle node failure and initiate recovery
    pub fn handle_node_failure(&self, failed_node: usize) -> LinalgResult<RecoveryPlan> {
        let mut state = self.sync_state.lock().unwrap();
        
        // Mark node as failed
        state.failed_nodes.insert(failed_node);
        
        // Create recovery plan
        let remaining_nodes: Vec<usize> = (0..self.num_nodes)
            .filter(|&n| !state.failed_nodes.contains(&n))
            .collect();
        
        let recovery_plan = RecoveryPlan {
            failed_node,
            remaining_nodes: remaining_nodes.clone(),
            redistribution_required: true,
            estimated_recovery_time: Duration::from_secs(30),
        };
        
        // Update active node count
        state.active_nodes = remaining_nodes.len();
        
        Ok(recovery_plan)
    }
    
    /// Get current coordination statistics
    pub fn get_stats(&self) -> CoordinationStats {
        let state = self.sync_state.lock().unwrap();
        CoordinationStats {
            active_nodes: state.active_nodes,
            failed_nodes: state.failed_nodes.len(),
            checkpoint_count: state.checkpoint_count,
            barrier_count: state.barrier_count,
            total_sync_time: state.total_sync_time,
        }
    }
    
    // Private helper methods
    
    fn simulate_barrier(&self, timeout: Duration) -> LinalgResult<()> {
        let start_time = Instant::now();
        let mut state = self.sync_state.lock().unwrap();
        
        state.barrier_participants.insert(self.node_rank);
        state.barrier_count += 1;
        
        // Simulate waiting for all nodes
        while state.barrier_participants.len() < self.num_nodes - state.failed_nodes.len() {
            if start_time.elapsed() > timeout {
                return Err(LinalgError::TimeoutError(
                    "Barrier timeout waiting for nodes".to_string()
                ));
            }
            
            // In a real implementation, we would wait for communication
            drop(state);
            std::thread::sleep(Duration::from_millis(1));
            state = self.sync_state.lock().unwrap();
        }
        
        // Clear barrier state
        state.barrier_participants.clear();
        state.total_sync_time += start_time.elapsed();
        
        Ok(())
    }
}

/// Internal coordination state
#[derive(Debug)]
struct CoordinationState {
    /// Total number of nodes
    total_nodes: usize,
    /// Currently active nodes
    active_nodes: usize,
    /// Failed nodes
    failed_nodes: std::collections::HashSet<usize>,
    /// Checkpoint status per node
    checkpoints: HashMap<usize, u64>,
    /// Barrier participants
    barrier_participants: std::collections::HashSet<usize>,
    /// Statistics
    checkpoint_count: usize,
    barrier_count: usize,
    total_sync_time: Duration,
}

impl CoordinationState {
    fn new(total_nodes: usize) -> Self {
        Self {
            total_nodes,
            active_nodes: total_nodes,
            failed_nodes: std::collections::HashSet::new(),
            checkpoints: HashMap::new(),
            barrier_participants: std::collections::HashSet::new(),
            checkpoint_count: 0,
            barrier_count: 0,
            total_sync_time: Duration::default(),
        }
    }
}

/// Distributed lock implementation
pub struct DistributedLock {
    /// Lock name/identifier
    name: String,
    /// Owner node rank
    owner: Option<usize>,
    /// Current node rank
    node_rank: usize,
    /// Total nodes
    num_nodes: usize,
    /// Lock state
    state: Arc<Mutex<LockState>>,
}

impl DistributedLock {
    /// Create a new distributed lock
    pub fn new(name: String, node_rank: usize, num_nodes: usize) -> LinalgResult<Self> {
        Ok(Self {
            name,
            owner: None,
            node_rank,
            num_nodes,
            state: Arc::new(Mutex::new(LockState::Unlocked)),
        })
    }
    
    /// Acquire the distributed lock
    pub fn acquire(&mut self) -> LinalgResult<()> {
        self.acquire_with_timeout(Duration::from_secs(30))
    }
    
    /// Acquire lock with timeout
    pub fn acquire_with_timeout(&mut self, timeout: Duration) -> LinalgResult<()> {
        let start_time = Instant::now();
        
        loop {
            let mut state = self.state.lock().unwrap();
            
            match *state {
                LockState::Unlocked => {
                    // Try to acquire lock
                    *state = LockState::Locked(self.node_rank);
                    self.owner = Some(self.node_rank);
                    return Ok(());
                }
                LockState::Locked(current_owner) if current_owner == self.node_rank => {
                    // Already own the lock
                    return Ok(());
                }
                LockState::Locked(_) => {
                    // Lock held by another node
                    if start_time.elapsed() > timeout {
                        return Err(LinalgError::TimeoutError(
                            format!("Failed to acquire lock '{}' within timeout", self.name)
                        ));
                    }
                    
                    drop(state);
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
    
    /// Release the distributed lock
    pub fn release(&mut self) -> LinalgResult<()> {
        let mut state = self.state.lock().unwrap();
        
        match *state {
            LockState::Locked(owner) if owner == self.node_rank => {
                *state = LockState::Unlocked;
                self.owner = None;
                Ok(())
            }
            LockState::Locked(_) => {
                Err(LinalgError::InvalidOperation(
                    "Cannot release lock owned by another node".to_string()
                ))
            }
            LockState::Unlocked => {
                // Already unlocked
                Ok(())
            }
        }
    }
    
    /// Check if this node owns the lock
    pub fn is_owned(&self) -> bool {
        self.owner == Some(self.node_rank)
    }
}

/// Lock state
#[derive(Debug, Clone, Copy)]
enum LockState {
    /// Lock is available
    Unlocked,
    /// Lock is held by specified node
    Locked(usize),
}

/// Synchronization barrier implementation
pub struct SynchronizationBarrier {
    /// Number of nodes that must participate
    expected_nodes: usize,
    /// Nodes that have arrived at barrier
    arrived_nodes: Arc<Mutex<std::collections::HashSet<usize>>>,
    /// Condition variable for waiting
    condition: Arc<Condvar>,
    /// Barrier ID for uniqueness
    barrier_id: u64,
}

impl SynchronizationBarrier {
    /// Create a new synchronization barrier
    pub fn new(expected_nodes: usize, barrier_id: u64) -> Self {
        Self {
            expected_nodes,
            arrived_nodes: Arc::new(Mutex::new(std::collections::HashSet::new())),
            condition: Arc::new(Condvar::new()),
            barrier_id,
        }
    }
    
    /// Wait at the barrier
    pub fn wait(&self, node_rank: usize) -> LinalgResult<()> {
        self.wait_timeout(node_rank, Duration::from_secs(60))
    }
    
    /// Wait at barrier with timeout
    pub fn wait_timeout(&self, node_rank: usize, timeout: Duration) -> LinalgResult<()> {
        let mut arrived = self.arrived_nodes.lock().unwrap();
        
        // Add this node to arrived set
        arrived.insert(node_rank);
        
        // Check if all nodes have arrived
        if arrived.len() >= self.expected_nodes {
            // All nodes arrived, notify everyone
            self.condition.notify_all();
            return Ok(());
        }
        
        // Wait for other nodes
        let (_guard, timeout_result) = self.condition
            .wait_timeout(arrived, timeout)
            .unwrap();
        
        if timeout_result.timed_out() {
            Err(LinalgError::TimeoutError(
                format!("Barrier {} timeout waiting for nodes", self.barrier_id)
            ))
        } else {
            Ok(())
        }
    }
    
    /// Reset the barrier for reuse
    pub fn reset(&self) {
        let mut arrived = self.arrived_nodes.lock().unwrap();
        arrived.clear();
    }
}

/// Reduction operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOperation {
    /// Sum all values
    Sum,
    /// Find maximum value
    Max,
    /// Find minimum value
    Min,
    /// Calculate average
    Average,
    /// Logical AND
    And,
    /// Logical OR
    Or,
}

/// Coordination for distributed reduction operations
pub struct ReductionCoordination {
    /// Type of reduction operation
    operation: ReductionOperation,
    /// Current node rank
    node_rank: usize,
    /// Total number of nodes
    num_nodes: usize,
    /// Reduction tree structure
    tree: ReductionTree,
}

impl ReductionCoordination {
    /// Create new reduction coordination
    pub fn new(operation: ReductionOperation, node_rank: usize, num_nodes: usize) -> LinalgResult<Self> {
        let tree = ReductionTree::new(num_nodes);
        
        Ok(Self {
            operation,
            node_rank,
            num_nodes,
            tree,
        })
    }
    
    /// Get nodes this node should receive from
    pub fn get_receive_nodes(&self) -> Vec<usize> {
        self.tree.get_children(self.node_rank)
    }
    
    /// Get node this node should send to
    pub fn get_send_node(&self) -> Option<usize> {
        self.tree.get_parent(self.node_rank)
    }
    
    /// Check if this node is the root of reduction tree
    pub fn is_root(&self) -> bool {
        self.node_rank == 0
    }
}

/// Binary tree structure for reduction operations
struct ReductionTree {
    num_nodes: usize,
}

impl ReductionTree {
    fn new(num_nodes: usize) -> Self {
        Self { num_nodes }
    }
    
    fn get_parent(&self, node: usize) -> Option<usize> {
        if node == 0 {
            None
        } else {
            Some((node - 1) / 2)
        }
    }
    
    fn get_children(&self, node: usize) -> Vec<usize> {
        let mut children = Vec::new();
        
        let left_child = 2 * node + 1;
        let right_child = 2 * node + 2;
        
        if left_child < self.num_nodes {
            children.push(left_child);
        }
        
        if right_child < self.num_nodes {
            children.push(right_child);
        }
        
        children
    }
}

/// Recovery plan for handling node failures
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// The node that failed
    pub failed_node: usize,
    /// Remaining active nodes
    pub remaining_nodes: Vec<usize>,
    /// Whether data redistribution is required
    pub redistribution_required: bool,
    /// Estimated time for recovery
    pub estimated_recovery_time: Duration,
}

impl RecoveryPlan {
    /// Execute the recovery plan
    pub fn execute(&self) -> LinalgResult<()> {
        // In a real implementation, this would:
        // 1. Redistribute data from failed node
        // 2. Update communication topologies
        // 3. Restart failed computations
        // 4. Update load balancing
        
        Ok(())
    }
    
    /// Get new node mapping after failure
    pub fn get_node_mapping(&self) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();
        
        // Map remaining nodes to new ranks
        for (new_rank, &old_rank) in self.remaining_nodes.iter().enumerate() {
            mapping.insert(old_rank, new_rank);
        }
        
        mapping
    }
}

/// Statistics for coordination operations
#[derive(Debug, Clone, Default)]
pub struct CoordinationStats {
    /// Number of currently active nodes
    pub active_nodes: usize,
    /// Number of failed nodes
    pub failed_nodes: usize,
    /// Total number of checkpoints
    pub checkpoint_count: usize,
    /// Total number of barriers
    pub barrier_count: usize,
    /// Total time spent in synchronization
    pub total_sync_time: Duration,
}

impl CoordinationStats {
    /// Calculate average synchronization time per operation
    pub fn avg_sync_time(&self) -> Duration {
        if self.barrier_count + self.checkpoint_count > 0 {
            self.total_sync_time / (self.barrier_count + self.checkpoint_count) as u32
        } else {
            Duration::default()
        }
    }
    
    /// Calculate node availability ratio
    pub fn availability_ratio(&self) -> f64 {
        let total_nodes = self.active_nodes + self.failed_nodes;
        if total_nodes > 0 {
            self.active_nodes as f64 / total_nodes as f64
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distributed_coordinator() {
        use super::super::DistributedConfig;
        
        let config = DistributedConfig::default().with_num_nodes(3).with_node_rank(0);
        let coordinator = DistributedCoordinator::new(&config).unwrap();
        
        let stats = coordinator.get_stats();
        assert_eq!(stats.active_nodes, 3);
        assert_eq!(stats.failed_nodes, 0);
    }
    
    #[test]
    fn test_distributed_lock() {
        let mut lock = DistributedLock::new("test_lock".to_string(), 0, 2).unwrap();
        
        // Should be able to acquire lock
        assert!(lock.acquire().is_ok());
        assert!(lock.is_owned());
        
        // Should be able to release lock
        assert!(lock.release().is_ok());
        assert!(!lock.is_owned());
    }
    
    #[test]
    fn test_synchronization_barrier() {
        let barrier = SynchronizationBarrier::new(2, 1);
        
        // First node arrives
        let start = Instant::now();
        let result = barrier.wait_timeout(0, Duration::from_millis(100));
        
        // Should timeout since second node didn't arrive
        assert!(result.is_err());
        assert!(start.elapsed() >= Duration::from_millis(90));
    }
    
    #[test]
    fn test_reduction_coordination() {
        let reduction = ReductionCoordination::new(ReductionOperation::Sum, 0, 4).unwrap();
        
        assert!(reduction.is_root());
        assert!(reduction.get_send_node().is_none());
        assert_eq!(reduction.get_receive_nodes(), vec![1, 2]);
    }
    
    #[test]
    fn test_recovery_plan() {
        let plan = RecoveryPlan {
            failed_node: 2,
            remaining_nodes: vec![0, 1, 3],
            redistribution_required: true,
            estimated_recovery_time: Duration::from_secs(30),
        };
        
        let mapping = plan.get_node_mapping();
        assert_eq!(mapping.len(), 3);
        assert_eq!(mapping[&0], 0);
        assert_eq!(mapping[&1], 1);
        assert_eq!(mapping[&3], 2);
    }
    
    #[test]
    fn test_reduction_tree() {
        let tree = ReductionTree::new(7);
        
        // Root node (0)
        assert_eq!(tree.get_parent(0), None);
        assert_eq!(tree.get_children(0), vec![1, 2]);
        
        // Internal node (1)
        assert_eq!(tree.get_parent(1), Some(0));
        assert_eq!(tree.get_children(1), vec![3, 4]);
        
        // Leaf node (6)
        assert_eq!(tree.get_parent(6), Some(2));
        assert_eq!(tree.get_children(6), vec![]);
    }
}