//! Communication primitives for distributed linear algebra
//!
//! This module provides efficient communication primitives for distributed
//! linear algebra operations, including point-to-point communication,
//! collective operations, and optimized data transfer protocols.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Communication backends for distributed operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationBackend {
    /// In-memory communication (for testing and single-node)
    InMemory,
    /// TCP/IP based communication
    TCP,
    /// MPI (Message Passing Interface)
    MPI,
    /// RDMA (Remote Direct Memory Access)
    RDMA,
    /// Custom communication backend
    Custom,
}

/// Message tags for different operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageTag {
    /// Data transfer
    Data = 0,
    /// Control message
    Control = 1,
    /// Synchronization
    Sync = 2,
    /// Error/status
    Status = 3,
    /// Matrix multiplication data
    MatMul = 4,
    /// Matrix decomposition data
    Decomp = 5,
    /// Solver data
    Solve = 6,
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Source node rank
    pub source: usize,
    /// Destination node rank  
    pub destination: usize,
    /// Message tag
    pub tag: MessageTag,
    /// Data size in bytes
    pub size_bytes: usize,
    /// Sequence number
    pub sequence: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Compression used
    pub compressed: bool,
}

/// Communication message
#[derive(Debug, Clone)]
pub struct Message<T> {
    /// Message metadata
    pub metadata: MessageMetadata,
    /// Message payload
    pub data: T,
}

impl<T> Message<T> {
    /// Create a new message
    pub fn new(
        source: usize,
        destination: usize,
        tag: MessageTag,
        data: T,
        sequence: u64,
    ) -> Self {
        let metadata = MessageMetadata {
            source,
            destination,
            tag,
            size_bytes: std::mem::size_of::<T>(),
            sequence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            compressed: false,
        };
        
        Self { metadata, data }
    }
}

/// Distributed communicator for linear algebra operations
pub struct DistributedCommunicator {
    /// Node rank
    pub rank: usize,
    /// Total number of nodes
    pub size: usize,
    /// Communication backend
    backend: CommunicationBackend,
    /// Message sequence counter
    sequence_counter: Arc<Mutex<u64>>,
    /// Pending messages
    message_buffer: Arc<Mutex<HashMap<(usize, MessageTag), Vec<u8>>>>,
    /// Communication statistics
    stats: Arc<Mutex<CommunicationStats>>,
}

impl DistributedCommunicator {
    /// Create a new distributed communicator
    pub fn new(config: &super::DistributedConfig) -> LinalgResult<Self> {
        Ok(Self {
            rank: config.node_rank,
            size: config.num_nodes,
            backend: config.backend,
            sequence_counter: Arc::new(Mutex::new(0)),
            message_buffer: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CommunicationStats::default())),
        })
    }
    
    /// Send a matrix to another node
    pub fn send_matrix<T>(&self, matrix: &ArrayView2<T>, dest: usize, tag: MessageTag) -> LinalgResult<()>
    where
        T: Clone + Send + Sync + Serialize,
    {
        let start_time = Instant::now();
        
        // Serialize matrix data
        let serialized = self.serialize_matrix(matrix)?;
        
        // Create message
        let sequence = self.next_sequence();
        let message = Message::new(self.rank, dest, tag, serialized, sequence);
        
        // Send based on backend
        match self.backend {
            CommunicationBackend::InMemory => {
                self.send_in_memory(message)?;
            },
            CommunicationBackend::TCP => {
                self.send_tcp(message)?;
            },
            CommunicationBackend::MPI => {
                self.send_mpi(message)?;
            },
            CommunicationBackend::RDMA => {
                self.send_rdma(message)?;
            },
            CommunicationBackend::Custom => {
                return Err(LinalgError::NotImplemented("Custom backend not implemented".to_string()));
            },
        }
        
        // Update statistics
        let elapsed = start_time.elapsed();
        self.update_send_stats(serialized.len(), elapsed);
        
        Ok(())
    }
    
    /// Receive a matrix from another node
    pub fn recv_matrix<T>(&self, source: usize, tag: MessageTag) -> LinalgResult<Array2<T>>
    where
        T: Clone + Send + Sync + for<'de> Deserialize<'de>,
    {
        let start_time = Instant::now();
        
        // Receive based on backend
        let message = match self.backend {
            CommunicationBackend::InMemory => {
                self.recv_in_memory(source, tag)?
            },
            CommunicationBackend::TCP => {
                self.recv_tcp(source, tag)?
            },
            CommunicationBackend::MPI => {
                self.recv_mpi(source, tag)?
            },
            CommunicationBackend::RDMA => {
                self.recv_rdma(source, tag)?
            },
            CommunicationBackend::Custom => {
                return Err(LinalgError::NotImplemented("Custom backend not implemented".to_string()));
            },
        };
        
        // Deserialize matrix
        let matrix = self.deserialize_matrix(&message.data)?;
        
        // Update statistics
        let elapsed = start_time.elapsed();
        self.update_recv_stats(message.data.len(), elapsed);
        
        Ok(matrix)
    }
    
    /// Broadcast a matrix to all nodes
    pub fn broadcast_matrix<T>(&self, matrix: &ArrayView2<T>) -> LinalgResult<()>
    where
        T: Clone + Send + Sync + Serialize,
    {
        if self.rank == 0 {
            // Root node sends to all others
            for dest in 1..self.size {
                self.send_matrix(matrix, dest, MessageTag::Data)?;
            }
        }
        Ok(())
    }
    
    /// Gather matrices from all nodes to root
    pub fn gather_matrices<T>(&self, local_matrix: &ArrayView2<T>) -> LinalgResult<Option<Vec<Array2<T>>>>
    where
        T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    {
        if self.rank == 0 {
            // Root node collects from all others
            let mut matrices = Vec::with_capacity(self.size);
            
            // Add local matrix
            matrices.push(local_matrix.to_owned());
            
            // Receive from other nodes
            for source in 1..self.size {
                let matrix = self.recv_matrix(source, MessageTag::Data)?;
                matrices.push(matrix);
            }
            
            Ok(Some(matrices))
        } else {
            // Non-root nodes send their data
            self.send_matrix(local_matrix, 0, MessageTag::Data)?;
            Ok(None)
        }
    }
    
    /// All-reduce operation (sum matrices across all nodes)
    pub fn allreduce_sum<T>(&self, local_matrix: &ArrayView2<T>) -> LinalgResult<Array2<T>>
    where
        T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + num_traits::Zero + std::ops::Add<Output = T>,
    {
        // Gather all matrices to root
        if let Some(matrices) = self.gather_matrices(local_matrix)? {
            // Sum all matrices (only on root)
            let mut result = matrices[0].clone();
            for matrix in matrices.iter().skip(1) {
                result = &result + matrix;
            }
            
            // Broadcast result to all nodes
            self.broadcast_matrix(&result.view())?;
            Ok(result)
        } else {
            // Non-root nodes receive the result
            self.recv_matrix(0, MessageTag::Data)
        }
    }
    
    /// Scatter operation (distribute parts of a matrix to all nodes)
    pub fn scatter_matrix<T>(&self, matrix: Option<&ArrayView2<T>>) -> LinalgResult<Array2<T>>
    where
        T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    {
        if self.rank == 0 {
            let matrix = matrix.ok_or_else(|| {
                LinalgError::InvalidInput("Root node must provide matrix for scatter".to_string())
            })?;
            
            let (rows, cols) = matrix.dim();
            let rows_per_node = rows / self.size;
            let remainder = rows % self.size;
            
            // Send chunks to other nodes
            let mut start_row = 0;
            for dest in 1..self.size {
                let chunk_rows = if dest <= remainder { rows_per_node + 1 } else { rows_per_node };
                let end_row = start_row + chunk_rows;
                
                let chunk = matrix.slice(ndarray::s![start_row..end_row, ..]);
                self.send_matrix(&chunk, dest, MessageTag::Data)?;
                
                start_row = end_row;
            }
            
            // Return root's portion
            let root_rows = if 0 < remainder { rows_per_node + 1 } else { rows_per_node };
            Ok(matrix.slice(ndarray::s![..root_rows, ..]).to_owned())
        } else {
            // Receive chunk from root
            self.recv_matrix(0, MessageTag::Data)
        }
    }
    
    /// Barrier synchronization
    pub fn barrier(&self) -> LinalgResult<()> {
        // Simple barrier implementation - in practice would use more efficient algorithms
        if self.rank == 0 {
            // Root waits for all nodes to arrive
            for source in 1..self.size {
                let _: Array2<f64> = self.recv_matrix(source, MessageTag::Sync)?;
            }
            
            // Send release signal to all nodes
            let dummy = Array2::<f64>::zeros((1, 1));
            for dest in 1..self.size {
                self.send_matrix(&dummy.view(), dest, MessageTag::Sync)?;
            }
        } else {
            // Non-root nodes signal arrival and wait for release
            let dummy = Array2::<f64>::zeros((1, 1));
            self.send_matrix(&dummy.view(), 0, MessageTag::Sync)?;
            let _: Array2<f64> = self.recv_matrix(0, MessageTag::Sync)?;
        }
        
        Ok(())
    }
    
    /// Finalize communication
    pub fn finalize(&self) -> LinalgResult<()> {
        // Cleanup based on backend
        match self.backend {
            CommunicationBackend::InMemory => {
                // Clear message buffer
                self.message_buffer.lock().unwrap().clear();
            },
            CommunicationBackend::MPI => {
                // MPI finalization would go here
            },
            _ => {
                // Other backends cleanup
            },
        }
        
        Ok(())
    }
    
    /// Get communication statistics
    pub fn get_stats(&self) -> CommunicationStats {
        self.stats.lock().unwrap().clone()
    }
    
    // Private helper methods
    
    fn next_sequence(&self) -> u64 {
        let mut counter = self.sequence_counter.lock().unwrap();
        *counter += 1;
        *counter
    }
    
    fn serialize_matrix<T>(&self, matrix: &ArrayView2<T>) -> LinalgResult<Vec<u8>>
    where
        T: Serialize,
    {
        bincode::serialize(matrix).map_err(|e| {
            LinalgError::SerializationError(format!("Failed to serialize matrix: {}", e))
        })
    }
    
    fn deserialize_matrix<T>(&self, data: &[u8]) -> LinalgResult<Array2<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        bincode::deserialize(data).map_err(|e| {
            LinalgError::SerializationError(format!("Failed to deserialize matrix: {}", e))
        })
    }
    
    fn send_in_memory(&self, message: Message<Vec<u8>>) -> LinalgResult<()> {
        let mut buffer = self.message_buffer.lock().unwrap();
        let key = (message.metadata.source, message.metadata.tag);
        buffer.insert(key, message.data);
        Ok(())
    }
    
    fn recv_in_memory(&self, source: usize, tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
        let mut buffer = self.message_buffer.lock().unwrap();
        let key = (source, tag);
        
        if let Some(data) = buffer.remove(&key) {
            let metadata = MessageMetadata {
                source,
                destination: self.rank,
                tag,
                size_bytes: data.len(),
                sequence: 0,
                timestamp: 0,
                compressed: false,
            };
            
            Ok(Message { metadata, data })
        } else {
            Err(LinalgError::CommunicationError(format!(
                "No message available from {} with tag {:?}",
                source, tag
            )))
        }
    }
    
    fn send_tcp(&self, _message: Message<Vec<u8>>) -> LinalgResult<()> {
        // TCP implementation would go here
        Err(LinalgError::NotImplemented("TCP backend not implemented".to_string()))
    }
    
    fn recv_tcp(&self, _source: usize, _tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
        // TCP implementation would go here
        Err(LinalgError::NotImplemented("TCP backend not implemented".to_string()))
    }
    
    fn send_mpi(&self, _message: Message<Vec<u8>>) -> LinalgResult<()> {
        // MPI implementation would go here
        Err(LinalgError::NotImplemented("MPI backend not implemented".to_string()))
    }
    
    fn recv_mpi(&self, _source: usize, _tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
        // MPI implementation would go here
        Err(LinalgError::NotImplemented("MPI backend not implemented".to_string()))
    }
    
    fn send_rdma(&self, _message: Message<Vec<u8>>) -> LinalgResult<()> {
        // RDMA implementation would go here
        Err(LinalgError::NotImplemented("RDMA backend not implemented".to_string()))
    }
    
    fn recv_rdma(&self, _source: usize, _tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
        // RDMA implementation would go here
        Err(LinalgError::NotImplemented("RDMA backend not implemented".to_string()))
    }
    
    fn update_send_stats(&self, bytes: usize, duration: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.messages_sent += 1;
        stats.bytes_sent += bytes;
        stats.total_send_time += duration;
    }
    
    fn update_recv_stats(&self, bytes: usize, duration: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.messages_received += 1;
        stats.bytes_received += bytes;
        stats.total_recv_time += duration;
    }
}

/// Communication statistics
#[derive(Debug, Clone, Default)]
pub struct CommunicationStats {
    /// Number of messages sent
    pub messages_sent: usize,
    /// Number of messages received
    pub messages_received: usize,
    /// Total bytes sent
    pub bytes_sent: usize,
    /// Total bytes received
    pub bytes_received: usize,
    /// Total time spent sending
    pub total_send_time: Duration,
    /// Total time spent receiving
    pub total_recv_time: Duration,
}

impl CommunicationStats {
    /// Calculate average send bandwidth (bytes/second)
    pub fn avg_send_bandwidth(&self) -> f64 {
        if self.total_send_time.as_secs_f64() > 0.0 {
            self.bytes_sent as f64 / self.total_send_time.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Calculate average receive bandwidth (bytes/second)
    pub fn avg_recv_bandwidth(&self) -> f64 {
        if self.total_recv_time.as_secs_f64() > 0.0 {
            self.bytes_received as f64 / self.total_recv_time.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Calculate communication efficiency ratio
    pub fn efficiency_ratio(&self) -> f64 {
        if self.messages_sent + self.messages_received > 0 {
            let total_messages = self.messages_sent + self.messages_received;
            let avg_time = (self.total_send_time + self.total_recv_time).as_secs_f64() / total_messages as f64;
            1.0 / (1.0 + avg_time)  // Higher efficiency = lower average time per message
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_message_creation() {
        let data = vec![1, 2, 3, 4];
        let message = Message::new(0, 1, MessageTag::Data, data.clone(), 1);
        
        assert_eq!(message.metadata.source, 0);
        assert_eq!(message.metadata.destination, 1);
        assert_eq!(message.metadata.tag, MessageTag::Data);
        assert_eq!(message.data, data);
    }
    
    #[test]
    fn test_communication_stats() {
        let mut stats = CommunicationStats::default();
        
        stats.messages_sent = 10;
        stats.bytes_sent = 1024;
        stats.total_send_time = Duration::from_millis(100);
        
        let bandwidth = stats.avg_send_bandwidth();
        assert!(bandwidth > 0.0);
        
        let efficiency = stats.efficiency_ratio();
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }
    
    #[test]
    fn test_in_memory_communication() {
        use super::super::DistributedConfig;
        
        let config = DistributedConfig::default()
            .with_backend(CommunicationBackend::InMemory);
        let comm = DistributedCommunicator::new(&config).unwrap();
        
        // Test serialization
        let matrix = Array2::from_shape_fn((3, 3), |(i, j)| (i + j) as f64);
        let serialized = comm.serialize_matrix(&matrix.view()).unwrap();
        let deserialized: Array2<f64> = comm.deserialize_matrix(&serialized).unwrap();
        
        assert_eq!(matrix, deserialized);
    }
}