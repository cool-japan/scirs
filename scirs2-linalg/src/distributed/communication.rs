//! Communication primitives for distributed linear algebra
//!
//! This module provides efficient communication primitives for distributed
//! linear algebra operations, including point-to-point communication,
//! collective operations, and optimized data transfer protocols.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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

/// TCP message header for reliable communication
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TcpMessageHeader {
    /// Source node rank
    source: usize,
    /// Destination node rank
    destination: usize,
    /// Message tag
    tag: u32,
    /// Data size in bytes
    datasize: usize,
    /// Sequence number
    sequence: u64,
    /// Timestamp
    timestamp: u64,
    /// Data checksum for integrity
    checksum: u64,
}

/// Connection pool for TCP connections
struct TcpConnectionPool {
    connections: HashMap<usize, std::net::TcpStream>,
    listener: Option<std::net::TcpListener>,
}

impl TcpConnectionPool {
    fn new() -> Self {
        Self {
            connections: HashMap::new(),
            listener: None,
        }
    }
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
    /// TCP connection pool (for TCP backend)
    tcp_pool: Arc<Mutex<TcpConnectionPool>>,
    /// Node address mapping (rank -> address)
    node_addresses: HashMap<usize, String>,
}

impl DistributedCommunicator {
    /// Create a new distributed communicator
    pub fn new(config: &super::DistributedConfig) -> LinalgResult<Self> {
        // Create node address mapping (simplified - in practice would use discovery service)
        let mut node_addresses = HashMap::new();
        for rank in 0.._config.num_nodes {
            let port = 7000 + rank; // Base port + rank
            node_addresses.insert(rank, format!("127.0.0.1:{}", port));
        }

        Ok(Self {
            rank: config.node_rank,
            size: config.num_nodes,
            backend: config.backend,
            sequence_counter: Arc::new(Mutex::new(0)),
            message_buffer: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CommunicationStats::default())),
            tcp_pool: Arc::new(Mutex::new(TcpConnectionPool::new())),
            node_addresses,
        })
    }
    
    /// Send a matrix to another node
    pub fn sendmatrix<T>(&self, matrix: &ArrayView2<T>, dest: usize, tag: MessageTag) -> LinalgResult<()>
    where
        T: Clone + Send + Sync + Serialize,
    {
        let start_time = Instant::now();
        
        // Serialize matrix data
        let serialized = self.serializematrix(matrix)?;
        
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
    pub fn recvmatrix<T>(&self, source: usize, tag: MessageTag) -> LinalgResult<Array2<T>>
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
        let matrix = self.deserializematrix(&message.data)?;
        
        // Update statistics
        let elapsed = start_time.elapsed();
        self.update_recv_stats(message.data.len(), elapsed);
        
        Ok(matrix)
    }
    
    /// Broadcast a matrix to all nodes
    pub fn broadcastmatrix<T>(&self, matrix: &ArrayView2<T>) -> LinalgResult<()>
    where
        T: Clone + Send + Sync + Serialize,
    {
        if self.rank == 0 {
            // Root node sends to all others
            for dest in 1..self.size {
                self.sendmatrix(matrix, dest, MessageTag::Data)?;
            }
        }
        Ok(())
    }
    
    /// Gather matrices from all nodes to root
    pub fn gather_matrices<T>(&self, localmatrix: &ArrayView2<T>) -> LinalgResult<Option<Vec<Array2<T>>>>
    where
        T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    {
        if self.rank == 0 {
            // Root node collects from all others
            let mut matrices = Vec::with_capacity(self.size);
            
            // Add local matrix
            matrices.push(localmatrix.to_owned());
            
            // Receive from other nodes
            for source in 1..self.size {
                let matrix = self.recvmatrix(source, MessageTag::Data)?;
                matrices.push(matrix);
            }
            
            Ok(Some(matrices))
        } else {
            // Non-root nodes send their data
            self.sendmatrix(localmatrix, 0, MessageTag::Data)?;
            Ok(None)
        }
    }
    
    /// All-reduce operation (sum matrices across all nodes)
    pub fn allreduce_sum<T>(&self, localmatrix: &ArrayView2<T>) -> LinalgResult<Array2<T>>
    where
        T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + scirs2_core::numeric::Zero + std::ops::Add<Output = T>,
    {
        // Gather all matrices to root
        if let Some(matrices) = self.gather_matrices(localmatrix)? {
            // Sum all matrices (only on root)
            let mut result = matrices[0].clone();
            for matrix in matrices.iter().skip(1) {
                result = &result + matrix;
            }
            
            // Broadcast result to all nodes
            self.broadcastmatrix(&result.view())?;
            Ok(result)
        } else {
            // Non-root nodes receive the result
            self.recvmatrix(0, MessageTag::Data)
        }
    }
    
    /// Scatter operation (distribute parts of a matrix to all nodes)
    pub fn scattermatrix<T>(&self, matrix: Option<&ArrayView2<T>>) -> LinalgResult<Array2<T>>
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
                
                let chunk = matrix.slice(scirs2_core::ndarray::s![start_row..end_row, ..]);
                self.sendmatrix(&chunk, dest, MessageTag::Data)?;
                
                start_row = end_row;
            }
            
            // Return root's portion
            let root_rows = if 0 < remainder { rows_per_node + 1 } else { rows_per_node };
            Ok(matrix.slice(scirs2_core::ndarray::s![..root_rows, ..]).to_owned())
        } else {
            // Receive chunk from root
            self.recvmatrix(0, MessageTag::Data)
        }
    }
    
    /// Barrier synchronization
    pub fn barrier(&self) -> LinalgResult<()> {
        // Simple barrier implementation - in practice would use more efficient algorithms
        if self.rank == 0 {
            // Root waits for all nodes to arrive
            for source in 1..self.size {
                let _: Array2<f64> = self.recvmatrix(source, MessageTag::Sync)?;
            }
            
            // Send release signal to all nodes
            let dummy = Array2::<f64>::zeros((1, 1));
            for dest in 1..self.size {
                self.sendmatrix(&dummy.view(), dest, MessageTag::Sync)?;
            }
        } else {
            // Non-root nodes signal arrival and wait for release
            let dummy = Array2::<f64>::zeros((1, 1));
            self.sendmatrix(&dummy.view(), 0, MessageTag::Sync)?;
            let _: Array2<f64> = self.recvmatrix(0, MessageTag::Sync)?;
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
            }_ => {
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
    
    fn serializematrix<T>(&self, matrix: &ArrayView2<T>) -> LinalgResult<Vec<u8>>
    where
        T: Serialize,
    {
        let cfg = bincode::config::standard();
        bincode::serde::encode_to_vec(matrix, cfg).map_err(|e| {
            LinalgError::SerializationError(format!("Failed to serialize matrix (bincode2): {}", e))
        })
    }
    
    fn deserializematrix<T>(&self, data: &[u8]) -> LinalgResult<Array2<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let cfg = bincode::config::standard();
        bincode::serde::decode_from_slice(data, cfg)
            .map(|(m, _len)| m)
            .map_err(|e| {
                LinalgError::SerializationError(format!("Failed to deserialize matrix (bincode2): {}", e))
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
    
    fn send_tcp(&self, message: Message<Vec<u8>>) -> LinalgResult<()> {
        use std::net::TcpStream;
        use std::io::Write;
        
        // Get connection info for destination node
        let dest_address = self.get_node_address(message.metadata.destination)?;
        
        // Establish TCP connection
        let mut stream = TcpStream::connect(&dest_address)
            .map_err(|e| LinalgError::CommunicationError(
                format!("Failed to connect to {}: {}", dest_address, e)
            ))?;
        
        // Send message header first
        let header = TcpMessageHeader {
            source: message.metadata.source,
            destination: message.metadata.destination,
            tag: message.metadata.tag as u32,
            datasize: message.data.len(),
            sequence: message.metadata.sequence,
            timestamp: message.metadata.timestamp,
            checksum: self.calculate_checksum(&message.data),
        };
        
        let cfg = bincode::config::standard();
        let header_bytes = bincode::serde::encode_to_vec(&header, cfg)
            .map_err(|e| LinalgError::SerializationError(format!("Header serialization failed (bincode2): {}", e)))?;
        
        // Send header size first (4 bytes)
        let headersize = header_bytes.len() as u32;
        stream.write_all(&headersize.to_be_bytes())
            .map_err(|e| LinalgError::CommunicationError(format!("Failed to send header size: {}", e)))?;
        
        // Send header
        stream.write_all(&header_bytes)
            .map_err(|e| LinalgError::CommunicationError(format!("Failed to send header: {}", e)))?;
        
        // Send data in chunks for large messages
        const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks
        for chunk in message.data.chunks(CHUNK_SIZE) {
            stream.write_all(chunk)
                .map_err(|e| LinalgError::CommunicationError(format!("Failed to send data chunk: {}", e)))?;
        }
        
        // Ensure all data is sent
        stream.flush()
            .map_err(|e| LinalgError::CommunicationError(format!("Failed to flush TCP stream: {}", e)))?;
        
        Ok(())
    }
    
    fn recv_tcp(&self, source: usize, tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
        use std::net::{TcpListener, TcpStream};
        use std::io::Read;
        
        // Listen for incoming connections on our port
        let listen_address = self.get_node_address(self.rank)?;
        let listener = TcpListener::bind(&listen_address)
            .map_err(|e| LinalgError::CommunicationError(
                format!("Failed to bind to {}: {}", listen_address, e)
            ))?;
        
        // Accept connection (in practice, would have connection pooling)
        let (mut stream, remote_addr) = listener.accept()
            .map_err(|e| LinalgError::CommunicationError(format!("Failed to accept connection: {}", e)))?;
        
        // Read header size
        let mut headersize_bytes = [0u8; 4];
        stream.read_exact(&mut headersize_bytes)
            .map_err(|e| LinalgError::CommunicationError(format!("Failed to read header size: {}", e)))?;
        let headersize = u32::from_be_bytes(headersize_bytes) as usize;
        
        // Read header
        let mut header_bytes = vec![0u8; headersize];
        stream.read_exact(&mut header_bytes)
            .map_err(|e| LinalgError::CommunicationError(format!("Failed to read header: {}", e)))?;
        
        let cfg = bincode::config::standard();
        let (header, _len): (TcpMessageHeader, usize) = bincode::serde::decode_from_slice(&header_bytes, cfg)
            .map_err(|e| LinalgError::SerializationError(format!("Header deserialization failed (bincode2): {}", e)))?;
        
        // Validate header
        if header.source != source {
            return Err(LinalgError::CommunicationError(format!(
                "Expected message from node {}, got from node {}",
                source, header.source
            )));
        }
        
        if header.tag != tag as u32 {
            return Err(LinalgError::CommunicationError(format!(
                "Expected message with tag {:?}, got tag {}",
                tag, header.tag
            )));
        }
        
        // Read data
        let mut data = vec![0u8; header.datasize];
        let mut bytes_read = 0;
        while bytes_read < header.datasize {
            let chunksize = std::cmp::min(header.datasize - bytes_read, 64 * 1024);
            let mut chunk = vec![0u8; chunksize];
            stream.read_exact(&mut chunk)
                .map_err(|e| LinalgError::CommunicationError(format!("Failed to read data chunk: {}", e)))?;
            
            data[bytes_read..bytes_read + chunksize].copy_from_slice(&chunk);
            bytes_read += chunksize;
        }
        
        // Verify checksum
        let received_checksum = self.calculate_checksum(&data);
        if received_checksum != header.checksum {
            return Err(LinalgError::CommunicationError(format!(
                "Checksum mismatch: expected {}, got {}",
                header.checksum, received_checksum
            )));
        }
        
        // Create message
        let metadata = MessageMetadata {
            source: header.source,
            destination: header.destination,
            tag,
            size_bytes: data.len(),
            sequence: header.sequence,
            timestamp: header.timestamp,
            compressed: false,
        };
        
        Ok(Message { metadata, data })
    }
    
    fn send_mpi(&selfmessage: Message<Vec<u8>>) -> LinalgResult<()> {
        // MPI implementation would go here
        Err(LinalgError::NotImplemented("MPI backend not implemented".to_string()))
    }
    
    fn recv_mpi(&self_source: usize, tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
        // MPI implementation would go here
        Err(LinalgError::NotImplemented("MPI backend not implemented".to_string()))
    }
    
    fn send_rdma(&selfmessage: Message<Vec<u8>>) -> LinalgResult<()> {
        // RDMA implementation would go here
        Err(LinalgError::NotImplemented("RDMA backend not implemented".to_string()))
    }
    
    fn recv_rdma(&self_source: usize, tag: MessageTag) -> LinalgResult<Message<Vec<u8>>> {
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
    
    /// Get node address for TCP communication
    fn get_node_address(&self, rank: usize) -> LinalgResult<String> {
        self.node_addresses.get(&rank)
            .cloned()
            .ok_or_else(|| LinalgError::CommunicationError(
                format!("No address found for node rank {}", rank)
            ))
    }
    
    /// Calculate checksum for data integrity
    fn calculate_checksum(&self, data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Compress data using LZ4 algorithm (mock implementation)
    fn compress_data(&self, data: &[u8]) -> LinalgResult<Vec<u8>> {
        // In a real implementation, this would use lz4 compression
        // For now, just return the original data
        Ok(data.to_vec())
    }
    
    /// Decompress data (mock implementation)
    fn decompress_data(&self, compresseddata: &[u8]) -> LinalgResult<Vec<u8>> {
        // In a real implementation, this would decompress LZ4 _data
        // For now, just return the original _data
        Ok(compressed_data.to_vec())
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
    use scirs2_core::ndarray::Array2;
    
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
        let serialized = comm.serializematrix(&matrix.view()).unwrap();
        let deserialized: Array2<f64> = comm.deserializematrix(&serialized).unwrap();
        
        assert_eq!(matrix, deserialized);
    }
}
