//! Distributed matrix and vector implementations
//!
//! This module provides distributed matrix and vector types that can
//! span multiple compute nodes, with automatic data partitioning,
//! load balancing, and distributed operations.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, Zero};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use super::communication::{DistributedCommunicator, MessageTag};
use super::coordination::DistributedCoordinator;
use super::distribution::{DataDistribution, DistributionStrategy, MatrixPartitioner};

/// Distributed matrix that spans multiple compute nodes
pub struct DistributedMatrix<T> {
    /// Local partition of the matrix
    local_data: Array2<T>,
    /// Distribution information
    distribution: DataDistribution,
    /// Communicator for inter-node communication
    communicator: Arc<DistributedCommunicator>,
    /// Coordinator for synchronization
    coordinator: Arc<DistributedCoordinator>,
    /// Current node rank
    node_rank: usize,
    /// Configuration
    config: super::DistributedConfig,
}

impl<T> DistributedMatrix<T>
where
    T: Float + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Create a distributed matrix from a local matrix
    pub fn from_local(
        local_matrix: Array2<T>,
        config: super::DistributedConfig,
    ) -> LinalgResult<Self> {
        let global_shape = local_matrix.dim();
        
        // Create distribution
        let distribution = match config.distribution {
            DistributionStrategy::RowWise => {
                DataDistribution::row_wise(global_shape, config.num_nodes, config.node_rank)?
            }
            DistributionStrategy::ColumnWise => {
                DataDistribution::column_wise(global_shape, config.num_nodes, config.node_rank)?
            }
            DistributionStrategy::BlockCyclic => {
                DataDistribution::block_cyclic(
                    global_shape,
                    config.num_nodes,
                    config.node_rank,
                    (config.block_size, config.block_size),
                )?
            }
            _ => {
                return Err(LinalgError::NotImplemented(
                    "Distribution strategy not implemented".to_string()
                ));
            }
        };
        
        // Extract local partition
        let local_data = MatrixPartitioner::partition(&local_matrix.view(), &distribution)?;
        
        // Create communicator and coordinator
        let communicator = Arc::new(DistributedCommunicator::new(&config)?);
        let coordinator = Arc::new(DistributedCoordinator::new(&config)?);
        
        Ok(Self {
            local_data,
            distribution,
            communicator,
            coordinator,
            node_rank: config.node_rank,
            config,
        })
    }
    
    /// Create a distributed matrix with specific distribution
    pub fn from_distribution(
        distribution: DataDistribution,
        config: super::DistributedConfig,
    ) -> LinalgResult<Self> {
        // Create zero matrix with local shape
        let local_data = Array2::zeros(distribution.local_shape);
        
        let communicator = Arc::new(DistributedCommunicator::new(&config)?);
        let coordinator = Arc::new(DistributedCoordinator::new(&config)?);
        
        Ok(Self {
            local_data,
            distribution,
            communicator,
            coordinator,
            node_rank: config.node_rank,
            config,
        })
    }
    
    /// Get global shape of the distributed matrix
    pub fn global_shape(&self) -> (usize, usize) {
        self.distribution.global_shape
    }
    
    /// Get local shape of this node's partition
    pub fn local_shape(&self) -> (usize, usize) {
        self.local_data.dim()
    }
    
    /// Get reference to local data
    pub fn local_data(&self) -> &Array2<T> {
        &self.local_data
    }
    
    /// Get mutable reference to local data
    pub fn local_data_mut(&mut self) -> &mut Array2<T> {
        &mut self.local_data
    }
    
    /// Distributed matrix multiplication: C = self * other
    pub fn multiply(&self, other: &DistributedMatrix<T>) -> LinalgResult<DistributedMatrix<T>> {
        let (m, k) = self.global_shape();
        let (k2, n) = other.global_shape();
        
        if k != k2 {
            return Err(LinalgError::DimensionError(format!(
                "Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})",
                m, k, k2, n
            )));
        }
        
        match (&self.distribution.strategy, &other.distribution.strategy) {
            (DistributionStrategy::RowWise, DistributionStrategy::ColumnWise) => {
                self.multiply_row_col(other)
            }
            (DistributionStrategy::RowWise, DistributionStrategy::RowWise) => {
                self.multiply_row_row(other)
            }
            (DistributionStrategy::ColumnWise, DistributionStrategy::ColumnWise) => {
                self.multiply_col_col(other)
            }
            _ => Err(LinalgError::NotImplemented(
                "Matrix multiplication for this distribution combination not implemented".to_string()
            )),
        }
    }
    
    /// Distributed matrix addition: C = self + other
    pub fn add(&self, other: &DistributedMatrix<T>) -> LinalgResult<DistributedMatrix<T>> {
        if self.global_shape() != other.global_shape() {
            return Err(LinalgError::DimensionError(
                "Matrix dimensions must match for addition".to_string()
            ));
        }
        
        if self.distribution.strategy != other.distribution.strategy {
            return Err(LinalgError::InvalidInput(
                "Matrices must have same distribution strategy for addition".to_string()
            ));
        }
        
        // Perform local addition
        let local_result = &self.local_data + &other.local_data;
        
        // Create result matrix with same distribution
        let mut result = DistributedMatrix::from_distribution(
            self.distribution.clone(),
            self.config.clone(),
        )?;
        result.local_data = local_result;
        
        Ok(result)
    }
    
    /// Distributed matrix transpose: B = self^T
    pub fn transpose(&self) -> LinalgResult<DistributedMatrix<T>> {
        match self.distribution.strategy {
            DistributionStrategy::RowWise => self.transpose_row_to_col(),
            DistributionStrategy::ColumnWise => self.transpose_col_to_row(),
            _ => Err(LinalgError::NotImplemented(
                "Transpose for this distribution not implemented".to_string()
            )),
        }
    }
    
    /// Gather all partitions to create a local matrix (only on root node)
    pub fn gather(&self) -> LinalgResult<Option<Array2<T>>> {
        if self.node_rank == 0 {
            // Root node gathers from all others
            let matrices = self.communicator.gather_matrices(&self.local_data.view())?;
            
            if let Some(partitions) = matrices {
                // Reconstruct global matrix
                let mut partition_map = HashMap::new();
                for (rank, matrix) in partitions.into_iter().enumerate() {
                    partition_map.insert(rank, matrix);
                }
                
                let global_matrix = MatrixPartitioner::reconstruct(&partition_map, &self.distribution)?;
                Ok(Some(global_matrix))
            } else {
                Ok(None)
            }
        } else {
            // Non-root nodes just send their data
            self.communicator.gather_matrices(&self.local_data.view())?;
            Ok(None)
        }
    }
    
    /// Broadcast a matrix from root to all nodes
    pub fn broadcast_from_root(
        global_matrix: Option<Array2<T>>,
        config: super::DistributedConfig,
    ) -> LinalgResult<DistributedMatrix<T>> {
        let communicator = Arc::new(DistributedCommunicator::new(&config)?);
        
        if config.node_rank == 0 {
            let matrix = global_matrix.ok_or_else(|| {
                LinalgError::InvalidInput("Root node must provide matrix for broadcast".to_string())
            })?;
            
            // Broadcast to all other nodes
            communicator.broadcast_matrix(&matrix.view())?;
            
            // Create distributed matrix on root
            Self::from_local(matrix, config)
        } else {
            // Receive from root
            let matrix = communicator.recv_matrix(0, MessageTag::Data)?;
            Self::from_local(matrix, config)
        }
    }
    
    /// Perform distributed GEMM using SIMD acceleration on local computations
    pub fn gemm_simd(
        &self,
        other: &DistributedMatrix<T>,
        alpha: T,
        beta: T,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: 'static,
    {
        // Check if SIMD is enabled
        if !self.config.enable_simd {
            return self.multiply(other);
        }
        
        // Use SIMD operations for local computations
        match (T::zero(), T::one()) {
            (_, _) if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() => {
                self.gemm_simd_f32(other, alpha, beta)
            }
            (_, _) if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() => {
                self.gemm_simd_f64(other, alpha, beta)
            }
            _ => {
                // Fallback to regular multiplication
                self.multiply(other)
            }
        }
    }
    
    // Private implementation methods
    
    fn multiply_row_col(&self, other: &DistributedMatrix<T>) -> LinalgResult<DistributedMatrix<T>> {
        // A is row-distributed, B is column-distributed
        // Result C will be row-distributed (same as A)
        
        let (_, k) = self.global_shape();
        let (_, n) = other.global_shape();
        
        // Each node computes its piece of the result
        let mut local_result = Array2::zeros((self.local_data.nrows(), n));
        
        // All-to-all communication to get all pieces of B
        for j in 0..self.config.num_nodes {
            let b_partition = if j == self.node_rank {
                other.local_data.clone()
            } else {
                self.communicator.recv_matrix(j, MessageTag::MatMul)?
            };
            
            if j != self.node_rank {
                self.communicator.send_matrix(&other.local_data.view(), j, MessageTag::MatMul)?;
            }
            
            // Compute local contribution
            let contrib = self.local_data.dot(&b_partition);
            local_result = local_result + contrib;
        }
        
        // Synchronize
        self.coordinator.barrier()?;
        
        // Create result distribution (row-wise like A)
        let result_distribution = DataDistribution::row_wise(
            (self.distribution.global_shape.0, n),
            self.config.num_nodes,
            self.node_rank,
        )?;
        
        let mut result = DistributedMatrix::from_distribution(result_distribution, self.config.clone())?;
        result.local_data = local_result;
        
        Ok(result)
    }
    
    fn multiply_row_row(&self, other: &DistributedMatrix<T>) -> LinalgResult<DistributedMatrix<T>> {
        // Both A and B are row-distributed
        // Need to redistribute B to column-wise for efficient multiplication
        
        let other_transposed = other.transpose()?; // This gives us column-distributed B^T
        let other_col_dist = other_transposed.transpose()?; // Convert to column-distributed B
        
        self.multiply_row_col(&other_col_dist)
    }
    
    fn multiply_col_col(&self, other: &DistributedMatrix<T>) -> LinalgResult<DistributedMatrix<T>> {
        // Both A and B are column-distributed
        // Need to redistribute A to row-wise for efficient multiplication
        
        let self_transposed = self.transpose()?; // This gives us row-distributed A^T
        let self_row_dist = self_transposed.transpose()?; // Convert to row-distributed A
        
        self_row_dist.multiply_row_col(other)
    }
    
    fn transpose_row_to_col(&self) -> LinalgResult<DistributedMatrix<T>> {
        // Convert from row-distributed to column-distributed
        let (m, n) = self.global_shape();
        
        // Create column-wise distribution for result
        let result_distribution = DataDistribution::column_wise(
            (n, m), // Transposed dimensions
            self.config.num_nodes,
            self.node_rank,
        )?;
        
        // All-to-all communication to redistribute data
        let mut result = DistributedMatrix::from_distribution(result_distribution, self.config.clone())?;
        
        // Gather transpose data from all nodes
        if let Some(global_matrix) = self.gather()? {
            // Only root computes transpose
            let transposed = global_matrix.t().to_owned();
            
            // Redistribute transposed matrix
            let redistributed = DistributedMatrix::broadcast_from_root(
                Some(transposed),
                self.config.clone(),
            )?;
            
            result.local_data = redistributed.local_data;
        }
        
        self.coordinator.barrier()?;
        
        Ok(result)
    }
    
    fn transpose_col_to_row(&self) -> LinalgResult<DistributedMatrix<T>> {
        // Convert from column-distributed to row-distributed
        let (m, n) = self.global_shape();
        
        // Create row-wise distribution for result
        let result_distribution = DataDistribution::row_wise(
            (n, m), // Transposed dimensions
            self.config.num_nodes,
            self.node_rank,
        )?;
        
        let mut result = DistributedMatrix::from_distribution(result_distribution, self.config.clone())?;
        
        // Similar to transpose_row_to_col but with different target distribution
        if let Some(global_matrix) = self.gather()? {
            let transposed = global_matrix.t().to_owned();
            let redistributed = DistributedMatrix::broadcast_from_root(
                Some(transposed),
                self.config.clone(),
            )?;
            result.local_data = redistributed.local_data;
        }
        
        self.coordinator.barrier()?;
        
        Ok(result)
    }
    
    fn gemm_simd_f32(&self, other: &DistributedMatrix<T>, alpha: T, beta: T) -> LinalgResult<DistributedMatrix<T>> {
        // This would use the SIMD GEMM operations from the simd_ops module
        // For now, fallback to regular multiplication
        self.multiply(other)
    }
    
    fn gemm_simd_f64(&self, other: &DistributedMatrix<T>, alpha: T, beta: T) -> LinalgResult<DistributedMatrix<T>> {
        // This would use the SIMD GEMM operations from the simd_ops module
        // For now, fallback to regular multiplication
        self.multiply(other)
    }
}

/// Distributed vector implementation
pub struct DistributedVector<T> {
    /// Local partition of the vector
    local_data: Array1<T>,
    /// Global length of the vector
    global_length: usize,
    /// Distribution information
    distribution: VectorDistribution,
    /// Communicator for inter-node communication
    communicator: Arc<DistributedCommunicator>,
    /// Current node rank
    node_rank: usize,
    /// Configuration
    config: super::DistributedConfig,
}

/// Distribution information for vectors
#[derive(Debug, Clone)]
pub struct VectorDistribution {
    /// Global vector length
    pub global_length: usize,
    /// Local vector length on this node
    pub local_length: usize,
    /// Start index for this node
    pub start_index: usize,
    /// End index for this node
    pub end_index: usize,
}

impl VectorDistribution {
    /// Create distribution for a vector
    pub fn new(global_length: usize, num_nodes: usize, node_rank: usize) -> Self {
        let elements_per_node = global_length / num_nodes;
        let remainder = global_length % num_nodes;
        
        let start_index = if node_rank < remainder {
            node_rank * (elements_per_node + 1)
        } else {
            node_rank * elements_per_node + remainder
        };
        
        let end_index = if node_rank < remainder {
            start_index + elements_per_node + 1
        } else {
            start_index + elements_per_node
        };
        
        let local_length = end_index - start_index;
        
        Self {
            global_length,
            local_length,
            start_index,
            end_index,
        }
    }
}

impl<T> DistributedVector<T>
where
    T: Float + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Create a distributed vector from a local vector
    pub fn from_local(
        local_vector: Array1<T>,
        config: super::DistributedConfig,
    ) -> LinalgResult<Self> {
        let global_length = local_vector.len();
        let distribution = VectorDistribution::new(global_length, config.num_nodes, config.node_rank);
        
        // Extract local partition
        let local_data = local_vector.slice(ndarray::s![distribution.start_index..distribution.end_index]).to_owned();
        
        let communicator = Arc::new(DistributedCommunicator::new(&config)?);
        
        Ok(Self {
            local_data,
            global_length,
            distribution,
            communicator,
            node_rank: config.node_rank,
            config,
        })
    }
    
    /// Get global length of the vector
    pub fn global_length(&self) -> usize {
        self.global_length
    }
    
    /// Get local length of this node's partition
    pub fn local_length(&self) -> usize {
        self.local_data.len()
    }
    
    /// Get reference to local data
    pub fn local_data(&self) -> &Array1<T> {
        &self.local_data
    }
    
    /// Get mutable reference to local data
    pub fn local_data_mut(&mut self) -> &mut Array1<T> {
        &mut self.local_data
    }
    
    /// Distributed dot product with another vector
    pub fn dot(&self, other: &DistributedVector<T>) -> LinalgResult<T> {
        if self.global_length != other.global_length {
            return Err(LinalgError::DimensionError(
                "Vector lengths must match for dot product".to_string()
            ));
        }
        
        // Compute local dot product
        let local_dot = self.local_data.dot(&other.local_data);
        
        // All-reduce to sum across all nodes
        let global_dot = self.allreduce_sum(local_dot)?;
        
        Ok(global_dot)
    }
    
    /// Distributed vector addition
    pub fn add(&self, other: &DistributedVector<T>) -> LinalgResult<DistributedVector<T>> {
        if self.global_length != other.global_length {
            return Err(LinalgError::DimensionError(
                "Vector lengths must match for addition".to_string()
            ));
        }
        
        // Perform local addition
        let local_result = &self.local_data + &other.local_data;
        
        // Create result vector
        let mut result = Self::from_local(
            Array1::zeros(self.global_length),
            self.config.clone(),
        )?;
        result.local_data = local_result;
        
        Ok(result)
    }
    
    /// Gather vector to root node
    pub fn gather(&self) -> LinalgResult<Option<Array1<T>>> {
        // Simple implementation - in practice would be more efficient
        let dummy_matrix = self.local_data.clone().insert_axis(Axis(0));
        
        if self.node_rank == 0 {
            let matrices = self.communicator.gather_matrices(&dummy_matrix.view())?;
            if let Some(parts) = matrices {
                let mut result = Array1::zeros(self.global_length);
                for (rank, matrix) in parts.into_iter().enumerate() {
                    let dist = VectorDistribution::new(self.global_length, self.config.num_nodes, rank);
                    let vector = matrix.index_axis(Axis(0), 0);
                    result.slice_mut(ndarray::s![dist.start_index..dist.end_index]).assign(&vector);
                }
                Ok(Some(result))
            } else {
                Ok(None)
            }
        } else {
            self.communicator.gather_matrices(&dummy_matrix.view())?;
            Ok(None)
        }
    }
    
    // Private helper methods
    
    fn allreduce_sum(&self, local_value: T) -> LinalgResult<T> {
        // Simple implementation using gather and broadcast
        let value_matrix = Array2::from_elem((1, 1), local_value);
        
        if let Some(gathered) = self.communicator.gather_matrices(&value_matrix.view())? {
            // Sum all values (only on root)
            let total: T = gathered.iter().map(|m| m[[0, 0]]).fold(T::zero(), |acc, x| acc + x);
            
            // Broadcast result
            let result_matrix = Array2::from_elem((1, 1), total);
            self.communicator.broadcast_matrix(&result_matrix.view())?;
            Ok(total)
        } else {
            // Receive result from root
            let result_matrix = self.communicator.recv_matrix(0, MessageTag::Data)?;
            Ok(result_matrix[[0, 0]])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::DistributedConfig;
    
    #[test]
    fn test_distributed_matrix_creation() {
        let matrix = Array2::from_shape_fn((6, 4), |(i, j)| (i * 4 + j) as f64);
        let config = DistributedConfig::default()
            .with_num_nodes(2)
            .with_node_rank(0)
            .with_distribution(DistributionStrategy::RowWise);
        
        let dist_matrix = DistributedMatrix::from_local(matrix.clone(), config).unwrap();
        
        assert_eq!(dist_matrix.global_shape(), (6, 4));
        assert_eq!(dist_matrix.local_shape().0, 3); // Half the rows
        assert_eq!(dist_matrix.local_shape().1, 4); // All columns
    }
    
    #[test]
    fn test_distributed_vector_creation() {
        let vector = Array1::from_shape_fn(10, |i| i as f64);
        let config = DistributedConfig::default()
            .with_num_nodes(2)
            .with_node_rank(0);
        
        let dist_vector = DistributedVector::from_local(vector, config).unwrap();
        
        assert_eq!(dist_vector.global_length(), 10);
        assert_eq!(dist_vector.local_length(), 5); // Half the elements
    }
    
    #[test]
    fn test_vector_distribution() {
        let dist = VectorDistribution::new(10, 3, 1);
        
        assert_eq!(dist.global_length, 10);
        assert_eq!(dist.start_index, 3);
        assert_eq!(dist.end_index, 6);
        assert_eq!(dist.local_length, 3);
    }
    
    #[test]
    fn test_matrix_local_operations() {
        let matrix1 = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
        let matrix2 = Array2::from_shape_fn((4, 4), |(i, j)| (i * j) as f64);
        
        let config = DistributedConfig::default();
        
        let dist1 = DistributedMatrix::from_local(matrix1, config.clone()).unwrap();
        let dist2 = DistributedMatrix::from_local(matrix2, config).unwrap();
        
        // Test addition (local operation)
        let result = dist1.add(&dist2).unwrap();
        assert_eq!(result.global_shape(), (4, 4));
    }
}