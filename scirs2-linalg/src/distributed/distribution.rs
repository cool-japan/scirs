//! Data distribution and load balancing for distributed linear algebra
//!
//! This module provides strategies for distributing data across nodes,
//! load balancing algorithms, and workload partitioning optimizations
//! to maximize performance in distributed linear algebra operations.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategies for distributing matrices across nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionStrategy {
    /// Distribute rows across nodes (row-wise partitioning)
    RowWise,
    /// Distribute columns across nodes (column-wise partitioning)  
    ColumnWise,
    /// Distribute in 2D blocks (block-cyclic distribution)
    BlockCyclic,
    /// Distribute based on computational load
    LoadBased,
    /// Custom distribution pattern
    Custom,
}

/// Data distribution information for a matrix
#[derive(Debug, Clone)]
pub struct DataDistribution {
    /// Distribution strategy used
    pub strategy: DistributionStrategy,
    /// Global matrix dimensions
    pub global_shape: (usize, usize),
    /// Local matrix dimensions on this node
    pub local_shape: (usize, usize),
    /// Global indices that this node owns
    pub owned_indices: IndexRange,
    /// Mapping of global indices to owning nodes
    pub index_map: HashMap<(usize, usize), usize>,
    /// Block size for block-based distributions
    pub block_size: (usize, usize),
}

/// Range of indices owned by a node
#[derive(Debug, Clone)]
pub struct IndexRange {
    /// Row range (start, end)
    pub rows: (usize, usize),
    /// Column range (start, end)
    pub columns: (usize, usize),
}

impl IndexRange {
    /// Create new index range
    pub fn new(row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        Self {
            rows: (row_start, row_end),
            columns: (col_start, col_end),
        }
    }
    
    /// Check if indices are in this range
    pub fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.rows.0 && row < self.rows.1 && col >= self.columns.0 && col < self.columns.1
    }
    
    /// Get number of rows in range
    pub fn num_rows(&self) -> usize {
        self.rows.1 - self.rows.0
    }
    
    /// Get number of columns in range
    pub fn num_cols(&self) -> usize {
        self.columns.1 - self.columns.0
    }
    
    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.num_rows() * self.num_cols()
    }
}

impl DataDistribution {
    /// Create row-wise distribution
    pub fn row_wise(
        global_shape: (usize, usize),
        num_nodes: usize,
        node_rank: usize,
    ) -> LinalgResult<Self> {
        let (global_rows, global_cols) = global_shape;
        let rows_per_node = global_rows / num_nodes;
        let remainder = global_rows % num_nodes;
        
        // Calculate this node's row range
        let start_row = if node_rank < remainder {
            node_rank * (rows_per_node + 1)
        } else {
            node_rank * rows_per_node + remainder
        };
        
        let end_row = if node_rank < remainder {
            start_row + rows_per_node + 1
        } else {
            start_row + rows_per_node
        };
        
        let local_rows = end_row - start_row;
        let local_shape = (local_rows, global_cols);
        let owned_indices = IndexRange::new(start_row, end_row, 0, global_cols);
        
        // Create index map
        let mut index_map = HashMap::new();
        for node in 0..num_nodes {
            let node_start = if node < remainder {
                node * (rows_per_node + 1)
            } else {
                node * rows_per_node + remainder
            };
            
            let node_end = if node < remainder {
                node_start + rows_per_node + 1
            } else {
                node_start + rows_per_node
            };
            
            for row in node_start..node_end {
                for col in 0..global_cols {
                    index_map.insert((row, col), node);
                }
            }
        }
        
        Ok(Self {
            strategy: DistributionStrategy::RowWise,
            global_shape,
            local_shape,
            owned_indices,
            index_map,
            block_size: (1, global_cols),
        })
    }
    
    /// Create column-wise distribution  
    pub fn column_wise(
        global_shape: (usize, usize),
        num_nodes: usize,
        node_rank: usize,
    ) -> LinalgResult<Self> {
        let (global_rows, global_cols) = global_shape;
        let cols_per_node = global_cols / num_nodes;
        let remainder = global_cols % num_nodes;
        
        // Calculate this node's column range
        let start_col = if node_rank < remainder {
            node_rank * (cols_per_node + 1)
        } else {
            node_rank * cols_per_node + remainder
        };
        
        let end_col = if node_rank < remainder {
            start_col + cols_per_node + 1
        } else {
            start_col + cols_per_node
        };
        
        let local_cols = end_col - start_col;
        let local_shape = (global_rows, local_cols);
        let owned_indices = IndexRange::new(0, global_rows, start_col, end_col);
        
        // Create index map
        let mut index_map = HashMap::new();
        for node in 0..num_nodes {
            let node_start = if node < remainder {
                node * (cols_per_node + 1)
            } else {
                node * cols_per_node + remainder
            };
            
            let node_end = if node < remainder {
                node_start + cols_per_node + 1
            } else {
                node_start + cols_per_node
            };
            
            for row in 0..global_rows {
                for col in node_start..node_end {
                    index_map.insert((row, col), node);
                }
            }
        }
        
        Ok(Self {
            strategy: DistributionStrategy::ColumnWise,
            global_shape,
            local_shape,
            owned_indices,
            index_map,
            block_size: (global_rows, 1),
        })
    }
    
    /// Create block-cyclic distribution
    pub fn block_cyclic(
        global_shape: (usize, usize),
        num_nodes: usize,
        node_rank: usize,
        block_size: (usize, usize),
    ) -> LinalgResult<Self> {
        let (global_rows, global_cols) = global_shape;
        let (block_rows, block_cols) = block_size;
        
        // Calculate grid dimensions
        let grid_rows = (global_rows + block_rows - 1) / block_rows;
        let grid_cols = (global_cols + block_cols - 1) / block_cols;
        
        // Determine 2D processor grid
        let proc_grid_rows = (num_nodes as f64).sqrt() as usize;
        let proc_grid_cols = (num_nodes + proc_grid_rows - 1) / proc_grid_rows;
        
        let proc_row = node_rank / proc_grid_cols;
        let proc_col = node_rank % proc_grid_cols;
        
        // Calculate owned blocks
        let mut owned_blocks = Vec::new();
        for grid_r in (proc_row..grid_rows).step_by(proc_grid_rows) {
            for grid_c in (proc_col..grid_cols).step_by(proc_grid_cols) {
                owned_blocks.push((grid_r, grid_c));
            }
        }
        
        // Calculate local shape (approximate)
        let local_rows = owned_blocks.iter()
            .map(|(gr, _)| {
                let start_row = gr * block_rows;
                let end_row = ((gr + 1) * block_rows).min(global_rows);
                end_row - start_row
            })
            .sum::<usize>();
            
        let local_cols = owned_blocks.iter()
            .map(|(_, gc)| {
                let start_col = gc * block_cols;
                let end_col = ((gc + 1) * block_cols).min(global_cols);
                end_col - start_col
            })
            .max()
            .unwrap_or(0);
        
        let local_shape = (local_rows, local_cols);
        
        // For simplicity, use first block's range as owned_indices
        let owned_indices = if !owned_blocks.is_empty() {
            let (first_gr, first_gc) = owned_blocks[0];
            let start_row = first_gr * block_rows;
            let end_row = ((first_gr + 1) * block_rows).min(global_rows);
            let start_col = first_gc * block_cols;
            let end_col = ((first_gc + 1) * block_cols).min(global_cols);
            IndexRange::new(start_row, end_row, start_col, end_col)
        } else {
            IndexRange::new(0, 0, 0, 0)
        };
        
        // Create simplified index map (full implementation would be more complex)
        let mut index_map = HashMap::new();
        for &(gr, gc) in &owned_blocks {
            let start_row = gr * block_rows;
            let end_row = ((gr + 1) * block_rows).min(global_rows);
            let start_col = gc * block_cols;
            let end_col = ((gc + 1) * block_cols).min(global_cols);
            
            for row in start_row..end_row {
                for col in start_col..end_col {
                    index_map.insert((row, col), node_rank);
                }
            }
        }
        
        Ok(Self {
            strategy: DistributionStrategy::BlockCyclic,
            global_shape,
            local_shape,
            owned_indices,
            index_map,
            block_size,
        })
    }
    
    /// Get the owner node for given global indices
    pub fn get_owner(&self, row: usize, col: usize) -> Option<usize> {
        self.index_map.get(&(row, col)).copied()
    }
    
    /// Check if this node owns the given indices
    pub fn owns(&self, row: usize, col: usize) -> bool {
        self.owned_indices.contains(row, col)
    }
    
    /// Convert global indices to local indices
    pub fn global_to_local(&self, global_row: usize, global_col: usize) -> Option<(usize, usize)> {
        if !self.owns(global_row, global_col) {
            return None;
        }
        
        match self.strategy {
            DistributionStrategy::RowWise => {
                let local_row = global_row - self.owned_indices.rows.0;
                Some((local_row, global_col))
            },
            DistributionStrategy::ColumnWise => {
                let local_col = global_col - self.owned_indices.columns.0;
                Some((global_row, local_col))
            },
            DistributionStrategy::BlockCyclic => {
                // Simplified implementation
                let local_row = global_row - self.owned_indices.rows.0;
                let local_col = global_col - self.owned_indices.columns.0;
                Some((local_row, local_col))
            },
            _ => None,
        }
    }
    
    /// Convert local indices to global indices
    pub fn local_to_global(&self, local_row: usize, local_col: usize) -> (usize, usize) {
        match self.strategy {
            DistributionStrategy::RowWise => {
                (local_row + self.owned_indices.rows.0, local_col)
            },
            DistributionStrategy::ColumnWise => {
                (local_row, local_col + self.owned_indices.columns.0)
            },
            DistributionStrategy::BlockCyclic => {
                // Simplified implementation
                (local_row + self.owned_indices.rows.0, local_col + self.owned_indices.columns.0)
            },
            _ => (local_row, local_col),
        }
    }
}

/// Load balancer for distributed computations
pub struct LoadBalancer {
    /// Node computational capabilities
    node_capabilities: HashMap<usize, f64>,
    /// Historical performance data
    performance_history: HashMap<usize, Vec<f64>>,
    /// Current workload distribution
    workload_distribution: HashMap<usize, f64>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(config: &super::DistributedConfig) -> LinalgResult<Self> {
        let mut node_capabilities = HashMap::new();
        for rank in 0..config.num_nodes {
            // Assume equal capabilities initially
            node_capabilities.insert(rank, 1.0);
        }
        
        Ok(Self {
            node_capabilities,
            performance_history: HashMap::new(),
            workload_distribution: HashMap::new(),
        })
    }
    
    /// Update node capability based on performance measurement
    pub fn update_capability(&mut self, node_rank: usize, performance_metric: f64) {
        // Add to performance history
        self.performance_history
            .entry(node_rank)
            .or_insert_with(Vec::new)
            .push(performance_metric);
        
        // Keep only recent measurements
        let history = self.performance_history.get_mut(&node_rank).unwrap();
        if history.len() > 10 {
            history.drain(0..history.len() - 10);
        }
        
        // Update capability as average of recent performance
        let avg_performance = history.iter().sum::<f64>() / history.len() as f64;
        self.node_capabilities.insert(node_rank, avg_performance);
    }
    
    /// Calculate optimal workload distribution
    pub fn calculate_workload_distribution(&self, total_work: f64) -> HashMap<usize, f64> {
        let total_capability: f64 = self.node_capabilities.values().sum();
        
        if total_capability == 0.0 {
            return HashMap::new();
        }
        
        let mut distribution = HashMap::new();
        for (&node_rank, &capability) in &self.node_capabilities {
            let work_fraction = capability / total_capability;
            distribution.insert(node_rank, total_work * work_fraction);
        }
        
        distribution
    }
    
    /// Suggest redistribution based on current performance
    pub fn suggest_redistribution(&self) -> Option<RedistributionPlan> {
        // Calculate load imbalance
        let workloads: Vec<f64> = self.workload_distribution.values().cloned().collect();
        if workloads.is_empty() {
            return None;
        }
        
        let avg_workload = workloads.iter().sum::<f64>() / workloads.len() as f64;
        let max_workload = workloads.iter().cloned().fold(0.0, f64::max);
        let min_workload = workloads.iter().cloned().fold(f64::INFINITY, f64::min);
        
        let imbalance_ratio = (max_workload - min_workload) / avg_workload;
        
        // Suggest redistribution if imbalance is significant
        if imbalance_ratio > 0.2 {
            // Find overloaded and underloaded nodes
            let mut overloaded = Vec::new();
            let mut underloaded = Vec::new();
            
            for (&node_rank, &workload) in &self.workload_distribution {
                if workload > avg_workload * 1.1 {
                    overloaded.push(node_rank);
                } else if workload < avg_workload * 0.9 {
                    underloaded.push(node_rank);
                }
            }
            
            Some(RedistributionPlan {
                from_nodes: overloaded,
                to_nodes: underloaded,
                suggested_strategy: DistributionStrategy::LoadBased,
                imbalance_ratio,
            })
        } else {
            None
        }
    }
    
    /// Record current workload for a node
    pub fn record_workload(&mut self, node_rank: usize, workload: f64) {
        self.workload_distribution.insert(node_rank, workload);
    }
    
    /// Get current load balance efficiency (0.0 to 1.0)
    pub fn get_efficiency(&self) -> f64 {
        if self.workload_distribution.is_empty() {
            return 1.0;
        }
        
        let workloads: Vec<f64> = self.workload_distribution.values().cloned().collect();
        let avg_workload = workloads.iter().sum::<f64>() / workloads.len() as f64;
        
        if avg_workload == 0.0 {
            return 1.0;
        }
        
        let variance: f64 = workloads
            .iter()
            .map(|w| (w - avg_workload).powi(2))
            .sum::<f64>() / workloads.len() as f64;
        
        let coefficient_of_variation = variance.sqrt() / avg_workload;
        
        // Efficiency is inversely related to coefficient of variation
        1.0 / (1.0 + coefficient_of_variation)
    }
}

/// Plan for redistributing workload across nodes
#[derive(Debug, Clone)]
pub struct RedistributionPlan {
    /// Nodes with excess workload
    pub from_nodes: Vec<usize>,
    /// Nodes that can take more workload
    pub to_nodes: Vec<usize>,
    /// Suggested distribution strategy
    pub suggested_strategy: DistributionStrategy,
    /// Current imbalance ratio
    pub imbalance_ratio: f64,
}

/// Partitioner for creating distributed matrix partitions
pub struct MatrixPartitioner;

impl MatrixPartitioner {
    /// Partition a matrix according to distribution strategy
    pub fn partition<T>(
        matrix: &ArrayView2<T>,
        distribution: &DataDistribution,
    ) -> LinalgResult<Array2<T>>
    where
        T: Clone,
    {
        let (global_rows, global_cols) = matrix.dim();
        
        // Validate that matrix matches distribution
        if (global_rows, global_cols) != distribution.global_shape {
            return Err(LinalgError::DimensionError(format!(
                "Matrix shape {:?} doesn't match distribution shape {:?}",
                (global_rows, global_cols),
                distribution.global_shape
            )));
        }
        
        // Extract local partition
        let IndexRange { rows, columns } = &distribution.owned_indices;
        let local_matrix = matrix.slice(ndarray::s![rows.0..rows.1, columns.0..columns.1]);
        
        Ok(local_matrix.to_owned())
    }
    
    /// Reconstruct global matrix from distributed partitions
    pub fn reconstruct<T>(
        partitions: &HashMap<usize, Array2<T>>,
        distribution: &DataDistribution,
    ) -> LinalgResult<Array2<T>>
    where
        T: Clone + Default,
    {
        let mut global_matrix = Array2::default(distribution.global_shape);
        
        // Place each partition in the correct location
        for (&node_rank, partition) in partitions {
            // Find this node's index range (simplified)
            if let Some(range) = Self::get_node_range(node_rank, distribution) {
                let target_slice = global_matrix.slice_mut(ndarray::s![
                    range.rows.0..range.rows.1,
                    range.columns.0..range.columns.1
                ]);
                
                // Copy partition data
                if target_slice.shape() == partition.shape() {
                    target_slice.assign(partition);
                }
            }
        }
        
        Ok(global_matrix)
    }
    
    /// Get index range for a specific node (helper method)
    fn get_node_range(node_rank: usize, distribution: &DataDistribution) -> Option<IndexRange> {
        // This is a simplified implementation
        // In practice, we'd need to reconstruct the range from the distribution strategy
        Some(distribution.owned_indices.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_index_range() {
        let range = IndexRange::new(10, 20, 5, 15);
        
        assert_eq!(range.num_rows(), 10);
        assert_eq!(range.num_cols(), 10);
        assert_eq!(range.size(), 100);
        
        assert!(range.contains(15, 10));
        assert!(!range.contains(5, 10));
        assert!(!range.contains(15, 20));
    }
    
    #[test]
    fn test_row_wise_distribution() {
        let distribution = DataDistribution::row_wise((100, 50), 4, 1).unwrap();
        
        assert_eq!(distribution.strategy, DistributionStrategy::RowWise);
        assert_eq!(distribution.global_shape, (100, 50));
        assert_eq!(distribution.local_shape.1, 50); // All columns
        
        // Node 1 should own rows 25-49
        assert!(distribution.owns(30, 10));
        assert!(!distribution.owns(10, 10));
    }
    
    #[test]
    fn test_column_wise_distribution() {
        let distribution = DataDistribution::column_wise((100, 50), 4, 2).unwrap();
        
        assert_eq!(distribution.strategy, DistributionStrategy::ColumnWise);
        assert_eq!(distribution.global_shape, (100, 50));
        assert_eq!(distribution.local_shape.0, 100); // All rows
        
        // Node 2 should own some columns
        assert!(distribution.local_shape.1 > 0);
    }
    
    #[test]
    fn test_load_balancer() {
        use super::super::DistributedConfig;
        
        let config = DistributedConfig::default().with_num_nodes(3);
        let mut balancer = LoadBalancer::new(&config).unwrap();
        
        // Update capabilities
        balancer.update_capability(0, 1.0);
        balancer.update_capability(1, 2.0);
        balancer.update_capability(2, 0.5);
        
        // Calculate workload distribution
        let distribution = balancer.calculate_workload_distribution(100.0);
        
        assert!(distribution.len() == 3);
        assert!(distribution[&1] > distribution[&0]); // Node 1 should get more work
        assert!(distribution[&2] < distribution[&0]); // Node 2 should get less work
    }
    
    #[test]
    fn test_matrix_partitioner() {
        let matrix = Array2::from_shape_fn((10, 8), |(i, j)| (i * 8 + j) as f64);
        let distribution = DataDistribution::row_wise((10, 8), 2, 0).unwrap();
        
        let partition = MatrixPartitioner::partition(&matrix.view(), &distribution).unwrap();
        
        assert_eq!(partition.nrows(), 5); // First half of rows
        assert_eq!(partition.ncols(), 8); // All columns
    }
}