//! NUMA-aware parallel computing
//!
//! This module provides NUMA (Non-Uniform Memory Access) aware algorithms
//! and utilities for optimizing performance on multi-socket systems.

use super::WorkerConfig;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Memory bandwidth between nodes (GB/s)
    pub memory_bandwidth: Vec<Vec<f64>>,
}

impl NumaTopology {
    /// Detect NUMA topology (simplified implementation)
    pub fn detect() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Simple heuristic: assume 2 NUMA nodes if more than 8 CPUs
        let num_nodes = if num_cpus > 8 { 2 } else { 1 };
        let cpus_per_node = if num_nodes == 2 {
            vec![
                (0..num_cpus / 2).collect(),
                (num_cpus / 2..num_cpus).collect(),
            ]
        } else {
            vec![(0..num_cpus).collect()]
        };

        // Simplified bandwidth matrix (local: 100 GB/s, remote: 50 GB/s)
        let mut memory_bandwidth = vec![vec![0.0; num_nodes]; num_nodes];
        for (i, row) in memory_bandwidth.iter_mut().enumerate().take(num_nodes) {
            for (j, item) in row.iter_mut().enumerate().take(num_nodes) {
                *item = if i == j { 100.0 } else { 50.0 };
            }
        }

        Self {
            num_nodes,
            cpus_per_node,
            memory_bandwidth,
        }
    }

    /// Get optimal thread distribution across NUMA nodes
    pub fn optimal_thread_distribution(&self, totalthreads: usize) -> Vec<usize> {
        let mut distribution = vec![0; self.num_nodes];
        let threads_per_node = totalthreads / self.num_nodes;
        let remaining = totalthreads % self.num_nodes;

        for (i, item) in distribution.iter_mut().enumerate().take(self.num_nodes) {
            *item = threads_per_node;
            if i < remaining {
                *item += 1;
            }
        }
        distribution
    }
}

/// NUMA-aware matrix partitioning strategy
#[derive(Debug, Clone, Copy)]
pub enum NumaPartitioning {
    /// Partition by rows across NUMA nodes
    RowWise,
    /// Partition by columns across NUMA nodes
    ColumnWise,
    /// 2D block partitioning across NUMA nodes
    Block2D,
    /// Automatic selection based on matrix shape
    Adaptive,
}

impl NumaPartitioning {
    /// Choose optimal partitioning strategy
    pub fn choose_optimal(_rows: usize, cols: usize, numnodes: usize) -> Self {
        if numnodes == 1 {
            return NumaPartitioning::RowWise;
        }

        let aspect_ratio = _rows as f64 / cols as f64;

        if aspect_ratio > 2.0 {
            // Tall matrix - prefer row-wise partitioning
            NumaPartitioning::RowWise
        } else if aspect_ratio < 0.5 {
            // Wide matrix - prefer column-wise partitioning
            NumaPartitioning::ColumnWise
        } else {
            // Square-ish matrix - use 2D block partitioning
            NumaPartitioning::Block2D
        }
    }
}

/// NUMA-aware parallel matrix-vector multiplication
///
/// This implementation partitions the matrix across NUMA nodes to minimize
/// cross-node memory access and improve cache locality.
pub fn numa_aware_matvec<F>(
    matrix: &ArrayView2<F>,
    vector: &ArrayView1<F>,
    config: &WorkerConfig,
    topology: &NumaTopology,
) -> LinalgResult<Array1<F>>
where
    F: Float + Send + Sync + Zero + std::iter::Sum + 'static,
{
    let (m, n) = matrix.dim();
    if n != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix-vector dimensions incompatible: {}x{} * {}",
            m,
            n,
            vector.len()
        )));
    }

    if topology.num_nodes == 1 {
        // Single NUMA node - use standard parallel implementation
        return super::algorithms::parallel_matvec(matrix, vector, config);
    }

    config.apply();

    // Partition matrix by rows across NUMA nodes
    let _node_distribution =
        topology.optimal_thread_distribution(config.workers.unwrap_or(topology.num_nodes * 2));

    let rows_per_node: Vec<usize> = (0..topology.num_nodes)
        .map(|i| {
            let start_ratio = i as f64 / topology.num_nodes as f64;
            let end_ratio = (i + 1) as f64 / topology.num_nodes as f64;
            let start_row = (start_ratio * m as f64) as usize;
            let end_row = (end_ratio * m as f64) as usize;
            end_row - start_row
        })
        .collect();

    // Compute result for each NUMA node in parallel
    let partial_results: Vec<Vec<F>> = (0..topology.num_nodes)
        .into_par_iter()
        .map(|node_id| {
            let start_row = rows_per_node.iter().take(node_id).sum::<usize>();
            let node_rows = rows_per_node[node_id];

            if node_rows == 0 {
                return Vec::new();
            }

            let nodematrix = matrix.slice(scirs2_core::ndarray::s![
                start_row..start_row + node_rows,
                ..
            ]);

            // Compute local result for this NUMA node
            (0..node_rows)
                .into_par_iter()
                .map(|local_row| {
                    nodematrix
                        .row(local_row)
                        .iter()
                        .zip(vector.iter())
                        .map(|(&a_ij, &x_j)| a_ij * x_j)
                        .sum()
                })
                .collect()
        })
        .collect();

    // Combine results from all NUMA nodes
    let mut result = Vec::with_capacity(m);
    for node_result in partial_results {
        result.extend(node_result);
    }

    Ok(Array1::from_vec(result))
}

/// NUMA-aware parallel matrix multiplication
///
/// Uses 2D block partitioning to distribute computation across NUMA nodes
/// while minimizing cross-node memory traffic.
pub fn numa_aware_gemm<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    config: &WorkerConfig,
    topology: &NumaTopology,
) -> LinalgResult<Array2<F>>
where
    F: Float + Send + Sync + Zero + std::iter::Sum + NumAssign + 'static,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();

    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible: {m}x{k} * {k2}x{n}"
        )));
    }

    if topology.num_nodes == 1 {
        return super::algorithms::parallel_gemm(a, b, config);
    }

    config.apply();

    let partitioning = NumaPartitioning::choose_optimal(m, n, topology.num_nodes);
    let mut result = Array2::zeros((m, n));

    match partitioning {
        NumaPartitioning::Block2D => {
            let nodes_sqrt = (topology.num_nodes as f64).sqrt() as usize;
            let block_rows = m.div_ceil(nodes_sqrt);
            let block_cols = n.div_ceil(nodes_sqrt);

            // Process blocks in parallel across NUMA nodes
            let block_results: Vec<((usize, usize), Array2<F>)> = (0..nodes_sqrt)
                .flat_map(|bi| (0..nodes_sqrt).map(move |bj| (bi, bj)))
                .collect::<Vec<_>>()
                .into_par_iter()
                .filter_map(|(bi, bj)| {
                    let i_start = bi * block_rows;
                    let i_end = std::cmp::min(i_start + block_rows, m);
                    let j_start = bj * block_cols;
                    let j_end = std::cmp::min(j_start + block_cols, n);

                    if i_start >= m || j_start >= n {
                        return None;
                    }

                    let a_block = a.slice(scirs2_core::ndarray::s![i_start..i_end, ..]);
                    let b_block = b.slice(scirs2_core::ndarray::s![.., j_start..j_end]);
                    let block_result = a_block.dot(&b_block);

                    Some(((i_start, j_start), block_result))
                })
                .collect();

            // Combine block results
            for ((i_start, j_start), block_result) in block_results {
                let (block_m, block_n) = block_result.dim();
                for i in 0..block_m {
                    for j in 0..block_n {
                        result[[i_start + i, j_start + j]] = block_result[[i, j]];
                    }
                }
            }
        }
        _ => {
            // Fall back to row-wise partitioning
            let rows_per_node = m / topology.num_nodes;
            let partial_results: Vec<Array2<F>> = (0..topology.num_nodes)
                .into_par_iter()
                .map(|node_id| {
                    let start_row = node_id * rows_per_node;
                    let end_row = if node_id == topology.num_nodes - 1 {
                        m
                    } else {
                        start_row + rows_per_node
                    };

                    let a_partition = a.slice(scirs2_core::ndarray::s![start_row..end_row, ..]);
                    a_partition.dot(b)
                })
                .collect();

            // Combine partial results
            let mut row_offset = 0;
            for partial_result in partial_results {
                let partial_rows = partial_result.nrows();
                for i in 0..partial_rows {
                    for j in 0..n {
                        result[[row_offset + i, j]] = partial_result[[i, j]];
                    }
                }
                row_offset += partial_rows;
            }
        }
    }

    Ok(result)
}

/// NUMA-aware memory allocation hints
///
/// Provides guidance for memory allocation strategies on NUMA systems.
pub struct NumaMemoryStrategy {
    topology: NumaTopology,
}

impl NumaMemoryStrategy {
    /// Create a new NUMA memory strategy
    pub fn new(topology: NumaTopology) -> Self {
        Self { topology }
    }

    /// Get recommended memory allocation for a matrix operation
    pub fn allocatematrix_memory<F>(&self, rows: usize, cols: usize) -> NumaAllocationHint
    where
        F: Float,
    {
        let elementsize = std::mem::size_of::<F>();
        let totalsize = rows * cols * elementsize;
        let size_per_node = totalsize / self.topology.num_nodes;

        NumaAllocationHint {
            strategy: if size_per_node > 1024 * 1024 {
                // Large matrices: distribute across nodes
                NumaAllocationStrategy::Distributed
            } else {
                // Small matrices: allocate locally
                NumaAllocationStrategy::Local
            },
            preferred_nodes: if self.topology.num_nodes > 1 {
                (0..self.topology.num_nodes).collect()
            } else {
                vec![0]
            },
            chunksize: std::cmp::max(4096, size_per_node / 8),
        }
    }

    /// Analyze memory access patterns for optimization
    pub fn analyze_access_pattern(
        &self,
        operation: NumaOperation,
        matrixsizes: &[(usize, usize)],
    ) -> NumaOptimizationHint {
        let total_memory = matrixsizes.iter()
            .map(|(r, c)| r * c * 8) // Assume f64
            .sum::<usize>();

        let memory_per_node = total_memory / self.topology.num_nodes;
        let local_bandwidth = self.topology.memory_bandwidth[0][0];
        let remote_bandwidth = if self.topology.num_nodes > 1 {
            self.topology.memory_bandwidth[0][1]
        } else {
            local_bandwidth
        };

        NumaOptimizationHint {
            operation,
            recommended_partitioning: NumaPartitioning::choose_optimal(
                matrixsizes[0].0,
                matrixsizes[0].1,
                self.topology.num_nodes,
            ),
            memory_per_node,
            expected_local_ratio: local_bandwidth / (local_bandwidth + remote_bandwidth),
            thread_affinity_recommended: memory_per_node > 1024 * 1024, // 1MB threshold
        }
    }
}

/// NUMA allocation strategy
#[derive(Debug, Clone, Copy)]
pub enum NumaAllocationStrategy {
    /// Allocate all memory on local node
    Local,
    /// Distribute memory across all nodes
    Distributed,
    /// Interleave memory across nodes
    Interleaved,
}

/// NUMA allocation hint
#[derive(Debug, Clone)]
pub struct NumaAllocationHint {
    pub strategy: NumaAllocationStrategy,
    pub preferred_nodes: Vec<usize>,
    pub chunksize: usize,
}

/// Type of NUMA operation
#[derive(Debug, Clone, Copy)]
pub enum NumaOperation {
    MatrixVector,
    MatrixMatrix,
    Decomposition,
    IterativeSolver,
}

/// NUMA optimization hint
#[derive(Debug, Clone)]
pub struct NumaOptimizationHint {
    pub operation: NumaOperation,
    pub recommended_partitioning: NumaPartitioning,
    pub memory_per_node: usize,
    pub expected_local_ratio: f64,
    pub thread_affinity_recommended: bool,
}

/// NUMA-aware parallel Cholesky decomposition
///
/// Implements a block-distributed Cholesky decomposition optimized for NUMA.
pub fn numa_aware_cholesky<F>(
    matrix: &ArrayView2<F>,
    config: &WorkerConfig,
    topology: &NumaTopology,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Send
        + Sync
        + Zero
        + One
        + NumAssign
        + scirs2_core::ndarray::ScalarOperand
        + std::iter::Sum
        + 'static,
{
    let (m, n) = matrix.dim();
    if m != n {
        return Err(LinalgError::ShapeError(
            "Cholesky decomposition requires square matrix".to_string(),
        ));
    }

    if topology.num_nodes == 1 {
        return super::algorithms::parallel_cholesky(matrix, config);
    }

    config.apply();

    let blocksize = n / topology.num_nodes;
    let mut l = Array2::zeros((n, n));

    // Distribute blocks across NUMA nodes
    for k in (0..n).step_by(blocksize) {
        let k_end = std::cmp::min(k + blocksize, n);
        let _current_node = k / blocksize;

        // Factorize diagonal block on local node
        for i in k..k_end {
            let mut sum = F::zero();
            for j in 0..i {
                sum += l[[i, j]] * l[[i, j]];
            }
            let aii = matrix[[i, i]] - sum;
            if aii <= F::zero() {
                return Err(LinalgError::ComputationError(
                    "Matrix is not positive definite".to_string(),
                ));
            }
            l[[i, i]] = aii.sqrt();

            // Update column in parallel within the node
            for j in (i + 1)..k_end {
                let mut sum = F::zero();
                for p in 0..i {
                    sum += l[[j, p]] * l[[i, p]];
                }
                l[[j, i]] = (matrix[[j, i]] - sum) / l[[i, i]];
            }
        }

        // Update remaining blocks in parallel across nodes
        if k_end < n {
            let remaining_blocks: Vec<Array2<F>> = (k_end..n)
                .step_by(blocksize)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|block_start| {
                    let block_end = std::cmp::min(block_start + blocksize, n);
                    let mut block_result = Array2::zeros((block_end - block_start, k_end - k));

                    for i in 0..(block_end - block_start) {
                        for j in 0..(k_end - k) {
                            let global_i = block_start + i;
                            let global_j = k + j;

                            let mut sum = F::zero();
                            for p in 0..global_j {
                                sum += l[[global_i, p]] * l[[global_j, p]];
                            }
                            block_result[[i, j]] =
                                (matrix[[global_i, global_j]] - sum) / l[[global_j, global_j]];
                        }
                    }
                    block_result
                })
                .collect();

            // Merge results back into L matrix
            for (block_idx, block_start) in (k_end..n).step_by(blocksize).enumerate() {
                let block_end = std::cmp::min(block_start + blocksize, n);
                let block_result = &remaining_blocks[block_idx];

                for i in 0..(block_end - block_start) {
                    for j in 0..(k_end - k) {
                        l[[block_start + i, k + j]] = block_result[[i, j]];
                    }
                }
            }
        }
    }

    Ok(l)
}

/// NUMA-aware workload balancer
///
/// Balances computational workload across NUMA nodes considering
/// memory bandwidth and CPU capabilities.
pub struct NumaWorkloadBalancer {
    topology: NumaTopology,
    load_history: Arc<Mutex<Vec<f64>>>,
}

impl NumaWorkloadBalancer {
    /// Create a new NUMA workload balancer
    pub fn new(topology: NumaTopology) -> Self {
        let load_history = Arc::new(Mutex::new(vec![0.0; topology.num_nodes]));
        Self {
            topology,
            load_history,
        }
    }

    /// Get optimal work distribution for a given workload
    pub fn distribute_work(&self, total_workunits: usize) -> Vec<usize> {
        let load_history = self.load_history.lock().unwrap();

        // Calculate load-adjusted capacity for each node
        let node_capacities: Vec<f64> = self
            .topology
            .cpus_per_node
            .iter()
            .enumerate()
            .map(|(i, cpus)| {
                let base_capacity = cpus.len() as f64;
                let load_factor = 1.0 - load_history[i].min(0.9); // Cap at 90% penalty
                base_capacity * load_factor
            })
            .collect();

        let total_capacity: f64 = node_capacities.iter().sum();

        // Distribute work proportionally
        let mut distribution = vec![0; self.topology.num_nodes];
        let mut remaining_work = total_workunits;

        for i in 0..self.topology.num_nodes {
            if i == self.topology.num_nodes - 1 {
                // Give remaining work to last node
                distribution[i] = remaining_work;
            } else {
                let node_share =
                    (node_capacities[i] / total_capacity * total_workunits as f64) as usize;
                distribution[i] = node_share;
                remaining_work -= node_share;
            }
        }

        distribution
    }

    /// Update load history after completing work
    pub fn update_load_history(&self, node_id: usize, completion_time: f64, expected_time: f64) {
        let mut load_history = self.load_history.lock().unwrap();

        // Exponential moving average with alpha = 0.1
        let load_ratio = completion_time / expected_time;
        load_history[node_id] = 0.9 * load_history[node_id] + 0.1 * load_ratio;
    }

    /// Get current load information
    pub fn get_load_info(&self) -> Vec<f64> {
        self.load_history.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes >= 1);
        assert_eq!(topology.cpus_per_node.len(), topology.num_nodes);
        assert_eq!(topology.memory_bandwidth.len(), topology.num_nodes);
    }

    #[test]
    fn test_numa_thread_distribution() {
        let topology = NumaTopology {
            num_nodes: 2,
            cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
        };

        let distribution = topology.optimal_thread_distribution(6);
        assert_eq!(distribution, vec![3, 3]);

        let distribution = topology.optimal_thread_distribution(5);
        assert_eq!(distribution, vec![3, 2]);
    }

    #[test]
    fn test_numa_partitioning_strategy() {
        // Tall matrix should prefer row-wise
        assert!(matches!(
            NumaPartitioning::choose_optimal(1000, 100, 2),
            NumaPartitioning::RowWise
        ));

        // Wide matrix should prefer column-wise
        assert!(matches!(
            NumaPartitioning::choose_optimal(100, 1000, 2),
            NumaPartitioning::ColumnWise
        ));

        // Square matrix should prefer 2D blocking
        assert!(matches!(
            NumaPartitioning::choose_optimal(500, 500, 4),
            NumaPartitioning::Block2D
        ));
    }

    #[test]
    fn test_numa_workload_balancer() {
        let topology = NumaTopology {
            num_nodes: 2,
            cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
        };

        let balancer = NumaWorkloadBalancer::new(topology);
        let distribution = balancer.distribute_work(100);

        // Should distribute roughly equally for balanced load
        assert_eq!(distribution.iter().sum::<usize>(), 100);
        assert!(distribution[0] >= 40 && distribution[0] <= 60);
        assert!(distribution[1] >= 40 && distribution[1] <= 60);
    }
}
