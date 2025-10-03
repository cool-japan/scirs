//! Performance optimizations for large graph operations
//!
//! This module provides performance-optimized algorithms and data structures
//! specifically designed for handling large graphs efficiently.

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (None = use all available cores)
    pub num_threads: Option<usize>,
    /// Chunk size for parallel operations
    pub chunk_size: usize,
    /// Enable SIMD optimizations where available
    pub enable_simd: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            num_threads: None, // Use all available cores
            chunk_size: 1000,
            enable_simd: true,
        }
    }
}

/// Memory-efficient iterator for large graph traversals
pub struct LargeGraphIterator<N: Node, E: EdgeWeight> {
    /// Current position in iteration
    position: usize,
    /// Graph reference
    graph_data: Vec<(N, N, E)>,
    /// Chunk size for memory efficiency
    chunk_size: usize,
}

impl<N: Node, E: EdgeWeight> LargeGraphIterator<N, E> {
    /// Create a new iterator for large graphs
    pub fn new<Ix>(graph: &Graph<N, E, Ix>, chunk_size: usize) -> Self
    where
        N: Clone + std::fmt::Debug,
        E: Clone,
        Ix: petgraph::graph::IndexType,
    {
        let graph_data = graph
            .edges()
            .into_iter()
            .map(|edge| (edge.source, edge.target, edge.weight))
            .collect();

        LargeGraphIterator {
            position: 0,
            graph_data,
            chunk_size,
        }
    }

    /// Get the next chunk of edges
    pub fn next_chunk(&mut self) -> Option<&[(N, N, E)]> {
        if self.position >= self.graph_data.len() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(self.graph_data.len());
        let chunk = &self.graph_data[self.position..end];
        self.position = end;

        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Parallel degree computation for large graphs
#[allow(dead_code)]
pub fn parallel_degree_computation<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &ParallelConfig,
) -> Result<HashMap<N, usize>>
where
    N: Node + Clone + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Send + Sync,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    // Note: Thread pool configuration is handled globally by scirs2-core
    // The num_threads config parameter is preserved for future use but currently ignored

    let nodes: Vec<_> = graph.nodes().into_iter().cloned().collect();

    // Parallel computation of degrees
    let degrees: HashMap<N, usize> = nodes
        .par_chunks(config.chunk_size)
        .map(|chunk| {
            let mut local_degrees = HashMap::new();
            for node in chunk {
                let degree = graph.degree(node);
                local_degrees.insert(node.clone(), degree);
            }
            local_degrees
        })
        .reduce(HashMap::new, |mut acc, local| {
            acc.extend(local);
            acc
        });

    Ok(degrees)
}

/// Memory-efficient parallel shortest path computation
#[allow(dead_code)]
pub fn parallel_shortest_paths<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    sources: &[N],
    _config: &ParallelConfig,
) -> Result<HashMap<N, HashMap<N, E>>>
where
    N: Node + Clone + Send + Sync + std::fmt::Debug,
    E: EdgeWeight
        + Clone
        + Send
        + Sync
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    use crate::algorithms::shortest_path::dijkstra_path;

    // Note: Thread pool configuration is handled globally by scirs2-core
    // The num_threads _config parameter is preserved for future use but currently ignored

    let all_nodes: Vec<_> = graph.nodes().into_iter().cloned().collect();

    // Parallel computation of shortest paths from multiple sources
    let results: HashMap<N, HashMap<N, E>> = sources
        .par_iter()
        .map(|source| {
            let mut paths_from_source = HashMap::new();

            for target in &all_nodes {
                if let Ok(Some(path)) = dijkstra_path(graph, source, target) {
                    paths_from_source.insert(target.clone(), path.total_weight);
                }
            }

            (source.clone(), paths_from_source)
        })
        .collect();

    Ok(results)
}

/// Cache-friendly adjacency matrix computation for large graphs
#[allow(dead_code)]
pub fn cache_friendly_adjacency_matrix<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<Vec<Vec<E>>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + scirs2_core::numeric::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n == 0 {
        return Ok(vec![]);
    }

    // Pre-allocate matrix with cache-friendly access patterns
    let mut matrix = vec![vec![E::zero(); n]; n];

    // Node to index mapping
    let node_to_index: HashMap<N, usize> = graph
        .nodes()
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    // Fill matrix in row-major order for cache efficiency
    for edge in graph.edges() {
        if let (Some(&src_idx), Some(&tgt_idx)) = (
            node_to_index.get(&edge.source),
            node_to_index.get(&edge.target),
        ) {
            matrix[src_idx][tgt_idx] = edge.weight;
            matrix[tgt_idx][src_idx] = edge.weight; // Undirected _graph
        }
    }

    Ok(matrix)
}

/// Streaming algorithm for processing very large graphs
pub struct StreamingGraphProcessor<N: Node, E: EdgeWeight> {
    /// Current batch of edges being processed
    current_batch: Vec<(N, N, E)>,
    /// Maximum batch size
    batch_size: usize,
    /// Running statistics
    edge_count: AtomicUsize,
    /// Degree accumulator
    degree_counter: Arc<parking_lot::Mutex<HashMap<N, usize>>>,
}

impl<N: Node, E: EdgeWeight> StreamingGraphProcessor<N, E>
where
    N: Clone + Send + Sync,
    E: Clone + Send + Sync,
{
    /// Create a new streaming processor
    pub fn new(batch_size: usize) -> Self {
        StreamingGraphProcessor {
            current_batch: Vec::with_capacity(batch_size),
            batch_size,
            edge_count: AtomicUsize::new(0),
            degree_counter: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        }
    }

    /// Add an edge to the streaming processor
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        self.current_batch.push((source, target, weight));

        if self.current_batch.len() >= self.batch_size {
            self.process_batch()?;
        }

        Ok(())
    }

    /// Process the current batch of edges
    fn process_batch(&mut self) -> Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Update edge count
        self.edge_count
            .fetch_add(self.current_batch.len(), Ordering::Relaxed);

        // Update degree counts
        {
            let mut degrees = self.degree_counter.lock();
            for (source, target_, _) in &self.current_batch {
                *degrees.entry(source.clone()).or_insert(0) += 1;
                *degrees.entry(target_.clone()).or_insert(0) += 1;
            }
        }

        // Clear current batch
        self.current_batch.clear();

        Ok(())
    }

    /// Finish processing and return final statistics
    pub fn finish(mut self) -> Result<(usize, HashMap<N, usize>)> {
        // Process remaining edges
        self.process_batch()?;

        let total_edges = self.edge_count.load(Ordering::Relaxed);
        let degrees = Arc::try_unwrap(self.degree_counter)
            .map_err(|_| GraphError::AlgorithmError("Failed to unwrap degree counter".to_string()))?
            .into_inner();

        Ok((total_edges, degrees))
    }

    /// Get current edge count
    pub fn edge_count(&self) -> usize {
        self.edge_count.load(Ordering::Relaxed)
    }
}

/// SIMD-optimized operations for numeric graph computations
#[cfg(target_arch = "x86_64")]
pub mod simd_ops {
    #[allow(unused_imports)]
    use super::*;
    use scirs2_core::simd_ops::SimdUnifiedOps;

    /// SIMD-optimized vector addition for graph metrics
    #[allow(dead_code)]
    pub fn simd_vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());

        // Convert slices to ArrayView1 for SIMD operations
        let a_view = scirs2_core::ndarray::ArrayView1::from(a);
        let b_view = scirs2_core::ndarray::ArrayView1::from(b);

        // Use scirs2-core SIMD operations for optimal performance
        let result = f64::simd_add(&a_view, &b_view);

        // Convert back to Vec<f64>
        result.to_vec()
    }

    /// SIMD-optimized dot product for similarity computations
    #[allow(dead_code)]
    pub fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let a_view = scirs2_core::ndarray::ArrayView1::from(a);
        let b_view = scirs2_core::ndarray::ArrayView1::from(b);

        // Use scirs2-core SIMD optimized dot product
        f64::simd_dot(&a_view, &b_view)
    }

    /// SIMD-optimized vector normalization
    #[allow(dead_code)]
    pub fn simd_normalize(vector: &mut [f64]) {
        let vector_view = scirs2_core::ndarray::ArrayView1::from(&*vector);
        let norm = f64::simd_norm(&vector_view);
        if norm > 0.0 {
            for val in vector.iter_mut() {
                *val /= norm;
            }
        }
    }

    /// SIMD-optimized cosine similarity
    #[allow(dead_code)]
    pub fn simd_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let a_view = scirs2_core::ndarray::ArrayView1::from(a);
        let b_view = scirs2_core::ndarray::ArrayView1::from(b);
        let dot_product = f64::simd_dot(&a_view, &b_view);
        let norm_a = f64::simd_norm(&a_view);
        let norm_b = f64::simd_norm(&b_view);
        dot_product / (norm_a * norm_b)
    }

    /// SIMD-optimized euclidean distance
    #[allow(dead_code)]
    pub fn simd_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let a_view = scirs2_core::ndarray::ArrayView1::from(a);
        let b_view = scirs2_core::ndarray::ArrayView1::from(b);
        let diff = f64::simd_sub(&a_view, &b_view);
        f64::simd_norm(&diff.view())
    }

    /// SIMD-optimized batch centrality computation
    #[allow(dead_code)]
    pub fn simd_batch_centrality_update(
        centralities: &mut [f64],
        contributions: &[f64],
        weights: &[f64],
    ) {
        assert_eq!(centralities.len(), contributions.len());
        assert_eq!(centralities.len(), weights.len());

        // Multiply contributions by weights and add to centralities
        let contrib_view = scirs2_core::ndarray::ArrayView1::from(contributions);
        let weights_view = scirs2_core::ndarray::ArrayView1::from(weights);
        let weighted_contribs = f64::simd_mul(&contrib_view, &weights_view);

        // Manual add-assign since there's no direct simd_add_assign
        for (c, w) in centralities.iter_mut().zip(weighted_contribs.iter()) {
            *c += *w;
        }
    }

    /// SIMD-optimized matrix-vector multiplication for PageRank-style algorithms
    #[allow(dead_code)]
    pub fn simd_sparse_matvec(
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
        x: &[f64],
        y: &mut [f64],
    ) {
        y.fill(0.0);

        for (i, y_i) in y.iter_mut().enumerate() {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];

            // Process row elements in SIMD-optimized chunks
            let row_values = &values[row_start..row_end];
            let row_indices = &col_idx[row_start..row_end];

            // Gather x values corresponding to column indices
            let x_vals: Vec<f64> = row_indices.iter().map(|&j| x[j]).collect();

            // SIMD dot product for this row
            let row_view = scirs2_core::ndarray::ArrayView1::from(row_values);
            let x_view = scirs2_core::ndarray::ArrayView1::from(&x_vals);
            *y_i = f64::simd_dot(&row_view, &x_view);
        }
    }

    /// SIMD-optimized degree computation for multiple nodes
    #[allow(dead_code)]
    pub fn simd_batch_degree_computation(_rowptr: &[usize], degrees: &mut [usize]) {
        for (i, degree) in degrees.iter_mut().enumerate() {
            *degree = _rowptr[i + 1] - _rowptr[i];
        }
    }
}

/// Non-x86_64 fallback implementations
#[cfg(not(target_arch = "x86_64"))]
pub mod simd_ops {
    /// Fallback vector addition
    #[allow(dead_code)]
    pub fn simd_vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Fallback dot product
    #[allow(dead_code)]
    pub fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Fallback normalization
    #[allow(dead_code)]
    pub fn simd_normalize(vector: &mut [f64]) {
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Fallback cosine similarity
    #[allow(dead_code)]
    pub fn simd_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Fallback euclidean distance
    #[allow(dead_code)]
    pub fn simd_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Fallback batch centrality update
    #[allow(dead_code)]
    pub fn simd_batch_centrality_update(
        centralities: &mut [f64],
        contributions: &[f64],
        weights: &[f64],
    ) {
        for ((cent, &contrib), &weight) in centralities
            .iter_mut()
            .zip(contributions.iter())
            .zip(weights.iter())
        {
            *cent += contrib * weight;
        }
    }

    /// Fallback sparse matrix-vector multiplication
    #[allow(dead_code)]
    pub fn simd_sparse_matvec(
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
        x: &[f64],
        y: &mut [f64],
    ) {
        y.fill(0.0);

        for (i, y_i) in y.iter_mut().enumerate() {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];

            for j in row_start..row_end {
                *y_i += values[j] * x[col_idx[j]];
            }
        }
    }

    /// Fallback batch degree computation
    #[allow(dead_code)]
    pub fn simd_batch_degree_computation(_rowptr: &[usize], degrees: &mut [usize]) {
        for (i, degree) in degrees.iter_mut().enumerate() {
            *degree = _rowptr[i + 1] - _rowptr[i];
        }
    }
}

/// Lazy evaluation wrapper for expensive graph computations
pub struct LazyGraphMetric<T> {
    /// The computed value stored in a thread-safe cell
    value: std::sync::OnceLock<std::result::Result<T, GraphError>>,
    /// Computation function stored in a mutex for thread safety
    #[allow(clippy::type_complexity)]
    compute_fn: std::sync::Mutex<Option<Box<dyn FnOnce() -> Result<T> + Send + 'static>>>,
}

impl<T> LazyGraphMetric<T>
where
    T: Send + 'static,
{
    /// Create a new lazy metric
    pub fn new<F>(_computefn: F) -> Self
    where
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        LazyGraphMetric {
            value: std::sync::OnceLock::new(),
            compute_fn: std::sync::Mutex::new(Some(Box::new(_computefn))),
        }
    }

    /// Get the value, computing it if necessary (thread-safe)
    pub fn get(&self) -> Result<&T> {
        let result = self.value.get_or_init(|| {
            // Extract the computation function from the mutex
            let mut fn_guard = self.compute_fn.lock().unwrap();
            if let Some(compute_fn) = fn_guard.take() {
                // Execute the computation
                compute_fn()
            } else {
                // Function already consumed, this shouldn't happen in normal usage
                Err(GraphError::AlgorithmError(
                    "Computation function already consumed".to_string(),
                ))
            }
        });

        match result {
            Ok(value) => Ok(value),
            Err(e) => Err(GraphError::AlgorithmError(format!(
                "Lazy computation failed: {e}"
            ))),
        }
    }

    /// Check if the value has been computed
    pub fn is_computed(&self) -> bool {
        self.value.get().is_some()
    }

    /// Force computation if not already done
    pub fn force(&self) -> Result<()> {
        self.get().map(|_| ())
    }

    /// Get the cached result if available, without triggering computation
    pub fn try_get(&self) -> Option<std::result::Result<&T, &GraphError>> {
        self.value.get().map(|result| match result {
            Ok(value) => Ok(value),
            Err(error) => Err(error),
        })
    }
}

/// Performance-focused memory profiling metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Peak memory usage during operation
    pub peak_bytes: usize,
    /// Average memory usage
    pub average_bytes: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Memory growth rate (bytes per second)
    pub growth_rate: f64,
    /// Potential memory leaks (allocations - deallocations)
    pub potential_leaks: isize,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        MemoryMetrics {
            current_bytes: 0,
            peak_bytes: 0,
            average_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            growth_rate: 0.0,
            potential_leaks: 0,
        }
    }
}

/// Real-time memory profiler for graph operations
pub struct RealTimeMemoryProfiler {
    /// Memory samples over time
    samples: Vec<(std::time::Instant, usize)>,
    /// Start time
    start_time: std::time::Instant,
    /// Allocation tracking
    allocations: AtomicUsize,
    /// Deallocation tracking
    deallocations: AtomicUsize,
    /// Sampling interval in milliseconds
    #[allow(dead_code)]
    sample_interval_ms: u64,
}

impl RealTimeMemoryProfiler {
    /// Create a new real-time profiler
    pub fn new(sample_interval_ms: u64) -> Self {
        RealTimeMemoryProfiler {
            samples: Vec::new(),
            start_time: std::time::Instant::now(),
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
            sample_interval_ms,
        }
    }

    /// Record a memory measurement
    pub fn sample_memory(&mut self, currentmemory: usize) {
        self.samples
            .push((std::time::Instant::now(), currentmemory));
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate comprehensive memory metrics
    pub fn generate_metrics(&self) -> MemoryMetrics {
        if self.samples.is_empty() {
            return MemoryMetrics::default();
        }

        let current_bytes = self.samples.last().map(|(_, mem)| *mem).unwrap_or(0);
        let peak_bytes = self.samples.iter().map(|(_, mem)| *mem).max().unwrap_or(0);
        let average_bytes = if !self.samples.is_empty() {
            self.samples.iter().map(|(_, mem)| *mem).sum::<usize>() / self.samples.len()
        } else {
            0
        };

        let allocation_count = self.allocations.load(Ordering::Relaxed);
        let deallocation_count = self.deallocations.load(Ordering::Relaxed);
        let potential_leaks = allocation_count as isize - deallocation_count as isize;

        // Calculate growth rate
        let growth_rate = if self.samples.len() >= 2 {
            let first = &self.samples[0];
            let last = &self.samples[self.samples.len() - 1];
            let time_diff = last.0.duration_since(first.0).as_secs_f64();
            let memory_diff = last.1 as f64 - first.1 as f64;
            if time_diff > 0.0 {
                memory_diff / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };

        MemoryMetrics {
            current_bytes,
            peak_bytes,
            average_bytes,
            allocation_count,
            deallocation_count,
            growth_rate,
            potential_leaks,
        }
    }

    /// Check for potential memory issues
    pub fn analyze_memory_health(&self) -> Vec<String> {
        let metrics = self.generate_metrics();
        let mut warnings = Vec::new();

        // Check for rapid memory growth
        if metrics.growth_rate > 1_000_000.0 {
            // 1MB/second
            warnings.push(format!(
                "High memory growth rate: {:.2} bytes/second",
                metrics.growth_rate
            ));
        }

        // Check for potential leaks
        if metrics.potential_leaks > 1000 {
            warnings.push(format!(
                "Potential memory leak detected: {} unmatched allocations",
                metrics.potential_leaks
            ));
        }

        // Check for excessive peak memory
        if metrics.peak_bytes > 1_000_000_000 {
            // 1GB
            warnings.push(format!(
                "High peak memory usage: {:.2} MB",
                metrics.peak_bytes as f64 / 1_000_000.0
            ));
        }

        warnings
    }

    /// Export memory timeline for visualization
    pub fn export_timeline(&self) -> Vec<(f64, usize)> {
        self.samples
            .iter()
            .map(|(time, memory)| {
                let elapsed = time.duration_since(self.start_time).as_secs_f64();
                (elapsed, *memory)
            })
            .collect()
    }
}

/// Performance monitoring utilities with enhanced memory profiling
pub struct PerformanceMonitor {
    /// Start time of current operation
    start_time: std::time::Instant,
    /// Operation name
    operation_name: String,
    /// Real-time memory profiler
    memory_profiler: RealTimeMemoryProfiler,
    /// Memory sampling thread handle
    sampling_active: Arc<std::sync::atomic::AtomicBool>,
}

impl PerformanceMonitor {
    /// Start monitoring a new operation with memory profiling
    pub fn start(_operationname: String) -> Self {
        Self::start_with_config(_operationname, 100) // Sample every 100ms by default
    }

    /// Start monitoring with custom sampling interval
    pub fn start_with_config(operation_name: String, sample_intervalms: u64) -> Self {
        PerformanceMonitor {
            start_time: std::time::Instant::now(),
            operation_name,
            memory_profiler: RealTimeMemoryProfiler::new(sample_intervalms),
            sampling_active: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Manually record current memory usage
    pub fn record_memory(&mut self, currentmemory: usize) {
        self.memory_profiler.sample_memory(currentmemory);
    }

    /// Record an allocation event
    pub fn record_allocation(&self, size: usize) {
        self.memory_profiler.record_allocation(size);
    }

    /// Record a deallocation event
    pub fn record_deallocation(&self, size: usize) {
        self.memory_profiler.record_deallocation(size);
    }

    /// Get current memory metrics
    pub fn get_memory_metrics(&self) -> MemoryMetrics {
        self.memory_profiler.generate_metrics()
    }

    /// Check for memory health issues
    pub fn check_memory_health(&self) -> Vec<String> {
        self.memory_profiler.analyze_memory_health()
    }

    /// Get memory timeline for analysis
    pub fn get_memory_timeline(&self) -> Vec<(f64, usize)> {
        self.memory_profiler.export_timeline()
    }

    /// Update peak memory usage (legacy method)
    pub fn update_memory(&mut self, currentmemory: usize) {
        self.record_memory(currentmemory);
    }

    /// Finish monitoring and return comprehensive performance metrics
    pub fn finish(self) -> PerformanceReport {
        self.sampling_active.store(false, Ordering::Relaxed);

        let duration = self.start_time.elapsed();
        let memory_metrics = self.memory_profiler.generate_metrics();
        let memory_warnings = self.memory_profiler.analyze_memory_health();
        let timeline = self.memory_profiler.export_timeline();

        let report = PerformanceReport {
            operation_name: self.operation_name.clone(),
            duration,
            memory_metrics,
            memory_warnings: memory_warnings.clone(),
            timeline,
        };

        println!(
            "Operation '{}' completed in {:?}",
            self.operation_name, duration
        );
        println!(
            "Memory: peak={:.2}MB, avg={:.2}MB, current={:.2}MB",
            report.memory_metrics.peak_bytes as f64 / 1_000_000.0,
            report.memory_metrics.average_bytes as f64 / 1_000_000.0,
            report.memory_metrics.current_bytes as f64 / 1_000_000.0
        );

        if !memory_warnings.is_empty() {
            println!("Memory warnings:");
            for warning in &memory_warnings {
                println!("  - {warning}");
            }
        }

        report
    }
}

/// Comprehensive performance report
#[derive(Debug)]
pub struct PerformanceReport {
    /// Operation name
    pub operation_name: String,
    /// Total execution duration
    pub duration: std::time::Duration,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Memory health warnings
    pub memory_warnings: Vec<String>,
    /// Memory usage timeline
    pub timeline: Vec<(f64, usize)>,
}

/// Optimized graph algorithms trait for large graphs
pub trait LargeGraphOps<N: Node, E: EdgeWeight> {
    /// Parallel computation of node degrees
    fn parallel_degrees(&self, config: &ParallelConfig) -> Result<HashMap<N, usize>>;

    /// Memory-efficient iteration over edges
    fn iter_edges_chunked(&self, chunksize: usize) -> LargeGraphIterator<N, E>;

    /// Cache-friendly matrix representation
    fn cache_friendly_matrix(&self) -> Result<Vec<Vec<E>>>;
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight, Ix: petgraph::graph::IndexType + Send + Sync>
    LargeGraphOps<N, E> for Graph<N, E, Ix>
where
    N: Clone + Send + Sync + std::fmt::Debug,
    E: Clone + Send + Sync + scirs2_core::numeric::Zero + Copy,
{
    fn parallel_degrees(&self, config: &ParallelConfig) -> Result<HashMap<N, usize>> {
        parallel_degree_computation(self, config)
    }

    fn iter_edges_chunked(&self, chunksize: usize) -> LargeGraphIterator<N, E> {
        LargeGraphIterator::new(self, chunksize)
    }

    fn cache_friendly_matrix(&self) -> Result<Vec<Vec<E>>> {
        cache_friendly_adjacency_matrix(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_large_graph_iterator() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();

        let mut iterator = LargeGraphIterator::new(&graph, 2);

        let chunk1 = iterator.next_chunk();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().len(), 2);

        let chunk2 = iterator.next_chunk();
        assert!(chunk2.is_some());
        assert_eq!(chunk2.unwrap().len(), 1);

        let chunk3 = iterator.next_chunk();
        assert!(chunk3.is_none());
    }

    #[test]
    fn test_parallel_degree_computation() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 1, 3.0).unwrap();

        let config = ParallelConfig::default();
        let degrees = graph.parallel_degrees(&config).unwrap();

        assert_eq!(degrees[&1], 2);
        assert_eq!(degrees[&2], 2);
        assert_eq!(degrees[&3], 2);
    }

    #[test]
    fn test_streaming_processor() {
        let mut processor: StreamingGraphProcessor<i32, f64> = StreamingGraphProcessor::new(2);

        processor.add_edge(1, 2, 1.0).unwrap();
        assert_eq!(processor.edge_count(), 0); // Not yet processed

        processor.add_edge(2, 3, 2.0).unwrap();
        assert_eq!(processor.edge_count(), 2); // Batch processed

        let (total_edges, degrees) = processor.finish().unwrap();
        assert_eq!(total_edges, 2);
        assert_eq!(degrees[&1], 1);
        assert_eq!(degrees[&2], 2);
        assert_eq!(degrees[&3], 1);
    }

    #[test]
    fn test_cache_friendly_matrix() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0).unwrap();
        graph.add_edge(1, 2, 2.0).unwrap();

        let matrix = graph.cache_friendly_matrix().unwrap();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0][1], 1.0);
        assert_eq!(matrix[1][2], 2.0);
        assert_eq!(matrix[2][1], 2.0); // Undirected
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::start("test_operation".to_string());

        // Simulate memory usage
        monitor.record_memory(1024);
        monitor.record_memory(2048);
        monitor.record_memory(1536);

        // Simulate allocations
        monitor.record_allocation(1024);
        monitor.record_allocation(512);
        monitor.record_deallocation(256);

        std::thread::sleep(std::time::Duration::from_millis(10));
        let report = monitor.finish();

        assert!(report.duration.as_millis() >= 10);
        assert_eq!(report.memory_metrics.peak_bytes, 2048);
        assert_eq!(report.memory_metrics.current_bytes, 1536);
        assert_eq!(report.memory_metrics.allocation_count, 2);
        assert_eq!(report.memory_metrics.deallocation_count, 1);
        assert_eq!(report.memory_metrics.potential_leaks, 1);
    }

    #[test]
    fn test_real_time_memory_profiler() {
        let mut profiler = RealTimeMemoryProfiler::new(50);

        // Record memory samples
        profiler.sample_memory(1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.sample_memory(2000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.sample_memory(1500);

        // Record allocations/deallocations
        profiler.record_allocation(1000);
        profiler.record_allocation(500);
        profiler.record_deallocation(200);

        let metrics = profiler.generate_metrics();
        assert_eq!(metrics.current_bytes, 1500);
        assert_eq!(metrics.peak_bytes, 2000);
        assert!(metrics.average_bytes > 0);
        assert_eq!(metrics.allocation_count, 2);
        assert_eq!(metrics.deallocation_count, 1);
        assert_eq!(metrics.potential_leaks, 1);

        // Test timeline export
        let timeline = profiler.export_timeline();
        assert_eq!(timeline.len(), 3);
        assert_eq!(timeline[0].1, 1000);
        assert_eq!(timeline[1].1, 2000);
        assert_eq!(timeline[2].1, 1500);
    }

    #[test]
    fn test_memory_health_analysis() {
        let mut profiler = RealTimeMemoryProfiler::new(100);

        // Simulate high memory growth
        profiler.sample_memory(100_000_000);
        std::thread::sleep(std::time::Duration::from_millis(50));
        profiler.sample_memory(200_000_000);

        // Simulate many unmatched allocations
        for _ in 0..1500 {
            profiler.record_allocation(1024);
        }

        let warnings = profiler.analyze_memory_health();
        assert!(!warnings.is_empty());

        // Should warn about high growth rate and potential leaks
        let has_growth_warning = warnings.iter().any(|w| w.contains("growth rate"));
        let has_leak_warning = warnings.iter().any(|w| w.contains("leak"));

        assert!(has_growth_warning);
        assert!(has_leak_warning);
    }

    #[test]
    fn test_memory_metrics_calculation() {
        let mut profiler = RealTimeMemoryProfiler::new(100);

        // Create a clear growth pattern
        profiler.sample_memory(1000);
        std::thread::sleep(std::time::Duration::from_millis(100));
        profiler.sample_memory(2000);
        std::thread::sleep(std::time::Duration::from_millis(100));
        profiler.sample_memory(3000);

        let metrics = profiler.generate_metrics();

        // Should have positive growth rate
        assert!(metrics.growth_rate > 0.0);

        // Average should be around 2000
        assert!(metrics.average_bytes >= 1500 && metrics.average_bytes <= 2500);

        // Peak should be 3000
        assert_eq!(metrics.peak_bytes, 3000);

        // Current should be 3000
        assert_eq!(metrics.current_bytes, 3000);
    }

    #[test]
    fn test_simd_operations() {
        use crate::performance::simd_ops::*;

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Test vector addition
        let sum = simd_vector_add(&a, &b);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        // Test dot product
        let dot = simd_dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6

        // Test cosine similarity
        let similarity = simd_cosine_similarity(&a, &b);
        assert!((similarity - 0.9746318461970762).abs() < 1e-10); // Known cosine similarity

        // Test euclidean distance
        let distance = simd_euclidean_distance(&a, &b);
        assert!((distance - 5.196152422706632).abs() < 1e-10); // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)

        // Test vector normalization
        let mut vector = vec![3.0, 4.0, 0.0];
        simd_normalize(&mut vector);
        let expected_norm =
            (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        assert!((expected_norm - 1.0).abs() < 1e-10);

        // Test batch centrality update
        let mut centralities = vec![1.0, 2.0, 3.0];
        let contributions = vec![0.5, 1.0, 1.5];
        let weights = vec![2.0, 2.0, 2.0];
        simd_batch_centrality_update(&mut centralities, &contributions, &weights);
        assert_eq!(centralities, vec![2.0, 4.0, 6.0]); // 1+0.5*2, 2+1*2, 3+1.5*2
    }

    #[test]
    fn test_sparse_matvec() {
        use crate::performance::simd_ops::*;

        // Create a simple 3x3 sparse matrix in CSR format:
        // [1 0 2]
        // [0 3 0]
        // [1 0 4]
        let row_ptr = vec![0, 2, 3, 5];
        let col_idx = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 1.0, 4.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];

        simd_sparse_matvec(&row_ptr, &col_idx, &values, &x, &mut y);

        // Expected: [1*1 + 2*1, 3*1, 1*1 + 4*1] = [3, 3, 5]
        assert_eq!(y, vec![3.0, 3.0, 5.0]);
    }

    #[test]
    fn test_batch_degree_computation() {
        use crate::performance::simd_ops::*;

        // Row pointers for nodes with degrees [2, 1, 2]
        let row_ptr = vec![0, 2, 3, 5];
        let mut degrees = vec![0; 3];

        simd_batch_degree_computation(&row_ptr, &mut degrees);

        assert_eq!(degrees, vec![2, 1, 2]);
    }

    #[test]
    fn test_lazy_graph_metric() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // Test basic lazy evaluation
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let lazy_metric = LazyGraphMetric::new(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            Ok(42i32)
        });

        // Initially not computed
        assert!(!lazy_metric.is_computed());
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        // First access computes the value
        let result1 = lazy_metric.get().unwrap();
        assert_eq!(*result1, 42);
        assert!(lazy_metric.is_computed());
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        // Second access returns cached value
        let result2 = lazy_metric.get().unwrap();
        assert_eq!(*result2, 42);
        assert_eq!(counter.load(Ordering::Relaxed), 1); // Not incremented again

        // try_get should return the cached value
        assert!(lazy_metric.try_get().is_some());
    }

    #[test]
    fn test_lazy_graph_metric_error() {
        let lazy_metric: LazyGraphMetric<String> =
            LazyGraphMetric::new(|| Err(GraphError::AlgorithmError("Test error".to_string())));

        // Should propagate the error
        let result = lazy_metric.get();
        assert!(result.is_err());

        // Subsequent calls should return the same error
        let result2 = lazy_metric.get();
        assert!(result2.is_err());
    }

    #[test]
    fn test_lazy_graph_metric_thread_safety() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread;

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let lazy_metric = Arc::new(LazyGraphMetric::new(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(10)); // Simulate work
            Ok(100i32)
        }));

        // Spawn multiple threads that try to access the value
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let metric = lazy_metric.clone();
                thread::spawn(move || *metric.get().unwrap())
            })
            .collect();

        // Wait for all threads and collect results
        let results: Vec<i32> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same value
        assert!(results.iter().all(|&x| x == 100));

        // Computation should only happen once
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
