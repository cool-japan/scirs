//! CPU affinity and thread pinning
//!
//! This module provides advanced thread affinity management for optimal
//! CPU utilization and NUMA-aware scheduling.

use super::{numa::NumaTopology, WorkerConfig};
use std::sync::{Arc, Mutex};

/// Thread affinity strategy
#[derive(Debug, Clone, Copy)]
pub enum AffinityStrategy {
    /// No specific affinity - let OS scheduler decide
    None,
    /// Pin threads to specific CPU cores
    Pinned,
    /// Spread threads across NUMA nodes
    NumaSpread,
    /// Compact threads within NUMA nodes
    NumaCompact,
    /// Custom affinity mapping
    Custom,
}

/// CPU core affinity mask
#[derive(Debug, Clone)]
pub struct CoreAffinity {
    /// CPU core IDs to bind threads to
    pub core_ids: Vec<usize>,
    /// Whether to allow thread migration
    pub allow_migration: bool,
    /// NUMA node preference
    pub numa_node: Option<usize>,
}

impl CoreAffinity {
    /// Create affinity for specific cores
    pub fn cores(_coreids: Vec<usize>) -> Self {
        Self {
            core_ids: _coreids,
            allow_migration: false,
            numa_node: None,
        }
    }

    /// Create affinity for a NUMA node
    pub fn numa_node(_nodeid: usize, topology: &NumaTopology) -> Self {
        let core_ids = if _nodeid < topology.cpus_per_node.len() {
            topology.cpus_per_node[_nodeid].clone()
        } else {
            vec![]
        };

        Self {
            core_ids,
            allow_migration: true,
            numa_node: Some(_nodeid),
        }
    }

    /// Allow thread migration between specified cores
    pub fn with_migration(mut self, allow: bool) -> Self {
        self.allow_migration = allow;
        self
    }
}

/// Thread affinity manager
pub struct AffinityManager {
    strategy: AffinityStrategy,
    topology: NumaTopology,
    thread_assignments: Arc<Mutex<Vec<Option<CoreAffinity>>>>,
}

impl AffinityManager {
    /// Create a new affinity manager
    pub fn new(strategy: AffinityStrategy, topology: NumaTopology) -> Self {
        let thread_assignments = Arc::new(Mutex::new(Vec::new()));
        Self {
            strategy,
            topology,
            thread_assignments,
        }
    }

    /// Generate thread affinity assignments based on strategy
    pub fn generate_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
        match self.strategy {
            AffinityStrategy::None => {
                // No specific affinity
                vec![]
            }
            AffinityStrategy::Pinned => self.generate_pinned_assignments(numthreads),
            AffinityStrategy::NumaSpread => self.generate_numa_spread_assignments(numthreads),
            AffinityStrategy::NumaCompact => self.generate_numa_compact_assignments(numthreads),
            AffinityStrategy::Custom => {
                // Use existing assignments
                self.thread_assignments
                    .lock()
                    .unwrap()
                    .iter()
                    .filter_map(|opt| opt.clone())
                    .collect()
            }
        }
    }

    /// Generate pinned affinity assignments (one thread per core)
    fn generate_pinned_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
        let total_cores: usize = self
            .topology
            .cpus_per_node
            .iter()
            .map(|node| node.len())
            .sum();

        let effective_threads = std::cmp::min(numthreads, total_cores);
        let mut all_cores: Vec<usize> = self
            .topology
            .cpus_per_node
            .iter()
            .flat_map(|node| node.iter().cloned())
            .collect();

        // Sort cores for consistent assignment
        all_cores.sort_unstable();

        (0..effective_threads)
            .map(|i| CoreAffinity::cores(vec![all_cores[i]]))
            .collect()
    }

    /// Generate NUMA-spread assignments (distribute across nodes)
    fn generate_numa_spread_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
        let mut assignments = Vec::new();
        let threads_per_node = numthreads / self.topology.num_nodes;
        let extra_threads = numthreads % self.topology.num_nodes;

        for (node_id, cores) in self.topology.cpus_per_node.iter().enumerate() {
            let node_threads = threads_per_node + if node_id < extra_threads { 1 } else { 0 };

            for i in 0..node_threads {
                if i < cores.len() {
                    assignments.push(CoreAffinity::cores(vec![cores[i]]).with_migration(false));
                } else {
                    // More _threads than cores in this node - allow migration
                    assignments.push(
                        CoreAffinity::numa_node(node_id, &self.topology).with_migration(true),
                    );
                }
            }
        }

        assignments
    }

    /// Generate NUMA-compact assignments (fill nodes sequentially)
    fn generate_numa_compact_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
        let mut assignments = Vec::new();
        let mut remaining_threads = numthreads;

        for (node_id, cores) in self.topology.cpus_per_node.iter().enumerate() {
            if remaining_threads == 0 {
                break;
            }

            let node_capacity = cores.len();
            let threads_for_node = std::cmp::min(remaining_threads, node_capacity);

            // Assign _threads to specific cores in this node
            for core in cores.iter().take(threads_for_node) {
                assignments.push(CoreAffinity::cores(vec![*core]).with_migration(false));
            }

            // If more _threads needed than cores, allow migration within node
            if remaining_threads > node_capacity {
                for _ in node_capacity..remaining_threads.min(node_capacity * 2) {
                    assignments.push(
                        CoreAffinity::numa_node(node_id, &self.topology).with_migration(true),
                    );
                }
            }

            remaining_threads = remaining_threads.saturating_sub(node_capacity * 2);
        }

        assignments
    }

    /// Set custom affinity for a specific thread
    pub fn set_thread_affinity(&self, threadid: usize, affinity: CoreAffinity) {
        let mut assignments = self.thread_assignments.lock().unwrap();

        // Expand vector if needed
        while assignments.len() <= threadid {
            assignments.push(None);
        }

        assignments[threadid] = Some(affinity);
    }

    /// Get affinity for a specific thread
    pub fn get_thread_affinity(&self, threadid: usize) -> Option<CoreAffinity> {
        let assignments = self.thread_assignments.lock().unwrap();
        assignments.get(threadid).and_then(|opt| opt.clone())
    }

    /// Apply affinity settings to current thread (platform-specific)
    pub fn apply_current_thread_affinity(&self, _affinity: &CoreAffinity) -> Result<(), String> {
        // Note: This is a simplified implementation
        // In a real implementation, you would use platform-specific APIs:
        // - On Linux: sched_setaffinity, pthread_setaffinity_np
        // - On Windows: SetThreadAffinityMask
        // - On macOS: thread_policy_set

        #[cfg(target_os = "linux")]
        {
            self.apply_linux_affinity(_affinity)
        }

        #[cfg(target_os = "windows")]
        {
            self.apply_windows_affinity(_affinity)
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // Fallback for unsupported platforms
            eprintln!("CPU affinity not supported on this platform");
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    fn apply_linux_affinity(&self, affinity: &CoreAffinity) -> Result<(), String> {
        // This would typically use libc::sched_setaffinity
        // For now, we'll just set environment variables that some libraries recognize
        if !affinity.core_ids.is_empty() {
            let core_list = affinity
                .core_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");

            std::env::set_var("GOMP_CPU_AFFINITY", &core_list);
            std::env::set_var("KMP_AFFINITY", format!("explicit,proclist=[{core_list}]"));
        }

        if let Some(numa_node) = affinity.numa_node {
            std::env::set_var("NUMA_NODE_HINT", numa_node.to_string());
        }

        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn apply_windows_affinity(&self, affinity: &CoreAffinity) -> Result<(), String> {
        // This would typically use Windows APIs like SetThreadAffinityMask
        // For now, we'll set environment variables
        if !affinity.core_ids.is_empty() {
            let core_mask: u64 = affinity
                .core_ids
                .iter()
                .fold(0u64, |mask, &core_id| mask | (1u64 << core_id));

            std::env::set_var("THREAD_AFFINITY_MASK", format!("0x{:x}", core_mask));
        }

        Ok(())
    }

    /// Get optimal affinity strategy for current system
    pub fn recommend_strategy(
        &self,
        num_threads: usize,
        workload_type: WorkloadType,
    ) -> AffinityStrategy {
        match workload_type {
            WorkloadType::CpuBound => {
                if num_threads <= self.total_cores() {
                    AffinityStrategy::Pinned
                } else {
                    AffinityStrategy::NumaSpread
                }
            }
            WorkloadType::MemoryBound => {
                if self.topology.num_nodes > 1 {
                    AffinityStrategy::NumaCompact
                } else {
                    AffinityStrategy::Pinned
                }
            }
            WorkloadType::Balanced => {
                if self.topology.num_nodes > 1 && num_threads >= self.topology.num_nodes {
                    AffinityStrategy::NumaSpread
                } else {
                    AffinityStrategy::Pinned
                }
            }
            WorkloadType::Latency => AffinityStrategy::Pinned,
        }
    }

    /// Get total number of CPU cores
    fn total_cores(&self) -> usize {
        self.topology
            .cpus_per_node
            .iter()
            .map(|node| node.len())
            .sum()
    }
}

/// Type of computational workload
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    /// CPU-intensive workload
    CpuBound,
    /// Memory-intensive workload
    MemoryBound,
    /// Balanced CPU and memory usage
    Balanced,
    /// Latency-sensitive workload
    Latency,
}

/// Thread pool with affinity support
pub struct AffinityThreadPool {
    affinity_manager: AffinityManager,
    config: WorkerConfig,
}

impl AffinityThreadPool {
    /// Create a new affinity-aware thread pool
    pub fn new(strategy: AffinityStrategy, topology: NumaTopology, config: WorkerConfig) -> Self {
        let affinity_manager = AffinityManager::new(strategy, topology);
        Self {
            affinity_manager,
            config,
        }
    }

    /// Execute work with affinity-pinned threads
    pub fn execute_with_affinity<F, R>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let num_threads = self.config.workers.unwrap_or(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        );

        let assignments = self.affinity_manager.generate_assignments(num_threads);

        // Apply affinity to current thread if assignments available
        if let Some(affinity) = assignments.first() {
            if let Err(e) = self
                .affinity_manager
                .apply_current_thread_affinity(affinity)
            {
                eprintln!("Warning: Failed to set thread affinity: {e}");
            }
        }

        // Execute the work
        work()
    }

    /// Get affinity information for debugging
    pub fn get_affinity_info(&self) -> AffinityInfo {
        let num_threads = self.config.workers.unwrap_or(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        );

        let assignments = self.affinity_manager.generate_assignments(num_threads);

        AffinityInfo {
            strategy: self.affinity_manager.strategy,
            num_threads,
            assignments,
            topology: self.affinity_manager.topology.clone(),
        }
    }
}

/// Affinity information for debugging and monitoring
#[derive(Debug, Clone)]
pub struct AffinityInfo {
    pub strategy: AffinityStrategy,
    pub num_threads: usize,
    pub assignments: Vec<CoreAffinity>,
    pub topology: NumaTopology,
}

impl AffinityInfo {
    /// Print detailed affinity information
    pub fn print_summary(&self) {
        println!("=== Thread Affinity Summary ===");
        println!("Strategy: {:?}", self.strategy);
        println!("Number of threads: {}", self.num_threads);
        println!("NUMA nodes: {}", self.topology.num_nodes);

        for (node_id, cores) in self.topology.cpus_per_node.iter().enumerate() {
            println!("  Node {node_id}: CPUs {cores:?}");
        }

        println!("Thread assignments:");
        for (thread_id, affinity) in self.assignments.iter().enumerate() {
            println!(
                "  Thread {}: cores {:?}, migration: {}, NUMA: {:?}",
                thread_id, affinity.core_ids, affinity.allow_migration, affinity.numa_node
            );
        }
        println!("==============================");
    }

    /// Get affinity efficiency metrics
    pub fn efficiency_metrics(&self) -> AffinityEfficiencyMetrics {
        let cores_used: std::collections::HashSet<usize> = self
            .assignments
            .iter()
            .flat_map(|affinity| affinity.core_ids.iter().cloned())
            .collect();

        let total_cores: usize = self
            .topology
            .cpus_per_node
            .iter()
            .map(|node| node.len())
            .sum();

        let numa_nodes_used: std::collections::HashSet<usize> = self
            .assignments
            .iter()
            .filter_map(|affinity| affinity.numa_node)
            .collect();

        let threads_with_migration: usize = self
            .assignments
            .iter()
            .filter(|affinity| affinity.allow_migration)
            .count();

        AffinityEfficiencyMetrics {
            core_utilization: cores_used.len() as f64 / total_cores as f64,
            numa_spread: numa_nodes_used.len() as f64 / self.topology.num_nodes as f64,
            migration_ratio: threads_with_migration as f64 / self.num_threads as f64,
            threads_per_core: self.num_threads as f64 / cores_used.len() as f64,
        }
    }
}

/// Metrics for evaluating affinity efficiency
#[derive(Debug, Clone)]
pub struct AffinityEfficiencyMetrics {
    /// Fraction of CPU cores being used (0.0 to 1.0)
    pub core_utilization: f64,
    /// Fraction of NUMA nodes being used (0.0 to 1.0)
    pub numa_spread: f64,
    /// Fraction of threads that allow migration (0.0 to 1.0)
    pub migration_ratio: f64,
    /// Average number of threads per CPU core
    pub threads_per_core: f64,
}

/// Utility functions for affinity management
pub mod utils {
    use super::*;

    /// Auto-detect optimal affinity strategy for a workload
    pub fn auto_detect_strategy(
        workload_type: WorkloadType,
        num_threads: usize,
        topology: &NumaTopology,
    ) -> AffinityStrategy {
        let manager = AffinityManager::new(AffinityStrategy::None, topology.clone());
        manager.recommend_strategy(num_threads, workload_type)
    }

    /// Create optimized thread pool for matrix operations
    pub fn creatematrix_thread_pool(
        matrixsize: (usize, usize),
        topology: NumaTopology,
    ) -> AffinityThreadPool {
        let workload_type = if matrixsize.0 * matrixsize.1 > 1_000_000 {
            WorkloadType::MemoryBound
        } else {
            WorkloadType::CpuBound
        };

        let num_threads = std::cmp::min(
            topology.cpus_per_node.iter().map(|node| node.len()).sum(),
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        );

        let strategy = auto_detect_strategy(workload_type, num_threads, &topology);
        let config = WorkerConfig::new().with_workers(num_threads);

        AffinityThreadPool::new(strategy, topology, config)
    }

    /// Benchmark different affinity strategies
    pub fn benchmark_affinity_strategies<F>(
        workload: F,
        topology: NumaTopology,
        config: WorkerConfig,
    ) -> Vec<(AffinityStrategy, f64)>
    where
        F: Fn() -> f64 + Clone + Send + Sync,
    {
        let strategies = vec![
            AffinityStrategy::None,
            AffinityStrategy::Pinned,
            AffinityStrategy::NumaSpread,
            AffinityStrategy::NumaCompact,
        ];

        let mut results = Vec::new();

        for strategy in strategies {
            let pool = AffinityThreadPool::new(strategy, topology.clone(), config.clone());

            // Warm up
            for _ in 0..3 {
                pool.execute_with_affinity(&workload);
            }

            // Benchmark
            let start = std::time::Instant::now();
            let iterations = 10;
            let mut total_work = 0.0;

            for _ in 0..iterations {
                total_work += pool.execute_with_affinity(&workload);
            }

            let elapsed = start.elapsed().as_secs_f64();
            let throughput = total_work / elapsed;

            results.push((strategy, throughput));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_affinity_creation() {
        let affinity = CoreAffinity::cores(vec![0, 1, 2]);
        assert_eq!(affinity.core_ids, vec![0, 1, 2]);
        assert!(!affinity.allow_migration);
        assert_eq!(affinity.numa_node, None);
    }

    #[test]
    fn test_numa_affinity_creation() {
        let topology = NumaTopology {
            num_nodes: 2,
            cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
        };

        let affinity = CoreAffinity::numa_node(1, &topology);
        assert_eq!(affinity.core_ids, vec![4, 5, 6, 7]);
        assert!(affinity.allow_migration);
        assert_eq!(affinity.numa_node, Some(1));
    }

    #[test]
    fn test_affinity_strategy_recommendation() {
        let topology = NumaTopology {
            num_nodes: 2,
            cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
        };

        let manager = AffinityManager::new(AffinityStrategy::None, topology);

        // CPU-bound with few threads should be pinned
        assert!(matches!(
            manager.recommend_strategy(4, WorkloadType::CpuBound),
            AffinityStrategy::Pinned
        ));

        // Memory-bound should prefer NUMA compact
        assert!(matches!(
            manager.recommend_strategy(4, WorkloadType::MemoryBound),
            AffinityStrategy::NumaCompact
        ));
    }

    #[test]
    fn test_pinned_assignments() {
        let topology = NumaTopology {
            num_nodes: 2,
            cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
        };

        let manager = AffinityManager::new(AffinityStrategy::Pinned, topology);
        let assignments = manager.generate_assignments(4);

        assert_eq!(assignments.len(), 4);
        for (i, assignment) in assignments.iter().enumerate() {
            assert_eq!(assignment.core_ids.len(), 1);
            assert_eq!(assignment.core_ids[0], i);
            assert!(!assignment.allow_migration);
        }
    }

    #[test]
    fn test_numa_spread_assignments() {
        let topology = NumaTopology {
            num_nodes: 2,
            cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
        };

        let manager = AffinityManager::new(AffinityStrategy::NumaSpread, topology);
        let assignments = manager.generate_assignments(4);

        assert_eq!(assignments.len(), 4);

        // Should have 2 threads per NUMA node
        let node0_threads = assignments
            .iter()
            .filter(|a| {
                a.core_ids.contains(&0)
                    || a.core_ids.contains(&1)
                    || a.core_ids.contains(&2)
                    || a.core_ids.contains(&3)
            })
            .count();
        let node1_threads = assignments
            .iter()
            .filter(|a| {
                a.core_ids.contains(&4)
                    || a.core_ids.contains(&5)
                    || a.core_ids.contains(&6)
                    || a.core_ids.contains(&7)
            })
            .count();

        assert_eq!(node0_threads, 2);
        assert_eq!(node1_threads, 2);
    }
}
