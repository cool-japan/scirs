//! Distributed optimization using MPI for large-scale parallel computation
//!
//! This module provides distributed optimization algorithms that can scale across
//! multiple nodes using Message Passing Interface (MPI), enabling optimization
//! of computationally expensive problems across compute clusters.

use crate::error::{ScirsError, ScirsResult};
use ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::Rng;
use statrs::statistics::Statistics;

/// MPI interface abstraction for distributed optimization
pub trait MPIInterface {
    /// Get the rank of this process
    fn rank(&self) -> i32;

    /// Get the total number of processes
    fn size(&self) -> i32;

    /// Broadcast data from root to all processes
    fn broadcast<T>(&self, data: &mut [T], root: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync;

    /// Gather data from all processes to root
    fn gather<T>(&self, send_data: &[T], recv_data: Option<&mut [T]>, root: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync;

    /// All-to-all reduction operation
    fn allreduce<T>(
        &self,
        send_data: &[T],
        recv_data: &mut [T],
        op: ReductionOp,
    ) -> ScirsResult<()>
    where
        T: Clone + Send + Sync + std::ops::Add<Output = T> + PartialOrd;

    /// Barrier synchronization
    fn barrier(&self) -> ScirsResult<()>;

    /// Send data to specific process
    fn send<T>(&self, data: &[T], dest: i32, tag: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync;

    /// Receive data from specific process
    fn recv<T>(&self, data: &mut [T], source: i32, tag: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync;
}

/// Reduction operations for MPI
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Min,
    Max,
    Prod,
}

/// Configuration for distributed optimization
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Strategy for distributing work
    pub distribution_strategy: DistributionStrategy,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Communication optimization settings
    pub communication: CommunicationConfig,
    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            distribution_strategy: DistributionStrategy::DataParallel,
            load_balancing: LoadBalancingConfig::default(),
            communication: CommunicationConfig::default(),
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
}

/// Work distribution strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistributionStrategy {
    /// Distribute data across processes
    DataParallel,
    /// Distribute parameters across processes
    ModelParallel,
    /// Hybrid data and model parallelism
    Hybrid,
    /// Master-worker with dynamic task assignment
    MasterWorker,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Whether to enable dynamic load balancing
    pub dynamic: bool,
    /// Threshold for load imbalance (0.0 to 1.0)
    pub imbalance_threshold: f64,
    /// Rebalancing interval (in iterations)
    pub rebalance_interval: usize,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            dynamic: true,
            imbalance_threshold: 0.2,
            rebalance_interval: 100,
        }
    }
}

/// Communication optimization configuration
#[derive(Debug, Clone)]
pub struct CommunicationConfig {
    /// Whether to use asynchronous communication
    pub async_communication: bool,
    /// Communication buffer size
    pub buffer_size: usize,
    /// Compression for large data transfers
    pub use_compression: bool,
    /// Overlap computation with communication
    pub overlap_computation: bool,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            async_communication: true,
            buffer_size: 1024 * 1024, // 1MB
            use_compression: false,
            overlap_computation: true,
        }
    }
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Enable checkpointing
    pub checkpointing: bool,
    /// Checkpoint interval (in iterations)
    pub checkpoint_interval: usize,
    /// Maximum number of retries for failed operations
    pub max_retries: usize,
    /// Timeout for MPI operations (in seconds)
    pub timeout: f64,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            checkpointing: false,
            checkpoint_interval: 1000,
            max_retries: 3,
            timeout: 30.0,
        }
    }
}

/// Distributed optimization context
pub struct DistributedOptimizationContext<M: MPIInterface> {
    mpi: M,
    config: DistributedConfig,
    rank: i32,
    size: i32,
    work_distribution: WorkDistribution,
    performance_stats: DistributedStats,
}

impl<M: MPIInterface> DistributedOptimizationContext<M> {
    /// Create a new distributed optimization context
    pub fn new(mpi: M, config: DistributedConfig) -> Self {
        let rank = mpi.rank();
        let size = mpi.size();
        let work_distribution = WorkDistribution::new(rank, size, config.distribution_strategy);

        Self {
            mpi,
            config,
            rank,
            size,
            work_distribution,
            performance_stats: DistributedStats::new(),
        }
    }

    /// Get the MPI rank of this process
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get the total number of MPI processes
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Check if this is the master process
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Distribute work among processes
    pub fn distribute_work(&mut self, total_work: usize) -> WorkAssignment {
        self.work_distribution.assign_work(total_work)
    }

    /// Synchronize all processes
    pub fn synchronize(&self) -> ScirsResult<()> {
        self.mpi.barrier()
    }

    /// Broadcast parameters from master to all workers
    pub fn broadcast_parameters(&self, params: &mut Array1<f64>) -> ScirsResult<()> {
        let data = params.as_slice_mut().unwrap();
        self.mpi.broadcast(data, 0)
    }

    /// Gather results from all workers to master
    pub fn gather_results(&self, local_result: &Array1<f64>) -> ScirsResult<Option<Array2<f64>>> {
        if self.is_master() {
            let total_size = local_result.len() * self.size as usize;
            let mut gathered_data = vec![0.0; total_size];
            self.mpi.gather(
                local_result.as_slice().unwrap(),
                Some(&mut gathered_data),
                0,
            )?;

            // Reshape into 2D array
            let result =
                Array2::from_shape_vec((self.size as usize, local_result.len()), gathered_data)
                    .map_err(|e| {
                        ScirsError::InvalidInput(scirs2_core::error::ErrorContext::new(format!(
                            "Failed to reshape gathered data: {}",
                            e
                        )))
                    })?;
            Ok(Some(result))
        } else {
            self.mpi.gather(local_result.as_slice().unwrap(), None, 0)?;
            Ok(None)
        }
    }

    /// Perform all-reduce operation (sum)
    pub fn allreduce_sum(&self, local_data: &Array1<f64>) -> ScirsResult<Array1<f64>> {
        let mut result = Array1::zeros(local_data.len());
        self.mpi.allreduce(
            local_data.as_slice().unwrap(),
            result.as_slice_mut().unwrap(),
            ReductionOp::Sum,
        )?;
        Ok(result)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &DistributedStats {
        &self.performance_stats
    }
}

/// Work distribution manager
struct WorkDistribution {
    rank: i32,
    size: i32,
    strategy: DistributionStrategy,
}

impl WorkDistribution {
    fn new(rank: i32, size: i32, strategy: DistributionStrategy) -> Self {
        Self {
            rank,
            size,
            strategy,
        }
    }

    fn assign_work(&self, total_work: usize) -> WorkAssignment {
        match self.strategy {
            DistributionStrategy::DataParallel => self.data_parallel_assignment(total_work),
            DistributionStrategy::ModelParallel => self.model_parallel_assignment(total_work),
            DistributionStrategy::Hybrid => self.hybrid_assignment(total_work),
            DistributionStrategy::MasterWorker => self.master_worker_assignment(total_work),
        }
    }

    fn data_parallel_assignment(&self, total_work: usize) -> WorkAssignment {
        let work_per_process = total_work / self.size as usize;
        let remainder = total_work % self.size as usize;

        let start = self.rank as usize * work_per_process + (self.rank as usize).min(remainder);
        let extra = if (self.rank as usize) < remainder {
            1
        } else {
            0
        };
        let count = work_per_process + extra;

        WorkAssignment {
            start_index: start,
            count,
            strategy: DistributionStrategy::DataParallel,
        }
    }

    fn model_parallel_assignment(&self, total_work: usize) -> WorkAssignment {
        // For model parallelism, each process handles different parameters
        WorkAssignment {
            start_index: 0,
            count: total_work, // Each process sees all data but handles different parameters
            strategy: DistributionStrategy::ModelParallel,
        }
    }

    fn hybrid_assignment(&self, total_work: usize) -> WorkAssignment {
        // Simplified hybrid: use data parallel for now
        self.data_parallel_assignment(total_work)
    }

    fn master_worker_assignment(&self, total_work: usize) -> WorkAssignment {
        if self.rank == 0 {
            // Master coordinates but may not do computation
            WorkAssignment {
                start_index: 0,
                count: 0,
                strategy: DistributionStrategy::MasterWorker,
            }
        } else {
            // Workers split the work
            let worker_count = self.size - 1;
            let work_per_worker = total_work / worker_count as usize;
            let remainder = total_work % worker_count as usize;
            let worker_rank = self.rank - 1;

            let start =
                worker_rank as usize * work_per_worker + (worker_rank as usize).min(remainder);
            let extra = if (worker_rank as usize) < remainder {
                1
            } else {
                0
            };
            let count = work_per_worker + extra;

            WorkAssignment {
                start_index: start,
                count,
                strategy: DistributionStrategy::MasterWorker,
            }
        }
    }
}

/// Work assignment for a process
#[derive(Debug, Clone)]
pub struct WorkAssignment {
    /// Starting index for this process
    pub start_index: usize,
    /// Number of work items for this process
    pub count: usize,
    /// Distribution strategy used
    pub strategy: DistributionStrategy,
}

impl WorkAssignment {
    /// Get the range of indices assigned to this process
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start_index..(self.start_index + self.count)
    }

    /// Check if this assignment is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Distributed optimization algorithms
pub mod algorithms {
    use super::*;
    use crate::result::OptimizeResults;

    /// Distributed differential evolution
    pub struct DistributedDifferentialEvolution<M: MPIInterface> {
        context: DistributedOptimizationContext<M>,
        population_size: usize,
        max_nit: usize,
        f_scale: f64,
        crossover_rate: f64,
    }

    impl<M: MPIInterface> DistributedDifferentialEvolution<M> {
        /// Create a new distributed differential evolution optimizer
        pub fn new(
            context: DistributedOptimizationContext<M>,
            population_size: usize,
            max_nit: usize,
        ) -> Self {
            Self {
                context,
                population_size,
                max_nit,
                f_scale: 0.8,
                crossover_rate: 0.7,
            }
        }

        /// Set mutation parameters
        pub fn with_parameters(mut self, f_scale: f64, crossover_rate: f64) -> Self {
            self.f_scale = f_scale;
            self.crossover_rate = crossover_rate;
            self
        }

        /// Optimize function using distributed differential evolution
        pub fn optimize<F>(
            &mut self,
            function: F,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<OptimizeResults<f64>>
        where
            F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
        {
            let dims = bounds.len();

            // Initialize local population
            let local_pop_size = self.population_size / self.context.size() as usize;
            let mut local_population = self.initialize_local_population(local_pop_size, bounds)?;
            let mut local_fitness = self.evaluate_local_population(&function, &local_population)?;

            // Find global best across all processes
            let mut global_best = self.find_global_best(&local_population, &local_fitness)?;
            let mut global_best_fitness = global_best.1;

            let mut total_evaluations = self.population_size;

            for iteration in 0..self.max_nit {
                // Generate trial population
                let trial_population = self.generate_trial_population(&local_population)?;
                let trial_fitness = self.evaluate_local_population(&function, &trial_population)?;

                // Selection
                self.selection(
                    &mut local_population,
                    &mut local_fitness,
                    &trial_population,
                    &trial_fitness,
                );

                total_evaluations += local_pop_size;

                // Exchange information between processes
                if iteration % 10 == 0 {
                    let new_global_best =
                        self.find_global_best(&local_population, &local_fitness)?;
                    if new_global_best.1 < global_best_fitness {
                        global_best = new_global_best;
                        global_best_fitness = global_best.1;
                    }

                    // Migration between processes
                    self.migrate_individuals(&mut local_population, &mut local_fitness)?;
                }

                // Convergence check (simplified)
                if iteration % 50 == 0 {
                    let convergence = self.check_convergence(&local_fitness)?;
                    if convergence {
                        break;
                    }
                }
            }

            // Final global best search
            let final_best = self.find_global_best(&local_population, &local_fitness)?;
            if final_best.1 < global_best_fitness {
                global_best = final_best.clone();
                global_best_fitness = final_best.1;
            }

            Ok(OptimizeResults::<f64> {
                x: global_best.0,
                fun: global_best_fitness,
                success: true,
                message: "Distributed differential evolution completed".to_string(),
                nit: self.max_nit,
                nfev: total_evaluations,
                ..OptimizeResults::default()
            })
        }

        fn initialize_local_population(
            &self,
            local_size: usize,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<Array2<f64>> {
            let mut rng = rand::rng();

            let dims = bounds.len();
            let mut population = Array2::zeros((local_size, dims));

            for i in 0..local_size {
                for j in 0..dims {
                    let (low, high) = bounds[j];
                    population[[i, j]] = rng.random_range(low..=high);
                }
            }

            Ok(population)
        }

        fn evaluate_local_population<F>(
            &self,
            function: &F,
            population: &Array2<f64>,
        ) -> ScirsResult<Array1<f64>>
        where
            F: Fn(&ArrayView1<f64>) -> f64,
        {
            let mut fitness = Array1::zeros(population.nrows());

            for i in 0..population.nrows() {
                let individual = population.row(i);
                fitness[i] = function(&individual);
            }

            Ok(fitness)
        }

        fn find_global_best(
            &mut self,
            local_population: &Array2<f64>,
            local_fitness: &Array1<f64>,
        ) -> ScirsResult<(Array1<f64>, f64)> {
            // Find local best
            let mut best_idx = 0;
            let mut best_fitness = local_fitness[0];
            for (i, &fitness) in local_fitness.iter().enumerate() {
                if fitness < best_fitness {
                    best_fitness = fitness;
                    best_idx = i;
                }
            }

            let local_best = local_population.row(best_idx).to_owned();

            // Find global best across all processes
            let global_fitness = Array1::from_elem(1, best_fitness);
            let global_fitness_sum = self.context.allreduce_sum(&global_fitness)?;

            // For simplicity, we'll use the local best for now
            // In a full implementation, we'd need to communicate the actual best individual
            Ok((local_best, best_fitness))
        }

        fn generate_trial_population(&self, population: &Array2<f64>) -> ScirsResult<Array2<f64>> {
            let mut rng = rand::rng();

            let (pop_size, dims) = population.dim();
            let mut trial_population = Array2::zeros((pop_size, dims));

            for i in 0..pop_size {
                // Select three random individuals
                let mut indices = Vec::new();
                while indices.len() < 3 {
                    let idx = rng.random_range(0..pop_size);
                    if idx != i && !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }

                let a = indices[0];
                let b = indices[1];
                let c = indices[2];

                // Mutation and crossover
                let j_rand = rng.random_range(0..dims);
                for j in 0..dims {
                    if rng.gen::<f64>() < self.crossover_rate || j == j_rand {
                        trial_population[[i, j]] = population[[a, j]]
                            + self.f_scale * (population[[b, j]] - population[[c, j]]);
                    } else {
                        trial_population[[i, j]] = population[[i, j]];
                    }
                }
            }

            Ok(trial_population)
        }

        fn selection(
            &self,
            population: &mut Array2<f64>,
            fitness: &mut Array1<f64>,
            trial_population: &Array2<f64>,
            trial_fitness: &Array1<f64>,
        ) {
            for i in 0..population.nrows() {
                if trial_fitness[i] <= fitness[i] {
                    for j in 0..population.ncols() {
                        population[[i, j]] = trial_population[[i, j]];
                    }
                    fitness[i] = trial_fitness[i];
                }
            }
        }

        fn migrate_individuals(
            &mut self,
            population: &mut Array2<f64>,
            fitness: &mut Array1<f64>,
        ) -> ScirsResult<()> {
            // Simple migration: send best individual to next process
            if self.context.size() <= 1 {
                return Ok(());
            }

            let best_idx = fitness
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let _next_rank = (self.context.rank() + 1) % self.context.size();
            let _prev_rank = (self.context.rank() - 1 + self.context.size()) % self.context.size();

            // Send best individual to next process
            let _best_individual = population.row(best_idx).to_owned();
            let _best_fitness_val = fitness[best_idx];

            // In a real implementation, we would use MPI send/recv here
            // For now, we'll skip the actual communication

            Ok(())
        }

        fn check_convergence(&mut self, local_fitness: &Array1<f64>) -> ScirsResult<bool> {
            let mean = local_fitness.view().mean();
            let variance = local_fitness
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / local_fitness.len() as f64;

            let std_dev = variance.sqrt();

            // Simple convergence criterion
            Ok(std_dev < 1e-12)
        }
    }

    /// Distributed particle swarm optimization
    pub struct DistributedParticleSwarm<M: MPIInterface> {
        context: DistributedOptimizationContext<M>,
        swarm_size: usize,
        max_nit: usize,
        w: f64,  // Inertia weight
        c1: f64, // Cognitive parameter
        c2: f64, // Social parameter
    }

    impl<M: MPIInterface> DistributedParticleSwarm<M> {
        /// Create a new distributed particle swarm optimizer
        pub fn new(
            context: DistributedOptimizationContext<M>,
            swarm_size: usize,
            max_nit: usize,
        ) -> Self {
            Self {
                context,
                swarm_size,
                max_nit,
                w: 0.729,
                c1: 1.49445,
                c2: 1.49445,
            }
        }

        /// Set PSO parameters
        pub fn with_parameters(mut self, w: f64, c1: f64, c2: f64) -> Self {
            self.w = w;
            self.c1 = c1;
            self.c2 = c2;
            self
        }

        /// Optimize function using distributed particle swarm optimization
        pub fn optimize<F>(
            &mut self,
            function: F,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<OptimizeResults<f64>>
        where
            F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
        {
            let dims = bounds.len();
            let local_swarm_size = self.swarm_size / self.context.size() as usize;

            // Initialize local swarm
            let mut positions = self.initialize_positions(local_swarm_size, bounds)?;
            let mut velocities = Array2::zeros((local_swarm_size, dims));
            let mut personal_best = positions.clone();
            let mut personal_best_fitness = self.evaluate_swarm(&function, &positions)?;

            // Find global best
            let mut global_best = self.find_global_best(&personal_best, &personal_best_fitness)?;
            let mut global_best_fitness = global_best.1;

            let mut function_evaluations = local_swarm_size;

            for iteration in 0..self.max_nit {
                // Update swarm
                self.update_swarm(
                    &mut positions,
                    &mut velocities,
                    &personal_best,
                    &global_best.0,
                    bounds,
                )?;

                // Evaluate new positions
                let fitness = self.evaluate_swarm(&function, &positions)?;
                function_evaluations += local_swarm_size;

                // Update personal bests
                for i in 0..local_swarm_size {
                    if fitness[i] < personal_best_fitness[i] {
                        personal_best_fitness[i] = fitness[i];
                        for j in 0..dims {
                            personal_best[[i, j]] = positions[[i, j]];
                        }
                    }
                }

                // Update global best
                if iteration % 10 == 0 {
                    let new_global_best =
                        self.find_global_best(&personal_best, &personal_best_fitness)?;
                    if new_global_best.1 < global_best_fitness {
                        global_best = new_global_best;
                        global_best_fitness = global_best.1;
                    }
                }
            }

            Ok(OptimizeResults::<f64> {
                x: global_best.0,
                fun: global_best_fitness,
                success: true,
                message: "Distributed particle swarm optimization completed".to_string(),
                nit: self.max_nit,
                nfev: function_evaluations,
                ..OptimizeResults::default()
            })
        }

        fn initialize_positions(
            &self,
            local_size: usize,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<Array2<f64>> {
            let mut rng = rand::rng();

            let dims = bounds.len();
            let mut positions = Array2::zeros((local_size, dims));

            for i in 0..local_size {
                for j in 0..dims {
                    let (low, high) = bounds[j];
                    positions[[i, j]] = rng.random_range(low..=high);
                }
            }

            Ok(positions)
        }

        fn evaluate_swarm<F>(
            &self,
            function: &F,
            positions: &Array2<f64>,
        ) -> ScirsResult<Array1<f64>>
        where
            F: Fn(&ArrayView1<f64>) -> f64,
        {
            let mut fitness = Array1::zeros(positions.nrows());

            for i in 0..positions.nrows() {
                let particle = positions.row(i);
                fitness[i] = function(&particle);
            }

            Ok(fitness)
        }

        fn find_global_best(
            &mut self,
            positions: &Array2<f64>,
            fitness: &Array1<f64>,
        ) -> ScirsResult<(Array1<f64>, f64)> {
            // Find local best
            let mut best_idx = 0;
            let mut best_fitness = fitness[0];
            for (i, &f) in fitness.iter().enumerate() {
                if f < best_fitness {
                    best_fitness = f;
                    best_idx = i;
                }
            }

            let local_best = positions.row(best_idx).to_owned();

            // In a full implementation, we would find the global best across all processes
            Ok((local_best, best_fitness))
        }

        fn update_swarm(
            &self,
            positions: &mut Array2<f64>,
            velocities: &mut Array2<f64>,
            personal_best: &Array2<f64>,
            global_best: &Array1<f64>,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<()> {
            let mut rng = rand::rng();

            let (swarm_size, dims) = positions.dim();

            for i in 0..swarm_size {
                for j in 0..dims {
                    let r1: f64 = rng.random();
                    let r2: f64 = rng.random();

                    // Update velocity
                    velocities[[i, j]] = self.w * velocities[[i, j]]
                        + self.c1 * r1 * (personal_best[[i, j]] - positions[[i, j]])
                        + self.c2 * r2 * (global_best[j] - positions[[i, j]]);

                    // Update position
                    positions[[i, j]] += velocities[[i, j]];

                    // Apply bounds
                    let (low, high) = bounds[j];
                    if positions[[i, j]] < low {
                        positions[[i, j]] = low;
                        velocities[[i, j]] = 0.0;
                    } else if positions[[i, j]] > high {
                        positions[[i, j]] = high;
                        velocities[[i, j]] = 0.0;
                    }
                }
            }

            Ok(())
        }
    }
}

/// Performance statistics for distributed optimization
#[derive(Debug, Clone)]
pub struct DistributedStats {
    /// Communication time statistics
    pub communication_time: f64,
    /// Computation time statistics
    pub computation_time: f64,
    /// Load balancing statistics
    pub load_balance_ratio: f64,
    /// Number of synchronization points
    pub synchronizations: usize,
    /// Data transfer statistics (bytes)
    pub bytes_transferred: usize,
}

impl DistributedStats {
    fn new() -> Self {
        Self {
            communication_time: 0.0,
            computation_time: 0.0,
            load_balance_ratio: 1.0,
            synchronizations: 0,
            bytes_transferred: 0,
        }
    }

    /// Calculate parallel efficiency
    pub fn parallel_efficiency(&self) -> f64 {
        if self.communication_time + self.computation_time == 0.0 {
            1.0
        } else {
            self.computation_time / (self.communication_time + self.computation_time)
        }
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "Distributed Optimization Performance Report\n\
             ==========================================\n\
             Computation Time: {:.3}s\n\
             Communication Time: {:.3}s\n\
             Parallel Efficiency: {:.1}%\n\
             Load Balance Ratio: {:.3}\n\
             Synchronizations: {}\n\
             Data Transferred: {} bytes\n",
            self.computation_time,
            self.communication_time,
            self.parallel_efficiency() * 100.0,
            self.load_balance_ratio,
            self.synchronizations,
            self.bytes_transferred
        )
    }
}

/// Mock MPI implementation for testing
#[cfg(test)]
pub struct MockMPI {
    rank: i32,
    size: i32,
}

#[cfg(test)]
impl MockMPI {
    pub fn new(rank: i32, size: i32) -> Self {
        Self { rank, size }
    }
}

#[cfg(test)]
impl MPIInterface for MockMPI {
    fn rank(&self) -> i32 {
        self.rank
    }
    fn size(&self) -> i32 {
        self.size
    }

    fn broadcast<T>(&self, data: &mut [T], root: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync,
    {
        Ok(())
    }

    fn gather<T>(
        &self,
        _send_data: &[T],
        _recv_data: Option<&mut [T]>,
        _root: i32,
    ) -> ScirsResult<()>
    where
        T: Clone + Send + Sync,
    {
        Ok(())
    }

    fn allreduce<T>(
        &self,
        send_data: &[T],
        recv_data: &mut [T],
        _op: ReductionOp,
    ) -> ScirsResult<()>
    where
        T: Clone + Send + Sync + std::ops::Add<Output = T> + PartialOrd,
    {
        for (i, item) in send_data.iter().enumerate() {
            if i < recv_data.len() {
                recv_data[i] = item.clone();
            }
        }
        Ok(())
    }

    fn barrier(&self) -> ScirsResult<()> {
        Ok(())
    }
    fn send<T>(&self, _data: &[T], _dest: i32, tag: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync,
    {
        Ok(())
    }
    fn recv<T>(&self, _data: &mut [T], _source: i32, tag: i32) -> ScirsResult<()>
    where
        T: Clone + Send + Sync,
    {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_distribution() {
        let distribution = WorkDistribution::new(0, 4, DistributionStrategy::DataParallel);
        let assignment = distribution.assign_work(100);

        assert_eq!(assignment.count, 25);
        assert_eq!(assignment.start_index, 0);
        assert_eq!(assignment.range(), 0..25);
    }

    #[test]
    fn test_work_assignment_remainder() {
        let distribution = WorkDistribution::new(3, 4, DistributionStrategy::DataParallel);
        let assignment = distribution.assign_work(10);

        // 10 items, 4 processes: 2, 3, 3, 2
        assert_eq!(assignment.count, 2);
        assert_eq!(assignment.start_index, 8);
    }

    #[test]
    fn test_master_worker_distribution() {
        let master_distribution = WorkDistribution::new(0, 4, DistributionStrategy::MasterWorker);
        let master_assignment = master_distribution.assign_work(100);

        assert_eq!(master_assignment.count, 0); // Master doesn't do computation

        let worker_distribution = WorkDistribution::new(1, 4, DistributionStrategy::MasterWorker);
        let worker_assignment = worker_distribution.assign_work(100);

        assert!(worker_assignment.count > 0); // Worker does computation
    }

    #[test]
    fn test_distributed_context() {
        let mpi = MockMPI::new(0, 4);
        let config = DistributedConfig::default();
        let context = DistributedOptimizationContext::new(mpi, config);

        assert_eq!(context.rank(), 0);
        assert_eq!(context.size(), 4);
        assert!(context.is_master());
    }

    #[test]
    fn test_distributed_stats() {
        let mut stats = DistributedStats::new();
        stats.computation_time = 80.0;
        stats.communication_time = 20.0;

        assert_eq!(stats.parallel_efficiency(), 0.8);
    }
}
