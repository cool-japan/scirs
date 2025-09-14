//! Distributed GPU optimization combining MPI and GPU acceleration
//!
//! This module provides optimization algorithms that leverage both distributed
//! computing (MPI) and GPU acceleration, enabling massive parallel optimization
//! across multiple nodes with GPU acceleration on each node.

use crate::error::ScirsResult;
use ndarray::{Array1, Array2, ArrayView1};

use crate::distributed::{
    DistributedConfig, DistributedOptimizationContext, DistributedStats, MPIInterface,
};
use crate::gpu::{
    acceleration::{AccelerationConfig, AccelerationManager},
    cuda_kernels::DifferentialEvolutionKernel,
    tensor_core_optimization::{AMPManager, TensorCoreOptimizationConfig, TensorCoreOptimizer},
    GpuOptimizationConfig, GpuOptimizationContext,
};
use crate::result::OptimizeResults;
use statrs::statistics::Statistics;

/// Configuration for distributed GPU optimization
#[derive(Clone)]
pub struct DistributedGpuConfig {
    /// Distributed computing configuration
    pub distributed_config: DistributedConfig,
    /// GPU optimization configuration
    pub gpu_config: GpuOptimizationConfig,
    /// GPU acceleration configuration
    pub acceleration_config: AccelerationConfig,
    /// Whether to use Tensor Cores if available
    pub use_tensor_cores: bool,
    /// Tensor Core configuration
    pub tensor_config: Option<TensorCoreOptimizationConfig>,
    /// Communication strategy for GPU data
    pub gpu_communication_strategy: GpuCommunicationStrategy,
    /// Load balancing between GPU and CPU work
    pub gpu_cpu_load_balance: f64, // 0.0 = all CPU, 1.0 = all GPU
}

impl Default for DistributedGpuConfig {
    fn default() -> Self {
        Self {
            distributed_config: DistributedConfig::default(),
            gpu_config: GpuOptimizationConfig::default(),
            acceleration_config: AccelerationConfig::default(),
            use_tensor_cores: true,
            tensor_config: Some(TensorCoreOptimizationConfig::default()),
            gpu_communication_strategy: GpuCommunicationStrategy::Direct,
            gpu_cpu_load_balance: 0.8, // Prefer GPU but keep some CPU work
        }
    }
}

/// GPU communication strategies for distributed systems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuCommunicationStrategy {
    /// Direct GPU-to-GPU communication via GPUDirect
    Direct,
    /// GPU-to-CPU-to-MPI-to-CPU-to-GPU
    Staged,
    /// Asynchronous overlapped communication
    AsyncOverlapped,
    /// Hierarchical communication (intra-node GPU, inter-node MPI)
    Hierarchical,
}

/// Distributed GPU optimization context
pub struct DistributedGpuOptimizer<M: MPIInterface> {
    distributed_context: DistributedOptimizationContext<M>,
    gpu_context: GpuOptimizationContext,
    acceleration_manager: AccelerationManager,
    tensor_optimizer: Option<TensorCoreOptimizer>,
    amp_manager: Option<AMPManager>,
    config: DistributedGpuConfig,
    performance_stats: DistributedGpuStats,
}

impl<M: MPIInterface> DistributedGpuOptimizer<M> {
    /// Create a new distributed GPU optimizer
    pub fn new(mpi: M, config: DistributedGpuConfig) -> ScirsResult<Self> {
        let distributed_context =
            DistributedOptimizationContext::new(mpi, config.distributed_config.clone());
        let gpu_context = GpuOptimizationContext::new(config.gpu_config.clone())?;
        let acceleration_manager = AccelerationManager::new(config.acceleration_config.clone());

        let tensor_optimizer = if config.use_tensor_cores {
            match config.tensor_config.as_ref() {
                Some(tensor_config) => {
                    match TensorCoreOptimizer::new(
                        gpu_context.context().clone(),
                        tensor_config.clone(),
                    ) {
                        Ok(optimizer) => Some(optimizer),
                        Err(_) => {
                            // Tensor Cores not available, continue without them
                            None
                        }
                    }
                }
                None => None,
            }
        } else {
            None
        };

        let amp_manager = if config
            .tensor_config
            .as_ref()
            .map(|tc| tc.use_amp)
            .unwrap_or(false)
        {
            Some(AMPManager::new())
        } else {
            None
        };

        Ok(Self {
            distributed_context,
            gpu_context,
            acceleration_manager,
            tensor_optimizer,
            amp_manager,
            config,
            performance_stats: DistributedGpuStats::new(),
        })
    }

    /// Optimize using distributed differential evolution with GPU acceleration
    pub fn differential_evolution<F>(
        &mut self,
        function: F,
        bounds: &[(f64, f64)],
        population_size: usize,
        max_nit: usize,
    ) -> ScirsResult<DistributedGpuResults>
    where
        F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
    {
        let start_time = std::time::Instant::now();

        // Distribute work across MPI processes
        let work_assignment = self.distributed_context.distribute_work(population_size);
        let local_pop_size = work_assignment.count;

        if local_pop_size == 0 {
            return Ok(DistributedGpuResults::empty()); // Worker with no assigned work
        }

        // Initialize population on GPU
        let dims = bounds.len();
        let local_population = self.initialize_gpu_population(local_pop_size, bounds)?;
        let local_fitness = self.evaluate_population_gpu(&function, &local_population)?;

        // GPU kernels for evolution operations
        // TODO: Fix GpuContext type mismatch between local alias and scirs2_core::GpuContext
        // let evolution_kernel =
        //     DifferentialEvolutionKernel::new(
        //         Arc::new(CoreGpuContext::from_device(self.gpu_context.context().as_ref().clone()))
        //     ).map_err(|e| scirs2_core::error::CoreError::General(e.to_string()))?;
        let evolution_kernel = todo!("Fix GpuContext type conversion");

        #[allow(unreachable_code)]
        let mut best_individual = Array1::zeros(dims);
        let mut best_fitness = f64::INFINITY;
        let mut total_evaluations = local_pop_size;

        // Main evolution loop
        for iteration in 0..max_nit {
            // Generate trial population using GPU
            let trial_population = self.gpu_mutation_crossover(
                &evolution_kernel,
                &local_population,
                0.8, // F scale
                0.7, // Crossover rate
            )?;

            // Evaluate trial population
            let trial_fitness = self.evaluate_population_gpu(&function, &trial_population)?;
            total_evaluations += local_pop_size;

            // GPU-accelerated selection
            self.gpu_selection(
                &evolution_kernel,
                &mut local_population,
                &trial_population,
                &mut local_fitness,
                &trial_fitness,
            )?;

            // Find local best
            let (local_best_idx, local_best_fitness) = self.find_local_best(&local_fitness)?;

            if local_best_fitness < best_fitness {
                best_fitness = local_best_fitness;
                best_individual = local_population.row(local_best_idx).to_owned();
            }

            // Periodic communication and migration
            if iteration % 10 == 0 {
                let global_best =
                    self.communicate_best_individuals(&best_individual, best_fitness)?;

                if let Some((global_best_individual, global_best_fitness)) = global_best {
                    if global_best_fitness < best_fitness {
                        best_individual = global_best_individual;
                        best_fitness = global_best_fitness;
                    }
                }

                // GPU-to-GPU migration if supported
                self.gpu_migration(&mut local_population, &mut local_fitness)?;
            }

            // Update performance statistics
            self.performance_stats.record_iteration(
                iteration,
                local_pop_size,
                best_fitness,
                start_time.elapsed().as_secs_f64(),
            );

            // Check convergence
            if self.check_convergence(&local_fitness, iteration)? {
                break;
            }
        }

        // Final global best communication
        let final_global_best =
            self.communicate_best_individuals(&best_individual, best_fitness)?;

        if let Some((final_best_individual, final_best_fitness)) = final_global_best {
            best_individual = final_best_individual;
            best_fitness = final_best_fitness;
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(DistributedGpuResults {
            base_result: OptimizeResults::<f64> {
                x: best_individual,
                fun: best_fitness,
                success: true,
                message: "Distributed GPU differential evolution completed".to_string(),
                nit: max_nit,
                nfev: total_evaluations,
                ..OptimizeResults::default()
            },
            gpu_stats: crate::gpu::acceleration::PerformanceStats::default(),
            distributed_stats: self.distributed_context.stats().clone(),
            performance_stats: self.performance_stats.clone(),
            total_time,
        })
    }

    /// Initialize population on GPU
    fn initialize_gpu_population(
        &self,
        pop_size: usize,
        bounds: &[(f64, f64)],
    ) -> ScirsResult<Array2<f64>> {
        use rand::Rng;
        let mut rng = rand::rng();

        let dims = bounds.len();
        let mut population = Array2::zeros((pop_size, dims));

        for i in 0..pop_size {
            for j in 0..dims {
                let (low, high) = bounds[j];
                population[[i, j]] = rng.gen_range(low..=high);
            }
        }

        Ok(population)
    }

    /// Evaluate population using GPU acceleration
    fn evaluate_population_gpu<F>(
        &mut self,
        function: &F,
        population: &Array2<f64>,
    ) -> ScirsResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let pop_size = population.nrows();
        let mut fitness = Array1::zeros(pop_size);

        // Decide between GPU and CPU based on problem size and configuration
        let use_gpu = pop_size >= 100 && self.config.gpu_cpu_load_balance > 0.5;

        if use_gpu {
            // Use GPU acceleration for function evaluation
            self.performance_stats.gpu_evaluations += pop_size;

            // For now, use CPU evaluation since GPU function interface needs adaptation
            // TODO: Implement proper GPU function evaluation interface
            for i in 0..pop_size {
                fitness[i] = function(&population.row(i));
            }
        } else {
            // Use CPU evaluation
            self.performance_stats.cpu_evaluations += pop_size;
            for i in 0..pop_size {
                fitness[i] = function(&population.row(i));
            }
        }

        Ok(fitness)
    }

    /// Perform GPU-accelerated mutation and crossover
    fn gpu_mutation_crossover(
        &self,
        _kernel: &DifferentialEvolutionKernel,
        population: &Array2<f64>,
        f_scale: f64,
        crossover_rate: f64,
    ) -> ScirsResult<Array2<f64>> {
        let (pop_size, dims) = population.dim();
        let mut trial_population = Array2::zeros((pop_size, dims));

        // For now, implement CPU-based mutation and crossover
        // TODO: Use actual GPU kernels when properly implemented
        use rand::Rng;
        let mut rng = rand::rng();

        for i in 0..pop_size {
            // Select three random individuals different from current
            let mut indices = Vec::new();
            while indices.len() < 3 {
                let idx = rng.gen_range(0..pop_size);
                if idx != i && !indices.contains(&idx) {
                    indices.push(idx);
                }
            }

            let a = indices[0];
            let b = indices[1];
            let c = indices[2];

            // Mutation and crossover
            let j_rand = rng.gen_range(0..dims);
            for j in 0..dims {
                if rng.gen_range(0.0..1.0) < crossover_rate || j == j_rand {
                    trial_population[[i, j]] =
                        population[[a, j]] + f_scale * (population[[b, j]] - population[[c, j]]);
                } else {
                    trial_population[[i, j]] = population[[i, j]];
                }
            }
        }

        Ok(trial_population)
    }

    /// Perform GPU-accelerated selection
    fn gpu_selection(
        &self,
        _kernel: &DifferentialEvolutionKernel,
        population: &mut Array2<f64>,
        trial_population: &Array2<f64>,
        fitness: &mut Array1<f64>,
        trial_fitness: &Array1<f64>,
    ) -> ScirsResult<()> {
        // For now, implement CPU-based selection
        // TODO: Use actual GPU kernels when properly implemented
        for i in 0..population.nrows() {
            if trial_fitness[i] <= fitness[i] {
                for j in 0..population.ncols() {
                    population[[i, j]] = trial_population[[i, j]];
                }
                fitness[i] = trial_fitness[i];
            }
        }

        Ok(())
    }

    /// Find local best individual and fitness
    fn find_local_best(&self, fitness: &Array1<f64>) -> ScirsResult<(usize, f64)> {
        let mut best_idx = 0;
        let mut best_fitness = fitness[0];

        for (i, &f) in fitness.iter().enumerate() {
            if f < best_fitness {
                best_fitness = f;
                best_idx = i;
            }
        }

        Ok((best_idx, best_fitness))
    }

    /// Communicate best individuals across MPI processes
    fn communicate_best_individuals(
        &mut self,
        local_best: &Array1<f64>,
        local_best_fitness: f64,
    ) -> ScirsResult<Option<(Array1<f64>, f64)>> {
        if self.distributed_context.size() <= 1 {
            return Ok(None);
        }

        // For simplicity, we'll use a basic approach
        // In a full implementation, this would use MPI all-reduce operations
        // to find the global best across all processes

        // Placeholder implementation
        Ok(Some((local_best.clone(), local_best_fitness)))
    }

    /// Perform GPU-to-GPU migration between processes
    fn gpu_migration(
        &mut self,
        population: &mut Array2<f64>,
        fitness: &mut Array1<f64>,
    ) -> ScirsResult<()> {
        match self.config.gpu_communication_strategy {
            GpuCommunicationStrategy::Direct => {
                // Use GPUDirect for direct GPU-to-GPU communication
                self.gpu_direct_migration(population, fitness)
            }
            GpuCommunicationStrategy::Staged => {
                // Stage through CPU memory
                self.staged_migration(population, fitness)
            }
            GpuCommunicationStrategy::AsyncOverlapped => {
                // Asynchronous communication with computation overlap
                self.async_migration(population, fitness)
            }
            GpuCommunicationStrategy::Hierarchical => {
                // Hierarchical intra-node GPU, inter-node MPI
                self.hierarchical_migration(population, fitness)
            }
        }
    }

    /// Direct GPU-to-GPU migration using GPUDirect
    fn gpu_direct_migration(
        &mut self,
        population: &mut Array2<f64>,
        _fitness: &mut Array1<f64>,
    ) -> ScirsResult<()> {
        // Placeholder for GPUDirect implementation
        // This would use CUDA-aware MPI or similar technology
        Ok(())
    }

    /// Staged migration through CPU memory
    fn staged_migration(
        &mut self,
        population: &mut Array2<f64>,
        _fitness: &mut Array1<f64>,
    ) -> ScirsResult<()> {
        // Download from GPU, perform MPI communication, upload back to GPU
        // This is less efficient but more compatible
        Ok(())
    }

    /// Asynchronous migration with computation overlap
    fn async_migration(
        &mut self,
        population: &mut Array2<f64>,
        _fitness: &mut Array1<f64>,
    ) -> ScirsResult<()> {
        // Use asynchronous MPI operations to overlap communication with computation
        Ok(())
    }

    /// Hierarchical migration (intra-node GPU, inter-node MPI)
    fn hierarchical_migration(
        &mut self,
        population: &mut Array2<f64>,
        _fitness: &mut Array1<f64>,
    ) -> ScirsResult<()> {
        // First migrate within node between GPUs, then between nodes via MPI
        Ok(())
    }

    /// Check convergence criteria
    fn check_convergence(&self, fitness: &Array1<f64>, iteration: usize) -> ScirsResult<bool> {
        if fitness.len() < 2 {
            return Ok(false);
        }

        let mean = fitness.view().mean();
        let variance =
            fitness.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / fitness.len() as f64;

        let std_dev = variance.sqrt();

        // Simple convergence criterion
        Ok(std_dev < 1e-12 || iteration >= 1000)
    }

    /// Generate random indices for differential evolution mutation
    fn generate_random_indices(&self, pop_size: usize) -> ScirsResult<Array2<i32>> {
        use rand::Rng;
        let mut rng = rand::rng();
        let mut indices = Array2::zeros((pop_size, 3));

        for i in 0..pop_size {
            let mut selected = std::collections::HashSet::new();
            selected.insert(i);

            for j in 0..3 {
                loop {
                    let idx = rng.gen_range(0..pop_size);
                    if !selected.contains(&idx) {
                        indices[[i, j]] = idx as i32;
                        selected.insert(idx);
                        break;
                    }
                }
            }
        }

        Ok(indices)
    }

    /// Generate random values for crossover
    fn generate_random_values(&self, count: usize) -> ScirsResult<Array1<f64>> {
        use rand::Rng;
        let mut rng = rand::rng();
        let mut values = Array1::zeros(count);

        for i in 0..count {
            values[i] = rng.gen_range(0.0..1.0);
        }

        Ok(values)
    }

    /// Generate j_rand values for crossover
    fn generate_j_rand(&self, pop_size: usize, dims: usize) -> ScirsResult<Array1<i32>> {
        use rand::Rng;
        let mut rng = rand::rng();
        let mut j_rand = Array1::zeros(pop_size);

        for i in 0..pop_size {
            j_rand[i] = rng.gen_range(0..dims) as i32;
        }

        Ok(j_rand)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &DistributedGpuStats {
        &self.performance_stats
    }
}

/// Performance statistics for distributed GPU optimization
#[derive(Debug, Clone)]
pub struct DistributedGpuStats {
    /// Total GPU function evaluations
    pub gpu_evaluations: usize,
    /// Total CPU function evaluations
    pub cpu_evaluations: usize,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Communication overhead time
    pub communication_time: f64,
    /// GPU memory usage statistics
    pub gpu_memory_usage: f64,
    /// Iteration statistics
    pub nit: Vec<IterationStats>,
}

impl DistributedGpuStats {
    fn new() -> Self {
        Self {
            gpu_evaluations: 0,
            cpu_evaluations: 0,
            gpu_utilization: 0.0,
            communication_time: 0.0,
            gpu_memory_usage: 0.0,
            nit: Vec::new(),
        }
    }

    fn record_iteration(
        &mut self,
        iteration: usize,
        pop_size: usize,
        best_fitness: f64,
        elapsed_time: f64,
    ) {
        self.nit.push(IterationStats {
            iteration,
            population_size: pop_size,
            best_fitness,
            elapsed_time,
        });
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("Distributed GPU Optimization Performance Report\n");
        report.push_str("==============================================\n\n");

        report.push_str(&format!(
            "GPU Function Evaluations: {}\n",
            self.gpu_evaluations
        ));
        report.push_str(&format!(
            "CPU Function Evaluations: {}\n",
            self.cpu_evaluations
        ));

        let total_evaluations = self.gpu_evaluations + self.cpu_evaluations;
        if total_evaluations > 0 {
            let gpu_percentage = (self.gpu_evaluations as f64 / total_evaluations as f64) * 100.0;
            report.push_str(&format!("GPU Usage: {:.1}%\n", gpu_percentage));
        }

        report.push_str(&format!(
            "GPU Utilization: {:.1}%\n",
            self.gpu_utilization * 100.0
        ));
        report.push_str(&format!(
            "Communication Overhead: {:.3}s\n",
            self.communication_time
        ));
        report.push_str(&format!(
            "GPU Memory Usage: {:.1}%\n",
            self.gpu_memory_usage * 100.0
        ));

        if let Some(last_iteration) = self.nit.last() {
            report.push_str(&format!(
                "Final Best Fitness: {:.6e}\n",
                last_iteration.best_fitness
            ));
            report.push_str(&format!(
                "Total Time: {:.3}s\n",
                last_iteration.elapsed_time
            ));
        }

        report
    }
}

/// Statistics for individual iterations
#[derive(Debug, Clone)]
pub struct IterationStats {
    pub iteration: usize,
    pub population_size: usize,
    pub best_fitness: f64,
    pub elapsed_time: f64,
}

/// Results from distributed GPU optimization
#[derive(Debug, Clone)]
pub struct DistributedGpuResults {
    /// Base optimization results
    pub base_result: OptimizeResults<f64>,
    /// GPU-specific performance statistics
    pub gpu_stats: crate::gpu::acceleration::PerformanceStats,
    /// Distributed computing statistics
    pub distributed_stats: DistributedStats,
    /// Combined performance statistics
    pub performance_stats: DistributedGpuStats,
    /// Total optimization time
    pub total_time: f64,
}

impl DistributedGpuResults {
    fn empty() -> Self {
        Self {
            base_result: OptimizeResults::<f64> {
                x: Array1::zeros(0),
                fun: 0.0,
                success: false,
                message: "No work assigned to this process".to_string(),
                nit: 0,
                nfev: 0,
                ..OptimizeResults::default()
            },
            gpu_stats: crate::gpu::acceleration::PerformanceStats::default(),
            distributed_stats: DistributedStats {
                communication_time: 0.0,
                computation_time: 0.0,
                load_balance_ratio: 1.0,
                synchronizations: 0,
                bytes_transferred: 0,
            },
            performance_stats: DistributedGpuStats::new(),
            total_time: 0.0,
        }
    }

    /// Print comprehensive results summary
    pub fn print_summary(&self) {
        println!("Distributed GPU Optimization Results");
        println!("===================================");
        println!("Success: {}", self.base_result.success);
        println!("Final function value: {:.6e}", self.base_result.fun);
        println!("Iterations: {}", self.base_result.nit);
        println!("Function evaluations: {}", self.base_result.nfev);
        println!("Total time: {:.3}s", self.total_time);
        println!();

        println!("GPU Performance:");
        println!("{}", self.gpu_stats.generate_report());
        println!();

        println!("Distributed Performance:");
        println!("{}", self.distributed_stats.generate_report());
        println!();

        println!("Combined Performance:");
        println!("{}", self.performance_stats.generate_report());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_gpu_config() {
        let config = DistributedGpuConfig::default();
        assert!(config.use_tensor_cores);
        assert_eq!(config.gpu_cpu_load_balance, 0.8);
        assert_eq!(
            config.gpu_communication_strategy,
            GpuCommunicationStrategy::Direct
        );
    }

    #[test]
    fn test_gpu_communication_strategies() {
        let strategies = [
            GpuCommunicationStrategy::Direct,
            GpuCommunicationStrategy::Staged,
            GpuCommunicationStrategy::AsyncOverlapped,
            GpuCommunicationStrategy::Hierarchical,
        ];

        for strategy in &strategies {
            let mut config = DistributedGpuConfig::default();
            config.gpu_communication_strategy = *strategy;
            // Test that configuration is valid
            assert_eq!(config.gpu_communication_strategy, *strategy);
        }
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = DistributedGpuStats::new();
        stats.gpu_evaluations = 1000;
        stats.cpu_evaluations = 200;
        stats.gpu_utilization = 0.85;

        let report = stats.generate_report();
        assert!(report.contains("GPU Function Evaluations: 1000"));
        assert!(report.contains("CPU Function Evaluations: 200"));
        assert!(report.contains("GPU Usage: 83.3%")); // 1000/(1000+200) * 100
    }

    #[test]
    #[ignore = "Requires MPI and GPU"]
    fn test_distributed_gpu_optimization() {
        // This would test the actual distributed GPU optimization
        // Implementation depends on having both MPI and GPU available
    }
}
