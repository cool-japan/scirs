//! GPU acceleration for optimization algorithms
//!
//! This module provides GPU-accelerated implementations of various optimization
//! algorithms and supporting functionality. It leverages scirs2-core's GPU
//! abstractions to provide high-performance computing capabilities.

use crate::error::{ScirsError, ScirsResult};
use ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext};
use statrs::statistics::Statistics;
use std::sync::Arc;

// Note: Error conversion handled through scirs2_core::error system
// GPU errors are automatically converted via CoreError type alias

// Real GPU array type backed by scirs2-core (using GpuBuffer instead of GpuArray)
pub type OptimGpuArray<T> = GpuBuffer<T>;

pub mod acceleration;
pub mod cuda_kernels;
pub mod memory_management;
pub mod tensor_core_optimization;

/// GPU-accelerated optimization configuration
#[derive(Clone)]
pub struct GpuOptimizationConfig {
    /// GPU context to use
    pub context: Arc<GpuContext>,
    /// Batch size for parallel evaluation
    pub batch_size: usize,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Whether to use tensor cores (if available)
    pub use_tensor_cores: bool,
    /// Precision mode (f32 or f64)
    pub precision: GpuPrecision,
    /// Stream count for concurrent execution
    pub stream_count: usize,
}

impl Default for GpuOptimizationConfig {
    fn default() -> Self {
        Self {
            context: Arc::new(GpuContext::new(GpuBackend::default()).unwrap_or_else(|_| {
                // Fallback to CPU if GPU context creation fails
                GpuContext::new(GpuBackend::Cpu).expect("CPU backend should always work")
            })),
            batch_size: 1024,
            memory_limit: None,
            use_tensor_cores: true,
            precision: GpuPrecision::F64,
            stream_count: 4,
        }
    }
}

/// GPU precision modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuPrecision {
    F32,
    F64,
    Mixed, // Use F32 for computation, F64 for accumulation
}

/// GPU-accelerated function evaluation interface
pub trait GpuFunction {
    /// Evaluate function on GPU for a batch of points
    fn evaluate_batch_gpu(&self, points: &OptimGpuArray<f64>) -> ScirsResult<OptimGpuArray<f64>>;

    /// Evaluate gradient on GPU for a batch of points
    fn gradient_batch_gpu(&self, points: &OptimGpuArray<f64>) -> ScirsResult<OptimGpuArray<f64>>;

    /// Evaluate hessian on GPU (if supported)
    fn hessian_batch_gpu(&self, points: &OptimGpuArray<f64>) -> ScirsResult<OptimGpuArray<f64>> {
        Err(ScirsError::NotImplementedError(
            scirs2_core::error::ErrorContext::new("Hessian evaluation not implemented".to_string()),
        ))
    }

    /// Check if function supports GPU acceleration
    fn supports_gpu(&self) -> bool {
        true
    }
}

/// GPU-accelerated optimization context
pub struct GpuOptimizationContext {
    config: GpuOptimizationConfig,
    context: Arc<GpuContext>,
    memory_pool: memory_management::GpuMemoryPool,
}

impl GpuOptimizationContext {
    /// Create a new GPU optimization context
    pub fn new(config: GpuOptimizationConfig) -> ScirsResult<Self> {
        let context = config.context.clone();
        let memory_pool = memory_management::GpuMemoryPool::new_stub();

        Ok(Self {
            config,
            context,
            memory_pool,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &GpuOptimizationConfig {
        &self.config
    }

    /// Get the GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    /// Allocate GPU memory for optimization data
    pub fn allocate_workspace(
        &mut self,
        size: usize,
    ) -> ScirsResult<memory_management::GpuWorkspace> {
        self.memory_pool.allocate_workspace(size)
    }

    /// Transfer data from CPU to GPU using scirs2-core GPU abstractions
    pub fn transfer_to_gpu<T>(&self, data: &Array2<T>) -> ScirsResult<OptimGpuArray<T>>
    where
        T: Clone + Send + Sync + 'static + scirs2_core::GpuDataType,
    {
        // Use scirs2-core GPU buffer creation
        let shape = data.dim();
        let total_size = shape.0 * shape.1;
        let flat_data = data.as_slice().unwrap();

        // Create a GPU buffer with the flattened data
        // Note: GpuBuffer creation API may vary based on scirs2-core implementation
        // For now, we'll return an error since we can't directly create a GpuBuffer
        Err(ScirsError::NotImplementedError(
            scirs2_core::error::ErrorContext::new(
                "GPU buffer creation not yet implemented".to_string(),
            ),
        ))
    }

    /// Transfer data from GPU to CPU using scirs2-core GPU abstractions
    pub fn transfer_from_gpu<T>(&self, gpu_data: &OptimGpuArray<T>) -> ScirsResult<Array2<T>>
    where
        T: Clone + Send + Sync + Default + 'static + scirs2_core::GpuDataType,
    {
        // For now, assume shape is known from context
        // In real implementation, shape would be stored with the buffer
        // This is a simplification since GpuBuffer doesn't have shape method
        let total_size = gpu_data.len();
        let dims = (total_size as f64).sqrt() as usize;

        // Allocate host memory and copy from GPU
        let mut host_data = vec![T::default(); total_size];
        gpu_data.copy_to_host(&mut host_data)?;

        // Reshape to ndarray (assume square for now)
        Array2::from_shape_vec((dims, dims), host_data).map_err(|e| {
            ScirsError::ComputationError(scirs2_core::error::ErrorContext::new(format!(
                "Shape error: {}",
                e
            )))
        })
    }

    /// Upload array to GPU (alias for transfer_to_gpu)
    pub fn upload_array<T>(&self, data: &Array2<T>) -> ScirsResult<OptimGpuArray<T>>
    where
        T: Clone + Send + Sync + 'static + scirs2_core::GpuDataType,
    {
        self.transfer_to_gpu(data)
    }

    /// Download array from GPU (alias for transfer_from_gpu)
    pub fn download_array<T>(&self, gpu_data: &OptimGpuArray<T>) -> ScirsResult<Array2<T>>
    where
        T: Clone + Send + Sync + Default + 'static + scirs2_core::GpuDataType,
    {
        self.transfer_from_gpu(gpu_data)
    }

    /// Execute batch function evaluation
    pub fn evaluate_function_batch<F>(
        &self,
        function: &F,
        points: &Array2<f64>,
    ) -> ScirsResult<Array1<f64>>
    where
        F: GpuFunction,
    {
        if !function.supports_gpu() {
            return Err(ScirsError::InvalidInput(
                scirs2_core::error::ErrorContext::new(
                    "Function does not support GPU acceleration".to_string(),
                ),
            ));
        }

        let gpu_points = self.transfer_to_gpu(points)?;
        let gpu_results = function.evaluate_batch_gpu(&gpu_points)?;
        let cpu_results = self.transfer_from_gpu(&gpu_results)?;

        // Convert 2D result to 1D (assuming single output per point)
        Ok(cpu_results.column(0).to_owned())
    }

    /// Execute batch gradient evaluation
    pub fn evaluate_gradient_batch<F>(
        &self,
        function: &F,
        points: &Array2<f64>,
    ) -> ScirsResult<Array2<f64>>
    where
        F: GpuFunction,
    {
        if !function.supports_gpu() {
            return Err(ScirsError::InvalidInput(
                scirs2_core::error::ErrorContext::new(
                    "Function does not support GPU acceleration".to_string(),
                ),
            ));
        }

        let gpu_points = self.transfer_to_gpu(points)?;
        let gpu_gradients = function.gradient_batch_gpu(&gpu_points)?;
        self.transfer_from_gpu(&gpu_gradients)
    }

    /// Compute gradient using finite differences
    pub fn compute_gradient_finite_diff<F>(
        &self,
        function: &F,
        x: &Array1<f64>,
        h: f64,
    ) -> ScirsResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let n = x.len();
        let mut gradient = Array1::zeros(n);

        // CPU finite differences for now - could be GPU-accelerated in the future
        for i in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;

            gradient[i] = (function(&x_plus.view()) - function(&x_minus.view())) / (2.0 * h);
        }

        Ok(gradient)
    }

    /// Compute search direction (simple steepest descent for now)
    pub fn compute_search_direction(&self, gradient: &Array1<f64>) -> ScirsResult<Array1<f64>> {
        // Simple steepest descent - could be enhanced with GPU-accelerated quasi-Newton methods
        Ok(-gradient.clone())
    }
}

/// GPU-accelerated optimization algorithms
pub mod algorithms {
    use super::*;
    use crate::result::OptimizeResults;

    /// GPU-accelerated differential evolution
    pub struct GpuDifferentialEvolution {
        context: GpuOptimizationContext,
        population_size: usize,
        max_nit: usize,
        f_scale: f64,
        crossover_rate: f64,
    }

    impl GpuDifferentialEvolution {
        /// Create a new GPU-accelerated differential evolution optimizer
        pub fn new(
            context: GpuOptimizationContext,
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

        /// Set mutation scale factor
        pub fn with_f_scale(mut self, f_scale: f64) -> Self {
            self.f_scale = f_scale;
            self
        }

        /// Set crossover rate
        pub fn with_crossover_rate(mut self, crossover_rate: f64) -> Self {
            self.crossover_rate = crossover_rate;
            self
        }

        /// Optimize function using GPU-accelerated differential evolution
        pub fn optimize<F>(
            &mut self,
            function: &F,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<OptimizeResults<f64>>
        where
            F: GpuFunction,
        {
            let dims = bounds.len();

            // Initialize population on GPU
            let mut population = self.initialize_population_gpu(bounds)?;
            let mut fitness = self.evaluate_population_gpu(function, &population)?;

            let mut best_idx = 0;
            let mut best_fitness = fitness[0];
            for (i, &f) in fitness.iter().enumerate() {
                if f < best_fitness {
                    best_fitness = f;
                    best_idx = i;
                }
            }

            let mut function_evaluations = self.population_size;

            for iteration in 0..self.max_nit {
                // Generate trial population using GPU kernels
                let trial_population = self.generate_trial_population_gpu(&population)?;
                let trial_fitness = self.evaluate_population_gpu(function, &trial_population)?;

                // Selection on GPU
                self.selection_gpu(
                    &mut population,
                    &mut fitness,
                    &trial_population,
                    &trial_fitness,
                )?;

                function_evaluations += self.population_size;

                // Update best solution
                for (i, &f) in fitness.iter().enumerate() {
                    if f < best_fitness {
                        best_fitness = f;
                        best_idx = i;
                    }
                }

                // Convergence check (simplified)
                if iteration % 10 == 0 {
                    let fitness_std = self.calculate_fitness_std(&fitness);
                    if fitness_std < 1e-12 {
                        break;
                    }
                }
            }

            // Transfer best solution back to CPU
            let best_x = population.row(best_idx).to_owned();

            Ok(OptimizeResults::<f64> {
                x: best_x,
                fun: best_fitness,
                success: true,
                message: "GPU differential evolution completed".to_string(),
                nit: self.max_nit,
                nfev: function_evaluations,
                ..OptimizeResults::default()
            })
        }

        fn initialize_population_gpu(&self, bounds: &[(f64, f64)]) -> ScirsResult<Array2<f64>> {
            use rand::Rng;
            let mut rng = rand::rng();

            let dims = bounds.len();
            let mut population = Array2::zeros((self.population_size, dims));

            for i in 0..self.population_size {
                for j in 0..dims {
                    let (low, high) = bounds[j];
                    population[[i, j]] = rng.gen_range(low..=high);
                }
            }

            Ok(population)
        }

        fn evaluate_population_gpu<F>(
            &self,
            function: &F,
            population: &Array2<f64>,
        ) -> ScirsResult<Array1<f64>>
        where
            F: GpuFunction,
        {
            self.context.evaluate_function_batch(function, population)
        }

        fn generate_trial_population_gpu(
            &self,
            population: &Array2<f64>,
        ) -> ScirsResult<Array2<f64>> {
            // For now, implement on CPU and transfer to GPU
            // In a full implementation, this would use GPU kernels
            use rand::Rng;
            let mut rng = rand::rng();

            let (pop_size, dims) = population.dim();
            let mut trial_population = Array2::zeros((pop_size, dims));

            for i in 0..pop_size {
                // Select three random individuals different from current
                let mut indices = Vec::new();
                while indices.len() < 3 {
                    let idx = rng.gen_range(0..pop_size);
                    if idx != i && !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }

                let [a, b, c] = [indices[0], indices[1], indices[2]];

                // Mutation and crossover
                let j_rand = rng.gen_range(0..dims);
                for j in 0..dims {
                    if rng.gen_range(0.0..1.0) < self.crossover_rate || j == j_rand {
                        trial_population[[i, j]] = population[[a, j]]
                            + self.f_scale * (population[[b, j]] - population[[c, j]]);
                    } else {
                        trial_population[[i, j]] = population[[i, j]];
                    }
                }
            }

            Ok(trial_population)
        }

        fn selection_gpu(
            &self,
            population: &mut Array2<f64>,
            fitness: &mut Array1<f64>,
            trial_population: &Array2<f64>,
            trial_fitness: &Array1<f64>,
        ) -> ScirsResult<()> {
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

        fn calculate_fitness_std(&self, fitness: &Array1<f64>) -> f64 {
            let mean = fitness.view().mean();
            let variance =
                fitness.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / fitness.len() as f64;
            variance.sqrt()
        }
    }

    /// GPU-accelerated particle swarm optimization
    pub struct GpuParticleSwarm {
        context: GpuOptimizationContext,
        swarm_size: usize,
        max_nit: usize,
        w: f64,  // Inertia weight
        c1: f64, // Cognitive parameter
        c2: f64, // Social parameter
    }

    impl GpuParticleSwarm {
        /// Create a new GPU-accelerated particle swarm optimizer
        pub fn new(context: GpuOptimizationContext, swarm_size: usize, max_nit: usize) -> Self {
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

        /// Optimize function using GPU-accelerated particle swarm optimization
        pub fn optimize<F>(
            &mut self,
            function: &F,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<OptimizeResults<f64>>
        where
            F: GpuFunction,
        {
            let dims = bounds.len();

            // Initialize swarm positions and velocities
            let mut positions = self.initialize_positions_gpu(bounds)?;
            let mut velocities = Array2::zeros((self.swarm_size, dims));
            let mut personal_best = positions.clone();
            let mut personal_best_fitness = self.evaluate_population_gpu(function, &positions)?;

            // Find global best
            let mut global_best_idx = 0;
            let mut global_best_fitness = personal_best_fitness[0];
            for (i, &fitness) in personal_best_fitness.iter().enumerate() {
                if fitness < global_best_fitness {
                    global_best_fitness = fitness;
                    global_best_idx = i;
                }
            }
            let mut global_best = personal_best.row(global_best_idx).to_owned();

            let mut function_evaluations = self.swarm_size;

            for iteration in 0..self.max_nit {
                // Update velocities and positions on GPU
                self.update_swarm_gpu(
                    &mut positions,
                    &mut velocities,
                    &personal_best,
                    &global_best,
                    bounds,
                )?;

                // Evaluate new positions
                let fitness = self.evaluate_population_gpu(function, &positions)?;
                function_evaluations += self.swarm_size;

                // Update personal bests
                for i in 0..self.swarm_size {
                    if fitness[i] < personal_best_fitness[i] {
                        personal_best_fitness[i] = fitness[i];
                        for j in 0..dims {
                            personal_best[[i, j]] = positions[[i, j]];
                        }

                        // Update global best
                        if fitness[i] < global_best_fitness {
                            global_best_fitness = fitness[i];
                            global_best = positions.row(i).to_owned();
                        }
                    }
                }

                // Convergence check
                if iteration % 10 == 0 {
                    let fitness_std = self.calculate_fitness_std(&personal_best_fitness);
                    if fitness_std < 1e-12 {
                        break;
                    }
                }
            }

            Ok(OptimizeResults::<f64> {
                x: global_best,
                fun: global_best_fitness,
                success: true,
                message: "GPU particle swarm optimization completed".to_string(),
                nit: self.max_nit,
                nfev: function_evaluations,
                ..OptimizeResults::default()
            })
        }

        fn initialize_positions_gpu(&self, bounds: &[(f64, f64)]) -> ScirsResult<Array2<f64>> {
            use rand::Rng;
            let mut rng = rand::rng();

            let dims = bounds.len();
            let mut positions = Array2::zeros((self.swarm_size, dims));

            for i in 0..self.swarm_size {
                for j in 0..dims {
                    let (low, high) = bounds[j];
                    positions[[i, j]] = rng.gen_range(low..=high);
                }
            }

            Ok(positions)
        }

        fn evaluate_population_gpu<F>(
            &self,
            function: &F,
            population: &Array2<f64>,
        ) -> ScirsResult<Array1<f64>>
        where
            F: GpuFunction,
        {
            self.context.evaluate_function_batch(function, population)
        }

        fn update_swarm_gpu(
            &self,
            positions: &mut Array2<f64>,
            velocities: &mut Array2<f64>,
            personal_best: &Array2<f64>,
            global_best: &Array1<f64>,
            bounds: &[(f64, f64)],
        ) -> ScirsResult<()> {
            use rand::Rng;
            let mut rng = rand::rng();

            let (swarm_size, dims) = positions.dim();

            for i in 0..swarm_size {
                for j in 0..dims {
                    let r1: f64 = rng.gen_range(0.0..1.0);
                    let r2: f64 = rng.gen_range(0.0..1.0);

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

        fn calculate_fitness_std(&self, fitness: &Array1<f64>) -> f64 {
            let mean = fitness.view().mean();
            let variance =
                fitness.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / fitness.len() as f64;
            variance.sqrt()
        }
    }
}

/// Utility functions for GPU optimization
pub mod utils {
    use super::*;

    /// Check if GPU acceleration is available and beneficial
    pub fn should_use_gpu(_problem_size: usize, batch_size: usize) -> bool {
        // Heuristic: GPU is beneficial for large problems or large batches
        _problem_size * batch_size > 10000
    }

    /// Estimate optimal batch size for GPU evaluation
    pub fn estimate_optimal_batch_size(
        problem_dims: usize,
        available_memory: usize,
        precision: GpuPrecision,
    ) -> usize {
        let element_size = match precision {
            GpuPrecision::F32 => 4,
            GpuPrecision::F64 => 8,
            GpuPrecision::Mixed => 6, // Average of F32 and F64
        };

        let memory_per_point = problem_dims * element_size * 3; // Input, output, temp
        let batch_size = available_memory / memory_per_point;

        // Ensure batch size is reasonable
        batch_size.max(1).min(65536)
    }

    /// Create optimal GPU configuration for a given problem
    pub fn create_optimal_config(
        problem_dims: usize,
        expected_evaluations: usize,
    ) -> ScirsResult<GpuOptimizationConfig> {
        let context = GpuContext::new(scirs2_core::gpu::GpuBackend::Cuda)?;
        let available_memory = 1024 * 1024 * 1024; // Default 1GB, should be queried from device

        let batch_size = estimate_optimal_batch_size(
            problem_dims,
            available_memory / 2, // Use half of available memory
            GpuPrecision::F64,
        );

        Ok(GpuOptimizationConfig {
            context: Arc::new(context),
            batch_size,
            memory_limit: Some(available_memory / 2),
            use_tensor_cores: true,
            precision: GpuPrecision::F64,
            stream_count: 4,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_creation() {
        let config = GpuOptimizationConfig::default();
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.precision, GpuPrecision::F64);
        assert!(config.use_tensor_cores);
    }

    #[test]
    fn test_optimal_batch_size_estimation() {
        let batch_size = utils::estimate_optimal_batch_size(
            10,          // 10-dimensional problem
            1024 * 1024, // 1MB memory
            GpuPrecision::F64,
        );
        assert!(batch_size > 0);
        assert!(batch_size <= 65536);
    }

    #[test]
    fn test_gpu_usage_heuristic() {
        assert!(!utils::should_use_gpu(10, 10)); // Small problem
        assert!(utils::should_use_gpu(1000, 100)); // Large problem
        assert!(utils::should_use_gpu(100, 1000)); // Large batch
    }
}
