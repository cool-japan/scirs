//! Parallel FFT Planning
//!
//! This module extends the planning system with multithreaded planning capabilities,
//! allowing for parallel plan creation and execution.

use crate::error::{FFTError, FFTResult};
use crate::planning::{
    AdvancedFftPlanner, FftPlan, FftPlanExecutor, PlannerBackend, PlanningConfig,
};
use crate::worker_pool::WorkerPool;

use num_complex::Complex64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration options for parallel planning
#[derive(Debug, Clone)]
pub struct ParallelPlanningConfig {
    /// Base planning configuration
    pub base_config: PlanningConfig,

    /// Maximum number of threads to use
    pub max_threads: Option<usize>,

    /// Minimum size threshold for parallel planning
    pub parallel_threshold: usize,

    /// Whether to use work stealing
    pub use_work_stealing: bool,

    /// Whether to enable parallel execution
    pub parallel_execution: bool,
}

impl Default for ParallelPlanningConfig {
    fn default() -> Self {
        Self {
            base_config: PlanningConfig::default(),
            max_threads: None,        // Use all available threads
            parallel_threshold: 1024, // Only use parallelism for FFTs >= 1024 elements
            use_work_stealing: true,
            parallel_execution: true,
        }
    }
}

/// Result of a parallel plan creation
// Custom Debug implementation because FftPlan doesn't implement Debug
pub struct ParallelPlanResult {
    /// The created plan
    pub plan: Arc<FftPlan>,

    /// Time taken to create the plan
    pub creation_time: Duration,

    /// Shape of the FFT for this plan
    pub shape: Vec<usize>,

    /// Thread ID that created this plan
    pub thread_id: usize,
}

impl std::fmt::Debug for ParallelPlanResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelPlanResult")
            .field("shape", &self.shape)
            .field("creation_time", &self.creation_time)
            .field("thread_id", &self.thread_id)
            .field("plan", &format!("<FftPlan: shape={:?}>", self.shape))
            .finish()
    }
}

/// Parallel planner that can create multiple plans simultaneously
pub struct ParallelPlanner {
    /// Base planner
    base_planner: Arc<Mutex<AdvancedFftPlanner>>,

    /// Configuration
    config: ParallelPlanningConfig,

    /// Worker pool for parallel execution
    worker_pool: Arc<WorkerPool>,
}

impl ParallelPlanner {
    /// Create a new parallel planner
    pub fn new(config: Option<ParallelPlanningConfig>) -> Self {
        let config = config.unwrap_or_default();
        let base_planner = Arc::new(Mutex::new(AdvancedFftPlanner::with_config(
            config.base_config.clone(),
        )));

        let worker_pool = match config.max_threads {
            Some(threads) => {
                let worker_config = crate::worker_pool::WorkerConfig {
                    num_workers: threads,
                    ..Default::default()
                };
                Arc::new(
                    WorkerPool::with_config(worker_config).unwrap_or_else(|_| WorkerPool::new()),
                )
            }
            None => Arc::new(WorkerPool::new()),
        };

        Self {
            base_planner,
            config,
            worker_pool,
        }
    }

    /// Create a single plan
    pub fn plan_fft(
        &self,
        shape: &[usize],
        forward: bool,
        backend: PlannerBackend,
    ) -> FFTResult<Arc<FftPlan>> {
        // For small FFTs, use the base planner directly
        let size = shape.iter().product::<usize>();
        if size < self.config.parallel_threshold || !self.config.parallel_execution {
            let mut planner = self.base_planner.lock().unwrap();
            return planner.plan_fft(shape, forward, backend);
        }

        // For larger FFTs, use the worker pool
        let planner_clone = self.base_planner.clone();
        let shape_clone = shape.to_vec();
        let backend_clone = backend.clone();

        let result = self.worker_pool.execute(move || {
            let mut planner = planner_clone.lock().unwrap();
            planner
                .plan_fft(&shape_clone, forward, backend_clone)
                .map_err(|e| format!("FFT planning error: {}", e))
        });

        match result {
            Ok(plan) => Ok(plan),
            Err(err) => Err(FFTError::PlanError(err)),
        }
    }

    /// Create multiple plans in parallel
    pub fn plan_multiple(
        &self,
        specs: &[(Vec<usize>, bool, PlannerBackend)],
    ) -> FFTResult<Vec<ParallelPlanResult>> {
        // Filter out small FFTs that would be processed serially
        let (small_specs, large_specs): (Vec<_>, Vec<_>) =
            specs.iter().enumerate().partition(|(_, (shape, _, _))| {
                shape.iter().product::<usize>() < self.config.parallel_threshold
            });

        // Process small FFTs serially
        let mut results = Vec::with_capacity(specs.len());
        for (idx, (shape, forward, backend)) in small_specs {
            let start = Instant::now();
            let plan = {
                let mut planner = self.base_planner.lock().unwrap();
                planner.plan_fft(shape, *forward, backend.clone())?
            };
            results.push((
                idx,
                ParallelPlanResult {
                    plan,
                    creation_time: start.elapsed(),
                    shape: shape.clone(),
                    thread_id: 0, // Main thread
                },
            ));
        }

        // Process large FFTs in parallel
        if !large_specs.is_empty() {
            let planner_clone = self.base_planner.clone();

            // Submit each plan creation as a separate task
            let plan_futures = large_specs
                .iter()
                .map(|(idx, (shape, forward, backend))| {
                    let planner = planner_clone.clone();
                    let shape_clone = shape.clone();
                    let backend_clone = backend.clone();
                    let forward_val = *forward;
                    let idx_val = *idx;

                    self.worker_pool.execute(move || {
                        let thread_id = 0; // Thread ID tracking handled by core parallel abstractions
                        let start = Instant::now();
                        let plan = {
                            let mut planner_guard = planner.lock().unwrap();
                            planner_guard
                                .plan_fft(&shape_clone, forward_val, backend_clone)
                                .map_err(|e| format!("FFT planning error: {}", e))?
                        };

                        Ok((
                            idx_val,
                            ParallelPlanResult {
                                plan,
                                creation_time: start.elapsed(),
                                shape: shape_clone,
                                thread_id,
                            },
                        ))
                    })
                })
                .collect::<Vec<_>>();

            // Plans are computed when executed - directly collect results
            for result in plan_futures {
                match result {
                    Ok((idx, result)) => results.push((idx, result)),
                    Err(err) => return Err(FFTError::PlanError(err)),
                }
            }
        }

        // Sort results by original index
        results.sort_by_key(|(idx, _)| *idx);
        Ok(results.into_iter().map(|(_, result)| result).collect())
    }

    /// Clear the plan cache
    pub fn clear_cache(&self) {
        let planner = self.base_planner.lock().unwrap();
        planner.clear_cache();
    }

    /// Save plans to disk
    pub fn save_plans(&self) -> FFTResult<()> {
        let planner = self.base_planner.lock().unwrap();
        planner.save_plans()
    }
}

/// Executor for parallel FFT operations
pub struct ParallelExecutor {
    /// The plan to execute
    plan: Arc<FftPlan>,

    /// Configuration
    config: ParallelPlanningConfig,

    /// Worker pool
    worker_pool: Arc<WorkerPool>,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(plan: Arc<FftPlan>, config: Option<ParallelPlanningConfig>) -> Self {
        let config = config.unwrap_or_default();

        let worker_pool = match config.max_threads {
            Some(threads) => {
                let worker_config = crate::worker_pool::WorkerConfig {
                    num_workers: threads,
                    ..Default::default()
                };
                Arc::new(
                    WorkerPool::with_config(worker_config).unwrap_or_else(|_| WorkerPool::new()),
                )
            }
            None => Arc::new(WorkerPool::new()),
        };

        Self {
            plan,
            config,
            worker_pool,
        }
    }

    /// Execute the plan in parallel
    pub fn execute(&self, input: &[Complex64], output: &mut [Complex64]) -> FFTResult<()> {
        // For small FFTs or if parallel execution is disabled, use the standard executor
        let size = self.plan.shape().iter().product::<usize>();
        if size < self.config.parallel_threshold || !self.config.parallel_execution {
            let executor = FftPlanExecutor::new(self.plan.clone());
            return executor.execute(input, output);
        }

        // For larger FFTs, use parallel execution
        // This is a simplified implementation - a real one would split the data
        // and distribute subtasks across threads

        // For now, we'll just offload the execution to a worker thread
        let plan_clone = self.plan.clone();
        let input_vec = input.to_vec(); // Copy input for thread safety

        let result = self.worker_pool.execute(move || {
            let mut output_vec = vec![Complex64::default(); input_vec.len()];
            let executor = FftPlanExecutor::new(plan_clone);

            executor
                .execute(&input_vec, &mut output_vec)
                .map_err(|e| format!("FFT execution error: {}", e))?;

            Ok(output_vec)
        });

        // Process the result and copy to output
        match result {
            Ok(result_vec) => {
                output.copy_from_slice(&result_vec);
                Ok(())
            }
            Err(err) => Err(FFTError::ComputationError(err)),
        }
    }

    /// Execute multiple FFTs in parallel
    pub fn execute_batch(
        &self,
        inputs: &[&[Complex64]],
        outputs: &mut [&mut [Complex64]],
    ) -> FFTResult<Vec<Duration>> {
        if inputs.len() != outputs.len() {
            return Err(FFTError::ValueError(
                "Input and output counts must match".to_string(),
            ));
        }

        // Verify all inputs/outputs have the correct size
        let expected_size = self.plan.shape().iter().product::<usize>();
        for (i, input) in inputs.iter().enumerate() {
            if input.len() != expected_size {
                return Err(FFTError::ValueError(format!(
                    "Input {} has wrong size: expected {}, got {}",
                    i,
                    expected_size,
                    input.len()
                )));
            }

            if outputs[i].len() != expected_size {
                return Err(FFTError::ValueError(format!(
                    "Output {} has wrong size: expected {}, got {}",
                    i,
                    expected_size,
                    outputs[i].len()
                )));
            }
        }

        // For small batch size, process serially
        if inputs.len() < 2 || !self.config.parallel_execution {
            let mut times = Vec::with_capacity(inputs.len());
            let executor = FftPlanExecutor::new(self.plan.clone());

            for i in 0..inputs.len() {
                let start = Instant::now();
                executor.execute(inputs[i], outputs[i])?;
                times.push(start.elapsed());
            }

            return Ok(times);
        }

        // Process batch in parallel
        let plan_clone = self.plan.clone();

        // Prepare futures for each FFT
        let futures = inputs
            .iter()
            .zip(outputs.iter_mut())
            .enumerate()
            .map(|(idx, (input, output))| {
                let plan = plan_clone.clone();
                let input_vec = input.to_vec(); // Copy for thread safety
                let output_len = output.len();

                self.worker_pool.execute(move || {
                    let mut local_output = vec![Complex64::default(); output_len];
                    let executor = FftPlanExecutor::new(plan);

                    let start = Instant::now();
                    executor
                        .execute(&input_vec, &mut local_output)
                        .map_err(|e| format!("FFT execution error for batch {}: {}", idx, e))?;
                    let elapsed = start.elapsed();

                    Ok((idx, local_output, elapsed))
                })
            })
            .collect::<Vec<_>>();

        // Collect results
        let mut times = vec![Duration::from_secs(0); inputs.len()];

        for result in futures {
            match result {
                Ok((idx, result_vec, elapsed)) => {
                    outputs[idx].copy_from_slice(&result_vec);
                    times[idx] = elapsed;
                }
                Err(err) => return Err(FFTError::ComputationError(err)),
            }
        }

        Ok(times)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_planner() {
        let planner = ParallelPlanner::new(None);

        // Create a plan
        let plan = planner
            .plan_fft(&[64], true, PlannerBackend::default())
            .unwrap();

        // Check the plan properties
        assert_eq!(plan.shape(), &[64]);
    }

    #[test]
    fn test_parallel_executor() {
        let planner = ParallelPlanner::new(None);
        let plan = planner
            .plan_fft(&[64], true, PlannerBackend::default())
            .unwrap();

        let executor = ParallelExecutor::new(plan, None);

        // Create test data
        let input = vec![Complex64::new(1.0, 0.0); 64];
        let mut output = vec![Complex64::default(); 64];

        // Execute the plan
        executor.execute(&input, &mut output).unwrap();

        // Basic validation - output should not be all zeros
        assert!(output.iter().any(|&val| val != Complex64::default()));
    }

    #[test]
    fn test_batch_execution() {
        let planner = ParallelPlanner::new(None);
        let plan = planner
            .plan_fft(&[32], true, PlannerBackend::default())
            .unwrap();

        let executor = ParallelExecutor::new(plan, None);

        // Create multiple test inputs
        let input1 = vec![Complex64::new(1.0, 0.0); 32];
        let input2 = vec![Complex64::new(0.0, 1.0); 32];

        let mut output1 = vec![Complex64::default(); 32];
        let mut output2 = vec![Complex64::default(); 32];

        let inputs = [&input1[..], &input2[..]];
        let mut outputs = [&mut output1[..], &mut output2[..]];

        // Execute batch
        let times = executor.execute_batch(&inputs, &mut outputs).unwrap();

        // Check that we got timing information
        assert_eq!(times.len(), 2);

        // Validate outputs
        assert!(output1.iter().any(|&val| val != Complex64::default()));
        assert!(output2.iter().any(|&val| val != Complex64::default()));

        // Outputs should be different since inputs were different
        assert_ne!(output1[0], output2[0]);
    }
}
