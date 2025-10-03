//! Memory-efficient sparse optimization for very large-scale problems
//!
//! This module provides optimization algorithms specifically designed for very large
//! problems (millions of variables) with sparse structure, using out-of-core storage,
//! progressive refinement, and advanced memory management techniques.

use crate::error::OptimizeError;
use crate::sparse_numdiff::SparseFiniteDiffOptions;
use crate::unconstrained::memory_efficient::MemoryOptions;
use crate::unconstrained::result::OptimizeResult;
// use crate::unconstrained::sparse_optimization::compute_sparse_gradient;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_sparse::csr_array::CsrArray;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Options for advanced-large-scale memory-efficient optimization
#[derive(Debug, Clone)]
pub struct AdvancedScaleOptions {
    /// Base memory options
    pub memory_options: MemoryOptions,
    /// Sparse finite difference options
    pub sparse_options: SparseFiniteDiffOptions,
    /// Maximum number of variables to keep in memory simultaneously
    pub max_variables_in_memory: usize,
    /// Block size for progressive refinement
    pub block_size: usize,
    /// Number of progressive refinement passes
    pub refinement_passes: usize,
    /// Use disk-based storage for very large problems
    pub use_disk_storage: bool,
    /// Memory mapping threshold (variables)
    pub mmap_threshold: usize,
    /// Compression level for disk storage (0-9)
    pub compression_level: u32,
}

impl Default for AdvancedScaleOptions {
    fn default() -> Self {
        Self {
            memory_options: MemoryOptions::default(),
            sparse_options: SparseFiniteDiffOptions::default(),
            max_variables_in_memory: 100_000,
            block_size: 10_000,
            refinement_passes: 3,
            use_disk_storage: false,
            mmap_threshold: 1_000_000,
            compression_level: 3,
        }
    }
}

/// Advanced-large-scale problem state manager
struct AdvancedScaleState {
    /// Current variable values (may be stored on disk)
    variables: VariableStorage,
    /// Current gradient (sparse representation)
    gradient: Option<CsrArray<f64>>,
    /// Variable blocks for progressive processing
    blocks: Vec<VariableBlock>,
    /// Active variable indices for current iteration
    #[allow(dead_code)]
    active_variables: Vec<usize>,
    /// Temporary directory for out-of-core storage
    #[allow(dead_code)]
    temp_dir: Option<PathBuf>,
}

/// Variable storage abstraction (memory or disk-based)
enum VariableStorage {
    /// All variables in memory
    Memory(Array1<f64>),
    /// Variables stored on disk with memory-mapped access
    Disk {
        file_path: PathBuf,
        size: usize,
        #[allow(dead_code)]
        buffer: Vec<f64>, // In-memory cache for active variables
        #[allow(dead_code)]
        active_indices: Vec<usize>,
    },
}

/// Block of variables for progressive processing
#[derive(Debug, Clone)]
struct VariableBlock {
    /// Starting index of the block
    start_idx: usize,
    /// Size of the block
    size: usize,
    /// Priority for processing (higher means more important)
    priority: f64,
    /// Last time this block was updated
    last_updated: usize,
}

impl AdvancedScaleState {
    fn new(x0: Array1<f64>, options: &AdvancedScaleOptions) -> Result<Self, OptimizeError> {
        let n = x0.len();

        // Determine storage strategy
        let variables = if options.use_disk_storage || n > options.mmap_threshold {
            // Create temporary file for large problems
            let temp_dir = options
                .memory_options
                .temp_dir
                .clone()
                .unwrap_or_else(std::env::temp_dir);

            std::fs::create_dir_all(&temp_dir).map_err(|e| {
                OptimizeError::ComputationError(format!("Failed to create temp directory: {}", e))
            })?;

            let file_path = temp_dir.join(format!("advancedscale_vars_{}.dat", std::process::id()));

            // Write initial variables to disk
            let mut file = File::create(&file_path).map_err(|e| {
                OptimizeError::ComputationError(format!("Failed to create temp file: {}", e))
            })?;

            let bytes: Vec<u8> = x0
                .as_slice()
                .unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            file.write_all(&bytes).map_err(|e| {
                OptimizeError::ComputationError(format!("Failed to write to temp file: {}", e))
            })?;

            VariableStorage::Disk {
                file_path,
                size: n,
                buffer: vec![0.0; options.max_variables_in_memory.min(n)],
                active_indices: Vec::new(),
            }
        } else {
            VariableStorage::Memory(x0)
        };

        // Create variable blocks for progressive processing
        let blocks = create_variable_blocks(n, options.block_size);

        Ok(Self {
            variables,
            gradient: None,
            blocks,
            active_variables: Vec::new(),
            temp_dir: options.memory_options.temp_dir.clone(),
        })
    }

    /// Get current variable values (loading from disk if necessary)
    fn get_variables(&mut self) -> Result<Array1<f64>, OptimizeError> {
        match &mut self.variables {
            VariableStorage::Memory(x) => Ok(x.clone()),
            VariableStorage::Disk {
                file_path,
                size,
                buffer: _,
                active_indices: _,
            } => {
                // Load all variables from disk (expensive, use sparingly)
                let mut file = File::open(file_path).map_err(|e| {
                    OptimizeError::ComputationError(format!("Failed to open temp file: {}", e))
                })?;

                let mut bytes = vec![0u8; *size * 8];
                file.read_exact(&mut bytes).map_err(|e| {
                    OptimizeError::ComputationError(format!("Failed to read from temp file: {}", e))
                })?;

                let values: Result<Vec<f64>, _> = bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let array: [u8; 8] = chunk.try_into().unwrap();
                        Ok(f64::from_le_bytes(array))
                    })
                    .collect();

                let values = values.map_err(|_: std::io::Error| {
                    OptimizeError::ComputationError("Failed to deserialize variables".to_string())
                })?;

                Ok(Array1::from_vec(values))
            }
        }
    }

    /// Update variables (writing to disk if necessary)
    fn update_variables(&mut self, new_x: &Array1<f64>) -> Result<(), OptimizeError> {
        match &mut self.variables {
            VariableStorage::Memory(x) => {
                x.assign(new_x);
                Ok(())
            }
            VariableStorage::Disk {
                file_path,
                size: _,
                buffer: _,
                active_indices: _,
            } => {
                // Write all variables to disk
                let mut file = OpenOptions::new()
                    .write(true)
                    .open(file_path)
                    .map_err(|e| {
                        OptimizeError::ComputationError(format!("Failed to open temp file: {}", e))
                    })?;

                let bytes: Vec<u8> = new_x
                    .as_slice()
                    .unwrap()
                    .iter()
                    .flat_map(|&x| x.to_le_bytes())
                    .collect();

                file.seek(SeekFrom::Start(0)).map_err(|e| {
                    OptimizeError::ComputationError(format!("Failed to seek in temp file: {}", e))
                })?;

                file.write_all(&bytes).map_err(|e| {
                    OptimizeError::ComputationError(format!("Failed to write to temp file: {}", e))
                })?;

                Ok(())
            }
        }
    }
}

impl Drop for AdvancedScaleState {
    fn drop(&mut self) {
        // Clean up temporary files
        if let VariableStorage::Disk { file_path, .. } = &self.variables {
            let _ = std::fs::remove_file(file_path);
        }
    }
}

/// Create variable blocks for progressive processing
#[allow(dead_code)]
fn create_variable_blocks(n: usize, block_size: usize) -> Vec<VariableBlock> {
    let mut _blocks = Vec::new();
    let num_blocks = n.div_ceil(block_size);

    for i in 0..num_blocks {
        let start_idx = i * block_size;
        let end_idx = std::cmp::min((i + 1) * block_size, n);
        let size = end_idx - start_idx;

        _blocks.push(VariableBlock {
            start_idx,
            size,
            priority: 1.0, // Initial priority
            last_updated: 0,
        });
    }

    _blocks
}

/// Advanced-large-scale optimization using progressive refinement and memory management
#[allow(dead_code)]
pub fn minimize_advanced_scale<F, S>(
    fun: F,
    x0: Array1<f64>,
    options: &AdvancedScaleOptions,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S + Clone + Sync,
    S: Into<f64> + Clone + Send,
{
    let n = x0.len();

    if n < options.max_variables_in_memory {
        // For smaller problems, use regular memory-efficient algorithm
        return super::memory_efficient::minimize_memory_efficient_lbfgs(
            fun,
            x0,
            &options.memory_options,
        );
    }

    println!(
        "Starting advanced-large-scale optimization for {} variables",
        n
    );

    // Initialize state manager
    let mut state = AdvancedScaleState::new(x0, options)?;
    let mut iteration = 0;
    let mut total_nfev = 0;

    // Progressive refinement loop
    for pass in 0..options.refinement_passes {
        println!(
            "Progressive refinement pass {}/{}",
            pass + 1,
            options.refinement_passes
        );

        // Update block priorities based on gradient information
        if let Some(ref gradient) = state.gradient {
            update_block_priorities(&mut state.blocks, gradient);
        }

        // Sort blocks by priority (highest first)
        state
            .blocks
            .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        // Process blocks in order of priority
        let num_blocks = state.blocks.len();
        for i in 0..num_blocks {
            iteration += 1;

            // Extract current block variables
            let current_x = state.get_variables()?;
            let block_start = state.blocks[i].start_idx;
            let block_size = state.blocks[i].size;
            let block_x = current_x
                .slice(scirs2_core::ndarray::s![
                    block_start..block_start + block_size
                ])
                .to_owned();

            // Optimize this block while keeping others fixed
            let block_result = optimize_block(
                &fun,
                &current_x,
                block_start,
                block_x,
                &options.memory_options,
            )?;

            total_nfev += block_result.nfev;

            // Update the full solution with optimized block
            let mut new_x = current_x.clone();
            new_x
                .slice_mut(scirs2_core::ndarray::s![
                    block_start..block_start + block_size
                ])
                .assign(&block_result.x);

            state.update_variables(&new_x)?;
            state.blocks[i].last_updated = iteration;

            // Check global convergence periodically
            if iteration % 10 == 0 {
                let f_val = fun(&new_x.view()).into();
                let grad_norm = estimate_sparse_gradient_norm(&fun, &new_x.view(), options)?;

                if grad_norm < options.memory_options.base_options.gtol {
                    println!("Converged at pass {}, iteration {}", pass + 1, iteration);

                    return Ok(OptimizeResult {
                        x: new_x.clone(),
                        fun: fun(&new_x.view()),
                        nit: iteration,
                        func_evals: total_nfev,
                        nfev: total_nfev,
                        success: true,
                        message: "Advanced-scale optimization converged successfully.".to_string(),
                        jacobian: None,
                        hessian: None,
                    });
                }

                println!(
                    "Iteration {}: f = {:.6e}, ||g|| ≈ {:.6e}",
                    iteration, f_val, grad_norm
                );
            }

            if iteration >= options.memory_options.base_options.max_iter {
                break;
            }
        }
    }

    let final_x = state.get_variables()?;

    Ok(OptimizeResult {
        x: final_x.clone(),
        fun: fun(&final_x.view()),
        nit: iteration,
        func_evals: total_nfev,
        nfev: total_nfev,
        success: iteration < options.memory_options.base_options.max_iter,
        message: if iteration < options.memory_options.base_options.max_iter {
            "Advanced-scale optimization completed successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: None,
        hessian: None,
    })
}

/// Optimize a single block of variables
#[allow(dead_code)]
fn optimize_block<F, S>(
    fun: &F,
    full_x: &Array1<f64>,
    block_start: usize,
    block_x0: Array1<f64>,
    options: &MemoryOptions,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    // Create a wrapper function that only varies the block variables
    let block_fun = |block_vars: &ArrayView1<f64>| -> f64 {
        let mut temp_x = full_x.clone();
        temp_x
            .slice_mut(scirs2_core::ndarray::s![
                block_start..block_start + block_vars.len()
            ])
            .assign(block_vars);
        fun(&temp_x.view()).into()
    };

    // Use memory-efficient L-BFGS for the block
    super::memory_efficient::minimize_memory_efficient_lbfgs(block_fun, block_x0, options)
}

/// Update block priorities based on gradient magnitude
#[allow(dead_code)]
fn update_block_priorities(blocks: &mut [VariableBlock], gradient: &CsrArray<f64>) {
    for block in blocks {
        // Compute average gradient magnitude in this block
        let mut total_grad_mag = 0.0;
        let mut count = 0;

        // Sparse gradient access
        for idx in block.start_idx..block.start_idx + block.size {
            if let Some(&grad_val) = gradient.get_data().get(idx) {
                total_grad_mag += grad_val.abs();
                count += 1;
            }
        }

        if count > 0 {
            block.priority = total_grad_mag / count as f64;
        } else {
            block.priority *= 0.9; // Decay priority for _blocks with no gradient info
        }
    }
}

/// Estimate sparse gradient norm efficiently
#[allow(dead_code)]
fn estimate_sparse_gradient_norm<F, S>(
    fun: &F,
    x: &ArrayView1<f64>,
    options: &AdvancedScaleOptions,
) -> Result<f64, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let n = x.len();

    // For very large problems, use sampling to estimate gradient norm
    if n > options.max_variables_in_memory {
        // Sample a subset of variables for gradient estimation
        let sample_size = (options.max_variables_in_memory / 10).min(1000).max(10);
        let step_size = options.sparse_options.rel_step.unwrap_or(1e-8);

        let mut gradient_norm_squared = 0.0;
        let f0 = fun(x).into();

        // Sample variables uniformly across the space
        let step = n / sample_size;
        for i in (0..n).step_by(step).take(sample_size) {
            // Compute finite difference for this variable
            let mut x_pert = x.to_owned();
            let h = step_size * (1.0 + x[i].abs());
            x_pert[i] += h;

            let f_pert = fun(&x_pert.view()).into();
            let grad_i = (f_pert - f0) / h;
            gradient_norm_squared += grad_i * grad_i;
        }

        // Scale by the sampling ratio to estimate full gradient norm
        let scaling_factor = n as f64 / sample_size as f64;
        Ok((gradient_norm_squared * scaling_factor).sqrt())
    } else {
        // For smaller problems, compute a more accurate sparse gradient
        compute_sparse_gradient_norm(fun, x, &options.sparse_options)
    }
}

/// Compute sparse gradient norm using finite differences
#[allow(dead_code)]
fn compute_sparse_gradient_norm<F, S>(
    fun: &F,
    x: &ArrayView1<f64>,
    sparse_options: &SparseFiniteDiffOptions,
) -> Result<f64, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> S,
    S: Into<f64>,
{
    let n = x.len();
    let step_size = sparse_options.rel_step.unwrap_or(1e-8);
    let _f0 = fun(x).into();

    // Use central differences for better accuracy
    let mut gradient_norm_squared = 0.0;

    for i in 0..n {
        let h = step_size * (1.0 + x[i].abs());

        // Forward difference
        let mut x_forward = x.to_owned();
        x_forward[i] += h;
        let f_forward = fun(&x_forward.view()).into();

        // Backward difference
        let mut x_backward = x.to_owned();
        x_backward[i] -= h;
        let f_backward = fun(&x_backward.view()).into();

        // Central difference
        let grad_i = (f_forward - f_backward) / (2.0 * h);
        gradient_norm_squared += grad_i * grad_i;
    }

    Ok(gradient_norm_squared.sqrt())
}

/// Create advanced-scale optimizer with automatic parameter selection
#[allow(dead_code)]
pub fn create_advanced_scale_optimizer(
    problem_size: usize,
    available_memory_mb: usize,
    estimated_sparsity: f64, // Fraction of non-zero elements
) -> AdvancedScaleOptions {
    let available_bytes = available_memory_mb * 1024 * 1024;

    // Estimate memory per variable
    let bytes_per_var = std::mem::size_of::<f64>() * 8; // Variable + gradient + temporaries
    let max_vars_in_memory = (available_bytes / bytes_per_var).min(problem_size);

    // Block _size based on memory and _sparsity
    let block_size = if estimated_sparsity < 0.1 {
        // Very sparse: larger blocks
        (max_vars_in_memory / 4).max(1000)
    } else {
        // Dense: smaller blocks
        (max_vars_in_memory / 10).max(100)
    };

    // Use disk storage for very large problems
    let use_disk = problem_size >= 1_000_000 || available_memory_mb < 512;

    // More refinement passes for larger problems
    let refinement_passes = if problem_size > 10_000_000 {
        5
    } else if problem_size > 1_000_000 {
        4
    } else {
        3
    };

    AdvancedScaleOptions {
        memory_options: super::memory_efficient::create_memory_efficient_optimizer(
            max_vars_in_memory,
            available_memory_mb / 2,
        ),
        sparse_options: SparseFiniteDiffOptions {
            max_group_size: (estimated_sparsity * 1000.0) as usize,
            ..Default::default()
        },
        max_variables_in_memory: max_vars_in_memory,
        block_size,
        refinement_passes,
        use_disk_storage: use_disk,
        mmap_threshold: available_bytes / (2 * std::mem::size_of::<f64>()),
        compression_level: if available_memory_mb < 256 { 6 } else { 3 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_advanced_scale_small_problem() {
        // Test that small problems fall back to regular memory-efficient method
        let quadratic = |x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi.powi(2)).sum() };

        let n = 50; // Small problem
        let x0 = Array1::ones(n);
        let options = AdvancedScaleOptions::default();

        let result = minimize_advanced_scale(quadratic, x0, &options).unwrap();

        assert!(result.success);
        // Should converge to origin
        for i in 0..n {
            assert_abs_diff_eq!(result.x[i], 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_variable_blocks() {
        let blocks = create_variable_blocks(1000, 100);
        assert_eq!(blocks.len(), 10);
        assert_eq!(blocks[0].start_idx, 0);
        assert_eq!(blocks[0].size, 100);
        assert_eq!(blocks[9].start_idx, 900);
        assert_eq!(blocks[9].size, 100);
    }

    #[test]
    fn test_auto_parameter_selection() {
        let options = create_advanced_scale_optimizer(1_000_000, 1024, 0.05);

        assert!(options.max_variables_in_memory > 0);
        assert!(options.block_size > 0);
        assert!(options.refinement_passes >= 3);
        assert!(options.use_disk_storage); // Should use disk for 1M variables
    }
}
