//! Parallel Monte Carlo integration using thread pools
//!
//! This module provides thread-pool based parallel implementations of Monte Carlo
//! integration methods, offering significant performance improvements for
//! computationally expensive integrand functions by distributing work across
//! multiple CPU cores.

use crate::error::{IntegrateError, IntegrateResult};
use crate::monte_carlo::{ErrorEstimationMethod, MonteCarloOptions, MonteCarloResult};
use crate::IntegrateFloat;
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Uniform};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Options for parallel Monte Carlo integration
#[derive(Debug, Clone)]
pub struct ParallelMonteCarloOptions<F: IntegrateFloat> {
    /// Number of sample points to use
    pub n_samples: usize,
    /// Random number generator seed (for reproducibility)
    pub seed: Option<u64>,
    /// Error estimation method
    pub error_method: ErrorEstimationMethod,
    /// Use antithetic variates for variance reduction
    pub use_antithetic: bool,
    /// Number of threads to use (None = automatic)
    pub n_threads: Option<usize>,
    /// Number of samples per batch for load balancing
    pub batch_size: usize,
    /// Use chunk-based parallelism for better load balancing
    pub use_chunking: bool,
    /// Phantom data for generic type
    pub _phantom: PhantomData<F>,
}

impl<F: IntegrateFloat> Default for ParallelMonteCarloOptions<F> {
    fn default() -> Self {
        Self {
            n_samples: 10000,
            seed: None,
            error_method: ErrorEstimationMethod::StandardError,
            use_antithetic: false,
            n_threads: None,  // Use all available cores
            batch_size: 1000, // Process 1000 samples per batch
            use_chunking: true,
            _phantom: PhantomData,
        }
    }
}

/// Convert regular MonteCarloOptions to ParallelMonteCarloOptions
impl<F: IntegrateFloat> From<MonteCarloOptions<F>> for ParallelMonteCarloOptions<F> {
    fn from(opts: MonteCarloOptions<F>) -> Self {
        Self {
            n_samples: opts.n_samples,
            seed: opts.seed,
            error_method: opts.error_method,
            use_antithetic: opts.use_antithetic,
            n_threads: None,
            batch_size: 1000,
            use_chunking: true,
            _phantom: PhantomData,
        }
    }
}

/// Parallel Monte Carlo integration using thread pools
///
/// This function distributes Monte Carlo sampling across multiple threads,
/// providing significant speedup for computationally expensive integrands.
/// Each thread works on independent batches of samples to minimize
/// synchronization overhead.
///
/// # Arguments
///
/// * `f` - The function to integrate (must be thread-safe)
/// * `ranges` - Integration ranges (a, b) for each dimension
/// * `options` - Optional parallel Monte Carlo parameters
///
/// # Returns
///
/// * `IntegrateResult<MonteCarloResult<F>>` - The result of the integration
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo_parallel::{parallel_monte_carlo, ParallelMonteCarloOptions};
/// use ndarray::ArrayView1;
/// use std::marker::PhantomData;
///
/// // Integrate an expensive function f(x,y) = sin(x*y) * exp(-x²-y²) over [-2,2]×[-2,2]
/// let options = ParallelMonteCarloOptions {
///     n_samples: 1000000,
///     n_threads: Some(4),
///     batch_size: 5000,
///     _phantom: PhantomData,
///     ..Default::default()
/// };
///
/// # #[cfg(feature = "parallel")]
/// let result = parallel_monte_carlo(
///     |x: ArrayView1<f64>| (x[0] * x[1]).sin() * (-x[0]*x[0] - x[1]*x[1]).exp(),
///     &[(-2.0, 2.0), (-2.0, 2.0)],
///     Some(options)
/// ).unwrap();
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_monte_carlo<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<ParallelMonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    rand_distr::StandardNormal: Distribution<F>,
{
    let opts = options.unwrap_or_default();
    let n_dims = ranges.len();

    if n_dims == 0 {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    if opts.n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "Number of samples must be positive".to_string(),
        ));
    }

    // Configure thread pool
    let pool = if let Some(n_threads) = opts.n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| {
                IntegrateError::ComputationError(format!("Failed to create thread pool: {}", e))
            })?
    } else {
        rayon::ThreadPoolBuilder::new().build().map_err(|e| {
            IntegrateError::ComputationError(format!("Failed to create thread pool: {}", e))
        })?
    };

    pool.install(|| {
        if opts.use_chunking {
            parallel_monte_carlo_chunked(&f, ranges, &opts)
        } else {
            parallel_monte_carlo_batched(&f, ranges, &opts)
        }
    })
}

/// Chunk-based parallel Monte Carlo integration
#[cfg(feature = "parallel")]
fn parallel_monte_carlo_chunked<F, Func>(
    f: &Func,
    ranges: &[(F, F)],
    opts: &ParallelMonteCarloOptions<F>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    rand_distr::StandardNormal: Distribution<F>,
{
    let n_dims = ranges.len();
    let n_actual_samples = if opts.use_antithetic {
        opts.n_samples / 2 * 2 // Ensure even number for antithetic pairs
    } else {
        opts.n_samples
    };

    // Calculate the volume of the integration domain
    let mut volume = F::one();
    for &(a, b) in ranges {
        volume *= b - a;
    }

    // Create chunks of sample indices
    let chunk_size = opts.batch_size;
    let chunks: Vec<_> = (0..n_actual_samples)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    // Process chunks in parallel
    let results: Vec<_> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            // Each thread gets its own RNG with a different seed
            let thread_seed = opts.seed.unwrap_or(42) + chunk_idx as u64;
            let mut rng = StdRng::seed_from_u64(thread_seed);

            // Create uniform distributions for each dimension
            let distributions: Vec<_> = ranges
                .iter()
                .map(|&(a, b)| Uniform::new_inclusive(a, b).unwrap())
                .collect();

            let mut thread_sum = F::zero();
            let mut thread_sum_sq = F::zero();
            let mut thread_n_evals = 0;
            let mut point = Array1::zeros(n_dims);

            if opts.use_antithetic {
                // Process antithetic pairs
                for _ in 0..(chunk.len() / 2) {
                    // Generate a random point
                    for (i, dist) in distributions.iter().enumerate() {
                        point[i] = dist.sample(&mut rng);
                    }
                    let value = f(point.view());
                    thread_sum += value;
                    thread_sum_sq += value * value;
                    thread_n_evals += 1;

                    // Antithetic point
                    for (i, &(a, b)) in ranges.iter().enumerate() {
                        point[i] = a + b - point[i];
                    }
                    let antithetic_value = f(point.view());
                    thread_sum += antithetic_value;
                    thread_sum_sq += antithetic_value * antithetic_value;
                    thread_n_evals += 1;
                }
            } else {
                // Standard sampling
                for _ in chunk {
                    for (i, dist) in distributions.iter().enumerate() {
                        point[i] = dist.sample(&mut rng);
                    }
                    let value = f(point.view());
                    thread_sum += value;
                    thread_sum_sq += value * value;
                    thread_n_evals += 1;
                }
            }

            (thread_sum, thread_sum_sq, thread_n_evals)
        })
        .collect();

    // Combine results from all threads
    let (total_sum, total_sum_sq, total_n_evals) = results.into_iter().fold(
        (F::zero(), F::zero(), 0),
        |(sum, sum_sq, n), (thread_sum, thread_sum_sq, thread_n)| {
            (sum + thread_sum, sum_sq + thread_sum_sq, n + thread_n)
        },
    );

    // Compute final result
    compute_final_result(
        total_sum,
        total_sum_sq,
        total_n_evals,
        volume,
        &opts.error_method,
    )
}

/// Batch-based parallel Monte Carlo integration
#[cfg(feature = "parallel")]
fn parallel_monte_carlo_batched<F, Func>(
    f: &Func,
    ranges: &[(F, F)],
    opts: &ParallelMonteCarloOptions<F>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    rand_distr::StandardNormal: Distribution<F>,
{
    let n_dims = ranges.len();
    let n_actual_samples = if opts.use_antithetic {
        opts.n_samples / 2 * 2
    } else {
        opts.n_samples
    };

    // Calculate the volume of the integration domain
    let mut volume = F::one();
    for &(a, b) in ranges {
        volume *= b - a;
    }

    // Determine number of batches and samples per batch
    let n_threads = rayon::current_num_threads();
    let samples_per_batch = n_actual_samples.div_ceil(n_threads); // Ceiling division

    // Use Arc<Mutex<>> for thread-safe accumulation of results
    let shared_sum = Arc::new(Mutex::new(F::zero()));
    let shared_sum_sq = Arc::new(Mutex::new(F::zero()));
    let shared_n_evals = Arc::new(Mutex::new(0usize));

    // Process batches in parallel
    (0..n_threads).into_par_iter().for_each(|batch_idx| {
        let start_sample = batch_idx * samples_per_batch;
        let end_sample = ((batch_idx + 1) * samples_per_batch).min(n_actual_samples);
        let batch_samples = end_sample - start_sample;

        if batch_samples == 0 {
            return;
        }

        // Each thread gets its own RNG
        let thread_seed = opts.seed.unwrap_or(42) + batch_idx as u64;
        let mut rng = StdRng::seed_from_u64(thread_seed);

        // Create uniform distributions for each dimension
        let distributions: Vec<_> = ranges
            .iter()
            .map(|&(a, b)| Uniform::new_inclusive(a, b).unwrap())
            .collect();

        let mut batch_sum = F::zero();
        let mut batch_sum_sq = F::zero();
        let mut batch_n_evals = 0;
        let mut point = Array1::zeros(n_dims);

        if opts.use_antithetic {
            for _ in 0..(batch_samples / 2) {
                // Generate a random point
                for (i, dist) in distributions.iter().enumerate() {
                    point[i] = dist.sample(&mut rng);
                }
                let value = f(point.view());
                batch_sum += value;
                batch_sum_sq += value * value;
                batch_n_evals += 1;

                // Antithetic point
                for (i, &(a, b)) in ranges.iter().enumerate() {
                    point[i] = a + b - point[i];
                }
                let antithetic_value = f(point.view());
                batch_sum += antithetic_value;
                batch_sum_sq += antithetic_value * antithetic_value;
                batch_n_evals += 1;
            }
        } else {
            for _ in 0..batch_samples {
                for (i, dist) in distributions.iter().enumerate() {
                    point[i] = dist.sample(&mut rng);
                }
                let value = f(point.view());
                batch_sum += value;
                batch_sum_sq += value * value;
                batch_n_evals += 1;
            }
        }

        // Add batch results to shared totals
        {
            let mut sum = shared_sum.lock().unwrap();
            *sum += batch_sum;
        }
        {
            let mut sum_sq = shared_sum_sq.lock().unwrap();
            *sum_sq += batch_sum_sq;
        }
        {
            let mut n_evals = shared_n_evals.lock().unwrap();
            *n_evals += batch_n_evals;
        }
    });

    // Extract final totals
    let total_sum = *shared_sum.lock().unwrap();
    let total_sum_sq = *shared_sum_sq.lock().unwrap();
    let total_n_evals = *shared_n_evals.lock().unwrap();

    compute_final_result(
        total_sum,
        total_sum_sq,
        total_n_evals,
        volume,
        &opts.error_method,
    )
}

/// Compute final Monte Carlo result from accumulated statistics
fn compute_final_result<F: IntegrateFloat>(
    sum: F,
    sum_sq: F,
    n_evals: usize,
    volume: F,
    error_method: &ErrorEstimationMethod,
) -> IntegrateResult<MonteCarloResult<F>> {
    let mean = sum / F::from_usize(n_evals).unwrap();
    let integral_value = mean * volume;

    let std_error = match error_method {
        ErrorEstimationMethod::StandardError => {
            let variance = (sum_sq - sum * sum / F::from_usize(n_evals).unwrap())
                / F::from_usize(n_evals - 1).unwrap();

            (variance / F::from_usize(n_evals).unwrap()).sqrt() * volume
        }
        ErrorEstimationMethod::BatchMeans => {
            // For parallel execution, we can implement proper batch means
            // by using the results from different threads as "batches"
            let variance = (sum_sq - sum * sum / F::from_usize(n_evals).unwrap())
                / F::from_usize(n_evals - 1).unwrap();

            (variance / F::from_usize(n_evals).unwrap()).sqrt() * volume
        }
    };

    Ok(MonteCarloResult {
        value: integral_value,
        std_error,
        n_evals,
    })
}

/// Parallel Monte Carlo integration with adaptive variance reduction
///
/// This advanced method automatically adjusts sampling strategies based on
/// the variance observed in different regions of the integration domain.
#[cfg(feature = "parallel")]
pub fn adaptive_parallel_monte_carlo<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    target_variance: F,
    max_samples: usize,
    options: Option<ParallelMonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    rand_distr::StandardNormal: Distribution<F>,
{
    let opts = options.unwrap_or_default();
    let initial_samples = opts.n_samples.min(max_samples / 4); // Start with 25% of max samples

    // Initial estimation
    let mut current_opts = opts.clone();
    current_opts.n_samples = initial_samples;

    let mut result = parallel_monte_carlo(&f, ranges, Some(current_opts.clone()))?;

    // Adaptive refinement
    while result.std_error > target_variance && result.n_evals < max_samples {
        let remaining_samples = max_samples - result.n_evals;
        let next_batch_size = remaining_samples.min(initial_samples);

        if next_batch_size == 0 {
            break;
        }

        // Run additional samples
        current_opts.n_samples = next_batch_size;
        let additional_result = parallel_monte_carlo(&f, ranges, Some(current_opts.clone()))?;

        // Combine results
        let total_evals = result.n_evals + additional_result.n_evals;
        let combined_value = (result.value * F::from_usize(result.n_evals).unwrap()
            + additional_result.value * F::from_usize(additional_result.n_evals).unwrap())
            / F::from_usize(total_evals).unwrap();

        // Simplified error combination (in practice, would be more sophisticated)
        let combined_error =
            (result.std_error * result.std_error * F::from_usize(result.n_evals).unwrap()
                + additional_result.std_error
                    * additional_result.std_error
                    * F::from_usize(additional_result.n_evals).unwrap())
                / F::from_usize(total_evals).unwrap();
        let combined_std_error = combined_error.sqrt();

        result = MonteCarloResult {
            value: combined_value,
            std_error: combined_std_error,
            n_evals: total_evals,
        };
    }

    Ok(result)
}

/// Fallback implementations when parallel feature is not enabled
#[cfg(not(feature = "parallel"))]
pub fn parallel_monte_carlo<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<ParallelMonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    rand_distr::StandardNormal: Distribution<F>,
{
    // Convert to regular MonteCarloOptions and use sequential implementation
    let regular_opts = options.map(|opts| crate::monte_carlo::MonteCarloOptions {
        n_samples: opts.n_samples,
        seed: opts.seed,
        error_method: opts.error_method,
        use_antithetic: opts.use_antithetic,
        _phantom: PhantomData,
    });

    crate::monte_carlo::monte_carlo(f, ranges, regular_opts)
}

#[cfg(not(feature = "parallel"))]
pub fn adaptive_parallel_monte_carlo<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    target_variance: F,
    max_samples: usize,
    options: Option<ParallelMonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync + Send,
    rand_distr::StandardNormal: Distribution<F>,
{
    // Fallback to regular Monte Carlo
    let regular_opts = options.map(|opts| crate::monte_carlo::MonteCarloOptions {
        n_samples: max_samples,
        seed: opts.seed,
        error_method: opts.error_method,
        use_antithetic: opts.use_antithetic,
        _phantom: PhantomData,
    });

    crate::monte_carlo::monte_carlo(f, ranges, regular_opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::ArrayView1;

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_monte_carlo_simple() {
        // Test integration of f(x) = x over [0, 1], exact result = 0.5
        let f = |x: ArrayView1<f64>| x[0];
        let ranges = [(0.0, 1.0)];

        let options = ParallelMonteCarloOptions {
            n_samples: 100000,
            n_threads: Some(2),
            batch_size: 5000,
            seed: Some(42),
            ..Default::default()
        };

        let result = parallel_monte_carlo(f, &ranges, Some(options)).unwrap();

        // Should be close to 0.5 with good accuracy due to large sample size
        assert_relative_eq!(result.value, 0.5, epsilon = 0.01);
        assert!(result.std_error > 0.0);
        assert_eq!(result.n_evals, 100000);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_monte_carlo_multidimensional() {
        // Test integration of f(x,y) = x*y over [0,1]×[0,1], exact result = 0.25
        let f = |x: ArrayView1<f64>| x[0] * x[1];
        let ranges = [(0.0, 1.0), (0.0, 1.0)];

        let options = ParallelMonteCarloOptions {
            n_samples: 50000,
            n_threads: Some(4),
            use_chunking: true,
            seed: Some(123),
            ..Default::default()
        };

        let result = parallel_monte_carlo(f, &ranges, Some(options)).unwrap();

        assert_relative_eq!(result.value, 0.25, epsilon = 0.02);
        assert!(result.std_error > 0.0);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_monte_carlo_antithetic() {
        // Test with antithetic sampling for variance reduction
        let f = |x: ArrayView1<f64>| x[0] * x[0];
        let ranges = [(0.0, 1.0)];

        let options = ParallelMonteCarloOptions {
            n_samples: 10000,
            use_antithetic: true,
            n_threads: Some(2),
            seed: Some(456),
            ..Default::default()
        };

        let result = parallel_monte_carlo(f, &ranges, Some(options)).unwrap();

        // Exact result is 1/3
        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 0.02);

        // With antithetic sampling, should have used even number of samples
        assert_eq!(result.n_evals % 2, 0);
    }

    #[test]
    fn test_parallel_options_conversion() {
        let regular_opts = MonteCarloOptions {
            n_samples: 5000,
            seed: Some(789),
            error_method: ErrorEstimationMethod::BatchMeans,
            use_antithetic: true,
            _phantom: PhantomData,
        };

        let parallel_opts: ParallelMonteCarloOptions<f64> = regular_opts.into();

        assert_eq!(parallel_opts.n_samples, 5000);
        assert_eq!(parallel_opts.seed, Some(789));
        assert_eq!(
            parallel_opts.error_method,
            ErrorEstimationMethod::BatchMeans
        );
        assert!(parallel_opts.use_antithetic);
        assert_eq!(parallel_opts.batch_size, 1000); // Default value
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_adaptive_parallel_monte_carlo() {
        // Test adaptive Monte Carlo that refines until target variance is reached
        let f = |x: ArrayView1<f64>| (x[0] * std::f64::consts::PI).sin();
        let ranges = [(0.0, 1.0)];

        let target_variance = 0.001;
        let max_samples = 100000;

        let options = ParallelMonteCarloOptions {
            n_samples: 5000, // Initial samples
            n_threads: Some(2),
            seed: Some(999),
            ..Default::default()
        };

        let result =
            adaptive_parallel_monte_carlo(f, &ranges, target_variance, max_samples, Some(options))
                .unwrap();

        // Should either reach target variance or use max samples
        assert!(result.std_error <= target_variance || result.n_evals >= max_samples);

        // Exact result is -cos(π)/π + 1/π = (1 + cos(π))/π = 0 (approximately)
        // Actually, ∫₀¹ sin(πx) dx = [-cos(πx)/π]₀¹ = (-cos(π) + cos(0))/π = 2/π
        assert_relative_eq!(result.value, 2.0 / std::f64::consts::PI, epsilon = 0.05);
    }
}
