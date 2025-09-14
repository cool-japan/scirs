//! Quasi-Monte Carlo
//!
//! This module provides functions for quasi-Monte Carlo integration,
//! following SciPy's `stats.qmc` module, with advanced sequences and
//! stratified sampling methods.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::validation::*;

/// Generate Sobol sequence
///
/// The Sobol sequence is a low-discrepancy sequence that fills space more uniformly
/// than random sampling for multi-dimensional integration and optimization.
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `d` - Dimension of each sample
/// * `scramble` - Whether to apply scrambling for better uniformity
/// * `seed` - Random seed for scrambling (if None, uses system time)
///
/// # Returns
/// * `Array2<f64>` - Matrix of shape (n, d) with samples in [0, 1]^d
#[allow(dead_code)]
pub fn sobol(n: usize, d: usize, scramble: bool, seed: Option<u64>) -> StatsResult<Array2<f64>> {
    check_positive(n, "n")?;
    check_positive(d, "d")?;

    if d > 32 {
        return Err(StatsError::InvalidArgument(
            "Dimension cannot exceed 32 for Sobol sequence".to_string(),
        ));
    }

    let mut sequence = SobolSequence::new(d, scramble, seed)?;
    sequence.generate(n)
}

/// Generate Halton sequence
///
/// The Halton sequence is a deterministic low-discrepancy sequence based on
/// prime number bases, providing good uniformity for moderate dimensions.
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `d` - Dimension of each sample
/// * `scramble` - Whether to apply scrambling
/// * `seed` - Random seed for scrambling (if None, uses system time)
///
/// # Returns
/// * `Array2<f64>` - Matrix of shape (n, d) with samples in [0, 1]^d
#[allow(dead_code)]
pub fn halton(n: usize, d: usize, scramble: bool, seed: Option<u64>) -> StatsResult<Array2<f64>> {
    check_positive(n, "n")?;
    check_positive(d, "d")?;

    if d > 100 {
        return Err(StatsError::InvalidArgument(
            "Dimension cannot exceed 100 for Halton sequence".to_string(),
        ));
    }

    let mut sequence = HaltonSequence::new(d, scramble, seed)?;
    sequence.generate(n)
}

/// Latin hypercube sampling
///
/// Generates samples that are evenly distributed across each dimension,
/// ensuring good coverage of the sample space.
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `d` - Dimension of each sample
/// * `seed` - Random seed (if None, uses system time)
///
/// # Returns
/// * `Array2<f64>` - Matrix of shape (n, d) with samples in [0, 1]^d
#[allow(dead_code)]
pub fn latin_hypercube(n: usize, d: usize, seed: Option<u64>) -> StatsResult<Array2<f64>> {
    check_positive(n, "n")?;
    check_positive(d, "d")?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let s = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            StdRng::seed_from_u64(s)
        }
    };

    let mut samples = Array2::zeros((n, d));

    for dim in 0..d {
        // Create stratified intervals
        let mut intervals: Vec<usize> = (0..n).collect();

        // Shuffle intervals
        for i in (1..n).rev() {
            let j = rng.gen_range(0..i);
            intervals.swap(i, j);
        }

        // Generate samples within each interval
        for (i, &interval) in intervals.iter().enumerate() {
            let u: f64 = rng.random();
            samples[[i, dim]] = (interval as f64 + u) / n as f64;
        }
    }

    Ok(samples)
}

/// Sobol sequence generator
pub struct SobolSequence {
    dimension: usize,
    direction_numbers: Vec<Vec<u32>>,
    current_index: usize,
    #[allow(dead_code)]
    scramble: bool,
    scramble_matrices: Option<Vec<Array2<u32>>>,
}

impl SobolSequence {
    /// Create a new Sobol sequence generator
    pub fn new(dimension: usize, scramble: bool, seed: Option<u64>) -> StatsResult<Self> {
        if dimension == 0 || dimension > 32 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be between 1 and 32".to_string(),
            ));
        }

        let direction_numbers = Self::initialize_direction_numbers(dimension)?;

        let scramble_matrices = if scramble {
            Some(Self::generate_scramble_matrices(dimension, seed)?)
        } else {
            None
        };

        Ok(Self {
            dimension,
            direction_numbers,
            current_index: 0,
            scramble,
            scramble_matrices,
        })
    }

    /// Generate n samples from the Sobol sequence
    pub fn generate(&mut self, n: usize) -> StatsResult<Array2<f64>> {
        let mut samples = Array2::zeros((n, self.dimension));

        for i in 0..n {
            let point = self.next_point()?;
            for (j, &val) in point.iter().enumerate() {
                samples[[i, j]] = val;
            }
        }

        Ok(samples)
    }

    /// Get the next point in the sequence
    pub fn next_point(&mut self) -> StatsResult<Array1<f64>> {
        let mut point = Array1::zeros(self.dimension);

        for dim in 0..self.dimension {
            let mut result = 0u32;
            let index = self.current_index;

            for bit in 0..32 {
                if (index >> bit) & 1 == 1 {
                    result ^= self.direction_numbers[dim][bit];
                }
            }

            // Apply scrambling if enabled
            if let Some(ref matrices) = self.scramble_matrices {
                result = Self::apply_scrambling(result, &matrices[dim]);
            }

            point[dim] = result as f64 / (1u64 << 32) as f64;
        }

        self.current_index += 1;
        Ok(point)
    }

    /// Initialize direction numbers for Sobol sequence
    fn initialize_direction_numbers(dimension: usize) -> StatsResult<Vec<Vec<u32>>> {
        let mut direction_numbers = vec![vec![0u32; 32]; dimension];

        // First dimension uses powers of 2
        for i in 0..32 {
            direction_numbers[0][i] = 1u32 << (31 - i);
        }

        // Additional dimensions use primitive polynomials
        // Simplified version - in practice, you'd use tabulated values
        let primitive_polynomials = [
            (1, vec![]),        // x (dimension 1, already handled)
            (2, vec![1]),       // x^2 + x + 1
            (3, vec![1, 3]),    // x^3 + x + 1
            (3, vec![2, 3]),    // x^3 + x^2 + 1
            (4, vec![1, 4]),    // x^4 + x + 1
            (4, vec![3, 4]),    // x^4 + x^3 + 1
            (4, vec![1, 2, 4]), // x^4 + x^2 + x + 1
            (4, vec![1, 3, 4]), // x^4 + x^3 + x + 1
        ];

        for dim in 1..dimension {
            let poly_idx = (dim - 1) % primitive_polynomials.len();
            let (degree, ref coeffs) = primitive_polynomials[poly_idx];

            // Initialize first few direction numbers
            for i in 0..degree {
                direction_numbers[dim][i] = (1u32 << (31 - i)) | (1u32 << (31 - degree));
            }

            // Generate remaining direction numbers using recurrence relation
            for i in degree..32 {
                let mut val = direction_numbers[dim][i - degree]
                    ^ (direction_numbers[dim][i - degree] >> degree);

                for &coeff in coeffs {
                    if coeff <= i {
                        val ^= direction_numbers[dim][i - coeff];
                    }
                }

                direction_numbers[dim][i] = val;
            }
        }

        Ok(direction_numbers)
    }

    /// Generate scrambling matrices for Owen scrambling
    fn generate_scramble_matrices(
        dimension: usize,
        seed: Option<u64>,
    ) -> StatsResult<Vec<Array2<u32>>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let s = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                StdRng::seed_from_u64(s)
            }
        };

        let mut matrices = Vec::with_capacity(dimension);

        for _ in 0..dimension {
            let mut matrix = Array2::zeros((32, 32));

            // Generate random permutation matrix for each bit level
            for i in 0..32 {
                let j = rng.gen_range(0..32);
                matrix[[i, j]] = 1;
            }

            matrices.push(matrix);
        }

        Ok(matrices)
    }

    /// Apply Owen scrambling to a value
    fn apply_scrambling(value: u32, matrix: &Array2<u32>) -> u32 {
        let mut result = 0u32;

        for i in 0..32 {
            let bit = (value >> (31 - i)) & 1;
            for j in 0..32 {
                if matrix[[i, j]] == 1 && bit == 1 {
                    result |= 1u32 << (31 - j);
                    break;
                }
            }
        }

        result
    }
}

/// Halton sequence generator  
pub struct HaltonSequence {
    dimension: usize,
    bases: Vec<u32>,
    current_index: usize,
    scramble: bool,
    permutations: Option<Vec<Vec<u32>>>,
}

impl HaltonSequence {
    /// Create a new Halton sequence generator
    pub fn new(dimension: usize, scramble: bool, seed: Option<u64>) -> StatsResult<Self> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be at least 1".to_string(),
            ));
        }

        let bases = Self::first_primes(dimension)?;

        let permutations = if scramble {
            Some(Self::generate_permutations(&bases, seed)?)
        } else {
            None
        };

        Ok(Self {
            dimension,
            bases,
            current_index: 0,
            scramble,
            permutations,
        })
    }

    /// Generate n samples from the Halton sequence
    pub fn generate(&mut self, n: usize) -> StatsResult<Array2<f64>> {
        let mut samples = Array2::zeros((n, self.dimension));

        for i in 0..n {
            let point = self.next_point()?;
            for (j, &val) in point.iter().enumerate() {
                samples[[i, j]] = val;
            }
        }

        Ok(samples)
    }

    /// Get the next point in the sequence
    pub fn next_point(&mut self) -> StatsResult<Array1<f64>> {
        let mut point = Array1::zeros(self.dimension);

        for dim in 0..self.dimension {
            let base = self.bases[dim];
            let value = if self.scramble {
                Self::scrambled_radical_inverse(
                    self.current_index,
                    base,
                    self.permutations.as_ref().unwrap()[dim].as_slice(),
                )?
            } else {
                Self::radical_inverse(self.current_index, base)?
            };

            point[dim] = value;
        }

        self.current_index += 1;
        Ok(point)
    }

    /// Compute radical inverse in given base
    fn radical_inverse(index: usize, base: u32) -> StatsResult<f64> {
        let mut result = 0.0;
        let mut fraction = 1.0 / base as f64;
        let mut i = index;

        while i > 0 {
            result += (i % base as usize) as f64 * fraction;
            i /= base as usize;
            fraction /= base as f64;
        }

        Ok(result)
    }

    /// Compute scrambled radical inverse
    fn scrambled_radical_inverse(index: usize, base: u32, permutation: &[u32]) -> StatsResult<f64> {
        let mut result = 0.0;
        let mut fraction = 1.0 / base as f64;
        let mut i = index;

        while i > 0 {
            let digit = i % base as usize;
            let scrambled_digit = permutation[digit];
            result += scrambled_digit as f64 * fraction;
            i /= base as usize;
            fraction /= base as f64;
        }

        Ok(result)
    }

    /// Generate first n prime numbers
    fn first_primes(n: usize) -> StatsResult<Vec<u32>> {
        if n == 0 {
            return Ok(vec![]);
        }

        let mut primes = Vec::with_capacity(n);
        let mut candidate = 2u32;

        while primes.len() < n {
            if Self::is_prime(candidate) {
                primes.push(candidate);
            }
            candidate += 1;
        }

        Ok(primes)
    }

    /// Check if a number is prime
    fn is_prime(n: u32) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u32;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }

        true
    }

    /// Generate random permutations for scrambling
    fn generate_permutations(bases: &[u32], seed: Option<u64>) -> StatsResult<Vec<Vec<u32>>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let s = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                StdRng::seed_from_u64(s)
            }
        };

        let mut permutations = Vec::with_capacity(bases.len());

        for &base in bases {
            let mut perm: Vec<u32> = (0..base).collect();

            // Fisher-Yates shuffle
            for i in (1..base).rev() {
                let j = rng.gen_range(0..i);
                perm.swap(i as usize, j as usize);
            }

            permutations.push(perm);
        }

        Ok(permutations)
    }
}

/// Discrepancy measures for QMC sequences
#[allow(dead_code)]
pub fn star_discrepancy(samples: &ArrayView1<Array1<f64>>) -> StatsResult<f64> {
    if samples.is_empty() {
        return Err(StatsError::InvalidArgument(
            "samples array cannot be empty".to_string(),
        ));
    }

    let n = samples.len();
    let d = samples[0].len();

    // Simplified star discrepancy calculation
    // In practice, this would use more sophisticated algorithms
    let mut max_discrepancy: f64 = 0.0;
    let num_test_points = 100; // Reduced for efficiency

    let mut rng = rand::rng();
    for _ in 0..num_test_points {
        let mut test_point = Array1::zeros(d);
        for j in 0..d {
            test_point[j] = (rng.random::<f64>() * 0.9) + 0.05; // Avoid exact boundaries
        }

        // Count points in box [0, test_point]
        let mut count = 0;
        for sample in samples.iter() {
            let mut in_box = true;
            for j in 0..d {
                if sample[j] > test_point[j] {
                    in_box = false;
                    break;
                }
            }
            if in_box {
                count += 1;
            }
        }

        // Expected volume
        let volume: f64 = test_point.iter().product();
        let expected = volume * n as f64;
        let discrepancy = (count as f64 - expected).abs() / n as f64;

        max_discrepancy = max_discrepancy.max(discrepancy);
    }

    Ok(max_discrepancy)
}

/// Advanced QMC sequences and stratified sampling
pub mod advanced;
pub mod enhanced_sequences;
pub use advanced::*;
pub use enhanced_sequences::*;
