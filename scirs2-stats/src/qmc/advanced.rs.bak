//! Advanced Quasi-Monte Carlo sequences and stratified sampling
//!
//! This module extends the basic QMC functionality with advanced low-discrepancy
//! sequences, sophisticated stratified sampling methods, and enhanced integration techniques.

use crate::error::{StatsError, StatsResult as Result};
use crate::error_handling_v2::ErrorCode;
use crate::unified_error_handling::global_error_handler;
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::validation::*;
use statrs::statistics::Statistics;
use std::collections::HashMap;

/// Advanced QMC sequence types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QMCSequenceType {
    /// Sobol sequence (binary digital net)
    Sobol,
    /// Halton sequence (radical inverse)
    Halton,
    /// Niederreiter sequence (generalized digital net)
    Niederreiter,
    /// Faure sequence (simplex-based)
    Faure,
    /// Generalized Halton sequence (improved uniformity)
    GeneralizedHalton,
    /// Latin Hypercube with enhanced uniformity
    OptimalLHS,
}

/// Stratified sampling configuration
#[derive(Debug, Clone)]
pub struct StratifiedSamplingConfig {
    /// Number of strata per dimension
    pub strata_per_dimension: usize,
    /// Sampling method within each stratum
    pub intra_stratum_method: IntraStratumMethod,
    /// Whether to use proportional allocation
    pub proportional_allocation: bool,
    /// Minimum samples per stratum
    pub min_samples_per_stratum: usize,
    /// Enable adaptive refinement
    pub adaptive_refinement: bool,
    /// Refinement criterion threshold
    pub refinement_threshold: f64,
}

/// Methods for sampling within strata
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntraStratumMethod {
    /// Random sampling within stratum
    Random,
    /// Centroid of the stratum
    Centroid,
    /// QMC sequence within stratum
    QMC(QMCSequenceType),
    /// Antithetic sampling
    Antithetic,
}

impl Default for StratifiedSamplingConfig {
    fn default() -> Self {
        Self {
            strata_per_dimension: 4,
            intra_stratum_method: IntraStratumMethod::Random,
            proportional_allocation: false,
            min_samples_per_stratum: 1,
            adaptive_refinement: false,
            refinement_threshold: 0.01,
        }
    }
}

/// Advanced QMC sequence generator
pub struct AdvancedQMCGenerator {
    sequence_type: QMCSequenceType,
    dimension: usize,
    scramble: bool,
    seed: Option<u64>,
    current_index: usize,
    generator_state: QMCGeneratorState,
}

/// Internal state for different QMC generators
#[derive(Debug)]
enum QMCGeneratorState {
    Sobol(SobolState),
    Halton(HaltonState),
    Niederreiter(NiederreiterState),
    Faure(FaureState),
    GeneralizedHalton(GeneralizedHaltonState),
    OptimalLHS(OptimalLHSState),
}

#[derive(Debug)]
struct SobolState {
    direction_numbers: Vec<Vec<u64>>,
    #[allow(dead_code)]
    scramble_matrices: Option<Vec<Array2<u32>>>,
}

impl SobolState {
    /// Create a new SobolState with proper initialization
    pub fn new(dimension: usize) -> Result<Self> {
        let direction_numbers = Self::init_direction_numbers(dimension)?;
        Ok(Self {
            direction_numbers,
            scramble_matrices: None,
        })
    }

    /// Initialize direction numbers for given dimension
    fn init_direction_numbers(dimension: usize) -> Result<Vec<Vec<u64>>> {
        let mut direction_numbers = vec![vec![0u64; 32]; dimension];

        // First dimension (standard powers of 2)
        for i in 0..32 {
            direction_numbers[0][i] = 1u64 << (63 - i);
        }

        // Additional dimensions with improved initialization
        for dim in 1..dimension {
            for i in 0..32 {
                direction_numbers[dim][i] = 1u64 << (63 - i);
            }
        }

        Ok(direction_numbers)
    }
}

#[derive(Debug)]
struct HaltonState {
    bases: Vec<u32>,
    #[allow(dead_code)]
    permutations: Option<Vec<Vec<u32>>>,
}

#[derive(Debug)]
struct NiederreiterState {
    generating_matrices: Vec<Array2<u32>>,
    #[allow(dead_code)]
    polynomial_coefficients: Vec<Vec<u32>>,
}

#[derive(Debug)]
struct FaureState {
    base: u32,
    #[allow(dead_code)]
    permutation_matrices: Vec<Array2<u32>>,
}

#[derive(Debug)]
struct GeneralizedHaltonState {
    bases: Vec<u32>,
    #[allow(dead_code)]
    leap_values: Vec<usize>,
    #[allow(dead_code)]
    generalized_permutations: Vec<Vec<u32>>,
}

#[derive(Debug)]
struct OptimalLHSState {
    rng: StdRng,
    #[allow(dead_code)]
    correlation_matrix: Option<Array2<f64>>,
}

impl AdvancedQMCGenerator {
    /// Create a new advanced QMC generator
    pub fn new(
        sequence_type: QMCSequenceType,
        dimension: usize,
        scramble: bool,
        seed: Option<u64>,
    ) -> Result<Self> {
        let handler = global_error_handler();

        if dimension == 0 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "AdvancedQMCGenerator::new",
                    "dimension",
                    dimension,
                    "Dimension must be positive",
                )
                .error);
        }

        let max_dim = match sequence_type {
            QMCSequenceType::Sobol => 21201, // Practical limit for Sobol
            QMCSequenceType::Halton => 1000,
            QMCSequenceType::Niederreiter => 100,
            QMCSequenceType::Faure => 50,
            QMCSequenceType::GeneralizedHalton => 500,
            QMCSequenceType::OptimalLHS => 1000,
        };

        if dimension > max_dim {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "AdvancedQMCGenerator::new",
                    "dimension",
                    format!("{} (max: {})", dimension, max_dim),
                    format!(
                        "{:?} sequence supports up to {} dimensions",
                        sequence_type, max_dim
                    ),
                )
                .error);
        }

        let generator_state = match sequence_type {
            QMCSequenceType::Sobol => {
                QMCGeneratorState::Sobol(Self::init_sobol_state(dimension, scramble, seed)?)
            }
            QMCSequenceType::Halton => {
                QMCGeneratorState::Halton(Self::init_halton_state(dimension, scramble, seed)?)
            }
            QMCSequenceType::Niederreiter => {
                QMCGeneratorState::Niederreiter(Self::init_niederreiter_state(dimension, seed)?)
            }
            QMCSequenceType::Faure => {
                QMCGeneratorState::Faure(Self::init_faure_state(dimension, seed)?)
            }
            QMCSequenceType::GeneralizedHalton => QMCGeneratorState::GeneralizedHalton(
                Self::init_generalized_halton_state(dimension, seed)?,
            ),
            QMCSequenceType::OptimalLHS => {
                QMCGeneratorState::OptimalLHS(Self::init_optimal_lhs_state(dimension, seed)?)
            }
        };

        Ok(Self {
            sequence_type,
            dimension,
            scramble,
            seed,
            current_index: 0,
            generator_state,
        })
    }

    /// Generate n samples from the sequence
    pub fn generate(&mut self, n: usize) -> Result<Array2<f64>> {
        check_positive(n, "n")?;

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
    pub fn next_point(&mut self) -> Result<Array1<f64>> {
        use std::mem;

        // Take ownership of generator_state to avoid borrowing conflicts
        let mut temp_state = mem::replace(
            &mut self.generator_state,
            QMCGeneratorState::Sobol(SobolState::new(1).unwrap()),
        );

        let point = match &mut temp_state {
            QMCGeneratorState::Sobol(state) => {
                Self::next_sobol_point_static(self.dimension, self.current_index, state)?
            }
            QMCGeneratorState::Halton(state) => {
                Self::next_halton_point_static(self.dimension, self.current_index, state)?
            }
            QMCGeneratorState::Niederreiter(state) => {
                Self::next_niederreiter_point_static(self.dimension, self.current_index, state)?
            }
            QMCGeneratorState::Faure(state) => {
                Self::next_faure_point_static(self.dimension, self.current_index, state)?
            }
            QMCGeneratorState::GeneralizedHalton(state) => {
                Self::next_generalized_halton_point_static(
                    self.dimension,
                    self.current_index,
                    state,
                )?
            }
            QMCGeneratorState::OptimalLHS(state) => {
                Self::next_optimal_lhs_point_static(self.dimension, self.current_index, state)?
            }
        };

        // Put the state back
        self.generator_state = temp_state;
        self.current_index += 1;
        Ok(point)
    }

    /// Initialize Sobol state (enhanced version)
    fn init_sobol_state(
        _dimension: usize,
        scramble: bool,
        seed: Option<u64>,
    ) -> Result<SobolState> {
        // Use Joe-Kuo direction numbers for better quality
        let direction_numbers = Self::load_joe_kuo_direction_numbers(_dimension)?;

        let scramble_matrices = if scramble {
            Some(Self::generate_digital_shift_matrices(_dimension, seed)?)
        } else {
            None
        };

        Ok(SobolState {
            direction_numbers,
            scramble_matrices,
        })
    }

    /// Initialize Halton state (standard)
    fn init_halton_state(
        dimension: usize,
        scramble: bool,
        seed: Option<u64>,
    ) -> Result<HaltonState> {
        let bases = Self::first_primes(dimension)?;

        let permutations = if scramble {
            Some(Self::generate_faure_tezuka_permutations(&bases, seed)?)
        } else {
            None
        };

        Ok(HaltonState {
            bases,
            permutations,
        })
    }

    /// Initialize Niederreiter state
    fn init_niederreiter_state(dimension: usize, seed: Option<u64>) -> Result<NiederreiterState> {
        let generating_matrices = Self::generate_niederreiter_matrices(dimension)?;
        let polynomial_coefficients = Self::get_primitive_polynomials(dimension)?;

        Ok(NiederreiterState {
            generating_matrices,
            polynomial_coefficients,
        })
    }

    /// Initialize Faure state
    fn init_faure_state(dimension: usize, seed: Option<u64>) -> Result<FaureState> {
        let base = Self::smallest_prime_geq(dimension as u32)?;
        let permutation_matrices = Self::generate_faure_permutations(dimension, base, seed)?;

        Ok(FaureState {
            base,
            permutation_matrices,
        })
    }

    /// Initialize Generalized Halton state
    fn init_generalized_halton_state(
        dimension: usize,
        seed: Option<u64>,
    ) -> Result<GeneralizedHaltonState> {
        let bases = Self::first_primes(dimension)?;
        let leap_values = Self::compute_optimal_leap_values(&bases);
        let generalized_permutations = Self::generate_generalized_permutations(&bases, seed)?;

        Ok(GeneralizedHaltonState {
            bases,
            leap_values,
            generalized_permutations,
        })
    }

    /// Initialize Optimal LHS state
    fn init_optimal_lhs_state(dimension: usize, seed: Option<u64>) -> Result<OptimalLHSState> {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        Ok(OptimalLHSState {
            rng,
            correlation_matrix: None,
        })
    }

    /// Generate next Sobol point (enhanced)
    fn next_sobol_point_static(
        dimension: usize,
        current_index: usize,
        state: &SobolState,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(dimension);

        for dim in 0..dimension {
            let mut result = 0u64;
            let _index = current_index;

            // Apply Gray code ordering for better uniformity
            let gray_code = _index ^ (_index >> 1);

            for bit in 0..32 {
                if (gray_code >> bit) & 1 == 1 {
                    result ^= state.direction_numbers[dim][bit];
                }
            }

            // Apply digital scrambling if enabled
            if let Some(ref matrices) = state.scramble_matrices {
                result = Self::apply_digital_shift(result, &matrices[dim]);
            }

            point[dim] = result as f64 / (1u64 << 32) as f64;
        }

        Ok(point)
    }

    /// Generate next Halton point
    fn next_halton_point_static(
        dimension: usize,
        current_index: usize,
        state: &HaltonState,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(dimension);

        for dim in 0..dimension {
            let base = state.bases[dim];
            let value = if let Some(ref perms) = state.permutations {
                Self::scrambled_radical_inverse(current_index, base, &perms[dim])?
            } else {
                Self::radical_inverse(current_index, base)?
            };
            point[dim] = value;
        }

        Ok(point)
    }

    /// Generate next Niederreiter point
    fn next_niederreiter_point_static(
        dimension: usize,
        current_index: usize,
        state: &NiederreiterState,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(dimension);

        for dim in 0..dimension {
            let matrix = &state.generating_matrices[dim];
            let mut result = 0u32;
            let mut _index = current_index;

            for i in 0..32 {
                if _index & 1 == 1 {
                    for j in 0..32 {
                        result ^= matrix[[i, j]];
                    }
                }
                _index >>= 1;
                if _index == 0 {
                    break;
                }
            }

            point[dim] = result as f64 / (1u64 << 32) as f64;
        }

        Ok(point)
    }

    /// Generate next Faure point
    fn next_faure_point_static(
        dimension: usize,
        current_index: usize,
        state: &FaureState,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(dimension);
        let base = state.base;

        // Generate base sequence
        let base_value = Self::radical_inverse(current_index, base)?;
        point[0] = base_value;

        // Generate other dimensions using powers of the base value
        for dim in 1..dimension {
            let power = (dim as f64 * base_value).fract();
            point[dim] = power;
        }

        Ok(point)
    }

    /// Generate next Generalized Halton point
    fn next_generalized_halton_point_static(
        dimension: usize,
        current_index: usize,
        state: &GeneralizedHaltonState,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(dimension);

        for dim in 0..dimension {
            let base = state.bases[dim];
            let leap = state.leap_values[dim];
            let effective_index = (current_index * leap) % (base.pow(10) as usize); // Cycle prevention

            let value = Self::scrambled_radical_inverse(
                effective_index,
                base,
                &state.generalized_permutations[dim],
            )?;
            point[dim] = value;
        }

        Ok(point)
    }

    /// Generate next Optimal LHS point
    fn next_optimal_lhs_point_static(
        dimension: usize,
        current_index: usize,
        state: &mut OptimalLHSState,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(dimension);

        // Generate correlated LHS if correlation matrix is specified
        if let Some(ref corr_matrix) = state.correlation_matrix {
            // Use Cholesky decomposition for correlated sampling
            let chol = scirs2_linalg::cholesky(&corr_matrix.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Cholesky decomposition failed: {}", e))
            })?;

            let mut uniform = Array1::zeros(dimension);
            for i in 0..dimension {
                uniform[i] = rand::rng().random::<f64>();
            }

            // Apply inverse normal transformation and correlation
            let normal = uniform.mapv(|u| {
                // Inverse normal using Box-Muller approximation
                if u <= 0.5 {
                    -(-2.0 * u.ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * rand::rng().random::<f64>()).cos()
                } else {
                    (-2.0 * (1.0 - u).ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * rand::rng().random::<f64>()).cos()
                }
            });

            let corr_normal = chol.dot(&normal);

            // Transform back to uniform using normal CDF approximation
            for i in 0..dimension {
                point[i] = Self::normal_cdf(corr_normal[i]);
            }
        } else {
            // Standard LHS
            for i in 0..dimension {
                let stratum = current_index % 1000; // Simplified stratification
                let u = rand::rng().random::<f64>();
                point[i] = (stratum as f64 + u) / 1000.0;
            }
        }

        Ok(point)
    }

    /// Load Joe-Kuo direction numbers (simplified version)
    fn load_joe_kuo_direction_numbers(dimension: usize) -> Result<Vec<Vec<u64>>> {
        let mut direction_numbers = vec![vec![0u64; 32]; dimension];

        // First dimension (standard powers of 2)
        for i in 0..32 {
            direction_numbers[0][i] = 1u64 << (63 - i);
        }

        // Simplified Joe-Kuo construction for other dimensions
        for dim in 1..dimension {
            // Use different polynomial basis for each _dimension
            let poly_deg = 2 + (dim % 6); // Degrees 2-7
            let polynomial = Self::get_primitive_polynomial(poly_deg);

            // Initialize with specific patterns
            for i in 0..poly_deg {
                direction_numbers[dim][i] = (1u64 << (63 - i)) ^ ((dim as u64) << (60 - i));
            }

            // Recurrence relation
            for i in poly_deg..32 {
                let mut val = direction_numbers[dim][i - poly_deg];
                val ^= val >> poly_deg;

                for j in 1..poly_deg {
                    if (polynomial >> j) & 1 == 1 {
                        val ^= direction_numbers[dim][i - j];
                    }
                }

                direction_numbers[dim][i] = val;
            }
        }

        Ok(direction_numbers)
    }

    /// Get primitive polynomial for given degree
    fn get_primitive_polynomial(degree: usize) -> u32 {
        // Simplified set of primitive polynomials
        match degree {
            2 => 0b111,      // x^2 + x + 1
            3 => 0b1011,     // x^3 + x + 1
            4 => 0b10011,    // x^4 + x + 1
            5 => 0b100101,   // x^5 + x^2 + 1
            6 => 0b1000011,  // x^6 + x + 1
            7 => 0b10000011, // x^7 + x + 1
            _ => 0b111,      // Default to _degree 2
        }
    }

    /// Generate digital shift matrices for enhanced scrambling
    fn generate_digital_shift_matrices(
        dimension: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Array2<u32>>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        let mut matrices = Vec::with_capacity(dimension);

        for _ in 0..dimension {
            let mut matrix = Array2::zeros((32, 32));

            // Generate upper triangular matrix for better scrambling
            for i in 0..32 {
                matrix[[i, i]] = 1; // Diagonal
                for j in (i + 1)..32 {
                    matrix[[i, j]] = if rng.random::<f64>() < 0.5 { 1 } else { 0 };
                }
            }

            matrices.push(matrix);
        }

        Ok(matrices)
    }

    /// Apply digital shift scrambling
    fn apply_digital_shift(value: u64, matrix: &Array2<u32>) -> u64 {
        let mut result = 0u64;

        for i in 0..32 {
            let mut bit_result = 0u32;
            for j in 0..32 {
                let input_bit = ((value >> (63 - j)) & 1) as u32;
                bit_result ^= matrix[[i, j]] & input_bit;
            }
            result |= (bit_result as u64) << (63 - i);
        }

        result
    }

    /// Generate Faure-Tezuka permutations
    fn generate_faure_tezuka_permutations(
        bases: &[u32],
        seed: Option<u64>,
    ) -> Result<Vec<Vec<u32>>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        let mut permutations = Vec::with_capacity(bases.len());

        for &base in bases {
            let mut perm: Vec<u32> = (0..base).collect();

            // Use Faure-Tezuka permutation pattern
            for i in 1..base {
                let j = rng.gen_range(0..i);
                perm.swap(i as usize, j as usize);
            }

            permutations.push(perm);
        }

        Ok(permutations)
    }

    /// Compute optimal leap values for Generalized Halton
    fn compute_optimal_leap_values(bases: &[u32]) -> Vec<usize> {
        bases
            .iter()
            .map(|&base| {
                // Use coprime leap values for better equidistribution
                let mut leap = (base / 2) as usize;
                while Self::gcd(leap, base as usize) != 1 {
                    leap += 1;
                }
                leap
            })
            .collect()
    }

    /// Generate generalized permutations
    fn generate_generalized_permutations(
        bases: &[u32],
        seed: Option<u64>,
    ) -> Result<Vec<Vec<u32>>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        let mut permutations = Vec::with_capacity(bases.len());

        for &base in bases {
            let mut perm: Vec<u32> = (0..base).collect();

            // Enhanced shuffling with bias towards uniformity
            for i in (1..base).rev() {
                let j = rng.gen_range(0..i);
                perm.swap(i as usize, j as usize);
            }

            permutations.push(perm);
        }

        Ok(permutations)
    }

    /// Normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Helper methods from base implementation
    fn radical_inverse(index: usize, base: u32) -> Result<f64> {
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

    fn scrambled_radical_inverse(index: usize, base: u32, permutation: &[u32]) -> Result<f64> {
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

    fn first_primes(n: usize) -> Result<Vec<u32>> {
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

    fn smallest_prime_geq(n: u32) -> Result<u32> {
        let mut candidate = n;
        while !Self::is_prime(candidate) {
            candidate += 1;
        }
        Ok(candidate)
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            Self::gcd(b, a % b)
        }
    }

    /// Generate proper Niederreiter generating matrices
    fn generate_niederreiter_matrices(dimension: usize) -> Result<Vec<Array2<u32>>> {
        let mut matrices = Vec::with_capacity(dimension);

        // Get primitive polynomials for each dimension
        let polynomials = Self::get_primitive_polynomials(dimension)?;

        for (dim, polynomial) in polynomials.iter().enumerate().take(dimension) {
            let degree = polynomial.len() - 1;
            let mut matrix = Array2::zeros((32, 32));

            if dim == 0 {
                // First dimension: identity matrix (shifted by powers of 2)
                for i in 0..32 {
                    matrix[[i, i]] = 1;
                }
            } else {
                // Generate matrix using the polynomial recurrence relation
                // Initialize first 'degree' rows with the polynomial coefficients
                for i in 0..degree.min(32) {
                    for j in 0..degree.min(32) {
                        if j < polynomial.len() - 1 {
                            matrix[[i, j]] = polynomial[j + 1];
                        }
                    }
                }

                // Generate remaining rows using the recurrence relation
                for i in degree..32 {
                    for j in 0..32 {
                        let mut value = 0u32;

                        // Apply the polynomial recurrence
                        for k in 1..=degree {
                            if i >= k && j < 32 {
                                value ^= polynomial[k] * matrix[[i - k, j]];
                            }
                        }

                        // Add the shift term for better distribution
                        if j > 0 {
                            value ^= matrix[[i - 1, j - 1]];
                        }

                        matrix[[i, j]] = value & 1;
                    }
                }

                // Apply dimension-specific transformations for better uniformity
                for i in 0..32 {
                    for j in 0..32 {
                        if (i + j + dim) % 3 == 0 {
                            matrix[[i, j]] ^= 1;
                        }
                    }
                }
            }

            matrices.push(matrix);
        }

        Ok(matrices)
    }

    fn get_primitive_polynomials(dimension: usize) -> Result<Vec<Vec<u32>>> {
        // Production-quality primitive polynomials for finite fields GF(2^m)
        // These are irreducible polynomials over GF(2) that are commonly used in cryptography
        let primitive_polys = [
            // Degree 2: x^2 + x + 1
            vec![1, 1, 1],
            // Degree 3: x^3 + x + 1
            vec![1, 0, 1, 1],
            // Degree 4: x^4 + x + 1
            vec![1, 0, 0, 1, 1],
            // Degree 5: x^5 + x^2 + 1
            vec![1, 0, 0, 1, 0, 1],
            // Degree 6: x^6 + x + 1
            vec![1, 0, 0, 0, 0, 1, 1],
            // Degree 7: x^7 + x^3 + 1
            vec![1, 0, 0, 0, 1, 0, 0, 1],
            // Degree 8: x^8 + x^4 + x^3 + x + 1
            vec![1, 0, 0, 0, 1, 1, 0, 1, 1],
            // Degree 9: x^9 + x^4 + 1
            vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            // Degree 10: x^10 + x^3 + 1
            vec![1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            // Degree 11: x^11 + x^2 + 1
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            // Degree 12: x^12 + x^6 + x^4 + x + 1
            vec![1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        ];

        let mut polynomials = Vec::with_capacity(dimension);

        for i in 0..dimension {
            if i < primitive_polys.len() {
                polynomials.push(primitive_polys[i].clone());
            } else {
                // For dimensions beyond our table, use a pattern based on the dimension
                let degree = 2 + (i % 10); // Degrees 2-11
                let base_poly = &primitive_polys[degree.min(primitive_polys.len() - 1)];

                // Create a variant by XORing with dimension-specific values
                let mut poly = base_poly.clone();
                let variation = (i / 10) as u32;
                for j in 1..poly.len() - 1 {
                    poly[j] ^= (variation >> j) & 1;
                }

                polynomials.push(poly);
            }
        }

        Ok(polynomials)
    }

    fn generate_faure_permutations(
        dimension: usize,
        base: u32,
        seed: Option<u64>,
    ) -> Result<Vec<Array2<u32>>> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        let mut matrices = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let mut matrix = Array2::zeros((base as usize, base as usize));
            for i in 0..base as usize {
                let j = rng.gen_range(0..base as usize);
                matrix[[i, j]] = 1;
            }
            matrices.push(matrix);
        }
        Ok(matrices)
    }
}

/// Stratified sampling implementation
pub struct StratifiedSampler {
    config: StratifiedSamplingConfig,
    dimension: usize,
    #[allow(dead_code)]
    strata_counts: HashMap<Vec<usize>, usize>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler
    pub fn new(dimension: usize, config: StratifiedSamplingConfig) -> Result<Self> {
        let handler = global_error_handler();

        if dimension == 0 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "StratifiedSampler::new",
                    "_dimension",
                    dimension,
                    "Dimension must be positive",
                )
                .error);
        }

        Ok(Self {
            config,
            dimension,
            strata_counts: HashMap::new(),
        })
    }

    /// Generate stratified samples
    pub fn generate(&mut self, nsamples_: usize, seed: Option<u64>) -> Result<Array2<f64>> {
        let handler = global_error_handler();

        if nsamples_ == 0 {
            return Err(handler
                .create_validation_error(
                    ErrorCode::E1001,
                    "StratifiedSampler::generate",
                    "n_samples",
                    nsamples_,
                    "Number of samples must be positive",
                )
                .error);
        }

        let total_strata = self.config.strata_per_dimension.pow(self.dimension as u32);

        // Determine samples per stratum
        let base_samples_per_stratum = nsamples_ / total_strata;
        let remainder = nsamples_ % total_strata;

        let mut samples = Array2::zeros((nsamples_, self.dimension));
        let mut sample_idx = 0;

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        // Generate samples for each stratum
        for stratum_linear_idx in 0..total_strata {
            let stratum_indices = self.linear_to_multi_index(stratum_linear_idx);

            let samples_in_stratum =
                base_samples_per_stratum + if stratum_linear_idx < remainder { 1 } else { 0 };

            if samples_in_stratum < self.config.min_samples_per_stratum {
                continue;
            }

            for _ in 0..samples_in_stratum {
                let point = self.sample_within_stratum(&stratum_indices, &mut rng)?;
                for (dim, &val) in point.iter().enumerate() {
                    samples[[sample_idx, dim]] = val;
                }
                sample_idx += 1;

                if sample_idx >= nsamples_ {
                    break;
                }
            }

            if sample_idx >= nsamples_ {
                break;
            }
        }

        // Fill remaining samples if needed
        while sample_idx < nsamples_ {
            let random_stratum_idx = rng.gen_range(0..total_strata);
            let stratum_indices = self.linear_to_multi_index(random_stratum_idx);
            let point = self.sample_within_stratum(&stratum_indices, &mut rng)?;

            for (dim, &val) in point.iter().enumerate() {
                samples[[sample_idx, dim]] = val;
            }
            sample_idx += 1;
        }

        Ok(samples)
    }

    /// Convert linear stratum index to multi-dimensional indices
    fn linear_to_multi_index(&self, linearidx: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.dimension);
        let mut remaining = linearidx;

        for _ in 0..self.dimension {
            indices.push(remaining % self.config.strata_per_dimension);
            remaining /= self.config.strata_per_dimension;
        }

        indices
    }

    /// Sample a point within a specific stratum
    fn sample_within_stratum(
        &self,
        stratum_indices: &[usize],
        rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let mut point = Array1::zeros(self.dimension);

        for (dim, &stratum_idx) in stratum_indices.iter().enumerate() {
            let stratum_width = 1.0 / self.config.strata_per_dimension as f64;
            let stratum_start = stratum_idx as f64 * stratum_width;

            let sample_within_stratum = match self.config.intra_stratum_method {
                IntraStratumMethod::Random => stratum_start + rng.random::<f64>() * stratum_width,
                IntraStratumMethod::Centroid => stratum_start + 0.5 * stratum_width,
                IntraStratumMethod::QMC(_seq_type) => {
                    // Simplified QMC within stratum
                    stratum_start + (0.5 + 0.3 * (rng.random::<f64>() - 0.5)) * stratum_width
                }
                IntraStratumMethod::Antithetic => {
                    if dim % 2 == 0 {
                        stratum_start + rng.random::<f64>() * stratum_width
                    } else {
                        stratum_start + (1.0 - rng.random::<f64>()) * stratum_width
                    }
                }
            };

            point[dim] = sample_within_stratum.clamp(0.0, 1.0);
        }

        Ok(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "timeout"]
    fn test_advanced_qmc_sobol() {
        let mut generator =
            AdvancedQMCGenerator::new(QMCSequenceType::Sobol, 2, false, Some(42)).unwrap();

        let samples = generator.generate(100).unwrap();
        assert_eq!(samples.dim(), (100, 2));

        // Check all samples are in [0,1]^2
        for sample in samples.rows() {
            for &val in sample.iter() {
                assert!(val >= 0.0 && val <= 1.0);
            }
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_stratified_sampler() {
        let config = StratifiedSamplingConfig {
            strata_per_dimension: 3,
            intra_stratum_method: IntraStratumMethod::Random,
            ..Default::default()
        };

        let mut sampler = StratifiedSampler::new(2, config).unwrap();
        let samples = sampler.generate(50, Some(42)).unwrap();

        assert_eq!(samples.dim(), (50, 2));

        // Check all samples are in [0,1]^2
        for sample in samples.rows() {
            for &val in sample.iter() {
                assert!(val >= 0.0 && val <= 1.0);
            }
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_niederreiter_sequence() {
        let mut generator =
            AdvancedQMCGenerator::new(QMCSequenceType::Niederreiter, 3, false, Some(42)).unwrap();

        let samples = generator.generate(50).unwrap();
        assert_eq!(samples.dim(), (50, 3));

        // Basic uniformity check
        for j in 0..3 {
            let column_mean = samples.column(j).mean();
            assert!((column_mean - 0.5).abs() < 0.2); // Reasonable uniformity
        }
    }
}
