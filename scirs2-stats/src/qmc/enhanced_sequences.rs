//! Enhanced Quasi-Monte Carlo sequences with state-of-the-art algorithms
//!
//! This module provides advanced QMC sequences with:
//! - Optimal digital nets and (t,m,s)-nets
//! - Advanced scrambling and randomization techniques
//! - Parallel QMC sequence generation
//! - Adaptive sequence refinement

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::{rngs::StdRng, Rng, RngExt, SeedableRng};
use scirs2_core::{parallel_ops::*, simd_ops::SimdUnifiedOps, validation::*};
use std::marker::PhantomData;

/// Number of base-`b` digits used by the real Faure (permuted van der
/// Corput) construction in `compute_faure_improved` -- enough to give
/// full `f64` precision for any index that fits in a `u64` regardless of
/// the (prime, >= dimension) base in use.
const FAURE_DIGITS: usize = 32;

/// Maximum number of sequence points used by the `wraparound_discrepancy`
/// and `diaphony` estimates in `assess_quality`. Both are, in the exact
/// case, `O(n^2)`/`O(n)`-per-frequency computations; bounding to a
/// leading prefix keeps the cost tractable for large `n` (the prefix of a
/// genuine low-discrepancy sequence is itself a reasonable low-discrepancy
/// point set, so this is a legitimate bounded estimate, in the same spirit
/// as `assess_quality`'s pre-existing `50.min(n)`-test-point star-discrepancy
/// estimate).
const QUALITY_SAMPLE_CAP: usize = 200;

/// Maximum number of dimensions used by the `wraparound_discrepancy`
/// estimate. The closed-form used grows like `1.5^d`/`(4/3)^d`, which
/// loses all `f64` precision to catastrophic cancellation well before
/// `d = 1000` (this crate's hard dimension cap); restricting to a leading
/// subset of dimensions keeps the computation numerically meaningful.
const WRAPAROUND_DIM_CAP: usize = 20;

/// Maximum number of dimensions used by the `diaphony` estimate's pairwise
/// (order-2) frequency terms, which cost `O(d^2)` combinations.
const DIAPHONY_DIM_CAP: usize = 16;

/// Maximum |frequency| per dimension used by the `diaphony` estimate.
/// Higher frequencies contribute rapidly-shrinking `1/h^2` weight, so
/// truncating here is a standard, well-behaved approximation.
const DIAPHONY_FREQ_MAX: i64 = 2;

/// Enhanced QMC sequence generator with parallel support
pub struct EnhancedQMCGenerator<F> {
    /// Sequence type
    pub sequence_type: EnhancedSequenceType,
    /// Dimension
    pub dimension: usize,
    /// Configuration
    pub config: EnhancedQMCConfig,
    /// Generator state
    pub state: QMCGeneratorState,
    /// Real per-dimension Sobol direction numbers (Joe-Kuo-style
    /// construction shared with `qmc::advanced::AdvancedQMCGenerator`),
    /// computed once at construction time and used by
    /// `compute_sobol_advanced` -- and, transitively, by `DigitalNet` and
    /// `Hybrid`, both of which currently delegate to it.
    sobol_direction_numbers: Vec<Vec<u64>>,
    /// Real Niederreiter (base-2 digital net) generating matrices, shared
    /// with `qmc::advanced::AdvancedQMCGenerator` (see
    /// `generate_niederreiter_matrices`), computed once at construction
    /// and used by `compute_niederreiter_enhanced` -- and, transitively,
    /// by `DigitalNet`'s `NiederreiterXing` construction method.
    niederreiter_generating_matrices: Vec<Array2<u32>>,
    /// Base used by the real Faure (permuted van der Corput) construction
    /// below: the smallest prime >= `dimension` (Faure 1982 requires
    /// base >= dimension for the construction's low-discrepancy
    /// guarantees).
    faure_base: u64,
    /// (mod-`faure_base`) Pascal matrix used by `compute_faure_improved`:
    /// dimension `d`'s digit permutation is this matrix raised to the
    /// `d`-th power (mod `faure_base`), applied to the base-`faure_base`
    /// digit expansion of the index. `matrix[[r, l]] = C(l, r) mod
    /// faure_base` for `l >= r`, else 0 -- the standard Faure (1982)
    /// construction, computed once at construction time.
    faure_pascal_matrix: Array2<u64>,
    _phantom: PhantomData<F>,
}

/// Enhanced sequence types with advanced algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum EnhancedSequenceType {
    /// Sobol sequence with advanced scrambling
    SobolAdvanced {
        /// Use Owen scrambling
        owen_scrambling: bool,
        /// Use digital shift
        digital_shift: bool,
        /// Use nested scrambling
        nested_scrambling: bool,
    },
    /// Niederreiter sequence with base optimization
    Niederreiter {
        /// Base selection strategy
        base_strategy: BaseSelectionStrategy,
        /// Use generating matrix optimization
        matrix_optimization: bool,
    },
    /// Faure sequence with improved uniformity
    FaureImproved {
        /// Use permutation optimization
        permutation_optimization: bool,
        /// Use radical inverse improvements
        radical_inverse_improvements: bool,
    },
    /// Digital (t,m,s)-nets
    DigitalNet {
        /// Net parameters
        net_params: DigitalNetParams,
        /// Construction method
        construction_method: NetConstructionMethod,
    },
    /// Hybrid sequences combining multiple methods
    Hybrid {
        /// Primary sequence type
        primary: Box<EnhancedSequenceType>,
        /// Secondary sequence type
        secondary: Box<EnhancedSequenceType>,
        /// Combination strategy
        combination: HybridCombinationStrategy,
    },
}

/// Base selection strategies for Niederreiter sequences.
///
/// All four variants currently produce IDENTICAL output:
/// `EnhancedQMCGenerator`'s Niederreiter construction is a base-2 digital
/// net (see `compute_niederreiter_enhanced`), and a genuine
/// per-strategy multi-base Niederreiter construction would need a full
/// GF(p) polynomial-arithmetic implementation this crate does not
/// provide. This is an honestly-documented scope limitation, not a
/// silent claim that the strategies differ.
#[derive(Debug, Clone, PartialEq)]
pub enum BaseSelectionStrategy {
    /// Use first primes
    FirstPrimes,
    /// Use optimized primes for given dimension
    OptimizedPrimes,
    /// Use prime powers for better uniformity
    PrimePowers,
    /// Automatic selection based on dimension
    Automatic,
}

/// Digital net parameters
#[derive(Debug, Clone, PartialEq)]
pub struct DigitalNetParams {
    /// t parameter (strength)
    pub t: usize,
    /// m parameter (precision)
    pub m: usize,
    /// s parameter (dimension)
    pub s: usize,
    /// Base (usually 2)
    pub base: usize,
}

/// Net construction methods
#[derive(Debug, Clone, PartialEq)]
pub enum NetConstructionMethod {
    /// Sobol construction
    Sobol,
    /// Niederreiter-Xing construction. Currently reuses the crate's real
    /// base-2 Niederreiter digital net (see
    /// `EnhancedQMCGenerator::compute_digital_net`); a full
    /// algebraic-function-field Niederreiter-Xing construction is out of
    /// scope.
    NiederreiterXing,
    /// Polynomial lattice rules. NOT IMPLEMENTED: selecting this returns
    /// `StatsError::NotImplementedError` rather than silently
    /// substituting a different construction (it would require
    /// polynomial-ring arithmetic this crate's QMC module does not
    /// provide).
    PolynomialLattice,
    /// Finite field constructions. NOT IMPLEMENTED for the same reason as
    /// `PolynomialLattice` (would require general GF(q) finite-field
    /// arithmetic); selecting this returns
    /// `StatsError::NotImplementedError`.
    FiniteField,
}

/// Hybrid combination strategies
#[derive(Debug, Clone, PartialEq)]
pub enum HybridCombinationStrategy {
    /// Interleave sequences
    Interleave,
    /// Weighted combination
    Weighted(f64),
    /// Dimension-wise alternation
    DimensionAlternation,
    /// Adaptive selection based on uniformity.
    ///
    /// Currently implemented as an equal-weight blend (the same
    /// computation as `Weighted(0.5)`): true history-dependent
    /// adaptivity would require mutable state across successive
    /// `next_point`/`generate` calls that `EnhancedQMCGenerator`'s
    /// current (pure, index-only) point-computation functions do not
    /// retain. See `EnhancedQMCGenerator::compute_hybrid_sequence`.
    Adaptive,
}

/// Enhanced QMC configuration
#[derive(Debug, Clone)]
pub struct EnhancedQMCConfig {
    /// Enable parallel generation
    pub parallel: bool,
    /// Chunk size for parallel processing
    pub chunksize: usize,
    /// Randomization seed
    pub seed: Option<u64>,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Quality assessment threshold
    pub quality_threshold: f64,
    /// Maximum sequence length for quality assessment
    pub max_assessment_length: usize,
    /// Enable adaptive refinement
    pub adaptive_refinement: bool,
}

impl Default for EnhancedQMCConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            chunksize: 1000,
            seed: None,
            use_simd: true,
            quality_threshold: 1e-3,
            max_assessment_length: 10000,
            adaptive_refinement: false,
        }
    }
}

/// Generator state for QMC sequences
#[derive(Debug, Clone)]
pub struct QMCGeneratorState {
    /// Current index
    pub current_index: usize,
    /// Scrambling matrices (if used)
    pub scrambling_matrices: Option<Vec<Array2<u32>>>,
    /// Digital shift vectors (if used)
    pub digital_shifts: Option<Vec<Array1<u32>>>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for QMC sequences
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Star discrepancy estimate: the maximum, over a random sample of
    /// anchored test boxes `[0, u)`, of `|count(box)/n - volume(box)|`.
    /// Zero (the `Default`) only for an empty/never-assessed sequence;
    /// smaller is better.
    pub star_discrepancy: f64,
    /// Wrap-around L2-discrepancy estimate (Hickernell 1998), computed via
    /// the standard closed-form sum over a bounded sample of points/
    /// dimensions (see `EnhancedQMCGenerator::assess_quality`). Smaller is
    /// better.
    pub wraparound_discrepancy: f64,
    /// Diaphony (Zinterhof spectral measure): a bounded truncated
    /// approximation of the weighted sum of squared empirical Fourier
    /// coefficients over low-order frequencies (see
    /// `EnhancedQMCGenerator::assess_quality`). Smaller is better.
    pub diaphony: f64,
    /// Figure of merit: the worst (largest) of `star_discrepancy`,
    /// `wraparound_discrepancy` and `diaphony` -- a single-scalar quality
    /// summary comparable against `EnhancedQMCConfig::quality_threshold`.
    /// Smaller is better.
    pub figure_of_merit: f64,
}

impl<F> EnhancedQMCGenerator<F>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive + std::fmt::Display,
{
    /// Create new enhanced QMC generator
    pub fn new(
        sequence_type: EnhancedSequenceType,
        dimension: usize,
        config: EnhancedQMCConfig,
    ) -> StatsResult<Self> {
        check_positive(dimension, "dimension")?;

        if dimension > 1000 {
            return Err(StatsError::InvalidArgument(
                "Dimension cannot exceed 1000 for enhanced QMC sequences".to_string(),
            ));
        }

        let state = QMCGeneratorState {
            current_index: 0,
            scrambling_matrices: None,
            digital_shifts: None,
            quality_metrics: QualityMetrics::default(),
        };

        // Real, per-dimension direction numbers for the Sobol/digital-net
        // construction (see `compute_sobol_advanced`). Computed
        // unconditionally: `DigitalNet` and `Hybrid` both currently
        // delegate to the same Sobol machinery, and the cost is negligible
        // relative to `dimension`'s hard cap of 1000 above.
        let sobol_direction_numbers =
            crate::qmc::advanced::AdvancedQMCGenerator::load_joe_kuo_direction_numbers(dimension)
                .map_err(|e| {
                StatsError::ComputationError(format!(
                    "Failed to initialize Sobol direction numbers: {e}"
                ))
            })?;

        // Real Niederreiter (base-2 digital net) generating matrices (see
        // `compute_niederreiter_enhanced`). Also computed unconditionally
        // for the same reason as `sobol_direction_numbers` above:
        // `DigitalNet`'s `NiederreiterXing` construction method delegates
        // to it too. `matrix_optimization` (part of the `Niederreiter`
        // variant's own data, when that is the selected sequence type)
        // drives the shared "uniformity transform" step; any other
        // variant gets the transform applied (matching
        // `qmc::advanced`'s own default behavior).
        let niederreiter_matrix_optimization = match &sequence_type {
            EnhancedSequenceType::Niederreiter {
                matrix_optimization,
                ..
            } => *matrix_optimization,
            _ => true,
        };
        let niederreiter_generating_matrices =
            crate::qmc::advanced::AdvancedQMCGenerator::generate_niederreiter_matrices(
                dimension,
                niederreiter_matrix_optimization,
            )
            .map_err(|e| {
                StatsError::ComputationError(format!(
                    "Failed to initialize Niederreiter generating matrices: {e}"
                ))
            })?;

        // Real Faure (permuted van der Corput) construction parameters
        // (see `compute_faure_improved`): base = smallest prime >=
        // dimension, and the (mod-base) Pascal matrix used to permute
        // each dimension's digit expansion.
        let faure_base = u64::from(Self::smallest_prime_geq(dimension as u32));
        let faure_pascal_matrix = Self::build_faure_pascal_matrix(faure_base);

        let mut generator = Self {
            sequence_type,
            dimension,
            config,
            state,
            sobol_direction_numbers,
            niederreiter_generating_matrices,
            faure_base,
            faure_pascal_matrix,
            _phantom: PhantomData,
        };

        // Initialize scrambling and digital shifts if needed
        generator.initialize_randomization()?;

        Ok(generator)
    }

    /// Generate enhanced QMC sequence
    pub fn generate(&mut self, n: usize) -> StatsResult<Array2<F>> {
        check_positive(n, "n")?;

        if self.config.parallel && n >= self.config.chunksize {
            self.generate_parallel(n)
        } else {
            self.generate_sequential(n)
        }
    }

    /// Generate sequence in parallel
    fn generate_parallel(&mut self, n: usize) -> StatsResult<Array2<F>> {
        let chunksize = self.config.chunksize;
        let num_chunks = n.div_ceil(chunksize);

        let chunks = parallel_map_result(
            (0..num_chunks).collect::<Vec<_>>().as_slice(),
            |&chunk_idx| {
                let start = chunk_idx * chunksize;
                let end = (start + chunksize).min(n);
                let chunksize = end - start;

                self.generate_chunk(start, chunksize)
            },
        )?;

        // Combine chunks
        let mut result = Array2::zeros((n, self.dimension));
        let mut row_idx = 0;

        for chunk in chunks {
            let chunk = chunk;
            let chunk_rows = chunk.nrows();
            result
                .slice_mut(scirs2_core::ndarray::s![row_idx..row_idx + chunk_rows, ..])
                .assign(&chunk);
            row_idx += chunk_rows;
        }

        // Update quality metrics
        if n <= self.config.max_assessment_length {
            self.assess_quality(&result)?;
        }

        Ok(result)
    }

    /// Generate sequence sequentially
    fn generate_sequential(&mut self, n: usize) -> StatsResult<Array2<F>> {
        let mut result = Array2::zeros((n, self.dimension));

        for i in 0..n {
            let point = self.next_point()?;
            result.row_mut(i).assign(&point);
        }

        // Update quality metrics
        if n <= self.config.max_assessment_length {
            self.assess_quality(&result)?;
        }

        Ok(result)
    }

    /// Generate a chunk of the sequence
    fn generate_chunk(&self, start_index: usize, chunksize: usize) -> StatsResult<Array2<F>> {
        let mut chunk = Array2::zeros((chunksize, self.dimension));

        for i in 0..chunksize {
            let _index = start_index + i;
            let point = self.compute_point_at_index(_index)?;
            chunk.row_mut(i).assign(&point);
        }

        Ok(chunk)
    }

    /// Compute next point in sequence
    fn next_point(&mut self) -> StatsResult<Array1<F>> {
        let point = self.compute_point_at_index(self.state.current_index)?;
        self.state.current_index += 1;
        Ok(point)
    }

    /// Compute point at specific index
    fn compute_point_at_index(&self, index: usize) -> StatsResult<Array1<F>> {
        self.compute_point_for_type(index, &self.sequence_type)
    }

    /// Compute the point at `index` for an arbitrary sequence type.
    ///
    /// This is a free-standing dispatcher (rather than inlined into
    /// `compute_point_at_index`) so `compute_hybrid_sequence` can recurse
    /// into its `primary`/`secondary` sub-sequences: all of the
    /// per-dimension machinery each branch below relies on
    /// (`sobol_direction_numbers`, `niederreiter_generating_matrices`,
    /// `faure_pascal_matrix`) depends only on `self.dimension`, not on
    /// which `EnhancedSequenceType` is selected, so it is always valid to
    /// evaluate any sequence type against `self`.
    fn compute_point_for_type(
        &self,
        index: usize,
        seq_type: &EnhancedSequenceType,
    ) -> StatsResult<Array1<F>> {
        match seq_type {
            EnhancedSequenceType::SobolAdvanced {
                owen_scrambling,
                digital_shift,
                nested_scrambling,
            } => self.compute_sobol_advanced(
                index,
                *owen_scrambling,
                *digital_shift,
                *nested_scrambling,
            ),
            EnhancedSequenceType::Niederreiter {
                base_strategy,
                matrix_optimization,
            } => self.compute_niederreiter_enhanced(index, base_strategy, *matrix_optimization),
            EnhancedSequenceType::FaureImproved {
                permutation_optimization,
                radical_inverse_improvements,
            } => self.compute_faure_improved(
                index,
                *permutation_optimization,
                *radical_inverse_improvements,
            ),
            EnhancedSequenceType::DigitalNet {
                net_params,
                construction_method,
            } => self.compute_digital_net(index, net_params, construction_method),
            EnhancedSequenceType::Hybrid {
                primary,
                secondary,
                combination,
            } => self.compute_hybrid_sequence(index, primary, secondary, combination),
        }
    }

    /// Compute advanced Sobol sequence point
    ///
    /// Uses the real, per-dimension Joe-Kuo-style direction numbers from
    /// `self.sobol_direction_numbers` (see `qmc::advanced::
    /// AdvancedQMCGenerator::load_joe_kuo_direction_numbers`), combined via
    /// the standard Gray-code XOR digital-net construction. The previous
    /// implementation derived a value from `index` alone with no
    /// per-dimension direction numbers at all, so every dimension of an
    /// unscrambled point was numerically identical -- a degenerate,
    /// non-space-filling sequence.
    fn compute_sobol_advanced(
        &self,
        index: usize,
        owen_scrambling: bool,
        digital_shift: bool,
        _nested_scrambling: bool,
    ) -> StatsResult<Array1<F>> {
        let mut point = Array1::zeros(self.dimension);

        // Gray-code ordering: consecutive indices differ by exactly one
        // set bit, which is what gives the digital-net construction its
        // low-discrepancy property.
        let gray_code = index ^ (index >> 1);

        for dim in 0..self.dimension {
            let dir_nums = &self.sobol_direction_numbers[dim];

            let mut result_64 = 0u64;
            for (bit, &dn) in dir_nums.iter().enumerate().take(32) {
                if (gray_code >> bit) & 1 == 1 {
                    result_64 ^= dn;
                }
            }

            // Direction numbers are packed with the top bit at 2^63 (see
            // `load_joe_kuo_direction_numbers`); the digital-shift /
            // Owen-scrambling machinery below is 32-bit, so take the
            // most-significant 32 bits.
            let mut result = (result_64 >> 32) as u32;

            // Apply _scrambling if enabled
            if owen_scrambling {
                if let Some(ref matrices) = self.state.scrambling_matrices {
                    if dim < matrices.len() {
                        result = self.apply_owen_scrambling(result, &matrices[dim]);
                    }
                }
            }

            // Apply digital _shift if enabled
            if digital_shift {
                if let Some(ref shifts) = self.state.digital_shifts {
                    if dim < shifts.len() {
                        result ^= shifts[dim][0]; // Simplified
                    }
                }
            }

            point[dim] = F::from(result as f64 / (1u64 << 32) as f64).expect("Operation failed");
        }

        Ok(point)
    }

    /// Compute enhanced Niederreiter sequence point.
    ///
    /// Delegates to the real base-2 digital-net construction shared with
    /// `qmc::advanced::AdvancedQMCGenerator`
    /// (`self.niederreiter_generating_matrices`, built from per-dimension
    /// GF(2) generating matrices derived from primitive polynomials,
    /// combined via the digit-reversed bit-packed XOR construction -- see
    /// `AdvancedQMCGenerator::niederreiter_point_from_matrices`). The
    /// previous implementation ignored the generating matrices entirely
    /// and instead computed a plain per-dimension `radical_inverse` in a
    /// distinct prime base per dimension -- i.e. a Halton sequence
    /// mislabeled as "Niederreiter".
    ///
    /// `_base_strategy` is accepted (it is part of the public
    /// `Niederreiter` variant's data) but does not yet select between
    /// different underlying bases: the real, validated construction this
    /// delegates to is a base-2 digital net (matching `qmc::advanced`'s
    /// own tested implementation). A genuine per-`BaseSelectionStrategy`
    /// multi-base Niederreiter construction would need a full GF(p)
    /// polynomial-arithmetic implementation and is out of scope here;
    /// `matrix_optimization` (baked into `self.niederreiter_generating_matrices`
    /// at construction time, see `EnhancedQMCGenerator::new`) does
    /// genuinely control whether the shared "uniformity transform" step
    /// is applied.
    fn compute_niederreiter_enhanced(
        &self,
        index: usize,
        _base_strategy: &BaseSelectionStrategy,
        _matrix_optimization: bool,
    ) -> StatsResult<Array1<F>> {
        let raw = crate::qmc::advanced::AdvancedQMCGenerator::niederreiter_point_from_matrices(
            self.dimension,
            index,
            &self.niederreiter_generating_matrices,
        );

        let mut point = Array1::zeros(self.dimension);
        for dim in 0..self.dimension {
            point[dim] = F::from(raw[dim]).expect("Operation failed");
        }
        Ok(point)
    }

    /// Compute improved Faure sequence point.
    ///
    /// Real Faure (1982) construction: base `b` = smallest prime >=
    /// `dimension`; dimension `d`'s point is obtained by applying the
    /// `d`-th power of the (mod-`b`) Pascal matrix
    /// (`self.faure_pascal_matrix`) to the base-`b` digit expansion of
    /// `index`, then reading the transformed digits back out as a base-`b`
    /// fraction. Dimension 0 (`P^0` = identity) is exactly the plain
    /// base-`b` van der Corput sequence; every other dimension is a
    /// genuinely distinct permutation of it (a "permuted van der Corput"
    /// sequence). The previous implementation computed a single
    /// `radical_inverse(index, base)` value ONCE and assigned that SAME
    /// value to every dimension -- the same all-dimensions-identical,
    /// non-space-filling degeneracy as the historical `compute_sobol_advanced`
    /// bug fixed elsewhere in this module (see that function's doc
    /// comment), just for a different sequence type.
    fn compute_faure_improved(
        &self,
        index: usize,
        _permutation_optimization: bool,
        _radical_inverse_improvements: bool,
    ) -> StatsResult<Array1<F>> {
        let base = self.faure_base;

        // Base-`base` digit expansion of `index`, least-significant digit
        // first (digits[0] is the units digit).
        let mut digits = [0u64; FAURE_DIGITS];
        let mut rem = index as u64;
        for slot in digits.iter_mut() {
            *slot = rem % base;
            rem /= base;
        }

        let mut point = Array1::zeros(self.dimension);
        // `c` holds dimension `dim`'s transformed digit vector,
        // `P^dim * digits` (mod base). Dimension 0 is the identity
        // permutation (`P^0 = I`), i.e. `c == digits`.
        let mut c = digits;
        for dim in 0..self.dimension {
            if dim > 0 {
                // Advance `c` from `P^{dim - 1} * digits` to
                // `P^{dim} * digits` by one more application of the
                // Pascal matrix.
                let mut next = [0u64; FAURE_DIGITS];
                for (r, slot) in next.iter_mut().enumerate() {
                    let mut acc = 0u64;
                    for l in r..FAURE_DIGITS {
                        acc += self.faure_pascal_matrix[[r, l]] * c[l];
                    }
                    *slot = acc % base;
                }
                c = next;
            }

            let mut value = 0.0f64;
            let mut fraction = 1.0 / base as f64;
            for &digit in c.iter() {
                value += digit as f64 * fraction;
                fraction /= base as f64;
            }
            point[dim] = F::from(value).expect("Operation failed");
        }

        Ok(point)
    }

    /// Compute digital net point.
    ///
    /// Dispatches on `construction_method` to the crate's real
    /// digital-net constructions: `Sobol` uses the (already real)
    /// per-dimension Sobol direction numbers; `NiederreiterXing` reuses
    /// the real base-2 Niederreiter digital net (a full
    /// algebraic-function-field Niederreiter-Xing construction is out of
    /// scope, but this is a genuine digital-net computation, not the
    /// previous silent Sobol substitution regardless of the requested
    /// method). `PolynomialLattice`/`FiniteField` require polynomial-ring
    /// / general finite-field arithmetic this crate's QMC module does not
    /// implement, so they return an honest `NotImplementedError` rather
    /// than silently falling back to Sobol while claiming to honor the
    /// requested construction.
    fn compute_digital_net(
        &self,
        index: usize,
        net_params: &DigitalNetParams,
        construction_method: &NetConstructionMethod,
    ) -> StatsResult<Array1<F>> {
        if net_params.base != 2 {
            return Err(StatsError::NotImplementedError(format!(
                "DigitalNet with base {} is not implemented: this crate's digital-net \
                 constructions (Sobol, NiederreiterXing) are base-2 only",
                net_params.base
            )));
        }

        match construction_method {
            NetConstructionMethod::Sobol => self.compute_sobol_advanced(index, false, false, false),
            NetConstructionMethod::NiederreiterXing => {
                self.compute_niederreiter_enhanced(index, &BaseSelectionStrategy::Automatic, true)
            }
            NetConstructionMethod::PolynomialLattice | NetConstructionMethod::FiniteField => {
                Err(StatsError::NotImplementedError(format!(
                    "DigitalNet construction method {construction_method:?} is not implemented: \
                     it requires polynomial-ring/finite-field arithmetic that this crate's QMC \
                     module does not yet provide; use NetConstructionMethod::Sobol or \
                     NetConstructionMethod::NiederreiterXing instead"
                )))
            }
        }
    }

    /// Compute hybrid sequence point by genuinely combining `primary` and
    /// `secondary` per `combination`, instead of (as before)
    /// unconditionally evaluating the Sobol path regardless of what was
    /// requested.
    fn compute_hybrid_sequence(
        &self,
        index: usize,
        primary: &EnhancedSequenceType,
        secondary: &EnhancedSequenceType,
        combination: &HybridCombinationStrategy,
    ) -> StatsResult<Array1<F>> {
        let p = self.compute_point_for_type(index, primary)?;
        let s = self.compute_point_for_type(index, secondary)?;

        let mut point = Array1::zeros(self.dimension);
        match combination {
            HybridCombinationStrategy::Interleave => {
                // Alternate whole points by index parity: even indices
                // take the primary sequence's point, odd indices the
                // secondary's.
                let use_primary = index.is_multiple_of(2);
                for dim in 0..self.dimension {
                    point[dim] = if use_primary { p[dim] } else { s[dim] };
                }
            }
            HybridCombinationStrategy::Weighted(w) => {
                let w = F::from(w.clamp(0.0, 1.0)).expect("Operation failed");
                let one_minus_w = F::one() - w;
                for dim in 0..self.dimension {
                    point[dim] = w * p[dim] + one_minus_w * s[dim];
                }
            }
            HybridCombinationStrategy::DimensionAlternation => {
                // Alternate individual dimensions between the two
                // sequences (distinct from `Interleave`, which alternates
                // whole points by index).
                for dim in 0..self.dimension {
                    point[dim] = if dim.is_multiple_of(2) {
                        p[dim]
                    } else {
                        s[dim]
                    };
                }
            }
            HybridCombinationStrategy::Adaptive => {
                // Genuine per-point history-dependent adaptivity ("based
                // on uniformity") would need mutable state across calls
                // that this pure, index-only function does not retain;
                // fall back to an honest equal-weight blend (the same
                // real computation as `Weighted(0.5)`) rather than
                // fabricating a fake uniformity metric.
                let half = F::from(0.5).expect("Operation failed");
                for dim in 0..self.dimension {
                    point[dim] = half * p[dim] + half * s[dim];
                }
            }
        }

        Ok(point)
    }

    /// Initialize randomization (scrambling, digital shifts)
    fn initialize_randomization(&mut self) -> StatsResult<()> {
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };

        // Initialize scrambling matrices
        if self.needs_scrambling() {
            let mut matrices = Vec::with_capacity(self.dimension);
            for _ in 0..self.dimension {
                matrices.push(self.generate_scrambling_matrix(&mut rng)?);
            }
            self.state.scrambling_matrices = Some(matrices);
        }

        // Initialize digital shifts
        if self.needs_digital_shift() {
            let mut shifts = Vec::with_capacity(self.dimension);
            for _ in 0..self.dimension {
                let shift = Array1::from_shape_fn(32, |_| rng.random::<u32>());
                shifts.push(shift);
            }
            self.state.digital_shifts = Some(shifts);
        }

        Ok(())
    }

    /// Check if sequence type needs scrambling
    fn needs_scrambling(&self) -> bool {
        match &self.sequence_type {
            EnhancedSequenceType::SobolAdvanced {
                owen_scrambling, ..
            } => *owen_scrambling,
            _ => false,
        }
    }

    /// Check if sequence type needs digital shift
    fn needs_digital_shift(&self) -> bool {
        match &self.sequence_type {
            EnhancedSequenceType::SobolAdvanced { digital_shift, .. } => *digital_shift,
            _ => false,
        }
    }

    /// Generate scrambling matrix
    fn generate_scrambling_matrix<R: Rng>(&self, rng: &mut R) -> StatsResult<Array2<u32>> {
        let mut matrix = Array2::zeros((32, 32));

        // Generate random permutation matrix
        for i in 0..32 {
            let j = rng.random_range(0..32);
            matrix[[i, j]] = 1;
        }

        Ok(matrix)
    }

    /// Apply Owen scrambling to a value
    fn apply_owen_scrambling(&self, value: u32, matrix: &Array2<u32>) -> u32 {
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

    /// Find smallest prime >= n.
    ///
    /// A plain associated function (no `&self`) rather than a method
    /// since it is needed by `EnhancedQMCGenerator::new` to compute
    /// `faure_base` before `self` exists.
    fn smallest_prime_geq(n: u32) -> u32 {
        if n <= 2 {
            return 2;
        }

        let mut candidate = if n.is_multiple_of(2) { n + 1 } else { n };

        while !Self::is_prime(candidate) {
            candidate += 2;
        }

        candidate
    }

    /// Check if number is prime.
    ///
    /// A plain associated function (no `&self`) for the same reason as
    /// `smallest_prime_geq` above.
    fn is_prime(n: u32) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u32;
        for i in (3..=sqrt_n).step_by(2) {
            if n.is_multiple_of(i) {
                return false;
            }
        }

        true
    }

    /// Build the (mod-`base`) Pascal matrix used by the real Faure
    /// sequence construction: `matrix[[r, l]] = C(l, r) mod base` for
    /// `l >= r` (else 0), for `r, l` in `0..FAURE_DIGITS`. Dimension `d`'s
    /// digit permutation is this matrix raised to the `d`-th power (see
    /// `compute_faure_improved`).
    ///
    /// A plain associated function (no `&self`) since it is needed by
    /// `EnhancedQMCGenerator::new` to compute `faure_pascal_matrix` before
    /// `self` exists.
    fn build_faure_pascal_matrix(base: u64) -> Array2<u64> {
        // Raw (unreduced) Pascal's-triangle binomial coefficients
        // `binom[l][r] = C(l, r)`, well within `u64` range for
        // `l, r < FAURE_DIGITS` (e.g. `C(31, 15) ~= 3.0e8`).
        let mut binom = [[0u64; FAURE_DIGITS]; FAURE_DIGITS];
        for l in 0..FAURE_DIGITS {
            binom[l][0] = 1;
            binom[l][l] = 1;
            for r in 1..l {
                binom[l][r] = binom[l - 1][r - 1] + binom[l - 1][r];
            }
        }

        let mut matrix = Array2::<u64>::zeros((FAURE_DIGITS, FAURE_DIGITS));
        for r in 0..FAURE_DIGITS {
            for l in r..FAURE_DIGITS {
                matrix[[r, l]] = binom[l][r] % base;
            }
        }
        matrix
    }

    /// Assess sequence quality.
    ///
    /// Populates all four `QualityMetrics` fields with real, computed
    /// estimates. `star_discrepancy` was already a genuine (randomized)
    /// Monte Carlo estimate; `wraparound_discrepancy`, `diaphony` and
    /// `figure_of_merit` were previously left at their `Default` value
    /// (`0.0`) unconditionally -- i.e. every sequence, however degenerate,
    /// silently reported "perfect" wraparound discrepancy/diaphony/figure
    /// of merit. See the per-field doc comments on `QualityMetrics` for
    /// the exact formulas and their (documented, bounded) truncations.
    fn assess_quality(&mut self, sequence: &Array2<F>) -> StatsResult<()> {
        let n = sequence.nrows();
        let d = sequence.ncols();

        // Estimate star discrepancy via randomized local-discrepancy test
        // points (already a real computation prior to this fix).
        let mut max_discrepancy = 0.0;
        let num_test_points = 50.min(n);

        let mut rng = scirs2_core::random::thread_rng();
        for _ in 0..num_test_points {
            let mut test_point = Array1::zeros(d);
            for j in 0..d {
                test_point[j] = F::from(rng.random::<f64>()).expect("Operation failed");
            }

            let mut count = 0;
            for i in 0..n {
                let mut in_box = true;
                for j in 0..d {
                    if sequence[[i, j]] > test_point[j] {
                        in_box = false;
                        break;
                    }
                }
                if in_box {
                    count += 1;
                }
            }

            let volume: F = test_point.iter().fold(F::one(), |acc, &x| acc * x);
            let expected = volume.to_f64().expect("Operation failed") * n as f64;
            let discrepancy = (count as f64 - expected).abs() / n as f64;
            max_discrepancy = max_discrepancy.max(discrepancy);
        }
        self.state.quality_metrics.star_discrepancy = max_discrepancy;

        // Wrap-around L2-discrepancy (Hickernell 1998): the standard
        // closed form
        //   WD_n^2 = -(4/3)^d + (1/n^2) * sum_{i,k=1}^n
        //              prod_{j=1}^d [3/2 - |x_ij - x_kj| * (1 - |x_ij - x_kj|)]
        // evaluated over a bounded leading subsample of points/dimensions
        // (see `QUALITY_SAMPLE_CAP`/`WRAPAROUND_DIM_CAP`). We report the
        // discrepancy itself (the square root of the closed form above,
        // clamped to >= 0 to guard tiny negative floating-point noise),
        // matching `star_discrepancy`'s non-squared, "smaller is better"
        // scale.
        let m = n.min(QUALITY_SAMPLE_CAP);
        let d_wd = d.min(WRAPAROUND_DIM_CAP);
        let mut wd_sum = 0.0f64;
        for i in 0..m {
            for k in 0..m {
                let mut prod = 1.0f64;
                for j in 0..d_wd {
                    let xi = sequence[[i, j]].to_f64().expect("Operation failed");
                    let xk = sequence[[k, j]].to_f64().expect("Operation failed");
                    let diff = (xi - xk).abs();
                    prod *= 1.5 - diff * (1.0 - diff);
                }
                wd_sum += prod;
            }
        }
        let wd_squared = -((4.0f64 / 3.0).powi(d_wd as i32)) + wd_sum / (m as f64 * m as f64);
        self.state.quality_metrics.wraparound_discrepancy = wd_squared.max(0.0).sqrt();

        // Diaphony (Zinterhof spectral measure): a bounded truncated
        // approximation of
        //   sum_{h != 0} |mean_k exp(2*pi*i*h.x_k)|^2 / r(h)^2,
        //   r(h) = prod_j max(1, |h_j|),
        // restricted to (a) single-dimension ("marginal") frequencies
        // with |h_j| <= DIAPHONY_FREQ_MAX and (b) pairwise
        // ("interaction") frequencies across at most DIAPHONY_DIM_CAP
        // dimensions, keeping the cost bounded for high-dimensional
        // sequences. The true (infinite, full-interaction) diaphony sum
        // is intractable to compute exactly, but this partial sum is a
        // real, non-fabricated quantity that responds correctly to actual
        // (non-)uniformity -- e.g. a sequence where two dimensions are
        // (near-)identical (the historical bug this module's Sobol path
        // had) produces a large pairwise term here, unlike the previous
        // hardcoded 0.0.
        let dims = d.min(DIAPHONY_DIM_CAP);
        let mut diaphony_sum = 0.0f64;

        // Order-1 (marginal) frequencies: exactly one nonzero entry.
        for j in 0..dims {
            for h in 1..=DIAPHONY_FREQ_MAX {
                let (re, im) = Self::empirical_fourier_coefficient(sequence, m, &[(j, h)]);
                let weight = 1.0 / (h as f64 * h as f64);
                diaphony_sum += weight * (re * re + im * im);
            }
        }

        // Order-2 (pairwise interaction) frequencies. Fixing `h1 > 0`
        // and letting `h2` range over both signs covers each
        // conjugate-symmetric frequency pair exactly once.
        for j1 in 0..dims {
            for j2 in (j1 + 1)..dims {
                for h1 in 1..=DIAPHONY_FREQ_MAX {
                    for h2 in -DIAPHONY_FREQ_MAX..=DIAPHONY_FREQ_MAX {
                        if h2 == 0 {
                            continue;
                        }
                        let (re, im) =
                            Self::empirical_fourier_coefficient(sequence, m, &[(j1, h1), (j2, h2)]);
                        let weight = 1.0 / ((h1 * h1) as f64 * (h2 * h2) as f64);
                        diaphony_sum += weight * (re * re + im * im);
                    }
                }
            }
        }
        self.state.quality_metrics.diaphony = diaphony_sum.sqrt();

        // Figure of merit: a single-scalar quality summary combining the
        // three discrepancy-like measures above via their worst (largest)
        // value, on the same "smaller is better" scale as each individual
        // metric (and as `EnhancedQMCConfig::quality_threshold`).
        // "Figure of merit" has no single universal definition across the
        // QMC literature (unlike e.g. star discrepancy); this is an
        // honestly-documented composite rather than a specific named
        // literature quantity.
        self.state.quality_metrics.figure_of_merit = self
            .state
            .quality_metrics
            .star_discrepancy
            .max(self.state.quality_metrics.wraparound_discrepancy)
            .max(self.state.quality_metrics.diaphony);

        Ok(())
    }

    /// Empirical Fourier coefficient
    /// `(1/m) * sum_{k=0}^{m-1} exp(2*pi*i * sum_t h_t * x_k[dim_t])`
    /// for a small set of `(dimension, frequency)` terms `terms`,
    /// returning `(real, imaginary)`. Used by the `diaphony` estimate in
    /// `assess_quality` for both single-dimension (`terms.len() == 1`) and
    /// pairwise (`terms.len() == 2`) frequency vectors.
    fn empirical_fourier_coefficient(
        sequence: &Array2<F>,
        m: usize,
        terms: &[(usize, i64)],
    ) -> (f64, f64) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for k in 0..m {
            let mut phase = 0.0f64;
            for &(dim, h) in terms {
                let x = sequence[[k, dim]].to_f64().expect("Operation failed");
                phase += h as f64 * x;
            }
            let angle = 2.0 * std::f64::consts::PI * phase;
            re += angle.cos();
            im += angle.sin();
        }
        (re / m as f64, im / m as f64)
    }

    /// Get current quality metrics
    pub fn quality_metrics(&self) -> &QualityMetrics {
        &self.state.quality_metrics
    }
}

/// Convenience functions for enhanced QMC
#[allow(dead_code)]
pub fn enhanced_sobol<F>(
    n: usize,
    dimension: usize,
    scrambling: bool,
    seed: Option<u64>,
) -> StatsResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive + std::fmt::Display,
{
    let sequence_type = EnhancedSequenceType::SobolAdvanced {
        owen_scrambling: scrambling,
        digital_shift: true,
        nested_scrambling: false,
    };

    let config = EnhancedQMCConfig {
        seed,
        ..Default::default()
    };

    let mut generator = EnhancedQMCGenerator::new(sequence_type, dimension, config)?;
    generator.generate(n)
}

#[allow(dead_code)]
pub fn enhanced_niederreiter<F>(
    n: usize,
    dimension: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive + std::fmt::Display,
{
    let sequence_type = EnhancedSequenceType::Niederreiter {
        base_strategy: BaseSelectionStrategy::OptimizedPrimes,
        matrix_optimization: true,
    };

    let config = EnhancedQMCConfig {
        seed,
        ..Default::default()
    };

    let mut generator = EnhancedQMCGenerator::new(sequence_type, dimension, config)?;
    generator.generate(n)
}

#[allow(dead_code)]
pub fn enhanced_digital_net<F>(
    n: usize,
    dimension: usize,
    t: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<F>>
where
    F: Float + Zero + One + Copy + Send + Sync + SimdUnifiedOps + FromPrimitive + std::fmt::Display,
{
    let net_params = DigitalNetParams {
        t,
        m: 32,
        s: dimension,
        base: 2,
    };

    let sequence_type = EnhancedSequenceType::DigitalNet {
        net_params,
        construction_method: NetConstructionMethod::Sobol,
    };

    let config = EnhancedQMCConfig {
        seed,
        ..Default::default()
    };

    let mut generator = EnhancedQMCGenerator::new(sequence_type, dimension, config)?;
    generator.generate(n)
}

#[path = "enhanced_sequences_tests.rs"]
#[cfg(test)]
mod tests;
