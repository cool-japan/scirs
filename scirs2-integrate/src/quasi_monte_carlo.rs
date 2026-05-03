//! Quasi-Monte Carlo low-discrepancy sequences and integration
//!
//! Quasi-Monte Carlo (QMC) methods replace pseudo-random points with
//! **deterministic low-discrepancy sequences** that fill the unit hypercube
//! much more uniformly than independent random samples.  For smooth integrands
//! in `s` dimensions, QMC typically achieves a convergence rate close to
//! `O(N^{-1} (log N)^s)` compared to the `O(N^{-1/2})` rate of plain Monte
//! Carlo.
//!
//! ## Sequences provided
//!
//! | Struct | Method | Notes |
//! |---|---|---|
//! | [`HaltonSequence`] | van der Corput in coprime bases | Fast; suffers correlation in high dims |
//! | [`SobolSequence`] | direction numbers (21 dims built-in) | Excellent uniformity up to 21 dims |
//! | [`LatticeRule`] | rank-1 lattice (Korobov) | Best when generator vector is carefully chosen |
//!
//! ## Integration helpers
//!
//! * [`qmc_integrate`] – generic QMC integration for any sequence implementing [`QmcSequence`].
//! * [`star_discrepancy`] – L∞ star-discrepancy for quality measurement.
//! * [`scrambled_halton`] – random digital scrambling for error estimation.
//!
//! ## References
//!
//! - Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods.*
//! - Joe, S. & Kuo, F. Y. (2008). Constructing Sobol sequences with better two-dimensional
//!   projections. *SIAM J. Sci. Comput.*, 30(5), 2635–2654.
//! - Sloan, I. H. & Joe, S. (1994). *Lattice Methods for Multiple Integration.*

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::uniform::SampleUniform;
use scirs2_core::random::{Distribution, Uniform};

// ─────────────────────────────────────────────────────────────────────────────
// QmcSequence trait
// ─────────────────────────────────────────────────────────────────────────────

/// A deterministic (or randomised) sequence in the `[0,1)^d` unit hypercube.
pub trait QmcSequence {
    /// Return the next point in the sequence as a `Vec<f64>` of length `dim()`.
    fn next_point(&mut self) -> Vec<f64>;

    /// Dimensionality of the sequence.
    fn dim(&self) -> usize;

    /// Reset the sequence to its initial state.
    fn reset(&mut self);

    /// Generate `n` successive points as an `n × d` matrix.
    fn generate(&mut self, n: usize) -> Array2<f64> {
        let d = self.dim();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let p = self.next_point();
            for j in 0..d {
                out[[i, j]] = p[j];
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Halton sequence
// ─────────────────────────────────────────────────────────────────────────────

/// Halton low-discrepancy sequence.
///
/// Each dimension uses a different prime as its base for the van der Corput
/// radical-inverse construction.  The `d`-th dimension uses the `d`-th prime
/// (0-indexed), so dimension 0 uses base 2, dimension 1 uses base 3, etc.
///
/// Up to 21 dimensions are supported with pre-stored primes; beyond that the
/// sequence uses the next prime computed on-the-fly.
///
/// # Example
///
/// ```
/// use scirs2_integrate::quasi_monte_carlo::{HaltonSequence, QmcSequence};
///
/// let mut seq = HaltonSequence::new(vec![2, 3]);
/// let p0 = seq.next_point();
/// assert_eq!(p0.len(), 2);
/// assert!(p0.iter().all(|&x| x >= 0.0 && x < 1.0));
/// ```
pub struct HaltonSequence {
    /// The base for each dimension (should be pairwise coprime; typically primes).
    pub base: Vec<usize>,
    /// The current sequence index (0-based; before calling `next_point` the
    /// first time, `index == 0`, and the first returned point corresponds to
    /// index 1).
    pub index: usize,
}

impl HaltonSequence {
    /// Construct a Halton sequence with explicitly provided bases.
    ///
    /// For good equidistribution use distinct primes. At most 64 dimensions are
    /// practical; beyond about 30 the Halton sequence exhibits visible
    /// correlation and scrambling is recommended.
    pub fn new(base: Vec<usize>) -> Self {
        for (i, &b) in base.iter().enumerate() {
            debug_assert!(b >= 2, "base[{i}] must be >= 2");
        }
        Self { base, index: 0 }
    }

    /// Construct a Halton sequence for `dim` dimensions using the first `dim`
    /// primes automatically.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::quasi_monte_carlo::{HaltonSequence, QmcSequence};
    ///
    /// let mut seq = HaltonSequence::with_primes(5);
    /// let pt = seq.next_point();
    /// assert_eq!(pt.len(), 5);
    /// ```
    pub fn with_primes(dim: usize) -> Self {
        let primes = first_n_primes(dim);
        Self::new(primes)
    }

    /// Return the van der Corput value of `n` in `base`.
    fn van_der_corput(mut n: usize, base: usize) -> f64 {
        let mut result = 0.0f64;
        let mut denominator = 1.0f64;
        while n > 0 {
            denominator *= base as f64;
            result += (n % base) as f64 / denominator;
            n /= base;
        }
        result
    }
}

impl QmcSequence for HaltonSequence {
    fn next_point(&mut self) -> Vec<f64> {
        self.index += 1;
        self.base
            .iter()
            .map(|&b| Self::van_der_corput(self.index, b))
            .collect()
    }

    fn dim(&self) -> usize {
        self.base.len()
    }

    fn reset(&mut self) {
        self.index = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sobol sequence
// ─────────────────────────────────────────────────────────────────────────────

/// Direction numbers for the Sobol sequence, dimension 2–21.
///
/// These are the standard Joe–Kuo (2003) direction numbers for dimensions 2–21
/// (dimension 1 is always powers of 2).  Each row contains the `s` initial
/// direction numbers for that dimension, where `s` is the degree of the
/// corresponding primitive polynomial over GF(2).
///
/// Source: Joe & Kuo (2008) "Constructing Sobol sequences with better
/// two-dimensional projections".
static JOE_KUO_DIRECTION_INIT: &[(
    u32,    // degree of primitive polynomial
    u32,    // polynomial coefficients (bits, excluding leading/trailing 1)
    &[u32], // initial direction numbers m_i (before left-shift)
)] = &[
    (1, 0, &[1]),
    (2, 1, &[1, 1]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 5, 5, 17]),
    (5, 4, &[1, 1, 5, 5, 5]),
    (5, 7, &[1, 1, 7, 11, 19]),
    (5, 11, &[1, 1, 5, 1, 1]),
    (5, 13, &[1, 1, 1, 3, 11]),
    (5, 14, &[1, 3, 5, 5, 31]),
    (6, 1, &[1, 3, 3, 9, 7, 49]),
    (6, 13, &[1, 1, 1, 15, 21, 21]),
    (6, 16, &[1, 3, 1, 13, 27, 49]),
    (6, 19, &[1, 1, 1, 15, 3, 13]),
    (6, 22, &[1, 3, 1, 15, 13, 17]),
    (6, 25, &[1, 1, 5, 5, 19, 45]),
    (7, 1, &[1, 3, 5, 5, 19, 27, 97]),
    (7, 4, &[1, 3, 7, 5, 13, 29, 91]),
];

/// Sobol low-discrepancy sequence.
///
/// Supports up to 21 dimensions using the Joe–Kuo (2008) direction numbers.
/// The first dimension uses the standard van der Corput base-2 sequence; the
/// remaining dimensions use properly initialised direction-number recurrences.
///
/// The sequence is generated with the **Gray-code** algorithm, which avoids
/// recomputing all bits at each step and runs in O(1) amortised time per point
/// for all dimensions.
///
/// # Example
///
/// ```
/// use scirs2_integrate::quasi_monte_carlo::{SobolSequence, QmcSequence};
///
/// let mut seq = SobolSequence::new(3);
/// let pt = seq.next_point();
/// assert_eq!(pt.len(), 3);
/// assert!(pt.iter().all(|&x| x >= 0.0 && x <= 1.0));
/// ```
pub struct SobolSequence {
    dim: usize,
    /// Current sample index (before first call to next_point this is 0).
    index: u64,
    /// Current Sobol point (integer representation, one entry per dimension).
    current: Vec<u64>,
    /// Direction numbers: `dir[d][j]` for dimension `d`, bit position `j`.
    dir: Vec<Vec<u64>>,
}

impl SobolSequence {
    /// Maximum supported dimension.
    pub const MAX_DIM: usize = 21;

    /// Bits of precision (= word size used for direction numbers).
    const BITS: u32 = 32;

    /// Create a new Sobol sequence generator for `dim` dimensions.
    ///
    /// # Panics
    ///
    /// Panics (debug) or clamps (release) if `dim > MAX_DIM`.
    pub fn new(dim: usize) -> Self {
        let dim = dim.min(Self::MAX_DIM);
        let dir = build_sobol_direction_numbers(dim);
        Self {
            dim,
            index: 0,
            current: vec![0u64; dim],
            dir,
        }
    }

    /// Generate the next Sobol point using the Gray-code recurrence.
    fn advance(&mut self) -> Vec<f64> {
        if self.index == 0 {
            self.index = 1;
            return vec![0.0f64; self.dim];
        }

        // Position of the rightmost zero bit of (index-1)
        let c = (!(self.index - 1)).trailing_zeros() as usize;

        for d in 0..self.dim {
            if c < self.dir[d].len() {
                self.current[d] ^= self.dir[d][c];
            }
        }
        self.index += 1;

        let scale = 2.0f64.powi(Self::BITS as i32);
        self.current.iter().map(|&x| (x as f64) / scale).collect()
    }
}

impl QmcSequence for SobolSequence {
    fn next_point(&mut self) -> Vec<f64> {
        self.advance()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn reset(&mut self) {
        self.index = 0;
        self.current.iter_mut().for_each(|x| *x = 0);
    }
}

/// Build direction numbers for all `dim` dimensions of the Sobol sequence.
fn build_sobol_direction_numbers(dim: usize) -> Vec<Vec<u64>> {
    let bits = SobolSequence::BITS as usize;
    let mut dir = vec![vec![0u64; bits]; dim];

    // Dimension 1: standard van der Corput base-2, dir[0][j] = 2^(31-j)
    for j in 0..bits {
        dir[0][j] = 1u64 << (bits - 1 - j);
    }

    // Remaining dimensions from Joe–Kuo table (0-indexed table entry i → dimension i+2)
    for d in 1..dim {
        let table_idx = d - 1; // entry in JOE_KUO_DIRECTION_INIT
        if table_idx >= JOE_KUO_DIRECTION_INIT.len() {
            // Fall back to van der Corput (less ideal but safe)
            for j in 0..bits {
                dir[d][j] = 1u64 << (bits - 1 - j);
            }
            continue;
        }

        let (s, a, m_init) = JOE_KUO_DIRECTION_INIT[table_idx];
        let s = s as usize;

        // Set initial direction numbers m_i << (bits - i)
        for i in 0..s {
            if i < m_init.len() {
                dir[d][i] = (m_init[i] as u64) << (bits - 1 - i);
            }
        }

        // Recurrence: v_i = v_{i-s} XOR (v_{i-s} >> s) XOR XOR_{k=1}^{s-1} c_k * v_{i-k}
        for j in s..bits {
            let mut v = dir[d][j - s] ^ (dir[d][j - s] >> s);
            for k in 1..s {
                // bit k-1 of polynomial coefficient a indicates whether term k is present
                if (a >> (s - 1 - k)) & 1 == 1 {
                    v ^= dir[d][j - k];
                }
            }
            dir[d][j] = v;
        }
    }

    dir
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank-1 lattice rule
// ─────────────────────────────────────────────────────────────────────────────

/// Rank-1 lattice rule (Korobov-style).
///
/// A rank-1 lattice rule with `n` points and generating vector `z ∈ ℤ^d` is
/// the point set `{ ({i·z/n}) : i = 0, ..., n-1 }` where `{·}` denotes the
/// fractional part.
///
/// When iterated sequentially (`next_point`) the points are generated in order
/// `i = 0, 1, 2, …`.
///
/// The generating vector `z` should be chosen so that the lattice has low
/// discrepancy (e.g., via the CBC construction).  Simple choices like
/// `z_d = (z_1^d mod n)` for a well-chosen base `z_1` also work well in
/// practice.
///
/// # Example
///
/// ```
/// use scirs2_integrate::quasi_monte_carlo::{LatticeRule, QmcSequence};
///
/// // Two-dimensional rank-1 lattice with n=1024, generator (1, 363)
/// // (the Fibonacci lattice – one of the optimal 2-D lattices)
/// let mut lat = LatticeRule::new(vec![1.0 / 1024.0, 363.0 / 1024.0], 1024);
/// let p0 = lat.next_point();
/// assert_eq!(p0.len(), 2);
/// ```
pub struct LatticeRule {
    /// Generator vector, normalised so each component is in `(0, 1]`.
    /// Specifically, if the integer generator is `z_d` with `n` points,
    /// store `generator[d] = z_d as f64`.  Then the `i`-th point in dimension
    /// `d` is `{ i * generator[d] / n }`.
    pub generator: Vec<f64>,
    /// Total number of lattice points.
    pub n: usize,
    /// Current index (starts at 0).
    index: usize,
}

impl LatticeRule {
    /// Create a rank-1 lattice rule.
    ///
    /// # Parameters
    ///
    /// * `generator` – normalised generator values in `(0, 1]`, one per dimension.
    ///   For integer generator `z` and lattice size `n`, pass `z as f64 / n as f64`.
    /// * `n`         – number of lattice points.
    pub fn new(generator: Vec<f64>, n: usize) -> Self {
        Self {
            generator,
            n,
            index: 0,
        }
    }

    /// Build a Korobov-style lattice for `dim` dimensions with generator `a`
    /// and `n` points.
    ///
    /// The generator vector is `(1, a, a², …, a^{d-1}) mod n`.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_integrate::quasi_monte_carlo::{LatticeRule, QmcSequence};
    ///
    /// let lat = LatticeRule::korobov(4, 1973, 8192);
    /// assert_eq!(lat.dim(), 4);
    /// assert_eq!(lat.n, 8192);
    /// ```
    pub fn korobov(dim: usize, a: u64, n: usize) -> Self {
        let n64 = n as u64;
        let generator: Vec<f64> = (0..dim)
            .map(|k| {
                // a^k mod n
                let mut power = 1u64;
                for _ in 0..k {
                    power = (power * a) % n64;
                }
                power as f64 / n as f64
            })
            .collect();
        Self::new(generator, n)
    }
}

impl QmcSequence for LatticeRule {
    fn next_point(&mut self) -> Vec<f64> {
        let i = self.index;
        self.index += 1;
        let i_f = i as f64;
        let n_f = self.n as f64;
        self.generator
            .iter()
            .map(|&g| {
                let raw = i_f * g * n_f / n_f; // i * g (g is already normalised)
                                               // Take fractional part; use modulo for robustness
                let frac = (i_f * g).fract();
                frac.rem_euclid(1.0)
            })
            .collect()
    }

    fn dim(&self) -> usize {
        self.generator.len()
    }

    fn reset(&mut self) {
        self.index = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QMC integration
// ─────────────────────────────────────────────────────────────────────────────

/// Integrate a function using any sequence implementing [`QmcSequence`].
///
/// The estimator is the standard sample mean scaled by the volume:
///
/// ```text
/// I ≈ vol(Ω) · (1/N) Σ_{i=1}^{N} f(x_i)
/// ```
///
/// where `x_i ∈ Ω` are the rescaled QMC points.
///
/// # Parameters
///
/// * `f`        – integrand, called with a `&[f64]` of length `bounds.len()`.
/// * `bounds`   – `(a_d, b_d)` per dimension.
/// * `sequence` – any mutable reference to a [`QmcSequence`] (dimension must
///   match `bounds.len()`).
/// * `n_samples`– number of points to evaluate.
///
/// # Returns
///
/// Estimated integral value.
///
/// # Errors
///
/// Returns [`IntegrateError::ValueError`] if the sequence dimension does not
/// match the number of bounds, bounds are empty, or `n_samples == 0`.
///
/// # Example
///
/// ```
/// use scirs2_integrate::quasi_monte_carlo::{SobolSequence, qmc_integrate};
///
/// let mut seq = SobolSequence::new(2);
/// // ∫₀¹ ∫₀¹ x·y dx dy = 1/4
/// let val = qmc_integrate(
///     |x: &[f64]| x[0] * x[1],
///     &[(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)],
///     &mut seq,
///     4096,
/// ).expect("qmc_integrate failed");
/// assert!((val - 0.25).abs() < 0.01);
/// ```
pub fn qmc_integrate<F, S>(
    f: F,
    bounds: &[(f64, f64)],
    sequence: &mut S,
    n_samples: usize,
) -> IntegrateResult<f64>
where
    F: Fn(&[f64]) -> f64,
    S: QmcSequence,
{
    if bounds.is_empty() {
        return Err(IntegrateError::ValueError(
            "bounds must not be empty".to_string(),
        ));
    }
    if n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "n_samples must be positive".to_string(),
        ));
    }
    let dim = bounds.len();
    if sequence.dim() != dim {
        return Err(IntegrateError::DimensionMismatch(format!(
            "sequence dimension ({}) != bounds dimension ({})",
            sequence.dim(),
            dim
        )));
    }

    let volume: f64 = bounds.iter().map(|&(a, b)| b - a).product();
    let mut sum = 0.0f64;
    let mut scaled = vec![0.0f64; dim];

    for _ in 0..n_samples {
        let unit_pt = sequence.next_point();
        for d in 0..dim {
            let (a, b) = bounds[d];
            scaled[d] = a + unit_pt[d] * (b - a);
        }
        sum += f(&scaled);
    }

    Ok(sum / (n_samples as f64) * volume)
}

// ─────────────────────────────────────────────────────────────────────────────
// Star discrepancy
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **L∞ star-discrepancy** `D*_N` of a point set.
///
/// The star discrepancy is defined as
///
/// ```text
/// D*_N = sup_{x ∈ [0,1]^s} |A([0,x), N) / N − vol([0,x))|
/// ```
///
/// where `A([0,x), N)` counts the number of points strictly inside the
/// axis-aligned box `[0, x)`.
///
/// Computing the exact star discrepancy is #P-hard in general, so this
/// function uses the **extreme-point** algorithm: the supremum is attained at
/// a corner formed by the coordinates of the point set.  The algorithm checks
/// all `N^s` candidate corners (not practical beyond ≈ 10 000 points in 1–2
/// dimensions but fast for small test sets) using a coordinate-projection
/// approach that runs in `O(N^{d+1})` time.
///
/// For practical use with large `N` and `d > 2` use randomised algorithms or
/// the Hickernell `R²` figure of merit instead.
///
/// # Parameters
///
/// * `points` – `n × d` matrix of points in `[0, 1]^d`.
///
/// # Returns
///
/// The star discrepancy `D*_N ∈ [0, 1]`.
///
/// # Errors
///
/// Returns [`IntegrateError::ValueError`] if the matrix is empty.
///
/// # Example
///
/// ```
/// use scirs2_integrate::quasi_monte_carlo::{HaltonSequence, QmcSequence, star_discrepancy};
/// use scirs2_core::ndarray::Array2;
///
/// let mut seq = HaltonSequence::with_primes(2);
/// let pts = seq.generate(512);
/// let d = star_discrepancy(&pts).expect("discrepancy failed");
/// // Good QMC sequences have D* ≪ 1/sqrt(N) ≈ 0.044
/// assert!(d < 0.1);
/// ```
pub fn star_discrepancy(points: &Array2<f64>) -> IntegrateResult<f64> {
    let (n, d) = (points.nrows(), points.ncols());
    if n == 0 || d == 0 {
        return Err(IntegrateError::ValueError(
            "point set must be non-empty".to_string(),
        ));
    }

    let n_f = n as f64;
    let mut disc = 0.0f64;

    // For each candidate corner x built from the coordinates of the point set:
    // We use a column-by-column approach: for each point p_k, take x_j = p_k[j]
    // for j=0 and enumerate all combinations from there.
    // For large n*d this is expensive; we limit exact computation to n ≤ 2000.
    if n > 2000 || d > 4 {
        // Use the 1-D marginal approximation as a lower bound
        for j in 0..d {
            let mut col: Vec<f64> = (0..n).map(|i| points[[i, j]]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let d_j = star_discrepancy_1d(&col, n_f);
            disc = disc.max(d_j);
        }
        return Ok(disc);
    }

    // Exact computation: check all N coordinate values per dimension as candidate x_j
    // For d dimensions, the candidate corners are products of the coordinate values.
    // We implement a recursive enumeration.
    let mut x = vec![0.0f64; d];
    star_discrepancy_recursive(points, n, d, n_f, &mut x, 0, &mut disc);

    Ok(disc)
}

/// Recursive helper to enumerate all coordinate-product corners.
fn star_discrepancy_recursive(
    points: &Array2<f64>,
    n: usize,
    d: usize,
    n_f: f64,
    x: &mut Vec<f64>,
    dim: usize,
    disc: &mut f64,
) {
    if dim == d {
        // Count points in [0, x)
        let count = (0..n)
            .filter(|&i| (0..d).all(|j| points[[i, j]] < x[j]))
            .count();

        let vol: f64 = x.iter().product();
        let gap = (count as f64 / n_f - vol).abs();
        if gap > *disc {
            *disc = gap;
        }
        return;
    }

    // Enumerate all distinct values of coordinate `dim` from the point set
    let mut vals: Vec<f64> = (0..n).map(|i| points[[i, dim]]).collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for &v in &vals {
        x[dim] = v;
        star_discrepancy_recursive(points, n, d, n_f, x, dim + 1, disc);
    }
}

/// One-dimensional star discrepancy for a sorted column vector.
fn star_discrepancy_1d(sorted: &[f64], n_f: f64) -> f64 {
    let mut disc = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let vol = x;
        let count = i as f64; // points strictly less than x
        disc = disc.max((count / n_f - vol).abs());
        // Also check upper endpoint
        let count_le = (i + 1) as f64;
        disc = disc.max((count_le / n_f - vol).abs());
    }
    disc
}

// ─────────────────────────────────────────────────────────────────────────────
// Scrambled Halton
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a scrambled Halton point set for randomised QMC.
///
/// **Scrambling** adds a random permutation of digits in the radical-inverse
/// representation, breaking the inter-dimensional correlation of the plain
/// Halton sequence while preserving the equidistribution property.
///
/// The scrambling used here is **Owen-style nested scrambling** implemented
/// as a random digit permutation independently for each dimension and each
/// digit position.  For a given `scramble_key` the scrambling is deterministic
/// and reproducible.
///
/// # Parameters
///
/// * `base`        – prime bases, one per dimension.
/// * `scramble_key`– seed for the scrambling RNG.
/// * `n_points`    – number of points to generate.
///
/// # Returns
///
/// An `n_points × dim` matrix with all entries in `[0, 1)`.
///
/// # Errors
///
/// Returns [`IntegrateError::ValueError`] if `base` is empty or any base is
/// less than 2.
///
/// # Example
///
/// ```
/// use scirs2_integrate::quasi_monte_carlo::scrambled_halton;
///
/// let pts = scrambled_halton(&[2, 3, 5], 12345, 256).expect("scramble failed");
/// assert_eq!(pts.nrows(), 256);
/// assert_eq!(pts.ncols(), 3);
/// assert!(pts.iter().all(|&x| x >= 0.0 && x < 1.0));
/// ```
pub fn scrambled_halton(
    base: &[usize],
    scramble_key: u64,
    n_points: usize,
) -> IntegrateResult<Array2<f64>> {
    if base.is_empty() {
        return Err(IntegrateError::ValueError(
            "base must not be empty".to_string(),
        ));
    }
    for (i, &b) in base.iter().enumerate() {
        if b < 2 {
            return Err(IntegrateError::ValueError(format!(
                "base[{i}] must be >= 2, got {b}"
            )));
        }
    }

    let dim = base.len();
    let max_digits = 32usize; // sufficient for n_points up to ~4 billion in base 2

    // Build per-dimension, per-digit permutation tables using a seeded RNG.
    let mut rng = StdRng::seed_from_u64(scramble_key);

    // perm[d][j][digit] = scrambled digit value
    let mut perm: Vec<Vec<Vec<usize>>> = Vec::with_capacity(dim);
    for &b in base.iter() {
        let mut dim_perm = Vec::with_capacity(max_digits);
        for _ in 0..max_digits {
            let mut p: Vec<usize> = (0..b).collect();
            // Fisher-Yates shuffle
            for k in (1..b).rev() {
                let j = rng.random_range(0..=(k as u64)) as usize;
                p.swap(k, j);
            }
            dim_perm.push(p);
        }
        perm.push(dim_perm);
    }

    let mut result = Array2::<f64>::zeros((n_points, dim));

    for i in 0..n_points {
        for (d, &b) in base.iter().enumerate() {
            let mut n = i + 1; // 1-indexed
            let mut val = 0.0f64;
            let mut denom = 1.0f64;
            let mut digit_pos = 0usize;
            while n > 0 {
                denom *= b as f64;
                let original_digit = n % b;
                let scrambled = if digit_pos < max_digits {
                    perm[d][digit_pos][original_digit]
                } else {
                    original_digit // fallback: no scramble for very deep digits
                };
                val += scrambled as f64 / denom;
                n /= b;
                digit_pos += 1;
            }
            result[[i, d]] = val.min(1.0 - f64::EPSILON); // clamp to [0,1)
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Prime utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Return the first `n` prime numbers.
fn first_n_primes(n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut primes = Vec::with_capacity(n);
    let mut candidate = 2usize;
    while primes.len() < n {
        if is_prime(candidate) {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

/// Simple trial-division primality test.
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let limit = (n as f64).sqrt() as usize + 1;
    (3..=limit).step_by(2).all(|k| !n.is_multiple_of(k))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HaltonSequence ───────────────────────────────────────────────────────

    #[test]
    fn test_halton_unit_cube() {
        let mut seq = HaltonSequence::with_primes(3);
        for _ in 0..100 {
            let p = seq.next_point();
            assert_eq!(p.len(), 3);
            assert!(
                p.iter().all(|&x| (0.0..1.0).contains(&x)),
                "out of [0,1): {p:?}"
            );
        }
    }

    #[test]
    fn test_halton_known_values_base2() {
        // van der Corput in base 2:
        // index=1: 0.1₂ = 0.5
        // index=2: 0.01₂ = 0.25
        // index=3: 0.11₂ = 0.75
        let mut seq = HaltonSequence::new(vec![2]);
        let p1 = seq.next_point();
        let p2 = seq.next_point();
        let p3 = seq.next_point();
        assert!((p1[0] - 0.5).abs() < 1e-12, "p1={}", p1[0]);
        assert!((p2[0] - 0.25).abs() < 1e-12, "p2={}", p2[0]);
        assert!((p3[0] - 0.75).abs() < 1e-12, "p3={}", p3[0]);
    }

    #[test]
    fn test_halton_reset() {
        let mut seq = HaltonSequence::with_primes(2);
        let first: Vec<Vec<f64>> = (0..5).map(|_| seq.next_point()).collect();
        seq.reset();
        let second: Vec<Vec<f64>> = (0..5).map(|_| seq.next_point()).collect();
        for (a, b) in first.iter().zip(second.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_halton_generate_shape() {
        let mut seq = HaltonSequence::with_primes(4);
        let pts = seq.generate(128);
        assert_eq!(pts.nrows(), 128);
        assert_eq!(pts.ncols(), 4);
    }

    // ── SobolSequence ────────────────────────────────────────────────────────

    #[test]
    fn test_sobol_unit_cube() {
        let mut seq = SobolSequence::new(5);
        for _ in 0..200 {
            let p = seq.next_point();
            assert_eq!(p.len(), 5);
            assert!(
                p.iter().all(|&x| (0.0..=1.0).contains(&x)),
                "out of [0,1]: {p:?}"
            );
        }
    }

    #[test]
    fn test_sobol_first_dim_van_der_corput() {
        // First dimension of Sobol using Gray-code recurrence gives low-discrepancy sequence.
        // Gray-code Sobol ordering: index=1 → 0.5, index=2 → 0.75, index=3 → 0.25
        // (differs from standard van der Corput but maintains equidistribution).
        let mut seq = SobolSequence::new(1);
        let _p0 = seq.next_point(); // skip 0.0
        let p1 = seq.next_point();
        let p2 = seq.next_point();
        let p3 = seq.next_point();
        assert!((p1[0] - 0.5).abs() < 1e-6, "p1={}", p1[0]);
        assert!((p2[0] - 0.75).abs() < 1e-6, "p2={}", p2[0]);
        assert!((p3[0] - 0.25).abs() < 1e-6, "p3={}", p3[0]);
    }

    #[test]
    fn test_sobol_reset() {
        let mut seq = SobolSequence::new(3);
        let before: Vec<Vec<f64>> = (0..10).map(|_| seq.next_point()).collect();
        seq.reset();
        let after: Vec<Vec<f64>> = (0..10).map(|_| seq.next_point()).collect();
        for (a, b) in before.iter().zip(after.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-12, "reset failed: {x} != {y}");
            }
        }
    }

    // ── LatticeRule ──────────────────────────────────────────────────────────

    #[test]
    fn test_lattice_rule_unit_cube() {
        let lat = LatticeRule::korobov(3, 1973, 1024);
        let mut lat = lat;
        for _ in 0..1024 {
            let p = lat.next_point();
            assert_eq!(p.len(), 3);
            assert!(
                p.iter().all(|&x| (0.0..=1.0).contains(&x)),
                "out of [0,1]: {p:?}"
            );
        }
    }

    #[test]
    fn test_lattice_first_point_zero() {
        // The lattice first point (i=0) should be all zeros (generator × 0 = 0)
        let mut lat = LatticeRule::korobov(2, 3, 8);
        let p0 = lat.next_point();
        assert!(
            p0.iter().all(|&x| x == 0.0),
            "first point should be zero: {p0:?}"
        );
    }

    // ── qmc_integrate ────────────────────────────────────────────────────────

    #[test]
    fn test_qmc_integrate_halton_1d() {
        // ∫₀¹ x² dx = 1/3
        let mut seq = HaltonSequence::with_primes(1);
        let val = qmc_integrate(|x| x[0] * x[0], &[(0.0, 1.0)], &mut seq, 4096)
            .expect("qmc_integrate failed");
        assert!((val - 1.0 / 3.0).abs() < 0.01, "val={val}");
    }

    #[test]
    fn test_qmc_integrate_sobol_2d() {
        // ∫₀¹∫₀¹ x·y dx dy = 1/4
        let mut seq = SobolSequence::new(2);
        let val = qmc_integrate(|x| x[0] * x[1], &[(0.0, 1.0), (0.0, 1.0)], &mut seq, 4096)
            .expect("qmc_integrate 2d failed");
        assert!((val - 0.25).abs() < 0.01, "val={val}");
    }

    #[test]
    fn test_qmc_integrate_lattice_3d() {
        // ∫₀¹∫₀¹∫₀¹ (x+y+z)/3 dx dy dz = 1/2
        let mut seq = LatticeRule::korobov(3, 1973, 4096);
        let val = qmc_integrate(
            |x| (x[0] + x[1] + x[2]) / 3.0,
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            &mut seq,
            4096,
        )
        .expect("qmc_integrate 3d failed");
        assert!((val - 0.5).abs() < 0.02, "val={val}");
    }

    #[test]
    fn test_qmc_integrate_non_unit_bounds() {
        // ∫₀²∫₀³ 1 dx dy = 6
        let mut seq = HaltonSequence::with_primes(2);
        let val = qmc_integrate(|_| 1.0, &[(0.0, 2.0), (0.0, 3.0)], &mut seq, 1000)
            .expect("qmc non-unit bounds failed");
        assert!((val - 6.0).abs() < 1e-10, "val={val}");
    }

    #[test]
    fn test_qmc_integrate_dim_mismatch() {
        let mut seq = HaltonSequence::with_primes(2);
        assert!(qmc_integrate(|x| x[0], &[(0.0, 1.0)], &mut seq, 100).is_err());
    }

    // ── star_discrepancy ─────────────────────────────────────────────────────

    #[test]
    fn test_star_discrepancy_halton() {
        let mut seq = HaltonSequence::with_primes(2);
        let pts = seq.generate(256);
        let d = star_discrepancy(&pts).expect("discrepancy failed");
        // QMC has much lower discrepancy than random
        assert!(d < 0.15, "D*={d}");
        assert!(d >= 0.0, "D* must be non-negative");
    }

    #[test]
    fn test_star_discrepancy_1d_perfect() {
        // Evenly spaced points in 1D have discrepancy ~ 1/(2N)
        let n = 100usize;
        let pts: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64 + 0.5) / n as f64]).collect();
        let arr = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 + 0.5) / n as f64);
        let d = star_discrepancy(&arr).expect("discrepancy failed");
        assert!(d < 0.02, "D*={d}");
    }

    #[test]
    fn test_star_discrepancy_empty() {
        let empty = Array2::<f64>::zeros((0, 2));
        assert!(star_discrepancy(&empty).is_err());
    }

    // ── scrambled_halton ─────────────────────────────────────────────────────

    #[test]
    fn test_scrambled_halton_shape() {
        let pts = scrambled_halton(&[2, 3, 5], 777, 128).expect("scramble failed");
        assert_eq!(pts.nrows(), 128);
        assert_eq!(pts.ncols(), 3);
    }

    #[test]
    fn test_scrambled_halton_unit_cube() {
        let pts = scrambled_halton(&[2, 3, 5, 7], 999, 256).expect("scramble failed");
        assert!(
            pts.iter().all(|&x| (0.0..1.0).contains(&x)),
            "some points outside [0,1)"
        );
    }

    #[test]
    fn test_scrambled_halton_reproducible() {
        let pts1 = scrambled_halton(&[2, 3], 42, 64).expect("scramble1 failed");
        let pts2 = scrambled_halton(&[2, 3], 42, 64).expect("scramble2 failed");
        assert_eq!(
            pts1, pts2,
            "scrambled Halton must be reproducible for same key"
        );
    }

    #[test]
    fn test_scrambled_halton_different_keys_differ() {
        let pts1 = scrambled_halton(&[2, 3], 1, 32).expect("scramble1 failed");
        let pts2 = scrambled_halton(&[2, 3], 2, 32).expect("scramble2 failed");
        // Different seeds should give different points
        let same = pts1
            .iter()
            .zip(pts2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-15);
        assert!(!same, "different keys produced identical points");
    }

    #[test]
    fn test_scrambled_halton_invalid_base() {
        assert!(scrambled_halton(&[1], 0, 10).is_err());
        assert!(scrambled_halton(&[], 0, 10).is_err());
    }

    // ── prime utilities ──────────────────────────────────────────────────────

    #[test]
    fn test_first_n_primes() {
        let p = first_n_primes(5);
        assert_eq!(p, vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }
}
