//! FFT Plan Serialization — Algorithm-Agnostic Plan Creation and Execution
//!
//! This module provides a pure-Rust FFT planning system that is independent of
//! any external backend (OxiFFT is the sole backend — COOLJAPAN Pure Rust policy).  It supports:
//!
//! - **Plan creation**: analyse an input size, choose a factorization tree, and
//!   pre-compute twiddle factors.
//! - **Plan execution**: apply a pre-built plan to data without recomputing the
//!   factorization or twiddles.
//! - **Serialization / deserialization**: persist a plan to bytes (via
//!   `serde` + JSON) for ahead-of-time compilation workflows.
//! - **Plan reuse**: the same plan object can be executed on any same-size input.
//!
//! # Example
//!
//! ```rust
//! use scirs2_fft::fft_plan::{FftPlanConfig, FftAlgorithm, create_plan, execute_plan};
//! use scirs2_fft::fft_plan::{serialize_plan, deserialize_plan};
//! use scirs2_core::numeric::Complex64;
//!
//! let config = FftPlanConfig {
//!     size: 8,
//!     algorithm: FftAlgorithm::CacheOblivious,
//!     precompute_twiddles: true,
//! };
//! let plan = create_plan(&config).expect("plan creation failed");
//!
//! let mut data: Vec<Complex64> = (0..8)
//!     .map(|k| Complex64::new(k as f64, 0.0))
//!     .collect();
//! execute_plan(&plan, &mut data).expect("execution failed");
//!
//! // Serialize / deserialize roundtrip
//! let bytes = serialize_plan(&plan).expect("serialize failed");
//! let plan2 = deserialize_plan(&bytes).expect("deserialize failed");
//! ```

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use scirs2_core::numeric::Complex64;

use crate::bluestein;
use crate::butterfly::{direct_dft, generate_twiddle_table};
use crate::cache_oblivious::{cache_oblivious_fft, cache_oblivious_ifft};
use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Supported FFT algorithm families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[derive(Default)]
pub enum FftAlgorithm {
    /// Cooley-Tukey radix-2 DIT (power-of-two sizes only).
    CooleyTukey,
    /// Split-radix algorithm (power-of-two sizes only).
    SplitRadix,
    /// Cache-oblivious four-step algorithm (any composite size).
    #[default]
    CacheOblivious,
    /// Bluestein's chirp-z transform (any size including primes).
    Bluestein,
    /// Rader's algorithm for prime sizes.
    Rader,
}

/// Configuration for plan creation.
#[derive(Debug, Clone)]
pub struct FftPlanConfig {
    /// Transform size (must be >= 1).
    pub size: usize,
    /// Algorithm to use.
    pub algorithm: FftAlgorithm,
    /// Whether to pre-compute and store twiddle factors in the plan.
    pub precompute_twiddles: bool,
}

impl Default for FftPlanConfig {
    fn default() -> Self {
        Self {
            size: 0,
            algorithm: FftAlgorithm::default(),
            precompute_twiddles: true,
        }
    }
}

/// Serializable helper for Complex64 (serde doesn't derive for num-complex by default).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SerComplex {
    re: f64,
    im: f64,
}

impl From<Complex64> for SerComplex {
    fn from(c: Complex64) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl From<SerComplex> for Complex64 {
    fn from(s: SerComplex) -> Self {
        Complex64::new(s.re, s.im)
    }
}

/// A node in the decomposition tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PlanNode {
    /// Leaf: use a direct DFT of size `n`.
    DirectDft {
        /// Transform size.
        n: usize,
    },
    /// Cooley-Tukey radix-2 decomposition.
    CooleyTukey {
        /// Total size (must be power of two).
        n: usize,
        /// Pre-computed twiddle factors (may be empty if not pre-computed).
        twiddles: Vec<SerComplex>,
    },
    /// Cache-oblivious four-step decomposition: `n = n1 * n2`.
    FourStep {
        /// Total size.
        n: usize,
        /// Row count.
        n1: usize,
        /// Column count.
        n2: usize,
        /// Twiddle factors for the twiddle-multiply step.
        twiddles: Vec<SerComplex>,
    },
    /// Bluestein's algorithm (any size).
    Bluestein {
        /// Transform size.
        n: usize,
    },
    /// Split-radix (power-of-two).
    SplitRadix {
        /// Transform size.
        n: usize,
        /// Pre-computed twiddle factors.
        twiddles: Vec<SerComplex>,
    },
    /// Rader's algorithm (prime size).
    Rader {
        /// Transform size.
        n: usize,
    },
}

/// An FFT plan: a decomposition tree plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FftPlan {
    /// The size this plan was built for.
    pub size: usize,
    /// The chosen algorithm.
    pub algorithm: FftAlgorithm,
    /// The root of the decomposition tree.
    pub root: PlanNode,
    /// Whether twiddles were pre-computed.
    pub precomputed: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Plan creation
// ─────────────────────────────────────────────────────────────────────────────

/// Create an FFT plan for the given configuration.
///
/// The plan pre-analyses the input size, selects a factorization strategy,
/// and optionally pre-computes twiddle factors.
///
/// # Errors
///
/// Returns [`FFTError::PlanError`] if the size is 0, or if the chosen
/// algorithm is incompatible with the given size.
pub fn create_plan(config: &FftPlanConfig) -> FFTResult<FftPlan> {
    let n = config.size;
    if n == 0 {
        return Err(FFTError::PlanError("create_plan: size must be >= 1".into()));
    }

    let root = match config.algorithm {
        FftAlgorithm::CooleyTukey => {
            if !n.is_power_of_two() {
                return Err(FFTError::PlanError(
                    "CooleyTukey requires power-of-two size".into(),
                ));
            }
            let twiddles = if config.precompute_twiddles {
                generate_twiddle_table(n)?
                    .into_iter()
                    .map(SerComplex::from)
                    .collect()
            } else {
                Vec::new()
            };
            PlanNode::CooleyTukey { n, twiddles }
        }
        FftAlgorithm::SplitRadix => {
            if !n.is_power_of_two() {
                return Err(FFTError::PlanError(
                    "SplitRadix requires power-of-two size".into(),
                ));
            }
            let twiddles = if config.precompute_twiddles {
                generate_twiddle_table(n)?
                    .into_iter()
                    .map(SerComplex::from)
                    .collect()
            } else {
                Vec::new()
            };
            PlanNode::SplitRadix { n, twiddles }
        }
        FftAlgorithm::CacheOblivious => {
            if n <= 16 {
                PlanNode::DirectDft { n }
            } else {
                // Try to factor
                let (n1, n2) = find_factor_pair(n);
                let twiddles = if config.precompute_twiddles {
                    compute_four_step_twiddles(n, n1, n2)
                        .into_iter()
                        .map(SerComplex::from)
                        .collect()
                } else {
                    Vec::new()
                };
                if n1 == 1 || n2 == 1 {
                    // Prime: fall back to Bluestein
                    PlanNode::Bluestein { n }
                } else {
                    PlanNode::FourStep {
                        n,
                        n1,
                        n2,
                        twiddles,
                    }
                }
            }
        }
        FftAlgorithm::Bluestein => PlanNode::Bluestein { n },
        FftAlgorithm::Rader => {
            if !is_prime(n) {
                return Err(FFTError::PlanError(
                    "Rader algorithm requires prime size".into(),
                ));
            }
            PlanNode::Rader { n }
        }
    };

    Ok(FftPlan {
        size: n,
        algorithm: config.algorithm,
        root,
        precomputed: config.precompute_twiddles,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Plan execution
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a pre-built FFT plan on `data` (in-place).
///
/// The data length **must** match `plan.size`.
///
/// # Errors
///
/// Returns [`FFTError::DimensionError`] if `data.len() != plan.size`.
pub fn execute_plan(plan: &FftPlan, data: &mut [Complex64]) -> FFTResult<()> {
    if data.len() != plan.size {
        return Err(FFTError::DimensionError(format!(
            "execute_plan: expected data of length {}, got {}",
            plan.size,
            data.len()
        )));
    }

    let result = execute_node(&plan.root, data)?;
    data.copy_from_slice(&result);
    Ok(())
}

/// Internal: execute a plan node and return the DFT result.
fn execute_node(node: &PlanNode, data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    match node {
        PlanNode::DirectDft { n } => {
            debug_assert_eq!(*n, data.len());
            if *n <= 1 {
                Ok(data.to_vec())
            } else {
                direct_dft(data)
            }
        }
        PlanNode::CooleyTukey { n, twiddles } => {
            debug_assert_eq!(*n, data.len());
            execute_cooley_tukey(data, *n, twiddles)
        }
        PlanNode::SplitRadix { n, .. } => {
            debug_assert_eq!(*n, data.len());
            // Use split-radix via butterfly module
            let mut buf = data.to_vec();
            crate::butterfly::split_radix_butterfly(&mut buf)?;
            Ok(buf)
        }
        PlanNode::FourStep { n, n1, n2, .. } => {
            debug_assert_eq!(*n, data.len());
            cache_oblivious_fft(data).or_else(|_| {
                // Fallback: direct DFT
                let _ = (n1, n2); // suppress unused warning
                direct_dft(data)
            })
        }
        PlanNode::Bluestein { n } => {
            debug_assert_eq!(*n, data.len());
            bluestein::bluestein_fft(data)
        }
        PlanNode::Rader { n } => {
            debug_assert_eq!(*n, data.len());
            // Rader falls back to Bluestein for now
            bluestein::bluestein_fft(data)
        }
    }
}

/// Cooley-Tukey radix-2 DIT FFT with optional pre-computed twiddles.
fn execute_cooley_tukey(
    data: &[Complex64],
    n: usize,
    precomputed_twiddles: &[SerComplex],
) -> FFTResult<Vec<Complex64>> {
    if n == 1 {
        return Ok(data.to_vec());
    }

    // Bit-reversal permutation
    let mut output = bit_reverse_copy(data, n);

    // Iterative butterfly passes
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let step = n / size;

        for k in 0..half {
            let twiddle = if !precomputed_twiddles.is_empty() {
                let idx = k * step;
                if idx < precomputed_twiddles.len() {
                    Complex64::from(precomputed_twiddles[idx])
                } else {
                    let angle = -2.0 * PI * (k * step) as f64 / n as f64;
                    Complex64::new(angle.cos(), angle.sin())
                }
            } else {
                let angle = -2.0 * PI * (k * step) as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            };

            let mut j = k;
            while j < n {
                let u = output[j];
                let t = twiddle * output[j + half];
                output[j] = u + t;
                output[j + half] = u - t;
                j += size;
            }
        }
        size *= 2;
    }

    Ok(output)
}

/// Bit-reversal permutation of `data`.
fn bit_reverse_copy(data: &[Complex64], n: usize) -> Vec<Complex64> {
    let bits = n.trailing_zeros();
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let rev = reverse_bits(i, bits);
        out[rev] = data[i];
    }
    out
}

/// Reverse the lower `bits` bits of `x`.
fn reverse_bits(x: usize, bits: u32) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
//  Serialization
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize an [`FftPlan`] to a byte vector (JSON format).
///
/// # Errors
///
/// Returns [`FFTError::IOError`] if serialization fails.
pub fn serialize_plan(plan: &FftPlan) -> FFTResult<Vec<u8>> {
    serde_json::to_vec(plan).map_err(|e| FFTError::IOError(format!("serialize_plan: {e}")))
}

/// Deserialize an [`FftPlan`] from a byte slice (JSON format).
///
/// # Errors
///
/// Returns [`FFTError::IOError`] if deserialization fails.
pub fn deserialize_plan(data: &[u8]) -> FFTResult<FftPlan> {
    serde_json::from_slice(data).map_err(|e| FFTError::IOError(format!("deserialize_plan: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find a factor pair `(n1, n2)` with `n1 * n2 == n` and `n1 ≈ sqrt(n)`.
/// If `n` is prime, returns `(1, n)`.
fn find_factor_pair(n: usize) -> (usize, usize) {
    if n <= 1 {
        return (1, n);
    }
    let sqrt_n = (n as f64).sqrt() as usize;
    for candidate in (2..=sqrt_n).rev() {
        if n % candidate == 0 {
            return (candidate, n / candidate);
        }
    }
    (1, n) // prime
}

/// Compute the N1*N2 twiddle factors for the four-step algorithm.
fn compute_four_step_twiddles(n: usize, n1: usize, n2: usize) -> Vec<Complex64> {
    let angle_base = -2.0 * PI / n as f64;
    let mut twiddles = Vec::with_capacity(n1 * n2);
    for i in 0..n1 {
        for j in 0..n2 {
            let angle = angle_base * (i * j) as f64;
            twiddles.push(Complex64::new(angle.cos(), angle.sin()));
        }
    }
    twiddles
}

/// Simple primality test.
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn max_abs_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    fn reference_dft(data: &[Complex64]) -> Vec<Complex64> {
        direct_dft(data).expect("direct_dft failed")
    }

    // ── plan creation ───────────────────────────────────────────────────
    #[test]
    fn test_create_plan_cooley_tukey() {
        let config = FftPlanConfig {
            size: 16,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: true,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        assert_eq!(plan.size, 16);
        assert_eq!(plan.algorithm, FftAlgorithm::CooleyTukey);
        assert!(plan.precomputed);
    }

    #[test]
    fn test_create_plan_cooley_tukey_non_pow2_fails() {
        let config = FftPlanConfig {
            size: 12,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: false,
        };
        assert!(create_plan(&config).is_err());
    }

    #[test]
    fn test_create_plan_cache_oblivious() {
        for &n in &[8, 16, 32, 64, 12, 15, 24] {
            let config = FftPlanConfig {
                size: n,
                algorithm: FftAlgorithm::CacheOblivious,
                precompute_twiddles: true,
            };
            let plan = create_plan(&config).expect("create_plan failed");
            assert_eq!(plan.size, n);
        }
    }

    #[test]
    fn test_create_plan_bluestein() {
        for &n in &[7, 11, 13, 17, 100] {
            let config = FftPlanConfig {
                size: n,
                algorithm: FftAlgorithm::Bluestein,
                precompute_twiddles: false,
            };
            let plan = create_plan(&config).expect("create_plan failed");
            assert_eq!(plan.size, n);
        }
    }

    #[test]
    fn test_create_plan_rader_prime() {
        let config = FftPlanConfig {
            size: 7,
            algorithm: FftAlgorithm::Rader,
            precompute_twiddles: false,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        assert_eq!(plan.size, 7);
    }

    #[test]
    fn test_create_plan_rader_non_prime_fails() {
        let config = FftPlanConfig {
            size: 12,
            algorithm: FftAlgorithm::Rader,
            precompute_twiddles: false,
        };
        assert!(create_plan(&config).is_err());
    }

    #[test]
    fn test_create_plan_zero_size_fails() {
        let config = FftPlanConfig {
            size: 0,
            algorithm: FftAlgorithm::CacheOblivious,
            precompute_twiddles: false,
        };
        assert!(create_plan(&config).is_err());
    }

    // ── plan execution ──────────────────────────────────────────────────
    #[test]
    fn test_execute_plan_cooley_tukey() {
        let config = FftPlanConfig {
            size: 8,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: true,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        let input: Vec<Complex64> = (0..8).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let expected = reference_dft(&input);

        let mut data = input;
        execute_plan(&plan, &mut data).expect("execute_plan failed");
        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-10, "CT execution error = {err}");
    }

    #[test]
    fn test_execute_plan_bluestein() {
        let config = FftPlanConfig {
            size: 7,
            algorithm: FftAlgorithm::Bluestein,
            precompute_twiddles: false,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        let input: Vec<Complex64> = (0..7).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let expected = reference_dft(&input);

        let mut data = input;
        execute_plan(&plan, &mut data).expect("execute_plan failed");
        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-10, "Bluestein execution error = {err}");
    }

    #[test]
    fn test_execute_plan_wrong_size() {
        let config = FftPlanConfig {
            size: 8,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: false,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        let mut data = vec![Complex64::new(1.0, 0.0); 16];
        assert!(execute_plan(&plan, &mut data).is_err());
    }

    #[test]
    fn test_execute_plan_matches_direct_fft() {
        // Test multiple algorithms and sizes
        let test_cases = vec![
            (8, FftAlgorithm::CooleyTukey),
            (16, FftAlgorithm::CooleyTukey),
            (8, FftAlgorithm::SplitRadix),
            (7, FftAlgorithm::Bluestein),
            (13, FftAlgorithm::Bluestein),
        ];
        for (n, algo) in test_cases {
            let config = FftPlanConfig {
                size: n,
                algorithm: algo,
                precompute_twiddles: true,
            };
            let plan = create_plan(&config).expect("plan creation failed");
            let input: Vec<Complex64> = (0..n)
                .map(|k| Complex64::new((k as f64 * 0.5).sin(), (k as f64 * 0.3).cos()))
                .collect();
            let expected = reference_dft(&input);
            let mut data = input;
            execute_plan(&plan, &mut data).expect("execution failed");
            let err = max_abs_err(&data, &expected);
            assert!(err < 1e-8, "{algo:?} n={n}: error = {err}");
        }
    }

    // ── serialization roundtrip ─────────────────────────────────────────
    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let config = FftPlanConfig {
            size: 16,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: true,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        let bytes = serialize_plan(&plan).expect("serialize failed");
        assert!(!bytes.is_empty());

        let plan2 = deserialize_plan(&bytes).expect("deserialize failed");
        assert_eq!(plan.size, plan2.size);
        assert_eq!(plan.algorithm, plan2.algorithm);
        assert_eq!(plan.precomputed, plan2.precomputed);
    }

    #[test]
    fn test_serialized_plan_produces_same_result() {
        let config = FftPlanConfig {
            size: 8,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: true,
        };
        let plan = create_plan(&config).expect("create_plan failed");
        let bytes = serialize_plan(&plan).expect("serialize failed");
        let plan2 = deserialize_plan(&bytes).expect("deserialize failed");

        let input: Vec<Complex64> = (0..8).map(|k| Complex64::new(k as f64, 0.0)).collect();

        let mut data1 = input.clone();
        let mut data2 = input;
        execute_plan(&plan, &mut data1).expect("exec1 failed");
        execute_plan(&plan2, &mut data2).expect("exec2 failed");

        let err = max_abs_err(&data1, &data2);
        assert!(err < 1e-14, "serialized plan diverges: {err}");
    }

    #[test]
    fn test_deserialize_invalid_data() {
        assert!(deserialize_plan(b"not json").is_err());
    }

    // ── plan reuse ──────────────────────────────────────────────────────
    #[test]
    fn test_plan_reuse_same_result() {
        let config = FftPlanConfig {
            size: 16,
            algorithm: FftAlgorithm::CooleyTukey,
            precompute_twiddles: true,
        };
        let plan = create_plan(&config).expect("create_plan failed");

        let input1: Vec<Complex64> = (0..16).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let input2: Vec<Complex64> = (0..16).map(|k| Complex64::new(0.0, k as f64)).collect();

        let mut data1 = input1.clone();
        let mut data2 = input2.clone();
        execute_plan(&plan, &mut data1).expect("exec1 failed");
        execute_plan(&plan, &mut data2).expect("exec2 failed");

        let expected1 = reference_dft(&input1);
        let expected2 = reference_dft(&input2);

        assert!(max_abs_err(&data1, &expected1) < 1e-10);
        assert!(max_abs_err(&data2, &expected2) < 1e-10);
    }

    // ── internal helpers ────────────────────────────────────────────────
    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(!is_prime(9));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(!is_prime(15));
        assert!(is_prime(17));
    }

    #[test]
    fn test_find_factor_pair() {
        let (a, b) = find_factor_pair(12);
        assert_eq!(a * b, 12);
        assert!(a >= 2);

        let (a, b) = find_factor_pair(16);
        assert_eq!(a * b, 16);
        assert!(a >= 2);

        // Prime: (1, n)
        let (a, b) = find_factor_pair(13);
        assert_eq!(a, 1);
        assert_eq!(b, 13);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(reverse_bits(0b000, 3), 0b000);
        assert_eq!(reverse_bits(0b001, 3), 0b100);
        assert_eq!(reverse_bits(0b010, 3), 0b010);
        assert_eq!(reverse_bits(0b011, 3), 0b110);
        assert_eq!(reverse_bits(0b100, 3), 0b001);
    }
}
