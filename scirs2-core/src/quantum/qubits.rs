//! Qubit state representation and multi-qubit register operations.
//!
//! This module provides the fundamental quantum state types used throughout
//! the quantum simulation library:
//!
//! - [`Qubit`]: a single-qubit pure state stored as a pair of complex amplitudes.
//! - [`QubitRegister`]: an n-qubit register stored as a statevector of 2^n amplitudes.
//!
//! # Conventions
//!
//! States are stored in the **computational basis** |0⟩, |1⟩, …, |2^n-1⟩ ordered
//! from least-significant qubit to most-significant qubit, i.e. qubit 0 is the
//! rightmost (fastest-varying) index.
//!
//! For a 2-qubit system the basis order is:
//! ```text
//! index 0 → |q1=0, q0=0⟩ = |00⟩
//! index 1 → |q1=0, q0=1⟩ = |01⟩
//! index 2 → |q1=1, q0=0⟩ = |10⟩
//! index 3 → |q1=1, q0=1⟩ = |11⟩
//! ```
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::quantum::qubits::{Qubit, QubitRegister};
//!
//! let q0 = Qubit::new_zero();
//! let q1 = Qubit::new_one();
//!
//! // Build a 2-qubit register |0⟩ ⊗ |1⟩ = |01⟩
//! let r0 = QubitRegister::from_qubit(&q0);
//! let r1 = QubitRegister::from_qubit(&q1);
//! let reg = QubitRegister::tensor_product(&r0, &r1);
//! assert_eq!(reg.n_qubits(), 2);
//! assert_eq!(reg.dim(), 4);
//! ```

use num_complex::Complex;
use rand::{Rng, RngExt};
use std::f64::consts::PI;

use super::error::{QuantumError, QuantumResult};

/// A single-qubit pure state: |ψ⟩ = α|0⟩ + β|1⟩.
///
/// Invariant: |α|² + |β|² = 1 (up to floating-point tolerance).
#[derive(Debug, Clone, PartialEq)]
pub struct Qubit {
    /// Amplitude for |0⟩.
    pub(crate) alpha: Complex<f64>,
    /// Amplitude for |1⟩.
    pub(crate) beta: Complex<f64>,
}

impl Qubit {
    /// Construct a qubit from raw amplitudes, normalising automatically.
    ///
    /// Returns an error if the norm is zero (unphysical).
    pub fn new(alpha: Complex<f64>, beta: Complex<f64>) -> QuantumResult<Self> {
        let norm_sq = alpha.norm_sqr() + beta.norm_sqr();
        if norm_sq < 1e-15 {
            return Err(QuantumError::ZeroStateVector);
        }
        let norm = norm_sq.sqrt();
        Ok(Self {
            alpha: alpha / norm,
            beta: beta / norm,
        })
    }

    /// |0⟩ state.
    pub fn new_zero() -> Self {
        Self {
            alpha: Complex::new(1.0, 0.0),
            beta: Complex::new(0.0, 0.0),
        }
    }

    /// |1⟩ state.
    pub fn new_one() -> Self {
        Self {
            alpha: Complex::new(0.0, 0.0),
            beta: Complex::new(1.0, 0.0),
        }
    }

    /// Bloch-sphere parametrisation: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
    ///
    /// - `theta` ∈ [0, π]: polar angle.
    /// - `phi`   ∈ [0, 2π): azimuthal angle.
    pub fn new_superposition(theta: f64, phi: f64) -> Self {
        let half = theta / 2.0;
        Self {
            alpha: Complex::new(half.cos(), 0.0),
            beta: Complex::from_polar(half.sin(), phi),
        }
    }

    /// Equal superposition: (|0⟩ + |1⟩) / √2.
    pub fn new_plus() -> Self {
        let s = 1.0 / 2.0_f64.sqrt();
        Self {
            alpha: Complex::new(s, 0.0),
            beta: Complex::new(s, 0.0),
        }
    }

    /// Equal superposition: (|0⟩ − |1⟩) / √2.
    pub fn new_minus() -> Self {
        let s = 1.0 / 2.0_f64.sqrt();
        Self {
            alpha: Complex::new(s, 0.0),
            beta: Complex::new(-s, 0.0),
        }
    }

    /// Amplitude for |0⟩.
    pub fn alpha(&self) -> Complex<f64> {
        self.alpha
    }

    /// Amplitude for |1⟩.
    pub fn beta(&self) -> Complex<f64> {
        self.beta
    }

    /// Probability of measuring |0⟩.
    pub fn prob_zero(&self) -> f64 {
        self.alpha.norm_sqr()
    }

    /// Probability of measuring |1⟩.
    pub fn prob_one(&self) -> f64 {
        self.beta.norm_sqr()
    }

    /// Check normalisation; returns `true` if within `tol` of 1.
    pub fn is_normalised(&self, tol: f64) -> bool {
        ((self.alpha.norm_sqr() + self.beta.norm_sqr()) - 1.0).abs() < tol
    }

    /// Perform a projective measurement in the computational basis.
    ///
    /// Returns `(outcome, post_measurement_state)` where `outcome` is 0 or 1
    /// and the post-measurement state has collapsed to the corresponding basis state.
    ///
    /// Uses `rng` to sample the Born-rule distribution.
    pub fn measure<R: Rng>(&self, rng: &mut R) -> (u8, Qubit) {
        let p0 = self.prob_zero();
        let sample: f64 = rng.random();
        if sample < p0 {
            (0, Qubit::new_zero())
        } else {
            (1, Qubit::new_one())
        }
    }

    /// Bloch-sphere angles (theta, phi) for this qubit.
    ///
    /// Returns `(theta, phi)` where θ ∈ [0, π] and φ ∈ [0, 2π).
    pub fn bloch_angles(&self) -> (f64, f64) {
        let theta = 2.0 * self.alpha.norm().acos().min(PI);
        let phi = {
            let raw = self.beta.arg() - self.alpha.arg();
            raw.rem_euclid(2.0 * PI)
        };
        (theta, phi)
    }

    /// Convert this single-qubit state into a [`QubitRegister`].
    pub fn to_register(&self) -> QubitRegister {
        QubitRegister {
            amplitudes: vec![self.alpha, self.beta],
            n_qubits: 1,
        }
    }
}

impl Default for Qubit {
    fn default() -> Self {
        Self::new_zero()
    }
}

impl std::fmt::Display for Qubit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:.6} + {:.6}i)|0⟩ + ({:.6} + {:.6}i)|1⟩",
            self.alpha.re, self.alpha.im, self.beta.re, self.beta.im
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QubitRegister
// ─────────────────────────────────────────────────────────────────────────────

/// An n-qubit quantum register stored as a 2^n-dimensional statevector.
///
/// The state is always kept normalised.  Amplitudes are ordered by the binary
/// representation of the basis index with qubit 0 as the least-significant bit.
#[derive(Debug, Clone, PartialEq)]
pub struct QubitRegister {
    /// Statevector amplitudes — length must equal 2^n_qubits.
    pub(crate) amplitudes: Vec<Complex<f64>>,
    /// Number of qubits.
    pub(crate) n_qubits: usize,
}

impl QubitRegister {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a register from a raw amplitude vector.
    ///
    /// `n_qubits` must satisfy `amplitudes.len() == 2^n_qubits`.
    /// The vector is automatically re-normalised.
    pub fn new(n_qubits: usize, amplitudes: Vec<Complex<f64>>) -> QuantumResult<Self> {
        let expected_dim = 1usize
            .checked_shl(n_qubits as u32)
            .ok_or(QuantumError::TooManyQubits(n_qubits))?;
        if amplitudes.len() != expected_dim {
            return Err(QuantumError::DimensionMismatch {
                expected: expected_dim,
                actual: amplitudes.len(),
            });
        }
        let norm_sq: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq < 1e-15 {
            return Err(QuantumError::ZeroStateVector);
        }
        let norm = norm_sq.sqrt();
        let normalised = amplitudes.iter().map(|a| a / norm).collect();
        Ok(Self {
            amplitudes: normalised,
            n_qubits,
        })
    }

    /// Wrap a single `Qubit` as a 1-qubit register.
    pub fn from_qubit(q: &Qubit) -> Self {
        Self {
            amplitudes: vec![q.alpha, q.beta],
            n_qubits: 1,
        }
    }

    /// All-zeros state |0…0⟩.
    pub fn new_zero_state(n_qubits: usize) -> QuantumResult<Self> {
        if n_qubits == 0 {
            return Err(QuantumError::InvalidQubitCount(n_qubits));
        }
        let dim = 1usize
            .checked_shl(n_qubits as u32)
            .ok_or(QuantumError::TooManyQubits(n_qubits))?;
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[0] = Complex::new(1.0, 0.0);
        Ok(Self {
            amplitudes: amps,
            n_qubits,
        })
    }

    /// Equal superposition (Hadamard applied to all qubits of |0…0⟩).
    ///
    /// Each amplitude has magnitude 1/√(2^n).
    pub fn new_uniform_superposition(n_qubits: usize) -> QuantumResult<Self> {
        if n_qubits == 0 {
            return Err(QuantumError::InvalidQubitCount(n_qubits));
        }
        let dim = 1usize
            .checked_shl(n_qubits as u32)
            .ok_or(QuantumError::TooManyQubits(n_qubits))?;
        let amp = Complex::new(1.0 / (dim as f64).sqrt(), 0.0);
        Ok(Self {
            amplitudes: vec![amp; dim],
            n_qubits,
        })
    }

    /// Computational basis state |k⟩ for a given integer `k` < 2^n.
    pub fn new_basis_state(n_qubits: usize, k: usize) -> QuantumResult<Self> {
        if n_qubits == 0 {
            return Err(QuantumError::InvalidQubitCount(n_qubits));
        }
        let dim = 1usize
            .checked_shl(n_qubits as u32)
            .ok_or(QuantumError::TooManyQubits(n_qubits))?;
        if k >= dim {
            return Err(QuantumError::BasisIndexOutOfRange { index: k, dim });
        }
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[k] = Complex::new(1.0, 0.0);
        Ok(Self {
            amplitudes: amps,
            n_qubits,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of qubits in this register.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Hilbert-space dimension: 2^n.
    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Slice over all amplitudes.
    pub fn amplitudes(&self) -> &[Complex<f64>] {
        &self.amplitudes
    }

    /// Mutable slice over all amplitudes (use with care — normalisation is not
    /// automatically maintained).
    pub fn amplitudes_mut(&mut self) -> &mut Vec<Complex<f64>> {
        &mut self.amplitudes
    }

    /// Amplitude for basis state `k`.
    pub fn amplitude(&self, k: usize) -> QuantumResult<Complex<f64>> {
        self.amplitudes
            .get(k)
            .copied()
            .ok_or(QuantumError::BasisIndexOutOfRange {
                index: k,
                dim: self.dim(),
            })
    }

    /// Probability of measuring basis state `k`.
    pub fn probability(&self, k: usize) -> QuantumResult<f64> {
        Ok(self.amplitude(k)?.norm_sqr())
    }

    /// All measurement probabilities |ψ_k|² in basis order.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Re-normalise the statevector in place.
    pub fn normalise(&mut self) -> QuantumResult<()> {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq < 1e-15 {
            return Err(QuantumError::ZeroStateVector);
        }
        let norm = norm_sq.sqrt();
        for a in &mut self.amplitudes {
            *a /= norm;
        }
        Ok(())
    }

    /// Check that the statevector is normalised within `tol`.
    pub fn is_normalised(&self, tol: f64) -> bool {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        (norm_sq - 1.0).abs() < tol
    }

    // ── Operations ───────────────────────────────────────────────────────────

    /// Tensor product (Kronecker product) of two registers: result = a ⊗ b.
    ///
    /// If `a` has n₁ qubits and `b` has n₂ qubits, the result has n₁+n₂ qubits.
    /// Qubit ordering: qubits of `a` are the *more* significant bits.
    pub fn tensor_product(a: &QubitRegister, b: &QubitRegister) -> QubitRegister {
        let n = a.n_qubits + b.n_qubits;
        let mut amps = Vec::with_capacity(a.dim() * b.dim());
        for &amp_a in &a.amplitudes {
            for &amp_b in &b.amplitudes {
                amps.push(amp_a * amp_b);
            }
        }
        QubitRegister {
            amplitudes: amps,
            n_qubits: n,
        }
    }

    /// Measure a single qubit at index `qubit_idx` and return `(outcome, collapsed_state)`.
    ///
    /// The returned register has the same number of qubits; amplitudes inconsistent
    /// with the measurement outcome are zeroed and the result is renormalised.
    pub fn measure_qubit<R: Rng>(
        &self,
        qubit_idx: usize,
        rng: &mut R,
    ) -> QuantumResult<(u8, QubitRegister)> {
        if qubit_idx >= self.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: qubit_idx,
                n_qubits: self.n_qubits,
            });
        }

        // Compute probability of measuring |1⟩ on this qubit.
        let mut prob_one: f64 = 0.0;
        for (k, amp) in self.amplitudes.iter().enumerate() {
            if (k >> qubit_idx) & 1 == 1 {
                prob_one += amp.norm_sqr();
            }
        }
        let sample: f64 = rng.random();
        let outcome: u8 = if sample < prob_one { 1 } else { 0 };

        // Project and renormalise.
        let mut new_amps = self.amplitudes.clone();
        for (k, amp) in new_amps.iter_mut().enumerate() {
            let bit = ((k >> qubit_idx) & 1) as u8;
            if bit != outcome {
                *amp = Complex::new(0.0, 0.0);
            }
        }
        let mut collapsed = QubitRegister {
            amplitudes: new_amps,
            n_qubits: self.n_qubits,
        };
        collapsed.normalise()?;
        Ok((outcome, collapsed))
    }

    /// Measure all qubits and return a bit-string outcome (qubit 0 first).
    ///
    /// Samples once from the Born-rule distribution.
    pub fn measure_all<R: Rng>(&self, rng: &mut R) -> Vec<u8> {
        // Build CDF and sample.
        let probs = self.probabilities();
        let sample: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut outcome_index = probs.len().saturating_sub(1);
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if sample < cumulative {
                outcome_index = i;
                break;
            }
        }
        // Decode basis index into bits (qubit 0 = LSB).
        (0..self.n_qubits)
            .map(|q| ((outcome_index >> q) & 1) as u8)
            .collect()
    }

    /// Inner product ⟨other|self⟩.
    pub fn inner_product(&self, other: &QubitRegister) -> QuantumResult<Complex<f64>> {
        if self.n_qubits != other.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }
        let ip = self
            .amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| b.conj() * a)
            .sum();
        Ok(ip)
    }

    /// Fidelity |⟨other|self⟩|² with another register.
    pub fn fidelity(&self, other: &QubitRegister) -> QuantumResult<f64> {
        let ip = self.inner_product(other)?;
        Ok(ip.norm_sqr())
    }

    /// Von-Neumann entropy (in nats) of the full pure state (always 0 for pure states).
    /// Provided for completeness; useful as a sanity-check (should return 0.0).
    pub fn entropy(&self) -> f64 {
        let probs = self.probabilities();
        -probs
            .iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

impl Default for QubitRegister {
    fn default() -> Self {
        // 1-qubit |0⟩ state
        Self {
            amplitudes: vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            n_qubits: 1,
        }
    }
}

impl std::fmt::Display for QubitRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QubitRegister({} qubits) [", self.n_qubits)?;
        for (i, amp) in self.amplitudes.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if amp.im >= 0.0 {
                write!(
                    f,
                    "|{:0>width$b}⟩: {:.4}+{:.4}i",
                    i,
                    amp.re,
                    amp.im,
                    width = self.n_qubits
                )?;
            } else {
                write!(
                    f,
                    "|{:0>width$b}⟩: {:.4}{:.4}i",
                    i,
                    amp.re,
                    amp.im,
                    width = self.n_qubits
                )?;
            }
        }
        write!(f, "]")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_qubit_zero_normalised() {
        let q = Qubit::new_zero();
        assert!(q.is_normalised(TOL));
        assert!((q.prob_zero() - 1.0).abs() < TOL);
        assert!(q.prob_one().abs() < TOL);
    }

    #[test]
    fn test_qubit_one_normalised() {
        let q = Qubit::new_one();
        assert!(q.is_normalised(TOL));
        assert!(q.prob_zero().abs() < TOL);
        assert!((q.prob_one() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_qubit_superposition() {
        let q = Qubit::new_plus();
        assert!(q.is_normalised(TOL));
        assert!((q.prob_zero() - 0.5).abs() < TOL);
        assert!((q.prob_one() - 0.5).abs() < TOL);
    }

    #[test]
    fn test_bloch_sphere_zero() {
        let q = Qubit::new_superposition(0.0, 0.0);
        // theta=0 → |0⟩
        assert!((q.prob_zero() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_bloch_sphere_one() {
        let q = Qubit::new_superposition(std::f64::consts::PI, 0.0);
        // theta=π → |1⟩
        assert!((q.prob_one() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_qubit_measure_deterministic_zero() {
        let q = Qubit::new_zero();
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let (outcome, post) = q.measure(&mut rng);
        assert_eq!(outcome, 0);
        assert!(post.is_normalised(TOL));
        assert!((post.prob_zero() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_qubit_measure_deterministic_one() {
        let q = Qubit::new_one();
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let (outcome, post) = q.measure(&mut rng);
        assert_eq!(outcome, 1);
        assert!(post.is_normalised(TOL));
    }

    #[test]
    fn test_register_zero_state() {
        let reg = QubitRegister::new_zero_state(3).expect("valid");
        assert_eq!(reg.n_qubits(), 3);
        assert_eq!(reg.dim(), 8);
        assert!((reg.probability(0).expect("ok") - 1.0).abs() < TOL);
    }

    #[test]
    fn test_register_uniform_superposition() {
        let reg = QubitRegister::new_uniform_superposition(2).expect("valid");
        let p = reg.probability(0).expect("ok");
        assert!((p - 0.25).abs() < TOL);
    }

    #[test]
    fn test_tensor_product_dims() {
        let r1 = QubitRegister::new_zero_state(2).expect("valid");
        let r2 = QubitRegister::new_zero_state(3).expect("valid");
        let combined = QubitRegister::tensor_product(&r1, &r2);
        assert_eq!(combined.n_qubits(), 5);
        assert_eq!(combined.dim(), 32);
    }

    #[test]
    fn test_measure_all_basis_state() {
        let reg = QubitRegister::new_basis_state(3, 5).expect("valid");
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let bits = reg.measure_all(&mut rng);
        // basis index 5 = 101 in binary → q0=1, q1=0, q2=1
        assert_eq!(bits, vec![1, 0, 1]);
    }

    #[test]
    fn test_fidelity_same_state() {
        let r = QubitRegister::new_zero_state(2).expect("valid");
        let f = r.fidelity(&r).expect("ok");
        assert!((f - 1.0).abs() < TOL);
    }

    #[test]
    fn test_fidelity_orthogonal() {
        let r0 = QubitRegister::new_basis_state(1, 0).expect("valid");
        let r1 = QubitRegister::new_basis_state(1, 1).expect("valid");
        let f = r0.fidelity(&r1).expect("ok");
        assert!(f.abs() < TOL);
    }
}
