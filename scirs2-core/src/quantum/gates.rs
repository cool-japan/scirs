//! Quantum gate definitions and statevector application.
//!
//! # Gate Types
//!
//! ## Single-Qubit Gates
//!
//! | Gate | Description |
//! |------|-------------|
//! | [`PauliX`] | Bit-flip (NOT) |
//! | [`PauliY`] | Y Pauli |
//! | [`PauliZ`] | Phase-flip |
//! | [`Hadamard`] | Equal superposition |
//! | [`PhaseS`] | S gate (π/2 phase) |
//! | [`PhaseT`] | T gate (π/4 phase) |
//! | [`RotX`] | Rotation around X-axis |
//! | [`RotY`] | Rotation around Y-axis |
//! | [`RotZ`] | Rotation around Z-axis |
//! | [`PhaseShift`] | Arbitrary phase shift |
//! | [`Identity`] | Identity |
//!
//! ## Two-Qubit Gates
//!
//! | Gate | Description |
//! |------|-------------|
//! | [`CNOT`] | Controlled-NOT |
//! | [`CZ`] | Controlled-Z |
//! | [`SWAP`] | SWAP |
//! | [`Toffoli`] | Toffoli (CCX, 3-qubit) |
//! | [`Fredkin`] | Fredkin (CSWAP, 3-qubit) |
//! | [`CU`] | Controlled-U (arbitrary 1-qubit gate) |
//!
//! # Applying Gates
//!
//! Use [`apply_gate`] to apply any [`QuantumGate`] to specific qubits of a
//! [`QubitRegister`]:
//!
//! ```rust
//! use scirs2_core::quantum::qubits::QubitRegister;
//! use scirs2_core::quantum::gates::{Hadamard, CNOT, apply_gate};
//!
//! let mut reg = QubitRegister::new_zero_state(2).expect("should succeed");
//! apply_gate(&mut reg, &Hadamard, &[0]).expect("should succeed");
//! apply_gate(&mut reg, &CNOT, &[0, 1]).expect("should succeed");
//! // reg is now in the Bell state (|00⟩ + |11⟩) / √2
//! ```

use ndarray::{s, Array2};
use num_complex::Complex;
use std::f64::consts::PI;

use super::error::{QuantumError, QuantumResult};
use super::qubits::QubitRegister;

// ─────────────────────────────────────────────────────────────────────────────
// QuantumGate trait
// ─────────────────────────────────────────────────────────────────────────────

/// A quantum gate represented by a unitary matrix in the computational basis.
///
/// Implementors must return the gate's 2^n × 2^n unitary matrix and declare
/// the number of qubits the gate acts on.
pub trait QuantumGate: Send + Sync {
    /// The 2^n × 2^n unitary matrix for this gate.
    fn matrix(&self) -> Array2<Complex<f64>>;

    /// Number of qubits this gate acts on.
    fn n_qubits(&self) -> usize;

    /// Human-readable name, used in circuit diagrams / debug output.
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build 2×2 matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 2×2 complex matrix from row-major entries.
fn mat2(
    a: Complex<f64>,
    b: Complex<f64>,
    c: Complex<f64>,
    d: Complex<f64>,
) -> Array2<Complex<f64>> {
    Array2::from_shape_vec((2, 2), vec![a, b, c, d]).expect("2x2 matrix construction is infallible")
}

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn cr(re: f64) -> Complex<f64> {
    Complex::new(re, 0.0)
}

fn ci(im: f64) -> Complex<f64> {
    Complex::new(0.0, im)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-qubit gates
// ─────────────────────────────────────────────────────────────────────────────

/// Identity gate I.
pub struct Identity;

impl QuantumGate for Identity {
    fn matrix(&self) -> Array2<Complex<f64>> {
        mat2(cr(1.0), cr(0.0), cr(0.0), cr(1.0))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "I"
    }
}

/// Pauli-X gate (bit-flip / NOT): |0⟩↔|1⟩.
pub struct PauliX;

impl QuantumGate for PauliX {
    fn matrix(&self) -> Array2<Complex<f64>> {
        mat2(cr(0.0), cr(1.0), cr(1.0), cr(0.0))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "X"
    }
}

/// Pauli-Y gate.
pub struct PauliY;

impl QuantumGate for PauliY {
    fn matrix(&self) -> Array2<Complex<f64>> {
        mat2(cr(0.0), ci(-1.0), ci(1.0), cr(0.0))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Y"
    }
}

/// Pauli-Z gate (phase-flip): |1⟩ → −|1⟩.
pub struct PauliZ;

impl QuantumGate for PauliZ {
    fn matrix(&self) -> Array2<Complex<f64>> {
        mat2(cr(1.0), cr(0.0), cr(0.0), cr(-1.0))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Z"
    }
}

/// Hadamard gate H: (|0⟩+|1⟩)/√2 ← |0⟩, (|0⟩−|1⟩)/√2 ← |1⟩.
pub struct Hadamard;

impl QuantumGate for Hadamard {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let s = 1.0 / 2.0_f64.sqrt();
        mat2(cr(s), cr(s), cr(s), cr(-s))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "H"
    }
}

/// S gate (phase gate): |1⟩ → i|1⟩ (π/2 phase shift).
pub struct PhaseS;

impl QuantumGate for PhaseS {
    fn matrix(&self) -> Array2<Complex<f64>> {
        mat2(cr(1.0), cr(0.0), cr(0.0), ci(1.0))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "S"
    }
}

/// S† (S-dagger / inverse S) gate: |1⟩ → −i|1⟩.
pub struct PhaseSdg;

impl QuantumGate for PhaseSdg {
    fn matrix(&self) -> Array2<Complex<f64>> {
        mat2(cr(1.0), cr(0.0), cr(0.0), ci(-1.0))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Sdg"
    }
}

/// T gate: |1⟩ → e^{iπ/4}|1⟩ (π/4 phase shift).
pub struct PhaseT;

impl QuantumGate for PhaseT {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let phase = Complex::from_polar(1.0, PI / 4.0);
        mat2(cr(1.0), cr(0.0), cr(0.0), phase)
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "T"
    }
}

/// T† (T-dagger / inverse T) gate: |1⟩ → e^{−iπ/4}|1⟩.
pub struct PhaseTdg;

impl QuantumGate for PhaseTdg {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let phase = Complex::from_polar(1.0, -PI / 4.0);
        mat2(cr(1.0), cr(0.0), cr(0.0), phase)
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Tdg"
    }
}

/// Rx(θ): rotation by angle `theta` around the X-axis.
///
/// Rx(θ) = cos(θ/2)I − i sin(θ/2)X
pub struct RotX {
    /// Rotation angle in radians.
    pub theta: f64,
}

impl QuantumGate for RotX {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let (s, co) = (self.theta / 2.0).sin_cos();
        mat2(cr(co), ci(-s), ci(-s), cr(co))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Rx"
    }
}

/// Ry(θ): rotation by angle `theta` around the Y-axis.
///
/// Ry(θ) = cos(θ/2)I − i sin(θ/2)Y
pub struct RotY {
    /// Rotation angle in radians.
    pub theta: f64,
}

impl QuantumGate for RotY {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let (s, co) = (self.theta / 2.0).sin_cos();
        mat2(cr(co), cr(-s), cr(s), cr(co))
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Ry"
    }
}

/// Rz(θ): rotation by angle `theta` around the Z-axis.
///
/// Rz(θ) = e^{−iθ/2}|0⟩⟨0| + e^{iθ/2}|1⟩⟨1|
pub struct RotZ {
    /// Rotation angle in radians.
    pub theta: f64,
}

impl QuantumGate for RotZ {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let neg = Complex::from_polar(1.0, -self.theta / 2.0);
        let pos = Complex::from_polar(1.0, self.theta / 2.0);
        mat2(neg, cr(0.0), cr(0.0), pos)
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "Rz"
    }
}

/// Arbitrary phase-shift gate P(λ): |1⟩ → e^{iλ}|1⟩.
pub struct PhaseShift {
    /// Phase angle λ in radians.
    pub lambda: f64,
}

impl QuantumGate for PhaseShift {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let phase = Complex::from_polar(1.0, self.lambda);
        mat2(cr(1.0), cr(0.0), cr(0.0), phase)
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "P"
    }
}

/// General single-qubit unitary U(θ, φ, λ) — IBM convention:
/// U = [[cos(θ/2), −e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
pub struct Unitary1Q {
    /// Polar rotation angle.
    pub theta: f64,
    /// Azimuthal phase φ.
    pub phi: f64,
    /// Phase λ.
    pub lambda: f64,
}

impl QuantumGate for Unitary1Q {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let (s, co) = (self.theta / 2.0).sin_cos();
        let eiphi = Complex::from_polar(1.0, self.phi);
        let eilambda = Complex::from_polar(1.0, self.lambda);
        let eiphilambda = Complex::from_polar(1.0, self.phi + self.lambda);
        mat2(cr(co), -eilambda * s, eiphi * s, eiphilambda * co)
    }
    fn n_qubits(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "U"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-qubit gates
// ─────────────────────────────────────────────────────────────────────────────

/// CNOT (CX) gate: flips target qubit when control qubit is |1⟩.
///
/// Matrix in the basis |00⟩, |01⟩, |10⟩, |11⟩ (control=0, target=1):
/// ```text
/// 1 0 0 0
/// 0 1 0 0
/// 0 0 0 1
/// 0 0 1 0
/// ```
pub struct CNOT;

impl QuantumGate for CNOT {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let o = cr(0.0);
        let i = cr(1.0);
        Array2::from_shape_vec((4, 4), vec![i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o])
            .expect("4x4 matrix construction is infallible")
    }
    fn n_qubits(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "CNOT"
    }
}

/// CZ (Controlled-Z) gate: applies Z to target when control is |1⟩.
pub struct CZ;

impl QuantumGate for CZ {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let o = cr(0.0);
        let i = cr(1.0);
        let m = cr(-1.0);
        Array2::from_shape_vec((4, 4), vec![i, o, o, o, o, i, o, o, o, o, i, o, o, o, o, m])
            .expect("4x4 matrix construction is infallible")
    }
    fn n_qubits(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "CZ"
    }
}

/// SWAP gate: swaps two qubits.
pub struct SWAP;

impl QuantumGate for SWAP {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let o = cr(0.0);
        let i = cr(1.0);
        Array2::from_shape_vec((4, 4), vec![i, o, o, o, o, o, i, o, o, i, o, o, o, o, o, i])
            .expect("4x4 matrix construction is infallible")
    }
    fn n_qubits(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "SWAP"
    }
}

/// iSWAP gate: SWAP with additional i phase on swapped states.
pub struct ISWAP;

impl QuantumGate for ISWAP {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let o = cr(0.0);
        let i_re = cr(1.0);
        let i_im = ci(1.0);
        Array2::from_shape_vec(
            (4, 4),
            vec![i_re, o, o, o, o, o, i_im, o, o, i_im, o, o, o, o, o, i_re],
        )
        .expect("4x4 matrix construction is infallible")
    }
    fn n_qubits(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "iSWAP"
    }
}

/// Controlled-U gate: applies an arbitrary 1-qubit gate `u` to the target
/// when the control qubit is |1⟩.
pub struct CU {
    inner: Box<dyn QuantumGate>,
}

impl CU {
    /// Construct a CU gate wrapping any single-qubit gate.
    pub fn new(gate: impl QuantumGate + 'static) -> QuantumResult<Self> {
        if gate.n_qubits() != 1 {
            return Err(QuantumError::GateArityMismatch {
                gate_qubits: gate.n_qubits(),
                supplied: 1,
            });
        }
        Ok(Self {
            inner: Box::new(gate),
        })
    }
}

impl QuantumGate for CU {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let u = self.inner.matrix();
        let o = cr(0.0);
        let i = cr(1.0);
        let u00 = u[[0, 0]];
        let u01 = u[[0, 1]];
        let u10 = u[[1, 0]];
        let u11 = u[[1, 1]];
        Array2::from_shape_vec(
            (4, 4),
            vec![i, o, o, o, o, i, o, o, o, o, u00, u01, o, o, u10, u11],
        )
        .expect("4x4 matrix construction is infallible")
    }
    fn n_qubits(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "CU"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Three-qubit gates
// ─────────────────────────────────────────────────────────────────────────────

/// Toffoli (CCX) gate: flips target when *both* control qubits are |1⟩.
///
/// Targets in the gate: `[control0, control1, target]`.
pub struct Toffoli;

impl QuantumGate for Toffoli {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let o = cr(0.0);
        let i = cr(1.0);
        // 8×8 matrix in basis |000⟩...|111⟩
        // Only rows/cols 6 and 7 are swapped (control0=1, control1=1 → flip target)
        let mut m = Array2::<Complex<f64>>::from_elem((8, 8), o);
        for k in 0..6usize {
            m[[k, k]] = i;
        }
        m[[6, 7]] = i;
        m[[7, 6]] = i;
        m
    }
    fn n_qubits(&self) -> usize {
        3
    }
    fn name(&self) -> &str {
        "Toffoli"
    }
}

/// Fredkin (CSWAP) gate: swaps target qubits when control qubit is |1⟩.
///
/// Targets in the gate: `[control, target0, target1]`.
pub struct Fredkin;

impl QuantumGate for Fredkin {
    fn matrix(&self) -> Array2<Complex<f64>> {
        let o = cr(0.0);
        let i = cr(1.0);
        // 8×8 in basis |000⟩…|111⟩
        // Rows/cols 5 and 6 are swapped (control=1 → swap the two targets)
        let mut m = Array2::<Complex<f64>>::from_elem((8, 8), o);
        for k in 0..8usize {
            m[[k, k]] = i;
        }
        m[[5, 5]] = o;
        m[[6, 6]] = o;
        m[[5, 6]] = i;
        m[[6, 5]] = i;
        m
    }
    fn n_qubits(&self) -> usize {
        3
    }
    fn name(&self) -> &str {
        "Fredkin"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate application
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `gate` to the qubits at positions `target_qubits` in `state`.
///
/// The gate's arity must equal `target_qubits.len()`.  All qubit indices must
/// be distinct and within range.
///
/// The statevector is updated in place.
pub fn apply_gate(
    state: &mut QubitRegister,
    gate: &dyn QuantumGate,
    target_qubits: &[usize],
) -> QuantumResult<()> {
    let gate_qubits = gate.n_qubits();
    if target_qubits.len() != gate_qubits {
        return Err(QuantumError::GateArityMismatch {
            gate_qubits,
            supplied: target_qubits.len(),
        });
    }

    // Range check.
    for &q in target_qubits {
        if q >= state.n_qubits() {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: q,
                n_qubits: state.n_qubits(),
            });
        }
    }

    // Duplicate check.
    for i in 0..target_qubits.len() {
        for j in (i + 1)..target_qubits.len() {
            if target_qubits[i] == target_qubits[j] {
                return Err(QuantumError::DuplicateQubitIndex {
                    index: target_qubits[i],
                });
            }
        }
    }

    let gate_mat = gate.matrix();
    let gate_dim = 1usize << gate_qubits;
    let total_qubits = state.n_qubits();
    let total_dim = state.dim();

    // For each group of 2^(gate_qubits) basis states that differ only in the
    // target qubit positions, compute the matrix-vector product.
    let mut new_amps = state.amplitudes.clone();

    // Iterate over all combinations of the non-target qubit values.
    let non_target_dim = total_dim / gate_dim;

    for outer in 0..non_target_dim {
        // Build the full amplitude vector for this sub-space slice.
        let mut sub_amps = vec![Complex::new(0.0, 0.0); gate_dim];

        // Map gate-space index → full-state index.
        let indices: Vec<usize> = (0..gate_dim)
            .map(|g| gate_idx_to_full_idx(g, outer, target_qubits, total_qubits))
            .collect();

        for (g, &full_idx) in indices.iter().enumerate() {
            sub_amps[g] = state.amplitudes[full_idx];
        }

        // Multiply by the gate matrix.
        let mut result = vec![Complex::new(0.0, 0.0); gate_dim];
        for row in 0..gate_dim {
            for col in 0..gate_dim {
                result[row] += gate_mat[[row, col]] * sub_amps[col];
            }
        }

        // Write back.
        for (g, &full_idx) in indices.iter().enumerate() {
            new_amps[full_idx] = result[g];
        }
    }

    state.amplitudes = new_amps;
    Ok(())
}

/// Convert a gate-space index and an "outer" index into the full statevector
/// index, placing the gate bits at the positions indicated by `target_qubits`.
///
/// `total_qubits` is the total number of qubits in the register.
fn gate_idx_to_full_idx(
    gate_idx: usize,
    outer: usize,
    target_qubits: &[usize],
    total_qubits: usize,
) -> usize {
    // We need to interleave the `gate_qubits` gate-index bits into the
    // `total_qubits`-bit full index, at the positions given by target_qubits.
    //
    // Algorithm:
    //   1. Start with the `outer` non-target bits spread across the non-target
    //      positions.
    //   2. Insert the gate-index bits at the target positions.

    let gate_qubits = target_qubits.len();
    let mut full = 0usize;

    // Collect target positions as a sorted set for easy "is this bit a target?"
    // lookups.
    let mut target_set = [usize::MAX; 64];
    for (i, &t) in target_qubits.iter().enumerate() {
        target_set[i] = t;
    }

    // outer_bit_pos iterates through the non-target qubit positions (0..total_qubits
    // minus targets) from LSB to MSB.
    let mut outer_idx = 0usize;

    for bit_pos in 0..total_qubits {
        // Is this bit position a target?
        let mut target_local = usize::MAX;
        for i in 0..gate_qubits {
            if target_set[i] == bit_pos {
                target_local = i;
                break;
            }
        }
        if target_local != usize::MAX {
            // Extract the corresponding gate-index bit.
            // The gate matrix is indexed in big-endian order: the first qubit in
            // target_qubits corresponds to the MSB of gate_idx.
            let gate_bit = (gate_idx >> (gate_qubits - 1 - target_local)) & 1;
            full |= gate_bit << bit_pos;
        } else {
            // Extract the corresponding outer bit.
            let outer_bit = (outer >> outer_idx) & 1;
            full |= outer_bit << bit_pos;
            outer_idx += 1;
        }
    }

    full
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate matrix utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the tensor product of two gate matrices: U₁ ⊗ U₂.
///
/// Produces a (2^(n₁+n₂)) × (2^(n₁+n₂)) matrix.
pub fn tensor_product_matrices(
    u1: &Array2<Complex<f64>>,
    u2: &Array2<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let (r1, c1) = (u1.nrows(), u1.ncols());
    let (r2, c2) = (u2.nrows(), u2.ncols());
    let rows = r1 * r2;
    let cols = c1 * c2;
    let mut result = Array2::zeros((rows, cols));
    for i in 0..r1 {
        for j in 0..c1 {
            for k in 0..r2 {
                for l in 0..c2 {
                    result[[i * r2 + k, j * c2 + l]] = u1[[i, j]] * u2[[k, l]];
                }
            }
        }
    }
    result
}

/// Verify that a matrix U is unitary: check that U†U ≈ I within `tol`.
///
/// Returns `Ok(())` if unitary, or `Err(QuantumError::NonUnitaryGate)` otherwise.
pub fn check_unitary(u: &Array2<Complex<f64>>, tol: f64) -> QuantumResult<()> {
    let n = u.nrows();
    if u.ncols() != n {
        return Err(QuantumError::DimensionMismatch {
            expected: n,
            actual: u.ncols(),
        });
    }
    let mut max_dev: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            // (U†U)_{ij} = Σ_k conj(U_{ki}) * U_{kj}
            let val: Complex<f64> = (0..n).map(|k| u[[k, i]].conj() * u[[k, j]]).sum();
            let expected = if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            let dev = (val - expected).norm();
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    if max_dev > tol {
        return Err(QuantumError::NonUnitaryGate { deviation: max_dev });
    }
    Ok(())
}

/// Compute the matrix product of two square matrices of the same dimension.
pub fn matrix_product(
    a: &Array2<Complex<f64>>,
    b: &Array2<Complex<f64>>,
) -> QuantumResult<Array2<Complex<f64>>> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n || b.ncols() != n {
        return Err(QuantumError::DimensionMismatch {
            expected: n,
            actual: b.nrows(),
        });
    }
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let val: Complex<f64> = (0..n).map(|k| a[[i, k]] * b[[k, j]]).sum();
            result[[i, j]] = val;
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::qubits::QubitRegister;
    use super::*;

    const TOL: f64 = 1e-12;

    fn assert_complex_close(a: Complex<f64>, b: Complex<f64>, tol: f64, msg: &str) {
        assert!(
            (a - b).norm() < tol,
            "{}: expected {:?}, got {:?}",
            msg,
            b,
            a
        );
    }

    #[test]
    fn test_pauli_x_unitary() {
        check_unitary(&PauliX.matrix(), 1e-12).expect("PauliX should be unitary");
    }

    #[test]
    fn test_hadamard_unitary() {
        check_unitary(&Hadamard.matrix(), 1e-12).expect("H should be unitary");
    }

    #[test]
    fn test_cnot_unitary() {
        check_unitary(&CNOT.matrix(), 1e-12).expect("CNOT should be unitary");
    }

    #[test]
    fn test_toffoli_unitary() {
        check_unitary(&Toffoli.matrix(), 1e-12).expect("Toffoli should be unitary");
    }

    #[test]
    fn test_x_flips_zero() {
        let mut reg = QubitRegister::new_zero_state(1).expect("valid");
        apply_gate(&mut reg, &PauliX, &[0]).expect("apply ok");
        assert!((reg.probability(1).expect("ok") - 1.0).abs() < TOL);
    }

    #[test]
    fn test_x_flips_one() {
        let mut reg = QubitRegister::new_basis_state(1, 1).expect("valid");
        apply_gate(&mut reg, &PauliX, &[0]).expect("apply ok");
        assert!((reg.probability(0).expect("ok") - 1.0).abs() < TOL);
    }

    #[test]
    fn test_hadamard_superposition() {
        let mut reg = QubitRegister::new_zero_state(1).expect("valid");
        apply_gate(&mut reg, &Hadamard, &[0]).expect("apply ok");
        let p0 = reg.probability(0).expect("ok");
        let p1 = reg.probability(1).expect("ok");
        assert!((p0 - 0.5).abs() < TOL);
        assert!((p1 - 0.5).abs() < TOL);
    }

    #[test]
    fn test_cnot_creates_bell_state() {
        let mut reg = QubitRegister::new_zero_state(2).expect("valid");
        apply_gate(&mut reg, &Hadamard, &[0]).expect("H ok");
        apply_gate(&mut reg, &CNOT, &[0, 1]).expect("CNOT ok");
        // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let p00 = reg.probability(0).expect("ok");
        let p11 = reg.probability(3).expect("ok");
        let p01 = reg.probability(1).expect("ok");
        let p10 = reg.probability(2).expect("ok");
        assert!((p00 - 0.5).abs() < TOL, "p00={}", p00);
        assert!((p11 - 0.5).abs() < TOL, "p11={}", p11);
        assert!(p01.abs() < TOL, "p01={}", p01);
        assert!(p10.abs() < TOL, "p10={}", p10);
    }

    #[test]
    fn test_z_phase_flip() {
        // Apply Z to |+⟩ = (|0⟩+|1⟩)/√2 → |−⟩ = (|0⟩−|1⟩)/√2
        let mut reg = QubitRegister::new_zero_state(1).expect("valid");
        apply_gate(&mut reg, &Hadamard, &[0]).expect("H ok");
        apply_gate(&mut reg, &PauliZ, &[0]).expect("Z ok");
        // Amplitude for |1⟩ should now be negative.
        let amp1 = reg.amplitude(1).expect("ok");
        assert!(amp1.re < 0.0);
    }

    #[test]
    fn test_duplicate_qubit_error() {
        let mut reg = QubitRegister::new_zero_state(2).expect("valid");
        let err = apply_gate(&mut reg, &CNOT, &[0, 0]);
        assert!(matches!(err, Err(QuantumError::DuplicateQubitIndex { .. })));
    }

    #[test]
    fn test_arity_error() {
        let mut reg = QubitRegister::new_zero_state(2).expect("valid");
        let err = apply_gate(&mut reg, &PauliX, &[0, 1]);
        assert!(matches!(err, Err(QuantumError::GateArityMismatch { .. })));
    }

    #[test]
    fn test_swap_swaps_qubits() {
        // |10⟩ → |01⟩ after SWAP
        // index 2 = |10⟩ (qubit0=0, qubit1=1)
        let mut reg = QubitRegister::new_basis_state(2, 2).expect("valid");
        apply_gate(&mut reg, &SWAP, &[0, 1]).expect("SWAP ok");
        // After swap: index 1 = |01⟩ (qubit0=1, qubit1=0)
        let p1 = reg.probability(1).expect("ok");
        assert!(
            (p1 - 1.0).abs() < TOL,
            "SWAP should move |10⟩ to |01⟩, got p1={}",
            p1
        );
    }

    #[test]
    fn test_rot_x_pi_equals_x() {
        let rx_pi = RotX { theta: PI };
        let mx = rx_pi.matrix();
        // Up to global phase, Rx(π) = −iX
        // |Rx(π)[0,1]| = 1.0
        assert!((mx[[0, 1]].norm() - 1.0).abs() < 1e-10);
        assert!((mx[[1, 0]].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli_flips_when_both_controls_set() {
        // |110⟩ → |111⟩
        // Toffoli(control0, control1, target) with target_qubits = [0, 1, 2]:
        //   control0 = qubit 0, control1 = qubit 1, target = qubit 2
        // Start with index 3 = |011⟩: qubit0=1, qubit1=1, qubit2=0 (both controls set).
        // Toffoli flips qubit2: index 3 -> index 7 = |111⟩
        let mut reg = QubitRegister::new_basis_state(3, 3).expect("valid");
        apply_gate(&mut reg, &Toffoli, &[0, 1, 2]).expect("ok");
        let p7 = reg.probability(7).expect("ok");
        assert!((p7 - 1.0).abs() < TOL, "p7={}", p7);
    }
}
