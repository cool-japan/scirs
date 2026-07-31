//! Quantum circuit construction, execution, and analysis.
//!
//! # Overview
//!
//! A [`QuantumCircuit`] is an ordered sequence of gate operations applied to
//! a fixed-width quantum register.  Each operation records the gate and the
//! qubit indices it targets.
//!
//! ```rust
//! use scirs2_core::quantum::circuit::QuantumCircuit;
//! use scirs2_core::quantum::qubits::QubitRegister;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha20Rng;
//!
//! // Build and run a Bell-pair circuit.
//! let mut circ = QuantumCircuit::new(2);
//! circ.h(0).expect("should succeed");
//! circ.cx(0, 1).expect("should succeed");
//!
//! let initial = QubitRegister::new_zero_state(2).expect("should succeed");
//! let final_state = circ.run(initial).expect("should succeed");
//!
//! // Measure several times (seeded for reproducibility).
//! let mut rng = ChaCha20Rng::seed_from_u64(42);
//! let bits = circ.measure_all(&final_state, &mut rng).expect("should succeed");
//! assert!(bits == vec![0, 0] || bits == vec![1, 1]);
//! ```

use rand::{Rng, RngExt};

use super::error::{QuantumError, QuantumResult};
use super::gates::{
    apply_gate, Fredkin, Hadamard, Identity, PauliX, PauliY, PauliZ, PhaseS, PhaseSdg, PhaseShift,
    PhaseT, PhaseTdg, QuantumGate, RotX, RotY, RotZ, Toffoli, Unitary1Q, CNOT, CU, CZ, SWAP,
};
use super::qubits::QubitRegister;

// ─────────────────────────────────────────────────────────────────────────────
// GateOp — a single gate applied to specific qubits
// ─────────────────────────────────────────────────────────────────────────────

/// A single gate application: the gate and the target qubit indices.
pub struct GateOp {
    /// Boxed gate (heap-allocated so we can store heterogeneous gate types).
    gate: Box<dyn QuantumGate>,
    /// Target qubit indices (length == gate.n_qubits()).
    qubits: Vec<usize>,
}

impl GateOp {
    fn new(gate: impl QuantumGate + 'static, qubits: Vec<usize>) -> Self {
        Self {
            gate: Box::new(gate),
            qubits,
        }
    }
}

impl std::fmt::Debug for GateOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GateOp({}, qubits={:?})", self.gate.name(), self.qubits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantumCircuit
// ─────────────────────────────────────────────────────────────────────────────

/// A quantum circuit: an ordered list of gate operations acting on an n-qubit
/// register.
///
/// Gates are appended via convenience methods (`h`, `cx`, `rx`, …) or via the
/// generic [`QuantumCircuit::add_gate`] method.  The circuit is executed by
/// calling [`QuantumCircuit::run`] which returns the final statevector.
pub struct QuantumCircuit {
    /// Number of qubits the circuit acts on.
    n_qubits: usize,
    /// Ordered gate operations.
    ops: Vec<GateOp>,
}

impl QuantumCircuit {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create an empty circuit acting on `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            ops: Vec::new(),
        }
    }

    // ── Properties ───────────────────────────────────────────────────────────

    /// Number of qubits in this circuit.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of gate operations in the circuit.
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }

    /// Circuit depth: the number of *sequential* layers when gates on disjoint
    /// qubits are parallelised.
    ///
    /// Uses a greedy layer-assignment algorithm: each gate is placed in the
    /// earliest layer that has no qubit overlap with it.
    pub fn circuit_depth(&self) -> usize {
        // layer_finish[q] = the layer in which qubit q was last used.
        let mut layer_finish = vec![0usize; self.n_qubits];
        let mut depth = 0usize;

        for op in &self.ops {
            // The gate must be placed after all layers that touch its qubits.
            let earliest = op
                .qubits
                .iter()
                .map(|&q| layer_finish[q])
                .max()
                .unwrap_or(0);
            let layer = earliest + 1;
            for &q in &op.qubits {
                layer_finish[q] = layer;
            }
            if layer > depth {
                depth = layer;
            }
        }
        depth
    }

    // ── Gate append ──────────────────────────────────────────────────────────

    /// Append an arbitrary gate acting on `qubits`.
    ///
    /// Validates that the qubit indices are in range before storing the gate.
    pub fn add_gate(
        &mut self,
        gate: impl QuantumGate + 'static,
        qubits: &[usize],
    ) -> QuantumResult<()> {
        if qubits.len() != gate.n_qubits() {
            return Err(QuantumError::GateArityMismatch {
                gate_qubits: gate.n_qubits(),
                supplied: qubits.len(),
            });
        }
        for &q in qubits {
            if q >= self.n_qubits {
                return Err(QuantumError::QubitIndexOutOfRange {
                    index: q,
                    n_qubits: self.n_qubits,
                });
            }
        }
        // Duplicate check.
        for i in 0..qubits.len() {
            for j in (i + 1)..qubits.len() {
                if qubits[i] == qubits[j] {
                    return Err(QuantumError::DuplicateQubitIndex { index: qubits[i] });
                }
            }
        }
        self.ops.push(GateOp::new(gate, qubits.to_vec()));
        Ok(())
    }

    // ── Execution ─────────────────────────────────────────────────────────────

    /// Execute the circuit on `initial_state` and return the final statevector.
    ///
    /// The register must have exactly `self.n_qubits` qubits.
    pub fn run(&self, initial_state: QubitRegister) -> QuantumResult<QubitRegister> {
        if initial_state.n_qubits() != self.n_qubits {
            return Err(QuantumError::CircuitRegisterMismatch {
                circuit_qubits: self.n_qubits,
                register_qubits: initial_state.n_qubits(),
            });
        }
        let mut state = initial_state;
        for op in &self.ops {
            apply_gate(&mut state, op.gate.as_ref(), &op.qubits)?;
        }
        Ok(state)
    }

    /// Measure all qubits of `state` once, returning a bit-string (qubit 0 first).
    ///
    /// Does *not* collapse the state; call this method on the result of [`Self::run`].
    pub fn measure_all<R: Rng>(
        &self,
        state: &QubitRegister,
        rng: &mut R,
    ) -> QuantumResult<Vec<u8>> {
        if state.n_qubits() != self.n_qubits {
            return Err(QuantumError::CircuitRegisterMismatch {
                circuit_qubits: self.n_qubits,
                register_qubits: state.n_qubits(),
            });
        }
        Ok(state.measure_all(rng))
    }

    /// Run the circuit `shots` times from `initial_state` and collect all
    /// measurement outcomes.
    ///
    /// Each shot independently executes the circuit and performs a single
    /// full-register measurement.
    pub fn sample<R: Rng>(
        &self,
        initial_state: &QubitRegister,
        shots: usize,
        rng: &mut R,
    ) -> QuantumResult<Vec<Vec<u8>>> {
        if initial_state.n_qubits() != self.n_qubits {
            return Err(QuantumError::CircuitRegisterMismatch {
                circuit_qubits: self.n_qubits,
                register_qubits: initial_state.n_qubits(),
            });
        }
        let final_state = self.run(initial_state.clone())?;
        let results = (0..shots).map(|_| final_state.measure_all(rng)).collect();
        Ok(results)
    }

    // ── Convenience single-qubit gates ────────────────────────────────────────

    /// Append Identity on qubit `q`.
    pub fn id(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(Identity, &[q])
    }

    /// Append Pauli-X (NOT) on qubit `q`.
    pub fn x(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PauliX, &[q])
    }

    /// Append Pauli-Y on qubit `q`.
    pub fn y(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PauliY, &[q])
    }

    /// Append Pauli-Z on qubit `q`.
    pub fn z(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PauliZ, &[q])
    }

    /// Append Hadamard on qubit `q`.
    pub fn h(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(Hadamard, &[q])
    }

    /// Append S gate on qubit `q`.
    pub fn s(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PhaseS, &[q])
    }

    /// Append S† gate on qubit `q`.
    pub fn sdg(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PhaseSdg, &[q])
    }

    /// Append T gate on qubit `q`.
    pub fn t(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PhaseT, &[q])
    }

    /// Append T† gate on qubit `q`.
    pub fn tdg(&mut self, q: usize) -> QuantumResult<()> {
        self.add_gate(PhaseTdg, &[q])
    }

    /// Append Rx(θ) on qubit `q`.
    pub fn rx(&mut self, theta: f64, q: usize) -> QuantumResult<()> {
        self.add_gate(RotX { theta }, &[q])
    }

    /// Append Ry(θ) on qubit `q`.
    pub fn ry(&mut self, theta: f64, q: usize) -> QuantumResult<()> {
        self.add_gate(RotY { theta }, &[q])
    }

    /// Append Rz(θ) on qubit `q`.
    pub fn rz(&mut self, theta: f64, q: usize) -> QuantumResult<()> {
        self.add_gate(RotZ { theta }, &[q])
    }

    /// Append P(λ) phase-shift on qubit `q`.
    pub fn p(&mut self, lambda: f64, q: usize) -> QuantumResult<()> {
        self.add_gate(PhaseShift { lambda }, &[q])
    }

    /// Append U(θ, φ, λ) on qubit `q`.
    pub fn u(&mut self, theta: f64, phi: f64, lambda: f64, q: usize) -> QuantumResult<()> {
        self.add_gate(Unitary1Q { theta, phi, lambda }, &[q])
    }

    // ── Convenience two-qubit gates ───────────────────────────────────────────

    /// Append CNOT with `control` controlling `target`.
    pub fn cx(&mut self, control: usize, target: usize) -> QuantumResult<()> {
        self.add_gate(CNOT, &[control, target])
    }

    /// Append CZ with `control` and `target`.
    pub fn cz(&mut self, control: usize, target: usize) -> QuantumResult<()> {
        self.add_gate(CZ, &[control, target])
    }

    /// Append SWAP of `q0` and `q1`.
    pub fn swap(&mut self, q0: usize, q1: usize) -> QuantumResult<()> {
        self.add_gate(SWAP, &[q0, q1])
    }

    /// Append Controlled-U where `control` triggers `gate` on `target`.
    pub fn cu(
        &mut self,
        gate: impl QuantumGate + 'static,
        control: usize,
        target: usize,
    ) -> QuantumResult<()> {
        let cu_gate = CU::new(gate)?;
        self.add_gate(cu_gate, &[control, target])
    }

    // ── Convenience three-qubit gates ─────────────────────────────────────────

    /// Append Toffoli (CCX) with two controls and one target.
    pub fn ccx(&mut self, c0: usize, c1: usize, target: usize) -> QuantumResult<()> {
        self.add_gate(Toffoli, &[c0, c1, target])
    }

    /// Append Fredkin (CSWAP) with one control and two targets.
    pub fn cswap(&mut self, control: usize, t0: usize, t1: usize) -> QuantumResult<()> {
        self.add_gate(Fredkin, &[control, t0, t1])
    }

    // ── Barrier (no-op annotation) ────────────────────────────────────────────

    /// Add a barrier (Identity gates) across all specified qubits.
    ///
    /// Barriers have no physical effect; they exist to delimit logical sections
    /// of a circuit and prevent gate optimisers from merging across the boundary.
    pub fn barrier(&mut self, qubits: &[usize]) -> QuantumResult<()> {
        for &q in qubits {
            self.add_gate(Identity, &[q])?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for QuantumCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QuantumCircuit({} qubits, {} ops)",
            self.n_qubits,
            self.ops.len()
        )?;
        for (i, op) in self.ops.iter().enumerate() {
            write!(f, "\n  [{i}] {} on {:?}", op.gate.name(), op.qubits)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in circuit factories
// ─────────────────────────────────────────────────────────────────────────────

/// Build the standard 2-qubit Bell pair circuit: H⊗I then CNOT(0→1).
///
/// When run on |00⟩ this produces the maximally-entangled state
/// |Φ+⟩ = (|00⟩ + |11⟩) / √2.
pub fn bell_pair_circuit() -> QuantumCircuit {
    let mut c = QuantumCircuit::new(2);
    c.h(0)
        .expect("h gate on qubit 0 is always valid for 2-qubit circuit");
    c.cx(0, 1)
        .expect("cx gate on qubits 0,1 is always valid for 2-qubit circuit");
    c
}

/// Build a GHZ-state circuit for `n_qubits` qubits.
///
/// Produces |GHZ⟩ = (|0…0⟩ + |1…1⟩) / √2 from |0…0⟩.
///
/// `n_qubits` must be ≥ 2.
pub fn ghz_circuit(n_qubits: usize) -> QuantumResult<QuantumCircuit> {
    if n_qubits < 2 {
        return Err(QuantumError::InvalidQubitCount(n_qubits));
    }
    let mut c = QuantumCircuit::new(n_qubits);
    c.h(0)?;
    for q in 1..n_qubits {
        c.cx(0, q)?;
    }
    Ok(c)
}

/// Build the n-qubit Quantum Fourier Transform (QFT) circuit.
///
/// Implements the standard decomposition:
/// ```text
/// QFT|j⟩ = (1/√N) Σ_{k=0}^{N-1} e^{2πijk/N} |k⟩
/// ```
/// using H gates followed by controlled phase rotations.
///
/// The circuit **does not** include the final SWAP reversal pass — if you need
/// the bit-reversed output convention used by most QFT literature, call
/// [`qft_circuit_with_swap`] instead.
pub fn qft_circuit(n_qubits: usize) -> QuantumResult<QuantumCircuit> {
    if n_qubits == 0 {
        return Err(QuantumError::InvalidQubitCount(n_qubits));
    }
    let mut c = QuantumCircuit::new(n_qubits);

    for target in 0..n_qubits {
        // Hadamard on target qubit.
        c.h(target)?;
        // Controlled phase rotations from all subsequent qubits.
        for control in (target + 1)..n_qubits {
            let k = (control - target + 1) as f64;
            let lambda = std::f64::consts::PI / (2.0_f64.powi((k - 1.0) as i32));
            c.cu(PhaseShift { lambda }, control, target)?;
        }
    }
    Ok(c)
}

/// Build the n-qubit QFT circuit with the output-reversal SWAP pass included.
///
/// This gives the conventional QFT output ordering where qubit 0 holds the
/// most-significant frequency component.
pub fn qft_circuit_with_swap(n_qubits: usize) -> QuantumResult<QuantumCircuit> {
    if n_qubits == 0 {
        return Err(QuantumError::InvalidQubitCount(n_qubits));
    }
    let mut c = qft_circuit(n_qubits)?;
    // Reverse qubit order with SWAP gates.
    for i in 0..(n_qubits / 2) {
        c.swap(i, n_qubits - 1 - i)?;
    }
    Ok(c)
}

/// Build the n-qubit inverse QFT circuit.
pub fn iqft_circuit(n_qubits: usize) -> QuantumResult<QuantumCircuit> {
    if n_qubits == 0 {
        return Err(QuantumError::InvalidQubitCount(n_qubits));
    }
    let mut c = QuantumCircuit::new(n_qubits);

    // IQFT is the QFT with conjugated phases applied in reverse order.
    for target in (0..n_qubits).rev() {
        // Controlled phase rotations (reversed, negated phase).
        for control in (target + 1..n_qubits).rev() {
            let k = (control - target + 1) as f64;
            let lambda = -std::f64::consts::PI / (2.0_f64.powi((k - 1.0) as i32));
            c.cu(PhaseShift { lambda }, control, target)?;
        }
        // Hadamard on target qubit.
        c.h(target)?;
    }
    Ok(c)
}

/// Build the quantum phase-estimation (QPE) circuit skeleton.
///
/// This creates `n_counting` counting qubits and `n_target` target qubits.
/// The returned circuit prepares the counting qubits in superposition and
/// leaves the eigenstate preparation as the caller's responsibility
/// (apply gates to the target register before calling this or extend the
/// circuit afterwards).
///
/// The last step is the IQFT on the counting register.
pub fn phase_estimation_circuit(
    n_counting: usize,
    n_target: usize,
) -> QuantumResult<QuantumCircuit> {
    if n_counting == 0 || n_target == 0 {
        return Err(QuantumError::InvalidQubitCount(n_counting + n_target));
    }
    let n_total = n_counting + n_target;
    let mut c = QuantumCircuit::new(n_total);

    // Hadamard on all counting qubits.
    for q in 0..n_counting {
        c.h(q)?;
    }
    // IQFT on counting register.
    let iqft = iqft_circuit(n_counting)?;
    for op in &iqft.ops {
        c.add_gate_raw(op.gate.as_ref(), &op.qubits)?;
    }
    Ok(c)
}

impl QuantumCircuit {
    /// Internal helper: append a gate operation by cloning the gate matrix into a
    /// `MatrixGate` wrapper.  Used to import sub-circuits.
    fn add_gate_raw(&mut self, gate: &dyn QuantumGate, qubits: &[usize]) -> QuantumResult<()> {
        // Bounds-check qubits (relative to this circuit's qubit count).
        for &q in qubits {
            if q >= self.n_qubits {
                return Err(QuantumError::QubitIndexOutOfRange {
                    index: q,
                    n_qubits: self.n_qubits,
                });
            }
        }
        let mat_gate = MatrixGate {
            matrix: gate.matrix(),
            n_qubits: gate.n_qubits(),
            name: gate.name().to_string(),
        };
        self.ops.push(GateOp::new(mat_gate, qubits.to_vec()));
        Ok(())
    }
}

/// A gate defined by an explicit matrix (used internally for sub-circuit import).
struct MatrixGate {
    matrix: ndarray::Array2<num_complex::Complex<f64>>,
    n_qubits: usize,
    name: String,
}

impl QuantumGate for MatrixGate {
    fn matrix(&self) -> ndarray::Array2<num_complex::Complex<f64>> {
        self.matrix.clone()
    }
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_bell_pair_probabilities() {
        let circ = bell_pair_circuit();
        let init = QubitRegister::new_zero_state(2).expect("valid");
        let state = circ.run(init).expect("run ok");

        let p00 = state.probability(0).expect("ok");
        let p11 = state.probability(3).expect("ok");
        let p01 = state.probability(1).expect("ok");
        let p10 = state.probability(2).expect("ok");

        assert!((p00 - 0.5).abs() < TOL, "p00={}", p00);
        assert!((p11 - 0.5).abs() < TOL, "p11={}", p11);
        assert!(p01.abs() < TOL, "p01={}", p01);
        assert!(p10.abs() < TOL, "p10={}", p10);
    }

    #[test]
    fn test_bell_pair_measurement_outcomes() {
        let circ = bell_pair_circuit();
        let init = QubitRegister::new_zero_state(2).expect("valid");
        let state = circ.run(init).expect("run ok");
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        for _ in 0..20 {
            let bits = circ.measure_all(&state, &mut rng).expect("ok");
            assert!(
                bits == vec![0, 0] || bits == vec![1, 1],
                "Bell pair must measure |00⟩ or |11⟩, got {:?}",
                bits
            );
        }
    }

    #[test]
    fn test_ghz_3qubits() {
        let circ = ghz_circuit(3).expect("valid");
        let init = QubitRegister::new_zero_state(3).expect("valid");
        let state = circ.run(init).expect("run ok");

        let p000 = state.probability(0).expect("ok"); // |000⟩
        let p111 = state.probability(7).expect("ok"); // |111⟩

        assert!((p000 - 0.5).abs() < TOL, "p000={}", p000);
        assert!((p111 - 0.5).abs() < TOL, "p111={}", p111);
        // All other probabilities should be ~0.
        for k in [1usize, 2, 3, 4, 5, 6] {
            let p = state.probability(k).expect("ok");
            assert!(p.abs() < TOL, "p[{}]={}", k, p);
        }
    }

    #[test]
    fn test_circuit_depth_sequential() {
        // Two X gates on the same qubit → depth 2.
        let mut c = QuantumCircuit::new(1);
        c.x(0).expect("ok");
        c.x(0).expect("ok");
        assert_eq!(c.circuit_depth(), 2);
    }

    #[test]
    fn test_circuit_depth_parallel() {
        // H on qubit 0 and H on qubit 1 can run in parallel → depth 1.
        let mut c = QuantumCircuit::new(2);
        c.h(0).expect("ok");
        c.h(1).expect("ok");
        assert_eq!(c.circuit_depth(), 1);
    }

    #[test]
    fn test_circuit_depth_mixed() {
        // H(0), H(1) [parallel] → CNOT(0,1) [serial] → depth 2.
        let mut c = QuantumCircuit::new(2);
        c.h(0).expect("ok");
        c.h(1).expect("ok");
        c.cx(0, 1).expect("ok");
        assert_eq!(c.circuit_depth(), 2);
    }

    #[test]
    fn test_circuit_register_mismatch() {
        let circ = bell_pair_circuit();
        let wrong = QubitRegister::new_zero_state(3).expect("valid");
        let err = circ.run(wrong);
        assert!(matches!(
            err,
            Err(QuantumError::CircuitRegisterMismatch { .. })
        ));
    }

    #[test]
    fn test_qft_two_qubits_normalised() {
        let circ = qft_circuit(2).expect("valid");
        let init = QubitRegister::new_zero_state(2).expect("valid");
        let state = circ.run(init).expect("run ok");
        assert!(
            state.is_normalised(1e-10),
            "QFT output should be normalised"
        );
    }

    #[test]
    fn test_qft_iqft_roundtrip() {
        // QFT then IQFT should return to the original state (up to global phase).
        let n = 3;
        let qft = qft_circuit(n).expect("valid");
        let iqft = iqft_circuit(n).expect("valid");

        let init = QubitRegister::new_basis_state(n, 3).expect("valid");
        let after_qft = qft.run(init.clone()).expect("qft ok");
        let after_iqft = iqft.run(after_qft).expect("iqft ok");

        let fidelity = init.fidelity(&after_iqft).expect("ok");
        assert!(
            (fidelity - 1.0).abs() < 1e-9,
            "QFT·IQFT fidelity should be 1, got {}",
            fidelity
        );
    }

    #[test]
    fn test_sample_returns_correct_shots() {
        let circ = bell_pair_circuit();
        let init = QubitRegister::new_zero_state(2).expect("valid");
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let shots = circ.sample(&init, 50, &mut rng).expect("ok");
        assert_eq!(shots.len(), 50);
    }

    #[test]
    fn test_toffoli_circuit() {
        // CCX(control0=0, control1=1, target=2) with big-endian gate mapping:
        //   control0=qubit0, control1=qubit1, target=qubit2
        // Start: index 3 (qubit0=1, qubit1=1, qubit2=0) - both controls set.
        // Toffoli flips qubit2: index 3 -> index 7.
        let mut c = QuantumCircuit::new(3);
        c.ccx(0, 1, 2).expect("ok");
        let init = QubitRegister::new_basis_state(3, 3).expect("valid");
        let state = c.run(init).expect("run ok");
        let p7 = state.probability(7).expect("ok");
        assert!((p7 - 1.0).abs() < TOL, "Toffoli p7={}", p7);
    }

    #[test]
    fn test_x_x_identity() {
        // X·X = I
        let mut c = QuantumCircuit::new(1);
        c.x(0).expect("ok");
        c.x(0).expect("ok");
        let init = QubitRegister::new_zero_state(1).expect("valid");
        let state = c.run(init.clone()).expect("run ok");
        let fidelity = init.fidelity(&state).expect("ok");
        assert!((fidelity - 1.0).abs() < TOL);
    }
}
