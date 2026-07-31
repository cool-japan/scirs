//! Quantum computing simulation primitives.
//!
//! This module provides a complete, pure-Rust quantum circuit simulator
//! based on statevector simulation.  It covers:
//!
//! | Submodule | Contents |
//! |-----------|----------|
//! | [`error`] | Error types and `QuantumResult` alias |
//! | [`qubits`] | [`qubits::Qubit`] and [`qubits::QubitRegister`] statevectors |
//! | [`gates`] | Gate trait + standard gate set + [`gates::apply_gate`] |
//! | [`circuit`] | [`circuit::QuantumCircuit`] builder and executor |
//!
//! # Design Decisions
//!
//! * **Statevector simulation** — the full 2^n complex-amplitude vector is
//!   kept in memory.  This is exact but exponential in qubit count; practical
//!   for n ≤ ~25 on a laptop.
//! * **Qubit-index convention** — qubit 0 is the *least*-significant bit of
//!   the basis index (rightmost qubit in ket notation |q_{n-1}…q_1 q_0⟩).
//! * **Pure Rust** — no C/Fortran dependencies; complex arithmetic via
//!   [`num-complex`](https://docs.rs/num-complex), arrays via
//!   [`ndarray`](https://docs.rs/ndarray).
//! * **No unwrap** — all fallible operations return `QuantumResult<T>`.
//!
//! # Quick Start
//!
//! ## Single qubit
//!
//! ```rust
//! use scirs2_core::quantum::qubits::Qubit;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha20Rng;
//!
//! let q = Qubit::new_superposition(std::f64::consts::PI / 2.0, 0.0);
//! let mut rng = ChaCha20Rng::seed_from_u64(0);
//! let (outcome, _post) = q.measure(&mut rng);
//! assert!(outcome == 0 || outcome == 1);
//! ```
//!
//! ## Bell pair
//!
//! ```rust
//! use scirs2_core::quantum::circuit::bell_pair_circuit;
//! use scirs2_core::quantum::qubits::QubitRegister;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha20Rng;
//!
//! let circ = bell_pair_circuit();
//! let init = QubitRegister::new_zero_state(2).expect("should succeed");
//! let state = circ.run(init).expect("should succeed");
//!
//! let mut rng = ChaCha20Rng::seed_from_u64(42);
//! let bits = circ.measure_all(&state, &mut rng).expect("should succeed");
//! // Bell pair always measures 00 or 11.
//! assert!(bits == vec![0, 0] || bits == vec![1, 1]);
//! ```
//!
//! ## Quantum Fourier Transform
//!
//! ```rust
//! use scirs2_core::quantum::circuit::{qft_circuit, iqft_circuit};
//! use scirs2_core::quantum::qubits::QubitRegister;
//!
//! let n = 4;
//! let qft = qft_circuit(n).expect("should succeed");
//! let iqft = iqft_circuit(n).expect("should succeed");
//!
//! let state = QubitRegister::new_basis_state(n, 5).expect("should succeed");
//! let transformed = qft.run(state.clone()).expect("should succeed");
//! let recovered = iqft.run(transformed).expect("should succeed");
//!
//! // QFT followed by IQFT should recover the original state.
//! let fidelity = state.fidelity(&recovered).expect("should succeed");
//! assert!((fidelity - 1.0).abs() < 1e-9);
//! ```

pub mod circuit;
pub mod error;
pub mod gates;
pub mod qubits;

// Convenience re-exports.
pub use circuit::{
    bell_pair_circuit, ghz_circuit, iqft_circuit, phase_estimation_circuit, qft_circuit,
    qft_circuit_with_swap, QuantumCircuit,
};
pub use error::{QuantumError, QuantumResult};
pub use gates::{
    apply_gate, check_unitary, matrix_product, tensor_product_matrices, Fredkin, Hadamard,
    Identity, PauliX, PauliY, PauliZ, PhaseS, PhaseSdg, PhaseShift, PhaseT, PhaseTdg, QuantumGate,
    RotX, RotY, RotZ, Toffoli, Unitary1Q, CNOT, CU, CZ, ISWAP, SWAP,
};
pub use qubits::{Qubit, QubitRegister};
