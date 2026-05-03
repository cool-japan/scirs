//! Network reliability analysis.
//!
//! This module provides algorithms for computing and estimating the reliability
//! of networks under random component (edge) failures.
//!
//! ## Problem setting
//!
//! Each edge `e` in the graph independently fails with probability `1 − p_e`
//! (survives with probability `p_e`).  Network reliability questions ask: what
//! is the probability that the surviving edges satisfy some connectivity
//! criterion?
//!
//! | Struct | Question |
//! |--------|----------|
//! | [`NetworkReliability`] | P(source `s` can reach terminal `t`) — two-terminal reliability |
//! | [`AllTerminalReliability`] | P(all nodes mutually reachable) — all-terminal reliability |
//! | [`ReliabilityPolynomial`] | Exact polynomial in `p` for small networks via inclusion-exclusion |
//! | [`BDD`] | Exact reliability via binary decision diagrams (BDD) |
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::reliability::{NetworkReliability, AllTerminalReliability};
//!
//! // Triangle graph: 0-1, 1-2, 0-2
//! let mut adj = Array2::<f64>::zeros((3, 3));
//! adj[[0,1]] = 0.9; adj[[1,0]] = 0.9;
//! adj[[1,2]] = 0.8; adj[[2,1]] = 0.8;
//! adj[[0,2]] = 0.7; adj[[2,0]] = 0.7;
//!
//! let rel = NetworkReliability::new(0, 2);
//! let estimate = rel.monte_carlo(&adj, 10000, Some(42)).unwrap();
//! println!("Two-terminal reliability ≈ {estimate:.4}");
//!
//! let all_rel = AllTerminalReliability::new();
//! let est2 = all_rel.monte_carlo(&adj, 10000, Some(42)).unwrap();
//! println!("All-terminal reliability ≈ {est2:.4}");
//! ```

pub mod network_reliability;

pub use network_reliability::{
    AllTerminalReliability, ComponentFailureTree, NetworkReliability, ReliabilityPolynomial, BDD,
};

pub mod reliability_extra;

pub use reliability_extra::{
    all_terminal_reliability_factoring, k_edge_connectivity, k_vertex_connectivity,
    min_cut_reliability_bound, reliability_polynomial, two_terminal_reliability,
};
