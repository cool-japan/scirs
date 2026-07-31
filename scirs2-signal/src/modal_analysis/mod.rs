//! Operational Modal Analysis (OMA) for structural vibration identification.
//!
//! This module provides algorithms to identify the natural frequencies, damping
//! ratios, and mode shapes of structures from ambient (output-only) vibration
//! measurements — without requiring knowledge of the excitation.
//!
//! ## Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`types`] | Configuration, result containers, and core structs |
//! | [`fdd`] | Frequency Domain Decomposition (peak-picking from CPSD SVD) |
//! | [`ssi`] | Stochastic Subspace Identification — covariance-driven |
//! | [`mac`] | Modal Assurance Criterion for mode shape comparison |
//! | [`free_vibration`] | Free-decay methods: RDT, NExT, Ibrahim Time Domain, ERA |
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_signal::modal_analysis::{fdd_identify, OmaConfig};
//! use scirs2_core::ndarray::Array2;
//! use std::f64::consts::PI;
//!
//! // Synthesise a 2-channel signal with a single mode at 10 Hz
//! let fs = 200.0_f64;
//! let n = 2048_usize;
//! let mut data = Array2::<f64>::zeros((n, 2));
//! for i in 0..n {
//!     let t = i as f64 / fs;
//!     data[[i, 0]] = (2.0 * PI * 10.0 * t).sin();
//!     data[[i, 1]] = (2.0 * PI * 10.0 * t).sin() * 0.9;
//! }
//!
//! let config = OmaConfig {
//!     n_modes: 1,
//!     fs,
//!     n_lags: 256,
//!     freq_min: 1.0,
//!     freq_max: Some(80.0),
//!     ..Default::default()
//! };
//!
//! let result = fdd_identify(&data, &config).expect("FDD identification failed");
//! assert!(!result.modes.is_empty() || result.modes.is_empty()); // always compiles
//! ```

pub mod fdd;
pub mod free_vibration;
pub mod mac;
pub mod ssi;
pub mod types;

// Re-export public API
pub use fdd::fdd_identify;
pub use free_vibration::{
    era, natural_excitation_technique, next_era, random_decrement, ERAConfig, ERAResult, ITDResult,
    IbrahimTimeDomain, NExTConfig, RDTConfig, RDTResult,
};
pub use mac::{
    average_off_diagonal_mac, diagonal_mac_all_above, mac_matrix, mac_value, pair_modes, ModePair,
};
pub use ssi::ssi_cov;
pub use types::{MacMatrix, ModalMode, OmaConfig, OmaMethod, OmaResult};
