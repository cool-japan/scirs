//! Modal Analysis and Operational Modal Analysis (OMA)
//!
//! This module provides methods for extracting modal parameters (natural frequencies,
//! damping ratios, and mode shapes) from measured vibration data.
//!
//! # Module Structure
//!
//! | Submodule | Description |
//! |-----------|-------------|
//! | [`fdd`] | Frequency Domain Decomposition (FDD, EFDD) |
//! | [`ssi`] | Stochastic Subspace Identification (SSI-Cov, SSI-Data) |
//! | [`free_vibration`] | Free vibration methods (ITD, RDT, NExT-ERA) |
//!
//! # Algorithms Overview
//!
//! ## Frequency Domain Decomposition (FDD)
//! - Estimate the power spectral density (PSD) matrix from multichannel data.
//! - Apply SVD at each frequency line; the singular values form a clear
//!   peak structure around the natural frequencies.
//! - Mode shapes are taken from the corresponding singular vectors.
//!
//! ## Stochastic Subspace Identification (SSI)
//! - **SSI-Cov**: uses output covariance functions as the projection.
//! - **SSI-Data**: directly uses the output data organised in a Hankel matrix.
//! - Both algorithms yield complex poles; natural frequencies and damping
//!   ratios are extracted from the imaginary and real parts.
//!
//! ## Free Vibration / NExT-ERA
//! - Random Decrement Technique (RDT) extracts free decay records.
//! - NExT converts random response cross-correlation into impulse responses.
//! - ERA (Eigensystem Realization Algorithm) fits a state-space model to
//!   the free-response / impulse-response data.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_signal::modal::fdd::{frequency_domain_decomposition, FDDConfig};
//! ```

pub mod fdd;
pub mod free_vibration;
pub mod mac;
pub mod ssi;

// Convenience re-exports

// FDD
pub use fdd::{
    enhanced_fdd, frequency_domain_decomposition, mac_criterion, svd_psd, EFDDResult, FDDConfig,
    FDDResult,
};

// SSI
pub use ssi::{
    build_block_hankel, covariance_driven_ssi, data_driven_ssi, stabilization_diagram, SSIConfig,
    SSIResult, StabilizationDiagram,
};

// Free vibration
pub use free_vibration::{
    era, natural_excitation_technique, random_decrement, ERAConfig, ERAResult,
    IbrahimTimeDomain, NExTConfig, RDTConfig, RDTResult,
};
