//! Multitaper spectral estimation.
//!
//! This module provides functions for spectral analysis using multitaper methods,
//! which provide robust spectral estimates with reduced variance and bias compared
//! to conventional approaches. The implementation includes Discrete Prolate
//! Spheroidal Sequences (DPSS) tapers, also known as Slepian sequences.

// Import internal modules
mod adaptive;
mod psd;
mod utils;
mod windows;

// Re-export public components
pub use adaptive::adaptive_psd;
pub use psd::{multitaper_spectrogram, pmtm};
pub use utils::{coherence, multitaper_filtfilt};
pub use windows::dpss;

// No direct imports needed for the module - submodules import their own dependencies
