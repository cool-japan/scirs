//! Advanced Fusion Intelligence Modules
//!
//! This module provides the modular implementation of the Advanced Fusion Intelligence
//! system, breaking down the large monolithic implementation into focused modules
//! for better maintainability and organization.

pub mod consciousness;
pub mod distributed;
pub mod evolution;
pub mod meta_learning;
pub mod neuromorphic;
pub mod quantum;
pub mod temporal;

// Re-export all public types - suppress ambiguity warnings
#[allow(ambiguous_glob_reexports)]
pub use consciousness::*;
#[allow(ambiguous_glob_reexports)]
pub use distributed::*;
#[allow(ambiguous_glob_reexports)]
pub use evolution::*;
#[allow(ambiguous_glob_reexports)]
pub use meta_learning::*;
#[allow(ambiguous_glob_reexports)]
pub use neuromorphic::*;
#[allow(ambiguous_glob_reexports)]
pub use quantum::*;
#[allow(ambiguous_glob_reexports)]
pub use temporal::*;
