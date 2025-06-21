//! ODE solver implementations.

mod adaptive;
mod enhanced_bdf;
mod enhanced_lsoda;
mod explicit;
mod implicit;
mod local_extrapolation;
mod lsoda;
mod radau_mass;
// Temporarily disabled SIMD module due to implementation complexity
// #[cfg(feature = "simd")]
// mod simd_explicit;

/// Re-exports
pub use adaptive::{dop853_method, rk23_method, rk45_method};
pub use enhanced_bdf::enhanced_bdf_method;
pub use enhanced_lsoda::enhanced_lsoda_method;
pub use explicit::{euler_method, rk4_method};
pub use implicit::{bdf_method, radau_method};
pub use local_extrapolation::{
    gragg_bulirsch_stoer_method, richardson_extrapolation_step, ExtrapolationBaseMethod,
    ExtrapolationOptions, ExtrapolationResult,
};
pub use lsoda::lsoda_method;
pub use radau_mass::radau_method_with_mass;

// Temporarily disabled SIMD methods due to implementation complexity
// #[cfg(feature = "simd")]
// pub use simd_explicit::{simd_rk45_method, simd_rk4_method};
