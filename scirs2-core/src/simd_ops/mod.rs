//! Auto-generated module structure

pub mod autooptimizer_traits;
pub mod bessel_functions;
pub mod functions;
pub mod functions_2;
pub mod functions_3;
pub mod functions_4;
pub mod functions_5;
pub mod functions_6;
pub mod gpu_detection;
pub mod matmul;
pub mod types;

// Re-export all types
pub use autooptimizer_traits::*;
pub use functions::*;
pub use functions_2::*;
pub use functions_3::*;
pub use functions_4::*;
pub use functions_5::*;
pub use functions_6::*;
pub use gpu_detection::*;
pub use matmul::*;
pub use types::*;

#[cfg(test)]
#[path = "simd_ops_tests.rs"]
mod tests;
