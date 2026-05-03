//! Auto-generated module structure

pub mod arimaparams_traits;
pub mod autoarimaoptions_traits;
pub mod expsmoothingparams_traits;
pub mod functions;
pub mod functions_2;
pub mod functions_3;
pub mod functions_4;
pub mod neural;
pub mod types;

// Re-export all types
pub use arimaparams_traits::*;
pub use autoarimaoptions_traits::*;
pub use expsmoothingparams_traits::*;
pub use functions::*;
pub use functions_2::*;
pub use functions_3::*;
pub use functions_4::*;
pub use types::*;

// Note: neural_tests.rs is included via #[path = "neural_tests.rs"] inside neural.rs.
// No need to declare it here as a separate module.
