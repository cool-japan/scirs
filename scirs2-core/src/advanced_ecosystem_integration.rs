//! Advanced Mode Ecosystem Integration (Refactored)
//!
//! This module has been refactored to comply with the 2000-line policy.
//! The original implementation has been split into multiple focused modules.

// Re-export everything from the submodules
pub use self::communication::*;
pub use self::coordinator::*;
pub use self::performance::*;
pub use self::resources::*;
pub use self::types::*;
pub use self::workflow::*;

// Module declarations
mod communication;
mod coordinator;
mod performance;
mod resources;
mod types;
mod workflow;
