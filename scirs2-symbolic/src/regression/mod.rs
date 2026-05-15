//! Symbolic regression — discover closed-form formulas from data.
//!
//! High-level API: [`fn@discover`] takes feature/target arrays and returns a
//! ranked list of [`DiscoveredFormula`]s. The default search uses beam-search
//! over a small grammar of `LoweredOp` building blocks; constants are seeded
//! into the population.
//!
//! For the FOUNDATIONAL Phase 1 API, this module ships single-output search
//! ([`fn@discover`]), multi-output search ([`fn@discover_multi`]), and SINDy-style
//! ODE discovery from trajectory data ([`fn@discover_ode`]). SMT-pruned
//! constrained search (`with_constraints`) and joint cross-output / joint
//! cross-dimension sparse-regression follow in v0.4.5.
//!
//! # Examples
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use scirs2_symbolic::regression::{discover, SrConfig};
//!
//! // y = x — discover the identity formula.
//! let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
//! let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
//! let targets = Array1::from_vec(xs);
//!
//! let config = SrConfig::default().with_max_iter(10);
//! let results = discover(features.view(), targets.view(), &config);
//! assert!(!results.is_empty());
//! ```

pub mod config;
pub mod discover;
pub mod discover_multi;
pub mod discover_ode;
pub mod fitness;
pub mod formula;
pub mod with_constraints;

pub use config::{BuildingBlock, SrConfig};
pub use discover::discover;
#[cfg(feature = "numa")]
pub use discover::NUMA_DISPATCH_THRESHOLD;
pub use discover_multi::{discover_multi, discover_multi_best};
pub use discover_ode::{discover_ode, discover_ode_best, OdeConfig};
pub use fitness::Fitness;
pub use formula::DiscoveredFormula;
pub use with_constraints::{with_constraints, ConstrainedConfig, Constraint};
