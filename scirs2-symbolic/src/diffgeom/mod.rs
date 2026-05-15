//! `diffgeom` — Cadabra2-class symbolic Riemannian differential geometry.
//!
//! This module provides Cadabra2-level symbolic differential geometry built on
//! the EML substrate ([`LoweredOp`]). All computations are purely symbolic —
//! they produce expression trees that can be evaluated numerically or
//! further manipulated by the CAS.
//!
//! # Architecture
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`tensor`] | [`Tensor`] with mixed valence, [`IndexKind`], [`IndexLabel`] |
//! | [`metric`] | [`Metric`] — covariant + contravariant metric with symbolic inverse |
//! | [`contraction`] | [`contract_indices`] — trace over one upper/lower index pair |
//! | [`mod@christoffel`] | [`fn@christoffel`] — Christoffel symbols `Γᵏᵢⱼ` |
//! | [`ricci`] | [`ricci_tensor`] — Ricci curvature tensor `Rᵢⱼ` |
//! | [`einstein`] | [`einstein_tensor`] — Einstein tensor `Gᵢⱼ = Rᵢⱼ − ½ g_{ij} R` |
//! | [`mod@covariant_derivative`] | [`fn@covariant_derivative`] — `∇_μ T` for arbitrary valence |
//!
//! # Example — Schwarzschild vacuum solution
//!
//! ```rust,no_run
//! use scirs2_symbolic::diffgeom::{Metric, christoffel, ricci_tensor, einstein_tensor};
//! use scirs2_symbolic::eml::LoweredOp;
//! use ndarray::{ArrayD, IxDyn};
//!
//! // Var(0)=r, Var(1)=θ, Var(2)=φ, Var(3)=t; Var(10)=rs (Schwarzschild radius)
//! // Build diagonal metric (simplified for illustration)
//! let rs = LoweredOp::Var(10);
//! let r  = LoweredOp::Var(0);
//! let one_minus_rs_over_r = LoweredOp::Sub(
//!     Box::new(LoweredOp::Const(1.0)),
//!     Box::new(LoweredOp::Div(Box::new(rs.clone()), Box::new(r.clone()))),
//! );
//! // g_tt = -(1 - rs/r)
//! let _g_tt = LoweredOp::Neg(Box::new(one_minus_rs_over_r.clone()));
//! // ... build full 4×4 metric, compute Christoffel, Ricci, Einstein
//! ```
//!
//! # No unwrap
//!
//! All fallible operations return `Result`. `Tensor::set`/`get` use
//! `ndarray::IxDyn` indexing (panics on out-of-bounds only in debug builds).
//!
//! [`LoweredOp`]: crate::eml::LoweredOp

pub mod christoffel;
pub mod contraction;
pub mod covariant_derivative;
pub mod einstein;
pub mod metric;
pub mod ricci;
pub mod tensor;

pub use christoffel::christoffel;
pub use contraction::contract_indices;
pub use covariant_derivative::covariant_derivative;
pub use einstein::einstein_tensor;
pub use metric::{Metric, MetricError};
pub use ricci::ricci_tensor;
pub use tensor::{IndexKind, IndexLabel, Tensor};
