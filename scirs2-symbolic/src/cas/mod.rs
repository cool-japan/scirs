//! Computer algebra system — built on the EML substrate.
//!
//! Phase 2 centerpiece:
//! - [`mod@canonicalize`] — the world-first EML-IR-native CAS canonical form.
//!   Hash equality on [`Canonical`] values implies mathematical equality on
//!   a documented decidable subset (polynomial subring + analytic identities).
//! - [`canonical_rules`] — additional rewrite rules beyond [`crate::eml::simplify`]
//!   that complete `canonicalize` (log/exp expansion, power identities, etc.).
//!
//! Phase 1 includes:
//! - [`smt`] — native OxiZ SMT solver wrapper for symbolic decision problems
//!   (feature-gated on `smt`).
//!
//! A0 prerequisite for A1/A3/A4:
//! - [`pattern`] — structural pattern-match / instantiation engine for
//!   `LoweredOp` expressions; prerequisite for the identity database (A1),
//!   certified rewrites (A3), and e-graphs (A4).
//!
//! Phase 2 + Phase 1 integration:
//! - [`identity_proof`] — discover whether `(x, y)` data matches a known
//!   closed-form identity by combining SR discovery, canonicalization, and
//!   numerical certification.

pub mod ad;
pub mod canonical_rules;
pub mod canonicalize;
pub mod cardano_ferrari;
pub mod certified_value;
pub mod cse_dag;
pub mod e_graph;
pub mod hermite_reduction;
pub mod identity_db;
pub mod identity_proof;
pub mod inverse_symbolic;
pub mod matrix_exp;
pub mod matrix_ops;
pub mod mle_catalog;
pub mod observed_fisher;
pub mod pattern;
pub mod quadratic_line_search;
pub mod reversible;
pub mod series;
pub mod solve;
pub mod solve_ode;
pub mod solve_system;
pub mod spectral_2x2;

#[cfg(feature = "smt")]
pub mod smt;

#[cfg(feature = "smt")]
pub mod certified_rewrite;

pub use ad::{
    batch_eval_grad, fourth_derivative, grad_canonical, hessian_canonical, higher_order_grad,
    jacobian_canonical, jvp, taylor_higher_order, third_derivative, vjp, AdError, GradGraph,
};
pub use canonical_rules::apply_canonical_rules;
pub use canonicalize::{canonicalize, Canonical};
pub use cardano_ferrari::{solve_cubic, solve_quartic};
pub use certified_value::{CertifiedInterval, CertifiedValue, CertifiedValueError};
pub use cse_dag::CseDag;
pub use e_graph::{canonicalize_egraph, SaturationBudget};
pub use identity_db::{
    apply_identity_db, apply_standard_identity_db, Identity, IdentityDb, IdentityKind,
    MAX_IDENTITY_ITERS,
};
pub use pattern::{
    instantiate, match_pattern, BinaryKind, Bindings, Pattern, PatternError, UnaryKind,
};
pub use reversible::{canonicalize_traced, RewriteStep, RewriteTrace};
pub use series::{pade, taylor, SeriesError, MAX_TAYLOR_ORDER};
pub use solve::{solve, solve_zero, SolveError, SolveResult};

// Track 1: Risch-LITE rational integration
pub mod integrate_rational;
pub use integrate_rational::{
    integrate_polynomial, integrate_rational, try_integrate, IntegrateRationalError,
};

// Track 2: Moments and Fisher information catalogs
pub mod moments_catalog;
pub use moments_catalog::{symbolic_moments_catalog, MomentsCatalog, MomentsError};
pub mod expected_fisher_catalog;
pub use expected_fisher_catalog::{expected_fisher_catalog, ExpectedFisherError};

// Track 3: Noether conservation via Poisson brackets
pub mod noether_conservation;
pub use identity_proof::{
    builtin_identity_db, discover_identity, KnownIdentity, ProofCertificate, ProofError,
};
pub use inverse_symbolic::{recover, Candidate, RecoverOpts};
pub use matrix_exp::{
    expm_2x2, expm_3x3, expm_diag_2x2, expm_diag_3x3, expm_nilpotent_2x2, MatrixExpError,
};
pub use matrix_ops::{
    adjugate_2x2, adjugate_3x3, adjugate_4x4, cofactor_3x3, det_2x2, det_3x3, det_4x4, inverse_2x2,
    inverse_3x3, inverse_4x4, trace_2x2, trace_3x3, trace_4x4, InverseResult,
};
pub use mle_catalog::{symbolic_mle_catalog, DistFamily, MleError, MleEstimator};
pub use noether_conservation::{
    check_conservation_1dof, check_conservation_ndof, first_integrals_1dof, poisson_bracket_1dof,
    poisson_bracket_ndof, ConservationCheck, NoetherError,
};
pub use observed_fisher::observed_fisher_matrix;
pub use quadratic_line_search::{closed_form_step, LineSearchError};
pub use spectral_2x2::{eig_symmetric_2x2, SymmetricEig2};

// A1: Multivariate system solver and symbolic ODE solver
pub use solve_ode::{solve_ode, OdeKind, OdeSolution, SolveOdeError};
pub use solve_system::{
    solve_system, SystemKind, SystemSolveError, SystemSolveResult, MAX_BUCHBERGER_STEPS,
};

#[cfg(feature = "smt")]
pub use smt::{EmlSmtSolver, SmtError, SmtResult};

#[cfg(feature = "smt")]
pub use certified_rewrite::{
    rewrite_certified, rewrite_certified_fixpoint, CertifiedRewriteError, CertifiedRule,
    MAX_CERT_ITER,
};
