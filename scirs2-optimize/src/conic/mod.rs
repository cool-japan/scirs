//! Conic programming: SDP and SOCP solvers.
//!
//! Provides interior-point methods for:
//! - Semidefinite Programming (SDP) via Mehrotra predictor-corrector
//! - Second-Order Cone Programming (SOCP) via NT-scaling interior point

pub mod sdp;
pub mod socp;

pub use sdp::{
    matrix_completion_sdp, max_cut_sdp, MatrixCompletionSdpResult, MaxCutSdpResult, SDPProblem,
    SDPResult, SDPSolver, SDPSolverConfig,
};
pub use socp::{
    portfolio_optimization_socp, robust_ls_socp, socp_to_sdp, PortfolioSocpResult, RobustLsResult,
    SOCConstraint, SOCPConfig, SOCPProblem, SOCPResult,
};
