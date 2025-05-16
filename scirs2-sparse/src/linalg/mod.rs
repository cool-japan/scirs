//! Linear algebra operations for sparse matrices
//!
//! This module provides linear algebra operations for sparse matrices,
//! including solvers, eigenvalue computations, and matrix functions.

mod cgs;
mod ic;
mod interface;
mod iterative;
mod lgmres;
mod matfuncs;
mod minres;
mod preconditioners;
mod qmr;
mod solvers;
mod spai;

pub use cgs::{cgs, CGSOptions, CGSResult};
pub use ic::IC0Preconditioner;
pub use interface::{
    AsLinearOperator, DiagonalOperator, IdentityOperator, LinearOperator, ScaledIdentityOperator,
};
pub use iterative::{
    bicg, bicgstab, cg, gmres, BiCGOptions, BiCGSTABOptions, BiCGSTABResult, CGOptions,
    GMRESOptions, IterationResult, IterativeSolver,
};
pub use lgmres::{lgmres, LGMRESOptions, LGMRESResult};
pub use matfuncs::{expm_multiply, onenormest};
pub use minres::{minres, MINRESOptions, MINRESResult};
pub use preconditioners::{ILU0Preconditioner, JacobiPreconditioner, SSORPreconditioner};
pub use qmr::{qmr, QMROptions, QMRResult};
pub use solvers::{
    add, diag_matrix, expm, eye, inv, matmul, matrix_power, multiply, norm, sparse_direct_solve,
    sparse_lstsq, spsolve,
};
pub use spai::{SpaiOptions, SpaiPreconditioner};
