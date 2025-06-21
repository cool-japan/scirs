//! Least squares submodule containing specialized algorithms and loss functions

pub mod bounded;
pub mod main;
pub mod robust;
pub mod separable;
pub mod sparse;
pub mod total;
pub mod weighted;

// Re-export main least squares functionality
pub use main::{least_squares, Method, Options};

// Re-export robust least squares functionality
pub use robust::{
    robust_least_squares, BisquareLoss, CauchyLoss, HuberLoss, RobustLoss, RobustOptions,
};

// Re-export weighted least squares functionality
pub use weighted::{weighted_least_squares, WeightedOptions};

// Re-export bounded least squares functionality
pub use bounded::{bounded_least_squares, BoundedOptions};

// Re-export separable least squares functionality
pub use separable::{separable_least_squares, LinearSolver, SeparableOptions, SeparableResult};

// Re-export total least squares functionality
pub use total::{
    total_least_squares, TLSMethod, TotalLeastSquaresOptions, TotalLeastSquaresResult,
};

// Re-export sparse least squares functionality
pub use sparse::{
    lsqr, sparse_least_squares, SparseInfo, SparseMatrix, SparseOptions, SparseResult,
};
