//! Extended precision eigenvalue computations
//!
//! This module provides functions for computing eigenvalues and eigenvectors
//! using extended precision arithmetic for improved accuracy.
//!
//! # Overview
//!
//! This module contains implementations for:
//!
//! * Computing eigenvalues of general matrices with extended precision
//! * Computing eigenvalues and eigenvectors of general matrices with extended precision
//! * Computing eigenvalues of symmetric/Hermitian matrices with extended precision
//! * Computing eigenvalues and eigenvectors of symmetric/Hermitian matrices with extended precision
//!
//! These functions are particularly useful for handling ill-conditioned matrices where
//! standard precision computations may suffer from numerical instability.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::extended_precision::eigen::{extended_eigvalsh, extended_eigh};
//!
//! // Create a symmetric matrix
//! let a = array![
//!     [2.0_f32, 1.0, 0.0],
//!     [1.0, 2.0, 1.0],
//!     [0.0, 1.0, 2.0]
//! ];
//!
//! // Compute eigenvalues only
//! let eigvals = extended_eigvalsh::<_, f64>(&a.view(), None, None).unwrap();
//! println!("Eigenvalues: {:?}", eigvals);
//!
//! // Compute both eigenvalues and eigenvectors
//! let (eigvals, eigvecs) = extended_eigh::<_, f64>(&a.view(), None, None).unwrap();
//! println!("Eigenvalues: {:?}", eigvals);
//! println!("Eigenvectors shape: {:?}", eigvecs.shape());
//! ```

mod generalized_eigen;
mod iterative;
mod standard_eigen;

// Re-export the main public functions
pub use standard_eigen::{
    extended_eig, extended_eigh, extended_eigvals, extended_eigvalsh, EigenResult,
};

pub use iterative::advanced_precision_eigh;

// Internal functions are available to each other
pub(super) use generalized_eigen::*;
pub(super) use iterative::*;
