//! Automatic differentiation support for linear algebra operations
//!
//! This module integrates the linear algebra operations with automatic
//! differentiation from the scirs2-autograd crate. It provides differentiable
//! versions of common matrix operations.

#![cfg(feature = "autograd")]

// Note: The autograd module is currently undergoing a major redesign to match
// the new API of scirs2-autograd crate. The following sub-modules are temporarily
// disabled until they can be updated:
//
// - batch: Batch matrix operations
// - factorizations: Matrix factorizations (LU, QR, etc.)
// - matrix_calculus: Gradient, Hessian, Jacobian calculations
// - special: Special matrix functions (log, sqrt, pseudo-inverse)
// - tensor_algebra: Tensor operations (contraction, outer product)
// - transformations: Matrix transformations (rotation, scaling, etc.)
//
// For basic autodiff functionality, see the examples in examples/autograd_simple_example.rs
// which demonstrates how to use scirs2-autograd directly with linear algebra operations.