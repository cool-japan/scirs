//! Tensor decompositions for dense N-way arrays.
//!
//! This module provides a unified, generic-float API for high-dimensional
//! tensor computations.  All algorithms operate on the [`Tensor<F>`] type
//! defined in [`core`] and accept any `F: Float + NumAssign + Sum +
//! ScalarOperand + Send + Sync + 'static`.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`core`] | Dense N-way tensor with shape/strides, unfold/fold, n-mode product |
//! | [`hosvd()`] | HOSVD, HOOI, and auto-rank truncated HOSVD |
//! | [`cp_decomp`] | CP-ALS and CP-gradient decompositions |
//! | [`tensor_train`] | TT-SVD, TT-Cross, TT-Rounding, and element-wise TT ops |
//!
//! ## Quick Tour
//!
//! ### Tucker decomposition (HOSVD / HOOI)
//!
//! ```rust
//! use scirs2_linalg::tensor::core::Tensor;
//! use scirs2_linalg::tensor::hosvd::{hosvd, hooi};
//!
//! let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
//! let t = Tensor::new(data, vec![2, 3, 4]).expect("valid");
//!
//! // HOSVD — fast initialisation
//! let r = hosvd(&t, &[2, 2, 3]).expect("hosvd");
//! println!("core shape: {:?}", r.core.shape);
//!
//! // HOOI — iterative refinement (better approximation)
//! let r2 = hooi(&t, &[2, 2, 3], 100).expect("hooi");
//! println!("relative error: {:.2e}", r2.relative_error(&t).expect("err"));
//! ```
//!
//! ### CP decomposition
//!
//! ```rust
//! use scirs2_linalg::tensor::core::Tensor;
//! use scirs2_linalg::tensor::cp_decomp::{cp_als, CPConfig};
//!
//! let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
//! let t = Tensor::new(data, vec![2, 3, 4]).expect("valid");
//!
//! let cfg = CPConfig { max_iter: 200, ..Default::default() };
//! let cp = cp_als(&t, 4, &cfg).expect("cp_als");
//! println!("final loss: {:.4e}", cp.loss.last().copied().unwrap_or(0.0));
//! ```
//!
//! ### Tensor-Train (TT / MPS)
//!
//! ```rust
//! use scirs2_linalg::tensor::core::Tensor;
//! use scirs2_linalg::tensor::tensor_train::{tt_svd, tt_add};
//!
//! let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
//! let t = Tensor::new(data, vec![2, 3, 4]).expect("valid");
//!
//! let tt = tt_svd(&t, 1e-10_f64).expect("tt_svd");
//! let tt2 = tt_add(&tt, &tt).expect("tt_add");  // 2 * t
//! println!("TT ranks: {:?}", tt.ranks);
//! ```

pub mod core;
pub mod cp_decomp;
pub mod hosvd;
pub mod tensor_train;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

pub use core::{Tensor, TensorScalar};
pub use cp_decomp::{cp_als, cp_grad, cp_reconstruct, CPConfig, CPResult};
pub use hosvd::{hooi, hosvd, truncated_hosvd, HOSVDResult};
pub use tensor_train::{tt_add, tt_cross, tt_dot, tt_hadamard, tt_round, tt_scale, tt_svd, TTCore};
