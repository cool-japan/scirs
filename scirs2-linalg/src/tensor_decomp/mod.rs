//! Tensor decompositions: Tucker, PARAFAC/CP, Tensor Train, HOSVD.
//!
//! This module provides a self-contained, lightweight API for 3-D tensor
//! decompositions using a simple [`Tensor3D`] type backed by `Vec<f64>`.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`tensor_utils`] | Core [`Tensor3D`] type, mode unfolding/folding, Khatri-Rao product, mode-n product |
//! | [`parafac`]      | PARAFAC/CP decomposition via ALS and regularised ALS |
//! | `hosvd`          | Higher-Order SVD (HOSVD) and Higher-Order Orthogonal Iteration (HOOI) |
//! | [`tucker`]       | Tucker decomposition via HOOI and core consistency diagnostic |
//! | [`tensor_train`] | Tensor Train (TT-SVD), rounding, addition, Hadamard product, inner product |
//!
//! ## Quick Example
//!
//! ```rust
//! use scirs2_linalg::tensor_decomp::tensor_utils::Tensor3D;
//! use scirs2_linalg::tensor_decomp::parafac::fit_als;
//! use scirs2_linalg::tensor_decomp::tucker::tucker_als;
//! use scirs2_linalg::tensor_decomp::tensor_train::tt_svd;
//!
//! // Build a simple 2×3×4 tensor
//! let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
//! let x = Tensor3D::new(data.clone(), [2, 3, 4]).expect("valid");
//!
//! // CP decomposition (rank 4)
//! let cp = fit_als(&x, 4, 200, 1e-6).expect("cp_als");
//! let err = cp.relative_error(&x).expect("err");
//! println!("CP relative error: {err:.2e}");
//!
//! // Tucker decomposition (ranks 2, 2, 3)
//! let tucker = tucker_als(&x, [2, 2, 3], 30, 1e-8).expect("tucker_als");
//! println!("Tucker core shape: {:?}", tucker.g.shape);
//!
//! // Tensor Train decomposition
//! let tt = tt_svd(&data, &[2, 3, 4], 10, 1e-8).expect("tt_svd");
//! println!("TT ranks: {:?}", tt.ranks());
//! ```

pub mod hosvd;
pub mod parafac;
pub mod tensor_train;
pub mod tensor_utils;
pub mod tucker;

// Convenience re-exports
pub use hosvd::{hooi, hosvd, hosvd_truncated, HOSVDDecomp};
pub use parafac::{cp_als, fit_als, fit_sparse_als, CPDecomp};
pub use tensor_train::{inner_product, tt_add, tt_hadamard, tt_round, tt_svd, TTCore, TensorTrain};
pub use tensor_utils::{mode_n_product, Tensor3D};
pub use tucker::{core_consistency_diagnostic, tucker_als, TuckerDecomp};

// N-dimensional tensor support (generalizes beyond 3D)
pub mod tensor_nd;
pub use tensor_nd::{
    cp_als as cp_als_nd, tensor_train_svd, tucker_hooi, tucker_hosvd, CpDecomposition, Tensor,
    TensorTrainDecomposition, TuckerDecomposition,
};
