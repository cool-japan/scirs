//! Out-of-core interpolation: disk-backed coefficient storage for large datasets.
//!
//! This module provides interpolation methods that can handle datasets too large
//! to keep entirely in RAM by writing fitted coefficients to disk and reading
//! them back in chunks during prediction.
//!
//! # Provided types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`DiskStorage`] | Raw flat binary disk store (`rows × cols` of `f64`) |
//! | [`OutOfCoreRbf`] | RBF interpolation with disk-backed coefficients |
//! | [`OutOfCoreRbfConfig`] | Configuration for [`OutOfCoreRbf`] |
//! | [`OocRbfKernel`] | Kernel selection for [`OutOfCoreRbf`] |
//! | [`OutOfCoreKriging`] | Simple kriging (Gaussian covariance) with disk-backed coefficients |
//! | [`OutOfCoreKrigingConfig`] | Configuration for [`OutOfCoreKriging`] |
//!
//! # Example: RBF
//!
//! ```rust
//! use std::env::temp_dir;
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_interpolate::outofcore::{OutOfCoreRbf, OutOfCoreRbfConfig, OocRbfKernel};
//!
//! let n = 30usize;
//! let centers = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
//! let values  = centers.column(0).mapv(|x| x * (1.0 - x));
//! let query   = Array2::from_shape_fn((5, 1), |(i, _)| i as f64 / 5.0);
//!
//! let scratch = temp_dir().join("mod_doctest_ooc");
//! std::fs::create_dir_all(&scratch).ok();
//!
//! let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
//!     scratch_dir: scratch.clone(),
//!     kernel: OocRbfKernel::Gaussian,
//!     ..Default::default()
//! });
//! ooc.fit(&centers, &values).expect("fit");
//! let _ = ooc.predict(&query).expect("predict");
//! ooc.cleanup().ok();
//! std::fs::remove_dir_all(&scratch).ok();
//! ```

pub mod kriging;
pub mod rbf;
pub mod storage;

pub use kriging::{OutOfCoreKriging, OutOfCoreKrigingConfig};
pub use rbf::{OocRbfKernel, OutOfCoreRbf, OutOfCoreRbfConfig};
pub use storage::DiskStorage;
