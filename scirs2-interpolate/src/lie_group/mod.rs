//! Kernel interpolation on Lie groups and homogeneous spaces.
//!
//! This module implements radial basis function (RBF) interpolation on curved
//! geometric spaces, using geodesic distances as the distance metric.
//!
//! # Supported Spaces
//!
//! | Space | Type | Distance |
//! |-------|------|----------|
//! | S² (2-sphere) | Homogeneous space SO(3)/SO(2) | Great-circle arc length |
//! | SO(3) | Lie group (rotations) | `2·arccos(|q₁·q₂|)` |
//! | SE(3) | Lie group (rigid motions) | Weighted product metric |
//!
//! # Kernel Functions
//!
//! Three kernels are available via [`kernel::GeometricKernel`]:
//!
//! - [`kernel::GeometricKernel::Heat`]: `k(d) = exp(-d²/σ²)` — smooth Gaussian-type.
//! - [`kernel::GeometricKernel::Matern`]: Matérn family (ν = 0.5, 1.5, 2.5) — stationary.
//! - [`kernel::GeometricKernel::SphericalHarmonic`]: Zonal kernel via Legendre polynomial
//!   expansion — positive definite on S².
//!
//! # Quick Start
//!
//! ## S² Interpolation
//!
//! ```rust
//! use scirs2_interpolate::lie_group::sphere::SphereRbfInterpolator;
//! use scirs2_interpolate::lie_group::kernel::GeometricKernel;
//!
//! // Data on the unit sphere (cardinal directions)
//! let points = vec![
//!     [1.0_f64, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0],
//! ];
//! let values = vec![1.0_f64, 2.0, 3.0];
//!
//! let interp = SphereRbfInterpolator::new(
//!     &points, &values,
//!     GeometricKernel::Heat { sigma: 1.0 },
//!     1e-6,
//! ).expect("construction should succeed");
//!
//! let v = interp.eval(&[1.0, 0.0, 0.0]);
//! assert!((v - 1.0).abs() < 0.1);
//! ```
//!
//! ## SO(3) Interpolation
//!
//! ```rust
//! use scirs2_interpolate::lie_group::so3::So3RbfInterpolator;
//! use scirs2_interpolate::lie_group::kernel::GeometricKernel;
//!
//! // Identity rotation and 90-degree rotation around Z
//! let identity = [1.0_f64, 0.0, 0.0, 0.0];
//! let rot_z90 = [0.7071_f64, 0.0, 0.0, 0.7071];
//! let points = vec![identity, rot_z90];
//! let values = vec![0.0_f64, 1.0];
//!
//! let interp = So3RbfInterpolator::new(
//!     &points, &values,
//!     GeometricKernel::Heat { sigma: 1.0 },
//!     1e-6,
//! ).expect("construction should succeed");
//! let v = interp.eval(&identity);
//! assert!(v.is_finite());
//! ```
//!
//! # References
//!
//! - Fuselier & Wright (2012). "Scattered data interpolation on embedded submanifolds
//!   with restricted positive definite kernels". SIAM J. Numer. Anal.
//! - Le Brigant & Puechmorel (2019). "Quantization and clustering on Riemannian manifolds
//!   with an application to air traffic analysis". JMIV.

pub mod kernel;
pub mod se3;
pub mod so3;
pub mod sphere;

pub use kernel::GeometricKernel;
pub use se3::{Se3Point, Se3RbfInterpolator};
pub use so3::So3RbfInterpolator;
pub use sphere::SphereRbfInterpolator;
