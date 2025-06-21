//! Advanced interpolation methods for specialized use cases.
//!
//! This module provides sophisticated interpolation algorithms that go beyond basic
//! linear and polynomial methods. These methods are designed for specific types of
//! interpolation problems and data characteristics.
//!
//! ## Available Methods
//!
//! ### Scattered Data Interpolation
//! - **[`RBFInterpolator`]** - Radial Basis Function interpolation for irregular data
//! - **[`EnhancedRBFInterpolator`]** - RBF with automatic parameter selection
//! - **[`ThinPlateSpline`]** - Smooth interpolation minimizing bending energy
//!
//! ### Uncertainty Quantification
//! - **[`KrigingInterpolator`]** - Gaussian process regression with error estimates
//! - **[`EnhancedKriging`]** - Advanced kriging with directional correlations
//! - **[`FastKriging`]** - Scalable kriging for large datasets
//!
//! ### Robust Methods
//! - **[`AkimaSpline`]** - Robust spline interpolation with outlier resistance
//! - **[`BarycentricInterpolator`]** - Numerically stable polynomial interpolation
//!
//! ## Choosing the Right Method
//!
//! ### For Scattered 2D/3D Data
//! ```rust
//! use scirs2_interpolate::advanced::rbf::{RBFInterpolator, RBFKernel};
//! // Use RBF with Gaussian kernel for smooth functions
//! // Use Thin Plate Spline for natural-looking surfaces
//! ```
//!
//! ### When You Need Uncertainty Estimates
//! ```rust
//! use scirs2_interpolate::advanced::kriging::KrigingInterpolator;
//! // Kriging provides both interpolated values and confidence intervals
//! ```
//!
//! ### For Noisy or Outlier-Prone Data
//! ```rust
//! use scirs2_interpolate::advanced::akima::AkimaSpline;
//! // Akima splines are less sensitive to outliers than cubic splines
//! ```
//!
//! ## Performance Considerations
//!
//! | Method | Construction Time | Evaluation Time | Memory Usage | Best For |
//! |--------|------------------|-----------------|--------------|----------|
//! | RBF | O(n³) | O(n) | O(n²) | Small to medium datasets |
//! | Kriging | O(n³) | O(n) | O(n²) | When uncertainty is needed |
//! | Fast Kriging | O(n log n) | O(log n) | O(n) | Large datasets |
//! | Thin Plate | O(n³) | O(n) | O(n²) | Smooth natural surfaces |
//! | Akima | O(n) | O(log n) | O(n) | 1D robust interpolation |

pub mod akima;
pub mod barycentric;
pub mod enhanced_kriging;
pub mod enhanced_rbf;
// Fast Kriging algorithms for large datasets
// Currently using a reexport with defined API but placeholder implementations
// This module is under active development with algorithms being optimized
// Full implementations will be available in future updates
pub mod fast_kriging_reexports;
// Aliasing to maintain API compatibility
pub use fast_kriging_reexports as fast_kriging;
pub mod kriging;
pub mod rbf;
pub mod thinplate;
