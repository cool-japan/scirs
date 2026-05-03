//! Advanced numerical quadrature rules.
//!
//! This module provides high-accuracy quadrature and cubature methods beyond the
//! basic rules already in `scirs2-integrate`:
//!
//! - `gaussian`: Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre,
//!   Gauss-Chebyshev T1/T2, Gauss-Jacobi, and Gauss-Kronrod G7K15.
//! - `cubature`: Monte Carlo, Quasi-Monte Carlo (Sobol), product Gauss-Legendre,
//!   Genz-Malik adaptive rule, and 1D Romberg integration.

pub mod contour_cc;
pub mod cubature;
pub mod filon;
pub mod filon_clenshaw;
pub mod gaussian;
pub mod smolyak;
pub mod sparse_grid;

pub use contour_cc::{
    adaptive_cc, contour_integrate_cc, filon_cc_oscillatory, filon_cc_oscillatory_sin,
    ContourConfig, ContourResult, ContourType,
};
pub use cubature::{genz_malik, monte_carlo, product_gauss, quasi_monte_carlo, romberg};
pub use gaussian::{
    gauss_chebyshev_t1, gauss_chebyshev_t2, gauss_hermite, gauss_jacobi, gauss_kronrod_g7k15,
    gauss_laguerre, gauss_legendre, quad_gauss_hermite, quad_gauss_legendre,
};
pub use sparse_grid::{SmolyakConfig, SmolyakGrid, UnivariateRule};
