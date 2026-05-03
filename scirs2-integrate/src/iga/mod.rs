//! Isogeometric Analysis (IGA) using NURBS and B-spline basis functions.
//!
//! IGA replaces classical finite element basis functions (Lagrange polynomials)
//! with the same NURBS basis functions used in CAD geometry representations.
//! This gives:
//!
//! - **Exact geometry**: circles, conic sections are represented exactly.
//! - **Higher-order continuity**: C^{p−1} continuity (for unrepeated knots).
//! - **Higher accuracy per DOF**: smooth, high-order approximation spaces.
//!
//! ## Modules
//!
//! - [`bspline`] — B-spline basis functions, curves, and surfaces.
//! - [`nurbs`] — NURBS (Non-Uniform Rational B-Splines) extending B-splines.
//! - [`iga_solver`] — IGA-based PDE solvers (1-D and 2-D Poisson problems).
//!
//! ## Example: 1-D Poisson solve
//!
//! ```rust
//! use scirs2_integrate::iga::{IGASolver, IGASolver1DConfig};
//!
//! // Solve −u'' = π² sin(πx) on [0,1], u(0)=u(1)=0
//! // Exact solution: u(x) = sin(πx)
//! let solver = IGASolver::solver_1d(3, 8).expect("solver");
//! let a = |_x: f64| 1.0_f64;
//! let f = |x: f64| {
//!     std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x).sin()
//! };
//! let sol = solver.solve(&a, &f, 0.0, 0.0).expect("solve");
//! let u_half = sol.eval(0.5);
//! ```

pub mod bspline;
pub mod iga_solver;
pub mod nurbs;

pub use bspline::{BSplineBasis, BSplineCurve, BSplineSurface};
pub use iga_solver::{
    IGASolution1D, IGASolution2D, IGASolver, IGASolver1D, IGASolver1DConfig, IGASolver2D,
};
pub use nurbs::{NurbsCurve, NurbsSurface};
