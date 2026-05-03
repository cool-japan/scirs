//! Boundary Element Method (BEM) for boundary value problems.
//!
//! The BEM reformulates a PDE in the volume into an integral equation on the
//! boundary Γ alone, reducing the dimensionality of the problem by one. This
//! module provides:
//!
//! - **Kernels** — fundamental solutions (Green's functions) for Laplace,
//!   Helmholtz, and biharmonic equations.
//! - **Boundary mesh** — linear panel discretization of 2-D boundaries,
//!   including canonical constructors for circles and rectangles.
//! - **Panel method** — source-panel solver for potential flow problems.
//! - **BEM solver** — full Galerkin/collocation solver assembling the H and G
//!   matrices and solving with Dirichlet, Neumann, or mixed BCs.
//!
//! ## Boundary Integral Equation
//!
//! For the 2-D Laplace equation −∇²u = 0, the BIE at a boundary point x is:
//!
//! ```text
//! ½ u(x) + ∫_Γ ∂G/∂n(x,y) u(y) dΓ(y) = ∫_Γ G(x,y) q(y) dΓ(y)
//! ```
//!
//! where q = ∂u/∂n is the normal flux, G(x,y) = −1/(2π) ln|x−y|.
//!
//! In matrix form this is **H** u = **G** q.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::bem::{BEMSolver, BoundaryMesh, LaplaceKernel};
//!
//! // Circular boundary with 16 panels
//! let mesh = BoundaryMesh::circle([0.0, 0.0], 1.0, 16);
//! let solver = BEMSolver::new(mesh, LaplaceKernel, 4);
//!
//! // Dirichlet BC: u = 1 on boundary
//! let u_bc = vec![1.0_f64; 16];
//! // Solve for q = ∂u/∂n
//! let q = solver.solve_dirichlet(&u_bc).unwrap();
//! ```

pub mod boundary_mesh;
pub mod kernels;
pub mod panel_method;
pub mod solver;

pub use boundary_mesh::{BoundaryElement, BoundaryMesh};
pub use kernels::{BEMKernel, BiharmonicKernel, HelmholtzKernel, LaplaceKernel};
pub use panel_method::{PanelMethod, PanelMethodConfig};
pub use solver::BEMSolver;
