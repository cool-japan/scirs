//! GPU-accelerated FEM stiffness matrix assembly dispatch.
//!
//! # Overview
//!
//! The CPU path (always available) uses Rayon work-stealing parallel assembly
//! of the sparse stiffness matrix for linear triangular (CST) elements.
//!
//! The GPU path (feature `gpu_fem`) provides a wgpu WGSL compute-shader backend
//! for shared-memory atomic scatter of the global stiffness matrix.  On any
//! machine where the GPU is unavailable or the element count is below the
//! threshold, the CPU path is used automatically.
//!
//! # Quick Example
//!
//! ```rust
//! use scirs2_integrate::gpu_fem::{
//!     FemAssemblyConfig, assemble_stiffness_auto,
//!     stiffness::{Element2D, MeshElement2D, assemble_stiffness_mesh},
//! };
//! use scirs2_core::ndarray::array;
//!
//! let d = {
//!     let e = 1.0_f64;
//!     let nu = 0.3_f64;
//!     let c = e / (1.0 - nu * nu);
//!     array![[c, c*nu, 0.0], [c*nu, c, 0.0], [0.0, 0.0, c*(1.0-nu)/2.0]]
//! };
//! let elements = vec![
//!     Element2D { nodes: [[0.0,0.0],[1.0,0.0],[0.0,1.0]], material_id: 0 },
//! ];
//! let config = FemAssemblyConfig::default();
//! let km = assemble_stiffness_auto(&elements, &d, 3, &config).unwrap();
//! assert!(!km.vals.is_empty());
//! ```

pub mod dispatch;
pub mod stiffness;

// GPU backend is only compiled when the `gpu_fem` feature is enabled.
// The stub file must exist for rustc to process the cfg-gated declaration.
#[cfg(feature = "gpu_fem")]
pub mod wgpu_backend;

// Re-exports
pub use dispatch::{
    assemble_stiffness_auto, assemble_stiffness_mesh_auto, FemAssemblyConfig, GpuFemError,
};
pub use stiffness::{
    assemble_stiffness_cpu, assemble_stiffness_mesh, element_stiffness, Element2D, MeshElement2D,
    StiffnessMatrix,
};
