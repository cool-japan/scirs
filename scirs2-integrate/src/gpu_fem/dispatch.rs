//! Auto-dispatch for GPU/CPU FEM stiffness assembly.
//!
//! Selects the GPU path when the `gpu_fem` feature is enabled and the mesh is
//! large enough; falls back to the CPU Rayon-parallel path otherwise.

use crate::error::IntegrateError;
use crate::gpu_fem::stiffness::{
    assemble_stiffness_cpu, Element2D, MeshElement2D, StiffnessMatrix,
};
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the FEM assembly dispatcher.
#[derive(Debug, Clone)]
pub struct FemAssemblyConfig {
    /// Number of elements above which the GPU path is attempted (default 10 000).
    pub gpu_threshold: usize,
    /// Whether to attempt GPU acceleration at all.
    pub use_gpu: bool,
}

impl Default for FemAssemblyConfig {
    fn default() -> Self {
        FemAssemblyConfig {
            gpu_threshold: 10_000,
            use_gpu: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to the GPU FEM dispatch layer.
#[derive(Debug, Clone)]
pub enum GpuFemError {
    /// GPU device not available on this machine.
    GpuNotAvailable,
    /// GPU assembly failed with a diagnostic message.
    AssemblyFailed(String),
}

impl std::fmt::Display for GpuFemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuFemError::GpuNotAvailable => write!(f, "GPU not available"),
            GpuFemError::AssemblyFailed(msg) => write!(f, "GPU assembly failed: {msg}"),
        }
    }
}

impl std::error::Error for GpuFemError {}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-dispatch (Element2D path)
// ─────────────────────────────────────────────────────────────────────────────

/// Assemble the stiffness matrix using the best available backend.
///
/// When `config.use_gpu` is `true`, the mesh is larger than
/// `config.gpu_threshold`, and the `gpu_fem` feature is compiled in, the GPU
/// path is tried first.  On failure (or if conditions are not met) the
/// function falls back to the CPU Rayon-parallel path, which always succeeds.
pub fn assemble_stiffness_auto(
    elements: &[Element2D],
    d_matrix: &Array2<f64>,
    n_nodes: usize,
    config: &FemAssemblyConfig,
) -> Result<StiffnessMatrix, IntegrateError> {
    #[cfg(feature = "gpu_fem")]
    if config.use_gpu && elements.len() >= config.gpu_threshold {
        match crate::gpu_fem::wgpu_backend::assemble_stiffness_gpu(elements, d_matrix, n_nodes) {
            Ok(result) => return Ok(result),
            Err(_) => {
                // Fall through to CPU path
            }
        }
    }

    // CPU path is always available
    assemble_stiffness_cpu(elements, d_matrix, n_nodes)
}

/// Assemble from a mesh with explicit node connectivity.
pub fn assemble_stiffness_mesh_auto(
    mesh_elements: &[MeshElement2D],
    d_matrix: &Array2<f64>,
    n_nodes: usize,
    config: &FemAssemblyConfig,
) -> Result<StiffnessMatrix, IntegrateError> {
    #[cfg(feature = "gpu_fem")]
    if config.use_gpu && mesh_elements.len() >= config.gpu_threshold {
        // GPU path for mesh-structured assembly (stub — falls through)
        let _ = (&mesh_elements, n_nodes); // suppress unused warning in stub
    }

    crate::gpu_fem::stiffness::assemble_stiffness_mesh(mesh_elements, d_matrix, n_nodes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn isotropic_d() -> Array2<f64> {
        let e = 1.0_f64;
        let nu = 0.3_f64;
        let c = e / (1.0 - nu * nu);
        array![
            [c, c * nu, 0.0],
            [c * nu, c, 0.0],
            [0.0, 0.0, c * (1.0 - nu) / 2.0],
        ]
    }

    fn single_triangle() -> Vec<Element2D> {
        vec![Element2D {
            nodes: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            material_id: 0,
        }]
    }

    #[test]
    fn test_auto_dispatch_routes_to_cpu_below_threshold() {
        let d = isotropic_d();
        let elements = single_triangle();
        let config = FemAssemblyConfig {
            gpu_threshold: 10_000,
            use_gpu: true,
        };
        // With only 1 element, must use CPU path regardless
        let km = assemble_stiffness_auto(&elements, &d, 3, &config).expect("auto dispatch failed");
        assert!(!km.vals.is_empty());
    }

    #[test]
    fn test_auto_dispatch_use_gpu_false() {
        let d = isotropic_d();
        let elements = single_triangle();
        let config = FemAssemblyConfig {
            gpu_threshold: 0,
            use_gpu: false,
        };
        let km = assemble_stiffness_auto(&elements, &d, 3, &config).expect("auto dispatch failed");
        assert!(!km.vals.is_empty());
    }

    #[test]
    fn test_auto_dispatch_large_mesh_cpu_fallback() {
        // 10 elements, each treated as 3 independent nodes → n_nodes = 30
        let d = isotropic_d();
        let config = FemAssemblyConfig {
            gpu_threshold: 5,
            use_gpu: false, // force CPU even if above threshold
        };
        let n_elems = 10_usize;
        let elements: Vec<Element2D> = (0..n_elems)
            .map(|k| Element2D {
                nodes: [
                    [k as f64, 0.0],
                    [k as f64 + 1.0, 0.0],
                    [k as f64 + 0.5, 1.0],
                ],
                material_id: 0,
            })
            .collect();
        // n_nodes must be 3 * n_elems so every element's DOFs fit within [0, 2*n_nodes)
        let km = assemble_stiffness_auto(&elements, &d, 3 * n_elems, &config)
            .expect("auto dispatch failed");
        assert!(!km.vals.is_empty());
    }

    #[test]
    fn test_cpu_two_element_assembly_distinct_dofs() {
        // Confirm that two Element2D assemblies write to *different* DOF ranges.
        // Element 0 → DOFs [0..5], element 1 → DOFs [6..11].
        let d = isotropic_d();
        let config = FemAssemblyConfig {
            gpu_threshold: 10_000,
            use_gpu: false,
        };
        let elements = vec![
            Element2D {
                nodes: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                material_id: 0,
            },
            Element2D {
                nodes: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                material_id: 0,
            },
        ];
        // 2 elements × 3 nodes = 6 independent nodes → n_nodes = 6
        let km = assemble_stiffness_auto(&elements, &d, 6, &config)
            .expect("two-element assembly failed");
        let dense = km.to_dense();
        // Use view to avoid move into closures
        let view = dense.view();
        // Top-left 6×6 block (element 0 DOFs) must be non-zero
        let block0_norm: f64 = (0..6)
            .flat_map(|i| (0..6).map(move |j| (i, j)))
            .map(|(i, j)| view[[i, j]].abs())
            .sum();
        // Bottom-right 6×6 block (element 1 DOFs) must also be non-zero
        let block1_norm: f64 = (6..12)
            .flat_map(|i| (6..12).map(move |j| (i, j)))
            .map(|(i, j)| view[[i, j]].abs())
            .sum();
        assert!(block0_norm > 0.0, "Element 0 DOF block is zero");
        assert!(
            block1_norm > 0.0,
            "Element 1 DOF block is zero — DOFs were not distinct"
        );
        // Cross-block must be zero (no shared nodes between independent elements)
        let cross_norm: f64 = (0..6)
            .flat_map(|i| (6..12).map(move |j| (i, j)))
            .map(|(i, j)| view[[i, j]].abs())
            .sum();
        assert!(
            cross_norm < 1e-12,
            "Cross-block is non-zero — shared DOFs detected unexpectedly"
        );
    }
}
