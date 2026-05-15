//! Linear triangular element stiffness matrix assembly.
//!
//! Implements the standard constant-strain triangle (CST) FEM stiffness
//! integral K_e = ∫ Bᵀ D B dΩ for 2-D linear triangular elements.
//!
//! # Reference
//!
//! Fish & Belytschko, "A First Course in Finite Elements" (2007), Chapter 9.

use crate::error::IntegrateError;
use scirs2_core::ndarray::Array2;
use scirs2_core::parallel_ops::*;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single linear triangular element with three nodes in 2-D.
///
/// `nodes[i] = [x, y]` for node *i* (0, 1, 2).
#[derive(Debug, Clone)]
pub struct Element2D {
    /// Node coordinates: `nodes[i] = [x, y]`
    pub nodes: [[f64; 2]; 3],
    /// Material identifier (indexes into a material lookup table)
    pub material_id: usize,
}

/// Sparse stiffness matrix in COO (coordinate) format.
///
/// Use [`StiffnessMatrix::to_dense`] for small problems or convert to CSR via
/// [`StiffnessMatrix::to_dense`] for dense conversion.
#[derive(Debug, Clone)]
pub struct StiffnessMatrix {
    /// Row indices (same length as `vals`)
    pub rows: Vec<usize>,
    /// Column indices (same length as `vals`)
    pub cols: Vec<usize>,
    /// Non-zero values
    pub vals: Vec<f64>,
    /// Matrix dimension (n × n, where n = number of DOF)
    pub size: usize,
}

impl StiffnessMatrix {
    /// Create an empty stiffness matrix of dimension `size × size`.
    pub fn empty(size: usize) -> Self {
        StiffnessMatrix {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
            size,
        }
    }

    /// Number of stored (possibly duplicate) entries.
    pub fn nnz_raw(&self) -> usize {
        self.vals.len()
    }

    /// Convert to a dense `n × n` matrix (sums duplicate entries).
    pub fn to_dense(&self) -> Array2<f64> {
        let mut mat = Array2::<f64>::zeros((self.size, self.size));
        for ((r, c), v) in self.rows.iter().zip(self.cols.iter()).zip(self.vals.iter()) {
            mat[[*r, *c]] += v;
        }
        mat
    }

    /// Merge another COO matrix into this one (concatenate triplets).
    fn merge(&mut self, other: StiffnessMatrix) {
        self.rows.extend(other.rows);
        self.cols.extend(other.cols);
        self.vals.extend(other.vals);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-level stiffness computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 6×6 element stiffness matrix for a single CST element.
///
/// Returns `Err` if the element is degenerate (zero or negative area).
pub fn element_stiffness(
    elem: &Element2D,
    d_matrix: &Array2<f64>,
) -> Result<[[f64; 6]; 6], IntegrateError> {
    let [x1, y1] = elem.nodes[0];
    let [x2, y2] = elem.nodes[1];
    let [x3, y3] = elem.nodes[2];

    // Twice the area (signed)
    let two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    if two_area.abs() < 1e-15 {
        return Err(IntegrateError::ValueError(
            "Degenerate triangle: zero or near-zero area".to_string(),
        ));
    }
    let inv_two_area = 1.0 / two_area;

    // Shape function gradients (constant over CST element)
    // ∂N1/∂x = (y2 - y3) / (2A)   ∂N1/∂y = (x3 - x2) / (2A)
    // ∂N2/∂x = (y3 - y1) / (2A)   ∂N2/∂y = (x1 - x3) / (2A)
    // ∂N3/∂x = (y1 - y2) / (2A)   ∂N3/∂y = (x2 - x1) / (2A)
    let b1x = (y2 - y3) * inv_two_area;
    let b1y = (x3 - x2) * inv_two_area;
    let b2x = (y3 - y1) * inv_two_area;
    let b2y = (x1 - x3) * inv_two_area;
    let b3x = (y1 - y2) * inv_two_area;
    let b3y = (x2 - x1) * inv_two_area;

    // B matrix (3×6): strain-displacement matrix for plane-stress/strain
    // ε = B u_e,  ε = [εxx, εyy, γxy]ᵀ
    // Columns: [u1, v1, u2, v2, u3, v3]
    let b = [
        [b1x, 0.0, b2x, 0.0, b3x, 0.0],
        [0.0, b1y, 0.0, b2y, 0.0, b3y],
        [b1y, b1x, b2y, b2x, b3y, b3x],
    ];

    // K_e = Bᵀ D B * area   (CST: constant B, so ∫dΩ = A = two_area/2)
    let area = two_area.abs() * 0.5;

    // D (3×3 material matrix)
    let d = [
        [d_matrix[[0, 0]], d_matrix[[0, 1]], d_matrix[[0, 2]]],
        [d_matrix[[1, 0]], d_matrix[[1, 1]], d_matrix[[1, 2]]],
        [d_matrix[[2, 0]], d_matrix[[2, 1]], d_matrix[[2, 2]]],
    ];

    // Compute D*B (3×6)
    let mut db = [[0.0_f64; 6]; 3];
    for i in 0..3 {
        for j in 0..6 {
            for k in 0..3 {
                db[i][j] += d[i][k] * b[k][j];
            }
        }
    }

    // Compute Bᵀ * D * B (6×6)
    let mut ke = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..3 {
                ke[i][j] += b[k][i] * db[k][j];
            }
        }
    }

    // Scale by element area
    for i in 0..6 {
        for j in 0..6 {
            ke[i][j] *= area;
        }
    }

    Ok(ke)
}

// ─────────────────────────────────────────────────────────────────────────────
// Global assembly (serial)
// ─────────────────────────────────────────────────────────────────────────────

/// Assemble the global stiffness matrix in COO format — serial path.
///
/// DOF ordering: element *k* contributes nodes `[3k, 3k+1, 3k+2]`, giving DOFs
/// `[6k, 6k+1, 6k+2, 6k+3, 6k+4, 6k+5]`.  Each element is treated as having
/// three *independent* nodes; if multiple elements share physical nodes the caller
/// should deduplicate using [`assemble_stiffness_mesh`] instead.
///
/// Returns a COO matrix of size `(2 * n_nodes) × (2 * n_nodes)`.
pub fn assemble_stiffness_cpu(
    elements: &[Element2D],
    d_matrix: &Array2<f64>,
    n_nodes: usize,
) -> Result<StiffnessMatrix, IntegrateError> {
    if d_matrix.shape() != [3, 3] {
        return Err(IntegrateError::DimensionMismatch(format!(
            "D matrix must be 3×3, got {:?}",
            d_matrix.shape()
        )));
    }
    // Each element contributes 3 independent nodes (DOFs 6k..6k+5 for element k).
    // If n_nodes < 3 * elements.len(), higher-index elements would be silently dropped.
    if n_nodes < 3 * elements.len() {
        return Err(IntegrateError::DimensionMismatch(format!(
            "assemble_stiffness_cpu requires n_nodes >= 3 * elements.len() \
             ({n_nodes} < {}); each element uses 3 independent nodes. \
             For shared nodes use assemble_stiffness_mesh.",
            3 * elements.len()
        )));
    }

    let n_dof = 2 * n_nodes;
    let n_threads = num_threads().max(1);
    let chunk_size = (elements.len() / n_threads).max(1);

    // Build chunks paired with their starting element index so each element
    // can derive its unique global DOF range: element k → nodes [3k, 3k+1, 3k+2].
    let chunks: Vec<(usize, &[Element2D])> = elements
        .chunks(chunk_size)
        .scan(0_usize, |offset, chunk| {
            let start = *offset;
            *offset += chunk.len();
            Some((start, chunk))
        })
        .collect();

    // Parallel assembly: each chunk builds a local COO, results collected
    let locals: Vec<StiffnessMatrix> = parallel_map_result(&chunks, |(start_idx, chunk)| {
        assemble_chunk(*start_idx, chunk, d_matrix, n_dof)
    })?;

    // Merge all local COO matrices
    let mut global = StiffnessMatrix::empty(n_dof);
    for local in locals {
        global.merge(local);
    }
    Ok(global)
}

/// Assemble a chunk of `Element2D` elements into a local COO matrix.
///
/// `elem_offset` is the index of the first element in this chunk within the
/// global element array.  Element at position `elem_offset + i` is assigned
/// nodes `[3*(elem_offset+i), 3*(elem_offset+i)+1, 3*(elem_offset+i)+2]`.
fn assemble_chunk(
    elem_offset: usize,
    elements: &[Element2D],
    d_matrix: &Array2<f64>,
    n_dof: usize,
) -> Result<StiffnessMatrix, IntegrateError> {
    let mut local = StiffnessMatrix::empty(n_dof);
    for (i, elem) in elements.iter().enumerate() {
        let ke = element_stiffness(elem, d_matrix)?;
        // Each element k = elem_offset + i gets three unique node IDs.
        // Node connectivity: element k → nodes [3k, 3k+1, 3k+2]
        //   → DOFs [6k, 6k+1, 6k+2, 6k+3, 6k+4, 6k+5]
        // Use assemble_stiffness_mesh for shared-node (connected mesh) problems.
        let k = elem_offset + i;
        let dofs: [usize; 6] = [6 * k, 6 * k + 1, 6 * k + 2, 6 * k + 3, 6 * k + 4, 6 * k + 5];
        for (li, &gi) in dofs.iter().enumerate() {
            for (lj, &gj) in dofs.iter().enumerate() {
                if gi < n_dof && gj < n_dof {
                    local.rows.push(gi);
                    local.cols.push(gj);
                    local.vals.push(ke[li][lj]);
                }
            }
        }
    }
    Ok(local)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh-level assembly with explicit connectivity
// ─────────────────────────────────────────────────────────────────────────────

/// Element with explicit global node indices for mesh assembly.
#[derive(Debug, Clone)]
pub struct MeshElement2D {
    /// Global node indices [n0, n1, n2]
    pub node_ids: [usize; 3],
    /// Node coordinates \[\[x0,y0\], \[x1,y1\], \[x2,y2\]\]
    pub nodes: [[f64; 2]; 3],
    /// Material identifier
    pub material_id: usize,
}

impl MeshElement2D {
    /// Convert to `Element2D` (dropping node connectivity).
    pub fn to_element2d(&self) -> Element2D {
        Element2D {
            nodes: self.nodes,
            material_id: self.material_id,
        }
    }
}

/// Assemble global stiffness matrix from a mesh with explicit connectivity.
///
/// This is the production-quality entry point for FEM assembly.
pub fn assemble_stiffness_mesh(
    mesh_elements: &[MeshElement2D],
    d_matrix: &Array2<f64>,
    n_nodes: usize,
) -> Result<StiffnessMatrix, IntegrateError> {
    if d_matrix.shape() != [3, 3] {
        return Err(IntegrateError::DimensionMismatch(format!(
            "D matrix must be 3×3, got {:?}",
            d_matrix.shape()
        )));
    }

    let n_dof = 2 * n_nodes;
    let chunk_size = (mesh_elements.len() / num_threads().max(1)).max(1);

    let chunks: Vec<&[MeshElement2D]> = mesh_elements.chunks(chunk_size).collect();

    let locals: Vec<StiffnessMatrix> =
        parallel_map_result(&chunks, |chunk| assemble_mesh_chunk(chunk, d_matrix, n_dof))?;

    let mut global = StiffnessMatrix::empty(n_dof);
    for local in locals {
        global.merge(local);
    }
    Ok(global)
}

fn assemble_mesh_chunk(
    chunk: &[MeshElement2D],
    d_matrix: &Array2<f64>,
    n_dof: usize,
) -> Result<StiffnessMatrix, IntegrateError> {
    let mut local = StiffnessMatrix::empty(n_dof);
    for me in chunk {
        let elem = me.to_element2d();
        let ke = element_stiffness(&elem, d_matrix)?;
        // Global DOF mapping: node k → DOFs [2k, 2k+1]
        let gdofs: [usize; 6] = [
            2 * me.node_ids[0],
            2 * me.node_ids[0] + 1,
            2 * me.node_ids[1],
            2 * me.node_ids[1] + 1,
            2 * me.node_ids[2],
            2 * me.node_ids[2] + 1,
        ];
        for (li, &gi) in gdofs.iter().enumerate() {
            for (lj, &gj) in gdofs.iter().enumerate() {
                if gi < n_dof && gj < n_dof {
                    local.rows.push(gi);
                    local.cols.push(gj);
                    local.vals.push(ke[li][lj]);
                }
            }
        }
    }
    Ok(local)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Isotropic plane-stress D matrix for E=1, ν=0.3
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

    #[test]
    fn test_element_stiffness_matrix_2d() {
        // Right triangle with nodes at (0,0), (1,0), (0,1)
        let elem = Element2D {
            nodes: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            material_id: 0,
        };
        let d = isotropic_d();
        let ke = element_stiffness(&elem, &d).expect("stiffness computation failed");

        // K_e must be 6×6 and symmetric
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (ke[i][j] - ke[j][i]).abs() < 1e-10,
                    "K_e not symmetric at [{i},{j}]: ke[i][j]={} ke[j][i]={}",
                    ke[i][j],
                    ke[j][i]
                );
            }
        }
    }

    #[test]
    fn test_element_stiffness_positive_semidefinite() {
        let elem = Element2D {
            nodes: [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            material_id: 0,
        };
        let d = isotropic_d();
        let ke = element_stiffness(&elem, &d).expect("stiffness computation failed");

        // All diagonal entries of a positive semi-definite matrix are ≥ 0
        for i in 0..6 {
            assert!(
                ke[i][i] >= -1e-12,
                "Negative diagonal at [{i}]: {}",
                ke[i][i]
            );
        }

        // Frobenius norm must be positive
        let frob: f64 = ke
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(frob > 0.0, "Stiffness matrix is all-zero");
    }

    #[test]
    fn test_cpu_assembly_mesh_of_10_elements() {
        let d = isotropic_d();
        // Simple 2×5 grid of triangles (4 nodes × 6 cols = 24 nodes)
        // Build a small mesh manually
        let mut elements = Vec::new();
        let n_nodes = 12_usize;
        for k in 0..10_usize {
            let x0 = (k % 5) as f64;
            let y0 = (k / 5) as f64;
            elements.push(MeshElement2D {
                node_ids: [k % n_nodes, (k + 1) % n_nodes, (k + 3) % n_nodes],
                nodes: [[x0, y0], [x0 + 1.0, y0], [x0 + 0.5, y0 + 1.0]],
                material_id: 0,
            });
        }
        let km = assemble_stiffness_mesh(&elements, &d, n_nodes).expect("mesh assembly failed");
        assert!(!km.vals.is_empty(), "No entries assembled");
        for (&r, &c) in km.rows.iter().zip(km.cols.iter()) {
            assert!(r < 2 * n_nodes, "Row index out of bounds: {r}");
            assert!(c < 2 * n_nodes, "Col index out of bounds: {c}");
        }
    }

    #[test]
    fn test_stiffness_matrix_symmetry_via_dense() {
        let d = isotropic_d();
        let elements = vec![
            MeshElement2D {
                node_ids: [0, 1, 2],
                nodes: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                material_id: 0,
            },
            MeshElement2D {
                node_ids: [1, 3, 2],
                nodes: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                material_id: 0,
            },
        ];
        let km = assemble_stiffness_mesh(&elements, &d, 4).expect("assembly failed");
        let dense = km.to_dense();
        let n = dense.shape()[0];
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dense[[i, j]] - dense[[j, i]]).abs() < 1e-10,
                    "Global K not symmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_degenerate_element_returns_error() {
        // Collinear nodes → zero area
        let elem = Element2D {
            nodes: [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            material_id: 0,
        };
        let d = isotropic_d();
        assert!(element_stiffness(&elem, &d).is_err());
    }
}
