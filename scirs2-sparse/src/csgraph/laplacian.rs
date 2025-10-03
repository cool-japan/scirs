//! Laplacian matrix computation for sparse graphs
//!
//! This module provides functions to compute various types of Laplacian matrices
//! from sparse graphs, including the standard Laplacian, normalized Laplacian,
//! and other variants used in spectral graph theory.

use super::{num_vertices, validate_graph};
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Laplacian matrix types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaplacianType {
    /// Standard Laplacian: L = D - A
    Standard,
    /// Normalized Laplacian: L = D^(-1/2) * (D - A) * D^(-1/2)
    Normalized,
    /// Random walk Laplacian: L = D^(-1) * (D - A)
    RandomWalk,
}

impl LaplacianType {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SparseResult<Self> {
        match s.to_lowercase().as_str() {
            "standard" | "unnormalized" => Ok(Self::Standard),
            "normalized" | "symmetric" => Ok(Self::Normalized),
            "random_walk" | "random-walk" | "randomwalk" => Ok(Self::RandomWalk),
            _ => Err(SparseError::ValueError(format!(
                "Unknown Laplacian type: {s}. Use 'standard', 'normalized', or 'random_walk'"
            ))),
        }
    }
}

/// Compute the Laplacian matrix of a graph
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `normed` - Whether to compute the normalized Laplacian
/// * `return_diag` - Whether to return the diagonal degree matrix
/// * `use_outdegree` - For directed graphs, whether to use out-degree (default) or in-degree
///
/// # Returns
///
/// A tuple containing:
/// - The Laplacian matrix
/// - Optional diagonal degree array (if requested)
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::laplacian;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a simple graph
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// // Compute standard Laplacian
/// let (laplacian, _) = laplacian(&graph, false, false, true).unwrap();
/// ```
#[allow(dead_code)]
pub fn laplacian<T, S>(
    graph: &S,
    normed: bool,
    return_diag: bool,
    use_outdegree: bool,
) -> SparseResult<(CsrArray<T>, Option<Array1<T>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    validate_graph(graph, true)?; // Allow both directed and undirected

    let laplaciantype = if normed {
        LaplacianType::Normalized
    } else {
        LaplacianType::Standard
    };

    compute_laplacianmatrix(graph, laplaciantype, return_diag, use_outdegree)
}

/// Compute a specific type of Laplacian matrix
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `laplaciantype` - Type of Laplacian to compute
/// * `return_diag` - Whether to return the diagonal degree matrix
/// * `use_outdegree` - For directed graphs, whether to use out-degree
///
/// # Returns
///
/// A tuple containing the Laplacian matrix and optional degree array
#[allow(dead_code)]
pub fn compute_laplacianmatrix<T, S>(
    graph: &S,
    laplaciantype: LaplacianType,
    return_diag: bool,
    use_outdegree: bool,
) -> SparseResult<(CsrArray<T>, Option<Array1<T>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);

    // Compute degrees
    let degrees = if use_outdegree {
        compute_out_degrees(graph)?
    } else {
        compute_in_degrees(graph)?
    };

    // Get the graph structure
    let (row_indices, col_indices, values) = graph.find();

    match laplaciantype {
        LaplacianType::Standard => compute_standard_laplacian(
            row_indices.as_slice().unwrap(),
            col_indices.as_slice().unwrap(),
            values.as_slice().unwrap(),
            &degrees,
            n,
        ),
        LaplacianType::Normalized => compute_normalized_laplacian(
            row_indices.as_slice().unwrap(),
            col_indices.as_slice().unwrap(),
            values.as_slice().unwrap(),
            &degrees,
            n,
        ),
        LaplacianType::RandomWalk => compute_random_walk_laplacian(
            row_indices.as_slice().unwrap(),
            col_indices.as_slice().unwrap(),
            values.as_slice().unwrap(),
            &degrees,
            n,
        ),
    }
    .map(|laplacian| {
        let diag = if return_diag { Some(degrees) } else { None };
        (laplacian, diag)
    })
}

/// Compute out-degrees for all vertices
#[allow(dead_code)]
fn compute_out_degrees<T, S>(graph: &S) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let mut degrees = Array1::zeros(n);

    let (row_indices, _, values) = graph.find();

    for (i, &row) in row_indices.iter().enumerate() {
        degrees[row] = degrees[row] + values[i];
    }

    Ok(degrees)
}

/// Compute in-degrees for all vertices
#[allow(dead_code)]
fn compute_in_degrees<T, S>(graph: &S) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let mut degrees = Array1::zeros(n);

    let (_, col_indices, values) = graph.find();

    for (i, &col) in col_indices.iter().enumerate() {
        degrees[col] = degrees[col] + values[i];
    }

    Ok(degrees)
}

/// Compute the standard Laplacian matrix: L = D - A
#[allow(dead_code)]
fn compute_standard_laplacian<T>(
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[T],
    degrees: &Array1<T>,
    n: usize,
) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy + 'static,
{
    let mut laplacianrows = Vec::new();
    let mut laplaciancols = Vec::new();
    let mut laplacianvalues = Vec::new();

    // Add diagonal elements (degrees)
    for (i, &degree) in degrees.iter().enumerate() {
        if !degree.is_zero() {
            laplacianrows.push(i);
            laplaciancols.push(i);
            laplacianvalues.push(degree);
        }
    }

    // Subtract off-diagonal elements (negative adjacency matrix entries)
    for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if row != col && !values[i].is_zero() {
            laplacianrows.push(row);
            laplaciancols.push(col);
            laplacianvalues.push(-values[i]);
        }
    }

    CsrArray::from_triplets(
        &laplacianrows,
        &laplaciancols,
        &laplacianvalues,
        (n, n),
        false, // Triplets are not sorted
    )
}

/// Compute the normalized Laplacian matrix: L = D^(-1/2) * (D - A) * D^(-1/2)
#[allow(dead_code)]
fn compute_normalized_laplacian<T>(
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[T],
    degrees: &Array1<T>,
    n: usize,
) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy + 'static,
{
    // Compute D^(-1/2)
    let mut sqrt_inv_degrees = Array1::zeros(n);
    for (i, &degree) in degrees.iter().enumerate() {
        if degree > T::zero() {
            sqrt_inv_degrees[i] = T::one() / degree.sqrt();
        }
    }

    let mut laplacianrows = Vec::new();
    let mut laplaciancols = Vec::new();
    let mut laplacianvalues = Vec::new();

    // Add diagonal elements (1 for vertices with positive degree, 0 for isolated vertices)
    for (i, &degree) in degrees.iter().enumerate() {
        if degree > T::zero() {
            laplacianrows.push(i);
            laplaciancols.push(i);
            laplacianvalues.push(T::one());
        }
    }

    // Add off-diagonal elements: -A_{ij} / sqrt(d_i * d_j)
    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if i != j && !values[k].is_zero() {
            let normalization = sqrt_inv_degrees[i] * sqrt_inv_degrees[j];
            if !normalization.is_zero() {
                laplacianrows.push(i);
                laplaciancols.push(j);
                laplacianvalues.push(-values[k] * normalization);
            }
        }
    }

    CsrArray::from_triplets(
        &laplacianrows,
        &laplaciancols,
        &laplacianvalues,
        (n, n),
        false, // Triplets are not sorted
    )
}

/// Compute the random walk Laplacian matrix: L = D^(-1) * (D - A)
#[allow(dead_code)]
fn compute_random_walk_laplacian<T>(
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[T],
    degrees: &Array1<T>,
    n: usize,
) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy + 'static,
{
    // Compute D^(-1)
    let mut inv_degrees = Array1::zeros(n);
    for (i, &degree) in degrees.iter().enumerate() {
        if degree > T::zero() {
            inv_degrees[i] = T::one() / degree;
        }
    }

    let mut laplacianrows = Vec::new();
    let mut laplaciancols = Vec::new();
    let mut laplacianvalues = Vec::new();

    // Add diagonal elements (1 for vertices with positive degree, 0 for isolated vertices)
    for (i, &degree) in degrees.iter().enumerate() {
        if degree > T::zero() {
            laplacianrows.push(i);
            laplaciancols.push(i);
            laplacianvalues.push(T::one());
        }
    }

    // Add off-diagonal elements: -A_{ij} / d_i
    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if i != j && !values[k].is_zero() && inv_degrees[i] > T::zero() {
            laplacianrows.push(i);
            laplaciancols.push(j);
            laplacianvalues.push(-values[k] * inv_degrees[i]);
        }
    }

    CsrArray::from_triplets(
        &laplacianrows,
        &laplaciancols,
        &laplacianvalues,
        (n, n),
        false, // Triplets are not sorted
    )
}

/// Compute the degree matrix of a graph
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `use_outdegree` - Whether to use out-degree (true) or in-degree (false)
///
/// # Returns
///
/// The degree array
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::degree_matrix;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let degrees = degree_matrix(&graph, true).unwrap();
/// ```
#[allow(dead_code)]
pub fn degree_matrix<T, S>(graph: &S, use_outdegree: bool) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    if use_outdegree {
        compute_out_degrees(graph)
    } else {
        compute_in_degrees(graph)
    }
}

/// Compute the algebraic connectivity (Fiedler value) of a graph
///
/// This is the second smallest eigenvalue of the Laplacian matrix.
/// For connected graphs, this value is positive and measures how well connected the graph is.
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `normalized` - Whether to use the normalized Laplacian
///
/// # Returns
///
/// The algebraic connectivity value
///
/// # Note
///
/// This function computes the second smallest eigenvalue of the Laplacian matrix
/// using sparse eigenvalue solvers. The smallest eigenvalue is always 0 for
/// connected graphs, so we find the k=2 smallest eigenvalues and return the second one.
#[allow(dead_code)]
pub fn algebraic_connectivity<T, S>(graph: &S, normalized: bool) -> SparseResult<T>
where
    T: Float
        + Debug
        + Copy
        + 'static
        + std::iter::Sum
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync,
    S: SparseArray<T>,
{
    use crate::linalg::{lanczos, LanczosOptions};
    use crate::sym_csr::SymCsrMatrix;

    validate_graph(graph, false)?; // Ensure it's a valid undirected graph

    // Compute the Laplacian matrix
    let (laplacian, _) = compute_laplacianmatrix(
        graph,
        if normalized {
            LaplacianType::Normalized
        } else {
            LaplacianType::Standard
        },
        false,
        true,
    )?;

    // Convert the Laplacian to symmetric CSR format
    // For undirected graphs, the Laplacian is already symmetric
    let (rows, cols) = laplacian.shape();
    let (row_indices, col_indices, values) = laplacian.find();

    // Extract lower triangular part for symmetric storage
    let mut sym_indices = Vec::new();
    let mut sym_data = Vec::new();
    let mut sym_indptr = vec![0; rows + 1];

    // Count non-zeros in lower triangle for each row
    let mut nnz_count = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..rows {
        sym_indptr[i] = nnz_count;
        for k in 0..row_indices.len() {
            let row = row_indices[k];
            let col = col_indices[k];
            if row == i && col <= i {
                // Include diagonal and lower triangular elements
                sym_indices.push(col);
                sym_data.push(values[k]);
                nnz_count += 1;
            }
        }
    }
    sym_indptr[rows] = nnz_count;

    // Create symmetric CSR matrix
    let sym_laplacian = SymCsrMatrix::new(sym_data, sym_indices, sym_indptr, (rows, cols))?;

    // Configure Lanczos options to find the 3 smallest eigenvalues
    // We need 3 to be safe in case the smallest is not exactly zero
    let options = LanczosOptions {
        max_iter: 1000,
        max_subspace_size: (rows / 4).clamp(10, 50), // Reasonable subspace size
        tol: 1e-12,
        numeigenvalues: 3.min(rows), // Find 3 smallest eigenvalues (or fewer if matrix is small)
        compute_eigenvectors: false, // We only need eigenvalues
    };

    // Use Lanczos algorithm to find smallest eigenvalues
    let eigen_result = lanczos(&sym_laplacian, &options, None)?;

    if !eigen_result.converged {
        return Err(SparseError::ValueError(
            "Eigenvalue computation did not converge".to_string(),
        ));
    }

    let eigenvalues = eigen_result.eigenvalues;

    // Find the second smallest eigenvalue
    // Note: Eigenvalues from Lanczos are typically in ascending order for smallest eigenvalues
    if eigenvalues.len() < 2 {
        return Err(SparseError::ValueError(
            "Not enough eigenvalues computed to determine algebraic connectivity".to_string(),
        ));
    }

    // Sort eigenvalues to ensure we get the correct ordering
    let mut sorted_eigenvals: Vec<T> = eigenvalues.iter().copied().collect();
    sorted_eigenvals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // The algebraic connectivity is the second smallest eigenvalue
    // The smallest should be approximately zero for connected graphs
    let algebraic_conn = sorted_eigenvals[1];

    // Verify that the smallest eigenvalue is close to zero (sanity check)
    let smallest_eigenval = sorted_eigenvals[0];
    let zero_threshold = T::from(1e-10).unwrap();

    if smallest_eigenval.abs() > zero_threshold {
        return Err(SparseError::ValueError(format!(
            "Graph may not be connected: smallest eigenvalue is {:.2e}, expected ~0",
            smallest_eigenval.to_f64().unwrap_or(0.0)
        )));
    }

    Ok(algebraic_conn)
}

/// Check if a matrix is a valid Laplacian matrix
///
/// A matrix L is a valid Laplacian if:
/// 1. L is symmetric (for undirected graphs)
/// 2. L has zero row sums
/// 3. L has non-positive off-diagonal entries
/// 4. L is positive semidefinite
///
/// # Arguments
///
/// * `matrix` - The matrix to check
/// * `tol` - Tolerance for numerical checks
///
/// # Returns
///
/// True if the matrix is a valid Laplacian, false otherwise
#[allow(dead_code)]
pub fn is_laplacian<T, S>(matrix: &S, tol: T) -> SparseResult<bool>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (n, m) = matrix.shape();

    // Must be square
    if n != m {
        return Ok(false);
    }

    let (row_indices, col_indices, values) = matrix.find();

    // Check row sums are approximately zero
    let mut row_sums = vec![T::zero(); n];
    for (i, (&row, &_col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        row_sums[row] = row_sums[row] + values[i];
    }

    for &sum in &row_sums {
        if sum.abs() > tol {
            return Ok(false);
        }
    }

    // Check off-diagonal entries are non-positive
    for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if row != col && values[i] > tol {
            return Ok(false);
        }
    }

    // For a complete check, we would also verify positive semidefiniteness,
    // but that requires eigenvalue computation which is expensive

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_simple_graph() -> CsrArray<f64> {
        // Create a simple triangle graph:
        //   0 -- 1
        //   |  / |
        //   | /  |
        //   2 ---+
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![1, 2, 0, 2, 0, 1];
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap()
    }

    #[test]
    fn test_compute_degrees() {
        let graph = create_simple_graph();

        let out_degrees = compute_out_degrees(&graph).unwrap();
        let in_degrees = compute_in_degrees(&graph).unwrap();

        // For undirected graphs, out-degrees and in-degrees should be the same
        assert_relative_eq!(out_degrees[0], 2.0); // Connected to 1 and 2
        assert_relative_eq!(out_degrees[1], 2.0); // Connected to 0 and 2
        assert_relative_eq!(out_degrees[2], 2.0); // Connected to 0 and 1

        for i in 0..3 {
            assert_relative_eq!(out_degrees[i], in_degrees[i]);
        }
    }

    #[test]
    fn test_standard_laplacian() {
        let graph = create_simple_graph();
        let (laplacian, degrees) =
            compute_laplacianmatrix(&graph, LaplacianType::Standard, true, true).unwrap();

        let degrees = degrees.unwrap();

        // Check degrees
        assert_relative_eq!(degrees[0], 2.0);
        assert_relative_eq!(degrees[1], 2.0);
        assert_relative_eq!(degrees[2], 2.0);

        // Check Laplacian matrix properties
        // Diagonal elements should equal degrees
        assert_relative_eq!(laplacian.get(0, 0), 2.0);
        assert_relative_eq!(laplacian.get(1, 1), 2.0);
        assert_relative_eq!(laplacian.get(2, 2), 2.0);

        // Off-diagonal elements should be negative adjacency
        assert_relative_eq!(laplacian.get(0, 1), -1.0);
        assert_relative_eq!(laplacian.get(0, 2), -1.0);
        assert_relative_eq!(laplacian.get(1, 0), -1.0);
        assert_relative_eq!(laplacian.get(1, 2), -1.0);
        assert_relative_eq!(laplacian.get(2, 0), -1.0);
        assert_relative_eq!(laplacian.get(2, 1), -1.0);
    }

    #[test]
    fn test_normalized_laplacian() {
        let graph = create_simple_graph();
        let (laplacian, _) =
            compute_laplacianmatrix(&graph, LaplacianType::Normalized, false, true).unwrap();

        // Check diagonal elements are 1
        assert_relative_eq!(laplacian.get(0, 0), 1.0);
        assert_relative_eq!(laplacian.get(1, 1), 1.0);
        assert_relative_eq!(laplacian.get(2, 2), 1.0);

        // Check off-diagonal normalization: -1/sqrt(d_i * d_j)
        // For this graph, all degrees are 2, so normalization factor is 1/sqrt(2*2) = 1/2
        assert_relative_eq!(laplacian.get(0, 1), -0.5);
        assert_relative_eq!(laplacian.get(1, 2), -0.5);
        assert_relative_eq!(laplacian.get(2, 0), -0.5);
    }

    #[test]
    fn test_random_walk_laplacian() {
        let graph = create_simple_graph();
        let (laplacian, _) =
            compute_laplacianmatrix(&graph, LaplacianType::RandomWalk, false, true).unwrap();

        // Check diagonal elements are 1
        assert_relative_eq!(laplacian.get(0, 0), 1.0);
        assert_relative_eq!(laplacian.get(1, 1), 1.0);
        assert_relative_eq!(laplacian.get(2, 2), 1.0);

        // Check off-diagonal normalization: -A_{ij}/d_i
        // For this graph, all degrees are 2, so normalization factor is 1/2
        assert_relative_eq!(laplacian.get(0, 1), -0.5);
        assert_relative_eq!(laplacian.get(1, 2), -0.5);
        assert_relative_eq!(laplacian.get(2, 0), -0.5);
    }

    #[test]
    fn test_laplacianapi() {
        let graph = create_simple_graph();

        // Test standard Laplacian
        let (lap_std_, _) = laplacian(&graph, false, false, true).unwrap();
        assert_relative_eq!(lap_std_.get(0, 0), 2.0);

        // Test normalized Laplacian
        let (lap_norm, degrees) = laplacian(&graph, true, true, true).unwrap();
        assert_relative_eq!(lap_norm.get(0, 0), 1.0);

        let degrees = degrees.unwrap();
        assert_relative_eq!(degrees[0], 2.0);
    }

    #[test]
    fn test_degree_matrix() {
        let graph = create_simple_graph();

        let degrees = degree_matrix(&graph, true).unwrap();
        assert_relative_eq!(degrees[0], 2.0);
        assert_relative_eq!(degrees[1], 2.0);
        assert_relative_eq!(degrees[2], 2.0);
    }

    #[test]
    fn test_laplacianrow_sums() {
        let graph = create_simple_graph();
        let (laplacian, _) =
            compute_laplacianmatrix(&graph, LaplacianType::Standard, false, true).unwrap();

        // Row sums of Laplacian should be zero
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                row_sum += laplacian.get(i, j);
            }
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_is_laplacian() {
        let graph = create_simple_graph();
        let (laplacian, _) =
            compute_laplacianmatrix(&graph, LaplacianType::Standard, false, true).unwrap();

        // Our computed Laplacian should pass the validation
        assert!(is_laplacian(&laplacian, 1e-10).unwrap());
    }

    #[test]
    fn test_isolated_vertex() {
        // Create a graph with an isolated vertex
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let data = vec![1.0, 1.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let (laplacian, degrees) =
            compute_laplacianmatrix(&graph, LaplacianType::Standard, true, true).unwrap();

        let degrees = degrees.unwrap();

        // Vertex 2 is isolated, so degree should be 0
        assert_relative_eq!(degrees[2], 0.0);

        // Isolated vertex should have 0 on diagonal in Laplacian
        assert_relative_eq!(laplacian.get(2, 2), 0.0);
        assert_relative_eq!(laplacian.get(2, 0), 0.0);
        assert_relative_eq!(laplacian.get(2, 1), 0.0);
    }

    #[test]
    fn test_directed_graph() {
        // Create a simple directed graph: 0 -> 1 -> 2
        let rows = vec![0, 1];
        let cols = vec![1, 2];
        let data = vec![1.0, 1.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        // Test out-degrees
        let out_degrees = compute_out_degrees(&graph).unwrap();
        assert_relative_eq!(out_degrees[0], 1.0);
        assert_relative_eq!(out_degrees[1], 1.0);
        assert_relative_eq!(out_degrees[2], 0.0);

        // Test in-degrees
        let in_degrees = compute_in_degrees(&graph).unwrap();
        assert_relative_eq!(in_degrees[0], 0.0);
        assert_relative_eq!(in_degrees[1], 1.0);
        assert_relative_eq!(in_degrees[2], 1.0);
    }
}
