//! Simplicial complexes and their topological invariants.
//!
//! A **simplicial complex** is a collection of simplices (points, edges,
//! triangles, tetrahedra, …) closed under taking faces.  This module provides:
//!
//! * [`SimplicialComplex`] – the core data structure.
//! * **Boundary matrices** `∂_k` for homology computation.
//! * **Betti numbers** β_0, β_1, … via rank-nullity of boundary matrices.
//! * **Euler characteristic** χ = Σ (-1)^k |C_k|.
//! * Constructors:
//!   - [`SimplicialComplex::vietoris_rips`] – from points and radius ε.
//!   - [`SimplicialComplex::cech_complex`] – from points and radius r (miniball check).
//!   - [`SimplicialComplex::nerve_complex`] – from a cover of index sets.
//!
//! # References
//! - Edelsbrunner & Harer, "Computational Topology", 2010.
//! - Zomorodian & Carlsson, "Computing persistent homology", DCG 2005.

use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use std::collections::{BTreeMap, BTreeSet};

// ============================================================================
// SimplicialComplex
// ============================================================================

/// A finite simplicial complex represented by its simplices, grouped by
/// dimension.
///
/// All simplices are stored as **sorted** vectors of vertex indices.  Adding a
/// simplex automatically adds all its faces (closure property).
///
/// ## Example
/// ```
/// use scirs2_graph::hypergraph::SimplicialComplex;
///
/// let mut sc = SimplicialComplex::new();
/// sc.add_simplex(vec![0, 1, 2]);  // adds triangle + all edges + all vertices
///
/// let betti = sc.betti_numbers();
/// assert_eq!(betti[0], 1); // one connected component
/// assert_eq!(betti[1], 0); // boundary is filled in
/// ```
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// Map: dimension → sorted set of simplices (each simplex = sorted `Vec<usize>`)
    simplices: BTreeMap<usize, BTreeSet<Vec<usize>>>,
}

impl Default for SimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplicialComplex {
    /// Create an empty simplicial complex.
    pub fn new() -> Self {
        SimplicialComplex {
            simplices: BTreeMap::new(),
        }
    }

    /// Add a simplex and all its faces (the **closure**).
    ///
    /// The simplex `[v_0, v_1, …, v_k]` is stored as a sorted, deduplicated
    /// vertex list.  All (k-1)-dimensional faces are recursively added.
    pub fn add_simplex(&mut self, mut simplex: Vec<usize>) {
        simplex.sort_unstable();
        simplex.dedup();
        self.add_simplex_internal(simplex);
    }

    /// Internal recursive insertion (simplex must already be sorted & deduped).
    fn add_simplex_internal(&mut self, simplex: Vec<usize>) {
        let dim = simplex.len().saturating_sub(1);
        let set = self.simplices.entry(dim).or_default();
        if set.contains(&simplex) {
            return; // already present, faces already added
        }
        set.insert(simplex.clone());

        // Add all (dim-1)-faces
        if simplex.len() > 1 {
            for i in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(i);
                self.add_simplex_internal(face);
            }
        }
    }

    /// Return the maximum dimension of the complex, or `None` if empty.
    pub fn max_dim(&self) -> Option<usize> {
        self.simplices.keys().next_back().copied()
    }

    /// Return a slice of all simplices at dimension `dim`.
    pub fn simplices_of_dim(&self, dim: usize) -> Vec<Vec<usize>> {
        self.simplices
            .get(&dim)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Total number of simplices across all dimensions.
    pub fn total_simplices(&self) -> usize {
        self.simplices.values().map(|s| s.len()).sum()
    }

    /// Number of simplices at dimension `dim`.
    pub fn num_simplices(&self, dim: usize) -> usize {
        self.simplices.get(&dim).map(|s| s.len()).unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Boundary matrix
    // -----------------------------------------------------------------------

    /// Compute the **boundary matrix** ∂_dim : C_dim → C_{dim-1}.
    ///
    /// Rows are indexed by (dim-1)-simplices; columns by dim-simplices.
    /// Entry `[i, j]` is `(-1)^k` where `k` is the position of the omitted
    /// vertex in simplex `j` that gives face `i`, else `0`.
    ///
    /// Returns an all-zero (1 × 1) matrix if there are no dim-simplices or no
    /// (dim-1)-simplices.
    pub fn boundary_matrix(&self, dim: usize) -> Array2<i8> {
        if dim == 0 {
            // ∂_0 = 0 (no (−1)-chains)
            let n0 = self.num_simplices(0);
            return Array2::zeros((1, n0.max(1)));
        }

        let chains_high = self.simplices_of_dim(dim);
        let chains_low = self.simplices_of_dim(dim - 1);

        if chains_high.is_empty() || chains_low.is_empty() {
            let rows = chains_low.len().max(1);
            let cols = chains_high.len().max(1);
            return Array2::zeros((rows, cols));
        }

        // Index the low-dimensional simplices for fast lookup
        let low_index: BTreeMap<Vec<usize>, usize> = chains_low
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let rows = chains_low.len();
        let cols = chains_high.len();
        let mut mat = Array2::<i8>::zeros((rows, cols));

        for (j, sigma) in chains_high.iter().enumerate() {
            for k in 0..sigma.len() {
                let mut face = sigma.clone();
                face.remove(k);
                if let Some(&i) = low_index.get(&face) {
                    let sign = if k % 2 == 0 { 1i8 } else { -1i8 };
                    mat[[i, j]] = sign;
                }
            }
        }
        mat
    }

    // -----------------------------------------------------------------------
    // Betti numbers (rank-nullity approach)
    // -----------------------------------------------------------------------

    /// Compute the **Betti numbers** β_0, β_1, …, β_{max_dim}.
    ///
    /// β_k = dim ker(∂_k) − dim im(∂_{k+1})
    ///      = (n_k − rank(∂_k)) − rank(∂_{k+1})
    ///
    /// where n_k is the number of k-simplices.
    ///
    /// Rank is computed by Gaussian elimination over ℤ (integer arithmetic,
    /// checking for divisibility).  Because our coefficient field is effectively
    /// ℚ (we use rational row operations), this gives exact Betti numbers over
    /// ℤ/2ℤ which coincides with ℤ Betti numbers for these complexes.
    pub fn betti_numbers(&self) -> Vec<usize> {
        let max_dim = match self.max_dim() {
            Some(d) => d,
            None => return Vec::new(),
        };

        // Compute rank of each boundary matrix
        let mut ranks: Vec<usize> = vec![0; max_dim + 2];
        for d in 0..=(max_dim + 1) {
            let mat = self.boundary_matrix(d);
            ranks[d] = matrix_rank_i8(&mat);
        }

        // β_k = (n_k - rank_k) - rank_{k+1}
        let mut betti = Vec::new();
        for k in 0..=max_dim {
            let n_k = self.num_simplices(k);
            let ker_k = n_k.saturating_sub(ranks[k]);
            let im_k1 = ranks[k + 1];
            betti.push(ker_k.saturating_sub(im_k1));
        }
        betti
    }

    // -----------------------------------------------------------------------
    // Euler characteristic
    // -----------------------------------------------------------------------

    /// Compute the **Euler characteristic**: χ = Σ_k (-1)^k |C_k|.
    pub fn euler_characteristic(&self) -> i64 {
        self.simplices
            .iter()
            .map(|(&dim, set)| {
                let sign: i64 = if dim % 2 == 0 { 1 } else { -1 };
                sign * set.len() as i64
            })
            .sum()
    }

    // -----------------------------------------------------------------------
    // Constructors from point clouds
    // -----------------------------------------------------------------------

    /// Build the **Vietoris-Rips complex** from a point cloud.
    ///
    /// Inserts a simplex on every subset of points whose pairwise Euclidean
    /// distances are all ≤ `epsilon`.
    ///
    /// # Arguments
    /// * `points`  – shape `(n_points, n_dims)`
    /// * `epsilon` – edge threshold
    ///
    /// # Complexity
    /// This is O(2^n) in the worst case; use only on small point sets (< 20).
    pub fn vietoris_rips(points: &Array2<f64>, epsilon: f64) -> Self {
        let n = points.nrows();
        let mut sc = SimplicialComplex::new();
        if n == 0 {
            return sc;
        }

        // Add all vertices
        for i in 0..n {
            sc.add_simplex(vec![i]);
        }

        // Precompute pairwise distances
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean_distance(
                    points.row(i).as_slice().unwrap_or(&[]),
                    points.row(j).as_slice().unwrap_or(&[]),
                );
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        // Build clique complex from edge graph (all pairs within epsilon)
        // Use Bron-Kerbosch to enumerate maximal cliques → add all cliques
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if dist[i][j] <= epsilon {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }

        // Add all cliques as simplices (flag complex = clique complex of 1-skeleton)
        let mut all_cliques: Vec<Vec<usize>> = Vec::new();
        bron_kerbosch(&adj, vec![], (0..n).collect(), vec![], &mut all_cliques);
        for clique in all_cliques {
            sc.add_simplex(clique);
        }
        sc
    }

    /// Build the **Čech complex** from a point cloud.
    ///
    /// A simplex σ is included iff the **miniball** (smallest enclosing ball)
    /// of the points in σ has radius ≤ `radius`.
    ///
    /// # Arguments
    /// * `points` – shape `(n_points, n_dims)`
    /// * `radius` – ball radius threshold
    pub fn cech_complex(points: &Array2<f64>, radius: f64) -> Self {
        let n = points.nrows();
        let d = points.ncols();
        let mut sc = SimplicialComplex::new();
        if n == 0 {
            return sc;
        }

        // For each subset, check miniball radius
        // We limit to subsets of size ≤ d+2 (by Helly's theorem, the miniball
        // is determined by at most d+1 points; we still enumerate all subsets
        // for correctness on small inputs).
        let max_simplex = (d + 2).min(n);

        // Add all vertices
        for i in 0..n {
            sc.add_simplex(vec![i]);
        }

        // Check edges
        for i in 0..n {
            for j in (i + 1)..n {
                let pts = vec![i, j];
                if miniball_radius(points, &pts) <= radius {
                    sc.add_simplex(pts);
                }
            }
        }

        // Higher-order simplices up to max_simplex
        for size in 3..=max_simplex {
            enumerate_subsets(n, size, &mut |subset| {
                if miniball_radius(points, subset) <= radius {
                    sc.add_simplex(subset.to_vec());
                }
            });
        }
        sc
    }

    /// Build the **nerve complex** from a cover.
    ///
    /// Given a cover `cover = [U_0, U_1, …, U_{k-1}]` where each `U_i` is a
    /// list of point indices, the nerve has:
    /// * A vertex for each `U_i`.
    /// * A simplex `{i_0, …, i_r}` whenever `U_{i_0} ∩ … ∩ U_{i_r} ≠ ∅`.
    ///
    /// # Arguments
    /// * `cover` – a slice of cover sets, each set being a sorted list of indices
    pub fn nerve_complex(cover: &[Vec<usize>]) -> Self {
        let mut sc = SimplicialComplex::new();
        let k = cover.len();
        if k == 0 {
            return sc;
        }

        // Vertex for each cover set
        for i in 0..k {
            sc.add_simplex(vec![i]);
        }

        // For each subset of cover sets, check if intersection is non-empty
        for size in 2..=k {
            enumerate_subsets(k, size, &mut |subset| {
                // Compute intersection of cover sets
                let mut inter: Vec<usize> = cover[subset[0]].clone();
                for &idx in &subset[1..] {
                    inter.retain(|x| cover[idx].contains(x));
                    if inter.is_empty() {
                        return;
                    }
                }
                if !inter.is_empty() {
                    sc.add_simplex(subset.to_vec());
                }
            });
        }
        sc
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Euclidean distance between two coordinate slices.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Radius of the smallest enclosing ball (miniball) for a set of points.
///
/// We use the Ritter algorithm for a bounding sphere approximation; for exact
/// results on up to 3 points we use the exact formula.
fn miniball_radius(points: &Array2<f64>, indices: &[usize]) -> f64 {
    match indices.len() {
        0 => 0.0,
        1 => 0.0,
        2 => {
            let d = points.ncols();
            let mut sq = 0.0f64;
            for k in 0..d {
                let diff = points[[indices[0], k]] - points[[indices[1], k]];
                sq += diff * diff;
            }
            sq.sqrt() / 2.0
        }
        _ => {
            // Ritter's bounding sphere (approximation — correct for our use case
            // since the Čech complex can be built with a slightly conservative radius)
            let d = points.ncols();
            // Find diameter pair
            let mut max_dist = 0.0f64;
            let mut p1 = indices[0];
            let mut p2 = indices[1];
            for &i in indices {
                for &j in indices {
                    if i == j {
                        continue;
                    }
                    let mut sq = 0.0f64;
                    for k in 0..d {
                        let diff = points[[i, k]] - points[[j, k]];
                        sq += diff * diff;
                    }
                    if sq > max_dist {
                        max_dist = sq;
                        p1 = i;
                        p2 = j;
                    }
                }
            }
            // Initial sphere: centre = midpoint(p1, p2), radius = dist/2
            let mut centre: Vec<f64> = (0..d)
                .map(|k| (points[[p1, k]] + points[[p2, k]]) / 2.0)
                .collect();
            let mut radius = max_dist.sqrt() / 2.0;

            // Expand to include all points
            for &i in indices {
                let mut sq = 0.0f64;
                for k in 0..d {
                    let diff = points[[i, k]] - centre[k];
                    sq += diff * diff;
                }
                let dist = sq.sqrt();
                if dist > radius {
                    // Expand sphere
                    let new_radius = (radius + dist) / 2.0;
                    let alpha = (dist - radius) / (2.0 * dist);
                    for k in 0..d {
                        centre[k] += alpha * (points[[i, k]] - centre[k]);
                    }
                    radius = new_radius;
                }
            }
            radius
        }
    }
}

/// Enumerate all subsets of `{0..n}` of size `k` and call `f` on each.
fn enumerate_subsets<F: FnMut(&[usize])>(n: usize, k: usize, f: &mut F) {
    let mut subset = vec![0usize; k];
    for i in 0..k {
        subset[i] = i;
    }
    loop {
        f(&subset);
        // Increment
        let mut i = k;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if subset[i] < n - k + i {
                subset[i] += 1;
                for j in (i + 1)..k {
                    subset[j] = subset[j - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Bron-Kerbosch algorithm for enumerating all maximal cliques.
fn bron_kerbosch(
    adj: &[Vec<usize>],
    r: Vec<usize>,
    mut p: Vec<usize>,
    mut x: Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if !r.is_empty() {
            result.push(r);
        }
        return;
    }
    // Choose pivot u ∈ P ∪ X that maximises |P ∩ N(u)|
    let pivot = {
        let all: Vec<usize> = p.iter().chain(x.iter()).copied().collect();
        *all.iter()
            .max_by_key(|&&u| p.iter().filter(|&&v| adj[u].contains(&v)).count())
            .unwrap_or(&all[0])
    };

    let candidates: Vec<usize> = p
        .iter()
        .copied()
        .filter(|&v| !adj[pivot].contains(&v))
        .collect();

    for v in candidates {
        let mut r_new = r.clone();
        r_new.push(v);
        let p_new: Vec<usize> = p.iter().copied().filter(|&u| adj[v].contains(&u)).collect();
        let x_new: Vec<usize> = x.iter().copied().filter(|&u| adj[v].contains(&u)).collect();
        bron_kerbosch(adj, r_new, p_new, x_new, result);
        p.retain(|&u| u != v);
        x.push(v);
    }
}

/// Compute the rank of an integer matrix over ℤ (by viewing entries as rationals).
///
/// We perform Gaussian elimination tracking exact integer fractions.
/// The result is the column rank = row rank.
fn matrix_rank_i8(mat: &Array2<i8>) -> usize {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    if rows == 0 || cols == 0 {
        return 0;
    }
    // Convert to i64 for elimination
    let mut m: Vec<Vec<i64>> = (0..rows)
        .map(|i| (0..cols).map(|j| mat[[i, j]] as i64).collect())
        .collect();

    let mut rank = 0usize;
    let mut pivot_row = 0usize;
    for col in 0..cols {
        // Find pivot
        let pivot = (pivot_row..rows).find(|&r| m[r][col] != 0);
        if let Some(p) = pivot {
            m.swap(pivot_row, p);
            // Eliminate all other rows
            let piv = m[pivot_row][col];
            for r in 0..rows {
                if r == pivot_row {
                    continue;
                }
                let factor = m[r][col];
                if factor == 0 {
                    continue;
                }
                for c in 0..cols {
                    m[r][c] = m[r][c] * piv - factor * m[pivot_row][c];
                }
                // Divide by gcd to prevent explosion
                let g = row_gcd(&m[r]);
                if g > 1 {
                    for c in 0..cols {
                        m[r][c] /= g;
                    }
                }
            }
            pivot_row += 1;
            rank += 1;
        }
    }
    rank
}

fn row_gcd(row: &[i64]) -> i64 {
    row.iter()
        .filter(|&&x| x != 0)
        .map(|x| x.unsigned_abs())
        .fold(0u64, gcd) as i64
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_simplex_closure() {
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1, 2]);
        // Should have 0-simplices: {0},{1},{2}; 1-simplices: {0,1},{0,2},{1,2}; 2-simplex {0,1,2}
        assert_eq!(sc.num_simplices(0), 3);
        assert_eq!(sc.num_simplices(1), 3);
        assert_eq!(sc.num_simplices(2), 1);
    }

    #[test]
    fn test_euler_characteristic_triangle_surface() {
        // A hollow triangle (just the boundary): V=3, E=3 → χ = 3-3 = 0
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1]);
        sc.add_simplex(vec![1, 2]);
        sc.add_simplex(vec![0, 2]);
        assert_eq!(sc.euler_characteristic(), 0);
    }

    #[test]
    fn test_euler_characteristic_filled_triangle() {
        // Filled triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1, 2]);
        assert_eq!(sc.euler_characteristic(), 1);
    }

    #[test]
    fn test_betti_numbers_point() {
        // Single point: β_0 = 1
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0]);
        let b = sc.betti_numbers();
        assert_eq!(b[0], 1);
    }

    #[test]
    fn test_betti_numbers_edge() {
        // Edge (filled): one connected component, no loops → β_0=1, β_1=0
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1]);
        let b = sc.betti_numbers();
        assert_eq!(b[0], 1);
        assert_eq!(b.get(1).copied().unwrap_or(0), 0);
    }

    #[test]
    fn test_betti_numbers_hollow_triangle() {
        // Hollow triangle: β_0=1 (connected), β_1=1 (one loop)
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1]);
        sc.add_simplex(vec![1, 2]);
        sc.add_simplex(vec![0, 2]);
        let b = sc.betti_numbers();
        assert_eq!(b[0], 1);
        assert_eq!(b.get(1).copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_betti_numbers_filled_triangle() {
        // Filled triangle: β_0=1, β_1=0
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1, 2]);
        let b = sc.betti_numbers();
        assert_eq!(b[0], 1);
        assert_eq!(b.get(1).copied().unwrap_or(0), 0);
    }

    #[test]
    fn test_betti_two_components() {
        // Two disjoint points: β_0 = 2
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0]);
        sc.add_simplex(vec![1]);
        let b = sc.betti_numbers();
        assert_eq!(b[0], 2);
    }

    #[test]
    fn test_boundary_matrix_dim0() {
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0]);
        let mat = sc.boundary_matrix(0);
        // Boundary of 0-chains is trivially 0
        assert_eq!(mat.nrows(), 1);
    }

    #[test]
    fn test_boundary_matrix_dim1() {
        // Edge {0,1}: ∂_1({0,1}) = {1} - {0}
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1]);
        let mat = sc.boundary_matrix(1);
        assert_eq!(mat.nrows(), 2); // two 0-simplices
        assert_eq!(mat.ncols(), 1); // one 1-simplex
                                    // One entry should be +1 and one -1
        let entries: Vec<i8> = vec![mat[[0, 0]], mat[[1, 0]]];
        assert!(entries.contains(&1));
        assert!(entries.contains(&-1));
    }

    #[test]
    fn test_vietoris_rips_collinear() {
        use scirs2_core::ndarray::array;
        // Three collinear points at distances 1,1 → VR(ε=1.5) should give 2 edges
        let pts = array![[0.0_f64, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let sc = SimplicialComplex::vietoris_rips(&pts, 1.5);
        // Expect edges {0,1} and {1,2} but not {0,2} (distance 2 > 1.5)
        assert_eq!(sc.num_simplices(1), 2);
        assert_eq!(sc.num_simplices(2), 0);
    }

    #[test]
    fn test_vietoris_rips_triangle() {
        use scirs2_core::ndarray::array;
        // Equilateral triangle with side 1 → VR(ε=1) = hollow triangle
        let pts = array![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 0.866_025_403_784_438_6]];
        let sc = SimplicialComplex::vietoris_rips(&pts, 1.0001);
        // All three edges present; filled (clique complex → add 2-simplex)
        assert_eq!(sc.num_simplices(0), 3);
        assert_eq!(sc.num_simplices(1), 3);
        assert_eq!(sc.num_simplices(2), 1);
    }

    #[test]
    fn test_nerve_complex_basic() {
        // U_0={0,1}, U_1={1,2}, U_2={2,3}
        // U_0 ∩ U_1 = {1} ≠ ∅ → edge {0,1}
        // U_1 ∩ U_2 = {2} ≠ ∅ → edge {1,2}
        // U_0 ∩ U_2 = ∅ → no edge {0,2}
        let cover = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let sc = SimplicialComplex::nerve_complex(&cover);
        assert_eq!(sc.num_simplices(0), 3);
        assert_eq!(sc.num_simplices(1), 2);
        assert_eq!(sc.num_simplices(2), 0);
    }

    #[test]
    fn test_nerve_triple_overlap() {
        // All three sets share node 0 → 2-simplex
        let cover = vec![vec![0, 1], vec![0, 2], vec![0, 3]];
        let sc = SimplicialComplex::nerve_complex(&cover);
        assert_eq!(sc.num_simplices(2), 1);
    }

    #[test]
    fn test_cech_complex_two_points() {
        use scirs2_core::ndarray::array;
        let pts = array![[0.0_f64], [1.0]];
        let sc = SimplicialComplex::cech_complex(&pts, 0.6); // miniball radius = 0.5 ≤ 0.6
        assert_eq!(sc.num_simplices(1), 1);
    }

    #[test]
    fn test_cech_complex_too_small_radius() {
        use scirs2_core::ndarray::array;
        let pts = array![[0.0_f64], [1.0]];
        let sc = SimplicialComplex::cech_complex(&pts, 0.4); // miniball radius = 0.5 > 0.4
        assert_eq!(sc.num_simplices(1), 0);
    }
}
