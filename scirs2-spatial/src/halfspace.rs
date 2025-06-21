//! Halfspace intersection and convex polytope construction
//!
//! This module provides algorithms for computing the intersection of halfspaces
//! to construct convex polytopes. Halfspace intersection is the dual problem
//! to convex hull computation and is fundamental in computational geometry.
//!
//! # Theory
//!
//! A halfspace in d-dimensional space is defined by a linear inequality:
//! a₁x₁ + a₂x₂ + ... + aₐxₐ ≤ b
//!
//! The intersection of multiple halfspaces forms a convex polytope.
//! This module implements algorithms to:
//! - Compute the vertices of the polytope
//! - Extract faces and facets
//! - Handle degenerate cases
//! - Check feasibility
//!
//! # Examples
//!
//! ```
//! # use scirs2_spatial::halfspace::{HalfspaceIntersection, Halfspace};
//! # use ndarray::array;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Define halfspaces for a unit square: x ≥ 0, y ≥ 0, x ≤ 1, y ≤ 1
//! let halfspaces = vec![
//!     Halfspace::new(array![-1.0, 0.0], 0.0),   // -x ≤ 0  =>  x ≥ 0
//!     Halfspace::new(array![0.0, -1.0], 0.0),   // -y ≤ 0  =>  y ≥ 0
//!     Halfspace::new(array![1.0, 0.0], 1.0),    //  x ≤ 1
//!     Halfspace::new(array![0.0, 1.0], 1.0),    //  y ≤ 1
//! ];
//!
//! let intersection = HalfspaceIntersection::new(&halfspaces, None)?;
//!
//! // Get the vertices of the resulting polytope
//! let vertices = intersection.vertices();
//! println!("Polytope vertices: {:?}", vertices);
//!
//! // Check if the polytope is bounded
//! println!("Is bounded: {}", intersection.is_bounded());
//! # Ok(())
//! # }
//! ```

use crate::convex_hull::ConvexHull;
use crate::error::{SpatialError, SpatialResult};
use ndarray::{arr1, Array1, Array2, ArrayView1};

/// Representation of a halfspace: a·x ≤ b
#[derive(Debug, Clone, PartialEq)]
pub struct Halfspace {
    /// Normal vector (coefficients a)
    normal: Array1<f64>,
    /// Offset value b
    offset: f64,
}

impl Halfspace {
    /// Create a new halfspace with normal vector and offset
    ///
    /// # Arguments
    ///
    /// * `normal` - Normal vector (a₁, a₂, ..., aₐ)
    /// * `offset` - Offset value b
    ///
    /// # Returns
    ///
    /// * A new Halfspace instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::halfspace::Halfspace;
    /// use ndarray::array;
    ///
    /// // Halfspace x + y ≤ 1
    /// let hs = Halfspace::new(array![1.0, 1.0], 1.0);
    /// ```
    pub fn new(normal: Array1<f64>, offset: f64) -> Self {
        Self { normal, offset }
    }

    /// Get the normal vector
    pub fn normal(&self) -> &Array1<f64> {
        &self.normal
    }

    /// Get the offset
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Get the dimension of the halfspace
    pub fn dim(&self) -> usize {
        self.normal.len()
    }

    /// Check if a point satisfies the halfspace constraint
    ///
    /// # Arguments
    ///
    /// * `point` - Point to test
    ///
    /// # Returns
    ///
    /// * true if point satisfies a·x ≤ b, false otherwise
    pub fn contains(&self, point: &ArrayView1<f64>) -> bool {
        if point.len() != self.normal.len() {
            return false;
        }

        let dot_product: f64 = self
            .normal
            .iter()
            .zip(point.iter())
            .map(|(a, x)| a * x)
            .sum();
        dot_product <= self.offset + 1e-10 // Small tolerance for numerical errors
    }

    /// Get the distance from a point to the halfspace boundary
    ///
    /// # Arguments
    ///
    /// * `point` - Point to measure distance from
    ///
    /// # Returns
    ///
    /// * Signed distance (negative if inside halfspace, positive if outside)
    pub fn distance(&self, point: &ArrayView1<f64>) -> SpatialResult<f64> {
        if point.len() != self.normal.len() {
            return Err(SpatialError::ValueError(
                "Point dimension must match halfspace dimension".to_string(),
            ));
        }

        let dot_product: f64 = self
            .normal
            .iter()
            .zip(point.iter())
            .map(|(a, x)| a * x)
            .sum();
        let normal_norm = (self.normal.iter().map(|x| x * x).sum::<f64>()).sqrt();

        if normal_norm < 1e-15 {
            return Err(SpatialError::ValueError(
                "Halfspace normal vector cannot be zero".to_string(),
            ));
        }

        Ok((dot_product - self.offset) / normal_norm)
    }

    /// Normalize the halfspace so that the normal vector has unit length
    ///
    /// # Returns
    ///
    /// * A new normalized Halfspace
    pub fn normalize(&self) -> SpatialResult<Self> {
        let normal_norm = (self.normal.iter().map(|x| x * x).sum::<f64>()).sqrt();

        if normal_norm < 1e-15 {
            return Err(SpatialError::ValueError(
                "Cannot normalize halfspace with zero normal vector".to_string(),
            ));
        }

        Ok(Self {
            normal: &self.normal / normal_norm,
            offset: self.offset / normal_norm,
        })
    }
}

/// Result of halfspace intersection computation
#[derive(Debug, Clone)]
pub struct HalfspaceIntersection {
    /// Input halfspaces
    halfspaces: Vec<Halfspace>,
    /// Vertices of the resulting polytope
    vertices: Array2<f64>,
    /// Faces of the polytope (indices into vertices array)
    faces: Vec<Vec<usize>>,
    /// Dimension of the space
    dim: usize,
    /// Whether the polytope is bounded
    is_bounded: bool,
    /// Interior point (if provided)
    #[allow(dead_code)]
    interior_point: Option<Array1<f64>>,
}

impl HalfspaceIntersection {
    /// Compute the intersection of halfspaces
    ///
    /// # Arguments
    ///
    /// * `halfspaces` - Vector of halfspaces to intersect
    /// * `interior_point` - Optional interior point for unbounded regions
    ///
    /// # Returns
    ///
    /// * HalfspaceIntersection result or error
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_spatial::halfspace::{HalfspaceIntersection, Halfspace};
    /// # use ndarray::array;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let halfspaces = vec![
    ///     Halfspace::new(array![-1.0, 0.0], 0.0),   // x ≥ 0
    ///     Halfspace::new(array![0.0, -1.0], 0.0),   // y ≥ 0
    ///     Halfspace::new(array![1.0, 1.0], 2.0),    // x + y ≤ 2
    /// ];
    ///
    /// let intersection = HalfspaceIntersection::new(&halfspaces, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        halfspaces: &[Halfspace],
        interior_point: Option<Array1<f64>>,
    ) -> SpatialResult<Self> {
        if halfspaces.is_empty() {
            return Err(SpatialError::ValueError(
                "At least one halfspace is required".to_string(),
            ));
        }

        let dim = halfspaces[0].dim();
        if halfspaces.iter().any(|hs| hs.dim() != dim) {
            return Err(SpatialError::ValueError(
                "All halfspaces must have the same dimension".to_string(),
            ));
        }

        if dim < 2 {
            return Err(SpatialError::ValueError(
                "Halfspace intersection requires at least 2D".to_string(),
            ));
        }

        // Validate interior point if provided
        if let Some(ref point) = interior_point {
            if point.len() != dim {
                return Err(SpatialError::ValueError(
                    "Interior point dimension must match halfspace dimension".to_string(),
                ));
            }

            // Check that the point is actually interior to all halfspaces
            for hs in halfspaces {
                if !hs.contains(&point.view()) {
                    return Err(SpatialError::ValueError(
                        "Provided point is not in the interior of all halfspaces".to_string(),
                    ));
                }
            }
        }

        // Use dual transformation to convert to convex hull problem
        let (vertices, faces, is_bounded) =
            if interior_point.is_some() || Self::is_likely_bounded(halfspaces) {
                Self::compute_bounded_intersection(halfspaces, interior_point.as_ref())?
            } else {
                Self::compute_unbounded_intersection(halfspaces)?
            };

        Ok(HalfspaceIntersection {
            halfspaces: halfspaces.to_vec(),
            vertices,
            faces,
            dim,
            is_bounded,
            interior_point,
        })
    }

    /// Check if the intersection is likely to be bounded by examining halfspaces
    fn is_likely_bounded(halfspaces: &[Halfspace]) -> bool {
        let dim = halfspaces[0].dim();

        // Check if we have enough "bounding" halfspaces in different directions
        let mut positive_count = vec![0; dim];
        let mut negative_count = vec![0; dim];

        for hs in halfspaces {
            for (i, &val) in hs.normal.iter().enumerate() {
                if val > 1e-10 {
                    positive_count[i] += 1;
                } else if val < -1e-10 {
                    negative_count[i] += 1;
                }
            }
        }

        // If we have constraints in both positive and negative directions for each dimension,
        // the polytope is likely bounded
        positive_count
            .iter()
            .zip(negative_count.iter())
            .all(|(&pos, &neg)| pos > 0 && neg > 0)
    }

    /// Compute 2D polygon intersection using direct vertex enumeration
    fn compute_2d_intersection(
        halfspaces: &[Halfspace],
    ) -> SpatialResult<(Array2<f64>, Vec<Vec<usize>>, bool)> {
        let mut vertices = Vec::new();
        let n = halfspaces.len();

        // Find all intersection points between pairs of halfspace boundaries
        for i in 0..n {
            for j in (i + 1)..n {
                let hs1 = &halfspaces[i];
                let hs2 = &halfspaces[j];

                // Solve the 2x2 system: hs1.normal · x = hs1.offset, hs2.normal · x = hs2.offset
                let det = hs1.normal[0] * hs2.normal[1] - hs1.normal[1] * hs2.normal[0];

                if det.abs() < 1e-15 {
                    continue; // Parallel halfspaces
                }

                let x = (hs2.normal[1] * hs1.offset - hs1.normal[1] * hs2.offset) / det;
                let y = (hs1.normal[0] * hs2.offset - hs2.normal[0] * hs1.offset) / det;

                let candidate = arr1(&[x, y]);

                // Check if this intersection point satisfies all other halfspaces
                let mut is_vertex = true;
                for (k, hs_k) in halfspaces.iter().enumerate() {
                    if k == i || k == j {
                        continue;
                    }
                    if !hs_k.contains(&candidate.view()) {
                        is_vertex = false;
                        break;
                    }
                }

                if is_vertex {
                    vertices.push((x, y));
                }
            }
        }

        if vertices.is_empty() {
            return Err(SpatialError::ComputationError(
                "No vertices found in intersection".to_string(),
            ));
        }

        // Remove duplicate vertices
        vertices.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        vertices.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10 && (a.1 - b.1).abs() < 1e-10);

        // Order vertices counter-clockwise
        let center_x = vertices.iter().map(|v| v.0).sum::<f64>() / vertices.len() as f64;
        let center_y = vertices.iter().map(|v| v.1).sum::<f64>() / vertices.len() as f64;

        vertices.sort_by(|a, b| {
            let angle_a = (a.1 - center_y).atan2(a.0 - center_x);
            let angle_b = (b.1 - center_y).atan2(b.0 - center_x);
            angle_a
                .partial_cmp(&angle_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Convert to Array2
        let vertex_array = Array2::from_shape_vec(
            (vertices.len(), 2),
            vertices.iter().flat_map(|&(x, y)| vec![x, y]).collect(),
        )
        .map_err(|_| SpatialError::ComputationError("Failed to create vertex array".to_string()))?;

        // Create simple face list for 2D polygon (single face with all vertices)
        let faces = if vertices.len() >= 3 {
            vec![(0..vertices.len()).collect()]
        } else {
            vec![]
        };

        Ok((vertex_array, faces, true))
    }

    /// Compute intersection for bounded polytopes using dual transformation
    fn compute_bounded_intersection(
        halfspaces: &[Halfspace],
        interior_point: Option<&Array1<f64>>,
    ) -> SpatialResult<(Array2<f64>, Vec<Vec<usize>>, bool)> {
        let dim = halfspaces[0].dim();

        // For 2D case, use direct vertex enumeration which is more reliable
        if dim == 2 {
            return Self::compute_2d_intersection(halfspaces);
        }

        // Find or use provided interior point
        let interior = if let Some(point) = interior_point {
            point.clone()
        } else {
            Self::find_interior_point(halfspaces)?
        };

        // Transform halfspaces to dual points using interior point as origin
        let mut dual_points = Vec::new();

        for hs in halfspaces {
            // Transform: each halfspace a·x ≤ b becomes point
            // p = (a₁, a₂, ..., aₐ) / (b - a·interior)
            let denominator = hs.offset - hs.normal.dot(&interior);

            if denominator.abs() < 1e-15 {
                // Halfspace passes through or very close to interior point
                continue;
            }

            if denominator < 0.0 {
                return Err(SpatialError::ComputationError(
                    "Interior point violates halfspace constraint".to_string(),
                ));
            }

            let dual_point: Vec<f64> = hs.normal.iter().map(|&a| a / denominator).collect();
            dual_points.push(dual_point);
        }

        if dual_points.len() < dim + 1 {
            return Err(SpatialError::ComputationError(
                "Insufficient halfspaces to form bounded polytope".to_string(),
            ));
        }

        // Convert to ndarray for convex hull computation
        let dual_array = Array2::from_shape_vec(
            (dual_points.len(), dim),
            dual_points.into_iter().flatten().collect(),
        )
        .map_err(|_| {
            SpatialError::ComputationError("Failed to create dual points array".to_string())
        })?;

        // Compute convex hull of dual points
        let hull = ConvexHull::new(&dual_array.view())?;

        // Transform hull vertices back to primal space
        let hull_vertices = hull.vertex_indices();
        let mut primal_vertices = Vec::new();

        for &vertex_idx in hull_vertices {
            let dual_vertex = dual_array.row(vertex_idx);

            // Transform back: dual point (p₁, p₂, ..., pₐ) becomes primal vertex
            // v = interior + p / ||p||²
            let p_norm_sq: f64 = dual_vertex.iter().map(|x| x * x).sum();

            if p_norm_sq < 1e-15 {
                continue; // Skip degenerate points
            }

            let primal_vertex: Vec<f64> = interior
                .iter()
                .zip(dual_vertex.iter())
                .map(|(&interior_i, &p_i)| interior_i + p_i / p_norm_sq)
                .collect();

            primal_vertices.push(primal_vertex);
        }

        if primal_vertices.is_empty() {
            return Err(SpatialError::ComputationError(
                "No valid vertices found in intersection".to_string(),
            ));
        }

        // Convert vertices to array
        let vertices = Array2::from_shape_vec(
            (primal_vertices.len(), dim),
            primal_vertices.into_iter().flatten().collect(),
        )
        .map_err(|_| {
            SpatialError::ComputationError("Failed to create vertices array".to_string())
        })?;

        // Extract faces from hull simplices
        let faces = Self::extract_faces_from_hull(&hull)?;

        Ok((vertices, faces, true))
    }

    /// Compute intersection for potentially unbounded polytopes
    fn compute_unbounded_intersection(
        halfspaces: &[Halfspace],
    ) -> SpatialResult<(Array2<f64>, Vec<Vec<usize>>, bool)> {
        let dim = halfspaces[0].dim();

        // For unbounded case, we need to find intersection vertices by
        // solving systems of linear equations
        let vertices = Self::find_intersection_vertices(halfspaces)?;

        if vertices.nrows() == 0 {
            return Err(SpatialError::ComputationError(
                "No intersection vertices found".to_string(),
            ));
        }

        // Check if polytope is bounded by examining vertex distribution
        let is_bounded = Self::check_boundedness(&vertices, halfspaces)?;

        // For simplicity, create basic face structure
        let faces = if vertices.nrows() > dim {
            // Compute convex hull to get proper face structure
            let hull = ConvexHull::new(&vertices.view())?;
            Self::extract_faces_from_hull(&hull)?
        } else {
            // Create simple face structure for degenerate cases
            vec![(0..vertices.nrows()).collect()]
        };

        Ok((vertices, faces, is_bounded))
    }

    /// Find an interior point for the given halfspaces using linear programming
    fn find_interior_point(halfspaces: &[Halfspace]) -> SpatialResult<Array1<f64>> {
        let dim = halfspaces[0].dim();

        // Try simple candidate points first, ensuring they are truly interior
        let candidates = vec![
            Array1::from_elem(dim, 0.1),    // Small positive values
            Array1::from_elem(dim, 0.01),   // Very small positive values
            Array1::from_elem(dim, 0.5),    // Medium values
            Array1::from_elem(dim, 0.3333), // 1/3 values
            Array1::from_elem(dim, 0.25),   // 1/4 values
            Array1::zeros(dim),             // Origin (try last)
        ];

        for candidate in candidates {
            // Check that point is strictly interior (not on boundary)
            let mut is_strictly_interior = true;
            for hs in halfspaces {
                let dot_product = hs.normal.dot(&candidate);
                if dot_product >= hs.offset - 1e-10 {
                    // Point is on or outside this halfspace
                    is_strictly_interior = false;
                    break;
                }
            }

            if is_strictly_interior {
                return Ok(candidate);
            }
        }

        // Try to find a point by solving a linear programming problem
        // Use Chebyshev center approach: find point that maximizes distance to closest constraint

        // For simple cases, try analytical solutions
        if dim == 2 && halfspaces.len() >= 3 {
            // Try intersection of first two constraints, shifted inward
            let hs1 = &halfspaces[0];
            let hs2 = &halfspaces[1];

            // Solve n1·x = b1 and n2·x = b2 system
            let det = hs1.normal[0] * hs2.normal[1] - hs1.normal[1] * hs2.normal[0];

            if det.abs() > 1e-10 {
                let x = (hs2.normal[1] * hs1.offset - hs1.normal[1] * hs2.offset) / det;
                let y = (hs1.normal[0] * hs2.offset - hs2.normal[0] * hs1.offset) / det;

                let candidate = arr1(&[x, y]);

                // Check if this intersection point is feasible for all constraints
                if halfspaces.iter().all(|hs| hs.contains(&candidate.view())) {
                    return Ok(candidate);
                }

                // If boundary point is feasible, move it slightly inward
                // Find the direction that moves away from the closest constraint
                let mut min_slack = f64::INFINITY;
                let mut worst_constraint_idx = 0;

                for (i, hs) in halfspaces.iter().enumerate() {
                    let slack = hs.offset - hs.normal.dot(&candidate);
                    if slack < min_slack {
                        min_slack = slack;
                        worst_constraint_idx = i;
                    }
                }

                if min_slack >= -1e-10 {
                    // Point is feasible or very close, shift inward slightly
                    let shift_direction = &halfspaces[worst_constraint_idx].normal * (-0.1);
                    let shifted_candidate = &candidate + &shift_direction;

                    if halfspaces
                        .iter()
                        .all(|hs| hs.contains(&shifted_candidate.view()))
                    {
                        return Ok(shifted_candidate);
                    }
                }
            }
        }

        Err(SpatialError::ComputationError(
            "Cannot find feasible interior point".to_string(),
        ))
    }

    /// Find intersection vertices by solving systems of linear equations
    fn find_intersection_vertices(halfspaces: &[Halfspace]) -> SpatialResult<Array2<f64>> {
        let dim = halfspaces[0].dim();
        let n = halfspaces.len();

        if n < dim {
            return Err(SpatialError::ComputationError(
                "Need at least d halfspaces to find intersection vertices in d dimensions"
                    .to_string(),
            ));
        }

        let mut vertices = Vec::new();

        // Generate all combinations of d halfspaces
        let combinations = Self::generate_combinations(n, dim);

        for combo in combinations {
            if let Ok(vertex) = Self::solve_intersection_system(halfspaces, &combo) {
                // Check if vertex satisfies all other halfspaces
                if halfspaces.iter().all(|hs| hs.contains(&vertex.view())) {
                    vertices.push(vertex.to_vec());
                }
            }
        }

        // Remove duplicate vertices
        vertices.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.partial_cmp(y) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    Some(order) => return order,
                    None => return std::cmp::Ordering::Equal,
                }
            }
            std::cmp::Ordering::Equal
        });
        vertices.dedup_by(|a, b| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10));

        if vertices.is_empty() {
            return Ok(Array2::zeros((0, dim)));
        }

        Array2::from_shape_vec(
            (vertices.len(), dim),
            vertices.into_iter().flatten().collect(),
        )
        .map_err(|_| SpatialError::ComputationError("Failed to create vertices array".to_string()))
    }

    /// Generate combinations of k elements from n elements
    fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        if k > n {
            return vec![];
        }

        if k == 0 {
            return vec![vec![]];
        }

        if k == 1 {
            return (0..n).map(|i| vec![i]).collect();
        }

        let mut result = Vec::new();

        fn backtrack(
            start: usize,
            n: usize,
            k: usize,
            current: &mut Vec<usize>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == k {
                result.push(current.clone());
                return;
            }

            for i in start..n {
                current.push(i);
                backtrack(i + 1, n, k, current, result);
                current.pop();
            }
        }

        let mut current = Vec::new();
        backtrack(0, n, k, &mut current, &mut result);
        result
    }

    /// Solve system of linear equations for intersection of d halfspaces
    fn solve_intersection_system(
        halfspaces: &[Halfspace],
        indices: &[usize],
    ) -> SpatialResult<Array1<f64>> {
        let dim = halfspaces[0].dim();

        if indices.len() != dim {
            return Err(SpatialError::ValueError(
                "Need exactly d halfspaces to solve for d-dimensional intersection".to_string(),
            ));
        }

        // Build matrix A and vector b for system Ax = b
        let mut matrix_data = Vec::with_capacity(dim * dim);
        let mut rhs = Vec::with_capacity(dim);

        for &idx in indices {
            let hs = &halfspaces[idx];
            matrix_data.extend(hs.normal.iter());
            rhs.push(hs.offset);
        }

        // Use simple Gaussian elimination for small systems
        Self::solve_linear_system(&matrix_data, &rhs, dim)
    }

    /// Simple Gaussian elimination solver for small linear systems
    fn solve_linear_system(
        matrix_data: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> SpatialResult<Array1<f64>> {
        let mut aug_matrix = vec![vec![0.0; n + 1]; n];

        // Build augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug_matrix[i][j] = matrix_data[i * n + j];
            }
            aug_matrix[i][n] = rhs[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            aug_matrix.swap(i, max_row);

            // Check for singular matrix
            if aug_matrix[i][i].abs() < 1e-15 {
                return Err(SpatialError::ComputationError(
                    "Singular matrix in intersection computation".to_string(),
                ));
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = aug_matrix[k][i] / aug_matrix[i][i];
                for j in i..(n + 1) {
                    aug_matrix[k][j] -= factor * aug_matrix[i][j];
                }
            }
        }

        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            solution[i] = aug_matrix[i][n];
            for j in (i + 1)..n {
                solution[i] -= aug_matrix[i][j] * solution[j];
            }
            solution[i] /= aug_matrix[i][i];
        }

        Ok(Array1::from(solution))
    }

    /// Check if a polytope is bounded by examining vertices
    fn check_boundedness(vertices: &Array2<f64>, _halfspaces: &[Halfspace]) -> SpatialResult<bool> {
        if vertices.nrows() == 0 {
            return Ok(false);
        }

        // Simple check: if all coordinates are finite and within reasonable bounds
        let max_coord = vertices
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f64, |acc, x| acc.max(x));

        Ok(max_coord.is_finite() && max_coord < 1e10)
    }

    /// Extract face structure from convex hull
    fn extract_faces_from_hull(hull: &ConvexHull) -> SpatialResult<Vec<Vec<usize>>> {
        // For now, create a simple face structure
        // A complete implementation would extract actual facets from the hull
        let vertices = hull.vertex_indices();
        if vertices.len() < 3 {
            Ok(vec![vertices.to_vec()])
        } else {
            // Create triangular faces for simplicity
            let mut faces = Vec::new();
            for i in 1..(vertices.len() - 1) {
                faces.push(vec![vertices[0], vertices[i], vertices[i + 1]]);
            }
            Ok(faces)
        }
    }

    /// Get the vertices of the intersection polytope
    pub fn vertices(&self) -> &Array2<f64> {
        &self.vertices
    }

    /// Get the faces of the intersection polytope
    pub fn faces(&self) -> &[Vec<usize>] {
        &self.faces
    }

    /// Get the dimension of the space
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if the polytope is bounded
    pub fn is_bounded(&self) -> bool {
        self.is_bounded
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.nrows()
    }

    /// Get the number of faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Check if the intersection is feasible (non-empty)
    pub fn is_feasible(&self) -> bool {
        self.vertices.nrows() > 0
    }

    /// Get the input halfspaces
    pub fn halfspaces(&self) -> &[Halfspace] {
        &self.halfspaces
    }

    /// Compute the volume of the polytope (2D area, 3D volume)
    pub fn volume(&self) -> SpatialResult<f64> {
        if !self.is_bounded {
            return Err(SpatialError::ComputationError(
                "Cannot compute volume of unbounded polytope".to_string(),
            ));
        }

        if self.vertices.nrows() == 0 {
            return Ok(0.0);
        }

        match self.dim {
            2 => self.compute_polygon_area(),
            3 => self.compute_polyhedron_volume(),
            _ => Err(SpatialError::NotImplementedError(
                "Volume computation only supports 2D and 3D".to_string(),
            )),
        }
    }

    /// Compute area of 2D polygon using shoelace formula
    fn compute_polygon_area(&self) -> SpatialResult<f64> {
        let vertices = &self.vertices;
        let n = vertices.nrows();

        if n < 3 {
            return Ok(0.0);
        }

        // Order vertices counter-clockwise
        let center_x = vertices.column(0).mean().unwrap();
        let center_y = vertices.column(1).mean().unwrap();

        let mut vertex_angles: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let dx = vertices[[i, 0]] - center_x;
                let dy = vertices[[i, 1]] - center_y;
                (i, dy.atan2(dx))
            })
            .collect();

        vertex_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Apply shoelace formula
        let mut area = 0.0;
        for i in 0..n {
            let curr = vertex_angles[i].0;
            let next = vertex_angles[(i + 1) % n].0;
            area += vertices[[curr, 0]] * vertices[[next, 1]];
            area -= vertices[[next, 0]] * vertices[[curr, 1]];
        }

        Ok(area.abs() / 2.0)
    }

    /// Compute volume of 3D polyhedron using triangulation
    fn compute_polyhedron_volume(&self) -> SpatialResult<f64> {
        if self.vertices.nrows() < 4 {
            return Ok(0.0);
        }

        // Use triangulation approach: compute convex hull and sum tetrahedron volumes
        let hull = ConvexHull::new(&self.vertices.view())?;
        let hull_vertices = hull.vertices();

        if hull_vertices.len() < 4 {
            return Ok(0.0);
        }

        // Pick a reference point (first vertex)
        let reference = self.vertices.row(0);
        let mut total_volume = 0.0;

        // Sum volumes of tetrahedra formed by reference point and triangular faces
        for face in &self.faces {
            if face.len() >= 3 {
                let v1 = self.vertices.row(face[0]);
                let v2 = self.vertices.row(face[1]);
                let v3 = self.vertices.row(face[2]);

                // Compute tetrahedron volume using scalar triple product
                let a = [
                    v1[0] - reference[0],
                    v1[1] - reference[1],
                    v1[2] - reference[2],
                ];
                let b = [
                    v2[0] - reference[0],
                    v2[1] - reference[1],
                    v2[2] - reference[2],
                ];
                let c = [
                    v3[0] - reference[0],
                    v3[1] - reference[1],
                    v3[2] - reference[2],
                ];

                let volume = (a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    .abs()
                    / 6.0;

                total_volume += volume;
            }
        }

        Ok(total_volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_halfspace_creation() {
        let hs = Halfspace::new(arr1(&[1.0, 2.0]), 3.0);
        assert_eq!(hs.normal(), &arr1(&[1.0, 2.0]));
        assert_eq!(hs.offset(), 3.0);
        assert_eq!(hs.dim(), 2);
    }

    #[test]
    fn test_halfspace_contains() {
        let hs = Halfspace::new(arr1(&[1.0, 1.0]), 1.0); // x + y ≤ 1

        assert!(hs.contains(&arr1(&[0.0, 0.0]).view())); // Origin
        assert!(hs.contains(&arr1(&[0.5, 0.5]).view())); // On boundary
        assert!(!hs.contains(&arr1(&[1.0, 1.0]).view())); // Outside (just barely)
        assert!(!hs.contains(&arr1(&[2.0, 0.0]).view())); // Clearly outside
    }

    #[test]
    fn test_halfspace_distance() {
        let hs = Halfspace::new(arr1(&[1.0, 0.0]), 1.0); // x ≤ 1

        let dist1 = hs.distance(&arr1(&[0.0, 0.0]).view()).unwrap();
        assert_relative_eq!(dist1, -1.0, epsilon = 1e-10); // Inside

        let dist2 = hs.distance(&arr1(&[1.0, 0.0]).view()).unwrap();
        assert_relative_eq!(dist2, 0.0, epsilon = 1e-10); // On boundary

        let dist3 = hs.distance(&arr1(&[2.0, 0.0]).view()).unwrap();
        assert_relative_eq!(dist3, 1.0, epsilon = 1e-10); // Outside
    }

    #[test]
    fn test_unit_square_intersection() {
        let halfspaces = vec![
            Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // x ≥ 0
            Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // y ≥ 0
            Halfspace::new(arr1(&[1.0, 0.0]), 1.0),  // x ≤ 1
            Halfspace::new(arr1(&[0.0, 1.0]), 1.0),  // y ≤ 1
        ];

        let intersection = HalfspaceIntersection::new(&halfspaces, None).unwrap();

        assert!(intersection.is_feasible());
        assert!(intersection.is_bounded());
        assert_eq!(intersection.dim(), 2);

        // Should have 4 vertices for unit square
        assert_eq!(intersection.num_vertices(), 4);

        // Check area
        let area = intersection.volume().unwrap();
        assert_relative_eq!(area, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triangle_intersection() {
        let halfspaces = vec![
            Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // x ≥ 0
            Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // y ≥ 0
            Halfspace::new(arr1(&[1.0, 1.0]), 1.0),  // x + y ≤ 1
        ];

        let intersection = HalfspaceIntersection::new(&halfspaces, None).unwrap();

        assert!(intersection.is_feasible());
        assert!(intersection.is_bounded());
        assert_eq!(intersection.num_vertices(), 3);

        // Triangle area should be 0.5
        let area = intersection.volume().unwrap();
        assert_relative_eq!(area, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_empty_intersection() {
        let halfspaces = vec![
            Halfspace::new(arr1(&[1.0, 0.0]), 0.0),   // x ≤ 0
            Halfspace::new(arr1(&[-1.0, 0.0]), -1.0), // x ≥ 1
        ];

        // These halfspaces have no intersection
        let result = HalfspaceIntersection::new(&halfspaces, None);
        // This should either fail or return empty intersection
        match result {
            Ok(intersection) => assert!(!intersection.is_feasible()),
            Err(_) => {} // Also acceptable
        }
    }

    #[test]
    fn test_halfspace_normalize() {
        let hs = Halfspace::new(arr1(&[3.0, 4.0]), 10.0);
        let normalized = hs.normalize().unwrap();

        let normal_norm = (normalized.normal()[0].powi(2) + normalized.normal()[1].powi(2)).sqrt();
        assert_relative_eq!(normal_norm, 1.0, epsilon = 1e-10);

        // The normalized offset should be 10/5 = 2
        assert_relative_eq!(normalized.offset(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_dimensions() {
        let halfspaces = vec![
            Halfspace::new(arr1(&[1.0, 0.0]), 1.0),
            Halfspace::new(arr1(&[1.0, 0.0, 0.0]), 1.0), // Different dimension
        ];

        let result = HalfspaceIntersection::new(&halfspaces, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_interior_point_validation() {
        let halfspaces = vec![
            Halfspace::new(arr1(&[-1.0, 0.0]), 0.0), // x ≥ 0
            Halfspace::new(arr1(&[0.0, -1.0]), 0.0), // y ≥ 0
            Halfspace::new(arr1(&[1.0, 1.0]), 1.0),  // x + y ≤ 1
        ];

        // Valid interior point
        let valid_interior = arr1(&[0.2, 0.2]);
        let result1 = HalfspaceIntersection::new(&halfspaces, Some(valid_interior));
        assert!(result1.is_ok());

        // Invalid interior point (outside)
        let invalid_interior = arr1(&[2.0, 2.0]);
        let result2 = HalfspaceIntersection::new(&halfspaces, Some(invalid_interior));
        assert!(result2.is_err());
    }
}
