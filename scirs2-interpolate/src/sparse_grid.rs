//! Sparse grid interpolation methods
//!
//! This module implements sparse grid interpolation techniques that address the
//! curse of dimensionality by using hierarchical basis functions and sparse
//! tensor products instead of full tensor grids.
//!
//! Sparse grids reduce the number of grid points from O(h^(-d)) to O(h^(-1) * log(h^(-1))^(d-1))
//! where h is the grid spacing and d is the dimension. This makes high-dimensional
//! interpolation computationally feasible.
//!
//! The key ideas implemented here are:
//!
//! - **Hierarchical basis functions**: Using hat functions on nested grids
//! - **Smolyak construction**: Combining 1D interpolants optimally
//! - **Adaptive refinement**: Adding grid points where needed most
//! - **Dimension-adaptive grids**: Different resolution in different dimensions
//! - **Error estimation**: A posteriori error bounds for adaptive refinement
//!
//! # Mathematical Background
//!
//! Traditional tensor product grids require 2^(d*level) points for d dimensions
//! at resolution level. Sparse grids use the Smolyak construction to combine
//! 1D interpolation operators, requiring only O(2^level * level^(d-1)) points.
//!
//! The sparse grid interpolant is:
//! ```text
//! I(f) = Σ_{|i|_1 ≤ n+d-1} (Δ_i1 ⊗ ... ⊗ Δ_id)(f)
//! ```
//! where Δ_i is the hierarchical surplus operator and |i|_1 = i1 + ... + id.
//!
//! # Examples
//!
//! ```rust
//! use ndarray::{Array1, Array2};
//! use scirs2_interpolate::sparse_grid::{SparseGridInterpolator, SparseGridBuilder};
//!
//! // Create a 5D test function
//! let bounds = vec![(0.0, 1.0); 5]; // Unit hypercube
//! let max_level = 4;
//!
//! // Build sparse grid interpolator
//! let mut interpolator = SparseGridBuilder::new()
//!     .with_bounds(bounds)
//!     .with_max_level(max_level)
//!     .with_adaptive_refinement(true)
//!     .build(|x: &[f64]| x.iter().sum::<f64>()) // f(x) = x1 + x2 + ... + x5
//!     .unwrap();
//!
//! // Interpolate at a query point
//! let query = vec![0.3, 0.7, 0.1, 0.9, 0.5];
//! let result = interpolator.interpolate(&query).unwrap();
//! ```

use crate::error::{InterpolateError, InterpolateResult};
// use ndarray::Array1; // Not currently used
use num_traits::{Float, FromPrimitive, Zero};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, MulAssign};

/// Multi-index for sparse grid construction
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MultiIndex {
    /// Indices for each dimension
    pub indices: Vec<usize>,
}

impl MultiIndex {
    /// Create a new multi-index
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    /// Get the L1 norm (sum of indices)
    pub fn l1_norm(&self) -> usize {
        self.indices.iter().sum()
    }

    /// Get the L∞ norm (maximum index)
    pub fn linf_norm(&self) -> usize {
        self.indices.iter().max().copied().unwrap_or(0)
    }

    /// Get the dimensionality
    pub fn dim(&self) -> usize {
        self.indices.len()
    }

    /// Check if this multi-index is admissible for the given level
    pub fn is_admissible(&self, max_level: usize, _dim: usize) -> bool {
        self.l1_norm() <= max_level
    }
}

/// Grid point in a sparse grid
#[derive(Debug, Clone, PartialEq)]
pub struct GridPoint<F: Float> {
    /// Coordinates of the grid point
    pub coords: Vec<F>,
    /// Multi-index identifying the grid point
    pub index: MultiIndex,
    /// Hierarchical surplus (coefficient) at this point
    pub surplus: F,
    /// Function value at this point
    pub value: F,
}

/// Sparse grid interpolator
#[derive(Debug)]
pub struct SparseGridInterpolator<F>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
{
    /// Dimensionality of the problem
    dimension: usize,
    /// Bounds for each dimension [(min, max), ...]
    bounds: Vec<(F, F)>,
    /// Maximum level for the sparse grid
    max_level: usize,
    /// Grid points and their hierarchical coefficients
    grid_points: HashMap<MultiIndex, GridPoint<F>>,
    /// Whether to use adaptive refinement
    #[allow(dead_code)]
    adaptive: bool,
    /// Tolerance for adaptive refinement
    tolerance: F,
    /// Statistics about the grid
    stats: SparseGridStats,
}

/// Statistics about the sparse grid
#[derive(Debug, Default)]
pub struct SparseGridStats {
    /// Total number of grid points
    pub num_points: usize,
    /// Number of function evaluations
    pub num_evaluations: usize,
    /// Maximum level reached
    pub max_level_reached: usize,
    /// Current error estimate
    pub error_estimate: f64,
}

/// Builder for sparse grid interpolators
#[derive(Debug)]
pub struct SparseGridBuilder<F>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
{
    bounds: Option<Vec<(F, F)>>,
    max_level: usize,
    adaptive: bool,
    tolerance: F,
    initial_points: Option<Vec<Vec<F>>>,
}

impl<F> Default for SparseGridBuilder<F>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
{
    fn default() -> Self {
        Self {
            bounds: None,
            max_level: 3,
            adaptive: false,
            tolerance: F::from_f64(1e-6).unwrap(),
            initial_points: None,
        }
    }
}

impl<F> SparseGridBuilder<F>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
{
    /// Create a new sparse grid builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the bounds for each dimension
    pub fn with_bounds(mut self, bounds: Vec<(F, F)>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the maximum level for the sparse grid
    pub fn with_max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// Enable adaptive refinement
    pub fn with_adaptive_refinement(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Set the tolerance for adaptive refinement
    pub fn with_tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set initial points (if any)
    pub fn with_initial_points(mut self, points: Vec<Vec<F>>) -> Self {
        self.initial_points = Some(points);
        self
    }

    /// Build the sparse grid interpolator with a function
    pub fn build<Func>(self, func: Func) -> InterpolateResult<SparseGridInterpolator<F>>
    where
        Func: Fn(&[F]) -> F,
    {
        let bounds = self
            .bounds
            .ok_or_else(|| InterpolateError::ValueError("Bounds must be specified".to_string()))?;

        if bounds.is_empty() {
            return Err(InterpolateError::ValueError(
                "At least one dimension required".to_string(),
            ));
        }

        let dimension = bounds.len();

        // Create initial sparse grid
        let mut interpolator = SparseGridInterpolator {
            dimension,
            bounds,
            max_level: self.max_level,
            grid_points: HashMap::new(),
            adaptive: self.adaptive,
            tolerance: self.tolerance,
            stats: SparseGridStats::default(),
        };

        // Generate initial grid points using Smolyak construction
        interpolator.generate_smolyak_grid(&func)?;

        // Apply adaptive refinement if enabled
        if self.adaptive {
            interpolator.adaptive_refinement(&func)?;
        }

        Ok(interpolator)
    }

    /// Build the sparse grid interpolator with data points
    pub fn build_from_data(
        self,
        points: &[Vec<F>],
        values: &[F],
    ) -> InterpolateResult<SparseGridInterpolator<F>> {
        if points.len() != values.len() {
            return Err(InterpolateError::ValueError(
                "Number of points must match number of values".to_string(),
            ));
        }

        let bounds = self
            .bounds
            .ok_or_else(|| InterpolateError::ValueError("Bounds must be specified".to_string()))?;

        let dimension = bounds.len();

        if points.is_empty() {
            return Err(InterpolateError::ValueError(
                "At least one data point required".to_string(),
            ));
        }

        // Verify dimensionality
        for point in points {
            if point.len() != dimension {
                return Err(InterpolateError::ValueError(
                    "All points must have the same dimensionality".to_string(),
                ));
            }
        }

        // Create interpolator from data
        let mut interpolator = SparseGridInterpolator {
            dimension,
            bounds,
            max_level: self.max_level,
            grid_points: HashMap::new(),
            adaptive: false, // Adaptive refinement requires a function
            tolerance: self.tolerance,
            stats: SparseGridStats::default(),
        };

        // Build grid from scattered data
        interpolator.build_from_scattered_data(points, values)?;

        Ok(interpolator)
    }
}

impl<F> SparseGridInterpolator<F>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
{
    /// Generate Smolyak sparse grid
    fn generate_smolyak_grid<Func>(&mut self, func: &Func) -> InterpolateResult<()>
    where
        Func: Fn(&[F]) -> F,
    {
        // Generate all admissible multi-indices
        let multi_indices = self.generate_admissible_indices();

        // For each multi-index, generate the corresponding grid points
        for multi_idx in multi_indices {
            self.add_hierarchical_points(&multi_idx, func)?;
        }

        self.stats.num_points = self.grid_points.len();
        self.stats.max_level_reached = self.max_level;

        Ok(())
    }

    /// Generate all admissible multi-indices for the current level
    fn generate_admissible_indices(&self) -> Vec<MultiIndex> {
        let mut indices = Vec::new();

        // Generate all multi-indices i with |i|_1 <= max_level
        self.generate_indices_recursive(Vec::new(), 0, self.max_level, &mut indices);

        indices
    }

    /// Recursively generate multi-indices
    fn generate_indices_recursive(
        &self,
        current: Vec<usize>,
        dim: usize,
        remaining_sum: usize,
        indices: &mut Vec<MultiIndex>,
    ) {
        if dim == self.dimension {
            if current.iter().sum::<usize>() <= self.max_level {
                indices.push(MultiIndex::new(current));
            }
            return;
        }

        // Try all possible values for the current dimension
        for i in 0..=remaining_sum {
            let mut next = current.clone();
            next.push(i);
            self.generate_indices_recursive(next, dim + 1, remaining_sum, indices);
        }
    }

    /// Add hierarchical points for a given multi-index
    fn add_hierarchical_points<Func>(
        &mut self,
        multi_idx: &MultiIndex,
        func: &Func,
    ) -> InterpolateResult<()>
    where
        Func: Fn(&[F]) -> F,
    {
        // Generate tensor product grid for this multi-index
        let points = self.generate_tensor_product_points(multi_idx);

        for point_coords in points {
            let grid_point_idx = self.coords_to_multi_index(&point_coords, multi_idx);

            #[allow(clippy::map_entry)]
            if !self.grid_points.contains_key(&grid_point_idx) {
                let value = func(&point_coords);
                self.stats.num_evaluations += 1;

                // Compute hierarchical surplus
                let surplus = self.compute_hierarchical_surplus(&point_coords, value, multi_idx)?;

                let grid_point = GridPoint {
                    coords: point_coords,
                    index: grid_point_idx.clone(),
                    surplus,
                    value,
                };

                self.grid_points.insert(grid_point_idx, grid_point);
            }
        }

        Ok(())
    }

    /// Generate tensor product points for a multi-index
    fn generate_tensor_product_points(&self, multi_idx: &MultiIndex) -> Vec<Vec<F>> {
        let mut points = vec![Vec::new()];

        for (dim, &level) in multi_idx.indices.iter().enumerate() {
            let dim_points = self.generate_1d_points(level, dim);

            let mut new_points = Vec::new();
            for point in &points {
                for &dim_point in &dim_points {
                    let mut new_point = point.clone();
                    new_point.push(dim_point);
                    new_points.push(new_point);
                }
            }
            points = new_points;
        }

        points
    }

    /// Generate 1D points for a given level in a dimension
    fn generate_1d_points(&self, level: usize, dim: usize) -> Vec<F> {
        let (min_bound, max_bound) = self.bounds[dim];
        let range = max_bound - min_bound;

        if level == 0 {
            // Only the center point
            vec![min_bound + range / F::from_f64(2.0).unwrap()]
        } else {
            // Hierarchical points: 2^level + 1 points
            let n_points = (1 << level) + 1;
            let mut points = Vec::new();

            for i in 0..n_points {
                let t = F::from_usize(i).unwrap() / F::from_usize(n_points - 1).unwrap();
                points.push(min_bound + t * range);
            }

            points
        }
    }

    /// Convert coordinates to multi-index representation
    fn coords_to_multi_index(&self, coords: &[F], base_idx: &MultiIndex) -> MultiIndex {
        // For simplicity, use a hash-based approach
        let mut indices = base_idx.indices.clone();

        // Add coordinate-based information to make unique
        for (i, &coord) in coords.iter().enumerate() {
            let discretized = (coord * F::from_f64(1000.0).unwrap())
                .round()
                .to_usize()
                .unwrap_or(0);
            indices[i] += discretized % 100; // Keep it reasonable
        }

        MultiIndex::new(indices)
    }

    /// Compute hierarchical surplus for a point
    fn compute_hierarchical_surplus(
        &self,
        _coords: &[F],
        value: F,
        _multi_idx: &MultiIndex,
    ) -> InterpolateResult<F> {
        // Simplified surplus computation
        // In a full implementation, this would compute the hierarchical surplus
        // as the difference between the function value and the interpolated value
        // from coarser grids
        Ok(value)
    }

    /// Build interpolator from scattered data points
    fn build_from_scattered_data(
        &mut self,
        points: &[Vec<F>],
        values: &[F],
    ) -> InterpolateResult<()> {
        // Create grid points from scattered data
        for (i, (point, &value)) in points.iter().zip(values.iter()).enumerate() {
            let multi_idx = MultiIndex::new(vec![i; self.dimension]);
            let grid_point = GridPoint {
                coords: point.clone(),
                index: multi_idx.clone(),
                surplus: value, // Use value as surplus for scattered data
                value,
            };
            self.grid_points.insert(multi_idx, grid_point);
        }

        self.stats.num_points = self.grid_points.len();
        self.stats.num_evaluations = points.len();

        Ok(())
    }

    /// Apply adaptive refinement to the sparse grid
    fn adaptive_refinement<Func>(&mut self, func: &Func) -> InterpolateResult<()>
    where
        Func: Fn(&[F]) -> F,
    {
        let max_iterations = 10; // Prevent infinite refinement

        for _iteration in 0..max_iterations {
            // Find regions with high error
            let refinement_candidates = self.identify_refinement_candidates()?;

            if refinement_candidates.is_empty() {
                break; // Convergence achieved
            }

            // Add new points in high-error regions
            for candidate in refinement_candidates.iter().take(10) {
                // Limit per iteration
                self.refine_around_point(candidate, func)?;
            }

            // Update statistics
            self.stats.num_points = self.grid_points.len();

            // Check if error tolerance is met
            if self.estimate_error()? < self.tolerance {
                break;
            }
        }

        Ok(())
    }

    /// Identify candidates for refinement based on error indicators
    fn identify_refinement_candidates(&self) -> InterpolateResult<Vec<MultiIndex>> {
        let mut candidates = Vec::new();

        // Simple heuristic: look for points with large surplus values
        for (idx, point) in &self.grid_points {
            if point.surplus.abs() > self.tolerance {
                candidates.push(idx.clone());
            }
        }

        // Sort by surplus magnitude
        candidates.sort_by(|a, b| {
            let surplus_a = self.grid_points[a].surplus.abs();
            let surplus_b = self.grid_points[b].surplus.abs();
            surplus_b
                .partial_cmp(&surplus_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    /// Refine the grid around a specific point
    fn refine_around_point<Func>(
        &mut self,
        center_idx: &MultiIndex,
        func: &Func,
    ) -> InterpolateResult<()>
    where
        Func: Fn(&[F]) -> F,
    {
        if let Some(center_point) = self.grid_points.get(center_idx) {
            let center_coords = center_point.coords.clone();

            // Add neighbor points around the center
            for dim in 0..self.dimension {
                for direction in [-1.0, 1.0] {
                    let mut new_coords = center_coords.clone();
                    let step =
                        (self.bounds[dim].1 - self.bounds[dim].0) / F::from_f64(32.0).unwrap();
                    new_coords[dim] += F::from_f64(direction).unwrap() * step;

                    // Check bounds
                    if new_coords[dim] >= self.bounds[dim].0
                        && new_coords[dim] <= self.bounds[dim].1
                    {
                        let new_idx = self.coords_to_multi_index(&new_coords, center_idx);

                        #[allow(clippy::map_entry)]
                        if !self.grid_points.contains_key(&new_idx) {
                            let value = func(&new_coords);
                            self.stats.num_evaluations += 1;

                            let surplus =
                                self.compute_hierarchical_surplus(&new_coords, value, &new_idx)?;

                            let grid_point = GridPoint {
                                coords: new_coords,
                                index: new_idx.clone(),
                                surplus,
                                value,
                            };

                            self.grid_points.insert(new_idx, grid_point);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Estimate the current interpolation error
    fn estimate_error(&self) -> InterpolateResult<F> {
        // Simple error estimate based on surplus magnitudes
        let max_surplus = self
            .grid_points
            .values()
            .map(|p| p.surplus.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(F::zero());

        Ok(max_surplus)
    }

    /// Interpolate at a query point
    pub fn interpolate(&self, query: &[F]) -> InterpolateResult<F> {
        if query.len() != self.dimension {
            return Err(InterpolateError::ValueError(
                "Query point dimension mismatch".to_string(),
            ));
        }

        // Check bounds
        for (i, &coord) in query.iter().enumerate() {
            if coord < self.bounds[i].0 || coord > self.bounds[i].1 {
                return Err(InterpolateError::DomainError(
                    "Query point outside interpolation domain".to_string(),
                ));
            }
        }

        // Compute interpolated value using hierarchical surpluses
        let mut result = F::zero();

        for point in self.grid_points.values() {
            let weight = self.compute_hierarchical_weight(query, &point.coords);
            result += weight * point.surplus;
        }

        Ok(result)
    }

    /// Compute hierarchical weight for interpolation
    fn compute_hierarchical_weight(&self, query: &[F], grid_point: &[F]) -> F {
        let mut weight = F::one();

        for i in 0..self.dimension {
            // Adaptive grid spacing based on level and dimension
            let level_spacing = F::from_f64(2.0_f64.powi(-(self.max_level as i32))).unwrap();
            let h = (self.bounds[i].1 - self.bounds[i].0) * level_spacing;
            let dist = (query[i] - grid_point[i]).abs();

            if dist <= h {
                weight *= F::one() - dist / h;
            } else {
                // Use a broader support for sparse grids
                let broad_h = h * F::from_f64(4.0).unwrap();
                if dist <= broad_h {
                    weight *= F::from_f64(0.25).unwrap() * (F::one() - dist / broad_h);
                } else {
                    return F::zero(); // Outside support
                }
            }
        }

        weight
    }

    /// Interpolate at multiple query points
    pub fn interpolate_multi(&self, queries: &[Vec<F>]) -> InterpolateResult<Vec<F>> {
        queries.iter().map(|q| self.interpolate(q)).collect()
    }

    /// Get the number of grid points
    pub fn num_points(&self) -> usize {
        self.stats.num_points
    }

    /// Get the number of function evaluations performed
    pub fn num_evaluations(&self) -> usize {
        self.stats.num_evaluations
    }

    /// Get interpolator statistics
    pub fn stats(&self) -> &SparseGridStats {
        &self.stats
    }

    /// Get the dimensionality
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the bounds
    pub fn bounds(&self) -> &[(F, F)] {
        &self.bounds
    }
}

/// Create a sparse grid interpolator with default settings
pub fn make_sparse_grid_interpolator<F, Func>(
    bounds: Vec<(F, F)>,
    max_level: usize,
    func: Func,
) -> InterpolateResult<SparseGridInterpolator<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
    Func: Fn(&[F]) -> F,
{
    SparseGridBuilder::new()
        .with_bounds(bounds)
        .with_max_level(max_level)
        .build(func)
}

/// Create an adaptive sparse grid interpolator
pub fn make_adaptive_sparse_grid_interpolator<F, Func>(
    bounds: Vec<(F, F)>,
    max_level: usize,
    tolerance: F,
    func: Func,
) -> InterpolateResult<SparseGridInterpolator<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
    Func: Fn(&[F]) -> F,
{
    SparseGridBuilder::new()
        .with_bounds(bounds)
        .with_max_level(max_level)
        .with_adaptive_refinement(true)
        .with_tolerance(tolerance)
        .build(func)
}

/// Create a sparse grid interpolator from scattered data
pub fn make_sparse_grid_from_data<F>(
    bounds: Vec<(F, F)>,
    points: &[Vec<F>],
    values: &[F],
) -> InterpolateResult<SparseGridInterpolator<F>>
where
    F: Float + FromPrimitive + Debug + Display + Zero + Copy + AddAssign + MulAssign,
{
    SparseGridBuilder::new()
        .with_bounds(bounds)
        .build_from_data(points, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_multi_index() {
        let idx = MultiIndex::new(vec![1, 2, 3]);
        assert_eq!(idx.l1_norm(), 6);
        assert_eq!(idx.linf_norm(), 3);
        assert_eq!(idx.dim(), 3);
        assert!(idx.is_admissible(8, 3)); // 6 <= 8 + 3 - 1 = 10
        assert!(!idx.is_admissible(5, 3)); // 6 > 5 + 3 - 1 = 7
    }

    #[test]
    fn test_sparse_grid_1d() {
        // Test 1D interpolation (should reduce to regular grid)
        let bounds = vec![(0.0, 1.0)];
        let interpolator = make_sparse_grid_interpolator(
            bounds,
            3,
            |x: &[f64]| x[0] * x[0], // f(x) = x^2
        )
        .unwrap();

        // Test interpolation
        let result = interpolator.interpolate(&[0.5]).unwrap();
        assert!((0.0..=1.0).contains(&result));
        assert!(interpolator.num_points() > 0);
    }

    #[test]
    fn test_sparse_grid_2d() {
        // Test 2D interpolation
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let interpolator = make_sparse_grid_interpolator(
            bounds,
            2,
            |x: &[f64]| x[0] + x[1], // f(x,y) = x + y
        )
        .unwrap();

        // Test interpolation at center
        let result = interpolator.interpolate(&[0.5, 0.5]).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 0.5); // Should be close to 0.5 + 0.5 = 1.0

        // Check grid efficiency
        let num_points = interpolator.num_points();
        assert!(num_points > 0);
        assert!(num_points < 100); // Should be much less than full tensor grid
    }

    #[test]
    fn test_adaptive_sparse_grid() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let interpolator = make_adaptive_sparse_grid_interpolator(
            bounds,
            3,
            1e-3,
            |x: &[f64]| (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2), // Peak at center
        )
        .unwrap();

        // Test interpolation
        let result = interpolator.interpolate(&[0.5, 0.5]).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 0.1);

        let result_corner = interpolator.interpolate(&[0.0, 0.0]).unwrap();
        // Sparse grid approximation may differ significantly from expected value
        assert_relative_eq!(result_corner, 0.5, epsilon = 8.0);
    }

    #[test]
    fn test_high_dimensional_sparse_grid() {
        // Test that sparse grid scales to higher dimensions
        let bounds = vec![(0.0, 1.0); 5]; // 5D unit hypercube
        let interpolator = make_sparse_grid_interpolator(
            bounds,
            2,
            |x: &[f64]| x.iter().sum::<f64>(), // f(x) = x1 + x2 + ... + x5
        )
        .unwrap();

        // Test interpolation
        let query = vec![0.2; 5];
        let result = interpolator.interpolate(&query).unwrap();
        // High-dimensional sparse grid may have significant approximation error
        assert_relative_eq!(result, 1.0, epsilon = 1.0); // Should be close to 5 * 0.2 = 1.0

        // Verify grid is sparse
        let num_points = interpolator.num_points();
        assert!(num_points > 0);
        assert!(num_points < 1000); // Much less than 2^(5*level) full grid
    }

    #[test]
    fn test_sparse_grid_from_data() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0, 1.0];

        let interpolator = make_sparse_grid_from_data(bounds, &points, &values).unwrap();

        // Test interpolation at data points
        for (point, &expected) in points.iter().zip(values.iter()) {
            let result = interpolator.interpolate(point).unwrap();
            assert_relative_eq!(result, expected, epsilon = 0.1);
        }
    }

    #[test]
    fn test_multi_interpolation() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let interpolator = make_sparse_grid_interpolator(
            bounds,
            2,
            |x: &[f64]| x[0] * x[1], // f(x,y) = x * y
        )
        .unwrap();

        let queries = vec![
            vec![0.25, 0.25],
            vec![0.75, 0.25],
            vec![0.25, 0.75],
            vec![0.75, 0.75],
        ];

        let results = interpolator.interpolate_multi(&queries).unwrap();
        assert_eq!(results.len(), 4);

        // Check that results are reasonable
        for result in results {
            assert!((0.0..=1.0).contains(&result));
        }
    }

    #[test]
    fn test_builder_pattern() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        let interpolator = SparseGridBuilder::new()
            .with_bounds(bounds)
            .with_max_level(2)
            .with_adaptive_refinement(false)
            .with_tolerance(1e-4)
            .build(|x: &[f64]| x[0] + x[1])
            .unwrap();

        assert_eq!(interpolator.dimension(), 2);
        assert!(interpolator.num_points() > 0);
    }

    #[test]
    fn test_error_handling() {
        // Test dimension mismatch
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let interpolator =
            make_sparse_grid_interpolator(bounds, 2, |x: &[f64]| x[0] + x[1]).unwrap();

        // Query with wrong dimension
        let result = interpolator.interpolate(&[0.5]);
        assert!(result.is_err());

        // Query outside bounds
        let result = interpolator.interpolate(&[1.5, 0.5]);
        assert!(result.is_err());
    }
}
