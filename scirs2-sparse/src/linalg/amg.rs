//! Algebraic Multigrid (AMG) preconditioner for sparse linear systems
//!
//! AMG is a powerful preconditioner for solving large sparse linear systems,
//! particularly effective for systems arising from discretizations of
//! elliptic PDEs and other problems with nice geometric structure.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Options for the AMG preconditioner
#[derive(Debug, Clone)]
pub struct AMGOptions {
    /// Maximum number of levels in the multigrid hierarchy
    pub max_levels: usize,
    /// Strong connection threshold for coarsening (typically 0.25-0.5)
    pub theta: f64,
    /// Maximum size of coarse grid before switching to direct solver
    pub max_coarse_size: usize,
    /// Interpolation method
    pub interpolation: InterpolationType,
    /// Smoother type
    pub smoother: SmootherType,
    /// Number of pre-smoothing steps
    pub pre_smooth_steps: usize,
    /// Number of post-smoothing steps
    pub post_smooth_steps: usize,
    /// Cycle type (V-cycle, W-cycle, etc.)
    pub cycle_type: CycleType,
}

impl Default for AMGOptions {
    fn default() -> Self {
        Self {
            max_levels: 10,
            theta: 0.25,
            max_coarse_size: 50,
            interpolation: InterpolationType::Classical,
            smoother: SmootherType::GaussSeidel,
            pre_smooth_steps: 1,
            post_smooth_steps: 1,
            cycle_type: CycleType::V,
        }
    }
}

/// Interpolation methods for AMG
#[derive(Debug, Clone, Copy)]
pub enum InterpolationType {
    /// Classical Ruge-Stuben interpolation
    Classical,
    /// Direct interpolation
    Direct,
    /// Standard interpolation
    Standard,
}

/// Smoother types for AMG
#[derive(Debug, Clone, Copy)]
pub enum SmootherType {
    /// Gauss-Seidel smoother
    GaussSeidel,
    /// Jacobi smoother
    Jacobi,
    /// SOR smoother
    SOR,
}

/// Cycle types for AMG
#[derive(Debug, Clone, Copy)]
pub enum CycleType {
    /// V-cycle
    V,
    /// W-cycle
    W,
    /// F-cycle
    F,
}

/// AMG preconditioner implementation
#[derive(Debug)]
pub struct AMGPreconditioner<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Matrices at each level
    operators: Vec<CsrArray<T>>,
    /// Prolongation operators (coarse to fine)
    prolongations: Vec<CsrArray<T>>,
    /// Restriction operators (fine to coarse)
    restrictions: Vec<CsrArray<T>>,
    /// AMG options
    options: AMGOptions,
    /// Number of levels in the hierarchy
    num_levels: usize,
}

impl<T> AMGPreconditioner<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Create a new AMG preconditioner from a sparse matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - The coefficient matrix
    /// * `options` - AMG options
    ///
    /// # Returns
    ///
    /// A new AMG preconditioner
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_sparse::csr_array::CsrArray;
    /// use scirs2_sparse::linalg::{AMGPreconditioner, AMGOptions};
    ///
    /// // Create a simple matrix
    /// let rows = vec![0, 0, 1, 1, 2, 2];
    /// let cols = vec![0, 1, 0, 1, 1, 2];
    /// let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
    /// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
    ///
    /// // Create AMG preconditioner
    /// let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();
    /// ```
    pub fn new(matrix: &CsrArray<T>, options: AMGOptions) -> SparseResult<Self> {
        let mut amg = AMGPreconditioner {
            operators: vec![matrix.clone()],
            prolongations: Vec::new(),
            restrictions: Vec::new(),
            options,
            num_levels: 1,
        };

        // Build the multigrid hierarchy
        amg.build_hierarchy()?;

        Ok(amg)
    }

    /// Build the multigrid hierarchy
    fn build_hierarchy(&mut self) -> SparseResult<()> {
        let mut level = 0;

        while level < self.options.max_levels - 1 {
            let currentmatrix = &self.operators[level];
            let (rows, _) = currentmatrix.shape();

            // Stop if matrix is small enough
            if rows <= self.options.max_coarse_size {
                break;
            }

            // Coarsen the matrix
            let (coarsematrix, prolongation, restriction) = self.coarsen_level(currentmatrix)?;

            // Check if coarsening was successful
            let (coarse_rows, _) = coarsematrix.shape();
            if coarse_rows >= rows {
                // Coarsening didn't reduce the problem size significantly
                break;
            }

            self.operators.push(coarsematrix);
            self.prolongations.push(prolongation);
            self.restrictions.push(restriction);
            self.num_levels += 1;
            level += 1;
        }

        Ok(())
    }

    /// Coarsen a single level using Ruge-Stuben algebraic coarsening
    fn coarsen_level(
        &self,
        matrix: &CsrArray<T>,
    ) -> SparseResult<(CsrArray<T>, CsrArray<T>, CsrArray<T>)> {
        let (n, _) = matrix.shape();

        // Step 1: Detect strong connections
        let strong_connections = self.detect_strong_connections(matrix)?;

        // Step 2: Perform C/F splitting using classical Ruge-Stuben algorithm
        let (c_points, f_points) = self.classical_cf_splitting(matrix, &strong_connections)?;

        // Step 3: Build coarse point mapping
        let mut fine_to_coarse = HashMap::new();
        for (coarse_idx, &fine_idx) in c_points.iter().enumerate() {
            fine_to_coarse.insert(fine_idx, coarse_idx);
        }

        let coarse_size = c_points.len();

        // Build prolongation operator (interpolation)
        let prolongation = self.build_prolongation(matrix, &fine_to_coarse, coarse_size)?;

        // Build restriction operator (typically transpose of prolongation)
        let restriction_box = prolongation.transpose()?;
        let restriction = restriction_box
            .as_any()
            .downcast_ref::<CsrArray<T>>()
            .ok_or_else(|| {
                SparseError::ValueError("Failed to downcast restriction to CsrArray".to_string())
            })?
            .clone();

        // Build coarse matrix: A_coarse = R * A * P
        let temp_box = restriction.dot(matrix)?;
        let temp = temp_box
            .as_any()
            .downcast_ref::<CsrArray<T>>()
            .ok_or_else(|| {
                SparseError::ValueError("Failed to downcast temp to CsrArray".to_string())
            })?;
        let coarsematrix_box = temp.dot(&prolongation)?;
        let coarsematrix = coarsematrix_box
            .as_any()
            .downcast_ref::<CsrArray<T>>()
            .ok_or_else(|| {
                SparseError::ValueError("Failed to downcast coarsematrix to CsrArray".to_string())
            })?
            .clone();

        Ok((coarsematrix, prolongation, restriction))
    }

    /// Detect strong connections in the matrix
    /// A connection i -> j is strong if |a_ij| >= theta * max_k(|a_ik|) for k != i
    fn detect_strong_connections(&self, matrix: &CsrArray<T>) -> SparseResult<Vec<Vec<usize>>> {
        let (n, _) = matrix.shape();
        let mut strong_connections = vec![Vec::new(); n];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let row_start = matrix.get_indptr()[i];
            let row_end = matrix.get_indptr()[i + 1];

            // Find maximum off-diagonal magnitude in this row
            let mut max_off_diag = T::zero();
            for j in row_start..row_end {
                let col = matrix.get_indices()[j];
                if col != i {
                    let val = matrix.get_data()[j].abs();
                    if val > max_off_diag {
                        max_off_diag = val;
                    }
                }
            }

            // Identify strong connections
            let threshold = T::from(self.options.theta).unwrap() * max_off_diag;
            for j in row_start..row_end {
                let col = matrix.get_indices()[j];
                if col != i {
                    let val = matrix.get_data()[j].abs();
                    if val >= threshold {
                        strong_connections[i].push(col);
                    }
                }
            }
        }

        Ok(strong_connections)
    }

    /// Classical Ruge-Stuben C/F splitting algorithm
    fn classical_cf_splitting(
        &self,
        matrix: &CsrArray<T>,
        strong_connections: &[Vec<usize>],
    ) -> SparseResult<(Vec<usize>, Vec<usize>)> {
        let (n, _) = matrix.shape();

        // Count strong _connections for each point (influence measure)
        let mut influence = vec![0; n];
        for i in 0..n {
            influence[i] = strong_connections[i].len();
        }

        // Track point types: 0 = undecided, 1 = C-point, 2 = F-point
        let mut point_type = vec![0; n];
        let mut c_points = Vec::new();
        let mut f_points = Vec::new();

        // Sort points by influence (high influence points become C-points first)
        let mut sorted_points: Vec<usize> = (0..n).collect();
        sorted_points.sort_by(|&a, &b| influence[b].cmp(&influence[a]));

        for &i in &sorted_points {
            if point_type[i] != 0 {
                continue; // Already processed
            }

            // Check if this point needs to be a C-point
            let mut needs_coarse = false;

            // If this point has strong F-point neighbors without coarse interpolatory set
            for &j in &strong_connections[i] {
                if point_type[j] == 2 {
                    // F-point
                    // Check if F-point j has a coarse interpolatory set
                    let mut has_coarse_interp = false;
                    for &k in &strong_connections[j] {
                        if point_type[k] == 1 {
                            // C-point
                            has_coarse_interp = true;
                            break;
                        }
                    }
                    if !has_coarse_interp {
                        needs_coarse = true;
                        break;
                    }
                }
            }

            if needs_coarse || influence[i] > 2 {
                // Make this a C-point
                point_type[i] = 1;
                c_points.push(i);

                // Make strongly connected neighbors F-points
                for &j in &strong_connections[i] {
                    if point_type[j] == 0 {
                        point_type[j] = 2;
                        f_points.push(j);
                    }
                }
            }
        }

        // Assign remaining undecided points as F-points
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if point_type[i] == 0 {
                point_type[i] = 2;
                f_points.push(i);
            }
        }

        Ok((c_points, f_points))
    }

    /// Build prolongation (interpolation) operator using algebraic interpolation
    fn build_prolongation(
        &self,
        matrix: &CsrArray<T>,
        fine_to_coarse: &HashMap<usize, usize>,
        coarse_size: usize,
    ) -> SparseResult<CsrArray<T>> {
        let (n, _) = matrix.shape();
        let mut prolongation_data = Vec::new();
        let mut prolongation_indices = Vec::new();
        let mut prolongation_indptr = vec![0];

        // Detect strong connections for interpolation
        let strong_connections = self.detect_strong_connections(matrix)?;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if let Some(&coarse_idx) = fine_to_coarse.get(&i) {
                // Direct injection for _coarse points
                prolongation_data.push(T::one());
                prolongation_indices.push(coarse_idx);
            } else {
                // Algebraic interpolation for fine points
                let interp_weights = self.compute_interpolation_weights(
                    i,
                    matrix,
                    &strong_connections[i],
                    fine_to_coarse,
                )?;

                if interp_weights.is_empty() {
                    // Fallback: direct injection to first _coarse point
                    prolongation_data.push(T::one());
                    prolongation_indices.push(0);
                } else {
                    // Add interpolation weights
                    for (coarse_idx, weight) in interp_weights {
                        prolongation_data.push(weight);
                        prolongation_indices.push(coarse_idx);
                    }
                }
            }
            prolongation_indptr.push(prolongation_data.len());
        }

        CsrArray::new(
            prolongation_data.into(),
            prolongation_indptr.into(),
            prolongation_indices.into(),
            (n, coarse_size),
        )
    }

    /// Compute interpolation weights for a fine point using classical interpolation
    fn compute_interpolation_weights(
        &self,
        fine_point: usize,
        matrix: &CsrArray<T>,
        strong_neighbors: &[usize],
        fine_to_coarse: &HashMap<usize, usize>,
    ) -> SparseResult<Vec<(usize, T)>> {
        let mut weights = Vec::new();

        // Find _coarse _neighbors that are strongly connected
        let mut coarse_neighbors = Vec::new();
        let mut coarse_weights = Vec::new();

        for &neighbor in strong_neighbors {
            if let Some(&coarse_idx) = fine_to_coarse.get(&neighbor) {
                coarse_neighbors.push(neighbor);
                coarse_weights.push(coarse_idx);
            }
        }

        if coarse_neighbors.is_empty() {
            return Ok(weights);
        }

        // Get the diagonal entry for fine _point
        let mut a_ii = T::zero();
        let row_start = matrix.get_indptr()[fine_point];
        let row_end = matrix.get_indptr()[fine_point + 1];

        for j in row_start..row_end {
            let col = matrix.get_indices()[j];
            if col == fine_point {
                a_ii = matrix.get_data()[j];
                break;
            }
        }

        if a_ii.is_zero() {
            return Ok(weights);
        }

        // Compute interpolation weights using classical formula
        // w_j = -a_ij / a_ii for _coarse _neighbors j
        let mut total_weight = T::zero();
        let mut temp_weights = Vec::new();

        for &coarse_neighbor in &coarse_neighbors {
            let mut a_ij = T::zero();
            for j in row_start..row_end {
                let col = matrix.get_indices()[j];
                if col == coarse_neighbor {
                    a_ij = matrix.get_data()[j];
                    break;
                }
            }

            if !a_ij.is_zero() {
                let weight = -a_ij / a_ii;
                temp_weights.push(weight);
                total_weight = total_weight + weight;
            } else {
                temp_weights.push(T::zero());
            }
        }

        // Normalize weights to sum to 1
        if !total_weight.is_zero() {
            for (i, &coarse_idx) in coarse_weights.iter().enumerate() {
                let normalized_weight = temp_weights[i] / total_weight;
                if !normalized_weight.is_zero() {
                    weights.push((coarse_idx, normalized_weight));
                }
            }
        }

        Ok(weights)
    }

    /// Apply the AMG preconditioner
    ///
    /// Solves the system M * x = b approximately, where M is the preconditioner
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// Approximate solution x
    pub fn apply(&self, b: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let (n, _) = self.operators[0].shape();
        if b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }

        let mut x = Array1::zeros(n);
        self.mg_cycle(&mut x, b, 0)?;
        Ok(x)
    }

    /// Perform one multigrid cycle
    fn mg_cycle(&self, x: &mut Array1<T>, b: &ArrayView1<T>, level: usize) -> SparseResult<()> {
        if level == self.num_levels - 1 {
            // Coarsest level - solve directly (simplified)
            self.coarse_solve(x, b, level)?;
            return Ok(());
        }

        let matrix = &self.operators[level];

        // Pre-smoothing
        for _ in 0..self.options.pre_smooth_steps {
            self.smooth(x, b, matrix)?;
        }

        // Compute residual
        let ax = matrix_vector_multiply(matrix, &x.view())?;
        let residual = b - &ax;

        // Restrict residual to coarse grid
        let restriction = &self.restrictions[level];
        let coarse_residual = matrix_vector_multiply(restriction, &residual.view())?;

        // Solve on coarse grid
        let coarse_size = coarse_residual.len();
        let mut coarse_correction = Array1::zeros(coarse_size);

        match self.options.cycle_type {
            CycleType::V => {
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
            }
            CycleType::W => {
                // Two recursive calls for W-cycle
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
            }
            CycleType::F => {
                // Full multigrid - not implemented here
                self.mg_cycle(&mut coarse_correction, &coarse_residual.view(), level + 1)?;
            }
        }

        // Prolongate correction to fine grid
        let prolongation = &self.prolongations[level];
        let fine_correction = matrix_vector_multiply(prolongation, &coarse_correction.view())?;

        // Add correction
        for i in 0..x.len() {
            x[i] = x[i] + fine_correction[i];
        }

        // Post-smoothing
        for _ in 0..self.options.post_smooth_steps {
            self.smooth(x, b, matrix)?;
        }

        Ok(())
    }

    /// Apply smoother
    fn smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
    ) -> SparseResult<()> {
        match self.options.smoother {
            SmootherType::GaussSeidel => self.gauss_seidel_smooth(x, b, matrix),
            SmootherType::Jacobi => self.jacobi_smooth(x, b, matrix),
            SmootherType::SOR => self.sor_smooth(x, b, matrix, T::from(1.2).unwrap()),
        }
    }

    /// Gauss-Seidel smoother
    fn gauss_seidel_smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
    ) -> SparseResult<()> {
        let n = x.len();

        for i in 0..n {
            let row_start = matrix.get_indptr()[i];
            let row_end = matrix.get_indptr()[i + 1];

            let mut sum = T::zero();
            let mut diag_val = T::zero();

            for j in row_start..row_end {
                let col = matrix.get_indices()[j];
                let val = matrix.get_data()[j];

                if col == i {
                    diag_val = val;
                } else {
                    sum = sum + val * x[col];
                }
            }

            if !diag_val.is_zero() {
                x[i] = (b[i] - sum) / diag_val;
            }
        }

        Ok(())
    }

    /// Jacobi smoother
    fn jacobi_smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
    ) -> SparseResult<()> {
        let n = x.len();
        let mut x_new = x.clone();

        for i in 0..n {
            let row_start = matrix.get_indptr()[i];
            let row_end = matrix.get_indptr()[i + 1];

            let mut sum = T::zero();
            let mut diag_val = T::zero();

            for j in row_start..row_end {
                let col = matrix.get_indices()[j];
                let val = matrix.get_data()[j];

                if col == i {
                    diag_val = val;
                } else {
                    sum = sum + val * x[col];
                }
            }

            if !diag_val.is_zero() {
                x_new[i] = (b[i] - sum) / diag_val;
            }
        }

        *x = x_new;
        Ok(())
    }

    /// SOR smoother
    fn sor_smooth(
        &self,
        x: &mut Array1<T>,
        b: &ArrayView1<T>,
        matrix: &CsrArray<T>,
        omega: T,
    ) -> SparseResult<()> {
        let n = x.len();

        for i in 0..n {
            let row_start = matrix.get_indptr()[i];
            let row_end = matrix.get_indptr()[i + 1];

            let mut sum = T::zero();
            let mut diag_val = T::zero();

            for j in row_start..row_end {
                let col = matrix.get_indices()[j];
                let val = matrix.get_data()[j];

                if col == i {
                    diag_val = val;
                } else {
                    sum = sum + val * x[col];
                }
            }

            if !diag_val.is_zero() {
                let x_gs = (b[i] - sum) / diag_val;
                x[i] = (T::one() - omega) * x[i] + omega * x_gs;
            }
        }

        Ok(())
    }

    /// Coarse grid solver (simplified direct method)
    fn coarse_solve(&self, x: &mut Array1<T>, b: &ArrayView1<T>, level: usize) -> SparseResult<()> {
        // For now, just use a few iterations of Gauss-Seidel
        let matrix = &self.operators[level];

        for _ in 0..10 {
            self.gauss_seidel_smooth(x, b, matrix)?;
        }

        Ok(())
    }

    /// Get the number of levels in the hierarchy
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Get the size of the matrix at a given level
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        if level < self.num_levels {
            Some(self.operators[level].shape())
        } else {
            None
        }
    }
}

/// Helper function for matrix-vector multiplication
#[allow(dead_code)]
fn matrix_vector_multiply<T>(matrix: &CsrArray<T>, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + 'static,
{
    let (rows, cols) = matrix.shape();
    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(rows);

    for i in 0..rows {
        for j in matrix.get_indptr()[i]..matrix.get_indptr()[i + 1] {
            let col = matrix.get_indices()[j];
            let val = matrix.get_data()[j];
            result[i] = result[i] + val * x[col];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;

    #[test]
    fn test_amg_preconditioner_creation() {
        // Create a simple 3x3 matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();

        assert!(amg.num_levels() >= 1);
        assert_eq!(amg.level_size(0), Some((3, 3)));
    }

    #[test]
    fn test_amg_apply() {
        // Create a diagonal system (easy test case)
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();

        let b = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let x = amg.apply(&b.view()).unwrap();

        // For a diagonal system, AMG should get close to the exact solution [1, 1, 1]
        assert!(x[0] > 0.5 && x[0] < 1.5);
        assert!(x[1] > 0.5 && x[1] < 1.5);
        assert!(x[2] > 0.5 && x[2] < 1.5);
    }

    #[test]
    fn test_amg_options() {
        let options = AMGOptions {
            max_levels: 5,
            theta: 0.5,
            smoother: SmootherType::Jacobi,
            cycle_type: CycleType::W,
            ..Default::default()
        };

        assert_eq!(options.max_levels, 5);
        assert_eq!(options.theta, 0.5);
        assert!(matches!(options.smoother, SmootherType::Jacobi));
        assert!(matches!(options.cycle_type, CycleType::W));
    }

    #[test]
    fn test_gauss_seidel_smoother() {
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let amg = AMGPreconditioner::new(&matrix, AMGOptions::default()).unwrap();

        let mut x = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        // Apply one Gauss-Seidel iteration
        amg.gauss_seidel_smooth(&mut x, &b.view(), &matrix).unwrap();

        // Solution should improve (move away from zero)
        assert!(x.iter().any(|&val| val.abs() > 1e-10));
    }

    #[test]
    fn test_enhanced_amg_coarsening() {
        // Create a larger test matrix to better test algebraic coarsening
        let rows = vec![0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4];
        let cols = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 3, 4, 0];
        let data = vec![
            4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, 4.0, -1.0,
        ];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (5, 5), false).unwrap();

        let options = AMGOptions {
            theta: 0.25, // Strong connection threshold
            ..Default::default()
        };

        let amg = AMGPreconditioner::new(&matrix, options).unwrap();

        // Should have created a hierarchy
        assert!(amg.num_levels() >= 1);

        // Test that it can be applied
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let x = amg.apply(&b.view()).unwrap();

        // Check that the result has the right size
        assert_eq!(x.len(), 5);

        // Check that the solution is reasonable (not all zeros)
        assert!(x.iter().any(|&val| val.abs() > 1e-10));
    }

    #[test]
    fn test_strong_connection_detection() {
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![4.0, -2.0, -2.0, 4.0, -2.0, 4.0];
        let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let options = AMGOptions {
            theta: 0.25,
            ..Default::default()
        };
        let amg = AMGPreconditioner::new(&matrix, options).unwrap();

        let strong_connections = amg.detect_strong_connections(&matrix).unwrap();

        // Each point should have some strong connections
        assert!(!strong_connections[0].is_empty());
        assert!(!strong_connections[1].is_empty());

        // Verify strong connections are bidirectional for symmetric matrix
        if strong_connections[0].contains(&1) {
            assert!(strong_connections[1].contains(&0));
        }
    }
}
