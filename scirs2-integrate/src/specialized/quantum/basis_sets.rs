//! Advanced basis sets for quantum calculations
//!
//! This module provides various basis sets and basis set operations
//! for quantum calculations and quantum chemistry applications.

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::collections::HashMap;

/// Advanced basis sets for quantum calculations
#[derive(Debug, Clone)]
pub struct AdvancedBasisSets {
    /// Number of basis functions
    pub n_basis: usize,
    /// Basis set type
    pub basis_type: BasisSetType,
    /// Basis function parameters
    pub parameters: Vec<BasisParameter>,
    /// Overlap matrix
    pub overlap_matrix: Array2<f64>,
}

impl AdvancedBasisSets {
    /// Create new advanced basis set
    pub fn new(n_basis: usize, basistype: BasisSetType) -> Self {
        let parameters = vec![BasisParameter::default(); n_basis];
        let overlap_matrix = Array2::eye(n_basis);

        Self {
            n_basis,
            basis_type: basistype,
            parameters,
            overlap_matrix,
        }
    }

    /// Generate basis functions
    pub fn generate_basis_functions(&self, coordinates: &Array2<f64>) -> Result<Array2<Complex64>> {
        let n_points = coordinates.nrows();
        let mut basis_functions = Array2::zeros((n_points, self.n_basis));

        match self.basis_type {
            BasisSetType::Gaussian => {
                self.generate_gaussian_basis(coordinates, &mut basis_functions)?;
            }
            BasisSetType::SlaterType => {
                self.generate_slater_basis(coordinates, &mut basis_functions)?;
            }
            BasisSetType::PlaneWave => {
                self.generate_plane_wave_basis(coordinates, &mut basis_functions)?;
            }
            BasisSetType::Atomic => {
                self.generate_atomic_basis(coordinates, &mut basis_functions)?;
            }
        }

        Ok(basis_functions)
    }

    /// Generate Gaussian basis functions
    fn generate_gaussian_basis(
        &self,
        coordinates: &Array2<f64>,
        basis_functions: &mut Array2<Complex64>,
    ) -> Result<()> {
        for (i, param) in self.parameters.iter().enumerate() {
            for (j, coord_row) in coordinates
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .enumerate()
            {
                let x = coord_row[0];
                let y = if coord_row.len() > 1 {
                    coord_row[1]
                } else {
                    0.0
                };
                let z = if coord_row.len() > 2 {
                    coord_row[2]
                } else {
                    0.0
                };

                let r_squared = (x - param.center_x).powi(2)
                    + (y - param.center_y).powi(2)
                    + (z - param.center_z).powi(2);

                let gaussian = (-param.exponent * r_squared).exp();
                basis_functions[[j, i]] = Complex64::new(gaussian * param.normalization, 0.0);
            }
        }

        Ok(())
    }

    /// Generate Slater-type basis functions
    fn generate_slater_basis(
        &self,
        coordinates: &Array2<f64>,
        basis_functions: &mut Array2<Complex64>,
    ) -> Result<()> {
        for (i, param) in self.parameters.iter().enumerate() {
            for (j, coord_row) in coordinates
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .enumerate()
            {
                let x = coord_row[0];
                let y = if coord_row.len() > 1 {
                    coord_row[1]
                } else {
                    0.0
                };
                let z = if coord_row.len() > 2 {
                    coord_row[2]
                } else {
                    0.0
                };

                let r = ((x - param.center_x).powi(2)
                    + (y - param.center_y).powi(2)
                    + (z - param.center_z).powi(2))
                .sqrt();

                let slater = r.powf(param.angular_momentum as f64) * (-param.exponent * r).exp();
                basis_functions[[j, i]] = Complex64::new(slater * param.normalization, 0.0);
            }
        }

        Ok(())
    }

    /// Generate plane wave basis functions
    fn generate_plane_wave_basis(
        &self,
        coordinates: &Array2<f64>,
        basis_functions: &mut Array2<Complex64>,
    ) -> Result<()> {
        use scirs2_core::constants::PI;

        for (i, param) in self.parameters.iter().enumerate() {
            for (j, coord_row) in coordinates
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .enumerate()
            {
                let x = coord_row[0];
                let y = if coord_row.len() > 1 {
                    coord_row[1]
                } else {
                    0.0
                };
                let z = if coord_row.len() > 2 {
                    coord_row[2]
                } else {
                    0.0
                };

                let k_dot_r = param.kx * x + param.ky * y + param.kz * z;
                let plane_wave = Complex64::new(
                    (k_dot_r).cos() * param.normalization,
                    (k_dot_r).sin() * param.normalization,
                );
                basis_functions[[j, i]] = plane_wave;
            }
        }

        Ok(())
    }

    /// Generate atomic orbital basis functions
    fn generate_atomic_basis(
        &self,
        coordinates: &Array2<f64>,
        basis_functions: &mut Array2<Complex64>,
    ) -> Result<()> {
        // Simplified atomic orbital generation
        for (i, param) in self.parameters.iter().enumerate() {
            for (j, coord_row) in coordinates
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .enumerate()
            {
                let x = coord_row[0];
                let y = if coord_row.len() > 1 {
                    coord_row[1]
                } else {
                    0.0
                };
                let z = if coord_row.len() > 2 {
                    coord_row[2]
                } else {
                    0.0
                };

                let r = ((x - param.center_x).powi(2)
                    + (y - param.center_y).powi(2)
                    + (z - param.center_z).powi(2))
                .sqrt();

                // Simplified hydrogen-like orbital
                let radial = r.powf(param.angular_momentum as f64) * (-param.exponent * r).exp();
                let orbital = radial * param.normalization;
                basis_functions[[j, i]] = Complex64::new(orbital, 0.0);
            }
        }

        Ok(())
    }

    /// Calculate overlap matrix
    pub fn calculate_overlap_matrix(&mut self, coordinates: &Array2<f64>) -> Result<()> {
        let basis_functions = self.generate_basis_functions(coordinates)?;
        let n_points = coordinates.nrows();

        self.overlap_matrix = Array2::zeros((self.n_basis, self.n_basis));

        for i in 0..self.n_basis {
            for j in 0..self.n_basis {
                let mut overlap = 0.0;
                for k in 0..n_points {
                    overlap += (basis_functions[[k, i]].conj() * basis_functions[[k, j]]).re;
                }
                self.overlap_matrix[[i, j]] = overlap;
            }
        }

        Ok(())
    }

    /// Orthogonalize basis functions using Gram-Schmidt
    pub fn orthogonalize_basis(&mut self) -> Result<()> {
        // Apply Gram-Schmidt orthogonalization to basis parameters
        for i in 1..self.n_basis {
            for j in 0..i {
                let overlap = self.overlap_matrix[[i, j]];
                if overlap.abs() > 1e-12 {
                    // Subtract projection
                    let norm_j = self.overlap_matrix[[j, j]].sqrt();
                    if norm_j > 1e-12 {
                        let projection_coeff = overlap / norm_j;
                        self.parameters[i].normalization -=
                            projection_coeff * self.parameters[j].normalization;
                    }
                }
            }
        }

        Ok(())
    }

    /// Transform basis functions
    pub fn transform_basis(
        &self,
        transformation_matrix: &Array2<f64>,
    ) -> Result<AdvancedBasisSets> {
        if transformation_matrix.nrows() != self.n_basis
            || transformation_matrix.ncols() != self.n_basis
        {
            return Err(IntegrateError::InvalidInput(
                "Transformation matrix dimension mismatch".to_string(),
            ));
        }

        let mut transformed_basis = self.clone();

        // Apply transformation to basis parameters
        for i in 0..self.n_basis {
            let mut new_normalization = 0.0;
            for j in 0..self.n_basis {
                new_normalization +=
                    transformation_matrix[[i, j]] * self.parameters[j].normalization;
            }
            transformed_basis.parameters[i].normalization = new_normalization;
        }

        // Transform overlap matrix
        let overlap_transformed = transformation_matrix
            .t()
            .dot(&self.overlap_matrix)
            .dot(transformation_matrix);
        transformed_basis.overlap_matrix = overlap_transformed;

        Ok(transformed_basis)
    }
}

/// Types of basis sets
#[derive(Debug, Clone, Copy)]
pub enum BasisSetType {
    /// Gaussian basis functions
    Gaussian,
    /// Slater-type orbitals
    SlaterType,
    /// Plane wave basis
    PlaneWave,
    /// Atomic orbital basis
    Atomic,
}

/// Parameters for individual basis functions
#[derive(Debug, Clone)]
pub struct BasisParameter {
    /// Exponent parameter
    pub exponent: f64,
    /// Normalization constant
    pub normalization: f64,
    /// Angular momentum quantum number
    pub angular_momentum: i32,
    /// Center coordinates
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    /// Wave vector components (for plane waves)
    pub kx: f64,
    pub ky: f64,
    pub kz: f64,
}

impl Default for BasisParameter {
    fn default() -> Self {
        Self {
            exponent: 1.0,
            normalization: 1.0,
            angular_momentum: 0,
            center_x: 0.0,
            center_y: 0.0,
            center_z: 0.0,
            kx: 0.0,
            ky: 0.0,
            kz: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basis_set_creation() {
        let basis = AdvancedBasisSets::new(5, BasisSetType::Gaussian);
        assert_eq!(basis.n_basis, 5);
        assert_eq!(basis.parameters.len(), 5);
        assert_eq!(basis.overlap_matrix.nrows(), 5);
        assert_eq!(basis.overlap_matrix.ncols(), 5);
    }

    #[test]
    fn test_gaussian_basis_generation() {
        let mut basis = AdvancedBasisSets::new(2, BasisSetType::Gaussian);

        // Set up simple Gaussian parameters
        basis.parameters[0].exponent = 1.0;
        basis.parameters[0].normalization = 1.0;
        basis.parameters[1].exponent = 2.0;
        basis.parameters[1].normalization = 1.0;
        basis.parameters[1].center_x = 1.0;

        let coordinates =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0])
                .unwrap();

        let basis_functions = basis.generate_basis_functions(&coordinates);
        assert!(basis_functions.is_ok());

        let functions = basis_functions.unwrap();
        assert_eq!(functions.nrows(), 3);
        assert_eq!(functions.ncols(), 2);
    }

    #[test]
    fn test_overlap_matrix_calculation() {
        let mut basis = AdvancedBasisSets::new(2, BasisSetType::Gaussian);

        let coordinates = Array2::from_shape_vec(
            (10, 3),
            vec![
                0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0, 0.0, 0.5,
                0.0, 0.0, 0.6, 0.0, 0.0, 0.7, 0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.0,
            ],
        )
        .unwrap();

        let result = basis.calculate_overlap_matrix(&coordinates);
        assert!(result.is_ok());

        // Diagonal elements should be positive
        for i in 0..basis.n_basis {
            assert!(basis.overlap_matrix[[i, i]] > 0.0);
        }
    }
}
