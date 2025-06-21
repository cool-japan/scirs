//! Kriging interpolation methods
//!
//! This module provides implementations of Kriging, also known as Gaussian process regression,
//! which is a method of spatial interpolation based on the theory of regionalized variables.
//! Kriging provides the best linear unbiased estimator (BLUE) for spatial data.
//!
//! # Theory
//!
//! Kriging assumes that the data follows a spatial stochastic process and uses
//! a variogram or covariance function to model spatial correlation. The main types
//! of Kriging implemented are:
//!
//! - **Simple Kriging**: Assumes a known constant mean
//! - **Ordinary Kriging**: Estimates the mean locally (most common)
//! - **Universal Kriging**: Models trend with basis functions
//!
//! The Kriging prediction at location x₀ is:
//! Z*(x₀) = Σᵢ λᵢ Z(xᵢ)
//!
//! where λᵢ are weights determined by solving the Kriging system.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::kriging::{OrdinaryKriging, VariogramModel};
//! use ndarray::array;
//!
//! // Sample data points (x, y, z)
//! let points = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [0.5, 0.5]
//! ];
//!
//! let values = array![1.0, 2.0, 3.0, 4.0, 2.5];
//!
//! // Create Kriging interpolator with spherical variogram
//! let variogram = VariogramModel::spherical(1.0, 0.1, 0.0);
//! let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();
//!
//! // Interpolate at new location
//! let prediction = kriging.predict(&[0.25, 0.25]).unwrap();
//! println!("Predicted value: {:.3}", prediction.value);
//! println!("Prediction variance: {:.3}", prediction.variance);
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

/// Variogram model types for Kriging
#[derive(Debug, Clone)]
pub enum VariogramModel {
    /// Spherical variogram: γ(h) = c₀ + c₁[1.5(h/a) - 0.5(h/a)³] for h ≤ a, γ(h) = c₀ + c₁ for h > a
    Spherical { range: f64, sill: f64, nugget: f64 },
    /// Exponential variogram: γ(h) = c₀ + c₁[1 - exp(-h/a)]
    Exponential { range: f64, sill: f64, nugget: f64 },
    /// Gaussian variogram: γ(h) = c₀ + c₁[1 - exp(-(h/a)²)]
    Gaussian { range: f64, sill: f64, nugget: f64 },
    /// Linear variogram: γ(h) = c₀ + c₁h (unbounded)
    Linear { slope: f64, nugget: f64 },
    /// Power variogram: γ(h) = c₀ + c₁h^α for 0 < α < 2
    Power {
        coefficient: f64,
        exponent: f64,
        nugget: f64,
    },
    /// Matérn variogram with parameter ν
    Matern {
        range: f64,
        sill: f64,
        nugget: f64,
        nu: f64,
    },
}

impl VariogramModel {
    /// Create a spherical variogram model
    ///
    /// # Arguments
    /// * `range` - Range parameter (distance where correlation becomes negligible)
    /// * `sill` - Sill parameter (maximum variance)
    /// * `nugget` - Nugget parameter (variance at zero distance)
    pub fn spherical(range: f64, sill: f64, nugget: f64) -> Self {
        Self::Spherical {
            range,
            sill,
            nugget,
        }
    }

    /// Create an exponential variogram model
    pub fn exponential(range: f64, sill: f64, nugget: f64) -> Self {
        Self::Exponential {
            range,
            sill,
            nugget,
        }
    }

    /// Create a Gaussian variogram model
    pub fn gaussian(range: f64, sill: f64, nugget: f64) -> Self {
        Self::Gaussian {
            range,
            sill,
            nugget,
        }
    }

    /// Create a linear variogram model
    pub fn linear(slope: f64, nugget: f64) -> Self {
        Self::Linear { slope, nugget }
    }

    /// Create a power variogram model
    pub fn power(coefficient: f64, exponent: f64, nugget: f64) -> Self {
        Self::Power {
            coefficient,
            exponent,
            nugget,
        }
    }

    /// Create a Matérn variogram model
    pub fn matern(range: f64, sill: f64, nugget: f64, nu: f64) -> Self {
        Self::Matern {
            range,
            sill,
            nugget,
            nu,
        }
    }

    /// Evaluate the variogram at distance h
    ///
    /// # Arguments
    /// * `h` - Distance
    ///
    /// # Returns
    /// * Variogram value
    pub fn evaluate(&self, h: f64) -> f64 {
        if h < 0.0 {
            return 0.0;
        }

        if h.abs() < 1e-15 {
            return match self {
                Self::Spherical { nugget, .. }
                | Self::Exponential { nugget, .. }
                | Self::Gaussian { nugget, .. }
                | Self::Linear { nugget, .. }
                | Self::Power { nugget, .. }
                | Self::Matern { nugget, .. } => *nugget,
            };
        }

        match self {
            Self::Spherical {
                range,
                sill,
                nugget,
            } => {
                if h >= *range {
                    nugget + sill
                } else {
                    let h_r = h / range;
                    nugget + sill * (1.5 * h_r - 0.5 * h_r.powi(3))
                }
            }
            Self::Exponential {
                range,
                sill,
                nugget,
            } => nugget + sill * (1.0 - (-h / range).exp()),
            Self::Gaussian {
                range,
                sill,
                nugget,
            } => nugget + sill * (1.0 - (-(h / range).powi(2)).exp()),
            Self::Linear { slope, nugget } => nugget + slope * h,
            Self::Power {
                coefficient,
                exponent,
                nugget,
            } => nugget + coefficient * h.powf(*exponent),
            Self::Matern {
                range,
                sill,
                nugget,
                nu,
            } => {
                let h_r = h / range;
                if h_r < 1e-10 {
                    *nugget
                } else {
                    // Simplified Matérn for common values of ν
                    let matern_val = if (nu - 0.5).abs() < 1e-10 {
                        // ν = 0.5: exponential
                        1.0 - (-h_r).exp()
                    } else if (nu - 1.5).abs() < 1e-10 {
                        // ν = 1.5
                        (1.0 + 3.0_f64.sqrt() * h_r) * (-3.0_f64.sqrt() * h_r).exp()
                    } else if (nu - 2.5).abs() < 1e-10 {
                        // ν = 2.5
                        (1.0 + 5.0_f64.sqrt() * h_r + 5.0 * h_r.powi(2) / 3.0)
                            * (-5.0_f64.sqrt() * h_r).exp()
                    } else {
                        // General case approximation
                        1.0 - ((-h_r).exp() * (1.0 + h_r))
                    };
                    nugget + sill * (1.0 - matern_val)
                }
            }
        }
    }

    /// Get the effective range of the variogram
    pub fn effective_range(&self) -> f64 {
        match self {
            Self::Spherical { range, .. } => *range,
            Self::Exponential { range, .. } => 3.0 * range, // Practical range
            Self::Gaussian { range, .. } => 3.0_f64.sqrt() * range, // Practical range
            Self::Linear { .. } => f64::INFINITY,
            Self::Power { .. } => f64::INFINITY,
            Self::Matern { range, .. } => 3.0 * range,
        }
    }

    /// Get the sill (maximum variance) of the variogram
    pub fn sill(&self) -> f64 {
        match self {
            Self::Spherical { sill, nugget, .. }
            | Self::Exponential { sill, nugget, .. }
            | Self::Gaussian { sill, nugget, .. }
            | Self::Matern { sill, nugget, .. } => sill + nugget,
            Self::Linear { .. } | Self::Power { .. } => f64::INFINITY,
        }
    }

    /// Get the nugget effect
    pub fn nugget(&self) -> f64 {
        match self {
            Self::Spherical { nugget, .. }
            | Self::Exponential { nugget, .. }
            | Self::Gaussian { nugget, .. }
            | Self::Linear { nugget, .. }
            | Self::Power { nugget, .. }
            | Self::Matern { nugget, .. } => *nugget,
        }
    }
}

/// Prediction result from Kriging interpolation
#[derive(Debug, Clone)]
pub struct KrigingPrediction {
    /// Predicted value
    pub value: f64,
    /// Prediction variance (uncertainty)
    pub variance: f64,
    /// Weights used in the prediction
    pub weights: Array1<f64>,
}

/// Ordinary Kriging interpolator
///
/// Ordinary Kriging assumes the mean is unknown but constant within a local neighborhood.
/// It provides the Best Linear Unbiased Estimator (BLUE) for spatial data.
#[derive(Debug, Clone)]
pub struct OrdinaryKriging {
    /// Data point locations
    points: Array2<f64>,
    /// Data values at points
    values: Array1<f64>,
    /// Variogram model
    variogram: VariogramModel,
    /// Number of data points
    n_points: usize,
    /// Dimension of space
    ndim: usize,
    /// Precomputed covariance matrix (inverse)
    cov_matrix_inv: Option<Array2<f64>>,
}

impl OrdinaryKriging {
    /// Create a new Ordinary Kriging interpolator
    ///
    /// # Arguments
    /// * `points` - Array of point coordinates, shape (n_points, ndim)
    /// * `values` - Array of values at points, shape (n_points,)
    /// * `variogram` - Variogram model to use
    ///
    /// # Returns
    /// * New OrdinaryKriging instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::kriging::{OrdinaryKriging, VariogramModel};
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    /// let values = array![1.0, 2.0, 3.0];
    /// let variogram = VariogramModel::spherical(1.0, 0.5, 0.1);
    ///
    /// let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();
    /// ```
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        variogram: VariogramModel,
    ) -> SpatialResult<Self> {
        let n_points = points.nrows();
        let ndim = points.ncols();

        if values.len() != n_points {
            return Err(SpatialError::ValueError(
                "Number of values must match number of points".to_string(),
            ));
        }

        if n_points < 3 {
            return Err(SpatialError::ValueError(
                "Need at least 3 points for Kriging".to_string(),
            ));
        }

        if !(1..=3).contains(&ndim) {
            return Err(SpatialError::ValueError(
                "Kriging supports 1D, 2D, and 3D points only".to_string(),
            ));
        }

        Ok(Self {
            points: points.to_owned(),
            values: values.to_owned(),
            variogram,
            n_points,
            ndim,
            cov_matrix_inv: None,
        })
    }

    /// Fit the Kriging model by precomputing the covariance matrix inverse
    ///
    /// This step is optional but recommended for multiple predictions
    /// as it avoids recomputing the matrix inverse each time.
    pub fn fit(&mut self) -> SpatialResult<()> {
        let cov_matrix = self.build_covariance_matrix()?;
        let inv_matrix = self.invert_matrix(&cov_matrix)?;
        self.cov_matrix_inv = Some(inv_matrix);
        Ok(())
    }

    /// Predict value at a new location
    ///
    /// # Arguments
    /// * `location` - Point where to predict, shape (ndim,)
    ///
    /// # Returns
    /// * KrigingPrediction with value, variance, and weights
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::kriging::{OrdinaryKriging, VariogramModel};
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    /// let values = array![1.0, 2.0, 3.0, 4.0];
    /// let variogram = VariogramModel::spherical(1.5, 1.0, 0.1);
    ///
    /// let mut kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();
    /// kriging.fit().unwrap();
    ///
    /// let prediction = kriging.predict(&[0.5, 0.5]).unwrap();
    /// println!("Predicted: {:.3} ± {:.3}", prediction.value, prediction.variance.sqrt());
    /// ```
    pub fn predict(&self, location: &[f64]) -> SpatialResult<KrigingPrediction> {
        if location.len() != self.ndim {
            return Err(SpatialError::ValueError(
                "Location dimension must match data dimension".to_string(),
            ));
        }

        // Build covariance matrix if not precomputed
        let cov_inv = if let Some(ref inv) = self.cov_matrix_inv {
            inv.clone()
        } else {
            let cov_matrix = self.build_covariance_matrix()?;
            self.invert_matrix(&cov_matrix)?
        };

        // Build covariance vector between new location and data points
        let mut cov_vector = Array1::zeros(self.n_points + 1);
        for i in 0..self.n_points {
            let dist = self.distance(location, &self.points.row(i).to_vec());
            cov_vector[i] = self.variogram.sill() - self.variogram.evaluate(dist);
        }
        cov_vector[self.n_points] = 1.0; // Lagrange multiplier for unbiasedness constraint

        // Solve for weights
        let weights_extended = cov_inv.dot(&cov_vector);
        let weights = weights_extended.slice(s![..self.n_points]).to_owned();

        // Calculate prediction
        let value = weights.dot(&self.values);

        // Calculate prediction variance
        let variance = self.variogram.sill() - weights_extended.dot(&cov_vector);

        Ok(KrigingPrediction {
            value,
            variance: variance.max(0.0), // Ensure non-negative variance
            weights,
        })
    }

    /// Predict values at multiple locations efficiently
    ///
    /// # Arguments
    /// * `locations` - Array of locations, shape (n_locations, ndim)
    ///
    /// # Returns
    /// * Vector of KrigingPrediction results
    pub fn predict_batch(
        &self,
        locations: &ArrayView2<f64>,
    ) -> SpatialResult<Vec<KrigingPrediction>> {
        if locations.ncols() != self.ndim {
            return Err(SpatialError::ValueError(
                "Location dimension must match data dimension".to_string(),
            ));
        }

        // Precompute covariance matrix inverse if not done
        let cov_inv = if let Some(ref inv) = self.cov_matrix_inv {
            inv.clone()
        } else {
            let cov_matrix = self.build_covariance_matrix()?;
            self.invert_matrix(&cov_matrix)?
        };

        let mut predictions = Vec::with_capacity(locations.nrows());

        for location_row in locations.outer_iter() {
            let location: Vec<f64> = location_row.to_vec();

            // Build covariance vector
            let mut cov_vector = Array1::zeros(self.n_points + 1);
            for i in 0..self.n_points {
                let dist = self.distance(&location, &self.points.row(i).to_vec());
                cov_vector[i] = self.variogram.sill() - self.variogram.evaluate(dist);
            }
            cov_vector[self.n_points] = 1.0;

            // Solve for weights
            let weights_extended = cov_inv.dot(&cov_vector);
            let weights = weights_extended.slice(s![..self.n_points]).to_owned();

            // Calculate prediction and variance
            let value = weights.dot(&self.values);
            let variance = (self.variogram.sill() - weights_extended.dot(&cov_vector)).max(0.0);

            predictions.push(KrigingPrediction {
                value,
                variance,
                weights,
            });
        }

        Ok(predictions)
    }

    /// Build the covariance matrix for the Kriging system
    fn build_covariance_matrix(&self) -> SpatialResult<Array2<f64>> {
        let size = self.n_points + 1;
        let mut matrix = Array2::zeros((size, size));

        // Fill covariance values
        for i in 0..self.n_points {
            for j in 0..self.n_points {
                let dist = if i == j {
                    0.0
                } else {
                    self.distance(&self.points.row(i).to_vec(), &self.points.row(j).to_vec())
                };
                // Covariance = Sill - Variogram
                matrix[[i, j]] = self.variogram.sill() - self.variogram.evaluate(dist);
            }
        }

        // Unbiasedness constraint (Lagrange multipliers)
        for i in 0..self.n_points {
            matrix[[i, self.n_points]] = 1.0;
            matrix[[self.n_points, i]] = 1.0;
        }
        matrix[[self.n_points, self.n_points]] = 0.0;

        Ok(matrix)
    }

    /// Compute Euclidean distance between two points
    fn distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Invert a matrix using Gaussian elimination with partial pivoting
    fn invert_matrix(&self, matrix: &Array2<f64>) -> SpatialResult<Array2<f64>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SpatialError::ComputationError(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // Create augmented matrix [A | I]
        let mut aug = Array2::zeros((n, 2 * n));

        // Fill A part
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = matrix[[i, j]];
            }
        }

        // Fill identity part
        for i in 0..n {
            aug[[i, n + i]] = 1.0;
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-12 {
                return Err(SpatialError::ComputationError(
                    "Matrix is singular (not invertible)".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = aug[[i, i]];
            for j in 0..(2 * n) {
                aug[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..(2 * n) {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = aug[[i, n + j]];
            }
        }

        Ok(inverse)
    }

    /// Get the variogram model
    pub fn variogram(&self) -> &VariogramModel {
        &self.variogram
    }

    /// Get the number of data points
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get the data points
    pub fn points(&self) -> &Array2<f64> {
        &self.points
    }

    /// Get the data values
    pub fn values(&self) -> &Array1<f64> {
        &self.values
    }

    /// Cross-validation: leave-one-out prediction errors
    ///
    /// # Returns
    /// * Array of prediction errors (predicted - actual)
    pub fn cross_validate(&self) -> SpatialResult<Array1<f64>> {
        let mut errors = Array1::zeros(self.n_points);

        for i in 0..self.n_points {
            // Create subset without point i
            let mut subset_points = Array2::zeros((self.n_points - 1, self.ndim));
            let mut subset_values = Array1::zeros(self.n_points - 1);

            let mut idx = 0;
            for j in 0..self.n_points {
                if j != i {
                    subset_points.row_mut(idx).assign(&self.points.row(j));
                    subset_values[idx] = self.values[j];
                    idx += 1;
                }
            }

            // Create Kriging model without point i
            let subset_kriging = OrdinaryKriging::new(
                &subset_points.view(),
                &subset_values.view(),
                self.variogram.clone(),
            )?;

            // Predict at point i
            let location: Vec<f64> = self.points.row(i).to_vec();
            let prediction = subset_kriging.predict(&location)?;

            errors[i] = prediction.value - self.values[i];
        }

        Ok(errors)
    }
}

/// Simple Kriging interpolator
///
/// Simple Kriging assumes a known constant mean value.
#[derive(Debug, Clone)]
pub struct SimpleKriging {
    /// Data point locations
    points: Array2<f64>,
    /// Data values at points
    values: Array1<f64>,
    /// Known mean value
    mean: f64,
    /// Variogram model
    variogram: VariogramModel,
    /// Number of data points
    n_points: usize,
    /// Dimension of space
    ndim: usize,
}

impl SimpleKriging {
    /// Create a new Simple Kriging interpolator
    ///
    /// # Arguments
    /// * `points` - Array of point coordinates
    /// * `values` - Array of values at points
    /// * `mean` - Known mean value
    /// * `variogram` - Variogram model
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        mean: f64,
        variogram: VariogramModel,
    ) -> SpatialResult<Self> {
        let n_points = points.nrows();
        let ndim = points.ncols();

        if values.len() != n_points {
            return Err(SpatialError::ValueError(
                "Number of values must match number of points".to_string(),
            ));
        }

        if n_points < 2 {
            return Err(SpatialError::ValueError(
                "Need at least 2 points for Simple Kriging".to_string(),
            ));
        }

        Ok(Self {
            points: points.to_owned(),
            values: values.to_owned(),
            mean,
            variogram,
            n_points,
            ndim,
        })
    }

    /// Predict value at a new location
    ///
    /// # Arguments
    /// * `location` - Point where to predict
    ///
    /// # Returns
    /// * KrigingPrediction with value, variance, and weights
    pub fn predict(&self, location: &[f64]) -> SpatialResult<KrigingPrediction> {
        if location.len() != self.ndim {
            return Err(SpatialError::ValueError(
                "Location dimension must match data dimension".to_string(),
            ));
        }

        // Build covariance matrix (without Lagrange multiplier)
        let mut cov_matrix = Array2::zeros((self.n_points, self.n_points));
        for i in 0..self.n_points {
            for j in 0..self.n_points {
                let dist = if i == j {
                    0.0
                } else {
                    self.distance(&self.points.row(i).to_vec(), &self.points.row(j).to_vec())
                };
                cov_matrix[[i, j]] = self.variogram.sill() - self.variogram.evaluate(dist);
            }
        }

        // Build covariance vector
        let mut cov_vector = Array1::zeros(self.n_points);
        for i in 0..self.n_points {
            let dist = self.distance(location, &self.points.row(i).to_vec());
            cov_vector[i] = self.variogram.sill() - self.variogram.evaluate(dist);
        }

        // Solve for weights
        let weights = self.solve_linear_system(&cov_matrix, &cov_vector)?;

        // Calculate prediction (residuals from mean)
        let residuals: Array1<f64> = &self.values - self.mean;
        let value = self.mean + weights.dot(&residuals);

        // Calculate prediction variance
        let variance = self.variogram.sill() - weights.dot(&cov_vector);

        Ok(KrigingPrediction {
            value,
            variance: variance.max(0.0),
            weights,
        })
    }

    /// Compute Euclidean distance between two points
    fn distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Solve linear system using Gaussian elimination
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> SpatialResult<Array1<f64>> {
        let n = a.nrows();

        // Create augmented matrix
        let mut aug = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(n + 1) {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-12 {
                return Err(SpatialError::ComputationError(
                    "Singular matrix in Kriging system".to_string(),
                ));
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..(n + 1) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut solution = Array1::zeros(n);
        for i in (0..n).rev() {
            solution[i] = aug[[i, n]];
            for j in (i + 1)..n {
                solution[i] -= aug[[i, j]] * solution[j];
            }
            solution[i] /= aug[[i, i]];
        }

        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_variogram_models() {
        let spherical = VariogramModel::spherical(1.0, 0.5, 0.1);

        // At distance 0, should return nugget
        assert_relative_eq!(spherical.evaluate(0.0), 0.1, epsilon = 1e-10);

        // At range, should approach sill + nugget
        assert_relative_eq!(spherical.evaluate(1.0), 0.6, epsilon = 1e-10);

        // Beyond range, should be sill + nugget
        assert_relative_eq!(spherical.evaluate(2.0), 0.6, epsilon = 1e-10);

        let exponential = VariogramModel::exponential(1.0, 0.5, 0.1);
        assert_relative_eq!(exponential.evaluate(0.0), 0.1, epsilon = 1e-10);
        assert!(exponential.evaluate(1.0) > 0.1);
        assert!(exponential.evaluate(1.0) < 0.6);

        let gaussian = VariogramModel::gaussian(1.0, 0.5, 0.1);
        assert_relative_eq!(gaussian.evaluate(0.0), 0.1, epsilon = 1e-10);
        assert!(gaussian.evaluate(1.0) > 0.1);

        let linear = VariogramModel::linear(0.2, 0.05);
        assert_relative_eq!(linear.evaluate(0.0), 0.05, epsilon = 1e-10);
        assert_relative_eq!(linear.evaluate(1.0), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_ordinary_kriging_basic() {
        // Simple 2D case
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let values = arr1(&[1.0, 2.0, 3.0, 4.0]);

        let variogram = VariogramModel::spherical(1.5, 1.0, 0.1);
        let mut kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();
        kriging.fit().unwrap();

        // Predict at center
        let prediction = kriging.predict(&[0.5, 0.5]).unwrap();

        // Should be close to the average of surrounding points
        assert!(prediction.value > 1.0);
        assert!(prediction.value < 4.0);
        assert!(prediction.variance >= 0.0);

        // Weights should sum to 1 (unbiasedness)
        let weight_sum: f64 = prediction.weights.sum();
        assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ordinary_kriging_exact_interpolation() {
        // Test that predictions at data locations are exact
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let values = arr1(&[1.0, 2.0, 3.0]);

        let variogram = VariogramModel::spherical(1.0, 0.5, 0.01); // Small nugget
        let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();

        // Predict at first data point
        let prediction = kriging.predict(&[0.0, 0.0]).unwrap();
        assert_relative_eq!(prediction.value, 1.0, epsilon = 1e-6);

        // Variance should be small at data locations
        assert!(prediction.variance < 0.1);
    }

    #[test]
    fn test_simple_kriging() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let values = arr1(&[1.5, 2.5, 3.5]);
        let mean = 2.0;

        let variogram = VariogramModel::exponential(1.0, 0.8, 0.1);
        let kriging = SimpleKriging::new(&points.view(), &values.view(), mean, variogram).unwrap();

        let prediction = kriging.predict(&[0.5, 0.5]).unwrap();

        // Should give reasonable prediction
        assert!(prediction.value > 1.0);
        assert!(prediction.value < 4.0);
        assert!(prediction.variance >= 0.0);
    }

    #[test]
    fn test_batch_prediction() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let values = arr1(&[1.0, 2.0, 3.0, 4.0]);

        let variogram = VariogramModel::spherical(1.5, 1.0, 0.1);
        let mut kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();
        kriging.fit().unwrap();

        let test_points =
            Array2::from_shape_vec((3, 2), vec![0.25, 0.25, 0.5, 0.5, 0.75, 0.75]).unwrap();

        let predictions = kriging.predict_batch(&test_points.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        for prediction in &predictions {
            assert!(prediction.value > 0.0);
            assert!(prediction.variance >= 0.0);

            // Weights should sum to 1
            let weight_sum: f64 = prediction.weights.sum();
            assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cross_validation() {
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let values = arr1(&[1.0, 2.0, 3.0, 4.0, 2.5]);

        let variogram = VariogramModel::spherical(1.5, 1.0, 0.1);
        let kriging = OrdinaryKriging::new(&points.view(), &values.view(), variogram).unwrap();

        let errors = kriging.cross_validate().unwrap();

        assert_eq!(errors.len(), 5);

        // Errors should be reasonable (not too large)
        for &error in errors.iter() {
            assert!(error.abs() < 5.0); // Reasonable bound for this test case
        }
    }

    #[test]
    fn test_variogram_properties() {
        let spherical = VariogramModel::spherical(2.0, 1.0, 0.2);

        assert_relative_eq!(spherical.effective_range(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(spherical.sill(), 1.2, epsilon = 1e-10);
        assert_relative_eq!(spherical.nugget(), 0.2, epsilon = 1e-10);

        let linear = VariogramModel::linear(0.5, 0.1);
        assert_eq!(linear.effective_range(), f64::INFINITY);
        assert_eq!(linear.sill(), f64::INFINITY);
        assert_relative_eq!(linear.nugget(), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_error_cases() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let values = arr1(&[1.0, 2.0, 3.0]); // Wrong length
        let variogram = VariogramModel::spherical(1.0, 0.5, 0.1);

        let result = OrdinaryKriging::new(&points.view(), &values.view(), variogram);
        assert!(result.is_err());

        // Too few points
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let values = arr1(&[1.0, 2.0]);
        let variogram = VariogramModel::spherical(1.0, 0.5, 0.1);

        let result = OrdinaryKriging::new(&points.view(), &values.view(), variogram);
        assert!(result.is_err());
    }
}
