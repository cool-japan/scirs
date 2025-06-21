//! N-dimensional interpolation methods
//!
//! This module provides functionality for interpolating multidimensional data.

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

/// Available grid types for N-dimensional interpolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridType {
    /// Regular grid (evenly spaced points in each dimension)
    Regular,
    /// Rectilinear grid (unevenly spaced points along each axis)
    Rectilinear,
    /// Unstructured grid (arbitrary point positions)
    Unstructured,
}

/// Extrapolation mode for N-dimensional interpolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolateMode {
    /// Return NaN for points outside the interpolation domain
    Nan,
    /// Raise an error for points outside the interpolation domain
    Error,
    /// Extrapolate based on the nearest edge points
    Extrapolate,
}

/// N-dimensional interpolation object for rectilinear grids
///
/// This interpolator works with data defined on a rectilinear grid,
/// where each dimension has its own set of coordinates.
#[derive(Debug, Clone)]
pub struct RegularGridInterpolator<F: Float + FromPrimitive + Debug + Display> {
    /// Grid points in each dimension
    points: Vec<Array1<F>>,
    /// Values at grid points
    values: Array<F, IxDyn>,
    /// Method to use for interpolation
    method: InterpolationMethod,
    /// How to handle points outside the domain
    extrapolate: ExtrapolateMode,
}

/// Available interpolation methods for N-dimensional interpolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    Linear,
    /// Spline interpolation
    Spline,
}

impl<F: Float + FromPrimitive + Debug + Display> RegularGridInterpolator<F> {
    /// Create a new RegularGridInterpolator
    ///
    /// # Arguments
    ///
    /// * `points` - A vector of arrays, where each array contains the points in one dimension
    /// * `values` - An N-dimensional array of values at the grid points
    /// * `method` - Interpolation method to use
    /// * `extrapolate` - How to handle points outside the domain
    ///
    /// # Returns
    ///
    /// A new RegularGridInterpolator object
    ///
    /// # Errors
    ///
    /// * If points dimensions don't match values dimensions
    /// * If any dimension has less than 2 points
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array, Array1, Dim, IxDyn};
    /// use scirs2_interpolate::interpnd::{
    ///     RegularGridInterpolator, InterpolationMethod, ExtrapolateMode
    /// };
    ///
    /// // Create a 3D grid
    /// let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    /// let y = Array1::from_vec(vec![0.0, 1.0]);
    /// let z = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    /// let points = vec![x, y, z];
    ///
    /// // Create values on the grid (3 × 2 × 4 = 24 values)
    /// let mut values = Array::zeros(IxDyn(&[3, 2, 4]));
    /// for i in 0..3 {
    ///     for j in 0..2 {
    ///         for k in 0..4 {
    ///             let idx = [i, j, k];
    ///             values[idx.as_slice()] = (i + j + k) as f64;
    ///         }
    ///     }
    /// }
    ///
    /// let interpolator = RegularGridInterpolator::new(
    ///     points,
    ///     values,
    ///     InterpolationMethod::Linear,
    ///     ExtrapolateMode::Extrapolate,
    /// ).unwrap();
    /// ```
    pub fn new(
        points: Vec<Array1<F>>,
        values: Array<F, IxDyn>,
        method: InterpolationMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        // Check that points dimensions match values dimensions
        if points.len() != values.ndim() {
            return Err(InterpolateError::ValueError(format!(
                "Points dimensions ({}) do not match values dimensions ({})",
                points.len(),
                values.ndim()
            )));
        }

        // Check that each dimension has at least 2 points
        for (i, p) in points.iter().enumerate() {
            if p.len() < 2 {
                return Err(InterpolateError::ValueError(format!(
                    "Dimension {} has less than 2 points",
                    i
                )));
            }

            // Check that points are sorted
            for j in 1..p.len() {
                if p[j] <= p[j - 1] {
                    return Err(InterpolateError::ValueError(format!(
                        "Points in dimension {} are not strictly increasing",
                        i
                    )));
                }
            }

            // Check that values dimension matches points dimension
            if p.len() != values.shape()[i] {
                return Err(InterpolateError::ValueError(format!(
                    "Values dimension {} size {} does not match points dimension size {}",
                    i,
                    values.shape()[i],
                    p.len()
                )));
            }
        }

        Ok(Self {
            points,
            values,
            method,
            extrapolate,
        })
    }

    /// Interpolate at the given points
    ///
    /// # Arguments
    ///
    /// * `xi` - Array of points to interpolate at, shape (n_points, n_dims)
    ///
    /// # Returns
    ///
    /// Interpolated values at the given points, shape (n_points,)
    ///
    /// # Errors
    ///
    /// * If xi dimensions don't match grid dimensions
    /// * If extrapolation is not allowed and points are outside the domain
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array, Array1, Array2, IxDyn};
    /// use scirs2_interpolate::interpnd::{
    ///     RegularGridInterpolator, InterpolationMethod, ExtrapolateMode
    /// };
    ///
    /// // Create a simple 2D grid
    /// let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    /// let y = Array1::from_vec(vec![0.0, 1.0]);
    /// let points = vec![x, y];
    ///
    /// let mut values = Array::zeros(IxDyn(&[3, 2]));
    /// for i in 0..3 {
    ///     for j in 0..2 {
    ///         let idx = [i, j];
    ///         values[idx.as_slice()] = (i * i + j * j) as f64;
    ///     }
    /// }
    ///
    /// let interpolator = RegularGridInterpolator::new(
    ///     points, values, InterpolationMethod::Linear, ExtrapolateMode::Extrapolate
    /// ).unwrap();
    ///
    /// // Interpolate at multiple points
    /// let xi = Array2::from_shape_vec((3, 2), vec![
    ///     0.5, 0.5,
    ///     1.0, 0.0,
    ///     1.5, 0.5,
    /// ]).unwrap();
    ///
    /// let results = interpolator.__call__(&xi.view()).unwrap();
    /// assert_eq!(results.len(), 3);
    /// ```
    pub fn __call__(&self, xi: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        // Check that xi dimensions match grid dimensions
        if xi.shape()[1] != self.points.len() {
            return Err(InterpolateError::ValueError(format!(
                "Dimensions of interpolation points ({}) do not match grid dimensions ({})",
                xi.shape()[1],
                self.points.len()
            )));
        }

        let n_points = xi.shape()[0];
        let mut result = Array1::zeros(n_points);

        for i in 0..n_points {
            let point = xi.slice(ndarray::s![i, ..]);
            result[i] = self.interpolate_point(&point)?;
        }

        Ok(result)
    }

    /// Interpolate at a single point
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates of the point to interpolate at
    ///
    /// # Returns
    ///
    /// Interpolated value at the given point
    fn interpolate_point(&self, point: &ArrayView1<F>) -> InterpolateResult<F> {
        // Find the grid cells containing the point and calculate the normalized distances
        let mut indices = Vec::with_capacity(self.points.len());
        let mut weights = Vec::with_capacity(self.points.len());

        for (dim, dim_points) in self.points.iter().enumerate() {
            let x = point[dim];

            // Check if point is outside the domain
            if x < dim_points[0] || x > dim_points[dim_points.len() - 1] {
                match self.extrapolate {
                    ExtrapolateMode::Error => {
                        return Err(InterpolateError::DomainError(format!(
                            "Point outside domain in dimension {}: {} not in [{}, {}]",
                            dim,
                            x,
                            dim_points[0],
                            dim_points[dim_points.len() - 1]
                        )));
                    }
                    ExtrapolateMode::Nan => {
                        return Ok(F::nan());
                    }
                    // For extrapolate and constant modes, we'll find the nearest edge
                    _ => {}
                }
            }

            // Find index of cell containing x
            let idx = match self.method {
                InterpolationMethod::Nearest => {
                    // For nearest, just find the closest point
                    let mut closest_idx = 0;
                    let mut min_dist = (x - dim_points[0]).abs();

                    for (j, &p) in dim_points.iter().enumerate().skip(1) {
                        let dist = (x - p).abs();
                        if dist < min_dist {
                            min_dist = dist;
                            closest_idx = j;
                        }
                    }

                    // Return just the index of the nearest point
                    indices.push(closest_idx);
                    weights.push(F::from_f64(1.0).unwrap());
                    continue;
                }
                _ => {
                    // For linear and spline, find the cell interval
                    let mut idx = dim_points.len() - 2;

                    // Find the cell that contains x (where x is between x[idx] and x[idx+1])
                    // Simply iterate through the points to find the right cell
                    let mut found = false;
                    for i in 0..dim_points.len() - 1 {
                        if x >= dim_points[i] && x <= dim_points[i + 1] {
                            idx = i;
                            found = true;
                            break;
                        }
                    }

                    // Handle extrapolation cases
                    if !found {
                        if x < dim_points[0] {
                            // Point is before the first grid point
                            if self.extrapolate == ExtrapolateMode::Extrapolate {
                                idx = 0;
                            } else if self.extrapolate == ExtrapolateMode::Error {
                                return Err(InterpolateError::out_of_domain(
                                    x,
                                    dim_points[0],
                                    dim_points[dim_points.len() - 1],
                                    "N-dimensional interpolation",
                                ));
                            } else {
                                // For Nan mode, clamp to boundary
                                idx = 0;
                            }
                        } else {
                            // Point is after the last grid point
                            idx = dim_points.len() - 2;
                        }
                    }

                    idx
                }
            };

            // For linear interpolation, compute the weights
            if self.method != InterpolationMethod::Nearest {
                // Get the lower and upper bounds of the cell
                let x0 = dim_points[idx];
                let x1 = dim_points[idx + 1];

                // Calculate the normalized distance for linear interpolation
                // t is the fraction of the distance between x0 and x1
                let t = if x1 == x0 {
                    F::from_f64(0.0).unwrap()
                } else {
                    (x - x0) / (x1 - x0)
                };

                // Ensure t is between 0 and 1 (this handles any numerical precision issues)
                let t = t
                    .max(F::from_f64(0.0).unwrap())
                    .min(F::from_f64(1.0).unwrap());

                indices.push(idx);
                weights.push(t);
            }
        }

        // Perform the interpolation based on the method
        match self.method {
            InterpolationMethod::Nearest => {
                // For nearest, we just return the value at the nearest grid point
                let idx_array = indices.to_vec();
                Ok(self.values[idx_array.as_slice()])
            }
            InterpolationMethod::Linear => {
                // For linear, we need to compute a weighted average of the surrounding cell vertices
                self.linear_interpolate(&indices, &weights)
            }
            InterpolationMethod::Spline => Err(InterpolateError::NotImplementedError(
                "Spline interpolation not yet implemented for N-dimensions".to_string(),
            )),
        }
    }

    /// Perform linear interpolation
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of the cell containing the point
    /// * `weights` - Normalized distances within the cell
    ///
    /// # Returns
    ///
    /// Interpolated value
    fn linear_interpolate(&self, indices: &[usize], weights: &[F]) -> InterpolateResult<F> {
        // For linear interpolation, we compute a weighted average of cell vertices
        // Each vertex has a weight that is a product of 1D weights

        // Handle the 2D case directly for better performance and correctness in test cases
        if indices.len() == 2 {
            // 2D case (rectangle)
            let i0 = indices[0];
            let i1 = indices[1];
            let t0 = weights[0];
            let t1 = weights[1];

            // Get the values at the 4 corners
            let idx00 = [i0, i1];
            let idx01 = [i0, i1 + 1];
            let idx10 = [i0 + 1, i1];
            let idx11 = [i0 + 1, i1 + 1];

            let v00 = self.values[idx00.as_slice()];
            let v01 = self.values[idx01.as_slice()];
            let v10 = self.values[idx10.as_slice()];
            let v11 = self.values[idx11.as_slice()];

            // Bilinear interpolation formula
            // (1-t0)(1-t1)v00 + (1-t0)t1v01 + t0(1-t1)v10 + t0t1v11
            let one = F::from_f64(1.0).unwrap();
            let result = (one - t0) * (one - t1) * v00
                + (one - t0) * t1 * v01
                + t0 * (one - t1) * v10
                + t0 * t1 * v11;

            return Ok(result);
        }

        // General case for N dimensions
        let n_dims = indices.len();
        let mut result = F::from_f64(0.0).unwrap();

        // We need to iterate through all 2^n_dims vertices of the hypercube
        // Each vertex is identified by a binary pattern of lower/upper indices
        let n_vertices = 1 << n_dims;

        for vertex in 0..n_vertices {
            // Build the index for this vertex and calculate its weight
            let mut vertex_index = Vec::with_capacity(n_dims);
            let mut vertex_weight = F::from_f64(1.0).unwrap();

            for dim in 0..n_dims {
                let use_upper = (vertex >> dim) & 1 == 1;
                let idx = indices[dim] + if use_upper { 1 } else { 0 };
                vertex_index.push(idx);

                // Weight is either weight (for upper) or (1-weight) for lower
                // For linear interpolation, weights represent normalized positions
                // e.g., weight 0.7 means 70% toward upper point, 30% toward lower point
                let dim_weight = if use_upper {
                    weights[dim]
                } else {
                    F::from_f64(1.0).unwrap() - weights[dim]
                };

                vertex_weight = vertex_weight * dim_weight;
            }

            // Add the weighted value to the result
            let vertex_value = self.values[vertex_index.as_slice()];
            result = result + vertex_weight * vertex_value;
        }

        Ok(result)
    }
}

/// N-dimensional interpolation on unstructured data (scattered points)
///
/// This interpolator works with data defined on scattered points without
/// a regular grid structure, using various methods.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScatteredInterpolator<F: Float + FromPrimitive + Debug + Display> {
    /// Points coordinates, shape (n_points, n_dims)
    points: Array2<F>,
    /// Values at points, shape (n_points,)
    values: Array1<F>,
    /// Method to use for interpolation
    method: ScatteredInterpolationMethod,
    /// How to handle points outside the domain
    extrapolate: ExtrapolateMode,
    /// Additional parameters for specific methods
    params: ScatteredInterpolatorParams<F>,
}

/// Parameters for scattered interpolation methods
#[derive(Debug, Clone)]
pub enum ScatteredInterpolatorParams<F: Float + FromPrimitive + Debug + Display> {
    /// No additional parameters
    None,
    /// Parameters for IDW (Inverse Distance Weighting)
    IDW {
        /// Power parameter for IDW (default: 2.0)
        power: F,
    },
    /// Parameters for RBF (Radial Basis Function)
    RBF {
        /// Epsilon parameter for RBF (default: 1.0)
        epsilon: F,
        /// Type of radial basis function
        rbf_type: RBFType,
    },
}

/// Types of radial basis functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RBFType {
    /// Gaussian: exp(-(εr)²)
    Gaussian,
    /// Multiquadric: sqrt(1 + (εr)²)
    Multiquadric,
    /// Inverse multiquadric: 1/sqrt(1 + (εr)²)
    InverseMultiquadric,
    /// Thin plate spline: (εr)² log(εr)
    ThinPlateSpline,
}

/// Available interpolation methods for scattered data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScatteredInterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Inverse Distance Weighting
    IDW,
    /// Radial Basis Function interpolation
    RBF,
}

impl<F: Float + FromPrimitive + Debug + Display> ScatteredInterpolator<F> {
    /// Create a new ScatteredInterpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Coordinates of sample points, shape (n_points, n_dims)
    /// * `values` - Values at sample points, shape (n_points,)
    /// * `method` - Interpolation method to use
    /// * `extrapolate` - How to handle points outside the domain
    /// * `params` - Additional parameters for specific methods
    ///
    /// # Returns
    ///
    /// A new ScatteredInterpolator object
    ///
    /// # Errors
    ///
    /// * If points and values dimensions don't match
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, Array2};
    /// use scirs2_interpolate::interpnd::{
    ///     ScatteredInterpolator, ScatteredInterpolationMethod,
    ///     ExtrapolateMode, ScatteredInterpolatorParams
    /// };
    ///
    /// // Create scattered 3D data
    /// let points = Array2::from_shape_vec((5, 3), vec![
    ///     0.0, 0.0, 0.0,
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    ///     0.5, 0.5, 0.5,
    /// ]).unwrap();
    /// let values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 1.5]);
    ///
    /// // Create IDW interpolator with custom power
    /// let interpolator = ScatteredInterpolator::new(
    ///     points,
    ///     values,
    ///     ScatteredInterpolationMethod::IDW,
    ///     ExtrapolateMode::Extrapolate,
    ///     Some(ScatteredInterpolatorParams::IDW { power: 3.0 }),
    /// ).unwrap();
    /// ```
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        method: ScatteredInterpolationMethod,
        extrapolate: ExtrapolateMode,
        params: Option<ScatteredInterpolatorParams<F>>,
    ) -> InterpolateResult<Self> {
        // Check that points and values have compatible dimensions
        if points.shape()[0] != values.len() {
            return Err(InterpolateError::ValueError(format!(
                "Number of points ({}) does not match number of values ({})",
                points.shape()[0],
                values.len()
            )));
        }

        // Set default parameters based on method if not provided
        let params = match params {
            Some(p) => p,
            None => match method {
                ScatteredInterpolationMethod::Nearest => ScatteredInterpolatorParams::None,
                ScatteredInterpolationMethod::IDW => ScatteredInterpolatorParams::IDW {
                    power: F::from_f64(2.0).unwrap(),
                },
                ScatteredInterpolationMethod::RBF => ScatteredInterpolatorParams::RBF {
                    epsilon: F::from_f64(1.0).unwrap(),
                    rbf_type: RBFType::Multiquadric,
                },
            },
        };

        Ok(Self {
            points,
            values,
            method,
            extrapolate,
            params,
        })
    }

    /// Interpolate at the given points
    ///
    /// # Arguments
    ///
    /// * `xi` - Array of points to interpolate at, shape (n_points, n_dims)
    ///
    /// # Returns
    ///
    /// Interpolated values at the given points, shape (n_points,)
    ///
    /// # Errors
    ///
    /// * If xi dimensions don't match input dimensions
    pub fn __call__(&self, xi: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        // Check that xi dimensions match input dimensions
        if xi.shape()[1] != self.points.shape()[1] {
            return Err(InterpolateError::ValueError(format!(
                "Dimensions of interpolation points ({}) do not match input dimensions ({})",
                xi.shape()[1],
                self.points.shape()[1]
            )));
        }

        let n_points = xi.shape()[0];
        let mut result = Array1::zeros(n_points);

        for i in 0..n_points {
            let point = xi.slice(ndarray::s![i, ..]);
            result[i] = self.interpolate_point(&point)?;
        }

        Ok(result)
    }

    /// Interpolate at a single point
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates of the point to interpolate at
    ///
    /// # Returns
    ///
    /// Interpolated value at the given point
    fn interpolate_point(&self, point: &ArrayView1<F>) -> InterpolateResult<F> {
        match self.method {
            ScatteredInterpolationMethod::Nearest => self.nearest_interpolate(point),
            ScatteredInterpolationMethod::IDW => self.idw_interpolate(point),
            ScatteredInterpolationMethod::RBF => Err(InterpolateError::NotImplementedError(
                "RBF interpolation not yet fully implemented".to_string(),
            )),
        }
    }

    /// Perform nearest neighbor interpolation
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates of the point to interpolate at
    ///
    /// # Returns
    ///
    /// Interpolated value at the given point
    fn nearest_interpolate(&self, point: &ArrayView1<F>) -> InterpolateResult<F> {
        let mut min_dist = F::infinity();
        let mut nearest_idx = 0;

        // Find the nearest point
        for i in 0..self.points.shape()[0] {
            let p = self.points.slice(ndarray::s![i, ..]);
            let dist = self.compute_distance(&p, point);

            if dist < min_dist {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        Ok(self.values[nearest_idx])
    }

    /// Perform Inverse Distance Weighting interpolation
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates of the point to interpolate at
    ///
    /// # Returns
    ///
    /// Interpolated value at the given point
    fn idw_interpolate(&self, point: &ArrayView1<F>) -> InterpolateResult<F> {
        // Get the power parameter
        let power = match self.params {
            ScatteredInterpolatorParams::IDW { power } => power,
            _ => F::from_f64(2.0).unwrap(), // Default to 2.0 if wrong params
        };

        let mut sum_weights = F::from_f64(0.0).unwrap();
        let mut sum_weighted_values = F::from_f64(0.0).unwrap();

        // Check for exact match with any input point
        for i in 0..self.points.shape()[0] {
            let p = self.points.slice(ndarray::s![i, ..]);
            let dist = self.compute_distance(&p, point);

            if dist.is_zero() {
                // Exact match found
                return Ok(self.values[i]);
            }

            // Calculate weight as 1/distance^power
            let weight = F::from_f64(1.0).unwrap() / dist.powf(power);
            sum_weights = sum_weights + weight;
            sum_weighted_values = sum_weighted_values + weight * self.values[i];
        }

        // Calculate weighted average
        if sum_weights.is_zero() {
            // This should not happen with non-zero distances
            return Err(InterpolateError::ComputationError(
                "Sum of weights is zero in IDW interpolation".to_string(),
            ));
        }

        Ok(sum_weighted_values / sum_weights)
    }

    /// Compute Euclidean distance between two points
    ///
    /// # Arguments
    ///
    /// * `p1` - First point
    /// * `p2` - Second point
    ///
    /// # Returns
    ///
    /// Euclidean distance between the points
    fn compute_distance(&self, p1: &ArrayView1<F>, p2: &ArrayView1<F>) -> F {
        let mut sum_sq = F::from_f64(0.0).unwrap();
        for i in 0..p1.len() {
            let diff = p1[i] - p2[i];
            sum_sq = sum_sq + diff * diff;
        }
        sum_sq.sqrt()
    }
}

/// Create an N-dimensional interpolator on a regular grid
///
/// # Arguments
///
/// * `points` - A vector of arrays, where each array contains the points in one dimension
/// * `values` - An N-dimensional array of values at the grid points
/// * `method` - Interpolation method to use
/// * `extrapolate` - How to handle points outside the domain
///
/// # Returns
///
/// A new RegularGridInterpolator object
///
/// # Errors
///
/// * If points dimensions don't match values dimensions
/// * If any dimension has less than 2 points
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Dim, IxDyn};
/// use num_traits::Float;
/// use scirs2_interpolate::interpnd::{
///     make_interp_nd, InterpolationMethod, ExtrapolateMode
/// };
///
/// // Create a 2D grid
/// let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
/// let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let points = vec![x, y];
///
/// // Create values on the grid (z = x^2 + y^2)
/// let mut values = Array::zeros(IxDyn(&[3, 4]));
/// for i in 0..3 {
///     for j in 0..4 {
///         let idx = [i, j];
///         values[idx.as_slice()] = (i * i + j * j) as f64;
///     }
/// }
///
/// // Create the interpolator
/// let interp = make_interp_nd(
///     points,
///     values,
///     InterpolationMethod::Linear,
///     ExtrapolateMode::Extrapolate,
/// ).unwrap();
///
/// // Interpolate at a point
/// use ndarray::Array2;
/// let points_to_interp = Array2::from_shape_vec((1, 2), vec![1.5, 2.5]).unwrap();
/// let result = interp.__call__(&points_to_interp.view()).unwrap();
/// assert!((result[0] - 9.0).abs() < 1e-10);
/// ```
pub fn make_interp_nd<F: Float + FromPrimitive + Debug + Display>(
    points: Vec<Array1<F>>,
    values: Array<F, IxDyn>,
    method: InterpolationMethod,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<RegularGridInterpolator<F>> {
    RegularGridInterpolator::new(points, values, method, extrapolate)
}

/// Create an N-dimensional interpolator for scattered data
///
/// # Arguments
///
/// * `points` - Coordinates of sample points, shape (n_points, n_dims)
/// * `values` - Values at sample points, shape (n_points,)
/// * `method` - Interpolation method to use
/// * `extrapolate` - How to handle points outside the domain
/// * `params` - Additional parameters for specific methods
///
/// # Returns
///
/// A new ScatteredInterpolator object
///
/// # Errors
///
/// * If points and values dimensions don't match
pub fn make_interp_scattered<F: Float + FromPrimitive + Debug + Display>(
    points: Array2<F>,
    values: Array1<F>,
    method: ScatteredInterpolationMethod,
    extrapolate: ExtrapolateMode,
    params: Option<ScatteredInterpolatorParams<F>>,
) -> InterpolateResult<ScatteredInterpolator<F>> {
    ScatteredInterpolator::new(points, values, method, extrapolate, params)
}

/// Map values on a rectilinear grid to a new grid
///
/// # Arguments
///
/// * `old_grid` - Vec of Arrays representing the old grid points in each dimension
/// * `old_values` - Values at old grid points
/// * `new_grid` - Vec of Arrays representing the new grid points in each dimension
/// * `method` - Interpolation method to use
///
/// # Returns
///
/// Values at new grid points
///
/// # Errors
///
/// * If dimensions don't match
/// * If any dimension has less than 2 points
pub fn map_coordinates<F: Float + FromPrimitive + Debug + Display>(
    old_grid: Vec<Array1<F>>,
    old_values: Array<F, IxDyn>,
    new_grid: Vec<Array1<F>>,
    method: InterpolationMethod,
) -> InterpolateResult<Array<F, IxDyn>> {
    // Create the interpolator
    let interp =
        RegularGridInterpolator::new(old_grid, old_values, method, ExtrapolateMode::Error)?;

    // Determine the shape of the output array
    let out_shape: Vec<usize> = new_grid.iter().map(|x| x.len()).collect();
    let n_dims = out_shape.len();

    // Create meshgrid of coordinates
    let mut indices = vec![Vec::<F>::new(); n_dims];
    let mut shape = vec![1; n_dims];

    for (i, grid) in new_grid.iter().enumerate() {
        let mut idx = vec![F::from_f64(0.0).unwrap(); grid.len()];
        for (j, val) in grid.iter().enumerate() {
            idx[j] = *val;
        }
        indices[i] = idx;
        shape[i] = grid.len();
    }

    // Calculate total number of points
    let total_points: usize = shape.iter().product();

    // Create the output array
    let mut out_values = Array::zeros(IxDyn(&out_shape));

    // Create a 2D array of all points to interpolate
    let mut points = Array2::zeros((total_points, n_dims));

    // Create a multi-index for traversing the grid
    let mut multi_index = vec![0; n_dims];

    for flat_idx in 0..total_points {
        // Convert flat index to multi-index
        let mut temp = flat_idx;
        for i in (0..n_dims).rev() {
            multi_index[i] = temp % shape[i];
            temp /= shape[i];
        }

        // Set point coordinates
        for i in 0..n_dims {
            points[[flat_idx, i]] = indices[i][multi_index[i]];
        }
    }

    // Perform interpolation for all points
    let values = interp.__call__(&points.view())?;

    // Reshape the result to match the output grid
    let mut out_idx_vec = Vec::with_capacity(n_dims);
    for flat_idx in 0..total_points {
        // Convert flat index to multi-index
        let mut temp = flat_idx;
        for i in (0..n_dims).rev() {
            multi_index[i] = temp % shape[i];
            temp /= shape[i];
        }

        // Convert multi-index to output index vector
        out_idx_vec.clear();
        out_idx_vec.extend_from_slice(&multi_index[..n_dims]);

        // Set the value in the output array
        *out_values.get_mut(out_idx_vec.as_slice()).unwrap() = values[flat_idx];
    }

    Ok(out_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, IxDyn}; // 配列操作用

    #[test]
    fn test_regular_grid_interpolator_2d() {
        // Create a 2D grid
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let points = vec![x, y];

        // Create values on the grid (z = x^2 + y^2)
        let mut values = Array::zeros(IxDyn(&[3, 4]));
        for i in 0..3 {
            for j in 0..4 {
                let idx = [i, j];
                values[idx.as_slice()] = (i * i + j * j) as f64;
            }
        }

        // Create the interpolator
        let interp = RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // Test interpolation at grid points
        let grid_point = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let result = interp.__call__(&grid_point.view()).unwrap();
        assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-10);

        // Test interpolation at non-grid points
        let non_grid_point = Array2::from_shape_vec((1, 2), vec![1.5, 2.5]).unwrap();
        let result = interp.__call__(&non_grid_point.view()).unwrap();

        // For point (1.5, 2.5):
        // We're interpolating between grid points:
        // (1,2) -> value = 5.0
        // (1,3) -> value = 10.0
        // (2,2) -> value = 8.0
        // (2,3) -> value = 13.0
        // With weights: x=0.5, y=0.5
        // Expected = (1-0.5)(1-0.5)*5.0 + (1-0.5)(0.5)*10.0 + (0.5)(1-0.5)*8.0 + (0.5)(0.5)*13.0
        //          = 0.25*5.0 + 0.25*10.0 + 0.25*8.0 + 0.25*13.0
        //          = 1.25 + 2.5 + 2.0 + 3.25 = 9.0
        assert_abs_diff_eq!(result[0], 9.0, epsilon = 1e-10);

        // Test multiple points at once
        let multiple_points = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let result = interp.__call__(&multiple_points.view()).unwrap();
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 8.0, epsilon = 1e-10);

        // Test nearest neighbor interpolation
        let interp_nearest = RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            InterpolationMethod::Nearest,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        let point = Array2::from_shape_vec((1, 2), vec![1.6, 1.7]).unwrap();
        let result = interp_nearest.__call__(&point.view()).unwrap();
        // Point (1.6, 1.7) is closest to grid point (2,2) which has value 8.0
        assert_abs_diff_eq!(result[0], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scattered_interpolator() {
        // Create scattered points in 2D
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();

        // Create values at those points (z = x^2 + y^2)
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 0.5]);

        // Create the interpolator with IDW
        let interp = ScatteredInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredInterpolationMethod::IDW,
            ExtrapolateMode::Extrapolate,
            Some(ScatteredInterpolatorParams::IDW { power: 2.0 }),
        )
        .unwrap();

        // Test interpolation at a point
        let test_point = Array2::from_shape_vec((1, 2), vec![0.5, 0.0]).unwrap();
        let result = interp.__call__(&test_point.view()).unwrap();
        // Value should be between 0.0 and 1.0, closer to 0.5
        assert!(result[0] > 0.0 && result[0] < 1.0);

        // Test nearest neighbor interpolator
        let interp_nearest = ScatteredInterpolator::new(
            points,
            values,
            ScatteredInterpolationMethod::Nearest,
            ExtrapolateMode::Extrapolate,
            None,
        )
        .unwrap();

        let test_point = Array2::from_shape_vec((1, 2), vec![0.6, 0.6]).unwrap();
        let result = interp_nearest.__call__(&test_point.view()).unwrap();
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10); // Should pick the center point
    }
}
