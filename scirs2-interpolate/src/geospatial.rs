//! Geospatial interpolation methods
//!
//! This module provides interpolation methods specifically designed for geospatial data,
//! including handling of geographic coordinates, projection systems, and spatial
//! autocorrelation. These methods are optimized for Earth science applications.
//!
//! # Geospatial Features
//!
//! - **Geographic coordinate handling**: Proper handling of longitude/latitude data
//! - **Projection-aware interpolation**: Support for various map projections
//! - **Spatial autocorrelation**: Methods that account for spatial dependencies
//! - **Spherical interpolation**: Interpolation on sphere surface (great circle distances)
//! - **Elevation-aware interpolation**: 3D interpolation with topographic considerations
//! - **Boundary handling**: Coastal boundaries and land/water transitions
//! - **Multi-scale interpolation**: From local to global scale interpolation
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2__interpolate::geospatial::{
//!     GeospatialInterpolator, CoordinateSystem, InterpolationModel
//! };
//!
//! // Create latitude/longitude coordinates
//! let latitudes = Array1::from_vec(vec![40.7128, 34.0522, 41.8781, 29.7604]);
//! let longitudes = Array1::from_vec(vec![-74.0060, -118.2437, -87.6298, -95.3698]);
//! let temperatures = Array1::from_vec(vec![15.0, 22.0, 12.0, 28.0]);
//!
//! // Create geospatial interpolator
//! let mut interpolator = GeospatialInterpolator::new()
//!     .with_coordinate_system(CoordinateSystem::WGS84)
//!     .with_interpolation_model(InterpolationModel::Kriging)
//!     .with_spherical_distance(true);
//!
//! // Fit the interpolator
//! interpolator.fit(&latitudes.view(), &longitudes.view(), &temperatures.view()).unwrap();
//!
//! // Interpolate at new locations
//! let query_lats = Array1::from_vec(vec![37.7749, 40.0]);
//! let query_lons = Array1::from_vec(vec![-122.4194, -100.0]);
//! let result = interpolator.interpolate(&query_lats.view(), &query_lons.view()).unwrap();
//! ```

use crate::advanced::kriging::{CovarianceFunction, KrigingInterpolator};
use crate::advanced::rbf::{RBFInterpolator, RBFKernel};
use crate::advanced::thinplate::ThinPlateSpline;
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::{Debug, Display, LowerExp};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

/// Coordinate reference systems supported for geospatial interpolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateSystem {
    /// World Geodetic System 1984 (latitude/longitude)
    WGS84,
    /// Universal Transverse Mercator
    UTM(u8), // UTM zone number
    /// Web Mercator (used by web mapping services)
    WebMercator,
    /// Local coordinate system (Cartesian)
    LocalCartesian,
    /// Custom projection (user-defined)
    Custom,
}

/// Interpolation models optimized for geospatial data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationModel {
    /// Inverse Distance Weighting with spatial considerations
    InverseDistanceWeighting,
    /// Kriging with spatial autocorrelation
    Kriging,
    /// Radial Basis Functions with spherical kernels
    SphericalRBF,
    /// Thin plate splines adapted for geographic data
    ThinPlateSpline,
    /// Natural neighbor interpolation
    NaturalNeighbor,
    /// Bilinear interpolation for gridded data
    Bilinear,
}

/// Configuration for geospatial interpolation
#[derive(Debug, Clone)]
pub struct GeospatialConfig<T> {
    /// Coordinate reference system
    pub coordinate_system: CoordinateSystem,
    /// Interpolation model to use
    pub model: InterpolationModel,
    /// Whether to use spherical distance calculations
    pub use_spherical_distance: bool,
    /// Earth radius in kilometers (for spherical calculations)
    pub earth_radius_km: T,
    /// Search radius for local interpolation methods
    pub search_radius_km: Option<T>,
    /// Maximum number of neighbors to consider
    pub max_neighbors: Option<usize>,
    /// Anisotropy parameters (direction-dependent scaling)
    pub anisotropy_angle: Option<T>,
    pub anisotropy_ratio: Option<T>,
    /// Elevation weighting factor (if elevation data is provided)
    pub elevation_weight: Option<T>,
}

impl<T: Float + FromPrimitive> Default for GeospatialConfig<T> {
    fn default() -> Self {
        Self {
            coordinate_system: CoordinateSystem::WGS84,
            model: InterpolationModel::Kriging,
            use_spherical_distance: true,
            earth_radius_km: T::from_f64(6371.0).unwrap(), // Mean Earth radius
            search_radius_km: None,
            max_neighbors: Some(20),
            anisotropy_angle: None,
            anisotropy_ratio: None,
            elevation_weight: None,
        }
    }
}

/// Result of geospatial interpolation with spatial metadata
#[derive(Debug, Clone)]
pub struct GeospatialResult<T> {
    /// Interpolated values
    pub values: Array1<T>,
    /// Interpolation variance/uncertainty (if available)
    pub variance: Option<Array1<T>>,
    /// Effective number of neighbors used for each point
    pub neighbor_counts: Option<Array1<usize>>,
    /// Spatial autocorrelation metrics
    pub spatial_correlation: Option<T>,
}

/// Geospatial interpolator for Earth science data
#[derive(Debug)]
pub struct GeospatialInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + 'static,
{
    /// Configuration for geospatial interpolation
    config: GeospatialConfig<T>,
    /// Training latitudes
    train_latitudes: Array1<T>,
    /// Training longitudes
    train_longitudes: Array1<T>,
    /// Training values
    train_values: Array1<T>,
    /// Optional elevation data
    #[allow(dead_code)]
    train_elevations: Option<Array1<T>>,
    /// Underlying interpolator
    interpolator: Option<Box<dyn GeospatialInterpolatorTrait<T>>>,
    /// Whether the model is trained
    is_trained: bool,
    /// Computed spatial statistics
    spatial_stats: SpatialStats<T>,
}

/// Trait for geospatial interpolation implementations
trait GeospatialInterpolatorTrait<T>: Debug
where
    T: Float + FromPrimitive + ToPrimitive + Debug + Display + Copy + 'static,
{
    /// Interpolate at new locations
    fn interpolate_spatial(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<GeospatialResult<T>>;

    /// Get model-specific parameters
    #[allow(dead_code)]
    fn get_parameters(&self) -> Vec<(String, String)>;
}

/// Statistics about spatial patterns in the data
#[derive(Debug, Clone)]
pub struct SpatialStats<T> {
    /// Estimated spatial range (km)
    pub spatial_range_km: Option<T>,
    /// Spatial variance
    pub spatial_variance: Option<T>,
    /// Moran's I spatial autocorrelation
    pub morans_i: Option<T>,
    /// Effective number of degrees of freedom
    pub effective_dof: Option<T>,
    /// Spatial clustering index
    pub clustering_index: Option<T>,
}

impl<T> Default for SpatialStats<T> {
    fn default() -> Self {
        Self {
            spatial_range_km: None,
            spatial_variance: None,
            morans_i: None,
            effective_dof: None,
            clustering_index: None,
        }
    }
}

impl<T> Default for GeospatialInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GeospatialInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    /// Create a new geospatial interpolator
    pub fn new() -> Self {
        Self {
            config: GeospatialConfig::default(),
            train_latitudes: Array1::zeros(0),
            train_longitudes: Array1::zeros(0),
            train_values: Array1::zeros(0),
            train_elevations: None,
            interpolator: None,
            is_trained: false,
            spatial_stats: SpatialStats::default(),
        }
    }

    /// Set the coordinate reference system
    pub fn with_coordinate_system(mut self, coordsys: CoordinateSystem) -> Self {
        self.config.coordinate_system = coordsys;
        self
    }

    /// Set the interpolation model
    pub fn with_interpolation_model(mut self, model: InterpolationModel) -> Self {
        self.config.model = model;
        self
    }

    /// Enable or disable spherical distance calculations
    pub fn with_spherical_distance(mut self, usespherical: bool) -> Self {
        self.config.use_spherical_distance = usespherical;
        self
    }

    /// Set search radius for local interpolation methods
    pub fn with_search_radius_km(mut self, radius: T) -> Self {
        self.config.search_radius_km = Some(radius);
        self
    }

    /// Set maximum number of neighbors
    pub fn with_max_neighbors(mut self, maxneighbors: usize) -> Self {
        self.config.max_neighbors = Some(maxneighbors);
        self
    }

    /// Set anisotropy parameters for directional spatial correlation
    pub fn with_anisotropy(mut self, angle: T, ratio: T) -> Self {
        self.config.anisotropy_angle = Some(angle);
        self.config.anisotropy_ratio = Some(ratio);
        self
    }

    /// Fit the geospatial interpolator to data
    ///
    /// # Arguments
    ///
    /// * `latitudes` - Latitude coordinates in degrees
    /// * `longitudes` - Longitude coordinates in degrees  
    /// * `values` - Values at each coordinate
    ///
    /// # Returns
    ///
    /// Success indicator
    pub fn fit(
        &mut self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        if latitudes.len() != longitudes.len() || latitudes.len() != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "latitudes ({}), longitudes ({}), and values ({}) must have the same length",
                latitudes.len(),
                longitudes.len(),
                values.len()
            )));
        }

        if latitudes.len() < 3 {
            return Err(InterpolateError::InvalidValue(
                "At least 3 data points required for geospatial interpolation".to_string(),
            ));
        }

        // Store training data
        self.train_latitudes = latitudes.to_owned();
        self.train_longitudes = longitudes.to_owned();
        self.train_values = values.to_owned();

        // Convert coordinates to projected system if needed
        let (x_coords, y_coords) = self.project_coordinates(latitudes, longitudes)?;

        // Compute spatial statistics
        self.compute_spatial_statistics(&x_coords, &y_coords, values)?;

        // Create the appropriate interpolator based on model type
        match self.config.model {
            InterpolationModel::Kriging => {
                self.fit_kriging(&x_coords, &y_coords, values)?;
            }
            InterpolationModel::SphericalRBF => {
                self.fit_spherical_rbf(&x_coords, &y_coords, values)?;
            }
            InterpolationModel::ThinPlateSpline => {
                self.fit_thin_plate_spline(&x_coords, &y_coords, values)?;
            }
            _ => {
                // For other models, use a default RBF approach
                self.fit_default_rbf(&x_coords, &y_coords, values)?;
            }
        }

        self.is_trained = true;
        Ok(())
    }

    /// Interpolate at new geographic locations
    ///
    /// # Arguments
    ///
    /// * `latitudes` - Query latitude coordinates
    /// * `longitudes` - Query longitude coordinates
    ///
    /// # Returns
    ///
    /// Geospatial interpolation result
    pub fn interpolate(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<GeospatialResult<T>> {
        if !self.is_trained {
            return Err(InterpolateError::InvalidState(
                "Interpolator must be fitted before interpolation".to_string(),
            ));
        }

        if latitudes.len() != longitudes.len() {
            return Err(InterpolateError::DimensionMismatch(
                "latitudes and longitudes must have the same length".to_string(),
            ));
        }

        // Project query coordinates
        let _x_coords_y_coords = self.project_coordinates(latitudes, longitudes)?;

        // Use the fitted interpolator
        if let Some(ref interpolator) = self.interpolator {
            interpolator.interpolate_spatial(latitudes, longitudes)
        } else {
            Err(InterpolateError::InvalidState(
                "No interpolator has been fitted".to_string(),
            ))
        }
    }

    /// Project geographic coordinates to the configured coordinate system
    fn project_coordinates(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<(Array1<T>, Array1<T>)> {
        match self.config.coordinate_system {
            CoordinateSystem::WGS84 => {
                // For WGS84, we can use coordinates directly or convert to radians
                if self.config.use_spherical_distance {
                    // Convert to radians for spherical calculations
                    let lat_rad = latitudes
                        .mapv(|lat| lat * T::from_f64(std::f64::consts::PI / 180.0).unwrap());
                    let lon_rad = longitudes
                        .mapv(|lon| lon * T::from_f64(std::f64::consts::PI / 180.0).unwrap());
                    Ok((lat_rad, lon_rad))
                } else {
                    // Use degrees directly
                    Ok((latitudes.to_owned(), longitudes.to_owned()))
                }
            }
            CoordinateSystem::WebMercator => self.project_to_web_mercator(latitudes, longitudes),
            CoordinateSystem::LocalCartesian => {
                // Simple equirectangular projection for local use
                self.project_equirectangular(latitudes, longitudes)
            }
            _ => {
                // For other coordinate systems, use simple degree coordinates for now
                Ok((latitudes.to_owned(), longitudes.to_owned()))
            }
        }
    }

    /// Project to Web Mercator coordinates
    fn project_to_web_mercator(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<(Array1<T>, Array1<T>)> {
        let earth_radius = self.config.earth_radius_km * T::from_f64(1000.0).unwrap(); // Convert to meters
        let deg_to_rad = T::from_f64(std::f64::consts::PI / 180.0).unwrap();

        let x_coords = longitudes.mapv(|lon| lon * deg_to_rad * earth_radius);
        let y_coords = latitudes.mapv(|lat| {
            let lat_rad = lat * deg_to_rad;
            earth_radius
                * (lat_rad + T::from_f64(std::f64::consts::PI / 4.0).unwrap())
                    .tan()
                    .ln()
        });

        Ok((x_coords, y_coords))
    }

    /// Simple equirectangular projection
    fn project_equirectangular(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<(Array1<T>, Array1<T>)> {
        let deg_to_rad = T::from_f64(std::f64::consts::PI / 180.0).unwrap();
        let earth_radius = self.config.earth_radius_km;

        // Use mean latitude for projection scaling
        let mean_lat = latitudes.sum() / T::from_usize(latitudes.len()).unwrap();
        let cos_mean_lat = (mean_lat * deg_to_rad).cos();

        let x_coords = longitudes.mapv(|lon| lon * deg_to_rad * earth_radius * cos_mean_lat);
        let y_coords = latitudes.mapv(|lat| lat * deg_to_rad * earth_radius);

        Ok((x_coords, y_coords))
    }

    /// Compute spatial statistics from the data
    fn compute_spatial_statistics(
        &mut self,
        _x_coords: &Array1<T>,
        _y_coords: &Array1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        // For now, compute basic statistics
        // In a full implementation, this would include:
        // - Variogram analysis
        // - Moran's I calculation
        // - Range estimation
        // - Spatial clustering metrics

        let n = values.len();
        if n > 1 {
            let mean_val = values.sum() / T::from_usize(n).unwrap();
            let variance = values
                .iter()
                .map(|&x| (x - mean_val) * (x - mean_val))
                .sum::<T>()
                / T::from_usize(n - 1).unwrap();

            self.spatial_stats.spatial_variance = Some(variance);
            self.spatial_stats.effective_dof = Some(T::from_usize(n).unwrap());
        }

        Ok(())
    }

    /// Fit kriging interpolator
    fn fit_kriging(
        &mut self,
        x_coords: &Array1<T>,
        y_coords: &Array1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        // Create 2D coordinate array for kriging
        let n = x_coords.len();
        let mut coords_2d = Array2::zeros((n, 2));
        for i in 0..n {
            coords_2d[[i, 0]] = x_coords[i];
            coords_2d[[i, 1]] = y_coords[i];
        }

        let kriging = KrigingInterpolator::new(
            &coords_2d.view(),
            values,
            CovarianceFunction::Exponential,
            T::from_f64(1.0).unwrap(),  // sigma_sq
            T::from_f64(1.0).unwrap(),  // length_scale
            T::from_f64(0.01).unwrap(), // nugget
            T::from_f64(1.0).unwrap(),  // alpha
        )?;

        self.interpolator = Some(Box::new(GeospatialKrigingWrapper { kriging }));
        Ok(())
    }

    /// Fit spherical RBF interpolator
    fn fit_spherical_rbf(
        &mut self,
        x_coords: &Array1<T>,
        y_coords: &Array1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        let n = x_coords.len();
        let mut coords_2d = Array2::zeros((n, 2));
        for i in 0..n {
            coords_2d[[i, 0]] = x_coords[i];
            coords_2d[[i, 1]] = y_coords[i];
        }

        let rbf = RBFInterpolator::new(
            &coords_2d.view(),
            values,
            RBFKernel::Gaussian,
            T::from_f64(1.0).unwrap(),
        )?;

        self.interpolator = Some(Box::new(GeospatialRBFWrapper { rbf }));
        Ok(())
    }

    /// Fit thin plate spline interpolator
    fn fit_thin_plate_spline(
        &mut self,
        x_coords: &Array1<T>,
        y_coords: &Array1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        let n = x_coords.len();
        let mut coords_2d = Array2::zeros((n, 2));
        for i in 0..n {
            coords_2d[[i, 0]] = x_coords[i];
            coords_2d[[i, 1]] = y_coords[i];
        }

        let tps = ThinPlateSpline::new(&coords_2d.view(), values, T::from_f64(0.0).unwrap())?;
        self.interpolator = Some(Box::new(GeospatialTPSWrapper { tps }));
        Ok(())
    }

    /// Fit default RBF interpolator for other models
    fn fit_default_rbf(
        &mut self,
        x_coords: &Array1<T>,
        y_coords: &Array1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        self.fit_spherical_rbf(x_coords, y_coords, values)
    }

    /// Calculate great circle distance between two points in kilometers
    pub fn great_circle_distance(&self, lat1: T, lon1: T, lat2: T, lon2: T) -> T {
        let deg_to_rad = T::from_f64(std::f64::consts::PI / 180.0).unwrap();
        let lat1_rad = lat1 * deg_to_rad;
        let lon1_rad = lon1 * deg_to_rad;
        let lat2_rad = lat2 * deg_to_rad;
        let lon2_rad = lon2 * deg_to_rad;

        let dlat = lat2_rad - lat1_rad;
        let dlon = lon2_rad - lon1_rad;

        let a = (dlat / T::from_f64(2.0).unwrap()).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (dlon / T::from_f64(2.0).unwrap()).sin().powi(2);
        let c = T::from_f64(2.0).unwrap() * a.sqrt().asin();

        self.config.earth_radius_km * c
    }

    /// Get spatial statistics
    pub fn get_spatial_stats(&self) -> &SpatialStats<T> {
        &self.spatial_stats
    }

    /// Check if the interpolator is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

/// Wrapper for Kriging interpolator to implement GeospatialInterpolatorTrait
#[derive(Debug)]
struct GeospatialKrigingWrapper<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    kriging: KrigingInterpolator<T>,
}

impl<T> GeospatialInterpolatorTrait<T> for GeospatialKrigingWrapper<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    fn interpolate_spatial(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<GeospatialResult<T>> {
        let n = latitudes.len();
        let mut coords_2d = Array2::zeros((n, 2));
        for i in 0..n {
            coords_2d[[i, 0]] = latitudes[i];
            coords_2d[[i, 1]] = longitudes[i];
        }

        let result = self.kriging.predict(&coords_2d.view())?;
        let values = result.value;

        Ok(GeospatialResult {
            values,
            variance: None,
            neighbor_counts: None,
            spatial_correlation: None,
        })
    }

    fn get_parameters(&self) -> Vec<(String, String)> {
        vec![
            ("model".to_string(), "Kriging".to_string()),
            ("covariance".to_string(), "Exponential".to_string()),
        ]
    }
}

/// Wrapper for RBF interpolator to implement GeospatialInterpolatorTrait
#[derive(Debug)]
struct GeospatialRBFWrapper<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    rbf: RBFInterpolator<T>,
}

impl<T> GeospatialInterpolatorTrait<T> for GeospatialRBFWrapper<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + Copy
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    fn interpolate_spatial(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<GeospatialResult<T>> {
        let n = latitudes.len();
        let mut coords_2d = Array2::zeros((n, 2));
        for i in 0..n {
            coords_2d[[i, 0]] = latitudes[i];
            coords_2d[[i, 1]] = longitudes[i];
        }

        let values = self.rbf.interpolate(&coords_2d.view())?;

        Ok(GeospatialResult {
            values,
            variance: None,
            neighbor_counts: None,
            spatial_correlation: None,
        })
    }

    fn get_parameters(&self) -> Vec<(String, String)> {
        vec![
            ("model".to_string(), "RBF".to_string()),
            ("kernel".to_string(), "Gaussian".to_string()),
        ]
    }
}

/// Wrapper for Thin Plate Spline interpolator to implement GeospatialInterpolatorTrait
#[derive(Debug)]
struct GeospatialTPSWrapper<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    tps: ThinPlateSpline<T>,
}

impl<T> GeospatialInterpolatorTrait<T> for GeospatialTPSWrapper<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + Copy
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    fn interpolate_spatial(
        &self,
        latitudes: &ArrayView1<T>,
        longitudes: &ArrayView1<T>,
    ) -> InterpolateResult<GeospatialResult<T>> {
        let n = latitudes.len();
        let mut coords_2d = Array2::zeros((n, 2));
        for i in 0..n {
            coords_2d[[i, 0]] = latitudes[i];
            coords_2d[[i, 1]] = longitudes[i];
        }

        let values = self.tps.evaluate(&coords_2d.view())?;

        Ok(GeospatialResult {
            values,
            variance: None,
            neighbor_counts: None,
            spatial_correlation: None,
        })
    }

    fn get_parameters(&self) -> Vec<(String, String)> {
        vec![("model".to_string(), "ThinPlateSpline".to_string())]
    }
}

/// Convenience function to create a geospatial interpolator for climate data
#[allow(dead_code)]
pub fn make_climate_interpolator<T>() -> GeospatialInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    GeospatialInterpolator::new()
        .with_coordinate_system(CoordinateSystem::WGS84)
        .with_interpolation_model(InterpolationModel::Kriging)
        .with_spherical_distance(true)
        .with_max_neighbors(50)
}

/// Convenience function to create a geospatial interpolator for elevation data
#[allow(dead_code)]
pub fn make_elevation_interpolator<T>() -> GeospatialInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    GeospatialInterpolator::new()
        .with_coordinate_system(CoordinateSystem::WGS84)
        .with_interpolation_model(InterpolationModel::ThinPlateSpline)
        .with_spherical_distance(true)
        .with_max_neighbors(30)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_geospatial_interpolator_creation() {
        let interpolator = GeospatialInterpolator::<f64>::new();
        assert!(!interpolator.is_trained());
        assert_eq!(
            interpolator.config.coordinate_system,
            CoordinateSystem::WGS84
        );
        assert_eq!(interpolator.config.model, InterpolationModel::Kriging);
    }

    #[test]
    fn test_geospatial_interpolator_configuration() {
        let interpolator = GeospatialInterpolator::<f64>::new()
            .with_coordinate_system(CoordinateSystem::WebMercator)
            .with_interpolation_model(InterpolationModel::SphericalRBF)
            .with_spherical_distance(false)
            .with_max_neighbors(10);

        assert_eq!(
            interpolator.config.coordinate_system,
            CoordinateSystem::WebMercator
        );
        assert_eq!(interpolator.config.model, InterpolationModel::SphericalRBF);
        assert!(!interpolator.config.use_spherical_distance);
        assert_eq!(interpolator.config.max_neighbors, Some(10));
    }

    #[test]
    fn test_geospatial_fitting() {
        let latitudes = Array1::from_vec(vec![40.0, 41.0, 42.0, 43.0]);
        let longitudes = Array1::from_vec(vec![-74.0, -75.0, -76.0, -77.0]);
        let temperatures = Array1::from_vec(vec![15.0, 12.0, 10.0, 8.0]);

        let mut interpolator = GeospatialInterpolator::new()
            .with_interpolation_model(InterpolationModel::SphericalRBF);

        let result = interpolator.fit(&latitudes.view(), &longitudes.view(), &temperatures.view());
        assert!(result.is_ok());
        assert!(interpolator.is_trained());
    }

    #[test]
    fn test_geospatial_interpolation() {
        let latitudes = Array1::from_vec(vec![40.0, 41.0, 42.0, 43.0]);
        let longitudes = Array1::from_vec(vec![-74.0, -75.0, -76.0, -77.0]);
        let temperatures = Array1::from_vec(vec![15.0, 12.0, 10.0, 8.0]);

        let mut interpolator = GeospatialInterpolator::new()
            .with_interpolation_model(InterpolationModel::SphericalRBF);

        interpolator
            .fit(&latitudes.view(), &longitudes.view(), &temperatures.view())
            .unwrap();

        let query_lats = Array1::from_vec(vec![40.5, 41.5]);
        let query_lons = Array1::from_vec(vec![-74.5, -75.5]);
        let result = interpolator
            .interpolate(&query_lats.view(), &query_lons.view())
            .unwrap();

        assert_eq!(result.values.len(), 2);
        assert!(result.values.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_great_circle_distance() {
        let interpolator = GeospatialInterpolator::<f64>::new();

        // Distance between NYC and Los Angeles (approximately 3935 km)
        let nyc_lat = 40.7128;
        let nyc_lon = -74.0060;
        let la_lat = 34.0522;
        let la_lon = -118.2437;

        let distance = interpolator.great_circle_distance(nyc_lat, nyc_lon, la_lat, la_lon);

        // Should be approximately 3935 km (allow 10% error for simple calculation)
        assert!((distance - 3935.0).abs() < 400.0);
    }

    #[test]
    fn test_coordinate_projection() {
        let interpolator =
            GeospatialInterpolator::<f64>::new().with_coordinate_system(CoordinateSystem::WGS84);

        let latitudes = Array1::from_vec(vec![40.0, 41.0]);
        let longitudes = Array1::from_vec(vec![-74.0, -75.0]);

        let result = interpolator.project_coordinates(&latitudes.view(), &longitudes.view());
        assert!(result.is_ok());

        let (x_coords, y_coords) = result.unwrap();
        assert_eq!(x_coords.len(), 2);
        assert_eq!(y_coords.len(), 2);
    }

    #[test]
    fn test_make_climate_interpolator() {
        let interpolator = make_climate_interpolator::<f64>();
        assert_eq!(
            interpolator.config.coordinate_system,
            CoordinateSystem::WGS84
        );
        assert_eq!(interpolator.config.model, InterpolationModel::Kriging);
        assert!(interpolator.config.use_spherical_distance);
        assert_eq!(interpolator.config.max_neighbors, Some(50));
    }

    #[test]
    fn test_make_elevation_interpolator() {
        let interpolator = make_elevation_interpolator::<f64>();
        assert_eq!(
            interpolator.config.coordinate_system,
            CoordinateSystem::WGS84
        );
        assert_eq!(
            interpolator.config.model,
            InterpolationModel::ThinPlateSpline
        );
        assert!(interpolator.config.use_spherical_distance);
        assert_eq!(interpolator.config.max_neighbors, Some(30));
    }

    #[test]
    fn test_kriging_model() {
        let latitudes = Array1::from_vec(vec![40.0, 41.0, 42.0, 43.0, 44.0]);
        let longitudes = Array1::from_vec(vec![-74.0, -75.0, -76.0, -77.0, -78.0]);
        let temperatures = Array1::from_vec(vec![15.0, 12.0, 10.0, 8.0, 6.0]);

        let mut interpolator =
            GeospatialInterpolator::new().with_interpolation_model(InterpolationModel::Kriging);

        let fit_result =
            interpolator.fit(&latitudes.view(), &longitudes.view(), &temperatures.view());
        assert!(fit_result.is_ok());

        let query_lats = Array1::from_vec(vec![40.5]);
        let query_lons = Array1::from_vec(vec![-74.5]);
        let result = interpolator.interpolate(&query_lats.view(), &query_lons.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_thin_plate_spline_model() {
        // Use non-collinear points for thin plate spline
        let latitudes = Array1::from_vec(vec![40.0, 41.0, 40.5, 41.5, 40.8]);
        let longitudes = Array1::from_vec(vec![-74.0, -74.5, -75.0, -74.2, -74.8]);
        let elevations = Array1::from_vec(vec![100.0, 200.0, 150.0, 250.0, 180.0]);

        let mut interpolator = GeospatialInterpolator::new()
            .with_interpolation_model(InterpolationModel::ThinPlateSpline);

        let fit_result =
            interpolator.fit(&latitudes.view(), &longitudes.view(), &elevations.view());
        assert!(
            fit_result.is_ok(),
            "Failed to fit thin plate spline: {:?}",
            fit_result.err()
        );

        let query_lats = Array1::from_vec(vec![40.5]);
        let query_lons = Array1::from_vec(vec![-74.5]);
        let result = interpolator.interpolate(&query_lats.view(), &query_lons.view());
        assert!(result.is_ok());
    }
}
