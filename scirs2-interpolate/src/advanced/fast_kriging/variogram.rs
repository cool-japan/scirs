//! Variogram Modeling for Fast Kriging
//!
//! This module provides functionality for estimating and modeling variograms,
//! which describe the spatial correlation structure of a dataset.

use crate::advanced::enhanced_kriging::AnisotropicCovariance;
use crate::error::InterpolateResult;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Variogram model types for kriging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariogramModel {
    /// Spherical variogram model
    Spherical,
    
    /// Exponential variogram model
    Exponential,
    
    /// Gaussian variogram model
    Gaussian,
    
    /// Matern variogram model with smoothness parameter
    Matern(f64),
    
    /// Power variogram model
    Power(f64),
}

/// Empirical variogram bin
#[derive(Debug, Clone)]
pub struct VariogramBin<F: Float> {
    /// Distance at center of bin
    pub distance: F,
    
    /// Average semivariance in bin
    pub semivariance: F,
    
    /// Number of point pairs in bin
    pub count: usize,
}

/// Compute empirical variogram from data
pub fn compute_empirical_variogram<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    n_bins: usize,
    max_distance: Option<F>,
) -> InterpolateResult<Vec<VariogramBin<F>>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let n_points = points.shape()[0];
    let n_dims = points.shape()[1];
    
    // Validate inputs
    if n_points != values.len() {
        return Err(crate::error::InterpolateError::DimensionMismatch(
            "Number of points must match number of values".to_string(),
        ));
    }
    
    if n_points < 2 {
        return Err(crate::error::InterpolateError::InvalidValue(
            "At least 2 points are required for variogram estimation".to_string(),
        ));
    }
    
    // Calculate maximum distance if not provided
    let max_dist = match max_distance {
        Some(dist) => dist,
        None => {
            // Estimate max distance as the diagonal of the bounding box
            let mut max_d = F::zero();
            for i in 0..n_points {
                for j in (i+1)..n_points {
                    let mut dist_sq = F::zero();
                    for d in 0..n_dims {
                        let diff = points[[i, d]] - points[[j, d]];
                        dist_sq = dist_sq + diff * diff;
                    }
                    let dist = dist_sq.sqrt();
                    if dist > max_d {
                        max_d = dist;
                    }
                }
            }
            max_d
        }
    };
    
    // Calculate bin width
    let bin_width = max_dist / F::from_usize(n_bins).unwrap();
    
    // Initialize bins
    let mut bins = vec![
        VariogramBin {
            distance: F::zero(),
            semivariance: F::zero(),
            count: 0,
        };
        n_bins
    ];
    
    // For each bin, set the center distance
    for i in 0..n_bins {
        bins[i].distance = F::from_usize(i).unwrap() * bin_width + bin_width / F::from(2).unwrap();
    }
    
    // Compute empirical variogram by comparing all pairs of points
    for i in 0..n_points {
        for j in (i+1)..n_points {
            // Calculate distance between points
            let mut dist_sq = F::zero();
            for d in 0..n_dims {
                let diff = points[[i, d]] - points[[j, d]];
                dist_sq = dist_sq + diff * diff;
            }
            let dist = dist_sq.sqrt();
            
            // Calculate squared difference in values
            let value_diff = values[i] - values[j];
            let semivariogram_value = value_diff * value_diff / F::from(2).unwrap();
            
            // Find appropriate bin
            let bin_idx = (dist / bin_width).to_usize().unwrap_or(n_bins-1);
            if bin_idx < n_bins {
                bins[bin_idx].semivariance = bins[bin_idx].semivariance + semivariogram_value;
                bins[bin_idx].count += 1;
            }
        }
    }
    
    // Normalize bins by count
    for bin in &mut bins {
        if bin.count > 0 {
            bin.semivariance = bin.semivariance / F::from_usize(bin.count).unwrap();
        }
    }
    
    // Filter out empty bins
    let valid_bins: Vec<VariogramBin<F>> = bins.into_iter()
        .filter(|bin| bin.count > 0)
        .collect();
    
    if valid_bins.is_empty() {
        return Err(crate::error::InterpolateError::ComputationError(
            "No valid bins found for variogram estimation".to_string(),
        ));
    }
    
    Ok(valid_bins)
}

/// Fit a variogram model to empirical data
pub fn fit_variogram_model<F>(
    bins: &[VariogramBin<F>],
    model: VariogramModel,
) -> InterpolateResult<(F, F, F)>
where
    F: Float + FromPrimitive + Debug + Display + 'static,
{
    if bins.is_empty() {
        return Err(crate::error::InterpolateError::InvalidValue(
            "Cannot fit variogram model to empty bins".to_string(),
        ));
    }
    
    #[cfg(feature = "linalg")]
    {
        use ndarray_linalg::LeastSquaresSvd;
        
        // Initial guess for parameters
        let max_semivariance = bins.iter()
            .map(|bin| bin.semivariance)
            .fold(F::zero(), |a, b| if a > b { a } else { b });
            
        let max_distance = bins.iter()
            .map(|bin| bin.distance)
            .fold(F::zero(), |a, b| if a > b { a } else { b });
            
        // Initial guess for parameters
        let mut nugget = F::from_f64(0.001).unwrap() * max_semivariance;
        let mut sill = max_semivariance - nugget;
        let mut range = max_distance / F::from_f64(3.0).unwrap();
        
        // Create design matrix and right-hand side for least squares
        let n_bins = bins.len();
        let mut a = Array2::<f64>::zeros((n_bins, 3));
        let mut b = Array1::<f64>::zeros(n_bins);
        
        for i in 0..n_bins {
            let h = bins[i].distance.to_f64().unwrap();
            let gamma = bins[i].semivariance.to_f64().unwrap();
            
            a[[i, 0]] = 1.0; // Nugget effect
            
            // Compute the variogram model value
            let model_val = match model {
                VariogramModel::Spherical => {
                    let range_val = range.to_f64().unwrap();
                    if h <= range_val {
                        1.5 * (h / range_val) - 0.5 * (h / range_val).powi(3)
                    } else {
                        1.0
                    }
                },
                VariogramModel::Exponential => {
                    let range_val = range.to_f64().unwrap();
                    1.0 - (-3.0 * h / range_val).exp()
                },
                VariogramModel::Gaussian => {
                    let range_val = range.to_f64().unwrap();
                    1.0 - (-3.0 * (h / range_val).powi(2)).exp()
                },
                VariogramModel::Matern(nu) => {
                    let range_val = range.to_f64().unwrap();
                    if h <= 1e-6 {
                        0.0
                    } else {
                        // For simplicity, we'll approximate Matern
                        // In a full implementation, this would use Bessel functions
                        1.0 - (-3.0 * h / range_val).powf(nu).exp()
                    }
                },
                VariogramModel::Power(exponent) => {
                    let range_val = range.to_f64().unwrap();
                    (h / range_val).powf(exponent)
                },
            };
            
            a[[i, 1]] = model_val; // Sill component
            a[[i, 2]] = h; // Range component (for optimization)
            b[i] = gamma;
        }
        
        // Solve least squares problem
        match a.least_squares(&b) {
            Ok(solution) => {
                // Update parameters
                nugget = F::from_f64(solution[0]).unwrap();
                sill = F::from_f64(solution[1]).unwrap();
                range = F::from_f64(solution[2]).unwrap();
                
                // Ensure parameters are sensible
                if nugget < F::zero() {
                    nugget = F::from_f64(0.001).unwrap() * max_semivariance;
                }
                
                if sill < F::zero() {
                    sill = max_semivariance - nugget;
                }
                
                if range < F::zero() {
                    range = max_distance / F::from_f64(3.0).unwrap();
                }
                
                Ok((nugget, sill, range))
            },
            Err(_) => {
                // Fallback to initial values if least squares fails
                Ok((nugget, sill, range))
            }
        }
    }
    
    #[cfg(not(feature = "linalg"))]
    {
        // Without linalg, use a simple heuristic
        let max_semivariance = bins.iter()
            .map(|bin| bin.semivariance)
            .fold(F::zero(), |a, b| if a > b { a } else { b });
            
        let max_distance = bins.iter()
            .map(|bin| bin.distance)
            .fold(F::zero(), |a, b| if a > b { a } else { b });
            
        // Estimate nugget as the y-intercept (value at distance near 0)
        let mut nugget = if !bins.is_empty() {
            // Find bin with smallest distance
            let min_dist_bin = bins.iter()
                .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
                .unwrap();
            
            min_dist_bin.semivariance
        } else {
            F::from_f64(0.05).unwrap() * max_semivariance
        };
        
        // Ensure nugget is positive but not too large
        if nugget <= F::zero() || nugget >= max_semivariance {
            nugget = F::from_f64(0.05).unwrap() * max_semivariance;
        }
        
        // Estimate sill as maximum semivariance minus nugget
        let sill = max_semivariance - nugget;
        
        // Estimate range as 1/3 of maximum distance
        let range = max_distance / F::from_f64(3.0).unwrap();
        
        Ok((nugget, sill, range))
    }
}

/// Convert variogram parameters to covariance parameters
pub fn variogram_to_covariance<F>(
    nugget: F,
    sill: F,
    range: F,
    model: VariogramModel,
) -> AnisotropicCovariance<F>
where
    F: Float + FromPrimitive + Debug + Display,
{
    use crate::advanced::kriging::CovarianceFunction;
    
    // Convert variogram model to covariance function
    let (cov_fn, extra_params) = match model {
        VariogramModel::Spherical => (CovarianceFunction::Matern52, F::zero()),
        VariogramModel::Exponential => (CovarianceFunction::Exponential, F::zero()),
        VariogramModel::Gaussian => (CovarianceFunction::SquaredExponential, F::zero()),
        VariogramModel::Matern(nu) => {
            if nu < 1.0 {
                (CovarianceFunction::Exponential, F::zero())
            } else if nu < 2.0 {
                (CovarianceFunction::Matern32, F::zero())
            } else {
                (CovarianceFunction::Matern52, F::zero())
            }
        },
        VariogramModel::Power(_) => (CovarianceFunction::RationalQuadratic, F::from_f64(0.5).unwrap()),
    };
    
    // Create anisotropic covariance object
    // For simplicity, we'll use isotropic scaling here
    let length_scales = vec![range];
    
    AnisotropicCovariance {
        cov_fn,
        length_scales,
        sigma_sq: sill,
        nugget,
        extra_params,
    }
}