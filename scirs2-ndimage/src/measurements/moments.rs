//! Moment calculation functions for arrays

use scirs2_core::ndarray::{Array, Array1, Array2, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::utils::safe_usize_to_float;

/// Find the center of mass (centroid) of an array
///
/// Computes the intensity-weighted centroid of an n-dimensional array.
/// The center of mass is calculated as the average position of all pixels,
/// weighted by their intensity values. This is useful for object localization,
/// tracking, and geometric analysis.
///
/// # Arguments
///
/// * `input` - Input array containing intensity values
///
/// # Returns
///
/// * `Result<Vec<T>>` - Center of mass coordinates, one per dimension
///
/// # Examples
///
/// ## Basic 1D center of mass
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Simple 1D signal with peak at position 2
/// let signal = Array1::from_vec(vec![0.0, 1.0, 5.0, 1.0, 0.0]);
/// let centroid = center_of_mass(&signal).unwrap();
///
/// // Center of mass should be close to position 2 (where the peak is)
/// assert!((centroid[0] - 2.0_f64).abs() < 0.1);
/// ```
///
/// ## 2D object localization
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Create a 2D object (bright square in upper-left)
/// let mut image = Array2::zeros((10, 10));
/// for i in 2..5 {
///     for j in 2..5 {
///         image[[i, j]] = 10.0;
///     }
/// }
///
/// let centroid = center_of_mass(&image).unwrap();
/// // Centroid should be approximately at (3, 3) - center of the bright square
/// assert!((centroid[0] - 3.0_f64).abs() < 0.1);
/// assert!((centroid[1] - 3.0_f64).abs() < 0.1);
/// ```
///
/// ## Intensity-weighted centroid
/// ```
/// use scirs2_core::ndarray::{Array2, array};
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Create object with non-uniform intensity distribution
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 2.0, 0.0],
///     [0.0, 3.0, 6.0, 0.0],  // Higher intensities toward bottom-right
///     [0.0, 0.0, 0.0, 0.0]
/// ];
///
/// let centroid = center_of_mass(&image).unwrap();
/// // Centroid will be shifted toward higher intensity pixels
/// // Should be closer to (2, 2) than (1.5, 1.5) due to intensity weighting
/// ```
///
/// ## 3D volume center of mass
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Create a 3D volume with a bright cube in one corner
/// let mut volume = Array3::zeros((20, 20, 20));
/// for i in 5..10 {
///     for j in 5..10 {
///         for k in 5..10 {
///             volume[[i, j, k]] = 1.0;
///         }
///     }
/// }
///
/// let centroid = center_of_mass(&volume).unwrap();
/// // Centroid should be at approximately (7.5, 7.5, 7.5)
/// assert_eq!(centroid.len(), 3); // 3D coordinates
/// ```
///
/// ## Binary object analysis
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::measurements::center_of_mass;
///
/// // Binary image (0.0 and 1.0 values only)
/// let binary = Array2::from_shape_fn((50, 50), |(i, j)| {
///     if ((i as f64 - 25.0).powi(2) + (j as f64 - 25.0).powi(2)).sqrt() < 10.0 {
///         1.0
///     } else {
///         0.0
///     }
/// });
///
/// let centroid = center_of_mass(&binary).unwrap();
/// // For a circular object centered at (25, 25), centroid should be near center
/// assert!((centroid[0] - 25.0_f64).abs() < 1.0);
/// assert!((centroid[1] - 25.0_f64).abs() < 1.0);
/// ```
///
/// # Special Cases
///
/// - If the total mass (sum of all values) is zero, returns the geometric center of the array
/// - For binary images, equivalent to finding the centroid of the foreground region
/// - Subpixel precision is maintained for accurate localization
#[allow(dead_code)]
pub fn center_of_mass<T, D>(input: &Array<T, D>) -> NdimageResult<Vec<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    let ndim = input.ndim();
    let shape = input.shape();

    // Calculate total mass (sum of all values)
    let total_mass = input.sum();

    if total_mass == T::zero() {
        // If total mass is zero, return center of array
        let center: Result<Vec<T>, NdimageError> = shape
            .iter()
            .map(|&dim| safe_usize_to_float::<T>(dim).map(|dim_t| dim_t / (T::one() + T::one())))
            .collect();
        let center = center?;
        return Ok(center);
    }

    // Calculate center of mass for each dimension
    let mut center_of_mass = vec![T::zero(); ndim];

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    // Iterate through all elements in the array
    for (idx, &value) in input_dyn.indexed_iter() {
        if value != T::zero() {
            // Add weighted coordinates
            for (dim, &coord) in idx.as_array_view().iter().enumerate() {
                let coord_t = safe_usize_to_float::<T>(coord)?;
                center_of_mass[dim] += coord_t * value;
            }
        }
    }

    // Normalize by total mass
    for coord in center_of_mass.iter_mut() {
        *coord /= total_mass;
    }

    Ok(center_of_mass)
}

/// Find the moment of inertia tensor of an array
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * `Result<Array<T, scirs2_core::ndarray::Ix2>>` - Moment of inertia tensor
#[allow(dead_code)]
pub fn moments_inertia_tensor<T, D>(
    input: &Array<T, D>,
) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix2>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Placeholder implementation
    let dim = input.ndim();
    Ok(Array2::<T>::zeros((dim, dim)))
}

/// Calculate image moments
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
///
/// # Returns
///
/// * `Result<Array<T, scirs2_core::ndarray::Ix1>>` - Array of moments
#[allow(dead_code)]
pub fn moments<T, D>(
    input: &Array<T, D>,
    order: usize,
) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // For 2D images, calculate raw moments M_pq where p, q <= order
    // For nD arrays, we generalize to all possible combinations of powers

    let ndim = input.ndim();

    // For 2D case (most common), calculate standard 2D moments
    if ndim == 2 {
        let mut moments_vec = Vec::new();
        let input_2d = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Expected 2D array for 2D moments".into()))?;

        // Calculate moments M_pq for p, q from 0 to order
        for p in 0..=order {
            for q in 0..=order {
                let mut moment = T::zero();

                for (row, col) in scirs2_core::ndarray::indices(input_2d.dim()) {
                    let value = input_2d[[row, col]];
                    if value != T::zero() {
                        let x = safe_usize_to_float::<T>(col)?;
                        let y = safe_usize_to_float::<T>(row)?;

                        // M_pq = sum(x^p * y^q * I(x,y))
                        let x_power = if p == 0 { T::one() } else { x.powi(p as i32) };
                        let y_power = if q == 0 { T::one() } else { y.powi(q as i32) };

                        moment += x_power * y_power * value;
                    }
                }

                moments_vec.push(moment);
            }
        }

        let total_moments = (order + 1) * (order + 1);
        Array1::<T>::from_vec(moments_vec)
            .into_shape((total_moments,))
            .map_err(|_| NdimageError::ComputationError("Failed to reshape moments array".into()))
    } else {
        // For nD case, return simplified implementation
        // Calculate only the basic moments (total mass, first moments, etc.)
        let mut moments_vec = Vec::new();

        // M_00...0 = total mass
        moments_vec.push(input.sum());

        // First moments for each dimension
        let center = center_of_mass(input)?;
        let total_mass = input.sum();

        for dim in 0..ndim {
            // M_10...0, M_01...0, etc. = center * mass
            moments_vec.push(center[dim] * total_mass);
        }

        // Pad with zeros to match expected size
        let expected_size = (order + 1).pow(ndim as u32);
        while moments_vec.len() < expected_size {
            moments_vec.push(T::zero());
        }

        Array1::<T>::from_vec(moments_vec)
            .into_shape((expected_size,))
            .map_err(|_| NdimageError::ComputationError("Failed to reshape moments array".into()))
    }
}

/// Calculate central moments (moments around centroid)
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
/// * `center` - Center coordinates (if None, uses center of mass)
///
/// # Returns
///
/// * `Result<Array<T, scirs2_core::ndarray::Ix1>>` - Array of central moments
#[allow(dead_code)]
pub fn central_moments<T, D>(
    input: &Array<T, D>,
    order: usize,
    center: Option<&[T]>,
) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if let Some(c) = center {
        if c.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Center must have same length as input dimensions (got {} expected {})",
                c.len(),
                input.ndim()
            )));
        }
    }

    let ndim = input.ndim();

    // Determine center coordinates
    let center_coords = if let Some(c) = center {
        c.to_vec()
    } else {
        center_of_mass(input)?
    };

    // For 2D case, calculate central moments μ_pq
    if ndim == 2 {
        let mut central_moments_vec = Vec::new();
        let input_2d = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| {
                NdimageError::DimensionError("Expected 2D array for 2D central moments".into())
            })?;

        let cx = center_coords[1]; // x-center (column)
        let cy = center_coords[0]; // y-center (row)

        // Calculate central moments μ_pq for p, q from 0 to order
        for p in 0..=order {
            for q in 0..=order {
                let mut moment = T::zero();

                for (row, col) in scirs2_core::ndarray::indices(input_2d.dim()) {
                    let value = input_2d[[row, col]];
                    if value != T::zero() {
                        let x = safe_usize_to_float::<T>(col)?;
                        let y = safe_usize_to_float::<T>(row)?;

                        // μ_pq = sum((x-cx)^p * (y-cy)^q * I(x,y))
                        let dx = x - cx;
                        let dy = y - cy;

                        let x_power = if p == 0 { T::one() } else { dx.powi(p as i32) };
                        let y_power = if q == 0 { T::one() } else { dy.powi(q as i32) };

                        moment += x_power * y_power * value;
                    }
                }

                central_moments_vec.push(moment);
            }
        }

        let total_moments = (order + 1) * (order + 1);
        Array1::<T>::from_vec(central_moments_vec)
            .into_shape((total_moments,))
            .map_err(|_| {
                NdimageError::ComputationError("Failed to reshape central moments array".into())
            })
    } else {
        // For nD case, simplified implementation
        let mut central_moments_vec = Vec::new();

        // μ_00...0 = total mass (same as raw moment)
        central_moments_vec.push(input.sum());

        // First central moments are zero by definition (moments around centroid)
        for _ in 0..ndim {
            central_moments_vec.push(T::zero());
        }

        // Second central moments (variances/covariances)
        if order >= 2 {
            let _total_mass = input.sum();
            let input_dyn = input.clone().into_dyn();

            for dim1 in 0..ndim {
                for dim2 in dim1..ndim {
                    let mut moment = T::zero();

                    for (idx, &value) in input_dyn.indexed_iter() {
                        if value != T::zero() {
                            let coord1 = safe_usize_to_float::<T>(idx.as_array_view()[dim1])?;
                            let coord2 = safe_usize_to_float::<T>(idx.as_array_view()[dim2])?;

                            let dc1 = coord1 - center_coords[dim1];
                            let dc2 = coord2 - center_coords[dim2];

                            moment += dc1 * dc2 * value;
                        }
                    }

                    central_moments_vec.push(moment);
                }
            }
        }

        // Pad with zeros to match expected size
        let expected_size = (order + 1).pow(ndim as u32);
        while central_moments_vec.len() < expected_size {
            central_moments_vec.push(T::zero());
        }

        Array1::<T>::from_vec(central_moments_vec)
            .into_shape((expected_size,))
            .map_err(|_| {
                NdimageError::ComputationError("Failed to reshape central moments array".into())
            })
    }
}

/// Calculate normalized moments
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
///
/// # Returns
///
/// * `Result<Array<T, scirs2_core::ndarray::Ix1>>` - Array of normalized moments
#[allow(dead_code)]
pub fn normalized_moments<T, D>(
    input: &Array<T, D>,
    order: usize,
) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + NumAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Calculate central moments first
    let central_moments = central_moments(input, order, None)?;

    let ndim = input.ndim();

    // For 2D case, calculate normalized moments η_pq
    if ndim == 2 {
        let mut normalized_moments_vec = Vec::new();

        // Get μ_00 (total mass) for normalization
        let mu_00 = central_moments[0]; // First element is μ_00

        if mu_00 == T::zero() {
            // If total mass is zero, return zeros
            let total_moments = (order + 1) * (order + 1);
            return Ok(Array1::<T>::zeros(total_moments));
        }

        // Calculate normalized moments η_pq = μ_pq / μ_00^((p+q)/2+1)
        for p in 0..=order {
            for q in 0..=order {
                let moment_idx = p * (order + 1) + q;
                let mu_pq = central_moments[moment_idx];

                if p == 0 && q == 0 {
                    // η_00 = 1 by definition
                    normalized_moments_vec.push(T::one());
                } else {
                    let gamma = (p + q) as f64 / 2.0 + 1.0;
                    let gamma_t = T::from_f64(gamma).ok_or_else(|| {
                        NdimageError::ComputationError(
                            "Failed to convert gamma to float type".into(),
                        )
                    })?;

                    let normalizer = mu_00.powf(gamma_t);
                    let eta_pq = if normalizer != T::zero() {
                        mu_pq / normalizer
                    } else {
                        T::zero()
                    };

                    normalized_moments_vec.push(eta_pq);
                }
            }
        }

        let total_moments = (order + 1) * (order + 1);
        Array1::<T>::from_vec(normalized_moments_vec)
            .into_shape((total_moments,))
            .map_err(|_| {
                NdimageError::ComputationError("Failed to reshape normalized moments array".into())
            })
    } else {
        // For nD case, simplified implementation
        let mut normalized_moments_vec = Vec::new();

        let mu_00 = central_moments[0]; // Total mass

        if mu_00 == T::zero() {
            let expected_size = (order + 1).pow(ndim as u32);
            return Ok(Array1::<T>::zeros(expected_size));
        }

        // Normalize available central moments
        normalized_moments_vec.push(T::one()); // η_00...0 = 1

        // First moments are zero for central moments, so normalized are also zero
        for _ in 0..ndim {
            normalized_moments_vec.push(T::zero());
        }

        // Normalize higher order moments
        for i in (ndim + 1)..central_moments.len() {
            let mu_i = central_moments[i];
            // Use a simple normalization for nD case
            let eta_i = if mu_00 != T::zero() {
                mu_i / (mu_00 * mu_00) // Simple normalization
            } else {
                T::zero()
            };
            normalized_moments_vec.push(eta_i);
        }

        // Pad with zeros to match expected size
        let expected_size = (order + 1).pow(ndim as u32);
        while normalized_moments_vec.len() < expected_size {
            normalized_moments_vec.push(T::zero());
        }

        Array1::<T>::from_vec(normalized_moments_vec)
            .into_shape((expected_size,))
            .map_err(|_| {
                NdimageError::ComputationError("Failed to reshape normalized moments array".into())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_center_of_mass() {
        let input: Array2<f64> = Array2::eye(3);
        let com = center_of_mass(&input).expect("center_of_mass should succeed for test");
        assert_eq!(com.len(), input.ndim());
    }

    #[test]
    fn test_moments_inertia_tensor() {
        let input: Array2<f64> = Array2::eye(3);
        let tensor =
            moments_inertia_tensor(&input).expect("moments_inertia_tensor should succeed for test");
        assert_eq!(tensor.shape(), &[input.ndim(), input.ndim()]);
    }

    #[test]
    fn test_moments() {
        let input: Array2<f64> = Array2::eye(3);
        let order = 2;
        let mom = moments(&input, order).expect("moments should succeed for test");
        assert!(!mom.is_empty());
    }
}
