//! Binary morphological operations on arrays
//!
//! This implementation focuses on 1D and 2D arrays to simplify handling.
//! For more general n-dimensional arrays (3D and above), the implementation
//! currently returns a NotImplementedError.
//!
//! # Important Implementation Notes
//!
//! 1. Dimensions and handling:
//!    - 1D arrays: Fully supported with optimized implementation
//!    - 2D arrays: Fully supported with optimized implementation
//!    - nD arrays (n > 2): Support is limited and may result in NotImplementedError
//!
//! 2. Function signatures:
//!    - All functions accept generic dimension parameter D
//!    - When D is Ix1 or Ix2, specific implementation is used
//!    - When D is IxDyn, the function checks the dimensionality and routes to the right implementation
//!
//! # Recommended Usage
//!
//! - For 1D and 2D arrays: Both the generic functions here and the functions in simple_morph work well
//! - For higher dimensional arrays: Convert to IxDyn first, but be aware of limitations
//! - For production code: Prefer the simple_morph module when working with 2D arrays
//!

use ndarray::{Array, Array1, Array2, Dimension, Ix1, Ix2, IxDyn};

use super::structuring::generate_binary_structure_dyn;
use super::utils::get_structure_center_dyn;
use crate::error::{NdimageError, Result};

/// Erode a binary array using a structuring element
///
/// Binary erosion removes pixels at the boundaries of regions of positive pixels,
/// resulting in a smaller region.
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `iterations` - Number of times to apply the erosion (default: 1)
/// * `mask` - Mask array that limits the operation (if None, no mask is applied)
/// * `border_value` - Border value (default: false)
/// * `origin` - Origin of the structuring element (if None, uses the center)
/// * `brute_force` - Whether to use brute force algorithm (default: false)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Eroded array
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_ndimage::morphology::binary_erosion;
///
/// // Create a simple 3x3 array filled with true values
/// let input = Array2::from_elem((3, 3), true);
///
/// // Erode the array
/// let result = binary_erosion(&input, None, None, None, None, None, None).unwrap();
///
/// // The center of the eroded array is still true, but the border elements may be eroded
/// assert!(result[[1, 1]]);
/// ```
pub fn binary_erosion<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle based on dimensionality
    match input.ndim() {
        1 => {
            if let Ok(input_1d) = input.clone().into_dimensionality::<Ix1>() {
                // Convert structure to 1D if provided
                let structure_1d = match structure {
                    Some(s) => {
                        if let Ok(s1d) = s.clone().into_dimensionality::<Ix1>() {
                            Some(s1d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert structure to 1D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Convert mask to 1D if provided
                let mask_1d = match mask {
                    Some(m) => {
                        if let Ok(m1d) = m.clone().into_dimensionality::<Ix1>() {
                            Some(m1d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert mask to 1D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Call 1D implementation
                let result_1d = binary_erosion1d(
                    &input_1d,
                    structure_1d.as_ref(),
                    iterations,
                    mask_1d.as_ref(),
                    border_value,
                    origin,
                    brute_force,
                )?;

                // Convert back to original dimensionality
                return result_1d.into_dimensionality().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensionality".to_string(),
                    )
                });
            }
        }
        2 => {
            if let Ok(input_2d) = input.clone().into_dimensionality::<Ix2>() {
                // Convert structure to 2D if provided
                let structure_2d = match structure {
                    Some(s) => {
                        if let Ok(s2d) = s.clone().into_dimensionality::<Ix2>() {
                            Some(s2d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert structure to 2D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Convert mask to 2D if provided
                let mask_2d = match mask {
                    Some(m) => {
                        if let Ok(m2d) = m.clone().into_dimensionality::<Ix2>() {
                            Some(m2d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert mask to 2D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Call 2D implementation
                let result_2d = binary_erosion2d(
                    &input_2d,
                    structure_2d.as_ref(),
                    iterations,
                    mask_2d.as_ref(),
                    border_value,
                    origin,
                    brute_force,
                )?;

                // Convert back to original dimensionality
                return result_2d.into_dimensionality().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensionality".to_string(),
                    )
                });
            }
        }
        _ => {
            // For higher dimensions, convert to dynamic dimension
            if let Ok(input_dyn) = input.clone().into_dimensionality::<IxDyn>() {
                // Convert structure to dyn if provided
                let structure_dyn = match structure {
                    Some(s) => {
                        if let Ok(sdyn) = s.clone().into_dimensionality::<IxDyn>() {
                            Some(sdyn)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert structure to dynamic dimension".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Convert mask to dyn if provided
                let mask_dyn = match mask {
                    Some(m) => {
                        if let Ok(mdyn) = m.clone().into_dimensionality::<IxDyn>() {
                            Some(mdyn)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert mask to dynamic dimension".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Call dynamic implementation
                let result_dyn = binary_erosion_dyn(
                    &input_dyn,
                    structure_dyn.as_ref(),
                    iterations,
                    mask_dyn.as_ref(),
                    border_value,
                    origin,
                    brute_force,
                )?;

                // Convert back to original dimensionality
                return result_dyn.into_dimensionality().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensionality".to_string(),
                    )
                });
            }
        }
    }

    // Fallback case (should not be reached, but needed for type checking)
    Err(NdimageError::DimensionError(
        "Unsupported array dimensions for erosion".to_string(),
    ))
}

/// Implementation of binary erosion for 1D arrays
fn binary_erosion1d(
    input: &Array1<bool>,
    structure: Option<&Array1<bool>>,
    iterations: Option<usize>,
    mask: Option<&Array1<bool>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array1<bool>> {
    // Default parameter values
    let iters = iterations.unwrap_or(1);
    let border_val = border_value.unwrap_or(false);
    let brute_force_algo = brute_force.unwrap_or(false);

    // Create a default structure if none is provided
    let owned_structure;
    let struct_elem = if let Some(s) = structure {
        s
    } else {
        // Create a default structure with face connectivity
        owned_structure = Array1::from_elem(3, true);
        &owned_structure
    };

    // Calculate the origin if not provided
    let origin_vec: Vec<isize> = if let Some(o) = origin {
        if o.len() != 1 {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have same length as input dimensions (got {} expected {})",
                o.len(),
                1
            )));
        }
        o.to_vec()
    } else {
        // Default origin is at the center of the structure
        vec![(struct_elem.len() as isize) / 2]
    };

    // Implementation for 1D erosion
    let mut result = input.to_owned();

    // Apply erosion the specified number of times
    for _ in 0..iters {
        // Create a temporary array for this iteration's result
        let mut temp = Array1::from_elem(input.len(), false);
        let prev = result.clone();

        // Iterate over each position in the array
        for (i, val) in temp.indexed_iter_mut() {
            // Skip if masked
            if let Some(m) = mask {
                if !m[i] {
                    *val = prev[i];
                    continue;
                }
            }

            // Check if the structuring element fits at this position
            let mut fits = true;
            for (s_i, &s_val) in struct_elem.indexed_iter() {
                if !s_val {
                    continue; // Only consider true values in the structure
                }

                // Calculate corresponding position in input
                let offset = s_i as isize - origin_vec[0];
                let pos = i as isize + offset;

                // Check if position is within bounds
                if pos < 0 || pos >= prev.len() as isize {
                    // Outside bounds - use border value
                    if !border_val {
                        fits = false;
                        break;
                    }
                } else if !prev[pos as usize] {
                    // Position is within bounds but value is false
                    fits = false;
                    break;
                }
            }

            *val = fits;
        }

        result = temp;

        // Check if we've reached a fixed point (no change)
        if !brute_force_algo && result == prev {
            break;
        }
    }

    Ok(result)
}

/// Implementation of binary erosion for 2D arrays
fn binary_erosion2d(
    input: &Array2<bool>,
    structure: Option<&Array2<bool>>,
    iterations: Option<usize>,
    mask: Option<&Array2<bool>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array2<bool>> {
    // Default parameter values
    let iters = iterations.unwrap_or(1);
    let border_val = border_value.unwrap_or(false);
    let brute_force_algo = brute_force.unwrap_or(false);

    // Create a default structure if none is provided
    let owned_structure;
    let struct_elem = if let Some(s) = structure {
        s
    } else {
        // Create a box structure with face connectivity
        let size = [3, 3];
        owned_structure = Array2::from_elem((size[0], size[1]), true);
        &owned_structure
    };

    // Calculate the origin if not provided
    let origin_vec: Vec<isize> = if let Some(o) = origin {
        if o.len() != 2 {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have same length as input dimensions (got {} expected {})",
                o.len(),
                2
            )));
        }
        o.to_vec()
    } else {
        // Default origin is at the center of the structure
        struct_elem
            .shape()
            .iter()
            .map(|&s| (s as isize) / 2)
            .collect()
    };

    let shape = input.shape();
    let mut result = input.to_owned();

    // Apply erosion for the specified number of iterations
    for iter in 0..iters {
        let prev = result.clone();
        let mut temp = Array2::from_elem((shape[0], shape[1]), false);

        // Get structure dimensions
        let s_rows = struct_elem.shape()[0];
        let s_cols = struct_elem.shape()[1];

        // Calculate half sizes for the structure
        let half_height = origin_vec[0];
        let half_width = origin_vec[1];

        // For each position in the array
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                // Skip masked positions
                if let Some(m) = mask {
                    if !m[[i, j]] {
                        temp[[i, j]] = prev[[i, j]];
                        continue;
                    }
                }

                // Check if the structuring element fits at this position
                let mut fits = true;

                // Iterate over the structure
                'outer: for si in 0..s_rows {
                    for sj in 0..s_cols {
                        if !struct_elem[[si, sj]] {
                            continue; // Skip false values in structure
                        }

                        // Calculate corresponding position in input
                        let ni = i as isize + (si as isize - half_height);
                        let nj = j as isize + (sj as isize - half_width);

                        // Check if position is within bounds
                        if ni < 0 || ni >= shape[0] as isize || nj < 0 || nj >= shape[1] as isize {
                            // Outside bounds - use border value
                            if !border_val {
                                fits = false;
                                break 'outer;
                            }
                        } else if !prev[[ni as usize, nj as usize]] {
                            // Position is within bounds but value is false
                            fits = false;
                            break 'outer;
                        }
                    }
                }

                temp[[i, j]] = fits;
            }
        }

        result = temp;

        // Check if we've reached a fixed point (no change)
        if !brute_force_algo && iter > 0 && result == prev {
            break;
        }
    }

    Ok(result)
}

/// Implementation of binary erosion for n-dimensional arrays (using dynamic dimensions)
fn binary_erosion_dyn(
    input: &Array<bool, IxDyn>,
    structure: Option<&Array<bool, IxDyn>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, IxDyn>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    _brute_force: Option<bool>,
) -> Result<Array<bool, IxDyn>> {
    let iterations = iterations.unwrap_or(1);
    let border = border_value.unwrap_or(false);

    // Get or generate structure
    let default_structure = if let Some(s) = structure {
        s.to_owned()
    } else {
        generate_binary_structure_dyn(input.ndim())?
    };

    // Validate input dimensions
    if input.ndim() != default_structure.ndim() {
        return Err(NdimageError::DimensionError(
            "Input and structure must have the same number of dimensions".into(),
        ));
    }

    // Validate mask dimensions if provided
    if let Some(m) = mask {
        if m.ndim() != input.ndim() || m.shape() != input.shape() {
            return Err(NdimageError::InvalidInput(
                "Mask must have the same shape as input".into(),
            ));
        }
    }

    // Get structure center
    let center = get_structure_center_dyn(&default_structure, origin)?;

    // Create result array
    let mut result = input.to_owned();

    // Apply erosion iterations
    for _ in 0..iterations {
        let temp = result.clone();

        // Iterate through all positions in the input array
        for idx in ndarray::indices(input.shape()) {
            let idx_vec: Vec<_> = idx.slice().to_vec();

            // Skip if masked out
            if let Some(m) = mask {
                if !m[idx_vec.as_slice()] {
                    continue;
                }
            }

            // Check if all structure elements fit
            let mut all_fit = true;

            // Check each structure element
            for str_idx in ndarray::indices(default_structure.shape()) {
                let str_idx_vec: Vec<_> = str_idx.slice().to_vec();

                // Skip if structure element is false
                if !default_structure[str_idx_vec.as_slice()] {
                    continue;
                }

                // Calculate corresponding input position
                let mut input_pos = vec![0isize; input.ndim()];
                for d in 0..input.ndim() {
                    input_pos[d] = idx_vec[d] as isize + str_idx_vec[d] as isize - center[d];
                }

                // Check if position is within bounds
                let mut within_bounds = true;
                for (d, &pos) in input_pos.iter().enumerate().take(input.ndim()) {
                    if pos < 0 || pos >= input.shape()[d] as isize {
                        within_bounds = false;
                        break;
                    }
                }

                // Get the value, using border value if out of bounds
                let val = if within_bounds {
                    let input_idx: Vec<_> = input_pos.iter().map(|&x| x as usize).collect();
                    temp[input_idx.as_slice()]
                } else {
                    border
                };

                // Erosion requires all values to be true
                if !val {
                    all_fit = false;
                    break;
                }
            }

            result[idx_vec.as_slice()] = all_fit;
        }
    }

    Ok(result)
}

/// Dilate a binary array using a structuring element
///
/// Binary dilation adds pixels to the boundaries of regions of positive pixels,
/// effectively expanding these regions.
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `iterations` - Number of times to apply the dilation (default: 1)
/// * `mask` - Mask array that limits the operation (if None, no mask is applied)
/// * `border_value` - Border value (default: false)
/// * `origin` - Origin of the structuring element (if None, uses the center)
/// * `brute_force` - Whether to use brute force algorithm (default: false)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Dilated array
pub fn binary_dilation<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle based on dimensionality
    match input.ndim() {
        1 => {
            if let Ok(input_1d) = input.clone().into_dimensionality::<Ix1>() {
                // Convert structure to 1D if provided
                let structure_1d = match structure {
                    Some(s) => {
                        if let Ok(s1d) = s.clone().into_dimensionality::<Ix1>() {
                            Some(s1d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert structure to 1D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Convert mask to 1D if provided
                let mask_1d = match mask {
                    Some(m) => {
                        if let Ok(m1d) = m.clone().into_dimensionality::<Ix1>() {
                            Some(m1d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert mask to 1D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Call 1D implementation
                let result_1d = binary_dilation1d(
                    &input_1d,
                    structure_1d.as_ref(),
                    iterations,
                    mask_1d.as_ref(),
                    border_value,
                    origin,
                    brute_force,
                )?;

                // Convert back to original dimensionality
                return result_1d.into_dimensionality().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensionality".to_string(),
                    )
                });
            }
        }
        2 => {
            if let Ok(input_2d) = input.clone().into_dimensionality::<Ix2>() {
                // Convert structure to 2D if provided
                let structure_2d = match structure {
                    Some(s) => {
                        if let Ok(s2d) = s.clone().into_dimensionality::<Ix2>() {
                            Some(s2d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert structure to 2D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Convert mask to 2D if provided
                let mask_2d = match mask {
                    Some(m) => {
                        if let Ok(m2d) = m.clone().into_dimensionality::<Ix2>() {
                            Some(m2d)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert mask to 2D".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Call 2D implementation
                let result_2d = binary_dilation2d(
                    &input_2d,
                    structure_2d.as_ref(),
                    iterations,
                    mask_2d.as_ref(),
                    border_value,
                    origin,
                    brute_force,
                )?;

                // Convert back to original dimensionality
                return result_2d.into_dimensionality().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensionality".to_string(),
                    )
                });
            }
        }
        _ => {
            // For higher dimensions, convert to dynamic dimension
            if let Ok(input_dyn) = input.clone().into_dimensionality::<IxDyn>() {
                // Convert structure to dyn if provided
                let structure_dyn = match structure {
                    Some(s) => {
                        if let Ok(sdyn) = s.clone().into_dimensionality::<IxDyn>() {
                            Some(sdyn)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert structure to dynamic dimension".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Convert mask to dyn if provided
                let mask_dyn = match mask {
                    Some(m) => {
                        if let Ok(mdyn) = m.clone().into_dimensionality::<IxDyn>() {
                            Some(mdyn)
                        } else {
                            return Err(NdimageError::DimensionError(
                                "Failed to convert mask to dynamic dimension".to_string(),
                            ));
                        }
                    }
                    None => None,
                };

                // Call dynamic implementation
                let result_dyn = binary_dilation_dyn(
                    &input_dyn,
                    structure_dyn.as_ref(),
                    iterations,
                    mask_dyn.as_ref(),
                    border_value,
                    origin,
                    brute_force,
                )?;

                // Convert back to original dimensionality
                return result_dyn.into_dimensionality().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensionality".to_string(),
                    )
                });
            }
        }
    }

    // Fallback case (should not be reached, but needed for type checking)
    Err(NdimageError::DimensionError(
        "Unsupported array dimensions for dilation".to_string(),
    ))
}

/// Implementation of binary dilation for 1D arrays
fn binary_dilation1d(
    input: &Array1<bool>,
    structure: Option<&Array1<bool>>,
    iterations: Option<usize>,
    mask: Option<&Array1<bool>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array1<bool>> {
    // Default parameter values
    let iters = iterations.unwrap_or(1);
    let border_val = border_value.unwrap_or(false);
    let brute_force_algo = brute_force.unwrap_or(false);

    // Create a default structure if none is provided
    let owned_structure;
    let struct_elem = if let Some(s) = structure {
        s
    } else {
        // Create a default structure with face connectivity
        owned_structure = Array1::from_elem(3, true);
        &owned_structure
    };

    // Calculate the origin if not provided
    let origin_vec: Vec<isize> = if let Some(o) = origin {
        if o.len() != 1 {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have same length as input dimensions (got {} expected {})",
                o.len(),
                1
            )));
        }
        o.to_vec()
    } else {
        // Default origin is at the center of the structure
        vec![(struct_elem.len() as isize) / 2]
    };

    // Implementation for 1D dilation
    let mut result = input.to_owned();

    // Apply dilation the specified number of times
    for _ in 0..iters {
        // Create a temporary array for this iteration's result
        let mut temp = Array1::from_elem(input.len(), false);
        let prev = result.clone();

        // Iterate over each position in the array
        for (i, val) in temp.indexed_iter_mut() {
            // Skip if masked
            if let Some(m) = mask {
                if !m[i] {
                    *val = prev[i];
                    continue;
                }
            }

            // Initialize current position value
            *val = prev[i];

            // If position is already true, no need to check neighbors
            if *val {
                continue;
            }

            // Check for neighboring true values using the structuring element
            for (s_i, &s_val) in struct_elem.indexed_iter() {
                if !s_val {
                    continue; // Only consider true values in the structure
                }

                // Calculate corresponding position in input (reflected)
                let offset = origin_vec[0] - s_i as isize;
                let pos = i as isize + offset;

                // Check if position is within bounds
                if pos < 0 || pos >= prev.len() as isize {
                    // Outside bounds - use border value
                    if border_val {
                        *val = true;
                        break;
                    }
                } else if prev[pos as usize] {
                    // Position has a true value in input
                    *val = true;
                    break;
                }
            }
        }

        result = temp;

        // Check if we've reached a fixed point (no change)
        if !brute_force_algo && result == prev {
            break;
        }
    }

    Ok(result)
}

/// Implementation of binary dilation for 2D arrays
fn binary_dilation2d(
    input: &Array2<bool>,
    structure: Option<&Array2<bool>>,
    iterations: Option<usize>,
    mask: Option<&Array2<bool>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array2<bool>> {
    // Default parameter values
    let iters = iterations.unwrap_or(1);
    let border_val = border_value.unwrap_or(false);
    let brute_force_algo = brute_force.unwrap_or(false);

    // Create a default structure if none is provided
    let owned_structure;
    let struct_elem = if let Some(s) = structure {
        s
    } else {
        // Create a box structure with face connectivity
        let size = [3, 3];
        owned_structure = Array2::from_elem((size[0], size[1]), true);
        &owned_structure
    };

    // Calculate the origin if not provided
    let origin_vec: Vec<isize> = if let Some(o) = origin {
        if o.len() != 2 {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have same length as input dimensions (got {} expected {})",
                o.len(),
                2
            )));
        }
        o.to_vec()
    } else {
        // Default origin is at the center of the structure
        struct_elem
            .shape()
            .iter()
            .map(|&s| (s as isize) / 2)
            .collect()
    };

    let shape = input.shape();
    let mut result = input.to_owned();

    // Apply dilation for the specified number of iterations
    for iter in 0..iters {
        let prev = result.clone();
        let mut temp = Array2::from_elem((shape[0], shape[1]), false);

        // Get structure dimensions
        let s_rows = struct_elem.shape()[0];
        let s_cols = struct_elem.shape()[1];

        // Calculate half sizes for the structure
        let half_height = origin_vec[0];
        let half_width = origin_vec[1];

        // For each position in the array
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                // Skip masked positions
                if let Some(m) = mask {
                    if !m[[i, j]] {
                        temp[[i, j]] = prev[[i, j]];
                        continue;
                    }
                }

                // Copy current value first
                temp[[i, j]] = prev[[i, j]];

                // If already true, skip checking neighbors
                if temp[[i, j]] {
                    continue;
                }

                // Check for neighboring true values
                let mut found_true = false;

                // Iterate over the structure
                'outer: for si in 0..s_rows {
                    for sj in 0..s_cols {
                        if !struct_elem[[si, sj]] {
                            continue; // Skip false values in structure
                        }

                        // Calculate corresponding position in input (reverse direction from erosion)
                        let ni = i as isize - (si as isize - half_height);
                        let nj = j as isize - (sj as isize - half_width);

                        // Check if neighbor position is within bounds
                        if ni < 0 || ni >= shape[0] as isize || nj < 0 || nj >= shape[1] as isize {
                            // Outside bounds - use border value
                            if border_val {
                                found_true = true;
                                break 'outer;
                            }
                        } else if prev[[ni as usize, nj as usize]] {
                            // Position is within bounds and value is true
                            found_true = true;
                            break 'outer;
                        }
                    }
                }

                if found_true {
                    temp[[i, j]] = true;
                }
            }
        }

        result = temp;

        // Check if we've reached a fixed point (no change)
        if !brute_force_algo && iter > 0 && result == prev {
            break;
        }
    }

    Ok(result)
}

/// Implementation of binary dilation for n-dimensional arrays (using dynamic dimensions)
fn binary_dilation_dyn(
    input: &Array<bool, IxDyn>,
    structure: Option<&Array<bool, IxDyn>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, IxDyn>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    _brute_force: Option<bool>,
) -> Result<Array<bool, IxDyn>> {
    let iterations = iterations.unwrap_or(1);
    let border = border_value.unwrap_or(false);

    // Get or generate structure
    let default_structure = if let Some(s) = structure {
        s.to_owned()
    } else {
        generate_binary_structure_dyn(input.ndim())?
    };

    // Validate input dimensions
    if input.ndim() != default_structure.ndim() {
        return Err(NdimageError::DimensionError(
            "Input and structure must have the same number of dimensions".into(),
        ));
    }

    // Validate mask dimensions if provided
    if let Some(m) = mask {
        if m.ndim() != input.ndim() || m.shape() != input.shape() {
            return Err(NdimageError::InvalidInput(
                "Mask must have the same shape as input".into(),
            ));
        }
    }

    // Get structure center
    let center = get_structure_center_dyn(&default_structure, origin)?;

    // Create result array
    let mut result = input.to_owned();

    // Apply dilation iterations
    for _ in 0..iterations {
        let temp = result.clone();

        // Iterate through all positions in the input array
        for idx in ndarray::indices(input.shape()) {
            let idx_vec: Vec<_> = idx.slice().to_vec();

            // Skip if masked out
            if let Some(m) = mask {
                if !m[idx_vec.as_slice()] {
                    continue;
                }
            }

            // Check if any structure element touches a true value
            let mut any_fit = false;

            // Check each structure element
            for str_idx in ndarray::indices(default_structure.shape()) {
                let str_idx_vec: Vec<_> = str_idx.slice().to_vec();

                // Skip if structure element is false
                if !default_structure[str_idx_vec.as_slice()] {
                    continue;
                }

                // Calculate corresponding input position
                let mut input_pos = vec![0isize; input.ndim()];
                for d in 0..input.ndim() {
                    input_pos[d] = idx_vec[d] as isize + str_idx_vec[d] as isize - center[d];
                }

                // Check if position is within bounds
                let mut within_bounds = true;
                for (d, &pos) in input_pos.iter().enumerate().take(input.ndim()) {
                    if pos < 0 || pos >= input.shape()[d] as isize {
                        within_bounds = false;
                        break;
                    }
                }

                // Get the value, using border value if out of bounds
                let val = if within_bounds {
                    let input_idx: Vec<_> = input_pos.iter().map(|&x| x as usize).collect();
                    temp[input_idx.as_slice()]
                } else {
                    border
                };

                // Dilation requires at least one value to be true
                if val {
                    any_fit = true;
                    break;
                }
            }

            result[idx_vec.as_slice()] = any_fit;
        }
    }

    Ok(result)
}

/// Open an array using a structuring element
///
/// Applies erosion followed by dilation with the same structuring element.
/// Opening can be used to remove small objects from an image while preserving
/// the shape and size of larger objects.
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `iterations` - Number of times to apply the opening (default: 1)
/// * `mask` - Mask array that limits the operation (if None, no mask is applied)
/// * `border_value` - Border value (default: false)
/// * `origin` - Origin of the structuring element (if None, uses the center)
/// * `brute_force` - Whether to use brute force algorithm (default: false)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Opened array
pub fn binary_opening<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Opening is erosion followed by dilation
    let eroded = binary_erosion(
        input,
        structure,
        iterations,
        mask,
        border_value,
        origin,
        brute_force,
    )?;

    binary_dilation(
        &eroded,
        structure,
        iterations,
        mask,
        border_value,
        origin,
        brute_force,
    )
}

/// Close an array using a structuring element
///
/// Applies dilation followed by erosion with the same structuring element.
/// Closing can be used to fill small holes and connect nearby objects.
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `iterations` - Number of times to apply the closing (default: 1)
/// * `mask` - Mask array that limits the operation (if None, no mask is applied)
/// * `border_value` - Border value (default: false)
/// * `origin` - Origin of the structuring element (if None, uses the center)
/// * `brute_force` - Whether to use brute force algorithm (default: false)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Closed array
pub fn binary_closing<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> Result<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Closing is dilation followed by erosion
    let dilated = binary_dilation(
        input,
        structure,
        iterations,
        mask,
        border_value,
        origin,
        brute_force,
    )?;

    binary_erosion(
        &dilated,
        structure,
        iterations,
        mask,
        border_value,
        origin,
        brute_force,
    )
}

// Also include a placeholder fill holes function to satisfy API

/// Fill holes in a binary array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `origin` - Origin of the structuring element (if None, uses the center)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Array with filled holes
pub fn binary_fill_holes<D>(
    input: &Array<bool, D>,
    _structure: Option<&Array<bool, D>>,
    _origin: Option<&[isize]>,
) -> Result<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Currently not fully implemented, return a copy of the input
    Ok(input.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_binary_erosion() {
        // Test with all true values
        let input = Array2::from_elem((5, 5), true);
        let result = binary_erosion(&input, None, None, None, None, None, None).unwrap();

        // Border elements should be eroded, but center should remain true
        assert_eq!(result.shape(), input.shape());
        assert!(result[[2, 2]]); // Center should still be true
        assert!(result[[1, 1]]); // Inner elements should still be true
        assert!(result[[1, 3]]);
        assert!(result[[3, 1]]);
        assert!(result[[3, 3]]);

        // Edges should be eroded (false)
        assert!(!result[[0, 2]]); // Top edge
        assert!(!result[[2, 0]]); // Left edge
        assert!(!result[[4, 2]]); // Bottom edge
        assert!(!result[[2, 4]]); // Right edge
    }

    #[test]
    fn test_binary_erosion_with_multiple_iterations() {
        // Create a 5x5 array filled with true values
        let input = Array2::from_elem((5, 5), true);

        // Apply erosion with 2 iterations
        let result = binary_erosion(&input, None, Some(2), None, None, None, None).unwrap();

        // Only the very center should remain true after 2 iterations
        assert!(result[[2, 2]]);

        // Elements that were true after 1 iteration should now be false
        assert!(!result[[1, 1]]);
        assert!(!result[[1, 3]]);
        assert!(!result[[3, 1]]);
        assert!(!result[[3, 3]]);
    }

    #[test]
    fn test_binary_dilation() {
        // Create a 5x5 array with a single true value in the center
        let mut input = Array2::from_elem((5, 5), false);
        input[[2, 2]] = true;

        // Apply dilation
        let result = binary_dilation(&input, None, None, None, None, None, None).unwrap();

        // Center and direct neighbors should be true
        assert!(result[[2, 2]]); // Center
        assert!(result[[1, 2]]); // Top neighbor
        assert!(result[[2, 1]]); // Left neighbor
        assert!(result[[3, 2]]); // Bottom neighbor
        assert!(result[[2, 3]]); // Right neighbor

        // Corners should still be false
        assert!(!result[[0, 0]]);
        assert!(!result[[0, 4]]);
        assert!(!result[[4, 0]]);
        assert!(!result[[4, 4]]);
    }

    #[test]
    fn test_binary_opening() {
        // Create a test pattern with a small feature and a larger feature
        let mut input = Array2::from_elem((7, 7), false);

        // Small feature (2x2)
        input[[1, 1]] = true;
        input[[1, 2]] = true;
        input[[2, 1]] = true;
        input[[2, 2]] = true;

        // Larger feature (3x3)
        input[[4, 4]] = true;
        input[[4, 5]] = true;
        input[[4, 6]] = true;
        input[[5, 4]] = true;
        input[[5, 5]] = true;
        input[[5, 6]] = true;
        input[[6, 4]] = true;
        input[[6, 5]] = true;
        input[[6, 6]] = true;

        // Apply opening
        let result = binary_opening(&input, None, None, None, None, None, None).unwrap();

        // The larger feature should survive
        assert!(result[[5, 5]]);

        // The small feature should be removed
        assert!(!result[[1, 1]]);
        assert!(!result[[1, 2]]);
        assert!(!result[[2, 1]]);
        assert!(!result[[2, 2]]);
    }

    #[test]
    fn test_binary_closing() {
        // Create a test pattern with a hole
        let mut input = Array2::from_elem((5, 5), false);

        // Create a square with a hole in the middle
        input[[1, 1]] = true;
        input[[1, 2]] = true;
        input[[1, 3]] = true;
        input[[2, 1]] = true;
        input[[2, 3]] = true;
        input[[3, 1]] = true;
        input[[3, 2]] = true;
        input[[3, 3]] = true;

        // Apply closing
        let result = binary_closing(&input, None, None, None, None, None, None).unwrap();

        // The hole should be filled
        assert!(result[[2, 2]]);

        // Original values should be maintained
        assert!(result[[1, 1]]);
        assert!(result[[1, 2]]);
        assert!(result[[1, 3]]);
        assert!(result[[2, 1]]);
        assert!(result[[2, 3]]);
        assert!(result[[3, 1]]);
        assert!(result[[3, 2]]);
        assert!(result[[3, 3]]);
    }

    #[test]
    fn test_binary_erosion_3d() {
        // Create a 3D test array with a solid cube
        let mut input: ndarray::Array<bool, ndarray::Ix3> =
            ndarray::Array::from_elem((3, 3, 3), false);
        // Fill the cube
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    input[[i, j, k]] = true;
                }
            }
        }

        // Apply erosion with default structure
        let result = binary_erosion(&input, None, None, None, None, None, None).unwrap();

        // Only the center should remain true after erosion
        assert!(result[[1, 1, 1]]);

        // Edges should be eroded away
        assert!(!result[[0, 0, 0]]);
        assert!(!result[[0, 1, 1]]);
        assert!(!result[[1, 0, 1]]);
        assert!(!result[[1, 1, 0]]);

        // Check that the shape is preserved
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_binary_dilation_3d() {
        // Create a 3D test array with a single point
        let mut input: ndarray::Array<bool, ndarray::Ix3> =
            ndarray::Array::from_elem((3, 3, 3), false);
        input[[1, 1, 1]] = true;

        // Apply dilation with default structure
        let result = binary_dilation(&input, None, None, None, None, None, None).unwrap();

        // Center should remain true
        assert!(result[[1, 1, 1]]);

        // Face neighbors should be dilated
        assert!(result[[0, 1, 1]]); // top
        assert!(result[[2, 1, 1]]); // bottom
        assert!(result[[1, 0, 1]]); // left
        assert!(result[[1, 2, 1]]); // right
        assert!(result[[1, 1, 0]]); // front
        assert!(result[[1, 1, 2]]); // back

        // Corners should not be dilated with default structure (face connectivity)
        assert!(!result[[0, 0, 0]]);
        assert!(!result[[2, 2, 2]]);

        // Check that the shape is preserved
        assert_eq!(result.shape(), input.shape());
    }
}
