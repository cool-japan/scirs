//! Binary dilation operations
//!
//! This module implements binary dilation for 1D, 2D, and n-dimensional arrays.
//! Dilation adds pixels to the boundaries of regions of positive pixels,
//! effectively expanding these regions.

use scirs2_core::ndarray::{Array, Array1, Array2, Dimension, Ix1, Ix2, IxDyn};

use crate::error::{NdimageError, NdimageResult};
use crate::morphology::structuring::generate_binary_structure_dyn;
use crate::morphology::utils::get_structure_center_dyn;

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
#[allow(dead_code)]
pub fn binary_dilation<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> NdimageResult<Array<bool, D>>
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
#[allow(dead_code)]
fn binary_dilation1d(
    input: &Array1<bool>,
    structure: Option<&Array1<bool>>,
    iterations: Option<usize>,
    mask: Option<&Array1<bool>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> NdimageResult<Array1<bool>> {
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

            // Initialize current position _value
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
                    // Outside bounds - use border _value
                    if border_val {
                        *val = true;
                        break;
                    }
                } else if prev[pos as usize] {
                    // Position has a true _value in input
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
#[allow(dead_code)]
fn binary_dilation2d(
    input: &Array2<bool>,
    structure: Option<&Array2<bool>>,
    iterations: Option<usize>,
    mask: Option<&Array2<bool>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    brute_force: Option<bool>,
) -> NdimageResult<Array2<bool>> {
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

                // Copy current _value first
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
                            // Outside bounds - use border _value
                            if border_val {
                                found_true = true;
                                break 'outer;
                            }
                        } else if prev[[ni as usize, nj as usize]] {
                            // Position is within bounds and _value is true
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
#[allow(dead_code)]
fn binary_dilation_dyn(
    input: &Array<bool, IxDyn>,
    structure: Option<&Array<bool, IxDyn>>,
    iterations: Option<usize>,
    mask: Option<&Array<bool, IxDyn>>,
    border_value: Option<bool>,
    origin: Option<&[isize]>,
    _brute_force: Option<bool>,
) -> NdimageResult<Array<bool, IxDyn>> {
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
        for idx in scirs2_core::ndarray::indices(input.shape()) {
            let idx_vec: Vec<_> = idx.slice().to_vec();

            // Skip if masked out
            if let Some(m) = mask {
                if !m[idx_vec.as_slice()] {
                    continue;
                }
            }

            // Check if any structure element touches a true _value
            let mut any_fit = false;

            // Check each structure element
            for str_idx in scirs2_core::ndarray::indices(default_structure.shape()) {
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

                // Get the value, using border _value if out of bounds
                let val = if within_bounds {
                    let input_idx: Vec<_> = input_pos.iter().map(|&x| x as usize).collect();
                    temp[input_idx.as_slice()]
                } else {
                    border
                };

                // Dilation requires at least one _value to be true
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
