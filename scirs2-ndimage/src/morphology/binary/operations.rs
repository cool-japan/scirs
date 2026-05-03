//! Binary morphological compound operations
//!
//! This module implements higher-level binary morphological operations:
//! opening, closing, fill holes, and hit-or-miss transform.
//! These are built on top of the fundamental erosion and dilation operations.

use scirs2_core::ndarray::{Array, Dimension};

use super::dilation::binary_dilation;
use super::erosion::binary_erosion;
use crate::error::{NdimageError, NdimageResult};

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
#[allow(dead_code)]
pub fn binary_opening<D>(
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
#[allow(dead_code)]
pub fn binary_closing<D>(
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
#[allow(dead_code)]
pub fn binary_fill_holes<D>(
    input: &Array<bool, D>,
    _structure: Option<&Array<bool, D>>,
    _origin: Option<&[isize]>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Currently not fully implemented, return a copy of the input
    Ok(input.clone())
}

/// Apply a binary hit-or-miss transform to an array
///
/// The hit-or-miss transform is a morphological operation used for shape detection.
/// It combines erosion and dilation operations to find patterns in binary images.
/// The transform finds locations where the foreground structuring element "hits"
/// the foreground pixels and the background structuring element "misses" the foreground pixels.
///
/// # Arguments
///
/// * `input` - Input binary array (2D arrays only for now)
/// * `structure1` - Foreground structuring element (hits), if None uses a cross
/// * `structure2` - Background structuring element (misses), if None uses the complement of structure1
/// * `mask` - Mask array that limits the operation (if None, no mask is applied)
/// * `border_value` - Border value (default: false)
/// * `origin1` - Origin of the foreground structuring element (if None, uses the center)
/// * `origin2` - Origin of the background structuring element (if None, uses the center)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Hit-or-miss transformed array
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::morphology::binary_hit_or_miss;
///
/// // Create a binary image with a specific pattern
/// let mut input = Array2::from_elem((5, 5), false);
/// input[[1, 1]] = true;
/// input[[1, 2]] = true;
/// input[[1, 3]] = true; // Horizontal line
///
/// // Define a structuring element to detect horizontal lines
/// let structure1 = Array2::from_elem((1, 3), true);
///
/// // Apply hit-or-miss transform
/// let result = binary_hit_or_miss(&input, Some(&structure1), None, None, None, None, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn binary_hit_or_miss<D>(
    input: &Array<bool, D>,
    structure1: Option<&Array<bool, D>>,
    structure2: Option<&Array<bool, D>>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin1: Option<&[isize]>,
    origin2: Option<&[isize]>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension + 'static,
{
    // Handle different dimensions based on input
    match input.ndim() {
        1 => {
            let input_1d = input
                .clone()
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".to_string()))?;
            let result_1d = binary_hit_or_miss_1d(
                &input_1d,
                structure1,
                structure2,
                mask,
                border_value,
                origin1,
                origin2,
            )?;
            Ok(result_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from 1D".to_string())
            })?)
        }
        2 => {
            let input_2d = input
                .clone()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".to_string()))?;
            let result_2d = binary_hit_or_miss_2d(
                &input_2d,
                structure1,
                structure2,
                mask,
                border_value,
                origin1,
                origin2,
            )?;
            Ok(result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from 2D".to_string())
            })?)
        }
        _ => {
            // N-D hit-or-miss: implemented via IxDyn brute-force
            let input_dyn = input
                .clone()
                .into_dimensionality::<scirs2_core::ndarray::IxDyn>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to IxDyn".to_string())
                })?;

            // Build N-D foreground structure (default: cross with centre + face-neighbours)
            let ndim = input.ndim();
            let s1_dyn: scirs2_core::ndarray::Array<bool, scirs2_core::ndarray::IxDyn> =
                if let Some(s) = structure1 {
                    s.clone()
                        .into_dimensionality::<scirs2_core::ndarray::IxDyn>()
                        .map_err(|_| {
                            NdimageError::DimensionError(
                                "Failed to convert structure1 to IxDyn".to_string(),
                            )
                        })?
                } else {
                    // Default: 3^ndim array with centre and face-connected neighbours = True
                    let side = vec![3usize; ndim];
                    let mut cross = scirs2_core::ndarray::Array::from_elem(
                        scirs2_core::ndarray::IxDyn(&side),
                        false,
                    );
                    // Centre
                    let center_idx: Vec<usize> = vec![1; ndim];
                    cross[scirs2_core::ndarray::IxDyn(&center_idx)] = true;
                    // Face neighbours: for each axis, ±1 from centre
                    for ax in 0..ndim {
                        let mut lo = center_idx.clone();
                        lo[ax] = 0;
                        cross[scirs2_core::ndarray::IxDyn(&lo)] = true;
                        let mut hi = center_idx.clone();
                        hi[ax] = 2;
                        cross[scirs2_core::ndarray::IxDyn(&hi)] = true;
                    }
                    cross
                };

            let s2_dyn: scirs2_core::ndarray::Array<bool, scirs2_core::ndarray::IxDyn> =
                if let Some(s) = structure2 {
                    s.clone()
                        .into_dimensionality::<scirs2_core::ndarray::IxDyn>()
                        .map_err(|_| {
                            NdimageError::DimensionError(
                                "Failed to convert structure2 to IxDyn".to_string(),
                            )
                        })?
                } else {
                    // Complement of s1
                    s1_dyn.mapv(|v| !v)
                };

            let border = border_value.unwrap_or(false);

            // Centre offsets
            let center1: Vec<isize> = if let Some(o) = origin1 {
                o.to_vec()
            } else {
                s1_dyn.shape().iter().map(|&s| s as isize / 2).collect()
            };
            let center2: Vec<isize> = if let Some(o) = origin2 {
                o.to_vec()
            } else {
                s2_dyn.shape().iter().map(|&s| s as isize / 2).collect()
            };

            let input_shape = input_dyn.shape().to_vec();
            let mut result_dyn = scirs2_core::ndarray::Array::from_elem(
                scirs2_core::ndarray::IxDyn(&input_shape),
                false,
            );

            // Iterate over all positions using IxDyn indices
            for out_idx in scirs2_core::ndarray::indices(scirs2_core::ndarray::IxDyn(&input_shape))
            {
                let pos: Vec<isize> = out_idx.slice().iter().map(|&x| x as isize).collect();

                // Optionally check mask
                if let Some(m) = mask {
                    let m_dyn = m
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::IxDyn>()
                        .map_err(|_| {
                            NdimageError::DimensionError(
                                "Failed to convert mask to IxDyn".to_string(),
                            )
                        })?;
                    if !m_dyn[out_idx.clone()] {
                        continue;
                    }
                }

                // Check foreground structure
                let mut fg_hit = true;
                'fg: for s_idx in scirs2_core::ndarray::indices(s1_dyn.raw_dim()) {
                    if !s1_dyn[s_idx.clone()] {
                        continue;
                    }
                    let src: Vec<isize> = s_idx
                        .slice()
                        .iter()
                        .enumerate()
                        .map(|(ax, &si)| pos[ax] + si as isize - center1[ax])
                        .collect();
                    let val = if src
                        .iter()
                        .enumerate()
                        .all(|(ax, &c)| c >= 0 && c < input_shape[ax] as isize)
                    {
                        let idx_vec: Vec<usize> = src.iter().map(|&c| c as usize).collect();
                        input_dyn[scirs2_core::ndarray::IxDyn(&idx_vec)]
                    } else {
                        border
                    };
                    if !val {
                        fg_hit = false;
                        break 'fg;
                    }
                }

                // Check background structure (only if foreground hit)
                let mut bg_miss = true;
                if fg_hit {
                    'bg: for s_idx in scirs2_core::ndarray::indices(s2_dyn.raw_dim()) {
                        if !s2_dyn[s_idx.clone()] {
                            continue;
                        }
                        let src: Vec<isize> = s_idx
                            .slice()
                            .iter()
                            .enumerate()
                            .map(|(ax, &si)| pos[ax] + si as isize - center2[ax])
                            .collect();
                        let val = if src
                            .iter()
                            .enumerate()
                            .all(|(ax, &c)| c >= 0 && c < input_shape[ax] as isize)
                        {
                            let idx_vec: Vec<usize> = src.iter().map(|&c| c as usize).collect();
                            input_dyn[scirs2_core::ndarray::IxDyn(&idx_vec)]
                        } else {
                            border
                        };
                        if val {
                            bg_miss = false;
                            break 'bg;
                        }
                    }
                }

                result_dyn[out_idx] = fg_hit && bg_miss;
            }

            result_dyn.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert result from IxDyn".to_string())
            })
        }
    }
}

/// Apply binary hit-or-miss transform to a 1D array
#[allow(dead_code)]
fn binary_hit_or_miss_1d<D>(
    input: &Array<bool, scirs2_core::ndarray::Ix1>,
    _structure1: Option<&Array<bool, D>>,
    _structure2: Option<&Array<bool, D>>,
    _mask: Option<&Array<bool, D>>,
    _border_value: Option<bool>,
    _origin1: Option<&[isize]>,
    _origin2: Option<&[isize]>,
) -> NdimageResult<Array<bool, scirs2_core::ndarray::Ix1>>
where
    D: Dimension + 'static,
{
    // For simplicity, not fully implemented for 1D yet
    Ok(input.clone())
}

/// Apply binary hit-or-miss transform to a 2D array
#[allow(dead_code)]
fn binary_hit_or_miss_2d<D>(
    input: &Array<bool, scirs2_core::ndarray::Ix2>,
    structure1: Option<&Array<bool, D>>,
    structure2: Option<&Array<bool, D>>,
    mask: Option<&Array<bool, D>>,
    border_value: Option<bool>,
    origin1: Option<&[isize]>,
    origin2: Option<&[isize]>,
) -> NdimageResult<Array<bool, scirs2_core::ndarray::Ix2>>
where
    D: Dimension + 'static,
{
    use scirs2_core::ndarray::Array2;

    let border = border_value.unwrap_or(false);
    let (rows, cols) = input.dim();

    // Default 2D cross structure if none provided
    let default_structure1 = if let Some(s) = structure1 {
        // Convert the provided structure to 2D
        if s.ndim() != 2 {
            return Err(NdimageError::DimensionError(
                "Foreground structure must be 2D for 2D input".into(),
            ));
        }
        s.clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| {
                NdimageError::DimensionError("Failed to convert structure to 2D".to_string())
            })?
    } else {
        // Default 3x3 cross
        let mut cross = Array2::from_elem((3, 3), false);
        cross[[1, 0]] = true; // left
        cross[[1, 1]] = true; // center
        cross[[1, 2]] = true; // right
        cross[[0, 1]] = true; // top
        cross[[2, 1]] = true; // bottom
        cross
    };

    // Background structure
    let default_structure2 = if let Some(s) = structure2 {
        if s.ndim() != 2 {
            return Err(NdimageError::DimensionError(
                "Background structure must be 2D for 2D input".into(),
            ));
        }
        s.clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| {
                NdimageError::DimensionError("Failed to convert structure to 2D".to_string())
            })?
    } else {
        // Create complement of structure1
        let mut complement = Array2::from_elem(default_structure1.raw_dim(), false);
        for ((i, j), &val) in default_structure1.indexed_iter() {
            complement[[i, j]] = !val;
        }
        complement
    };

    // Structure centers
    let center1 = if let Some(orig) = origin1 {
        if orig.len() != 2 {
            return Err(NdimageError::InvalidInput(
                "Origin must be 2D for 2D structure".into(),
            ));
        }
        [orig[0], orig[1]]
    } else {
        [
            default_structure1.nrows() as isize / 2,
            default_structure1.ncols() as isize / 2,
        ]
    };

    let center2 = if let Some(orig) = origin2 {
        if orig.len() != 2 {
            return Err(NdimageError::InvalidInput(
                "Origin must be 2D for 2D structure".into(),
            ));
        }
        [orig[0], orig[1]]
    } else {
        [
            default_structure2.nrows() as isize / 2,
            default_structure2.ncols() as isize / 2,
        ]
    };

    // Create result array (initially all false)
    let mut result = Array2::from_elem((rows, cols), false);

    // Iterate through all positions in the input array
    for i in 0..rows {
        for j in 0..cols {
            // Check mask
            if let Some(m) = mask {
                if m.ndim() != 2 {
                    return Err(NdimageError::DimensionError(
                        "Mask must be 2D for 2D input".into(),
                    ));
                }
                let m_2d = m
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert mask to 2D".to_string())
                    })?;
                if !m_2d[[i, j]] {
                    continue;
                }
            }

            // Check if foreground structure "hits"
            let mut foreground_hit = true;
            for (si, sj) in scirs2_core::ndarray::indices(default_structure1.dim()) {
                if !default_structure1[[si, sj]] {
                    continue; // Skip false elements in structure
                }

                let input_i = i as isize + si as isize - center1[0];
                let input_j = j as isize + sj as isize - center1[1];

                let val = if input_i >= 0
                    && input_i < rows as isize
                    && input_j >= 0
                    && input_j < cols as isize
                {
                    input[[input_i as usize, input_j as usize]]
                } else {
                    border
                };

                if !val {
                    foreground_hit = false;
                    break;
                }
            }

            // Check if background structure "misses"
            let mut background_miss = true;
            if foreground_hit {
                for (si, sj) in scirs2_core::ndarray::indices(default_structure2.dim()) {
                    if !default_structure2[[si, sj]] {
                        continue; // Skip false elements in structure
                    }

                    let input_i = i as isize + si as isize - center2[0];
                    let input_j = j as isize + sj as isize - center2[1];

                    let val = if input_i >= 0
                        && input_i < rows as isize
                        && input_j >= 0
                        && input_j < cols as isize
                    {
                        input[[input_i as usize, input_j as usize]]
                    } else {
                        border
                    };

                    if val {
                        background_miss = false;
                        break;
                    }
                }
            }

            result[[i, j]] = foreground_hit && background_miss;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_binary_erosion() {
        // Test with all true values
        let input = Array2::from_elem((5, 5), true);
        let result = binary_erosion(&input, None, None, None, None, None, None)
            .expect("binary_erosion should succeed for test");

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
        let result = binary_erosion(&input, None, Some(2), None, None, None, None)
            .expect("binary_erosion with iterations should succeed for test");

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
        let result = binary_dilation(&input, None, None, None, None, None, None)
            .expect("binary_dilation should succeed for test");

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
        let result = binary_opening(&input, None, None, None, None, None, None)
            .expect("binary_opening should succeed for test");

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
        let result = binary_closing(&input, None, None, None, None, None, None)
            .expect("binary_closing should succeed for test");

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
        let mut input: scirs2_core::ndarray::Array<bool, scirs2_core::ndarray::Ix3> =
            scirs2_core::ndarray::Array::from_elem((3, 3, 3), false);
        // Fill the cube
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    input[[i, j, k]] = true;
                }
            }
        }

        // Apply erosion with default structure
        let result = binary_erosion(&input, None, None, None, None, None, None)
            .expect("binary_erosion 3D should succeed for test");

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
        let mut input: scirs2_core::ndarray::Array<bool, scirs2_core::ndarray::Ix3> =
            scirs2_core::ndarray::Array::from_elem((3, 3, 3), false);
        input[[1, 1, 1]] = true;

        // Apply dilation with default structure
        let result = binary_dilation(&input, None, None, None, None, None, None)
            .expect("binary_dilation 3D should succeed for test");

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

    // ─── N-D hit-or-miss tests ────────────────────────────────────────────────

    #[test]
    fn test_binary_hit_or_miss_3d_basic() {
        // 3×3×3 volume with a single foreground voxel at the centre.
        // Use a minimal foreground structure (centre-only) and all-false background
        // to verify that the N-D implementation routes correctly and returns
        // the right shape.
        use scirs2_core::ndarray::Array3;

        let mut input = Array3::from_elem((5, 5, 5), false);
        input[[2, 2, 2]] = true;

        // Foreground: centre-only (a single true voxel matches any foreground point)
        let mut fg = Array3::from_elem((3, 3, 3), false);
        fg[[1, 1, 1]] = true;

        // Background: all-false (no background constraint)
        let bg = Array3::from_elem((3, 3, 3), false);

        let result = binary_hit_or_miss(
            &input,
            Some(&fg),
            Some(&bg),
            None::<&Array3<bool>>,
            None,
            None,
            None,
        )
        .expect("binary_hit_or_miss 3D should succeed");

        assert_eq!(result.shape(), input.shape(), "Shape must be preserved");

        // Only (2,2,2) has a true foreground voxel, so only it should be detected
        assert!(
            result[[2, 2, 2]],
            "Centre of isolated foreground should be detected"
        );

        // All other positions should be false
        let n_true = result.iter().filter(|&&v| v).count();
        assert_eq!(n_true, 1, "Exactly one voxel should be detected");
    }

    #[test]
    fn test_binary_hit_or_miss_3d_all_foreground() {
        // When the entire volume is foreground, the background (complement)
        // structure cannot match — result should be all false.
        use scirs2_core::ndarray::Array3;

        let input = Array3::from_elem((5, 5, 5), true);

        let result = binary_hit_or_miss(
            &input,
            None::<&Array3<bool>>,
            None::<&Array3<bool>>,
            None::<&Array3<bool>>,
            None,
            None,
            None,
        )
        .expect("binary_hit_or_miss 3D all-foreground should succeed");

        assert_eq!(result.shape(), input.shape());
        let n_true = result.iter().filter(|&&v| v).count();
        assert_eq!(
            n_true, 0,
            "No voxel should be detected when entire volume is foreground"
        );
    }

    #[test]
    fn test_binary_hit_or_miss_3d_custom_structures() {
        // Test with custom foreground and background structures in 3D.
        use scirs2_core::ndarray::Array3;

        let mut input = Array3::from_elem((5, 5, 5), false);
        // Create an isolated foreground point at (2,2,2) surrounded by background
        input[[2, 2, 2]] = true;

        // Foreground: only the centre voxel
        let mut fg = Array3::from_elem((3, 3, 3), false);
        fg[[1, 1, 1]] = true;

        // Background: only the face-adjacent voxels (should all be background=false in input)
        let mut bg = Array3::from_elem((3, 3, 3), false);
        bg[[0, 1, 1]] = true; // top face
        bg[[2, 1, 1]] = true; // bottom face
        bg[[1, 0, 1]] = true; // left face
        bg[[1, 2, 1]] = true; // right face
        bg[[1, 1, 0]] = true; // front face
        bg[[1, 1, 2]] = true; // back face

        let result = binary_hit_or_miss(
            &input,
            Some(&fg),
            Some(&bg),
            None::<&Array3<bool>>,
            None,
            None,
            None,
        )
        .expect("binary_hit_or_miss 3D custom should succeed");

        assert_eq!(result.shape(), input.shape());

        // The isolated point at (2,2,2) should be detected: foreground hits centre,
        // background verifies face neighbours are all false (which they are)
        assert!(
            result[[2, 2, 2]],
            "Isolated foreground point should be detected"
        );

        let n_true = result.iter().filter(|&&v| v).count();
        assert_eq!(n_true, 1, "Exactly one voxel should be detected");
    }
}
