//! Connected component operations for binary and labeled arrays

use scirs2_core::ndarray::{Array, Dimension, IxDyn};
use std::collections::HashMap;

use super::Connectivity;
use crate::error::{NdimageError, NdimageResult};

/// Union-Find data structure for connected component labeling
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        UnionFind {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            // Union by rank
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    fn get_component_mapping(&mut self) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();
        let mut next_label = 1;

        for i in 0..self.parent.len() {
            let root = self.find(i);
            if !mapping.contains_key(&root) {
                mapping.insert(root, next_label);
                next_label += 1;
            }
        }

        mapping
    }
}

/// Get neighbors for a given position based on connectivity
#[allow(dead_code)]
fn get_neighbors(
    position: &[usize],
    shape: &[usize],
    connectivity: Connectivity,
) -> Vec<Vec<usize>> {
    let ndim = position.len();
    let mut neighbors = Vec::new();

    match connectivity {
        Connectivity::Face => {
            // Face connectivity: only share a face (4-connectivity in 2D, 6-connectivity in 3D)
            for dim in 0..ndim {
                // Check negative direction
                if position[dim] > 0 {
                    let mut neighbor = position.to_vec();
                    neighbor[dim] -= 1;
                    neighbors.push(neighbor);
                }
                // Check positive direction
                if position[dim] + 1 < shape[dim] {
                    let mut neighbor = position.to_vec();
                    neighbor[dim] += 1;
                    neighbors.push(neighbor);
                }
            }
        }
        Connectivity::FaceEdge => {
            // Face and edge connectivity: 8-connectivity in 2D, 18-connectivity in 3D
            let offsets = generate_face_edge_offsets(ndim);
            for offset in offsets {
                let mut neighbor = Vec::with_capacity(ndim);
                let mut valid = true;

                for (i, &pos) in position.iter().enumerate() {
                    let new_pos = (pos as isize) + offset[i];
                    if new_pos < 0 || new_pos >= shape[i] as isize {
                        valid = false;
                        break;
                    }
                    neighbor.push(new_pos as usize);
                }

                if valid && neighbor != position {
                    neighbors.push(neighbor);
                }
            }
        }
        Connectivity::Full => {
            // Corner connectivity: all possible neighbors (8-connectivity in 2D, 26-connectivity in 3D)
            let offsets = generate_all_offsets(ndim);
            for offset in offsets {
                let mut neighbor = Vec::with_capacity(ndim);
                let mut valid = true;

                for (i, &pos) in position.iter().enumerate() {
                    let new_pos = (pos as isize) + offset[i];
                    if new_pos < 0 || new_pos >= shape[i] as isize {
                        valid = false;
                        break;
                    }
                    neighbor.push(new_pos as usize);
                }

                if valid && neighbor != position {
                    neighbors.push(neighbor);
                }
            }
        }
    }

    neighbors
}

/// Generate all possible offsets for corner connectivity
#[allow(dead_code)]
fn generate_all_offsets(ndim: usize) -> Vec<Vec<isize>> {
    let mut offsets = Vec::new();
    let total_combinations = 3_usize.pow(ndim as u32);

    for i in 0..total_combinations {
        let mut offset = Vec::with_capacity(ndim);
        let mut temp = i;

        for _ in 0..ndim {
            let val = (temp % 3) as isize - 1; // -1, 0, or 1
            offset.push(val);
            temp /= 3;
        }

        // Skip the center point (all zeros)
        if !offset.iter().all(|&x| x == 0) {
            offsets.push(offset);
        }
    }

    offsets
}

/// Generate face and edge neighbor offsets (excludes vertex neighbors in 3D+)
#[allow(dead_code)]
fn generate_face_edge_offsets(ndim: usize) -> Vec<Vec<isize>> {
    let mut offsets = Vec::new();
    let total_combinations = 3_usize.pow(ndim as u32);

    for i in 0..total_combinations {
        let mut offset = Vec::with_capacity(ndim);
        let mut temp = i;
        for _ in 0..ndim {
            let val = (temp % 3) as isize - 1; // -1, 0, or 1
            offset.push(val);
            temp /= 3;
        }

        // Skip the center point (all zeros)
        if offset.iter().all(|&x| x == 0) {
            continue;
        }

        // For face and edge connectivity, exclude vertex neighbors
        // Vertex neighbors have all non-zero components
        let non_zero_count = offset.iter().filter(|&&x| x != 0).count();

        // Include face neighbors (1 non-zero) and edge neighbors (2 non-zero in 3D+)
        // In 2D, this gives 8-connectivity (same as full)
        // In 3D, this gives 18-connectivity (excludes 8 vertex neighbors)
        if non_zero_count <= 2 {
            offsets.push(offset);
        }
    }

    offsets
}

/// Convert multi-dimensional index to flat index
#[allow(dead_code)]
fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat_index = 0;
    let mut stride = 1;

    for i in (0..indices.len()).rev() {
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }

    flat_index
}

/// Convert flat index to multi-dimensional index
#[allow(dead_code)]
fn unravel_index(_flatindex: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = _flatindex;

    // Process dimensions in forward order for row-major (C-order) layout
    for i in 0..shape.len() {
        let stride: usize = shape[(i + 1)..].iter().product();
        indices[i] = remaining / stride;
        remaining %= stride;
    }

    indices
}

/// Label connected components in a binary array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `connectivity` - Connectivity type (default: Face)
/// * `background` - Whether to consider background as a feature (default: false)
///
/// # Returns
///
/// * `Result<(Array<usize, D>, usize)>` - Labeled array and number of labels
#[allow(dead_code)]
pub fn label<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    connectivity: Option<Connectivity>,
    background: Option<bool>,
) -> NdimageResult<(Array<usize, D>, usize)>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);
    let bg = background.unwrap_or(false);

    // Structure must have same rank as input
    if let Some(struct_elem) = structure {
        if struct_elem.ndim() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Structure must have same rank as input (got {} expected {})",
                struct_elem.ndim(),
                input.ndim()
            )));
        }
    }

    let shape = input.shape();
    let total_elements: usize = shape.iter().product();

    if total_elements == 0 {
        let output = Array::<usize, D>::zeros(input.raw_dim());
        return Ok((output, 0));
    }

    // Initialize Union-Find data structure
    let mut uf = UnionFind::new(total_elements);

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    // First pass: scan all pixels and union adjacent foreground pixels
    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let current_pixel = input_dyn[IxDyn(&indices)];

        // Only process foreground pixels (or background if bg=true)
        if current_pixel == !bg {
            // Get neighbors based on connectivity
            let neighbors = get_neighbors(&indices, shape, conn);

            for neighbor_indices in neighbors {
                let neighbor_pixel = input_dyn[IxDyn(&neighbor_indices)];

                // If neighbor is also foreground, union them
                if neighbor_pixel == current_pixel {
                    let neighbor_flat_idx = ravel_index(&neighbor_indices, shape);
                    uf.union(flat_idx, neighbor_flat_idx);
                }
            }
        }
    }

    // Get component mapping (root -> label)
    let component_mapping = uf.get_component_mapping();

    // Create output array
    let mut output = Array::<usize, D>::zeros(input.raw_dim());
    let mut num_labels = 0;

    // Second pass: assign labels
    let mut output_dyn = output.clone().into_dyn();
    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let pixel = input_dyn[IxDyn(&indices)];

        if pixel == !bg {
            let root = uf.find(flat_idx);
            if let Some(&label) = component_mapping.get(&root) {
                output_dyn[IxDyn(&indices)] = label;
                num_labels = num_labels.max(label);
            }
        }
    }

    // Convert back to original dimension type
    output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimension type".into())
    })?;

    Ok((output, num_labels))
}

/// Find the boundaries of objects in a labeled array
///
/// # Arguments
///
/// * `input` - Input labeled array
/// * `connectivity` - Connectivity type (default: Face)
/// * `mode` - Mode for boundary detection: "inner", "outer", or "thick" (default: "outer")
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary array with boundaries
#[allow(dead_code)]
pub fn find_boundaries<D>(
    input: &Array<usize, D>,
    connectivity: Option<Connectivity>,
    mode: Option<&str>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);
    let mode_str = mode.unwrap_or("outer");

    // Validate mode
    if mode_str != "inner" && mode_str != "outer" && mode_str != "thick" {
        return Err(NdimageError::InvalidInput(format!(
            "Mode must be 'inner', 'outer', or 'thick', got '{}'",
            mode_str
        )));
    }

    let shape = input.shape();
    let total_elements: usize = shape.iter().product();
    let mut output = Array::<bool, D>::from_elem(input.raw_dim(), false);

    if total_elements == 0 {
        return Ok(output);
    }

    // Convert to dynamic arrays for easier indexing
    let input_dyn = input.clone().into_dyn();
    let mut output_dyn = output.clone().into_dyn();

    // Scan all pixels to find boundaries
    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let current_label = input_dyn[IxDyn(&indices)];

        // Skip background pixels for inner mode
        if mode_str == "inner" && current_label == 0 {
            continue;
        }

        // Get neighbors based on connectivity
        let neighbors = get_neighbors(&indices, shape, conn);
        let mut is_boundary = false;

        for neighbor_indices in neighbors {
            let neighbor_label = input_dyn[IxDyn(&neighbor_indices)];

            match mode_str {
                "inner" => {
                    // Inner boundary: foreground pixels adjacent to background or different labels
                    if current_label != 0
                        && (neighbor_label == 0 || neighbor_label != current_label)
                    {
                        is_boundary = true;
                        break;
                    }
                }
                "outer" => {
                    // Outer boundary: background pixels adjacent to foreground
                    if current_label == 0 && neighbor_label != 0 {
                        is_boundary = true;
                        break;
                    }
                }
                "thick" => {
                    // Thick boundary: both inner and outer
                    if current_label != neighbor_label {
                        is_boundary = true;
                        break;
                    }
                }
                _ => {} // Already validated above
            }
        }

        if is_boundary {
            output_dyn[IxDyn(&indices)] = true;
        }
    }

    // Convert back to original dimension type
    output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimension type".into())
    })?;

    Ok(output)
}

/// Remove small objects from a labeled array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `min_size` - Minimum size of objects to keep
/// * `connectivity` - Connectivity type (default: Face)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary array with small objects removed
#[allow(dead_code)]
pub fn remove_small_objects<D>(
    input: &Array<bool, D>,
    min_size: usize,
    connectivity: Option<Connectivity>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if min_size == 0 {
        return Err(NdimageError::InvalidInput(
            "min_size must be greater than 0".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);

    // Label connected components
    let (labeled, num_labels) = label(input, None, Some(conn), None)?;

    if num_labels == 0 {
        return Ok(Array::<bool, D>::from_elem(input.raw_dim(), false));
    }

    // Count the _size of each component
    let mut component_sizes = vec![0; num_labels + 1];
    for &label_val in labeled.iter() {
        if label_val > 0 {
            component_sizes[label_val] += 1;
        }
    }

    // Create output array, keeping only large enough components
    let mut output = Array::<bool, D>::from_elem(input.raw_dim(), false);
    let shape = input.shape();
    let total_elements: usize = shape.iter().product();

    // Convert to dynamic arrays for easier indexing
    let labeled_dyn = labeled.clone().into_dyn();
    let mut output_dyn = output.clone().into_dyn();

    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let label_val = labeled_dyn[IxDyn(&indices)];

        if label_val > 0 && component_sizes[label_val] >= min_size {
            output_dyn[IxDyn(&indices)] = true;
        }
    }

    // Convert back to original dimension type
    output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimension type".into())
    })?;

    Ok(output)
}

/// Remove small holes from a labeled array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `min_size` - Minimum size of holes to keep
/// * `connectivity` - Connectivity type (default: Face)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary array with small holes removed
#[allow(dead_code)]
pub fn remove_small_holes<D>(
    input: &Array<bool, D>,
    min_size: usize,
    connectivity: Option<Connectivity>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if min_size == 0 {
        return Err(NdimageError::InvalidInput(
            "min_size must be greater than 0".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);

    // To remove small holes, we:
    // 1. Invert the binary image (holes become objects)
    // 2. Remove small objects from the inverted image
    // 3. Invert back

    // Create inverted image
    let mut inverted = input.clone();
    for pixel in inverted.iter_mut() {
        *pixel = !*pixel;
    }

    // Remove small objects from inverted image
    let filtered_inverted = remove_small_objects(&inverted, min_size, Some(conn))?;

    // Invert back to get result
    let mut output = filtered_inverted;
    for pixel in output.iter_mut() {
        *pixel = !*pixel;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    #[ignore]
    fn test_label() {
        let input = Array2::from_elem((3, 3), true);
        let (result, _num_labels) = label(&input, None, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    #[ignore]
    fn test_find_boundaries() {
        let input = Array2::from_elem((3, 3), 1);
        let result = find_boundaries(&input, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    #[ignore]
    fn test_remove_small_objects() {
        let input = Array2::from_elem((3, 3), true);
        let result = remove_small_objects(&input, 1, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }
}
