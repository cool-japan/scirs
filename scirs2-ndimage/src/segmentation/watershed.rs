//! Watershed segmentation algorithm
//!
//! This module provides the watershed segmentation algorithm for image segmentation.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array, Ix2};
use scirs2_core::numeric::{Float, NumAssign};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Point with coordinates and priority for the watershed queue
///
/// For floating point types, we create a special wrapper that implements Eq
#[derive(Clone, Debug)]
struct PriorityPoint<T> {
    coords: Vec<usize>,
    priority: T,
}

impl<T: PartialEq> PartialEq for PriorityPoint<T> {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl<T: PartialEq> Eq for PriorityPoint<T> {}

impl<T: PartialEq> Hash for PriorityPoint<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coords.hash(state);
    }
}

impl<T: PartialOrd> PartialOrd for PriorityPoint<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // For NaN values, prioritize by coordinates to ensure deterministic behavior
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for PriorityPoint<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order to make BinaryHeap a min-heap
        // Use partial_cmp but provide a fallback for NaN values
        match other.priority.partial_cmp(&self.priority) {
            Some(ordering) => ordering,
            None => Ordering::Equal, // Fallback for NaN comparisons
        }
    }
}

/// Watersheds segmentation for 2D arrays
///
/// The watershed algorithm treats the image as a topographic surface,
/// where bright pixels are high and dark pixels are low. It segments
/// the image into catchment basins.
///
/// # Arguments
///
/// * `image` - Input array (intensity gradient image)
/// * `markers` - Initial markers array (same shape as input, with unique positive values
///   for each region to be segmented, and 0 for unknown regions)
///
/// # Returns
///
/// * Result containing the labeled segmented image
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::watershed;
///
/// let image = array![
///     [0.5, 0.6, 0.7],
///     [0.4, 0.1, 0.2],
///     [0.3, 0.4, 0.5],
/// ];
///
/// let markers = array![
///     [0, 0, 0],
///     [0, 1, 0],
///     [0, 0, 2],
/// ];
///
/// let segmented = watershed(&image, &markers).unwrap();
/// ```
#[allow(dead_code)]
pub fn watershed<T>(
    image: &Array<T, Ix2>,
    markers: &Array<i32, Ix2>,
) -> NdimageResult<Array<i32, Ix2>>
where
    T: Float + NumAssign + std::fmt::Debug + std::ops::DivAssign + 'static,
{
    // Check shapes match
    if image.shape() != markers.shape() {
        return Err(NdimageError::DimensionError(
            "Input image and markers must have the same shape".to_string(),
        ));
    }

    // Create output array with initial markers
    let mut output = markers.clone();

    // Create priority queue with markers
    let mut queue = BinaryHeap::new();

    // Keep track of processed points
    let mut processed = HashSet::new();

    // Process each marker
    for ((r, c), &marker) in markers.indexed_iter() {
        if marker != 0 {
            let pos = vec![r, c];
            processed.insert(pos.clone());

            // Get neighbor points of this marker and add to queue
            for neighbor in get_neighbors_2d(image, r, c) {
                let neighbor_pos = neighbor.coords.clone();

                // Only process neighbors that aren't already marked
                if !processed.contains(&neighbor_pos) {
                    let nr = neighbor_pos[0];
                    let nc = neighbor_pos[1];

                    let value = image[[nr, nc]];
                    queue.push(PriorityPoint {
                        coords: neighbor_pos.clone(),
                        priority: value,
                    });
                    processed.insert(neighbor_pos);
                }
            }
        }
    }

    // Process the queue
    while let Some(point) = queue.pop() {
        let pos = point.coords.clone();
        let r = pos[0];
        let c = pos[1];

        // Get the neighbors of this point that are already labeled
        let mut neighbor_labels = HashMap::new();

        for neighbor in get_neighbors_2d(image, r, c) {
            let nr = neighbor.coords[0];
            let nc = neighbor.coords[1];

            let label = output[[nr, nc]];
            if label > 0 {
                *neighbor_labels.entry(label).or_insert(0) += 1;
            }
        }

        // Determine the label for this point
        if let Some((most_common_label_, _)) =
            neighbor_labels.iter().max_by_key(|&(_, count)| count)
        {
            output[[r, c]] = *most_common_label_;

            // Add new neighbors to the queue
            for neighbor in get_neighbors_2d(image, r, c) {
                let neighbor_pos = neighbor.coords.clone();
                let nr = neighbor_pos[0];
                let nc = neighbor_pos[1];

                if !processed.contains(&neighbor_pos) {
                    let value = image[[nr, nc]];
                    let label = output[[nr, nc]];

                    if label == 0 {
                        queue.push(PriorityPoint {
                            coords: neighbor_pos.clone(),
                            priority: value,
                        });
                        processed.insert(neighbor_pos);
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Get the neighbors of a point in a 2D array
#[allow(dead_code)]
fn get_neighbors_2d<T>(array: &Array<T, Ix2>, row: usize, col: usize) -> Vec<PriorityPoint<T>>
where
    T: Float + std::fmt::Debug + Copy + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let mut neighbors = Vec::new();
    let shape = array.shape();
    let rows = shape[0];
    let cols = shape[1];

    // Define the 8 possible neighbors (including diagonals)
    let offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    for (dr, dc) in offsets.iter() {
        let new_row = row as isize + dr;
        let new_col = col as isize + dc;

        // Check boundaries
        if new_row >= 0 && new_row < rows as isize && new_col >= 0 && new_col < cols as isize {
            let nr = new_row as usize;
            let nc = new_col as usize;

            let value = array[[nr, nc]];
            neighbors.push(PriorityPoint {
                coords: vec![nr, nc],
                priority: value,
            });
        }
    }

    neighbors
}

/// Get the face-connected neighbors of a point in a 2D array (4-connectivity)
#[allow(dead_code)]
fn get_face_neighbors_2d<T>(array: &Array<T, Ix2>, row: usize, col: usize) -> Vec<PriorityPoint<T>>
where
    T: Float + std::fmt::Debug + Copy + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let mut neighbors = Vec::new();
    let shape = array.shape();
    let rows = shape[0];
    let cols = shape[1];

    // Define the 4 possible neighbors (no diagonals)
    let offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)];

    for (dr, dc) in offsets.iter() {
        let new_row = row as isize + dr;
        let new_col = col as isize + dc;

        // Check boundaries
        if new_row >= 0 && new_row < rows as isize && new_col >= 0 && new_col < cols as isize {
            let nr = new_row as usize;
            let nc = new_col as usize;

            let value = array[[nr, nc]];
            neighbors.push(PriorityPoint {
                coords: vec![nr, nc],
                priority: value,
            });
        }
    }

    neighbors
}

/// Marker-controlled watershed for 2D arrays
///
/// A variant of the watershed algorithm that uses specified markers
/// as seeds and a gradient image to find the boundaries.
///
/// # Arguments
///
/// * `image` - Input array (intensity image)
/// * `markers` - Initial markers array (same shape as input, with unique positive values
///   for each region to be segmented, and 0 for unknown regions)
/// * `connectivity` - Connectivity for considering neighbors (1: face-connected, 2: fully-connected)
///
/// # Returns
///
/// * Result containing the labeled segmented image
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::marker_watershed;
///
/// let image = array![
///     [0.5, 0.5, 0.5],
///     [0.5, 0.2, 0.5],
///     [0.5, 0.5, 0.5],
/// ];
///
/// let markers = array![
///     [0, 0, 0],
///     [1, 0, 2],
///     [0, 0, 0],
/// ];
///
/// let segmented = marker_watershed(&image, &markers, 1).unwrap();
/// ```
#[allow(dead_code)]
pub fn marker_watershed<T>(
    image: &Array<T, Ix2>,
    markers: &Array<i32, Ix2>,
    connectivity: usize,
) -> NdimageResult<Array<i32, Ix2>>
where
    T: Float + NumAssign + std::fmt::Debug + Copy + 'static,
{
    // Check shapes match
    if image.shape() != markers.shape() {
        return Err(NdimageError::DimensionError(
            "Input image and markers must have the same shape".to_string(),
        ));
    }

    // Check connectivity
    if connectivity != 1 && connectivity != 2 {
        return Err(NdimageError::InvalidInput(
            "Connectivity must be 1 or 2".to_string(),
        ));
    }

    // Create output array with initial markers
    let mut output = markers.clone();

    // Create a queue of pixels sorted by value (lowest first)
    let mut q = BinaryHeap::new();

    // Add marker pixels to the queue
    for ((r, c), &marker) in markers.indexed_iter() {
        if marker > 0 {
            // Get neighbors based on connectivity
            let neighbors = if connectivity == 1 {
                get_face_neighbors_2d(image, r, c)
            } else {
                get_neighbors_2d(image, r, c)
            };

            for neighbor in neighbors {
                let nr = neighbor.coords[0];
                let nc = neighbor.coords[1];

                // Only consider unlabeled neighbors
                if output[[nr, nc]] == 0 {
                    q.push(Reverse(PriorityPoint {
                        coords: neighbor.coords,
                        priority: neighbor.priority,
                    }));
                }
            }
        }
    }

    // Process the queue
    while let Some(Reverse(point)) = q.pop() {
        let pos = point.coords.clone();
        let r = pos[0];
        let c = pos[1];

        // Skip if this pixel is already labeled
        if output[[r, c]] != 0 {
            continue;
        }

        // Get the neighbors of this point that are already labeled
        let mut labels = HashMap::new();

        let neighbors = if connectivity == 1 {
            get_face_neighbors_2d(image, r, c)
        } else {
            get_neighbors_2d(image, r, c)
        };

        for neighbor in neighbors {
            let nr = neighbor.coords[0];
            let nc = neighbor.coords[1];

            let label = output[[nr, nc]];
            if label > 0 {
                *labels.entry(label).or_insert(0) += 1;
            }
        }

        // If we have labeled neighbors, assign this pixel to the most frequent label
        if !labels.is_empty() {
            let (most_common_label_, _) = labels.iter().max_by_key(|&(_, count)| count).unwrap();

            output[[r, c]] = *most_common_label_;

            // Add unlabeled neighbors to the queue
            let new_neighbors = if connectivity == 1 {
                get_face_neighbors_2d(image, r, c)
            } else {
                get_neighbors_2d(image, r, c)
            };

            for neighbor in new_neighbors {
                let nr = neighbor.coords[0];
                let nc = neighbor.coords[1];

                if output[[nr, nc]] == 0 {
                    q.push(Reverse(PriorityPoint {
                        coords: neighbor.coords.clone(),
                        priority: neighbor.priority,
                    }));
                }
            }
        }
    }

    Ok(output)
}
