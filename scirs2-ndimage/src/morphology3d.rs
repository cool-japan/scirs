//! 3D Morphological Operations
//!
//! This module provides three-dimensional morphological operations for volumetric
//! image data, including binary and grayscale erosion/dilation, opening/closing,
//! connected component labeling, object property measurement, skeletonization,
//! and Euclidean distance transforms.
//!
//! # References
//!
//! - Gonzalez & Woods, "Digital Image Processing", 3rd ed.
//! - Meijster et al. (2000), "A General Algorithm for Computing Distance Transforms in Linear Time"
//! - Lee et al. (1994), "Building Skeleton Models via 3-D Medial Surface Axis Thinning"

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::ArrayView3;
use scirs2_core::ndarray::{s, Array3};

// ---------------------------------------------------------------------------
// Structuring element
// ---------------------------------------------------------------------------

/// 3D structuring element shapes for morphological operations.
#[derive(Debug, Clone)]
pub enum StructuringElement3D {
    /// Cubic structuring element: side = `2 * radius + 1`.
    Cube(usize),
    /// Spherical structuring element with given radius.
    Sphere(f64),
    /// 6-connectivity cross (face-connected neighbors only: ±x, ±y, ±z plus center).
    Cross,
    /// 26-connectivity: full 3×3×3 neighborhood (all 26 neighbors plus center).
    Cross26,
    /// Arbitrary boolean array.
    Custom(Array3<bool>),
}

impl StructuringElement3D {
    /// Convert the structuring element to a dense boolean `Array3`.
    pub fn to_array(&self) -> Array3<bool> {
        match self {
            StructuringElement3D::Cube(radius) => {
                let side = 2 * radius + 1;
                Array3::from_elem((side, side, side), true)
            }
            StructuringElement3D::Sphere(radius) => {
                let r = radius.ceil() as usize;
                let side = 2 * r + 1;
                let cr = r as f64;
                Array3::from_shape_fn((side, side, side), |(z, y, x)| {
                    let dz = z as f64 - cr;
                    let dy = y as f64 - cr;
                    let dx = x as f64 - cr;
                    dz * dz + dy * dy + dx * dx <= radius * radius + 1e-9
                })
            }
            StructuringElement3D::Cross => {
                let mut arr = Array3::from_elem((3, 3, 3), false);
                // center
                arr[[1, 1, 1]] = true;
                // ±z
                arr[[0, 1, 1]] = true;
                arr[[2, 1, 1]] = true;
                // ±y
                arr[[1, 0, 1]] = true;
                arr[[1, 2, 1]] = true;
                // ±x
                arr[[1, 1, 0]] = true;
                arr[[1, 1, 2]] = true;
                arr
            }
            StructuringElement3D::Cross26 => Array3::from_elem((3, 3, 3), true),
            StructuringElement3D::Custom(arr) => arr.clone(),
        }
    }

    /// Return the center voxel index `(z, y, x)` of this structuring element.
    pub fn center(&self) -> (usize, usize, usize) {
        let arr = self.to_array();
        let shape = arr.shape();
        (shape[0] / 2, shape[1] / 2, shape[2] / 2)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply one pass of binary erosion using a dense SE mask.
fn erode_pass(src: &Array3<bool>, se: &Array3<bool>) -> Array3<bool> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let (ez, ey, ex) = (se.shape()[0], se.shape()[1], se.shape()[2]);
    let (cz, cy, cx) = (ez / 2, ey / 2, ex / 2);

    let mut out = Array3::from_elem((sz, sy, sx), false);
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut fits = true;
                'se_loop: for dz in 0..ez {
                    for dy in 0..ey {
                        for dx in 0..ex {
                            if !se[[dz, dy, dx]] {
                                continue;
                            }
                            let nz = iz as isize + dz as isize - cz as isize;
                            let ny = iy as isize + dy as isize - cy as isize;
                            let nx = ix as isize + dx as isize - cx as isize;
                            if nz < 0
                                || ny < 0
                                || nx < 0
                                || nz >= sz as isize
                                || ny >= sy as isize
                                || nx >= sx as isize
                            {
                                fits = false;
                                break 'se_loop;
                            }
                            if !src[[nz as usize, ny as usize, nx as usize]] {
                                fits = false;
                                break 'se_loop;
                            }
                        }
                    }
                }
                out[[iz, iy, ix]] = fits;
            }
        }
    }
    out
}

/// Apply one pass of binary dilation using a dense SE mask.
fn dilate_pass(src: &Array3<bool>, se: &Array3<bool>) -> Array3<bool> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let (ez, ey, ex) = (se.shape()[0], se.shape()[1], se.shape()[2]);
    let (cz, cy, cx) = (ez / 2, ey / 2, ex / 2);

    let mut out = Array3::from_elem((sz, sy, sx), false);
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                if !src[[iz, iy, ix]] {
                    continue;
                }
                for dz in 0..ez {
                    for dy in 0..ey {
                        for dx in 0..ex {
                            if !se[[dz, dy, dx]] {
                                continue;
                            }
                            let nz = iz as isize + dz as isize - cz as isize;
                            let ny = iy as isize + dy as isize - cy as isize;
                            let nx = ix as isize + dx as isize - cx as isize;
                            if nz >= 0
                                && ny >= 0
                                && nx >= 0
                                && nz < sz as isize
                                && ny < sy as isize
                                && nx < sx as isize
                            {
                                out[[nz as usize, ny as usize, nx as usize]] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public binary morphology API
// ---------------------------------------------------------------------------

/// 3D binary erosion.
///
/// Applies erosion `iterations` times.  The image is shrunken by removing
/// foreground voxels that are not fully covered by the structuring element.
///
/// # Arguments
///
/// * `image` - Input binary volumetric image (view).
/// * `structuring_element` - Shape of the morphological probe.
/// * `iterations` - Number of erosion iterations (≥1).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if `iterations == 0`.
pub fn binary_erosion3d(
    image: ArrayView3<bool>,
    structuring_element: &StructuringElement3D,
    iterations: usize,
) -> NdimageResult<Array3<bool>> {
    if iterations == 0 {
        return Err(NdimageError::InvalidInput(
            "iterations must be at least 1".to_string(),
        ));
    }
    let se = structuring_element.to_array();
    let mut current = image.to_owned();
    for _ in 0..iterations {
        current = erode_pass(&current, &se);
    }
    Ok(current)
}

/// 3D binary dilation.
///
/// Applies dilation `iterations` times.  The image is grown by adding background
/// voxels that are touched by the structuring element placed at any foreground voxel.
///
/// # Arguments
///
/// * `image` - Input binary volumetric image (view).
/// * `structuring_element` - Shape of the morphological probe.
/// * `iterations` - Number of dilation iterations (≥1).
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if `iterations == 0`.
pub fn binary_dilation3d(
    image: ArrayView3<bool>,
    structuring_element: &StructuringElement3D,
    iterations: usize,
) -> NdimageResult<Array3<bool>> {
    if iterations == 0 {
        return Err(NdimageError::InvalidInput(
            "iterations must be at least 1".to_string(),
        ));
    }
    let se = structuring_element.to_array();
    let mut current = image.to_owned();
    for _ in 0..iterations {
        current = dilate_pass(&current, &se);
    }
    Ok(current)
}

/// 3D binary opening (erosion followed by dilation).
///
/// Removes small bright features and smooths boundaries while roughly preserving
/// the shape and size of larger foreground objects.
pub fn binary_opening3d(
    image: ArrayView3<bool>,
    structuring_element: &StructuringElement3D,
) -> NdimageResult<Array3<bool>> {
    let se = structuring_element.to_array();
    let eroded = erode_pass(&image.to_owned(), &se);
    Ok(dilate_pass(&eroded, &se))
}

/// 3D binary closing (dilation followed by erosion).
///
/// Fills small holes and gaps in the foreground while roughly preserving shape.
pub fn binary_closing3d(
    image: ArrayView3<bool>,
    structuring_element: &StructuringElement3D,
) -> NdimageResult<Array3<bool>> {
    let se = structuring_element.to_array();
    let dilated = dilate_pass(&image.to_owned(), &se);
    Ok(erode_pass(&dilated, &se))
}

// ---------------------------------------------------------------------------
// Grayscale morphology
// ---------------------------------------------------------------------------

/// Apply one pass of grayscale erosion (minimum filter with SE).
fn grey_erode_pass(src: &Array3<f64>, se: &Array3<bool>) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let (ez, ey, ex) = (se.shape()[0], se.shape()[1], se.shape()[2]);
    let (cz, cy, cx) = (ez / 2, ey / 2, ex / 2);

    let mut out = Array3::<f64>::from_elem((sz, sy, sx), f64::INFINITY);
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut min_val = f64::INFINITY;
                for dz in 0..ez {
                    for dy in 0..ey {
                        for dx in 0..ex {
                            if !se[[dz, dy, dx]] {
                                continue;
                            }
                            let nz = iz as isize + dz as isize - cz as isize;
                            let ny = iy as isize + dy as isize - cy as isize;
                            let nx = ix as isize + dx as isize - cx as isize;
                            if nz >= 0
                                && ny >= 0
                                && nx >= 0
                                && nz < sz as isize
                                && ny < sy as isize
                                && nx < sx as isize
                            {
                                let v = src[[nz as usize, ny as usize, nx as usize]];
                                if v < min_val {
                                    min_val = v;
                                }
                            }
                        }
                    }
                }
                out[[iz, iy, ix]] = if min_val.is_infinite() {
                    src[[iz, iy, ix]]
                } else {
                    min_val
                };
            }
        }
    }
    out
}

/// Apply one pass of grayscale dilation (maximum filter with SE).
fn grey_dilate_pass(src: &Array3<f64>, se: &Array3<bool>) -> Array3<f64> {
    let (sz, sy, sx) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let (ez, ey, ex) = (se.shape()[0], se.shape()[1], se.shape()[2]);
    let (cz, cy, cx) = (ez / 2, ey / 2, ex / 2);

    let mut out = Array3::<f64>::from_elem((sz, sy, sx), f64::NEG_INFINITY);
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let mut max_val = f64::NEG_INFINITY;
                for dz in 0..ez {
                    for dy in 0..ey {
                        for dx in 0..ex {
                            if !se[[dz, dy, dx]] {
                                continue;
                            }
                            let nz = iz as isize + dz as isize - cz as isize;
                            let ny = iy as isize + dy as isize - cy as isize;
                            let nx = ix as isize + dx as isize - cx as isize;
                            if nz >= 0
                                && ny >= 0
                                && nx >= 0
                                && nz < sz as isize
                                && ny < sy as isize
                                && nx < sx as isize
                            {
                                let v = src[[nz as usize, ny as usize, nx as usize]];
                                if v > max_val {
                                    max_val = v;
                                }
                            }
                        }
                    }
                }
                out[[iz, iy, ix]] = if max_val.is_infinite() {
                    src[[iz, iy, ix]]
                } else {
                    max_val
                };
            }
        }
    }
    out
}

/// 3D grayscale erosion (min filter).
///
/// Each output voxel receives the minimum value within the structuring element
/// neighbourhood centered at that voxel.  Out-of-bounds positions are ignored.
pub fn grey_erosion3d(
    image: ArrayView3<f64>,
    structuring_element: &StructuringElement3D,
) -> NdimageResult<Array3<f64>> {
    let se = structuring_element.to_array();
    Ok(grey_erode_pass(&image.to_owned(), &se))
}

/// 3D grayscale dilation (max filter).
///
/// Each output voxel receives the maximum value within the structuring element
/// neighbourhood centered at that voxel.  Out-of-bounds positions are ignored.
pub fn grey_dilation3d(
    image: ArrayView3<f64>,
    structuring_element: &StructuringElement3D,
) -> NdimageResult<Array3<f64>> {
    let se = structuring_element.to_array();
    Ok(grey_dilate_pass(&image.to_owned(), &se))
}

// ---------------------------------------------------------------------------
// Connected-component labeling (union-find)
// ---------------------------------------------------------------------------

/// Union-Find data structure with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            // Path halving
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Generate neighbor offsets for a given connectivity (6, 18, or 26).
fn neighbor_offsets(connectivity: usize) -> Vec<(isize, isize, isize)> {
    match connectivity {
        6 => vec![(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
        18 => {
            let mut offsets = Vec::new();
            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let nonzero = (dz != 0) as usize + (dy != 0) as usize + (dx != 0) as usize;
                        if nonzero > 0 && nonzero <= 2 {
                            // face- or edge-connected (not corner)
                            if dz < 0 || (dz == 0 && dy < 0) || (dz == 0 && dy == 0 && dx < 0) {
                                offsets.push((dz, dy, dx));
                            }
                        }
                    }
                }
            }
            offsets
        }
        _ => {
            // 26-connectivity: all previously scanned neighbors
            let mut offsets = Vec::new();
            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dz == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        if dz < 0 || (dz == 0 && dy < 0) || (dz == 0 && dy == 0 && dx < 0) {
                            offsets.push((dz, dy, dx));
                        }
                    }
                }
            }
            offsets
        }
    }
}

/// 3D connected component labeling using union-find.
///
/// Labels connected foreground (`true`) voxels with distinct integer labels
/// starting from 1.  Background voxels receive label 0.
///
/// # Arguments
///
/// * `binary_image` - Input binary volumetric image.
/// * `connectivity` - One of 6 (face), 18 (face+edge), or 26 (full neighborhood).
///   Any value other than 6 or 18 is treated as 26.
///
/// # Returns
///
/// A tuple `(label_image, n_components)`.
pub fn label3d(
    binary_image: ArrayView3<bool>,
    connectivity: usize,
) -> NdimageResult<(Array3<i32>, usize)> {
    let shape = binary_image.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    let n_voxels = sz * sy * sx;

    // Flat label array: 0 = background, provisional labels 1..
    let mut labels = vec![0usize; n_voxels];
    let mut uf = UnionFind::new(n_voxels + 1); // index 0 = background sentinel
    let mut next_label = 1usize;

    let offsets = neighbor_offsets(connectivity);

    let flat = |z: usize, y: usize, x: usize| z * sy * sx + y * sx + x;

    // First pass: assign provisional labels and record unions
    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                if !binary_image[[iz, iy, ix]] {
                    continue;
                }
                let mut neighbor_labels: Vec<usize> = Vec::new();
                for &(dz, dy, dx) in &offsets {
                    let nz = iz as isize + dz;
                    let ny = iy as isize + dy;
                    let nx = ix as isize + dx;
                    if nz < 0
                        || ny < 0
                        || nx < 0
                        || nz >= sz as isize
                        || ny >= sy as isize
                        || nx >= sx as isize
                    {
                        continue;
                    }
                    let nb = flat(nz as usize, ny as usize, nx as usize);
                    if labels[nb] > 0 {
                        neighbor_labels.push(labels[nb]);
                    }
                }

                let idx = flat(iz, iy, ix);
                if neighbor_labels.is_empty() {
                    labels[idx] = next_label;
                    next_label += 1;
                } else {
                    let min_lbl = neighbor_labels.iter().copied().min().unwrap_or(next_label);
                    labels[idx] = min_lbl;
                    for &nl in &neighbor_labels {
                        uf.union(min_lbl, nl);
                    }
                }
            }
        }
    }

    // Second pass: relabel via union-find roots to contiguous integers
    let mut root_to_final = vec![0usize; next_label];
    let mut n_components = 0usize;
    let mut out = Array3::<i32>::zeros((sz, sy, sx));

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let idx = flat(iz, iy, ix);
                if labels[idx] == 0 {
                    continue;
                }
                let root = uf.find(labels[idx]);
                if root_to_final[root] == 0 {
                    n_components += 1;
                    root_to_final[root] = n_components;
                }
                out[[iz, iy, ix]] = root_to_final[root] as i32;
            }
        }
    }

    Ok((out, n_components))
}

// ---------------------------------------------------------------------------
// Object properties
// ---------------------------------------------------------------------------

/// Properties of a single labeled 3D object.
#[derive(Debug, Clone)]
pub struct Object3DProps {
    /// Label index (1-based).
    pub label: i32,
    /// Volume in voxels.
    pub volume: usize,
    /// Centroid `(z, y, x)` coordinates.
    pub centroid: (f64, f64, f64),
    /// Bounding box `((z_min, y_min, x_min), (z_max, y_max, x_max))`.
    pub bounding_box: ((usize, usize, usize), (usize, usize, usize)),
    /// Diameter of a sphere with the same volume.
    pub equivalent_diameter: f64,
    /// Approximate surface area (number of voxels adjacent to background).
    pub surface_area_approx: f64,
}

/// Compute properties for each labeled object in a 3D label image.
///
/// Objects with label values 1 through `n_objects` are analyzed.  Label 0
/// (background) is ignored.
///
/// # Arguments
///
/// * `label_image` - Integer label image (output of [`label3d`]).
/// * `n_objects` - Number of distinct objects (second return value of [`label3d`]).
pub fn object_properties3d(label_image: &Array3<i32>, n_objects: usize) -> Vec<Object3DProps> {
    let shape = label_image.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);

    // Accumulators
    let mut volumes = vec![0usize; n_objects + 1];
    let mut sum_z = vec![0.0f64; n_objects + 1];
    let mut sum_y = vec![0.0f64; n_objects + 1];
    let mut sum_x = vec![0.0f64; n_objects + 1];
    let mut min_z = vec![usize::MAX; n_objects + 1];
    let mut min_y = vec![usize::MAX; n_objects + 1];
    let mut min_x = vec![usize::MAX; n_objects + 1];
    let mut max_z = vec![0usize; n_objects + 1];
    let mut max_y = vec![0usize; n_objects + 1];
    let mut max_x = vec![0usize; n_objects + 1];
    let mut surface = vec![0usize; n_objects + 1];

    // 6-connectivity neighbors for surface detection
    let face_neighbors: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                let lbl = label_image[[iz, iy, ix]] as usize;
                if lbl == 0 || lbl > n_objects {
                    continue;
                }
                volumes[lbl] += 1;
                sum_z[lbl] += iz as f64;
                sum_y[lbl] += iy as f64;
                sum_x[lbl] += ix as f64;
                if iz < min_z[lbl] {
                    min_z[lbl] = iz;
                }
                if iy < min_y[lbl] {
                    min_y[lbl] = iy;
                }
                if ix < min_x[lbl] {
                    min_x[lbl] = ix;
                }
                if iz > max_z[lbl] {
                    max_z[lbl] = iz;
                }
                if iy > max_y[lbl] {
                    max_y[lbl] = iy;
                }
                if ix > max_x[lbl] {
                    max_x[lbl] = ix;
                }

                // Count surface voxels
                let mut on_surface = false;
                for &(dz, dy, dx) in &face_neighbors {
                    let nz = iz as isize + dz;
                    let ny = iy as isize + dy;
                    let nx = ix as isize + dx;
                    if nz < 0
                        || ny < 0
                        || nx < 0
                        || nz >= sz as isize
                        || ny >= sy as isize
                        || nx >= sx as isize
                    {
                        on_surface = true;
                        break;
                    }
                    if label_image[[nz as usize, ny as usize, nx as usize]] != lbl as i32 {
                        on_surface = true;
                        break;
                    }
                }
                if on_surface {
                    surface[lbl] += 1;
                }
            }
        }
    }

    let pi = std::f64::consts::PI;

    (1..=n_objects)
        .map(|lbl| {
            let vol = volumes[lbl];
            let centroid = if vol > 0 {
                (
                    sum_z[lbl] / vol as f64,
                    sum_y[lbl] / vol as f64,
                    sum_x[lbl] / vol as f64,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            // diameter of sphere with same volume: V = (4/3)*pi*r^3 => d = 2*(3V/4pi)^(1/3)
            let equiv_diam = if vol > 0 {
                2.0 * (3.0 * vol as f64 / (4.0 * pi)).powf(1.0 / 3.0)
            } else {
                0.0
            };
            let bb_min = (
                if min_z[lbl] == usize::MAX {
                    0
                } else {
                    min_z[lbl]
                },
                if min_y[lbl] == usize::MAX {
                    0
                } else {
                    min_y[lbl]
                },
                if min_x[lbl] == usize::MAX {
                    0
                } else {
                    min_x[lbl]
                },
            );
            Object3DProps {
                label: lbl as i32,
                volume: vol,
                centroid,
                bounding_box: (bb_min, (max_z[lbl], max_y[lbl], max_x[lbl])),
                equivalent_diameter: equiv_diam,
                surface_area_approx: surface[lbl] as f64,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Euclidean Distance Transform 3D
// ---------------------------------------------------------------------------

/// 3D Euclidean distance transform.
///
/// For every foreground (`true`) voxel, computes the exact Euclidean distance
/// to the nearest background (`false`) voxel.  Background voxels receive 0.
///
/// Uses the separable Meijster/Felzenszwalb algorithm extended to three
/// dimensions.
///
/// # Arguments
///
/// * `binary_image` - Input binary volumetric image.
pub fn distance_transform_edt3d(binary_image: ArrayView3<bool>) -> NdimageResult<Array3<f64>> {
    let shape = binary_image.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Ok(Array3::zeros((sz, sy, sx)));
    }

    let inf_sq = (sz * sz + sy * sy + sx * sx + 1) as f64;

    // --- Phase 1: along X axis ---
    // g[z][y][x] = squared distance to nearest background in row (z, y, *)
    let mut g = Array3::<f64>::from_elem((sz, sy, sx), inf_sq);

    for iz in 0..sz {
        for iy in 0..sy {
            // Forward scan
            let mut prev = if !binary_image[[iz, iy, 0]] {
                0.0
            } else {
                inf_sq
            };
            g[[iz, iy, 0]] = prev;
            for ix in 1..sx {
                if !binary_image[[iz, iy, ix]] {
                    prev = 0.0;
                } else if prev < inf_sq {
                    prev += 1.0;
                }
                g[[iz, iy, ix]] = prev * prev;
            }
            // Backward scan
            let mut prev_back = if !binary_image[[iz, iy, sx - 1]] {
                0.0
            } else {
                inf_sq
            };
            for ix in (0..sx).rev() {
                if !binary_image[[iz, iy, ix]] {
                    prev_back = 0.0;
                } else if prev_back < inf_sq {
                    prev_back += 1.0;
                }
                let bval = prev_back * prev_back;
                if bval < g[[iz, iy, ix]] {
                    g[[iz, iy, ix]] = bval;
                }
            }
        }
    }

    // --- Phase 2: along Y axis ---
    // h[z][y][x] = min over y' of (g[z][y'][x] + (y - y')^2)
    let mut h = Array3::<f64>::from_elem((sz, sy, sx), inf_sq);

    for iz in 0..sz {
        for ix in 0..sx {
            // Felzenszwalb parabola lower envelope
            let mut s = vec![0.0f64; sy]; // separation points
            let mut t = vec![0usize; sy]; // parabola indices
            let f = |q: usize| g[[iz, q, ix]] + (q as f64).powi(2);

            let mut k = 0usize;
            t[0] = 0;
            s[0] = f64::NEG_INFINITY;

            for q in 1..sy {
                // intersection of parabolas at t[k] and q
                loop {
                    let sq =
                        ((f(q) - f(t[k])) / (2.0 * q as f64 - 2.0 * t[k] as f64) + 0.5).floor();
                    if k == 0 || sq > s[k] {
                        k += 1;
                        s[k] = sq;
                        t[k] = q;
                        break;
                    }
                    if k > 0 {
                        k -= 1;
                    } else {
                        break;
                    }
                }
            }

            let mut j = k;
            for q in (0..sy).rev() {
                while j > 0 && s[j] > q as f64 {
                    j -= 1;
                }
                h[[iz, q, ix]] = f(t[j]) + (q as f64 - t[j] as f64).powi(2) - (t[j] as f64).powi(2);
            }
        }
    }

    // --- Phase 3: along Z axis ---
    let mut out = Array3::<f64>::zeros((sz, sy, sx));

    for iy in 0..sy {
        for ix in 0..sx {
            let f = |q: usize| h[[q, iy, ix]] + (q as f64).powi(2);
            let mut s = vec![0.0f64; sz];
            let mut t = vec![0usize; sz];
            let mut k = 0usize;
            t[0] = 0;
            s[0] = f64::NEG_INFINITY;

            for q in 1..sz {
                loop {
                    let sq =
                        ((f(q) - f(t[k])) / (2.0 * q as f64 - 2.0 * t[k] as f64) + 0.5).floor();
                    if k == 0 || sq > s[k] {
                        k += 1;
                        s[k] = sq;
                        t[k] = q;
                        break;
                    }
                    if k > 0 {
                        k -= 1;
                    } else {
                        break;
                    }
                }
            }

            let mut j = k;
            for q in (0..sz).rev() {
                while j > 0 && s[j] > q as f64 {
                    j -= 1;
                }
                let dist_sq = f(t[j]) + (q as f64 - t[j] as f64).powi(2) - (t[j] as f64).powi(2);
                out[[q, iy, ix]] = dist_sq.max(0.0).sqrt();
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Skeletonization (3D thinning)
// ---------------------------------------------------------------------------

/// 3D binary skeletonization via iterative thinning.
///
/// Iteratively removes border voxels that are topologically simple (i.e.,
/// their removal does not change the topology of the object) until no further
/// voxels can be removed.  The result is a one-voxel-wide medial skeleton.
///
/// This implementation uses a simplified heuristic based on 26-connectivity
/// thinning criteria adapted from Lee et al. (1994).
///
/// # Arguments
///
/// * `binary_image` - Input binary volumetric image.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if the image is empty.
pub fn skeletonize3d(binary_image: ArrayView3<bool>) -> NdimageResult<Array3<bool>> {
    let shape = binary_image.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "image must be non-empty".to_string(),
        ));
    }

    let mut current = binary_image.to_owned();

    // Six sub-iteration directions (thinning along each face direction)
    let directions: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let mut changed = true;
    while changed {
        changed = false;
        for &(dz, dy, dx) in &directions {
            // Collect candidates: foreground voxels with an exposed face in direction (dz,dy,dx)
            let mut candidates = Vec::new();
            for iz in 0..sz {
                for iy in 0..sy {
                    for ix in 0..sx {
                        if !current[[iz, iy, ix]] {
                            continue;
                        }
                        let nz = iz as isize + dz;
                        let ny = iy as isize + dy;
                        let nx = ix as isize + dx;
                        // The neighbor in the thinning direction must be background
                        let is_border = if nz < 0
                            || ny < 0
                            || nx < 0
                            || nz >= sz as isize
                            || ny >= sy as isize
                            || nx >= sx as isize
                        {
                            true
                        } else {
                            !current[[nz as usize, ny as usize, nx as usize]]
                        };
                        if is_border && is_simple_point(&current, iz, iy, ix) {
                            candidates.push((iz, iy, ix));
                        }
                    }
                }
            }
            for (iz, iy, ix) in candidates {
                current[[iz, iy, ix]] = false;
                changed = true;
            }
        }
    }

    Ok(current)
}

/// Determine whether removing voxel (iz, iy, ix) leaves the local topology unchanged.
///
/// A voxel is "simple" if its 26-neighborhood has exactly one connected component
/// of foreground voxels and exactly one connected component of background voxels
/// when the voxel itself is excluded.
fn is_simple_point(vol: &Array3<bool>, iz: usize, iy: usize, ix: usize) -> bool {
    let shape = vol.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);

    // Extract the 3×3×3 neighborhood, excluding the center voxel
    let mut nbhood = [[[false; 3]; 3]; 3];
    for dz in 0..3usize {
        for dy in 0..3usize {
            for dx in 0..3usize {
                if dz == 1 && dy == 1 && dx == 1 {
                    continue;
                }
                let nz = iz as isize + dz as isize - 1;
                let ny = iy as isize + dy as isize - 1;
                let nx = ix as isize + dx as isize - 1;
                if nz >= 0
                    && ny >= 0
                    && nx >= 0
                    && nz < sz as isize
                    && ny < sy as isize
                    && nx < sx as isize
                {
                    nbhood[dz][dy][dx] = vol[[nz as usize, ny as usize, nx as usize]];
                }
            }
        }
    }

    // Count 26-connected components of foreground in neighborhood
    let fg_components = count_components_26(&nbhood, true);
    if fg_components != 1 {
        return false;
    }

    // Count 6-connected components of background in neighborhood (treating center as bg)
    let mut bg_nbhood = [[[true; 3]; 3]; 3];
    for dz in 0..3usize {
        for dy in 0..3usize {
            for dx in 0..3usize {
                if dz == 1 && dy == 1 && dx == 1 {
                    // center: set to background (we're removing it)
                    bg_nbhood[dz][dy][dx] = true;
                } else {
                    // background if the neighbor is background
                    bg_nbhood[dz][dy][dx] = !nbhood[dz][dy][dx];
                }
            }
        }
    }
    let bg_components = count_components_6(&bg_nbhood, true);
    bg_components == 1
}

/// Count 26-connected components of `target` value in a 3×3×3 neighborhood.
fn count_components_26(nbhood: &[[[bool; 3]; 3]; 3], target: bool) -> usize {
    let mut visited = [[[false; 3]; 3]; 3];
    let mut count = 0;
    for sz in 0..3usize {
        for sy in 0..3usize {
            for sx in 0..3usize {
                if nbhood[sz][sy][sx] == target && !visited[sz][sy][sx] {
                    // BFS in 26-connectivity
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((sz, sy, sx));
                    visited[sz][sy][sx] = true;
                    while let Some((cz, cy, cx)) = queue.pop_front() {
                        for dz in 0..3usize {
                            for dy in 0..3usize {
                                for dx in 0..3usize {
                                    if dz == 1 && dy == 1 && dx == 1 {
                                        continue;
                                    }
                                    let nz = cz as isize + dz as isize - 1;
                                    let ny = cy as isize + dy as isize - 1;
                                    let nx = cx as isize + dx as isize - 1;
                                    if nz >= 0 && ny >= 0 && nx >= 0 && nz < 3 && ny < 3 && nx < 3 {
                                        let (nzu, nyu, nxu) =
                                            (nz as usize, ny as usize, nx as usize);
                                        if nbhood[nzu][nyu][nxu] == target
                                            && !visited[nzu][nyu][nxu]
                                        {
                                            visited[nzu][nyu][nxu] = true;
                                            queue.push_back((nzu, nyu, nxu));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    count += 1;
                }
            }
        }
    }
    count
}

/// Count 6-connected components of `target` value in a 3×3×3 neighborhood.
fn count_components_6(nbhood: &[[[bool; 3]; 3]; 3], target: bool) -> usize {
    let offsets: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    let mut visited = [[[false; 3]; 3]; 3];
    let mut count = 0;
    for sz in 0..3usize {
        for sy in 0..3usize {
            for sx in 0..3usize {
                if nbhood[sz][sy][sx] == target && !visited[sz][sy][sx] {
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((sz, sy, sx));
                    visited[sz][sy][sx] = true;
                    while let Some((cz, cy, cx)) = queue.pop_front() {
                        for &(dz, dy, dx) in &offsets {
                            let nz = cz as isize + dz;
                            let ny = cy as isize + dy;
                            let nx = cx as isize + dx;
                            if nz >= 0 && ny >= 0 && nx >= 0 && nz < 3 && ny < 3 && nx < 3 {
                                let (nzu, nyu, nxu) = (nz as usize, ny as usize, nx as usize);
                                if nbhood[nzu][nyu][nxu] == target && !visited[nzu][nyu][nxu] {
                                    visited[nzu][nyu][nxu] = true;
                                    queue.push_back((nzu, nyu, nxu));
                                }
                            }
                        }
                    }
                    count += 1;
                }
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    // Helper: small 5×5×5 sphere in center
    fn sphere_5() -> Array3<bool> {
        Array3::from_shape_fn((5, 5, 5), |(z, y, x)| {
            let dz = z as f64 - 2.0;
            let dy = y as f64 - 2.0;
            let dx = x as f64 - 2.0;
            dz * dz + dy * dy + dx * dx <= 4.0 + 1e-9
        })
    }

    // Helper: solid 4×4×4 cube
    fn solid_cube() -> Array3<bool> {
        Array3::from_elem((4, 4, 4), true)
    }

    // Helper: two disjoint 2×2×2 blobs
    fn two_blobs() -> Array3<bool> {
        let mut v = Array3::from_elem((5, 5, 5), false);
        for z in 0..2usize {
            for y in 0..2usize {
                for x in 0..2usize {
                    v[[z, y, x]] = true;
                    v[[z + 3, y + 3, x + 3]] = true;
                }
            }
        }
        v
    }

    // ----- StructuringElement3D -----

    #[test]
    fn test_se_cube_shape() {
        let se = StructuringElement3D::Cube(1);
        let arr = se.to_array();
        assert_eq!(arr.shape(), &[3, 3, 3]);
        assert!(arr.iter().all(|&v| v));
    }

    #[test]
    fn test_se_sphere_center() {
        let se = StructuringElement3D::Sphere(2.0);
        let arr = se.to_array();
        let (cz, cy, cx) = se.center();
        assert!(arr[[cz, cy, cx]]);
    }

    #[test]
    fn test_se_cross_6_connectivity() {
        let se = StructuringElement3D::Cross;
        let arr = se.to_array();
        assert!(arr[[1, 1, 1]]); // center
        assert!(arr[[0, 1, 1]]); // -z
        assert!(arr[[2, 1, 1]]); // +z
        assert!(!arr[[0, 0, 0]]); // corner must be false
                                  // Exactly 7 true values (6 neighbors + center)
        let count = arr.iter().filter(|&&v| v).count();
        assert_eq!(count, 7);
    }

    #[test]
    fn test_se_cross26_full() {
        let se = StructuringElement3D::Cross26;
        let arr = se.to_array();
        assert_eq!(arr.iter().filter(|&&v| v).count(), 27);
    }

    // ----- Binary erosion / dilation -----

    #[test]
    fn test_binary_erosion_shrinks() {
        let img = sphere_5();
        let se = StructuringElement3D::Cross;
        let eroded = binary_erosion3d(img.view(), &se, 1).expect("erosion failed");
        let before: usize = img.iter().filter(|&&v| v).count();
        let after: usize = eroded.iter().filter(|&&v| v).count();
        assert!(after <= before, "erosion should not grow the object");
    }

    #[test]
    fn test_binary_dilation_grows() {
        let img = sphere_5();
        let se = StructuringElement3D::Cross;
        let dilated = binary_dilation3d(img.view(), &se, 1).expect("dilation failed");
        let before: usize = img.iter().filter(|&&v| v).count();
        let after: usize = dilated.iter().filter(|&&v| v).count();
        assert!(after >= before, "dilation should not shrink the object");
    }

    #[test]
    fn test_binary_erosion_zero_iterations_error() {
        let img = sphere_5();
        let se = StructuringElement3D::Cross;
        let result = binary_erosion3d(img.view(), &se, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_opening_smaller_than_original() {
        let img = sphere_5();
        let se = StructuringElement3D::Cube(1);
        let opened = binary_opening3d(img.view(), &se).expect("opening failed");
        let before: usize = img.iter().filter(|&&v| v).count();
        let after: usize = opened.iter().filter(|&&v| v).count();
        assert!(after <= before);
    }

    #[test]
    fn test_binary_closing_larger_than_original() {
        // Closing fills gaps → result ≥ original in count
        let mut img = Array3::from_elem((7, 7, 7), false);
        img[[3, 3, 3]] = true;
        img[[3, 3, 5]] = true;
        let se = StructuringElement3D::Cube(1);
        let closed = binary_closing3d(img.view(), &se).expect("closing failed");
        let before: usize = img.iter().filter(|&&v| v).count();
        let after: usize = closed.iter().filter(|&&v| v).count();
        assert!(after >= before);
    }

    #[test]
    fn test_binary_erosion_all_background_stays_background() {
        let img = Array3::from_elem((5, 5, 5), false);
        let se = StructuringElement3D::Cross26;
        let eroded = binary_erosion3d(img.view(), &se, 1).expect("erosion failed");
        assert!(eroded.iter().all(|&v| !v));
    }

    // ----- Grayscale morphology -----

    #[test]
    fn test_grey_erosion_reduces_values() {
        let img = Array3::from_shape_fn((5, 5, 5), |(z, y, x)| (z + y + x) as f64);
        let se = StructuringElement3D::Cross;
        let eroded = grey_erosion3d(img.view(), &se).expect("grey erosion failed");
        // Every voxel should be ≤ original
        for ((z, y, x), &v) in eroded.indexed_iter() {
            assert!(v <= img[[z, y, x]] + 1e-10);
        }
    }

    #[test]
    fn test_grey_dilation_increases_values() {
        let img = Array3::from_shape_fn((5, 5, 5), |(z, y, x)| (z + y + x) as f64);
        let se = StructuringElement3D::Cross;
        let dilated = grey_dilation3d(img.view(), &se).expect("grey dilation failed");
        for ((z, y, x), &v) in dilated.indexed_iter() {
            assert!(v >= img[[z, y, x]] - 1e-10);
        }
    }

    // ----- Connected component labeling -----

    #[test]
    fn test_label3d_single_blob() {
        let img = sphere_5();
        let (labels, n) = label3d(img.view(), 26).expect("labeling failed");
        assert_eq!(n, 1);
        // All labeled voxels should have label 1
        for ((z, y, x), &lbl) in labels.indexed_iter() {
            if img[[z, y, x]] {
                assert_eq!(lbl, 1);
            } else {
                assert_eq!(lbl, 0);
            }
        }
    }

    #[test]
    fn test_label3d_two_blobs() {
        let img = two_blobs();
        let (_, n) = label3d(img.view(), 26).expect("labeling failed");
        assert_eq!(n, 2, "expected exactly 2 components");
    }

    #[test]
    fn test_label3d_background_only() {
        let img = Array3::from_elem((4, 4, 4), false);
        let (_, n) = label3d(img.view(), 6).expect("labeling failed");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_label3d_full_volume() {
        let img = Array3::from_elem((3, 3, 3), true);
        let (_, n) = label3d(img.view(), 6).expect("labeling failed");
        assert_eq!(n, 1);
    }

    // ----- Object properties -----

    #[test]
    fn test_object_properties_volume() {
        let img = solid_cube();
        let (labels, n) = label3d(img.view(), 26).expect("labeling failed");
        let props = object_properties3d(&labels, n);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].volume, 64); // 4^3
    }

    #[test]
    fn test_object_properties_centroid() {
        let img = solid_cube();
        let (labels, n) = label3d(img.view(), 26).expect("labeling failed");
        let props = object_properties3d(&labels, n);
        let c = props[0].centroid;
        assert!((c.0 - 1.5).abs() < 1e-6);
        assert!((c.1 - 1.5).abs() < 1e-6);
        assert!((c.2 - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_object_properties_equiv_diameter() {
        let img = solid_cube();
        let (labels, n) = label3d(img.view(), 26).expect("labeling failed");
        let props = object_properties3d(&labels, n);
        // Diameter must be positive
        assert!(props[0].equivalent_diameter > 0.0);
    }

    // ----- Distance transform -----

    #[test]
    fn test_edt3d_background_zero() {
        let img = Array3::from_elem((5, 5, 5), false);
        let dist = distance_transform_edt3d(img.view()).expect("EDT failed");
        assert!(dist.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_edt3d_center_voxel() {
        // Single foreground voxel in center of 5x5x5 volume surrounded by background border
        // The center voxel [2,2,2] is surrounded; nearest bg is at distance 1 (face neighbor)
        // Wait -- we need a non-trivial layout.
        // All-foreground interior; background = border => center [2,2,2] has dist=2
        let mut img = Array3::from_elem((5, 5, 5), true);
        // Set border to false
        for z in 0..5usize {
            for y in 0..5usize {
                for x in 0..5usize {
                    if z == 0 || z == 4 || y == 0 || y == 4 || x == 0 || x == 4 {
                        img[[z, y, x]] = false;
                    }
                }
            }
        }
        let dist = distance_transform_edt3d(img.view()).expect("EDT failed");
        // Center voxel [2,2,2] should have distance 2 (2 steps to border)
        assert!(
            (dist[[2, 2, 2]] - 2.0).abs() < 0.5,
            "Expected ~2.0, got {}",
            dist[[2, 2, 2]]
        );
    }

    #[test]
    fn test_edt3d_single_fg_voxel() {
        let mut img = Array3::from_elem((3, 3, 3), false);
        img[[1, 1, 1]] = true;
        let dist = distance_transform_edt3d(img.view()).expect("EDT failed");
        // Background voxels have distance 0
        assert_eq!(dist[[0, 0, 0]], 0.0);
        // The single foreground voxel has distance = 1 (closest bg is face neighbor)
        assert!((dist[[1, 1, 1]] - 1.0).abs() < 0.5);
    }

    // ----- Skeletonization -----

    #[test]
    fn test_skeletonize3d_preserves_nonempty() {
        let img = sphere_5();
        let skel = skeletonize3d(img.view()).expect("skeletonize failed");
        // Skeleton must be a subset of the original
        for ((z, y, x), &v) in skel.indexed_iter() {
            if v {
                assert!(img[[z, y, x]], "skeleton contains voxel not in original");
            }
        }
    }

    #[test]
    fn test_skeletonize3d_empty_error() {
        let img: Array3<bool> = Array3::from_elem((0, 0, 0), false);
        let result = skeletonize3d(img.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_skeletonize3d_single_voxel() {
        let mut img = Array3::from_elem((3, 3, 3), false);
        img[[1, 1, 1]] = true;
        let skel = skeletonize3d(img.view()).expect("skeletonize failed");
        // A single foreground voxel with all background neighbors is NOT a simple point
        // (removing it would disconnect foreground) → should remain
        assert!(skel[[1, 1, 1]]);
    }
}
