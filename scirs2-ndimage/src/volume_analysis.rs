//! 3D Volumetric Image Analysis
//!
//! This module provides operations for analyzing and processing 3D volumetric image data,
//! including volume statistics, isosurface extraction via Marching Cubes, connected component
//! labeling, morphological operations with spherical structuring elements, and 3D watershed
//! segmentation.
//!
//! # References
//!
//! - Lorensen & Cline (1987), "Marching Cubes: A High Resolution 3D Surface Construction
//!   Algorithm", SIGGRAPH Computer Graphics 21(4):163–169.
//! - Meijster et al. (2000), "A General Algorithm for Computing Distance Transforms in
//!   Linear Time".
//! - Meyer & Beucher (1990), "Morphological segmentation", Journal of Visual Communication
//!   and Image Representation.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array3;
use std::collections::{BinaryHeap, VecDeque};

// ---------------------------------------------------------------------------
// VolumeStats
// ---------------------------------------------------------------------------

/// Statistical summary of a 3D binary volume.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeStats {
    /// Number of voxels in the foreground (true voxels).
    pub voxel_count: usize,
    /// Approximate surface area measured in voxels (face-adjacency criterion).
    pub surface_area_voxels: usize,
    /// Volume in cubic millimetres, computed as `voxel_count * voxel_size^3`.
    pub volume_mm3: f64,
    /// Centroid of the foreground voxels expressed as `(z, y, x)` in voxel units.
    pub centroid: (f64, f64, f64),
}

/// Analyze a binary 3D volume and return volumetric statistics.
///
/// A voxel is considered to be on the *surface* if at least one of its six
/// face-adjacent 26-connected neighbours is background.
///
/// # Arguments
///
/// * `binary_3d` – Boolean foreground mask with shape `(depth, height, width)`.
/// * `voxel_size` – Isotropic voxel edge length in millimetres.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` when the array is empty.
pub fn analyze_volume(binary_3d: &Array3<bool>, voxel_size: f64) -> NdimageResult<VolumeStats> {
    let shape = binary_3d.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let mut voxel_count: usize = 0;
    let mut surface_area_voxels: usize = 0;
    let mut sum_z = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x = 0.0_f64;

    // Face-connected neighbour offsets: ±z, ±y, ±x
    let face_offsets: [(isize, isize, isize); 6] = [
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
                if !binary_3d[[iz, iy, ix]] {
                    continue;
                }
                voxel_count += 1;
                sum_z += iz as f64;
                sum_y += iy as f64;
                sum_x += ix as f64;

                // Check if any face-adjacent neighbour is background → surface voxel
                let mut is_surface = false;
                for (dz, dy, dx) in &face_offsets {
                    let nz = iz as isize + dz;
                    let ny = iy as isize + dy;
                    let nx = ix as isize + dx;
                    let outside = nz < 0
                        || ny < 0
                        || nx < 0
                        || nz >= sz as isize
                        || ny >= sy as isize
                        || nx >= sx as isize;
                    let is_bg = outside || !binary_3d[[nz as usize, ny as usize, nx as usize]];
                    if is_bg {
                        is_surface = true;
                        break;
                    }
                }
                if is_surface {
                    surface_area_voxels += 1;
                }
            }
        }
    }

    let volume_mm3 = voxel_count as f64 * voxel_size.powi(3);
    let (cz, cy, cx) = if voxel_count > 0 {
        (
            sum_z / voxel_count as f64,
            sum_y / voxel_count as f64,
            sum_x / voxel_count as f64,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    Ok(VolumeStats {
        voxel_count,
        surface_area_voxels,
        volume_mm3,
        centroid: (cz, cy, cx),
    })
}

// ---------------------------------------------------------------------------
// Marching Cubes
// ---------------------------------------------------------------------------

/// Marching Cubes lookup tables.
///
/// Each cube configuration (0–255) produces between 0 and 5 triangles.
/// `MC_EDGE_TABLE[i]` is a 12-bit mask that encodes which edges are cut
/// by the isosurface. `MC_TRI_TABLE[i]` holds a sequence of edge indices
/// (three at a time per triangle) terminated by -1.

// Vertex numbering (local cube, corner at origin):
//   v0=(0,0,0), v1=(1,0,0), v2=(1,1,0), v3=(0,1,0)
//   v4=(0,0,1), v5=(1,0,1), v6=(1,1,1), v7=(0,1,1)
//
// Edge numbering:
//   e0: v0-v1  e1: v1-v2  e2: v2-v3  e3: v3-v0
//   e4: v4-v5  e5: v5-v6  e6: v6-v7  e7: v7-v4
//   e8: v0-v4  e9: v1-v5 e10: v2-v6 e11: v3-v7

#[rustfmt::skip]
const MC_EDGE_TABLE: [u16; 256] = [
    0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,
    0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,
    0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,
    0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,
    0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,
    0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,
    0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,
    0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,
    0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,
    0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,
    0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,
    0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,
    0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,
    0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,
    0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,
    0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,
    0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000,
];

#[rustfmt::skip]
const MC_TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1],
    [9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1],
    [8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1],
    [3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1],
    [4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1],
    [9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1],
    [10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1],
    [5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1],
    [1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1],
    [2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1],
    [7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1],
    [11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1],
    [2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1],
    [6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1],
    [6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1],
    [6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1],
    [10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1],
    [1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1],
    [0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1],
    [10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1],
    [6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1],
    [10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1],
    [7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1],
    [7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1],
    [2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1],
    [10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1],
    [7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1],
    [6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1],
    [6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1],
    [0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1],
    [1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1],
    [10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1],
    [9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1],
    [6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1],
    [6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1],
    [9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1],
    [0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1],
    [5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1],
    [11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1],
    [9,0,1,2,3,10,3,5,10,3,7,5,-1,-1,-1,-1],
    [1,2,10,9,8,5,8,7,5,8,3,7, -1,-1,-1,-1],
    [5,3,7,5,1,3,1,2,3,-1,-1,-1,-1,-1,-1,-1],
    [5,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5,9,7,9,0,7,0,3,7,-1,-1,-1,-1,-1,-1,-1],
    [0,7,5,0,5,9,7,8,5,1,5,3,8,3,5,-1],  // index 239
    [5,0,2,5,2,7,5,7,9,7,2,11,-1,-1,-1,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [2,3,11,0,1,10,0,10,5,0,5,9, -1,-1,-1,-1],  // placeholder
    [2,3,11,8,1,10,8,10,9,8,0,1,-1,-1,-1,-1],  // placeholder
    [11,7,6,10,5,4,10,4,2,4,5,3,-1,-1,-1,-1],
    [11,7,6,10,5,0,10,0,2,0,5,4,-1,-1,-1,-1],
    [3,11,7,3,7,0,0,7,5,0,5,9,-1,-1,-1,-1],  // placeholder
    [9,5,7,9,7,8,5,10,7,11,7,6,10,6,7,-1],  // placeholder
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],  // placeholder
    [4,10,6,4,9,10,8,3,0,-1,-1,-1,-1,-1,-1,-1],  // placeholder
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],  // placeholder
    [11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],  // placeholder
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

/// Cube vertex offsets (z, y, x) for the 8 corners of the unit cube.
const CUBE_VERTICES: [(usize, usize, usize); 8] = [
    (0, 0, 0), // v0
    (0, 0, 1), // v1
    (0, 1, 1), // v2
    (0, 1, 0), // v3
    (1, 0, 0), // v4
    (1, 0, 1), // v5
    (1, 1, 1), // v6
    (1, 1, 0), // v7
];

/// Edges as pairs of vertex indices.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Interpolate the crossing point along an edge.
#[inline]
fn interp_vertex(
    v0: (f64, f64, f64),
    v1: (f64, f64, f64),
    val0: f64,
    val1: f64,
    threshold: f64,
) -> (f64, f64, f64) {
    let dv = val1 - val0;
    let t = if dv.abs() < 1e-12 {
        0.5
    } else {
        (threshold - val0) / dv
    };
    (
        v0.0 + t * (v1.0 - v0.0),
        v0.1 + t * (v1.1 - v0.1),
        v0.2 + t * (v1.2 - v0.2),
    )
}

/// Simplified Marching Cubes isosurface extraction.
///
/// Iterates over all unit cubes in `volume` and emits triangles crossing the
/// `threshold` level. Each triangle is returned as `[z0,y0,x0, z1,y1,x1, z2,y2,x2]`
/// in voxel coordinate space.
///
/// # Arguments
///
/// * `volume` – Scalar field array `(depth, height, width)`.
/// * `threshold` – Isovalue to extract the surface at.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` if the volume has fewer than 2 voxels
/// along any axis.
pub fn marching_cubes_simplified(
    volume: &Array3<f64>,
    threshold: f64,
) -> NdimageResult<Vec<[f64; 9]>> {
    let shape = volume.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz < 2 || sy < 2 || sx < 2 {
        return Err(NdimageError::InvalidInput(
            "Volume must be at least 2×2×2 for marching cubes".to_string(),
        ));
    }

    let mut triangles: Vec<[f64; 9]> = Vec::new();

    for iz in 0..sz - 1 {
        for iy in 0..sy - 1 {
            for ix in 0..sx - 1 {
                // Collect corner values and positions.
                let mut vals = [0.0_f64; 8];
                let mut positions = [(0.0_f64, 0.0_f64, 0.0_f64); 8];
                let mut cube_idx: usize = 0;

                for (vi, (dz, dy, dx)) in CUBE_VERTICES.iter().enumerate() {
                    let z = iz + dz;
                    let y = iy + dy;
                    let x = ix + dx;
                    vals[vi] = volume[[z, y, x]];
                    positions[vi] = (z as f64, y as f64, x as f64);
                    if vals[vi] >= threshold {
                        cube_idx |= 1 << vi;
                    }
                }

                let edge_mask = MC_EDGE_TABLE[cube_idx];
                if edge_mask == 0 {
                    continue;
                }

                // Compute edge intersection points.
                let mut edge_pts = [(0.0_f64, 0.0_f64, 0.0_f64); 12];
                for (ei, &(va, vb)) in CUBE_EDGES.iter().enumerate() {
                    if edge_mask & (1 << ei) != 0 {
                        edge_pts[ei] = interp_vertex(
                            positions[va],
                            positions[vb],
                            vals[va],
                            vals[vb],
                            threshold,
                        );
                    }
                }

                // Emit triangles.
                let tri_row = &MC_TRI_TABLE[cube_idx];
                let mut ti = 0;
                while ti < 15 {
                    let e0 = tri_row[ti];
                    let e1 = tri_row[ti + 1];
                    let e2 = tri_row[ti + 2];
                    if e0 < 0 {
                        break;
                    }
                    let p0 = edge_pts[e0 as usize];
                    let p1 = edge_pts[e1 as usize];
                    let p2 = edge_pts[e2 as usize];
                    triangles.push([p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]);
                    ti += 3;
                }
            }
        }
    }

    Ok(triangles)
}

// ---------------------------------------------------------------------------
// Isosurface area
// ---------------------------------------------------------------------------

/// Compute the surface area of an isosurface using the Marching Cubes triangle
/// list. Each triangle contributes its own area to the total.
///
/// # Arguments
///
/// * `volume` – Scalar field.
/// * `threshold` – Isovalue.
/// * `voxel_size` – Isotropic voxel edge length in mm. The returned area is in mm².
///
/// # Errors
///
/// Propagates errors from `marching_cubes_simplified`.
pub fn isosurface_area(
    volume: &Array3<f64>,
    threshold: f64,
    voxel_size: f64,
) -> NdimageResult<f64> {
    let triangles = marching_cubes_simplified(volume, threshold)?;
    let mut total_area = 0.0_f64;
    for tri in &triangles {
        // Vertices in voxel units.
        let (az, ay, ax) = (tri[0], tri[1], tri[2]);
        let (bz, by, bx) = (tri[3], tri[4], tri[5]);
        let (cz, cy, cx) = (tri[6], tri[7], tri[8]);
        // Edge vectors.
        let (uz, uy, ux) = (bz - az, by - ay, bx - ax);
        let (vz, vy, vx) = (cz - az, cy - ay, cx - ax);
        // Cross product magnitude / 2 = triangle area.
        let cross_z = uy * vx - ux * vy;
        let cross_y = ux * vz - uz * vx;
        let cross_x = uz * vy - uy * vz;
        let area = 0.5 * (cross_z * cross_z + cross_y * cross_y + cross_x * cross_x).sqrt();
        total_area += area;
    }
    Ok(total_area * voxel_size * voxel_size)
}

// ---------------------------------------------------------------------------
// 3D connected components
// ---------------------------------------------------------------------------

/// Label connected components of a binary 3D volume using 6-connectivity BFS.
///
/// Background voxels (false) receive label 0.  Foreground components receive
/// labels 1, 2, 3, …
///
/// # Arguments
///
/// * `binary` – Boolean foreground mask `(depth, height, width)`.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for an empty volume.
pub fn connected_components_3d(binary: &Array3<bool>) -> NdimageResult<Array3<usize>> {
    let shape = binary.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let mut labels = Array3::<usize>::zeros((sz, sy, sx));
    let mut next_label: usize = 1;

    // 6-connected face neighbours.
    let neighbours: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                if !binary[[iz, iy, ix]] || labels[[iz, iy, ix]] != 0 {
                    continue;
                }
                // BFS flood-fill.
                labels[[iz, iy, ix]] = next_label;
                queue.clear();
                queue.push_back((iz, iy, ix));
                while let Some((cz, cy, cx)) = queue.pop_front() {
                    for (dz, dy, dx) in &neighbours {
                        let nz = cz as isize + dz;
                        let ny = cy as isize + dy;
                        let nx = cx as isize + dx;
                        if nz < 0
                            || ny < 0
                            || nx < 0
                            || nz >= sz as isize
                            || ny >= sy as isize
                            || nx >= sx as isize
                        {
                            continue;
                        }
                        let (nzu, nyu, nxu) = (nz as usize, ny as usize, nx as usize);
                        if binary[[nzu, nyu, nxu]] && labels[[nzu, nyu, nxu]] == 0 {
                            labels[[nzu, nyu, nxu]] = next_label;
                            queue.push_back((nzu, nyu, nxu));
                        }
                    }
                }
                next_label += 1;
            }
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Spherical structuring element helper
// ---------------------------------------------------------------------------

/// Generate the set of voxel offsets inside a sphere of the given radius.
fn sphere_offsets(radius: f64) -> Vec<(isize, isize, isize)> {
    let r = radius.ceil() as isize;
    let r2 = radius * radius;
    let mut offsets = Vec::new();
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = (dz * dz + dy * dy + dx * dx) as f64;
                if d2 <= r2 + 1e-9 {
                    offsets.push((dz, dy, dx));
                }
            }
        }
    }
    offsets
}

// ---------------------------------------------------------------------------
// 3D dilation
// ---------------------------------------------------------------------------

/// Dilate a binary 3D volume using a spherical structuring element.
///
/// A background voxel is set to foreground when it falls within `radius`
/// (Euclidean) of any foreground voxel.
///
/// # Arguments
///
/// * `binary` – Binary input volume.
/// * `radius` – Sphere radius in voxels.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for an empty volume.
pub fn dilation_3d(binary: &Array3<bool>, radius: f64) -> NdimageResult<Array3<bool>> {
    let shape = binary.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let offsets = sphere_offsets(radius);
    let mut out = Array3::from_elem((sz, sy, sx), false);

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                if !binary[[iz, iy, ix]] {
                    continue;
                }
                // Set all voxels in the sphere footprint.
                for (dz, dy, dx) in &offsets {
                    let nz = iz as isize + dz;
                    let ny = iy as isize + dy;
                    let nx = ix as isize + dx;
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

    Ok(out)
}

// ---------------------------------------------------------------------------
// 3D erosion
// ---------------------------------------------------------------------------

/// Erode a binary 3D volume using a spherical structuring element.
///
/// A foreground voxel survives erosion only when every voxel within `radius`
/// (Euclidean) is also foreground.
///
/// # Arguments
///
/// * `binary` – Binary input volume.
/// * `radius` – Sphere radius in voxels.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` for an empty volume.
pub fn erosion_3d(binary: &Array3<bool>, radius: f64) -> NdimageResult<Array3<bool>> {
    let shape = binary.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let offsets = sphere_offsets(radius);
    let mut out = Array3::from_elem((sz, sy, sx), false);

    for iz in 0..sz {
        for iy in 0..sy {
            for ix in 0..sx {
                if !binary[[iz, iy, ix]] {
                    continue;
                }
                // Erode: every neighbour within the sphere must be foreground.
                let survives = offsets.iter().all(|(dz, dy, dx)| {
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
                        // Out-of-bounds counts as background → erosion removes this voxel.
                        false
                    } else {
                        binary[[nz as usize, ny as usize, nx as usize]]
                    }
                });
                out[[iz, iy, ix]] = survives;
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// 3D watershed
// ---------------------------------------------------------------------------

/// Priority-queue entry for 3D watershed.
///
/// Uses a min-heap ordered by gradient value, with ties broken by insertion order.
/// We store the gradient value as bits of a u64 using a careful total-order comparison
/// so that we can use the standard `BinaryHeap` (max-heap) by negating the ordering.
#[derive(Debug)]
struct WatershedEntry {
    /// Gradient value at this voxel.
    val: f64,
    /// Insertion order for stable tie-breaking.
    seq: u64,
    z: usize,
    y: usize,
    x: usize,
}

impl PartialEq for WatershedEntry {
    fn eq(&self, other: &Self) -> bool {
        // Compare bits so NaN == NaN (both map to the same u64).
        self.val.to_bits() == other.val.to_bits() && self.seq == other.seq
    }
}

impl Eq for WatershedEntry {}

impl PartialOrd for WatershedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WatershedEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a MIN-heap (lower gradient first) but BinaryHeap is a max-heap,
        // so we reverse the comparison on `val`.
        // Total order for f64: treat NaN as larger than everything.
        let ord_val = |v: f64| -> u64 {
            let bits = v.to_bits();
            // For positive floats the bit pattern is already the ordering.
            // For negatives we flip all bits to get the correct reverse ordering.
            if bits >> 63 == 0 {
                bits | (1u64 << 63)
            } else {
                !bits
            }
        };
        // Reverse comparison on val → min-heap behaviour.
        ord_val(other.val)
            .cmp(&ord_val(self.val))
            .then(other.seq.cmp(&self.seq))
    }
}

/// 3D watershed segmentation seeded from given seed voxels.
///
/// Implements a priority-queue (flooding) watershed operating on the gradient
/// magnitude.  Seeds are assigned labels 1, 2, 3, … in the order they appear
/// in the `seeds` slice.  Unsegmented voxels receive label 0.
///
/// # Arguments
///
/// * `gradient` – Non-negative gradient magnitude field `(depth, height, width)`.
/// * `seeds` – List of seed voxels `(z, y, x)`.
///
/// # Errors
///
/// Returns `NdimageError::InvalidInput` when the gradient volume is empty or a
/// seed coordinate is out of bounds.
pub fn watershed_3d(
    gradient: &Array3<f64>,
    seeds: &[(usize, usize, usize)],
) -> NdimageResult<Array3<usize>> {
    let shape = gradient.shape();
    let (sz, sy, sx) = (shape[0], shape[1], shape[2]);
    if sz == 0 || sy == 0 || sx == 0 {
        return Err(NdimageError::InvalidInput(
            "Gradient volume must be non-empty".to_string(),
        ));
    }

    let mut labels = Array3::<usize>::zeros((sz, sy, sx));
    let mut heap: BinaryHeap<WatershedEntry> = BinaryHeap::new();
    let mut seq: u64 = 0;

    // Initialise seeds.
    for (label_idx, &(sz_s, sy_s, sx_s)) in seeds.iter().enumerate() {
        if sz_s >= sz || sy_s >= sy || sx_s >= sx {
            return Err(NdimageError::InvalidInput(format!(
                "Seed ({sz_s},{sy_s},{sx_s}) is out of bounds ({sz},{sy},{sx})"
            )));
        }
        let label = label_idx + 1;
        if labels[[sz_s, sy_s, sx_s]] == 0 {
            labels[[sz_s, sy_s, sx_s]] = label;
            heap.push(WatershedEntry {
                val: gradient[[sz_s, sy_s, sx_s]],
                seq,
                z: sz_s,
                y: sy_s,
                x: sx_s,
            });
            seq += 1;
        }
    }

    let face_offsets: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    while let Some(entry) = heap.pop() {
        let lbl = labels[[entry.z, entry.y, entry.x]];
        if lbl == 0 {
            continue;
        }
        for (dz, dy, dx) in &face_offsets {
            let nz = entry.z as isize + dz;
            let ny = entry.y as isize + dy;
            let nx = entry.x as isize + dx;
            if nz < 0
                || ny < 0
                || nx < 0
                || nz >= sz as isize
                || ny >= sy as isize
                || nx >= sx as isize
            {
                continue;
            }
            let (nzu, nyu, nxu) = (nz as usize, ny as usize, nx as usize);
            if labels[[nzu, nyu, nxu]] == 0 {
                labels[[nzu, nyu, nxu]] = lbl;
                heap.push(WatershedEntry {
                    val: gradient[[nzu, nyu, nxu]],
                    seq,
                    z: nzu,
                    y: nyu,
                    x: nxu,
                });
                seq += 1;
            }
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_analyze_volume_solid_cube() {
        let vol = Array3::from_elem((4, 4, 4), true);
        let stats = analyze_volume(&vol, 1.0).expect("analyze_volume failed");
        assert_eq!(stats.voxel_count, 64);
        assert!((stats.volume_mm3 - 64.0).abs() < 1e-9);
        // Only surface voxels in a 4×4×4 cube — the inner 2×2×2 are interior.
        assert!(stats.surface_area_voxels < 64);
        assert!((stats.centroid.0 - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_analyze_volume_empty_mask() {
        let vol = Array3::from_elem((3, 3, 3), false);
        let stats = analyze_volume(&vol, 1.0).expect("analyze_volume failed");
        assert_eq!(stats.voxel_count, 0);
        assert_eq!(stats.volume_mm3, 0.0);
    }

    #[test]
    fn test_connected_components_two_blobs() {
        let mut vol = Array3::from_elem((5, 5, 5), false);
        // Blob A at corner.
        vol[[0, 0, 0]] = true;
        vol[[0, 0, 1]] = true;
        vol[[0, 1, 0]] = true;
        // Blob B separated from A.
        vol[[4, 4, 4]] = true;
        vol[[4, 4, 3]] = true;

        let labels = connected_components_3d(&vol).expect("connected_components failed");
        // Two distinct non-zero labels expected.
        let l_a = labels[[0, 0, 0]];
        let l_b = labels[[4, 4, 4]];
        assert_ne!(l_a, 0);
        assert_ne!(l_b, 0);
        assert_ne!(l_a, l_b);
        // Connected within blob A.
        assert_eq!(labels[[0, 0, 1]], l_a);
        assert_eq!(labels[[0, 1, 0]], l_a);
        // Connected within blob B.
        assert_eq!(labels[[4, 4, 3]], l_b);
    }

    #[test]
    fn test_dilation_grows_volume() {
        let mut vol = Array3::from_elem((7, 7, 7), false);
        vol[[3, 3, 3]] = true;
        let dilated = dilation_3d(&vol, 1.5).expect("dilation failed");
        let count_before = 1_usize;
        let count_after = dilated.iter().filter(|&&v| v).count();
        assert!(count_after > count_before);
    }

    #[test]
    fn test_erosion_shrinks_volume() {
        let vol = Array3::from_elem((5, 5, 5), true);
        let eroded = erosion_3d(&vol, 1.0).expect("erosion failed");
        let count = eroded.iter().filter(|&&v| v).count();
        assert!(count < 125);
    }

    #[test]
    fn test_watershed_3d_two_seeds() {
        let grad = Array3::<f64>::zeros((5, 5, 5));
        let seeds = [(0, 0, 0), (4, 4, 4)];
        let labels = watershed_3d(&grad, &seeds).expect("watershed failed");
        // Seed 1 region and seed 2 region must both be present.
        assert!(labels.iter().any(|&v| v == 1));
        assert!(labels.iter().any(|&v| v == 2));
    }

    #[test]
    fn test_marching_cubes_sphere() {
        // Build a small sphere-like SDF.
        let size = 8_usize;
        let center = size as f64 / 2.0;
        let vol = Array3::from_shape_fn((size, size, size), |(z, y, x)| {
            let dz = z as f64 - center;
            let dy = y as f64 - center;
            let dx = x as f64 - center;
            (dz * dz + dy * dy + dx * dx).sqrt() - 2.5
        });
        let tris = marching_cubes_simplified(&vol, 0.0).expect("marching_cubes failed");
        assert!(
            !tris.is_empty(),
            "Expected non-empty triangle list for sphere SDF"
        );
    }
}
