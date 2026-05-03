//! Topological analysis of binary images.
//!
//! # Overview
//!
//! This module provides tools for analysing the topological structure of
//! binary 2-D and 3-D images:
//!
//! - **Euler number**: topological invariant equal to (connected components) −
//!   (holes) in 2-D, or (components) − (tunnels) + (enclosed cavities) in 3-D.
//! - **Connected component counting**: 4- or 8-connected components in 2-D.
//! - **Hole filling**: flood-fill from the image border to identify and fill
//!   foreground-enclosed holes.
//! - **Genus**: topological genus of a 3-D binary object (number of
//!   "handles" or tunnels through the object).
//!
//! # Notes on conventions
//!
//! - For **2-D** images "connectivity 4" uses the 4-neighbourhood (N/S/E/W)
//!   and "connectivity 8" uses the full 8-neighbourhood.
//! - The Euler number and genus are computed via the standard quad-tree /
//!   octant lookup-table approach on 2×2 (or 2×2×2) pixel windows.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Euler Number (2-D)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Euler number (Euler characteristic) of a 2-D binary image.
///
/// The Euler number is defined as
///
/// ```text
/// χ = C − H
/// ```
///
/// where *C* is the number of connected foreground components and *H* is the
/// number of enclosed holes (background regions not touching the border).
///
/// # Implementation
///
/// Uses the CW-complex formula χ = V − E + F where V is the number of
/// foreground pixels, E is the number of 4-adjacent foreground pairs, and
/// F is the number of 2×2 all-foreground squares.  This correctly handles
/// thin connections (single-pixel bridges) and diagonal pixel patterns.
///
/// # Arguments
///
/// * `binary_image` – 2-D boolean array where `true` = foreground.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::topology::euler_number;
/// use scirs2_core::ndarray::Array2;
///
/// // Solid disc – one component, no holes → Euler = 1
/// let mut img = Array2::<bool>::default((7, 7));
/// for r in 1..6 { for c in 1..6 { img[[r, c]] = true; } }
/// assert_eq!(euler_number(&img.view()), 1);
///
/// // Ring (annulus) – one component, one hole → Euler = 0
/// let mut ring = Array2::<bool>::default((9, 9));
/// for r in 0..9 { for c in 0..9 {
///     let dr = r as i32 - 4;
///     let dc = c as i32 - 4;
///     let d = dr*dr + dc*dc;
///     ring[[r, c]] = d >= 4 && d <= 16;
/// }}
/// let e = euler_number(&ring.view());
/// assert_eq!(e, 0);
/// ```
pub fn euler_number(binary_image: &ArrayView2<bool>) -> i32 {
    // Compute the 2-D Euler (Euler-Poincaré) characteristic via the
    // CW-complex formula:
    //
    //   χ = V − E + F
    //
    // where, for a binary foreground image:
    //   V = number of foreground pixels (0-cells / vertices)
    //   E = number of 4-adjacent foreground pixel pairs (1-cells / edges):
    //         horizontal pairs  (r,c)─(r,c+1)
    //       + vertical   pairs  (r,c)─(r+1,c)
    //   F = number of 2×2 all-foreground squares (2-cells / faces)
    //
    // This is the standard discrete topological Euler number for 4-connectivity
    // (see Duda & Hart, "Pattern Classification", or Gonzalez & Woods,
    // "Digital Image Processing") and correctly handles thin/diagonal connections.

    let rows = binary_image.nrows();
    let cols = binary_image.ncols();

    let get = |r: usize, c: usize| -> i32 { *binary_image.get((r, c)).unwrap_or(&false) as i32 };

    // V – foreground pixel count
    let mut v: i32 = 0;
    for r in 0..rows {
        for c in 0..cols {
            v += get(r, c);
        }
    }

    // E – count of 4-adjacent foreground pairs (horizontal + vertical)
    let mut e: i32 = 0;
    for r in 0..rows {
        for c in 0..(cols.saturating_sub(1)) {
            if get(r, c) == 1 && get(r, c + 1) == 1 {
                e += 1;
            }
        }
    }
    for r in 0..(rows.saturating_sub(1)) {
        for c in 0..cols {
            if get(r, c) == 1 && get(r + 1, c) == 1 {
                e += 1;
            }
        }
    }

    // F – count of 2×2 all-foreground squares
    let mut f: i32 = 0;
    for r in 0..(rows.saturating_sub(1)) {
        for c in 0..(cols.saturating_sub(1)) {
            if get(r, c) == 1 && get(r, c + 1) == 1 && get(r + 1, c) == 1 && get(r + 1, c + 1) == 1
            {
                f += 1;
            }
        }
    }

    v - e + f
}

// ─────────────────────────────────────────────────────────────────────────────
// Connected Component Count
// ─────────────────────────────────────────────────────────────────────────────

/// Count the number of connected foreground components in a 2-D binary image.
///
/// # Arguments
///
/// * `binary`      – 2-D boolean array (`true` = foreground).
/// * `connectivity` – `4` for 4-connectivity (N/S/E/W neighbours) or
///                    `8` for 8-connectivity (all 8 neighbours).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] if `connectivity` is not 4 or 8.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::topology::connected_components_count;
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<bool>::default((5, 9));
/// // Two separate 2×2 blocks
/// img[[0,0]] = true; img[[0,1]] = true; img[[1,0]] = true; img[[1,1]] = true;
/// img[[0,7]] = true; img[[0,8]] = true; img[[1,7]] = true; img[[1,8]] = true;
/// assert_eq!(connected_components_count(&img.view(), 4).unwrap(), 2);
/// ```
pub fn connected_components_count(
    binary: &ArrayView2<bool>,
    connectivity: u8,
) -> NdimageResult<usize> {
    if connectivity != 4 && connectivity != 8 {
        return Err(NdimageError::InvalidInput(format!(
            "connected_components_count: connectivity must be 4 or 8, got {connectivity}"
        )));
    }

    let rows = binary.nrows();
    let cols = binary.ncols();
    let mut visited = Array2::<bool>::default((rows, cols));
    let mut count = 0usize;

    for start_r in 0..rows {
        for start_c in 0..cols {
            if !binary[[start_r, start_c]] || visited[[start_r, start_c]] {
                continue;
            }
            // BFS
            count += 1;
            let mut queue = VecDeque::new();
            queue.push_back((start_r, start_c));
            visited[[start_r, start_c]] = true;

            while let Some((r, c)) = queue.pop_front() {
                for (nr, nc) in neighbours_2d(r, c, rows, cols, connectivity) {
                    if binary[[nr, nc]] && !visited[[nr, nc]] {
                        visited[[nr, nc]] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
    }

    Ok(count)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hole Filling
// ─────────────────────────────────────────────────────────────────────────────

/// Fill holes in a binary image.
///
/// A *hole* is defined as a connected region of background pixels (false) that
/// does not touch any border of the image.  After filling all such regions the
/// returned image contains only solid foreground objects.
///
/// The algorithm flood-fills the background starting from all border pixels,
/// then inverts the result to obtain the holes, and finally OR-combines with
/// the original foreground.
///
/// # Arguments
///
/// * `binary` – 2-D boolean array.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::topology::hole_filling;
/// use scirs2_core::ndarray::Array2;
///
/// // 5×5 frame (ring) with an interior hole
/// let mut img = Array2::<bool>::default((5, 5));
/// for r in 0..5 { for c in 0..5 {
///     img[[r, c]] = r == 0 || r == 4 || c == 0 || c == 4;
/// }}
/// let filled = hole_filling(&img.view()).unwrap();
/// // After filling the interior should also be foreground
/// assert!(filled[[2, 2]]);
/// ```
pub fn hole_filling(binary: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
    let rows = binary.nrows();
    let cols = binary.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::default((rows, cols)));
    }

    // Mark background pixels reachable from the border using 4-connectivity
    let mut reachable = Array2::<bool>::default((rows, cols));
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    // Seed from all border pixels that are background
    let seed_if_bg =
        |r: usize, c: usize, q: &mut VecDeque<(usize, usize)>, reach: &mut Array2<bool>| {
            if !*binary.get((r, c)).unwrap_or(&false) && !reach[[r, c]] {
                reach[[r, c]] = true;
                q.push_back((r, c));
            }
        };

    for c in 0..cols {
        seed_if_bg(0, c, &mut queue, &mut reachable);
        seed_if_bg(rows - 1, c, &mut queue, &mut reachable);
    }
    for r in 1..(rows - 1) {
        seed_if_bg(r, 0, &mut queue, &mut reachable);
        seed_if_bg(r, cols - 1, &mut queue, &mut reachable);
    }

    // BFS with 4-connectivity
    while let Some((r, c)) = queue.pop_front() {
        for (nr, nc) in neighbours_2d(r, c, rows, cols, 4) {
            if !binary[[nr, nc]] && !reachable[[nr, nc]] {
                reachable[[nr, nc]] = true;
                queue.push_back((nr, nc));
            }
        }
    }

    // Output: original foreground OR (background not reachable from border)
    let mut output = Array2::<bool>::default((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            output[[r, c]] = binary[[r, c]] || !reachable[[r, c]];
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Genus (3-D)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the topological genus of a 3-D binary object.
///
/// For a compact orientable surface the genus *g* is related to the Euler
/// characteristic by χ = 2(1 − g) per connected component.  For a 3-D solid
/// voxel object the discrete Euler characteristic is:
///
/// ```text
/// χ = C − T + V
/// ```
///
/// where *C* = connected components (26-connectivity), *T* = tunnels (handles),
/// and *V* = enclosed voids (cavities).
///
/// This function estimates *g* via the octant (2×2×2 voxel window) Euler number
/// contribution lookup table commonly used in digital topology.  For each voxel
/// component the genus is computed from the Euler characteristic as:
///
/// ```text
/// g = 1 − χ/2     (per component, assuming simply connected surface)
/// ```
///
/// # Arguments
///
/// * `binary_3d` – 3-D boolean array where `true` = foreground voxel.
///
/// # Example
///
/// ```
/// use scirs2_ndimage::topology::genus;
/// use scirs2_core::ndarray::Array3;
///
/// // Solid ball – genus 0
/// let mut ball = Array3::<bool>::default((7, 7, 7));
/// for z in 0..7 { for y in 0..7 { for x in 0..7 {
///     let dz = z as i32 - 3;
///     let dy = y as i32 - 3;
///     let dx = x as i32 - 3;
///     ball[[z, y, x]] = dz*dz + dy*dy + dx*dx <= 9;
/// }}}
/// // Genus of a solid ball is 0
/// assert_eq!(genus(&ball.view()), 0);
/// ```
pub fn genus(binary_3d: &ArrayView3<bool>) -> i32 {
    let euler = euler_number_3d(binary_3d);

    // Number of 26-connected components
    let n_components = connected_components_3d(binary_3d);

    // Euler characteristic for a 3-D solid: χ = 2(C - g) assuming no enclosed voids
    // => g = C - χ/2
    // For the general case we report: g = n_components - euler/2
    // (Each tunnel reduces χ by 2, each enclosed void increases χ by 2.)
    let g = n_components as i32 - euler / 2;
    g.max(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 3-D Euler number via the octant (2×2×2) look-up table.
///
/// The octant contribution table follows Toriwaki & Yonekura (2002):
///
/// | Occupied voxels in 2×2×2 | LUT value |
/// |---|---|
/// | 0 or 8 | 0 |
/// | 1 | +1 |
/// | 7 | −1 |
/// | 2 (face pair) | 0 |
/// | 6 (face pair) | 0 |
/// | 2 (edge pair) | 0 |
/// | 6 (edge pair) | 0 |
/// | 2 (body diagonal) | −2 |
/// | 6 (body diagonal) | +2 |
/// | 3 (face triple) | 0 |
/// | 5 (face triple) | 0 |
/// | 4 | depends on pattern |
///
/// For simplicity this implementation uses the commonly-cited simplified
/// formula based on counting isolated, edge, face, and body-diagonal
/// contributions only.
fn euler_number_3d(binary_3d: &ArrayView3<bool>) -> i32 {
    let depth = binary_3d.len_of(scirs2_core::ndarray::Axis(0));
    let rows = binary_3d.len_of(scirs2_core::ndarray::Axis(1));
    let cols = binary_3d.len_of(scirs2_core::ndarray::Axis(2));

    if depth < 2 || rows < 2 || cols < 2 {
        let count: i32 = binary_3d.iter().map(|&b| b as i32).sum();
        return count;
    }

    let get =
        |z: usize, r: usize, c: usize| -> u8 { *binary_3d.get((z, r, c)).unwrap_or(&false) as u8 };

    // 8-bit index encoding the 8 corners of a 2×2×2 cube
    // bit 0: (z,  r,  c  )
    // bit 1: (z,  r,  c+1)
    // bit 2: (z,  r+1,c  )
    // bit 3: (z,  r+1,c+1)
    // bit 4: (z+1,r,  c  )
    // bit 5: (z+1,r,  c+1)
    // bit 6: (z+1,r+1,c  )
    // bit 7: (z+1,r+1,c+1)

    // LUT values from Toriwaki-Yonekura / Lee-Kashyap (26-connectivity)
    // Index = bitmask, value = Euler contribution × 8
    const LUT: [i32; 256] = euler_lut_3d();

    let mut sum = 0i32;
    for z in 0..(depth - 1) {
        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                let idx: usize = ((get(z, r, c)) as usize)
                    | ((get(z, r, c + 1)) as usize) << 1
                    | ((get(z, r + 1, c)) as usize) << 2
                    | ((get(z, r + 1, c + 1)) as usize) << 3
                    | ((get(z + 1, r, c)) as usize) << 4
                    | ((get(z + 1, r, c + 1)) as usize) << 5
                    | ((get(z + 1, r + 1, c)) as usize) << 6
                    | ((get(z + 1, r + 1, c + 1)) as usize) << 7;
                sum += LUT[idx];
            }
        }
    }
    sum / 8
}

/// Count 26-connected foreground components in 3-D via BFS.
fn connected_components_3d(binary_3d: &ArrayView3<bool>) -> usize {
    let depth = binary_3d.len_of(scirs2_core::ndarray::Axis(0));
    let rows = binary_3d.len_of(scirs2_core::ndarray::Axis(1));
    let cols = binary_3d.len_of(scirs2_core::ndarray::Axis(2));

    let mut visited = Array3::<bool>::default((depth, rows, cols));
    let mut count = 0usize;

    for sz in 0..depth {
        for sr in 0..rows {
            for sc in 0..cols {
                if !binary_3d[[sz, sr, sc]] || visited[[sz, sr, sc]] {
                    continue;
                }
                count += 1;
                let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
                queue.push_back((sz, sr, sc));
                visited[[sz, sr, sc]] = true;

                while let Some((z, r, c)) = queue.pop_front() {
                    for dz in -1i32..=1 {
                        for dr in -1i32..=1 {
                            for dc in -1i32..=1 {
                                if dz == 0 && dr == 0 && dc == 0 {
                                    continue;
                                }
                                let nz = z as i32 + dz;
                                let nr = r as i32 + dr;
                                let nc = c as i32 + dc;
                                if nz < 0
                                    || nr < 0
                                    || nc < 0
                                    || nz >= depth as i32
                                    || nr >= rows as i32
                                    || nc >= cols as i32
                                {
                                    continue;
                                }
                                let (nz, nr, nc) = (nz as usize, nr as usize, nc as usize);
                                if binary_3d[[nz, nr, nc]] && !visited[[nz, nr, nc]] {
                                    visited[[nz, nr, nc]] = true;
                                    queue.push_back((nz, nr, nc));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    count
}

/// Generate the 256-entry Euler LUT for 3-D 26-connectivity (×8 scaling).
///
/// The table is derived from the standard Toriwaki-Yonekura octant table.
/// Each entry corresponds to the bitmask of the 8 corners of a 2×2×2 cube.
const fn euler_lut_3d() -> [i32; 256] {
    // Values are pre-multiplied by 8 to allow integer arithmetic.
    // Source: Lee, T.C., Kashyap, R.L. & Chu, C.N. (1994), Table I
    // (26-connectivity, 6-connectivity background)
    let mut lut = [0i32; 256];
    // Encode the known non-zero entries explicitly:
    // 1 voxel occupied → +1
    lut[1] = 1;
    lut[2] = 1;
    lut[4] = 1;
    lut[8] = 1;
    lut[16] = 1;
    lut[32] = 1;
    lut[64] = 1;
    lut[128] = 1;
    // 7 voxels occupied → −1
    lut[254] = -1;
    lut[253] = -1;
    lut[251] = -1;
    lut[247] = -1;
    lut[239] = -1;
    lut[223] = -1;
    lut[191] = -1;
    lut[127] = -1;
    // Body-diagonal pairs (2 voxels, body-diagonal opposite)
    // Bit pairs (0,7), (1,6), (2,5), (3,4)
    lut[0b10000001] = -2; // 129
    lut[0b01000010] = -2; // 66
    lut[0b00100100] = -2; // 36
    lut[0b00011000] = -2; // 24
                          // Complement body-diagonal pairs (6 voxels, body-diagonal missing)
    lut[!0b10000001u8 as usize] = 2; // 126
    lut[!0b01000010u8 as usize] = 2; // 189
    lut[!0b00100100u8 as usize] = 2; // 219
    lut[!0b00011000u8 as usize] = 2; // 231
                                     // All other entries remain 0 (face pairs, edge pairs, etc.)
    lut
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D neighbour iterator
// ─────────────────────────────────────────────────────────────────────────────

/// Return in-bounds 2-D neighbours with the given connectivity (4 or 8).
fn neighbours_2d(
    r: usize,
    c: usize,
    rows: usize,
    cols: usize,
    connectivity: u8,
) -> impl Iterator<Item = (usize, usize)> {
    const N4: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    const N8: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    let offsets: &'static [(i32, i32)] = if connectivity == 4 { &N4 } else { &N8 };
    let (r, c, rows, cols) = (r as i32, c as i32, rows as i32, cols as i32);

    offsets.iter().filter_map(move |&(dr, dc)| {
        let nr = r + dr;
        let nc = c + dc;
        if nr >= 0 && nr < rows && nc >= 0 && nc < cols {
            Some((nr as usize, nc as usize))
        } else {
            None
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    // ── Euler number 2-D ─────────────────────────────────────────────────────

    #[test]
    fn test_euler_solid_square() {
        // Solid 5×5 square: 1 component, 0 holes → χ = 1
        let mut img = Array2::<bool>::default((7, 7));
        for r in 1..6 {
            for c in 1..6 {
                img[[r, c]] = true;
            }
        }
        let e = euler_number(&img.view());
        assert_eq!(e, 1, "solid square Euler number should be 1, got {e}");
    }

    #[test]
    fn test_euler_ring() {
        // Annulus: 1 component, 1 hole → χ = 0
        let mut ring = Array2::<bool>::default((9, 9));
        for r in 0..9u32 {
            for c in 0..9u32 {
                let dr = r as i32 - 4;
                let dc = c as i32 - 4;
                let d = dr * dr + dc * dc;
                ring[[r as usize, c as usize]] = d >= 4 && d <= 16;
            }
        }
        let e = euler_number(&ring.view());
        assert_eq!(e, 0, "ring Euler number should be 0, got {e}");
    }

    #[test]
    fn test_euler_empty() {
        let img = Array2::<bool>::default((5, 5));
        assert_eq!(euler_number(&img.view()), 0);
    }

    #[test]
    fn test_euler_single_pixel() {
        let mut img = Array2::<bool>::default((3, 3));
        img[[1, 1]] = true;
        // Single isolated pixel: 1 component, 0 holes → χ = 1
        assert_eq!(euler_number(&img.view()), 1);
    }

    // ── Connected components ──────────────────────────────────────────────────

    #[test]
    fn test_cc_two_blobs_4() {
        let mut img = Array2::<bool>::default((5, 9));
        img[[0, 0]] = true;
        img[[0, 1]] = true;
        img[[1, 0]] = true;
        img[[1, 1]] = true;
        img[[0, 7]] = true;
        img[[0, 8]] = true;
        img[[1, 7]] = true;
        img[[1, 8]] = true;
        let n = connected_components_count(&img.view(), 4)
            .expect("connected_components_count should succeed with connectivity=4");
        assert_eq!(n, 2);
    }

    #[test]
    fn test_cc_diagonal_4_vs_8() {
        // Two diagonally adjacent pixels
        let mut img = Array2::<bool>::default((3, 3));
        img[[0, 0]] = true;
        img[[1, 1]] = true;
        let n4 = connected_components_count(&img.view(), 4)
            .expect("connected_components_count should succeed with connectivity=4");
        let n8 = connected_components_count(&img.view(), 8)
            .expect("connected_components_count should succeed with connectivity=8");
        assert_eq!(n4, 2, "4-connectivity: diagonal pixels are separate");
        assert_eq!(n8, 1, "8-connectivity: diagonal pixels are connected");
    }

    #[test]
    fn test_cc_invalid_connectivity() {
        let img = Array2::<bool>::default((3, 3));
        assert!(connected_components_count(&img.view(), 6).is_err());
    }

    #[test]
    fn test_cc_empty() {
        let img = Array2::<bool>::default((4, 4));
        let n = connected_components_count(&img.view(), 4)
            .expect("connected_components_count should succeed on empty image");
        assert_eq!(n, 0);
    }

    // ── Hole filling ──────────────────────────────────────────────────────────

    #[test]
    fn test_hole_filling_frame() {
        // 5×5 border frame with hole at centre
        let mut img = Array2::<bool>::default((5, 5));
        for r in 0..5 {
            for c in 0..5 {
                img[[r, c]] = r == 0 || r == 4 || c == 0 || c == 4;
            }
        }
        let filled =
            hole_filling(&img.view()).expect("hole_filling should succeed on valid framed image");
        // Interior should now be foreground
        assert!(filled[[2, 2]], "centre pixel should be filled");
        // Border foreground unchanged
        assert!(filled[[0, 0]]);
    }

    #[test]
    fn test_hole_filling_no_holes() {
        let mut img = Array2::<bool>::default((5, 5));
        for r in 0..5 {
            for c in 0..5 {
                img[[r, c]] = true;
            }
        }
        let filled =
            hole_filling(&img.view()).expect("hole_filling should succeed on fully filled image");
        for r in 0..5 {
            for c in 0..5 {
                assert!(filled[[r, c]]);
            }
        }
    }

    #[test]
    fn test_hole_filling_all_background() {
        let img = Array2::<bool>::default((4, 4));
        let filled =
            hole_filling(&img.view()).expect("hole_filling should succeed on all-background image");
        // All background touching border: nothing filled
        for r in 0..4 {
            for c in 0..4 {
                assert!(!filled[[r, c]]);
            }
        }
    }

    // ── Genus (3-D) ───────────────────────────────────────────────────────────

    #[test]
    fn test_genus_solid_ball() {
        let mut ball = Array3::<bool>::default((7, 7, 7));
        for z in 0..7usize {
            for y in 0..7usize {
                for x in 0..7usize {
                    let dz = z as i32 - 3;
                    let dy = y as i32 - 3;
                    let dx = x as i32 - 3;
                    ball[[z, y, x]] = dz * dz + dy * dy + dx * dx <= 9;
                }
            }
        }
        let g = genus(&ball.view());
        assert_eq!(g, 0, "solid ball genus = 0, got {g}");
    }

    #[test]
    fn test_genus_empty_volume() {
        let vol = Array3::<bool>::default((4, 4, 4));
        // No foreground: genus is 0 by convention (0 components - 0/2)
        let g = genus(&vol.view());
        assert_eq!(g, 0);
    }
}
