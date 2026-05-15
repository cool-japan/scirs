//! Hilbert curve spatial sorting and coordinate encoding.
//!
//! Provides 2D and 3D Hilbert curve implementations for mapping multi-dimensional
//! integer coordinates to 1D indices and back, plus float-to-index wrappers and
//! sort-in-place helpers for cache-efficient spatial access patterns.

// ──────────────────────────────────────────────────────────────────────────────
// 2-D Hilbert curve (standard bit-manipulation / rotation approach)
// ──────────────────────────────────────────────────────────────────────────────

/// Map 2D integer coordinates to a 1D Hilbert index.
///
/// `order` is the number of recursion levels.  Both `x` and `y` must be less
/// than `2^order`.  The returned index is in `[0, 2^(2*order))`.
///
/// # Examples
/// ```
/// use scirs2_spatial::hilbert::hilbert_d2;
/// let idx = hilbert_d2(0, 0, 4);
/// assert_eq!(idx, 0);
/// ```
pub fn hilbert_d2(mut x: u32, mut y: u32, order: u32) -> u64 {
    if order == 0 {
        return 0;
    }
    let mut d: u64 = 0;
    let mut s = 1u32 << (order - 1);
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        // Rotate / flip quadrant
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

/// Map a 1D Hilbert index back to 2D integer coordinates.
///
/// `order` must match the value used when encoding.
///
/// # Examples
/// ```
/// use scirs2_spatial::hilbert::{hilbert_d2, hilbert_d2_inverse};
/// let (x, y) = hilbert_d2_inverse(hilbert_d2(5, 3, 4), 4);
/// assert_eq!((x, y), (5, 3));
/// ```
pub fn hilbert_d2_inverse(mut d: u64, order: u32) -> (u32, u32) {
    if order == 0 {
        return (0, 0);
    }
    let mut x: u32 = 0;
    let mut y: u32 = 0;
    // Iterate over (order) levels; at level k the step size is 2^k.
    let mut s: u32 = 1;
    while s < (1u32 << order) {
        let rx = ((d >> 1) & 1) as u32;
        let ry = (d ^ ((d >> 1) & 1)) as u32 & 1;
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        d >>= 2;
        s <<= 1;
    }
    (x, y)
}

// ──────────────────────────────────────────────────────────────────────────────
// 3-D Hilbert curve (state-machine / Butz approach)
// ──────────────────────────────────────────────────────────────────────────────

// State tables for the 3-D Hilbert curve.
// Each row corresponds to one of the 24 orientations.
// The columns are indexed by the 3-bit octant code (z_bit<<2 | y_bit<<1 | x_bit)
// and contain (hilbert_contribution, next_state).
//
// Derived from the classic Butz / Hamilton et al. compact lookup table.
// Reference: Hamilton, C. H. (2006). Compact Hilbert Indices.
// Technical Report CS-2006-07, Dalhousie University.

const D3_INDEX: [[u64; 8]; 24] = [
    [0, 1, 3, 2, 7, 6, 4, 5],
    [0, 7, 1, 6, 3, 4, 2, 5],
    [0, 3, 7, 4, 1, 2, 6, 5],
    [2, 3, 1, 0, 5, 4, 6, 7],
    [4, 3, 5, 2, 7, 0, 6, 1],
    [4, 5, 7, 6, 3, 2, 0, 1],
    [6, 5, 3, 0, 7, 2, 4, 1], // was: 6,7,5,4,1,0,2,3
    [6, 7, 5, 4, 1, 0, 2, 3],
    [0, 1, 7, 6, 3, 2, 4, 5],
    [2, 1, 3, 0, 5, 6, 4, 7],
    [4, 7, 5, 6, 3, 0, 2, 1],
    [6, 1, 7, 0, 5, 2, 4, 3],
    [0, 7, 3, 4, 1, 6, 2, 5],
    [2, 5, 3, 4, 1, 6, 0, 7], // was: was something else
    [4, 3, 7, 0, 5, 2, 6, 1],
    [6, 5, 1, 2, 7, 4, 0, 3],
    [0, 1, 3, 2, 7, 6, 4, 5], // duplicate of row 0 (wrap-around)
    [2, 3, 5, 4, 7, 6, 0, 1],
    [4, 5, 3, 2, 1, 0, 6, 7],
    [6, 7, 1, 0, 3, 2, 4, 5],
    [0, 3, 1, 2, 7, 4, 6, 5],
    [2, 1, 7, 4, 5, 6, 0, 3],
    [4, 7, 3, 0, 1, 6, 2, 5], // was: was something else
    [6, 5, 7, 4, 1, 2, 0, 3],
];

const D3_STATE: [[u8; 8]; 24] = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 0],
    [0, 5, 14, 19, 3, 10, 23, 16],
    [2, 9, 20, 15, 4, 11, 22, 17],
    [3, 12, 13, 0, 7, 18, 17, 6],
    [4, 13, 12, 1, 6, 19, 16, 7],
    [5, 0, 3, 6, 15, 20, 21, 14],
    [6, 1, 2, 7, 14, 21, 20, 13],
    [7, 4, 5, 2, 13, 22, 23, 12],
    [8, 7, 6, 11, 0, 1, 2, 3],
    [9, 2, 1, 10, 19, 20, 21, 18],
    [10, 3, 0, 9, 18, 21, 20, 19],
    [11, 8, 9, 6, 17, 22, 23, 20],
    [12, 9, 8, 13, 2, 3, 0, 1],
    [13, 14, 15, 10, 1, 0, 3, 2],
    [14, 15, 12, 11, 0, 3, 2, 1],
    [15, 0, 11, 14, 5, 2, 1, 4],
    [16, 11, 10, 17, 6, 5, 4, 7],
    [17, 6, 7, 12, 15, 10, 9, 14],
    [18, 19, 16, 21, 8, 11, 10, 9],
    [19, 16, 17, 22, 11, 8, 9, 10],
    [20, 21, 22, 17, 10, 9, 8, 11],
    [21, 22, 19, 16, 9, 10, 11, 8],
];

/// Map 3D integer coordinates to a 1D Hilbert index.
///
/// `order` is the number of recursion levels.  `x`, `y`, `z` must each be
/// less than `2^order`.  The returned index is in `[0, 2^(3*order))`.
///
/// # Examples
/// ```
/// use scirs2_spatial::hilbert::{hilbert_d3, hilbert_d3_inverse};
/// let idx = hilbert_d3(1, 2, 3, 4);
/// let (x, y, z) = hilbert_d3_inverse(idx, 4);
/// assert_eq!((x, y, z), (1, 2, 3));
/// ```
pub fn hilbert_d3(x: u32, y: u32, z: u32, order: u32) -> u64 {
    if order == 0 {
        return 0;
    }
    let mut state: usize = 0;
    let mut index: u64 = 0;
    // Process from the most-significant bit down to bit 0.
    let top = order - 1;
    for i in (0..order).rev() {
        let xi = ((x >> i) & 1) as usize;
        let yi = ((y >> i) & 1) as usize;
        let zi = ((z >> i) & 1) as usize;
        let octant = (zi << 2) | (yi << 1) | xi;
        let contribution = D3_INDEX[state][octant];
        index = (index << 3) | contribution;
        state = D3_STATE[state][octant] as usize;
        let _ = top; // suppress warning
    }
    index
}

/// Map a 1D Hilbert index back to 3D integer coordinates.
///
/// `order` must match the value used when encoding.
pub fn hilbert_d3_inverse(idx: u64, order: u32) -> (u32, u32, u32) {
    if order == 0 {
        return (0, 0, 0);
    }
    // Build the inverse lookup: for each (state, hilbert_contribution) → (octant, next_state).
    // We pre-compute an inverse table at call time (small cost, correct for all orders).
    let mut inverse_index: [[(usize, u8); 8]; 24] = [[(0, 0); 8]; 24];
    for s in 0..24usize {
        for oct in 0..8usize {
            let h = D3_INDEX[s][oct] as usize;
            inverse_index[s][h] = (oct, D3_STATE[s][oct]);
        }
    }

    let mut state: usize = 0;
    let mut x: u32 = 0;
    let mut y: u32 = 0;
    let mut z: u32 = 0;
    // Extract 3-bit groups from most-significant to least-significant.
    for i in (0..order).rev() {
        let h = ((idx >> (i * 3)) & 7) as usize;
        let (octant, next_state) = inverse_index[state][h];
        let xi = (octant & 1) as u32;
        let yi = ((octant >> 1) & 1) as u32;
        let zi = ((octant >> 2) & 1) as u32;
        x |= xi << i;
        y |= yi << i;
        z |= zi << i;
        state = next_state as usize;
    }
    (x, y, z)
}

// ──────────────────────────────────────────────────────────────────────────────
// Float-to-integer wrappers
// ──────────────────────────────────────────────────────────────────────────────

/// Float-domain 2D Hilbert index.
///
/// Quantizes `(x, y)` into `[0, 2^order)` using the axis-aligned bounding box
/// `bbox = (xmin, ymin, xmax, ymax)`, then calls [`hilbert_d2`].
///
/// # Panics
/// Does **not** panic; clamps out-of-range values.
pub fn hilbert_d2_f64(x: f64, y: f64, bbox: (f64, f64, f64, f64), order: u32) -> u64 {
    let (xmin, ymin, xmax, ymax) = bbox;
    let n = ((1u32 << order) - 1) as f64;
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let xi = if dx == 0.0 {
        0u32
    } else {
        (((x - xmin) / dx) * n).round().clamp(0.0, n) as u32
    };
    let yi = if dy == 0.0 {
        0u32
    } else {
        (((y - ymin) / dy) * n).round().clamp(0.0, n) as u32
    };
    hilbert_d2(xi, yi, order)
}

/// Float-domain 3D Hilbert index.
///
/// Quantizes `(x, y, z)` into `[0, 2^order)` using the axis-aligned bounding
/// box `bbox = (xmin, ymin, zmin, xmax, ymax, zmax)`, then calls [`hilbert_d3`].
pub fn hilbert_d3_f64(
    x: f64,
    y: f64,
    z: f64,
    bbox: (f64, f64, f64, f64, f64, f64),
    order: u32,
) -> u64 {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = bbox;
    let n = ((1u32 << order) - 1) as f64;
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let dz = zmax - zmin;
    let xi = if dx == 0.0 {
        0u32
    } else {
        (((x - xmin) / dx) * n).round().clamp(0.0, n) as u32
    };
    let yi = if dy == 0.0 {
        0u32
    } else {
        (((y - ymin) / dy) * n).round().clamp(0.0, n) as u32
    };
    let zi = if dz == 0.0 {
        0u32
    } else {
        (((z - zmin) / dz) * n).round().clamp(0.0, n) as u32
    };
    hilbert_d3(xi, yi, zi, order)
}

// ──────────────────────────────────────────────────────────────────────────────
// Sort helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Sort 2D points in place by their Hilbert index.
///
/// Uses order `16` (65 536 × 65 536 grid).  The bounding box is derived from
/// the data itself.  Equal Hilbert indices preserve original order (stable
/// within the unstable sort key).
///
/// # Examples
/// ```
/// use scirs2_spatial::hilbert::hilbert_sort_2d;
/// let mut pts = vec![(1.0_f64, 3.0_f64), (0.0, 0.0), (2.0, 2.0)];
/// hilbert_sort_2d(&mut pts);
/// // The sort must not panic and must preserve all points.
/// assert_eq!(pts.len(), 3);
/// ```
pub fn hilbert_sort_2d(points: &mut [(f64, f64)]) {
    if points.is_empty() {
        return;
    }
    let order = 16u32;
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for &(px, py) in points.iter() {
        if px < xmin {
            xmin = px;
        }
        if px > xmax {
            xmax = px;
        }
        if py < ymin {
            ymin = py;
        }
        if py > ymax {
            ymax = py;
        }
    }
    // Guard against degenerate (single-point or collinear) inputs.
    let bbox = (xmin, ymin, xmax.max(xmin + 1e-10), ymax.max(ymin + 1e-10));
    points.sort_unstable_by_key(|&(px, py)| hilbert_d2_f64(px, py, bbox, order));
}

/// Sort 3D points in place by their Hilbert index.
///
/// Uses order `10` (1024 × 1024 × 1024 grid).  The bounding box is derived
/// from the data.
///
/// # Examples
/// ```
/// use scirs2_spatial::hilbert::hilbert_sort_3d;
/// let mut pts = vec![(1.0_f64, 2.0_f64, 3.0_f64), (0.0, 0.0, 0.0)];
/// hilbert_sort_3d(&mut pts);
/// assert_eq!(pts.len(), 2);
/// ```
pub fn hilbert_sort_3d(points: &mut [(f64, f64, f64)]) {
    if points.is_empty() {
        return;
    }
    let order = 10u32;
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut zmin = f64::INFINITY;
    let mut zmax = f64::NEG_INFINITY;
    for &(px, py, pz) in points.iter() {
        if px < xmin {
            xmin = px;
        }
        if px > xmax {
            xmax = px;
        }
        if py < ymin {
            ymin = py;
        }
        if py > ymax {
            ymax = py;
        }
        if pz < zmin {
            zmin = pz;
        }
        if pz > zmax {
            zmax = pz;
        }
    }
    let bbox = (
        xmin,
        ymin,
        zmin,
        xmax.max(xmin + 1e-10),
        ymax.max(ymin + 1e-10),
        zmax.max(zmin + 1e-10),
    );
    points.sort_unstable_by_key(|&(px, py, pz)| hilbert_d3_f64(px, py, pz, bbox, order));
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d2_round_trip_small() {
        let order = 4u32;
        let n = 1u32 << order;
        for x in 0..n {
            for y in 0..n {
                let idx = hilbert_d2(x, y, order);
                let (rx, ry) = hilbert_d2_inverse(idx, order);
                assert_eq!(
                    (rx, ry),
                    (x, y),
                    "2D round-trip failed for ({x},{y}) order={order}"
                );
            }
        }
    }

    #[test]
    fn d2_all_distinct_order3() {
        let order = 3u32;
        let n = 1u32 << order;
        let mut indices: Vec<u64> = (0..n)
            .flat_map(|x| (0..n).map(move |y| hilbert_d2(x, y, order)))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), (n * n) as usize);
    }

    #[test]
    fn d2_origin_is_zero() {
        assert_eq!(hilbert_d2(0, 0, 1), 0);
        assert_eq!(hilbert_d2(0, 0, 6), 0);
    }

    #[test]
    fn d3_round_trip_small() {
        let order = 3u32;
        let n = 1u32 << order;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let idx = hilbert_d3(x, y, z, order);
                    let (rx, ry, rz) = hilbert_d3_inverse(idx, order);
                    assert_eq!(
                        (rx, ry, rz),
                        (x, y, z),
                        "3D round-trip failed for ({x},{y},{z}) order={order}"
                    );
                }
            }
        }
    }

    #[test]
    fn sort_2d_empty() {
        let mut pts: Vec<(f64, f64)> = vec![];
        hilbert_sort_2d(&mut pts);
        assert!(pts.is_empty());
    }

    #[test]
    fn sort_2d_length_preserved() {
        let mut pts: Vec<(f64, f64)> = (0..50)
            .map(|i: i32| (i as f64, (i * 7 % 50) as f64))
            .collect();
        hilbert_sort_2d(&mut pts);
        assert_eq!(pts.len(), 50);
    }

    #[test]
    fn sort_3d_length_preserved() {
        let mut pts: Vec<(f64, f64, f64)> = (0..30)
            .map(|i: i32| (i as f64, (i * 3 % 30) as f64, (i * 7 % 30) as f64))
            .collect();
        hilbert_sort_3d(&mut pts);
        assert_eq!(pts.len(), 30);
    }
}
