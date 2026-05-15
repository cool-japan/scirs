use scirs2_spatial::hilbert::{
    hilbert_d2, hilbert_d2_f64, hilbert_d2_inverse, hilbert_d3, hilbert_d3_inverse,
    hilbert_sort_2d, hilbert_sort_3d,
};

#[test]
fn hilbert_d2_round_trip() {
    let order = 6;
    for x in 0..64u32 {
        for y in 0..64u32 {
            let idx = hilbert_d2(x, y, order);
            let (rx, ry) = hilbert_d2_inverse(idx, order);
            assert_eq!((rx, ry), (x, y), "round trip failed for ({x},{y})");
        }
    }
}

#[test]
fn hilbert_d2_locality() {
    // The Hilbert curve provides locality in a statistical sense: the average
    // distance between adjacent cells is much smaller than for a random permutation.
    // We verify: (a) the average neighbor diff is < 25% of the total cell count,
    // and (b) the number of "close" neighbors (diff <= total/4) is at least 50%.
    let order = 4u32;
    let n = 1u32 << order;
    let total_cells = (n * n) as u64;
    let mut sum_diff: u64 = 0;
    let mut count: u64 = 0;
    let mut close_pairs: u64 = 0;
    for x in 0..n {
        for y in 0..n {
            let idx0 = hilbert_d2(x, y, order);
            for (nx, ny) in [(x.wrapping_add(1), y), (x, y.wrapping_add(1))] {
                if nx < n && ny < n {
                    let idx1 = hilbert_d2(nx, ny, order);
                    let diff = (idx0 as i64 - idx1 as i64).unsigned_abs();
                    sum_diff += diff;
                    count += 1;
                    if diff <= total_cells / 4 {
                        close_pairs += 1;
                    }
                }
            }
        }
    }
    let avg_diff = sum_diff / count;
    assert!(
        avg_diff < total_cells / 4,
        "avg locality diff {avg_diff} should be < {} (25% of total {total_cells})",
        total_cells / 4
    );
    // At least half of adjacent pairs should be within 25% of the range.
    assert!(
        close_pairs * 2 >= count,
        "only {close_pairs}/{count} adjacent pairs are within 25% of range"
    );
}

#[test]
fn hilbert_d2_indices_all_distinct() {
    let order = 3;
    let n = 1u32 << order;
    let mut indices: Vec<u64> = (0..n)
        .flat_map(|x| (0..n).map(move |y| hilbert_d2(x, y, order)))
        .collect();
    indices.sort_unstable();
    indices.dedup();
    assert_eq!(indices.len(), (n * n) as usize);
}

#[test]
fn hilbert_d2_f64_quantises_correctly() {
    let order = 4;
    let bbox = (0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64);
    // (0.5, 0.5) should map to a valid index within the total range.
    let idx = hilbert_d2_f64(0.5, 0.5, bbox, order);
    assert!(idx < (1u64 << (2 * order)));
}

#[test]
fn hilbert_sort_2d_preserves_length() {
    let mut pts: Vec<(f64, f64)> = (0..100)
        .map(|i: i32| (i as f64, (i * 37 % 100) as f64))
        .collect();
    let original_len = pts.len();
    hilbert_sort_2d(&mut pts);
    assert_eq!(pts.len(), original_len);
}

#[test]
fn hilbert_sort_2d_empty() {
    let mut pts: Vec<(f64, f64)> = vec![];
    hilbert_sort_2d(&mut pts); // must not panic
}

#[test]
fn hilbert_d3_round_trip() {
    let order = 4;
    for x in 0..8u32 {
        for y in 0..8u32 {
            for z in 0..8u32 {
                let idx = hilbert_d3(x, y, z, order);
                let (rx, ry, rz) = hilbert_d3_inverse(idx, order);
                assert_eq!(
                    (rx, ry, rz),
                    (x, y, z),
                    "3D round trip failed for ({x},{y},{z})"
                );
            }
        }
    }
}

#[test]
fn hilbert_sort_3d_preserves_length_and_no_panic() {
    let mut pts: Vec<(f64, f64, f64)> = (0..50)
        .map(|i: i32| (i as f64, (i * 17 % 50) as f64, (i * 31 % 50) as f64))
        .collect();
    let original_len = pts.len();
    hilbert_sort_3d(&mut pts);
    assert_eq!(pts.len(), original_len);
}
