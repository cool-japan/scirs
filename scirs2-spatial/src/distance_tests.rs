use super::*;

use super::*;
use approx::assert_relative_eq;
use scirs2_core::ndarray::arr2;

#[test]
fn test_euclidean_distance() {
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[4.0, 5.0, 6.0];

    assert_relative_eq!(euclidean(point1, point2), 5.196152422706632, epsilon = 1e-6);
}

#[test]
fn test_manhattan_distance() {
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[4.0, 5.0, 6.0];

    assert_relative_eq!(manhattan(point1, point2), 9.0, epsilon = 1e-6);
}

#[test]
fn test_chebyshev_distance() {
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[4.0, 5.0, 6.0];

    assert_relative_eq!(chebyshev(point1, point2), 3.0, epsilon = 1e-6);
}

#[test]
fn test_minkowski_distance() {
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[4.0, 5.0, 6.0];

    // p = 1 (Manhattan)
    assert_relative_eq!(minkowski(point1, point2, 1.0), 9.0, epsilon = 1e-6);

    // p = 2 (Euclidean)
    assert_relative_eq!(
        minkowski(point1, point2, 2.0),
        5.196152422706632,
        epsilon = 1e-6
    );

    // p = infinity (Chebyshev)
    assert_relative_eq!(
        minkowski(point1, point2, f64::INFINITY),
        3.0,
        epsilon = 1e-6
    );

    // p = 3
    assert_relative_eq!(minkowski(point1, point2, 3.0), 4.3267, epsilon = 1e-4);
}

#[test]
fn test_canberra_distance() {
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[4.0, 5.0, 6.0];

    assert_relative_eq!(canberra(point1, point2), 1.5, epsilon = 1e-6);
}

#[test]
fn test_cosine_distance() {
    // Orthogonal vectors should have distance 1
    let point1 = &[1.0, 0.0];
    let point2 = &[0.0, 1.0];

    assert_relative_eq!(cosine(point1, point2), 1.0, epsilon = 1e-6);

    // Parallel vectors should have distance 0
    let point3 = &[1.0, 2.0];
    let point4 = &[2.0, 4.0];

    assert_relative_eq!(cosine(point3, point4), 0.0, epsilon = 1e-6);

    // 45 degree angle should have distance 1 - sqrt(2)/2
    let point5 = &[1.0, 1.0];
    let point6 = &[1.0, 0.0];

    assert_relative_eq!(cosine(point5, point6), 0.2928932188134525, epsilon = 1e-6);
}

#[test]
fn test_correlation_distance() {
    // Perfectly anti-correlated should have distance 2
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[3.0, 2.0, 1.0];

    assert_relative_eq!(correlation(point1, point2), 2.0, epsilon = 1e-6);

    // Perfectly correlated should have distance 0
    let point3 = &[1.0, 2.0, 3.0];
    let point4 = &[2.0, 4.0, 6.0];

    assert_relative_eq!(correlation(point3, point4), 0.0, epsilon = 1e-6);
}

#[test]
fn test_jaccard_distance() {
    let point1 = &[1.0, 0.0, 1.0];
    let point2 = &[0.0, 1.0, 1.0];

    // 1 element in common, 2 elements different = 2/3
    assert_relative_eq!(jaccard(point1, point2), 2.0 / 3.0, epsilon = 1e-6);

    // Empty sets should have distance 0
    let point3 = &[0.0, 0.0, 0.0];
    let point4 = &[0.0, 0.0, 0.0];

    assert_relative_eq!(jaccard(point3, point4), 0.0, epsilon = 1e-6);

    // No elements in common should have distance 1
    let point5 = &[1.0, 1.0, 0.0];
    let point6 = &[0.0, 0.0, 1.0];

    assert_relative_eq!(jaccard(point5, point6), 1.0, epsilon = 1e-6);
}

#[test]
fn test_hamming_distance() {
    let point1 = &[1.0, 0.0, 1.0, 1.0, 0.0];
    let point2 = &[1.0, 1.0, 0.0, 1.0, 0.0];

    // 2 of 5 coordinates differ = 0.4
    assert_relative_eq!(hamming(point1, point2), 0.4, epsilon = 1e-6);

    // Identical vectors should have distance 0
    let point3 = &[1.0, 1.0, 0.0];
    let point4 = &[1.0, 1.0, 0.0];
    assert_relative_eq!(hamming(point3, point4), 0.0, epsilon = 1e-6);

    // Completely different vectors should have distance 1
    let point5 = &[1.0, 1.0, 1.0];
    let point6 = &[0.0, 0.0, 0.0];
    assert_relative_eq!(hamming(point5, point6), 1.0, epsilon = 1e-6);

    // Mismatched lengths should yield NaN
    let point7 = &[1.0, 0.0];
    let point8 = &[1.0, 0.0, 1.0];
    let mismatched_dist: f64 = hamming(point7, point8);
    assert!(mismatched_dist.is_nan());
}

#[test]
fn test_pdist() {
    let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    let dist_matrix = pdist(&points, euclidean);

    assert_eq!(dist_matrix.shape(), &[3, 3]);

    // Check diagonal is zero
    assert_relative_eq!(dist_matrix[(0, 0)], 0.0, epsilon = 1e-6);
    assert_relative_eq!(dist_matrix[(1, 1)], 0.0, epsilon = 1e-6);
    assert_relative_eq!(dist_matrix[(2, 2)], 0.0, epsilon = 1e-6);

    // Check off-diagonal elements
    assert_relative_eq!(dist_matrix[(0, 1)], 1.0, epsilon = 1e-6);
    assert_relative_eq!(dist_matrix[(0, 2)], 1.0, epsilon = 1e-6);
    assert_relative_eq!(
        dist_matrix[(1, 2)],
        std::f64::consts::SQRT_2,
        epsilon = 1e-6
    );

    // Check symmetry
    assert_relative_eq!(dist_matrix[(1, 0)], dist_matrix[(0, 1)], epsilon = 1e-6);
    assert_relative_eq!(dist_matrix[(2, 0)], dist_matrix[(0, 2)], epsilon = 1e-6);
    assert_relative_eq!(dist_matrix[(2, 1)], dist_matrix[(1, 2)], epsilon = 1e-6);
}

#[test]
fn test_cdist() {
    let x_a = arr2(&[[0.0, 0.0], [1.0, 0.0]]);

    let xb = arr2(&[[0.0, 1.0], [1.0, 1.0]]);

    let dist_matrix = cdist(&x_a, &xb, euclidean).expect("Operation failed");

    assert_eq!(dist_matrix.shape(), &[2, 2]);

    assert_relative_eq!(dist_matrix[(0, 0)], 1.0, epsilon = 1e-6);
    assert_relative_eq!(
        dist_matrix[(0, 1)],
        std::f64::consts::SQRT_2,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        dist_matrix[(1, 0)],
        std::f64::consts::SQRT_2,
        epsilon = 1e-6
    );
    assert_relative_eq!(dist_matrix[(1, 1)], 1.0, epsilon = 1e-6);
}

#[test]
fn test_braycurtis_distance() {
    let point1 = &[1.0, 2.0, 3.0];
    let point2 = &[2.0, 3.0, 4.0];

    let dist = braycurtis(point1, point2);
    // Sum of differences: |1-2| + |2-3| + |3-4| = 3
    // Sum of absolute sums: |1+2| + |2+3| + |3+4| = 3 + 5 + 7 = 15
    // Distance: 3/15 = 1/5 = 0.2
    assert_relative_eq!(dist, 0.2, epsilon = 1e-6);

    // Test identical vectors (should be 0)
    let point3 = &[1.0, 2.0, 3.0];
    let point4 = &[1.0, 2.0, 3.0];
    assert_relative_eq!(braycurtis(point3, point4), 0.0, epsilon = 1e-6);

    // Test with zeros in both vectors
    let point5 = &[0.0, 0.0];
    let point6 = &[0.0, 0.0];
    assert_relative_eq!(braycurtis(point5, point6), 0.0, epsilon = 1e-6);
}

#[test]
fn test_squareform() {
    // Test conversion from condensed to square form
    let condensed = vec![1.0, 2.0, 3.0];
    let square = squareform(&condensed).expect("Operation failed");

    assert_eq!(square.shape(), &[3, 3]);
    assert_relative_eq!(square[(0, 1)], 1.0, epsilon = 1e-6);
    assert_relative_eq!(square[(0, 2)], 2.0, epsilon = 1e-6);
    assert_relative_eq!(square[(1, 2)], 3.0, epsilon = 1e-6);

    // Test conversion from square to condensed form
    let condensed2 = squareform_to_condensed(&square).expect("Operation failed");
    assert_eq!(condensed2, condensed);
}
