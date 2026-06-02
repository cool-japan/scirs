//! Test: 5D Voronoi cells via VoronoiDiagram nD dispatch
//!
//! Tests that the nD Voronoi implementation correctly handles 5-dimensional data.
//! Uses simple well-separated sites that the Delaunay algorithm can handle reliably.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::voronoi::VoronoiDiagram;

/// Generate pseudo-random sites in [0,1]^5 using a simple LCG
fn random_sites_5d(n: usize, seed: u64) -> Array2<f64> {
    let mut data = Vec::with_capacity(n * 5);
    let mut state = seed;
    for _ in 0..n * 5 {
        // LCG parameters (Knuth)
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let val = ((state >> 33) as f64) / (u32::MAX as f64);
        data.push(val);
    }
    Array2::from_shape_vec((n, 5), data).expect("random_sites_5d: shape_vec failed")
}

/// Create a small 5D simplex-like configuration (6 points: standard simplex vertices)
fn simple_sites_5d() -> Array2<f64> {
    // 6 vertices of a regular simplex in 5D (scaled unit simplex)
    // First vertex at origin, remaining at standard basis vectors
    Array2::from_shape_vec(
        (7, 5),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, // origin
            1.0, 0.0, 0.0, 0.0, 0.0, // e1
            0.0, 1.0, 0.0, 0.0, 0.0, // e2
            0.0, 0.0, 1.0, 0.0, 0.0, // e3
            0.0, 0.0, 0.0, 1.0, 0.0, // e4
            0.0, 0.0, 0.0, 0.0, 1.0, // e5
            0.2, 0.2, 0.2, 0.2, 0.2, // interior point
        ],
    )
    .expect("simple_sites_5d: shape_vec failed")
}

#[test]
fn test_5d_voronoi_diagram_creates() {
    let sites = simple_sites_5d();
    let values = Array1::from_vec(vec![1.0_f64; 7]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed for 5D sites");

    assert_eq!(diagram.cells.len(), 7, "Should have 7 cells");
    assert_eq!(diagram.dim, 5, "Dimension should be 5");
}

#[test]
fn test_5d_voronoi_cells_have_valid_data() {
    let sites = simple_sites_5d();
    let values = Array1::from_vec(vec![1.0_f64; 7]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        // Vertices should be accessible and dimension-correct
        let verts = cell.vertices_nd().expect("vertices_nd should succeed");
        for v in &verts {
            assert_eq!(v.len(), 5, "Cell {} vertex should be 5D", i);
        }

        // Volume should be non-negative and finite
        let vol = cell.volume_nd().expect("volume_nd should succeed");
        assert!(
            vol >= 0.0,
            "Cell {} volume should be non-negative, got {}",
            i,
            vol
        );
        assert!(
            vol.is_finite(),
            "Cell {} volume should be finite, got {}",
            i,
            vol
        );

        // Neighbour indices should be valid
        let nb = cell.neighbours_nd().expect("neighbours_nd should succeed");
        for &idx in &nb {
            assert!(idx < 7, "Cell {} neighbour index {} out of range", i, idx);
            assert_ne!(idx, i, "Cell {} lists itself as neighbour", i);
        }
    }
}

#[test]
fn test_5d_voronoi_at_least_some_cells_have_vertices() {
    let sites = simple_sites_5d();
    let values = Array1::from_vec(vec![1.0_f64; 7]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    let cells_with_verts = diagram
        .cells
        .iter()
        .filter(|c| c.vertices_nd().map(|v| !v.is_empty()).unwrap_or(false))
        .count();

    // At least some cells should have Voronoi vertices computed
    assert!(
        cells_with_verts >= 1,
        "At least 1 cell should have Voronoi vertices, got {}",
        cells_with_verts
    );
}

#[test]
fn test_5d_voronoi_at_least_some_cells_have_neighbours() {
    let sites = simple_sites_5d();
    let values = Array1::from_vec(vec![1.0_f64; 7]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    let cells_with_nb = diagram
        .cells
        .iter()
        .filter(|c| c.neighbours_nd().map(|n| !n.is_empty()).unwrap_or(false))
        .count();

    // At least some cells should have Delaunay neighbours
    assert!(
        cells_with_nb >= 1,
        "At least 1 cell should have neighbours, got {}",
        cells_with_nb
    );
}

#[test]
fn test_5d_random_voronoi_diagram_creates() {
    // Test that VoronoiDiagram::new works for 5D random sites
    let sites = random_sites_5d(32, 42);
    let values = Array1::from_vec(vec![1.0_f64; 32]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed for 5D random sites");

    assert_eq!(diagram.cells.len(), 32, "Should have 32 cells");
    assert_eq!(diagram.dim, 5, "Dimension should be 5");
}

#[test]
fn test_5d_random_all_volumes_non_negative() {
    let sites = random_sites_5d(32, 42);
    let values = Array1::from_vec(vec![1.0_f64; 32]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        let vol = cell.volume_nd().expect("volume_nd should succeed");
        assert!(
            vol >= 0.0,
            "Cell {} volume should be non-negative, got {}",
            i,
            vol
        );
        assert!(
            vol.is_finite(),
            "Cell {} volume should be finite, got {}",
            i,
            vol
        );
    }
}

#[test]
fn test_5d_random_vertices_have_correct_dimension() {
    let sites = random_sites_5d(32, 42);
    let values = Array1::from_vec(vec![1.0_f64; 32]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        let verts = cell.vertices_nd().expect("vertices_nd should succeed");
        for (vi, v) in verts.iter().enumerate() {
            assert_eq!(
                v.len(),
                5,
                "Cell {} vertex {} should have 5 coordinates",
                i,
                vi
            );
        }
    }
}

#[test]
fn test_5d_random_neighbours_are_valid_indices() {
    let sites = random_sites_5d(32, 42);
    let values = Array1::from_vec(vec![1.0_f64; 32]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    let n = diagram.cells.len();
    for (i, cell) in diagram.cells.iter().enumerate() {
        let neighbours = cell.neighbours_nd().expect("neighbours_nd should succeed");
        for &nb in &neighbours {
            assert!(nb < n, "Cell {} has out-of-range neighbour index {}", i, nb);
            assert_ne!(nb, i, "Cell {} lists itself as neighbour", i);
        }
    }
}
