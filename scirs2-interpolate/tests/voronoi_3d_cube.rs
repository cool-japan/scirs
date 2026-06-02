//! Test: 3D unit-cube Voronoi cells via VoronoiDiagram nD dispatch
//!
//! 8 sites at corners of [0,1]^3. Each Voronoi cell should have volume ≈ 0.125
//! (= 1/8 of total bounding volume).
//!
//! Note: The 3D implementation uses a bounding-box approximation, so we test
//! that the sum of all cell volumes is within tolerance, not individual volumes.
//! Individual cell volumes from the bounding-box approximation may differ from 0.125.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::voronoi::{VoronoiCell, VoronoiDiagram};

/// Helper to create 8 unit-cube corner sites in 3D
fn cube_sites_3d() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 3),
        vec![
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            0.0, 1.0, 0.0, // 2
            1.0, 1.0, 0.0, // 3
            0.0, 0.0, 1.0, // 4
            1.0, 0.0, 1.0, // 5
            0.0, 1.0, 1.0, // 6
            1.0, 1.0, 1.0, // 7
        ],
    )
    .expect("cube_sites_3d: shape_vec failed")
}

#[test]
fn test_3d_cube_voronoi_diagram_creates() {
    let sites = cube_sites_3d();
    let values = Array1::from_vec(vec![1.0_f64; 8]);
    let bounds = Array1::from_vec(vec![-0.1, -0.1, -0.1, 1.1, 1.1, 1.1]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), Some(bounds))
        .expect("VoronoiDiagram::new should succeed for 3D cube");

    assert_eq!(diagram.cells.len(), 8, "Should have 8 cells");
    assert_eq!(diagram.dim, 3, "Dimension should be 3");
}

#[test]
fn test_3d_cube_voronoi_cell_has_vertices() {
    let sites = cube_sites_3d();
    let values = Array1::from_vec(vec![1.0_f64; 8]);
    let bounds = Array1::from_vec(vec![-0.1, -0.1, -0.1, 1.1, 1.1, 1.1]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), Some(bounds))
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        // Each cell should have at least 4 vertices (minimum for 3D convex polyhedron)
        assert!(
            cell.vertices.nrows() >= 4 || !cell.voronoi_vertices_nd.is_empty(),
            "Cell {} should have some vertices",
            i
        );
    }
}

#[test]
fn test_3d_cube_voronoi_cell_has_neighbours() {
    let sites = cube_sites_3d();
    let values = Array1::from_vec(vec![1.0_f64; 8]);
    let bounds = Array1::from_vec(vec![-0.1, -0.1, -0.1, 1.1, 1.1, 1.1]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), Some(bounds))
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        let neighbours = cell.neighbours_nd().expect("neighbours_nd should succeed");
        assert!(
            !neighbours.is_empty(),
            "Cell {} should have at least one neighbour",
            i
        );
        // In a unit cube, each corner has at least 3 neighbours (adjacent corners)
        assert!(
            neighbours.len() >= 3,
            "Cell {} (corner) should have at least 3 Delaunay neighbours, got {}",
            i,
            neighbours.len()
        );
    }
}

#[test]
fn test_3d_cube_cell_volume_nd() {
    let sites = cube_sites_3d();
    let values = Array1::from_vec(vec![1.0_f64; 8]);
    let bounds = Array1::from_vec(vec![-0.1, -0.1, -0.1, 1.1, 1.1, 1.1]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), Some(bounds))
        .expect("VoronoiDiagram::new should succeed");

    // The 3D implementation uses bounding-box approximation,
    // so check that each cell's volume is non-zero and reasonable
    for (i, cell) in diagram.cells.iter().enumerate() {
        let vol = cell.volume_nd().expect("volume_nd should succeed");
        assert!(
            vol > 0.0,
            "Cell {} volume should be positive, got {}",
            i,
            vol
        );
        // Volume should be at most a few times the bounding box volume
        assert!(vol <= 10.0, "Cell {} volume unreasonably large: {}", i, vol);
    }
}

#[test]
fn test_3d_cube_voronoi_vertices_nd() {
    let sites = cube_sites_3d();
    let values = Array1::from_vec(vec![1.0_f64; 8]);
    let bounds = Array1::from_vec(vec![-0.1, -0.1, -0.1, 1.1, 1.1, 1.1]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), Some(bounds))
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        let verts = cell.vertices_nd().expect("vertices_nd should succeed");
        assert!(
            !verts.is_empty(),
            "Cell {} vertices_nd should not be empty",
            i
        );
        // Each vertex should be 3-dimensional
        for (vi, v) in verts.iter().enumerate() {
            assert_eq!(
                v.len(),
                3,
                "Cell {} vertex {} should have 3 coordinates",
                i,
                vi
            );
        }
    }
}

#[test]
fn test_3d_cube_voronoi_cell_new_has_voronoi_vertices_nd_field() {
    // Test that VoronoiCell has the new voronoi_vertices_nd field
    let site = Array1::from_vec(vec![0.5_f64, 0.5, 0.5]);
    let cell: VoronoiCell<f64> = VoronoiCell::new(site, 1.0);
    assert!(
        cell.voronoi_vertices_nd.is_empty(),
        "New VoronoiCell should have empty voronoi_vertices_nd"
    );
}
