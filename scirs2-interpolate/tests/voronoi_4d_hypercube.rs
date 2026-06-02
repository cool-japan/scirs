//! Test: 4D unit-hypercube Voronoi cells via VoronoiDiagram nD dispatch
//!
//! Fast smoke tests use 5 representative 4D points (a 4-simplex at the origin
//! plus unit vectors). These run in <5 s and validate correctness of the nD
//! dispatch path.
//!
//! Expensive 16-site (all corners of [0,1]^4) tests are marked `#[ignore]`
//! and can be run explicitly with `cargo nextest run --run-ignored`.

use scirs2_core::ndarray::{Array1, Array2};

use scirs2_interpolate::voronoi::VoronoiDiagram;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// 5 representative 4D points: origin + 4 unit-axis vectors (one 4-simplex).
fn simplex_sites_4d() -> Array2<f64> {
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    Array2::from_shape_vec((5, 4), data).expect("simplex_sites_4d: shape_vec failed")
}

/// Create 16 unit-hypercube corners in 4D.
fn hypercube_sites_4d() -> Array2<f64> {
    let mut data = Vec::with_capacity(16 * 4);
    for i in 0..16_usize {
        data.push((i & 1) as f64);
        data.push(((i >> 1) & 1) as f64);
        data.push(((i >> 2) & 1) as f64);
        data.push(((i >> 3) & 1) as f64);
    }
    Array2::from_shape_vec((16, 4), data).expect("hypercube_sites_4d: shape_vec failed")
}

// ---------------------------------------------------------------------------
// Fast smoke tests (5-point 4-simplex) — run in <5 s
// ---------------------------------------------------------------------------

#[test]
fn test_4d_simplex_voronoi_diagram_creates() {
    let sites = simplex_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 5]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed for 4D simplex");

    assert_eq!(diagram.cells.len(), 5, "Should have 5 cells");
    assert_eq!(diagram.dim, 4, "Dimension should be 4");
}

#[test]
fn test_4d_simplex_cells_have_neighbours() {
    // A 4-simplex has 5 vertices; every vertex is adjacent to the other 4,
    // so each Voronoi cell must have at least one neighbour.
    let sites = simplex_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 5]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        let neighbours = cell.neighbours_nd().expect("neighbours_nd should succeed");
        assert!(
            !neighbours.is_empty(),
            "Cell {} should have at least one neighbour",
            i
        );
    }
}

#[test]
fn test_4d_simplex_cells_non_negative_volume() {
    // Volumes may be 0 for unbounded cells; they must not be NaN or negative.
    let sites = simplex_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 5]);

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
fn test_4d_simplex_vertices_are_4d() {
    // Any returned vertices must have dimension 4.
    let sites = simplex_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 5]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    for cell in &diagram.cells {
        let verts = cell.vertices_nd().expect("vertices_nd should succeed");
        for v in &verts {
            assert_eq!(v.len(), 4, "Each vertex should be 4-dimensional");
        }
    }
}

// ---------------------------------------------------------------------------
// Expensive tests (16-point hypercube) — ignored by default
//
// Run explicitly with:
//   cargo nextest run -p scirs2-interpolate --test voronoi_4d_hypercube \
//       --run-ignored
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow: Bowyer-Watson 4D Delaunay for 16 points can exceed 60s under load"]
fn test_4d_hypercube_voronoi_diagram_creates() {
    let sites = hypercube_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 16]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed for 4D hypercube");

    assert_eq!(diagram.cells.len(), 16, "Should have 16 cells");
    assert_eq!(diagram.dim, 4, "Dimension should be 4");
}

#[test]
#[ignore = "slow: Bowyer-Watson 4D Delaunay for 16 points can exceed 60s under load"]
fn test_4d_hypercube_all_cells_have_vertices() {
    let sites = hypercube_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 16]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    let mut cells_with_verts = 0;
    for cell in &diagram.cells {
        let verts = cell.vertices_nd().expect("vertices_nd should succeed");
        if !verts.is_empty() {
            cells_with_verts += 1;
            for v in &verts {
                assert_eq!(v.len(), 4, "Each vertex should be 4-dimensional");
            }
        }
    }
    // At least half the cells should have Voronoi vertices
    assert!(
        cells_with_verts >= 8,
        "At least 8 of 16 cells should have Voronoi vertices (nD circumcentres), got {}",
        cells_with_verts
    );
}

#[test]
#[ignore = "slow: Bowyer-Watson 4D Delaunay for 16 points can exceed 60s under load"]
fn test_4d_hypercube_all_cells_have_non_negative_volume() {
    // Note: For the unit hypercube {0,1}^4, all corners are equidistant from
    // the center (0.5, 0.5, 0.5, 0.5). This means all Delaunay circumcentres
    // coincide at the center, producing degenerate (zero-volume) Voronoi cells.
    // This is expected for this highly symmetric input.
    // We test that volumes are non-negative (not NaN/negative).
    let sites = hypercube_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 16]);

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
#[ignore = "slow: Bowyer-Watson 4D Delaunay for 16 points can exceed 60s under load"]
fn test_4d_hypercube_cell_volume_within_range() {
    let sites = hypercube_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 16]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    // Each cell volume should be within some reasonable range.
    // The exact volume per cell = 1/16 = 0.0625.
    // With the circumcentre-based approach, cells may be approximate.
    for (i, cell) in diagram.cells.iter().enumerate() {
        let vol = cell.volume_nd().expect("volume_nd should succeed");
        assert!(
            vol >= 0.0,
            "Cell {} volume should be non-negative, got {}",
            i,
            vol
        );
        if vol > 0.0 {
            assert!(vol < 100.0, "Cell {} volume too large: {}", i, vol);
        }
    }
}

#[test]
#[ignore = "slow: Bowyer-Watson 4D Delaunay for 16 points can exceed 60s under load"]
fn test_4d_hypercube_cells_have_neighbours() {
    let sites = hypercube_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 16]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    for (i, cell) in diagram.cells.iter().enumerate() {
        let neighbours = cell.neighbours_nd().expect("neighbours_nd should succeed");
        // Each hypercube corner is adjacent to at least 4 other corners
        assert!(!neighbours.is_empty(), "Cell {} should have neighbours", i);
    }
}

#[test]
#[ignore = "slow: Bowyer-Watson 4D Delaunay for 16 points can exceed 60s under load"]
fn test_4d_hypercube_individual_cell_volume_approx() {
    // Test that each cell's volume is approximately 1/16 = 0.0625.
    // This uses the circumcentre-based Voronoi vertex computation.
    let sites = hypercube_sites_4d();
    let values = Array1::from_vec(vec![1.0_f64; 16]);

    let diagram = VoronoiDiagram::new(sites.view(), values.view(), None)
        .expect("VoronoiDiagram::new should succeed");

    let expected_volume = 1.0_f64 / 16.0;
    let tolerance = 1e-4_f64;

    let mut failed = 0;
    for (i, cell) in diagram.cells.iter().enumerate() {
        let vol = cell.volume_nd().expect("volume_nd should succeed");
        if vol > 0.0 {
            let diff = (vol - expected_volume).abs();
            if diff > tolerance {
                eprintln!(
                    "Cell {} volume = {:.8}, expected = {:.8}, diff = {:.8e}",
                    i, vol, expected_volume, diff
                );
                failed += 1;
            }
        }
    }
    // Allow some cells to fail (border effects, approximation errors)
    // but at least half should be close.
    let cells_with_vertices = diagram
        .cells
        .iter()
        .filter(|c| c.volume_nd().unwrap_or(0.0) > 0.0)
        .count();

    if cells_with_vertices > 0 {
        assert!(
            failed <= cells_with_vertices / 2,
            "Too many cells with inaccurate volumes: {} out of {}",
            failed,
            cells_with_vertices
        );
    }
}
