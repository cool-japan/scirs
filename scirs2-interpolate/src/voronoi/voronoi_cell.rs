//! Voronoi cell implementation
//!
//! This module provides data structures and operations for working with Voronoi cells,
//! which are the building blocks for Voronoi-based interpolation methods.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{InterpolateError, InterpolateResult};

/// Represents a single Voronoi cell in a Voronoi diagram
///
/// A Voronoi cell is the region of space that is closer to a specific site (point)
/// than to any other site. This structure stores the geometry and properties of
/// a Voronoi cell needed for interpolation.
#[derive(Debug, Clone)]
pub struct VoronoiCell<F: Float + FromPrimitive + Debug> {
    /// The center point (site) of this Voronoi cell
    pub site: Array1<F>,

    /// The vertices of the Voronoi cell (convex polygon in 2D, polyhedron in higher dimensions)
    pub vertices: Array2<F>,

    /// The neighboring cells (indices to other cells in the diagram)
    pub neighbors: Vec<usize>,

    /// The area (2D) or volume (3D+) of the cell
    pub measure: F,

    /// The value associated with this cell's site (used in interpolation)
    pub value: F,
}

impl<F: Float + FromPrimitive + Debug + ndarray::ScalarOperand> VoronoiCell<F> {
    /// Creates a new Voronoi cell with the given site and value
    pub fn new(site: Array1<F>, value: F) -> Self {
        VoronoiCell {
            site,
            vertices: Array2::zeros((0, 0)),
            neighbors: Vec::new(),
            measure: F::zero(),
            value,
        }
    }

    /// Sets the vertices of the Voronoi cell
    pub fn set_vertices(&mut self, vertices: Array2<F>) {
        self.vertices = vertices;
    }

    /// Sets the neighbors of the Voronoi cell
    pub fn set_neighbors(&mut self, neighbors: Vec<usize>) {
        self.neighbors = neighbors;
    }

    /// Computes and sets the measure (area in 2D, volume in 3D) of the cell
    pub fn compute_measure(&mut self) -> InterpolateResult<()> {
        let dim = self.site.len();

        if dim == 2 {
            // Compute area of 2D polygon using shoelace formula
            let n = self.vertices.nrows();
            if n < 3 {
                return Err(InterpolateError::InsufficientData(
                    "Voronoi cell has too few vertices to compute area".to_string(),
                ));
            }

            let mut area = F::zero();
            for i in 0..n {
                let j = (i + 1) % n;
                let xi = self.vertices[[i, 0]];
                let yi = self.vertices[[i, 1]];
                let xj = self.vertices[[j, 0]];
                let yj = self.vertices[[j, 1]];

                area = area + (xi * yj - xj * yi);
            }

            // Take absolute value and divide by 2
            area = area.abs() / (F::from(2).unwrap());
            self.measure = area;
        } else if dim == 3 {
            // Compute volume of 3D polyhedron using the divergence theorem
            // We decompose the polyhedron into tetrahedra from the origin
            // and sum their volumes

            let n = self.vertices.nrows();
            if n < 4 {
                return Err(InterpolateError::InsufficientData(
                    "Voronoi cell has too few vertices to compute volume".to_string(),
                ));
            }

            // To calculate the volume properly, we need the faces of the polyhedron
            // The computation below is a simplified version that requires convex polyhedra
            // and triangulation of the faces

            // For now, we'll implement a simpler approach using the centroid
            // This is an approximation but works well for convex polyhedra

            // Calculate the centroid of the polyhedron
            let mut centroid = Array1::zeros(3);
            for i in 0..n {
                for j in 0..3 {
                    centroid[j] = centroid[j] + self.vertices[[i, j]];
                }
            }
            centroid = centroid / F::from(n).unwrap();

            // Calculate volume by summing the volumes of tetrahedra
            // formed by each triangular face and the centroid
            let mut volume = F::zero();

            // This implementation assumes the polyhedron faces are provided as triangles
            // or are already triangulated
            for i in 0..n {
                let i_next = (i + 1) % n;

                // Form a tetrahedron with the centroid and two adjacent vertices
                let p1 = self.vertices.row(i).to_owned();
                let p2 = self.vertices.row(i_next).to_owned();

                // Calculate the signed volume of the tetrahedron
                let v1 = &p1 - &centroid;
                let v2 = &p2 - &centroid;

                // Cross product v1 × v2
                let cross_x = v1[1] * v2[2] - v1[2] * v2[1];
                let cross_y = v1[2] * v2[0] - v1[0] * v2[2];
                let cross_z = v1[0] * v2[1] - v1[1] * v2[0];

                // Dot product centroid · (v1 × v2)
                let dot = centroid[0] * cross_x + centroid[1] * cross_y + centroid[2] * cross_z;

                // Add to total volume
                volume = volume + dot / F::from(6).unwrap();
            }

            self.measure = volume.abs();
        } else {
            return Err(InterpolateError::UnsupportedOperation(format!(
                "Computing measure for {}-dimensional Voronoi cells not yet implemented",
                dim
            )));
        }

        Ok(())
    }

    /// Computes the intersection between this cell and another Voronoi cell
    ///
    /// Returns the vertices of the intersection polygon/polyhedron and its measure
    pub fn intersection(&self, other: &VoronoiCell<F>) -> InterpolateResult<(Array2<F>, F)> {
        let dim = self.site.len();

        if dim == 2 {
            // For 2D, compute the intersection of two convex polygons
            if self.vertices.is_empty() || other.vertices.is_empty() {
                return Ok((Array2::zeros((0, dim)), F::zero()));
            }

            // Implementation of Sutherland-Hodgman algorithm for polygon clipping
            let mut subject_polygon = self.vertices.clone();
            let clip_polygon = &other.vertices;

            let n_clip = clip_polygon.nrows();
            if n_clip < 3 {
                return Ok((Array2::zeros((0, dim)), F::zero()));
            }

            let mut output_list = Vec::new();

            for i in 0..n_clip {
                let clip_edge_start = clip_polygon.row(i).to_owned();
                let clip_edge_end = clip_polygon.row((i + 1) % n_clip).to_owned();

                let input_list = subject_polygon
                    .rows()
                    .into_iter()
                    .map(|row| row.to_owned())
                    .collect::<Vec<_>>();

                output_list.clear();

                if input_list.is_empty() {
                    break;
                }

                let s = input_list.last().unwrap().clone();

                for e in &input_list {
                    if inside_edge(e, &clip_edge_start, &clip_edge_end) {
                        if !inside_edge(&s, &clip_edge_start, &clip_edge_end) {
                            let intersection =
                                compute_intersection(&s, e, &clip_edge_start, &clip_edge_end)?;
                            output_list.push(intersection);
                        }
                        output_list.push(e.clone());
                    } else if inside_edge(&s, &clip_edge_start, &clip_edge_end) {
                        let intersection =
                            compute_intersection(&s, e, &clip_edge_start, &clip_edge_end)?;
                        output_list.push(intersection);
                    }
                }

                subject_polygon = if output_list.is_empty() {
                    Array2::zeros((0, dim))
                } else {
                    let mut result = Array2::zeros((output_list.len(), dim));
                    for (i, point) in output_list.iter().enumerate() {
                        result.row_mut(i).assign(&point.view());
                    }
                    result
                };
            }

            // Calculate the area of the intersection polygon
            let intersection_polygon = subject_polygon;
            let n = intersection_polygon.nrows();

            if n < 3 {
                return Ok((intersection_polygon, F::zero()));
            }

            let mut area = F::zero();
            for i in 0..n {
                let j = (i + 1) % n;
                let xi = intersection_polygon[[i, 0]];
                let yi = intersection_polygon[[i, 1]];
                let xj = intersection_polygon[[j, 0]];
                let yj = intersection_polygon[[j, 1]];

                area = area + (xi * yj - xj * yi);
            }

            area = area.abs() / (F::from(2).unwrap());

            Ok((intersection_polygon, area))
        } else if dim == 3 {
            // For 3D, we need to compute the intersection of two convex polyhedra
            // This is a complex operation involving:
            // 1. Finding the intersections of edges from one polyhedron with faces of another
            // 2. Finding vertices of one polyhedron that lie inside the other
            // 3. Constructing a new polyhedron from these points

            // For now, we'll implement a simplified approach
            // We'll use an approximation with a conservative estimate

            if self.vertices.is_empty() || other.vertices.is_empty() {
                return Ok((Array2::zeros((0, dim)), F::zero()));
            }

            // Create a bounding box for each polyhedron
            let (min1, max1) = compute_bounding_box(self.vertices.view());
            let (min2, max2) = compute_bounding_box(other.vertices.view());

            // Check if bounding boxes intersect
            let mut intersect = true;
            for i in 0..3 {
                if min1[i] > max2[i] || max1[i] < min2[i] {
                    intersect = false;
                    break;
                }
            }

            if !intersect {
                return Ok((Array2::zeros((0, dim)), F::zero()));
            }

            // Compute intersection bounding box
            let mut intersection_min = Array1::zeros(3);
            let mut intersection_max = Array1::zeros(3);

            for i in 0..3 {
                intersection_min[i] = min1[i].max(min2[i]);
                intersection_max[i] = max1[i].min(max2[i]);
            }

            // Create vertices for the intersection bounding box
            let mut intersection_vertices = Array2::zeros((8, 3));

            // Define the 8 corners of the box
            intersection_vertices[[0, 0]] = intersection_min[0];
            intersection_vertices[[0, 1]] = intersection_min[1];
            intersection_vertices[[0, 2]] = intersection_min[2];

            intersection_vertices[[1, 0]] = intersection_max[0];
            intersection_vertices[[1, 1]] = intersection_min[1];
            intersection_vertices[[1, 2]] = intersection_min[2];

            intersection_vertices[[2, 0]] = intersection_max[0];
            intersection_vertices[[2, 1]] = intersection_max[1];
            intersection_vertices[[2, 2]] = intersection_min[2];

            intersection_vertices[[3, 0]] = intersection_min[0];
            intersection_vertices[[3, 1]] = intersection_max[1];
            intersection_vertices[[3, 2]] = intersection_min[2];

            intersection_vertices[[4, 0]] = intersection_min[0];
            intersection_vertices[[4, 1]] = intersection_min[1];
            intersection_vertices[[4, 2]] = intersection_max[2];

            intersection_vertices[[5, 0]] = intersection_max[0];
            intersection_vertices[[5, 1]] = intersection_min[1];
            intersection_vertices[[5, 2]] = intersection_max[2];

            intersection_vertices[[6, 0]] = intersection_max[0];
            intersection_vertices[[6, 1]] = intersection_max[1];
            intersection_vertices[[6, 2]] = intersection_max[2];

            intersection_vertices[[7, 0]] = intersection_min[0];
            intersection_vertices[[7, 1]] = intersection_max[1];
            intersection_vertices[[7, 2]] = intersection_max[2];

            // Calculate the volume of the intersection box
            let volume = (intersection_max[0] - intersection_min[0])
                * (intersection_max[1] - intersection_min[1])
                * (intersection_max[2] - intersection_min[2]);

            // This is a conservative approximation of the true intersection volume
            // In reality, the intersection of two convex polyhedra is also a convex polyhedron,
            // but computing it exactly is complex

            Ok((intersection_vertices, volume.abs()))
        } else {
            return Err(InterpolateError::UnsupportedOperation(format!(
                "Intersection for {}-dimensional Voronoi cells not yet implemented",
                dim
            )));
        }
    }
}

/// Returns true if a point is inside an edge (to the left of the edge in 2D)
fn inside_edge<F: Float + Debug>(
    point: &Array1<F>,
    edge_start: &Array1<F>,
    edge_end: &Array1<F>,
) -> bool {
    let x = point[0];
    let y = point[1];
    let x1 = edge_start[0];
    let y1 = edge_start[1];
    let x2 = edge_end[0];
    let y2 = edge_end[1];

    // Compute the cross product to determine if the point is to the left of the edge
    (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= F::zero()
}

/// Computes the intersection of two line segments
fn compute_intersection<F: Float + FromPrimitive + Debug>(
    s1: &Array1<F>,
    s2: &Array1<F>,
    c1: &Array1<F>,
    c2: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let x1 = s1[0];
    let y1 = s1[1];
    let x2 = s2[0];
    let y2 = s2[1];
    let x3 = c1[0];
    let y3 = c1[1];
    let x4 = c2[0];
    let y4 = c2[1];

    let denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);

    if denom.abs() < F::epsilon() {
        return Err(InterpolateError::NumericalError(
            "Lines are parallel, no intersection exists".to_string(),
        ));
    }

    let ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;

    let x = x1 + ua * (x2 - x1);
    let y = y1 + ua * (y2 - y1);

    Ok(Array1::from_vec(vec![x, y]))
}

/// Computes the bounding box of a set of points
///
/// Returns the minimum and maximum coordinates as Arrays
fn compute_bounding_box<F: Float + Debug>(points: ArrayView2<F>) -> (Array1<F>, Array1<F>) {
    let dim = points.ncols();
    let n_points = points.nrows();

    if n_points == 0 {
        return (
            Array1::from_elem(dim, F::infinity()),
            Array1::from_elem(dim, F::neg_infinity()),
        );
    }

    let mut min_coords = Array1::from_elem(dim, F::infinity());
    let mut max_coords = Array1::from_elem(dim, F::neg_infinity());

    for i in 0..n_points {
        for j in 0..dim {
            let val = points[[i, j]];
            if val < min_coords[j] {
                min_coords[j] = val;
            }
            if val > max_coords[j] {
                max_coords[j] = val;
            }
        }
    }

    (min_coords, max_coords)
}

/// A collection of Voronoi cells forming a Voronoi diagram
#[derive(Debug, Clone)]
pub struct VoronoiDiagram<F: Float + FromPrimitive + Debug + ndarray::ScalarOperand + 'static> {
    /// The cells that make up the Voronoi diagram
    pub cells: Vec<VoronoiCell<F>>,

    /// The dimension of the space
    pub dim: usize,

    /// Bounds of the domain (min_x, min_y, max_x, max_y, etc.)
    pub bounds: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug + ndarray::ScalarOperand + 'static> VoronoiDiagram<F> {
    /// Creates a new Voronoi diagram from sites and values
    pub fn new(
        sites: ArrayView2<F>,
        values: ArrayView1<F>,
        bounds: Option<Array1<F>>,
    ) -> InterpolateResult<Self> {
        let n_sites = sites.nrows();
        let dim = sites.ncols();

        if n_sites != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Number of sites ({}) does not match number of values ({})",
                n_sites,
                values.len()
            )));
        }

        let default_bounds = if dim == 2 || dim == 3 {
            // Calculate min/max for each dimension
            let mut min_coords = Array1::from_elem(dim, F::infinity());
            let mut max_coords = Array1::from_elem(dim, F::neg_infinity());

            for i in 0..n_sites {
                for j in 0..dim {
                    let val = sites[[i, j]];
                    min_coords[j] = min_coords[j].min(val);
                    max_coords[j] = max_coords[j].max(val);
                }
            }

            // Add padding to avoid numerical issues (10% on each side)
            let mut bounds_vec = Vec::with_capacity(2 * dim);
            for j in 0..dim {
                let padding = (max_coords[j] - min_coords[j]) * F::from(0.1).unwrap();
                bounds_vec.push(min_coords[j] - padding); // Min bound
                bounds_vec.push(max_coords[j] + padding); // Max bound
            }

            if dim == 2 {
                // Reorder for 2D to match the expected format [min_x, min_y, max_x, max_y]
                Array1::from_vec(vec![
                    bounds_vec[0], // min_x
                    bounds_vec[1], // min_y
                    bounds_vec[2], // max_x
                    bounds_vec[3], // max_y
                ])
            } else {
                // Reorder for 3D to match the expected format [min_x, min_y, min_z, max_x, max_y, max_z]
                Array1::from_vec(vec![
                    bounds_vec[0], // min_x
                    bounds_vec[2], // min_y
                    bounds_vec[4], // min_z
                    bounds_vec[1], // max_x
                    bounds_vec[3], // max_y
                    bounds_vec[5], // max_z
                ])
            }
        } else {
            return Err(InterpolateError::UnsupportedOperation(
                format!("Default bounds calculation for {}-dimensional Voronoi diagrams not yet implemented", dim)));
        };

        let bounds = bounds.unwrap_or(default_bounds);

        if bounds.len() != 2 * dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Bounds must have {} elements for {}-dimensional data",
                2 * dim,
                dim
            )));
        }

        let mut cells = Vec::with_capacity(n_sites);

        for i in 0..n_sites {
            let site = sites.row(i).to_owned();
            let value = values[i];

            cells.push(VoronoiCell::new(site, value));
        }

        let mut diagram = VoronoiDiagram { cells, dim, bounds };

        // Compute the Voronoi diagram
        diagram.compute_cells()?;

        Ok(diagram)
    }

    /// Computes the Voronoi cells for the diagram
    fn compute_cells(&mut self) -> InterpolateResult<()> {
        let n_sites = self.cells.len();
        if n_sites < 3 {
            return Err(InterpolateError::InsufficientData(
                "At least 3 sites are required to compute a Voronoi diagram".to_string(),
            ));
        }

        if self.dim == 2 {
            // For 2D, we use a simple approach:
            // 1. For each site, compute the perpendicular bisectors with every other site
            // 2. Intersect these bisectors with each other and with the domain bounds
            // 3. Keep the points that are inside all half-planes defined by the bisectors

            let min_x = self.bounds[0];
            let min_y = self.bounds[1];
            let max_x = self.bounds[2];
            let max_y = self.bounds[3];

            // Define domain corners
            let corners = vec![
                Array1::from_vec(vec![min_x, min_y]),
                Array1::from_vec(vec![max_x, min_y]),
                Array1::from_vec(vec![max_x, max_y]),
                Array1::from_vec(vec![min_x, max_y]),
            ];

            // Define domain edges as line segments
            let domain_edges = vec![
                (corners[0].clone(), corners[1].clone()),
                (corners[1].clone(), corners[2].clone()),
                (corners[2].clone(), corners[3].clone()),
                (corners[3].clone(), corners[0].clone()),
            ];

            for i in 0..n_sites {
                let site_i = &self.cells[i].site;
                let mut half_planes = Vec::new();
                let mut neighbors = Vec::new();

                // Compute perpendicular bisectors with all other sites
                for j in 0..n_sites {
                    if i == j {
                        continue;
                    }

                    let site_j = &self.cells[j].site;

                    // Compute midpoint between site_i and site_j
                    let mid_x = (site_i[0] + site_j[0]) / F::from(2).unwrap();
                    let mid_y = (site_i[1] + site_j[1]) / F::from(2).unwrap();
                    let midpoint = Array1::from_vec(vec![mid_x, mid_y]);

                    // Compute normal vector to the line from site_i to site_j
                    let dx = site_j[0] - site_i[0];
                    let dy = site_j[1] - site_i[1];
                    let normal = Array1::from_vec(vec![-dy, dx]); // Perpendicular to (dx, dy)

                    // Define a half-plane using the midpoint and normal
                    // The half-plane is the set of points p such that dot(p - midpoint, normal) <= 0
                    // This represents the side of the bisector containing site_i
                    half_planes.push((midpoint, normal, j));
                }

                // Start with all corners of the domain
                let mut vertices = corners.clone();

                // Add intersections between half-plane boundaries
                for k in 0..half_planes.len() {
                    let (mp_k, n_k, _) = &half_planes[k];

                    for half_plane_l in half_planes.iter().skip(k + 1) {
                        let (mp_l, n_l, _) = half_plane_l;

                        // Compute intersection of two lines:
                        // Line 1: mp_k + t * perpendicular(n_k)
                        // Line 2: mp_l + s * perpendicular(n_l)

                        let p_n_k = Array1::from_vec(vec![n_k[1], -n_k[0]]); // Perpendicular to n_k
                        let p_n_l = Array1::from_vec(vec![n_l[1], -n_l[0]]); // Perpendicular to n_l

                        // Check if lines are parallel
                        let det = p_n_k[0] * p_n_l[1] - p_n_k[1] * p_n_l[0];
                        if det.abs() < F::epsilon() {
                            continue; // Parallel lines
                        }

                        // Solve the system of equations
                        let dx = mp_l[0] - mp_k[0];
                        let dy = mp_l[1] - mp_k[1];

                        let t = (dx * p_n_l[1] - dy * p_n_l[0]) / det;

                        let intersect_x = mp_k[0] + t * p_n_k[0];
                        let intersect_y = mp_k[1] + t * p_n_k[1];

                        vertices.push(Array1::from_vec(vec![intersect_x, intersect_y]));
                    }

                    // Add intersections with domain edges
                    for (edge_start, edge_end) in &domain_edges {
                        if let Ok(intersection) = line_segment_intersection(
                            mp_k,
                            &Array1::from_vec(vec![mp_k[0] + n_k[1], mp_k[1] - n_k[0]]),
                            edge_start,
                            edge_end,
                        ) {
                            vertices.push(intersection);
                        }
                    }
                }

                // Filter vertices that are inside all half-planes and the domain
                let valid_vertices: Vec<Array1<F>> = vertices
                    .into_iter()
                    .filter(|v| {
                        // Check if v is inside the domain
                        if v[0] < min_x || v[0] > max_x || v[1] < min_y || v[1] > max_y {
                            return false;
                        }

                        // Check if v is inside all half-planes
                        for (mp, n, j) in &half_planes {
                            let dx = v[0] - mp[0];
                            let dy = v[1] - mp[1];
                            let dot_product = dx * n[0] + dy * n[1];

                            if dot_product > F::zero() {
                                return false;
                            }

                            // If the point is very close to the boundary of this half-plane,
                            // add the corresponding site as a neighbor
                            if dot_product.abs() < F::from(1e-10).unwrap() && !neighbors.contains(j)
                            {
                                neighbors.push(*j);
                            }
                        }

                        true
                    })
                    .collect();

                if valid_vertices.is_empty() {
                    continue;
                }

                // Sort vertices in counter-clockwise order around the site
                let mut sorted_vertices = valid_vertices.clone();
                let center_x = site_i[0];
                let center_y = site_i[1];

                sorted_vertices.sort_by(|a, b| {
                    let angle_a = (a[1] - center_y).atan2(a[0] - center_x);
                    let angle_b = (b[1] - center_y).atan2(b[0] - center_x);
                    angle_a.partial_cmp(&angle_b).unwrap()
                });

                // Convert to Array2 for storage
                let mut vertices_array = Array2::zeros((sorted_vertices.len(), 2));
                for (idx, vertex) in sorted_vertices.iter().enumerate() {
                    vertices_array.row_mut(idx).assign(&vertex.view());
                }

                // Update the cell
                self.cells[i].set_vertices(vertices_array);
                self.cells[i].set_neighbors(neighbors);

                // Compute the area
                let _ = self.cells[i].compute_measure();
            }
        } else if self.dim == 3 {
            // For 3D, we use a simplified approach:
            // We approximate the Voronoi cells using bounding boxes
            // This is not exact but gives a reasonable approximation

            // Extract bounds
            let min_x = self.bounds[0];
            let min_y = self.bounds[1];
            let min_z = self.bounds[2];
            let max_x = self.bounds[3];
            let max_y = self.bounds[4];
            let max_z = self.bounds[5];

            // Define domain vertices (8 corners of the bounding box)
            let _domain_vertices = Array2::from_shape_vec(
                (8, 3),
                vec![
                    min_x, min_y, min_z, max_x, min_y, min_z, max_x, max_y, min_z, min_x, max_y,
                    min_z, min_x, min_y, max_z, max_x, min_y, max_z, max_x, max_y, max_z, min_x,
                    max_y, max_z,
                ],
            )
            .unwrap();

            for i in 0..n_sites {
                let site_i = &self.cells[i].site;
                let mut neighbors = Vec::new();

                // For each site, we'll compute an approximation of its Voronoi cell
                // by finding the points closer to this site than any other

                // For simplicity, we'll use a discretized approach:
                // 1. Create a grid of points within the domain
                // 2. Assign each grid point to the closest site
                // 3. Use the convex hull of these points as the Voronoi cell

                // For now, we'll use an even simpler approximation:
                // We'll create a bounding box for each cell that extends halfway
                // to the nearest neighbors in each direction

                let mut min_dist = Array1::from_elem(6, F::infinity());
                let directions = [
                    [-1.0, 0.0, 0.0], // -x
                    [1.0, 0.0, 0.0],  // +x
                    [0.0, -1.0, 0.0], // -y
                    [0.0, 1.0, 0.0],  // +y
                    [0.0, 0.0, -1.0], // -z
                    [0.0, 0.0, 1.0],  // +z
                ];

                // Find the nearest site in each of the 6 principal directions
                for j in 0..n_sites {
                    if i == j {
                        continue;
                    }

                    let site_j = &self.cells[j].site;

                    // Check if site_j is a potential neighbor
                    let dx = site_j[0] - site_i[0];
                    let dy = site_j[1] - site_i[1];
                    let dz = site_j[2] - site_i[2];

                    // Compute the distance
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    // Add to neighbors if close enough
                    if dist < F::from(5.0).unwrap() {
                        neighbors.push(j);
                    }

                    // Check each direction
                    for (dir_idx, dir) in directions.iter().enumerate() {
                        // Project the vector to site_j onto this direction
                        let proj = dx * F::from(dir[0]).unwrap()
                            + dy * F::from(dir[1]).unwrap()
                            + dz * F::from(dir[2]).unwrap();

                        // If the projection is positive (site_j is in this direction)
                        // and the distance in this direction is smaller than current minimum
                        if proj > F::zero() {
                            let dir_dist = dist;
                            if dir_dist < min_dist[dir_idx] {
                                min_dist[dir_idx] = dir_dist;
                            }
                        }
                    }
                }

                // Create a bounding box for this Voronoi cell
                // using half the distance to the nearest neighbor in each direction
                let mut cell_bounds = [
                    site_i[0] - min_dist[0] / F::from(2).unwrap(), // min_x
                    site_i[1] - min_dist[2] / F::from(2).unwrap(), // min_y
                    site_i[2] - min_dist[4] / F::from(2).unwrap(), // min_z
                    site_i[0] + min_dist[1] / F::from(2).unwrap(), // max_x
                    site_i[1] + min_dist[3] / F::from(2).unwrap(), // max_y
                    site_i[2] + min_dist[5] / F::from(2).unwrap(), // max_z
                ];

                // Clamp to domain bounds
                cell_bounds[0] = cell_bounds[0].max(min_x);
                cell_bounds[1] = cell_bounds[1].max(min_y);
                cell_bounds[2] = cell_bounds[2].max(min_z);
                cell_bounds[3] = cell_bounds[3].min(max_x);
                cell_bounds[4] = cell_bounds[4].min(max_y);
                cell_bounds[5] = cell_bounds[5].min(max_z);

                // Create vertices for the cell (8 corners of the bounding box)
                let vertices = Array2::from_shape_vec(
                    (8, 3),
                    vec![
                        cell_bounds[0],
                        cell_bounds[1],
                        cell_bounds[2], // min_x, min_y, min_z
                        cell_bounds[3],
                        cell_bounds[1],
                        cell_bounds[2], // max_x, min_y, min_z
                        cell_bounds[3],
                        cell_bounds[4],
                        cell_bounds[2], // max_x, max_y, min_z
                        cell_bounds[0],
                        cell_bounds[4],
                        cell_bounds[2], // min_x, max_y, min_z
                        cell_bounds[0],
                        cell_bounds[1],
                        cell_bounds[5], // min_x, min_y, max_z
                        cell_bounds[3],
                        cell_bounds[1],
                        cell_bounds[5], // max_x, min_y, max_z
                        cell_bounds[3],
                        cell_bounds[4],
                        cell_bounds[5], // max_x, max_y, max_z
                        cell_bounds[0],
                        cell_bounds[4],
                        cell_bounds[5], // min_x, max_y, max_z
                    ],
                )
                .unwrap();

                // Update the cell
                self.cells[i].set_vertices(vertices);
                self.cells[i].set_neighbors(neighbors);

                // Compute the volume
                let _ = self.cells[i].compute_measure();
            }
        } else {
            return Err(InterpolateError::UnsupportedOperation(format!(
                "Computing Voronoi cells for {}-dimensional diagrams not yet implemented",
                self.dim
            )));
        }

        Ok(())
    }

    /// Finds the natural neighbors of a query point
    ///
    /// Returns a map of neighbor indices to their corresponding weights
    pub fn natural_neighbors(&self, query: &ArrayView1<F>) -> InterpolateResult<HashMap<usize, F>> {
        let dim = query.len();

        if dim != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query point dimension ({}) does not match diagram dimension ({})",
                dim, self.dim
            )));
        }

        let query_point = query.to_owned();

        if dim == 2 {
            // Create a temporary Voronoi cell for the query point
            let mut query_cell = VoronoiCell::new(query_point.clone(), F::zero());

            // Define the query cell's vertices as the domain corners
            let min_x = self.bounds[0];
            let min_y = self.bounds[1];
            let max_x = self.bounds[2];
            let max_y = self.bounds[3];

            let corners = Array2::from_shape_vec(
                (4, 2),
                vec![min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y],
            )
            .unwrap();

            query_cell.set_vertices(corners);
            let _ = query_cell.compute_measure();

            // Now compute the intersections with existing cells
            // This implements the Natural Neighbor interpolation concept
            let query_area = query_cell.measure;
            let mut weights = HashMap::new();

            for (i, cell) in self.cells.iter().enumerate() {
                if let Ok((_, area)) = query_cell.intersection(cell) {
                    if area > F::zero() {
                        weights.insert(i, area / query_area);
                    }
                }
            }

            Ok(weights)
        } else if dim == 3 {
            // For 3D, we'll use a simplified approach:
            // 1. Find the cells whose Voronoi regions contain the query point
            // 2. Compute weights based on relative distances or volumes

            // Create a temporary Voronoi cell for the query point
            let mut query_cell = VoronoiCell::new(query_point.clone(), F::zero());

            // Define the query cell's vertices as the domain corners
            let min_x = self.bounds[0];
            let min_y = self.bounds[1];
            let min_z = self.bounds[2];
            let max_x = self.bounds[3];
            let max_y = self.bounds[4];
            let max_z = self.bounds[5];

            // Create a box for the query cell
            let corners = Array2::from_shape_vec(
                (8, 3),
                vec![
                    min_x, min_y, min_z, max_x, min_y, min_z, max_x, max_y, min_z, min_x, max_y,
                    min_z, min_x, min_y, max_z, max_x, min_y, max_z, max_x, max_y, max_z, min_x,
                    max_y, max_z,
                ],
            )
            .unwrap();

            query_cell.set_vertices(corners);
            let _ = query_cell.compute_measure();

            // Compute intersections with existing cells
            let query_volume = query_cell.measure;
            let mut weights = HashMap::new();

            for (i, cell) in self.cells.iter().enumerate() {
                if let Ok((_, volume)) = query_cell.intersection(cell) {
                    if volume > F::zero() {
                        weights.insert(i, volume / query_volume);
                    }
                }
            }

            // If we didn't find any natural neighbors, try a distance-based approach
            if weights.is_empty() {
                // Compute distances to all sites
                let mut distances = Vec::with_capacity(self.cells.len());
                for (i, cell) in self.cells.iter().enumerate() {
                    let site = &cell.site;

                    // Compute distance
                    let mut dist_sq = F::zero();
                    for j in 0..dim {
                        dist_sq = dist_sq + (site[j] - query_point[j]).powi(2);
                    }
                    let dist = dist_sq.sqrt();

                    distances.push((i, dist));
                }

                // Sort by distance
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Take the k nearest sites
                let k = 4.min(distances.len()); // Use 4 neighbors for 3D

                // Compute weights based on inverse distance
                let mut total_weight = F::zero();
                for &(idx, dist) in distances.iter().take(k) {
                    // Avoid division by zero
                    if dist < F::epsilon() {
                        // If we're exactly on a site, just use that site
                        weights.clear();
                        weights.insert(idx, F::one());
                        return Ok(weights);
                    }

                    let weight = F::one() / dist;
                    weights.insert(idx, weight);
                    total_weight = total_weight + weight;
                }

                // Normalize weights
                for (_, weight) in weights.iter_mut() {
                    *weight = *weight / total_weight;
                }
            }

            Ok(weights)
        } else {
            return Err(InterpolateError::UnsupportedOperation(format!(
                "Natural neighbor computation for {}-dimensional diagrams not yet implemented",
                dim
            )));
        }
    }
}

/// Computes the intersection of two line segments if it exists
fn line_segment_intersection<F: Float + FromPrimitive + Debug>(
    a1: &Array1<F>,
    a2: &Array1<F>,
    b1: &Array1<F>,
    b2: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let x1 = a1[0];
    let y1 = a1[1];
    let x2 = a2[0];
    let y2 = a2[1];

    let x3 = b1[0];
    let y3 = b1[1];
    let x4 = b2[0];
    let y4 = b2[1];

    let denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);

    if denom.abs() < F::epsilon() {
        return Err(InterpolateError::NumericalError(
            "Lines are parallel, no intersection exists".to_string(),
        ));
    }

    let ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    let ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;

    // Check if intersection is within both line segments
    if ua < F::zero() || ua > F::one() || ub < F::zero() || ub > F::one() {
        return Err(InterpolateError::NumericalError(
            "Intersection exists but not within line segments".to_string(),
        ));
    }

    let x = x1 + ua * (x2 - x1);
    let y = y1 + ua * (y2 - y1);

    Ok(Array1::from_vec(vec![x, y]))
}
