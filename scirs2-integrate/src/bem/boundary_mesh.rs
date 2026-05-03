//! Boundary mesh discretization for the Boundary Element Method.
//!
//! Provides linear (constant and linear) boundary elements in 2-D, along with
//! canonical mesh constructors for circles, rectangles, and general polygon
//! boundaries.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature tables (nodes on [-1,1])
// ---------------------------------------------------------------------------

/// Returns (nodes, weights) for n-point Gauss-Legendre quadrature on [-1, 1].
pub(crate) fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.577_350_269_189_625_8, 0.577_350_269_189_625_8],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
            vec![
                0.555_555_555_555_556,
                0.888_888_888_888_889,
                0.555_555_555_555_556,
            ],
        ),
        4 => (
            vec![
                -0.861_136_311_594_052_6,
                -0.339_981_043_584_856_3,
                0.339_981_043_584_856_3,
                0.861_136_311_594_052_6,
            ],
            vec![
                0.347_854_845_137_453_8,
                0.652_145_154_862_546_1,
                0.652_145_154_862_546_1,
                0.347_854_845_137_453_8,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.538_469_310_105_683_1,
                0.0,
                0.538_469_310_105_683_1,
                0.906_179_845_938_664,
            ],
            vec![
                0.236_926_885_056_189_1,
                0.478_628_670_499_366_5,
                0.568_888_888_888_888_9,
                0.478_628_670_499_366_5,
                0.236_926_885_056_189_1,
            ],
        ),
        _ => {
            // Default to 5-point rule for unsupported counts
            gauss_legendre_1d(5)
        }
    }
}

// ---------------------------------------------------------------------------
// BoundaryElement
// ---------------------------------------------------------------------------

/// A linear boundary element (panel) in 2-D, defined by two endpoint nodes.
///
/// The element has a constant interpolation of the unknowns, with the
/// collocation point at the midpoint.
#[derive(Debug, Clone)]
pub struct BoundaryElement {
    /// Start node p1 and end node p2 in 2-D.
    pub nodes: [[f64; 2]; 2],
    /// Outward unit normal (perpendicular to element, pointing outward from domain).
    pub normal: [f64; 2],
    /// Length of the element |p2 − p1|.
    pub length: f64,
    /// Midpoint (p1 + p2) / 2 — used as the collocation point.
    pub midpoint: [f64; 2],
}

impl BoundaryElement {
    /// Create a boundary element from two endpoint coordinates.
    ///
    /// The outward normal is computed by rotating the tangent vector 90° to the
    /// right (counterclockwise boundary → outward normal points right of travel).
    pub fn new(p1: [f64; 2], p2: [f64; 2]) -> Self {
        let tx = p2[0] - p1[0];
        let ty = p2[1] - p1[1];
        let length = (tx * tx + ty * ty).sqrt();

        // Outward normal: rotate tangent 90° clockwise (right of travel)
        // For a CCW boundary this gives the outward normal.
        let (nx, ny) = if length > 1e-30 {
            (ty / length, -tx / length)
        } else {
            (0.0, 1.0)
        };

        let midpoint = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5];

        Self {
            nodes: [p1, p2],
            normal: [nx, ny],
            length,
            midpoint,
        }
    }

    /// Produce Gauss quadrature points and weights mapped onto this element.
    ///
    /// Returns a `Vec` of `(point, weight)` pairs where `point` is a 2-D
    /// coordinate on the element and `weight` accounts for the element Jacobian
    /// (half-length = length / 2).
    pub fn quadrature_points(&self, n_points: usize) -> Vec<([f64; 2], f64)> {
        let (xi, w) = gauss_legendre_1d(n_points);
        let p1 = self.nodes[0];
        let p2 = self.nodes[1];
        let half = self.length * 0.5;

        xi.iter()
            .zip(w.iter())
            .map(|(&xi_k, &w_k)| {
                // Map ξ ∈ [-1,1] → physical point on element
                let t = (1.0 + xi_k) * 0.5; // t ∈ [0, 1]
                let pt = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
                (pt, w_k * half)
            })
            .collect()
    }

    /// Return the parametric coordinate for a given parameter t ∈ [0, 1].
    pub fn point_at(&self, t: f64) -> [f64; 2] {
        let p1 = self.nodes[0];
        let p2 = self.nodes[1];
        [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]
    }
}

// ---------------------------------------------------------------------------
// BoundaryMesh
// ---------------------------------------------------------------------------

/// A 2-D boundary mesh composed of linear boundary elements (panels).
///
/// The mesh stores an ordered sequence of elements that together form a closed
/// boundary Γ. Elements are ordered counterclockwise so that the computed
/// normals point outward.
#[derive(Debug, Clone)]
pub struct BoundaryMesh {
    /// Ordered list of boundary elements.
    pub elements: Vec<BoundaryElement>,
    /// Total number of elements (panels).
    pub n_elements: usize,
}

impl BoundaryMesh {
    /// Create an empty boundary mesh.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            n_elements: 0,
        }
    }

    /// Add a boundary element to the mesh.
    pub fn add_element(&mut self, elem: BoundaryElement) {
        self.elements.push(elem);
        self.n_elements += 1;
    }

    /// Create a circular boundary mesh.
    ///
    /// The circle is discretized into `n_panels` equal-length linear elements
    /// ordered counterclockwise. The normals are computed to point outward
    /// (away from the center).
    pub fn circle(center: [f64; 2], radius: f64, n_panels: usize) -> Self {
        let mut mesh = Self::new();
        for i in 0..n_panels {
            let theta1 = 2.0 * PI * i as f64 / n_panels as f64;
            let theta2 = 2.0 * PI * (i + 1) as f64 / n_panels as f64;
            let p1 = [
                center[0] + radius * theta1.cos(),
                center[1] + radius * theta1.sin(),
            ];
            let p2 = [
                center[0] + radius * theta2.cos(),
                center[1] + radius * theta2.sin(),
            ];
            let mut elem = BoundaryElement::new(p1, p2);
            // For a circle, the exact outward normal points away from center.
            let mid_theta = (theta1 + theta2) * 0.5;
            elem.normal = [mid_theta.cos(), mid_theta.sin()];
            mesh.add_element(elem);
        }
        mesh
    }

    /// Create a rectangular boundary mesh.
    ///
    /// The rectangle with corners `(x_min, y_min)` to `(x_max, y_max)` is
    /// discretized with `n_panels_per_side` elements on each of the four sides,
    /// traversed counterclockwise. Normals point outward.
    pub fn rectangle(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        n_panels_per_side: usize,
    ) -> Self {
        let mut mesh = Self::new();
        let n = n_panels_per_side;

        // Bottom edge: left → right, normal = (0, -1)
        for i in 0..n {
            let x1 = x_min + (x_max - x_min) * i as f64 / n as f64;
            let x2 = x_min + (x_max - x_min) * (i + 1) as f64 / n as f64;
            let mut elem = BoundaryElement::new([x1, y_min], [x2, y_min]);
            elem.normal = [0.0, -1.0];
            mesh.add_element(elem);
        }
        // Right edge: bottom → top, normal = (1, 0)
        for i in 0..n {
            let y1 = y_min + (y_max - y_min) * i as f64 / n as f64;
            let y2 = y_min + (y_max - y_min) * (i + 1) as f64 / n as f64;
            let mut elem = BoundaryElement::new([x_max, y1], [x_max, y2]);
            elem.normal = [1.0, 0.0];
            mesh.add_element(elem);
        }
        // Top edge: right → left, normal = (0, 1)
        for i in 0..n {
            let x1 = x_max - (x_max - x_min) * i as f64 / n as f64;
            let x2 = x_max - (x_max - x_min) * (i + 1) as f64 / n as f64;
            let mut elem = BoundaryElement::new([x1, y_max], [x2, y_max]);
            elem.normal = [0.0, 1.0];
            mesh.add_element(elem);
        }
        // Left edge: top → bottom, normal = (-1, 0)
        for i in 0..n {
            let y1 = y_max - (y_max - y_min) * i as f64 / n as f64;
            let y2 = y_max - (y_max - y_min) * (i + 1) as f64 / n as f64;
            let mut elem = BoundaryElement::new([x_min, y1], [x_min, y2]);
            elem.normal = [-1.0, 0.0];
            mesh.add_element(elem);
        }

        mesh
    }

    /// Create a mesh from an ordered polygon (list of vertices, CCW).
    ///
    /// Each consecutive pair of vertices forms one element. The last vertex
    /// is connected back to the first to close the polygon.
    pub fn from_polygon(vertices: &[[f64; 2]]) -> Self {
        let mut mesh = Self::new();
        let n = vertices.len();
        for i in 0..n {
            let p1 = vertices[i];
            let p2 = vertices[(i + 1) % n];
            mesh.add_element(BoundaryElement::new(p1, p2));
        }
        mesh
    }

    /// Compute the total arc length of the boundary.
    pub fn total_length(&self) -> f64 {
        self.elements.iter().map(|e| e.length).sum()
    }

    /// Return the collocation points (midpoints of elements) as a `Vec<[f64; 2]>`.
    pub fn collocation_points(&self) -> Vec<[f64; 2]> {
        self.elements.iter().map(|e| e.midpoint).collect()
    }
}

impl Default for BoundaryMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_normal_circle() {
        // On a unit circle, the outward normal at each panel should point outward.
        let mesh = BoundaryMesh::circle([0.0, 0.0], 1.0, 8);
        for elem in &mesh.elements {
            let n = elem.normal;
            let len = (n[0] * n[0] + n[1] * n[1]).sqrt();
            assert!((len - 1.0).abs() < 1e-10, "Normal should be unit length");
            // midpoint · normal should be positive (outward)
            let m = elem.midpoint;
            let dot = m[0] * n[0] + m[1] * n[1];
            assert!(dot > 0.0, "Normal should point outward from center");
        }
    }

    #[test]
    fn test_circle_total_length() {
        let r = 1.5;
        let n = 64;
        let mesh = BoundaryMesh::circle([0.0, 0.0], r, n);
        let total = mesh.total_length();
        let exact = 2.0 * PI * r;
        assert!(
            (total - exact).abs() / exact < 0.002,
            "Circle perimeter: got {total}, expected {exact}"
        );
    }

    #[test]
    fn test_rectangle_mesh_count() {
        let mesh = BoundaryMesh::rectangle(0.0, 1.0, 0.0, 1.0, 4);
        assert_eq!(mesh.n_elements, 16);
    }

    #[test]
    fn test_quadrature_points_count() {
        let elem = BoundaryElement::new([0.0, 0.0], [1.0, 0.0]);
        let pts = elem.quadrature_points(4);
        assert_eq!(pts.len(), 4);
    }

    #[test]
    fn test_quadrature_weights_sum() {
        let elem = BoundaryElement::new([0.0, 0.0], [2.0, 0.0]);
        let pts = elem.quadrature_points(5);
        let weight_sum: f64 = pts.iter().map(|(_, w)| w).sum();
        // Sum of weights should equal element length
        assert!(
            (weight_sum - elem.length).abs() < 1e-12,
            "Weight sum {weight_sum} != length {}",
            elem.length
        );
    }
}
