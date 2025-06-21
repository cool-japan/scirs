//! Higher-order finite element implementations
//!
//! This module provides quadratic and cubic triangular elements for finite element methods.
//! Higher-order elements provide better accuracy and can better represent curved boundaries.

use super::{ElementType, Point, TriangularMesh};
use crate::pde::{PDEError, PDEResult};
use ndarray::Array1;

/// Extended triangle element for higher-order elements
#[derive(Debug, Clone)]
pub struct HigherOrderTriangle {
    /// All node indices for the element
    pub nodes: Vec<usize>,
    /// Element type (Linear, Quadratic, or Cubic)
    pub element_type: ElementType,
    /// Marker for domain/boundary identification
    pub marker: Option<i32>,
}

impl HigherOrderTriangle {
    /// Create a new higher-order triangle
    pub fn new(
        nodes: Vec<usize>,
        element_type: ElementType,
        marker: Option<i32>,
    ) -> PDEResult<Self> {
        let expected_nodes = match element_type {
            ElementType::Linear => 3,
            ElementType::Quadratic => 6,
            ElementType::Cubic => 10,
        };

        if nodes.len() != expected_nodes {
            return Err(PDEError::FiniteElementError(format!(
                "Expected {} nodes for {:?} element, got {}",
                expected_nodes,
                element_type,
                nodes.len()
            )));
        }

        Ok(HigherOrderTriangle {
            nodes,
            element_type,
            marker,
        })
    }

    /// Get corner nodes (first 3 nodes are always corners)
    pub fn corner_nodes(&self) -> [usize; 3] {
        [self.nodes[0], self.nodes[1], self.nodes[2]]
    }

    /// Get edge nodes (if any)
    pub fn edge_nodes(&self) -> Vec<usize> {
        match self.element_type {
            ElementType::Linear => vec![],
            ElementType::Quadratic => self.nodes[3..6].to_vec(),
            ElementType::Cubic => self.nodes[3..9].to_vec(),
        }
    }

    /// Get internal nodes (if any)
    pub fn internal_nodes(&self) -> Vec<usize> {
        match self.element_type {
            ElementType::Linear | ElementType::Quadratic => vec![],
            ElementType::Cubic => self.nodes[9..10].to_vec(),
        }
    }
}

/// Shape function evaluator for higher-order elements
pub struct ShapeFunctions;

impl ShapeFunctions {
    /// Evaluate shape functions at a point (xi, eta) in reference coordinates
    /// Reference triangle: (0,0), (1,0), (0,1)
    pub fn evaluate(element_type: ElementType, xi: f64, eta: f64) -> PDEResult<Array1<f64>> {
        let zeta = 1.0 - xi - eta; // Third barycentric coordinate

        match element_type {
            ElementType::Linear => Ok(Self::linear_shape_functions(xi, eta, zeta)),
            ElementType::Quadratic => Ok(Self::quadratic_shape_functions(xi, eta, zeta)),
            ElementType::Cubic => Ok(Self::cubic_shape_functions(xi, eta, zeta)),
        }
    }

    /// Evaluate shape function derivatives at a point (xi, eta)
    /// Returns [dN/dxi, dN/deta] for each shape function
    pub fn evaluate_derivatives(
        element_type: ElementType,
        xi: f64,
        eta: f64,
    ) -> PDEResult<(Array1<f64>, Array1<f64>)> {
        let zeta = 1.0 - xi - eta;

        match element_type {
            ElementType::Linear => Ok(Self::linear_shape_derivatives(xi, eta, zeta)),
            ElementType::Quadratic => Ok(Self::quadratic_shape_derivatives(xi, eta, zeta)),
            ElementType::Cubic => Ok(Self::cubic_shape_derivatives(xi, eta, zeta)),
        }
    }

    /// Linear shape functions (P1)
    fn linear_shape_functions(xi: f64, eta: f64, zeta: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            zeta, // N1 = 1 - xi - eta
            xi,   // N2 = xi
            eta,  // N3 = eta
        ])
    }

    /// Linear shape function derivatives
    fn linear_shape_derivatives(_xi: f64, _eta: f64, _zeta: f64) -> (Array1<f64>, Array1<f64>) {
        let dxi = Array1::from_vec(vec![-1.0, 1.0, 0.0]);
        let deta = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        (dxi, deta)
    }

    /// Quadratic shape functions (P2)
    /// Node ordering: 3 corners + 3 mid-edge nodes
    fn quadratic_shape_functions(xi: f64, eta: f64, zeta: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            zeta * (2.0 * zeta - 1.0), // N1: corner node 1
            xi * (2.0 * xi - 1.0),     // N2: corner node 2
            eta * (2.0 * eta - 1.0),   // N3: corner node 3
            4.0 * xi * zeta,           // N4: mid-edge node 1-2
            4.0 * xi * eta,            // N5: mid-edge node 2-3
            4.0 * eta * zeta,          // N6: mid-edge node 3-1
        ])
    }

    /// Quadratic shape function derivatives
    fn quadratic_shape_derivatives(xi: f64, eta: f64, zeta: f64) -> (Array1<f64>, Array1<f64>) {
        let dxi = Array1::from_vec(vec![
            1.0 - 4.0 * zeta,  // dN1/dxi
            4.0 * xi - 1.0,    // dN2/dxi
            0.0,               // dN3/dxi
            4.0 * (zeta - xi), // dN4/dxi
            4.0 * eta,         // dN5/dxi
            -4.0 * eta,        // dN6/dxi
        ]);

        let deta = Array1::from_vec(vec![
            1.0 - 4.0 * zeta,   // dN1/deta
            0.0,                // dN2/deta
            4.0 * eta - 1.0,    // dN3/deta
            -4.0 * xi,          // dN4/deta
            4.0 * xi,           // dN5/deta
            4.0 * (zeta - eta), // dN6/deta
        ]);

        (dxi, deta)
    }

    /// Cubic shape functions (P3)
    /// Node ordering: 3 corners + 6 edge nodes (2 per edge) + 1 internal node
    fn cubic_shape_functions(xi: f64, eta: f64, zeta: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            // Corner nodes
            zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0) / 2.0, // N1
            xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0) / 2.0,       // N2
            eta * (3.0 * eta - 1.0) * (3.0 * eta - 2.0) / 2.0,    // N3
            // Edge nodes (2 nodes per edge)
            9.0 * xi * zeta * (3.0 * zeta - 1.0) / 2.0, // N4: edge 1-2, node 1
            9.0 * xi * zeta * (3.0 * xi - 1.0) / 2.0,   // N5: edge 1-2, node 2
            9.0 * xi * eta * (3.0 * xi - 1.0) / 2.0,    // N6: edge 2-3, node 1
            9.0 * xi * eta * (3.0 * eta - 1.0) / 2.0,   // N7: edge 2-3, node 2
            9.0 * eta * zeta * (3.0 * eta - 1.0) / 2.0, // N8: edge 3-1, node 1
            9.0 * eta * zeta * (3.0 * zeta - 1.0) / 2.0, // N9: edge 3-1, node 2
            // Internal node
            27.0 * xi * eta * zeta, // N10: internal node
        ])
    }

    /// Cubic shape function derivatives
    fn cubic_shape_derivatives(xi: f64, eta: f64, zeta: f64) -> (Array1<f64>, Array1<f64>) {
        let dxi = Array1::from_vec(vec![
            // Corner nodes derivatives w.r.t xi
            -(3.0 * zeta - 1.0) * (3.0 * zeta - 2.0) / 2.0
                - zeta * 3.0 * (3.0 * zeta - 2.0) / 2.0
                - zeta * (3.0 * zeta - 1.0) * 3.0 / 2.0,
            (3.0 * xi - 1.0) * (3.0 * xi - 2.0) / 2.0
                + xi * 3.0 * (3.0 * xi - 2.0) / 2.0
                + xi * (3.0 * xi - 1.0) * 3.0 / 2.0,
            0.0,
            // Edge nodes derivatives w.r.t xi
            9.0 * zeta * (3.0 * zeta - 1.0) / 2.0 - 9.0 * xi * zeta * 3.0 / 2.0,
            9.0 * zeta * (3.0 * xi - 1.0) / 2.0 + 9.0 * xi * zeta * 3.0 / 2.0,
            9.0 * eta * (3.0 * xi - 1.0) / 2.0 + 9.0 * xi * eta * 3.0 / 2.0,
            9.0 * eta * (3.0 * eta - 1.0) / 2.0,
            -9.0 * eta * zeta * 3.0 / 2.0,
            -9.0 * eta * zeta * 3.0 / 2.0,
            // Internal node derivative w.r.t xi
            27.0 * eta * zeta,
        ]);

        let deta = Array1::from_vec(vec![
            // Corner nodes derivatives w.r.t eta
            -(3.0 * zeta - 1.0) * (3.0 * zeta - 2.0) / 2.0
                - zeta * 3.0 * (3.0 * zeta - 2.0) / 2.0
                - zeta * (3.0 * zeta - 1.0) * 3.0 / 2.0,
            0.0,
            (3.0 * eta - 1.0) * (3.0 * eta - 2.0) / 2.0
                + eta * 3.0 * (3.0 * eta - 2.0) / 2.0
                + eta * (3.0 * eta - 1.0) * 3.0 / 2.0,
            // Edge nodes derivatives w.r.t eta
            -9.0 * xi * zeta * 3.0 / 2.0,
            -9.0 * xi * zeta * 3.0 / 2.0,
            9.0 * xi * (3.0 * xi - 1.0) / 2.0,
            9.0 * xi * (3.0 * eta - 1.0) / 2.0 + 9.0 * xi * eta * 3.0 / 2.0,
            9.0 * zeta * (3.0 * eta - 1.0) / 2.0 + 9.0 * eta * zeta * 3.0 / 2.0,
            9.0 * zeta * (3.0 * zeta - 1.0) / 2.0 - 9.0 * eta * zeta * 3.0 / 2.0,
            // Internal node derivative w.r.t eta
            27.0 * xi * zeta,
        ]);

        (dxi, deta)
    }
}

/// Gaussian quadrature rules for triangular elements
pub struct TriangularQuadrature;

impl TriangularQuadrature {
    /// Get quadrature points and weights for a given order
    /// Returns (xi_coords, eta_coords, weights)
    pub fn get_rule(order: usize) -> PDEResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        match order {
            1 => Ok(Self::order_1()),
            3 => Ok(Self::order_3()),
            6 => Ok(Self::order_6()),
            12 => Ok(Self::order_12()),
            _ => Err(PDEError::FiniteElementError(format!(
                "Quadrature rule of order {} is not implemented",
                order
            ))),
        }
    }

    /// 1-point rule (exact for linear functions)
    fn order_1() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let xi = Array1::from_vec(vec![1.0 / 3.0]);
        let eta = Array1::from_vec(vec![1.0 / 3.0]);
        let weights = Array1::from_vec(vec![0.5]); // Area of reference triangle is 0.5
        (xi, eta, weights)
    }

    /// 3-point rule (exact for quadratic functions)
    fn order_3() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let xi = Array1::from_vec(vec![0.5, 0.0, 0.5]);
        let eta = Array1::from_vec(vec![0.0, 0.5, 0.5]);
        let weights = Array1::from_vec(vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]);
        (xi, eta, weights)
    }

    /// 6-point rule (exact for cubic functions)
    fn order_6() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let a = 0.816847572980459;
        let b = 0.091576213509771;
        let c = 0.108103018168070;
        let d = 0.445948490915965;

        let xi = Array1::from_vec(vec![b, a, b, d, c, d]);
        let eta = Array1::from_vec(vec![b, b, a, c, d, d]);
        let weights = Array1::from_vec(vec![
            0.109951743655322,
            0.109951743655322,
            0.109951743655322,
            0.223381589678011,
            0.223381589678011,
            0.223381589678011,
        ]);

        // Scale weights by area of reference triangle
        let weights = weights * 0.5;
        (xi, eta, weights)
    }

    /// 12-point rule (exact for 5th order functions)
    fn order_12() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let a = 0.873821971016996;
        let b = 0.063089014491502;
        let c = 0.501426509658179;
        let d = 0.249286745170910;
        let e = 0.636502499121399;
        let f = 0.310352451033785;
        let g = 0.053145049844817;

        let xi = Array1::from_vec(vec![b, a, b, d, c, d, f, e, g, g, f, e]);
        let eta = Array1::from_vec(vec![b, b, a, d, d, c, g, g, f, e, e, f]);
        let weights = Array1::from_vec(vec![
            0.050844906370207,
            0.050844906370207,
            0.050844906370207,
            0.116786275726379,
            0.116786275726379,
            0.116786275726379,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
            0.082851075618374,
        ]);

        // Scale weights by area of reference triangle
        let weights = weights * 0.5;
        (xi, eta, weights)
    }
}

/// Mesh refinement utilities for higher-order elements
pub struct HigherOrderMeshGenerator;

impl HigherOrderMeshGenerator {
    /// Generate a quadratic mesh from a linear mesh
    pub fn linear_to_quadratic(
        linear_mesh: &TriangularMesh,
    ) -> PDEResult<(Vec<Point>, Vec<HigherOrderTriangle>)> {
        let mut new_points = linear_mesh.points.clone();
        let mut quadratic_elements = Vec::new();
        let mut edge_midpoints = std::collections::HashMap::new();

        // Process each linear triangle
        for element in &linear_mesh.elements {
            let [n1, n2, n3] = element.nodes;

            // Get or create midpoint nodes for each edge
            let mid12 = Self::get_or_create_midpoint(&mut new_points, &mut edge_midpoints, n1, n2);
            let mid23 = Self::get_or_create_midpoint(&mut new_points, &mut edge_midpoints, n2, n3);
            let mid31 = Self::get_or_create_midpoint(&mut new_points, &mut edge_midpoints, n3, n1);

            // Create quadratic element with 6 nodes
            let nodes = vec![n1, n2, n3, mid12, mid23, mid31];
            let quad_element =
                HigherOrderTriangle::new(nodes, ElementType::Quadratic, element.marker)?;

            quadratic_elements.push(quad_element);
        }

        Ok((new_points, quadratic_elements))
    }

    /// Generate a cubic mesh from a linear mesh
    pub fn linear_to_cubic(
        linear_mesh: &TriangularMesh,
    ) -> PDEResult<(Vec<Point>, Vec<HigherOrderTriangle>)> {
        let mut new_points = linear_mesh.points.clone();
        let mut cubic_elements = Vec::new();
        let mut edge_nodes = std::collections::HashMap::new();

        // Process each linear triangle
        for element in &linear_mesh.elements {
            let [n1, n2, n3] = element.nodes;

            // Get or create two nodes on each edge (at 1/3 and 2/3 positions)
            let (edge12_1, edge12_2) =
                Self::get_or_create_edge_nodes(&mut new_points, &mut edge_nodes, n1, n2);
            let (edge23_1, edge23_2) =
                Self::get_or_create_edge_nodes(&mut new_points, &mut edge_nodes, n2, n3);
            let (edge31_1, edge31_2) =
                Self::get_or_create_edge_nodes(&mut new_points, &mut edge_nodes, n3, n1);

            // Create internal node at centroid
            let p1 = &linear_mesh.points[n1];
            let p2 = &linear_mesh.points[n2];
            let p3 = &linear_mesh.points[n3];
            let centroid = Point::new((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0);
            let internal_node = new_points.len();
            new_points.push(centroid);

            // Create cubic element with 10 nodes
            let nodes = vec![
                n1,
                n2,
                n3, // Corner nodes
                edge12_1,
                edge12_2, // Edge 1-2 nodes
                edge23_1,
                edge23_2, // Edge 2-3 nodes
                edge31_1,
                edge31_2,      // Edge 3-1 nodes
                internal_node, // Internal node
            ];
            let cubic_element =
                HigherOrderTriangle::new(nodes, ElementType::Cubic, element.marker)?;

            cubic_elements.push(cubic_element);
        }

        Ok((new_points, cubic_elements))
    }

    /// Get or create midpoint between two nodes
    fn get_or_create_midpoint(
        points: &mut Vec<Point>,
        midpoints: &mut std::collections::HashMap<(usize, usize), usize>,
        n1: usize,
        n2: usize,
    ) -> usize {
        let key = if n1 < n2 { (n1, n2) } else { (n2, n1) };

        if let Some(&midpoint_idx) = midpoints.get(&key) {
            midpoint_idx
        } else {
            let p1 = &points[n1];
            let p2 = &points[n2];
            let midpoint = Point::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
            let midpoint_idx = points.len();
            points.push(midpoint);
            midpoints.insert(key, midpoint_idx);
            midpoint_idx
        }
    }

    /// Get or create two nodes on an edge (at 1/3 and 2/3 positions)
    fn get_or_create_edge_nodes(
        points: &mut Vec<Point>,
        edge_nodes: &mut std::collections::HashMap<(usize, usize), (usize, usize)>,
        n1: usize,
        n2: usize,
    ) -> (usize, usize) {
        let key = if n1 < n2 { (n1, n2) } else { (n2, n1) };
        let reversed = n1 > n2;

        if let Some(&(node1, node2)) = edge_nodes.get(&key) {
            if reversed {
                (node2, node1)
            } else {
                (node1, node2)
            }
        } else {
            let p1 = &points[n1];
            let p2 = &points[n2];

            // Create two nodes at 1/3 and 2/3 along the edge
            let node1_pos = Point::new(p1.x + (p2.x - p1.x) / 3.0, p1.y + (p2.y - p1.y) / 3.0);
            let node2_pos = Point::new(
                p1.x + 2.0 * (p2.x - p1.x) / 3.0,
                p1.y + 2.0 * (p2.y - p1.y) / 3.0,
            );

            let node1_idx = points.len();
            points.push(node1_pos);
            let node2_idx = points.len();
            points.push(node2_pos);

            edge_nodes.insert(key, (node1_idx, node2_idx));

            if reversed {
                (node2_idx, node1_idx)
            } else {
                (node1_idx, node2_idx)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_shape_functions() {
        // Test at corners of reference triangle
        let n = ShapeFunctions::evaluate(ElementType::Linear, 0.0, 0.0).unwrap();
        assert_relative_eq!(n[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(n[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(n[2], 0.0, epsilon = 1e-12);

        let n = ShapeFunctions::evaluate(ElementType::Linear, 1.0, 0.0).unwrap();
        assert_relative_eq!(n[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(n[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(n[2], 0.0, epsilon = 1e-12);

        let n = ShapeFunctions::evaluate(ElementType::Linear, 0.0, 1.0).unwrap();
        assert_relative_eq!(n[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(n[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(n[2], 1.0, epsilon = 1e-12);

        // Test partition of unity at center
        let n = ShapeFunctions::evaluate(ElementType::Linear, 1.0 / 3.0, 1.0 / 3.0).unwrap();
        let sum: f64 = n.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_quadratic_shape_functions() {
        // Test partition of unity at center
        let n = ShapeFunctions::evaluate(ElementType::Quadratic, 1.0 / 3.0, 1.0 / 3.0).unwrap();
        let sum: f64 = n.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);

        // Test at corner nodes
        let n = ShapeFunctions::evaluate(ElementType::Quadratic, 0.0, 0.0).unwrap();
        assert_relative_eq!(n[0], 1.0, epsilon = 1e-12);
        for i in 1..6 {
            assert_relative_eq!(n[i], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_cubic_shape_functions() {
        // Test partition of unity at center
        let n = ShapeFunctions::evaluate(ElementType::Cubic, 1.0 / 3.0, 1.0 / 3.0).unwrap();
        let sum: f64 = n.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);

        // Test at corner nodes
        let n = ShapeFunctions::evaluate(ElementType::Cubic, 0.0, 0.0).unwrap();
        assert_relative_eq!(n[0], 1.0, epsilon = 1e-12);
        for i in 1..10 {
            assert_relative_eq!(n[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_quadrature_rules() {
        // Test that weights sum to area of reference triangle (0.5)
        let (_, _, weights) = TriangularQuadrature::get_rule(1).unwrap();
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 0.5, epsilon = 1e-12);

        let (_, _, weights) = TriangularQuadrature::get_rule(3).unwrap();
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 0.5, epsilon = 1e-12);

        let (_, _, weights) = TriangularQuadrature::get_rule(6).unwrap();
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_higher_order_triangle_creation() {
        // Test linear triangle
        let linear = HigherOrderTriangle::new(vec![0, 1, 2], ElementType::Linear, None).unwrap();
        assert_eq!(linear.nodes.len(), 3);

        // Test quadratic triangle
        let quadratic =
            HigherOrderTriangle::new(vec![0, 1, 2, 3, 4, 5], ElementType::Quadratic, None).unwrap();
        assert_eq!(quadratic.nodes.len(), 6);

        // Test cubic triangle
        let cubic =
            HigherOrderTriangle::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ElementType::Cubic, None)
                .unwrap();
        assert_eq!(cubic.nodes.len(), 10);

        // Test error case
        let result = HigherOrderTriangle::new(vec![0, 1], ElementType::Linear, None);
        assert!(result.is_err());
    }
}
