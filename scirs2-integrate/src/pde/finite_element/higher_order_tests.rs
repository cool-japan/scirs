//! Comprehensive tests for higher-order finite elements

#[cfg(test)]
mod tests {
    // use super::*; // Unused import, commenting out
    use crate::pde::finite_element::{
        ElementType, FEMOptions, FEMPoissonSolver, HigherOrderMeshGenerator, Point, ShapeFunctions,
        TriangularMesh, TriangularQuadrature,
    };
    use crate::pde::{
        BoundaryCondition as GenericBoundaryCondition, BoundaryConditionType, BoundaryLocation,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_higher_order_mesh_generation() {
        // Create a simple 2x2 triangular mesh
        let linear_mesh = TriangularMesh::generate_rectangular((0.0, 1.0), (0.0, 1.0), 2, 2);

        // Test quadratic mesh generation
        let (quad_points, quad_elements) =
            HigherOrderMeshGenerator::linear_to_quadratic(&linear_mesh).unwrap();

        // Original mesh had 9 points (3x3 grid), quadratic should have more
        assert!(quad_points.len() > linear_mesh.points.len());
        assert_eq!(quad_elements.len(), linear_mesh.elements.len());

        // Each quadratic element should have 6 nodes
        for element in &quad_elements {
            assert_eq!(element.nodes.len(), 6);
            assert_eq!(element.element_type, ElementType::Quadratic);
        }

        // Test cubic mesh generation
        let (cubic_points, cubic_elements) =
            HigherOrderMeshGenerator::linear_to_cubic(&linear_mesh).unwrap();

        // Cubic mesh should have even more points
        assert!(cubic_points.len() > quad_points.len());
        assert_eq!(cubic_elements.len(), linear_mesh.elements.len());

        // Each cubic element should have 10 nodes
        for element in &cubic_elements {
            assert_eq!(element.nodes.len(), 10);
            assert_eq!(element.element_type, ElementType::Cubic);
        }
    }

    #[test]
    fn test_quadrature_integration() {
        // Test that quadrature rules integrate polynomials exactly

        // Test 1-point rule integrates constants exactly
        let (xi, _eta, w) = TriangularQuadrature::get_rule(1).unwrap();
        let mut integral = 0.0;
        for i in 0..xi.len() {
            integral += w[i]; // integrating f(xi,eta) = 1
        }
        assert_relative_eq!(integral, 0.5, epsilon = 1e-12); // area of reference triangle

        // Test 3-point rule integrates linear functions exactly
        let (xi, eta, w) = TriangularQuadrature::get_rule(3).unwrap();

        // Integrate f(xi,eta) = xi
        let mut integral_xi = 0.0;
        for i in 0..xi.len() {
            integral_xi += xi[i] * w[i];
        }
        assert_relative_eq!(integral_xi, 1.0 / 6.0, epsilon = 1e-12); // ∫∫ xi dxi deta = 1/6

        // Integrate f(xi,eta) = eta
        let mut integral_eta = 0.0;
        for i in 0..xi.len() {
            integral_eta += eta[i] * w[i];
        }
        assert_relative_eq!(integral_eta, 1.0 / 6.0, epsilon = 1e-12); // ∫∫ eta dxi deta = 1/6
    }

    #[test]
    fn test_shape_function_properties() {
        // Test partition of unity for all element types
        let test_points = vec![
            (0.0, 0.0),             // corner
            (1.0, 0.0),             // corner
            (0.0, 1.0),             // corner
            (1.0 / 3.0, 1.0 / 3.0), // center
            (0.5, 0.0),             // edge midpoint
            (0.2, 0.3),             // arbitrary point
        ];

        for &(xi, eta) in &test_points {
            // Skip points outside reference triangle
            if xi >= 0.0 && eta >= 0.0 && xi + eta <= 1.0 {
                // Test linear elements
                let n_linear = ShapeFunctions::evaluate(ElementType::Linear, xi, eta).unwrap();
                let sum_linear: f64 = n_linear.iter().sum();
                assert_relative_eq!(sum_linear, 1.0, epsilon = 1e-12);

                // Test quadratic elements
                let n_quad = ShapeFunctions::evaluate(ElementType::Quadratic, xi, eta).unwrap();
                let sum_quad: f64 = n_quad.iter().sum();
                assert_relative_eq!(sum_quad, 1.0, epsilon = 1e-12);

                // Test cubic elements
                let n_cubic = ShapeFunctions::evaluate(ElementType::Cubic, xi, eta).unwrap();
                let sum_cubic: f64 = n_cubic.iter().sum();
                assert_relative_eq!(sum_cubic, 1.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_higher_order_fem_solver_creation() {
        // Create a small mesh
        let mesh = TriangularMesh::generate_rectangular((0.0, 1.0), (0.0, 1.0), 2, 2);

        let source_term = |_x: f64, _y: f64| 1.0;

        let boundary_conditions = vec![GenericBoundaryCondition {
            dimension: 0,
            location: BoundaryLocation::Lower,
            bc_type: BoundaryConditionType::Dirichlet,
            value: 0.0,
            coefficients: None,
        }];

        // Test creation with different element types
        for element_type in &[
            ElementType::Linear,
            ElementType::Quadratic,
            ElementType::Cubic,
        ] {
            let options = FEMOptions {
                element_type: *element_type,
                quadrature_order: 3,
                ..Default::default()
            };

            let solver_result = FEMPoissonSolver::new(
                mesh.clone(),
                source_term,
                boundary_conditions.clone(),
                Some(options),
            );

            assert!(
                solver_result.is_ok(),
                "Failed to create solver for {:?}",
                element_type
            );
        }
    }

    #[test]
    fn test_jacobian_transformation() {
        // Test coordinate transformation for a simple triangle
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(1.0, 0.0);
        let p3 = Point::new(0.0, 1.0);

        // Manual Jacobian computation
        let jacobian = [[p2.x - p1.x, p3.x - p1.x], [p2.y - p1.y, p3.y - p1.y]];

        let det_j = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
        assert_relative_eq!(det_j, 1.0, epsilon = 1e-12); // Unit triangle has determinant 1

        // Test that reference triangle maps correctly
        let xi_ref = [0.0, 1.0, 0.0, 1.0 / 3.0];
        let eta_ref = [0.0, 0.0, 1.0, 1.0 / 3.0];
        let expected_x = [0.0, 1.0, 0.0, 1.0 / 3.0];
        let expected_y = [0.0, 0.0, 1.0, 1.0 / 3.0];

        for i in 0..xi_ref.len() {
            let xi = xi_ref[i];
            let eta = eta_ref[i];
            let zeta = 1.0 - xi - eta;

            // Map from reference to physical coordinates
            let x_phys = zeta * p1.x + xi * p2.x + eta * p3.x;
            let y_phys = zeta * p1.y + xi * p2.y + eta * p3.y;

            assert_relative_eq!(x_phys, expected_x[i], epsilon = 1e-12);
            assert_relative_eq!(y_phys, expected_y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_element_matrix_symmetry() {
        // Create a simple mesh and test that stiffness matrices are symmetric
        let mesh = TriangularMesh::generate_rectangular((0.0, 1.0), (0.0, 1.0), 3, 3);

        let source_term = |_x: f64, _y: f64| 0.0;
        let boundary_conditions = vec![GenericBoundaryCondition {
            dimension: 0,
            location: BoundaryLocation::Lower,
            bc_type: BoundaryConditionType::Dirichlet,
            value: 0.0,
            coefficients: None,
        }];

        // Test with quadratic elements
        let options = FEMOptions {
            element_type: ElementType::Quadratic,
            quadrature_order: 6,
            ..Default::default()
        };

        let solver =
            FEMPoissonSolver::new(mesh, source_term, boundary_conditions, Some(options)).unwrap();

        // Get the first higher-order element for testing
        if let Some(ho_elements) = &solver.higher_order_elements {
            if !ho_elements.is_empty() {
                let (a_e, _b_e) = solver
                    .element_matrices_higher_order(&ho_elements[0])
                    .unwrap();

                // Check symmetry
                let n = a_e.shape()[0];
                for i in 0..n {
                    for j in 0..n {
                        assert_relative_eq!(a_e[[i, j]], a_e[[j, i]], epsilon = 1e-12);
                    }
                }
            }
        }
    }
}
