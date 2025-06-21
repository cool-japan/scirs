//! Automatic Mesh Generation Examples
//!
//! This example demonstrates the automatic mesh generation capabilities for
//! various geometric domains. The mesh generator can create structured and
//! unstructured meshes with quality control and refinement options.

use scirs2_integrate::pde::finite_element::ElementType;
use scirs2_integrate::pde::mesh_generation::{
    AutoMeshGenerator, BoundarySpecification, Domain, MeshGenerationParams,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Automatic Mesh Generation Examples\n");

    // Example 1: Rectangle mesh with default parameters
    println!("1. Rectangle Mesh Generation");
    let generator = AutoMeshGenerator::default();
    let rectangle_domain = Domain::Rectangle {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 2.0,
        y_max: 1.0,
    };

    let boundary_spec = BoundarySpecification::default();
    let rect_mesh = generator.generate_mesh(&rectangle_domain, &boundary_spec)?;

    println!("   Generated rectangular mesh:");
    println!("   - Points: {}", rect_mesh.points.len());
    println!("   - Elements: {}", rect_mesh.elements.len());

    let rect_quality = generator.assess_mesh_quality(&rect_mesh);
    println!("   - Quality score: {:.3}", rect_quality.quality_score);
    println!("   - Min angle: {:.1}°", rect_quality.min_angle);
    println!("   - Max angle: {:.1}°", rect_quality.max_angle);
    println!(
        "   - Average element size: {:.4}",
        rect_quality.avg_element_size
    );
    println!();

    // Example 2: Circle mesh with custom parameters
    println!("2. Circle Mesh Generation (Custom Parameters)");
    let custom_params = MeshGenerationParams {
        element_size: 0.05,
        min_angle: 25.0,
        max_angle: 135.0,
        quality_iterations: 8,
        element_type: ElementType::Linear,
        boundary_refinement_iterations: 3,
    };

    let fine_generator = AutoMeshGenerator::new(custom_params);
    let circle_domain = Domain::Circle {
        center_x: 0.0,
        center_y: 0.0,
        radius: 1.0,
    };

    let circle_mesh = fine_generator.generate_mesh(&circle_domain, &boundary_spec)?;

    println!("   Generated circular mesh:");
    println!("   - Points: {}", circle_mesh.points.len());
    println!("   - Elements: {}", circle_mesh.elements.len());

    let circle_quality = fine_generator.assess_mesh_quality(&circle_mesh);
    println!("   - Quality score: {:.3}", circle_quality.quality_score);
    println!("   - Min angle: {:.1}°", circle_quality.min_angle);
    println!("   - Max angle: {:.1}°", circle_quality.max_angle);
    println!(
        "   - Average element size: {:.4}",
        circle_quality.avg_element_size
    );
    println!();

    // Example 3: Ellipse mesh
    println!("3. Ellipse Mesh Generation");
    let ellipse_domain = Domain::Ellipse {
        center_x: 0.0,
        center_y: 0.0,
        a: 2.0,                               // Semi-major axis
        b: 1.0,                               // Semi-minor axis
        rotation: std::f64::consts::PI / 6.0, // 30 degrees
    };

    let ellipse_mesh = generator.generate_mesh(&ellipse_domain, &boundary_spec)?;

    println!("   Generated ellipse mesh:");
    println!("   - Points: {}", ellipse_mesh.points.len());
    println!("   - Elements: {}", ellipse_mesh.elements.len());

    let ellipse_quality = generator.assess_mesh_quality(&ellipse_mesh);
    println!("   - Quality score: {:.3}", ellipse_quality.quality_score);
    println!("   - Min angle: {:.1}°", ellipse_quality.min_angle);
    println!("   - Max angle: {:.1}°", ellipse_quality.max_angle);
    println!();

    // Example 4: L-shaped domain
    println!("4. L-shaped Domain Mesh Generation");
    let l_shape_domain = Domain::LShape {
        width: 2.0,
        height: 2.0,
        notch_width: 1.0,
        notch_height: 1.0,
    };

    let l_shape_mesh = generator.generate_mesh(&l_shape_domain, &boundary_spec)?;

    println!("   Generated L-shaped mesh:");
    println!("   - Points: {}", l_shape_mesh.points.len());
    println!("   - Elements: {}", l_shape_mesh.elements.len());

    let l_shape_quality = generator.assess_mesh_quality(&l_shape_mesh);
    println!("   - Quality score: {:.3}", l_shape_quality.quality_score);
    println!("   - Min angle: {:.1}°", l_shape_quality.min_angle);
    println!("   - Max angle: {:.1}°", l_shape_quality.max_angle);
    println!();

    // Example 5: Annulus (ring) mesh
    println!("5. Annulus (Ring) Mesh Generation");
    let annulus_domain = Domain::Annulus {
        center_x: 0.0,
        center_y: 0.0,
        inner_radius: 0.5,
        outer_radius: 1.5,
    };

    let annulus_mesh = generator.generate_mesh(&annulus_domain, &boundary_spec)?;

    println!("   Generated annulus mesh:");
    println!("   - Points: {}", annulus_mesh.points.len());
    println!("   - Elements: {}", annulus_mesh.elements.len());

    let annulus_quality = generator.assess_mesh_quality(&annulus_mesh);
    println!("   - Quality score: {:.3}", annulus_quality.quality_score);
    println!("   - Min angle: {:.1}°", annulus_quality.min_angle);
    println!("   - Max angle: {:.1}°", annulus_quality.max_angle);
    println!();

    // Example 6: Custom polygon mesh
    println!("6. Custom Polygon Mesh Generation");
    use scirs2_integrate::pde::finite_element::Point;

    // Create a pentagon
    let mut pentagon_vertices = Vec::new();
    let n_sides = 5;
    let radius = 1.0;

    for i in 0..n_sides {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_sides as f64;
        pentagon_vertices.push(Point::new(radius * angle.cos(), radius * angle.sin()));
    }

    let polygon_domain = Domain::Polygon {
        vertices: pentagon_vertices,
    };

    let polygon_mesh = generator.generate_mesh(&polygon_domain, &boundary_spec)?;

    println!("   Generated pentagon mesh:");
    println!("   - Points: {}", polygon_mesh.points.len());
    println!("   - Elements: {}", polygon_mesh.elements.len());

    let polygon_quality = generator.assess_mesh_quality(&polygon_mesh);
    println!("   - Quality score: {:.3}", polygon_quality.quality_score);
    println!("   - Min angle: {:.1}°", polygon_quality.min_angle);
    println!("   - Max angle: {:.1}°", polygon_quality.max_angle);
    println!();

    // Example 7: Quality comparison between different element sizes
    println!("7. Element Size Comparison");
    let coarse_params = MeshGenerationParams {
        element_size: 0.2,
        ..Default::default()
    };
    let medium_params = MeshGenerationParams {
        element_size: 0.1,
        ..Default::default()
    };
    let fine_params = MeshGenerationParams {
        element_size: 0.05,
        ..Default::default()
    };

    let test_domain = Domain::Circle {
        center_x: 0.0,
        center_y: 0.0,
        radius: 1.0,
    };

    let generators = [
        ("Coarse", AutoMeshGenerator::new(coarse_params)),
        ("Medium", AutoMeshGenerator::new(medium_params)),
        ("Fine", AutoMeshGenerator::new(fine_params)),
    ];

    for (name, gen) in &generators {
        let mesh = gen.generate_mesh(&test_domain, &boundary_spec)?;
        let quality = gen.assess_mesh_quality(&mesh);

        println!(
            "   {} mesh: {} points, {} elements, avg size: {:.4}",
            name,
            mesh.points.len(),
            mesh.elements.len(),
            quality.avg_element_size
        );
    }
    println!();

    // Example 8: Advanced quality settings
    println!("8. High Quality Mesh Generation");
    let high_quality_params = MeshGenerationParams {
        element_size: 0.08,
        min_angle: 30.0, // Stricter angle constraints
        max_angle: 120.0,
        quality_iterations: 15, // More quality improvement iterations
        element_type: ElementType::Linear,
        boundary_refinement_iterations: 5,
    };

    let hq_generator = AutoMeshGenerator::new(high_quality_params);
    let hq_mesh = hq_generator.generate_mesh(&circle_domain, &boundary_spec)?;
    let hq_quality = hq_generator.assess_mesh_quality(&hq_mesh);

    println!("   High-quality mesh results:");
    println!("   - Points: {}", hq_mesh.points.len());
    println!("   - Elements: {}", hq_mesh.elements.len());
    println!("   - Quality score: {:.3}", hq_quality.quality_score);
    println!("   - Min angle: {:.1}°", hq_quality.min_angle);
    println!("   - Max angle: {:.1}°", hq_quality.max_angle);
    println!(
        "   - Poor quality elements: {}",
        hq_quality.poor_quality_elements
    );
    println!("   - Min aspect ratio: {:.3}", hq_quality.min_aspect_ratio);
    println!();

    println!("All automatic mesh generation examples completed successfully!");
    println!();
    println!("Mesh Generation Summary:");
    println!("- Rectangle: Basic structured mesh for simple domains");
    println!("- Circle: Radial structured mesh with good element distribution");
    println!("- Ellipse: Transformed circular mesh for elliptical domains");
    println!("- L-shape: Combined mesh approach for complex geometries");
    println!("- Annulus: Specialized mesh for ring-shaped domains");
    println!("- Polygon: Flexible mesh generation for arbitrary shapes");
    println!("- Quality control: Automated mesh refinement and smoothing");
    println!("- Customizable parameters: Element size, angle constraints, iterations");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangle_mesh_basic() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Rectangle {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(mesh.points.len() >= 4); // At least corners
        assert!(mesh.elements.len() >= 2); // At least 2 triangles

        // All points should be within domain
        for point in &mesh.points {
            assert!(point.x >= 0.0 && point.x <= 1.0);
            assert!(point.y >= 0.0 && point.y <= 1.0);
        }
    }

    #[test]
    fn test_circle_mesh_basic() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Circle {
            center_x: 0.0,
            center_y: 0.0,
            radius: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(mesh.points.len() > 0);
        assert!(mesh.elements.len() > 0);

        // All points should be within or on circle
        for point in &mesh.points {
            let distance = (point.x.powi(2) + point.y.powi(2)).sqrt();
            assert!(distance <= 1.01); // Allow small numerical tolerance
        }
    }

    #[test]
    fn test_mesh_quality_metrics() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Rectangle {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();
        let quality = generator.assess_mesh_quality(&mesh);

        assert!(quality.min_angle > 0.0 && quality.min_angle < 180.0);
        assert!(quality.max_angle > 0.0 && quality.max_angle < 180.0);
        assert!(quality.min_angle <= quality.max_angle);
        assert!(quality.avg_element_size > 0.0);
        assert!(quality.quality_score >= 0.0 && quality.quality_score <= 1.0);
        assert!(quality.min_aspect_ratio >= 1.0); // Aspect ratio is >= 1
    }

    #[test]
    fn test_element_size_effect() {
        let coarse_params = MeshGenerationParams {
            element_size: 0.2,
            ..Default::default()
        };
        let fine_params = MeshGenerationParams {
            element_size: 0.05,
            ..Default::default()
        };

        let domain = Domain::Circle {
            center_x: 0.0,
            center_y: 0.0,
            radius: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let coarse_generator = AutoMeshGenerator::new(coarse_params);
        let fine_generator = AutoMeshGenerator::new(fine_params);

        let coarse_mesh = coarse_generator
            .generate_mesh(&domain, &boundary_spec)
            .unwrap();
        let fine_mesh = fine_generator
            .generate_mesh(&domain, &boundary_spec)
            .unwrap();

        // Fine mesh should have more elements
        assert!(fine_mesh.elements.len() > coarse_mesh.elements.len());
        assert!(fine_mesh.points.len() > coarse_mesh.points.len());

        let coarse_quality = coarse_generator.assess_mesh_quality(&coarse_mesh);
        let fine_quality = fine_generator.assess_mesh_quality(&fine_mesh);

        // Fine mesh should have smaller average element size
        assert!(fine_quality.avg_element_size < coarse_quality.avg_element_size);
    }

    #[test]
    fn test_annulus_mesh() {
        let generator = AutoMeshGenerator::default();
        let domain = Domain::Annulus {
            center_x: 0.0,
            center_y: 0.0,
            inner_radius: 0.5,
            outer_radius: 1.0,
        };
        let boundary_spec = BoundarySpecification::default();

        let mesh = generator.generate_mesh(&domain, &boundary_spec).unwrap();

        assert!(mesh.points.len() > 0);
        assert!(mesh.elements.len() > 0);

        // All points should be within annulus
        for point in &mesh.points {
            let distance = (point.x.powi(2) + point.y.powi(2)).sqrt();
            assert!(distance >= 0.49 && distance <= 1.01); // Allow tolerance
        }
    }
}
