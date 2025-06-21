//! Mesh Generation Demo
//!
//! A simple demonstration of the automatic mesh generation capabilities.

use scirs2_integrate::pde::finite_element::{ElementType, Point};
use scirs2_integrate::pde::mesh_generation::{
    AutoMeshGenerator, BoundarySpecification, Domain, MeshGenerationParams,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mesh Generation Demo\n");

    // Create a mesh generator with custom parameters
    let params = MeshGenerationParams {
        element_size: 0.1,
        min_angle: 25.0,
        max_angle: 135.0,
        quality_iterations: 5,
        element_type: ElementType::Linear,
        boundary_refinement_iterations: 3,
    };

    let generator = AutoMeshGenerator::new(params);
    let boundary_spec = BoundarySpecification::default();

    // Generate mesh for a circle
    println!("Generating mesh for a unit circle...");
    let circle_domain = Domain::Circle {
        center_x: 0.0,
        center_y: 0.0,
        radius: 1.0,
    };

    let circle_mesh = generator.generate_mesh(&circle_domain, &boundary_spec)?;
    let circle_quality = generator.assess_mesh_quality(&circle_mesh);

    println!("Circle mesh generated successfully!");
    println!("  Points: {}", circle_mesh.points.len());
    println!("  Elements: {}", circle_mesh.elements.len());
    println!("  Quality score: {:.3}", circle_quality.quality_score);
    println!("  Min angle: {:.1}°", circle_quality.min_angle);
    println!("  Max angle: {:.1}°", circle_quality.max_angle);
    println!();

    // Generate mesh for a rectangle
    println!("Generating mesh for a rectangle...");
    let rect_domain = Domain::Rectangle {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 2.0,
        y_max: 1.0,
    };

    let rect_mesh = generator.generate_mesh(&rect_domain, &boundary_spec)?;
    let rect_quality = generator.assess_mesh_quality(&rect_mesh);

    println!("Rectangle mesh generated successfully!");
    println!("  Points: {}", rect_mesh.points.len());
    println!("  Elements: {}", rect_mesh.elements.len());
    println!("  Quality score: {:.3}", rect_quality.quality_score);
    println!("  Min angle: {:.1}°", rect_quality.min_angle);
    println!("  Max angle: {:.1}°", rect_quality.max_angle);
    println!();

    // Generate mesh for a custom pentagon
    println!("Generating mesh for a pentagon...");
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
    let polygon_quality = generator.assess_mesh_quality(&polygon_mesh);

    println!("Pentagon mesh generated successfully!");
    println!("  Points: {}", polygon_mesh.points.len());
    println!("  Elements: {}", polygon_mesh.elements.len());
    println!("  Quality score: {:.3}", polygon_quality.quality_score);
    println!("  Min angle: {:.1}°", polygon_quality.min_angle);
    println!("  Max angle: {:.1}°", polygon_quality.max_angle);
    println!();

    println!("Mesh generation demo completed successfully!");

    Ok(())
}
