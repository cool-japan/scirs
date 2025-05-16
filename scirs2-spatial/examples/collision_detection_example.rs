//! Example demonstrating collision detection in spatial module
//!
//! This example demonstrates various collision detection capabilities
//! including:
//! - 2D and 3D primitive collision detection
//! - Ray-primitive collision tests
//! - Continuous collision detection

// No imports needed
use scirs2_spatial::collision::*;

fn main() {
    println!("Collision Detection Examples");
    println!("===========================\n");

    // 2D Collision Examples
    run_2d_examples();

    // 3D Collision Examples
    run_3d_examples();

    // Continuous Collision Detection
    run_continuous_examples();
}

fn run_2d_examples() {
    println!("2D Collision Detection");
    println!("---------------------");

    // Create a 2D box
    let box2d = Box2D::new([0.0, 0.0], [2.0, 2.0]);
    println!("Box2D: min={:?}, max={:?}", box2d.min, box2d.max);
    println!(
        "  Width: {}, Height: {}, Area: {}",
        box2d.width(),
        box2d.height(),
        box2d.area()
    );
    println!("  Center: {:?}", box2d.center());

    // Create a circle
    let circle = Circle::new([3.0, 1.0], 1.5);
    println!(
        "\nCircle: center={:?}, radius={}",
        circle.center, circle.radius
    );
    println!("  Area: {}", circle.area());

    // Create a triangle
    let triangle = Triangle2D::new([0.0, 0.0], [2.0, 0.0], [1.0, 2.0]);
    println!(
        "\nTriangle2D: a={:?}, b={:?}, c={:?}",
        triangle.v1, triangle.v2, triangle.v3
    );
    println!("  Area: {}", triangle.area());

    // Point collision tests
    let test_points = [
        [1.0, 1.0], // Inside box and triangle
        [3.0, 1.0], // Inside circle
        [4.0, 4.0], // Outside all shapes
    ];

    for (i, point) in test_points.iter().enumerate() {
        println!("\nTest point {}: {:?}", i + 1, point);
        println!("  Point inside box: {}", box2d.contains_point(point));
        println!("  Point inside circle: {}", circle.contains_point(point));
        println!(
            "  Point inside triangle: {}",
            triangle.contains_point(point)
        );
    }

    // Shape-Shape collision tests
    let box2d_2 = Box2D::new([1.0, 1.0], [3.0, 3.0]);
    let box2d_3 = Box2D::new([5.0, 5.0], [6.0, 6.0]);
    let circle_2 = Circle::new([1.0, 1.0], 1.0);

    println!("\nShape-Shape Collisions:");
    println!(
        "  Box-Box collision (overlapping): {}",
        box2d_box2d_collision(&box2d, &box2d_2)
    );
    println!(
        "  Box-Box collision (separated): {}",
        box2d_box2d_collision(&box2d, &box2d_3)
    );
    println!(
        "  Circle-Circle collision: {}",
        circle_circle_collision(&circle, &circle_2)
    );
    println!(
        "  Circle-Box collision: {}",
        circle_box2d_collision(&circle, &box2d)
    );

    // Line segment collision
    let line1 = LineSegment2D::new([0.0, 0.0], [2.0, 2.0]);
    let line2 = LineSegment2D::new([0.0, 2.0], [2.0, 0.0]);
    let line3 = LineSegment2D::new([3.0, 3.0], [4.0, 4.0]);

    println!("\nLine Segment Collisions:");
    println!(
        "  Line1-Line2 collision (intersecting): {}",
        line2d_line2d_collision(&line1, &line2)
    );
    println!(
        "  Line1-Line3 collision (separated): {}",
        line2d_line2d_collision(&line1, &line3)
    );

    println!("\n");
}

fn run_3d_examples() {
    println!("3D Collision Detection");
    println!("---------------------");

    // Create a 3D box
    let box3d = Box3D::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    println!("Box3D: min={:?}, max={:?}", box3d.min, box3d.max);
    println!(
        "  Width: {}, Height: {}, Depth: {}",
        box3d.width(),
        box3d.height(),
        box3d.depth()
    );
    println!("  Volume: {}", box3d.volume());
    println!("  Center: {:?}", box3d.center());

    // Create a sphere
    let sphere = Sphere::new([3.0, 1.0, 1.0], 1.5);
    println!(
        "\nSphere: center={:?}, radius={}",
        sphere.center, sphere.radius
    );
    println!("  Volume: {}", sphere.volume());

    // Create a triangle
    let triangle = Triangle3D::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]);
    println!(
        "\nTriangle3D: a={:?}, b={:?}, c={:?}",
        triangle.v1, triangle.v2, triangle.v3
    );
    println!("  Area: {}", triangle.area());
    println!("  Normal: {:?}", triangle.normal());

    // Point collision tests
    let test_points = [
        [1.0, 1.0, 1.0], // Inside box
        [3.0, 1.0, 1.0], // Inside sphere (at center)
        [5.0, 5.0, 5.0], // Outside all shapes
    ];

    for (i, point) in test_points.iter().enumerate() {
        println!("\nTest point {}: {:?}", i + 1, point);
        println!("  Point inside box: {}", box3d.contains_point(point));
        println!("  Point inside sphere: {}", sphere.contains_point(point));
    }

    // Shape-Shape collision tests
    let box3d_2 = Box3D::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
    let box3d_3 = Box3D::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
    let sphere_2 = Sphere::new([1.0, 1.0, 1.0], 1.0);

    println!("\nShape-Shape Collisions:");
    println!(
        "  Box-Box collision (overlapping): {}",
        box3d_box3d_collision(&box3d, &box3d_2)
    );
    println!(
        "  Box-Box collision (separated): {}",
        box3d_box3d_collision(&box3d, &box3d_3)
    );
    println!(
        "  Sphere-Sphere collision: {}",
        sphere_sphere_collision(&sphere, &sphere_2)
    );
    println!(
        "  Sphere-Box collision: {}",
        sphere_box3d_collision(&sphere, &box3d)
    );

    // Ray collision tests
    println!("\nRay Collision Tests:");

    // Ray that hits the sphere
    let ray_origin = [-5.0, 1.0, 1.0];
    let ray_direction = [1.0, 0.0, 0.0];

    match ray_sphere_collision(&ray_origin, &ray_direction, &sphere) {
        Some((t, hit_point)) => {
            println!("  Ray hits sphere at time t={} at point {:?}", t, hit_point);
        }
        None => {
            println!("  Ray misses sphere");
        }
    }

    // Ray that hits the box
    match ray_box3d_collision(&ray_origin, &ray_direction, &box3d) {
        Some((t_enter, t_exit, hit_point)) => {
            println!(
                "  Ray enters box at time t={} at point {:?}",
                t_enter, hit_point
            );
            println!("  Ray exits box at time t={}", t_exit);
        }
        None => {
            println!("  Ray misses box");
        }
    }

    // Ray that hits the triangle
    match ray_triangle3d_collision(&ray_origin, &ray_direction, &triangle) {
        Some((t, hit_point, barycentric)) => {
            println!(
                "  Ray hits triangle at time t={} at point {:?}",
                t, hit_point
            );
            println!("  Barycentric coordinates: {:?}", barycentric);
        }
        None => {
            println!("  Ray misses triangle");
        }
    }

    println!("\n");
}

fn run_continuous_examples() {
    println!("Continuous Collision Detection");
    println!("-----------------------------");

    // Create two moving spheres
    let sphere1 = Sphere::new([0.0, 0.0, 0.0], 1.0);
    let velocity1 = [1.0, 0.0, 0.0]; // Moving to the right

    let sphere2 = Sphere::new([5.0, 0.0, 0.0], 1.0);
    let velocity2 = [-1.0, 0.0, 0.0]; // Moving to the left

    println!(
        "Sphere 1: position={:?}, radius={}, velocity={:?}",
        sphere1.center, sphere1.radius, velocity1
    );
    println!(
        "Sphere 2: position={:?}, radius={}, velocity={:?}",
        sphere2.center, sphere2.radius, velocity2
    );

    match continuous_sphere_sphere_collision(&sphere1, &velocity1, &sphere2, &velocity2, 10.0) {
        Some((time, pos1, pos2)) => {
            println!("\nSpheres will collide:");
            println!("  Time of collision: {}", time);
            println!("  Sphere 1 position at collision: {:?}", pos1);
            println!("  Sphere 2 position at collision: {:?}", pos2);
        }
        None => {
            println!("\nSpheres will not collide within time limit");
        }
    }

    // Create two spheres that won't collide
    let sphere3 = Sphere::new([0.0, 5.0, 0.0], 1.0);
    let velocity3 = [1.0, 0.0, 0.0]; // Moving to the right, but offset in y

    println!(
        "\nSphere 1: position={:?}, radius={}, velocity={:?}",
        sphere1.center, sphere1.radius, velocity1
    );
    println!(
        "Sphere 3: position={:?}, radius={}, velocity={:?}",
        sphere3.center, sphere3.radius, velocity3
    );

    match continuous_sphere_sphere_collision(&sphere1, &velocity1, &sphere3, &velocity3, 10.0) {
        Some((time, pos1, pos3)) => {
            println!("\nSpheres will collide:");
            println!("  Time of collision: {}", time);
            println!("  Sphere 1 position at collision: {:?}", pos1);
            println!("  Sphere 3 position at collision: {:?}", pos3);
        }
        None => {
            println!("\nSpheres will not collide within time limit");
        }
    }
}
