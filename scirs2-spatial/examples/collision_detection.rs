use scirs2_spatial::collision::{
    circle_circle_collision, continuous_sphere_sphere_collision, point_box2d_collision,
    point_box3d_collision, point_circle_collision, point_sphere_collision, ray_box3d_collision,
    ray_sphere_collision, ray_triangle3d_collision, sphere_sphere_collision, Box2D, Box3D, Circle,
    Sphere, Triangle2D, Triangle3D,
};

fn main() {
    println!("Collision Detection Examples");
    println!("===========================\n");

    // Example 1: Simple point-shape collision tests
    println!("Example 1: Point-Shape Collision Tests");
    println!("-------------------------------------");

    let circle = Circle {
        center: [0.0, 0.0],
        radius: 2.0,
    };

    let box2d = Box2D {
        min: [-1.0, -1.0],
        max: [1.0, 1.0],
    };

    let sphere = Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 2.0,
    };

    let box3d = Box3D {
        min: [-1.0, -1.0, -1.0],
        max: [1.0, 1.0, 1.0],
    };

    let _triangle2d = Triangle2D {
        v1: [0.0, 0.0],
        v2: [2.0, 0.0],
        v3: [1.0, 2.0],
    };

    // Test point-circle collision
    let points_2d = [
        ([0.0, 0.0], "center"),
        ([1.0, 1.0], "inside"),
        ([2.0, 0.0], "on boundary"),
        ([3.0, 0.0], "outside"),
    ];

    println!(
        "Circle at {:?} with radius {}",
        circle.center, circle.radius
    );
    for (point, description) in &points_2d {
        let collision = point_circle_collision(point, &circle);
        println!(
            "  Point {:?} ({}) collides with circle: {}",
            point, description, collision
        );
    }

    // Test point-box2d collision
    println!("\nBox2D from {:?} to {:?}", box2d.min, box2d.max);
    for (point, description) in &points_2d {
        let collision = point_box2d_collision(point, &box2d);
        println!(
            "  Point {:?} ({}) collides with box2d: {}",
            point, description, collision
        );
    }

    // Test point-sphere collision
    let points_3d = [
        ([0.0, 0.0, 0.0], "center"),
        ([1.0, 1.0, 1.0], "inside"),
        ([0.0, 0.0, 2.0], "on boundary"),
        ([0.0, 0.0, 3.0], "outside"),
    ];

    println!(
        "\nSphere at {:?} with radius {}",
        sphere.center, sphere.radius
    );
    for (point, description) in &points_3d {
        let collision = point_sphere_collision(point, &sphere);
        println!(
            "  Point {:?} ({}) collides with sphere: {}",
            point, description, collision
        );
    }

    // Test point-box3d collision
    println!("\nBox3D from {:?} to {:?}", box3d.min, box3d.max);
    for (point, description) in &points_3d {
        let collision = point_box3d_collision(point, &box3d);
        println!(
            "  Point {:?} ({}) collides with box3d: {}",
            point, description, collision
        );
    }

    // Example 2: Shape-Shape collision tests
    println!("\n\nExample 2: Shape-Shape Collision Tests");
    println!("---------------------------------");

    // Define some shapes for collision testing
    let circle1 = Circle {
        center: [0.0, 0.0],
        radius: 2.0,
    };

    let circle2 = Circle {
        center: [3.0, 0.0],
        radius: 1.5,
    };

    let circle3 = Circle {
        center: [5.0, 0.0],
        radius: 1.0,
    };

    // Test circle-circle collision
    println!(
        "Circle1: center {:?}, radius {}",
        circle1.center, circle1.radius
    );
    println!(
        "Circle2: center {:?}, radius {}",
        circle2.center, circle2.radius
    );
    println!(
        "Circle3: center {:?}, radius {}",
        circle3.center, circle3.radius
    );

    println!(
        "  Circle1 collides with itself: {}",
        circle_circle_collision(&circle1, &circle1)
    );
    println!(
        "  Circle1 collides with Circle2: {}",
        circle_circle_collision(&circle1, &circle2)
    );
    println!(
        "  Circle1 collides with Circle3: {}",
        circle_circle_collision(&circle1, &circle3)
    );

    // Test sphere-sphere collision
    let sphere1 = Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 2.0,
    };

    let sphere2 = Sphere {
        center: [3.0, 0.0, 0.0],
        radius: 1.5,
    };

    let sphere3 = Sphere {
        center: [5.0, 0.0, 0.0],
        radius: 1.0,
    };

    println!(
        "\nSphere1: center {:?}, radius {}",
        sphere1.center, sphere1.radius
    );
    println!(
        "Sphere2: center {:?}, radius {}",
        sphere2.center, sphere2.radius
    );
    println!(
        "Sphere3: center {:?}, radius {}",
        sphere3.center, sphere3.radius
    );

    println!(
        "  Sphere1 collides with itself: {}",
        sphere_sphere_collision(&sphere1, &sphere1)
    );
    println!(
        "  Sphere1 collides with Sphere2: {}",
        sphere_sphere_collision(&sphere1, &sphere2)
    );
    println!(
        "  Sphere1 collides with Sphere3: {}",
        sphere_sphere_collision(&sphere1, &sphere3)
    );

    // Example 3: Ray collision tests
    println!("\n\nExample 3: Ray Collision Tests");
    println!("---------------------------");

    let ray_origin1 = [5.0, 0.0, 0.0];
    let ray_direction1 = [-1.0, 0.0, 0.0];

    let ray_origin2 = [5.0, 2.0, 0.0];
    let ray_direction2 = [-1.0, 0.0, 0.0];

    println!(
        "Ray1: origin {:?}, direction {:?}",
        ray_origin1, ray_direction1
    );
    println!(
        "Ray2: origin {:?}, direction {:?}",
        ray_origin2, ray_direction2
    );

    // Test ray-sphere collision
    let ray_sphere = Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    };

    println!(
        "\nRay-Sphere collision tests with sphere at {:?}, radius {}",
        ray_sphere.center, ray_sphere.radius
    );

    match ray_sphere_collision(&ray_origin1, &ray_direction1, &ray_sphere) {
        Some(distance) => println!("  Ray1 intersects sphere at distance {:.2?}", distance),
        None => println!("  Ray1 does not intersect sphere"),
    }

    match ray_sphere_collision(&ray_origin2, &ray_direction2, &ray_sphere) {
        Some(distance) => println!("  Ray2 intersects sphere at distance {:.2?}", distance),
        None => println!("  Ray2 does not intersect sphere"),
    }

    // Test ray-box collision
    let ray_box = Box3D {
        min: [-1.0, -1.0, -1.0],
        max: [1.0, 1.0, 1.0],
    };

    println!(
        "\nRay-Box collision tests with box from {:?} to {:?}",
        ray_box.min, ray_box.max
    );

    match ray_box3d_collision(&ray_origin1, &ray_direction1, &ray_box) {
        Some(distance) => println!("  Ray1 intersects box at distance {:.2?}", distance),
        None => println!("  Ray1 does not intersect box"),
    }

    match ray_box3d_collision(&ray_origin2, &ray_direction2, &ray_box) {
        Some(distance) => println!("  Ray2 intersects box at distance {:.2?}", distance),
        None => println!("  Ray2 does not intersect box"),
    }

    // Test ray-triangle collision
    let ray_triangle = Triangle3D {
        v1: [-1.0, -1.0, 0.0],
        v2: [1.0, -1.0, 0.0],
        v3: [0.0, 1.0, 0.0],
    };

    println!(
        "\nRay-Triangle collision tests with triangle: {:?}, {:?}, {:?}",
        ray_triangle.v1, ray_triangle.v2, ray_triangle.v3
    );

    match ray_triangle3d_collision(&ray_origin1, &ray_direction1, &ray_triangle) {
        Some(distance) => println!("  Ray1 intersects triangle at distance {:.2?}", distance),
        None => println!("  Ray1 does not intersect triangle"),
    }

    match ray_triangle3d_collision(&ray_origin2, &ray_direction2, &ray_triangle) {
        Some(distance) => println!("  Ray2 intersects triangle at distance {:.2?}", distance),
        None => println!("  Ray2 does not intersect triangle"),
    }

    // Example 4: Continuous collision detection
    println!("\n\nExample 4: Continuous Collision Detection");
    println!("---------------------------------------");

    let moving_sphere1 = Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    };

    let velocity1 = [1.0, 0.0, 0.0];

    let moving_sphere2 = Sphere {
        center: [5.0, 0.0, 0.0],
        radius: 1.0,
    };

    let velocity2 = [-1.0, 0.0, 0.0];

    println!("Moving spheres:");
    println!(
        "  Sphere1: center {:?}, radius {}, velocity {:?}",
        moving_sphere1.center, moving_sphere1.radius, velocity1
    );
    println!(
        "  Sphere2: center {:?}, radius {}, velocity {:?}",
        moving_sphere2.center, moving_sphere2.radius, velocity2
    );

    // Test with different time steps
    let time_steps = [1.0, 2.0, 5.0];

    for &time_step in &time_steps {
        println!("\nChecking collision within time step {:.1}:", time_step);

        match continuous_sphere_sphere_collision(
            &moving_sphere1,
            &velocity1,
            &moving_sphere2,
            &velocity2,
            time_step,
        ) {
            Some(collision_time) => {
                println!("  Spheres will collide at time {:.2?}", collision_time);

                // Extract the collision time as a float (the first element of the tuple)
                let collision_time_float = match collision_time {
                    (t, _, _) => t,
                };

                // Calculate positions at collision time
                let position1 = [
                    moving_sphere1.center[0] + velocity1[0] * collision_time_float,
                    moving_sphere1.center[1] + velocity1[1] * collision_time_float,
                    moving_sphere1.center[2] + velocity1[2] * collision_time_float,
                ];

                let position2 = [
                    moving_sphere2.center[0] + velocity2[0] * collision_time_float,
                    moving_sphere2.center[1] + velocity2[1] * collision_time_float,
                    moving_sphere2.center[2] + velocity2[2] * collision_time_float,
                ];

                println!("  Sphere1 position at collision: {:?}", position1);
                println!("  Sphere2 position at collision: {:?}", position2);

                // Calculate distance between centers at collision
                let dx = position1[0] - position2[0];
                let dy = position1[1] - position2[1];
                let dz = position1[2] - position2[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                println!("  Distance between centers: {:.2}", distance);
                println!(
                    "  Sum of radii: {:.2}",
                    moving_sphere1.radius + moving_sphere2.radius
                );
            }
            None => println!("  No collision within the time step"),
        }
    }
}
