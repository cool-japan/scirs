use ndarray::Array1;
use scirs2_spatial::pathplanning::{
    PotentialConfig, PotentialField2DPlanner, PotentialFieldPlanner,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Potential Field Pathfinding Examples");
    println!("===================================\n");

    // Example 1: Simple 2D environment with circular obstacles
    println!("Example 1: Simple 2D environment with circular obstacles");
    println!("-----------------------------------------------------");

    // Create a configuration for the potential field planner
    let config = PotentialConfig::new()
        .with_attractive_gain(1.0)
        .with_repulsive_gain(100.0)
        .with_influence_radius(3.0)
        .with_step_size(0.1)
        .with_max_iterations(500);

    // Create a potential field planner
    let mut planner = PotentialFieldPlanner::new_2d(config);

    // Add some circular obstacles
    planner.add_circular_obstacle([3.0, 3.0], 1.0);
    planner.add_circular_obstacle([5.0, 7.0], 1.5);
    planner.add_circular_obstacle([7.0, 3.0], 1.2);

    // Define start and goal points
    let start = Array1::from_vec(vec![1.0, 1.0]);
    let goal = Array1::from_vec(vec![9.0, 9.0]);

    println!(
        "Finding path from [{:.1}, {:.1}] to [{:.1}, {:.1}]",
        start[0], start[1], goal[0], goal[1]
    );
    println!("Obstacles:");
    println!("- Circle at (3,3) with radius 1.0");
    println!("- Circle at (5,7) with radius 1.5");
    println!("- Circle at (7,3) with radius 1.2");

    // Time the path planning
    let start_time = Instant::now();
    let path = planner.plan(&start, &goal)?;
    let plan_time = start_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and length {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                if i == 0 || i == path.nodes.len() - 1 || i % 5 == 0 {
                    // Print fewer points
                    println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
                }
            }
            println!("Path finding time: {:.2?}", plan_time);

            // Simple ASCII visualization
            print_ascii_visualization_circles(
                &path.nodes,
                &[([3.0, 3.0], 1.0), ([5.0, 7.0], 1.5), ([7.0, 3.0], 1.2)],
            );
        }
        None => println!("No path found - stuck in local minimum!"),
    }

    // Example 2: Environment with polygon obstacles
    println!("\n\nExample 2: Environment with polygon obstacles");
    println!("-------------------------------------------");

    // Create a configuration with different parameters
    let config = PotentialConfig::new()
        .with_attractive_gain(1.0)
        .with_repulsive_gain(200.0)
        .with_influence_radius(2.0)
        .with_step_size(0.1)
        .with_max_iterations(1000);

    // Create a specialized 2D planner
    let mut planner = PotentialField2DPlanner::new(config);

    // Add polygon obstacles
    planner.add_polygon_obstacle(vec![
        [3.0, 3.0],
        [5.0, 3.0],
        [5.0, 5.0],
        [3.0, 5.0], // Square
    ]);

    planner.add_polygon_obstacle(vec![
        [6.0, 6.0],
        [8.0, 6.0],
        [7.0, 8.0], // Triangle
    ]);

    // Define start and goal
    let start = [1.0, 1.0];
    let goal = [9.0, 9.0];

    println!(
        "Finding path from [{:.1}, {:.1}] to [{:.1}, {:.1}]",
        start[0], start[1], goal[0], goal[1]
    );
    println!("Obstacles:");
    println!("- Square at (3,3) to (5,5)");
    println!("- Triangle at (6,6), (8,6), (7,8)");

    // Time the path planning
    let start_time = Instant::now();
    let path = planner.plan(start, goal)?;
    let plan_time = start_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and length {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                if i == 0 || i == path.nodes.len() - 1 || i % 5 == 0 {
                    // Print fewer points
                    println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
                }
            }
            println!("Path finding time: {:.2?}", plan_time);

            // Simple ASCII visualization
            print_ascii_visualization_polygons(
                &path.nodes,
                &[
                    vec![[3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0]],
                    vec![[6.0, 6.0], [8.0, 6.0], [7.0, 8.0]],
                ],
            );
        }
        None => println!("No path found - stuck in local minimum!"),
    }

    // Example 3: Local minimum trap
    println!("\n\nExample 3: Local minimum trap (U-shaped obstacle)");
    println!("-----------------------------------------------");

    // Create a configuration that will likely get stuck
    let config = PotentialConfig::new()
        .with_attractive_gain(0.8)
        .with_repulsive_gain(300.0)
        .with_influence_radius(3.0)
        .with_step_size(0.05)
        .with_min_force_threshold(0.1) // Higher threshold to detect minima
        .with_max_iterations(200);

    let mut planner = PotentialField2DPlanner::new(config);

    // Add a U-shaped obstacle that will trap the algorithm
    planner.add_polygon_obstacle(vec![
        [3.0, 3.0],
        [7.0, 3.0], // Bottom of U
        [7.0, 7.0],
        [6.0, 7.0], // Right side of U
        [6.0, 4.0],
        [4.0, 4.0], // Inside of U
        [4.0, 7.0],
        [3.0, 7.0], // Left side of U
    ]);

    // Start inside the U, goal outside
    let start = [5.0, 5.0];
    let goal = [5.0, 9.0];

    println!(
        "Finding path from [{:.1}, {:.1}] to [{:.1}, {:.1}]",
        start[0], start[1], goal[0], goal[1]
    );
    println!("Obstacle: U-shaped polygon");

    // Time the path planning
    let start_time = Instant::now();
    let path = planner.plan(start, goal)?;
    let plan_time = start_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and length {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                if i == 0 || i == path.nodes.len() - 1 || i % 5 == 0 {
                    // Print fewer points
                    println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
                }
            }
            println!("Path finding time: {:.2?}", plan_time);

            // Simple ASCII visualization
            print_ascii_visualization_polygons(
                &path.nodes,
                &[vec![
                    [3.0, 3.0],
                    [7.0, 3.0], // Bottom of U
                    [7.0, 7.0],
                    [6.0, 7.0], // Right side of U
                    [6.0, 4.0],
                    [4.0, 4.0], // Inside of U
                    [4.0, 7.0],
                    [3.0, 7.0], // Left side of U
                ]],
            );
        }
        None => {
            println!("No path found - stuck in local minimum as expected!");
            println!("Path finding time: {:.2?}", plan_time);

            // Simple ASCII visualization with just the obstacles
            print_local_minimum_visualization(
                start,
                goal,
                &[vec![
                    [3.0, 3.0],
                    [7.0, 3.0], // Bottom of U
                    [7.0, 7.0],
                    [6.0, 7.0], // Right side of U
                    [6.0, 4.0],
                    [4.0, 4.0], // Inside of U
                    [4.0, 7.0],
                    [3.0, 7.0], // Left side of U
                ]],
            );
        }
    }

    // Example 4: Escaping local minima with random noise
    println!("\n\nExample 4: Solution - Escaping local minima with random perturbations");
    println!("--------------------------------------------------------------");

    // Create a configuration with a higher step size to avoid getting stuck
    let config = PotentialConfig::new()
        .with_attractive_gain(0.8)
        .with_repulsive_gain(200.0)
        .with_influence_radius(3.0)
        .with_step_size(0.2)  // Larger step size
        .with_min_force_threshold(0.01)
        .with_max_iterations(1000);

    // Create a planner with the same U-shaped obstacle
    let mut planner = PotentialFieldPlanner::new_2d(config.clone());

    // Add the U-shaped obstacle
    planner.add_polygon_obstacle(vec![
        [3.0, 3.0],
        [7.0, 3.0], // Bottom of U
        [7.0, 7.0],
        [6.0, 7.0], // Right side of U
        [6.0, 4.0],
        [4.0, 4.0], // Inside of U
        [4.0, 7.0],
        [3.0, 7.0], // Left side of U
    ]);

    // Same start and goal as before
    let start = Array1::from_vec(vec![5.0, 5.0]);
    let goal = Array1::from_vec(vec![5.0, 9.0]);

    println!(
        "Finding path from [{:.1}, {:.1}] to [{:.1}, {:.1}]",
        start[0], start[1], goal[0], goal[1]
    );
    println!("Obstacle: Same U-shaped polygon");
    println!("Approach: Using a larger step size and more iterations");

    // Time the path planning
    let start_time = Instant::now();
    let path = planner.plan(&start, &goal)?;
    let plan_time = start_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and length {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                if i == 0 || i == path.nodes.len() - 1 || i % 5 == 0 {
                    // Print fewer points
                    println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
                }
            }
            println!("Path finding time: {:.2?}", plan_time);

            // Simple ASCII visualization
            print_ascii_visualization_polygons(
                &path.nodes,
                &[vec![
                    [3.0, 3.0],
                    [7.0, 3.0], // Bottom of U
                    [7.0, 7.0],
                    [6.0, 7.0], // Right side of U
                    [6.0, 4.0],
                    [4.0, 4.0], // Inside of U
                    [4.0, 7.0],
                    [3.0, 7.0], // Left side of U
                ]],
            );
        }
        None => println!("No path found - still stuck in local minimum!"),
    }

    Ok(())
}

/// Simple ASCII visualization of a path with circular obstacles
fn print_ascii_visualization_circles(path: &[Array1<f64>], circles: &[([f64; 2], f64)]) {
    const SIZE: usize = 20;
    let mut grid = vec![vec![' '; SIZE]; SIZE];

    // Draw the circular obstacles
    for y in 0..SIZE {
        for x in 0..SIZE {
            let px = x as f64 / (SIZE as f64) * 10.0;
            let py = y as f64 / (SIZE as f64) * 10.0;

            // Check if the point is inside any circle
            for &(center, radius) in circles {
                let dx = px - center[0];
                let dy = py - center[1];
                let dist_squared = dx * dx + dy * dy;

                if dist_squared < radius * radius {
                    grid[SIZE - 1 - y][x] = '#';
                    break;
                }
            }
        }
    }

    // Draw the path
    for i in 0..path.len() - 1 {
        let (x1, y1) = (path[i][0], path[i][1]);
        let (x2, y2) = (path[i + 1][0], path[i + 1][1]);

        // Scale to grid coordinates
        let (gx1, gy1) = (
            (x1 / 10.0 * SIZE as f64) as usize,
            (y1 / 10.0 * SIZE as f64) as usize,
        );
        let (gx2, gy2) = (
            (x2 / 10.0 * SIZE as f64) as usize,
            (y2 / 10.0 * SIZE as f64) as usize,
        );

        // Draw line segments
        let steps = ((gx2 as i32 - gx1 as i32)
            .abs()
            .max((gy2 as i32 - gy1 as i32).abs())
            + 1) as usize;

        for j in 0..steps {
            let t = if steps == 1 {
                0.0
            } else {
                j as f64 / (steps - 1) as f64
            };
            let x = (gx1 as f64 * (1.0 - t) + gx2 as f64 * t).round() as usize;
            let y = (gy1 as f64 * (1.0 - t) + gy2 as f64 * t).round() as usize;

            if x < SIZE && y < SIZE {
                grid[SIZE - 1 - y][x] = '*';
            }
        }
    }

    // Mark start and goal
    let (start_x, start_y) = (
        (path[0][0] / 10.0 * SIZE as f64) as usize,
        (path[0][1] / 10.0 * SIZE as f64) as usize,
    );
    let (goal_x, goal_y) = (
        (path.last().unwrap()[0] / 10.0 * SIZE as f64) as usize,
        (path.last().unwrap()[1] / 10.0 * SIZE as f64) as usize,
    );

    if start_x < SIZE && start_y < SIZE {
        grid[SIZE - 1 - start_y][start_x] = 'S';
    }

    if goal_x < SIZE && goal_y < SIZE {
        grid[SIZE - 1 - goal_y][goal_x] = 'G';
    }

    // Print the grid
    println!("\nASCII Visualization (S=start, G=goal, #=obstacle, *=path):");
    for row in &grid {
        println!("{}", row.iter().collect::<String>());
    }
}

/// Simple ASCII visualization of a path with polygon obstacles
fn print_ascii_visualization_polygons(path: &[Array1<f64>], polygons: &[Vec<[f64; 2]>]) {
    const SIZE: usize = 20;
    let mut grid = vec![vec![' '; SIZE]; SIZE];

    // Draw obstacles
    for y in 0..SIZE {
        for x in 0..SIZE {
            let px = x as f64 / (SIZE as f64) * 10.0;
            let py = y as f64 / (SIZE as f64) * 10.0;

            // Check if the point is inside any polygon
            for polygon in polygons {
                if point_in_polygon(&[px, py], polygon) {
                    grid[SIZE - 1 - y][x] = '#';
                    break;
                }
            }
        }
    }

    // Draw the path
    for i in 0..path.len() - 1 {
        let (x1, y1) = (path[i][0], path[i][1]);
        let (x2, y2) = (path[i + 1][0], path[i + 1][1]);

        // Scale to grid coordinates
        let (gx1, gy1) = (
            (x1 / 10.0 * SIZE as f64) as usize,
            (y1 / 10.0 * SIZE as f64) as usize,
        );
        let (gx2, gy2) = (
            (x2 / 10.0 * SIZE as f64) as usize,
            (y2 / 10.0 * SIZE as f64) as usize,
        );

        // Draw line segments
        let steps = ((gx2 as i32 - gx1 as i32)
            .abs()
            .max((gy2 as i32 - gy1 as i32).abs())
            + 1) as usize;

        for j in 0..steps {
            let t = if steps == 1 {
                0.0
            } else {
                j as f64 / (steps - 1) as f64
            };
            let x = (gx1 as f64 * (1.0 - t) + gx2 as f64 * t).round() as usize;
            let y = (gy1 as f64 * (1.0 - t) + gy2 as f64 * t).round() as usize;

            if x < SIZE && y < SIZE {
                grid[SIZE - 1 - y][x] = '*';
            }
        }
    }

    // Mark start and goal
    let (start_x, start_y) = (
        (path[0][0] / 10.0 * SIZE as f64) as usize,
        (path[0][1] / 10.0 * SIZE as f64) as usize,
    );
    let (goal_x, goal_y) = (
        (path.last().unwrap()[0] / 10.0 * SIZE as f64) as usize,
        (path.last().unwrap()[1] / 10.0 * SIZE as f64) as usize,
    );

    if start_x < SIZE && start_y < SIZE {
        grid[SIZE - 1 - start_y][start_x] = 'S';
    }

    if goal_x < SIZE && goal_y < SIZE {
        grid[SIZE - 1 - goal_y][goal_x] = 'G';
    }

    // Print the grid
    println!("\nASCII Visualization (S=start, G=goal, #=obstacle, *=path):");
    for row in &grid {
        println!("{}", row.iter().collect::<String>());
    }
}

/// Visualize a local minimum case with no path
fn print_local_minimum_visualization(start: [f64; 2], goal: [f64; 2], polygons: &[Vec<[f64; 2]>]) {
    const SIZE: usize = 20;
    let mut grid = vec![vec![' '; SIZE]; SIZE];

    // Draw obstacles
    for y in 0..SIZE {
        for x in 0..SIZE {
            let px = x as f64 / (SIZE as f64) * 10.0;
            let py = y as f64 / (SIZE as f64) * 10.0;

            // Check if the point is inside any polygon
            for polygon in polygons {
                if point_in_polygon(&[px, py], polygon) {
                    grid[SIZE - 1 - y][x] = '#';
                    break;
                }
            }
        }
    }

    // Mark start and goal
    let (start_x, start_y) = (
        (start[0] / 10.0 * SIZE as f64) as usize,
        (start[1] / 10.0 * SIZE as f64) as usize,
    );
    let (goal_x, goal_y) = (
        (goal[0] / 10.0 * SIZE as f64) as usize,
        (goal[1] / 10.0 * SIZE as f64) as usize,
    );

    if start_x < SIZE && start_y < SIZE {
        grid[SIZE - 1 - start_y][start_x] = 'S';
    }

    if goal_x < SIZE && goal_y < SIZE {
        grid[SIZE - 1 - goal_y][goal_x] = 'G';
    }

    // Print the grid
    println!("\nASCII Visualization (S=start, G=goal, #=obstacle):");
    for row in &grid {
        println!("{}", row.iter().collect::<String>());
    }
}

/// Check if a point is inside a polygon using the ray casting algorithm
fn point_in_polygon(point: &[f64; 2], polygon: &[[f64; 2]]) -> bool {
    let (x, y) = (point[0], point[1]);
    let mut inside = false;

    // Ray casting algorithm determines if the point is inside the polygon
    let n = polygon.len();
    for i in 0..n {
        let (x1, y1) = (polygon[i][0], polygon[i][1]);
        let (x2, y2) = (polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]);

        let intersects = ((y1 > y) != (y2 > y)) && (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1);

        if intersects {
            inside = !inside;
        }
    }

    inside
}
