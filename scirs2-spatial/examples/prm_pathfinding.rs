use ndarray::Array1;
use scirs2_spatial::pathplanning::{PRM2DPlanner, PRMConfig, PRMPlanner};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Probabilistic Roadmap (PRM) Pathfinding Examples");
    println!("===============================================\n");

    // Example 1: Simple 2D environment with circle obstacle
    println!("Example 1: Simple 2D environment with a circle obstacle");
    println!("------------------------------------------------------");

    // Create configuration for the PRM planner
    let config = PRMConfig::new()
        .with_num_samples(500)
        .with_connection_radius(2.0)
        .with_max_connections(10)
        .with_seed(42);

    // Define bounds of the configuration space
    let lower_bounds = Array1::from_vec(vec![0.0, 0.0]);
    let upper_bounds = Array1::from_vec(vec![10.0, 10.0]);

    // Create a PRM planner with a circle obstacle at (5,5) with radius 2
    let mut planner = PRMPlanner::new(config, lower_bounds, upper_bounds);

    planner.set_collision_checker(Box::new(|p: &Array1<f64>| {
        let dx = p[0] - 5.0;
        let dy = p[1] - 5.0;
        let dist_squared = dx * dx + dy * dy;
        dist_squared < 4.0 // Inside the circle is in collision
    }));

    // Time the roadmap construction
    let start_time = Instant::now();
    planner.build_roadmap()?;
    let build_time = start_time.elapsed();

    println!("Roadmap built in {:.2?}", build_time);

    // Find path from start to goal
    let start = Array1::from_vec(vec![1.0, 1.0]);
    let goal = Array1::from_vec(vec![9.0, 9.0]);

    let path_time = Instant::now();
    let path = planner.find_path(&start, &goal)?;
    let path_find_time = path_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and cost {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
            }
            println!("Path finding time: {:.2?}", path_find_time);

            // Simple ASCII visualization
            print_ascii_visualization_circle(&path.nodes, 5.0, 5.0, 2.0);
        }
        None => println!("No path found!"),
    }

    // Example 2: 2D environment with polygon obstacles
    println!("\n\nExample 2: 2D environment with polygon obstacles");
    println!("------------------------------------------------");

    // Create multiple polygon obstacles
    let obstacles = vec![
        // Rectangle in the middle
        vec![[4.0, 3.0], [6.0, 3.0], [6.0, 7.0], [4.0, 7.0]],
        // Triangle at the top
        vec![[1.0, 8.0], [3.0, 8.0], [2.0, 9.0]],
        // L-shape at the bottom right
        vec![
            [7.0, 1.0],
            [9.0, 1.0],
            [9.0, 4.0],
            [8.0, 4.0],
            [8.0, 2.0],
            [7.0, 2.0],
        ],
    ];

    let config = PRMConfig::new()
        .with_num_samples(1000)
        .with_connection_radius(2.0)
        .with_max_connections(20)
        .with_seed(42);

    // Create a 2D PRM planner with polygon obstacles
    let mut planner = PRM2DPlanner::new(config, obstacles, (0.0, 10.0), (0.0, 10.0));

    // Time the roadmap construction
    let start_time = Instant::now();
    planner.build_roadmap()?;
    let build_time = start_time.elapsed();

    println!("Roadmap built in {:.2?}", build_time);

    // Find path from start to goal
    let start = [1.0, 5.0];
    let goal = [9.0, 5.0];

    println!(
        "Finding path from [{:.1}, {:.1}] to [{:.1}, {:.1}]",
        start[0], start[1], goal[0], goal[1]
    );

    let path_time = Instant::now();
    let path = planner.find_path(start, goal)?;
    let path_find_time = path_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and cost {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
            }
            println!("Path finding time: {:.2?}", path_find_time);

            // Simple ASCII visualization
            print_ascii_visualization_polygons(&path.nodes, planner.obstacles());
        }
        None => println!("No path found!"),
    }

    // Example 3: Narrow passages challenge
    println!("\n\nExample 3: Narrow passage challenge");
    println!("---------------------------------");

    // Create obstacles that form a narrow passage in the middle
    let obstacles = vec![
        // Top wall
        vec![[0.0, 6.0], [4.0, 6.0], [4.0, 10.0], [0.0, 10.0]],
        // Bottom wall
        vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        // Narrow passage
        // Top wall
        vec![[6.0, 6.0], [10.0, 6.0], [10.0, 10.0], [6.0, 10.0]],
        // Bottom wall
        vec![[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]],
    ];

    // Create a more dense PRM for the narrow passage
    let config = PRMConfig::new()
        .with_num_samples(2000)
        .with_connection_radius(1.0)
        .with_max_connections(30)
        .with_seed(42);

    let mut planner = PRM2DPlanner::new(config, obstacles, (0.0, 10.0), (0.0, 10.0));

    // Time the roadmap construction
    let start_time = Instant::now();
    planner.build_roadmap()?;
    let build_time = start_time.elapsed();

    println!("Roadmap built in {:.2?}", build_time);

    // Find path from left to right through the narrow passage
    let start = [2.0, 5.0];
    let goal = [8.0, 5.0];

    println!(
        "Finding path from [{:.1}, {:.1}] to [{:.1}, {:.1}]",
        start[0], start[1], goal[0], goal[1]
    );

    let path_time = Instant::now();
    let path = planner.find_path(start, goal)?;
    let path_find_time = path_time.elapsed();

    match &path {
        Some(path) => {
            println!(
                "Path found with {} points and cost {:.2}:",
                path.nodes.len(),
                path.cost
            );
            for (i, point) in path.nodes.iter().enumerate() {
                println!("  {}: [{:.2}, {:.2}]", i, point[0], point[1]);
            }
            println!("Path finding time: {:.2?}", path_find_time);

            // Simple ASCII visualization
            print_ascii_visualization_polygons(&path.nodes, planner.obstacles());
        }
        None => println!("No path found through the narrow passage!"),
    }

    Ok(())
}

/// Simple ASCII visualization of a path with a circle obstacle
fn print_ascii_visualization_circle(
    path: &[Array1<f64>],
    circle_x: f64,
    circle_y: f64,
    circle_radius: f64,
) {
    const SIZE: usize = 20;
    let mut grid = vec![vec![' '; SIZE]; SIZE];

    // Draw the circle obstacle
    for y in 0..SIZE {
        for x in 0..SIZE {
            let px = x as f64 / (SIZE as f64) * 10.0;
            let py = y as f64 / (SIZE as f64) * 10.0;

            // Check if the point is inside the circle
            let dx = px - circle_x;
            let dy = py - circle_y;
            let dist_squared = dx * dx + dy * dy;

            if dist_squared < circle_radius * circle_radius {
                grid[SIZE - 1 - y][x] = '#';
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
fn print_ascii_visualization_polygons(path: &[Array1<f64>], obstacles: &[Vec<[f64; 2]>]) {
    const SIZE: usize = 20;
    let mut grid = vec![vec![' '; SIZE]; SIZE];

    // Draw obstacles
    for y in 0..SIZE {
        for x in 0..SIZE {
            let px = x as f64 / (SIZE as f64) * 10.0;
            let py = y as f64 / (SIZE as f64) * 10.0;

            // Check if the point is inside any obstacle
            for obstacle in obstacles {
                if point_in_polygon(&[px, py], obstacle) {
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
