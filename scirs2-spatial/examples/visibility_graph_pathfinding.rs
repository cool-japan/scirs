use ndarray::array;
use scirs2_spatial::pathplanning::VisibilityGraphPlanner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Visibility Graph Pathfinding Example");
    println!("====================================\n");

    // Create several polygon obstacles
    let obstacles = vec![
        // Square obstacle
        array![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0],],
        // Triangle obstacle
        array![[6.0, 6.0], [8.0, 6.0], [7.0, 8.0],],
        // L-shaped obstacle
        array![
            [3.0, 7.0],
            [5.0, 7.0],
            [5.0, 8.0],
            [6.0, 8.0],
            [6.0, 10.0],
            [3.0, 10.0],
        ],
    ];

    // Create a visibility graph planner
    let mut planner = VisibilityGraphPlanner::new(obstacles);

    // Define start and goal points
    let start = [1.0, 1.0];
    let goal = [9.0, 9.0];

    println!("Finding path from {:?} to {:?}", start, goal);
    println!("Obstacles:");
    println!("- Square at (2,2) to (4,4)");
    println!("- Triangle at (6,6), (8,6), (7,8)");
    println!("- L-shape at (3,7) to (6,10)\n");

    // Find a path
    let result = planner.find_path(start, goal)?;

    match result {
        Some(path) => {
            println!(
                "Path found with {} points and cost {:.2}:",
                path.len(),
                path.cost
            );

            for (i, point) in path.nodes.iter().enumerate() {
                println!("  {}: {:?}", i, point);
            }

            // Print ASCII visualization (simple)
            println!("\nASCII Visualization (S=start, G=goal, #=obstacle, *=path):");
            print_ascii_visualization(&path.nodes, &planner);
        }
        None => {
            println!("No path found!");
        }
    }

    // Example with no path possible
    println!("\n\nExample with no path possible:");

    // Create a wall of obstacles that completely blocks the path
    let mut wall_obstacles = Vec::new();

    // Create a wall from y=0 to y=10 at x=5
    for i in 0..10 {
        wall_obstacles.push(array![
            [5.0, i as f64],
            [5.0, (i + 1) as f64],
            [6.0, (i + 1) as f64],
            [6.0, i as f64],
        ]);
    }

    let mut wall_planner = VisibilityGraphPlanner::new(wall_obstacles);

    let wall_start = [2.0, 5.0];
    let wall_goal = [9.0, 5.0];

    println!("Finding path from {:?} to {:?}", wall_start, wall_goal);
    println!("Obstacle: Wall from (5,0) to (6,10)\n");

    let wall_result = wall_planner.find_path(wall_start, wall_goal)?;

    match wall_result {
        Some(path) => {
            println!(
                "Path found with {} points and cost {:.2}:",
                path.len(),
                path.cost
            );

            for (i, point) in path.nodes.iter().enumerate() {
                println!("  {}: {:?}", i, point);
            }
        }
        None => {
            println!("No path found, as expected!");
        }
    }

    Ok(())
}

/// Simple ASCII visualization of the path and obstacles
fn print_ascii_visualization(path: &[[f64; 2]], planner: &VisibilityGraphPlanner) {
    const SIZE: usize = 20;
    let mut grid = vec![vec![' '; SIZE]; SIZE];

    // Draw obstacles
    for obstacle in &planner.obstacles {
        // Fill polygon with '#'
        let min_x = obstacle
            .column(0)
            .fold(f64::INFINITY, |a, &b| a.min(b))
            .floor() as usize;
        let max_x = obstacle
            .column(0)
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            .ceil() as usize;
        let min_y = obstacle
            .column(1)
            .fold(f64::INFINITY, |a, &b| a.min(b))
            .floor() as usize;
        let max_y = obstacle
            .column(1)
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            .ceil() as usize;

        // Mark the obstacle area
        for y in min_y..=max_y {
            if y >= SIZE {
                continue;
            }
            for x in min_x..=max_x {
                if x >= SIZE {
                    continue;
                }

                // Simple check if point is inside obstacle - for visualization only
                // This is a very rough approximation
                grid[SIZE - 1 - y][x] = '#';
            }
        }
    }

    // Draw path
    for i in 0..path.len() - 1 {
        let (x1, y1) = (path[i][0] as usize, path[i][1] as usize);
        let (x2, y2) = (path[i + 1][0] as usize, path[i + 1][1] as usize);

        // Draw line segments
        let steps = ((x2 as i32 - x1 as i32)
            .abs()
            .max((y2 as i32 - y1 as i32).abs())
            + 1) as usize;

        for j in 0..steps {
            let t = if steps == 1 {
                0.0
            } else {
                j as f64 / (steps - 1) as f64
            };
            let x = (x1 as f64 * (1.0 - t) + x2 as f64 * t).round() as usize;
            let y = (y1 as f64 * (1.0 - t) + y2 as f64 * t).round() as usize;

            if x < SIZE && y < SIZE {
                grid[SIZE - 1 - y][x] = '*';
            }
        }
    }

    // Mark start and goal
    let (start_x, start_y) = (path[0][0] as usize, path[0][1] as usize);
    let (goal_x, goal_y) = (
        path.last().unwrap()[0] as usize,
        path.last().unwrap()[1] as usize,
    );

    if start_x < SIZE && start_y < SIZE {
        grid[SIZE - 1 - start_y][start_x] = 'S';
    }

    if goal_x < SIZE && goal_y < SIZE {
        grid[SIZE - 1 - goal_y][goal_x] = 'G';
    }

    // Print the grid
    for row in &grid {
        println!("{}", row.iter().collect::<String>());
    }
}
