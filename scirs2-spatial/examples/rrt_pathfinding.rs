//! Example demonstrating the RRT and RRT* path planning algorithms

use scirs2_spatial::error::SpatialResult;
use scirs2_spatial::pathplanning::{RRT2DPlanner, RRTConfig};

fn main() -> SpatialResult<()> {
    println!("RRT Path Planning Examples");
    println!("========================\n");

    // Example 1: Basic RRT in an environment with obstacles
    basic_rrt_example()?;

    // Example 2: RRT* for optimized paths
    rrt_star_example()?;

    // Example 3: Bidirectional RRT for faster convergence
    bidirectional_rrt_example()?;

    Ok(())
}

fn basic_rrt_example() -> SpatialResult<()> {
    println!("Example 1: Basic RRT with Obstacles");
    println!("----------------------------------");

    // Create RRT configuration
    let config = RRTConfig {
        max_iterations: 2000,
        step_size: 0.3,
        goal_bias: 0.1,
        seed: Some(42), // For reproducibility
        use_rrt_star: false,
        neighborhood_radius: None,
        bidirectional: false,
    };

    // Define obstacles (polygons)
    let obstacles = vec![
        // Vertical wall with a gap
        vec![[4.0, 0.0], [5.0, 0.0], [5.0, 3.0], [4.0, 3.0]],
        vec![[4.0, 5.0], [5.0, 5.0], [5.0, 10.0], [4.0, 10.0]],
        // Circular-like obstacle (approximated with a polygon)
        vec![
            [7.0, 4.0],
            [7.5, 3.5],
            [8.0, 3.2],
            [8.5, 3.5],
            [9.0, 4.0],
            [8.5, 4.5],
            [8.0, 4.8],
            [7.5, 4.5],
        ],
    ];

    // Create a 2D RRT planner
    let mut planner = RRT2DPlanner::new(
        config,
        obstacles.clone(),
        [0.0, 0.0],   // Min bounds
        [10.0, 10.0], // Max bounds
        0.1,          // Collision step size
    )?;

    // Define start and goal positions
    let start = [1.0, 5.0];
    let goal = [9.0, 5.0];
    let goal_threshold = 0.5;

    println!("Finding path from {:?} to {:?}...", start, goal);

    // Find a path
    let path = planner.find_path(start, goal, goal_threshold)?;

    if let Some(path) = path {
        println!(
            "Path found with {} segments and cost {:.2}:",
            path.len() - 1,
            path.cost
        );
        for (i, pos) in path.nodes.iter().enumerate() {
            println!("  Point {}: [{:.2}, {:.2}]", i, pos[0], pos[1]);
        }

        // Visualize the path
        visualize_path(&path.nodes, &obstacles, [0.0, 0.0], [10.0, 10.0]);
    } else {
        println!("No path found!");
    }

    println!();
    Ok(())
}

fn rrt_star_example() -> SpatialResult<()> {
    println!("Example 2: RRT* for Optimized Paths");
    println!("---------------------------------");

    // Create RRT* configuration
    let config = RRTConfig {
        max_iterations: 3000,
        step_size: 0.3,
        goal_bias: 0.05,
        seed: Some(42), // For reproducibility
        use_rrt_star: true,
        neighborhood_radius: Some(1.0),
        bidirectional: false,
    };

    // Define obstacles (polygons)
    let obstacles = vec![
        // Vertical wall with a gap
        vec![[4.0, 0.0], [5.0, 0.0], [5.0, 3.0], [4.0, 3.0]],
        vec![[4.0, 5.0], [5.0, 5.0], [5.0, 10.0], [4.0, 10.0]],
        // Circular-like obstacle (approximated with a polygon)
        vec![
            [7.0, 4.0],
            [7.5, 3.5],
            [8.0, 3.2],
            [8.5, 3.5],
            [9.0, 4.0],
            [8.5, 4.5],
            [8.0, 4.8],
            [7.5, 4.5],
        ],
    ];

    // Create a 2D RRT* planner
    let mut planner = RRT2DPlanner::new(
        config,
        obstacles.clone(),
        [0.0, 0.0],   // Min bounds
        [10.0, 10.0], // Max bounds
        0.1,          // Collision step size
    )?;

    // Define start and goal positions
    let start = [1.0, 5.0];
    let goal = [9.0, 5.0];
    let goal_threshold = 0.5;

    println!(
        "Finding optimal path from {:?} to {:?} using RRT*...",
        start, goal
    );

    // Find a path
    let path = planner.find_path(start, goal, goal_threshold)?;

    if let Some(path) = path {
        println!(
            "Path found with {} segments and cost {:.2}:",
            path.len() - 1,
            path.cost
        );
        for (i, pos) in path.nodes.iter().enumerate() {
            println!("  Point {}: [{:.2}, {:.2}]", i, pos[0], pos[1]);
        }

        // Visualize the path
        visualize_path(&path.nodes, &obstacles, [0.0, 0.0], [10.0, 10.0]);
    } else {
        println!("No path found!");
    }

    println!();
    Ok(())
}

fn bidirectional_rrt_example() -> SpatialResult<()> {
    println!("Example 3: Bidirectional RRT for Faster Convergence");
    println!("-----------------------------------------------");

    // Create bidirectional RRT configuration
    let config = RRTConfig {
        max_iterations: 2000,
        step_size: 0.3,
        goal_bias: 0.0, // No goal bias needed for bidirectional RRT
        seed: Some(42), // For reproducibility
        use_rrt_star: false,
        neighborhood_radius: None,
        bidirectional: true,
    };

    // Define obstacles (polygons)
    let obstacles = vec![
        // Complex environment with multiple obstacles
        // Maze-like structure
        vec![[2.0, 0.0], [3.0, 0.0], [3.0, 7.0], [2.0, 7.0]],
        vec![[2.0, 8.0], [3.0, 8.0], [3.0, 10.0], [2.0, 10.0]],
        vec![[3.0, 3.0], [7.0, 3.0], [7.0, 4.0], [3.0, 4.0]],
        vec![[5.0, 6.0], [10.0, 6.0], [10.0, 7.0], [5.0, 7.0]],
        vec![[7.0, 0.0], [8.0, 0.0], [8.0, 3.0], [7.0, 3.0]],
    ];

    // Create a 2D bidirectional RRT planner
    let mut planner = RRT2DPlanner::new(
        config,
        obstacles.clone(),
        [0.0, 0.0],   // Min bounds
        [10.0, 10.0], // Max bounds
        0.1,          // Collision step size
    )?;

    // Define start and goal positions
    let start = [1.0, 1.0];
    let goal = [9.0, 9.0];
    let goal_threshold = 0.5;

    println!(
        "Finding path from {:?} to {:?} using bidirectional RRT...",
        start, goal
    );

    // Find a path
    let path = planner.find_path(start, goal, goal_threshold)?;

    if let Some(path) = path {
        println!(
            "Path found with {} segments and cost {:.2}:",
            path.len() - 1,
            path.cost
        );
        for (i, pos) in path.nodes.iter().enumerate() {
            println!("  Point {}: [{:.2}, {:.2}]", i, pos[0], pos[1]);
        }

        // Visualize the path
        visualize_path(&path.nodes, &obstacles, [0.0, 0.0], [10.0, 10.0]);
    } else {
        println!("No path found!");
    }

    println!();
    Ok(())
}

// Helper function to visualize the path in a simple ASCII grid
fn visualize_path(
    path: &[[f64; 2]],
    obstacles: &[Vec<[f64; 2]>],
    min_bounds: [f64; 2],
    max_bounds: [f64; 2],
) {
    let grid_size = 25;
    let scale_x = grid_size as f64 / (max_bounds[0] - min_bounds[0]);
    let scale_y = grid_size as f64 / (max_bounds[1] - min_bounds[1]);

    // Create an empty grid
    let mut grid = vec![vec![' '; grid_size]; grid_size];

    // Fill in obstacle cells
    for obstacle in obstacles {
        // For simplicity, we just fill cells that are inside the polygon
        for y in 0..grid_size {
            for x in 0..grid_size {
                // Convert grid coordinates to world coordinates
                let world_x = (x as f64) / scale_x + min_bounds[0];
                let world_y = (y as f64) / scale_y + min_bounds[1];

                // Check if this point is inside any obstacle
                if is_point_in_polygon(&[world_x, world_y], obstacle) {
                    grid[grid_size - 1 - y][x] = '#';
                }
            }
        }
    }

    // Draw the path
    for (i, point) in path.iter().enumerate() {
        // Convert world coordinates to grid indices
        let grid_x = ((point[0] - min_bounds[0]) * scale_x) as usize;
        let grid_y = ((point[1] - min_bounds[1]) * scale_y) as usize;

        // Ensure coordinates are within grid bounds
        if grid_x < grid_size && grid_y < grid_size {
            let y = grid_size - 1 - grid_y;
            if i == 0 {
                grid[y][grid_x] = 'S'; // Start
            } else if i == path.len() - 1 {
                grid[y][grid_x] = 'G'; // Goal
            } else {
                grid[y][grid_x] = '*'; // Path point
            }
        }
    }

    // Print the visualization
    println!("\nPath visualization (S=Start, G=Goal, *=Path point, #=Obstacle):");
    for row in &grid {
        for &cell in row {
            print!("{} ", cell);
        }
        println!();
    }
}

// Helper function for point-in-polygon test
fn is_point_in_polygon(point: &[f64; 2], polygon: &[[f64; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = polygon.len() - 1;

    for i in 0..polygon.len() {
        let xi = polygon[i][0];
        let yi = polygon[i][1];
        let xj = polygon[j][0];
        let yj = polygon[j][1];

        let intersect = ((yi > point[1]) != (yj > point[1]))
            && (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi);

        if intersect {
            inside = !inside;
        }

        j = i;
    }

    inside
}
