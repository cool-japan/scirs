//! Example demonstrating A* search algorithm for pathfinding

#![allow(clippy::needless_range_loop)]

use scirs2_spatial::error::SpatialResult;
use scirs2_spatial::pathplanning::{ContinuousAStarPlanner, GridAStarPlanner};

fn main() -> SpatialResult<()> {
    println!("A* Pathfinding Examples");
    println!("======================\n");

    // Example 1: Grid-based A* with obstacles
    grid_astar_example()?;

    // Example 2: Grid-based A* with diagonal movement
    grid_astar_diagonal_example()?;

    // Example 3: Continuous space A* with polygon obstacles
    continuous_astar_example()?;

    Ok(())
}

fn grid_astar_example() -> SpatialResult<()> {
    println!("Example 1: Grid-based A* with obstacles");
    println!("---------------------------------------");

    // Create a 10x10 grid with some obstacles
    let mut grid = vec![vec![false; 10]; 10];

    // Add obstacles (walls)
    for i in 2..8 {
        grid[i][3] = true;
    }
    for i in 3..9 {
        grid[2][i] = true;
    }
    for i in 5..9 {
        grid[5][i] = true;
    }
    for i in 2..6 {
        grid[7][i] = true;
    }

    // Print the grid
    println!("Grid (X = obstacle, . = free):");
    for row in &grid {
        for &cell in row {
            print!("{}", if cell { "X " } else { ". " });
        }
        println!();
    }

    // Create A* planner
    let planner = GridAStarPlanner::new(grid, false);
    let start = [1, 1];
    let goal = [8, 8];

    println!("\nFinding path from {:?} to {:?}...", start, goal);

    // Find a path
    match planner.find_path(start, goal)? {
        Some(path) => {
            println!(
                "Path found with {} steps and cost {:.2}:",
                path.len() - 1,
                path.cost
            );
            for (i, pos) in path.nodes.iter().enumerate() {
                println!("  Step {}: {:?}", i, pos);
            }

            // Visualize the path on the grid
            let mut grid_with_path = vec![vec![' '; 10]; 10];
            for i in 0..10 {
                for j in 0..10 {
                    if planner.grid[i][j] {
                        grid_with_path[i][j] = 'X'; // Obstacle
                    } else {
                        grid_with_path[i][j] = '.'; // Free space
                    }
                }
            }

            // Mark the path
            for pos in &path.nodes {
                grid_with_path[pos[0] as usize][pos[1] as usize] = '*';
            }
            // Mark start and goal
            grid_with_path[start[0] as usize][start[1] as usize] = 'S';
            grid_with_path[goal[0] as usize][goal[1] as usize] = 'G';

            println!("\nPath visualization:");
            for row in &grid_with_path {
                for &cell in row {
                    print!("{} ", cell);
                }
                println!();
            }
        }
        None => {
            println!("No path found!");
        }
    }

    println!();
    Ok(())
}

fn grid_astar_diagonal_example() -> SpatialResult<()> {
    println!("Example 2: Grid-based A* with diagonal movement");
    println!("----------------------------------------------");

    // Create a 10x10 grid with some obstacles
    let mut grid = vec![vec![false; 10]; 10];

    // Add obstacles (walls)
    for i in 2..8 {
        grid[i][3] = true;
    }
    for i in 3..9 {
        grid[2][i] = true;
    }
    for i in 5..9 {
        grid[5][i] = true;
    }
    for i in 2..6 {
        grid[7][i] = true;
    }

    // Create A* planner with diagonal movement allowed
    let planner = GridAStarPlanner::new(grid, true);
    let start = [1, 1];
    let goal = [8, 8];

    println!(
        "\nFinding path from {:?} to {:?} with diagonal movement...",
        start, goal
    );

    // Find a path
    match planner.find_path(start, goal)? {
        Some(path) => {
            println!(
                "Path found with {} steps and cost {:.2}:",
                path.len() - 1,
                path.cost
            );
            for (i, pos) in path.nodes.iter().enumerate() {
                println!("  Step {}: {:?}", i, pos);
            }

            // Visualize the path on the grid
            let mut grid_with_path = vec![vec![' '; 10]; 10];
            for i in 0..10 {
                for j in 0..10 {
                    if planner.grid[i][j] {
                        grid_with_path[i][j] = 'X'; // Obstacle
                    } else {
                        grid_with_path[i][j] = '.'; // Free space
                    }
                }
            }

            // Mark the path
            for pos in &path.nodes {
                grid_with_path[pos[0] as usize][pos[1] as usize] = '*';
            }
            // Mark start and goal
            grid_with_path[start[0] as usize][start[1] as usize] = 'S';
            grid_with_path[goal[0] as usize][goal[1] as usize] = 'G';

            println!("\nPath visualization:");
            for row in &grid_with_path {
                for &cell in row {
                    print!("{} ", cell);
                }
                println!();
            }
        }
        None => {
            println!("No path found!");
        }
    }

    println!();
    Ok(())
}

fn continuous_astar_example() -> SpatialResult<()> {
    println!("Example 3: Continuous space A* with polygon obstacles");
    println!("-------------------------------------------------");

    // Define some polygon obstacles
    let obstacles = vec![
        // Rectangle obstacle
        vec![[2.0, 2.0], [2.0, 6.0], [3.0, 6.0], [3.0, 2.0]],
        // Triangle obstacle
        vec![[5.0, 5.0], [7.0, 7.0], [7.0, 3.0]],
        // Polygon obstacle
        vec![[8.0, 2.0], [9.0, 3.0], [10.0, 2.0], [9.0, 1.0]],
    ];

    // Create continuous space A* planner
    let planner = ContinuousAStarPlanner::new(obstacles, 0.1, 0.1);
    let start = [1.0, 1.0];
    let goal = [9.0, 9.0];
    let neighbor_radius = 1.0;

    println!(
        "\nFinding path from {:?} to {:?} in continuous space...",
        start, goal
    );

    // Find a path
    match planner.find_path(start, goal, neighbor_radius)? {
        Some(path) => {
            println!(
                "Path found with {} segments and cost {:.2}:",
                path.len() - 1,
                path.cost
            );
            for (i, pos) in path.nodes.iter().enumerate() {
                println!("  Point {}: [{:.2}, {:.2}]", i, pos[0], pos[1]);
            }

            // We can't easily visualize the continuous path in the terminal,
            // but we could generate a simple ASCII art representation
            println!("\nNote: In a real application, you would visualize this path graphically.");
        }
        None => {
            println!("No path found!");
        }
    }

    println!();
    Ok(())
}
