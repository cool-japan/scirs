//! Example demonstrating the A* path planning algorithm in continuous space with polygon obstacles

use scirs2_spatial::error::SpatialResult;
use scirs2_spatial::pathplanning::ContinuousAStarPlanner;

fn main() -> SpatialResult<()> {
    println!("Continuous Space A* Path Planning Example");
    println!("========================================\n");

    // Define some polygon obstacles
    let obstacles = vec![
        // Rectangle obstacle
        vec![[2.0, 2.0], [2.0, 6.0], [3.0, 6.0], [3.0, 2.0]],
        // Triangle obstacle
        vec![[5.0, 5.0], [7.0, 7.0], [7.0, 3.0]],
        // L-shaped obstacle
        vec![
            [8.0, 8.0],
            [12.0, 8.0],
            [12.0, 9.0],
            [10.0, 9.0],
            [10.0, 12.0],
            [8.0, 12.0],
        ],
    ];

    // Create a continuous space A* planner
    let planner = ContinuousAStarPlanner::new(obstacles, 0.1, 0.1);

    // Test cases with different start and goal positions
    let test_cases = vec![
        ([1.0, 1.0], [9.0, 9.0], "around multiple obstacles"),
        ([2.5, 1.0], [2.5, 7.0], "around rectangle obstacle"),
        ([4.0, 4.0], [8.0, 6.0], "around triangle obstacle"),
        ([7.0, 10.0], [13.0, 10.0], "around L-shaped obstacle"),
    ];

    // Neighbor radius parameter determines how far each step can be
    let neighbor_radius = 0.5;

    // Run each test case
    for (i, (start, goal, description)) in test_cases.iter().enumerate() {
        println!("\nTest Case {}: Path planning {}", i + 1, description);
        println!("--------------------------------------------");
        println!("Start: {:?}", start);
        println!("Goal:  {:?}", goal);

        match planner.find_path(*start, *goal, neighbor_radius)? {
            Some(path) => {
                println!(
                    "\nFound a path with {} segments and total cost {:.2}:",
                    path.len() - 1,
                    path.cost
                );

                // Print the path waypoints
                for (j, point) in path.nodes.iter().enumerate() {
                    println!("  Waypoint {}: [{:.2}, {:.2}]", j, point[0], point[1]);
                }

                // Create a simple ASCII visualization of the path
                let grid_size = 20;
                let scale = grid_size as f64 / 15.0; // Scale to fit in grid

                let mut grid = vec![vec![' '; grid_size]; grid_size];

                // Mark obstacles
                for obstacle in &planner.obstacles {
                    for point in obstacle {
                        let x = (point[0] * scale) as usize;
                        let y = (point[1] * scale) as usize;
                        if x < grid_size && y < grid_size {
                            grid[grid_size - 1 - y][x] = 'X';
                        }
                    }
                }

                // Mark path
                for (j, point) in path.nodes.iter().enumerate() {
                    let x = (point[0] * scale) as usize;
                    let y = (point[1] * scale) as usize;
                    if x < grid_size && y < grid_size {
                        if j == 0 {
                            grid[grid_size - 1 - y][x] = 'S'; // Start
                        } else if j == path.nodes.len() - 1 {
                            grid[grid_size - 1 - y][x] = 'G'; // Goal
                        } else {
                            grid[grid_size - 1 - y][x] = '*'; // Path
                        }
                    }
                }

                // Print the grid
                println!("\nPath visualization (S=Start, G=Goal, *=Path, X=Obstacle):");
                for row in &grid {
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
    }

    Ok(())
}
