//! Broad-phase collision detection algorithms
//!
//! This module provides efficient algorithms for quickly filtering out pairs of objects
//! that cannot possibly be colliding, reducing the number of detailed collision tests needed.

use super::shapes::{Box2D, Box3D, Circle, Sphere};

/// A simple AABB (Axis-Aligned Bounding Box) for spatial partitioning
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    /// Minimum corner of the bounding box
    pub min: [f64; 3],
    /// Maximum corner of the bounding box
    pub max: [f64; 3],
}

impl AABB {
    /// Creates a new AABB with the given minimum and maximum corners
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        AABB { min, max }
    }

    /// Creates an AABB from a 3D box
    pub fn from_box3d(box3d: &Box3D) -> Self {
        AABB {
            min: box3d.min,
            max: box3d.max,
        }
    }

    /// Creates an AABB from a sphere
    pub fn from_sphere(sphere: &Sphere) -> Self {
        let radius = sphere.radius;
        AABB {
            min: [
                sphere.center[0] - radius,
                sphere.center[1] - radius,
                sphere.center[2] - radius,
            ],
            max: [
                sphere.center[0] + radius,
                sphere.center[1] + radius,
                sphere.center[2] + radius,
            ],
        }
    }

    /// Creates an AABB from a 2D box, setting z-coordinates to 0
    pub fn from_box2d(box2d: &Box2D) -> Self {
        AABB {
            min: [box2d.min[0], box2d.min[1], 0.0],
            max: [box2d.max[0], box2d.max[1], 0.0],
        }
    }

    /// Creates an AABB from a circle, setting z-coordinates to 0
    pub fn from_circle(circle: &Circle) -> Self {
        let radius = circle.radius;
        AABB {
            min: [circle.center[0] - radius, circle.center[1] - radius, 0.0],
            max: [circle.center[0] + radius, circle.center[1] + radius, 0.0],
        }
    }

    /// Tests if this AABB intersects with another AABB
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }
}

/// A simple grid-based spatial partitioning structure for 2D space
pub struct SpatialGrid2D {
    /// Cell size (width and height)
    cell_size: f64,
    /// Total width of the grid
    width: f64,
    /// Total height of the grid
    height: f64,
    /// Number of cells in the x-direction
    cells_x: usize,
    /// Number of cells in the y-direction
    cells_y: usize,
}

impl SpatialGrid2D {
    /// Creates a new 2D spatial grid with the given dimensions and cell size
    pub fn new(width: f64, height: f64, cell_size: f64) -> Self {
        let cells_x = (width / cell_size).ceil() as usize;
        let cells_y = (height / cell_size).ceil() as usize;
        SpatialGrid2D {
            cell_size,
            width,
            height,
            cells_x,
            cells_y,
        }
    }

    /// Returns the cell indices for a given 2D position
    pub fn get_cell_indices(&self, pos: &[f64; 2]) -> Option<(usize, usize)> {
        if pos[0] < 0.0 || pos[0] >= self.width || pos[1] < 0.0 || pos[1] >= self.height {
            return None;
        }

        let x = (pos[0] / self.cell_size) as usize;
        let y = (pos[1] / self.cell_size) as usize;

        // Ensure indices are within bounds
        if x >= self.cells_x || y >= self.cells_y {
            return None;
        }

        Some((x, y))
    }

    /// Returns the cell indices for a 2D circle, potentially spanning multiple cells
    pub fn get_circle_cell_indices(&self, circle: &Circle) -> Vec<(usize, usize)> {
        let min_x = (circle.center[0] - circle.radius).max(0.0);
        let min_y = (circle.center[1] - circle.radius).max(0.0);
        let max_x = (circle.center[0] + circle.radius).min(self.width);
        let max_y = (circle.center[1] + circle.radius).min(self.height);

        let min_cell_x = (min_x / self.cell_size) as usize;
        let min_cell_y = (min_y / self.cell_size) as usize;
        let max_cell_x = (max_x / self.cell_size) as usize;
        let max_cell_y = (max_y / self.cell_size) as usize;

        let mut cells = Vec::new();
        for y in min_cell_y..=max_cell_y {
            for x in min_cell_x..=max_cell_x {
                if x < self.cells_x && y < self.cells_y {
                    cells.push((x, y));
                }
            }
        }

        cells
    }

    /// Returns the cell indices for a 2D box, potentially spanning multiple cells
    pub fn get_box_cell_indices(&self, box2d: &Box2D) -> Vec<(usize, usize)> {
        let min_x = box2d.min[0].max(0.0);
        let min_y = box2d.min[1].max(0.0);
        let max_x = box2d.max[0].min(self.width);
        let max_y = box2d.max[1].min(self.height);

        let min_cell_x = (min_x / self.cell_size) as usize;
        let min_cell_y = (min_y / self.cell_size) as usize;
        let max_cell_x = (max_x / self.cell_size) as usize;
        let max_cell_y = (max_y / self.cell_size) as usize;

        let mut cells = Vec::new();
        for y in min_cell_y..=max_cell_y {
            for x in min_cell_x..=max_cell_x {
                if x < self.cells_x && y < self.cells_y {
                    cells.push((x, y));
                }
            }
        }

        cells
    }
}

/// A simple grid-based spatial partitioning structure for 3D space
pub struct SpatialGrid3D {
    /// Cell size (width, height, and depth)
    cell_size: f64,
    /// Total width of the grid (x-dimension)
    width: f64,
    /// Total height of the grid (y-dimension)
    height: f64,
    /// Total depth of the grid (z-dimension)
    depth: f64,
    /// Number of cells in the x-direction
    cells_x: usize,
    /// Number of cells in the y-direction
    cells_y: usize,
    /// Number of cells in the z-direction
    cells_z: usize,
}

impl SpatialGrid3D {
    /// Creates a new 3D spatial grid with the given dimensions and cell size
    pub fn new(width: f64, height: f64, depth: f64, cell_size: f64) -> Self {
        let cells_x = (width / cell_size).ceil() as usize;
        let cells_y = (height / cell_size).ceil() as usize;
        let cells_z = (depth / cell_size).ceil() as usize;
        SpatialGrid3D {
            cell_size,
            width,
            height,
            depth,
            cells_x,
            cells_y,
            cells_z,
        }
    }

    /// Returns the cell indices for a given 3D position
    pub fn get_cell_indices(&self, pos: &[f64; 3]) -> Option<(usize, usize, usize)> {
        if pos[0] < 0.0
            || pos[0] >= self.width
            || pos[1] < 0.0
            || pos[1] >= self.height
            || pos[2] < 0.0
            || pos[2] >= self.depth
        {
            return None;
        }

        let x = (pos[0] / self.cell_size) as usize;
        let y = (pos[1] / self.cell_size) as usize;
        let z = (pos[2] / self.cell_size) as usize;

        // Ensure indices are within bounds
        if x >= self.cells_x || y >= self.cells_y || z >= self.cells_z {
            return None;
        }

        Some((x, y, z))
    }

    /// Returns the cell indices for a 3D sphere, potentially spanning multiple cells
    pub fn get_sphere_cell_indices(&self, sphere: &Sphere) -> Vec<(usize, usize, usize)> {
        let min_x = (sphere.center[0] - sphere.radius).max(0.0);
        let min_y = (sphere.center[1] - sphere.radius).max(0.0);
        let min_z = (sphere.center[2] - sphere.radius).max(0.0);
        let max_x = (sphere.center[0] + sphere.radius).min(self.width);
        let max_y = (sphere.center[1] + sphere.radius).min(self.height);
        let max_z = (sphere.center[2] + sphere.radius).min(self.depth);

        let min_cell_x = (min_x / self.cell_size) as usize;
        let min_cell_y = (min_y / self.cell_size) as usize;
        let min_cell_z = (min_z / self.cell_size) as usize;
        let max_cell_x = (max_x / self.cell_size) as usize;
        let max_cell_y = (max_y / self.cell_size) as usize;
        let max_cell_z = (max_z / self.cell_size) as usize;

        let mut cells = Vec::new();
        for z in min_cell_z..=max_cell_z {
            for y in min_cell_y..=max_cell_y {
                for x in min_cell_x..=max_cell_x {
                    if x < self.cells_x && y < self.cells_y && z < self.cells_z {
                        cells.push((x, y, z));
                    }
                }
            }
        }

        cells
    }

    /// Returns the cell indices for a 3D box, potentially spanning multiple cells
    pub fn get_box_cell_indices(&self, box3d: &Box3D) -> Vec<(usize, usize, usize)> {
        let min_x = box3d.min[0].max(0.0);
        let min_y = box3d.min[1].max(0.0);
        let min_z = box3d.min[2].max(0.0);
        let max_x = box3d.max[0].min(self.width);
        let max_y = box3d.max[1].min(self.height);
        let max_z = box3d.max[2].min(self.depth);

        let min_cell_x = (min_x / self.cell_size) as usize;
        let min_cell_y = (min_y / self.cell_size) as usize;
        let min_cell_z = (min_z / self.cell_size) as usize;
        let max_cell_x = (max_x / self.cell_size) as usize;
        let max_cell_y = (max_y / self.cell_size) as usize;
        let max_cell_z = (max_z / self.cell_size) as usize;

        let mut cells = Vec::new();
        for z in min_cell_z..=max_cell_z {
            for y in min_cell_y..=max_cell_y {
                for x in min_cell_x..=max_cell_x {
                    if x < self.cells_x && y < self.cells_y && z < self.cells_z {
                        cells.push((x, y, z));
                    }
                }
            }
        }

        cells
    }
}

/// A sweep and prune algorithm for broad-phase collision detection in 1D
pub struct SweepAndPrune1D {
    /// The axis to use for sorting objects (0 = x, 1 = y, 2 = z)
    axis: usize,
}

impl SweepAndPrune1D {
    /// Creates a new sweep and prune algorithm for the given axis
    pub fn new(axis: usize) -> Self {
        SweepAndPrune1D { axis }
    }

    /// Checks if two AABBs could be colliding along the chosen axis
    pub fn may_collide(&self, aabb1: &AABB, aabb2: &AABB) -> bool {
        aabb1.min[self.axis] <= aabb2.max[self.axis] && aabb1.max[self.axis] >= aabb2.min[self.axis]
    }

    /// Gets the starting point of an AABB along the chosen axis
    pub fn get_start(&self, aabb: &AABB) -> f64 {
        aabb.min[self.axis]
    }

    /// Gets the ending point of an AABB along the chosen axis
    pub fn get_end(&self, aabb: &AABB) -> f64 {
        aabb.max[self.axis]
    }
}

/// The standard swept-prune algorithm for broad-phase collision detection
pub struct SweepAndPrune {
    /// Sweep and prune for the x-axis
    x_axis: SweepAndPrune1D,
    /// Sweep and prune for the y-axis
    y_axis: SweepAndPrune1D,
    /// Sweep and prune for the z-axis
    z_axis: SweepAndPrune1D,
}

impl Default for SweepAndPrune {
    fn default() -> Self {
        Self::new()
    }
}

impl SweepAndPrune {
    /// Creates a new sweep and prune algorithm
    pub fn new() -> Self {
        SweepAndPrune {
            x_axis: SweepAndPrune1D::new(0),
            y_axis: SweepAndPrune1D::new(1),
            z_axis: SweepAndPrune1D::new(2),
        }
    }

    /// Checks if two AABBs could potentially be colliding
    pub fn may_collide(&self, aabb1: &AABB, aabb2: &AABB) -> bool {
        self.x_axis.may_collide(aabb1, aabb2)
            && self.y_axis.may_collide(aabb1, aabb2)
            && self.z_axis.may_collide(aabb1, aabb2)
    }
}
