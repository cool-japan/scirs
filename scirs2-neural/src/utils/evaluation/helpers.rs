//! Helper functions for visualization components
//!
//! This module provides utility functions for creating ASCII visualizations,
//! including line drawing algorithms and formatting helpers for evaluation tools.

/// Draw a line between two points using a Bresenham-like algorithm
///
/// This function draws a line between two points and returns the coordinates of all
/// points along the line. It uses a modified Bresenham algorithm, which is an
/// efficient line drawing algorithm that uses only integer arithmetic.
///
/// # Arguments
///
/// * `x1` - X coordinate of the starting point
/// * `y1` - Y coordinate of the starting point
/// * `x2` - X coordinate of the ending point
/// * `y2` - Y coordinate of the ending point
/// * `max_width` - Maximum width of the drawing area (optional)
/// * `max_height` - Maximum height of the drawing area (optional)
///
/// # Returns
///
/// A vector of (x, y) coordinates representing points along the line
pub(crate) fn draw_line_with_coords(
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    max_width: Option<usize>,
    max_height: Option<usize>,
) -> Vec<(usize, usize)> {
    let mut coords = Vec::new();

    // Simple Bresenham-like algorithm
    let dx = (x2 as isize - x1 as isize).abs();
    let dy = (y2 as isize - y1 as isize).abs();

    let sx = if x1 < x2 { 1isize } else { -1isize };
    let sy = if y1 < y2 { 1isize } else { -1isize };

    let mut err = dx - dy;
    let mut x = x1 as isize;
    let mut y = y1 as isize;

    while x != x2 as isize || y != y2 as isize {
        // Check bounds if max dimensions are provided
        let in_bounds = x >= 0
            && y >= 0
            && (max_width.is_none() || x < max_width.unwrap() as isize)
            && (max_height.is_none() || y < max_height.unwrap() as isize);

        if in_bounds {
            coords.push((x as usize, y as usize));
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }

    // Add the endpoint if in bounds
    let end_in_bounds = (max_width.is_none() || x2 < max_width.unwrap())
        && (max_height.is_none() || y2 < max_height.unwrap());

    if end_in_bounds {
        coords.push((x2, y2));
    }

    coords
}
