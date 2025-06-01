use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, ArrayView1};
use std::cmp::Ordering;
// No imports from std::collections needed

/// Rectangle represents a minimum bounding rectangle (MBR) in N-dimensional space.
#[derive(Clone, Debug)]
pub struct Rectangle {
    /// Minimum coordinates in each dimension
    pub(crate) min: Array1<f64>,
    /// Maximum coordinates in each dimension
    pub(crate) max: Array1<f64>,
}

impl Rectangle {
    /// Create a new rectangle from min and max coordinates
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum coordinates in each dimension
    /// * `max` - Maximum coordinates in each dimension
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the rectangle if valid, or an error if the input is invalid
    pub fn new(min: Array1<f64>, max: Array1<f64>) -> SpatialResult<Self> {
        if min.len() != max.len() {
            return Err(SpatialError::DimensionError(format!(
                "Min and max must have the same dimensions. Got {} and {}",
                min.len(),
                max.len()
            )));
        }

        // Validate that min ≤ max in all dimensions
        for i in 0..min.len() {
            if min[i] > max[i] {
                return Err(SpatialError::ValueError(
                    format!("Min must be less than or equal to max in all dimensions. Dimension {}: {} > {}", 
                           i, min[i], max[i])
                ));
            }
        }

        Ok(Rectangle { min, max })
    }

    /// Create a rectangle from a point (zero-area rectangle)
    ///
    /// # Arguments
    ///
    /// * `point` - The point coordinates
    ///
    /// # Returns
    ///
    /// A rectangle containing only the given point
    pub fn from_point(point: &ArrayView1<f64>) -> Self {
        Rectangle {
            min: point.to_owned(),
            max: point.to_owned(),
        }
    }

    /// Get the dimensionality of the rectangle
    pub fn ndim(&self) -> usize {
        self.min.len()
    }

    /// Calculate the area of the rectangle
    pub fn area(&self) -> f64 {
        let mut area = 1.0;
        for i in 0..self.ndim() {
            area *= self.max[i] - self.min[i];
        }
        area
    }

    /// Calculate the margin (perimeter) of the rectangle
    pub fn margin(&self) -> f64 {
        let mut margin = 0.0;
        for i in 0..self.ndim() {
            margin += self.max[i] - self.min[i];
        }
        2.0 * margin
    }

    /// Check if this rectangle contains a point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    ///
    /// # Returns
    ///
    /// `true` if the point is inside or on the boundary of the rectangle, `false` otherwise
    pub fn contains_point(&self, point: &ArrayView1<f64>) -> SpatialResult<bool> {
        if point.len() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Point dimension {} does not match rectangle dimension {}",
                point.len(),
                self.ndim()
            )));
        }

        for i in 0..self.ndim() {
            if point[i] < self.min[i] || point[i] > self.max[i] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if this rectangle contains another rectangle
    ///
    /// # Arguments
    ///
    /// * `other` - The other rectangle to check
    ///
    /// # Returns
    ///
    /// `true` if the other rectangle is completely inside this rectangle, `false` otherwise
    pub fn contains_rectangle(&self, other: &Rectangle) -> SpatialResult<bool> {
        if other.ndim() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Rectangle dimensions do not match: {} and {}",
                self.ndim(),
                other.ndim()
            )));
        }

        for i in 0..self.ndim() {
            if other.min[i] < self.min[i] || other.max[i] > self.max[i] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if this rectangle intersects another rectangle
    ///
    /// # Arguments
    ///
    /// * `other` - The other rectangle to check
    ///
    /// # Returns
    ///
    /// `true` if the rectangles intersect, `false` otherwise
    pub fn intersects(&self, other: &Rectangle) -> SpatialResult<bool> {
        if other.ndim() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Rectangle dimensions do not match: {} and {}",
                self.ndim(),
                other.ndim()
            )));
        }

        for i in 0..self.ndim() {
            if self.max[i] < other.min[i] || self.min[i] > other.max[i] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Calculate the intersection of this rectangle with another rectangle
    ///
    /// # Arguments
    ///
    /// * `other` - The other rectangle
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the intersection rectangle if it exists,
    /// or an error if the rectangles do not intersect or have different dimensions
    pub fn intersection(&self, other: &Rectangle) -> SpatialResult<Rectangle> {
        if other.ndim() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Rectangle dimensions do not match: {} and {}",
                self.ndim(),
                other.ndim()
            )));
        }

        let intersects = self.intersects(other)?;
        if !intersects {
            return Err(SpatialError::ValueError(
                "Rectangles do not intersect".into(),
            ));
        }

        let mut min = Array1::zeros(self.ndim());
        let mut max = Array1::zeros(self.ndim());

        for i in 0..self.ndim() {
            min[i] = f64::max(self.min[i], other.min[i]);
            max[i] = f64::min(self.max[i], other.max[i]);
        }

        Rectangle::new(min, max)
    }

    /// Calculate the minimum bounding rectangle (MBR) containing this rectangle and another
    ///
    /// # Arguments
    ///
    /// * `other` - The other rectangle
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the enlarged rectangle containing both rectangles,
    /// or an error if the rectangles have different dimensions
    pub fn enlarge(&self, other: &Rectangle) -> SpatialResult<Rectangle> {
        if other.ndim() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Rectangle dimensions do not match: {} and {}",
                self.ndim(),
                other.ndim()
            )));
        }

        let mut min = Array1::zeros(self.ndim());
        let mut max = Array1::zeros(self.ndim());

        for i in 0..self.ndim() {
            min[i] = f64::min(self.min[i], other.min[i]);
            max[i] = f64::max(self.max[i], other.max[i]);
        }

        Rectangle::new(min, max)
    }

    /// Calculate the enlargement area needed to include another rectangle
    ///
    /// # Arguments
    ///
    /// * `other` - The other rectangle
    ///
    /// # Returns
    ///
    /// The difference between the area of the enlarged rectangle and this rectangle,
    /// or an error if the rectangles have different dimensions
    pub fn enlargement_area(&self, other: &Rectangle) -> SpatialResult<f64> {
        let enlarged = self.enlarge(other)?;
        Ok(enlarged.area() - self.area())
    }

    /// Calculate the minimum distance from this rectangle to a point
    ///
    /// # Arguments
    ///
    /// * `point` - The point
    ///
    /// # Returns
    ///
    /// The minimum distance from the rectangle to the point,
    /// or an error if the point has a different dimension
    pub fn min_distance_to_point(&self, point: &ArrayView1<f64>) -> SpatialResult<f64> {
        if point.len() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Point dimension {} does not match rectangle dimension {}",
                point.len(),
                self.ndim()
            )));
        }

        let mut distance_sq = 0.0;

        for i in 0..self.ndim() {
            if point[i] < self.min[i] {
                distance_sq += (point[i] - self.min[i]).powi(2);
            } else if point[i] > self.max[i] {
                distance_sq += (point[i] - self.max[i]).powi(2);
            }
        }

        Ok(distance_sq.sqrt())
    }

    /// Calculate the minimum distance from this rectangle to another rectangle
    ///
    /// # Arguments
    ///
    /// * `other` - The other rectangle
    ///
    /// # Returns
    ///
    /// The minimum distance between the rectangles,
    /// or an error if they have different dimensions
    pub fn min_distance_to_rectangle(&self, other: &Rectangle) -> SpatialResult<f64> {
        if other.ndim() != self.ndim() {
            return Err(SpatialError::DimensionError(format!(
                "Rectangle dimensions do not match: {} and {}",
                self.ndim(),
                other.ndim()
            )));
        }

        // If the rectangles intersect, the distance is 0
        if self.intersects(other)? {
            return Ok(0.0);
        }

        let mut distance_sq = 0.0;

        for i in 0..self.ndim() {
            if self.max[i] < other.min[i] {
                distance_sq += (other.min[i] - self.max[i]).powi(2);
            } else if self.min[i] > other.max[i] {
                distance_sq += (self.min[i] - other.max[i]).powi(2);
            }
        }

        Ok(distance_sq.sqrt())
    }
}

/// Entry type in R-tree node
#[derive(Clone, Debug)]
pub(crate) enum Entry<T>
where
    T: Clone,
{
    /// Leaf entry containing a data record
    Leaf {
        /// Bounding rectangle
        mbr: Rectangle,
        /// Data associated with this entry
        data: T,
        /// Index of the data in the original array
        index: usize,
    },

    /// Non-leaf entry pointing to a child node
    NonLeaf {
        /// Bounding rectangle
        mbr: Rectangle,
        /// Child node
        child: Box<Node<T>>,
    },
}

impl<T: Clone> Entry<T> {
    /// Get the minimum bounding rectangle of this entry
    pub(crate) fn mbr(&self) -> &Rectangle {
        match self {
            Entry::Leaf { mbr, .. } => mbr,
            Entry::NonLeaf { mbr, .. } => mbr,
        }
    }
}

/// Node in the R-tree
#[derive(Clone, Debug)]
pub(crate) struct Node<T>
where
    T: Clone,
{
    /// Entries stored in this node
    pub entries: Vec<Entry<T>>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
    /// Node level in the tree (0 for leaf nodes, increasing towards the root)
    pub level: usize,
}

impl<T: Clone> Default for Node<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            is_leaf: true,
            level: 0,
        }
    }
}

impl<T: Clone> Node<T> {
    /// Create a new node
    pub fn new(is_leaf: bool, level: usize) -> Self {
        Node {
            entries: Vec::new(),
            is_leaf,
            level,
        }
    }

    /// Get the number of entries in the node
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Calculate the bounding rectangle for the node
    pub fn mbr(&self) -> Option<Rectangle> {
        if self.entries.is_empty() {
            return None;
        }

        let mut result = self.entries[0].mbr().clone();

        for i in 1..self.size() {
            result = result.enlarge(self.entries[i].mbr()).unwrap();
        }

        Some(result)
    }
}

/// Entry with distance for nearest neighbor search priority queue
#[derive(Clone, Debug)]
pub(crate) struct EntryWithDistance<T>
where
    T: Clone,
{
    /// Entry
    pub entry: Entry<T>,
    /// Distance to the query point
    pub distance: f64,
}

impl<T: Clone> PartialEq for EntryWithDistance<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: Clone> Eq for EntryWithDistance<T> {}

impl<T: Clone> PartialOrd for EntryWithDistance<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Clone> Ord for EntryWithDistance<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering - smaller distances come first
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// R-tree for spatial indexing and queries
///
/// # Examples
///
/// ```
/// use scirs2_spatial::rtree::RTree;
/// use ndarray::array;
///
/// // Create points
/// let points = array![
///     [0.0, 0.0],
///     [1.0, 1.0],
///     [2.0, 2.0],
///     [3.0, 3.0],
///     [4.0, 4.0],
/// ];
///
/// // Build R-tree with points
/// let mut rtree = RTree::new(2, 4, 8).unwrap();
/// for (i, point) in points.rows().into_iter().enumerate() {
///     rtree.insert(point.to_owned(), i).unwrap();
/// }
///
/// // Search for points within a range
/// let query_min = array![0.5, 0.5];
/// let query_max = array![2.5, 2.5];
///
/// // The views need to be passed explicitly in the actual code
/// let query_min_view = query_min.view();
/// let query_max_view = query_max.view();
/// let results = rtree.search_range(&query_min_view, &query_max_view).unwrap();
///
/// println!("Found {} points in range", results.len());
///
/// // Find the nearest neighbors to a point
/// let query_point = array![2.5, 2.5];
/// let query_point_view = query_point.view();
/// let nearest = rtree.nearest(&query_point_view, 2).unwrap();
///
/// println!("Nearest points: {:?}", nearest);
/// ```
#[derive(Clone, Debug)]
pub struct RTree<T>
where
    T: Clone,
{
    /// Root node of the tree
    pub(crate) root: Node<T>,
    /// Number of dimensions
    ndim: usize,
    /// Minimum number of entries in each node (except the root)
    pub(crate) min_entries: usize,
    /// Maximum number of entries in each node
    pub(crate) max_entries: usize,
    /// Number of data points in the tree
    size: usize,
    /// Height of the tree
    height: usize,
}

impl<T: Clone> RTree<T> {
    /// Create a new R-tree
    ///
    /// # Arguments
    ///
    /// * `ndim` - Number of dimensions
    /// * `min_entries` - Minimum number of entries in each node (except the root)
    /// * `max_entries` - Maximum number of entries in each node
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the new R-tree, or an error if the parameters are invalid
    pub fn new(ndim: usize, min_entries: usize, max_entries: usize) -> SpatialResult<Self> {
        if ndim == 0 {
            return Err(SpatialError::ValueError(
                "Number of dimensions must be positive".into(),
            ));
        }

        if min_entries < 1 || min_entries > max_entries / 2 {
            return Err(SpatialError::ValueError(format!(
                "min_entries must be between 1 and max_entries/2, got: {}",
                min_entries
            )));
        }

        if max_entries < 2 {
            return Err(SpatialError::ValueError(format!(
                "max_entries must be at least 2, got: {}",
                max_entries
            )));
        }

        Ok(RTree {
            root: Node::new(true, 0),
            ndim,
            min_entries,
            max_entries,
            size: 0,
            height: 1,
        })
    }

    /// Get the number of dimensions in the R-tree
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get the number of data points in the R-tree
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the height of the R-tree
    pub fn height(&self) -> usize {
        self.height
    }

    /// Check if the R-tree is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear the R-tree, removing all data points
    pub fn clear(&mut self) {
        self.root = Node::new(true, 0);
        self.size = 0;
        self.height = 1;
    }

    /// Increment the size of the tree (internal use only)
    pub(crate) fn increment_size(&mut self) {
        self.size += 1;
    }

    /// Decrement the size of the tree (internal use only)
    pub(crate) fn decrement_size(&mut self) {
        if self.size > 0 {
            self.size -= 1;
        }
    }

    /// Increment the height of the tree (internal use only)
    pub(crate) fn increment_height(&mut self) {
        self.height += 1;
    }

    /// Decrement the height of the tree (internal use only)
    #[allow(dead_code)]
    pub(crate) fn decrement_height(&mut self) {
        if self.height > 0 {
            self.height -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_rectangle_creation() {
        let min = array![0.0, 0.0, 0.0];
        let max = array![1.0, 2.0, 3.0];

        let rect = Rectangle::new(min.clone(), max.clone()).unwrap();

        assert_eq!(rect.ndim(), 3);
        assert_eq!(rect.min, min);
        assert_eq!(rect.max, max);

        // Test invalid rectangle (min > max)
        let min_invalid = array![0.0, 3.0, 0.0];
        let max_invalid = array![1.0, 2.0, 3.0];

        let result = Rectangle::new(min_invalid, max_invalid);
        assert!(result.is_err());

        // Test dimension mismatch
        let min_dim_mismatch = array![0.0, 0.0];
        let max_dim_mismatch = array![1.0, 2.0, 3.0];

        let result = Rectangle::new(min_dim_mismatch, max_dim_mismatch);
        assert!(result.is_err());
    }

    #[test]
    fn test_rectangle_area_and_margin() {
        let min = array![0.0, 0.0, 0.0];
        let max = array![1.0, 2.0, 3.0];

        let rect = Rectangle::new(min, max).unwrap();

        assert_relative_eq!(rect.area(), 6.0);
        assert_relative_eq!(rect.margin(), 12.0);
    }

    #[test]
    fn test_rectangle_contains_point() {
        let min = array![0.0, 0.0];
        let max = array![1.0, 1.0];

        let rect = Rectangle::new(min, max).unwrap();

        // Inside
        let point_inside = array![0.5, 0.5];
        assert!(rect.contains_point(&point_inside.view()).unwrap());

        // On boundary
        let point_boundary = array![0.0, 0.5];
        assert!(rect.contains_point(&point_boundary.view()).unwrap());

        // Outside
        let point_outside = array![2.0, 0.5];
        assert!(!rect.contains_point(&point_outside.view()).unwrap());

        // Dimension mismatch
        let point_dim_mismatch = array![0.5, 0.5, 0.5];
        assert!(rect.contains_point(&point_dim_mismatch.view()).is_err());
    }

    #[test]
    fn test_rectangle_intersects() {
        let rect1 = Rectangle::new(array![0.0, 0.0], array![2.0, 2.0]).unwrap();

        // Overlap
        let rect2 = Rectangle::new(array![1.0, 1.0], array![3.0, 3.0]).unwrap();
        assert!(rect1.intersects(&rect2).unwrap());

        // Touch
        let rect3 = Rectangle::new(array![2.0, 0.0], array![3.0, 2.0]).unwrap();
        assert!(rect1.intersects(&rect3).unwrap());

        // No intersection
        let rect4 = Rectangle::new(array![3.0, 3.0], array![4.0, 4.0]).unwrap();
        assert!(!rect1.intersects(&rect4).unwrap());

        // Dimension mismatch
        let rect5 = Rectangle::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0]).unwrap();
        assert!(rect1.intersects(&rect5).is_err());
    }

    #[test]
    fn test_rectangle_enlarge() {
        let rect1 = Rectangle::new(array![1.0, 1.0], array![2.0, 2.0]).unwrap();
        let rect2 = Rectangle::new(array![0.0, 0.0], array![3.0, 1.5]).unwrap();

        let enlarged = rect1.enlarge(&rect2).unwrap();

        assert_eq!(enlarged.min, array![0.0, 0.0]);
        assert_eq!(enlarged.max, array![3.0, 2.0]);

        // Calculate enlargement area
        let enlargement = rect1.enlargement_area(&rect2).unwrap();
        let original_area = rect1.area();
        let enlarged_area = enlarged.area();

        assert_relative_eq!(enlargement, enlarged_area - original_area);
    }

    #[test]
    fn test_rectangle_distance() {
        let rect = Rectangle::new(array![1.0, 1.0], array![3.0, 3.0]).unwrap();

        // Inside: distance should be 0
        let point_inside = array![2.0, 2.0];
        assert_relative_eq!(
            rect.min_distance_to_point(&point_inside.view()).unwrap(),
            0.0
        );

        // Outside
        let point_outside = array![0.0, 0.0];
        assert_relative_eq!(
            rect.min_distance_to_point(&point_outside.view()).unwrap(),
            2.0_f64.sqrt()
        );

        // Other rectangle - no intersection
        let rect2 = Rectangle::new(array![4.0, 4.0], array![5.0, 5.0]).unwrap();
        assert_relative_eq!(
            rect.min_distance_to_rectangle(&rect2).unwrap(),
            2.0_f64.sqrt()
        );

        // Other rectangle - intersection (distance = 0)
        let rect3 = Rectangle::new(array![2.0, 2.0], array![4.0, 4.0]).unwrap();
        assert_relative_eq!(rect.min_distance_to_rectangle(&rect3).unwrap(), 0.0);
    }

    #[test]
    fn test_rtree_creation() {
        // Valid parameters
        let rtree: RTree<usize> = RTree::new(2, 2, 5).unwrap();
        assert_eq!(rtree.ndim(), 2);
        assert_eq!(rtree.size(), 0);
        assert_eq!(rtree.height(), 1);
        assert!(rtree.is_empty());

        // Invalid parameters
        assert!(RTree::<usize>::new(0, 2, 5).is_err()); // ndim = 0
        assert!(RTree::<usize>::new(2, 0, 5).is_err()); // min_entries = 0
        assert!(RTree::<usize>::new(2, 3, 5).is_err()); // min_entries > max_entries/2
        assert!(RTree::<usize>::new(2, 2, 1).is_err()); // max_entries < 2
    }
}
