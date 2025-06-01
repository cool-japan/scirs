use crate::error::SpatialResult;
use crate::rtree::node::{Entry, Node, RTree};
use ndarray::Array1;

impl<T: Clone> RTree<T> {
    /// Optimize the R-tree by rebuilding it with the current data
    ///
    /// This can significantly improve query performance by reducing overlap
    /// and creating a more balanced tree.
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing nothing if successful
    pub fn optimize(&mut self) -> SpatialResult<()> {
        // Collect all data points
        let data_points = self.collect_all_data_points()?;

        if data_points.is_empty() {
            return Ok(());
        }

        // Create a new, empty R-tree
        // These parameters are not used in this method but would be used
        // if we were creating a new R-tree
        // let _ndim = self.ndim();
        // let _min_entries = self.min_entries;
        // let _max_entries = self.max_entries;

        // Save current size to check at the end
        let size = self.size();

        // Clear current tree
        self.clear();

        // Re-insert all data points (bulk loading would be more efficient)
        for (point, data, _index) in data_points {
            self.insert(point, data)?;
        }

        // Verify integrity
        assert_eq!(self.size(), size, "Size mismatch after optimization");

        Ok(())
    }

    /// Collect all data points in the R-tree
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing a vector of (point, data, index) tuples
    fn collect_all_data_points(&self) -> SpatialResult<Vec<(Array1<f64>, T, usize)>> {
        let mut points = Vec::new();
        self.collect_data_points_recursive(&self.root, &mut points)?;
        Ok(points)
    }

    /// Recursively collect data points from a node
    #[allow(clippy::only_used_in_recursion)]
    fn collect_data_points_recursive(
        &self,
        node: &Node<T>,
        points: &mut Vec<(Array1<f64>, T, usize)>,
    ) -> SpatialResult<()> {
        for entry in &node.entries {
            match entry {
                Entry::Leaf { mbr, data, index } => {
                    // For leaf entries, the MBR should be a point (min == max)
                    points.push((mbr.min.clone(), data.clone(), *index));
                }
                Entry::NonLeaf { child, .. } => {
                    // For non-leaf entries, recursively collect from children
                    self.collect_data_points_recursive(child, points)?;
                }
            }
        }
        Ok(())
    }

    /// Perform bulk loading of the R-tree with sorted data points
    ///
    /// This is more efficient than inserting points one by one.
    ///
    /// # Arguments
    ///
    /// * `points` - A vector of (point, data) pairs to insert
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing a new R-tree built from the data points
    pub fn bulk_load(
        ndim: usize,
        min_entries: usize,
        max_entries: usize,
        points: Vec<(Array1<f64>, T)>,
    ) -> SpatialResult<Self> {
        // Create a new, empty R-tree
        let mut rtree = RTree::new(ndim, min_entries, max_entries)?;

        if points.is_empty() {
            return Ok(rtree);
        }

        // TODO: Implement Sort-Tile-Recursive (STR) bulk loading algorithm
        // For now, just insert points one by one
        for (i, (point, data)) in points.into_iter().enumerate() {
            // Validate dimensions
            if point.len() != ndim {
                return Err(crate::error::SpatialError::DimensionError(format!(
                    "Point at index {} has dimension {} but tree dimension is {}",
                    i,
                    point.len(),
                    ndim
                )));
            }

            rtree.insert(point, data)?;
        }

        Ok(rtree)
    }

    /// Calculate the total overlap in the R-tree
    ///
    /// This is a quality metric for the tree. Lower overlap generally means
    /// better query performance.
    ///
    /// # Returns
    ///
    /// The total overlap area between all pairs of nodes at each level
    pub fn calculate_total_overlap(&self) -> SpatialResult<f64> {
        let mut total_overlap = 0.0;

        // Calculate overlap at each level, starting from the root
        let mut current_level_nodes = vec![&self.root];

        while !current_level_nodes.is_empty() {
            // Calculate overlap between nodes at this level
            for i in 0..current_level_nodes.len() - 1 {
                let node_i_mbr = match current_level_nodes[i].mbr() {
                    Some(mbr) => mbr,
                    None => continue,
                };

                for node_j in current_level_nodes.iter().skip(i + 1) {
                    let node_j_mbr = match node_j.mbr() {
                        Some(mbr) => mbr,
                        None => continue,
                    };

                    // Check if MBRs intersect
                    if node_i_mbr.intersects(&node_j_mbr)? {
                        // Calculate intersection area
                        if let Ok(intersection) = node_i_mbr.intersection(&node_j_mbr) {
                            total_overlap += intersection.area();
                        }
                    }
                }
            }

            // Move to the next level
            let mut next_level_nodes = Vec::new();
            for node in current_level_nodes {
                for entry in &node.entries {
                    if let Entry::NonLeaf { child, .. } = entry {
                        next_level_nodes.push(&**child);
                    }
                }
            }

            current_level_nodes = next_level_nodes;
        }

        Ok(total_overlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rtree_optimize() {
        // Create a new R-tree
        let mut rtree: RTree<i32> = RTree::new(2, 2, 4).unwrap();

        // Insert some points
        let points = vec![
            (array![0.0, 0.0], 0),
            (array![1.0, 0.0], 1),
            (array![0.0, 1.0], 2),
            (array![1.0, 1.0], 3),
            (array![0.5, 0.5], 4),
            (array![2.0, 2.0], 5),
            (array![3.0, 3.0], 6),
            (array![4.0, 4.0], 7),
            (array![5.0, 5.0], 8),
            (array![6.0, 6.0], 9),
        ];

        for (point, value) in points {
            rtree.insert(point, value).unwrap();
        }

        // Optimize the tree
        rtree.optimize().unwrap();

        // Check that all data is still present
        assert_eq!(rtree.size(), 10);

        // Try to search for a point
        let results = rtree
            .search_range(&array![0.4, 0.4].view(), &array![0.6, 0.6].view())
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 4);
    }

    #[test]
    fn test_rtree_bulk_load() {
        // Create points
        let points = vec![
            (array![0.0, 0.0], 0),
            (array![1.0, 0.0], 1),
            (array![0.0, 1.0], 2),
            (array![1.0, 1.0], 3),
            (array![0.5, 0.5], 4),
            (array![2.0, 2.0], 5),
            (array![3.0, 3.0], 6),
            (array![4.0, 4.0], 7),
            (array![5.0, 5.0], 8),
            (array![6.0, 6.0], 9),
        ];

        // Bulk load
        let rtree = RTree::bulk_load(2, 2, 4, points).unwrap();

        // Check that all data is present
        assert_eq!(rtree.size(), 10);

        // Try to search for a point
        let results = rtree
            .search_range(&array![0.4, 0.4].view(), &array![0.6, 0.6].view())
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 4);
    }
}
