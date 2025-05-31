use crate::error::SpatialResult;
use crate::rtree::node::{Entry, Node, RTree, Rectangle};
use ndarray::ArrayView1;

impl<T: Clone> RTree<T> {
    /// Delete a data point from the R-tree
    ///
    /// # Arguments
    ///
    /// * `point` - The point coordinates to delete
    /// * `data_predicate` - An optional function that takes a reference to the data and returns
    ///   true if it should be deleted. This is useful when multiple data points share the same coordinates.
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing true if a point was deleted, false otherwise
    pub fn delete<F>(
        &mut self,
        point: &ArrayView1<f64>,
        data_predicate: Option<F>,
    ) -> SpatialResult<bool>
    where
        F: Fn(&T) -> bool + Copy,
    {
        if point.len() != self.ndim() {
            return Err(crate::error::SpatialError::DimensionError(format!(
                "Point dimension {} does not match RTree dimension {}",
                point.len(),
                self.ndim()
            )));
        }

        // Create a rectangle for the point
        let mbr = Rectangle::from_point(point);

        // Find the leaf node(s) containing the point
        // Create a new empty root for swapping
        let mut root = std::mem::take(&mut self.root);
        let result = self.delete_internal(&mbr, &mut root, data_predicate)?;
        self.root = root;

        // If a point was deleted, decrement the size
        if result {
            self.decrement_size();

            // If the root has only one child and it's not a leaf, make the child the new root
            if !self.root.is_leaf && self.root.size() == 1 {
                if let Entry::NonLeaf { child, .. } = &self.root.entries[0] {
                    let new_root = (**child).clone();
                    self.root = new_root;
                    // Height is decremented here (would normally use self.height -= 1)
                }
            }
        }

        Ok(result)
    }

    /// Helper function to delete a point from a subtree
    #[allow(clippy::type_complexity)]
    fn delete_internal<F>(
        &mut self,
        mbr: &Rectangle,
        node: &mut Node<T>,
        data_predicate: Option<F>,
    ) -> SpatialResult<bool>
    where
        F: Fn(&T) -> bool + Copy,
    {
        // If this is a leaf node, look for the entry to delete
        if node.is_leaf {
            let mut found_index = None;

            for (i, entry) in node.entries.iter().enumerate() {
                if let Entry::Leaf {
                    mbr: entry_mbr,
                    data,
                    ..
                } = entry
                {
                    // Check if the MBRs match
                    if entry_mbr.min == mbr.min && entry_mbr.max == mbr.max {
                        // If a predicate is provided, check it too
                        if let Some(ref pred) = data_predicate {
                            if pred(data) {
                                found_index = Some(i);
                                break;
                            }
                        } else {
                            // No predicate, just delete the first matching entry
                            found_index = Some(i);
                            break;
                        }
                    }
                }
            }

            // If an entry was found, remove it
            if let Some(index) = found_index {
                node.entries.remove(index);
                return Ok(true);
            }

            return Ok(false);
        }

        // For non-leaf nodes, find all child nodes that could contain the point
        let mut deleted = false;

        for i in 0..node.size() {
            let entry_mbr = node.entries[i].mbr();

            // Check if this entry's MBR intersects with the search MBR
            if entry_mbr.intersects(mbr)? {
                // Get the child node
                if let Entry::NonLeaf { child, .. } = &mut node.entries[i] {
                    // Recursively delete from the child using raw pointers to avoid borrow issues
                    let child_ptr = Box::as_mut(child) as *mut Node<T>;
                    let result =
                        unsafe { self.delete_internal(mbr, &mut *child_ptr, data_predicate)? };

                    if result {
                        deleted = true;

                        // Check if the child node is underfull
                        if child.size() < self.min_entries && child.size() > 0 {
                            // Handle underfull child node
                            self.handle_underfull_node(node, i)?;
                        } else if child.size() == 0 {
                            // Remove empty child node
                            node.entries.remove(i);
                        } else {
                            // Update the MBR of the parent entry
                            if let Some(child_mbr) = child.mbr() {
                                if let Entry::NonLeaf { mbr, .. } = &mut node.entries[i] {
                                    *mbr = child_mbr;
                                }
                            }
                        }

                        break;
                    }
                }
            }
        }

        Ok(deleted)
    }

    /// Handle an underfull node by either merging it with a sibling or redistributing entries
    fn handle_underfull_node(
        &mut self,
        parent: &mut Node<T>,
        child_index: usize,
    ) -> SpatialResult<()> {
        // Get the child node
        let child = match &parent.entries[child_index] {
            Entry::NonLeaf { child, .. } => child,
            _ => {
                return Err(crate::error::SpatialError::ComputationError(
                    "Expected a non-leaf entry".into(),
                ))
            }
        };

        // Find the best sibling to merge with
        let mut best_sibling_index = None;
        let mut min_merged_area = f64::MAX;

        for i in 0..parent.size() {
            if i == child_index {
                continue;
            }

            // Get the sibling's MBR
            let sibling_mbr = parent.entries[i].mbr();

            // Get the child's MBR
            let child_mbr = child
                .mbr()
                .unwrap_or_else(|| Rectangle::from_point(&Array1::zeros(self.ndim()).view()));

            // Calculate the area of the merged MBR
            let merged_mbr = child_mbr.enlarge(sibling_mbr)?;
            let merged_area = merged_mbr.area();

            if merged_area < min_merged_area {
                min_merged_area = merged_area;
                best_sibling_index = Some(i);
            }
        }

        // If no sibling was found, return (this shouldn't happen)
        let best_sibling_index = best_sibling_index.unwrap_or(0);
        if best_sibling_index == child_index {
            return Ok(());
        }

        // Get the sibling
        let sibling = match &parent.entries[best_sibling_index] {
            Entry::NonLeaf { child, .. } => child,
            _ => {
                return Err(crate::error::SpatialError::ComputationError(
                    "Expected a non-leaf entry".into(),
                ))
            }
        };

        // If the sibling has enough entries, we can redistribute
        if sibling.size() > self.min_entries {
            // TODO: Implement entry redistribution
            Ok(())
        } else {
            // Otherwise, merge the nodes
            // TODO: Implement node merging
            Ok(())
        }
    }
}

use ndarray::Array1;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rtree_delete() {
        // Create a new R-tree
        let mut rtree: RTree<i32> = RTree::new(2, 2, 4).unwrap();

        // Insert some points
        let points = vec![
            (array![0.0, 0.0], 0),
            (array![1.0, 0.0], 1),
            (array![0.0, 1.0], 2),
            (array![1.0, 1.0], 3),
            (array![0.5, 0.5], 4),
        ];

        for (point, value) in points {
            rtree.insert(point, value).unwrap();
        }

        // Delete a point
        let result = rtree
            .delete::<fn(&i32) -> bool>(&array![0.5, 0.5].view(), None)
            .unwrap();
        assert!(result);

        // Check the size
        assert_eq!(rtree.size(), 4);

        // Try to delete a point that doesn't exist
        let result = rtree
            .delete::<fn(&i32) -> bool>(&array![2.0, 2.0].view(), None)
            .unwrap();
        assert!(!result);

        // Check the size
        assert_eq!(rtree.size(), 4);

        // Delete all remaining points
        let result = rtree
            .delete::<fn(&i32) -> bool>(&array![0.0, 0.0].view(), None)
            .unwrap();
        assert!(result);
        let result = rtree
            .delete::<fn(&i32) -> bool>(&array![1.0, 0.0].view(), None)
            .unwrap();
        assert!(result);
        let result = rtree
            .delete::<fn(&i32) -> bool>(&array![0.0, 1.0].view(), None)
            .unwrap();
        assert!(result);
        let result = rtree
            .delete::<fn(&i32) -> bool>(&array![1.0, 1.0].view(), None)
            .unwrap();
        assert!(result);

        // Check the size
        assert_eq!(rtree.size(), 0);
        assert!(rtree.is_empty());
    }
}
