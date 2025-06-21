//! Disjoint Set (Union-Find) data structure for connectivity queries
//!
//! This module provides a disjoint set data structure that efficiently supports
//! union and find operations. It's particularly useful for clustering algorithms
//! that need to track connected components or merge clusters.
//!
//! The implementation uses path compression and union by rank optimizations
//! for nearly O(1) amortized performance.

use std::collections::HashMap;

/// Disjoint Set (Union-Find) data structure
///
/// This data structure maintains a collection of disjoint sets and supports
/// efficient union and find operations. It's commonly used in clustering
/// algorithms for tracking connected components.
///
/// # Examples
///
/// ```
/// use scirs2_cluster::hierarchy::DisjointSet;
///
/// let mut ds = DisjointSet::new();
///
/// // Add some elements
/// ds.make_set(1);
/// ds.make_set(2);
/// ds.make_set(3);
/// ds.make_set(4);
///
/// // Union some sets
/// ds.union(1, 2);
/// ds.union(3, 4);
///
/// // Check connectivity
/// assert_eq!(ds.find(&1), ds.find(&2)); // 1 and 2 are connected
/// assert_eq!(ds.find(&3), ds.find(&4)); // 3 and 4 are connected
/// assert_ne!(ds.find(&1), ds.find(&3)); // 1 and 3 are in different sets
/// ```
#[derive(Debug, Clone)]
pub struct DisjointSet<T: Clone + std::hash::Hash + Eq> {
    /// Parent pointers for each element
    parent: HashMap<T, T>,
    /// Rank (approximate depth) of each tree
    rank: HashMap<T, usize>,
    /// Number of disjoint sets
    num_sets: usize,
}

impl<T: Clone + std::hash::Hash + Eq> DisjointSet<T> {
    /// Create a new empty disjoint set
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let ds: DisjointSet<i32> = DisjointSet::new();
    /// ```
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
            num_sets: 0,
        }
    }

    /// Create a new disjoint set with a specified capacity
    ///
    /// This can improve performance when you know approximately how many
    /// elements you'll be adding.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of elements
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            parent: HashMap::with_capacity(capacity),
            rank: HashMap::with_capacity(capacity),
            num_sets: 0,
        }
    }

    /// Add a new element as its own singleton set
    ///
    /// If the element already exists, this operation has no effect.
    ///
    /// # Arguments
    ///
    /// * `x` - Element to add
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(42);
    /// assert!(ds.contains(&42));
    /// ```
    pub fn make_set(&mut self, x: T) {
        if !self.parent.contains_key(&x) {
            self.parent.insert(x.clone(), x.clone());
            self.rank.insert(x, 0);
            self.num_sets += 1;
        }
    }

    /// Find the representative (root) of the set containing the given element
    ///
    /// Uses path compression for optimization: all nodes on the path to the root
    /// are made to point directly to the root.
    ///
    /// # Arguments
    ///
    /// * `x` - Element to find the representative for
    ///
    /// # Returns
    ///
    /// * `Some(representative)` if the element exists in the structure
    /// * `None` if the element doesn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(1);
    /// ds.make_set(2);
    /// ds.union(1, 2);
    ///
    /// let root1 = ds.find(&1).unwrap();
    /// let root2 = ds.find(&2).unwrap();
    /// assert_eq!(root1, root2); // Same representative
    /// ```
    pub fn find(&mut self, x: &T) -> Option<T> {
        if !self.parent.contains_key(x) {
            return None;
        }

        // Path compression: make all nodes on path point to root
        let mut current = x.clone();
        let mut path = Vec::new();

        // Find root
        while self.parent[&current] != current {
            path.push(current.clone());
            current = self.parent[&current].clone();
        }

        // Compress path
        for node in path {
            self.parent.insert(node, current.clone());
        }

        Some(current)
    }

    /// Union two sets containing the given elements
    ///
    /// Uses union by rank: the root of the tree with smaller rank becomes
    /// a child of the root with larger rank.
    ///
    /// # Arguments
    ///
    /// * `x` - Element from first set
    /// * `y` - Element from second set
    ///
    /// # Returns
    ///
    /// * `true` if the sets were successfully unioned (they were different sets)
    /// * `false` if the elements were already in the same set or don't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(1);
    /// ds.make_set(2);
    ///
    /// assert!(ds.union(1, 2)); // Successfully unioned
    /// assert!(!ds.union(1, 2)); // Already in same set
    /// ```
    pub fn union(&mut self, x: T, y: T) -> bool {
        let root_x = match self.find(&x) {
            Some(root) => root,
            None => return false,
        };

        let root_y = match self.find(&y) {
            Some(root) => root,
            None => return false,
        };

        if root_x == root_y {
            return false; // Already in same set
        }

        // Union by rank
        let rank_x = self.rank[&root_x];
        let rank_y = self.rank[&root_y];

        match rank_x.cmp(&rank_y) {
            std::cmp::Ordering::Less => {
                self.parent.insert(root_x, root_y);
            }
            std::cmp::Ordering::Greater => {
                self.parent.insert(root_y, root_x);
            }
            std::cmp::Ordering::Equal => {
                // Same rank, make one root and increase its rank
                self.parent.insert(root_y, root_x.clone());
                self.rank.insert(root_x, rank_x + 1);
            }
        }

        self.num_sets -= 1;
        true
    }

    /// Check if two elements are in the same set
    ///
    /// # Arguments
    ///
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// # Returns
    ///
    /// * `true` if both elements exist and are in the same set
    /// * `false` if they're in different sets or don't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(1);
    /// ds.make_set(2);
    /// ds.make_set(3);
    /// ds.union(1, 2);
    ///
    /// assert!(ds.connected(&1, &2)); // Connected
    /// assert!(!ds.connected(&1, &3)); // Not connected
    /// ```
    pub fn connected(&mut self, x: &T, y: &T) -> bool {
        match (self.find(x), self.find(y)) {
            (Some(root_x), Some(root_y)) => root_x == root_y,
            _ => false,
        }
    }

    /// Check if an element exists in the disjoint set
    ///
    /// # Arguments
    ///
    /// * `x` - Element to check
    ///
    /// # Returns
    ///
    /// * `true` if the element exists
    /// * `false` otherwise
    pub fn contains(&self, x: &T) -> bool {
        self.parent.contains_key(x)
    }

    /// Get the number of disjoint sets
    ///
    /// # Returns
    ///
    /// The number of disjoint sets currently in the structure
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// assert_eq!(ds.num_sets(), 0);
    ///
    /// ds.make_set(1);
    /// ds.make_set(2);
    /// assert_eq!(ds.num_sets(), 2);
    ///
    /// ds.union(1, 2);
    /// assert_eq!(ds.num_sets(), 1);
    /// ```
    pub fn num_sets(&self) -> usize {
        self.num_sets
    }

    /// Get the total number of elements
    ///
    /// # Returns
    ///
    /// The total number of elements in all sets
    pub fn size(&self) -> usize {
        self.parent.len()
    }

    /// Check if the disjoint set is empty
    ///
    /// # Returns
    ///
    /// * `true` if no elements have been added
    /// * `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Get all elements in the same set as the given element
    ///
    /// # Arguments
    ///
    /// * `x` - Element to find set members for
    ///
    /// # Returns
    ///
    /// * `Some(Vec<T>)` containing all elements in the same set
    /// * `None` if the element doesn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(1);
    /// ds.make_set(2);
    /// ds.make_set(3);
    /// ds.union(1, 2);
    ///
    /// let set_members = ds.get_set_members(&1).unwrap();
    /// assert_eq!(set_members.len(), 2);
    /// assert!(set_members.contains(&1));
    /// assert!(set_members.contains(&2));
    /// assert!(!set_members.contains(&3));
    /// ```
    pub fn get_set_members(&mut self, x: &T) -> Option<Vec<T>> {
        let target_root = self.find(x)?;

        let mut members = Vec::new();
        let elements_to_check: Vec<T> = self.parent.keys().cloned().collect();

        for element in elements_to_check {
            if let Some(root) = self.find(&element) {
                if root == target_root {
                    members.push(element);
                }
            }
        }

        Some(members)
    }

    /// Get all disjoint sets as a vector of vectors
    ///
    /// # Returns
    ///
    /// A vector where each inner vector contains the elements of one set
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(1);
    /// ds.make_set(2);
    /// ds.make_set(3);
    /// ds.union(1, 2);
    ///
    /// let all_sets = ds.get_all_sets();
    /// assert_eq!(all_sets.len(), 2); // Two disjoint sets
    /// ```
    pub fn get_all_sets(&mut self) -> Vec<Vec<T>> {
        let mut sets_map: HashMap<T, Vec<T>> = HashMap::new();

        // Group elements by their root
        for element in self.parent.keys().cloned().collect::<Vec<_>>() {
            if let Some(root) = self.find(&element) {
                sets_map.entry(root).or_default().push(element);
            }
        }

        sets_map.into_values().collect()
    }

    /// Clear all elements from the disjoint set
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_cluster::hierarchy::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.make_set(1);
    /// ds.make_set(2);
    ///
    /// assert_eq!(ds.size(), 2);
    /// ds.clear();
    /// assert_eq!(ds.size(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.num_sets = 0;
    }
}

impl<T: Clone + std::hash::Hash + Eq> Default for DisjointSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut ds = DisjointSet::new();

        // Initially empty
        assert_eq!(ds.size(), 0);
        assert_eq!(ds.num_sets(), 0);
        assert!(ds.is_empty());

        // Add elements
        ds.make_set(1);
        ds.make_set(2);
        ds.make_set(3);

        assert_eq!(ds.size(), 3);
        assert_eq!(ds.num_sets(), 3);
        assert!(!ds.is_empty());

        // Check individual sets
        assert!(ds.contains(&1));
        assert!(ds.contains(&2));
        assert!(ds.contains(&3));
        assert!(!ds.contains(&4));
    }

    #[test]
    fn test_union_find() {
        let mut ds = DisjointSet::new();
        ds.make_set(1);
        ds.make_set(2);
        ds.make_set(3);
        ds.make_set(4);

        // Initially all separate
        assert!(!ds.connected(&1, &2));
        assert!(!ds.connected(&3, &4));

        // Union 1 and 2
        assert!(ds.union(1, 2));
        assert_eq!(ds.num_sets(), 3);
        assert!(ds.connected(&1, &2));
        assert!(!ds.connected(&1, &3));

        // Union 3 and 4
        assert!(ds.union(3, 4));
        assert_eq!(ds.num_sets(), 2);
        assert!(ds.connected(&3, &4));
        assert!(!ds.connected(&1, &3));

        // Union the two sets
        assert!(ds.union(1, 3));
        assert_eq!(ds.num_sets(), 1);
        assert!(ds.connected(&1, &3));
        assert!(ds.connected(&2, &4));

        // Redundant union
        assert!(!ds.union(1, 2));
        assert_eq!(ds.num_sets(), 1);
    }

    #[test]
    fn test_path_compression() {
        let mut ds = DisjointSet::new();

        // Create a chain: 1 -> 2 -> 3 -> 4
        ds.make_set(1);
        ds.make_set(2);
        ds.make_set(3);
        ds.make_set(4);

        ds.union(1, 2);
        ds.union(2, 3);
        ds.union(3, 4);

        // After find operations, path should be compressed
        let root1 = ds.find(&1).unwrap();
        let root2 = ds.find(&2).unwrap();
        let root3 = ds.find(&3).unwrap();
        let root4 = ds.find(&4).unwrap();

        assert_eq!(root1, root2);
        assert_eq!(root2, root3);
        assert_eq!(root3, root4);
    }

    #[test]
    fn test_get_set_members() {
        let mut ds = DisjointSet::new();
        ds.make_set(1);
        ds.make_set(2);
        ds.make_set(3);
        ds.make_set(4);

        ds.union(1, 2);
        ds.union(3, 4);

        let members1 = ds.get_set_members(&1).unwrap();
        assert_eq!(members1.len(), 2);
        assert!(members1.contains(&1));
        assert!(members1.contains(&2));

        let members3 = ds.get_set_members(&3).unwrap();
        assert_eq!(members3.len(), 2);
        assert!(members3.contains(&3));
        assert!(members3.contains(&4));

        // Non-existent element
        assert!(ds.get_set_members(&5).is_none());
    }

    #[test]
    fn test_get_all_sets() {
        let mut ds = DisjointSet::new();
        ds.make_set(1);
        ds.make_set(2);
        ds.make_set(3);
        ds.make_set(4);
        ds.make_set(5);

        ds.union(1, 2);
        ds.union(3, 4);
        // 5 remains alone

        let all_sets = ds.get_all_sets();
        assert_eq!(all_sets.len(), 3); // Three disjoint sets

        // Find which set contains which elements
        let mut set_sizes: Vec<usize> = all_sets.iter().map(|s| s.len()).collect();
        set_sizes.sort();
        assert_eq!(set_sizes, vec![1, 2, 2]);
    }

    #[test]
    fn test_edge_cases() {
        let mut ds = DisjointSet::new();

        // Union with non-existent elements
        assert!(!ds.union(1, 2));

        // Find non-existent element
        assert!(ds.find(&1).is_none());

        // Connected with non-existent elements
        assert!(!ds.connected(&1, &2));

        // Add same element twice
        ds.make_set(1);
        ds.make_set(1); // Should have no effect
        assert_eq!(ds.size(), 1);
        assert_eq!(ds.num_sets(), 1);
    }

    #[test]
    fn test_clear() {
        let mut ds = DisjointSet::new();
        ds.make_set(1);
        ds.make_set(2);
        ds.union(1, 2);

        assert_eq!(ds.size(), 2);
        assert_eq!(ds.num_sets(), 1);

        ds.clear();

        assert_eq!(ds.size(), 0);
        assert_eq!(ds.num_sets(), 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_with_strings() {
        let mut ds = DisjointSet::new();
        ds.make_set("alice".to_string());
        ds.make_set("bob".to_string());
        ds.make_set("charlie".to_string());

        ds.union("alice".to_string(), "bob".to_string());

        assert!(ds.connected(&"alice".to_string(), &"bob".to_string()));
        assert!(!ds.connected(&"alice".to_string(), &"charlie".to_string()));
    }

    #[test]
    fn test_large_dataset() {
        let mut ds = DisjointSet::with_capacity(1000);

        // Add many elements
        for i in 0..1000 {
            ds.make_set(i);
        }

        assert_eq!(ds.size(), 1000);
        assert_eq!(ds.num_sets(), 1000);

        // Union them in pairs
        for i in (0..1000).step_by(2) {
            ds.union(i, i + 1);
        }

        assert_eq!(ds.num_sets(), 500);

        // Check some connections
        assert!(ds.connected(&0, &1));
        assert!(ds.connected(&998, &999));
        assert!(!ds.connected(&0, &2));
    }
}
