//! Union-Find with path compression and union-by-rank.
//!
//! This module provides a classic disjoint-set union-find data structure used
//! internally by the e-graph to track equivalence classes. Both `find` and
//! `union` run in near O(1) amortized time (inverse-Ackermann).
//!
//! # Invariants
//!
//! - `parent[i] == i` iff `i` is a root (representative of its class).
//! - `rank[winner] >= rank[loser]` after any union.
//! - `find` applies path compression: after `find(x)`, `parent[x]` points
//!   directly to the root.

/// Disjoint-set union-find structure with path compression and union-by-rank.
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    /// Create an empty union-find.
    pub fn new() -> Self {
        UnionFind {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    /// Create a new singleton set and return its id.
    ///
    /// Ids are assigned sequentially starting at 0.
    pub fn make_set(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    /// Find the representative (root) of the set containing `x`.
    ///
    /// Applies iterative path compression: on the way back up every node in
    /// the path is pointed directly at the root.
    pub fn find(&mut self, mut x: u32) -> u32 {
        // Walk up to the root.
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Path compression: point all nodes along the path directly at root.
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }
        root
    }

    /// Merge the sets containing `x` and `y`.
    ///
    /// Returns `(winner, loser)` where winner is the new representative.
    /// If `x` and `y` are already in the same set, returns `(find(x), find(x))`
    /// — callers should check if winner == loser to detect no-op merges.
    ///
    /// Union-by-rank: the set with the higher rank absorbs the other.
    /// On equal rank, `x`'s root becomes the winner and its rank increases by 1.
    pub fn union(&mut self, x: u32, y: u32) -> (u32, u32) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return (rx, rx);
        }
        let rank_x = self.rank[rx as usize];
        let rank_y = self.rank[ry as usize];
        if rank_x < rank_y {
            // ry wins
            self.parent[rx as usize] = ry;
            (ry, rx)
        } else if rank_x > rank_y {
            // rx wins
            self.parent[ry as usize] = rx;
            (rx, ry)
        } else {
            // Equal rank — rx wins; increment rank.
            self.parent[ry as usize] = rx;
            self.rank[rx as usize] += 1;
            (rx, ry)
        }
    }

    /// Return the current number of elements (including merged ones).
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Return the rank of element `x` (for testing).
    pub fn rank_of(&self, x: u32) -> u8 {
        self.rank.get(x as usize).copied().unwrap_or(0)
    }

    /// Non-mutating root lookup (no path compression).
    ///
    /// Useful for read-only passes that need a shared borrow of `UnionFind`
    /// (e.g. during extraction where `EGraph` is borrowed immutably).
    /// Bounded to avoid infinite loops on corrupt state.
    pub fn find_root_immutable(&self, mut x: u32) -> u32 {
        let mut depth = 0u32;
        while depth < 128 {
            let next = self.parent.get(x as usize).copied().unwrap_or(x);
            if next == x {
                break;
            }
            x = next;
            depth += 1;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new();
        // Create 5 sets: 0, 1, 2, 3, 4
        for _ in 0..5 {
            uf.make_set();
        }
        // Chain-union: 0-1, 1-2, 2-3, 3-4
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);
        // All should have same root.
        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(3), root);
        assert_eq!(uf.find(4), root);
    }

    #[test]
    fn test_union_find_rank() {
        let mut uf = UnionFind::new();
        // Create 4 sets.
        for _ in 0..4 {
            uf.make_set();
        }
        // Union 0 and 1 — equal rank, root 0 wins, rank[0] becomes 1.
        let (winner, _loser) = uf.union(0, 1);
        assert_eq!(winner, uf.find(0));
        assert_eq!(winner, uf.find(1));
        // After equal-rank union, winner should have rank 1.
        assert_eq!(uf.rank_of(winner), 1);
    }

    #[test]
    fn test_make_set_returns_sequential_ids() {
        let mut uf = UnionFind::new();
        assert_eq!(uf.make_set(), 0);
        assert_eq!(uf.make_set(), 1);
        assert_eq!(uf.make_set(), 2);
        assert_eq!(uf.len(), 3);
    }

    #[test]
    fn test_union_same_set_noop() {
        let mut uf = UnionFind::new();
        uf.make_set();
        let (w, l) = uf.union(0, 0);
        assert_eq!(w, l); // same set, no-op
    }
}
