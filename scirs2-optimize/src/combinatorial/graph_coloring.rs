//! Graph coloring algorithms.
//!
//! Provides greedy (Welsh-Powell), DSATUR, exact backtracking, and chromatic
//! number computation for undirected graphs.

use crate::error::OptimizeError;

/// Result type for graph coloring operations.
pub type ColoringResult<T> = Result<T, OptimizeError>;

// ── Graph structure ───────────────────────────────────────────────────────────

/// Undirected graph represented as an adjacency list.
#[derive(Debug, Clone)]
pub struct GraphColoring {
    /// Adjacency list: `adj[u]` contains all neighbours of `u`.
    pub adj: Vec<Vec<usize>>,
    /// Number of vertices.
    pub n: usize,
}

impl GraphColoring {
    /// Create a new empty graph with `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            adj: vec![vec![]; n],
            n,
        }
    }

    /// Add an undirected edge between `u` and `v`.
    ///
    /// Duplicate edges are silently ignored.
    ///
    /// # Errors
    /// Returns an error if `u` or `v` are out of range.
    pub fn add_edge(&mut self, u: usize, v: usize) -> ColoringResult<()> {
        if u >= self.n || v >= self.n {
            return Err(OptimizeError::InvalidInput(format!(
                "Edge ({u},{v}) out of range for graph with {} vertices",
                self.n
            )));
        }
        if u == v {
            return Ok(()); // ignore self-loops
        }
        if !self.adj[u].contains(&v) {
            self.adj[u].push(v);
        }
        if !self.adj[v].contains(&u) {
            self.adj[v].push(u);
        }
        Ok(())
    }

    /// Degree of vertex `v`.
    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    // ── Welsh-Powell greedy coloring ──────────────────────────────────────────

    /// Greedy coloring using the Welsh-Powell heuristic (sort by degree desc).
    ///
    /// Assigns the smallest available color to each vertex in degree-descending
    /// order.  Returns a color array where `coloring[v]` is the color (0-indexed)
    /// assigned to vertex `v`.
    pub fn greedy_coloring(&self) -> Vec<usize> {
        if self.n == 0 {
            return vec![];
        }

        // Sort vertices by degree descending
        let mut order: Vec<usize> = (0..self.n).collect();
        order.sort_by(|&a, &b| self.degree(b).cmp(&self.degree(a)));

        let mut color = vec![usize::MAX; self.n];

        for &v in &order {
            let mut neighbor_colors = std::collections::HashSet::new();
            for &u in &self.adj[v] {
                if color[u] != usize::MAX {
                    neighbor_colors.insert(color[u]);
                }
            }
            // Assign smallest available color
            let mut c = 0;
            while neighbor_colors.contains(&c) {
                c += 1;
            }
            color[v] = c;
        }

        color
    }

    // ── DSATUR coloring ───────────────────────────────────────────────────────

    /// DSATUR coloring: at each step, color the vertex with maximum saturation
    /// (number of distinct colors among already-colored neighbours), breaking
    /// ties by degree.
    ///
    /// Often produces optimal or near-optimal colorings.
    pub fn dsatur_coloring(&self) -> Vec<usize> {
        if self.n == 0 {
            return vec![];
        }

        let mut color = vec![usize::MAX; self.n];
        // saturation[v] = number of distinct colors in N(v) that are already assigned
        let mut saturation = vec![0usize; self.n];
        // neighbor_color_sets[v] = set of colors seen in N(v)
        let mut neighbor_colors: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); self.n];
        let mut colored = 0usize;

        while colored < self.n {
            // Pick uncolored vertex with max saturation, break ties by degree
            // The loop condition (colored < self.n) guarantees an uncolored vertex exists.
            let v = match (0..self.n)
                .filter(|&u| color[u] == usize::MAX)
                .max_by(|&a, &b| {
                    saturation[a]
                        .cmp(&saturation[b])
                        .then(self.degree(a).cmp(&self.degree(b)))
                }) {
                Some(vertex) => vertex,
                None => break, // all vertices colored (should not occur given loop condition)
            };

            // Assign smallest available color
            let mut c = 0;
            while neighbor_colors[v].contains(&c) {
                c += 1;
            }
            color[v] = c;
            colored += 1;

            // Update saturation of uncolored neighbours
            for &u in &self.adj[v] {
                if color[u] == usize::MAX && !neighbor_colors[u].contains(&c) {
                    neighbor_colors[u].insert(c);
                    saturation[u] += 1;
                }
            }
        }

        color
    }

    // ── Backtracking exact k-coloring ─────────────────────────────────────────

    /// Try to find a valid k-coloring via backtracking with forward checking.
    ///
    /// Returns `Some(coloring)` if a k-coloring exists, `None` otherwise.
    pub fn backtrack_coloring(&self, k: usize) -> Option<Vec<usize>> {
        if self.n == 0 {
            return Some(vec![]);
        }
        if k == 0 {
            return None;
        }

        let mut color = vec![k; self.n]; // k = uncolored sentinel
                                         // Order by degree descending (Welsh-Powell ordering)
        let mut order: Vec<usize> = (0..self.n).collect();
        order.sort_by(|&a, &b| self.degree(b).cmp(&self.degree(a)));
        let pos_of: Vec<usize> = {
            let mut p = vec![0usize; self.n];
            for (pos, &v) in order.iter().enumerate() {
                p[v] = pos;
            }
            p
        };

        if self.backtrack_inner(&mut color, &order, &pos_of, k, 0) {
            Some(color)
        } else {
            None
        }
    }

    fn backtrack_inner(
        &self,
        color: &mut Vec<usize>,
        order: &[usize],
        pos_of: &[usize],
        k: usize,
        idx: usize,
    ) -> bool {
        if idx == self.n {
            return true;
        }
        let v = order[idx];

        // Collect forbidden colors
        let mut forbidden = vec![false; k];
        for &u in &self.adj[v] {
            if color[u] < k {
                // already colored
                forbidden[color[u]] = true;
            }
        }

        for c in 0..k {
            if forbidden[c] {
                continue;
            }
            // Forward checking: ensure all uncolored neighbours still have at
            // least one available color after assigning c to v
            let ok = self.adj[v].iter().all(|&u| {
                if color[u] < k {
                    return true; // already colored, no change
                }
                // u is uncolored; check if any color other than c is available
                let used: std::collections::HashSet<usize> = self.adj[u]
                    .iter()
                    .filter(|&&w| color[w] < k && w != v)
                    .map(|&w| color[w])
                    .collect();
                // u's forbidden set = used ∪ {c}
                let available = (0..k).filter(|&x| x != c && !used.contains(&x)).count();
                // Check neighbours that are later in order
                if pos_of[u] > idx {
                    available > 0
                } else {
                    true
                }
            });

            if !ok {
                continue;
            }

            color[v] = c;
            if self.backtrack_inner(color, order, pos_of, k, idx + 1) {
                return true;
            }
            color[v] = k; // uncolor
        }

        false
    }

    // ── Chromatic number ──────────────────────────────────────────────────────

    /// Compute the chromatic number χ(G) via binary search + backtracking.
    ///
    /// The greedy coloring gives an upper bound; 1 is the lower bound (unless
    /// there are edges, in which case 2 is the lower bound).
    pub fn chromatic_number(&self) -> usize {
        if self.n == 0 {
            return 0;
        }
        // Lower bound: clique number lower bound via degree
        let has_edges = self.adj.iter().any(|nbrs| !nbrs.is_empty());
        let lower = if has_edges { 2 } else { 1 };

        // Upper bound from DSATUR
        let upper_coloring = self.dsatur_coloring();
        let upper = upper_coloring
            .iter()
            .cloned()
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        if lower >= upper {
            return lower;
        }

        // Binary search in [lower, upper]
        let mut lo = lower;
        let mut hi = upper;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.backtrack_coloring(mid).is_some() {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    // ── Validation ────────────────────────────────────────────────────────────

    /// Check that `coloring` is a valid proper coloring (no two adjacent
    /// vertices share the same color).
    pub fn is_valid_coloring(&self, coloring: &[usize]) -> bool {
        if coloring.len() != self.n {
            return false;
        }
        for u in 0..self.n {
            for &v in &self.adj[u] {
                if coloring[u] == coloring[v] {
                    return false;
                }
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn cycle_graph(n: usize) -> GraphColoring {
        let mut g = GraphColoring::new(n);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n).expect("unexpected None or Err");
        }
        g
    }

    fn complete_graph(n: usize) -> GraphColoring {
        let mut g = GraphColoring::new(n);
        for i in 0..n {
            for j in i + 1..n {
                g.add_edge(i, j).expect("unexpected None or Err");
            }
        }
        g
    }

    #[test]
    fn test_greedy_valid() {
        let g = cycle_graph(6);
        let c = g.greedy_coloring();
        assert!(g.is_valid_coloring(&c));
    }

    #[test]
    fn test_dsatur_valid() {
        let g = complete_graph(5);
        let c = g.dsatur_coloring();
        assert!(g.is_valid_coloring(&c));
        // K5 needs 5 colors
        let num_colors = c
            .iter()
            .cloned()
            .max()
            .expect("failed to create num_colors")
            + 1;
        assert_eq!(num_colors, 5);
    }

    #[test]
    fn test_bipartite_2_colorable() {
        // Complete bipartite K3,3
        let mut g = GraphColoring::new(6);
        for u in 0..3 {
            for v in 3..6 {
                g.add_edge(u, v).expect("unexpected None or Err");
            }
        }
        let c = g.backtrack_coloring(2);
        assert!(c.is_some());
        assert!(g.is_valid_coloring(c.as_ref().expect("unexpected None or Err")));
    }

    #[test]
    fn test_odd_cycle_not_2_colorable() {
        let g = cycle_graph(5); // C5 needs 3 colors
        assert!(g.backtrack_coloring(2).is_none());
        assert!(g.backtrack_coloring(3).is_some());
    }

    #[test]
    fn test_chromatic_number_cycle() {
        let even = cycle_graph(6);
        assert_eq!(even.chromatic_number(), 2);

        let odd = cycle_graph(5);
        assert_eq!(odd.chromatic_number(), 3);
    }

    #[test]
    fn test_chromatic_number_complete() {
        for n in 1..=5 {
            let g = complete_graph(n);
            assert_eq!(g.chromatic_number(), n);
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = GraphColoring::new(4);
        let c = g.greedy_coloring();
        // All vertices get color 0 (no edges)
        assert!(c.iter().all(|&x| x == 0));
        assert!(g.is_valid_coloring(&c));
        assert_eq!(g.chromatic_number(), 1);
    }

    #[test]
    fn test_self_loop_ignored() {
        let mut g = GraphColoring::new(3);
        g.add_edge(0, 0).expect("unexpected None or Err"); // self-loop ignored
        g.add_edge(0, 1).expect("unexpected None or Err");
        assert_eq!(g.degree(0), 1);
    }

    #[test]
    fn test_invalid_edge() {
        let mut g = GraphColoring::new(3);
        assert!(g.add_edge(0, 5).is_err());
    }
}
